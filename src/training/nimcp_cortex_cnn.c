/**
 * @file nimcp_cortex_cnn.c
 * @brief Per-cortex CNN processors — modality-specific feature extraction
 *
 * WHAT: Independent CNN processors for visual, audio, speech, somatosensory modalities
 * WHY:  Replace single classifier head with per-modality convolutional architectures
 * HOW:  Each processor wraps a cnn_trainer_t with modality-specific layers,
 *       provides forward/backward wrappers, attention fusion, and UTM adapter
 */

#include "training/nimcp_cortex_cnn.h"
#include "training/nimcp_cnn_training.h"
#include "training/nimcp_unified_training.h"
#include "utils/tensor/nimcp_tensor.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ========================================================================= */
/* Default embedding dimensions per modality                                  */
/* ========================================================================= */

#define VISUAL_DEFAULT_EMBED_DIM   64
#define AUDIO_DEFAULT_EMBED_DIM    64
#define SPEECH_DEFAULT_EMBED_DIM   32
#define SOMATO_DEFAULT_EMBED_DIM   32

#define CORTEX_CNN_MAX_LABELS     4096
#define CORTEX_CNN_ATTENTION_TEMP  0.5f

/* ========================================================================= */
/* Internal processor structure                                               */
/* ========================================================================= */

struct cortex_cnn_processor {
    cortex_cnn_type_t type;
    cnn_trainer_t* trainer;           /* Owned CNN trainer */
    uint32_t embedding_dim;           /* Output embedding size */

    /* Last forward result (cached for backward) */
    cnn_forward_result_t last_fwd;
    bool has_fwd_result;

    /* Embedding buffer (extracted from last forward output) */
    float* embedding;                 /* [embedding_dim] */

    /* Label management (simple index for one-hot) */
    char** labels;
    uint32_t num_labels;
    uint32_t max_labels;

    /* Metrics */
    float last_loss;
    float ema_loss;
    uint64_t forward_steps;
    uint64_t backward_steps;
    float confidence;                 /* Softmax max from last forward */
    uint32_t num_params;              /* Approximate param count */

    /* FNO audio processor (alternative to CNN trainer for audio modality) */
    void* fno_audio;                  /* fno_audio_processor_t* (NULL if using CNN) */
};

/* ========================================================================= */
/* Internal helpers                                                           */
/* ========================================================================= */

static const char* cortex_type_names[CORTEX_CNN_COUNT] = {
    "Visual", "Audio", "Speech", "Somato"
};

/**
 * @brief Get or create label index for a string label
 */
static uint32_t cortex_get_or_create_label(cortex_cnn_processor_t* proc, const char* label) {
    if (!proc || !label) return 0;

    /* Search existing labels */
    for (uint32_t i = 0; i < proc->num_labels; i++) {
        if (proc->labels[i] && strcmp(proc->labels[i], label) == 0) {
            return i;
        }
    }

    /* Create new label */
    if (proc->num_labels >= proc->max_labels) return 0;

    size_t len = strlen(label);
    proc->labels[proc->num_labels] = (char*)nimcp_malloc(len + 1);
    if (!proc->labels[proc->num_labels]) return 0;
    memcpy(proc->labels[proc->num_labels], label, len + 1);

    return proc->num_labels++;
}

/**
 * @brief Extract embedding from CNN forward output
 *
 * The last dense layer's output IS the embedding. Copy it to proc->embedding.
 * Also compute confidence as softmax max.
 */
static void extract_embedding(cortex_cnn_processor_t* proc) {
    if (!proc->has_fwd_result || !proc->last_fwd.output) return;

    const float* out = (const float*)nimcp_tensor_data_const(proc->last_fwd.output);
    size_t numel = nimcp_tensor_numel(proc->last_fwd.output);
    if (!out || numel == 0) return;
    if (!proc->embedding) return;  /* Guard against NULL embedding buffer */

    uint32_t copy_dim = (proc->embedding_dim < (uint32_t)numel) ?
                         proc->embedding_dim : (uint32_t)numel;
    memcpy(proc->embedding, out, copy_dim * sizeof(float));

    /* Zero-pad if output smaller than embedding_dim */
    if (copy_dim < proc->embedding_dim) {
        memset(proc->embedding + copy_dim, 0,
               (proc->embedding_dim - copy_dim) * sizeof(float));
    }

    /* Compute confidence = max of softmax-like values.
     * Use exp(x)/sum(exp(x)) but for efficiency just find max output. */
    float max_val = -1e30f;
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < copy_dim; i++) {
        if (out[i] > max_val) max_val = out[i];
    }
    for (uint32_t i = 0; i < copy_dim; i++) {
        sum_exp += expf(out[i] - max_val);
    }
    proc->confidence = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
}

/**
 * @brief Run CNN forward on a 1D float tensor
 */
static const float* cortex_forward_1d(cortex_cnn_processor_t* proc,
                                       const float* data, uint32_t size) {
    if (!proc || !proc->trainer || !data || size == 0) return NULL;

    uint32_t dims[2] = {1, size};
    nimcp_tensor_t* input = nimcp_tensor_create(dims, 2, NIMCP_DTYPE_F32);
    if (!input) return NULL;

    memcpy(nimcp_tensor_data(input), data, size * sizeof(float));

    memset(&proc->last_fwd, 0, sizeof(proc->last_fwd));
    nimcp_error_t err = cnn_trainer_forward(proc->trainer, input, &proc->last_fwd);
    nimcp_tensor_destroy(input);

    if (err != NIMCP_SUCCESS) {
        proc->has_fwd_result = false;
        return NULL;
    }

    proc->has_fwd_result = true;
    proc->forward_steps++;
    extract_embedding(proc);
    return proc->embedding;
}

/* ========================================================================= */
/* Factory — build modality-specific architectures                            */
/* ========================================================================= */

/**
 * @brief Build visual cortex CNN architecture
 *
 * Conv2d(3->16,3x3)->BN->ReLU->Pool(2x2)
 * ->Conv2d(16->32,3x3)->BN->ReLU->Pool(2x2)
 * ->Conv2d(32->64,3x3)->BN->ReLU->GlobalAvgPool
 * ->Dense(64->embed_dim)
 */
static int build_visual_architecture(cnn_trainer_t* trainer, uint32_t embed_dim) {
    /* Conv1: 3->16, 3x3, padding=1 */
    cnn_conv_config_t conv1 = {
        .kernel_h = 3, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 1, .padding_w = 1,
        .dilation_h = 1, .dilation_w = 1,
        .in_channels = 3, .out_channels = 16,
        .groups = 1,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_conv_layer(trainer, &conv1)) return -1;

    cnn_batch_norm_config_t bn1 = {
        .num_features = 16, .epsilon = 1e-5f,
        .momentum = 0.1f, .affine = true, .track_running_stats = true
    };
    if (!cnn_trainer_add_batch_norm_layer(trainer, &bn1)) return -1;
    if (!cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU)) return -1;

    cnn_pool_config_t pool1 = {
        .type = CNN_POOL_MAX,
        .pool_h = 2, .pool_w = 2,
        .stride_h = 2, .stride_w = 2
    };
    if (!cnn_trainer_add_pool_layer(trainer, &pool1)) return -1;

    /* Conv2: 16->32, 3x3, padding=1 */
    cnn_conv_config_t conv2 = {
        .kernel_h = 3, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 1, .padding_w = 1,
        .dilation_h = 1, .dilation_w = 1,
        .in_channels = 16, .out_channels = 32,
        .groups = 1,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_conv_layer(trainer, &conv2)) return -1;

    cnn_batch_norm_config_t bn2 = {
        .num_features = 32, .epsilon = 1e-5f,
        .momentum = 0.1f, .affine = true, .track_running_stats = true
    };
    if (!cnn_trainer_add_batch_norm_layer(trainer, &bn2)) return -1;
    if (!cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU)) return -1;

    cnn_pool_config_t pool2 = {
        .type = CNN_POOL_MAX,
        .pool_h = 2, .pool_w = 2,
        .stride_h = 2, .stride_w = 2
    };
    if (!cnn_trainer_add_pool_layer(trainer, &pool2)) return -1;

    /* Conv3: 32->64, 3x3, padding=1 */
    cnn_conv_config_t conv3 = {
        .kernel_h = 3, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 1, .padding_w = 1,
        .dilation_h = 1, .dilation_w = 1,
        .in_channels = 32, .out_channels = 64,
        .groups = 1,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_conv_layer(trainer, &conv3)) return -1;

    cnn_batch_norm_config_t bn3 = {
        .num_features = 64, .epsilon = 1e-5f,
        .momentum = 0.1f, .affine = true, .track_running_stats = true
    };
    if (!cnn_trainer_add_batch_norm_layer(trainer, &bn3)) return -1;
    if (!cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU)) return -1;

    /* Global Average Pool */
    cnn_pool_config_t gap = {
        .type = CNN_POOL_GLOBAL_AVERAGE,
        .pool_h = 0, .pool_w = 0,  /* Global = entire spatial extent */
        .stride_h = 1, .stride_w = 1
    };
    if (!cnn_trainer_add_pool_layer(trainer, &gap)) return -1;

    /* Flatten + Dense(64->embed_dim) */
    if (!cnn_trainer_add_flatten_layer(trainer)) return -1;

    cnn_dense_config_t dense = {
        .in_features = 64,
        .out_features = embed_dim,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_dense_layer(trainer, &dense)) return -1;

    return 0;
}

/**
 * @brief Build audio cortex CNN architecture
 *
 * Conv2d(1->16,1x5)->BN->ReLU->Pool(1x2)
 * ->Conv2d(16->32,1x3)->BN->ReLU->GlobalAvgPool
 * ->Dense(32->embed_dim)
 */
static int build_audio_architecture(cnn_trainer_t* trainer, uint32_t embed_dim) {
    cnn_conv_config_t conv1 = {
        .kernel_h = 1, .kernel_w = 5,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 0, .padding_w = 2,
        .dilation_h = 1, .dilation_w = 1,
        .in_channels = 1, .out_channels = 16,
        .groups = 1,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_conv_layer(trainer, &conv1)) return -1;

    cnn_batch_norm_config_t bn1 = {
        .num_features = 16, .epsilon = 1e-5f,
        .momentum = 0.1f, .affine = true, .track_running_stats = true
    };
    if (!cnn_trainer_add_batch_norm_layer(trainer, &bn1)) return -1;
    if (!cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU)) return -1;

    cnn_pool_config_t pool1 = {
        .type = CNN_POOL_MAX,
        .pool_h = 1, .pool_w = 2,
        .stride_h = 1, .stride_w = 2
    };
    if (!cnn_trainer_add_pool_layer(trainer, &pool1)) return -1;

    cnn_conv_config_t conv2 = {
        .kernel_h = 1, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 0, .padding_w = 1,
        .dilation_h = 1, .dilation_w = 1,
        .in_channels = 16, .out_channels = 32,
        .groups = 1,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_conv_layer(trainer, &conv2)) return -1;

    cnn_batch_norm_config_t bn2 = {
        .num_features = 32, .epsilon = 1e-5f,
        .momentum = 0.1f, .affine = true, .track_running_stats = true
    };
    if (!cnn_trainer_add_batch_norm_layer(trainer, &bn2)) return -1;
    if (!cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU)) return -1;

    cnn_pool_config_t gap = {
        .type = CNN_POOL_GLOBAL_AVERAGE,
        .pool_h = 0, .pool_w = 0,
        .stride_h = 1, .stride_w = 1
    };
    if (!cnn_trainer_add_pool_layer(trainer, &gap)) return -1;

    if (!cnn_trainer_add_flatten_layer(trainer)) return -1;

    cnn_dense_config_t dense = {
        .in_features = 32,
        .out_features = embed_dim,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_dense_layer(trainer, &dense)) return -1;

    return 0;
}

/**
 * @brief Build speech cortex CNN architecture
 *
 * Conv2d(1->16,1x3)->BN->ReLU
 * ->Conv2d(16->32,1x3)->BN->ReLU->GlobalAvgPool
 * ->Dense(32->embed_dim)
 */
static int build_speech_architecture(cnn_trainer_t* trainer, uint32_t embed_dim) {
    cnn_conv_config_t conv1 = {
        .kernel_h = 1, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 0, .padding_w = 1,
        .dilation_h = 1, .dilation_w = 1,
        .in_channels = 1, .out_channels = 16,
        .groups = 1,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_conv_layer(trainer, &conv1)) return -1;

    cnn_batch_norm_config_t bn1 = {
        .num_features = 16, .epsilon = 1e-5f,
        .momentum = 0.1f, .affine = true, .track_running_stats = true
    };
    if (!cnn_trainer_add_batch_norm_layer(trainer, &bn1)) return -1;
    if (!cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU)) return -1;

    cnn_conv_config_t conv2 = {
        .kernel_h = 1, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 0, .padding_w = 1,
        .dilation_h = 1, .dilation_w = 1,
        .in_channels = 16, .out_channels = 32,
        .groups = 1,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_conv_layer(trainer, &conv2)) return -1;

    cnn_batch_norm_config_t bn2 = {
        .num_features = 32, .epsilon = 1e-5f,
        .momentum = 0.1f, .affine = true, .track_running_stats = true
    };
    if (!cnn_trainer_add_batch_norm_layer(trainer, &bn2)) return -1;
    if (!cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU)) return -1;

    cnn_pool_config_t gap = {
        .type = CNN_POOL_GLOBAL_AVERAGE,
        .pool_h = 0, .pool_w = 0,
        .stride_h = 1, .stride_w = 1
    };
    if (!cnn_trainer_add_pool_layer(trainer, &gap)) return -1;

    if (!cnn_trainer_add_flatten_layer(trainer)) return -1;

    cnn_dense_config_t dense = {
        .in_features = 32,
        .out_features = embed_dim,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_dense_layer(trainer, &dense)) return -1;

    return 0;
}

/**
 * @brief Build somatosensory cortex architecture (dense-only, no conv)
 *
 * Dense(45->64)->ReLU->Dense(64->embed_dim)
 */
static int build_somato_architecture(cnn_trainer_t* trainer, uint32_t embed_dim) {
    cnn_dense_config_t dense1 = {
        .in_features = 45,
        .out_features = 64,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_dense_layer(trainer, &dense1)) return -1;
    if (!cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU)) return -1;

    cnn_dense_config_t dense2 = {
        .in_features = 64,
        .out_features = embed_dim,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    if (!cnn_trainer_add_dense_layer(trainer, &dense2)) return -1;

    return 0;
}

static uint32_t default_embed_dim(cortex_cnn_type_t type) {
    switch (type) {
        case CORTEX_CNN_VISUAL:  return VISUAL_DEFAULT_EMBED_DIM;
        case CORTEX_CNN_AUDIO:   return AUDIO_DEFAULT_EMBED_DIM;
        case CORTEX_CNN_SPEECH:  return SPEECH_DEFAULT_EMBED_DIM;
        case CORTEX_CNN_SOMATO:  return SOMATO_DEFAULT_EMBED_DIM;
        default: return 32;
    }
}

static uint32_t estimate_params(cortex_cnn_type_t type) {
    switch (type) {
        case CORTEX_CNN_VISUAL:  return 30000;
        case CORTEX_CNN_AUDIO:   return 8000;
        case CORTEX_CNN_SPEECH:  return 4000;
        case CORTEX_CNN_SOMATO:  return 4000;
        default: return 0;
    }
}

/* ========================================================================= */
/* Public API: Lifecycle                                                      */
/* ========================================================================= */

cortex_cnn_processor_t* cortex_cnn_create(cortex_cnn_type_t type, uint32_t embedding_dim) {
    if (type < 0 || type >= CORTEX_CNN_COUNT) {
        NIMCP_LOGGING_WARN("cortex_cnn_create: invalid type %d", (int)type);
        return NULL;
    }

    if (embedding_dim == 0) {
        embedding_dim = default_embed_dim(type);
    }

    cortex_cnn_processor_t* proc = (cortex_cnn_processor_t*)nimcp_calloc(
        1, sizeof(cortex_cnn_processor_t));
    if (!proc) return NULL;

    proc->type = type;
    proc->embedding_dim = embedding_dim;
    proc->ema_loss = -1.0f;  /* Sentinel: not yet computed */

    /* Allocate embedding buffer */
    proc->embedding = (float*)nimcp_calloc(embedding_dim, sizeof(float));
    if (!proc->embedding) {
        nimcp_free(proc);
        return NULL;
    }

    /* Allocate label storage */
    proc->max_labels = CORTEX_CNN_MAX_LABELS;
    proc->labels = (char**)nimcp_calloc(proc->max_labels, sizeof(char*));
    if (!proc->labels) {
        nimcp_free(proc->embedding);
        nimcp_free(proc);
        return NULL;
    }

    /* Create CNN trainer */
    cnn_trainer_config_t cfg;
    cnn_trainer_default_config(&cfg);
    cfg.learning_rate = 0.001f;
    cfg.gradient_clip_value = 5.0f;
    cfg.diversity_loss_weight = 0.1f;
    cfg.use_gradient_normalization = true;
    cfg.gradient_target_norm = 1.0f;

    proc->trainer = cnn_trainer_create(&cfg);
    if (!proc->trainer) {
        nimcp_free(proc->labels);
        nimcp_free(proc->embedding);
        nimcp_free(proc);
        return NULL;
    }

    /* Build modality-specific architecture */
    int rc = -1;
    switch (type) {
        case CORTEX_CNN_VISUAL:  rc = build_visual_architecture(proc->trainer, embedding_dim); break;
        case CORTEX_CNN_AUDIO:   rc = build_audio_architecture(proc->trainer, embedding_dim); break;
        case CORTEX_CNN_SPEECH:  rc = build_speech_architecture(proc->trainer, embedding_dim); break;
        case CORTEX_CNN_SOMATO:  rc = build_somato_architecture(proc->trainer, embedding_dim); break;
        default: break;
    }

    if (rc != 0) {
        NIMCP_LOGGING_WARN("cortex_cnn_create: failed to build %s architecture",
                          cortex_type_names[type]);
        cnn_trainer_destroy(proc->trainer);
        nimcp_free(proc->labels);
        nimcp_free(proc->embedding);
        nimcp_free(proc);
        return NULL;
    }

    proc->num_params = estimate_params(type);

    NIMCP_LOGGING_INFO("Created %s cortex CNN processor: embed_dim=%u, ~%u params",
                      cortex_type_names[type], embedding_dim, proc->num_params);
    return proc;
}

void cortex_cnn_destroy(cortex_cnn_processor_t* proc) {
    if (!proc) return;

    if (proc->trainer) {
        cnn_trainer_destroy(proc->trainer);
    }
    if (proc->labels) {
        for (uint32_t i = 0; i < proc->num_labels; i++) {
            nimcp_free(proc->labels[i]);
        }
        nimcp_free(proc->labels);
    }
    nimcp_free(proc->embedding);
    nimcp_free(proc);
}

/* ========================================================================= */
/* Public API: Forward Pass                                                   */
/* ========================================================================= */

const float* cortex_cnn_forward_visual(cortex_cnn_processor_t* proc,
                                        const uint8_t* pixels,
                                        uint32_t w, uint32_t h, uint32_t ch) {
    if (!proc || proc->type != CORTEX_CNN_VISUAL || !pixels) return NULL;
    if (w == 0 || h == 0 || ch == 0) return NULL;

    /* Convert uint8 pixels to float [0,1] and pack as 1D tensor (H*W*C) */
    uint32_t total = w * h * ch;
    float* float_data = (float*)nimcp_malloc(total * sizeof(float));
    if (!float_data) return NULL;

    for (uint32_t i = 0; i < total; i++) {
        float_data[i] = (float)pixels[i] / 255.0f;
    }

    /* Create 4D tensor: [1, ch, h, w] for CNN forward */
    uint32_t dims[4] = {1, ch, h, w};
    nimcp_tensor_t* input = nimcp_tensor_create(dims, 4, NIMCP_DTYPE_F32);
    if (!input) {
        nimcp_free(float_data);
        return NULL;
    }

    /* Reorder from HWC to CHW */
    float* tensor_data = (float*)nimcp_tensor_data(input);
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            for (uint32_t c = 0; c < ch; c++) {
                uint32_t hwc_idx = (y * w + x) * ch + c;
                uint32_t chw_idx = c * (h * w) + y * w + x;
                tensor_data[chw_idx] = float_data[hwc_idx];
            }
        }
    }
    nimcp_free(float_data);

    /* Forward through CNN */
    memset(&proc->last_fwd, 0, sizeof(proc->last_fwd));
    nimcp_error_t err = cnn_trainer_forward(proc->trainer, input, &proc->last_fwd);
    nimcp_tensor_destroy(input);

    if (err != NIMCP_SUCCESS) {
        proc->has_fwd_result = false;
        return NULL;
    }

    proc->has_fwd_result = true;
    proc->forward_steps++;
    extract_embedding(proc);
    return proc->embedding;
}

const float* cortex_cnn_forward_audio(cortex_cnn_processor_t* proc,
                                       const float* mel, uint32_t mel_size) {
    if (!proc || proc->type != CORTEX_CNN_AUDIO || !mel) return NULL;

    /* FNO path — spectral convolution for native frequency-domain processing */
    if (proc->fno_audio) {
        #include "training/nimcp_fno_layer.h"
        fno_audio_processor_t* fno = (fno_audio_processor_t*)proc->fno_audio;
        if (!proc->embedding) {
            proc->embedding = nimcp_calloc(proc->embedding_dim, sizeof(float));
            if (!proc->embedding) return NULL;
        }
        int rc = fno_audio_forward(fno, mel, mel_size, proc->embedding);
        if (rc != 0) {
            proc->has_fwd_result = false;
            return NULL;
        }
        proc->has_fwd_result = true;
        proc->forward_steps++;
        /* Compute confidence from embedding norm */
        float norm = 0.0f;
        for (uint32_t i = 0; i < proc->embedding_dim; i++)
            norm += proc->embedding[i] * proc->embedding[i];
        proc->confidence = (norm > 0.0f) ? 1.0f / (1.0f + expf(-sqrtf(norm))) : 0.0f;
        return proc->embedding;
    }

    /* CNN path — Conv2d layers (original architecture) */
    uint32_t dims[4] = {1, 1, 1, mel_size};
    nimcp_tensor_t* input = nimcp_tensor_create(dims, 4, NIMCP_DTYPE_F32);
    if (!input) return NULL;

    memcpy(nimcp_tensor_data(input), mel, mel_size * sizeof(float));

    memset(&proc->last_fwd, 0, sizeof(proc->last_fwd));
    nimcp_error_t err = cnn_trainer_forward(proc->trainer, input, &proc->last_fwd);
    nimcp_tensor_destroy(input);

    if (err != NIMCP_SUCCESS) {
        proc->has_fwd_result = false;
        return NULL;
    }

    proc->has_fwd_result = true;
    proc->forward_steps++;
    extract_embedding(proc);
    return proc->embedding;
}

const float* cortex_cnn_forward_speech(cortex_cnn_processor_t* proc,
                                        const float* phonemes, uint32_t size) {
    if (!proc || proc->type != CORTEX_CNN_SPEECH || !phonemes) return NULL;

    /* Pack as [1, 1, 1, size] */
    uint32_t dims[4] = {1, 1, 1, size};
    nimcp_tensor_t* input = nimcp_tensor_create(dims, 4, NIMCP_DTYPE_F32);
    if (!input) return NULL;

    memcpy(nimcp_tensor_data(input), phonemes, size * sizeof(float));

    memset(&proc->last_fwd, 0, sizeof(proc->last_fwd));
    nimcp_error_t err = cnn_trainer_forward(proc->trainer, input, &proc->last_fwd);
    nimcp_tensor_destroy(input);

    if (err != NIMCP_SUCCESS) {
        proc->has_fwd_result = false;
        return NULL;
    }

    proc->has_fwd_result = true;
    proc->forward_steps++;
    extract_embedding(proc);
    return proc->embedding;
}

const float* cortex_cnn_forward_somato(cortex_cnn_processor_t* proc,
                                        const float* segments, uint32_t n_segments) {
    if (!proc || proc->type != CORTEX_CNN_SOMATO || !segments) return NULL;

    /* Somato uses dense-only architecture, pass as 1D [1, n_segments] */
    return cortex_forward_1d(proc, segments, n_segments);
}

/* ========================================================================= */
/* Public API: Backward Pass                                                  */
/* ========================================================================= */

float cortex_cnn_backward(cortex_cnn_processor_t* proc,
                           const char* label, uint32_t num_outputs) {
    if (!proc || !proc->trainer || !proc->has_fwd_result || !label) return -1.0f;
    if (num_outputs == 0) num_outputs = proc->embedding_dim;

    uint32_t label_idx = cortex_get_or_create_label(proc, label);

    /* Build one-hot target */
    uint32_t target_dim = num_outputs;
    uint32_t target_dims[1] = {target_dim};
    nimcp_tensor_t* target = nimcp_tensor_create(target_dims, 1, NIMCP_DTYPE_F32);
    if (!target) return -1.0f;

    float* target_data = (float*)nimcp_tensor_data(target);
    memset(target_data, 0, target_dim * sizeof(float));
    if (label_idx < target_dim) {
        target_data[label_idx] = 1.0f;
    }

    /* Backward + step */
    nimcp_error_t err = cnn_trainer_backward(proc->trainer, target, &proc->last_fwd);
    nimcp_tensor_destroy(target);

    if (err != NIMCP_SUCCESS) return -1.0f;

    cnn_trainer_step(proc->trainer);
    proc->backward_steps++;

    /* Compute loss approximation from last forward output vs one-hot */
    float loss = 0.0f;
    if (proc->last_fwd.output) {
        const float* out = (const float*)nimcp_tensor_data_const(proc->last_fwd.output);
        size_t numel = nimcp_tensor_numel(proc->last_fwd.output);
        uint32_t n = (target_dim < (uint32_t)numel) ? target_dim : (uint32_t)numel;
        for (uint32_t i = 0; i < n; i++) {
            float tgt = (i == label_idx) ? 1.0f : 0.0f;
            float diff = out[i] - tgt;
            loss += diff * diff;
        }
        if (n > 0) loss /= (float)n;
    }

    proc->last_loss = loss;
    if (proc->ema_loss < 0.0f) {
        proc->ema_loss = loss;
    } else {
        proc->ema_loss = 0.99f * proc->ema_loss + 0.01f * loss;
    }

    return loss;
}

/* ========================================================================= */
/* Public API: Cross-Modal Attention Fusion                                   */
/* ========================================================================= */

uint32_t cortex_cnn_fuse(cortex_cnn_processor_t* procs[], uint32_t count,
                          float* fused_out, uint32_t max_dim) {
    if (!procs || !fused_out || max_dim == 0 || count == 0) return 0;

    /* Count active processors and compute total embedding dim */
    uint32_t active_count = 0;
    uint32_t total_dim = 0;
    float confidences[CORTEX_CNN_COUNT] = {0};
    uint32_t active_indices[CORTEX_CNN_COUNT] = {0};

    for (uint32_t i = 0; i < count && i < CORTEX_CNN_COUNT; i++) {
        if (procs[i] && procs[i]->has_fwd_result && procs[i]->embedding) {
            confidences[active_count] = procs[i]->confidence;
            active_indices[active_count] = i;
            total_dim += procs[i]->embedding_dim;
            active_count++;
        }
    }

    if (active_count == 0) return 0;
    if (total_dim > max_dim) total_dim = max_dim;

    /* Compute attention weights via softmax(confidence / temperature) */
    float attention[CORTEX_CNN_COUNT] = {0};
    float max_conf = -1e30f;
    for (uint32_t i = 0; i < active_count; i++) {
        float c = confidences[i] / CORTEX_CNN_ATTENTION_TEMP;
        if (c > max_conf) max_conf = c;
    }
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < active_count; i++) {
        attention[i] = expf(confidences[i] / CORTEX_CNN_ATTENTION_TEMP - max_conf);
        sum_exp += attention[i];
    }
    if (sum_exp > 0.0f) {
        for (uint32_t i = 0; i < active_count; i++) {
            attention[i] /= sum_exp;
        }
    }

    /* Concatenate attention-weighted embeddings */
    uint32_t offset = 0;
    for (uint32_t i = 0; i < active_count && offset < max_dim; i++) {
        uint32_t idx = active_indices[i];
        cortex_cnn_processor_t* p = procs[idx];
        uint32_t dim = p->embedding_dim;
        if (offset + dim > max_dim) dim = max_dim - offset;

        float w = attention[i];
        for (uint32_t j = 0; j < dim; j++) {
            fused_out[offset + j] = w * p->embedding[j];
        }
        offset += dim;
    }

    return offset;
}

/* ========================================================================= */
/* Public API: Metrics                                                        */
/* ========================================================================= */

int cortex_cnn_get_metrics(const cortex_cnn_processor_t* proc, cortex_cnn_metrics_t* out) {
    if (!proc || !out) return -1;

    out->type = proc->type;
    out->last_loss = proc->last_loss;
    out->ema_loss = (proc->ema_loss >= 0.0f) ? proc->ema_loss : 0.0f;
    out->forward_steps = proc->forward_steps;
    out->backward_steps = proc->backward_steps;
    out->confidence = proc->confidence;
    out->embedding_dim = proc->embedding_dim;
    out->num_params = proc->num_params;

    /* Compute embedding L2 norm */
    float norm = 0.0f;
    if (proc->embedding) {
        for (uint32_t i = 0; i < proc->embedding_dim; i++) {
            norm += proc->embedding[i] * proc->embedding[i];
        }
    }
    out->embedding_norm = sqrtf(norm);

    return 0;
}

const float* cortex_cnn_get_embedding(const cortex_cnn_processor_t* proc, uint32_t* dim) {
    if (!proc || !proc->has_fwd_result) {
        if (dim) *dim = 0;
        return NULL;
    }
    if (dim) *dim = proc->embedding_dim;
    return proc->embedding;
}

cortex_cnn_type_t cortex_cnn_get_type(const cortex_cnn_processor_t* proc) {
    return proc ? proc->type : CORTEX_CNN_VISUAL;
}

const char* cortex_cnn_type_name(cortex_cnn_type_t type) {
    if (type >= 0 && type < CORTEX_CNN_COUNT) return cortex_type_names[type];
    return "Unknown";
}

/* ========================================================================= */
/* UTM Adapter                                                                */
/* ========================================================================= */

typedef struct {
    cortex_cnn_processor_t* proc;  /* NOT owned */
    uint32_t input_dim;
    uint32_t output_dim;
} cortex_cnn_adapter_ctx_t;

static int cortex_adapter_forward(void* ctx, const float* input, uint32_t input_dim,
                                   float* output, uint32_t output_dim) {
    cortex_cnn_adapter_ctx_t* a = (cortex_cnn_adapter_ctx_t*)ctx;
    if (!a || !a->proc || !input || !output) return -1;

    /* Use the generic 1D forward — works for all modalities when data is
     * already pre-processed to flat float representation */
    const float* emb = cortex_forward_1d(a->proc, input, input_dim);
    if (!emb) return -1;

    uint32_t copy_dim = (output_dim < a->proc->embedding_dim) ?
                         output_dim : a->proc->embedding_dim;
    memcpy(output, emb, copy_dim * sizeof(float));
    if (output_dim > copy_dim) {
        memset(output + copy_dim, 0, (output_dim - copy_dim) * sizeof(float));
    }
    return 0;
}

static int cortex_adapter_backward(void* ctx, const float* dl_doutput, uint32_t output_dim,
                                    float* dl_dinput, uint32_t input_dim) {
    cortex_cnn_adapter_ctx_t* a = (cortex_cnn_adapter_ctx_t*)ctx;
    if (!a || !a->proc || !a->proc->trainer || !dl_doutput) return -1;
    if (!a->proc->has_fwd_result) return -1;

    /* Pass external gradient to CNN backward */
    uint32_t dims[1] = {output_dim};
    nimcp_tensor_t* grad_t = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    if (!grad_t) return -1;

    memcpy(nimcp_tensor_data(grad_t), dl_doutput, output_dim * sizeof(float));

    nimcp_error_t err = cnn_trainer_backward_with_gradient(
        a->proc->trainer, grad_t, &a->proc->last_fwd);

    if (err == NIMCP_SUCCESS) {
        cnn_trainer_step(a->proc->trainer);
        a->proc->backward_steps++;
    }

    /* Copy input gradients if available */
    if (dl_dinput) {
        const nimcp_tensor_t* in_grad = cnn_trainer_get_input_grad(a->proc->trainer);
        if (in_grad) {
            const float* grad_data = (const float*)nimcp_tensor_data_const(in_grad);
            size_t grad_numel = nimcp_tensor_numel(in_grad);
            uint32_t copy = (input_dim < (uint32_t)grad_numel) ?
                             input_dim : (uint32_t)grad_numel;
            memcpy(dl_dinput, grad_data, copy * sizeof(float));
            if (input_dim > copy) {
                memset(dl_dinput + copy, 0, (input_dim - copy) * sizeof(float));
            }
        } else {
            memset(dl_dinput, 0, input_dim * sizeof(float));
        }
    }

    nimcp_tensor_destroy(grad_t);
    return (err == NIMCP_SUCCESS) ? 0 : -1;
}

static int cortex_adapter_get_param_groups(void* ctx,
                                            nimcp_utm_param_group_t** groups,
                                            uint32_t* num_groups) {
    (void)ctx;
    /* Cortex CNNs manage their own weights via internal tensors — same pattern as CNN adapter */
    if (groups) *groups = NULL;
    if (num_groups) *num_groups = 0;
    return 0;
}

static int cortex_adapter_zero_grad(void* ctx) {
    cortex_cnn_adapter_ctx_t* a = (cortex_cnn_adapter_ctx_t*)ctx;
    if (!a || !a->proc || !a->proc->trainer) return -1;
    cnn_trainer_zero_grad(a->proc->trainer);
    return 0;
}

static uint32_t cortex_adapter_get_output_dim(void* ctx) {
    cortex_cnn_adapter_ctx_t* a = (cortex_cnn_adapter_ctx_t*)ctx;
    return a ? a->output_dim : 0;
}

static uint32_t cortex_adapter_get_input_dim(void* ctx) {
    cortex_cnn_adapter_ctx_t* a = (cortex_cnn_adapter_ctx_t*)ctx;
    return a ? a->input_dim : 0;
}

static float cortex_adapter_auxiliary_loss(void* ctx) {
    cortex_cnn_adapter_ctx_t* a = (cortex_cnn_adapter_ctx_t*)ctx;
    if (!a || !a->proc) return 0.0f;
    /* Use tracked loss as auxiliary loss signal */
    return (a->proc->last_loss >= 0.0f) ? 0.001f * a->proc->last_loss : 0.0f;
}

static void cortex_adapter_destroy(void* ctx) {
    cortex_cnn_adapter_ctx_t* a = (cortex_cnn_adapter_ctx_t*)ctx;
    if (!a) return;
    /* Don't free proc — not owned */
    nimcp_free(a);
}

static const char* cortex_adapter_names[CORTEX_CNN_COUNT] = {
    "CortexCNN_Visual", "CortexCNN_Audio", "CortexCNN_Speech", "CortexCNN_Somato"
};

/* Per-type vtable instances */
static const nimcp_trainable_network_ops_t cortex_cnn_ops[CORTEX_CNN_COUNT] = {
    [CORTEX_CNN_VISUAL] = {
        .name = "CortexCNN_Visual",
        .type = NIMCP_TRAINABLE_CUSTOM,
        .forward = cortex_adapter_forward,
        .backward = cortex_adapter_backward,
        .get_param_groups = cortex_adapter_get_param_groups,
        .zero_grad = cortex_adapter_zero_grad,
        .get_output_dim = cortex_adapter_get_output_dim,
        .get_input_dim = cortex_adapter_get_input_dim,
        .compute_auxiliary_loss = cortex_adapter_auxiliary_loss,
        .destroy = cortex_adapter_destroy,
        .sync_params = NULL,
    },
    [CORTEX_CNN_AUDIO] = {
        .name = "CortexCNN_Audio",
        .type = NIMCP_TRAINABLE_CUSTOM,
        .forward = cortex_adapter_forward,
        .backward = cortex_adapter_backward,
        .get_param_groups = cortex_adapter_get_param_groups,
        .zero_grad = cortex_adapter_zero_grad,
        .get_output_dim = cortex_adapter_get_output_dim,
        .get_input_dim = cortex_adapter_get_input_dim,
        .compute_auxiliary_loss = cortex_adapter_auxiliary_loss,
        .destroy = cortex_adapter_destroy,
        .sync_params = NULL,
    },
    [CORTEX_CNN_SPEECH] = {
        .name = "CortexCNN_Speech",
        .type = NIMCP_TRAINABLE_CUSTOM,
        .forward = cortex_adapter_forward,
        .backward = cortex_adapter_backward,
        .get_param_groups = cortex_adapter_get_param_groups,
        .zero_grad = cortex_adapter_zero_grad,
        .get_output_dim = cortex_adapter_get_output_dim,
        .get_input_dim = cortex_adapter_get_input_dim,
        .compute_auxiliary_loss = cortex_adapter_auxiliary_loss,
        .destroy = cortex_adapter_destroy,
        .sync_params = NULL,
    },
    [CORTEX_CNN_SOMATO] = {
        .name = "CortexCNN_Somato",
        .type = NIMCP_TRAINABLE_CUSTOM,
        .forward = cortex_adapter_forward,
        .backward = cortex_adapter_backward,
        .get_param_groups = cortex_adapter_get_param_groups,
        .zero_grad = cortex_adapter_zero_grad,
        .get_output_dim = cortex_adapter_get_output_dim,
        .get_input_dim = cortex_adapter_get_input_dim,
        .compute_auxiliary_loss = cortex_adapter_auxiliary_loss,
        .destroy = cortex_adapter_destroy,
        .sync_params = NULL,
    },
};

int cortex_cnn_utm_adapter_create(cortex_cnn_processor_t* proc,
                                   const nimcp_trainable_network_ops_t** ops,
                                   void** ctx) {
    if (!proc || !ops || !ctx) return -1;

    cortex_cnn_adapter_ctx_t* a = (cortex_cnn_adapter_ctx_t*)nimcp_calloc(
        1, sizeof(cortex_cnn_adapter_ctx_t));
    if (!a) return -1;

    a->proc = proc;
    a->output_dim = proc->embedding_dim;

    /* Input dim depends on modality */
    switch (proc->type) {
        case CORTEX_CNN_VISUAL:  a->input_dim = 64 * 64 * 3; break;
        case CORTEX_CNN_AUDIO:   a->input_dim = 128; break;
        case CORTEX_CNN_SPEECH:  a->input_dim = 64; break;
        case CORTEX_CNN_SOMATO:  a->input_dim = 45; break;
        default: a->input_dim = 64; break;
    }

    *ops = &cortex_cnn_ops[proc->type];
    *ctx = a;

    NIMCP_LOGGING_INFO("Created UTM adapter for %s cortex CNN (in=%u, out=%u)",
                      cortex_type_names[proc->type], a->input_dim, a->output_dim);
    return 0;
}

/* ========================================================================= */
/* Checkpoint Save/Load                                                       */
/* ========================================================================= */

int cortex_cnn_save(const cortex_cnn_processor_t* proc, const char* path) {
    if (!proc || !proc->trainer || !path) return -1;
    return cnn_trainer_save(proc->trainer, path);
}

int cortex_cnn_load(cortex_cnn_processor_t* proc, const char* path) {
    if (!proc || !proc->trainer || !path) return -1;
    return cnn_trainer_load_weights(proc->trainer, path);
}
