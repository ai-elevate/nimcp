/**
 * @file nimcp_cnn_training.c
 * @brief CNN Training Implementation
 *
 * WHAT: Convolutional Neural Network training with SNN conversion
 * WHY:  Enable hybrid CNN+SNN architectures in NIMCP
 * HOW:  Classical backpropagation + rate/spike encoding bridges
 *
 * @author NIMCP Development Team
 * @date 2025-12-20
 */

#include "training/nimcp_cnn_training.h"
#include "utils/validation/nimcp_common.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief CNN layer internal structure
 */
struct cnn_layer_s {
    cnn_layer_type_t type;          /**< Layer type */
    nimcp_tensor_t* weights;        /**< Layer weights */
    nimcp_tensor_t* bias;           /**< Layer bias */
    nimcp_tensor_t* weight_grad;    /**< Weight gradients */
    nimcp_tensor_t* bias_grad;      /**< Bias gradients */

    /* Batch norm running stats */
    nimcp_tensor_t* running_mean;
    nimcp_tensor_t* running_var;

    /* Layer-specific config (union to save memory) */
    union {
        cnn_conv_config_t conv;
        cnn_pool_config_t pool;
        cnn_batch_norm_config_t batch_norm;
        cnn_dropout_config_t dropout;
        cnn_dense_config_t dense;
    } config;

    /* Shape tracking */
    nimcp_tensor_shape_t input_shape;
    nimcp_tensor_shape_t output_shape;

    cnn_layer_t* next;              /**< Linked list */
};

/**
 * @brief CNN trainer internal structure
 */
struct cnn_trainer_s {
    /* Architecture */
    cnn_layer_t* layers_head;       /**< Linked list of layers */
    cnn_layer_t* layers_tail;
    uint32_t num_layers;

    /* Training infrastructure */
    nimcp_optimizer_ctx_t* optimizer;
    nimcp_gradient_manager_ctx_t* grad_manager;
    nimcp_loss_ctx_t* loss_fn;

    /* Configuration */
    cnn_trainer_config_t config;

    /* Training state */
    uint32_t current_epoch;
    uint32_t global_step;
    float best_val_loss;
    uint32_t epochs_no_improve;
    bool training_mode;             /**< true=train, false=eval */

    /* Bio-async integration */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    /* Statistics */
    uint64_t total_forward_calls;
    uint64_t total_backward_calls;
    double total_forward_time_ms;
    double total_backward_time_ms;
};

/**
 * @brief Data loader internal structure
 */
struct cnn_dataloader_s {
    nimcp_tensor_t* data;           /**< Full dataset */
    nimcp_tensor_t* labels;         /**< Full labels */
    uint32_t* indices;              /**< Shuffled indices */
    uint32_t dataset_size;
    uint32_t current_index;
    cnn_dataloader_config_t config;
    uint32_t seed;                  /**< Random seed for shuffling */
};

/**
 * @brief CNN-to-SNN converter internal structure
 */
struct cnn_to_snn_converter_s {
    cnn_to_snn_config_t config;

    /* Calibration statistics */
    nimcp_tensor_t** activation_stats; /**< Per-layer activation statistics */
    uint32_t num_layers;

    /* Conversion mappings */
    float* firing_rate_scales;      /**< Per-layer rate normalization factors */
    float* threshold_values;        /**< Per-layer SNN thresholds */
};

//=============================================================================
// Default Configurations
//=============================================================================

/**
 * @brief Get default trainer configuration
 *
 * WHAT: Initialize config with sensible hyperparameters
 * WHY:  Simplify setup for common use cases
 * HOW:  Set Adam optimizer, cross-entropy loss, standard batch size
 */
nimcp_error_t cnn_trainer_default_config(cnn_trainer_config_t* config) {
    /* Guard clause: null check */
    if (!config) {
        NIMCP_LOGGING_ERROR("cnn_trainer_default_config: NULL config pointer");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Zero-initialize */
    memset(config, 0, sizeof(cnn_trainer_config_t));

    /* Optimizer: Adam with standard hyperparameters */
    config->optimizer_config.type = NIMCP_OPTIMIZER_ADAM;
    config->optimizer_config.config.adam = (nimcp_adam_config_t){
        .learning_rate = 0.001f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .epsilon = 1e-8f,
        .weight_decay = 0.0f,
        .amsgrad = false
    };

    /* Loss: Cross-entropy for classification */
    config->loss_type = NIMCP_LOSS_CROSS_ENTROPY;

    /* Training hyperparameters */
    config->max_epochs = 100;
    config->learning_rate = 0.001f;
    config->weight_decay = 0.0f;
    config->gradient_clip_value = 5.0f;

    /* Data loader */
    config->dataloader.batch_size = 32;
    config->dataloader.shuffle = true;
    config->dataloader.num_workers = 1;
    config->dataloader.pin_memory = false;
    config->dataloader.drop_last = false;

    /* Validation */
    config->use_validation = true;
    config->validation_split = 0.2f;
    config->validation_frequency = 1;

    /* Early stopping */
    config->use_early_stopping = true;
    config->patience = 10;
    config->min_delta = 1e-4f;

    /* Checkpointing */
    config->save_checkpoints = false;
    config->checkpoint_frequency = 5;
    config->checkpoint_dir = "./checkpoints";

    /* Bio-async */
    config->enable_bio_async = false;

    /* Logging */
    config->verbose = true;
    config->log_frequency = 10;

    return NIMCP_SUCCESS;
}

//=============================================================================
// Trainer Lifecycle
//=============================================================================

/**
 * @brief Create CNN trainer
 *
 * WHAT: Allocate and initialize CNN trainer context
 * WHY:  Set up infrastructure for backpropagation-based training
 * HOW:  Create optimizer, gradient manager, loss function
 */
cnn_trainer_t* cnn_trainer_create(const cnn_trainer_config_t* config) {
    /* Guard clause: null check */
    if (!config) {
        NIMCP_LOGGING_ERROR("cnn_trainer_create: NULL config");
        return NULL;
    }

    /* Allocate trainer */
    cnn_trainer_t* trainer = (cnn_trainer_t*)nimcp_malloc(sizeof(cnn_trainer_t));
    if (!trainer) {
        NIMCP_LOGGING_ERROR("cnn_trainer_create: Failed to allocate trainer");
        return NULL;
    }

    /* Zero-initialize */
    memset(trainer, 0, sizeof(cnn_trainer_t));

    /* Copy configuration */
    memcpy(&trainer->config, config, sizeof(cnn_trainer_config_t));

    /* Initialize optimizer */
    trainer->optimizer = nimcp_optimizer_create(&config->optimizer_config);
    if (!trainer->optimizer) {
        NIMCP_LOGGING_ERROR("cnn_trainer_create: Failed to create optimizer");
        nimcp_free(trainer);
        return NULL;
    }

    /* Initialize gradient manager */
    trainer->grad_manager = nimcp_gradient_manager_create(&config->gradient_config);
    if (!trainer->grad_manager) {
        NIMCP_LOGGING_ERROR("cnn_trainer_create: Failed to create gradient manager");
        nimcp_optimizer_destroy(trainer->optimizer);
        nimcp_free(trainer);
        return NULL;
    }

    /* Initialize loss function (requires loss-specific config) */
    /* TODO: Create loss function based on config->loss_type */

    /* Initialize state */
    trainer->current_epoch = 0;
    trainer->global_step = 0;
    trainer->best_val_loss = FLT_MAX;
    trainer->epochs_no_improve = 0;
    trainer->training_mode = true;

    /* Bio-async integration */
    if (config->enable_bio_async) {
        cnn_connect_bio_async(trainer);
    }

    NIMCP_LOGGING_INFO("CNN trainer created successfully");
    return trainer;
}

/**
 * @brief Destroy CNN trainer
 *
 * WHAT: Free all trainer resources
 * WHY:  Prevent memory leaks
 * HOW:  Release layers, optimizer, gradient manager
 */
void cnn_trainer_destroy(cnn_trainer_t* trainer) {
    /* Guard clause: null check */
    if (!trainer) {
        return;
    }

    /* Disconnect bio-async */
    if (trainer->bio_async_enabled) {
        cnn_disconnect_bio_async(trainer);
    }

    /* Destroy layers */
    cnn_layer_t* layer = trainer->layers_head;
    while (layer) {
        cnn_layer_t* next = layer->next;

        /* Free layer tensors */
        if (layer->weights) nimcp_tensor_destroy(layer->weights);
        if (layer->bias) nimcp_tensor_destroy(layer->bias);
        if (layer->weight_grad) nimcp_tensor_destroy(layer->weight_grad);
        if (layer->bias_grad) nimcp_tensor_destroy(layer->bias_grad);
        if (layer->running_mean) nimcp_tensor_destroy(layer->running_mean);
        if (layer->running_var) nimcp_tensor_destroy(layer->running_var);

        nimcp_free(layer);
        layer = next;
    }

    /* Destroy optimizer */
    if (trainer->optimizer) {
        nimcp_optimizer_destroy(trainer->optimizer);
    }

    /* Destroy gradient manager */
    if (trainer->grad_manager) {
        nimcp_gradient_manager_destroy(trainer->grad_manager);
    }

    /* Destroy loss function */
    if (trainer->loss_fn) {
        nimcp_loss_destroy(trainer->loss_fn);
    }

    nimcp_free(trainer);
    NIMCP_LOGGING_INFO("CNN trainer destroyed");
}

//=============================================================================
// Layer Construction Stubs
//=============================================================================

/**
 * @brief Add convolution layer
 *
 * WHAT: Append Conv2D layer to network
 * WHY:  Build hierarchical feature extraction
 * HOW:  Allocate weights (He initialization), append to layer list
 */
cnn_layer_t* cnn_trainer_add_conv_layer(cnn_trainer_t* trainer,
                                        const cnn_conv_config_t* config) {
    /* Guard clauses */
    if (!trainer) {
        NIMCP_LOGGING_ERROR("cnn_trainer_add_conv_layer: NULL trainer");
        return NULL;
    }
    if (!config) {
        NIMCP_LOGGING_ERROR("cnn_trainer_add_conv_layer: NULL config");
        return NULL;
    }

    /* TODO: Implement layer creation */
    /* - Allocate cnn_layer_t */
    /* - Initialize weights: (out_channels, in_channels, kernel_h, kernel_w) */
    /* - He initialization: std = sqrt(2 / fan_in) */
    /* - Append to trainer->layers list */

    NIMCP_LOGGING_WARN("cnn_trainer_add_conv_layer: Not yet implemented");
    return NULL;
}

cnn_layer_t* cnn_trainer_add_pool_layer(cnn_trainer_t* trainer,
                                        const cnn_pool_config_t* config) {
    if (!trainer || !config) return NULL;
    NIMCP_LOGGING_WARN("cnn_trainer_add_pool_layer: Not yet implemented");
    return NULL;
}

cnn_layer_t* cnn_trainer_add_batch_norm_layer(cnn_trainer_t* trainer,
                                               const cnn_batch_norm_config_t* config) {
    if (!trainer || !config) return NULL;
    NIMCP_LOGGING_WARN("cnn_trainer_add_batch_norm_layer: Not yet implemented");
    return NULL;
}

cnn_layer_t* cnn_trainer_add_dropout_layer(cnn_trainer_t* trainer,
                                            const cnn_dropout_config_t* config) {
    if (!trainer || !config) return NULL;
    NIMCP_LOGGING_WARN("cnn_trainer_add_dropout_layer: Not yet implemented");
    return NULL;
}

cnn_layer_t* cnn_trainer_add_dense_layer(cnn_trainer_t* trainer,
                                          const cnn_dense_config_t* config) {
    if (!trainer || !config) return NULL;
    NIMCP_LOGGING_WARN("cnn_trainer_add_dense_layer: Not yet implemented");
    return NULL;
}

cnn_layer_t* cnn_trainer_add_flatten_layer(cnn_trainer_t* trainer) {
    if (!trainer) return NULL;
    NIMCP_LOGGING_WARN("cnn_trainer_add_flatten_layer: Not yet implemented");
    return NULL;
}

//=============================================================================
// Training API Stubs
//=============================================================================

/**
 * @brief Forward propagation
 *
 * WHAT: Compute network output and cache activations
 * WHY:  Generate predictions for loss computation
 * HOW:  Sequential layer-wise forward passes
 */
nimcp_error_t cnn_trainer_forward(cnn_trainer_t* trainer,
                                   const nimcp_tensor_t* input,
                                   cnn_forward_result_t* result) {
    /* Guard clauses */
    if (!trainer) {
        NIMCP_LOGGING_ERROR("cnn_trainer_forward: NULL trainer");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (!input) {
        NIMCP_LOGGING_ERROR("cnn_trainer_forward: NULL input");
        return NIMCP_ERROR_NULL_POINTER;
    }
    if (!result) {
        NIMCP_LOGGING_ERROR("cnn_trainer_forward: NULL result");
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement forward pass */
    /* - Loop through layers in order */
    /* - Apply layer operations (conv, pool, activation) */
    /* - Cache activations for backprop */
    /* - Return final output */

    NIMCP_LOGGING_WARN("cnn_trainer_forward: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

nimcp_error_t cnn_trainer_backward(cnn_trainer_t* trainer,
                                    const nimcp_tensor_t* target,
                                    const cnn_forward_result_t* forward_result) {
    if (!trainer || !target || !forward_result) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement backpropagation */
    /* - Compute loss gradient */
    /* - Reverse-order layer-wise gradient computation */
    /* - Store gradients in layer->weight_grad, layer->bias_grad */

    NIMCP_LOGGING_WARN("cnn_trainer_backward: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

nimcp_error_t cnn_trainer_step(cnn_trainer_t* trainer) {
    if (!trainer) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement optimizer step */
    /* - Call optimizer to update weights */
    /* - Clear gradients for next iteration */

    NIMCP_LOGGING_WARN("cnn_trainer_step: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

nimcp_error_t cnn_trainer_train_epoch(cnn_trainer_t* trainer,
                                       cnn_dataloader_t* dataloader,
                                       cnn_epoch_result_t* result) {
    if (!trainer || !dataloader || !result) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement epoch training loop */
    /* - Reset dataloader */
    /* - Loop over batches: */
    /*   - Get batch */
    /*   - Forward pass */
    /*   - Compute loss */
    /*   - Backward pass */
    /*   - Optimizer step */
    /*   - Accumulate statistics */

    NIMCP_LOGGING_WARN("cnn_trainer_train_epoch: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

nimcp_error_t cnn_trainer_validate(cnn_trainer_t* trainer,
                                    cnn_dataloader_t* dataloader,
                                    float* val_loss,
                                    float* val_accuracy) {
    if (!trainer || !dataloader || !val_loss || !val_accuracy) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement validation */
    /* - Set training_mode = false (disable dropout) */
    /* - Loop over validation batches */
    /* - Forward only (no backprop) */
    /* - Compute metrics */
    /* - Restore training_mode = true */

    NIMCP_LOGGING_WARN("cnn_trainer_validate: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

nimcp_error_t cnn_trainer_fit(cnn_trainer_t* trainer,
                               cnn_dataloader_t* train_loader,
                               cnn_dataloader_t* val_loader) {
    if (!trainer || !train_loader) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement full training loop */
    /* - Epoch loop up to max_epochs */
    /* - Train epoch */
    /* - Validate if val_loader provided */
    /* - Check early stopping */
    /* - Save checkpoints */

    NIMCP_LOGGING_WARN("cnn_trainer_fit: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

//=============================================================================
// Data Pipeline Stubs
//=============================================================================

cnn_dataloader_t* cnn_dataloader_create(const nimcp_tensor_t* data,
                                         const nimcp_tensor_t* labels,
                                         const cnn_dataloader_config_t* config) {
    if (!data || !labels || !config) {
        NIMCP_LOGGING_ERROR("cnn_dataloader_create: NULL arguments");
        return NULL;
    }

    /* TODO: Implement dataloader */
    /* - Allocate structure */
    /* - Store data and labels */
    /* - Initialize shuffled indices */

    NIMCP_LOGGING_WARN("cnn_dataloader_create: Not yet implemented");
    return NULL;
}

void cnn_dataloader_destroy(cnn_dataloader_t* loader) {
    if (!loader) return;

    if (loader->indices) {
        nimcp_free(loader->indices);
    }
    nimcp_free(loader);
}

nimcp_error_t cnn_dataloader_next_batch(cnn_dataloader_t* loader,
                                         nimcp_tensor_t** batch_data,
                                         nimcp_tensor_t** batch_labels) {
    if (!loader || !batch_data || !batch_labels) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement batch sampling */
    /* - Check if epoch complete */
    /* - Extract batch slice from data/labels */
    /* - Apply augmentation if enabled */
    /* - Return batch tensors */

    NIMCP_LOGGING_WARN("cnn_dataloader_next_batch: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

void cnn_dataloader_reset(cnn_dataloader_t* loader) {
    if (!loader) return;

    loader->current_index = 0;

    /* TODO: Shuffle indices if configured */
}

nimcp_error_t cnn_augment_batch(nimcp_tensor_t* batch,
                                 const cnn_augmentation_config_t* config) {
    if (!batch || !config) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement data augmentation */
    /* - Apply random transformations based on config flags */
    /* - Horizontal/vertical flips */
    /* - Rotations */
    /* - Color jitter */
    /* - Cutout */

    NIMCP_LOGGING_WARN("cnn_augment_batch: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

//=============================================================================
// CNN-to-SNN Conversion Stubs
//=============================================================================

cnn_to_snn_converter_t* cnn_to_snn_converter_create(const cnn_to_snn_config_t* config) {
    if (!config) {
        NIMCP_LOGGING_ERROR("cnn_to_snn_converter_create: NULL config");
        return NULL;
    }

    /* TODO: Implement converter creation */

    NIMCP_LOGGING_WARN("cnn_to_snn_converter_create: Not yet implemented");
    return NULL;
}

void cnn_to_snn_converter_destroy(cnn_to_snn_converter_t* converter) {
    if (!converter) return;

    /* TODO: Free conversion resources */

    nimcp_free(converter);
}

nimcp_error_t cnn_to_snn_convert(cnn_to_snn_converter_t* converter,
                                  const cnn_trainer_t* trainer,
                                  const nimcp_tensor_t* calibration_data,
                                  cnn_to_snn_result_t* result) {
    if (!converter || !trainer || !result) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement CNN-to-SNN conversion */
    /* ALGORITHM:
     * 1. Calibration: Run CNN on calibration_data, collect activation stats
     * 2. Normalization: Compute per-layer firing rate scales
     * 3. Threshold: Set SNN thresholds based on activation quantiles
     * 4. Weight conversion: Map CNN weights to SNN synapses
     * 5. Network construction: Build SNN with spatial connectivity
     * 6. Verification: Run test inputs, measure accuracy
     * 7. Optional STDP fine-tuning
     */

    NIMCP_LOGGING_WARN("cnn_to_snn_convert: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

nimcp_error_t cnn_to_snn_finetune_stdp(cnn_to_snn_result_t* result,
                                        const nimcp_tensor_t* train_data,
                                        uint32_t epochs) {
    if (!result || !train_data) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement STDP fine-tuning */
    /* - Run SNN on train_data */
    /* - Apply STDP learning rule */
    /* - Monitor accuracy improvement */

    NIMCP_LOGGING_WARN("cnn_to_snn_finetune_stdp: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

//=============================================================================
// NIMCP Integration Stubs
//=============================================================================

nimcp_error_t cnn_connect_visual_cortex(cnn_trainer_t* trainer,
                                         void* visual_cortex) {
    if (!trainer || !visual_cortex) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Integrate with visual cortex module */

    NIMCP_LOGGING_WARN("cnn_connect_visual_cortex: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

nimcp_error_t cnn_connect_audio_cortex(cnn_trainer_t* trainer,
                                        void* audio_cortex) {
    if (!trainer || !audio_cortex) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Integrate with audio cortex module */

    NIMCP_LOGGING_WARN("cnn_connect_audio_cortex: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

nimcp_error_t cnn_connect_bio_async(cnn_trainer_t* trainer) {
    if (!trainer) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* Guard clause: already connected */
    if (trainer->bio_async_enabled) {
        return NIMCP_SUCCESS;
    }

    /* Register with bio-async router */
    bio_module_info_t info = {
        .module_id = BIO_MODULE_CNN_TRAINING,
        .module_name = "cnn_training",
        .inbox_capacity = 32,
        .user_data = trainer
    };

    trainer->bio_ctx = bio_router_register_module(&info);
    if (trainer->bio_ctx) {
        trainer->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("CNN trainer connected to bio-async router");
        return NIMCP_SUCCESS;
    }

    NIMCP_LOGGING_WARN("Bio-async router not available");
    return NIMCP_ERROR_NOT_FOUND;
}

void cnn_disconnect_bio_async(cnn_trainer_t* trainer) {
    if (!trainer || !trainer->bio_async_enabled) {
        return;
    }

    if (trainer->bio_ctx) {
        bio_router_unregister_module(trainer->bio_ctx);
        trainer->bio_ctx = NULL;
    }

    trainer->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("CNN trainer disconnected from bio-async");
}

//=============================================================================
// Utility Functions
//=============================================================================

nimcp_error_t cnn_get_output_shape(cnn_layer_type_t layer_type,
                                    const nimcp_tensor_shape_t* input_shape,
                                    const void* config,
                                    nimcp_tensor_shape_t* output_shape) {
    if (!input_shape || !output_shape) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    /* TODO: Implement shape computation for each layer type */
    /* - CONV2D: (H - K + 2P) / S + 1 */
    /* - POOLING: (H - P) / S + 1 */
    /* - BATCH_NORM: Same as input */
    /* - etc. */

    NIMCP_LOGGING_WARN("cnn_get_output_shape: Not yet implemented");
    return NIMCP_ERROR_NOT_IMPLEMENTED;
}

size_t cnn_count_parameters(const cnn_trainer_t* trainer) {
    if (!trainer) return 0;

    size_t total = 0;
    cnn_layer_t* layer = trainer->layers_head;

    while (layer) {
        if (layer->weights) {
            total += layer->weights->shape.numel;
        }
        if (layer->bias) {
            total += layer->bias->shape.numel;
        }
        layer = layer->next;
    }

    return total;
}

cnn_layer_t* cnn_get_layer(const cnn_trainer_t* trainer, uint32_t layer_idx) {
    if (!trainer) return NULL;

    cnn_layer_t* layer = trainer->layers_head;
    uint32_t idx = 0;

    while (layer && idx < layer_idx) {
        layer = layer->next;
        idx++;
    }

    return layer;
}

uint32_t cnn_get_layer_count(const cnn_trainer_t* trainer) {
    return trainer ? trainer->num_layers : 0;
}
