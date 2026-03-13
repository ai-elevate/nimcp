/**
 * @file nimcp_cortex_cnn.h
 * @brief Per-cortex CNN processors for modality-specific feature extraction
 *
 * WHAT: Independent CNN processors for visual, audio, speech, and somatosensory cortices
 * WHY:  Replace single classifier head with per-modality convolutional architectures
 *       that learn from raw sensory data via gradient-based training
 * HOW:  Each cortex CNN wraps a cnn_trainer_t with modality-specific architecture,
 *       registers via UTM vtable for composite loss / cross-network bridges,
 *       and fuses embeddings via cross-modal attention before adaptive network
 *
 * Architecture:
 *   Visual (64x64x3): Conv2d(3->16,3x3)->BN->ReLU->Pool->Conv2d(16->32)->BN->ReLU->Pool
 *                      ->Conv2d(32->64)->BN->ReLU->GlobalAvgPool->Dense(64->64) => 64-dim
 *   Audio (1x128):     Conv2d(1->16,1x5)->BN->ReLU->Pool->Conv2d(16->32,1x3)->BN->ReLU
 *                      ->GlobalAvgPool->Dense(32->64) => 64-dim
 *   Speech (1x64):     Conv2d(1->16,1x3)->BN->ReLU->Conv2d(16->32,1x3)->BN->ReLU
 *                      ->GlobalAvgPool->Dense(32->32) => 32-dim
 *   Somato (45):       Dense(45->64)->ReLU->Dense(64->32) => 32-dim
 *
 * ~46K total params, <1MB memory, CPU-first.
 */

#ifndef NIMCP_CORTEX_CNN_H
#define NIMCP_CORTEX_CNN_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct cnn_trainer_s;
struct nimcp_trainable_network_ops;
typedef struct nimcp_trainable_network_ops nimcp_trainable_network_ops_t;

/* ========================================================================= */
/* Types                                                                      */
/* ========================================================================= */

/** Cortex modality types */
typedef enum cortex_cnn_type {
    CORTEX_CNN_VISUAL  = 0,  /**< Visual cortex: processes raw pixel data */
    CORTEX_CNN_AUDIO   = 1,  /**< Audio cortex: processes mel spectrograms */
    CORTEX_CNN_SPEECH  = 2,  /**< Speech cortex: processes phoneme features */
    CORTEX_CNN_SOMATO  = 3,  /**< Somatosensory cortex: processes body state */
    CORTEX_CNN_COUNT   = 4
} cortex_cnn_type_t;

/** Per-cortex metrics for monitoring */
typedef struct cortex_cnn_metrics {
    cortex_cnn_type_t type;
    float last_loss;             /**< Loss from last backward pass */
    float ema_loss;              /**< EMA of loss (alpha=0.01) */
    uint64_t forward_steps;      /**< Total forward passes */
    uint64_t backward_steps;     /**< Total backward passes */
    float embedding_norm;        /**< L2 norm of last embedding */
    float confidence;            /**< Last forward softmax max (for attention) */
    uint32_t embedding_dim;      /**< Output embedding dimension */
    uint32_t num_params;         /**< Approximate parameter count */
} cortex_cnn_metrics_t;

/** Cortex CNN processor (opaque) */
typedef struct cortex_cnn_processor cortex_cnn_processor_t;

/* ========================================================================= */
/* Lifecycle                                                                  */
/* ========================================================================= */

/**
 * @brief Create a cortex CNN processor for a specific modality
 *
 * WHAT: Factory that builds modality-specific CNN architecture
 * WHY:  Each modality has different input shapes and optimal architectures
 * HOW:  Creates cnn_trainer_t, adds modality-specific layers via cnn_trainer_add_*
 *
 * @param type       Modality type (VISUAL, AUDIO, SPEECH, SOMATO)
 * @param embedding_dim  Output embedding dimension (0 = use default for type)
 * @return Processor handle or NULL on failure
 */
cortex_cnn_processor_t* cortex_cnn_create(cortex_cnn_type_t type, uint32_t embedding_dim);

/**
 * @brief Destroy a cortex CNN processor and free all resources
 * @param proc  Processor to destroy (NULL-safe)
 */
void cortex_cnn_destroy(cortex_cnn_processor_t* proc);

/* ========================================================================= */
/* Forward Pass (modality-specific input formats)                             */
/* ========================================================================= */

/**
 * @brief Forward pass for visual cortex (raw pixel data)
 *
 * @param proc    Visual cortex processor
 * @param pixels  Raw pixel data (uint8, HWC layout)
 * @param w       Image width
 * @param h       Image height
 * @param ch      Number of channels (1 or 3)
 * @return Embedding pointer (owned by proc, valid until next forward), NULL on error
 */
const float* cortex_cnn_forward_visual(cortex_cnn_processor_t* proc,
                                        const uint8_t* pixels,
                                        uint32_t w, uint32_t h, uint32_t ch);

/**
 * @brief Forward pass for audio cortex (mel spectrogram)
 *
 * @param proc      Audio cortex processor
 * @param mel       Mel spectrogram features (float array)
 * @param mel_size  Number of mel bins
 * @return Embedding pointer (owned by proc), NULL on error
 */
const float* cortex_cnn_forward_audio(cortex_cnn_processor_t* proc,
                                       const float* mel, uint32_t mel_size);

/**
 * @brief Forward pass for speech cortex (phoneme features)
 *
 * @param proc  Speech cortex processor
 * @param phonemes  Phoneme feature vector
 * @param size      Number of features
 * @return Embedding pointer (owned by proc), NULL on error
 */
const float* cortex_cnn_forward_speech(cortex_cnn_processor_t* proc,
                                        const float* phonemes, uint32_t size);

/**
 * @brief Forward pass for somatosensory cortex (body state)
 *
 * @param proc        Somato cortex processor
 * @param segments    Per-segment body state vector
 * @param n_segments  Number of segments
 * @return Embedding pointer (owned by proc), NULL on error
 */
const float* cortex_cnn_forward_somato(cortex_cnn_processor_t* proc,
                                        const float* segments, uint32_t n_segments);

/* ========================================================================= */
/* Backward Pass / Training                                                   */
/* ========================================================================= */

/**
 * @brief Train the cortex CNN on a labeled example
 *
 * WHAT: Run backward pass + optimizer step on the most recent forward result
 * WHY:  Each cortex CNN learns independently from its modality data
 * HOW:  Build one-hot from label, run cnn_trainer_backward + cnn_trainer_step
 *
 * @param proc        Processor with a valid last forward result
 * @param label       Label string (looked up or created in internal label map)
 * @param num_outputs Number of output classes
 * @return Loss value, or -1.0f on error
 */
float cortex_cnn_backward(cortex_cnn_processor_t* proc,
                           const char* label, uint32_t num_outputs);

/* ========================================================================= */
/* Cross-Modal Attention Fusion                                               */
/* ========================================================================= */

/**
 * @brief Fuse embeddings from multiple cortex CNNs via attention-weighted concatenation
 *
 * WHAT: Combine per-cortex embeddings into a single fused vector
 * WHY:  Multi-modal integration weighted by per-cortex confidence
 * HOW:  attention[i] = softmax(confidence[i] / temperature)
 *       fused = concat(attention[i] * embedding[i] for each active cortex)
 *
 * @param procs     Array of cortex processors (NULL entries skipped)
 * @param count     Number of entries in procs array
 * @param fused_out Output buffer for fused embedding
 * @param max_dim   Maximum output dimension (buffer size)
 * @return Actual fused dimension written, or 0 on error
 */
uint32_t cortex_cnn_fuse(cortex_cnn_processor_t* procs[], uint32_t count,
                          float* fused_out, uint32_t max_dim);

/* ========================================================================= */
/* Metrics / Introspection                                                    */
/* ========================================================================= */

/**
 * @brief Get current metrics for a cortex CNN processor
 * @param proc  Processor to query
 * @param out   Output metrics struct
 * @return 0 on success, -1 on error
 */
int cortex_cnn_get_metrics(const cortex_cnn_processor_t* proc, cortex_cnn_metrics_t* out);

/**
 * @brief Get the embedding from the last forward pass
 * @param proc  Processor to query
 * @param dim   Output: embedding dimension
 * @return Embedding pointer (owned by proc), NULL if no forward pass done
 */
const float* cortex_cnn_get_embedding(const cortex_cnn_processor_t* proc, uint32_t* dim);

/**
 * @brief Get the cortex type of a processor
 */
cortex_cnn_type_t cortex_cnn_get_type(const cortex_cnn_processor_t* proc);

/**
 * @brief Get human-readable name for cortex type
 */
const char* cortex_cnn_type_name(cortex_cnn_type_t type);

/* ========================================================================= */
/* UTM Integration                                                            */
/* ========================================================================= */

/**
 * @brief Create a UTM trainable adapter for a cortex CNN
 *
 * WHAT: Wrap cortex CNN behind nimcp_trainable_network_ops_t vtable
 * WHY:  Register in UTM for composite loss, cross-network bridges, anti-collapse
 * HOW:  Adapter translates flat float I/O to cortex CNN's tensor-based API
 *
 * @param proc  Cortex CNN processor (must outlive adapter)
 * @param ops   Output: vtable pointer
 * @param ctx   Output: adapter context (opaque)
 * @return 0 on success, -1 on error
 */
int cortex_cnn_utm_adapter_create(cortex_cnn_processor_t* proc,
                                   const nimcp_trainable_network_ops_t** ops,
                                   void** ctx);

/* ========================================================================= */
/* Checkpoint                                                                 */
/* ========================================================================= */

/**
 * @brief Save cortex CNN weights to file
 * @return 0 on success, -1 on error
 */
int cortex_cnn_save(const cortex_cnn_processor_t* proc, const char* path);

/**
 * @brief Load cortex CNN weights from file
 * @return 0 on success, -1 on error
 */
int cortex_cnn_load(cortex_cnn_processor_t* proc, const char* path);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_CORTEX_CNN_H */
