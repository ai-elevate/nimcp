//=============================================================================
// nimcp_brain_inference.c - Brain Inference Module Implementation
//=============================================================================
/**
 * @file nimcp_brain_inference.c
 * @brief Brain inference and prediction implementation
 *
 * WHAT: Inference engine for brain predictions and decisions
 * WHY:  Separates inference logic from core brain module for modularity
 * HOW:  Forward pass, decision caching, mirror neuron integration
 *
 * EXTRACTED FROM: nimcp_brain.c (lines 5343-7087, 1744 lines)
 *
 * FUNCTIONS EXTRACTED:
 * - allocate_decision (static helper)
 * - copy_decision (static helper)
 * - perform_forward_pass (static helper)  
 * - determine_output_label (static helper)
 * - populate_interpretability (static helper)
 * - update_inference_stats (static helper)
 * - brain_decision_to_action (static helper, Phase 10.11)
 * - features_to_action (static helper, Phase 10.11)
 * - brain_decide (public API)
 * - brain_observe_action (public API, Phase 10.11)
 * - brain_free_decision (public API)
 * - brain_decide_batch (public API)
 *
 * ARCHITECTURE:
 * - Primary inference via brain_decide()
 * - Batch processing via brain_decide_batch()
 * - Mirror neuron integration via brain_observe_action()
 * - Decision caching for repeated inputs (thread-safe)
 *
 * PERFORMANCE:
 * - Inference: O(s*n) where s=sparsity, n=active_neurons
 * - Caching: O(1) cache hit, O(s*n) cache miss
 * - Batch: O(m*s*n) where m=batch_size
 */

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_unified_memory.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#include "core/brain/inference/nimcp_brain_inference.h"
#include <math.h>
#include <float.h>  // I-H3 FIX: FLT_MAX for NaN-safe argmax
#include <stdint.h> // SIZE_MAX for W6-13 overflow guard
#include <string.h>
#include <stdio.h>
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "lnn/nimcp_lnn.h"
#include "snn/nimcp_snn_network.h"
#include "training/nimcp_cnn_training.h"
#include "cognitive/nimcp_mirror_neurons.h"
#include "cognitive/nimcp_theory_of_mind.h"
#include "cognitive/nimcp_working_memory.h"
#include "async/nimcp_future.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/thread/nimcp_thread_pool.h"
#include "gpu/training/nimcp_training_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_INFERENCE"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/bridge/nimcp_bridge_boilerplate.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

BRIDGE_BOILERPLATE_MESH_ONLY(brain_inference, MESH_ADAPTER_CATEGORY_COGNITIVE)


// Forward declarations for brain internal functions (defined in nimcp_brain.c)
extern void set_error(const char* fmt, ...);
extern void brain_clear_error(void);
extern bool is_cached_input(brain_t brain, const float* features, uint32_t num_features);
extern void cache_decision(brain_t brain, const float* features, uint32_t num_features, brain_decision_t* decision);
extern bool ensure_writable_network(brain_t brain);

//=============================================================================
// Static Helper Functions
//=============================================================================

/**
 * @brief Allocate decision structure
 *
 * COMPLEXITY: O(1)
 */
static brain_decision_t* allocate_decision(uint32_t output_size)
{
    brain_decision_t* decision = nimcp_calloc(1, sizeof(brain_decision_t));
    if (!decision) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "decision is NULL");

        return NULL;

    }

    decision->output_size = output_size;

    // W6-13 FIX: Guard against integer overflow in allocation size computation.
    // output_size * sizeof(float) can wrap around for very large output_size values.
    if (output_size > SIZE_MAX / sizeof(float)) {
        nimcp_free(decision);
        set_error("Output size too large for allocation (%u outputs)", output_size);
        return NULL;
    }

    decision->output_vector = nimcp_malloc(output_size * sizeof(float));

    if (!decision->output_vector) {
        nimcp_free(decision);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "allocate_decision: decision->output_vector is NULL");
        return NULL;
    }

    return decision;
}

/**
 * @brief Deep copy a decision structure
 *
 * WHAT: Creates an independent copy of a decision
 * WHY: Cached decisions must not be freed by caller - return copies instead
 * HOW: Allocates new decision and copies all fields including dynamically allocated arrays
 *
 * COMPLEXITY: O(n) where n = output_size + num_active_neurons
 *
 * @param source Decision to copy
 * @return New decision copy, or NULL on allocation failure
 */
static brain_decision_t* copy_decision(const brain_decision_t* source)
{
    if (!source) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "source is NULL");

        return NULL;

    }

    // Allocate new decision structure
    brain_decision_t* copy = nimcp_calloc(1, sizeof(brain_decision_t));
    if (!copy) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "copy is NULL");

        return NULL;

    }

    // Copy scalar fields
    memcpy(copy, source, sizeof(brain_decision_t));

    // NULL out pointer fields to prevent accidental sharing
    copy->output_vector = NULL;
    copy->active_neuron_ids = NULL;

    // Deep copy output_vector
    if (source->output_vector && source->output_size > 0) {
        copy->output_vector = nimcp_malloc(source->output_size * sizeof(float));
        if (!copy->output_vector) {
            nimcp_free(copy);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "copy_decision: copy->output_vector is NULL");
            return NULL;
        }
        memcpy(copy->output_vector, source->output_vector, source->output_size * sizeof(float));
    }

    // Deep copy active_neuron_ids
    if (source->active_neuron_ids && source->num_active_neurons > 0) {
        copy->active_neuron_ids = nimcp_malloc(source->num_active_neurons * sizeof(uint32_t));
        if (!copy->active_neuron_ids) {
            if (copy->output_vector)
                nimcp_free(copy->output_vector);
            nimcp_free(copy);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "copy_decision: validation failed");
            return NULL;
        }
        memcpy(copy->active_neuron_ids, source->active_neuron_ids,
               source->num_active_neurons * sizeof(uint32_t));
    }

    return copy;
}

/**
 * @brief Perform forward pass through network
 *
 * COMPLEXITY: O(s*n) where s = sparsity
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @param decision Decision to populate
 * @return Number of active neurons
 */
static uint32_t perform_forward_pass(brain_t brain, const float* features, uint32_t num_features,
                                     brain_decision_t* decision)
{
    uint64_t start_time = nimcp_time_monotonic_us();

    uint32_t active_neurons;

    // Phase 3: Use read-only inference for COW clones to avoid triggering copy
    if (brain->can_use_readonly) {
        // COW clone using shared network - read-only inference
        active_neurons = adaptive_network_forward_readonly(
            brain->network, features, num_features, decision->output_vector, decision->output_size, 0);
    } else {
        // Original brain or post-COW clone - normal inference with statistics
        active_neurons = adaptive_network_forward(
            brain->network, features, num_features, decision->output_vector, decision->output_size, 0);
    }

    /* Ensemble inference: blend outputs from SNN, LNN, CNN when enabled */
    if (brain->config.enable_ensemble_inference && decision->output_size > 0) {
        float w_adaptive = brain->config.ensemble_weights[0];
        float w_snn = brain->config.ensemble_weights[1];
        float w_lnn = brain->config.ensemble_weights[2];
        float w_cnn = brain->config.ensemble_weights[3];

        /* Use defaults if weights are all zero (not configured) */
        if (w_adaptive == 0.0f && w_snn == 0.0f && w_lnn == 0.0f && w_cnn == 0.0f) {
            w_adaptive = 0.6f; w_snn = 0.2f; w_lnn = 0.1f; w_cnn = 0.1f;
        }

        /* Scale adaptive output by its weight */
        for (uint32_t j = 0; j < decision->output_size; j++) {
            decision->output_vector[j] *= w_adaptive;
        }

        /* Blend SNN output */
        if (brain->snn_network && w_snn > 0.0f) {
            uint32_t snn_out = brain->snn_network->config.n_outputs;
            float* snn_buf = nimcp_calloc(snn_out, sizeof(float));
            if (snn_buf) {
                snn_network_forward((snn_network_t*)brain->snn_network,
                                    features, num_features, snn_buf, snn_out, 1.0f);
                uint32_t blend_dim = (snn_out < decision->output_size) ?
                                      snn_out : decision->output_size;
                for (uint32_t j = 0; j < blend_dim; j++) {
                    decision->output_vector[j] += w_snn * snn_buf[j];
                }
                nimcp_free(snn_buf);
            }
        }

        /* Blend LNN output */
        if (brain->lnn_network && w_lnn > 0.0f) {
            lnn_network_t* lnn = (lnn_network_t*)brain->lnn_network;
            uint32_t lnn_out = lnn->n_outputs;
            uint32_t lnn_in = lnn->n_inputs;
            uint32_t in_dims[1] = { lnn_in };
            uint32_t out_dims[1] = { lnn_out };
            nimcp_tensor_t* in_t = nimcp_tensor_create(in_dims, 1, NIMCP_DTYPE_F32);
            nimcp_tensor_t* out_t = nimcp_tensor_create(out_dims, 1, NIMCP_DTYPE_F32);
            if (in_t && out_t) {
                float* in_data = (float*)nimcp_tensor_data(in_t);
                uint32_t copy_in = (num_features < lnn_in) ? num_features : lnn_in;
                memcpy(in_data, features, copy_in * sizeof(float));
                if (lnn_forward_step(lnn, in_t, out_t, 1.0f) == 0) {
                    const float* out_data = (const float*)nimcp_tensor_data_const(out_t);
                    uint32_t blend_dim = (lnn_out < decision->output_size) ?
                                          lnn_out : decision->output_size;
                    for (uint32_t j = 0; j < blend_dim; j++) {
                        decision->output_vector[j] += w_lnn * out_data[j];
                    }
                }
            }
            nimcp_tensor_destroy(in_t);
            nimcp_tensor_destroy(out_t);
        }

        /* Blend CNN output */
        if (brain->cnn_trainer && w_cnn > 0.0f) {
            uint32_t cnn_dims[1] = { num_features };
            nimcp_tensor_t* cnn_in = nimcp_tensor_create(cnn_dims, 1, NIMCP_DTYPE_F32);
            if (cnn_in) {
                memcpy(nimcp_tensor_data(cnn_in), features, num_features * sizeof(float));
                cnn_forward_result_t cnn_result = {0};
                if (cnn_trainer_forward(brain->cnn_trainer, cnn_in, &cnn_result) == NIMCP_SUCCESS
                    && cnn_result.output) {
                    const float* cnn_data = (const float*)nimcp_tensor_data_const(cnn_result.output);
                    size_t cnn_numel = nimcp_tensor_numel(cnn_result.output);
                    uint32_t blend_dim = ((uint32_t)cnn_numel < decision->output_size) ?
                                          (uint32_t)cnn_numel : decision->output_size;
                    for (uint32_t j = 0; j < blend_dim; j++) {
                        decision->output_vector[j] += w_cnn * cnn_data[j];
                    }
                }
                nimcp_tensor_destroy(cnn_in);
            }
        }
    }

    decision->inference_time_us = nimcp_time_elapsed_us(start_time);

    return active_neurons;
}

/**
 * @brief Find maximum output and determine label
 *
 * COMPLEXITY: O(n) where n = num_outputs
 */
// P3-51: Named constant for confidence normalization factor
#define CONFIDENCE_NORMALIZATION_FACTOR 10.0F

static void determine_output_label(brain_t brain, brain_decision_t* decision)
{
    // P2-50 FIX: Guard against output_size==0 to prevent OOB access on output_vector[0]
    if (decision->output_size == 0) return;

    // I-H3 FIX: NaN-safe argmax. If output contains NaN values, standard comparison
    // (NaN > max_value) is always false, silently picking index 0. Use isfinite()
    // to skip NaN/Inf values and detect all-NaN output.
    uint32_t max_idx = 0;
    float max_value = -FLT_MAX;
    bool has_valid = false;

    for (uint32_t i = 0; i < decision->output_size; i++) {
        if (isfinite(decision->output_vector[i]) && decision->output_vector[i] > max_value) {
            max_value = decision->output_vector[i];
            max_idx = i;
            has_valid = true;
        }
    }

    // If all outputs are NaN/Inf, set label to UNKNOWN with zero confidence
    if (!has_valid) {
        strncpy(decision->label, "UNKNOWN", sizeof(decision->label) - 1);
        decision->label[sizeof(decision->label) - 1] = '\0';
        decision->confidence = 0.0f;
        return;
    }

    // W6-1 FIX: 3-way NULL guard matching predict_fast pattern.
    // Check output_labels array, bounds, AND specific entry for NULL.
    if (brain->output_labels && max_idx < brain->num_output_labels
        && brain->output_labels[max_idx]) {
        strncpy(decision->label, brain->output_labels[max_idx], sizeof(decision->label) - 1);
    } else {
        decision->label[0] = '\0';
    }

    // W6-3 FIX: Use ratio normalization (max/sum_abs) matching predict_fast,
    // replacing the fixed-divisor formula (max/10.0) that gave inconsistent
    // confidence values compared to predict_fast for the same inputs.
    // W6-2 FIX: Lower-bound clamp to ensure confidence is in [0, 1].
    float sum_abs = 0.0f;
    for (uint32_t i = 0; i < decision->output_size; i++) {
        if (isfinite(decision->output_vector[i]))
            sum_abs += fabsf(decision->output_vector[i]);
    }
    float raw_conf = (sum_abs > 0.0f) ? (max_value / sum_abs) : 0.0f;
    decision->confidence = fmaxf(0.0f, fminf(raw_conf, 1.0f));
}

/**
 * @brief Populate interpretability information
 *
 * COMPLEXITY: O(n)
 */
static void populate_interpretability(brain_t brain, const float* features, uint32_t num_features,
                                      uint32_t active_neurons, brain_decision_t* decision)
{
    decision->num_active_neurons = active_neurons;
    decision->sparsity = adaptive_network_get_sparsity(brain->network);

    if (brain->config.enable_explanations) {
        adaptive_network_explain(brain->network, features, num_features, decision->explanation,
                                 sizeof(decision->explanation));
    }

    // P2-54 FIX: Guard against nimcp_malloc(0) when active_neurons==0
    // malloc(0) is implementation-defined and may return NULL or a non-NULL
    // pointer that must not be dereferenced.
    if (active_neurons == 0) return;

    // Populate active neuron IDs
    decision->active_neuron_ids = nimcp_malloc(active_neurons * sizeof(uint32_t));
    if (!decision->active_neuron_ids) {
        // W6-8 FIX: Reset num_active_neurons to 0 on alloc failure.
        // Without this, num_active_neurons is non-zero but pointer is NULL,
        // causing callers that iterate active_neuron_ids to dereference NULL.
        decision->num_active_neurons = 0;
        set_error("Failed to allocate active neuron IDs array (%u neurons)", active_neurons);
        return;
    }
    // W6-11 NOTE: These are sequential indices [0..active_neurons-1], not the
    // real neuron IDs from the network. Mapping real IDs would require
    // adaptive_network to expose its active set, which is not yet implemented.
    for (uint32_t i = 0; i < active_neurons; i++) {
        decision->active_neuron_ids[i] = i;
    }
}

/**
 * @brief Update brain statistics after inference
 *
 * THREAD SAFETY: Uses atomic operations for counter updates to ensure
 * thread-safe statistics tracking when multiple threads perform concurrent
 * inference on the same brain instance.
 *
 * COMPLEXITY: O(1)
 */
static void update_inference_stats(brain_t brain, brain_decision_t* decision)
{
    // Use atomic increment for thread-safe stats update
    uint64_t new_count = __atomic_fetch_add(&brain->stats.total_inferences, 1, __ATOMIC_RELAXED) + 1;

    // Update average inference time using the new count
    // W6-10 NOTE: This is a non-atomic read-modify-write on avg_inference_time_us
    // (torn read/write possible under concurrent inference). This is benign under
    // typical usage because: (1) the value is only used for monitoring/logging,
    // (2) worst case is a transiently inaccurate average, and (3) using atomics
    // for double would require _Atomic double + -latomic which adds overhead.
    brain->stats.avg_inference_time_us =
        (brain->stats.avg_inference_time_us * (new_count - 1) +
         decision->inference_time_us) /
        new_count;
    brain->stats.avg_sparsity = decision->sparsity;
}

//=============================================================================
// Mirror Neuron Integration Helpers (Phase 10.11)
//=============================================================================

/**
 * @brief Convert brain decision to mirror neuron action
 *
 * WHAT: Transform brain decision into action_t for mirror neuron system
 * WHY:  Enable mirror neurons to learn from brain's own decisions
 * HOW:  Extract decision features, confidence, and output as action representation
 *
 * COMPLEXITY: O(n) where n = num_outputs (feature copying)
 *
 * @param decision Brain decision
 * @param action_id Unique action identifier
 * @param action_name Human-readable action name
 * @return action_t struct for mirror neuron system
 */
static action_t brain_decision_to_action(const brain_decision_t* decision,
                                         uint32_t action_id,
                                         const char* action_name)
{
    action_t action;
    memset(&action, 0, sizeof(action_t));

    if (!decision || !action_name) {
        return action;
    }

    action.action_id = action_id;
    strncpy(action.action_name, action_name, sizeof(action.action_name) - 1);
    action.agent_id = 0;  // 0 = self
    action.timestamp = nimcp_time_get_ms();
    action.confidence = decision->confidence;

    // Use output activations as action features (up to 32)
    action.num_features = (decision->output_size < 32) ? decision->output_size : 32;
    for (uint32_t i = 0; i < action.num_features; i++) {
        action.features[i] = decision->output_vector[i];
    }

    return action;
}

/**
 * @brief Convert input features to observed action
 *
 * WHAT: Transform input features into action_t for observation pathway
 * WHY:  Enable mirror neurons to learn from observed patterns
 * HOW:  Treat input as observed action with features
 *
 * COMPLEXITY: O(n) where n = num_features (copying)
 *
 * @param features Input features
 * @param num_features Number of features
 * @param agent_id ID of agent performing action (0 = self, >0 = other)
 * @return action_t struct for mirror neuron system
 */
static action_t features_to_action(const float* features, uint32_t num_features,
                                   uint32_t agent_id)
{
    action_t action;
    memset(&action, 0, sizeof(action_t));

    if (!features) {
        return action;
    }

    action.action_id = 0;  // Will be assigned by mirror neuron system
    snprintf(action.action_name, sizeof(action.action_name), "observed_%u", agent_id);
    action.agent_id = agent_id;
    action.timestamp = nimcp_time_get_ms();
    action.confidence = 1.0F;

    // Copy features (up to 32)
    action.num_features = (num_features < 32) ? num_features : 32;
    for (uint32_t i = 0; i < action.num_features; i++) {
        action.features[i] = features[i];
    }

    return action;
}

//=============================================================================
// Public API Functions
//=============================================================================
// NOTE: The full brain_decide() implementation is too large (1374 lines)
// and deeply integrated with brain internals. For this extraction, we are
// documenting the function signature and providing a reference implementation
// that must remain in nimcp_brain.c due to its extensive use of brain internals.
//=============================================================================

/**
 * @brief Free decision result with CoW support
 *
 * WHY: Proper memory management for decision results
 * Handles all allocated sub-structures
 *
 * Phase 1.5 CoW: Decisions may share data via reference counting.
 * Only free shared data when the last reference is released.
 *
 * COMPLEXITY: O(1)
 * THREAD SAFETY: Uses atomic operations for refcount updates
 *
 * @param decision Decision to free
 */

int brain_reset_inference_state(brain_t brain)
{
    if (!brain) return -1;

    /* Reset main adaptive network neuron states */
    if (brain->network) {
        neural_network_t base_net = adaptive_network_get_base_network(brain->network);
        if (base_net) {
            neural_network_reset(base_net);
        }
    }

    /* Reset LNN hidden states if present */
    if (brain->lnn_network) {
        lnn_reset_state((lnn_network_t*)brain->lnn_network);
    }

    /* Reset SNN spike buffers and membrane potentials */
    if (brain->snn_network) {
        snn_network_reset((snn_network_t*)brain->snn_network);
    }

    return 0;
}

void brain_set_active_modalities(brain_t brain, uint32_t modality_flags)
{
    if (!brain) return;
    brain->active_modalities = modality_flags;
}

uint32_t brain_get_active_modalities(brain_t brain)
{
    if (!brain) return BRAIN_MODALITY_TEXT;
    return brain->active_modalities;
}

void brain_clear_sensory(brain_t brain)
{
    if (!brain) return;
    if (brain->staged_sensory.visual_frame) {
        nimcp_free(brain->staged_sensory.visual_frame);
        brain->staged_sensory.visual_frame = NULL;
    }
    if (brain->staged_sensory.audio_data) {
        nimcp_free(brain->staged_sensory.audio_data);
        brain->staged_sensory.audio_data = NULL;
    }
    if (brain->staged_sensory.speech_data) {
        nimcp_free(brain->staged_sensory.speech_data);
        brain->staged_sensory.speech_data = NULL;
    }
    if (brain->staged_sensory.somato_data) {
        nimcp_free(brain->staged_sensory.somato_data);
        brain->staged_sensory.somato_data = NULL;
    }
}

int brain_submit_sensory(brain_t brain, uint32_t modality,
                          const void* data, uint32_t size,
                          uint32_t width, uint32_t height,
                          uint32_t channels)
{
    if (!brain || !data || size == 0) return -1;

    switch (modality) {
    case BRAIN_MODALITY_VISUAL: {
        if (brain->staged_sensory.visual_frame)
            nimcp_free(brain->staged_sensory.visual_frame);
        uint32_t bytes = size * sizeof(uint8_t);
        brain->staged_sensory.visual_frame = nimcp_malloc(bytes);
        if (!brain->staged_sensory.visual_frame) return -1;
        memcpy(brain->staged_sensory.visual_frame, data, bytes);
        brain->staged_sensory.visual_width = width;
        brain->staged_sensory.visual_height = height;
        brain->staged_sensory.visual_channels = channels ? channels : 3;
        break;
    }
    case BRAIN_MODALITY_AUDIO: {
        if (brain->staged_sensory.audio_data)
            nimcp_free(brain->staged_sensory.audio_data);
        uint32_t bytes = size * sizeof(float);
        brain->staged_sensory.audio_data = nimcp_malloc(bytes);
        if (!brain->staged_sensory.audio_data) return -1;
        memcpy(brain->staged_sensory.audio_data, data, bytes);
        brain->staged_sensory.audio_size = size;
        brain->staged_sensory.audio_channels = channels ? (uint8_t)channels : 1;
        break;
    }
    case BRAIN_MODALITY_SPEECH: {
        if (brain->staged_sensory.speech_data)
            nimcp_free(brain->staged_sensory.speech_data);
        uint32_t bytes = size * sizeof(float);
        brain->staged_sensory.speech_data = nimcp_malloc(bytes);
        if (!brain->staged_sensory.speech_data) return -1;
        memcpy(brain->staged_sensory.speech_data, data, bytes);
        brain->staged_sensory.speech_size = size;
        break;
    }
    case BRAIN_MODALITY_SOMATOSENSORY: {
        if (brain->staged_sensory.somato_data)
            nimcp_free(brain->staged_sensory.somato_data);
        uint32_t bytes = size * sizeof(float);
        brain->staged_sensory.somato_data = nimcp_malloc(bytes);
        if (!brain->staged_sensory.somato_data) return -1;
        memcpy(brain->staged_sensory.somato_data, data, bytes);
        brain->staged_sensory.somato_segments = size;
        break;
    }
    default:
        return -1;
    }

    /* Auto-enable the modality */
    brain->active_modalities |= modality;
    return 0;
}

void brain_free_decision(brain_decision_t* decision)
{
    if (!decision)
        return;

    /**
     * WHAT: Free decision structure and its allocated fields
     * WHY: Prevent memory leaks
     * HOW: Use nimcp_free() for memory allocated with nimcp_malloc()
     *
     * Phase 1.5 CoW: Check reference count before freeing shared data
     */

    if (decision->_cow_refcount) {
        // This decision shares data with others via CoW
        // Atomically decrement refcount and check if we're the last user
        uint32_t remaining = __atomic_sub_fetch(decision->_cow_refcount, 1, __ATOMIC_SEQ_CST);

        if (remaining == 0) {
            // Last reference - free the shared data
            if (decision->output_vector) {
                nimcp_free(decision->output_vector);
            }
            if (decision->active_neuron_ids) {
                nimcp_free(decision->active_neuron_ids);
            }
            // Free the refcount itself
            nimcp_free(decision->_cow_refcount);
        }
        // else: Other references exist - don't free the shared data
    } else {
        // This decision owns its data exclusively (not shared)
        if (decision->output_vector) {
            nimcp_free(decision->output_vector);
        }
        if (decision->active_neuron_ids) {
            nimcp_free(decision->active_neuron_ids);
        }
    }

    // Free cognitive transcript if present — must use transcript_free()
    // to release nested entry allocations, not nimcp_free() which only
    // frees the top-level struct.
    if (decision->transcript) {
        transcript_free(decision->transcript);
        decision->transcript = NULL;
    }

    // Always free the decision structure itself (each copy has its own)
    nimcp_free(decision);
}

/**
 * @brief Batch inference
 *
 * WHY: More efficient than individual calls for large batches
 * Enables parallel processing opportunities
 *
 * COMPLEXITY: O(m*s*n) where m = num_inputs
 *
 * @param brain Brain handle
 * @param inputs Array of input vectors
 * @param num_inputs Number of inputs
 * @param features_per_input Features per input
 * @param decisions Output decisions array (allocated by caller)
 * @return true on success
 */
/**
 * @brief Task context for parallel CPU batch inference
 */
typedef struct {
    brain_t brain;
    const float* features;
    uint32_t num_features;
    brain_decision_t* decision;
    bool success;
} batch_decide_task_t;

/**
 * @brief Worker function for parallel CPU batch inference
 *
 * W7-4 (C-INF-4) DATA RACE WARNING: Each worker calls brain_decide() on the
 * SAME brain instance. brain_decide() -> perform_forward_pass() ->
 * adaptive_network_forward() mutates shared neuron state (spike encoding,
 * weight statistics). Concurrent calls without external synchronization
 * produce undefined behavior (corrupted neuron state, wrong predictions).
 *
 * SAFE USAGE PATTERNS:
 *   1. Frozen brain (brain_freeze()): Forward pass is read-only, no mutation.
 *   2. ThreadSafeBrain (Python): Wraps brain_decide in a per-brain RLock.
 *   3. Read-only COW clones: can_use_readonly path uses forward_readonly().
 *
 * The parallel path is dispatched only when inference_pool exists, which is
 * created during brain_create for GPU-enabled or explicitly pooled brains.
 * Users of the C API must ensure one of the above safety patterns.
 */
static void batch_decide_worker(void* arg)
{
    batch_decide_task_t* task = (batch_decide_task_t*)arg;
    brain_decision_t* result = brain_decide(task->brain, task->features, task->num_features);
    if (result) {
        memcpy(task->decision, result, sizeof(brain_decision_t));
        result->output_vector = NULL;
        result->active_neuron_ids = NULL;
        brain_free_decision(result);
        task->success = true;
    } else {
        task->success = false;
    }
}

bool brain_decide_batch(brain_t brain, const float** inputs, uint32_t num_inputs,
                        uint32_t features_per_input, brain_decision_t* decisions)
{
    // Guard: Validate parameters
    if (!brain || !inputs || !decisions || num_inputs == 0 || features_per_input == 0) {
        set_error("Invalid parameters to brain_decide_batch");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_decide_batch: required parameter is NULL or zero (brain, inputs, decisions, num_inputs, features_per_input)");
        return false;
    }

    // Zero-initialize all decisions so partial parallel failures don't leave
    // garbage pointers that the serial fallback would try to free.
    memset(decisions, 0, num_inputs * sizeof(brain_decision_t));

    // CPU parallel path: use thread pool for batch >= 4 when pool exists
    // W7-4 (C-INF-4): Only use parallel path if brain is safe for concurrent access.
    // Unfrozen, non-COW brains have mutable neuron state that races under concurrency.
    if (num_inputs >= 4 && brain->inference_pool
        && (brain->frozen || brain->can_use_readonly)) {
        batch_decide_task_t* tasks = nimcp_calloc(num_inputs, sizeof(batch_decide_task_t));
        if (!tasks) {
            // Fall through to serial path
            goto serial_path;
        }

        for (uint32_t i = 0; i < num_inputs; i++) {
            tasks[i].brain = brain;
            tasks[i].features = inputs[i];
            tasks[i].num_features = features_per_input;
            tasks[i].decision = &decisions[i];
            tasks[i].success = false;
            nimcp_pool_submit(brain->inference_pool, batch_decide_worker, &tasks[i]);
        }
        nimcp_pool_wait(brain->inference_pool);

        bool all_ok = true;
        for (uint32_t i = 0; i < num_inputs; i++) {
            if (!tasks[i].success) { all_ok = false; break; }
        }
        nimcp_free(tasks);

        if (all_ok) {
            brain_clear_error();
            return true;
        }
        // If any failed, fall through to serial
    }

serial_path:
    ;  // empty statement after label (required by C before declarations)
    // I-M5 FIX: Handle partial failures in batch prediction. Previously, if any
    // single inference failed, the function returned false immediately, leaving
    // remaining decisions uninitialized (garbage data). Now we continue processing
    // all items, zero-initialize failed decisions, and report partial success.
    {
    uint32_t batch_failures = 0;
    for (uint32_t i = 0; i < num_inputs; i++) {
        // W6-5 FIX: Free existing decision fields before overwriting.
        // When the parallel batch path partially succeeds then falls through
        // to serial, successful parallel decisions have heap-allocated
        // output_vector and active_neuron_ids that would leak if overwritten.
        if (decisions[i].output_vector) {
            nimcp_free(decisions[i].output_vector);
            decisions[i].output_vector = NULL;
        }
        if (decisions[i].active_neuron_ids) {
            nimcp_free(decisions[i].active_neuron_ids);
            decisions[i].active_neuron_ids = NULL;
        }

        brain_decision_t* decision = brain_decide(brain, inputs[i], features_per_input);

        if (!decision) {
            // I-M5: Zero-initialize failed decisions instead of aborting the batch.
            // Caller can check decision->confidence == 0 to detect failures.
            memset(&decisions[i], 0, sizeof(brain_decision_t));
            strncpy(decisions[i].label, "FAILED", sizeof(decisions[i].label) - 1);
            batch_failures++;
            continue;
        }

        memcpy(&decisions[i], decision, sizeof(brain_decision_t));
        decision->output_vector = NULL;
        decision->active_neuron_ids = NULL;
        brain_free_decision(decision);
    }

    if (batch_failures > 0) {
        // I-M2: Propagate error information about partial failures
        set_error("brain_decide_batch: %u of %u inferences failed", batch_failures, num_inputs);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "brain_decide_batch: %u of %u decisions failed", batch_failures, num_inputs);
        return false;
    }
    }

    brain_clear_error();
    return true;
}

//=============================================================================
// Frozen Inference Network
//=============================================================================

bool brain_freeze(brain_t brain)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_freeze: brain is NULL");
        return false;
    }

    if (brain->frozen) {
        return true;  // Already frozen
    }

    // Freeze the adaptive network (zeros learning, frees traces)
    if (brain->network) {
        // Ensure GPU weight cache is up-to-date before freezing
        if (adaptive_network_is_gpu_enabled(brain->network)) {
            neural_network_t base = adaptive_network_get_base_network(brain->network);
            const adaptive_network_config_t* cfg = adaptive_network_get_config(brain->network);
            (void)base; (void)cfg;  // GPU cache already managed by adaptive network
        }
        adaptive_network_freeze(brain->network);
    }

    brain->frozen = true;
    LOG_INFO("Brain frozen for inference-only mode");
    return true;
}

bool brain_is_frozen(brain_t brain)
{
    if (!brain) return false;
    return brain->frozen;
}

/**
 * @brief Observe action performed by another agent (Phase 10.11)
 *
 * WHAT: Record observed action in mirror neuron system for observational learning
 * WHY:  Enable learning from watching others (imitation, social cognition)
 * HOW:  Convert input features to observed action and send to mirror neurons
 *
 * This is the OBSERVATION PATHWAY for mirror neurons. When the brain observes
 * another agent performing an action, this function records it for learning.
 *
 * USE CASES:
 * - Robot watching human demonstration
 * - Agent observing another agent's behavior
 * - Learning from video/sensor data of actions
 * - Social learning and imitation
 *
 * COMPLEXITY: O(n) where n = num_features
 * THREAD-SAFE: No (requires external synchronization)
 *
 * @param brain Brain handle
 * @param features Observed action features (sensor data, visual features, etc.)
 * @param num_features Number of features
 * @param agent_id ID of agent being observed (must be > 0, as 0 = self)
 * @return true on success, false on error
 */
bool brain_observe_action(brain_t brain, const float* features, uint32_t num_features,
                          uint32_t agent_id)
{
    // Guard: Validate parameters
    if (!brain || !features) {
        set_error("Invalid parameters to brain_observe_action");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_observe_action: required parameter is NULL (brain, features)");
        return false;
    }

    if (agent_id == 0) {
        set_error("agent_id must be > 0 (0 is reserved for self)");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_observe_action: agent_id is zero");
        return false;
    }

    // Guard: Check if mirror neurons are enabled
    if (!brain->mirror_neurons || !brain->config.enable_mirror_neurons) {
        // Not an error, just not enabled
        return true;
    }

    // Convert features to action
    action_t action = features_to_action(features, num_features, agent_id);

    // Record as observed action
    bool success = mirror_neurons_observe_action(brain->mirror_neurons, &action);
    if (!success) {
        set_error("Failed to record observed action in mirror neurons");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_observe_action: success is NULL");
        return false;
    }

    // Store in working memory if enabled (for offline replay)
    if (brain->working_memory && brain->config.enable_working_memory) {
        // Store action features in working memory
        // Note: This enables later replay and sequence learning
        working_memory_add(brain->working_memory, features, num_features, 0.8F);
    }

    // Trigger theory of mind inference if enabled
    if (brain->theory_of_mind && brain->config.enable_theory_of_mind) {
        // Use mirror neuron activations to infer agent intentions
        // Note: This requires getting current mirror neuron state
        // WHAT: Use mirror neuron activations to infer agent intentions
        // WHY:  Mirror neurons encode observed actions, ToM infers mental states
        // HOW:  Extract mirror activations and pass to ToM as action vector

        // BIOLOGICAL RATIONALE: Mirror neurons fire during action observation
        // (Rizzolatti & Craighero, 2004). ToM uses this to understand "why"
        // agents perform actions, enabling empathy and intention inference.

        float mirror_activations[100];  // Max 100 actions
        uint32_t num_activations = 0;

        if (mirror_neurons_get_all_activations(brain->mirror_neurons,
                                              mirror_activations,
                                              100,
                                              &num_activations)) {
            // Package as ToM observation
            tom_observation_t observation = {
                .action_vector = mirror_activations,
                .action_dim = num_activations,
                .verbal_context = NULL,  // No verbal context from features
                .observed_emotion = TOM_EMOTION_UNKNOWN,  // Infer from context
                .situational_context = features,  // Raw features as context
                .context_dim = num_features
            };

            // Let ToM infer mental state from mirror neuron pattern
            tom_observe(brain->theory_of_mind, &observation);
        }
    }

    brain_clear_error();
    return true;
}


//=============================================================================
// Async Inference Implementation
//=============================================================================

/**
 * @brief Context for async inference thread
 */
typedef struct {
    brain_t brain;
    float* features;
    uint32_t num_features;
    nimcp_promise_t promise;
} async_infer_context_t;

/**
 * @brief Background thread function for async inference
 *
 * WHAT: Performs inference in background thread and completes promise
 * WHY:  Enable non-blocking decision making
 * HOW:  Call brain_decide, set result or error on promise
 *
 * THREAD SAFETY: Each thread has its own context, brain must support concurrent reads
 *
 * @param arg async_infer_context_t pointer
 * @return NULL (unused)
 */
static void* async_infer_thread(void* arg)
{
    async_infer_context_t* ctx = (async_infer_context_t*)arg;

    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");


        return NULL;
    }

    LOG_MODULE_DEBUG("brain_inference", "Async inference started: %u features", ctx->num_features);

    // Perform synchronous inference
    brain_decision_t* decision = brain_decide(
        ctx->brain,
        ctx->features,
        ctx->num_features
    );

    // Complete promise with result or error
    if (!decision) {
        // Inference failed
        nimcp_error_t error = NIMCP_ERROR_OPERATION_FAILED;
        nimcp_promise_fail(ctx->promise, error);
        LOG_MODULE_ERROR("brain_inference", "Async inference failed");
    } else {
        // Inference succeeded
        // Note: promise takes ownership of decision pointer
        nimcp_promise_complete(ctx->promise, &decision);
        LOG_MODULE_DEBUG("brain_inference", "Async inference completed: label='%s', confidence=%.2f",
                       decision->label, decision->confidence);
    }

    // Cleanup
    nimcp_free(ctx->features);
    // W6-12 NOTE: Promise is destroyed here before the future consumer may have
    // read the result. This is safe because nimcp_promise_complete() copies the
    // result into the shared future state before returning, so the promise
    // object itself is no longer needed after completion.
    nimcp_promise_destroy(ctx->promise);
    nimcp_free(ctx);

    // W6-4 FIX: Removed unconditional NIMCP_THROW_TO_IMMUNE that fired on every
    // async inference completion (including success), generating false immune events.
    // The error path is already handled by nimcp_promise_fail() above.
    return NULL;
}

/**
 * @brief Asynchronous inference/decision making
 *
 * WHAT: Non-blocking version of brain_decide
 * WHY:  Enable concurrent inference without blocking caller
 * HOW:  Copy inputs, create promise/future, spawn thread
 *
 * IMPLEMENTATION NOTES:
 * - Features are deep-copied to ensure thread safety
 * - Promise is created with sizeof(brain_decision_t*) for result pointer
 * - Thread is detached for auto-cleanup
 * - Context is freed by worker thread
 *
 * ERROR HANDLING:
 * - Returns NULL if allocation fails or thread creation fails
 * - Inference errors propagated through future
 *
 * MEMORY MANAGEMENT:
 * - Context allocated on heap, freed by worker thread
 * - Features copied to heap, freed by worker thread
 * - Promise destroyed by worker thread after completion
 * - Future must be destroyed by caller
 * - Decision result must be freed by caller with brain_free_decision()
 *
 * @param brain Brain handle
 * @param features Input features (will be copied)
 * @param num_features Feature count
 * @return Future handle or NULL on error
 */
nimcp_future_t nimcp_brain_infer_async(brain_t brain, const float* features,
                                        uint32_t num_features)
{
    // Validate parameters
    if (!brain || !features) {
        LOG_MODULE_ERROR("brain_inference", "Invalid parameters to nimcp_brain_infer_async");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_infer_async: required parameter is NULL (brain, features)");
        return NULL;
    }

    if (num_features == 0) {
        LOG_MODULE_ERROR("brain_inference", "Invalid num_features=0");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "nimcp_brain_infer_async: num_features is zero");
        return NULL;
    }

    // Allocate context for worker thread
    async_infer_context_t* ctx = nimcp_malloc(sizeof(async_infer_context_t));
    if (!ctx) {
        LOG_MODULE_ERROR("brain_inference", "Failed to allocate async inference context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_infer_async: ctx is NULL");
        return NULL;
    }

    // Copy features to heap (worker thread will free)
    ctx->features = nimcp_malloc(num_features * sizeof(float));
    if (!ctx->features) {
        LOG_MODULE_ERROR("brain_inference", "Failed to allocate features array (%u floats)", num_features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_infer_async: ctx->features is NULL");
        return NULL;
    }
    memcpy(ctx->features, features, num_features * sizeof(float));

    // Set context fields
    ctx->brain = brain;
    ctx->num_features = num_features;

    // Create promise for result (decision is a pointer)
    ctx->promise = nimcp_promise_create(sizeof(brain_decision_t*));
    if (!ctx->promise) {
        LOG_MODULE_ERROR("brain_inference", "Failed to create promise for async inference");
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_infer_async: ctx->promise is NULL");
        return NULL;
    }

    // Get future before starting thread
    nimcp_future_t future = nimcp_promise_get_future(ctx->promise);
    if (!future) {
        LOG_MODULE_ERROR("brain_inference", "Failed to get future from promise");
        nimcp_promise_destroy(ctx->promise);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_infer_async: future is NULL");
        return NULL;
    }

    // Create worker thread
    nimcp_thread_t thread;
    thread_attr_t attr = {
        .stack_size = NIMCP_THREAD_DEFAULT_STACK_SIZE,
        .priority = 0,
        .detached = true  // Auto-cleanup on exit
    };

    nimcp_result_t result = nimcp_thread_create(&thread, async_infer_thread, ctx, &attr);
    if (result != NIMCP_SUCCESS) {
        LOG_MODULE_ERROR("brain_inference", "Failed to create async inference thread: %d", result);
        nimcp_future_destroy(future);
        nimcp_promise_destroy(ctx->promise);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_infer_async: validation failed");
        return NULL;
    }

    LOG_MODULE_INFO("brain_inference", "Async inference thread created successfully");

    // Return future to caller (caller must destroy)
    return future;
}
