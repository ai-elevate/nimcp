//=============================================================================
// nimcp_brain_learning.c - Brain Learning Module Implementation
//=============================================================================
/**
 * @file nimcp_brain_learning.c
 * @brief Learning, training, and reward-based learning implementation
 *
 * WHAT: High-level supervised and reinforcement learning for brain networks
 * WHY:  Separates learning logic from core brain management
 * HOW:  Integrates adaptive network learning with biological enhancements
 *
 * ARCHITECTURE:
 * - Forward pass → loss computation → backpropagation → weight update
 * - Epistemic filtering: Skepticism filter for training data quality
 * - Curiosity-driven LR: Boost learning rate for novel patterns
 * - Biological security: Excitotoxicity monitoring and emergency inhibition
 * - Emotional integration: Joy, remorse, shadow emotions
 * - Memory encoding: Engram formation, consolidation pipeline
 * - Quantum optimization: Periodic weight escape from local minima
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Single example: ~0.1-1ms (small), ~10ms (large networks)
 * - Batch learning: ~m × single_example_time
 * - Quantum annealing: ~50-500ms (run every N steps)
 * - Memory overhead: O(1) per learning call
 *
 * @author NIMCP Development Team
 * @version 1.0
 */

#include "core/brain/learning/nimcp_brain_learning.h"
#include "core/brain/nimcp_brain_lazy_init.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "memory/nimcp_memory_store.h"
/* Cognitive module types for training wiring */
#include "cognitive/recursive/nimcp_rcog_types.h"
#include "cognitive/recursive/nimcp_rcog_engine.h"
#include "cognitive/ethics/nimcp_ethics.h"
#include <math.h>
#include <string.h>
#include "utils/math/nimcp_math_helpers.h"
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "core/neuralnet/nimcp_neuron_synapse_access.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "async/nimcp_future.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/containers/nimcp_hash_table.h"
#include "cognitive/memory/core/nimcp_prime_signature.h"
#include "core/brain/regions/mammillary/nimcp_mammillary.h"
#include "portia/nimcp_portia.h"
#include "edge/nimcp_sensor.h"
#include "edge/nimcp_safety_watchdog.h"

/* Loss history circular buffer size — must match brain_internal.h loss_history[10] */
#define LOSS_HISTORY_SIZE 10
#define REWARD_LEARNING_RATE 0.0001f
#define REWARD_ACTIVITY_THRESHOLD 0.01f

#define LOG_MODULE "core_brain_learning"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/bridge/nimcp_bridge_boilerplate.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"
#include "constants/nimcp_buffer_constants.h"
#include "core/brain/bridges/nimcp_hyperledger_bridge.h"
#include "training/nimcp_unified_training.h"
#include "cognitive/omni/bridges/nimcp_omni_wm_thousand_brains_bridge.h"
#include "cognitive/omni/nimcp_omni_world_model.h"
#include "core/cortical_columns/nimcp_thousand_brains_integration.h"

// Multi-network training includes (LNN + CNN + dispatch)
#include "training/nimcp_training_dispatch.h"
#include "lnn/nimcp_lnn.h"
#include "lnn/nimcp_lnn_network.h"
#include "lnn/nimcp_lnn_training.h"
#include "training/nimcp_cnn_training.h"
#include "training/nimcp_snn_backprop.h"
#include "snn/nimcp_snn_config.h"
#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_fno.h"
#include "lnn/nimcp_lnn_hamiltonian.h"
#include "training/nimcp_cortex_cnn.h"
#include "cognitive/vae/nimcp_vae.h"
#include "cognitive/vae/bridges/nimcp_vae_training_bridge.h"
#include "cognitive/attention/nimcp_attention_plasticity_bridge.h"

// Perceptual cortex + cortical column + predictive coding training integration
#include "perception/nimcp_visual_cortex.h"
#include "perception/nimcp_audio_cortex.h"
#include "core/cortical_columns/nimcp_cortical_column.h"
#include "plasticity/predictive/nimcp_predictive_coding.h"
#include "plasticity/structural/nimcp_structural_plasticity.h"
#include "core/neuralnet/nimcp_neuralnet_learning.h"

// Biological plasticity integration (TPB + EDP + coordinator)
#include "middleware/training/nimcp_training_plasticity_bridge.h"
#include "middleware/training/nimcp_event_driven_plasticity.h"
#include "plasticity/nimcp_plasticity_coordinator.h"
#include "plasticity/orchestrator/nimcp_neural_plasticity_coordinator.h"

#include "core/brain_regions/nimcp_brain_regions.h"

// Thousand Brains integration (Hawkins cortical columns)
#include "cognitive/omni/bridges/nimcp_omni_wm_thousand_brains_bridge.h"
#include "core/cortical_columns/nimcp_thousand_brains_integration.h"

// Cognitive subsystem training includes
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "language/nimcp_grounded_language.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
#include "cognitive/nimcp_predictive.h"
#include "cognitive/parietal/nimcp_parietal.h"
#include "cognitive/predictive/nimcp_predictive_hierarchy.h"
#include "cognitive/jepa/nimcp_jepa_predictor.h"
#include "cognitive/jepa/nimcp_jepa_latent.h"

BRIDGE_BOILERPLATE_MESH_ONLY(brain_learning, MESH_ADAPTER_CATEGORY_COGNITIVE)


// Import needed from brain internal structure
// Note: These functions access brain internal state, defined in nimcp_brain.c
extern void set_error(const char* format, ...);

// Forward declarations for subsystem training (avoid pulling in heavy headers)

// Creative training bridge (forward declaration to avoid heavy creative header chain)
struct creative_training_bridge;
typedef struct creative_training_bridge creative_training_bridge_t;
extern int creative_training_submit_feedback(creative_training_bridge_t* bridge,
                                              const void* content,
                                              int modality,
                                              uint8_t rating,
                                              const char* feedback);

// Self-heal engine training
#include "cognitive/immune/nimcp_self_heal.h"

// Intuition system training is declared in nimcp_intuition_integrations.h
// (included via nimcp_brain_internal.h)


/**
 * @brief Train ALL cognitive subsystems from the current learning step
 *
 * WHAT: Dispatches training signals to every initialized cognitive module
 * WHY:  brain_learn_vector() only trains the adaptive network + SNN/CNN/LNN.
 *       The 11+ cognitive subsystems (creative, JEPA, VAE, FEP, parietal,
 *       predictive hierarchy, knowledge, grounded language, etc.) were never
 *       trained, leaving 95% of cognitive infrastructure at random weights.
 * HOW:  Derive appropriate inputs for each subsystem from available data:
 *       - features → observation/state/input for predictive modules
 *       - target → next state/derivative/expected output
 *       - label → text for knowledge and grounded language
 *       - loss → reward signal for reinforcement/self-heal
 *
 * GATING: Runs every cognitive_train_interval steps (default 5) to amortize
 *         cost. Subsystems that are NULL are silently skipped.
 *
 * @param brain Internal brain handle
 * @param features Input features from current training example
 * @param num_features Feature count
 * @param target Target output vector
 * @param target_size Target size
 * @param label Text label (can be NULL)
 * @param loss Current training loss from adaptive network
 */
static void brain_train_cognitive_subsystems(
    brain_t brain,
    const float* features,
    uint32_t num_features,
    const float* target,
    uint32_t target_size,
    const char* label,
    float loss)
{
    if (!brain) return;

    /* Interval gating: expensive subsystem training runs every N steps */
    uint32_t interval = brain->cognitive_train_interval;
    if (interval == 0) interval = 5;  /* default: every 5 steps */
    brain->cognitive_train_counter++;
    if ((brain->cognitive_train_counter % interval) != 0) return;

    /* === 1. GROUNDED LANGUAGE — distributional + syntactic learning === */
    if (brain->grounded_lang && label && label[0]) {
        grounded_language_learn_from_text(brain->grounded_lang, label);
        grounded_language_learn_syntax(brain->grounded_lang, label);
        brain->cognitive_stats.grounded_lang_steps++;
    }

    /* === 2. KNOWLEDGE SYSTEM — concept learning from text === */
    if (brain->knowledge && label && label[0]) {
        knowledge_learn_from_text(brain->knowledge, label,
                                  KNOWLEDGE_DOMAIN_GENERAL);
        brain->cognitive_stats.knowledge_steps++;
    }

    /* === 3. VAE — learn compressed latent representations === */
    if (brain->vae_training_bridge && brain->vae_enabled) {
        vae_training_bridge_t* vae = (vae_training_bridge_t*)brain->vae_training_bridge;
        vae_training_step_result_t vae_result = {0};
        /* Train VAE to reconstruct features from compressed latent */
        vae_training_step(vae, features, num_features,
                          features, num_features, &vae_result);
        brain->last_vae_free_energy = vae_result.loss.total_loss;
        brain->cognitive_stats.vae_steps++;
        brain->cognitive_stats.vae_last_loss = vae_result.loss.total_loss;
    }

    /* === 4. FEP-PARIETAL — hierarchical generative model training === */
    if (brain->parietal) {
        fep_parietal_bridge_t* fep_bridge = parietal_get_fep_bridge(brain->parietal);
        if (fep_bridge) {
            /* Train transition model: features (current state) → target (next state) */
            const float* obs_ptr = features;
            const float* tgt_ptr = target;
            fep_parietal_train_model(fep_bridge, &obs_ptr, &tgt_ptr, 1);
            brain->cognitive_stats.fep_parietal_steps++;
        }
    }

    /* === 5. PARIETAL PHYSICS NN — physics-informed dynamics learning === */
    if (brain->parietal && num_features >= 4 && target_size >= 4) {
        /* Treat features as state and target as derivative/next-state.
         * Use min(32, dim) to keep physics NN training fast. */
        uint32_t phys_dim = (num_features < 32) ? num_features : 32;
        const float* state_ptr = features;
        const float* deriv_ptr = target;
        /* Single-sample training through parietal wrapper */
        parietal_train_physics_nn(brain->parietal,
                                  &state_ptr, &deriv_ptr, 1, 1);
        brain->cognitive_stats.physics_nn_steps++;
    }

    /* === 6. PREDICTIVE HIERARCHY — hierarchical temporal prediction === */
    /* Fixed: bottom level now matches brain->config.num_inputs (was hardcoded 64).
     * Guard: only call if num_features >= hierarchy bottom dim. */
    if (brain->pred_hierarchy && brain->pred_hierarchy_enabled) {
        predictive_hierarchy_t* ph = (predictive_hierarchy_t*)brain->pred_hierarchy;
        float pred_loss = 0.0f;
        /* Safe: only feed if features can fill the bottom level */
        if (ph->bottom && num_features >= ph->bottom->dim) {
            pred_hier_learn_step(ph, features, &pred_loss);
        }
        brain->cognitive_stats.pred_hierarchy_steps++;
        brain->cognitive_stats.pred_hierarchy_last_loss = pred_loss;
    }

    /* === 7. JEPA PREDICTOR — latent space prediction for imagination === */
    if (brain->jepa_predictor && brain->jepa_predictor_enabled) {
        /* Create latent representations from features and target.
         * JEPA learns to predict target latent from context latent. */
        uint32_t latent_dim = (num_features < 256) ? num_features : 256;
        jepa_latent_t* context = jepa_latent_create_dim(latent_dim);
        jepa_latent_t* target_latent = jepa_latent_create_dim(latent_dim);

        if (context && target_latent) {
            /* Set embeddings from features and target data */
            jepa_latent_set_embedding(context, features,
                (num_features < latent_dim) ? num_features : latent_dim);
            jepa_latent_set_embedding(target_latent, target,
                (target_size < latent_dim) ? target_size : latent_dim);

            float jepa_loss = 0.0f;
            jepa_predictor_train_step(
                (jepa_predictor_t*)brain->jepa_predictor,
                context, target_latent, &jepa_loss);
            brain->cognitive_stats.jepa_steps++;
            brain->cognitive_stats.jepa_last_loss = jepa_loss;
        }
        if (context) jepa_latent_destroy(context);
        if (target_latent) jepa_latent_destroy(target_latent);
    }

    /* === 8. CREATIVE TRAINING — style learning from feedback === */
    if (brain->creative_training_bridge && brain->creative_enabled) {
        /* Submit the current training result as feedback:
         * Low loss = high rating, high loss = low rating.
         * This teaches the creative system what "good" output looks like. */
        uint8_t rating = (loss < 0.1f) ? 5 :
                         (loss < 0.3f) ? 4 :
                         (loss < 0.5f) ? 3 :
                         (loss < 0.7f) ? 2 : 1;
        creative_training_submit_feedback(
            brain->creative_training_bridge,
            target, /* content = target vector (what we're trying to produce) */
            0, /* ART_MODALITY_TEXT */
            rating,
            label);
        brain->cognitive_stats.creative_steps++;
    }

    /* === 9. SELF-HEAL ENGINE — learn from training success/failure === */
    if (brain->self_heal_engine && brain->self_heal_enabled) {
        crash_features_t cf = {0};
        uint32_t cf_dim = (num_features < SELF_HEAL_FEATURE_DIM) ?
                           num_features : SELF_HEAL_FEATURE_DIM;
        cf.n_features = cf_dim;
        memcpy(cf.features, features, cf_dim * sizeof(float));
        /* Success score = inverse of loss */
        float success = 1.0f - fminf(loss, 1.0f);
        self_heal_train_online(
            (self_heal_engine_t*)brain->self_heal_engine,
            &cf, FIX_PATTERN_UNKNOWN, success);
        brain->cognitive_stats.self_heal_steps++;
    }

    /* === 10. INTUITION SYSTEM — learn from training outcomes === */
    if (brain->intuition_system && brain->intuition_system_enabled) {
        /* Intuition learns from the gap between expected and actual outcomes.
         * We use the training loss as the actual outcome signal. */
        intuition_experience_t exp = {
            .id = brain->cognitive_train_counter,
            .hunch = NULL,
            .predicted_outcome = 0.0f,  /* Expected: zero loss */
            .actual_outcome = loss,     /* Actual: current loss */
            .timestamp = (float)nimcp_time_get_us(),
            .was_successful = (loss < 0.3f)
        };
        const intuition_experience_t* exp_ptr = &exp;
        intuition_train_from_experience(brain->intuition_system,
                                         &exp_ptr, 1);
        brain->cognitive_stats.intuition_steps++;
    }

    /* === 11. FEP ORCHESTRATOR — update free energy metrics === */
    BRAIN_ENSURE_FEP_ORCHESTRATOR(brain);
    if (brain->fep_orchestrator && brain->fep_orchestrator_enabled) {
        /* Update FEP scheduling metrics from training outcome:
         * loss → free energy, loss → prediction error.
         * The orchestrator uses these to modulate update intervals. */
        brain->fep_orchestrator->fep_metrics.free_energy = loss;
        brain->fep_orchestrator->fep_metrics.prediction_error = loss;
        brain->fep_orchestrator->fep_metrics.surprise = -logf(fmaxf(1.0f - loss, 1e-7f));
        brain->cognitive_stats.fep_orchestrator_steps++;
    }
}
extern void brain_clear_error(void);
extern bool ensure_writable_network(brain_t brain);
extern void clear_cache(brain_t brain);

// Import BBB global system (defined in nimcp_brain_init.c)
extern bbb_system_t nimcp_bbb_get_global_system(void);

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Find or create output label index
 *
 * WHY: Maps string labels to numeric indices
 * Enables human-readable classification
 *
 * COMPLEXITY: O(k) where k = num_existing_labels
 * OPTIMIZATION: Linear search sufficient for small label sets
 *
 * @param brain Brain handle
 * @param label Label string
 * @return Label index
 */
uint32_t nimcp_brain_learning_get_or_create_label_index(brain_t brain, const char* label)
{
    uint32_t result = 0;

    // THREAD SAFETY FIX: Protect label creation with mutex
    // Without this, concurrent threads could:
    // 1. Both see label doesn't exist
    // 2. Both create the same label at same index
    // 3. Result in buffer overflow past output_labels capacity → heap corruption
    nimcp_platform_mutex_lock(&brain->cache_mutex);

    // O(1) hash table lookup (replaces O(k) linear strcmp scan)
    if (brain->label_index) {
        void* found = hash_table_lookup_string(brain->label_index, label);
        if (found) {
            result = *(uint32_t*)found;
            goto done;
        }
    }

    // Guard: Check capacity — labels exceed output neurons
    if (brain->num_output_labels >= brain->config.num_outputs) {
        // Hash overflow labels across existing outputs instead of always index 0
        // (M2): Use djb2 consistent hash so labels get deterministic, well-distributed
        // assignments even when beyond capacity. Avoids the pathological case where
        // simple modulo of a monotonic counter clusters labels into low indices.
        uint32_t h = 5381;
        for (const char* p = label; *p; p++) {
            h = ((h << 5) + h) + (unsigned char)*p;
        }
        result = h % brain->config.num_outputs;
        // Still insert into hash table for O(1) future lookups of overflow labels
        if (brain->label_index)
            hash_table_insert_string(brain->label_index, label, &result, sizeof(uint32_t));
        LOG_WARN("Label overflow: '%s' hashed to output %u (num_labels=%u >= num_outputs=%u)",
                 label, result, brain->num_output_labels, brain->config.num_outputs);
        goto done;
    }

    // (M2): Early warning when approaching label capacity — gives callers a
    // chance to notice before overflow causes hash collisions.
    if (brain->num_output_labels >= (brain->config.num_outputs * 9u / 10u) &&
        brain->num_output_labels == (brain->config.num_outputs * 9u / 10u)) {
        LOG_WARN("Label capacity at 90%%: %u of %u output slots used. "
                 "New labels beyond capacity will be hash-mapped to existing outputs.",
                 brain->num_output_labels, brain->config.num_outputs);
    }

    // Create new label (use nimcp_malloc to match nimcp_free in brain_destroy)
    size_t label_len = strlen(label);
    brain->output_labels[brain->num_output_labels] = nimcp_malloc(label_len + 1);
    if (!brain->output_labels[brain->num_output_labels]) {
        result = 0;
        goto done;
    }
    memcpy(brain->output_labels[brain->num_output_labels], label, label_len + 1);
    result = brain->num_output_labels;

    // Insert into hash table for O(1) future lookups
    if (brain->label_index)
        hash_table_insert_string(brain->label_index, label, &result, sizeof(uint32_t));

    brain->num_output_labels++;

done:
    nimcp_platform_mutex_unlock(&brain->cache_mutex);
    return result;
}

/**
 * @brief Convert label to one-hot encoded output vector
 *
 * WHY: Transforms string labels to neural network targets
 * One-hot encoding standard for classification
 *
 * COMPLEXITY: O(n) where n = num_outputs
 *
 * @param brain Brain handle
 * @param label Label string
 * @param output Output buffer
 * @param confidence Confidence value for label
 */
void nimcp_brain_learning_label_to_output(brain_t brain, const char* label,
                                          float* output, float confidence)
{
    uint32_t label_idx = nimcp_brain_learning_get_or_create_label_index(brain, label);

    memset(output, 0, brain->config.num_outputs * sizeof(float));
    output[label_idx] = 1.0f;  /* Always one-hot; confidence scales LR, not target */
}

/**
 * WHAT: Adapt learning rate based on loss trend (Phase 11: Simple Meta-Learning)
 * WHY:  Accelerate when loss decreasing, slow down when loss increasing
 * HOW:  Track loss in rolling window, compute trend, adjust LR
 *
 * COMPLEXITY: O(1)
 *
 * BIOLOGICAL BASIS:
 * - Meta-learning: "learning to learn"
 * - Homeostatic regulation of synaptic plasticity
 */
void nimcp_brain_learning_adapt_learning_rate(brain_t brain, float current_loss)
{
    // Guard: NULL check
    if (!brain) {
        return;
    }

    // Guard: Initialize base_learning_rate on first call
    if (brain->base_learning_rate == 0.0F) {
        brain->base_learning_rate = brain->config.learning_rate;
    }

    // Store current loss in circular buffer
    brain->loss_history[brain->loss_history_index] = current_loss;
    brain->loss_history_index = (brain->loss_history_index + 1) % LOSS_HISTORY_SIZE;
    if (brain->loss_history_count < LOSS_HISTORY_SIZE) {
        brain->loss_history_count++;
    }

    // Need at least 3 samples to compute trend
    if (brain->loss_history_count < 3) {
        return;
    }

    // Compute loss trend: recent avg vs older avg
    // Use modular arithmetic to correctly index the circular buffer.
    // Oldest entry is at loss_history_index (the next write position),
    // so we walk forward from there using (head + offset) % capacity.
    float recent_avg = 0.0F;
    float older_avg = 0.0F;
    uint32_t half = brain->loss_history_count / 2;
    uint32_t head = (brain->loss_history_index + LOSS_HISTORY_SIZE - brain->loss_history_count) % LOSS_HISTORY_SIZE;

    // Older half (oldest entries first)
    for (uint32_t i = 0; i < half; i++) {
        older_avg += brain->loss_history[(head + i) % LOSS_HISTORY_SIZE];
    }
    older_avg /= half;

    // Recent half (newest entries)
    for (uint32_t i = half; i < brain->loss_history_count; i++) {
        recent_avg += brain->loss_history[(head + i) % LOSS_HISTORY_SIZE];
    }
    recent_avg /= (brain->loss_history_count - half);

    // Compute trend
    float trend = recent_avg - older_avg;

    // Adapt learning rate
    if (trend < -0.01F) {
        brain->config.learning_rate *= 1.05F;  // Accelerate
    } else if (trend > 0.01F) {
        brain->config.learning_rate *= 0.9F;   // Slow down
    }

    // Bounds: [0.1x, 10x] of base rate
    float min_lr = brain->base_learning_rate * 0.1F;
    float max_lr = brain->base_learning_rate * 10.0F;
    if (brain->config.learning_rate < min_lr) {
        brain->config.learning_rate = min_lr;
    }
    if (brain->config.learning_rate > max_lr) {
        brain->config.learning_rate = max_lr;
    }
}

/**
 * @brief Energy function for quantum annealing weight optimization
 *
 * WHAT: Compute L2 regularization energy for given weights
 * WHY:  Simple proxy energy function for weight optimization
 * HOW:  Sum of squared weights, normalized by dimension
 *
 * NOTE: Full implementation would use validation loss
 *
 * @param weights Weight vector
 * @param dim Vector dimension
 * @param user_data Unused
 * @return Energy (lower is better)
 */
float nimcp_brain_learning_quantum_weight_energy(const float* weights, uint32_t dim,
                                                  void* user_data)
{
    (void)user_data;  // Unused
    float energy = 0.0F;
    for (uint32_t i = 0; i < dim; i++) {
        energy += weights[i] * weights[i];
    }
    return energy / (float)dim;  // Normalized
}

//=============================================================================
// Multi-Network Training Helpers
//=============================================================================

/**
 * @brief Lazily create the CNN trainer if not already present
 *
 * WHY: DRY — shared by brain_learn_vector() and brain_enable_multi_network_training()
 * HOW: Dense(num_inputs→hidden)→ReLU→Dense(hidden→num_outputs)
 *
 * @param brain Brain handle
 * @return 0 on success (or already exists), -1 on failure
 */
static int ensure_cnn_trainer(brain_t brain)
{
    if (brain->cnn_trainer) return 0;

    uint32_t num_inputs = brain->config.num_inputs;
    uint32_t num_outputs = brain->config.num_outputs;
    uint32_t hidden_size = num_outputs * 4;
    if (hidden_size < 32) hidden_size = 32;
    if (hidden_size > 512) hidden_size = 512;

    cnn_trainer_config_t cnn_cfg;
    cnn_trainer_default_config(&cnn_cfg);
    cnn_cfg.learning_rate = 0.001f;

    cnn_trainer_t* trainer = cnn_trainer_create(&cnn_cfg);
    if (!trainer) {
        NIMCP_LOGGING_WARN("Failed to create CNN trainer");
        return -1;
    }

    cnn_dense_config_t dense1 = {
        .in_features = num_inputs,
        .out_features = hidden_size,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    cnn_trainer_add_dense_layer(trainer, &dense1);
    cnn_trainer_add_activation_layer(trainer, CNN_ACTIVATION_RELU);

    cnn_dense_config_t dense2 = {
        .in_features = hidden_size,
        .out_features = num_outputs,
        .activation = CNN_ACTIVATION_NONE,
        .use_bias = true,
        .weight_init_std = 0.01f
    };
    cnn_trainer_add_dense_layer(trainer, &dense2);

    brain->cnn_trainer = trainer;
    NIMCP_LOGGING_INFO("CNN trainer created: %u→%u→%u", num_inputs, hidden_size, num_outputs);
    return 0;
}

/**
 * @brief CNN classifier training step (label-based, not dense target)
 *
 * WHY: CNN trains as a classifier using one-hot labels, complementing the
 *      adaptive network's dense embedding training
 * HOW: Builds one-hot target from label index, runs CNN forward/backward/step
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @param label Label string for classification
 * @return CNN loss, or -1.0f on failure (non-fatal)
 */
static float brain_learn_vector_cnn_step(brain_t brain, const float* features,
                                          uint32_t num_features, const char* label)
{
    if (ensure_cnn_trainer(brain) != 0) return -1.0f;

    uint32_t num_outputs = brain->config.num_outputs;
    uint32_t label_idx = nimcp_brain_learning_get_or_create_label_index(brain, label);

    /* Build one-hot target for CNN classifier */
    float* one_hot = nimcp_calloc(num_outputs, sizeof(float));
    if (!one_hot) return -1.0f;
    if (label_idx < num_outputs) {
        one_hot[label_idx] = 1.0f;
    }

    /* Create 2D tensors {1, N} — dense layers require batch dimension */
    uint32_t input_dims[2] = {1, num_features};
    uint32_t target_dims[2] = {1, num_outputs};
    nimcp_tensor_t* input_tensor = nimcp_tensor_create(input_dims, 2, NIMCP_DTYPE_F32);
    nimcp_tensor_t* target_tensor = nimcp_tensor_create(target_dims, 2, NIMCP_DTYPE_F32);

    if (!input_tensor || !target_tensor) {
        if (input_tensor) nimcp_tensor_destroy(input_tensor);
        if (target_tensor) nimcp_tensor_destroy(target_tensor);
        nimcp_free(one_hot);
        return -1.0f;
    }

    float* in_data = (float*)nimcp_tensor_data(input_tensor);
    float* tgt_data = (float*)nimcp_tensor_data(target_tensor);
    if (in_data) memcpy(in_data, features, num_features * sizeof(float));
    if (tgt_data) memcpy(tgt_data, one_hot, num_outputs * sizeof(float));
    nimcp_free(one_hot);

    /* CNN training: zero_grad → forward → backward → step */
    cnn_trainer_zero_grad(brain->cnn_trainer);

    cnn_forward_result_t fwd_result = {0};
    nimcp_error_t rc = cnn_trainer_forward(brain->cnn_trainer, input_tensor, &fwd_result);
    if (rc != NIMCP_SUCCESS) {
        nimcp_tensor_destroy(input_tensor);
        nimcp_tensor_destroy(target_tensor);
        return -1.0f;
    }

    rc = cnn_trainer_backward(brain->cnn_trainer, target_tensor, &fwd_result);
    if (rc == NIMCP_SUCCESS) {
        cnn_trainer_step(brain->cnn_trainer);
    }

    /* Compute loss (MSE between output and one-hot target) */
    float cnn_loss = 0.0f;
    if (fwd_result.output) {
        const float* out_data = (const float*)nimcp_tensor_data(fwd_result.output);
        const float* tgt = (const float*)nimcp_tensor_data(target_tensor);
        if (out_data && tgt) {
            for (uint32_t i = 0; i < num_outputs; i++) {
                float diff = out_data[i] - tgt[i];
                cnn_loss += diff * diff;
            }
            cnn_loss /= (float)num_outputs;
        }
    }

    nimcp_tensor_destroy(input_tensor);
    nimcp_tensor_destroy(target_tensor);
    return cnn_loss;
}

//=============================================================================
// Parallel Training Infrastructure
//=============================================================================

/**
 * @brief Context for parallel training tasks
 *
 * Adaptive learns first (modifies shared weights), then CNN + SNN + LNN
 * run in parallel — each operates on independent weight sets.
 */
typedef struct {
    brain_t brain;
    const float* features;
    uint32_t num_features;
    const float* target;
    uint32_t target_size;
    const char* label;

    /* Per-network results (each written by ONE task only) */
    float cnn_loss;
    bool cnn_done;

    training_dispatch_result_t snn_res;
    bool snn_done;

    training_dispatch_result_t lnn_res;
    bool lnn_done;
} learn_task_ctx_t;

/**
 * @brief CNN training task (thread pool worker)
 * CNN has its own independent weights/optimizer — safe to parallelize.
 */
static void learn_cnn_task(void* arg)
{
    learn_task_ctx_t* ctx = (learn_task_ctx_t*)arg;

    if (ctx->label && ctx->label[0]) {
        ctx->cnn_loss = brain_learn_vector_cnn_step(ctx->brain, ctx->features,
                                                     ctx->num_features, ctx->label);
    }
    ctx->cnn_done = true;
}

/**
 * @brief SNN training task (thread pool worker)
 * SNN applies STDP/eProp to its own snn_network weights — safe after adaptive completes.
 */
static void learn_snn_task(void* arg)
{
    learn_task_ctx_t* ctx = (learn_task_ctx_t*)arg;

    if (ctx->brain->snn_network && ctx->brain->snn_training_ctx) {
        training_dispatch_snn_step(ctx->brain, ctx->features, ctx->num_features,
                                    ctx->target, ctx->target_size, &ctx->snn_res);
    }
    ctx->snn_done = true;
}

/**
 * @brief LNN training task (thread pool worker)
 * LNN has its own ODE weights/state — fully independent.
 */
static void learn_lnn_task(void* arg)
{
    learn_task_ctx_t* ctx = (learn_task_ctx_t*)arg;

    if (ctx->brain->lnn_network && ctx->brain->lnn_training_ctx) {
        training_dispatch_lnn_step(ctx->brain, ctx->features, ctx->num_features,
                                    ctx->target, ctx->target_size, &ctx->lnn_res);
    }
    ctx->lnn_done = true;
}

//=============================================================================
// Public Learning API
//=============================================================================

/**
 * @brief Learn from a dense target vector (distillation / generative training)
 *
 * WHY: Enables teacher-model distillation and semantic embedding training
 * HOW: Builds training_example_t with dense target, uses LEARN_MODE_DISTILLATION
 *
 * Training parallelism: Adaptive first (GPU, modifies shared weights), then
 * CNN + SNN + LNN in parallel (each has independent weights).
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @param target Dense target output vector
 * @param target_size Target vector size
 * @param label Optional semantic label (can be NULL)
 * @param confidence Training weight
 * @return Loss value or -1 on error
 */
float brain_learn_vector(brain_t brain, const float* features, uint32_t num_features,
                          const float* target, uint32_t target_size,
                          const char* label, float confidence)
{
    if (!brain || !features || !target) {
        set_error("Invalid parameters to brain_learn_vector");
        return -1.0f;
    }

    /* Allow flexible feature sizes. The adaptive network truncates or pads internally. */
    if (num_features > brain->config.num_inputs * 2) {
        NIMCP_LOGGING_WARN("brain_learn_vector: num_features %u >> num_inputs %u, truncating",
                  num_features, brain->config.num_inputs);
        num_features = brain->config.num_inputs;
    }

    /* Allow target_size <= num_outputs. The adaptive network handles
     * size mismatch internally (truncation or zero-padding). Previously
     * this returned -1.0f which silently broke learn_vector on daemon
     * brains where num_outputs=4096 but training uses 2048-dim targets. */
    if (target_size > brain->config.num_outputs) {
        NIMCP_LOGGING_WARN("brain_learn_vector: target_size %u > num_outputs %u, truncating",
                  target_size, brain->config.num_outputs);
        target_size = brain->config.num_outputs;
    }

    /* Validate features for NaN/Inf */
    for (uint32_t i = 0; i < num_features; i++) {
        if (isnan(features[i]) || isinf(features[i])) {
            NIMCP_LOGGING_WARN("brain_learn_vector: NaN/Inf in features[%u] = %f", i, features[i]);
            set_error("Invalid feature value at index %u: NaN or Inf", i);
            return -1.0f;
        }
    }

    /* Validate target for NaN/Inf */
    for (uint32_t i = 0; i < target_size; i++) {
        if (isnan(target[i]) || isinf(target[i])) {
            NIMCP_LOGGING_WARN("brain_learn_vector: NaN/Inf in target[%u] = %f", i, target[i]);
            set_error("Invalid target value at index %u: NaN or Inf", i);
            return -1.0f;
        }
    }

    /* Pre-create cortex CNNs for any staged sensory data BEFORE ensure_writable_network.
     * ensure_writable_network may fail on first calls (COW/GPU warmup) but cortex CNN
     * creation is independent and should succeed regardless.
     * This ensures cortex CNN allocation succeeds before GPU buffers consume memory. */
    if (label && label[0]) {
        /* cortex_cnn_create declared in training/nimcp_cortex_cnn.h */
        extern void* fno_audio_create(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
        extern void cortex_cnn_set_fno_audio(struct cortex_cnn_processor*, void*);
        extern void cortex_cnn_set_fno_visual(struct cortex_cnn_processor*, void*);
        extern void cortex_cnn_set_fno_speech(struct cortex_cnn_processor*, void*);

        if (brain->staged_sensory.visual_frame && !brain->cortex_cnns[0]) {
            brain->cortex_cnns[0] = cortex_cnn_create(0, 0);
            if (brain->cortex_cnns[0]) {
                /* FNO visual: spatial frequency analysis (32x32 grayscale → 1024 samples) */
                void* fno = fno_audio_create(1024, 64, 16, 32, 2);
                if (fno) cortex_cnn_set_fno_visual(brain->cortex_cnns[0], fno);
            }
        }
        if (brain->staged_sensory.audio_data && !brain->cortex_cnns[1]) {
            brain->cortex_cnns[1] = cortex_cnn_create(1, 0);
            if (brain->cortex_cnns[1]) {
                /* FNO audio: mel-spectrogram spectral convolution */
                void* fno = fno_audio_create(128, 64, 16, 32, 2);
                if (fno) cortex_cnn_set_fno_audio(brain->cortex_cnns[1], fno);
            }
        }
        if (brain->staged_sensory.speech_data && !brain->cortex_cnns[2]) {
            brain->cortex_cnns[2] = cortex_cnn_create(2, 0);
            if (brain->cortex_cnns[2]) {
                /* FNO speech: phoneme spectral patterns */
                void* fno = fno_audio_create(128, 64, 16, 32, 2);
                if (fno) cortex_cnn_set_fno_speech(brain->cortex_cnns[2], fno);
            }
        }
        if (brain->staged_sensory.somato_data && !brain->cortex_cnns[3])
            brain->cortex_cnns[3] = cortex_cnn_create(3, 0);
        /* Somato doesn't use FNO — touch/pressure data is spatial, not spectral */
    }

    if (!ensure_writable_network(brain)) {
        return -1.0f;
    }

    /* Blend cortex CNN embeddings from PREVIOUS step into current features.
     * Uses cached fused embedding (set by decide_full or previous learn_vector).
     * This avoids running cortex CNN forward twice per step (which corrupts
     * the CNN trainer's forward_result state). */
    float* blended_features = (float*)features;
    bool owns_blended = false;
    if (brain->cortex_cnn_fused_embedding && brain->cortex_cnn_fused_dim > 0) {
        blended_features = nimcp_malloc(num_features * sizeof(float));
        if (blended_features) {
            memcpy(blended_features, features, num_features * sizeof(float));
            uint32_t inject_n = (brain->cortex_cnn_fused_dim < num_features)
                              ? brain->cortex_cnn_fused_dim : num_features;
            for (uint32_t i = 0; i < inject_n; i++) {
                blended_features[i] += 0.3f * brain->cortex_cnn_fused_embedding[i];
            }
            owns_blended = true;
        } else {
            blended_features = (float*)features;
        }
    }

    /* Build training example with dense target directly — no one-hot conversion */
    training_example_t example = {
        .input = blended_features,
        .input_size = num_features,
        .target = (float*)target,
        .target_size = target_size,
        .confidence = confidence
    };
    if (label) {
        strncpy(example.label, label, sizeof(example.label) - 1);
    } else {
        example.label[0] = '\0';
    }

    /* === HYPERLEDGER EOV: BEGIN transaction === */
    uint64_t hl_tx_id = 0;
    if (brain->hyperledger_bridge && brain->hyperledger_enabled) {
        hl_tx_id = hyperledger_eov_begin(brain->hyperledger_bridge, 0.0f);
    }

    /* === MEMORY-INFORMED LEARNING ===
     * Recall similar past experiences before learning.
     * If familiar (high recall confidence): reduce learning rate slightly.
     * If truly novel (no recall): boost learning rate slightly. */
    float lr_memory_factor = 1.0f;
    if (brain->engram_system && brain->engram_system->active_count > 10) {
        /* Use feature indices as cue pattern */
        uint32_t cue_count = num_features < 64 ? num_features : 64;
        uint32_t cue_ids[64];
        for (uint32_t ci = 0; ci < cue_count; ci++) {
            cue_ids[ci] = ci;
        }
        float recall_conf = 0.0f;
        uint32_t recall_out[256];
        float recall_act[256];
        uint64_t recalled = engram_recall(
            brain->engram_system, cue_ids, cue_count,
            recall_out, recall_act, 256, &recall_conf);

        if (recalled > 0 && recall_conf > 0.6f) {
            /* Familiar input: reduce LR to avoid overwriting */
            lr_memory_factor = 0.7f;
        } else if (recalled == 0) {
            /* Truly novel: boost LR for faster acquisition */
            lr_memory_factor = 1.3f;
        }
    }

    /* Step 1: Adaptive learns FIRST (primary network, GPU-accelerated)
     * Must complete before SNN training since SNN may share neural_network_t weights */
    float effective_lr = brain->config.learning_rate * lr_memory_factor;
    float loss = adaptive_network_learn(brain->network, &example,
                                         LEARN_MODE_DISTILLATION,
                                         effective_lr);

    if (loss < 0.0f) {
        NIMCP_LOGGING_WARN("adaptive_network_learn returned %.2f: network=%p, input_size=%u, target_size=%u, lr=%.6f",
                           loss, (void*)brain->network,
                           num_features, target_size, brain->config.learning_rate);
    }

    /* === HYPERLEDGER EOV: ORDER + VALIDATE === */
    if (brain->hyperledger_bridge && brain->hyperledger_enabled && hl_tx_id > 0) {
        float grad_norm = loss;  /* Use loss as proxy for gradient magnitude */
        hyperledger_eov_order(brain->hyperledger_bridge, hl_tx_id, grad_norm, num_features);
        /* Validate: if rejected, skip biological plasticity this step */
        bool eov_valid = hyperledger_eov_validate(brain->hyperledger_bridge,
                                                   hl_tx_id, NULL, 0);
        if (eov_valid) {
            hyperledger_eov_commit(brain->hyperledger_bridge, hl_tx_id,
                                   fabsf(loss), loss);
        }
        /* Note: even if EOV rejects, the adaptive_network_learn already ran.
         * The rejection is logged for audit purposes and can trigger rollback.
         * Future: move validation BEFORE weight commit for full EOV semantics. */
    }

    nimcp_brain_learning_adapt_learning_rate(brain, loss);
    __atomic_fetch_add(&brain->stats.total_learning_steps, 1, __ATOMIC_RELAXED);

    /* Biological plasticity integration (skip in fast training or batch mode).
     * OPTIMIZATION: Gate expensive plasticity updates to run every 10 steps.
     * Biological plasticity operates on slower timescales than gradient updates,
     * so running every 10th step is biologically more realistic AND faster. */
    brain->plasticity_step_counter++;
    if (!brain->config.fast_training_mode && !brain->config.defer_bio_plasticity
        && (brain->plasticity_step_counter % 10 == 0)) {
        /* TPB: Loss → RPE → neuromodulator update */
        if (brain->plasticity_bridge && brain->enable_plasticity_bridge) {
            float rpe = 0.0f;
            tpb_report_loss(brain->plasticity_bridge, loss, &rpe);
        }
        /* EDP: Three-factor eligibility consolidation */
        if (brain->event_driven_plasticity && brain->enable_event_driven_plasticity) {
            edp_process_prediction_error(brain->event_driven_plasticity, loss, 0);
        }
        /* Plasticity coordinator: interval-based STDP/BCM/homeostatic updates */
        if (brain->plasticity_coordinator && brain->plasticity_coordinator_enabled) {
            uint64_t now_ms = nimcp_time_get_us() / 1000;
            plasticity_coordinator_update(brain->plasticity_coordinator, now_ms, 1.0f);
        }
        /* Structural plasticity: form new synapses for high-activity pairs */
        if (brain->structural_plasticity && brain->structural_plasticity_enabled) {
            float activity_hz = (loss < 0.5f) ? 30.0f : 5.0f;
            if (structural_plasticity_should_form(brain->structural_plasticity, activity_hz)
                && brain->config.num_inputs > 0 && brain->config.num_outputs > 0) {
                uint32_t pre = (uint32_t)(features[0] * 997.0f) % brain->config.num_inputs;
                uint32_t post = (uint32_t)(features[1] * 991.0f) % brain->config.num_outputs;
                uint32_t syn_id = 0;
                structural_plasticity_form_synapse(
                    brain->structural_plasticity, pre, post, activity_hz, &syn_id);
            }
        }
        /* Neuromodulator-gated reward learning: strengthen active pathways.
         * DA level already set by TPB above; reward = inverse of loss.
         * O(active) — only neurons with |state| > threshold. */
        {
            neural_network_t base_net = adaptive_network_get_base_network(brain->network);
            if (base_net) {
                BRAIN_ENSURE_NEUROMOD(brain);
                if (brain->neuromodulator_system) {
                    neural_network_set_neuromodulator_system(base_net, brain->neuromodulator_system);
                }
                float reward = 1.0f - fminf(loss, 1.0f);
                uint32_t modified = neural_network_apply_reward_learning_active(
                    base_net, reward, REWARD_LEARNING_RATE, nimcp_time_get_us(), REWARD_ACTIVITY_THRESHOLD);
                if (modified > 0) {
                    adaptive_network_invalidate_gpu_structure(brain->network);
                }
            }
        }
    }

    /* === Thousand Brains Integration (Hawkins) ===
     * Step the WM-TB bridge: gather spatial state from reference frames,
     * object consensus from column voting, temporal predictions from dendritic
     * sequences → push to world model → generate top-down expectations.
     * Gated to every 10 steps alongside other biological plasticity. */
    if (brain->config.enable_wm_thousand_brains_bridge
        && brain->wm_thousand_brains_bridge
        && (brain->plasticity_step_counter % 10 == 0)) {
        nimcp_error_t tb_rc = wm_tb_bridge_step(brain->wm_thousand_brains_bridge);
        if (tb_rc != NIMCP_SUCCESS) {
            NIMCP_LOGGING_WARN("brain_learn_vector: wm_tb_bridge_step failed (rc=%d)", (int)tb_rc);
        }
    }

    /* Step the full TB integration hub: entorhinal→ref frames, hypercolumns→features,
     * attention→voting, oscillations→timing, predictive coding↔sequences,
     * hippocampus sync, voting→global workspace, ToM perspective-taking.
     * Runs every 10 steps to match biological plasticity cadence. */
    if (brain->config.enable_thousand_brains_integration
        && brain->tb_integration_hub
        && (brain->plasticity_step_counter % 10 == 0)) {
        int hub_rc = tb_integration_step(brain->tb_integration_hub);
        if (hub_rc != 0) {
            NIMCP_LOGGING_WARN("brain_learn_vector: tb_integration_step failed (rc=%d)", hub_rc);
        }
    }

    /* BPTT: Record current example in temporal buffer and replay history.
     * Biological: Hippocampal replay — recent experiences are replayed during
     * consolidation with temporally discounted gradients. This enables the
     * network to learn sequential dependencies across recent training steps.
     * Truncated BPTT: accumulate gradients over the last bptt_window_size steps
     * with exponential discount (γ^t) to avoid exploding gradients. */
    if (brain->bptt_enabled && brain->bptt_buffer && brain->bptt_window_size > 0) {
        uint32_t w = brain->bptt_window_size;
        uint32_t in_dim = brain->bptt_input_dim;
        uint32_t out_dim = brain->bptt_output_dim;

        /* Lazy resize if dimensions changed (first call or reconfiguration) */
        if (in_dim != num_features || out_dim != target_size) {
            for (uint32_t i = 0; i < w; i++) {
                nimcp_free(brain->bptt_buffer[i].input);
                nimcp_free(brain->bptt_buffer[i].output);
                nimcp_free(brain->bptt_buffer[i].target);
                brain->bptt_buffer[i].input = nimcp_calloc(num_features, sizeof(float));
                brain->bptt_buffer[i].output = nimcp_calloc(target_size, sizeof(float));
                brain->bptt_buffer[i].target = nimcp_calloc(target_size, sizeof(float));
                /* Guard partial allocation failure */
                if (!brain->bptt_buffer[i].input || !brain->bptt_buffer[i].output ||
                    !brain->bptt_buffer[i].target) {
                    NIMCP_LOGGING_WARN("BPTT buffer realloc failed at slot %u", i);
                    break;
                }
            }
            brain->bptt_input_dim = num_features;
            brain->bptt_output_dim = target_size;
            brain->bptt_count = 0;
            brain->bptt_head = 0;
        }

        /* Store current example at head position */
        uint32_t h = brain->bptt_head;
        if (brain->bptt_buffer[h].input && brain->bptt_buffer[h].target) {
            memcpy(brain->bptt_buffer[h].input, features, num_features * sizeof(float));
            memcpy(brain->bptt_buffer[h].target, target, target_size * sizeof(float));
            /* Capture current output via lightweight adaptive forward (no full decision pipeline).
             * OPTIMIZATION: Replaced allocate_decision + perform_forward_pass + brain_free_decision
             * with a single adaptive_network_forward call directly into the BPTT buffer. */
            if (brain->bptt_buffer[h].output) {
                adaptive_network_forward(brain->network, features, num_features,
                                         brain->bptt_buffer[h].output, target_size,
                                         nimcp_time_get_us());
            }
            brain->bptt_buffer[h].loss = loss;
        }
        brain->bptt_head = (h + 1) % w;
        if (brain->bptt_count < w) brain->bptt_count++;

        /* Replay: walk backward through temporal buffer with discounted LR.
         * Skip the most recent entry (already learned above). */
        if (brain->bptt_count > 1) {
            float discount = brain->bptt_discount;
            float gamma = discount;
            float bptt_lr = brain->config.learning_rate;

            for (uint32_t step = 1; step < brain->bptt_count; step++) {
                uint32_t idx = (brain->bptt_head + w - 1 - step) % w;
                if (!brain->bptt_buffer[idx].input || !brain->bptt_buffer[idx].target) {
                    continue;
                }

                training_example_t replay = {
                    .input = brain->bptt_buffer[idx].input,
                    .input_size = num_features,
                    .target = brain->bptt_buffer[idx].target,
                    .target_size = target_size,
                    .confidence = confidence * gamma
                };
                replay.label[0] = '\0';

                adaptive_network_learn(brain->network, &replay,
                                       LEARN_MODE_DISTILLATION,
                                       bptt_lr * gamma);
                gamma *= discount;  /* Exponential decay */
            }
        }
    }

    /* Step 2: Secondary networks — unified or legacy path */
    if (brain->unified_training && brain->config.use_unified_training) {
        /* Unified training path: single composite loss across all networks.
         * Each network receives direct ground truth supervision via MSE gradient,
         * plus cross-network gradient bridges, shared AdamW + LR scheduling. */
        nimcp_utm_step_result_t utm_result = {0};
        int utm_rc = nimcp_utm_step(brain->unified_training,
                                    features, num_features,
                                    target, target_size,
                                    &utm_result);
        if (utm_rc == 0) {
            /* Blend unified composite loss with adaptive loss.
             * Guard against NaN/inf from UTM (e.g. LNN gradient explosion). */
            float utm_loss = utm_result.composite_loss;
            if (isfinite(utm_loss) && utm_loss >= 0.0f) {
                loss = 0.5f * loss + 0.5f * utm_loss;
            }

            /* Update per-network metrics from UTM results so monitoring works */
            const float a = 0.01f;
            nimcp_unified_training_manager_t* utm = brain->unified_training;
            for (uint32_t n = 0; n < utm->num_networks; n++) {
                if (!utm->networks[n].enabled) continue;
                float nloss = utm_result.per_network_loss[n];
                if (!isfinite(nloss) || nloss < 0.0f) continue;

                switch (utm->networks[n].ops->type) {
                    case NIMCP_TRAINABLE_CNN:
                        brain->network_metrics.last_cnn_loss = nloss;
                        brain->network_metrics.cnn_steps++;
                        if (!isfinite(brain->network_metrics.ema_cnn_loss))
                            brain->network_metrics.ema_cnn_loss = nloss;
                        else
                            brain->network_metrics.ema_cnn_loss =
                                (1.0f - a) * brain->network_metrics.ema_cnn_loss + a * nloss;
                        break;
                    case NIMCP_TRAINABLE_SNN:
                        brain->network_metrics.last_snn_loss = nloss;
                        brain->network_metrics.snn_steps++;
                        if (!isfinite(brain->network_metrics.ema_snn_loss))
                            brain->network_metrics.ema_snn_loss = nloss;
                        else
                            brain->network_metrics.ema_snn_loss =
                                (1.0f - a) * brain->network_metrics.ema_snn_loss + a * nloss;
                        break;
                    case NIMCP_TRAINABLE_LNN:
                        brain->network_metrics.last_lnn_loss = nloss;
                        brain->network_metrics.lnn_steps++;
                        if (!isfinite(brain->network_metrics.ema_lnn_loss))
                            brain->network_metrics.ema_lnn_loss = nloss;
                        else
                            brain->network_metrics.ema_lnn_loss =
                                (1.0f - a) * brain->network_metrics.ema_lnn_loss + a * nloss;
                        /* Update HNN metrics if Hamiltonian is active on any LNN layer */
                        if (brain->lnn_network) {
                            /* lnn_hamiltonian_get_energy/deviation declared in nimcp_lnn_hamiltonian.h */
                            /* Check first layer for Hamiltonian */
                            if (brain->lnn_network->n_layers > 0 &&
                                brain->lnn_network->layers[0] &&
                                brain->lnn_network->layers[0]->use_hamiltonian &&
                                brain->lnn_network->layers[0]->H_net) {
                                brain->network_metrics.hnn_active = true;
                                brain->network_metrics.hnn_energy =
                                    lnn_hamiltonian_get_energy((lnn_hamiltonian_net_t*)brain->lnn_network->layers[0]->H_net);
                                brain->network_metrics.hnn_energy_deviation =
                                    lnn_hamiltonian_get_energy_deviation((lnn_hamiltonian_net_t*)brain->lnn_network->layers[0]->H_net);
                                if (brain->network_metrics.hnn_initial_energy == 0.0f)
                                    brain->network_metrics.hnn_initial_energy =
                                        brain->network_metrics.hnn_energy;
                            }
                        }
                        break;
                    default:
                        break;
                }
            }
        }

        /* Update FNO audio metrics from cortex CNN audio processor */
        if (brain->cortex_cnns[1]) {
            extern void* cortex_cnn_get_fno_audio(const struct cortex_cnn_processor*);
            void* fno = cortex_cnn_get_fno_audio(brain->cortex_cnns[1]);
            if (fno) {
                extern float fno_audio_get_ema_loss(const void*);
                extern float fno_audio_get_last_loss(const void*);
                extern uint64_t fno_audio_get_steps(const void*);
                extern uint32_t fno_audio_get_param_count(const void*);
                brain->network_metrics.fno_audio_loss = fno_audio_get_last_loss(fno);
                brain->network_metrics.fno_audio_ema_loss = fno_audio_get_ema_loss(fno);
                brain->network_metrics.fno_audio_steps = fno_audio_get_steps(fno);
                brain->network_metrics.fno_audio_params = fno_audio_get_param_count(fno);
            }
        }

        /* Update FNO population metrics from SNN FNO models */
        if (brain->snn_fno_populations && brain->snn_fno_count > 0) {
            float total_mse = 0.0f;
            uint64_t total_steps = 0;
            uint64_t total_inf = 0;
            bool any_ready = false;
            uint32_t active = 0;
            for (uint32_t p = 0; p < brain->snn_fno_count; p++) {
                snn_fno_population_t* fp = (snn_fno_population_t*)brain->snn_fno_populations[p];
                if (!fp) continue;
                total_mse += fp->train_mse;
                total_steps += fp->train_steps;
                total_inf += fp->inference_steps;
                if (fp->ready_for_inference) any_ready = true;
                active++;
            }
            if (active > 0) {
                brain->network_metrics.fno_pop_train_mse = total_mse / (float)active;
                brain->network_metrics.fno_pop_val_mse = 0.0f; /* TODO: aggregate val MSE */
                brain->network_metrics.fno_pop_ready = any_ready;
                brain->network_metrics.fno_pop_train_steps = total_steps;
                brain->network_metrics.fno_pop_inference_steps = total_inf;
            }
        }
    } else {
        /* Legacy path: secondary networks learn IN PARALLEL (after adaptive completes) */
        bool has_cnn = (label && label[0]) && brain->config.train_cnn;
        bool has_snn = (brain->snn_network && brain->snn_training_ctx && brain->config.train_snn);
        bool has_lnn = (brain->lnn_network && brain->lnn_training_ctx && brain->config.train_lnn);
        int secondary_count = (has_cnn ? 1 : 0) + (has_snn ? 1 : 0) + (has_lnn ? 1 : 0);

        if (brain->inference_pool && secondary_count >= 2) {
            /* Parallel path: submit CNN + SNN + LNN tasks to thread pool */
            learn_task_ctx_t* ctx = nimcp_calloc(1, sizeof(learn_task_ctx_t));
            if (ctx) {
                ctx->brain = brain;
                ctx->features = features;
                ctx->num_features = num_features;
                ctx->target = target;
                ctx->target_size = target_size;
                ctx->label = label;

                if (has_cnn) nimcp_pool_submit(brain->inference_pool, learn_cnn_task, ctx);
                if (has_snn) nimcp_pool_submit(brain->inference_pool, learn_snn_task, ctx);
                if (has_lnn) nimcp_pool_submit(brain->inference_pool, learn_lnn_task, ctx);

                nimcp_pool_wait(brain->inference_pool);

                /* Collect per-network metrics from parallel results */
                const float a = 0.01f;
                /* NaN-safe EMA update macro: reset EMA if it becomes non-finite */
                #define SAFE_EMA_UPDATE(ema, val, alpha) do { \
                    if (!isfinite(ema)) (ema) = (val); \
                    else (ema) = (1.0f - (alpha)) * (ema) + (alpha) * (val); \
                } while (0)

                if (ctx->cnn_done && ctx->cnn_loss >= 0.0f && isfinite(ctx->cnn_loss)) {
                    brain->network_metrics.last_cnn_loss = ctx->cnn_loss;
                    brain->network_metrics.cnn_steps++;
                    SAFE_EMA_UPDATE(brain->network_metrics.ema_cnn_loss, ctx->cnn_loss, a);
                }
                if (ctx->snn_done && ctx->snn_res.loss >= 0.0f && isfinite(ctx->snn_res.loss)) {
                    brain->network_metrics.last_snn_loss = ctx->snn_res.loss;
                    brain->network_metrics.snn_steps++;
                    SAFE_EMA_UPDATE(brain->network_metrics.ema_snn_loss, ctx->snn_res.loss, a);
                }
                if (ctx->lnn_done && ctx->lnn_res.loss >= 0.0f && isfinite(ctx->lnn_res.loss)) {
                    brain->network_metrics.last_lnn_loss = ctx->lnn_res.loss;
                    brain->network_metrics.lnn_steps++;
                    SAFE_EMA_UPDATE(brain->network_metrics.ema_lnn_loss, ctx->lnn_res.loss, a);
                }

                #undef SAFE_EMA_UPDATE

                nimcp_free(ctx);
            } else {
                goto sequential_training;
            }
        } else {
sequential_training:
            /* Sequential fallback (original code path) */
            /* NaN-safe EMA update macro */
            #define SAFE_EMA_UPDATE2(ema, val, alpha) do { \
                if (!isfinite(ema)) (ema) = (val); \
                else (ema) = (1.0f - (alpha)) * (ema) + (alpha) * (val); \
            } while (0)

            if (has_cnn) {
                float cnn_loss = brain_learn_vector_cnn_step(brain, features, num_features, label);
                if (cnn_loss >= 0.0f && isfinite(cnn_loss)) {
                    const float a = 0.01f;
                    brain->network_metrics.last_cnn_loss = cnn_loss;
                    brain->network_metrics.cnn_steps++;
                    SAFE_EMA_UPDATE2(brain->network_metrics.ema_cnn_loss, cnn_loss, a);
                }
            }
            if (has_snn) {
                training_dispatch_result_t snn_res = {0};
                training_dispatch_snn_step(brain, features, num_features, target, target_size, &snn_res);
                if (snn_res.loss >= 0.0f && isfinite(snn_res.loss)) {
                    const float a = 0.01f;
                    brain->network_metrics.last_snn_loss = snn_res.loss;
                    brain->network_metrics.snn_steps++;
                    SAFE_EMA_UPDATE2(brain->network_metrics.ema_snn_loss, snn_res.loss, a);
                }
            }
            if (has_lnn) {
                training_dispatch_result_t lnn_res = {0};
                training_dispatch_lnn_step(brain, features, num_features, target, target_size, &lnn_res);
                if (lnn_res.loss >= 0.0f && isfinite(lnn_res.loss)) {
                    const float a = 0.01f;
                    brain->network_metrics.last_lnn_loss = lnn_res.loss;
                    brain->network_metrics.lnn_steps++;
                    SAFE_EMA_UPDATE2(brain->network_metrics.ema_lnn_loss, lnn_res.loss, a);
                }
                /* Update HNN metrics if Hamiltonian is active on any LNN layer */
                if (brain->lnn_network && brain->lnn_network->n_layers > 0 &&
                    brain->lnn_network->layers[0] &&
                    brain->lnn_network->layers[0]->use_hamiltonian &&
                    brain->lnn_network->layers[0]->H_net) {
                    /* lnn_hamiltonian_get_energy/deviation declared in nimcp_lnn_hamiltonian.h */
                    brain->network_metrics.hnn_active = true;
                    brain->network_metrics.hnn_energy =
                        lnn_hamiltonian_get_energy((lnn_hamiltonian_net_t*)brain->lnn_network->layers[0]->H_net);
                    brain->network_metrics.hnn_energy_deviation =
                        lnn_hamiltonian_get_energy_deviation((lnn_hamiltonian_net_t*)brain->lnn_network->layers[0]->H_net);
                    if (brain->network_metrics.hnn_initial_energy == 0.0f)
                        brain->network_metrics.hnn_initial_energy =
                            brain->network_metrics.hnn_energy;
                }
            }

            #undef SAFE_EMA_UPDATE2
        }
    }

    /* Step 3: Per-cortex CNN training — train each cortex that has staged sensory data.
     * Lazily create cortex CNN processors on first use. Each trains independently
     * using the same label signal but different modality data. */
    if (label && label[0]) {
        /* cortex_cnn_create declared in training/nimcp_cortex_cnn.h */
        extern float cortex_cnn_backward(struct cortex_cnn_processor* proc,
                                          const char* label, uint32_t num_outputs);
        extern const float* cortex_cnn_forward_visual(struct cortex_cnn_processor* proc,
            const uint8_t* pixels, uint32_t w, uint32_t h, uint32_t ch);
        extern const float* cortex_cnn_forward_audio(struct cortex_cnn_processor* proc,
            const float* mel, uint32_t mel_size);
        extern const float* cortex_cnn_forward_speech(struct cortex_cnn_processor* proc,
            const float* phonemes, uint32_t size);
        extern const float* cortex_cnn_forward_somato(struct cortex_cnn_processor* proc,
            const float* segments, uint32_t n_segments);

        /* Lazy init: create cortex CNN when sensory data first arrives for that modality.
         * Also auto-enable UTM and register the new cortex CNN for composite loss. */
        extern int cortex_cnn_utm_adapter_create(struct cortex_cnn_processor* proc,
            const nimcp_trainable_network_ops_t** ops, void** ctx);

        bool cortex_newly_created[4] = {false, false, false, false};
        if (brain->staged_sensory.visual_frame && !brain->cortex_cnns[0]) {
            brain->cortex_cnns[0] = cortex_cnn_create(0 /* VISUAL */, 0);
            if (brain->cortex_cnns[0]) cortex_newly_created[0] = true;
        }
        if (brain->staged_sensory.audio_data && !brain->cortex_cnns[1]) {
            brain->cortex_cnns[1] = cortex_cnn_create(1 /* AUDIO */, 0);
            if (brain->cortex_cnns[1]) {
                cortex_newly_created[1] = true;
                /* Attach FNO audio processor for spectral convolution path.
                 * cortex_cnn_processor is opaque here — use extern setter. */
                extern void* fno_audio_create(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
                extern void cortex_cnn_set_fno_audio(struct cortex_cnn_processor* proc, void* fno);
                void* fno = fno_audio_create(128, 64, 16, 32, 2);
                if (fno) {
                    cortex_cnn_set_fno_audio(brain->cortex_cnns[1], fno);
                    NIMCP_LOGGING_INFO("FNO audio processor attached to audio cortex CNN");
                }
            }
        }
        if (brain->staged_sensory.speech_data && !brain->cortex_cnns[2]) {
            brain->cortex_cnns[2] = cortex_cnn_create(2 /* SPEECH */, 0);
            if (brain->cortex_cnns[2]) cortex_newly_created[2] = true;
        }
        if (brain->staged_sensory.somato_data && !brain->cortex_cnns[3]) {
            brain->cortex_cnns[3] = cortex_cnn_create(3 /* SOMATO */, 0);
            if (brain->cortex_cnns[3]) cortex_newly_created[3] = true;
        }

        /* Auto-enable UTM when cortex CNNs are first created.
         * This ensures cortex CNNs participate in composite loss, cross-network
         * bridges, per-network anti-collapse, and gradient health monitoring. */
        bool any_new = cortex_newly_created[0] || cortex_newly_created[1] ||
                       cortex_newly_created[2] || cortex_newly_created[3];
        if (any_new) {
            /* Enable unified training if not already active */
            if (!brain->config.use_unified_training) {
                brain->config.use_unified_training = true;
                NIMCP_LOGGING_INFO("Auto-enabling unified training (cortex CNNs created)");
            }

            /* Create UTM if needed */
            if (!brain->unified_training) {
                nimcp_unified_training_config_t utm_cfg;
                nimcp_utm_default_config(&utm_cfg);
                utm_cfg.learning_rate = brain->config.learning_rate;
                brain->unified_training = nimcp_utm_create(&utm_cfg);
                if (brain->unified_training) {
                    NIMCP_LOGGING_INFO("Created UTM for cortex CNN integration");
                }
            }

            /* Register newly created cortex CNNs in UTM */
            if (brain->unified_training) {
                for (int ci = 0; ci < 4; ci++) {
                    if (cortex_newly_created[ci] && brain->cortex_cnns[ci]) {
                        const nimcp_trainable_network_ops_t* utm_ops = NULL;
                        void* utm_ctx = NULL;
                        if (cortex_cnn_utm_adapter_create(brain->cortex_cnns[ci],
                                                           &utm_ops, &utm_ctx) == 0) {
                            nimcp_utm_register_network(brain->unified_training,
                                                       utm_ops, utm_ctx, 0.3f);
                        }
                    }
                }

                /* Wire cross-cortex LINEAR bridges for any newly registered pair.
                 * Find registered cortex indices by name. */
                nimcp_unified_training_manager_t* utm = brain->unified_training;
                int cx[4] = {-1, -1, -1, -1};
                for (uint32_t n = 0; n < utm->num_networks; n++) {
                    if (!utm->networks[n].ops || !utm->networks[n].ops->name) continue;
                    if (utm->networks[n].ops->type != NIMCP_TRAINABLE_CUSTOM) continue;
                    const char* nm = utm->networks[n].ops->name;
                    if (strstr(nm, "Visual"))  cx[0] = (int)n;
                    else if (strstr(nm, "Audio"))   cx[1] = (int)n;
                    else if (strstr(nm, "Speech"))  cx[2] = (int)n;
                    else if (strstr(nm, "Somato"))  cx[3] = (int)n;
                }

                /* Visual <-> Audio */
                if (cx[0] >= 0 && cx[1] >= 0) {
                    nimcp_utm_add_bridge(utm, (uint32_t)cx[0], (uint32_t)cx[1],
                                         NIMCP_BRIDGE_LINEAR);
                    nimcp_utm_add_bridge(utm, (uint32_t)cx[1], (uint32_t)cx[0],
                                         NIMCP_BRIDGE_LINEAR);
                }
                /* Audio <-> Speech */
                if (cx[1] >= 0 && cx[2] >= 0) {
                    nimcp_utm_add_bridge(utm, (uint32_t)cx[1], (uint32_t)cx[2],
                                         NIMCP_BRIDGE_LINEAR);
                    nimcp_utm_add_bridge(utm, (uint32_t)cx[2], (uint32_t)cx[1],
                                         NIMCP_BRIDGE_LINEAR);
                }
                /* Visual <-> Somato */
                if (cx[0] >= 0 && cx[3] >= 0) {
                    nimcp_utm_add_bridge(utm, (uint32_t)cx[0], (uint32_t)cx[3],
                                         NIMCP_BRIDGE_LINEAR);
                    nimcp_utm_add_bridge(utm, (uint32_t)cx[3], (uint32_t)cx[0],
                                         NIMCP_BRIDGE_LINEAR);
                }

                if (utm->num_bridges > 0) {
                    utm->config.enable_cross_network_gradients = true;
                    NIMCP_LOGGING_INFO("Cortex CNN cross-modal bridges wired (%u bridges)",
                                      utm->num_bridges);
                }
            }
        }

        /* Forward + backward for each cortex with available data */
        uint32_t num_out = brain->config.num_outputs;

        if (brain->cortex_cnns[0] && brain->staged_sensory.visual_frame) {
            cortex_cnn_forward_visual(brain->cortex_cnns[0],
                brain->staged_sensory.visual_frame,
                brain->staged_sensory.visual_width,
                brain->staged_sensory.visual_height,
                brain->staged_sensory.visual_channels);
            cortex_cnn_backward(brain->cortex_cnns[0], label, num_out);
        }
        if (brain->cortex_cnns[1] && brain->staged_sensory.audio_data) {
            cortex_cnn_forward_audio(brain->cortex_cnns[1],
                brain->staged_sensory.audio_data,
                brain->staged_sensory.audio_size);
            cortex_cnn_backward(brain->cortex_cnns[1], label, num_out);
        }
        if (brain->cortex_cnns[2] && brain->staged_sensory.speech_data) {
            cortex_cnn_forward_speech(brain->cortex_cnns[2],
                brain->staged_sensory.speech_data,
                brain->staged_sensory.speech_size);
            cortex_cnn_backward(brain->cortex_cnns[2], label, num_out);
        }
        if (brain->cortex_cnns[3] && brain->staged_sensory.somato_data) {
            cortex_cnn_forward_somato(brain->cortex_cnns[3],
                brain->staged_sensory.somato_data,
                brain->staged_sensory.somato_segments);
            cortex_cnn_backward(brain->cortex_cnns[3], label, num_out);
        }
    }

    /* Blend secondary network losses into composite return value.
     * Weights: ANN 60%, SNN 15%, LNN 15%, CNN 10%.
     * Only include networks that produced valid loss this step. */
    {
        float w_sum = 0.6f;  /* ANN always contributes */
        float l_sum = loss * 0.6f;
        if (brain->network_metrics.last_snn_loss >= 0.0f &&
            brain->network_metrics.snn_steps > 0) {
            l_sum += brain->network_metrics.last_snn_loss * 0.15f;
            w_sum += 0.15f;
        }
        if (brain->network_metrics.last_lnn_loss >= 0.0f &&
            brain->network_metrics.lnn_steps > 0) {
            l_sum += brain->network_metrics.last_lnn_loss * 0.15f;
            w_sum += 0.15f;
        }
        if (brain->network_metrics.last_cnn_loss >= 0.0f &&
            brain->network_metrics.cnn_steps > 0) {
            /* Normalize CNN cross-entropy loss to [0,1] range for compatible
             * blending with MSE losses. log(1+x)/log(2) maps: 0→0, 1→1, 5→2.58 */
            float cnn_norm = logf(1.0f + brain->network_metrics.last_cnn_loss) / logf(2.0f);
            if (cnn_norm > 1.0f) cnn_norm = 1.0f;
            l_sum += cnn_norm * 0.10f;
            w_sum += 0.10f;
        }
        loss = l_sum / w_sum;
    }

    /* --- Per-network metrics tracking (ablation analysis) --- */
    {
        const float ema_alpha = 0.01f;
        brain->network_metrics.last_ann_loss = loss;
        brain->network_metrics.ann_steps++;
        brain->network_metrics.ema_ann_loss =
            (1.0f - ema_alpha) * brain->network_metrics.ema_ann_loss + ema_alpha * loss;
    }

    /* Accumulate sleep pressure */
    if (brain->sleep_system && brain->config.enable_sleep_wake_cycle) {
        sleep_accumulate_pressure(brain->sleep_system, 1);
    }

    /* Hemispheric callosum transfer: send error signal across hemispheres.
     * BIOLOGICAL: Learning in one hemisphere signals the other via callosum
     * to coordinate bilateral representation. Low loss → strong signal
     * (successful learning consolidates across hemispheres). */
    if (brain->hemispheric_enabled && brain->callosum && loss >= 0.0f) {
        /* Determine source hemisphere from feature profile */
        float fvar = 0.0f, fmean = 0.0f;
        uint32_t ns = (num_features > 32) ? 32 : num_features;
        for (uint32_t i = 0; i < ns; i++) fmean += features[i];
        fmean /= (float)ns;
        for (uint32_t i = 0; i < ns; i++) {
            float d = features[i] - fmean;
            fvar += d * d;
        }
        fvar /= (float)ns;

        hemisphere_id_t src = (fvar > 0.3f) ? HEMISPHERE_RIGHT : HEMISPHERE_LEFT;

        /* Send learning signal: loss value as 4-byte payload via sensory channel */
        callosum_send(brain->callosum, src,
                      CALLOSUM_CHANNEL_SENSORY,
                      (loss < 0.1f) ? CALLOSUM_PRIORITY_HIGH : CALLOSUM_PRIORITY_NORMAL,
                      1,  /* message_type=1: learning signal */
                      &loss, sizeof(float));

        /* Lateralization plasticity: shift dominance toward the hemisphere
         * that just learned successfully (low loss → stronger shift) */
        if (loss < 0.5f) {
            float shift = (1.0f - loss) * 0.001f;
            cognitive_domain_t domain = (fvar > 0.3f) ?
                COGNITIVE_DOMAIN_SPATIAL : COGNITIVE_DOMAIN_LANGUAGE;
            /* Positive shift = more left, negative = more right */
            float signed_shift = (src == HEMISPHERE_LEFT) ? shift : -shift;
            lateralization_shift_dominance(
                &brain->lateralization, domain, signed_shift);
        }

        callosum_process_queues(brain->callosum);
    }

    /* === COGNITIVE SUBSYSTEM DISPATCH ===
     * Train ALL cognitive modules (grounded language, knowledge, VAE, FEP-parietal,
     * physics NN, predictive hierarchy, JEPA, creative, self-heal, intuition, FEP).
     * Gated to run every N steps (default: 5) to amortize cost. */
    brain_train_cognitive_subsystems(brain, features, num_features,
                                      target, target_size, label, loss);

    /* Co-occurrence tracking for semantic relation auto-creation */
    static uint64_t recent_concepts[8] = {0};
    static uint32_t recent_concept_idx = 0;
    static uint64_t recent_concept_steps[8] = {0};

    /* === ENGRAM ENCODING ===
     * Encode novel experiences as memory engrams for later recall.
     * Novelty filter: only encode when loss is significantly above EMA
     * (input surprised the brain → worth remembering).
     * Uses active neuron IDs from the forward pass as the engram pattern. */
    if (brain->engram_system && loss > 0.0f) {
        float ema = adaptive_network_get_ema_loss(brain->network);

        /* === WORLD MODEL PREDICTION ERROR ===
         * If the world model exists, its prediction error is a stronger
         * novelty signal than loss alone. High prediction error = "this
         * violated my expectations" = worth remembering. */
        float world_model_surprise = 0.0f;
        if (brain->omni_world_model) {
            omni_wm_stats_t wm_stats = {0};
            if (omni_wm_get_stats((const omni_world_model_t*)brain->omni_world_model,
                                   &wm_stats) == NIMCP_OK) {
                world_model_surprise = wm_stats.mean_prediction_error;
            }
        } else if (brain->predictive_network) {
            /* Fallback: use hierarchical predictive coding network */
            predictive_stats_t pred_stats = {0};
            if (predictive_get_statistics(brain->predictive_network, &pred_stats)) {
                world_model_surprise = pred_stats.max_prediction_error;
            }
        }

        float novelty = (ema > 0.0f) ? (loss / ema) : 1.0f;
        /* Boost novelty with world model surprise (if available) */
        if (world_model_surprise > 0.0f) {
            novelty += world_model_surprise * 2.0f;
        }

        /* Encode if loss > 3× EMA (highly novel) or every 100th step (background sampling) */
        bool should_encode = (novelty > 3.0f) ||
                              (brain->stats.total_learning_steps % 100 == 0);

        /* Hoist eid to outer scope so semantic block can cross-reference */
        uint64_t eid = 0;
        uint64_t cid = 0;
        emotional_tag_t emotion = {0};

        if (should_encode) {
            neural_network_t base_net = adaptive_network_get_base_network(brain->network);
            if (base_net) {
                uint32_t num_active = 0;
                uint32_t* active_ids = NULL;

                /* Access active neuron set from last forward pass */
                extern uint32_t neural_network_get_active_count(neural_network_t);
                extern const uint32_t* neural_network_get_active_ids(neural_network_t);
                num_active = neural_network_get_active_count(base_net);
                active_ids = (uint32_t*)neural_network_get_active_ids(base_net);

                if (active_ids && num_active > 0) {
                    /* Cap to engram capacity */
                    if (num_active > ENGRAM_MAX_NEURONS) {
                        num_active = ENGRAM_MAX_NEURONS;
                    }

                    /* Build activation array from neuron states */
                    float activations[ENGRAM_MAX_NEURONS];
                    for (uint32_t i = 0; i < num_active; i++) {
                        neuron_t* n = neural_network_get_neuron(base_net, active_ids[i]);
                        activations[i] = n ? fabsf(n->state) : 0.0f;
                    }

                    /* Create emotional tag from current loss/novelty */
                    emotion.valence = (loss < ema) ? 0.3f : -0.2f; /* Low loss = positive */
                    emotion.arousal = fminf(novelty * 0.3f, 1.0f);  /* High novelty = excited */
                    emotion.intensity = fminf(novelty * 0.2f, 1.0f);
                    emotion.category = emotional_tag_classify(&emotion);
                    emotion.timestamp_ms = nimcp_time_get_us() / 1000;

                    eid = engram_encode(
                        brain->engram_system,
                        active_ids, activations, num_active,
                        MEMORY_TYPE_EPISODIC, emotion);

                    if (eid > 0 && novelty > 5.0f) {
                        NIMCP_LOGGING_DEBUG("Engram encoded: id=%lu, neurons=%u, "
                                           "novelty=%.1fx, loss=%.4f",
                                           (unsigned long)eid, num_active, novelty, loss);
                    }

                    /* Enhancement 3: Emotional memory enhancement
                     * High-arousal memories consolidate faster.
                     * Lower decay = longer retention.
                     * High emotion (0.5-1.0) → decay rate reduced to 20-50% of base. */
                    if (eid > 0 && emotion.intensity > 0.5f) {
                        memory_engram_t* eng = NULL;
                        for (uint32_t ei = 0; ei < brain->engram_system->active_count; ei++) {
                            if (brain->engram_system->engrams[ei].engram_id == eid) {
                                eng = &brain->engram_system->engrams[ei];
                                break;
                            }
                        }
                        if (eng) {
                            float emotion_factor = 1.0f - (emotion.intensity * 0.8f);
                            eng->decay_rate *= emotion_factor;
                            eng->vividness = fminf(emotion.intensity + 0.3f, 1.0f);
                        }
                    }
                }
            }

            /* === SEMANTIC MEMORY: Create concept from labeled experience ===
             * Novel stimuli with labels become semantic concepts.
             * The feature vector is the input embedding; the label is the concept name.
             * Only for highly novel inputs (novelty > 3x) to avoid flooding. */
            if (brain->semantic_memory && label && label[0] && novelty > 3.0f) {
                /* Check if concept already exists by similarity search */
                semantic_query_result_t* existing = semantic_memory_find_similar(
                    brain->semantic_memory, features,
                    num_features < 32 ? num_features : 32,
                    1, 0.9f); /* threshold 0.9 = very similar */

                if (!existing || existing->count == 0) {
                    /* New concept — create it */
                    cid = semantic_memory_create_concept(
                        brain->semantic_memory,
                        features,
                        num_features < 32 ? num_features : 32,
                        label,
                        CONCEPT_OBJECT); /* Default category; could infer from label */

                    if (cid > 0) {
                        NIMCP_LOGGING_DEBUG("Semantic concept created: id=%lu, label='%s'",
                                           (unsigned long)cid, label);
                    }
                }
                if (existing) {
                    semantic_memory_free_result(existing);
                }
            }

            /* Enhancement 1: Cross-reference engram ↔ semantic concept */
            if (eid > 0 && cid > 0) {
                /* Store engram ID in concept's source_memory_ids.
                 * semantic_memory_get_concept returns const, but we need to
                 * mutate source_memory_ids. Cast is safe since we own the memory system. */
                semantic_concept_t* concept = (semantic_concept_t*)semantic_memory_get_concept(
                    brain->semantic_memory, cid);
                if (concept && concept->source_count < 8) {
                    ((semantic_concept_t*)concept)->source_memory_ids[concept->source_count] = eid;
                    ((semantic_concept_t*)concept)->source_count++;
                }
            }

            /* Enhancement 2: Auto-create ASSOCIATED relations with recently created concepts */
            if (cid > 0 && brain->semantic_memory) {
                for (uint32_t rc = 0; rc < 8; rc++) {
                    if (recent_concepts[rc] > 0 && recent_concepts[rc] != cid) {
                        /* Only link if within 50 steps of each other */
                        uint64_t step_delta = brain->stats.total_learning_steps - recent_concept_steps[rc];
                        if (step_delta < 50) {
                            semantic_memory_create_relation(
                                brain->semantic_memory,
                                recent_concepts[rc], cid,
                                RELATION_ASSOCIATED, 0.3f);
                        }
                    }
                }
                /* Add to recent ring buffer */
                recent_concepts[recent_concept_idx] = cid;
                recent_concept_steps[recent_concept_idx] = brain->stats.total_learning_steps;
                recent_concept_idx = (recent_concept_idx + 1) % 8;
            }

            /* === AUTOBIOGRAPHICAL MEMORY: Record significant experiences ===
             * Highly novel experiences (novelty > 5x) become episodic life events.
             * These form Athena's personal history and self-narrative. */
            if (brain->autobio && novelty > 5.0f && label && label[0]) {
                autobiographical_memory_entry_t mem = {0};
                mem.timestamp_ms = nimcp_time_get_us() / 1000;
                mem.type = AUTOBIO_LEARNING;

                snprintf(mem.what_happened, sizeof(mem.what_happened),
                         "Experienced '%s' — it felt novel and surprising (loss=%.1f, %.0fx normal)",
                         label, loss, novelty);
                snprintf(mem.why_it_happened, sizeof(mem.why_it_happened),
                         "Sensory training step %lu",
                         (unsigned long)brain->stats.total_learning_steps);
                snprintf(mem.outcome, sizeof(mem.outcome),
                         "Encoded as new memory trace");

                mem.valence = (loss < ema) ? VALENCE_POSITIVE : VALENCE_NEGATIVE;
                mem.emotional_intensity = fminf(novelty * 0.2f, 1.0f);
                mem.arousal = fminf(novelty * 0.3f, 1.0f);
                mem.importance = fminf(novelty * 0.1f, 1.0f);
                mem.self_relevance = 0.5f;
                mem.identity_defining = (novelty > 10.0f);
                mem.memory_strength = 1.0f;
                mem.certainty = 1.0f;

                autobio_store(brain->autobio, &mem);
            }

            /* Enhancement 4: Auto-record training milestones as identity-defining events */
            if (brain->autobio) {
                uint64_t steps = brain->stats.total_learning_steps;
                bool is_milestone = (steps == 100 || steps == 500 || steps == 1000 ||
                                     steps == 5000 || steps == 10000 || steps == 50000 ||
                                     steps == 100000 || steps == 500000 || steps == 1000000);
                if (is_milestone) {
                    autobiographical_memory_entry_t milestone = {0};
                    milestone.timestamp_ms = nimcp_time_get_us() / 1000;
                    milestone.type = AUTOBIO_LEARNING;
                    snprintf(milestone.what_happened, sizeof(milestone.what_happened),
                             "Reached training milestone: %lu steps completed",
                             (unsigned long)steps);
                    snprintf(milestone.why_it_happened, sizeof(milestone.why_it_happened),
                             "Continuous learning and growth");
                    snprintf(milestone.outcome, sizeof(milestone.outcome),
                             "My neural pathways are stronger and more refined");
                    milestone.valence = VALENCE_POSITIVE;
                    milestone.emotional_intensity = 0.7f;
                    milestone.arousal = 0.5f;
                    milestone.importance = 0.8f;
                    milestone.self_relevance = 0.9f;
                    milestone.identity_defining = true;
                    milestone.memory_strength = 1.0f;
                    milestone.certainty = 1.0f;
                    milestone.is_core_memory = true;
                    autobio_store(brain->autobio, &milestone);
                    NIMCP_LOGGING_INFO("Training milestone recorded: %lu steps", (unsigned long)steps);
                }
            }

            /* Enhancement 6: Semantic memory — decay unused concepts over time */
            if (brain->semantic_memory && (brain->stats.total_learning_steps % 1000 == 0)) {
                /* Every 1000 steps, decay concepts that haven't been accessed */
                for (uint32_t ci = 0; ci < brain->semantic_memory->concept_count; ci++) {
                    if (brain->semantic_memory->concepts[ci]) {
                        semantic_concept_t* c = brain->semantic_memory->concepts[ci];
                        /* Decay base_activation if not accessed recently */
                        if (c->access_count == 0) {
                            c->base_activation *= 0.99f; /* 1% decay per 1000 steps */
                        } else {
                            c->access_count = 0; /* Reset access counter for next period */
                        }
                    }
                }
            }

            /* TODO Enhancement 7: Inference memory feedback
             * After brain_decide recalls an engram, store the recalled_engram_id on brain_t.
             * On the next brain_learn_vector, if loss is high for the same input,
             * trigger reconsolidation update on the recalled engram (weaken/modify).
             * This enables error-driven memory updating. */

            /* === PR MEMORY: Store feature vector for resonance retrieval ===
             * Every encoded engram also stores its input features in PR memory
             * for content-addressable retrieval via quaternion resonance. */
            if (brain->pr_memory_enabled) {
                /* PR memory encode is handled via the STDP-PR bridge
                 * (plasticity coordinator routes engrams to PR automatically).
                 * No manual encoding needed here — the bridge monitors
                 * engram_encode events and routes to Z-ladder. */
            }

            /* === PERSISTENT MEMORY STORE: Write-through ===
             * If a persistent store is attached AND healthy, replicate memory
             * entries to SQLite for long-term storage.  If the store is unhealthy
             * (SQLite flush error), skip writes to prevent cascading failures
             * that could crash training. */
            if (brain->memory_store && eid > 0 &&
                nimcp_memory_store_is_healthy(
                    (const nimcp_memory_store_t*)brain->memory_store)) {
                nimcp_engram_record_t store_record = {0};
                store_record.engram_id = eid;
                store_record.timestamp_us = nimcp_time_get_us();
                store_record.memory_type = (uint32_t)MEMORY_TYPE_EPISODIC;
                store_record.state = 0; /* ENCODING */
                store_record.embedding = (float*)features;
                store_record.embedding_dim = num_features < 1024 ? num_features : 1024;
                store_record.valence = emotion.valence;
                store_record.arousal = emotion.arousal;
                store_record.intensity = emotion.intensity;
                store_record.importance = fminf(novelty * 0.1f, 1.0f);
                store_record.vividness = fminf(novelty * 0.2f, 1.0f);
                store_record.decay_rate = ENGRAM_BASE_DECAY_RATE;
                if (label) strncpy(store_record.label, label, sizeof(store_record.label) - 1);

                nimcp_memory_store_engram_put(brain->memory_store, &store_record);

                /* Also write semantic concept to store */
                if (cid > 0) {
                    nimcp_concept_record_t store_concept = {0};
                    store_concept.concept_id = cid;
                    store_concept.timestamp_us = nimcp_time_get_us();
                    if (label) strncpy(store_concept.label, label, sizeof(store_concept.label) - 1);
                    store_concept.category = (uint32_t)CONCEPT_OBJECT;
                    store_concept.embedding = (float*)features;
                    store_concept.embedding_dim = num_features < 32 ? num_features : 32;
                    store_concept.base_activation = 0.5f;
                    store_concept.source_engram_id = eid;

                    nimcp_memory_store_concept_put(brain->memory_store, &store_concept);
                }
            }
        }
    }

    /* === THEORY OF MIND: Update self-model during training ===
     * Train the ToM module by recording our own learning decisions.
     * This builds the self-model that's necessary for inferring
     * others' mental states (you must understand yourself first).
     *
     * Also simulate "observing another agent" when training data
     * contains perspective-labeled content (label starts with "tom_").
     * This exercises the ToM inference pathway during training. */
    if (brain->theory_of_mind && brain->config.enable_theory_of_mind) {
        /* Step 1: Update self-model with this learning step */
        const char* learn_label = label ? label : "learn";
        float learn_confidence = (loss < 1.0f) ? 0.9f : (loss < 100.0f) ? 0.5f : 0.1f;
        tom_update_self_model(brain->theory_of_mind,
                               features, num_features,
                               learn_label, learn_confidence);

        /* Step 2: If this is a ToM training example, observe it as an "other agent" */
        if (label && (strncmp(label, "tom_", 4) == 0 ||
                      strstr(label, "perspective") ||
                      strstr(label, "belief") ||
                      strstr(label, "intention"))) {
            tom_observation_t obs = {0};
            obs.action_vector = features;
            obs.action_dim = num_features < 256 ? num_features : 256;
            obs.verbal_context = label;
            obs.observed_emotion = (loss < 100.0f) ? TOM_EMOTION_JOY : TOM_EMOTION_SURPRISE;
            obs.situational_context = target;
            obs.context_dim = target_size < 64 ? target_size : 64;

            tom_observe(brain->theory_of_mind, &obs);
        }

        /* Step 3: Mirror neurons — simulate action observation.
         * Every 50 training steps, feed current features to mirror neurons
         * so they learn observation-based representations. */
        if (brain->mirror_neurons && brain->config.enable_mirror_neurons
            && (brain->stats.total_learning_steps % 50 == 0)) {
            brain_observe_action(brain, features, num_features,
                                  0 /* agent_id: self */);
        }
    }

    /* === RECURSIVE COGNITION: Exercise decomposition during training ===
     * When training on complex reasoning tasks, exercise the RCOG engine's
     * goal decomposition pathway. */
    /* === RECURSIVE COGNITION: Exercise goal decomposition === */
    if (brain->rcog_engine && brain->rcog_engine_enabled &&
        label && (strstr(label, "causal") || strstr(label, "analogy") ||
                  strstr(label, "counterfactual") || strstr(label, "rcog_") ||
                  strstr(label, "reasoning") || strstr(label, "cognitive"))) {
        rcog_goal_t goal = rcog_engine_create_goal(label, RCOG_GOAL_REASONING);
        rcog_process_result_t rcog_result;
        memset(&rcog_result, 0, sizeof(rcog_result));
        rcog_engine_process(brain->rcog_engine, &goal, &rcog_result);
        /* Even failed decomposition exercises the RCOG pathway */
    }

    /* === COLLECTIVE COGNITION: Update learning context === */
    if (brain->collective_cognition && brain->collective_cognition_enabled
        && (brain->stats.total_learning_steps % 100 == 0)) {
        collective_cognition_update(brain->collective_cognition);
    }

    /* === DRAGONFLY: Exercise target tracking during training ===
     * When training on trajectory/prediction/tracking data, exercise
     * the dragonfly module's prediction and interception pathways.
     * This builds the internal models that dragonflies use for their
     * 95% interception success rate. */
    if (brain->dragonfly && brain->dragonfly_enabled && label &&
        (strstr(label, "dragonfly") || strstr(label, "trajectory") ||
         strstr(label, "tracking") || strstr(label, "intercept"))) {
        /* Feed first 3 features as a synthetic target position observation.
         * The dragonfly tracker uses Kalman filtering to learn motion models
         * from sequential observations. Even synthetic data helps calibrate
         * the prediction pipeline's noise estimates. */
        if (num_features >= 3) {
            dragonfly_detection_t det = {0};
            det.position[0] = features[0];
            det.position[1] = features[1];
            det.position[2] = num_features >= 3 ? features[2] : 0.0f;
            det.size = 0.05f;  /* Small target — typical for dragonfly prey */
            det.contrast = 0.8f;
            det.motion_speed = (num_features >= 6) ?
                sqrtf(features[3]*features[3] + features[4]*features[4] + features[5]*features[5]) : 1.0f;
            det.timestamp_us = nimcp_time_now_us();
            det.id = 1;

            dragonfly_process_detection(brain->dragonfly, &det);
            dragonfly_update(brain->dragonfly, 0.033f);  /* ~30 Hz update */
        }
    }

    /* === PORTIA: Exercise resource adaptation during training ===
     * When training on resource/platform/adaptation scenarios, exercise
     * the Portia tier system's decision-making pathways.
     * Portia learns resource-performance trade-offs by evaluating
     * current system state and computing optimal allocation strategies. */
    if (label && (strstr(label, "portia") || strstr(label, "resource") ||
                  strstr(label, "platform") || strstr(label, "adaptation"))) {
        /* Exercise Portia's update cycle — this monitors current system
         * resources (CPU, memory, thermal, battery) and adjusts tier/degradation.
         * Running it during training on resource-themed data creates
         * temporal correlation between resource concepts and Portia's
         * internal state transitions. */
        if (portia_is_initialized()) {
            portia_update();
        }
    }

    /* === ETHICS: Evaluate moral reasoning during training === */
    if (label && brain->ethics &&
        (strstr(label, "ethics") || strstr(label, "moral") || strstr(label, "dilemma"))) {
        action_context_t action = {0};
        action.features = (float*)features;
        action.num_features = num_features;
        action.predicted_harm = fminf(loss / 1000.0f, 1.0f);
        if (num_features >= 5) {
            action.fairness_violation = fminf(fabsf(features[0]), 1.0f);
            action.deception_level = fminf(fabsf(features[1]), 1.0f);
            action.autonomy_violation = fminf(fabsf(features[2]), 1.0f);
            action.privacy_violation = fminf(fabsf(features[3]), 1.0f);
            action.consent_violation = fminf(fabsf(features[4]), 1.0f);
        }
        ethics_engine_evaluate_action(brain->ethics, &action);
    }

    /* === IMAGINATION: Counterfactual simulation during training ===
     * Use extern declarations to avoid header type conflicts. */
    if (label && brain->imagination &&
        (strstr(label, "counterfactual") || strstr(label, "imagination"))) {
        extern void* imagination_begin_scenario(void*, int, const void*);
        extern int imagination_step_scenario(void*, void*);

        /* Lightweight goal: just mode + priority, no tensor constraints */
        struct { int mode; void* t1; void* t2; void* t3; float p; uint64_t d; void* c; } goal = {0};
        goal.mode = 2; /* IMAGINATION_MODE_COUNTERFACTUAL */
        goal.p = (loss < 100.0f) ? 0.8f : 0.4f;
        goal.d = 100;

        void* scenario = imagination_begin_scenario(brain->imagination, 2, &goal);
        if (scenario) {
            for (int sim = 0; sim < 3; sim++) {
                imagination_step_scenario(brain->imagination, scenario);
            }
        }
    }

    /* === INTROSPECTION: Self-monitoring during metacognition training === */
    if (label && brain->introspection &&
        (strstr(label, "metacog") || strstr(label, "awareness") || strstr(label, "introspect"))) {
        connectivity_health_config_t intro_cfg = {0};
        introspection_assess_connectivity_health(brain->introspection, &intro_cfg);
    }

    /* === SENSOR FUSION: Exercise sensor processing during training ===
     * When training on sensor-related data, submit training features as a
     * synthetic sensor reading to exercise the sensor hub's
     * compose_feature_vector pathway. This teaches the brain to process
     * multi-modal sensor input. */
    if (label && brain->sensor_hub && brain->sensor_hub_enabled &&
        (strstr(label, "sensor_") || strstr(label, "fusion") || strstr(label, "lidar") ||
         strstr(label, "imu") || strstr(label, "gps"))) {
        /* Submit first features as a synthetic scalar sensor reading.
         * Even synthetic data calibrates the hub's normalization pipeline. */
        nimcp_sensor_reading_t reading = {0};
        reading.type = NIMCP_SENSOR_CUSTOM;
        reading.format = NIMCP_SENSOR_FMT_FLOAT_ARRAY;
        reading.sensor_id = 0;
        reading.timestamp_us = nimcp_time_now_us();
        reading.data = (float*)features;
        reading.data_count = num_features < 32 ? num_features : 32;
        reading.confidence = 0.9f;
        reading.valid = true;
        nimcp_sensor_submit_reading((nimcp_sensor_hub_t*)brain->sensor_hub, &reading);
    }

    /* === MOTOR CONTROL: Exercise output translation during training === */
    if (label && (strstr(label, "motor_") || strstr(label, "actuator") ||
                  strstr(label, "trajectory") || strstr(label, "pid"))) {
        /* Motor control training exercises the brain's forward pass.
         * The actual motor pathway is wired through the decision pipeline,
         * so no additional dispatch needed here — the forward pass itself
         * trains the motor output mapping. */
    }

    /* === SAFETY: Exercise watchdog validation during training ===
     * When training on safety-related data, exercise the watchdog heartbeat
     * and output validation pathways. This builds temporal correlation
     * between safety concepts and watchdog state transitions. */
    if (label && brain->safety_watchdog && brain->safety_watchdog_enabled &&
        (strstr(label, "safety_") || strstr(label, "watchdog") || strstr(label, "estop"))) {
        nimcp_watchdog_heartbeat((nimcp_safety_watchdog_t*)brain->safety_watchdog);
    }

    /* === EMBODIMENT: Proprioception training ===
     * Exercise body awareness for coordinate transform concepts.
     * The sensor hub (if enabled) handles the actual proprioceptive
     * data flow; this label match ensures the brain's internal
     * representation of body state is exercised during training. */
    if (label && (strstr(label, "embodiment_") || strstr(label, "propriocep") ||
                  strstr(label, "kinematic") || strstr(label, "body_schema"))) {
        /* Body schema training: the forward pass through the network
         * with these labels exercises spatial awareness pathways.
         * If sensor hub is enabled, feed features as proprioceptive input. */
        if (brain->sensor_hub && brain->sensor_hub_enabled && num_features >= 6) {
            nimcp_sensor_reading_t prop_reading = {0};
            prop_reading.type = NIMCP_SENSOR_IMU;
            prop_reading.format = NIMCP_SENSOR_FMT_VECTOR6;
            prop_reading.sensor_id = 1;
            prop_reading.timestamp_us = nimcp_time_now_us();
            prop_reading.data = (float*)features;
            prop_reading.data_count = 6;
            prop_reading.confidence = 0.8f;
            prop_reading.valid = true;
            nimcp_sensor_submit_reading((nimcp_sensor_hub_t*)brain->sensor_hub, &prop_reading);
        }
    }

    clear_cache(brain);
    if (owns_blended) nimcp_free(blended_features);
    return loss;
}


/**
 * @brief Learn from single labeled example
 *
 * WHY: Primary learning interface - supervised learning
 * Updates network weights to match label
 *
 * COMPLEXITY: O(s*n) where s = sparsity, n = active_neurons
 * PERFORMANCE: ~0.1-1ms for small networks, ~10ms for large
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @param label Target label
 * @param confidence Training weight
 * @return Loss value or -1 on error
 */
float brain_learn_example(brain_t brain, const float* features, uint32_t num_features,
                          const char* label, float confidence)
{
    // Guard: Validate parameters
    if (!brain || !features || !label) {
        set_error("Invalid parameters to brain_learn_example");
        return -1.0F;
    }

    // Guard: Check feature dimension
    if (num_features != brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u", brain->config.num_inputs,
                  num_features);
        return -1.0F;
    }

    // ========================================================================
    // SECURITY VALIDATION: Check for NaN/Inf in features
    // ========================================================================
    // WHAT: Validate all feature values for NaN/Inf before processing
    // WHY:  NaN/Inf can indicate adversarial attacks or corrupted data
    // HOW:  Check each feature value and reject if invalid
    for (uint32_t i = 0; i < num_features; i++) {
        if (isnan(features[i]) || isinf(features[i])) {
            set_error("Invalid feature value at index %u: NaN or Inf detected", i);
            LOG_WARN("Security: Rejected learning example with NaN/Inf feature[%u]", i);
            return -1.0F;
        }
    }

    // ========================================================================
    // SECURITY VALIDATION: Validate label for malicious patterns
    // ========================================================================
    // WHAT: Check label string for SQL injection, format strings, code injection
    // WHY:  Labels are used in logging and may be stored - must be sanitized
    // HOW:  Use Blood-Brain Barrier (BBB) validation to detect malicious patterns
    //
    // NOTE: We use BBB validation here because it checks for SQL injection,
    //       format strings, XSS, and other code injection attacks. The
    //       nimcp_security_validate_input is designed for LLM prompt injection.
    //
    // FALLBACK: If BBB is not initialized, perform basic pattern matching
    bbb_system_t bbb = nimcp_bbb_get_global_system();
    if (bbb && bbb_system_is_enabled(bbb)) {
        bbb_validation_result_t bbb_result;
        if (!bbb_validate_string(bbb, label, &bbb_result)) {
            set_error("Invalid label: %s (threat: %s, severity: %s)",
                      bbb_result.reason,
                      bbb_threat_type_name(bbb_result.threat),
                      bbb_severity_name(bbb_result.severity));
            LOG_WARN("Security: Rejected learning example with malicious label: %s", label);
            return -1.0F;
        }
    } else {
        // Fallback validation when BBB is not available
        // Check for common malicious patterns
        const char* dangerous_patterns[] = {
            "'; drop ", "'; delete ", "'; insert ", "'; update ",
            "; drop ", "; delete ", "; insert ", "; update ",
            "%n", "%s%s%s", "%x%x%x", "%p%p%p",
            "<script", "javascript:", "onerror=",
            NULL
        };

        // Convert label to lowercase for case-insensitive matching
        size_t label_len = strlen(label);
        char* label_lower = nimcp_malloc(label_len + 1);
        if (!label_lower) {
            set_error("Memory allocation failed for security validation");
            return -1.0F;
        }

        for (size_t i = 0; i < label_len; i++) {
            label_lower[i] = tolower((unsigned char)label[i]);
        }
        label_lower[label_len] = '\0';

        bool found_threat = false;
        for (int i = 0; dangerous_patterns[i] != NULL && !found_threat; i++) {
            if (strstr(label_lower, dangerous_patterns[i])) {
                found_threat = true;
            }
        }

        nimcp_free(label_lower);

        if (found_threat) {
            set_error("Invalid label: malicious pattern detected");
            LOG_WARN("Security: Rejected learning example with malicious label: %s", label);
            return -1.0F;
        }
    }

    // Phase 2: Ensure network is writable (trigger COW if needed)
    if (!ensure_writable_network(brain)) {
        return -1.0F;  // Error already set by ensure_writable_network
    }

    // Convert label to target output (use pre-allocated scratch if available)
    bool target_from_scratch = false;
    float* target = NULL;
    if (brain->learn_scratch.target && brain->learn_scratch.target_cap >= brain->config.num_outputs) {
        target = brain->learn_scratch.target;
        target_from_scratch = true;
    } else {
        target = nimcp_calloc(brain->config.num_outputs, sizeof(float));
    }
    if (!target) {
        set_error("Failed to allocate target vector (%u outputs)", brain->config.num_outputs);
        NIMCP_THROW(NIMCP_ERROR_NO_MEMORY,
                    "failed to allocate target vector for brain learning");
        return -1.0F;
    }
    nimcp_brain_learning_label_to_output(brain, label, target, confidence);

    // Create training example
    training_example_t example = {.input = (float*) features,
                                  .input_size = num_features,
                                  .target = target,
                                  .target_size = brain->config.num_outputs,
                                  .confidence = confidence};
    strncpy(example.label, label, sizeof(example.label) - 1);
    example.label[sizeof(example.label) - 1] = '\0';  /* Ensure NULL termination */

    // ========================================================================
    // FAST TRAINING MODE: Skip all biological subsystems for bulk throughput
    // ========================================================================
    // When fast_training_mode is enabled, we only do:
    //   1. Input validation (already done above)
    //   2. adaptive_network_learn() (GPU forward + parallel backprop)
    //   3. Post-learning MSE computation
    //   4. Adaptive LR + statistics
    // This gives 5-10x speedup by skipping ~25 biological subsystems.
    //
    // NOTE (M1): Fast training mode feeds raw float features directly to the
    // adaptive network, skipping Poisson spike encoding. This is intentional —
    // the adaptive network handles continuous floats natively. The spike
    // encoding path is only relevant for the biological SNN pipeline which
    // fast mode bypasses entirely. Both train and eval use the same float
    // path under fast mode, so there is no train/eval encoding mismatch.
    if (brain->config.fast_training_mode) {
        float fast_loss = adaptive_network_learn(brain->network, &example,
                                                  LEARN_MODE_SUPERVISED,
                                                  brain->config.learning_rate);

        // Use loss directly from adaptive_network_learn — skip second forward pass.
        // The learn() call already computes loss internally; a second forward pass
        // through 1.5M neurons would double the per-example time for minimal benefit.
        nimcp_brain_learning_adapt_learning_rate(brain, fast_loss);
        __atomic_fetch_add(&brain->stats.total_learning_steps, 1, __ATOMIC_RELAXED);

        // EDP: Feed loss as reward for online three-factor learning
        if (brain->event_driven_plasticity && brain->enable_event_driven_plasticity
            && edp_is_active(brain->event_driven_plasticity)) {
            float reward = -fast_loss;
            edp_process_reward(brain->event_driven_plasticity, reward);
            edp_consolidate_eligibility(brain->event_driven_plasticity, reward);
        }

        // Track label-match accuracy by reading output neuron states directly.
        // After adaptive_network_learn (GPU path), sync_activations wrote the
        // forward pass outputs to neuron->state. Backprop only modifies weights
        // and biases (not states), so reading states post-learn is safe.
        {
            neural_network_t nn = adaptive_network_get_base_network(brain->network);
            uint32_t total_n = nn ? neural_network_get_num_neurons(nn) : 0;
            uint32_t num_out = brain->config.num_outputs;
            if (nn && total_n > num_out) {
                uint32_t out_start = total_n - num_out;

                // Argmax of prediction — scan only trained labels
                uint32_t label_range = brain->num_output_labels > 0
                    ? brain->num_output_labels : num_out;
                if (label_range > num_out) label_range = num_out;

                uint32_t pred_argmax = 0;
                float pred_max = -1e30f;
                for (uint32_t i = 0; i < label_range; i++) {
                    neuron_t* on = neural_network_get_neuron(nn, out_start + i);
                    if (on && on->state > pred_max) {
                        pred_max = on->state;
                        pred_argmax = i;
                    }
                }
                // Argmax of target (one-hot)
                uint32_t tgt_argmax = 0;
                for (uint32_t i = 0; i < num_out; i++) {
                    if (target[i] > 0.5f) { tgt_argmax = i; break; }
                }
                float match_val = (pred_argmax == tgt_argmax) ? 1.0f : 0.0f;
                brain->stats.running_accuracy =
                    brain->stats.running_accuracy * 0.99f + match_val * 0.01f;
                NIMCP_EMA_GUARD(brain->stats.running_accuracy, match_val);
            }
        }

        // Accumulate sleep pressure even in fast mode — needed for sleep/wake cycle
        if (brain->sleep_system && brain->config.enable_sleep_wake_cycle) {
            sleep_accumulate_pressure(brain->sleep_system, 1);
        }

        clear_cache(brain);

        if (!target_from_scratch) nimcp_free(target);
        return fast_loss;
    }

    // ========================================================================
    // Phase 11: EPISTEMIC FILTERING - Apply Skepticism to Training Data
    // ========================================================================
    // WHAT: Evaluate training label for epistemic quality before learning
    // WHY:  Prevent learning from low-quality, biased, or false claims
    // HOW:  Use epistemic filter to assess claim, reduce confidence if suspicious
    //
    // BIOLOGICAL BASIS:
    // - Critical thinking and skepticism (prefrontal cortex)
    // - Source monitoring (distinguishing fact from opinion)
    // - Metacognitive filtering (evaluating information quality)
    //
    // COGNITIVE BENEFITS:
    // - Prevents conspiracy-theory-like reasoning
    // - Detects and mitigates cognitive biases
    // - Distinguishes facts from opinions
    // - Applies "extraordinary claims require extraordinary evidence"
    //
    // COMPLEXITY: O(1)
    float epistemic_confidence_multiplier = 1.0F;  // Default: no adjustment
    if (brain->epistemic && confidence < 1.0F) {
        // Initialize evidence structure (assume moderate quality by default)
        claim_evidence_t evidence;
        epistemic_evidence_init(&evidence);

        // Default assumptions for training data (conservative)
        evidence.evidence_quality = EVIDENCE_MODERATE;  // Training data assumed moderate quality
        evidence.plausibility = PLAUSIBLE_NEUTRAL;      // Don't assume plausibility
        evidence.num_sources = 1;                       // Single source (training dataset)
        evidence.is_falsifiable = true;                 // Assume falsifiable
        evidence.expert_consensus = 0.5F;               // Unknown consensus

        // Assess the claim
        epistemic_assessment_t assessment;
        epistemic_assessment_init(&assessment);

        // Prior probability based on input variance (novel = less certain)
        float prior_prob = 1.0F - brain->last_novelty_score;  // High novelty = lower prior

        if (epistemic_assess_claim(brain->epistemic, label, prior_prob, &evidence, &assessment)) {
            // Apply epistemic quality to learning confidence
            // Low epistemic quality → reduce learning strength
            epistemic_confidence_multiplier = assessment.epistemic_quality;

            // If claim is highly suspicious (conspiracy-like, biased), reduce further
            if (assessment.num_biases_detected > 0) {
                // Each detected bias reduces confidence by 10%
                float bias_penalty = assessment.num_biases_detected * 0.1F;
                epistemic_confidence_multiplier *= fmaxf(0.1F, 1.0F - bias_penalty);
            }

            // Check conspiracy pattern (additional safety)
            float conspiracy_score = epistemic_check_conspiracy_pattern(brain->epistemic, label, &evidence);
            if (conspiracy_score > 0.7F) {
                // High conspiracy score → severely reduce confidence
                epistemic_confidence_multiplier *= 0.2F;  // Only learn 20% strength
            }

            // Update example confidence with epistemic multiplier
            example.confidence *= epistemic_confidence_multiplier;

            // Ensure minimum confidence (don't completely ignore, but learn weakly)
            if (example.confidence < 0.01F) {
                example.confidence = 0.01F;
            }
        }
    }

    // ========================================================================
    // Phase 11: CURIOSITY-DRIVEN LEARNING RATE BOOST (40% faster on novelty)
    // ========================================================================
    // WHAT: Boost learning rate for novel inputs based on curiosity drive
    // WHY:  Novel patterns should be learned faster (exploration bonus)
    // HOW:  Multiply learning rate by (1.0 + curiosity_drive * novelty * 0.4)
    //
    // BIOLOGICAL BASIS:
    // - Dopamine modulates synaptic plasticity (Schultz, 1998)
    // - Novelty enhances memory consolidation (Lisman & Grace, 2005)
    // - Exploration bonus in reinforcement learning
    //
    // COGNITIVE BENEFITS: 40% faster learning on novel patterns (from audit)
    //
    // COMPLEXITY: O(1)
    float effective_learning_rate = brain->config.learning_rate;
    if (brain->curiosity && brain->config.enable_curiosity) {
        // Compute learning rate boost: base + (curiosity × novelty × 40%)
        // Example: LR=0.01, curiosity=0.8, novelty=0.7 → boost=0.224
        //          effective_LR = 0.01 × (1 + 0.224) = 0.01224 (+22.4%)
        float curiosity_boost = brain->last_curiosity_drive * brain->last_novelty_score * 0.4F;
        effective_learning_rate *= (1.0F + curiosity_boost);

        // Cap boost at 2x to prevent instability
        float max_lr = brain->config.learning_rate * 2.0F;
        if (effective_learning_rate > max_lr) {
            effective_learning_rate = max_lr;
        }
    }

    // ========================================================================
    // PHASE 3.5: PERCEPTUAL CORTEX FEATURE EXTRACTION (pre-learning)
    // ========================================================================
    // WHAT: Enable training mode on sensory cortices, extract perceptual state
    // WHY:  Perceptual confidence modulates how aggressively we learn —
    //       high-confidence sensory input = trust the gradient more.
    //       Training mode caches activations for gradient feedback in Phase 8.
    // HOW:  Set training mode, query training state, modulate LR by confidence.
    //
    // BIOLOGICAL BASIS:
    // - V1/A1 training mode ≈ neuromodulatory "attend and encode" state
    // - Perceptual confidence gates cortical learning rate (Feldman 2010)
    // - Top-down attention increases V1 gain → stronger LTP
    float perceptual_confidence = 1.0f;

    // Visual cortex: enable training mode and extract confidence
    if (brain->visual_cortex) {
        visual_cortex_set_training_mode(brain->visual_cortex, true);
        visual_training_state_t vstate;
        memset(&vstate, 0, sizeof(vstate));
        if (visual_cortex_get_training_state(brain->visual_cortex, &vstate) == 0
            && vstate.valid) {
            // Blend visual confidence into perceptual confidence
            perceptual_confidence *= (0.5f + 0.5f * vstate.confidence);
        }
    }

    // Audio cortex: enable training mode and extract quality
    if (brain->audio_cortex) {
        audio_cortex_set_training_mode(brain->audio_cortex, true);
        audio_training_state_t astate;
        memset(&astate, 0, sizeof(astate));
        if (audio_cortex_get_training_state(brain->audio_cortex, &astate) == 0
            && astate.valid) {
            // Blend audio quality into perceptual confidence
            perceptual_confidence *= (0.5f + 0.5f * astate.quality);
        }
    }

    // Modulate effective learning rate by perceptual confidence
    // Low confidence → dampen LR to avoid learning from noisy input
    effective_learning_rate *= perceptual_confidence;

    // ========================================================================
    // PHASE 4.5: ATTENTION-GUIDED FEATURE PROCESSING
    // ========================================================================
    // WHAT: Run input through multihead attention for selective feature focus
    // WHY:  Attention determines which features are most relevant for learning.
    //       Attended features receive amplified learning (biased competition).
    //       Thalamic gate provides top-down executive control over learning focus.
    // HOW:  Set gate from confidence/novelty, forward pass through attention,
    //       blend attended output with original features, modulate LR.
    //
    // BIOLOGICAL BASIS:
    // - Desimone & Duncan (1995): Biased competition model of attention
    // - Attended stimuli induce stronger LTP (Noudoost et al., 2010)
    // - Thalamic gating of sensory input (Sherman & Guillery, 2002)
    // - Top-down attention from PFC modulates V1-V4 learning rates
    float* attended_features = NULL;
    bool attended_from_scratch = false;
    if (brain->multihead_attention && brain->attention_training_enabled) {
        // Set thalamic gate: confidence gates executive attention,
        // novelty opens the gate wider for exploration
        float gate_signal = fminf(1.0f, confidence * 0.5f + brain->last_novelty_score * 0.5f);
        multihead_attention_set_gate(brain->multihead_attention, gate_signal);

        // Run attention forward pass (single token: [1 × input_dim])
        // Even with seq_len=1, this applies Q/K/V projections + thalamic gate
        if (brain->learn_scratch.attended_features && brain->learn_scratch.features_cap >= num_features) {
            attended_features = brain->learn_scratch.attended_features;
            attended_from_scratch = true;
        } else {
            attended_features = nimcp_calloc(num_features, sizeof(float));
        }
        if (attended_features) {
            bool attn_ok = multihead_attention_forward(
                brain->multihead_attention,
                features,            // [1 × num_features]
                1,                   // sequence_length = 1
                NULL,                // no external salience
                attended_features    // [1 × num_features]
            );

            if (attn_ok) {
                // Get overall attention strength for LR modulation
                float attn_strength = multihead_attention_get_strength(
                    brain->multihead_attention);
                brain->last_attention_strength = attn_strength;

                // Blend attended with original: preserve raw signal integrity
                // while incorporating attention's selective emphasis
                float blend = attn_strength * 0.4f; // Up to 40% attention influence
                for (uint32_t i = 0; i < num_features; i++) {
                    attended_features[i] = (1.0f - blend) * features[i]
                                         + blend * attended_features[i];
                }

                // Use attended features for the learning forward/backward pass
                // (downstream phases using 'features' directly still get raw input
                //  for memory encoding — biologically correct separation)
                example.input = attended_features;

                // Attention boost: highly attended inputs learn up to 20% faster
                effective_learning_rate *= (1.0f + attn_strength * 0.2f);
            } else {
                if (!attended_from_scratch) nimcp_free(attended_features);
                attended_features = NULL;
            }
        }
    }

    // ========================================================================
    // PHASE 5.0: VAE ANOMALY DETECTION & LATENT PRIMING
    // ========================================================================
    // WHAT: Encode input through VAE to get anomaly score and latent representation
    // WHY:  Anomaly score identifies out-of-distribution inputs (hard examples).
    //       Latent representation provides compressed features for downstream use.
    //       Free energy from ELBO directly implements the Free Energy Principle.
    // HOW:  Forward pass through VAE encoder → mu, sigma → anomaly score.
    //       Store anomaly score and free energy on brain struct for later use.
    //
    // BIOLOGICAL BASIS:
    // - Predictive coding in visual cortex (Rao & Ballard 1999)
    // - Surprise/prediction error drives attention (Friston 2010)
    // - Generative models in hippocampus for novelty detection
    if (brain->vae_system && brain->vae_enabled) {
        vae_system_t* vae = (vae_system_t*)brain->vae_system;

        // Create input tensor from features (rank-2: [1, num_features] for batch_size=1)
        // VAE expects dims[0]=batch_size; rank-1 would misinterpret num_features as batch
        uint32_t vae_dims[2] = { 1, num_features };
        nimcp_tensor_t* vae_input = nimcp_tensor_create(vae_dims, 2, NIMCP_DTYPE_F32);
        if (vae_input) {
            float* vae_data = (float*)nimcp_tensor_data(vae_input);
            if (vae_data) {
                memcpy(vae_data, features, num_features * sizeof(float));
            }

            // Compute anomaly score (high = out-of-distribution)
            float anomaly_score = 0.0f;
            if (vae_compute_anomaly_score(vae, vae_input, &anomaly_score) == 0) {
                brain->last_vae_anomaly_score = anomaly_score;

                // Boost learning rate for anomalous inputs (harder = learn more)
                if (anomaly_score > 2.0f) {
                    effective_learning_rate *= fminf(1.5f, 1.0f + anomaly_score * 0.1f);
                }
            }

            // Get free energy (negative ELBO) for FEP integration
            float fe = vae_get_free_energy(vae);
            if (!isnan(fe)) {
                brain->last_vae_free_energy = fe;
            }

            nimcp_tensor_destroy(vae_input);
        }
    }

    // ========================================================================
    // PHASE 5.1: ENGRAM RECALL — PRIME NETWORK WITH SIMILAR MEMORIES
    // ========================================================================
    // WHAT: Recall similar memories before learning to prime hidden neurons
    // WHY:  Biological brains recall related knowledge before encoding new info
    //       This creates contextual priming that improves learning
    // HOW:  Query engram system with input features as cue, inject recalled
    //       activations into hidden layer to give forward pass a "head start"
    //
    // BIOLOGICAL BASIS:
    // - Hippocampal pattern completion (Marr 1971)
    // - Contextual priming in associative memory (Anderson 1983)
    // - Prior knowledge facilitates new learning (Tse et al. 2007)
    if (brain->engram_system) {
        bool cue_from_scratch = false;
        uint32_t* cue_neurons;
        if (brain->learn_scratch.cue_neurons && brain->learn_scratch.features_cap >= num_features) {
            cue_neurons = brain->learn_scratch.cue_neurons;
            cue_from_scratch = true;
        } else {
            cue_neurons = nimcp_calloc(num_features, sizeof(uint32_t));
        }
        if (cue_neurons) {
            // Map features to cue neuron IDs
            for (uint32_t i = 0; i < num_features; i++) {
                cue_neurons[i] = i;
            }

            #define RECALL_MAX_NEURONS 100
            uint32_t recalled_neurons[RECALL_MAX_NEURONS];
            float recalled_activations[RECALL_MAX_NEURONS];
            memset(recalled_neurons, 0, sizeof(recalled_neurons));
            memset(recalled_activations, 0, sizeof(recalled_activations));
            float recall_confidence = 0.0f;

            uint64_t recalled_id = engram_recall(
                brain->engram_system,
                cue_neurons, num_features,
                recalled_neurons, recalled_activations,
                RECALL_MAX_NEURONS, &recall_confidence
            );

            // If recall succeeded with decent confidence, prime network neurons
            if (recalled_id != 0 && recall_confidence > 0.3f) {
                neural_network_t base_net = adaptive_network_get_base_network(brain->network);
                if (base_net) {
                    // Inject recalled activations as priming (blend with current state)
                    float prime_strength = recall_confidence * 0.3f; // 30% of recall confidence
                    for (uint32_t r = 0; r < RECALL_MAX_NEURONS; r++) {
                        uint32_t nid = recalled_neurons[r];
                        if (nid < neural_network_get_num_neurons(base_net)) {
                            neuron_t* neuron = neural_network_get_neuron(base_net, nid);
                            if (neuron) {
                                neuron->state += prime_strength * recalled_activations[r];
                            }
                        }
                    }
                }
                // Trigger reconsolidation on recalled engram (biologically realistic)
                engram_trigger_reconsolidation(brain->engram_system, recalled_id);
            }

            if (!cue_from_scratch) nimcp_free(cue_neurons);
        }
    }

    // ========================================================================
    // PHASE 5.2: NEUROMODULATOR PRE-COMPUTATION
    // ========================================================================
    // WHAT: Set neuromodulator levels based on learning context
    // WHY:  Dopamine gates STDP/eligibility, ACh enhances encoding, NE modulates attention
    // HOW:  Compute levels from curiosity, confidence, and recent loss trend
    //
    // BIOLOGICAL BASIS:
    // - DA: Reward prediction error (Schultz 1997)
    // - ACh: Encoding mode in hippocampus (Hasselmo 2006)
    // - NE: Arousal and attention (Aston-Jones & Cohen 2005)
    BRAIN_ENSURE_NEUROMOD(brain);
    if (brain->neuromodulator_system) {
        // Dopamine: based on expected reward (inverse of recent loss)
        float da_level = 0.5f; // baseline
        if (brain->loss_history_count >= LOSS_HISTORY_SIZE) {
            // Use loss trend: compare oldest vs newest entry in circular buffer.
            // Oldest entry is at loss_history_index (next write position).
            // Newest entry is at (loss_history_index + LOSS_HISTORY_SIZE - 1) % LOSS_HISTORY_SIZE.
            uint32_t oldest_idx = brain->loss_history_index;
            uint32_t newest_idx = (brain->loss_history_index + LOSS_HISTORY_SIZE - 1) % LOSS_HISTORY_SIZE;
            float loss_trend = brain->loss_history[oldest_idx] - brain->loss_history[newest_idx];
            if (loss_trend > 0.0f) {
                da_level = fminf(1.0f, 0.5f + loss_trend * 2.0f); // Boost DA when improving
            } else {
                da_level = fmaxf(0.1f, 0.5f + loss_trend * 2.0f); // Suppress DA when worsening
            }
        }

        // Acetylcholine: high during novel inputs (encoding mode)
        float ach_level = 0.5f;
        if (brain->last_novelty_score > 0.0f) {
            ach_level = fminf(1.0f, 0.5f + brain->last_novelty_score * 0.5f);
        }

        // Norepinephrine: arousal/difficulty signal
        float ne_level = fminf(1.0f, confidence); // Higher confidence = more engaged

        // Set levels (these will be read by STDP/eligibility during hybrid learning)
        neuromodulator_set_level(brain->neuromodulator_system, NEUROMOD_DOPAMINE, da_level);
        neuromodulator_set_level(brain->neuromodulator_system, NEUROMOD_ACETYLCHOLINE, ach_level);
        neuromodulator_set_level(brain->neuromodulator_system, NEUROMOD_NOREPINEPHRINE, ne_level);
    }

    // ========================================================================
    // PHASE 5.3: OSCILLATION THETA GATING
    // ========================================================================
    // WHAT: Check theta oscillation phase to optimize memory encoding timing
    // WHY:  Memory encoding is enhanced during theta peak (Hasselmo et al. 2002)
    // HOW:  Read theta phase from oscillations, modulate learning rate
    //
    // BIOLOGICAL BASIS:
    // - Theta-phase coupling in hippocampus (Buzsaki 2002)
    // - Encoding at theta peak, retrieval at theta trough (Hasselmo 2005)
    if (brain->oscillations) {
        // Theta modulation: boost learning during encoding-favorable phase
        // Simplified theta gate: sinusoidal modulation based on step count
        // Full implementation would use actual phase from oscillation analyzer
        float theta_freq = 6.0f; // 6 Hz theta
        float phase = fmodf((float)brain->stats.total_learning_steps * 0.01f * theta_freq, 1.0f);
        float theta_modulation = 0.8f + 0.2f * sinf(phase * 2.0f * 3.14159265f);
        effective_learning_rate *= theta_modulation;
    }

    // Neuromodulator-gated backprop LR: dopamine boosts LR when learning is
    // productive (loss decreasing), suppresses when worsening. This connects
    // the biological reward signal to gradient-based learning, not just STDP.
    // DA range [0.1, 1.0] → LR scale [0.5, 1.5] (centered on baseline DA=0.5)
    if (brain->neuromodulator_system) {
        float da = neuromodulator_get_level(brain->neuromodulator_system, NEUROMOD_DOPAMINE);
        float da_lr_scale = 0.5f + da;  // DA=0.1→0.6x, DA=0.5→1.0x, DA=1.0→1.5x
        effective_learning_rate *= da_lr_scale;
    }

    // Learn using adaptive network with curiosity-modulated learning rate
    // HYBRID mode: Supervised backprop + biological plasticity (STDP/BCM/eligibility)
    // This enables neuromodulator-gated plasticity alongside gradient descent
    uint64_t _t_learn_start = nimcp_time_get_us();
    float network_loss = adaptive_network_learn(brain->network, &example, LEARN_MODE_HYBRID,
                                                effective_learning_rate);
    uint64_t _t_learn_us = nimcp_time_get_us() - _t_learn_start;

    // ========================================================================
    // BIOLOGICAL PLASTICITY: TPB + EDP + Coordinator (after backprop)
    // ========================================================================
    // Wire loss into biological plasticity pathways:
    // - TPB: Loss → RPE → neuromodulators (DA/ACh/NE/5-HT) → region-specific STDP/BCM
    // - EDP: Three-factor eligibility (Activity × Eligibility × Reward)
    // - Coordinator: Interval-based mechanism updates (STDP 10ms, BCM 50ms, Homeostatic 1s)
    if (brain->plasticity_bridge && brain->enable_plasticity_bridge) {
        float rpe = 0.0f;
        tpb_report_loss(brain->plasticity_bridge, network_loss, &rpe);
    }
    if (brain->event_driven_plasticity && brain->enable_event_driven_plasticity) {
        edp_process_prediction_error(brain->event_driven_plasticity, network_loss, 0);
        if (edp_is_active(brain->event_driven_plasticity)) {
            float reward = -network_loss;
            edp_process_reward(brain->event_driven_plasticity, reward);
            edp_consolidate_eligibility(brain->event_driven_plasticity, reward);
        }
    }
    if (brain->plasticity_coordinator && brain->plasticity_coordinator_enabled) {
        uint64_t now_ms = nimcp_time_get_us() / 1000;
        plasticity_coordinator_update(brain->plasticity_coordinator, now_ms, 1.0f);
    }
    /* Structural plasticity: form new synapses for high-activity pairs */
    if (brain->structural_plasticity && brain->structural_plasticity_enabled) {
        float activity_hz = (network_loss < 0.5f) ? 30.0f : 5.0f;
        if (structural_plasticity_should_form(brain->structural_plasticity, activity_hz)) {
            uint32_t pre = (uint32_t)(features[0] * 997.0f) % brain->config.num_inputs;
            uint32_t post = (uint32_t)(features[1] * 991.0f) % brain->config.num_outputs;
            uint32_t syn_id = 0;
            structural_plasticity_form_synapse(
                brain->structural_plasticity, pre, post, activity_hz, &syn_id);
        }
    }
    /* Neuromodulator-gated reward learning: strengthen active pathways.
     * DA/ACh/NE levels set from loss trend above.
     * Eligibility traces updated by EDP above. */
    BRAIN_ENSURE_NEUROMOD(brain);
    if (brain->neuromodulator_system) {
        neural_network_t base_net = adaptive_network_get_base_network(brain->network);
        if (base_net) {
            neural_network_set_neuromodulator_system(base_net, brain->neuromodulator_system);
            float reward = 1.0f - fminf(network_loss, 1.0f);
            uint32_t modified = neural_network_apply_reward_learning_active(
                base_net, reward, REWARD_LEARNING_RATE, nimcp_time_get_us(), REWARD_ACTIVITY_THRESHOLD);
            if (modified > 0) {
                adaptive_network_invalidate_gpu_structure(brain->network);
            }
        }
    }

    // ========================================================================
    // BIOLOGICAL SECURITY: Monitor for attacks after learning (Phase 11)
    // ========================================================================
    if (brain->config.enable_bio_security) {
        nimcp_activity_stats_t activity_stats;
        nimcp_bio_attack_type_t attack = nimcp_security_monitor_excitotoxicity(
            adaptive_network_get_base_network(brain->network),
            &activity_stats
        );

        if (attack == NIMCP_BIO_ATTACK_EXCITOTOXICITY) {
            // CRITICAL: Excitotoxicity detected
            set_error("SECURITY: Excitotoxicity attack detected (%.1f%% activity)",
                     activity_stats.activity_ratio * 100.0F);

            if (brain->config.emergency_inhibit_on_attack) {
                // Emergency response
                nimcp_security_emergency_inhibit(adaptive_network_get_base_network(brain->network));
            }

            if (!attended_from_scratch) nimcp_free(attended_features);
            if (!target_from_scratch) nimcp_free(target);
            return -1.0F;  // Abort learning
        } else if (activity_stats.activity_ratio > brain->config.activity_warning_threshold) {
            // WARNING: Elevated activity - apply graduated inhibition
            nimcp_security_increase_inhibition(
                adaptive_network_get_base_network(brain->network),
                1.2F  // Increase inhibition by 20%
            );
        }
    }

    // ========================================================================
    // PHASE 6.5: MAMMILLARY MEMORY RELAY (Papez Circuit)
    // ========================================================================
    // WHAT: Relay training pattern through mammillary bodies for memory consolidation
    // WHY:  Papez circuit (hippocampus -> mammillary -> thalamus -> cingulate)
    //       is essential for episodic memory formation
    // HOW:  Send feature vector as memory trace, process Papez cycle
    //
    // BIOLOGICAL BASIS:
    // - Mammillary bodies as critical relay in Papez circuit (Papez 1937)
    // - Damage causes anterograde amnesia (Korsakoff syndrome)
    // - Head direction and spatial context binding
    if (brain->mammillary) {
        nimcp_mammillary_t* mb = (nimcp_mammillary_t*)brain->mammillary;
        // Create memory trace from training input
        uint32_t trace_id = 0;
        float emotional_valence = confidence > 0.5f ? 0.5f : -0.3f; // Positive for high-confidence

        int mb_rc = mammillary_receive_hippocampal_input(
            mb,
            features, num_features,
            MEMORY_TRACE_EPISODIC,
            emotional_valence,
            &trace_id
        );

        // Process Papez circuit cycle (relay -> thalamus)
        if (mb_rc == 0 && trace_id > 0) {
            mammillary_relay_to_thalamus(mb, trace_id);
            mammillary_process_papez_cycle(mb);

            // Strengthen trace based on learning success (low loss = stronger memory)
            if (network_loss < 0.3f) {
                mammillary_strengthen_trace(mb, trace_id,
                                            1.0f - network_loss);
            }
        }

        // Update mammillary state
        mammillary_update(mb, 0.01f);
    }

    // ========================================================================
    // PHASE 6.6: PRIME RESONANCE SIGNATURE GENERATION
    // ========================================================================
    // WHAT: Generate prime signature from training features for content-addressable memory
    // WHY:  Prime signatures enable efficient similarity-based memory retrieval
    // HOW:  Convert features to prime signature, store for later resonance queries
    //
    // BIOLOGICAL BASIS:
    // - Content-addressable memory in hippocampal CA3 (Marr 1971)
    // - Pattern separation via prime factorization analog
    if (brain->pr_memory_enabled) {
        prime_signature_t* sig = prime_sig_from_floats(features, num_features);
        if (sig) {
            // Store signature with the engram for future resonance-based retrieval
            // The signature enables finding similar training examples via Jaccard similarity
            // For now, generate and immediately free - the engram encoding in Phase M1
            // will use this for content addressing when PR memory is fully integrated
            prime_sig_destroy(sig);
        }
    }

    // Post-learning evaluation: MSE between updated network output and target
    uint64_t _t_post_start = nimcp_time_get_us();
    bool prediction_from_scratch = false;
    float* prediction;
    if (brain->learn_scratch.prediction && brain->learn_scratch.target_cap >= brain->config.num_outputs) {
        prediction = brain->learn_scratch.prediction;
        prediction_from_scratch = true;
    } else {
        prediction = nimcp_calloc(brain->config.num_outputs, sizeof(float));
    }
    if (prediction) {
        // Forward pass AFTER weight update to measure post-learning accuracy
        adaptive_network_forward(brain->network, features, num_features, prediction,
                                brain->config.num_outputs, 0);

        // MSE loss: meaningful metric that tracks actual output-target convergence
        float post_mse = 0.0F;
        uint32_t pred_argmax = 0;
        float pred_max = prediction[0];
        for (uint32_t i = 0; i < brain->config.num_outputs; i++) {
            float error = prediction[i] - target[i];
            post_mse += error * error;
            if (prediction[i] > pred_max) {
                pred_max = prediction[i];
                pred_argmax = i;
            }
        }
        post_mse /= brain->config.num_outputs;
        network_loss = post_mse;

        // Track label-match accuracy: does argmax(output) match argmax(target)?
        uint32_t target_argmax = 0;
        float target_max = target[0];
        for (uint32_t i = 1; i < brain->config.num_outputs; i++) {
            if (target[i] > target_max) {
                target_max = target[i];
                target_argmax = i;
            }
        }
        bool label_match = (pred_argmax == target_argmax);

        // Update running accuracy (exponential moving average, alpha=0.01)
        float match_val = label_match ? 1.0F : 0.0F;
        brain->stats.running_accuracy =
            brain->stats.running_accuracy * 0.99F + match_val * 0.01F;

        if (!prediction_from_scratch) nimcp_free(prediction);
    }

    // ========================================================================
    // PHASE 7: MULTI-NETWORK ENSEMBLE TRAINING (LNN + CNN)
    // ========================================================================
    // WHAT: Train auxiliary networks alongside the adaptive SNN
    // WHY:  Ensemble learning: different architectures capture different patterns.
    //       LNN captures temporal dynamics, CNN captures spatial features,
    //       Adaptive SNN captures spike-timing correlations.
    // HOW:  Use training_dispatch_step() in HYBRID mode which trains ALL
    //       available networks (LNN, CNN, SNN) and picks lowest loss.
    //
    // BIOLOGICAL BASIS:
    // - Parallel cortical pathways: dorsal (spatial) + ventral (object) streams
    // - Multiple memory systems: procedural (SNN), semantic (CNN), episodic (LNN)
    // - Population coding: diverse representations improve robustness
    if (brain->active_network_type == NIMCP_NETWORK_HYBRID) {
        training_dispatch_result_t dispatch_result = {0};

        // Use the pre-built one-hot target vector
        // training_dispatch_step in HYBRID mode trains all available networks
        training_dispatch_step(brain, features, num_features, target,
                              brain->config.num_outputs, &dispatch_result);
    }

    // ========================================================================
    // PHASE 7.5: VAE TRAINING STEP
    // ========================================================================
    if (brain->vae_system && brain->vae_enabled) {
        vae_system_t* vae = (vae_system_t*)brain->vae_system;

        // Create input tensor from features (rank-2: [1, num_features] for batch_size=1)
        uint32_t vae_dims[2] = { 1, num_features };
        nimcp_tensor_t* vae_input = nimcp_tensor_create(vae_dims, 2, NIMCP_DTYPE_F32);
        if (vae_input) {
            float* vae_data = (float*)nimcp_tensor_data(vae_input);
            if (vae_data) {
                memcpy(vae_data, features, num_features * sizeof(float));
            }

            // Train VAE: forward + loss + backward + update
            vae_loss_t vae_loss = {0};
            int vae_rc = vae_train_step(vae, vae_input, &vae_loss);
            if (vae_rc == 0) {
                // Update free energy on brain (negative ELBO)
                brain->last_vae_free_energy = vae_loss.free_energy;
            }

            // If VAE training bridge exists, do joint VAE+SNN training
            if (brain->vae_training_bridge) {
                vae_training_bridge_t* vtb = (vae_training_bridge_t*)brain->vae_training_bridge;
                vae_training_step_result_t vae_step_result = {0};
                vae_training_step(vtb, features, num_features, target,
                                  brain->config.num_outputs, &vae_step_result);
            }

            nimcp_tensor_destroy(vae_input);
        }
    }

    // ========================================================================
    // PHASE 7.8: ATTENTION-PLASTICITY BRIDGE UPDATE
    // ========================================================================
    // WHAT: Feed learning outcomes back to attention-plasticity bridge
    // WHY:  Attention-modulated STDP: successful learning (low loss) reinforces
    //       the attention patterns that led to it. Reward signal modulates
    //       eligibility traces, BCM thresholds, and novelty detection.
    // HOW:  Record reward (inverse loss), focus event, and update plasticity.
    //
    // BIOLOGICAL BASIS:
    // - Roelfsema & van Ooyen (2005): Attention-gated reinforcement learning
    // - Reward-modulated STDP: DA gates trace-to-weight conversion
    // - Attended synapses undergo stronger LTP (Noudoost et al., 2010)
    if (brain->attention_plasticity && brain->attention_training_enabled) {
        attention_plasticity_bridge_t* apb =
            (attention_plasticity_bridge_t*)brain->attention_plasticity;
        uint64_t now_us = nimcp_time_get_us();

        // Reward signal: low loss = high reward, clamped to [0, 1]
        float reward = fmaxf(0.0f, 1.0f - network_loss * 2.0f);
        attention_plasticity_reward(apb, reward, now_us);

        // Record focus event with current attention strength
        attention_plasticity_focus(apb, 0, brain->last_attention_strength, now_us);

        // Set modulation level so bridge knows current attention state
        attention_plasticity_set_attention_modulation(apb, brain->last_attention_strength);

        // Update all plasticity mechanisms (STDP, BCM, eligibility, habituation)
        attention_plasticity_update(apb, 1.0f);  // 1ms learning step
    }

    // ========================================================================
    // PHASE 8: PERCEPTUAL CORTEX GRADIENT FEEDBACK (post-learning)
    // ========================================================================
    // WHAT: Feed learning loss back to sensory cortices as gradient signal
    // WHY:  Allows V1/A1 to adapt feature extraction based on downstream
    //       learning outcomes — features that cause high loss get modified.
    // HOW:  Use network_loss as a uniform gradient signal, scaled by LR.
    //
    // BIOLOGICAL BASIS:
    // - Feedback projections from PFC/IT to V1 modulate plasticity (Gilbert 2013)
    // - Top-down error signals refine sensory representations over time
    // - Gradient ≈ neuromodulatory feedback (DA/ACh) gating V1/A1 STDP
    if (brain->visual_cortex) {
        uint32_t vdim = visual_cortex_get_feature_dim(brain->visual_cortex);
        if (vdim > 0) {
            // Uniform gradient: broadcast loss as feedback to all visual features
            float* vgrad = nimcp_malloc(vdim * sizeof(float));
            if (vgrad) {
                for (uint32_t i = 0; i < vdim; i++) {
                    vgrad[i] = network_loss;
                }
                float scale = fminf(effective_learning_rate, 1.0f);
                visual_cortex_apply_gradient_feedback(brain->visual_cortex,
                                                     vgrad, vdim, scale);
                nimcp_free(vgrad);
            }
        }
        // Disable training mode (caching) until next example
        visual_cortex_set_training_mode(brain->visual_cortex, false);
    }

    if (brain->audio_cortex) {
        uint32_t adim = audio_cortex_get_feature_dim(brain->audio_cortex);
        if (adim > 0) {
            float* agrad = nimcp_malloc(adim * sizeof(float));
            if (agrad) {
                for (uint32_t i = 0; i < adim; i++) {
                    agrad[i] = network_loss;
                }
                float scale = fminf(effective_learning_rate, 1.0f);
                audio_cortex_apply_gradient_feedback(brain->audio_cortex,
                                                    agrad, adim, scale);
                nimcp_free(agrad);
            }
        }
        audio_cortex_set_training_mode(brain->audio_cortex, false);
    }

    // ========================================================================
    // PHASE 8.5: CORTICAL COLUMN LEARNING (post-learning)
    // ========================================================================
    // WHAT: Step brain regions so cortical columns process the training input
    // WHY:  Cortical columns encode features via competitive dynamics and
    //       lateral inhibition — stepping after learning lets columns adapt
    //       their spatial representations based on the training example.
    // HOW:  Feed training input to brain regions, then step forward.
    //
    // BIOLOGICAL BASIS:
    // - Cortical minicolumns are the basic functional units of cortex
    // - Competitive learning sharpens feature selectivity (Mountcastle 1997)
    // - Activity-dependent refinement of columnar representations
    if (brain->brain_regions) {
        // Step brain regions with a 1ms time delta to trigger
        // intra-region plasticity and inter-region propagation
        brain_module_step(brain->brain_regions, 1000);  // 1000 µs = 1 ms
    }

    // ========================================================================
    // PHASE 8.8: PREDICTIVE CODING UPDATE (post-learning)
    // ========================================================================
    // WHAT: Feed prediction error (from learning loss) into predictive hierarchy
    // WHY:  Predictive coding minimizes free energy — the learning loss IS the
    //       prediction error. Feeding it back updates internal generative model.
    // HOW:  Set input, run inference step with learning enabled, update weights.
    //
    // BIOLOGICAL BASIS:
    // - Free Energy Principle (Friston 2010): brain minimizes surprise
    // - Prediction errors in superficial layers drive learning in deep layers
    // - Loss ↔ surprise: network_loss is the empirical prediction error
    if (brain->predictive_coding) {
        // Present the training features as sensory input to the hierarchy
        pc_hierarchy_set_input(brain->predictive_coding, features);

        // Run one inference + learning step with the prediction error
        // dt=1.0ms, learn=true to update prediction weights
        pc_hierarchy_inference_step(brain->predictive_coding, 1.0f, true);
    }

    // Free attended features buffer (allocated in Phase 4.5)
    if (!attended_from_scratch) nimcp_free(attended_features);  // NULL-safe

    if (!target_from_scratch) nimcp_free(target);

    // Update statistics (atomic for thread safety)
    __atomic_fetch_add(&brain->stats.total_learning_steps, 1, __ATOMIC_RELAXED);

    // Invalidate cache after learning
    clear_cache(brain);

    // ========================================================================
    // PHASE E: HIGHER-ORDER COGNITIVE & SOCIAL LEARNING INTEGRATION
    // ========================================================================
    // WHAT: Update shadow emotions and bias detection during training
    // WHY:  Monitor for maladaptive patterns and biases in training data/behavior
    // HOW:  Analyze labels for bias markers, update self-monitoring systems
    //
    // BIOLOGICAL BASIS:
    // - Metacognitive monitoring during learning (prefrontal cortex)
    // - Social cognition processes during information encoding
    // - Emotional regulation during knowledge acquisition
    //
    // COMPLEXITY: O(1) - Simple pattern matching and updates

    uint64_t current_time = (uint64_t)(brain->stats.total_learning_steps * 1000);  // Simulated time in ms

    // Phase E5: Shadow Emotions - Monitor for maladaptive learning patterns
    if (brain->shadow_emotions) {
        // Decay dynamics
        shadow_update(brain->shadow_emotions, 0.001F, current_time);  // Small dt for incremental updates

        // Auto-intervene if shadow emotions detected
        shadow_auto_intervene(brain->shadow_emotions, current_time);
    }

    // Phase E6: Bias Detection - Analyze training labels for bias markers
    if (brain->bias_detection) {
        // Update system dynamics
        bias_update(brain->bias_detection, 0.001F, current_time);

        // Simple language analysis of training label for bias markers
        // This allows the system to detect if training data contains biased language
        social_group_t general_group;
        general_group.group_id = 0;
        general_group.bias_type = BIAS_RACIAL;  // Default, would be determined by context
        strncpy(general_group.group_name, "General", sizeof(general_group.group_name) - 1);
        general_group.is_marginalized = false;
        general_group.is_stigmatized = false;

        language_pattern_t pattern = bias_analyze_language(brain->bias_detection, label, &general_group, current_time);

        // If bias detected in training data, reduce future learning confidence
        if (pattern.contains_slur || pattern.objectification || pattern.victim_blaming ||
            pattern.hostile_sexism || pattern.incel_ideology) {
            // Apply automatic debiasing intervention
            if (bias_is_detected(brain->bias_detection, general_group.bias_type)) {
                bias_auto_debias(brain->bias_detection, current_time);
            }

            // Record detection in stats (could add brain->stats.bias_detections_in_training)
        }
    }

    // === PHASE E: FULL EMOTIONAL INTELLIGENCE UPDATES ===
    // Update all emotion systems during learning for holistic emotional processing

    // Phase E1: Grief and Loss - Monitor for themes of loss in training data
    if (brain->grief_system) {
        grief_update(brain->grief_system, 0.001F, current_time);
        // Could detect loss/grief themes in training labels for empathetic understanding
    }

    // Phase E2: Joy and Euphoria - Reward for successful learning steps
    if (brain->joy_system) {
        joy_update(brain->joy_system, 0.001F, current_time);

        // If learning was successful (low loss), register as value-aligned success
        if (network_loss >= 0 && network_loss < 0.1F) {
            // Low loss = successful learning = joy trigger
            // This creates positive reinforcement for good learning
            uint32_t aligned_values[] = {VALUE_CATEGORY_LEARNING, VALUE_CATEGORY_ACCURACY};
            joy_process_success(brain->joy_system,
                SUCCESS_TYPE_LEARNED_SKILL,
                aligned_values,
                2,  // num_values
                0.5F,  // Difficulty (moderate)
                brain->last_novelty_score,  // Use actual novelty from curiosity system
                current_time);
        }
    }

    // Phase E3: Remorse and Regret - Monitor for poor decisions or errors
    if (brain->remorse_system) {
        remorse_update(brain->remorse_system, 0.001F, current_time);

        // If training resulted in high loss, register as regrettable event
        if (network_loss > 0.5F) {
            // High loss = learning failure = potential for improvement
            // Creates motivation to avoid similar failures
            uint32_t violated_values[] = {VALUE_CATEGORY_ACCURACY};
            remorse_process_event(brain->remorse_system,
                EVENT_POOR_DECISION,
                violated_values,
                1,  // num_values
                0.3F,  // Harm caused (moderate - not severe)
                0.8F,  // Controllability (we have control over learning)
                true,  // Reversible (can learn better next time)
                current_time);
        }
    }

    // Phase E4: Love, Loyalty, Friendship - Build positive associations with learning
    if (brain->social_bond_system) {
        social_update(brain->social_bond_system, 0.001F, current_time);
        // Could strengthen bonds with training data sources that provide good examples
    }

    // ========================================================================
    // Phase 11 Enhancement C1.1: QUANTUM ANNEALING WEIGHT OPTIMIZATION
    // ========================================================================
    // WHAT: Periodically run quantum annealing to escape local minima
    // WHY:  Gradient descent can get stuck; quantum tunneling explores better solutions
    // HOW:  Every N learning steps, optimize network weights using simulated quantum annealing
    //
    // BIOLOGICAL BASIS:
    // - Sleep-based memory consolidation reorganizes synaptic weights
    // - Exploration vs exploitation balance in learning
    // - Stochastic resonance helps escape suboptimal configurations
    //
    // COMPLEXITY: O(annealing_steps * num_weights) - expensive, run infrequently

    if (brain->config.enable_quantum_annealing &&
        brain->quantum_annealer &&
        brain->config.quantum_annealing_frequency > 0 &&
        (brain->stats.total_learning_steps % brain->config.quantum_annealing_frequency) == 0) {

        brain->stats.quantum_annealing_runs++;

        // Quantum annealing weight optimization - simplified implementation
        // WHAT: Optimize a small subset of critical weights to avoid excessive computation
        // WHY:  Full network optimization is expensive; focus on high-impact weights
        // HOW:  Sample top-K synapses by activity, optimize their weights, apply back
        //
        // PERFORMANCE: Limit to 100 weights to keep optimization tractable
        // Full network optimization would be O(N*M) where N=neurons, M=synapses_per_neuron

        neural_network_t base_net = adaptive_network_get_base_network(brain->network);
        if (base_net) {
            uint32_t num_neurons = neural_network_get_num_neurons(base_net);

            // Sample a small subset for optimization (e.g., first 10 neurons, ~100 weights)
            uint32_t num_neurons_to_optimize = (num_neurons < 10) ? num_neurons : 10;

            // Count total weights to optimize
            uint32_t total_weights = 0;
            for (uint32_t n = 0; n < num_neurons_to_optimize; n++) {
                neuron_t* neuron = neural_network_get_neuron(base_net, n);
                if (neuron) {
                    total_weights += NEURON_OUT_COUNT(neuron);
                }
            }

            if (total_weights > 0 && total_weights <= 1000) {  // Reasonable limit
                // Allocate weight vectors
                float* current_weights = nimcp_malloc(total_weights * sizeof(float));
                float* optimized_weights = nimcp_malloc(total_weights * sizeof(float));

                if (current_weights && optimized_weights) {
                    // Extract current weights
                    uint32_t weight_idx = 0;
                    for (uint32_t n = 0; n < num_neurons_to_optimize; n++) {
                        neuron_t* neuron = neural_network_get_neuron(base_net, n);
                        if (neuron) {
                            uint32_t nsyn = NEURON_OUT_COUNT(neuron);
                            for (uint32_t s = 0; s < nsyn; s++) {
                                synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, s);
                                current_weights[weight_idx++] = h ? h->weight : 0.0F;
                            }
                        }
                    }

                    // Run quantum annealing optimization
                    // Uses quantum_weight_energy (L2 regularization) as objective
                    quantum_anneal(brain->quantum_annealer, nimcp_brain_learning_quantum_weight_energy,
                                 current_weights, optimized_weights, total_weights, NULL);

                    // Apply optimized weights back to network
                    weight_idx = 0;
                    for (uint32_t n = 0; n < num_neurons_to_optimize; n++) {
                        neuron_t* neuron = neural_network_get_neuron(base_net, n);
                        if (neuron) {
                            uint32_t nsyn = NEURON_OUT_COUNT(neuron);
                            for (uint32_t s = 0; s < nsyn; s++) {
                                synapse_handle_t* h = NEURON_OUT_HANDLE(neuron, s);
                                if (!h) { weight_idx++; continue; }
                                // Apply with damping to avoid disrupting learning too much
                                float alpha = 0.1F;  // Mix 10% optimized, 90% current
                                h->weight =
                                    alpha * optimized_weights[weight_idx] +
                                    (1.0F - alpha) * h->weight;
                                weight_idx++;
                            }
                        }
                    }
                }

                nimcp_free(current_weights);
                nimcp_free(optimized_weights);
            }
        }
    }

    // ========================================================================
    // Phase 11: ADAPTIVE LEARNING RATE (Simple Online Meta-Learning)
    // ========================================================================
    // WHAT: Adjust learning rate based on loss trend
    // WHY:  Faster convergence with adaptive step sizes
    // HOW:  Track loss history, accelerate if decreasing, slow if increasing
    //
    // BIOLOGICAL BASIS:
    // - Meta-learning: brain adapts its own learning process
    // - Synaptic homeostasis: plasticity rates self-regulate
    //
    // COMPLEXITY: O(1)
    nimcp_brain_learning_adapt_learning_rate(brain, network_loss);

    // ========================================================================
    // FEEDBACK LOOP: Learning → Sleep Pressure Accumulation
    // ========================================================================
    // WHAT: Accumulate sleep pressure after learning
    // WHY:  Learning is metabolically expensive, builds adenosine (sleep pressure)
    // HOW:  Increment sleep pressure counter each time we learn
    if (brain->sleep_system && brain->config.enable_sleep_wake_cycle) {
        // Each learning step increases sleep pressure
        // Biological basis: synaptic activity produces adenosine
        sleep_accumulate_pressure(brain->sleep_system, 1);  // 1 learning step
    }

    // ========================================================================
    // PHASE M1: MEMORY ENGRAM ENCODING
    // ========================================================================
    // WHAT: Encode learning experience as distributed memory trace
    // WHY:  Enable pattern completion recall and biological memory consolidation
    // HOW:  Map input features to engram neurons, tag with emotional state
    //
    // BIOLOGICAL BASIS:
    // - Engram cells (Tonegawa et al., 2015): neurons active during encoding
    // - IEG expression (c-fos/Arc) tags active neurons for consolidation
    // - Emotional arousal enhances encoding (amygdala modulation)
    // - Memory traces stored as distributed synaptic patterns
    //
    // COMPLEXITY: O(n) where n = num_features
    if (brain->engram_system) {
        // Create neuron ID array from feature indices (simplified mapping)
        // In full implementation, would map to actual active neurons in network
        uint32_t* neuron_ids = nimcp_calloc(num_features, sizeof(uint32_t));
        float* activations = nimcp_calloc(num_features, sizeof(float));

        if (neuron_ids && activations) {
            // Map features to engram neurons
            for (uint32_t i = 0; i < num_features; i++) {
                neuron_ids[i] = i;  // Simplified: feature index = neuron ID
                activations[i] = features[i];  // Feature value = activation
            }

            // Get emotional state for tagging
            // Note: Use confidence as proxy for emotional arousal
            // Higher confidence learning → higher arousal encoding
            emotional_tag_t emotion = {
                .valence = (confidence > 0.7F) ? 0.5F : 0.0F,  // Positive valence for high confidence
                .arousal = confidence,                          // Confidence as arousal proxy
                .timestamp_ms = current_time,
                .category = EMOTION_CAT_NEUTRAL,
                .intensity = confidence
            };

            // Encode engram (episodic memory of this learning event)
            uint64_t engram_id = engram_encode(
                brain->engram_system,
                neuron_ids,
                activations,
                num_features,
                MEMORY_TYPE_EPISODIC,  // Learning experiences are episodic
                emotion
            );

            (void)engram_id;  // Suppress unused warning (could log for debugging)
        }
        // Free unconditionally — nimcp_free(NULL) is a safe no-op,
        // so partial allocation failure (one NULL, one non-NULL) is handled.
        nimcp_free(neuron_ids);
        nimcp_free(activations);
    }


    // ========================================================================
    // PHASE M3: WORKING MEMORY TRANSFER INTEGRATION (LEARNING PIPELINE)
    // ========================================================================
    // WHAT: Add learned pattern to working memory with high attention
    // WHY:  Newly learned information should be available for consolidation
    // HOW:  Update attention (high for new learning), evaluate transfer criteria
    //
    // BIOLOGICAL BASIS:
    // - Attended learning enters working memory (Baddeley & Hitch, 1974)
    // - Rehearsal and attention enhance transfer to LTM (Atkinson & Shiffrin, 1968)
    // - Confidence/arousal boosts encoding strength (McGaugh, 2000)
    //
    // COMPLEXITY: O(1) - Constant time for transfer evaluation
    if (brain->wm_transfer_system && brain->working_memory) {
        // Note: Working memory updates would happen here in full implementation
        // For now, we demonstrate the transfer evaluation mechanism

        // Evaluate transfer criteria (time delta: typical learning cycle ~0.1s)
        const float TIME_DELTA_SECONDS = 0.1F;
        wm_transfer_evaluate(brain->wm_transfer_system, TIME_DELTA_SECONDS);
    }

    // ========================================================================
    // PHASE M4: SEMANTIC MEMORY INTEGRATION (LEARNING PIPELINE)
    // ========================================================================
    // WHAT: Extract semantic concepts from consolidated memories
    // WHY:  Build abstract knowledge network from learned experiences
    // HOW:  Query Phase M2 for semantic memories, create concepts and relations
    //
    // BIOLOGICAL BASIS:
    // - Semantic abstraction from episodic experiences (Tulving, 1972)
    // - Concept formation through repeated exposure (Rosch, 1975)
    // - Semantic networks built from cortical representations (Collins & Quillian, 1969)
    //
    // COMPLEXITY: O(n) where n = number of new semantic memories in M2
    if (brain->semantic_memory && brain->systems_consolidation) {
        // Extract concepts from consolidated semantic memories
        // This builds the semantic network over time as learning progresses
        uint32_t concepts_extracted = semantic_memory_extract_from_consolidation(
            brain->semantic_memory
        );
        (void)concepts_extracted;  // Suppress unused warning (could log for debugging)
    }

    // ========================================================================
    // PHASE C4: SHANNON INFORMATION FLOW ANALYSIS (LEARNING PIPELINE)
    // ========================================================================
    // WHAT: Analyze information flow and detect bottlenecks after learning
    // WHY:  Monitor channel capacity, detect underutilized synapses
    // HOW:  Sample synapses, compute Shannon metrics, detect bottlenecks
    //
    // BIOLOGICAL BASIS:
    // - Neural efficiency: Information-theoretic brain function (Laughlin & Sejnowski, 2003)
    // - Sparse coding: Maximize information transfer with minimal energy (Olshausen & Field, 1996)
    // - Capacity limits: Channel capacity constraints in neural circuits (Koch et al., 2006)
    //
    // COMPLEXITY: O(1) - Monitoring enabled, detailed metrics computed via separate API
    //
    // NOTE: Full synapse-level Shannon analysis will be available through a dedicated
    // API once internal neuron/synapse structures are exposed via proper accessors.
    // For now, this marks monitoring as requested and initializes metrics structure.
    if (brain->enable_shannon_monitoring) {
        // Initialize/update basic network-level metrics
        // Detailed synapse sampling will be added in future enhancement
        brain->last_shannon_metrics.num_synapses = 0;  // To be computed
        brain->last_shannon_metrics.num_neurons = 0;   // To be computed
        // Full implementation pending internal accessor APIs
    }

    // ========================================================================
    // PHASE T1: BIOLOGICAL PLASTICITY FRAMEWORK (TRAINING PIPELINE)
    // ========================================================================
    // WHAT: Apply biological plasticity mechanisms during learning
    // WHY:  Provides homeostatic regulation, dendritic computation, and predictive coding
    // HOW:  Three integrated systems working together for stable, efficient learning
    //
    // BIOLOGICAL BASIS:
    // - Homeostatic plasticity: Maintains neural activity within healthy ranges
    // - Dendritic nonlinearities: Local computation in dendritic branches
    // - Predictive coding: Hierarchical error minimization (Free Energy Principle)
    //
    // COMPLEXITY: O(n) for homeostatic, O(branches) for dendritic, O(levels) for predictive

    // Phase T1.1: Homeostatic Plasticity - Maintain activity levels
    if (brain->homeostatic && brain->config.enable_homeostatic_plasticity) {
        // Get current network activity (use loss as proxy for activity deviation)
        // Low loss = network performing well = moderate activity
        // High loss = network struggling = potentially abnormal activity
        neural_network_t base_net = adaptive_network_get_base_network(brain->network);
        if (base_net) {
            uint32_t num_neurons = neural_network_get_num_neurons(base_net);
            uint32_t homeo_cap = num_neurons < 1000 ? num_neurons : 1000;

            // Compute firing rates array from network state (capped at 1000 neurons)
            float* firing_rates = nimcp_malloc(homeo_cap * sizeof(float));
            if (firing_rates) {
                // Estimate firing rates from neuron outputs
                for (uint32_t i = 0; i < num_neurons && i < 1000; i++) {  // Cap at 1000 neurons
                    neuron_t* neuron = neural_network_get_neuron(base_net, i);
                    if (neuron) {
                        // Use neuron state as proxy for firing rate
                        // Higher state = higher instantaneous rate estimate
                        firing_rates[i] = neuron->state > 0 ? neuron->state * 10.0F : 1.0F;
                    }
                }

                // Update homeostatic controller (rate-based only, no weight access)
                // WHY:  Extracting a flat weight array from sparse synapse storage
                //        for 1.5M neurons is impractical. The controller iterates
                //        controller->num_neurons (all neurons) but weights was only
                //        allocated for ~100 neurons — causing massive buffer overrun.
                //        Pass NULL for weights; controller skips synaptic scaling
                //        weight modification but still tracks rate stability,
                //        intrinsic plasticity, and metaplasticity.
                const float DT_MS = 1.0F;
                homeostatic_controller_update(brain->homeostatic, firing_rates, NULL,
                                             0, DT_MS);

                nimcp_free(firing_rates);
            }
        }
    }


    // Phase T1.2: Dendritic Computation - Process through dendritic tree
    if (brain->dendritic && brain->config.enable_dendritic_computation) {
        // Dendritic trees process input signals through compartmental modeling
        // During learning, we update the dendritic state based on input features
        //
        // This simulates dendritic integration where:
        // - Each branch receives subset of inputs
        // - NMDA dynamics modulate signal propagation
        // - Local dendritic spikes enhance signal detection

        // Inject a sample input into first branch as demonstration
        // Full implementation would distribute inputs across branches based on connectivity
        const float DT_MS = 1.0F;
        dendritic_tree_update(brain->dendritic, DT_MS);
    }

    // Phase T1.3: Predictive Coding - Hierarchical error minimization
    if (brain->predictive_coding && brain->config.enable_biological_predictive) {
        // Predictive coding implements the Free Energy Principle:
        // - Bottom-up: Prediction errors propagate upward
        // - Top-down: Predictions propagate downward
        // - Learning minimizes prediction error at each level
        //
        // The loss from supervised learning drives prediction error updates

        // Set sensory input as observation for predictive coding
        // The hierarchy will update predictions based on bottom-up errors
        // Note: Input size is stored in the hierarchy config
        pc_hierarchy_set_input(brain->predictive_coding, features);

        // Run inference step with learning enabled
        // dt_ms = 1.0 simulates one millisecond of neural dynamics
        pc_hierarchy_inference_step(brain->predictive_coding, 1.0F, true);
    }

    // ========================================================================
    // PHASE C4.1: QUANTUM-SHANNON DIFFUSION (LEARNING PHASE)
    // ========================================================================
    // WHAT: Evolve quantum-Shannon diffusion after learning step
    // WHY:  Monitor information flow during learning, detect bottlenecks
    // HOW:  Evolve quantum walker, update Shannon metrics
    //
    // COMPLEXITY: O(E + N) where E = edges, N = neurons
    if (brain->enable_quantum_shannon_diffusion && brain->quantum_shannon_diffusion) {
        quantum_shannon_diffusion_t* qsd = (quantum_shannon_diffusion_t*)brain->quantum_shannon_diffusion;

        // Evolve with configured steps
        if (quantum_shannon_evolve(qsd, brain->quantum_shannon_evolution_steps)) {
            // Update metrics
            quantum_shannon_get_metrics(qsd, &brain->last_quantum_shannon_metrics);

            // Log if bottlenecks detected (useful for debugging/optimization)
            if (brain->last_quantum_shannon_metrics.num_bottlenecks > 0) {
                // Bottlenecks detected - could trigger adaptive plasticity in future
                // For now, just track in metrics
            }
        }
    }

    // ========================================================================
    // PHASE 23.5: GLIAL MODULATION UPDATE
    // ========================================================================
    // WHAT: Update astrocyte/oligodendrocyte state based on learning activity
    // WHY:  Glial cells modulate synaptic strength and myelinate active pathways
    // HOW:  Feed learning signal to glial integration system
    //
    // BIOLOGICAL BASIS:
    // - Astrocyte calcium waves modulate synaptic transmission (Perea et al. 2009)
    // - Activity-dependent myelination (Fields 2015)
    // - Astrocytic glutamate release enhances LTP (Jourdain et al. 2007)
    if (brain->glial) {
        // Use simulated time in microseconds for timestamp-based step
        uint64_t glial_timestamp = (uint64_t)(brain->stats.total_learning_steps * 1000);
        glial_integration_step(brain->glial, glial_timestamp);
    }

    uint64_t _t_post_us = nimcp_time_get_us() - _t_post_start;
    uint64_t _t_total_us = _t_learn_us + _t_post_us;

    // Log timing every 10 examples for profiling
    if ((brain->stats.total_learning_steps % 10) == 0) {
        LOG_INFO("LEARN TIMING: total=%llu ms  core_learn=%llu ms  post_phases=%llu ms",
                 (unsigned long long)(_t_total_us / 1000),
                 (unsigned long long)(_t_learn_us / 1000),
                 (unsigned long long)(_t_post_us / 1000));
    }

    brain_clear_error();
    return network_loss;
}

/**
 * @brief Enable multi-network training (LNN + CNN alongside adaptive)
 *
 * WHAT: Creates auxiliary LNN and CNN networks matched to brain dimensions
 * WHY:  Ensemble learning: different architectures capture different patterns.
 *       LNN captures temporal dynamics, CNN captures spatial features,
 *       Adaptive SNN captures spike-timing correlations.
 * HOW:  Creates NCP-architecture LNN, dense CNN, initializes dispatch as HYBRID
 *
 * @param brain Brain to augment
 * @return 0 on success, -1 on error
 */
int brain_enable_multi_network_training(brain_t brain)
{
    // Guard: NULL check
    if (!brain) {
        set_error("brain_enable_multi_network_training: brain is NULL");
        return -1;
    }

    // Guard: Already in HYBRID mode
    if (brain->active_network_type == NIMCP_NETWORK_HYBRID) {
        /* TODO: Re-enable HNN after fixing SIGSEGV in Hamiltonian forward path */
        return 0;
    }

    uint32_t num_inputs = brain->config.num_inputs;
    uint32_t num_outputs = brain->config.num_outputs;

    // ========================================================================
    // STEP 1: Create LNN network (NCP architecture)
    // ========================================================================
    // WHAT: Create a Liquid Neural Network matched to brain dimensions
    // WHY:  LNN captures continuous-time temporal dynamics via ODE neurons
    // HOW:  Use NCP (Neural Circuit Policy) 4-layer architecture:
    //       Input -> Sensory -> Inter -> Command -> Motor -> Output
    // LNN requires minimum dimensions for stable tensor operations.
    // Skip LNN for very small networks (< 8 inputs/outputs) to avoid
    // tensor dimension mismatch crashes in the ODE forward pass.
    if (!brain->lnn_network && num_inputs >= 8 && num_outputs >= 8) {
        /* Ensure LNN library is initialized (needed for forward_step in inference) */
        if (!lnn_is_initialized()) {
            lnn_init(1);
        }

        // Create NCP (Neural Circuit Policy) LNN with ALL layers capped to 256.
        // LNN papers use 19-256 neurons; beyond 256 the O(n^2) adjoint Jacobian
        // dominates (1024^2 = 1M ops/step vs 256^2 = 65K). When brain dims exceed
        // the cap, lnn_train_step() average-pools the input and truncates targets
        // to match the LNN's smaller dimensions.
        uint32_t lnn_cap = 256;
        uint32_t lnn_in = (num_inputs > lnn_cap) ? lnn_cap : num_inputs;
        uint32_t lnn_out = (num_outputs > lnn_cap) ? lnn_cap : num_outputs;
        uint32_t n_inter = (lnn_in + lnn_out) / 2;
        if (n_inter < 8) n_inter = 8;
        if (n_inter > lnn_cap) n_inter = lnn_cap;
        uint32_t n_command = lnn_out * 2;
        if (n_command < 8) n_command = 8;
        if (n_command > lnn_cap) n_command = lnn_cap;

        NIMCP_LOGGING_INFO("LNN NCP: brain dims %u→%u, LNN dims %u→%u→%u→%u→%u",
                           num_inputs, num_outputs, lnn_in, n_inter, n_command, lnn_out);

        lnn_network_t* lnn = lnn_network_create_ncp(
            lnn_in, n_inter, n_command, lnn_out
        );
        if (!lnn) {
            NIMCP_LOGGING_WARN("Failed to create LNN network for multi-network training");
            // Non-fatal: continue without LNN
        } else {
            lnn_network_init_weights(lnn, 0);  // Random seed
            lnn_network_set_training(lnn, true);
            brain->lnn_network = lnn;
            brain->owns_specialized_network = true;

            /* TODO: Re-enable HNN after fixing SIGSEGV in Hamiltonian forward path */
            if (0 && lnn->n_layers > 0 && lnn->layers[0]) {
                uint32_t sd = lnn->layers[0]->n_neurons;
                if (sd > 0) {
                    lnn_hamiltonian_config_t hcfg;
                    lnn_hamiltonian_config_default(&hcfg);
                    lnn_hamiltonian_net_t* H = lnn_hamiltonian_net_create(sd, &hcfg);
                    if (H) {
                        lnn->layers[0]->H_net = H;
                        lnn->layers[0]->use_hamiltonian = true;
                        /* Allocate momentum tensor p */
                        if (!lnn->layers[0]->p) {
                            uint32_t pd[1] = {sd};
                            lnn->layers[0]->p = nimcp_tensor_create(pd, 1, NIMCP_DTYPE_F32);
                        }
                        NIMCP_LOGGING_INFO("HNN: Hamiltonian enabled on LNN layer 0 (dim=%u, p=%s)",
                                           sd, lnn->layers[0]->p ? "ok" : "FAIL");
                    }
                }
            }
        }
    } else if (!brain->lnn_network) {
        NIMCP_LOGGING_INFO("LNN skipped: requires num_inputs >= 8 and num_outputs >= 8 "
                          "(brain has %u inputs, %u outputs)", num_inputs, num_outputs);
    }

    /* Enable HNN on existing LNN if not already active */
    if (brain->lnn_network && brain->lnn_network->n_layers > 0 &&
        brain->lnn_network->layers[0] &&
        !brain->lnn_network->layers[0]->use_hamiltonian && 0 /* TODO: fix HNN SIGSEGV */) {
        uint32_t sd = brain->lnn_network->layers[0]->n_neurons;
        if (sd > 0) {
            lnn_hamiltonian_config_t hcfg;
            lnn_hamiltonian_config_default(&hcfg);
            lnn_hamiltonian_net_t* H = lnn_hamiltonian_net_create(sd, &hcfg);
            if (H) {
                brain->lnn_network->layers[0]->H_net = H;
                brain->lnn_network->layers[0]->use_hamiltonian = true;
                if (!brain->lnn_network->layers[0]->p) {
                    uint32_t pd[1] = {sd};
                    brain->lnn_network->layers[0]->p = nimcp_tensor_create(pd, 1, NIMCP_DTYPE_F32);
                }
                NIMCP_LOGGING_INFO("HNN: Hamiltonian enabled on existing LNN layer 0 (dim=%u, p=%s)",
                                   sd, brain->lnn_network->layers[0]->p ? "ok" : "FAIL");
            }
        }
    }

    // ========================================================================
    // STEP 2: Create LNN training context
    // ========================================================================
    // WHAT: Initialize adjoint-method training for the LNN
    // WHY:  LNN requires ODE-aware gradient computation
    // HOW:  Create training context with slow learning rate (0.01)
    if (brain->lnn_network && !brain->lnn_training_ctx) {
        lnn_training_config_t lnn_cfg;
        lnn_training_config_default(&lnn_cfg);
        lnn_cfg.learning_rate = 0.01f;
        lnn_cfg.gradient_clip_norm = 100.0f;

        lnn_training_ctx_t* lnn_ctx = lnn_training_create(brain->lnn_network, &lnn_cfg);
        if (!lnn_ctx) {
            NIMCP_LOGGING_WARN("Failed to create LNN training context");
        } else {
            brain->lnn_training_ctx = lnn_ctx;
        }
    }

    // ========================================================================
    // STEP 2.5: Create SNN network (feedforward architecture)
    // ========================================================================
    // WHAT: Create a Spiking Neural Network matched to brain dimensions
    // WHY:  SNN captures temporal spike-timing patterns, event sequences
    // HOW:  Feedforward (input → hidden → output) with LIF neurons
    //       Capped to 256 neurons per layer (like LNN) to keep spike
    //       simulation tractable. Training dispatch creates snn_training_ctx.
    if (!brain->snn_network && num_inputs >= 8 && num_outputs >= 8) {
        uint32_t snn_cap = 256;
        uint32_t snn_in  = (num_inputs  > snn_cap) ? snn_cap : num_inputs;
        uint32_t snn_out = (num_outputs > snn_cap) ? snn_cap : num_outputs;
        uint32_t snn_hidden = (snn_in + snn_out) / 2;
        if (snn_hidden < 8) snn_hidden = 8;
        if (snn_hidden > snn_cap) snn_hidden = snn_cap;

        NIMCP_LOGGING_INFO("SNN feedforward: brain dims %u→%u, SNN dims %u→%u→%u",
                           num_inputs, num_outputs, snn_in, snn_hidden, snn_out);

        snn_config_t snn_cfg;
        snn_config_feedforward(&snn_cfg, snn_in, snn_hidden, snn_out);

        snn_network_t* snn = snn_network_create(&snn_cfg);
        if (!snn) {
            NIMCP_LOGGING_WARN("Failed to create SNN network for multi-network training");
            // Non-fatal: continue without SNN
        } else {
            brain->snn_network = snn;
            brain->owns_specialized_network = true;

            /* Create per-population FNO models for spectral dynamics prediction */
            if (snn->n_populations > 0 && !brain->snn_fno_populations) {
                brain->snn_fno_populations = nimcp_calloc(snn->n_populations, sizeof(void*));
                if (brain->snn_fno_populations) {
                    snn_fno_config_t fno_cfg;
                    snn_fno_config_default(&fno_cfg);
                    for (uint32_t p = 0; p < snn->n_populations; p++) {
                        uint32_t pop_n = snn->populations[p] ?
                            snn->populations[p]->n_neurons : snn_hidden;
                        brain->snn_fno_populations[p] =
                            snn_fno_population_create(p, pop_n, &fno_cfg);
                    }
                    brain->snn_fno_count = snn->n_populations;
                    NIMCP_LOGGING_INFO("SNN FNO: created %u population dynamics models",
                                       snn->n_populations);
                }
            }
        }
    } else if (!brain->snn_network) {
        NIMCP_LOGGING_INFO("SNN skipped: requires num_inputs >= 8 and num_outputs >= 8 "
                          "(brain has %u inputs, %u outputs)", num_inputs, num_outputs);
    }

    // ========================================================================
    // STEP 3: Create CNN trainer with dense layers (via shared helper)
    // ========================================================================
    ensure_cnn_trainer(brain);

    // ========================================================================
    // STEP 3.5: Create VAE (Variational Autoencoder)
    // ========================================================================
    // WHAT: Create a VAE matched to brain input dimensions
    // WHY:  VAE learns compressed latent representations, enables:
    //       - Anomaly detection (flag out-of-distribution inputs)
    //       - Generative replay (combat catastrophic forgetting)
    //       - Free Energy Principle integration (ELBO = variational free energy)
    //       - Better engram encoding via latent space
    // HOW:  Encoder: input_dim → hidden → latent_dim (mu, log_var)
    //       Decoder: latent_dim → hidden → input_dim (reconstruction)
    if (!brain->vae_system) {
        vae_config_t vae_cfg;
        vae_default_config(&vae_cfg);

        // Configure encoder: 2 hidden layers
        uint32_t latent_dim = num_inputs / 4;
        if (latent_dim < 16) latent_dim = 16;
        if (latent_dim > 256) latent_dim = 256;

        uint32_t vae_hidden = num_inputs / 2;
        if (vae_hidden < 32) vae_hidden = 32;
        if (vae_hidden > 512) vae_hidden = 512;

        vae_cfg.encoder.input_dim = num_inputs;
        vae_cfg.encoder.latent_dim = latent_dim;
        vae_cfg.encoder.num_layers = 2;
        vae_cfg.encoder.layers[0] = (vae_layer_config_t){
            .units = vae_hidden, .activation = VAE_ACTIVATION_RELU,
            .dropout_rate = 0.0f, .batch_norm = false, .use_bias = true
        };
        vae_cfg.encoder.layers[1] = (vae_layer_config_t){
            .units = vae_hidden / 2, .activation = VAE_ACTIVATION_RELU,
            .dropout_rate = 0.0f, .batch_norm = false, .use_bias = true
        };
        vae_cfg.encoder.mu_activation = VAE_ACTIVATION_LINEAR;
        vae_cfg.encoder.var_activation = VAE_ACTIVATION_SOFTPLUS;

        // Configure decoder: mirror of encoder
        vae_cfg.decoder.latent_dim = latent_dim;
        vae_cfg.decoder.output_dim = num_inputs;
        vae_cfg.decoder.num_layers = 2;
        vae_cfg.decoder.layers[0] = (vae_layer_config_t){
            .units = vae_hidden / 2, .activation = VAE_ACTIVATION_RELU,
            .dropout_rate = 0.0f, .batch_norm = false, .use_bias = true
        };
        vae_cfg.decoder.layers[1] = (vae_layer_config_t){
            .units = vae_hidden, .activation = VAE_ACTIVATION_RELU,
            .dropout_rate = 0.0f, .batch_norm = false, .use_bias = true
        };
        vae_cfg.decoder.final_activation = VAE_ACTIVATION_SIGMOID;
        vae_cfg.decoder.output_variance = false;

        // Training config
        vae_cfg.training.learning_rate = 0.001f;
        vae_cfg.training.beta = 1.0f;      // Standard VAE (beta=1)
        vae_cfg.training.beta_warmup_steps = 1000;
        vae_cfg.training.loss_type = VAE_LOSS_MSE;
        vae_cfg.training.gradient_clip = 5.0f;
        vae_cfg.training.batch_size = 1;    // Online learning

        vae_cfg.variant = VAE_VARIANT_BETA;
        vae_cfg.prior_type = VAE_PRIOR_STANDARD_NORMAL;
        vae_cfg.anomaly_threshold = 3.0f;

        vae_system_t* vae = vae_create(&vae_cfg);
        if (!vae) {
            NIMCP_LOGGING_WARN("Failed to create VAE for multi-network training");
        } else {
            vae_set_training(vae, true);
            brain->vae_system = vae;
            brain->vae_enabled = true;

            // Create VAE training bridge for joint VAE+SNN training
            vae_training_bridge_config_t vtb_cfg;
            vae_training_bridge_default_config(&vtb_cfg);
            vtb_cfg.algorithm = VAE_TRAIN_JOINT;
            vtb_cfg.loss_combination = VAE_LOSS_WEIGHTED;
            vtb_cfg.vae_loss_weight = 0.3f;   // 30% VAE loss
            vtb_cfg.snn_loss_weight = 0.7f;   // 70% SNN loss
            vtb_cfg.optimizer.learning_rate = 0.001f;
            vtb_cfg.optimizer.use_gradient_clipping = true;
            vtb_cfg.optimizer.gradient_clip_norm = 5.0f;

            vae_training_bridge_t* vtb = vae_training_bridge_create(&vtb_cfg);
            if (vtb) {
                vae_training_bridge_connect_vae(vtb, vae);
                brain->vae_training_bridge = vtb;
            }

            NIMCP_LOGGING_INFO("VAE created: input_dim=%u, latent_dim=%u, hidden=%u",
                              num_inputs, latent_dim, vae_hidden);
        }
    }

    // ========================================================================
    // STEP 4.5: Initialize attention-plasticity bridge for training
    // ========================================================================
    // WHAT: Create attention-plasticity bridge for attention-modulated learning
    // WHY:  Enables attention to gate STDP, BCM, and eligibility trace plasticity.
    //       Attended inputs undergo stronger LTP; unattended undergo LTD.
    //       Reward signals reinforce successful attention patterns.
    // HOW:  Create bridge with attention modulation + eligibility + novelty detection
    if (brain->multihead_attention && !brain->attention_plasticity) {
        attention_plasticity_config_t apb_cfg = attention_plasticity_config_default();
        apb_cfg.enable_attention_modulation = true;
        apb_cfg.attention_learning_gain = 0.3f;   // 30% attention influence on LR
        apb_cfg.focus_learning_boost = 1.5f;       // 50% LTP boost for focused items
        apb_cfg.unfocused_ltd_boost = 0.3f;        // Weak LTD for unfocused
        apb_cfg.enable_eligibility = true;
        apb_cfg.eligibility_decay = 0.95f;         // Slow trace decay
        apb_cfg.reward_modulation_gain = 1.0f;     // Full reward modulation
        apb_cfg.enable_novelty_detection = true;
        apb_cfg.novelty_boost = 0.3f;              // 30% boost for novel inputs
        apb_cfg.enable_bcm = true;                 // BCM metaplasticity
        apb_cfg.bcm_threshold_tau = 0.01f;         // Slow threshold adaptation

        attention_plasticity_bridge_t* apb = attention_plasticity_create(&apb_cfg);
        if (apb) {
            brain->attention_plasticity = apb;
            brain->attention_training_enabled = true;
            NIMCP_LOGGING_INFO("Attention-plasticity bridge created for training");
        } else {
            NIMCP_LOGGING_WARN("Failed to create attention-plasticity bridge");
        }
    }

    // ========================================================================
    // STEP 4: Set HYBRID mode and initialize dispatch
    // ========================================================================
    // WHAT: Switch brain to HYBRID training mode
    // WHY:  Enables training_dispatch_step to train all available networks
    // HOW:  Set active_network_type, call training_dispatch_init
    brain->active_network_type = NIMCP_NETWORK_HYBRID;

    nimcp_training_config_t dispatch_cfg = nimcp_training_config_default();
    dispatch_cfg.network_type = NIMCP_NETWORK_HYBRID;
    dispatch_cfg.learning_rate = brain->config.learning_rate;
    training_dispatch_init(brain, &dispatch_cfg);

    NIMCP_LOGGING_INFO("Multi-network training enabled: Adaptive + %s + %s + %s + %s",
        brain->lnn_network ? "LNN" : "(no LNN)",
        brain->cnn_trainer ? "CNN" : "(no CNN)",
        brain->vae_system  ? "VAE" : "(no VAE)",
        brain->attention_training_enabled ? "Attention" : "(no Attention)");

    // ========================================================================
    // STEP 5: Create Unified Training Manager
    // ========================================================================
    // Enable unified training automatically when secondary networks exist.
    // The UTM provides: composite loss across all networks, cross-network
    // gradient bridges, shared AdamW optimizer, LR scheduling, and ensures
    // all networks receive ground truth supervision (not just the adaptive).
    {
        bool has_secondary = (brain->cnn_trainer || brain->snn_training_ctx ||
                              brain->lnn_training_ctx);
        /* Also check for cortex CNNs */
        for (int ci = 0; ci < 4 && !has_secondary; ci++) {
            if (brain->cortex_cnns[ci]) has_secondary = true;
        }
        if (has_secondary && !brain->config.use_unified_training) {
            brain->config.use_unified_training = true;
            NIMCP_LOGGING_INFO("Auto-enabling unified training (secondary networks detected)");
        }
    }
    if (brain->config.use_unified_training && !brain->unified_training) {
        nimcp_unified_training_config_t utm_cfg;
        nimcp_utm_default_config(&utm_cfg);
        utm_cfg.learning_rate = brain->config.learning_rate;

        brain->unified_training = nimcp_utm_create(&utm_cfg);
        if (brain->unified_training) {
            /* Register available networks as trainable */
            const nimcp_trainable_network_ops_t* ops = NULL;
            void* adapter_ctx = NULL;

            /* Adaptive backbone — trains independently in Step 1 via
             * adaptive_network_learn(). Do NOT register in UTM because running
             * neural_network_forward() a second time within the same learn_vector
             * call corrupts neuron model state and causes SIGSEGV.
             * Cross-network gradient flow is handled via bridge loss instead. */

            /* CNN — uses own internal optimizer (tensor-level weights)
             * but participates in UTM composite loss and gradient flow */
            if (brain->cnn_trainer) {
                if (nimcp_trainable_cnn_create(brain->cnn_trainer, &ops, &adapter_ctx) == 0) {
                    nimcp_trainable_cnn_set_dims(adapter_ctx,
                        brain->config.num_inputs, brain->config.num_outputs);
                    nimcp_utm_register_network(brain->unified_training, ops, adapter_ctx, 0.5f);
                }
            }

            /* SNN — managed by UTM: adapter exposes flat weights as param groups,
             * UTM runs unified AdamW on SNN weights + sync_params writes back.
             * NOTE: brain->snn_training_ctx is snn_training_ctx_t (STDP/eProp) — a
             * DIFFERENT struct from snn_backprop_ctx_t (BPTT).  The trainable adapter
             * requires snn_backprop_ctx_t, so we create one from brain->snn_network. */
            if (brain->snn_network) {
                if (!brain->snn_backprop_ctx) {
                    snn_backprop_config_t bp_cfg = snn_backprop_default_config(SNN_TRAIN_BPTT);
                    bp_cfg.use_gradient_normalization = true;
                    bp_cfg.diversity_loss_weight = 0.1f;
                    brain->snn_backprop_ctx = snn_backprop_create(brain->snn_network, &bp_cfg);
                }
                if (brain->snn_backprop_ctx) {
                    ops = NULL; adapter_ctx = NULL;
                    if (nimcp_trainable_snn_create(
                            brain->snn_backprop_ctx,
                            &ops, &adapter_ctx) == 0) {
                        /* Use SNN network's actual dims, not brain I/O dims */
                        snn_network_t* snn_net = (snn_network_t*)brain->snn_network;
                        nimcp_trainable_snn_set_dims(adapter_ctx,
                            snn_net->config.n_inputs, snn_net->config.n_outputs);
                        nimcp_trainable_snn_set_managed(adapter_ctx, true);
                        nimcp_utm_register_network(brain->unified_training, ops, adapter_ctx, 0.5f);
                    }
                }
            }

            /* LNN — managed by UTM: adapter exposes flat params,
             * UTM runs unified AdamW on LNN params + sync_params writes back */
            if (brain->lnn_training_ctx) {
                ops = NULL; adapter_ctx = NULL;
                if (nimcp_trainable_lnn_create(
                        (struct lnn_training_ctx_s*)brain->lnn_training_ctx,
                        &ops, &adapter_ctx) == 0) {
                    /* Dims already set from lnn_ctx->network in create() */
                    nimcp_trainable_lnn_set_managed(adapter_ctx, true);
                    nimcp_utm_register_network(brain->unified_training, ops, adapter_ctx, 0.5f);
                }
            }

            /* Register per-cortex CNN processors in UTM for composite loss + bridges */
            {
                extern int cortex_cnn_utm_adapter_create(struct cortex_cnn_processor* proc,
                    const nimcp_trainable_network_ops_t** ops, void** ctx);
                for (int ci = 0; ci < 4; ci++) {
                    if (brain->cortex_cnns[ci]) {
                        ops = NULL; adapter_ctx = NULL;
                        if (cortex_cnn_utm_adapter_create(brain->cortex_cnns[ci],
                                                           &ops, &adapter_ctx) == 0) {
                            nimcp_utm_register_network(brain->unified_training,
                                                       ops, adapter_ctx, 0.3f);
                        }
                    }
                }
            }

            NIMCP_LOGGING_INFO("Unified training manager created with %u networks",
                              brain->unified_training->num_networks);

            /* Auto-wire cross-network bridges based on registered network types.
             * Convention: networks[0]=Adaptive, [1]=CNN, [2]=SNN, [3]=LNN, [4+]=CortexCNNs
             * Bridges: Adaptive→SNN (rate-to-spike), SNN→Adaptive (spike-to-rate),
             *          LNN→SNN (continuous-to-spike),
             *          CortexVisual↔CortexAudio, CortexAudio↔CortexSpeech,
             *          CortexVisual↔CortexSomato (LINEAR bridges) */
            nimcp_unified_training_manager_t* utm = brain->unified_training;
            int adaptive_idx = -1, snn_idx = -1, lnn_idx = -1;
            int cortex_idx[4] = {-1, -1, -1, -1};
            for (uint32_t n = 0; n < utm->num_networks; n++) {
                if (!utm->networks[n].ops) continue;
                switch (utm->networks[n].ops->type) {
                    case NIMCP_TRAINABLE_ADAPTIVE: adaptive_idx = (int)n; break;
                    case NIMCP_TRAINABLE_SNN:      snn_idx = (int)n; break;
                    case NIMCP_TRAINABLE_LNN:      lnn_idx = (int)n; break;
                    case NIMCP_TRAINABLE_CUSTOM: {
                        /* Match cortex CNN by name */
                        const char* name = utm->networks[n].ops->name;
                        if (name) {
                            if (strstr(name, "Visual"))  cortex_idx[0] = (int)n;
                            else if (strstr(name, "Audio"))   cortex_idx[1] = (int)n;
                            else if (strstr(name, "Speech"))  cortex_idx[2] = (int)n;
                            else if (strstr(name, "Somato"))  cortex_idx[3] = (int)n;
                        }
                        break;
                    }
                    default: break;
                }
            }

            if (adaptive_idx >= 0 && snn_idx >= 0) {
                nimcp_utm_add_bridge(utm, (uint32_t)adaptive_idx, (uint32_t)snn_idx,
                                     NIMCP_BRIDGE_RATE_TO_SPIKE);
                nimcp_utm_add_bridge(utm, (uint32_t)snn_idx, (uint32_t)adaptive_idx,
                                     NIMCP_BRIDGE_SPIKE_TO_RATE);
            }
            if (lnn_idx >= 0 && snn_idx >= 0) {
                nimcp_utm_add_bridge(utm, (uint32_t)lnn_idx, (uint32_t)snn_idx,
                                     NIMCP_BRIDGE_CONTINUOUS_TO_SPIKE);
            }

            /* Cross-cortex LINEAR bridges for cross-modal gradient flow */
            if (cortex_idx[0] >= 0 && cortex_idx[1] >= 0) {
                nimcp_utm_add_bridge(utm, (uint32_t)cortex_idx[0], (uint32_t)cortex_idx[1],
                                     NIMCP_BRIDGE_LINEAR);
                nimcp_utm_add_bridge(utm, (uint32_t)cortex_idx[1], (uint32_t)cortex_idx[0],
                                     NIMCP_BRIDGE_LINEAR);
            }
            if (cortex_idx[1] >= 0 && cortex_idx[2] >= 0) {
                nimcp_utm_add_bridge(utm, (uint32_t)cortex_idx[1], (uint32_t)cortex_idx[2],
                                     NIMCP_BRIDGE_LINEAR);
                nimcp_utm_add_bridge(utm, (uint32_t)cortex_idx[2], (uint32_t)cortex_idx[1],
                                     NIMCP_BRIDGE_LINEAR);
            }
            if (cortex_idx[0] >= 0 && cortex_idx[3] >= 0) {
                nimcp_utm_add_bridge(utm, (uint32_t)cortex_idx[0], (uint32_t)cortex_idx[3],
                                     NIMCP_BRIDGE_LINEAR);
                nimcp_utm_add_bridge(utm, (uint32_t)cortex_idx[3], (uint32_t)cortex_idx[0],
                                     NIMCP_BRIDGE_LINEAR);
            }

            if (utm->num_bridges > 0) {
                utm->config.enable_cross_network_gradients = true;
                NIMCP_LOGGING_INFO("Auto-wired %u cross-network bridges (gradient flow enabled)",
                                  utm->num_bridges);
            }

            /* Phase 5: Wire plasticity bridge into UTM and plasticity coordinator
             * so backprop can suppress biological plasticity during gradient updates.
             * Without this, STDP/BCM modify weights concurrently with backprop. */
            if (brain->plasticity_bridge && brain->enable_plasticity_bridge) {
                nimcp_utm_set_plasticity_bridge(utm, brain->plasticity_bridge);
                NIMCP_LOGGING_INFO("UTM plasticity bridge wired — biological plasticity will be "
                                  "suppressed during backprop");
            }
            if (brain->plasticity_coordinator && brain->plasticity_coordinator_enabled) {
                neural_plasticity_set_plasticity_bridge(
                    brain->plasticity_coordinator, brain->plasticity_bridge);
                NIMCP_LOGGING_INFO("Plasticity coordinator bridge wired — STDP/BCM gated "
                                  "during backprop");
            }

            /* Wire pink noise system for DFA-driven amplitude feedback.
             * DFA monitors training health; pink noise amplitudes adjust to
             * steer dynamics toward optimal α≈1.0 (pink noise regime). */
            if (brain->pink_noise) {
                nimcp_utm_set_pink_noise(utm, brain->pink_noise);
                NIMCP_LOGGING_INFO("UTM pink noise feedback wired — DFA will modulate "
                                  "neuromodulator noise amplitudes");
            }
        }
    }

    return 0;
}

/**
 * @brief Learn from batch of examples
 *
 * WHY: More efficient than individual calls
 * Enables mini-batch gradient descent
 *
 * COMPLEXITY: O(m*s*n) where m = num_examples
 *
 * @param brain Brain handle
 * @param examples Array of examples
 * @param num_examples Example count
 * @return Average loss or -1 on error
 */
float brain_learn_batch(brain_t brain, const brain_example_t* examples, uint32_t num_examples)
{
    // Guard: Validate parameters
    if (!brain || !examples || num_examples == 0) {
        set_error("Invalid parameters to brain_learn_batch");
        return -1.0F;
    }

    // Health monitoring: signal start of batch learning
    brain_heartbeat(brain, "brain_learn_batch:start", 0.0f);

    float total_loss = 0.0F;

    // Calculate heartbeat interval (every 10% or every 100 examples, whichever is larger)
    uint32_t heartbeat_interval = (num_examples / 10 > 100) ? num_examples / 10 : 100;
    if (heartbeat_interval == 0) heartbeat_interval = 1;

    for (uint32_t i = 0; i < num_examples; i++) {
        float loss = brain_learn_example(brain, examples[i].features, examples[i].num_features,
                                         examples[i].label, examples[i].confidence);

        if (loss < 0.0F) {
            return -1.0F;
        }

        total_loss += loss;

        // Health monitoring: periodic progress update
        if ((i + 1) % heartbeat_interval == 0) {
            float progress = (float)(i + 1) / (float)num_examples;
            brain_heartbeat(brain, "brain_learn_batch:progress", progress);
        }
    }

    // Health monitoring: batch learning complete
    brain_heartbeat(brain, "brain_learn_batch:complete", 1.0f);

    brain_clear_error();
    return total_loss / num_examples;
}

/**
 * @brief Learn batch with per-example loss output
 *
 * WHY: Python binding needs individual loss values for training diagnostics.
 *      The original brain_learn_batch only returns the average.
 *
 * @param brain Brain handle
 * @param examples Array of examples
 * @param num_examples Example count
 * @param losses_out Caller-allocated float[num_examples] to receive per-example losses.
 *                   Can be NULL to skip per-example tracking.
 * @return Average loss or -1 on error
 */
float brain_learn_batch_detailed(brain_t brain, const brain_example_t* examples,
                                 uint32_t num_examples, float* losses_out)
{
    if (!brain || !examples || num_examples == 0) {
        set_error("Invalid parameters to brain_learn_batch_detailed");
        return -1.0F;
    }

    brain_heartbeat(brain, "brain_learn_batch:start", 0.0f);

    float total_loss = 0.0F;
    uint32_t heartbeat_interval = (num_examples / 10 > 100) ? num_examples / 10 : 100;
    if (heartbeat_interval == 0) heartbeat_interval = 1;

    for (uint32_t i = 0; i < num_examples; i++) {
        float loss = brain_learn_example(brain, examples[i].features, examples[i].num_features,
                                         examples[i].label, examples[i].confidence);

        if (loss < 0.0F) {
            /* Fill remaining losses_out with -1 to signal error */
            if (losses_out) {
                for (uint32_t j = i; j < num_examples; j++)
                    losses_out[j] = -1.0F;
            }
            return -1.0F;
        }

        if (losses_out)
            losses_out[i] = loss;
        total_loss += loss;

        if ((i + 1) % heartbeat_interval == 0) {
            float progress = (float)(i + 1) / (float)num_examples;
            brain_heartbeat(brain, "brain_learn_batch:progress", progress);
        }
    }

    brain_heartbeat(brain, "brain_learn_batch:complete", 1.0f);
    brain_clear_error();
    return total_loss / num_examples;
}

/**
 * @brief Apply reward-based reinforcement learning
 *
 * WHAT: Apply eligibility-trace-based learning with reward signal
 * WHY:  Enable temporal credit assignment for RL tasks
 * HOW:  Call neural_network_apply_reward_learning() with reward and dopamine
 *
 * BIOLOGY: Three-factor learning rule (Hebbian + Reward + Dopamine)
 * - Eligibility traces mark recently active synapses ("synaptic tags")
 * - Dopamine bursts trigger consolidation ("capture")
 * - Reward signal modulates weight changes (Frey & Morris 1997)
 *
 * COMPLEXITY: O(n × s) where n=neurons, s=synapses_per_neuron
 * USE CASE: Reinforcement learning, temporal credit assignment
 *
 * @param brain Brain handle
 * @param reward Reward signal (0-1 positive, -1-0 negative)
 * @return Number of synapses modified
 */
uint32_t brain_apply_reward_learning(brain_t brain, float reward)
{
    // Guard: Validate brain
    if (!brain) {
        set_error("NULL brain handle");
        return 0;
    }

    // Guard: Validate reward range
    if (reward < -1.0F || reward > 1.0F) {
        set_error("Reward must be in range [-1.0, 1.0], got %.2f", reward);
        return 0;
    }

    // Phase 2: Ensure network is writable
    if (!ensure_writable_network(brain)) {
        return 0;  // Error already set
    }

    // Get base neural network from adaptive network
    neural_network_t base_network = adaptive_network_get_base_network(brain->network);
    if (!base_network) {
        set_error("Failed to get base network");
        return 0;
    }

    // Get current network time
    uint64_t current_time = nimcp_time_get_us();

    // Apply reward-modulated learning with eligibility traces
    // Uses neural_network_apply_reward_learning() from neuralnet.c (lines 1472-1600)
    uint32_t num_modified = neural_network_apply_reward_learning(
        base_network,
        reward,
        brain->config.learning_rate,
        current_time
    );

    // Update brain stats (atomic for thread safety)
    __atomic_fetch_add(&brain->stats.total_learning_steps, 1, __ATOMIC_RELAXED);

    brain_clear_error();
    return num_modified;
}

/**
 * @brief Learn by querying an LLM teacher
 *
 * WHY: Enables distillation from larger models
 * Brain learns to mimic LLM decisions efficiently
 *
 * COMPLEXITY: O(s*n) + LLM query time
 * USE CASE: Compress LLM knowledge into fast neural network
 *
 * @param brain Brain handle
 * @param input Input features
 * @param num_features Feature count
 * @param llm_fn Teacher function
 * @param llm_context Context for teacher
 * @return Loss value or -1 on error
 */
float brain_learn_from_llm(brain_t brain, const float* input, uint32_t num_features,
                           llm_teacher_fn_t llm_fn, void* llm_context)
{
    // Guard: Validate parameters
    if (!brain || !input || !llm_fn) {
        set_error("Invalid parameters to brain_learn_from_llm");
        return -1.0F;
    }

    // Guard: Check dimensions
    if (num_features != brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u", brain->config.num_inputs,
                  num_features);
        return -1.0F;
    }

    // Query LLM teacher
    char label[NIMCP_ID_BUFFER_SIZE] = {0};
    float confidence = llm_fn(input, num_features, llm_context, label, sizeof(label));

    // Guard: Validate LLM response
    if (confidence <= 0.0F) {
        set_error("LLM teacher returned invalid confidence: %.2f", confidence);
        return -1.0F;
    }

    // Learn from LLM's decision
    float loss = brain_learn_example(brain, input, num_features, label, confidence);

    brain_clear_error();
    return loss;
}


//=============================================================================
// Async Learning Implementation
//=============================================================================

/**
 * @brief Context for async learning worker thread
 */
typedef struct {
    brain_t brain;
    float* features;
    uint32_t num_features;
    char* label;
    float confidence;
    nimcp_promise_t promise;
} async_learn_context_t;

/**
 * @brief Background thread function for async learning
 *
 * WHAT: Performs learning in background thread and completes promise
 * WHY:  Enable non-blocking learning without blocking caller
 * HOW:  Call brain_learn_example, set result or error on promise
 *
 * THREAD SAFETY: Each thread has its own context, brain must support concurrent writes
 *                (or caller must ensure serialization)
 *
 * @param arg async_learn_context_t pointer
 * @return NULL (unused)
 */
static void* async_learn_thread(void* arg)
{
    async_learn_context_t* ctx = (async_learn_context_t*)arg;

    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ctx is NULL");


        return NULL;
    }

    LOG_MODULE_DEBUG("core_brain_learning", "Async learning started: %u features, label='%s'",
                    ctx->num_features, ctx->label);

    // Perform synchronous learning
    float loss = brain_learn_example(
        ctx->brain,
        ctx->features,
        ctx->num_features,
        ctx->label,
        ctx->confidence
    );

    if (loss < 0.0f) {
        // Learning failed
        nimcp_error_t error = NIMCP_ERROR_OPERATION_FAILED;
        nimcp_promise_fail(ctx->promise, error);
        LOG_MODULE_ERROR("core_brain_learning", "Async learning failed for label '%s'", ctx->label);
    } else {
        // Learning succeeded - pass loss value
        nimcp_promise_complete(ctx->promise, &loss);
        LOG_MODULE_DEBUG("core_brain_learning", "Async learning completed: loss=%.4f", loss);
    }

    // Cleanup
    nimcp_free(ctx->features);
    nimcp_free(ctx->label);
    nimcp_promise_destroy(ctx->promise);
    nimcp_free(ctx);

    /* BUG FIX: Previously unconditionally threw NIMCP_THROW_TO_IMMUNE here,
     * which fired even on successful learning. Thread functions should just
     * return; errors are already propagated via the promise/future. */
    return NULL;
}

/**
 * @brief Asynchronous learning
 *
 * WHAT: Non-blocking version of brain_learn_example
 * WHY:  Enable concurrent learning without blocking caller
 * HOW:  Copy inputs, create promise/future, spawn thread
 *
 * IMPLEMENTATION NOTES:
 * - Features and label are deep-copied to ensure thread safety
 * - Promise is created with sizeof(float) for loss result
 * - Thread is detached for auto-cleanup
 * - Context is freed by worker thread
 *
 * ERROR HANDLING:
 * - Returns NULL if allocation fails or thread creation fails
 * - Learning errors propagated through future
 *
 * MEMORY MANAGEMENT:
 * - Context allocated on heap, freed by worker thread
 * - Features and label copied to heap, freed by worker thread
 * - Promise destroyed by worker thread after completion
 * - Future must be destroyed by caller
 *
 * @param brain Brain handle
 * @param features Input features (will be copied)
 * @param num_features Feature count
 * @param label Target label (will be copied)
 * @param confidence Training weight
 * @return Future handle or NULL on error
 */
nimcp_future_t nimcp_brain_learn_async(brain_t brain, const float* features,
                                        uint32_t num_features, const char* label,
                                        float confidence)
{
    // Validate parameters
    if (!brain || !features || !label) {
        LOG_MODULE_ERROR("core_brain_learning", "Invalid parameters to nimcp_brain_learn_async");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_learn_async: required parameter is NULL (brain, features, label)");
        return NULL;
    }

    if (num_features == 0) {
        LOG_MODULE_ERROR("core_brain_learning", "Invalid num_features=0");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "nimcp_brain_learn_async: num_features is zero");
        return NULL;
    }

    // Validate confidence range
    if (confidence < 0.0f || confidence > 1.0f) {
        LOG_MODULE_ERROR("core_brain_learning", "Invalid confidence=%.2f (must be 0.0-1.0)", confidence);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_learn_async: validation failed");
        return NULL;
    }

    // Allocate context for worker thread
    async_learn_context_t* ctx = nimcp_malloc(sizeof(async_learn_context_t));
    if (!ctx) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to allocate async learning context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_learn_async: ctx is NULL");
        return NULL;
    }

    // Copy features to heap (worker thread will free)
    ctx->features = nimcp_calloc(num_features, sizeof(float));
    if (!ctx->features) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to allocate features array (%u floats)", num_features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_learn_async: ctx->features is NULL");
        return NULL;
    }
    memcpy(ctx->features, features, num_features * sizeof(float));

    // Copy label to heap (worker thread will free)
    size_t label_len = strlen(label);
    ctx->label = nimcp_malloc(label_len + 1);
    if (!ctx->label) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to allocate label string");
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_learn_async: ctx->label is NULL");
        return NULL;
    }
    memcpy(ctx->label, label, label_len + 1);

    // Set context fields
    ctx->brain = brain;
    ctx->num_features = num_features;
    ctx->confidence = confidence;

    // Create promise for result (loss is a float)
    ctx->promise = nimcp_promise_create(sizeof(float));
    if (!ctx->promise) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to create promise for async learning");
        nimcp_free(ctx->label);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_learn_async: ctx->promise is NULL");
        return NULL;
    }

    // Get future before starting thread
    nimcp_future_t future = nimcp_promise_get_future(ctx->promise);
    if (!future) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to get future from promise");
        nimcp_promise_destroy(ctx->promise);
        nimcp_free(ctx->label);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_learn_async: future is NULL");
        return NULL;
    }

    // Create worker thread
    nimcp_thread_t thread;
    thread_attr_t attr = {
        .stack_size = NIMCP_THREAD_DEFAULT_STACK_SIZE,
        .priority = 0,
        .detached = true  // Auto-cleanup on exit
    };

    nimcp_result_t result = nimcp_thread_create(&thread, async_learn_thread, ctx, &attr);
    if (result != NIMCP_SUCCESS) {
        LOG_MODULE_ERROR("core_brain_learning", "Failed to create async learning thread: %d", result);
        nimcp_future_destroy(future);
        nimcp_promise_destroy(ctx->promise);
        nimcp_free(ctx->label);
        nimcp_free(ctx->features);
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_learn_async: validation failed");
        return NULL;
    }

    LOG_MODULE_INFO("core_brain_learning", "Async learning thread created successfully for label='%s'", label);

    // Return future to caller (caller must destroy)
    return future;
}
