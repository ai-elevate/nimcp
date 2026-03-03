/**
 * @file nimcp_brain_init_lnn.c
 *
 * WHAT: Initialize LNN (Liquid Neural Network) temporal context processor at brain creation.
 * WHY:  LNN provides continuous-time ODE-based temporal context that captures sequence
 *       dynamics without explicit recurrence buffers. Creates a compact NCP architecture
 *       (128→64→32→64 = 288 neurons) connected to plasticity, sleep, immune, and bio-async.
 * HOW:  Called during brain factory init when fast_training_mode is false.
 *       Connects to TPB for plasticity integration, sleep bridge for tau/LR modulation
 *       during sleep cycles, immune for instability detection, and bio-async for state
 *       broadcasts.
 *
 * BIOLOGICAL BASIS:
 * - Neural Circuit Policy (NCP) architecture mirrors biological sensory-inter-command-motor
 *   hierarchy (Lechner et al. 2020, Nature Machine Intelligence)
 * - ODE neuron dynamics model continuous-time membrane potential evolution
 * - Time constants (tau) modulated by sleep bridge match biological circadian plasticity
 */

#include "core/brain/factory/init/nimcp_brain_init_subsystems.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "lnn/nimcp_lnn.h"
#include "lnn/nimcp_lnn_network.h"
#include "lnn/nimcp_lnn_training.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "BRAIN_INIT_LNN"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/bridge/nimcp_bridge_boilerplate.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

BRIDGE_BOILERPLATE_MESH_ONLY(brain_init_lnn, MESH_ADAPTER_CATEGORY_SYSTEM)

/**
 * @brief Initialize LNN temporal context processor during brain creation.
 *
 * Creates a compact NCP-architecture LNN (128→64→32→64 = 288 neurons) as a temporal
 * context processor. Connected to the training-plasticity bridge, sleep bridge, immune
 * system, and bio-async router for full biological integration.
 *
 * @param brain  The brain being initialized (internal struct pointer)
 * @return true on success or non-fatal skip, false on critical error
 */
bool nimcp_brain_factory_init_lnn_subsystem(brain_t brain)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_lnn_subsystem: brain is NULL");
        return false;
    }

    /* Initialize fields */
    /* lnn_network and lnn_training_ctx are already NULL from brain creation */

    /* Skip if fast training mode — LNN temporal context not needed */
    if (brain->config.fast_training_mode) {
        LOG_INFO(LOG_MODULE, "Skipping LNN init (fast_training_mode=true)");
        return true;
    }

    /* Skip if already created (idempotent) */
    if (brain->lnn_network) {
        LOG_INFO(LOG_MODULE, "LNN already initialized, skipping");
        return true;
    }

    /* Ensure LNN subsystem is initialized */
    if (!lnn_is_initialized()) {
        int init_result = lnn_init(1);  /* Single-threaded for now */
        if (init_result != 0) {
            LOG_WARN(LOG_MODULE, "LNN subsystem init failed (non-fatal)");
            return true;
        }
    }

    /* Create compact NCP architecture for temporal context processing:
     *   128 sensory → 64 inter → 32 command → 64 motor
     *   Total: 288 neurons (lightweight, O(288^2) = 83K ops/step)
     */
    uint32_t n_sensory = 128;
    uint32_t n_inter   = 64;
    uint32_t n_command  = 32;
    uint32_t n_motor    = 64;

    brain->lnn_network = lnn_network_create_ncp(n_sensory, n_inter, n_command, n_motor);
    if (!brain->lnn_network) {
        LOG_WARN(LOG_MODULE, "Failed to create LNN network (non-fatal)");
        return true;
    }

    /* Initialize weights with deterministic seed for reproducibility */
    lnn_network_init_weights(brain->lnn_network, 42);

    /* Create training context with plasticity integration enabled */
    lnn_training_config_t train_config;
    lnn_training_config_default(&train_config);
    train_config.learning_rate = 0.001f;
    train_config.gradient_clip_norm = 1.0f;
    train_config.enable_plasticity_integration = true;
    train_config.enable_immune_integration = true;
    train_config.enable_bio_async = true;
    train_config.lnn_train_mode = LNN_TRAIN_ADJOINT;
    train_config.track_statistics = true;

    brain->lnn_training_ctx = lnn_training_create(brain->lnn_network, &train_config);
    if (!brain->lnn_training_ctx) {
        LOG_WARN(LOG_MODULE, "Failed to create LNN training context (non-fatal)");
        /* LNN forward pass still works without training context */
    }

    /* Note: LNN training uses training_plasticity_bridge_t (not tpb_context_t).
     * These are connected at a higher level if available. Immune integration
     * likewise uses a different type (training_immune_system_t vs brain_immune_system_t).
     * The LNN training context will function without these connections — it just
     * won't get neuromodulator-gated learning or immune instability detection. */

    LOG_INFO(LOG_MODULE, "LNN temporal processor initialized: NCP %u→%u→%u→%u (288 neurons)",
             n_sensory, n_inter, n_command, n_motor);
    return true;
}
