/**
 * @file nimcp_memory_sleep_bridge.c
 * @brief Sleep-Systems Consolidation Integration Bridge Implementation
 * @version 1.0.0
 * @date 2025-12-21
 */

#include "cognitive/memory/nimcp_memory_sleep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include <string.h>

/**
 * @struct memory_sleep_bridge_struct
 * @brief Internal bridge structure
 *
 * WHAT: Contains all bridge state and configuration
 * WHY:  Encapsulate implementation details
 * HOW:  Opaque struct pattern (pimpl)
 */
struct memory_sleep_bridge_struct {
    memory_sleep_config_t config;        /* Bridge configuration */
    sleep_system_t sleep_system;          /* Non-owning pointer to sleep system */
    memory_sleep_effects_t effects;       /* Current sleep effects */
    nimcp_mutex_t* mutex;                 /* Thread safety */
    bool callback_registered;             /* Track callback registration for cleanup */

    /* Bio-async integration */
    bio_module_context_t bio_ctx;         /* Bio-async module context */
    bool bio_async_enabled;                /* Whether bio-async is active */
};

/* Forward declarations */
static void memory_on_sleep_state_change(sleep_state_t new_state, void* user_data);

/**
 * WHAT: Callback invoked when sleep state changes
 * WHY:  Immediately update consolidation parameters for new sleep state
 * HOW:  Called by sleep system via observer pattern
 *
 * BIOLOGICAL BASIS:
 * - Sharp-wave ripples in deep NREM trigger immediate consolidation
 * - Transition to REM enables semantic extraction/abstraction
 * - Awake state minimizes replay to prevent interference
 *
 * SLEEP STATE TRANSITIONS:
 * - Awake → Drowsy: Replay begins to ramp up
 * - Drowsy → Light NREM: Active hippocampal-cortical dialogue starts
 * - Light → Deep NREM: Peak consolidation window opens
 * - Deep NREM → REM: Consolidation continues, semantic extraction peaks
 * - REM → Awake: Replay stops, encoding resumes
 */
static void memory_on_sleep_state_change(sleep_state_t new_state, void* user_data)
{
    memory_sleep_bridge_t bridge = (memory_sleep_bridge_t)user_data;

    if (!bridge) {
        NIMCP_LOGGING_ERROR("NULL bridge in sleep state callback");
        return;
    }

    NIMCP_LOGGING_DEBUG("Memory bridge received sleep state: %d", new_state);

    nimcp_mutex_lock(bridge->mutex);

    bridge->effects.current_state = new_state;

    /* Update replay frequency modulation */
    if (bridge->config.enable_replay_modulation) {
        bridge->effects.replay_frequency_factor = memory_sleep_replay_for_state(new_state);
    }

    /* Update transfer rate modulation */
    if (bridge->config.enable_transfer_modulation) {
        bridge->effects.transfer_rate_factor = memory_sleep_transfer_for_state(new_state);
    }

    /* Update consolidation strength modulation */
    if (bridge->config.enable_consolidation_modulation) {
        bridge->effects.consolidation_strength_factor = memory_sleep_consolidation_for_state(new_state);
    }

    /* Update semantic extraction strength */
    if (bridge->config.enable_semantic_extraction) {
        bridge->effects.semantic_extraction_factor = memory_sleep_semantic_for_state(new_state);
    }

    /* Set replay activity flags */
    bridge->effects.replay_active = (new_state == SLEEP_STATE_LIGHT_NREM ||
                                      new_state == SLEEP_STATE_DEEP_NREM ||
                                      new_state == SLEEP_STATE_REM);

    bridge->effects.peak_consolidation = (new_state == SLEEP_STATE_DEEP_NREM);

    nimcp_mutex_unlock(bridge->mutex);

    NIMCP_LOGGING_DEBUG("Memory consolidation modulated: replay=%.2f, transfer=%.3f, "
                        "consolidation=%.2f, semantic=%.2f, peak=%d",
                        bridge->effects.replay_frequency_factor,
                        bridge->effects.transfer_rate_factor,
                        bridge->effects.consolidation_strength_factor,
                        bridge->effects.semantic_extraction_factor,
                        bridge->effects.peak_consolidation);
}

/**
 * WHAT: Initialize configuration with biological defaults
 * WHY:  Provide research-backed default parameters
 * HOW:  Enable all modulation pathways with full strength
 *
 * BIOLOGICAL DEFAULTS:
 * - All modulation enabled (matches biological systems)
 * - Full modulation strength (1.0)
 * - Semantic extraction enabled (critical for memory abstraction)
 */
int memory_sleep_default_config(memory_sleep_config_t* config)
{
    if (!config) return -1;

    config->enable_replay_modulation = true;
    config->enable_transfer_modulation = true;
    config->enable_consolidation_modulation = true;
    config->enable_semantic_extraction = true;
    config->modulation_strength = 1.0f;

    return 0;
}

/**
 * WHAT: Create and initialize memory-sleep bridge
 * WHY:  Connect sleep system to consolidation modulation
 * HOW:  Allocate structure, register callback, get initial state
 *
 * INITIALIZATION SEQUENCE:
 * 1. Allocate bridge structure
 * 2. Copy/default configuration
 * 3. Initialize effects to awake state
 * 4. Create mutex for thread safety
 * 5. Register sleep state change callback
 * 6. Get initial sleep state and compute effects
 *
 * ERROR HANDLING:
 * - Returns NULL on allocation failure
 * - Logs warning if callback registration fails (falls back to polling)
 * - Continues even if callback fails (graceful degradation)
 */
memory_sleep_bridge_t memory_sleep_bridge_create(
    const memory_sleep_config_t* config,
    sleep_system_t sleep)
{
    if (!sleep) return NULL;

    struct memory_sleep_bridge_struct* bridge =
        (struct memory_sleep_bridge_struct*)nimcp_malloc(
            sizeof(struct memory_sleep_bridge_struct));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(struct memory_sleep_bridge_struct));

    /* Copy or default configuration */
    if (config) {
        bridge->config = *config;
    } else {
        memory_sleep_default_config(&bridge->config);
    }

    bridge->sleep_system = sleep;

    /* Initialize effects to awake state defaults */
    bridge->effects.replay_frequency_factor = MEMORY_SLEEP_REPLAY_AWAKE;
    bridge->effects.transfer_rate_factor = MEMORY_SLEEP_TRANSFER_AWAKE;
    bridge->effects.consolidation_strength_factor = MEMORY_SLEEP_CONSOLIDATION_AWAKE;
    bridge->effects.semantic_extraction_factor = MEMORY_SLEEP_SEMANTIC_AWAKE;
    bridge->effects.current_state = SLEEP_STATE_AWAKE;
    bridge->effects.sleep_pressure = 0.0f;
    bridge->effects.replay_active = false;
    bridge->effects.peak_consolidation = false;

    /* Create thread safety mutex */
    bridge->mutex = nimcp_platform_mutex_create();
    if (!bridge->mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Register callback for automatic state updates */
    bridge->callback_registered = sleep_register_state_callback(
        sleep,
        memory_on_sleep_state_change,
        bridge);

    if (!bridge->callback_registered) {
        NIMCP_LOGGING_WARN("Failed to register sleep state callback - will use polling");
    } else {
        NIMCP_LOGGING_DEBUG("Registered sleep state callback for memory bridge");
    }

    /* Get initial state immediately */
    sleep_state_t initial_state = sleep_get_current_state(sleep);
    memory_on_sleep_state_change(initial_state, bridge);

    NIMCP_LOGGING_INFO("Memory-sleep bridge created");
    return bridge;
}

/**
 * WHAT: Destroy bridge and free resources
 * WHY:  Prevent memory leaks and dangling callbacks
 * HOW:  Unregister callback, free mutex, free structure
 *
 * CLEANUP SEQUENCE:
 * 1. Check for NULL (safe to call on NULL)
 * 2. Unregister sleep state callback if registered
 * 3. Disconnect bio-async if connected
 * 4. Destroy mutex
 * 5. Free bridge structure
 *
 * THREAD-SAFE: Yes (no mutex needed - bridge being destroyed)
 */
void memory_sleep_bridge_destroy(memory_sleep_bridge_t bridge)
{
    if (!bridge) return;

    /* Unregister callback if it was registered */
    if (bridge->callback_registered && bridge->sleep_system) {
        bool unregistered = sleep_unregister_state_callback(
            bridge->sleep_system,
            memory_on_sleep_state_change,
            bridge);

        if (unregistered) {
            NIMCP_LOGGING_DEBUG("Unregistered sleep state callback for memory bridge");
        }
    }

    /* Disconnect bio-async if connected */
    if (bridge->bio_async_enabled) {
        memory_sleep_disconnect_bio_async(bridge);
    }

    if (bridge->mutex) {
        nimcp_mutex_destroy(bridge->mutex);
    }

    nimcp_free(bridge);
}

/**
 * WHAT: Manually update bridge effects from current sleep state
 * WHY:  Support polling mode if callback fails
 * HOW:  Query sleep system, recompute all effects
 *
 * POLLING MODE:
 * - Use this if callback registration fails
 * - Call periodically from consolidation system update loop
 * - Less efficient than callback but functionally equivalent
 *
 * UPDATES:
 * - Current sleep state
 * - Sleep pressure (affects consolidation even when awake)
 * - All modulation factors
 * - Activity flags
 *
 * THREAD-SAFE: Yes (mutex protected)
 */
int memory_sleep_update(memory_sleep_bridge_t bridge)
{
    if (!bridge) return -1;

    nimcp_mutex_lock(bridge->mutex);

    sleep_state_t state = sleep_get_current_state(bridge->sleep_system);
    float pressure = sleep_get_pressure(bridge->sleep_system);

    bridge->effects.current_state = state;
    bridge->effects.sleep_pressure = pressure;

    /* Update replay frequency */
    if (bridge->config.enable_replay_modulation) {
        float replay_base = memory_sleep_replay_for_state(state);
        bridge->effects.replay_frequency_factor = replay_base;

        /* High sleep pressure increases replay even when awake */
        if (state == SLEEP_STATE_AWAKE && pressure > 0.7f) {
            bridge->effects.replay_frequency_factor *= (1.0f + (pressure - 0.7f) / 0.3f);
        }
    }

    /* Update transfer rate */
    if (bridge->config.enable_transfer_modulation) {
        bridge->effects.transfer_rate_factor = memory_sleep_transfer_for_state(state);
    }

    /* Update consolidation strength */
    if (bridge->config.enable_consolidation_modulation) {
        bridge->effects.consolidation_strength_factor = memory_sleep_consolidation_for_state(state);
    }

    /* Update semantic extraction */
    if (bridge->config.enable_semantic_extraction) {
        bridge->effects.semantic_extraction_factor = memory_sleep_semantic_for_state(state);
    }

    /* Update activity flags */
    bridge->effects.replay_active = (state == SLEEP_STATE_LIGHT_NREM ||
                                      state == SLEEP_STATE_DEEP_NREM ||
                                      state == SLEEP_STATE_REM);

    bridge->effects.peak_consolidation = (state == SLEEP_STATE_DEEP_NREM);

    nimcp_mutex_unlock(bridge->mutex);
    return 0;
}

/**
 * WHAT: Retrieve current sleep effects on consolidation
 * WHY:  Consolidation system needs modulation parameters
 * HOW:  Thread-safe copy of effects structure
 *
 * USAGE PATTERN:
 * memory_sleep_effects_t effects;
 * memory_sleep_get_effects(bridge, &effects);
 * float effective_replay_hz = base_hz * effects.replay_frequency_factor;
 * float effective_transfer = base_rate * effects.transfer_rate_factor;
 *
 * THREAD-SAFE: Yes (mutex protected read)
 */
int memory_sleep_get_effects(
    const memory_sleep_bridge_t bridge,
    memory_sleep_effects_t* effects)
{
    if (!bridge || !effects) return -1;

    nimcp_mutex_lock(bridge->mutex);
    *effects = bridge->effects;
    nimcp_mutex_unlock(bridge->mutex);

    return 0;
}

/**
 * WHAT: Get current replay frequency factor
 * WHY:  Convenience function for most common query
 * HOW:  Thread-safe read of single field
 *
 * USAGE:
 * float factor = memory_sleep_get_replay_frequency(bridge);
 * float replay_hz = 10.0f * factor;  // Scale base replay rate
 *
 * THREAD-SAFE: Yes (mutex protected)
 */
float memory_sleep_get_replay_frequency(const memory_sleep_bridge_t bridge)
{
    if (!bridge) return 1.0f;

    nimcp_mutex_lock(bridge->mutex);
    float result = bridge->effects.replay_frequency_factor;
    nimcp_mutex_unlock(bridge->mutex);

    return result;
}

/**
 * WHAT: Check if memory replay is currently active
 * WHY:  Consolidation system needs to know when to trigger replay
 * HOW:  Returns activity flag (true during NREM/REM)
 *
 * BIOLOGICAL BASIS:
 * - Replay is active during all sleep stages (NREM and REM)
 * - Minimal replay during drowsiness (transitional)
 * - Rare spontaneous replay when awake
 *
 * USAGE:
 * if (memory_sleep_is_replay_active(bridge)) {
 *     systems_consolidation_execute_replays(system, dt, is_sws, is_rem);
 * }
 *
 * THREAD-SAFE: Yes (mutex protected)
 */
bool memory_sleep_is_replay_active(const memory_sleep_bridge_t bridge)
{
    if (!bridge) return false;

    nimcp_mutex_lock(bridge->mutex);
    bool result = bridge->effects.replay_active;
    nimcp_mutex_unlock(bridge->mutex);

    return result;
}

/**
 * WHAT: Lookup replay frequency for given sleep state
 * WHY:  Enable state-specific logic and testing
 * HOW:  Switch on state, return constant
 *
 * BIOLOGICAL VALUES:
 * - Awake: 0.1 Hz (rare spontaneous replay)
 * - Drowsy: 0.5 Hz (preparation for sleep)
 * - Light NREM: 5.0 Hz (hippocampal-cortical dialogue)
 * - Deep NREM: 10.0 Hz (sharp-wave ripples, peak consolidation)
 * - REM: 3.0 Hz (integration/abstraction)
 *
 * STATELESS: Pure function (no side effects, no state mutation)
 */
float memory_sleep_replay_for_state(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return MEMORY_SLEEP_REPLAY_AWAKE;
        case SLEEP_STATE_DROWSY:     return MEMORY_SLEEP_REPLAY_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return MEMORY_SLEEP_REPLAY_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return MEMORY_SLEEP_REPLAY_DEEP_NREM;
        case SLEEP_STATE_REM:        return MEMORY_SLEEP_REPLAY_REM;
        default:                     return MEMORY_SLEEP_REPLAY_AWAKE;
    }
}

/**
 * WHAT: Lookup transfer rate for given sleep state
 * WHY:  Enable state-specific logic and testing
 * HOW:  Switch on state, return constant
 *
 * BIOLOGICAL VALUES:
 * - Awake: 0.001 (0.1% per hour)
 * - Drowsy: 0.005 (0.5% per hour)
 * - Light NREM: 0.03 (3% per hour)
 * - Deep NREM: 0.05 (5% per hour, peak transfer)
 * - REM: 0.02 (2% per hour)
 *
 * REFERENCE: McClelland et al. (1995) - Complementary learning systems theory
 *
 * STATELESS: Pure function (no side effects, no state mutation)
 */
float memory_sleep_transfer_for_state(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return MEMORY_SLEEP_TRANSFER_AWAKE;
        case SLEEP_STATE_DROWSY:     return MEMORY_SLEEP_TRANSFER_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return MEMORY_SLEEP_TRANSFER_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return MEMORY_SLEEP_TRANSFER_DEEP_NREM;
        case SLEEP_STATE_REM:        return MEMORY_SLEEP_TRANSFER_REM;
        default:                     return MEMORY_SLEEP_TRANSFER_AWAKE;
    }
}

/**
 * WHAT: Lookup consolidation strength for given sleep state
 * WHY:  Enable state-specific logic and testing
 * HOW:  Switch on state, return constant
 *
 * BIOLOGICAL VALUES:
 * - Awake: 0.1 (minimal consolidation)
 * - Drowsy: 0.2 (slightly increased)
 * - Light NREM: 0.7 (active consolidation)
 * - Deep NREM: 1.0 (peak consolidation)
 * - REM: 0.8 (continued consolidation + integration)
 *
 * REFERENCE: Born & Wilhelm (2012) - System consolidation during sleep
 *
 * STATELESS: Pure function (no side effects, no state mutation)
 */
float memory_sleep_consolidation_for_state(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return MEMORY_SLEEP_CONSOLIDATION_AWAKE;
        case SLEEP_STATE_DROWSY:     return MEMORY_SLEEP_CONSOLIDATION_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return MEMORY_SLEEP_CONSOLIDATION_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return MEMORY_SLEEP_CONSOLIDATION_DEEP_NREM;
        case SLEEP_STATE_REM:        return MEMORY_SLEEP_CONSOLIDATION_REM;
        default:                     return MEMORY_SLEEP_CONSOLIDATION_AWAKE;
    }
}

/**
 * WHAT: Lookup semantic extraction strength for given sleep state
 * WHY:  Enable state-specific logic and testing
 * HOW:  Switch on state, return constant
 *
 * BIOLOGICAL VALUES:
 * - Awake: 0.1 (minimal abstraction)
 * - Drowsy: 0.2 (slightly increased)
 * - Light NREM: 0.5 (moderate abstraction)
 * - Deep NREM: 0.7 (strong abstraction)
 * - REM: 1.0 (peak semantic extraction/generalization)
 *
 * BIOLOGICAL BASIS:
 * - REM sleep is optimal for semantic abstraction and schema formation
 * - Theta oscillations (4-8 Hz) in REM enable pattern extraction
 * - Deep NREM enables consolidation but less abstraction than REM
 * - Awake state preserves episodic details, minimal semantic transformation
 *
 * REFERENCE: Winocur & Moscovitch (2011) - Memory transformation and systems consolidation
 *
 * STATELESS: Pure function (no side effects, no state mutation)
 */
float memory_sleep_semantic_for_state(sleep_state_t state)
{
    switch (state) {
        case SLEEP_STATE_AWAKE:      return MEMORY_SLEEP_SEMANTIC_AWAKE;
        case SLEEP_STATE_DROWSY:     return MEMORY_SLEEP_SEMANTIC_DROWSY;
        case SLEEP_STATE_LIGHT_NREM: return MEMORY_SLEEP_SEMANTIC_LIGHT_NREM;
        case SLEEP_STATE_DEEP_NREM:  return MEMORY_SLEEP_SEMANTIC_DEEP_NREM;
        case SLEEP_STATE_REM:        return MEMORY_SLEEP_SEMANTIC_REM;
        default:                     return MEMORY_SLEEP_SEMANTIC_AWAKE;
    }
}

/**
 * WHAT: Connect bridge to bio-async router
 * WHY:  Enable inter-module messaging for distributed memory signals
 * HOW:  Register with bio_router using BIO_MODULE_MEMORY_SLEEP
 *
 * BIOLOGICAL BASIS:
 * - Sleep-dependent memory consolidation signals can be broadcast
 * - Allows coordination with other sleep-sensitive modules
 * - Enables memory replay notifications to learning systems
 */
int memory_sleep_connect_bio_async(memory_sleep_bridge_t bridge)
{
    /* Guard clauses */
    if (!bridge) return -1;
    if (bridge->bio_async_enabled) return 0;  /* Already connected */

    /* Register with bio-async router */
    bio_module_info_t info = {
        .module_id = BIO_MODULE_MEMORY_SLEEP,
        .module_name = "memory_sleep_bridge",
        .inbox_capacity = 32,
        .user_data = bridge
    };

    bridge->bio_ctx = bio_router_register_module(&info);
    if (bridge->bio_ctx) {
        bridge->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Memory-sleep bridge connected to bio-async router");
        return 0;
    }

    NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    return 0;  /* Not an error if router unavailable */
}

/**
 * WHAT: Disconnect from bio-async router
 * WHY:  Clean shutdown of messaging
 * HOW:  Unregister from bio_router
 *
 * CLEANUP:
 * - Unregisters module from router
 * - Clears bio context
 * - Disables bio-async flag
 */
int memory_sleep_disconnect_bio_async(memory_sleep_bridge_t bridge)
{
    /* Guard clauses */
    if (!bridge) return -1;
    if (!bridge->bio_async_enabled) return 0;  /* Not connected */

    /* Unregister from bio-async router */
    if (bridge->bio_ctx) {
        bio_router_unregister_module(bridge->bio_ctx);
        bridge->bio_ctx = NULL;
    }

    bridge->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("Memory-sleep bridge disconnected from bio-async router");

    return 0;
}

/**
 * WHAT: Check if bio-async is connected
 * WHY:  Allow conditional bio-async usage
 * HOW:  Return bio_async_enabled flag
 *
 * THREAD-SAFE: Yes (reads single boolean flag)
 *
 * USE CASE:
 * - Check before sending bio-async messages
 * - Determine if inter-module communication is available
 */
bool memory_sleep_is_bio_async_connected(const memory_sleep_bridge_t bridge)
{
    /* Guard clause */
    if (!bridge) return false;

    return bridge->bio_async_enabled;
}
