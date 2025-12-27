/**
 * @file nimcp_astrocytes.c
 * @brief Implementation of biological astrocyte glial cells
 *
 * STATUS: Bio-async integrated - event-driven calcium wave coordination
 * FEATURES: Glutamate/D-serine release via predictive signals, glial wave API
 */

#include "glial/astrocytes/nimcp_astrocytes.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/spatial/nimcp_kdtree.h"
#include "utils/thread/nimcp_atomic.h"
#include "utils/platform/nimcp_platform_once.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
// Global Bio-Async Context
//=============================================================================

static bio_module_context_t g_astrocyte_bio_ctx = NULL;
static unified_mem_manager_t g_astrocyte_mem_mgr = NULL;
static nimcp_atomic_bool_t g_astrocyte_bio_initialized = { .value = false };
static nimcp_platform_once_t g_astrocyte_bio_once = NIMCP_PLATFORM_ONCE_INIT;
static nimcp_error_t g_astrocyte_bio_init_result = NIMCP_SUCCESS;

//=============================================================================
// Bio-Async Message Handlers
//=============================================================================

/**
 * @brief Handle BIO_MSG_ASTROCYTE_CALCIUM_WAVE message
 *
 * Initiates a calcium wave from the specified region with given initial concentration.
 * Uses glial wave API for slow system-wide coordination.
 */
static nimcp_error_t handle_calcium_wave_message(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    if (!msg || msg_size < sizeof(bio_message_header_t)) {
        LOG_MODULE_ERROR("ASTROCYTE", "Invalid calcium wave message: msg=%p, size=%zu",
                         msg, msg_size);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;

    // Parse message payload (expecting region ID and calcium level)
    if (msg_size < sizeof(bio_message_header_t) + sizeof(uint32_t) + sizeof(float)) {
        LOG_MODULE_ERROR("ASTROCYTE", "Calcium wave message too small: size=%zu", msg_size);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const uint8_t* payload = (const uint8_t*)msg + sizeof(bio_message_header_t);
    uint32_t source_region = *(const uint32_t*)payload;
    float initial_calcium = *(const float*)(payload + sizeof(uint32_t));

    LOG_MODULE_INFO("ASTROCYTE", "Initiating calcium wave from region %u, calcium=%.2f μM",
                    source_region, initial_calcium);

    // Initiate glial wave for slow system-wide coordination
    // Note: nimcp_glial_wave_initiate would need to be implemented in bio_async
    // For now, we publish a predictive signal
    nimcp_error_t result = bio_router_publish_signal(g_astrocyte_bio_ctx,
        "astrocyte.calcium_wave", initial_calcium);

    if (result == NIMCP_SUCCESS) {
        LOG_MODULE_DEBUG("ASTROCYTE", "Calcium wave signal published successfully");
    } else {
        LOG_MODULE_WARN("ASTROCYTE", "Failed to publish calcium wave signal: error=%d", result);
    }

    return result;
}

/**
 * @brief Handle BIO_MSG_ASTROCYTE_GLUTAMATE_UPTAKE message
 *
 * Processes glutamate uptake request and updates internal glutamate pool.
 */
static nimcp_error_t handle_glutamate_uptake_message(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    if (!msg || msg_size < sizeof(bio_message_header_t)) {
        LOG_MODULE_ERROR("ASTROCYTE", "Invalid glutamate uptake message");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;

    // Parse uptake amount
    if (msg_size < sizeof(bio_message_header_t) + sizeof(float)) {
        LOG_MODULE_ERROR("ASTROCYTE", "Glutamate uptake message too small");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const uint8_t* payload = (const uint8_t*)msg + sizeof(bio_message_header_t);
    float uptake_amount = *(const float*)payload;

    LOG_MODULE_DEBUG("ASTROCYTE", "Processing glutamate uptake: amount=%.4f", uptake_amount);

    // Publish uptake event
    nimcp_error_t result = bio_router_publish_signal(g_astrocyte_bio_ctx,
        "astrocyte.glutamate_uptake", uptake_amount);

    return result;
}

/**
 * @brief Handle BIO_MSG_METABOLIC_DEMAND message
 *
 * Processes metabolic demand from neurons and responds with metabolic supply.
 * Astrocytes provide glucose and lactate to neurons for ATP production.
 */
static nimcp_error_t handle_metabolic_demand_message(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    if (!msg || msg_size < sizeof(bio_msg_metabolic_demand_t)) {
        LOG_MODULE_ERROR("ASTROCYTE", "Invalid metabolic demand message");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const bio_msg_metabolic_demand_t* demand = (const bio_msg_metabolic_demand_t*)msg;

    LOG_MODULE_DEBUG("ASTROCYTE", "Metabolic demand from region %u: glucose=%.2f, oxygen=%.2f, ATP deficit=%.2f, urgency=%.2f",
                     demand->region_id, demand->glucose_demand, demand->oxygen_demand,
                     demand->atp_deficit, demand->urgency);

    // Calculate supply based on current ATP level and demand urgency
    // High urgency demands get priority allocation
    float supply_factor = 1.0F; // This would be calculated based on astrocyte ATP level
    float glucose_supply = demand->glucose_demand * supply_factor;
    float oxygen_supply = demand->oxygen_demand * supply_factor;

    // Publish metabolic supply response via SEROTONIN (slow, metabolic coordination)
    bio_router_publish_signal(g_astrocyte_bio_ctx, "astrocyte.metabolic_supply", supply_factor);

    // If urgency is high, also publish via NOREPINEPHRINE (alert channel)
    if (demand->urgency > 0.7F) {
        LOG_MODULE_WARN("ASTROCYTE", "High urgency metabolic demand (%.2f) - escalating priority", demand->urgency);
        bio_router_publish_signal(g_astrocyte_bio_ctx, "astrocyte.metabolic_alert", demand->urgency);
    }

    return NIMCP_SUCCESS;
}

/**
 * @brief Handle BIO_MSG_GLIAL_SYNC_REQUEST message
 *
 * Handles synchronization requests for coordinated glial cell activity.
 * Used to coordinate calcium waves, metabolic support, and homeostatic adjustments.
 */
static nimcp_error_t handle_glial_sync_message(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{
    if (!msg || msg_size < sizeof(bio_message_header_t)) {
        LOG_MODULE_ERROR("ASTROCYTE", "Invalid glial sync message");
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;

    LOG_MODULE_DEBUG("ASTROCYTE", "Glial sync request received from module %u", header->source_module);

    // Publish sync acknowledgment - this helps coordinate timing across glial cells
    nimcp_error_t result = bio_router_publish_signal(g_astrocyte_bio_ctx,
        "astrocyte.sync_ack", 1.0F);

    // Respond to sync request if response promise is provided
    if (response_promise) {
        // TODO: Could send back current state (calcium level, ATP, etc.)
        LOG_MODULE_DEBUG("ASTROCYTE", "Sync acknowledgment sent");
    }

    return result;
}

/**
 * @brief Internal initialization function called by nimcp_platform_once
 *
 * Thread-safe initialization of bio-async integration.
 * Result is stored in g_astrocyte_bio_init_result for callers to check.
 */
static void astrocyte_bio_init_once_impl(void)
{
    // Initialize unified memory manager
    unified_mem_config_t mem_config = unified_mem_default_config();
    g_astrocyte_mem_mgr = unified_mem_create(&mem_config);
    if (!g_astrocyte_mem_mgr) {
        LOG_MODULE_ERROR("ASTROCYTE", "Failed to create unified memory manager");
        g_astrocyte_bio_init_result = NIMCP_ERROR_MEMORY;
        return;
    }

    // Register module with bio-router
    bio_module_info_t info = {
        .module_id = BIO_MODULE_ASTROCYTE,
        .module_name = "Astrocyte",
        .inbox_capacity = 256,
        .user_data = NULL
    };

    g_astrocyte_bio_ctx = bio_router_register_module(&info);
    if (!g_astrocyte_bio_ctx) {
        LOG_MODULE_ERROR("ASTROCYTE", "Failed to register with bio-router");
        unified_mem_destroy(g_astrocyte_mem_mgr);
        g_astrocyte_mem_mgr = NULL;
        g_astrocyte_bio_init_result = NIMCP_ERROR_INVALID_PARAM;
        return;
    }

    // Register message handlers
    nimcp_error_t result;

    result = bio_router_register_handler(g_astrocyte_bio_ctx,
                                          BIO_MSG_ASTROCYTE_CALCIUM_WAVE,
                                          handle_calcium_wave_message);
    if (result != NIMCP_SUCCESS) {
        LOG_MODULE_ERROR("ASTROCYTE", "Failed to register calcium wave handler: %d", result);
        goto cleanup;
    }

    result = bio_router_register_handler(g_astrocyte_bio_ctx,
                                          BIO_MSG_ASTROCYTE_GLUTAMATE_UPTAKE,
                                          handle_glutamate_uptake_message);
    if (result != NIMCP_SUCCESS) {
        LOG_MODULE_ERROR("ASTROCYTE", "Failed to register glutamate uptake handler: %d", result);
        goto cleanup;
    }

    result = bio_router_register_handler(g_astrocyte_bio_ctx,
                                          BIO_MSG_METABOLIC_DEMAND,
                                          handle_metabolic_demand_message);
    if (result != NIMCP_SUCCESS) {
        LOG_MODULE_ERROR("ASTROCYTE", "Failed to register metabolic demand handler: %d", result);
        goto cleanup;
    }

    result = bio_router_register_handler(g_astrocyte_bio_ctx,
                                          BIO_MSG_GLIAL_SYNC_REQUEST,
                                          handle_glial_sync_message);
    if (result != NIMCP_SUCCESS) {
        LOG_MODULE_ERROR("ASTROCYTE", "Failed to register glial sync handler: %d", result);
        goto cleanup;
    }

    nimcp_atomic_store_bool(&g_astrocyte_bio_initialized, true, NIMCP_MEMORY_ORDER_RELEASE);
    g_astrocyte_bio_init_result = NIMCP_SUCCESS;
    LOG_MODULE_INFO("ASTROCYTE", "Bio-async integration initialized successfully (4 handlers registered)");
    return;

cleanup:
    bio_router_unregister_module(g_astrocyte_bio_ctx);
    g_astrocyte_bio_ctx = NULL;
    unified_mem_destroy(g_astrocyte_mem_mgr);
    g_astrocyte_mem_mgr = NULL;
    g_astrocyte_bio_init_result = result;
}

/**
 * @brief Initialize bio-async integration for astrocytes module (thread-safe)
 *
 * Uses nimcp_platform_once to ensure initialization runs exactly once
 * across all threads calling this function.
 */
static nimcp_error_t astrocyte_bio_init(void)
{
    nimcp_platform_once(&g_astrocyte_bio_once, astrocyte_bio_init_once_impl);
    return g_astrocyte_bio_init_result;
}

/**
 * @brief Shutdown bio-async integration
 */
static void astrocyte_bio_shutdown(void)
{
    if (!nimcp_atomic_load_bool(&g_astrocyte_bio_initialized, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        return;
    }

    LOG_MODULE_INFO("ASTROCYTE", "Shutting down bio-async integration");

    if (g_astrocyte_bio_ctx) {
        bio_router_unregister_module(g_astrocyte_bio_ctx);
        g_astrocyte_bio_ctx = NULL;
    }

    if (g_astrocyte_mem_mgr) {
        unified_mem_destroy(g_astrocyte_mem_mgr);
        g_astrocyte_mem_mgr = NULL;
    }

    nimcp_atomic_store_bool(&g_astrocyte_bio_initialized, false, NIMCP_MEMORY_ORDER_RELEASE);
}

//=============================================================================
// Public Bio-Async API
//=============================================================================

nimcp_result_t astrocyte_register_bio_handlers(void)
{
    LOG_MODULE_INFO("ASTROCYTE", "Registering bio-async message handlers");
    return astrocyte_bio_init();
}

void astrocyte_unregister_bio_handlers(void)
{
    LOG_MODULE_INFO("ASTROCYTE", "Unregistering bio-async message handlers");
    astrocyte_bio_shutdown();
}

//=============================================================================
// Creation and Destruction
//=============================================================================

astrocyte_t* astrocyte_create(uint32_t id, astrocyte_type_t type, float x, float y, float z, float coverage_radius)
{
    /* Initialize bio-async on first create (thread-safe via nimcp_platform_once) */
    nimcp_error_t init_result = astrocyte_bio_init();
    if (init_result != NIMCP_SUCCESS) {
        LOG_MODULE_WARN("ASTROCYTE", "Bio-async init failed: %d (continuing anyway)", init_result);
    }

    /* Validate spatial coordinates are finite (not NaN or Inf) */
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
        LOG_MODULE_ERROR("ASTROCYTE", "Invalid spatial coordinates: (%.2f, %.2f, %.2f)", x, y, z);
        return NULL;
    }

    /* Validate coverage radius is positive and finite */
    if (coverage_radius < 0.0F || !isfinite(coverage_radius)) {
        LOG_MODULE_ERROR("ASTROCYTE", "Invalid coverage radius: %.2f", coverage_radius);
        return NULL;
    }

    astrocyte_t* astro = (astrocyte_t*) nimcp_malloc(sizeof(astrocyte_t));
    if (!astro) {
        LOG_MODULE_ERROR("ASTROCYTE", "Failed to allocate astrocyte structure");
        return NULL;
    }

    memset(astro, 0, sizeof(astrocyte_t));

    // Identity
    astro->id = id;
    astro->type = type;

    // Calcium dynamics - initialize to resting baseline
    astro->calcium_concentration = ASTROCYTE_BASELINE_CALCIUM_UM;
    astro->calcium_baseline = ASTROCYTE_BASELINE_CALCIUM_UM;
    astro->ip3_concentration = 0.0F;
    astro->last_calcium_spike = 0;

    // Neurotransmitter pools - initialize to full
    astro->glutamate_pool = 1.0F;
    astro->d_serine_pool = 1.0F;
    astro->atp_level = 1.0F;

    // Spatial location
    astro->position[0] = x;
    astro->position[1] = y;
    astro->position[2] = z;
    astro->coverage_radius = coverage_radius;

    // Synaptic coverage - initially empty
    astro->num_covered_synapses = 0;
    astro->covered_synapse_ids = NULL;
    astro->synapse_calcium_levels = NULL;

    // Gap junction coupling - initially empty
    astro->num_coupled_astrocytes = 0;
    astro->coupled_astrocyte_ids = NULL;
    astro->coupling_strengths = NULL;

    // Homeostatic regulation
    astro->target_activity_level = 0.3F; // Default target
    astro->scaling_factor = 1.0F;

    // Thread safety
    nimcp_spinlock_init(&astro->lock);

    return astro;
}

void astrocyte_destroy(astrocyte_t* astro)
{
    if (!astro) {
        return;
    }

    // Free synapse arrays
    if (astro->covered_synapse_ids) {
        nimcp_free(astro->covered_synapse_ids);
    }
    if (astro->synapse_calcium_levels) {
        nimcp_free(astro->synapse_calcium_levels);
    }

    // Free coupling arrays
    if (astro->coupled_astrocyte_ids) {
        nimcp_free(astro->coupled_astrocyte_ids);
    }
    if (astro->coupling_strengths) {
        nimcp_free(astro->coupling_strengths);
    }

    // Destroy spinlock
    nimcp_spinlock_destroy(&astro->lock);

    // Free astrocyte itself
    nimcp_free(astro);
}

//=============================================================================
// Calcium Dynamics - STUB IMPLEMENTATION
//=============================================================================

void astrocyte_update_calcium(astrocyte_t* astro, float dt, float external_stimulus)
{
    if (!astro || dt < 0.0F) {
        return;
    }

    // Li-Rinzel calcium dynamics model
    // d[Ca²⁺]/dt = J_channel + J_leak - J_pump - J_ER
    //
    // BIOLOGICAL PARAMETERS:
    // - v1 = 6.0: IP3R channel max flux
    // - v2 = 0.11: Leak flux coefficient
    // - v3 = 2.2: Pump max flux
    // - k1 = 0.3 µM: Ca²⁺ activation for IP3R
    // - k2 = 0.1 µM: Ca²⁺ half-max for pump
    // - k3 = 0.5 µM: IP3 half-max for IP3R

    nimcp_spinlock_lock(&astro->lock);

    // Model parameters - TUNED for faster decay to baseline
    const float v1 = 6.0F;   // IP3R channel coefficient
    const float v2 = 0.11F;  // Leak coefficient
    const float v3 = 8.0F;   // Pump coefficient - DOUBLED from 4.0 for even faster decay
    const float k1 = 0.3F;   // Ca²⁺ activation constant (µM)
    const float k2 = 0.1F;   // Ca²⁺ pump half-max (µM)
    const float k3 = 0.5F;   // IP3 half-max (µM)

    // ER calcium store (assumed constant for simplified model)
    const float ca_er = 10.0F; // µM

    // Additional decay term to ensure return to baseline
    const float baseline_decay_rate = 5.0F; // Direct exponential decay to baseline - INCREASED

    // Current state
    float ca = astro->calcium_concentration;
    float ip3 = astro->ip3_concentration;

    // IP3 dynamics: production from stimulus, degradation
    const float ip3_degradation_rate = 1.0F; // /s (from biological constants)
    float ip3_production = external_stimulus * dt * 0.5F; // Stimulus produces IP3
    float ip3_degradation = ip3_degradation_rate * ip3 * dt;
    ip3 += ip3_production - ip3_degradation;
    ip3 = fmaxf(0.0F, fminf(5.0F, ip3)); // Clamp IP3

    // J_channel: IP3-receptor mediated Ca²⁺ release from ER
    // J_channel = v1 * (IP3³/(IP3³ + k3³)) * (Ca³/(Ca³ + k1³)) * (Ca_ER - Ca)
    float ip3_3 = ip3 * ip3 * ip3;
    float k3_3 = k3 * k3 * k3;
    float ip3_factor = ip3_3 / (ip3_3 + k3_3);

    float ca_3 = ca * ca * ca;
    float k1_3 = k1 * k1 * k1;
    float ca_factor = ca_3 / (ca_3 + k1_3);

    float J_channel = v1 * ip3_factor * ca_factor * (ca_er - ca);

    // J_leak: Passive leak from ER
    float J_leak = v2 * (ca_er - ca);

    // J_pump: ATP-dependent Ca²⁺ reuptake
    // J_pump = v3 * Ca²/(Ca² + k2²)
    float ca_2 = ca * ca;
    float k2_2 = k2 * k2;
    float J_pump = v3 * ca_2 / (ca_2 + k2_2);

    // Integrate calcium dynamics
    float d_ca = J_channel + J_leak - J_pump;

    // Add exponential decay term to ensure return to baseline
    // This dominates when IP3 is low, pulling Ca back to baseline
    float decay_to_baseline = -baseline_decay_rate * (ca - astro->calcium_baseline);
    d_ca += decay_to_baseline;

    ca += d_ca * dt;

    // Add stimulus contribution (direct calcium injection)
    ca += external_stimulus * dt * 0.1F;

    // Clamp to physiological range
    ca = fmaxf(0.0F, fminf(10.0F, ca));

    // Update astrocyte state
    astro->calcium_concentration = ca;
    astro->ip3_concentration = ip3;

    // Detect calcium spike (3x baseline)
    if (ca > astro->calcium_baseline * 3.0F) {
        astro->last_calcium_spike = nimcp_time_monotonic_us();
    }

    nimcp_spinlock_unlock(&astro->lock);

    // Process pending bio-async messages
    if (nimcp_atomic_load_bool(&g_astrocyte_bio_initialized, NIMCP_MEMORY_ORDER_ACQUIRE) && g_astrocyte_bio_ctx) {
        bio_router_process_inbox(g_astrocyte_bio_ctx, 5);
    }

    // Publish calcium concentration via predictive signal
    if (nimcp_atomic_load_bool(&g_astrocyte_bio_initialized, NIMCP_MEMORY_ORDER_ACQUIRE) && g_astrocyte_bio_ctx) {
        bio_router_publish_signal(g_astrocyte_bio_ctx, "astrocyte.calcium", ca);
        LOG_MODULE_DEBUG("ASTROCYTE", "Published calcium signal: %.2f μM", ca);
    }
}

void astrocyte_propagate_calcium_wave(astrocyte_t* astro,
                                      astrocyte_network_t* network,
                                      float dt)
{
    if (!astro || !network) {
        return;
    }

    // Only propagate if calcium is above threshold
    if (astro->calcium_concentration < network->calcium_threshold_um) {
        return;
    }

    // Propagate calcium to coupled neighbors via gap junctions
    nimcp_spinlock_lock(&astro->lock);

    for (uint32_t i = 0; i < astro->num_coupled_astrocytes; i++) {
        uint32_t neighbor_id = astro->coupled_astrocyte_ids[i];
        float coupling_strength = astro->coupling_strengths[i];

        // Find neighbor astrocyte in network
        astrocyte_t* neighbor = NULL;
        for (uint32_t j = 0; j < network->num_astrocytes; j++) {
            if (network->astrocytes[j]->id == neighbor_id) {
                neighbor = network->astrocytes[j];
                break;
            }
        }

        if (neighbor) {
            // Calcium diffusion through gap junction
            float ca_diff = astro->calcium_concentration - neighbor->calcium_concentration;
            float flux = coupling_strength * ca_diff * network->coupling_decay_rate * dt;

            // Update neighbor calcium (with their lock)
            nimcp_spinlock_lock(&neighbor->lock);
            neighbor->calcium_concentration += flux;
            neighbor->calcium_concentration = fmaxf(0.0F, fminf(50.0F, neighbor->calcium_concentration));
            nimcp_spinlock_unlock(&neighbor->lock);
        }
    }

    nimcp_spinlock_unlock(&astro->lock);
}

//=============================================================================
// Neurotransmitter Release - STUB IMPLEMENTATION
//=============================================================================

float astrocyte_compute_glutamate_release(astrocyte_t* astro, uint32_t synapse_idx)
{
    if (!astro || synapse_idx >= astro->num_covered_synapses) {
        return 0.0F;
    }

    // Calcium-dependent glutamate release with pool depletion
    float ca_threshold = astro->calcium_baseline * 2.0F;
    if (astro->calcium_concentration > ca_threshold) {
        float excess = astro->calcium_concentration - ca_threshold;
        float release_fraction = fminf(1.0F, excess / 5.0F);

        // Release glutamate from pool (depletes pool)
        // Higher release rate for biological accuracy
        float release_rate = 0.15F; // 15% of available pool × release_fraction
        float release_amount = release_fraction * astro->glutamate_pool * release_rate;
        astro->glutamate_pool -= release_amount;
        astro->glutamate_pool = fmaxf(0.0F, astro->glutamate_pool); // Clamp to [0, 1]

        // Publish glutamate release via predictive signal
        if (nimcp_atomic_load_bool(&g_astrocyte_bio_initialized, NIMCP_MEMORY_ORDER_ACQUIRE) && g_astrocyte_bio_ctx) {
            bio_router_publish_signal(g_astrocyte_bio_ctx, "astrocyte.glutamate", release_amount);
            LOG_MODULE_DEBUG("ASTROCYTE", "Published glutamate release: %.4f", release_amount);
        }

        return release_amount;
    }

    return 0.0F;
}

float astrocyte_compute_d_serine_release(astrocyte_t* astro, uint32_t synapse_idx)
{
    if (!astro || synapse_idx >= astro->num_covered_synapses) {
        return 0.0F;
    }

    // Hill equation for D-serine release
    // release = pool * Ca^n / (Ca^n + K^n)
    //
    // BIOLOGICAL PARAMETERS:
    // - n = 4: Hill coefficient (cooperative binding)
    // - K = 1.5 µM: Half-maximal calcium concentration
    // - Depletion rate: Pool depletes with release
    // - Refill rate: 0.01/s (slow recovery)

    nimcp_spinlock_lock(&astro->lock);

    const float n = 4.0F;           // Hill coefficient
    const float K = 1.5F;           // Half-max calcium (µM)
    const float depletion_rate = 0.1F;  // Pool depletion per release
    const float refill_rate = 0.01F;    // Pool refill rate (1/s)

    float ca = astro->calcium_concentration;
    float pool = astro->d_serine_pool;

    // Hill equation: fractional release
    float ca_n = powf(ca, n);
    float K_n = powf(K, n);
    float release_fraction = ca_n / (ca_n + K_n);

    // Actual release amount (from available pool)
    float release = pool * release_fraction;

    // Pool depletion (assume dt ~ 0.001s for small depletion)
    // Note: dt should be passed in for proper dynamics, but using small constant here
    float dt = 0.001F; // 1ms timestep assumption
    pool -= release * dt * depletion_rate;

    // Pool refill (slow recovery toward 1.0)
    pool += dt * refill_rate * (1.0F - pool);

    // Clamp pool to [0, 1]
    pool = fmaxf(0.0F, fminf(1.0F, pool));

    // Update pool state
    astro->d_serine_pool = pool;

    nimcp_spinlock_unlock(&astro->lock);

    return release;
}

//=============================================================================
// Synaptic Modulation - STUB IMPLEMENTATION
//=============================================================================

void astrocyte_modulate_synapse_strength(astrocyte_t* astro,
                                         synapse_t* synapse,
                                         uint32_t synapse_idx)
{
    if (!astro || !synapse || synapse_idx >= astro->num_covered_synapses) {
        return;
    }

    // Dual modulation: Glutamate enhances AMPA, D-serine gates NMDA
    //
    // BIOLOGICAL MECHANISM:
    // - Glutamate: Spillover enhances AMPA receptors (multiplicative)
    // - D-serine: Obligatory NMDA co-agonist (threshold function)
    //
    // MODULATION FORMULA:
    // modulation = (1 + 0.5 * glutamate) * (d_serine > 0.1 ? 1.2 : 1.0)
    //
    // - Glutamate effect: Up to 50% enhancement
    // - D-serine effect: 20% boost if present (threshold 0.1)

    float glutamate = astrocyte_compute_glutamate_release(astro, synapse_idx);
    float d_serine = astrocyte_compute_d_serine_release(astro, synapse_idx);

    // Glutamate: multiplicative enhancement of AMPA transmission
    float glutamate_factor = 1.0F + 0.5F * glutamate;

    // D-serine: threshold gating for NMDA (binary-like, but smooth threshold)
    float d_serine_factor = (d_serine > 0.1F) ? 1.2F : 1.0F;

    // Combined modulation
    float modulation = glutamate_factor * d_serine_factor;

    // Apply to synapse strength
    synapse->strength *= modulation;
}

nimcp_result_t astrocyte_assign_synapse(astrocyte_t* astro, uint32_t synapse_id)
{
    if (!astro) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Reallocate arrays
    uint32_t new_count = astro->num_covered_synapses + 1;

    uint32_t* new_ids = (uint32_t*) nimcp_realloc(astro->covered_synapse_ids,
                                                    new_count * sizeof(uint32_t));
    if (!new_ids) {
        return NIMCP_ERROR_MEMORY;
    }
    astro->covered_synapse_ids = new_ids;

    float* new_ca = (float*) nimcp_realloc(astro->synapse_calcium_levels,
                                            new_count * sizeof(float));
    if (!new_ca) {
        return NIMCP_ERROR_MEMORY;
    }
    astro->synapse_calcium_levels = new_ca;

    // Add synapse
    astro->covered_synapse_ids[astro->num_covered_synapses] = synapse_id;
    astro->synapse_calcium_levels[astro->num_covered_synapses] = astro->calcium_baseline;
    astro->num_covered_synapses = new_count;

    return NIMCP_SUCCESS;
}

//=============================================================================
// Homeostatic Plasticity - STUB IMPLEMENTATION
//=============================================================================

float astrocyte_compute_synaptic_scaling(astrocyte_t* astro, neural_network_t network)
{
    if (!astro) {
        return 1.0F;
    }

    // Homeostatic synaptic scaling with integrated calcium activity
    //
    // BIOLOGICAL MECHANISM:
    // - Astrocytes integrate calcium over time to estimate network activity
    // - Compare to target activity level
    // - Use PID controller to compute scaling factor
    //
    // ALGORITHM:
    // 1. Sliding window average of calcium (last 1000 timesteps)
    // 2. Error signal: target_activity - avg_calcium_normalized
    // 3. PID: scaling = 1.0 + Kp*error + Ki*integral_error
    // 4. Clamp to [0.3, 3.0] for stability
    //
    // PARAMETERS:
    // - Kp = 0.1: Proportional gain
    // - Ki = 0.01: Integral gain
    // - Window = 1000 timesteps (simplified to moving average)

    nimcp_spinlock_lock(&astro->lock);

    const float Kp = 0.1F;  // Proportional gain
    const float Ki = 0.01F; // Integral gain

    // Estimate average calcium activity
    // In full implementation, would use sliding window buffer
    // Here, simplified to current calcium normalized by expected max
    float avg_calcium_normalized = astro->calcium_concentration / 5.0F;

    // Error signal
    float error = astro->target_activity_level - avg_calcium_normalized;

    // Integral error (accumulated in scaling_factor over time as simple integrator)
    // Note: This is a simplification. Full implementation would track integral separately
    static float integral_error = 0.0F;
    integral_error += error * 0.001F; // Accumulate (assuming 1ms timestep)
    integral_error = fmaxf(-1.0F, fminf(1.0F, integral_error)); // Anti-windup

    // PID controller
    float scaling = 1.0F + Kp * error + Ki * integral_error;

    // Clamp to stability range [0.3, 3.0]
    scaling = fmaxf(0.3F, fminf(3.0F, scaling));

    // Update internal state
    astro->scaling_factor = scaling;

    nimcp_spinlock_unlock(&astro->lock);

    return scaling;
}

//=============================================================================
// BCM Threshold Modulation - STUB IMPLEMENTATION
//=============================================================================

float astrocyte_compute_bcm_threshold_shift(astrocyte_t* astro, float default_threshold)
{
    if (!astro) {
        return 0.0F;
    }

    // Sigmoidal BCM threshold modulation
    //
    // BIOLOGICAL MECHANISM:
    // - Elevated astrocyte calcium shifts BCM threshold upward
    // - Implements metaplasticity (plasticity of plasticity)
    // - Prevents runaway potentiation during high activity
    //
    // FORMULA:
    // shift = max_shift * tanh((calcium - baseline) / sensitivity)
    //
    // PARAMETERS:
    // - max_shift = 0.3: Maximum threshold shift
    // - sensitivity = 2.0: Sensitivity to calcium changes
    //
    // EFFECT:
    // - Positive shift raises threshold (favors LTD)
    // - Negative shift lowers threshold (favors LTP)

    nimcp_spinlock_lock(&astro->lock);

    const float max_shift = 0.3F;      // Maximum shift magnitude
    const float sensitivity = 2.0F;    // Sensitivity parameter

    float calcium_deviation = astro->calcium_concentration - astro->calcium_baseline;
    float normalized_deviation = calcium_deviation / sensitivity;

    // Sigmoidal modulation using tanh
    float shift = max_shift * tanhf(normalized_deviation);

    nimcp_spinlock_unlock(&astro->lock);

    return shift;
}

//=============================================================================
// Metabolic Support - STUB IMPLEMENTATION
//=============================================================================

void astrocyte_update_atp_level(astrocyte_t* astro, float neural_activity, float dt)
{
    if (!astro || dt < 0.0F) {
        return;
    }

    // ATP production and consumption dynamics
    //
    // BIOLOGICAL MECHANISM:
    // - Astrocytes convert glucose to lactate (glycolysis)
    // - Transfer lactate to neurons for ATP production
    // - High neural activity depletes astrocyte ATP
    // - Calcium pumping also consumes ATP
    //
    // PRODUCTION:
    // - Glycolysis: 0.5 + 0.3 * lactate_availability (basal + activity-dependent)
    //
    // CONSUMPTION:
    // - Neural support: 0.1 * neural_activity
    // - Calcium pumping: 0.05 * (calcium - baseline)²
    //
    // DYNAMICS:
    // d_atp = (production - consumption) * dt
    // Clamp to [0.0, 1.5] (can exceed 1.0 temporarily during high glycolysis)

    nimcp_spinlock_lock(&astro->lock);

    // Lactate availability (simplified to constant for now)
    float lactate_availability = 1.0F;

    // Glycolysis production
    float production = 0.5F + 0.3F * lactate_availability;

    // Activity-dependent consumption
    float neural_consumption = 0.1F * neural_activity;

    // Calcium pumping consumption (quadratic in calcium excess)
    float calcium_excess = astro->calcium_concentration - astro->calcium_baseline;
    float calcium_pumping = 0.05F * calcium_excess * calcium_excess;

    // Total consumption
    float consumption = neural_consumption + calcium_pumping;

    // Integrate ATP dynamics
    float d_atp = (production - consumption) * dt;
    astro->atp_level += d_atp;

    // Clamp to [0.0, 1.0]
    astro->atp_level = fmaxf(0.0F, fminf(1.0F, astro->atp_level));

    nimcp_spinlock_unlock(&astro->lock);

    // Publish ATP level via predictive signal
    if (nimcp_atomic_load_bool(&g_astrocyte_bio_initialized, NIMCP_MEMORY_ORDER_ACQUIRE) && g_astrocyte_bio_ctx) {
        bio_router_publish_signal(g_astrocyte_bio_ctx, "astrocyte.atp", astro->atp_level);
        LOG_MODULE_DEBUG("ASTROCYTE", "Published ATP level: %.3f", astro->atp_level);
    }
}

//=============================================================================
// Astrocyte Network Management - STUB IMPLEMENTATION
//=============================================================================

astrocyte_network_t* astrocyte_network_create(uint32_t capacity)
{
    astrocyte_network_t* network = (astrocyte_network_t*) nimcp_malloc(sizeof(astrocyte_network_t));
    if (!network) {
        return NULL;
    }

    memset(network, 0, sizeof(astrocyte_network_t));

    network->capacity = capacity;
    network->num_astrocytes = 0;
    network->astrocytes = (astrocyte_t**) nimcp_malloc(capacity * sizeof(astrocyte_t*));
    if (!network->astrocytes) {
        nimcp_free(network);
        return NULL;
    }

    // Global parameters
    network->calcium_threshold_um = ASTROCYTE_CALCIUM_WAVE_THRESHOLD_UM;
    network->coupling_decay_rate = 0.01F; // Biologically realistic coupling (target: 10-30 µm/s wave speed)
    network->coupling_radius_um = ASTROCYTE_COUPLING_RADIUS_UM;

    // Spatial index (NULL for now)
    network->spatial_index = NULL;

    nimcp_mutex_init(&network->lock, NULL);

    return network;
}

void astrocyte_network_destroy(astrocyte_network_t* network)
{
    if (!network) {
        return;
    }

    // Destroy all astrocytes
    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_destroy(network->astrocytes[i]);
    }

    // Free arrays
    if (network->astrocytes) {
        nimcp_free(network->astrocytes);
    }

    // Destroy spatial index (KD-tree)
    if (network->spatial_index) {
        kdtree_destroy((kdtree_t*)network->spatial_index);
    }

    nimcp_mutex_destroy(&network->lock);
    nimcp_free(network);
}

nimcp_result_t astrocyte_network_add(astrocyte_network_t* network, astrocyte_t* astro)
{
    if (!network || !astro) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (network->num_astrocytes >= network->capacity) {
        // Resize
        uint32_t new_capacity = network->capacity * 2;
        astrocyte_t** new_array = (astrocyte_t**) nimcp_realloc(network->astrocytes,
                                                                 new_capacity * sizeof(astrocyte_t*));
        if (!new_array) {
            return NIMCP_ERROR_MEMORY;
        }
        network->astrocytes = new_array;
        network->capacity = new_capacity;
    }

    network->astrocytes[network->num_astrocytes++] = astro;
    return NIMCP_SUCCESS;
}

nimcp_result_t astrocyte_network_build_spatial_index(astrocyte_network_t* network)
{
    if (!network) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // Clear existing spatial index if present
    if (network->spatial_index) {
        kdtree_destroy((kdtree_t*)network->spatial_index);
        network->spatial_index = NULL;
    }

    // No astrocytes to index
    if (network->num_astrocytes == 0) {
        return NIMCP_SUCCESS;
    }

    // Create KD-tree
    kdtree_t* kdtree = kdtree_create();
    if (!kdtree) {
        return NIMCP_ERROR_MEMORY;
    }

    // Prepare point and user data arrays
    kdtree_point_t* points = (kdtree_point_t*)nimcp_malloc(
        network->num_astrocytes * sizeof(kdtree_point_t));
    void** user_data = (void**)nimcp_malloc(
        network->num_astrocytes * sizeof(void*));

    if (!points || !user_data) {
        nimcp_free(points);
        nimcp_free(user_data);
        kdtree_destroy(kdtree);
        return NIMCP_ERROR_MEMORY;
    }

    // Copy astrocyte positions and pointers
    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        memcpy(points[i], network->astrocytes[i]->position, sizeof(kdtree_point_t));
        user_data[i] = network->astrocytes[i];
    }

    // Build KD-tree
    bool success = kdtree_build(kdtree, points, user_data, network->num_astrocytes);

    nimcp_free(points);
    nimcp_free(user_data);

    if (!success) {
        kdtree_destroy(kdtree);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    network->spatial_index = kdtree;
    return NIMCP_SUCCESS;
}

astrocyte_t* astrocyte_network_find_nearest(astrocyte_network_t* network, const float point[3])
{
    if (!network || !point || network->num_astrocytes == 0) {
        return NULL;
    }
    if (!network->astrocytes) {
        return NULL;
    }
    /* Validate spatial coordinates are finite */
    if (!isfinite(point[0]) || !isfinite(point[1]) || !isfinite(point[2])) {
        return NULL;
    }
    /* Bounds check: num_astrocytes should not exceed capacity */
    if (network->num_astrocytes > network->capacity) {
        return NULL;
    }

    /* Use KD-tree for O(log N) lookup if available */
    if (network->spatial_index) {
        kdtree_t* kdtree = (kdtree_t*)network->spatial_index;
        return (astrocyte_t*)kdtree_nearest(kdtree, point, NULL);
    }

    // Fallback to linear search O(N) if KD-tree not built
    astrocyte_t* nearest = network->astrocytes[0];
    float min_dist_sq = INFINITY;

    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_t* astro = network->astrocytes[i];
        float dx = astro->position[0] - point[0];
        float dy = astro->position[1] - point[1];
        float dz = astro->position[2] - point[2];
        float dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest = astro;
        }
    }

    return nearest;
}

nimcp_result_t astrocyte_network_establish_coupling(astrocyte_network_t* network)
{
    if (!network) {
        return NIMCP_ERROR_INVALID_PARAM;
    }
    if (!network->astrocytes || network->num_astrocytes == 0) {
        return NIMCP_SUCCESS;  /* Nothing to couple */
    }
    /* Bounds check: num_astrocytes should not exceed capacity */
    if (network->num_astrocytes > network->capacity) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    /* For each astrocyte, find neighbors within coupling radius */
    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_t* astro = network->astrocytes[i];

        // Count neighbors first
        uint32_t neighbor_count = 0;
        for (uint32_t j = 0; j < network->num_astrocytes; j++) {
            if (i == j) continue;

            astrocyte_t* neighbor = network->astrocytes[j];
            float dx = astro->position[0] - neighbor->position[0];
            float dy = astro->position[1] - neighbor->position[1];
            float dz = astro->position[2] - neighbor->position[2];
            float distance = sqrtf(dx*dx + dy*dy + dz*dz);

            if (distance <= network->coupling_radius_um) {
                neighbor_count++;
            }
        }

        if (neighbor_count == 0) {
            continue;
        }

        // Allocate arrays for neighbors
        astro->coupled_astrocyte_ids = (uint32_t*) nimcp_malloc(neighbor_count * sizeof(uint32_t));
        astro->coupling_strengths = (float*) nimcp_malloc(neighbor_count * sizeof(float));

        if (!astro->coupled_astrocyte_ids || !astro->coupling_strengths) {
            return NIMCP_ERROR_MEMORY;
        }

        // Fill arrays
        uint32_t idx = 0;
        for (uint32_t j = 0; j < network->num_astrocytes; j++) {
            if (i == j) continue;

            astrocyte_t* neighbor = network->astrocytes[j];
            float dx = astro->position[0] - neighbor->position[0];
            float dy = astro->position[1] - neighbor->position[1];
            float dz = astro->position[2] - neighbor->position[2];
            float distance = sqrtf(dx*dx + dy*dy + dz*dz);

            if (distance <= network->coupling_radius_um) {
                astro->coupled_astrocyte_ids[idx] = neighbor->id;

                // Coupling strength decreases with distance
                // Use max(0.1, ...) to ensure minimum coupling even at boundary
                float normalized_dist = distance / network->coupling_radius_um;
                float strength = fmaxf(0.1F, 1.0F - normalized_dist);

                // Scale by distance² to compensate for Laplacian normalization
                // The graph Laplacian divides by distance², so we multiply here
                // to maintain the effective diffusion coefficient
                float distance_sq = dx*dx + dy*dy + dz*dz;
                astro->coupling_strengths[idx] = strength * distance_sq;

                idx++;
            }
        }

        astro->num_coupled_astrocytes = neighbor_count;
    }

    return NIMCP_SUCCESS;
}

nimcp_result_t astrocyte_network_assign_synapses(astrocyte_network_t* network,
                                                 neural_network_t* nn)
{
    if (!network || !nn) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    // TODO: Implement full synapse assignment based on spatial proximity
    // For now, stub returns success
    return NIMCP_SUCCESS;
}

void astrocyte_network_step(astrocyte_network_t* network, float dt)
{
    if (!network || dt <= 0.0F) {
        return;
    }
    if (!network->astrocytes) {
        return;
    }
    /* Bounds check: num_astrocytes should not exceed capacity */
    if (network->num_astrocytes > network->capacity) {
        return;
    }
    /* Validate dt is finite and within reasonable range */
    if (!isfinite(dt) || dt > 1.0F) {
        return;  /* Cap at 1 second per step */
    }

    /* TODO: Implement full network dynamics */
    /* For now, simple update of all astrocytes */

    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_t* astro = network->astrocytes[i];

        // Update calcium
        astrocyte_update_calcium(astro, dt, 0.0F);

        // Propagate waves
        astrocyte_propagate_calcium_wave(astro, network, dt);

        // Update ATP
        astrocyte_update_atp_level(astro, 1.0F, dt);
    }
}

//=============================================================================
// Utility Functions
//=============================================================================

void astrocyte_network_get_stats(astrocyte_network_t* network,
                                 float* avg_calcium,
                                 float* max_calcium,
                                 float* avg_glutamate)
{
    if (!network) {
        if (avg_calcium) *avg_calcium = 0.0F;
        if (max_calcium) *max_calcium = 0.0F;
        if (avg_glutamate) *avg_glutamate = 0.0F;
        return;
    }

    float sum_ca = 0.0F;
    float sum_glu = 0.0F;
    float max_ca = 0.0F;

    for (uint32_t i = 0; i < network->num_astrocytes; i++) {
        astrocyte_t* astro = network->astrocytes[i];
        sum_ca += astro->calcium_concentration;
        sum_glu += astro->glutamate_pool;
        if (astro->calcium_concentration > max_ca) {
            max_ca = astro->calcium_concentration;
        }
    }

    if (avg_calcium) *avg_calcium = sum_ca / network->num_astrocytes;
    if (max_calcium) *max_calcium = max_ca;
    if (avg_glutamate) *avg_glutamate = sum_glu / network->num_astrocytes;
}
