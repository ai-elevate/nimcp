/**
 * @file nimcp_astrocyte_calcium.c
 * @brief Implementation of reaction-diffusion calcium dynamics in astrocytes
 *
 * ENHANCEMENT: A4.1 - Reaction-Diffusion Calcium in Astrocytes
 *
 * BIOLOGICAL MODEL:
 * - Coupled Ca²⁺ and IP3 reaction-diffusion equations
 * - Graph-based discretization on astrocyte network topology
 * - Calcium wave propagation at 10-20 µm/s (biological range)
 *
 * MATHEMATICAL FORMULATION:
 * ∂Ca²⁺/∂t = D_Ca∇²Ca²⁺ + J_release - J_uptake
 * ∂IP3/∂t = D_IP3∇²IP3 + production - degradation
 *
 * where:
 * - ∇²: Graph Laplacian (sum over neighbors)
 * - J_release: IP3-dependent Ca²⁺ release from ER stores
 * - J_uptake: ATP-dependent Ca²⁺ pumps
 *
 * PERFORMANCE:
 * - Target overhead: < 15% vs simple decay model
 * - O(N × K) complexity per timestep (K = avg neighbors ~3-6)
 * - Thread-safe with spinlocks
 *
 * INTEGRATION:
 * - glial_integration module: Calcium waves → gliotransmitter release
 * - astrocyte_network: Uses gap junction topology for diffusion
 *
 * @version 1.0
 * @date 2025-11-11
 */

#include "nimcp_astrocytes.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
// Constants
//=============================================================================

/** ER calcium baseline (µM) */
#define ER_CALCIUM_BASELINE 400.0f

/** IP3 threshold for calcium release (µM) */
#define IP3_THRESHOLD_FOR_RELEASE 0.5f

/** Hill coefficient for IP3 receptor */
#define IP3_HILL_COEFFICIENT 2.0f

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Compute IP3-dependent calcium release from ER
 *
 * Uses Hill equation: J = J_max * (IP3^n / (K^n + IP3^n)) * (Ca_ER - Ca)
 */
static inline float compute_ca_release_flux(float ip3, float ca_cytosol, float ca_er, float flux_coeff) {
    if (ip3 < 0.01f) return 0.0f; // No release without IP3

    // Hill function for IP3 receptor activation
    float ip3_n = powf(ip3, IP3_HILL_COEFFICIENT);
    float K_n = powf(IP3_THRESHOLD_FOR_RELEASE, IP3_HILL_COEFFICIENT);
    float activation = ip3_n / (K_n + ip3_n);

    // Driving force (Ca gradient)
    float driving_force = ca_er - ca_cytosol;
    if (driving_force < 0.0f) driving_force = 0.0f;

    return flux_coeff * activation * driving_force;
}

/**
 * @brief Compute calcium uptake by pumps
 *
 * ATP-dependent pumps with Michaelis-Menten kinetics
 */
static inline float compute_ca_uptake_flux(float ca_cytosol, float uptake_rate) {
    // Simplified linear uptake (full model would use Michaelis-Menten)
    return uptake_rate * ca_cytosol;
}

/**
 * @brief Compute graph Laplacian for diffusion
 *
 * Laplacian = Σ_neighbors (C_neighbor - C_self) × coupling_strength
 */
static float compute_graph_laplacian(
    float* concentration,
    uint32_t astrocyte_idx,
    astrocyte_t* astro,
    astrocyte_network_t* network)
{
    float laplacian = 0.0f;

    // Sum contributions from all coupled neighbors
    for (uint32_t i = 0; i < astro->num_coupled_astrocytes; i++) {
        uint32_t neighbor_id = astro->coupled_astrocyte_ids[i];
        float coupling = astro->coupling_strengths[i];

        // Find neighbor in network
        astrocyte_t* neighbor = NULL;
        uint32_t neighbor_idx = 0;
        for (uint32_t j = 0; j < network->num_astrocytes; j++) {
            if (network->astrocytes[j]->id == neighbor_id) {
                neighbor = network->astrocytes[j];
                neighbor_idx = j;
                break;
            }
        }

        if (neighbor) {
            float diff = concentration[neighbor_idx] - concentration[astrocyte_idx];
            laplacian += coupling * diff;
        }
    }

    return laplacian;
}

//=============================================================================
// Creation and Destruction
//=============================================================================

astrocyte_calcium_system_t* astrocyte_calcium_system_create(astrocyte_network_t* network)
{
    if (!network || network->num_astrocytes == 0) {
        return NULL;
    }

    astrocyte_calcium_system_t* system = (astrocyte_calcium_system_t*) nimcp_malloc(
        sizeof(astrocyte_calcium_system_t));
    if (!system) {
        return NULL;
    }

    memset(system, 0, sizeof(astrocyte_calcium_system_t));

    uint32_t num = network->num_astrocytes;
    system->num_astrocytes = num;
    system->network = network;

    // Allocate state arrays
    system->calcium = (float*) nimcp_malloc(num * sizeof(float));
    system->ip3 = (float*) nimcp_malloc(num * sizeof(float));
    system->calcium_er = (float*) nimcp_malloc(num * sizeof(float));
    system->last_wave_time = (uint64_t*) nimcp_malloc(num * sizeof(uint64_t));

    if (!system->calcium || !system->ip3 || !system->calcium_er || !system->last_wave_time) {
        astrocyte_calcium_system_destroy(system);
        return NULL;
    }

    // Initialize state arrays
    for (uint32_t i = 0; i < num; i++) {
        system->calcium[i] = ASTROCYTE_BASELINE_CALCIUM_UM;
        system->ip3[i] = 0.0f;
        system->calcium_er[i] = ER_CALCIUM_BASELINE;
        system->last_wave_time[i] = 0;
    }

    // Set parameters from biological constants
    system->D_ca = CALCIUM_DIFFUSION_COEFF;
    system->D_ip3 = IP3_DIFFUSION_COEFF;
    system->ip3_production_rate = IP3_PRODUCTION_RATE;
    system->ip3_degradation_rate = IP3_DEGRADATION_RATE;
    system->ca_release_flux = CALCIUM_RELEASE_FLUX;
    system->ca_uptake_rate = CALCIUM_UPTAKE_RATE;

    // Initialize performance tracking
    system->total_update_time_us = 0;
    system->num_updates = 0;
    system->wave_speed_measured = 0.0f;

    // Initialize lock
    nimcp_spinlock_init(&system->lock);

    return system;
}

void astrocyte_calcium_system_destroy(astrocyte_calcium_system_t* system)
{
    if (!system) {
        return;
    }

    if (system->calcium) {
        nimcp_free(system->calcium);
    }
    if (system->ip3) {
        nimcp_free(system->ip3);
    }
    if (system->calcium_er) {
        nimcp_free(system->calcium_er);
    }
    if (system->last_wave_time) {
        nimcp_free(system->last_wave_time);
    }

    nimcp_spinlock_destroy(&system->lock);
    nimcp_free(system);
}

//=============================================================================
// Reaction-Diffusion Update
//=============================================================================

void astrocyte_calcium_system_update(
    astrocyte_calcium_system_t* system,
    float dt,
    float* external_stimulus)
{
    if (!system || !system->network || dt <= 0.0f) {
        return;
    }

    uint64_t start_time = nimcp_time_monotonic_us();

    nimcp_spinlock_lock(&system->lock);

    uint32_t num = system->num_astrocytes;
    astrocyte_network_t* network = system->network;

    // Allocate temporary arrays for derivatives (prevents aliasing)
    float* dCa_dt = (float*) nimcp_malloc(num * sizeof(float));
    float* dIP3_dt = (float*) nimcp_malloc(num * sizeof(float));
    float* dCa_ER_dt = (float*) nimcp_malloc(num * sizeof(float));

    if (!dCa_dt || !dIP3_dt || !dCa_ER_dt) {
        if (dCa_dt) nimcp_free(dCa_dt);
        if (dIP3_dt) nimcp_free(dIP3_dt);
        if (dCa_ER_dt) nimcp_free(dCa_ER_dt);
        nimcp_spinlock_unlock(&system->lock);
        return;
    }

    // Compute derivatives for all astrocytes
    for (uint32_t i = 0; i < num; i++) {
        astrocyte_t* astro = network->astrocytes[i];
        if (!astro) continue;

        float ca = system->calcium[i];
        float ip3 = system->ip3[i];
        float ca_er = system->calcium_er[i];

        // === CALCIUM DYNAMICS ===

        // Diffusion term (graph Laplacian)
        float ca_laplacian = compute_graph_laplacian(system->calcium, i, astro, network);
        float ca_diffusion = system->D_ca * ca_laplacian;

        // Reaction terms
        float J_release = compute_ca_release_flux(ip3, ca, ca_er, system->ca_release_flux);
        float J_uptake = compute_ca_uptake_flux(ca, system->ca_uptake_rate);

        // External stimulus (if provided)
        float stimulus = external_stimulus ? external_stimulus[i] : 0.0f;

        // dCa/dt = D∇²Ca + J_release - J_uptake + stimulus
        dCa_dt[i] = ca_diffusion + J_release - J_uptake + stimulus;

        // === IP3 DYNAMICS ===

        // Diffusion term
        float ip3_laplacian = compute_graph_laplacian(system->ip3, i, astro, network);
        float ip3_diffusion = system->D_ip3 * ip3_laplacian;

        // Reaction terms
        float ip3_production = system->ip3_production_rate * stimulus; // Stimulus-driven
        float ip3_degradation = system->ip3_degradation_rate * ip3;

        // dIP3/dt = D∇²IP3 + production - degradation
        dIP3_dt[i] = ip3_diffusion + ip3_production - ip3_degradation;

        // === ER CALCIUM DYNAMICS ===

        // ER loses Ca when cytosol gains Ca from release
        dCa_ER_dt[i] = -J_release;
    }

    // Integrate using forward Euler (could upgrade to RK4 with A1.1)
    uint64_t current_time = nimcp_time_monotonic_us();

    for (uint32_t i = 0; i < num; i++) {
        system->calcium[i] += dt * dCa_dt[i];
        system->ip3[i] += dt * dIP3_dt[i];
        system->calcium_er[i] += dt * dCa_ER_dt[i];

        // Clamp to reasonable ranges
        system->calcium[i] = fmaxf(0.0f, fminf(50.0f, system->calcium[i]));
        system->ip3[i] = fmaxf(0.0f, fminf(10.0f, system->ip3[i]));
        system->calcium_er[i] = fmaxf(0.0f, fminf(1000.0f, system->calcium_er[i]));

        // Track wave events (calcium above threshold)
        if (system->calcium[i] > ASTROCYTE_CALCIUM_WAVE_THRESHOLD_UM) {
            system->last_wave_time[i] = current_time;

            // Update individual astrocyte state (for compatibility)
            astrocyte_t* astro = network->astrocytes[i];
            if (astro) {
                astro->calcium_concentration = system->calcium[i];
                astro->ip3_concentration = system->ip3[i];
            }
        }
    }

    // Estimate wave speed (simple method: track wavefront spread)
    // This is a simplified measurement - full method would track wavefront position
    float max_ca = 0.0f;
    uint32_t max_idx = 0;
    for (uint32_t i = 0; i < num; i++) {
        if (system->calcium[i] > max_ca) {
            max_ca = system->calcium[i];
            max_idx = i;
        }
    }

    if (max_ca > ASTROCYTE_CALCIUM_WAVE_THRESHOLD_UM) {
        // Count neighbors above threshold
        astrocyte_t* source = network->astrocytes[max_idx];
        if (source) {
            uint32_t neighbors_active = 0;
            for (uint32_t i = 0; i < source->num_coupled_astrocytes; i++) {
                uint32_t neighbor_id = source->coupled_astrocyte_ids[i];
                for (uint32_t j = 0; j < num; j++) {
                    if (network->astrocytes[j]->id == neighbor_id) {
                        if (system->calcium[j] > ASTROCYTE_CALCIUM_WAVE_THRESHOLD_UM) {
                            neighbors_active++;
                        }
                        break;
                    }
                }
            }

            // Rough estimate: wave speed based on activation rate
            // Real calculation would track wavefront distance/time
            if (neighbors_active > 0) {
                float coupling_dist = network->coupling_radius_um / 2.0f; // Typical distance
                system->wave_speed_measured = coupling_dist / (dt * 1000.0f); // Convert to µm/s
            }
        }
    }

    // Free temporary arrays
    nimcp_free(dCa_dt);
    nimcp_free(dIP3_dt);
    nimcp_free(dCa_ER_dt);

    // Update performance metrics
    uint64_t end_time = nimcp_time_monotonic_us();
    system->total_update_time_us += (end_time - start_time);
    system->num_updates++;

    nimcp_spinlock_unlock(&system->lock);
}

//=============================================================================
// Wave Stimulation and Measurement
//=============================================================================

void astrocyte_calcium_system_stimulate(
    astrocyte_calcium_system_t* system,
    uint32_t astrocyte_id,
    float intensity)
{
    if (!system || astrocyte_id >= system->num_astrocytes) {
        return;
    }

    nimcp_spinlock_lock(&system->lock);

    // Increase both Ca²⁺ and IP3 to initiate wave
    system->calcium[astrocyte_id] += intensity * 1.0f; // µM
    system->ip3[astrocyte_id] += intensity * 0.5f; // µM

    // Clamp
    system->calcium[astrocyte_id] = fminf(50.0f, system->calcium[astrocyte_id]);
    system->ip3[astrocyte_id] = fminf(10.0f, system->ip3[astrocyte_id]);

    // Mark wave time
    system->last_wave_time[astrocyte_id] = nimcp_time_monotonic_us();

    nimcp_spinlock_unlock(&system->lock);
}

float astrocyte_calcium_system_get_wave_speed(astrocyte_calcium_system_t* system)
{
    if (!system) {
        return 0.0f;
    }

    nimcp_spinlock_lock(&system->lock);
    float speed = system->wave_speed_measured;
    nimcp_spinlock_unlock(&system->lock);

    return speed;
}

//=============================================================================
// Performance Monitoring
//=============================================================================

float astrocyte_calcium_system_get_overhead_percent(
    astrocyte_calcium_system_t* system,
    uint64_t total_sim_time_us)
{
    if (!system || total_sim_time_us == 0 || system->num_updates == 0) {
        return 0.0f;
    }

    nimcp_spinlock_lock(&system->lock);
    float overhead = (100.0f * system->total_update_time_us) / (float)total_sim_time_us;
    nimcp_spinlock_unlock(&system->lock);

    return overhead;
}

//=============================================================================
// Integration with Glial System
//=============================================================================

bool astrocyte_calcium_system_should_release_gliotransmitter(
    astrocyte_calcium_system_t* system,
    uint32_t astrocyte_id)
{
    if (!system || astrocyte_id >= system->num_astrocytes) {
        return false;
    }

    nimcp_spinlock_lock(&system->lock);
    bool should_release = system->calcium[astrocyte_id] > ASTROCYTE_CALCIUM_WAVE_THRESHOLD_UM;
    nimcp_spinlock_unlock(&system->lock);

    return should_release;
}

//=============================================================================
// State Queries
//=============================================================================

float astrocyte_calcium_system_get_calcium(
    astrocyte_calcium_system_t* system,
    uint32_t astrocyte_id)
{
    if (!system || astrocyte_id >= system->num_astrocytes) {
        return 0.0f;
    }

    nimcp_spinlock_lock(&system->lock);
    float ca = system->calcium[astrocyte_id];
    nimcp_spinlock_unlock(&system->lock);

    return ca;
}

float astrocyte_calcium_system_get_ip3(
    astrocyte_calcium_system_t* system,
    uint32_t astrocyte_id)
{
    if (!system || astrocyte_id >= system->num_astrocytes) {
        return 0.0f;
    }

    nimcp_spinlock_lock(&system->lock);
    float ip3 = system->ip3[astrocyte_id];
    nimcp_spinlock_unlock(&system->lock);

    return ip3;
}
