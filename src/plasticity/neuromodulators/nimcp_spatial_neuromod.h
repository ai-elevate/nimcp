/**
 * @file nimcp_spatial_neuromod.h
 * @brief Graph-Based Neuromodulator Diffusion System (Enhancement A2.1)
 *
 * WHAT: Spatial dynamics for neuromodulators on neural network graph
 *       Models concentration fields that diffuse and decay over time
 *       Implements reaction-diffusion PDE on graph topology
 *
 * WHY:  Realistic neuromodulator propagation requires spatial structure:
 *       - Dopamine diffuses from reward-processing regions
 *       - Serotonin spreads from raphe nuclei
 *       - Acetylcholine gradients guide attention
 *       - Local vs. global neuromodulation effects
 *
 * HOW:  Graph-based discretization of reaction-diffusion equation:
 *       ∂c/∂t = D * Laplacian(c) - k*c + S(x,t)
 *
 *       Discretized:
 *       c_i(t+Δt) = c_i(t) + Δt * [D * Σ_j (c_j - c_i) - k*c_i + S_i]
 *
 *       Where:
 *       - c_i: concentration at neuron i
 *       - D: diffusion coefficient (spatial spread)
 *       - k: decay rate (clearance/reuptake)
 *       - S_i: source term (release at neuron i)
 *       - j: neighbors of neuron i
 *
 * BIOLOGICAL MAPPING:
 * - Graph nodes: Neurons (each has concentration field)
 * - Graph edges: Synaptic connections (diffusion pathways)
 * - Diffusion: Volume transmission through extracellular space
 * - Decay: Reuptake by transporters, enzymatic degradation
 * - Sources: VTA (dopamine), raphe (serotonin), basal forebrain (ACh)
 *
 * MATHEMATICAL ENHANCEMENTS:
 * Enhancement A2.1 from MATHEMATICAL_ENHANCEMENTS_CHECKLIST.md
 * - Graph Laplacian operator for irregular topology
 * - Explicit Euler scheme (stable for small timesteps)
 * - Neumann boundary conditions (zero flux at borders)
 *
 * DESIGN PATTERNS:
 * - Strategy: Pluggable diffusion kernels
 * - Observer: Neurons observe local concentration
 * - Template Method: Update loop with pluggable dynamics
 * - Facade: Simplified API over complex PDE solver
 *
 * PERFORMANCE: O(E) per timestep where E = number of synapses
 *              ~10% overhead vs. point-based neuromodulation
 *
 * INTEGRATION:
 * - Works with existing neuromodulator_system_t
 * - Can be enabled/disabled via config flag
 * - Backward compatible (falls back to global broadcast)
 *
 * @author NIMCP Development Team
 * @date 2025-11-11
 */

#ifndef NIMCP_SPATIAL_NEUROMOD_H
#define NIMCP_SPATIAL_NEUROMOD_H

#include <stdbool.h>
#include <stdint.h>
#include "common/nimcp_export.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations
//=============================================================================

/**
 * @brief Forward declaration of neural network (opaque pointer type)
 * WHY: Avoid circular dependency with nimcp_neuralnet.h
 * NOTE: neural_network_t is already defined as opaque pointer in nimcp_neuralnet.h
 */
typedef struct neural_network_struct* neural_network_t;

//=============================================================================
// Spatial Neuromodulator Field
//=============================================================================

/**
 * @brief Spatial concentration field for one neuromodulator type
 *
 * WHAT: Per-neuron concentration array with diffusion dynamics
 * WHY:  Models spatial propagation of neuromodulators
 * HOW:  Each neuron has local concentration, diffuses to neighbors
 *
 * MEMORY: O(N) where N = number of neurons
 * COMPUTATION: O(E) per update where E = edges (synapses)
 */
typedef struct {
    // === CONCENTRATION FIELD ===
    float* concentration;        /**< [num_neurons] current concentration at each neuron (0-1) */
    float* source_rate;          /**< [num_neurons] release rate at each neuron (mol/s) */
    float* laplacian_buffer;     /**< [num_neurons] temporary buffer for Laplacian computation */
    uint32_t num_neurons;        /**< Number of neurons in network */

    // === PHYSICAL PARAMETERS ===
    float diffusion_coeff;       /**< D: diffusion coefficient (0.01-1.0, default: 0.1) */
    float decay_rate;            /**< k: decay/clearance rate (1/s, default: 0.01) */
    float baseline;              /**< Homeostatic baseline concentration (0-1, default: 0.1) */

    // === NUMERICAL PARAMETERS ===
    float timestep;              /**< Integration timestep (ms, default: 1.0) */
    uint32_t substeps;           /**< Substeps per timestep for stability (default: 1) */
    float max_concentration;     /**< Upper bound for concentration (default: 1.0) */
    float min_concentration;     /**< Lower bound for concentration (default: 0.0) */

    // === STATISTICS ===
    float total_released;        /**< Cumulative amount released */
    float total_decayed;         /**< Cumulative amount decayed */
    float avg_concentration;     /**< Average concentration across network */
    float max_gradient;          /**< Maximum spatial gradient */
    uint64_t update_count;       /**< Number of updates performed */

    // === TYPE IDENTIFICATION ===
    neuromodulator_type_t type;  /**< Which neuromodulator (DA, 5-HT, ACh, NE) */

} spatial_neuromod_field_t;

/**
 * @brief Complete spatial neuromodulator system
 *
 * WHAT: Manages all neuromodulator fields with spatial dynamics
 * WHY:  Centralizes spatial diffusion computation
 * HOW:  One field per neuromodulator type
 */
typedef struct {
    spatial_neuromod_field_t* fields[NEUROMOD_COUNT];  /**< One field per neuromodulator */
    bool enabled[NEUROMOD_COUNT];                      /**< Which fields are active */
    neural_network_t network;                          /**< Reference to network topology */
    bool use_substeps;                                 /**< Enable substeps for stability */
    float global_diffusion_scale;                      /**< Global scaling for all diffusion */
} spatial_neuromod_system_t;

//=============================================================================
// Configuration
//=============================================================================

/**
 * @brief Configuration for spatial neuromodulator field
 */
typedef struct {
    neuromodulator_type_t type;   /**< Which neuromodulator */
    float diffusion_coeff;        /**< D: diffusion coefficient (default: 0.1) */
    float decay_rate;             /**< k: decay rate (default: 0.01) */
    float baseline;               /**< Baseline concentration (default: 0.1) */
    float timestep;               /**< Integration timestep (default: 1.0 ms) */
    uint32_t substeps;            /**< Substeps for stability (default: 1) */
} spatial_neuromod_config_t;

/**
 * @brief Get default configuration for spatial neuromodulator field
 *
 * WHAT: Returns sensible defaults for given neuromodulator type
 * WHY:  Different neuromodulators have different kinetics
 * HOW:  Calibrated from literature values
 *
 * DEFAULTS:
 * - Dopamine: Fast diffusion (D=0.2), fast decay (k=0.5), baseline=0.05
 * - Serotonin: Slow diffusion (D=0.05), slow decay (k=0.1), baseline=0.3
 * - Acetylcholine: Fast diffusion (D=0.3), very fast decay (k=2.0), baseline=0.1
 * - Norepinephrine: Medium diffusion (D=0.15), medium decay (k=0.3), baseline=0.05
 *
 * @param type Neuromodulator type
 * @return Default configuration
 */
spatial_neuromod_config_t spatial_neuromod_default_config(neuromodulator_type_t type);

//=============================================================================
// Lifecycle Functions
//=============================================================================

/**
 * @brief Create spatial neuromodulator field
 *
 * WHAT: Allocates concentration arrays for N neurons
 * WHY:  Initialize spatial dynamics system
 * HOW:  malloc concentration[N], set parameters
 *
 * COMPLEXITY: O(N) memory, O(1) time
 *
 * @param num_neurons Number of neurons in network
 * @param config Field configuration
 * @return Spatial field or NULL on error
 */
spatial_neuromod_field_t* spatial_neuromod_create(uint32_t num_neurons,
                                                   const spatial_neuromod_config_t* config);

/**
 * @brief Destroy spatial neuromodulator field
 *
 * WHAT: Frees all allocated memory
 * WHY:  Clean shutdown, prevent memory leaks
 * HOW:  free concentration arrays
 *
 * @param field Field to destroy (can be NULL)
 */
void spatial_neuromod_destroy(spatial_neuromod_field_t* field);

/**
 * @brief Create complete spatial neuromodulator system
 *
 * WHAT: Creates fields for all enabled neuromodulators
 * WHY:  Unified interface for multi-field diffusion
 * HOW:  Calls spatial_neuromod_create for each type
 *
 * @param network Neural network (for topology)
 * @param enabled_types Array of booleans indicating which types to enable
 * @param configs Array of configurations (one per type)
 * @return Spatial system or NULL on error
 */
spatial_neuromod_system_t* spatial_neuromod_system_create(
    neural_network_t network,
    const bool enabled_types[NEUROMOD_COUNT],
    const spatial_neuromod_config_t configs[NEUROMOD_COUNT]
);

/**
 * @brief Destroy spatial neuromodulator system
 *
 * @param system System to destroy (can be NULL)
 */
void spatial_neuromod_system_destroy(spatial_neuromod_system_t* system);

//=============================================================================
// Diffusion Dynamics
//=============================================================================

/**
 * @brief Update spatial concentration field (single timestep)
 *
 * WHAT: Applies diffusion + decay + sources for one timestep
 * WHY:  Core PDE integration step
 * HOW:  Explicit Euler scheme on graph Laplacian
 *
 * ALGORITHM:
 * 1. Compute graph Laplacian: L_i = Σ_j (c_j - c_i) for neighbors j
 * 2. Apply reaction-diffusion: dc/dt = D*L - k*c + S
 * 3. Euler step: c(t+dt) = c(t) + dt * dc/dt
 * 4. Clamp to [0, 1]
 *
 * STABILITY: Requires dt < 1/(2*D*max_degree) for explicit Euler
 *            Use substeps if network has high-degree hubs
 *
 * COMPLEXITY: O(E) where E = number of synapses
 *
 * @param field Spatial field to update
 * @param network Network topology (for neighbor iteration)
 * @param dt Timestep (seconds)
 * @return true on success
 */
bool spatial_neuromod_update(spatial_neuromod_field_t* field,
                              neural_network_t network,
                              float dt);

/**
 * @brief Compute graph Laplacian at each node
 *
 * WHAT: Computes discrete Laplacian operator on graph
 * WHY:  Core operator for diffusion PDE
 * HOW:  For each neuron i: L_i = Σ_j (c_j - c_i) over neighbors j
 *
 * MATHEMATICAL DEFINITION:
 * Graph Laplacian: L = D - A
 * Where D = degree matrix, A = adjacency matrix
 * Applied to concentration: Lc_i = Σ_j A_ij (c_j - c_i)
 *
 * COMPLEXITY: O(E) where E = edges
 *
 * @param field Spatial field
 * @param network Network topology
 * @param laplacian Output array [num_neurons]
 * @return true on success
 */
bool spatial_neuromod_compute_laplacian(const spatial_neuromod_field_t* field,
                                         neural_network_t network,
                                         float* laplacian);

/**
 * @brief Release neuromodulator at specific neuron
 *
 * WHAT: Adds neuromodulator to source term at neuron
 * WHY:  Models phasic release (e.g., VTA dopamine burst)
 * HOW:  S_i += amount
 *
 * BIOLOGICAL: Vesicular release from presynaptic terminals
 *
 * @param field Spatial field
 * @param neuron_id Target neuron for release
 * @param amount Amount to release (mol/s or normalized units)
 * @return true on success
 */
bool spatial_neuromod_release(spatial_neuromod_field_t* field,
                               uint32_t neuron_id,
                               float amount);

/**
 * @brief Release neuromodulator at multiple neurons
 *
 * WHAT: Batch release operation
 * WHY:  Efficient for multiple simultaneous releases
 * HOW:  Updates source terms for all specified neurons
 *
 * USE CASE: Reward signal → dopamine release in striatal neurons
 *
 * @param field Spatial field
 * @param neuron_ids Array of neuron IDs
 * @param amounts Array of release amounts (parallel to neuron_ids)
 * @param count Number of neurons
 * @return true on success
 */
bool spatial_neuromod_release_batch(spatial_neuromod_field_t* field,
                                     const uint32_t* neuron_ids,
                                     const float* amounts,
                                     uint32_t count);

//=============================================================================
// Queries
//=============================================================================

/**
 * @brief Get concentration at specific neuron
 *
 * WHAT: Reads local concentration at neuron
 * WHY:  Synapses/neurons need local concentration for modulation
 * HOW:  Direct array access
 *
 * COMPLEXITY: O(1)
 *
 * @param field Spatial field
 * @param neuron_id Neuron to query
 * @return Concentration at neuron (0-1), or 0.0 on error
 */
float spatial_neuromod_get_concentration(const spatial_neuromod_field_t* field,
                                          uint32_t neuron_id);

/**
 * @brief Set concentration at specific neuron
 *
 * WHAT: Directly sets concentration (for initialization/testing)
 * WHY:  Useful for setting up initial conditions
 * HOW:  Direct array write with clamping
 *
 * @param field Spatial field
 * @param neuron_id Neuron to modify
 * @param concentration New concentration (will be clamped to [0,1])
 * @return true on success
 */
bool spatial_neuromod_set_concentration(spatial_neuromod_field_t* field,
                                         uint32_t neuron_id,
                                         float concentration);

/**
 * @brief Get spatial gradient at neuron
 *
 * WHAT: Computes concentration gradient (magnitude)
 * WHY:  Quantify spatial non-uniformity, gradient-following behavior
 * HOW:  |∇c_i| ≈ Σ_j |c_j - c_i| / degree_i
 *
 * USE CASE: Curiosity-driven exploration following dopamine gradients
 *
 * @param field Spatial field
 * @param network Network topology
 * @param neuron_id Neuron to query
 * @return Gradient magnitude, or 0.0 on error
 */
float spatial_neuromod_get_gradient(const spatial_neuromod_field_t* field,
                                     neural_network_t network,
                                     uint32_t neuron_id);

/**
 * @brief Get average concentration across network
 *
 * WHAT: Mean concentration over all neurons
 * WHY:  Global measure of neuromodulator level
 * HOW:  Sum(c_i) / N
 *
 * @param field Spatial field
 * @return Average concentration
 */
float spatial_neuromod_get_average(const spatial_neuromod_field_t* field);

/**
 * @brief Get maximum concentration in network
 *
 * WHAT: Peak concentration value
 * WHY:  Track hotspots, saturation
 * HOW:  max(c_i)
 *
 * @param field Spatial field
 * @param neuron_id_out Optional: location of maximum (can be NULL)
 * @return Maximum concentration
 */
float spatial_neuromod_get_max(const spatial_neuromod_field_t* field,
                                uint32_t* neuron_id_out);

//=============================================================================
// Integration with Global Neuromodulator System
//=============================================================================

/**
 * @brief Synchronize spatial field with global neuromodulator level
 *
 * WHAT: Updates global concentration from spatial field average
 * WHY:  Maintain consistency between spatial and global systems
 * HOW:  global_level = avg(spatial_concentrations)
 *
 * USE CASE: When spatial diffusion is enabled but some code still queries global
 *
 * @param field Spatial field
 * @param system Global neuromodulator system
 * @return true on success
 */
bool spatial_neuromod_sync_to_global(const spatial_neuromod_field_t* field,
                                      neuromodulator_system_t system);

/**
 * @brief Initialize spatial field from global neuromodulator level
 *
 * WHAT: Sets all spatial concentrations to global level
 * WHY:  Bootstrap spatial field from existing global state
 * HOW:  c_i = global_level for all i
 *
 * @param field Spatial field
 * @param system Global neuromodulator system
 * @return true on success
 */
bool spatial_neuromod_init_from_global(spatial_neuromod_field_t* field,
                                        neuromodulator_system_t system);

//=============================================================================
// Visualization & Analysis
//=============================================================================

/**
 * @brief Export concentration field for visualization
 *
 * WHAT: Exports concentration values for external plotting
 * WHY:  Visualize spatial distribution, gradients
 * HOW:  Writes to provided buffer
 *
 * @param field Spatial field
 * @param buffer Output buffer [num_neurons]
 * @param buffer_size Size of buffer (must be >= num_neurons)
 * @return Number of values written, or 0 on error
 */
uint32_t spatial_neuromod_export(const spatial_neuromod_field_t* field,
                                  float* buffer,
                                  uint32_t buffer_size);

/**
 * @brief Compute spatial statistics
 *
 * WHAT: Various spatial measures (mean, variance, gradient, etc.)
 * WHY:  Quantitative analysis of spatial distribution
 * HOW:  Single pass over concentration array
 *
 * @param field Spatial field
 * @param network Network topology
 * @param mean_out Output: mean concentration (can be NULL)
 * @param variance_out Output: variance (can be NULL)
 * @param max_gradient_out Output: maximum gradient (can be NULL)
 * @return true on success
 */
bool spatial_neuromod_compute_stats(const spatial_neuromod_field_t* field,
                                     neural_network_t network,
                                     float* mean_out,
                                     float* variance_out,
                                     float* max_gradient_out);

//=============================================================================
// Reset & Debugging
//=============================================================================

/**
 * @brief Reset field to baseline concentration
 *
 * WHAT: Sets all concentrations to baseline, clears sources
 * WHY:  Reset between experiments/trials
 * HOW:  c_i = baseline, S_i = 0 for all i
 *
 * @param field Spatial field
 * @return true on success
 */
bool spatial_neuromod_reset(spatial_neuromod_field_t* field);

/**
 * @brief Validate field state (for debugging)
 *
 * WHAT: Checks invariants (concentrations in [0,1], no NaN/inf, etc.)
 * WHY:  Catch numerical errors early
 * HOW:  Scans arrays for invalid values
 *
 * @param field Spatial field
 * @return true if valid, false if errors detected
 */
bool spatial_neuromod_validate(const spatial_neuromod_field_t* field);

#ifdef __cplusplus
}
#endif

#endif  // NIMCP_SPATIAL_NEUROMOD_H
