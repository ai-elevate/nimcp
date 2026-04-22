//=============================================================================
// nimcp_snn_network.h - SNN Network API
//=============================================================================
/**
 * @file nimcp_snn_network.h
 * @brief Main SNN network creation, simulation, and training API
 *
 * WHAT: Top-level SNN orchestration facade
 * WHY:  Unified API for spiking neural network operations
 * HOW:  Integrates with existing neural_network_t, adds spike-specific features
 *
 * DESIGN PATTERN: Facade
 * - SNN network is a facade over neural_network_t
 * - Adds spike encoding/decoding, population management, training
 * - Does NOT duplicate neuron/synapse infrastructure
 *
 * @author NIMCP Development Team
 * @date 2025-12-20
 * @version 1.0.0
 */

#ifndef NIMCP_SNN_NETWORK_H
#define NIMCP_SNN_NETWORK_H

#include "snn/nimcp_snn_types.h"
#include "snn/nimcp_snn_config.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Network Lifecycle
//=============================================================================

/**
 * @brief Create SNN network from configuration
 *
 * WHAT: Allocate and initialize a complete SNN
 * WHY:  Factory function for SNN creation
 * HOW:  Creates underlying neural_network, populations, encoder/decoder
 *
 * @param config Network configuration (validated before use)
 * @return New SNN network, or NULL on failure
 *
 * COMPLEXITY: O(n_inputs + n_outputs + sum(population_sizes))
 * MEMORY: ~500 bytes base + ~2KB per neuron (spike trains)
 */
snn_network_t* snn_network_create(const snn_config_t* config);

/**
 * @brief Destroy SNN network and free all resources
 *
 * WHAT: Clean up SNN and all owned resources
 * WHY:  Prevent memory leaks
 * HOW:  Destroys populations, encoder/decoder, simulation, training context
 *
 * NOTE: Does NOT destroy the underlying neural_network if it was provided
 *       externally. Only destroys if created internally.
 *
 * @param network Network to destroy (can be NULL)
 *
 * COMPLEXITY: O(n_populations + n_neurons)
 */
void snn_network_destroy(snn_network_t* network);

/**
 * @brief Reset SNN network to initial state
 *
 * WHAT: Clear all spike history, reset membrane potentials
 * WHY:  Prepare for new input sequence without recreating network
 * HOW:  Resets all neurons to resting state, clears spike trains
 *
 * @param network Network to reset
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(n_neurons)
 */
int snn_network_reset(snn_network_t* network);

/**
 * @brief Attach a neural substrate to an SNN for biological modulation.
 *
 * WHAT: Sets the borrowed substrate pointer that the SNN's step function
 *       consults each tick for ATP/temperature/ion/membrane-driven
 *       modulation of tau, refractory, spike survival, and plasticity LR.
 * WHY:  The substrate carries slow-varying biological state that governs
 *       metabolic constraints; the SNN needs a handle to read it without
 *       owning its lifecycle.
 * HOW:  Null-tolerant pass-through; resets the update period counter so
 *       effects are recomputed on the next step. The network does NOT
 *       destroy the substrate in snn_network_destroy.
 *
 * @param net Network to attach to (NULL-tolerant: returns silently).
 * @param sub Substrate pointer or NULL to detach.
 */
void snn_network_attach_substrate(snn_network_t* net,
                                  struct neural_substrate* sub);

//=============================================================================
// Simulation
//=============================================================================

/**
 * @brief Run one simulation timestep
 *
 * WHAT: Advance simulation by dt milliseconds
 * WHY:  Discrete-time SNN simulation
 * HOW:  Update membrane potentials, generate spikes, propagate spikes
 *
 * ALGORITHM:
 * 1. For each population:
 *    a. Compute synaptic input currents
 *    b. Update membrane potentials (LIF/Izhikevich)
 *    c. Generate spikes where V >= V_thresh
 *    d. Apply refractory period
 *    e. Record spikes to spike trains
 * 2. If training: update eligibility traces
 * 3. If bio-async: broadcast spike events
 *
 * @param network SNN network
 * @param dt Timestep in milliseconds (uses config.dt if 0)
 * @return Number of spikes generated, or negative on error
 *
 * COMPLEXITY: O(n_neurons × avg_synapses)
 */
int snn_network_step(snn_network_t* network, float dt);

/**
 * @brief Run simulation for specified duration
 *
 * WHAT: Simulate network for T milliseconds
 * WHY:  Convenience function for running multiple steps
 * HOW:  Calls snn_network_step repeatedly
 *
 * @param network SNN network
 * @param duration_ms Simulation duration in milliseconds
 * @return Total spikes generated, or negative on error
 *
 * COMPLEXITY: O((duration/dt) × n_neurons × avg_synapses)
 */
int snn_network_run(snn_network_t* network, float duration_ms);

/**
 * @brief Update network->stats with derived firing-rate metrics.
 *
 * Shared helper so both the inference path (snn_network_run) and the BPTT
 * training path (snn_backprop_forward) can update the same visible
 * metrics — mean_firing_rate, max_firing_rate, sparsity, silent_neurons,
 * hyperactive_neurons, spikes_per_sample. Previously only the inference
 * path updated these, which made training-time SNN activity invisible
 * via get_snn_stats RPC.
 *
 * @param network       The SNN network (stats will be overwritten).
 * @param total_spikes  Total spikes accumulated during the run.
 * @param duration_ms   Simulated duration in milliseconds (for Hz conversion).
 */
void snn_network_update_stats(snn_network_t* network, int total_spikes, float duration_ms);

//=============================================================================
// Performance-Optimized Stepping
//=============================================================================

/**
 * @brief Per-step performance statistics
 */
typedef struct snn_step_stats_s {
    uint32_t total_neurons;         /**< Total neurons across populations */
    uint32_t neurons_updated;       /**< Neurons that were actually updated */
    uint32_t neurons_skipped;       /**< Neurons skipped (far from threshold) */
    uint32_t neurons_refractory;    /**< Neurons in refractory period */
    uint32_t spikes_generated;      /**< Spikes this step */
    float    compute_ratio;         /**< neurons_updated / total_neurons */
} snn_step_stats_t;

/**
 * @brief Spike-driven sparse step — only updates active neurons
 *
 * WHAT: Event-driven neuron update that skips quiescent neurons
 * WHY:  At 2-5% firing rate, 95-98% of neurons are silent; O(spikes) not O(N)
 * HOW:  Skip neurons whose membrane potential is far below threshold AND
 *       have no incoming synaptic current. Only neurons that received a spike
 *       or are within threshold_margin of firing are updated.
 *
 * @param network SNN network
 * @param dt Timestep in milliseconds (uses config.dt if 0)
 * @param threshold_margin Neurons within this margin of v_thresh are always updated (mV).
 *                         Use 0 for default (5.0 mV).
 * @param stats [out] Optional step statistics (may be NULL)
 * @return Number of spikes generated, or negative on error
 *
 * COMPLEXITY: O(active_neurons) where active = spiked + near-threshold + has-input
 */
int snn_network_step_sparse(snn_network_t* network, float dt,
                             float threshold_margin,
                             snn_step_stats_t* stats);

/**
 * @brief Population-parallel step — independent populations step concurrently
 *
 * WHAT: Step populations in parallel using thread pool
 * WHY:  Independent populations have no cross-dependencies within a step
 * HOW:  Partition populations into independent groups, step each in parallel
 *
 * @param network SNN network
 * @param dt Timestep in milliseconds (uses config.dt if 0)
 * @param n_threads Number of threads (0 = use config.n_threads or auto-detect)
 * @return Number of spikes generated, or negative on error
 *
 * COMPLEXITY: O(n_neurons / n_threads × avg_synapses)
 */
int snn_network_step_parallel(snn_network_t* network, float dt,
                               uint32_t n_threads);

/**
 * @brief Run simulation using sparse (event-driven) stepping
 *
 * WHAT: Run for duration using snn_network_step_sparse for each timestep
 * WHY:  Combines sparse stepping with full simulation convenience
 *
 * @param network SNN network
 * @param duration_ms Simulation duration
 * @param threshold_margin Skip margin (0 = default 5.0 mV)
 * @param stats [out] Accumulated statistics (may be NULL)
 * @return Total spikes generated, or negative on error
 */
int snn_network_run_sparse(snn_network_t* network, float duration_ms,
                            float threshold_margin,
                            snn_step_stats_t* stats);

//=============================================================================
// Input/Output (Encoding/Decoding)
//=============================================================================

/**
 * @brief Set input values (will be encoded to spikes)
 *
 * WHAT: Provide continuous input values for spike encoding
 * WHY:  Convert external signals to spike trains
 * HOW:  Store values, encode to spikes during simulation
 *
 * @param network SNN network
 * @param inputs Input values [n_inputs]
 * @param n_inputs Number of input values
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(n_inputs)
 */
int snn_network_set_inputs(snn_network_t* network,
                           const float* inputs,
                           uint32_t n_inputs);

/**
 * @brief Set input as tensor
 *
 * WHAT: Provide input as nimcp_tensor_t
 * WHY:  Integration with tensor pipeline
 * HOW:  Extract data from tensor, call snn_network_set_inputs
 *
 * @param network SNN network
 * @param input Input tensor [n_inputs] or [batch, n_inputs]
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(n_inputs)
 */
int snn_network_set_input_tensor(snn_network_t* network,
                                 const nimcp_tensor_t* input);

/**
 * @brief Get decoded output values
 *
 * WHAT: Decode output spike trains to continuous values
 * WHY:  Convert spikes to usable output
 * HOW:  Apply configured decoding method to output population
 *
 * @param network SNN network
 * @param outputs Output buffer [n_outputs]
 * @param n_outputs Size of output buffer
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(n_outputs × spike_count)
 */
int snn_network_get_outputs(snn_network_t* network,
                            float* outputs,
                            uint32_t n_outputs);

/**
 * @brief Get output as tensor
 *
 * WHAT: Return decoded outputs as tensor
 * WHY:  Integration with tensor pipeline
 * HOW:  Decode spikes, fill tensor
 *
 * @param network SNN network
 * @param output Output tensor (must be pre-allocated [n_outputs])
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(n_outputs × spike_count)
 */
int snn_network_get_output_tensor(snn_network_t* network,
                                  nimcp_tensor_t* output);

/**
 * @brief Forward pass (set input, run, get output)
 *
 * WHAT: Complete forward inference
 * WHY:  Convenience function for single inference
 * HOW:  Combines set_inputs, run, get_outputs
 *
 * @param network SNN network
 * @param inputs Input values [n_inputs]
 * @param n_inputs Number of inputs
 * @param outputs Output buffer [n_outputs]
 * @param n_outputs Number of outputs
 * @param duration_ms Simulation duration
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O((duration/dt) × n_neurons × avg_synapses)
 */
int snn_network_forward(snn_network_t* network,
                        const float* inputs,
                        uint32_t n_inputs,
                        float* outputs,
                        uint32_t n_outputs,
                        float duration_ms);

//=============================================================================
// Training
//=============================================================================

/**
 * @brief Set training mode
 *
 * WHAT: Enable or disable training mode
 * WHY:  Control whether learning updates are applied
 * HOW:  Creates/destroys training context
 *
 * @param network SNN network
 * @param training true to enable training
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(1) if no context change, O(n_synapses) if allocating traces
 */
int snn_network_set_training(snn_network_t* network, bool training);

/**
 * @brief Apply STDP learning (uses existing synapse_t infrastructure)
 *
 * WHAT: Apply spike-timing dependent plasticity
 * WHY:  Local, biological learning rule
 * HOW:  Uses spike times from spike_trains, updates synapse weights
 *
 * NOTE: This leverages existing stdp_params in synapse_t, not duplicating
 *
 * @param network SNN network
 * @return Number of synapses modified
 *
 * COMPLEXITY: O(n_neurons × n_synapses × spike_count)
 */
int snn_network_apply_stdp(snn_network_t* network);

/**
 * @brief Apply reward-modulated STDP
 *
 * WHAT: Apply R-STDP with reward signal
 * WHY:  Reinforcement learning with biological plausibility
 * HOW:  STDP eligibility traces × reward signal
 *
 * @param network SNN network
 * @param reward Reward signal [-1, 1]
 * @return Number of synapses modified
 *
 * COMPLEXITY: O(n_synapses)
 */
int snn_network_apply_rstdp(snn_network_t* network, float reward);

/**
 * @brief Compute surrogate gradients (for backprop-through-time)
 *
 * WHAT: Compute gradients using surrogate gradient function
 * WHY:  Enable gradient-based training of SNNs
 * HOW:  Smooth approximation of spike derivative
 *
 * @param network SNN network
 * @param target Target output values [n_outputs]
 * @param n_targets Number of target values
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(n_neurons × n_synapses)
 */
int snn_network_compute_gradients(snn_network_t* network,
                                  const float* target,
                                  uint32_t n_targets);

/**
 * @brief Apply computed gradients to weights
 *
 * WHAT: Update weights using computed gradients
 * WHY:  Complete the training step
 * HOW:  w -= lr × grad_w
 *
 * @param network SNN network
 * @param learning_rate Learning rate (uses config if 0)
 * @return Number of weights updated
 *
 * COMPLEXITY: O(n_synapses)
 */
int snn_network_apply_gradients(snn_network_t* network, float learning_rate);

/**
 * @brief Complete training step (forward + backward + update)
 *
 * WHAT: Single training iteration
 * WHY:  Convenience function for training loop
 * HOW:  Forward pass, compute loss, compute gradients, update weights
 *
 * @param network SNN network
 * @param inputs Input values [n_inputs]
 * @param n_inputs Number of inputs
 * @param targets Target values [n_outputs]
 * @param n_targets Number of targets
 * @param duration_ms Simulation duration
 * @return Training loss, or negative on error
 *
 * COMPLEXITY: O((duration/dt) × n_neurons × avg_synapses)
 */
float snn_network_train_step(snn_network_t* network,
                             const float* inputs,
                             uint32_t n_inputs,
                             const float* targets,
                             uint32_t n_targets,
                             float duration_ms);

//=============================================================================
// Population Management
//=============================================================================

/**
 * @brief Add population to network
 *
 * WHAT: Create and add a new population of neurons
 * WHY:  Build network architecture incrementally
 * HOW:  Allocates population, adds neurons to underlying neural_network
 *
 * @param network SNN network
 * @param n_neurons Number of neurons in population
 * @param neuron_type Type of neurons (LIF, Izhikevich, etc.)
 * @param name Human-readable name
 * @return Population ID, or negative on error
 *
 * COMPLEXITY: O(n_neurons)
 */
int snn_network_add_population(snn_network_t* network,
                               uint32_t n_neurons,
                               neuron_type_t neuron_type,
                               const char* name);

/**
 * @brief Add population in lightweight/CSR mode (no dense allocation)
 *
 * @param network SNN network
 * @param n_neurons Number of neurons
 * @param neuron_type Type of neurons
 * @param name Human-readable name
 * @return Population ID, or negative on error
 */
int snn_network_add_population_lightweight(snn_network_t* network,
                                           uint32_t n_neurons,
                                           neuron_type_t neuron_type,
                                           const char* name);

/**
 * @brief Connect two populations
 *
 * WHAT: Create synaptic connections between populations
 * WHY:  Define network topology
 * HOW:  Creates synapses in underlying neural_network
 *
 * @param network SNN network
 * @param src_pop Source population ID
 * @param dst_pop Destination population ID
 * @param topology Connection topology
 * @param connectivity Connectivity ratio [0, 1] for sparse topologies
 * @param synapse_type Type of synapses (AMPA, NMDA, GABA, etc.)
 * @param weight_mean Mean initial weight
 * @param weight_std Weight std for random initialization
 * @return Number of synapses created, or negative on error
 *
 * COMPLEXITY: O(n_src × n_dst × connectivity)
 */
int snn_network_connect_populations(snn_network_t* network,
                                    uint32_t src_pop,
                                    uint32_t dst_pop,
                                    snn_topology_t topology,
                                    float connectivity,
                                    synapse_type_t synapse_type,
                                    float weight_mean,
                                    float weight_std);

/**
 * @brief Get population by ID
 *
 * WHAT: Access population structure
 * WHY:  Inspect or modify population state
 * HOW:  Look up in populations array
 *
 * @param network SNN network
 * @param pop_id Population ID
 * @return Population pointer, or NULL if not found
 *
 * COMPLEXITY: O(1)
 */
snn_population_t* snn_network_get_population(snn_network_t* network,
                                             uint32_t pop_id);

//=============================================================================
// Statistics and Monitoring
//=============================================================================

/**
 * @brief Get network statistics
 *
 * WHAT: Fill statistics structure with current metrics
 * WHY:  Monitor network health and performance
 * HOW:  Aggregate metrics from all populations
 *
 * @param network SNN network
 * @param stats Statistics structure to fill
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(n_populations)
 */
int snn_network_get_stats(snn_network_t* network, snn_stats_t* stats);

/**
 * @brief Check network health
 *
 * WHAT: Detect simulation issues (silence, explosion, NaN)
 * WHY:  Early warning for training problems
 * HOW:  Check firing rates, weight norms, voltage ranges
 *
 * @param network SNN network
 * @return Health status enum
 *
 * COMPLEXITY: O(n_populations)
 */
snn_state_health_t snn_network_check_health(snn_network_t* network);

/**
 * @brief Get firing rate for a neuron
 *
 * WHAT: Compute instantaneous firing rate
 * WHY:  Monitor individual neuron activity
 * HOW:  Count spikes in recent time window
 *
 * @param network SNN network
 * @param pop_id Population ID
 * @param neuron_idx Neuron index within population
 * @param window_ms Time window for rate calculation
 * @return Firing rate in Hz
 *
 * COMPLEXITY: O(spike_count in window)
 */
float snn_network_get_firing_rate(snn_network_t* network,
                                  uint32_t pop_id,
                                  uint32_t neuron_idx,
                                  float window_ms);

/**
 * @brief Get population mean firing rate
 *
 * WHAT: Average firing rate across population
 * WHY:  Monitor population activity level
 *
 * @param network SNN network
 * @param pop_id Population ID
 * @param window_ms Time window for rate calculation
 * @return Mean firing rate in Hz
 *
 * COMPLEXITY: O(n_neurons × spike_count)
 */
float snn_network_get_population_rate(snn_network_t* network,
                                      uint32_t pop_id,
                                      float window_ms);

/**
 * @brief Copy a population's temporal spike-count history into a caller buffer.
 *
 * Returns the last SNN_POP_HISTORY_LEN steps of population-aggregate spike
 * counts, time-ordered oldest→newest. Used by Python eval tooling for FFT,
 * cross-correlation, and autocorrelation analysis of population dynamics.
 *
 * @param network    SNN network
 * @param pop_id     Population index
 * @param out_counts Caller-allocated buffer of SNN_POP_HISTORY_LEN uint32_t
 * @param out_total_steps Optional — receives the monotonic step counter
 * @return Number of valid entries in out_counts (may be less than
 *         SNN_POP_HISTORY_LEN if the population hasn't run that many steps),
 *         or 0 on error.
 */
uint32_t snn_network_get_population_history(snn_network_t* network,
                                            uint32_t pop_id,
                                            uint32_t* out_counts,
                                            uint64_t* out_total_steps);

/**
 * @brief Emergency saturation rescue — run homeostatic apply N times in a row.
 *
 * When the SNN drifts far from biological range (e.g., 90%+ firing after
 * a bad training episode), normal homeostatic cadence (every 10 R-STDP
 * applies) is too slow to recover — can take hours. This force-applies the
 * homeostatic scaling N times in rapid succession, scaling weights by
 * ~0.9^N. 20 iterations drops firing rate by 8×.
 *
 * @param network SNN network
 * @param ctx     snn_training_ctx_t*
 * @param n_iter  How many back-to-back applies to run
 * @return Total populations scaled across all iterations
 */
uint32_t snn_network_force_homeostasis(snn_network_t* network,
                                       void* ctx,
                                       uint32_t n_iter);

/**
 * @brief Get population firing rate directly from population pointer
 *
 * WHAT: Average firing rate across population (convenience wrapper)
 * WHY:  Allow bridges with population pointers to query firing rate
 * HOW:  Computes mean rate from population's neurons
 *
 * @param population Population pointer
 * @return Mean firing rate in Hz (0.0 if NULL or no spikes)
 *
 * COMPLEXITY: O(n_neurons)
 */
float snn_population_get_firing_rate(const snn_population_t* population);

//=============================================================================
// Integration
//=============================================================================

/**
 * @brief Connect to bio-async router
 *
 * WHAT: Register SNN with bio-async messaging system
 * WHY:  Enable inter-module spike event communication
 * HOW:  Registers with bio_router, sets up message handlers
 *
 * @param network SNN network
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(1)
 */
int snn_network_connect_bio_async(snn_network_t* network);

/**
 * @brief Disconnect from bio-async router
 *
 * @param network SNN network
 * @return SNN_SUCCESS on success
 */
int snn_network_disconnect_bio_async(snn_network_t* network);

/**
 * @brief Connect to brain immune system
 *
 * WHAT: Integrate SNN with immune system
 * WHY:  Cytokine modulation of excitability and plasticity
 * HOW:  Creates immune bridge, registers for cytokine updates
 *
 * @param network SNN network
 * @param immune Brain immune system handle
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(1)
 */
int snn_network_connect_immune(snn_network_t* network, void* immune);

/**
 * @brief Apply immune modulation effects
 *
 * WHAT: Update SNN parameters based on immune state
 * WHY:  Model inflammatory effects on neural activity
 * HOW:  Modulate threshold, learning rate, time constants
 *
 * @param network SNN network
 * @return SNN_SUCCESS on success
 *
 * COMPLEXITY: O(n_populations)
 */
int snn_network_apply_immune_modulation(snn_network_t* network);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get underlying neural_network handle
 *
 * WHAT: Access the underlying neural_network_t
 * WHY:  Direct access for advanced operations
 * HOW:  Return internal handle
 *
 * @param network SNN network
 * @return neural_network_t handle, or NULL
 */
neural_network_t snn_network_get_neural_net(snn_network_t* network);

/**
 * @brief Validate network structure
 *
 * WHAT: Check network integrity
 * WHY:  Debug and validation
 * HOW:  Verify magic number, population references, etc.
 *
 * @param network SNN network
 * @return SNN_SUCCESS if valid
 */
int snn_network_validate(const snn_network_t* network);

/**
 * @brief Save SNN network (config + neuron weights) to file
 * @return 0 on success, -1 on error
 */
int snn_network_save(snn_network_t* network, const char* path);

/**
 * @brief Load SNN network from file
 * @return Network handle, or NULL on error
 */
snn_network_t* snn_network_load(const char* path);

/**
 * @brief Create a hierarchical SNN with cortical-inspired population topology
 *
 * WHAT: Creates a 1.8M-neuron SNN organized into 46 populations across 8 tiers
 * WHY:  The SNN is the primary brain — needs biological cortical hierarchy
 * HOW:  Builds populations tier-by-tier with feedforward, recurrent, and skip
 *       connections using efficient direct-sampling connectivity
 *
 * @param n_inputs Brain input dimension (for input population sizing)
 * @param n_outputs Brain output dimension (for output population sizing)
 * @param target_total_neurons Target total neuron count (actual may differ slightly)
 * @return Network handle, or NULL on error
 */
snn_network_t* snn_create_hierarchical_network(
    uint32_t n_inputs,
    uint32_t n_outputs,
    uint32_t target_total_neurons);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SNN_NETWORK_H */
