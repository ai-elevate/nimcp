//=============================================================================
// nimcp_snn_types.h - Spiking Neural Network Core Types
//=============================================================================
/**
 * @file nimcp_snn_types.h
 * @brief Core type definitions for Spiking Neural Networks in NIMCP
 *
 * WHAT: Spiking Neural Network (SNN) orchestration layer types
 * WHY:  SNNs are the core computational substrate of NIMCP, providing
 *       biologically-accurate spike-based computation with precise timing
 * HOW:  Integrates with existing neuron_t, synapse_t, axon, and dendrite
 *       infrastructure - does NOT duplicate these structures
 *
 * INTEGRATION ARCHITECTURE:
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                        SNN MODULE (THIS FILE)                            │
 * │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
 * │  │ snn_network_t │  │snn_population_t│ │ snn_encoder_t │                │
 * │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘                │
 * │          │                  │                  │                        │
 * └──────────┼──────────────────┼──────────────────┼────────────────────────┘
 *            │                  │                  │
 * ┌──────────┼──────────────────┼──────────────────┼────────────────────────┐
 * │          ▼                  ▼                  ▼                        │
 * │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
 * │  │   neuron_t    │  │   synapse_t   │  │    axon_t     │                │
 * │  │ (neuralnet.h) │  │ (neuralnet.h) │  │  (axon.h)     │                │
 * │  └───────────────┘  └───────────────┘  └───────────────┘                │
 * │                    EXISTING INFRASTRUCTURE                               │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * BIOLOGICAL BASIS:
 * - Discrete spike events with precise timing (µs resolution)
 * - Spike-timing dependent plasticity (STDP) for learning
 * - Population coding for robust information representation
 * - Temporal coding for fine-grained information
 *
 * CODING STANDARDS:
 * - Guard clauses (no nested ifs)
 * - WHAT-WHY-HOW documentation
 * - Single Responsibility Principle
 * - Integration with existing infrastructure
 *
 * @author NIMCP Development Team
 * @date 2025-12-20
 * @version 1.0.0
 */

#ifndef NIMCP_SNN_TYPES_H
#define NIMCP_SNN_TYPES_H

#include <stdint.h>
#include "utils/bridge/nimcp_bridge_base.h"
#include <stdbool.h>
#include <stddef.h>

// Include existing infrastructure
#include "core/neuralnet/nimcp_neuralnet.h"
#include "core/neuron_types/nimcp_neuron_types.h"
#include "core/synapse_types/nimcp_synapse_types.h"
#include "utils/tensor/nimcp_tensor.h"

/* Substrate effects struct types (axon/dendrite). We embed values (not
 * pointers) in snn_network_s so the full definitions are required. Pulled
 * in via the small effects header — which itself only forward-declares
 * neural_substrate_t, so no cycle. */
#include "core/nimcp_axon_dendrite_substrate_bridge.h"  /* axon_/dendrite_substrate_effects_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Forward decl of the full substrate state struct. The pointer on
 * snn_network_s is borrowed (not owned) — whoever created the substrate
 * is responsible for its lifecycle. */
struct neural_substrate;

//=============================================================================
// Constants
//=============================================================================

/** Maximum number of populations in an SNN network.
 *  Raised from 64 to 128 for hierarchical cortical-layer topology
 *  at 1.8M-neuron scale (46 populations across 8 tiers). */
#define SNN_MAX_POPULATIONS 128

/** Maximum spike buffer size per neuron */
#define SNN_SPIKE_BUFFER_SIZE 256  /* DO NOT CHANGE — baked into checkpoint format (snn_neuron_t.spike_times[]) */

/** Default simulation timestep (ms) */
#define SNN_DT_DEFAULT 0.1f

/** Minimum timestep (ms) - for numerical stability */
#define SNN_DT_MIN 0.01f

/** Maximum timestep (ms) - for accuracy */
#define SNN_DT_MAX 1.0f

/** Default refractory period (ms) */
#define SNN_REFRACTORY_DEFAULT 2.0f

/** Magic number for validation */
#define SNN_MAGIC 0x534E4E00  /* "SNN\0" */

//=============================================================================
// Forward Declarations
//=============================================================================

typedef struct snn_network_s snn_network_t;
typedef struct snn_population_s snn_population_t;
typedef struct snn_encoder_s snn_encoder_t;
typedef struct snn_decoder_s snn_decoder_t;
typedef struct snn_config_s snn_config_t;
typedef struct snn_simulation_s snn_simulation_t;
typedef struct snn_training_ctx_s snn_training_ctx_t;
typedef struct snn_spike_s snn_spike_t;
typedef struct snn_spike_train_s snn_spike_train_t;

//=============================================================================
// Enumerations
//=============================================================================

/**
 * @brief Spike encoding methods
 *
 * WHAT: Methods to convert continuous values to spike trains
 * WHY:  SNNs operate on discrete spikes, not continuous values
 * HOW:  Each method has different properties for rate/temporal coding
 */
typedef enum {
    SNN_ENCODE_RATE = 0,        /**< Rate coding: value → firing rate */
    SNN_ENCODE_TEMPORAL,        /**< Temporal coding: value → spike timing */
    SNN_ENCODE_POPULATION,      /**< Population coding: value → population pattern */
    SNN_ENCODE_LATENCY,         /**< Latency coding: value → time-to-first-spike */
    SNN_ENCODE_BURST,           /**< Burst coding: value → burst count */
    SNN_ENCODE_PHASE,           /**< Phase coding: value → spike phase relative to oscillation */
    SNN_ENCODE_POISSON,         /**< Poisson process: value → stochastic spike rate */
    SNN_ENCODE_COUNT
} snn_encoding_t;

/**
 * @brief Spike decoding methods
 *
 * WHAT: Methods to convert spike trains to continuous values
 * WHY:  Output layer must produce usable values for downstream systems
 * HOW:  Integrate spikes over time or use first-spike timing
 */
typedef enum {
    SNN_DECODE_RATE = 0,        /**< Rate decoding: count spikes / time window */
    SNN_DECODE_FIRST_SPIKE,     /**< First-spike: earliest spiking neuron wins */
    SNN_DECODE_POPULATION,      /**< Population vector: weighted population activity */
    SNN_DECODE_MEMBRANE,        /**< Membrane potential: use final voltage */
    SNN_DECODE_COUNT
} snn_decoding_t;

/**
 * @brief SNN training modes
 *
 * WHAT: Methods to train SNNs with spike-based learning
 * WHY:  Standard backprop doesn't work with discrete spikes
 * HOW:  Surrogate gradients, local learning rules, or hybrid approaches
 *
 * BIOLOGICAL BASIS:
 * - STDP: Spike-timing dependent plasticity (local, biological)
 * - R-STDP: Reward-modulated STDP (reinforcement learning)
 * - eProp: Eligibility propagation (bio-plausible backprop)
 * - Surrogate: Smooth approximation of spike gradient
 */
typedef enum {
    SNN_TRAIN_STDP = 0,         /**< Local STDP rules (already in synapse_t) */
    SNN_TRAIN_R_STDP,           /**< Reward-modulated STDP (TD learning) */
    SNN_TRAIN_EPROP,            /**< Eligibility propagation (bio-plausible) */
    SNN_TRAIN_SURROGATE,        /**< Surrogate gradient backprop */
    SNN_TRAIN_SLAYER,           /**< SLAYER temporal credit assignment */
    SNN_TRAIN_DECOLLE,          /**< Deep Continuous Local Learning */
    SNN_TRAIN_COUNT
} snn_train_mode_t;

/**
 * @brief Surrogate gradient functions
 *
 * WHAT: Smooth approximations of the spike function derivative
 * WHY:  The derivative of a spike (step function) is a Dirac delta
 * HOW:  Replace delta with smooth function for gradient computation
 *
 * REFERENCE: Neftci et al. (2019) "Surrogate Gradient Learning in SNNs"
 */
typedef enum {
    SNN_SURROGATE_SIGMOID = 0,  /**< σ'(x) = σ(βx)(1-σ(βx)) */
    SNN_SURROGATE_FAST_SIGMOID, /**< x / (1 + |x|)² - faster computation */
    SNN_SURROGATE_ARCTAN,       /**< 1 / (1 + (πx)²) */
    SNN_SURROGATE_SUPERSPIKE,   /**< 1 / (β|x| + 1)² - SuperSpike paper */
    SNN_SURROGATE_TRIANGULAR,   /**< max(0, 1 - |x|/a) - simple piecewise */
    SNN_SURROGATE_RECTANGULAR,  /**< 1 if |x| < a else 0 - STE */
    SNN_SURROGATE_COUNT
} snn_surrogate_t;

/**
 * @brief Population topology types
 *
 * WHAT: Connectivity patterns within and between populations
 * WHY:  Different topologies serve different computational purposes
 * HOW:  Applied when creating populations or connecting them
 */
typedef enum {
    SNN_TOPO_FULL = 0,          /**< All-to-all connectivity */
    SNN_TOPO_RANDOM,            /**< Random sparse (p% connected) */
    SNN_TOPO_FEEDFORWARD,       /**< Strictly feedforward (no recurrence) */
    SNN_TOPO_RECURRENT,         /**< Full recurrent within population */
    SNN_TOPO_SMALL_WORLD,       /**< Watts-Strogatz small-world */
    SNN_TOPO_RESERVOIR,         /**< Echo state network (fixed random) */
    SNN_TOPO_COLUMN,            /**< Cortical column (6 layers) */
    SNN_TOPO_CUSTOM,            /**< User-defined connectivity */
    SNN_TOPO_COUNT
} snn_topology_t;

/**
 * @brief SNN simulation state health indicators
 *
 * WHAT: Health status of the spiking network simulation
 * WHY:  Detect numerical issues, runaway firing, or dead networks
 * HOW:  Monitor spike rates, membrane potentials, and weight norms
 */
typedef enum {
    SNN_STATE_HEALTHY = 0,      /**< Normal operation */
    SNN_STATE_SILENT,           /**< No spikes (dead network) */
    SNN_STATE_EXPLOSION,        /**< Runaway firing (too many spikes) */
    SNN_STATE_NAN_DETECTED,     /**< NaN in membrane potential or weights */
    SNN_STATE_INF_DETECTED,     /**< Inf detected */
    SNN_STATE_WEIGHT_EXPLOSION, /**< Weights exceeding bounds */
    SNN_STATE_UNSTABLE          /**< Oscillating or divergent */
} snn_state_health_t;

//=============================================================================
// Core Structures
//=============================================================================

/**
 * @brief Single spike event
 *
 * WHAT: Discrete spike with precise timing and source neuron
 * WHY:  SNNs communicate via discrete spike events, not continuous values
 * HOW:  Timestamp + neuron ID for event-driven simulation
 *
 * MEMORY: 16 bytes per spike
 */
struct snn_spike_s {
    uint64_t timestamp_us;      /**< Spike time in microseconds */
    uint32_t neuron_id;         /**< Source neuron ID */
    uint32_t population_id;     /**< Source population ID (for routing) */
};

/**
 * @brief Spike train for a single neuron
 *
 * WHAT: Time series of spikes for one neuron
 * WHY:  Store spike history for STDP, analysis, and decoding
 * HOW:  Circular buffer of spike timestamps
 *
 * MEMORY: ~2KB per neuron (256 spikes × 8 bytes)
 */
struct snn_spike_train_s {
    uint32_t neuron_id;                             /**< Neuron this train belongs to */
    uint64_t spike_times[SNN_SPIKE_BUFFER_SIZE];    /**< Circular buffer of spike times (µs) */
    uint32_t write_idx;                             /**< Next write position */
    uint32_t count;                                 /**< Number of spikes in buffer */
    uint32_t total_spikes;                          /**< Total spikes since reset */
    float inst_rate;                                /**< Instantaneous firing rate (Hz) */
    float avg_rate;                                 /**< Average firing rate (Hz) */
};

/**
 * @brief Spike encoder configuration
 *
 * WHAT: Configuration for converting values to spike trains
 * WHY:  Control encoding parameters (rate scaling, timing, etc.)
 */
typedef struct snn_encoder_config_s {
    snn_encoding_t method;      /**< Encoding method */
    float max_rate;             /**< Maximum firing rate for rate coding (Hz) */
    float min_rate;             /**< Minimum firing rate (Hz) */
    float time_window;          /**< Encoding time window (ms) */
    float threshold;            /**< Threshold for latency coding */
    uint32_t population_size;   /**< Neurons per encoded value */
    float sigma;                /**< Tuning curve width for population coding */
} snn_encoder_config_t;

/**
 * @brief Spike decoder configuration
 *
 * WHAT: Configuration for converting spike trains to values
 * WHY:  Control decoding parameters (time window, method, etc.)
 */
typedef struct snn_decoder_config_s {
    snn_decoding_t method;      /**< Decoding method */
    float time_window;          /**< Decoding time window (ms) */
    float decay_tau;            /**< Exponential decay time constant (ms) */
    bool use_softmax;           /**< Apply softmax for classification */
} snn_decoder_config_t;

/**
 * @brief Population of neurons in SNN
 *
 * WHAT: Group of neurons with shared properties and connectivity
 * WHY:  Organize neurons into functional groups (layers, columns, etc.)
 * HOW:  References existing neuron_t structures from neural_network
 *
 * DESIGN: This does NOT duplicate neurons - it references them!
 * The actual neuron_t structures live in neural_network_t
 */
struct snn_population_s {
    /* Identity */
    uint32_t id;                    /**< Population ID */
    char name[64];                  /**< Human-readable name */

    /* Neuron references (NOT owned - owned by neural_network_t) */
    uint32_t* neuron_ids;           /**< Array of neuron IDs in this population */
    uint32_t n_neurons;             /**< Number of neurons */
    neuron_type_t neuron_type;      /**< Neuron type (LIF, Izhikevich, etc.) */

    /* Spike trains for this population */
    snn_spike_train_t* spike_trains; /**< Spike history per neuron [n_neurons] */

    /* Population-level state (for efficient access) */
    nimcp_tensor_t* membrane_v;     /**< Membrane potentials [n_neurons] */
    nimcp_tensor_t* spike_output;   /**< Binary spike output [n_neurons] */
    nimcp_tensor_t* refractory;     /**< Remaining refractory time [n_neurons] */

    /* Topology */
    snn_topology_t topology;        /**< Internal connectivity pattern */
    float connectivity;             /**< Connectivity ratio [0, 1] for sparse */

    /* Statistics */
    float mean_rate;                /**< Population mean firing rate (Hz) */
    float population_synchrony;     /**< Synchrony measure [0, 1] */
    uint64_t total_spikes;          /**< Total spikes since reset */

    /* Lightweight mode (CSR synapse storage — replaces neuron_t dependency).
     * When lightweight=true, neurons are NOT in the neural_network_t.
     * All state lives in population tensors + CSR storage. */
    bool lightweight;                        /**< true = CSR mode, false = legacy */
    float* external_current;                 /**< [n_neurons] input currents */
    struct snn_csr_storage_s* incoming_csr;  /**< CSR incoming synapses (owned) */

    /* Homeostatic plasticity state (Turrigiano-style synaptic scaling).
     * Tracked per-population, updated every SNN step, applied every N
     * training steps by snn_homeostatic_apply() to keep firing rate near
     * the biological target. Without this, R-STDP drives the network
     * into either silent or saturated regimes. */
    float firing_rate_ema;   /**< EMA of per-step firing fraction [0, 1] */
    uint64_t rate_samples;   /**< Number of EMA updates since creation */

    /* Temporal history — per-step spike count ring buffer.
     * Enables Python-side FFT, cross-correlation, ISI analysis of
     * population dynamics without having to store full per-neuron
     * spike trains (which would be 100s of MB for 1.8M neurons).
     * Per-population aggregate is ~256 × 4 bytes = 1 KB per pop. */
    uint32_t spike_count_history[256];  /**< Last 256 steps of spike counts */
    uint32_t history_write_idx;         /**< Ring buffer write head */
    uint64_t history_total_steps;       /**< Monotonic counter for alignment */

    /* Intrinsic plasticity — per-neuron threshold adaptation.
     * Each neuron adjusts its own firing threshold based on recent activity:
     *   Fires too often → threshold rises (harder to fire)
     *   Fires too little → threshold falls (easier to fire)
     * Operates on ~second timescale, complementing synaptic scaling.
     * v_thresh_effective[n] = v_thresh_config + threshold_offset[n]
     * Per-neuron cost: 4 bytes × 1.8M = 7.2 MB total (trivial). */
    float* threshold_offset;        /**< [n_neurons] learned threshold shift, init 0 */
    float* neuron_rate_ema;         /**< [n_neurons] per-neuron firing rate EMA */

    /* Short-term synaptic depression — per-presynaptic-neuron.
     * When a neuron fires, its outgoing synapses become transiently
     * weaker (like neurotransmitter depletion in biology). This prevents
     * single hot pathways from runaway on the millisecond timescale,
     * long before synaptic scaling can notice.
     * Stored per-neuron (not per-synapse) for memory efficiency:
     * 1.8M × 4 bytes = 7.2 MB vs ~6 GB for per-synapse.
     * Effective synaptic drive = weight × (1 - depression_amount[src_neuron])
     * Depression recovers exponentially with time constant tau_d. */
    float* depression;              /**< [n_neurons] current depression [0..0.5] */

    /* Biophysical stability mechanisms — added to prevent saturation/collapse
     * oscillation observed in 1.8M-neuron pod runs (see
     * project_snn_structural_fixes_2026-04-21.md). Each is allocated only
     * when its feature is enabled via snn_tune_set_*_enabled(1.0). Freed
     * in snn_population_destroy. Opaque pointers; forward-declared so this
     * header does not depend on their internals. */
    struct snn_adaptation_state_s* ahp;     /**< fast spike-rate adaptation (AHP / M-current) */
    struct snn_adaptation_state_s* pump;    /**< slow Na+/K+ pump adaptation */
    struct snn_basket_pool_s*       basket; /**< fast-spiking inhibitory interneurons */
};

#define SNN_POP_HISTORY_LEN 256

/**
 * @brief SNN network configuration
 *
 * WHAT: Configuration parameters for SNN network
 * WHY:  Centralize network settings for creation and validation
 */
struct snn_config_s {
    /* Network dimensions */
    uint32_t n_inputs;              /**< Number of input neurons */
    uint32_t n_outputs;             /**< Number of output neurons */
    uint32_t n_hidden;              /**< Number of hidden neurons (0 = no hidden layer) */
    uint32_t n_populations;         /**< Number of populations (layers) */

    /* Simulation parameters */
    float dt;                       /**< Simulation timestep (ms) */
    float t_ref;                    /**< Default refractory period (ms) */
    float v_thresh;                 /**< Default spike threshold (mV) */
    float v_reset;                  /**< Default reset potential (mV) */
    float v_rest;                   /**< Default resting potential (mV) */
    float tau_mem;                  /**< Default membrane time constant (ms) */
    float tau_syn;                  /**< Default synaptic time constant (ms) */

    /* Input scaling */
    float input_current_scale;      /**< Scale factor for external input currents (mV per unit input).
                                     *   Default 30.0: maps input 0.7 → 21mV (suprathreshold),
                                     *   input 0.3 → 9mV (subthreshold). Must exceed
                                     *   (v_thresh - v_rest) for max input to produce spikes. */

    /* Encoding/Decoding */
    snn_encoder_config_t encoder;   /**< Input encoding configuration */
    snn_decoder_config_t decoder;   /**< Output decoding configuration */

    /* Learning */
    snn_train_mode_t train_mode;    /**< Training algorithm */
    snn_surrogate_t surrogate;      /**< Surrogate gradient function */
    float surrogate_beta;           /**< Surrogate gradient sharpness */
    float learning_rate;            /**< Base learning rate */
    bool enable_stdp;               /**< Enable STDP (use existing synapse_t.stdp_params) */
    bool enable_reward_modulation;  /**< Enable reward-modulated learning */

    /* Integration flags */
    bool enable_bio_async;          /**< Enable bio-async messaging */
    bool enable_immune;             /**< Enable immune system integration */
    bool use_axon_delays;           /**< Use existing axon infrastructure for delays */
    bool use_dendritic_integration; /**< Use existing dendrite infrastructure */

    /* Parallelization */
    bool enable_simd;               /**< Enable SIMD vectorization */
    uint32_t n_threads;             /**< Number of worker threads */
};

/**
 * @brief SNN simulation context
 *
 * WHAT: State for discrete-time spiking simulation
 * WHY:  Track simulation progress, buffer spikes, manage timing
 */
struct snn_simulation_s {
    /* Timing */
    uint64_t current_time_us;       /**< Current simulation time (µs) */
    uint64_t start_time_us;         /**< Simulation start time (µs) */
    uint64_t step_count;            /**< Number of simulation steps */
    float dt_ms;                    /**< Timestep in milliseconds */

    /* Spike event queue (priority queue by time) */
    snn_spike_t* spike_queue;       /**< Pending spike events */
    uint32_t queue_size;            /**< Current queue size */
    uint32_t queue_capacity;        /**< Queue capacity */

    /* Global state */
    snn_state_health_t health;      /**< Simulation health status */
    float total_energy;             /**< Cumulative spike energy (for efficiency) */

    /* Random state for stochastic operations */
    uint64_t rng_state;             /**< Random number generator state */
};

/**
 * @brief SNN training context
 *
 * WHAT: State for training SNNs with gradient-based or local learning
 * WHY:  Track eligibility traces, surrogate gradients, and credit assignment
 */
struct snn_training_ctx_s {
    snn_train_mode_t mode;          /**< Training mode */
    snn_surrogate_t surrogate;      /**< Surrogate gradient function */
    float surrogate_beta;           /**< Surrogate gradient sharpness */

    /* Eligibility traces (for eProp/R-STDP) */
    nimcp_tensor_t* eligibility;    /**< Eligibility traces [n_synapses] */
    float eligibility_decay;        /**< Trace decay rate */

    /* Surrogate gradients */
    nimcp_tensor_t* grad_membrane;  /**< Gradients w.r.t. membrane potential */
    nimcp_tensor_t* grad_weights;   /**< Gradients w.r.t. weights */

    /* Reward signal (for R-STDP) */
    float reward;                   /**< Current reward signal [-1, 1] */
    float reward_baseline;          /**< Baseline for variance reduction */

    /* Loss tracking */
    float current_loss;             /**< Current training loss */
    float smoothed_loss;            /**< Exponential moving average of loss */
};

/**
 * @brief SNN network statistics
 *
 * WHAT: Performance and activity metrics for SNN
 * WHY:  Monitor network health, efficiency, and training progress
 */
typedef struct snn_stats_s {
    /* Simulation metrics */
    uint64_t total_steps;           /**< Total simulation steps */
    uint64_t total_spikes;          /**< Total spikes generated */
    double total_compute_time_ms;   /**< Cumulative compute time */
    double avg_step_time_ms;        /**< Average time per step */

    /* Activity metrics */
    float mean_firing_rate;         /**< Network mean firing rate (Hz) */
    float max_firing_rate;          /**< Maximum neuron firing rate (Hz) */
    float sparsity;                 /**< Spike sparsity (% silent neurons) */
    float synchrony;                /**< Network synchrony [0, 1] */

    /* Energy efficiency */
    float spikes_per_sample;        /**< Average spikes per input sample */
    float energy_per_spike;         /**< Estimated energy per spike (pJ) */

    /* Health */
    snn_state_health_t health;      /**< Current health status */
    uint32_t silent_neurons;        /**< Number of never-firing neurons */
    uint32_t hyperactive_neurons;   /**< Neurons above max rate */

    /* GPU metrics */
    bool last_step_gpu;             /**< true if last step used GPU */
    uint32_t gpu_steps;             /**< Steps executed on GPU */
    uint32_t cpu_fallback_steps;    /**< Steps that fell back to CPU */
    double last_step_time_ms;       /**< Wall-clock time of last step */
    double last_isyn_time_ms;       /**< I_syn computation time of last step */

    /* Memory */
    size_t memory_usage_bytes;      /**< Current memory usage */
} snn_stats_t;

/**
 * @brief Complete SNN network
 *
 * WHAT: Top-level SNN orchestration structure
 * WHY:  Manage populations, simulation, training, and integrations
 * HOW:  References existing neural_network_t for actual neurons/synapses
 *
 * DESIGN PATTERN: Facade
 * - SNN is a facade over neural_network_t + additional spiking-specific features
 * - Does NOT duplicate neuron/synapse structures
 * - Provides spike encoding/decoding and training algorithms
 */
struct snn_network_s {
    /* Identity */
    uint32_t magic;                 /**< Magic number for validation (SNN_MAGIC) */
    uint32_t id;                    /**< Network instance ID */
    char name[64];                  /**< Network name */

    /* Underlying neural network (NOT owned - existing infrastructure) */
    neural_network_t neural_net;    /**< Existing neural network with neurons/synapses */

    /* Population organization */
    snn_population_t** populations; /**< Array of population pointers [n_populations] */
    uint32_t n_populations;         /**< Number of populations */

    /* Special populations */
    snn_population_t* input_pop;    /**< Input population (encoding layer) */
    snn_population_t* output_pop;   /**< Output population (decoding layer) */

    /* Encoding/Decoding */
    snn_encoder_t* encoder;         /**< Input spike encoder */
    snn_decoder_t* decoder;         /**< Output spike decoder */

    /* Configuration */
    snn_config_t config;            /**< Network configuration */

    /* Simulation context */
    snn_simulation_t* sim;          /**< Simulation state */

    /* Training context */
    snn_training_ctx_t* train_ctx;  /**< Training state (NULL if inference-only) */
    bool is_training;               /**< Training mode flag */

    /* Integration handles */
    void* bio_ctx;                  /**< bio_module_context_t (bio-async) */
    void* immune_bridge;            /**< snn_immune_bridge_t* */
    void* thread_pool;              /**< nimcp_thread_pool_t* */

    /* Thread safety */
    void* mutex;                    /**< nimcp_mutex_t* */

    /* GPU acceleration */
    void* gpu_lif_state;            /**< nimcp_lif_state_t* for GPU SNN (NULL if CPU-only) */
    void* gpu_ctx;                  /**< nimcp_gpu_context_t* for GPU SNN (NULL if CPU-only) */

    /* Persistent GPU tensors for I_syn computation (allocated once, reused per step).
     * Eliminates per-step tensor create/upload/download/destroy overhead. */
    void* gpu_spike_vector;         /**< nimcp_gpu_tensor_t*: flat spike vector [total_neurons] */
    void** gpu_csr_weights;         /**< nimcp_gpu_tensor_t*[n_populations]: CSR weights per pop */
    void** gpu_csr_col_idx;         /**< nimcp_gpu_tensor_t*[n_populations]: CSR col indices per pop */
    void** gpu_csr_row_ptr;         /**< nimcp_gpu_tensor_t*[n_populations]: CSR row pointers per pop */
    void** gpu_ext_current;         /**< nimcp_gpu_tensor_t*[n_populations]: external current per pop */
    void** gpu_isyn_output;         /**< nimcp_gpu_tensor_t*[n_populations]: I_syn output per pop */
    uint32_t gpu_n_persistent_pops; /**< Number of populations with persistent GPU tensors */

    /* Statistics */
    snn_stats_t stats;              /**< Network statistics */

    /* Substrate integration (Phase 1 biological-substrate adapter).
     * The substrate pointer is BORROWED — the network does not own it.
     * Cached effect structs are recomputed every N steps (controlled by
     * g_snn_substrate_update_period) and read inside the hot per-neuron
     * loop to modulate tau_mem, refractory period, spike survival, and
     * plasticity learning rate. See snn_network_attach_substrate. */
    /* not part of checkpoint schema — runtime pointers rebuilt on load */
    struct neural_substrate*     substrate;
    axon_substrate_effects_t     cached_axon_effects;
    dendrite_substrate_effects_t cached_dend_effects;
    uint32_t                     substrate_steps_since_update;
};

//=============================================================================
// Error Codes
//=============================================================================

#define SNN_SUCCESS                     0
#define SNN_ERROR_NULL_POINTER         -1
#define SNN_ERROR_INVALID_CONFIG       -2
#define SNN_ERROR_INVALID_DIMENSION    -3
#define SNN_ERROR_OUT_OF_MEMORY        -4
#define SNN_ERROR_INVALID_STATE        -5
#define SNN_ERROR_INVALID_NEURON       -6
#define SNN_ERROR_INVALID_POPULATION   -7
#define SNN_ERROR_SIMULATION_FAILED    -8
#define SNN_ERROR_ENCODING_FAILED      -9
#define SNN_ERROR_DECODING_FAILED     -10
#define SNN_ERROR_TRAINING_FAILED     -11
#define SNN_ERROR_THREAD_FAILURE      -12
#define SNN_ERROR_NOT_INITIALIZED     -13
#define SNN_ERROR_HEALTH_CHECK_FAILED -14
#define SNN_ERROR_OPERATION_FAILED    -15

//=============================================================================
// Bio-Async Module IDs for SNN
//=============================================================================

/** Bio-async module IDs for SNN subsystem (0x0600 - 0x060F) */
#define BIO_MODULE_SNN_CORE         0x0600
#define BIO_MODULE_SNN_POPULATION   0x0601
#define BIO_MODULE_SNN_ENCODER      0x0602
#define BIO_MODULE_SNN_DECODER      0x0603
#define BIO_MODULE_SNN_SIMULATION   0x0604
#define BIO_MODULE_SNN_TRAINING     0x0605
#define BIO_MODULE_SNN_IMMUNE       0x0606

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SNN_TYPES_H */
