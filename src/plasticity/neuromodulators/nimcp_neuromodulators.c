/**
 * @file nimcp_neuromodulators.c
 * @brief High-performance thread-safe neuromodulator system implementation
 *
 * ARCHITECTURAL OVERVIEW:
 * This module implements a biologically-inspired neuromodulator system using:
 * - Singleton Pattern: Global neuromodulator pool (volume transmission)
 * - Thread-Local Storage Pattern: Per-thread effect computation buffers
 * - Strategy Pattern: Receptor-specific modulation strategies
 * - Template Method Pattern: Common update loop with variable policies
 * - Monitor Pattern: Synchronized access to shared state
 *
 * DESIGN PATTERNS:
 * - Singleton: Global neuromodulator concentrations
 * - Thread-Local Storage: Effect buffers (eliminates contention)
 * - Strategy: Function pointers for receptor-specific effects
 * - Template Method: Update algorithm with pluggable decay functions
 * - Monitor: Reader-writer locks for concurrent access
 *
 * THREAD SAFETY STRATEGY:
 * - Reader-Writer Lock: Multiple concurrent readers, exclusive writers
 * - Atomic Operations: Lock-free statistics counters
 * - Thread-Local Storage: Per-thread effect pools (zero contention)
 * - Platform once: Safe lazy initialization
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Concentration update: O(1) - fixed number of neurotransmitters
 * - Effect computation: O(1) - direct formula, no loops, no locks
 * - Release events: O(1) - simple arithmetic with atomic counters
 * - Memory: O(1) per thread - thread-local allocation
 *
 * COMPLEXITY OPTIMIZATIONS:
 * - No malloc in hot paths: Thread-local static arrays
 * - Cache-friendly: Struct-of-arrays layout for SIMD
 * - Branchless: Use floating-point math instead of conditionals
 * - Inline: Hot path functions inlined
 * - Lock-free reads: Reader-writer lock allows concurrent queries
 *
 * BIOLOGICAL FIDELITY:
 * - Exponential decay: Matches reuptake/metabolism kinetics
 * - Reward prediction error: Temporal difference learning
 * - Volume transmission: Global broadcast with local receptors
 * - Receptor specificity: Different effects per receptor type
 *
 * @author NIMCP Development Team
 * @date 2025-11-01
 */

#include "plasticity/neuromodulators/nimcp_neuromodulators.h"
#include <stddef.h>  /* for NULL */
#include "plasticity/neuromodulators/nimcp_phasic_tonic.h"       // Phase C2.2 Enhancement #2
#include "plasticity/neuromodulators/nimcp_receptor_subtypes.h"  // Phase C2.2 Enhancement #1
#include "plasticity/neuromodulators/nimcp_vesicle_packaging.h"  // Phase C2.3 Enhancement #3
#include "plasticity/neuromodulators/nimcp_metabolic_pathways.h" // Phase C2.4 Enhancement #4
#include "plasticity/neuromodulators/nimcp_neuromodulators_sleep_bridge.h"  // Sleep integration
#include "cognitive/nimcp_sleep_wake.h"                          // Sleep state
#include "cognitive/nimcp_grief_and_loss.h"                      // Phase E1: Grief and Loss Understanding
#include "cognitive/nimcp_joy_euphoria.h"                        // Phase E2: Joy and Euphoria System Integration
#include "cognitive/nimcp_love_loyalty_friendship.h"             // Phase E4: Love, Loyalty, Friendship System Integration
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/platform/nimcp_platform_rwlock.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_wiring_helpers.h"
#include "security/nimcp_security.h"
#include "utils/tensor/nimcp_tensor.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <math.h>
#include <string.h>
#include <stdatomic.h>

#define LOG_MODULE "plasticity_neuromodulators"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(neuromodulators)

//=============================================================================
// Bio-Async Integration State (Global for module registration)
//=============================================================================

/**
 * @brief Bio-async state for neuromodulator system
 *
 * WHAT: Global singleton state for bio-async integration
 * WHY:  Bio-router requires static module registration
 * HOW:  Registered once, routes messages to appropriate system instance
 */
typedef struct {
    bio_module_context_t module_ctx;        /**< Bio-router module context */
    neuromodulator_system_t current_system; /**< Currently active system (set by last create) */
    bool initialized;                       /**< Whether bio-async is initialized */
    atomic_uint_fast64_t messages_processed;/**< Message processing counter */
    nimcp_mutex_t init_mutex;             /**< Mutex protecting initialization */
    bool mutex_initialized;                 /**< Whether mutex is initialized */
} neuromod_bio_state_t;

/** Global bio-async state for neuromodulator module */
static neuromod_bio_state_t g_neuromod_bio_state = {
    .module_ctx = NULL,
    .current_system = NULL,
    .initialized = false,
    .messages_processed = 0,
    .init_mutex = NIMCP_MUTEX_INITIALIZER,
    .mutex_initialized = true
};

//=============================================================================
// Medulla Integration: Forward Declarations
//=============================================================================

/* Forward declare brain type to avoid include conflicts with brain_region_t */
#ifndef BRAIN_T_DEFINED
#define BRAIN_T_DEFINED
typedef struct brain_struct* brain_t;
#endif

/* Forward declare medulla function - implemented in nimcp_brain_init_medulla.c */
float nimcp_brain_get_arousal_level(brain_t brain);

//=============================================================================
// Constants for Performance and Biological Realism
//=============================================================================

/**
 * WHAT: Epsilon for floating-point comparisons
 * WHY:  Prevents division by zero and handles numerical precision
 */
#define EPSILON 1e-10f

/**
 * WHAT: Maximum neuromodulator concentration
 * WHY:  Biological constraint - receptors saturate at high concentrations
 */
#define MAX_CONCENTRATION 1.0f

/**
 * WHAT: Minimum neuromodulator concentration
 * WHY:  Cannot have negative neurotransmitter levels
 */
#define MIN_CONCENTRATION 0.0f

/**
 * WHAT: Object pool size for effect computation buffers
 * WHY:  Eliminates malloc overhead in hot path. 32 concurrent computations
 *       should handle typical workloads without exhaustion.
 */
#define EFFECT_POOL_SIZE 32

/**
 * WHAT: Moving average window for statistics
 * WHY:  Smooths out transient spikes, matches biological timescales (~10s)
 */
#define STATS_WINDOW_SIZE 100

/**
 * WHAT: Default receptor density for unspecified receptors
 * WHY:  Most neurons have some baseline sensitivity to all neuromodulators
 */
#define DEFAULT_RECEPTOR_DENSITY 0.1f

//=============================================================================
// Thread-Local Storage Pattern - Zero-Contention Effect Computation
//=============================================================================

/**
 * @brief Thread-local effect computation buffer
 *
 * WHAT: Each thread has its own effect buffer
 * WHY:  Eliminates all contention and locking overhead. Threads never share
 *       effect buffers, so no synchronization needed.
 * HOW:  _Thread_local storage class specifier (C11)
 *
 * DESIGN PATTERN: Thread-Local Storage
 * COMPLEXITY: O(1) access, no synchronization overhead
 * THREAD SAFETY: Perfect - each thread has independent instance
 * PERFORMANCE: ~100x faster than mutex-protected pool
 *
 * BIOLOGICAL ANALOGY: Like each neuron having its own receptor machinery
 */
static _Thread_local modulation_effects_t thread_effect_buffer = {0};

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Neuromodulator system state (Singleton pattern with Monitor pattern)
 *
 * WHAT: Contains all system state, configuration, and synchronization primitives
 * WHY:  Encapsulates global state with thread-safe access control
 * HOW:  Opaque pointer pattern + reader-writer locks + atomics
 *
 * THREAD SAFETY ARCHITECTURE:
 * - rwlock: Protects concentrations (multiple readers, exclusive writers)
 * - Atomic counters: Lock-free statistics (release_counts, update_count)
 * - Regular mutex: Would serialize all access (too slow)
 * - Reader-writer lock: Allows parallel reads during non-update periods
 *
 * PERFORMANCE CONSIDERATIONS:
 * - Read path (get_levels): Only acquires read lock (parallel with other reads)
 * - Write path (update, release): Acquires write lock (exclusive)
 * - Effect computation: No locks (uses thread-local storage)
 *
 * INVARIANTS:
 * - All concentrations in [0, 1] (maintained under lock)
 * - Decay rates > 0
 * - Gains in [0, 1]
 * - Atomic counters monotonically increasing
 * - Lock ordering: Always acquire rwlock before accessing concentrations
 */
struct neuromodulator_system_struct {
    /* WHAT: Reader-writer lock for concentration access
     * WHY:  Allow multiple concurrent readers (common case), exclusive writers
     * PERFORMANCE: ~10x faster than mutex for read-heavy workloads
     * PATTERN: Monitor Pattern (synchronized shared state)
     * USES: NIMCP platform abstraction for cross-platform compatibility
     */
    nimcp_platform_rwlock_t rwlock;

    /* WHAT: Current global concentrations (volume transmission)
     * WHY:  Models cerebrospinal fluid / extracellular space
     * LAYOUT: Struct-of-arrays for SIMD vectorization
     * PROTECTED BY: rwlock (read for query, write for modification)
     */
    float concentrations[NEUROMOD_COUNT];

    /* WHAT: Baseline concentrations (homeostatic set points)
     * WHY:  System returns to baseline via decay
     * PROTECTED BY: rwlock (read-only after initialization)
     */
    float baselines[NEUROMOD_COUNT];

    /* WHAT: Decay time constants (seconds)
     * WHY:  Different neurotransmitters have different clearance rates
     * PROTECTED BY: rwlock (read-only after initialization)
     */
    float decay_times[NEUROMOD_COUNT];

    /* WHAT: Release gains (response magnitude per unit stimulus)
     * WHY:  Calibrates neuromodulator response strength
     * PROTECTED BY: rwlock (read-only after initialization)
     */
    float reward_dopamine_gain;
    float threat_norepinephrine_gain;
    float salience_acetylcholine_gain;
    float punishment_serotonin_gain;

    /* WHAT: Volume transmission parameters
     * WHY:  Models spatial diffusion of neuromodulators
     * PROTECTED BY: rwlock (read-only after initialization)
     */
    bool enable_volume_transmission;
    float diffusion_rate;

    /* WHAT: Statistics for monitoring and analysis
     * WHY:  Track system behavior over time
     * THREAD SAFETY: Atomic counters for lock-free increments
     */
    struct {
        /* WHAT: Non-atomic statistics (protected by rwlock)
         * WHY:  Updated during update() under write lock
         */
        float moving_averages[NEUROMOD_COUNT];  // Exponential moving average
        float variances[NEUROMOD_COUNT];         // Variance for each modulator
        float reward_prediction_error_sum;       // Sum of RPEs for accuracy

        /* WHAT: Atomic counters (lock-free)
         * WHY:  Incremented in release functions without lock overhead
         * PERFORMANCE: ~50x faster than mutex-protected increments
         */
        atomic_uint_fast64_t release_counts[NEUROMOD_COUNT]; // Release events
        atomic_uint_fast64_t update_count;                   // Update calls
        atomic_uint_fast64_t rpe_count;                      // RPE computations
    } stats;

    /* WHAT: Last update timestamp
     * WHY:  For computing Δt in decay equations
     * PROTECTED BY: rwlock (write)
     */
    uint64_t last_update_time;

    /* WHAT: Current sleep state for neuromodulator modulation
     * WHY:  Sleep states fundamentally alter neuromodulator profiles
     * HOW:  Sleep state modulates baseline concentrations and release dynamics
     * BIOLOGICAL: ACh, NE, 5-HT, DA all vary dramatically across sleep stages
     * PROTECTED BY: rwlock (read/write)
     */
    sleep_state_t current_sleep_state;

    // ===========================================================================
    // PHASE C2.2 ENHANCEMENTS: Phasic-Tonic Dynamics + Receptor Subtypes
    // ===========================================================================

    /* WHAT: Phasic-tonic dynamics for each neurotransmitter system
     * WHY:  Models burst (phasic) vs baseline (tonic) release
     * HOW:  Replaces simple concentration with dual-mode dynamics
     * PROTECTED BY: rwlock (read/write)
     */
    phasic_tonic_state_t dopamine_phasic_tonic;
    phasic_tonic_state_t serotonin_phasic_tonic;
    phasic_tonic_state_t norepinephrine_phasic_tonic;
    phasic_tonic_state_t acetylcholine_phasic_tonic;

    /* WHAT: Default receptor profiles for different brain regions
     * WHY:  Provides regional specialization (cortex vs striatum, etc.)
     * HOW:  Pre-computed profiles reduce per-neuron memory
     * PROTECTED BY: rwlock (read-only after initialization)
     */
    neuron_receptor_profile_t cortical_profile;
    neuron_receptor_profile_t striatal_profile;

    /* WHAT: Flag to enable Phase C2.2 enhancements
     * WHY:  Allows fallback to simple model for compatibility
     * HOW:  When true, uses phasic-tonic + receptors; when false, uses simple concentrations
     * PROTECTED BY: rwlock (read-only after initialization)
     */
    bool use_enhanced_dynamics;

    // ===========================================================================
    // PHASE C2.3 ENHANCEMENT: Synaptic Vesicle Packaging
    // ===========================================================================

    /* WHAT: Vesicle pools for each neurotransmitter system
     * WHY:  Models short-term synaptic plasticity (facilitation & depression)
     * HOW:  Each neurotransmitter has independent vesicle dynamics
     * BIOLOGICAL: Three-pool model (RRP, recycling, reserve) with quantal release
     * PROTECTED BY: rwlock (read/write)
     */
    vesicle_pool_state_t dopamine_vesicles;
    vesicle_pool_state_t serotonin_vesicles;
    vesicle_pool_state_t norepinephrine_vesicles;
    vesicle_pool_state_t acetylcholine_vesicles;

    /* WHAT: Flag to enable Phase C2.3 vesicle packaging
     * WHY:  Allows fallback for compatibility or performance
     * HOW:  When true, uses vesicle-based release; when false, uses direct release
     * PROTECTED BY: rwlock (read-only after initialization)
     */
    bool use_vesicle_packaging;

    // ===========================================================================
    // PHASE C2.4 ENHANCEMENT: Metabolic Pathways
    // ===========================================================================

    /* WHAT: Metabolic state for each neurotransmitter system
     * WHY:  Models complete lifecycle (synthesis, degradation, reuptake)
     * HOW:  Each neurotransmitter has independent metabolism
     * BIOLOGICAL: Synthesis (precursor→NT), degradation (MAO/COMT), reuptake (DAT/SERT/NET)
     * PROTECTED BY: rwlock (read/write)
     */
    metabolic_state_t dopamine_metabolism;
    metabolic_state_t serotonin_metabolism;
    metabolic_state_t norepinephrine_metabolism;
    metabolic_state_t acetylcholine_metabolism;

    /* WHAT: Flag to enable Phase C2.4 metabolic pathways
     * WHY:  Allows fallback for compatibility or performance
     * HOW:  When true, uses metabolic dynamics; when false, uses direct release
     * PROTECTED BY: rwlock (read-only after initialization)
     */
    bool use_metabolic_pathways;

    /* WHAT: Phase E1 grief system integration (cognitive pipeline)
     * WHY:  Grief affects neuromodulator levels (serotonin↓, dopamine↓, norepinephrine↑)
     * HOW:  Query grief system for modulation factors, apply to neuromodulator release
     * BIOLOGICAL: Grief causes measurable changes in neurotransmitter levels
     * PROTECTED BY: rwlock (read/write)
     */
    grief_system_t* grief_system;
    bool use_grief_integration;

    /* WHAT: Phase E2 joy/euphoria system integration (cognitive pipeline)
     * WHY:  Joy/euphoria enhance neuromodulator levels (dopamine↑, serotonin↑)
     * HOW:  Query joy system for modulation factors, apply to neuromodulator release
     * BIOLOGICAL: Positive emotions boost dopamine and serotonin
     * PROTECTED BY: rwlock (read/write)
     */
    joy_system_t* joy_system;
    bool use_joy_integration;

    /* WHAT: Phase E4 social bond system integration (cognitive pipeline)
     * WHY:  Social bonds affect neuromodulator levels (dopamine↑, oxytocin↑)
     * HOW:  Query social system for modulation factors, apply to neuromodulator release
     * BIOLOGICAL: Love/friendship boost dopamine and oxytocin, loneliness reduces dopamine
     * PROTECTED BY: rwlock (read/write)
     */
    social_bond_system_t* social_system;
    bool use_social_integration;

    /* WHAT: Brain reference for medulla integration
     * WHY:  Medulla arousal modulates norepinephrine (stress response)
     * HOW:  Query medulla arousal, adjust NE release accordingly
     * BIOLOGICAL: Locus coeruleus NE release scales with brainstem arousal
     * PROTECTED BY: rwlock (read/write)
     */
    void* brain_ref;
    bool use_medulla_integration;
};


// Forward declarations for static functions (SRP split)
static inline float clamp(float value, float min, float max);
static inline float exponential_decay(float current, float baseline, float dt, float tau);
static inline float update_ema(float current_avg, float new_value, float alpha);
static neuromodulator_type_t bio_channel_to_neuromod_type(nimcp_bio_channel_type_t channel);
static nimcp_error_t neuromod_handle_release_message( const void* msg, size_t msg_size, nimcp_bio_promise_t response_promise, void* user_data);
static nimcp_error_t neuromod_handle_learning_rate_message( const void* msg, size_t msg_size, nimcp_bio_promise_t response_promise, void* user_data);
static int neuromodulator_wiring_handler_callback( bio_module_context_t ctx, const bio_message_type_t* message_types, uint32_t message_count, void* user_data );
static nimcp_error_t neuromod_bio_async_init(neuromodulator_system_t system);
static void neuromod_bio_async_shutdown(void);

//=============================================================================
// SRP Split: Function implementations organized by responsibility
//=============================================================================
#include "nimcp_neuromodulators_part_helpers.c"  // 4 functions: helpers
#include "nimcp_neuromodulators_part_processing.c"  // 5 functions: processing
#include "nimcp_neuromodulators_part_lifecycle.c"  // 11 functions: lifecycle
#include "nimcp_neuromodulators_part_core.c"  // 15 functions: core
#include "nimcp_neuromodulators_part_accessors.c"  // 43 functions: accessors
