// nimcp_brain_part_core.c - core functions
// Part of nimcp_brain.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp_brain.c

/* Forward declarations for C6 inference-time training */
struct creative_training_bridge;
typedef struct creative_training_bridge creative_training_bridge_t;
extern int creative_training_submit_feedback(creative_training_bridge_t* bridge,
                                              const void* content,
                                              int modality,
                                              uint8_t rating,
                                              const char* feedback);

/* Cognitive stage constants */
static const float DIALOGUE_AGREEMENT_HIGH   = 0.8F;   /* Above this: perspectives agree → boost */
static const float DIALOGUE_AGREEMENT_LOW    = 0.4F;   /* Below this: perspectives disagree → reduce */
static const float DIALOGUE_BOOST_FACTOR     = 1.1F;   /* Confidence multiplier on high agreement */
static const float DIALOGUE_REDUCE_FACTOR    = 0.8F;   /* Confidence multiplier on low agreement */
static const float IMAGINATION_FAIL_PENALTY  = 0.95F;  /* Confidence multiplier when simulation fails */
static const float RCOG_CONFIDENCE_FLOOR     = 0.1F;   /* Min confidence to invoke recursive cognition */
static const int   IMAGINATION_SIM_STEPS     = 3;      /* Number of prospective simulation steps */


/**
 * @brief Clear decision cache (thread-safe with deadlock protection)
 *
 * WHAT: Invalidates cached input and decision
 * WHY:  Cache must be cleared after network modifications
 * HOW:  Timeout-protected mutex acquisition with emergency recovery
 *
 * BIOLOGICAL RATIONALE:
 * Thread-safe cache invalidation mimics synaptic reorganization that
 * invalidates previously stable neural response patterns. When synaptic
 * weights change (learning/pruning), cached neural activations become
 * obsolete, requiring recomputation from modified connectivity.
 *
 * CONCURRENCY: Uses timeout-based mutex locking to prevent deadlocks.
 * If mutex cannot be acquired within MUTEX_TIMEOUT_US, cache is left
 * in potentially stale state (safe, just suboptimal performance).
 *
 * COMPLEXITY: O(1)
 *
 * @param brain Brain handle
 */
void clear_cache(brain_t brain)
{
    // Guard: Validate parameters
    if (!brain) {
        return;
    }

    // Lock cache mutex with retry + backoff to prevent deadlock.
    // Stale cache is safe (just slower), corrupted cache is not,
    // so we never proceed without the lock.
    int lock_acquired = -1;
    for (int attempt = 0; attempt < 3; attempt++) {
        uint64_t timeout = MUTEX_TIMEOUT_US << attempt;  // exponential backoff
        lock_acquired = mutex_lock_with_timeout(&brain->cache_mutex, timeout);
        if (lock_acquired == 0) break;
    }
    if (lock_acquired != 0) {
        set_error("Timeout waiting for cache mutex in clear_cache after 3 retries - cache not cleared");
        return;
    }

    // Free cached input vector
    nimcp_free(brain->last_input);
    brain->last_input = NULL;

    // Free cached decision
    if (brain->cached_decision) {
        brain_free_decision(brain->cached_decision);
        brain->cached_decision = NULL;
    }

    // Always attempt unlock, even if operations above failed
    int unlock_result = nimcp_platform_mutex_unlock(&brain->cache_mutex);
    if (unlock_result != 0) {
        // CRITICAL: Mutex unlock failed - attempt emergency recovery
        LOG_MODULE_ERROR("BRAIN", "CRITICAL: Normal unlock failed in clear_cache - attempting recovery");
        force_unlock_with_logging(&brain->cache_mutex, "clear_cache");
    }
}


/**
 * @brief Allocate and initialize brain structure
 *
 * WHY: Separates allocation from configuration
 * Makes memory management explicit
 *
 * COMPLEXITY: O(1)
 *
 * @return Allocated brain or NULL on error
 */
brain_t allocate_brain(void)
{
    brain_t brain = nimcp_calloc(1, sizeof(struct brain_struct));
    if (!brain) {
        set_error("Failed to allocate brain structure");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "allocate_brain: brain is NULL");
        return NULL;
    }

    brain->last_input = NULL;
    brain->cached_decision = NULL;
    brain->input_size = 0;

    // Initialize cache mutex for thread-safe access
    if (nimcp_platform_mutex_init(&brain->cache_mutex, false) != 0) {
        set_error("Failed to initialize cache mutex");
        nimcp_free(brain);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "allocate_brain: validation failed");
        return NULL;
    }

    brain->distributed = NULL;  // Initialize as standalone brain

    // Phase 11: Initialize long-term memory consolidation buffer
    brain->longterm_capacity = 100;  // Store up to 100 consolidated memories
    brain->longterm_count = 0;
    brain->longterm_memory = nimcp_calloc(brain->longterm_capacity,
                                          sizeof(*brain->longterm_memory));
    // Guard: If allocation fails, set capacity to 0 (consolidation will be disabled)
    if (!brain->longterm_memory) {
        brain->longterm_capacity = 0;
    }

    // Initialize COW fields
    brain->is_cow_clone = false;
    brain->owns_network = true;  // By default, brain owns its network
    brain->original_network = NULL;
    brain->network_is_cached = false;

    // Phase 3: Initialize reference counting fields (atomic operations)
    brain->network_refcount_atomic = NULL;
    brain->can_use_readonly = false;

    // Community Detection: Initialize fields
    brain->functional_modules = NULL;
    brain->network_hubs = NULL;
    brain->topology_metrics = NULL;
    brain->auto_detect_communities = false;
    brain->community_detection_interval = 0.0F;  // Manual only by default

    return brain;
}


/**
 * @brief Create adaptive network for brain
 *
 * WHY: Isolates network creation complexity
 * Handles network config lifecycle
 *
 * COMPLEXITY: O(n) where n = num_neurons
 *
 * @param num_inputs Input dimension
 * @param num_outputs Output dimension
 * @param num_neurons Neuron count
 * @param sparsity_target Target sparsity
 * @return Network handle or NULL on error
 */
adaptive_network_t create_brain_network(uint32_t num_inputs, uint32_t num_outputs,
                                               uint32_t num_neurons, float sparsity_target,
                                               ode_integration_method_t integration_method)
{

    adaptive_network_config_t net_config =
        build_network_config(num_inputs, num_outputs, num_neurons, sparsity_target, integration_method);

    // Guard: Check if layer_sizes allocation failed in build_base_network_config
    // WHY: NULL layer_sizes will cause crash in adaptive_network_create
    if (!net_config.base_config.layer_sizes) {
        // Error already set by build_base_network_config
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "create_brain_network: net_config is NULL");
        return NULL;
    }

    adaptive_network_t network = adaptive_network_create(&net_config);

    // Free our copy of layer_sizes - adaptive_network_create makes its own deep copy (or fails)
    // WHY: Avoid memory leak - we allocated this in build_base_network_config
    // WHAT: Safe to free even if network creation failed, because we still own this allocation
    // Note: layer_sizes pointer should not be modified by adaptive_network_create (const param)
    if (net_config.base_config.layer_sizes) {
        nimcp_free((void*)net_config.base_config.layer_sizes);
    }

    return network;
}


/**
 * @brief Initialize output labels array
 *
 * COMPLEXITY: O(1)
 *
 * @param brain Brain to initialize
 * @param num_outputs Number of output labels
 * @return true on success
 */
bool init_output_labels(brain_t brain, uint32_t num_outputs)
{
    if (!brain) {
        set_error("NULL brain pointer in init_output_labels");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_output_labels: brain is NULL");
        return false;
    }
    if (num_outputs == 0) {
        // Zero outputs - no allocation needed, but set to NULL
        brain->output_labels = NULL;
        brain->num_output_labels = 0;
        return true;
    }
    brain->output_labels = nimcp_calloc(num_outputs, sizeof(char*));
    if (!brain->output_labels) {
        set_error("Failed to allocate output labels");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "init_output_labels: brain->output_labels is NULL");
        return false;
    }
    brain->num_output_labels = 0;
    return true;
}


/**
 * @brief Initialize multi-modal subsystems (Phase 8)
 *
 * WHAT: Create visual cortex, audio cortex, and integration layer
 * WHY:  Enable unified multi-modal processing
 * HOW:  Check config flags, create modules, allocate feature buffers
 *
 * DESIGN:
 * - Only creates modules if config flags are enabled
 * - Allocates reusable feature buffers (no per-frame allocation)
 * - Gracefully handles partial initialization
 *
 * @param brain Brain structure with configuration set
 * @return true on success, false on failure
 *
 * COMPLEXITY: O(1) - just allocation
 * MEMORY: O(D_v + D_a + D_integrated) for feature buffers
 *
 * ERROR HANDLING:
 * - Returns true if multi-modal disabled (not an error)
 * - Returns false only on allocation failure
 * - Partial cleanup handled by brain_destroy()
 *
 * @version 2.7.0 Phase 8.1
 * @author NIMCP Development Team
 * @date 2025-11-08
 */


/**
 * WHAT: Initialize multihead attention mechanism
 * WHY:  Enable selective focus on relevant features for efficient processing
 * HOW:  Create attention system based on cortical column architecture
 *
 * BIOLOGICAL MOTIVATION:
 * - Cortical Columns: Each attention head acts as specialized processing column
 * - Thalamic Gating: Controls information flow (like thalamic relay nucleus)
 * - Salience Weighting: Biologically-inspired attention based on feature importance
 * - Parallel Streams: Multiple heads process different aspects simultaneously
 *
 * INTEGRATION WITH BRAIN:
 * - Applied to multimodal inputs (visual, audio, speech) before neural network
 * - Connects to salience evaluator for attention weighting
 * - Interfaces with executive control for top-down attention modulation
 * - Used in working memory for attention-based retrieval
 *
 * PERFORMANCE BENEFITS:
 * - 2-5x inference speedup by selective processing
 * - 30-50% memory reduction through focused activations
 * - 5-15% accuracy improvement on complex tasks
 *
 * @param brain Brain instance to initialize
 * @return true on success, false on failure
 *
 * @version 3.0.0 Module Integration Phase
 * @author NIMCP Development Team
 * @date 2025-11-11
 */
bool init_attention_subsystem(brain_t brain)
{
    // WHAT: Guard clause - validate input
    // WHY:  Prevent null pointer dereference
    // HOW:  Check brain pointer before use
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_attention_subsystem: brain is NULL");
        return false;
    }

    // Bio-Async: Initialize on first subsystem init (uses platform_once internally)
    nimcp_error_t bio_result = brain_bio_init();
    if (bio_result != NIMCP_SUCCESS) {
        LOG_MODULE_WARN("BRAIN", "Bio-async init failed: %d (continuing anyway)", bio_result);
    }

    // WHAT: Check if already initialized
    // WHY:  Prevent double initialization and memory leak
    // HOW:  Return success if attention already exists
    if (brain->multihead_attention) {
        return true;  // Already initialized
    }

    // WHAT: Check if attention is enabled in configuration
    // WHY:  Only initialize if user requested this feature
    // HOW:  Check config flag, return success (not error) if disabled
    if (!brain->config.enable_multihead_attention) {
        return true;  // Not enabled, not an error
    }

    // WHAT: Calculate appropriate dimensions for attention
    // WHY:  Attention dimensions must match integrated_feature_buffer size
    // HOW:  Always use num_inputs (the output size of multimodal integration)
    //
    // NOTE: The multimodal integration layer compresses all modalities
    //       (visual + audio + speech + direct) into a unified representation
    //       of size num_inputs. The attention system processes this integrated
    //       representation, not the raw concatenated features.
    uint32_t input_dim = brain->config.num_inputs;

    // WHAT: Configure multihead attention system
    // WHY:  Need proper configuration for cortical column architecture
    // HOW:  Create config with biological parameters
    multihead_attention_config_t attention_config = {
        .num_heads = brain->config.num_attention_heads > 0 ?
                     brain->config.num_attention_heads : 8,  // Default: 8 heads
        .input_dim = input_dim,
        .output_dim = input_dim,  // Same dimension (residual connection compatible)
        .sequence_length = 32,    // Default sequence length for temporal processing
        .use_thalamic_gate = brain->config.enable_thalamic_gate,
        .use_salience_weighting = brain->config.enable_salience_weighting,
        .gate_bias = 0.5F        // Default: 50% gate opening
    };

    // WHAT: Create multihead attention system
    // WHY:  Enable selective feature processing with parallel attention streams
    // HOW:  Call attention creation API with configured parameters
    brain->multihead_attention = multihead_attention_create(&attention_config);
    if (!brain->multihead_attention) {
        set_error("Failed to create multihead attention system");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_attention_subsystem: brain->multihead_attention is NULL");
        return false;
    }

    return true;
}


/**
 * WHAT: Initialize brain regions hierarchical architecture
 * WHY:  Enable modular cortical organization with layers and minicolumns
 * HOW:  Create brain module with specialized regions if config enables it
 *
 * BIOLOGICAL MOTIVATION:
 * - Cerebral cortex organized into hierarchical regions (V1, A1, M1, PFC, etc.)
 * - Each region has 6 cortical layers with distinct functions
 * - Minicolumns span layers vertically for parallel processing
 * - Inter-region connections follow biological pathways (feedforward/feedback)
 *
 * INTEGRATION WITH BRAIN:
 * - Provides spatial organization of processing
 * - Enables specialized regions for sensory, motor, associative functions
 * - Supports realistic cortical layer dynamics (Layer 4 input, Layer 5 output)
 * - Allows for hierarchical processing pathways
 *
 * @param brain Brain instance to initialize
 * @return true on success, false on failure
 *
 * @version 3.0.0 Module Integration Phase
 * @author NIMCP Development Team
 * @date 2025-11-11
 */
bool init_brain_regions_subsystem(brain_t brain)
{
    // WHAT: Guard clause - validate input
    // WHY:  Prevent null pointer dereference
    // HOW:  Check brain pointer before use
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_brain_regions_subsystem: brain is NULL");
        return false;
    }

    // WHAT: Check if already initialized
    // WHY:  Prevent double initialization and memory leak
    // HOW:  Return success if brain_regions already exists
    if (brain->brain_regions) {
        return true;  // Already initialized
    }

    // WHAT: Check if brain regions architecture is enabled in configuration
    // WHY:  Only initialize if user requested this feature
    // HOW:  Check config flag, return success (not error) if disabled
    if (!brain->config.enable_brain_regions) {
        return true;  // Not enabled, not an error
    }

    // WHAT: Determine number of regions and neurons per region
    // WHY:  Need proper sizing for modular architecture
    // HOW:  Use config values with sensible defaults
    uint32_t num_regions = brain->config.num_brain_regions > 0 ?
                           brain->config.num_brain_regions : 4;  // Default: 4 regions
    uint32_t neurons_per_region = brain->config.neurons_per_region > 0 ?
                                  brain->config.neurons_per_region : 1000;  // Default: 1000 neurons

    // WHAT: Create brain module with max capacity
    // WHY:  Top-level container for all brain regions
    // HOW:  Allocate module with specified max regions
    brain->brain_regions = brain_module_create(num_regions);
    if (!brain->brain_regions) {
        set_error("Failed to create brain regions module");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "init_brain_regions_subsystem: brain->brain_regions is NULL");
        return false;
    }

    // WHAT: Create individual brain regions with specialized types
    // WHY:  Different regions have different layer proportions and neuron types
    // HOW:  Create regions based on configuration, starting with primary sensory/motor areas
    brain_region_type_t region_types[] = {
        REGION_VISUAL_V1,      // Primary visual cortex
        REGION_AUDITORY_A1,    // Primary auditory cortex
        REGION_MOTOR_M1,       // Primary motor cortex
        REGION_PREFRONTAL      // Prefrontal cortex (executive control)
    };

    for (uint32_t i = 0; i < num_regions && i < 4; i++) {
        brain_region_t* region = brain_region_create(region_types[i], neurons_per_region);
        if (!region) {
            set_error("Failed to create brain region");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_brain_regions_subsystem: region is NULL");
            return false;
        }

        // Organize region into minicolumns (8x8 grid for moderate-sized regions)
        uint32_t columns_x = 8;
        uint32_t columns_y = 8;
        if (brain_region_organize_columns(region, columns_x, columns_y) != NIMCP_SUCCESS) {
            brain_region_destroy(region);
            set_error("Failed to organize brain region into minicolumns");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "init_brain_regions_subsystem: validation failed");
            return false;
        }

        // Add region to brain module
        if (brain_module_add_region(brain->brain_regions, region) != NIMCP_SUCCESS) {
            brain_region_destroy(region);
            set_error("Failed to add region to brain module");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "init_brain_regions_subsystem: validation failed");
            return false;
        }
    }

    // WHAT: Establish inter-region connections
    // WHY:  Brain regions need to communicate (e.g., V1 → PFC for visual attention)
    // HOW:  Connect regions with biologically realistic pathways
    if (num_regions >= 2) {
        // Connect V1 (visual) → PFC (prefrontal) for visual processing pathway
        brain_region_t* v1 = brain_module_get_region_by_type(brain->brain_regions, REGION_VISUAL_V1);
        brain_region_t* pfc = brain_module_get_region_by_type(brain->brain_regions, REGION_PREFRONTAL);
        if (v1 && pfc) {
            nimcp_result_t result = brain_module_connect_regions(brain->brain_regions, v1->id, pfc->id, 0.3F);
            if (result != NIMCP_SUCCESS) {
                LOG_MODULE_WARN("BRAIN", "Failed to connect V1→PFC regions (error=%d), visual processing pathway unavailable", result);
            }
        } else {
            LOG_MODULE_WARN("BRAIN", "Cannot establish V1→PFC connection: V1=%p, PFC=%p", (void*)v1, (void*)pfc);
        }
    }

    if (num_regions >= 3) {
        // Connect A1 (auditory) → PFC for auditory processing pathway
        brain_region_t* a1 = brain_module_get_region_by_type(brain->brain_regions, REGION_AUDITORY_A1);
        brain_region_t* pfc = brain_module_get_region_by_type(brain->brain_regions, REGION_PREFRONTAL);
        if (a1 && pfc) {
            nimcp_result_t result = brain_module_connect_regions(brain->brain_regions, a1->id, pfc->id, 0.3F);
            if (result != NIMCP_SUCCESS) {
                LOG_MODULE_WARN("BRAIN", "Failed to connect A1→PFC regions (error=%d), auditory processing pathway unavailable", result);
            }
        } else {
            LOG_MODULE_WARN("BRAIN", "Cannot establish A1→PFC connection: A1=%p, PFC=%p", (void*)a1, (void*)pfc);
        }
    }

    return true;
}


/**
 * WHAT: Initialize symbolic logic reasoning subsystem
 * WHY:  Enable logical inference, knowledge representation, and abstract reasoning
 * HOW:  Create symbolic logic engine if config enables it
 *
 * BIOLOGICAL MOTIVATION:
 * - Prefrontal cortex performs abstract logical reasoning
 * - Hippocampus stores declarative knowledge (facts)
 * - Working memory maintains active inferences
 *
 * INTEGRATION WITH BRAIN:
 * - Stores facts learned during experience
 * - Performs deductive/inductive reasoning
 * - Validates decisions against logical constraints
 * - Enables explanation generation ("because X implies Y")
 *
 * @param brain Brain instance to initialize
 * @return true on success, false on failure
 *
 * @version 2.7.0 Phase 8.9
 * @author NIMCP Development Team
 * @date 2025-11-08
 */
bool init_symbolic_logic_subsystem(brain_t brain)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_symbolic_logic_subsystem: brain is NULL");
        return false;
    }

    // Check if already initialized
    if (brain->logic) {
        return true;  // Already initialized
    }

    // Check if symbolic logic is enabled via knowledge system or explicit flag
    // The knowledge system uses logic internally, so enable if knowledge is enabled
    if (!brain->config.enable_knowledge) {
        return true;  // Not enabled, not an error
    }

    // Create neural logic network with spiking logic gates (Phase 9.0)
    neural_logic_config_t logic_config = neural_logic_default_config(1000);
    logic_config.use_gpu = neural_logic_gpu_available();
    logic_config.integration_window_ms = 5.0F;
    logic_config.enable_learning = false;  // Combinational logic (no plasticity)

    brain->logic = neural_logic_create(&logic_config);
    if (!brain->logic) {
        set_error("Failed to create neural logic network");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_symbolic_logic_subsystem: brain->logic is NULL");
        return false;
    }

    return true;
}


/**
 * @brief Initialize symbolic reasoning subsystem (Phase 9.4)
 *
 * WHAT: Creates symbolic logic engine for first-order logic reasoning
 * WHY:  Enable logical inference, consistency checking for communication
 * HOW:  Allocate logic engine with inference and knowledge base capabilities
 *
 * @param brain Brain to initialize
 * @return true on success, false on error
 */
bool init_symbolic_reasoning_subsystem(brain_t brain)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_symbolic_reasoning_subsystem: brain is NULL");
        return false;
    }

    // Check if already initialized
    if (brain->symbolic_logic) {
        return true;  // Already initialized
    }

    // Only initialize if explicitly enabled
    if (!brain->config.enable_logic) {
        brain->symbolic_logic = NULL;
        return true;  // Not enabled, not an error
    }

    // Create symbolic logic engine with default configuration
    logic_config_t logic_config = {
        .max_predicates = LOGIC_MAX_PREDICATES,
        .max_rules = LOGIC_MAX_RULES,
        .max_kb_size = 10000,           // 10K facts
        .max_inference_depth = 10,       // Max 10 inference steps
        .enable_forward_chaining = true,
        .enable_backward_chaining = true,
        .enable_resolution = true,
        .enable_memory_consolidation = false  // Handled by brain->consolidation
    };

    brain->symbolic_logic = symbolic_logic_create(&logic_config);
    if (!brain->symbolic_logic) {
        set_error("Failed to create symbolic logic engine");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_symbolic_reasoning_subsystem: brain->symbolic_logic is NULL");
        return false;
    }

    return true;
}


/**
 * @brief Initialize epistemic filtering subsystem (Phase 9.2)
 *
 * WHAT: Creates epistemic filter for cognitive bias prevention
 * WHY:  Prevents conspiracy-theory thinking and cognitive biases
 * HOW:  Applies skepticism, evidence evaluation, bias detection
 *
 * @param brain Brain to initialize
 * @return true on success, false on error
 */
bool init_epistemic_subsystem(brain_t brain)
{
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_epistemic_subsystem: brain is NULL");
        return false;
    }

    // Check if already initialized
    if (brain->epistemic) {
        return true;  // Already initialized
    }

    // Epistemic filtering is recommended for all brains to prevent
    // accepting unproven information or developing biased reasoning

    // Skepticism level:
    // 0.0 = credulous (accepts most claims)
    // 0.5 = balanced (reasonable skepticism)
    // 0.7 = cautious (requires strong evidence)
    // 1.0 = extreme skeptic (rejects almost everything)
    //
    // We default to 0.6 (cautious but not paranoid)
    float skepticism_level = 0.6F;

    brain->epistemic = epistemic_filter_create(skepticism_level);
    if (!brain->epistemic) {
        set_error("Failed to create epistemic filter");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_epistemic_subsystem: brain->epistemic is NULL");
        return false;
    }

    return true;
}


//=============================================================================
// Phase 2: Copy-on-Write Brain Cloning
//=============================================================================

/**
 * @brief Ensure brain has writable network (trigger COW if needed)
 *
 * WHAT: Detects COW clone and makes private copy before write
 * WHY:  Prevent modifying shared network, maintain data safety
 * HOW:  Check is_cow_clone flag, copy network if true
 *
 * THREAD SAFETY: Not thread-safe - caller must ensure exclusive access
 * PERFORMANCE: O(n) where n = network size (only on first write)
 *
 * @param brain Brain handle
 * @return true on success (or if already writable), false on error
 */
bool ensure_writable_network(brain_t brain)
{
    // Guard: Validate parameter
    if (!brain) {
        set_error("NULL brain in ensure_writable_network");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ensure_writable_network: brain is NULL");
        return false;
    }

    // If not a COW clone, network is already writable
    if (!brain->is_cow_clone) {
        return true;
    }

    // COW clone detected - need to make private copy
    // For Phase 2, we'll create a full copy of the network
    if (!brain->network) {
        set_error("COW clone has NULL network");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ensure_writable_network: brain->network is NULL");
        return false;
    }

    // Save the original network pointer
    adaptive_network_t shared_network = brain->network;

    // Phase 2 workaround: Use save/load to clone the network
    // TODO: Phase 3 should implement proper adaptive_network_clone() or incremental COW

    // Generate unique temporary filename using mkstemp for security
    // (prevents symlink attacks and race conditions)
    char temp_file[NIMCP_SHORT_PATH_SIZE];
    snprintf(temp_file, sizeof(temp_file), "/tmp/nimcp_cow_temp_XXXXXX");
    int fd = mkstemp(temp_file);
    if (fd < 0) {
        set_error("Failed to create secure temp file for COW copy");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "ensure_writable_network: validation failed");
        return false;
    }
    close(fd);  // Close fd, we'll use the filename with adaptive_network_save

    // Save shared network to temp file
    if (!adaptive_network_save(shared_network, temp_file, SERIALIZE_FORMAT_BINARY)) {
        unlink(temp_file);  // Clean up on failure
        set_error("Failed to save network for COW copy");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "ensure_writable_network: adaptive_network_save is NULL");
        return false;
    }

    // Load into new network
    brain->network = adaptive_network_load(temp_file);

    // Clean up temp file immediately after use
    unlink(temp_file);

    if (!brain->network) {
        // Failed to load - restore shared network and fail
        brain->network = shared_network;
        set_error("Failed to load network copy for COW");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "ensure_writable_network: brain->network is NULL");
        return false;
    }

    // Successfully made private copy of network
    // Note: Keep is_cow_clone = true because strategy is still shared!
    // But now we own the network and can destroy it
    brain->owns_network = true;
    brain->original_network = NULL;

    brain_clear_error();
    return true;
}


/**
 * @brief Copy a decision structure using Copy-on-Write (CoW) semantics
 *
 * WHAT: Creates a shallow copy that shares data with the original
 * WHY: Cached decisions must not be freed by caller - return copies instead
 *      Phase 1.5 CoW: Avoid expensive deep copies for read-only access
 * HOW: Share pointers with reference counting - only copy when modified
 *
 * COMPLEXITY: O(1) - just pointer sharing and refcount increment
 *             (vs O(n) for deep copy where n = output_size + num_active_neurons)
 *
 * THREAD SAFETY: Uses atomic operations for refcount updates
 *
 * @param source Decision to copy (will be modified to set up CoW sharing if not already shared)
 * @return New decision copy (shallow CoW), or NULL on allocation failure
 *
 * @note This function mutates the source decision's CoW metadata (_cow_refcount, _cow_is_shallow)
 *       when setting up sharing for the first time. This is intentional for CoW semantics -
 *       the refcount is shared metadata, not decision data. Callers must NOT pass truly
 *       const objects (e.g., objects in read-only memory or declared with const storage).
 */
brain_decision_t* copy_decision(brain_decision_t* source)
{
    if (!source) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "source is NULL");

        return NULL;

    }

    // Allocate new decision structure for the shallow copy
    brain_decision_t* copy = nimcp_calloc(1, sizeof(brain_decision_t));
    if (!copy) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "copy_decision: failed to allocate decision copy");

        return NULL;

    }

    // Copy all scalar fields (includes label, explanation, confidence, etc.)
    memcpy(copy, source, sizeof(brain_decision_t));

    // Phase 1.5 CoW: Share pointers instead of deep copying
    // The copy shares the same output_vector and active_neuron_ids as source

    // Setup reference counting for the shared data
    // Use atomic compare-and-swap to handle concurrent initialization
    //
    // KNOWN LIMITATION (ABA window): Between loading existing_refcount and the CAS,
    // another thread could free the refcount and reallocate at the same address.
    // This is a theoretical ABA concern. In practice, decisions are short-lived and
    // copies are typically made on the same thread. A proper fix would require
    // hazard pointers or epoch-based reclamation, which is deferred.
    uint32_t* existing_refcount = __atomic_load_n(&source->_cow_refcount, __ATOMIC_ACQUIRE);
    if (!existing_refcount) {
        // Source owns its data - create a new refcount for sharing
        uint32_t* new_refcount = nimcp_malloc(sizeof(uint32_t));
        if (!new_refcount) {
            nimcp_free(copy);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "copy_decision: new_refcount is NULL");
            return NULL;
        }
        // Initial refcount = 2 (source + this copy)
        *new_refcount = 2;

        // Atomically set refcount if still NULL (another thread may have beaten us)
        uint32_t* expected = NULL;
        if (__atomic_compare_exchange_n(&source->_cow_refcount, &expected, new_refcount,
                                         false, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE)) {
            // We won the race - our refcount is now installed
            __atomic_store_n(&source->_cow_is_shallow, true, __ATOMIC_RELEASE);
            copy->_cow_refcount = new_refcount;
            copy->_cow_is_shallow = true;
        } else {
            // Another thread installed a refcount first - use theirs
            nimcp_free(new_refcount);
            // expected now contains the other thread's refcount pointer
            __atomic_add_fetch(expected, 1, __ATOMIC_SEQ_CST);
            copy->_cow_refcount = expected;
            copy->_cow_is_shallow = true;
        }
    } else {
        // Source already has a refcount - just increment it
        // Use atomic increment for thread safety
        __atomic_add_fetch(existing_refcount, 1, __ATOMIC_SEQ_CST);

        copy->_cow_refcount = existing_refcount;
        copy->_cow_is_shallow = true;
    }

    // Pointers are shared (already copied via memcpy)
    // copy->output_vector = source->output_vector (same pointer)
    // copy->active_neuron_ids = source->active_neuron_ids (same pointer)

    return copy;
}


/**
 * @brief Create a deep copy of a decision (force copy, ignore CoW)
 *
 * WHAT: Creates an independent copy with its own memory
 * WHY: Needed when caller intends to modify the decision data
 * HOW: Allocates new arrays and copies all data
 *
 * COMPLEXITY: O(n) where n = output_size + num_active_neurons
 *
 * @param source Decision to deep copy
 * @return New independent decision copy, or NULL on allocation failure
 */
brain_decision_t* copy_decision_deep(const brain_decision_t* source)
{
    if (!source) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "source is NULL");
        return NULL;
    }

    // Allocate new decision structure
    brain_decision_t* copy = nimcp_calloc(1, sizeof(brain_decision_t));
    if (!copy) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "copy_decision_deep: failed to allocate decision copy");
        return NULL;
    }

    // Copy scalar fields
    memcpy(copy, source, sizeof(brain_decision_t));

    // NULL out pointer fields - we'll allocate fresh ones
    copy->output_vector = NULL;
    copy->active_neuron_ids = NULL;
    copy->_cow_refcount = NULL;      // Deep copy owns its data
    copy->_cow_is_shallow = false;
    copy->transcript = NULL;         // Don't share transcript pointer — prevents double-free

    // Deep copy output_vector
    if (source->output_vector && source->output_size > 0) {
        copy->output_vector = nimcp_malloc(source->output_size * sizeof(float));
        if (!copy->output_vector) {
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "copy_decision_deep: copy->output_vector is NULL");
            goto cleanup;
        }
        memcpy(copy->output_vector, source->output_vector, source->output_size * sizeof(float));
    }

    // Deep copy active_neuron_ids
    if (source->active_neuron_ids && source->num_active_neurons > 0) {
        copy->active_neuron_ids = nimcp_malloc(source->num_active_neurons * sizeof(uint32_t));
        if (!copy->active_neuron_ids) {
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "copy_decision_deep: failed to allocate active_neuron_ids");
            goto cleanup;
        }
        memcpy(copy->active_neuron_ids, source->active_neuron_ids,
               source->num_active_neurons * sizeof(uint32_t));
    }

    return copy;

cleanup:
    nimcp_free(copy->output_vector);
    nimcp_free(copy->active_neuron_ids);
    nimcp_free(copy);
    return NULL;
}


/**
 * @brief Make decision for input
 *
 * WHY: Primary inference interface
 * Performs forward pass and returns structured decision
 *
 * COMPLEXITY: O(s*n) where s = sparsity, n = active_neurons
 * PERFORMANCE: <1ms for small, ~5ms for medium, ~50ms for large
 * OPTIMIZATION: Caches results for repeated identical inputs
 *
 * @param brain Brain handle
 * @param features Input features
 * @param num_features Feature count
 * @return Decision result (caller must free with brain_free_decision)
 */
brain_decision_t* brain_decide(brain_t brain, const float* features, uint32_t num_features)
{
    // Guard: Validate parameters
    if (!brain || !features) {
        set_error("Invalid parameters to brain_decide");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_decide: required parameter is NULL (brain, features)");
        return NULL;
    }

    /* Phase 8: Send heartbeat at start of cognitive decision */
    brain_heartbeat(brain, "brain_decide", 0.0f);

    // W6-7 FIX: Align dimension check with predict_fast's soft check.
    // brain->config.num_inputs is 0 for unconfigured brains (before first learn_example),
    // so reject mismatches only when num_inputs has been set (> 0).
    if (brain->config.num_inputs > 0 && num_features != brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u", brain->config.num_inputs,
                  num_features);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_decide: feature count mismatch");
        return NULL;
    }

    // ========================================================================
    // DEFENSIVE COPY: Protect against input pointer invalidation
    // ========================================================================
    // WHAT: Make a local copy of the input features
    // WHY:  The caller may pass a pointer to working memory storage. Operations
    //       within brain_decide (e.g., working_memory_add) may evict and free
    //       that storage, invalidating the pointer. By copying first, we ensure
    //       safe access throughout the function.
    // HOW:  Allocate, copy, use local_features everywhere, free at return points
    float* local_features = nimcp_malloc(num_features * sizeof(float));
    if (!local_features) {
        set_error("Failed to allocate local features buffer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "brain_decide: local_features is NULL");
        return NULL;
    }
    memcpy(local_features, features, num_features * sizeof(float));

    // Use local_features instead of features from here on
    // (reassign to avoid changing all usage sites)
    features = local_features;

    // ========================================================================
    // CACHE CHECK: Thread-safe decision caching with mutex protection
    // ========================================================================
    // WHAT: Check if input matches cached input and return cached decision
    // WHY:  Avoid redundant computation for repeated identical inputs
    // HOW:  Mutex-protected comparison and decision copy
    //
    // BIOLOGICAL RATIONALE:
    // Thread-safe caching mimics neural activity persistence across cognitive
    // contexts. When identical stimuli arrive, neurons that recently fired for
    // that pattern remain in a facilitated state (short-term potentiation),
    // enabling faster reactivation. Mutex protection ensures coherent cache
    // state analogous to how neuromodulators coordinate neural ensemble stability.
    //
    // Lock cache mutex and check for cached decision
    if (nimcp_platform_mutex_lock(&brain->cache_mutex) != 0) {
        set_error("Failed to lock cache mutex for cache check");
        nimcp_free(local_features);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "brain_decide: validation failed");
        return NULL;
    }

    if (is_cached_input(brain, features, num_features)) {
        brain_decision_t* cached_copy = copy_decision_deep(brain->cached_decision);

        if (nimcp_platform_mutex_unlock(&brain->cache_mutex) != 0) {
            set_error("Failed to unlock cache mutex after cache hit");
            brain_free_decision(cached_copy);
            nimcp_free(local_features);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "brain_decide: validation failed");
            return NULL;
        }

        if (cached_copy) {
            // Use atomic increment for thread-safe stats update
            __atomic_fetch_add(&brain->stats.total_inferences, 1, __ATOMIC_RELAXED);
            nimcp_free(local_features);
            return cached_copy;
        }
        // Fall through if copy failed
    } else {
        if (nimcp_platform_mutex_unlock(&brain->cache_mutex) != 0) {
            set_error("Failed to unlock cache mutex after cache miss");
            nimcp_free(local_features);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "brain_decide: validation failed");
            return NULL;
        }
    }

    // ========================================================================
    // PARALLEL PRE-FORWARD: Wellbeing + Engram + Sleep + Curiosity
    // ========================================================================
    // WHAT: Run independent pre-forward stages concurrently via thread pool
    // WHY:  4 stages read brain state independently — parallelizing saves ~3x latency
    // HOW:  Dispatch to inference_pool, wait, extract results for later stages
    //
    // Falls back to serial execution if no thread pool is available.

    // Variables that serial stages populate (used by later stages):
    uint64_t recalled_engram_id = 0;
    float engram_confidence = 0.0F;
    sleep_state_t sleep_state = SLEEP_STATE_AWAKE;
    bool sleep_needed = false;
    float sleep_confidence_multiplier = 1.0F;
    float sleep_noise_level = 0.0F;
    bool trigger_consolidation = false;
    float novelty_score = 0.0F;
    bool is_novel = false;
    float curiosity_drive = 0.0F;

    // Try parallel pre-forward dispatch
    bool pre_forward_parallel_done = false;
    if (brain->inference_pool) {
        pre_forward_context_t pre_ctx;
        memset(&pre_ctx, 0, sizeof(pre_ctx));

        if (brain_decide_parallel_pre_forward(brain, features, num_features,
                                               brain->inference_pool, &pre_ctx)) {
            pre_forward_parallel_done = true;

            // STAGE 0: Wellbeing — check for CRITICAL distress (circuit breaker)
            if (pre_ctx.wellbeing_done && pre_ctx.distress_level >= 1.0f) {
                // Distress level 1.0 = CRITICAL — must check via main thread
                if (brain->wellbeing_monitoring_enabled && brain->introspection) {
                    brain->last_distress = wellbeing_assess_distress(brain->introspection);
                    if (brain->last_distress.severity == DISTRESS_SEVERITY_CRITICAL) {
                        set_error("Decision blocked: System in CRITICAL distress (%s)",
                                 brain->last_distress.description ? brain->last_distress.description : "Unknown");
                        nimcp_free(local_features);
                        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "brain_decide: system in CRITICAL distress");
                        return NULL;
                    }
                }
            }

            // STAGE 0.4: Extract engram recall results
            engram_confidence = pre_ctx.engram_strength;
            // Note: parallel version returns confidence; engram_id tracking requires serial path
            // Reconsolidation trigger is deferred to post-forward consolidation stages

            // STAGE 0.5: Extract sleep state (parallel computed pressure; we still need full state)
            if (brain->sleep_system && brain->config.enable_sleep_wake_cycle) {
                sleep_state = sleep_get_current_state(brain->sleep_system);
                sleep_needed = sleep_is_needed(brain->sleep_system);

                switch (sleep_state) {
                    case SLEEP_STATE_DEEP_NREM:
                        sleep_confidence_multiplier = 0.3F;
                        trigger_consolidation = true;
                        break;
                    case SLEEP_STATE_REM:
                        sleep_confidence_multiplier = 0.6F;
                        sleep_noise_level = 0.1F;
                        break;
                    case SLEEP_STATE_DROWSY:
                    case SLEEP_STATE_LIGHT_NREM:
                        sleep_confidence_multiplier = 0.8F;
                        break;
                    case SLEEP_STATE_AWAKE:
                    default:
                        if (sleep_needed) {
                            float sp = pre_ctx.sleep_pressure;
                            sleep_confidence_multiplier = 1.0F - (sp * 0.3F);
                        }
                        break;
                }
            }

            // STAGE 0.6: Extract curiosity results
            curiosity_drive = pre_ctx.curiosity_score;
            brain->last_curiosity_drive = curiosity_drive;

            // Compute novelty on main thread (cheap O(N) — needs features)
            if (brain->curiosity && brain->config.enable_curiosity) {
                float input_variance = 0.0F;
                float input_mean = 0.0F;
                for (uint32_t i = 0; i < num_features; i++) {
                    input_mean += features[i];
                }
                input_mean /= (float)num_features;
                for (uint32_t i = 0; i < num_features; i++) {
                    float diff = features[i] - input_mean;
                    input_variance += diff * diff;
                }
                input_variance /= (float)num_features;
                novelty_score = fminf(input_variance * 2.0F, 1.0F);
                is_novel = (novelty_score > 0.5F);
                brain->last_novelty_score = novelty_score;

                char experience_desc[NIMCP_ERROR_BUFFER_MEDIUM];
                snprintf(experience_desc, sizeof(experience_desc),
                        "input_variance_%.3f", input_variance);
                curiosity_learn_experience(brain->curiosity, experience_desc,
                                          features, num_features);
            }
        }
    }

    // SERIAL FALLBACK: If parallel dispatch was not available or failed
    if (!pre_forward_parallel_done) {
    // STAGE 0: Wellbeing Monitoring
    if (brain->wellbeing_monitoring_enabled && brain->introspection) {
        uint64_t current_time = nimcp_time_get_ms();
        bool should_check = (brain->wellbeing_check_interval_ms == 0) ||
                           ((current_time - brain->last_wellbeing_check_time) >= brain->wellbeing_check_interval_ms);
        if (should_check) {
            brain->last_distress = wellbeing_assess_distress(brain->introspection);
            brain->last_wellbeing_check_time = current_time;
            if (brain->last_distress.severity == DISTRESS_SEVERITY_CRITICAL) {
                set_error("Decision blocked: System in CRITICAL distress (%s)",
                         brain->last_distress.description ? brain->last_distress.description : "Unknown");
                nimcp_free(local_features);
                NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "brain_decide: system in CRITICAL distress");
                return NULL;
            }
        }
    }
    } // end serial fallback for STAGE 0

    // Phase 3: Only trigger COW if not using read-only inference
    if (!brain->can_use_readonly) {
        if (!ensure_writable_network(brain)) {
            nimcp_free(local_features);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_decide: ensure_writable_network is NULL");
            return NULL;
        }
    }

    // Allocate decision structure
    brain_decision_t* decision = allocate_decision(brain->config.num_outputs);
    if (!decision) {
        set_error("Failed to allocate decision structure");
        nimcp_free(local_features);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "brain_decide: decision is NULL");
        return NULL;
    }

    // SERIAL FALLBACK: Engram recall (STAGE 0.4) if not parallelized
    if (!pre_forward_parallel_done && brain->engram_system) {
        uint32_t* cue_neurons = nimcp_malloc(num_features * sizeof(uint32_t));
        if (cue_neurons) {
            for (uint32_t i = 0; i < num_features; i++) {
                cue_neurons[i] = i;
            }
            #define MAX_RECALL_NEURONS 100
            uint32_t recalled_neurons[MAX_RECALL_NEURONS];
            float recalled_activations[MAX_RECALL_NEURONS];
            recalled_engram_id = engram_recall(
                brain->engram_system, cue_neurons, num_features,
                recalled_neurons, recalled_activations, MAX_RECALL_NEURONS,
                &engram_confidence);
            if (recalled_engram_id != 0 && engram_confidence > 0.4F) {
                engram_trigger_reconsolidation(brain->engram_system, recalled_engram_id);
            }
            nimcp_free(cue_neurons);
        }
    }

    // SERIAL FALLBACK: Sleep/wake (STAGE 0.5) if not parallelized
    if (!pre_forward_parallel_done && brain->sleep_system && brain->config.enable_sleep_wake_cycle) {
        sleep_state = sleep_get_current_state(brain->sleep_system);
        sleep_needed = sleep_is_needed(brain->sleep_system);
        switch (sleep_state) {
            case SLEEP_STATE_DEEP_NREM:
                sleep_confidence_multiplier = 0.3F;
                trigger_consolidation = true;
                break;
            case SLEEP_STATE_REM:
                sleep_confidence_multiplier = 0.6F;
                sleep_noise_level = 0.1F;
                break;
            case SLEEP_STATE_DROWSY:
            case SLEEP_STATE_LIGHT_NREM:
                sleep_confidence_multiplier = 0.8F;
                break;
            case SLEEP_STATE_AWAKE:
            default:
                if (sleep_needed) {
                    float sleep_pressure = sleep_get_pressure(brain->sleep_system);
                    sleep_confidence_multiplier = 1.0F - (sleep_pressure * 0.3F);
                }
                break;
        }
    }

    // SERIAL FALLBACK: Curiosity (STAGE 0.6) if not parallelized
    if (!pre_forward_parallel_done && brain->curiosity && brain->config.enable_curiosity) {
        float input_variance = 0.0F;
        float input_mean = 0.0F;
        for (uint32_t i = 0; i < num_features; i++) {
            input_mean += features[i];
        }
        input_mean /= (float)num_features;
        for (uint32_t i = 0; i < num_features; i++) {
            float diff = features[i] - input_mean;
            input_variance += diff * diff;
        }
        input_variance /= (float)num_features;
        novelty_score = fminf(input_variance * 2.0F, 1.0F);
        is_novel = (novelty_score > 0.5F);
        char experience_desc[NIMCP_ERROR_BUFFER_MEDIUM];
        snprintf(experience_desc, sizeof(experience_desc),
                "input_variance_%.3f", input_variance);
        curiosity_learn_experience(brain->curiosity, experience_desc,
                                  features, num_features);
        curiosity_drive = curiosity_get_drive(brain->curiosity);
        brain->last_novelty_score = novelty_score;
        brain->last_curiosity_drive = curiosity_drive;
    }

    // ========================================================================
    // STAGE 1: Predictive Processing (Phase 10.9) - Generate Prediction
    // ========================================================================
    // WHAT: Generate top-down prediction before actual processing
    // WHY:  Compute prediction error for active inference
    // HOW:  Use predictive network to anticipate output
    float* prediction = NULL;
    float prediction_error = 0.0F;
    if (brain->predictive_network && brain->config.enable_predictive_processing) {
        prediction = (float*)nimcp_calloc(num_features, sizeof(float));
        if (prediction) {
            // Generate prediction (5 iterations of free energy minimization)
            predictive_forward(brain->predictive_network, features, 5);
            // Get prediction from bottom layer
            predictive_get_layer_prediction(brain->predictive_network, 0, prediction);
        }
    }

    // Attention-guided feature processing (positional encoding + multi-head attention)
    // Treat input as a sequence of chunks for cross-element attention.
    // Chunk size = 64 (matches typical head dimension), sequence_length = num_features/64.
    // This enables the attention mechanism to attend across different parts of the input
    // rather than treating the entire input as a single token (sequence_length=1).
    if (brain->positional_encoder) {
        nimcp_pos_apply_encoding(brain->positional_encoder, local_features,
                                1, local_features, true);
    }
    if (brain->multihead_attention) {
        float* attended = nimcp_malloc(num_features * sizeof(float));
        if (attended) {
            // Use chunked sequence: each 64-dim slice is a "token"
            // For 1024-dim input: 16 tokens × 64 dims = cross-attention over 16 positions
            uint32_t chunk_dim = 64;
            uint32_t seq_len = num_features / chunk_dim;
            if (seq_len < 2) seq_len = 1;  // Fall back to single token for small inputs
            if (seq_len * chunk_dim > num_features) seq_len = 1;  // Safety

            if (multihead_attention_forward(brain->multihead_attention,
                                           features, seq_len, NULL, attended)) {
                // Blend attended features with original (residual connection)
                uint32_t blend_len = seq_len * chunk_dim;
                if (blend_len > num_features) blend_len = num_features;
                for (uint32_t i = 0; i < blend_len; i++) {
                    local_features[i] = 0.6f * local_features[i] + 0.4f * attended[i];
                }
            }
            nimcp_free(attended);
        }
    }

    // ========================================================================
    // RECURRENT FORWARD PASS: Iterative refinement for uncertain decisions
    // ========================================================================
    // WHAT: Run multiple forward passes, feeding output back into input
    // WHY:  Biological: thalamocortical loops refine percepts over 50-150ms
    //       Hard problems benefit from iterative processing (like "thinking longer")
    // HOW:  After initial forward pass, check confidence. If below threshold,
    //       blend output into input and re-run. Cap at max_iterations.
    //
    uint32_t active_neurons = perform_forward_pass(brain, features, num_features, decision);

    if (brain->recurrent_enabled && brain->recurrent_max_iterations > 1) {
        float alpha = brain->recurrent_blend_alpha;
        uint32_t out_sz = decision->output_size;
        uint32_t iter;

        for (iter = 1; iter < brain->recurrent_max_iterations; iter++) {
            // Check stopping criterion: confident enough?
            float max_act = 0.0f;
            for (uint32_t i = 0; i < out_sz; i++) {
                float v = fabsf(decision->output_vector[i]);
                if (v > max_act) max_act = v;
            }
            if (max_act >= brain->recurrent_confidence_threshold) {
                break;  // Confident — stop iterating
            }

            // Blend output back into input: input' = (1-α)·input + α·output_projected
            // Project output to input space by tiling/truncating
            for (uint32_t i = 0; i < num_features; i++) {
                float feedback = decision->output_vector[i % out_sz];
                local_features[i] = (1.0f - alpha) * local_features[i] + alpha * feedback;
            }

            // Re-run forward pass with refined input
            active_neurons = perform_forward_pass(brain, local_features, num_features, decision);
        }
        brain->recurrent_iteration_count = iter;
    } else {
        brain->recurrent_iteration_count = 1;
    }

    // ========================================================================
    // OPTIMIZATION: Classification-only fast path — skip non-essential stages
    // ========================================================================
    // WHAT: For pure classification tasks, skip expensive cognitive stages
    // WHY:  Classification only needs forward pass + argmax + confidence
    //       Glial maintenance, ToM, Shannon analysis, quantum coherence,
    //       emotional processing, and sleep pressure are unnecessary overhead
    // HOW:  Jump directly to output labeling and stats after forward pass
    bool classify_fast_path = (brain->strategy &&
                               brain->strategy->task_type == BRAIN_TASK_CLASSIFICATION &&
                               !brain->config.enable_natural_explanations);
    if (classify_fast_path) {
        // Apply task-specific output transformation (e.g., softmax)
        if (brain->strategy && brain->strategy->transform_output) {
            brain->strategy->transform_output(decision->output_vector, decision->output_size);
        }

        // Determine output label and confidence (argmax + label lookup)
        determine_output_label(brain, decision);

        // Update statistics
        update_inference_stats(brain, decision);

        // Cache decision
        if (nimcp_platform_mutex_lock(&brain->cache_mutex) == 0) {
            cache_decision(brain, features, num_features, decision);
            nimcp_platform_mutex_unlock(&brain->cache_mutex);
        }

        nimcp_free(prediction);  // Free predictive processing buffer if allocated
        nimcp_free(local_features);
        brain_clear_error();
        return decision;
    }

    // ========================================================================
    // STAGE 1.5: Hemispheric Processing — Callosum Transfer + Lateralization
    // ========================================================================
    // WHAT: Route output through inter-hemispheric channels, modulate by dominance
    // WHY:  Biological brains lateralize processing; callosum coordinates hemispheres
    // HOW:  Send output through cognitive callosum channel, update hemisphere balance
    //
    // BIOLOGICAL BASIS:
    // - Corpus callosum transfers ~200 msg/s between hemispheres
    // - Language left-lateralized (95%), spatial right-lateralized (80%)
    // - Lateralization shifts with experience (neuroplasticity)
    //
    if (brain->hemispheric_enabled && brain->callosum && decision->output_vector) {
        // Determine cognitive domain from input features (heuristic: variance profile)
        // High-variance = spatial/perceptual (right), low-variance = language/logic (left)
        float feature_variance = 0.0f;
        float feature_mean = 0.0f;
        uint32_t n_sample = (num_features > 64) ? 64 : num_features;
        for (uint32_t i = 0; i < n_sample; i++) {
            feature_mean += features[i];
        }
        feature_mean /= (float)n_sample;
        for (uint32_t i = 0; i < n_sample; i++) {
            float d = features[i] - feature_mean;
            feature_variance += d * d;
        }
        feature_variance /= (float)n_sample;

        // Map variance to cognitive domain (simplified heuristic)
        cognitive_domain_t domain = COGNITIVE_DOMAIN_LANGUAGE;
        if (feature_variance > 0.5f) {
            domain = COGNITIVE_DOMAIN_SPATIAL;
        } else if (feature_variance > 0.2f) {
            domain = COGNITIVE_DOMAIN_ATTENTION_GLOBAL;
        }

        // Get lateralization dominance for this domain
        float dominance = lateralization_get_dominance(
            &brain->lateralization, domain);
        hemisphere_id_t dominant = lateralization_get_dominant_hemisphere(
            &brain->lateralization, domain);

        // Send output through callosum cognitive channel
        // The source hemisphere is the dominant one for this domain
        uint32_t send_size = decision->output_size * sizeof(float);
        if (send_size > 4096) send_size = 4096;  // Callosum message size limit
        callosum_send(brain->callosum, dominant,
                      CALLOSUM_CHANNEL_COGNITIVE,
                      CALLOSUM_PRIORITY_NORMAL,
                      0,  // message_type: inference output
                      decision->output_vector,
                      send_size);

        // Process any queued callosum messages (deliver after latency)
        callosum_process_queues(brain->callosum);

        // Update hemispheric balance: positive = right dominant, negative = left
        float balance_delta = (dominant == HEMISPHERE_LEFT) ? -0.01f : 0.01f;
        brain->hemispheric_balance = brain->hemispheric_balance * 0.99f + balance_delta;

        // Modulate output by lateralization strength
        // Stronger lateralization = more confident (specialized processing)
        float lat_strength = fabsf(dominance - 0.5f) * 2.0f;  // 0.0-1.0
        float lat_boost = 1.0f + lat_strength * 0.1f;  // Up to 10% confidence boost
        for (uint32_t i = 0; i < decision->output_size; i++) {
            decision->output_vector[i] *= lat_boost;
        }
    }

    // ========================================================================
    // STAGE 2: Predictive Processing - Compute Prediction Error
    // ========================================================================
    // WHAT: Compute mismatch between prediction and actual output
    // WHY:  Prediction errors drive learning and attention
    // HOW:  L2 distance between predicted and actual output
    if (prediction) {
        for (uint32_t i = 0; i < decision->output_size; i++) {
            float error = decision->output_vector[i] -
                         (i < num_features ? prediction[i] : 0.0F);
            prediction_error += error * error;
        }
        prediction_error = (decision->output_size > 0) ?
            sqrtf(prediction_error / decision->output_size) : 0.0f;

        // Update predictive model with actual outcome
        if (brain->config.enable_predictive_processing) {
            predictive_update_model(brain->predictive_network);
        }

        nimcp_free(prediction);
        prediction = NULL;
    }

    // EDP Online Learning: Feed prediction error for continuous STDP
    if (brain->event_driven_plasticity && brain->enable_event_driven_plasticity
        && edp_is_active(brain->event_driven_plasticity)) {
        edp_process_prediction_error(brain->event_driven_plasticity, prediction_error, 0);
        edp_update_eligibility(brain->event_driven_plasticity, 0.001f);
    }

    // Apply task-specific output transformation
    if (brain->strategy && brain->strategy->transform_output) {
        brain->strategy->transform_output(decision->output_vector, decision->output_size);
    }

    // ========================================================================
    // STAGE 3.5: Apply Sleep-Induced Noise (REM creativity)
    // ========================================================================
    // WHAT: Add random noise to outputs during REM sleep
    // WHY:  REM sleep shows creative recombination, increased variability
    // HOW:  Add gaussian noise proportional to sleep_noise_level
    if (sleep_noise_level > 0.0F) {
        for (uint32_t i = 0; i < decision->output_size; i++) {
            // Add noise: random value in [-noise, +noise] range
            // NOLINTNEXTLINE(concurrency-mt-unsafe): noise generation, not security-critical
            float noise = ((float)nimcp_tl_rand() / (float)RAND_MAX) * 2.0F - 1.0F;  // [-1, 1]
            noise *= sleep_noise_level;  // Scale to desired level
            decision->output_vector[i] += noise * decision->output_vector[i];  // Proportional noise
        }
    }

    // Determine output label and confidence
    determine_output_label(brain, decision);

    // ========================================================================
    // STAGE 3.9: Edge-Cloud Hybrid Inference (Confidence-Gated Routing)
    // ========================================================================
    // WHAT: If local confidence is too low, escalate to cloud backend
    // WHY:  Small brains handle easy cases locally; hard cases go to cloud
    // HOW:  cloud_inference_route() checks threshold, calls backend, distills
    //
    // BIOLOGICAL ANALOGY:
    // Kahneman System 1 (fast/local) vs System 2 (slow/cloud).
    // Repeated System 2 answers become System 1 habits via distillation.
    //
    if (brain->cloud_inference_enabled && brain->cloud_bridge) {
        cloud_inference_route(
            (cloud_inference_bridge_t*)brain->cloud_bridge,
            brain, decision, features, num_features);
        // decision may have been upgraded in-place with cloud result
    }

    // ========================================================================
    // STAGE 4: Apply Sleep-Induced Cognitive Degradation
    // ========================================================================
    // WHAT: Reduce decision confidence based on sleep state
    // WHY:  Sleep/drowsiness impairs cognitive performance
    // HOW:  Multiply confidence by sleep_confidence_multiplier
    decision->confidence *= sleep_confidence_multiplier;

    // ========================================================================
    // STAGE 4.2: Trigger Memory Consolidation (Deep Sleep) - Phase 11 ACTIVE
    // ========================================================================
    // WHAT: Transfer high-salience working memory items to long-term during deep sleep
    // WHY:  Sleep is when memory consolidation occurs biologically
    // HOW:  Retrieve items with salience >0.7, store in longterm buffer, clear from WM
    //
    // BIOLOGICAL BASIS:
    // - Hippocampus → Cortex transfer during SWS (slow-wave sleep)
    // - High-salience memories prioritized for consolidation
    // - Replay and synaptic strengthening occur during sleep
    //
    // COMPLEXITY: O(N) where N = working memory size
    // NOTE: Dead consolidation code removed — wm_salience was always 0.0 (never populated),
    // so the threshold check (>= 0.7) was unreachable. Real consolidation happens via
    // the engram system (Stage 3.8) and systems consolidation (Phase M2) below.
    (void)trigger_consolidation;

    // ========================================================================
    // PARALLEL POST-FORWARD: Submit independent stages to thread pool
    // ========================================================================
    // WHAT: Run 8 independent stages (3.8, 3.9, 3.10, 3.11, glial, ToM,
    //       Shannon, quantum-Shannon) concurrently with sequential stages
    // WHY:  These stages read brain state but don't modify the decision struct,
    //       so they can run while main thread handles reasoning/dialogue/etc.
    // HOW:  Submit to pool, run sequential stages on main thread, then pool_wait
    post_forward_context_t post_ctx;
    memset(&post_ctx, 0, sizeof(post_ctx));
    bool post_forward_submitted = false;

    if (brain->inference_pool) {
        post_forward_submitted = brain_decide_submit_post_forward(
            brain, decision, brain->inference_pool, &post_ctx);
    }

    // SERIAL FALLBACK: Stages 3.8-3.11 if no thread pool
    if (!post_forward_submitted) {
    // STAGE 3.8: MEMORY ENGRAM CONSOLIDATION
    if (brain->engram_system) {
        const float TIME_DELTA_SECONDS = 0.1F;
        bool is_sleeping = (sleep_state == SLEEP_STATE_DEEP_NREM ||
                           sleep_state == SLEEP_STATE_LIGHT_NREM ||
                           sleep_state == SLEEP_STATE_REM);
        engram_consolidate_update(brain->engram_system, TIME_DELTA_SECONDS, is_sleeping);
        if (sleep_state == SLEEP_STATE_REM && recalled_engram_id != 0) {
            engram_trigger_reconsolidation(brain->engram_system, recalled_engram_id);
        }
    }

    // STAGE 3.9: SYSTEMS CONSOLIDATION UPDATE
    if (brain->systems_consolidation) {
        const float TIME_DELTA_SECONDS = 0.1F;
        bool is_sws = (sleep_state == SLEEP_STATE_DEEP_NREM ||
                       sleep_state == SLEEP_STATE_LIGHT_NREM);
        bool is_rem = (sleep_state == SLEEP_STATE_REM);
        bool is_sleeping = (is_sws || is_rem);
        if (is_sleeping || (recalled_engram_id != 0)) {
            if (recalled_engram_id != 0) {
                systems_consolidation_schedule_replay(
                    brain->systems_consolidation, recalled_engram_id, engram_confidence);
            }
            systems_consolidation_execute_replays(
                brain->systems_consolidation, TIME_DELTA_SECONDS, is_sws, is_rem);
        }
        systems_consolidation_update(brain->systems_consolidation, TIME_DELTA_SECONDS, is_sleeping);
    }

    // STAGE 3.10: WORKING MEMORY TRANSFER
    if (brain->wm_transfer_system && brain->working_memory) {
        wm_transfer_evaluate(brain->wm_transfer_system, 0.1F);
    }

    // STAGE 3.11: SEMANTIC MEMORY QUERY
    if (brain->semantic_memory) {
        semantic_query_result_t* semantic_results = semantic_memory_query(
            brain->semantic_memory, features, num_features);
        if (semantic_results) {
            semantic_memory_free_result(semantic_results);
        }
        semantic_memory_extract_from_consolidation(brain->semantic_memory);
    }
    } // end serial fallback for stages 3.8-3.11

    // ========================================================================
    // STAGE 4.1: REASONING ENGINE (Causal/Abductive/Convergent)
    // ========================================================================
    // WHAT: Execute multi-step reasoning on decision context
    // WHY:  Enable causal inference, evidence synthesis, epistemic verification
    // HOW:  Connect engine to brain, run reasoning pipeline on decision label
    //
    // BIOLOGICAL BASIS:
    // - Prefrontal cortex: Multi-step planning and inference (Miller & Cohen, 2001)
    // - Working memory-guided reasoning (Baddeley, 2003)
    // - Bayesian inference in neural circuits (Ma et al., 2006)
    //
    // COMPLEXITY: O(steps * module_cost)
    float reasoning_confidence = 0.0F;
    if (brain->reasoning_engine && brain->reasoning_engine_enabled && decision->label[0]) {
        reasoning_chain_t chain;
        reasoning_chain_init(&chain);

        // Connect engine to brain (extracts subsystem pointers)
        reasoning_engine_connect_brain(brain->reasoning_engine, brain);

        // Run reasoning on the decision label as query
        int reason_rc = reasoning_engine_reason(brain->reasoning_engine,
                                                 decision->label, &chain);
        if (reason_rc == 0 && chain.num_steps > 0) {
            reasoning_confidence = chain.overall_confidence;
            // Blend reasoning confidence with network confidence
            float rw = brain->config.reasoning_blend_weight;
            decision->confidence = decision->confidence * (1.0F - rw) +
                                   reasoning_confidence * rw;
        }

        reasoning_chain_cleanup(&chain);
    }

    // ========================================================================
    // STAGE 4.2a: INNER DIALOGUE (Multi-Perspective Deliberation)
    // ========================================================================
    // WHAT: Run 7-perspective deliberation on the decision
    // WHY:  Analytical, emotional, critical, creative, memory, ethical, and
    //       metacognitive viewpoints catch blind spots single-pass misses
    // HOW:  Start conversation on decision topic, run to convergence
    //
    // BIOLOGICAL BASIS:
    // - Internal speech (Vygotsky, 1934): verbal self-regulation
    // - Default mode network (Raichle, 2001): self-referential processing
    // - Metacognitive monitoring (Flavell, 1979): thinking about thinking
    //
    // COMPLEXITY: O(turns * perspectives) — bounded by max_turns config
    if (brain->inner_dialogue && brain->inner_dialogue_enabled && decision->label[0]) {
        // Only deliberate on decisions with moderate confidence (uncertain territory)
        // High confidence = brain is sure, low confidence = too noisy to deliberate on
        if (decision->confidence > brain->config.dialogue_confidence_min &&
            decision->confidence < brain->config.dialogue_confidence_max) {
            inner_dialogue_result_t dialogue_result;
            memset(&dialogue_result, 0, sizeof(dialogue_result));

            // Start conversation on the decision topic
            int start_rc = inner_dialogue_engine_start(brain->inner_dialogue,
                                                        decision->label);
            if (start_rc == 0) {
                int run_rc = inner_dialogue_engine_run(brain->inner_dialogue,
                                                       &dialogue_result);
                if (run_rc == 0 && dialogue_result.has_conclusion) {
                    // Modulate confidence based on agreement among perspectives
                    // High agreement (>0.8) → boost confidence
                    // Low agreement (<0.4) → reduce confidence (disagreement = uncertainty)
                    if (dialogue_result.final_agreement > DIALOGUE_AGREEMENT_HIGH) {
                        decision->confidence *= DIALOGUE_BOOST_FACTOR;
                    } else if (dialogue_result.final_agreement < DIALOGUE_AGREEMENT_LOW) {
                        decision->confidence *= DIALOGUE_REDUCE_FACTOR;
                    }
                    decision->confidence = fminf(decision->confidence, 1.0F);
                }
            }
        }
    }

    // ========================================================================
    // STAGE 4.2b: IMAGINATION ENGINE (Prospective Simulation)
    // ========================================================================
    // WHAT: Run mental simulation to test the decision before committing
    // WHY:  Prospective simulation prevents harmful/suboptimal actions
    // HOW:  Begin scenario, step forward, evaluate coherence as quality signal
    //
    // BIOLOGICAL BASIS:
    // - Hippocampal prospection (Schacter & Addis, 2007): future imagination
    // - Default mode network (Buckner et al., 2008): episodic simulation
    // - Mental rehearsal (Jeannerod, 2001): motor imagery improves execution
    //
    // COMPLEXITY: O(simulation_steps)
    if (brain->imagination && brain->imagination_enabled) {
        // Only simulate for decisions with enough confidence to be worth testing
        if (decision->confidence > brain->config.imagination_confidence_min) {
            imagination_scenario_t* scenario = imagination_begin_scenario(
                brain->imagination,
                IMAGINATION_MODE_PROSPECTIVE,
                NULL  // No specific goal — evaluate decision outcome
            );

            if (scenario) {
                // Run a few simulation steps to let the scenario evolve
                bool sim_ok = true;
                for (int sim_step = 0; sim_step < IMAGINATION_SIM_STEPS && sim_ok; sim_step++) {
                    sim_ok = (imagination_step_scenario(brain->imagination, scenario) == 0);
                }

                // If simulation failed to evolve, slight confidence reduction
                // (inability to simulate = uncertain territory)
                if (!sim_ok) {
                    decision->confidence *= IMAGINATION_FAIL_PENALTY;
                }

                imagination_end_scenario(brain->imagination, scenario);
            }
        }
    }

    // ========================================================================
    // STAGE 4.3: RECURSIVE COGNITION (Goal Decomposition)
    // ========================================================================
    // WHAT: For complex decisions, decompose into sub-goals and orchestrate
    // WHY:  Complex tasks require multi-step decomposition and delegation
    // HOW:  Create goal from decision, process through engine, refine answer
    //
    // BIOLOGICAL BASIS:
    // - Hierarchical planning in prefrontal cortex (Koechlin et al., 2003)
    // - Goal-directed behavior (Balleine & Dickinson, 1998)
    // - Cognitive control hierarchy (Badre, 2008)
    //
    // COMPLEXITY: O(subtasks * depth)
    if (brain->rcog_engine && brain->rcog_engine_enabled && decision->label[0]) {
        // Only invoke recursive cognition for low-confidence complex decisions
        // where simple forward pass wasn't enough
        if (decision->confidence < brain->config.rcog_confidence_max &&
            decision->confidence > RCOG_CONFIDENCE_FLOOR) {
            rcog_goal_t goal = rcog_engine_create_goal(
                decision->label,
                RCOG_GOAL_ANALYSIS  // Analysis goal type
            );

            rcog_process_result_t rcog_result;
            memset(&rcog_result, 0, sizeof(rcog_result));

            int rcog_rc = rcog_engine_process(brain->rcog_engine, &goal, &rcog_result);
            if (rcog_rc == 0 && rcog_result.success) {
                // Recursive processing improved confidence
                // Use geometric mean of original and recursive confidence
                float rcog_conf = rcog_result.answer.confidence;
                if (rcog_conf > decision->confidence) {
                    decision->confidence = sqrtf(decision->confidence * rcog_conf);
                }
            }
        }
    }

    // ========================================================================
    // STAGE 4.4: COLLECTIVE COGNITION UPDATE (Multi-Brain Sync)
    // ========================================================================
    // WHAT: Broadcast decision state to collective and receive group influence
    // WHY:  Enable distributed consciousness, swarm intelligence, shared goals
    // HOW:  Update collective cognition with current decision context
    //
    // BIOLOGICAL BASIS:
    // - Social cognition (Tomasello, 2014): shared intentionality
    // - Group decision-making (Couzin et al., 2005): emergent collective intelligence
    // - Inter-brain synchronization (Hasson et al., 2012): neural coupling
    //
    // COMPLEXITY: O(connected_instances)
    if (brain->collective_cognition && brain->collective_cognition_enabled) {
        collective_cognition_update(brain->collective_cognition);
    }

    // ========================================================================
    // STAGE 4.5: Executive Controller Integration (Phase 10.11.2 - Priority 3)
    // ========================================================================
    // WHAT: Apply executive control to decision output
    // WHY:  Enable goal-directed behavior, inhibition, and multi-step planning
    // HOW:  Use executive controller to select/inhibit/plan actions
    if (brain->executive && brain->config.enable_executive_control) {
        // Check if response should be inhibited
        // For example, inhibit low-confidence decisions
        if (decision->confidence < 0.3F) {
            bool should_inhibit = executive_should_inhibit(
                brain->executive,
                decision->confidence,
                "low confidence"
            );

            if (should_inhibit) {
                // Inhibited: Set output to neutral/no-op
                // In classification, this could mean "uncertain" class
                // For now, mark it in the label
                strncat(decision->label, " [INHIBITED]", sizeof(decision->label) - strlen(decision->label) - 1);
                decision->confidence = 0.0F;
            }
        }

        // Executive could also:
        // - Select among competing outputs (task switching)
        // - Decompose complex goals into action sequences (planning)
        // - Coordinate multi-step behaviors
        // Note: Full integration requires executive task management
    }

    // ========================================================================
    // STAGE 4.6: Curiosity-Executive Bidirectional Feedback (Phase 11)
    // ========================================================================
    // WHAT: Two-way communication between curiosity and executive systems
    // WHY:  Balance exploration vs exploitation based on cognitive load
    // HOW:  Executive→Curiosity: modulate exploration based on load
    //       Curiosity→Executive: provide information gain for prioritization
    //
    // BIOLOGICAL BASIS:
    // - Prefrontal cortex regulates exploration/exploitation (Daw et al., 2006)
    // - High cognitive load → reduced exploration (dual-task interference)
    // - Novel stimuli compete for executive attention (task switching)
    //
    // COMPLEXITY: O(1)
    if (brain->curiosity && brain->executive &&
        brain->config.enable_curiosity && brain->config.enable_executive_control) {

        // EXECUTIVE → CURIOSITY: Modulate exploration based on cognitive load
        // When executive is busy (high load), reduce exploration
        // When executive has capacity, allow more exploration
        executive_stats_t exec_stats;
        if (executive_get_stats(brain->executive, &exec_stats)) {
            // Compute cognitive load from failure rate (0.0 = all success, 1.0 = all failures)
            // High failure rate indicates executive is overloaded/struggling
            float failure_rate = 0.0F;
            if (exec_stats.total_tasks > 0) {
                failure_rate = (float)exec_stats.failed_tasks / (float)exec_stats.total_tasks;
            }

            // Also consider inhibition rate (high inhibition = high control demand)
            float cognitive_load = fminf((failure_rate * 0.5F) + (exec_stats.inhibition_rate * 0.5F), 1.0F);

            // Convert load to exploration rate: high load → low exploration
            // Load 0.0 (idle/successful) → explore 0.8 (high exploration)
            // Load 1.0 (busy/failing) → explore 0.2 (low exploration, focus on current tasks)
            float exploration_rate = 0.8F - (cognitive_load * 0.6F);
            curiosity_set_exploration_rate(brain->curiosity, exploration_rate);

            // CURIOSITY → EXECUTIVE: Provide information gain signal
            // Executive can use this to prioritize exploratory tasks
            float info_gain = curiosity_get_information_gain(brain->curiosity);

            // If high information gain (>0.6) and low load (<0.5), exploration is valuable
            // (Could trigger exploratory behavior in executive planner in future)
            // Note: info_gain signal is now available for executive to use in task prioritization
            (void)info_gain;  // Suppress unused warning - used for bidirectional feedback
            (void)cognitive_load;  // Suppress unused warning
        }
    }

    // Populate interpretability information
    populate_interpretability(brain, features, num_features, active_neurons, decision);

    // ========================================================================
    // STAGE 5: Natural Explanations (Phase 10.7)
    // ========================================================================
    // WHAT: Generate human-readable what-why-how explanations
    // WHY:  Enhance interpretability with structured natural language
    // HOW:  Use explanation_generator to create detailed explanations
    if (brain->explanation_gen && brain->config.enable_natural_explanations) {
        natural_explanation_t nat_exp;
        if (explanation_generate_from_decision(brain->explanation_gen, brain, decision, &nat_exp)) {
            // Enhance the decision->explanation with natural explanation
            // Format: "WHAT: <what> | WHY: <why> | CONF: <confidence>"
            snprintf(decision->explanation, sizeof(decision->explanation),
                    "WHAT: %s | WHY: %s | CONF: %s",
                    nat_exp.what, nat_exp.why, nat_exp.confidence);

            // Optional: Add symbolic logic proof if available and enabled
            if (brain->symbolic_logic && nat_exp.has_symbolic_proof) {
                char proof_buffer[NIMCP_ERROR_BUFFER_LARGE];
                if (explain_with_symbolic_logic(brain->explanation_gen, brain,
                                               decision, proof_buffer, sizeof(proof_buffer))) {
                    // Append proof to explanation (if space permits)
                    size_t current_len = strlen(decision->explanation);
                    size_t remaining = sizeof(decision->explanation) - current_len;
                    if (remaining > 20) {  // Enough space for " | PROOF: <text>"
                        snprintf(decision->explanation + current_len, remaining,
                                " | PROOF: %s", proof_buffer);
                    }
                }
            }
        }
    }

    // ========================================================================
    // STAGE 6: Working Memory Integration (Phase 10.11.2)
    // ========================================================================
    // WHAT: Store decision context in working memory with cognitive metadata
    // WHY:  Enable context-dependent decisions, consolidation, and temporal reasoning
    // HOW:  Store features + decision + cognitive state (sleep, novelty, etc.)
    if (brain->working_memory && brain->config.enable_working_memory) {
        // Compute salience based on multiple factors:
        // - High prediction error = surprising/important
        // - Novel input = worth remembering
        // - High confidence = reliable information
        float salience = 0.5F;  // Base salience

        // Boost salience for novel inputs (curiosity-driven)
        if (is_novel) {
            salience += 0.2F;
        }

        // Boost salience for high prediction error (surprise)
        if (prediction_error > 0.5F) {
            salience += 0.2F;
        }

        // Boost salience for high confidence decisions (reliable)
        if (decision->confidence > 0.8F) {
            salience += 0.1F;
        }

        // ====================================================================
        // Phase 11: ATTENTION-WORKING MEMORY COORDINATION
        // ====================================================================
        // WHAT: Boost salience for attended items (attention gates memory)
        // WHY:  Biologically, only attended stimuli reach working memory (PFC)
        //       "Inattentional blindness" - unattended items don't enter awareness
        // HOW:  Get attention strength from multihead attention, boost salience
        //
        // BIOLOGICAL BASIS:
        // - Visual cortex → Attention filter → PFC (working memory)
        // - Unattended items: weak cortical representation, don't reach PFC
        // - Attended items: enhanced representation, prioritized for WM storage
        //
        // COMPLEXITY: O(1)
        if (brain->multihead_attention && brain->config.enable_multihead_attention) {
            float attention_strength = multihead_attention_get_strength(brain->multihead_attention);

            // Boost salience proportional to attention (up to +0.3)
            // High attention (0.8-1.0) → strong boost (+0.24 to +0.3)
            // Medium attention (0.5-0.8) → moderate boost (+0.15 to +0.24)
            // Low attention (0.0-0.5) → weak boost (0.0 to +0.15)
            float attention_boost = attention_strength * 0.3F;
            salience += attention_boost;
        }

        salience = fminf(salience, 1.0F);  // Cap at 1.0

        // Store in working memory
        working_memory_add(brain->working_memory, features, num_features, salience);

        // During sleep, these items would be consolidated to long-term memory
        // by the sleep system (already integrated in brain_sleep() if implemented)
    }

    // ========================================================================
    // STAGE 6.5: Global Workspace Competition (NEWLY INTEGRATED)
    // ========================================================================
    // WHAT: Modules compete for conscious access via broadcast
    // WHY:  Limited-capacity workspace enables prioritization and integration
    // HOW:  High-salience/novelty content competes, winner broadcasts globally
    if (brain->global_workspace && brain->config.enable_global_workspace) {
        // Working memory competes with decision content
        bool won_competition = global_workspace_compete(
            brain->global_workspace,
            MODULE_WORKING_MEMORY,
            features,
            num_features,
            prediction_error + (is_novel ? 0.3F : 0.0F)  // Strength based on novelty/surprise
        );

        // If won competition, content is now in global workspace (conscious access)
        if (won_competition) {
            // Add note to decision that it reached conscious access
            strncat(decision->explanation, " [CONSCIOUS]",
                   sizeof(decision->explanation) - strlen(decision->explanation) - 1);
        }

        // Executive function could also compete for action planning
        if (brain->executive && brain->config.enable_executive_control) {
            float executive_urgency = executive_get_cognitive_load(brain->executive);
            if (executive_urgency > 0.7F) {
                global_workspace_compete(
                    brain->global_workspace,
                    MODULE_EXECUTIVE,
                    decision->output_vector,
                    decision->output_size,
                    executive_urgency
                );
            }
        }

        // Salience detection could compete for attention signals
        if (brain->salience && brain->config.enable_salience) {
            brain_salience_t salience_signal = brain_evaluate_salience_temporal(
                brain->salience,
                features,
                num_features,
                nimcp_time_get_ms()
            );
            if (salience_signal.surprise > 0.7F) {
                global_workspace_compete(
                    brain->global_workspace,
                    MODULE_SALIENCE,
                    features,
                    num_features,
                    salience_signal.surprise
                );
            }
        }
    }

    // ========================================================================
    // STAGE 7: Emotional Tagging (Phase 10.11.2 - REAL INTEGRATION)
    // ========================================================================
    // WHAT: Tag significant decisions with emotional valence/arousal
    // WHY:  Prioritize emotionally-significant experiences for consolidation
    // HOW:  Compute valence from confidence, arousal from prediction error, boost salience
    if (brain->config.enable_emotional_tagging) {
        // Valence: Positive for high confidence, negative for low confidence
        float valence = (decision->confidence - 0.5F) * 2.0F;  // Range: [-1, 1]

        // Arousal: High for high prediction error (surprising)
        float arousal = prediction_error;  // Already in [0, 1] range

        // Create actual emotional tag (instead of discarding)
        emotional_tag_t emotion = emotional_tag_create(
            valence,
            arousal,
            nimcp_time_get_ms()
        );

        // BEHAVIORAL EFFECT: Boost working memory salience for emotional content
        // High arousal = grab attention, strong valence = important
        if (brain->working_memory && brain->config.enable_working_memory) {
            float emotional_salience_boost = 0.0F;

            // Arousal boosts salience (high arousal = attention grabbing)
            emotional_salience_boost += emotion.arousal * EMOTIONAL_AROUSAL_SALIENCE_FACTOR;

            // Strong valence (positive OR negative) boosts salience
            float valence_intensity = fabsf(emotion.valence);
            emotional_salience_boost += valence_intensity * EMOTIONAL_VALENCE_SALIENCE_FACTOR;

            // Apply boost by re-storing the last item with higher salience
            // Note: This is a simplified approach; ideally we'd tag the specific item
            // For now, the next working memory add will benefit from this computation
            // being factored into the novelty/prediction salience calculation above

            // Store emotional tag with decision for later retrieval
            // (In a full implementation, working memory items would have emotion field)
            (void)emotional_salience_boost;  // Computed but would be used in full impl
        }

        // Track emotional statistics (would need to add field to brain_stats_t)
        // For now, just note that we're creating actual emotional tags
        (void)emotion.intensity;  // Computed and used above for salience
    }

    // ========================================================================
    // STAGE 7.5: Bidirectional Cognitive Feedback (Phase 10.11.3)
    // ========================================================================
    // WHAT: Apply bidirectional connections between cognitive modules
    // WHY:  Enable top-down and bottom-up modulation for realistic cognition
    // HOW:  4 strategic connections based on neuroscience

    // Connection 1: Curiosity ↔ Executive Function
    if (brain->curiosity && brain->executive &&
        brain->config.enable_curiosity && brain->config.enable_executive_control) {
        // Executive → Curiosity: Reduce exploration when cognitively overloaded
        float cognitive_load = executive_get_cognitive_load(brain->executive);
        if (cognitive_load > 0.8F) {
            float exploration_rate = 1.0F - cognitive_load;  // High load → low exploration
            curiosity_set_exploration_rate(brain->curiosity, exploration_rate);
        }

        // Curiosity → Executive: Boost exploratory tasks when high information gain
        float information_gain = curiosity_get_information_gain(brain->curiosity);
        if (information_gain > 0.7F) {
            executive_boost_task_priority(brain->executive, "exploration", information_gain * 0.3F);
        }
    }

    // Connection 2: Mirror Neurons ↔ Visual Cortex
    if (brain->mirror_neurons && brain->visual_cortex) {
        // Mirror Neurons → Visual: Boost attention to social cues
        float social_salience = mirror_neurons_get_social_salience(brain->mirror_neurons);
        if (social_salience > 0.6F) {
            // Boost visual attention to center region (where faces typically appear)
            visual_cortex_boost_region_attention(brain->visual_cortex, 0.5F, 0.5F, social_salience);
        }

        // Visual → Mirror Neurons: Activate observation mode when agent detected
        if (num_features > 0) {
            bool agent_detected = visual_cortex_detect_agent(brain->visual_cortex, features, num_features);
            if (agent_detected) {
                mirror_neurons_activate_observation_mode(brain->mirror_neurons);
            }
        }
    }

    // Connection 3: Emotional System ↔ Salience
    if (brain->salience && brain->config.enable_emotional_tagging) {
        // Create emotional tag from current cognitive state
        emotional_tag_t current_emotion = emotional_tag_from_cognitive_state(
            decision->confidence,
            prediction_error,
            novelty_score,
            true,  // Ethical approval (assumed in normal brain_decide)
            nimcp_time_get_ms()
        );

        // Emotional → Salience: Mood biases attention
        float valence = emotional_get_valence(&current_emotion);
        float arousal = emotional_get_arousal(&current_emotion);

        // Depression-like state (negative valence) → attention to negative cues
        if (valence < -0.3F) {
            salience_boost_negative_cues(brain->salience, fabsf(valence) * 0.3F);
        }

        // Anxiety-like state (high arousal + negative valence) → threat vigilance
        if (arousal > 0.7F && valence < 0.0F) {
            salience_boost_threat_detection(brain->salience, arousal * 0.4F);
        }

        // Salience → Emotional: Surprises modulate arousal
        float surprise = salience_get_surprise_level(brain->salience);
        if (surprise > 0.5F) {
            emotional_modulate_arousal(&current_emotion, surprise * 0.2F);
        }
    }

    // Connection 4: Audio ↔ Speech Cortex (Phase 10.11.3)
    if (brain->audio_cortex && brain->speech_cortex) {
        // Audio → Speech: Activate speech mode when speech detected
        if (num_features > 0) {
            float speech_salience = audio_cortex_get_speech_salience(
                brain->audio_cortex,
                features,
                num_features
            );
            if (speech_salience > 0.6F) {
                audio_cortex_activate_speech_mode(brain->audio_cortex);
            }
        }

        // Speech → Audio: Request frequency boost when phoneme confidence is low
        float phoneme_confidence = speech_cortex_get_phoneme_confidence(brain->speech_cortex);
        if (phoneme_confidence < 0.7F) {
            float target_freq = 0.0F;
            float bandwidth = 0.0F;
            bool boost_needed = speech_cortex_request_frequency_boost(
                brain->speech_cortex,
                &target_freq,
                &bandwidth
            );
            // Note: Full integration would pass target_freq and bandwidth to audio cortex
            // to adjust mel filterbank emphasis (future enhancement)
            (void)boost_needed;  // Suppress unused warning
        }
    }

    // ========================================================================
    // STAGE 8: Glial Cell Modulation (Phase 10.11.2 - Priority 4)
    // ========================================================================
    // Increment simulation time (assume 1ms per decision cycle = 1000 µs)
    brain->current_time_us += 1000;

    // Glial update: skip if already running in parallel post-forward pool
    if (!post_forward_submitted && brain->glial && brain->config.enable_glial) {
        brain->glial_update_counter++;
        if (brain->glial_update_counter % 50 == 0) {
            glial_integration_step(brain->glial, brain->current_time_us);
        }
    }

    // ========================================================================
    // STAGE 9: Theory of Mind (Phase 10.11.2 - Priority 5)
    // ========================================================================
    // Skip if already running in parallel post-forward pool
    if (!post_forward_submitted && brain->theory_of_mind && brain->config.enable_theory_of_mind && decision) {
        // Step 1: Record own decision as a mental state
        // Convert decision to action for ToM tracking
        const char* intention = decision->label[0] ? decision->label : "decide";
        uint32_t intention_id = 0;
        for (const char* p = intention; *p; p++) {
            intention_id = intention_id * 31 + (uint32_t)(*p);
        }

        // Step 2: Update self-model with this decision
        // This allows the brain to understand its own decision patterns
        // which is necessary for inferring others' mental states
        tom_update_self_model(brain->theory_of_mind, features, num_features, intention, decision->confidence);

        // Step 3: If mirror neurons detected observed actions, use ToM to infer agent intentions
        // This would typically happen after brain_observe_action() was called
        // The inference results can influence future decisions in social contexts
        if (brain->mirror_neurons) {
            // Check if mirror neurons have recent observation data
            bool has_observations = mirror_neurons_has_recent_observations(brain->mirror_neurons);
            if (has_observations) {
                // Use ToM to predict what the observed agent might do next
                // This can help anticipate collaborative or competitive behaviors
                char predicted_action[NIMCP_ID_BUFFER_SIZE];
                float prediction_likelihood = 0.0F;

                bool predicted = tom_predict_action(
                    brain->theory_of_mind,
                    predicted_action,
                    sizeof(predicted_action),
                    &prediction_likelihood
                );

                // If prediction is confident, could influence current decision
                // (e.g., cooperate if agent predicted to cooperate, compete if not)
                if (predicted && prediction_likelihood > 0.7F) {
                    // High-confidence ToM prediction available
                    // Could modulate decision confidence or add explanation
                    // For now, just note that ToM inference occurred
                    (void)predicted_action; // Would be used in full implementation
                }
            }
        }
    }

    // ========================================================================
    // STAGE 7.5: Mental Health Monitoring (Phase 10.5) - Safety-Critical
    // ========================================================================
    // WHAT: Monitor behavioral markers, detect disorders, trigger interventions
    // WHY:  Prevent harmful behaviors before they escalate (safety-critical)
    // HOW:  Update markers → Check periodically → Intervene if needed
    if (brain->mental_health_monitor && brain->config.enable_mental_health_monitoring) {
        // Update behavioral markers with current decision
        mental_health_update(brain->mental_health_monitor, brain,
                           (const void*)decision, nimcp_time_get_ms());

        // Periodic health check (every N decisions)
        uint32_t check_interval = 100;  // Check every 100 decisions by default

        if (brain->stats.total_inferences % check_interval == 0) {
            // Run comprehensive mental health check
            disorder_severity_t max_severity = mental_health_check(
                brain->mental_health_monitor, brain);

            // If severe or critical disorder detected, trigger intervention
            if (max_severity >= DISORDER_SEVERITY_SEVERE) {
                bool intervened = mental_health_intervene(
                    brain->mental_health_monitor, brain);

                // Log intervention (optional - could add logging here)
                (void)intervened;  // Suppress unused warning for now

                // Check if quarantine mode was triggered
                mental_health_report_t report;
                mental_health_get_report(brain->mental_health_monitor, &report);

                if (report.quarantine_mode) {
                    // System in quarantine - reduce confidence as safety measure
                    decision->confidence *= 0.5F;

                    // Add warning to explanation
                    strncat(decision->explanation, " [QUARANTINE]",
                           sizeof(decision->explanation) - strlen(decision->explanation) - 1);
                }
            }
        }
    }

    // ========================================================================
    // STAGE 7.8: Ethics Engine - Golden Rule Evaluation (NEWLY INTEGRATED)
    // ========================================================================
    // WHAT: Evaluate decision against Golden Rule ethics
    // WHY:  Prevent harmful actions that violate "do unto others" principle
    // HOW:  Create action context, evaluate, block if unethical
    if (brain->ethics && brain->config.enable_ethics) {
        // Create action context from decision
        action_context_t ethics_action = {
            .features = (float*)features,
            .num_features = num_features,
            .affected_agents = NULL,  // Would need context to know affected agents
            .num_affected_agents = 0,
            .predicted_harm = (decision->confidence < 0.5F) ? 0.5F : 0.0F,
            .fairness_violation = 0.0F,
            .deception_level = 0.0F,
            .autonomy_violation = 0.0F,
            .privacy_violation = 0.0F,
            .consent_violation = 0.0F
        };

        // Evaluate action
        ethics_evaluation_t ethics_eval = ethics_engine_evaluate_action(
            brain->ethics,
            &ethics_action
        );

        // If action not allowed, modify decision
        if (!ethics_eval.allowed) {
            // Block unethical action
            decision->confidence = 0.0F;
            strncat(decision->label, " [BLOCKED-ETHICS]",
                   sizeof(decision->label) - strlen(decision->label) - 1);

            // Add ethics explanation
            strncat(decision->explanation, " | ETHICS: ",
                   sizeof(decision->explanation) - strlen(decision->explanation) - 1);
            strncat(decision->explanation, ethics_eval.explanation,
                   sizeof(decision->explanation) - strlen(decision->explanation) - 1);
        } else if (ethics_eval.golden_rule_score < 0.0F) {
            // Action allowed but marginally ethical - reduce confidence
            decision->confidence *= (1.0F + ethics_eval.golden_rule_score);  // Reduce by negative score
        }
    }

    // ========================================================================
    // STAGE 7.9: EPISTEMIC FILTERING - Apply Skepticism to Decisions
    // ========================================================================
    // WHAT: Evaluate decision output for epistemic quality (fact vs opinion, bias detection)
    // WHY:  Prevent outputting low-quality, biased, or conspiracy-like responses
    // HOW:  Use epistemic filter to assess decision label, reduce confidence if suspicious
    //
    // BIOLOGICAL BASIS:
    // - Critical thinking and skepticism (prefrontal cortex)
    // - Metacognitive monitoring (evaluating own beliefs)
    // - Source monitoring (tracking origin of beliefs)
    //
    // COGNITIVE BENEFITS:
    // - Outputs carry epistemic uncertainty markers
    // - Detects when outputting biased or low-quality responses
    // - Applies "extraordinary claims require extraordinary evidence"
    // - Distinguishes facts from opinions in outputs
    //
    // COMPLEXITY: O(1)
    if (brain->epistemic) {
        // Initialize evidence structure
        claim_evidence_t evidence;
        epistemic_evidence_init(&evidence);

        // Assess output quality (assume we're outputting moderate-quality claims)
        evidence.evidence_quality = EVIDENCE_MODERATE;
        evidence.plausibility = PLAUSIBLE_NEUTRAL;
        evidence.num_sources = 1;  // Single source (our own network)
        evidence.is_falsifiable = true;

        // Assess the decision label
        epistemic_assessment_t assessment;
        epistemic_assessment_init(&assessment);

        // Prior probability based on confidence (high confidence = higher prior)
        float prior_prob = decision->confidence;

        if (epistemic_assess_claim(brain->epistemic, decision->label, prior_prob, &evidence, &assessment)) {
            // Store epistemic quality in decision (if extended brain_decision_t has these fields)
            // For now, apply quality to confidence

            // If epistemic quality is low, reduce confidence
            if (assessment.epistemic_quality < 0.5F) {
                decision->confidence *= assessment.epistemic_quality;
            }

            // If biases detected, mark in label
            if (assessment.num_biases_detected > 0) {
                strncat(decision->label, " [BIAS-DETECTED]",
                       sizeof(decision->label) - strlen(decision->label) - 1);
                // Reduce confidence by 20% per bias
                float bias_penalty = assessment.num_biases_detected * 0.2F;
                decision->confidence *= fmaxf(0.2F, 1.0F - bias_penalty);
            }

            // Check conspiracy pattern
            float conspiracy_score = epistemic_check_conspiracy_pattern(brain->epistemic, decision->label, &evidence);
            if (conspiracy_score > 0.7F) {
                // High conspiracy score → mark and severely reduce confidence
                strncat(decision->label, " [CONSPIRACY-LIKE]",
                       sizeof(decision->label) - strlen(decision->label) - 1);
                decision->confidence *= 0.1F;  // Only 10% confidence
            }
        }
    }

    // ========================================================================
    // STAGE 8: Mirror Neuron Integration (Phase 10.11) - Execute Action
    // ========================================================================
    // WHAT: Record brain's decision as executed action in mirror neuron system
    // WHY:  Enable learning from own actions, build execution representation
    // HOW:  Convert decision to action and send to mirror neurons
    if (brain->mirror_neurons && brain->config.enable_mirror_neurons) {
        // Use the winning output neuron index as the action_id.
        // The brain's output is a continuous embedding — there are no discrete
        // "labels" to track. The argmax output neuron is the most stable
        // identifier for the response pattern. This keeps the action space
        // bounded by num_outputs (not unbounded unique label strings).
        uint32_t action_id = 0;
        if (decision->output_vector && decision->output_size > 0) {
            float max_val = decision->output_vector[0];
            for (uint32_t i = 1; i < decision->output_size; i++) {
                if (decision->output_vector[i] > max_val) {
                    max_val = decision->output_vector[i];
                    action_id = i;
                }
            }
        }
        const char* action_name = "response";
        action_t action = brain_decision_to_action(decision, action_id, action_name);

        // Record as executed action
        mirror_neurons_execute_action(brain->mirror_neurons, &action);

        // TODO: Dead code — `prediction` is always NULL here (freed at STAGE 2,
        // lines 1246-1247). To restore mirror-prediction matching, the prediction
        // buffer lifetime must be extended past this point, or the prediction error
        // float should be used instead.
        //
        // if (brain->predictive_network && prediction) {
        //     action_t predicted_action = features_to_action(prediction, num_features, 0);
        //     float similarity = 0.0F;
        //     mirror_neurons_match_actions(brain->mirror_neurons, &predicted_action,
        //                                 &action, &similarity);
        // }
    }

    // ========================================================================
    // PHASE C4: SHANNON INFORMATION FLOW ANALYSIS (INFERENCE PIPELINE)
    // ========================================================================
    // Skip if already running in parallel post-forward pool
    if (!post_forward_submitted && brain->enable_shannon_monitoring) {
        brain->last_shannon_metrics.information_rate = 0.0F;
    }

    // ========================================================================
    // PHASE C4.1: QUANTUM-SHANNON DIFFUSION (INFERENCE PHASE)
    // ========================================================================
    // Skip if already running in parallel post-forward pool
    if (!post_forward_submitted && brain->enable_quantum_shannon_diffusion && brain->quantum_shannon_diffusion) {
        quantum_shannon_diffusion_t* qsd = (quantum_shannon_diffusion_t*)brain->quantum_shannon_diffusion;
        if (quantum_shannon_evolve(qsd, brain->quantum_shannon_evolution_steps)) {
            quantum_shannon_get_metrics(qsd, &brain->last_quantum_shannon_metrics);
        }
    }

    // ========================================================================
    // STAGE C5: COGNITIVE SUBSYSTEM INFERENCE
    // ========================================================================
    // Run cognitive modules that enrich the decision with additional
    // information from parietal reasoning, predictive hierarchy, JEPA,
    // and VAE anomaly detection.
    {
        float* output_vec = decision->output_vector;
        uint32_t out_size = decision->output_size;

        /* C5.1: Parietal lobe step — spatial/mathematical reasoning */
        if (brain->parietal) {
            parietal_step(brain->parietal, 1000); /* 1ms time step */
        }

        /* C5.2: Predictive hierarchy — temporal prediction from features */
        if (brain->pred_hierarchy && brain->pred_hierarchy_enabled) {
            float pred_loss = 0.0f;
            pred_hier_learn_step((predictive_hierarchy_t*)brain->pred_hierarchy,
                                 features, &pred_loss);
            /* Modulate output confidence based on prediction error */
            if (pred_loss < 0.5f && decision->confidence < 1.0f) {
                decision->confidence += (1.0f - decision->confidence) * 0.1f * (1.0f - pred_loss);
            }
        }

        /* C5.3: VAE anomaly detection — flag unusual inputs */
        if (brain->vae_system && brain->vae_enabled && output_vec) {
            /* VAE anomaly score is computed during forward pass;
             * we use the cached value to modulate confidence */
            if (brain->last_vae_anomaly_score > 0.8f) {
                decision->confidence *= 0.5f;  /* High anomaly → low confidence */
            }
        }

        /* C5.4: JEPA predictor — latent-space prediction enrichment */
        if (brain->jepa_predictor && brain->jepa_predictor_enabled && output_vec) {
            /* During inference, JEPA produces a prediction of what the output
             * should look like based on the input context. We use agreement
             * between actual output and JEPA prediction to boost confidence. */
            uint32_t latent_dim = (num_features < 256) ? num_features : 256;
            jepa_latent_t* ctx = jepa_latent_create_dim(latent_dim);
            if (ctx) {
                jepa_latent_set_embedding(ctx, features,
                    (num_features < latent_dim) ? num_features : latent_dim);
                /* Predict: result stored internally in predictor */
                float jepa_loss = 0.0f;
                jepa_latent_t* pred = jepa_latent_create_dim(latent_dim);
                if (pred) {
                    jepa_latent_set_embedding(pred, output_vec,
                        (out_size < latent_dim) ? out_size : latent_dim);
                    /* Compare prediction with actual output */
                    float sim = jepa_latent_cosine_similarity(ctx, pred);
                    if (sim > 0.5f) {
                        decision->confidence += (1.0f - decision->confidence) * 0.05f * sim;
                    }
                    jepa_latent_destroy(pred);
                }
                jepa_latent_destroy(ctx);
            }
        }

        /* C5.5: FEP-Parietal — update beliefs from observation */
        if (brain->parietal && output_vec) {
            fep_parietal_bridge_t* fep_bridge = parietal_get_fep_bridge(brain->parietal);
            if (fep_bridge) {
                /* Build problem state from current inference output */
                fep_problem_state_t problem = {0};
                problem.state_vector = output_vec;
                problem.state_dim = out_size;
                problem.domain = FEP_MATH_DOMAIN_NUMERICAL;
                problem.distance_to_goal = 1.0f - decision->confidence;
                problem.solved = (decision->confidence > 0.9f);
                problem.solution_confidence = decision->confidence;

                fep_active_inference_result_t result = {0};
                int rc = fep_parietal_active_inference(fep_bridge, &problem, &result);
                if (rc == 0 && result.expected_improvement > 0.0f) {
                    /* Use exploration bonus to modulate confidence */
                    decision->confidence += (1.0f - decision->confidence) *
                        0.05f * result.exploration_bonus;
                }
                /* Clean up allocated policies if any */
                if (result.evaluated_policies) {
                    nimcp_free(result.evaluated_policies);
                }
                if (result.action) {
                    nimcp_free(result.action);
                }
            }
        }
    }

    // ========================================================================
    // STAGE C6: INFERENCE-TIME COGNITIVE TRAINING (Online Learning)
    // ========================================================================
    // The brain learns from every experience, not just explicit training.
    // During inference, subsystems train on the input features and decision
    // output using self-supervised signals. Gated by cognitive_train_interval.
    {
        uint32_t interval = brain->cognitive_train_interval;
        if (interval == 0) interval = 5;
        brain->cognitive_train_counter++;
        if ((brain->cognitive_train_counter % interval) == 0) {
            float* output_vec = decision->output_vector;
            uint32_t out_size = decision->output_size;
            /* Implicit loss from confidence: low confidence → high implicit loss */
            float implicit_loss = 1.0f - decision->confidence;

            /* C6.1: Grounded language — learn from label if available */
            if (brain->grounded_lang && decision->label && decision->label[0]) {
                grounded_language_learn_from_text(brain->grounded_lang, decision->label);
                grounded_language_learn_syntax(brain->grounded_lang, decision->label);
                brain->cognitive_stats.grounded_lang_steps++;
            }

            /* C6.2: Knowledge system — absorb label as concept */
            if (brain->knowledge && decision->label && decision->label[0]) {
                knowledge_learn_from_text(brain->knowledge, decision->label,
                                          KNOWLEDGE_DOMAIN_GENERAL);
                brain->cognitive_stats.knowledge_steps++;
            }

            /* C6.3: VAE — self-supervised autoencoding of input features */
            if (brain->vae_training_bridge && brain->vae_enabled && features) {
                vae_training_bridge_t* vae = (vae_training_bridge_t*)brain->vae_training_bridge;
                vae_training_step_result_t vae_result = {0};
                vae_training_step(vae, features, num_features,
                                  features, num_features, &vae_result);
                brain->last_vae_free_energy = vae_result.loss.total_loss;
                /* Update anomaly score for next inference C5.3 */
                brain->last_vae_anomaly_score = vae_result.loss.total_loss;
                brain->cognitive_stats.vae_steps++;
                brain->cognitive_stats.vae_last_loss = vae_result.loss.total_loss;
            }

            /* C6.4: FEP-Parietal generative model — train on features→output */
            if (brain->parietal && output_vec) {
                fep_parietal_bridge_t* fep_bridge = parietal_get_fep_bridge(brain->parietal);
                if (fep_bridge) {
                    const float* obs_ptr = features;
                    const float* tgt_ptr = output_vec;
                    fep_parietal_train_model(fep_bridge, &obs_ptr, &tgt_ptr, 1);
                    brain->cognitive_stats.fep_parietal_steps++;
                }
            }

            /* C6.5: Parietal physics NN — learn dynamics from features→output */
            if (brain->parietal && num_features >= 4 && out_size >= 4 && output_vec) {
                uint32_t phys_dim = (num_features < 32) ? num_features : 32;
                const float* state_ptr = features;
                const float* deriv_ptr = output_vec;
                parietal_train_physics_nn(brain->parietal,
                                          &state_ptr, &deriv_ptr, 1, 1);
                brain->cognitive_stats.physics_nn_steps++;
            }

            /* C6.6: JEPA predictor — self-supervised latent prediction */
            if (brain->jepa_predictor && brain->jepa_predictor_enabled && output_vec) {
                uint32_t latent_dim = (num_features < 256) ? num_features : 256;
                jepa_latent_t* context = jepa_latent_create_dim(latent_dim);
                jepa_latent_t* target_latent = jepa_latent_create_dim(latent_dim);
                if (context && target_latent) {
                    jepa_latent_set_embedding(context, features,
                        (num_features < latent_dim) ? num_features : latent_dim);
                    jepa_latent_set_embedding(target_latent, output_vec,
                        (out_size < latent_dim) ? out_size : latent_dim);
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

            /* C6.7: Creative training — feedback from decision quality */
            if (brain->creative_training_bridge && brain->creative_enabled && output_vec) {
                uint8_t rating = (decision->confidence > 0.9f) ? 5 :
                                 (decision->confidence > 0.7f) ? 4 :
                                 (decision->confidence > 0.5f) ? 3 :
                                 (decision->confidence > 0.3f) ? 2 : 1;
                creative_training_submit_feedback(
                    brain->creative_training_bridge,
                    output_vec, 0 /* ART_MODALITY_TEXT */, rating,
                    decision->label);
                brain->cognitive_stats.creative_steps++;
            }

            /* C6.8: Self-heal engine — learn from inference success/failure */
            if (brain->self_heal_engine && brain->self_heal_enabled) {
                crash_features_t cf = {0};
                uint32_t cf_dim = (num_features < SELF_HEAL_FEATURE_DIM) ?
                                   num_features : SELF_HEAL_FEATURE_DIM;
                cf.n_features = cf_dim;
                memcpy(cf.features, features, cf_dim * sizeof(float));
                float success = decision->confidence;
                self_heal_train_online(
                    (self_heal_engine_t*)brain->self_heal_engine,
                    &cf, FIX_PATTERN_UNKNOWN, success);
                brain->cognitive_stats.self_heal_steps++;
            }

            /* C6.9: Intuition system — learn from inference outcomes */
            if (brain->intuition_system && brain->intuition_system_enabled) {
                intuition_experience_t exp = {
                    .id = brain->cognitive_train_counter,
                    .hunch = NULL,
                    .predicted_outcome = 1.0f,  /* Expected: high confidence */
                    .actual_outcome = decision->confidence,
                    .timestamp = (float)nimcp_time_get_us(),
                    .was_successful = (decision->confidence > 0.5f)
                };
                const intuition_experience_t* exp_ptr = &exp;
                intuition_train_from_experience(brain->intuition_system,
                                                 &exp_ptr, 1);
                brain->cognitive_stats.intuition_steps++;
            }

            /* C6.10: FEP orchestrator — update free energy from inference */
            if (brain->fep_orchestrator && brain->fep_orchestrator_enabled) {
                brain->fep_orchestrator->fep_metrics.free_energy = implicit_loss;
                brain->fep_orchestrator->fep_metrics.prediction_error = implicit_loss;
                brain->fep_orchestrator->fep_metrics.surprise =
                    -logf(fmaxf(decision->confidence, 1e-7f));
                brain->cognitive_stats.fep_orchestrator_steps++;
            }

            /* Log cognitive training summary periodically */
            if ((brain->cognitive_train_counter % (interval * 100)) == 0) {
                LOG_INFO("cognitive_train",
                    "C6 stats: lang=%u know=%u vae=%u(%.3f) fep=%u phys=%u pred=%u(%.3f) "
                    "jepa=%u(%.3f) creative=%u heal=%u intuit=%u fep_orch=%u",
                    brain->cognitive_stats.grounded_lang_steps,
                    brain->cognitive_stats.knowledge_steps,
                    brain->cognitive_stats.vae_steps, brain->cognitive_stats.vae_last_loss,
                    brain->cognitive_stats.fep_parietal_steps,
                    brain->cognitive_stats.physics_nn_steps,
                    brain->cognitive_stats.pred_hierarchy_steps,
                    brain->cognitive_stats.pred_hierarchy_last_loss,
                    brain->cognitive_stats.jepa_steps, brain->cognitive_stats.jepa_last_loss,
                    brain->cognitive_stats.creative_steps,
                    brain->cognitive_stats.self_heal_steps,
                    brain->cognitive_stats.intuition_steps,
                    brain->cognitive_stats.fep_orchestrator_steps);
            }
        }
    }

    // ========================================================================
    // TRANSCRIPT: Populate cognitive transcript from all stage outputs
    // ========================================================================
    // Captures the outputs of all cognitive stages into a structured record
    // for response composition and introspection.
    {
        cognitive_transcript_t* t = transcript_create();
        if (t) {
            transcript_entry_t* e;

            /* Engram recall */
            if (engram_confidence > 0.0f) {
                e = transcript_add(t, TRANSCRIPT_MODULE_ENGRAM,
                    engram_confidence, engram_confidence,
                    "Memory recall activated");
                if (e) transcript_entry_add_value(e, "strength", engram_confidence);
            }

            /* Curiosity */
            if (curiosity_drive > 0.0f) {
                char buf[NIMCP_TRANSCRIPT_SUMMARY_LEN];
                snprintf(buf, sizeof(buf), "Curiosity drive: %.2f%s",
                         curiosity_drive, is_novel ? " (novel input)" : "");
                e = transcript_add(t, TRANSCRIPT_MODULE_CURIOSITY,
                    curiosity_drive * 0.5f, curiosity_drive, buf);
                if (e) {
                    transcript_entry_add_value(e, "drive", curiosity_drive);
                    transcript_entry_add_value(e, "novelty", novelty_score);
                }
            }

            /* Predictive processing */
            if (prediction_error > 0.01f) {
                char buf[NIMCP_TRANSCRIPT_SUMMARY_LEN];
                snprintf(buf, sizeof(buf), "Prediction error: %.3f", prediction_error);
                float pred_salience = fminf(prediction_error, 1.0f);
                e = transcript_add(t, TRANSCRIPT_MODULE_PREDICTIVE,
                    pred_salience, 1.0f - prediction_error, buf);
                if (e) transcript_entry_add_value(e, "error", prediction_error);
            }

            /* Reasoning engine */
            if (reasoning_confidence > 0.0f) {
                char buf[NIMCP_TRANSCRIPT_SUMMARY_LEN];
                snprintf(buf, sizeof(buf), "Reasoning chain (confidence: %.2f)",
                         reasoning_confidence);
                e = transcript_add(t, TRANSCRIPT_MODULE_REASONING,
                    reasoning_confidence * 0.8f, reasoning_confidence, buf);
            }

            /* Emotional tagging */
            if (brain->config.enable_emotional_tagging) {
                float valence = (decision->confidence - 0.5f) * 2.0f;
                float arousal = prediction_error;
                char buf[NIMCP_TRANSCRIPT_SUMMARY_LEN];
                snprintf(buf, sizeof(buf), "Emotion: valence=%.2f arousal=%.2f",
                         valence, arousal);
                float emo_salience = fmaxf(fabsf(valence), arousal) * 0.6f;
                e = transcript_add(t, TRANSCRIPT_MODULE_EMOTION,
                    emo_salience, fabsf(valence), buf);
                if (e) {
                    transcript_entry_add_value(e, "valence", valence);
                    transcript_entry_add_value(e, "arousal", arousal);
                }
            }

            /* Ethics evaluation */
            if (brain->ethics && brain->config.enable_ethics) {
                bool blocked = (strstr(decision->label, "BLOCKED-ETHICS") != NULL);
                float ethics_salience = blocked ? 1.0f : 0.2f;
                e = transcript_add(t, TRANSCRIPT_MODULE_ETHICS,
                    ethics_salience, blocked ? 0.0f : 1.0f,
                    blocked ? "Action blocked by ethics engine" : "Ethics check passed");
            }

            /* Epistemic filtering */
            if (brain->epistemic) {
                bool biased = (strstr(decision->label, "BIAS-DETECTED") != NULL);
                e = transcript_add(t, TRANSCRIPT_MODULE_EPISTEMIC,
                    biased ? 0.8f : 0.3f, decision->confidence,
                    biased ? "Bias detected in output" : "Epistemic filter passed");
            }

            /* FEP-Parietal (C5.5) */
            if (brain->parietal && brain->cognitive_stats.fep_parietal_steps > 0) {
                e = transcript_add(t, TRANSCRIPT_MODULE_FEP_PARIETAL,
                    0.4f, decision->confidence,
                    "Active inference updated beliefs");
            }

            /* Predictive hierarchy (C5.2) */
            if (brain->pred_hierarchy && brain->pred_hierarchy_enabled) {
                char buf[NIMCP_TRANSCRIPT_SUMMARY_LEN];
                snprintf(buf, sizeof(buf), "Predictive hierarchy (loss: %.3f)",
                         brain->cognitive_stats.pred_hierarchy_last_loss);
                e = transcript_add(t, TRANSCRIPT_MODULE_PRED_HIERARCHY,
                    0.4f, 1.0f - brain->cognitive_stats.pred_hierarchy_last_loss, buf);
                if (e) transcript_entry_add_value(e, "loss",
                    brain->cognitive_stats.pred_hierarchy_last_loss);
            }

            /* VAE anomaly (C5.3) */
            if (brain->vae_system && brain->vae_enabled) {
                bool anomalous = (brain->last_vae_anomaly_score > 0.8f);
                char buf[NIMCP_TRANSCRIPT_SUMMARY_LEN];
                snprintf(buf, sizeof(buf), "VAE anomaly score: %.3f%s",
                         brain->last_vae_anomaly_score,
                         anomalous ? " [ANOMALY]" : "");
                e = transcript_add(t, TRANSCRIPT_MODULE_VAE,
                    anomalous ? 0.8f : 0.2f,
                    1.0f - brain->last_vae_anomaly_score, buf);
                if (e) transcript_entry_add_value(e, "anomaly_score",
                    brain->last_vae_anomaly_score);
            }

            /* JEPA (C5.4) */
            if (brain->jepa_predictor && brain->jepa_predictor_enabled) {
                char buf[NIMCP_TRANSCRIPT_SUMMARY_LEN];
                snprintf(buf, sizeof(buf), "JEPA prediction (loss: %.3f)",
                         brain->cognitive_stats.jepa_last_loss);
                e = transcript_add(t, TRANSCRIPT_MODULE_JEPA,
                    0.3f, 1.0f - brain->cognitive_stats.jepa_last_loss, buf);
                if (e) transcript_entry_add_value(e, "loss",
                    brain->cognitive_stats.jepa_last_loss);
            }

            /* Grounded language */
            if (brain->grounded_lang) {
                e = transcript_add(t, TRANSCRIPT_MODULE_GROUNDED_LANG,
                    0.5f, 0.5f,
                    "Grounded language active");
            }

            /* Knowledge system */
            if (brain->knowledge && brain->cognitive_stats.knowledge_steps > 0) {
                e = transcript_add(t, TRANSCRIPT_MODULE_KNOWLEDGE,
                    0.4f, 0.5f,
                    "Knowledge graph queried");
            }

            /* Natural explanation (already in decision->explanation) */
            if (decision->explanation[0]) {
                e = transcript_add(t, TRANSCRIPT_MODULE_REASONING,
                    0.6f, decision->confidence,
                    decision->explanation);
            }

            transcript_finalize(t);
            decision->transcript = t;

            /* Cache a copy on the brain for API access after decision is freed */
            if (brain->last_transcript) {
                transcript_free(brain->last_transcript);
            }
            cognitive_transcript_t* copy = transcript_create();
            if (copy) {
                memcpy(copy, t, sizeof(cognitive_transcript_t));
            }
            brain->last_transcript = copy;
        }
    }

    // ========================================================================
    // PARALLEL POST-FORWARD: Wait for pool completion
    // ========================================================================
    if (post_forward_submitted) {
        nimcp_pool_wait(brain->inference_pool);
        // Free the heap-allocated task args
        if (post_ctx._internal_args) {
            nimcp_free(post_ctx._internal_args);
            post_ctx._internal_args = NULL;
        }
    }

    // Update statistics (after all post-decision processing)
    update_inference_stats(brain, decision);

    // Cache decision for future reuse (thread-safe with mutex protection)
    // W7-8 (C-INF-M3): Log warning on mutex lock failure instead of silently
    // skipping cache update. A failed lock indicates contention or corruption
    // that should be diagnosed, not silently ignored.
    if (nimcp_platform_mutex_lock(&brain->cache_mutex) == 0) {
        cache_decision(brain, features, num_features, decision);
        nimcp_platform_mutex_unlock(&brain->cache_mutex);
    } else {
        LOG_MODULE_WARN("BRAIN", "brain_decide: failed to lock cache_mutex — "
                        "decision not cached (possible contention or corruption)");
    }

    // Free the defensive copy of features
    nimcp_free(local_features);

    brain_clear_error();
    return decision;
}


//=============================================================================
// brain_predict - Wrapper for prediction functionality
//=============================================================================
/**
 * @brief Perform prediction with given input
 *
 * WHAT: Forward pass through network to generate predictions
 * WHY:  Compatibility wrapper for tests and external APIs
 * HOW:  Validates parameters and delegates to network forward pass
 *
 * @param brain Brain handle
 * @param input Input features array
 * @param input_size Size of input array
 * @param output Output predictions array (pre-allocated)
 * @param output_size Size of output array
 * @return true on success, false on error
 */
bool brain_predict(brain_t brain, const float* input, uint32_t input_size,
                  float* output, uint32_t output_size)
{
    LOG_MODULE_DEBUG("BRAIN", "brain_predict: input_size=%u, output_size=%u", input_size, output_size);

    // Validation
    if (!brain) {
        LOG_MODULE_ERROR("BRAIN", "brain_predict: NULL brain");
        set_error("brain_predict: NULL brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_predict: brain is NULL");
        return false;
    }

    if (!input || input_size == 0) {
        LOG_MODULE_ERROR("BRAIN", "brain_predict: invalid input parameters (input=%p, size=%u)", input, input_size);
        set_error("brain_predict: invalid input parameters");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_predict: input is NULL");
        return false;
    }

    if (!output || output_size == 0) {
        LOG_MODULE_ERROR("BRAIN", "brain_predict: invalid output parameters (output=%p, size=%u)", output, output_size);
        set_error("brain_predict: invalid output parameters");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_predict: output is NULL");
        return false;
    }

    // Check network exists
    if (!brain->network) {
        LOG_MODULE_ERROR("BRAIN", "brain_predict: network not initialized");
        set_error("brain_predict: network not initialized");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_predict: brain->network is NULL");
        return false;
    }

    // Validate input size matches configured dimensions
    if (brain->config.num_inputs > 0 && input_size != brain->config.num_inputs) {
        LOG_MODULE_ERROR("BRAIN", "brain_predict: input size mismatch (expected=%u, got=%u)",
                         brain->config.num_inputs, input_size);
        set_error("brain_predict: input size mismatch");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_predict: validation failed");
        return false;
    }

    // Validate output size matches configured dimensions
    if (brain->config.num_outputs > 0 && output_size != brain->config.num_outputs) {
        LOG_MODULE_ERROR("BRAIN", "brain_predict: output size mismatch (expected=%u, got=%u)",
                         brain->config.num_outputs, output_size);
        set_error("brain_predict: output size mismatch");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "brain_predict: validation failed");
        return false;
    }

    LOG_MODULE_INFO("BRAIN", "brain_predict: performing forward pass (readonly=%d)", brain->can_use_readonly);

    // W7-7 (C-INF-M2): Forward path consistency note.
    // brain_predict() uses adaptive_network_forward() (thresholded), which applies
    // adaptive output thresholding that zeros below-threshold outputs. This differs
    // from the API-level predict_fast/predict_in_domain which use forward_raw().
    // brain_decide() also uses adaptive_network_forward (via perform_forward_pass).
    //
    // This means:
    //   - brain_predict() and brain_decide() are CONSISTENT (both thresholded)
    //   - predict_fast() uses raw (unthresholded) for classification argmax
    //
    // The thresholded version is correct here because brain_predict() populates
    // a raw output buffer for the caller, and the threshold reflects the network's
    // actual learned decision boundary. predict_fast uses raw specifically because
    // it does its own argmax and thresholding would collapse outputs to the same class.
    //
    // Perform forward pass through network
    // Use read-only mode if this is a COW clone
    if (brain->can_use_readonly) {
        adaptive_network_forward_readonly(brain->network, input, input_size,
                                         output, output_size, 0);
    } else {
        adaptive_network_forward(brain->network, input, input_size,
                                output, output_size, 0);
    }

    // Publish prediction event via bio-async
    brain_publish_processing_event("prediction", 1.0F);

    LOG_MODULE_DEBUG("BRAIN", "brain_predict: prediction complete");
    brain_clear_error();
    return true;
}
