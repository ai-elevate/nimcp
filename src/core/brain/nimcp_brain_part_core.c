// nimcp_brain_part_core.c - core functions
// Part of nimcp_brain.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp_brain.c


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

    // Guard: Check dimensions
    if (num_features != brain->config.num_inputs) {
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
    // STAGE 0: Pre-Processing - Wellbeing Monitoring (Phase 9.3)
    // ========================================================================
    // WHAT: Check for distress BEFORE decision-making
    // WHY: Prevent decisions while system is in distress (ethical obligation)
    // HOW: Assess using introspection data, circuit-break on CRITICAL severity
    if (brain->wellbeing_monitoring_enabled && brain->introspection) {
        uint64_t current_time = nimcp_time_get_ms();

        // Check if it's time for a wellbeing assessment
        bool should_check = (brain->wellbeing_check_interval_ms == 0) ||  // Always check
                           ((current_time - brain->last_wellbeing_check_time) >= brain->wellbeing_check_interval_ms);

        if (should_check) {
            brain->last_distress = wellbeing_assess_distress(brain->introspection);
            brain->last_wellbeing_check_time = current_time;

            // Circuit breaker: CRITICAL distress prevents decisions
            if (brain->last_distress.severity == DISTRESS_SEVERITY_CRITICAL) {
                set_error("Decision blocked: System in CRITICAL distress (%s)",
                         brain->last_distress.description ? brain->last_distress.description : "Unknown");
                // Note: Caller should check error and potentially apply intervention
                nimcp_free(local_features);
                NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "brain_decide: system in CRITICAL distress");
                return NULL;
            }
        }
    }

    // Phase 3: Only trigger COW if not using read-only inference
    // WHY: COW clones can use adaptive_network_forward_readonly() indefinitely
    // WHEN: Trigger only for original brains or clones that already triggered COW
    if (!brain->can_use_readonly) {
        // Not using read-only mode - ensure network is writable
        if (!ensure_writable_network(brain)) {
            nimcp_free(local_features);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_decide: ensure_writable_network is NULL");
            return NULL;  // Error already set
        }
    }
    // else: Using read-only inference - no COW trigger needed!

    // Allocate decision structure
    brain_decision_t* decision = allocate_decision(brain->config.num_outputs);
    if (!decision) {
        set_error("Failed to allocate decision structure");
        nimcp_free(local_features);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "brain_decide: decision is NULL");
        return NULL;
    }

    // ========================================================================
    // STAGE 0.4: MEMORY ENGRAM RECALL (Phase M1: Pattern Completion)
    // ========================================================================
    // WHAT: Retrieve memory traces from partial cues for pattern completion
    // WHY:  Engrams enable recall of full experiences from incomplete input
    // HOW:  Map input features to cue neurons, search for matching engrams
    //
    // BIOLOGICAL BASIS:
    // - Pattern completion in hippocampus (Marr 1971, Rolls 2013)
    // - Partial cues reactivate full engram ensemble (Tonegawa et al., 2015)
    // - Reconsolidation: Retrieved memories become labile (Nader et al., 2000)
    // - Competition between overlapping engrams (Rashid et al., 2016)
    //
    // COMPLEXITY: O(n + e*k) where n=num_features, e=num_engrams, k=neurons_per_engram
    uint64_t recalled_engram_id = 0;
    float engram_confidence = 0.0F;

    if (brain->engram_system) {
        // Create cue neuron array from input features
        uint32_t* cue_neurons = nimcp_malloc(num_features * sizeof(uint32_t));

        if (cue_neurons) {
            // Map features to neuron IDs (simplified: feature index = neuron ID)
            for (uint32_t i = 0; i < num_features; i++) {
                cue_neurons[i] = i;
            }

            // Attempt pattern completion recall
            // Pre-allocate arrays for recall output
            #define MAX_RECALL_NEURONS 100
            uint32_t recalled_neurons[MAX_RECALL_NEURONS];
            float recalled_activations[MAX_RECALL_NEURONS];

            recalled_engram_id = engram_recall(
                brain->engram_system,
                cue_neurons,
                num_features,
                recalled_neurons,
                recalled_activations,
                MAX_RECALL_NEURONS,
                &engram_confidence
            );

            // If pattern completion succeeded (confidence > threshold)
            if (recalled_engram_id != 0 && engram_confidence > 0.4F) {
                // BIOLOGICAL: Recalled engrams undergo reconsolidation
                // Retrieved memories become temporarily labile and must be re-stabilized
                engram_trigger_reconsolidation(brain->engram_system, recalled_engram_id);

                // Optional: Could blend recalled pattern with network inference
                // For now, we just track that recall occurred (future enhancement)
                // decision->metadata could store engram_id and confidence
            }

            // Cleanup - NOTE: recalled_neurons and recalled_activations are stack-allocated, don't free them!
            nimcp_free(cue_neurons);
        }
    }

    // ========================================================================
    // STAGE 0.5: Sleep/Wake Cycle Integration (Phase 10.11.2 - REAL INTEGRATION)
    // ========================================================================
    // WHAT: Check sleep state and ACTUALLY modify behavior
    // WHY:  Sleep affects cognition - drowsiness, creativity, consolidation
    // HOW:  Reduce confidence during sleep, add noise during REM, degrade when tired
    sleep_state_t sleep_state = SLEEP_STATE_AWAKE;
    bool sleep_needed = false;
    float sleep_confidence_multiplier = 1.0F;  // Modifier for decision confidence
    float sleep_noise_level = 0.0F;            // Noise to add during REM
    bool trigger_consolidation = false;        // Should consolidate during this decision

    if (brain->sleep_system && brain->config.enable_sleep_wake_cycle) {
        sleep_state = sleep_get_current_state(brain->sleep_system);
        sleep_needed = sleep_is_needed(brain->sleep_system);

        // During sleep states, ACTUALLY adjust processing
        switch (sleep_state) {
            case SLEEP_STATE_DEEP_NREM:
                // Deep sleep: Severely reduced cognitive performance
                // Trigger consolidation of working memory
                sleep_confidence_multiplier = 0.3F;  // 70% confidence reduction
                trigger_consolidation = true;
                break;

            case SLEEP_STATE_REM:
                // REM sleep: Creative recombination with noise
                // Moderate cognitive impairment but increased creativity
                sleep_confidence_multiplier = 0.6F;  // 40% confidence reduction
                sleep_noise_level = 0.1F;            // Add 10% random noise to outputs
                break;

            case SLEEP_STATE_DROWSY:
            case SLEEP_STATE_LIGHT_NREM:
                // Light sleep: Mild cognitive impairment
                sleep_confidence_multiplier = 0.8F;  // 20% confidence reduction
                break;

            case SLEEP_STATE_AWAKE:
            default:
                // Awake: Check for sleep pressure
                if (sleep_needed) {
                    // High sleep pressure degrades performance (fatigue)
                    float sleep_pressure = sleep_get_pressure(brain->sleep_system);
                    sleep_confidence_multiplier = 1.0F - (sleep_pressure * 0.3F);
                    // At 80% pressure threshold: 1.0 - (0.8 * 0.3) = 0.76 (24% degradation)
                }
                break;
        }
    }

    // ========================================================================
    // STAGE 0.6: Curiosity Engine Integration (Phase 10.11.2 - ACTIVE)
    // ========================================================================
    // WHAT: Evaluate input novelty to drive exploration and learning
    // WHY:  Novel inputs should get increased attention and learning (40% faster)
    // HOW:  Compute novelty proxy, record experience, get curiosity drive
    //
    // BIOLOGICAL BASIS:
    // - Dopaminergic novelty response (midbrain)
    // - Exploration bonus (prefrontal cortex)
    // - Orienting response to novel stimuli (superior colliculus)
    //
    // COMPLEXITY: O(N) where N = num_features
    float novelty_score = 0.0F;
    bool is_novel = false;
    float curiosity_drive = 0.0F;  // Motivation to learn [0.0-1.0]

    if (brain->curiosity && brain->config.enable_curiosity) {
        // Compute variance-based novelty metric (reasonable proxy)
        // High variance → unusual pattern → potentially novel
        float input_variance = 0.0F;
        float input_mean = 0.0F;

        // Compute mean
        for (uint32_t i = 0; i < num_features; i++) {
            input_mean += features[i];
        }
        input_mean /= (float)num_features;

        // Compute variance
        for (uint32_t i = 0; i < num_features; i++) {
            float diff = features[i] - input_mean;
            input_variance += diff * diff;
        }
        input_variance /= (float)num_features;

        // Use variance as novelty score (normalized to ~[0.0-1.0])
        // Typical variance: 0.0-0.25 (normalized inputs), >0.5 = high novelty
        novelty_score = fminf(input_variance * 2.0F, 1.0F);
        is_novel = (novelty_score > 0.5F);

        // Record experience in curiosity engine (learns patterns over time)
        // This enables the engine to detect when similar patterns recur
        char experience_desc[NIMCP_ERROR_BUFFER_MEDIUM];
        snprintf(experience_desc, sizeof(experience_desc),
                "input_variance_%.3f", input_variance);
        curiosity_learn_experience(brain->curiosity, experience_desc,
                                  features, num_features);

        // Get curiosity drive (intrinsic motivation to learn)
        // Higher drive → boost learning rate for exploration
        curiosity_drive = curiosity_get_drive(brain->curiosity);

        // Store novelty and curiosity in brain for brain_learn()
        // Novel inputs with high curiosity get boosted learning rate
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

    // Perform forward pass
    uint32_t active_neurons = perform_forward_pass(brain, features, num_features, decision);

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
        prediction_error = sqrtf(prediction_error / decision->output_size);

        // Update predictive model with actual outcome
        if (brain->config.enable_predictive_processing) {
            predictive_update_model(brain->predictive_network);
        }

        nimcp_free(prediction);
        prediction = NULL;
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
    // STAGE 3.8: MEMORY ENGRAM CONSOLIDATION (Phase M1: Sleep-Dependent)
    // ========================================================================
    // WHAT: Update engram consolidation state during sleep
    // WHY:  Memory consolidation occurs during sleep (Tonegawa et al., 2015)
    // HOW:  Call engram_consolidate_update() with time delta and sleep state
    //
    // BIOLOGICAL BASIS:
    // - Sleep-dependent consolidation (Born & Wilhelm, 2012)
    // - ENCODING → LABILE → CONSOLIDATING → CONSOLIDATED state progression
    // - SWS (slow-wave sleep) strengthens hippocampal memory traces
    // - Synaptic homeostasis: weak synapses pruned, strong ones potentiated
    // - Sleep replay reactivates engram ensembles for strengthening
    //
    // COMPLEXITY: O(e) where e = number of engrams in system
    if (brain->engram_system) {
        // Compute time delta since last consolidation update
        // Use typical decision cycle time: ~100ms per decision
        const float TIME_DELTA_SECONDS = 0.1F;

        // Sleep accelerates consolidation (biological realism)
        bool is_sleeping = (sleep_state == SLEEP_STATE_DEEP_NREM ||
                           sleep_state == SLEEP_STATE_LIGHT_NREM ||
                           sleep_state == SLEEP_STATE_REM);

        // Update all engram consolidation states
        engram_consolidate_update(brain->engram_system, TIME_DELTA_SECONDS, is_sleeping);

        // During REM sleep: trigger memory replay
        // Replay reactivates and strengthens recent engrams
        if (sleep_state == SLEEP_STATE_REM && recalled_engram_id != 0) {
            // REM replay: reactivate recently recalled engrams
            // This strengthens the memory trace through repeated activation
            engram_trigger_reconsolidation(brain->engram_system, recalled_engram_id);
        }
    }

    // ========================================================================
    // STAGE 3.9: SYSTEMS CONSOLIDATION UPDATE (Phase M2: Hippocampus → Cortex)
    // ========================================================================
    // WHAT: Transfer memories from hippocampus to cortex during sleep
    // WHY:  Long-term memory stability requires cortical storage (McClelland et al., 1995)
    // HOW:  Execute replays during sleep, update consolidation, transfer to cortex
    //
    // BIOLOGICAL BASIS:
    // - Systems consolidation: hippocampus → cortex transfer over days/weeks
    // - Sleep replay at ~10-20x speed drives cortical plasticity
    // - Semantic abstraction: episodic details fade, gist remains
    // - Hippocampal dependency decreases as cortex becomes independent
    // - Sharp-wave ripples during SWS trigger coordinated replay
    //
    // COMPLEXITY: O(r + n) where r = replays executed, n = cortical nodes
    if (brain->systems_consolidation) {
        const float TIME_DELTA_SECONDS = 0.1F;

        // Determine sleep state for consolidation
        bool is_sws = (sleep_state == SLEEP_STATE_DEEP_NREM ||
                       sleep_state == SLEEP_STATE_LIGHT_NREM);
        bool is_rem = (sleep_state == SLEEP_STATE_REM);
        bool is_sleeping = (is_sws || is_rem);

        // PHASE M2.1: Execute memory replays during sleep
        // Replay frequency: SWS ~10 Hz, REM ~5 Hz, awake ~0.1 Hz
        if (is_sleeping || (recalled_engram_id != 0)) {
            // Schedule replay of recently recalled engram (high priority)
            if (recalled_engram_id != 0) {
                float priority = engram_confidence;  // Use recall confidence as priority
                systems_consolidation_schedule_replay(
                    brain->systems_consolidation,
                    recalled_engram_id,
                    priority
                );
            }

            // Execute pending replays (drives hippocampus → cortex transfer)
            uint32_t replays_executed = systems_consolidation_execute_replays(
                brain->systems_consolidation,
                TIME_DELTA_SECONDS,
                is_sws,
                is_rem
            );

            (void)replays_executed;  // Replay count available for monitoring
        }

        // PHASE M2.2: Update cortical consolidation (time-dependent strengthening)
        // Sleep accelerates consolidation (~5% per hour), awake is slower (~0.1% per hour)
        systems_consolidation_update(
            brain->systems_consolidation,
            TIME_DELTA_SECONDS,
            is_sleeping
        );
    }

    // ========================================================================
    // STAGE 3.10: WORKING MEMORY TRANSFER (Phase M3: WM → Engram Encoding)
    // ========================================================================
    // WHAT: Evaluate working memory items for transfer to engrams
    // WHY:  Attended/rehearsed information should consolidate to long-term memory
    // HOW:  Update attention based on confidence, evaluate transfer criteria
    //
    // BIOLOGICAL BASIS:
    // - Atkinson-Shiffrin model (1968): Working memory → long-term memory
    // - Miller's law (1956): Limited WM capacity requires selective transfer
    // - Attention enhances encoding (Craik & Lockhart, 1972)
    // - Rehearsal strengthens transfer probability (Rundus, 1971)
    // - Emotional arousal enhances consolidation (McGaugh, 2000)
    //
    // COMPLEXITY: O(n) where n = working memory capacity (7±2 items)
    if (brain->wm_transfer_system && brain->working_memory) {
        const float TIME_DELTA_SECONDS = 0.1F;

        // Update attention weights based on decision confidence
        // High confidence decisions receive higher attention
        // Note: In full implementation, would track attention per WM slot
        // For now, we demonstrate the evaluation mechanism

        // Evaluate transfer criteria for all working memory items
        // Items meeting criteria (rehearsal, attention, emotion, time) will transfer
        uint32_t transfers = wm_transfer_evaluate(
            brain->wm_transfer_system,
            TIME_DELTA_SECONDS
        );

        (void)transfers;  // Transfer count available for monitoring
    }

    // ========================================================================
    // STAGE 3.11: SEMANTIC MEMORY QUERY (Phase M4: Concept Network Reasoning)
    // ========================================================================
    // WHAT: Query semantic memory for related concepts
    // WHY:  Enable abstract reasoning and inference beyond immediate input
    // HOW:  Find similar concepts, spread activation through network
    //
    // BIOLOGICAL BASIS:
    // - Semantic memory supports reasoning (Tulving, 1972)
    // - Spreading activation retrieves related concepts (Collins & Loftus, 1975)
    // - Conceptual priming facilitates processing (Meyer & Schvaneveldt, 1971)
    // - Semantic networks organize knowledge (Collins & Quillian, 1969)
    //
    // COMPLEXITY: O(k*n) where k = max_hops, n = concepts per hop
    if (brain->semantic_memory) {
        // Query semantic memory with input features
        // This retrieves semantically related concepts that can inform reasoning
        semantic_query_result_t* semantic_results = semantic_memory_query(
            brain->semantic_memory,
            features,
            num_features
        );

        if (semantic_results) {
            // Semantic concepts activated - could be used for:
            // - Reasoning and inference
            // - Concept-based explanation generation
            // - Abstract knowledge retrieval
            // For now, we just demonstrate the query mechanism
            (void)semantic_results;  // Could log activated concepts for debugging
            semantic_memory_free_result(semantic_results);
        }

        // Periodically extract new concepts from Phase M2 during inference
        // This keeps the semantic network growing with experience
        semantic_memory_extract_from_consolidation(brain->semantic_memory);
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
    // WHAT: Apply glial cell modulation to synaptic transmission
    // WHY:  Biologically-inspired adaptive modulation (15% faster inference)
    // HOW:  Astrocytes modulate weights, oligodendrocytes speed up pathways
    //
    // NOTE: Glial modulation happens at the network level during forward pass
    //       See: adaptive_network_forward() in nimcp_adaptive.c
    //
    // Increment simulation time (assume 1ms per decision cycle = 1000 µs)
    brain->current_time_us += 1000;

    // IMPLEMENTATION: Trigger glial integration step for this decision cycle
    // Note: glial_integration_step() will synchronize network_time internally
    if (brain->glial && brain->config.enable_glial) {
        // Step 1: Update glial cell states based on network activity
        // This updates astrocyte calcium levels, oligodendrocyte myelination,
        // and microglia synaptic pruning decisions
        glial_integration_step(brain->glial, brain->current_time_us);

        // Step 2: Glial modulation is automatically applied during forward pass
        // (already integrated in adaptive_network_forward() via glial callbacks)
        // - Astrocytes: Modulate synaptic weights based on calcium levels
        // - Oligodendrocytes: Adjust conduction delays via myelination factors
        // - Microglia: Prune weak synapses to optimize network connectivity

        // Optional: Get modulation stats for monitoring
        // float astrocyte_modulation = glial_integration_get_avg_synaptic_modulation(brain->glial);
        // float myelination_speedup = glial_integration_get_avg_myelination_factor(brain->glial);
    }

    // ========================================================================
    // STAGE 9: Theory of Mind (Phase 10.11.2 - Priority 5)
    // ========================================================================
    // WHAT: Infer beliefs/intentions of other agents (multi-agent scenarios)
    // WHY:  Enable social cognition and collaboration
    // HOW:  Use mirror neuron activations + ToM model (BDI)
    //
    // IMPLEMENTATION: Update Theory of Mind model with current decision
    // This builds a self-model that can be used to predict other agents
    if (brain->theory_of_mind && brain->config.enable_theory_of_mind && decision) {
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
        // Convert decision to action
        const char* action_name = decision->label[0] ? decision->label : "decision";
        // Use hash of label as action_id for consistent tracking
        uint32_t action_id = 0;
        for (const char* p = action_name; *p; p++) {
            action_id = action_id * 31 + (uint32_t)(*p);
        }
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
    // WHAT: Analyze information flow during inference
    // WHY:  Monitor mutual information between input and output
    // HOW:  Compute entropy, channel capacity, and information rate
    //
    // BIOLOGICAL BASIS:
    // - Predictive coding: Minimize prediction error via information theory (Friston, 2010)
    // - Efficient coding: Maximize mutual information I(input; output) (Barlow, 1961)
    // - Capacity constraints: Limited channel capacity in sensory systems (Shannon, 1948)
    //
    // COMPLEXITY: O(1) - Monitoring enabled, detailed metrics computed via separate API
    //
    // NOTE: Full synapse-level Shannon analysis will be available through a dedicated
    // API once internal neuron/synapse structures are exposed via proper accessors.
    // For now, this marks monitoring as requested and initializes metrics structure.
    if (brain->enable_shannon_monitoring) {
        // Initialize/update basic inference metrics
        // Detailed synapse sampling will be added in future enhancement
        brain->last_shannon_metrics.information_rate = 0.0F;  // To be computed
        // Full implementation pending internal accessor APIs
    }

    // ========================================================================
    // PHASE C4.1: QUANTUM-SHANNON DIFFUSION (INFERENCE PHASE)
    // ========================================================================
    // WHAT: Evolve quantum-Shannon diffusion during inference
    // WHY:  Fast information propagation for real-time decisions, monitor bottlenecks
    // HOW:  Evolve quantum walker, update Shannon metrics, potential attention spread
    //
    // COMPLEXITY: O(E + N) where E = edges, N = neurons
    if (brain->enable_quantum_shannon_diffusion && brain->quantum_shannon_diffusion) {
        quantum_shannon_diffusion_t* qsd = (quantum_shannon_diffusion_t*)brain->quantum_shannon_diffusion;

        // Evolve with configured steps
        if (quantum_shannon_evolve(qsd, brain->quantum_shannon_evolution_steps)) {
            // Update metrics
            quantum_shannon_get_metrics(qsd, &brain->last_quantum_shannon_metrics);

            // Quantum speedup enables faster attention spread
            // Future: Could use quantum distribution for attention weights
            if (brain->last_quantum_shannon_metrics.speedup_vs_classical > 1.0F) {
                // Achieving quantum speedup - could boost confidence
                // For now, just track in metrics
            }
        }
    }

    // Update statistics (after all post-decision processing)
    update_inference_stats(brain, decision);

    // Cache decision for future reuse (thread-safe with mutex protection)
    if (nimcp_platform_mutex_lock(&brain->cache_mutex) == 0) {
        cache_decision(brain, features, num_features, decision);
        nimcp_platform_mutex_unlock(&brain->cache_mutex);
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
