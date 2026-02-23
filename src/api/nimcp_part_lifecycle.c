// nimcp_part_lifecycle.c - lifecycle functions
// Part of nimcp.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp.c


//=============================================================================
// Initialization
//=============================================================================

/**
 * @brief Internal initialization function
 *
 * NOTE: Previously used nimcp_platform_once() which prevented re-initialization
 * after shutdown. Now uses atomic compare-exchange for thread-safe init that
 * can be repeated after shutdown.
 *
 * @return NIMCP_OK on success, NIMCP_ERROR on failure
 */
static nimcp_status_t nimcp_init_internal(void) {
    LOG_INFO("Initializing NIMCP library version %s", NIMCP_VERSION_STRING);

    // Initialize memory tracking (unified memory management)
    LOG_DEBUG("Initializing memory tracking system");
    nimcp_memory_init();

    // Initialize bio-async system (core async communication infrastructure)
    LOG_INFO("Initializing bio-async communication system");
    nimcp_bio_async_config_t bio_async_config = {0};  // Use default config
    if (nimcp_bio_async_init(&bio_async_config) != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to initialize bio-async system");
        nimcp_memory_cleanup();
        set_error("Failed to initialize bio-async system");
        return NIMCP_ERROR;
    }

    // Initialize bio-async router (message routing for modules)
    LOG_DEBUG("Initializing bio-async router");
    if (bio_router_init(NULL) != NIMCP_SUCCESS) {
        LOG_ERROR("Failed to initialize bio-async router");
        nimcp_bio_async_shutdown();
        nimcp_memory_cleanup();
        set_error("Failed to initialize bio-async router");
        return NIMCP_ERROR;
    }

    // Initialize COW cache system
    LOG_DEBUG("Initializing COW cache system");
    nimcp_cache_init();

    set_error("No error");
    LOG_INFO("NIMCP library initialized successfully");
    return NIMCP_OK;
}


nimcp_status_t nimcp_init(void) {
    // Fast path: already initialized
    if (nimcp_atomic_load_bool(&g_initialized, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        return g_init_result;
    }

    // Try to acquire init lock using compare-exchange
    // Only one thread will succeed; others will spin-wait
    bool expected = false;
    if (!nimcp_atomic_compare_exchange_bool(&g_init_in_progress, &expected, true,
                                             NIMCP_MEMORY_ORDER_ACQ_REL)) {
        // Another thread is initializing - wait for it to complete
        while (nimcp_atomic_load_bool(&g_init_in_progress, NIMCP_MEMORY_ORDER_ACQUIRE)) {
            // Spin-wait (could use yield/sleep for production)
        }
        return g_init_result;
    }

    // Double-check after acquiring lock (another thread may have initialized)
    if (nimcp_atomic_load_bool(&g_initialized, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        nimcp_atomic_store_bool(&g_init_in_progress, false, NIMCP_MEMORY_ORDER_RELEASE);
        return g_init_result;
    }

    // Perform actual initialization
    g_init_result = nimcp_init_internal();

    if (g_init_result == NIMCP_OK) {
        nimcp_atomic_store_bool(&g_initialized, true, NIMCP_MEMORY_ORDER_RELEASE);
    }

    // Release init lock
    nimcp_atomic_store_bool(&g_init_in_progress, false, NIMCP_MEMORY_ORDER_RELEASE);

    return g_init_result;
}

void nimcp_shutdown(void) {
    LOG_INFO("Shutting down NIMCP library");

    if (!nimcp_atomic_load_bool(&g_initialized, NIMCP_MEMORY_ORDER_ACQUIRE)) {
        LOG_DEBUG("NIMCP not initialized, nothing to shutdown");
        return;
    }

    // Cleanup cache system
    LOG_DEBUG("Cleaning up COW cache system");
    nimcp_cache_cleanup();

    // P1-8 FIX: Reset brain probe module context before shutting down bio-router
    nimcp_api_reset_brain_probe_ctx();

    // Shutdown bio-async router
    LOG_DEBUG("Shutting down bio-async router");
    bio_router_shutdown();

    // Shutdown bio-async system
    LOG_DEBUG("Shutting down bio-async communication system");
    nimcp_bio_async_shutdown();

    // Shutdown constant-time module (before memory cleanup to avoid double-free)
    LOG_DEBUG("Shutting down constant-time module");
    nimcp_ct_shutdown();

    // Cleanup adaptive module memory pool (before memory cleanup)
    LOG_DEBUG("Cleaning up adaptive network memory pool");
    adaptive_pool_cleanup();

    // Shutdown exception system (before memory cleanup to avoid dangling references)
    LOG_DEBUG("Shutting down exception system");
    bbb_helpers_shutdown();
    nimcp_exception_system_shutdown();

    // Cleanup memory tracking (last)
    LOG_DEBUG("Cleaning up memory tracking");
    nimcp_memory_cleanup();

    // Reset init state to allow re-initialization
    g_init_result = NIMCP_OK;  // Reset to default for next init
    nimcp_atomic_store_bool(&g_initialized, false, NIMCP_MEMORY_ORDER_RELEASE);
    LOG_INFO("NIMCP library shutdown complete");
}


//=============================================================================
// Brain API Implementation
//=============================================================================

nimcp_brain_t nimcp_brain_create(
    const char* name,
    nimcp_brain_size_t size,
    nimcp_brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs)
{
    LOG_INFO("Creating brain: name='%s', size=%d, task=%d, inputs=%u, outputs=%u",
             name ? name : "NULL", size, task, num_inputs, num_outputs);

    /* Use exception-integrated validation (returns NULL on error) */
    NIMCP_API_CHECK_NULL_RET_NULL(name, "Brain name cannot be NULL");

    /* Allocate handle with exception-integrated check */
    LOG_DEBUG("Allocating brain handle (%zu bytes)", sizeof(struct nimcp_brain_handle));
    nimcp_brain_t handle = (nimcp_brain_t)nimcp_malloc(sizeof(struct nimcp_brain_handle));
    NIMCP_API_CHECK_ALLOC_SIZE(handle, sizeof(struct nimcp_brain_handle),
                               "Failed to allocate brain handle");

    // Map public enums to internal enums
    brain_size_t internal_size = (brain_size_t)size;
    brain_task_t internal_task = (brain_task_t)task;
    LOG_DEBUG("Mapped enums: internal_size=%d, internal_task=%d", internal_size, internal_task);

    // Create internal brain
    LOG_DEBUG("Creating internal brain structure");
    handle->internal_brain = brain_create(name, internal_size, internal_task,
                                          num_inputs, num_outputs);

    if (!handle->internal_brain) {
        LOG_ERROR("Failed to create internal brain for '%s'", name);
        set_error("Failed to create internal brain");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_brain_create: handle->internal_brain is NULL");
        return NULL;
    }

    set_error("No error");
    LOG_INFO("Brain '%s' created successfully (handle=%p)", name, (void*)handle);
    return handle;
}

void nimcp_brain_destroy(nimcp_brain_t brain) {
    if (!brain) {
        set_error("nimcp_brain_destroy: NULL brain handle");
        LOG_DEBUG("nimcp_brain_destroy called with NULL brain, ignoring");
        return;
    }

    LOG_INFO("Destroying brain (handle=%p)", (void*)brain);

    // Clean up training pipeline state BEFORE destroying internal brain
    // This prevents memory leaks from the global g_training_states array
    nimcp_api_training_cleanup_brain(brain);

    if (brain->internal_brain) {
        LOG_DEBUG("Destroying internal brain structure");
        brain_destroy(brain->internal_brain);
    } else {
        LOG_WARN("Brain handle has NULL internal_brain");
    }

    LOG_DEBUG("Freeing brain handle");
    nimcp_free(brain);
    LOG_DEBUG("Brain destroyed successfully");
}


nimcp_brain_t nimcp_brain_create_with_neurons(
    const char* name,
    nimcp_brain_task_t task,
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t neuron_count)
{
    LOG_INFO("Creating brain with %u neurons: name='%s', task=%d, inputs=%u, outputs=%u",
             neuron_count, name ? name : "NULL", task, num_inputs, num_outputs);

    NIMCP_API_CHECK_NULL_RET_NULL(name, "Brain name cannot be NULL");

    nimcp_brain_t handle = (nimcp_brain_t)nimcp_malloc(sizeof(struct nimcp_brain_handle));
    NIMCP_API_CHECK_ALLOC_SIZE(handle, sizeof(struct nimcp_brain_handle),
                               "Failed to allocate brain handle");

    brain_size_t internal_size = BRAIN_SIZE_LARGE;
    brain_task_t internal_task = (brain_task_t)task;

    // Build config with neuron_count override
    brain_config_t config = {0};
    task_strategy_t* strategy = strategy_create(internal_task);
    if (!strategy) {
        nimcp_free(handle);
        return NULL;
    }
    nimcp_brain_factory_init_brain_config(&config, name, internal_size, internal_task,
                                          num_inputs, num_outputs, strategy);
    strategy_destroy(strategy);
    config.neuron_count = neuron_count;

    handle->internal_brain = brain_create_custom(&config);
    if (!handle->internal_brain) {
        nimcp_free(handle);
        return NULL;
    }

    set_error("No error");
    LOG_INFO("Brain '%s' created with %u neurons (handle=%p)", name, neuron_count, (void*)handle);
    return handle;
}

nimcp_brain_t nimcp_brain_create_from_config(const char* config_filepath) {
    if (!config_filepath) {
        set_error("Config filepath is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config_filepath is NULL");

        return NULL;
    }

    // Load configuration from YAML/JSON
    nimcp_brain_config_t config;
    if (!nimcp_config_load(config_filepath, &config)) {
        set_error("Failed to load config from %s", config_filepath);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_create_from_config: nimcp_config_load is NULL");
        return NULL;
    }

    // Create brain using loaded configuration
    nimcp_brain_t brain = nimcp_brain_create(
        config.name,
        (nimcp_brain_size_t)config.size,
        (nimcp_brain_task_t)config.task,
        config.num_inputs,
        config.num_outputs
    );

    if (!brain) {
        set_error("Failed to create brain from config");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");


        return NULL;
    }

    // Note: Additional configuration like BCM, STDP, ethics could be applied here
    // For now, we just create the basic brain structure
    // TODO: Apply plasticity settings, ethics config, etc.

    set_error("No error");
    return brain;
}

static void brain_probe_module_init_once(void) {
    if (!bio_router_is_initialized()) {
        return;
    }
    bio_module_info_t info = {
        .module_id = BIO_MODULE_BRAIN,
        .module_name = "brain_probe",
        .inbox_capacity = 64,
        .user_data = NULL
    };
    g_brain_probe_module_ctx = bio_router_register_module(&info);
}


/**
 * @brief Reset brain probe module state for re-initialization after shutdown.
 * Called from nimcp_shutdown() (forward-declared there).
 */
void nimcp_api_reset_brain_probe_ctx(void) {
    g_brain_probe_module_ctx = NULL;
    g_brain_probe_once = (nimcp_platform_once_t)NIMCP_PLATFORM_ONCE_INIT;
}


/**
 * WHAT: Destroy brain snapshot and release COW references
 * WHY:  Free snapshot resources and decrement reference counts
 * HOW:  Release cached references and free snapshot handle
 */
void nimcp_brain_snapshot_destroy(nimcp_brain_snapshot_t snapshot) {
    if (!snapshot) {
        return;
    }

    // Phase 1: Free the cloned internal brain
    if (snapshot->internal_brain_snapshot) {
        brain_destroy(snapshot->internal_brain_snapshot);
    }

    // Free snapshot handle
    nimcp_free(snapshot);
}


//=============================================================================
// Neural Network API Implementation
//=============================================================================

nimcp_network_t nimcp_network_create(
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t num_hidden,
    float learning_rate)
{
    // Allocate handle
    nimcp_network_t handle = (nimcp_network_t)nimcp_malloc(sizeof(struct nimcp_network_handle));
    if (!handle) {
        set_error("Failed to allocate network handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "handle is NULL");

        return NULL;
    }

    // Create config for internal API
    network_config_t config = {0};
    config.input_size = num_inputs;
    config.output_size = num_outputs;
    // Calculate total neurons: inputs + hidden layers + outputs
    config.num_neurons = num_inputs + num_hidden + num_outputs;
    config.learning_rate = learning_rate;

    // Create internal neural network
    handle->internal_network = neural_network_create(&config);

    if (!handle->internal_network) {
        set_error("Failed to create internal neural network");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_network_create: handle->internal_network is NULL");
        return NULL;
    }

    set_error("No error");
    return handle;
}


void nimcp_network_destroy(nimcp_network_t network) {
    if (!network) {
        return;
    }

    if (network->internal_network) {
        neural_network_destroy(network->internal_network);
    }

    nimcp_free(network);
}


//=============================================================================
// Ethics API Implementation
//=============================================================================

nimcp_ethics_t nimcp_ethics_create(void) {
    nimcp_ethics_t handle = (nimcp_ethics_t)nimcp_malloc(sizeof(struct nimcp_ethics_handle));
    if (!handle) {
        set_error("Failed to allocate ethics handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "handle is NULL");

        return NULL;
    }

    // Create default ethics configuration
    ethics_config_t config = {0};
    config.policies = NULL;
    config.num_policies = 0;
    config.callback = NULL;
    config.callback_context = NULL;
    config.default_severity = 0.5F;
    config.enable_learning = true;
    config.action_feature_size = 32;
    config.max_agents = 16;
    config.golden_rule_threshold = 0.0F;
    config.empathy_weight = 0.5F;

    // Create internal ethics engine
    handle->internal_ethics = ethics_engine_create(&config);

    if (!handle->internal_ethics) {
        set_error("Failed to create internal ethics engine");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_ethics_create: handle->internal_ethics is NULL");
        return NULL;
    }

    set_error("No error");
    return handle;
}


void nimcp_ethics_destroy(nimcp_ethics_t ethics) {
    if (!ethics) {
        return;
    }

    if (ethics->internal_ethics) {
        ethics_engine_destroy(ethics->internal_ethics);
    }

    nimcp_free(ethics);
}


//=============================================================================
// Knowledge API Implementation
//=============================================================================

nimcp_knowledge_t nimcp_knowledge_create(void) {
    nimcp_knowledge_t handle = (nimcp_knowledge_t)nimcp_malloc(sizeof(struct nimcp_knowledge_handle));
    if (!handle) {
        set_error("Failed to allocate knowledge handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "handle is NULL");

        return NULL;
    }

    // Create internal knowledge system
    handle->internal_knowledge = knowledge_system_create("default");

    if (!handle->internal_knowledge) {
        set_error("Failed to create internal knowledge system");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "nimcp_knowledge_create: handle->internal_knowledge is NULL");
        return NULL;
    }

    set_error("No error");
    return handle;
}


void nimcp_knowledge_destroy(nimcp_knowledge_t knowledge) {
    if (!knowledge) {
        return;
    }

    if (knowledge->internal_knowledge) {
        knowledge_system_destroy(knowledge->internal_knowledge);
    }

    nimcp_free(knowledge);
}


/**
 * @brief Cleanup training state when brain is destroyed
 *
 * Called from nimcp_brain_destroy to prevent memory leaks from training state.
 */
void nimcp_api_training_cleanup_brain(nimcp_brain_t brain) {
    if (!brain) return;
    clear_training_state(brain);
}
