/**
 * @file nimcp_reasoning_factory.c
 * @brief MODULE 6: Reasoning Factory implementation
 *
 * @author NIMCP Development Team - SRP Refactoring
 * @date 2025-11-20
 * @version 3.0.0
 */

#include "cognitive/reasoning/nimcp_reasoning_factory.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "api/nimcp_api_exception.h"

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "nimcp.h"  // For error codes
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#define LOG_MODULE "reasoning"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(reasoning_factory)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_reasoning_factory_mesh_id = 0;
static mesh_participant_registry_t* g_reasoning_factory_mesh_registry = NULL;

nimcp_error_t reasoning_factory_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_reasoning_factory_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "reasoning_factory", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "reasoning_factory";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_reasoning_factory_mesh_id);
    if (err == NIMCP_SUCCESS) g_reasoning_factory_mesh_registry = registry;
    return err;
}

void reasoning_factory_mesh_unregister(void) {
    if (g_reasoning_factory_mesh_registry && g_reasoning_factory_mesh_id != 0) {
        mesh_participant_unregister(g_reasoning_factory_mesh_registry, g_reasoning_factory_mesh_id);
        g_reasoning_factory_mesh_id = 0;
        g_reasoning_factory_mesh_registry = NULL;
    }
}


/** @brief Send heartbeat from reasoning_factory module (instance-level) */
static inline void reasoning_factory_heartbeat_instance(
    nimcp_health_agent_t* instance_agent, const char* operation, float progress)
{
    if (g_reasoning_factory_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_reasoning_factory_health_agent, operation, progress);
    }
    if (instance_agent && instance_agent != g_reasoning_factory_health_agent) {
        nimcp_health_agent_heartbeat_ex(instance_agent, operation, progress);
    }
}



static __thread char last_error[256] = {0};

static void set_error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(last_error, sizeof(last_error), fmt, args);
    va_end(args);
}

const char* reasoning_factory_get_last_error(void)
{
    return last_error;
}

//=============================================================================
// Bio-Async Integration (Singleton with thread-safe initialization)
//=============================================================================

#include "utils/platform/nimcp_platform_once.h"
#include <stdatomic.h>

static _Atomic(bio_module_context_t) g_factory_bio_ctx = NULL;
static atomic_bool g_factory_bio_async_enabled = false;
static nimcp_platform_once_t g_factory_bio_once = NIMCP_PLATFORM_ONCE_INIT;

/**
 * @brief One-time initialization of bio-async for reasoning factory
 */
static void factory_init_bio_async_once(void)
{
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_KNOWLEDGE_FACTORY,
            .module_name = "reasoning_factory",
            .inbox_capacity = 64,
            .user_data = NULL
        };
        bio_module_context_t ctx = bio_router_register_module(&bio_info);
        if (ctx) {
            atomic_store(&g_factory_bio_ctx, ctx);
            atomic_store(&g_factory_bio_async_enabled, true);
            NIMCP_LOGGING_INFO("Bio-async enabled for reasoning factory");
        } else {
            NIMCP_LOGGING_WARN("Bio-async registration failed for reasoning factory");
        }
    }
}

/**
 * @brief Initialize bio-async for reasoning factory (thread-safe)
 */
static void factory_init_bio_async(void)
{
    nimcp_platform_once(&g_factory_bio_once, factory_init_bio_async_once);
}

static logic_config_t get_config_for_size(reasoning_size_t size)
{
    logic_config_t config = {0};

    switch (size) {
        case REASONING_SIZE_SMALL:
            config.max_predicates = 200;
            config.max_rules = 50;
            config.max_kb_size = 100;
            config.max_inference_depth = 5;
            break;
        case REASONING_SIZE_MEDIUM:
            config.max_predicates = 500;
            config.max_rules = 250;
            config.max_kb_size = 500;
            config.max_inference_depth = 10;
            break;
        case REASONING_SIZE_LARGE:
            config.max_predicates = 1000;
            config.max_rules = 500;
            config.max_kb_size = 1000;
            config.max_inference_depth = 15;
            break;
        default:
            config.max_predicates = 500;
            config.max_rules = 250;
            config.max_kb_size = 500;
            config.max_inference_depth = 10;
    }

    config.enable_forward_chaining = true;
    config.enable_backward_chaining = true;
    config.enable_resolution = true;
    config.enable_memory_consolidation = true;

    return config;
}

symbolic_logic_t* create_default_symbolic_logic(reasoning_size_t size)
{
    // Initialize bio-async on first use
    /* Phase 8: Heartbeat at operation start */
    reasoning_factory_heartbeat("reasoning_fa_create_default_symbo", 0.0f);


    factory_init_bio_async();

    // Process pending bio-async messages
    if (atomic_load(&g_factory_bio_async_enabled)) {
        bio_module_context_t ctx = atomic_load(&g_factory_bio_ctx);
        if (ctx) {
            bio_router_process_inbox(ctx, 5);
        }
    }

    logic_config_t config = get_config_for_size(size);
    symbolic_logic_t* engine = symbolic_logic_create(&config);

    if (!engine) {
        set_error("Failed to create symbolic logic engine");
        NIMCP_LOGGING_ERROR("create_default_symbolic_logic: creation failed");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");


        return NULL;
    }

    NIMCP_LOGGING_INFO("Created symbolic logic engine (size=%d)", size);
    return engine;
}

symbolic_logic_t* create_symbolic_logic_with_config(const logic_config_t* config)
{
    if (!config) {
        set_error("Configuration is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return NULL;
    }

    /* Phase 8: Heartbeat at operation start */
    reasoning_factory_heartbeat("reasoning_fa_create_symbolic_logi", 0.0f);


    symbolic_logic_t* engine = symbolic_logic_create(config);

    if (!engine) {
        set_error("Failed to create symbolic logic engine");
        NIMCP_LOGGING_ERROR("create_symbolic_logic_with_config: creation failed");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");


        return NULL;
    }

    NIMCP_LOGGING_INFO("Created symbolic logic engine with custom config");
    return engine;
}

symbolic_logic_t* create_forward_chaining_engine(reasoning_size_t size)
{
    /* Phase 8: Heartbeat at operation start */
    reasoning_factory_heartbeat("reasoning_fa_create_forward_chain", 0.0f);


    logic_config_t config = get_config_for_size(size);
    config.enable_forward_chaining = true;
    config.enable_backward_chaining = false;

    symbolic_logic_t* engine = symbolic_logic_create(&config);

    if (!engine) {
        set_error("Failed to create forward chaining engine");
        NIMCP_LOGGING_ERROR("create_forward_chaining_engine: creation failed");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");


        return NULL;
    }

    NIMCP_LOGGING_INFO("Created forward chaining engine (size=%d)", size);
    return engine;
}

symbolic_logic_t* create_backward_chaining_engine(reasoning_size_t size)
{
    /* Phase 8: Heartbeat at operation start */
    reasoning_factory_heartbeat("reasoning_fa_create_backward_chai", 0.0f);


    logic_config_t config = get_config_for_size(size);
    config.enable_forward_chaining = false;
    config.enable_backward_chaining = true;

    symbolic_logic_t* engine = symbolic_logic_create(&config);

    if (!engine) {
        set_error("Failed to create backward chaining engine");
        NIMCP_LOGGING_ERROR("create_backward_chaining_engine: creation failed");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "engine is NULL");


        return NULL;
    }

    NIMCP_LOGGING_INFO("Created backward chaining engine (size=%d)", size);
    return engine;
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

int reasoning_factory_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    /* Phase 8: Heartbeat at operation start */
    reasoning_factory_heartbeat("reasoning_fa_query_self_knowledge", 0.0f);


    const kg_entity_t* self = kg_reader_get_entity(kg, "Reasoning_Factory");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && self->num_observations > 256) {
                reasoning_factory_heartbeat("reasoning_fa_loop",
                                 (float)(i + 1) / (float)self->num_observations);
            }

            NIMCP_LOGGING_DEBUG("Reasoning_Factory self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Reasoning_Factory");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Reasoning_Factory");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}

/* ============================================================================
 * Phase 8: Instance-Level Health Agent
 * ============================================================================ */

void reasoning_factory_set_instance_health_agent(void* instance, nimcp_health_agent_t* agent) {
    if (instance) {
        (void)agent;
        g_reasoning_factory_health_agent = agent;
    }
}

/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int reasoning_factory_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "reasoning_factory_training_begin: NULL argument");
        return -1;
    }
    reasoning_factory_heartbeat_instance(NULL, "reasoning_factory_training_begin", 0.0f);
    return 0;
}

int reasoning_factory_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "reasoning_factory_training_end: NULL argument");
        return -1;
    }
    reasoning_factory_heartbeat_instance(NULL, "reasoning_factory_training_end", 1.0f);
    return 0;
}

int reasoning_factory_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "reasoning_factory_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    reasoning_factory_heartbeat_instance(NULL, "reasoning_factory_training_step", progress);
    return 0;
}
