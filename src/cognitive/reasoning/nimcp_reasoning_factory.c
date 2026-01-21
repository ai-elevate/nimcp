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
        return NULL;
    }

    NIMCP_LOGGING_INFO("Created symbolic logic engine (size=%d)", size);
    return engine;
}

symbolic_logic_t* create_symbolic_logic_with_config(const logic_config_t* config)
{
    if (!config) {
        set_error("Configuration is NULL");
        return NULL;
    }

    symbolic_logic_t* engine = symbolic_logic_create(config);

    if (!engine) {
        set_error("Failed to create symbolic logic engine");
        NIMCP_LOGGING_ERROR("create_symbolic_logic_with_config: creation failed");
        return NULL;
    }

    NIMCP_LOGGING_INFO("Created symbolic logic engine with custom config");
    return engine;
}

symbolic_logic_t* create_forward_chaining_engine(reasoning_size_t size)
{
    logic_config_t config = get_config_for_size(size);
    config.enable_forward_chaining = true;
    config.enable_backward_chaining = false;

    symbolic_logic_t* engine = symbolic_logic_create(&config);

    if (!engine) {
        set_error("Failed to create forward chaining engine");
        NIMCP_LOGGING_ERROR("create_forward_chaining_engine: creation failed");
        return NULL;
    }

    NIMCP_LOGGING_INFO("Created forward chaining engine (size=%d)", size);
    return engine;
}

symbolic_logic_t* create_backward_chaining_engine(reasoning_size_t size)
{
    logic_config_t config = get_config_for_size(size);
    config.enable_forward_chaining = false;
    config.enable_backward_chaining = true;

    symbolic_logic_t* engine = symbolic_logic_create(&config);

    if (!engine) {
        set_error("Failed to create backward chaining engine");
        NIMCP_LOGGING_ERROR("create_backward_chaining_engine: creation failed");
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
    const kg_entity_t* self = kg_reader_get_entity(kg, "Reasoning_Factory");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("Reasoning_Factory self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Reasoning_Factory");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Reasoning_Factory");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
