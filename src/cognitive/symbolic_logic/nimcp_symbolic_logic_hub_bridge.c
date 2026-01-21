/**
 * @file nimcp_symbolic_logic_hub_bridge.c
 * @brief Symbolic Logic - Cognitive Hub Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-08
 */

#include "cognitive/symbolic_logic/nimcp_symbolic_logic_hub_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/validation/nimcp_common.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

#define LOG_MODULE "symbolic_logic_hub_bridge"

/* ============================================================================
 * Event Callback Declarations
 * ============================================================================ */

static int on_memory_access(const cognitive_event_data_t* event, void* user_data);
static int on_attention_shift(const cognitive_event_data_t* event, void* user_data);
static int on_learning_complete(const cognitive_event_data_t* event, void* user_data);

/* ============================================================================
 * Lifecycle API Implementation
 * ============================================================================ */

int symbolic_logic_hub_bridge_default_config(symbolic_logic_hub_config_t* config) {
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");

    /* Event subscriptions */
    config->subscribe_memory_access = true;
    config->subscribe_attention_shift = true;
    config->subscribe_emotion_update = true;
    config->subscribe_decision_made = true;
    config->subscribe_learning_complete = true;

    /* Event publishing */
    config->publish_inference_results = true;
    config->publish_fact_additions = true;
    config->publish_contradiction_found = true;

    /* Processing parameters */
    config->inference_priority = 0.7f;
    config->max_forward_chain_depth = 10;
    config->max_backward_chain_depth = 15;
    config->novelty_threshold = 0.3f;

    /* Integration enables */
    config->enable_memory_tagging = true;
    config->enable_attention_bias = true;
    config->enable_emotional_weights = true;

    return 0;
}

symbolic_logic_hub_bridge_t* symbolic_logic_hub_bridge_create(
    const symbolic_logic_hub_config_t* config)
{
    symbolic_logic_hub_bridge_t* bridge = nimcp_malloc(sizeof(symbolic_logic_hub_bridge_t));
    if (!bridge) return NULL;

    memset(bridge, 0, sizeof(symbolic_logic_hub_bridge_t));

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        symbolic_logic_hub_bridge_default_config(&bridge->config);
    }

    /* Initialize base bridge */
    bridge->base.mutex = nimcp_platform_mutex_create();
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state.is_registered = false;
    bridge->state.is_active = false;
    bridge->state.pending_inferences = 0;
    bridge->state.current_load = 0.0f;

    NIMCP_LOGGING_INFO("Symbolic logic hub bridge created");
    return bridge;
}

void symbolic_logic_hub_bridge_destroy(symbolic_logic_hub_bridge_t* bridge) {
    if (!bridge) return;

    /* Disconnect from hub */
    if (bridge->state.is_registered) {
        symbolic_logic_hub_bridge_disconnect(bridge);
    }

    /* Disconnect bio-async */
    if (bridge->base.bio_async_enabled) {
        symbolic_logic_hub_bridge_disconnect_bio_async(bridge);
    }

    /* Free mutex */
    if (bridge->base.mutex) {
        nimcp_platform_mutex_destroy(bridge->base.mutex);
    }

    nimcp_free(bridge);
    NIMCP_LOGGING_INFO("Symbolic logic hub bridge destroyed");
}

/* ============================================================================
 * Connection API Implementation
 * ============================================================================ */

int symbolic_logic_hub_bridge_connect_hub(
    symbolic_logic_hub_bridge_t* bridge,
    cognitive_integration_hub_t hub)
{
    NIMCP_CHECK_THROW(bridge && hub, NIMCP_ERROR_NULL_POINTER, "bridge or hub is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->hub = hub;

    /* Register module with hub */
    int result = cognitive_hub_register_module(
        hub,
        LOGIC_HUB_MODULE_ID,
        COG_CATEGORY_REASONING,
        LOGIC_HUB_MODULE_NAME,
        bridge);

    if (result != 0) {
        nimcp_mutex_unlock(bridge->base.mutex);
        NIMCP_LOGGING_ERROR("Failed to register with cognitive hub");
        return NIMCP_ERROR_OPERATION_FAILED;
    }

    bridge->module_id = LOGIC_HUB_MODULE_ID;

    /* Subscribe to events based on configuration */
    if (bridge->config.subscribe_memory_access) {
        cognitive_hub_subscribe(hub, LOGIC_HUB_MODULE_ID,
            COG_EVENT_MEMORY_ACCESS, on_memory_access, bridge);
    }

    if (bridge->config.subscribe_attention_shift) {
        cognitive_hub_subscribe(hub, LOGIC_HUB_MODULE_ID,
            COG_EVENT_ATTENTION_SHIFT, on_attention_shift, bridge);
    }

    if (bridge->config.subscribe_learning_complete) {
        cognitive_hub_subscribe(hub, LOGIC_HUB_MODULE_ID,
            COG_EVENT_LEARNING_COMPLETE, on_learning_complete, bridge);
    }

    bridge->state.is_registered = true;
    bridge->state.is_active = true;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Connected to cognitive hub (module_id=0x%04X)", LOGIC_HUB_MODULE_ID);
    return 0;
}

int symbolic_logic_hub_bridge_connect_logic(
    symbolic_logic_hub_bridge_t* bridge,
    symbolic_logic_t* logic)
{
    NIMCP_CHECK_THROW(bridge && logic, NIMCP_ERROR_NULL_POINTER, "bridge or logic is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->logic = logic;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Connected to symbolic logic system");
    return 0;
}

int symbolic_logic_hub_bridge_disconnect(symbolic_logic_hub_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);

    if (bridge->hub && bridge->state.is_registered) {
        cognitive_hub_unregister_module(bridge->hub, bridge->module_id);
    }

    bridge->hub = NULL;
    bridge->logic = NULL;
    bridge->state.is_registered = false;
    bridge->state.is_active = false;

    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_INFO("Disconnected from cognitive hub");
    return 0;
}

/* ============================================================================
 * Event Handling Implementation
 * ============================================================================ */

int symbolic_logic_hub_bridge_handle_event(
    symbolic_logic_hub_bridge_t* bridge,
    const cognitive_event_data_t* event)
{
    NIMCP_CHECK_THROW(bridge && event, NIMCP_ERROR_NULL_POINTER, "bridge or event is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.events_received++;
    nimcp_mutex_unlock(bridge->base.mutex);

    /* Process event based on type - trigger appropriate inference */
    /* This is called by the event callbacks */

    return 0;
}

int symbolic_logic_hub_bridge_publish_inference(
    symbolic_logic_hub_bridge_t* bridge,
    const char* conclusion,
    float confidence)
{
    NIMCP_CHECK_THROW(bridge && conclusion, NIMCP_ERROR_NULL_POINTER, "bridge or conclusion is NULL");
    if (!bridge->hub || !bridge->config.publish_inference_results) return 0;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Create event data */
    cognitive_event_data_t event = {0};
    event.priority = (confidence > 0.8f) ? COG_PRIORITY_HIGH : COG_PRIORITY_NORMAL;
    event.timestamp = 0; /* Hub will set */

    /* Release lock before publishing to avoid deadlock */
    nimcp_mutex_unlock(bridge->base.mutex);

    /* Publish to hub */
    int result = cognitive_hub_publish(
        bridge->hub,
        LOGIC_HUB_MODULE_ID,
        COG_EVENT_OUTPUT_READY,
        &event);

    /* Update stats after publishing */
    nimcp_mutex_lock(bridge->base.mutex);
    if (result == 0) {
        bridge->stats.events_published++;
    }
    nimcp_mutex_unlock(bridge->base.mutex);

    return result;
}

int symbolic_logic_hub_bridge_publish_contradiction(
    symbolic_logic_hub_bridge_t* bridge,
    const char* fact1,
    const char* fact2)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (!bridge->hub || !bridge->config.publish_contradiction_found) return 0;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.contradictions_found++;

    cognitive_event_data_t event = {0};
    event.priority = COG_PRIORITY_HIGH; /* Contradictions are important */

    nimcp_mutex_unlock(bridge->base.mutex);

    /* Publish contradiction event */
    int result = cognitive_hub_publish(
        bridge->hub,
        LOGIC_HUB_MODULE_ID,
        COG_EVENT_STATE_CHANGE,
        &event);

    if (result == 0) {
        nimcp_mutex_lock(bridge->base.mutex);
        bridge->stats.events_published++;
        nimcp_mutex_unlock(bridge->base.mutex);
    }

    NIMCP_LOGGING_DEBUG("Published contradiction event");
    return result;
}

/* ============================================================================
 * Query Handling Implementation
 * ============================================================================ */

int symbolic_logic_hub_bridge_handle_query(
    symbolic_logic_hub_bridge_t* bridge,
    cognitive_query_type_t query_type,
    const void* query_data,
    void* result)
{
    NIMCP_CHECK_THROW(bridge && result, NIMCP_ERROR_NULL_POINTER, "bridge or result is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.queries_answered++;
    nimcp_mutex_unlock(bridge->base.mutex);

    switch (query_type) {
        case COG_QUERY_STATUS:
            /* Return operational status */
            *((bool*)result) = bridge->state.is_active;
            break;

        case COG_QUERY_STATE:
            /* Return current state */
            memcpy(result, &bridge->state, sizeof(symbolic_logic_hub_state_t));
            break;

        case COG_QUERY_METRICS:
            /* Return statistics */
            memcpy(result, &bridge->stats, sizeof(symbolic_logic_hub_stats_t));
            break;

        default:
            return NIMCP_ERROR_INVALID_SEQUENCE;
    }

    return 0;
}

/* ============================================================================
 * Update API Implementation
 * ============================================================================ */

int symbolic_logic_hub_bridge_update(
    symbolic_logic_hub_bridge_t* bridge,
    uint64_t delta_ms)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    (void)delta_ms;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Process any pending inference triggers */
    if (bridge->state.pending_inferences > 0 && bridge->logic) {
        /* Trigger forward chaining for pending facts */
        bridge->stats.inferences_triggered++;
        bridge->state.pending_inferences--;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int symbolic_logic_hub_bridge_force_update(symbolic_logic_hub_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    return symbolic_logic_hub_bridge_update(bridge, 0);
}

/* ============================================================================
 * State and Stats Implementation
 * ============================================================================ */

int symbolic_logic_hub_bridge_get_state(
    const symbolic_logic_hub_bridge_t* bridge,
    symbolic_logic_hub_state_t* state)
{
    NIMCP_CHECK_THROW(bridge && state, NIMCP_ERROR_NULL_POINTER, "bridge or state is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *state = bridge->state;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int symbolic_logic_hub_bridge_get_stats(
    const symbolic_logic_hub_bridge_t* bridge,
    symbolic_logic_hub_stats_t* stats)
{
    NIMCP_CHECK_THROW(bridge && stats, NIMCP_ERROR_NULL_POINTER, "bridge or stats is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int symbolic_logic_hub_bridge_reset_stats(symbolic_logic_hub_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(symbolic_logic_hub_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Bio-Async Implementation
 * ============================================================================ */

int symbolic_logic_hub_bridge_connect_bio_async(symbolic_logic_hub_bridge_t* bridge) {
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");
    if (bridge->base.bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_SYMBOLIC_LOGIC_HUB,
        .module_name = "symbolic_logic_hub_bridge",
        .inbox_capacity = 64,
        .user_data = bridge
    };

    bridge->base.bio_ctx = bio_router_register_module(&info);
    if (bridge->base.bio_ctx) {
        bridge->base.bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Connected to bio-async router");
    }

    return 0;
}

int symbolic_logic_hub_bridge_disconnect_bio_async(symbolic_logic_hub_bridge_t* bridge) {
    if (!bridge || !bridge->base.bio_async_enabled) return 0;

    if (bridge->base.bio_ctx) {
        bio_router_unregister_module(bridge->base.bio_ctx);
    }

    bridge->base.bio_ctx = NULL;
    bridge->base.bio_async_enabled = false;

    return 0;
}

bool symbolic_logic_hub_bridge_is_bio_async_connected(
    const symbolic_logic_hub_bridge_t* bridge)
{
    return bridge ? bridge->base.bio_async_enabled : false;
}

/* ============================================================================
 * Event Callbacks
 * ============================================================================ */

static int on_memory_access(const cognitive_event_data_t* event, void* user_data) {
    (void)event;
    symbolic_logic_hub_bridge_t* bridge = (symbolic_logic_hub_bridge_t*)user_data;
    if (!bridge) return -1;

    /* Memory access may provide new facts to incorporate */
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.events_received++;
    bridge->state.pending_inferences++;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Memory access event received");
    return 0;
}

static int on_attention_shift(const cognitive_event_data_t* event, void* user_data) {
    (void)event;
    symbolic_logic_hub_bridge_t* bridge = (symbolic_logic_hub_bridge_t*)user_data;
    if (!bridge) return -1;

    /* Attention shift may redirect inference focus */
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.events_received++;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Attention shift event received");
    return 0;
}

static int on_learning_complete(const cognitive_event_data_t* event, void* user_data) {
    (void)event;
    symbolic_logic_hub_bridge_t* bridge = (symbolic_logic_hub_bridge_t*)user_data;
    if (!bridge) return -1;

    /* Learning completion may trigger knowledge consolidation */
    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.events_received++;
    nimcp_mutex_unlock(bridge->base.mutex);

    NIMCP_LOGGING_DEBUG("Learning complete event received");
    return 0;
}
