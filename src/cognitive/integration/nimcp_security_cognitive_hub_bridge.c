/**
 * @file nimcp_security_cognitive_hub_bridge.c
 * @brief Implementation of Security-Cognitive Hub Bridge
 * @version 1.0.0
 * @date 2026-01-10
 */

/* Include actual type definitions BEFORE the bridge header to avoid forward decl conflicts */
#include "security/nimcp_security_orchestrator.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/integration/nimcp_cognitive_integration_hub.h"
#include "cognitive/integration/nimcp_cognitive_event_types.h"
#include "cognitive/integration/nimcp_security_cognitive_hub_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/error/nimcp_error_codes.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdio.h>

/* ============================================================================
 * INTERNAL STRUCTURES
 * ============================================================================ */

/**
 * @brief Internal bridge structure
 */
struct security_cognitive_bridge_struct {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */

    /* Configuration */
    security_cognitive_config_t config;

    /* State */
    security_cognitive_state_t state;
    bool security_connected;
    bool cognitive_connected;
    bool in_lockdown;

    /* Connections */
    security_orchestrator_t security_orch;
    cognitive_integration_hub_t cognitive_hub;
    uint32_t security_bridge_id;    /* Bridge ID in security orchestrator */
    uint32_t cognitive_module_id;   /* Module ID in cognitive hub */

    /* Statistics */
    security_cognitive_stats_t stats;

    /* Timestamps */
    uint64_t create_time;
};

/* ============================================================================
 * HELPER MACROS
 * ============================================================================ */

#define BRIDGE_LOCK(b) nimcp_mutex_lock((b)->base.mutex)
#define BRIDGE_UNLOCK(b) nimcp_mutex_unlock((b)->base.mutex)

/* ============================================================================
 * INTERNAL HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Get current timestamp
 */
static uint64_t get_timestamp_us(void)
{
    return nimcp_time_monotonic_us();
}

/**
 * @brief Security event callback from orchestrator
 */
static int security_event_callback(
    const security_event_data_t* event,
    void* user_data
)
{
    security_cognitive_bridge_t bridge = (security_cognitive_bridge_t)user_data;
    if (!bridge || !event) return -1;

    /* Translate security event to cognitive domain if configured */
    if (bridge->config.translate_security_to_cognitive && bridge->cognitive_connected) {
        security_cognitive_translate_security_event(bridge, event);
    }

    return 0;
}

/**
 * @brief Cognitive event callback from hub
 */
static int cognitive_event_callback(
    const cognitive_event_data_t* event,
    void* user_data
)
{
    security_cognitive_bridge_t bridge = (security_cognitive_bridge_t)user_data;
    if (!bridge || !event) return -1;

    /* Translate cognitive event to security domain if configured */
    if (bridge->config.translate_cognitive_to_security && bridge->security_connected) {
        /* Check if event indicates anomaly based on priority */
        float anomaly_score = 0.0f;
        bool should_report = false;

        /* Derive anomaly score from event priority - high priority may indicate anomalies */
        switch (event->event_type) {
            case COG_EVENT_MEMORY_ACCESS:
                if (bridge->config.report_memory_anomalies) {
                    /* Critical/high priority memory access may indicate anomaly */
                    if (event->priority >= COG_PRIORITY_HIGH) {
                        anomaly_score = (event->priority == COG_PRIORITY_CRITICAL) ? 0.9f : 0.7f;
                        should_report = true;
                    }
                }
                break;

            case COG_EVENT_DECISION_MADE:
            case COG_EVENT_STATE_CHANGE:
                if (bridge->config.report_reasoning_anomalies) {
                    /* Critical decisions may need security review */
                    if (event->priority >= COG_PRIORITY_HIGH) {
                        anomaly_score = (event->priority == COG_PRIORITY_CRITICAL) ? 0.8f : 0.6f;
                        should_report = true;
                    }
                }
                break;

            default:
                break;
        }

        /* Report if above threshold */
        if (should_report && anomaly_score > bridge->config.cognitive_anomaly_threshold) {
            security_cognitive_translate_cognitive_event(
                bridge,
                event->event_type,
                COG_CATEGORY_SELF,  /* Default category since event doesn't have this field */
                anomaly_score,
                "Cognitive anomaly detected"
            );
        }
    }

    return 0;
}

/**
 * @brief Query handler for security queries from cognitive modules
 */
static int security_query_handler(
    const cognitive_query_t* query,
    cognitive_query_result_t* result,
    void* context
)
{
    security_cognitive_bridge_t bridge = (security_cognitive_bridge_t)context;
    if (!bridge || !query || !result) return -1;

    BRIDGE_LOCK(bridge);
    bridge->stats.security_queries_handled++;

    /* Static result data for each query type */
    static struct {
        bool is_active;
        uint32_t state_code;
    } status_result;

    static struct {
        float threat_level;
        bool in_lockdown;
    } state_result;

    static struct {
        uint64_t threats_detected;
        uint64_t attacks_blocked;
    } metrics_result;

    /* Handle different query types */
    switch (query->query_type) {
        case COG_QUERY_STATUS: {
            /* Return security operational status */
            security_orch_state_t orch_state;
            if (bridge->security_orch) {
                security_orch_get_state(bridge->security_orch, &orch_state);
                result->status = 0;
                status_result.is_active = (orch_state != SEC_ORCH_STATE_ERROR);
                status_result.state_code = (uint32_t)orch_state;
                result->result_data = &status_result;
                result->result_size = sizeof(status_result);
            }
            break;
        }

        case COG_QUERY_STATE: {
            /* Return security threat state */
            if (bridge->security_orch) {
                float threat_level;
                security_orch_get_threat_level(bridge->security_orch, &threat_level);
                result->status = 0;
                state_result.threat_level = threat_level;
                state_result.in_lockdown = bridge->in_lockdown;
                result->result_data = &state_result;
                result->result_size = sizeof(state_result);
            }
            break;
        }

        case COG_QUERY_METRICS: {
            /* Return security statistics */
            if (bridge->security_orch) {
                security_orch_stats_t stats;
                security_orch_get_stats(bridge->security_orch, &stats);
                result->status = 0;
                metrics_result.threats_detected = stats.threats_detected;
                metrics_result.attacks_blocked = stats.attacks_blocked;
                result->result_data = &metrics_result;
                result->result_size = sizeof(metrics_result);
            }
            break;
        }

        default:
            result->status = -1;
            result->result_data = NULL;
            result->result_size = 0;
            break;
    }

    BRIDGE_UNLOCK(bridge);
    return result->status;
}

/* ============================================================================
 * PUBLIC API IMPLEMENTATION
 * ============================================================================ */

int security_cognitive_default_config(security_cognitive_config_t* config)
{
    NIMCP_CHECK_THROW(config, NIMCP_ERROR_NULL_POINTER, "config is NULL");

    memset(config, 0, sizeof(*config));

    /* Event translation - enabled by default */
    config->translate_security_to_cognitive = true;
    config->translate_cognitive_to_security = true;
    config->enable_async_translation = true;

    /* Response coordination - enabled by default */
    config->coordinate_lockdown = true;
    config->protect_memory_on_attack = true;
    config->restrict_reasoning_on_threat = true;

    /* Query handling - enabled by default */
    config->enable_security_queries = true;
    config->enable_cognitive_queries = true;

    /* Thresholds */
    config->attention_shift_threshold = 0.5f;
    config->reasoning_restrict_threshold = 0.7f;
    config->lockdown_notify_threshold = 0.9f;

    /* Anomaly detection */
    config->cognitive_anomaly_threshold = 0.6f;
    config->report_memory_anomalies = true;
    config->report_reasoning_anomalies = true;

    return 0;
}

security_cognitive_bridge_t security_cognitive_bridge_create(
    const security_cognitive_config_t* config
)
{
    security_cognitive_bridge_t bridge = nimcp_calloc(1, sizeof(*bridge));
    if (!bridge) return NULL;

    /* Apply configuration */
    if (config) {
        memcpy(&bridge->config, config, sizeof(bridge->config));
    } else {
        security_cognitive_default_config(&bridge->config);
    }

    /* Initialize base bridge infrastructure */
    if (bridge_base_init(&bridge->base, 0, "security_cognitive") != 0) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->state = SEC_COG_STATE_DISCONNECTED;
    bridge->create_time = get_timestamp_us();

    return bridge;
}

void security_cognitive_bridge_destroy(security_cognitive_bridge_t bridge)
{
    if (!bridge) return;

    /* Disconnect from both systems */
    security_cognitive_disconnect_security(bridge);
    security_cognitive_disconnect_cognitive(bridge);

    /* Cleanup base bridge infrastructure */
    bridge_base_cleanup(&bridge->base);

    nimcp_free(bridge);
}

int security_cognitive_bridge_reset(security_cognitive_bridge_t bridge)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    BRIDGE_LOCK(bridge);

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(bridge->stats));

    /* Reset state */
    bridge->in_lockdown = false;
    if (bridge->security_connected && bridge->cognitive_connected) {
        bridge->state = SEC_COG_STATE_CONNECTED;
    } else if (bridge->security_connected || bridge->cognitive_connected) {
        bridge->state = SEC_COG_STATE_DISCONNECTED;
    }

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_connect_security(
    security_cognitive_bridge_t bridge,
    security_orchestrator_t orchestrator
)
{
    NIMCP_CHECK_THROW(bridge && orchestrator, NIMCP_ERROR_NULL_POINTER, "bridge or orchestrator is NULL");

    BRIDGE_LOCK(bridge);

    if (bridge->security_connected) {
        BRIDGE_UNLOCK(bridge);
        return NIMCP_ERROR_ALREADY_EXISTS;
    }

    /* Register as a bridge with the security orchestrator */
    uint32_t bridge_id = 0;
    int result = security_orch_register_bridge(
        orchestrator,
        SEC_BRIDGE_IMMUNE,  /* Use immune type for integration bridges */
        "security_cognitive_bridge",
        bridge,
        bridge,
        &bridge_id
    );

    if (result != 0) {
        BRIDGE_UNLOCK(bridge);
        return result;
    }

    /* Subscribe to all threat events */
    security_orch_subscribe(orchestrator, bridge_id, SEC_EVENT_THREAT_DETECTED,
                           security_event_callback, bridge);
    security_orch_subscribe(orchestrator, bridge_id, SEC_EVENT_THREAT_ESCALATED,
                           security_event_callback, bridge);
    security_orch_subscribe(orchestrator, bridge_id, SEC_EVENT_ATTACK_STARTED,
                           security_event_callback, bridge);
    security_orch_subscribe(orchestrator, bridge_id, SEC_EVENT_ATTACK_BLOCKED,
                           security_event_callback, bridge);
    security_orch_subscribe(orchestrator, bridge_id, SEC_EVENT_ORCHESTRATOR_STATE,
                           security_event_callback, bridge);

    bridge->security_orch = orchestrator;
    bridge->security_bridge_id = bridge_id;
    bridge->security_connected = true;

    /* Update state */
    if (bridge->cognitive_connected) {
        bridge->state = SEC_COG_STATE_CONNECTED;
    }

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_connect_cognitive(
    security_cognitive_bridge_t bridge,
    cognitive_integration_hub_t cognitive_hub
)
{
    NIMCP_CHECK_THROW(bridge && cognitive_hub, NIMCP_ERROR_NULL_POINTER, "bridge or cognitive_hub is NULL");

    BRIDGE_LOCK(bridge);

    if (bridge->cognitive_connected) {
        BRIDGE_UNLOCK(bridge);
        return NIMCP_ERROR_ALREADY_EXISTS;
    }

    /* Register as a cognitive module */
    int result = cognitive_hub_register_module(
        cognitive_hub,
        SEC_COG_MODULE_ID,
        COG_CATEGORY_SELF,  /* Security as self-monitoring */
        SEC_COG_MODULE_NAME,
        bridge
    );

    if (result != 0) {
        BRIDGE_UNLOCK(bridge);
        return result;
    }

    /* Register query handler if enabled */
    if (bridge->config.enable_security_queries) {
        cognitive_hub_register_query_handler(
            cognitive_hub,
            SEC_COG_MODULE_ID,
            security_query_handler
        );
    }

    /* Subscribe to cognitive events we care about */
    cognitive_hub_subscribe(cognitive_hub, SEC_COG_MODULE_ID,
                           COG_EVENT_MEMORY_ACCESS, cognitive_event_callback, bridge);
    cognitive_hub_subscribe(cognitive_hub, SEC_COG_MODULE_ID,
                           COG_EVENT_DECISION_MADE, cognitive_event_callback, bridge);
    cognitive_hub_subscribe(cognitive_hub, SEC_COG_MODULE_ID,
                           COG_EVENT_STATE_CHANGE, cognitive_event_callback, bridge);

    bridge->cognitive_hub = cognitive_hub;
    bridge->cognitive_module_id = SEC_COG_MODULE_ID;
    bridge->cognitive_connected = true;

    /* Update state */
    if (bridge->security_connected) {
        bridge->state = SEC_COG_STATE_CONNECTED;
    }

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_disconnect_security(security_cognitive_bridge_t bridge)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    BRIDGE_LOCK(bridge);

    if (!bridge->security_connected) {
        BRIDGE_UNLOCK(bridge);
        return 0;
    }

    /* Unregister from security orchestrator */
    if (bridge->security_orch && bridge->security_bridge_id > 0) {
        security_orch_unregister_bridge(bridge->security_orch, bridge->security_bridge_id);
    }

    bridge->security_orch = NULL;
    bridge->security_bridge_id = 0;
    bridge->security_connected = false;

    /* Update state */
    bridge->state = SEC_COG_STATE_DISCONNECTED;

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_disconnect_cognitive(security_cognitive_bridge_t bridge)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    BRIDGE_LOCK(bridge);

    if (!bridge->cognitive_connected) {
        BRIDGE_UNLOCK(bridge);
        return 0;
    }

    /* Unregister from cognitive hub */
    if (bridge->cognitive_hub) {
        cognitive_hub_unregister_module(bridge->cognitive_hub, bridge->cognitive_module_id);
    }

    bridge->cognitive_hub = NULL;
    bridge->cognitive_module_id = 0;
    bridge->cognitive_connected = false;

    /* Update state */
    bridge->state = SEC_COG_STATE_DISCONNECTED;

    BRIDGE_UNLOCK(bridge);
    return 0;
}

bool security_cognitive_is_connected(security_cognitive_bridge_t bridge)
{
    if (!bridge) return false;

    BRIDGE_LOCK(bridge);
    bool connected = bridge->security_connected && bridge->cognitive_connected;
    BRIDGE_UNLOCK(bridge);

    return connected;
}

int security_cognitive_translate_security_event(
    security_cognitive_bridge_t bridge,
    const security_event_data_t* security_event
)
{
    NIMCP_CHECK_THROW(bridge && security_event, NIMCP_ERROR_NULL_POINTER, "bridge or security_event is NULL");

    BRIDGE_LOCK(bridge);

    if (!bridge->cognitive_connected || !bridge->cognitive_hub) {
        BRIDGE_UNLOCK(bridge);
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    uint64_t start_time = get_timestamp_us();

    /* Map security event to cognitive event */
    cognitive_event_data_t cog_event = {};
    cog_event.source_module_id = bridge->cognitive_module_id;
    cog_event.timestamp = security_event->timestamp;

    bool should_publish = true;

    switch (security_event->event_type) {
        case SEC_EVENT_THREAT_DETECTED:
        case SEC_EVENT_THREAT_ESCALATED:
            /* Map to attention shift if above threshold */
            if (security_event->threat.threat_level >= bridge->config.attention_shift_threshold) {
                cog_event.event_type = COG_EVENT_ATTENTION_SHIFT;
                cog_event.priority = (security_event->severity >= SEC_SEVERITY_HIGH) ?
                    COG_PRIORITY_CRITICAL : COG_PRIORITY_HIGH;
                bridge->stats.attention_shifts_triggered++;
            } else {
                cog_event.event_type = COG_EVENT_STATE_CHANGE;
                cog_event.priority = COG_PRIORITY_NORMAL;
            }
            break;

        case SEC_EVENT_ATTACK_STARTED:
        case SEC_EVENT_ATTACK_ONGOING:
            cog_event.event_type = COG_EVENT_ATTENTION_SHIFT;
            cog_event.priority = COG_PRIORITY_CRITICAL;
            bridge->stats.attention_shifts_triggered++;

            /* Trigger memory protection if configured */
            if (bridge->config.protect_memory_on_attack) {
                security_cognitive_protect_memory(bridge, security_event->threat.threat_level);
            }
            break;

        case SEC_EVENT_MEMORY_CORRUPTION:
            cog_event.event_type = COG_EVENT_MEMORY_ACCESS;
            cog_event.priority = COG_PRIORITY_CRITICAL;
            break;

        case SEC_EVENT_ORCHESTRATOR_STATE:
            /* Handle lockdown state changes */
            if (security_event->custom.code == SEC_ORCH_STATE_LOCKDOWN) {
                if (bridge->config.coordinate_lockdown) {
                    security_cognitive_coordinate_lockdown(bridge, "Security lockdown");
                }
            } else if (security_event->custom.code == SEC_ORCH_STATE_RECOVERY) {
                security_cognitive_release_lockdown(bridge);
            }
            cog_event.event_type = COG_EVENT_STATE_CHANGE;
            cog_event.priority = COG_PRIORITY_HIGH;
            break;

        default:
            should_publish = false;
            break;
    }

    if (should_publish) {
        cognitive_hub_publish(
            bridge->cognitive_hub,
            bridge->cognitive_module_id,
            cog_event.event_type,
            &cog_event
        );
        bridge->stats.security_events_translated++;
    }

    /* Update timing stats */
    uint64_t elapsed = get_timestamp_us() - start_time;
    bridge->stats.avg_translation_latency_us =
        (bridge->stats.avg_translation_latency_us *
         bridge->stats.security_events_translated + elapsed) /
        (bridge->stats.security_events_translated + 1);

    bridge->state = SEC_COG_STATE_ACTIVE;

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_translate_cognitive_event(
    security_cognitive_bridge_t bridge,
    cognitive_event_type_t cognitive_event_type,
    cognitive_category_t category,
    float anomaly_score,
    const char* description
)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    BRIDGE_LOCK(bridge);

    if (!bridge->security_connected || !bridge->security_orch) {
        BRIDGE_UNLOCK(bridge);
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    /* Map cognitive event to security event */
    security_event_type_t sec_event_type;
    security_severity_t severity;

    switch (cognitive_event_type) {
        case COG_EVENT_MEMORY_ACCESS:
            sec_event_type = SEC_EVENT_MEMORY_CORRUPTION;
            break;

        case COG_EVENT_DECISION_MADE:
            sec_event_type = SEC_EVENT_STRATEGY_MANIPULATION;
            break;

        case COG_EVENT_STATE_CHANGE:
            sec_event_type = SEC_EVENT_THREAT_DETECTED;
            break;

        default:
            sec_event_type = SEC_EVENT_THREAT_DETECTED;
            break;
    }

    /* Map anomaly score to severity */
    if (anomaly_score >= 0.9f) {
        severity = SEC_SEVERITY_CRITICAL;
    } else if (anomaly_score >= 0.7f) {
        severity = SEC_SEVERITY_HIGH;
    } else if (anomaly_score >= 0.5f) {
        severity = SEC_SEVERITY_MEDIUM;
    } else {
        severity = SEC_SEVERITY_LOW;
    }

    /* Report to security orchestrator */
    security_orch_report_threat(
        bridge->security_orch,
        bridge->security_bridge_id,
        anomaly_score,
        severity,
        description ? description : "Cognitive anomaly"
    );

    bridge->stats.cognitive_events_translated++;
    bridge->state = SEC_COG_STATE_ACTIVE;

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_coordinate_lockdown(
    security_cognitive_bridge_t bridge,
    const char* reason
)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    BRIDGE_LOCK(bridge);

    if (!bridge->cognitive_connected || !bridge->cognitive_hub) {
        BRIDGE_UNLOCK(bridge);
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    /* Publish lockdown event to cognitive hub */
    cognitive_event_data_t event = {};
    event.event_type = COG_EVENT_STATE_CHANGE;
    event.source_module_id = bridge->cognitive_module_id;
    event.priority = COG_PRIORITY_CRITICAL;
    event.timestamp = get_timestamp_us();

    cognitive_hub_publish(
        bridge->cognitive_hub,
        bridge->cognitive_module_id,
        COG_EVENT_STATE_CHANGE,
        &event
    );

    bridge->in_lockdown = true;
    bridge->state = SEC_COG_STATE_LOCKDOWN;
    bridge->stats.lockdowns_coordinated++;

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_release_lockdown(security_cognitive_bridge_t bridge)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    BRIDGE_LOCK(bridge);

    bridge->in_lockdown = false;
    bridge->state = SEC_COG_STATE_CONNECTED;

    /* Publish recovery event */
    if (bridge->cognitive_connected && bridge->cognitive_hub) {
        cognitive_event_data_t event = {};
        event.event_type = COG_EVENT_STATE_CHANGE;
        event.source_module_id = bridge->cognitive_module_id;
        event.priority = COG_PRIORITY_NORMAL;
        event.timestamp = get_timestamp_us();

        cognitive_hub_publish(
            bridge->cognitive_hub,
            bridge->cognitive_module_id,
            COG_EVENT_STATE_CHANGE,
            &event
        );
    }

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_protect_memory(
    security_cognitive_bridge_t bridge,
    float threat_level
)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    BRIDGE_LOCK(bridge);

    if (!bridge->cognitive_connected || !bridge->cognitive_hub) {
        BRIDGE_UNLOCK(bridge);
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    /* Publish memory consolidation event */
    cognitive_event_data_t event = {};
    event.event_type = COG_EVENT_CONSOLIDATION;
    event.source_module_id = bridge->cognitive_module_id;
    event.priority = COG_PRIORITY_CRITICAL;
    event.timestamp = get_timestamp_us();

    cognitive_hub_publish(
        bridge->cognitive_hub,
        bridge->cognitive_module_id,
        COG_EVENT_CONSOLIDATION,
        &event
    );

    bridge->stats.memory_protections_triggered++;

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_shift_attention(
    security_cognitive_bridge_t bridge,
    uint32_t priority,
    cognitive_category_t target_category
)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    BRIDGE_LOCK(bridge);

    if (!bridge->cognitive_connected || !bridge->cognitive_hub) {
        BRIDGE_UNLOCK(bridge);
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    /* Publish attention shift event */
    static cognitive_category_t attention_target;  /* Static to ensure pointer validity */
    attention_target = target_category;

    cognitive_event_data_t event = {};
    event.event_type = COG_EVENT_ATTENTION_SHIFT;
    event.source_module_id = bridge->cognitive_module_id;
    event.priority = (cognitive_event_priority_t)priority;
    event.timestamp = get_timestamp_us();
    event.payload = &attention_target;
    event.payload_size = sizeof(attention_target);

    cognitive_hub_publish(
        bridge->cognitive_hub,
        bridge->cognitive_module_id,
        COG_EVENT_ATTENTION_SHIFT,
        &event
    );

    bridge->stats.attention_shifts_triggered++;

    BRIDGE_UNLOCK(bridge);
    return 0;
}

int security_cognitive_query_cognitive(
    security_cognitive_bridge_t bridge,
    uint32_t target_module,
    uint32_t query_type,
    void* result_out
)
{
    NIMCP_CHECK_THROW(bridge && result_out, NIMCP_ERROR_NULL_POINTER, "bridge or result_out is NULL");

    BRIDGE_LOCK(bridge);

    if (!bridge->cognitive_connected || !bridge->cognitive_hub) {
        BRIDGE_UNLOCK(bridge);
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    cognitive_query_t query = {
        .query_type = (cognitive_query_type_t)query_type,
        .query_params = NULL,
        .params_size = 0
    };

    cognitive_query_result_t result;
    int status = cognitive_hub_query_module(
        bridge->cognitive_hub,
        bridge->cognitive_module_id,
        target_module,
        &query,
        &result
    );

    if (status == 0) {
        memcpy(result_out, &result, sizeof(result));
        bridge->stats.cognitive_queries_made++;
    } else {
        bridge->stats.query_failures++;
    }

    BRIDGE_UNLOCK(bridge);
    return status;
}

int security_cognitive_get_security_assessment(
    security_cognitive_bridge_t bridge,
    security_threat_assessment_t* assessment_out
)
{
    NIMCP_CHECK_THROW(bridge && assessment_out, NIMCP_ERROR_NULL_POINTER, "bridge or assessment_out is NULL");

    BRIDGE_LOCK(bridge);

    if (!bridge->security_connected || !bridge->security_orch) {
        BRIDGE_UNLOCK(bridge);
        return NIMCP_ERROR_NOT_INITIALIZED;
    }

    int status = security_orch_get_threat_assessment(bridge->security_orch, assessment_out);

    BRIDGE_UNLOCK(bridge);
    return status;
}

int security_cognitive_get_state(
    security_cognitive_bridge_t bridge,
    security_cognitive_state_t* state_out
)
{
    NIMCP_CHECK_THROW(bridge && state_out, NIMCP_ERROR_NULL_POINTER, "bridge or state_out is NULL");

    BRIDGE_LOCK(bridge);
    *state_out = bridge->state;
    BRIDGE_UNLOCK(bridge);

    return 0;
}

int security_cognitive_get_stats(
    security_cognitive_bridge_t bridge,
    security_cognitive_stats_t* stats_out
)
{
    NIMCP_CHECK_THROW(bridge && stats_out, NIMCP_ERROR_NULL_POINTER, "bridge or stats_out is NULL");

    BRIDGE_LOCK(bridge);
    memcpy(stats_out, &bridge->stats, sizeof(*stats_out));
    stats_out->uptime_us = get_timestamp_us() - bridge->create_time;
    BRIDGE_UNLOCK(bridge);

    return 0;
}

int security_cognitive_reset_stats(security_cognitive_bridge_t bridge)
{
    NIMCP_CHECK_THROW(bridge, NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

    BRIDGE_LOCK(bridge);
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    BRIDGE_UNLOCK(bridge);

    return 0;
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

const char* security_cognitive_state_name(security_cognitive_state_t state)
{
    static const char* names[] = {
        [SEC_COG_STATE_UNINITIALIZED] = "Uninitialized",
        [SEC_COG_STATE_DISCONNECTED] = "Disconnected",
        [SEC_COG_STATE_CONNECTED] = "Connected",
        [SEC_COG_STATE_ACTIVE] = "Active",
        [SEC_COG_STATE_COORDINATING] = "Coordinating",
        [SEC_COG_STATE_LOCKDOWN] = "Lockdown",
        [SEC_COG_STATE_ERROR] = "Error"
    };

    if (state > SEC_COG_STATE_ERROR) return "Invalid";
    return names[state];
}

void security_cognitive_print_summary(security_cognitive_bridge_t bridge)
{
    if (!bridge) {
        printf("Security-Cognitive Bridge: NULL\n");
        return;
    }

    BRIDGE_LOCK(bridge);

    printf("\n=== Security-Cognitive Hub Bridge Summary ===\n");
    printf("State: %s\n", security_cognitive_state_name(bridge->state));
    printf("Security Connected: %s\n", bridge->security_connected ? "Yes" : "No");
    printf("Cognitive Connected: %s\n", bridge->cognitive_connected ? "Yes" : "No");
    printf("In Lockdown: %s\n", bridge->in_lockdown ? "Yes" : "No");

    printf("\nConfiguration:\n");
    printf("  Translate Security->Cognitive: %s\n",
           bridge->config.translate_security_to_cognitive ? "Yes" : "No");
    printf("  Translate Cognitive->Security: %s\n",
           bridge->config.translate_cognitive_to_security ? "Yes" : "No");
    printf("  Coordinate Lockdown: %s\n",
           bridge->config.coordinate_lockdown ? "Yes" : "No");

    BRIDGE_UNLOCK(bridge);
}

void security_cognitive_print_stats(const security_cognitive_stats_t* stats)
{
    if (!stats) {
        printf("Statistics: NULL\n");
        return;
    }

    printf("\n=== Security-Cognitive Bridge Statistics ===\n");
    printf("Event Translation:\n");
    printf("  Security Events Translated: %lu\n",
           (unsigned long)stats->security_events_translated);
    printf("  Cognitive Events Translated: %lu\n",
           (unsigned long)stats->cognitive_events_translated);
    printf("  Events Dropped: %lu\n",
           (unsigned long)stats->events_dropped);

    printf("\nQuery Handling:\n");
    printf("  Security Queries Handled: %lu\n",
           (unsigned long)stats->security_queries_handled);
    printf("  Cognitive Queries Made: %lu\n",
           (unsigned long)stats->cognitive_queries_made);
    printf("  Query Failures: %lu\n",
           (unsigned long)stats->query_failures);

    printf("\nCoordination:\n");
    printf("  Lockdowns Coordinated: %u\n", stats->lockdowns_coordinated);
    printf("  Attention Shifts Triggered: %u\n", stats->attention_shifts_triggered);
    printf("  Memory Protections Triggered: %u\n", stats->memory_protections_triggered);

    printf("\nTiming:\n");
    printf("  Avg Translation Latency: %lu us\n",
           (unsigned long)stats->avg_translation_latency_us);
    printf("  Uptime: %lu us\n",
           (unsigned long)stats->uptime_us);
}
