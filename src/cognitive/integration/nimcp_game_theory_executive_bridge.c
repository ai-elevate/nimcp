/**
 * @file nimcp_game_theory_executive_bridge.c
 * @brief Game Theory-Executive Integration Bridge Implementation
 * @version 1.0.0
 * @date 2026-01-08
 *
 * WHAT: Bridge connecting game theory to executive function via cognitive hub
 * WHY:  Enable strategic reasoning to inform executive decisions
 * HOW:  Implements event subscription, publication, and query handling
 *
 * BIOLOGICAL BASIS:
 * - Models prefrontal-parietal integration for strategic planning
 * - Dorsolateral PFC (executive) integrates input from strategic reasoning
 * - Anterior cingulate cortex monitors decision conflicts
 *
 * @author NIMCP Development Team
 */

#include "cognitive/integration/nimcp_game_theory_executive_bridge.h"
#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/integration/nimcp_cognitive_integration_hub.h"
#include "cognitive/integration/nimcp_cognitive_event_types.h"
#include "utils/thread/nimcp_thread.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define GT_EXEC_BRIDGE_NAME "GameTheoryExecutive"

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Internal game theory-executive bridge structure
 */
struct game_theory_executive_bridge {
    bridge_base_t base;                       /**< MUST be first: base bridge infrastructure */
    game_theory_executive_config_t config;    /**< Bridge configuration */
    cognitive_integration_hub_t hub;          /**< Connected cognitive hub */

    /* State */
    bool initialized;                         /**< Initialization flag */
    bool connected;                           /**< Connection status */
    game_theory_executive_state_t state;      /**< Current operational state */

    /* Pending recommendation */
    gt_strategic_recommendation_t pending_recommendation;
    bool has_pending_recommendation;

    /* Analysis state */
    float* current_utilities;                 /**< Current utility matrix */
    uint32_t current_num_actions;             /**< Current number of actions */
    uint32_t current_num_outcomes;            /**< Current number of outcomes */

    /* Opponent models */
    gt_exec_opponent_model_t opponent_models[GT_EXEC_MAX_OPPONENT_MODELS];
    uint32_t num_opponent_models;

    /* Statistics */
    game_theory_executive_stats_t stats;      /**< Bridge statistics */
};

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Get current timestamp in milliseconds
 */
static uint64_t get_timestamp_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
    }
    return 0;
}

/**
 * @brief Clamp a float value to a range
 */
static float clamp_float(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief Find or create opponent model
 */
static gt_exec_opponent_model_t* find_or_create_opponent_model(
    game_theory_executive_bridge_t* bridge,
    uint32_t opponent_id
) {
    /* Search for existing model */
    for (uint32_t i = 0; i < bridge->num_opponent_models; i++) {
        if (bridge->opponent_models[i].opponent_id == opponent_id) {
            return &bridge->opponent_models[i];
        }
    }

    /* Create new model if space available */
    if (bridge->num_opponent_models < GT_EXEC_MAX_OPPONENT_MODELS) {
        gt_exec_opponent_model_t* model =
            &bridge->opponent_models[bridge->num_opponent_models++];
        memset(model, 0, sizeof(gt_exec_opponent_model_t));
        model->opponent_id = opponent_id;
        model->cooperation_tendency = 0.5f;
        model->aggression_level = 0.5f;
        model->predictability = 0.5f;
        model->num_strategies = GT_EXEC_MAX_STRATEGIES;

        /* Initialize uniform prior over strategies */
        float uniform_prob = 1.0f / (float)model->num_strategies;
        for (uint32_t i = 0; i < model->num_strategies; i++) {
            model->strategy_probs[i] = uniform_prob;
        }

        model->last_update = get_timestamp_ms();
        return model;
    }

    return NULL;
}

/* ============================================================================
 * Hub Event Callback (internal)
 * ============================================================================ */

/**
 * @brief Internal callback for cognitive hub events
 */
static int gt_exec_on_event(const cognitive_event_data_t* event, void* user_data) {
    if (!event || !user_data) {
        return -1;
    }

    game_theory_executive_bridge_t* bridge = (game_theory_executive_bridge_t*)user_data;

    nimcp_mutex_lock(bridge->base.mutex);

    /* Update statistics */
    bridge->stats.events_received++;
    bridge->stats.total_events++;
    bridge->stats.last_event_timestamp = get_timestamp_ms();

    nimcp_mutex_unlock(bridge->base.mutex);

    /* Dispatch based on event type */
    switch (event->event_type) {
        case COG_EVENT_DECISION_MADE: {
            nimcp_mutex_lock(bridge->base.mutex);
            bridge->stats.decisions_received++;

            /* Check if recommendation was followed */
            if (bridge->has_pending_recommendation && event->payload) {
                /* Simplified: assume payload contains action taken */
                const uint32_t* action = (const uint32_t*)event->payload;
                if (*action == bridge->pending_recommendation.action_index) {
                    bridge->stats.recommendations_followed++;
                } else {
                    bridge->stats.executive_overrides++;
                }
                bridge->has_pending_recommendation = false;
            }

            bridge->state = GT_EXEC_STATE_UPDATING;
            nimcp_mutex_unlock(bridge->base.mutex);
            break;
        }

        case COG_EVENT_ATTENTION_SHIFT: {
            /* Attention shift might require strategy re-evaluation */
            nimcp_mutex_lock(bridge->base.mutex);
            /* Could trigger re-analysis based on new attention focus */
            nimcp_mutex_unlock(bridge->base.mutex);
            break;
        }

        case COG_EVENT_STATE_CHANGE: {
            /* Strategy state update from game theory module */
            nimcp_mutex_lock(bridge->base.mutex);
            bridge->stats.strategic_decisions++;
            nimcp_mutex_unlock(bridge->base.mutex);
            break;
        }

        default:
            /* Unhandled event type - not an error */
            break;
    }

    return 0;
}

/**
 * @brief Internal query handler for bridge queries
 */
static int gt_exec_query_handler(
    const cognitive_query_t* query,
    cognitive_query_result_t* result,
    void* context
) {
    if (!query || !result || !context) {
        return -1;
    }

    game_theory_executive_bridge_t* bridge = (game_theory_executive_bridge_t*)context;

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->stats.queries_handled++;
    nimcp_mutex_unlock(bridge->base.mutex);

    /* Initialize result */
    result->status = 0;
    result->result_data = NULL;
    result->result_size = 0;
    result->error_message[0] = '\0';

    switch (query->query_type) {
        case COG_QUERY_STATUS: {
            nimcp_mutex_lock(bridge->base.mutex);
            game_theory_executive_state_t state = bridge->state;
            nimcp_mutex_unlock(bridge->base.mutex);

            const char* state_names[] = {
                "IDLE", "ANALYZING", "RECOMMENDING",
                "AWAITING_DECISION", "UPDATING", "ERROR"
            };
            const char* state_name = (state < 6) ? state_names[state] : "UNKNOWN";

            size_t len = strlen(state_name) + 1;
            char* status_str = nimcp_malloc(len);
            if (status_str) {
                strncpy(status_str, state_name, len);
                result->result_data = status_str;
                result->result_size = len;
            }
            break;
        }

        case COG_QUERY_STATE: {
            nimcp_mutex_lock(bridge->base.mutex);
            gt_strategic_recommendation_t* rec = NULL;
            if (bridge->has_pending_recommendation) {
                rec = nimcp_malloc(sizeof(gt_strategic_recommendation_t));
                if (rec) {
                    *rec = bridge->pending_recommendation;
                }
            }
            nimcp_mutex_unlock(bridge->base.mutex);

            if (rec) {
                result->result_data = rec;
                result->result_size = sizeof(gt_strategic_recommendation_t);
            } else {
                strncpy(result->error_message, "No pending recommendation",
                        sizeof(result->error_message) - 1);
            }
            break;
        }

        case COG_QUERY_METRICS: {
            game_theory_executive_stats_t* stats =
                nimcp_malloc(sizeof(game_theory_executive_stats_t));
            if (stats) {
                nimcp_mutex_lock(bridge->base.mutex);
                *stats = bridge->stats;
                nimcp_mutex_unlock(bridge->base.mutex);
                result->result_data = stats;
                result->result_size = sizeof(game_theory_executive_stats_t);
            }
            break;
        }

        default:
            strncpy(result->error_message, "Unsupported query type",
                    sizeof(result->error_message) - 1);
            result->status = -1;
            break;
    }

    return result->status;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int game_theory_executive_bridge_default_config(game_theory_executive_config_t* config) {
    if (!config) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "config is NULL");

        return -1;
    }

    config->module_id = GT_EXEC_DEFAULT_MODULE_ID;
    config->enable_logging = false;
    config->strategic_weight = 0.4f;
    config->risk_assessment_weight = 0.3f;
    config->decision_integration_weight = 0.3f;
    config->max_strategies = GT_EXEC_MAX_STRATEGIES;
    config->risk_tolerance = 0.5f;
    config->time_pressure_factor = 0.5f;
    config->auto_subscribe_decision = true;
    config->auto_subscribe_attention = true;
    config->auto_subscribe_query = true;
    config->auto_subscribe_state_change = true;
    config->enable_mixed_strategies = true;
    config->enable_learning = true;
    config->enable_query_handler = true;
    config->event_buffer_size = GT_EXEC_MAX_EVENT_BUFFER;
    config->opponent_model_decay = 0.1f;

    return 0;
}

game_theory_executive_bridge_t* game_theory_executive_bridge_create(
    const game_theory_executive_config_t* config
) {
    /* Allocate bridge structure */
    game_theory_executive_bridge_t* bridge =
        (game_theory_executive_bridge_t*)nimcp_calloc(
            1, sizeof(game_theory_executive_bridge_t));
    if (!bridge) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bridge is NULL");

        return NULL;
    }

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        game_theory_executive_bridge_default_config(&bridge->config);
    }

    /* Create mutex for thread safety */
    if (bridge_base_init(&bridge->base, 0, "game_theory_executive") != 0) { nimcp_free(bridge); return NULL; }
    if (!bridge->base.mutex) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->hub = NULL;
    bridge->connected = false;
    bridge->state = GT_EXEC_STATE_IDLE;
    bridge->has_pending_recommendation = false;
    bridge->current_utilities = NULL;
    bridge->current_num_actions = 0;
    bridge->current_num_outcomes = 0;
    bridge->num_opponent_models = 0;

    /* Initialize stats */
    memset(&bridge->stats, 0, sizeof(game_theory_executive_stats_t));

    bridge->initialized = true;

    return bridge;
}

void game_theory_executive_bridge_destroy(game_theory_executive_bridge_t* bridge) {
    if (!bridge) {
        return;
    }

    /* Disconnect from hub if connected */
    if (bridge->connected) {
        game_theory_executive_bridge_disconnect(bridge);
    }

    /* Free utilities buffer if allocated */
    if (bridge->current_utilities) {
        nimcp_free(bridge->current_utilities);
        bridge->current_utilities = NULL;
    }

    /* Destroy mutex */
    if (bridge->base.mutex) {
        bridge_base_cleanup(&bridge->base);
        bridge->base.mutex = NULL;
    }

    bridge->initialized = false;

    nimcp_free(bridge);
}

/* ============================================================================
 * Connection Implementation
 * ============================================================================ */

int game_theory_executive_bridge_connect(
    game_theory_executive_bridge_t* bridge,
    cognitive_integration_hub_t hub
) {
    if (!bridge || !bridge->initialized || !hub) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Check if already connected */
    if (bridge->connected) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Store hub reference */
    bridge->hub = hub;

    nimcp_mutex_unlock(bridge->base.mutex);

    /* Register module with hub */
    int result = cognitive_hub_register_module(
        hub,
        bridge->config.module_id,
        COG_CATEGORY_REASONING,
        GT_EXEC_BRIDGE_NAME,
        bridge
    );

    if (result != 0) {
        nimcp_mutex_lock(bridge->base.mutex);
        bridge->hub = NULL;
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Subscribe to configured event types */
    if (bridge->config.auto_subscribe_decision) {
        cognitive_hub_subscribe(
            hub,
            bridge->config.module_id,
            COG_EVENT_DECISION_MADE,
            gt_exec_on_event,
            bridge
        );
    }

    if (bridge->config.auto_subscribe_attention) {
        cognitive_hub_subscribe(
            hub,
            bridge->config.module_id,
            COG_EVENT_ATTENTION_SHIFT,
            gt_exec_on_event,
            bridge
        );
    }

    if (bridge->config.auto_subscribe_state_change) {
        cognitive_hub_subscribe(
            hub,
            bridge->config.module_id,
            COG_EVENT_STATE_CHANGE,
            gt_exec_on_event,
            bridge
        );
    }

    /* Register query handler if enabled */
    if (bridge->config.enable_query_handler) {
        cognitive_hub_register_query_handler(
            hub,
            bridge->config.module_id,
            gt_exec_query_handler
        );
    }

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->connected = true;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int game_theory_executive_bridge_disconnect(game_theory_executive_bridge_t* bridge) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->connected || !bridge->hub) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    cognitive_integration_hub_t hub = bridge->hub;
    uint32_t module_id = bridge->config.module_id;

    nimcp_mutex_unlock(bridge->base.mutex);

    /* Unsubscribe from events */
    if (bridge->config.auto_subscribe_decision) {
        cognitive_hub_unsubscribe(hub, module_id, COG_EVENT_DECISION_MADE);
    }
    if (bridge->config.auto_subscribe_attention) {
        cognitive_hub_unsubscribe(hub, module_id, COG_EVENT_ATTENTION_SHIFT);
    }
    if (bridge->config.auto_subscribe_state_change) {
        cognitive_hub_unsubscribe(hub, module_id, COG_EVENT_STATE_CHANGE);
    }

    /* Unregister module from hub */
    cognitive_hub_unregister_module(hub, module_id);

    nimcp_mutex_lock(bridge->base.mutex);
    bridge->hub = NULL;
    bridge->connected = false;
    bridge->state = GT_EXEC_STATE_IDLE;
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

bool game_theory_executive_bridge_is_connected(
    const game_theory_executive_bridge_t* bridge
) {
    if (!bridge || !bridge->initialized) {
        return false;
    }

    /* Cast away const for mutex lock */
    game_theory_executive_bridge_t* mutable_bridge =
        (game_theory_executive_bridge_t*)bridge;

    nimcp_mutex_lock(mutable_bridge->base.mutex);
    bool connected = bridge->connected;
    nimcp_mutex_unlock(mutable_bridge->base.mutex);

    return connected;
}

/* ============================================================================
 * Strategy API Implementation
 * ============================================================================ */

int game_theory_executive_analyze_options(
    game_theory_executive_bridge_t* bridge,
    uint32_t num_actions,
    const float* utilities,
    uint32_t num_outcomes
) {
    if (!bridge || !bridge->initialized || !utilities) {
        return -1;
    }

    if (num_actions == 0 || num_outcomes == 0) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->state = GT_EXEC_STATE_ANALYZING;

    /* Free existing utilities if any */
    if (bridge->current_utilities) {
        nimcp_free(bridge->current_utilities);
    }

    /* Allocate and copy utilities */
    size_t utilities_size = num_actions * num_outcomes * sizeof(float);
    bridge->current_utilities = nimcp_malloc(utilities_size);
    if (!bridge->current_utilities) {
        bridge->state = GT_EXEC_STATE_ERROR;
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    memcpy(bridge->current_utilities, utilities, utilities_size);
    bridge->current_num_actions = num_actions;
    bridge->current_num_outcomes = num_outcomes;

    bridge->stats.strategies_analyzed += num_actions;

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int game_theory_executive_get_recommendation(
    game_theory_executive_bridge_t* bridge,
    gt_strategic_recommendation_t* recommendation_out
) {
    if (!bridge || !bridge->initialized || !recommendation_out) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->current_utilities || bridge->current_num_actions == 0) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    bridge->state = GT_EXEC_STATE_RECOMMENDING;

    /* Find best action using expected utility */
    float best_utility = -1e10f;
    uint32_t best_action = 0;
    float total_utility = 0.0f;

    for (uint32_t a = 0; a < bridge->current_num_actions; a++) {
        float expected_utility = 0.0f;
        for (uint32_t o = 0; o < bridge->current_num_outcomes; o++) {
            /* Assume uniform distribution over outcomes for simplicity */
            float prob = 1.0f / (float)bridge->current_num_outcomes;
            expected_utility += prob *
                bridge->current_utilities[a * bridge->current_num_outcomes + o];
        }

        total_utility += expected_utility;

        if (expected_utility > best_utility) {
            best_utility = expected_utility;
            best_action = a;
        }
    }

    /* Calculate confidence based on how much better best action is */
    float avg_utility = total_utility / (float)bridge->current_num_actions;
    float confidence = 0.5f;
    if (avg_utility > 0.0f) {
        confidence = clamp_float(best_utility / (avg_utility * 2.0f), 0.0f, 1.0f);
    }

    /* Estimate risk level (inverse of outcome variance) */
    float variance = 0.0f;
    for (uint32_t o = 0; o < bridge->current_num_outcomes; o++) {
        float diff = bridge->current_utilities[
            best_action * bridge->current_num_outcomes + o] - best_utility;
        variance += diff * diff;
    }
    variance /= (float)bridge->current_num_outcomes;
    float risk_level = clamp_float(variance / 10.0f, 0.0f, 1.0f);

    /* Determine strategy type */
    gt_strategy_type_t strategy_type = GT_STRATEGY_DOMINANT;
    if (bridge->config.enable_mixed_strategies && confidence < 0.7f) {
        strategy_type = GT_STRATEGY_MIXED;
    }

    /* Fill recommendation */
    static uint64_t rec_id_counter = 0;
    recommendation_out->recommendation_id = ++rec_id_counter;
    recommendation_out->strategy_type = strategy_type;
    recommendation_out->expected_utility = clamp_float(best_utility, 0.0f, 1.0f);
    recommendation_out->confidence = confidence;
    recommendation_out->risk_level = risk_level;
    recommendation_out->action_index = best_action;

    /* Store as pending */
    bridge->pending_recommendation = *recommendation_out;
    bridge->has_pending_recommendation = true;

    bridge->state = GT_EXEC_STATE_AWAITING_DECISION;
    bridge->stats.recommendations_made++;

    /* Update average expected utility */
    float n = (float)bridge->stats.recommendations_made;
    bridge->stats.avg_expected_utility =
        ((n - 1.0f) * bridge->stats.avg_expected_utility + best_utility) / n;

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int game_theory_executive_notify_outcome(
    game_theory_executive_bridge_t* bridge,
    const gt_decision_outcome_t* outcome
) {
    if (!bridge || !bridge->initialized || !outcome) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    bridge->stats.decisions_received++;

    /* Check if this matches our pending recommendation */
    if (bridge->has_pending_recommendation &&
        outcome->recommendation_id == bridge->pending_recommendation.recommendation_id) {

        if (outcome->followed_recommendation) {
            bridge->stats.recommendations_followed++;
        } else {
            bridge->stats.executive_overrides++;
        }

        /* Update average realized utility */
        float n = (float)bridge->stats.decisions_received;
        bridge->stats.avg_realized_utility =
            ((n - 1.0f) * bridge->stats.avg_realized_utility +
             outcome->outcome_utility) / n;

        /* Update accuracy rate */
        if (bridge->stats.recommendations_made > 0) {
            bridge->stats.recommendation_accuracy =
                (float)bridge->stats.recommendations_followed /
                (float)bridge->stats.recommendations_made;
        }

        bridge->has_pending_recommendation = false;
    }

    bridge->state = GT_EXEC_STATE_UPDATING;

    /* Learning: adjust strategy based on outcome */
    if (bridge->config.enable_learning) {
        /* Simple learning: not implemented in this version */
    }

    bridge->state = GT_EXEC_STATE_IDLE;

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int game_theory_executive_publish_recommendation(
    game_theory_executive_bridge_t* bridge,
    const gt_strategic_recommendation_t* recommendation
) {
    if (!bridge || !bridge->initialized || !recommendation) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    if (!bridge->connected || !bridge->hub) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    cognitive_integration_hub_t hub = bridge->hub;
    uint32_t module_id = bridge->config.module_id;

    nimcp_mutex_unlock(bridge->base.mutex);

    /* Create event data */
    cognitive_event_data_t event;
    event.event_type = COG_EVENT_OUTPUT_READY;
    event.source_module_id = module_id;
    event.timestamp = get_timestamp_ms() * 1000;
    event.priority = (recommendation->confidence > 0.7f) ?
        COG_PRIORITY_HIGH : COG_PRIORITY_NORMAL;
    event.payload = (void*)recommendation;
    event.payload_size = sizeof(gt_strategic_recommendation_t);

    /* Publish to hub */
    int result = cognitive_hub_publish(hub, module_id, COG_EVENT_OUTPUT_READY, &event);

    if (result == 0) {
        nimcp_mutex_lock(bridge->base.mutex);
        bridge->stats.events_published++;
        nimcp_mutex_unlock(bridge->base.mutex);
    }

    return result;
}

/* ============================================================================
 * Risk Assessment Implementation
 * ============================================================================ */

int game_theory_executive_request_risk_assessment(
    game_theory_executive_bridge_t* bridge,
    uint64_t action_id,
    const void* context,
    gt_exec_risk_assessment_t* assessment
) {
    (void)context;  /* Reserved for future use */

    if (!bridge || !bridge->initialized || !assessment) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Initialize assessment */
    memset(assessment, 0, sizeof(gt_exec_risk_assessment_t));
    assessment->action_id = action_id;
    assessment->timestamp = get_timestamp_ms();

    /* Calculate risk based on current analysis state */
    if (bridge->current_utilities && bridge->current_num_actions > 0 &&
        action_id < bridge->current_num_actions) {

        /* Calculate variance of outcomes for this action */
        float mean = 0.0f;
        for (uint32_t o = 0; o < bridge->current_num_outcomes; o++) {
            mean += bridge->current_utilities[
                action_id * bridge->current_num_outcomes + o];
        }
        mean /= (float)bridge->current_num_outcomes;

        float variance = 0.0f;
        for (uint32_t o = 0; o < bridge->current_num_outcomes; o++) {
            float diff = bridge->current_utilities[
                action_id * bridge->current_num_outcomes + o] - mean;
            variance += diff * diff;
        }
        variance /= (float)bridge->current_num_outcomes;

        /* Strategic risk based on variance */
        assessment->strategic_risk = clamp_float(variance / 5.0f, 0.0f, 1.0f);

        /* Execution risk (simplified - based on action complexity) */
        assessment->execution_risk = clamp_float(
            (float)action_id / (float)bridge->current_num_actions, 0.0f, 0.5f);

        /* Opportunity cost (best alternative - this action's expected value) */
        float best_value = -1e10f;
        for (uint32_t a = 0; a < bridge->current_num_actions; a++) {
            if (a == action_id) continue;
            float value = 0.0f;
            for (uint32_t o = 0; o < bridge->current_num_outcomes; o++) {
                value += bridge->current_utilities[
                    a * bridge->current_num_outcomes + o];
            }
            value /= (float)bridge->current_num_outcomes;
            if (value > best_value) {
                best_value = value;
            }
        }
        assessment->opportunity_cost = clamp_float(best_value - mean, 0.0f, 1.0f);

        /* Overall risk */
        assessment->overall_risk = clamp_float(
            0.4f * assessment->strategic_risk +
            0.3f * assessment->execution_risk +
            0.3f * assessment->opportunity_cost,
            0.0f, 1.0f
        );

        snprintf(assessment->risk_factors, sizeof(assessment->risk_factors),
                 "Strategic: %.2f, Execution: %.2f, Opportunity: %.2f",
                 assessment->strategic_risk, assessment->execution_risk,
                 assessment->opportunity_cost);
    } else {
        /* No analysis data - moderate default risk */
        assessment->overall_risk = 0.5f;
        assessment->strategic_risk = 0.5f;
        assessment->execution_risk = 0.3f;
        assessment->opportunity_cost = 0.2f;
        strncpy(assessment->risk_factors, "No analysis data available",
                sizeof(assessment->risk_factors) - 1);
    }

    bridge->stats.risk_assessments++;

    /* Update average risk score */
    float n = (float)bridge->stats.risk_assessments;
    bridge->stats.avg_risk_score =
        ((n - 1.0f) * bridge->stats.avg_risk_score +
         assessment->overall_risk) / n;

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Decision Notification Implementation
 * ============================================================================ */

int game_theory_executive_notify_decision_made(
    game_theory_executive_bridge_t* bridge,
    uint64_t decision_id,
    uint64_t recommendation_id,
    uint32_t action_taken,
    bool followed_recommendation
) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    /* Create outcome structure */
    gt_decision_outcome_t outcome;
    outcome.decision_id = decision_id;
    outcome.recommendation_id = recommendation_id;
    outcome.action_taken = action_taken;
    outcome.outcome_utility = 0.0f;  /* Unknown at notification time */
    outcome.followed_recommendation = followed_recommendation;

    return game_theory_executive_notify_outcome(bridge, &outcome);
}

/* ============================================================================
 * Opponent Modeling Implementation
 * ============================================================================ */

int game_theory_executive_request_opponent_model(
    game_theory_executive_bridge_t* bridge,
    uint32_t opponent_id,
    const void* context,
    gt_exec_opponent_model_t* model
) {
    (void)context;  /* Reserved for future use */

    if (!bridge || !bridge->initialized || !model) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    gt_exec_opponent_model_t* internal_model =
        find_or_create_opponent_model(bridge, opponent_id);

    if (!internal_model) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Copy model to output */
    *model = *internal_model;

    bridge->stats.opponent_model_requests++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int game_theory_executive_update_opponent_model(
    game_theory_executive_bridge_t* bridge,
    uint32_t opponent_id,
    uint32_t observed_strategy,
    float outcome
) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    gt_exec_opponent_model_t* model =
        find_or_create_opponent_model(bridge, opponent_id);

    if (!model || observed_strategy >= model->num_strategies) {
        nimcp_mutex_unlock(bridge->base.mutex);
        return -1;
    }

    /* Bayesian-like update of strategy probabilities */
    float learning_rate = 0.1f;
    float decay = bridge->config.opponent_model_decay;

    /* Decay all probabilities */
    for (uint32_t i = 0; i < model->num_strategies; i++) {
        model->strategy_probs[i] *= (1.0f - decay);
    }

    /* Boost observed strategy */
    model->strategy_probs[observed_strategy] += learning_rate;

    /* Normalize */
    float sum = 0.0f;
    for (uint32_t i = 0; i < model->num_strategies; i++) {
        sum += model->strategy_probs[i];
    }
    if (sum > 0.0f) {
        for (uint32_t i = 0; i < model->num_strategies; i++) {
            model->strategy_probs[i] /= sum;
        }
    }

    /* Update predictability based on entropy */
    float entropy = 0.0f;
    for (uint32_t i = 0; i < model->num_strategies; i++) {
        if (model->strategy_probs[i] > 0.0f) {
            entropy -= model->strategy_probs[i] *
                logf(model->strategy_probs[i]);
        }
    }
    float max_entropy = logf((float)model->num_strategies);
    model->predictability = 1.0f - (entropy / max_entropy);

    /* Update cooperation tendency based on outcome */
    if (outcome > 0.5f) {
        model->cooperation_tendency =
            model->cooperation_tendency * 0.9f + 0.1f;
    } else {
        model->cooperation_tendency =
            model->cooperation_tendency * 0.9f;
    }
    model->cooperation_tendency = clamp_float(
        model->cooperation_tendency, 0.0f, 1.0f);

    model->last_update = get_timestamp_ms();
    model->interaction_count++;

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

/* ============================================================================
 * Situation Analysis Implementation
 * ============================================================================ */

int game_theory_executive_request_strategic_analysis(
    game_theory_executive_bridge_t* bridge,
    const gt_exec_situation_t* situation,
    gt_strategic_recommendation_t* recommendation
) {
    if (!bridge || !bridge->initialized || !situation || !recommendation) {
        return -1;
    }

    /* Set up the analysis */
    int result = game_theory_executive_analyze_options(
        bridge,
        situation->num_actions,
        situation->utilities,
        situation->num_outcomes
    );

    if (result != 0) {
        return result;
    }

    /* Get recommendation */
    return game_theory_executive_get_recommendation(bridge, recommendation);
}

/* ============================================================================
 * Query API Implementation
 * ============================================================================ */

game_theory_executive_state_t game_theory_executive_bridge_get_state(
    const game_theory_executive_bridge_t* bridge
) {
    if (!bridge || !bridge->initialized) {
        return GT_EXEC_STATE_ERROR;
    }

    game_theory_executive_bridge_t* mutable_bridge =
        (game_theory_executive_bridge_t*)bridge;

    nimcp_mutex_lock(mutable_bridge->base.mutex);
    game_theory_executive_state_t state = bridge->state;
    nimcp_mutex_unlock(mutable_bridge->base.mutex);

    return state;
}

uint32_t game_theory_executive_bridge_get_module_id(
    const game_theory_executive_bridge_t* bridge
) {
    if (!bridge || !bridge->initialized) {
        return 0;
    }

    return bridge->config.module_id;
}

uint32_t game_theory_executive_bridge_get_pending_count(
    const game_theory_executive_bridge_t* bridge
) {
    if (!bridge || !bridge->initialized) {
        return 0;
    }

    game_theory_executive_bridge_t* mutable_bridge =
        (game_theory_executive_bridge_t*)bridge;

    nimcp_mutex_lock(mutable_bridge->base.mutex);
    uint32_t count = bridge->has_pending_recommendation ? 1 : 0;
    nimcp_mutex_unlock(mutable_bridge->base.mutex);

    return count;
}

/* ============================================================================
 * Statistics Implementation
 * ============================================================================ */

int game_theory_executive_bridge_get_stats(
    const game_theory_executive_bridge_t* bridge,
    game_theory_executive_stats_t* stats
) {
    if (!bridge || !bridge->initialized || !stats) {
        return -1;
    }

    game_theory_executive_bridge_t* mutable_bridge =
        (game_theory_executive_bridge_t*)bridge;

    nimcp_mutex_lock(mutable_bridge->base.mutex);
    *stats = bridge->stats;
    nimcp_mutex_unlock(mutable_bridge->base.mutex);

    return 0;
}

int game_theory_executive_bridge_reset_stats(game_theory_executive_bridge_t* bridge) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);
    memset(&bridge->stats, 0, sizeof(game_theory_executive_stats_t));
    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}

int game_theory_executive_bridge_force_update(game_theory_executive_bridge_t* bridge) {
    if (!bridge || !bridge->initialized) {
        return -1;
    }

    nimcp_mutex_lock(bridge->base.mutex);

    /* Force state transition if stuck */
    if (bridge->state == GT_EXEC_STATE_AWAITING_DECISION) {
        /* Timeout waiting for decision - reset */
        bridge->has_pending_recommendation = false;
        bridge->state = GT_EXEC_STATE_IDLE;
    }

    nimcp_mutex_unlock(bridge->base.mutex);

    return 0;
}
