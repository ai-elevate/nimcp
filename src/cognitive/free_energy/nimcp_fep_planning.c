/**
 * @file nimcp_fep_planning.c
 * @brief Multi-step MCTS Planning Module for Free Energy Principle
 * @version 1.0.0
 * @date 2025-12-12
 *
 * WHAT: Implementation of MCTS-based planning for FEP
 * WHY:  Multi-step planning enables goal-directed behavior
 * HOW:  Monte Carlo Tree Search with FEP's generative model and EFE as value
 */

#include "cognitive/free_energy/nimcp_fep_planning.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "api/nimcp_api_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static inline float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static inline float safe_sqrt(float x) {
    return x > 0.0f ? sqrtf(x) : 0.0f;
}

static inline float safe_log(float x) {
    return x > 0.0f ? logf(x) : -100.0f;
}

/* Compute UCB1 value for node selection */
static float compute_ucb(const mcts_node_t* node, uint32_t parent_visits, float c) {
    if (!node || node->visit_count == 0) return FLT_MAX;

    float q = node->q_value;
    float exploration = c * safe_sqrt(safe_log((float)parent_visits) / (float)node->visit_count);
    return q + exploration;
}

/* Allocate new MCTS node */
static mcts_node_t* alloc_node(fep_planning_system_t* sys) {
    if (!sys || sys->num_nodes >= sys->max_nodes) return NULL;

    mcts_node_t* node = (mcts_node_t*)nimcp_calloc(1, sizeof(mcts_node_t));
    if (!node) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "node is NULL");

        return NULL;

    }

    node->node_id = sys->num_nodes;
    node->state_type = MCTS_NODE_UNVISITED;
    node->visit_count = 0;
    node->q_value = 0.0f;

    sys->tree_nodes[sys->num_nodes++] = node;
    sys->stats.nodes_created++;

    return node;
}

static void free_node(mcts_node_t* node) {
    if (!node) return;

    if (node->children_ids) nimcp_free(node->children_ids);
    if (node->state) nimcp_free(node->state);

    nimcp_free(node);
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

void fep_planning_default_config(fep_planning_config_t* config) {
    if (!config) return;

    config->method = PLANNING_MCTS;
    config->planning_horizon = FEP_PLANNING_DEFAULT_HORIZON;
    config->num_simulations = FEP_PLANNING_DEFAULT_SIMULATIONS;
    config->exploration_constant = FEP_PLANNING_DEFAULT_EXPLORATION;
    config->discount_factor = FEP_PLANNING_DEFAULT_DISCOUNT;
    config->use_prior = false;
    config->beam_width = FEP_PLANNING_DEFAULT_BEAM_WIDTH;
    config->pruning_threshold = FEP_PLANNING_DEFAULT_PRUNING;
    config->enable_caching = true;
    config->enable_parallel_rollouts = false;
    config->enable_progressive_widening = false;
    config->convergence_threshold = 0.01f;
    config->max_tree_nodes = FEP_PLANNING_MAX_TREE_NODES;
}

fep_planning_system_t* fep_planning_create(const fep_planning_config_t* config) {
    fep_planning_system_t* sys = (fep_planning_system_t*)nimcp_calloc(
        1, sizeof(fep_planning_system_t));
    NIMCP_API_CHECK_ALLOC(sys, "Failed to allocate planning system");

    /* Apply configuration */
    fep_planning_config_t default_cfg;
    if (!config) {
        fep_planning_default_config(&default_cfg);
        config = &default_cfg;
    }
    sys->config = *config;

    /* Allocate tree nodes array */
    sys->max_nodes = config->max_tree_nodes;
    sys->tree_nodes = (mcts_node_t**)nimcp_calloc(sys->max_nodes, sizeof(mcts_node_t*));
    if (!sys->tree_nodes) {
        fep_planning_destroy(sys);
        return NULL;
    }

    /* Create root node */
    mcts_node_t* root = alloc_node(sys);
    if (!root) {
        fep_planning_destroy(sys);
        return NULL;
    }
    sys->root_id = root->node_id;

    /* Create mutex */
    sys->mutex = nimcp_platform_mutex_create();
    if (!sys->mutex) {
        fep_planning_destroy(sys);
        return NULL;
    }

    NIMCP_LOGGING_INFO("Planning system created: horizon=%u, simulations=%u",
                      config->planning_horizon, config->num_simulations);
    return sys;
}

void fep_planning_destroy(fep_planning_system_t* sys) {
    if (!sys) return;

    if (sys->bio_async_enabled) {
        fep_planning_disconnect_bio_async(sys);
    }

    /* Free all tree nodes */
    if (sys->tree_nodes) {
        for (uint32_t i = 0; i < sys->num_nodes; i++) {
            free_node(sys->tree_nodes[i]);
        }
        nimcp_free(sys->tree_nodes);
    }

    /* Free current plan */
    fep_plan_destroy(&sys->current_plan);

    if (sys->mutex) {
        nimcp_platform_mutex_destroy(sys->mutex);
    }

    nimcp_free(sys);
    NIMCP_LOGGING_INFO("Planning system destroyed");
}

int fep_planning_reset(fep_planning_system_t* sys) {
    if (!sys) return -1;

    nimcp_platform_mutex_lock(sys->mutex);

    /* Free all nodes except root */
    for (uint32_t i = 1; i < sys->num_nodes; i++) {
        free_node(sys->tree_nodes[i]);
        sys->tree_nodes[i] = NULL;
    }

    /* Reset root */
    if (sys->tree_nodes[0]) {
        sys->tree_nodes[0]->visit_count = 0;
        sys->tree_nodes[0]->q_value = 0.0f;
        sys->tree_nodes[0]->total_value = 0.0f;
        sys->tree_nodes[0]->num_children = 0;
        sys->tree_nodes[0]->state_type = MCTS_NODE_UNVISITED;
    }

    sys->num_nodes = 1;
    sys->plan_step = 0;

    /* Reset stats */
    memset(&sys->stats, 0, sizeof(sys->stats));

    nimcp_platform_mutex_unlock(sys->mutex);
    return 0;
}

/* ============================================================================
 * FEP Integration Implementation
 * ============================================================================ */

int fep_planning_connect(fep_planning_system_t* planning, fep_system_t* fep) {
    if (!planning || !fep) return -1;

    nimcp_platform_mutex_lock(planning->mutex);
    planning->fep = fep;
    planning->fep_connected = true;
    nimcp_platform_mutex_unlock(planning->mutex);

    NIMCP_LOGGING_INFO("Planning connected to FEP");
    return 0;
}

int fep_planning_disconnect(fep_planning_system_t* planning) {
    if (!planning) return -1;

    nimcp_platform_mutex_lock(planning->mutex);
    planning->fep = NULL;
    planning->fep_connected = false;
    nimcp_platform_mutex_unlock(planning->mutex);

    return 0;
}

/* ============================================================================
 * MCTS Implementation
 * ============================================================================ */

int fep_mcts_select(fep_planning_system_t* sys, uint32_t node_id, uint32_t* action) {
    if (!sys || !action || node_id >= sys->num_nodes) return -1;

    mcts_node_t* node = sys->tree_nodes[node_id];
    if (!node || node->num_children == 0) return -1;

    /* Find child with highest UCB value */
    float best_ucb = -FLT_MAX;
    uint32_t best_child = 0;
    uint32_t best_action = 0;

    for (uint32_t i = 0; i < node->num_children; i++) {
        uint32_t child_id = node->children_ids[i];
        if (child_id >= sys->num_nodes) continue;

        mcts_node_t* child = sys->tree_nodes[child_id];
        if (!child) continue;

        float ucb = compute_ucb(child, node->visit_count, sys->config.exploration_constant);
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_child = child_id;
            best_action = child->action_id;
        }
    }

    *action = best_action;
    return best_child;
}

int fep_mcts_expand(fep_planning_system_t* sys, uint32_t node_id) {
    if (!sys || node_id >= sys->num_nodes) return -1;

    mcts_node_t* node = sys->tree_nodes[node_id];
    if (!node) return -1;
    if (node->state_type == MCTS_NODE_TERMINAL) return 0;

    /* Get number of available actions */
    uint32_t num_actions = sys->fep ? sys->fep->num_actions : 4;
    if (num_actions > FEP_PLANNING_MAX_ACTIONS) {
        num_actions = FEP_PLANNING_MAX_ACTIONS;
    }

    /* Allocate children array */
    node->children_ids = (uint32_t*)nimcp_calloc(num_actions, sizeof(uint32_t));
    if (!node->children_ids) return -1;

    /* Create child for each action */
    uint32_t created = 0;
    for (uint32_t a = 0; a < num_actions; a++) {
        mcts_node_t* child = alloc_node(sys);
        if (!child) break;

        child->parent_id = node_id;
        child->action_id = a;
        child->depth = node->depth + 1;
        child->state_dim = node->state_dim;

        /* Check for terminal (horizon reached) */
        if (child->depth >= sys->config.planning_horizon) {
            child->state_type = MCTS_NODE_TERMINAL;
        }

        node->children_ids[created++] = child->node_id;
    }

    node->num_children = created;
    node->state_type = MCTS_NODE_EXPANDED;

    return (int)created;
}

int fep_mcts_simulate(
    fep_planning_system_t* sys,
    fep_system_t* fep,
    uint32_t node_id,
    float* value
) {
    if (!sys || !value) return -1;

    mcts_node_t* node = sys->tree_nodes[node_id];
    if (!node) return -1;

    /* Simulate rollout from this node */
    float total_value = 0.0f;
    float discount = 1.0f;

    uint32_t current_depth = node->depth;
    uint32_t horizon = sys->config.planning_horizon;

    /* Simple rollout: estimate value based on depth and random actions */
    for (uint32_t t = current_depth; t < horizon; t++) {
        /* Use FEP's EFE as reward signal if available */
        float step_value = 0.0f;

        if (fep && fep->num_levels > 0) {
            /* Estimate value from prediction error (lower is better) */
            float pe = fep->levels[0].errors.magnitude;
            step_value = 1.0f / (1.0f + pe);  /* Convert to positive reward */
        } else {
            step_value = 0.5f;  /* Default moderate value */
        }

        total_value += discount * step_value;
        discount *= sys->config.discount_factor;
    }

    *value = total_value;
    sys->stats.simulations_run++;

    return 0;
}

int fep_mcts_backpropagate(fep_planning_system_t* sys, uint32_t node_id, float value) {
    if (!sys) return -1;

    /* Backpropagate value up the tree */
    uint32_t current_id = node_id;

    while (current_id < sys->num_nodes) {
        mcts_node_t* node = sys->tree_nodes[current_id];
        if (!node) break;

        node->visit_count++;
        node->total_value += value;
        node->q_value = node->total_value / (float)node->visit_count;

        if (current_id == sys->root_id) break;
        current_id = node->parent_id;
    }

    return 0;
}

uint32_t fep_mcts_get_best_child(const fep_planning_system_t* sys, uint32_t node_id) {
    if (!sys || node_id >= sys->num_nodes) return 0;

    const mcts_node_t* node = sys->tree_nodes[node_id];
    if (!node || node->num_children == 0) return 0;

    /* Find most visited child (robust selection) */
    uint32_t best_child = 0;
    uint32_t max_visits = 0;

    for (uint32_t i = 0; i < node->num_children; i++) {
        uint32_t child_id = node->children_ids[i];
        if (child_id >= sys->num_nodes) continue;

        const mcts_node_t* child = sys->tree_nodes[child_id];
        if (child && child->visit_count > max_visits) {
            max_visits = child->visit_count;
            best_child = child_id;
        }
    }

    return best_child;
}

/* ============================================================================
 * Planning Implementation
 * ============================================================================ */

int fep_planning_generate_plan(
    fep_planning_system_t* sys,
    fep_system_t* fep,
    const float* current_state,
    size_t state_dim,
    fep_plan_t* plan
) {
    if (!sys || !plan) return -1;

    nimcp_platform_mutex_lock(sys->mutex);

    /* Reset tree for new planning (inline to avoid deadlock) */
    /* Free all nodes except root */
    for (uint32_t i = 1; i < sys->num_nodes; i++) {
        free_node(sys->tree_nodes[i]);
        sys->tree_nodes[i] = NULL;
    }

    /* Reset root */
    if (sys->tree_nodes[0]) {
        sys->tree_nodes[0]->visit_count = 0;
        sys->tree_nodes[0]->q_value = 0.0f;
        sys->tree_nodes[0]->total_value = 0.0f;
        sys->tree_nodes[0]->num_children = 0;
        sys->tree_nodes[0]->state_type = MCTS_NODE_UNVISITED;
        if (sys->tree_nodes[0]->children_ids) {
            nimcp_free(sys->tree_nodes[0]->children_ids);
            sys->tree_nodes[0]->children_ids = NULL;
        }
    }

    sys->num_nodes = 1;
    sys->plan_step = 0;
    memset(&sys->stats, 0, sizeof(sys->stats));

    /* Store current state in root */
    mcts_node_t* root = sys->tree_nodes[sys->root_id];
    if (current_state && state_dim > 0) {
        root->state = (float*)nimcp_calloc(state_dim, sizeof(float));
        if (root->state) {
            memcpy(root->state, current_state, state_dim * sizeof(float));
            root->state_dim = (uint32_t)state_dim;
        }
    }

    /* Run MCTS iterations */
    for (uint32_t sim = 0; sim < sys->config.num_simulations; sim++) {
        /* Selection: traverse to leaf */
        uint32_t node_id = sys->root_id;
        uint32_t action;

        while (sys->tree_nodes[node_id]->state_type == MCTS_NODE_EXPANDED &&
               sys->tree_nodes[node_id]->num_children > 0) {
            int child_id = fep_mcts_select(sys, node_id, &action);
            if (child_id <= 0) break;
            node_id = (uint32_t)child_id;
        }

        /* Expansion: add children if not terminal */
        mcts_node_t* node = sys->tree_nodes[node_id];
        if (node->state_type == MCTS_NODE_UNVISITED) {
            fep_mcts_expand(sys, node_id);
            if (node->num_children > 0) {
                node_id = node->children_ids[0];
            }
        }

        /* Simulation: rollout to horizon */
        float value;
        fep_mcts_simulate(sys, fep, node_id, &value);

        /* Backpropagation */
        fep_mcts_backpropagate(sys, node_id, value);
    }

    /* Extract plan from tree */
    fep_plan_create(plan, sys->config.planning_horizon);

    uint32_t current_id = sys->root_id;
    uint32_t step = 0;

    while (step < sys->config.planning_horizon &&
           sys->tree_nodes[current_id]->num_children > 0) {
        uint32_t best_child = fep_mcts_get_best_child(sys, current_id);
        if (best_child == 0) break;

        mcts_node_t* child = sys->tree_nodes[best_child];
        plan->action_sequence[step] = child->action_id;
        plan->step_values[step] = child->q_value;
        step++;

        current_id = best_child;
    }

    plan->sequence_length = step;
    plan->expected_value = root->q_value;

    sys->stats.plans_generated++;
    sys->stats.avg_plan_length = (sys->stats.avg_plan_length *
        (sys->stats.plans_generated - 1) + step) / sys->stats.plans_generated;

    /* Store as current plan */
    fep_plan_copy(&sys->current_plan, plan);
    sys->plan_step = 0;

    nimcp_platform_mutex_unlock(sys->mutex);
    return 0;
}

int fep_planning_evaluate_plan(
    fep_planning_system_t* sys,
    fep_system_t* fep,
    const fep_plan_t* plan,
    float* value
) {
    if (!sys || !plan || !value) return -1;

    float total = 0.0f;
    float discount = 1.0f;

    for (size_t i = 0; i < plan->sequence_length; i++) {
        total += discount * plan->step_values[i];
        discount *= sys->config.discount_factor;
    }

    *value = total;
    return 0;
}

int fep_planning_replan(
    fep_planning_system_t* sys,
    fep_system_t* fep,
    const float* new_state,
    size_t state_dim,
    fep_plan_t* plan
) {
    /* For now, just regenerate from scratch */
    return fep_planning_generate_plan(sys, fep, new_state, state_dim, plan);
}

/* ============================================================================
 * Plan Management Implementation
 * ============================================================================ */

int fep_plan_create(fep_plan_t* plan, size_t max_length) {
    if (!plan) return -1;

    memset(plan, 0, sizeof(fep_plan_t));

    plan->action_sequence = (uint32_t*)nimcp_calloc(max_length, sizeof(uint32_t));
    plan->step_values = (float*)nimcp_calloc(max_length, sizeof(float));
    plan->step_efe = (float*)nimcp_calloc(max_length, sizeof(float));

    if (!plan->action_sequence || !plan->step_values || !plan->step_efe) {
        fep_plan_destroy(plan);
        return -1;
    }

    return 0;
}

void fep_plan_destroy(fep_plan_t* plan) {
    if (!plan) return;

    if (plan->action_sequence) nimcp_free(plan->action_sequence);
    if (plan->step_values) nimcp_free(plan->step_values);
    if (plan->step_efe) nimcp_free(plan->step_efe);

    memset(plan, 0, sizeof(fep_plan_t));
}

int fep_plan_get_next_action(const fep_plan_t* plan, uint32_t step, uint32_t* action) {
    if (!plan || !action) return -1;
    if (step >= plan->sequence_length) return -1;

    *action = plan->action_sequence[step];
    return 0;
}

int fep_plan_copy(fep_plan_t* dest, const fep_plan_t* src) {
    if (!dest || !src) return -1;

    if (dest->action_sequence) nimcp_free(dest->action_sequence);
    if (dest->step_values) nimcp_free(dest->step_values);
    if (dest->step_efe) nimcp_free(dest->step_efe);

    dest->sequence_length = src->sequence_length;
    dest->expected_value = src->expected_value;
    dest->uncertainty = src->uncertainty;

    if (src->sequence_length > 0) {
        dest->action_sequence = (uint32_t*)nimcp_calloc(src->sequence_length, sizeof(uint32_t));
        dest->step_values = (float*)nimcp_calloc(src->sequence_length, sizeof(float));
        dest->step_efe = (float*)nimcp_calloc(src->sequence_length, sizeof(float));

        if (dest->action_sequence) {
            memcpy(dest->action_sequence, src->action_sequence,
                   src->sequence_length * sizeof(uint32_t));
        }
        if (dest->step_values && src->step_values) {
            memcpy(dest->step_values, src->step_values,
                   src->sequence_length * sizeof(float));
        }
        if (dest->step_efe && src->step_efe) {
            memcpy(dest->step_efe, src->step_efe,
                   src->sequence_length * sizeof(float));
        }
    }

    return 0;
}

bool fep_plan_is_valid(const fep_plan_t* plan) {
    return plan && plan->action_sequence && plan->sequence_length > 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

int fep_planning_get_stats(const fep_planning_system_t* sys, fep_planning_stats_t* stats) {
    if (!sys || !stats) return -1;
    *stats = sys->stats;
    return 0;
}

uint32_t fep_planning_get_tree_size(const fep_planning_system_t* sys) {
    return sys ? sys->num_nodes : 0;
}

uint32_t fep_planning_get_tree_depth(const fep_planning_system_t* sys) {
    if (!sys) return 0;

    uint32_t max_depth = 0;
    for (uint32_t i = 0; i < sys->num_nodes; i++) {
        if (sys->tree_nodes[i] && sys->tree_nodes[i]->depth > max_depth) {
            max_depth = sys->tree_nodes[i]->depth;
        }
    }
    return max_depth;
}

/* ============================================================================
 * Bio-Async Implementation
 * ============================================================================ */

int fep_planning_connect_bio_async(fep_planning_system_t* sys) {
    if (!sys) return -1;
    if (sys->bio_async_enabled) return 0;

    bio_module_info_t info = {
        .module_id = BIO_MODULE_FEP_PLANNING,
        .module_name = "fep_planning",
        .inbox_capacity = FEP_PLANNING_BIO_INBOX_SIZE,
        .user_data = sys
    };

    sys->bio_ctx = bio_router_register_module(&info);
    if (sys->bio_ctx) {
        sys->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Planning connected to bio-async");
    } else {
        NIMCP_LOGGING_WARN("Bio-async router not available, skipping registration");
    }
    return 0;
}

int fep_planning_disconnect_bio_async(fep_planning_system_t* sys) {
    if (!sys) return -1;
    if (!sys->bio_async_enabled) return 0;

    if (sys->bio_ctx) {
        bio_router_unregister_module(sys->bio_ctx);
        sys->bio_ctx = NULL;
    }
    sys->bio_async_enabled = false;
    return 0;
}

bool fep_planning_is_bio_async_connected(const fep_planning_system_t* sys) {
    return sys && sys->bio_async_enabled;
}

/* ============================================================================
 * String Conversion Implementation
 * ============================================================================ */

const char* fep_planning_method_to_string(fep_planning_method_t method) {
    switch (method) {
        case PLANNING_MCTS:         return "MCTS";
        case PLANNING_BEAM_SEARCH:  return "BEAM_SEARCH";
        case PLANNING_DYNAMIC_PROG: return "DYNAMIC_PROGRAMMING";
        case PLANNING_ROLLOUT:      return "ROLLOUT";
        default:                    return "UNKNOWN";
    }
}

const char* fep_planning_node_state_to_string(mcts_node_state_t state) {
    switch (state) {
        case MCTS_NODE_UNVISITED: return "UNVISITED";
        case MCTS_NODE_EXPANDED:  return "EXPANDED";
        case MCTS_NODE_TERMINAL:  return "TERMINAL";
        default:                  return "UNKNOWN";
    }
}

/* ============================================================================
 * KG Self-Awareness Integration
 * ============================================================================ */

int fep_planning_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "FEP_Planning");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            NIMCP_LOGGING_DEBUG("FEP Planning self-knowledge: %s", self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "FEP_Planning");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "FEP_Planning");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
