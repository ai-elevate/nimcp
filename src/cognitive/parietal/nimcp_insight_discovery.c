/**
 * @file nimcp_insight_discovery.c
 * @brief Insight and discovery engine implementation
 */

#include "cognitive/parietal/nimcp_insight_discovery.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * INTERNAL TYPES
 * ============================================================================ */

typedef struct {
    uint32_t id;
    insight_problem_t* problem;
    float progress;
    uint64_t start_time;
    uint32_t iterations;
    bool ready;
    insight_eureka_t* result;
} incubation_entry_t;

struct insight_engine {
    insight_config_t config;
    insight_stats_t stats;

    incubation_entry_t* incubation_queue;
    uint32_t num_incubating;
    uint32_t next_problem_id;
    uint32_t next_eureka_id;

    float inflammation;
    float fatigue;
};

static __thread char g_last_error[256] = {0};

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

static void set_error(const char* msg) {
    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
}

static uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static float apply_modulation(const insight_engine_t* e, float v) {
    float f = 1.0f - e->inflammation * e->config.inflammation_sensitivity * 0.3f
                   - e->fatigue * e->config.fatigue_sensitivity * 0.4f;
    return v * fmaxf(0.3f, f);
}

/* ============================================================================
 * LIFECYCLE
 * ============================================================================ */

insight_config_t insight_engine_default_config(void) {
    return (insight_config_t){
        .impasse_threshold = 0.7f,
        .incubation_rate = 0.05f,
        .constraint_relaxation_rate = 0.1f,
        .restructuring_threshold = 0.4f,
        .enable_incubation = true,
        .enable_constraint_relaxation = true,
        .enable_perspective_shifting = true,
        .max_restructuring_attempts = 10,
        .incubation_queue_size = INSIGHT_MAX_INCUBATION_QUEUE,
        .inflammation_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f
    };
}

insight_engine_t* insight_engine_create(void) {
    insight_config_t config = insight_engine_default_config();
    return insight_engine_create_custom(&config);
}

insight_engine_t* insight_engine_create_custom(const insight_config_t* config) {
    if (!config) { set_error("NULL config"); return NULL; }

    insight_engine_t* engine = nimcp_calloc(1, sizeof(insight_engine_t));
    if (!engine) { set_error("Alloc failed"); return NULL; }

    engine->config = *config;

    engine->incubation_queue = nimcp_calloc(config->incubation_queue_size,
                                            sizeof(incubation_entry_t));
    if (!engine->incubation_queue) {
        nimcp_free(engine);
        set_error("Queue alloc failed");
        return NULL;
    }

    engine->next_problem_id = 1;
    engine->next_eureka_id = 1;

    return engine;
}

void insight_engine_destroy(insight_engine_t* engine) {
    if (!engine) return;

    if (engine->incubation_queue) {
        for (uint32_t i = 0; i < engine->num_incubating; i++) {
            if (engine->incubation_queue[i].result) {
                insight_free_eureka(engine->incubation_queue[i].result);
            }
        }
        nimcp_free(engine->incubation_queue);
    }
    nimcp_free(engine);
}

/* ============================================================================
 * PROBLEM MANAGEMENT
 * ============================================================================ */

insight_problem_t* insight_create_problem(const char* description) {
    insight_problem_t* problem = nimcp_calloc(1, sizeof(insight_problem_t));
    if (!problem) return NULL;

    if (description) {
        strncpy(problem->description, description, sizeof(problem->description) - 1);
    }

    problem->constraints = nimcp_calloc(INSIGHT_MAX_CONSTRAINTS, sizeof(insight_constraint_t));
    problem->perspectives = nimcp_calloc(INSIGHT_MAX_PERSPECTIVES, sizeof(insight_perspective_t));

    return problem;
}

int insight_add_constraint(insight_problem_t* problem, const char* description,
                          float binding_strength, bool is_explicit) {
    if (!problem || problem->num_constraints >= INSIGHT_MAX_CONSTRAINTS) return -1;

    insight_constraint_t* c = &problem->constraints[problem->num_constraints];
    c->id = problem->num_constraints + 1;
    if (description) strncpy(c->description, description, sizeof(c->description) - 1);
    c->binding_strength = fmaxf(0, fminf(1, binding_strength));
    c->is_explicit = is_explicit;
    c->is_relaxable = !is_explicit;
    c->relaxation_cost = c->binding_strength * 0.5f;

    problem->num_constraints++;
    return 0;
}

int insight_add_perspective(insight_problem_t* problem, const char* description,
                           const float* representation, uint32_t dim) {
    if (!problem || problem->num_perspectives >= INSIGHT_MAX_PERSPECTIVES) return -1;

    insight_perspective_t* p = &problem->perspectives[problem->num_perspectives];
    p->id = problem->num_perspectives + 1;
    if (description) strncpy(p->description, description, sizeof(p->description) - 1);

    if (representation && dim > 0) {
        p->representation = nimcp_calloc(dim, sizeof(float));
        if (p->representation) {
            memcpy(p->representation, representation, dim * sizeof(float));
            p->repr_dim = dim;
        }
    }

    p->novelty = (problem->num_perspectives == 0) ? 0.0f : 0.5f;
    p->utility = 0.5f;

    problem->num_perspectives++;
    return 0;
}

void insight_free_problem(insight_problem_t* problem) {
    if (!problem) return;

    if (problem->state) nimcp_free(problem->state);
    if (problem->goal) nimcp_free(problem->goal);
    if (problem->constraints) nimcp_free(problem->constraints);

    if (problem->perspectives) {
        for (uint32_t i = 0; i < problem->num_perspectives; i++) {
            if (problem->perspectives[i].representation) {
                nimcp_free(problem->perspectives[i].representation);
            }
        }
        nimcp_free(problem->perspectives);
    }

    nimcp_free(problem);
}

/* ============================================================================
 * INCUBATION
 * ============================================================================ */

uint32_t insight_incubate(insight_engine_t* engine, insight_problem_t* problem) {
    if (!engine || !problem) return 0;
    if (!engine->config.enable_incubation) return 0;
    if (engine->num_incubating >= engine->config.incubation_queue_size) return 0;

    incubation_entry_t* entry = &engine->incubation_queue[engine->num_incubating];
    entry->id = engine->next_problem_id++;
    entry->problem = problem;
    entry->progress = 0.0f;
    entry->start_time = get_timestamp_us();
    entry->iterations = 0;
    entry->ready = false;
    entry->result = NULL;

    engine->num_incubating++;
    return entry->id;
}

int insight_check_incubation(insight_engine_t* engine, uint32_t problem_id,
                            insight_eureka_t** result) {
    if (!engine || !result) return -1;
    *result = NULL;

    for (uint32_t i = 0; i < engine->num_incubating; i++) {
        if (engine->incubation_queue[i].id == problem_id) {
            if (engine->incubation_queue[i].ready) {
                *result = engine->incubation_queue[i].result;
                engine->incubation_queue[i].result = NULL;
                return 1;
            }
            return 0;
        }
    }
    return -1;
}

int insight_process_incubation_step(insight_engine_t* engine) {
    if (!engine) return -1;

    int insights = 0;
    float rate = engine->config.incubation_rate * (1.0f - engine->fatigue * 0.5f);

    for (uint32_t i = 0; i < engine->num_incubating; i++) {
        incubation_entry_t* e = &engine->incubation_queue[i];
        if (e->ready) continue;

        e->iterations++;
        e->progress += rate * (1.0f - e->problem->impasse_level * 0.3f);

        if (e->progress >= 1.0f) {
            e->ready = true;

            /* Generate eureka */
            e->result = nimcp_calloc(1, sizeof(insight_eureka_t));
            if (e->result) {
                e->result->id = engine->next_eureka_id++;
                e->result->problem = e->problem;
                e->result->surprise_magnitude = 0.6f + 0.4f * ((float)rand() / RAND_MAX);
                e->result->elegance = 0.5f + 0.5f * e->progress;
                e->result->confidence = apply_modulation(engine, 0.7f);
                e->result->incubation_time_us = get_timestamp_us() - e->start_time;
                e->result->restructuring_attempts = e->iterations;
                snprintf(e->result->description, sizeof(e->result->description),
                        "Insight after %u iterations", e->iterations);

                engine->stats.insights_generated++;
                insights++;
            }
        }
    }

    return insights;
}

int insight_cancel_incubation(insight_engine_t* engine, uint32_t problem_id) {
    if (!engine) return -1;

    for (uint32_t i = 0; i < engine->num_incubating; i++) {
        if (engine->incubation_queue[i].id == problem_id) {
            if (engine->incubation_queue[i].result) {
                insight_free_eureka(engine->incubation_queue[i].result);
            }
            /* Shift remaining */
            for (uint32_t j = i; j < engine->num_incubating - 1; j++) {
                engine->incubation_queue[j] = engine->incubation_queue[j + 1];
            }
            engine->num_incubating--;
            return 0;
        }
    }
    return -1;
}

/* ============================================================================
 * CONSTRAINT RELAXATION
 * ============================================================================ */

int insight_identify_blocking_constraints(insight_engine_t* engine,
                                         const insight_problem_t* problem,
                                         insight_constraint_t* blocking,
                                         uint32_t max_constraints,
                                         uint32_t* num_found) {
    if (!engine || !problem || !blocking || !num_found) return -1;
    *num_found = 0;

    for (uint32_t i = 0; i < problem->num_constraints && *num_found < max_constraints; i++) {
        if (problem->constraints[i].binding_strength > 0.5f) {
            blocking[(*num_found)++] = problem->constraints[i];
        }
    }

    engine->stats.impasses_detected++;
    return 0;
}

int insight_relax_constraint(insight_engine_t* engine,
                            insight_problem_t* problem,
                            uint32_t constraint_id) {
    if (!engine || !problem) return -1;
    if (!engine->config.enable_constraint_relaxation) return -1;

    for (uint32_t i = 0; i < problem->num_constraints; i++) {
        if (problem->constraints[i].id == constraint_id) {
            if (!problem->constraints[i].is_relaxable) return -1;

            problem->constraints[i].binding_strength *= 0.5f;
            problem->impasse_level *= 0.8f;

            engine->stats.constraints_relaxed++;
            return 0;
        }
    }
    return -1;
}

int insight_find_relaxable_constraints(insight_engine_t* engine,
                                       const insight_problem_t* problem,
                                       uint32_t* constraint_ids,
                                       uint32_t max_constraints,
                                       uint32_t* num_found) {
    if (!engine || !problem || !constraint_ids || !num_found) return -1;
    *num_found = 0;

    for (uint32_t i = 0; i < problem->num_constraints && *num_found < max_constraints; i++) {
        if (problem->constraints[i].is_relaxable) {
            constraint_ids[(*num_found)++] = problem->constraints[i].id;
        }
    }
    return 0;
}

/* ============================================================================
 * RESTRUCTURING
 * ============================================================================ */

insight_restructuring_t* insight_attempt_restructure(insight_engine_t* engine,
                                                    insight_problem_t* problem) {
    if (!engine || !problem) return NULL;

    insight_restructuring_t* r = nimcp_calloc(1, sizeof(insight_restructuring_t));
    if (!r) return NULL;

    /* Find and relax a constraint */
    for (uint32_t i = 0; i < problem->num_constraints; i++) {
        if (problem->constraints[i].is_relaxable &&
            problem->constraints[i].binding_strength > 0.3f) {
            r->constraint_relaxed = problem->constraints[i].id;
            problem->constraints[i].binding_strength *= 0.5f;
            break;
        }
    }

    /* Shift perspective if available */
    if (problem->num_perspectives > 1) {
        r->new_perspective = (problem->current_perspective + 1) % problem->num_perspectives;
        problem->current_perspective = r->new_perspective;
        engine->stats.perspectives_shifted++;
    }

    r->novelty = 0.5f + 0.5f * ((float)rand() / RAND_MAX);
    snprintf(r->description, sizeof(r->description), "Restructured problem");

    return r;
}

int insight_shift_perspective(insight_engine_t* engine,
                             insight_problem_t* problem,
                             uint32_t new_perspective) {
    if (!engine || !problem) return -1;
    if (!engine->config.enable_perspective_shifting) return -1;
    if (new_perspective >= problem->num_perspectives) return -1;

    problem->current_perspective = new_perspective;
    problem->impasse_level *= 0.7f;

    engine->stats.perspectives_shifted++;
    return 0;
}

int insight_generate_perspectives(insight_engine_t* engine,
                                 insight_problem_t* problem,
                                 uint32_t max_perspectives) {
    if (!engine || !problem) return -1;

    uint32_t generated = 0;
    while (problem->num_perspectives < INSIGHT_MAX_PERSPECTIVES &&
           generated < max_perspectives) {
        char desc[256];
        snprintf(desc, sizeof(desc), "Generated perspective %u",
                problem->num_perspectives + 1);
        insight_add_perspective(problem, desc, NULL, 0);
        generated++;
    }

    return (int)generated;
}

void insight_free_restructuring(insight_restructuring_t* r) {
    if (r) nimcp_free(r);
}

/* ============================================================================
 * EUREKA DETECTION
 * ============================================================================ */

bool insight_check_eureka(insight_engine_t* engine,
                         const insight_problem_t* problem,
                         insight_eureka_t** result) {
    if (!engine || !problem || !result) return false;
    *result = NULL;

    if (problem->is_solved) {
        *result = nimcp_calloc(1, sizeof(insight_eureka_t));
        if (*result) {
            (*result)->id = engine->next_eureka_id++;
            (*result)->surprise_magnitude = 0.8f;
            (*result)->elegance = 0.7f;
            (*result)->confidence = apply_modulation(engine, 0.9f);
            engine->stats.insights_generated++;
            return true;
        }
    }

    return false;
}

float insight_estimate_surprise(insight_engine_t* engine,
                               const insight_eureka_t* eureka) {
    if (!engine || !eureka) return 0.0f;
    return apply_modulation(engine, eureka->surprise_magnitude);
}

int insight_verify_eureka(insight_engine_t* engine, insight_eureka_t* eureka) {
    if (!engine || !eureka) return -1;
    eureka->verified = true;
    eureka->confidence = fminf(1.0f, eureka->confidence * 1.2f);
    return 0;
}

void insight_free_eureka(insight_eureka_t* eureka) {
    if (!eureka) return;
    if (eureka->solution) nimcp_free(eureka->solution);
    nimcp_free(eureka);
}

/* ============================================================================
 * IMPASSE
 * ============================================================================ */

int insight_detect_impasse(insight_engine_t* engine,
                          const insight_problem_t* problem,
                          insight_impasse_t* impasse) {
    if (!engine || !problem || !impasse) return -1;

    memset(impasse, 0, sizeof(insight_impasse_t));
    impasse->stuckness_level = problem->impasse_level;
    impasse->ready_for_incubation = (impasse->stuckness_level >= engine->config.impasse_threshold);

    return 0;
}

int insight_resolve_impasse(insight_engine_t* engine,
                           insight_problem_t* problem,
                           const insight_impasse_t* impasse) {
    if (!engine || !problem || !impasse) return -1;

    /* Try relaxing constraints */
    for (uint32_t i = 0; i < problem->num_constraints; i++) {
        if (problem->constraints[i].is_relaxable) {
            insight_relax_constraint(engine, problem, problem->constraints[i].id);
            break;
        }
    }

    /* Shift perspective */
    if (problem->num_perspectives > 1) {
        insight_shift_perspective(engine, problem,
            (problem->current_perspective + 1) % problem->num_perspectives);
    }

    problem->impasse_level *= 0.5f;
    engine->stats.impasses_resolved++;

    return 0;
}

/* ============================================================================
 * MODULATION
 * ============================================================================ */

int insight_set_inflammation(insight_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->inflammation = fmaxf(0, fminf(1, level));
    return 0;
}

int insight_set_fatigue(insight_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->fatigue = fmaxf(0, fminf(1, level));
    return 0;
}

/* ============================================================================
 * STATISTICS
 * ============================================================================ */

int insight_get_stats(const insight_engine_t* engine, insight_stats_t* stats) {
    if (!engine || !stats) return -1;
    *stats = engine->stats;
    return 0;
}

void insight_reset_stats(insight_engine_t* engine) {
    if (engine) memset(&engine->stats, 0, sizeof(engine->stats));
}

const char* insight_get_last_error(void) {
    return g_last_error;
}

/* ============================================================================
 * KG SELF-AWARENESS INTEGRATION
 * ============================================================================ */

int insight_discovery_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Insight_Discovery_Module");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG("Insight discovery self-knowledge: %s", self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Insight_Discovery_Module");
    if (connections) {
        LOG_DEBUG("Insight discovery has %u outgoing connections", connections->count);
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Insight_Discovery_Module");
    if (incoming) {
        LOG_DEBUG("Insight discovery has %u incoming connections", incoming->count);
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
