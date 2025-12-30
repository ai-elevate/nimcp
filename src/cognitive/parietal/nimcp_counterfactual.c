/**
 * @file nimcp_counterfactual.c
 * @brief Counterfactual reasoning engine implementation
 */

#include "cognitive/parietal/nimcp_counterfactual.h"
#include "utils/memory/nimcp_memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

struct counterfactual_engine {
    cf_config_t config;
    cf_stats_t stats;
    float inflammation;
    float fatigue;
    uint32_t next_cf_id;
};

static __thread char g_last_error[256] = {0};

static void set_error(const char* msg) {
    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
}

static float apply_mod(const counterfactual_engine_t* e, float v) {
    float f = 1.0f - e->inflammation * e->config.inflammation_sensitivity * 0.2f
                   - e->fatigue * e->config.fatigue_sensitivity * 0.3f;
    return v * fmaxf(0.4f, f);
}

cf_config_t counterfactual_engine_default_config(void) {
    return (cf_config_t){
        .min_plausibility = 0.2f,
        .consequence_threshold = 0.1f,
        .enable_causal_tracing = true,
        .use_minimal_change = true,
        .max_trace_depth = 5,
        .inflammation_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f
    };
}

counterfactual_engine_t* counterfactual_engine_create(void) {
    cf_config_t c = counterfactual_engine_default_config();
    return counterfactual_engine_create_custom(&c);
}

counterfactual_engine_t* counterfactual_engine_create_custom(const cf_config_t* config) {
    if (!config) { set_error("NULL config"); return NULL; }
    counterfactual_engine_t* e = nimcp_calloc(1, sizeof(counterfactual_engine_t));
    if (!e) return NULL;
    e->config = *config;
    e->next_cf_id = 1;
    return e;
}

void counterfactual_engine_destroy(counterfactual_engine_t* engine) {
    if (engine) nimcp_free(engine);
}

cf_state_t* counterfactual_create_state(const float* values, uint32_t dim,
    const char* description) {
    cf_state_t* s = nimcp_calloc(1, sizeof(cf_state_t));
    if (!s) return NULL;

    if (values && dim > 0) {
        s->values = nimcp_calloc(dim, sizeof(float));
        if (s->values) {
            memcpy(s->values, values, dim * sizeof(float));
            s->dim = dim;
        }
    }
    if (description) strncpy(s->description, description, sizeof(s->description) - 1);

    return s;
}

void counterfactual_free_state(cf_state_t* state) {
    if (!state) return;
    if (state->values) nimcp_free(state->values);
    nimcp_free(state);
}

cf_counterfactual_t* counterfactual_imagine(counterfactual_engine_t* engine,
    const cf_state_t* actual, const cf_intervention_t* what_if) {
    if (!engine || !actual || !what_if) return NULL;

    cf_counterfactual_t* cf = nimcp_calloc(1, sizeof(cf_counterfactual_t));
    if (!cf) return NULL;

    cf->id = engine->next_cf_id++;

    /* Copy actual state */
    cf->actual_world = counterfactual_create_state(actual->values, actual->dim, actual->description);

    /* Create counterfactual state with intervention applied */
    cf->counterfactual_world = counterfactual_create_state(actual->values, actual->dim,
        "Counterfactual world");
    if (cf->counterfactual_world && what_if->target_variable < cf->counterfactual_world->dim) {
        cf->counterfactual_world->values[what_if->target_variable] = what_if->counterfactual_value;
    }

    /* Copy intervention */
    cf->intervention = nimcp_calloc(1, sizeof(cf_intervention_t));
    if (cf->intervention) {
        memcpy(cf->intervention, what_if, sizeof(cf_intervention_t));
    }

    /* Trace consequences */
    cf->consequences = nimcp_calloc(CF_MAX_CONSEQUENCES, sizeof(cf_consequence_t));
    counterfactual_trace_effects(engine, cf, cf->consequences, CF_MAX_CONSEQUENCES, &cf->num_consequences);

    /* Compute distance and plausibility */
    cf->distance = counterfactual_distance(engine, actual, cf->counterfactual_world);
    cf->plausibility = apply_mod(engine, 1.0f / (1.0f + cf->distance));

    engine->stats.counterfactuals_generated++;
    engine->stats.interventions_analyzed++;

    return cf;
}

cf_counterfactual_t** counterfactual_explore_space(counterfactual_engine_t* engine,
    const cf_state_t* actual, uint32_t max_alternatives, uint32_t* num_found) {
    if (!engine || !actual || !num_found) return NULL;

    cf_counterfactual_t** cfs = nimcp_calloc(max_alternatives, sizeof(cf_counterfactual_t*));
    if (!cfs) return NULL;

    *num_found = 0;
    for (uint32_t i = 0; i < actual->dim && *num_found < max_alternatives; i++) {
        cf_intervention_t intervention = {
            .id = i + 1,
            .target_variable = i,
            .original_value = actual->values[i],
            .counterfactual_value = actual->values[i] * 1.5f
        };
        snprintf(intervention.description, sizeof(intervention.description),
                "Increase var %u", i);

        cfs[*num_found] = counterfactual_imagine(engine, actual, &intervention);
        if (cfs[*num_found]) (*num_found)++;
    }

    return cfs;
}

int counterfactual_trace_effects(counterfactual_engine_t* engine,
    const cf_counterfactual_t* cf, cf_consequence_t* consequences,
    uint32_t max_consequences, uint32_t* num_found) {
    if (!engine || !cf || !consequences || !num_found) return -1;
    if (!engine->config.enable_causal_tracing) { *num_found = 0; return 0; }

    *num_found = 0;

    if (!cf->actual_world || !cf->counterfactual_world) return 0;

    /* Find variables that differ between actual and counterfactual */
    for (uint32_t i = 0; i < cf->actual_world->dim && *num_found < max_consequences; i++) {
        float diff = cf->counterfactual_world->values[i] - cf->actual_world->values[i];
        if (fabsf(diff) > engine->config.consequence_threshold) {
            consequences[*num_found].id = *num_found + 1;
            consequences[*num_found].affected_variable = i;
            consequences[*num_found].magnitude = diff;
            consequences[*num_found].probability = apply_mod(engine, 0.8f);
            consequences[*num_found].is_direct = (cf->intervention &&
                cf->intervention->target_variable == i);
            snprintf(consequences[*num_found].description, sizeof(consequences[*num_found].description),
                    "Change in var %u: %.2f", i, diff);
            (*num_found)++;
            engine->stats.consequences_traced++;
        }
    }

    return 0;
}

float counterfactual_estimate_probability(counterfactual_engine_t* engine,
    const cf_counterfactual_t* cf) {
    if (!engine || !cf) return 0;
    return apply_mod(engine, cf->plausibility);
}

float counterfactual_causal_strength(counterfactual_engine_t* engine,
    uint32_t cause_var, uint32_t effect_var, const cf_state_t* context) {
    if (!engine || !context) return 0;
    if (cause_var >= context->dim || effect_var >= context->dim) return 0;

    /* Simple correlation-based estimate */
    float strength = fabsf(context->values[cause_var] * context->values[effect_var]);
    return apply_mod(engine, fminf(1.0f, strength));
}

int counterfactual_find_causes(counterfactual_engine_t* engine,
    const cf_state_t* state, uint32_t effect_var,
    uint32_t* cause_vars, uint32_t max_causes, uint32_t* num_found) {
    if (!engine || !state || !cause_vars || !num_found) return -1;
    if (effect_var >= state->dim) return -1;

    *num_found = 0;
    for (uint32_t i = 0; i < state->dim && *num_found < max_causes; i++) {
        if (i == effect_var) continue;
        float strength = counterfactual_causal_strength(engine, i, effect_var, state);
        if (strength > 0.3f) {
            cause_vars[(*num_found)++] = i;
        }
    }

    return 0;
}

float counterfactual_distance(counterfactual_engine_t* engine,
    const cf_state_t* s1, const cf_state_t* s2) {
    if (!engine || !s1 || !s2) return 0;
    if (!s1->values || !s2->values) return 0;

    uint32_t min_dim = (s1->dim < s2->dim) ? s1->dim : s2->dim;
    float dist = 0;

    for (uint32_t i = 0; i < min_dim; i++) {
        float diff = s1->values[i] - s2->values[i];
        dist += diff * diff;
    }

    return sqrtf(dist);
}

cf_counterfactual_t* counterfactual_find_closest(counterfactual_engine_t* engine,
    const cf_state_t* actual, const cf_state_t* target) {
    if (!engine || !actual || !target) return NULL;

    /* Create intervention that moves toward target */
    uint32_t best_var = 0;
    float best_improvement = 0;

    for (uint32_t i = 0; i < actual->dim && i < target->dim; i++) {
        float improvement = fabsf(target->values[i] - actual->values[i]);
        if (improvement > best_improvement) {
            best_improvement = improvement;
            best_var = i;
        }
    }

    cf_intervention_t intervention = {
        .id = 1,
        .target_variable = best_var,
        .original_value = actual->values[best_var],
        .counterfactual_value = target->values[best_var]
    };
    snprintf(intervention.description, sizeof(intervention.description),
            "Move var %u toward target", best_var);

    return counterfactual_imagine(engine, actual, &intervention);
}

void counterfactual_free(cf_counterfactual_t* cf) {
    if (!cf) return;
    if (cf->actual_world) counterfactual_free_state(cf->actual_world);
    if (cf->counterfactual_world) counterfactual_free_state(cf->counterfactual_world);
    if (cf->intervention) nimcp_free(cf->intervention);
    if (cf->consequences) nimcp_free(cf->consequences);
    nimcp_free(cf);
}

int counterfactual_set_inflammation(counterfactual_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->inflammation = fmaxf(0, fminf(1, level));
    return 0;
}

int counterfactual_set_fatigue(counterfactual_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->fatigue = fmaxf(0, fminf(1, level));
    return 0;
}

int counterfactual_get_stats(const counterfactual_engine_t* engine, cf_stats_t* stats) {
    if (!engine || !stats) return -1;
    *stats = engine->stats;
    return 0;
}

void counterfactual_reset_stats(counterfactual_engine_t* engine) {
    if (engine) memset(&engine->stats, 0, sizeof(engine->stats));
}

const char* counterfactual_get_last_error(void) { return g_last_error; }
