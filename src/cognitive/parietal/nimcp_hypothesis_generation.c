/**
 * @file nimcp_hypothesis_generation.c
 * @brief Hypothesis generation engine implementation
 */

#include "cognitive/parietal/nimcp_hypothesis_generation.h"
#include "utils/memory/nimcp_memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

struct hypothesis_engine {
    hypogen_config_t config;
    hypogen_stats_t stats;
    float inflammation;
    float fatigue;
    uint32_t next_theory_id;
    uint32_t next_prediction_id;
};

static __thread char g_last_error[256] = {0};

static void set_error(const char* msg) {
    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
}

static float apply_mod(const hypothesis_engine_t* e, float v) {
    float f = 1.0f - e->inflammation * e->config.inflammation_sensitivity * 0.2f
                   - e->fatigue * e->config.fatigue_sensitivity * 0.3f;
    return v * fmaxf(0.4f, f);
}

hypogen_config_t hypothesis_engine_default_config(void) {
    return (hypogen_config_t){
        .min_explanatory_power = 0.3f,
        .parsimony_weight = 0.3f,
        .novelty_bonus = 0.1f,
        .enable_abduction = true,
        .enable_prediction = true,
        .max_hypotheses = 10,
        .inflammation_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f
    };
}

hypothesis_engine_t* hypothesis_engine_create(void) {
    hypogen_config_t c = hypothesis_engine_default_config();
    return hypothesis_engine_create_custom(&c);
}

hypothesis_engine_t* hypothesis_engine_create_custom(const hypogen_config_t* config) {
    if (!config) { set_error("NULL config"); return NULL; }
    hypothesis_engine_t* e = nimcp_calloc(1, sizeof(hypothesis_engine_t));
    if (!e) return NULL;
    e->config = *config;
    e->next_theory_id = 1;
    e->next_prediction_id = 1;
    return e;
}

void hypothesis_engine_destroy(hypothesis_engine_t* engine) {
    if (engine) nimcp_free(engine);
}

hypogen_theory_t** hypothesis_generate_explanations(hypothesis_engine_t* engine,
    const hypogen_observation_t* observations, uint32_t num_obs, uint32_t* num_theories) {
    if (!engine || !observations || !num_theories) return NULL;

    uint32_t max = engine->config.max_hypotheses;
    hypogen_theory_t** theories = nimcp_calloc(max, sizeof(hypogen_theory_t*));
    if (!theories) return NULL;

    *num_theories = 0;
    for (uint32_t i = 0; i < max && i < num_obs; i++) {
        hypogen_theory_t* t = nimcp_calloc(1, sizeof(hypogen_theory_t));
        if (!t) break;

        t->id = engine->next_theory_id++;
        snprintf(t->statement, sizeof(t->statement),
                "Theory explaining observation %u", i);
        t->explanatory_power = 0.5f + 0.3f * ((float)rand() / RAND_MAX);
        t->parsimony = 0.6f + 0.2f * ((float)rand() / RAND_MAX);
        t->falsifiability = 0.7f;
        t->prior = 0.5f;
        t->likelihood = apply_mod(engine, t->explanatory_power);
        t->posterior = t->prior * t->likelihood;

        theories[(*num_theories)++] = t;
        engine->stats.hypotheses_generated++;
    }

    return theories;
}

hypogen_theory_t* hypothesis_abductive_inference(hypothesis_engine_t* engine,
    const hypogen_observation_t* surprising_fact) {
    if (!engine || !surprising_fact) return NULL;
    if (!engine->config.enable_abduction) return NULL;

    hypogen_theory_t* t = nimcp_calloc(1, sizeof(hypogen_theory_t));
    if (!t) return NULL;

    t->id = engine->next_theory_id++;
    snprintf(t->statement, sizeof(t->statement),
            "Abductive explanation for: %s", surprising_fact->description);
    t->explanatory_power = apply_mod(engine, 0.7f);
    t->parsimony = 0.5f;
    t->falsifiability = 0.8f;
    t->prior = 0.3f;
    t->likelihood = 0.8f;
    t->posterior = t->prior * t->likelihood;

    engine->stats.hypotheses_generated++;
    return t;
}

int hypothesis_rank_theories(hypothesis_engine_t* engine,
    hypogen_theory_t** theories, uint32_t num_theories, uint32_t* rankings) {
    if (!engine || !theories || !rankings) return -1;

    for (uint32_t i = 0; i < num_theories; i++) rankings[i] = i;

    /* Bubble sort by posterior */
    for (uint32_t i = 0; i < num_theories - 1; i++) {
        for (uint32_t j = 0; j < num_theories - i - 1; j++) {
            if (theories[rankings[j]]->posterior < theories[rankings[j+1]]->posterior) {
                uint32_t tmp = rankings[j];
                rankings[j] = rankings[j+1];
                rankings[j+1] = tmp;
            }
        }
    }
    return 0;
}

hypogen_prediction_t** hypothesis_derive_predictions(hypothesis_engine_t* engine,
    const hypogen_theory_t* theory, uint32_t* num_predictions) {
    if (!engine || !theory || !num_predictions) return NULL;
    if (!engine->config.enable_prediction) return NULL;

    uint32_t n = 3;  /* Generate 3 predictions */
    hypogen_prediction_t** preds = nimcp_calloc(n, sizeof(hypogen_prediction_t*));
    if (!preds) return NULL;

    *num_predictions = 0;
    for (uint32_t i = 0; i < n; i++) {
        hypogen_prediction_t* p = nimcp_calloc(1, sizeof(hypogen_prediction_t));
        if (!p) break;

        p->id = engine->next_prediction_id++;
        snprintf(p->description, sizeof(p->description),
                "Prediction %u from theory %u", i+1, theory->id);
        p->confidence = apply_mod(engine, theory->posterior * 0.8f);
        p->is_testable = true;
        p->is_novel = (i == 0);  /* First prediction is novel */

        preds[(*num_predictions)++] = p;
        engine->stats.predictions_made++;
    }

    return preds;
}

int hypothesis_test_prediction(hypothesis_engine_t* engine,
    hypogen_prediction_t* prediction, const hypogen_observation_t* observation) {
    if (!engine || !prediction || !observation) return -1;

    /* Simple test: compare confidence */
    if (prediction->confidence > 0.5f && observation->confidence > 0.5f) {
        engine->stats.hypotheses_confirmed++;
        return 1;  /* Confirmed */
    }
    engine->stats.hypotheses_rejected++;
    return 0;  /* Rejected */
}

hypogen_theory_t* hypothesis_revise_theory(hypothesis_engine_t* engine,
    hypogen_theory_t* theory, const hypogen_observation_t* new_evidence) {
    if (!engine || !theory || !new_evidence) return NULL;

    /* Bayesian update */
    float likelihood = new_evidence->confidence;
    theory->posterior = theory->posterior * likelihood;
    theory->posterior = fmaxf(0.01f, fminf(0.99f, theory->posterior));

    return theory;
}

int hypothesis_evaluate_theory(hypothesis_engine_t* engine,
    const hypogen_theory_t* theory, float* score) {
    if (!engine || !theory || !score) return -1;

    *score = theory->explanatory_power * 0.4f +
             theory->parsimony * engine->config.parsimony_weight +
             theory->falsifiability * 0.2f +
             theory->posterior * 0.1f;
    *score = apply_mod(engine, *score);

    engine->stats.avg_explanatory_power =
        (engine->stats.avg_explanatory_power * engine->stats.hypotheses_generated +
         theory->explanatory_power) / (engine->stats.hypotheses_generated + 1);

    return 0;
}

void hypothesis_free_theory(hypogen_theory_t* theory) {
    if (!theory) return;
    if (theory->parameters) nimcp_free(theory->parameters);
    if (theory->assumptions) {
        for (uint32_t i = 0; i < theory->num_assumptions; i++) {
            if (theory->assumptions[i]) nimcp_free(theory->assumptions[i]);
        }
        nimcp_free(theory->assumptions);
    }
    nimcp_free(theory);
}

void hypothesis_free_prediction(hypogen_prediction_t* prediction) {
    if (!prediction) return;
    if (prediction->predicted_values) nimcp_free(prediction->predicted_values);
    nimcp_free(prediction);
}

int hypothesis_set_inflammation(hypothesis_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->inflammation = fmaxf(0, fminf(1, level));
    return 0;
}

int hypothesis_set_fatigue(hypothesis_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->fatigue = fmaxf(0, fminf(1, level));
    return 0;
}

int hypothesis_get_stats(const hypothesis_engine_t* engine, hypogen_stats_t* stats) {
    if (!engine || !stats) return -1;
    *stats = engine->stats;
    return 0;
}

void hypothesis_reset_stats(hypothesis_engine_t* engine) {
    if (engine) memset(&engine->stats, 0, sizeof(engine->stats));
}

const char* hypothesis_get_last_error(void) { return g_last_error; }
