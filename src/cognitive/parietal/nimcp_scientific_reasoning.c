/**
 * @file nimcp_scientific_reasoning.c
 * @brief Scientific reasoning and hypothesis testing implementation
 *
 * Implements dimensional analysis, hypothesis testing with Bayesian
 * updating, and basic causal inference for the parietal lobe module.
 */

#include "cognitive/parietal/nimcp_scientific_reasoning.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/thread/nimcp_thread.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define EPSILON 1e-6f
#define PI 3.14159265358979323846f

/* ============================================================================
 * INTERNAL STRUCTURES
 * ============================================================================ */

/**
 * @brief Internal scientific reasoning state
 */
struct scientific_reasoning {
    /* Configuration */
    scientific_config_t config;

    /* Hypothesis tracking */
    hypothesis_t* hypotheses;
    uint32_t num_hypotheses;
    uint32_t next_hypothesis_id;

    /* Modulation state */
    float inflammation_level;
    float sleep_deprivation_level;

    /* Statistics */
    uint64_t hypotheses_generated;
    uint64_t hypotheses_rejected;
    uint64_t dimensional_analyses;
    uint64_t causal_inferences;
    double total_posterior;

    /* Thread safety */
    nimcp_mutex_t* lock;
};

/* Thread-local error message */
static _Thread_local char g_scientific_error[256] = {0};

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

static void set_scientific_error(const char* msg) {
    strncpy(g_scientific_error, msg, sizeof(g_scientific_error) - 1);
    g_scientific_error[sizeof(g_scientific_error) - 1] = '\0';
}

static float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static float compute_correlation(const float* x, const float* y, uint32_t n) {
    if (n < 2) return 0.0f;

    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;

    for (uint32_t i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    float num = (float)n * sum_xy - sum_x * sum_y;
    float den = sqrtf(((float)n * sum_x2 - sum_x * sum_x) *
                      ((float)n * sum_y2 - sum_y * sum_y));

    if (fabsf(den) < EPSILON) return 0.0f;

    return num / den;
}

/* ============================================================================
 * LIFECYCLE API
 * ============================================================================ */

scientific_config_t scientific_default_config(void) {
    scientific_config_t config = {
        .hypothesis_prior_default = 0.5f,
        .evidence_threshold = 0.1f,
        .significance_level = 0.05f,
        .max_hypotheses = SCIENTIFIC_MAX_HYPOTHESES,
        .enable_causal_inference = true,
        .enable_bio_async = false,
        .inflammation_sensitivity = 0.5f,
        .sleep_deprivation_effect = 0.5f
    };
    return config;
}

bool scientific_validate_config(const scientific_config_t* config) {
    if (!config) return false;

    if (config->hypothesis_prior_default < 0.0f ||
        config->hypothesis_prior_default > 1.0f) {
        set_scientific_error("Prior default must be in [0, 1]");
        return false;
    }

    if (config->evidence_threshold < 0.0f || config->evidence_threshold > 1.0f) {
        set_scientific_error("Evidence threshold must be in [0, 1]");
        return false;
    }

    if (config->significance_level <= 0.0f || config->significance_level > 0.5f) {
        set_scientific_error("Significance level must be in (0, 0.5]");
        return false;
    }

    if (config->max_hypotheses == 0) {
        set_scientific_error("Max hypotheses cannot be 0");
        return false;
    }

    return true;
}

scientific_reasoning_t* scientific_reasoning_create(void) {
    return scientific_reasoning_create_custom(NULL);
}

scientific_reasoning_t* scientific_reasoning_create_custom(
    const scientific_config_t* config
) {
    scientific_config_t cfg;

    if (config) {
        if (!scientific_validate_config(config)) {
            return NULL;
        }
        cfg = *config;
    } else {
        cfg = scientific_default_config();
    }

    scientific_reasoning_t* sr = calloc(1, sizeof(scientific_reasoning_t));
    if (!sr) {
        set_scientific_error("Failed to allocate scientific reasoning");
        return NULL;
    }

    sr->config = cfg;
    sr->next_hypothesis_id = 1;

    /* Allocate hypothesis array */
    sr->hypotheses = calloc(cfg.max_hypotheses, sizeof(hypothesis_t));
    if (!sr->hypotheses) {
        set_scientific_error("Failed to allocate hypothesis array");
        free(sr);
        return NULL;
    }

    /* Create mutex */
    mutex_attr_t attr = {.type = MUTEX_TYPE_NORMAL};
    sr->lock = nimcp_mutex_create(&attr);
    if (!sr->lock) {
        set_scientific_error("Failed to create mutex");
        free(sr->hypotheses);
        free(sr);
        return NULL;
    }

    return sr;
}

void scientific_reasoning_destroy(scientific_reasoning_t* sr) {
    if (!sr) return;

    /* Free hypothesis parameters */
    for (uint32_t i = 0; i < sr->num_hypotheses; i++) {
        if (sr->hypotheses[i].parameters) {
            free(sr->hypotheses[i].parameters);
        }
    }

    free(sr->hypotheses);

    if (sr->lock) {
        nimcp_mutex_free(sr->lock);
    }

    free(sr);
}

/* ============================================================================
 * DIMENSIONAL ANALYSIS API
 * ============================================================================ */

physical_dimension_t scientific_dimensionless(void) {
    physical_dimension_t d = {0, 0, 0, 0, 0, 0, 0};
    return d;
}

physical_quantity_t scientific_create_quantity(
    float value,
    physical_dimension_t dimension,
    const char* symbol
) {
    physical_quantity_t q;
    q.value = value;
    q.dimension = dimension;

    if (symbol) {
        strncpy(q.symbol, symbol, sizeof(q.symbol) - 1);
        q.symbol[sizeof(q.symbol) - 1] = '\0';
    } else {
        q.symbol[0] = '\0';
    }

    return q;
}

physical_dimension_t scientific_multiply_dimensions(
    physical_dimension_t a,
    physical_dimension_t b
) {
    physical_dimension_t result;
    result.length = a.length + b.length;
    result.mass = a.mass + b.mass;
    result.time = a.time + b.time;
    result.current = a.current + b.current;
    result.temperature = a.temperature + b.temperature;
    result.amount = a.amount + b.amount;
    result.luminosity = a.luminosity + b.luminosity;
    return result;
}

physical_dimension_t scientific_divide_dimensions(
    physical_dimension_t a,
    physical_dimension_t b
) {
    physical_dimension_t result;
    result.length = a.length - b.length;
    result.mass = a.mass - b.mass;
    result.time = a.time - b.time;
    result.current = a.current - b.current;
    result.temperature = a.temperature - b.temperature;
    result.amount = a.amount - b.amount;
    result.luminosity = a.luminosity - b.luminosity;
    return result;
}

physical_dimension_t scientific_power_dimension(
    physical_dimension_t d,
    int8_t power
) {
    physical_dimension_t result;
    result.length = d.length * power;
    result.mass = d.mass * power;
    result.time = d.time * power;
    result.current = d.current * power;
    result.temperature = d.temperature * power;
    result.amount = d.amount * power;
    result.luminosity = d.luminosity * power;
    return result;
}

bool scientific_dimensions_equal(
    physical_dimension_t a,
    physical_dimension_t b
) {
    return a.length == b.length &&
           a.mass == b.mass &&
           a.time == b.time &&
           a.current == b.current &&
           a.temperature == b.temperature &&
           a.amount == b.amount &&
           a.luminosity == b.luminosity;
}

bool scientific_is_dimensionless(physical_dimension_t d) {
    return d.length == 0 && d.mass == 0 && d.time == 0 &&
           d.current == 0 && d.temperature == 0 &&
           d.amount == 0 && d.luminosity == 0;
}

const char* scientific_dimension_to_string(
    physical_dimension_t d,
    char* buffer,
    uint32_t buffer_size
) {
    if (!buffer || buffer_size < 64) return NULL;

    char* p = buffer;
    *p = '\0';

    if (scientific_is_dimensionless(d)) {
        strcpy(buffer, "1");
        return buffer;
    }

    const char* units[] = {"m", "kg", "s", "A", "K", "mol", "cd"};
    int8_t exps[] = {d.length, d.mass, d.time, d.current,
                     d.temperature, d.amount, d.luminosity};

    bool first = true;
    for (int i = 0; i < 7; i++) {
        if (exps[i] != 0) {
            if (!first) {
                strcat(p, "*");
            }
            first = false;

            if (exps[i] == 1) {
                strcat(p, units[i]);
            } else {
                char tmp[16];
                snprintf(tmp, sizeof(tmp), "%s^%d", units[i], exps[i]);
                strcat(p, tmp);
            }
        }
    }

    return buffer;
}

uint32_t scientific_buckingham_pi(
    scientific_reasoning_t* sr,
    const physical_quantity_t* quantities,
    uint32_t num_quantities,
    float** pi_groups,
    uint32_t max_groups
) {
    if (!sr || !quantities || !pi_groups || num_quantities < 2) {
        return 0;
    }

    nimcp_mutex_lock(sr->lock);

    /* Build dimension matrix [7 x num_quantities] */
    int8_t dim_matrix[7][16];  /* Max 16 quantities */

    if (num_quantities > 16) {
        nimcp_mutex_unlock(sr->lock);
        return 0;
    }

    for (uint32_t j = 0; j < num_quantities; j++) {
        dim_matrix[0][j] = quantities[j].dimension.length;
        dim_matrix[1][j] = quantities[j].dimension.mass;
        dim_matrix[2][j] = quantities[j].dimension.time;
        dim_matrix[3][j] = quantities[j].dimension.current;
        dim_matrix[4][j] = quantities[j].dimension.temperature;
        dim_matrix[5][j] = quantities[j].dimension.amount;
        dim_matrix[6][j] = quantities[j].dimension.luminosity;
    }

    /* Count non-zero dimensions (rank of dimension matrix) */
    uint32_t rank = 0;
    for (int i = 0; i < 7; i++) {
        bool non_zero = false;
        for (uint32_t j = 0; j < num_quantities; j++) {
            if (dim_matrix[i][j] != 0) {
                non_zero = true;
                break;
            }
        }
        if (non_zero) rank++;
    }

    /* Number of dimensionless groups = n - rank */
    uint32_t num_groups = (num_quantities > rank) ? num_quantities - rank : 0;
    if (num_groups > max_groups) num_groups = max_groups;

    /* Simple approach: find combinations that yield dimensionless products */
    /* For a proper implementation, use null space computation */

    /* Allocate output */
    for (uint32_t g = 0; g < num_groups; g++) {
        pi_groups[g] = calloc(num_quantities, sizeof(float));
        if (!pi_groups[g]) {
            /* Clean up on failure */
            for (uint32_t k = 0; k < g; k++) {
                free(pi_groups[k]);
            }
            nimcp_mutex_unlock(sr->lock);
            return 0;
        }

        /* Simple heuristic: first quantities are repeating, others get exponent 1 */
        /* This is a placeholder - real implementation needs null space */
        pi_groups[g][g + rank] = 1.0f;
    }

    sr->dimensional_analyses++;

    nimcp_mutex_unlock(sr->lock);

    return num_groups;
}

/* ============================================================================
 * HYPOTHESIS API
 * ============================================================================ */

hypothesis_t scientific_create_hypothesis(
    scientific_reasoning_t* sr,
    const char* description,
    float prior
) {
    hypothesis_t h = {0};

    if (!sr || !description) {
        return h;
    }

    nimcp_mutex_lock(sr->lock);

    h.id = sr->next_hypothesis_id++;
    strncpy(h.description, description, SCIENTIFIC_MAX_DESCRIPTION - 1);
    h.description[SCIENTIFIC_MAX_DESCRIPTION - 1] = '\0';

    h.prior = clamp01(prior);
    h.posterior = h.prior;
    h.likelihood = 1.0f;
    h.evidence_strength = 0.0f;
    h.observations = 0;
    h.active = true;

    /* Add to tracking array if space available */
    if (sr->num_hypotheses < sr->config.max_hypotheses) {
        sr->hypotheses[sr->num_hypotheses++] = h;
    }

    sr->hypotheses_generated++;

    nimcp_mutex_unlock(sr->lock);

    return h;
}

float scientific_update_hypothesis(
    scientific_reasoning_t* sr,
    hypothesis_t* hypothesis,
    const data_sample_t* samples,
    uint32_t num_samples
) {
    if (!sr || !hypothesis || !samples || num_samples == 0) {
        return 0.0f;
    }

    nimcp_mutex_lock(sr->lock);

    /* Apply cognitive impairment from inflammation/sleep */
    float cognitive_factor = 1.0f -
        sr->inflammation_level * sr->config.inflammation_sensitivity * 0.2f -
        sr->sleep_deprivation_level * sr->config.sleep_deprivation_effect * 0.15f;

    /* Compute likelihood from data */
    /* Simplified: use variance as evidence */
    float total_evidence = 0.0f;
    uint32_t total_values = 0;

    for (uint32_t i = 0; i < num_samples; i++) {
        const data_sample_t* s = &samples[i];

        /* Compute sample variance */
        if (s->num_values > 1) {
            float mean = 0.0f;
            for (uint32_t j = 0; j < s->num_values; j++) {
                mean += s->values[j];
            }
            mean /= (float)s->num_values;

            float var = 0.0f;
            for (uint32_t j = 0; j < s->num_values; j++) {
                float diff = s->values[j] - mean;
                var += diff * diff;
            }
            var /= (float)(s->num_values - 1);

            /* Low variance = high evidence for hypothesis */
            float evidence = 1.0f / (1.0f + sqrtf(var));
            total_evidence += evidence * s->weight;
            total_values += s->num_values;
        }
    }

    if (total_values > 0) {
        hypothesis->evidence_strength += total_evidence * cognitive_factor;
    }

    /* Bayesian update */
    /* P(H|D) ∝ P(D|H) * P(H) */
    float likelihood = total_evidence / (float)num_samples;
    likelihood = 0.5f + 0.5f * likelihood;  /* Scale to [0.5, 1] */

    hypothesis->likelihood = likelihood;
    hypothesis->observations += num_samples;

    /* Update posterior */
    float prior = hypothesis->posterior;  /* Use current posterior as prior */
    float evidence_factor = likelihood * cognitive_factor;

    /* Simplified Bayesian update */
    hypothesis->posterior = prior * evidence_factor /
                            (prior * evidence_factor + (1.0f - prior) * (1.0f - evidence_factor + 0.1f));

    hypothesis->posterior = clamp01(hypothesis->posterior);

    sr->total_posterior += hypothesis->posterior;

    nimcp_mutex_unlock(sr->lock);

    return hypothesis->posterior;
}

float scientific_compare_hypotheses(
    scientific_reasoning_t* sr,
    const hypothesis_t* h1,
    const hypothesis_t* h2
) {
    if (!sr || !h1 || !h2) return 1.0f;

    /* Bayes factor: ratio of posteriors divided by ratio of priors */
    float post_ratio = (h1->posterior + EPSILON) / (h2->posterior + EPSILON);
    float prior_ratio = (h1->prior + EPSILON) / (h2->prior + EPSILON);

    return post_ratio / prior_ratio;
}

uint32_t scientific_best_hypothesis(
    scientific_reasoning_t* sr,
    const hypothesis_t* hypotheses,
    uint32_t num_hypotheses
) {
    if (!sr || !hypotheses || num_hypotheses == 0) return 0;

    uint32_t best_idx = 0;
    float best_posterior = hypotheses[0].posterior;

    for (uint32_t i = 1; i < num_hypotheses; i++) {
        if (hypotheses[i].active && hypotheses[i].posterior > best_posterior) {
            best_posterior = hypotheses[i].posterior;
            best_idx = i;
        }
    }

    return best_idx;
}

bool scientific_reject_hypothesis(
    scientific_reasoning_t* sr,
    hypothesis_t* hypothesis
) {
    if (!sr || !hypothesis) return false;

    nimcp_mutex_lock(sr->lock);

    /* Reject if posterior is below significance level */
    bool reject = hypothesis->posterior < sr->config.significance_level &&
                  hypothesis->observations >= 10;  /* Need enough observations */

    if (reject) {
        hypothesis->active = false;
        sr->hypotheses_rejected++;
    }

    nimcp_mutex_unlock(sr->lock);

    return reject;
}

/* ============================================================================
 * CAUSAL INFERENCE API
 * ============================================================================ */

causal_graph_t* scientific_create_causal_graph(
    scientific_reasoning_t* sr,
    const char** variable_names,
    uint32_t num_variables
) {
    if (!sr || !variable_names || num_variables == 0 ||
        num_variables > SCIENTIFIC_MAX_CAUSAL_VARS) {
        return NULL;
    }

    causal_graph_t* graph = calloc(1, sizeof(causal_graph_t));
    if (!graph) {
        set_scientific_error("Failed to allocate causal graph");
        return NULL;
    }

    graph->num_variables = num_variables;

    /* Copy variable names */
    for (uint32_t i = 0; i < num_variables; i++) {
        if (variable_names[i]) {
            graph->variable_names[i] = strdup(variable_names[i]);
        }
    }

    /* Allocate adjacency matrix */
    graph->adjacency = malloc(num_variables * sizeof(float*));
    if (!graph->adjacency) {
        scientific_destroy_causal_graph(graph);
        return NULL;
    }

    for (uint32_t i = 0; i < num_variables; i++) {
        graph->adjacency[i] = calloc(num_variables, sizeof(float));
        if (!graph->adjacency[i]) {
            scientific_destroy_causal_graph(graph);
            return NULL;
        }
    }

    /* Allocate relations array */
    graph->relations = calloc(num_variables * num_variables, sizeof(causal_relation_t));

    return graph;
}

void scientific_destroy_causal_graph(causal_graph_t* graph) {
    if (!graph) return;

    for (uint32_t i = 0; i < graph->num_variables; i++) {
        free(graph->variable_names[i]);
        if (graph->adjacency) {
            free(graph->adjacency[i]);
        }
    }

    free(graph->adjacency);
    free(graph->relations);
    free(graph);
}

int scientific_learn_causal_structure(
    scientific_reasoning_t* sr,
    causal_graph_t* graph,
    const float** data,
    uint32_t num_samples
) {
    if (!sr || !graph || !data || num_samples < 10) {
        return -1;
    }

    nimcp_mutex_lock(sr->lock);

    uint32_t n = graph->num_variables;

    /* Phase 1: Compute correlation matrix and create skeleton */
    float** corr = malloc(n * sizeof(float*));
    for (uint32_t i = 0; i < n; i++) {
        corr[i] = calloc(n, sizeof(float));
    }

    /* Extract column data for correlation computation */
    for (uint32_t i = 0; i < n; i++) {
        float* col_i = malloc(num_samples * sizeof(float));
        for (uint32_t s = 0; s < num_samples; s++) {
            col_i[s] = data[s][i];
        }

        for (uint32_t j = i + 1; j < n; j++) {
            float* col_j = malloc(num_samples * sizeof(float));
            for (uint32_t s = 0; s < num_samples; s++) {
                col_j[s] = data[s][j];
            }

            float r = compute_correlation(col_i, col_j, num_samples);
            corr[i][j] = r;
            corr[j][i] = r;

            /* If significant correlation, add edge */
            if (fabsf(r) > 0.3f) {
                graph->adjacency[i][j] = r;
                graph->adjacency[j][i] = r;
            }

            free(col_j);
        }
        free(col_i);
    }

    /* Phase 2: Orient edges (simplified - use temporal ordering or domain knowledge) */
    /* For now, orient based on correlation strength asymmetry */
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = i + 1; j < n; j++) {
            if (graph->adjacency[i][j] != 0.0f) {
                /* Simple heuristic: lower index is cause */
                scientific_add_causal_relation(graph, i, j, graph->adjacency[i][j]);
            }
        }
    }

    /* Cleanup */
    for (uint32_t i = 0; i < n; i++) {
        free(corr[i]);
    }
    free(corr);

    sr->causal_inferences++;

    nimcp_mutex_unlock(sr->lock);

    return 0;
}

int scientific_add_causal_relation(
    causal_graph_t* graph,
    uint32_t cause_id,
    uint32_t effect_id,
    float strength
) {
    if (!graph || cause_id >= graph->num_variables ||
        effect_id >= graph->num_variables) {
        return -1;
    }

    /* Add to relations array */
    graph->relations[graph->num_relations].cause_id = cause_id;
    graph->relations[graph->num_relations].effect_id = effect_id;
    graph->relations[graph->num_relations].strength = strength;
    graph->relations[graph->num_relations].confidence = fabsf(strength);
    graph->relations[graph->num_relations].is_direct = true;
    graph->num_relations++;

    return 0;
}

float scientific_estimate_causal_effect(
    scientific_reasoning_t* sr,
    const causal_graph_t* graph,
    uint32_t treatment_id,
    uint32_t outcome_id,
    float treatment_value
) {
    if (!sr || !graph || treatment_id >= graph->num_variables ||
        outcome_id >= graph->num_variables) {
        return 0.0f;
    }

    nimcp_mutex_lock(sr->lock);

    /* Find direct effect from adjacency matrix */
    float direct_effect = graph->adjacency[treatment_id][outcome_id];

    /* Apply do-calculus: remove incoming edges to treatment */
    /* Simplified: just use direct effect scaled by treatment value */
    float causal_effect = direct_effect * treatment_value;

    sr->causal_inferences++;

    nimcp_mutex_unlock(sr->lock);

    return causal_effect;
}

bool scientific_is_path_blocked(
    const causal_graph_t* graph,
    uint32_t from_id,
    uint32_t to_id,
    const uint32_t* conditioning_set,
    uint32_t num_conditioning
) {
    if (!graph || from_id >= graph->num_variables ||
        to_id >= graph->num_variables) {
        return true;  /* Invalid = blocked */
    }

    /* Simplified d-separation check */
    /* A path is blocked if any non-collider is in conditioning set */

    /* Check if direct path exists */
    if (graph->adjacency[from_id][to_id] != 0.0f) {
        /* Check if to_id is in conditioning set */
        for (uint32_t i = 0; i < num_conditioning; i++) {
            if (conditioning_set[i] == to_id) {
                return true;  /* Blocked by conditioning on intermediate */
            }
        }
        return false;  /* Direct path, not blocked */
    }

    return true;  /* No direct path */
}

/* ============================================================================
 * EXPERIMENTAL DESIGN API
 * ============================================================================ */

int scientific_suggest_experiment(
    scientific_reasoning_t* sr,
    const causal_graph_t* graph,
    uint32_t target_effect_id,
    experimental_design_t* design
) {
    if (!sr || !graph || !design || target_effect_id >= graph->num_variables) {
        return -1;
    }

    memset(design, 0, sizeof(experimental_design_t));
    strcpy(design->name, "Suggested Experiment");

    /* Find causes of target */
    design->treatment_vars = malloc(graph->num_variables * sizeof(uint32_t));
    design->control_vars = malloc(graph->num_variables * sizeof(uint32_t));
    design->outcome_vars = malloc(sizeof(uint32_t));

    design->outcome_vars[0] = target_effect_id;
    design->num_outcomes = 1;

    for (uint32_t i = 0; i < graph->num_relations; i++) {
        if (graph->relations[i].effect_id == target_effect_id) {
            design->treatment_vars[design->num_treatments++] = graph->relations[i].cause_id;
        }
    }

    /* Suggest sample size based on effect size assumption */
    design->sample_size = scientific_required_sample_size(sr, 0.5f, 0.8f, 0.05f);
    design->power = 0.8f;

    return 0;
}

uint32_t scientific_required_sample_size(
    scientific_reasoning_t* sr,
    float effect_size,
    float power,
    float alpha
) {
    if (!sr) return 100;

    /* Cohen's formula approximation for two-sample t-test */
    /* n = 2 * ((z_alpha + z_beta) / d)^2 */

    /* Z-scores for common values */
    float z_alpha = 1.96f;  /* alpha = 0.05 */
    if (alpha > 0.1f) z_alpha = 1.28f;
    else if (alpha < 0.01f) z_alpha = 2.58f;

    float z_beta = 0.84f;   /* power = 0.8 */
    if (power > 0.9f) z_beta = 1.28f;
    else if (power < 0.7f) z_beta = 0.52f;

    float d = effect_size;
    if (d < 0.1f) d = 0.1f;

    float ratio = (z_alpha + z_beta) / d;
    uint32_t n = (uint32_t)(2.0f * ratio * ratio + 0.5f);

    if (n < 10) n = 10;
    if (n > 10000) n = 10000;

    return n;
}

/* ============================================================================
 * MODULATION API
 * ============================================================================ */

int scientific_set_inflammation(scientific_reasoning_t* sr, float level) {
    if (!sr) return -1;

    nimcp_mutex_lock(sr->lock);
    sr->inflammation_level = clamp01(level);
    nimcp_mutex_unlock(sr->lock);

    return 0;
}

int scientific_set_sleep_deprivation(scientific_reasoning_t* sr, float level) {
    if (!sr) return -1;

    nimcp_mutex_lock(sr->lock);
    sr->sleep_deprivation_level = clamp01(level);
    nimcp_mutex_unlock(sr->lock);

    return 0;
}

/* ============================================================================
 * STATISTICS API
 * ============================================================================ */

int scientific_get_stats(const scientific_reasoning_t* sr, scientific_stats_t* stats) {
    if (!sr || !stats) return -1;

    nimcp_mutex_lock(((scientific_reasoning_t*)sr)->lock);

    stats->hypotheses_generated = sr->hypotheses_generated;
    stats->hypotheses_rejected = sr->hypotheses_rejected;
    stats->dimensional_analyses = sr->dimensional_analyses;
    stats->causal_inferences = sr->causal_inferences;

    /* Count active hypotheses */
    stats->active_hypotheses = 0;
    for (uint32_t i = 0; i < sr->num_hypotheses; i++) {
        if (sr->hypotheses[i].active) {
            stats->active_hypotheses++;
        }
    }

    if (sr->hypotheses_generated > 0) {
        stats->avg_posterior = (float)(sr->total_posterior /
                                        (double)sr->hypotheses_generated);
    } else {
        stats->avg_posterior = 0.0f;
    }

    nimcp_mutex_unlock(((scientific_reasoning_t*)sr)->lock);

    return 0;
}

void scientific_reset_stats(scientific_reasoning_t* sr) {
    if (!sr) return;

    nimcp_mutex_lock(sr->lock);

    sr->hypotheses_generated = 0;
    sr->hypotheses_rejected = 0;
    sr->dimensional_analyses = 0;
    sr->causal_inferences = 0;
    sr->total_posterior = 0.0;

    nimcp_mutex_unlock(sr->lock);
}

const char* scientific_get_last_error(void) {
    return g_scientific_error;
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int scientific_reasoning_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Scientific_Reasoning");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Module self-knowledge logged */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Scientific_Reasoning");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Scientific_Reasoning");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
