/**
 * @file nimcp_intuitive_reasoning.c
 * @brief Intuitive reasoning engine implementation
 *
 * WHAT: Core engine for intuitive, heuristic-based reasoning
 * WHY:  Enable formation of NEW knowledge through educated guesses
 * HOW:  Pattern recognition, plausibility estimation, incubation
 *
 * IMPLEMENTATION NOTES:
 * - Uses neural pattern matching for hunch formation
 * - Employs Bayesian-like updates for plausibility
 * - Implements background incubation queue
 * - Supports gestalt perception principles
 */

#include "cognitive/parietal/nimcp_intuitive_reasoning.h"
#include "utils/memory/nimcp_memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * INTERNAL CONSTANTS
 * ============================================================================ */

#define MAX_INCUBATION_QUEUE    32
#define MAX_PATTERN_MEMORY      256
#define PATTERN_MATCH_THRESHOLD 0.6f
#define CONFIDENCE_DECAY_BASE   0.95f
#define MIN_HUNCH_PLAUSIBILITY  0.1f

/* ============================================================================
 * INTERNAL TYPES
 * ============================================================================ */

/**
 * @brief Stored pattern for matching
 */
typedef struct {
    uint32_t id;
    char name[64];
    float* data;
    uint32_t dim;
    uint32_t match_count;
    float avg_match_strength;
} stored_pattern_t;

/**
 * @brief Incubating problem entry
 */
typedef struct {
    uint32_t id;
    problem_t problem;
    float progress;
    uint32_t iterations;
    float* partial_solution;
    uint32_t solution_dim;
    bool ready;
    insight_t* result;
} incubation_entry_t;

/**
 * @brief Internal engine structure
 */
struct intuitive_engine {
    intuitive_config_t config;
    intuitive_stats_t stats;

    /* Pattern memory */
    stored_pattern_t* patterns;
    uint32_t num_patterns;
    uint32_t pattern_capacity;
    uint32_t next_pattern_id;

    /* Incubation queue */
    incubation_entry_t* incubation_queue;
    uint32_t num_incubating;
    uint32_t next_incubation_id;

    /* Modulation state */
    float inflammation;
    float fatigue;
    float emotional_valence;

    /* Processing state */
    uint64_t last_process_time;
    float accumulated_insight_potential;
};

/* Thread-local error message */
static __thread char g_last_error[256] = {0};

/* ============================================================================
 * INTERNAL HELPERS
 * ============================================================================ */

static void set_error(const char* msg) {
    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
}

static uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

/**
 * @brief Compute cosine similarity between vectors
 */
static float cosine_similarity(const float* a, const float* b, uint32_t dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-10f) return 0.0f;

    return dot / denom;
}

/**
 * @brief Compute pattern entropy (measure of structure)
 */
static float compute_entropy(const float* data, uint32_t dim) {
    if (dim == 0) return 0.0f;

    /* Compute histogram-based entropy */
    const int NUM_BINS = 16;
    int bins[16] = {0};

    float min_val = data[0], max_val = data[0];
    for (uint32_t i = 1; i < dim; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    float range = max_val - min_val;
    if (range < 1e-10f) return 0.0f;  /* All same value = no entropy */

    for (uint32_t i = 0; i < dim; i++) {
        int bin = (int)((data[i] - min_val) / range * (NUM_BINS - 1));
        if (bin >= NUM_BINS) bin = NUM_BINS - 1;
        bins[bin]++;
    }

    float entropy = 0.0f;
    for (int i = 0; i < NUM_BINS; i++) {
        if (bins[i] > 0) {
            float p = (float)bins[i] / dim;
            entropy -= p * log2f(p);
        }
    }

    return entropy / log2f(NUM_BINS);  /* Normalize to [0,1] */
}

/**
 * @brief Detect trends in data
 */
static float detect_trend(const float* data, uint32_t length) {
    if (length < 2) return 0.0f;

    /* Simple linear regression slope */
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (uint32_t i = 0; i < length; i++) {
        sum_x += i;
        sum_y += data[i];
        sum_xy += i * data[i];
        sum_xx += i * i;
    }

    float n = (float)length;
    float denom = n * sum_xx - sum_x * sum_x;
    if (fabsf(denom) < 1e-10f) return 0.0f;

    float slope = (n * sum_xy - sum_x * sum_y) / denom;

    /* Normalize slope to [-1, 1] range */
    float range = 0.0f;
    for (uint32_t i = 0; i < length; i++) {
        float abs_val = fabsf(data[i]);
        if (abs_val > range) range = abs_val;
    }
    if (range < 1e-10f) return 0.0f;

    return fmaxf(-1.0f, fminf(1.0f, slope / range * length));
}

/**
 * @brief Detect periodicity in data
 */
static float detect_periodicity(const float* data, uint32_t length) {
    if (length < 4) return 0.0f;

    float max_autocorr = 0.0f;

    /* Check autocorrelation at different lags */
    for (uint32_t lag = 1; lag < length / 2; lag++) {
        float autocorr = 0.0f;
        uint32_t count = 0;

        for (uint32_t i = 0; i < length - lag; i++) {
            autocorr += data[i] * data[i + lag];
            count++;
        }

        if (count > 0) {
            autocorr /= count;
            if (autocorr > max_autocorr) {
                max_autocorr = autocorr;
            }
        }
    }

    /* Normalize */
    float variance = 0.0f;
    float mean = 0.0f;
    for (uint32_t i = 0; i < length; i++) mean += data[i];
    mean /= length;
    for (uint32_t i = 0; i < length; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= length;

    if (variance < 1e-10f) return 0.0f;
    return fmaxf(0.0f, fminf(1.0f, max_autocorr / variance));
}

/**
 * @brief Apply modulation to confidence
 */
static float apply_modulation(const intuitive_engine_t* engine, float value) {
    /* Inflammation increases caution (reduces confidence) */
    float inflammation_factor = 1.0f - engine->inflammation *
                                engine->config.inflammation_sensitivity * 0.3f;

    /* Fatigue reduces clarity */
    float fatigue_factor = 1.0f - engine->fatigue *
                           engine->config.fatigue_sensitivity * 0.4f;

    return value * inflammation_factor * fatigue_factor;
}

/* ============================================================================
 * LIFECYCLE IMPLEMENTATION
 * ============================================================================ */

intuitive_config_t intuitive_engine_default_config(void) {
    intuitive_config_t config = {
        .plausibility_threshold = 0.3f,
        .confidence_threshold = 0.5f,
        .novelty_weight = 0.2f,
        .coherence_weight = 0.4f,
        .fertility_weight = 0.2f,
        .enable_gestalt = true,
        .enable_incubation = true,
        .enable_emotional_markers = true,
        .prior_strength = 0.5f,
        .learning_rate = 0.1f,
        .inflammation_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f,
        .max_incubation_problems = MAX_INCUBATION_QUEUE,
        .pattern_memory_size = MAX_PATTERN_MEMORY
    };
    return config;
}

intuitive_engine_t* intuitive_engine_create(void) {
    intuitive_config_t config = intuitive_engine_default_config();
    return intuitive_engine_create_custom(&config);
}

intuitive_engine_t* intuitive_engine_create_custom(const intuitive_config_t* config) {
    if (!config) {
        set_error("NULL config");
        return NULL;
    }

    intuitive_engine_t* engine = nimcp_calloc(1, sizeof(intuitive_engine_t));
    if (!engine) {
        set_error("Failed to allocate engine");
        return NULL;
    }

    engine->config = *config;
    memset(&engine->stats, 0, sizeof(engine->stats));

    /* Allocate pattern memory */
    engine->pattern_capacity = config->pattern_memory_size;
    engine->patterns = nimcp_calloc(engine->pattern_capacity, sizeof(stored_pattern_t));
    if (!engine->patterns) {
        set_error("Failed to allocate pattern memory");
        nimcp_free(engine);
        return NULL;
    }
    engine->num_patterns = 0;
    engine->next_pattern_id = 1;

    /* Allocate incubation queue */
    engine->incubation_queue = nimcp_calloc(config->max_incubation_problems,
                                            sizeof(incubation_entry_t));
    if (!engine->incubation_queue) {
        set_error("Failed to allocate incubation queue");
        nimcp_free(engine->patterns);
        nimcp_free(engine);
        return NULL;
    }
    engine->num_incubating = 0;
    engine->next_incubation_id = 1;

    /* Initialize modulation state */
    engine->inflammation = 0.0f;
    engine->fatigue = 0.0f;
    engine->emotional_valence = 0.0f;

    engine->last_process_time = get_timestamp_us();
    engine->accumulated_insight_potential = 0.0f;

    return engine;
}

void intuitive_engine_destroy(intuitive_engine_t* engine) {
    if (!engine) return;

    /* Free pattern memory */
    if (engine->patterns) {
        for (uint32_t i = 0; i < engine->num_patterns; i++) {
            if (engine->patterns[i].data) {
                nimcp_free(engine->patterns[i].data);
            }
        }
        nimcp_free(engine->patterns);
    }

    /* Free incubation queue */
    if (engine->incubation_queue) {
        for (uint32_t i = 0; i < engine->num_incubating; i++) {
            incubation_entry_t* entry = &engine->incubation_queue[i];
            if (entry->partial_solution) {
                nimcp_free(entry->partial_solution);
            }
            if (entry->result) {
                intuitive_free_insight(entry->result);
            }
            if (entry->problem.initial_state) {
                nimcp_free(entry->problem.initial_state);
            }
            if (entry->problem.goal_state) {
                nimcp_free(entry->problem.goal_state);
            }
            if (entry->problem.constraints) {
                nimcp_free(entry->problem.constraints);
            }
        }
        nimcp_free(engine->incubation_queue);
    }

    nimcp_free(engine);
}

/* ============================================================================
 * HUNCH FORMATION IMPLEMENTATION
 * ============================================================================ */

hunch_t* intuitive_form_hunch(
    intuitive_engine_t* engine,
    const observation_t* observations,
    uint32_t num_observations
) {
    if (!engine || !observations || num_observations == 0) {
        set_error("Invalid parameters for hunch formation");
        return NULL;
    }

    uint64_t start_time = get_timestamp_us();

    hunch_t* hunch = nimcp_calloc(1, sizeof(hunch_t));
    if (!hunch) {
        set_error("Failed to allocate hunch");
        return NULL;
    }

    hunch->id = engine->stats.hunches_formed + 1;
    hunch->formation_time = get_timestamp_us();

    /* Analyze observations to extract pattern */
    uint32_t max_dim = 0;
    float total_salience = 0.0f;
    float total_reliability = 0.0f;

    for (uint32_t i = 0; i < num_observations; i++) {
        if (observations[i].dim > max_dim) {
            max_dim = observations[i].dim;
        }
        total_salience += observations[i].salience;
        total_reliability += observations[i].reliability;
    }

    if (max_dim == 0) {
        set_error("No observation data");
        nimcp_free(hunch);
        return NULL;
    }

    /* Allocate pattern storage */
    hunch->predicted_pattern = nimcp_calloc(max_dim, sizeof(float));
    hunch->pattern_dim = max_dim;

    /* Compute weighted average pattern */
    float* weights = nimcp_calloc(num_observations, sizeof(float));
    float weight_sum = 0.0f;

    for (uint32_t i = 0; i < num_observations; i++) {
        weights[i] = observations[i].salience * observations[i].reliability;
        weight_sum += weights[i];
    }

    if (weight_sum > 0.0f) {
        for (uint32_t i = 0; i < num_observations; i++) {
            weights[i] /= weight_sum;
            for (uint32_t j = 0; j < observations[i].dim && j < max_dim; j++) {
                hunch->predicted_pattern[j] += weights[i] * observations[i].data[j];
            }
        }
    }

    nimcp_free(weights);

    /* Compute intuition scores */
    float pattern_entropy = compute_entropy(hunch->predicted_pattern, max_dim);
    float pattern_trend = fabsf(detect_trend(hunch->predicted_pattern, max_dim));
    float pattern_periodicity = detect_periodicity(hunch->predicted_pattern, max_dim);

    /* Plausibility: higher if pattern has structure */
    hunch->score.plausibility = (1.0f - pattern_entropy * 0.5f) *
                                (0.5f + pattern_trend * 0.25f + pattern_periodicity * 0.25f);

    /* Novelty: higher if less similar to known patterns */
    float max_similarity = 0.0f;
    for (uint32_t i = 0; i < engine->num_patterns; i++) {
        if (engine->patterns[i].dim == max_dim) {
            float sim = cosine_similarity(hunch->predicted_pattern,
                                         engine->patterns[i].data, max_dim);
            if (sim > max_similarity) max_similarity = sim;
        }
    }
    hunch->score.novelty = 1.0f - max_similarity;

    /* Coherence: how well observations agree */
    float variance = 0.0f;
    for (uint32_t i = 0; i < num_observations && i < max_dim; i++) {
        for (uint32_t j = 0; j < observations[i].dim; j++) {
            float diff = observations[i].data[j] - hunch->predicted_pattern[j];
            variance += diff * diff;
        }
    }
    variance /= (num_observations * max_dim);
    hunch->score.coherence = expf(-variance);

    /* Fertility: based on pattern complexity */
    hunch->score.fertility = pattern_entropy * 0.5f + 0.5f * (pattern_trend + pattern_periodicity) / 2.0f;

    /* Overall confidence */
    hunch->score.confidence = (
        hunch->score.plausibility * 0.3f +
        hunch->score.coherence * engine->config.coherence_weight +
        hunch->score.novelty * engine->config.novelty_weight +
        hunch->score.fertility * engine->config.fertility_weight
    );

    /* Apply modulation */
    hunch->score.confidence = apply_modulation(engine, hunch->score.confidence);
    hunch->score.plausibility = apply_modulation(engine, hunch->score.plausibility);

    /* Set probabilities */
    hunch->prior_probability = engine->config.prior_strength;
    hunch->posterior_probability = hunch->prior_probability * hunch->score.coherence;

    /* Determine actionability */
    hunch->is_actionable = (hunch->score.plausibility >= engine->config.plausibility_threshold);
    hunch->needs_verification = (hunch->score.confidence < engine->config.confidence_threshold);

    /* Create description */
    snprintf(hunch->description, sizeof(hunch->description),
             "Pattern hunch: dim=%u, plaus=%.2f, novel=%.2f, coher=%.2f",
             max_dim, hunch->score.plausibility, hunch->score.novelty, hunch->score.coherence);

    /* Track supporting observations */
    hunch->supporting_obs = nimcp_calloc(num_observations, sizeof(uint32_t));
    hunch->num_supporting = num_observations;
    for (uint32_t i = 0; i < num_observations; i++) {
        hunch->supporting_obs[i] = i;
    }

    /* Update statistics */
    engine->stats.hunches_formed++;
    uint64_t elapsed = get_timestamp_us() - start_time;
    engine->stats.avg_processing_time_us =
        (engine->stats.avg_processing_time_us * (engine->stats.hunches_formed - 1) + elapsed) /
        engine->stats.hunches_formed;

    return hunch;
}

hunch_t* intuitive_form_hunch_from_data(
    intuitive_engine_t* engine,
    const float* data,
    uint32_t length
) {
    if (!engine || !data || length == 0) {
        set_error("Invalid parameters");
        return NULL;
    }

    /* Create a single observation from raw data */
    observation_t obs;
    obs.data = (float*)data;  /* Const cast - we won't modify */
    obs.dim = length;
    obs.salience = 1.0f;
    obs.reliability = 1.0f;
    obs.timestamp = get_timestamp_us();
    strncpy(obs.source, "raw_data", sizeof(obs.source));

    return intuitive_form_hunch(engine, &obs, 1);
}

int intuitive_update_hunch(
    intuitive_engine_t* engine,
    hunch_t* hunch,
    const observation_t* observation
) {
    if (!engine || !hunch || !observation) {
        set_error("Invalid parameters");
        return -1;
    }

    /* Update pattern with new observation */
    float learning_rate = engine->config.learning_rate;
    float weight = observation->salience * observation->reliability;

    for (uint32_t i = 0; i < hunch->pattern_dim && i < observation->dim; i++) {
        hunch->predicted_pattern[i] += learning_rate * weight *
                                       (observation->data[i] - hunch->predicted_pattern[i]);
    }

    /* Update coherence based on how well observation fits */
    float error = 0.0f;
    for (uint32_t i = 0; i < observation->dim && i < hunch->pattern_dim; i++) {
        float diff = observation->data[i] - hunch->predicted_pattern[i];
        error += diff * diff;
    }
    error /= observation->dim;

    float observation_coherence = expf(-error);
    hunch->score.coherence = 0.9f * hunch->score.coherence + 0.1f * observation_coherence;

    /* Update posterior probability (Bayesian update) */
    hunch->posterior_probability *= observation_coherence;
    hunch->posterior_probability = fminf(1.0f, fmaxf(0.0f, hunch->posterior_probability));

    /* Update confidence */
    hunch->score.confidence = (
        hunch->score.plausibility * 0.3f +
        hunch->score.coherence * engine->config.coherence_weight +
        hunch->score.novelty * engine->config.novelty_weight +
        hunch->score.fertility * engine->config.fertility_weight
    );
    hunch->score.confidence = apply_modulation(engine, hunch->score.confidence);

    return 0;
}

int intuitive_refine_hunch(
    intuitive_engine_t* engine,
    hunch_t* hunch,
    const float* context,
    uint32_t context_dim
) {
    if (!engine || !hunch || !context || context_dim == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    /* Use context to adjust pattern */
    uint32_t min_dim = (context_dim < hunch->pattern_dim) ? context_dim : hunch->pattern_dim;

    for (uint32_t i = 0; i < min_dim; i++) {
        /* Context modulates pattern */
        hunch->predicted_pattern[i] *= (1.0f + 0.1f * context[i]);
    }

    /* Recalculate scores */
    float pattern_entropy = compute_entropy(hunch->predicted_pattern, hunch->pattern_dim);
    hunch->score.fertility = pattern_entropy * 0.5f + 0.5f;

    return 0;
}

void intuitive_free_hunch(hunch_t* hunch) {
    if (!hunch) return;

    if (hunch->predicted_pattern) {
        nimcp_free(hunch->predicted_pattern);
    }
    if (hunch->supporting_obs) {
        nimcp_free(hunch->supporting_obs);
    }
    if (hunch->conflicting_obs) {
        nimcp_free(hunch->conflicting_obs);
    }

    nimcp_free(hunch);
}

/* ============================================================================
 * PLAUSIBILITY ESTIMATION IMPLEMENTATION
 * ============================================================================ */

float intuitive_estimate_plausibility(
    intuitive_engine_t* engine,
    const hypothesis_t* hypothesis
) {
    if (!engine || !hypothesis) {
        set_error("Invalid parameters");
        return 0.0f;
    }

    float plausibility = 0.5f;  /* Start neutral */

    /* Factor in explanatory power */
    plausibility += 0.2f * hypothesis->explanatory_power;

    /* Factor in parsimony (simpler is more plausible) */
    plausibility += 0.15f * hypothesis->parsimony;

    /* Factor in falsifiability (testable is more plausible) */
    plausibility += 0.1f * hypothesis->falsifiability;

    /* Factor in prior belief */
    plausibility *= (0.5f + 0.5f * hypothesis->prior);

    /* Apply modulation */
    plausibility = apply_modulation(engine, plausibility);

    return fmaxf(0.0f, fminf(1.0f, plausibility));
}

int intuitive_score_hypothesis(
    intuitive_engine_t* engine,
    const hypothesis_t* hypothesis,
    intuition_score_t* score
) {
    if (!engine || !hypothesis || !score) {
        set_error("Invalid parameters");
        return -1;
    }

    score->plausibility = intuitive_estimate_plausibility(engine, hypothesis);

    /* Novelty: inverse of how common such hypotheses are */
    score->novelty = 1.0f - hypothesis->prior;

    /* Coherence: based on likelihood */
    score->coherence = hypothesis->likelihood;

    /* Fertility: based on explanatory power */
    score->fertility = hypothesis->explanatory_power;

    /* Confidence: weighted combination */
    score->confidence = (
        score->plausibility * 0.3f +
        score->coherence * engine->config.coherence_weight +
        score->novelty * engine->config.novelty_weight +
        score->fertility * engine->config.fertility_weight
    );

    /* Urgency: based on falsifiability (more testable = more urgent to test) */
    score->urgency = hypothesis->falsifiability;

    return 0;
}

int intuitive_rank_hypotheses(
    intuitive_engine_t* engine,
    const hypothesis_t* hypotheses,
    uint32_t num_hypotheses,
    uint32_t* rankings
) {
    if (!engine || !hypotheses || !rankings || num_hypotheses == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    /* Compute plausibility for each */
    float* scores = nimcp_calloc(num_hypotheses, sizeof(float));
    if (!scores) {
        set_error("Failed to allocate scores");
        return -1;
    }

    for (uint32_t i = 0; i < num_hypotheses; i++) {
        scores[i] = intuitive_estimate_plausibility(engine, &hypotheses[i]);
        rankings[i] = i;
    }

    /* Simple bubble sort (fine for small N) */
    for (uint32_t i = 0; i < num_hypotheses - 1; i++) {
        for (uint32_t j = 0; j < num_hypotheses - i - 1; j++) {
            if (scores[rankings[j]] < scores[rankings[j + 1]]) {
                uint32_t temp = rankings[j];
                rankings[j] = rankings[j + 1];
                rankings[j + 1] = temp;
            }
        }
    }

    nimcp_free(scores);
    return 0;
}

float intuitive_estimate_statement_plausibility(
    intuitive_engine_t* engine,
    const char* statement,
    const float* domain_features,
    uint32_t num_features
) {
    if (!engine || !statement) {
        set_error("Invalid parameters");
        return 0.0f;
    }

    float plausibility = 0.5f;

    /* Simple heuristics based on statement length */
    size_t len = strlen(statement);
    if (len < 10) {
        plausibility -= 0.1f;  /* Too short to be meaningful */
    } else if (len > 500) {
        plausibility -= 0.05f;  /* Very long might be overspecified */
    }

    /* Use domain features if provided */
    if (domain_features && num_features > 0) {
        float feature_sum = 0.0f;
        for (uint32_t i = 0; i < num_features; i++) {
            feature_sum += domain_features[i];
        }
        float avg_feature = feature_sum / num_features;
        plausibility += 0.2f * avg_feature;
    }

    return apply_modulation(engine, fmaxf(0.0f, fminf(1.0f, plausibility)));
}

/* ============================================================================
 * CONFIDENCE GRADIENT IMPLEMENTATION
 * ============================================================================ */

int intuitive_track_confidence(
    intuitive_engine_t* engine,
    const float* step_confidences,
    uint32_t num_steps,
    confidence_gradient_t* gradient
) {
    if (!engine || !step_confidences || !gradient || num_steps == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    gradient->confidence_values = nimcp_calloc(num_steps, sizeof(float));
    if (!gradient->confidence_values) {
        set_error("Failed to allocate gradient");
        return -1;
    }

    memcpy(gradient->confidence_values, step_confidences, num_steps * sizeof(float));
    gradient->num_steps = num_steps;

    /* Find min, max, and weakest link */
    gradient->min_confidence = step_confidences[0];
    gradient->max_confidence = step_confidences[0];
    gradient->weakest_link = 0;

    for (uint32_t i = 1; i < num_steps; i++) {
        if (step_confidences[i] < gradient->min_confidence) {
            gradient->min_confidence = step_confidences[i];
            gradient->weakest_link = i;
        }
        if (step_confidences[i] > gradient->max_confidence) {
            gradient->max_confidence = step_confidences[i];
        }
    }

    /* Compute decay rate */
    if (num_steps > 1) {
        float total_decay = 0.0f;
        for (uint32_t i = 1; i < num_steps; i++) {
            if (step_confidences[i - 1] > 0) {
                total_decay += step_confidences[i] / step_confidences[i - 1];
            }
        }
        gradient->decay_rate = total_decay / (num_steps - 1);
    } else {
        gradient->decay_rate = 1.0f;
    }

    return 0;
}

float intuitive_propagate_confidence(
    intuitive_engine_t* engine,
    float current_confidence,
    float step_reliability
) {
    if (!engine) return current_confidence;

    /* Confidence decays based on step reliability */
    float decay = CONFIDENCE_DECAY_BASE + 0.05f * step_reliability;
    float new_confidence = current_confidence * decay * step_reliability;

    return apply_modulation(engine, fmaxf(0.0f, fminf(1.0f, new_confidence)));
}

int intuitive_find_weak_links(
    intuitive_engine_t* engine,
    const confidence_gradient_t* gradient,
    uint32_t* weak_indices,
    uint32_t max_weak,
    uint32_t* num_found
) {
    if (!engine || !gradient || !weak_indices || !num_found) {
        set_error("Invalid parameters");
        return -1;
    }

    *num_found = 0;
    float threshold = gradient->min_confidence + 0.2f * (gradient->max_confidence - gradient->min_confidence);

    for (uint32_t i = 0; i < gradient->num_steps && *num_found < max_weak; i++) {
        if (gradient->confidence_values[i] < threshold) {
            weak_indices[*num_found] = i;
            (*num_found)++;
        }
    }

    return 0;
}

void intuitive_free_gradient(confidence_gradient_t* gradient) {
    if (!gradient) return;
    if (gradient->confidence_values) {
        nimcp_free(gradient->confidence_values);
        gradient->confidence_values = NULL;
    }
}

/* ============================================================================
 * INTUITIVE LEAP IMPLEMENTATION
 * ============================================================================ */

insight_t* intuitive_leap(
    intuitive_engine_t* engine,
    const problem_t* problem
) {
    return intuitive_leap_with_strategy(engine, problem, INTUITIVE_STRATEGY_RECOGNITION);
}

insight_t* intuitive_leap_with_strategy(
    intuitive_engine_t* engine,
    const problem_t* problem,
    intuitive_strategy_t strategy
) {
    if (!engine || !problem) {
        set_error("Invalid parameters");
        return NULL;
    }

    uint64_t start_time = get_timestamp_us();

    /* Check if leap is possible */
    float leap_probability = intuitive_can_leap(engine, problem);
    if (leap_probability < 0.3f) {
        set_error("Leap probability too low");
        return NULL;
    }

    insight_t* insight = nimcp_calloc(1, sizeof(insight_t));
    if (!insight) {
        set_error("Failed to allocate insight");
        return NULL;
    }

    insight->id = engine->stats.insights_generated + 1;

    /* Copy problem reference */
    insight->original_problem = nimcp_calloc(1, sizeof(problem_t));
    if (insight->original_problem) {
        memcpy(insight->original_problem, problem, sizeof(problem_t));
    }

    /* Generate solution based on strategy */
    insight->solution_dim = problem->state_dim;
    insight->solution = nimcp_calloc(insight->solution_dim, sizeof(float));

    switch (strategy) {
        case INTUITIVE_STRATEGY_RECOGNITION:
            /* Pattern matching: look for similar patterns in memory */
            if (engine->num_patterns > 0 && problem->initial_state) {
                float best_sim = 0.0f;
                uint32_t best_idx = 0;

                for (uint32_t i = 0; i < engine->num_patterns; i++) {
                    if (engine->patterns[i].dim == problem->state_dim) {
                        float sim = cosine_similarity(problem->initial_state,
                                                     engine->patterns[i].data,
                                                     problem->state_dim);
                        if (sim > best_sim) {
                            best_sim = sim;
                            best_idx = i;
                        }
                    }
                }

                if (best_sim > PATTERN_MATCH_THRESHOLD) {
                    /* Use matched pattern as template */
                    memcpy(insight->solution, engine->patterns[best_idx].data,
                           insight->solution_dim * sizeof(float));
                    insight->steps_skipped = 5;  /* Estimate */
                }
            }
            break;

        case INTUITIVE_STRATEGY_SIMULATION:
            /* Mental simulation: interpolate toward goal */
            if (problem->initial_state && problem->goal_state && problem->goal_known) {
                for (uint32_t i = 0; i < insight->solution_dim; i++) {
                    insight->solution[i] = problem->initial_state[i] +
                        0.8f * (problem->goal_state[i] - problem->initial_state[i]);
                }
                insight->steps_skipped = 3;
            }
            break;

        case INTUITIVE_STRATEGY_HEURISTIC:
            /* Apply simple heuristics */
            if (problem->initial_state) {
                for (uint32_t i = 0; i < insight->solution_dim; i++) {
                    /* Heuristic: move away from extremes */
                    float val = problem->initial_state[i];
                    if (val > 0.5f) {
                        insight->solution[i] = val - 0.1f;
                    } else {
                        insight->solution[i] = val + 0.1f;
                    }
                }
                insight->steps_skipped = 2;
            }
            break;

        default:
            /* Default: weighted combination of initial and goal */
            if (problem->initial_state) {
                memcpy(insight->solution, problem->initial_state,
                       insight->solution_dim * sizeof(float));
            }
            break;
    }

    /* Compute insight quality metrics */
    insight->surprise_factor = 1.0f - problem->estimated_difficulty;
    insight->elegance = (insight->steps_skipped > 0) ?
                        fminf(1.0f, insight->steps_skipped / 10.0f) : 0.5f;
    insight->generalizability = leap_probability;

    /* Create key realization description */
    insight->key_realization = nimcp_calloc(128, sizeof(char));
    if (insight->key_realization) {
        snprintf(insight->key_realization, 128,
                 "Pattern recognition via %s strategy",
                 intuitive_strategy_name(strategy));
    }

    strncpy(insight->description, "Intuitive leap solution",
            sizeof(insight->description));

    insight->time_saved_estimate = insight->steps_skipped * 100.0f;  /* Arbitrary units */
    insight->verified = false;
    insight->verification_confidence = 0.0f;

    /* Update statistics */
    engine->stats.insights_generated++;
    engine->stats.intuitive_leaps++;

    uint64_t elapsed = get_timestamp_us() - start_time;
    float total_ops = engine->stats.hunches_formed + engine->stats.insights_generated;
    engine->stats.avg_processing_time_us =
        (engine->stats.avg_processing_time_us * (total_ops - 1) + elapsed) / total_ops;

    return insight;
}

float intuitive_can_leap(
    intuitive_engine_t* engine,
    const problem_t* problem
) {
    if (!engine || !problem) return 0.0f;

    float probability = 0.5f;

    /* Higher probability if problem is simpler */
    probability += 0.2f * (1.0f - problem->estimated_difficulty);

    /* Higher if we have relevant patterns */
    if (engine->num_patterns > 0 && problem->initial_state) {
        float max_sim = 0.0f;
        for (uint32_t i = 0; i < engine->num_patterns; i++) {
            if (engine->patterns[i].dim == problem->state_dim) {
                float sim = cosine_similarity(problem->initial_state,
                                             engine->patterns[i].data,
                                             problem->state_dim);
                if (sim > max_sim) max_sim = sim;
            }
        }
        probability += 0.3f * max_sim;
    }

    /* Higher if goal is known */
    if (problem->goal_known) {
        probability += 0.1f;
    }

    /* Apply modulation (fatigue reduces leap ability) */
    probability = apply_modulation(engine, probability);

    /* Emotional valence affects willingness to leap */
    if (engine->emotional_valence > 0) {
        probability += 0.1f * engine->emotional_valence;  /* Positive emotion helps */
    }

    return fmaxf(0.0f, fminf(1.0f, probability));
}

void intuitive_free_insight(insight_t* insight) {
    if (!insight) return;

    if (insight->solution) {
        nimcp_free(insight->solution);
    }
    if (insight->key_realization) {
        nimcp_free(insight->key_realization);
    }
    if (insight->original_problem) {
        if (insight->original_problem->initial_state) {
            nimcp_free(insight->original_problem->initial_state);
        }
        if (insight->original_problem->goal_state) {
            nimcp_free(insight->original_problem->goal_state);
        }
        if (insight->original_problem->constraints) {
            nimcp_free(insight->original_problem->constraints);
        }
        nimcp_free(insight->original_problem);
    }

    nimcp_free(insight);
}

/* ============================================================================
 * GESTALT PERCEPTION IMPLEMENTATION
 * ============================================================================ */

int intuitive_gestalt_perceive(
    intuitive_engine_t* engine,
    const float* data,
    uint32_t dim,
    gestalt_result_t* result
) {
    if (!engine || !data || !result || dim == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    if (!engine->config.enable_gestalt) {
        set_error("Gestalt perception disabled");
        return -1;
    }

    memset(result, 0, sizeof(gestalt_result_t));

    /* Compute whole representation (mean + variance summary) */
    result->repr_dim = 4;  /* Summary statistics */
    result->whole_representation = nimcp_calloc(result->repr_dim, sizeof(float));
    if (!result->whole_representation) {
        set_error("Failed to allocate representation");
        return -1;
    }

    /* Mean */
    float mean = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        mean += data[i];
    }
    mean /= dim;
    result->whole_representation[0] = mean;

    /* Variance */
    float variance = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= dim;
    result->whole_representation[1] = sqrtf(variance);

    /* Range */
    float min_val = data[0], max_val = data[0];
    for (uint32_t i = 1; i < dim; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    result->whole_representation[2] = max_val - min_val;

    /* Trend */
    result->whole_representation[3] = detect_trend(data, dim);

    /* Closure strength: how complete is the pattern? */
    float entropy = compute_entropy(data, dim);
    result->closure_strength = 1.0f - entropy;

    /* Figure-ground separation: based on variance */
    result->figure_ground_separation = fminf(1.0f, sqrtf(variance) * 2.0f);

    /* Grouping strength: based on autocorrelation */
    result->grouping_strength = detect_periodicity(data, dim);

    /* Identify pattern type */
    if (result->grouping_strength > 0.7f) {
        strncpy(result->pattern_type, "periodic", sizeof(result->pattern_type));
    } else if (fabsf(result->whole_representation[3]) > 0.7f) {
        strncpy(result->pattern_type, "trending", sizeof(result->pattern_type));
    } else if (variance < 0.1f) {
        strncpy(result->pattern_type, "uniform", sizeof(result->pattern_type));
    } else {
        strncpy(result->pattern_type, "complex", sizeof(result->pattern_type));
    }

    result->num_parts = (uint32_t)(dim / 4) + 1;  /* Rough estimate */

    engine->stats.gestalt_perceptions++;

    return 0;
}

int intuitive_gestalt_group(
    intuitive_engine_t* engine,
    const float* elements,
    uint32_t num_elements,
    uint32_t element_dim,
    uint32_t* group_assignments,
    uint32_t* num_groups
) {
    if (!engine || !elements || !group_assignments || !num_groups ||
        num_elements == 0 || element_dim == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    /* Simple proximity-based grouping */
    const float PROXIMITY_THRESHOLD = 0.5f;

    *num_groups = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        group_assignments[i] = UINT32_MAX;  /* Unassigned */
    }

    for (uint32_t i = 0; i < num_elements; i++) {
        if (group_assignments[i] == UINT32_MAX) {
            /* Start new group */
            group_assignments[i] = *num_groups;

            /* Find similar elements */
            for (uint32_t j = i + 1; j < num_elements; j++) {
                if (group_assignments[j] == UINT32_MAX) {
                    float sim = cosine_similarity(
                        &elements[i * element_dim],
                        &elements[j * element_dim],
                        element_dim
                    );
                    if (sim > PROXIMITY_THRESHOLD) {
                        group_assignments[j] = *num_groups;
                    }
                }
            }

            (*num_groups)++;
        }
    }

    return 0;
}

void intuitive_free_gestalt(gestalt_result_t* result) {
    if (!result) return;
    if (result->whole_representation) {
        nimcp_free(result->whole_representation);
        result->whole_representation = NULL;
    }
}

/* ============================================================================
 * PATTERN MATCHING IMPLEMENTATION
 * ============================================================================ */

int intuitive_match_patterns(
    intuitive_engine_t* engine,
    const float* input,
    uint32_t input_dim,
    pattern_match_t* matches,
    uint32_t max_matches,
    uint32_t* num_found
) {
    if (!engine || !input || !matches || !num_found || input_dim == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    *num_found = 0;

    for (uint32_t i = 0; i < engine->num_patterns && *num_found < max_matches; i++) {
        stored_pattern_t* pattern = &engine->patterns[i];

        if (pattern->dim != input_dim) continue;

        float similarity = cosine_similarity(input, pattern->data, input_dim);

        if (similarity > PATTERN_MATCH_THRESHOLD) {
            pattern_match_t* match = &matches[*num_found];

            match->structural_similarity = similarity;
            match->surface_similarity = similarity;
            match->functional_similarity = similarity * 0.8f;  /* Estimate */
            match->overall_similarity = similarity;
            match->matched_pattern_id = pattern->id;
            match->matched_pattern_name = pattern->name;

            /* Update pattern statistics */
            pattern->match_count++;
            pattern->avg_match_strength =
                (pattern->avg_match_strength * (pattern->match_count - 1) + similarity) /
                pattern->match_count;

            (*num_found)++;
            engine->stats.patterns_matched++;
        }
    }

    return 0;
}

uint32_t intuitive_register_pattern(
    intuitive_engine_t* engine,
    const float* pattern,
    uint32_t pattern_dim,
    const char* name
) {
    if (!engine || !pattern || pattern_dim == 0) {
        set_error("Invalid parameters");
        return 0;
    }

    if (engine->num_patterns >= engine->pattern_capacity) {
        set_error("Pattern memory full");
        return 0;
    }

    stored_pattern_t* stored = &engine->patterns[engine->num_patterns];

    stored->id = engine->next_pattern_id++;
    stored->dim = pattern_dim;
    stored->match_count = 0;
    stored->avg_match_strength = 0.0f;

    stored->data = nimcp_calloc(pattern_dim, sizeof(float));
    if (!stored->data) {
        set_error("Failed to allocate pattern data");
        return 0;
    }
    memcpy(stored->data, pattern, pattern_dim * sizeof(float));

    if (name) {
        strncpy(stored->name, name, sizeof(stored->name) - 1);
    } else {
        snprintf(stored->name, sizeof(stored->name), "pattern_%u", stored->id);
    }

    engine->num_patterns++;

    return stored->id;
}

int intuitive_forget_pattern(
    intuitive_engine_t* engine,
    uint32_t pattern_id
) {
    if (!engine || pattern_id == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    for (uint32_t i = 0; i < engine->num_patterns; i++) {
        if (engine->patterns[i].id == pattern_id) {
            if (engine->patterns[i].data) {
                nimcp_free(engine->patterns[i].data);
            }

            /* Shift remaining patterns */
            for (uint32_t j = i; j < engine->num_patterns - 1; j++) {
                engine->patterns[j] = engine->patterns[j + 1];
            }
            engine->num_patterns--;

            return 0;
        }
    }

    set_error("Pattern not found");
    return -1;
}

/* ============================================================================
 * INCUBATION IMPLEMENTATION
 * ============================================================================ */

uint32_t intuitive_incubate(
    intuitive_engine_t* engine,
    const problem_t* problem
) {
    if (!engine || !problem) {
        set_error("Invalid parameters");
        return 0;
    }

    if (!engine->config.enable_incubation) {
        set_error("Incubation disabled");
        return 0;
    }

    if (engine->num_incubating >= engine->config.max_incubation_problems) {
        set_error("Incubation queue full");
        return 0;
    }

    incubation_entry_t* entry = &engine->incubation_queue[engine->num_incubating];

    entry->id = engine->next_incubation_id++;
    entry->progress = 0.0f;
    entry->iterations = 0;
    entry->ready = false;
    entry->result = NULL;

    /* Copy problem */
    memcpy(&entry->problem, problem, sizeof(problem_t));

    if (problem->initial_state && problem->state_dim > 0) {
        entry->problem.initial_state = nimcp_calloc(problem->state_dim, sizeof(float));
        if (entry->problem.initial_state) {
            memcpy(entry->problem.initial_state, problem->initial_state,
                   problem->state_dim * sizeof(float));
        }
    }

    if (problem->goal_state && problem->state_dim > 0) {
        entry->problem.goal_state = nimcp_calloc(problem->state_dim, sizeof(float));
        if (entry->problem.goal_state) {
            memcpy(entry->problem.goal_state, problem->goal_state,
                   problem->state_dim * sizeof(float));
        }
    }

    /* Initialize partial solution */
    entry->solution_dim = problem->state_dim;
    entry->partial_solution = nimcp_calloc(entry->solution_dim, sizeof(float));
    if (problem->initial_state && entry->partial_solution) {
        memcpy(entry->partial_solution, problem->initial_state,
               entry->solution_dim * sizeof(float));
    }

    engine->num_incubating++;

    return entry->id;
}

int intuitive_check_incubation(
    intuitive_engine_t* engine,
    uint32_t problem_id,
    insight_t** insight
) {
    if (!engine || !insight || problem_id == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    *insight = NULL;

    for (uint32_t i = 0; i < engine->num_incubating; i++) {
        if (engine->incubation_queue[i].id == problem_id) {
            if (engine->incubation_queue[i].ready) {
                *insight = engine->incubation_queue[i].result;
                engine->incubation_queue[i].result = NULL;  /* Transfer ownership */
                return 1;
            }
            return 0;  /* Still incubating */
        }
    }

    set_error("Problem not found in incubation queue");
    return -1;
}

int intuitive_process_incubation(intuitive_engine_t* engine) {
    if (!engine) return -1;

    int insights_generated = 0;
    float progress_rate = 0.1f * (1.0f - engine->fatigue);  /* Fatigue slows incubation */

    for (uint32_t i = 0; i < engine->num_incubating; i++) {
        incubation_entry_t* entry = &engine->incubation_queue[i];

        if (entry->ready) continue;

        entry->iterations++;
        entry->progress += progress_rate * (1.0f - entry->problem.estimated_difficulty);

        /* Update partial solution */
        if (entry->partial_solution && entry->problem.goal_state && entry->problem.goal_known) {
            for (uint32_t j = 0; j < entry->solution_dim; j++) {
                entry->partial_solution[j] +=
                    progress_rate * (entry->problem.goal_state[j] - entry->partial_solution[j]);
            }
        }

        /* Check if insight emerges */
        if (entry->progress >= 1.0f) {
            entry->ready = true;
            entry->result = intuitive_leap(engine, &entry->problem);
            if (entry->result) {
                /* Update solution with incubated result */
                if (entry->partial_solution && entry->result->solution) {
                    memcpy(entry->result->solution, entry->partial_solution,
                           entry->solution_dim * sizeof(float));
                }
                entry->result->time_saved_estimate = entry->iterations * 10.0f;
                insights_generated++;
            }
        }
    }

    return insights_generated;
}

int intuitive_cancel_incubation(
    intuitive_engine_t* engine,
    uint32_t problem_id
) {
    if (!engine || problem_id == 0) {
        set_error("Invalid parameters");
        return -1;
    }

    for (uint32_t i = 0; i < engine->num_incubating; i++) {
        if (engine->incubation_queue[i].id == problem_id) {
            incubation_entry_t* entry = &engine->incubation_queue[i];

            /* Free resources */
            if (entry->partial_solution) {
                nimcp_free(entry->partial_solution);
            }
            if (entry->result) {
                intuitive_free_insight(entry->result);
            }
            if (entry->problem.initial_state) {
                nimcp_free(entry->problem.initial_state);
            }
            if (entry->problem.goal_state) {
                nimcp_free(entry->problem.goal_state);
            }
            if (entry->problem.constraints) {
                nimcp_free(entry->problem.constraints);
            }

            /* Shift remaining entries */
            for (uint32_t j = i; j < engine->num_incubating - 1; j++) {
                engine->incubation_queue[j] = engine->incubation_queue[j + 1];
            }
            engine->num_incubating--;

            return 0;
        }
    }

    set_error("Problem not found");
    return -1;
}

/* ============================================================================
 * MODULATION IMPLEMENTATION
 * ============================================================================ */

int intuitive_set_inflammation(intuitive_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->inflammation = fmaxf(0.0f, fminf(1.0f, level));
    return 0;
}

int intuitive_set_fatigue(intuitive_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->fatigue = fmaxf(0.0f, fminf(1.0f, level));
    return 0;
}

int intuitive_set_emotional_valence(intuitive_engine_t* engine, float valence) {
    if (!engine) return -1;
    engine->emotional_valence = fmaxf(-1.0f, fminf(1.0f, valence));
    return 0;
}

/* ============================================================================
 * STATISTICS IMPLEMENTATION
 * ============================================================================ */

int intuitive_get_stats(const intuitive_engine_t* engine, intuitive_stats_t* stats) {
    if (!engine || !stats) return -1;
    *stats = engine->stats;
    return 0;
}

void intuitive_reset_stats(intuitive_engine_t* engine) {
    if (!engine) return;
    memset(&engine->stats, 0, sizeof(engine->stats));
}

const char* intuitive_get_last_error(void) {
    return g_last_error;
}

/* ============================================================================
 * HELPER FUNCTIONS IMPLEMENTATION
 * ============================================================================ */

observation_t intuitive_create_observation(
    const float* data,
    uint32_t dim,
    float salience
) {
    observation_t obs;
    memset(&obs, 0, sizeof(obs));

    obs.data = (float*)data;  /* Note: caller owns the data */
    obs.dim = dim;
    obs.salience = fmaxf(0.0f, fminf(1.0f, salience));
    obs.reliability = 1.0f;
    obs.timestamp = get_timestamp_us();
    strncpy(obs.source, "user", sizeof(obs.source));

    return obs;
}

problem_t intuitive_create_problem(
    const float* initial,
    const float* goal,
    uint32_t dim,
    const char* description
) {
    problem_t prob;
    memset(&prob, 0, sizeof(prob));

    prob.initial_state = (float*)initial;  /* Caller owns */
    prob.goal_state = (float*)goal;
    prob.state_dim = dim;
    prob.goal_known = (goal != NULL);
    prob.estimated_difficulty = 0.5f;

    if (description) {
        strncpy(prob.description, description, sizeof(prob.description) - 1);
    }

    return prob;
}

hypothesis_t intuitive_create_hypothesis(
    const char* statement,
    const float* parameters,
    uint32_t num_params
) {
    hypothesis_t hyp;
    memset(&hyp, 0, sizeof(hyp));

    if (statement) {
        strncpy(hyp.statement, statement, sizeof(hyp.statement) - 1);
    }

    hyp.parameters = (float*)parameters;  /* Caller owns */
    hyp.num_params = num_params;
    hyp.explanatory_power = 0.5f;
    hyp.parsimony = 0.5f;
    hyp.falsifiability = 0.5f;
    hyp.prior = 0.5f;
    hyp.likelihood = 0.5f;
    hyp.posterior = 0.5f;

    return hyp;
}

const char* intuitive_strategy_name(intuitive_strategy_t strategy) {
    switch (strategy) {
        case INTUITIVE_STRATEGY_RECOGNITION:  return "recognition";
        case INTUITIVE_STRATEGY_SIMULATION:   return "simulation";
        case INTUITIVE_STRATEGY_ANALOGY:      return "analogy";
        case INTUITIVE_STRATEGY_HEURISTIC:    return "heuristic";
        case INTUITIVE_STRATEGY_GESTALT:      return "gestalt";
        case INTUITIVE_STRATEGY_EMOTIONAL:    return "emotional";
        case INTUITIVE_STRATEGY_ASSOCIATIVE:  return "associative";
        case INTUITIVE_STRATEGY_INCUBATION:   return "incubation";
        default:                              return "unknown";
    }
}
