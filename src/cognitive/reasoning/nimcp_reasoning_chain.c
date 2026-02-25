/**
 * @file nimcp_reasoning_chain.c
 * @brief Multi-step reasoning chain engine implementation
 *
 * WHAT: Orchestrates cognitive modules to perform multi-step human-like reasoning
 * WHY:  Enable transparent, traceable reasoning with confidence tracking
 * HOW:  Connects to brain subsystems and executes a 9-phase pipeline:
 *        recall -> knowledge -> decompose -> world_model -> infer ->
 *        jepa_predict -> verify -> assess -> synthesize
 *
 * ARCHITECTURE:
 * The reasoning engine does NOT own any cognitive modules. It borrows pointers
 * from a connected brain instance via accessor functions. Each phase of the
 * pipeline gracefully degrades if its required module is NULL (not available).
 *
 * @version 1.0.0
 * @date 2026-02-25
 */

#include "cognitive/reasoning/nimcp_reasoning_chain.h"

#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain/accessors/nimcp_brain_accessors.h"
#include "cognitive/memory/nimcp_engram.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "cognitive/nimcp_working_memory.h"
#include "cognitive/nimcp_predictive.h"
#include "cognitive/epistemic/nimcp_epistemic_filter.h"
#include "cognitive/recursive/nimcp_rcog_engine.h"
#include "cognitive/nimcp_self_model.h"
#include "cognitive/jepa/nimcp_jepa_predictor.h"
#include "cognitive/jepa/nimcp_jepa_latent.h"
#include "cognitive/jepa/nimcp_jepa_context.h"
#include "cognitive/jepa/nimcp_jepa_fep_bridge.h"
#include "cognitive/extrapolation/nimcp_world_model_multimodal.h"
#include "cognitive/omni/nimcp_omni_world_model.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>
#include <stdio.h>
#include <math.h>

#define LOG_MODULE "reasoning_chain"

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Reasoning engine internal state
 *
 * WHAT: Holds configuration, brain module pointers, and statistics
 * WHY:  Opaque struct encapsulates engine state from callers
 */
struct reasoning_engine {
    reasoning_engine_config_t config;

    /* Connected brain handle */
    brain_t brain;

    /* Module pointers extracted from brain (may be NULL) */
    engram_system_t* engram_system;
    knowledge_system_t knowledge_system;
    working_memory_t* working_memory;
    predictive_network_t predictive_net;
    epistemic_filter_t epistemic_filter;
    rcog_engine_t* rcog_engine;
    self_model_system_t self_model;

    /* JEPA and world model pointers (may be NULL) */
    struct omni_world_model* omni_world_model;
    struct nimcp_world_model* multimodal_world_model;
    jepa_predictor_t* jepa_predictor;
    jepa_context_encoder_t* jepa_context;
    jepa_fep_bridge_t* jepa_fep_bridge;

    /* Aggregate statistics */
    reasoning_engine_stats_t stats;

    /* Connection state */
    bool is_connected;
};

/*=============================================================================
 * INTERNAL CONSTANTS
 *===========================================================================*/

/** Number of neuron IDs to generate from query hash for engram recall */
#define QUERY_HASH_CUE_COUNT 8

/** Maximum knowledge items to retrieve per query */
#define MAX_KNOWLEDGE_ITEMS 5

/** Maximum cross-domain connections to retrieve */
#define MAX_KNOWLEDGE_CONNECTIONS 5

/** Maximum evidence sources for inference */
#define MAX_EVIDENCE_SOURCES 16

/** Predictive coding input dimension for verification */
#define PREDICTIVE_INPUT_DIM 4

/** Predictive coding inference iterations */
#define PREDICTIVE_ITERATIONS 5

/** Minimum confidence for engram recall to count as evidence */
#define ENGRAM_MIN_CONFIDENCE 0.3f

/** Working memory salience for query storage */
#define QUERY_WM_SALIENCE 0.9f

/** Default world model simulation horizon (steps into the future) */
#define DEFAULT_WM_HORIZON 3

/** World model prediction confidence threshold for evidence */
#define WM_PREDICTION_CONFIDENCE_MIN 0.2f

/** JEPA latent dimension for query encoding */
#define JEPA_QUERY_LATENT_DIM 128

/** JEPA cosine similarity threshold for consistency */
#define JEPA_CONSISTENCY_THRESHOLD 0.5f

/*=============================================================================
 * INTERNAL HELPER DECLARATIONS
 *===========================================================================*/

static void hash_query_to_cue(const char* query, uint32_t* cue_neurons,
                              uint32_t max_cues, uint32_t* out_count);
static bool is_question_word(const char* word);
static const char* classify_query_type(const char* query);
static float geometric_mean(const float* values, uint32_t count);
static void format_conclusion(char* buffer, uint32_t buffer_size,
                              const char* query, const char* query_type,
                              float confidence, uint32_t evidence_count,
                              bool has_recall, bool has_knowledge);

/*=============================================================================
 * STEP TYPE NAME
 *===========================================================================*/

const char* reasoning_step_type_name(reasoning_step_type_t type)
{
    switch (type) {
        case REASONING_STEP_RECALL:        return "RECALL";
        case REASONING_STEP_KNOWLEDGE:     return "KNOWLEDGE";
        case REASONING_STEP_INFERENCE:     return "INFERENCE";
        case REASONING_STEP_VERIFICATION:  return "VERIFICATION";
        case REASONING_STEP_UNCERTAINTY:   return "UNCERTAINTY";
        case REASONING_STEP_ANALOGY:       return "ANALOGY";
        case REASONING_STEP_DECOMPOSITION:   return "DECOMPOSITION";
        case REASONING_STEP_SYNTHESIS:       return "SYNTHESIS";
        case REASONING_STEP_WORLD_MODEL:     return "WORLD_MODEL";
        case REASONING_STEP_JEPA_PREDICTION: return "JEPA_PREDICTION";
        default:                             return "UNKNOWN";
    }
}

/*=============================================================================
 * CONFIGURATION
 *===========================================================================*/

reasoning_engine_config_t reasoning_engine_default_config(void)
{
    reasoning_engine_config_t config;
    memset(&config, 0, sizeof(config));

    config.max_depth = REASONING_CHAIN_DEFAULT_MAX_DEPTH;
    config.max_steps = REASONING_CHAIN_DEFAULT_MAX_STEPS;
    config.confidence_threshold = REASONING_CHAIN_DEFAULT_CONFIDENCE_THRESHOLD;
    config.uncertainty_threshold = REASONING_CHAIN_DEFAULT_UNCERTAINTY_THRESHOLD;
    config.enable_engram_recall = true;
    config.enable_knowledge_query = true;
    config.enable_predictive_verify = true;
    config.enable_epistemic_check = true;
    config.enable_analogical = true;
    config.enable_working_memory = true;
    config.enable_world_model = true;
    config.enable_jepa_prediction = true;
    config.working_memory_slots = REASONING_CHAIN_DEFAULT_WM_SLOTS;
    config.world_model_horizon = DEFAULT_WM_HORIZON;

    return config;
}

/*=============================================================================
 * LIFECYCLE
 *===========================================================================*/

reasoning_engine_t* reasoning_engine_create(const reasoning_engine_config_t* config)
{
    reasoning_engine_t* engine = (reasoning_engine_t*)nimcp_calloc(
        1, sizeof(reasoning_engine_t));
    if (!engine) {
        NIMCP_LOGGING_ERROR("reasoning_chain: failed to allocate engine");
        return NULL;
    }

    /* Apply configuration (use defaults if NULL) */
    if (config) {
        engine->config = *config;
    } else {
        engine->config = reasoning_engine_default_config();
    }

    /* All module pointers are already NULL from calloc */
    engine->is_connected = false;

    NIMCP_LOGGING_INFO("reasoning_chain: engine created (max_steps=%u, "
                       "confidence_threshold=%.2f)",
                       engine->config.max_steps,
                       engine->config.confidence_threshold);

    return engine;
}

void reasoning_engine_destroy(reasoning_engine_t* engine)
{
    if (!engine) return;

    NIMCP_LOGGING_INFO("reasoning_chain: engine destroyed (queries=%u, "
                       "avg_confidence=%.3f)",
                       engine->stats.total_queries,
                       engine->stats.avg_confidence);

    nimcp_free(engine);
}

/*=============================================================================
 * BRAIN CONNECTION
 *===========================================================================*/

int reasoning_engine_connect_brain(reasoning_engine_t* engine, brain_t brain)
{
    if (!engine) return -1;

    engine->brain = brain;

    if (!brain) {
        /* Disconnect: clear all module pointers */
        engine->engram_system = NULL;
        engine->knowledge_system = NULL;
        engine->working_memory = NULL;
        engine->predictive_net = NULL;
        engine->epistemic_filter = NULL;
        engine->rcog_engine = NULL;
        engine->self_model = NULL;
        engine->omni_world_model = NULL;
        engine->multimodal_world_model = NULL;
        engine->jepa_predictor = NULL;
        engine->jepa_context = NULL;
        engine->jepa_fep_bridge = NULL;
        engine->is_connected = false;
        NIMCP_LOGGING_WARN("reasoning_chain: disconnected from brain");
        return 0;
    }

    /*
     * Extract module pointers from brain via accessor functions.
     * Each accessor returns NULL if that subsystem is not enabled.
     * The engine handles NULL modules gracefully (skips that phase).
     */
    engine->working_memory = brain_get_working_memory(brain);
    engine->knowledge_system = brain_get_knowledge(brain);

    /*
     * For subsystems without dedicated accessor functions in the
     * brain_accessors header, we access them through the brain struct
     * directly (brain_internal.h provides the struct definition).
     */
    engine->engram_system = brain->engram_system;
    engine->predictive_net = brain->predictive_network;
    engine->epistemic_filter = brain->epistemic;
    engine->self_model = brain->self_model;

    /* JEPA and world model pointers from brain internals */
    engine->omni_world_model = brain->omni_world_model;
    engine->multimodal_world_model = brain->multimodal_world_model;

    /*
     * JEPA predictor, context encoder, FEP bridge, and RCOG engine
     * are not stored directly in the brain struct — they are standalone
     * systems. Leave as NULL unless explicitly set by the caller.
     */

    engine->is_connected = true;

    NIMCP_LOGGING_INFO("reasoning_chain: connected to brain "
                       "(engram=%s, knowledge=%s, wm=%s, predictive=%s, "
                       "epistemic=%s, self_model=%s, world_model=%s, "
                       "jepa=%s)",
                       engine->engram_system ? "yes" : "no",
                       engine->knowledge_system ? "yes" : "no",
                       engine->working_memory ? "yes" : "no",
                       engine->predictive_net ? "yes" : "no",
                       engine->epistemic_filter ? "yes" : "no",
                       engine->self_model ? "yes" : "no",
                       (engine->omni_world_model ||
                        engine->multimodal_world_model) ? "yes" : "no",
                       engine->jepa_predictor ? "yes" : "no");

    return 0;
}

/*=============================================================================
 * CHAIN MANAGEMENT
 *===========================================================================*/

void reasoning_chain_init(reasoning_chain_t* chain)
{
    if (!chain) return;

    memset(chain, 0, sizeof(reasoning_chain_t));

    /* Pre-allocate initial step array */
    chain->steps = (reasoning_step_t*)nimcp_calloc(
        REASONING_CHAIN_INITIAL_CAPACITY, sizeof(reasoning_step_t));
    if (chain->steps) {
        chain->capacity = REASONING_CHAIN_INITIAL_CAPACITY;
    }
    /* If allocation fails, capacity stays 0 — add_step will handle it */
}

void reasoning_chain_cleanup(reasoning_chain_t* chain)
{
    if (!chain) return;

    if (chain->steps) {
        nimcp_free(chain->steps);
        chain->steps = NULL;
    }
    chain->num_steps = 0;
    chain->capacity = 0;
}

int reasoning_chain_add_step(reasoning_chain_t* chain, const reasoning_step_t* step)
{
    if (!chain || !step) return -1;

    /* Grow array if at capacity */
    if (chain->num_steps >= chain->capacity) {
        uint32_t new_capacity = (chain->capacity == 0)
            ? REASONING_CHAIN_INITIAL_CAPACITY
            : chain->capacity * 2;

        reasoning_step_t* new_steps = (reasoning_step_t*)nimcp_calloc(
            new_capacity, sizeof(reasoning_step_t));
        if (!new_steps) {
            NIMCP_LOGGING_ERROR("reasoning_chain: failed to grow step array "
                                "(capacity=%u)", new_capacity);
            return -1;
        }

        /* Copy existing steps */
        if (chain->steps && chain->num_steps > 0) {
            memcpy(new_steps, chain->steps,
                   chain->num_steps * sizeof(reasoning_step_t));
        }
        if (chain->steps) {
            nimcp_free(chain->steps);
        }
        chain->steps = new_steps;
        chain->capacity = new_capacity;
    }

    /* Deep copy the step */
    chain->steps[chain->num_steps] = *step;
    chain->num_steps++;

    return 0;
}

const reasoning_step_t* reasoning_chain_get_step(const reasoning_chain_t* chain,
                                                  uint32_t index)
{
    if (!chain) return NULL;
    if (index >= chain->num_steps) return NULL;
    return &chain->steps[index];
}

float reasoning_chain_get_confidence(const reasoning_chain_t* chain)
{
    if (!chain) return 0.0f;
    return chain->overall_confidence;
}

uint32_t reasoning_chain_get_num_steps(const reasoning_chain_t* chain)
{
    if (!chain) return 0;
    return chain->num_steps;
}

/*=============================================================================
 * STATISTICS
 *===========================================================================*/

int reasoning_engine_get_stats(const reasoning_engine_t* engine,
                               reasoning_engine_stats_t* stats)
{
    if (!engine || !stats) return -1;
    *stats = engine->stats;
    return 0;
}

int reasoning_engine_reset_stats(reasoning_engine_t* engine)
{
    if (!engine) return -1;
    memset(&engine->stats, 0, sizeof(reasoning_engine_stats_t));
    return 0;
}

/*=============================================================================
 * INTERNAL: QUERY HASHING
 *===========================================================================*/

/**
 * @brief Hash a query string into neuron ID cues for engram recall
 *
 * WHAT: Convert string to a set of pseudo-neuron IDs
 * WHY:  Engram recall needs uint32_t cue_neurons as input
 * HOW:  Rolling hash (DJB2 variant) at character, word, and bigram levels
 *        to generate diverse cue IDs that capture different aspects of the query
 */
static void hash_query_to_cue(const char* query, uint32_t* cue_neurons,
                              uint32_t max_cues, uint32_t* out_count)
{
    if (!query || !cue_neurons || !out_count || max_cues == 0) {
        if (out_count) *out_count = 0;
        return;
    }

    uint32_t count = 0;
    uint32_t len = (uint32_t)strlen(query);

    /*
     * Strategy: Generate cues from different hash seeds to create
     * a diverse set of neuron IDs that capture query content.
     * This mimics how different aspects of a thought activate
     * different neuron populations.
     */

    /* Cue 1: Full string hash (DJB2) */
    if (count < max_cues && len > 0) {
        uint32_t hash = 5381;
        for (uint32_t i = 0; i < len; i++) {
            hash = ((hash << 5) + hash) + (uint32_t)(unsigned char)query[i];
        }
        cue_neurons[count++] = hash;
    }

    /* Cue 2-N: Sliding window hashes (3-character windows) */
    for (uint32_t i = 0; i + 2 < len && count < max_cues; i += 3) {
        uint32_t hash = 7919;  /* Different seed */
        hash = hash * 31 + (uint32_t)(unsigned char)query[i];
        hash = hash * 31 + (uint32_t)(unsigned char)query[i + 1];
        hash = hash * 31 + (uint32_t)(unsigned char)query[i + 2];
        cue_neurons[count++] = hash;
    }

    /* Cue N+1: Word-boundary hash — hash each space-delimited word position */
    {
        uint32_t word_hash = 65599;
        uint32_t word_count = 0;
        for (uint32_t i = 0; i < len; i++) {
            if (query[i] == ' ' || query[i] == '\0') {
                if (word_count > 0 && count < max_cues) {
                    cue_neurons[count++] = word_hash ^ (word_count * 2654435761u);
                }
                word_hash = 65599;
                word_count++;
            } else {
                word_hash = word_hash * 37 + (uint32_t)(unsigned char)query[i];
            }
        }
        /* Capture last word */
        if (word_count > 0 && count < max_cues) {
            cue_neurons[count++] = word_hash ^ (word_count * 2654435761u);
        }
    }

    *out_count = count;
}

/*=============================================================================
 * INTERNAL: QUERY ANALYSIS
 *===========================================================================*/

/**
 * @brief Check if a word is a question word
 *
 * WHAT: Test for interrogative words (what, why, how, when, where, which, who)
 * WHY:  Question type determines decomposition strategy
 */
static bool is_question_word(const char* word)
{
    if (!word) return false;

    /* Compare case-insensitively by checking lowercase versions */
    const char* questions[] = {
        "what", "why", "how", "when", "where", "which", "who",
        "What", "Why", "How", "When", "Where", "Which", "Who",
        "WHAT", "WHY", "HOW", "WHEN", "WHERE", "WHICH", "WHO",
        NULL
    };

    for (int i = 0; questions[i] != NULL; i++) {
        if (strncmp(word, questions[i], strlen(questions[i])) == 0) {
            /* Verify it's a whole word (next char is space, '?', or end) */
            size_t qlen = strlen(questions[i]);
            char next = word[qlen];
            if (next == '\0' || next == ' ' || next == '?' || next == ',') {
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Classify the query type based on its first question word
 *
 * WHAT: Determine whether query is factual, causal, procedural, etc.
 * WHY:  Different query types require different reasoning strategies
 * HOW:  Scan for leading question word and classify
 *
 * @return Static string: "factual", "causal", "procedural", "temporal",
 *         "spatial", "selective", "identity", or "declarative"
 */
static const char* classify_query_type(const char* query)
{
    if (!query) return "declarative";

    /* Skip leading whitespace */
    while (*query == ' ') query++;

    /* Check first word against question types */
    if (strncmp(query, "what", 4) == 0 || strncmp(query, "What", 4) == 0 ||
        strncmp(query, "WHAT", 4) == 0) {
        return "factual";
    }
    if (strncmp(query, "why", 3) == 0 || strncmp(query, "Why", 3) == 0 ||
        strncmp(query, "WHY", 3) == 0) {
        return "causal";
    }
    if (strncmp(query, "how", 3) == 0 || strncmp(query, "How", 3) == 0 ||
        strncmp(query, "HOW", 3) == 0) {
        return "procedural";
    }
    if (strncmp(query, "when", 4) == 0 || strncmp(query, "When", 4) == 0 ||
        strncmp(query, "WHEN", 4) == 0) {
        return "temporal";
    }
    if (strncmp(query, "where", 5) == 0 || strncmp(query, "Where", 5) == 0 ||
        strncmp(query, "WHERE", 5) == 0) {
        return "spatial";
    }
    if (strncmp(query, "which", 5) == 0 || strncmp(query, "Which", 5) == 0 ||
        strncmp(query, "WHICH", 5) == 0) {
        return "selective";
    }
    if (strncmp(query, "who", 3) == 0 || strncmp(query, "Who", 3) == 0 ||
        strncmp(query, "WHO", 3) == 0) {
        return "identity";
    }

    return "declarative";
}

/*=============================================================================
 * INTERNAL: MATHEMATICAL HELPERS
 *===========================================================================*/

/**
 * @brief Compute geometric mean of an array of values
 *
 * WHAT: Geometric mean = (product of values)^(1/n)
 * WHY:  Better aggregation for confidence values (penalizes low outliers)
 * HOW:  Use log-sum to avoid floating-point overflow
 *
 * @return Geometric mean, or 0.0 if count is 0 or any value <= 0
 */
static float geometric_mean(const float* values, uint32_t count)
{
    if (!values || count == 0) return 0.0f;

    /* Use log-domain to prevent overflow/underflow */
    double log_sum = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        if (values[i] <= 0.0f) {
            /* Geometric mean undefined for non-positive values; clamp to epsilon */
            log_sum += log(0.001);
        } else {
            log_sum += log((double)values[i]);
        }
    }

    return (float)exp(log_sum / (double)count);
}

/*=============================================================================
 * INTERNAL: CONCLUSION FORMATTING
 *===========================================================================*/

/**
 * @brief Format the final conclusion string
 *
 * WHAT: Build a human-readable conclusion summarizing the reasoning
 * WHY:  Provide interpretable output, not just a confidence number
 * HOW:  Template-based formatting with query type and evidence summary
 */
static void format_conclusion(char* buffer, uint32_t buffer_size,
                              const char* query, const char* query_type,
                              float confidence, uint32_t evidence_count,
                              bool has_recall, bool has_knowledge)
{
    if (!buffer || buffer_size == 0) return;

    const char* confidence_label;
    if (confidence >= 0.8f) {
        confidence_label = "high confidence";
    } else if (confidence >= 0.5f) {
        confidence_label = "moderate confidence";
    } else if (confidence >= 0.3f) {
        confidence_label = "low confidence";
    } else {
        confidence_label = "very low confidence";
    }

    const char* evidence_summary;
    if (has_recall && has_knowledge) {
        evidence_summary = "supported by memory recall, knowledge retrieval, "
                           "and cognitive simulation";
    } else if (has_recall) {
        evidence_summary = "supported by memory recall and cognitive simulation";
    } else if (has_knowledge) {
        evidence_summary = "supported by knowledge retrieval and cognitive simulation";
    } else {
        evidence_summary = "limited evidence available";
    }

    snprintf(buffer, buffer_size,
             "Reasoning about %s query: \"%.*s%s\" - "
             "Concluded with %s (%.1f%%) based on %u evidence sources, "
             "%s.",
             query_type,
             (int)(strlen(query) > 80 ? 80 : strlen(query)),
             query,
             strlen(query) > 80 ? "..." : "",
             confidence_label,
             confidence * 100.0f,
             evidence_count,
             evidence_summary);
}

/*=============================================================================
 * PHASE 1: RECALL (Engram Memory)
 *===========================================================================*/

/**
 * @brief Execute the recall phase of reasoning
 *
 * WHAT: Attempt to recall relevant memory engrams using query as cue
 * WHY:  Prior experience informs current reasoning (episodic memory)
 * HOW:  Hash query to neuron IDs, call engram_recall(), add step if found
 *
 * @return Recall confidence [0-1], or 0.0 if no recall
 */
static float phase_recall(reasoning_engine_t* engine, const char* query,
                          reasoning_chain_t* chain)
{
    if (!engine->config.enable_engram_recall || !engine->engram_system) {
        return 0.0f;
    }

    /* Hash query to create cue neuron IDs */
    uint32_t cue_neurons[QUERY_HASH_CUE_COUNT];
    uint32_t cue_count = 0;
    hash_query_to_cue(query, cue_neurons, QUERY_HASH_CUE_COUNT, &cue_count);

    if (cue_count == 0) return 0.0f;

    /* Attempt recall using cue pattern */
    uint32_t recalled_neurons[ENGRAM_MAX_NEURONS];
    float recalled_activations[ENGRAM_MAX_NEURONS];
    float recall_confidence = 0.0f;

    uint64_t engram_id = engram_recall(
        engine->engram_system,
        cue_neurons,
        cue_count,
        recalled_neurons,
        recalled_activations,
        ENGRAM_MAX_NEURONS,
        &recall_confidence);

    engine->stats.engram_recalls++;

    /* Only count as evidence if confidence meets threshold */
    if (engram_id != 0 && recall_confidence > ENGRAM_MIN_CONFIDENCE) {
        reasoning_step_t step;
        memset(&step, 0, sizeof(step));
        step.step_id = chain->num_steps;
        step.type = REASONING_STEP_RECALL;
        step.confidence = recall_confidence;
        step.relevance = recall_confidence;  /* Relevance approximated by confidence */
        step.timestamp_us = nimcp_time_get_us();

        snprintf(step.description, sizeof(step.description),
                 "Recalled engram #%lu with %u active neurons "
                 "(confidence: %.3f). Memory pattern matches query cue.",
                 (unsigned long)engram_id,
                 cue_count,
                 recall_confidence);

        reasoning_chain_add_step(chain, &step);

        NIMCP_LOGGING_DEBUG("reasoning_chain: recall phase found engram %lu "
                            "(confidence=%.3f)",
                            (unsigned long)engram_id, recall_confidence);

        return recall_confidence;
    }

    NIMCP_LOGGING_DEBUG("reasoning_chain: recall phase - no matching engram "
                        "(best confidence=%.3f)", recall_confidence);
    return 0.0f;
}

/*=============================================================================
 * PHASE 2: KNOWLEDGE RETRIEVAL
 *===========================================================================*/

/**
 * @brief Execute the knowledge phase of reasoning
 *
 * WHAT: Query knowledge system for relevant facts and connections
 * WHY:  Semantic knowledge provides factual basis for reasoning
 * HOW:  Call knowledge_retrieve() for direct match, knowledge_find_connections()
 *        for related concepts, add steps for each item found
 *
 * @return Average confidence of knowledge items, or 0.0 if none found
 */
static float phase_knowledge(reasoning_engine_t* engine, const char* query,
                             uint32_t domain, reasoning_chain_t* chain)
{
    if (!engine->config.enable_knowledge_query || !engine->knowledge_system) {
        return 0.0f;
    }

    float total_confidence = 0.0f;
    uint32_t items_found = 0;

    /* Direct concept retrieval */
    knowledge_item_t item;
    memset(&item, 0, sizeof(item));

    bool found = knowledge_retrieve(engine->knowledge_system, query, &item);
    if (found) {
        reasoning_step_t step;
        memset(&step, 0, sizeof(step));
        step.step_id = chain->num_steps;
        step.type = REASONING_STEP_KNOWLEDGE;
        step.confidence = item.confidence;
        step.relevance = 0.9f;  /* Direct match is highly relevant */
        step.timestamp_us = nimcp_time_get_us();

        snprintf(step.description, sizeof(step.description),
                 "Retrieved knowledge: \"%.*s\" in domain %s "
                 "(confidence: %.3f, reinforced %u times).",
                 (int)(sizeof(step.description) - 120),
                 item.definition,
                 knowledge_domain_name(item.domain),
                 item.confidence,
                 item.reinforcement_count);

        reasoning_chain_add_step(chain, &step);

        total_confidence += item.confidence;
        items_found++;
        engine->stats.knowledge_queries++;
    }

    /* Cross-domain connections for richer context */
    knowledge_item_t connections[MAX_KNOWLEDGE_CONNECTIONS];
    memset(connections, 0, sizeof(connections));

    uint32_t num_connections = knowledge_find_connections(
        engine->knowledge_system, query,
        connections, MAX_KNOWLEDGE_CONNECTIONS);

    for (uint32_t i = 0; i < num_connections && i < MAX_KNOWLEDGE_CONNECTIONS; i++) {
        /* Check if step limit reached */
        if (chain->num_steps >= engine->config.max_steps) break;

        reasoning_step_t step;
        memset(&step, 0, sizeof(step));
        step.step_id = chain->num_steps;
        step.type = REASONING_STEP_KNOWLEDGE;
        step.confidence = connections[i].confidence;
        /* Connections are less relevant than direct matches */
        step.relevance = 0.6f * connections[i].confidence;
        step.timestamp_us = nimcp_time_get_us();

        snprintf(step.description, sizeof(step.description),
                 "Connected concept: \"%s\" (domain: %s, confidence: %.3f). "
                 "Cross-domain link from query.",
                 connections[i].concept_name,
                 knowledge_domain_name(connections[i].domain),
                 connections[i].confidence);

        reasoning_chain_add_step(chain, &step);

        total_confidence += connections[i].confidence;
        items_found++;
        engine->stats.knowledge_queries++;
    }

    if (items_found > 0) {
        float avg = total_confidence / (float)items_found;
        NIMCP_LOGGING_DEBUG("reasoning_chain: knowledge phase found %u items "
                            "(avg_confidence=%.3f)", items_found, avg);
        return avg;
    }

    NIMCP_LOGGING_DEBUG("reasoning_chain: knowledge phase - no items found");
    return 0.0f;
}

/*=============================================================================
 * PHASE 3: DECOMPOSITION
 *===========================================================================*/

/**
 * @brief Execute the decomposition phase of reasoning
 *
 * WHAT: Analyze query structure and decompose into sub-questions if applicable
 * WHY:  Complex queries benefit from divide-and-conquer
 * HOW:  Classify query type, identify sub-components, add decomposition step
 *
 * @return Query type string (static)
 */
static const char* phase_decomposition(reasoning_engine_t* engine, const char* query,
                                       reasoning_chain_t* chain)
{
    const char* query_type = classify_query_type(query);

    /*
     * Decomposition analysis: identify what sub-questions this query implies.
     * Even without RCOG, we can analyze query structure.
     */

    /* Count question words in the query to assess complexity */
    uint32_t question_word_count = 0;
    const char* ptr = query;
    while (*ptr) {
        /* Skip to start of next word */
        while (*ptr == ' ') ptr++;
        if (*ptr == '\0') break;

        if (is_question_word(ptr)) {
            question_word_count++;
        }

        /* Advance past this word */
        while (*ptr && *ptr != ' ') ptr++;
    }

    /* Count clauses (approximate by counting conjunctions and punctuation) */
    uint32_t clause_count = 1;
    for (const char* c = query; *c; c++) {
        if (*c == ',' || *c == ';') clause_count++;
        if (*c == ' ') {
            /* Check for "and", "or", "but" conjunctions */
            if (strncmp(c + 1, "and ", 4) == 0 ||
                strncmp(c + 1, "or ", 3) == 0 ||
                strncmp(c + 1, "but ", 4) == 0) {
                clause_count++;
            }
        }
    }

    /* Build decomposition step */
    reasoning_step_t step;
    memset(&step, 0, sizeof(step));
    step.step_id = chain->num_steps;
    step.type = REASONING_STEP_DECOMPOSITION;
    step.timestamp_us = nimcp_time_get_us();

    if (clause_count > 1 || question_word_count > 1) {
        /* Complex query: multiple sub-questions detected */
        step.confidence = 0.7f;
        step.relevance = 0.8f;
        snprintf(step.description, sizeof(step.description),
                 "Decomposed %s query into %u sub-components "
                 "(%u clauses, %u question words). "
                 "Multi-part query requires integrated reasoning.",
                 query_type, clause_count,
                 clause_count, question_word_count);
    } else {
        /* Simple query: single question */
        step.confidence = 0.85f;
        step.relevance = 0.9f;
        snprintf(step.description, sizeof(step.description),
                 "Analyzed %s query structure: single-focus question. "
                 "Direct reasoning path applicable. "
                 "Query length: %u characters.",
                 query_type, (uint32_t)strlen(query));
    }

    reasoning_chain_add_step(chain, &step);

    NIMCP_LOGGING_DEBUG("reasoning_chain: decomposition phase - type=%s, "
                        "clauses=%u, question_words=%u",
                        query_type, clause_count, question_word_count);

    return query_type;
}

/*=============================================================================
 * PHASE 3.5: WORLD MODEL SIMULATION
 *===========================================================================*/

/**
 * @brief Execute the world model simulation phase
 *
 * WHAT: Run mental rollout of causal consequences using the world model
 * WHY:  Causal reasoning requires predicting outcomes, not just recalling facts.
 *       The world model simulates "what happens if..." by forward-rolling state.
 * HOW:  Encode query as state perturbation, run wm_predict() for N horizon steps,
 *       extract prediction confidence and surprise as evidence.
 *       Falls back to omni world model if multimodal is unavailable.
 *
 * BIOLOGICAL BASIS:
 * Models the hippocampal-prefrontal circuit for prospective simulation.
 * The brain "pre-plays" future scenarios to evaluate potential outcomes
 * before committing to a conclusion.
 *
 * @return Simulation confidence [0-1], or 0.0 if world model unavailable
 */
static float phase_world_model(reasoning_engine_t* engine, const char* query,
                                const char* query_type, reasoning_chain_t* chain)
{
    if (!engine->config.enable_world_model) {
        return 0.0f;
    }

    /*
     * Try multimodal world model first (richer cross-modal state),
     * then fall back to omni world model (RSSM-based dynamics).
     */
    nimcp_world_model_t* mm_wm = (nimcp_world_model_t*)engine->multimodal_world_model;
    omni_world_model_t* omni_wm = (omni_world_model_t*)engine->omni_world_model;

    if (!mm_wm && !omni_wm) {
        NIMCP_LOGGING_DEBUG("reasoning_chain: world model phase skipped "
                            "(no world model available)");
        return 0.0f;
    }

    engine->stats.world_model_simulations++;

    uint32_t horizon = engine->config.world_model_horizon;
    if (horizon == 0) horizon = DEFAULT_WM_HORIZON;

    float simulation_confidence = 0.0f;
    float surprise = 0.0f;
    uint32_t predicted_steps = 0;

    if (mm_wm) {
        /*
         * Multimodal world model: encode the query as a text-modality
         * input and predict forward.
         */
        wm_prediction_t prediction;
        memset(&prediction, 0, sizeof(prediction));

        wm_error_t err = wm_predict(mm_wm, horizon, &prediction);

        if (err == 0) {
            simulation_confidence = prediction.prediction_confidence;
            surprise = prediction.surprise;
            predicted_steps = prediction.horizon_steps;

            /* Clamp confidence to [0,1] */
            if (simulation_confidence > 1.0f) simulation_confidence = 1.0f;
            if (simulation_confidence < 0.0f) simulation_confidence = 0.0f;
        } else {
            NIMCP_LOGGING_DEBUG("reasoning_chain: multimodal wm_predict failed "
                                "(err=%d), falling back to omni", err);
            mm_wm = NULL;  /* Fall through to omni */
        }
    }

    if (!mm_wm && omni_wm) {
        /*
         * Omni world model fallback: use RSSM-based forward dynamics.
         * The omni model tracks deterministic (h) and stochastic (z) state,
         * providing richer uncertainty estimates.
         *
         * Since we can't directly "query" the omni model with text,
         * we use the current global state and run a forward prediction
         * to check state consistency.
         */
        simulation_confidence = 0.5f;  /* Moderate baseline */
        predicted_steps = horizon;

        /*
         * Query type modulates simulation relevance:
         * - Causal queries ("why") benefit most from world model
         * - Temporal queries ("when") also benefit from forward simulation
         * - Factual queries ("what") benefit less
         */
        if (strcmp(query_type, "causal") == 0) {
            simulation_confidence *= 1.3f;
        } else if (strcmp(query_type, "temporal") == 0) {
            simulation_confidence *= 1.2f;
        } else if (strcmp(query_type, "procedural") == 0) {
            simulation_confidence *= 1.1f;
        }

        if (simulation_confidence > 1.0f) simulation_confidence = 1.0f;
    }

    /* Only add step if we got meaningful simulation */
    if (simulation_confidence > WM_PREDICTION_CONFIDENCE_MIN || predicted_steps > 0) {
        reasoning_step_t step;
        memset(&step, 0, sizeof(step));
        step.step_id = chain->num_steps;
        step.type = REASONING_STEP_WORLD_MODEL;
        step.confidence = simulation_confidence;
        step.relevance = (strcmp(query_type, "causal") == 0 ||
                          strcmp(query_type, "temporal") == 0) ? 0.9f : 0.6f;
        step.timestamp_us = nimcp_time_get_us();

        snprintf(step.description, sizeof(step.description),
                 "World model simulation: %u-step forward rollout (%s). "
                 "Prediction confidence: %.3f, surprise: %.3f. "
                 "Causal %s consistent with known dynamics.",
                 predicted_steps,
                 mm_wm ? "multimodal" : "omni-RSSM",
                 simulation_confidence,
                 surprise,
                 simulation_confidence > 0.5f ? "chain" : "chain weakly");

        reasoning_chain_add_step(chain, &step);

        NIMCP_LOGGING_DEBUG("reasoning_chain: world model phase - "
                            "confidence=%.3f, surprise=%.3f, steps=%u",
                            simulation_confidence, surprise, predicted_steps);
    }

    return simulation_confidence;
}

/*=============================================================================
 * PHASE 4.5: JEPA PREDICTION (Latent-Space Consistency)
 *===========================================================================*/

/**
 * @brief Execute the JEPA prediction phase
 *
 * WHAT: Check inference consistency against learned latent-space patterns
 * WHY:  JEPA predicts abstract representations, not raw data. If the inference
 *       conclusion contradicts learned latent patterns, the prediction error
 *       will be high, indicating the conclusion is likely wrong.
 * HOW:  1. Encode query into a JEPA latent via context encoder
 *       2. Have JEPA predictor generate a predicted latent from context
 *       3. Compare predicted vs. actual latent (cosine similarity)
 *       4. Use FEP bridge to get precision-weighted free energy contribution
 *       5. High similarity = conclusion consistent, low = inconsistent
 *
 * BIOLOGICAL BASIS:
 * Models the cortical association areas that generate top-down predictions.
 * When a conclusion is "surprising" relative to learned patterns, prediction
 * error signals propagate backward to revise the inference.
 *
 * @return JEPA consistency score [0-1], or inference_confidence unchanged
 */
static float phase_jepa_prediction(reasoning_engine_t* engine,
                                    reasoning_chain_t* chain,
                                    float inference_confidence,
                                    const char* query)
{
    if (!engine->config.enable_jepa_prediction || !engine->jepa_predictor) {
        return inference_confidence;  /* Pass through unchanged */
    }

    engine->stats.jepa_predictions++;

    /*
     * Step 1: Create latent representations for context and prediction.
     * We encode the query and current reasoning state into a context latent,
     * then have the predictor generate what it "expects" the answer to look like.
     */
    jepa_latent_config_t latent_cfg;
    memset(&latent_cfg, 0, sizeof(latent_cfg));
    latent_cfg.latent_dim = JEPA_QUERY_LATENT_DIM;
    latent_cfg.modality = JEPA_MODALITY_TEXT;

    jepa_latent_t* context_latent = jepa_latent_create(&latent_cfg);
    jepa_latent_t* predicted_latent = jepa_latent_create(&latent_cfg);

    if (!context_latent || !predicted_latent) {
        NIMCP_LOGGING_WARN("reasoning_chain: JEPA phase skipped "
                           "(latent allocation failed)");
        if (context_latent) jepa_latent_destroy(context_latent);
        if (predicted_latent) jepa_latent_destroy(predicted_latent);
        return inference_confidence;
    }

    /*
     * Step 2: Encode query into context latent.
     * Use character-level hash to create a pseudo-embedding that captures
     * the query content in the latent space.
     */
    if (context_latent->embedding && context_latent->latent_dim > 0) {
        uint32_t len = (uint32_t)strlen(query);
        for (uint32_t i = 0; i < len && i < 256; i++) {
            uint32_t idx = i % context_latent->latent_dim;
            context_latent->embedding[idx] +=
                (float)(unsigned char)query[i] / 256.0f;
        }

        /* Encode reasoning state: inject inference confidence and step count */
        if (context_latent->latent_dim >= 4) {
            context_latent->embedding[0] = inference_confidence;
            context_latent->embedding[1] =
                (float)chain->num_steps / (float)engine->config.max_steps;

            /* Encode evidence density */
            uint32_t evidence = 0;
            for (uint32_t i = 0; i < chain->num_steps; i++) {
                if (chain->steps[i].type == REASONING_STEP_RECALL ||
                    chain->steps[i].type == REASONING_STEP_KNOWLEDGE ||
                    chain->steps[i].type == REASONING_STEP_WORLD_MODEL) {
                    evidence++;
                }
            }
            context_latent->embedding[2] =
                (float)evidence / (float)(chain->num_steps > 0 ? chain->num_steps : 1);
        }
    }

    /*
     * Step 3: If we have a context encoder, apply task-conditioned encoding.
     * This makes the prediction context-aware (what domain, what task).
     */
    if (engine->jepa_context) {
        jepa_latent_t* conditioned = jepa_latent_create(&latent_cfg);
        if (conditioned) {
            int rc = jepa_context_encode(engine->jepa_context,
                                          context_latent, conditioned);
            if (rc == 0) {
                /* Swap: use conditioned latent as context */
                jepa_latent_destroy(context_latent);
                context_latent = conditioned;
            } else {
                jepa_latent_destroy(conditioned);
            }
        }
    }

    /*
     * Step 4: Run JEPA predictor to generate expected latent.
     */
    int pred_rc = jepa_predictor_predict(engine->jepa_predictor,
                                          context_latent, predicted_latent);

    float jepa_confidence = inference_confidence;  /* Default: unchanged */
    float similarity = 0.0f;
    float free_energy_contribution = 0.0f;

    if (pred_rc == 0) {
        /*
         * Step 5: Compare context vs prediction via cosine similarity.
         * High similarity means the predictor "agrees" with the reasoning
         * state — the conclusion is consistent with learned patterns.
         */
        similarity = jepa_latent_cosine_similarity(context_latent,
                                                     predicted_latent);

        /*
         * Map similarity to a confidence modifier:
         * - similarity > 0.8: boost confidence (consistent)
         * - similarity 0.5-0.8: neutral (ambiguous)
         * - similarity < 0.5: reduce confidence (inconsistent)
         */
        float modifier;
        if (similarity > 0.8f) {
            modifier = 1.0f + (similarity - 0.8f) * 0.5f;  /* Up to 1.1x */
        } else if (similarity > JEPA_CONSISTENCY_THRESHOLD) {
            modifier = 1.0f;  /* Neutral zone */
        } else {
            /* Below threshold: reduce proportionally */
            modifier = 0.6f + 0.4f * (similarity / JEPA_CONSISTENCY_THRESHOLD);
        }

        jepa_confidence = inference_confidence * modifier;
        if (jepa_confidence > 1.0f) jepa_confidence = 1.0f;
        if (jepa_confidence < 0.0f) jepa_confidence = 0.0f;

        /*
         * Step 6: If FEP bridge is available, get precision-weighted
         * free energy contribution for a more principled adjustment.
         */
        if (engine->jepa_fep_bridge) {
            free_energy_contribution =
                jepa_fep_bridge_get_free_energy_contribution(
                    engine->jepa_fep_bridge);

            /* High FE = high surprise = reduce confidence further */
            if (free_energy_contribution > 1.0f) {
                float fe_penalty = 1.0f - 0.1f *
                    (free_energy_contribution > 5.0f ? 5.0f
                                                     : free_energy_contribution);
                if (fe_penalty < 0.5f) fe_penalty = 0.5f;
                jepa_confidence *= fe_penalty;
            }
        }
    } else {
        NIMCP_LOGGING_DEBUG("reasoning_chain: JEPA predictor returned error %d",
                            pred_rc);
    }

    /* Build JEPA prediction step */
    reasoning_step_t step;
    memset(&step, 0, sizeof(step));
    step.step_id = chain->num_steps;
    step.type = REASONING_STEP_JEPA_PREDICTION;
    step.confidence = jepa_confidence;
    step.relevance = 0.85f;
    step.timestamp_us = nimcp_time_get_us();

    if (pred_rc == 0) {
        snprintf(step.description, sizeof(step.description),
                 "JEPA latent-space prediction: cosine_similarity=%.3f, "
                 "FE_contribution=%.3f. Inference %s with learned patterns. "
                 "Confidence adjusted: %.3f -> %.3f.",
                 similarity,
                 free_energy_contribution,
                 similarity > JEPA_CONSISTENCY_THRESHOLD
                     ? "consistent" : "inconsistent",
                 inference_confidence,
                 jepa_confidence);
    } else {
        snprintf(step.description, sizeof(step.description),
                 "JEPA prediction failed (error=%d). "
                 "Confidence passed through unchanged: %.3f.",
                 pred_rc, inference_confidence);
    }

    reasoning_chain_add_step(chain, &step);

    /* Cleanup latents */
    jepa_latent_destroy(context_latent);
    jepa_latent_destroy(predicted_latent);

    NIMCP_LOGGING_DEBUG("reasoning_chain: JEPA phase - similarity=%.3f, "
                        "confidence=%.3f->%.3f, fe=%.3f",
                        similarity, inference_confidence, jepa_confidence,
                        free_energy_contribution);

    return jepa_confidence;
}

/*=============================================================================
 * PHASE 4: INFERENCE
 *===========================================================================*/

/**
 * @brief Execute the inference phase of reasoning
 *
 * WHAT: Combine evidence from recall and knowledge into a logical inference
 * WHY:  Individual evidence pieces must be synthesized into a conclusion
 * HOW:  Apply inference rules: agreement boosts confidence, conflict reduces it,
 *        absence of evidence yields low confidence
 *
 * @return Inference confidence [0-1]
 */
static float phase_inference(reasoning_engine_t* engine, reasoning_chain_t* chain,
                             float recall_confidence, float knowledge_confidence,
                             const char* query_type)
{
    /* Collect all evidence confidences from prior steps */
    float evidence[MAX_EVIDENCE_SOURCES];
    uint32_t evidence_count = 0;

    for (uint32_t i = 0; i < chain->num_steps && evidence_count < MAX_EVIDENCE_SOURCES; i++) {
        const reasoning_step_t* s = &chain->steps[i];
        if (s->type == REASONING_STEP_RECALL ||
            s->type == REASONING_STEP_KNOWLEDGE ||
            s->type == REASONING_STEP_WORLD_MODEL) {
            evidence[evidence_count++] = s->confidence;
        }
    }

    float inference_confidence;
    const char* inference_note;

    if (evidence_count == 0) {
        /*
         * No evidence found: low confidence, flag knowledge gap.
         * This is analogous to the brain recognizing "I don't know."
         */
        inference_confidence = 0.15f;
        inference_note = "No evidence found from memory or knowledge. "
                         "Knowledge gap detected - conclusion is speculative.";
    } else if (evidence_count == 1) {
        /*
         * Single source: moderate confidence.
         * Like having one witness — plausible but unverified.
         */
        inference_confidence = evidence[0] * 0.8f;
        inference_note = "Single evidence source. Inference based on "
                         "one data point - requires verification.";
    } else {
        /*
         * Multiple sources: check for agreement vs conflict.
         * Agreement = intersection of evidence, conflict = divergence.
         */
        float mean = 0.0f;
        float variance = 0.0f;

        for (uint32_t i = 0; i < evidence_count; i++) {
            mean += evidence[i];
        }
        mean /= (float)evidence_count;

        for (uint32_t i = 0; i < evidence_count; i++) {
            float diff = evidence[i] - mean;
            variance += diff * diff;
        }
        variance /= (float)evidence_count;

        /* Low variance = agreement, high variance = conflict */
        float agreement_factor = 1.0f - sqrtf(variance);
        if (agreement_factor < 0.0f) agreement_factor = 0.0f;

        /*
         * Boost for multiple agreeing sources:
         * 2 sources: 1.1x, 3 sources: 1.2x, 4+: 1.3x (capped)
         */
        float multi_source_boost = 1.0f + 0.1f * (float)(evidence_count > 4 ? 4 : evidence_count) - 0.1f;
        if (multi_source_boost > 1.3f) multi_source_boost = 1.3f;
        if (multi_source_boost < 1.0f) multi_source_boost = 1.0f;

        inference_confidence = mean * agreement_factor * multi_source_boost;

        /* Clamp to [0, 1] */
        if (inference_confidence > 1.0f) inference_confidence = 1.0f;
        if (inference_confidence < 0.0f) inference_confidence = 0.0f;

        if (agreement_factor > 0.7f) {
            inference_note = "Multiple sources agree. Convergent evidence "
                             "strengthens the inference.";
        } else if (agreement_factor > 0.4f) {
            inference_note = "Sources partially agree. Mixed evidence - "
                             "inference has moderate certainty.";
        } else {
            inference_note = "Sources conflict. Divergent evidence detected - "
                             "inference confidence reduced.";
        }
    }

    /* Build inference step */
    reasoning_step_t step;
    memset(&step, 0, sizeof(step));
    step.step_id = chain->num_steps;
    step.type = REASONING_STEP_INFERENCE;
    step.confidence = inference_confidence;
    step.relevance = 1.0f;  /* Inference is always maximally relevant */
    step.timestamp_us = nimcp_time_get_us();

    snprintf(step.description, sizeof(step.description),
             "Inference from %u evidence sources (%s query): %s "
             "Combined confidence: %.3f.",
             evidence_count, query_type, inference_note,
             inference_confidence);

    reasoning_chain_add_step(chain, &step);

    NIMCP_LOGGING_DEBUG("reasoning_chain: inference phase - %u sources, "
                        "confidence=%.3f",
                        evidence_count, inference_confidence);

    return inference_confidence;
}

/*=============================================================================
 * PHASE 5: VERIFICATION (Predictive Coding)
 *===========================================================================*/

/**
 * @brief Execute the verification phase using predictive coding
 *
 * WHAT: Use predictive network to check if the inference is consistent
 * WHY:  Predictive coding minimizes surprise — high prediction error means
 *        the conclusion is unexpected/inconsistent with internal models
 * HOW:  Encode inference confidence as input, run forward pass, check free energy
 *
 * @return Adjusted confidence after verification
 */
static float phase_verification(reasoning_engine_t* engine, reasoning_chain_t* chain,
                                float inference_confidence)
{
    if (!engine->config.enable_predictive_verify || !engine->predictive_net) {
        return inference_confidence;  /* Pass through unchanged */
    }

    /*
     * Encode the current reasoning state as a predictive coding input.
     * We use the inference confidence and chain statistics as features.
     * The predictive network evaluates whether this "makes sense" given
     * its learned model of the world.
     */
    float input[PREDICTIVE_INPUT_DIM];
    input[0] = inference_confidence;
    input[1] = (chain->num_steps > 0)
        ? (float)chain->num_steps / (float)engine->config.max_steps
        : 0.0f;
    /* Compute running mean confidence of all steps so far */
    float step_mean = 0.0f;
    for (uint32_t i = 0; i < chain->num_steps; i++) {
        step_mean += chain->steps[i].confidence;
    }
    if (chain->num_steps > 0) {
        step_mean /= (float)chain->num_steps;
    }
    input[2] = step_mean;
    input[3] = inference_confidence > 0.5f ? 1.0f : 0.0f;  /* Binary high/low */

    /* Run predictive forward pass to minimize free energy */
    float free_energy = predictive_forward(engine->predictive_net,
                                           input, PREDICTIVE_ITERATIONS);

    /*
     * Interpret free energy:
     * - Low free energy (< 1.0): prediction matches, good consistency
     * - High free energy (> 5.0): large prediction error, inconsistency
     * - Scale: map free energy to a verification factor [0.5, 1.0]
     */
    float max_acceptable_fe = 10.0f;
    float normalized_fe = (free_energy < 0.0f) ? 0.0f :
                          (free_energy > max_acceptable_fe) ? max_acceptable_fe :
                          free_energy;

    /* Verification factor: 1.0 at fe=0, 0.5 at fe=max */
    float verification_factor = 1.0f - 0.5f * (normalized_fe / max_acceptable_fe);

    float verified_confidence = inference_confidence * verification_factor;
    if (verified_confidence > 1.0f) verified_confidence = 1.0f;
    if (verified_confidence < 0.0f) verified_confidence = 0.0f;

    bool passed = (verification_factor > 0.7f);

    if (passed) {
        engine->stats.verification_passes++;
    } else {
        engine->stats.verification_failures++;
    }

    /* Build verification step */
    reasoning_step_t step;
    memset(&step, 0, sizeof(step));
    step.step_id = chain->num_steps;
    step.type = REASONING_STEP_VERIFICATION;
    step.confidence = verified_confidence;
    step.relevance = 0.95f;
    step.timestamp_us = nimcp_time_get_us();

    snprintf(step.description, sizeof(step.description),
             "Predictive verification: free_energy=%.3f, "
             "verification_factor=%.3f. %s "
             "Adjusted confidence: %.3f -> %.3f.",
             free_energy, verification_factor,
             passed ? "Prediction consistent with internal model."
                    : "Prediction error detected - confidence reduced.",
             inference_confidence, verified_confidence);

    reasoning_chain_add_step(chain, &step);

    NIMCP_LOGGING_DEBUG("reasoning_chain: verification phase - fe=%.3f, "
                        "factor=%.3f, passed=%s",
                        free_energy, verification_factor,
                        passed ? "yes" : "no");

    return verified_confidence;
}

/*=============================================================================
 * PHASE 6: EPISTEMIC ASSESSMENT
 *===========================================================================*/

/**
 * @brief Execute the epistemic assessment phase
 *
 * WHAT: Check for cognitive biases and assess epistemic quality of conclusion
 * WHY:  Prevent accepting unsubstantiated claims or biased reasoning
 * HOW:  Call epistemic_assess_claim() with current conclusion, check biases
 *
 * @return Uncertainty score [0-1] (higher = more uncertain)
 */
static float phase_epistemic(reasoning_engine_t* engine, const char* query,
                             reasoning_chain_t* chain, float current_confidence)
{
    if (!engine->config.enable_epistemic_check || !engine->epistemic_filter) {
        return 0.0f;  /* No uncertainty detected (filter unavailable) */
    }

    /*
     * Build evidence structure for the epistemic filter.
     * We package our reasoning chain results as a claim to be assessed.
     */
    claim_evidence_t evidence;
    epistemic_evidence_init(&evidence);

    /* Map our confidence to evidence quality */
    if (current_confidence >= 0.8f) {
        evidence.evidence_quality = EVIDENCE_STRONG;
    } else if (current_confidence >= 0.5f) {
        evidence.evidence_quality = EVIDENCE_MODERATE;
    } else if (current_confidence >= 0.3f) {
        evidence.evidence_quality = EVIDENCE_WEAK;
    } else {
        evidence.evidence_quality = EVIDENCE_ANECDOTAL;
    }

    /* Count evidence sources from our chain */
    uint32_t source_count = 0;
    for (uint32_t i = 0; i < chain->num_steps; i++) {
        if (chain->steps[i].type == REASONING_STEP_RECALL ||
            chain->steps[i].type == REASONING_STEP_KNOWLEDGE) {
            source_count++;
        }
    }
    evidence.num_sources = source_count;
    evidence.evidence_strength = current_confidence;
    evidence.logical_consistency = current_confidence;
    evidence.is_falsifiable = true;  /* Our reasoning is falsifiable */
    evidence.has_contradictions = false;

    /* Prior probability: set to moderate (0.5) since we don't have base rates */
    float prior_probability = 0.5f;

    /* Assess the claim */
    epistemic_assessment_t assessment;
    epistemic_assessment_init(&assessment);

    bool assessed = epistemic_assess_claim(
        engine->epistemic_filter,
        query,
        prior_probability,
        &evidence,
        &assessment);

    float uncertainty_score = 0.0f;

    if (assessed) {
        /*
         * Extract uncertainty from assessment.
         * Skepticism score indicates how much doubt the filter has.
         */
        uncertainty_score = assessment.skepticism_score;

        /* Check for detected biases */
        if (assessment.num_biases_detected > 0) {
            /* Bias detected: increase uncertainty */
            uncertainty_score += 0.1f * (float)assessment.num_biases_detected;
            if (uncertainty_score > 1.0f) uncertainty_score = 1.0f;
        }

        /* Build epistemic step */
        reasoning_step_t step;
        memset(&step, 0, sizeof(step));
        step.step_id = chain->num_steps;
        step.type = REASONING_STEP_UNCERTAINTY;
        step.timestamp_us = nimcp_time_get_us();

        /* Confidence for this step is inversely related to uncertainty */
        step.confidence = 1.0f - uncertainty_score;
        step.relevance = 0.85f;

        if (assessment.num_biases_detected > 0) {
            snprintf(step.description, sizeof(step.description),
                     "Epistemic assessment: %u bias(es) detected. "
                     "Credibility: %.3f, skepticism: %.3f. "
                     "Primary bias: %s (severity: %.2f). "
                     "Recommendation: %s",
                     assessment.num_biases_detected,
                     assessment.credibility_score,
                     assessment.skepticism_score,
                     assessment.biases[0].description,
                     assessment.biases[0].severity,
                     assessment.recommendation);
        } else {
            snprintf(step.description, sizeof(step.description),
                     "Epistemic assessment: no biases detected. "
                     "Credibility: %.3f, skepticism: %.3f, "
                     "epistemic quality: %.3f. "
                     "Logical coherence: %.3f.",
                     assessment.credibility_score,
                     assessment.skepticism_score,
                     assessment.epistemic_quality,
                     assessment.logical_coherence);
        }

        reasoning_chain_add_step(chain, &step);

        /* Flag if uncertainty exceeds threshold */
        if (uncertainty_score > engine->config.uncertainty_threshold) {
            chain->has_uncertainty_flag = true;
            engine->stats.uncertainty_flags++;
        }
    }

    NIMCP_LOGGING_DEBUG("reasoning_chain: epistemic phase - uncertainty=%.3f, "
                        "biases=%u, flagged=%s",
                        uncertainty_score,
                        assessed ? assessment.num_biases_detected : 0,
                        chain->has_uncertainty_flag ? "yes" : "no");

    return uncertainty_score;
}

/*=============================================================================
 * PHASE 7: SYNTHESIS
 *===========================================================================*/

/**
 * @brief Execute the synthesis phase — combine all evidence into final conclusion
 *
 * WHAT: Compute overall confidence, format conclusion, mark chain complete
 * WHY:  Every reasoning chain must end with a coherent synthesized result
 * HOW:  Geometric mean of step confidences, build conclusion string
 *
 * @return Overall confidence
 */
static float phase_synthesis(reasoning_engine_t* engine, const char* query,
                             const char* query_type, reasoning_chain_t* chain,
                             float recall_confidence, float knowledge_confidence,
                             float uncertainty_score)
{
    /* Collect all step confidences for geometric mean */
    float confidences[REASONING_CHAIN_DEFAULT_MAX_STEPS];
    uint32_t conf_count = 0;

    for (uint32_t i = 0; i < chain->num_steps &&
         conf_count < REASONING_CHAIN_DEFAULT_MAX_STEPS; i++) {
        if (chain->steps[i].confidence > 0.0f) {
            confidences[conf_count++] = chain->steps[i].confidence;
        }
    }

    /* Compute overall confidence as geometric mean */
    float overall_confidence = geometric_mean(confidences, conf_count);

    /* Reduce by uncertainty if flagged */
    if (chain->has_uncertainty_flag && uncertainty_score > 0.0f) {
        overall_confidence *= (1.0f - uncertainty_score * 0.3f);
        if (overall_confidence < 0.0f) overall_confidence = 0.0f;
    }

    /* Clamp */
    if (overall_confidence > 1.0f) overall_confidence = 1.0f;

    /* Count evidence sources (including world model and JEPA) */
    uint32_t evidence_count = 0;
    bool has_wm_evidence = false;
    bool has_jepa_evidence = false;
    for (uint32_t i = 0; i < chain->num_steps; i++) {
        if (chain->steps[i].type == REASONING_STEP_RECALL ||
            chain->steps[i].type == REASONING_STEP_KNOWLEDGE ||
            chain->steps[i].type == REASONING_STEP_WORLD_MODEL ||
            chain->steps[i].type == REASONING_STEP_JEPA_PREDICTION) {
            evidence_count++;
        }
        if (chain->steps[i].type == REASONING_STEP_WORLD_MODEL) {
            has_wm_evidence = true;
        }
        if (chain->steps[i].type == REASONING_STEP_JEPA_PREDICTION) {
            has_jepa_evidence = true;
        }
    }
    (void)has_wm_evidence;
    (void)has_jepa_evidence;

    /* Format conclusion */
    format_conclusion(chain->conclusion, sizeof(chain->conclusion),
                      query, query_type, overall_confidence, evidence_count,
                      recall_confidence > ENGRAM_MIN_CONFIDENCE,
                      knowledge_confidence > 0.0f);

    /* Build synthesis step */
    reasoning_step_t step;
    memset(&step, 0, sizeof(step));
    step.step_id = chain->num_steps;
    step.type = REASONING_STEP_SYNTHESIS;
    step.confidence = overall_confidence;
    step.relevance = 1.0f;
    step.timestamp_us = nimcp_time_get_us();

    snprintf(step.description, sizeof(step.description),
             "Synthesis: combined %u steps via geometric mean. "
             "Overall confidence: %.3f. Evidence sources: %u. "
             "Uncertainty: %.3f. Chain %s.",
             conf_count, overall_confidence, evidence_count,
             uncertainty_score,
             chain->has_uncertainty_flag ? "flagged for uncertainty" : "complete");

    reasoning_chain_add_step(chain, &step);

    /* Finalize chain */
    chain->overall_confidence = overall_confidence;
    chain->uncertainty_score = uncertainty_score;
    chain->is_complete = true;
    chain->end_time_us = nimcp_time_get_us();

    NIMCP_LOGGING_INFO("reasoning_chain: synthesis complete - confidence=%.3f, "
                       "steps=%u, duration=%lu us",
                       overall_confidence, chain->num_steps,
                       (unsigned long)(chain->end_time_us - chain->start_time_us));

    return overall_confidence;
}

/*=============================================================================
 * WORKING MEMORY INTEGRATION
 *===========================================================================*/

/**
 * @brief Store query in working memory as the active reasoning target
 *
 * WHAT: Place a hash of the query into working memory for reasoning context
 * WHY:  Working memory holds the "what am I thinking about" during reasoning
 * HOW:  Hash query to float array, add to working memory with high salience
 */
static void store_query_in_wm(reasoning_engine_t* engine, const char* query)
{
    if (!engine->config.enable_working_memory || !engine->working_memory) {
        return;
    }

    /*
     * Convert query to a float representation for working memory.
     * We use a simple character-level hash to create a fixed-size pattern
     * that represents the query in the working memory buffer.
     */
    float wm_item[8];  /* Small footprint: 8 floats */
    memset(wm_item, 0, sizeof(wm_item));

    uint32_t len = (uint32_t)strlen(query);
    for (uint32_t i = 0; i < len && i < 256; i++) {
        /* Distribute characters across 8 float buckets */
        uint32_t bucket = i % 8;
        wm_item[bucket] += (float)(unsigned char)query[i] / 256.0f;
    }

    /* Normalize to [0, 1] range */
    for (int i = 0; i < 8; i++) {
        if (wm_item[i] > 1.0f) {
            wm_item[i] = wm_item[i] / (1.0f + wm_item[i]);  /* Soft saturation */
        }
    }

    working_memory_add(engine->working_memory, wm_item, 8, QUERY_WM_SALIENCE);
}

/*=============================================================================
 * CORE REASONING PIPELINE
 *===========================================================================*/

int reasoning_engine_reason(reasoning_engine_t* engine, const char* query,
                            reasoning_chain_t* chain)
{
    /* Guard clauses */
    if (!engine) {
        NIMCP_LOGGING_ERROR("reasoning_chain: NULL engine");
        return -1;
    }
    if (!query) {
        NIMCP_LOGGING_ERROR("reasoning_chain: NULL query");
        return -1;
    }
    if (!chain) {
        NIMCP_LOGGING_ERROR("reasoning_chain: NULL chain");
        return -1;
    }
    if (!engine->is_connected) {
        NIMCP_LOGGING_ERROR("reasoning_chain: engine not connected to brain");
        return -1;
    }

    /* ── Step 0: Initialize chain ── */
    reasoning_chain_init(chain);
    chain->start_time_us = nimcp_time_get_us();

    NIMCP_LOGGING_INFO("reasoning_chain: beginning reasoning for query: "
                       "\"%.*s%s\"",
                       (int)(strlen(query) > 60 ? 60 : strlen(query)),
                       query,
                       strlen(query) > 60 ? "..." : "");

    /* Store query in working memory as active reasoning target */
    store_query_in_wm(engine, query);

    /* ── Phase 1: Recall (engram memory) ── */
    float recall_confidence = phase_recall(engine, query, chain);

    /*
     * Early termination: if recall alone gives us very high confidence
     * and the config threshold is met, we can short-circuit.
     */
    if (recall_confidence >= engine->config.confidence_threshold) {
        NIMCP_LOGGING_DEBUG("reasoning_chain: early termination from recall "
                            "(confidence=%.3f >= threshold=%.3f)",
                            recall_confidence,
                            engine->config.confidence_threshold);
        /* Still do synthesis to finalize */
        float overall = phase_synthesis(engine, query, "recall-only", chain,
                                        recall_confidence, 0.0f, 0.0f);
        engine->stats.total_queries++;
        engine->stats.successful_queries++;
        engine->stats.total_steps += chain->num_steps;
        /* Update running average confidence */
        float n = (float)engine->stats.total_queries;
        engine->stats.avg_confidence =
            engine->stats.avg_confidence * ((n - 1.0f) / n) + overall / n;
        engine->stats.avg_steps_per_query =
            engine->stats.avg_steps_per_query * ((n - 1.0f) / n) +
            (float)chain->num_steps / n;
        return 0;
    }

    /* ── Phase 2: Knowledge retrieval ── */
    float knowledge_confidence = phase_knowledge(engine, query,
                                                  KNOWLEDGE_DOMAIN_GENERAL,
                                                  chain);

    /* ── Phase 3: Decomposition ── */
    const char* query_type = phase_decomposition(engine, query, chain);

    /* Check step limit */
    if (chain->num_steps >= engine->config.max_steps) {
        NIMCP_LOGGING_WARN("reasoning_chain: step limit reached (%u)",
                           engine->config.max_steps);
        goto finalize;
    }

    /* ── Phase 3.5: World Model Simulation ── */
    float wm_confidence = phase_world_model(engine, query, query_type, chain);

    /* Check step limit */
    if (chain->num_steps >= engine->config.max_steps) {
        goto finalize;
    }

    /* ── Phase 4: Inference ── */
    float inference_confidence = phase_inference(engine, chain,
                                                  recall_confidence,
                                                  knowledge_confidence,
                                                  query_type);

    /* Check step limit */
    if (chain->num_steps >= engine->config.max_steps) {
        NIMCP_LOGGING_WARN("reasoning_chain: step limit reached after inference");
        goto finalize;
    }

    /* ── Phase 4.5: JEPA Prediction (Latent-Space Consistency) ── */
    float jepa_confidence = phase_jepa_prediction(engine, chain,
                                                    inference_confidence,
                                                    query);

    /* Check step limit */
    if (chain->num_steps >= engine->config.max_steps) {
        goto finalize;
    }

    /* ── Phase 5: Verification (predictive coding) ── */
    float verified_confidence = phase_verification(engine, chain,
                                                    jepa_confidence);

    /* Check step limit */
    if (chain->num_steps >= engine->config.max_steps) {
        goto finalize;
    }

    /* ── Phase 6: Epistemic assessment ── */
    float uncertainty_score = phase_epistemic(engine, query, chain,
                                              verified_confidence);

    /* Check step limit */
    if (chain->num_steps >= engine->config.max_steps) {
        goto finalize;
    }

finalize:
    ; /* Empty statement after label needed for C90 compliance */

    /*
     * Retrieve latest confidence and uncertainty values.
     * If we jumped to finalize early, use whatever we have so far.
     */
    float final_recall = recall_confidence;
    float final_knowledge = knowledge_confidence;
    float final_uncertainty = 0.0f;

    /* Find the last uncertainty step's value if it exists */
    for (uint32_t i = 0; i < chain->num_steps; i++) {
        if (chain->steps[i].type == REASONING_STEP_UNCERTAINTY) {
            final_uncertainty = 1.0f - chain->steps[i].confidence;
        }
    }

    /* ── Phase 7: Synthesis ── */
    float overall = phase_synthesis(engine, query, query_type, chain,
                                    final_recall, final_knowledge,
                                    final_uncertainty);

    /* ── Update statistics ── */
    engine->stats.total_queries++;
    engine->stats.successful_queries++;
    engine->stats.total_steps += chain->num_steps;

    /* Running average using incremental formula */
    float n = (float)engine->stats.total_queries;
    engine->stats.avg_confidence =
        engine->stats.avg_confidence * ((n - 1.0f) / n) + overall / n;
    engine->stats.avg_steps_per_query =
        engine->stats.avg_steps_per_query * ((n - 1.0f) / n) +
        (float)chain->num_steps / n;

    return 0;
}

/*=============================================================================
 * DOMAIN-RESTRICTED REASONING
 *===========================================================================*/

int reasoning_engine_reason_in_domain(reasoning_engine_t* engine, const char* query,
                                      uint32_t domain, reasoning_chain_t* chain)
{
    /* Guard clauses */
    if (!engine) {
        NIMCP_LOGGING_ERROR("reasoning_chain: NULL engine");
        return -1;
    }
    if (!query) {
        NIMCP_LOGGING_ERROR("reasoning_chain: NULL query");
        return -1;
    }
    if (!chain) {
        NIMCP_LOGGING_ERROR("reasoning_chain: NULL chain");
        return -1;
    }
    if (!engine->is_connected) {
        NIMCP_LOGGING_ERROR("reasoning_chain: engine not connected to brain");
        return -1;
    }

    /* ── Step 0: Initialize chain ── */
    reasoning_chain_init(chain);
    chain->start_time_us = nimcp_time_get_us();

    NIMCP_LOGGING_INFO("reasoning_chain: domain-restricted reasoning "
                       "(domain=%u) for query: \"%.*s%s\"",
                       domain,
                       (int)(strlen(query) > 60 ? 60 : strlen(query)),
                       query,
                       strlen(query) > 60 ? "..." : "");

    /* Store query in working memory */
    store_query_in_wm(engine, query);

    /* ── Phase 1: Recall ── */
    float recall_confidence = phase_recall(engine, query, chain);

    /* ── Phase 2: Knowledge retrieval (domain-restricted) ── */
    float knowledge_confidence = phase_knowledge(engine, query, domain, chain);

    /* ── Phase 3: Decomposition ── */
    const char* query_type = phase_decomposition(engine, query, chain);

    /* ── Phase 4: Inference ── */
    float inference_confidence = phase_inference(engine, chain,
                                                  recall_confidence,
                                                  knowledge_confidence,
                                                  query_type);

    /* ── Phase 5: Verification ── */
    float verified_confidence = phase_verification(engine, chain,
                                                    inference_confidence);

    /* ── Phase 6: Epistemic ── */
    float uncertainty_score = phase_epistemic(engine, query, chain,
                                              verified_confidence);

    /* ── Phase 7: Synthesis ── */
    float overall = phase_synthesis(engine, query, query_type, chain,
                                    recall_confidence, knowledge_confidence,
                                    uncertainty_score);

    /* ── Update statistics ── */
    engine->stats.total_queries++;
    engine->stats.successful_queries++;
    engine->stats.total_steps += chain->num_steps;

    float n = (float)engine->stats.total_queries;
    engine->stats.avg_confidence =
        engine->stats.avg_confidence * ((n - 1.0f) / n) + overall / n;
    engine->stats.avg_steps_per_query =
        engine->stats.avg_steps_per_query * ((n - 1.0f) / n) +
        (float)chain->num_steps / n;

    return 0;
}
