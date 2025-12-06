/**
 * @file nimcp_language_production_bridge.c
 * @brief Implementation of Language Production Bridge
 *
 * WHAT: Connects Broca's region with Speech Cortex and NLP systems
 * WHY:  Enable end-to-end language production pipeline
 * HOW:  Orchestrate data flow and transformations between systems
 *
 * @version Phase B3: Language Production Integration
 * @date 2025-11-23
 */

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

// Logging integration
#include "utils/logging/nimcp_logging.h"

// Unified memory integration
#include "utils/memory/nimcp_unified_memory.h"

#include "core/brain/regions/broca/nimcp_language_production_bridge.h"
#include "core/brain/regions/broca/nimcp_broca_adapter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils/memory/nimcp_memory_guards.h"  // For nimcp_calloc/nimcp_free

#define LOG_MODULE "BROCA_LANG_PROD"

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Internal bridge state
 */
struct language_production_bridge {
    /* Configuration */
    lpb_config_t config;

    /* Core component */
    broca_adapter_t* broca;

    /* Connected systems (opaque references) */
    speech_cortex_t* speech_cortex;
    nlp_network_t* nlp;
    working_memory_t* wm;

    /* State */
    lpb_status_t status;
    lpb_error_t last_error;

    /* Current production context */
    lpb_semantic_intent_t current_intent;
    float* priming_vector;
    uint32_t priming_dim;
    float priming_strength;

    /* Output buffers */
    lpb_token_t* token_buffer;
    uint32_t token_count;
    uint8_t* phoneme_buffer;
    uint32_t phoneme_count;

    /* Callbacks */
    lpb_motor_output_callback_t motor_callback;
    void* motor_callback_data;
    lpb_event_callback_t event_callback;
    void* event_callback_data;

    /* Statistics */
    lpb_stats_t stats;
};

/*=============================================================================
 * INTERNAL HELPERS
 *===========================================================================*/

/**
 * @brief Set error and return false
 */
static bool lpb_set_error(language_production_bridge_t* bridge, lpb_error_t error) {
    if (bridge) {
        bridge->last_error = error;
        bridge->status = LPB_STATUS_ERROR;
    }
    return false;
}

/**
 * @brief Emit event to callback
 */
static void lpb_emit_event(language_production_bridge_t* bridge,
                           uint32_t event_type,
                           const void* event_data) {
    if (bridge && bridge->event_callback) {
        bridge->event_callback(event_type, event_data, bridge->event_callback_data);
    }
}

/**
 * @brief Clear current production context
 */
static void lpb_clear_context(language_production_bridge_t* bridge) {
    if (!bridge) return;

    /* Clear intent */
    if (bridge->current_intent.semantic_vector) {
        nimcp_free(bridge->current_intent.semantic_vector);
        bridge->current_intent.semantic_vector = NULL;
    }
    memset(&bridge->current_intent, 0, sizeof(lpb_semantic_intent_t));

    /* Clear tokens */
    bridge->token_count = 0;

    /* Clear phonemes */
    bridge->phoneme_count = 0;

    bridge->status = LPB_STATUS_IDLE;
    bridge->last_error = LPB_ERROR_NONE;
}

/**
 * @brief Simple hash for lexicon lookup
 */
static uint32_t lpb_hash_string(const char* str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + (uint32_t)c;
    }
    return hash;
}

/**
 * @brief Map semantic vector to tokens (simplified spreading activation)
 *
 * BIOLOGICAL BASIS: Semantic vectors activate lemmas in mental lexicon
 * Higher activation = better semantic match = more likely selection
 */
static uint32_t lpb_semantic_to_tokens(language_production_bridge_t* bridge,
                                       const float* semantic_vector,
                                       uint32_t dim,
                                       lpb_token_t* tokens,
                                       uint32_t max_tokens) {
    if (!bridge || !semantic_vector || !tokens || max_tokens == 0) {
        return 0;
    }

    /*
     * Simplified lexical selection:
     * - Compute activation from semantic vector magnitude/features
     * - Map to common function words and content words
     * - Apply priming if available
     */

    uint32_t count = 0;
    float activation_sum = 0.0f;

    /* Compute semantic magnitude for activation */
    for (uint32_t i = 0; i < dim && i < 256; i++) {
        activation_sum += semantic_vector[i] * semantic_vector[i];
    }
    float magnitude = sqrtf(activation_sum);

    /* Scale activations */
    float base_activation = magnitude > 0.0f ? magnitude / sqrtf((float)dim) : 0.5f;

    /* Apply priming boost if available */
    float prime_boost = 1.0f;
    if (bridge->priming_vector && bridge->priming_dim > 0) {
        float dot = 0.0f;
        uint32_t min_dim = (dim < bridge->priming_dim) ? dim : bridge->priming_dim;
        for (uint32_t i = 0; i < min_dim; i++) {
            dot += semantic_vector[i] * bridge->priming_vector[i];
        }
        prime_boost = 1.0f + (bridge->priming_strength * fmaxf(0.0f, dot));
    }

    /* Generate tokens based on semantic features */
    /* Feature indices (simplified model):
     * 0-31: entity features
     * 32-63: action features
     * 64-95: property features
     * 96-127: relation features
     */

    /* Check for subject entity (high activation in entity features) */
    float entity_activation = 0.0f;
    for (uint32_t i = 0; i < 32 && i < dim; i++) {
        entity_activation += fabsf(semantic_vector[i]);
    }
    entity_activation /= 32.0f;

    if (entity_activation > 0.1f && count < max_tokens) {
        tokens[count].token_id = lpb_hash_string("entity") % 10000;
        strncpy(tokens[count].token_str, "it", sizeof(tokens[count].token_str) - 1);
        tokens[count].token_str[sizeof(tokens[count].token_str) - 1] = '\0';
        tokens[count].pos = 0; /* Noun/pronoun */
        tokens[count].activation = entity_activation * base_activation * prime_boost;
        tokens[count].frequency = 0.9f; /* High frequency pronoun */
        count++;
    }

    /* Check for action (high activation in action features) */
    float action_activation = 0.0f;
    for (uint32_t i = 32; i < 64 && i < dim; i++) {
        action_activation += fabsf(semantic_vector[i]);
    }
    action_activation /= 32.0f;

    if (action_activation > 0.1f && count < max_tokens) {
        tokens[count].token_id = lpb_hash_string("action") % 10000;
        strncpy(tokens[count].token_str, "does", sizeof(tokens[count].token_str) - 1);
        tokens[count].token_str[sizeof(tokens[count].token_str) - 1] = '\0';
        tokens[count].pos = 1; /* Verb */
        tokens[count].activation = action_activation * base_activation * prime_boost;
        tokens[count].frequency = 0.85f;
        count++;
    }

    /* Check for property (adjective/adverb features) */
    float property_activation = 0.0f;
    for (uint32_t i = 64; i < 96 && i < dim; i++) {
        property_activation += fabsf(semantic_vector[i]);
    }
    property_activation /= 32.0f;

    if (property_activation > 0.15f && count < max_tokens) {
        tokens[count].token_id = lpb_hash_string("property") % 10000;
        strncpy(tokens[count].token_str, "good", sizeof(tokens[count].token_str) - 1);
        tokens[count].token_str[sizeof(tokens[count].token_str) - 1] = '\0';
        tokens[count].pos = 2; /* Adjective */
        tokens[count].activation = property_activation * base_activation * prime_boost;
        tokens[count].frequency = 0.8f;
        count++;
    }

    return count;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

lpb_config_t lpb_default_config(void) {
    lpb_config_t config = {
        .max_tokens = LPB_DEFAULT_MAX_TOKENS,
        .semantic_dim = LPB_DEFAULT_SEMANTIC_DIM,
        .comprehension_threshold = LPB_DEFAULT_COMPREHENSION_THRESHOLD,
        .production_delay_ms = LPB_DEFAULT_PRODUCTION_DELAY_MS,
        .enable_wernicke_connection = true,
        .enable_nlp_connection = true,
        .enable_working_memory = true,
        .enable_semantic_priming = true,
        .enable_repetition = true,
        .enable_paraphrase = false,
        .enable_self_monitoring = true,
        .enable_error_correction = true
    };
    return config;
}

language_production_bridge_t* lpb_create(const lpb_config_t* config,
                                          broca_adapter_t* broca) {
    if (!broca) {
        return NULL;
    }

    language_production_bridge_t* bridge = nimcp_calloc(1, sizeof(language_production_bridge_t));
    if (!bridge) {
        return NULL;
    }

    /* Apply configuration */
    if (config) {
        bridge->config = *config;
    } else {
        bridge->config = lpb_default_config();
    }

    /* Store Broca reference */
    bridge->broca = broca;

    /* Allocate buffers */
    bridge->token_buffer = nimcp_calloc(bridge->config.max_tokens, sizeof(lpb_token_t));
    if (!bridge->token_buffer) {
        nimcp_free(bridge);
        return NULL;
    }

    /* Phoneme buffer: estimate 4 phonemes per token average */
    uint32_t phoneme_capacity = bridge->config.max_tokens * 4;
    bridge->phoneme_buffer = nimcp_calloc(phoneme_capacity, sizeof(uint8_t));
    if (!bridge->phoneme_buffer) {
        nimcp_free(bridge->token_buffer);
        nimcp_free(bridge);
        return NULL;
    }

    /* Initialize state */
    bridge->status = LPB_STATUS_IDLE;
    bridge->last_error = LPB_ERROR_NONE;

    return bridge;
}

void lpb_destroy(language_production_bridge_t* bridge) {
    if (!bridge) return;

    /* Clear context (frees intent vector) */
    lpb_clear_context(bridge);

    /* Free priming vector */
    if (bridge->priming_vector) {
        nimcp_free(bridge->priming_vector);
    }

    /* Free buffers */
    if (bridge->token_buffer) {
        nimcp_free(bridge->token_buffer);
    }
    if (bridge->phoneme_buffer) {
        nimcp_free(bridge->phoneme_buffer);
    }

    /* Note: We don't destroy broca, speech_cortex, nlp, or wm
     * as they are owned externally */

    nimcp_free(bridge);
}

bool lpb_reset(language_production_bridge_t* bridge) {
    if (!bridge) return false;

    lpb_clear_context(bridge);

    /* Clear priming */
    if (bridge->priming_vector) {
        nimcp_free(bridge->priming_vector);
        bridge->priming_vector = NULL;
    }
    bridge->priming_dim = 0;
    bridge->priming_strength = 0.0f;

    /* Reset statistics */
    memset(&bridge->stats, 0, sizeof(lpb_stats_t));

    /* Reset Broca adapter */
    if (bridge->broca) {
        broca_reset(bridge->broca);
    }

    return true;
}

/*=============================================================================
 * SYSTEM CONNECTIONS
 *===========================================================================*/

bool lpb_connect_speech_cortex(language_production_bridge_t* bridge,
                                speech_cortex_t* speech_cortex) {
    if (!bridge) return false;

    if (!bridge->config.enable_wernicke_connection) {
        return lpb_set_error(bridge, LPB_ERROR_NO_SPEECH_CORTEX);
    }

    bridge->speech_cortex = speech_cortex;
    return true;
}

bool lpb_connect_nlp(language_production_bridge_t* bridge,
                     nlp_network_t* nlp) {
    if (!bridge) return false;

    if (!bridge->config.enable_nlp_connection) {
        return lpb_set_error(bridge, LPB_ERROR_NO_NLP);
    }

    bridge->nlp = nlp;
    return true;
}

bool lpb_connect_working_memory(language_production_bridge_t* bridge,
                                 working_memory_t* wm) {
    if (!bridge) return false;

    if (!bridge->config.enable_working_memory) {
        return false;
    }

    bridge->wm = wm;
    return true;
}

/*=============================================================================
 * PRODUCTION PIPELINE
 *===========================================================================*/

bool lpb_produce_from_intent(language_production_bridge_t* bridge,
                              const lpb_semantic_intent_t* intent,
                              lpb_production_result_t* result) {
    if (!bridge || !intent) {
        return lpb_set_error(bridge, LPB_ERROR_INVALID_INPUT);
    }

    if (!bridge->broca) {
        return lpb_set_error(bridge, LPB_ERROR_NO_BROCA);
    }

    bridge->stats.productions_attempted++;

    /* Stage 1: Receive intent */
    bridge->status = LPB_STATUS_RECEIVING_INTENT;
    lpb_emit_event(bridge, LPB_STATUS_RECEIVING_INTENT, intent);

    /* Copy intent */
    if (bridge->current_intent.semantic_vector) {
        nimcp_free(bridge->current_intent.semantic_vector);
    }
    bridge->current_intent = *intent;
    if (intent->semantic_vector && intent->semantic_dim > 0) {
        bridge->current_intent.semantic_vector = nimcp_malloc(intent->semantic_dim * sizeof(float));
        if (bridge->current_intent.semantic_vector) {
            memcpy(bridge->current_intent.semantic_vector,
                   intent->semantic_vector,
                   intent->semantic_dim * sizeof(float));
        }
    }

    /* Stage 2: Lexical selection */
    bridge->status = LPB_STATUS_LEXICAL_SELECTION;
    lpb_emit_event(bridge, LPB_STATUS_LEXICAL_SELECTION, NULL);

    bridge->token_count = lpb_semantic_to_tokens(
        bridge,
        intent->semantic_vector,
        intent->semantic_dim,
        bridge->token_buffer,
        bridge->config.max_tokens
    );

    if (bridge->token_count == 0) {
        return lpb_set_error(bridge, LPB_ERROR_LEXICAL_FAILURE);
    }

    bridge->stats.lexical_selections++;

    /* Stage 3: Syntactic encoding via Broca */
    bridge->status = LPB_STATUS_SYNTACTIC_ENCODING;
    lpb_emit_event(bridge, LPB_STATUS_SYNTACTIC_ENCODING, NULL);

    if (!broca_begin_utterance(bridge->broca)) {
        return lpb_set_error(bridge, LPB_ERROR_PRODUCTION_FAILURE);
    }

    /* Add tokens to Broca */
    for (uint32_t i = 0; i < bridge->token_count; i++) {
        broca_input_word_t word = {
            .word_id = bridge->token_buffer[i].token_id,
            .pos = bridge->token_buffer[i].pos,
            .number = 0, /* Singular */
            .person = 3, /* Third person */
            .tense = 0   /* Present */
        };
        strncpy(word.word, bridge->token_buffer[i].token_str, sizeof(word.word) - 1);
        word.word[sizeof(word.word) - 1] = '\0';

        if (!broca_add_word(bridge->broca, &word)) {
            /* Continue with partial utterance */
        }
    }

    /* Stage 4: Phonological encoding */
    bridge->status = LPB_STATUS_PHONOLOGICAL_ENCODING;
    lpb_emit_event(bridge, LPB_STATUS_PHONOLOGICAL_ENCODING, NULL);

    /* Stage 5: Articulation planning */
    bridge->status = LPB_STATUS_ARTICULATION;
    lpb_emit_event(bridge, LPB_STATUS_ARTICULATION, NULL);

    /* Process through Broca's pipeline */
    broca_utterance_result_t broca_result;
    if (!broca_process_utterance(bridge->broca, &broca_result)) {
        return lpb_set_error(bridge, LPB_ERROR_PRODUCTION_FAILURE);
    }

    /* Stage 6: Self-monitoring */
    if (bridge->config.enable_self_monitoring) {
        bridge->status = LPB_STATUS_SELF_MONITORING;
        lpb_emit_event(bridge, LPB_STATUS_SELF_MONITORING, NULL);

        float match_score;
        if (lpb_check_production(bridge, &match_score)) {
            if (result) {
                result->self_monitoring_passed = true;
                result->semantic_match = match_score;
            }
        } else if (bridge->config.enable_error_correction) {
            /* Attempt correction - simplified: just note the issue */
            bridge->stats.self_corrections++;
            if (result) {
                result->self_monitoring_passed = false;
            }
        }
    }

    /* Extract motor commands */
    if (bridge->motor_callback) {
        broca_output_command_t cmd;
        while (broca_get_next_command(bridge->broca, &cmd)) {
            bridge->motor_callback(&cmd, bridge->motor_callback_data);
        }
    }

    /* Fill result */
    if (result) {
        result->tokens = bridge->token_buffer;
        result->token_count = bridge->token_count;
        result->phonemes = bridge->phoneme_buffer;
        result->phoneme_count = broca_result.phoneme_count;
        result->motor_command_count = broca_result.command_count;
        result->estimated_duration_ms = broca_result.total_duration_ms;
        /* Compute fluency score from syntax/agreement validation */
        result->fluency_score = (broca_result.syntax_valid ? 0.5f : 0.0f) +
                                (broca_result.agreement_valid ? 0.5f : 0.0f);
        if (!result->self_monitoring_passed) {
            result->semantic_match = 0.5f; /* Default if monitoring disabled */
        }
    }

    bridge->status = LPB_STATUS_READY;
    bridge->stats.productions_successful++;
    bridge->stats.avg_production_latency_ms =
        (bridge->stats.avg_production_latency_ms * (bridge->stats.productions_successful - 1) +
         bridge->config.production_delay_ms) / bridge->stats.productions_successful;

    if (result) {
        bridge->stats.avg_fluency_score =
            (bridge->stats.avg_fluency_score * (bridge->stats.productions_successful - 1) +
             result->fluency_score) / bridge->stats.productions_successful;
    }

    return true;
}

bool lpb_produce_from_tokens(language_production_bridge_t* bridge,
                              const lpb_token_t* tokens,
                              uint32_t num_tokens,
                              lpb_production_result_t* result) {
    if (!bridge || !tokens || num_tokens == 0) {
        return lpb_set_error(bridge, LPB_ERROR_INVALID_INPUT);
    }

    if (!bridge->broca) {
        return lpb_set_error(bridge, LPB_ERROR_NO_BROCA);
    }

    bridge->stats.productions_attempted++;

    /* Copy tokens to internal buffer */
    uint32_t copy_count = (num_tokens < bridge->config.max_tokens) ?
                          num_tokens : bridge->config.max_tokens;
    memcpy(bridge->token_buffer, tokens, copy_count * sizeof(lpb_token_t));
    bridge->token_count = copy_count;

    /* Skip lexical selection, go directly to syntax */
    bridge->status = LPB_STATUS_SYNTACTIC_ENCODING;

    if (!broca_begin_utterance(bridge->broca)) {
        return lpb_set_error(bridge, LPB_ERROR_PRODUCTION_FAILURE);
    }

    for (uint32_t i = 0; i < bridge->token_count; i++) {
        broca_input_word_t word = {
            .word_id = bridge->token_buffer[i].token_id,
            .pos = bridge->token_buffer[i].pos,
            .number = 0,
            .person = 3,
            .tense = 0
        };
        strncpy(word.word, bridge->token_buffer[i].token_str, sizeof(word.word) - 1);
        word.word[sizeof(word.word) - 1] = '\0';

        broca_add_word(bridge->broca, &word);
    }

    /* Process */
    bridge->status = LPB_STATUS_PHONOLOGICAL_ENCODING;
    bridge->status = LPB_STATUS_ARTICULATION;

    broca_utterance_result_t broca_result;
    if (!broca_process_utterance(bridge->broca, &broca_result)) {
        return lpb_set_error(bridge, LPB_ERROR_PRODUCTION_FAILURE);
    }

    /* Motor output */
    if (bridge->motor_callback) {
        broca_output_command_t cmd;
        while (broca_get_next_command(bridge->broca, &cmd)) {
            bridge->motor_callback(&cmd, bridge->motor_callback_data);
        }
    }

    if (result) {
        result->tokens = bridge->token_buffer;
        result->token_count = bridge->token_count;
        result->phonemes = bridge->phoneme_buffer;
        result->phoneme_count = broca_result.phoneme_count;
        result->motor_command_count = broca_result.command_count;
        result->estimated_duration_ms = broca_result.total_duration_ms;
        /* Compute fluency score from syntax/agreement validation */
        result->fluency_score = (broca_result.syntax_valid ? 0.5f : 0.0f) +
                                (broca_result.agreement_valid ? 0.5f : 0.0f);
        result->semantic_match = 1.0f; /* Direct tokens = exact match */
        result->self_monitoring_passed = true;
    }

    bridge->status = LPB_STATUS_READY;
    bridge->stats.productions_successful++;

    return true;
}

bool lpb_repeat_last_heard(language_production_bridge_t* bridge,
                            lpb_production_result_t* result) {
    if (!bridge) {
        return lpb_set_error(bridge, LPB_ERROR_INVALID_INPUT);
    }

    if (!bridge->config.enable_repetition) {
        return lpb_set_error(bridge, LPB_ERROR_INVALID_INPUT);
    }

    if (!bridge->speech_cortex) {
        return lpb_set_error(bridge, LPB_ERROR_NO_SPEECH_CORTEX);
    }

    bridge->stats.wernicke_inputs++;

    /*
     * In a full implementation, we would:
     * 1. Query speech_cortex for last comprehended utterance
     * 2. Get phoneme sequence directly (echoic pathway)
     * 3. Feed to Broca for motor planning only
     *
     * For now, return error as speech cortex interface is opaque
     */
    return lpb_set_error(bridge, LPB_ERROR_NO_SPEECH_CORTEX);
}

bool lpb_generate_response(language_production_bridge_t* bridge,
                            lpb_production_result_t* result) {
    if (!bridge) {
        return lpb_set_error(bridge, LPB_ERROR_INVALID_INPUT);
    }

    if (!bridge->nlp) {
        return lpb_set_error(bridge, LPB_ERROR_NO_NLP);
    }

    bridge->stats.nlp_inputs++;

    /*
     * In a full implementation, we would:
     * 1. Query NLP for response to last input
     * 2. Get semantic intent from NLP
     * 3. Call lpb_produce_from_intent
     *
     * For now, return error as NLP interface is opaque
     */
    return lpb_set_error(bridge, LPB_ERROR_NO_NLP);
}

/*=============================================================================
 * LEXICAL ACCESS
 *===========================================================================*/

bool lpb_select_lexical_items(language_production_bridge_t* bridge,
                               const float* semantic_vector,
                               uint32_t dim,
                               lpb_token_t* tokens,
                               uint32_t max_tokens,
                               uint32_t* num_selected) {
    if (!bridge || !semantic_vector || !tokens || !num_selected) {
        return lpb_set_error(bridge, LPB_ERROR_INVALID_INPUT);
    }

    *num_selected = lpb_semantic_to_tokens(bridge, semantic_vector, dim, tokens, max_tokens);

    if (*num_selected == 0) {
        return lpb_set_error(bridge, LPB_ERROR_LEXICAL_FAILURE);
    }

    bridge->stats.lexical_selections++;
    return true;
}

bool lpb_prime_lexical_access(language_production_bridge_t* bridge,
                               const float* context_vector,
                               uint32_t dim,
                               float prime_strength) {
    if (!bridge || !context_vector || dim == 0) {
        return lpb_set_error(bridge, LPB_ERROR_INVALID_INPUT);
    }

    if (!bridge->config.enable_semantic_priming) {
        return true; /* Silently succeed if priming disabled */
    }

    /* Allocate/reallocate priming vector */
    if (bridge->priming_vector == NULL || bridge->priming_dim != dim) {
        float* new_vec = nimcp_realloc(bridge->priming_vector, dim * sizeof(float));
        if (!new_vec) {
            return lpb_set_error(bridge, LPB_ERROR_INTERNAL);
        }
        bridge->priming_vector = new_vec;
        bridge->priming_dim = dim;
    }

    /* Copy and normalize */
    float norm = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        norm += context_vector[i] * context_vector[i];
    }
    norm = sqrtf(norm);

    if (norm > 0.0f) {
        for (uint32_t i = 0; i < dim; i++) {
            bridge->priming_vector[i] = context_vector[i] / norm;
        }
    } else {
        memcpy(bridge->priming_vector, context_vector, dim * sizeof(float));
    }

    bridge->priming_strength = fmaxf(0.0f, fminf(1.0f, prime_strength));

    return true;
}

/*=============================================================================
 * SELF-MONITORING
 *===========================================================================*/

bool lpb_set_self_monitoring(language_production_bridge_t* bridge, bool enable) {
    if (!bridge) return false;

    bridge->config.enable_self_monitoring = enable;
    return true;
}

bool lpb_check_production(language_production_bridge_t* bridge, float* match_score) {
    if (!bridge) return false;

    if (!bridge->config.enable_self_monitoring) {
        if (match_score) *match_score = 1.0f;
        return true;
    }

    /*
     * Self-monitoring: Compare planned output semantics to input intent
     *
     * BIOLOGICAL BASIS: Perceptual loop monitors inner speech before
     * articulation, comparing planned phonemes/words to intended meaning
     */

    if (!bridge->current_intent.semantic_vector || bridge->token_count == 0) {
        if (match_score) *match_score = 0.0f;
        return false;
    }

    /* Simplified check: compute expected semantic match from token activations */
    float total_activation = 0.0f;
    float max_activation = 0.0f;
    for (uint32_t i = 0; i < bridge->token_count; i++) {
        total_activation += bridge->token_buffer[i].activation;
        if (bridge->token_buffer[i].activation > max_activation) {
            max_activation = bridge->token_buffer[i].activation;
        }
    }

    /* Average activation as proxy for semantic match */
    float avg_activation = (bridge->token_count > 0) ?
                           total_activation / bridge->token_count : 0.0f;

    /* Scale to [0,1] range */
    float score = fminf(1.0f, avg_activation * 2.0f);

    if (match_score) {
        *match_score = score;
    }

    /* Pass if score exceeds comprehension threshold */
    return (score >= bridge->config.comprehension_threshold);
}

/*=============================================================================
 * CALLBACKS
 *===========================================================================*/

bool lpb_set_motor_callback(language_production_bridge_t* bridge,
                             lpb_motor_output_callback_t callback,
                             void* user_data) {
    if (!bridge) return false;

    bridge->motor_callback = callback;
    bridge->motor_callback_data = user_data;
    return true;
}

bool lpb_set_event_callback(language_production_bridge_t* bridge,
                             lpb_event_callback_t callback,
                             void* user_data) {
    if (!bridge) return false;

    bridge->event_callback = callback;
    bridge->event_callback_data = user_data;
    return true;
}

/*=============================================================================
 * STATUS AND DIAGNOSTICS
 *===========================================================================*/

lpb_status_t lpb_get_status(const language_production_bridge_t* bridge) {
    if (!bridge) return LPB_STATUS_ERROR;
    return bridge->status;
}

lpb_error_t lpb_get_last_error(const language_production_bridge_t* bridge) {
    if (!bridge) return LPB_ERROR_INTERNAL;
    return bridge->last_error;
}

const char* lpb_error_string(lpb_error_t error) {
    switch (error) {
        case LPB_ERROR_NONE:              return "No error";
        case LPB_ERROR_INVALID_INPUT:     return "Invalid input";
        case LPB_ERROR_NO_BROCA:          return "Broca's region not connected";
        case LPB_ERROR_NO_SPEECH_CORTEX:  return "Speech cortex not connected";
        case LPB_ERROR_NO_NLP:            return "NLP network not connected";
        case LPB_ERROR_LEXICAL_FAILURE:   return "Lexical selection failed";
        case LPB_ERROR_PRODUCTION_FAILURE:return "Production pipeline failed";
        case LPB_ERROR_MONITORING_FAILURE:return "Self-monitoring failed";
        case LPB_ERROR_INTERNAL:          return "Internal error";
        default:                          return "Unknown error";
    }
}

const char* lpb_status_string(lpb_status_t status) {
    switch (status) {
        case LPB_STATUS_IDLE:                return "Idle";
        case LPB_STATUS_RECEIVING_INTENT:    return "Receiving intent";
        case LPB_STATUS_LEXICAL_SELECTION:   return "Lexical selection";
        case LPB_STATUS_SYNTACTIC_ENCODING:  return "Syntactic encoding";
        case LPB_STATUS_PHONOLOGICAL_ENCODING:return "Phonological encoding";
        case LPB_STATUS_ARTICULATION:        return "Articulation planning";
        case LPB_STATUS_SELF_MONITORING:     return "Self-monitoring";
        case LPB_STATUS_READY:               return "Ready";
        case LPB_STATUS_ERROR:               return "Error";
        default:                             return "Unknown";
    }
}

bool lpb_get_stats(const language_production_bridge_t* bridge, lpb_stats_t* stats) {
    if (!bridge || !stats) return false;

    *stats = bridge->stats;
    return true;
}

bool lpb_get_config(const language_production_bridge_t* bridge, lpb_config_t* config) {
    if (!bridge || !config) return false;

    *config = bridge->config;
    return true;
}

/*=============================================================================
 * DIRECT ACCESS
 *===========================================================================*/

broca_adapter_t* lpb_get_broca_adapter(language_production_bridge_t* bridge) {
    if (!bridge) return NULL;
    return bridge->broca;
}
