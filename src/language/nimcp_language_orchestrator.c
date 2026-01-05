//=============================================================================
// nimcp_language_orchestrator.c - Language Layer Central Orchestrator
//=============================================================================
/**
 * @file nimcp_language_orchestrator.c
 * @brief Implementation of the Language Layer central orchestrator
 *
 * Implements state machine, lifecycle management, subsystem coordination,
 * and bridge integration for unified language processing.
 *
 * @version 1.0.0 - Phase L5: Orchestrator Implementation
 * @author NIMCP Development Team
 * @date 2026-01-05
 */

#include "language/nimcp_language_orchestrator.h"
#include "language/nimcp_language_bio_async.h"
#include "language/bridges/nimcp_language_perception_bridge.h"
#include "language/bridges/nimcp_language_cognitive_bridge.h"
#include "language/bridges/nimcp_language_training_bridge.h"
#include "language/bridges/nimcp_language_omni_bridge.h"
#include "language/bridges/nimcp_language_immune_bridge.h"
#include "language/bridges/nimcp_language_gpu_bridge.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

//=============================================================================
// Internal Constants
//=============================================================================

#define ORCHESTRATOR_MAX_CALLBACKS          8
#define ORCHESTRATOR_MAX_PENDING_PHONEMES   256
#define ORCHESTRATOR_MAX_PENDING_WORDS      64
#define ORCHESTRATOR_STATE_TIMEOUT_MS       5000

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Callback registration entry
 */
typedef struct {
    language_event_callback_t callback;
    void* user_data;
    bool active;
} callback_entry_t;

/**
 * @brief Pending input buffer
 */
typedef struct {
    language_phoneme_t* phonemes;
    uint32_t phoneme_count;
    uint32_t phoneme_capacity;

    language_word_t* words;
    uint32_t word_count;
    uint32_t word_capacity;

    char* text_buffer;
    uint32_t text_length;
    uint32_t text_capacity;
} input_buffer_t;

/**
 * @brief Internal orchestrator structure
 */
struct language_orchestrator {
    /* Configuration */
    language_orchestrator_config_t config;
    bool initialized;
    bool running;

    /* State machine */
    language_state_t state;
    language_state_t prev_state;
    language_mode_t mode;
    uint64_t state_entry_time_ms;
    uint64_t last_update_ms;

    /* Subsystem connections */
    wernicke_adapter_t* wernicke;
    broca_adapter_t* broca;
    nlp_network_t nlp_network;
    speech_cortex_t* speech_cortex;
    multimodal_nlp_bridge_t* multimodal;

    /* Bridge connections */
    language_perception_bridge_t* perception_bridge;
    language_cognitive_bridge_t* cognitive_bridge;
    language_training_bridge_t* training_bridge;
    language_omni_bridge_t* omni_bridge;
    language_immune_bridge_t* immune_bridge;
    language_gpu_bridge_t* gpu_bridge;

    /* Input buffers */
    input_buffer_t input;

    /* Current results */
    language_comprehension_result_t current_comprehension;
    bool comprehension_valid;

    language_production_plan_t current_production;
    bool production_valid;

    /* Event callbacks */
    callback_entry_t callbacks[ORCHESTRATOR_MAX_CALLBACKS];
    uint32_t num_callbacks;

    /* Statistics */
    language_orchestrator_stats_t stats;

    /* Bio-async */
    bool bio_async_registered;
};

//=============================================================================
// Forward Declarations
//=============================================================================

static int orchestrator_init_buffers(language_orchestrator_t* orch);
static void orchestrator_free_buffers(language_orchestrator_t* orch);
static int orchestrator_transition_state(language_orchestrator_t* orch,
                                         language_state_t new_state,
                                         uint64_t current_time_ms);
static int orchestrator_process_state(language_orchestrator_t* orch,
                                      uint64_t current_time_ms);
static void orchestrator_emit_event(language_orchestrator_t* orch,
                                    const language_event_t* event);
static int orchestrator_update_bridges(language_orchestrator_t* orch,
                                       uint64_t current_time_ms);

//=============================================================================
// Configuration API
//=============================================================================

void language_orchestrator_default_config(language_orchestrator_config_t* config)
{
    if (!config) return;

    memset(config, 0, sizeof(*config));

    /* Processing mode */
    config->default_mode = LANGUAGE_MODE_DIALOGUE;

    /* Subsystem enables */
    config->enable_wernicke = true;
    config->enable_broca = true;
    config->enable_nlp_core = true;
    config->enable_spike_nlp = false;
    config->enable_multimodal = true;

    /* Bridge enables */
    config->enable_perception_bridge = true;
    config->enable_cognitive_bridge = true;
    config->enable_training_bridge = true;
    config->enable_omni_bridge = true;
    config->enable_immune_bridge = true;
    config->enable_gpu_bridge = false;  /* Opt-in */

    /* Processing settings */
    config->max_utterance_words = LANGUAGE_MAX_WORDS;
    config->phoneme_buffer_size = LANGUAGE_MAX_PHONEMES;
    config->semantic_dim = LANGUAGE_SEMANTIC_DIM;
    config->comprehension_threshold = 0.5f;
    config->production_threshold = 0.5f;

    /* Timing settings */
    config->update_interval_ms = 20;
    config->comprehension_timeout_ms = 5000;
    config->production_timeout_ms = 3000;

    /* Bio-async settings */
    config->enable_bio_async = true;
    config->message_inbox_capacity = 64;

    /* Logging */
    config->enable_logging = false;
    config->enable_stats = true;
}

//=============================================================================
// Lifecycle API
//=============================================================================

language_orchestrator_t* language_orchestrator_create(
    const language_orchestrator_config_t* config)
{
    language_orchestrator_t* orch = calloc(1, sizeof(language_orchestrator_t));
    if (!orch) return NULL;

    /* Apply configuration */
    if (config) {
        orch->config = *config;
    } else {
        language_orchestrator_default_config(&orch->config);
    }

    /* Initialize state machine */
    orch->state = LANGUAGE_STATE_IDLE;
    orch->prev_state = LANGUAGE_STATE_IDLE;
    orch->mode = orch->config.default_mode;
    orch->state_entry_time_ms = 0;
    orch->last_update_ms = 0;

    /* Initialize buffers */
    if (orchestrator_init_buffers(orch) != 0) {
        free(orch);
        return NULL;
    }

    /* Clear results */
    memset(&orch->current_comprehension, 0, sizeof(orch->current_comprehension));
    orch->comprehension_valid = false;

    memset(&orch->current_production, 0, sizeof(orch->current_production));
    orch->production_valid = false;

    /* Clear callbacks */
    memset(orch->callbacks, 0, sizeof(orch->callbacks));
    orch->num_callbacks = 0;

    /* Initialize statistics */
    memset(&orch->stats, 0, sizeof(orch->stats));
    orch->stats.current_state = LANGUAGE_STATE_IDLE;
    orch->stats.current_mode = orch->mode;

    orch->initialized = true;
    orch->running = false;
    orch->bio_async_registered = false;

    return orch;
}

void language_orchestrator_destroy(language_orchestrator_t* orchestrator)
{
    if (!orchestrator) return;

    /* Stop if running */
    if (orchestrator->running) {
        language_orchestrator_stop(orchestrator);
    }

    /* Free buffers */
    orchestrator_free_buffers(orchestrator);

    /* Free result resources */
    language_comprehension_result_free(&orchestrator->current_comprehension);
    language_production_plan_free(&orchestrator->current_production);

    free(orchestrator);
}

int language_orchestrator_start(language_orchestrator_t* orchestrator)
{
    if (!orchestrator || !orchestrator->initialized) {
        return -1;
    }

    if (orchestrator->running) {
        return 0;  /* Already running */
    }

    /* Reset state */
    orchestrator->state = LANGUAGE_STATE_IDLE;
    orchestrator->state_entry_time_ms = 0;

    orchestrator->running = true;

    /* Start bridges if connected */
    if (orchestrator->perception_bridge) {
        language_perception_bridge_start(orchestrator->perception_bridge);
    }
    if (orchestrator->cognitive_bridge) {
        language_cognitive_bridge_start(orchestrator->cognitive_bridge);
    }
    if (orchestrator->training_bridge) {
        language_training_bridge_start(orchestrator->training_bridge);
    }
    if (orchestrator->omni_bridge) {
        language_omni_bridge_start(orchestrator->omni_bridge);
    }
    if (orchestrator->immune_bridge) {
        language_immune_bridge_start(orchestrator->immune_bridge);
    }
    if (orchestrator->gpu_bridge) {
        language_gpu_bridge_start(orchestrator->gpu_bridge);
    }

    return 0;
}

int language_orchestrator_stop(language_orchestrator_t* orchestrator)
{
    if (!orchestrator) {
        return -1;
    }

    if (!orchestrator->running) {
        return 0;  /* Already stopped */
    }

    /* Stop bridges */
    if (orchestrator->perception_bridge) {
        language_perception_bridge_stop(orchestrator->perception_bridge);
    }
    if (orchestrator->cognitive_bridge) {
        language_cognitive_bridge_stop(orchestrator->cognitive_bridge);
    }
    if (orchestrator->training_bridge) {
        language_training_bridge_stop(orchestrator->training_bridge);
    }
    if (orchestrator->omni_bridge) {
        language_omni_bridge_stop(orchestrator->omni_bridge);
    }
    if (orchestrator->immune_bridge) {
        language_immune_bridge_stop(orchestrator->immune_bridge);
    }
    if (orchestrator->gpu_bridge) {
        language_gpu_bridge_stop(orchestrator->gpu_bridge);
    }

    orchestrator->running = false;
    orchestrator->state = LANGUAGE_STATE_IDLE;

    return 0;
}

bool language_orchestrator_is_running(const language_orchestrator_t* orchestrator)
{
    return orchestrator && orchestrator->running;
}

//=============================================================================
// Subsystem Connection API
//=============================================================================

int language_orchestrator_connect_wernicke(
    language_orchestrator_t* orchestrator,
    wernicke_adapter_t* wernicke)
{
    if (!orchestrator) return -1;

    orchestrator->wernicke = wernicke;
    orchestrator->stats.wernicke_connected = (wernicke != NULL);

    return 0;
}

int language_orchestrator_connect_broca(
    language_orchestrator_t* orchestrator,
    broca_adapter_t* broca)
{
    if (!orchestrator) return -1;

    orchestrator->broca = broca;
    orchestrator->stats.broca_connected = (broca != NULL);

    return 0;
}

int language_orchestrator_connect_nlp(
    language_orchestrator_t* orchestrator,
    nlp_network_t nlp)
{
    if (!orchestrator) return -1;

    orchestrator->nlp_network = nlp;
    orchestrator->stats.nlp_connected = (nlp != NULL);

    return 0;
}

int language_orchestrator_connect_speech_cortex(
    language_orchestrator_t* orchestrator,
    speech_cortex_t* speech)
{
    if (!orchestrator) return -1;

    orchestrator->speech_cortex = speech;

    return 0;
}

int language_orchestrator_connect_multimodal(
    language_orchestrator_t* orchestrator,
    multimodal_nlp_bridge_t* multimodal)
{
    if (!orchestrator) return -1;

    orchestrator->multimodal = multimodal;

    return 0;
}

//=============================================================================
// Bridge Connection API
//=============================================================================

int language_orchestrator_connect_perception_bridge(
    language_orchestrator_t* orchestrator,
    language_perception_bridge_t* bridge)
{
    if (!orchestrator) return -1;

    orchestrator->perception_bridge = bridge;
    orchestrator->stats.perception_bridge_connected = (bridge != NULL);

    /* Connect bridge back to orchestrator */
    if (bridge) {
        language_perception_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}

int language_orchestrator_connect_cognitive_bridge(
    language_orchestrator_t* orchestrator,
    language_cognitive_bridge_t* bridge)
{
    if (!orchestrator) return -1;

    orchestrator->cognitive_bridge = bridge;
    orchestrator->stats.cognitive_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_cognitive_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}

int language_orchestrator_connect_training_bridge(
    language_orchestrator_t* orchestrator,
    language_training_bridge_t* bridge)
{
    if (!orchestrator) return -1;

    orchestrator->training_bridge = bridge;
    orchestrator->stats.training_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_training_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}

int language_orchestrator_connect_omni_bridge(
    language_orchestrator_t* orchestrator,
    language_omni_bridge_t* bridge)
{
    if (!orchestrator) return -1;

    orchestrator->omni_bridge = bridge;
    orchestrator->stats.omni_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_omni_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}

int language_orchestrator_connect_immune_bridge(
    language_orchestrator_t* orchestrator,
    language_immune_bridge_t* bridge)
{
    if (!orchestrator) return -1;

    orchestrator->immune_bridge = bridge;
    orchestrator->stats.immune_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_immune_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}

int language_orchestrator_connect_gpu_bridge(
    language_orchestrator_t* orchestrator,
    language_gpu_bridge_t* bridge)
{
    if (!orchestrator) return -1;

    orchestrator->gpu_bridge = bridge;
    orchestrator->stats.gpu_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_gpu_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}

//=============================================================================
// Processing API
//=============================================================================

int language_orchestrator_process_input(
    language_orchestrator_t* orchestrator,
    const void* input,
    uint32_t input_size,
    language_input_type_t input_type)
{
    if (!orchestrator || !orchestrator->running || !input) {
        return -1;
    }

    switch (input_type) {
        case LANGUAGE_INPUT_PHONEMES:
            return language_orchestrator_process_phonemes(
                orchestrator,
                (const language_phoneme_t*)input,
                input_size
            );

        case LANGUAGE_INPUT_TEXT:
            return language_orchestrator_process_text(
                orchestrator,
                (const char*)input
            );

        case LANGUAGE_INPUT_AUDIO:
            /* Route to perception bridge for phoneme extraction */
            if (orchestrator->perception_bridge) {
                /* Audio processing would go through speech cortex */
                /* For now, just transition to listening state */
                orchestrator_transition_state(orchestrator,
                    LANGUAGE_STATE_LISTENING,
                    orchestrator->last_update_ms);
            }
            break;

        case LANGUAGE_INPUT_SEMANTIC:
            /* Direct semantic input for production */
            if (orchestrator->mode == LANGUAGE_MODE_PRODUCTION ||
                orchestrator->mode == LANGUAGE_MODE_DIALOGUE) {
                orchestrator_transition_state(orchestrator,
                    LANGUAGE_STATE_GENERATING,
                    orchestrator->last_update_ms);
            }
            break;

        default:
            return -1;
    }

    return 0;
}

int language_orchestrator_process_phonemes(
    language_orchestrator_t* orchestrator,
    const language_phoneme_t* phonemes,
    uint32_t count)
{
    if (!orchestrator || !orchestrator->running || !phonemes || count == 0) {
        return -1;
    }

    /* Buffer phonemes */
    uint32_t space = orchestrator->input.phoneme_capacity -
                     orchestrator->input.phoneme_count;
    uint32_t to_copy = (count < space) ? count : space;

    if (to_copy > 0) {
        memcpy(&orchestrator->input.phonemes[orchestrator->input.phoneme_count],
               phonemes,
               to_copy * sizeof(language_phoneme_t));
        orchestrator->input.phoneme_count += to_copy;
        orchestrator->stats.phonemes_processed += to_copy;
    }

    /* Transition to comprehending if not already */
    if (orchestrator->state == LANGUAGE_STATE_IDLE ||
        orchestrator->state == LANGUAGE_STATE_LISTENING) {
        orchestrator_transition_state(orchestrator,
            LANGUAGE_STATE_COMPREHENDING,
            orchestrator->last_update_ms);
    }

    return (int)to_copy;
}

int language_orchestrator_process_text(
    language_orchestrator_t* orchestrator,
    const char* text)
{
    if (!orchestrator || !orchestrator->running || !text) {
        return -1;
    }

    uint32_t len = (uint32_t)strlen(text);
    uint32_t space = orchestrator->input.text_capacity -
                     orchestrator->input.text_length;
    uint32_t to_copy = (len < space) ? len : space;

    if (to_copy > 0) {
        memcpy(&orchestrator->input.text_buffer[orchestrator->input.text_length],
               text,
               to_copy);
        orchestrator->input.text_length += to_copy;
        orchestrator->input.text_buffer[orchestrator->input.text_length] = '\0';
    }

    /* Transition to comprehending */
    if (orchestrator->state == LANGUAGE_STATE_IDLE) {
        orchestrator_transition_state(orchestrator,
            LANGUAGE_STATE_COMPREHENDING,
            orchestrator->last_update_ms);
    }

    return 0;
}

int language_orchestrator_generate_output(
    language_orchestrator_t* orchestrator,
    const float* semantic_input,
    uint32_t semantic_dim,
    void* output,
    uint32_t max_output,
    uint32_t* output_size,
    language_output_type_t output_type)
{
    if (!orchestrator || !orchestrator->running || !semantic_input) {
        return -1;
    }

    /* Store semantic input for production */
    orchestrator->current_production.semantic_input = (float*)semantic_input;
    orchestrator->current_production.semantic_dim = semantic_dim;

    /* Transition to generating state */
    orchestrator_transition_state(orchestrator,
        LANGUAGE_STATE_GENERATING,
        orchestrator->last_update_ms);

    /* Production would be handled by Broca adapter */
    /* For now, mark as pending */

    if (output_size) {
        *output_size = 0;  /* Will be filled when production completes */
    }

    return 0;
}

int language_orchestrator_get_comprehension(
    const language_orchestrator_t* orchestrator,
    language_comprehension_result_t* result)
{
    if (!orchestrator || !result) {
        return -1;
    }

    if (!orchestrator->comprehension_valid) {
        return -1;
    }

    *result = orchestrator->current_comprehension;
    return 0;
}

int language_orchestrator_get_production_plan(
    const language_orchestrator_t* orchestrator,
    language_production_plan_t* plan)
{
    if (!orchestrator || !plan) {
        return -1;
    }

    if (!orchestrator->production_valid) {
        return -1;
    }

    *plan = orchestrator->current_production;
    return 0;
}

//=============================================================================
// Update Cycle API
//=============================================================================

int language_orchestrator_update(
    language_orchestrator_t* orchestrator,
    uint64_t current_time_ms)
{
    if (!orchestrator || !orchestrator->running) {
        return -1;
    }

    orchestrator->last_update_ms = current_time_ms;
    orchestrator->stats.last_update_ms = current_time_ms;

    /* Update bridges */
    orchestrator_update_bridges(orchestrator, current_time_ms);

    /* Process state machine */
    orchestrator_process_state(orchestrator, current_time_ms);

    /* Check for state timeout */
    if (orchestrator->state != LANGUAGE_STATE_IDLE &&
        orchestrator->state != LANGUAGE_STATE_ERROR) {
        uint64_t elapsed = current_time_ms - orchestrator->state_entry_time_ms;
        if (elapsed > ORCHESTRATOR_STATE_TIMEOUT_MS) {
            /* Timeout - return to idle */
            orchestrator_transition_state(orchestrator,
                LANGUAGE_STATE_IDLE,
                current_time_ms);
        }
    }

    return 0;
}

int language_orchestrator_process_messages(language_orchestrator_t* orchestrator)
{
    if (!orchestrator) {
        return -1;
    }

    /* Process bio-async messages */
    int count = 0;

    /* Message processing would be handled by bio-async integration */
    orchestrator->stats.bio_async_messages += (uint64_t)count;

    return count;
}

//=============================================================================
// State API
//=============================================================================

language_state_t language_orchestrator_get_state(
    const language_orchestrator_t* orchestrator)
{
    if (!orchestrator) {
        return LANGUAGE_STATE_ERROR;
    }
    return orchestrator->state;
}

language_mode_t language_orchestrator_get_mode(
    const language_orchestrator_t* orchestrator)
{
    if (!orchestrator) {
        return LANGUAGE_MODE_COMPREHENSION;
    }
    return orchestrator->mode;
}

int language_orchestrator_set_mode(
    language_orchestrator_t* orchestrator,
    language_mode_t mode)
{
    if (!orchestrator || mode >= LANGUAGE_MODE_COUNT) {
        return -1;
    }

    orchestrator->mode = mode;
    orchestrator->stats.current_mode = mode;

    return 0;
}

int language_orchestrator_reset(language_orchestrator_t* orchestrator)
{
    if (!orchestrator) {
        return -1;
    }

    /* Clear buffers */
    orchestrator->input.phoneme_count = 0;
    orchestrator->input.word_count = 0;
    orchestrator->input.text_length = 0;

    /* Clear results */
    orchestrator->comprehension_valid = false;
    orchestrator->production_valid = false;

    /* Return to idle */
    orchestrator->state = LANGUAGE_STATE_IDLE;
    orchestrator->stats.current_state = LANGUAGE_STATE_IDLE;

    return 0;
}

//=============================================================================
// Statistics API
//=============================================================================

int language_orchestrator_get_stats(
    const language_orchestrator_t* orchestrator,
    language_orchestrator_stats_t* stats)
{
    if (!orchestrator || !stats) {
        return -1;
    }

    *stats = orchestrator->stats;
    return 0;
}

void language_orchestrator_reset_stats(language_orchestrator_t* orchestrator)
{
    if (!orchestrator) return;

    /* Preserve connection status */
    bool wernicke = orchestrator->stats.wernicke_connected;
    bool broca = orchestrator->stats.broca_connected;
    bool nlp = orchestrator->stats.nlp_connected;
    bool perception = orchestrator->stats.perception_bridge_connected;
    bool cognitive = orchestrator->stats.cognitive_bridge_connected;
    bool training = orchestrator->stats.training_bridge_connected;
    bool omni = orchestrator->stats.omni_bridge_connected;
    bool immune = orchestrator->stats.immune_bridge_connected;
    bool gpu = orchestrator->stats.gpu_bridge_connected;
    bool bio_async = orchestrator->stats.bio_async_connected;

    memset(&orchestrator->stats, 0, sizeof(orchestrator->stats));

    /* Restore connection status */
    orchestrator->stats.wernicke_connected = wernicke;
    orchestrator->stats.broca_connected = broca;
    orchestrator->stats.nlp_connected = nlp;
    orchestrator->stats.perception_bridge_connected = perception;
    orchestrator->stats.cognitive_bridge_connected = cognitive;
    orchestrator->stats.training_bridge_connected = training;
    orchestrator->stats.omni_bridge_connected = omni;
    orchestrator->stats.immune_bridge_connected = immune;
    orchestrator->stats.gpu_bridge_connected = gpu;
    orchestrator->stats.bio_async_connected = bio_async;

    orchestrator->stats.current_state = orchestrator->state;
    orchestrator->stats.current_mode = orchestrator->mode;
}

//=============================================================================
// Event API
//=============================================================================

int language_orchestrator_register_callback(
    language_orchestrator_t* orchestrator,
    language_event_callback_t callback,
    void* user_data)
{
    if (!orchestrator || !callback) {
        return -1;
    }

    /* Find empty slot */
    for (uint32_t i = 0; i < ORCHESTRATOR_MAX_CALLBACKS; i++) {
        if (!orchestrator->callbacks[i].active) {
            orchestrator->callbacks[i].callback = callback;
            orchestrator->callbacks[i].user_data = user_data;
            orchestrator->callbacks[i].active = true;
            orchestrator->num_callbacks++;
            return 0;
        }
    }

    return -1;  /* No space */
}

int language_orchestrator_unregister_callback(
    language_orchestrator_t* orchestrator,
    language_event_callback_t callback)
{
    if (!orchestrator || !callback) {
        return -1;
    }

    for (uint32_t i = 0; i < ORCHESTRATOR_MAX_CALLBACKS; i++) {
        if (orchestrator->callbacks[i].active &&
            orchestrator->callbacks[i].callback == callback) {
            orchestrator->callbacks[i].active = false;
            orchestrator->callbacks[i].callback = NULL;
            orchestrator->callbacks[i].user_data = NULL;
            orchestrator->num_callbacks--;
            return 0;
        }
    }

    return -1;  /* Not found */
}

//=============================================================================
// Memory Management
//=============================================================================

void language_comprehension_result_free(language_comprehension_result_t* result)
{
    if (!result) return;

    if (result->words) {
        free(result->words);
        result->words = NULL;
    }
    if (result->concepts) {
        free(result->concepts);
        result->concepts = NULL;
    }
    if (result->parse_tree) {
        /* Recursive tree free would go here */
        free(result->parse_tree);
        result->parse_tree = NULL;
    }
    if (result->semantic_vector) {
        free(result->semantic_vector);
        result->semantic_vector = NULL;
    }
    if (result->prosody.pitch_contour) {
        free(result->prosody.pitch_contour);
        result->prosody.pitch_contour = NULL;
    }
    if (result->prosody.intensity_contour) {
        free(result->prosody.intensity_contour);
        result->prosody.intensity_contour = NULL;
    }
}

void language_production_plan_free(language_production_plan_t* plan)
{
    if (!plan) return;

    /* Note: semantic_input is not owned by plan */
    if (plan->words) {
        free(plan->words);
        plan->words = NULL;
    }
    if (plan->phonemes) {
        free(plan->phonemes);
        plan->phonemes = NULL;
    }
    if (plan->motor_commands) {
        free(plan->motor_commands);
        plan->motor_commands = NULL;
    }
    if (plan->prosody.pitch_contour) {
        free(plan->prosody.pitch_contour);
        plan->prosody.pitch_contour = NULL;
    }
    if (plan->prosody.intensity_contour) {
        free(plan->prosody.intensity_contour);
        plan->prosody.intensity_contour = NULL;
    }
}

//=============================================================================
// Internal Functions
//=============================================================================

static int orchestrator_init_buffers(language_orchestrator_t* orch)
{
    /* Allocate phoneme buffer */
    orch->input.phoneme_capacity = orch->config.phoneme_buffer_size;
    orch->input.phonemes = calloc(orch->input.phoneme_capacity,
                                   sizeof(language_phoneme_t));
    if (!orch->input.phonemes) {
        return -1;
    }
    orch->input.phoneme_count = 0;

    /* Allocate word buffer */
    orch->input.word_capacity = orch->config.max_utterance_words;
    orch->input.words = calloc(orch->input.word_capacity,
                                sizeof(language_word_t));
    if (!orch->input.words) {
        free(orch->input.phonemes);
        return -1;
    }
    orch->input.word_count = 0;

    /* Allocate text buffer */
    orch->input.text_capacity = 4096;
    orch->input.text_buffer = calloc(orch->input.text_capacity, 1);
    if (!orch->input.text_buffer) {
        free(orch->input.phonemes);
        free(orch->input.words);
        return -1;
    }
    orch->input.text_length = 0;

    return 0;
}

static void orchestrator_free_buffers(language_orchestrator_t* orch)
{
    if (orch->input.phonemes) {
        free(orch->input.phonemes);
        orch->input.phonemes = NULL;
    }
    if (orch->input.words) {
        free(orch->input.words);
        orch->input.words = NULL;
    }
    if (orch->input.text_buffer) {
        free(orch->input.text_buffer);
        orch->input.text_buffer = NULL;
    }
}

static int orchestrator_transition_state(
    language_orchestrator_t* orch,
    language_state_t new_state,
    uint64_t current_time_ms)
{
    if (orch->state == new_state) {
        return 0;  /* No transition needed */
    }

    /* Store previous state */
    orch->prev_state = orch->state;
    orch->state = new_state;
    orch->state_entry_time_ms = current_time_ms;

    /* Update stats */
    orch->stats.current_state = new_state;
    orch->stats.state_entry_time_ms = current_time_ms;
    orch->stats.state_transitions++;

    /* Emit state change event */
    language_event_t event;
    memset(&event, 0, sizeof(event));
    event.type = LANGUAGE_EVENT_STATE_CHANGE;
    event.timestamp_ms = current_time_ms;
    event.source_module = BIO_MODULE_LANGUAGE_LAYER;
    event.data.state_change.old_state = orch->prev_state;
    event.data.state_change.new_state = new_state;

    orchestrator_emit_event(orch, &event);

    return 0;
}

static int orchestrator_process_state(
    language_orchestrator_t* orch,
    uint64_t current_time_ms)
{
    switch (orch->state) {
        case LANGUAGE_STATE_IDLE:
            /* Check for pending input */
            if (orch->input.phoneme_count > 0 ||
                orch->input.text_length > 0) {
                orchestrator_transition_state(orch,
                    LANGUAGE_STATE_LISTENING,
                    current_time_ms);
            }
            break;

        case LANGUAGE_STATE_LISTENING:
            /* Transition to comprehending once we have sufficient input */
            if (orch->input.phoneme_count >= 3 ||
                orch->input.text_length > 0) {
                orchestrator_transition_state(orch,
                    LANGUAGE_STATE_COMPREHENDING,
                    current_time_ms);
            }
            break;

        case LANGUAGE_STATE_COMPREHENDING:
            /* Process through Wernicke if connected */
            if (orch->wernicke) {
                /* Wernicke processing would happen here */
                /* For now, simulate completion */
            }

            /* Transition to integrating */
            orchestrator_transition_state(orch,
                LANGUAGE_STATE_INTEGRATING,
                current_time_ms);
            break;

        case LANGUAGE_STATE_INTEGRATING:
            /* Integration with cognitive systems */
            if (orch->cognitive_bridge) {
                /* Cognitive integration would happen here */
            }

            /* Check mode for next state */
            if (orch->mode == LANGUAGE_MODE_PRODUCTION ||
                orch->mode == LANGUAGE_MODE_DIALOGUE) {
                /* If production needed, go to generating */
                if (orch->production_valid ||
                    orch->current_production.semantic_input) {
                    orchestrator_transition_state(orch,
                        LANGUAGE_STATE_GENERATING,
                        current_time_ms);
                } else {
                    /* Done comprehending, return to idle */
                    orchestrator_transition_state(orch,
                        LANGUAGE_STATE_IDLE,
                        current_time_ms);

                    /* Mark comprehension complete */
                    orch->comprehension_valid = true;
                    orch->stats.utterances_comprehended++;
                }
            } else {
                /* Comprehension only - done */
                orchestrator_transition_state(orch,
                    LANGUAGE_STATE_IDLE,
                    current_time_ms);
                orch->comprehension_valid = true;
                orch->stats.utterances_comprehended++;
            }
            break;

        case LANGUAGE_STATE_GENERATING:
            /* Process through Broca if connected */
            if (orch->broca) {
                /* Broca processing would happen here */
            }

            /* Transition to producing */
            orchestrator_transition_state(orch,
                LANGUAGE_STATE_PRODUCING,
                current_time_ms);
            break;

        case LANGUAGE_STATE_PRODUCING:
            /* Motor output phase */
            /* Production completion */
            orch->production_valid = true;
            orch->stats.utterances_produced++;

            /* Clear input buffers */
            orch->input.phoneme_count = 0;
            orch->input.word_count = 0;
            orch->input.text_length = 0;

            /* Return to idle */
            orchestrator_transition_state(orch,
                LANGUAGE_STATE_IDLE,
                current_time_ms);
            break;

        case LANGUAGE_STATE_ERROR:
            /* Error recovery - return to idle after delay */
            {
                uint64_t error_duration = current_time_ms -
                                          orch->state_entry_time_ms;
                if (error_duration > 1000) {  /* 1 second recovery */
                    orchestrator_transition_state(orch,
                        LANGUAGE_STATE_IDLE,
                        current_time_ms);
                }
            }
            break;

        default:
            break;
    }

    return 0;
}

static void orchestrator_emit_event(
    language_orchestrator_t* orch,
    const language_event_t* event)
{
    for (uint32_t i = 0; i < ORCHESTRATOR_MAX_CALLBACKS; i++) {
        if (orch->callbacks[i].active && orch->callbacks[i].callback) {
            orch->callbacks[i].callback(event, orch->callbacks[i].user_data);
        }
    }
}

static int orchestrator_update_bridges(
    language_orchestrator_t* orch,
    uint64_t current_time_ms)
{
    /* Update all connected bridges */
    if (orch->perception_bridge) {
        language_perception_bridge_update(orch->perception_bridge,
                                          current_time_ms);
    }
    if (orch->cognitive_bridge) {
        language_cognitive_bridge_update(orch->cognitive_bridge,
                                         current_time_ms);
    }
    if (orch->training_bridge) {
        language_training_bridge_update(orch->training_bridge,
                                        current_time_ms);
    }
    if (orch->omni_bridge) {
        language_omni_bridge_update(orch->omni_bridge,
                                    current_time_ms);
    }
    if (orch->immune_bridge) {
        language_immune_bridge_update(orch->immune_bridge,
                                      current_time_ms);
    }
    if (orch->gpu_bridge) {
        language_gpu_bridge_update(orch->gpu_bridge,
                                   current_time_ms);
    }

    return 0;
}

//=============================================================================
// String Conversion - from nimcp_language_types.h
//=============================================================================

const char* language_state_to_string(language_state_t state)
{
    switch (state) {
        case LANGUAGE_STATE_IDLE:          return "IDLE";
        case LANGUAGE_STATE_LISTENING:     return "LISTENING";
        case LANGUAGE_STATE_COMPREHENDING: return "COMPREHENDING";
        case LANGUAGE_STATE_INTEGRATING:   return "INTEGRATING";
        case LANGUAGE_STATE_GENERATING:    return "GENERATING";
        case LANGUAGE_STATE_PRODUCING:     return "PRODUCING";
        case LANGUAGE_STATE_ERROR:         return "ERROR";
        default:                           return "UNKNOWN";
    }
}

const char* language_mode_to_string(language_mode_t mode)
{
    switch (mode) {
        case LANGUAGE_MODE_COMPREHENSION: return "COMPREHENSION";
        case LANGUAGE_MODE_PRODUCTION:    return "PRODUCTION";
        case LANGUAGE_MODE_DIALOGUE:      return "DIALOGUE";
        case LANGUAGE_MODE_REPETITION:    return "REPETITION";
        case LANGUAGE_MODE_TRANSLATION:   return "TRANSLATION";
        default:                          return "UNKNOWN";
    }
}

const char* language_input_type_to_string(language_input_type_t type)
{
    switch (type) {
        case LANGUAGE_INPUT_AUDIO:    return "AUDIO";
        case LANGUAGE_INPUT_TEXT:     return "TEXT";
        case LANGUAGE_INPUT_TOKENS:   return "TOKENS";
        case LANGUAGE_INPUT_PHONEMES: return "PHONEMES";
        case LANGUAGE_INPUT_SEMANTIC: return "SEMANTIC";
        case LANGUAGE_INPUT_VISUAL:   return "VISUAL";
        default:                      return "UNKNOWN";
    }
}

const char* language_output_type_to_string(language_output_type_t type)
{
    switch (type) {
        case LANGUAGE_OUTPUT_MOTOR_COMMANDS: return "MOTOR_COMMANDS";
        case LANGUAGE_OUTPUT_PHONEMES:       return "PHONEMES";
        case LANGUAGE_OUTPUT_TOKENS:         return "TOKENS";
        case LANGUAGE_OUTPUT_TEXT:           return "TEXT";
        case LANGUAGE_OUTPUT_SEMANTIC:       return "SEMANTIC";
        default:                             return "UNKNOWN";
    }
}

const char* phoneme_category_to_string(phoneme_category_t cat)
{
    switch (cat) {
        case PHONEME_CAT_VOWEL:       return "VOWEL";
        case PHONEME_CAT_STOP:        return "STOP";
        case PHONEME_CAT_FRICATIVE:   return "FRICATIVE";
        case PHONEME_CAT_AFFRICATE:   return "AFFRICATE";
        case PHONEME_CAT_NASAL:       return "NASAL";
        case PHONEME_CAT_APPROXIMANT: return "APPROXIMANT";
        case PHONEME_CAT_SILENCE:     return "SILENCE";
        case PHONEME_CAT_UNKNOWN:     return "UNKNOWN";
        default:                      return "UNKNOWN";
    }
}

const char* part_of_speech_to_string(part_of_speech_t pos)
{
    switch (pos) {
        case POS_NOUN:          return "NOUN";
        case POS_VERB:          return "VERB";
        case POS_ADJECTIVE:     return "ADJECTIVE";
        case POS_ADVERB:        return "ADVERB";
        case POS_DETERMINER:    return "DETERMINER";
        case POS_PREPOSITION:   return "PREPOSITION";
        case POS_CONJUNCTION:   return "CONJUNCTION";
        case POS_PRONOUN:       return "PRONOUN";
        case POS_AUXILIARY:     return "AUXILIARY";
        case POS_COMPLEMENTIZER:return "COMPLEMENTIZER";
        case POS_NEGATION:      return "NEGATION";
        case POS_PUNCTUATION:   return "PUNCTUATION";
        case POS_UNKNOWN:       return "UNKNOWN";
        default:                return "UNKNOWN";
    }
}

const char* thematic_role_to_string(thematic_role_t role)
{
    switch (role) {
        case THEMATIC_ROLE_AGENT:       return "AGENT";
        case THEMATIC_ROLE_PATIENT:     return "PATIENT";
        case THEMATIC_ROLE_THEME:       return "THEME";
        case THEMATIC_ROLE_EXPERIENCER: return "EXPERIENCER";
        case THEMATIC_ROLE_BENEFICIARY: return "BENEFICIARY";
        case THEMATIC_ROLE_INSTRUMENT:  return "INSTRUMENT";
        case THEMATIC_ROLE_LOCATION:    return "LOCATION";
        case THEMATIC_ROLE_SOURCE:      return "SOURCE";
        case THEMATIC_ROLE_GOAL:        return "GOAL";
        case THEMATIC_ROLE_TIME:        return "TIME";
        case THEMATIC_ROLE_MANNER:      return "MANNER";
        case THEMATIC_ROLE_CAUSE:       return "CAUSE";
        case THEMATIC_ROLE_NONE:        return "NONE";
        default:                        return "UNKNOWN";
    }
}

const char* phrase_type_to_string(phrase_type_t type)
{
    switch (type) {
        case PHRASE_NP:   return "NP";
        case PHRASE_VP:   return "VP";
        case PHRASE_PP:   return "PP";
        case PHRASE_AP:   return "AP";
        case PHRASE_ADVP: return "ADVP";
        case PHRASE_S:    return "S";
        case PHRASE_SBAR: return "SBAR";
        case PHRASE_CP:   return "CP";
        case PHRASE_IP:   return "IP";
        case PHRASE_DP:   return "DP";
        default:          return "UNKNOWN";
    }
}

const char* parse_state_to_string(parse_state_t state)
{
    switch (state) {
        case PARSE_STATE_INIT:        return "INIT";
        case PARSE_STATE_ACTIVE:      return "ACTIVE";
        case PARSE_STATE_COMPLETE:    return "COMPLETE";
        case PARSE_STATE_AMBIGUOUS:   return "AMBIGUOUS";
        case PARSE_STATE_GARDEN_PATH: return "GARDEN_PATH";
        case PARSE_STATE_ERROR:       return "ERROR";
        default:                      return "UNKNOWN";
    }
}

const char* language_event_type_to_string(language_event_type_t type)
{
    switch (type) {
        case LANGUAGE_EVENT_UTTERANCE_START:       return "UTTERANCE_START";
        case LANGUAGE_EVENT_PHONEME_RECOGNIZED:    return "PHONEME_RECOGNIZED";
        case LANGUAGE_EVENT_WORD_RECOGNIZED:       return "WORD_RECOGNIZED";
        case LANGUAGE_EVENT_CONCEPT_ACTIVATED:     return "CONCEPT_ACTIVATED";
        case LANGUAGE_EVENT_COMPREHENSION_COMPLETE:return "COMPREHENSION_COMPLETE";
        case LANGUAGE_EVENT_PRODUCTION_START:      return "PRODUCTION_START";
        case LANGUAGE_EVENT_PRODUCTION_COMPLETE:   return "PRODUCTION_COMPLETE";
        case LANGUAGE_EVENT_AMBIGUITY_DETECTED:    return "AMBIGUITY_DETECTED";
        case LANGUAGE_EVENT_ANOMALY_DETECTED:      return "ANOMALY_DETECTED";
        case LANGUAGE_EVENT_ERROR:                 return "ERROR";
        case LANGUAGE_EVENT_STATE_CHANGE:          return "STATE_CHANGE";
        case LANGUAGE_EVENT_TRAINING_UPDATE:       return "TRAINING_UPDATE";
        default:                                   return "UNKNOWN";
    }
}
