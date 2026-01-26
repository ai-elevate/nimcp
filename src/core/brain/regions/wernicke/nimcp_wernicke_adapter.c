/**
 * @file nimcp_wernicke_adapter.c
 * @brief Implementation of Wernicke's area adapter for language comprehension
 *
 * WHAT: Core implementation of language comprehension processing
 * WHY:  Enable speech understanding in NIMCP brain
 * HOW:  Phonological → Lexical → Semantic → Syntactic pipeline
 *
 * @version Phase W1: Wernicke's Area Core Integration
 * @date 2026-01-04
 */

#include "core/brain/regions/wernicke/nimcp_wernicke_adapter.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for wernicke_adapter module */
static nimcp_health_agent_t* g_wernicke_adapter_health_agent = NULL;

/**
 * @brief Set health agent for wernicke_adapter heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void wernicke_adapter_set_health_agent(nimcp_health_agent_t* agent) {
    g_wernicke_adapter_health_agent = agent;
}

/** @brief Send heartbeat from wernicke_adapter module */
static inline void wernicke_adapter_heartbeat(const char* operation, float progress) {
    if (g_wernicke_adapter_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_wernicke_adapter_health_agent, operation, progress);
    }
}


/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Lexicon entry for hash table
 */
typedef struct lexicon_entry {
    wernicke_word_t word;
    struct lexicon_entry* next;
} lexicon_entry_t;

/**
 * @brief Working memory state
 */
typedef struct {
    phoneme_t buffer[WERNICKE_DEFAULT_WORKING_MEMORY_SLOTS * 4];
    uint32_t count;
    uint32_t capacity;
    float decay_rate;
    uint64_t last_rehearsal_ms;
} working_memory_t;

/**
 * @brief Phoneme buffer for processing
 */
typedef struct {
    phoneme_event_t* events;
    uint32_t count;
    uint32_t capacity;
} phoneme_buffer_t;

/**
 * @brief Wernicke adapter internal structure
 */
struct wernicke_adapter {
    /* Configuration */
    wernicke_config_t config;

    /* Processing state */
    wernicke_status_t status;
    wernicke_error_t last_error;

    /* Sub-modules (Phase 2 will implement these fully) */
    phonological_analyzer_t* phonological;
    lexical_access_t* lexical;
    semantic_integrator_t* semantic;
    syntactic_comprehension_t* syntactic;

    /* Lexicon (simple hash table) */
    lexicon_entry_t** lexicon_table;
    uint32_t lexicon_count;
    uint32_t lexicon_capacity;

    /* Working memory */
    working_memory_t working_memory;

    /* Phoneme buffer */
    phoneme_buffer_t phoneme_buffer;

    /* External connections */
    broca_adapter_t* broca;
    speech_cortex_t* speech_cortex;
    semantic_memory_t* semantic_memory;
    brain_kg_t* kg;

    /* Callbacks */
    wernicke_word_callback_t word_callback;
    void* word_callback_data;
    wernicke_concept_callback_t concept_callback;
    void* concept_callback_data;
    wernicke_comprehension_callback_t comprehension_callback;
    void* comprehension_callback_data;
    wernicke_event_callback_t event_callback;
    void* event_callback_data;

    /* Statistics */
    wernicke_stats_t stats;

    /* Bio-async communication */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;
    nimcp_bio_channel_type_t default_channel;
};

/*=============================================================================
 * LEXICON HASH TABLE HELPERS
 *===========================================================================*/

/**
 * @brief Simple string hash function
 */
static uint32_t hash_string(const char* str, uint32_t table_size)
{
    uint32_t hash = 5381;
    int c;
    while ((c = *str++) != 0) {
        hash = ((hash << 5) + hash) + (uint32_t)c;
    }
    return hash % table_size;
}

/**
 * @brief Hash phoneme sequence
 */
static uint32_t hash_phonemes(const uint8_t* phonemes, uint32_t count, uint32_t table_size)
{
    uint32_t hash = 5381;
    for (uint32_t i = 0; i < count; i++) {
        hash = ((hash << 5) + hash) + phonemes[i];
    }
    return hash % table_size;
}

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

wernicke_config_t wernicke_default_config(void)
{
    wernicke_config_t config = {
        /* Capacity limits */
        .max_phonemes = WERNICKE_DEFAULT_MAX_PHONEMES,
        .max_words = WERNICKE_DEFAULT_MAX_WORDS,
        .max_concepts = WERNICKE_DEFAULT_MAX_CONCEPTS,

        /* Lexicon configuration */
        .lexicon_size = WERNICKE_DEFAULT_LEXICON_SIZE,
        .enable_lexicon = true,

        /* Working memory */
        .working_memory_slots = WERNICKE_DEFAULT_WORKING_MEMORY_SLOTS,
        .enable_working_memory = true,

        /* Processing layers */
        .enable_phonological = true,
        .enable_lexical = true,
        .enable_semantic = true,
        .enable_syntactic = false,  /* Phase 5 */

        /* Feature configuration */
        .embedding_dim = WERNICKE_DEFAULT_EMBEDDING_DIM,
        .formant_count = WERNICKE_DEFAULT_FORMANT_COUNT,

        /* Cross-modal */
        .enable_audiovisual = false,  /* Phase 4 */
        .enable_prosody = true,

        /* External connections */
        .enable_broca_connection = true,
        .enable_semantic_memory = true,
        .enable_kg_registration = true,

        /* Event system */
        .enable_events = true,

        /* Training */
        .enable_training = true,
        .learning_rate = 0.01f,

        /* Timing */
        .processing_window_ms = WERNICKE_DEFAULT_PROCESSING_WINDOW_MS,

        /* Bio-async */
        .enable_bio_async = true,
        .default_channel = BIO_CHANNEL_DOPAMINE
    };
    return config;
}

wernicke_adapter_t* wernicke_create(const wernicke_config_t* config)
{
    wernicke_config_t cfg = config ? *config : wernicke_default_config();

    /* Allocate adapter */
    wernicke_adapter_t* adapter = (wernicke_adapter_t*)calloc(1, sizeof(wernicke_adapter_t));
    if (!adapter) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adapter is NULL");

        return NULL;
    }

    adapter->config = cfg;
    adapter->status = WERNICKE_STATUS_IDLE;
    adapter->last_error = WERNICKE_ERROR_NONE;

    /* Allocate lexicon hash table */
    if (cfg.enable_lexicon) {
        adapter->lexicon_capacity = cfg.lexicon_size;
        adapter->lexicon_table = (lexicon_entry_t**)calloc(cfg.lexicon_size, sizeof(lexicon_entry_t*));
        if (!adapter->lexicon_table) {
            free(adapter);
            return NULL;
        }
        adapter->lexicon_count = 0;
    }

    /* Initialize working memory */
    if (cfg.enable_working_memory) {
        adapter->working_memory.capacity = cfg.working_memory_slots * 4; /* 4 phonemes per slot avg */
        adapter->working_memory.count = 0;
        adapter->working_memory.decay_rate = 0.1f;
        adapter->working_memory.last_rehearsal_ms = 0;
    }

    /* Allocate phoneme buffer */
    adapter->phoneme_buffer.capacity = cfg.max_phonemes;
    adapter->phoneme_buffer.events = (phoneme_event_t*)calloc(cfg.max_phonemes, sizeof(phoneme_event_t));
    if (!adapter->phoneme_buffer.events) {
        if (adapter->lexicon_table) free(adapter->lexicon_table);
        free(adapter);
        return NULL;
    }
    adapter->phoneme_buffer.count = 0;

    /* Initialize bio-async if enabled */
    adapter->bio_async_enabled = cfg.enable_bio_async;
    adapter->default_channel = cfg.default_channel;
    adapter->bio_ctx = NULL;  /* Will be registered via wernicke_register_kg */

    /* Initialize statistics */
    memset(&adapter->stats, 0, sizeof(wernicke_stats_t));

    NIMCP_LOG_INFO("wernicke", "Created Wernicke's area adapter (lexicon=%u, wm=%u slots)",
                   cfg.lexicon_size, cfg.working_memory_slots);

    return adapter;
}

void wernicke_destroy(wernicke_adapter_t* adapter)
{
    if (!adapter) return;

    /* Free lexicon entries */
    if (adapter->lexicon_table) {
        for (uint32_t i = 0; i < adapter->lexicon_capacity; i++) {
            lexicon_entry_t* entry = adapter->lexicon_table[i];
            while (entry) {
                lexicon_entry_t* next = entry->next;
                free(entry);
                entry = next;
            }
        }
        free(adapter->lexicon_table);
    }

    /* Free phoneme buffer */
    if (adapter->phoneme_buffer.events) {
        free(adapter->phoneme_buffer.events);
    }

    /* Free sub-modules when implemented */
    /* Phase 2: phonological, lexical, semantic, syntactic */

    NIMCP_LOG_INFO("wernicke", "Destroyed Wernicke's area adapter");

    free(adapter);
}

bool wernicke_reset(wernicke_adapter_t* adapter)
{
    if (!adapter) return false;

    adapter->status = WERNICKE_STATUS_IDLE;
    adapter->last_error = WERNICKE_ERROR_NONE;

    /* Clear phoneme buffer */
    adapter->phoneme_buffer.count = 0;

    /* Clear working memory */
    adapter->working_memory.count = 0;

    NIMCP_LOG_DEBUG("wernicke", "Reset adapter state");
    return true;
}

/*=============================================================================
 * PHONOLOGICAL PROCESSING
 *===========================================================================*/

bool wernicke_process_audio(
    wernicke_adapter_t* adapter,
    const float* audio,
    uint32_t num_samples,
    uint32_t sample_rate,
    phoneme_event_t* phonemes,
    uint32_t max_phonemes,
    uint32_t* num_detected)
{
    if (!adapter || !audio || !phonemes || !num_detected) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    adapter->status = WERNICKE_STATUS_PHONOLOGICAL;

    /* Phase 2 will implement full formant analysis
     * For now, delegate to speech cortex if connected */
    if (adapter->speech_cortex) {
        /* Use speech cortex for phoneme detection */
        bool result = speech_cortex_detect_phonemes(
            adapter->speech_cortex,
            audio,
            num_samples,
            phonemes,
            max_phonemes,
            num_detected
        );
        if (result) {
            adapter->stats.phonemes_processed += *num_detected;
        }
        adapter->status = WERNICKE_STATUS_IDLE;
        return result;
    }

    /* Fallback: No speech cortex connected */
    /* Phase 2 will implement standalone phoneme detection */
    *num_detected = 0;
    adapter->status = WERNICKE_STATUS_IDLE;
    NIMCP_LOG_WARN("wernicke", "No speech cortex connected for phoneme detection");
    return true;
}

bool wernicke_process_phonemes(
    wernicke_adapter_t* adapter,
    const phoneme_event_t* phonemes,
    uint32_t count)
{
    if (!adapter || !phonemes || count == 0) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    if (count > adapter->phoneme_buffer.capacity) {
        adapter->last_error = WERNICKE_ERROR_BUFFER_OVERFLOW;
        return false;
    }

    adapter->status = WERNICKE_STATUS_PHONOLOGICAL;

    /* Store phonemes in buffer */
    memcpy(adapter->phoneme_buffer.events, phonemes, count * sizeof(phoneme_event_t));
    adapter->phoneme_buffer.count = count;
    adapter->stats.phonemes_processed += count;

    /* Calculate average confidence */
    float total_conf = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        total_conf += phonemes[i].confidence;
    }

    NIMCP_LOG_DEBUG("wernicke", "Processed %u phonemes (avg conf=%.2f)",
                   count, total_conf / (float)count);

    adapter->status = WERNICKE_STATUS_IDLE;
    return true;
}

/*=============================================================================
 * LEXICAL ACCESS
 *===========================================================================*/

bool wernicke_recognize_word(
    wernicke_adapter_t* adapter,
    const phoneme_t* phonemes,
    uint32_t count,
    wernicke_word_result_t* result)
{
    if (!adapter || !phonemes || count == 0 || !result) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    adapter->status = WERNICKE_STATUS_LEXICAL_ACCESS;

    /* Hash the phoneme sequence */
    uint32_t hash = hash_phonemes((const uint8_t*)phonemes, count, adapter->lexicon_capacity);

    /* Search hash bucket */
    lexicon_entry_t* entry = adapter->lexicon_table[hash];
    while (entry) {
        if (entry->word.phoneme_count == count) {
            bool match = true;
            for (uint32_t i = 0; i < count; i++) {
                if (entry->word.phonemes[i] != (uint8_t)phonemes[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                /* Found word */
                result->word = entry->word;
                result->confidence = 1.0f;
                result->onset_time_ms = 0;  /* Would be set from phoneme events */
                result->offset_time_ms = 0;
                result->position_in_utterance = 0;

                adapter->stats.words_recognized++;
                adapter->stats.successful_recognitions++;

                /* Fire callback if set */
                if (adapter->word_callback) {
                    adapter->word_callback(result, adapter->word_callback_data);
                }

                NIMCP_LOG_DEBUG("wernicke", "Recognized word: %s (conf=%.2f)",
                               result->word.word, result->confidence);

                adapter->status = WERNICKE_STATUS_IDLE;
                return true;
            }
        }
        entry = entry->next;
    }

    /* Word not found */
    adapter->last_error = WERNICKE_ERROR_WORD_NOT_FOUND;
    adapter->stats.lexical_misses++;
    adapter->status = WERNICKE_STATUS_IDLE;
    return false;
}

bool wernicke_add_word(
    wernicke_adapter_t* adapter,
    const wernicke_word_t* word)
{
    if (!adapter || !word || word->phoneme_count == 0) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    if (!adapter->lexicon_table) {
        adapter->last_error = WERNICKE_ERROR_INTERNAL;
        return false;
    }

    /* Hash by phoneme sequence */
    uint32_t hash = hash_phonemes(word->phonemes, word->phoneme_count, adapter->lexicon_capacity);

    /* Create new entry */
    lexicon_entry_t* entry = (lexicon_entry_t*)malloc(sizeof(lexicon_entry_t));
    if (!entry) {
        adapter->last_error = WERNICKE_ERROR_INTERNAL;
        return false;
    }

    entry->word = *word;
    entry->next = adapter->lexicon_table[hash];
    adapter->lexicon_table[hash] = entry;
    adapter->lexicon_count++;

    NIMCP_LOG_DEBUG("wernicke", "Added word to lexicon: %s (%u phonemes)",
                   word->word, word->phoneme_count);

    return true;
}

bool wernicke_lookup_word(
    const wernicke_adapter_t* adapter,
    const char* word_str,
    wernicke_word_t* entry)
{
    if (!adapter || !word_str || !entry) {
        return false;
    }

    if (!adapter->lexicon_table) {
        return false;
    }

    /* Lexicon is indexed by phoneme hash, so we must search all buckets
     * to find a word by its string. This is O(n) but acceptable for
     * typical lexicon sizes. A secondary string index could be added
     * for O(1) string lookups if needed. */
    for (uint32_t i = 0; i < adapter->lexicon_capacity; i++) {
        lexicon_entry_t* e = adapter->lexicon_table[i];
        while (e) {
            if (strcmp(e->word.word, word_str) == 0) {
                *entry = e->word;
                return true;
            }
            e = e->next;
        }
    }

    return false;
}

bool wernicke_predict_next_word(
    wernicke_adapter_t* adapter,
    const wernicke_context_t* context,
    wernicke_word_pred_t* prediction)
{
    if (!adapter || !prediction) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    /* Phase 3 will implement n-gram prediction and semantic priming */
    /* For now, return empty prediction */
    prediction->num_candidates = 0;
    return true;
}

/*=============================================================================
 * SEMANTIC INTEGRATION
 *===========================================================================*/

bool wernicke_get_meaning(
    wernicke_adapter_t* adapter,
    const wernicke_word_result_t* word,
    wernicke_concept_t* concept_out)
{
    if (!adapter || !word || !concept_out) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    adapter->status = WERNICKE_STATUS_SEMANTIC;

    /* Phase 3 will implement full semantic memory integration */
    /* For now, create a basic concept from the word */
    concept_out->concept_id = word->word.concept_id;
    strncpy(concept_out->concept_name, word->word.word, sizeof(concept_out->concept_name) - 1);
    concept_out->concept_name[sizeof(concept_out->concept_name) - 1] = '\0';
    concept_out->activation = word->confidence;
    concept_out->embedding = NULL;
    concept_out->embedding_dim = 0;
    concept_out->related_concepts = NULL;
    concept_out->num_related = 0;

    adapter->stats.concepts_activated++;

    /* Fire callback if set */
    if (adapter->concept_callback) {
        adapter->concept_callback(concept_out, adapter->concept_callback_data);
    }

    adapter->status = WERNICKE_STATUS_IDLE;
    return true;
}

bool wernicke_disambiguate(
    wernicke_adapter_t* adapter,
    const wernicke_word_result_t* word,
    const wernicke_context_t* context,
    wernicke_concept_t* concept_out)
{
    if (!adapter || !word || !concept_out) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    /* Phase 3 will implement context-based disambiguation */
    /* For now, just return the first meaning */
    return wernicke_get_meaning(adapter, word, concept_out);
}

bool wernicke_spread_activation(
    wernicke_adapter_t* adapter,
    uint32_t concept_id,
    uint32_t depth,
    wernicke_concept_t* activated,
    uint32_t max_concepts,
    uint32_t* num_activated)
{
    if (!adapter || !activated || !num_activated) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    /* Phase 3 will implement spreading activation over semantic graph */
    *num_activated = 0;
    return true;
}

/*=============================================================================
 * SENTENCE COMPREHENSION
 *===========================================================================*/

bool wernicke_comprehend(
    wernicke_adapter_t* adapter,
    const float* audio,
    uint32_t num_samples,
    uint32_t sample_rate,
    wernicke_comprehension_t* result)
{
    if (!adapter || !audio || !result) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    /* Initialize result */
    memset(result, 0, sizeof(wernicke_comprehension_t));

    /* Step 1: Phonological processing */
    phoneme_event_t* phonemes = adapter->phoneme_buffer.events;
    uint32_t num_phonemes = 0;

    if (!wernicke_process_audio(adapter, audio, num_samples, sample_rate,
                                phonemes, adapter->phoneme_buffer.capacity, &num_phonemes)) {
        adapter->last_error = WERNICKE_ERROR_PHONOLOGICAL_FAILURE;
        return false;
    }
    result->phoneme_count = num_phonemes;

    /* Step 2: Lexical access - word segmentation */
    /* Phase 2 will implement proper word boundary detection */
    adapter->status = WERNICKE_STATUS_LEXICAL_ACCESS;

    /* Step 3: Semantic integration */
    adapter->status = WERNICKE_STATUS_SEMANTIC;

    /* Step 4: Syntactic parsing (if enabled) */
    if (adapter->config.enable_syntactic) {
        adapter->status = WERNICKE_STATUS_SYNTACTIC;
        /* Phase 5 will implement parsing */
    }

    adapter->status = WERNICKE_STATUS_COMPREHENSION_READY;
    adapter->stats.utterances_comprehended++;

    /* Fire callback if set */
    if (adapter->comprehension_callback) {
        adapter->comprehension_callback(result, adapter->comprehension_callback_data);
    }

    adapter->status = WERNICKE_STATUS_IDLE;
    return true;
}

bool wernicke_parse_sentence(
    wernicke_adapter_t* adapter,
    const wernicke_word_result_t* words,
    uint32_t count,
    wernicke_parse_t* parse)
{
    if (!adapter || !words || !parse) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    /* Phase 5 will implement CKY/Earley parsing */
    parse->root = NULL;
    parse->is_valid = false;
    parse->parse_confidence = 0.0f;
    parse->semantic_representation = NULL;

    return true;
}

void wernicke_free_comprehension(wernicke_comprehension_t* result)
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
    if (result->parse) {
        wernicke_free_parse(result->parse);
        result->parse = NULL;
    }
}

void wernicke_free_parse(wernicke_parse_t* parse)
{
    if (!parse) return;

    /* Recursively free parse tree nodes */
    if (parse->root) {
        /* Phase 5 will implement proper tree deallocation */
    }
    if (parse->semantic_representation) {
        free(parse->semantic_representation);
    }
}

/*=============================================================================
 * CROSS-MODAL INTEGRATION
 *===========================================================================*/

bool wernicke_integrate_audiovisual(
    wernicke_adapter_t* adapter,
    const phoneme_event_t* audio_phonemes,
    const float* visual_lip_shapes,
    uint32_t num_frames,
    phoneme_event_t* fused_phonemes,
    uint32_t* num_fused)
{
    if (!adapter || !audio_phonemes || !fused_phonemes || !num_fused) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    /* Phase 4 will implement McGurk effect and audiovisual fusion */
    /* For now, just copy audio phonemes */
    memcpy(fused_phonemes, audio_phonemes, num_frames * sizeof(phoneme_event_t));
    *num_fused = num_frames;
    adapter->stats.audiovisual_fusions++;

    return true;
}

/*=============================================================================
 * BROCA CONNECTION
 *===========================================================================*/

bool wernicke_connect_broca(
    wernicke_adapter_t* adapter,
    broca_adapter_t* broca)
{
    if (!adapter) return false;

    adapter->broca = broca;
    NIMCP_LOG_INFO("wernicke", "Connected to Broca's area (arcuate fasciculus)");
    return true;
}

bool wernicke_send_to_broca(
    wernicke_adapter_t* adapter,
    const wernicke_comprehension_t* comprehension)
{
    if (!adapter || !comprehension) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    if (!adapter->broca) {
        adapter->last_error = WERNICKE_ERROR_BROCA_DISCONNECTED;
        return false;
    }

    /* Phase 3 will implement bio-async message to Broca */
    NIMCP_LOG_DEBUG("wernicke", "Sent comprehension to Broca's area");
    return true;
}

bool wernicke_receive_efference_copy(
    wernicke_adapter_t* adapter,
    const broca_efference_copy_t* efference)
{
    if (!adapter || !efference) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    /* Phase 3 will implement self-monitoring via efference copy comparison */
    NIMCP_LOG_DEBUG("wernicke", "Received efference copy from Broca (%u phonemes)",
                   efference->phoneme_count);
    return true;
}

/*=============================================================================
 * WORKING MEMORY
 *===========================================================================*/

bool wernicke_wm_store(
    wernicke_adapter_t* adapter,
    const phoneme_t* phonemes,
    uint32_t count)
{
    if (!adapter || !phonemes) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    if (adapter->working_memory.count + count > adapter->working_memory.capacity) {
        adapter->last_error = WERNICKE_ERROR_WORKING_MEMORY_FULL;
        return false;
    }

    memcpy(&adapter->working_memory.buffer[adapter->working_memory.count],
           phonemes, count * sizeof(phoneme_t));
    adapter->working_memory.count += count;

    NIMCP_LOG_DEBUG("wernicke", "Stored %u phonemes in working memory (total=%u)",
                   count, adapter->working_memory.count);
    return true;
}

bool wernicke_wm_rehearse(wernicke_adapter_t* adapter)
{
    if (!adapter) return false;

    /* Reset decay timer */
    /* Phase 3 will implement proper rehearsal with re-activation */
    adapter->working_memory.last_rehearsal_ms = 0; /* Would use current time */

    NIMCP_LOG_DEBUG("wernicke", "Rehearsed working memory (%u items)",
                   adapter->working_memory.count);
    return true;
}

bool wernicke_wm_get_contents(
    const wernicke_adapter_t* adapter,
    phoneme_t* phonemes,
    uint32_t max_count,
    uint32_t* count)
{
    if (!adapter || !phonemes || !count) {
        return false;
    }

    uint32_t copy_count = adapter->working_memory.count;
    if (copy_count > max_count) {
        copy_count = max_count;
    }

    memcpy(phonemes, adapter->working_memory.buffer, copy_count * sizeof(phoneme_t));
    *count = copy_count;

    return true;
}

void wernicke_wm_clear(wernicke_adapter_t* adapter)
{
    if (!adapter) return;

    adapter->working_memory.count = 0;
    NIMCP_LOG_DEBUG("wernicke", "Cleared working memory");
}

/*=============================================================================
 * EVENT INTEGRATION
 *===========================================================================*/

bool wernicke_set_word_callback(
    wernicke_adapter_t* adapter,
    wernicke_word_callback_t callback,
    void* user_data)
{
    if (!adapter) return false;

    adapter->word_callback = callback;
    adapter->word_callback_data = user_data;
    return true;
}

bool wernicke_set_concept_callback(
    wernicke_adapter_t* adapter,
    wernicke_concept_callback_t callback,
    void* user_data)
{
    if (!adapter) return false;

    adapter->concept_callback = callback;
    adapter->concept_callback_data = user_data;
    return true;
}

bool wernicke_set_comprehension_callback(
    wernicke_adapter_t* adapter,
    wernicke_comprehension_callback_t callback,
    void* user_data)
{
    if (!adapter) return false;

    adapter->comprehension_callback = callback;
    adapter->comprehension_callback_data = user_data;
    return true;
}

bool wernicke_set_event_callback(
    wernicke_adapter_t* adapter,
    wernicke_event_callback_t callback,
    void* user_data)
{
    if (!adapter) return false;

    adapter->event_callback = callback;
    adapter->event_callback_data = user_data;
    return true;
}

/*=============================================================================
 * TRAINING INTERFACE
 *===========================================================================*/

bool wernicke_train_word(
    wernicke_adapter_t* adapter,
    const phoneme_t* phonemes,
    uint32_t num_phonemes,
    const char* target_word,
    float learning_rate)
{
    if (!adapter || !phonemes || !target_word) {
        if (adapter) adapter->last_error = WERNICKE_ERROR_INVALID_INPUT;
        return false;
    }

    /* Phase 3 will implement contrastive learning */
    /* For now, just add the word to lexicon if not present */
    wernicke_word_t word;
    if (!wernicke_lookup_word(adapter, target_word, &word)) {
        memset(&word, 0, sizeof(word));
        word.word_id = adapter->lexicon_count + 1;
        strncpy(word.word, target_word, sizeof(word.word) - 1);
        word.word[sizeof(word.word) - 1] = '\0';

        if (num_phonemes > sizeof(word.phonemes)) {
            num_phonemes = sizeof(word.phonemes);
        }
        for (uint32_t i = 0; i < num_phonemes; i++) {
            word.phonemes[i] = (uint8_t)phonemes[i];
        }
        word.phoneme_count = num_phonemes;
        word.frequency = 0.5f;

        wernicke_add_word(adapter, &word);
    }

    adapter->stats.training_iterations++;
    return true;
}

bool wernicke_train_semantic(
    wernicke_adapter_t* adapter,
    uint32_t word_id,
    uint32_t concept_id,
    float strength)
{
    if (!adapter) {
        return false;
    }

    /* Phase 3 will implement Hebbian semantic learning */
    adapter->stats.training_iterations++;
    return true;
}

/*=============================================================================
 * STATUS AND DIAGNOSTICS
 *===========================================================================*/

wernicke_status_t wernicke_get_status(const wernicke_adapter_t* adapter)
{
    if (!adapter) return WERNICKE_STATUS_ERROR;
    return adapter->status;
}

wernicke_error_t wernicke_get_last_error(const wernicke_adapter_t* adapter)
{
    if (!adapter) return WERNICKE_ERROR_INTERNAL;
    return adapter->last_error;
}

const char* wernicke_error_string(wernicke_error_t error)
{
    switch (error) {
        case WERNICKE_ERROR_NONE:               return "No error";
        case WERNICKE_ERROR_INVALID_INPUT:      return "Invalid input";
        case WERNICKE_ERROR_PHONOLOGICAL_FAILURE: return "Phonological processing failed";
        case WERNICKE_ERROR_LEXICAL_FAILURE:    return "Lexical access failed";
        case WERNICKE_ERROR_SEMANTIC_FAILURE:   return "Semantic integration failed";
        case WERNICKE_ERROR_SYNTACTIC_FAILURE:  return "Syntactic parsing failed";
        case WERNICKE_ERROR_WORKING_MEMORY_FULL: return "Working memory full";
        case WERNICKE_ERROR_WORD_NOT_FOUND:     return "Word not in lexicon";
        case WERNICKE_ERROR_CONCEPT_NOT_FOUND:  return "Concept not found";
        case WERNICKE_ERROR_BUFFER_OVERFLOW:    return "Buffer overflow";
        case WERNICKE_ERROR_BROCA_DISCONNECTED: return "Broca's area not connected";
        case WERNICKE_ERROR_INTERNAL:           return "Internal error";
        default:                                 return "Unknown error";
    }
}

const char* wernicke_status_string(wernicke_status_t status)
{
    switch (status) {
        case WERNICKE_STATUS_IDLE:               return "Idle";
        case WERNICKE_STATUS_PHONOLOGICAL:       return "Phonological analysis";
        case WERNICKE_STATUS_LEXICAL_ACCESS:     return "Lexical access";
        case WERNICKE_STATUS_SEMANTIC:           return "Semantic integration";
        case WERNICKE_STATUS_SYNTACTIC:          return "Syntactic parsing";
        case WERNICKE_STATUS_COMPREHENSION_READY: return "Comprehension ready";
        case WERNICKE_STATUS_ERROR:              return "Error";
        default:                                  return "Unknown status";
    }
}

bool wernicke_get_stats(const wernicke_adapter_t* adapter, wernicke_stats_t* stats)
{
    if (!adapter || !stats) return false;

    *stats = adapter->stats;
    return true;
}

bool wernicke_get_config(const wernicke_adapter_t* adapter, wernicke_config_t* config)
{
    if (!adapter || !config) return false;

    *config = adapter->config;
    return true;
}

/*=============================================================================
 * SUB-MODULE ACCESS
 *===========================================================================*/

phonological_analyzer_t* wernicke_get_phonological_analyzer(wernicke_adapter_t* adapter)
{
    if (!adapter) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adapter is NULL");

        return NULL;

    }
    return adapter->phonological;
}

lexical_access_t* wernicke_get_lexical_access(wernicke_adapter_t* adapter)
{
    if (!adapter) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adapter is NULL");

        return NULL;

    }
    return adapter->lexical;
}

semantic_integrator_t* wernicke_get_semantic_integrator(wernicke_adapter_t* adapter)
{
    if (!adapter) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adapter is NULL");

        return NULL;

    }
    return adapter->semantic;
}

syntactic_comprehension_t* wernicke_get_syntactic_comprehension(wernicke_adapter_t* adapter)
{
    if (!adapter) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adapter is NULL");

        return NULL;

    }
    return adapter->syntactic;
}

/*=============================================================================
 * BIO-ASYNC COMMUNICATION
 *===========================================================================*/

bio_module_context_t wernicke_get_bio_context(wernicke_adapter_t* adapter)
{
    if (!adapter || !adapter->bio_async_enabled) return NULL;
    return adapter->bio_ctx;
}

uint32_t wernicke_process_bio_messages(wernicke_adapter_t* adapter, uint32_t max_messages)
{
    if (!adapter || !adapter->bio_async_enabled) return 0;

    /* Phase 3 will implement full bio-async message handling */
    return 0;
}

nimcp_bio_future_t wernicke_request_word_async(
    wernicke_adapter_t* adapter,
    const phoneme_t* phonemes,
    uint32_t count)
{
    (void)count;
    if (!adapter || !phonemes) return NULL;

    /* Phase 3 will implement async word recognition */
    return NULL;
}

nimcp_error_t wernicke_broadcast_word(
    wernicke_adapter_t* adapter,
    const wernicke_word_result_t* word)
{
    if (!adapter || !word) return NIMCP_ERROR_INVALID_PARAM;

    /* Phase 3 will implement bio-async broadcast */
    return NIMCP_SUCCESS;
}

nimcp_error_t wernicke_broadcast_comprehension(
    wernicke_adapter_t* adapter,
    const wernicke_comprehension_t* comprehension)
{
    if (!adapter || !comprehension) return NIMCP_ERROR_INVALID_PARAM;

    /* Phase 3 will implement bio-async broadcast */
    return NIMCP_SUCCESS;
}

/*=============================================================================
 * KNOWLEDGE GRAPH INTEGRATION
 *===========================================================================*/

bool wernicke_register_kg(
    wernicke_adapter_t* adapter,
    brain_kg_t* kg)
{
    if (!adapter || !kg) return false;

    adapter->kg = kg;

    /* Phase 3 will implement full KG registration:
     * - Create Wernicke node
     * - Add edges to Broca, Semantic Memory, Temporal
     */

    NIMCP_LOG_INFO("wernicke", "Registered with brain knowledge graph");
    return true;
}

bool wernicke_connect_semantic_memory(
    wernicke_adapter_t* adapter,
    semantic_memory_t* semantic)
{
    if (!adapter) return false;

    adapter->semantic_memory = semantic;
    NIMCP_LOG_INFO("wernicke", "Connected to semantic memory");
    return true;
}
