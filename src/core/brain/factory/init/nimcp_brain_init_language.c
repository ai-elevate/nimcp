//=============================================================================
// nimcp_brain_init_language.c - Language Layer Initialization
//=============================================================================
/**
 * @file nimcp_brain_init_language.c
 * @brief Language Layer Initialization Implementation
 *
 * WHAT: Initialization functions for unified Language Layer
 * WHY:  Enable unified language processing capabilities in the brain
 * HOW:  Creates language orchestrator and connects all integration bridges
 *
 * @version Phase L6: Language Layer Brain Integration
 * @author NIMCP Development Team
 * @date 2026-01-05
 */

//=============================================================================
// Includes
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_language.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"

/* Language Layer includes */
#include "language/nimcp_language_orchestrator.h"
#include "language/nimcp_language_config.h"
#include "generation/nimcp_language_generator.h"
#include "generation/nimcp_tokenizer.h"
#include "generation/nimcp_embedding.h"
#include "training/nimcp_cortex_cnn.h"
#include "language/nimcp_grounded_language.h"
#include "core/brain/nimcp_brain_kg.h"
#include "snn/bridges/nimcp_snn_language_bridge.h"
#include "snn/bridges/nimcp_snn_speech_bridge.h"
#include "snn/bridges/nimcp_snn_audio_bridge.h"
#include "snn/bridges/nimcp_snn_cross_modal_align.h"
#include "snn/bridges/nimcp_snn_visual_bridge.h"
#include "snn/bridges/nimcp_snn_somatosensory_bridge.h"
#include "core/brain/bridges/nimcp_hyperledger_bridge.h"
#include "audio/synthesis/nimcp_formant_synth.h"
#include "lnn/nimcp_lnn_network.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <string.h>
#include <stdlib.h>

#define LOG_MODULE "BRAIN_INIT_LANGUAGE"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "utils/bridge/nimcp_bridge_boilerplate.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

BRIDGE_BOILERPLATE_MESH_ONLY(brain_init_language, MESH_ADAPTER_CATEGORY_SYSTEM)


/* Compatibility macro for set_error */
#ifndef set_error
#define set_error(msg) LOG_ERROR(LOG_MODULE, "%s", msg)
#endif

/* Forward declarations for bridge types (avoid header conflicts) */
struct language_perception_bridge;
struct language_cognitive_bridge;
struct language_training_bridge;
struct language_omni_bridge;
struct language_immune_bridge;
struct language_gpu_bridge;
struct language_thalamic_bridge;
struct language_substrate_bridge;
struct language_logic_bridge;

/* Bio-async forward declarations */
struct bio_router;

/* Bridge API forward declarations - stub implementations */
static struct language_perception_bridge* language_perception_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_perception_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - returns NULL, actual implementation in bridge .c files */
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_cognitive_bridge* language_cognitive_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_cognitive_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_training_bridge* language_training_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_training_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_omni_bridge* language_omni_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_omni_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_immune_bridge* language_immune_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_immune_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_gpu_bridge* language_gpu_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_gpu_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_thalamic_bridge* language_thalamic_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_thalamic_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_substrate_bridge* language_substrate_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_substrate_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

static struct language_logic_bridge* language_logic_bridge_create_stub(
    language_orchestrator_t* orchestrator,
    const language_logic_config_t* config)
{
    (void)orchestrator;
    (void)config;
    /* Stub - callers handle NULL gracefully */
    return NULL;
}

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Check prerequisites for language layer initialization
 */
static bool check_prerequisites(brain_t brain) {
    if (!brain) {
        return false;
    }

    /* Language layer can work standalone - speech/multimodal enhances it but isn't required */

    return true;
}

/**
 * @brief Create default orchestrator configuration from brain config
 */
static void create_orchestrator_config(
    brain_t brain,
    language_orchestrator_config_t* config)
{
    if (!config) return;

    /* Get defaults first */
    language_orchestrator_default_config(config);

    /* Adjust based on brain capabilities */
    config->enable_wernicke = (brain->wernicke != NULL || brain->wernicke_enabled);
    config->enable_broca = (brain->broca != NULL || brain->broca_enabled);
    config->enable_nlp_core = true;
    config->enable_multimodal = brain->config.enable_multimodal_integration;

    /* Bridge enables */
    config->enable_perception_bridge = brain->config.enable_speech_cortex;
    config->enable_cognitive_bridge = true;
    config->enable_training_bridge = true;
    config->enable_omni_bridge = (brain->omni_wernicke_bridge != NULL);
    config->enable_immune_bridge = (brain->immune_system != NULL);
    config->enable_gpu_bridge = brain->gpu_enabled;

    /* Adjust working memory size */
    if (brain->config.working_memory_capacity > 0) {
        config->phoneme_buffer_size = brain->config.working_memory_capacity * 4;
    }
}

/**
 * @brief Connect orchestrator to Wernicke adapter
 */
static bool connect_wernicke(brain_t brain) {
    if (!brain || !brain->language_layer) return true;

    if (!brain->wernicke) {
        LOG_DEBUG(LOG_MODULE, "Wernicke not available, skipping connection");
        return true;
    }

    int result = language_orchestrator_connect_wernicke(
        brain->language_layer,
        brain->wernicke
    );

    if (result != 0) {
        LOG_WARN(LOG_MODULE, "Failed to connect Wernicke to language layer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "connect_wernicke: validation failed");
        return false;
    }

    return true;
}

/**
 * @brief Connect orchestrator to Broca adapter
 */
static bool connect_broca(brain_t brain) {
    if (!brain || !brain->language_layer) return true;

    if (!brain->broca) {
        LOG_DEBUG(LOG_MODULE, "Broca not available, skipping connection");
        return true;
    }

    int result = language_orchestrator_connect_broca(
        brain->language_layer,
        brain->broca
    );

    if (result != 0) {
        LOG_WARN(LOG_MODULE, "Failed to connect Broca to language layer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "connect_broca: validation failed");
        return false;
    }

    return true;
}

/**
 * @brief Connect orchestrator to NLP network
 */
static bool connect_nlp_network(brain_t brain) {
    if (!brain || !brain->language_layer) return true;

    /* nlp_network is a struct member, not a pointer */
    int result = language_orchestrator_connect_nlp(
        brain->language_layer,
        brain->nlp_network
    );

    if (result != 0) {
        LOG_WARN(LOG_MODULE, "Failed to connect NLP network to language layer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "connect_nlp_network: validation failed");
        return false;
    }

    return true;
}

/**
 * @brief Connect orchestrator to speech cortex
 */
static bool connect_speech_cortex(brain_t brain) {
    if (!brain || !brain->language_layer) return true;

    if (!brain->speech_cortex) {
        LOG_DEBUG(LOG_MODULE, "Speech cortex not available, skipping connection");
        return true;
    }

    int result = language_orchestrator_connect_speech_cortex(
        brain->language_layer,
        brain->speech_cortex
    );

    if (result != 0) {
        LOG_WARN(LOG_MODULE, "Failed to connect speech cortex to language layer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "connect_speech_cortex: validation failed");
        return false;
    }

    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

bool nimcp_brain_factory_init_language_subsystem(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_language_subsystem: brain is NULL");
        return false;
    }

    /* Check if already initialized */
    if (brain->language_layer) {
        return true;  /* Already initialized */
    }

    /* Check prerequisites */
    if (!check_prerequisites(brain)) {
        brain->language_layer_enabled = false;
        return true;  /* Not needed, not an error */
    }

    /* Create orchestrator configuration */
    language_orchestrator_config_t config;
    create_orchestrator_config(brain, &config);

    /* Create language orchestrator */
    brain->language_layer = language_orchestrator_create(&config);
    if (!brain->language_layer) {
        set_error("Failed to create language orchestrator");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_subsystem: brain->language_layer is NULL");
        return false;
    }

    brain->language_layer_enabled = true;
    brain->last_language_update_us = 0;

    /* Connect core subsystems */
    if (!connect_wernicke(brain)) {
        LOG_WARN(LOG_MODULE, "Wernicke connection failed (non-fatal)");
    }

    if (!connect_broca(brain)) {
        LOG_WARN(LOG_MODULE, "Broca connection failed (non-fatal)");
    }

    if (!connect_nlp_network(brain)) {
        LOG_WARN(LOG_MODULE, "NLP network connection failed (non-fatal)");
    }

    if (!connect_speech_cortex(brain)) {
        LOG_WARN(LOG_MODULE, "Speech cortex connection failed (non-fatal)");
    }

    /* Initialize bridges */
    if (!nimcp_brain_factory_init_language_perception_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Perception bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_cognitive_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Cognitive bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_training_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Training bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_omni_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Omni bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_immune_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Immune bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_gpu_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "GPU bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_thalamic_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Thalamic bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_substrate_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Substrate bridge init failed (non-fatal)");
    }

    if (!nimcp_brain_factory_init_language_logic_bridge(brain)) {
        LOG_WARN(LOG_MODULE, "Logic bridge init failed (non-fatal)");
    }

    /* Connect to bio-async */
    if (!nimcp_brain_factory_connect_language_to_bio_async(brain)) {
        LOG_WARN(LOG_MODULE, "Bio-async connection failed (non-fatal)");
    }

    /* Start the orchestrator */
    if (language_orchestrator_start(brain->language_layer) != 0) {
        LOG_WARN(LOG_MODULE, "Orchestrator start failed (non-fatal)");
    }

    /* --- Initialize LNN-based language generator --- */
    {
        /* Create tokenizer if not already present */
        if (!brain->tokenizer) {
            tokenizer_config_t tok_cfg = tokenizer_default_config();
            brain->tokenizer = tokenizer_create(&tok_cfg);
            if (!brain->tokenizer) {
                LOG_WARN(LOG_MODULE, "Tokenizer creation failed (non-fatal)");
            }
        }

        if (brain->tokenizer) {
            uint32_t vocab_size = tokenizer_get_vocab_size(brain->tokenizer);
            if (vocab_size < 16) {
                /* Build a basic vocabulary from common English words */
                const char* basic_vocab[] = {
                    "the", "a", "an", "is", "are", "was", "were", "be", "been",
                    "have", "has", "had", "do", "does", "did", "will", "would",
                    "can", "could", "shall", "should", "may", "might", "must",
                    "i", "you", "he", "she", "it", "we", "they", "me", "him",
                    "her", "us", "them", "my", "your", "his", "its", "our",
                    "their", "this", "that", "these", "those", "what", "which",
                    "who", "whom", "where", "when", "why", "how", "not", "no",
                    "yes", "and", "or", "but", "if", "then", "else", "so",
                    "in", "on", "at", "to", "for", "of", "with", "by", "from",
                    "up", "out", "off", "over", "under", "again", "further",
                    "all", "each", "every", "both", "few", "more", "most",
                    "other", "some", "such", "only", "own", "same", "than",
                    "too", "very", "just", "about", "also", "back", "after",
                    "before", "between", "here", "there", "now", "then",
                    "think", "know", "see", "say", "tell", "make", "go",
                    "come", "take", "get", "give", "find", "look", "want",
                    "need", "feel", "try", "leave", "call", "keep", "let",
                    "begin", "seem", "help", "show", "hear", "play", "run",
                    "move", "live", "believe", "bring", "happen", "write",
                    "provide", "sit", "stand", "lose", "pay", "meet", "set",
                    "learn", "change", "lead", "understand", "watch", "follow",
                    "stop", "create", "speak", "read", "allow", "add", "spend",
                    "grow", "open", "walk", "win", "offer", "remember",
                    "love", "consider", "appear", "buy", "wait", "serve",
                    "die", "send", "expect", "build", "stay", "fall", "cut",
                    "reach", "kill", "remain", "suggest", "raise", "pass",
                    "good", "new", "first", "last", "long", "great", "little",
                    "right", "old", "big", "high", "small", "large", "next",
                    "early", "young", "important", "public", "bad", "real",
                    "best", "better", "sure", "free", "true", "whole", "clear",
                    ".", ",", "!", "?", ":", ";", "'", "\"", "-",
                    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "world", "life", "time", "day", "way", "man", "woman",
                    "child", "people", "thing", "place", "work", "part",
                    "case", "point", "home", "water", "room", "mother",
                    "area", "money", "story", "fact", "month", "lot",
                    "right", "study", "book", "eye", "job", "word", "side",
                    "kind", "head", "house", "service", "friend", "father",
                    "power", "hour", "game", "line", "end", "member", "law",
                    "car", "city", "community", "name", "system", "program",
                    "question", "during", "plan", "group", "number", "state",
                    "hand", "school", "night", "light", "problem", "family",
                    "brain", "neural", "network", "language", "model", "data",
                    "input", "output", "process", "signal", "pattern", "memory"
                };
                uint32_t n_words = sizeof(basic_vocab) / sizeof(basic_vocab[0]);
                for (uint32_t i = 0; i < n_words; i++) {
                    tokenizer_add_token(brain->tokenizer, basic_vocab[i]);
                }
                vocab_size = tokenizer_get_vocab_size(brain->tokenizer);
                LOG_INFO(LOG_MODULE, "Built basic vocabulary: %u tokens", vocab_size);
            }

            /* Create embedding layer */
            uint32_t embed_dim = 128;  /* Lightweight for on-device generation */
            embedding_config_t emb_cfg = embedding_default_config(vocab_size, embed_dim);
            brain->lang_embedding = embedding_create(&emb_cfg);
            if (!brain->lang_embedding) {
                LOG_WARN(LOG_MODULE, "Embedding layer creation failed (non-fatal)");
            }

            /* Create language generator */
            if (brain->lang_embedding) {
                generator_config_t gen_cfg = generator_default_config();
                gen_cfg.hidden_dim = 128;
                gen_cfg.num_lnn_neurons = 64;
                gen_cfg.max_sequence_length = 256;
                gen_cfg.strategy = GENERATION_TOP_P;
                gen_cfg.temperature = 0.8f;
                gen_cfg.top_p = 0.9f;

                brain->lang_generator = language_generator_create(
                    &gen_cfg,
                    brain->tokenizer,
                    brain->lang_embedding,
                    vocab_size,
                    embed_dim
                );

                if (brain->lang_generator) {
                    LOG_INFO(LOG_MODULE, "Language generator created: vocab=%u, embed=%u, hidden=128",
                             vocab_size, embed_dim);
                } else {
                    LOG_WARN(LOG_MODULE, "Language generator creation failed (non-fatal)");
                }
            }
        }
    }

    /* ===================================================================
     * Grounded Language System (human-like word-concept binding)
     * ===================================================================
     * Creates the grounded lexicon that maps words to semantic concepts
     * through cross-modal Hebbian binding rather than token statistics.
     */
    brain->grounded_lang = grounded_language_create(
        128, /* semantic_dim — matches brain embedding dim */
        brain->semantic_memory
    );

    if (brain->grounded_lang) {
        /* Wire cross-modal connections */
        if (brain->visual_cortex)
            grounded_language_connect_visual(brain->grounded_lang, brain->visual_cortex);
        if (brain->audio_cortex)
            grounded_language_connect_auditory(brain->grounded_lang, brain->audio_cortex);
        if (brain->speech_cortex)
            grounded_language_connect_speech(brain->grounded_lang, brain->speech_cortex);
        if (brain->cortical_column_pool)
            grounded_language_connect_columns(brain->grounded_lang, brain->cortical_column_pool);

        /* Per-network bridges — feed the comprehend semantic_vector
         * to LNN/CNN/FNO/ANN forward passes so each network sees
         * language input + contributes confidence modulation. ANN
         * predictor is left for the runtime to wire (no clean default
         * predictor available at init time). */
        if (brain->cortex_cnns[CORTEX_CNN_SPEECH]) {
            grounded_language_attach_cortex_cnn(brain->grounded_lang,
                                                  brain->cortex_cnns[CORTEX_CNN_SPEECH]);
        }
        /* LNN + FNO: brain->lnn_network and the SNN-FNO populations are
         * not the layer/processor handles the bridge expects. Leaving
         * unwired here — runtime callers (daemon, training scripts)
         * can attach via grounded_language_attach_lnn / _fno when
         * they have direct handles. */

        /* Cognitive subscriber bus — register every present cognitive
         * module so it observes NEW_WORD/GROUNDED/COMPREHENDED/
         * PRODUCED events. Each attach is NULL-tolerant (skips when
         * the module isn't created). */
        if (brain->inner_speech)
            grounded_language_attach_inner_speech(brain->grounded_lang,
                                                    brain->inner_speech);
        if (brain->imagination)
            grounded_language_attach_imagination(brain->grounded_lang,
                                                   brain->imagination);
        /* theory_of_mind is a struct-by-value typedef on brain_t — pass
         * its address so the wrapper has a stable ctx pointer. */
        grounded_language_attach_theory_of_mind(brain->grounded_lang,
                                                  &brain->theory_of_mind);
        if (brain->empathetic_response_engine)
            grounded_language_attach_empathy(brain->grounded_lang,
                                               brain->empathetic_response_engine);
        grounded_language_attach_introspection(brain->grounded_lang,
                                                 &brain->introspection);
        if (brain->analogical_transfer)
            grounded_language_attach_analogical(brain->grounded_lang,
                                                  brain->analogical_transfer);

        /* Memory-system attachments — every successful grounding event
         * fans out to working memory + episodic replay + hippocampus
         * (each independently optional). Without these, the trained
         * vocabulary is invisible to the rest of the memory hierarchy. */
        if (brain->working_memory)
            grounded_language_connect_working_memory(brain->grounded_lang,
                                                      brain->working_memory);
        if (brain->episodic_replay)
            grounded_language_connect_episodic_replay(brain->grounded_lang,
                                                       brain->episodic_replay);
        if (brain->hippocampus)
            grounded_language_connect_hippocampus(brain->grounded_lang,
                                                   brain->hippocampus);

        /* Region-adapter attachments — every newly-created lexicon entry
         * is mirrored into broca/wernicke. Single source of truth =
         * grounded_language; broca/wernicke get copies for production
         * + comprehension respectively. No-op when adapters absent. */
        if (brain->broca)
            grounded_language_connect_broca(brain->grounded_lang,
                                             brain->broca);
        if (brain->wernicke)
            grounded_language_connect_wernicke(brain->grounded_lang,
                                                brain->wernicke);

        /* Internal KG self-registration — register grounded_language as
         * a COGNITIVE node so KG analytics + downstream consumers can
         * see the lexicon subsystem exists. Mirrors the surface_geometry
         * registration pattern (init_surface_geometry.c:212-255):
         * elevate to ADMIN, find-or-add node, add edges to peer
         * subsystems, restore READ. Edges express dependencies:
         *   grounded_language → semantic_memory (DEPENDS_ON)
         *   grounded_language → snn_language_bridge (INTEGRATES_WITH)
         *   grounded_language → broca / wernicke   (PROVIDES_TO) */
        if (brain->internal_kg_enabled && brain->internal_kg) {
            uint64_t admin_token = brain->internal_kg_admin_token;
            brain_kg_set_access_level(brain->internal_kg,
                                       BRAIN_KG_ACCESS_ADMIN, admin_token);

            brain_kg_node_id_t gl_node =
                brain_kg_find_node(brain->internal_kg, "grounded_language_subsystem");
            if (gl_node == BRAIN_KG_INVALID_NODE) {
                gl_node = brain_kg_add_node(
                    brain->internal_kg,
                    "grounded_language_subsystem",
                    BRAIN_KG_NODE_COGNITIVE,
                    "Grounded language: word-concept binding lexicon + "
                    "syntactic templates + cross-modal grounding events");
            }

            if (gl_node != BRAIN_KG_INVALID_NODE) {
                brain_kg_node_id_t sm_node =
                    brain_kg_find_node(brain->internal_kg, "semantic_memory");
                if (sm_node != BRAIN_KG_INVALID_NODE) {
                    brain_kg_add_edge(brain->internal_kg, gl_node, sm_node,
                        BRAIN_KG_EDGE_DEPENDS_ON,
                        "lexicon binds words to semantic_memory concepts",
                        0.9f);
                }
                brain_kg_node_id_t snn_node =
                    brain_kg_find_node(brain->internal_kg, "snn_language_bridge");
                if (snn_node != BRAIN_KG_INVALID_NODE) {
                    brain_kg_add_edge(brain->internal_kg, gl_node, snn_node,
                        BRAIN_KG_EDGE_INTEGRATES_WITH,
                        "spike-driven dual-path produce/comprehend",
                        0.7f);
                }
                brain_kg_node_id_t broca_node =
                    brain_kg_find_node(brain->internal_kg, "broca");
                if (broca_node != BRAIN_KG_INVALID_NODE) {
                    brain_kg_add_edge(brain->internal_kg, gl_node, broca_node,
                        BRAIN_KG_EDGE_PROVIDES_TO,
                        "grounded_language mirrors lexicon entries to broca",
                        0.6f);
                }
                brain_kg_node_id_t wernicke_node =
                    brain_kg_find_node(brain->internal_kg, "wernicke");
                if (wernicke_node != BRAIN_KG_INVALID_NODE) {
                    brain_kg_add_edge(brain->internal_kg, gl_node, wernicke_node,
                        BRAIN_KG_EDGE_PROVIDES_TO,
                        "grounded_language mirrors lexicon entries to wernicke",
                        0.6f);
                }
            }

            brain_kg_set_access_level(brain->internal_kg,
                                       BRAIN_KG_ACCESS_READ, 0);
        }

        LOG_INFO(LOG_MODULE, "Grounded language system created (dim=128)");

        /* SNN Language Bridge — spike-driven word-concept binding via STDP */
        snn_lang_config_t snn_lang_cfg = snn_lang_config_default();
        brain->snn_lang_bridge = snn_language_bridge_create(&snn_lang_cfg);
        if (brain->snn_lang_bridge) {
            snn_language_bridge_connect_grounded(brain->snn_lang_bridge, brain->grounded_lang);
            if (brain->neuromodulator_system)
                snn_language_bridge_connect_neuromod(brain->snn_lang_bridge,
                                                     brain->neuromodulator_system);
            /* Wire bidirectional: GL → SNN bridge for dual-path production */
            grounded_language_connect_snn_bridge(brain->grounded_lang,
                                                  brain->snn_lang_bridge);
            LOG_INFO(LOG_MODULE, "SNN language bridge created (STDP word-concept binding)");
        } else {
            LOG_WARN(LOG_MODULE, "SNN language bridge creation failed (non-fatal)");
        }

        /* Create SNN speech bridge and wire to language bridge */
        if (brain->speech_cortex && brain->snn_lang_bridge) {
            snn_speech_config_t speech_cfg;
            snn_speech_config_default(&speech_cfg);
            brain->snn_speech_bridge = (struct snn_speech_bridge*)
                snn_speech_bridge_create(&speech_cfg, NULL, brain->speech_cortex);
            if (brain->snn_speech_bridge) {
                snn_speech_bridge_set_language_bridge(
                    (snn_speech_bridge_t*)brain->snn_speech_bridge,
                    brain->snn_lang_bridge);
                LOG_INFO(LOG_MODULE, "SNN speech bridge created and wired to language bridge");
            }
        }

        /* =============================================================
         * Formant Voice Synthesizer + LNN Prosody Network
         * =============================================================
         * Creates Athena's voice: Klatt-style formant synthesis driven
         * by an LNN network that predicts F0 contour, stress, and rate
         * from emotional state + linguistic context.
         *
         * LNN architecture (NCP wiring):
         *   Input (6): arousal, valence, word_pos, utterance_progress,
         *              phoneme_class, stress_mark
         *   Inter (16): interneurons for temporal dynamics
         *   Command (8): prosody decision layer
         *   Output (4): F0_scale, duration_scale, intensity_scale, breathiness
         */
        if (brain->snn_speech_bridge) {
            /* Create formant synthesizer with default config */
            brain->formant_synth = (struct formant_synth*)
                formant_synth_create(NULL);
            if (brain->formant_synth) {
                LOG_INFO(LOG_MODULE, "Formant voice synthesizer created");

                /* Create LNN prosody network: 6 inputs → 16 inter → 8 cmd → 4 outputs */
                brain->lnn_prosody = (struct lnn_network_s*)
                    lnn_network_create_ncp(6, 16, 8, 4);
                if (brain->lnn_prosody) {
                    lnn_network_init_weights((lnn_network_t*)brain->lnn_prosody, 42);
                    LOG_INFO(LOG_MODULE, "LNN prosody network created (6→16→8→4 NCP)");
                } else {
                    LOG_WARN(LOG_MODULE, "LNN prosody creation failed (non-fatal, using static prosody)");
                }
            } else {
                LOG_WARN(LOG_MODULE, "Formant synth creation failed (non-fatal)");
            }
        }

        /* Create SNN audio bridge for spike-based auditory processing */
        if (brain->audio_cortex) {
            snn_audio_config_t audio_cfg;
            snn_audio_config_default(&audio_cfg);
            brain->snn_audio_bridge = (struct snn_audio_bridge*)
                snn_audio_bridge_create(&audio_cfg, NULL, brain->audio_cortex);
            if (brain->snn_audio_bridge) {
                LOG_INFO(LOG_MODULE, "SNN audio bridge created for auditory spike processing");
            }
        }

        /* Create SNN visual bridge for spike-based visual processing */
        if (brain->visual_cortex) {
            snn_visual_config_t vis_cfg;
            snn_visual_config_default(&vis_cfg);
            brain->snn_visual_bridge = (struct snn_visual_bridge*)
                snn_visual_bridge_create(&vis_cfg, NULL, brain->visual_cortex);
            if (brain->snn_visual_bridge) {
                LOG_INFO(LOG_MODULE, "SNN visual bridge created for visual spike processing");
            }
        }

        /* Create SNN somatosensory bridge for touch/proprioception spike encoding */
        if (brain->somatosensory) {
            snn_somatosensory_config_t soma_cfg;
            snn_somatosensory_config_default(&soma_cfg);
            brain->snn_somatosensory_bridge = (struct snn_somatosensory_bridge*)
                snn_somatosensory_bridge_create(&soma_cfg, NULL);
            if (brain->snn_somatosensory_bridge) {
                LOG_INFO(LOG_MODULE, "SNN somatosensory bridge created for touch/proprioception spike encoding");
            }
        }

        /* Create cross-modal temporal alignment (latency compensation) */
        cross_modal_align_config_t cma_cfg;
        cross_modal_align_config_default(&cma_cfg);
        brain->cross_modal_aligner = (struct cross_modal_align*)
            cross_modal_align_create(&cma_cfg);
        if (brain->cross_modal_aligner) {
            cross_modal_align_t* cma = (cross_modal_align_t*)brain->cross_modal_aligner;
            /* register_modality(aligner, name, inherent_latency, processing_latency, stdp_tau) */
            cross_modal_align_register_modality(cma, "visual", 60.0f, 15.0f, 30.0f);
            cross_modal_align_register_modality(cma, "auditory", 20.0f, 10.0f, 20.0f);
            cross_modal_align_register_modality(cma, "somatosensory", 40.0f, 12.0f, 25.0f);
            cross_modal_align_register_modality(cma, "speech", 50.0f, 15.0f, 50.0f);
            LOG_INFO(LOG_MODULE, "Cross-modal aligner created (4 modalities registered)");
        }
    } else {
        LOG_WARN(LOG_MODULE, "Grounded language system creation failed (non-fatal)");
    }

    /* Hyperledger Bridge — EOV training + consensus-gated inference + audit */
    hyperledger_bridge_config_t hl_cfg = hyperledger_bridge_default_config();
    brain->hyperledger_bridge = hyperledger_bridge_create(&hl_cfg);
    if (brain->hyperledger_bridge) {
        /* Wire to collective cognition if available */
        if (brain->collective_cognition) {
            hyperledger_bridge_connect_collective(brain->hyperledger_bridge,
                                                   brain->collective_cognition);
        }
        brain->hyperledger_enabled = true;
        LOG_INFO(LOG_MODULE, "Hyperledger bridge created (EOV training + audit)");
    }

    LOG_INFO(LOG_MODULE, "Language layer initialized successfully");
    return true;
}

bool nimcp_brain_factory_init_language_perception_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_perception_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Get default perception config */
    language_perception_config_t config;
    language_perception_default_config(&config);

    /* Adjust based on brain capabilities */
    config.enable_speech_cortex = (brain->speech_cortex != NULL);
    config.enable_audio_cortex = (brain->audio_cortex != NULL);
    config.enable_visual_cortex = (brain->visual_cortex != NULL);
    config.enable_omni_sensory = brain->config.enable_multimodal_integration;

    /* Create perception bridge (stub for now) */
    brain->language_perception_bridge = language_perception_bridge_create_stub(
        brain->language_layer,
        &config
    );

    /* Note: Stub returns NULL, but that's okay for now */
    if (!brain->language_perception_bridge) {
        LOG_DEBUG(LOG_MODULE, "Perception bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_cognitive_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_cognitive_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Get default cognitive config */
    language_cognitive_config_t config;
    language_cognitive_default_config(&config);

    /* Adjust working memory based on brain config */
    if (brain->config.working_memory_capacity > 0) {
        config.phonological_buffer_size = brain->config.working_memory_capacity;
    }

    /* Create cognitive bridge (stub for now) */
    brain->language_cognitive_bridge = language_cognitive_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_cognitive_bridge) {
        LOG_DEBUG(LOG_MODULE, "Cognitive bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_training_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_training_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Get default training config */
    language_training_config_t config;
    language_training_default_config(&config);

    /* Create training bridge (stub for now) */
    brain->language_training_bridge = language_training_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_training_bridge) {
        LOG_DEBUG(LOG_MODULE, "Training bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_omni_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_omni_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    if (!brain->omni_wernicke_bridge) {
        return true;  /* Not enabled */
    }

    /* Get default omni config */
    language_omni_config_t config;
    language_omni_default_config(&config);

    /* Create omni bridge (stub for now) */
    brain->language_omni_bridge = language_omni_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_omni_bridge) {
        LOG_DEBUG(LOG_MODULE, "Omni bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_immune_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_immune_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    if (!brain->immune_system) {
        return true;  /* Not enabled */
    }

    /* Get default immune config */
    language_immune_config_t config;
    language_immune_default_config(&config);

    /* Create immune bridge (stub for now) */
    brain->language_immune_bridge = language_immune_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_immune_bridge) {
        LOG_DEBUG(LOG_MODULE, "Immune bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_gpu_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_gpu_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    if (!brain->gpu_enabled) {
        return true;  /* Not enabled */
    }

    /* Get default GPU config */
    language_gpu_config_t config;
    language_gpu_default_config(&config);
    config.enable_gpu = true;

    /* Set device ID from brain config */
    config.device_id = brain->config.gpu_device_id;

    /* Create GPU bridge (stub for now) */
    brain->language_gpu_bridge = language_gpu_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_gpu_bridge) {
        LOG_DEBUG(LOG_MODULE, "GPU bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_thalamic_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_thalamic_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Thalamic bridge doesn't require a specific prerequisite - always try to init */

    /* Get default thalamic config */
    language_thalamic_config_t config;
    language_thalamic_default_config(&config);

    /* Create thalamic bridge (stub for now) */
    brain->language_thalamic_bridge = language_thalamic_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_thalamic_bridge) {
        LOG_DEBUG(LOG_MODULE, "Thalamic bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_substrate_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_substrate_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Substrate bridge doesn't require a specific prerequisite - always try to init */

    /* Get default substrate config */
    language_substrate_config_t config;
    language_substrate_default_config(&config);

    /* Create substrate bridge (stub for now) */
    brain->language_substrate_bridge = language_substrate_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_substrate_bridge) {
        LOG_DEBUG(LOG_MODULE, "Substrate bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_init_language_logic_bridge(brain_t brain) {
    if (!brain || !brain->language_layer) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_factory_init_language_logic_bridge: required parameter is NULL (brain, brain->language_layer)");
        return false;
    }

    /* Logic bridge doesn't require a specific prerequisite - always try to init */

    /* Get default logic config */
    language_logic_config_t config;
    language_logic_default_config(&config);

    /* Create logic bridge (stub for now) */
    brain->language_logic_bridge = language_logic_bridge_create_stub(
        brain->language_layer,
        &config
    );

    if (!brain->language_logic_bridge) {
        LOG_DEBUG(LOG_MODULE, "Logic bridge not yet implemented");
    }

    return true;
}

bool nimcp_brain_factory_connect_language_to_bio_async(brain_t brain) {
    if (!brain || !brain->language_layer) {
        return true;  /* Non-fatal */
    }

    if (!brain->bio_async_enabled || !brain->bio_async_ctx) {
        return true;  /* Bio-async not enabled */
    }

    /* Registration handled by orchestrator start */
    LOG_DEBUG(LOG_MODULE, "Language layer bio-async connection deferred to orchestrator");
    return true;
}

void nimcp_brain_factory_destroy_language_subsystem(brain_t brain) {
    if (!brain) {
        return;
    }

    /* Stop orchestrator */
    if (brain->language_layer) {
        language_orchestrator_stop(brain->language_layer);
    }

    /* Clear bridge pointers (stubs, no actual destroy needed yet) */
    brain->language_logic_bridge = NULL;
    brain->language_substrate_bridge = NULL;
    brain->language_thalamic_bridge = NULL;
    brain->language_gpu_bridge = NULL;
    brain->language_immune_bridge = NULL;
    brain->language_omni_bridge = NULL;
    brain->language_training_bridge = NULL;
    brain->language_cognitive_bridge = NULL;
    brain->language_perception_bridge = NULL;

    /* Destroy formant synth and LNN prosody */
    if (brain->lnn_prosody) {
        lnn_network_destroy((lnn_network_t*)brain->lnn_prosody);
        brain->lnn_prosody = NULL;
    }
    if (brain->formant_synth) {
        formant_synth_destroy((formant_synth_t*)brain->formant_synth);
        brain->formant_synth = NULL;
    }

    /* Destroy orchestrator */
    if (brain->language_layer) {
        language_orchestrator_destroy(brain->language_layer);
        brain->language_layer = NULL;
    }

    brain->language_layer_enabled = false;

    LOG_INFO(LOG_MODULE, "Language layer destroyed");
}

bool nimcp_brain_factory_language_is_initialized(brain_t brain) {
    if (!brain) {
        return false;
    }
    return brain->language_layer_enabled && brain->language_layer != NULL;
}

struct language_orchestrator* nimcp_brain_factory_get_language_orchestrator(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;
    }
    return brain->language_layer;
}
