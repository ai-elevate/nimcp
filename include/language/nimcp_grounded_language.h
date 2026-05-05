/**
 * @file nimcp_grounded_language.h
 * @brief Grounded Language System - Human-like language acquisition and production
 *
 * WHAT: Language grounded in multimodal semantic concepts, not token statistics
 * WHY:  Humans learn language from ~30M words (not trillions) by grounding words
 *       in sensory experience, emotion, and social interaction. This system models
 *       that process: words become pointers into the brain's shared semantic space.
 * HOW:  Bidirectional word-concept bindings with Hebbian strengthening, cross-modal
 *       grounding, compositional structure via cortical columns, and production
 *       through semantic navigation rather than autoregressive token prediction.
 *
 * BIOLOGICAL BASIS:
 * - Embodied cognition (Barsalou, 1999): Concepts are grounded in sensorimotor
 *   experience, not amodal symbols
 * - Statistical learning (Saffran et al., 1996): Infants extract patterns from
 *   sparse input via distributional regularities
 * - Cross-situational word learning (Smith & Yu, 2008): Word meanings inferred
 *   from co-occurrence across situations
 * - Fast mapping (Carey & Bartlett, 1978): One-shot word-concept binding in
 *   children (the "gavagai" problem)
 * - Semantic hubs (Patterson et al., 2007): Anterior temporal lobe as convergence
 *   zone for multimodal concept representations
 *
 * ARCHITECTURE:
 *
 *   COMPREHENSION (Wernicke's pathway):
 *   text --> tokenize --> word_form lookup --> concept activation (semantic memory)
 *        --> spreading activation --> context --> understanding
 *
 *   GROUNDING (Cross-modal binding):
 *   word_form + sensory_input --> hebbian_bind() --> strengthened association
 *   Modalities: visual, auditory, motor, emotional, spatial
 *
 *   PRODUCTION (Broca's pathway):
 *   semantic_intent --> activate concepts --> select words by association strength
 *        --> syntactic ordering (SVO templates + learned patterns) --> text
 *
 *   CREATIVE PRODUCTION:
 *   conceptual_blend(A, B) --> novel representation --> describe()
 *   semantic_chain(seed, steps) --> narrative walk --> text
 *   analogy(A:B :: C:?) --> relational transfer --> text
 *
 * @version 1.0.0
 * @date 2026-03-07
 */

#ifndef NIMCP_GROUNDED_LANGUAGE_H
#define NIMCP_GROUNDED_LANGUAGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * Constants
 *===========================================================================*/

/** Maximum vocabulary size for the grounded lexicon */
#define GL_MAX_VOCAB              16384

/** Maximum word form length in bytes */
#define GL_MAX_WORD_LEN           64

/** Semantic vector dimension (matches brain's internal representation) */
#define GL_SEMANTIC_DIM           128

/** Maximum modalities for cross-modal binding */
#define GL_MAX_MODALITIES         6

/** Maximum concepts activated during comprehension */
#define GL_MAX_ACTIVE_CONCEPTS    64

/** Maximum words in production buffer */
#define GL_MAX_PRODUCTION_WORDS   256

/** Maximum syntactic templates */
#define GL_MAX_TEMPLATES          128

/** Maximum words per template slot */
#define GL_MAX_TEMPLATE_SLOTS     8

/** Hebbian learning rate for word-concept binding */
#define GL_HEBBIAN_LR_DEFAULT     0.1f

/** Association decay rate per timestep */
#define GL_DECAY_RATE_DEFAULT     0.001f

/** Minimum association strength before pruning */
#define GL_ASSOC_PRUNE_THRESHOLD  0.01f

/** Context window for distributional learning */
#define GL_CONTEXT_WINDOW         7

/** Fast mapping threshold (one-shot if above this) */
#define GL_FAST_MAP_THRESHOLD     0.8f

/** Magic number for serialization */
#define GL_MAGIC                  0x474C4E47  /* "GLNG" */

/** Maximum dialect tag length (e.g., "en-US", "fr-CA", "zh-CN", "es-MX"). */
#define GL_MAX_DIALECT_LEN        16

/*=============================================================================
 * Enumerations
 *===========================================================================*/

/**
 * @brief Sensory modality for grounding
 */
typedef enum {
    GL_MODALITY_VISUAL = 0,     /**< Visual cortex features */
    GL_MODALITY_AUDITORY,       /**< Auditory cortex features */
    GL_MODALITY_MOTOR,          /**< Motor/action representations */
    GL_MODALITY_EMOTIONAL,      /**< Emotional valence/arousal */
    GL_MODALITY_SPATIAL,        /**< Spatial/proprioceptive */
    GL_MODALITY_LINGUISTIC,     /**< Cross-linguistic (word-to-word) */
    GL_MODALITY_COUNT
} gl_modality_t;

/**
 * @brief Word class (part of speech, learned not assigned)
 */
typedef enum {
    GL_CLASS_UNKNOWN = 0,
    GL_CLASS_NOUN,              /**< Object/entity words */
    GL_CLASS_VERB,              /**< Action/state words */
    GL_CLASS_ADJECTIVE,         /**< Property words */
    GL_CLASS_ADVERB,            /**< Manner words */
    GL_CLASS_FUNCTION,          /**< Determiners, prepositions, conjunctions */
    GL_CLASS_PRONOUN,           /**< Reference words */
    GL_CLASS_COUNT
} gl_word_class_t;

/**
 * @brief Production mode
 */
typedef enum {
    GL_PRODUCE_DESCRIBE = 0,    /**< Describe a concept/scene */
    GL_PRODUCE_NARRATE,         /**< Generate narrative from concept chain */
    GL_PRODUCE_RESPOND,         /**< Respond to comprehended input */
    GL_PRODUCE_ELABORATE,       /**< Expand on a concept with detail */
    GL_PRODUCE_ANALOGIZE,       /**< Generate analogy-based text */
    GL_PRODUCE_CREATE           /**< Free creative generation */
} gl_production_mode_t;

/**
 * @brief Syntactic pattern type
 */
typedef enum {
    GL_PATTERN_SVO = 0,         /**< Subject-Verb-Object */
    GL_PATTERN_SV,              /**< Subject-Verb (intransitive) */
    GL_PATTERN_SVA,             /**< Subject-Verb-Adjunct */
    GL_PATTERN_SVOO,            /**< Subject-Verb-Object-Object (ditransitive) */
    GL_PATTERN_NP,              /**< Noun phrase (Det-Adj-Noun) */
    GL_PATTERN_COPULA,          /**< Subject-is-Predicate */
    GL_PATTERN_CONDITIONAL,     /**< If-then pattern */
    GL_PATTERN_COMPARATIVE,     /**< More/less-than pattern */
    GL_PATTERN_LEARNED,         /**< Extracted from input (not built-in) */
    GL_PATTERN_COUNT
} gl_syntactic_pattern_t;

/*=============================================================================
 * Data Structures
 *===========================================================================*/

/**
 * @brief A single word-concept association (grounded binding)
 *
 * WHAT: Maps a word form to a semantic concept with per-modality strength
 * WHY:  Models how "red" activates visual redness, "crash" activates auditory
 *       and emotional features, etc.
 */
typedef struct {
    uint64_t concept_id;                        /**< Semantic memory concept ID */
    float    strength;                          /**< Overall association strength [0,1] */
    float    modality_strength[GL_MODALITY_COUNT]; /**< Per-modality grounding */
    uint32_t exposure_count;                    /**< Times this binding was reinforced */
    uint64_t last_activation_ms;                /**< Last time this binding fired */
    float    confidence;                        /**< How confident the mapping is */
} gl_word_binding_t;

/**
 * @brief A word in the grounded lexicon
 *
 * WHAT: Word form with all its concept bindings and distributional statistics
 * WHY:  Words are polysemous (bank = river bank, money bank) -- one word maps
 *       to multiple concepts with different strengths
 */
typedef struct {
    char     form[GL_MAX_WORD_LEN];             /**< Surface form (lowercase) */
    uint32_t form_hash;                         /**< Hash for fast lookup */

    /* Concept bindings (one word -> many concepts) */
    gl_word_binding_t* bindings;                /**< Array of concept associations */
    uint32_t           binding_count;           /**< Current number of bindings */
    uint32_t           binding_capacity;        /**< Allocated capacity */

    /* Distributional statistics */
    uint32_t           frequency;               /**< Total exposure count */
    gl_word_class_t    learned_class;           /**< Inferred part of speech */
    float              class_confidence;         /**< Confidence in class assignment */

    /* Contextual co-occurrence (distributional semantics) */
    float*             context_vector;          /**< Distributional embedding [GL_SEMANTIC_DIM] */
    bool               context_initialized;     /**< Has been seen in context */

    /* Emotional grounding */
    float              valence;                 /**< Emotional valence [-1, +1] */
    float              arousal;                 /**< Emotional arousal [0, 1] */

} gl_lexicon_entry_t;

/**
 * @brief Syntactic template (learned or built-in)
 *
 * WHAT: A sentence pattern with typed slots
 * WHY:  Children learn syntax by extracting patterns, not memorizing rules.
 *       "The [NOUN] [VERB]s the [NOUN]" is learned from examples.
 */
typedef struct {
    gl_syntactic_pattern_t type;                /**< Pattern type */
    gl_word_class_t        slots[GL_MAX_TEMPLATE_SLOTS]; /**< Expected word class per slot */
    uint32_t               slot_count;          /**< Number of slots */
    float                  frequency;           /**< How often this pattern seen */
    float                  confidence;          /**< How reliable this pattern is */
} gl_template_t;

/**
 * @brief Comprehension result
 */
typedef struct {
    uint64_t* activated_concepts;               /**< Concept IDs activated */
    float*    activation_levels;                /**< Activation per concept */
    uint32_t  concept_count;                    /**< Number of concepts activated */
    float*    semantic_vector;                  /**< Integrated meaning [GL_SEMANTIC_DIM] */
    float     comprehension_confidence;         /**< Overall comprehension score */
    float     novelty;                          /**< How novel the input is */
} gl_comprehension_result_t;

/**
 * @brief Production result
 */
typedef struct {
    char*    text;                              /**< Generated text (heap, NUL-terminated) */
    uint32_t word_count;                        /**< Number of words produced */
    float    fluency;                           /**< Self-assessed fluency [0,1] */
    float    relevance;                         /**< Relevance to intent [0,1] */
    float    creativity;                        /**< Novelty of word combinations [0,1] */
    float*   semantic_vector;                   /**< Semantic meaning of output [GL_SEMANTIC_DIM] */
} gl_production_result_t;

/**
 * @brief Grounding event — a cross-modal learning experience
 *
 * WHAT: Represents a moment of grounded learning (e.g., seeing a cat while hearing "cat")
 * WHY:  This is how humans learn words — through situated experience
 */
typedef struct {
    const char*  word;                          /**< Word form being learned */
    gl_modality_t modality;                     /**< Which sensory modality */
    const float* sensory_features;              /**< Feature vector from that modality */
    uint32_t     feature_dim;                   /**< Dimension of feature vector */
    float        emotional_valence;             /**< Emotional context [-1, +1] */
    float        emotional_arousal;             /**< Arousal context [0, 1] */
    float        attention;                     /**< Attentional weight [0, 1] */
    const char*  context_sentence;              /**< Sentence context (optional) */
    /* #7 Negative grounding (contrast pair). When true, this is an
     * "anti-example": the word does NOT mean the bound concept under
     * these features. Engages negative-Hebbian: existing word↔concept
     * binding for the matching modality is weakened by attention*lr,
     * no new entry is created, frequency is NOT incremented. Fires
     * GROUNDED with negative confidence to signal anti-learning. */
    bool         negative;
} gl_grounding_event_t;

/**
 * @brief Statistics for the grounded language system
 */
typedef struct {
    uint32_t vocab_size;                        /**< Words in lexicon */
    uint32_t total_bindings;                    /**< Word-concept associations */
    uint32_t total_groundings;                  /**< Grounding events processed */
    uint32_t total_comprehensions;              /**< Comprehension operations */
    uint32_t total_productions;                 /**< Production operations */
    uint32_t templates_learned;                 /**< Syntactic patterns discovered */
    float    avg_binding_strength;              /**< Mean association strength */
    float    avg_comprehension_confidence;      /**< Mean comprehension score */
    float    vocabulary_growth_rate;            /**< New words per 1000 events */
    /* Forgetting-curve telemetry (#15) — bumped by sleep_consolidate
     * when a lexicon entry is decayed below its retention threshold.
     * Ring of last-24h counts (24 buckets of 1h each); _last_24h is
     * the rolling sum, _all_time is the cumulative. */
    uint32_t entries_decayed_last_24h;
    uint64_t entries_decayed_all_time;
    /* Negative-grounding telemetry (#7) — counts ground() calls that
     * arrived with event->negative=true and successfully weakened
     * an existing binding. */
    uint64_t total_negative_groundings;
    /* Active-learning telemetry (#10) — counts NEEDS_GROUNDING events
     * fired by comprehend on low-confidence input. */
    uint64_t total_needs_grounding;
} gl_stats_t;

/**
 * @brief Grounded language system (opaque handle)
 */
typedef struct grounded_language grounded_language_t;

/*=============================================================================
 * Lifecycle
 *===========================================================================*/

/**
 * @brief Create the grounded language system
 *
 * WHAT: Initialize the grounded lexicon, syntactic templates, and cross-modal binder
 * WHY:  Prepare the system for human-like language acquisition
 * HOW:  Allocate lexicon, seed with basic function words, create built-in templates
 *
 * @param semantic_dim     Dimension of semantic vectors (0 = GL_SEMANTIC_DIM)
 * @param semantic_memory  Pointer to semantic_memory_system_t (can be NULL, wired later)
 * @return System handle, or NULL on failure
 */
grounded_language_t* grounded_language_create(
    uint32_t semantic_dim,
    void* semantic_memory);

/**
 * @brief Destroy the grounded language system
 * @param gl System to destroy (NULL-safe)
 */
void grounded_language_destroy(grounded_language_t* gl);

/**
 * @brief Wire semantic memory after creation
 * @param gl System handle
 * @param semantic_memory Pointer to semantic_memory_system_t
 */
void grounded_language_set_semantic_memory(
    grounded_language_t* gl,
    void* semantic_memory);

/*=============================================================================
 * Comprehension (Wernicke's pathway)
 *===========================================================================*/

/**
 * @brief Comprehend text by activating grounded concepts
 *
 * WHAT: Parse text into words, activate concept bindings, integrate meaning
 * WHY:  Understanding = activating the right concepts in semantic memory
 * HOW:  For each word: lookup in lexicon -> activate bound concepts ->
 *       spreading activation -> context integration -> semantic vector
 *
 * @param gl       System handle
 * @param text     Input text to comprehend
 * @param result   Output comprehension result (caller allocates, system fills)
 * @return 0 on success, -1 on error
 */
int grounded_language_comprehend(
    grounded_language_t* gl,
    const char* text,
    gl_comprehension_result_t* result);

/**
 * @brief Free comprehension result internals
 * @param result Result to clean up (struct itself not freed)
 */
void gl_comprehension_result_cleanup(gl_comprehension_result_t* result);

/*=============================================================================
 * Grounding (Cross-modal binding)
 *===========================================================================*/

/**
 * @brief Ground a word in sensory experience
 *
 * WHAT: Create or strengthen a word-concept binding through cross-modal association
 * WHY:  This is how humans learn words — by experiencing the referent
 * HOW:  1. Find or create concept from sensory features (semantic memory)
 *       2. Find or create lexicon entry for word
 *       3. Hebbian binding: strengthen word->concept association
 *       4. Update modality-specific grounding weights
 *
 * @param gl     System handle
 * @param event  Grounding event (word + sensory input + context)
 * @return 0 on success, -1 on error
 */
int grounded_language_ground(
    grounded_language_t* gl,
    const gl_grounding_event_t* event);

/**
 * @brief Learn word meaning from text context (distributional learning)
 *
 * WHAT: Update word representations from co-occurrence patterns
 * WHY:  "You shall know a word by the company it keeps" (Firth, 1957)
 * HOW:  Sliding window over text, update context vectors via co-occurrence,
 *       infer word class from distributional patterns
 *
 * @param gl    System handle
 * @param text  Text to learn from
 * @return Number of word bindings updated, or -1 on error
 */
int grounded_language_learn_from_text(
    grounded_language_t* gl,
    const char* text);

/**
 * @brief Teach a word-concept pair directly (fast mapping)
 *
 * WHAT: One-shot word learning ("This is a gavagai")
 * WHY:  Children can learn a word from a single labeled example
 * HOW:  Create strong binding between word and concept immediately
 *
 * @param gl          System handle
 * @param word        Word form to learn
 * @param concept_features  Semantic features of the concept
 * @param feature_dim       Dimension of features
 * @param category    Concept category (CONCEPT_OBJECT, etc.)
 * @return Concept ID of bound concept, or 0 on error
 */
uint64_t grounded_language_fast_map(
    grounded_language_t* gl,
    const char* word,
    const float* concept_features,
    uint32_t feature_dim,
    uint32_t category);

/*=============================================================================
 * Production (Broca's pathway)
 *===========================================================================*/

/**
 * @brief Produce text from a semantic intent
 *
 * WHAT: Convert internal semantic representation to natural language
 * WHY:  Expression = finding words that point to the intended concepts
 * HOW:  1. Activate concepts closest to semantic intent
 *       2. For each concept, find highest-strength word bindings
 *       3. Select syntactic template matching word classes
 *       4. Fill template slots, apply morphology
 *       5. Output text
 *
 * @param gl           System handle
 * @param intent       Semantic vector representing intended meaning [semantic_dim]
 * @param intent_dim   Dimension of intent vector
 * @param mode         Production mode (describe, narrate, respond, etc.)
 * @param result       Output production result
 * @return 0 on success, -1 on error
 */
int grounded_language_produce(
    grounded_language_t* gl,
    const float* intent,
    uint32_t intent_dim,
    gl_production_mode_t mode,
    gl_production_result_t* result);

/**
 * @brief Produce text describing a specific concept
 *
 * WHAT: Generate a description of a concept using grounded vocabulary
 * WHY:  "What is X?" — describe by activating related concepts and verbalizing
 * HOW:  Activate concept, spread to related concepts, select descriptive words
 *
 * @param gl          System handle
 * @param concept_id  Concept to describe
 * @param result      Output production result
 * @return 0 on success, -1 on error
 */
int grounded_language_describe_concept(
    grounded_language_t* gl,
    uint64_t concept_id,
    gl_production_result_t* result);

/**
 * @brief Produce text by blending two concepts (creative generation)
 *
 * WHAT: Conceptual blending — combine two concepts to create novel expression
 * WHY:  Creativity = novel combinations. "Time is money" blends TIME + MONEY.
 * HOW:  Interpolate concept features, find words for blended space, generate text
 *
 * @param gl          System handle
 * @param concept_a   First concept ID (or 0 to use vector_a)
 * @param concept_b   Second concept ID (or 0 to use vector_b)
 * @param vector_a    Feature vector for concept A (used if concept_a == 0)
 * @param vector_b    Feature vector for concept B (used if concept_b == 0)
 * @param vec_dim     Dimension of feature vectors
 * @param blend_ratio Interpolation ratio [0=all A, 1=all B, 0.5=even blend]
 * @param result      Output production result
 * @return 0 on success, -1 on error
 */
int grounded_language_blend(
    grounded_language_t* gl,
    uint64_t concept_a,
    uint64_t concept_b,
    const float* vector_a,
    const float* vector_b,
    uint32_t vec_dim,
    float blend_ratio,
    gl_production_result_t* result);

/**
 * @brief Generate narrative by walking through semantic space
 *
 * WHAT: Produce multi-sentence text by chaining related concepts
 * WHY:  Stories and essays work by traversing a path through meaning space
 * HOW:  Start from seed concept, follow strongest relations, produce sentence
 *       per concept cluster, maintain coherence through context vector
 *
 * @param gl            System handle
 * @param seed_concept  Starting concept ID
 * @param num_sentences Target number of sentences
 * @param creativity    Exploration vs exploitation [0=safe, 1=wild]
 * @param result        Output production result
 * @return 0 on success, -1 on error
 */
int grounded_language_narrate(
    grounded_language_t* gl,
    uint64_t seed_concept,
    uint32_t num_sentences,
    float creativity,
    gl_production_result_t* result);

/**
 * @brief Free production result internals
 * @param result Result to clean up (struct itself not freed)
 */
void gl_production_result_cleanup(gl_production_result_t* result);

/*=============================================================================
 * Conversation (Comprehension + Production loop)
 *===========================================================================*/

/**
 * @brief Process input and generate a response (full conversation turn)
 *
 * WHAT: Comprehend input text, then produce a relevant response
 * WHY:  This is the core conversational loop
 * HOW:  comprehend(input) -> integrate with context -> produce(response)
 *
 * @param gl           System handle
 * @param input_text   What was said to the brain
 * @param response     Output buffer for response text
 * @param response_max Maximum response buffer size
 * @param confidence   Output: response confidence [0,1]
 * @return Number of characters written, or -1 on error
 */
int grounded_language_respond(
    grounded_language_t* gl,
    const char* input_text,
    char* response,
    uint32_t response_max,
    float* confidence);

/*=============================================================================
 * Training / Learning
 *===========================================================================*/

/**
 * @brief Learn from a sentence pair (teacher-guided)
 *
 * WHAT: Learn associations from input-target pairs
 * WHY:  Social learning — a teacher provides correct responses
 * HOW:  Comprehend input, comprehend target, strengthen associations between
 *       input concepts and target words (and vice versa)
 *
 * @param gl           System handle
 * @param input_text   Input/stimulus text
 * @param target_text  Expected/correct response
 * @param learning_rate Hebbian learning rate (0 = default)
 * @return Learning loss (lower = better), or -1.0 on error
 */
float grounded_language_learn_pair(
    grounded_language_t* gl,
    const char* input_text,
    const char* target_text,
    float learning_rate);

/**
 * @brief Learn syntactic patterns from text
 *
 * WHAT: Extract sentence structure templates from observed text
 * WHY:  Children learn grammar from examples, not rules
 * HOW:  Parse text into word classes, identify recurring patterns,
 *       create/strengthen template entries
 *
 * @param gl    System handle
 * @param text  Text to extract patterns from
 * @return Number of new patterns learned, or -1 on error
 */
int grounded_language_learn_syntax(
    grounded_language_t* gl,
    const char* text);

/*=============================================================================
 * Cross-modal Integration
 *===========================================================================*/

/**
 * @brief Connect to visual cortex for visual grounding
 * @param gl       System handle
 * @param vis_ctx  Visual cortex context (opaque pointer)
 */
void grounded_language_connect_visual(
    grounded_language_t* gl,
    void* vis_ctx);

/**
 * @brief Connect to auditory cortex for auditory grounding
 * @param gl        System handle
 * @param aud_ctx   Auditory cortex context (opaque pointer)
 */
void grounded_language_connect_auditory(
    grounded_language_t* gl,
    void* aud_ctx);

/**
 * @brief Connect to speech cortex for speech grounding
 * @param gl          System handle
 * @param speech_ctx  Speech cortex context (opaque pointer)
 */
void grounded_language_connect_speech(
    grounded_language_t* gl,
    void* speech_ctx);

/**
 * @brief Connect to cortical column pool for hierarchical feature extraction
 * @param gl        System handle
 * @param col_pool  Cortical column pool (opaque pointer)
 */
void grounded_language_connect_columns(
    grounded_language_t* gl,
    void* col_pool);

/**
 * @brief Connect to emotional system for affective grounding
 * @param gl        System handle
 * @param emo_ctx   Emotional context (opaque pointer, e.g., metabolic state)
 */
void grounded_language_connect_emotional(
    grounded_language_t* gl,
    void* emo_ctx);

/** Connect SNN language bridge for spike-driven dual-path production/comprehension */
struct snn_language_bridge;
void grounded_language_connect_snn_bridge(
    grounded_language_t* gl,
    struct snn_language_bridge* bridge);

/**
 * @brief Connect to working memory — active words get pushed to the
 *        Miller-7±2 buffer with attention-derived salience, so the rest
 *        of the cognitive stack can reason over recently-grounded words.
 * @param gl  System handle
 * @param wm  working_memory_t* (opaque here to avoid include surface)
 */
void grounded_language_connect_working_memory(
    grounded_language_t* gl,
    void* wm);

/**
 * @brief Connect to episodic replay buffer — every grounding event gets
 *        recorded as a replay-eligible experience. During sleep the
 *        replay system re-presents high-importance grounding events
 *        through brain.learn_vector to consolidate the binding.
 * @param gl      System handle
 * @param replay  nimcp_episodic_replay_t* (opaque)
 */
void grounded_language_connect_episodic_replay(
    grounded_language_t* gl,
    void* replay);

/**
 * @brief Connect to the hippocampus adapter — high-attention grounding
 *        events train word→concept associations in the hippocampus,
 *        producing genuine episodic encoding of when a word was learned.
 * @param gl           System handle
 * @param hippocampus  hippocampus_adapter_t* (opaque)
 */
void grounded_language_connect_hippocampus(
    grounded_language_t* gl,
    void* hippocampus);

/**
 * @brief Connect Broca's area adapter — every new lexicon entry is
 *        mirrored into the production lexicon so broca can articulate
 *        words that grounded_language has learned. SOLID single-source-
 *        of-truth: grounded_language owns the canonical lexicon; broca
 *        receives copies for production-side phoneme planning.
 * @param gl     System handle
 * @param broca  broca_adapter_t* (opaque)
 */
void grounded_language_connect_broca(
    grounded_language_t* gl,
    void* broca);

/**
 * @brief Connect Wernicke's area adapter — every new lexicon entry is
 *        mirrored into the comprehension lexicon so wernicke can
 *        recognize words that grounded_language has learned. Also
 *        provides the entry point for routing wernicke comprehension
 *        results back into grounded_language as auditory grounding
 *        events (see grounded_language_ingest_wernicke_result).
 * @param gl        System handle
 * @param wernicke  wernicke_adapter_t* (opaque)
 */
void grounded_language_connect_wernicke(
    grounded_language_t* gl,
    void* wernicke);

/**
 * @brief Ingest a wernicke comprehension result as auditory grounding
 *        events. For each recognized word in the result, a
 *        gl_grounding_event_t is constructed with modality=AUDITORY,
 *        attention=word.confidence, and pushed through the standard
 *        grounding pipeline. Closes the audio→language loop.
 * @param gl              System handle
 * @param comp_result     wernicke_comprehension_t* (opaque cast inside)
 * @param audio_features  Sensory feature vector that produced the
 *                        comprehension (modality=audio); NULL skips.
 * @param feature_dim     Length of audio_features
 * @return Number of grounding events successfully recorded, or -1.
 */
int grounded_language_ingest_wernicke_result(
    grounded_language_t* gl,
    const void* comp_result,
    const float* audio_features,
    uint32_t feature_dim);

/**
 * @brief Sleep-state consolidation pass over the lexicon.
 *
 * WHAT: Apply consolidation effects matching the current sleep stage.
 *       NREM stages strengthen frequent / high-confidence bindings and
 *       decay rarely-fired ones (memory replay + forgetting curve).
 *       REM does light associative spreading on the recent context
 *       buffer (creative recombination).
 * WHY:  Without this, the lexicon never participates in sleep
 *       consolidation — vocabulary growth is wake-only and decay is
 *       absent, so noise accumulates and the trained lexicon drifts
 *       from biological plausibility.
 *
 * Stage effects:
 *   - SLEEP_STATE_AWAKE / DROWSY: no-op
 *   - SLEEP_STATE_LIGHT_NREM (strength≈0.3): mild decay of unused bindings
 *   - SLEEP_STATE_DEEP_NREM  (strength≈0.8): strong reinforcement of
 *     frequent bindings + decay of stale ones
 *   - SLEEP_STATE_REM        (strength≈0.5): mild spreading via
 *     context vectors (no decay)
 *
 * @param gl              System handle (NULL → no-op)
 * @param sleep_state_int sleep_state_t value (kept as int to keep
 *                        nimcp_sleep_wake.h out of the public include
 *                        surface; impl casts internally)
 * @param strength        Consolidation magnitude [0, 1]
 * @return 0 on success, -1 on bad parameters.
 */
int grounded_language_sleep_consolidate(
    grounded_language_t* gl,
    int sleep_state_int,
    float strength);

/**
 * @brief Modulation taps from the connected cortexes — used to bias
 *        grounding strength and comprehension confidence in real time.
 *        All values are 0 when the corresponding cortex isn't attached
 *        or its scalar state isn't readable (e.g. visual_cortex has no
 *        scalar attention tap, only stats counters — left at 0).
 */
typedef struct {
    float visual_activity;    /**< Visual cortex activity proxy [0, 1] */
    float audio_salience;     /**< Audio cortex speech-salience tap [0, 1] */
    float speech_confidence;  /**< Speech cortex phoneme confidence [0, 1] */
} gl_cortex_modulation_t;

/**
 * @brief Read current cortex modulation values into out.
 *
 * Cheap (~3 scalar reads). Called from grounded_language_ground() and
 * grounded_language_comprehend() to bias strength/confidence by which
 * sensory cortexes are currently active. Zero-fills out on NULL gl
 * or NULL out.
 *
 * @param gl   System handle
 * @param out  Caller-supplied modulation struct (zero-filled on entry)
 * @return 0 on success, -1 on bad parameters.
 */
int grounded_language_get_cortex_modulation(
    grounded_language_t* gl,
    gl_cortex_modulation_t* out);

/*=============================================================================
 * Dialect / accent conditioning (#14)
 *
 * The lexicon is stored language-agnostic by design — words from any
 * locale can coexist. Dialect tagging lets callers mark the *current*
 * input/output context (e.g., "en-US", "fr-CA", "zh-CN") so that
 * fuzzy lookup, production templates, and SNN bridge prefer entries
 * that were learned under that dialect when ranking ties. Default is
 * the empty string — dialect-agnostic mode. Both setter and getter
 * are NULL-safe; setter truncates to GL_MAX_DIALECT_LEN-1 chars.
 *===========================================================================*/

/** Set the current dialect context. NULL or "" clears the tag. */
void grounded_language_set_dialect(grounded_language_t* gl,
                                     const char* dialect);

/** Read the current dialect tag (NUL-terminated). Returns "" when
 *  unset or gl is NULL. The pointer is owned by GL — copy if you
 *  need to retain it past a subsequent set call. */
const char* grounded_language_get_dialect(const grounded_language_t* gl);

/*=============================================================================
 * NLP Frontend (Morphology / Embeddings / Subword Tokenizer)
 *
 * Three-stage lookup chain for unknown words in comprehend:
 *   1. Exact lexicon hit                 (fast, O(1))
 *   2. Morphological normalization       (suffix-strip, then exact retry)
 *   3. Fuzzy match                       (phonological + character-set)
 *   4. BPE subword fallback              (only when tokenizer is connected)
 * Stages 2+4 + the embedding blend below are no-ops when the relevant
 * subsystems aren't connected — the legacy behavior is preserved.
 *===========================================================================*/

/**
 * @brief Strip common English inflectional suffixes to produce a
 *        normalized lookup form (a lightweight stemmer — not full
 *        lemmatization, but enough to make running→run, cats→cat,
 *        quickly→quick land on the same lexicon entry).
 *
 * @param word    Input word (NUL-terminated, ASCII-lowercased recommended)
 * @param out     Caller-supplied buffer for stripped form
 * @param out_sz  Size of out buffer (>= GL_MAX_WORD_LEN recommended)
 * @return Number of suffix chars stripped (0 if unchanged), -1 on error.
 */
int gl_morph_normalize(const char* word, char* out, size_t out_sz);

/**
 * @brief Suffix-based POS hint. Returns a gl_word_class_t guess from
 *        morphological cues alone (-ing/-ed→VERB, -ly→ADVERB,
 *        -tion/-ness/-ment→NOUN, -ous/-ive/-ful/-able→ADJECTIVE).
 *        Returns GL_CLASS_UNKNOWN when no suffix matches.
 *
 *        Used to bootstrap word-class inference for never-seen-in-context
 *        words. Combined with the existing positional heuristic in
 *        learn_from_text — morphology takes priority when confidence is
 *        higher.
 */
gl_word_class_t gl_morph_pos_hint(const char* word);

/**
 * @brief Word-form → integer-token-id callback used by the embedding
 *        bridge. The caller knows their tokenizer's vocab; GL just asks
 *        "what id is this word?" Return 0 for unknown.
 */
typedef uint32_t (*gl_word_to_id_fn)(void* ctx, const char* word);

/**
 * @brief Connect a word-embedding layer + lookup callback. When wired,
 *        grounded_language_comprehend() blends the per-word embedding
 *        into the result's semantic_vector (subject to dim match).
 *
 *        Pass NULL for emb to disconnect. emb_dim must equal GL's
 *        semantic_dim or the embedding contribution is skipped per word.
 *
 * @param gl              System handle
 * @param emb             embedding_layer_t* (opaque)
 * @param emb_dim         Embedding output dimension
 * @param word_to_id_fn   Callback to map word → token_id (caller-owned)
 * @param ctx             Opaque ctx forwarded to word_to_id_fn
 */
void grounded_language_connect_embeddings(
    grounded_language_t* gl,
    void* emb,
    uint32_t emb_dim,
    gl_word_to_id_fn word_to_id_fn,
    void* ctx);

/**
 * @brief Connect a BPE tokenizer for subword fallback on totally OOV
 *        words. When wired, comprehend's last-resort path decomposes
 *        unrecognized words into subword token ids and pulls embeddings
 *        for those (when the embedding layer is also connected).
 *
 * @param gl   System handle
 * @param tok  tokenizer_t* (opaque), NULL to disconnect
 */
void grounded_language_connect_tokenizer(
    grounded_language_t* gl,
    void* tok);

/*=============================================================================
 * Named Entity Recognition + Shallow Chunking
 *
 * NER:       per-word classifier driven by capitalization / numeric form /
 *            sentence-position cues. Pure function — no lexicon mutation.
 * Chunking:  groups consecutive words into NP/VP/PP/ADJP/ADVP based on the
 *            POS sequence (lexicon-known classes + morph hints + NER).
 * Chinking:  hard breaks inside chunks at commas / conjunctions / verb
 *            tokens that interrupt an otherwise contiguous NP.
 *===========================================================================*/

/**
 * @brief Entity types produced by gl_ner_classify(). Coarse-grained on
 *        purpose — finer-grained NER (date subtypes, ORG vs LOC) is a
 *        downstream KG concern.
 */
typedef enum {
    GL_ENTITY_NONE   = 0,
    GL_ENTITY_PERSON,    /**< Capitalized, alphabetic, mid-sentence */
    GL_ENTITY_PLACE,     /**< Caps + known place suffix (-ville, -burg) */
    GL_ENTITY_ORG,       /**< All-caps acronym, or caps + corp suffix */
    GL_ENTITY_NUMBER,    /**< Pure numeric tokens */
    GL_ENTITY_DATE,      /**< 4-digit year, or numeric+slash forms */
    GL_ENTITY_OTHER,     /**< Capitalized but not classifiable */
} gl_entity_type_t;

/**
 * @brief Classify a single token as an entity (or not).
 *
 * @param word               Input token (caller already split text)
 * @param prev_word_or_null  Preceding token, or NULL at sentence start
 * @param is_sentence_start  True if this is the first token of a sentence
 *                           (sentence-initial caps don't imply entity)
 */
gl_entity_type_t gl_ner_classify(const char* word,
                                  const char* prev_word_or_null,
                                  bool is_sentence_start);

/**
 * @brief Chunk type produced by the shallow chunker.
 */
typedef enum {
    GL_CHUNK_NONE = 0,
    GL_CHUNK_NP,         /**< Noun phrase: (DT|PRON)? ADJ* (NOUN|ENTITY)+ */
    GL_CHUNK_VP,         /**< Verb phrase: (AUX)? VERB ADV* */
    GL_CHUNK_PP,         /**< Prepositional phrase: prep + NP */
    GL_CHUNK_ADJP,       /**< Adjective phrase: ADV? ADJ */
    GL_CHUNK_ADVP,       /**< Adverb phrase: ADV+ */
} gl_chunk_type_t;

/** Maximum words per chunk reported by the API. */
#define GL_CHUNK_MAX_WORDS 12

/**
 * @brief Single chunk emitted by grounded_language_chunk(). Index into
 *        the caller's tokenized input is via [start_word, end_word).
 */
typedef struct {
    gl_chunk_type_t  type;
    uint32_t         start_word;       /**< First word index, inclusive */
    uint32_t         end_word;         /**< Last word index, exclusive */
    char             head_word[32];    /**< Head form (NOUN for NP, etc.) */
    gl_entity_type_t head_entity;      /**< Entity tag on the head, if any */
    bool             chinked;          /**< True if a chink rule split this */
} gl_chunk_t;

/**
 * @brief Run the shallow chunker over a sentence. Tokenizes the input,
 *        classifies each token (lexicon → morph hint → NER fallback),
 *        then greedily groups tokens into NP/VP/PP/ADJP/ADVP using
 *        regex-style POS patterns. Chinking rules break NPs at commas,
 *        conjunctions, and unrelated verbs.
 *
 * @param gl              System handle
 * @param text            Input sentence (caller-owned NUL-terminated)
 * @param chunks_out      Caller-supplied buffer
 * @param max_chunks      Capacity of chunks_out
 * @param chunk_count_out [OUT] Number of chunks written
 * @return 0 on success, -1 on error.
 */
int grounded_language_chunk(grounded_language_t* gl,
                             const char* text,
                             gl_chunk_t* chunks_out,
                             uint32_t max_chunks,
                             uint32_t* chunk_count_out);

/*=============================================================================
 * Per-Network Bridges (LNN / cortex-CNN / FNO / ANN)
 *
 * grounded_language already feeds into SNN via nimcp_snn_language_bridge.
 * These bridges close the gap with the other networks: the comprehend
 * semantic_vector gets broadcast to each connected network's forward
 * pass, and per-network response magnitudes feed back as modulation
 * factors on confidence + binding strength.
 *
 * Each attach function takes an opaque pointer; broadcast/modulation
 * are no-ops when no network is wired (the legacy single-tier path
 * is preserved). Networks can be wired and unwired independently —
 * passing NULL to any attach function is the disconnect operation.
 *===========================================================================*/

/**
 * @brief Per-network response magnitudes from the most recent
 *        broadcast. Each scalar in [0, 1]. Zero when the corresponding
 *        network isn't attached or its forward pass produced no
 *        meaningful output.
 */
typedef struct {
    float lnn_magnitude;      /**< L2 norm of LNN forward output (normalized) */
    float cnn_magnitude;      /**< Cortex-CNN feature norm (normalized) */
    float fno_magnitude;      /**< FNO embedding norm (normalized) */
    float ann_magnitude;      /**< ANN/adaptive predictor norm (reserved) */
} gl_network_modulation_t;

/** Attach an LNN layer. lnn_layer_t* (opaque). NULL = disconnect. */
void grounded_language_attach_lnn(grounded_language_t* gl, void* lnn_layer);

/** Attach a cortex-CNN processor (uses the speech variant for 1D float input). */
void grounded_language_attach_cortex_cnn(grounded_language_t* gl, void* cnn_proc);

/** Attach an FNO audio processor (uses fno_audio_forward for 1D float input). */
void grounded_language_attach_fno(grounded_language_t* gl, void* fno_proc);

/** Attach a generic ANN predictor — reserved hook (uses callback). */
typedef int (*gl_ann_predict_fn)(void* ctx, const float* in, uint32_t in_dim,
                                   float* out, uint32_t out_dim);
void grounded_language_attach_ann(grounded_language_t* gl,
                                   gl_ann_predict_fn fn,
                                   void* ctx);

/**
 * @brief Broadcast a semantic vector to every attached network. Runs
 *        each forward pass with the vector as input, captures response
 *        magnitudes into the GL's last-modulation cache. Returns the
 *        number of networks that successfully responded.
 *
 * @param gl   System handle
 * @param vec  Semantic vector (length = gl->semantic_dim)
 * @param dim  Dimension of vec
 * @return Count of networks that produced a valid response, -1 on error.
 */
int grounded_language_broadcast_to_networks(grounded_language_t* gl,
                                              const float* vec,
                                              uint32_t dim);

/**
 * @brief Read the last-broadcast modulation values. Cheap scalar read.
 *        Used internally by comprehend to bias confidence.
 *
 * @return 0 on success, -1 on bad parameters. Out is zero-filled on entry.
 */
int grounded_language_get_network_modulation(grounded_language_t* gl,
                                               gl_network_modulation_t* out);

/*=============================================================================
 * Cognitive subscriber bus
 *
 * GL fires lightweight events at four points (new word created, grounding
 * succeeded, comprehension finished, production finished). Cognitive
 * modules subscribe a callback to receive these events and react in
 * their own terms — inner speech may rehearse the word, imagination
 * may register the concept, theory-of-mind may treat the utterance as
 * an observation, etc.
 *
 * Pull direction (cognitive module → GL) uses the existing public APIs
 * (grounded_language_lookup, _produce, _comprehend, _ground). The bus
 * here covers the push direction without GL needing to know each
 * module's internals. Up to 16 subscribers; out-of-order unsubscribe
 * is supported via the ctx pointer.
 *===========================================================================*/

typedef enum {
    GL_EVENT_NEW_WORD     = 0,  /**< A new lexicon entry was just created */
    GL_EVENT_GROUNDED,           /**< grounded_language_ground succeeded */
    GL_EVENT_COMPREHENDED,       /**< grounded_language_comprehend finished */
    GL_EVENT_PRODUCED,           /**< grounded_language_produce finished */
    /* #10 Active-learning curriculum signal. Fired when comprehend hits
     * a word with confidence below GL_LOW_CONFIDENCE_THRESHOLD —
     * downstream curriculum modules can register their interest in
     * grounding that word the next time it shows up in sensory input. */
    GL_EVENT_NEEDS_GROUNDING,
} gl_event_type_t;

/**
 * @brief Event payload pushed to subscribers. Pointer fields are
 *        owned by GL and valid only for the duration of the callback.
 *        Subscribers must copy any data they want to retain.
 */
typedef struct {
    gl_event_type_t  type;
    const char*      word;          /**< NEW_WORD/GROUNDED: the word; else NULL */
    const char*      text;          /**< COMPREHENDED/PRODUCED: full text; else NULL */
    const float*     semantic_vec;  /**< COMPREHENDED/PRODUCED: [gl->semantic_dim] */
    uint64_t         concept_id;    /**< GROUNDED: bound concept; else 0 */
    float            valence;       /**< Emotional valence at event time */
    float            arousal;       /**< Emotional arousal */
    float            confidence;    /**< [0,1]; comprehension/production confidence */
} gl_event_t;

/** Subscriber callback. Return 0 to continue, non-zero to log+continue. */
typedef int (*gl_event_callback_t)(void* ctx, const gl_event_t* event);

/* Event-type bitmask (#4 — per-subscriber filter). Use these to limit
 * which events a subscriber receives — the bus skips wrappers whose
 * mask doesn't match, removing the if-else boilerplate from every
 * wrapper and cutting bus-walk cost from O(N) to ~O(N_relevant). */
#define GL_EVENT_MASK_NEW_WORD          (1u << 0)
#define GL_EVENT_MASK_GROUNDED          (1u << 1)
#define GL_EVENT_MASK_COMPREHENDED      (1u << 2)
#define GL_EVENT_MASK_PRODUCED          (1u << 3)
#define GL_EVENT_MASK_NEEDS_GROUNDING   (1u << 4)
#define GL_EVENT_MASK_ALL               (0xFFFFFFFFu)

/** Confidence floor below which comprehend fires a NEEDS_GROUNDING
 *  event for the lowest-confidence word it just processed (#10). */
#define GL_LOW_CONFIDENCE_THRESHOLD     0.20f

/**
 * @brief Subscribe to GL events. Returns 0 on success, -1 if the
 *        subscriber table is full or arguments invalid.
 *
 *  Equivalent to subscribe_ex(gl, fn, ctx, GL_EVENT_MASK_ALL, 0).
 *
 * @param gl   System handle
 * @param fn   Callback (must be non-NULL)
 * @param ctx  Opaque ctx forwarded on every callback. Used as the
 *             dedup key — registering the same ctx twice replaces
 *             the prior callback.
 */
int grounded_language_subscribe(grounded_language_t* gl,
                                  gl_event_callback_t fn,
                                  void* ctx);

/**
 * @brief Subscribe with extended options (#3 priority + #4 type mask).
 *
 *  - type_mask: bitwise-OR of GL_EVENT_MASK_* constants. The bus skips
 *    callbacks whose mask doesn't match the firing event's type.
 *  - priority: higher fires first. Use 0 for default. Use a higher
 *    value for the hemispheric-bridge / lateralization callback so it
 *    can mute downstream observers before they react. Range [-127,127].
 *
 *  Behaviorally equivalent to subscribe() when type_mask=GL_EVENT_MASK_ALL
 *  and priority=0; both are stable identity for legacy callers.
 *
 * @return 0 on success, -1 on invalid args or table full.
 */
int grounded_language_subscribe_ex(grounded_language_t* gl,
                                     gl_event_callback_t fn,
                                     void* ctx,
                                     uint32_t type_mask,
                                     int8_t priority);

/** Unsubscribe by ctx pointer. Returns 0 if removed, -1 if not found. */
int grounded_language_unsubscribe(grounded_language_t* gl, void* ctx);

/** How many subscribers are currently registered. */
uint32_t grounded_language_subscriber_count(const grounded_language_t* gl);

/**
 * @brief Test whether a word currently has a lexicon entry (#1).
 *        Lower-cases on entry. Read-only — never creates a binding.
 *        Used by region adapters (broca/wernicke) to fall through to
 *        GL when their local lexicons miss without paying the cost
 *        of a full comprehend.
 * @return true if the word is in the lexicon, false otherwise (also
 *         false for NULL or empty input).
 */
bool grounded_language_has_word(const grounded_language_t* gl, const char* word);

/*=============================================================================
 * Probes & Metrics (operational telemetry beyond gl_stats_t)
 *===========================================================================*/

/**
 * @brief Operational probe metrics for the language module. Returned
 *        by grounded_language_get_probe_metrics — covers everything a
 *        Grafana/dashboard query would want without exposing internal
 *        struct layout.
 *
 *  All counters are cumulative since gl_create. All gauges are
 *  instantaneous (current state). Caller-owned out struct, zero-filled
 *  on entry by the impl so missing fields stay defaulted.
 */
typedef struct {
    /* === Lexicon gauges === */
    uint32_t vocab_count;             /**< Current entries in lexicon */
    uint32_t templates_count;         /**< Current syntactic templates */
    float    avg_binding_strength;    /**< Mean across all bindings */
    float    avg_binding_confidence;  /**< Mean across all bindings */

    /* === Bus state === */
    uint32_t subscriber_count;        /**< Total subscribers right now */
    uint32_t subscriber_high_priority;/**< Subs with priority > 0 */
    uint32_t subscriber_filtered;     /**< Subs with mask != ALL */
    uint64_t events_dropped_reentry;  /**< Inner fire blocked by guard (#11) */
    bool     in_fire_event;           /**< True if we're inside fire right now */

    /* === Forgetting curve (#15) === */
    uint32_t entries_decayed_last_24h;
    uint64_t entries_decayed_all_time;

    /* === Network bridges === */
    float    last_lnn_response_mag;   /**< [0,1] last LNN bridge magnitude */
    float    last_cnn_response_mag;
    float    last_fno_response_mag;
    float    last_ann_response_mag;

    /* === Cortex modulation taps (current state, not cumulative) === */
    float    visual_activity;
    float    audio_salience;
    float    speech_confidence;

    /* === Throughput rates (since boot) === */
    uint64_t total_groundings;
    uint64_t total_comprehensions;
    uint64_t total_productions;
    uint64_t total_new_words;

    /* === Region wiring health === */
    bool     broca_attached;
    bool     wernicke_attached;
    bool     working_memory_attached;
    bool     hippocampus_attached;
    bool     embedding_attached;
    bool     tokenizer_attached;

    /* === Dialect / accent (#14) === */
    char     context_dialect[GL_MAX_DIALECT_LEN];

    /* === Active-learning + negative-grounding throughput === */
    uint64_t total_negative_groundings;
    uint64_t total_needs_grounding;
} gl_probe_metrics_t;

/**
 * @brief Snapshot operational probe metrics. Out is zero-filled
 *        on entry so future field additions remain backwards-safe
 *        (missing reads see 0, not stale stack data).
 *
 * @param gl   System handle (NULL → no-op, returns -1)
 * @param out  Destination (NULL → no-op, returns -1)
 * @return 0 on success, -1 on bad params.
 */
int grounded_language_get_probe_metrics(const grounded_language_t* gl,
                                          gl_probe_metrics_t* out);

/*=============================================================================
 * Per-cognitive-module attach helpers — convenience wrappers that
 * subscribe a wrapper callback translating gl_event_t into the
 * module's native API. Each accepts NULL for the module pointer to
 * make brain-init wiring tolerant of missing modules.
 *===========================================================================*/

/** Attach inner-speech: every NEW_WORD + COMPREHENDED event becomes
 *  a candidate for silent rehearsal. mod is the inner_speech_t* opaque
 *  handle (NULL = no-op). */
void grounded_language_attach_inner_speech(grounded_language_t* gl, void* mod);

/** Attach imagination engine: GROUNDED events register the new
 *  concept as a candidate scene element. */
void grounded_language_attach_imagination(grounded_language_t* gl, void* mod);

/** Attach theory of mind: COMPREHENDED utterances are observations
 *  about an external speaker's beliefs. */
void grounded_language_attach_theory_of_mind(grounded_language_t* gl, void* mod);

/** Attach empathetic response: emotionally-charged GROUNDED events
 *  (high arousal) seed empathy generation. */
void grounded_language_attach_empathy(grounded_language_t* gl, void* mod);

/** Attach introspection: every PRODUCED event is logged as a
 *  self-narration sample. */
void grounded_language_attach_introspection(grounded_language_t* gl, void* mod);

/** Attach reasoning subsystem: COMPREHENDED utterances feed forward
 *  chaining as new premises. */
void grounded_language_attach_reasoning(grounded_language_t* gl, void* mod);

/** Attach narrative system: COMPREHENDED text appended to the
 *  current narrative buffer. */
void grounded_language_attach_narrative(grounded_language_t* gl, void* mod);

/** Attach metacognition: every event ticks the metacognitive
 *  reflection counter. */
void grounded_language_attach_metacognition(grounded_language_t* gl, void* mod);

/** Attach analogical-transfer: NEW_WORD events trigger analogy
 *  search against existing concepts. */
void grounded_language_attach_analogical(grounded_language_t* gl, void* mod);

/** Attach emergent-language module: GROUNDED + PRODUCED events
 *  feed the emergent-symbol learner. */
void grounded_language_attach_emergent_language(grounded_language_t* gl, void* mod);

/*--------------------------------------------------------------------------
 * Brain-region subscribers — anatomical regions that should observe
 * language events in their own modality. All four are NULL-tolerant
 * (mod = NULL is a disconnect).
 *------------------------------------------------------------------------*/

/** Attach prefrontal cortex: COMPREHENDED + PRODUCED + high-arousal
 *  GROUNDED events feed executive monitoring of language. */
void grounded_language_attach_prefrontal(grounded_language_t* gl, void* mod);

/** Attach insula: COMPREHENDED events with non-zero valence feed
 *  interoceptive integration (gut-feel evaluation of an utterance). */
void grounded_language_attach_insula(grounded_language_t* gl, void* mod);

/** Attach cingulate cortex: COMPREHENDED + PRODUCED events feed
 *  conflict-monitoring (low-confidence comprehensions or production
 *  errors are flagged for ACC attention). */
void grounded_language_attach_cingulate(grounded_language_t* gl, void* mod);

/** Attach amygdala: GROUNDED events with arousal > 0.3 feed
 *  emotional tagging of the bound concept (fear/threat conditioning
 *  on language-encoded experiences). */
void grounded_language_attach_amygdala(grounded_language_t* gl, void* mod);

/** Attach orbitofrontal cortex (OFC): GROUNDED + COMPREHENDED events
 *  with non-trivial valence feed reward-value bindings (stimulus
 *  value learning + RPE updates). */
void grounded_language_attach_ofc(grounded_language_t* gl, void* mod);

/*=============================================================================
 * Query / Introspection
 *===========================================================================*/

/**
 * @brief Look up word in lexicon
 * @param gl    System handle
 * @param word  Word to look up
 * @return Lexicon entry (read-only), or NULL if unknown word
 */
const gl_lexicon_entry_t* grounded_language_lookup(
    const grounded_language_t* gl,
    const char* word);

/**
 * @brief Get words associated with a concept
 * @param gl          System handle
 * @param concept_id  Concept to find words for
 * @param words       Output array of word forms (caller provides)
 * @param max_words   Maximum words to return
 * @return Number of words found
 */
uint32_t grounded_language_words_for_concept(
    const grounded_language_t* gl,
    uint64_t concept_id,
    const char** words,
    uint32_t max_words);

/**
 * @brief Get system statistics
 * @param gl    System handle
 * @param stats Output statistics
 */
void grounded_language_get_stats(
    const grounded_language_t* gl,
    gl_stats_t* stats);

/**
 * @brief Get the semantic vector dimension this system was created with.
 *
 * WHAT: Exposes the immutable semantic_dim chosen at create() time.
 * WHY:  Diagnostic probes (probe_comprehend) need this to bound reads
 *       — comprehension_result.semantic_vector is sized to this dim and
 *       walking past it is undefined behaviour.
 * @return semantic_dim, or 0 if gl is NULL.
 */
uint32_t grounded_language_get_semantic_dim(
    const grounded_language_t* gl);

/**
 * @brief Minimum production confidence below which respond() returns
 *        an "I don't have words for that yet" fallback instead of
 *        emitting a degenerate template.
 *
 * Confidence here = fluency × relevance from gl_production_result_t.
 * Below this floor the lexicon hasn't accumulated enough word-concept
 * binding strength to justify producing — emitting a top-1 template
 * looks like authoritative speech but is just the strongest seeded
 * attractor, which causes mode collapse in the user-visible output.
 */
#define GL_RESPOND_MIN_CONFIDENCE 0.05f

/*=============================================================================
 * Serialization
 *===========================================================================*/

/**
 * @brief Save grounded language state to file
 * @param gl   System handle
 * @param path File path to save to
 * @return 0 on success, -1 on error
 */
int grounded_language_save(
    const grounded_language_t* gl,
    const char* path);

/**
 * @brief Load grounded language state from file
 * @param path File path to load from
 * @param semantic_memory Semantic memory to connect (can be NULL)
 * @return Loaded system, or NULL on error
 */
grounded_language_t* grounded_language_load(
    const char* path,
    void* semantic_memory);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_GROUNDED_LANGUAGE_H */
