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
