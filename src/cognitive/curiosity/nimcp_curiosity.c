//=============================================================================
// nimcp_curiosity.c - Curiosity-Driven Learning Implementation
// Refactored for maintainability, performance, and clarity
//=============================================================================

#include "nimcp_curiosity.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/brain/nimcp_brain.h"
#include "utils/containers/nimcp_hash_table.h"
#include "utils/memory/nimcp_memory.h"  // CRITICAL: Declares nimcp_calloc/nimcp_free return types
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"  // Dopamine modulation

//=============================================================================
// Hash Table Configuration for O(1) Concept Lookup
//=============================================================================

#define HASH_TABLE_SIZE 4096  // Power of 2 for fast modulo via bitwise AND

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Concept data stored in hash table
 *
 * NOTE: Hash table now handles chaining internally via nimcp_hash_table utility.
 * No need for explicit 'next' pointer.
 */
typedef struct concept_bucket_struct {
    char concept[256];
    float familiarity;
    uint32_t exposure_count;
    uint64_t last_encountered;
    char** related_concepts;
    uint32_t num_related;
} concept_bucket_t;

/**
 * @brief Value destructor for concept_bucket_t
 *
 * WHAT: Clean up concept_str bucket and related concepts
 * WHY: Called by hash table when entry is removed or table destroyed
 * HOW: Free related_concepts array and strings
 */
static void concept_bucket_destructor(void* value, size_t value_size)
{
    (void) value_size;  // Unused parameter
    if (!value)
        return;

    concept_bucket_t* bucket = (concept_bucket_t*) value;

    // Free related concepts
    if (bucket->related_concepts) {
        for (uint32_t j = 0; j < bucket->num_related; j++) {
            nimcp_free(bucket->related_concepts[j]);
        }
        nimcp_free(bucket->related_concepts);
    }
}

/**
 * @brief Question history entry
 */
typedef struct {
    char question[512];
    char answer[2048];
    bool answered;
    float learning_value;
    uint64_t timestamp;
} question_history_t;

/**
 * @brief Knowledge source registration
 */
typedef struct {
    char name[128];
    knowledge_search_fn_t search_fn;
    void* context;
    bool active;
} knowledge_source_t;

//=============================================================================
// State Pattern - Learning Stage Behaviors
//=============================================================================

/**
 * @brief Virtual function table for learning stage behaviors
 *
 * Each stage implements these methods to customize learning behavior.
 * This eliminates switch statements and enables polymorphic stage behavior.
 */
typedef struct {
    float (*calculate_learning_potential)(const char* concept_str, float gap_size);
    float (*get_baseline_curiosity)(void);
    uint32_t (*get_question_types_count)(void);
    const question_type_t* (*get_question_types)(void);
} learning_stage_strategy_t;

// Forward declarations for stage strategies
static const learning_stage_strategy_t* get_stage_strategy(learning_stage_t stage);

//=============================================================================
// Strategy Pattern - Question Generation
//=============================================================================

/**
 * @brief Function pointer type for question generation strategies
 *
 * Each question type has its own generation strategy.
 * Eliminates switch statements and enables extensible question types.
 */
typedef void (*question_generator_fn_t)(const char* topic, char* question, size_t max_len,
                                        float* priority_multiplier, float* difficulty);

/**
 * @brief Question generation strategy entry
 */
typedef struct {
    question_type_t type;
    question_generator_fn_t generator;
    float priority_multiplier;
    float difficulty;
} question_strategy_t;

// Forward declaration for strategy table
static const question_strategy_t* get_question_strategy(question_type_t type);

//=============================================================================
// Main Engine Structure
//=============================================================================

struct curiosity_engine_struct {
    char learner_name[128];

    // Hash table for O(1) concept_str lookup (using nimcp_hash_table utility)
    hash_table_t* concept_hash_table;
    uint32_t total_concepts;

    // Question history
    question_history_t* questions;
    uint32_t num_questions;
    uint32_t questions_capacity;

    // Knowledge sources
    knowledge_source_t* sources;
    uint32_t num_sources;
    uint32_t sources_capacity;

    // Neural networks for prediction
    brain_t gap_detector;
    brain_t question_prioritizer;

    // Motivation state
    float baseline_curiosity;
    float current_motivation;

    // Learning stage with strategy pattern
    learning_stage_t stage;
    const learning_stage_strategy_t* stage_strategy;

    // Statistics
    learning_progress_t progress;
};

//=============================================================================
// Stage Pattern Implementation - Infant Stage
//=============================================================================

/**
 * @brief Infant stage learning potential calculation
 *
 * Infants find nearly everything highly valuable for learning.
 * High baseline potential regardless of gap size.
 *
 * Complexity: O(1)
 */
static float infant_learning_potential(const char* concept_str, float gap_size)
{
    return 0.9f;  // Everything is exciting and worth learning
}

static float infant_baseline_curiosity(void)
{
    return 0.95f;
}

static uint32_t infant_question_types_count(void)
{
    return 3;  // WHAT, WHY, HOW only
}

static const question_type_t infant_question_types[] = {QUESTION_WHAT, QUESTION_WHY, QUESTION_HOW};

static const question_type_t* infant_get_question_types(void)
{
    return infant_question_types;
}

static const learning_stage_strategy_t infant_strategy = {
    .calculate_learning_potential = infant_learning_potential,
    .get_baseline_curiosity = infant_baseline_curiosity,
    .get_question_types_count = infant_question_types_count,
    .get_question_types = infant_get_question_types};

//=============================================================================
// Stage Pattern Implementation - Toddler Stage
//=============================================================================

/**
 * @brief Toddler stage learning potential calculation
 *
 * Toddlers prefer concrete, simple concepts.
 * Penalizes complex or abstract concepts (approximated by length).
 *
 * Complexity: O(1) - strlen is O(n) but concept_str length bounded
 */
static float toddler_learning_potential(const char* concept_str, float gap_size)
{
    size_t len = strlen(concept_str);
    float simplicity_bonus = (len < 15) ? 1.0f : 0.5f;
    return 0.8f * simplicity_bonus;
}

static float toddler_baseline_curiosity(void)
{
    return 0.90f;
}

static uint32_t toddler_question_types_count(void)
{
    return 5;  // Add WHERE and WHEN
}

static const question_type_t toddler_question_types[] = {QUESTION_WHAT, QUESTION_WHY, QUESTION_HOW,
                                                         QUESTION_WHERE, QUESTION_WHEN};

static const question_type_t* toddler_get_question_types(void)
{
    return toddler_question_types;
}

static const learning_stage_strategy_t toddler_strategy = {
    .calculate_learning_potential = toddler_learning_potential,
    .get_baseline_curiosity = toddler_baseline_curiosity,
    .get_question_types_count = toddler_question_types_count,
    .get_question_types = toddler_get_question_types};

//=============================================================================
// Stage Pattern Implementation - Child Stage
//=============================================================================

/**
 * @brief Child stage learning potential calculation
 *
 * Children can handle abstract concepts and are driven by curiosity intensity.
 * Learning potential scales with how curious they are about the topic.
 *
 * Complexity: O(1)
 */
static float child_learning_potential(const char* concept_str, float gap_size)
{
    return 0.7f + 0.3f * gap_size;  // Scales with gap/curiosity
}

static float child_baseline_curiosity(void)
{
    return 0.80f;
}

static uint32_t child_question_types_count(void)
{
    return 7;  // All question types
}

static const question_type_t child_question_types[] = {QUESTION_WHAT,  QUESTION_WHY,  QUESTION_HOW,
                                                       QUESTION_WHERE, QUESTION_WHEN, QUESTION_WHO,
                                                       QUESTION_WHICH};

static const question_type_t* child_get_question_types(void)
{
    return child_question_types;
}

static const learning_stage_strategy_t child_strategy = {
    .calculate_learning_potential = child_learning_potential,
    .get_baseline_curiosity = child_baseline_curiosity,
    .get_question_types_count = child_question_types_count,
    .get_question_types = child_get_question_types};

//=============================================================================
// Stage Pattern Implementation - Adolescent Stage
//=============================================================================

static float adolescent_learning_potential(const char* concept_str, float gap_size)
{
    return 0.6f + 0.4f * gap_size;
}

static float adolescent_baseline_curiosity(void)
{
    return 0.70f;
}

static uint32_t adolescent_question_types_count(void)
{
    return 7;
}

static const question_type_t* adolescent_get_question_types(void)
{
    return child_question_types;  // Same as child
}

static const learning_stage_strategy_t adolescent_strategy = {
    .calculate_learning_potential = adolescent_learning_potential,
    .get_baseline_curiosity = adolescent_baseline_curiosity,
    .get_question_types_count = adolescent_question_types_count,
    .get_question_types = adolescent_get_question_types};

//=============================================================================
// Stage Pattern Implementation - Adult Stage
//=============================================================================

static float adult_learning_potential(const char* concept_str, float gap_size)
{
    return 0.5f + 0.5f * gap_size;  // More selective
}

static float adult_baseline_curiosity(void)
{
    return 0.60f;
}

static uint32_t adult_question_types_count(void)
{
    return 7;
}

static const question_type_t* adult_get_question_types(void)
{
    return child_question_types;
}

static const learning_stage_strategy_t adult_strategy = {
    .calculate_learning_potential = adult_learning_potential,
    .get_baseline_curiosity = adult_baseline_curiosity,
    .get_question_types_count = adult_question_types_count,
    .get_question_types = adult_get_question_types};

//=============================================================================
// Stage Pattern Implementation - Expert Stage
//=============================================================================

static float expert_learning_potential(const char* concept_str, float gap_size)
{
    return 0.4f + 0.6f * gap_size;  // Only interested in significant gaps
}

static float expert_baseline_curiosity(void)
{
    return 0.50f;
}

static uint32_t expert_question_types_count(void)
{
    return 7;
}

static const question_type_t* expert_get_question_types(void)
{
    return child_question_types;
}

static const learning_stage_strategy_t expert_strategy = {
    .calculate_learning_potential = expert_learning_potential,
    .get_baseline_curiosity = expert_baseline_curiosity,
    .get_question_types_count = expert_question_types_count,
    .get_question_types = expert_get_question_types};

//=============================================================================
// Stage Strategy Dispatch Table
//=============================================================================

/**
 * @brief Get strategy for learning stage
 *
 * Returns function table for stage-specific behaviors.
 * Replaces switch statements with polymorphism.
 *
 * Complexity: O(1) - array lookup
 */
static const learning_stage_strategy_t* get_stage_strategy(learning_stage_t stage)
{
    static const learning_stage_strategy_t* strategies[] = {&infant_strategy, &toddler_strategy,
                                                            &child_strategy,  &adolescent_strategy,
                                                            &adult_strategy,  &expert_strategy};

    if (stage < 0 || stage >= 6) {
        return &infant_strategy;  // Safe default
    }

    return strategies[stage];
}

//=============================================================================
// Question Generation Strategies
//=============================================================================

/**
 * @brief Generate "What" question
 *
 * Fundamental definitional question. Highest priority as it establishes
 * basic understanding of a concept_str.
 *
 * Complexity: O(n) where n is topic length (due to snprintf)
 */
static void generate_what_question(const char* topic, char* question, size_t max_len,
                                   float* priority_multiplier, float* difficulty)
{
    snprintf(question, max_len, "What is %s?", topic);
    *priority_multiplier = 1.0f;
    *difficulty = 0.3f;
}

/**
 * @brief Generate "Why" question
 *
 * Explores causality and reasoning. Slightly lower priority than "what"
 * but higher difficulty due to abstract reasoning required.
 *
 * Complexity: O(n)
 */
static void generate_why_question(const char* topic, char* question, size_t max_len,
                                  float* priority_multiplier, float* difficulty)
{
    snprintf(question, max_len, "Why does %s happen?", topic);
    *priority_multiplier = 0.9f;
    *difficulty = 0.6f;
}

/**
 * @brief Generate "How" question
 *
 * Explores mechanisms and processes. High difficulty due to procedural
 * understanding required.
 *
 * Complexity: O(n)
 */
static void generate_how_question(const char* topic, char* question, size_t max_len,
                                  float* priority_multiplier, float* difficulty)
{
    snprintf(question, max_len, "How does %s work?", topic);
    *priority_multiplier = 0.8f;
    *difficulty = 0.7f;
}

/**
 * @brief Generate "Where" question
 *
 * Spatial/location question. Medium priority and difficulty.
 *
 * Complexity: O(n)
 */
static void generate_where_question(const char* topic, char* question, size_t max_len,
                                    float* priority_multiplier, float* difficulty)
{
    snprintf(question, max_len, "Where is %s?", topic);
    *priority_multiplier = 0.6f;
    *difficulty = 0.4f;
}

/**
 * @brief Generate "When" question
 *
 * Temporal question. Medium priority and difficulty.
 *
 * Complexity: O(n)
 */
static void generate_when_question(const char* topic, char* question, size_t max_len,
                                   float* priority_multiplier, float* difficulty)
{
    snprintf(question, max_len, "When did %s happen?", topic);
    *priority_multiplier = 0.7f;
    *difficulty = 0.5f;
}

/**
 * @brief Generate "Who" question
 *
 * Asks about agents/actors involved. Medium priority and difficulty.
 *
 * Complexity: O(n)
 */
static void generate_who_question(const char* topic, char* question, size_t max_len,
                                  float* priority_multiplier, float* difficulty)
{
    snprintf(question, max_len, "Who is involved with %s?", topic);
    *priority_multiplier = 0.7f;
    *difficulty = 0.5f;
}

/**
 * @brief Generate "Which" question
 *
 * Discriminative question about types/categories. Lower priority as it
 * assumes some base knowledge.
 *
 * Complexity: O(n)
 */
static void generate_which_question(const char* topic, char* question, size_t max_len,
                                    float* priority_multiplier, float* difficulty)
{
    snprintf(question, max_len, "Which type of %s?", topic);
    *priority_multiplier = 0.5f;
    *difficulty = 0.6f;
}

//=============================================================================
// Question Strategy Dispatch Table
//=============================================================================

/**
 * @brief Question generation strategy dispatch table
 *
 * Maps question types to their generation functions.
 * Eliminates switch statements in question generation.
 *
 * Complexity: O(1) lookup
 */
static const question_strategy_t question_strategies[] = {
    {QUESTION_WHAT, generate_what_question, 1.0f, 0.3f},
    {QUESTION_WHY, generate_why_question, 0.9f, 0.6f},
    {QUESTION_HOW, generate_how_question, 0.8f, 0.7f},
    {QUESTION_WHERE, generate_where_question, 0.6f, 0.4f},
    {QUESTION_WHEN, generate_when_question, 0.7f, 0.5f},
    {QUESTION_WHO, generate_who_question, 0.7f, 0.5f},
    {QUESTION_WHICH, generate_which_question, 0.5f, 0.6f}};

static const size_t num_question_strategies =
    sizeof(question_strategies) / sizeof(question_strategies[0]);

/**
 * @brief Get question generation strategy by type
 *
 * Returns strategy for given question type, or NULL if invalid.
 *
 * Complexity: O(1) - fixed-size array lookup
 */
static const question_strategy_t* get_question_strategy(question_type_t type)
{
    if (type < 0 || (size_t) type >= num_question_strategies) {
        return NULL;
    }
    return &question_strategies[type];
}

//=============================================================================
// Hash Table Operations - O(1) Concept Lookup
//=============================================================================

/**
 * @brief Normalize string for consistent hashing and comparison
 *
 * Converts to lowercase and removes non-alphanumeric except spaces.
 * Ensures consistent lookup regardless of input formatting.
 *
 * Complexity: O(n) where n is input length
 */
static void normalize_string(const char* input, char* output, size_t max_len)
{
    if (!input || !output || max_len == 0) {
        return;
    }

    size_t j = 0;
    for (size_t i = 0; input[i] && j < max_len - 1; i++) {
        if (isalnum(input[i]) || input[i] == ' ') {
            output[j++] = tolower(input[i]);
        }
    }
    output[j] = '\0';
}

/**
 * @brief Find concept_str in hash table
 *
 * Uses hash table for O(1) average case lookup instead of O(n) linear search.
 * Handles collisions via chaining (linked list per bucket).
 *
 * Time Complexity:
 *   - Average: O(1) with good hash distribution
 *   - Worst: O(n) if all concepts hash to same bucket (rare)
 *
 * Space Complexity: O(1)
 *
 * @param engine Curiosity engine
 * @param concept_str Concept to find
 * @return Pointer to concept_str bucket, or NULL if not found
 */
static concept_bucket_t* find_concept_bucket(curiosity_engine_t engine, const char* concept_str)
{
    if (!engine || !concept_str || !engine->concept_hash_table) {
        return NULL;
    }

    return (concept_bucket_t*) hash_table_lookup_string(engine->concept_hash_table, concept_str);
}

/**
 * @brief Add new concept_str to hash table
 *
 * Creates new bucket and inserts at head of chain for O(1) insertion.
 * Automatically handles collisions via chaining.
 *
 * Time Complexity: O(1) average case
 * Space Complexity: O(1) per concept_str
 *
 * @param engine Curiosity engine
 * @param concept_str Concept to add
 * @return Pointer to new or existing concept_str bucket
 */
static concept_bucket_t* add_concept_to_hash_table(curiosity_engine_t engine, const char* concept_str)
{
    if (!engine || !concept_str) {
        return NULL;
    }

    // Check if already exists
    concept_bucket_t* existing = find_concept_bucket(engine, concept_str);
    if (existing) {
        return existing;
    }

    // Create new bucket
    concept_bucket_t new_bucket = {0};
    strncpy(new_bucket.concept, concept_str, sizeof(new_bucket.concept) - 1);
    new_bucket.familiarity = 0.0f;
    new_bucket.exposure_count = 1;
    new_bucket.last_encountered = 0;
    new_bucket.related_concepts = NULL;
    new_bucket.num_related = 0;

    // Insert into hash table
    if (!hash_table_insert_string(engine->concept_hash_table, concept_str, &new_bucket,
                                  sizeof(concept_bucket_t))) {
        return NULL;
    }

    engine->total_concepts++;

    // Return pointer to the stored bucket
    return find_concept_bucket(engine, concept_str);
}

/**
 * @brief Free all concepts in hash table
 *
 * WHAT: Destroy hash table and all concepts
 * WHY: Called during engine destruction
 * HOW: Delegate to hash_table_destroy (calls concept_bucket_destructor)
 *
 * Complexity: O(n) where n is total concepts
 */
static void free_hash_table(curiosity_engine_t engine)
{
    if (!engine || !engine->concept_hash_table) {
        return;
    }

    hash_table_destroy(engine->concept_hash_table);
    engine->concept_hash_table = NULL;
}

//=============================================================================
// Curiosity Engine Lifecycle
//=============================================================================

/**
 * @brief Create new curiosity engine
 *
 * Initializes all subsystems:
 * - Hash table for O(1) concept_str lookup
 * - Neural networks for gap detection and question prioritization
 * - Learning stage strategy pattern
 * - Question and source storage
 *
 * Time Complexity: O(1)
 * Space Complexity: O(1) - initial allocation, grows with use
 *
 * @param learner_name Name identifier for this learner
 * @return New engine instance, or NULL on failure
 */
curiosity_engine_t curiosity_engine_create(const char* learner_name)
{
    if (!learner_name) {
        return NULL;
    }

    curiosity_engine_t engine = nimcp_calloc(1, sizeof(struct curiosity_engine_struct));
    if (!engine) {
        return NULL;
    }

    strncpy(engine->learner_name, learner_name, sizeof(engine->learner_name) - 1);

    // Initialize hash table with nimcp_hash_table utility
    hash_table_config_t ht_config = {.initial_buckets = HASH_TABLE_SIZE,
                                     .key_type = HASH_KEY_STRING,
                                     .hash_algorithm = HASH_ALG_FNV1A,
                                     .case_insensitive = true,
                                     .value_destructor = concept_bucket_destructor,
                                     .thread_safe = false};
    engine->concept_hash_table = hash_table_create(&ht_config);
    if (!engine->concept_hash_table) {
        nimcp_free(engine);
        return NULL;
    }
    engine->total_concepts = 0;

    // Allocate question storage
    engine->questions_capacity = 10000;
    engine->questions = nimcp_calloc(engine->questions_capacity, sizeof(question_history_t));
    if (!engine->questions) {
        nimcp_free(engine);
        return NULL;
    }

    // Allocate source storage
    engine->sources_capacity = 10;
    engine->sources = nimcp_calloc(engine->sources_capacity, sizeof(knowledge_source_t));
    if (!engine->sources) {
        nimcp_free(engine->questions);
        nimcp_free(engine);
        return NULL;
    }

    // Create neural networks
    engine->gap_detector =
        brain_create("gap_detector", BRAIN_SIZE_SMALL, BRAIN_TASK_REGRESSION, 10, 1);

    engine->question_prioritizer =
        brain_create("question_priority", BRAIN_SIZE_SMALL, BRAIN_TASK_REGRESSION, 15, 1);

    // Initialize with infant stage
    engine->stage = STAGE_INFANT;
    engine->stage_strategy = get_stage_strategy(STAGE_INFANT);
    engine->baseline_curiosity = engine->stage_strategy->get_baseline_curiosity();
    engine->current_motivation = engine->baseline_curiosity;

    // Initialize progress
    engine->progress.total_questions_asked = 0;
    engine->progress.total_answers_learned = 0;
    engine->progress.concepts_learned = 0;
    engine->progress.avg_curiosity = engine->baseline_curiosity;
    snprintf(engine->progress.current_focus, sizeof(engine->progress.current_focus),
             "exploring world");

    return engine;
}

/**
 * @brief Destroy curiosity engine and free all resources
 *
 * Frees hash table, questions, sources, and neural networks.
 * Safe to call with NULL pointer.
 *
 * Complexity: O(n) where n is total concepts
 */
void curiosity_engine_destroy(curiosity_engine_t engine)
{
    if (!engine) {
        return;
    }

    free_hash_table(engine);

    nimcp_free(engine->questions);
    nimcp_free(engine->sources);

    brain_destroy(engine->gap_detector);
    brain_destroy(engine->question_prioritizer);

    nimcp_free(engine);
}

//=============================================================================
// Knowledge Gap Detection
//=============================================================================

/**
 * @brief Calculate curiosity intensity for knowledge gap
 *
 * WHAT: Compute how intensely curious we are about this gap
 * WHY:  Dopamine modulates intrinsic motivation and curiosity
 * HOW:  gap_size × baseline × dopamine_modulation
 *
 * BIOLOGY: Dopamine encodes reward prediction and motivation
 *          Higher DA → more curious and exploratory
 *
 * Complexity: O(1)
 */
static float calculate_curiosity_intensity(const curiosity_engine_t engine, float gap_size)
{
    float base_intensity = gap_size * engine->baseline_curiosity;

    // Modulate by dopamine if brain available
    if (engine->gap_detector) {
        neuromodulator_system_t neuromod = brain_get_neuromodulator_system(engine->gap_detector);
        if (neuromod) {
            float dopamine = neuromodulator_get_level(neuromod, NEUROMOD_DOPAMINE);
            // Dopamine range [0.3, 0.7], map to modulation [0.6, 1.4]
            // Low DA (0.3) → 0.6× curiosity (apathy, depression)
            // High DA (0.7) → 1.4× curiosity (motivated, manic)
            float modulation = 0.6f + (dopamine - 0.3f) * 2.0f;
            base_intensity *= modulation;
        }
    }

    return fminf(base_intensity, 1.0f);  // Clamp to [0, 1]
}

/**
 * @brief Get familiarity with concept_str
 *
 * Uses hash table for O(1) average case lookup.
 * Returns 0.0 for unknown concepts.
 *
 * Time Complexity: O(1) average case
 *
 * @param engine Curiosity engine
 * @param concept_str Concept to check
 * @return Familiarity score [0.0-1.0]
 */
float curiosity_check_familiarity(curiosity_engine_t engine, const char* concept_str)
{
    if (!engine || !concept_str) {
        return 0.0f;
    }

    concept_bucket_t* bucket = find_concept_bucket(engine, concept_str);
    if (!bucket) {
        return 0.0f;
    }

    return bucket->familiarity;
}

/**
 * @brief Count related concepts for a topic
 *
 * Uses hash table for O(1) concept_str lookup.
 *
 * Complexity: O(1) average case
 */
static uint32_t count_related_concepts(curiosity_engine_t engine, const char* concept_str)
{
    concept_bucket_t* bucket = find_concept_bucket(engine, concept_str);
    if (!bucket) {
        return 0;
    }
    return bucket->num_related;
}

/**
 * @brief Detect knowledge gap for concept_str
 *
 * Analyzes how much is unknown about a concept_str and how valuable
 * learning it would be. Uses stage strategy pattern to calculate
 * learning potential appropriate for developmental stage.
 *
 * Time Complexity: O(1) average case (hash lookup + strategy dispatch)
 *
 * @param engine Curiosity engine
 * @param concept_str Concept to analyze
 * @return Knowledge gap analysis
 */
knowledge_gap_t curiosity_detect_knowledge_gap(curiosity_engine_t engine, const char* concept_str)
{
    knowledge_gap_t gap = {0};

    if (!engine || !concept_str) {
        return gap;
    }

    strncpy(gap.topic, concept_str, sizeof(gap.topic) - 1);

    float familiarity = curiosity_check_familiarity(engine, concept_str);
    gap.gap_size = 1.0f - familiarity;
    gap.curiosity_intensity = calculate_curiosity_intensity(engine, gap.gap_size);

    // Use stage strategy for learning potential
    gap.learning_potential =
        engine->stage_strategy->calculate_learning_potential(concept_str, gap.gap_size);

    gap.related_concepts = count_related_concepts(engine, concept_str);

    return gap;
}

/**
 * @brief Get related concepts for a topic
 *
 * Retrieves concepts that are associated with the given concept_str.
 * Uses hash table for O(1) concept_str lookup.
 *
 * Time Complexity: O(min(k, max_related)) where k is number of related concepts
 *
 * @param engine Curiosity engine
 * @param concept_str Concept to query
 * @param related Output array for related concept_str strings
 * @param max_related Maximum concepts to return
 * @return Number of related concepts found
 */
uint32_t curiosity_get_related_concepts(curiosity_engine_t engine, const char* concept_str,
                                        char** related, uint32_t max_related)
{
    if (!engine || !concept_str) {
        return 0;
    }

    concept_bucket_t* bucket = find_concept_bucket(engine, concept_str);
    if (!bucket) {
        return 0;
    }

    if (!related) {
        return bucket->num_related;
    }

    uint32_t count = (bucket->num_related < max_related) ? bucket->num_related : max_related;

    for (uint32_t i = 0; i < count; i++) {
        related[i] = bucket->related_concepts[i];
    }

    return count;
}

//=============================================================================
// Question Generation
//=============================================================================

/**
 * @brief Generate single question using strategy pattern
 *
 * Uses strategy dispatch table to eliminate switch statements.
 * Each question type has its own generation function.
 *
 * Complexity: O(1) strategy lookup + O(n) string formatting
 */
static bool generate_single_question(const knowledge_gap_t* gap, question_type_t type,
                                     generated_question_t* output)
{
    if (!gap || !output) {
        return false;
    }

    const question_strategy_t* strategy = get_question_strategy(type);
    if (!strategy) {
        return false;
    }

    memset(output, 0, sizeof(generated_question_t));
    output->type = type;

    float priority_mult = 0.0f;
    float difficulty = 0.0f;

    strategy->generator(gap->topic, output->question, sizeof(output->question), &priority_mult,
                        &difficulty);

    output->priority = gap->curiosity_intensity * priority_mult;
    output->difficulty = difficulty;

    return true;
}

/**
 * @brief Generate questions for knowledge gap
 *
 * Creates multiple questions appropriate for current learning stage.
 * Uses stage strategy to determine which question types to generate.
 * Uses question strategy pattern to generate each question.
 *
 * Time Complexity: O(k) where k is number of question types for stage
 *
 * @param engine Curiosity engine
 * @param gap Knowledge gap to question
 * @param questions Output array for generated questions
 * @param max_questions Maximum questions to generate
 * @return Number of questions generated
 */
uint32_t curiosity_generate_questions(curiosity_engine_t engine, const knowledge_gap_t* gap,
                                      generated_question_t* questions, uint32_t max_questions)
{
    if (!engine || !gap || !questions || max_questions == 0) {
        return 0;
    }

    const question_type_t* types = engine->stage_strategy->get_question_types();
    uint32_t num_types = engine->stage_strategy->get_question_types_count();
    uint32_t generated = 0;

    for (uint32_t i = 0; i < num_types && generated < max_questions; i++) {
        if (generate_single_question(gap, types[i], &questions[generated])) {
            generated++;
        }
    }

    return generated;
}

/**
 * @brief Generate follow-up question based on previous answer
 *
 * In full implementation, would use NLP to extract key concepts
 * and generate relevant follow-ups. Currently returns generic prompt.
 *
 * Complexity: O(1)
 *
 * @param engine Curiosity engine
 * @param previous_answer Answer to follow up on
 * @return Follow-up question string
 */
const char* curiosity_generate_followup(curiosity_engine_t engine, const char* previous_answer)
{
    if (!engine || !previous_answer) {
        return NULL;
    }

    static char followup[512];
    snprintf(followup, sizeof(followup), "Can you tell me more about that?");

    return followup;
}

//=============================================================================
// Motivation Assessment
//=============================================================================

/**
 * @brief Calculate weighted motivation score
 *
 * Combines multiple motivation factors with empirically-determined weights.
 *
 * Complexity: O(1)
 */
static float calculate_overall_motivation(const motivation_state_t* state)
{
    return (state->intrinsic_curiosity * 0.4f + state->goal_relevance * 0.2f +
            state->social_importance * 0.1f + state->survival_value * 0.2f +
            state->aesthetic_appeal * 0.1f);
}

/**
 * @brief Calculate aesthetic appeal based on novelty
 *
 * Novel concepts are more aesthetically appealing.
 *
 * Complexity: O(1)
 */
static float calculate_aesthetic_appeal(float familiarity)
{
    return (1.0f - familiarity) * 0.8f;
}

/**
 * @brief Assess motivation to learn concept_str
 *
 * Evaluates multiple factors that drive learning:
 * - Intrinsic curiosity (baseline trait)
 * - Goal relevance (alignment with objectives)
 * - Social importance (what others value)
 * - Survival value (safety/wellbeing)
 * - Aesthetic appeal (novelty/interest)
 *
 * Time Complexity: O(1) average case (hash lookup)
 *
 * @param engine Curiosity engine
 * @param concept_str Concept to assess
 * @return Motivation state breakdown
 */
motivation_state_t curiosity_assess_motivation(curiosity_engine_t engine, const char* concept_str)
{
    motivation_state_t state = {0};

    if (!engine || !concept_str) {
        return state;
    }

    state.intrinsic_curiosity = engine->baseline_curiosity;

    // DOPAMINE MODULATION: High DA enhances intrinsic motivation
    if (engine->gap_detector) {
        neuromodulator_system_t neuromod = brain_get_neuromodulator_system(engine->gap_detector);
        if (neuromod) {
            float dopamine = neuromodulator_get_level(neuromod, NEUROMOD_DOPAMINE);
            // Dopamine amplifies intrinsic curiosity
            state.intrinsic_curiosity *= (0.5f + dopamine);  // 0.8-1.2× modulation
        }
    }

    state.goal_relevance = 0.3f;     // Simplified - would check actual goals
    state.social_importance = 0.4f;  // Simplified - would check social context
    state.survival_value = 0.1f;     // Most concepts have low survival value

    float familiarity = curiosity_check_familiarity(engine, concept_str);
    state.aesthetic_appeal = calculate_aesthetic_appeal(familiarity);
    state.overall_motivation = calculate_overall_motivation(&state);

    return state;
}

/**
 * @brief Set baseline curiosity level
 *
 * Adjusts intrinsic motivation to learn. Clamped to [0.0, 1.0].
 *
 * Complexity: O(1)
 */
void curiosity_set_baseline(curiosity_engine_t engine, float level)
{
    if (!engine) {
        return;
    }

    engine->baseline_curiosity = fminf(fmaxf(level, 0.0f), 1.0f);
    engine->current_motivation = engine->baseline_curiosity;
}

/**
 * @brief Get current learning drive
 *
 * Returns current motivation level.
 *
 * Complexity: O(1)
 */
float curiosity_get_drive(curiosity_engine_t engine)
{
    if (!engine) {
        return 0.0f;
    }
    return engine->current_motivation;
}

//=============================================================================
// Incremental Learning
//=============================================================================

/**
 * @brief Record question in history
 *
 * Adds question-answer pair to learning history.
 * Used for tracking progress and preventing repetition.
 *
 * Complexity: O(1) assuming capacity available
 */
static bool record_question_history(curiosity_engine_t engine, const char* question,
                                    const char* answer)
{
    if (engine->num_questions >= engine->questions_capacity) {
        return false;
    }

    question_history_t* q = &engine->questions[engine->num_questions++];
    strncpy(q->question, question, sizeof(q->question) - 1);
    strncpy(q->answer, answer, sizeof(q->answer) - 1);
    q->answered = true;
    q->learning_value = 0.5f;
    q->timestamp = 0;

    return true;
}

/**
 * @brief Learn from question-answer pair
 *
 * Records answer and updates internal knowledge representation.
 * In full implementation, would extract concepts and update familiarity.
 *
 * Time Complexity: O(1) for recording
 *
 * @param engine Curiosity engine
 * @param question Question that was asked
 * @param answer Answer received
 * @return true on success
 */
bool curiosity_learn_answer(curiosity_engine_t engine, const char* question, const char* answer)
{
    if (!engine || !question || !answer) {
        return false;
    }

    if (!record_question_history(engine, question, answer)) {
        return false;
    }

    engine->progress.total_answers_learned++;

    // INTRINSIC REWARD: Release dopamine for learning
    // BIOLOGY: Learning triggers dopamine release in reward circuits
    if (engine->gap_detector) {
        neuromodulator_system_t neuromod = brain_get_neuromodulator_system(engine->gap_detector);
        if (neuromod) {
            // Release dopamine proportional to curiosity
            float reward = 0.3f;  // Moderate intrinsic reward for learning
            neuromodulator_release_dopamine(neuromod, reward, 0.0f);
        }
    }

    return true;
}

/**
 * @brief Learn from direct experience
 *
 * Processes sensory data and associates with concepts.
 * Enables multi-modal learning (vision, audio, etc.).
 *
 * Complexity: O(1) for recording
 *
 * @param engine Curiosity engine
 * @param experience_description Text description of experience
 * @param sensory_data Feature vector from sensors
 * @param num_features Size of feature vector
 * @return true on success
 */
bool curiosity_learn_experience(curiosity_engine_t engine, const char* experience_description,
                                const float* sensory_data, uint32_t num_features)
{
    if (!engine || !experience_description) {
        return false;
    }

    engine->progress.total_experiences++;

    // INTRINSIC REWARD: Release dopamine for experiential learning
    // BIOLOGY: Novel experiences trigger dopamine release
    if (engine->gap_detector) {
        neuromodulator_system_t neuromod = brain_get_neuromodulator_system(engine->gap_detector);
        if (neuromod) {
            float reward = 0.4f;  // Higher reward for direct experience
            neuromodulator_release_dopamine(neuromod, reward, 0.0f);
        }
    }

    return true;
}

/**
 * @brief Learn from observation
 *
 * Social learning from watching others.
 * Updates concept_str knowledge based on observed interactions.
 *
 * Complexity: O(1)
 *
 * @param engine Curiosity engine
 * @param what_observed What was observed
 * @param context Context of observation
 * @return true on success
 */
bool curiosity_learn_observation(curiosity_engine_t engine, const char* what_observed,
                                 const char* context)
{
    if (!engine || !what_observed) {
        return false;
    }

    // Would update concept_str familiarity and relationships
    return true;
}

//=============================================================================
// Knowledge Source Management
//=============================================================================

/**
 * @brief Expand knowledge source array if needed
 *
 * Doubles capacity when full.
 *
 * Complexity: O(n) when resizing, amortized O(1)
 */
static bool ensure_source_capacity(curiosity_engine_t engine)
{
    if (engine->num_sources < engine->sources_capacity) {
        return true;
    }

    engine->sources_capacity *= 2;
    knowledge_source_t* new_sources =
        nimcp_realloc(engine->sources, engine->sources_capacity * sizeof(knowledge_source_t));

    if (!new_sources) {
        return false;
    }

    engine->sources = new_sources;
    return true;
}

/**
 * @brief Register external knowledge source
 *
 * Adds searchable knowledge source (e.g., database, API, documents).
 * Engine will query registered sources when seeking knowledge.
 *
 * Time Complexity: O(1) amortized
 *
 * @param engine Curiosity engine
 * @param source_name Name of source
 * @param search_fn Function to search this source
 * @param context Opaque context for search function
 * @return true on success
 */
bool curiosity_register_knowledge_source(curiosity_engine_t engine, const char* source_name,
                                         knowledge_search_fn_t search_fn, void* context)
{
    if (!engine || !source_name || !search_fn) {
        return false;
    }

    if (!ensure_source_capacity(engine)) {
        return false;
    }

    knowledge_source_t* source = &engine->sources[engine->num_sources++];
    strncpy(source->name, source_name, sizeof(source->name) - 1);
    source->search_fn = search_fn;
    source->context = context;
    source->active = true;

    return true;
}

/**
 * @brief Query single knowledge source
 *
 * Executes search on one source and collects results.
 *
 * Complexity: O(k) where k is results from this source
 */
static uint32_t query_knowledge_source(const knowledge_source_t* source, const char* topic,
                                       char** results, uint32_t max_results, uint32_t current_count)
{
    if (!source->active) {
        return 0;
    }

    uint32_t num_found = 0;
    char** source_results =
        source->search_fn(topic, source->context, max_results - current_count, &num_found);

    if (!source_results) {
        return 0;
    }

    uint32_t copied = 0;
    for (uint32_t i = 0; i < num_found && current_count + copied < max_results; i++) {
        results[current_count + copied] = source_results[i];
        copied++;
    }

    return copied;
}

/**
 * @brief Search all knowledge sources for information
 *
 * Queries all registered sources and aggregates results.
 * Sources are queried sequentially until result limit reached.
 *
 * Time Complexity: O(s * k) where s is sources, k is results per source
 *
 * @param engine Curiosity engine
 * @param gap Knowledge gap to fill
 * @param results Output array for search results
 * @param max_results Maximum results to return
 * @return Number of results found
 */
uint32_t curiosity_seek_knowledge(curiosity_engine_t engine, const knowledge_gap_t* gap,
                                  char** results, uint32_t max_results)
{
    if (!engine || !gap || !results) {
        return 0;
    }

    uint32_t total_results = 0;

    for (uint32_t i = 0; i < engine->num_sources && total_results < max_results; i++) {
        uint32_t found = query_knowledge_source(&engine->sources[i], gap->topic, results,
                                                max_results, total_results);
        total_results += found;
    }

    return total_results;
}

//=============================================================================
// Learning Progress Tracking
//=============================================================================

/**
 * @brief Update computed progress fields
 *
 * Refreshes derived statistics before returning progress.
 *
 * Complexity: O(1)
 */
static void update_progress_statistics(curiosity_engine_t engine, learning_progress_t* progress)
{
    progress->concepts_learned = engine->total_concepts;
    progress->avg_curiosity = engine->baseline_curiosity;
}

/**
 * @brief Get learning progress statistics
 *
 * Returns comprehensive learning metrics.
 *
 * Complexity: O(1)
 *
 * @param engine Curiosity engine
 * @param progress Output structure for progress data
 * @return true on success
 */
bool curiosity_get_progress(curiosity_engine_t engine, learning_progress_t* progress)
{
    if (!engine || !progress) {
        return false;
    }

    *progress = engine->progress;
    update_progress_statistics(engine, progress);

    return true;
}

/**
 * @brief Check if concept_str belongs to domain
 *
 * Simplified domain matching via substring search.
 * Full implementation would use proper ontology/taxonomy.
 *
 * Complexity: O(n * m) where n is concept_str length, m is domain length
 */
static bool concept_in_domain(const char* concept_str, const char* domain)
{
    return strstr(concept_str, domain) != NULL;
}

/**
 * @brief Count concepts in domain
 *
 * Traverses hash table counting domain-related concepts.
 *
 * Complexity: O(n) where n is total concepts
 */
/**
 * @brief Context for count_domain_concepts iteration
 */
typedef struct {
    const char* domain;
    uint32_t count;
} count_domain_context_t;

/**
 * @brief Iterator callback for counting domain concepts
 */
static bool count_domain_callback(const void* key, size_t key_size, void* value, size_t value_size,
                                  void* user_data)
{
    concept_bucket_t* bucket = (concept_bucket_t*) value;
    count_domain_context_t* ctx = (count_domain_context_t*) user_data;

    if (concept_in_domain(bucket->concept, ctx->domain)) {
        ctx->count++;
    }

    return true;  // Continue iteration
}

static uint32_t count_domain_concepts(curiosity_engine_t engine, const char* domain)
{
    if (!engine || !engine->concept_hash_table) {
        return 0;
    }

    count_domain_context_t ctx = {.domain = domain, .count = 0};

    hash_table_iterate(engine->concept_hash_table, count_domain_callback, &ctx);

    return ctx.count;
}

/**
 * @brief Estimate domain coverage
 *
 * Calculates what proportion of domain concepts are known.
 * Uses heuristic estimate of domain size (100 concepts).
 *
 * Time Complexity: O(n) where n is total concepts
 *
 * @param engine Curiosity engine
 * @param domain Domain to assess
 * @return Coverage fraction [0.0-1.0]
 */
float curiosity_get_domain_coverage(curiosity_engine_t engine, const char* domain)
{
    if (!engine || !domain) {
        return 0.0f;
    }

    uint32_t domain_concepts = count_domain_concepts(engine, domain);
    return fminf((float) domain_concepts / 100.0f, 1.0f);
}

//=============================================================================
// Developmental Stage Management
//=============================================================================

/**
 * @brief Get current learning stage
 *
 * Returns developmental stage (infant, toddler, etc.).
 *
 * Complexity: O(1)
 */
learning_stage_t curiosity_get_stage(curiosity_engine_t engine)
{
    if (!engine) {
        return STAGE_INFANT;
    }
    return engine->stage;
}

/**
 * @brief Set learning stage and update strategy
 *
 * Changes developmental stage and switches to appropriate strategy.
 * Updates baseline curiosity based on stage characteristics.
 * Uses state pattern to modify all stage-dependent behaviors.
 *
 * Time Complexity: O(1)
 *
 * @param engine Curiosity engine
 * @param stage New learning stage
 */
void curiosity_set_stage(curiosity_engine_t engine, learning_stage_t stage)
{
    if (!engine) {
        return;
    }

    engine->stage = stage;
    engine->stage_strategy = get_stage_strategy(stage);
    engine->baseline_curiosity = engine->stage_strategy->get_baseline_curiosity();
    engine->current_motivation = engine->baseline_curiosity;
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Print knowledge gap analysis
 *
 * Debugging utility to display gap information.
 *
 * Complexity: O(1)
 */
void curiosity_print_gap(const knowledge_gap_t* gap)
{
    if (!gap) {
        return;
    }

    printf("Knowledge Gap:\n");
    printf("  Topic: %s\n", gap->topic);
    printf("  Gap size: %.2f\n", gap->gap_size);
    printf("  Curiosity intensity: %.2f\n", gap->curiosity_intensity);
    printf("  Learning potential: %.2f\n", gap->learning_potential);
    printf("  Related concepts: %u\n", gap->related_concepts);
}

/**
 * @brief Print generated question
 *
 * Debugging utility to display question information.
 *
 * Complexity: O(1)
 */
void curiosity_print_question(const generated_question_t* question)
{
    if (!question) {
        return;
    }

    printf("Question:\n");
    printf("  Text: %s\n", question->question);
    printf("  Type: %d\n", question->type);
    printf("  Priority: %.2f\n", question->priority);
    printf("  Difficulty: %.2f\n", question->difficulty);
}

/**
 * @brief Print learning progress
 *
 * Debugging utility to display progress statistics.
 *
 * Complexity: O(1)
 */
void curiosity_print_progress(const learning_progress_t* progress)
{
    if (!progress) {
        return;
    }

    printf("Learning Progress:\n");
    printf("  Questions asked: %lu\n", progress->total_questions_asked);
    printf("  Answers learned: %lu\n", progress->total_answers_learned);
    printf("  Experiences: %lu\n", progress->total_experiences);
    printf("  Concepts learned: %lu\n", progress->concepts_learned);
    printf("  Knowledge growth rate: %.3f\n", progress->knowledge_growth_rate);
    printf("  Average curiosity: %.2f\n", progress->avg_curiosity);
    printf("  Current focus: %s\n", progress->current_focus);
}

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

/**
 * @brief Set exploration rate based on cognitive load
 *
 * WHAT: Adjust exploration vs exploitation based on executive load
 * WHY:  Prevent exploration when cognitively overloaded
 * HOW:  Modulate curiosity intensity and baseline
 *
 * BIOLOGY: Prefrontal cortex regulates exploration/exploitation trade-off
 *          High load → exploit known strategies (lower exploration)
 *          Low load → explore novel options (higher exploration)
 *
 * COMPLEXITY: O(1)
 *
 * @param engine Curiosity engine
 * @param exploration_rate Exploration rate [0, 1] (0=exploit, 1=explore)
 */
void curiosity_set_exploration_rate(curiosity_engine_t engine, float exploration_rate)
{
    // Guard: NULL engine
    if (!engine) {
        return;
    }

    // Clamp exploration rate to [0, 1]
    exploration_rate = fminf(fmaxf(exploration_rate, 0.0f), 1.0f);

    // WHAT: Modulate baseline curiosity based on exploration rate
    // WHY:  Higher exploration rate → more curious
    // HOW:  Scale baseline by exploration rate
    engine->baseline_curiosity = engine->stage_strategy->get_baseline_curiosity() * exploration_rate;
    engine->current_motivation = engine->baseline_curiosity;
}

/**
 * @brief Get information gain from recent learning
 *
 * WHAT: Query expected information gain from exploring
 * WHY:  Executive can prioritize exploratory tasks
 * HOW:  Estimate gain from recent learning progress
 *
 * COMPLEXITY: O(1)
 *
 * @param engine Curiosity engine
 * @return Information gain [0, 1]
 */
float curiosity_get_information_gain(curiosity_engine_t engine)
{
    // Guard: NULL engine
    if (!engine) {
        return 0.0f;
    }

    // WHAT: Estimate information gain from recent learning
    // WHY:  If we're learning a lot, exploration is paying off
    // HOW:  Use knowledge growth rate as proxy for information gain
    if (engine->progress.total_questions_asked == 0) {
        return engine->baseline_curiosity;  // No data yet, use baseline
    }

    // Calculate learning rate: answers / questions
    float learning_rate = (float)engine->progress.total_answers_learned /
                         (float)engine->progress.total_questions_asked;

    // Scale by current motivation
    float information_gain = learning_rate * engine->current_motivation;

    // Clamp to [0, 1]
    return fminf(fmaxf(information_gain, 0.0f), 1.0f);
}
