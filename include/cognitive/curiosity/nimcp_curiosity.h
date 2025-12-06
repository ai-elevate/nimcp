//=============================================================================
// nimcp_curiosity.h - Curiosity-Driven Learning System
//=============================================================================

#ifndef NIMCP_CURIOSITY_H
#define NIMCP_CURIOSITY_H

#include <stdbool.h>
#include <stdint.h>
#include "core/brain/nimcp_brain.h"
#include "common/nimcp_export.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

/**
 * @file nimcp_curiosity.h
 * @brief Curiosity-driven learning system for human-like knowledge acquisition
 *
 * NIMCP Philosophy: Learn like a human infant
 * - Start with minimal knowledge
 * - Curiosity drives exploration
 * - Learn incrementally from experiences
 * - No massive pre-training required
 * - CPU-friendly, no GPUs needed
 *
 * Example usage:
 * ```c
 * // Create curiosity engine (like an infant's drive to explore)
 * curiosity_engine_t engine = curiosity_engine_create("infant_learner");
 *
 * // Infant encounters something new
 * knowledge_gap_t gap = curiosity_detect_knowledge_gap(engine, "What is a cat?");
 *
 * // Curiosity generates questions
 * const char* questions[10];
 * uint32_t num_q = curiosity_generate_questions(engine, &gap, questions, 10);
 * // Output: "What does a cat look like?", "What sounds does it make?",
 * //         "How does it move?", "What does it eat?"
 *
 * // Learn from answers (incrementally)
 * curiosity_learn_answer(engine, questions[0], "A cat is a small furry animal");
 * curiosity_learn_answer(engine, questions[1], "Cats meow and purr");
 *
 * // Knowledge accumulates like an infant learning
 * ```
 */

//=============================================================================
// Core Types
//=============================================================================

/**
 * @brief Curiosity engine handle
 */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct curiosity_engine_struct* curiosity_engine_t;

/**
 * @brief Knowledge gap detection
 */
typedef struct {
    char topic[256];           /**< What we don't know about */
    float gap_size;            /**< How much we don't know (0-1) */
    float curiosity_intensity; /**< How curious we are about it (0-1) */
    float learning_potential;  /**< Expected learning value (0-1) */
    uint32_t related_concepts; /**< Number of related known concepts */
    char** prerequisites;      /**< What we need to know first */
    uint32_t num_prerequisites;
} knowledge_gap_t;

/**
 * @brief Question types for knowledge seeking
 */
typedef enum {
    QUESTION_WHAT,  /**< "What is X?" - definitional */
    QUESTION_WHY,   /**< "Why does X happen?" - causal */
    QUESTION_HOW,   /**< "How does X work?" - mechanistic */
    QUESTION_WHEN,  /**< "When did X happen?" - temporal */
    QUESTION_WHERE, /**< "Where is X?" - spatial */
    QUESTION_WHO,   /**< "Who did X?" - agentive */
    QUESTION_WHICH  /**< "Which X?" - discriminative */
} question_type_t;

/**
 * @brief Generated question for knowledge seeking
 */
typedef struct {
    char question[512];   /**< The question text */
    question_type_t type; /**< Question type */
    float priority;       /**< How important to answer (0-1) */
    float difficulty;     /**< Expected difficulty (0-1) */
    char** search_terms;  /**< Suggested search terms */
    uint32_t num_search_terms;
} generated_question_t;

/**
 * @brief Learning motivation sources
 */
typedef enum {
    MOTIVATION_INTRINSIC, /**< Pure curiosity */
    MOTIVATION_GOAL,      /**< Needed for a goal */
    MOTIVATION_SOCIAL,    /**< Others find it important */
    MOTIVATION_SURVIVAL,  /**< Needed for safety/wellbeing */
    MOTIVATION_AESTHETIC  /**< Beautiful/interesting */
} motivation_source_t;

/**
 * @brief Learning motivation state
 */
typedef struct {
    float intrinsic_curiosity; /**< Pure desire to know (0-1) */
    float goal_relevance;      /**< Helps achieve goals (0-1) */
    float social_importance;   /**< Others value it (0-1) */
    float survival_value;      /**< Survival/safety importance (0-1) */
    float aesthetic_appeal;    /**< Beauty/interest (0-1) */
    float overall_motivation;  /**< Combined motivation (0-1) */
} motivation_state_t;

//=============================================================================
// Curiosity Engine Creation
//=============================================================================

/**
 * @brief Create curiosity engine
 *
 * BREAKING CHANGE: Now requires parent brain reference (module pattern)
 * Curiosity is a cognitive module that uses the parent brain's neuromodulator system.
 *
 * @param parent_brain Parent brain that owns this curiosity module
 * @param learner_name Name for this learner (e.g., "infant", "student")
 * @return Curiosity engine or NULL on error
 */
curiosity_engine_t curiosity_engine_create(brain_t parent_brain, const char* learner_name);

/**
 * @brief Destroy curiosity engine
 *
 * @param engine Engine to destroy
 */
void curiosity_engine_destroy(curiosity_engine_t engine);

//=============================================================================
// Knowledge Gap Detection
//=============================================================================

/**
 * @brief Detect knowledge gap
 *
 * When encountering new information, assess what we don't know
 *
 * @param engine Curiosity engine
 * @param concept Concept or topic encountered
 * @return Knowledge gap assessment
 */
knowledge_gap_t curiosity_detect_knowledge_gap(curiosity_engine_t engine, const char* concept_str);

/**
 * @brief Check if concept is known
 *
 * @param engine Curiosity engine
 * @param concept Concept to check
 * @return Familiarity score (0=unknown, 1=fully known)
 */
float curiosity_check_familiarity(curiosity_engine_t engine, const char* concept_str);

/**
 * @brief Get related concepts
 *
 * Find concepts related to current knowledge gap
 *
 * @param engine Curiosity engine
 * @param concept Central concept
 * @param related Output array for related concepts
 * @param max_related Maximum concepts to return
 * @return Number of related concepts found
 */
uint32_t curiosity_get_related_concepts(curiosity_engine_t engine, const char* concept_str,
                                        char** related, uint32_t max_related);

//=============================================================================
// Question Generation
//=============================================================================

/**
 * @brief Generate questions about knowledge gap
 *
 * Like an infant asking "What's that?" "Why?" "How?"
 *
 * @param engine Curiosity engine
 * @param gap Knowledge gap to explore
 * @param questions Output array for questions
 * @param max_questions Maximum questions to generate
 * @return Number of questions generated
 */
uint32_t curiosity_generate_questions(curiosity_engine_t engine, const knowledge_gap_t* gap,
                                      generated_question_t* questions, uint32_t max_questions);

/**
 * @brief Generate follow-up question
 *
 * After learning something, ask deeper questions
 *
 * @param engine Curiosity engine
 * @param previous_answer Previous answer learned
 * @return Follow-up question or NULL
 */
const char* curiosity_generate_followup(curiosity_engine_t engine, const char* previous_answer);

//=============================================================================
// Motivation & Drive
//=============================================================================

/**
 * @brief Assess learning motivation
 *
 * Why do we want to learn this? (intrinsic, goal, social, survival)
 *
 * @param engine Curiosity engine
 * @param concept Concept to learn
 * @return Motivation state
 */
motivation_state_t curiosity_assess_motivation(curiosity_engine_t engine, const char* concept_str);

/**
 * @brief Set intrinsic curiosity level
 *
 * Adjust the baseline drive to explore (like infant curiosity)
 *
 * @param engine Curiosity engine
 * @param level Curiosity level (0-1, typically 0.7-0.9 for infants)
 */
void curiosity_set_baseline(curiosity_engine_t engine, float level);

/**
 * @brief Get current motivation to learn
 *
 * @param engine Curiosity engine
 * @return Current overall motivation (0-1)
 */
float curiosity_get_drive(curiosity_engine_t engine);

//=============================================================================
// Incremental Learning
//=============================================================================

/**
 * @brief Learn from answer to question
 *
 * Incremental learning like an infant acquiring knowledge piece by piece
 *
 * @param engine Curiosity engine
 * @param question Question that was asked
 * @param answer Answer received
 * @return true on success
 */
bool curiosity_learn_answer(curiosity_engine_t engine, const char* question, const char* answer);

/**
 * @brief Learn from experience
 *
 * Learn from direct interaction (not just text)
 *
 * @param engine Curiosity engine
 * @param experience_description Description of experience
 * @param sensory_data Optional sensory features
 * @param num_features Number of features
 * @return true on success
 */
bool curiosity_learn_experience(curiosity_engine_t engine, const char* experience_description,
                                const float* sensory_data, uint32_t num_features);

/**
 * @brief Learn from observation
 *
 * Learn by watching (social learning)
 *
 * @param engine Curiosity engine
 * @param what_observed What was observed
 * @param context Context of observation
 * @return true on success
 */
bool curiosity_learn_observation(curiosity_engine_t engine, const char* what_observed,
                                 const char* context);

//=============================================================================
// Knowledge Search & Acquisition
//=============================================================================

/**
 * @brief Search callback for external knowledge sources
 *
 * Function signature for searching external sources (web, books, etc.)
 */
typedef char** (*knowledge_search_fn_t)(const char* query, void* context, uint32_t max_results,
                                        uint32_t* num_results);

/**
 * @brief Register knowledge source
 *
 * Connect to external knowledge (Wikipedia, books, web, etc.)
 *
 * @param engine Curiosity engine
 * @param source_name Name of knowledge source
 * @param search_fn Search function callback
 * @param context Context for search function
 * @return true on success
 */
bool curiosity_register_knowledge_source(curiosity_engine_t engine, const char* source_name,
                                         knowledge_search_fn_t search_fn, void* context);

/**
 * @brief Seek knowledge actively
 *
 * Given a knowledge gap, actively search for answers
 *
 * @param engine Curiosity engine
 * @param gap Knowledge gap to fill
 * @param results Output array for search results
 * @param max_results Maximum results
 * @return Number of results found
 */
uint32_t curiosity_seek_knowledge(curiosity_engine_t engine, const knowledge_gap_t* gap,
                                  char** results, uint32_t max_results);

//=============================================================================
// Learning Progress
//=============================================================================

/**
 * @brief Learning progress tracking
 */
typedef struct {
    uint64_t total_questions_asked;
    uint64_t total_answers_learned;
    uint64_t total_experiences;
    uint64_t concepts_learned;
    float knowledge_growth_rate; /**< Recent learning rate */
    float avg_curiosity;         /**< Average curiosity level */
    char current_focus[256];     /**< Current learning focus */
    bool enable_bio_async;       /**< Enable bio-async communication */
} learning_progress_t;

/**
 * @brief Get learning progress
 *
 * Track how much has been learned (like developmental milestones)
 *
 * @param engine Curiosity engine
 * @param progress Output progress structure
 * @return true on success
 */
bool curiosity_get_progress(curiosity_engine_t engine, learning_progress_t* progress);

/**
 * @brief Get knowledge domain coverage
 *
 * How much is known in different domains (literature, science, art, etc.)
 *
 * @param engine Curiosity engine
 * @param domain Domain name (e.g., "literature", "science", "ethics")
 * @return Coverage percentage (0-1)
 */
float curiosity_get_domain_coverage(curiosity_engine_t engine, const char* domain);

//=============================================================================
// Developmental Stages
//=============================================================================

/**
 * @brief Learning stage (like human development)
 */
typedef enum {
    STAGE_INFANT,     /**< Basic concepts, sensory learning */
    STAGE_TODDLER,    /**< Language acquisition, basic reasoning */
    STAGE_CHILD,      /**< Structured learning, reading, writing */
    STAGE_ADOLESCENT, /**< Abstract reasoning, complex domains */
    STAGE_ADULT,      /**< Specialized knowledge, expertise */
    STAGE_EXPERT      /**< Deep expertise in domains */
} learning_stage_t;

/**
 * @brief Get current learning stage
 *
 * @param engine Curiosity engine
 * @return Current developmental stage
 */
learning_stage_t curiosity_get_stage(curiosity_engine_t engine);

/**
 * @brief Set learning stage constraints
 *
 * Limit learning to age-appropriate concepts
 *
 * @param engine Curiosity engine
 * @param stage Target learning stage
 */
void curiosity_set_stage(curiosity_engine_t engine, learning_stage_t stage);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Print knowledge gap summary
 */
void curiosity_print_gap(const knowledge_gap_t* gap);

/**
 * @brief Print generated question
 */
void curiosity_print_question(const generated_question_t* question);

/**
 * @brief Print learning progress
 */
void curiosity_print_progress(const learning_progress_t* progress);

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

/**
 * @brief Set exploration rate based on cognitive load
 *
 * WHAT: Adjust exploration vs exploitation based on executive load
 * WHY:  Prevent exploration when cognitively overloaded
 * HOW:  Modulate curiosity intensity and exploration rate
 *
 * BIOLOGY: Prefrontal cortex regulates exploration/exploitation trade-off
 *
 * @param engine Curiosity engine
 * @param exploration_rate Exploration rate [0, 1] (0=exploit, 1=explore)
 */
void curiosity_set_exploration_rate(curiosity_engine_t engine, float exploration_rate);

/**
 * @brief Get information gain from recent learning
 *
 * WHAT: Query expected information gain from exploring
 * WHY:  Executive can prioritize exploratory tasks
 * HOW:  Return recent learning value
 *
 * @param engine Curiosity engine
 * @return Information gain [0, 1]
 */
float curiosity_get_information_gain(curiosity_engine_t engine);


#ifdef __cplusplus
}
#endif
#endif  // NIMCP_CURIOSITY_H
