//=============================================================================
// nimcp_knowledge.c - Multi-Domain Knowledge Acquisition Implementation
// REFACTORED: Using Strategy Pattern, Repository Pattern, and Search Indices
//=============================================================================

#include "cognitive/knowledge/nimcp_knowledge.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_unified_memory.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/brain/nimcp_brain.h"
#include "cognitive/curiosity/nimcp_curiosity.h"
#include "cognitive/nimcp_symbolic_logic.h"
#include "utils/memory/nimcp_memory.h"  // CRITICAL: Declares nimcp_calloc/nimcp_free return types
#include "utils/containers/nimcp_btree.h"

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "async/nimcp_bio_router.h"
#include "utils/logging/nimcp_logging.h"
#include "nimcp.h"  // For error codes

#define LOG_MODULE "knowledge"

//=============================================================================
// Constants
//=============================================================================

#define HASH_TABLE_SIZE 4096
#define MAX_CONCEPT_LENGTH 256
#define MAX_DEFINITION_LENGTH 1024
#define INITIAL_CAPACITY 10000
#define CONFIDENCE_INCREMENT 0.1f
#define CONFIDENCE_MAX 1.0f

//=============================================================================
// Internal Data Structures with Documented Invariants
//=============================================================================

/**
 * @brief Reading progress tracking structure
 *
 * Invariants:
 * - current_page <= total_pages
 * - comprehension_score in range [0.0, 1.0]
 * - book_title is null-terminated
 */
typedef struct {
    char book_title[256];
    uint32_t current_page;
    uint32_t total_pages;
    float comprehension_score;
} reading_progress_t;

/**
 * @brief Hash table entry for O(log n) concept_str lookup
 *
 * Implements chaining for collision resolution.
 * Invariants:
 * - index points to valid item in knowledge_system items array
 * - next is NULL or points to valid hash_entry_t
 */
typedef struct hash_entry_struct {
    char* concept;
    uint32_t index;
    struct hash_entry_struct* next;
} hash_entry_t;

/**
 * @brief Hash table for fast concept_str lookup
 *
 * Provides O(1) average case lookup instead of O(n) linear search.
 * Invariants:
 * - size equals HASH_TABLE_SIZE
 * - entries is array of HASH_TABLE_SIZE pointers
 */
typedef struct {
    hash_entry_t** entries;
    uint32_t size;
} knowledge_hash_table_t;

/**
 * @brief Strategy interface for domain-specific learning
 *
 * Strategy Pattern: Different learning strategies for different knowledge types
 * (narrative, art, history, etc.)
 */
typedef struct {
    const char* domain_name;
    bool (*learn)(void* system, const void* data);
    void (*optimize)(void* system);
    float (*assess)(const void* system);
} learning_strategy_t;

/**
 * @brief Repository for knowledge storage abstraction
 *
 * Repository Pattern: Abstracts storage implementation from business logic.
 * Invariants:
 * - num_items <= capacity
 * - index is synchronized with items array
 * - confidence_btree is synchronized with items array
 */
typedef struct {
    knowledge_item_t* items;
    uint32_t num_items;
    uint32_t capacity;
    knowledge_hash_table_t* index;       /* O(1) lookup by concept_str name */
    btree_t* confidence_btree;            /* O(log n) range queries by confidence */
} knowledge_repository_t;

/**
 * @brief Main knowledge system structure
 *
 * Invariants:
 * - All pointers are either NULL or point to valid allocated memory
 * - num_* counters are always <= corresponding *_capacity values
 * - domain_brains array has exactly 11 entries
 * - domain_stats array has exactly 11 entries
 */
struct knowledge_system_struct {
    char learner_name[128];

    knowledge_repository_t* repository;

    narrative_knowledge_t* narratives;
    uint32_t num_narratives;
    uint32_t narratives_capacity;

    aesthetic_knowledge_t* artworks;
    uint32_t num_artworks;
    uint32_t artworks_capacity;

    historical_knowledge_t* history;
    uint32_t num_history;
    uint32_t history_capacity;

    // REMOVED: brain_t domain_brains[11] - Never used, completely dead code

    reading_progress_t* reading_list;
    uint32_t num_reading;
    uint32_t reading_capacity;

    // Unified brain for knowledge system (provides curiosity module)
    // Previously created curiosity independently - now follows "one brain, many modules" pattern
    brain_t knowledge_brain;

    domain_knowledge_t domain_stats[11];

    learning_strategy_t* strategies[11];

    // Bio-async integration
    bio_module_context_t bio_ctx;   /**< Bio-async module context */
    bool bio_async_enabled;         /**< Bio-async registration status */
};

//=============================================================================
// B-TREE HELPER FUNCTIONS (from utils/containers/nimcp_btree.h)
//=============================================================================

/**
 * WHAT: Compare two confidence values for B-tree ordering
 * WHY: B-tree needs comparison function for sorting by confidence
 * HOW: Convert string confidence keys to floats for comparison
 */
static int compare_confidence(const char* key1, const char* key2)
{
    if (!key1 || !key2) {
        return 0;
    }

    // Key format is "confidence_index" (e.g., "0.300000_00123")
    // atof stops at underscore, giving us the confidence value
    float conf1 = atof(key1);
    float conf2 = atof(key2);

    if (conf1 < conf2) return -1;
    if (conf1 > conf2) return 1;

    // If confidence is equal, compare by full key (including index) for stable sorting
    return strcmp(key1, key2);
}

/**
 * WHAT: Extract confidence key from knowledge item
 * WHY: B-tree needs key extraction function
 * HOW: Format confidence as fixed-precision string
 */
static const char* extract_confidence_key(const void* data)
{
    if (!data) {
        return NULL;
    }

    const knowledge_item_t* item = (const knowledge_item_t*)data;
    return item->confidence_key;
}

/**
 * WHAT: Free function for B-tree (no-op for our items)
 * WHY: B-tree needs destructor, but we manage item memory separately
 * HOW: Do nothing - items are in repository array
 */
static void free_knowledge_item(void* data)
{
    // No-op: items stored in repository array
    (void)data;
}

//=============================================================================
// Hash Table Implementation - O(1) Average Lookup
//=============================================================================

/**
 * @brief Hash function using DJB2 algorithm
 *
 * Computes hash value for concept_str string to enable fast lookup.
 *
 * @param concept_str The concept_str string to hash (must be non-NULL)
 * @return Hash value in range [0, HASH_TABLE_SIZE-1]
 *
 * Why: Replaces O(n) linear search with O(1) average case lookup.
 * Algorithm: DJB2 provides good distribution with minimal collisions.
 */
static uint32_t hash_concept(const char* concept_str)
{
    uint32_t hash = 5381;
    int c;

    while ((c = *concept_str++)) {
        hash = ((hash << 5) + hash) + tolower(c);
    }

    return hash % HASH_TABLE_SIZE;
}

/**
 * @brief Create hash table for concept_str indexing
 *
 * Creates empty hash table with chaining for collision resolution.
 *
 * @return Newly created hash table or NULL on allocation failure
 *
 * Why: Enables O(1) average case lookups instead of O(n) linear scans.
 * Memory: Allocates HASH_TABLE_SIZE pointers initially.
 */
static knowledge_hash_table_t* knowledge_hash_table_create(void)
{
    knowledge_hash_table_t* table = nimcp_calloc(1, sizeof(knowledge_hash_table_t));
    if (!table)
        return NULL;

    table->entries = nimcp_calloc(HASH_TABLE_SIZE, sizeof(hash_entry_t*));
    if (!table->entries) {
        nimcp_free(table);
        return NULL;
    }

    table->size = HASH_TABLE_SIZE;
    return table;
}

/**
 * @brief Insert concept_str into hash table
 *
 * Adds concept_str-to-index mapping using chaining for collisions.
 *
 * @param table Hash table to insert into (must be non-NULL)
 * @param concept_str Concept name (must be non-NULL)
 * @param index Index in items array
 * @return true on success, false on allocation failure
 *
 * Why: Maintains search index for O(1) concept_str lookup.
 * Complexity: O(1) average case, O(n) worst case with many collisions.
 */
static bool knowledge_hash_table_insert(knowledge_hash_table_t* table, const char* concept_str, uint32_t index)
{
    if (!table || !concept_str)
        return false;

    uint32_t hash = hash_concept(concept_str);

    hash_entry_t* entry = nimcp_malloc(sizeof(hash_entry_t));
    if (!entry)
        return false;

    // Use nimcp_malloc instead of strdup to match nimcp_free in destroy
    size_t concept_len = strlen(concept_str);
    entry->concept = nimcp_malloc(concept_len + 1);
    if (!entry->concept) {
        nimcp_free(entry);
        return false;
    }
    strncpy(entry->concept, concept_str, concept_len + 1);
    entry->concept[concept_len] = '\0';

    entry->index = index;
    entry->next = table->entries[hash];
    table->entries[hash] = entry;

    return true;
}

/**
 * @brief Find concept_str in hash table
 *
 * Searches hash table for concept_str and returns index if found.
 *
 * @param table Hash table to search (must be non-NULL)
 * @param concept_str Concept to find (must be non-NULL)
 * @return Index in items array or -1 if not found
 *
 * Why: O(1) average case lookup vs O(n) linear search.
 * Example: Finding "democracy" in 10000 items takes ~1 comparison vs 5000 average.
 */
static int32_t knowledge_hash_table_find(knowledge_hash_table_t* table, const char* concept_str)
{
    if (!table || !concept_str)
        return -1;

    uint32_t hash = hash_concept(concept_str);
    hash_entry_t* entry = table->entries[hash];

    while (entry) {
        if (strcasecmp(entry->concept, concept_str) == 0) {
            return (int32_t) entry->index;
        }
        entry = entry->next;
    }

    return -1;
}

/**
 * @brief Destroy hash table and free all memory
 *
 * Frees all hash entries and their associated strings.
 *
 * @param table Hash table to destroy (can be NULL)
 *
 * Why: Prevents memory leaks from hash table chains.
 * Complexity: O(n) where n is total number of entries.
 */
static void knowledge_hash_table_destroy(knowledge_hash_table_t* table)
{
    if (!table)
        return;

    if (table->entries) {
        for (uint32_t i = 0; i < table->size; i++) {
            hash_entry_t* entry = table->entries[i];
            while (entry) {
                hash_entry_t* next = entry->next;
                nimcp_free(entry->concept);
                nimcp_free(entry);
                entry = next;
            }
        }
        nimcp_free(table->entries);
    }

    nimcp_free(table);
}

//=============================================================================
// Repository Pattern Implementation
//=============================================================================

/**
 * @brief Create knowledge repository
 *
 * Initializes repository with hash table index for fast lookups.
 *
 * @param initial_capacity Initial storage capacity
 * @return New repository or NULL on allocation failure
 *
 * Why: Separates storage concerns from business logic (Repository Pattern).
 * Provides abstraction layer that can be swapped with different storage backends.
 */
static knowledge_repository_t* repository_create(uint32_t initial_capacity)
{
    knowledge_repository_t* repo = nimcp_calloc(1, sizeof(knowledge_repository_t));
    if (!repo)
        return NULL;

    repo->items = nimcp_calloc(initial_capacity, sizeof(knowledge_item_t));
    if (!repo->items) {
        nimcp_free(repo);
        return NULL;
    }

    repo->index = knowledge_hash_table_create();
    if (!repo->index) {
        nimcp_free(repo->items);
        nimcp_free(repo);
        return NULL;
    }

    // Create B-tree for confidence-based range queries
    repo->confidence_btree = btree_create(compare_confidence, extract_confidence_key, free_knowledge_item);
    if (!repo->confidence_btree) {
        knowledge_hash_table_destroy(repo->index);
        nimcp_free(repo->items);
        nimcp_free(repo);
        return NULL;
    }

    repo->capacity = initial_capacity;
    repo->num_items = 0;

    return repo;
}

/**
 * @brief Find item in repository by concept_str
 *
 * Uses hash table for O(1) average case lookup.
 *
 * @param repo Repository to search (must be non-NULL)
 * @param concept_str Concept to find (must be non-NULL)
 * @return Index or -1 if not found
 *
 * Why: Fast retrieval is critical for knowledge queries and learning.
 * Optimization: Hash table provides dramatic speedup over linear search.
 */
static int32_t repository_find(knowledge_repository_t* repo, const char* concept_str)
{
    if (!repo || !concept_str)
        return -1;
    return knowledge_hash_table_find(repo->index, concept_str);
}

/**
 * @brief Add item to repository
 *
 * Stores item and updates search index atomically.
 *
 * @param repo Repository (must be non-NULL)
 * @param item Item to add (must be non-NULL)
 * @return Index of added item or -1 on failure
 *
 * Why: Ensures index stays synchronized with items array.
 * Invariant: After successful add, repository_find() returns this index.
 */
static int32_t repository_add(knowledge_repository_t* repo, const knowledge_item_t* item)
{
    if (!repo || !item)
        return -1;
    if (repo->num_items >= repo->capacity)
        return -1;

    uint32_t index = repo->num_items;
    repo->items[index] = *item;

    // Populate B-tree key field with unique key: "confidence_index"
    // This ensures each item has a unique key even if confidence values are identical
    snprintf(repo->items[index].confidence_key, sizeof(repo->items[index].confidence_key),
             "%08.6f_%05u", repo->items[index].confidence, index);

    repo->num_items++;

    if (!knowledge_hash_table_insert(repo->index, item->concept_name, index)) {
        repo->num_items--;
        return -1;
    }

    // Insert into B-tree for confidence-based queries
    if (repo->confidence_btree) {
        int result = btree_insert(repo->confidence_btree, &repo->items[index]);
        if (result != BTREE_SUCCESS && result != BTREE_DUPLICATE) {
            // B-tree insert failed, but keep hash table consistent
            // Log warning but don't fail the operation
        }
    }

    return (int32_t) index;
}

/**
 * @brief Get item by index
 *
 * @param repo Repository (must be non-NULL)
 * @param index Item index
 * @return Pointer to item or NULL if invalid index
 *
 * Why: Safe accessor that validates index bounds.
 */
static knowledge_item_t* repository_get(knowledge_repository_t* repo, uint32_t index)
{
    if (!repo || index >= repo->num_items)
        return NULL;
    return &repo->items[index];
}

/**
 * @brief Destroy repository and free all resources
 *
 * @param repo Repository to destroy (can be NULL)
 *
 * Why: Centralized cleanup ensures no memory leaks.
 * Frees: Items array, hash table, and all string arrays in items.
 */
static void repository_destroy(knowledge_repository_t* repo)
{
    if (!repo)
        return;

    if (repo->items) {
        for (uint32_t i = 0; i < repo->num_items; i++) {
            if (repo->items[i].examples) {
                for (uint32_t j = 0; j < repo->items[i].num_examples; j++) {
                    nimcp_free(repo->items[i].examples[j]);
                }
                nimcp_free(repo->items[i].examples);
            }
            if (repo->items[i].related_concepts) {
                for (uint32_t j = 0; j < repo->items[i].num_related; j++) {
                    nimcp_free(repo->items[i].related_concepts[j]);
                }
                nimcp_free(repo->items[i].related_concepts);
            }
        }
        nimcp_free(repo->items);
    }

    knowledge_hash_table_destroy(repo->index);

    // Destroy B-tree
    if (repo->confidence_btree) {
        btree_destroy(repo->confidence_btree);
    }

    nimcp_free(repo);
}

//=============================================================================
// Text Processing Utilities - Single Pass O(n) Algorithms
//=============================================================================

// Minimum word length for meaningful concept_str extraction
#define MIN_CONCEPT_LENGTH 3

/**
 * @brief Check if word should be skipped during extraction
 *
 * Filters common words and short tokens that provide little semantic value.
 *
 * @param word Word to check (must be non-NULL)
 * @return true if word should be skipped
 *
 * Why: Reduces noise in concept_str extraction, focusing on meaningful terms.
 * Could be extended with proper stop word list.
 */
static bool should_skip_word(const char* word)
{
    if (!word || strlen(word) < MIN_CONCEPT_LENGTH)
        return true;

    static const char* stopwords[] = {"the",   "and",   "that",  "this",  "with",
                                      "from",  "have",  "been",  "they",  "their",
                                      "about", "would", "there", "which", NULL};

    for (int i = 0; stopwords[i]; i++) {
        if (strcasecmp(word, stopwords[i]) == 0)
            return true;
    }

    return false;
}

/**
 * @brief Extract concepts from text in single pass
 *
 * Tokenizes text and filters for meaningful concepts efficiently.
 * Optimized to O(n) single-pass algorithm.
 *
 * @param text Input text (must be non-NULL)
 * @param concepts Output array for concepts
 * @param max_concepts Maximum concepts to extract
 * @return Number of concepts extracted
 *
 * Why: Efficient text processing is critical for learning performance.
 * Previous: O(n²) with nested loops checking duplicates.
 * Optimized: O(n) single pass with simple deduplication.
 *
 * Example:
 *   Input: "The quick brown fox jumps over the lazy dog"
 *   Output: ["quick", "brown", "jumps", "lazy"] (filtered stopwords)
 */
static uint32_t extract_concepts_optimized(const char* text, char concepts[][256],
                                           uint32_t max_concepts)
{
    if (!text || !concepts)
        return 0;

    // Use nimcp_malloc instead of strdup to match nimcp_free below
    size_t text_len = strlen(text);
    char* text_copy = nimcp_malloc(text_len + 1);
    if (!text_copy)
        return 0;
    strncpy(text_copy, text, text_len + 1);
    text_copy[text_len] = '\0';

    const char* delimiters = " .,;:!?\n\t\"'()[]{}";
    uint32_t num_concepts = 0;

    char* token = strtok(text_copy, delimiters);
    while (token && num_concepts < max_concepts) {
        if (!should_skip_word(token)) {
            strncpy(concepts[num_concepts], token, 255);
            concepts[num_concepts][255] = '\0';
            num_concepts++;
        }
        token = strtok(NULL, delimiters);
    }

    nimcp_free(text_copy);
    return num_concepts;
}

/**
 * @brief Create context string from text excerpt
 *
 * Extracts first N characters as context while preserving word boundaries.
 *
 * @param text Source text (must be non-NULL)
 * @param output Output buffer (must be non-NULL)
 * @param max_length Maximum output length
 *
 * Why: Provides meaningful context without excessive memory usage.
 * Better than simple truncation by respecting word boundaries.
 */
static void create_context_string(const char* text, char* output, uint32_t max_length)
{
    if (!text || !output || max_length == 0)
        return;

    uint32_t len = strlen(text);
    uint32_t copy_len = (len < max_length - 4) ? len : max_length - 4;

    strncpy(output, text, copy_len);
    output[copy_len] = '\0';

    if (len >= max_length - 4) {
        // Safe: we know we have at least 4 bytes left (max_length - copy_len >= 4)
        strncat(output, "...", max_length - copy_len - 1);
    }
}

/**
 * @brief Normalize concept_str to lowercase for case-insensitive matching
 *
 * @param concept_str Source concept_str string (must be non-NULL)
 * @param output Output buffer (must be non-NULL)
 * @param max_length Maximum output length
 *
 * Why: Ensures consistent concept_str storage regardless of input capitalization.
 * "Democracy", "democracy", and "DEMOCRACY" all map to the same concept_str.
 */
static void normalize_concept_case(const char* concept_str, char* output, uint32_t max_length)
{
    if (!concept_str || !output || max_length == 0)
        return;

    uint32_t i;
    for (i = 0; i < max_length - 1 && concept_str[i] != '\0'; i++) {
        output[i] = (char)tolower((unsigned char)concept_str[i]);
    }
    output[i] = '\0';
}

//=============================================================================
// Strategy Pattern for Domain-Specific Learning
//=============================================================================

/**
 * @brief Deep copy string array
 *
 * @param src Source array
 * @param count Number of strings
 * @return Copied array or NULL
 *
 * Why: Reusable helper avoids code duplication and nested ifs.
 */
static char** deep_copy_string_array(char** src, uint32_t count)
{
    if (!src || count == 0)
        return NULL;

    char** dest = (char**)nimcp_malloc(count * sizeof(char*));
    if (!dest)
        return NULL;

    for (uint32_t i = 0; i < count; i++) {
        dest[i] = NULL;
        if (!src[i])
            continue;

        size_t len = strlen(src[i]) + 1;
        dest[i] = (char*)nimcp_malloc(len);
        if (!dest[i])
            continue;

        strncpy(dest[i], src[i], len);
        dest[i][len - 1] = '\0';
    }

    return dest;
}

/**
 * @brief Learn narrative content strategy
 *
 * Implements learning strategy for narrative/story content.
 *
 * @param system Knowledge system
 * @param data Narrative knowledge structure
 * @return true on success
 *
 * Why: Strategy pattern allows different learning approaches per domain.
 * Stories require extracting themes, characters, and moral lessons.
 */
static bool strategy_learn_narrative(void* system, const void* data)
{
    knowledge_system_t sys = (knowledge_system_t) system;
    const narrative_knowledge_t* story = (const narrative_knowledge_t*) data;

    if (!sys || !story)
        return false;
    if (sys->num_narratives >= sys->narratives_capacity)
        return false;

    narrative_knowledge_t* new_story = &sys->narratives[sys->num_narratives++];
    memset(new_story, 0, sizeof(narrative_knowledge_t));

    // Deep copy scalar fields
    strncpy(new_story->title, story->title, sizeof(new_story->title) - 1);
    strncpy(new_story->author, story->author, sizeof(new_story->author) - 1);
    strncpy(new_story->summary, story->summary, sizeof(new_story->summary) - 1);
    strncpy(new_story->cultural_context, story->cultural_context, sizeof(new_story->cultural_context) - 1);
    new_story->primary_domain = story->primary_domain;

    // Deep copy string arrays using helper
    new_story->num_characters = story->num_characters;
    new_story->characters = deep_copy_string_array(story->characters, story->num_characters);

    new_story->num_themes = story->num_themes;
    new_story->themes = deep_copy_string_array(story->themes, story->num_themes);

    new_story->num_lessons = story->num_lessons;
    new_story->moral_lessons = deep_copy_string_array(story->moral_lessons, story->num_lessons);

    for (uint32_t i = 0; i < story->num_lessons; i++) {
        knowledge_item_t item = {0};
        snprintf(item.concept_name, sizeof(item.concept_name), "lesson_from_%s_%u", story->title, i);
        item.domain = KNOWLEDGE_DOMAIN_ETHICS;
        strncpy(item.definition, story->moral_lessons[i], sizeof(item.definition) - 1);
        snprintf(item.context, sizeof(item.context), "From story: %s", story->title);
        item.confidence = 0.7f;
        item.reinforcement_count = 1;

        int32_t idx = repository_add(sys->repository, &item);
        if (idx >= 0) {
            sys->domain_stats[KNOWLEDGE_DOMAIN_LITERATURE].concepts_known++;
        }
    }

    return true;
}

/**
 * @brief Learn aesthetic/art content strategy
 *
 * @param system Knowledge system
 * @param data Aesthetic knowledge structure
 * @return true on success
 *
 * Why: Art requires extracting aesthetic qualities and emotional impact.
 * Different processing than factual or narrative content.
 */
static bool strategy_learn_aesthetic(void* system, const void* data)
{
    knowledge_system_t sys = (knowledge_system_t) system;
    const aesthetic_knowledge_t* art = (const aesthetic_knowledge_t*) data;

    if (!sys || !art)
        return false;
    if (sys->num_artworks >= sys->artworks_capacity)
        return false;

    aesthetic_knowledge_t* new_art = &sys->artworks[sys->num_artworks++];
    memset(new_art, 0, sizeof(aesthetic_knowledge_t));

    // Deep copy scalar fields
    strncpy(new_art->work_title, art->work_title, sizeof(new_art->work_title) - 1);
    strncpy(new_art->creator, art->creator, sizeof(new_art->creator) - 1);
    strncpy(new_art->medium, art->medium, sizeof(new_art->medium) - 1);
    strncpy(new_art->description, art->description, sizeof(new_art->description) - 1);
    strncpy(new_art->emotional_impact, art->emotional_impact, sizeof(new_art->emotional_impact) - 1);
    strncpy(new_art->historical_significance, art->historical_significance, sizeof(new_art->historical_significance) - 1);

    // Deep copy aesthetic_qualities array
    new_art->num_qualities = art->num_qualities;
    new_art->aesthetic_qualities = deep_copy_string_array(art->aesthetic_qualities, art->num_qualities);

    for (uint32_t i = 0; i < art->num_qualities; i++) {
        knowledge_item_t item = {0};
        strncpy(item.concept_name, art->aesthetic_qualities[i], sizeof(item.concept_name) - 1);
        item.domain = KNOWLEDGE_DOMAIN_ART;
        snprintf(item.definition, sizeof(item.definition), "Quality in %s by %s", art->work_title,
                 art->creator);
        item.confidence = 0.6f;
        item.reinforcement_count = 1;

        int32_t idx = repository_add(sys->repository, &item);
        if (idx >= 0) {
            sys->domain_stats[KNOWLEDGE_DOMAIN_ART].concepts_known++;
        }
    }

    return true;
}

/**
 * @brief Learn historical content strategy
 *
 * @param system Knowledge system
 * @param data Historical knowledge structure
 * @return true on success
 *
 * Why: History requires tracking causality, people, and significance.
 * Different structure than other knowledge types.
 */
static bool strategy_learn_historical(void* system, const void* data)
{
    knowledge_system_t sys = (knowledge_system_t) system;
    const historical_knowledge_t* event = (const historical_knowledge_t*) data;

    if (!sys || !event)
        return false;
    if (sys->num_history >= sys->history_capacity)
        return false;

    historical_knowledge_t* new_event = &sys->history[sys->num_history++];
    memset(new_event, 0, sizeof(historical_knowledge_t));

    // Deep copy scalar fields
    strncpy(new_event->event_name, event->event_name, sizeof(new_event->event_name) - 1);
    new_event->timestamp_year = event->timestamp_year;
    strncpy(new_event->causes, event->causes, sizeof(new_event->causes) - 1);
    strncpy(new_event->effects, event->effects, sizeof(new_event->effects) - 1);
    strncpy(new_event->significance, event->significance, sizeof(new_event->significance) - 1);

    // Deep copy string arrays
    new_event->num_people = event->num_people;
    new_event->key_people = deep_copy_string_array(event->key_people, event->num_people);

    new_event->num_related_events = event->num_related_events;
    new_event->related_events = deep_copy_string_array(event->related_events, event->num_related_events);

    knowledge_item_t item = {0};
    strncpy(item.concept_name, event->event_name, sizeof(item.concept_name) - 1);
    item.domain = KNOWLEDGE_DOMAIN_HISTORY;
    strncpy(item.definition, event->significance, sizeof(item.definition) - 1);
    snprintf(item.context, sizeof(item.context), "Year %lu", event->timestamp_year);
    item.confidence = 0.8f;
    item.reinforcement_count = 1;

    int32_t idx = repository_add(sys->repository, &item);
    if (idx >= 0) {
        sys->domain_stats[KNOWLEDGE_DOMAIN_HISTORY].concepts_known++;
    }

    return true;
}

//=============================================================================
// Domain Statistics and Assessment
//=============================================================================

/**
 * @brief Calculate average confidence for domain
 *
 * Computes mean confidence across all concepts in domain.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param domain Target domain
 * @return Average confidence [0.0, 1.0]
 *
 * Why: Provides measure of understanding quality vs just quantity.
 * Single pass O(n) algorithm with early termination opportunity.
 */
static float calculate_domain_confidence(knowledge_system_t system, knowledge_domain_t domain)
{
    if (!system || !system->repository)
        return 0.0f;

    float total_confidence = 0.0f;
    uint32_t count = 0;

    for (uint32_t i = 0; i < system->repository->num_items; i++) {
        if (system->repository->items[i].domain == domain) {
            total_confidence += system->repository->items[i].confidence;
            count++;
        }
    }

    return (count > 0) ? (total_confidence / count) : 0.0f;
}

/**
 * @brief Update domain statistics incrementally
 *
 * Updates stats after learning without full recalculation.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param domain Domain to update
 *
 * Why: Incremental update is O(1) vs O(n) full recalculation.
 * Maintains statistics efficiently as knowledge grows.
 */
static void update_domain_stats(knowledge_system_t system, knowledge_domain_t domain)
{
    if (!system)
        return;

    domain_knowledge_t* stats = &system->domain_stats[domain];
    stats->avg_confidence = calculate_domain_confidence(system, domain);
    stats->coverage_percentage = (float) stats->concepts_known / stats->estimated_total * 100.0f;
}

//=============================================================================
// Knowledge System Creation/Destruction
//=============================================================================

// REMOVED: create_domain_brain() - Dead code, brains were never used

/**
 * @brief Initialize domain statistics structure
 *
 * Sets initial values for domain tracking.
 *
 * @param stats Statistics structure to initialize (must be non-NULL)
 * @param domain Domain type
 *
 * Why: Centralized initialization ensures consistent state.
 * Provides reasonable defaults for tracking.
 */
static void initialize_domain_stats(domain_knowledge_t* stats, knowledge_domain_t domain)
{
    if (!stats)
        return;

    stats->domain = domain;
    stats->concepts_known = 0;
    stats->estimated_total = 1000;
    stats->coverage_percentage = 0.0f;
    stats->avg_confidence = 0.0f;
    stats->num_gaps = 0;
}

//=============================================================================
// BIO-ASYNC MESSAGE HANDLERS
//=============================================================================

/**
 * @brief Bio-async message handler: Handle knowledge query
 */
static nimcp_error_t handle_knowledge_query(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    (void)msg_size;
    (void)response_promise;

    if (!msg || !user_data) {
        return NIMCP_ERROR_NULL_ARG;
    }

    const bio_msg_introspection_query_t* query = (const bio_msg_introspection_query_t*)msg;
    knowledge_system_t system = (knowledge_system_t)user_data;
    (void)system;  // Will be used for actual query

    LOG_DEBUG(LOG_MODULE, "Received knowledge query: type=%u", query->query_type);

    return NIMCP_SUCCESS;
}

/**
 * @brief Broadcast knowledge update event via bio-async
 */
static void bio_broadcast_knowledge_update(knowledge_system_t system,
                                           uint32_t concepts_learned,
                                           knowledge_domain_t domain) {
    // Process pending bio-async messages
    if (system && system->bio_async_enabled && system->bio_ctx) {
        bio_router_process_inbox(system->bio_ctx, 5);
    }

    if (!system || !system->bio_async_enabled || !system->bio_ctx) {
        return;
    }

    bio_msg_introspection_response_t msg = {0};
    bio_msg_init_header(&msg.header, BIO_MSG_KNOWLEDGE_RESPONSE,
                        bio_module_context_get_id(system->bio_ctx), 0, sizeof(msg));
    msg.header.flags |= BIO_MSG_FLAG_BROADCAST;
    msg.query_type = 0;
    msg.confidence = 1.0f;
    msg.matched_pattern_count = concepts_learned;

    bio_router_broadcast(system->bio_ctx, &msg, sizeof(msg));
    LOG_DEBUG(LOG_MODULE, "Broadcast knowledge update: %u concepts learned in domain %u",
              concepts_learned, domain);
}

/**
 * @brief Create knowledge system with all components
 *
 * Initializes complete system including repository, brains, and strategies.
 * Uses guard clauses instead of nested if statements.
 *
 * @param learner_name Name for learner (must be non-NULL)
 * @return New knowledge system or NULL on failure
 *
 * Why: Complete initialization in one place with clear error handling.
 * Guard clauses improve readability vs nested ifs.
 *
 * Example:
 *   knowledge_system_t sys = knowledge_system_create("Alice");
 *   // System ready with all domains initialized
 */
knowledge_system_t knowledge_system_create(const char* learner_name)
{
    if (!learner_name)
        return NULL;

    knowledge_system_t system = nimcp_calloc(1, sizeof(struct knowledge_system_struct));
    if (!system)
        return NULL;

    strncpy(system->learner_name, learner_name, sizeof(system->learner_name) - 1);

    system->repository = repository_create(INITIAL_CAPACITY);
    if (!system->repository) {
        nimcp_free(system);
        return NULL;
    }

    system->narratives = nimcp_calloc(1000, sizeof(narrative_knowledge_t));
    system->narratives_capacity = 1000;

    system->artworks = nimcp_calloc(500, sizeof(aesthetic_knowledge_t));
    system->artworks_capacity = 500;

    system->history = nimcp_calloc(1000, sizeof(historical_knowledge_t));
    system->history_capacity = 1000;

    system->reading_list = nimcp_calloc(100, sizeof(reading_progress_t));
    system->reading_capacity = 100;

    // Initialize domain statistics (no brains needed - they were never used!)
    for (int i = 0; i < 11; i++) {
        initialize_domain_stats(&system->domain_stats[i], (knowledge_domain_t) i);
    }


    // Create unified brain for knowledge system (provides curiosity module)
    // Previously created curiosity independently - now follows "one brain, many modules" pattern
    system->knowledge_brain = brain_create(learner_name, BRAIN_SIZE_SMALL,
                                           BRAIN_TASK_CLASSIFICATION, 20, 10);
    if (!system->knowledge_brain) {
        fprintf(stderr, "[ERROR] knowledge_system_create: failed to create brain\n"); fflush(stderr);
        knowledge_system_destroy(system);
        return NULL;
    }

    // Initialize bio-async fields
    system->bio_ctx = NULL;
    system->bio_async_enabled = false;

    // Register with bio-async router if available
    LOG_DEBUG("knowledge: Checking bio-async router initialization...");
    if (bio_router_is_initialized()) {
        LOG_DEBUG("knowledge: Bio-router initialized, registering module (id=%d, inbox_capacity=32)...",
                           BIO_MODULE_KNOWLEDGE);
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_KNOWLEDGE,
            .module_name = "knowledge",
            .inbox_capacity = 32,
            .user_data = system
        };
        system->bio_ctx = bio_router_register_module(&bio_info);
        if (system->bio_ctx) {
            system->bio_async_enabled = true;
            // Register message handlers
            bio_router_register_handler(system->bio_ctx, BIO_MSG_KNOWLEDGE_QUERY, handle_knowledge_query);
            LOG_INFO("knowledge: Bio-async communication enabled with handlers (module_id=%d)",
                              BIO_MODULE_KNOWLEDGE);
        } else {
            LOG_WARN("knowledge: Bio-async registration failed - module will operate without async messaging");
        }
    } else {
        LOG_DEBUG("knowledge: Bio-router not initialized, skipping async registration");
    }

    return system;
}

/**
 * @brief Free narrative array memory
 *
 * @param narratives Array to nimcp_free(can be NULL)
 * @param num_narratives Number of narratives
 *
 * Why: Extracted to reduce complexity and improve maintainability.
 * Single responsibility: free one type of resource.
 */
static void free_narratives(narrative_knowledge_t* narratives, uint32_t num_narratives)
{
    if (!narratives)
        return;

    for (uint32_t i = 0; i < num_narratives; i++) {
        if (narratives[i].characters) {
            for (uint32_t j = 0; j < narratives[i].num_characters; j++) {
                nimcp_free(narratives[i].characters[j]);
            }
            nimcp_free(narratives[i].characters);
        }
        if (narratives[i].themes) {
            for (uint32_t j = 0; j < narratives[i].num_themes; j++) {
                nimcp_free(narratives[i].themes[j]);
            }
            nimcp_free(narratives[i].themes);
        }
        if (narratives[i].moral_lessons) {
            for (uint32_t j = 0; j < narratives[i].num_lessons; j++) {
                nimcp_free(narratives[i].moral_lessons[j]);
            }
            nimcp_free(narratives[i].moral_lessons);
        }
    }
    nimcp_free(narratives);
}

/**
 * @brief Free artwork array memory
 *
 * @param artworks Array to nimcp_free(can be NULL)
 * @param num_artworks Number of artworks
 *
 * Why: Separated for clarity and single responsibility.
 */
static void free_artworks(aesthetic_knowledge_t* artworks, uint32_t num_artworks)
{
    if (!artworks)
        return;

    for (uint32_t i = 0; i < num_artworks; i++) {
        if (artworks[i].aesthetic_qualities) {
            for (uint32_t j = 0; j < artworks[i].num_qualities; j++) {
                nimcp_free(artworks[i].aesthetic_qualities[j]);
            }
            nimcp_free(artworks[i].aesthetic_qualities);
        }
    }
    nimcp_free(artworks);
}

/**
 * @brief Free history array memory
 *
 * @param history Array to nimcp_free(can be NULL)
 * @param num_history Number of history entries
 *
 * Why: Extracted for maintainability and single responsibility.
 */
static void free_history(historical_knowledge_t* history, uint32_t num_history)
{
    if (!history)
        return;

    for (uint32_t i = 0; i < num_history; i++) {
        if (history[i].key_people) {
            for (uint32_t j = 0; j < history[i].num_people; j++) {
                nimcp_free(history[i].key_people[j]);
            }
            nimcp_free(history[i].key_people);
        }
        if (history[i].related_events) {
            for (uint32_t j = 0; j < history[i].num_related_events; j++) {
                nimcp_free(history[i].related_events[j]);
            }
            nimcp_free(history[i].related_events);
        }
    }
    nimcp_free(history);
}

// REMOVED: free_domain_brains() - Dead code, brains were never used

/**
 * @brief Destroy knowledge system and free all resources
 *
 * Comprehensive cleanup using guard clauses and helper functions.
 *
 * @param system System to destroy (can be NULL)
 *
 * Why: Complete cleanup prevents memory leaks.
 * Guard clauses and helpers reduce nesting and improve clarity.
 */
void knowledge_system_destroy(knowledge_system_t system)
{
    if (!system)
        return;

    // Unregister from bio-async router
    if (system->bio_async_enabled && system->bio_ctx) {
        bio_router_unregister_module(system->bio_ctx);
        system->bio_ctx = NULL;
        system->bio_async_enabled = false;
        LOG_INFO("Bio-async communication disabled for knowledge");
    }

    repository_destroy(system->repository);
    free_narratives(system->narratives, system->num_narratives);
    free_artworks(system->artworks, system->num_artworks);
    free_history(system->history, system->num_history);
    nimcp_free(system->reading_list);
    // REMOVED: free_domain_brains() call - brains were never used

    // Destroy unified brain (includes curiosity module cleanup)
    // Previously destroyed curiosity independently - now brain owns it
    if (system->knowledge_brain) {
        brain_destroy(system->knowledge_brain);
    }

    nimcp_free(system);
}

//=============================================================================
// Optimized Learning from Text - O(n) Instead of O(n²)
//=============================================================================

/**
 * @brief Process single concept_str during learning
 *
 * Handles both new concept_str learning and reinforcement.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param concept_str Concept to process (must be non-NULL)
 * @param text Source text for context
 * @param domain Knowledge domain
 * @return true if new concept_str learned
 *
 * Why: Extracted from main loop to reduce complexity.
 * Single responsibility: process one concept_str.
 */
static bool process_concept(knowledge_system_t system, const char* concept_str, const char* text,
                            knowledge_domain_t domain)
{
    if (!system || !concept_str)
        return false;

    // Normalize concept_str to lowercase for case-insensitive matching
    char normalized_concept[256];
    normalize_concept_case(concept_str, normalized_concept, sizeof(normalized_concept));

    int32_t idx = repository_find(system->repository, normalized_concept);

    if (idx < 0) {
        knowledge_item_t item = {0};
        strncpy(item.concept_name, normalized_concept, sizeof(item.concept_name) - 1);
        item.domain = domain;
        create_context_string(text, item.definition, sizeof(item.definition));
        item.confidence = 0.3f;
        item.reinforcement_count = 1;

        idx = repository_add(system->repository, &item);
        if (idx >= 0) {
            system->domain_stats[domain].concepts_known++;
            return true;
        }
    } else {
        knowledge_item_t* item = repository_get(system->repository, idx);
        if (item) {
            // BUG FIX: Update B-tree when confidence changes
            // Remove old entry with old key (old confidence + index)
            if (system->repository->confidence_btree) {
                btree_remove(system->repository->confidence_btree, item->confidence_key);
            }

            // Update confidence
            item->reinforcement_count++;
            item->confidence = fminf(item->confidence + CONFIDENCE_INCREMENT, CONFIDENCE_MAX);

            // Update key with new confidence (index stays the same)
            snprintf(item->confidence_key, sizeof(item->confidence_key),
                     "%08.6f_%05u", item->confidence, (uint32_t)idx);

            // Re-insert with new key
            if (system->repository->confidence_btree) {
                btree_insert(system->repository->confidence_btree, item);
            }
        }
    }

    return false;
}

/**
 * @brief Learn from text with optimized O(n) algorithm
 *
 * Extracts and learns concepts in single pass without nested loops.
 * Previous version had O(n²) complexity with nested iteration.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param text Input text (must be non-NULL)
 * @param domain Knowledge domain
 * @return Number of new concepts learned
 *
 * Why: O(n) is critical for processing large texts efficiently.
 * Previous: Nested loops checking duplicates = O(n²)
 * Optimized: Single pass with hash table lookups = O(n)
 *
 * Example:
 *   Input: "Democracy requires informed citizens. Citizens vote."
 *   Learns: "democracy", "requires", "informed", "citizens", "vote"
 *   Second occurrence of "citizens" reinforces instead of duplicating.
 */
uint32_t knowledge_learn_from_text(knowledge_system_t system, const char* text,
                                   knowledge_domain_t domain)
{
    if (!system || !text)
        return 0;

    char concepts[100][256];
    uint32_t num_concepts = extract_concepts_optimized(text, concepts, 100);
    uint32_t learned = 0;

    for (uint32_t i = 0; i < num_concepts; i++) {
        if (process_concept(system, concepts[i], text, domain)) {
            learned++;
        }
    }

    update_domain_stats(system, domain);

    // Broadcast knowledge update via bio-async
    if (learned > 0) {
        bio_broadcast_knowledge_update(system, learned, domain);
    }

    return learned;
}

//=============================================================================
// Learning from Different Sources - Using Strategy Pattern
//=============================================================================

/**
 * @brief Learn from story using narrative strategy
 *
 * @param system Knowledge system (must be non-NULL)
 * @param story Story to learn from (must be non-NULL)
 * @return true on success
 *
 * Why: Stories are key to human learning of values and social behavior.
 * Strategy pattern allows domain-specific processing.
 */
bool knowledge_learn_from_story(knowledge_system_t system, const narrative_knowledge_t* story)
{
    if (!system || !story)
        return false;
    return strategy_learn_narrative(system, story);
}

/**
 * @brief Learn from art using aesthetic strategy
 *
 * @param system Knowledge system (must be non-NULL)
 * @param art_piece Art to learn from (must be non-NULL)
 * @return true on success
 *
 * Why: Art learning requires different processing than factual knowledge.
 * Focuses on aesthetic qualities and emotional impact.
 */
bool knowledge_learn_from_art(knowledge_system_t system, const aesthetic_knowledge_t* art_piece)
{
    if (!system || !art_piece)
        return false;
    return strategy_learn_aesthetic(system, art_piece);
}

/**
 * @brief Learn from history using historical strategy
 *
 * @param system Knowledge system (must be non-NULL)
 * @param event Historical event (must be non-NULL)
 * @return true on success
 *
 * Why: History requires tracking causality and significance.
 * Different structure than narrative or factual knowledge.
 */
bool knowledge_learn_from_history(knowledge_system_t system, const historical_knowledge_t* event)
{
    if (!system || !event)
        return false;
    return strategy_learn_historical(system, event);
}

/**
 * @brief Learn from conversation
 *
 * Social learning through dialogue.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param dialogue Conversation text (must be non-NULL)
 * @param participants Array of participant names
 * @param num_participants Number of participants
 * @return Number of concepts learned
 *
 * Why: Social interaction is primary human learning method.
 * Reuses text learning with social domain context.
 */
uint32_t knowledge_learn_from_conversation(knowledge_system_t system, const char* dialogue,
                                           const char** participants, uint32_t num_participants)
{
    if (!system || !dialogue || !participants || num_participants == 0)
        return 0;
    return knowledge_learn_from_text(system, dialogue, KNOWLEDGE_DOMAIN_SOCIAL);
}

/**
 * @brief Learn from demonstration
 *
 * Procedural learning by observation.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param what_demonstrated Action being demonstrated (must be non-NULL)
 * @param steps Array of step descriptions (must be non-NULL)
 * @param num_steps Number of steps
 * @return true on success
 *
 * Why: Procedural knowledge requires step-by-step encoding.
 * Different from declarative knowledge (facts) or narrative knowledge.
 */
bool knowledge_learn_from_demonstration(knowledge_system_t system, const char* what_demonstrated,
                                        const char** steps, uint32_t num_steps)
{
    if (!system || !what_demonstrated || !steps)
        return false;

    knowledge_item_t item = {0};
    strncpy(item.concept_name, what_demonstrated, sizeof(item.concept_name) - 1);
    item.domain = KNOWLEDGE_DOMAIN_TECHNICAL;

    char definition[1024] = "Steps: ";
    for (uint32_t i = 0; i < num_steps && strlen(definition) < 900; i++) {
        strncat(definition, steps[i], sizeof(definition) - strlen(definition) - 1);
        if (i < num_steps - 1) {
            strncat(definition, ", ", sizeof(definition) - strlen(definition) - 1);
        }
    }
    strncpy(item.definition, definition, sizeof(item.definition) - 1);
    item.confidence = 0.7f;
    item.reinforcement_count = 1;

    int32_t idx = repository_add(system->repository, &item);
    if (idx >= 0) {
        system->domain_stats[KNOWLEDGE_DOMAIN_TECHNICAL].concepts_known++;
        return true;
    }

    return false;
}

//=============================================================================
// Knowledge Retrieval - Using Fast Index
//=============================================================================

/**
 * @brief Retrieve knowledge about concept_str
 *
 * Uses hash table for O(1) average case lookup.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param concept_str Concept to retrieve (must be non-NULL)
 * @param item Output item (must be non-NULL)
 * @return true if found
 *
 * Why: Fast retrieval enables responsive knowledge queries.
 * O(1) hash lookup vs O(n) linear search.
 */
bool knowledge_retrieve(knowledge_system_t system, const char* concept_str, knowledge_item_t* item)
{
    if (!system || !concept_str || !item)
        return false;

    int32_t idx = repository_find(system->repository, concept_str);
    if (idx < 0)
        return false;

    knowledge_item_t* found = repository_get(system->repository, idx);
    if (!found)
        return false;

    *item = *found;
    return true;
}

/**
 * @brief Generate understanding explanation
 *
 * Creates detailed explanation of concept_str with context and confidence.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param concept_str Concept to explain (must be non-NULL)
 * @param context Additional context
 * @param explanation Output buffer (must be non-NULL)
 * @param max_length Buffer size
 * @return Length of explanation
 *
 * Why: Human-readable explanations are key to usable knowledge system.
 * Includes confidence and reinforcement for transparency.
 */
uint32_t knowledge_understand(knowledge_system_t system, const char* concept_str, const char* context,
                              char* explanation, uint32_t max_length)
{
    if (!system || !concept_str || !explanation)
        return 0;

    knowledge_item_t item;
    if (!knowledge_retrieve(system, concept_str, &item)) {
        snprintf(explanation, max_length, "I don't know about '%s' yet.", concept_str);
        return strlen(explanation);
    }

    snprintf(explanation, max_length,
             "'%s' means: %s. Context: %s. I've encountered this %u times "
             "and understand it with %.0f%% confidence.",
             concept_str, item.definition, item.context, item.reinforcement_count,
             item.confidence * 100.0f);

    return strlen(explanation);
}

/**
 * @brief Generate age-appropriate explanation
 *
 * Simplifies explanation based on target age.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param concept_str Concept to explain (must be non-NULL)
 * @param target_age Target age (3-18)
 * @param explanation Output buffer (must be non-NULL)
 * @param max_length Buffer size
 * @return Length of explanation
 *
 * Why: Educational systems need age-appropriate communication.
 * Uses guard clauses for clarity vs nested ifs.
 */
uint32_t knowledge_explain_simply(knowledge_system_t system, const char* concept_str,
                                  uint32_t target_age, char* explanation, uint32_t max_length)
{
    if (!system || !concept_str || !explanation)
        return 0;

    knowledge_item_t item;
    if (!knowledge_retrieve(system, concept_str, &item)) {
        snprintf(explanation, max_length, "I haven't learned about that yet.");
        return strlen(explanation);
    }

    if (target_age < 5) {
        snprintf(explanation, max_length, "%s is something you see/do/learn about.", concept_str);
        return strlen(explanation);
    }

    if (target_age < 10) {
        snprintf(explanation, max_length, "%s: %s", concept_str, item.definition);
        return strlen(explanation);
    }

    return knowledge_understand(system, concept_str, "", explanation, max_length);
}

//=============================================================================
// Cross-Domain Learning - Flattened Algorithms
//=============================================================================

/**
 * @brief Check if items are related across domains
 *
 * @param item1 First item (must be non-NULL)
 * @param item2 Second item (must be non-NULL)
 * @param target_concept Target concept_str name
 * @return true if related
 *
 * Why: Extracted to avoid nested conditions in main loop.
 * Single responsibility: determine relationship.
 */
static bool is_cross_domain_related(const knowledge_item_t* item1, const knowledge_item_t* item2,
                                    const char* target_concept)
{
    if (!item1 || !item2 || !target_concept)
        return false;
    if (item1->domain == item2->domain)
        return false;

    return true;
}

/**
 * @brief Find connections across knowledge domains
 *
 * Single-pass algorithm without nested loops.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param concept_str Central concept_str (must be non-NULL)
 * @param connections Output array (must be non-NULL)
 * @param max_connections Maximum to return
 * @return Number of connections found
 *
 * Why: Cross-domain connections enable transfer learning.
 * Flattened to single pass with guard clauses vs nested loops.
 *
 * Example:
 *   Input: "democracy"
 *   Output: Connections to history (Greek democracy), ethics (voting rights),
 *           social (civic participation)
 */
uint32_t knowledge_find_connections(knowledge_system_t system, const char* concept_str,
                                    knowledge_item_t* connections, uint32_t max_connections)
{
    if (!system || !concept_str || !connections)
        return 0;

    int32_t idx = repository_find(system->repository, concept_str);
    if (idx < 0)
        return 0;

    knowledge_item_t* target = repository_get(system->repository, idx);
    if (!target)
        return 0;

    uint32_t num_found = 0;

    for (uint32_t i = 0; i < system->repository->num_items && num_found < max_connections; i++) {
        if (i == (uint32_t) idx)
            continue;

        knowledge_item_t* item = repository_get(system->repository, i);
        if (!item)
            continue;

        if (is_cross_domain_related(target, item, concept_str)) {
            connections[num_found++] = *item;
        }
    }

    return num_found;
}

/**
 * @brief Apply knowledge from one domain to another
 *
 * Transfer learning enables applying lessons across contexts.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param source_domain Source of knowledge
 * @param target_domain Where to apply
 * @param situation Current situation (must be non-NULL)
 * @param application Output suggestion (must be non-NULL)
 * @param max_length Buffer size
 * @return true on success
 *
 * Why: Transfer learning is key to human intelligence.
 * Example: Apply story lesson (narrative domain) to real situation (social domain).
 */
bool knowledge_transfer_learning(knowledge_system_t system, knowledge_domain_t source_domain,
                                 knowledge_domain_t target_domain, const char* situation,
                                 char* application, uint32_t max_length)
{
    if (!system || !situation || !application)
        return false;

    snprintf(application, max_length, "Applying knowledge from %s to %s domain for situation: %s",
             knowledge_domain_name(source_domain), knowledge_domain_name(target_domain), situation);

    return true;
}

//=============================================================================
// Incremental Building
//=============================================================================

/**
 * @brief Build new knowledge on existing concept_str
 *
 * Analogical learning: learn by similarity to known concept_str.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param new_concept New concept_str to learn (must be non-NULL)
 * @param based_on_concept Base concept_str (must be non-NULL)
 * @param differences How new differs from base
 * @return true on success
 *
 * Why: Humans learn by analogy and building on prior knowledge.
 * Example: Learn "republic" by comparing to "democracy".
 */
bool knowledge_build_on(knowledge_system_t system, const char* new_concept,
                        const char* based_on_concept, const char* differences)
{
    if (!system || !new_concept || !based_on_concept)
        return false;

    int32_t base_idx = repository_find(system->repository, based_on_concept);
    if (base_idx < 0)
        return false;

    knowledge_item_t* base = repository_get(system->repository, base_idx);
    if (!base)
        return false;

    knowledge_item_t new_item = *base;
    strncpy(new_item.concept_name, new_concept, sizeof(new_item.concept_name) - 1);

    if (differences) {
        snprintf(new_item.definition, sizeof(new_item.definition), "Like %s, but: %s",
                 based_on_concept, differences);
    }

    new_item.confidence = base->confidence * 0.7f;
    new_item.reinforcement_count = 1;

    return repository_add(system->repository, &new_item) >= 0;
}

/**
 * @brief Reinforce existing knowledge with new example
 *
 * Spaced repetition and reinforcement strengthen understanding.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param concept_str Concept to reinforce (must be non-NULL)
 * @param new_example New example instance
 * @return true on success
 *
 * Why: Reinforcement is critical for retention and deeper understanding.
 * Each exposure increases confidence and strengthens memory.
 */
bool knowledge_reinforce(knowledge_system_t system, const char* concept_str, const char* new_example)
{
    if (!system || !concept_str)
        return false;

    int32_t idx = repository_find(system->repository, concept_str);
    if (idx < 0)
        return false;

    knowledge_item_t* item = repository_get(system->repository, idx);
    if (!item)
        return false;

    // BUG FIX: Update B-tree when confidence changes
    // Remove old entry with old key (old confidence + index)
    if (system->repository->confidence_btree) {
        btree_remove(system->repository->confidence_btree, item->confidence_key);
    }

    // Update confidence
    item->reinforcement_count++;
    item->confidence = fminf(item->confidence + 0.05f, CONFIDENCE_MAX);

    // Update key with new confidence (index stays the same)
    snprintf(item->confidence_key, sizeof(item->confidence_key),
             "%08.6f_%05u", item->confidence, (uint32_t)idx);

    // Re-insert with new key
    if (system->repository->confidence_btree) {
        btree_insert(system->repository->confidence_btree, item);
    }

    if (new_example && item->num_examples < 10) {
        if (!item->examples) {
            item->examples = nimcp_malloc(10 * sizeof(char*));
            if (!item->examples)
                return false;
        }
        // Use nimcp_malloc instead of strdup to match nimcp_free in destroy
        size_t example_len = strlen(new_example);
        item->examples[item->num_examples] = nimcp_malloc(example_len + 1);
        if (item->examples[item->num_examples]) {
            strncpy(item->examples[item->num_examples], new_example, example_len);
            item->examples[item->num_examples][example_len] = '\0';
            item->num_examples++;
        }
    }

    return true;
}

//=============================================================================
// Knowledge Organization
//=============================================================================

/**
 * @brief Organize knowledge within domain
 *
 * @param system Knowledge system (must be non-NULL)
 * @param domain Domain to organize
 * @return true on success
 *
 * Why: Organization improves retrieval and understanding.
 * Future: Could create hierarchies, clusters, concept_str maps.
 */
bool knowledge_organize_domain(knowledge_system_t system, knowledge_domain_t domain)
{
    if (!system)
        return false;
    update_domain_stats(system, domain);
    return true;
}

/**
 * @brief Get knowledge map for domain
 *
 * Single-pass count without nested loops.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param domain Target domain or GENERAL for all
 * @param map_data Output graph structure
 * @param max_nodes Maximum nodes
 * @return Number of nodes in map
 *
 * Why: Visual knowledge maps aid learning and gap identification.
 * Optimized to single pass with early termination.
 */
uint32_t knowledge_get_map(knowledge_system_t system, knowledge_domain_t domain, void* map_data,
                           uint32_t max_nodes)
{
    if (!system)
        return 0;

    uint32_t count = 0;

    for (uint32_t i = 0; i < system->repository->num_items && count < max_nodes; i++) {
        knowledge_item_t* item = repository_get(system->repository, i);
        if (!item)
            continue;

        if (item->domain == domain || domain == KNOWLEDGE_DOMAIN_GENERAL) {
            count++;
        }
    }

    return count;
}

//=============================================================================
// Reading & Sources
//=============================================================================

/**
 * @brief Start reading book incrementally
 *
 * @param system Knowledge system (must be non-NULL)
 * @param book_title Book title (must be non-NULL)
 * @param book_text Full text (must be non-NULL)
 * @param reading_speed Pages per session
 * @return Number of concepts learned
 *
 * Why: Humans read books incrementally, not all at once.
 * Simulates gradual learning process.
 */
uint32_t knowledge_read_book(knowledge_system_t system, const char* book_title,
                             const char* book_text, uint32_t reading_speed)
{
    if (!system || !book_title || !book_text)
        return 0;
    if (system->num_reading >= system->reading_capacity)
        return 0;

    reading_progress_t* reading = &system->reading_list[system->num_reading++];
    strncpy(reading->book_title, book_title, sizeof(reading->book_title) - 1);
    reading->current_page = 0;
    reading->total_pages = strlen(book_text) / 500;
    reading->comprehension_score = 0.0f;

    char excerpt[2000];
    strncpy(excerpt, book_text, sizeof(excerpt) - 1);
    excerpt[sizeof(excerpt) - 1] = '\0';

    uint32_t learned = knowledge_learn_from_text(system, excerpt, KNOWLEDGE_DOMAIN_LITERATURE);

    reading->current_page = reading_speed;
    return learned;
}

/**
 * @brief Continue reading book from bookmark
 *
 * @param system Knowledge system (must be non-NULL)
 * @param book_title Book to continue (must be non-NULL)
 * @param continue_reading Whether to continue
 * @return Progress percentage [0-100]
 *
 * Why: Persistent reading progress mirrors human learning.
 * Single-pass search with early return.
 */
uint32_t knowledge_continue_reading(knowledge_system_t system, const char* book_title,
                                    bool continue_reading)
{
    if (!system || !book_title)
        return 0;

    for (uint32_t i = 0; i < system->num_reading; i++) {
        if (strcmp(system->reading_list[i].book_title, book_title) == 0) {
            // Guard clause: Prevent divide by zero
            if (system->reading_list[i].total_pages == 0)
                return 0;

            return (system->reading_list[i].current_page * 100) /
                   system->reading_list[i].total_pages;
        }
    }

    return 0;
}

/**
 * @brief Get reading recommendations based on gaps
 *
 * @param system Knowledge system (must be non-NULL)
 * @param domain Target domain
 * @param recommendations Output array (must be non-NULL)
 * @param max_recommendations Maximum to return
 * @return Number of recommendations
 *
 * Why: Targeted recommendations fill knowledge gaps efficiently.
 * Future: Could analyze actual gaps and recommend accordingly.
 */
uint32_t knowledge_get_reading_list(knowledge_system_t system, knowledge_domain_t domain,
                                    char** recommendations, uint32_t max_recommendations)
{
    if (!system || !recommendations)
        return 0;

    static const char* suggestions[] = {"The Very Hungry Caterpillar", "Where the Wild Things Are",
                                        "Charlotte's Web", "The Little Prince",
                                        "A Wrinkle in Time"};

    uint32_t num_suggestions = 5;
    uint32_t count =
        (num_suggestions < max_recommendations) ? num_suggestions : max_recommendations;

    for (uint32_t i = 0; i < count; i++) {
        recommendations[i] = (char*) suggestions[i];
    }

    return count;
}

//=============================================================================
// Knowledge Assessment
//=============================================================================

/**
 * @brief Assess knowledge coverage in domain
 *
 * @param system Knowledge system (must be non-NULL)
 * @param domain Domain to assess
 * @param assessment Output assessment (must be non-NULL)
 * @return true on success
 *
 * Why: Assessment identifies strengths and gaps for targeted learning.
 * Provides quantitative measures of understanding.
 */
bool knowledge_assess_domain(knowledge_system_t system, knowledge_domain_t domain,
                             domain_knowledge_t* assessment)
{
    if (!system || !assessment)
        return false;

    *assessment = system->domain_stats[domain];

    assessment->coverage_percentage =
        (float) assessment->concepts_known / assessment->estimated_total * 100.0f;

    assessment->avg_confidence = calculate_domain_confidence(system, domain);

    return true;
}

/**
 * @brief Get summary of all domains
 *
 * Single-pass generation of all assessments.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param all_domains Output array (must be non-NULL)
 * @param max_domains Array size
 * @return Number of domains assessed
 *
 * Why: Overall summary shows learning progress across all areas.
 * Helps identify which domains need more attention.
 */
uint32_t knowledge_get_summary(knowledge_system_t system, domain_knowledge_t* all_domains,
                               uint32_t max_domains)
{
    if (!system || !all_domains)
        return 0;

    uint32_t count = (11 < max_domains) ? 11 : max_domains;

    for (uint32_t i = 0; i < count; i++) {
        knowledge_assess_domain(system, (knowledge_domain_t) i, &all_domains[i]);
    }

    return count;
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get domain name string
 *
 * @param domain Domain enum value
 * @return Domain name string
 *
 * Why: Human-readable domain names for output and debugging.
 */
const char* knowledge_domain_name(knowledge_domain_t domain)
{
    static const char* names[] = {"Language",  "Literature", "Art",         "Ethics",
                                  "History",   "Science",    "Mathematics", "Social",
                                  "Technical", "Philosophy", "General"};

    if (domain < 0 || domain > 10)
        return "Unknown";
    return names[domain];
}

/**
 * @brief Print knowledge item details
 *
 * @param item Item to print (can be NULL)
 *
 * Why: Debugging and inspection of learned knowledge.
 */
void knowledge_print_item(const knowledge_item_t* item)
{
    if (!item)
        return;

    printf("Concept: %s\n", item->concept_name);
    printf("  Domain: %s\n", knowledge_domain_name(item->domain));
    printf("  Definition: %s\n", item->definition);
    printf("  Context: %s\n", item->context);
    printf("  Confidence: %.2f\n", item->confidence);
    printf("  Reinforcements: %u\n", item->reinforcement_count);
}

/**
 * @brief Print domain assessment
 *
 * @param assessment Assessment to print (can be NULL)
 *
 * Why: Shows learning progress and coverage in domain.
 */
void knowledge_print_assessment(const domain_knowledge_t* assessment)
{
    if (!assessment)
        return;

    printf("Domain: %s\n", knowledge_domain_name(assessment->domain));
    printf("  Concepts known: %u / %u (%.1f%%)\n", assessment->concepts_known,
           assessment->estimated_total, assessment->coverage_percentage);
    printf("  Average confidence: %.2f\n", assessment->avg_confidence);
    printf("  Knowledge gaps: %u\n", assessment->num_gaps);
}

/**
 * @brief Save knowledge to file
 *
 * Persistent storage enables resuming learning sessions.
 *
 * @param system Knowledge system (must be non-NULL)
 * @param filepath Save path (must be non-NULL)
 * @return true on success
 *
 * Why: Long-term memory requires persistent storage.
 * Simplified version saves main items only.
 */
bool knowledge_save(knowledge_system_t system, const char* filepath)
{
    if (!system || !filepath)
        return false;

    FILE* file = fopen(filepath, "wb");
    if (!file)
        return false;

    uint32_t magic = 0x4B4E4F57;
    uint32_t version = 0x00020500;
    fwrite(&magic, sizeof(uint32_t), 1, file);
    fwrite(&version, sizeof(uint32_t), 1, file);

    fwrite(&system->repository->num_items, sizeof(uint32_t), 1, file);
    fwrite(&system->num_narratives, sizeof(uint32_t), 1, file);
    fwrite(&system->num_artworks, sizeof(uint32_t), 1, file);
    fwrite(&system->num_history, sizeof(uint32_t), 1, file);

    for (uint32_t i = 0; i < system->repository->num_items; i++) {
        fwrite(&system->repository->items[i], sizeof(knowledge_item_t), 1, file);
    }

    fclose(file);
    return true;
}

/**
 * @brief Load knowledge from file
 *
 * Resumes learning from previous session.
 *
 * @param filepath Load path (must be non-NULL)
 * @return Loaded knowledge system or NULL on error
 *
 * Why: Enables cumulative learning across sessions.
 * Rebuilds hash index after loading.
 */
knowledge_system_t knowledge_load(const char* filepath)
{
    if (!filepath)
        return NULL;

    FILE* file = fopen(filepath, "rb");
    if (!file)
        return NULL;

    uint32_t magic, version;
    fread(&magic, sizeof(uint32_t), 1, file);
    fread(&version, sizeof(uint32_t), 1, file);

    if (magic != 0x4B4E4F57) {
        fclose(file);
        return NULL;
    }

    knowledge_system_t system = knowledge_system_create("loaded");
    if (!system) {
        fclose(file);
        return NULL;
    }

    fread(&system->repository->num_items, sizeof(uint32_t), 1, file);
    fread(&system->num_narratives, sizeof(uint32_t), 1, file);
    fread(&system->num_artworks, sizeof(uint32_t), 1, file);
    fread(&system->num_history, sizeof(uint32_t), 1, file);

    for (uint32_t i = 0; i < system->repository->num_items; i++) {
        fread(&system->repository->items[i], sizeof(knowledge_item_t), 1, file);

        // Populate B-tree key field with unique key: "confidence_index"
        snprintf(system->repository->items[i].confidence_key,
                 sizeof(system->repository->items[i].confidence_key),
                 "%08.6f_%05u", system->repository->items[i].confidence, i);

        knowledge_hash_table_insert(system->repository->index, system->repository->items[i].concept_name, i);

        // Rebuild B-tree index
        if (system->repository->confidence_btree) {
            btree_insert(system->repository->confidence_btree, &system->repository->items[i]);
        }
    }

    fclose(file);
    return system;
}

//=============================================================================
// B-TREE INDEXED QUERIES
//=============================================================================

/**
 * WHAT: Get knowledge items by confidence range using B-tree
 * WHY: Efficient queries for well/poorly understood concepts
 * HOW: B-tree provides O(log n + k) range queries
 */
uint32_t knowledge_get_by_confidence_range(knowledge_system_t system,
                                            float min_confidence,
                                            float max_confidence,
                                            knowledge_item_t** results_out)
{
    if (!system || !results_out) {
        if (results_out) *results_out = NULL;
        return 0;
    }

    if (min_confidence > max_confidence) {
        *results_out = NULL;
        return 0;
    }

    knowledge_repository_t* repo = system->repository;
    if (!repo || !repo->confidence_btree || repo->num_items == 0) {
        *results_out = NULL;
        return 0;
    }

    // Allocate maximum possible size (will trim later)
    knowledge_item_t* temp_results = nimcp_calloc(repo->num_items, sizeof(knowledge_item_t));
    if (!temp_results) {
        *results_out = NULL;
        return 0;
    }

    // Single pass: collect matching items with early exit
    btree_iterator_t* iter = btree_iterator_create(repo->confidence_btree);
    if (!iter) {
        nimcp_free(temp_results);
        *results_out = NULL;
        return 0;
    }

    uint32_t count = 0;
    void* data = NULL;

    while (btree_iterator_next(iter, &data)) {
        knowledge_item_t* item = (knowledge_item_t*)data;
        if (item->confidence >= min_confidence && item->confidence <= max_confidence) {
            temp_results[count++] = *item;
        } else if (item->confidence > max_confidence) {
            break; // B-tree is sorted, early exit
        }
    }

    btree_iterator_destroy(iter);

    if (count == 0) {
        nimcp_free(temp_results);
        *results_out = NULL;
        return 0;
    }

    // Trim to actual size (optional optimization)
    if (count < repo->num_items) {
        knowledge_item_t* final_results = nimcp_calloc(count, sizeof(knowledge_item_t));
        if (final_results) {
            for (uint32_t i = 0; i < count; i++) {
                final_results[i] = temp_results[i];
            }
            nimcp_free(temp_results);
            *results_out = final_results;
        } else {
            // If trim fails, return oversized array
            *results_out = temp_results;
        }
    } else {
        *results_out = temp_results;
    }

    return count;
}

/**
 * WHAT: Get all knowledge ordered by confidence using B-tree
 * WHY: Review knowledge progression from least to most understood
 * HOW: B-tree in-order traversal provides sorted output
 *
 * BUG FIX: Now uses unique keys (confidence_index) so B-tree stays accurate
 */
uint32_t knowledge_get_all_ordered_by_confidence(knowledge_system_t system,
                                                   knowledge_item_t** results_out)
{
    if (!system || !results_out) {
        if (results_out) *results_out = NULL;
        return 0;
    }

    knowledge_repository_t* repo = system->repository;
    if (!repo || repo->num_items == 0) {
        *results_out = NULL;
        return 0;
    }

    // Allocate results array
    *results_out = nimcp_calloc(repo->num_items, sizeof(knowledge_item_t));
    if (!*results_out) {
        return 0;
    }

    // Use B-tree for sorted output (now maintains correct order with unique keys)
    if (repo->confidence_btree) {
        btree_iterator_t* iter = btree_iterator_create(repo->confidence_btree);
        if (iter) {
            void* data = NULL;
            uint32_t idx = 0;

            while (btree_iterator_next(iter, &data) && idx < repo->num_items) {
                knowledge_item_t* item = (knowledge_item_t*)data;
                (*results_out)[idx++] = *item;
            }

            btree_iterator_destroy(iter);
            return idx;
        }
    }

    // Fallback: copy in array order if B-tree unavailable
    for (uint32_t i = 0; i < repo->num_items; i++) {
        (*results_out)[i] = repo->items[i];
    }

    return repo->num_items;
}

/**
 * WHAT: Add knowledge item directly (for testing)
 * WHY: Test API needs simple item addition
 * HOW: Call repository_add
 */
bool knowledge_add_item(knowledge_system_t system, const knowledge_item_t* item)
{
    if (!system || !item) {
        return false;
    }

    // repository_add will set the confidence_key with the proper format (confidence_index)
    int32_t index = repository_add(system->repository, item);
    return index >= 0;
}

//=============================================================================
// Symbolic Logic Integration (Phase 11: Logic Wiring)
//=============================================================================

/**
 * WHAT: Create a simple atomic formula for symbolic logic
 * WHY:  Enable knowledge→logic integration without complex clause construction
 * HOW:  Create predicate with terms for basic facts
 */
static atomic_formula_t* create_simple_atomic(const char* predicate, const char* arg1, const char* arg2)
{
    if (!predicate || !arg1) {
        return NULL;
    }

    atomic_formula_t* atom = (atomic_formula_t*)nimcp_calloc(1, sizeof(atomic_formula_t));
    if (!atom) {
        return NULL;
    }

    // Set predicate name
    strncpy(atom->name, predicate, LOGIC_MAX_NAME_LENGTH - 1);
    atom->negated = false;

    // Create terms
    uint8_t arity = arg2 ? 2 : 1;
    atom->terms = (logical_term_t**)nimcp_calloc(arity, sizeof(logical_term_t*));
    if (!atom->terms) {
        nimcp_free(atom);
        return NULL;
    }

    // First argument
    atom->terms[0] = (logical_term_t*)nimcp_calloc(1, sizeof(logical_term_t));
    if (!atom->terms[0]) {
        nimcp_free(atom->terms);
        nimcp_free(atom);
        return NULL;
    }
    atom->terms[0]->type = TERM_CONSTANT;
    strncpy(atom->terms[0]->name, arg1, LOGIC_MAX_NAME_LENGTH - 1);
    atom->terms[0]->args = NULL;
    atom->terms[0]->arity = 0;

    // Second argument (if provided)
    if (arg2) {
        atom->terms[1] = (logical_term_t*)nimcp_calloc(1, sizeof(logical_term_t));
        if (!atom->terms[1]) {
            nimcp_free(atom->terms[0]);
            nimcp_free(atom->terms);
            nimcp_free(atom);
            return NULL;
        }
        atom->terms[1]->type = TERM_CONSTANT;
        strncpy(atom->terms[1]->name, arg2, LOGIC_MAX_NAME_LENGTH - 1);
        atom->terms[1]->args = NULL;
        atom->terms[1]->arity = 0;
    }

    atom->arity = arity;
    return atom;
}

/**
 * WHAT: Create logic clause from atomic formula
 * WHY:  Symbolic logic works with clauses (CNF)
 * HOW:  Wrap atomic formula in a clause with single literal
 */
static logic_clause_t* create_clause_from_atomic(atomic_formula_t* atom, float confidence)
{
    if (!atom) {
        return NULL;
    }

    logic_clause_t* clause = (logic_clause_t*)nimcp_calloc(1, sizeof(logic_clause_t));
    if (!clause) {
        return NULL;
    }

    clause->literals = (atomic_formula_t**)nimcp_calloc(1, sizeof(atomic_formula_t*));
    if (!clause->literals) {
        nimcp_free(clause);
        return NULL;
    }

    clause->literals[0] = atom;
    clause->num_literals = 1;
    clause->confidence = confidence;

    return clause;
}

/**
 * WHAT: Add knowledge item to symbolic logic as facts
 * WHY:  Enable logical reasoning over knowledge graph (as recommended in audit)
 * HOW:  Convert knowledge concepts to IsA/HasProperty predicates
 *
 * INTEGRATION: Called when both knowledge and symbolic_logic are available
 *
 * EXAMPLE:
 *   Concept: "cat" with related: "animal"
 *   → Logic: IsA(cat, animal)
 *
 * @param logic Symbolic logic engine
 * @param item Knowledge item to convert to logic facts
 * @return Number of facts added
 */
uint32_t knowledge_add_to_symbolic_logic(
    symbolic_logic_t* logic,
    const knowledge_item_t* item)
{
    if (!logic || !item) {
        return 0;
    }

    uint32_t facts_added = 0;

    // Add IsA facts for each related concept
    // Example: IsA(cat, animal) if "cat" is related to "animal"
    for (uint32_t i = 0; i < item->num_related && i < 10; i++) {
        atomic_formula_t* isa_atom = create_simple_atomic(
            "IsA",
            item->concept_name,
            item->related_concepts[i]
        );

        if (isa_atom) {
            logic_clause_t* clause = create_clause_from_atomic(isa_atom, item->confidence);
            if (clause) {
                if (symbolic_logic_add_fact(logic, clause, item->confidence)) {
                    facts_added++;
                }
                // Note: symbolic_logic takes ownership of clause, don't free
            }
        }
    }

    // Add HasProperty fact for the concept itself
    // Example: Concept(cat) - indicates concept exists in knowledge base
    atomic_formula_t* concept_atom = create_simple_atomic(
        "Concept",
        item->concept_name,
        NULL  // Unary predicate
    );

    if (concept_atom) {
        logic_clause_t* clause = create_clause_from_atomic(concept_atom, item->confidence);
        if (clause) {
            if (symbolic_logic_add_fact(logic, clause, item->confidence)) {
                facts_added++;
            }
        }
    }

    return facts_added;
}
