//=============================================================================
// nimcp_brain_persistence.c - Brain Persistence & Snapshot Management
//=============================================================================
/**
 * @file nimcp_brain_persistence.c
 * @brief Implementation of brain state persistence and snapshot management
 *
 * EXTRACTED FROM: nimcp_brain.c lines 7088-8226
 * EXTRACTION DATE: 2025-11-19
 *
 * WHAT: Implements save/load/snapshot APIs for brain state persistence
 * WHY:  Modularize persistence logic to reduce nimcp_brain.c size
 * HOW:  Serialize brain state to files with versioning, compression, encryption
 *
 * ARCHITECTURE:
 * - Persistence: brain_save() → save_metadata() → save subsystems
 * - Loading: brain_load() → load_metadata() → restore subsystems
 * - Snapshots: Wrappers around persistence with directory management
 *
 * DEPENDENCIES:
 * - Brain structure (nimcp_brain.h)
 * - Subsystem save/load APIs (knowledge, executive, mirror neurons, etc.)
 * - File I/O, directory scanning (stdio.h, dirent.h, sys/stat.h)
 */

#include "nimcp_brain_persistence.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

// Platform-specific directory operations
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// Core dependencies
#include "plasticity/adaptive/nimcp_adaptive.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/validation/nimcp_validate.h"

// Subsystem dependencies for save/load
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "cognitive/nimcp_executive.h"
#include "cognitive/nimcp_mirror_neurons.h"
#include "plasticity/neuromodulators/nimcp_neuromod_pink_noise.h"
#include "cognitive/nimcp_working_memory.h"

// Additional subsystems that may need initialization
#include "glial/integration/nimcp_glial_integration.h"
#include "plasticity/neuromodulators/nimcp_spatial_neuromod.h"
#include "optimization/quantum_annealing/nimcp_quantum_annealing.h"

// Error handling (forward declarations from nimcp_brain.c)
// Note: These are defined in nimcp_brain.c, we'll use them here
extern void set_error(const char* format, ...);
extern void brain_clear_error(void);

// Forward declarations of nimcp_brain.c internal functions we need
// Note: These would need to be exposed or we duplicate them here
extern brain_t allocate_brain(void);
extern task_strategy_t* strategy_create(brain_task_t task);
extern void init_brain_stats(brain_stats_t* stats, const char* task_name,
                            brain_size_t size, uint32_t num_inputs, float learning_rate);
extern bool init_working_memory_subsystem(brain_t brain);
extern bool init_mirror_neurons(brain_t brain);
extern bool init_glial_subsystem(brain_t brain);
extern bool init_spatial_neuromod_system(brain_t brain);
extern void executive_set_brain(executive_controller_t* exec, brain_t brain);
extern void mirror_neurons_set_brain(mirror_neurons_t mirror, brain_t brain);  // Fixed: mirror_neurons_t not pointer

//=============================================================================
// Persistence API
//=============================================================================

/**
 * @brief Save working memory state to file (Phase 10.2)
 *
 * WHAT: Serialize working memory items for COW snapshot persistence
 * WHY:  Preserve active representations across save/load/snapshot operations
 * HOW:  Write marker → size/capacity → each item's data
 *
 * COMPLEXITY: O(n*m) where n = items, m = avg item size
 *
 * @param wm Working memory instance (nullable)
 * @param file File handle (non-NULL)
 * @return true on success, false on error
 */
bool nimcp_brain_save_working_memory_state(working_memory_t* wm, FILE* file)
{
    // Guard: NULL file handle
    if (!file) {
        return false;
    }

    // Guard: No working memory → write marker and return
    if (!wm) {
        uint8_t has_wm = 0;
        fwrite(&has_wm, sizeof(uint8_t), 1, file);
        return true;
    }

    // Write existence marker
    uint8_t has_wm = 1;
    fwrite(&has_wm, sizeof(uint8_t), 1, file);

    // Get current state
    working_memory_stats_t stats;
    working_memory_get_stats(wm, &stats);

    // Write metadata
    fwrite(&stats.current_size, sizeof(uint32_t), 1, file);
    fwrite(&stats.capacity, sizeof(uint32_t), 1, file);

    // Write each item
    for (uint32_t i = 0; i < stats.current_size; i++) {
        uint32_t item_size = 0;
        const float* item = working_memory_get(wm, i, &item_size);

        // Guard: Invalid item → skip
        if (!item || item_size == 0) {
            continue;
        }

        fwrite(&item_size, sizeof(uint32_t), 1, file);
        fwrite(item, sizeof(float), item_size, file);
    }

    return true;
}

/**
 * @brief Save metadata file
 *
 * WHAT: Persist brain configuration and output labels
 * WHY:  Enable full state reconstruction on load
 * HOW:  Write config → labels → working memory state
 *
 * COMPLEXITY: O(k + n*m) where k = labels, n = WM items, m = item size
 *
 * @param brain Brain instance (non-NULL)
 * @param filepath Base filepath (non-NULL)
 * @return true on success, false on error
 */
bool nimcp_brain_save_metadata(brain_t brain, const char* filepath)
{
    // Guard: NULL parameters handled by caller

    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", filepath);

    FILE* meta_file = fopen(meta_path, "wb");
    if (!meta_file) {
        return false;
    }

    // Write version header (v1.0 format)
    nimcp_file_header_t header = {
        .magic = {NIMCP_MAGIC_0, NIMCP_MAGIC_1, NIMCP_MAGIC_2, NIMCP_MAGIC_3},
        .version_major = NIMCP_FORMAT_VERSION_MAJOR,
        .version_minor = NIMCP_FORMAT_VERSION_MINOR,
        .flags = 0,  // No compression/encryption yet
        .reserved = 0
    };
    fwrite(&header, sizeof(nimcp_file_header_t), 1, meta_file);

    // Write configuration
    fwrite(&brain->config, sizeof(brain_config_t), 1, meta_file);
    fwrite(&brain->num_output_labels, sizeof(uint32_t), 1, meta_file);

    // Write output labels
    for (uint32_t i = 0; i < brain->num_output_labels; i++) {
        uint32_t len = strlen(brain->output_labels[i]) + 1;
        fwrite(&len, sizeof(uint32_t), 1, meta_file);
        fwrite(brain->output_labels[i], len, 1, meta_file);
    }

    // Phase 10.2: Save working memory state
    bool wm_success = nimcp_brain_save_working_memory_state(brain->working_memory, meta_file);
    if (!wm_success) {
        fclose(meta_file);
        return false;
    }

    // Save brain statistics (performance metrics)
    fwrite(&brain->stats, sizeof(brain_stats_t), 1, meta_file);

    // Save wellbeing state (Phase 9.3)
    fwrite(&brain->last_distress, sizeof(distress_assessment_t), 1, meta_file);
    fwrite(&brain->wellbeing_monitoring_enabled, sizeof(bool), 1, meta_file);
    fwrite(&brain->wellbeing_check_interval_ms, sizeof(uint64_t), 1, meta_file);
    fwrite(&brain->last_wellbeing_check_time, sizeof(uint64_t), 1, meta_file);

    // Save simulation time tracking
    fwrite(&brain->current_time_us, sizeof(uint64_t), 1, meta_file);
    fwrite(&brain->last_glial_update_us, sizeof(uint64_t), 1, meta_file);

    // Save knowledge system state (if exists)
    bool has_knowledge = (brain->knowledge != NULL);
    fwrite(&has_knowledge, sizeof(bool), 1, meta_file);
    if (has_knowledge) {
        char knowledge_path[512];
        snprintf(knowledge_path, sizeof(knowledge_path), "%s.knowledge", filepath);
        knowledge_save(brain->knowledge, knowledge_path);
    }

    // Save emotional system state (Phase 10.2 - NOT A MODULE)
    // Note: Emotional tagging uses stateless utility functions, not a system object
    bool has_emotional = false;  // No emotional_system module (just tagging functions)
    fwrite(&has_emotional, sizeof(bool), 1, meta_file);

    // Save executive controller state (Phase 10.3 - if exists)
    bool has_executive = (brain->executive != NULL);
    fwrite(&has_executive, sizeof(bool), 1, meta_file);
    if (has_executive) {
        // WHAT: Save executive controller state to separate file
        // WHY:  Preserve task queue, statistics, and configuration
        // HOW:  Use executive_save API with dedicated file
        char executive_path[512];
        snprintf(executive_path, sizeof(executive_path), "%s.executive", filepath);
        FILE* exec_file = fopen(executive_path, "wb");
        if (exec_file) {
            executive_save(brain->executive, exec_file);
            fclose(exec_file);
        }
    }

    // Save sleep system state (Phase 10.4)
    // Sleep system is embedded struct, always save
    // TODO: Add sleep_system_save API when available
    // For now, skip to maintain backward compatibility

    // Save pink noise neuromodulator state (if exists)
    bool has_pink_noise = (brain->pink_noise != NULL);
    fwrite(&has_pink_noise, sizeof(bool), 1, meta_file);
    if (has_pink_noise) {
        // WHAT: Save pink noise neuromodulator state to separate file
        // WHY:  Preserve neuromodulator levels and pink noise generators
        // HOW:  Use neuromod_pink_save API with dedicated file
        char pink_noise_path[512];
        snprintf(pink_noise_path, sizeof(pink_noise_path), "%s.pink_noise", filepath);
        FILE* pink_file = fopen(pink_noise_path, "wb");
        if (pink_file) {
            neuromod_pink_save(brain->pink_noise, pink_file);
            fclose(pink_file);
        }
    }

    // Save mirror neurons state (Phase 10.11 - if exists)
    bool has_mirror_neurons = (brain->mirror_neurons != NULL);
    fwrite(&has_mirror_neurons, sizeof(bool), 1, meta_file);
    if (has_mirror_neurons) {
        // WHAT: Save mirror neuron system state to separate file
        // WHY:  Preserve learned action associations and statistics
        // HOW:  Use mirror_neurons_save API with dedicated file
        char mirror_path[512];
        snprintf(mirror_path, sizeof(mirror_path), "%s.mirror_neurons", filepath);
        FILE* mirror_file = fopen(mirror_path, "wb");
        if (mirror_file) {
            mirror_neurons_save(brain->mirror_neurons, mirror_file);
            fclose(mirror_file);
        }
    }

    fclose(meta_file);
    return true;
}

/**
 * @brief Save brain to file
 *
 * WHY: Enables model persistence across sessions
 * Saves both network and metadata
 *
 * COMPLEXITY: O(n*c) where n = neurons, c = connections
 *
 * @param brain Brain handle
 * @param filepath Path to save to
 * @return true on success
 */
bool brain_save(brain_t brain, const char* filepath)
{
    // Guard: Validate parameters
    if (!brain || !filepath) {
        set_error("Invalid parameters to brain_save");
        return false;
    }

    // Save adaptive network
    bool success = adaptive_network_save(brain->network, filepath, SERIALIZE_FORMAT_BINARY);

    if (!success) {
        set_error("Failed to save adaptive network to %s", filepath);
        return false;
    }

    // Save metadata
    if (!nimcp_brain_save_metadata(brain, filepath)) {
        set_error("Failed to save metadata");
        return false;
    }

    brain_clear_error();
    return true;
}

/**
 * @brief Load single working memory item from file (Phase 10.2)
 *
 * WHAT: Deserialize one item and add to working memory buffer
 * WHY:  Restore individual active representations
 * HOW:  Read size → allocate → read data → add to buffer → free temp
 *
 * COMPLEXITY: O(m) where m = item size
 *
 * @param wm Working memory instance (non-NULL)
 * @param file File handle (non-NULL)
 * @return true on success, false on error
 */
static bool load_working_memory_item(working_memory_t* wm, FILE* file)
{
    #define MAX_ITEM_SIZE 10000  // Sanity check limit

    // Guard: NULL parameters
    if (!wm || !file) {
        return false;
    }

    uint32_t item_size = 0;
    if (fread(&item_size, sizeof(uint32_t), 1, file) != 1) {
        return false;
    }

    // Guard: Invalid size
    if (item_size == 0 || item_size > MAX_ITEM_SIZE) {
        return false;
    }

    // Allocate temporary buffer
    float* item = nimcp_malloc(item_size * sizeof(float));
    if (!item) {
        return false;
    }

    // Read item data
    if (fread(item, sizeof(float), item_size, file) != item_size) {
        nimcp_free(item);
        return false;
    }

    // Add to working memory (use default salience since not persisted)
    const float DEFAULT_SALIENCE = 0.5f;
    bool success = working_memory_add(wm, item, item_size, DEFAULT_SALIENCE);

    nimcp_free(item);
    return success;

    #undef MAX_ITEM_SIZE
}

/**
 * @brief Load working memory state from file (Phase 10.2)
 *
 * WHAT: Deserialize working memory items from COW snapshot
 * WHY:  Restore active representations after load/restore
 * HOW:  Read marker → initialize if needed → load each item
 *
 * COMPLEXITY: O(n*m) where n = items, m = avg item size
 *
 * @param brain Brain instance (non-NULL)
 * @param file File handle (non-NULL)
 * @return true on success (non-fatal on WM failure)
 */
bool nimcp_brain_load_working_memory_state(brain_t brain, FILE* file)
{
    // Guard: NULL parameters
    if (!brain || !file) {
        return false;
    }

    // Read existence marker
    uint8_t has_wm = 0;
    if (fread(&has_wm, sizeof(uint8_t), 1, file) != 1) {
        return true;  // EOF or old format → non-fatal
    }

    // Guard: No working memory in snapshot
    if (has_wm == 0) {
        return true;  // Nothing to load → success
    }

    // Read metadata
    uint32_t wm_size = 0, wm_capacity = 0;
    if (fread(&wm_size, sizeof(uint32_t), 1, file) != 1) {
        return true;  // Non-fatal
    }
    if (fread(&wm_capacity, sizeof(uint32_t), 1, file) != 1) {
        return true;  // Non-fatal
    }

    // Initialize working memory if enabled but not yet created
    if (!brain->working_memory && brain->config.enable_working_memory) {
        if (!init_working_memory_subsystem(brain)) {
            fprintf(stderr, "WARNING: Failed to initialize working memory on load\n");
            return true;  // Non-fatal: continue without WM
        }
    }

    // Guard: Working memory not available
    if (!brain->working_memory) {
        return true;  // Skip loading → non-fatal
    }

    // Load each item
    for (uint32_t i = 0; i < wm_size; i++) {
        load_working_memory_item(brain->working_memory, file);
        // Errors loading individual items are non-fatal
    }

    return true;
}

/**
 * @brief Load metadata file
 *
 * WHAT: Deserialize brain configuration and output labels
 * WHY:  Reconstruct full brain state from persistent storage
 * HOW:  Read config → validate → load labels → load working memory
 *
 * COMPLEXITY: O(k + n*m) where k = labels, n = WM items, m = item size
 *
 * @param brain Brain instance (non-NULL)
 * @param filepath Base filepath (non-NULL)
 * @return true on success, false on error
 */
bool nimcp_brain_load_metadata(brain_t brain, const char* filepath)
{
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", filepath);

    FILE* meta_file = fopen(meta_path, "rb");
    if (!meta_file)
        return false;

    // Try to read version header
    nimcp_file_header_t header;
    size_t header_read = fread(&header, sizeof(nimcp_file_header_t), 1, meta_file);

    bool has_version_header = false;
    if (header_read == 1) {
        // Check magic bytes
        if (header.magic[0] == NIMCP_MAGIC_0 &&
            header.magic[1] == NIMCP_MAGIC_1 &&
            header.magic[2] == NIMCP_MAGIC_2 &&
            header.magic[3] == NIMCP_MAGIC_3) {

            has_version_header = true;

            // Validate version compatibility
            if (header.version_major != NIMCP_FORMAT_VERSION_MAJOR) {
                fprintf(stderr, "ERROR: Incompatible format version %u.%u (expected %u.x)\n",
                        header.version_major, header.version_minor,
                        NIMCP_FORMAT_VERSION_MAJOR);
                fclose(meta_file);
                return false;
            }

            fprintf(stderr, "[INFO] Loading brain metadata v%u.%u\n",
                    header.version_major, header.version_minor);

            // TODO: Handle format flags (compression, encryption)
            if (header.flags & NIMCP_FORMAT_FLAG_COMPRESSED) {
                fprintf(stderr, "[WARN] Compressed format not yet supported, skipping\n");
            }
            if (header.flags & NIMCP_FORMAT_FLAG_ENCRYPTED) {
                fprintf(stderr, "[WARN] Encrypted format not yet supported, skipping\n");
            }
        } else {
            // Not a versioned file - rewind and read as legacy format
            has_version_header = false;
            fseek(meta_file, 0, SEEK_SET);
        }
    } else {
        // File too small for header - legacy format
        fseek(meta_file, 0, SEEK_SET);
    }

    if (!has_version_header) {
        fprintf(stderr, "[INFO] Loading brain metadata (legacy format, no version header)\n");
    }

    fread(&brain->config, sizeof(brain_config_t), 1, meta_file);

    // Validate brain->config fields after reading
    // Validate learning_rate (float field - NaN/Inf check)
    if (!nimcp_validate_float_field(&brain->config.learning_rate,
                                    sizeof(brain->config.learning_rate))) {
        fprintf(stderr, "ERROR: Invalid learning_rate in loaded config (NaN or Inf)\n");
        fclose(meta_file);
        return false;
    }

    // Validate sparsity_target (float field - NaN/Inf check)
    if (!nimcp_validate_float_field(&brain->config.sparsity_target,
                                    sizeof(brain->config.sparsity_target))) {
        fprintf(stderr, "ERROR: Invalid sparsity_target in loaded config (NaN or Inf)\n");
        fclose(meta_file);
        return false;
    }

    // Validate num_inputs (integer field, range 1-10000)
    if (!nimcp_validate_integer_field(&brain->config.num_inputs,
                                      sizeof(brain->config.num_inputs))) {
        fprintf(stderr, "ERROR: Invalid num_inputs in loaded config\n");
        fclose(meta_file);
        return false;
    }
    if (brain->config.num_inputs < 1 || brain->config.num_inputs > 10000) {
        fprintf(stderr, "ERROR: num_inputs out of range (1-10000): %u\n", brain->config.num_inputs);
        fclose(meta_file);
        return false;
    }

    // Validate num_outputs (integer field, range 1-10000)
    if (!nimcp_validate_integer_field(&brain->config.num_outputs,
                                      sizeof(brain->config.num_outputs))) {
        fprintf(stderr, "ERROR: Invalid num_outputs in loaded config\n");
        fclose(meta_file);
        return false;
    }
    if (brain->config.num_outputs < 1 || brain->config.num_outputs > 10000) {
        fprintf(stderr, "ERROR: num_outputs out of range (1-10000): %u\n",
                brain->config.num_outputs);
        fclose(meta_file);
        return false;
    }

    fread(&brain->num_output_labels, sizeof(uint32_t), 1, meta_file);

    // SECURITY: Strict validation limits to prevent buffer overflow attacks
    #define MAX_OUTPUT_LABELS 10000     // Maximum number of labels
    #define MAX_LABEL_LENGTH 256        // Maximum length of a single label

    // Validate num_output_labels (range 0-10000, 0 means no labels)
    if (!nimcp_validate_integer_field(&brain->num_output_labels,
                                      sizeof(brain->num_output_labels))) {
        fprintf(stderr, "ERROR: Invalid num_output_labels in loaded metadata\n");
        fclose(meta_file);
        return false;
    }
    if (brain->num_output_labels > MAX_OUTPUT_LABELS) {
        fprintf(stderr, "SECURITY ERROR: num_output_labels %u exceeds maximum %d\n",
                brain->num_output_labels, MAX_OUTPUT_LABELS);
        fprintf(stderr, "This file may be maliciously crafted\n");
        fclose(meta_file);
        return false;
    }

    // Handle case where there are no labels
    if (brain->num_output_labels == 0) {
        brain->output_labels = NULL;
        fclose(meta_file);
        return true;
    }

    brain->output_labels = nimcp_malloc(brain->num_output_labels * sizeof(char*));
    if (!brain->output_labels) {
        fprintf(stderr, "ERROR: Failed to allocate output_labels array\n");
        fclose(meta_file);
        return false;
    }

    uint32_t i;
    for (i = 0; i < brain->num_output_labels; i++) {
        uint32_t len;
        fread(&len, sizeof(uint32_t), 1, meta_file);

        // SECURITY: Validate label length to prevent buffer overflow
        if (len == 0 || len > MAX_LABEL_LENGTH) {
            fprintf(stderr, "SECURITY ERROR: Label %u length %u exceeds maximum %d\n",
                    i, len, MAX_LABEL_LENGTH);
            fprintf(stderr, "This file may be maliciously crafted\n");
            goto cleanup;
        }

        // Validate integer field integrity
        if (!nimcp_validate_integer_field(&len, sizeof(len))) {
            fprintf(stderr, "ERROR: Invalid label length at index %u\n", i);
            goto cleanup;
        }

        brain->output_labels[i] = nimcp_malloc(len);
        if (!brain->output_labels[i]) {
            fprintf(stderr, "ERROR: Failed to allocate label at index %u\n", i);
            goto cleanup;
        }

        fread(brain->output_labels[i], len, 1, meta_file);
    }

    // Phase 10.2: Load working memory state
    nimcp_brain_load_working_memory_state(brain, meta_file);

    // Load brain statistics (performance metrics)
    if (fread(&brain->stats, sizeof(brain_stats_t), 1, meta_file) != 1) {
        // Non-fatal: use default stats if not available (backward compatibility)
        init_brain_stats(&brain->stats, brain->config.task_name, brain->config.size,
                        brain->config.num_inputs, brain->config.learning_rate);
    }

    // Load wellbeing state (Phase 9.3)
    if (fread(&brain->last_distress, sizeof(distress_assessment_t), 1, meta_file) == 1 &&
        fread(&brain->wellbeing_monitoring_enabled, sizeof(bool), 1, meta_file) == 1 &&
        fread(&brain->wellbeing_check_interval_ms, sizeof(uint64_t), 1, meta_file) == 1 &&
        fread(&brain->last_wellbeing_check_time, sizeof(uint64_t), 1, meta_file) == 1) {
        // Successfully loaded wellbeing state
    }

    // Load simulation time tracking (may not exist in old snapshots)
    if (fread(&brain->current_time_us, sizeof(uint64_t), 1, meta_file) == 1 &&
        fread(&brain->last_glial_update_us, sizeof(uint64_t), 1, meta_file) == 1) {
        // Successfully loaded time tracking
    } else {
        // Old snapshot, initialize to 0
        brain->current_time_us = 0;
        brain->last_glial_update_us = 0;
    }

    // Load knowledge system state (if exists)
    bool has_knowledge = false;
    if (fread(&has_knowledge, sizeof(bool), 1, meta_file) == 1 && has_knowledge) {
        char knowledge_path[512];
        snprintf(knowledge_path, sizeof(knowledge_path), "%s.knowledge", filepath);
        brain->knowledge = knowledge_load(knowledge_path);
        // Non-fatal if knowledge load fails
    }

    // Load emotional system state (Phase 10.2 - NOT A MODULE)
    // Note: Emotional tagging uses stateless utility functions, not a system object
    bool has_emotional = false;
    if (fread(&has_emotional, sizeof(bool), 1, meta_file) == 1 && has_emotional) {
        // Placeholder for backward compatibility (old saves might have this flag set)
        // No action needed - emotional tagging uses stateless functions
    }

    // Load executive controller state (Phase 10.3 - if exists)
    bool has_executive = false;
    if (fread(&has_executive, sizeof(bool), 1, meta_file) == 1 && has_executive) {
        // WHAT: Load executive controller state from separate file
        // WHY:  Restore task queue, statistics, and configuration
        // HOW:  Use executive_load API with dedicated file
        char executive_path[512];
        snprintf(executive_path, sizeof(executive_path), "%s.executive", filepath);
        FILE* exec_file = fopen(executive_path, "rb");
        if (exec_file) {
            brain->executive = executive_load(exec_file);
            fclose(exec_file);
            // Set brain reference for neuromodulation integration
            if (brain->executive) {
                executive_set_brain(brain->executive, brain);
            }
        }
    }

    // Load pink noise neuromodulator state (if exists)
    bool has_pink_noise = false;
    if (fread(&has_pink_noise, sizeof(bool), 1, meta_file) == 1 && has_pink_noise) {
        // WHAT: Load pink noise neuromodulator state from separate file
        // WHY:  Restore neuromodulator levels and pink noise generators
        // HOW:  Use neuromod_pink_load API with dedicated file
        char pink_noise_path[512];
        snprintf(pink_noise_path, sizeof(pink_noise_path), "%s.pink_noise", filepath);
        FILE* pink_file = fopen(pink_noise_path, "rb");
        if (pink_file) {
            brain->pink_noise = neuromod_pink_load(pink_file);
            fclose(pink_file);
        }
    }

    // Load mirror neurons state (Phase 10.11 - if exists)
    bool has_mirror_neurons = false;
    if (fread(&has_mirror_neurons, sizeof(bool), 1, meta_file) == 1 && has_mirror_neurons) {
        // WHAT: Load mirror neuron system state from separate file
        // WHY:  Restore learned action associations and statistics
        // HOW:  Use mirror_neurons_load API with dedicated file
        char mirror_path[512];
        snprintf(mirror_path, sizeof(mirror_path), "%s.mirror_neurons", filepath);
        FILE* mirror_file = fopen(mirror_path, "rb");
        if (mirror_file) {
            brain->mirror_neurons = mirror_neurons_load(mirror_file);
            fclose(mirror_file);
            // Set brain reference for neuromodulation integration
            if (brain->mirror_neurons) {
                mirror_neurons_set_brain(brain->mirror_neurons, brain);
            }
        }
    }

    fclose(meta_file);
    return true;

cleanup:
    // Free any allocated labels before the failed one
    for (uint32_t j = 0; j < i; j++) {
        nimcp_free(brain->output_labels[j]);
    }
    nimcp_free(brain->output_labels);
    brain->output_labels = NULL;
    fclose(meta_file);
    return false;
}

/**
 * @brief Load brain from file
 *
 * WHY: Restores saved brain state
 * Reconstructs network and metadata
 *
 * COMPLEXITY: O(n*c) where n = neurons, c = connections
 *
 * @param filepath Path to load from
 * @return Brain handle or NULL on error
 */
brain_t brain_load(const char* filepath)
{
    // Guard: Validate filepath
    if (!filepath) {
        set_error("Null filepath provided to brain_load");
        return NULL;
    }

    // Load adaptive network
    adaptive_network_t network = adaptive_network_load(filepath);
    if (!network) {
        set_error("Failed to load adaptive network from %s", filepath);
        return NULL;
    }

    // Allocate brain structure
    brain_t brain = allocate_brain();
    if (!brain) {
        adaptive_network_destroy(network);
        return NULL;
    }

    brain->network = network;

    // Load metadata
    if (!nimcp_brain_load_metadata(brain, filepath)) {
        // Use defaults if no metadata
        brain->config.size = BRAIN_SIZE_SMALL;
        brain->config.task = BRAIN_TASK_CLASSIFICATION;
        brain->config.learning_rate = 0.01f;
        brain->config.sparsity_target = 0.8f;
        brain->config.enable_explanations = true;
        snprintf(brain->config.task_name, sizeof(brain->config.task_name), "loaded_brain");
    }

    // CRITICAL: Restore actual dimensions from saved network config
    // This is essential for resize+persistence compatibility
    const adaptive_network_config_t* net_config = adaptive_network_get_config(network);
    if (net_config) {
        brain->config.num_inputs = net_config->base_config.input_size;
        brain->config.num_outputs = net_config->base_config.output_size;
    } else {
        // Fallback if config not available
        brain->config.num_inputs = 1;
        brain->config.num_outputs = 1;
    }

    // Create strategy for task
    brain->strategy = strategy_create(brain->config.task);
    if (!brain->strategy) {
        set_error("Failed to create task strategy");
        brain_destroy(brain);
        return NULL;
    }

    // Initialize statistics
    init_brain_stats(&brain->stats, brain->config.task_name, brain->config.size,
                     brain->config.num_inputs, brain->config.learning_rate);

    // Re-initialize mirror neurons if enabled but not loaded from file
    // (Handles case where brain was saved without mirror neurons but loaded config has them enabled)
    if (brain->config.enable_mirror_neurons && !brain->mirror_neurons) {
        fprintf(stderr, "[INFO] Re-initializing mirror neurons from config (enable_mirror_neurons=true but not in save file)\n");
        if (!init_mirror_neurons(brain)) {
            fprintf(stderr, "[WARN] Failed to re-initialize mirror neurons\n");
            // Non-fatal: continue without mirror neurons
        }
    }

    // Re-initialize quantum annealer if enabled but not initialized
    // (Handles case where brain was saved without quantum annealer but loaded config has it enabled)
    if (brain->config.enable_quantum_annealing && !brain->quantum_annealer) {
        fprintf(stderr, "[INFO] Re-initializing quantum annealer from config (enable_quantum_annealing=true but not in save file)\n");
        quantum_annealing_config_t qa_config = {
            .initial_temperature = brain->config.annealing_temperature_init,
            .final_temperature = brain->config.annealing_temperature_final,
            .num_iterations = brain->config.annealing_steps,
            .cooling_schedule = COOLING_EXPONENTIAL,
            .quantum_strength = 0.5f,
            .enable_tunneling = true,
            .seed = (uint32_t)time(NULL)
        };

        brain->quantum_annealer = quantum_annealer_create(&qa_config);
        if (!brain->quantum_annealer) {
            fprintf(stderr, "[WARN] Failed to re-initialize quantum annealer\n");
            brain->config.enable_quantum_annealing = false;
        } else {
            fprintf(stderr, "[INFO] Quantum annealer re-initialized successfully\n");
        }
    }

    // Re-initialize glial subsystem if enabled but not initialized
    // (Handles case where brain was saved with glial but loaded without it)
    if (brain->config.enable_glial && !brain->glial) {
        fprintf(stderr, "[INFO] Re-initializing glial subsystem from config (enable_glial=true but not in save file)\n");
        if (!init_glial_subsystem(brain)) {
            fprintf(stderr, "[WARN] Failed to re-initialize glial subsystem\n");
            brain->config.enable_glial = false;
        } else {
            fprintf(stderr, "[INFO] Glial subsystem re-initialized successfully\n");

            // Re-initialize spatial neuromodulator system if glial was created
            // Spatial neuromod is part of glial integration, so always try to initialize it
            if (brain->glial) {
                fprintf(stderr, "[INFO] Re-initializing spatial neuromodulator system\n");
                if (!init_spatial_neuromod_system(brain)) {
                    fprintf(stderr, "[WARN] Failed to re-initialize spatial neuromodulator system\n");
                    // Non-fatal: continue without spatial neuromod
                }
            }
        }
    }

    brain_clear_error();
    return brain;
}

//=============================================================================
// Snapshot API - Named State Snapshots
//=============================================================================

/**
 * @brief Create snapshot directory if it doesn't exist
 *
 * @param snapshot_dir Directory path
 * @return true on success, false on error
 */
static bool ensure_snapshot_dir(const char* snapshot_dir)
{
    if (!snapshot_dir) {
        return false;
    }

    // Try to create directory (will fail silently if already exists)
    #ifdef _WIN32
    _mkdir(snapshot_dir);
    #else
    mkdir(snapshot_dir, 0755);
    #endif

    return true;
}

/**
 * @brief Get default snapshot directory
 *
 * @param brain Brain instance
 * @return Snapshot directory path
 */
static const char* get_snapshot_dir(brain_t brain)
{
    if (brain->config.snapshot_dir) {
        return brain->config.snapshot_dir;
    }
    return "./snapshots";  // Default
}

bool brain_save_snapshot(brain_t brain, const char* name, const char* description)
{
    // Guard: Validate parameters
    if (!brain || !name) {
        set_error("Invalid parameters to brain_save_snapshot");
        return false;
    }

    // Ensure snapshot directory exists
    const char* snapshot_dir = get_snapshot_dir(brain);
    if (!ensure_snapshot_dir(snapshot_dir)) {
        set_error("Failed to create snapshot directory: %s", snapshot_dir);
        return false;
    }

    // Generate snapshot filename with timestamp
    time_t now = time(NULL);
    char snapshot_path[1024];
    snprintf(snapshot_path, sizeof(snapshot_path), "%s/%s_%ld.snapshot",
             snapshot_dir, name, (long)now);

    // Save brain state to snapshot file
    if (!brain_save(brain, snapshot_path)) {
        set_error("Failed to save snapshot to %s", snapshot_path);
        return false;
    }

    // Save snapshot metadata
    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s/%s_%ld.snapshot.info",
             snapshot_dir, name, (long)now);
    FILE* meta_file = fopen(meta_path, "w");
    if (meta_file) {
        fprintf(meta_file, "name=%s\n", name);
        fprintf(meta_file, "timestamp=%ld\n", (long)now);
        if (description) {
            fprintf(meta_file, "description=%s\n", description);
        }
        fprintf(meta_file, "compressed=%d\n", brain->config.compress_snapshots ? 1 : 0);
        fprintf(meta_file, "encrypted=%d\n", brain->config.encrypt_snapshots ? 1 : 0);
        fclose(meta_file);
    }

    brain_clear_error();
    return true;
}

brain_t brain_restore_snapshot(brain_t brain, const char* name)
{
    // Guard: Validate parameters
    if (!name) {
        set_error("Null snapshot name provided");
        return NULL;
    }

    // Get snapshot directory (use default if brain is NULL)
    const char* snapshot_dir = brain ? get_snapshot_dir(brain) : "snapshots";

    // Find most recent snapshot with this name
    // Snapshots are named: {name}_{timestamp}.snapshot
    // We need to find the one with the highest timestamp
    DIR* dir = opendir(snapshot_dir);
    if (!dir) {
        set_error("Failed to open snapshot directory: %s", snapshot_dir);
        return NULL;
    }

    char best_snapshot[1024] = {0};
    time_t best_timestamp = 0;
    size_t name_len = strlen(name);

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        // Check if this is a snapshot file matching our name
        if (strncmp(entry->d_name, name, name_len) != 0) {
            continue;
        }

        // Check if it has the format: name_timestamp.snapshot
        if (entry->d_name[name_len] != '_') {
            continue;
        }

        size_t entry_len = strlen(entry->d_name);
        if (entry_len < 9 || strcmp(entry->d_name + entry_len - 9, ".snapshot") != 0) {
            continue;
        }

        // Extract timestamp
        char* timestamp_str = entry->d_name + name_len + 1;
        time_t timestamp = strtol(timestamp_str, NULL, 10);

        // Keep the most recent snapshot
        if (timestamp > best_timestamp) {
            best_timestamp = timestamp;
            strncpy(best_snapshot, entry->d_name, sizeof(best_snapshot) - 1);
        }
    }

    closedir(dir);

    if (best_snapshot[0] == '\0') {
        set_error("No snapshot found with name: %s", name);
        return NULL;
    }

    // Load brain from snapshot
    char snapshot_path[1024];
    snprintf(snapshot_path, sizeof(snapshot_path), "%s/%s", snapshot_dir, best_snapshot);

    brain_t loaded_brain = brain_load(snapshot_path);
    if (!loaded_brain) {
        set_error("Failed to load snapshot: %s", snapshot_path);
        return NULL;
    }

    // If brain provided, we'd need to copy state into it
    // For now, return new brain instance
    if (brain) {
        fprintf(stderr, "WARNING: In-place restore not yet implemented, returning new brain instance\n");
    }

    brain_clear_error();
    return loaded_brain;
}

bool brain_list_snapshots(brain_t brain, brain_snapshot_info_t* infos,
                         uint32_t max_count, uint32_t* out_count)
{
    // Guard: Validate parameters
    if (!brain || !infos || !out_count) {
        set_error("Invalid parameters to brain_list_snapshots");
        return false;
    }

    *out_count = 0;

    // Get snapshot directory
    const char* snapshot_dir = get_snapshot_dir(brain);
    if (!snapshot_dir) {
        set_error("Failed to get snapshot directory");
        return false;
    }

    // Open directory for scanning
    DIR* dir = opendir(snapshot_dir);
    if (!dir) {
        // Directory doesn't exist - not an error, just no snapshots
        brain_clear_error();
        return true;
    }

    // Scan directory for .snapshot files
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL && *out_count < max_count) {
        // Look for .snapshot files
        size_t name_len = strlen(entry->d_name);
        if (name_len < 9 || strcmp(entry->d_name + name_len - 9, ".snapshot") != 0) {
            continue;  // Not a snapshot file
        }

        // Read metadata from .info file
        char info_path[1024];
        snprintf(info_path, sizeof(info_path), "%s/%s.info", snapshot_dir, entry->d_name);

        FILE* info_file = fopen(info_path, "r");
        if (!info_file) {
            continue;  // No metadata, skip this snapshot
        }

        brain_snapshot_info_t* info = &infos[*out_count];
        memset(info, 0, sizeof(brain_snapshot_info_t));

        // Parse metadata file
        char line[1024];
        while (fgets(line, sizeof(line), info_file)) {
            char* equals = strchr(line, '=');
            if (!equals) continue;

            *equals = '\0';
            char* key = line;
            char* value = equals + 1;

            // Trim newline from value
            char* newline = strchr(value, '\n');
            if (newline) *newline = '\0';

            if (strcmp(key, "name") == 0) {
                strncpy(info->name, value, sizeof(info->name) - 1);
            } else if (strcmp(key, "description") == 0) {
                strncpy(info->description, value, sizeof(info->description) - 1);
            } else if (strcmp(key, "timestamp") == 0) {
                info->timestamp = (uint64_t)strtol(value, NULL, 10);
            } else if (strcmp(key, "compressed") == 0) {
                info->is_compressed = (atoi(value) != 0);
            } else if (strcmp(key, "encrypted") == 0) {
                info->is_encrypted = (atoi(value) != 0);
            }
        }

        fclose(info_file);

        // Get file size
        char snapshot_path[1024];
        snprintf(snapshot_path, sizeof(snapshot_path), "%s/%s", snapshot_dir, entry->d_name);
        struct stat st;
        if (stat(snapshot_path, &st) == 0) {
            info->file_size = (uint32_t)st.st_size;
        }

        (*out_count)++;
    }

    closedir(dir);

    brain_clear_error();
    return true;
}

bool brain_delete_snapshot(brain_t brain, const char* name)
{
    // Guard: Validate parameters
    if (!brain || !name) {
        set_error("Invalid parameters to brain_delete_snapshot");
        return false;
    }

    const char* snapshot_dir = get_snapshot_dir(brain);

    // Find most recent snapshot with this name
    // Snapshots are named: {name}_{timestamp}.snapshot
    DIR* dir = opendir(snapshot_dir);
    if (!dir) {
        set_error("Failed to open snapshot directory: %s", snapshot_dir);
        return false;
    }

    char best_snapshot[1024] = {0};
    time_t best_timestamp = 0;
    size_t name_len = strlen(name);

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        // Check if this is a snapshot file matching our name
        if (strncmp(entry->d_name, name, name_len) != 0) {
            continue;
        }

        // Check if it has the format: name_timestamp.snapshot
        if (entry->d_name[name_len] != '_') {
            continue;
        }

        size_t entry_len = strlen(entry->d_name);
        if (entry_len < 9 || strcmp(entry->d_name + entry_len - 9, ".snapshot") != 0) {
            continue;
        }

        // Extract timestamp
        char* timestamp_str = entry->d_name + name_len + 1;
        time_t timestamp = strtol(timestamp_str, NULL, 10);

        // Keep the most recent snapshot
        if (timestamp > best_timestamp) {
            best_timestamp = timestamp;
            strncpy(best_snapshot, entry->d_name, sizeof(best_snapshot) - 1);
        }
    }

    closedir(dir);

    if (best_snapshot[0] == '\0') {
        set_error("No snapshot found with name: %s", name);
        return false;
    }

    // Delete snapshot file
    char snapshot_path[1024];
    snprintf(snapshot_path, sizeof(snapshot_path), "%s/%s", snapshot_dir, best_snapshot);

    if (remove(snapshot_path) != 0) {
        set_error("Failed to delete snapshot: %s", snapshot_path);
        return false;
    }

    // Delete metadata file if it exists
    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s.info", snapshot_path);
    remove(meta_path);  // Ignore error

    // Delete .meta file if it exists
    snprintf(meta_path, sizeof(meta_path), "%s.meta", snapshot_path);
    remove(meta_path);  // Ignore error

    // Delete .knowledge file if it exists
    char knowledge_path[1024];
    snprintf(knowledge_path, sizeof(knowledge_path), "%s.knowledge", snapshot_path);
    remove(knowledge_path);  // Ignore error

    brain_clear_error();
    return true;
}
