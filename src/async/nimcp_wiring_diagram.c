/**
 * @file nimcp_wiring_diagram.c
 * @brief KG-Based Runtime Module Wiring Diagram System Implementation
 * @version 1.0.0
 * @date 2025-12-31
 *
 * WHAT: Implementation of dynamic module wiring using JSONL diagrams
 * WHY:  Enable self-assembling module topology without hardcoded dependencies
 * HOW:  Parse JSONL files, merge by priority, maintain module config cache
 *
 * @author NIMCP Development Team
 */

#include "async/nimcp_wiring_diagram.h"
#include "core/brain/nimcp_brain_kg.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/logging/nimcp_logging.h"
#include <cjson/cJSON.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define WIRING_LINE_BUFFER_SIZE    4096
#define WIRING_INITIAL_CAPACITY    64
#define WIRING_ARRAY_GROW_FACTOR   2

/* Subsystem names for file loading */
static const char* SUBSYSTEM_NAMES[] = {
    "core",
    "ethics",
    "perception",
    "cognition",
    "memory",
    "emotion",
    "immune",
    "plasticity",
    "recursive",
    "social"
};

/* Platform tier names for file loading */
static const char* TIER_NAMES[] = {
    "basic",
    "minimal",
    "constrained",
    "medium",
    "full",
    "neuromorphic",
    "quantum"
};

/* Hardware flag names for file loading */
static const char* HW_FLAG_NAMES[] = {
    "cuda",
    "rocm",
    "oneapi",
    "metal",
    "opencl",
    "loihi",
    "spinnaker",
    "akida",
    "brainscales",
    "tpu",
    "npu"
};

/* ============================================================================
 * Internal Structures
 * ============================================================================ */

/**
 * @brief Internal wiring diagram structure
 *
 * WHAT: Main state for the wiring diagram manager
 * WHY:  Track loaded modules, persistence settings, thread safety
 * HOW:  Dynamic array of module configs with mutex protection
 */
struct wiring_diagram {
    /* Configuration */
    char base_path[WIRING_MAX_PATH_LENGTH];  /**< Base path to wiring directory */
    bool auto_persist;                        /**< Auto-save on changes */
    bool dirty;                               /**< Changes pending persistence */

    /* Module storage */
    wiring_module_config_t** module_configs;  /**< Array of module config pointers */
    uint32_t module_count;                    /**< Number of loaded modules */
    uint32_t module_capacity;                 /**< Array capacity */

    /* Thread safety */
    nimcp_platform_mutex_t* mutex;            /**< Protects all state */
};

/* ============================================================================
 * Forward Declarations
 * ============================================================================ */

static int ensure_capacity(wiring_diagram_t* wd);
static wiring_module_config_t* find_module_by_name_unlocked(
    const wiring_diagram_t* wd, const char* name);
static wiring_module_config_t* find_module_by_id_unlocked(
    const wiring_diagram_t* wd, bio_module_id_t id);
static int parse_jsonl_file(wiring_diagram_t* wd, const char* filepath);
static int parse_entity_line(wiring_diagram_t* wd, const char* line);
static int parse_relation_line(wiring_diagram_t* wd, const char* line);
static int add_to_id_array(bio_module_id_t** array, uint32_t* count,
    uint32_t* capacity, bio_module_id_t id);
static int add_to_msgtype_array(bio_message_type_t** array, uint32_t* count,
    uint32_t* capacity, bio_message_type_t msg_type);
static wiring_subsystem_t parse_subsystem_string(const char* str);
static bio_module_id_t parse_module_id(const char* name);
static int persist_subsystem_unlocked(wiring_diagram_t* wd, wiring_subsystem_t subsystem);
static void auto_persist_if_enabled(wiring_diagram_t* wd);
static void write_entity_json(FILE* file, const wiring_module_config_t* config,
    wiring_subsystem_t subsystem);
static void write_relation_json(FILE* file, const char* from, const char* to,
    const char* rel_type);
static void populate_config_from_json(wiring_module_config_t* config, cJSON* json,
    const char* name);

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

/**
 * @brief Create wiring diagram manager
 *
 * WHAT: Initialize wiring diagram system
 * WHY:  Central manager for loading and querying wiring configurations
 * HOW:  Allocate structures, set base path, prepare for loading
 */
wiring_diagram_t* wiring_diagram_create(const char* base_path) {
    wiring_diagram_t* wd = (wiring_diagram_t*)nimcp_calloc(
        1, sizeof(wiring_diagram_t));
    if (!wd) {
        NIMCP_LOGGING_ERROR("Failed to allocate wiring diagram");
        return NULL;
    }

    /* Set base path */
    if (base_path && base_path[0] != '\0') {
        strncpy(wd->base_path, base_path, WIRING_MAX_PATH_LENGTH - 1);
    } else {
        strncpy(wd->base_path, WIRING_DEFAULT_PATH, WIRING_MAX_PATH_LENGTH - 1);
    }
    wd->base_path[WIRING_MAX_PATH_LENGTH - 1] = '\0';

    /* Create mutex */
    wd->mutex = nimcp_platform_mutex_create();
    if (!wd->mutex) {
        NIMCP_LOGGING_ERROR("Failed to create wiring diagram mutex");
        nimcp_free(wd);
        return NULL;
    }

    /* Allocate initial module array */
    wd->module_capacity = WIRING_INITIAL_CAPACITY;
    wd->module_configs = (wiring_module_config_t**)nimcp_calloc(
        wd->module_capacity, sizeof(wiring_module_config_t*));
    if (!wd->module_configs) {
        NIMCP_LOGGING_ERROR("Failed to allocate module config array");
        nimcp_platform_mutex_destroy(wd->mutex);
        nimcp_free(wd);
        return NULL;
    }

    /* Initialize state */
    wd->module_count = 0;
    wd->auto_persist = true;  /* Default: auto-persist enabled */
    wd->dirty = false;

    NIMCP_LOGGING_DEBUG("Wiring diagram created with base path: %s", wd->base_path);
    return wd;
}

/**
 * @brief Destroy wiring diagram manager
 *
 * WHAT: Clean up wiring diagram resources
 * WHY:  Proper resource deallocation
 * HOW:  Free all module configs, destroy mutex, free manager
 */
void wiring_diagram_destroy(wiring_diagram_t* wd) {
    if (!wd) return;

    /* Persist if dirty */
    if (wd->dirty && wd->auto_persist) {
        wiring_diagram_persist(wd);
    }

    /* Free all module configs */
    for (uint32_t i = 0; i < wd->module_count; i++) {
        if (wd->module_configs[i]) {
            wiring_module_config_cleanup(wd->module_configs[i]);
            nimcp_free(wd->module_configs[i]);
        }
    }
    nimcp_free(wd->module_configs);

    /* Destroy mutex */
    if (wd->mutex) {
        nimcp_platform_mutex_destroy(wd->mutex);
    }

    nimcp_free(wd);
    NIMCP_LOGGING_DEBUG("Wiring diagram destroyed");
}

/* ============================================================================
 * Loading Implementation
 * ============================================================================ */

/**
 * @brief Load master wiring diagram
 *
 * WHAT: Load base module wiring from master.jsonl
 * WHY:  Establish baseline connectivity for all modules
 * HOW:  Parse JSONL, populate module configs
 */
int wiring_diagram_load_master(wiring_diagram_t* wd) {
    if (!wd) return -1;

    char filepath[WIRING_MAX_PATH_LENGTH * 2];
    snprintf(filepath, sizeof(filepath), "%s/master.jsonl", wd->base_path);

    nimcp_platform_mutex_lock(wd->mutex);
    int result = parse_jsonl_file(wd, filepath);
    nimcp_platform_mutex_unlock(wd->mutex);

    if (result == 0) {
        NIMCP_LOGGING_INFO("Loaded master wiring diagram");
    }
    return result;
}

/**
 * @brief Load subsystem-specific wiring
 *
 * WHAT: Load wiring for specific subsystem from subsystems/<name>.jsonl
 * WHY:  Add subsystem-specific connectivity
 * HOW:  Merge subsystem entries with existing configs
 */
int wiring_diagram_load_subsystem(wiring_diagram_t* wd, wiring_subsystem_t subsystem) {
    if (!wd) return -1;
    if (subsystem >= WIRING_SUBSYSTEM_COUNT) return -1;

    char filepath[WIRING_MAX_PATH_LENGTH * 2];
    snprintf(filepath, sizeof(filepath), "%s/subsystems/%s.jsonl",
        wd->base_path, SUBSYSTEM_NAMES[subsystem]);

    nimcp_platform_mutex_lock(wd->mutex);
    int result = parse_jsonl_file(wd, filepath);
    nimcp_platform_mutex_unlock(wd->mutex);

    /* Missing file is not an error - subsystem may not have custom wiring */
    if (result == -1) {
        return 0;
    }

    NIMCP_LOGGING_DEBUG("Loaded subsystem wiring: %s", SUBSYSTEM_NAMES[subsystem]);
    return 0;
}

/**
 * @brief Load all subsystem wiring diagrams
 *
 * WHAT: Load all subsystem/*.jsonl files
 * WHY:  Convenience function to load all subsystems at once
 * HOW:  Iterate WIRING_SUBSYSTEM_* and load each
 */
int wiring_diagram_load_all_subsystems(wiring_diagram_t* wd) {
    if (!wd) return -1;

    for (int i = 0; i < WIRING_SUBSYSTEM_COUNT; i++) {
        wiring_diagram_load_subsystem(wd, (wiring_subsystem_t)i);
    }

    return 0;
}

/**
 * @brief Load platform tier-specific wiring
 *
 * WHAT: Load wiring for specific platform tier from platforms/<tier>.jsonl
 * WHY:  Apply tier-specific overrides (enable/disable modules)
 * HOW:  Merge tier entries with existing configs
 */
int wiring_diagram_load_platform(wiring_diagram_t* wd, platform_tier_t tier) {
    if (!wd) return -1;
    if (tier > PLATFORM_TIER_FULL) return -1;

    char filepath[WIRING_MAX_PATH_LENGTH * 2];
    snprintf(filepath, sizeof(filepath), "%s/platforms/%s.jsonl",
        wd->base_path, TIER_NAMES[tier]);

    nimcp_platform_mutex_lock(wd->mutex);
    int result = parse_jsonl_file(wd, filepath);
    nimcp_platform_mutex_unlock(wd->mutex);

    /* Missing file is not an error */
    if (result == -1) {
        return 0;
    }

    NIMCP_LOGGING_DEBUG("Loaded platform wiring: %s", TIER_NAMES[tier]);
    return 0;
}

/**
 * @brief Load hardware-specific wiring
 *
 * WHAT: Load wiring for specific hardware from hardware/<hw>.jsonl
 * WHY:  Apply hardware-specific configurations
 * HOW:  Load based on hardware flags (may load multiple files)
 */
int wiring_diagram_load_hardware(wiring_diagram_t* wd, wiring_hardware_flags_t hw) {
    if (!wd) return -1;
    if (hw == WIRING_HW_NONE) return 0;

    char filepath[WIRING_MAX_PATH_LENGTH * 2];

    nimcp_platform_mutex_lock(wd->mutex);

    /* Load file for each set hardware flag */
    for (int i = 0; i < 11; i++) {
        if (hw & (1 << i)) {
            snprintf(filepath, sizeof(filepath), "%s/hardware/%s.jsonl",
                wd->base_path, HW_FLAG_NAMES[i]);
            parse_jsonl_file(wd, filepath);  /* Ignore missing files */
        }
    }

    nimcp_platform_mutex_unlock(wd->mutex);
    return 0;
}

/**
 * @brief Load custom wiring diagram
 *
 * WHAT: Load user-provided custom wiring from custom/<name>.jsonl
 * WHY:  Allow user overrides and extensions
 * HOW:  Merge custom entries (override existing, add new)
 */
int wiring_diagram_load_custom(wiring_diagram_t* wd, const char* filename) {
    if (!wd || !filename) return -1;

    char filepath[WIRING_MAX_PATH_LENGTH * 2];
    snprintf(filepath, sizeof(filepath), "%s/custom/%s", wd->base_path, filename);

    nimcp_platform_mutex_lock(wd->mutex);
    int result = parse_jsonl_file(wd, filepath);
    nimcp_platform_mutex_unlock(wd->mutex);

    return result;
}

/**
 * @brief Load all custom wiring diagrams
 *
 * WHAT: Load all custom/*.jsonl files
 * WHY:  Apply all user customizations
 * HOW:  Iterate custom/ directory and load each
 */
int wiring_diagram_load_all_custom(wiring_diagram_t* wd) {
    if (!wd) return -1;

    char dirpath[WIRING_MAX_PATH_LENGTH * 2];
    snprintf(dirpath, sizeof(dirpath), "%s/custom", wd->base_path);

    DIR* dir = opendir(dirpath);
    if (!dir) return 0;  /* No custom directory is fine */

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Skip hidden files and directories */
        if (entry->d_name[0] == '.') continue;

        /* Only process .jsonl files */
        size_t len = strlen(entry->d_name);
        if (len < 7 || strcmp(entry->d_name + len - 6, ".jsonl") != 0) {
            continue;
        }

        wiring_diagram_load_custom(wd, entry->d_name);
    }

    closedir(dir);
    return 0;
}

/**
 * @brief Load composite wiring for hardware profile
 *
 * WHAT: Load and merge all applicable diagrams for given profile
 * WHY:  One-shot loading with correct priority ordering
 * HOW:  Load master -> subsystems -> platform -> hardware -> custom
 */
int wiring_diagram_load_for_profile(
    wiring_diagram_t* wd,
    const wiring_hardware_profile_t* profile
) {
    if (!wd || !profile) return -1;

    /* 1. Load master (base wiring) */
    wiring_diagram_load_master(wd);

    /* 2. Load all subsystems */
    wiring_diagram_load_all_subsystems(wd);

    /* 3. Load platform tier */
    wiring_diagram_load_platform(wd, profile->tier);

    /* 4. Load hardware-specific */
    wiring_diagram_load_hardware(wd, profile->hw_flags);

    /* 5. Load all custom (highest priority) */
    wiring_diagram_load_all_custom(wd);

    NIMCP_LOGGING_INFO("Loaded wiring for profile: tier=%s, hw=0x%x",
        TIER_NAMES[profile->tier], profile->hw_flags);

    return 0;
}

/* ============================================================================
 * Query Implementation
 * ============================================================================ */

/**
 * @brief Get module wiring configuration
 *
 * WHAT: Query wiring configuration for a specific module
 * WHY:  Enable modules to discover their connectivity at runtime
 * HOW:  Lookup module by name in loaded configurations
 */
int wiring_diagram_get_module_config(
    const wiring_diagram_t* wd,
    const char* module_name,
    wiring_module_config_t* config
) {
    if (!wd || !module_name || !config) return -1;

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)wd->mutex);

    wiring_module_config_t* found = find_module_by_name_unlocked(wd, module_name);
    if (!found) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
        return -1;
    }

    /* Copy the config (shallow copy - arrays are shared) */
    *config = *found;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
    return 0;
}

/**
 * @brief Get module wiring configuration by ID
 *
 * WHAT: Query wiring configuration by bio-async module ID
 * WHY:  Lookup by ID is often more convenient than by name
 * HOW:  Lookup module by ID in loaded configurations
 */
int wiring_diagram_get_module_config_by_id(
    const wiring_diagram_t* wd,
    bio_module_id_t module_id,
    wiring_module_config_t* config
) {
    if (!wd || !config) return -1;

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)wd->mutex);

    wiring_module_config_t* found = find_module_by_id_unlocked(wd, module_id);
    if (!found) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
        return -1;
    }

    *config = *found;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
    return 0;
}

/**
 * @brief Check if module is available for current profile
 *
 * WHAT: Test if module should be enabled for given hardware profile
 * WHY:  Determine if module meets tier/hardware requirements
 * HOW:  Check min_tier and required_hw against profile
 */
bool wiring_diagram_module_available(
    const wiring_diagram_t* wd,
    const char* module_name,
    const wiring_hardware_profile_t* profile
) {
    if (!wd || !module_name || !profile) return false;

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)wd->mutex);

    wiring_module_config_t* config = find_module_by_name_unlocked(wd, module_name);
    if (!config) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
        return false;
    }

    /* Check tier requirement */
    if (profile->tier < config->min_tier) {
        nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
        return false;
    }

    /* Check hardware requirements */
    if (config->required_hw != WIRING_HW_NONE) {
        if ((profile->hw_flags & config->required_hw) != config->required_hw) {
            nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
            return false;
        }
    }

    /* Check enabled flag */
    bool available = config->enabled;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
    return available;
}

/**
 * @brief Get all modules in a subsystem
 *
 * WHAT: Query all modules belonging to a subsystem
 * WHY:  Enable subsystem-level operations
 * HOW:  Filter modules by subsystem field
 */
uint32_t wiring_diagram_get_subsystem_modules(
    const wiring_diagram_t* wd,
    wiring_subsystem_t subsystem,
    const char** module_names,
    uint32_t max_modules
) {
    if (!wd || !module_names || max_modules == 0) return 0;

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)wd->mutex);

    uint32_t count = 0;
    for (uint32_t i = 0; i < wd->module_count && count < max_modules; i++) {
        if (wd->module_configs[i] &&
            wd->module_configs[i]->subsystem == subsystem) {
            module_names[count++] = wd->module_configs[i]->module_name;
        }
    }

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
    return count;
}

/**
 * @brief Get all loaded modules
 *
 * WHAT: Enumerate all modules in the wiring diagram
 * WHY:  System introspection
 * HOW:  Return array of all module names
 */
uint32_t wiring_diagram_get_all_modules(
    const wiring_diagram_t* wd,
    const char** module_names,
    uint32_t max_modules
) {
    if (!wd || !module_names || max_modules == 0) return 0;

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)wd->mutex);

    uint32_t count = 0;
    for (uint32_t i = 0; i < wd->module_count && count < max_modules; i++) {
        if (wd->module_configs[i]) {
            module_names[count++] = wd->module_configs[i]->module_name;
        }
    }

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
    return count;
}

/* ============================================================================
 * Dynamic Update Implementation
 * ============================================================================ */

/**
 * @brief Add or update module in wiring diagram
 *
 * WHAT: Add new module or update existing module configuration
 * WHY:  Enable dynamic wiring modifications at runtime
 * HOW:  Insert/update in module map, auto-persist if enabled
 */
int wiring_diagram_add_module(
    wiring_diagram_t* wd,
    const char* module_name,
    const wiring_module_config_t* config
) {
    if (!wd || !module_name || !config) return -1;

    /* Reject empty module names */
    if (module_name[0] == '\0') return -1;

    nimcp_platform_mutex_lock(wd->mutex);

    /* Check if module exists */
    wiring_module_config_t* existing = find_module_by_name_unlocked(wd, module_name);
    if (existing) {
        /* Update existing */
        wiring_module_config_cleanup(existing);
        *existing = *config;
        strncpy(existing->module_name, module_name, sizeof(existing->module_name) - 1);
    } else {
        /* Add new */
        if (ensure_capacity(wd) != 0) {
            nimcp_platform_mutex_unlock(wd->mutex);
            return -1;
        }

        wiring_module_config_t* new_config = (wiring_module_config_t*)nimcp_calloc(
            1, sizeof(wiring_module_config_t));
        if (!new_config) {
            nimcp_platform_mutex_unlock(wd->mutex);
            return -1;
        }

        *new_config = *config;
        strncpy(new_config->module_name, module_name,
            sizeof(new_config->module_name) - 1);

        wd->module_configs[wd->module_count++] = new_config;
    }

    wd->dirty = true;
    nimcp_platform_mutex_unlock(wd->mutex);

    auto_persist_if_enabled(wd);
    return 0;
}

/**
 * @brief Add relation between modules
 *
 * WHAT: Add a wiring relation between two modules
 * WHY:  Enable dynamic connectivity changes
 * HOW:  Update source module's sends_to/depends_on arrays
 */
int wiring_diagram_add_relation(
    wiring_diagram_t* wd,
    const char* from,
    const char* to,
    wiring_relation_type_t relation_type
) {
    if (!wd || !from || !to) return -1;

    nimcp_platform_mutex_lock(wd->mutex);

    wiring_module_config_t* from_config = find_module_by_name_unlocked(wd, from);
    wiring_module_config_t* to_config = find_module_by_name_unlocked(wd, to);

    if (!from_config || !to_config) {
        nimcp_platform_mutex_unlock(wd->mutex);
        return -1;
    }

    int result = 0;
    switch (relation_type) {
        case WIRING_RELATION_DEPENDS_ON:
            result = add_to_id_array(&from_config->depends_on,
                &from_config->depends_on_count,
                &from_config->depends_on_capacity,
                to_config->module_id);
            break;

        case WIRING_RELATION_SENDS_TO:
            result = add_to_id_array(&from_config->sends_to,
                &from_config->sends_to_count,
                &from_config->sends_to_capacity,
                to_config->module_id);
            break;

        case WIRING_RELATION_RECEIVES_FROM:
            result = add_to_id_array(&from_config->receives_from,
                &from_config->receives_from_count,
                &from_config->receives_from_capacity,
                to_config->module_id);
            break;

        default:
            result = -1;
            break;
    }

    if (result == 0) {
        wd->dirty = true;
    }

    nimcp_platform_mutex_unlock(wd->mutex);

    if (result == 0) {
        auto_persist_if_enabled(wd);
    }
    return result;
}

/**
 * @brief Add message handler to module
 *
 * WHAT: Register that a module handles a message type
 * WHY:  Enable dynamic handler registration
 * HOW:  Add to module's handles_messages array
 */
int wiring_diagram_add_handler(
    wiring_diagram_t* wd,
    const char* module_name,
    bio_message_type_t message_type
) {
    if (!wd || !module_name) return -1;

    nimcp_platform_mutex_lock(wd->mutex);

    wiring_module_config_t* config = find_module_by_name_unlocked(wd, module_name);
    if (!config) {
        nimcp_platform_mutex_unlock(wd->mutex);
        return -1;
    }

    int result = add_to_msgtype_array(&config->handles_messages,
        &config->handles_message_count,
        &config->handles_message_capacity,
        message_type);

    if (result == 0) {
        wd->dirty = true;
    }

    nimcp_platform_mutex_unlock(wd->mutex);

    if (result == 0) {
        auto_persist_if_enabled(wd);
    }
    return result;
}

/**
 * @brief Remove module from wiring diagram
 *
 * WHAT: Remove a module and its relations
 * WHY:  Enable module removal at runtime
 * HOW:  Remove from map, clean up relations, auto-persist
 */
int wiring_diagram_remove_module(wiring_diagram_t* wd, const char* module_name) {
    if (!wd || !module_name) return -1;

    nimcp_platform_mutex_lock(wd->mutex);

    /* Find module index */
    int found_idx = -1;
    for (uint32_t i = 0; i < wd->module_count; i++) {
        if (wd->module_configs[i] &&
            strcmp(wd->module_configs[i]->module_name, module_name) == 0) {
            found_idx = (int)i;
            break;
        }
    }

    if (found_idx < 0) {
        nimcp_platform_mutex_unlock(wd->mutex);
        return -1;
    }

    /* Clean up and free */
    wiring_module_config_cleanup(wd->module_configs[found_idx]);
    nimcp_free(wd->module_configs[found_idx]);

    /* Shift remaining */
    if ((uint32_t)found_idx < wd->module_count - 1) {
        memmove(&wd->module_configs[found_idx],
            &wd->module_configs[found_idx + 1],
            (wd->module_count - found_idx - 1) * sizeof(wiring_module_config_t*));
    }

    wd->module_count--;
    wd->dirty = true;

    nimcp_platform_mutex_unlock(wd->mutex);

    auto_persist_if_enabled(wd);
    return 0;
}

/**
 * @brief Enable or disable module
 *
 * WHAT: Toggle module enabled state
 * WHY:  Selective activation without removing wiring
 * HOW:  Set module's enabled flag, auto-persist
 */
int wiring_diagram_set_enabled(
    wiring_diagram_t* wd,
    const char* module_name,
    bool enabled
) {
    if (!wd || !module_name) return -1;

    nimcp_platform_mutex_lock(wd->mutex);

    wiring_module_config_t* config = find_module_by_name_unlocked(wd, module_name);
    if (!config) {
        nimcp_platform_mutex_unlock(wd->mutex);
        return -1;
    }

    config->enabled = enabled;
    wd->dirty = true;

    nimcp_platform_mutex_unlock(wd->mutex);

    auto_persist_if_enabled(wd);
    return 0;
}

/* ============================================================================
 * Persistence Implementation
 * ============================================================================ */

/**
 * @brief Persist wiring diagram to disk
 *
 * WHAT: Save current wiring state to JSONL files
 * WHY:  Maintain configuration across restarts
 * HOW:  Write to subsystem files based on module membership
 */
int wiring_diagram_persist(wiring_diagram_t* wd) {
    if (!wd) return -1;

    nimcp_platform_mutex_lock(wd->mutex);

    /* Persist each subsystem */
    for (int i = 0; i < WIRING_SUBSYSTEM_COUNT; i++) {
        persist_subsystem_unlocked(wd, (wiring_subsystem_t)i);
    }

    wd->dirty = false;

    nimcp_platform_mutex_unlock(wd->mutex);

    NIMCP_LOGGING_DEBUG("Wiring diagram persisted");
    return 0;
}

/**
 * @brief Persist specific subsystem to disk
 *
 * WHAT: Save only specified subsystem's wiring
 * WHY:  Selective persistence for partial updates
 * HOW:  Write to subsystems/<name>.jsonl
 */
int wiring_diagram_persist_subsystem(wiring_diagram_t* wd, wiring_subsystem_t subsystem) {
    if (!wd) return -1;
    if (subsystem >= WIRING_SUBSYSTEM_COUNT) return -1;

    nimcp_platform_mutex_lock(wd->mutex);
    int result = persist_subsystem_unlocked(wd, subsystem);
    nimcp_platform_mutex_unlock(wd->mutex);

    return result;
}

/**
 * @brief Set auto-persist behavior
 *
 * WHAT: Enable or disable automatic persistence on changes
 * WHY:  Control when wiring is saved to disk
 * HOW:  Set internal flag (default: enabled)
 */
void wiring_diagram_set_auto_persist(wiring_diagram_t* wd, bool enabled) {
    if (!wd) return;

    nimcp_platform_mutex_lock(wd->mutex);
    wd->auto_persist = enabled;
    nimcp_platform_mutex_unlock(wd->mutex);
}

/**
 * @brief Check if auto-persist is enabled
 */
bool wiring_diagram_get_auto_persist(const wiring_diagram_t* wd) {
    if (!wd) return false;
    return wd->auto_persist;
}

/* ============================================================================
 * Hardware Detection Implementation (Stub)
 * ============================================================================ */

/**
 * @brief Detect hardware profile for current system
 *
 * WHAT: Auto-detect platform tier and available hardware
 * WHY:  Enable automatic wiring configuration selection
 * HOW:  Query system resources (stub implementation)
 */
/**
 * @brief Helper to check if a command exists and returns 0
 */
static bool check_command_exists(const char* cmd) {
    char buf[256];
    snprintf(buf, sizeof(buf), "which %s > /dev/null 2>&1", cmd);
    return (system(buf) == 0);
}

/**
 * @brief Helper to run a command and check if it succeeds
 */
static bool run_command_success(const char* cmd) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s > /dev/null 2>&1", cmd);
    return (system(buf) == 0);
}

/**
 * @brief Detect GPU hardware availability
 */
static wiring_hardware_flags_t detect_gpu_hardware(void) {
    wiring_hardware_flags_t flags = WIRING_HW_NONE;

    /* NVIDIA CUDA - check nvidia-smi or nvcc */
    if (check_command_exists("nvidia-smi") &&
        run_command_success("nvidia-smi")) {
        flags |= WIRING_HW_CUDA;
        NIMCP_LOGGING_INFO("Detected NVIDIA CUDA GPU");
    }

    /* AMD ROCm - check rocm-smi or rocminfo */
    if (check_command_exists("rocm-smi") ||
        check_command_exists("rocminfo")) {
        if (run_command_success("rocm-smi") ||
            run_command_success("rocminfo")) {
            flags |= WIRING_HW_ROCM;
            NIMCP_LOGGING_INFO("Detected AMD ROCm GPU");
        }
    }

    /* Intel oneAPI - check sycl-ls or icpx */
    if (check_command_exists("sycl-ls") &&
        run_command_success("sycl-ls")) {
        flags |= WIRING_HW_ONEAPI;
        NIMCP_LOGGING_INFO("Detected Intel oneAPI");
    }

#ifdef __APPLE__
    /* Apple Metal - available on macOS by default */
    flags |= WIRING_HW_METAL;
    NIMCP_LOGGING_INFO("Detected Apple Metal (macOS)");
#endif

    /* OpenCL - check clinfo */
    if (check_command_exists("clinfo") &&
        run_command_success("clinfo")) {
        flags |= WIRING_HW_OPENCL;
        NIMCP_LOGGING_INFO("Detected OpenCL");
    }

    return flags;
}

/**
 * @brief Detect neuromorphic hardware availability
 */
static wiring_hardware_flags_t detect_neuromorphic_hardware(void) {
    wiring_hardware_flags_t flags = WIRING_HW_NONE;

    /* Intel Loihi - check for nxsdk or lava */
    if (check_command_exists("nxsdk") ||
        run_command_success("python3 -c 'import nxsdk' 2>/dev/null") ||
        run_command_success("python3 -c 'import lava' 2>/dev/null")) {
        flags |= WIRING_HW_LOIHI;
        NIMCP_LOGGING_INFO("Detected Intel Loihi (nxsdk/lava)");
    }

    /* SpiNNaker - check for spynnaker */
    if (run_command_success("python3 -c 'import spynnaker' 2>/dev/null")) {
        flags |= WIRING_HW_SPINNAKER;
        NIMCP_LOGGING_INFO("Detected SpiNNaker");
    }

    /* BrainChip Akida - check for akida */
    if (run_command_success("python3 -c 'import akida' 2>/dev/null")) {
        flags |= WIRING_HW_AKIDA;
        NIMCP_LOGGING_INFO("Detected BrainChip Akida");
    }

    return flags;
}

/**
 * @brief Detect CPU cores
 */
static uint32_t detect_cpu_cores(void) {
#ifdef _SC_NPROCESSORS_ONLN
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores > 0) return (uint32_t)cores;
#endif

#ifdef __linux__
    /* Fallback: parse /proc/cpuinfo */
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        uint32_t count = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "processor", 9) == 0) count++;
        }
        fclose(f);
        if (count > 0) return count;
    }
#endif

    return 4;  /* Default fallback */
}

/**
 * @brief Detect available memory in MB
 */
static size_t detect_memory_mb(void) {
#ifdef _SC_PHYS_PAGES
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        return (size_t)((pages * page_size) / (1024 * 1024));
    }
#endif

#ifdef __linux__
    /* Fallback: parse /proc/meminfo */
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            size_t mem_kb;
            if (sscanf(line, "MemTotal: %zu kB", &mem_kb) == 1) {
                fclose(f);
                return mem_kb / 1024;
            }
        }
        fclose(f);
    }
#endif

    return 4096;  /* Default 4GB fallback */
}

/**
 * @brief Determine platform tier from available resources
 */
static platform_tier_t detect_platform_tier(size_t memory_mb, uint32_t cpu_cores,
                                            wiring_hardware_flags_t hw_flags) {
    /* FULL: 16+ cores, 32GB+ RAM, or GPU */
    if ((cpu_cores >= 16 && memory_mb >= 32768) ||
        (hw_flags & WIRING_HW_ANY_GPU)) {
        return PLATFORM_TIER_FULL;
    }

    /* MEDIUM: 4+ cores, 8GB+ RAM */
    if (cpu_cores >= 4 && memory_mb >= 8192) {
        return PLATFORM_TIER_MEDIUM;
    }

    /* CONSTRAINED: 2+ cores, 2GB+ RAM */
    if (cpu_cores >= 2 && memory_mb >= 2048) {
        return PLATFORM_TIER_CONSTRAINED;
    }

    /* MINIMAL: Everything else */
    return PLATFORM_TIER_MINIMAL;
}

int wiring_detect_hardware_profile(wiring_hardware_profile_t* profile) {
    if (!profile) return -1;

    /* Initialize with defaults first */
    wiring_get_default_profile(profile);

    /* Detect CPU and memory */
    profile->cpu_cores = detect_cpu_cores();
    profile->memory_mb = detect_memory_mb();

    NIMCP_LOGGING_INFO("Detected %u CPU cores, %zu MB memory",
        profile->cpu_cores, profile->memory_mb);

    /* Detect GPU hardware */
    profile->hw_flags = detect_gpu_hardware();

    /* Detect neuromorphic hardware */
    profile->hw_flags |= detect_neuromorphic_hardware();

    /* Determine platform tier based on resources */
    profile->tier = detect_platform_tier(
        profile->memory_mb,
        profile->cpu_cores,
        profile->hw_flags
    );

    NIMCP_LOGGING_INFO("Hardware profile: tier=%d, hw_flags=0x%x",
        (int)profile->tier, profile->hw_flags);

    return 0;
}

/**
 * @brief Get default hardware profile
 *
 * WHAT: Get safe default profile for unknown hardware
 * WHY:  Fallback when detection fails
 * HOW:  Return conservative CPU-only profile
 */
void wiring_get_default_profile(wiring_hardware_profile_t* profile) {
    if (!profile) return;

    memset(profile, 0, sizeof(*profile));
    /* Use conservative defaults - CONSTRAINED tier is safe fallback */
    profile->tier = PLATFORM_TIER_CONSTRAINED;
    profile->hw_flags = WIRING_HW_NONE;
    profile->memory_mb = 2048;
    profile->cpu_cores = 2;
    profile->gpu_compute_units = 0;
    profile->neuromorphic_cores = 0;
}

/* ============================================================================
 * Utility Implementation
 * ============================================================================ */

/**
 * @brief Convert subsystem enum to string
 */
const char* wiring_subsystem_to_string(wiring_subsystem_t subsystem) {
    if (subsystem >= WIRING_SUBSYSTEM_COUNT) return "unknown";
    return SUBSYSTEM_NAMES[subsystem];
}

/**
 * @brief Convert relation type to string
 */
const char* wiring_relation_to_string(wiring_relation_type_t relation) {
    static const char* RELATION_NAMES[] = {
        "DEPENDS_ON",
        "SENDS_TO",
        "RECEIVES_FROM",
        "HANDLES_MESSAGE",
        "BELONGS_TO",
        "REQUIRES_HW",
        "AVAILABLE_ON_TIER"
    };

    if (relation >= WIRING_RELATION_COUNT) return "unknown";
    return RELATION_NAMES[relation];
}

/**
 * @brief Convert hardware flags to string
 */
const char* wiring_hardware_flags_to_string(
    wiring_hardware_flags_t hw,
    char* buffer,
    size_t buffer_size
) {
    if (!buffer || buffer_size == 0) return "";

    buffer[0] = '\0';
    if (hw == WIRING_HW_NONE) {
        strncpy(buffer, "none", buffer_size - 1);
        return buffer;
    }

    size_t pos = 0;
    for (int i = 0; i < 11 && pos < buffer_size - 1; i++) {
        if (hw & (1 << i)) {
            if (pos > 0 && pos < buffer_size - 1) {
                buffer[pos++] = '|';
            }
            size_t len = strlen(HW_FLAG_NAMES[i]);
            if (pos + len < buffer_size) {
                strncpy(buffer + pos, HW_FLAG_NAMES[i], buffer_size - pos - 1);
                pos += len;
            }
        }
    }
    buffer[buffer_size - 1] = '\0';

    return buffer;
}

/**
 * @brief Initialize module config structure
 *
 * WHAT: Zero-initialize a module config with proper defaults
 * WHY:  Ensure clean initialization before use
 * HOW:  memset to zero, set defaults
 */
void wiring_module_config_init(wiring_module_config_t* config) {
    if (!config) return;

    memset(config, 0, sizeof(*config));
    config->min_tier = PLATFORM_TIER_MINIMAL;
    config->required_hw = WIRING_HW_NONE;
    config->enabled = true;
    config->discovered = false;
}

/**
 * @brief Clean up module config arrays
 *
 * WHAT: Free dynamically allocated arrays in module config
 * WHY:  Proper memory cleanup
 * HOW:  Free depends_on, sends_to, receives_from, handles_messages arrays
 */
void wiring_module_config_cleanup(wiring_module_config_t* config) {
    if (!config) return;

    if (config->depends_on) {
        nimcp_free(config->depends_on);
        config->depends_on = NULL;
    }
    if (config->sends_to) {
        nimcp_free(config->sends_to);
        config->sends_to = NULL;
    }
    if (config->receives_from) {
        nimcp_free(config->receives_from);
        config->receives_from = NULL;
    }
    if (config->handles_messages) {
        nimcp_free(config->handles_messages);
        config->handles_messages = NULL;
    }

    config->depends_on_count = 0;
    config->depends_on_capacity = 0;
    config->sends_to_count = 0;
    config->sends_to_capacity = 0;
    config->receives_from_count = 0;
    config->receives_from_capacity = 0;
    config->handles_message_count = 0;
    config->handles_message_capacity = 0;
}

/**
 * @brief Get wiring diagram statistics
 *
 * WHAT: Get counts and status of loaded wiring
 * WHY:  System introspection and debugging
 * HOW:  Count modules, relations, enabled/disabled
 */
int wiring_diagram_get_stats(
    const wiring_diagram_t* wd,
    uint32_t* total_modules,
    uint32_t* enabled_modules,
    uint32_t* total_relations
) {
    if (!wd) return -1;

    nimcp_platform_mutex_lock((nimcp_platform_mutex_t*)wd->mutex);

    uint32_t enabled = 0;
    uint32_t relations = 0;

    for (uint32_t i = 0; i < wd->module_count; i++) {
        if (!wd->module_configs[i]) continue;

        if (wd->module_configs[i]->enabled) {
            enabled++;
        }

        relations += wd->module_configs[i]->depends_on_count;
        relations += wd->module_configs[i]->sends_to_count;
        relations += wd->module_configs[i]->receives_from_count;
    }

    if (total_modules) *total_modules = wd->module_count;
    if (enabled_modules) *enabled_modules = enabled;
    if (total_relations) *total_relations = relations;

    nimcp_platform_mutex_unlock((nimcp_platform_mutex_t*)wd->mutex);
    return 0;
}

/* ============================================================================
 * Internal Helper Implementation
 * ============================================================================ */

/**
 * @brief Ensure module array has capacity for one more entry
 */
static int ensure_capacity(wiring_diagram_t* wd) {
    if (wd->module_count < wd->module_capacity) return 0;

    uint32_t new_capacity = wd->module_capacity * WIRING_ARRAY_GROW_FACTOR;
    if (new_capacity > WIRING_MAX_MODULES) {
        new_capacity = WIRING_MAX_MODULES;
    }
    if (new_capacity <= wd->module_capacity) return -1;

    wiring_module_config_t** new_array = (wiring_module_config_t**)nimcp_realloc(
        wd->module_configs,
        new_capacity * sizeof(wiring_module_config_t*));
    if (!new_array) return -1;

    wd->module_configs = new_array;
    wd->module_capacity = new_capacity;
    return 0;
}

/**
 * @brief Find module by name (caller must hold mutex)
 */
static wiring_module_config_t* find_module_by_name_unlocked(
    const wiring_diagram_t* wd,
    const char* name
) {
    for (uint32_t i = 0; i < wd->module_count; i++) {
        if (wd->module_configs[i] &&
            strcmp(wd->module_configs[i]->module_name, name) == 0) {
            return wd->module_configs[i];
        }
    }
    return NULL;
}

/**
 * @brief Find module by ID (caller must hold mutex)
 */
static wiring_module_config_t* find_module_by_id_unlocked(
    const wiring_diagram_t* wd,
    bio_module_id_t id
) {
    for (uint32_t i = 0; i < wd->module_count; i++) {
        if (wd->module_configs[i] &&
            wd->module_configs[i]->module_id == id) {
            return wd->module_configs[i];
        }
    }
    return NULL;
}

/**
 * @brief Parse a JSONL file and merge into wiring diagram
 *
 * WHAT: Read JSONL file line by line and parse entities/relations
 * WHY:  Load wiring configuration from disk
 * HOW:  Parse each line as JSON, dispatch to entity/relation handler
 */
static int parse_jsonl_file(wiring_diagram_t* wd, const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        /* Missing file is not necessarily an error */
        return -1;
    }

    char line[WIRING_LINE_BUFFER_SIZE];
    int line_count = 0;
    int error_count = 0;

    while (fgets(line, sizeof(line), file)) {
        line_count++;

        /* Skip empty lines and comments */
        size_t len = strlen(line);
        if (len == 0) continue;

        /* Remove trailing newline */
        if (line[len - 1] == '\n') {
            line[len - 1] = '\0';
            len--;
        }
        if (len == 0 || line[0] == '#') continue;

        /* Parse JSON */
        cJSON* json = cJSON_Parse(line);
        if (!json) {
            NIMCP_LOGGING_WARN("Failed to parse line %d in %s", line_count, filepath);
            error_count++;
            continue;
        }

        /* Determine type and dispatch */
        cJSON* type_item = cJSON_GetObjectItem(json, "type");
        if (type_item && cJSON_IsString(type_item)) {
            if (strcmp(type_item->valuestring, "entity") == 0) {
                if (parse_entity_line(wd, line) != 0) {
                    error_count++;
                }
            } else if (strcmp(type_item->valuestring, "relation") == 0) {
                if (parse_relation_line(wd, line) != 0) {
                    error_count++;
                }
            }
        }

        cJSON_Delete(json);
    }

    fclose(file);

    if (error_count > 0) {
        NIMCP_LOGGING_WARN("Parsed %s with %d errors", filepath, error_count);
    }

    return 0;
}

/**
 * @brief Populate config from parsed JSON
 *
 * WHAT: Extract fields from JSON into module config
 * WHY:  Helper to keep parse_entity_line under 50 lines
 * HOW:  Read each optional field from JSON
 */
static void populate_config_from_json(
    wiring_module_config_t* config,
    cJSON* json,
    const char* name
) {
    cJSON* subsystem = cJSON_GetObjectItem(json, "subsystem");
    if (subsystem && cJSON_IsString(subsystem)) {
        config->subsystem = parse_subsystem_string(subsystem->valuestring);
    }

    cJSON* module_id = cJSON_GetObjectItem(json, "module_id");
    if (module_id && cJSON_IsNumber(module_id)) {
        config->module_id = (bio_module_id_t)module_id->valueint;
    } else {
        config->module_id = parse_module_id(name);
    }

    cJSON* min_tier = cJSON_GetObjectItem(json, "min_tier");
    if (min_tier && cJSON_IsNumber(min_tier)) {
        config->min_tier = (platform_tier_t)min_tier->valueint;
    }

    cJSON* required_hw = cJSON_GetObjectItem(json, "required_hw");
    if (required_hw && cJSON_IsNumber(required_hw)) {
        config->required_hw = (wiring_hardware_flags_t)required_hw->valueint;
    }

    cJSON* enabled = cJSON_GetObjectItem(json, "enabled");
    if (enabled && cJSON_IsBool(enabled)) {
        config->enabled = cJSON_IsTrue(enabled);
    }
}

/**
 * @brief Parse an entity line from JSONL
 *
 * WHAT: Parse JSON entity and create/update module config
 * WHY:  Load module definitions from wiring files
 * HOW:  Extract fields from JSON, populate config struct
 */
static int parse_entity_line(wiring_diagram_t* wd, const char* line) {
    cJSON* json = cJSON_Parse(line);
    if (!json) return -1;

    /* Get required fields */
    cJSON* name = cJSON_GetObjectItem(json, "name");
    if (!name || !cJSON_IsString(name)) {
        cJSON_Delete(json);
        return -1;
    }

    /* Find or create config */
    wiring_module_config_t* config = find_module_by_name_unlocked(wd, name->valuestring);
    if (!config) {
        if (ensure_capacity(wd) != 0) {
            cJSON_Delete(json);
            return -1;
        }

        config = (wiring_module_config_t*)nimcp_calloc(1, sizeof(wiring_module_config_t));
        if (!config) {
            cJSON_Delete(json);
            return -1;
        }

        wiring_module_config_init(config);
        strncpy(config->module_name, name->valuestring,
            sizeof(config->module_name) - 1);

        wd->module_configs[wd->module_count++] = config;
    }

    /* Populate fields from JSON */
    populate_config_from_json(config, json, name->valuestring);

    cJSON_Delete(json);
    return 0;
}

/**
 * @brief Parse a relation line from JSONL
 *
 * WHAT: Parse JSON relation and update module connectivity
 * WHY:  Load module relationships from wiring files
 * HOW:  Extract from/to/type, update appropriate arrays
 */
static int parse_relation_line(wiring_diagram_t* wd, const char* line) {
    cJSON* json = cJSON_Parse(line);
    if (!json) return -1;

    /* Get required fields */
    cJSON* from = cJSON_GetObjectItem(json, "from");
    cJSON* to = cJSON_GetObjectItem(json, "to");
    cJSON* rel_type = cJSON_GetObjectItem(json, "relationType");

    if (!from || !cJSON_IsString(from) ||
        !to || !cJSON_IsString(to) ||
        !rel_type || !cJSON_IsString(rel_type)) {
        cJSON_Delete(json);
        return -1;
    }

    /* Find source module - always required */
    wiring_module_config_t* from_config = find_module_by_name_unlocked(
        wd, from->valuestring);
    if (!from_config) {
        cJSON_Delete(json);
        return -1;
    }

    /* Determine relation type and update arrays */
    int result = 0;

    if (strcmp(rel_type->valuestring, "HANDLES_MESSAGE") == 0) {
        /* For HANDLES_MESSAGE, 'to' is a message type name, not a module */
        cJSON* msg_type = cJSON_GetObjectItem(json, "message_type");
        if (msg_type && cJSON_IsNumber(msg_type)) {
            result = add_to_msgtype_array(&from_config->handles_messages,
                &from_config->handles_message_count,
                &from_config->handles_message_capacity,
                (bio_message_type_t)msg_type->valueint);
        }
    } else {
        /* All other relations require target module */
        wiring_module_config_t* to_config = find_module_by_name_unlocked(
            wd, to->valuestring);
        if (!to_config) {
            cJSON_Delete(json);
            return -1;
        }

        if (strcmp(rel_type->valuestring, "DEPENDS_ON") == 0) {
            result = add_to_id_array(&from_config->depends_on,
                &from_config->depends_on_count,
                &from_config->depends_on_capacity,
                to_config->module_id);
        } else if (strcmp(rel_type->valuestring, "SENDS_TO") == 0) {
            result = add_to_id_array(&from_config->sends_to,
                &from_config->sends_to_count,
                &from_config->sends_to_capacity,
                to_config->module_id);
        } else if (strcmp(rel_type->valuestring, "RECEIVES_FROM") == 0) {
            result = add_to_id_array(&from_config->receives_from,
                &from_config->receives_from_count,
                &from_config->receives_from_capacity,
                to_config->module_id);
        }
    }

    cJSON_Delete(json);
    return result;
}

/**
 * @brief Add module ID to dynamic array
 */
static int add_to_id_array(
    bio_module_id_t** array,
    uint32_t* count,
    uint32_t* capacity,
    bio_module_id_t id
) {
    /* Check for duplicates */
    for (uint32_t i = 0; i < *count; i++) {
        if ((*array)[i] == id) return 0;
    }

    /* Grow if needed */
    if (*count >= *capacity) {
        uint32_t new_cap = *capacity == 0 ? 8 : *capacity * 2;
        if (new_cap > WIRING_MAX_DEPENDENCIES) new_cap = WIRING_MAX_DEPENDENCIES;

        bio_module_id_t* new_array = (bio_module_id_t*)nimcp_realloc(
            *array, new_cap * sizeof(bio_module_id_t));
        if (!new_array) return -1;

        *array = new_array;
        *capacity = new_cap;
    }

    (*array)[(*count)++] = id;
    return 0;
}

/**
 * @brief Add message type to dynamic array
 */
static int add_to_msgtype_array(
    bio_message_type_t** array,
    uint32_t* count,
    uint32_t* capacity,
    bio_message_type_t msg_type
) {
    /* Check for duplicates */
    for (uint32_t i = 0; i < *count; i++) {
        if ((*array)[i] == msg_type) return 0;
    }

    /* Grow if needed */
    if (*count >= *capacity) {
        uint32_t new_cap = *capacity == 0 ? 8 : *capacity * 2;
        if (new_cap > WIRING_MAX_HANDLERS) new_cap = WIRING_MAX_HANDLERS;

        bio_message_type_t* new_array = (bio_message_type_t*)nimcp_realloc(
            *array, new_cap * sizeof(bio_message_type_t));
        if (!new_array) return -1;

        *array = new_array;
        *capacity = new_cap;
    }

    (*array)[(*count)++] = msg_type;
    return 0;
}

/**
 * @brief Parse subsystem string to enum
 */
static wiring_subsystem_t parse_subsystem_string(const char* str) {
    if (!str) return WIRING_SUBSYSTEM_CORE;

    for (int i = 0; i < WIRING_SUBSYSTEM_COUNT; i++) {
        if (strcasecmp(str, SUBSYSTEM_NAMES[i]) == 0) {
            return (wiring_subsystem_t)i;
        }
    }

    return WIRING_SUBSYSTEM_CORE;
}

/**
 * @brief Generate module ID from name (simple hash)
 */
static bio_module_id_t parse_module_id(const char* name) {
    if (!name) return 0;

    /* Simple hash for generating IDs from names */
    uint32_t hash = 0x811c9dc5;  /* FNV-1a offset basis */
    while (*name) {
        hash ^= (uint8_t)*name++;
        hash *= 0x01000193;  /* FNV-1a prime */
    }

    return (bio_module_id_t)(hash & 0xFFFF);
}

/**
 * @brief Write entity JSON to file
 *
 * WHAT: Serialize module config as JSON entity line
 * WHY:  Helper to keep persist function under 50 lines
 * HOW:  Create cJSON object, print, cleanup
 */
static void write_entity_json(
    FILE* file,
    const wiring_module_config_t* config,
    wiring_subsystem_t subsystem
) {
    cJSON* entity = cJSON_CreateObject();
    cJSON_AddStringToObject(entity, "type", "entity");
    cJSON_AddStringToObject(entity, "name", config->module_name);
    cJSON_AddStringToObject(entity, "subsystem", SUBSYSTEM_NAMES[subsystem]);
    cJSON_AddNumberToObject(entity, "module_id", config->module_id);
    cJSON_AddNumberToObject(entity, "min_tier", config->min_tier);
    cJSON_AddNumberToObject(entity, "required_hw", config->required_hw);
    cJSON_AddBoolToObject(entity, "enabled", config->enabled);

    char* json_str = cJSON_PrintUnformatted(entity);
    if (json_str) {
        fprintf(file, "%s\n", json_str);
        cJSON_free(json_str);
    }
    cJSON_Delete(entity);
}

/**
 * @brief Write relation JSON to file
 *
 * WHAT: Serialize relation as JSON line
 * WHY:  Helper to keep persist function under 50 lines
 * HOW:  Create cJSON object, print, cleanup
 */
static void write_relation_json(
    FILE* file,
    const char* from,
    const char* to,
    const char* rel_type
) {
    cJSON* rel = cJSON_CreateObject();
    cJSON_AddStringToObject(rel, "type", "relation");
    cJSON_AddStringToObject(rel, "from", from);
    cJSON_AddStringToObject(rel, "to", to);
    cJSON_AddStringToObject(rel, "relationType", rel_type);

    char* rel_str = cJSON_PrintUnformatted(rel);
    if (rel_str) {
        fprintf(file, "%s\n", rel_str);
        cJSON_free(rel_str);
    }
    cJSON_Delete(rel);
}

/**
 * @brief Persist subsystem to JSONL file (caller must hold mutex)
 */
static int persist_subsystem_unlocked(wiring_diagram_t* wd, wiring_subsystem_t subsystem) {
    char filepath[WIRING_MAX_PATH_LENGTH * 2];
    snprintf(filepath, sizeof(filepath), "%s/subsystems/%s.jsonl",
        wd->base_path, SUBSYSTEM_NAMES[subsystem]);

    /* Create directory if needed */
    char dirpath[WIRING_MAX_PATH_LENGTH * 2];
    snprintf(dirpath, sizeof(dirpath), "%s/subsystems", wd->base_path);
    mkdir(dirpath, 0755);

    FILE* file = fopen(filepath, "w");
    if (!file) {
        NIMCP_LOGGING_WARN("Failed to open %s for writing: %s", filepath, strerror(errno));
        return -1;
    }

    /* Write each module in this subsystem */
    for (uint32_t i = 0; i < wd->module_count; i++) {
        wiring_module_config_t* config = wd->module_configs[i];
        if (!config || config->subsystem != subsystem) continue;

        /* Write entity */
        write_entity_json(file, config, subsystem);

        /* Write depends_on relations */
        for (uint32_t j = 0; j < config->depends_on_count; j++) {
            wiring_module_config_t* target = find_module_by_id_unlocked(
                wd, config->depends_on[j]);
            if (target) {
                write_relation_json(file, config->module_name, target->module_name, "DEPENDS_ON");
            }
        }

        /* Write sends_to relations */
        for (uint32_t j = 0; j < config->sends_to_count; j++) {
            wiring_module_config_t* target = find_module_by_id_unlocked(
                wd, config->sends_to[j]);
            if (target) {
                write_relation_json(file, config->module_name, target->module_name, "SENDS_TO");
            }
        }
    }

    fclose(file);
    return 0;
}

/**
 * @brief Auto-persist if enabled
 */
static void auto_persist_if_enabled(wiring_diagram_t* wd) {
    if (wd->auto_persist && wd->dirty) {
        wiring_diagram_persist(wd);
    }
}

/* ============================================================================
 * Brain KG Integration Implementation (Phase 8)
 * ============================================================================ */

/**
 * @brief Sync wiring diagram handlers to brain_kg message index
 *
 * WHAT: Register all wiring diagram message handlers with brain_kg
 * WHY:  Enable KG-driven message dispatch (BIO_MODULE_KG_DISPATCH)
 * HOW:  Iterate modules, call brain_kg_add_message_handler for each handler
 *
 * @param wd Wiring diagram source
 * @param kg Brain KG to sync to
 * @return Number of handlers synced, or -1 on error
 */
int wiring_diagram_sync_to_brain_kg(wiring_diagram_t* wd, brain_kg_t* kg) {
    if (!wd || !kg) return -1;

    int handlers_synced = 0;

    /* Lock wiring diagram for thread-safe iteration */
    if (wd->mutex) {
        nimcp_platform_mutex_lock(wd->mutex);
    }

    /* Iterate all modules in the wiring diagram */
    for (uint32_t i = 0; i < wd->module_count; i++) {
        wiring_module_config_t* module = wd->module_configs[i];
        if (!module || !module->enabled) {
            continue;
        }

        /* Register each message handler for this module */
        for (uint32_t j = 0; j < module->handles_message_count; j++) {
            bio_message_type_t msg_type = module->handles_messages[j];

            /* Use module_id as the handler ID (bio_module_id_t is compatible
             * with brain_kg_node_id_t for message dispatch purposes) */
            int result = brain_kg_add_message_handler(kg,
                (brain_kg_node_id_t)module->module_id, msg_type);

            if (result == 0) {
                handlers_synced++;
                NIMCP_LOGGING_DEBUG("wiring_diagram_sync_to_brain_kg: "
                    "registered module %s (0x%04X) for message 0x%04X",
                    module->module_name, module->module_id, msg_type);
            }
        }
    }

    if (wd->mutex) {
        nimcp_platform_mutex_unlock(wd->mutex);
    }

    NIMCP_LOGGING_INFO("wiring_diagram_sync_to_brain_kg: synced %d handlers from %u modules",
        handlers_synced, wd->module_count);

    return handlers_synced;
}

/**
 * @brief Sync message handlers from brain_kg back to wiring diagram
 *
 * WHAT: Update wiring diagram with handlers registered in brain_kg
 * WHY:  Enable bidirectional sync for runtime-modified handlers
 * HOW:  Query brain_kg for each module's handlers, update wiring config
 *
 * @param wd Wiring diagram to update
 * @param kg Brain KG source (const - read-only)
 * @return Number of handlers synced, or -1 on error
 */
int wiring_diagram_sync_from_brain_kg(wiring_diagram_t* wd, const brain_kg_t* kg) {
    if (!wd || !kg) return -1;

    int handlers_synced = 0;

    /* Lock wiring diagram for thread-safe update */
    if (wd->mutex) {
        nimcp_platform_mutex_lock(wd->mutex);
    }

    /* Iterate all modules in the wiring diagram */
    for (uint32_t i = 0; i < wd->module_count; i++) {
        wiring_module_config_t* module = wd->module_configs[i];
        if (!module) {
            continue;
        }

        /* Query brain_kg for messages this module handles
         * Note: We cast away const because the API doesn't take const,
         * but brain_kg_get_module_handled_messages is read-only */
        uint32_t msg_types[WIRING_MAX_HANDLERS];
        uint32_t count = brain_kg_get_module_handled_messages(
            (brain_kg_t*)kg,
            (brain_kg_node_id_t)module->module_id,
            msg_types,
            WIRING_MAX_HANDLERS);

        if (count == 0) {
            continue;
        }

        /* Ensure capacity for new handlers */
        if (count > module->handles_message_capacity) {
            bio_message_type_t* new_handlers = (bio_message_type_t*)
                nimcp_realloc(module->handles_messages,
                    count * sizeof(bio_message_type_t));
            if (!new_handlers) {
                NIMCP_LOGGING_ERROR("wiring_diagram_sync_from_brain_kg: "
                    "failed to resize handlers for %s", module->module_name);
                continue;
            }
            module->handles_messages = new_handlers;
            module->handles_message_capacity = count;
        }

        /* Copy handlers from brain_kg */
        for (uint32_t j = 0; j < count; j++) {
            module->handles_messages[j] = (bio_message_type_t)msg_types[j];
        }
        module->handles_message_count = count;
        handlers_synced += count;

        NIMCP_LOGGING_DEBUG("wiring_diagram_sync_from_brain_kg: "
            "synced %u handlers to module %s", count, module->module_name);
    }

    /* Mark dirty for persistence */
    if (handlers_synced > 0) {
        wd->dirty = true;
        auto_persist_if_enabled(wd);
    }

    if (wd->mutex) {
        nimcp_platform_mutex_unlock(wd->mutex);
    }

    NIMCP_LOGGING_INFO("wiring_diagram_sync_from_brain_kg: synced %d handlers to %u modules",
        handlers_synced, wd->module_count);

    return handlers_synced;
}
