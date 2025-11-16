/**
 * @file nimcp_dynamic_config.c
 * @brief Dynamic configuration with INI parser
 *
 * WHAT: Runtime-reconfigurable hyperparameters via config file
 * WHY:  Allow tuning without restarting (critical for production)
 * HOW:  INI config file + SIGHUP signal triggers reload
 *
 * @author NIMCP Team
 * @date 2025-11-09
 */

#include "nimcp_dynamic_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>

//=============================================================================
// Internal Data Structures
//=============================================================================

#define MAX_CONFIG_ENTRIES 256
#define MAX_KEY_LENGTH 128
#define MAX_STRING_VALUE 512
#define MAX_CALLBACKS 64

typedef struct {
    char key[MAX_KEY_LENGTH];
    config_value_type_t type;
    config_value_t value;
    bool in_use;
} config_entry_internal_t;

/**
 * @brief Callback registration entry
 *
 * WHAT: Stores callback function and metadata for config change notifications
 * WHY:  Enable runtime reaction to config changes (e.g., adjust learning rates)
 * HOW:  Array of registrations with thread-safe access
 */
typedef struct {
    uint32_t id;                        /**< Unique registration ID */
    char key[MAX_KEY_LENGTH];           /**< Config key to watch (empty = all) */
    config_change_callback_t callback;  /**< Callback function */
    void* user_data;                    /**< User data passed to callback */
    bool in_use;                        /**< Is this slot active? */
} callback_registration_t;

static config_entry_internal_t g_config_table[MAX_CONFIG_ENTRIES];
static callback_registration_t g_callbacks[MAX_CALLBACKS];
static pthread_rwlock_t g_config_lock = PTHREAD_RWLOCK_INITIALIZER;
static pthread_mutex_t g_callback_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_config_path[512] = {0};
static config_stats_t g_stats = {0};
static uint32_t g_next_callback_id = 1;

//=============================================================================
// INI Parser (Simple, No Dependencies)
//=============================================================================

static void trim_whitespace(char* str) {
    char* start = str;
    char* end;

    // Trim leading space
    while (isspace((unsigned char)*start)) start++;

    if (*start == 0) {
        *str = 0;
        return;
    }

    // Trim trailing space
    end = start + strlen(start) - 1;
    while (end > start && isspace((unsigned char)*end)) end--;
    end[1] = '\0';

    // Move trimmed string to beginning if needed
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

static bool parse_config_file(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open config file: %s\n", path);
        return false;
    }

    char line[1024];
    int line_num = 0;

    while (fgets(line, sizeof(line), file)) {
        line_num++;
        trim_whitespace(line);

        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == ';' || line[0] == '\0') {
            continue;
        }

        // Parse key=value
        char* equals = strchr(line, '=');
        if (!equals) {
            fprintf(stderr, "Config parse error line %d: missing '='\n", line_num);
            continue;
        }

        *equals = '\0';
        char* key = line;
        char* value = equals + 1;

        trim_whitespace(key);
        trim_whitespace(value);

        if (key[0] == '\0' || value[0] == '\0') {
            fprintf(stderr, "Config parse error line %d: empty key or value\n", line_num);
            continue;
        }

        // Store in config table
        pthread_rwlock_wrlock(&g_config_lock);

        for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
            if (!g_config_table[i].in_use ||
                strcmp(g_config_table[i].key, key) == 0) {

                // Free old string value if replacing
                if (g_config_table[i].in_use &&
                    g_config_table[i].type == CONFIG_TYPE_STRING &&
                    g_config_table[i].value.string_val) {
                    free(g_config_table[i].value.string_val);
                    g_config_table[i].value.string_val = NULL;
                }

                strncpy(g_config_table[i].key, key, MAX_KEY_LENGTH - 1);
                g_config_table[i].key[MAX_KEY_LENGTH - 1] = '\0';
                g_config_table[i].in_use = true;

                // Detect type and parse value
                if (strcmp(value, "true") == 0 || strcmp(value, "false") == 0) {
                    g_config_table[i].type = CONFIG_TYPE_BOOL;
                    g_config_table[i].value.bool_val = (strcmp(value, "true") == 0);
                } else if (strchr(value, '.') != NULL) {
                    g_config_table[i].type = CONFIG_TYPE_FLOAT;
                    g_config_table[i].value.float_val = atof(value);
                } else if (value[0] == '-' || isdigit((unsigned char)value[0])) {
                    g_config_table[i].type = CONFIG_TYPE_INT;
                    g_config_table[i].value.int_val = atoll(value);
                } else {
                    g_config_table[i].type = CONFIG_TYPE_STRING;
                    g_config_table[i].value.string_val = strdup(value);
                }
                break;
            }
        }

        pthread_rwlock_unlock(&g_config_lock);
    }

    fclose(file);
    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

bool config_init(const char* config_path) {
    if (!config_path) {
        fprintf(stderr, "config_init: NULL config path\n");
        return false;
    }

    strncpy(g_config_path, config_path, sizeof(g_config_path) - 1);
    g_config_path[sizeof(g_config_path) - 1] = '\0';

    // Parse initial config
    if (!parse_config_file(g_config_path)) {
        fprintf(stderr, "config_init: Failed to parse config file\n");
        return false;
    }

    g_stats.config_version = 1;
    g_stats.last_reload_time_ms = 0;

    printf("Config initialized from: %s\n", g_config_path);
    return true;
}

void config_shutdown(void) {
    pthread_rwlock_wrlock(&g_config_lock);

    // Free string values
    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use &&
            g_config_table[i].type == CONFIG_TYPE_STRING &&
            g_config_table[i].value.string_val) {
            free(g_config_table[i].value.string_val);
        }
        g_config_table[i].in_use = false;
    }

    pthread_rwlock_unlock(&g_config_lock);
}

bool config_reload(void) {
    printf("Reloading config from: %s\n", g_config_path);

    // Parse config file (replaces values in place)
    bool success = parse_config_file(g_config_path);

    if (success) {
        g_stats.reload_count++;
        g_stats.config_version++;
        printf("Config reloaded successfully (version %u)\n", g_stats.config_version);
    } else {
        g_stats.reload_failures++;
        fprintf(stderr, "Config reload failed\n");
    }

    return success;
}

int64_t config_get_int(const char* key, int64_t default_value) {
    if (!key) return default_value;

    pthread_rwlock_rdlock(&g_config_lock);

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use &&
            strcmp(g_config_table[i].key, key) == 0 &&
            g_config_table[i].type == CONFIG_TYPE_INT) {
            int64_t value = g_config_table[i].value.int_val;
            pthread_rwlock_unlock(&g_config_lock);
            return value;
        }
    }

    pthread_rwlock_unlock(&g_config_lock);
    return default_value;
}

double config_get_float(const char* key, double default_value) {
    if (!key) return default_value;

    pthread_rwlock_rdlock(&g_config_lock);

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use &&
            strcmp(g_config_table[i].key, key) == 0 &&
            g_config_table[i].type == CONFIG_TYPE_FLOAT) {
            double value = g_config_table[i].value.float_val;
            pthread_rwlock_unlock(&g_config_lock);
            return value;
        }
    }

    pthread_rwlock_unlock(&g_config_lock);
    return default_value;
}

bool config_get_bool(const char* key, bool default_value) {
    if (!key) return default_value;

    pthread_rwlock_rdlock(&g_config_lock);

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use &&
            strcmp(g_config_table[i].key, key) == 0 &&
            g_config_table[i].type == CONFIG_TYPE_BOOL) {
            bool value = g_config_table[i].value.bool_val;
            pthread_rwlock_unlock(&g_config_lock);
            return value;
        }
    }

    pthread_rwlock_unlock(&g_config_lock);
    return default_value;
}

const char* config_get_string(const char* key, const char* default_value) {
    if (!key) return default_value;

    pthread_rwlock_rdlock(&g_config_lock);

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use &&
            strcmp(g_config_table[i].key, key) == 0 &&
            g_config_table[i].type == CONFIG_TYPE_STRING) {
            const char* value = g_config_table[i].value.string_val;
            pthread_rwlock_unlock(&g_config_lock);
            return value;
        }
    }

    pthread_rwlock_unlock(&g_config_lock);
    return default_value;
}

bool config_set_int(const char* key, int64_t value) {
    if (!key) return false;

    pthread_rwlock_wrlock(&g_config_lock);

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use && strcmp(g_config_table[i].key, key) == 0) {
            if (g_config_table[i].type != CONFIG_TYPE_INT) {
                pthread_rwlock_unlock(&g_config_lock);
                return false; // Type mismatch
            }
            g_config_table[i].value.int_val = value;
            pthread_rwlock_unlock(&g_config_lock);
            return true;
        }
    }

    pthread_rwlock_unlock(&g_config_lock);
    return false; // Key not found
}

bool config_set_float(const char* key, double value) {
    if (!key) return false;

    pthread_rwlock_wrlock(&g_config_lock);

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use && strcmp(g_config_table[i].key, key) == 0) {
            if (g_config_table[i].type != CONFIG_TYPE_FLOAT) {
                pthread_rwlock_unlock(&g_config_lock);
                return false; // Type mismatch
            }
            g_config_table[i].value.float_val = value;
            pthread_rwlock_unlock(&g_config_lock);
            return true;
        }
    }

    pthread_rwlock_unlock(&g_config_lock);
    return false; // Key not found
}

bool config_set_bool(const char* key, bool value) {
    if (!key) return false;

    pthread_rwlock_wrlock(&g_config_lock);

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use && strcmp(g_config_table[i].key, key) == 0) {
            if (g_config_table[i].type != CONFIG_TYPE_BOOL) {
                pthread_rwlock_unlock(&g_config_lock);
                return false; // Type mismatch
            }
            g_config_table[i].value.bool_val = value;
            pthread_rwlock_unlock(&g_config_lock);
            return true;
        }
    }

    pthread_rwlock_unlock(&g_config_lock);
    return false; // Key not found
}

bool config_set_string(const char* key, const char* value) {
    if (!key || !value) return false;

    pthread_rwlock_wrlock(&g_config_lock);

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use && strcmp(g_config_table[i].key, key) == 0) {
            if (g_config_table[i].type != CONFIG_TYPE_STRING) {
                pthread_rwlock_unlock(&g_config_lock);
                return false; // Type mismatch
            }

            // Free old value and set new one
            if (g_config_table[i].value.string_val) {
                free(g_config_table[i].value.string_val);
            }
            g_config_table[i].value.string_val = strdup(value);
            pthread_rwlock_unlock(&g_config_lock);
            return true;
        }
    }

    pthread_rwlock_unlock(&g_config_lock);
    return false; // Key not found
}

config_stats_t config_get_stats(void) {
    return g_stats;
}

void config_print(void) {
    pthread_rwlock_rdlock(&g_config_lock);

    printf("\n=== NIMCP Configuration ===\n");
    printf("Version: %u\n", g_stats.config_version);
    printf("Reload count: %lu\n", (unsigned long)g_stats.reload_count);
    printf("\nCurrent Values:\n");

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use) {
            printf("  %s = ", g_config_table[i].key);
            switch (g_config_table[i].type) {
                case CONFIG_TYPE_INT:
                    printf("%lld\n", (long long)g_config_table[i].value.int_val);
                    break;
                case CONFIG_TYPE_FLOAT:
                    printf("%f\n", g_config_table[i].value.float_val);
                    break;
                case CONFIG_TYPE_BOOL:
                    printf("%s\n", g_config_table[i].value.bool_val ? "true" : "false");
                    break;
                case CONFIG_TYPE_STRING:
                    printf("%s\n", g_config_table[i].value.string_val);
                    break;
            }
        }
    }

    printf("===========================\n\n");

    pthread_rwlock_unlock(&g_config_lock);
}

bool config_dump(const char* output_path) {
    if (!output_path) return false;

    FILE* file = fopen(output_path, "w");
    if (!file) {
        fprintf(stderr, "Failed to open output file: %s\n", output_path);
        return false;
    }

    pthread_rwlock_rdlock(&g_config_lock);

    fprintf(file, "# NIMCP Configuration Dump\n");
    fprintf(file, "# Version: %u\n", g_stats.config_version);
    fprintf(file, "# Generated automatically\n\n");

    for (int i = 0; i < MAX_CONFIG_ENTRIES; i++) {
        if (g_config_table[i].in_use) {
            fprintf(file, "%s = ", g_config_table[i].key);
            switch (g_config_table[i].type) {
                case CONFIG_TYPE_INT:
                    fprintf(file, "%lld\n", (long long)g_config_table[i].value.int_val);
                    break;
                case CONFIG_TYPE_FLOAT:
                    fprintf(file, "%f\n", g_config_table[i].value.float_val);
                    break;
                case CONFIG_TYPE_BOOL:
                    fprintf(file, "%s\n", g_config_table[i].value.bool_val ? "true" : "false");
                    break;
                case CONFIG_TYPE_STRING:
                    fprintf(file, "%s\n", g_config_table[i].value.string_val);
                    break;
            }
        }
    }

    pthread_rwlock_unlock(&g_config_lock);
    fclose(file);

    printf("Config dumped to: %s\n", output_path);
    return true;
}

bool config_validate(const char* config_path) {
    if (!config_path) return false;

    FILE* file = fopen(config_path, "r");
    if (!file) {
        fprintf(stderr, "Cannot open config file: %s\n", config_path);
        return false;
    }

    char line[1024];
    int line_num = 0;
    bool valid = true;

    while (fgets(line, sizeof(line), file)) {
        line_num++;
        trim_whitespace(line);

        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == ';' || line[0] == '\0') {
            continue;
        }

        // Validate key=value format
        char* equals = strchr(line, '=');
        if (!equals) {
            fprintf(stderr, "Validation error line %d: missing '='\n", line_num);
            valid = false;
            continue;
        }

        *equals = '\0';
        char* key = line;
        char* value = equals + 1;

        trim_whitespace(key);
        trim_whitespace(value);

        if (key[0] == '\0' || value[0] == '\0') {
            fprintf(stderr, "Validation error line %d: empty key or value\n", line_num);
            valid = false;
        }
    }

    fclose(file);
    return valid;
}

/**
 * @brief Trigger callbacks for a config value change
 *
 * WHAT: Notify all registered callbacks about a config change
 * WHY:  Enable runtime adaptation to config changes
 * HOW:  Iterate callbacks, invoke matching ones
 *
 * THREAD-SAFETY: Must be called with g_config_lock held (read or write)
 *
 * @param key Config key that changed
 * @param old_value Previous value
 * @param new_value New value
 */
static void trigger_callbacks(const char* key, const config_value_t* old_value,
                              const config_value_t* new_value) {
    if (!key) {
        return;
    }

    // Lock callback table
    pthread_mutex_lock(&g_callback_lock);

    // Iterate all registered callbacks
    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (!g_callbacks[i].in_use || !g_callbacks[i].callback) {
            continue;
        }

        // Check if callback matches this key
        // Empty key means callback wants all changes
        bool matches = (g_callbacks[i].key[0] == '\0') ||
                      (strcmp(g_callbacks[i].key, key) == 0);

        if (matches) {
            // Invoke callback (unlock first to prevent deadlock)
            pthread_mutex_unlock(&g_callback_lock);
            g_callbacks[i].callback(key, old_value, new_value, g_callbacks[i].user_data);
            pthread_mutex_lock(&g_callback_lock);
        }
    }

    pthread_mutex_unlock(&g_callback_lock);
}

uint32_t config_register_callback(const char* key, config_change_callback_t callback,
                                   void* user_data) {
    // WHAT: Register callback for config change notifications
    // WHY:  Enable runtime reaction to config updates (e.g., adjust learning rate)
    // HOW:  Store callback in thread-safe registry with unique ID

    // Input validation
    if (!callback) {
        fprintf(stderr, "config_register_callback: NULL callback\n");
        return 0;
    }

    pthread_mutex_lock(&g_callback_lock);

    // Find free slot
    uint32_t slot = 0;
    bool found = false;
    for (uint32_t i = 0; i < MAX_CALLBACKS; i++) {
        if (!g_callbacks[i].in_use) {
            slot = i;
            found = true;
            break;
        }
    }

    if (!found) {
        pthread_mutex_unlock(&g_callback_lock);
        fprintf(stderr, "config_register_callback: No free callback slots (max %d)\n",
                MAX_CALLBACKS);
        return 0;
    }

    // Generate unique ID
    uint32_t id = g_next_callback_id++;

    // Register callback
    g_callbacks[slot].id = id;
    g_callbacks[slot].callback = callback;
    g_callbacks[slot].user_data = user_data;
    g_callbacks[slot].in_use = true;

    // Copy key (empty string means watch all keys)
    if (key) {
        strncpy(g_callbacks[slot].key, key, MAX_KEY_LENGTH - 1);
        g_callbacks[slot].key[MAX_KEY_LENGTH - 1] = '\0';
    } else {
        g_callbacks[slot].key[0] = '\0';  // Watch all keys
    }

    pthread_mutex_unlock(&g_callback_lock);

    return id;
}

void config_unregister_callback(uint32_t registration_id) {
    // WHAT: Unregister callback by ID
    // WHY:  Cleanup when callback no longer needed
    // HOW:  Find callback by ID and mark slot as free

    // Validate ID
    if (registration_id == 0) {
        return;
    }

    pthread_mutex_lock(&g_callback_lock);

    // Find and remove callback
    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (g_callbacks[i].in_use && g_callbacks[i].id == registration_id) {
            // Mark slot as free
            g_callbacks[i].in_use = false;
            g_callbacks[i].callback = NULL;
            g_callbacks[i].user_data = NULL;
            g_callbacks[i].key[0] = '\0';
            break;
        }
    }

    pthread_mutex_unlock(&g_callback_lock);
}
