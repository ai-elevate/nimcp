/**
 * @file nimcp_dynamic_config.h
 * @brief Dynamic configuration system with SIGHUP reload
 *
 * WHAT: Runtime-reconfigurable hyperparameters via config file
 * WHY:  Allow tuning without restarting (critical for production)
 * HOW:  JSON/INI config file + SIGHUP signal triggers reload
 *
 * FEATURES:
 * - Hot reload on SIGHUP (no restart required)
 * - Thread-safe access to config values
 * - Validation of config ranges
 * - Change notifications for subscribers
 * - Rollback on invalid config
 * - Config versioning and history
 *
 * USAGE:
 * 1. Initialize: config_init("/path/to/config.json")
 * 2. Read values: config_get_float("learning_rate")
 * 3. Reload: Send SIGHUP signal to process
 * 4. Config automatically reloads and validates
 *
 * @author NIMCP Team
 * @date 2025-11-09
 */

#ifndef NIMCP_DYNAMIC_CONFIG_H
#define NIMCP_DYNAMIC_CONFIG_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Configuration Types
//=============================================================================

/**
 * @brief Config value types
 */
typedef enum {
    CONFIG_TYPE_INT,
    CONFIG_TYPE_FLOAT,
    CONFIG_TYPE_BOOL,
    CONFIG_TYPE_STRING
} config_value_type_t;

/**
 * @brief Config value union
 */
typedef union {
    int64_t int_val;
    double float_val;
    bool bool_val;
    char* string_val;
} config_value_t;

/**
 * @brief Config entry with validation
 */
typedef struct {
    const char* key;              /**< Config key name */
    config_value_type_t type;     /**< Value type */
    config_value_t value;         /**< Current value */
    config_value_t default_value; /**< Default value */
    config_value_t min_value;     /**< Min value (for numeric types) */
    config_value_t max_value;     /**< Max value (for numeric types) */
    bool has_min;                 /**< Whether min is set */
    bool has_max;                 /**< Whether max is set */
    const char* description;      /**< Human-readable description */
} config_entry_t;

/**
 * @brief Config change notification callback
 */
typedef void (*config_change_callback_t)(const char* key, const config_value_t* old_value,
                                          const config_value_t* new_value, void* user_data);

/**
 * @brief Dynamic config statistics
 */
typedef struct {
    uint64_t reload_count;        /**< Number of reloads */
    uint64_t reload_failures;     /**< Number of failed reloads */
    uint64_t validation_failures; /**< Number of validation failures */
    uint64_t last_reload_time_ms; /**< Last successful reload timestamp */
    uint32_t config_version;      /**< Current config version */
} config_stats_t;

//=============================================================================
// Public API
//=============================================================================

/**
 * @brief Initialize config system
 *
 * WHAT: Load config from file and setup SIGHUP handler
 * WHY:  Enable dynamic configuration
 * HOW:  Parse config file, register signal handler
 *
 * @param config_path Path to config file
 * @return true on success, false on failure
 */
bool config_init(const char* config_path);

/**
 * @brief Shutdown config system
 *
 * WHAT: Clean up resources and uninstall signal handler
 * WHY:  Proper cleanup
 * HOW:  Free memory, restore default signal handler
 */
void config_shutdown(void);

/**
 * @brief Reload config from file
 *
 * WHAT: Re-parse config file and update values
 * WHY:  Manual reload or SIGHUP handler
 * HOW:  Parse file, validate, update atomically
 *
 * @return true on success, false on failure (old config retained)
 */
bool config_reload(void);

/**
 * @brief Get integer config value
 *
 * @param key Config key
 * @param default_value Default if key not found
 * @return Config value or default
 */
int64_t config_get_int(const char* key, int64_t default_value);

/**
 * @brief Get float config value
 *
 * @param key Config key
 * @param default_value Default if key not found
 * @return Config value or default
 */
double config_get_float(const char* key, double default_value);

/**
 * @brief Get boolean config value
 *
 * @param key Config key
 * @param default_value Default if key not found
 * @return Config value or default
 */
bool config_get_bool(const char* key, bool default_value);

/**
 * @brief Get string config value
 *
 * @param key Config key
 * @param default_value Default if key not found
 * @return Config value or default (caller does NOT own memory)
 */
const char* config_get_string(const char* key, const char* default_value);

/**
 * @brief Set integer config value (runtime override)
 *
 * @param key Config key
 * @param value New value
 * @return true on success, false on validation failure
 */
bool config_set_int(const char* key, int64_t value);

/**
 * @brief Set float config value (runtime override)
 *
 * @param key Config key
 * @param value New value
 * @return true on success, false on validation failure
 */
bool config_set_float(const char* key, double value);

/**
 * @brief Set boolean config value (runtime override)
 *
 * @param key Config key
 * @param value New value
 * @return true on success
 */
bool config_set_bool(const char* key, bool value);

/**
 * @brief Set string config value (runtime override)
 *
 * @param key Config key
 * @param value New value (copied internally)
 * @return true on success
 */
bool config_set_string(const char* key, const char* value);

/**
 * @brief Register change notification callback
 *
 * WHAT: Get notified when config value changes
 * WHY:  React to dynamic config changes
 * HOW:  Store callback, call on each reload
 *
 * @param key Config key to watch (NULL for all keys)
 * @param callback Callback function
 * @param user_data User data passed to callback
 * @return Registration ID (use to unregister)
 */
uint32_t config_register_callback(const char* key, config_change_callback_t callback,
                                   void* user_data);

/**
 * @brief Unregister change notification callback
 *
 * @param registration_id ID returned from register_callback
 */
void config_unregister_callback(uint32_t registration_id);

/**
 * @brief Get config statistics
 *
 * @return Config statistics
 */
config_stats_t config_get_stats(void);

/**
 * @brief Dump current config to file
 *
 * WHAT: Save current config values to file
 * WHY:  Backup or generate template
 * HOW:  Serialize config to JSON/INI format
 *
 * @param output_path Output file path
 * @return true on success
 */
bool config_dump(const char* output_path);

/**
 * @brief Print current config to stdout
 */
void config_print(void);

/**
 * @brief Validate config file without loading
 *
 * WHAT: Check if config file is valid
 * WHY:  Pre-validation before reload
 * HOW:  Parse and validate without updating
 *
 * @param config_path Path to config file
 * @return true if valid, false otherwise
 */
bool config_validate(const char* config_path);

//=============================================================================
// Helper Macros for Common Hyperparameters
//=============================================================================

// Learning rates
#define CONFIG_LEARNING_RATE            config_get_float("learning_rate", 0.001)
#define CONFIG_LEARNING_RATE_SENSORY    config_get_float("learning_rate_sensory", 0.0001)
#define CONFIG_LEARNING_RATE_ASSOCIATION config_get_float("learning_rate_association", 0.001)
#define CONFIG_LEARNING_RATE_PREFRONTAL config_get_float("learning_rate_prefrontal", 0.01)

// Network architecture
#define CONFIG_NUM_INPUTS               config_get_int("num_inputs", 256)
#define CONFIG_NUM_HIDDEN               config_get_int("num_hidden", 1024)
#define CONFIG_NUM_OUTPUTS              config_get_int("num_outputs", 10)

// Training parameters
#define CONFIG_BATCH_SIZE               config_get_int("batch_size", 32)
#define CONFIG_NUM_EPOCHS               config_get_int("num_epochs", 10)
#define CONFIG_DROPOUT_RATE             config_get_float("dropout_rate", 0.5)

// Plasticity parameters
#define CONFIG_STDP_WINDOW_MS           config_get_int("stdp_window_ms", 20)
#define CONFIG_STDP_A_PLUS              config_get_float("stdp_a_plus", 0.01)
#define CONFIG_STDP_A_MINUS             config_get_float("stdp_a_minus", 0.012)
#define CONFIG_STDP_TAU_PLUS_MS         config_get_float("stdp_tau_plus_ms", 20.0)
#define CONFIG_STDP_TAU_MINUS_MS        config_get_float("stdp_tau_minus_ms", 20.0)

// Neuromodulators
#define CONFIG_DOPAMINE_BASELINE        config_get_float("dopamine_baseline", 0.2)
#define CONFIG_SEROTONIN_BASELINE       config_get_float("serotonin_baseline", 0.5)
#define CONFIG_ACETYLCHOLINE_BASELINE   config_get_float("acetylcholine_baseline", 0.3)

// Phase 10 parameters
#define CONFIG_WORKING_MEMORY_CAPACITY  config_get_int("working_memory_capacity", 7)
#define CONFIG_WORKING_MEMORY_DECAY     config_get_float("working_memory_decay", 0.1)
#define CONFIG_PREDICTION_ERROR_THRESHOLD config_get_float("prediction_error_threshold", 0.5)
#define CONFIG_META_LEARNING_K_SHOT     config_get_int("meta_learning_k_shot", 5)

// Feature flags
#define CONFIG_ENABLE_COW               config_get_bool("enable_cow", true)
#define CONFIG_ENABLE_CACHE             config_get_bool("enable_cache", true)
#define CONFIG_ENABLE_WORKING_MEMORY    config_get_bool("enable_working_memory", true)
#define CONFIG_ENABLE_PREDICTIVE        config_get_bool("enable_predictive_processing", true)

#ifdef __cplusplus
}
#endif

#endif // NIMCP_DYNAMIC_CONFIG_H
