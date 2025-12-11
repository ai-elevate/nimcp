/**
 * @file nimcp_config.c
 * @brief NIMCP Configuration file parser implementation
 */

#include "utils/config/nimcp_config.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/json/nimcp_json.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#define MAX_LINE 1024

// Helper to parse size enum
static int parse_size(const char* str) {
    if (strcmp(str, "tiny") == 0) return 0;
    if (strcmp(str, "small") == 0) return 1;
    if (strcmp(str, "medium") == 0) return 2;
    if (strcmp(str, "large") == 0) return 3;
    return 1; // default to small
}

// Helper to parse task enum
static int parse_task(const char* str) {
    if (strcmp(str, "classification") == 0) return 0;
    if (strcmp(str, "regression") == 0) return 1;
    if (strcmp(str, "pattern_matching") == 0) return 2;
    if (strcmp(str, "sequence") == 0) return 3;
    if (strcmp(str, "association") == 0) return 4;
    return 0; // default to classification
}

// Helper to trim whitespace
static char* trim(char* str) {
    char* end;
    while(isspace((unsigned char)*str)) str++;
    if(*str == 0) return str;
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

// Helper to remove quotes
static char* unquote(char* str) {
    size_t len = strlen(str);
    if (len >= 2 && ((str[0] == '"' && str[len-1] == '"') ||
                      (str[0] == '\'' && str[len-1] == '\''))) {
        str[len-1] = '\0';
        return str + 1;
    }
    return str;
}

void nimcp_config_init_defaults(nimcp_brain_config_t* config) {
    memset(config, 0, sizeof(nimcp_brain_config_t));

    strcpy(config->name, "default_brain");
    config->size = 1;  // small
    config->task = 0;  // classification

    config->num_inputs = 10;
    config->num_outputs = 3;
    config->num_hidden = 100;
    config->learning_rate = 0.01F;

    config->max_epochs = 100;
    config->batch_size = 32;
    config->validation_split = 0.2F;
    config->early_stopping = true;
    config->patience = 10;

    config->enable_bcm = true;
    config->bcm_tau = 1000.0F;
    config->enable_stdp = false;
    config->stdp_window = 20.0F;

    config->ethics_enabled = false;
    config->golden_rule_threshold = 0.0F;
    config->empathy_weight = 0.5F;

    strcpy(config->model_path, "/tmp/brain.model");
    config->checkpoint_interval = 10;
}

bool nimcp_config_load_yaml(const char* filepath, nimcp_brain_config_t* config) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Failed to open config file: %s\n", filepath);
        return false;
    }

    nimcp_config_init_defaults(config);

    char line[MAX_LINE];
    char section[64] = "";

    while (fgets(line, sizeof(line), file)) {
        char* trimmed = trim(line);

        // Skip comments and empty lines
        if (trimmed[0] == '#' || trimmed[0] == '\0') continue;

        // Check for section headers
        if (strstr(trimmed, "brain:")) {
            strcpy(section, "brain");
            continue;
        } else if (strstr(trimmed, "architecture:")) {
            strcpy(section, "architecture");
            continue;
        } else if (strstr(trimmed, "training:")) {
            strcpy(section, "training");
            continue;
        } else if (strstr(trimmed, "plasticity:")) {
            strcpy(section, "plasticity");
            continue;
        } else if (strstr(trimmed, "ethics:")) {
            strcpy(section, "ethics");
            continue;
        }

        // Parse key-value pairs
        char* colon = strchr(trimmed, ':');
        if (!colon) continue;

        *colon = '\0';
        char* key = trim(trimmed);
        char* value = trim(colon + 1);
        value = unquote(value);

        // Parse based on section and key
        if (strcmp(section, "brain") == 0 || strcmp(section, "") == 0) {
            if (strcmp(key, "name") == 0) {
                strncpy(config->name, value, sizeof(config->name) - 1);
            } else if (strcmp(key, "size") == 0) {
                config->size = parse_size(value);
            } else if (strcmp(key, "task") == 0) {
                config->task = parse_task(value);
            } else if (strcmp(key, "model_path") == 0) {
                strncpy(config->model_path, value, sizeof(config->model_path) - 1);
            } else if (strcmp(key, "checkpoint_interval") == 0) {
                config->checkpoint_interval = (uint32_t)atoi(value);
            }
        } else if (strcmp(section, "architecture") == 0) {
            if (strcmp(key, "num_inputs") == 0) {
                config->num_inputs = (uint32_t)atoi(value);
            } else if (strcmp(key, "num_outputs") == 0) {
                config->num_outputs = (uint32_t)atoi(value);
            } else if (strcmp(key, "num_hidden") == 0) {
                config->num_hidden = (uint32_t)atoi(value);
            } else if (strcmp(key, "learning_rate") == 0) {
                config->learning_rate = (float)atof(value);
            }
        } else if (strcmp(section, "training") == 0) {
            if (strcmp(key, "max_epochs") == 0) {
                config->max_epochs = (uint32_t)atoi(value);
            } else if (strcmp(key, "batch_size") == 0) {
                config->batch_size = (uint32_t)atoi(value);
            } else if (strcmp(key, "validation_split") == 0) {
                config->validation_split = (float)atof(value);
            } else if (strcmp(key, "early_stopping") == 0) {
                config->early_stopping = (strcmp(value, "true") == 0);
            } else if (strcmp(key, "patience") == 0) {
                config->patience = (uint32_t)atoi(value);
            }
        } else if (strcmp(section, "plasticity") == 0) {
            if (strcmp(key, "enable_bcm") == 0) {
                config->enable_bcm = (strcmp(value, "true") == 0);
            } else if (strcmp(key, "bcm_tau") == 0) {
                config->bcm_tau = (float)atof(value);
            } else if (strcmp(key, "enable_stdp") == 0) {
                config->enable_stdp = (strcmp(value, "true") == 0);
            } else if (strcmp(key, "stdp_window") == 0) {
                config->stdp_window = (float)atof(value);
            }
        } else if (strcmp(section, "ethics") == 0) {
            if (strcmp(key, "enabled") == 0) {
                config->ethics_enabled = (strcmp(value, "true") == 0);
            } else if (strcmp(key, "golden_rule_threshold") == 0) {
                config->golden_rule_threshold = (float)atof(value);
            } else if (strcmp(key, "empathy_weight") == 0) {
                config->empathy_weight = (float)atof(value);
            }
        }
    }

    fclose(file);
    return true;
}

bool nimcp_config_load_json(const char* filepath, nimcp_brain_config_t* config) {
    // Use the nimcp_json module for proper JSON parsing
    JsonContext* ctx = NULL;

    if (nimcp_json_create_context(&ctx) != JSON_SUCCESS) {
        fprintf(stderr, "Failed to create JSON context\n");
        return false;
    }

    if (nimcp_json_load_file(ctx, filepath, 0) != JSON_SUCCESS) {
        fprintf(stderr, "Failed to load JSON config file: %s\n", filepath);
        nimcp_json_destroy_context(ctx);
        return false;
    }

    nimcp_config_init_defaults(config);

    // Parse brain configuration fields
    char str_buffer[256];
    int64_t int_value = 0;
    double double_value = 0.0;
    bool bool_value = false;

    // Brain section fields
    if (nimcp_json_get_string_value(ctx, "name", str_buffer, sizeof(str_buffer)) == JSON_SUCCESS) {
        strncpy(config->name, str_buffer, sizeof(config->name) - 1);
    }

    // Architecture section fields
    if (nimcp_json_get_integer_value(ctx, "num_inputs", &int_value) == JSON_SUCCESS) {
        config->num_inputs = (uint32_t)int_value;
    }
    if (nimcp_json_get_integer_value(ctx, "num_outputs", &int_value) == JSON_SUCCESS) {
        config->num_outputs = (uint32_t)int_value;
    }
    if (nimcp_json_get_integer_value(ctx, "num_hidden", &int_value) == JSON_SUCCESS) {
        config->num_hidden = (uint32_t)int_value;
    }
    if (nimcp_json_get_number_value(ctx, "learning_rate", &double_value) == JSON_SUCCESS) {
        config->learning_rate = (float)double_value;
    }

    // Training section fields
    if (nimcp_json_get_integer_value(ctx, "max_epochs", &int_value) == JSON_SUCCESS) {
        config->max_epochs = (uint32_t)int_value;
    }
    if (nimcp_json_get_integer_value(ctx, "batch_size", &int_value) == JSON_SUCCESS) {
        config->batch_size = (uint32_t)int_value;
    }
    if (nimcp_json_get_number_value(ctx, "validation_split", &double_value) == JSON_SUCCESS) {
        config->validation_split = (float)double_value;
    }
    if (nimcp_json_get_boolean_value(ctx, "early_stopping", &bool_value) == JSON_SUCCESS) {
        config->early_stopping = bool_value;
    }
    if (nimcp_json_get_integer_value(ctx, "patience", &int_value) == JSON_SUCCESS) {
        config->patience = (uint32_t)int_value;
    }

    // Plasticity section fields
    if (nimcp_json_get_boolean_value(ctx, "enable_bcm", &bool_value) == JSON_SUCCESS) {
        config->enable_bcm = bool_value;
    }
    if (nimcp_json_get_number_value(ctx, "bcm_tau", &double_value) == JSON_SUCCESS) {
        config->bcm_tau = (float)double_value;
    }
    if (nimcp_json_get_boolean_value(ctx, "enable_stdp", &bool_value) == JSON_SUCCESS) {
        config->enable_stdp = bool_value;
    }
    if (nimcp_json_get_number_value(ctx, "stdp_window", &double_value) == JSON_SUCCESS) {
        config->stdp_window = (float)double_value;
    }

    // Ethics section fields
    if (nimcp_json_get_boolean_value(ctx, "ethics_enabled", &bool_value) == JSON_SUCCESS) {
        config->ethics_enabled = bool_value;
    }
    if (nimcp_json_get_number_value(ctx, "golden_rule_threshold", &double_value) == JSON_SUCCESS) {
        config->golden_rule_threshold = (float)double_value;
    }
    if (nimcp_json_get_number_value(ctx, "empathy_weight", &double_value) == JSON_SUCCESS) {
        config->empathy_weight = (float)double_value;
    }

    nimcp_json_destroy_context(ctx);
    return true;
}

bool nimcp_config_load(const char* filepath, nimcp_brain_config_t* config) {
    // Auto-detect format based on extension
    size_t len = strlen(filepath);

    if (len > 5 && (strcmp(filepath + len - 5, ".yaml") == 0 ||
                    strcmp(filepath + len - 4, ".yml") == 0)) {
        return nimcp_config_load_yaml(filepath, config);
    } else if (len > 5 && strcmp(filepath + len - 5, ".json") == 0) {
        return nimcp_config_load_json(filepath, config);
    }

    // Default to YAML
    return nimcp_config_load_yaml(filepath, config);
}
