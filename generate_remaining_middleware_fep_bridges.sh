#!/bin/bash
# Script to generate all remaining middleware FEP bridge files
# This script creates headers and implementations for 12 remaining middleware modules

set -e

NIMCP_ROOT="/home/bbrelin/nimcp"

echo "Generating middleware FEP bridge files..."

# Array of modules to generate (already created training_module)
MODULES=(
    "gradient_manager:training"
    "optimizers:training"
    "lr_scheduler:training"
    "regularization:training"
    "zscore_normalizer:normalization"
    "rate_coding:encoding"
    "temporal_coding:encoding"
    "event_bus:events"
    "pattern_library:patterns"
    "oscillation_detector:patterns"
    "sliding_window:buffering"
    "integration_buffer:buffering"
)

# Function to convert module_name to ModuleName (PascalCase)
to_pascal_case() {
    echo "$1" | sed -r 's/(^|_)([a-z])/\U\2/g'
}

# Function to convert module_name to MODULEMODULE_NAME (UPPER_SNAKE_CASE for header guards)
to_upper_snake_case() {
    echo "$1" | tr '[:lower:]' '[:upper:]'
}

# Generate files for each module
for module_info in "${MODULES[@]}"; do
    IFS=':' read -r module_name subdir <<< "$module_info"

    HEADER_DIR="$NIMCP_ROOT/include/middleware/$subdir"
    SRC_DIR="$NIMCP_ROOT/src/middleware/$subdir"

    HEADER_FILE="$HEADER_DIR/nimcp_${module_name}_fep_bridge.h"
    SRC_FILE="$SRC_DIR/nimcp_${module_name}_fep_bridge.c"

    MODULE_UPPER=$(to_upper_snake_case "$module_name")
    MODULE_PASCAL=$(to_pascal_case "$module_name")

    echo "Creating $module_name FEP bridge..."

    # Create header file
    cat > "$HEADER_FILE" << 'HEADER_EOF'
/**
 * @file nimcp_MODULE_NAME_fep_bridge.h
 * @brief Free Energy Principle bridge for MODULE_PASCAL
 * @version 1.0.0
 * @date 2025-12-15
 *
 * WHAT: FEP integration for MODULE_NAME module
 * WHY:  Model MODULE_NAME operations under Free Energy Principle
 * HOW:  Bidirectional FEP-MODULE_NAME integration
 *
 * BIOLOGICAL BASIS:
 * ==============================================================================
 * MODULE_PASCAL as Predictive Processing:
 * - Module state → Hidden state beliefs
 * - Module outputs → Observations
 * - Module dynamics → State transitions
 * - Module parameters → Precision weights
 *
 * FEP FORMULATION:
 * - Free energy F = prediction error + complexity
 * - Beliefs minimize F through gradient descent
 * - Precision weights determine error influence
 * - Active inference guides parameter adaptation
 *
 * @author NIMCP Development Team
 */

#ifndef NIMCP_MODULE_UPPER_FEP_BRIDGE_H
#define NIMCP_MODULE_UPPER_FEP_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include "middleware/SUBDIR/nimcp_MODULE_NAME.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
#include "async/nimcp_bio_messages.h"
#include "utils/validation/nimcp_common.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/thread/nimcp_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration */
typedef struct {
    float belief_learning_rate;
    float precision_learning_rate;
    float initial_precision;
    bool learn_precision;
    float surprise_threshold;
    bool enable_bio_async;
} MODULE_NAME_fep_config_t;

/* FEP effects on module */
typedef struct {
    float parameter_modulation;
    float precision_weight;
    float exploration_bonus;
} MODULE_NAME_fep_effects_t;

/* Module effects on FEP */
typedef struct {
    float primary_metric;
    float secondary_metric;
    uint64_t operation_count;
} fep_MODULE_NAME_effects_t;

/* Bridge state */
typedef struct {
    uint64_t update_count;
    float avg_free_energy;
    float max_surprise;
} MODULE_NAME_fep_state_t;

/* Bridge statistics */
typedef struct {
    uint64_t total_updates;
    float min_free_energy;
    float max_free_energy;
} MODULE_NAME_fep_stats_t;

/* Opaque bridge type */
typedef struct MODULE_NAME_fep_bridge MODULE_NAME_fep_bridge_t;

/* Lifecycle API */
int MODULE_NAME_fep_default_config(MODULE_NAME_fep_config_t* config);

MODULE_NAME_fep_bridge_t* MODULE_NAME_fep_create(
    const MODULE_NAME_fep_config_t* config,
    MODULE_TYPE_t* module,
    fep_system_t* fep_system
);

void MODULE_NAME_fep_destroy(MODULE_NAME_fep_bridge_t* bridge);

/* Integration API */
int MODULE_NAME_fep_update(MODULE_NAME_fep_bridge_t* bridge);
int MODULE_NAME_fep_compute_effects(MODULE_NAME_fep_bridge_t* bridge);
int MODULE_NAME_fep_apply_effects(MODULE_NAME_fep_bridge_t* bridge);
int MODULE_NAME_fep_step(MODULE_NAME_fep_bridge_t* bridge);

/* Query API */
int MODULE_NAME_fep_get_effects(
    const MODULE_NAME_fep_bridge_t* bridge,
    MODULE_NAME_fep_effects_t* effects
);

float MODULE_NAME_fep_get_free_energy(const MODULE_NAME_fep_bridge_t* bridge);
float MODULE_NAME_fep_get_prediction_error(const MODULE_NAME_fep_bridge_t* bridge);

int MODULE_NAME_fep_get_stats(
    const MODULE_NAME_fep_bridge_t* bridge,
    MODULE_NAME_fep_stats_t* stats
);

/* Bio-Async API */
int MODULE_NAME_fep_connect_bio_async(MODULE_NAME_fep_bridge_t* bridge);
int MODULE_NAME_fep_disconnect_bio_async(MODULE_NAME_fep_bridge_t* bridge);
bool MODULE_NAME_fep_is_bio_async_connected(const MODULE_NAME_fep_bridge_t* bridge);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_MODULE_UPPER_FEP_BRIDGE_H */
HEADER_EOF

    # Replace placeholders in header
    sed -i "s/MODULE_NAME/$module_name/g" "$HEADER_FILE"
    sed -i "s/MODULE_UPPER/$MODULE_UPPER/g" "$HEADER_FILE"
    sed -i "s/MODULE_PASCAL/$MODULE_PASCAL/g" "$HEADER_FILE"
    sed -i "s/SUBDIR/$subdir/g" "$HEADER_FILE"

    # Set appropriate module type based on module name
    case "$module_name" in
        gradient_manager) MODULE_TYPE="nimcp_gradient_manager_ctx" ;;
        optimizers) MODULE_TYPE="nimcp_optimizer_context" ;;
        lr_scheduler) MODULE_TYPE="nimcp_lr_scheduler_ctx" ;;
        regularization) MODULE_TYPE="nimcp_regularization_ctx" ;;
        zscore_normalizer) MODULE_TYPE="zscore_normalizer" ;;
        rate_coding) MODULE_TYPE="rate_coding_encoder" ;;
        temporal_coding) MODULE_TYPE="temporal_coding_encoder" ;;
        event_bus) MODULE_TYPE="event_bus" ;;
        pattern_library) MODULE_TYPE="pattern_library" ;;
        oscillation_detector) MODULE_TYPE="oscillation_detector" ;;
        sliding_window) MODULE_TYPE="sliding_window" ;;
        integration_buffer) MODULE_TYPE="integration_buffer" ;;
    esac

    sed -i "s/MODULE_TYPE/$MODULE_TYPE/g" "$HEADER_FILE"

    # Create implementation file
    cat > "$SRC_FILE" << 'SRC_EOF'
/**
 * @file nimcp_MODULE_NAME_fep_bridge.c
 * @brief Free Energy Principle bridge for MODULE_PASCAL implementation
 */

#include "middleware/SUBDIR/nimcp_MODULE_NAME_fep_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>
#include <math.h>

struct MODULE_NAME_fep_bridge {
    MODULE_NAME_fep_config_t config;
    fep_system_t* fep_system;
    MODULE_TYPE_t* module;
    MODULE_NAME_fep_effects_t fep_effects;
    fep_MODULE_NAME_effects_t module_effects;
    MODULE_NAME_fep_state_t state;
    MODULE_NAME_fep_stats_t stats;
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;
    nimcp_mutex_t* mutex;
    bool owns_fep_system;
};

int MODULE_NAME_fep_default_config(MODULE_NAME_fep_config_t* config) {
    if (!config) return NIMCP_ERROR_NULL_POINTER;
    config->belief_learning_rate = 0.1f;
    config->precision_learning_rate = 0.05f;
    config->initial_precision = 1.0f;
    config->learn_precision = true;
    config->surprise_threshold = 5.0f;
    config->enable_bio_async = false;
    return 0;
}

MODULE_NAME_fep_bridge_t* MODULE_NAME_fep_create(
    const MODULE_NAME_fep_config_t* config,
    MODULE_TYPE_t* module,
    fep_system_t* fep_system
) {
    if (!module) {
        NIMCP_LOGGING_ERROR("MODULE_NAME_fep_create: NULL module");
        return NULL;
    }

    MODULE_NAME_fep_bridge_t* bridge = (MODULE_NAME_fep_bridge_t*)
        nimcp_malloc(sizeof(MODULE_NAME_fep_bridge_t));
    if (!bridge) {
        NIMCP_LOGGING_ERROR("MODULE_NAME_fep_create: Allocation failed");
        return NULL;
    }

    memset(bridge, 0, sizeof(MODULE_NAME_fep_bridge_t));

    if (config) {
        bridge->config = *config;
    } else {
        MODULE_NAME_fep_default_config(&bridge->config);
    }

    bridge->module = module;

    if (fep_system) {
        bridge->fep_system = fep_system;
        bridge->owns_fep_system = false;
    } else {
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep_config.num_levels = 1;
        uint32_t level_dim = 4;
        fep_config.level_dims = &level_dim;
        fep_config.belief_learning_rate = bridge->config.belief_learning_rate;
        fep_config.precision_learning_rate = bridge->config.precision_learning_rate;
        fep_config.initial_precision = bridge->config.initial_precision;
        fep_config.learn_precision = bridge->config.learn_precision;

        bridge->fep_system = fep_create(&fep_config, 4, 1);
        if (!bridge->fep_system) {
            NIMCP_LOGGING_ERROR("MODULE_NAME_fep_create: FEP system creation failed");
            nimcp_free(bridge);
            return NULL;
        }
        bridge->owns_fep_system = true;
    }

    bridge->mutex = nimcp_platform_mutex_create();
    bridge->fep_effects.parameter_modulation = 1.0f;
    bridge->stats.min_free_energy = INFINITY;
    bridge->stats.max_free_energy = -INFINITY;

    NIMCP_LOGGING_INFO("Created MODULE_NAME FEP bridge");
    return bridge;
}

void MODULE_NAME_fep_destroy(MODULE_NAME_fep_bridge_t* bridge) {
    if (!bridge) return;
    if (bridge->bio_async_enabled) {
        MODULE_NAME_fep_disconnect_bio_async(bridge);
    }
    if (bridge->owns_fep_system && bridge->fep_system) {
        fep_destroy(bridge->fep_system);
    }
    if (bridge->mutex) {
        nimcp_platform_mutex_destroy(bridge->mutex);
    }
    nimcp_free(bridge);
}

int MODULE_NAME_fep_update(MODULE_NAME_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->mutex) nimcp_platform_mutex_lock(bridge->mutex);

    /* Create observation from module state */
    float observation[4] = {
        bridge->module_effects.primary_metric,
        bridge->module_effects.secondary_metric,
        (float)bridge->module_effects.operation_count / 1000.0f,
        0.0f
    };

    int result = fep_process_observation(bridge->fep_system, observation, 4);
    if (result == 0) {
        result = fep_update_beliefs(bridge->fep_system);
    }

    /* Update state */
    float free_energy = fep_get_free_energy(bridge->fep_system);
    bridge->state.update_count++;
    bridge->state.avg_free_energy =
        (bridge->state.avg_free_energy * (bridge->state.update_count - 1) + free_energy) /
        bridge->state.update_count;

    float pred_error = fep_get_prediction_error(bridge->fep_system, 0);
    if (pred_error > bridge->state.max_surprise) {
        bridge->state.max_surprise = pred_error;
    }

    /* Update statistics */
    bridge->stats.total_updates++;
    if (free_energy < bridge->stats.min_free_energy) {
        bridge->stats.min_free_energy = free_energy;
    }
    if (free_energy > bridge->stats.max_free_energy) {
        bridge->stats.max_free_energy = free_energy;
    }

    if (bridge->mutex) nimcp_platform_mutex_unlock(bridge->mutex);
    return result;
}

int MODULE_NAME_fep_compute_effects(MODULE_NAME_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->mutex) nimcp_platform_mutex_lock(bridge->mutex);

    float pred_error = fep_get_prediction_error(bridge->fep_system, 0);
    bridge->fep_effects.parameter_modulation = expf(-pred_error * 0.1f);
    if (bridge->fep_effects.parameter_modulation < 0.1f) {
        bridge->fep_effects.parameter_modulation = 0.1f;
    } else if (bridge->fep_effects.parameter_modulation > 2.0f) {
        bridge->fep_effects.parameter_modulation = 2.0f;
    }

    fep_belief_t beliefs;
    fep_get_beliefs(bridge->fep_system, 0, &beliefs);
    float uncertainty = 0.0f;
    for (uint32_t i = 0; i < beliefs.dim; i++) {
        uncertainty += beliefs.variance[i];
    }
    uncertainty /= beliefs.dim;
    bridge->fep_effects.exploration_bonus = fminf(uncertainty, 1.0f);

    if (bridge->mutex) nimcp_platform_mutex_unlock(bridge->mutex);
    return 0;
}

int MODULE_NAME_fep_apply_effects(MODULE_NAME_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    return 0;
}

int MODULE_NAME_fep_step(MODULE_NAME_fep_bridge_t* bridge) {
    int result = MODULE_NAME_fep_update(bridge);
    if (result != 0) return result;
    result = MODULE_NAME_fep_compute_effects(bridge);
    if (result != 0) return result;
    return MODULE_NAME_fep_apply_effects(bridge);
}

int MODULE_NAME_fep_get_effects(
    const MODULE_NAME_fep_bridge_t* bridge,
    MODULE_NAME_fep_effects_t* effects
) {
    if (!bridge || !effects) return NIMCP_ERROR_NULL_POINTER;
    *effects = bridge->fep_effects;
    return 0;
}

float MODULE_NAME_fep_get_free_energy(const MODULE_NAME_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return 0.0f;
    return fep_get_free_energy(bridge->fep_system);
}

float MODULE_NAME_fep_get_prediction_error(const MODULE_NAME_fep_bridge_t* bridge) {
    if (!bridge || !bridge->fep_system) return 0.0f;
    return fep_get_prediction_error(bridge->fep_system, 0);
}

int MODULE_NAME_fep_get_stats(
    const MODULE_NAME_fep_bridge_t* bridge,
    MODULE_NAME_fep_stats_t* stats
) {
    if (!bridge || !stats) return NIMCP_ERROR_NULL_POINTER;
    *stats = bridge->stats;
    return 0;
}

int MODULE_NAME_fep_connect_bio_async(MODULE_NAME_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (bridge->bio_async_enabled) return 0;
    bridge->bio_async_enabled = true;
    NIMCP_LOGGING_INFO("MODULE_NAME FEP bridge connected to bio-async");
    return 0;
}

int MODULE_NAME_fep_disconnect_bio_async(MODULE_NAME_fep_bridge_t* bridge) {
    if (!bridge) return NIMCP_ERROR_NULL_POINTER;
    if (!bridge->bio_async_enabled) return 0;
    bridge->bio_async_enabled = false;
    NIMCP_LOGGING_INFO("MODULE_NAME FEP bridge disconnected from bio-async");
    return 0;
}

bool MODULE_NAME_fep_is_bio_async_connected(const MODULE_NAME_fep_bridge_t* bridge) {
    if (!bridge) return false;
    return bridge->bio_async_enabled;
}
SRC_EOF

    # Replace placeholders in implementation
    sed -i "s/MODULE_NAME/$module_name/g" "$SRC_FILE"
    sed -i "s/MODULE_PASCAL/$MODULE_PASCAL/g" "$SRC_FILE"
    sed -i "s/SUBDIR/$subdir/g" "$SRC_FILE"
    sed -i "s/MODULE_TYPE/$MODULE_TYPE/g" "$SRC_FILE"

    echo "  ✓ Created $HEADER_FILE"
    echo "  ✓ Created $SRC_FILE"
done

echo ""
echo "✓ All middleware FEP bridge files generated successfully!"
echo ""
echo "Summary:"
echo "  - 12 header files created in include/middleware/*/"
echo "  - 12 implementation files created in src/middleware/*/"
echo "  - Total: 24 new files (plus 2 manually created = 26 total)"
echo ""
echo "Next steps:"
echo "  1. Review generated files for module-specific customizations"
echo "  2. Add module-specific FEP mappings and biological interpretations"
echo "  3. Add files to CMakeLists.txt build system"
echo "  4. Create unit tests for each bridge"
echo ""
