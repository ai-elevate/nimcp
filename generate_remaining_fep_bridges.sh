#!/bin/bash

# Script to generate remaining FEP bridge implementations
# Run this from /home/bbrelin/nimcp directory

set -e

echo "Generating remaining FEP bridge implementations..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to generate header file
generate_header() {
    local module=$1
    local MODULE=$(echo $module | tr '[:lower:]' '[:upper:]')
    local header_path="include/cognitive/${module}/nimcp_${module}_fep_bridge.h"

    echo -e "${BLUE}Generating ${header_path}...${NC}"

    cat > "$header_path" <<'EOF'
/**
 * @file nimcp_${module}_fep_bridge.h
 * @brief Free Energy Principle - ${Module} Integration Bridge
 * @version 1.0.0
 * @date 2025-12-12
 */

#ifndef NIMCP_${MODULE}_FEP_BRIDGE_H
#define NIMCP_${MODULE}_FEP_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "cognitive/free_energy/nimcp_free_energy.h"
#include "cognitive/nimcp_${module}.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration */
typedef struct {
    /* FEP → Module */
    bool enable_precision_modulation;
    float precision_sensitivity;

    /* Module → FEP */
    bool enable_state_feedback;
    float module_sensitivity;
} ${module}_fep_config_t;

/* FEP effects */
typedef struct {
    float precision_value;
    float prediction_error;
} ${module}_fep_effects_t;

/* Module effects on FEP */
typedef struct {
    float module_state;
    float precision_modulation;
} fep_${module}_effects_t;

/* State */
typedef struct {
    float current_precision;
    float current_module_state;
    uint64_t last_update_time;
} ${module}_fep_state_t;

/* Statistics */
typedef struct {
    uint64_t update_count;
    float avg_precision;
    float avg_free_energy;
} ${module}_fep_stats_t;

/* Bridge */
typedef struct ${module}_fep_bridge {
    ${module}_fep_config_t config;
    fep_system_t* fep_system;
    ${module}_t* ${module};
    ${module}_fep_effects_t fep_effects;
    fep_${module}_effects_t module_effects;
    ${module}_fep_state_t state;
    ${module}_fep_stats_t stats;
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;
    void* mutex;
} ${module}_fep_bridge_t;

/* Lifecycle */
int ${module}_fep_bridge_default_config(${module}_fep_config_t* config);
${module}_fep_bridge_t* ${module}_fep_bridge_create(const ${module}_fep_config_t* config);
void ${module}_fep_bridge_destroy(${module}_fep_bridge_t* bridge);

/* Connection */
int ${module}_fep_bridge_connect_fep(${module}_fep_bridge_t* bridge, fep_system_t* fep);
int ${module}_fep_bridge_connect_${module}(${module}_fep_bridge_t* bridge, ${module}_t* mod);
int ${module}_fep_bridge_disconnect(${module}_fep_bridge_t* bridge);

/* Update */
int ${module}_fep_bridge_update(${module}_fep_bridge_t* bridge, uint64_t delta_ms);

/* State/Stats */
int ${module}_fep_bridge_get_state(const ${module}_fep_bridge_t* bridge, ${module}_fep_state_t* state);
int ${module}_fep_bridge_get_stats(const ${module}_fep_bridge_t* bridge, ${module}_fep_stats_t* stats);

/* Bio-Async */
int ${module}_fep_bridge_connect_bio_async(${module}_fep_bridge_t* bridge);
int ${module}_fep_bridge_disconnect_bio_async(${module}_fep_bridge_t* bridge);
bool ${module}_fep_bridge_is_bio_async_connected(const ${module}_fep_bridge_t* bridge);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_${MODULE}_FEP_BRIDGE_H */
EOF

    # Replace placeholders
    sed -i "s/\${module}/${module}/g" "$header_path"
    sed -i "s/\${MODULE}/${MODULE}/g" "$header_path"
    sed -i "s/\${Module}/$(echo ${module:0:1} | tr '[:lower:]' '[:upper:]')${module:1}/g" "$header_path"
}

# Generate implementations for remaining modules
modules=("wellbeing" "sleep_wake" "meta_learning" "consolidation")

for module in "${modules[@]}"; do
    echo -e "${GREEN}Processing $module...${NC}"
    generate_header "$module"
done

echo -e "${GREEN}Done! Remember to:${NC}"
echo "1. Review and customize each generated file"
echo "2. Add module-specific biological pathways"
echo "3. Implement specific FEP integration logic"
echo "4. Create unit tests"
echo "5. Update bio_messages.h with BIO_MODULE_FEP_* definitions"
echo "6. Update CMakeLists.txt"
