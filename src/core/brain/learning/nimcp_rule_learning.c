//=============================================================================
// nimcp_rule_learning.c - Inductive Rule Learning Implementation
//=============================================================================

#include "core/brain/learning/nimcp_rule_learning.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "core_rule_learning"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(rule_learning)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_rule_learning_mesh_id = 0;
static mesh_participant_registry_t* g_rule_learning_mesh_registry = NULL;

nimcp_error_t rule_learning_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_rule_learning_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "rule_learning", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "rule_learning";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_rule_learning_mesh_id);
    if (err == NIMCP_SUCCESS) g_rule_learning_mesh_registry = registry;
    return err;
}

void rule_learning_mesh_unregister(void) {
    if (g_rule_learning_mesh_registry && g_rule_learning_mesh_id != 0) {
        mesh_participant_unregister(g_rule_learning_mesh_registry, g_rule_learning_mesh_id);
        g_rule_learning_mesh_id = 0;
        g_rule_learning_mesh_registry = NULL;
    }
}


#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

//=============================================================================
// Rule Learning Implementation
//=============================================================================

int brain_learn_rule_from_examples(brain_t brain, const rule_example_t* examples,
                                     const char** labels, uint32_t count) {
    if (!brain || !examples || !labels || count == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
            "brain_learn_rule_from_examples: invalid parameters");
        return -1;
    }

    int rules_learned = 0;

    // Group examples by label
    for (uint32_t i = 0; i < count; i++) {
        const char* current_label = labels[i];

        // Find all examples with this label
        rule_example_t* group = (rule_example_t*)nimcp_malloc(count * sizeof(rule_example_t));
        uint32_t group_size = 0;

        for (uint32_t j = 0; j < count; j++) {
            if (strcmp(labels[j], current_label) == 0) {
                group[group_size++] = examples[j];
            }
        }

        // Extract pattern from grouped examples
        char rule_str[512];
        if (extract_rule_pattern(group, group_size, current_label, rule_str, sizeof(rule_str))) {
            // Compute confidence
            float confidence = compute_rule_confidence(group_size, count);

            // Add to KB
            if (add_learned_rule_to_kb(brain, rule_str, confidence)) {
                rules_learned++;
                LOG_INFO("rule_learning: Learned rule: %s (confidence: %.2f)",
                         rule_str, confidence);
            }
        }

        nimcp_free(group);
    }

    return rules_learned;
}

bool extract_rule_pattern(const rule_example_t* examples, uint32_t count,
                          const char* label, char* rule_out, size_t rule_size) {
    if (!examples || count == 0 || !label || !rule_out) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,

                "extract_rule_pattern: invalid parameters");

            return false;
    }

    // Find common features (threshold: present in >80% of examples)
    uint32_t num_features = examples[0].num_features;
    bool* common_features = (bool*)nimcp_calloc(num_features, sizeof(bool));

    for (uint32_t f = 0; f < num_features; f++) {
        uint32_t present_count = 0;

        // Count how many examples have this feature active
        for (uint32_t i = 0; i < count; i++) {
            if (examples[i].features[f] > 0.5F) {
                present_count++;
            }
        }

        // Feature is common if present in >80% of examples
        if (present_count >= (count * 4) / 5) {
            common_features[f] = true;
        }
    }

    // Build rule string
    int offset = snprintf(rule_out, rule_size, "IF ");
    // SECURITY FIX: Clamp offset to prevent buffer overflow (snprintf returns would-be length, not actual)
    if (offset >= (int)rule_size) offset = rule_size - 1;
    bool first = true;

    for (uint32_t f = 0; f < num_features; f++) {
        if (common_features[f]) {
            if (!first) {
                int written = snprintf(rule_out + offset, rule_size - offset, " AND ");
                offset += written;
                if (offset >= (int)rule_size) offset = rule_size - 1;  // Clamp to prevent overflow
            }
            int written = snprintf(rule_out + offset, rule_size - offset, "feature_%u", f);
            offset += written;
            if (offset >= (int)rule_size) offset = rule_size - 1;  // Clamp to prevent overflow
            first = false;
        }
    }

    if (first) {
        // No common features found
        nimcp_free(common_features);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "rule_learning_mesh_unregister: validation failed");
        return false;
    }

    snprintf(rule_out + offset, rule_size - offset, " THEN %s", label);
    nimcp_free(common_features);

    return true;
}

bool add_learned_rule_to_kb(brain_t brain, const char* rule, float confidence) {
    if (!brain || !rule) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,

                "add_learned_rule_to_kb: invalid parameters");

            return false;
    }

    // In a full implementation, this would integrate with the KB module
    // For now, we log the rule
    LOG_INFO("rule_learning: Adding rule to KB: %s (confidence: %.2f)",
             rule, confidence);

    // TODO: Integration with knowledge base module
    // kb_add_rule(b->knowledge_base, rule, confidence);

    return true;
}

float compute_rule_confidence(uint32_t support_count, uint32_t total_count) {
    if (total_count == 0) {
        return 0.0F;
    }

    // Laplace smoothing
    const float alpha = 1.0F;
    const float beta = 2.0F;

    float confidence = (float)(support_count + alpha) / (float)(total_count + beta);

    // Clamp to [0.0, 1.0]
    if (confidence < 0.0F) confidence = 0.0F;
    if (confidence > 1.0F) confidence = 1.0F;

    return confidence;
}
