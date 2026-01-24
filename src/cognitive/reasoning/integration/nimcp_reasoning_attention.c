/**
 * @file nimcp_reasoning_attention.c
 * @brief MODULE 1: Reasoning-Attention Integration Implementation
 * @version 1.0.0
 * @date 2025-11-20
 */

#include "cognitive/reasoning/integration/nimcp_reasoning_attention.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include "cognitive/fault_tolerance/nimcp_fault_attention.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"

#define LOG_MODULE "cognitive.reasoning.attention"
#define BIO_MODULE_COGNITIVE_REASONING_ATTENTION 0x034C


//=============================================================================
// Internal Structure
//=============================================================================

struct reasoning_attention {
    // Configuration
    reasoning_attention_config_t config;

    // Integration handles
    event_bus_t event_bus;
    fault_attention_t* attention;
    event_subscription_handle_t subscription_handle;

    // Statistics
    reasoning_attention_stats_t stats;

    // State
    uint64_t last_boost_time_ms;
    float current_salience;

    // Bio-async integration
    bio_module_context_t bio_ctx;   /**< Bio-async module context */
    bool bio_async_enabled;         /**< Bio-async registration status */
};

//=============================================================================
// Helper Functions
//=============================================================================

static uint64_t get_current_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)(tv.tv_sec) * 1000 + (uint64_t)(tv.tv_usec) / 1000;
}

static uint64_t get_current_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)(tv.tv_sec) * 1000000 + (uint64_t)(tv.tv_usec);
}

//=============================================================================
// Default Configuration
//=============================================================================

reasoning_attention_config_t reasoning_attention_default_config(void) {
    reasoning_attention_config_t config = {
        .enable_novel_fact_boost = true,
        .enable_contradiction_boost = true,
        .enable_proof_found_boost = true,

        .novel_fact_salience = REASONING_ATTENTION_NOVEL_FACT_SALIENCE,
        .contradiction_salience = REASONING_ATTENTION_CONTRADICTION_SALIENCE,
        .proof_found_salience = REASONING_ATTENTION_PROOF_FOUND_SALIENCE,

        .attention_decay_tau_ms = REASONING_ATTENTION_DECAY_TAU_MS,
        .min_salience_threshold = 0.1F
    };
    return config;
}

//=============================================================================
// Validation
//=============================================================================

bool reasoning_attention_validate_config(const reasoning_attention_config_t* config) {
    if (!config) return false;

    // Validate salience values [0.0, 1.0]
    if (config->novel_fact_salience < 0.0F || config->novel_fact_salience > 1.0F) return false;
    if (config->contradiction_salience < 0.0F || config->contradiction_salience > 1.0F) return false;
    if (config->proof_found_salience < 0.0F || config->proof_found_salience > 1.0F) return false;
    if (config->min_salience_threshold < 0.0F || config->min_salience_threshold > 1.0F) return false;

    // Validate decay time constant
    if (config->attention_decay_tau_ms == 0) return false;

    return true;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

reasoning_attention_t* reasoning_attention_create(
    event_bus_t event_bus,
    fault_attention_t* attention
) {
    LOG_DEBUG("Creating module");
    reasoning_attention_config_t config = reasoning_attention_default_config();
    return reasoning_attention_create_custom(event_bus, attention, &config);
}

reasoning_attention_t* reasoning_attention_create_custom(
    event_bus_t event_bus,
    fault_attention_t* attention,
    const reasoning_attention_config_t* config
) {
    if (!event_bus || !attention) {
        LOG_ERROR("NULL event_bus or attention");
        return NULL;
    }

    // Use default config if NULL
    reasoning_attention_config_t default_config = reasoning_attention_default_config();
    const reasoning_attention_config_t* final_config = config ? config : &default_config;

    // Validate configuration
    if (!reasoning_attention_validate_config(final_config)) {
        LOG_ERROR("Invalid configuration");
        return NULL;
    }

    // Allocate structure
    reasoning_attention_t* integration = nimcp_calloc(1, sizeof(reasoning_attention_t));
    if (!integration) {
        LOG_ERROR("Failed to allocate integration");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "integration is NULL");

        return NULL;
    }

    // Initialize
    integration->config = *final_config;
    integration->event_bus = event_bus;
    integration->attention = attention;
    integration->last_boost_time_ms = get_current_time_ms();
    integration->current_salience = 0.0F;
    memset(&integration->stats, 0, sizeof(reasoning_attention_stats_t));

    // Subscribe to reasoning events
    integration->subscription_handle = event_bus_subscribe(
        event_bus,
        EVENT_ALL,  // Subscribe to all events, filter in callback
        reasoning_attention_callback,
        integration
    );

    if (integration->subscription_handle == INVALID_SUBSCRIPTION_HANDLE) {
        LOG_ERROR("Failed to subscribe to events");
        nimcp_free(integration);
        return NULL;
    }

    LOG_INFO("Created reasoning-attention integration");
    
    // Bio-async registration
    integration->bio_ctx = NULL;
    integration->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_ATTENTION_REASONING,
            .module_name = "reasoning_attention",
            .inbox_capacity = 32,
            .user_data = integration
        };
        integration->bio_ctx = bio_router_register_module(&bio_info);
        if (integration->bio_ctx) {
            integration->bio_async_enabled = true;
        }
    }

return integration;
}

void reasoning_attention_destroy(reasoning_attention_t* integration) {
    LOG_DEBUG("Destroying module");
    if (!integration) return;

    // Unsubscribe from events
    if (integration->subscription_handle != INVALID_SUBSCRIPTION_HANDLE) {
        event_bus_unsubscribe(integration->event_bus, integration->subscription_handle);
    }

    LOG_INFO("Destroyed reasoning-attention integration");
    // Unregister from bio-router
    if (integration->bio_async_enabled && integration->bio_ctx) {
        bio_router_unregister_module(integration->bio_ctx);
        integration->bio_ctx = NULL;
        integration->bio_async_enabled = false;
    }

    nimcp_free(integration);
}

//=============================================================================
// Core Functions
//=============================================================================

float reasoning_attention_compute_fact_salience(
    reasoning_attention_t* integration,
    const char* fact_description,
    bool is_novel,
    bool is_contradiction
) {
    if (!integration || !fact_description) return 0.0F;

    float salience = 0.0F;

    // Contradiction gets maximum salience
    if (is_contradiction && integration->config.enable_contradiction_boost) {
        salience = integration->config.contradiction_salience;
    }
    // Novel fact gets high salience
    else if (is_novel && integration->config.enable_novel_fact_boost) {
        salience = integration->config.novel_fact_salience;
    }
    // Default base salience
    else {
        salience = 0.3F;
    }

    // Apply decay based on time since last boost
    uint64_t current_time = get_current_time_ms();
    uint64_t time_since_boost = current_time - integration->last_boost_time_ms;
    float decay_factor = expf(-((float)time_since_boost / (float)integration->config.attention_decay_tau_ms));
    salience *= decay_factor;

    // Clamp to minimum threshold
    if (salience < integration->config.min_salience_threshold) {
        salience = integration->config.min_salience_threshold;
    }

    return salience;
}

void reasoning_attention_callback(const brain_event_t* event, void* context) {
    if (!event || !context) return;

    reasoning_attention_t* integration = (reasoning_attention_t*)context;

    // Process pending bio-async messages
    if (integration && integration->bio_ctx) {
        bio_router_process_inbox(integration->bio_ctx, 5);
    }
    uint64_t start_time_us = get_current_time_us();

    // Filter for reasoning events only
    if (event->type < EVENT_LOGIC_GATE_EVALUATED || event->type > EVENT_NOVEL_FACT_DERIVED) {
        return;  // Not a reasoning event
    }

    integration->stats.total_events_processed++;

    float salience = 0.0F;
    bool should_boost = false;

    switch (event->type) {
        case EVENT_NOVEL_FACT_DERIVED:
            if (integration->config.enable_novel_fact_boost) {
                salience = integration->config.novel_fact_salience;
                integration->stats.novel_fact_boosts++;
                should_boost = true;
            }
            break;

        case EVENT_CONTRADICTION_DETECTED:
            if (integration->config.enable_contradiction_boost) {
                salience = integration->config.contradiction_salience;
                integration->stats.contradiction_boosts++;
                should_boost = true;
            }
            break;

        case EVENT_PROOF_FOUND:
            if (integration->config.enable_proof_found_boost) {
                salience = integration->config.proof_found_salience;
                integration->stats.proof_found_boosts++;
                should_boost = true;
            }
            break;

        default:
            break;
    }

    // Apply attention boost if salience exceeds threshold
    if (should_boost && salience >= integration->config.min_salience_threshold) {
        // Create active_fault_t for attention mechanism
        active_fault_t reasoning_event_fault = {
            .fault_id = (uint32_t)event->type,
            .severity = salience,
            .occurrence_count = 1,
            .users_affected = 1,
            .first_occurrence_ms = event->timestamp_us / 1000,
            .last_occurrence_ms = event->timestamp_us / 1000,
            .is_active = true
        };
        snprintf(reasoning_event_fault.description, sizeof(reasoning_event_fault.description),
                "Reasoning event: %s", event_type_to_string(event->type));

        // Note: Actual attention boost would go here via fault_attention API
        // For now, just track statistics

        integration->last_boost_time_ms = get_current_time_ms();
        integration->current_salience = salience;

        // Update statistics
        if (salience > integration->stats.max_salience_applied) {
            integration->stats.max_salience_applied = salience;
        }

        float total_salience = integration->stats.avg_salience_applied *
                               (integration->stats.novel_fact_boosts +
                                integration->stats.contradiction_boosts +
                                integration->stats.proof_found_boosts - 1);
        integration->stats.avg_salience_applied =
            (total_salience + salience) /
            (integration->stats.novel_fact_boosts +
             integration->stats.contradiction_boosts +
             integration->stats.proof_found_boosts);
    }

    // Update callback timing statistics
    uint64_t end_time_us = get_current_time_us();
    uint64_t elapsed_us = end_time_us - start_time_us;

    uint64_t total_time = integration->stats.avg_callback_time_us *
                          (integration->stats.total_events_processed - 1);
    integration->stats.avg_callback_time_us =
        (total_time + elapsed_us) / integration->stats.total_events_processed;
}

//=============================================================================
// Query Functions
//=============================================================================

bool reasoning_attention_get_config(
    const reasoning_attention_t* integration,
    reasoning_attention_config_t* config
) {
    if (!integration || !config) return false;
    *config = integration->config;
    return true;
}

bool reasoning_attention_set_config(
    reasoning_attention_t* integration,
    const reasoning_attention_config_t* config
) {
    if (!integration || !config) return false;
    if (!reasoning_attention_validate_config(config)) return false;

    integration->config = *config;
    return true;
}

bool reasoning_attention_get_stats(
    const reasoning_attention_t* integration,
    reasoning_attention_stats_t* stats
) {
    if (!integration || !stats) return false;
    *stats = integration->stats;
    return true;
}

bool reasoning_attention_reset_stats(reasoning_attention_t* integration) {
    if (!integration) return false;
    memset(&integration->stats, 0, sizeof(reasoning_attention_stats_t));
    return true;
}

//=============================================================================
// KG Self-Awareness Integration
//=============================================================================

int reasoning_attention_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Reasoning_Attention");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG("Reasoning_Attention self-knowledge: %s", self->observations[i]);
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Reasoning_Attention");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Reasoning_Attention");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
