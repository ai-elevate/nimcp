//=============================================================================
// nimcp_creative_emotion_bridge.c - Creative-Emotion Bridge Implementation
//=============================================================================
/**
 * @file nimcp_creative_emotion_bridge.c
 * @brief Bridge connecting creative system to emotion system
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-01-30
 */

#include "cognitive/creative/appreciation/nimcp_creative_emotion_bridge.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

#define LOG_MODULE "CREATIVE_EMO_BRIDGE"

//=============================================================================
// Configuration Defaults
//=============================================================================

void creative_emotion_bridge_config_defaults(creative_emotion_bridge_config_t* config) {
    if (!config) return;

    config->emotion_sensitivity = 1.0f;
    config->awe_threshold = 0.8f;
    config->chills_threshold = 0.9f;
    config->propagate_to_emotion_system = false;
    config->receive_from_emotion_system = false;
    config->emotion_decay_rate = 0.1f;
    config->store_emotional_memories = false;
    config->memory_threshold = 0.7f;
    config->max_emotional_memories = 100;
}

//=============================================================================
// Lifecycle API
//=============================================================================

creative_emotion_bridge_t* creative_emotion_bridge_create(
    const creative_emotion_bridge_config_t* config) {

    creative_emotion_bridge_t* bridge = nimcp_calloc(1, sizeof(creative_emotion_bridge_t));
    if (!bridge) {
        LOG_ERROR(LOG_MODULE, "Failed to allocate emotion bridge");
        return NULL;
    }

    if (config) {
        memcpy(&bridge->config, config, sizeof(creative_emotion_bridge_config_t));
    } else {
        creative_emotion_bridge_config_defaults(&bridge->config);
    }

    bridge->events_processed = 0;
    bridge->avg_engagement = 0.0f;
    bridge->last_update_us = 0;

    LOG_INFO(LOG_MODULE, "Creative emotion bridge created");
    return bridge;
}

void creative_emotion_bridge_destroy(creative_emotion_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->event_history) {
        nimcp_free(bridge->event_history);
    }

    nimcp_free(bridge);
    LOG_INFO(LOG_MODULE, "Creative emotion bridge destroyed");
}

//=============================================================================
// Processing API - Stubs
//=============================================================================

int creative_emotion_bridge_process(creative_emotion_bridge_t* bridge,
                                     const aesthetic_evaluation_t* eval,
                                     aesthetic_emotional_state_t* out) {
    if (!bridge || !eval || !out) return -1;

    memset(out, 0, sizeof(aesthetic_emotional_state_t));
    bridge->events_processed++;

    return 0;
}

int creative_emotion_bridge_extract(creative_emotion_bridge_t* bridge,
                                     const void* content,
                                     art_modality_t modality,
                                     aesthetic_emotional_response_t* out) {
    if (!bridge || !content || !out) return -1;

    memset(out, 0, sizeof(aesthetic_emotional_response_t));

    return 0;
}

void creative_emotion_bridge_set_emotion_system(creative_emotion_bridge_t* bridge,
                                                  void* emotion_system) {
    if (!bridge) return;
    bridge->emotion_system = emotion_system;
}

void creative_emotion_bridge_set_hippocampus(creative_emotion_bridge_t* bridge,
                                               void* hippocampus) {
    if (!bridge) return;
    bridge->hippocampus = hippocampus;
}

const aesthetic_emotional_state_t* creative_emotion_bridge_get_state(
    const creative_emotion_bridge_t* bridge) {
    return bridge ? &bridge->current_state : NULL;
}
