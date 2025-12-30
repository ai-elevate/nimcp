/**
 * @file nimcp_knowledge_thalamic_bridge.c
 * @brief Knowledge-Thalamic Bridge Implementation
 */

#include "cognitive/knowledge/nimcp_knowledge_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct knowledge_thalamic_bridge {
    void* knowledge;
    thalamic_router_t* router;
    knowledge_thalamic_config_t config;
    knowledge_thalamic_stats_t stats;
    float attention_weight;
};

knowledge_thalamic_config_t knowledge_thalamic_default_config(void) {
    knowledge_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_relevance_boost = true,
        .min_retrieval_strength = 0.3f,
        .inference_threshold = 0.5f
    };
    return cfg;
}

knowledge_thalamic_bridge_t* knowledge_thalamic_bridge_create(void* knowledge, thalamic_router_t* router, const knowledge_thalamic_config_t* config) {
    knowledge_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(knowledge_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->knowledge = knowledge;
    bridge->router = router;
    bridge->config = config ? *config : knowledge_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void knowledge_thalamic_bridge_destroy(knowledge_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int knowledge_thalamic_bridge_reset(knowledge_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int knowledge_thalamic_route_retrieval(knowledge_thalamic_bridge_t* bridge, const knowledge_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->retrieval_strength < bridge->config.min_retrieval_strength) {
        return 0;
    }
    bridge->stats.retrievals_routed++;
    bridge->stats.avg_retrieval_strength = (bridge->stats.avg_retrieval_strength * (bridge->stats.retrievals_routed - 1) +
                                            signal->retrieval_strength) / bridge->stats.retrievals_routed;
    return 0;
}

int knowledge_thalamic_route_inference(knowledge_thalamic_bridge_t* bridge, const void* inference, float confidence) {
    if (!bridge) return -1;
    if (confidence < bridge->config.inference_threshold) return 0;
    bridge->stats.inferences_routed++;
    return 0;
}

int knowledge_thalamic_set_attention(knowledge_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int knowledge_thalamic_get_attention(const knowledge_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int knowledge_thalamic_bridge_get_stats(const knowledge_thalamic_bridge_t* bridge, knowledge_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
