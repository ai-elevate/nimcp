/**
 * @file nimcp_theory_of_mind_thalamic_bridge.c
 * @brief Theory of Mind-Thalamic Bridge Implementation
 */

#include "cognitive/theory_of_mind/nimcp_theory_of_mind_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct tom_thalamic_bridge {
    void* tom;
    thalamic_router_t* router;
    tom_thalamic_config_t config;
    tom_thalamic_stats_t stats;
    float attention_weight;
};

tom_thalamic_config_t tom_thalamic_default_config(void) {
    tom_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_salience_boost = true,
        .min_salience_threshold = 0.25f,
        .deception_boost = 0.5f  /* Deception signals get strong priority */
    };
    return cfg;
}

tom_thalamic_bridge_t* tom_thalamic_bridge_create(void* tom, thalamic_router_t* router, const tom_thalamic_config_t* config) {
    tom_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(tom_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->tom = tom;
    bridge->router = router;
    bridge->config = config ? *config : tom_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void tom_thalamic_bridge_destroy(tom_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int tom_thalamic_bridge_reset(tom_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int tom_thalamic_route_inference(tom_thalamic_bridge_t* bridge, const tom_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    /* Deception signals bypass attention gating */
    if (bridge->config.enable_attention_gating &&
        signal->social_salience < bridge->config.min_salience_threshold &&
        signal->signal_type != TOM_SIGNAL_DECEPTION) {
        return 0;
    }
    bridge->stats.inferences_routed++;
    bridge->stats.avg_confidence = (bridge->stats.avg_confidence * (bridge->stats.inferences_routed - 1) +
                                    signal->confidence) / bridge->stats.inferences_routed;
    if (signal->signal_type == TOM_SIGNAL_DECEPTION) {
        bridge->stats.deceptions_detected++;
    }
    return 0;
}

int tom_thalamic_route_prediction(tom_thalamic_bridge_t* bridge, const void* prediction, float confidence) {
    if (!bridge) return -1;
    bridge->stats.predictions_made++;
    return 0;
}

int tom_thalamic_set_attention(tom_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int tom_thalamic_get_attention(const tom_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int tom_thalamic_bridge_get_stats(const tom_thalamic_bridge_t* bridge, tom_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
