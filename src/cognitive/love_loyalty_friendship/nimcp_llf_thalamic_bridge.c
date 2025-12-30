/**
 * @file nimcp_llf_thalamic_bridge.c
 * @brief Love/Loyalty/Friendship-Thalamic Bridge Implementation
 */

#include "cognitive/love_loyalty_friendship/nimcp_llf_thalamic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>

struct llf_thalamic_bridge {
    void* llf;
    thalamic_router_t* router;
    llf_thalamic_config_t config;
    llf_thalamic_stats_t stats;
    float attention_weight;
};

llf_thalamic_config_t llf_thalamic_default_config(void) {
    llf_thalamic_config_t cfg = {
        .enable_attention_gating = true,
        .enable_bond_strengthening = true,
        .min_bond_threshold = 0.3f,
        .trust_threshold = 0.5f
    };
    return cfg;
}

llf_thalamic_bridge_t* llf_thalamic_bridge_create(void* llf, thalamic_router_t* router, const llf_thalamic_config_t* config) {
    llf_thalamic_bridge_t* bridge = nimcp_calloc(1, sizeof(llf_thalamic_bridge_t));
    if (!bridge) return NULL;
    bridge->llf = llf;
    bridge->router = router;
    bridge->config = config ? *config : llf_thalamic_default_config();
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return bridge;
}

void llf_thalamic_bridge_destroy(llf_thalamic_bridge_t* bridge) {
    if (bridge) nimcp_free(bridge);
}

int llf_thalamic_bridge_reset(llf_thalamic_bridge_t* bridge) {
    if (!bridge) return -1;
    bridge->attention_weight = 1.0f;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    return 0;
}

int llf_thalamic_route_attachment(llf_thalamic_bridge_t* bridge, const llf_thalamic_signal_t* signal) {
    if (!bridge || !signal) return -1;
    if (bridge->config.enable_attention_gating && signal->bond_strength < bridge->config.min_bond_threshold) {
        return 0;
    }
    bridge->stats.attachments_routed++;
    bridge->stats.avg_bond_strength = (bridge->stats.avg_bond_strength * (bridge->stats.attachments_routed - 1) +
                                       signal->bond_strength) / bridge->stats.attachments_routed;
    if (signal->signal_type == LLF_SIGNAL_TRUST) {
        bridge->stats.trust_updates++;
    }
    return 0;
}

int llf_thalamic_route_care(llf_thalamic_bridge_t* bridge, const void* target, float motivation) {
    if (!bridge) return -1;
    bridge->stats.care_expressions++;
    return 0;
}

int llf_thalamic_set_attention(llf_thalamic_bridge_t* bridge, float attention) {
    if (!bridge) return -1;
    bridge->attention_weight = attention < 0.0f ? 0.0f : (attention > 1.0f ? 1.0f : attention);
    return 0;
}

int llf_thalamic_get_attention(const llf_thalamic_bridge_t* bridge, float* attention) {
    if (!bridge || !attention) return -1;
    *attention = bridge->attention_weight;
    return 0;
}

int llf_thalamic_bridge_get_stats(const llf_thalamic_bridge_t* bridge, llf_thalamic_stats_t* stats) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}
