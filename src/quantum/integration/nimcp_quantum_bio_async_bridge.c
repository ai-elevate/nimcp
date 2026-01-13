/**
 * @file nimcp_quantum_bio_async_bridge.c
 * @brief Implementation of Quantum Module Bio-Async Integration Bridge
 * @version 1.0.0
 * @date 2026-01-13
 */

#include "quantum/integration/nimcp_quantum_bio_async_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ============================================================================
 * Internal State Cache
 * ============================================================================ */

typedef struct {
    float coherence_level;
    float min_coherence;
    uint32_t num_qubits;
    uint32_t errors_detected;
    uint32_t entangled_pairs;
} quantum_internal_state_t;

/* ============================================================================
 * Internal Bridge Structure
 * ============================================================================ */

struct quantum_bio_async_bridge_struct {
    quantum_bio_async_config_t config;
    bio_router_t router;

    quantum_bio_subscription_t* subscriptions;
    uint32_t subscription_count;
    uint32_t subscription_capacity;

    bool connected;
    uint64_t last_broadcast_us;
    uint32_t time_since_broadcast_ms;

    /* Cached state */
    quantum_internal_state_t state;

    quantum_bio_async_stats_t stats;
};

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static uint64_t quantum_get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static quantum_bio_subscription_t* quantum_find_subscription(
    quantum_bio_async_bridge_t* b, uint32_t module_id
) {
    for (uint32_t i = 0; i < b->subscription_count; i++) {
        if (b->subscriptions[i].module_id == module_id && b->subscriptions[i].active) {
            return &b->subscriptions[i];
        }
    }
    return NULL;
}

/* ============================================================================
 * Lifecycle API
 * ============================================================================ */

int quantum_bio_async_default_config(quantum_bio_async_config_t* config) {
    if (!config) return -1;

    config->state_broadcast_interval_ms = QUANTUM_BIO_DEFAULT_BROADCAST_INTERVAL_MS;
    config->enable_auto_broadcast = true;
    config->max_inbox_process_per_update = 32;
    config->message_ttl_ms = QUANTUM_BIO_MESSAGE_TTL_MS;
    config->default_channel = BIO_CHANNEL_ACETYLCHOLINE;
    config->error_channel = BIO_CHANNEL_NOREPINEPHRINE;
    config->coherence_warning_threshold = QUANTUM_BIO_COHERENCE_WARNING;
    config->coherence_critical_threshold = QUANTUM_BIO_COHERENCE_CRITICAL;
    config->max_subscriptions = QUANTUM_BIO_MAX_SUBSCRIPTIONS;
    config->enable_coherence_routing = true;
    config->enable_entanglement_routing = true;
    config->enable_measurement_routing = true;
    config->enable_error_routing = true;
    config->enable_logging = false;

    return 0;
}

quantum_bio_async_bridge_t* quantum_bio_async_bridge_create(
    const quantum_bio_async_config_t* config
) {
    quantum_bio_async_bridge_t* bridge = calloc(1, sizeof(quantum_bio_async_bridge_t));
    if (!bridge) return NULL;

    if (config) {
        bridge->config = *config;
    } else {
        quantum_bio_async_default_config(&bridge->config);
    }

    bridge->subscription_capacity = bridge->config.max_subscriptions;
    bridge->subscriptions = calloc(
        bridge->subscription_capacity, sizeof(quantum_bio_subscription_t)
    );
    if (!bridge->subscriptions) {
        free(bridge);
        return NULL;
    }

    bridge->last_broadcast_us = quantum_get_timestamp_us();
    bridge->state.coherence_level = 1.0f;
    bridge->state.min_coherence = 1.0f;
    bridge->stats.avg_coherence = 1.0f;
    bridge->stats.min_coherence = 1.0f;

    return bridge;
}

void quantum_bio_async_bridge_destroy(quantum_bio_async_bridge_t* bridge) {
    if (!bridge) return;

    if (bridge->connected) {
        quantum_bio_async_disconnect(bridge);
    }

    free(bridge->subscriptions);
    free(bridge);
}

/* ============================================================================
 * Connection API
 * ============================================================================ */

int quantum_bio_async_connect(
    quantum_bio_async_bridge_t* bridge,
    bio_router_t router
) {
    if (!bridge) return -1;

    bridge->router = router;
    bridge->connected = true;

    return 0;
}

int quantum_bio_async_disconnect(quantum_bio_async_bridge_t* bridge) {
    if (!bridge) return -1;

    bridge->router = NULL;
    bridge->connected = false;

    return 0;
}

bool quantum_bio_async_is_connected(const quantum_bio_async_bridge_t* bridge) {
    return bridge ? bridge->connected : false;
}

/* ============================================================================
 * Message Processing API
 * ============================================================================ */

int quantum_bio_async_process_inbox(
    quantum_bio_async_bridge_t* bridge,
    uint32_t max_messages
) {
    if (!bridge || !bridge->connected) return -1;

    uint32_t processed = 0;
    (void)max_messages;

    bridge->stats.messages_received += processed;
    return (int)processed;
}

int quantum_bio_async_update(
    quantum_bio_async_bridge_t* bridge,
    uint32_t delta_ms
) {
    if (!bridge || !bridge->connected) return -1;

    bridge->time_since_broadcast_ms += delta_ms;

    if (bridge->config.enable_auto_broadcast &&
        bridge->time_since_broadcast_ms >= bridge->config.state_broadcast_interval_ms) {
        /* Broadcast coherence as periodic update */
        quantum_bio_async_broadcast_coherence(
            bridge,
            bridge->state.coherence_level,
            bridge->state.num_qubits
        );
        bridge->time_since_broadcast_ms = 0;
    }

    return 0;
}

/* ============================================================================
 * Broadcast API
 * ============================================================================ */

int quantum_bio_async_broadcast_coherence(
    quantum_bio_async_bridge_t* bridge,
    float coherence_level,
    uint32_t num_qubits
) {
    if (!bridge || !bridge->connected) return -1;
    if (!bridge->config.enable_coherence_routing) return 0;

    quantum_bio_coherence_msg_t msg = {0};
    msg.header.type = BIO_MSG_ELIG_UQ_COHERENCE_UPDATE;
    msg.header.source_module = BIO_MODULE_ELIGIBILITY_QUANTUM;
    msg.header.channel = bridge->config.default_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.timestamp_us = quantum_get_timestamp_us();

    msg.coherence_level = coherence_level;
    msg.previous_coherence = bridge->state.coherence_level;
    msg.decoherence_rate = (bridge->state.coherence_level - coherence_level) * 1000.0f;
    msg.num_qubits = num_qubits;
    msg.avg_t1_time_us = 50.0f;
    msg.avg_t2_time_us = 30.0f;
    msg.is_warning = (coherence_level < bridge->config.coherence_warning_threshold);
    msg.is_critical = (coherence_level < bridge->config.coherence_critical_threshold);
    msg.timestamp_us = msg.header.timestamp_us;

    if (msg.is_critical) {
        msg.header.channel = bridge->config.error_channel;
        msg.header.flags |= BIO_MSG_FLAG_URGENT;
        bridge->stats.decoherence_warnings++;
    } else if (msg.is_warning) {
        bridge->stats.decoherence_warnings++;
    }

    /* Update internal state */
    bridge->state.coherence_level = coherence_level;
    bridge->state.num_qubits = num_qubits;
    if (coherence_level < bridge->state.min_coherence) {
        bridge->state.min_coherence = coherence_level;
    }

    /* Update stats */
    bridge->stats.avg_coherence = (bridge->stats.avg_coherence * 0.99f) + (coherence_level * 0.01f);
    bridge->stats.min_coherence = bridge->state.min_coherence;

    bridge->stats.coherence_updates_sent++;
    bridge->stats.broadcasts_sent++;
    bridge->last_broadcast_us = msg.timestamp_us;

    return 0;
}

int quantum_bio_async_broadcast_entanglement(
    quantum_bio_async_bridge_t* bridge,
    uint32_t qubit_a,
    uint32_t qubit_b,
    float fidelity,
    bool is_creation
) {
    if (!bridge || !bridge->connected) return -1;
    if (!bridge->config.enable_entanglement_routing) return 0;

    quantum_bio_entanglement_msg_t msg = {0};
    msg.header.type = BIO_MSG_ELIG_UQ_COHERENCE_UPDATE;
    msg.header.source_module = BIO_MODULE_ELIGIBILITY_QUANTUM;
    msg.header.channel = bridge->config.default_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.timestamp_us = quantum_get_timestamp_us();

    msg.qubit_a = qubit_a;
    msg.qubit_b = qubit_b;
    msg.entanglement_fidelity = fidelity;
    msg.bell_state = 0; /* Phi+ by default */
    msg.concurrence = fidelity;
    msg.is_creation = is_creation;
    msg.timestamp_us = msg.header.timestamp_us;

    if (is_creation) {
        bridge->state.entangled_pairs++;
    } else {
        if (bridge->state.entangled_pairs > 0) {
            bridge->state.entangled_pairs--;
        }
    }

    bridge->stats.entanglement_events_sent++;
    bridge->stats.broadcasts_sent++;

    return 0;
}

int quantum_bio_async_broadcast_measurement(
    quantum_bio_async_bridge_t* bridge,
    uint32_t qubit_id,
    int32_t outcome,
    float probability
) {
    if (!bridge || !bridge->connected) return -1;
    if (!bridge->config.enable_measurement_routing) return 0;

    quantum_bio_measurement_msg_t msg = {0};
    msg.header.type = BIO_MSG_ELIG_UQ_COHERENCE_UPDATE;
    msg.header.source_module = BIO_MODULE_ELIGIBILITY_QUANTUM;
    msg.header.channel = bridge->config.default_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.timestamp_us = quantum_get_timestamp_us();

    msg.qubit_id = qubit_id;
    msg.basis = 0; /* Z basis by default */
    msg.outcome = outcome;
    msg.probability = probability;
    msg.confidence = 0.95f;
    msg.timestamp_us = msg.header.timestamp_us;

    bridge->stats.measurements_sent++;
    bridge->stats.broadcasts_sent++;

    return 0;
}

int quantum_bio_async_broadcast_walk_update(
    quantum_bio_async_bridge_t* bridge,
    uint32_t walk_id,
    uint32_t step_number,
    float amplitude_at_target
) {
    if (!bridge || !bridge->connected) return -1;

    quantum_bio_walk_msg_t msg = {0};
    msg.header.type = BIO_MSG_ELIG_QUANTUM_WALK_DIFFUSION;
    msg.header.source_module = BIO_MODULE_ELIGIBILITY_QUANTUM;
    msg.header.channel = bridge->config.default_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.timestamp_us = quantum_get_timestamp_us();

    msg.walk_id = walk_id;
    msg.step_number = step_number;
    msg.amplitude_at_target = amplitude_at_target;
    msg.target_found = (amplitude_at_target > 0.5f);
    msg.timestamp_us = msg.header.timestamp_us;

    bridge->stats.walk_updates_sent++;
    bridge->stats.broadcasts_sent++;

    return 0;
}

int quantum_bio_async_broadcast_annealing(
    quantum_bio_async_bridge_t* bridge,
    uint32_t annealing_id,
    uint32_t step_number,
    float temperature,
    float current_energy
) {
    if (!bridge || !bridge->connected) return -1;

    quantum_bio_annealing_msg_t msg = {0};
    msg.header.type = BIO_MSG_ELIG_QUANTUM_ANNEAL_STATE;
    msg.header.source_module = BIO_MODULE_ELIGIBILITY_QUANTUM;
    msg.header.channel = bridge->config.default_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.timestamp_us = quantum_get_timestamp_us();

    msg.annealing_id = annealing_id;
    msg.step_number = step_number;
    msg.temperature = temperature;
    msg.current_energy = current_energy;
    msg.is_complete = (temperature < 0.01f);
    msg.timestamp_us = msg.header.timestamp_us;

    bridge->stats.broadcasts_sent++;

    return 0;
}

int quantum_bio_async_broadcast_gate(
    quantum_bio_async_bridge_t* bridge,
    uint32_t gate_type,
    uint32_t target_qubit,
    float gate_fidelity
) {
    if (!bridge || !bridge->connected) return -1;

    quantum_bio_gate_msg_t msg = {0};
    msg.header.type = BIO_MSG_ELIG_UQ_COHERENCE_UPDATE;
    msg.header.source_module = BIO_MODULE_ELIGIBILITY_QUANTUM;
    msg.header.channel = bridge->config.default_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.timestamp_us = quantum_get_timestamp_us();

    msg.gate_type = gate_type;
    msg.target_qubit = target_qubit;
    msg.gate_fidelity = gate_fidelity;
    msg.gate_time_us = 0.1f; /* Typical single-qubit gate */
    msg.timestamp_us = msg.header.timestamp_us;

    bridge->stats.broadcasts_sent++;

    return 0;
}

int quantum_bio_async_broadcast_error(
    quantum_bio_async_bridge_t* bridge,
    uint32_t error_type,
    uint32_t qubit_id,
    bool is_correctable
) {
    if (!bridge || !bridge->connected) return -1;
    if (!bridge->config.enable_error_routing) return 0;

    quantum_bio_error_msg_t msg = {0};
    msg.header.type = BIO_MSG_ELIG_UQ_COHERENCE_UPDATE;
    msg.header.source_module = BIO_MODULE_ELIGIBILITY_QUANTUM;
    msg.header.channel = bridge->config.error_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST | BIO_MSG_FLAG_URGENT;
    msg.header.timestamp_us = quantum_get_timestamp_us();

    msg.error_type = error_type;
    msg.qubit_id = qubit_id;
    msg.is_correctable = is_correctable;
    msg.was_corrected = false;
    msg.error_count = ++bridge->state.errors_detected;
    msg.timestamp_us = msg.header.timestamp_us;

    bridge->stats.errors_detected++;
    bridge->stats.broadcasts_sent++;

    return 0;
}

int quantum_bio_async_broadcast_amplitude(
    quantum_bio_async_bridge_t* bridge,
    uint32_t target_state,
    float estimated_amplitude,
    float variance
) {
    if (!bridge || !bridge->connected) return -1;

    quantum_bio_amplitude_msg_t msg = {0};
    msg.header.type = BIO_MSG_ELIG_QUANTUM_CREDIT_ASSIGNED;
    msg.header.source_module = BIO_MODULE_ELIGIBILITY_QUANTUM;
    msg.header.channel = bridge->config.default_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;
    msg.header.timestamp_us = quantum_get_timestamp_us();

    msg.target_state = target_state;
    msg.estimated_amplitude = estimated_amplitude;
    msg.estimated_probability = estimated_amplitude * estimated_amplitude;
    msg.variance = variance;
    msg.std_error = (variance > 0.0f) ? sqrtf(variance) : 0.0f;
    msg.timestamp_us = msg.header.timestamp_us;

    bridge->stats.broadcasts_sent++;

    return 0;
}

/* ============================================================================
 * Subscription Management API
 * ============================================================================ */

int quantum_bio_async_subscribe_module(
    quantum_bio_async_bridge_t* bridge,
    uint32_t module_id,
    uint32_t msg_types
) {
    if (!bridge) return -1;

    quantum_bio_subscription_t* existing = quantum_find_subscription(bridge, module_id);
    if (existing) {
        existing->msg_type_mask |= msg_types;
        return 0;
    }

    if (bridge->subscription_count >= bridge->subscription_capacity) {
        return -1;
    }

    quantum_bio_subscription_t* sub = &bridge->subscriptions[bridge->subscription_count++];
    sub->module_id = module_id;
    sub->msg_type_mask = msg_types;
    sub->active = true;
    sub->subscription_time = quantum_get_timestamp_us();
    sub->messages_sent = 0;

    bridge->stats.active_subscriptions = bridge->subscription_count;
    if (bridge->subscription_count > bridge->stats.peak_subscriptions) {
        bridge->stats.peak_subscriptions = bridge->subscription_count;
    }

    return 0;
}

int quantum_bio_async_unsubscribe_module(
    quantum_bio_async_bridge_t* bridge,
    uint32_t module_id
) {
    if (!bridge) return -1;

    for (uint32_t i = 0; i < bridge->subscription_count; i++) {
        if (bridge->subscriptions[i].module_id == module_id) {
            bridge->subscriptions[i].active = false;
            bridge->stats.active_subscriptions--;
            return 0;
        }
    }

    return -1;
}

uint32_t quantum_bio_async_get_subscriber_count(
    const quantum_bio_async_bridge_t* bridge,
    quantum_bio_msg_type_t msg_type
) {
    if (!bridge) return 0;

    uint32_t count = 0;
    uint32_t mask = (1U << msg_type);

    for (uint32_t i = 0; i < bridge->subscription_count; i++) {
        if (bridge->subscriptions[i].active &&
            (bridge->subscriptions[i].msg_type_mask & mask)) {
            count++;
        }
    }

    return count;
}

/* ============================================================================
 * Statistics and Diagnostics API
 * ============================================================================ */

int quantum_bio_async_get_stats(
    const quantum_bio_async_bridge_t* bridge,
    quantum_bio_async_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

int quantum_bio_async_reset_stats(quantum_bio_async_bridge_t* bridge) {
    if (!bridge) return -1;

    uint32_t active = bridge->stats.active_subscriptions;
    uint32_t peak = bridge->stats.peak_subscriptions;
    memset(&bridge->stats, 0, sizeof(bridge->stats));
    bridge->stats.active_subscriptions = active;
    bridge->stats.peak_subscriptions = peak;
    bridge->stats.avg_coherence = 1.0f;
    bridge->stats.min_coherence = 1.0f;

    return 0;
}

static const char* quantum_bio_msg_type_names[] = {
    "COHERENCE_UPDATE", "ENTANGLEMENT", "MEASUREMENT",
    "WALK_UPDATE", "ANNEALING_STEP", "GATE_APPLIED",
    "ERROR_DETECTED", "ERROR_CORRECTED", "STATE_PREPARED",
    "AMPLITUDE_ESTIMATE", "REQUEST_QUBITS", "QUERY_STATE"
};

const char* quantum_bio_msg_type_name(quantum_bio_msg_type_t msg_type) {
    if (msg_type >= QUANTUM_BIO_MSG_COUNT) return "UNKNOWN";
    return quantum_bio_msg_type_names[msg_type];
}

void quantum_bio_async_print_summary(const quantum_bio_async_bridge_t* bridge) {
    if (!bridge) {
        printf("Quantum Bio-Async Bridge: NULL\n");
        return;
    }

    printf("Quantum Bio-Async Bridge Summary:\n");
    printf("  Connected: %s\n", bridge->connected ? "yes" : "no");
    printf("  Coherence: %.3f (min: %.3f, avg: %.3f)\n",
           bridge->state.coherence_level,
           bridge->stats.min_coherence,
           bridge->stats.avg_coherence);
    printf("  Qubits: %u, Entangled pairs: %u\n",
           bridge->state.num_qubits,
           bridge->state.entangled_pairs);
    printf("  Subscriptions: %u (peak: %u)\n",
           bridge->stats.active_subscriptions,
           bridge->stats.peak_subscriptions);
    printf("  Messages sent: %lu, received: %lu\n",
           (unsigned long)bridge->stats.messages_sent,
           (unsigned long)bridge->stats.messages_received);
    printf("  Broadcasts: %lu (coherence: %lu, entangle: %lu, measure: %lu)\n",
           (unsigned long)bridge->stats.broadcasts_sent,
           (unsigned long)bridge->stats.coherence_updates_sent,
           (unsigned long)bridge->stats.entanglement_events_sent,
           (unsigned long)bridge->stats.measurements_sent);
    printf("  Errors: detected=%lu, decoherence_warnings=%lu\n",
           (unsigned long)bridge->stats.errors_detected,
           (unsigned long)bridge->stats.decoherence_warnings);
    printf("  Routing errors: handler=%lu, routing=%lu\n",
           (unsigned long)bridge->stats.handler_errors,
           (unsigned long)bridge->stats.routing_errors);
}

/* For sqrtf if not already included via math.h */
#include <math.h>
