//=============================================================================
// nimcp_thalamic_router.c - Attention-Gated Neural Routing
//=============================================================================

#include "middleware/routing/nimcp_thalamic_router.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "middleware/routing/nimcp_attention_gate.h"
#include "middleware/routing/nimcp_routing_table.h"
#include "middleware/routing/nimcp_signal_wrapper.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/tensor/nimcp_tensor.h"
#include <string.h>
#include <stdio.h>
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "plasticity/nimcp_second_messengers.h"
#include "plasticity/neuromodulators/nimcp_neuromodulators.h"

/* Version 1.1.0 - Added tensor library integration for vectorized signal modulation */



#define LOG_MODULE "nimcp_thalamic_router"
#define LOG_MODULE_ID 0x052C

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

typedef struct {
    signal_wrapper_t wrapper;  // CoW-based signal reference (zero-copy)
    uint32_t source_id;
    float attention_weight;
    signal_priority_t priority;
    uint64_t timestamp_ms;
    double enqueue_time_ms;
} queued_signal_t;

typedef struct {
    signal_delivery_callback_t callback;
    void* user_data;
} delivery_registration_t;

#define MAX_DESTINATIONS 256
#define MAX_QUEUE 1000

struct thalamic_router {
    // Configuration
    thalamic_router_config_t config;

    // Routing components
    attention_gate_t* attention_gate;
    routing_table_t* routing_table;

    // Signal queue (priority queue)
    queued_signal_t* queue;
    uint32_t queue_capacity;
    uint32_t queue_size;

    // Delivery callbacks
    delivery_registration_t* callbacks;
    uint32_t num_callbacks;

    // Statistics
    routing_stats_t stats;
    double last_process_time_ms;
    uint64_t total_signal_count;

    // Bio-async integration
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;

    // Second messenger cascades
    second_messenger_system_t* second_messengers;
    bool second_messengers_enabled;
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static double get_current_time_ms(void) {
    // WHAT: Get current wall-clock time in milliseconds
    // WHY:  Track signal timing and latency
    // HOW:  Use NIMCP time utility for high-precision timing
    return (double)nimcp_time_get_ms();
}

static void init_stats(routing_stats_t* stats) {
    stats->signals_routed = 0;
    stats->signals_dropped = 0;
    stats->signals_bypassed = 0;
    stats->avg_latency_ms = 0.0F;
    stats->throughput_hz = 0.0F;
    stats->queue_depth = 0;
}

static bool enqueue_signal(thalamic_router_t* router, const routed_signal_t* signal) {
    if (router->queue_size >= router->queue_capacity) {
        router->stats.signals_dropped++;
        return false;
    }

    queued_signal_t* qs = &router->queue[router->queue_size];

    // Zero-copy signal wrapper (CoW-based, ~30ns vs 1500ns deep copy)
    qs->wrapper = signal_wrapper_create(
        signal->dest_ids, signal->num_dests,
        signal->signal_data, signal->signal_size);

    if (!qs->wrapper) return false;

    // Copy metadata (small, fixed-size)
    qs->source_id = signal->source_id;
    qs->attention_weight = signal->attention_weight;
    qs->priority = signal->priority;
    qs->timestamp_ms = signal->timestamp_ms;
    qs->enqueue_time_ms = get_current_time_ms();

    router->queue_size++;
    router->stats.queue_depth = router->queue_size;

    // Sort by priority (simple insertion sort)
    for (uint32_t i = router->queue_size - 1; i > 0; i--) {
        if (router->queue[i].priority > router->queue[i-1].priority) {
            queued_signal_t temp = router->queue[i];
            router->queue[i] = router->queue[i-1];
            router->queue[i-1] = temp;
        } else {
            break;
        }
    }

    return true;
}

// Helper: Get PKA-modulated attention threshold
static float get_attention_threshold(thalamic_router_t* router, uint32_t dest_id) {
    // WHAT: Compute PKA-modulated attention gating threshold
    // WHY:  ACh/NE increase PKA -> lower threshold -> more signals pass
    // HOW:  Higher PKA activity reduces threshold (inverse relationship)

    float base_threshold = router->config.min_attention_threshold;

    if (!router->second_messengers_enabled || !router->second_messengers) {
        return base_threshold;
    }

    // Query PKA activity for destination neuron
    second_messenger_state_t sm_state;
    nimcp_result_t result = second_messenger_get_state(
        router->second_messengers, dest_id, &sm_state);

    if (result != NIMCP_SUCCESS) {
        return base_threshold;
    }

    float pka_activity = sm_state.camp.pka_activity;

    // Modulation: PKA [0,1] -> threshold reduction [1.0, 0.1]
    // High PKA = low threshold = more attention
    float modulation = 1.0F - (0.9F * pka_activity);

    return base_threshold * modulation;
}

// Helper: Deliver signal using CoW wrapper
static bool deliver_signal_wrapper(thalamic_router_t* router,
                                     signal_wrapper_t wrapper,
                                     uint32_t source_id,
                                     float attention_weight) {
    bool delivered = false;

    // Read destinations and signal data (zero-copy)
    uint32_t num_dests = 0;
    const uint32_t* dest_ids = signal_wrapper_read_destinations(wrapper, &num_dests);

    uint32_t signal_size = 0;
    const float* signal_data = signal_wrapper_read_data(wrapper, &signal_size);

    if (!dest_ids || !signal_data) return false;

    for (uint32_t i = 0; i < num_dests; i++) {
        uint32_t dest_id = dest_ids[i];

        // Find callback for destination
        signal_delivery_callback_t callback = NULL;
        void* user_data = NULL;

        for (uint32_t j = 0; j < router->num_callbacks; j++) {
            if (j == dest_id) {
                callback = router->callbacks[j].callback;
                user_data = router->callbacks[j].user_data;
                break;
            }
        }

        if (callback) {
            // Apply attention gating if enabled
            float attention = attention_weight;

            if (router->config.enable_attention_gating && router->attention_gate) {
                float gate_weight = 1.0F;
                attention_gate_get_weight(router->attention_gate, source_id,
                                        dest_id, &gate_weight);
                attention *= gate_weight;
            }

            // Check PKA-modulated attention threshold
            float attention_threshold = get_attention_threshold(router, dest_id);
            if (attention >= attention_threshold) {
                // Apply attention to signal using tensor operations
                float* modulated_signal = (float*)nimcp_malloc(signal_size * sizeof(float));
                if (modulated_signal) {
                    memcpy(modulated_signal, signal_data, signal_size * sizeof(float));

                    /* Use tensor library for vectorized scalar multiplication */
                    uint32_t dims[] = {signal_size};
                    nimcp_tensor_t* t = nimcp_tensor_from_data(modulated_signal, dims, 1,
                                                              NIMCP_DTYPE_F32, false);
                    if (t) {
                        nimcp_tensor_mul_scalar_(t, (double)attention);
                        nimcp_tensor_destroy(t);
                    } else {
                        /* Fallback to scalar loop */
                        for (uint32_t k = 0; k < signal_size; k++) {
                            modulated_signal[k] *= attention;
                        }
                    }

                    callback(dest_id, modulated_signal, signal_size,
                           attention, user_data);

                    nimcp_free(modulated_signal);
                    delivered = true;
                }
            }
        }
    }

    return delivered;
}

static bool deliver_signal(thalamic_router_t* router, const routed_signal_t* signal) {
    // Create temporary wrapper for delivery (will be released immediately)
    signal_wrapper_t wrapper = signal_wrapper_create(
        signal->dest_ids, signal->num_dests,
        signal->signal_data, signal->signal_size);

    if (!wrapper) return false;

    bool delivered = deliver_signal_wrapper(router, wrapper,
                                           signal->source_id,
                                           signal->attention_weight);

    signal_wrapper_release(wrapper);
    return delivered;
}

// ============================================================================
// PUBLIC API
// ============================================================================

thalamic_router_config_t thalamic_router_default_config(void) {
    thalamic_router_config_t config;
    config.max_queue_size = THALAMIC_MAX_QUEUE_SIZE;
    config.max_destinations = THALAMIC_MAX_DESTINATIONS;
    config.enable_attention_gating = true;
    config.enable_priority_routing = true;
    config.enable_statistics = true;
    config.min_attention_threshold = 0.01F;
    config.enable_learning = true;
    config.enable_second_messengers = true;  /* Enable by default - biological fidelity */
    config.num_neurons = 0;
    return config;
}

thalamic_router_t* thalamic_router_create(const thalamic_router_config_t* config) {
    if (!config) return NULL;

    thalamic_router_t* router = (thalamic_router_t*)nimcp_calloc(1, sizeof(thalamic_router_t));
    if (!router) return NULL;

    router->config = *config;

    // Create attention gate
    if (config->enable_attention_gating) {
        attention_gate_config_t gate_config = attention_gate_default_config();
        router->attention_gate = attention_gate_create(&gate_config);
        if (!router->attention_gate) {
            thalamic_router_destroy(router);
            return NULL;
        }
    }

    // Create routing table
    routing_table_config_t table_config = routing_table_default_config();
    router->routing_table = routing_table_create(&table_config);
    if (!router->routing_table) {
        thalamic_router_destroy(router);
        return NULL;
    }

    // Allocate queue
    router->queue_capacity = config->max_queue_size;
    router->queue = (queued_signal_t*)nimcp_calloc(router->queue_capacity,
                                            sizeof(queued_signal_t));
    if (!router->queue) {
        thalamic_router_destroy(router);
        return NULL;
    }

    router->queue_size = 0;

    // Allocate callback storage
    router->callbacks = (delivery_registration_t*)nimcp_calloc(MAX_DESTINATIONS,
                                                         sizeof(delivery_registration_t));
    if (!router->callbacks) {
        thalamic_router_destroy(router);
        return NULL;
    }

    router->num_callbacks = 0;

    // Initialize statistics
    init_stats(&router->stats);
    router->last_process_time_ms = get_current_time_ms();
    router->total_signal_count = 0;

    // Register with bio-async router
    router->bio_ctx = NULL;
    router->bio_async_enabled = false;
    if (bio_router_is_initialized()) {
        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_SIGNAL_ROUTER,
            .module_name = "thalamic_router",
            .inbox_capacity = 64,
            .user_data = router
        };
        router->bio_ctx = bio_router_register_module(&bio_info);
        if (router->bio_ctx) {
            router->bio_async_enabled = true;
            LOG_DEBUG(LOG_MODULE, "Registered with bio-async router");
        }
    }

    // Initialize second messenger cascades
    router->second_messengers = NULL;
    router->second_messengers_enabled = false;
    if (config->enable_second_messengers && config->num_neurons > 0) {
        second_messenger_config_t sm_config = second_messenger_default_config();
        sm_config.enable_bio_async = true;
        sm_config.enable_security = true;

        router->second_messengers = second_messenger_create(config->num_neurons, &sm_config);
        if (router->second_messengers) {
            router->second_messengers_enabled = true;
            LOG_INFO(LOG_MODULE, "Initialized second messenger cascades for %u neurons",
                     config->num_neurons);
        } else {
            LOG_WARN(LOG_MODULE, "Failed to initialize second messenger cascades");
        }
    }

    return router;
}

void thalamic_router_destroy(thalamic_router_t* router) {
    if (!router) return;

    // Destroy second messenger system
    if (router->second_messengers_enabled && router->second_messengers) {
        second_messenger_destroy(router->second_messengers);
        router->second_messengers = NULL;
        router->second_messengers_enabled = false;
    }

    // Unregister from bio-async router
    if (router->bio_async_enabled && router->bio_ctx) {
        bio_router_unregister_module(router->bio_ctx);
        router->bio_ctx = NULL;
        router->bio_async_enabled = false;
    }

    // Release queued signal wrappers
    if (router->queue) {
        for (uint32_t i = 0; i < router->queue_size; i++) {
            signal_wrapper_release(router->queue[i].wrapper);
        }
        nimcp_free(router->queue);
    }

    nimcp_free(router->callbacks);

    attention_gate_destroy(router->attention_gate);
    routing_table_destroy(router->routing_table);

    nimcp_free(router);
}

bool thalamic_router_route_signal(thalamic_router_t* router,
                                   const routed_signal_t* signal) {
    if (!router || !signal || !signal->dest_ids || signal->num_dests == 0) {
        return false;
    }

    router->total_signal_count++;

    // High priority or bypass flag: deliver immediately
    if (signal->bypass_queue ||
        (router->config.enable_priority_routing && signal->priority == PRIORITY_HIGH)) {

        bool delivered = deliver_signal(router, signal);

        if (delivered) {
            router->stats.signals_routed++;
            router->stats.signals_bypassed++;

            // Update routing table (Hebbian learning)
            if (router->config.enable_learning) {
                for (uint32_t i = 0; i < signal->num_dests; i++) {
                    routing_table_add_route(router->routing_table,
                                          signal->source_id,
                                          signal->dest_ids[i],
                                          1.0F);  // Strengthen route
                }
            }
        }

        return delivered;
    }

    // Normal/low priority: enqueue
    bool enqueued = enqueue_signal(router, signal);

    if (enqueued && router->config.enable_learning) {
        // Register route in table
        for (uint32_t i = 0; i < signal->num_dests; i++) {
            routing_table_add_route(router->routing_table,
                                  signal->source_id,
                                  signal->dest_ids[i],
                                  0.5F);
        }
    }

    return enqueued;
}

bool thalamic_router_process_queue(thalamic_router_t* router,
                                    uint32_t max_signals,
                                    uint32_t* num_processed) {
    if (!router) return false;

    // Process bio-async messages first
    if (router->bio_async_enabled && router->bio_ctx) {
        bio_router_process_inbox(router->bio_ctx, 8);
    }

    // Update second messenger cascades
    double current_time = get_current_time_ms();
    if (router->second_messengers_enabled && router->second_messengers) {
        double dt_ms = current_time - router->last_process_time_ms;
        if (dt_ms > 0.0) {
            second_messenger_update(router->second_messengers, (float)dt_ms,
                                  (uint64_t)current_time);
        }
    }

    uint32_t processed = 0;

    while (processed < max_signals && router->queue_size > 0) {
        // Dequeue highest priority (front of queue)
        queued_signal_t* qs = &router->queue[0];

        // Deliver using CoW wrapper (zero-copy)
        bool delivered = deliver_signal_wrapper(router, qs->wrapper,
                                               qs->source_id,
                                               qs->attention_weight);

        if (delivered) {
            router->stats.signals_routed++;

            // Update latency statistics
            double latency = current_time - qs->enqueue_time_ms;
            float alpha = 0.1F;
            router->stats.avg_latency_ms =
                (1.0F - alpha) * router->stats.avg_latency_ms + alpha * (float)latency;
        }

        // Release wrapper (decrements refcount, frees if last reference)
        signal_wrapper_release(qs->wrapper);

        // Shift queue
        for (uint32_t i = 0; i < router->queue_size - 1; i++) {
            router->queue[i] = router->queue[i + 1];
        }

        router->queue_size--;
        processed++;
    }

    router->stats.queue_depth = router->queue_size;

    // Update throughput
    double elapsed = current_time - router->last_process_time_ms;
    if (elapsed > 0.0) {
        router->stats.throughput_hz = (float)processed / (float)(elapsed / 1000.0);
    }
    router->last_process_time_ms = current_time;

    if (num_processed) *num_processed = processed;

    return true;
}

bool thalamic_router_set_callback(thalamic_router_t* router,
                                   uint32_t dest_id,
                                   signal_delivery_callback_t callback,
                                   void* user_data) {
    if (!router || dest_id >= MAX_DESTINATIONS) {
        return false;
    }

    // Extend callbacks array if needed
    if (dest_id >= router->num_callbacks) {
        router->num_callbacks = dest_id + 1;
    }

    router->callbacks[dest_id].callback = callback;
    router->callbacks[dest_id].user_data = user_data;

    return true;
}

bool thalamic_router_set_attention(thalamic_router_t* router,
                                    uint32_t source_id,
                                    uint32_t dest_id,
                                    float attention) {
    if (!router || !router->attention_gate) {
        return false;
    }

    return attention_gate_set_weight(router->attention_gate,
                                    source_id, dest_id, attention);
}

bool thalamic_router_get_attention(const thalamic_router_t* router,
                                    uint32_t source_id,
                                    uint32_t dest_id,
                                    float* attention) {
    if (!router || !router->attention_gate || !attention) {
        return false;
    }

    return attention_gate_get_weight(router->attention_gate,
                                    source_id, dest_id, attention);
}

bool thalamic_router_get_stats(const thalamic_router_t* router,
                                routing_stats_t* stats) {
    if (!router || !stats) return false;

    *stats = router->stats;
    return true;
}

void thalamic_router_reset_stats(thalamic_router_t* router) {
    if (!router) return;
    init_stats(&router->stats);
}

void thalamic_router_clear_queue(thalamic_router_t* router) {
    if (!router) return;

    // Release all queued wrappers
    for (uint32_t i = 0; i < router->queue_size; i++) {
        signal_wrapper_release(router->queue[i].wrapper);
    }

    router->queue_size = 0;
    router->stats.queue_depth = 0;
}

routed_signal_t* thalamic_router_create_signal(uint32_t source_id,
                                                const uint32_t* dest_ids,
                                                uint32_t num_dests,
                                                const float* signal_data,
                                                uint32_t signal_size,
                                                signal_priority_t priority) {
    if (!dest_ids || num_dests == 0 || !signal_data || signal_size == 0) {
        return NULL;
    }

    routed_signal_t* signal = (routed_signal_t*)nimcp_calloc(1, sizeof(routed_signal_t));
    if (!signal) return NULL;

    signal->dest_ids = (uint32_t*)nimcp_malloc(num_dests * sizeof(uint32_t));
    signal->signal_data = (float*)nimcp_malloc(signal_size * sizeof(float));

    if (!signal->dest_ids || !signal->signal_data) {
        thalamic_router_free_signal(signal);
        return NULL;
    }

    memcpy(signal->dest_ids, dest_ids, num_dests * sizeof(uint32_t));
    memcpy(signal->signal_data, signal_data, signal_size * sizeof(float));

    signal->source_id = source_id;
    signal->num_dests = num_dests;
    signal->signal_size = signal_size;
    signal->attention_weight = 1.0F;
    signal->priority = priority;
    signal->timestamp_ms = get_current_time_ms();
    signal->bypass_queue = false;

    return signal;
}

void thalamic_router_free_signal(routed_signal_t* signal) {
    if (!signal) return;

    nimcp_free(signal->dest_ids);
    nimcp_free(signal->signal_data);
    nimcp_free(signal);
}

bool thalamic_router_trigger_receptor(thalamic_router_t* router,
                                       uint32_t neuron_id,
                                       uint32_t receptor,
                                       float occupancy,
                                       uint64_t timestamp_ms) {
    // WHAT: Activate receptor and trigger second messenger cascade
    // WHY:  Neuromodulators (ACh, NE) shift thalamic burst/tonic modes
    // HOW:  Route receptor activation to appropriate G-protein cascade

    // Guard clauses
    if (!router) {
        LOG_ERROR(LOG_MODULE, "NULL router in trigger_receptor");
        return false;
    }

    if (!router->second_messengers_enabled || !router->second_messengers) {
        LOG_DEBUG(LOG_MODULE, "Second messengers not enabled");
        return false;
    }

    if (occupancy < 0.0F || occupancy > 1.0F) {
        LOG_WARN(LOG_MODULE, "Invalid occupancy %.3f, clamping to [0,1]", occupancy);
        occupancy = (occupancy < 0.0F) ? 0.0F : 1.0F;
    }

    // Determine G-protein coupling and activate cascade
    receptor_type_t receptor_type = (receptor_type_t)receptor;
    gpcr_coupling_t coupling = second_messenger_receptor_coupling(receptor_type);

    nimcp_result_t result = NIMCP_ERROR;

    switch (coupling) {
        case GPCR_GS_COUPLED:
            // ACh (M1), beta-adrenergic -> cAMP increase -> PKA activation
            result = second_messenger_activate_gs(router->second_messengers,
                                                 neuron_id, occupancy, timestamp_ms);
            LOG_DEBUG(LOG_MODULE, "Activated Gs cascade (neuron=%u, occupancy=%.3f)",
                     neuron_id, occupancy);
            break;

        case GPCR_GI_COUPLED:
            // Alpha2-adrenergic, D2 -> cAMP decrease -> PKA inhibition
            result = second_messenger_activate_gi(router->second_messengers,
                                                 neuron_id, occupancy, timestamp_ms);
            LOG_DEBUG(LOG_MODULE, "Activated Gi cascade (neuron=%u, occupancy=%.3f)",
                     neuron_id, occupancy);
            break;

        case GPCR_GQ_COUPLED:
            // 5-HT2A, mGluR1/5 -> IP3/DAG -> Ca2+ release + PKC
            result = second_messenger_activate_gq(router->second_messengers,
                                                 neuron_id, occupancy, timestamp_ms);
            LOG_DEBUG(LOG_MODULE, "Activated Gq cascade (neuron=%u, occupancy=%.3f)",
                     neuron_id, occupancy);
            break;

        default:
            LOG_WARN(LOG_MODULE, "Unsupported receptor coupling type: %d", coupling);
            return false;
    }

    if (result != NIMCP_SUCCESS) {
        LOG_ERROR(LOG_MODULE, "Failed to activate cascade (result=%d)", result);
        return false;
    }

    return true;
}

bool thalamic_router_get_second_messenger_state(const thalamic_router_t* router,
                                                 uint32_t neuron_id,
                                                 void* state) {
    // WHAT: Query current cascade state for neuron
    // WHY:  Allow inspection of PKA/PKC/CaMKII activity
    // HOW:  Forward to second messenger system query

    // Guard clauses
    if (!router) {
        LOG_ERROR(LOG_MODULE, "NULL router in get_second_messenger_state");
        return false;
    }

    if (!state) {
        LOG_ERROR(LOG_MODULE, "NULL state buffer in get_second_messenger_state");
        return false;
    }

    if (!router->second_messengers_enabled || !router->second_messengers) {
        LOG_DEBUG(LOG_MODULE, "Second messengers not enabled");
        return false;
    }

    // Query cascade state
    second_messenger_state_t* sm_state = (second_messenger_state_t*)state;
    nimcp_result_t result = second_messenger_get_state(
        router->second_messengers, neuron_id, sm_state);

    if (result != NIMCP_SUCCESS) {
        LOG_WARN(LOG_MODULE, "Failed to get cascade state for neuron %u (result=%d)",
                 neuron_id, result);
        return false;
    }

    return true;
}
