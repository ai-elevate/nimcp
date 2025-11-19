//=============================================================================
// nimcp_event_bus.c - Event Bus Implementation
//=============================================================================

#include "middleware/events/nimcp_event_bus.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/thread/nimcp_thread.h"
#include <unistd.h>

//=============================================================================
// Internal Structure
//=============================================================================

struct event_bus_struct {
    event_queue_t queue;
    subscriber_manager_t subscribers;

    // Async delivery thread
    bool async_delivery;
    nimcp_thread_t delivery_thread;
    bool running;
    uint32_t delivery_thread_sleep_us;

    // Statistics
    uint64_t events_published;
    uint64_t events_delivered;
    uint64_t events_dropped;

    nimcp_platform_mutex_t mutex;
};

//=============================================================================
// Delivery Thread
//=============================================================================

static void* delivery_thread_fn(void* arg) {
    event_bus_t bus = (event_bus_t)arg;

    while (bus->running) {
        // Process batch of events
        event_t events[32];
        uint32_t count = event_queue_dequeue_batch(bus->queue, events, 32);

        for (uint32_t i = 0; i < count; i++) {
            subscriber_dispatch_event(bus->subscribers, &events[i]);
            event_free(&events[i]);

            nimcp_platform_mutex_lock(&bus->mutex);
            bus->events_delivered++;
            nimcp_platform_mutex_unlock(&bus->mutex);
        }

        // Sleep if no events
        if (count == 0) {
            usleep(bus->delivery_thread_sleep_us);
        }
    }

    return NULL;
}

//=============================================================================
// Lifecycle
//=============================================================================

event_bus_config_t event_bus_default_config(void) {
    event_bus_config_t config = {0};
    config.queue_capacity = 1024;
    config.overflow_policy = OVERFLOW_POLICY_DROP_OLDEST;
    config.async_delivery = false; // Manual by default
    config.delivery_thread_sleep_us = 1000; // 1ms
    return config;
}

event_bus_t event_bus_create(const event_bus_config_t* config) {
    event_bus_config_t cfg = config ? *config : event_bus_default_config();

    event_bus_t bus = nimcp_calloc(1, sizeof(struct event_bus_struct));
    if (!bus) return NULL;

    // Create queue
    event_queue_config_t queue_cfg = event_queue_default_config();
    queue_cfg.capacity = cfg.queue_capacity;
    queue_cfg.overflow_policy = cfg.overflow_policy;

    bus->queue = event_queue_create(&queue_cfg);
    if (!bus->queue) {
        nimcp_free(bus);
        return NULL;
    }

    // Create subscriber manager
    bus->subscribers = subscriber_manager_create();
    if (!bus->subscribers) {
        event_queue_destroy(bus->queue);
        nimcp_free(bus);
        return NULL;
    }

    if (nimcp_platform_mutex_init(&bus->mutex, false) != 0) {
        subscriber_manager_destroy(bus->subscribers);
        event_queue_destroy(bus->queue);
        nimcp_free(bus);
        return NULL;
    }

    bus->async_delivery = cfg.async_delivery;
    bus->delivery_thread_sleep_us = cfg.delivery_thread_sleep_us;

    // Start async delivery thread if enabled
    if (bus->async_delivery) {
        bus->running = true;
        if (nimcp_thread_create(&bus->delivery_thread, delivery_thread_fn, bus, NULL) != NIMCP_SUCCESS) {
            nimcp_platform_mutex_destroy(&bus->mutex);
            subscriber_manager_destroy(bus->subscribers);
            event_queue_destroy(bus->queue);
            nimcp_free(bus);
            return NULL;
        }
    }

    return bus;
}

void event_bus_destroy(event_bus_t bus) {
    if (!bus) return;

    // Stop delivery thread
    if (bus->async_delivery) {
        bus->running = false;
        nimcp_thread_join(bus->delivery_thread, NULL);
    }

    subscriber_manager_destroy(bus->subscribers);
    event_queue_destroy(bus->queue);
    nimcp_platform_mutex_destroy(&bus->mutex);
    nimcp_free(bus);
}

//=============================================================================
// Pub/Sub
//=============================================================================

bool event_bus_publish(event_bus_t bus, const event_t* event) {
    if (!bus || !event) return false;

    bool success = event_queue_enqueue(bus->queue, event);

    nimcp_platform_mutex_lock(&bus->mutex);
    if (success) {
        bus->events_published++;
    } else {
        bus->events_dropped++;
    }
    nimcp_platform_mutex_unlock(&bus->mutex);

    return success;
}

subscription_handle_t event_bus_subscribe(event_bus_t bus,
                                          event_callback_fn callback,
                                          void* context,
                                          const subscription_config_t* config) {
    if (!bus) return SUBSCRIPTION_HANDLE_INVALID;
    return subscriber_subscribe(bus->subscribers, callback, context, config);
}

bool event_bus_unsubscribe(event_bus_t bus, subscription_handle_t handle) {
    if (!bus) return false;
    return subscriber_unsubscribe(bus->subscribers, handle);
}

//=============================================================================
// Manual Processing
//=============================================================================

uint32_t event_bus_process_events(event_bus_t bus, uint32_t max_events) {
    if (!bus) return 0;

    event_t events[256];
    uint32_t batch_size = (max_events < 256) ? max_events : 256;
    uint32_t count = event_queue_dequeue_batch(bus->queue, events, batch_size);

    for (uint32_t i = 0; i < count; i++) {
        subscriber_dispatch_event(bus->subscribers, &events[i]);
        event_free(&events[i]);

        nimcp_platform_mutex_lock(&bus->mutex);
        bus->events_delivered++;
        nimcp_platform_mutex_unlock(&bus->mutex);
    }

    return count;
}

//=============================================================================
// Statistics
//=============================================================================

bool event_bus_get_stats(event_bus_t bus, event_bus_stats_t* stats) {
    if (!bus || !stats) return false;

    nimcp_platform_mutex_lock(&bus->mutex);
    stats->events_published = bus->events_published;
    stats->events_delivered = bus->events_delivered;
    stats->active_subscribers = subscriber_get_count(bus->subscribers);
    stats->queue_size = event_queue_size(bus->queue);

    // Get dropped count from queue stats (handles DROP_OLDEST policy correctly)
    event_queue_stats_t queue_stats;
    if (event_queue_get_stats(bus->queue, &queue_stats)) {
        stats->events_dropped = queue_stats.total_dropped;
    } else {
        stats->events_dropped = bus->events_dropped;
    }

    nimcp_platform_mutex_unlock(&bus->mutex);

    return true;
}
