//=============================================================================
// nimcp_event_bus.h - Event Bus (Pub/Sub System)
//=============================================================================

#ifndef NIMCP_EVENT_BUS_H
#define NIMCP_EVENT_BUS_H

#include "nimcp_event_types.h"
#include "nimcp_event_queue.h"
#include "nimcp_event_subscriber.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file nimcp_event_bus.h
 * @brief Thread-safe publish/subscribe event bus
 *
 * WHAT: Central event routing and delivery system
 * WHY:  Decouple event producers from consumers
 * HOW:  Queue + subscriber manager + async delivery thread
 */

//=============================================================================
// Types
//=============================================================================

typedef struct event_bus_struct* event_bus_t;

typedef struct {
    uint32_t queue_capacity;
    overflow_policy_t overflow_policy;
    bool async_delivery;        /**< Async delivery thread? */
    uint32_t delivery_thread_sleep_us; /**< Sleep between checks */
} event_bus_config_t;

typedef struct {
    uint64_t events_published;
    uint64_t events_delivered;
    uint64_t events_dropped;
    uint32_t active_subscribers;
    uint32_t queue_size;
} event_bus_stats_t;

//=============================================================================
// Lifecycle
//=============================================================================

/**
 * @brief Create event bus
 */
event_bus_t event_bus_create(const event_bus_config_t* config);

/**
 * @brief Destroy event bus
 */
void event_bus_destroy(event_bus_t bus);

//=============================================================================
// Publish/Subscribe
//=============================================================================

/**
 * @brief Publish event (add to delivery queue)
 *
 * COMPLEXITY: O(log n) queue insertion
 * THREAD-SAFE: Yes
 */
bool event_bus_publish(event_bus_t bus, const event_t* event);

/**
 * @brief Subscribe to events
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes
 *
 * @return Subscription handle for unsubscribe
 */
subscription_handle_t event_bus_subscribe(event_bus_t bus,
                                          event_callback_fn callback,
                                          void* context,
                                          const subscription_config_t* config);

/**
 * @brief Unsubscribe
 */
bool event_bus_unsubscribe(event_bus_t bus, subscription_handle_t handle);

//=============================================================================
// Control
//=============================================================================

/**
 * @brief Process pending events (manual delivery)
 *
 * WHAT: Dequeue and deliver up to max_events
 * WHY:  For sync delivery mode or manual control
 * HOW:  Dequeue batch, dispatch each
 *
 * @return Number of events processed
 */
uint32_t event_bus_process_events(event_bus_t bus, uint32_t max_events);

/**
 * @brief Get statistics
 */
bool event_bus_get_stats(event_bus_t bus, event_bus_stats_t* stats);

/**
 * @brief Default config
 */
event_bus_config_t event_bus_default_config(void);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_EVENT_BUS_H
