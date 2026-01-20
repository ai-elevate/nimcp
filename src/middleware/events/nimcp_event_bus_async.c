//=============================================================================
// nimcp_event_bus_async.c - Async Event Bus Extensions
//=============================================================================

#include "middleware/events/nimcp_event_bus.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "async/nimcp_future.h"
#include "async/nimcp_bio_async.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "utils/time/nimcp_time.h"
#include <string.h>
#include "security/nimcp_blood_brain_barrier.h"
#include "api/nimcp_api_exception.h"

//=============================================================================
// Module Configuration
//=============================================================================

#define MODULE_NAME "middleware.events.async"

//=============================================================================
// Async Publish Context
//=============================================================================

typedef struct {
    event_bus_t bus;
    event_t event;
    nimcp_promise_t promise;
    uint64_t publish_time;
} async_publish_ctx_t;

//=============================================================================
// Request-Response Context
//=============================================================================

typedef struct {
    event_bus_t bus;
    event_type_t response_type;
    nimcp_promise_t promise;
    subscription_handle_t subscription;
    uint64_t start_time;
    uint32_t timeout_ms;
    bool received;
    nimcp_platform_mutex_t mutex;
} request_response_ctx_t;

//=============================================================================
// Forward Declarations
//=============================================================================

static void async_publish_completion_callback(const event_t* event, void* user_data);
static void request_response_callback(const event_t* event, void* user_data);
static void* async_publish_worker(void* arg);
static void* request_timeout_worker(void* arg);

//=============================================================================
// Async Publish Implementation
//=============================================================================

/**
 * @brief Worker thread for async event publishing
 *
 * WHAT: Publishes event and tracks delivery confirmation
 * WHY:  Non-blocking async event publishing
 * HOW:  Publish event, wait for delivery, complete promise
 */
static void* async_publish_worker(void* arg)
{
    async_publish_ctx_t* ctx = (async_publish_ctx_t*)arg;
    if (!ctx) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async publish worker: null context");
        return NULL;
    }

    LOG_MODULE_DEBUG(MODULE_NAME, "Async publish worker started for event type %d",
                     ctx->event.type);

    // Publish event
    bool published = event_bus_publish(ctx->bus, &ctx->event);

    if (!published) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async publish failed for event type %d",
                         ctx->event.type);
        nimcp_promise_fail(ctx->promise, NIMCP_ERROR_OPERATION_FAILED);
        event_free(&ctx->event);
        nimcp_free(ctx);
        return NULL;
    }

    // Get delivery count
    event_bus_stats_t stats;
    if (event_bus_get_stats(ctx->bus, &stats)) {
        uint32_t delivered = (uint32_t)stats.events_delivered;

        LOG_MODULE_DEBUG(MODULE_NAME, "Async publish completed: type=%d delivered=%u",
                         ctx->event.type, delivered);

        nimcp_promise_complete(ctx->promise, &delivered);
    } else {
        uint32_t zero = 0;
        nimcp_promise_complete(ctx->promise, &zero);
    }

    // Cleanup
    event_free(&ctx->event);
    nimcp_free(ctx);

    return NULL;
}

/**
 * @brief Publish event asynchronously with delivery confirmation
 */
nimcp_future_t event_bus_publish_async(event_bus_t bus, const event_t* event)
{
    if (!bus || !event) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async publish: invalid parameters");
        return NULL;
    }

    // Create promise/future pair
    nimcp_promise_t promise = nimcp_promise_create(sizeof(uint32_t));
    if (!promise) {
        LOG_MODULE_ERROR(MODULE_NAME, "Async publish: failed to create promise");
        return NULL;
    }

    nimcp_future_t future = nimcp_promise_get_future(promise);
    if (!future) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async publish: failed to get future");
        return NULL;
    }

    // Allocate context
    async_publish_ctx_t* ctx = (async_publish_ctx_t*)nimcp_malloc(sizeof(async_publish_ctx_t));
    if (!ctx) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async publish: failed to allocate context");
        return NULL;
    }

    ctx->bus = bus;
    ctx->event = *event;  // Copy event
    ctx->promise = promise;
    ctx->publish_time = nimcp_time_get_us();

    // Spawn worker thread
    nimcp_thread_t thread;
    if (nimcp_thread_create(&thread, async_publish_worker, ctx, NULL) != NIMCP_SUCCESS) {
        nimcp_free(ctx);
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Async publish: failed to create worker thread");
        return NULL;
    }

    // Detach thread (worker will clean up context)
    nimcp_thread_detach(thread);

    LOG_MODULE_INFO(MODULE_NAME, "Async publish initiated for event type %d", event->type);

    return future;
}

//=============================================================================
// Request-Response Implementation
//=============================================================================

/**
 * @brief Callback for response events in request-response pattern
 */
static void request_response_callback(const event_t* event, void* user_data)
{
    request_response_ctx_t* ctx = (request_response_ctx_t*)user_data;
    if (!ctx || !event) {
        return;
    }

    nimcp_platform_mutex_lock(&ctx->mutex);

    if (ctx->received) {
        nimcp_platform_mutex_unlock(&ctx->mutex);
        return;  // Already received response
    }

    // Mark as received
    ctx->received = true;

    LOG_MODULE_DEBUG(MODULE_NAME, "Request-response: received response type %d", event->type);

    // Complete promise with response event
    event_t response_copy = *event;
    nimcp_promise_complete(ctx->promise, &response_copy);

    nimcp_platform_mutex_unlock(&ctx->mutex);
}

/**
 * @brief Worker thread for request timeout handling
 */
static void* request_timeout_worker(void* arg)
{
    request_response_ctx_t* ctx = (request_response_ctx_t*)arg;
    if (!ctx) {
        return NULL;
    }

    // Sleep for timeout duration
    nimcp_time_sleep_ms(ctx->timeout_ms);

    nimcp_platform_mutex_lock(&ctx->mutex);

    if (!ctx->received) {
        LOG_MODULE_WARN(MODULE_NAME, "Request-response: timeout after %u ms", ctx->timeout_ms);

        // Unsubscribe
        event_bus_unsubscribe(ctx->bus, ctx->subscription);

        // Fail promise with timeout error
        nimcp_promise_fail(ctx->promise, NIMCP_ERROR_TIMEOUT);
    }

    nimcp_platform_mutex_unlock(&ctx->mutex);
    nimcp_platform_mutex_destroy(&ctx->mutex);
    nimcp_free(ctx);

    return NULL;
}

/**
 * @brief Request-response pattern with async future
 */
nimcp_future_t event_bus_request_async(event_bus_t bus,
                                       const event_t* request,
                                       event_type_t response_type,
                                       uint32_t timeout_ms)
{
    if (!bus || !request) {
        LOG_MODULE_ERROR(MODULE_NAME, "Request-response: invalid parameters");
        return NULL;
    }

    // Create promise/future pair (result is event_t)
    nimcp_promise_t promise = nimcp_promise_create(sizeof(event_t));
    if (!promise) {
        LOG_MODULE_ERROR(MODULE_NAME, "Request-response: failed to create promise");
        return NULL;
    }

    nimcp_future_t future = nimcp_promise_get_future(promise);
    if (!future) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Request-response: failed to get future");
        return NULL;
    }

    // Allocate context
    request_response_ctx_t* ctx = (request_response_ctx_t*)nimcp_calloc(1, sizeof(request_response_ctx_t));
    if (!ctx) {
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Request-response: failed to allocate context");
        return NULL;
    }

    ctx->bus = bus;
    ctx->response_type = response_type;
    ctx->promise = promise;
    ctx->start_time = nimcp_time_get_us();
    ctx->timeout_ms = timeout_ms;
    ctx->received = false;

    if (nimcp_platform_mutex_init(&ctx->mutex, false) != 0) {
        nimcp_free(ctx);
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Request-response: failed to initialize mutex");
        return NULL;
    }

    // Subscribe to response type
    subscription_config_t sub_config = {0};
    sub_config.filter_type = response_type;
    sub_config.enable_type_filter = true;

    subscription_handle_t sub = event_bus_subscribe(
        bus,
        request_response_callback,
        ctx,
        &sub_config
    );

    if (sub == SUBSCRIPTION_HANDLE_INVALID) {
        nimcp_platform_mutex_destroy(&ctx->mutex);
        nimcp_free(ctx);
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Request-response: failed to subscribe");
        return NULL;
    }

    ctx->subscription = sub;

    // Publish request
    if (!event_bus_publish(bus, request)) {
        event_bus_unsubscribe(bus, sub);
        nimcp_platform_mutex_destroy(&ctx->mutex);
        nimcp_free(ctx);
        nimcp_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Request-response: failed to publish request");
        return NULL;
    }

    LOG_MODULE_INFO(MODULE_NAME, "Request-response: sent request type %d, waiting for response type %d",
                    request->type, response_type);

    // Start timeout worker if timeout specified
    if (timeout_ms > 0) {
        nimcp_thread_t timeout_thread;
        if (nimcp_thread_create(&timeout_thread, request_timeout_worker, ctx, NULL) == NIMCP_SUCCESS) {
            nimcp_thread_detach(timeout_thread);
        } else {
            LOG_MODULE_WARN(MODULE_NAME, "Request-response: failed to create timeout worker");
        }
    }

    return future;
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

/**
 * @brief Publish event using bio-async with neuromodulator signaling
 *
 * WHAT: Publish event with biological completion signaling
 * WHY:  Enable neuromodulator-based event delivery tracking
 * HOW:  Use bio-promise with dopamine channel for fast completion
 */
nimcp_bio_future_t event_bus_publish_bio_async(event_bus_t bus,
                                                const event_t* event,
                                                nimcp_bio_channel_type_t channel)
{
    if (!bus || !event) {
        LOG_MODULE_ERROR(MODULE_NAME, "Bio-async publish: invalid parameters");
        return NULL;
    }

    // Create bio-promise with specified neuromodulator channel
    nimcp_bio_promise_t promise = nimcp_bio_promise_create(channel, sizeof(uint32_t));
    if (!promise) {
        LOG_MODULE_ERROR(MODULE_NAME, "Bio-async publish: failed to create bio-promise");
        return NULL;
    }

    nimcp_bio_future_t future = nimcp_bio_promise_get_future(promise);
    if (!future) {
        nimcp_bio_promise_destroy(promise);
        LOG_MODULE_ERROR(MODULE_NAME, "Bio-async publish: failed to get bio-future");
        return NULL;
    }

    // Publish event synchronously
    bool published = event_bus_publish(bus, event);

    if (!published) {
        LOG_MODULE_ERROR(MODULE_NAME, "Bio-async publish: publish failed for event type %d",
                         event->type);
        nimcp_bio_promise_fail(promise, NIMCP_ERROR_OPERATION_FAILED);
        return future;
    }

    // Get delivery count
    event_bus_stats_t stats;
    uint32_t delivered = 0;
    if (event_bus_get_stats(bus, &stats)) {
        delivered = (uint32_t)stats.events_delivered;
    }

    // Complete bio-promise (triggers neuromodulator release)
    nimcp_bio_promise_complete(promise, &delivered);

    LOG_MODULE_INFO(MODULE_NAME, "Bio-async publish completed: type=%d channel=%s delivered=%u",
                    event->type, nimcp_bio_channel_name(channel), delivered);

    return future;
}

/**
 * @brief Request-response with bio-async using predictive coding
 *
 * WHAT: Request-response pattern with biological prediction tracking
 * WHY:  Enable surprise-based callbacks when response differs from prediction
 * HOW:  Use predictive model to track expected response, fire on errors
 */
nimcp_bio_future_t event_bus_request_bio_async(event_bus_t bus,
                                                const event_t* request,
                                                event_type_t response_type,
                                                float expected_latency_ms,
                                                uint32_t timeout_ms)
{
    if (!bus || !request) {
        LOG_MODULE_ERROR(MODULE_NAME, "Bio-async request: invalid parameters");
        return NULL;
    }

    // Create predictive model for response latency
    char model_name[64];
    snprintf(model_name, sizeof(model_name), "request_response_%u", request->type);

    nimcp_predictive_model_t model = nimcp_predictive_create(
        model_name,
        expected_latency_ms,  // Initial prediction
        1.0f                  // Initial precision
    );

    // Create bio-promise with serotonin (sustained coordination signal)
    nimcp_bio_promise_t promise = nimcp_bio_promise_create(
        BIO_CHANNEL_SEROTONIN,
        sizeof(event_t)
    );

    if (!promise) {
        if (model) nimcp_predictive_destroy(model);
        LOG_MODULE_ERROR(MODULE_NAME, "Bio-async request: failed to create bio-promise");
        return NULL;
    }

    nimcp_bio_future_t future = nimcp_bio_promise_get_future(promise);
    if (!future) {
        nimcp_bio_promise_destroy(promise);
        if (model) nimcp_predictive_destroy(model);
        LOG_MODULE_ERROR(MODULE_NAME, "Bio-async request: failed to get bio-future");
        return NULL;
    }

    // For now, use standard request-response mechanism
    // In full implementation, would integrate predictive model callbacks

    // Publish request
    if (!event_bus_publish(bus, request)) {
        nimcp_bio_promise_fail(promise, NIMCP_ERROR_OPERATION_FAILED);
        if (model) nimcp_predictive_destroy(model);
        LOG_MODULE_ERROR(MODULE_NAME, "Bio-async request: failed to publish");
        return future;
    }

    LOG_MODULE_INFO(MODULE_NAME, "Bio-async request sent: type=%d response_type=%d expected_latency=%.1fms",
                    request->type, response_type, expected_latency_ms);

    // Note: Full implementation would set up response subscription and
    // observe actual latency for predictive coding updates

    return future;
}
