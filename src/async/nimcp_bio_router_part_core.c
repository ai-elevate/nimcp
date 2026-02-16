// nimcp_bio_router_part_core.c - core functions
// Part of nimcp_bio_router.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp_bio_router.c


/*=============================================================================
 * MODULE REGISTRATION
 *============================================================================*/

bio_module_context_t bio_router_register_module(const bio_module_info_t* info) {
    if (!g_router || !info) {
        LOG_ERROR("Cannot register module: router not initialized or invalid info");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bio_router_register_module: required parameter is NULL (g_router, info)");
        return NULL;
    }

    nimcp_platform_mutex_lock(&g_router->modules_mutex);

    // Check for duplicate module ID - if already registered, create a NEW context
    // wrapping the existing entry (allows multiple consumers of same module ID)
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].magic == BIO_MODULE_MAGIC &&
            g_router->modules[i].module_id == info->module_id) {
            bio_module_entry_t* existing = &g_router->modules[i];
            /* P1-25 fix: Update user_data INSIDE the lock to prevent race condition.
             * Previously this was done after unlocking, allowing concurrent access
             * to see a partially-updated entry. */
            existing->user_data = info->user_data;
            nimcp_platform_mutex_unlock(&g_router->modules_mutex);
            /* P1 fix: Warn about potential memory leak on re-registration.
             * Each call allocates a new context. Callers MUST free the
             * previously returned context via bio_router_unregister_module()
             * before re-registering, or they will leak the old context. */
            LOG_WARN("Module ID %u (%s) already registered - caller must free "
                     "previous context to avoid memory leak",
                     info->module_id, existing->module_name);
            // CRITICAL FIX: Must allocate a proper context struct, NOT cast entry!
            // The old code returned (bio_module_context_t)existing which was wrong because
            // bio_module_entry_t and bio_module_context_struct are different structs.
            bio_module_context_t ctx = nimcp_calloc(1, sizeof(struct bio_module_context_struct));
            if (!ctx) {
                LOG_ERROR("Failed to allocate context for existing module");
                NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "ctx is NULL");

                return NULL;
            }
            ctx->magic = BIO_MODULE_MAGIC;
            ctx->entry = existing;
            return ctx;
        }
    }

    // Find a free slot (either a reusable invalid slot, or a new slot at the end)
    bio_module_entry_t* entry = NULL;

    // First try to reuse an invalid slot
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].magic != BIO_MODULE_MAGIC) {
            entry = &g_router->modules[i];
            LOG_DEBUG("Reusing module slot %u for new module %u", i, info->module_id);
            break;
        }
    }

    // If no reusable slot, allocate at the end
    if (!entry) {
        if (g_router->module_count >= g_router->module_capacity) {
            nimcp_platform_mutex_unlock(&g_router->modules_mutex);
            LOG_ERROR("Module registry full (max=%u)", g_router->module_capacity);
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "entry is NULL");

        return NULL;
        }
        entry = &g_router->modules[g_router->module_count];
        g_router->module_count++;
    }

    memset(entry, 0, sizeof(*entry));

    entry->magic = BIO_MODULE_MAGIC;
    entry->module_id = info->module_id;
    strncpy(entry->module_name, info->module_name ? info->module_name : "unknown",
            MAX_MODULE_NAME - 1);
    entry->user_data = info->user_data;

    // Initialize inbox with tier-optimized capacity
    // Cap at NIMCP_BIO_INBOX_CAPACITY to enforce tier-based memory limits
    uint32_t inbox_cap = info->inbox_capacity > 0 ?
                         info->inbox_capacity : g_router->config.inbox_capacity;
    // Enforce tier-based maximum (saves 150KB+ on MINIMAL tier)
    if (inbox_cap > NIMCP_BIO_INBOX_CAPACITY) {
        inbox_cap = NIMCP_BIO_INBOX_CAPACITY;
    }
    if (bio_msg_queue_init(&entry->inbox, inbox_cap) != NIMCP_SUCCESS) {
        nimcp_platform_mutex_unlock(&g_router->modules_mutex);
        LOG_ERROR("Failed to initialize inbox for module %s", entry->module_name);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "bio_router_register_module: validation failed");
        return NULL;
    }

    // Initialize handler mutex
    if (nimcp_platform_mutex_init(&entry->handler_mutex, false) != 0) {
        bio_msg_queue_destroy(&entry->inbox);
        nimcp_platform_mutex_unlock(&g_router->modules_mutex);
        LOG_ERROR("Failed to initialize handler mutex for module %s", entry->module_name);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "bio_router_register_module: validation failed");
        return NULL;
    }

    // Count active modules for statistics
    uint32_t active_count = 0;
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].magic == BIO_MODULE_MAGIC) {
            active_count++;
        }
    }

    // Update statistics
    nimcp_platform_mutex_lock(&g_router->stats_mutex);
    g_router->stats.active_modules = active_count;
    nimcp_platform_mutex_unlock(&g_router->stats_mutex);

    nimcp_platform_mutex_unlock(&g_router->modules_mutex);

    // Create context handle
    bio_module_context_t ctx = nimcp_calloc(1, sizeof(struct bio_module_context_struct));
    if (!ctx) {
        LOG_ERROR("Failed to allocate module context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "bio_router_register_module: failed to allocate context");
        /* Undo registration to avoid orphaned entry */
        nimcp_platform_mutex_lock(&g_router->modules_mutex);
        entry->magic = 0;
        nimcp_platform_mutex_unlock(&g_router->modules_mutex);
        return NULL;
    }

    ctx->magic = BIO_MODULE_MAGIC;
    ctx->entry = entry;

    LOG_INFO("Registered module: id=%u, name=%s, inbox_capacity=%u",
             info->module_id, entry->module_name, inbox_cap);

    return ctx;
}


void bio_router_unregister_module(bio_module_context_t ctx) {
    if (!g_router || !ctx || ctx->magic != BIO_MODULE_MAGIC) return;

    bio_module_entry_t* entry = ctx->entry;
    if (!entry || entry->magic != BIO_MODULE_MAGIC) return;

    LOG_INFO("Unregistering module: id=%u, name=%s",
             entry->module_id, entry->module_name);

    nimcp_platform_mutex_lock(&g_router->modules_mutex);

    // Destroy inbox
    bio_msg_queue_destroy(&entry->inbox);

    // Clear all handlers to prevent accumulation on re-registration
    entry->handler_count = 0;
    memset(entry->handlers, 0, sizeof(entry->handlers));

    // Destroy handler mutex
    nimcp_platform_mutex_destroy(&entry->handler_mutex);

    // Mark as invalid (DO NOT compact array - other contexts have pointers to entries)
    entry->magic = 0;

    /* P2-55 note: module_count is the high-water mark, NOT the active count.
     * We do NOT decrement it because other contexts may hold pointers to entries
     * at indices < module_count. Instead, invalid entries (magic=0) are skipped
     * by find_module, shutdown, and iteration. Active count is tracked separately
     * in stats.active_modules (updated below). */

    // Count active modules for statistics
    uint32_t active_count = 0;
    for (uint32_t i = 0; i < g_router->module_count; i++) {
        if (g_router->modules[i].magic == BIO_MODULE_MAGIC) {
            active_count++;
        }
    }

    // Update statistics
    nimcp_platform_mutex_lock(&g_router->stats_mutex);
    g_router->stats.active_modules = active_count;
    nimcp_platform_mutex_unlock(&g_router->stats_mutex);

    nimcp_platform_mutex_unlock(&g_router->modules_mutex);

    // Free context
    ctx->magic = 0;
    nimcp_free(ctx);
}


nimcp_error_t bio_router_send(bio_module_context_t ctx,
                               const void* msg,
                               size_t msg_size,
                               uint32_t timeout_ms) {
    NIMCP_CHECK_THROW(g_router && ctx && ctx->magic == BIO_MODULE_MAGIC && msg && msg_size > 0,
                      NIMCP_ERROR_INVALID_PARAM,
                      "bio_router_send: invalid router, context, or message");

    const bio_message_header_t* header = (const bio_message_header_t*)msg;
    uint32_t target_id = header->target_module;

    uint64_t start_time = nimcp_platform_time_monotonic_us();

    LOG_TRACE("Routing message type=0x%04X from=%u to=%u size=%zu",
              header->type, header->source_module, target_id, msg_size);

    // BBB Security Gate: Validate message content before dispatch
    // WHAT: Apply Blood-Brain Barrier validation to all messages
    // WHY:  Prevent malicious or malformed messages from affecting modules
    // HOW:  Use global BBB system to validate raw message bytes
    bbb_system_t bbb = nimcp_bbb_get_global_system();
    if (bbb && bbb_system_is_enabled(bbb)) {
        bbb_validation_result_t bbb_result;
        if (!bbb_validate_input(bbb, msg, msg_size, &bbb_result)) {
            LOG_WARN("BBB validation failed for message type=0x%04X from=%u: threat=%d severity=%d",
                     header->type, header->source_module, bbb_result.threat, bbb_result.severity);

            // Update dropped message stats
            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);

            NIMCP_CHECK_THROW(false, NIMCP_ERROR_PERMISSION_DENIED,
                              "bio_router_send: BBB validation failed");
        }
        LOG_TRACE("BBB validation passed for message type=0x%04X", header->type);
    }

    /* Phase 15: Mesh integration hook - try mesh routing first if available
     * WHAT: Route messages through mesh network for consensus-based delivery
     * WHY:  Enable distributed consensus for inter-module messages
     * HOW:  Check global mesh integration, route if available, fall back if not
     */
    if (mesh_bio_integration_global_available()) {
        nimcp_error_t mesh_result = mesh_bio_integration_route_message(
            mesh_bio_integration_get_global(),
            msg, msg_size, NULL, NULL);

        if (mesh_result == NIMCP_SUCCESS) {
            /* Message routed through mesh - update stats and return */
            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_routed++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);

            if (ctx && ctx->entry) {
                ctx->entry->messages_sent++;
            }

            LOG_TRACE("Message type=0x%04X routed through mesh", header->type);
            return NIMCP_SUCCESS;
        }
        /* mesh_result == NIMCP_ERROR_NOT_FOUND means fall through to direct routing */
        else if (mesh_result != NIMCP_ERROR_NOT_FOUND) {
            LOG_WARN("Mesh routing failed for type=0x%04X: %d, falling back",
                    header->type, mesh_result);
        }
        /* Fall through to normal routing */
    }

    nimcp_platform_mutex_lock(&g_router->modules_mutex);

    nimcp_error_t result = NIMCP_SUCCESS;

    if (target_id == BIO_MODULE_MESH_ROUTE) {
        /* Phase 15: Explicit mesh routing - MUST route through mesh */
        nimcp_platform_mutex_unlock(&g_router->modules_mutex);

        if (!mesh_bio_integration_global_available()) {
            LOG_ERROR("Mesh routing requested but mesh integration not available");
            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);
            NIMCP_THROW(NIMCP_ERROR_NOT_INITIALIZED,
                        "bio_router_send: mesh routing requested but not available");
        }

        result = mesh_bio_integration_route_message(
            mesh_bio_integration_get_global(),
            msg, msg_size, NULL, NULL);

        if (result != NIMCP_SUCCESS) {
            LOG_ERROR("Mesh routing failed for type=0x%04X: %d", header->type, result);
            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);
            return result;
        }

        nimcp_platform_mutex_lock(&g_router->stats_mutex);
        g_router->stats.messages_routed++;
        nimcp_platform_mutex_unlock(&g_router->stats_mutex);

        LOG_DEBUG("Mesh route: msg type 0x%04X successfully routed", header->type);
        return NIMCP_SUCCESS;

    } else if (target_id == BIO_MODULE_KG_DISPATCH) {
        /* Phase 7: KG-driven dispatch - route to all handlers for message type */
        nimcp_platform_mutex_unlock(&g_router->modules_mutex);

        if (!g_router_brain_kg) {
            LOG_WARN("KG dispatch requested but brain_kg not set");
            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_NOT_INITIALIZED,
                              "KG dispatch requested but brain_kg not set");
        }

        int dispatched = bio_router_kg_dispatch_internal(msg, msg_size, timeout_ms);
        if (dispatched < 0) {
            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_OPERATION_FAILED,
                              "bio_router_send: KG dispatch failed");
        }

        LOG_DEBUG("KG dispatch: msg type 0x%04X delivered to %d handlers",
                  header->type, dispatched);
        return NIMCP_SUCCESS;

    } else if (target_id == 0) {
        // Broadcast to all modules except sender
        for (uint32_t i = 0; i < g_router->module_count; i++) {
            bio_module_entry_t* entry = &g_router->modules[i];
            if (entry->magic == BIO_MODULE_MAGIC &&
                entry->module_id != header->source_module) {

                nimcp_error_t enq_result = bio_msg_queue_enqueue(&entry->inbox,
                                                                  msg, msg_size,
                                                                  NULL, timeout_ms);
                if (enq_result == NIMCP_SUCCESS) {
                    entry->messages_received++;
                } else {
                    result = NIMCP_ERROR_OUT_OF_RANGE;
                    // Queue overflow expected during high-load scenarios (stress tests)
                    LOG_DEBUG("Failed to enqueue broadcast to module %u (queue full)", entry->module_id);
                }
            }
        }

        // Update broadcast stats
        nimcp_platform_mutex_lock(&g_router->stats_mutex);
        g_router->stats.broadcasts_sent++;
        nimcp_platform_mutex_unlock(&g_router->stats_mutex);

    } else {
        // Send to specific module
        bio_module_entry_t* target = bio_router_find_module(target_id);
        if (!target) {
            nimcp_platform_mutex_unlock(&g_router->modules_mutex);
            LOG_ERROR("Target module %u not found", target_id);

            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);

            NIMCP_CHECK_THROW(false, NIMCP_ERROR_NOT_FOUND,
                              "bio_router_send: target module not found");
        }

        result = bio_msg_queue_enqueue(&target->inbox, msg, msg_size,
                                        NULL, timeout_ms);
        if (result == NIMCP_SUCCESS) {
            target->messages_received++;
        } else {
            LOG_WARN("Failed to enqueue to module %u inbox", target_id);

            nimcp_platform_mutex_lock(&g_router->stats_mutex);
            g_router->stats.messages_dropped++;
            nimcp_platform_mutex_unlock(&g_router->stats_mutex);
        }
    }

    nimcp_platform_mutex_unlock(&g_router->modules_mutex);

    // Update routing statistics
    if (result == NIMCP_SUCCESS) {
        uint64_t latency_us = nimcp_platform_time_monotonic_us() - start_time;
        float latency = (float)latency_us;

        nimcp_platform_mutex_lock(&g_router->stats_mutex);
        g_router->stats.messages_routed++;

        // Update latency stats
        if (g_router->stats.messages_routed == 1) {
            g_router->stats.avg_routing_latency_us = latency;
            g_router->stats.max_routing_latency_us = latency;
        } else {
            g_router->stats.avg_routing_latency_us =
                (g_router->stats.avg_routing_latency_us * 0.95F) + (latency * 0.05F);
            if (latency > g_router->stats.max_routing_latency_us) {
                g_router->stats.max_routing_latency_us = latency;
            }
        }

        nimcp_platform_mutex_unlock(&g_router->stats_mutex);

        // Observe message with predictive protocol
        if (g_router->predictive_proto) {
            predictive_protocol_observe(g_router->predictive_proto, header);

            // Generate predictions and prefetch
            prediction_t predictions[5];
            uint32_t pred_count = predictive_protocol_predict_next(g_router->predictive_proto,
                                                                    header,
                                                                    predictions, 5);

            // Prefetch high-confidence predictions
            for (uint32_t i = 0; i < pred_count; i++) {
                if (predictions[i].confidence >= 0.8F) {
                    predictive_protocol_prefetch(g_router->predictive_proto, &predictions[i]);
                }
            }
        }
    }

    return result;
}


nimcp_bio_promise_t bio_router_send_async(bio_module_context_t ctx,
                                           const void* msg,
                                           size_t msg_size,
                                           nimcp_bio_channel_type_t channel) {
    if (!g_router || !ctx || !msg || msg_size == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "unknown: required parameter is NULL (g_router, ctx, msg)");
        return NULL;
    }

    // Create promise for response
    // Note: result_size is used as capacity - actual copy size determined by
    // nimcp_bio_promise_complete_sized(). Use 0 to indicate handler will
    // provide the actual size.
    nimcp_bio_promise_t promise = nimcp_bio_promise_create(channel, 0);
    if (!promise) {
        LOG_ERROR("Failed to create promise for async send");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "promise is NULL");

        return NULL;
    }

    // Send message with promise attached so handler can complete it
    nimcp_error_t result = bio_router_send_with_promise(ctx, msg, msg_size, promise, 0);
    if (result != NIMCP_SUCCESS) {
        nimcp_bio_promise_fail(promise, result);
    }

    return promise;
}


nimcp_error_t bio_router_request(bio_module_context_t ctx,
                                  const void* request,
                                  size_t request_size,
                                  void* response,
                                  size_t response_size,
                                  uint32_t timeout_ms) {
    NIMCP_CHECK_THROW(ctx && request && response, NIMCP_ERROR_INVALID_PARAM,
                      "bio_router_request: invalid context, request, or response buffer");

    bio_module_entry_t* entry = ctx->entry;
    NIMCP_CHECK_THROW(entry && entry->magic == BIO_MODULE_MAGIC, NIMCP_ERROR_INVALID_STATE,
                      "bio_router_request: invalid module entry");

    // Create a promise for the response
    nimcp_bio_promise_t promise = nimcp_bio_promise_create(BIO_CHANNEL_DOPAMINE, response_size);
    if (!promise) {
        LOG_ERROR("Failed to create promise for request");
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_NO_MEMORY,
                          "bio_router_request: failed to create promise");
    }

    // Send request with promise
    nimcp_error_t send_result = bio_router_send_with_promise(ctx, request, request_size,
                                                               promise, timeout_ms);
    if (send_result != NIMCP_SUCCESS) {
        nimcp_bio_promise_destroy(promise);
        LOG_ERROR("Failed to send request");
        return send_result;
    }

    // Get future from promise
    nimcp_bio_future_t future = nimcp_bio_promise_get_future(promise);
    if (!future) {
        nimcp_bio_promise_destroy(promise);
        LOG_ERROR("Failed to get future from promise");
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_OPERATION_FAILED,
                          "bio_router_request: failed to get future from promise");
    }

    // Wait for response with timeout
    nimcp_error_t wait_result = nimcp_bio_future_wait(future, response,
                                                        timeout_ms > 0 ? timeout_ms : DEFAULT_TIMEOUT_MS);

    // Cleanup - destroy both future and promise to avoid memory leak
    nimcp_bio_future_destroy(future);
    nimcp_bio_promise_destroy(promise);

    if (wait_result != NIMCP_SUCCESS) {
        LOG_DEBUG("Request timed out or failed");

        // Update timeout statistics
        nimcp_platform_mutex_lock(&g_router->stats_mutex);
        g_router->stats.timeouts++;
        nimcp_platform_mutex_unlock(&g_router->stats_mutex);

        return wait_result;
    }

    LOG_TRACE("Request completed successfully");
    return NIMCP_SUCCESS;
}


nimcp_error_t bio_router_broadcast(bio_module_context_t ctx,
                                    const void* msg,
                                    size_t msg_size) {
    NIMCP_CHECK_THROW(ctx && msg && msg_size > 0, NIMCP_ERROR_INVALID_PARAM,
                      "bio_router_broadcast: invalid context or message");

    // Create a copy with target set to 0 (broadcast)
    bio_message_header_t* header = (bio_message_header_t*)msg;
    uint32_t original_target = header->target_module;
    header->target_module = 0;
    header->flags |= BIO_MSG_FLAG_BROADCAST;

    nimcp_error_t result = bio_router_send(ctx, msg, msg_size, 0);

    // Restore original target
    header->target_module = original_target;
    header->flags &= ~BIO_MSG_FLAG_BROADCAST;

    return result;
}


nimcp_error_t bio_router_observe_signal(bio_module_context_t ctx,
                                         const char* signal_name,
                                         float initial_prediction,
                                         float precision,
                                         bio_prediction_observer_t callback) {
    NIMCP_CHECK_THROW(ctx && signal_name && callback, NIMCP_ERROR_INVALID_PARAM,
                      "bio_router_observe_signal: invalid context, signal_name, or callback");

    // Initialize mutex on first use (thread-safe via pthread_once)
    nimcp_platform_once(&g_signal_mutex_once, init_signal_mutex_once);

    nimcp_platform_mutex_lock(&g_signal_mutex);

    if (g_signal_observer_count >= 256) {
        nimcp_platform_mutex_unlock(&g_signal_mutex);
        LOG_ERROR("Signal observer registry full");
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_OUT_OF_RANGE,
                          "bio_router_observe_signal: observer registry full");
    }

    // Register observer
    signal_observer_t* obs = &g_signal_observers[g_signal_observer_count];
    strncpy(obs->signal_name, signal_name, sizeof(obs->signal_name) - 1);
    obs->signal_name[sizeof(obs->signal_name) - 1] = '\0';
    obs->prediction = initial_prediction;
    obs->precision = precision;
    obs->callback = callback;
    obs->user_data = bio_module_context_get_user_data(ctx);
    obs->active = true;

    g_signal_observer_count++;

    nimcp_platform_mutex_unlock(&g_signal_mutex);

    LOG_DEBUG("Registered observer for signal '%s' (prediction=%.3f, precision=%.3f)",
              signal_name, initial_prediction, precision);

    return NIMCP_SUCCESS;
}


nimcp_error_t bio_router_publish_signal(bio_module_context_t ctx,
                                         const char* signal_name,
                                         float value) {
    NIMCP_CHECK_THROW(ctx && signal_name, NIMCP_ERROR_INVALID_PARAM,
                      "bio_router_publish_signal: invalid context or signal_name");

    // Initialize mutex on first use (thread-safe via pthread_once)
    nimcp_platform_once(&g_signal_mutex_once, init_signal_mutex_once);

    nimcp_platform_mutex_lock(&g_signal_mutex);

    // Notify all observers of this signal
    uint32_t notified_count = 0;
    for (uint32_t i = 0; i < g_signal_observer_count; i++) {
        signal_observer_t* obs = &g_signal_observers[i];

        if (obs->active && strcmp(obs->signal_name, signal_name) == 0) {
            // Compute prediction error
            float error = value - obs->prediction;

            // Update prediction with learning (simple exponential moving average)
            obs->prediction = obs->prediction + 0.1F * error;

            // Call observer callback
            obs->callback(signal_name, value, obs->user_data);

            notified_count++;
        }
    }

    nimcp_platform_mutex_unlock(&g_signal_mutex);

    if (notified_count > 0) {
        LOG_TRACE("Published signal '%s' value=%.3f (notified %u observers)",
                  signal_name, value, notified_count);
    }

    return NIMCP_SUCCESS;
}

nimcp_phase_sync_t bio_router_sync_request(bio_module_context_t ctx,
                                            nimcp_oscillation_band_t band,
                                            const bio_module_id_t* targets,
                                            size_t target_count,
                                            const void* request,
                                            size_t request_size) {
    if (!ctx || !targets || target_count == 0 || !request || request_size == 0) {
        LOG_ERROR("Invalid parameters for sync request");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "init_signal_mutex_once: required parameter is NULL (ctx, targets, request)");
        return NULL;
    }

    if (!g_router) {
        LOG_ERROR("Router not initialized");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "g_router is NULL");


        return NULL;
    }

    // Allocate sync context
    phase_sync_context_t* sync_ctx = nimcp_calloc(1, sizeof(phase_sync_context_t));
    if (!sync_ctx) {
        LOG_ERROR("Failed to allocate phase sync context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "sync_ctx is NULL");

        return NULL;
    }

    sync_ctx->band = band;
    sync_ctx->promise_count = (uint32_t)target_count;
    sync_ctx->completed_count = 0;
    sync_ctx->all_ready = false;

    // Allocate promise array
    sync_ctx->promises = nimcp_calloc(target_count, sizeof(nimcp_bio_promise_t));
    if (!sync_ctx->promises) {
        nimcp_free(sync_ctx);
        LOG_ERROR("Failed to allocate promise array");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "init_signal_mutex_once: sync_ctx->promises is NULL");
        return NULL;
    }

    // Initialize synchronization primitives
    nimcp_platform_mutex_init(&sync_ctx->mutex, false);
    nimcp_platform_cond_init(&sync_ctx->cond);

    // Send request to all targets and collect promises
    nimcp_bio_channel_type_t channel = BIO_CHANNEL_DOPAMINE;
    for (size_t i = 0; i < target_count; i++) {
        // Create a promise for this target's response
        sync_ctx->promises[i] = nimcp_bio_promise_create(channel, 0);

        if (!sync_ctx->promises[i]) {
            LOG_WARN("Failed to create promise for target %u", targets[i]);
            continue;
        }

        // Send request with promise
        nimcp_error_t send_result = bio_router_send_with_promise(
            ctx, request, request_size, sync_ctx->promises[i], 0);

        if (send_result != NIMCP_SUCCESS) {
            LOG_WARN("Failed to send sync request to target %u", targets[i]);
        }
    }

    LOG_DEBUG("Created phase sync request for %zu targets on %s band",
              target_count,
              band == BIO_OSC_GAMMA ? "GAMMA" :
              band == BIO_OSC_BETA ? "BETA" :
              band == BIO_OSC_ALPHA ? "ALPHA" : "OTHER");

    return (nimcp_phase_sync_t)sync_ctx;
}


nimcp_error_t bio_router_on_wave_arrival(bio_module_context_t ctx,
                                          nimcp_wave_callback_t callback,
                                          void* user_data) {
    NIMCP_CHECK_THROW(ctx && callback, NIMCP_ERROR_INVALID_PARAM,
                      "bio_router_on_wave_arrival: invalid context or callback");

    // Initialize mutex on first use (thread-safe via pthread_once)
    nimcp_platform_once(&g_wave_mutex_once, init_wave_mutex_once);

    nimcp_platform_mutex_lock(&g_wave_mutex);

    if (g_wave_callback_count >= 128) {
        nimcp_platform_mutex_unlock(&g_wave_mutex);
        LOG_ERROR("Wave callback registry full");
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_OUT_OF_RANGE,
                          "bio_router_on_wave_arrival: wave callback registry full");
    }

    // Register callback
    wave_callback_entry_t* entry = &g_wave_callbacks[g_wave_callback_count];
    entry->module_id = bio_module_context_get_id(ctx);
    entry->callback = callback;
    entry->user_data = user_data;
    entry->active = true;

    g_wave_callback_count++;

    nimcp_platform_mutex_unlock(&g_wave_mutex);

    LOG_DEBUG("Registered wave arrival callback for module %u", entry->module_id);

    return NIMCP_SUCCESS;
}


/*=============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

const char* bio_msg_type_name(bio_message_type_t type) {
    switch (type) {
        // Brain messages
        case BIO_MSG_BRAIN_STATE_QUERY: return "BRAIN_STATE_QUERY";
        case BIO_MSG_BRAIN_STATE_RESPONSE: return "BRAIN_STATE_RESPONSE";
        case BIO_MSG_NEURON_ACTIVATION_REQUEST: return "NEURON_ACTIVATION_REQUEST";
        case BIO_MSG_NEURON_ACTIVATION_RESPONSE: return "NEURON_ACTIVATION_RESPONSE";

        // Plasticity messages
        case BIO_MSG_WEIGHT_UPDATE_REQUEST: return "WEIGHT_UPDATE_REQUEST";
        case BIO_MSG_WEIGHT_UPDATE_RESPONSE: return "WEIGHT_UPDATE_RESPONSE";
        case BIO_MSG_STDP_EVENT: return "STDP_EVENT";
        case BIO_MSG_NEUROMODULATOR_RELEASE: return "NEUROMODULATOR_RELEASE";

        // Cognitive messages
        case BIO_MSG_INTROSPECTION_QUERY: return "INTROSPECTION_QUERY";
        case BIO_MSG_ETHICS_EVALUATION_REQUEST: return "ETHICS_EVALUATION_REQUEST";
        case BIO_MSG_SALIENCE_QUERY: return "SALIENCE_QUERY";
        case BIO_MSG_ATTENTION_SHIFT: return "ATTENTION_SHIFT";

        // System messages
        case BIO_MSG_HEALTH_CHECK: return "HEALTH_CHECK";
        case BIO_MSG_ERROR_REPORT: return "ERROR_REPORT";

        default:
            if (type >= 0x0100 && type < 0x0200) return "BRAIN_MSG";
            if (type >= 0x0200 && type < 0x0300) return "PLASTICITY_MSG";
            if (type >= 0x0300 && type < 0x0400) return "COGNITIVE_MSG";
            if (type >= 0x0400 && type < 0x0500) return "GLIAL_MSG";
            if (type >= 0x0500 && type < 0x0600) return "MIDDLEWARE_MSG";
            if (type >= 0x0600 && type < 0x0700) return "TRAINING_MSG";
            return "UNKNOWN";
    }
}


const char* bio_module_name(bio_module_id_t module) {
    switch (module) {
        case BIO_MODULE_BRAIN: return "BRAIN";
        case BIO_MODULE_INTROSPECTION: return "INTROSPECTION";
        case BIO_MODULE_ETHICS: return "ETHICS";
        case BIO_MODULE_SALIENCE: return "SALIENCE";
        case BIO_MODULE_ATTENTION: return "ATTENTION";
        case BIO_MODULE_STDP: return "STDP";
        case BIO_MODULE_ASTROCYTE: return "ASTROCYTE";
        case BIO_MODULE_PIPELINE: return "PIPELINE";
        case BIO_MODULE_TRAINING: return "TRAINING";
        case BIO_MODULE_SYSTEM: return "SYSTEM";
        default: return "UNKNOWN";
    }
}


/**
 * @brief Subscribe to a message type on a bio-async context
 *
 * WHAT: Subscribe to receive messages of a specific type
 * WHY:  Enables modules to receive bio-async messages for inter-module communication
 * HOW:  Add subscription to global registry with thread-safe mutex protection
 *
 * @param ctx Bio-async context (module context pointer)
 * @param msg_type Message type to subscribe to (BIO_MSG_* constant)
 * @return true on success, false if subscription limit reached or invalid params
 */
bool bio_router_subscribe(void* ctx, uint32_t msg_type) {
    /* WHAT: Guard clause - validate parameters */
    if (ctx == NULL) {
        LOG_WARNING("bio_router_subscribe: NULL context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "bio_router_subscribe: validation failed");
        return false;
    }

    /* WHAT: Initialize subscription system on first use (thread-safe via pthread_once) */
    nimcp_platform_once(&g_subscription_once, subscription_init);

    /* WHAT: Acquire lock for thread-safe modification */
    nimcp_platform_mutex_lock(&g_subscription_mutex);

    /* WHAT: Check subscription limit */
    if (g_subscription_count >= MAX_SUBSCRIPTIONS) {
        nimcp_platform_mutex_unlock(&g_subscription_mutex);
        LOG_ERROR("bio_router_subscribe: subscription limit reached (%u)",
                  MAX_SUBSCRIPTIONS);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OUT_OF_RANGE, "bio_router_subscribe: capacity exceeded");
        return false;
    }

    /* WHAT: Check for duplicate subscription */
    for (uint32_t i = 0; i < g_subscription_count; i++) {
        if (g_subscriptions[i].active &&
            g_subscriptions[i].msg_type == msg_type &&
            g_subscriptions[i].user_data == ctx) {
            /* Already subscribed - not an error */
            nimcp_platform_mutex_unlock(&g_subscription_mutex);
            LOG_DEBUG("bio_router_subscribe: already subscribed to 0x%04x", msg_type);
            return true;
        }
    }

    /* WHAT: Find free slot (prefer reusing inactive entries) */
    uint32_t slot = g_subscription_count;
    for (uint32_t i = 0; i < g_subscription_count; i++) {
        if (!g_subscriptions[i].active) {
            slot = i;
            break;
        }
    }

    /* WHAT: Create subscription entry */
    g_subscriptions[slot].msg_type = msg_type;
    g_subscriptions[slot].callback = NULL;  /* Callback set via bio_module_register_handler */
    g_subscriptions[slot].user_data = ctx;
    g_subscriptions[slot].channel = -1;     /* Default channel */
    g_subscriptions[slot].active = true;

    if (slot == g_subscription_count) {
        g_subscription_count++;
    }

    nimcp_platform_mutex_unlock(&g_subscription_mutex);

    LOG_DEBUG("bio_router_subscribe: subscribed to msg_type=0x%04x (slot=%u)",
              msg_type, slot);
    return true;
}


/**
 * @brief Unsubscribe from a message channel
 *
 * WHAT: Remove subscription from bio-async message system
 * WHY:  Clean up subscriptions when module shuts down to prevent dangling callbacks
 * HOW:  Find matching subscription and mark inactive with thread-safe access
 *
 * @param channel Neuromodulator channel (BIO_CHANNEL_*)
 * @param msg_type Message type to unsubscribe from (BIO_MSG_* constant)
 * @param callback Callback function that was registered
 * @param user_data User context data that was provided during subscription
 */
void bio_async_unsubscribe(int channel, uint32_t msg_type, void* callback, void* user_data) {
    /* WHAT: Initialize subscription system on first use (thread-safe via pthread_once) */
    nimcp_platform_once(&g_subscription_once, subscription_init);

    /* WHAT: Acquire lock for thread-safe modification */
    nimcp_platform_mutex_lock(&g_subscription_mutex);

    /* WHAT: Find and deactivate matching subscription */
    bool found = false;
    for (uint32_t i = 0; i < g_subscription_count; i++) {
        if (g_subscriptions[i].active &&
            g_subscriptions[i].msg_type == msg_type &&
            (callback == NULL || g_subscriptions[i].callback == callback) &&
            (user_data == NULL || g_subscriptions[i].user_data == user_data) &&
            (channel < 0 || g_subscriptions[i].channel == channel)) {

            g_subscriptions[i].active = false;
            found = true;
            LOG_DEBUG("bio_async_unsubscribe: removed subscription at slot %u "
                      "(channel=%d, msg_type=0x%04x)", i, channel, msg_type);
            break;
        }
    }

    nimcp_platform_mutex_unlock(&g_subscription_mutex);

    if (!found) {
        LOG_DEBUG("bio_async_unsubscribe: no matching subscription found "
                  "(channel=%d, msg_type=0x%04x)", channel, msg_type);
    }
}

nimcp_error_t bio_async_connect_immune(void* immune_system) {
    NIMCP_CHECK_THROW(immune_system != NULL, NIMCP_ERROR_NULL_POINTER,
                      "bio_async_connect_immune: NULL immune system");

    NIMCP_CHECK_THROW(g_router && g_router->initialized, NIMCP_ERROR_NOT_INITIALIZED,
                      "bio_async_connect_immune: Router not initialized");

    /* Store immune system reference (simplified implementation) */
    g_router->brain_immune_system = immune_system;

    LOG_INFO("bio_async_connect_immune: Brain immune system connected (simplified)");
    return NIMCP_SUCCESS;
}


/**
 * @brief Broadcast cytokine release via bio-async (stub)
 *
 * WHAT: Send immune cytokine signal to all modules
 * WHY:  Coordinate immune response across system
 * HOW:  Currently a stub - full implementation pending
 */
nimcp_error_t bio_async_broadcast_cytokine(
    uint32_t cytokine_type,
    float concentration,
    uint32_t source_cell
) {
    (void)cytokine_type;
    (void)concentration;
    (void)source_cell;
    /* Stub - full implementation pending bio-async immune integration */
    return NIMCP_SUCCESS;
}


/**
 * @brief Send inflammation alert as high-priority message (stub)
 *
 * WHAT: Send inflammation escalation alert
 * WHY:  Notify system of immune response escalation
 * HOW:  Currently a stub - full implementation pending
 */
nimcp_error_t bio_async_inflammation_alert(
    uint32_t region_id,
    uint32_t severity,
    uint32_t antigen_id
) {
    (void)region_id;
    (void)severity;
    (void)antigen_id;
    /* Stub - full implementation pending bio-async immune integration */
    return NIMCP_SUCCESS;
}


/**
 * @brief Notify immune phase change (stub)
 *
 * WHAT: Broadcast immune system phase transition
 * WHY:  Coordinate system-wide immune state awareness
 * HOW:  Currently a stub - full implementation pending
 */
nimcp_error_t bio_async_immune_phase_change(
    uint32_t old_phase,
    uint32_t new_phase
) {
    (void)old_phase;
    (void)new_phase;
    /* Stub - full implementation pending bio-async immune integration */
    return NIMCP_SUCCESS;
}


/**
 * @brief Register with BBB for emotion queries
 *
 * WHAT: Register a module with the Blood-Brain Barrier for emotion-related queries
 * WHY:  Security validation ensures only authorized modules can access emotion state
 * HOW:  Add module to registration table with mutex protection for thread safety
 *
 * @param system System/module context pointer
 * @param module_name Human-readable module name for logging and auditing
 */
void bbb_register_emotion_query(void* system, const char* module_name) {
    /* WHAT: Guard clause - validate system pointer */
    if (system == NULL) {
        LOG_WARNING("bbb_register_emotion_query: NULL system pointer");
        return;
    }

    /* WHAT: Initialize registration system on first use (thread-safe via pthread_once) */
    nimcp_platform_once(&g_emotion_reg_once, emotion_registration_init);

    /* WHAT: Acquire lock for thread-safe modification */
    nimcp_platform_mutex_lock(&g_emotion_reg_mutex);

    /* WHAT: Check registration limit */
    if (g_emotion_registration_count >= MAX_EMOTION_REGISTRATIONS) {
        nimcp_platform_mutex_unlock(&g_emotion_reg_mutex);
        LOG_ERROR("bbb_register_emotion_query: registration limit reached (%u)",
                  MAX_EMOTION_REGISTRATIONS);
        return;
    }

    /* WHAT: Check for duplicate registration */
    for (uint32_t i = 0; i < g_emotion_registration_count; i++) {
        if (g_emotion_registrations[i].active &&
            g_emotion_registrations[i].system == system) {
            /* Already registered - update name if provided */
            if (module_name != NULL) {
                strncpy(g_emotion_registrations[i].module_name, module_name,
                        sizeof(g_emotion_registrations[i].module_name) - 1);
            }
            nimcp_platform_mutex_unlock(&g_emotion_reg_mutex);
            LOG_DEBUG("bbb_register_emotion_query: updated registration for %s",
                      module_name ? module_name : "unknown");
            return;
        }
    }

    /* WHAT: Find free slot */
    uint32_t slot = g_emotion_registration_count;
    for (uint32_t i = 0; i < g_emotion_registration_count; i++) {
        if (!g_emotion_registrations[i].active) {
            slot = i;
            break;
        }
    }

    /* WHAT: Create registration entry */
    g_emotion_registrations[slot].system = system;
    if (module_name != NULL) {
        strncpy(g_emotion_registrations[slot].module_name, module_name,
                sizeof(g_emotion_registrations[slot].module_name) - 1);
        g_emotion_registrations[slot].module_name[
            sizeof(g_emotion_registrations[slot].module_name) - 1] = '\0';
    } else {
        snprintf(g_emotion_registrations[slot].module_name,
                 sizeof(g_emotion_registrations[slot].module_name),
                 "module_%p", system);
    }
    g_emotion_registrations[slot].query_count = 0;
    g_emotion_registrations[slot].active = true;

    if (slot == g_emotion_registration_count) {
        g_emotion_registration_count++;
    }

    nimcp_platform_mutex_unlock(&g_emotion_reg_mutex);

    LOG_INFO("bbb_register_emotion_query: registered module '%s' (slot=%u)",
             g_emotion_registrations[slot].module_name, slot);
}


/*=============================================================================
 * KNOWLEDGE GRAPH SELF-AWARENESS INTEGRATION
 *============================================================================*/

/**
 * @brief Query self-knowledge from the knowledge graph
 *
 * WHAT: Retrieves structural self-knowledge about the Bio_Router module
 * WHY:  Enables runtime introspection and self-awareness capabilities
 * HOW:  Queries KG for Bio_Router entity and logs observations/relations
 *
 * @param kg Knowledge graph reader handle
 * @return 1 if self-knowledge was found, 0 otherwise
 */
int bio_router_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Bio_Router");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            LOG_DEBUG("Bio_Router self-knowledge: %s", self->observations[i]);
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Bio_Router");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Bio_Router");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
