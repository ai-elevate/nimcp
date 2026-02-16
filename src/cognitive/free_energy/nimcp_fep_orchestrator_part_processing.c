// nimcp_fep_orchestrator_part_processing.c - processing functions
// Part of nimcp_fep_orchestrator.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp_fep_orchestrator.c


/**
 * @brief Check if category is due for update
 */
static bool category_needs_update(
    const fep_orchestrator_t* orchestrator,
    fep_bridge_category_t category,
    uint64_t current_time_ms
) {
    const fep_category_config_t* cat_cfg = &orchestrator->config.categories[category];
    if (!cat_cfg->enabled) {
        return false;
    }
    
    uint64_t elapsed = current_time_ms - cat_cfg->last_update_time;
    return elapsed >= cat_cfg->update_interval_ms;
}


/* ============================================================================
 * Update API Implementation
 * ============================================================================ */

int fep_orchestrator_update(
    fep_orchestrator_t* orchestrator,
    uint64_t current_time_ms
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_update", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->state != FEP_ORCHESTRATOR_RUNNING) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return 0;
    }
    
    uint64_t cycle_start_us = nimcp_platform_time_monotonic_us();
    int total_updated = 0;
    
    /* Process bio-async inbox if connected */
    if (orchestrator->bio_async_connected && orchestrator->bio_context) {
        uint32_t processed = bio_router_process_inbox(orchestrator->bio_context, 10);
        orchestrator->stats.bio_async_messages_received += processed;
    }
    
    /* Update each category that is due */
    for (int cat = 0; cat < FEP_BRIDGE_CATEGORY_COUNT; cat++) {
        /* Phase 8: Loop progress heartbeat */
        if ((cat & 0xFF) == 0 && FEP_BRIDGE_CATEGORY_COUNT > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(cat + 1) / (float)FEP_BRIDGE_CATEGORY_COUNT);
        }

        if (!category_needs_update(orchestrator, (fep_bridge_category_t)cat, current_time_ms)) {
            continue;
        }
        
        /* Update all bridges in this category */
        for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
            /* Phase 8: Loop progress heartbeat */
            if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
                fep_orchestrator_heartbeat("fep_orchestr_loop",
                                 (float)(i + 1) / (float)orchestrator->bridge_count);
            }

            fep_bridge_entry_t* entry = &orchestrator->bridges[i];
            if (entry->category != cat) continue;
            
            int result = update_single_bridge(orchestrator, entry, current_time_ms);
            if (result > 0) {
                total_updated++;
                orchestrator->stats.total_bridge_updates++;
                orchestrator->stats.categories[cat].total_updates++;
            }
        }
        
        /* Update category last update time */
        orchestrator->config.categories[cat].last_update_time = current_time_ms;
    }
    
    /* Update orchestrator statistics */
    uint64_t cycle_time_us = nimcp_platform_time_monotonic_us() - cycle_start_us;
    orchestrator->stats.total_update_cycles++;
    
    float cycle_time_ms = (float)cycle_time_us / 1000.0f;
    if (orchestrator->stats.max_cycle_time_us < (float)cycle_time_us) {
        orchestrator->stats.max_cycle_time_us = (float)cycle_time_us;
    }
    
    /* Rolling average */
    uint64_t n = orchestrator->stats.total_update_cycles;
    orchestrator->stats.avg_cycle_time_us = 
        (orchestrator->stats.avg_cycle_time_us * (n - 1) + (float)cycle_time_us) / n;
    
    /* Check for time budget overrun */
    if (orchestrator->config.update_time_budget_ms > 0 &&
        cycle_time_ms > orchestrator->config.update_time_budget_ms) {
        orchestrator->stats.overrun_count++;
    }
    
    orchestrator->last_update_time = current_time_ms;
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return total_updated;
}


int fep_orchestrator_update_category(
    fep_orchestrator_t* orchestrator,
    fep_bridge_category_t category,
    uint64_t current_time_ms
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_update_category", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    NIMCP_CHECK_THROW(category < FEP_BRIDGE_CATEGORY_COUNT, NIMCP_ERROR_INVALID_PARAM, "invalid bridge category");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    if (orchestrator->state != FEP_ORCHESTRATOR_RUNNING) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return 0;
    }
    
    int updated = 0;
    for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)orchestrator->bridge_count);
        }

        fep_bridge_entry_t* entry = &orchestrator->bridges[i];
        if (entry->category != category) continue;
        
        int result = update_single_bridge(orchestrator, entry, current_time_ms);
        if (result > 0) {
            updated++;
            orchestrator->stats.total_bridge_updates++;
            orchestrator->stats.categories[category].total_updates++;
        }
    }
    
    orchestrator->config.categories[category].last_update_time = current_time_ms;
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return updated;
}


int fep_orchestrator_update_bridge(
    fep_orchestrator_t* orchestrator,
    uint32_t bridge_id
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_update_bridge", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    
    fep_bridge_entry_t* entry = find_bridge_by_id(orchestrator, bridge_id);
    if (!entry) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return NIMCP_ERROR_INVALID_PARAM;
    }
    
    uint64_t current_time_ms = nimcp_platform_time_monotonic_ms();
    int result = update_single_bridge(orchestrator, entry, current_time_ms);
    
    if (result > 0) {
        orchestrator->stats.total_bridge_updates++;
        orchestrator->stats.categories[entry->category].total_updates++;
    }
    
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return result > 0 ? 0 : NIMCP_ERROR_OPERATION_FAILED;
}


int fep_orchestrator_force_update_all(fep_orchestrator_t* orchestrator) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_force_update_all", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");

    nimcp_platform_mutex_lock(orchestrator->mutex);

    /* Check if orchestrator is in a state that allows updates */
    if (orchestrator->state != FEP_ORCHESTRATOR_RUNNING) {
        nimcp_platform_mutex_unlock(orchestrator->mutex);
        return 0;  /* No updates when not running */
    }

    uint64_t cycle_start_us = nimcp_platform_time_monotonic_us();
    uint64_t current_time_ms = nimcp_platform_time_monotonic_ms();
    int updated = 0;

    for (uint32_t i = 0; i < orchestrator->bridge_count; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && orchestrator->bridge_count > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)orchestrator->bridge_count);
        }

        fep_bridge_entry_t* entry = &orchestrator->bridges[i];
        int result = update_single_bridge(orchestrator, entry, current_time_ms);
        if (result > 0) {
            updated++;
            orchestrator->stats.total_bridge_updates++;
            orchestrator->stats.categories[entry->category].total_updates++;
        }
    }

    /* Reset all category timers */
    for (int i = 0; i < FEP_BRIDGE_CATEGORY_COUNT; i++) {
        /* Phase 8: Loop progress heartbeat */
        if ((i & 0xFF) == 0 && FEP_BRIDGE_CATEGORY_COUNT > 256) {
            fep_orchestrator_heartbeat("fep_orchestr_loop",
                             (float)(i + 1) / (float)FEP_BRIDGE_CATEGORY_COUNT);
        }

        orchestrator->config.categories[i].last_update_time = current_time_ms;
    }

    /* Track cycle time statistics */
    uint64_t cycle_time_us = nimcp_platform_time_monotonic_us() - cycle_start_us;
    uint32_t n = orchestrator->stats.total_update_cycles + 1;
    orchestrator->stats.total_update_cycles = n;
    orchestrator->stats.avg_cycle_time_us =
        (orchestrator->stats.avg_cycle_time_us * (n - 1) + (float)cycle_time_us) / n;
    if ((float)cycle_time_us > orchestrator->stats.max_cycle_time_us) {
        orchestrator->stats.max_cycle_time_us = (float)cycle_time_us;
    }

    nimcp_platform_mutex_unlock(orchestrator->mutex);
    return updated;
}


/* ============================================================================
 * Category Configuration API Implementation
 * ============================================================================ */

int fep_orchestrator_set_update_interval(
    fep_orchestrator_t* orchestrator,
    fep_bridge_category_t category,
    uint64_t interval_ms
) {
    /* Phase 8: Heartbeat at operation start */
    fep_orchestrator_heartbeat("fep_orchestr_set_update_interval", 0.0f);


    NIMCP_CHECK_THROW(orchestrator, NIMCP_ERROR_NULL_POINTER, "orchestrator is NULL");
    NIMCP_CHECK_THROW(category < FEP_BRIDGE_CATEGORY_COUNT, NIMCP_ERROR_INVALID_PARAM, "invalid bridge category");
    
    nimcp_platform_mutex_lock(orchestrator->mutex);
    orchestrator->config.categories[category].update_interval_ms = interval_ms;
    nimcp_platform_mutex_unlock(orchestrator->mutex);
    
    return 0;
}


int fep_orchestrator_training_step(void* instance, float progress) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "fep_orchestrator_training_step: NULL argument");
        return -1;
    }
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    fep_orchestrator_heartbeat_instance(NULL, "fep_orchestrator_training_step", progress);
    return 0;
}
