// nimcp_brain_part_processing.c - processing functions
// Part of nimcp_brain.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp_brain.c


/**
 * @brief Publish brain processing event
 *
 * @param processing_type Type of processing (inference, learning, etc.)
 * @param confidence Processing confidence [0,1]
 */
static void brain_publish_processing_event(const char* processing_type, float confidence)
{
    // Use atomic load for thread-safe access to global state
    if (!__atomic_load_n(&g_brain_bio_initialized, __ATOMIC_ACQUIRE) ||
        !__atomic_load_n(&g_brain_bio_ctx, __ATOMIC_ACQUIRE)) {
        return;  // Graceful degradation
    }

    LOG_MODULE_DEBUG("BRAIN", "Publishing processing event: type=%s, confidence=%.3f",
                     processing_type, confidence);

    // Use predictive coding signal for processing events
    char signal_name[NIMCP_ID_BUFFER_SIZE];
    snprintf(signal_name, sizeof(signal_name), "brain.processing.%s", processing_type);
    bio_router_publish_signal(g_brain_bio_ctx, signal_name, confidence);
}


//=============================================================================
// Brain Resize Helper (called from nimcp_brain_resize.c)
//=============================================================================

/**
 * @brief Internal helper for brain_resize - update subsystems after network swap
 *
 * WHAT: Updates glial integration and other subsystems to reference new network
 * WHY:  brain_resize.c can't access full brain struct, so needs helper with full access
 * HOW:  Destroys/recreates glial integration with new network reference
 *
 * @param brain Brain handle
 * @param new_base_network New neural network after resize
 * @param new_neuron_count New neuron count after resize
 * @return true on success
 */
bool brain_resize_update_subsystems_internal(brain_t brain, neural_network_t new_base_network, uint32_t new_neuron_count)
{
    if (!brain || !new_base_network) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain_resize_update_subsystems_internal: required parameter is NULL (brain, new_base_network)");
        return false;
    }

    // Destroy and recreate glial integration (if glial system exists)
    if (brain->glial) {
        LOG_INFO(LOG_MODULE, "brain_resize: Destroying old glial integration system");

        // Save configuration flags before destroying
        bool enable_glial = brain->config.enable_glial;

        // Destroy entire glial integration (frees all nested structures)
        glial_integration_destroy(brain->glial);
        brain->glial = NULL;

        // Recreate glial integration with new network
        if (enable_glial) {
            LOG_INFO(LOG_MODULE, "brain_resize: Creating new glial integration system for %u neurons", new_neuron_count);

            // Create new glial integration system
            brain->glial = glial_integration_create(new_base_network, 1000);  // 1000 = max_mappings

            if (brain->glial) {
                LOG_INFO(LOG_MODULE, "brain_resize: Glial integration system created successfully");

                // Recreate spatial neuromodulator system
                LOG_INFO(LOG_MODULE, "brain_resize: Creating new spatial neuromodulator system");

                // Enable all neuromodulator types by default
                bool enabled_types[NEUROMOD_COUNT] = {true, true, true, true};
                spatial_neuromod_config_t configs[NEUROMOD_COUNT];

                // Use default configs for all types
                configs[0] = spatial_neuromod_default_config(NEUROMOD_DOPAMINE);
                configs[1] = spatial_neuromod_default_config(NEUROMOD_SEROTONIN);
                configs[2] = spatial_neuromod_default_config(NEUROMOD_ACETYLCHOLINE);
                configs[3] = spatial_neuromod_default_config(NEUROMOD_NOREPINEPHRINE);

                spatial_neuromod_system_t* new_spatial = spatial_neuromod_system_create(new_base_network, enabled_types, configs);
                if (new_spatial) {
                    brain->glial->spatial_neuromod = new_spatial;
                    LOG_INFO(LOG_MODULE, "brain_resize: Spatial neuromodulator system created successfully");
                } else {
                    LOG_WARN(LOG_MODULE, "brain_resize: Failed to create spatial neuromodulator system");
                }
            } else {
                LOG_WARN(LOG_MODULE, "brain_resize: Failed to create glial integration system");
                brain->config.enable_glial = false;
            }
        }
    }

    // Update brain oscillations network reference (if oscillations exist)
    // Brain oscillations doesn't store network directly, it queries from brain
    // No update needed

    return true;
}
