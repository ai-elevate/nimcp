// nimcp_language_orchestrator_part_core.c - core functions
// Part of nimcp_language_orchestrator.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp_language_orchestrator.c


int language_orchestrator_start(language_orchestrator_t* orchestrator)
{
    if (!orchestrator || !orchestrator->initialized) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_start: NULL or uninitialized orchestrator");
        return -1;
    }

    if (orchestrator->running) {
        return 0;  /* Already running */
    }

    /* Reset state */
    orchestrator->state = LANGUAGE_STATE_IDLE;
    orchestrator->state_entry_time_ms = 0;

    orchestrator->running = true;

    /* Start bridges if connected */
    if (orchestrator->perception_bridge) {
        language_perception_bridge_start(orchestrator->perception_bridge);
    }
    if (orchestrator->cognitive_bridge) {
        language_cognitive_bridge_start(orchestrator->cognitive_bridge);
    }
    if (orchestrator->training_bridge) {
        language_training_bridge_start(orchestrator->training_bridge);
    }
    if (orchestrator->omni_bridge) {
        language_omni_bridge_start(orchestrator->omni_bridge);
    }
    if (orchestrator->immune_bridge) {
        language_immune_bridge_start(orchestrator->immune_bridge);
    }
    if (orchestrator->gpu_bridge) {
        language_gpu_bridge_start(orchestrator->gpu_bridge);
    }

    return 0;
}


int language_orchestrator_stop(language_orchestrator_t* orchestrator)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_stop: NULL orchestrator");
        return -1;
    }

    if (!orchestrator->running) {
        return 0;  /* Already stopped */
    }

    /* Stop bridges */
    if (orchestrator->perception_bridge) {
        language_perception_bridge_stop(orchestrator->perception_bridge);
    }
    if (orchestrator->cognitive_bridge) {
        language_cognitive_bridge_stop(orchestrator->cognitive_bridge);
    }
    if (orchestrator->training_bridge) {
        language_training_bridge_stop(orchestrator->training_bridge);
    }
    if (orchestrator->omni_bridge) {
        language_omni_bridge_stop(orchestrator->omni_bridge);
    }
    if (orchestrator->immune_bridge) {
        language_immune_bridge_stop(orchestrator->immune_bridge);
    }
    if (orchestrator->gpu_bridge) {
        language_gpu_bridge_stop(orchestrator->gpu_bridge);
    }

    orchestrator->running = false;
    orchestrator->state = LANGUAGE_STATE_IDLE;

    return 0;
}


//=============================================================================
// Subsystem Connection API
//=============================================================================

int language_orchestrator_connect_wernicke(
    language_orchestrator_t* orchestrator,
    wernicke_adapter_t* wernicke)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_wernicke: NULL orchestrator");
        return -1;
    }

    orchestrator->wernicke = wernicke;
    orchestrator->stats.wernicke_connected = (wernicke != NULL);

    return 0;
}


int language_orchestrator_connect_broca(
    language_orchestrator_t* orchestrator,
    broca_adapter_t* broca)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_broca: NULL orchestrator");
        return -1;
    }

    orchestrator->broca = broca;
    orchestrator->stats.broca_connected = (broca != NULL);

    return 0;
}


int language_orchestrator_connect_nlp(
    language_orchestrator_t* orchestrator,
    nlp_network_t nlp)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_nlp: NULL orchestrator");
        return -1;
    }

    orchestrator->nlp_network = nlp;
    orchestrator->stats.nlp_connected = (nlp != NULL);

    return 0;
}


int language_orchestrator_connect_speech_cortex(
    language_orchestrator_t* orchestrator,
    speech_cortex_t* speech)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_speech_cortex: NULL orchestrator");
        return -1;
    }

    orchestrator->speech_cortex = speech;

    return 0;
}


int language_orchestrator_connect_multimodal(
    language_orchestrator_t* orchestrator,
    multimodal_nlp_bridge_t* multimodal)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_multimodal: NULL orchestrator");
        return -1;
    }

    orchestrator->multimodal = multimodal;

    return 0;
}


//=============================================================================
// Bridge Connection API
//=============================================================================

int language_orchestrator_connect_perception_bridge(
    language_orchestrator_t* orchestrator,
    language_perception_bridge_t* bridge)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_perception_bridge: NULL orchestrator");
        return -1;
    }

    orchestrator->perception_bridge = bridge;
    orchestrator->stats.perception_bridge_connected = (bridge != NULL);

    /* Connect bridge back to orchestrator */
    if (bridge) {
        language_perception_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}


int language_orchestrator_connect_cognitive_bridge(
    language_orchestrator_t* orchestrator,
    language_cognitive_bridge_t* bridge)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_cognitive_bridge: NULL orchestrator");
        return -1;
    }

    orchestrator->cognitive_bridge = bridge;
    orchestrator->stats.cognitive_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_cognitive_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}


int language_orchestrator_connect_training_bridge(
    language_orchestrator_t* orchestrator,
    language_training_bridge_t* bridge)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_training_bridge: NULL orchestrator");
        return -1;
    }

    orchestrator->training_bridge = bridge;
    orchestrator->stats.training_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_training_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}


int language_orchestrator_connect_omni_bridge(
    language_orchestrator_t* orchestrator,
    language_omni_bridge_t* bridge)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_omni_bridge: NULL orchestrator");
        return -1;
    }

    orchestrator->omni_bridge = bridge;
    orchestrator->stats.omni_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_omni_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}


int language_orchestrator_connect_immune_bridge(
    language_orchestrator_t* orchestrator,
    language_immune_bridge_t* bridge)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_immune_bridge: NULL orchestrator");
        return -1;
    }

    orchestrator->immune_bridge = bridge;
    orchestrator->stats.immune_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_immune_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}


int language_orchestrator_connect_gpu_bridge(
    language_orchestrator_t* orchestrator,
    language_gpu_bridge_t* bridge)
{
    if (!orchestrator) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_connect_gpu_bridge: NULL orchestrator");
        return -1;
    }

    orchestrator->gpu_bridge = bridge;
    orchestrator->stats.gpu_bridge_connected = (bridge != NULL);

    if (bridge) {
        language_gpu_bridge_connect_orchestrator(bridge, orchestrator);
    }

    return 0;
}


int language_orchestrator_generate_output(
    language_orchestrator_t* orchestrator,
    const float* semantic_input,
    uint32_t semantic_dim,
    void* output,
    uint32_t max_output,
    uint32_t* output_size,
    language_output_type_t output_type)
{
    if (!orchestrator || !orchestrator->running || !semantic_input) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_generate_output: Invalid parameters");
        return -1;
    }

    /* Store semantic input for production */
    orchestrator->current_production.semantic_input = (float*)semantic_input;
    orchestrator->current_production.semantic_dim = semantic_dim;

    /* Transition to generating state */
    orchestrator_transition_state(orchestrator,
        LANGUAGE_STATE_GENERATING,
        orchestrator->last_update_ms);

    /* Production would be handled by Broca adapter */
    /* For now, mark as pending */

    if (output_size) {
        *output_size = 0;  /* Will be filled when production completes */
    }

    return 0;
}


//=============================================================================
// Event API
//=============================================================================

int language_orchestrator_register_callback(
    language_orchestrator_t* orchestrator,
    language_event_callback_t callback,
    void* user_data)
{
    if (!orchestrator || !callback) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_register_callback: NULL parameter");
        return -1;
    }

    /* Find empty slot */
    for (uint32_t i = 0; i < ORCHESTRATOR_MAX_CALLBACKS; i++) {
        if (!orchestrator->callbacks[i].active) {
            orchestrator->callbacks[i].callback = callback;
            orchestrator->callbacks[i].user_data = user_data;
            orchestrator->callbacks[i].active = true;
            orchestrator->num_callbacks++;
            return 0;
        }
    }

    return -1;  /* No callback slots available - normal capacity behavior */
}


int language_orchestrator_unregister_callback(
    language_orchestrator_t* orchestrator,
    language_event_callback_t callback)
{
    if (!orchestrator || !callback) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "language_orchestrator_unregister_callback: NULL parameter");
        return -1;
    }

    for (uint32_t i = 0; i < ORCHESTRATOR_MAX_CALLBACKS; i++) {
        if (orchestrator->callbacks[i].active &&
            orchestrator->callbacks[i].callback == callback) {
            orchestrator->callbacks[i].active = false;
            orchestrator->callbacks[i].callback = NULL;
            orchestrator->callbacks[i].user_data = NULL;
            orchestrator->num_callbacks--;
            return 0;
        }
    }

    return -1;  /* Callback not found - normal lookup behavior */
}
