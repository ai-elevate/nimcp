// nimcp_part_io.c - io functions
// Part of nimcp.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp.c


nimcp_status_t nimcp_brain_save(nimcp_brain_t brain, const char* filepath) {
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    API_CHECK_THROW(filepath, NIMCP_ERROR_NULL_ARG, "Filepath is NULL");
    API_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Call internal brain API
    bool success = brain_save(brain->internal_brain, filepath);

    if (!success) {
        NIMCP_CHECK_THROW(false, NIMCP_ERROR_IO, "Failed to save brain");
    }

    set_error("No error");
    return NIMCP_OK;
}


nimcp_brain_t nimcp_brain_load(const char* filepath) {
    if (!filepath) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_ARG, "Filepath is NULL");
        set_error("Filepath is NULL");
        return NULL;
    }

    // Allocate handle (calloc to zero-init last_loss/last_gradient_norm)
    nimcp_brain_t handle = (nimcp_brain_t)nimcp_calloc(1, sizeof(struct nimcp_brain_handle));
    if (!handle) {
        set_error("Failed to allocate brain handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "handle is NULL");

        return NULL;
    }

    // Load internal brain
    handle->internal_brain = brain_load(filepath);

    if (!handle->internal_brain) {
        set_error("Failed to load brain from file");
        nimcp_free(handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_load: handle->internal_brain is NULL");
        return NULL;
    }

    set_error("No error");
    return handle;
}


//=============================================================================
// Brain Snapshot API Implementation
//=============================================================================

nimcp_status_t nimcp_brain_snapshot_save(
    nimcp_brain_t brain,
    const char* name,
    const char* description)
{
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    API_CHECK_THROW(name, NIMCP_ERROR_NULL_ARG, "Snapshot name is NULL");
    API_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Call internal brain snapshot API
    bool success = brain_save_snapshot(
        brain->internal_brain,
        name,
        description ? description : ""
    );

    if (!success) {
        set_error("Failed to save snapshot");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_SUCCESS;
}


nimcp_status_t nimcp_brain_workspace_read(
    nimcp_brain_t brain,
    float* content,
    uint32_t max_dim,
    uint32_t* actual_dim,
    nimcp_cognitive_module_t* source_module)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_read");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Validate parameters FIRST (before checking subsystem availability)
    NIMCP_CHECK_THROW(content && actual_dim && source_module, NIMCP_ERROR_NULL_ARG,
                      "NULL output parameter provided to workspace_read");
    NIMCP_CHECK_THROW(max_dim != 0, NIMCP_ERROR_INVALID, "Invalid max_dim (0)");

    // Guard: Check if global workspace enabled (after parameter validation)
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Read broadcast
    cognitive_module_t internal_source;
    bool success = global_workspace_read_broadcast(
        workspace, content, max_dim, actual_dim, &internal_source
    );

    if (success) {
        *source_module = (nimcp_cognitive_module_t)internal_source;
        set_error("No error");
        return NIMCP_OK;
    } else {
        set_error("No broadcast available or buffer too small");
        return NIMCP_ERROR;
    }
}
