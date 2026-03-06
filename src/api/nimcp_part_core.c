// nimcp_part_core.c - core functions
// Part of nimcp.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp.c


//=============================================================================
// Version Functions
//=============================================================================

const char* nimcp_version(void) {
    return NIMCP_VERSION_STRING;
}


int nimcp_version_int(void) {
    return NIMCP_VERSION_MAJOR * 10000 + NIMCP_VERSION_MINOR * 100 + NIMCP_VERSION_PATCH;
}


nimcp_status_t nimcp_brain_learn_example(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const char* label,
    float confidence)
{
    LOG_DEBUG("Learning example: label='%s', num_features=%u, confidence=%.3f",
              label ? label : "NULL", num_features, confidence);

    /* Exception-integrated parameter validation */
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_API_CHECK_NULL(label, NIMCP_ERROR_NULL_ARG, "Label is NULL");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    /* I-H1 FIX: Validate input dimensions at API layer */
    if (num_features == 0) {
        set_error("num_features must be > 0");
        return NIMCP_ERROR_INVALID;
    }

    /* === PHASE IS-1: BBB INPUT VALIDATION === */
    /* Validate external input data through Blood-Brain Barrier before processing */
    /* Note: brain->internal_brain already validated non-NULL above */
    if (brain->internal_brain->bbb_enabled &&
        brain->internal_brain->bbb_system) {
        LOG_DEBUG("BBB enabled, validating inputs");
        bbb_validation_result_t result;

        /* Validate features array (external input data) */
        if (!bbb_validate_input(brain->internal_brain->bbb_system,
                               features, num_features * sizeof(float), &result)) {
            NIMCP_API_CHECK_BBB(false, result, NIMCP_ERROR_INVALID);
        }

        /* Validate label string (external string input) */
        if (!bbb_validate_string(brain->internal_brain->bbb_system, label, &result)) {
            NIMCP_API_CHECK_BBB(false, result, NIMCP_ERROR_INVALID);
        }
        LOG_DEBUG("BBB validation passed");
    }

    /* Call internal brain API */
    LOG_DEBUG("Invoking internal brain_learn_example");
    float loss = brain_learn_example(brain->internal_brain, features, num_features, label, confidence);

    /* brain_learn_example returns -1.0f on error, >= 0.0f on success */
    NIMCP_API_CHECK_FLOAT(loss, NIMCP_ERROR,
                          "Brain learning failed for label");

    /* Store loss for retrieval via nimcp_brain_get_last_loss() */
    brain->last_loss = loss;

    /* Store gradient norm from the adaptive network */
    if (brain->internal_brain && brain->internal_brain->network) {
        brain->last_gradient_norm = adaptive_network_get_last_grad_norm(brain->internal_brain->network);
    } else {
        brain->last_gradient_norm = 0.0f;
    }

    set_error("No error");
    LOG_DEBUG("Learning example completed successfully (loss=%.6f)", loss);
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_learn_vector(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const float* target,
    uint32_t target_size,
    const char* label,
    float confidence)
{
    LOG_DEBUG("Learning vector: target_size=%u, num_features=%u, confidence=%.3f",
              target_size, num_features, confidence);

    /* Exception-integrated parameter validation */
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_API_CHECK_NULL(target, NIMCP_ERROR_NULL_ARG, "Target array is NULL");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    if (num_features == 0) {
        set_error("num_features must be > 0");
        return NIMCP_ERROR_INVALID;
    }
    if (target_size == 0) {
        set_error("target_size must be > 0");
        return NIMCP_ERROR_INVALID;
    }

    /* === BBB INPUT VALIDATION === */
    if (brain->internal_brain->bbb_enabled &&
        brain->internal_brain->bbb_system) {
        bbb_validation_result_t result;

        /* Validate features array */
        if (!bbb_validate_input(brain->internal_brain->bbb_system,
                               features, num_features * sizeof(float), &result)) {
            NIMCP_API_CHECK_BBB(false, result, NIMCP_ERROR_INVALID);
        }

        /* Validate target array */
        if (!bbb_validate_input(brain->internal_brain->bbb_system,
                               target, target_size * sizeof(float), &result)) {
            NIMCP_API_CHECK_BBB(false, result, NIMCP_ERROR_INVALID);
        }

        /* Validate label string if provided */
        if (label && !bbb_validate_string(brain->internal_brain->bbb_system, label, &result)) {
            NIMCP_API_CHECK_BBB(false, result, NIMCP_ERROR_INVALID);
        }
    }

    /* Call internal brain API */
    float loss = brain_learn_vector(brain->internal_brain, features, num_features,
                                    target, target_size, label, confidence);

    NIMCP_API_CHECK_FLOAT(loss, NIMCP_ERROR,
                          "Brain vector learning failed");

    /* Store loss and gradient norm */
    brain->last_loss = loss;
    if (brain->internal_brain && brain->internal_brain->network) {
        brain->last_gradient_norm = adaptive_network_get_last_grad_norm(brain->internal_brain->network);
    } else {
        brain->last_gradient_norm = 0.0f;
    }

    set_error("No error");
    LOG_DEBUG("Learning vector completed successfully (loss=%.6f)", loss);
    return NIMCP_OK;
}


float nimcp_brain_get_last_loss(nimcp_brain_t brain)
{
    if (!brain) return -1.0F;
    return brain->last_loss;
}


float nimcp_brain_get_last_gradient_norm(nimcp_brain_t brain)
{
    if (!brain) return -1.0f;
    return brain->last_gradient_norm;
}


float nimcp_brain_get_accuracy(nimcp_brain_t brain)
{
    if (!brain || !brain->internal_brain) return 0.0F;
    return brain->internal_brain->stats.running_accuracy;
}


nimcp_status_t nimcp_brain_learn_batch(
    nimcp_brain_t brain,
    const float** features_array,
    const uint32_t* num_features_array,
    const char** labels,
    const float* confidences,
    uint32_t num_examples,
    float* losses_out)
{
    LOG_DEBUG("Learning batch: num_examples=%u", num_examples);

    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(features_array, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    NIMCP_API_CHECK_NULL(labels, NIMCP_ERROR_NULL_ARG, "Labels array is NULL");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    if (num_examples == 0) {
        set_error("num_examples must be > 0");
        return NIMCP_ERROR_INVALID;
    }

    /* BBB validation on the entire batch */
    if (brain->internal_brain->bbb_enabled &&
        brain->internal_brain->bbb_system) {
        bbb_validation_result_t result;
        for (uint32_t i = 0; i < num_examples; i++) {
            if (!features_array[i] || !labels[i]) continue;
            uint32_t nf = num_features_array ? num_features_array[i] : 0;
            if (!bbb_validate_input(brain->internal_brain->bbb_system,
                                   features_array[i], nf * sizeof(float), &result)) {
                NIMCP_API_CHECK_BBB(false, result, NIMCP_ERROR_INVALID);
            }
            if (!bbb_validate_string(brain->internal_brain->bbb_system, labels[i], &result)) {
                NIMCP_API_CHECK_BBB(false, result, NIMCP_ERROR_INVALID);
            }
        }
    }

    /* Build brain_example_t array and call internal batch learn */
    brain_example_t* examples = (brain_example_t*)nimcp_calloc(num_examples, sizeof(brain_example_t));
    if (!examples) {
        set_error("Failed to allocate batch example array");
        return NIMCP_ERROR_OUT_OF_MEMORY;
    }

    for (uint32_t i = 0; i < num_examples; i++) {
        examples[i].features = (float*)features_array[i];
        examples[i].num_features = num_features_array ? num_features_array[i] : 0;
        strncpy(examples[i].label, labels[i] ? labels[i] : "", 63);
        examples[i].label[63] = '\0';
        examples[i].confidence = confidences ? confidences[i] : 1.0f;
    }

    float avg_loss = brain_learn_batch_detailed(brain->internal_brain, examples,
                                                 num_examples, losses_out);

    nimcp_free(examples);

    if (avg_loss < 0.0f) {
        set_error("Batch learning failed");
        return NIMCP_ERROR;
    }

    brain->last_loss = avg_loss;

    /* Update gradient norm from last example in batch */
    if (brain->internal_brain && brain->internal_brain->network) {
        brain->last_gradient_norm = adaptive_network_get_last_grad_norm(brain->internal_brain->network);
    }

    LOG_DEBUG("Batch learning completed: avg_loss=%.6f", avg_loss);
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_predict(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    char* out_label,
    float* out_confidence)
{
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain handle");
    API_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    API_CHECK_THROW(out_label, NIMCP_ERROR_NULL_ARG, "Output label buffer is NULL");
    API_CHECK_THROW(out_confidence, NIMCP_ERROR_NULL_ARG, "Output confidence pointer is NULL");
    API_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // I-H1 FIX: Validate input dimensions
    if (num_features == 0) {
        set_error("num_features must be > 0");
        return NIMCP_ERROR_INVALID;
    }
    if (brain->internal_brain->config.num_inputs > 0 &&
        num_features != brain->internal_brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u",
                  brain->internal_brain->config.num_inputs, num_features);
        return NIMCP_ERROR_INVALID;
    }

    // === PHASE IS-1: BBB INPUT VALIDATION ===
    // Validate external input data through Blood-Brain Barrier before processing
    if (brain->internal_brain->bbb_enabled &&
        brain->internal_brain->bbb_system) {
        bbb_validation_result_t result;

        // Validate features array (external input data)
        if (!bbb_validate_input(brain->internal_brain->bbb_system,
                               features, num_features * sizeof(float), &result)) {
            NIMCP_CHECK_THROW(false, NIMCP_ERROR_INVALID, "BBB rejected features: %s", result.reason);
        }
    }

    // Call internal brain API
    brain_decision_t* decision = brain_decide(brain->internal_brain, features, num_features);

    if (!decision) {
        NIMCP_CHECK_THROW(false, NIMCP_ERROR, "Brain prediction failed");
    }

    /* P1-48: Use NIMCP_MAX_LABEL_SIZE instead of hardcoded 63 */
    strncpy(out_label, decision->label, NIMCP_MAX_LABEL_SIZE - 1);
    out_label[NIMCP_MAX_LABEL_SIZE - 1] = '\0';
    *out_confidence = decision->confidence;

    // Deep-copy decision for rubric access (last_decision owns this copy)
    // Note: internal_brain already validated non-NULL above
    brain_decision_t* deep_copy = copy_decision_deep(decision);
    if (deep_copy) {
        if (brain->internal_brain->last_decision) {
            brain_free_decision(brain->internal_brain->last_decision);
        }
        brain->internal_brain->last_decision = deep_copy;
    }
    // If copy_decision_deep fails (OOM), keep the old last_decision intact

    // Free original decision (caller's copy)
    brain_free_decision(decision);

    set_error("No error");
    return NIMCP_OK;
}


/**
 * @brief Fast prediction (no decision caching, no mirror neuron integration)
 *
 * I-M4: THREAD SAFETY CONTRACT
 * This function is NOT thread-safe for concurrent calls on the same brain instance.
 * forward_raw mutates neuron states (spike encoding + weight reads). Callers must
 * provide external synchronization (e.g., Python ThreadSafeBrain uses RLock).
 * Concurrent reads on different brain instances are safe.
 *
 * @param brain Brain handle
 * @param features Input feature vector
 * @param num_features Number of features (must match brain->config.num_inputs)
 * @param out_label Output buffer for predicted label (at least NIMCP_MAX_LABEL_SIZE bytes)
 * @param out_confidence Output confidence [0.0, 1.0]
 * @return NIMCP_OK on success, error code on failure
 */
nimcp_status_t nimcp_brain_predict_fast(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    char* out_label,
    float* out_confidence)
{
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain handle");
    API_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    API_CHECK_THROW(out_label, NIMCP_ERROR_NULL_ARG, "Output label buffer is NULL");
    API_CHECK_THROW(out_confidence, NIMCP_ERROR_NULL_ARG, "Output confidence pointer is NULL");

    brain_t ib = brain->internal_brain;
    if (!ib || !ib->network) {
        // I-L2: Consistent error logging for all error paths
        LOG_ERROR("predict_fast: brain not initialized (ib=%p, network=%p)",
                  (void*)ib, ib ? (void*)ib->network : NULL);
        set_error("Brain not initialized");
        return NIMCP_ERROR;
    }

    // I-H1 FIX: Validate input dimensions to prevent out-of-bounds reads in forward pass.
    // Check both zero and mismatch against brain's expected input size.
    if (num_features == 0) {
        LOG_ERROR("predict_fast: num_features must be > 0");
        set_error("num_features must be > 0");
        return NIMCP_ERROR_INVALID;
    }
    if (ib->config.num_inputs > 0 && num_features != ib->config.num_inputs) {
        LOG_ERROR("predict_fast: feature count mismatch: expected %u, got %u",
                  ib->config.num_inputs, num_features);
        set_error("Feature count mismatch: expected %u, got %u",
                  ib->config.num_inputs, num_features);
        return NIMCP_ERROR_INVALID;
    }

    // Allocate output buffer on stack for small outputs, heap for large
    uint32_t num_outputs = ib->config.num_outputs;
    if (num_outputs == 0) {
        LOG_ERROR("predict_fast: brain has 0 outputs");
        set_error("Brain has 0 outputs");
        return NIMCP_ERROR_INVALID;
    }
    // W6-6 FIX: Zero-initialize stack buffer. If forward_raw partially writes
    // (e.g., writes fewer outputs than num_outputs), argmax would read garbage
    // from uninitialized slots, potentially selecting a wrong output index.
    float stack_buf[4096] = {0};
    float* output = (num_outputs <= 4096) ? stack_buf : (float*)nimcp_calloc(num_outputs, sizeof(float));
    if (!output) {
        set_error("Failed to allocate output buffer");
        return NIMCP_ERROR_MEMORY;
    }

    // I-C1: Use forward_raw (no output thresholding) for classification.
    // adaptive_network_forward() applies adaptive thresholding that zeros below-threshold
    // outputs, collapsing all predictions to the same class. forward_raw preserves the
    // raw network activations needed for accurate argmax classification.
    // I-H2: Note -- forward_raw mutates neuron states (spike encoding + weight reads).
    // Concurrent predict_fast calls from multiple threads require external synchronization.
    // W6-9 FIX: Store return value (active neuron count) for diagnostics.
    // forward_raw returning 0 active neurons with valid outputs is normal for sparse
    // networks, so we only log it at debug level rather than treating it as an error.
    uint32_t active_count = adaptive_network_forward_raw(ib->network, features, num_features,
                                 output, num_outputs);
    if (active_count == 0) {
        LOG_DEBUG("predict_fast: forward_raw returned 0 active neurons");
    }

    // I-H3 FIX: NaN-safe argmax. If output contains NaN values, standard comparison
    // (NaN > max_val) is always false, silently picking index 0. Use isfinite() to
    // skip NaN/Inf values and detect all-NaN output as an error condition.
    uint32_t max_idx = 0;
    float max_val = -FLT_MAX;
    bool has_valid_output = false;
    for (uint32_t i = 0; i < num_outputs; i++) {
        if (isfinite(output[i]) && output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
            has_valid_output = true;
        }
    }

    // If all outputs are NaN/Inf, return error with zero confidence
    if (!has_valid_output) {
        LOG_WARN("predict_fast: all %u outputs are NaN/Inf, returning confidence 0", num_outputs);
        snprintf(out_label, NIMCP_MAX_LABEL_SIZE, "UNKNOWN");
        out_label[NIMCP_MAX_LABEL_SIZE - 1] = '\0';
        *out_confidence = 0.0f;
        if (output != stack_buf) nimcp_free(output);
        set_error("All network outputs are NaN/Inf");
        return NIMCP_OK;  // Not a system error, just degenerate output
    }

    // Map to label: use output_labels if available, else "output_N"
    // H1-FIX: Check max_idx < num_output_labels (not num_outputs). The output layer
    // has num_outputs neurons but only num_output_labels label mappings are populated.
    // Using num_outputs could access uninitialized entries beyond num_output_labels.
    if (ib->output_labels && max_idx < ib->num_output_labels && ib->output_labels[max_idx]) {
        strncpy(out_label, ib->output_labels[max_idx], NIMCP_MAX_LABEL_SIZE - 1);
    } else {
        snprintf(out_label, NIMCP_MAX_LABEL_SIZE, "output_%u", max_idx);
    }
    out_label[NIMCP_MAX_LABEL_SIZE - 1] = '\0';

    // I-M2: Confidence uses max/sum_abs normalization (not softmax). This is faster
    // than softmax and provides a reasonable confidence estimate for classification.
    // Softmax would be more principled but requires exp() per output neuron.
    float sum = 0.0f;
    for (uint32_t i = 0; i < num_outputs; i++) {
        if (isfinite(output[i])) sum += fabsf(output[i]);
    }
    float raw_conf = (sum > 0.0f) ? (max_val / sum) : 0.0f;
    if (raw_conf < 0.0f) raw_conf = 0.0f;
    if (raw_conf > 1.0f) raw_conf = 1.0f;
    // I-M1: Guard against NaN/Inf propagating from network outputs
    if (!isfinite(raw_conf)) raw_conf = 0.0f;
    *out_confidence = raw_conf;

    if (output != stack_buf) nimcp_free(output);

    set_error("No error");
    return NIMCP_OK;
}


/**
 * @brief Domain-scoped prediction (I-L3: documentation for complex logic)
 *
 * WHAT: Predicts using only output neurons whose labels start with domain_prefix.
 * WHY:  When a brain has outputs for multiple domains (e.g., "math_add", "math_sub",
 *       "history_ww2"), domain-scoped prediction restricts the argmax to the relevant
 *       domain, preventing cross-domain interference.
 * HOW:  Forward pass through full network, then filter outputs by domain prefix before
 *       argmax. Confidence is computed only among same-domain outputs.
 *
 * FALLBACK: If no output labels match the domain prefix, falls back to predict_fast
 *           (global argmax across all outputs).
 *
 * THREAD SAFETY: Same as predict_fast -- NOT thread-safe for same brain instance.
 */
nimcp_status_t nimcp_brain_predict_in_domain(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    const char* domain_prefix,
    char* out_label,
    float* out_confidence)
{
    // If no domain prefix, fall back to standard predict_fast
    if (!domain_prefix || domain_prefix[0] == '\0') {
        return nimcp_brain_predict_fast(brain, features, num_features,
                                         out_label, out_confidence);
    }

    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain handle");
    API_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    API_CHECK_THROW(out_label, NIMCP_ERROR_NULL_ARG, "Output label buffer is NULL");
    API_CHECK_THROW(out_confidence, NIMCP_ERROR_NULL_ARG, "Output confidence pointer is NULL");

    brain_t ib = brain->internal_brain;
    if (!ib || !ib->network) {
        // I-L2: Consistent error logging
        LOG_ERROR("predict_in_domain: brain not initialized");
        set_error("Brain not initialized");
        return NIMCP_ERROR;
    }

    // I-H1 FIX: Validate input dimensions to prevent out-of-bounds reads in forward pass.
    // Check both zero and mismatch against brain's expected input size.
    if (num_features == 0) {
        LOG_ERROR("predict_in_domain: num_features must be > 0");
        set_error("num_features must be > 0");
        return NIMCP_ERROR_INVALID;
    }
    if (ib->config.num_inputs > 0 && num_features != ib->config.num_inputs) {
        LOG_ERROR("predict_in_domain: feature count mismatch: expected %u, got %u",
                  ib->config.num_inputs, num_features);
        set_error("Feature count mismatch: expected %u, got %u",
                  ib->config.num_inputs, num_features);
        return NIMCP_ERROR_INVALID;
    }

    uint32_t num_outputs = ib->config.num_outputs;
    if (num_outputs == 0) {
        LOG_ERROR("predict_in_domain: brain has 0 outputs");
        set_error("Brain has 0 outputs");
        return NIMCP_ERROR_INVALID;
    }
    // W6-6 FIX: Zero-initialize stack buffer to prevent argmax reading garbage
    // if forward_raw partially writes the output array.
    float stack_buf[4096] = {0};
    float* output = (num_outputs <= 4096) ? stack_buf : (float*)nimcp_calloc(num_outputs, sizeof(float));
    if (!output) {
        set_error("Failed to allocate output buffer");
        return NIMCP_ERROR_MEMORY;
    }

    // I-C1: Use forward_raw for classification (same reasoning as predict_fast)
    // W6-9 FIX: Store return value for diagnostics (same as predict_fast).
    uint32_t active_count = adaptive_network_forward_raw(ib->network, features, num_features,
                                 output, num_outputs);
    if (active_count == 0) {
        LOG_DEBUG("predict_in_domain: forward_raw returned 0 active neurons");
    }

    // I-H3 FIX: NaN-safe domain-filtered argmax. Use isfinite() to skip NaN/Inf values.
    size_t prefix_len = strlen(domain_prefix);
    uint32_t max_idx = UINT32_MAX;
    float max_val = -FLT_MAX;
    float domain_sum = 0.0f;
    bool has_valid_domain_output = false;

    for (uint32_t i = 0; i < num_outputs; i++) {
        if (!ib->output_labels || i >= ib->num_output_labels || !ib->output_labels[i])
            continue;
        if (strncmp(ib->output_labels[i], domain_prefix, prefix_len) != 0)
            continue;

        // I-H3: Skip NaN/Inf outputs in domain argmax
        if (!isfinite(output[i])) continue;

        domain_sum += fabsf(output[i]);
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
            has_valid_domain_output = true;
        }
    }

    // If no labels matched the domain, fall back to global argmax
    // W7-5 (C-INF-5): Use the already-computed output buffer instead of calling
    // predict_fast, which would execute a redundant second forward pass.
    if (max_idx == UINT32_MAX) {
        // I-M4: Log warning when no domain labels match -- silent fallback makes
        // debugging domain configuration issues difficult
        if (!has_valid_domain_output) {
            LOG_WARN("predict_in_domain: no valid labels match domain prefix '%s', "
                     "falling back to global argmax", domain_prefix);
        }

        // Global argmax on already-computed output (same logic as predict_fast)
        uint32_t global_max_idx = 0;
        float global_max_val = -FLT_MAX;
        bool global_has_valid = false;
        for (uint32_t i = 0; i < num_outputs; i++) {
            if (isfinite(output[i]) && output[i] > global_max_val) {
                global_max_val = output[i];
                global_max_idx = i;
                global_has_valid = true;
            }
        }

        if (!global_has_valid) {
            snprintf(out_label, NIMCP_MAX_LABEL_SIZE, "UNKNOWN");
            out_label[NIMCP_MAX_LABEL_SIZE - 1] = '\0';
            *out_confidence = 0.0f;
            if (output != stack_buf) nimcp_free(output);
            set_error("All network outputs are NaN/Inf");
            return NIMCP_OK;
        }

        if (ib->output_labels && global_max_idx < ib->num_output_labels
            && ib->output_labels[global_max_idx]) {
            strncpy(out_label, ib->output_labels[global_max_idx], NIMCP_MAX_LABEL_SIZE - 1);
        } else {
            snprintf(out_label, NIMCP_MAX_LABEL_SIZE, "output_%u", global_max_idx);
        }
        out_label[NIMCP_MAX_LABEL_SIZE - 1] = '\0';

        // Confidence via sum_abs normalization (matching predict_fast)
        float global_sum = 0.0f;
        for (uint32_t i = 0; i < num_outputs; i++) {
            if (isfinite(output[i])) global_sum += fabsf(output[i]);
        }
        float global_conf = (global_sum > 0.0f) ? (global_max_val / global_sum) : 0.0f;
        if (global_conf < 0.0f) global_conf = 0.0f;
        if (global_conf > 1.0f) global_conf = 1.0f;
        if (!isfinite(global_conf)) global_conf = 0.0f;
        *out_confidence = global_conf;

        if (output != stack_buf) nimcp_free(output);
        set_error("No error");
        return NIMCP_OK;
    }

    strncpy(out_label, ib->output_labels[max_idx], NIMCP_MAX_LABEL_SIZE - 1);
    out_label[NIMCP_MAX_LABEL_SIZE - 1] = '\0';

    // Domain-scoped confidence (only among same-domain outputs)
    float raw_conf = (domain_sum > 0.0f) ? (max_val / domain_sum) : 0.0f;
    if (raw_conf < 0.0f) raw_conf = 0.0f;
    if (raw_conf > 1.0f) raw_conf = 1.0f;
    // I-M1: Guard against NaN/Inf propagating from network outputs
    if (!isfinite(raw_conf)) raw_conf = 0.0f;
    *out_confidence = raw_conf;

    if (output != stack_buf) nimcp_free(output);

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_infer(
    nimcp_brain_t brain,
    const float* features,
    uint32_t num_features,
    float* outputs,
    uint32_t num_outputs)
{
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain handle");
    API_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    API_CHECK_THROW(outputs, NIMCP_ERROR_NULL_ARG, "Outputs array is NULL");
    API_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");
    if (num_features == 0) {
        set_error("num_features must be > 0");
        return NIMCP_ERROR_INVALID;
    }
    // I-H1 FIX: Validate input dimension against brain config
    if (brain->internal_brain->config.num_inputs > 0 &&
        num_features != brain->internal_brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u",
                  brain->internal_brain->config.num_inputs, num_features);
        return NIMCP_ERROR_INVALID;
    }
    if (num_outputs == 0) {
        set_error("num_outputs must be > 0");
        return NIMCP_ERROR_INVALID;
    }

    // Call internal brain API to get decision (which includes output vector)
    brain_decision_t* decision = brain_decide(brain->internal_brain, features, num_features);

    if (!decision) {
        set_error("Brain inference failed");
        return NIMCP_ERROR;
    }

    // Guard: output_vector must be valid
    if (!decision->output_vector) {
        brain_free_decision(decision);
        set_error("Decision output_vector is NULL");
        return NIMCP_ERROR;
    }

    // Copy raw output vector
    uint32_t copy_size = (decision->output_size < num_outputs) ? decision->output_size : num_outputs;
    for (uint32_t i = 0; i < copy_size; i++) {
        outputs[i] = decision->output_vector[i];
    }

    // Fill remaining with zeros if requested more outputs than available
    for (uint32_t i = copy_size; i < num_outputs; i++) {
        outputs[i] = 0.0F;
    }

    // Free decision
    brain_free_decision(decision);

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_decide_full(
    nimcp_brain_t brain,
    const float* features, uint32_t num_features,
    char* out_label, float* out_confidence,
    char* out_explanation,
    float* out_output_vector, uint32_t* out_output_size,
    uint32_t* out_num_active_neurons, float* out_sparsity,
    uint64_t* out_inference_time_us)
{
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain handle");
    API_CHECK_THROW(features, NIMCP_ERROR_NULL_ARG, "Features array is NULL");
    API_CHECK_THROW(out_label, NIMCP_ERROR_NULL_ARG, "Output label buffer is NULL");
    API_CHECK_THROW(out_confidence, NIMCP_ERROR_NULL_ARG, "Output confidence is NULL");
    API_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // I-H1 FIX: Validate input dimensions
    if (num_features == 0) {
        set_error("num_features must be > 0");
        return NIMCP_ERROR_INVALID;
    }
    if (brain->internal_brain->config.num_inputs > 0 &&
        num_features != brain->internal_brain->config.num_inputs) {
        set_error("Feature count mismatch: expected %u, got %u",
                  brain->internal_brain->config.num_inputs, num_features);
        return NIMCP_ERROR_INVALID;
    }

    brain_decision_t* decision = brain_decide(brain->internal_brain, features, num_features);
    if (!decision) {
        set_error("Brain decide_full failed");
        return NIMCP_ERROR;
    }

    strncpy(out_label, decision->label, NIMCP_MAX_LABEL_SIZE - 1);
    out_label[NIMCP_MAX_LABEL_SIZE - 1] = '\0';
    *out_confidence = decision->confidence;

    if (out_explanation) {
        strncpy(out_explanation, decision->explanation, NIMCP_MAX_EXPLANATION_SIZE - 1);
        out_explanation[NIMCP_MAX_EXPLANATION_SIZE - 1] = '\0';
    }

    if (out_output_vector && out_output_size) {
        uint32_t cap = *out_output_size;
        uint32_t copy_n = (decision->output_size < cap) ? decision->output_size : cap;
        for (uint32_t i = 0; i < copy_n; i++) {
            out_output_vector[i] = decision->output_vector[i];
        }
        // W7-6 (C-INF-M1): Report actual number of elements copied, not full
        // output_size. Previously reported decision->output_size which could be
        // larger than the caller's buffer capacity, causing the caller to read
        // beyond what was actually written.
        *out_output_size = copy_n;
    }

    if (out_num_active_neurons) *out_num_active_neurons = decision->num_active_neurons;
    if (out_sparsity) *out_sparsity = decision->sparsity;
    if (out_inference_time_us) *out_inference_time_us = decision->inference_time_us;

    brain_free_decision(decision);
    set_error("No error");
    return NIMCP_OK;
}


nimcp_brain_t nimcp_brain_snapshot_restore(
    nimcp_brain_t brain,
    const char* name)
{
    if (!name) {
        set_error("Snapshot name is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "name is NULL");

        return NULL;
    }

    // Load from snapshot
    brain_t restored_brain = brain_restore_snapshot(
        brain ? brain->internal_brain : NULL,
        name
    );

    if (!restored_brain) {
        set_error("Failed to restore from snapshot");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "restored_brain is NULL");


        return NULL;
    }

    // Allocate new handle (calloc to zero-init last_loss/last_gradient_norm)
    nimcp_brain_t handle = (nimcp_brain_t)nimcp_calloc(1, sizeof(struct nimcp_brain_handle));
    if (!handle) {
        set_error("Failed to allocate brain handle");
        brain_destroy(restored_brain);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "handle is NULL");

        return NULL;
    }

    handle->internal_brain = restored_brain;
    set_error("No error");
    return handle;
}


nimcp_status_t nimcp_brain_snapshot_list(
    nimcp_brain_t brain,
    nimcp_brain_snapshot_info_t* infos,
    uint32_t max_count,
    uint32_t* out_count)
{
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    API_CHECK_THROW(infos, NIMCP_ERROR_NULL_ARG, "Infos array is NULL");

    // Call internal brain list API
    // Note: brain_snapshot_info_t and nimcp_brain_snapshot_info_t have same layout
    bool success = brain_list_snapshots(
        brain->internal_brain,
        (brain_snapshot_info_t*)infos,
        max_count,
        out_count
    );

    if (!success) {
        set_error("Failed to list snapshots");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_snapshot_delete(
    nimcp_brain_t brain,
    const char* name)
{
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    API_CHECK_THROW(name, NIMCP_ERROR_NULL_ARG, "Snapshot name is NULL");

    // Call internal brain delete API
    bool success = brain_delete_snapshot(brain->internal_brain, name);

    if (!success) {
        set_error("Failed to delete snapshot");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_probe(nimcp_brain_t brain, nimcp_brain_probe_t* probe) {
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain is NULL");
    NIMCP_CHECK_THROW(probe, NIMCP_ERROR_NULL_ARG, "Probe output structure is NULL");

    // Get internal brain statistics
    brain_stats_t internal_stats;
    NIMCP_CHECK_THROW(brain_get_stats(brain->internal_brain, &internal_stats),
                      NIMCP_ERROR, "Failed to get brain statistics");

    // Map internal stats to public probe structure
    strncpy(probe->task_name, internal_stats.task_name, sizeof(probe->task_name) - 1);
    probe->task_name[sizeof(probe->task_name) - 1] = '\0';

    // Map internal size enum to public enum
    switch (internal_stats.size) {
        case BRAIN_SIZE_TINY:   probe->size = NIMCP_BRAIN_TINY; break;
        case BRAIN_SIZE_SMALL:  probe->size = NIMCP_BRAIN_SMALL; break;
        case BRAIN_SIZE_MEDIUM: probe->size = NIMCP_BRAIN_MEDIUM; break;
        case BRAIN_SIZE_LARGE:  probe->size = NIMCP_BRAIN_LARGE; break;
        default:                probe->size = NIMCP_BRAIN_SMALL; break;
    }

    // P3 FIX: Removed unused brain_config_t internal_config variable (dead code).
    // Note: We use size as a proxy for task type since internal API doesn't expose it directly
    probe->task = NIMCP_TASK_CLASSIFICATION; // Default

    probe->num_neurons = internal_stats.num_neurons;
    probe->num_synapses = internal_stats.num_synapses;
    probe->num_active_synapses = internal_stats.num_active_synapses;

    probe->total_inferences = internal_stats.total_inferences;
    probe->total_learning_steps = internal_stats.total_learning_steps;

    probe->avg_sparsity = internal_stats.avg_sparsity;
    probe->avg_inference_time_us = internal_stats.avg_inference_time_us;
    probe->current_learning_rate = internal_stats.current_learning_rate;

    probe->accuracy = internal_stats.running_accuracy > 0.0f
                    ? internal_stats.running_accuracy
                    : internal_stats.accuracy;
    probe->memory_bytes = internal_stats.memory_bytes;

    // Get input/output sizes from internal brain
    probe->num_inputs = brain_get_num_inputs(brain->internal_brain);
    probe->num_outputs = brain_get_num_outputs(brain->internal_brain);

    // Get COW statistics from internal brain
    brain_cow_stats_t cow_stats;
    if (brain_get_cow_stats(brain->internal_brain, &cow_stats)) {
        probe->is_cow_clone = cow_stats.is_cow_clone;
        probe->cow_ref_count = cow_stats.cow_ref_count;
        probe->cow_shared_bytes = cow_stats.cow_shared_bytes;
        probe->cow_private_bytes = cow_stats.cow_private_bytes;
    } else {
        // Fallback to defaults if COW stats retrieval fails
        probe->is_cow_clone = false;
        probe->cow_ref_count = 0;
        probe->cow_shared_bytes = 0;
        probe->cow_private_bytes = internal_stats.memory_bytes;
    }

    // GPU status
    probe->gpu_available = brain->internal_brain->gpu_enabled;

    set_error("No error");
    return NIMCP_OK;
}


/**
 * @brief Broadcast brain probe data via bio-async for decoupled metrics collection
 *
 * WHAT: Sends brain probe metrics to all interested subscribers via bio-async
 * WHY:  Enables loose coupling - metrics module receives data without direct dependency
 * HOW:  Fills bio_msg_brain_probe_data_t and broadcasts via BIO_MSG_BRAIN_PROBE_DATA
 *
 * @param brain Brain handle to probe and broadcast
 * @return NIMCP_OK on success, error code otherwise
 */
nimcp_status_t nimcp_brain_broadcast_probe(nimcp_brain_t brain) {
    NIMCP_CHECK_THROW(brain && brain->internal_brain, NIMCP_ERROR_NULL_ARG, "Invalid brain handle");

    // Get probe data
    nimcp_brain_probe_t probe;
    nimcp_status_t status = nimcp_brain_probe(brain, &probe);
    if (status != NIMCP_OK) {
        LOG_ERROR("Failed to get brain probe data for broadcast");
        return status;
    }

    // Get module context for bio-async
    bio_module_context_t ctx = get_brain_probe_module_ctx();
    if (!ctx) {
        LOG_DEBUG("Bio-router not available, skipping probe broadcast");
        return NIMCP_OK;  // Not an error - router may not be initialized
    }

    // Build bio-async message
    bio_msg_brain_probe_data_t msg;
    memset(&msg, 0, sizeof(msg));

    // Initialize header
    bio_msg_init_header(&msg.header, BIO_MSG_BRAIN_PROBE_DATA,
                        BIO_MODULE_BRAIN, BIO_MODULE_ALL,
                        sizeof(bio_msg_brain_probe_data_t));
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;

    // Fill probe data
    msg.brain_id = (uint64_t)(uintptr_t)brain;  // Use pointer as unique ID
    strncpy(msg.task_name, probe.task_name, sizeof(msg.task_name) - 1);
    msg.task_name[sizeof(msg.task_name) - 1] = '\0';
    msg.size = (uint32_t)probe.size;
    msg.task = (uint32_t)probe.task;
    msg.num_neurons = probe.num_neurons;
    msg.num_synapses = probe.num_synapses;
    msg.num_active_synapses = probe.num_active_synapses;
    msg.total_inferences = probe.total_inferences;
    msg.total_learning_steps = probe.total_learning_steps;
    msg.avg_sparsity = probe.avg_sparsity;
    msg.avg_inference_time_us = probe.avg_inference_time_us;
    msg.current_learning_rate = probe.current_learning_rate;
    msg.accuracy = probe.accuracy;
    msg.memory_bytes = probe.memory_bytes;
    msg.num_inputs = probe.num_inputs;
    msg.num_outputs = probe.num_outputs;
    msg.is_cow_clone = probe.is_cow_clone;
    msg.cow_ref_count = probe.cow_ref_count;
    msg.cow_shared_bytes = probe.cow_shared_bytes;
    msg.cow_private_bytes = probe.cow_private_bytes;

    // Broadcast to all subscribers (best-effort - failure is non-fatal)
    nimcp_error_t err = bio_router_broadcast(ctx, &msg, sizeof(msg));
    if (err != NIMCP_SUCCESS) {
        // Log warning but don't fail - probe data was collected successfully,
        // broadcast is secondary and optional
        LOG_DEBUG("Probe broadcast incomplete (some modules may not have received): %d", err);
    } else {
        LOG_DEBUG("Brain probe broadcast: brain_id=%llu, neurons=%u, synapses=%u",
                  (unsigned long long)msg.brain_id, msg.num_neurons, msg.num_synapses);
    }

    set_error("No error");
    return NIMCP_OK;
}


//=============================================================================
// Cognitive Output Rubric API Implementation
//=============================================================================

nimcp_status_t nimcp_brain_rubric(nimcp_brain_t brain, nimcp_rubric_t* rubric) {
    NIMCP_CHECK_THROW(brain && brain->internal_brain, NIMCP_ERROR_NULL_ARG, "Invalid brain handle");
    NIMCP_CHECK_THROW(rubric, NIMCP_ERROR_NULL_ARG, "Rubric output structure is NULL");

    brain_t ib = brain->internal_brain;

    /* Lazy-init rubric evaluator on first call */
    if (!ib->rubric_evaluator) {
        ib->rubric_evaluator = rubric_evaluator_create(NULL);
        if (!ib->rubric_evaluator) {
            set_error("Failed to create rubric evaluator");
            return NIMCP_ERROR_MEMORY;
        }
    }

    /* Need a decision to evaluate */
    if (!ib->last_decision) {
        set_error("No decision to evaluate — call predict or decide_full first");
        return NIMCP_ERROR_INVALID;
    }

    /* Run evaluation */
    rubric_result_t result;
    int rc = rubric_evaluate_decision(ib->rubric_evaluator, ib,
                                       ib->last_decision, &result);
    if (rc != 0) {
        set_error("Rubric evaluation failed");
        return NIMCP_ERROR;
    }

    /* Map internal result to public flat struct */
    rubric->internal_consistency   = result.tier1.internal_consistency;
    rubric->confidence_calibration = result.tier1.confidence_calibration;
    rubric->completeness           = result.tier1.completeness;
    rubric->reasoning_chain_quality = result.tier1.reasoning_chain_quality;
    rubric->epistemic_quality      = result.tier1.epistemic_quality;
    rubric->ethical_alignment      = result.tier1.ethical_alignment;
    rubric->tier1_score            = result.tier1.tier1_score;

    rubric->originality            = result.tier2.originality;
    rubric->integration_depth      = result.tier2.integration_depth;
    rubric->communication_clarity  = result.tier2.communication_clarity;
    rubric->engagement_quality     = result.tier2.engagement_quality;
    rubric->empathetic_accuracy    = result.tier2.empathetic_accuracy;
    rubric->information_density    = result.tier2.information_density;
    rubric->tier2_score            = result.tier2.tier2_score;

    rubric->overall_score          = result.overall_score;
    rubric->grade                  = result.grade;
    rubric->grade_modifier         = result.grade_modifier;
    rubric->subsystems_available   = result.subsystems_available;
    rubric->evaluation_time_us     = result.evaluation_time_us;

    set_error("No error");
    return NIMCP_OK;
}

nimcp_status_t nimcp_brain_broadcast_rubric(nimcp_brain_t brain) {
    NIMCP_CHECK_THROW(brain && brain->internal_brain, NIMCP_ERROR_NULL_ARG, "Invalid brain handle");

    /* Get rubric data */
    nimcp_rubric_t rubric;
    nimcp_status_t status = nimcp_brain_rubric(brain, &rubric);
    if (status != NIMCP_OK) {
        LOG_ERROR("Failed to get rubric data for broadcast");
        return status;
    }

    /* Get module context for bio-async */
    bio_module_context_t ctx = get_brain_probe_module_ctx();
    if (!ctx) {
        LOG_DEBUG("Bio-router not available, skipping rubric broadcast");
        return NIMCP_OK;
    }

    /* Build bio-async message */
    bio_msg_rubric_data_t msg;
    memset(&msg, 0, sizeof(msg));

    bio_msg_init_header(&msg.header, BIO_MSG_RUBRIC_DATA,
                        BIO_MODULE_RUBRIC, BIO_MODULE_ALL,
                        sizeof(bio_msg_rubric_data_t));
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;

    msg.brain_id = (uint64_t)(uintptr_t)brain;

    /* Tier 1 */
    msg.internal_consistency   = rubric.internal_consistency;
    msg.confidence_calibration = rubric.confidence_calibration;
    msg.completeness           = rubric.completeness;
    msg.reasoning_chain_quality = rubric.reasoning_chain_quality;
    msg.epistemic_quality      = rubric.epistemic_quality;
    msg.ethical_alignment      = rubric.ethical_alignment;
    msg.tier1_score            = rubric.tier1_score;

    /* Tier 2 */
    msg.originality            = rubric.originality;
    msg.integration_depth      = rubric.integration_depth;
    msg.communication_clarity  = rubric.communication_clarity;
    msg.engagement_quality     = rubric.engagement_quality;
    msg.empathetic_accuracy    = rubric.empathetic_accuracy;
    msg.information_density    = rubric.information_density;
    msg.tier2_score            = rubric.tier2_score;

    /* Overall */
    msg.overall_score          = rubric.overall_score;
    msg.grade                  = rubric.grade;
    msg.grade_modifier         = rubric.grade_modifier;
    msg.subsystems_available   = rubric.subsystems_available;
    msg.evaluation_time_us     = rubric.evaluation_time_us;

    /* Broadcast */
    nimcp_error_t err = bio_router_broadcast(ctx, &msg, sizeof(msg));
    if (err != NIMCP_SUCCESS) {
        LOG_DEBUG("Rubric broadcast incomplete: %d", err);
    }

    set_error("No error");
    return NIMCP_OK;
}


//=============================================================================
// Copy-on-Write (COW) Cache API Implementation
//=============================================================================

/**
 * WHAT: Clone brain using copy-on-write caching
 * WHY:  Enable efficient replication with 86% memory savings
 * HOW:  Use internal brain_clone_cow which shares network structures
 *
 * PERFORMANCE: <10ms clone time vs ~1000ms for full copy
 * MEMORY: ~1MB overhead vs ~50MB for full copy
 *
 * Phase 2: True COW sharing via pointer sharing
 */
nimcp_brain_t nimcp_brain_clone_cow(nimcp_brain_t original) {
    // Guard: Validate parameters
    if (!original) {
        set_error("NULL brain provided to nimcp_brain_clone_cow");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "original is NULL");

        return NULL;
    }

    if (!original->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_clone_cow: original->internal_brain is NULL");
        return NULL;
    }

    // Allocate handle (calloc to zero-init last_loss/last_gradient_norm)
    nimcp_brain_t clone_handle = (nimcp_brain_t)nimcp_calloc(1, sizeof(struct nimcp_brain_handle));
    if (!clone_handle) {
        set_error("Failed to allocate brain handle for COW clone");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "clone_handle is NULL");

        return NULL;
    }

    // Use internal COW clone function
    clone_handle->internal_brain = brain_clone_cow(original->internal_brain);

    if (!clone_handle->internal_brain) {
        set_error("Failed to clone internal brain");
        nimcp_free(clone_handle);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_clone_cow: clone_handle->internal_brain is NULL");
        return NULL;
    }

    set_error("No error");
    return clone_handle;
}


/**
 * WHAT: Create instant snapshot of brain state using COW with advanced reference tracking
 * WHY:  Enable zero-copy checkpointing for rollback with complete isolation guarantees
 * HOW:  Use brain_clone_cow with enhanced cache reference tracking for snapshot isolation
 *
 * PERFORMANCE: <1ms snapshot time (zero-copy with cache reference)
 * MEMORY: ~64 bytes overhead + O(1) reference count increment
 *
 * IMPLEMENTATION:
 * 1. Create COW clone using brain_clone_cow (shares network via reference counting)
 * 2. Track shared memory size for cache statistics
 * 3. Initialize snapshot refcount for multi-snapshot scenarios
 * 4. Mark snapshot as isolated to prevent unwanted modifications
 * 5. Multiple snapshots share underlying network data until modifications
 *
 * ADVANCED FEATURES:
 * - Snapshot isolation: Modifications to original don't affect snapshot
 * - Reference tracking: Accurate memory usage reporting
 * - Multi-snapshot support: Create multiple snapshots with shared data
 */
nimcp_brain_snapshot_t nimcp_brain_snapshot_cow(nimcp_brain_t brain) {
    // Guard: Validate parameters
    if (!brain) {
        set_error("NULL brain provided to nimcp_brain_snapshot_cow");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");

        return NULL;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_snapshot_cow: brain->internal_brain is NULL");
        return NULL;
    }

    // Allocate snapshot handle
    nimcp_brain_snapshot_t snapshot =
        (nimcp_brain_snapshot_t)nimcp_malloc(sizeof(struct nimcp_brain_snapshot_handle));
    if (!snapshot) {
        set_error("Failed to allocate snapshot handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snapshot is NULL");

        return NULL;
    }

    // WHAT: Capture current stats before creating snapshot
    // WHY:  Snapshot should preserve stats at snapshot time, not reflect future changes
    // HOW:  Get stats from original brain before cloning
    brain_stats_t current_stats;
    if (!brain_get_stats(brain->internal_brain, &current_stats)) {
        set_error("Failed to get brain stats for snapshot");
        nimcp_free(snapshot);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_snapshot_cow: brain_get_stats is NULL");
        return NULL;
    }

    // WHAT: Create COW clone of brain
    // WHY:  Enables zero-copy snapshot with reference counting
    // HOW:  brain_clone_cow shares network and increments reference count
    brain_t snapshot_brain = brain_clone_cow(brain->internal_brain);

    if (!snapshot_brain) {
        set_error("Failed to create COW clone for snapshot");
        nimcp_free(snapshot);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "snapshot_brain is NULL");


        return NULL;
    }

    // CRITICAL: Mark this brain as a snapshot and preserve stats
    // This prevents brain_get_stats from reading the shared network's current stats
    brain_mark_as_snapshot(snapshot_brain, &current_stats);

    // WHAT: Calculate shared memory size for tracking
    // WHY:  Enables accurate memory usage reporting and cache statistics
    // HOW:  Use the preserved stats to estimate network size
    size_t shared_size = (size_t)(current_stats.num_neurons * 100 + current_stats.num_synapses * 20);

    // WHAT: Store snapshot with enhanced tracking metadata
    // WHY:  Track when snapshot was created and memory usage for monitoring
    // HOW:  Save cloned brain, timestamp, and reference tracking info
    snapshot->internal_brain_snapshot = snapshot_brain;
    snapshot->timestamp_us = nimcp_time_monotonic_us();
    snapshot->shared_memory_size = shared_size;
    snapshot->snapshot_refcount = 1;  // This snapshot has one reference
    snapshot->is_isolated = true;     // Snapshot is isolated from original modifications

    // WHAT: Record cache reference for accurate COW statistics
    // WHY:  Track memory savings achieved by COW snapshots
    // HOW:  Use nimcp_cache_record_reference with shared memory size
    if (shared_size > 0) {
        nimcp_cache_record_reference(shared_size);
    }

    set_error("No error");
    return snapshot;
}


/**
 * WHAT: Restore brain state from COW snapshot
 * WHY:  Enable instant rollback to previous state
 * HOW:  Swap brain state with snapshot state
 *
 * PERFORMANCE: <1ms restore time (pointer swap)
 * MEMORY: O(1)
 */
nimcp_status_t nimcp_brain_restore_cow(nimcp_brain_t brain, nimcp_brain_snapshot_t snapshot) {
    // Guard: Validate parameters
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to nimcp_brain_restore_cow");
    API_CHECK_THROW(snapshot, NIMCP_ERROR_NULL_ARG, "NULL snapshot provided to nimcp_brain_restore_cow");
    API_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");
    API_CHECK_THROW(snapshot->internal_brain_snapshot, NIMCP_ERROR_INVALID,
                      "Snapshot has NULL internal_brain_snapshot");

    // CRITICAL FIX: Use brain_clone_cow() which properly handles COW refcounting
    // The snapshot was created via save/load, so it owns its network independently
    // We can safely clone it, and the clone will share the snapshot's network

    // Save the old brain pointer for cleanup
    brain_t old_brain = brain->internal_brain;

    // Clone the snapshot to create new brain state (creates COW reference)
    brain_t new_brain = brain_clone_cow(snapshot->internal_brain_snapshot);
    NIMCP_CHECK_THROW(new_brain, NIMCP_ERROR, "Failed to clone snapshot for restore");

    // CRITICAL: Mark the restored brain as a snapshot with the same preserved stats
    // This ensures brain_get_stats returns the snapshot's stats, not the current network stats
    brain_stats_t snapshot_stats;
    if (brain_get_stats(snapshot->internal_brain_snapshot, &snapshot_stats)) {
        brain_mark_as_snapshot(new_brain, &snapshot_stats);
    }

    // Assign new brain (do this before destroying old brain to avoid use-after-free)
    brain->internal_brain = new_brain;

    // Now destroy the old brain (will decrement refcounts properly)
    brain_destroy(old_brain);

    set_error("No error");
    return NIMCP_OK;
}


//=============================================================================
// Phase 10.2: Working Memory API Implementation
//=============================================================================

/**
 * @brief Add item to brain's working memory
 *
 * WHAT: Wrapper for working_memory_add() on brain's working memory
 * WHY:  Provide public API for adding items to working memory
 * HOW:  Validate brain → Get working memory → Call internal API
 */
nimcp_status_t nimcp_brain_working_memory_add(
    nimcp_brain_t brain,
    const float* data,
    uint32_t size,
    float salience)
{
    // Guard: Validate brain
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to working_memory_add");
    API_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Validate parameters FIRST (before checking subsystem availability)
    API_CHECK_THROW(data, NIMCP_ERROR_NULL_ARG, "NULL data provided to working_memory_add");
    API_CHECK_THROW(size != 0, NIMCP_ERROR_INVALID, "Invalid size (0) provided to working_memory_add");

    // Guard: Check if working memory enabled (after parameter validation)
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    API_CHECK_THROW(wm, NIMCP_ERROR_INVALID, "Working memory not enabled in brain config");

    // Add to working memory
    bool success = working_memory_add(wm, data, size, salience);
    API_CHECK_THROW(success, NIMCP_ERROR, "Failed to add item to working memory");

    set_error("No error");
    return NIMCP_OK;
}


/**
 * @brief Get item from brain's working memory
 *
 * WHAT: Wrapper for working_memory_get() on brain's working memory
 * WHY:  Provide public API for accessing working memory items
 * HOW:  Validate brain → Get working memory → Call internal API
 */
const float* nimcp_brain_working_memory_get(
    nimcp_brain_t brain,
    uint32_t index,
    uint32_t* size_out)
{
    // Guard: Validate brain
    if (!brain || !brain->internal_brain) {
        set_error("Invalid brain handle");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_working_memory_get: required parameter is NULL (brain, brain->internal_brain)");
        return NULL;
    }

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    if (!wm) {
        set_error("Working memory not enabled");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "wm is NULL");

        return NULL;
    }

    // Get item
    const float* item = working_memory_get(wm, index, size_out);
    if (!item) {
        set_error("Invalid index or empty working memory");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "item is NULL");

        return NULL;
    }

    set_error("No error");
    return item;
}


/**
 * @brief Refresh item in brain's working memory
 *
 * WHAT: Wrapper for working_memory_refresh() on brain's working memory
 * WHY:  Provide public API for preventing decay (rehearsal)
 * HOW:  Validate brain → Get working memory → Call internal API
 */
nimcp_status_t nimcp_brain_working_memory_refresh(
    nimcp_brain_t brain,
    uint32_t index)
{
    // Guard: Validate brain
    API_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided");
    API_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if working memory enabled
    working_memory_t* wm = brain_get_working_memory(brain->internal_brain);
    API_CHECK_THROW(wm, NIMCP_ERROR_INVALID, "Working memory not enabled");

    // Refresh item
    bool success = working_memory_refresh(wm, index);
    API_CHECK_THROW(success, NIMCP_ERROR_INVALID, "Invalid index for refresh");

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_workspace_compete(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module,
    const float* content,
    uint32_t content_dim,
    float strength)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_compete");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Validate parameters FIRST (before checking workspace availability)
    NIMCP_CHECK_THROW(content, NIMCP_ERROR_NULL_ARG, "NULL content provided to workspace_compete");
    NIMCP_CHECK_THROW(content_dim != 0, NIMCP_ERROR_INVALID, "Invalid content_dim (0)");
    NIMCP_CHECK_THROW(strength >= 0.0F && strength <= 1.0F, NIMCP_ERROR_INVALID,
                      "Strength must be in range [0.0, 1.0]");

    // Guard: Check if global workspace enabled (after parameter validation)
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Convert module enum and compete
    cognitive_module_t internal_module = convert_module_enum(module);
    bool won = global_workspace_compete(workspace, internal_module, content, content_dim, strength);

    if (won) {
        set_error("No error");
        return NIMCP_OK;
    } else {
        set_error("Did not win workspace competition");
        return NIMCP_ERROR;
    }
}


nimcp_status_t nimcp_brain_workspace_subscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_subscribe");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Subscribe module
    cognitive_module_t internal_module = convert_module_enum(module);
    bool success = global_workspace_subscribe(workspace, internal_module);
    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Failed to subscribe module");

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_workspace_unsubscribe(
    nimcp_brain_t brain,
    nimcp_cognitive_module_t module)
{
    // Guard: Validate brain
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "NULL brain provided to workspace_unsubscribe");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    // Guard: Check if global workspace enabled
    global_workspace_t* workspace = brain_get_global_workspace(brain->internal_brain);
    NIMCP_CHECK_THROW(workspace, NIMCP_ERROR_INVALID, "Global workspace not enabled in brain config");

    // Unsubscribe module
    cognitive_module_t internal_module = convert_module_enum(module);
    bool success = global_workspace_unsubscribe(workspace, internal_module);
    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Failed to unsubscribe module");

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_network_forward(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    float* outputs,
    uint32_t num_outputs)
{
    NIMCP_CHECK_THROW(network, NIMCP_ERROR_NULL_ARG, "Network handle is NULL");
    NIMCP_CHECK_THROW(inputs, NIMCP_ERROR_NULL_ARG, "Inputs array is NULL");
    NIMCP_CHECK_THROW(outputs, NIMCP_ERROR_NULL_ARG, "Outputs array is NULL");

    // Call internal network API
    bool success = neural_network_forward(network->internal_network,
                                         inputs, num_inputs,
                                         outputs, num_outputs);

    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Network forward pass failed");

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_network_train(
    nimcp_network_t network,
    const float* inputs,
    uint32_t num_inputs,
    const float* targets,
    uint32_t num_targets)
{
    NIMCP_CHECK_THROW(network, NIMCP_ERROR_NULL_ARG, "Network handle is NULL");

    // Training not yet implemented in internal API
    NIMCP_THROW(NIMCP_ERROR, "Training not yet implemented");
    return NIMCP_ERROR;
}


nimcp_status_t nimcp_ethics_check(
    nimcp_ethics_t ethics,
    const float* situation,
    uint32_t num_features,
    float* out_score)
{
    NIMCP_CHECK_THROW(ethics, NIMCP_ERROR_NULL_ARG, "Ethics handle is NULL");
    NIMCP_CHECK_THROW(situation, NIMCP_ERROR_NULL_ARG, "Situation array is NULL");
    NIMCP_CHECK_THROW(out_score, NIMCP_ERROR_NULL_ARG, "Output score pointer is NULL");

    /* P2-1: The internal API accepts non-const float* but we guarantee
     * the features are not modified during evaluation. This const_cast
     * is safe because ethics_engine_evaluate_action() is read-only.
     * TODO: Fix internal API to accept const float*. */
    action_context_t action = {0};
    action.features = (float*)(uintptr_t)situation;  /* const_cast - see above */
    action.num_features = num_features;
    action.affected_agents = NULL;
    action.num_affected_agents = 0;
    action.predicted_harm = 0.0F;

    // Evaluate using internal ethics engine
    ethics_evaluation_t eval = ethics_engine_evaluate_action(ethics->internal_ethics, &action);

    // Convert evaluation to simple score
    // Golden Rule score: -1 (harmful) to +1 (beneficial)
    *out_score = eval.golden_rule_score;

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_knowledge_add_fact(
    nimcp_knowledge_t knowledge,
    const char* subject,
    const char* predicate,
    const char* object)
{
    NIMCP_CHECK_THROW(knowledge, NIMCP_ERROR_NULL_ARG, "Knowledge handle is NULL");
    NIMCP_CHECK_THROW(subject && predicate && object, NIMCP_ERROR_NULL_ARG,
                      "Subject, predicate, or object is NULL");

    // Create a knowledge item from the fact
    knowledge_item_t item = {0};

    // Use subject as concept
    strncpy(item.concept_name, subject, sizeof(item.concept_name) - 1);

    // Create definition from predicate and object
    snprintf(item.definition, sizeof(item.definition), "%s %s", predicate, object);

    // Set defaults
    item.domain = KNOWLEDGE_DOMAIN_GENERAL;
    item.confidence = 1.0F;
    item.examples = NULL;
    item.num_examples = 0;
    item.related_concepts = NULL;
    item.num_related = 0;
    item.learned_timestamp = 0;
    item.reinforcement_count = 1;

    // Add to internal knowledge system
    bool success = knowledge_add_item(knowledge->internal_knowledge, &item);

    NIMCP_CHECK_THROW(success, NIMCP_ERROR, "Failed to add knowledge item");

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_knowledge_query(
    nimcp_knowledge_t knowledge,
    const char* query,
    char* out_result,
    uint32_t max_result_len)
{
    NIMCP_CHECK_THROW(knowledge, NIMCP_ERROR_NULL_ARG, "Knowledge handle is NULL");
    NIMCP_CHECK_THROW(query, NIMCP_ERROR_NULL_ARG, "Query is NULL");
    NIMCP_CHECK_THROW(out_result, NIMCP_ERROR_NULL_ARG, "Output result buffer is NULL");

    // Try to retrieve knowledge about the query concept
    knowledge_item_t item;
    bool found = knowledge_retrieve(knowledge->internal_knowledge, query, &item);

    if (found) {
        // Return the definition
        snprintf(out_result, max_result_len, "%s", item.definition);
    } else {
        // Concept not found
        snprintf(out_result, max_result_len, "No knowledge found about '%s'", query);
    }

    set_error("No error");
    return NIMCP_OK;
}

bool nimcp_enable_complex_oscillations(nimcp_brain_t brain, bool enable) {
    if (!brain) {
        set_error("Brain handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_enable_complex_oscillations: brain is NULL");
        return false;
    }

    if (!brain->internal_brain) {
        set_error("Brain has NULL internal_brain");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_enable_complex_oscillations: brain->internal_brain is NULL");
        return false;
    }

    // Get brain's internal structure to access complex oscillation state
    // Note: This requires brain_internal_t access, which would need to be exposed
    // For now, we'll use a hypothetical brain configuration function
    // This would typically be: brain_config_set_complex_oscillations(brain->internal_brain, enable)

    // Since we may not have direct config access, we'll check if it's already enabled
    // and return true if enabling and already enabled, or if disabling and already disabled
    bool currently_enabled = brain_complex_oscillation_is_enabled(brain->internal_brain);

    if (enable == currently_enabled) {
        set_error("No error");
        return true;
    }

    // For actual enable/disable, we would need brain_config_t access
    // This is a placeholder implementation that at least validates the state
    set_error("Complex oscillation enable/disable requires brain reconfiguration");
    NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "nimcp_enable_complex_oscillations: validation failed");
    return false;
}


//=============================================================================
// Phase 2.8: Dynamic Brain Resizing API
//=============================================================================

bool nimcp_brain_resize(nimcp_brain_t brain, uint32_t new_neuron_count) {
    if (!brain) {
        set_error("Brain handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_resize: brain is NULL");
        return false;
    }
    if (!brain->internal_brain) {
        set_error("Brain internal state is NULL");
        return false;
    }

    return brain_resize(brain->internal_brain, new_neuron_count);
}


bool nimcp_brain_auto_resize(nimcp_brain_t brain) {
    if (!brain) {
        set_error("Brain handle is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "nimcp_brain_auto_resize: brain is NULL");
        return false;
    }
    if (!brain->internal_brain) {
        set_error("Brain internal state is NULL");
        return false;
    }

    return brain_auto_resize(brain->internal_brain);
}


nimcp_status_t nimcp_brain_train_batch(
    nimcp_brain_t brain,
    const float* features,
    const float* targets,
    uint32_t batch_size,
    uint32_t num_features,
    uint32_t num_targets,
    nimcp_training_result_t* result)
{
    NIMCP_CHECK_THROW(brain && features && targets, NIMCP_ERROR_NULL_ARG, "NULL argument provided");
    NIMCP_CHECK_THROW(batch_size != 0, NIMCP_ERROR_INVALID, "Batch size cannot be zero");
    // I-M1 FIX: Validate feature/target counts and check for multiplication overflow
    // in batch indexing. batch_size * num_features could overflow uint32_t, causing
    // out-of-bounds reads from the features/targets arrays.
    NIMCP_CHECK_THROW(num_features != 0, NIMCP_ERROR_INVALID, "num_features cannot be zero");
    NIMCP_CHECK_THROW(num_targets != 0, NIMCP_ERROR_INVALID, "num_targets cannot be zero");
    if (batch_size > UINT32_MAX / num_features) {
        set_error("batch_size * num_features overflow");
        return NIMCP_ERROR_INVALID;
    }
    if (batch_size > UINT32_MAX / num_targets) {
        set_error("batch_size * num_targets overflow");
        return NIMCP_ERROR_INVALID;
    }

    // Train on each example and average results
    float total_loss = 0.0F;
    uint32_t completed = 0;
    nimcp_training_result_t step_result = {0};

    for (uint32_t i = 0; i < batch_size; i++) {
        const float* sample_features = features + ((size_t)i * num_features);
        const float* sample_targets = targets + ((size_t)i * num_targets);

        nimcp_status_t res = nimcp_brain_train_step(
            brain, sample_features, num_features,
            sample_targets, num_targets, &step_result);

        if (res != NIMCP_OK) {
            return res;
        }

        total_loss += step_result.loss;
        completed++;

        if (step_result.early_stopped) {
            break;
        }
    }

    if (result) {
        result->loss = (completed > 0) ? (total_loss / completed) : 0.0F;
        result->learning_rate = step_result.learning_rate;
        result->step = step_result.step;
        result->early_stopped = step_result.early_stopped;
        result->gradient_norm = step_result.gradient_norm;
    }

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_enable_callbacks(
    nimcp_brain_t brain,
    const nimcp_callback_config_t* config)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");

    training_pipeline_state_t* state = get_training_state(brain);
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_MEMORY, "Failed to get training state");

    // Destroy existing callbacks if present
    if (state->callbacks) {
        tcb_destroy(state->callbacks);
        state->callbacks = NULL;
    }

    // Build internal config from public config
    tcb_config_t internal_config = tcb_config_default();

    if (config) {
        internal_config.enable_auto_checkpoint = config->enable_auto_checkpoint;
        internal_config.checkpoint_interval = config->checkpoint_interval;
        internal_config.enable_early_stopping = config->enable_early_stopping;
        internal_config.patience = config->patience;
        internal_config.min_delta = config->min_delta;
        internal_config.divergence_threshold = config->divergence_threshold;
        if (config->log_interval > 0) {
            internal_config.enable_auto_logging = true;
            internal_config.log_interval = config->log_interval;
        }
    }

    // Create callback manager
    state->callbacks = tcb_create(&internal_config);
    NIMCP_CHECK_THROW(state->callbacks, NIMCP_ERROR_MEMORY, "Failed to create callback manager");

    state->callbacks_enabled = true;
    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_disable_callbacks(nimcp_brain_t brain) {
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");

    training_pipeline_state_t* state = get_training_state(brain);
    NIMCP_CHECK_THROW(state, NIMCP_ERROR_INVALID, "No training state");

    state->callbacks_enabled = false;
    set_error("No error");
    return NIMCP_OK;
}

uint32_t nimcp_brain_register_callback(
    nimcp_brain_t brain,
    nimcp_callback_event_t event,
    nimcp_training_callback_fn callback,
    void* user_data,
    const char* name)
{
    if (!brain) {
        set_error("Brain handle is NULL");
        return 0;
    }

    if (!callback) {
        set_error("Callback function is NULL");
        return 0;
    }

    training_pipeline_state_t* state = get_training_state(brain);
    if (!state) {
        set_error("No training state");
        return 0;
    }

    // Auto-enable callbacks if not already enabled
    if (!state->callbacks) {
        nimcp_status_t res = nimcp_brain_enable_callbacks(brain, NULL);
        if (res != NIMCP_OK) {
            return 0;
        }
    }

    // Allocate wrapper
    callback_wrapper_t* wrapper = nimcp_malloc(sizeof(callback_wrapper_t));
    if (!wrapper) {
        set_error("Failed to allocate callback wrapper");
        return 0;
    }

    wrapper->public_callback = callback;
    wrapper->user_data = user_data;

    /* P1-47: Find first free slot instead of wrapping counter, to avoid
     * freeing a live wrapper that's still referenced by the callback manager */
    nimcp_mutex_lock(&g_callback_wrappers_mutex);
    uint32_t wrapper_idx = MAX_CALLBACK_WRAPPERS; /* sentinel = not found */
    for (uint32_t i = 0; i < MAX_CALLBACK_WRAPPERS; i++) {
        if (g_callback_wrappers[i] == NULL) {
            wrapper_idx = i;
            break;
        }
    }
    if (wrapper_idx >= MAX_CALLBACK_WRAPPERS) {
        nimcp_mutex_unlock(&g_callback_wrappers_mutex);
        nimcp_free(wrapper);
        set_error("Callback wrapper table full");
        return 0;
    }
    g_callback_wrappers[wrapper_idx] = wrapper;
    g_next_wrapper_id++;
    nimcp_mutex_unlock(&g_callback_wrappers_mutex);

    // Map public event type to internal event type
    tcb_event_type_t internal_event;
    switch (event) {
        case NIMCP_CB_STEP_COMPLETE:    internal_event = TCB_EVENT_STEP_COMPLETE; break;
        case NIMCP_CB_EPOCH_COMPLETE:   internal_event = TCB_EVENT_EPOCH_COMPLETE; break;
        case NIMCP_CB_LOSS_COMPUTED:    internal_event = TCB_EVENT_LOSS_COMPUTED; break;
        case NIMCP_CB_WEIGHTS_UPDATED:  internal_event = TCB_EVENT_WEIGHTS_UPDATED; break;
        case NIMCP_CB_LR_CHANGED:       internal_event = TCB_EVENT_LR_CHANGED; break;
        case NIMCP_CB_CONVERGENCE:      internal_event = TCB_EVENT_CONVERGENCE; break;
        case NIMCP_CB_DIVERGENCE:       internal_event = TCB_EVENT_DIVERGENCE; break;
        case NIMCP_CB_CHECKPOINT:       internal_event = TCB_EVENT_CHECKPOINT; break;
        default:                        internal_event = TCB_EVENT_STEP_COMPLETE; break;
    }

    // Register with internal callback manager
    tcb_callback_info_t info = {
        .callback = callback_bridge,
        .user_data = wrapper,
        .event_type = internal_event,
        .mode = TCB_MODE_SYNC,
        .priority = TCB_PRIORITY_NORMAL,
        .name = name,
        .enabled = true
    };

    uint32_t cb_id = tcb_register(state->callbacks, &info);
    if (cb_id == 0) {
        set_error("Failed to register callback");
        return 0;
    }

    set_error("No error");
    return cb_id;
}


nimcp_status_t nimcp_brain_unregister_callback(
    nimcp_brain_t brain,
    uint32_t callback_id)
{
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");

    training_pipeline_state_t* state = get_training_state(brain);
    NIMCP_CHECK_THROW(state && state->callbacks, NIMCP_ERROR_INVALID, "Callbacks not enabled");

    // Unregister from internal manager
    if (!tcb_unregister(state->callbacks, callback_id)) {
        NIMCP_CHECK_THROW(false, NIMCP_ERROR, "Failed to unregister callback");
    }

    set_error("No error");
    return NIMCP_OK;
}


//=============================================================================
// Brain Health Probes - System-Level Metrics Collection
//=============================================================================

size_t nimcp_brain_get_memory_rss(void) {
#ifdef __linux__
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;

    unsigned long rss_kb = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            /* Format: "VmRSS:    <number> kB\n" */
            if (sscanf(line + 6, " %lu", &rss_kb) != 1) {
                rss_kb = 0;
            }
            break;
        }
    }
    fclose(f);
    return (size_t)rss_kb * 1024;  /* Convert kB to bytes */
#else
    return 0;  /* Not supported on non-Linux platforms */
#endif
}


size_t nimcp_brain_get_gpu_vram_used(nimcp_brain_t brain) {
    if (!brain || !brain->internal_brain) return 0;
    if (!brain->internal_brain->gpu_enabled) return 0;

    struct nimcp_gpu_context_s* gpu_ctx = brain->internal_brain->gpu_ctx;
    if (!gpu_ctx) return 0;

    return gpu_ctx->allocated_memory;
}


float nimcp_brain_get_neuron_utilization(nimcp_brain_t brain) {
    if (!brain || !brain->internal_brain) return 0.0f;

    neural_network_t net = adaptive_network_get_base_network(brain->internal_brain->network);
    if (!net) return 0.0f;

    uint32_t total = neural_network_get_num_neurons(net);
    if (total == 0) return 0.0f;

    /* Sample every Nth neuron for speed (same pattern as weight_stats) */
    uint32_t step = total / 15000;
    if (step < 1) step = 1;

    uint32_t sampled = 0;
    uint32_t connected = 0;

    for (uint32_t i = 0; i < total; i += step) {
        neuron_t* n = neural_network_get_neuron(net, i);
        if (!n) continue;
        sampled++;
        if (NEURON_OUT_COUNT(n) > 0) {
            connected++;
        }
    }

    if (sampled == 0) return 0.0f;
    return (float)connected / (float)sampled;
}


nimcp_status_t nimcp_brain_get_immune_metrics(nimcp_brain_t brain, nimcp_immune_metrics_t* out) {
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(out, NIMCP_ERROR_NULL_ARG, "Output structure is NULL");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    memset(out, 0, sizeof(*out));

    /* If immune system is not enabled, return zeroed struct successfully */
    if (!brain->internal_brain->immune_enabled ||
        !brain->internal_brain->immune_system) {
        set_error("No error");
        return NIMCP_OK;
    }

    brain_immune_stats_t istats;
    int rc = brain_immune_get_stats(brain->internal_brain->immune_system, &istats);
    if (rc != 0) {
        set_error("Failed to get immune stats");
        return NIMCP_ERROR;
    }

    out->total_exceptions = (uint32_t)istats.antigens_processed;
    out->recovered_exceptions = (uint32_t)istats.threats_neutralized;
    out->inflammation_level = istats.inflammation_level_continuous;
    out->active_antibodies = istats.active_antibodies;

    set_error("No error");
    return NIMCP_OK;
}


/**
 * @brief Per-brain synapse count from previous probe call
 *
 * WHAT: Static storage for computing synapse growth delta
 * WHY:  The delta is per-call, not per-session — simplest approach is a static
 * HOW:  Stores previous synapse count, computes diff on next call
 *
 * NOTE: This is a single static — if multiple brains are probed, the delta
 * reflects the difference since the last call regardless of which brain.
 * A per-brain approach would require a hash map; this is adequate for the
 * common single-brain monitoring case.
 */
static uint64_t s_last_synapse_count = 0;

nimcp_status_t nimcp_brain_get_synapse_stats(nimcp_brain_t brain, nimcp_synapse_stats_t* out) {
    NIMCP_CHECK_THROW(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_CHECK_THROW(out, NIMCP_ERROR_NULL_ARG, "Output structure is NULL");
    NIMCP_CHECK_THROW(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    memset(out, 0, sizeof(*out));

    brain_stats_t stats;
    if (!brain_get_stats(brain->internal_brain, &stats)) {
        set_error("Failed to get brain stats");
        return NIMCP_ERROR;
    }

    uint64_t current = (uint64_t)stats.num_synapses;
    out->total_synapses = current;
    out->growth_since_last = (int64_t)current - (int64_t)s_last_synapse_count;
    s_last_synapse_count = current;

    set_error("No error");
    return NIMCP_OK;
}


/* ============================================================================
 * Unified Experience API (Developmental Learning)
 * ============================================================================ */

nimcp_status_t nimcp_brain_innate_hardwire(
    nimcp_brain_t brain,
    const innate_config_t* config)
{
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(config, NIMCP_ERROR_NULL_ARG, "Config is NULL");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    int rc = brain_innate_hardwire(brain->internal_brain, config);
    if (rc != 0) {
        set_error("brain_innate_hardwire failed");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_experience(
    nimcp_brain_t brain,
    const float* input,
    uint32_t input_size,
    float* output,
    uint32_t output_size,
    float teacher_reward,
    brain_experience_result_t* result)
{
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(input, NIMCP_ERROR_NULL_ARG, "Input array is NULL");
    NIMCP_API_CHECK_NULL(output, NIMCP_ERROR_NULL_ARG, "Output array is NULL");
    NIMCP_API_CHECK_NULL(result, NIMCP_ERROR_NULL_ARG, "Result struct is NULL");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    bool ok = brain_experience(brain->internal_brain, input, input_size,
                                output, output_size, teacher_reward, result);
    if (!ok) {
        set_error("brain_experience failed");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}


nimcp_status_t nimcp_brain_experience_configure(
    nimcp_brain_t brain,
    const brain_experience_config_t* config)
{
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(config, NIMCP_ERROR_NULL_ARG, "Config is NULL");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    int rc = brain_experience_configure(brain->internal_brain, config);
    if (rc != 0) {
        set_error("brain_experience_configure failed");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}


float nimcp_brain_experience_correct(
    nimcp_brain_t brain,
    const float* expected,
    uint32_t expected_size)
{
    if (!brain || !expected || !brain->internal_brain) return -1.0f;
    return brain_experience_correct(brain->internal_brain, expected, expected_size);
}


nimcp_status_t nimcp_brain_experience_attend(
    nimcp_brain_t brain,
    const char* modality,
    float strength)
{
    NIMCP_API_CHECK_NULL(brain, NIMCP_ERROR_NULL_ARG, "Brain handle is NULL");
    NIMCP_API_CHECK_NULL(modality, NIMCP_ERROR_NULL_ARG, "Modality is NULL");
    NIMCP_API_CHECK_NULL(brain->internal_brain, NIMCP_ERROR_INVALID, "Brain has NULL internal_brain");

    int rc = brain_experience_attend(brain->internal_brain, modality, strength);
    if (rc != 0) {
        set_error("brain_experience_attend failed");
        return NIMCP_ERROR;
    }

    set_error("No error");
    return NIMCP_OK;
}
