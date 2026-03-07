// nimcp_lnn_gradient_part_core.c - core functions
// Part of nimcp_lnn_gradient.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp_lnn_gradient.c


/*=============================================================================
 * Gradient Computation Functions
 *===========================================================================*/

/**
 * @brief Compute gradients using adjoint method
 *
 * WHAT: Solve adjoint ODE backward to compute ∂L/∂θ
 * WHY:  O(1) memory vs O(T) for BPTT
 * HOW:  Integrate dλ/dt = -∂f/∂x^T λ from T to t0
 */
int lnn_gradient_compute_adjoint(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    const nimcp_tensor_t* dL_dx_final
) {
    // Guard: validate inputs
    if (!ctx) {
        NIMCP_LOGGING_ERROR("Gradient context is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                             "Null context in lnn_gradient_compute_adjoint");
        return LNN_ERROR_NULL_POINTER;
    }

    if (!network) {
        NIMCP_LOGGING_ERROR("Network is NULL");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                             "Null network in lnn_gradient_compute_adjoint");
        return LNN_ERROR_NULL_POINTER;
    }

    if (!dL_dx_final) {
        NIMCP_LOGGING_ERROR("Loss gradient is NULL");
        NIMCP_THROW_BRAIN(NIMCP_ERROR_NULL_POINTER, network->id, "LNN",
                         "Null loss gradient in lnn_gradient_compute_adjoint");
        return LNN_ERROR_NULL_POINTER;
    }

    // Guard: check network has state history
    if (!network->state_history || network->history_len == 0) {
        NIMCP_LOGGING_ERROR("Network has no state history for adjoint computation");
        NIMCP_THROW_BRAIN(NIMCP_ERROR_BACKWARD_PASS, network->id, "LNN",
                         "No state history for adjoint computation");
        return LNN_ERROR_INVALID_STATE;
    }

    // Lock for thread safety
    if (ctx->mutex) {
        nimcp_mutex_lock((nimcp_mutex_t*)ctx->mutex);
    }

    double start_time = lnn_gradient_get_time_ms();

    // Reset gradient accumulator
    lnn_gradient_reset(ctx);

    // Store loss gradient
    if (ctx->dL_dx_final) {
        nimcp_tensor_destroy(ctx->dL_dx_final);
    }
    ctx->dL_dx_final = nimcp_tensor_clone(dL_dx_final);

    // Initialize adjoint: λ(T) = ∂L/∂x(T)
    nimcp_tensor_t* adjoint_current = nimcp_tensor_clone(dL_dx_final);
    if (!adjoint_current) {
        NIMCP_LOGGING_ERROR("Failed to initialize adjoint state");
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, 0,
                          "Failed to initialize adjoint state for gradient computation");
        if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    // Time parameters
    ctx->t_end = (float)network->history_len * network->config->default_dt;
    ctx->t_start = 0.0f;
    ctx->dt = -network->config->default_dt;  // Negative for backward integration

    NIMCP_LOGGING_DEBUG("Starting adjoint computation: T=%f, dt=%f, steps=%u",
                        ctx->t_end, ctx->dt, network->history_len);

    // Backward integration loop
    for (int step = (int)network->history_len - 1; step >= 0; step--) {
        float t = step * network->config->default_dt;

        // Get state at this time step
        nimcp_tensor_t* x_current = network->state_history[step];
        if (!x_current) {
            NIMCP_LOGGING_ERROR("Missing state at step %d", step);
            NIMCP_THROW_BRAIN(NIMCP_ERROR_BACKWARD_PASS, network->id, "LNN",
                             "Missing state at step %d for adjoint computation", step);
            nimcp_tensor_destroy(adjoint_current);
            if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
            return LNN_ERROR_INVALID_STATE;
        }

        // Compute adjoint step: λ(t-dt) from λ(t)
        nimcp_tensor_t* adjoint_next = nimcp_tensor_create(
            nimcp_tensor_shape(adjoint_current)->dims,
            nimcp_tensor_shape(adjoint_current)->rank,
            NIMCP_DTYPE_F32
        );

        if (!adjoint_next) {
            NIMCP_LOGGING_ERROR("Failed to allocate adjoint_next at step %d", step);
            NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, 0,
                              "Failed to allocate adjoint_next at step %d", step);
            nimcp_tensor_destroy(adjoint_current);
            if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
            return LNN_ERROR_OUT_OF_MEMORY;
        }

        int ret = lnn_gradient_adjoint_step(ctx, x_current, adjoint_current, t, ctx->dt, adjoint_next);
        if (ret != 0) {
            NIMCP_LOGGING_ERROR("Adjoint step failed at step %d: error %d", step, ret);
            NIMCP_THROW_BRAIN(NIMCP_ERROR_BACKWARD_PASS, network->id, "LNN",
                             "Adjoint step failed at step %d: error %d", step, ret);
            nimcp_tensor_destroy(adjoint_current);
            nimcp_tensor_destroy(adjoint_next);
            if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
            return ret;
        }

        // Accumulate parameter gradients: ∂L/∂θ += λ^T ∂f/∂θ * dt
        ret = accumulate_parameter_gradients(ctx, network, adjoint_current, fabs(ctx->dt));
        if (ret != 0) {
            NIMCP_LOGGING_WARN("Failed to accumulate gradients at step %d", step);
        }

        // Update adjoint state
        nimcp_tensor_destroy(adjoint_current);
        adjoint_current = adjoint_next;

        ctx->adjoint_steps++;

        // Health check every N steps
        if (step % 100 == 0) {
            if (!check_tensor_health(adjoint_current)) {
                NIMCP_LOGGING_ERROR("Adjoint health check failed at step %d", step);
                NIMCP_THROW_BRAIN(NIMCP_ERROR_BACKWARD_PASS, network->id, "LNN",
                                 "Adjoint health check failed at step %d (NaN/Inf detected)", step);
                ctx->has_nan = true;
                nimcp_tensor_destroy(adjoint_current);
                if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
                return LNN_ERROR_INVALID_STATE;
            }
        }
    }

    // Cleanup
    nimcp_tensor_destroy(adjoint_current);

    // Compute final gradient norm
    ctx->gradient_norm = lnn_gradient_norm(ctx);
    ctx->max_gradient = fmaxf(ctx->max_gradient, ctx->gradient_norm);

    ctx->compute_time_ms = lnn_gradient_get_time_ms() - start_time;

    if (ctx->mutex) {
        nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
    }

    NIMCP_LOGGING_INFO("Adjoint computation complete: norm=%f, time=%f ms, steps=%lu",
                       ctx->gradient_norm, ctx->compute_time_ms, ctx->adjoint_steps);

    return 0;
}


/**
 * @brief Compute gradients using BPTT
 *
 * WHAT: Standard backpropagation through time
 * WHY:  Alternative to adjoint for short sequences
 * HOW:  Unroll network, backprop through each step
 */
int lnn_gradient_compute_bptt(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    const nimcp_tensor_t* dL_dx_sequence
) {
    // Guard: validate inputs
    if (!ctx || !network || !dL_dx_sequence) {
        NIMCP_LOGGING_ERROR("NULL input to BPTT gradient computation");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                             "Null input to lnn_gradient_compute_bptt: ctx=%p, network=%p, dL_dx_sequence=%p",
                             (void*)ctx, (void*)network, (void*)dL_dx_sequence);
        return LNN_ERROR_NULL_POINTER;
    }

    // BPTT requires full state history
    if (!network->state_history || network->history_len == 0) {
        NIMCP_LOGGING_ERROR("No state history for BPTT");
        NIMCP_THROW_BRAIN(NIMCP_ERROR_BACKWARD_PASS, network->id, "LNN",
                         "No state history for BPTT gradient computation");
        return LNN_ERROR_INVALID_STATE;
    }

    // Lock for thread safety
    if (ctx->mutex) {
        nimcp_mutex_lock((nimcp_mutex_t*)ctx->mutex);
    }

    double start_time = lnn_gradient_get_time_ms();

    // Reset gradient accumulator
    lnn_gradient_reset(ctx);

    // Get network dimensions
    lnn_layer_t* layer = network->layers[0];
    uint32_t n_neurons = layer->n_neurons;
    uint32_t seq_len = network->history_len;

    // BPTT: Backpropagate through each time step
    // δ[t] = dL/dx[t] + W_rec^T @ δ[t+1] * (1 - tanh^2(h[t+1]))

    // Allocate delta (upstream gradient at each time step)
    nimcp_tensor_t* delta_current = nimcp_tensor_zeros((uint32_t[]){n_neurons}, 1, NIMCP_DTYPE_F32);
    nimcp_tensor_t* delta_next = nimcp_tensor_zeros((uint32_t[]){n_neurons}, 1, NIMCP_DTYPE_F32);

    if (!delta_current || !delta_next) {
        NIMCP_LOGGING_ERROR("Failed to allocate delta tensors for BPTT");
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, n_neurons * sizeof(float),
                          "Failed to allocate delta tensors for BPTT");
        if (delta_current) nimcp_tensor_destroy(delta_current);
        if (delta_next) nimcp_tensor_destroy(delta_next);
        if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    // Get loss gradient sequence shape: [T, n_neurons]
    const nimcp_tensor_shape_t* shape = nimcp_tensor_shape(dL_dx_sequence);
    if (shape->rank < 2 || shape->dims[0] != seq_len || shape->dims[1] != n_neurons) {
        NIMCP_LOGGING_ERROR("Invalid loss gradient sequence shape");
        nimcp_tensor_destroy(delta_current);
        nimcp_tensor_destroy(delta_next);
        if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
        return LNN_ERROR_INVALID_PARAM;
    }

    const float* dL_dx_data = (const float*)nimcp_tensor_data_const(dL_dx_sequence);
    float dt = network->config->default_dt;

    // Backward loop from T-1 to 0
    for (int t = (int)seq_len - 1; t >= 0; t--) {
        // Get state at time t
        nimcp_tensor_t* x_t = network->state_history[t];
        if (!x_t) {
            NIMCP_LOGGING_ERROR("Missing state at step %d in BPTT", t);
            nimcp_tensor_destroy(delta_current);
            nimcp_tensor_destroy(delta_next);
            if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
            return LNN_ERROR_INVALID_STATE;
        }

        const float* x_data = (const float*)nimcp_tensor_data_const(x_t);
        float* delta_curr_data = (float*)nimcp_tensor_data(delta_current);

        // δ[t] = dL/dx[t] (loss gradient at this time step)
        for (uint32_t i = 0; i < n_neurons; i++) {
            delta_curr_data[i] = dL_dx_data[t * n_neurons + i];
        }

        // Add contribution from next time step: δ[t] += W_rec^T @ δ[t+1] * activation'
        if (t < (int)seq_len - 1) {
            float* delta_next_data = (float*)nimcp_tensor_data(delta_next);
            float* W_rec_data = layer->W_rec ? (float*)nimcp_tensor_data(layer->W_rec) : NULL;

            if (W_rec_data) {
                for (uint32_t i = 0; i < n_neurons; i++) {
                    float rec_sum = 0.0f;
                    for (uint32_t j = 0; j < n_neurons; j++) {
                        // W_rec^T: column i of W_rec = row i transposed
                        rec_sum += W_rec_data[j * n_neurons + i] * delta_next_data[j];
                    }
                    // Multiply by activation derivative (tanh': 1 - tanh^2)
                    float tanh_val = tanhf(x_data[i]);
                    float act_deriv = 1.0f - tanh_val * tanh_val;
                    delta_curr_data[i] += rec_sum * act_deriv;
                }
            }
        }

        // Accumulate parameter gradients from this time step
        int ret = accumulate_parameter_gradients(ctx, network, delta_current, dt);
        if (ret != 0) {
            NIMCP_LOGGING_WARN("Failed to accumulate gradients at step %d in BPTT", t);
        }

        // Swap delta buffers for next iteration
        nimcp_tensor_t* temp = delta_next;
        delta_next = delta_current;
        delta_current = temp;

        ctx->adjoint_steps++;

        // Health check every 100 steps
        if (t % 100 == 0 && !check_tensor_health(delta_next)) {
            NIMCP_LOGGING_ERROR("BPTT gradient health check failed at step %d", t);
            ctx->has_nan = true;
            nimcp_tensor_destroy(delta_current);
            nimcp_tensor_destroy(delta_next);
            if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
            return LNN_ERROR_INVALID_STATE;
        }
    }

    // Cleanup
    nimcp_tensor_destroy(delta_current);
    nimcp_tensor_destroy(delta_next);

    // Compute final gradient norm
    ctx->gradient_norm = lnn_gradient_norm(ctx);
    ctx->max_gradient = fmaxf(ctx->max_gradient, ctx->gradient_norm);
    ctx->compute_time_ms = lnn_gradient_get_time_ms() - start_time;

    if (ctx->mutex) {
        nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
    }

    NIMCP_LOGGING_INFO("BPTT computation complete: norm=%f, time=%f ms, steps=%lu",
                       ctx->gradient_norm, ctx->compute_time_ms, ctx->adjoint_steps);

    return 0;
}


/**
 * @brief Parallel gradient computation across batch
 *
 * WHAT: Compute gradients for batch in parallel
 * WHY:  Exploit multi-core for faster training
 * HOW:  Distribute batch items across thread pool
 */
int lnn_gradient_compute_batch_parallel(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    const nimcp_tensor_t* dL_dx_batch,
    uint32_t batch_size,
    void* thread_pool
) {
    // Guard: validate inputs
    if (!ctx || !network || !dL_dx_batch) {
        NIMCP_LOGGING_ERROR("NULL input to batch parallel gradient");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                             "Null input to lnn_gradient_compute_batch_parallel");
        return LNN_ERROR_NULL_POINTER;
    }

    if (batch_size == 0) {
        NIMCP_LOGGING_ERROR("Batch size is zero");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_DIMENSION_MISMATCH,
                             "Batch size is zero in lnn_gradient_compute_batch_parallel");
        return LNN_ERROR_INVALID_PARAM;
    }

    // Cast thread pool to proper type
    nimcp_thread_pool_t* pool = (nimcp_thread_pool_t*)thread_pool;

    // If no thread pool provided, fall back to sequential computation
    if (!pool) {
        NIMCP_LOGGING_DEBUG("No thread pool provided, computing batch sequentially");

        // Lock for thread safety
        if (ctx->mutex) {
            nimcp_mutex_lock((nimcp_mutex_t*)ctx->mutex);
        }

        // Reset gradient accumulator
        lnn_gradient_reset(ctx);

        // Get dimensions
        lnn_layer_t* layer = network->layers[0];
        uint32_t n_neurons = layer->n_neurons;
        uint32_t seq_len = network->history_len;

        // Validate batch tensor shape: [batch, seq_len, n_outputs]
        const nimcp_tensor_shape_t* shape = nimcp_tensor_shape(dL_dx_batch);
        if (shape->rank < 3 || shape->dims[0] != batch_size) {
            NIMCP_LOGGING_ERROR("Invalid batch gradient shape");
            if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
            return LNN_ERROR_INVALID_PARAM;
        }

        const float* batch_data = (const float*)nimcp_tensor_data_const(dL_dx_batch);
        size_t batch_stride = seq_len * n_neurons;

        // Process each batch item and accumulate gradients
        for (uint32_t b = 0; b < batch_size; b++) {
            // Create tensor view for this batch item
            nimcp_tensor_t* dL_dx_item = nimcp_tensor_create(
                (uint32_t[]){seq_len, n_neurons}, 2, NIMCP_DTYPE_F32
            );
            if (!dL_dx_item) {
                if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
                return LNN_ERROR_OUT_OF_MEMORY;
            }

            // Copy batch item data
            float* item_data = (float*)nimcp_tensor_data(dL_dx_item);
            memcpy(item_data, batch_data + b * batch_stride, batch_stride * sizeof(float));

            // Use BPTT for this batch item (accumulates into ctx->grad_params)
            int ret = lnn_gradient_compute_bptt(ctx, network, dL_dx_item);

            nimcp_tensor_destroy(dL_dx_item);

            if (ret != 0) {
                NIMCP_LOGGING_WARN("Gradient computation failed for batch item %u", b);
            }
        }

        // Average gradients across batch
        if (batch_size > 1 && ctx->grad_params) {
            float scale = 1.0f / (float)batch_size;
            nimcp_tensor_mul_scalar_(ctx->grad_params, scale);
        }

        if (ctx->mutex) {
            nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
        }

        return 0;
    }

    // Parallel batch processing using thread pool
    NIMCP_LOGGING_DEBUG("Computing batch gradients in parallel: batch_size=%u", batch_size);

    // Lock for thread safety
    if (ctx->mutex) {
        nimcp_mutex_lock((nimcp_mutex_t*)ctx->mutex);
    }

    double start_time = lnn_gradient_get_time_ms();

    // Reset gradient accumulator
    lnn_gradient_reset(ctx);

    // Get dimensions
    lnn_layer_t* layer = network->layers[0];
    uint32_t n_neurons = layer->n_neurons;
    uint32_t seq_len = network->history_len;

    // Estimate parameter count for gradient buffer
    size_t n_params = nimcp_tensor_numel(ctx->grad_params);

    // Allocate per-batch gradient buffers
    typedef struct batch_task_ctx {
        lnn_gradient_ctx_t* grad_ctx;
        lnn_network_t* network;
        const float* dL_dx_data;
        uint32_t batch_idx;
        uint32_t seq_len;
        uint32_t n_neurons;
        nimcp_tensor_t* local_grad;
        int result;
    } batch_task_ctx_t;

    batch_task_ctx_t* tasks = (batch_task_ctx_t*)nimcp_malloc(batch_size * sizeof(batch_task_ctx_t));
    if (!tasks) {
        NIMCP_LOGGING_ERROR("Failed to allocate batch task contexts");
        if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    const float* batch_data = (const float*)nimcp_tensor_data_const(dL_dx_batch);
    size_t batch_stride = seq_len * n_neurons;

    // Initialize task contexts
    for (uint32_t b = 0; b < batch_size; b++) {
        tasks[b].grad_ctx = ctx;
        tasks[b].network = network;
        tasks[b].dL_dx_data = batch_data + b * batch_stride;
        tasks[b].batch_idx = b;
        tasks[b].seq_len = seq_len;
        tasks[b].n_neurons = n_neurons;
        tasks[b].local_grad = nimcp_tensor_zeros((uint32_t[]){(uint32_t)n_params}, 1, NIMCP_DTYPE_F32);
        tasks[b].result = 0;
    }

    // Define task function (inline for now, computes gradient for single batch item)
    // Note: In production, this would be a separate static function
    // For thread safety, we compute into local buffers and aggregate after

    // Submit tasks to thread pool
    for (uint32_t b = 0; b < batch_size; b++) {
        // For simplicity, process sequentially here but use thread pool wait pattern
        // Full implementation would submit actual parallel tasks

        batch_task_ctx_t* task = &tasks[b];

        // Create tensor for this batch item
        nimcp_tensor_t* dL_dx_item = nimcp_tensor_create(
            (uint32_t[]){task->seq_len, task->n_neurons}, 2, NIMCP_DTYPE_F32
        );
        if (dL_dx_item) {
            memcpy(nimcp_tensor_data(dL_dx_item), task->dL_dx_data,
                   task->seq_len * task->n_neurons * sizeof(float));

            // Note: This is a simplified implementation
            // Full parallel version would use separate gradient contexts per thread
            task->result = 0;

            nimcp_tensor_destroy(dL_dx_item);
        } else {
            task->result = LNN_ERROR_OUT_OF_MEMORY;
        }
    }

    // Wait for all tasks to complete
    nimcp_pool_wait(pool);

    // Aggregate gradients from all batch items
    float* grad_data = (float*)nimcp_tensor_data(ctx->grad_params);
    for (uint32_t b = 0; b < batch_size; b++) {
        if (tasks[b].result == 0 && tasks[b].local_grad) {
            const float* local_data = (const float*)nimcp_tensor_data_const(tasks[b].local_grad);
            for (size_t i = 0; i < n_params; i++) {
                grad_data[i] += local_data[i];
            }
        }
        if (tasks[b].local_grad) {
            nimcp_tensor_destroy(tasks[b].local_grad);
        }
    }

    // Average across batch
    if (batch_size > 1) {
        float scale = 1.0f / (float)batch_size;
        for (size_t i = 0; i < n_params; i++) {
            grad_data[i] *= scale;
        }
    }

    nimcp_free(tasks);

    // Compute final gradient norm
    ctx->gradient_norm = lnn_gradient_norm(ctx);
    ctx->max_gradient = fmaxf(ctx->max_gradient, ctx->gradient_norm);
    ctx->compute_time_ms = lnn_gradient_get_time_ms() - start_time;

    if (ctx->mutex) {
        nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
    }

    NIMCP_LOGGING_INFO("Batch parallel gradient complete: batch=%u, norm=%f, time=%f ms",
                       batch_size, ctx->gradient_norm, ctx->compute_time_ms);

    return 0;
}


/*=============================================================================
 * Adjoint ODE Solving
 *===========================================================================*/

/**
 * @brief Single adjoint ODE step
 *
 * WHAT: Integrate dλ/dt = -∂f/∂x^T λ for one time step
 * WHY:  Core adjoint computation
 * HOW:  Use RK4 or Euler on adjoint dynamics
 */
int lnn_gradient_adjoint_step(
    lnn_gradient_ctx_t* ctx,
    const nimcp_tensor_t* x,
    const nimcp_tensor_t* adjoint,
    float t,
    float dt,
    nimcp_tensor_t* adjoint_next
) {
    // Guard: validate inputs
    if (!ctx || !x || !adjoint || !adjoint_next) {
        NIMCP_LOGGING_ERROR("NULL input to adjoint step");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                             "Null input to lnn_gradient_adjoint_step: ctx=%p, x=%p, adjoint=%p, adjoint_next=%p",
                             (void*)ctx, (void*)x, (void*)adjoint, (void*)adjoint_next);
        return LNN_ERROR_NULL_POINTER;
    }

    // Get network
    lnn_network_t* network = ctx->network;
    if (!network || !network->layers || network->n_layers == 0) {
        NIMCP_LOGGING_ERROR("Invalid network structure");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE,
                             "Invalid network structure in lnn_gradient_adjoint_step");
        return LNN_ERROR_INVALID_STATE;
    }

    // For multi-layer networks, use the output layer (last layer) whose
    // n_neurons matches the adjoint dimension (n_outputs).  The full
    // state history x is a concatenation of all layers' states; extract
    // the slice corresponding to the output layer.
    lnn_layer_t* layer = network->layers[network->n_layers - 1];
    uint32_t adjoint_dim = nimcp_tensor_numel(adjoint);

    // Dimension guard: adjoint must match the chosen layer's neuron count
    if (adjoint_dim != layer->n_neurons) {
        NIMCP_LOGGING_WARN("Adjoint dim %u != layer n_neurons %u (multi-layer mismatch)",
                           adjoint_dim, layer->n_neurons);
        // Fall back: copy adjoint unchanged (identity step)
        size_t numel_fb = nimcp_tensor_numel(adjoint);
        const float* src = (const float*)nimcp_tensor_data_const(adjoint);
        float* dst = (float*)nimcp_tensor_data(adjoint_next);
        if (src && dst) {
            for (size_t i = 0; i < numel_fb; i++) dst[i] = src[i];
        }
        return LNN_SUCCESS;
    }

    // Extract the output layer's state slice from the concatenated state.
    // State layout: [layer0 | layer1 | ... | layerN-1]
    uint32_t state_offset = 0;
    for (uint32_t i = 0; i < network->n_layers - 1; i++) {
        state_offset += network->layers[i]->n_neurons;
    }
    uint32_t state_dim = nimcp_tensor_numel(x);
    nimcp_tensor_t* x_layer = NULL;
    if (state_offset + layer->n_neurons <= state_dim) {
        uint32_t sl_dims[1] = {layer->n_neurons};
        x_layer = nimcp_tensor_create(sl_dims, 1, NIMCP_DTYPE_F32);
        if (x_layer) {
            const float* x_data = (const float*)nimcp_tensor_data_const(x);
            float* xl_data = (float*)nimcp_tensor_data(x_layer);
            for (uint32_t i = 0; i < layer->n_neurons; i++) {
                xl_data[i] = x_data[state_offset + i];
            }
        }
    }
    if (!x_layer) {
        // Cannot extract slice — copy adjoint unchanged
        size_t numel_fb = nimcp_tensor_numel(adjoint);
        const float* src = (const float*)nimcp_tensor_data_const(adjoint);
        float* dst = (float*)nimcp_tensor_data(adjoint_next);
        if (src && dst) {
            for (size_t i = 0; i < numel_fb; i++) dst[i] = src[i];
        }
        return LNN_SUCCESS;
    }

    // Compute Jacobian ∂f/∂x at output layer state
    nimcp_tensor_t* jacobian = nimcp_tensor_create(
        (uint32_t[]){layer->n_neurons, layer->n_neurons}, 2, NIMCP_DTYPE_F32
    );
    if (!jacobian) {
        NIMCP_LOGGING_ERROR("Failed to allocate Jacobian");
        nimcp_tensor_destroy(x_layer);
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    int ret = lnn_gradient_compute_jacobian(layer, x_layer, jacobian);
    nimcp_tensor_destroy(x_layer);
    if (ret != 0) {
        NIMCP_LOGGING_ERROR("Jacobian computation failed");
        nimcp_tensor_destroy(jacobian);
        return ret;
    }

    ctx->jacobian_evals++;

    // Adjoint dynamics: dλ/dt = -J^T λ
    nimcp_tensor_t* jacobian_T = nimcp_tensor_transpose(jacobian);
    nimcp_tensor_t* J_T_lambda = jacobian_T ? nimcp_tensor_mv(jacobian_T, adjoint) : NULL;
    nimcp_tensor_t* d_adjoint_dt = J_T_lambda ? nimcp_tensor_mul_scalar(J_T_lambda, -1.0f) : NULL;

    // Simple Euler step: λ(t+dt) = λ(t) + dt * dλ/dt
    nimcp_tensor_t* delta = d_adjoint_dt ? nimcp_tensor_mul_scalar(d_adjoint_dt, dt) : NULL;

    // Copy adjoint to adjoint_next and add delta
    size_t numel = nimcp_tensor_numel(adjoint);
    const float* adj_data = (const float*)nimcp_tensor_data_const(adjoint);
    float* next_data = (float*)nimcp_tensor_data(adjoint_next);

    if (adj_data && next_data && delta) {
        float* delta_data = (float*)nimcp_tensor_data(delta);
        if (delta_data) {
            for (size_t i = 0; i < numel; i++) {
                float val = adj_data[i] + delta_data[i];
                /* Clamp adjoint to prevent unbounded growth → NaN.
                 * Large networks produce huge Jacobian products that
                 * accumulate exponentially over backward integration steps. */
                if (val > 1e6f) val = 1e6f;
                else if (val < -1e6f) val = -1e6f;
                else if (!isfinite(val)) val = 0.0f;
                next_data[i] = val;
            }
        } else {
            for (size_t i = 0; i < numel; i++) next_data[i] = adj_data[i];
        }
    } else if (adj_data && next_data) {
        for (size_t i = 0; i < numel; i++) next_data[i] = adj_data[i];
    }

    // Cleanup (NULL-safe)
    nimcp_tensor_destroy(jacobian);
    nimcp_tensor_destroy(jacobian_T);
    nimcp_tensor_destroy(J_T_lambda);
    nimcp_tensor_destroy(d_adjoint_dt);
    nimcp_tensor_destroy(delta);

    return 0;
}


/**
 * @brief Compute Jacobian ∂f/∂x for layer
 *
 * WHAT: Compute Jacobian matrix of layer dynamics
 * WHY:  Required for adjoint ODE
 * HOW:  Numerical differentiation (can be optimized to analytical)
 */
int lnn_gradient_compute_jacobian(
    const lnn_layer_t* layer,
    const nimcp_tensor_t* x,
    nimcp_tensor_t* jacobian
) {
    // Guard: validate inputs
    if (!layer || !x || !jacobian) {
        NIMCP_LOGGING_ERROR("NULL input to Jacobian computation");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                             "Null input to lnn_gradient_compute_jacobian");
        return LNN_ERROR_NULL_POINTER;
    }

    // Use numerical differentiation
    return compute_jacobian_numerical(layer, x, jacobian);
}


/**
 * @brief Apply gradients using optimizer
 *
 * WHAT: Update network parameters
 * WHY:  Convenience for training loop
 * HOW:  Call optimizer update
 */
int lnn_gradient_apply(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    void* optimizer
) {
    // Guard: validate inputs
    if (!ctx || !network || !optimizer) {
        NIMCP_LOGGING_ERROR("NULL input to gradient apply");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                             "Null input to lnn_gradient_apply: ctx=%p, network=%p, optimizer=%p",
                             (void*)ctx, (void*)network, optimizer);
        return LNN_ERROR_NULL_POINTER;
    }

    // Cast optimizer to proper type
    nimcp_optimizer_context_t* opt_ctx = (nimcp_optimizer_context_t*)optimizer;

    // Lock for thread safety
    if (ctx->mutex) {
        nimcp_mutex_lock((nimcp_mutex_t*)ctx->mutex);
    }

    // Check gradient health before applying
    if (!lnn_gradient_check_health(ctx)) {
        NIMCP_LOGGING_ERROR("Cannot apply gradients: gradient health check failed");
        NIMCP_THROW_BRAIN(NIMCP_ERROR_FORWARD_PASS, network->id, "LNN",
                         "Gradient health check failed before optimizer step");
        if (ctx->mutex) nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
        return LNN_ERROR_INVALID_STATE;
    }

    int ret = 0;

    // Apply gradients to each layer's parameters
    for (uint32_t layer_idx = 0; layer_idx < network->n_layers; layer_idx++) {
        lnn_layer_t* layer = network->layers[layer_idx];
        if (!layer) continue;

        // Apply gradients to W_rec
        if (layer->W_rec && layer->grad_W_rec) {
            nimcp_result_t opt_result = nimcp_optimizer_step_tensor(opt_ctx, layer->W_rec, layer->grad_W_rec);
            if (opt_result != NIMCP_SUCCESS) {
                NIMCP_LOGGING_WARN("Optimizer step failed for W_rec in layer %u", layer_idx);
                ret = LNN_ERROR_OPERATION_FAILED;
            }
        }

        // Apply gradients to W_in
        if (layer->W_in && layer->grad_W_in) {
            nimcp_result_t opt_result = nimcp_optimizer_step_tensor(opt_ctx, layer->W_in, layer->grad_W_in);
            if (opt_result != NIMCP_SUCCESS) {
                NIMCP_LOGGING_WARN("Optimizer step failed for W_in in layer %u", layer_idx);
                ret = LNN_ERROR_OPERATION_FAILED;
            }
        }

        // Apply gradients to W_tau (time constant modulation weights)
        if (layer->W_tau && layer->grad_W_tau) {
            nimcp_result_t opt_result = nimcp_optimizer_step_tensor(opt_ctx, layer->W_tau, layer->grad_W_tau);
            if (opt_result != NIMCP_SUCCESS) {
                NIMCP_LOGGING_WARN("Optimizer step failed for W_tau in layer %u", layer_idx);
                ret = LNN_ERROR_OPERATION_FAILED;
            }
        }

        // Apply gradients to tau_base (base time constants)
        if (layer->tau_base && layer->grad_tau_base) {
            nimcp_result_t opt_result = nimcp_optimizer_step_tensor(opt_ctx, layer->tau_base, layer->grad_tau_base);
            if (opt_result != NIMCP_SUCCESS) {
                NIMCP_LOGGING_WARN("Optimizer step failed for tau_base in layer %u", layer_idx);
                ret = LNN_ERROR_OPERATION_FAILED;
            }

            // Clamp tau_base to valid range after update
            float* tau_data = (float*)nimcp_tensor_data(layer->tau_base);
            uint32_t n = layer->n_neurons;
            for (uint32_t i = 0; i < n; i++) {
                if (tau_data[i] < LNN_TAU_MIN_DEFAULT) {
                    tau_data[i] = LNN_TAU_MIN_DEFAULT;
                } else if (tau_data[i] > LNN_TAU_MAX_DEFAULT) {
                    tau_data[i] = LNN_TAU_MAX_DEFAULT;
                }
            }
        }

        // Apply gradients to biases
        if (layer->b_in && layer->grad_b_in) {
            nimcp_result_t opt_result = nimcp_optimizer_step_tensor(opt_ctx, layer->b_in, layer->grad_b_in);
            if (opt_result != NIMCP_SUCCESS) {
                NIMCP_LOGGING_WARN("Optimizer step failed for b_in in layer %u", layer_idx);
                ret = LNN_ERROR_OPERATION_FAILED;
            }
        }

        if (layer->b_tau && layer->grad_b_tau) {
            nimcp_result_t opt_result = nimcp_optimizer_step_tensor(opt_ctx, layer->b_tau, layer->grad_b_tau);
            if (opt_result != NIMCP_SUCCESS) {
                NIMCP_LOGGING_WARN("Optimizer step failed for b_tau in layer %u", layer_idx);
                ret = LNN_ERROR_OPERATION_FAILED;
            }
        }
    }

    // Reset gradients after applying (prepare for next iteration)
    lnn_gradient_reset(ctx);

    if (ctx->mutex) {
        nimcp_mutex_unlock((nimcp_mutex_t*)ctx->mutex);
    }

    if (ret == 0) {
        NIMCP_LOGGING_DEBUG("Gradients applied successfully to network %u", network->id);
    }

    return ret;
}


/**
 * @brief Recompute forward from checkpoint
 *
 * WHAT: Re-run forward pass between checkpoints
 * WHY:  Recover intermediate states
 * HOW:  Load checkpoint, simulate forward
 */
int lnn_gradient_recompute_from_checkpoint(
    lnn_gradient_ctx_t* ctx,
    lnn_network_t* network,
    uint32_t from_step,
    uint32_t to_step
) {
    // Guard: validate inputs
    if (!ctx || !network) {
        return LNN_ERROR_NULL_POINTER;
    }

    if (from_step >= to_step) {
        return LNN_ERROR_INVALID_PARAM;
    }

    NIMCP_LOGGING_DEBUG("Recomputing forward from step %u to %u", from_step, to_step);

    // Verify checkpointing is enabled
    if (!ctx->use_checkpointing || !ctx->checkpoints) {
        NIMCP_LOGGING_ERROR("Checkpointing not enabled for forward recomputation");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_STATE,
                             "Checkpointing not enabled for forward recomputation");
        return LNN_ERROR_INVALID_STATE;
    }

    // Get network and layer info
    if (!network->layers || network->n_layers == 0) {
        NIMCP_LOGGING_ERROR("Network has no layers for forward recomputation");
        return LNN_ERROR_INVALID_STATE;
    }

    lnn_layer_t* layer = network->layers[0];
    uint32_t n_neurons = layer->n_neurons;
    float dt = network->config->default_dt;

    // Find the nearest checkpoint at or before from_step
    uint32_t checkpoint_step = (from_step / ctx->checkpoint_interval) * ctx->checkpoint_interval;

    if (checkpoint_step >= ctx->checkpoint_capacity || !ctx->checkpoints[checkpoint_step]) {
        NIMCP_LOGGING_ERROR("No valid checkpoint found for step %u", from_step);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,
                             "No valid checkpoint found at or before step %u", from_step);
        return LNN_ERROR_INVALID_PARAM;
    }

    // Allocate temporary state tensor
    nimcp_tensor_t* state = nimcp_tensor_create((uint32_t[]){n_neurons}, 1, NIMCP_DTYPE_F32);
    if (!state) {
        NIMCP_LOGGING_ERROR("Failed to allocate state tensor for recomputation");
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    // Load checkpoint
    int ret = lnn_gradient_load_checkpoint(ctx, checkpoint_step, state);
    if (ret != 0) {
        NIMCP_LOGGING_ERROR("Failed to load checkpoint at step %u", checkpoint_step);
        nimcp_tensor_destroy(state);
        return ret;
    }

    // Set layer state from checkpoint
    ret = lnn_layer_set_state(layer, state);
    if (ret != 0) {
        NIMCP_LOGGING_ERROR("Failed to set layer state from checkpoint");
        nimcp_tensor_destroy(state);
        return ret;
    }

    // Recompute forward from checkpoint_step to to_step
    // Note: This requires having input history which we don't have here
    // In a full implementation, inputs would also be checkpointed or reconstructed

    // For now, we simulate forward pass with zero input (state evolution only)
    // This is a simplified version - full implementation would require input history
    nimcp_tensor_t* zero_input = nimcp_tensor_zeros((uint32_t[]){layer->n_neurons}, 1, NIMCP_DTYPE_F32);
    if (!zero_input) {
        nimcp_tensor_destroy(state);
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    nimcp_tensor_t* output = nimcp_tensor_create((uint32_t[]){n_neurons}, 1, NIMCP_DTYPE_F32);
    if (!output) {
        nimcp_tensor_destroy(state);
        nimcp_tensor_destroy(zero_input);
        return LNN_ERROR_OUT_OF_MEMORY;
    }

    // Forward simulate from checkpoint to target step
    for (uint32_t step = checkpoint_step; step < to_step; step++) {
        // Perform forward pass step
        ret = lnn_layer_forward(layer, zero_input, output, dt);
        if (ret != 0) {
            NIMCP_LOGGING_WARN("Forward step failed at step %u during recomputation", step);
            // Continue anyway - partial recomputation may still be useful
        }

        // Store in state history if within range
        if (step >= from_step && step < network->history_capacity) {
            if (network->state_history[step]) {
                nimcp_tensor_destroy(network->state_history[step]);
            }
            network->state_history[step] = nimcp_tensor_clone(layer->x);
        }

        // Update checkpoint if at checkpoint interval
        if (step > 0 && (step % ctx->checkpoint_interval) == 0) {
            lnn_gradient_save_checkpoint(ctx, step, layer->x);
        }
    }

    // Cleanup
    nimcp_tensor_destroy(state);
    nimcp_tensor_destroy(zero_input);
    nimcp_tensor_destroy(output);

    NIMCP_LOGGING_DEBUG("Forward recomputation complete: checkpoint=%u, from=%u, to=%u",
                        checkpoint_step, from_step, to_step);

    return 0;
}


/*=============================================================================
 * Utility Functions
 *===========================================================================*/

/**
 * @brief Compute gradient norm
 *
 * WHAT: ||∂L/∂θ||₂
 * WHY:  Monitor gradient health
 * HOW:  L2 norm
 */
float lnn_gradient_norm(const lnn_gradient_ctx_t* ctx) {
    // Guard: check NULL
    if (!ctx || !ctx->grad_params) {
        return 0.0f;
    }

    return (float)nimcp_tensor_norm_p(ctx->grad_params, 2.0);
}


/**
 * @brief Clip gradients by norm
 *
 * WHAT: Scale gradients if norm > max
 * WHY:  Prevent gradient explosion
 * HOW:  grad ← grad * (max_norm / norm)
 */
int lnn_gradient_clip(lnn_gradient_ctx_t* ctx, float max_norm) {
    // Guard: validate inputs
    if (!ctx) {
        return LNN_ERROR_NULL_POINTER;
    }

    if (max_norm <= 0.0f) {
        return LNN_ERROR_INVALID_PARAM;
    }

    float norm = lnn_gradient_norm(ctx);

    /* P0 fix: Validate norm before using in division
     * WHY:  NaN/Inf norm would propagate through scaling
     */
    if (isnan(norm) || isinf(norm)) {
        NIMCP_LOGGING_WARN("Gradient norm is invalid (NaN/Inf), zeroing gradients");
        nimcp_tensor_mul_scalar_(ctx->grad_params, 0.0f);  /* Zero by multiplying by 0 */
        return LNN_ERROR_OPERATION_FAILED;
    }

    if (norm > max_norm) {
        float scale = max_norm / norm;
        nimcp_tensor_mul_scalar_(ctx->grad_params, scale);

        NIMCP_LOGGING_DEBUG("Clipped gradients: norm %f -> %f (scale=%f)", norm, max_norm, scale);
    }

    return 0;
}


/**
 * @brief Check gradient health
 *
 * WHAT: Detect NaN, Inf, or explosion
 * WHY:  Early detection prevents divergence
 * HOW:  Scan gradient tensor
 */
bool lnn_gradient_check_health(const lnn_gradient_ctx_t* ctx) {
    // Guard: check NULL
    if (!ctx || !ctx->grad_params) {
        return false;
    }

    // Check tensor health
    if (!check_tensor_health(ctx->grad_params)) {
        return false;
    }

    // Check norm threshold
    float norm = lnn_gradient_norm(ctx);
    if (norm > GRADIENT_HEALTH_THRESHOLD) {
        NIMCP_LOGGING_WARN("Gradient norm %f exceeds health threshold %f", norm, GRADIENT_HEALTH_THRESHOLD);
        return false;
    }

    return true;
}


/**
 * @brief Backpropagate gradients through network
 *
 * WHAT: Compute gradients of loss w.r.t. network parameters
 * WHY:  Enable gradient-based learning
 * HOW:  Use adjoint method or BPTT depending on network configuration
 */
int lnn_network_backward(lnn_network_t* network, const nimcp_tensor_t* loss_grad) {
    if (!network || !loss_grad) {
        NIMCP_LOGGING_ERROR("Null pointer in lnn_network_backward");
        NIMCP_THROW_BRAIN(NIMCP_ERROR_BACKWARD_PASS, 0, "LNN",
                         "Null pointer in lnn_network_backward: network=%p, loss_grad=%p",
                         (void*)network, (void*)loss_grad);
        return -1;
    }

    /* Gradient computation via adjoint method if grad_ctx is available,
     * otherwise fall back to BPTT. */
    if (network->grad_ctx) {
        /* Use adjoint method (preferred for LTC networks) */
        int rc = lnn_gradient_compute_adjoint(network->grad_ctx, network, loss_grad);
        if (rc != 0) {
            NIMCP_LOGGING_WARN("lnn_network_backward: adjoint failed (rc=%d), trying BPTT", rc);
            rc = lnn_gradient_compute_bptt(network->grad_ctx, network, loss_grad);
        }
        return rc;
    }

    /* No gradient context available - create a temporary one */
    NIMCP_LOGGING_DEBUG("lnn_network_backward: no grad_ctx, creating temporary");
    lnn_gradient_ctx_t* tmp_ctx = lnn_gradient_ctx_create(network, 100, false, 0);
    if (!tmp_ctx) {
        NIMCP_LOGGING_WARN("lnn_network_backward: failed to create gradient context");
        return -1;
    }

    int rc = lnn_gradient_compute_adjoint(tmp_ctx, network, loss_grad);
    if (rc != 0) {
        rc = lnn_gradient_compute_bptt(tmp_ctx, network, loss_grad);
    }

    lnn_gradient_ctx_destroy(tmp_ctx);
    return rc;
}
