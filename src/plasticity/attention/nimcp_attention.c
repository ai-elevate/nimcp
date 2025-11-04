/**
 * @file nimcp_attention.c
 * @brief Implementation of biology-based multihead attention
 *
 * WHAT: Cortical column-inspired parallel attention with thalamic gating
 * WHY:  Enable selective, biologically-plausible focus on relevant features
 * HOW:  Multiple specialized attention heads process input in parallel,
 *       modulated by thalamic gate for top-down control
 *
 * DESIGN: Strict Single Responsibility Principle
 * - Each function does ONE thing
 * - No nested if/for structures
 * - Early returns with guard clauses
 * - Extract helper functions liberally
 */

#include "nimcp_attention.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/containers/nimcp_vector.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/logging/nimcp_logging.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * WHAT: Single attention head (cortical column)
 * WHY:  Processes input in specialized representational space
 */
struct attention_head_struct {
    attention_head_config_t config;

    /* WHAT: Learned projection matrices (Q/K/V)
     * WHY:  Project input to query/key/value spaces
     */
    float* query_weights;     // [input_dim × key_dim]
    float* key_weights;       // [input_dim × key_dim]
    float* value_weights;     // [input_dim × value_dim]
    float* output_weights;    // [value_dim × output_dim]
};

/**
 * WHAT: Multihead attention system (cortical column network)
 * WHY:  Coordinate multiple parallel attention streams
 */
struct multihead_attention_struct {
    multihead_attention_config_t config;

    /* WHAT: Array of attention heads (cortical columns)
     * WHY:  Each head specializes in different features
     */
    attention_head_t* heads;

    /* WHAT: Thalamic gate state
     * WHY:  Top-down attention modulation
     */
    atomic_uint_fast32_t gate_signal;  // Fixed-point: [0, 1000] → [0.0, 1.0]

    /* WHAT: Statistics (atomic for thread-safe monitoring)
     * WHY:  Track system behavior without locks
     */
    atomic_uint_fast64_t forward_calls;
    atomic_uint_fast32_t avg_entropy_scaled;  // Fixed-point entropy
    atomic_uint_fast32_t avg_gate_scaled;     // Fixed-point gate
};

//=============================================================================
// Helper Functions - Matrix Operations
//=============================================================================

/**
 * WHAT: Matrix-vector multiplication: y = A * x
 * WHY:  Core linear algebra operation for projections
 * @param matrix Matrix A [m × n]
 * @param vector Vector x [n]
 * @param output Output y [m]
 * @param m Number of rows
 * @param n Number of columns
 * PERFORMANCE: ~O(m*n), SIMD-friendly loop order
 * SRP: Only does matrix-vector multiplication
 */
static void matmul_mv(const float* matrix, const float* vector,
                     float* output, uint32_t m, uint32_t n)
{
    /* WHAT: Guard against NULL pointers
     * WHY:  Prevent segfaults, enable early return
     */
    if (!matrix || !vector || !output) {
        return;
    }

    /* WHAT: Compute each output element
     * WHY:  y[i] = sum(A[i][j] * x[j])
     * NOTE: Single loop, no nesting
     */
    for (uint32_t i = 0; i < m; i++) {
        float sum = 0.0f;
        const float* row = matrix + (i * n);

        /* WHAT: Dot product of row with vector
         * WHY:  This is what matrix-vector multiplication means
         * PERFORMANCE: Compiler can auto-vectorize this
         */
        for (uint32_t j = 0; j < n; j++) {
            sum += row[j] * vector[j];
        }

        output[i] = sum;
    }
}

/**
 * WHAT: Compute dot product of two vectors
 * WHY:  Needed for attention score computation
 * @param a First vector [n]
 * @param b Second vector [n]
 * @param n Vector length
 * @return Dot product value
 * PERFORMANCE: ~O(n), SIMD-friendly
 * SRP: Only computes dot product
 */
static float dot_product(const float* a, const float* b, uint32_t n)
{
    /* WHAT: Guard clause for NULL
     * WHY:  Early return prevents nested if
     */
    if (!a || !b) {
        return 0.0f;
    }

    float sum = 0.0f;

    /* WHAT: Accumulate products
     * WHY:  dot(a,b) = sum(a[i] * b[i])
     */
    for (uint32_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

/**
 * WHAT: Apply softmax to vector in-place
 * WHY:  Convert attention scores to probability distribution
 * @param vector Vector to softmax [n]
 * @param n Vector length
 * @param temperature Softmax temperature (biological gain)
 * PERFORMANCE: ~O(n), numerically stable
 * SRP: Only does softmax transformation
 */
static void softmax_inplace(float* vector, uint32_t n, float temperature)
{
    /* WHAT: Guard clauses
     * WHY:  Early returns avoid nested ifs
     */
    if (!vector || n == 0) {
        return;
    }

    if (temperature <= 0.0f) {
        temperature = 1.0f;
    }

    /* WHAT: Find maximum for numerical stability
     * WHY:  Prevents overflow in exp()
     */
    float max_val = vector[0];
    for (uint32_t i = 1; i < n; i++) {
        if (vector[i] > max_val) {
            max_val = vector[i];
        }
    }

    /* WHAT: Compute exp(scaled values) and sum
     * WHY:  Numerically stable softmax computation
     */
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float scaled = (vector[i] - max_val) / temperature;
        vector[i] = expf(scaled);
        sum += vector[i];
    }

    /* WHAT: Normalize by sum
     * WHY:  Make it a probability distribution
     */
    if (sum > 0.0f) {
        for (uint32_t i = 0; i < n; i++) {
            vector[i] /= sum;
        }
    }
}

/**
 * WHAT: Apply element-wise scalar multiplication
 * WHY:  Scale vector by thalamic gate or salience
 * @param vector Vector to scale [n]
 * @param scalar Scaling factor
 * @param n Vector length
 * PERFORMANCE: ~O(n)
 * SRP: Only does element-wise scaling
 */
static void scale_vector(float* vector, float scalar, uint32_t n)
{
    if (!vector) {
        return;
    }

    for (uint32_t i = 0; i < n; i++) {
        vector[i] *= scalar;
    }
}

//=============================================================================
// Helper Functions - Projection & Attention
//=============================================================================

/**
 * WHAT: Project input sequence to query space
 * WHY:  Transform input for attention computation
 * @param head Attention head with query weights
 * @param input Input sequence [seq_len × input_dim]
 * @param seq_len Sequence length
 * @param output Query buffer [seq_len × key_dim]
 * PERFORMANCE: ~O(seq_len * input_dim * key_dim)
 * SRP: Only does query projection
 */
static void project_to_query(attention_head_t head,
                            const float* input,
                            uint32_t seq_len,
                            float* output)
{
    if (!head || !input || !output) {
        return;
    }

    const uint32_t input_dim = head->config.input_dim;
    const uint32_t key_dim = head->config.key_dim;

    /* WHAT: Project each token to query space
     * WHY:  Q = Input * W_q
     * NOTE: Single loop, no nesting
     */
    for (uint32_t t = 0; t < seq_len; t++) {
        const float* token = input + (t * input_dim);
        float* query = output + (t * key_dim);
        matmul_mv(head->query_weights, token, query, key_dim, input_dim);
    }
}

/**
 * WHAT: Project input sequence to key space
 * WHY:  Transform input for attention computation
 * SRP: Only does key projection
 */
static void project_to_key(attention_head_t head,
                          const float* input,
                          uint32_t seq_len,
                          float* output)
{
    if (!head || !input || !output) {
        return;
    }

    const uint32_t input_dim = head->config.input_dim;
    const uint32_t key_dim = head->config.key_dim;

    for (uint32_t t = 0; t < seq_len; t++) {
        const float* token = input + (t * input_dim);
        float* key = output + (t * key_dim);
        matmul_mv(head->key_weights, token, key, key_dim, input_dim);
    }
}

/**
 * WHAT: Project input sequence to value space
 * WHY:  Create values to be weighted by attention
 * SRP: Only does value projection
 */
static void project_to_value(attention_head_t head,
                            const float* input,
                            uint32_t seq_len,
                            float* output)
{
    if (!head || !input || !output) {
        return;
    }

    const uint32_t input_dim = head->config.input_dim;
    const uint32_t value_dim = head->config.value_dim;

    for (uint32_t t = 0; t < seq_len; t++) {
        const float* token = input + (t * input_dim);
        float* value = output + (t * value_dim);
        matmul_mv(head->value_weights, token, value, value_dim, input_dim);
    }
}

/**
 * WHAT: Compute attention scores for single query
 * WHY:  Measure relevance of each key to query
 * @param query Query vector [key_dim]
 * @param keys All key vectors [seq_len × key_dim]
 * @param seq_len Number of keys
 * @param key_dim Key dimension
 * @param scores Output scores [seq_len]
 * PERFORMANCE: ~O(seq_len * key_dim)
 * SRP: Only computes raw attention scores
 */
static void compute_attention_scores(const float* query,
                                    const float* keys,
                                    uint32_t seq_len,
                                    uint32_t key_dim,
                                    float* scores)
{
    if (!query || !keys || !scores) {
        return;
    }

    /* WHAT: Compute scaled dot-product for each key
     * WHY:  score(q, k) = (q · k) / sqrt(d_k)
     * NOTE: Single loop, no nesting
     */
    const float scale = 1.0f / sqrtf((float)key_dim);

    for (uint32_t i = 0; i < seq_len; i++) {
        const float* key = keys + (i * key_dim);
        scores[i] = dot_product(query, key, key_dim) * scale;
    }
}

/**
 * WHAT: Apply attention weights to values
 * WHY:  Weight values by attention distribution
 * @param attention_weights Attention distribution [seq_len]
 * @param values Value vectors [seq_len × value_dim]
 * @param seq_len Sequence length
 * @param value_dim Value dimension
 * @param output Weighted output [value_dim]
 * PERFORMANCE: ~O(seq_len * value_dim)
 * SRP: Only does weighted sum of values
 */
static void apply_attention_to_values(const float* attention_weights,
                                     const float* values,
                                     uint32_t seq_len,
                                     uint32_t value_dim,
                                     float* output)
{
    if (!attention_weights || !values || !output) {
        return;
    }

    /* WHAT: Initialize output to zero
     * WHY:  We'll accumulate weighted values
     */
    memset(output, 0, value_dim * sizeof(float));

    /* WHAT: Accumulate weighted values
     * WHY:  output = sum(attention[i] * value[i])
     * NOTE: Single loop, no nesting
     */
    for (uint32_t i = 0; i < seq_len; i++) {
        const float weight = attention_weights[i];
        const float* value = values + (i * value_dim);

        for (uint32_t d = 0; d < value_dim; d++) {
            output[d] += weight * value[d];
        }
    }
}

//=============================================================================
// Helper Functions - Biological Modulation
//=============================================================================

/**
 * WHAT: Apply salience-based weighting to attention scores
 * WHY:  Amplify attention to biologically important stimuli
 * @param scores Attention scores [seq_len]
 * @param salience Salience values [seq_len]
 * @param seq_len Sequence length
 * PERFORMANCE: ~O(seq_len)
 * SRP: Only applies salience weighting
 */
static void apply_salience_weighting(float* scores,
                                    const float* salience,
                                    uint32_t seq_len)
{
    if (!scores || !salience) {
        return;
    }

    /* WHAT: Multiply each score by salience
     * WHY:  Boost attention to salient features
     */
    for (uint32_t i = 0; i < seq_len; i++) {
        scores[i] *= salience[i];
    }
}

/**
 * WHAT: Apply thalamic gate to attention output
 * WHY:  Top-down modulation of attention strength
 * @param output Attention output [output_dim]
 * @param gate_signal Gate opening [0.0-1.0]
 * @param output_dim Output dimension
 * PERFORMANCE: ~O(output_dim)
 * SRP: Only applies gating
 */
static void apply_thalamic_gate(float* output,
                               float gate_signal,
                               uint32_t output_dim)
{
    if (!output) {
        return;
    }

    /* WHAT: Guard against invalid gate
     * WHY:  Clamp to valid range
     */
    if (gate_signal < 0.0f) {
        gate_signal = 0.0f;
    }
    if (gate_signal > 1.0f) {
        gate_signal = 1.0f;
    }

    /* WHAT: Scale output by gate
     * WHY:  Gate controls information flow (like thalamus)
     */
    scale_vector(output, gate_signal, output_dim);
}

//=============================================================================
// Attention Head Implementation
//=============================================================================

/**
 * WHAT: Initialize projection matrices with Xavier initialization
 * WHY:  Proper initialization prevents vanishing/exploding gradients
 * SRP: Only initializes weights
 */
static bool initialize_weights(attention_head_t head)
{
    if (!head) {
        return false;
    }

    const uint32_t input_dim = head->config.input_dim;
    const uint32_t key_dim = head->config.key_dim;
    const uint32_t value_dim = head->config.value_dim;
    const uint32_t output_dim = head->config.output_dim;

    /* WHAT: Xavier initialization scale
     * WHY:  Maintains variance across layers
     */
    const float query_scale = sqrtf(2.0f / (input_dim + key_dim));
    const float key_scale = sqrtf(2.0f / (input_dim + key_dim));
    const float value_scale = sqrtf(2.0f / (input_dim + value_dim));
    const float output_scale = sqrtf(2.0f / (value_dim + output_dim));

    /* WHAT: Initialize each weight matrix
     * WHY:  Random initialization for symmetry breaking
     * NOTE: Single-level loops, no nesting
     */
    for (uint32_t i = 0; i < input_dim * key_dim; i++) {
        head->query_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * query_scale;
    }

    for (uint32_t i = 0; i < input_dim * key_dim; i++) {
        head->key_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * key_scale;
    }

    for (uint32_t i = 0; i < input_dim * value_dim; i++) {
        head->value_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * value_scale;
    }

    for (uint32_t i = 0; i < value_dim * output_dim; i++) {
        head->output_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * output_scale;
    }

    return true;
}

attention_head_t attention_head_create(const attention_head_config_t* config)
{
    /* WHAT: Validate configuration
     * WHY:  Early return prevents allocation if invalid
     */
    if (!config) {
        NIMCP_LOGGING_ERROR("Attention head config is NULL");
        return NULL;
    }

    if (config->input_dim == 0 || config->output_dim == 0) {
        NIMCP_LOGGING_ERROR("Invalid dimensions in attention head config");
        return NULL;
    }

    /* WHAT: Allocate head structure
     * WHY:  Need storage for head state
     */
    attention_head_t head = nimcp_malloc(sizeof(struct attention_head_struct));
    if (!head) {
        NIMCP_LOGGING_ERROR("Failed to allocate attention head");
        return NULL;
    }

    memcpy(&head->config, config, sizeof(attention_head_config_t));

    /* WHAT: Allocate weight matrices
     * WHY:  Need storage for learned parameters
     */
    const uint32_t input_dim = config->input_dim;
    const uint32_t key_dim = config->key_dim;
    const uint32_t value_dim = config->value_dim;
    const uint32_t output_dim = config->output_dim;

    head->query_weights = nimcp_malloc(input_dim * key_dim * sizeof(float));
    head->key_weights = nimcp_malloc(input_dim * key_dim * sizeof(float));
    head->value_weights = nimcp_malloc(input_dim * value_dim * sizeof(float));
    head->output_weights = nimcp_malloc(value_dim * output_dim * sizeof(float));

    /* WHAT: Check all allocations succeeded
     * WHY:  Early cleanup if any allocation failed
     */
    if (!head->query_weights || !head->key_weights ||
        !head->value_weights || !head->output_weights) {
        attention_head_destroy(head);
        return NULL;
    }

    /* WHAT: Initialize weights
     * WHY:  Proper initialization for training
     */
    if (!initialize_weights(head)) {
        attention_head_destroy(head);
        return NULL;
    }

    NIMCP_LOGGING_DEBUG("Created attention head: input_dim=%u, output_dim=%u",
                       input_dim, output_dim);

    return head;
}

void attention_head_destroy(attention_head_t head)
{
    /* WHAT: Guard clause for NULL
     * WHY:  Early return if nothing to destroy
     */
    if (!head) {
        return;
    }

    /* WHAT: Free all allocated resources
     * WHY:  Prevent memory leaks
     * NOTE: nimcp_free handles NULL gracefully
     */
    nimcp_free(head->query_weights);
    nimcp_free(head->key_weights);
    nimcp_free(head->value_weights);
    nimcp_free(head->output_weights);
    nimcp_free(head);
}

bool attention_head_forward(attention_head_t head,
                           const float* query,
                           const float* key,
                           const float* value,
                           uint32_t sequence_length,
                           float* output,
                           float* attention_weights)
{
    /* WHAT: Validate all inputs
     * WHY:  Early returns prevent crashes
     */
    if (!head || !query || !key || !value || !output) {
        NIMCP_LOGGING_ERROR("NULL parameter in attention_head_forward");
        return false;
    }

    if (sequence_length == 0) {
        NIMCP_LOGGING_ERROR("Zero sequence length");
        return false;
    }

    /* WHAT: Get dimensions from config
     * WHY:  Used throughout computation
     */
    const uint32_t key_dim = head->config.key_dim;
    const uint32_t value_dim = head->config.value_dim;
    const uint32_t output_dim = head->config.output_dim;
    const float temperature = head->config.temperature;

    /* WHAT: Allocate temporary buffers
     * WHY:  Need workspace for Q/K/V projections
     */
    float* query_proj = nimcp_malloc(sequence_length * key_dim * sizeof(float));
    float* key_proj = nimcp_malloc(sequence_length * key_dim * sizeof(float));
    float* value_proj = nimcp_malloc(sequence_length * value_dim * sizeof(float));
    float* scores = nimcp_malloc(sequence_length * sizeof(float));
    float* output_proj = nimcp_malloc(value_dim * sizeof(float));

    /* WHAT: Check allocations
     * WHY:  Early cleanup if allocation failed
     */
    if (!query_proj || !key_proj || !value_proj || !scores || !output_proj) {
        nimcp_free(query_proj);
        nimcp_free(key_proj);
        nimcp_free(value_proj);
        nimcp_free(scores);
        nimcp_free(output_proj);
        return false;
    }

    /* WHAT: Project inputs to Q/K/V spaces
     * WHY:  Transform for attention computation
     */
    project_to_query(head, query, sequence_length, query_proj);
    project_to_key(head, key, sequence_length, key_proj);
    project_to_value(head, value, sequence_length, value_proj);

    /* WHAT: Compute attention for each query token
     * WHY:  Each output depends on all inputs via attention
     * NOTE: Single loop, no nesting
     */
    for (uint32_t t = 0; t < sequence_length; t++) {
        const float* query_vec = query_proj + (t * key_dim);

        /* WHAT: Compute attention scores
         * WHY:  Measure relevance of each key to this query
         */
        compute_attention_scores(query_vec, key_proj, sequence_length,
                                key_dim, scores);

        /* WHAT: Apply softmax to get attention distribution
         * WHY:  Convert scores to probabilities
         */
        softmax_inplace(scores, sequence_length, temperature);

        /* WHAT: Store attention weights if requested
         * WHY:  Useful for visualization and debugging
         */
        if (attention_weights) {
            memcpy(attention_weights + (t * sequence_length),
                  scores, sequence_length * sizeof(float));
        }

        /* WHAT: Apply attention to values
         * WHY:  Weighted combination of values
         */
        apply_attention_to_values(scores, value_proj, sequence_length,
                                 value_dim, output_proj);

        /* WHAT: Project to output space
         * WHY:  Final linear transformation
         */
        float* output_vec = output + (t * output_dim);
        matmul_mv(head->output_weights, output_proj, output_vec,
                 output_dim, value_dim);
    }

    /* WHAT: Free temporary buffers
     * WHY:  Prevent memory leaks
     */
    nimcp_free(query_proj);
    nimcp_free(key_proj);
    nimcp_free(value_proj);
    nimcp_free(scores);
    nimcp_free(output_proj);

    return true;
}

//=============================================================================
// Multihead Attention Implementation
//=============================================================================

multihead_attention_t multihead_attention_create(const multihead_attention_config_t* config)
{
    /* WHAT: Validate configuration
     * WHY:  Early return prevents allocation if invalid
     */
    if (!attention_validate_config(config)) {
        NIMCP_LOGGING_ERROR("Invalid multihead attention config");
        return NULL;
    }

    /* WHAT: Allocate system structure
     * WHY:  Need storage for multihead state
     */
    multihead_attention_t mha = nimcp_malloc(sizeof(struct multihead_attention_struct));
    if (!mha) {
        NIMCP_LOGGING_ERROR("Failed to allocate multihead attention");
        return NULL;
    }

    memcpy(&mha->config, config, sizeof(multihead_attention_config_t));

    /* WHAT: Initialize atomic counters
     * WHY:  Lock-free statistics tracking
     */
    atomic_init(&mha->forward_calls, 0);
    atomic_init(&mha->avg_entropy_scaled, 0);
    atomic_init(&mha->gate_signal, (uint32_t)(config->gate_bias * 1000.0f));

    /* WHAT: Allocate attention heads
     * WHY:  Need multiple parallel heads (cortical columns)
     */
    mha->heads = nimcp_malloc(config->num_heads * sizeof(attention_head_t));
    if (!mha->heads) {
        nimcp_free(mha);
        return NULL;
    }

    /* WHAT: Create each attention head
     * WHY:  Each head specializes in different features
     * NOTE: Single loop, no nesting
     */
    const uint32_t head_dim = config->input_dim / config->num_heads;

    for (uint32_t i = 0; i < config->num_heads; i++) {
        attention_head_config_t head_config = {
            .input_dim = config->input_dim,
            .output_dim = head_dim,
            .key_dim = head_dim,
            .value_dim = head_dim,
            .temperature = 1.0f,
            .dropout_rate = 0.0f
        };

        mha->heads[i] = attention_head_create(&head_config);

        /* WHAT: Check if head creation failed
         * WHY:  Early cleanup if any head fails
         */
        if (!mha->heads[i]) {
            multihead_attention_destroy(mha);
            return NULL;
        }
    }

    NIMCP_LOGGING_INFO("Created multihead attention: num_heads=%u, input_dim=%u",
                      config->num_heads, config->input_dim);

    return mha;
}

void multihead_attention_destroy(multihead_attention_t mha)
{
    /* WHAT: Guard clause for NULL
     * WHY:  Early return if nothing to destroy
     */
    if (!mha) {
        return;
    }

    /* WHAT: Destroy all attention heads
     * WHY:  Free head resources
     */
    if (mha->heads) {
        for (uint32_t i = 0; i < mha->config.num_heads; i++) {
            attention_head_destroy(mha->heads[i]);
        }
        nimcp_free(mha->heads);
    }

    nimcp_free(mha);
}

bool multihead_attention_forward(multihead_attention_t mha,
                                const float* input,
                                uint32_t sequence_length,
                                const float* salience,
                                float* output)
{
    /* WHAT: Validate inputs
     * WHY:  Early returns prevent crashes
     */
    if (!mha || !input || !output) {
        NIMCP_LOGGING_ERROR("NULL parameter in multihead_attention_forward");
        return false;
    }

    if (sequence_length == 0 || sequence_length > mha->config.sequence_length) {
        NIMCP_LOGGING_ERROR("Invalid sequence length");
        return false;
    }

    /* WHAT: Get configuration values
     * WHY:  Used throughout computation
     */
    const uint32_t num_heads = mha->config.num_heads;
    const uint32_t input_dim = mha->config.input_dim;
    const uint32_t output_dim = mha->config.output_dim;
    const uint32_t head_dim = input_dim / num_heads;

    /* WHAT: Get thalamic gate signal
     * WHY:  Top-down attention modulation
     */
    const uint32_t gate_scaled = atomic_load(&mha->gate_signal);
    const float gate = (float)gate_scaled / 1000.0f;

    /* WHAT: Allocate temporary buffers
     * WHY:  Need workspace for head outputs and attention weights (for entropy)
     */
    float* head_outputs = nimcp_malloc(num_heads * sequence_length * head_dim * sizeof(float));
    float* attention_weights = nimcp_malloc(sequence_length * sequence_length * sizeof(float));

    if (!head_outputs || !attention_weights) {
        nimcp_free(head_outputs);
        nimcp_free(attention_weights);
        return false;
    }

    /* WHAT: Process each head independently
     * WHY:  Each head computes attention in parallel (logically)
     * NOTE: Single loop, no nesting
     */
    for (uint32_t h = 0; h < num_heads; h++) {
        float* head_output = head_outputs + (h * sequence_length * head_dim);

        /* WHAT: Capture attention weights from first head for entropy calculation
         * WHY:  Track attention focus characteristics
         */
        float* weights_ptr = (h == 0) ? attention_weights : NULL;

        /* WHAT: Run forward pass for this head
         * WHY:  Compute attention-weighted representation
         */
        bool success = attention_head_forward(mha->heads[h],
                                             input, input, input,
                                             sequence_length,
                                             head_output,
                                             weights_ptr);

        /* WHAT: Check if forward pass failed
         * WHY:  Early cleanup and return on error
         */
        if (!success) {
            nimcp_free(head_outputs);
            nimcp_free(attention_weights);
            return false;
        }

        /* WHAT: Apply thalamic gate if enabled
         * WHY:  Top-down modulation of attention
         */
        if (mha->config.use_thalamic_gate) {
            apply_thalamic_gate(head_output, gate, sequence_length * head_dim);
        }

        /* WHAT: Apply salience weighting if provided
         * WHY:  Boost biologically important features
         */
        if (mha->config.use_salience_weighting && salience) {
            for (uint32_t t = 0; t < sequence_length; t++) {
                float* token_output = head_output + (t * head_dim);
                scale_vector(token_output, salience[t], head_dim);
            }
        }
    }

    /* WHAT: Concatenate head outputs
     * WHY:  Combine representations from all heads
     * NOTE: Single loop, no nesting
     */
    for (uint32_t t = 0; t < sequence_length; t++) {
        float* token_output = output + (t * output_dim);

        for (uint32_t h = 0; h < num_heads; h++) {
            const float* head_output = head_outputs + (h * sequence_length * head_dim) + (t * head_dim);
            memcpy(token_output + (h * head_dim), head_output, head_dim * sizeof(float));
        }
    }

    /* WHAT: Compute and update entropy statistics
     * WHY:  Track attention focus characteristics
     */
    float entropy = attention_compute_entropy(attention_weights, sequence_length);
    uint32_t entropy_scaled = (uint32_t)(entropy * 1000.0f);

    /* WHAT: Update running average of entropy (simple moving average)
     * WHY:  Track typical attention behavior over time
     */
    uint32_t old_entropy = atomic_load(&mha->avg_entropy_scaled);
    uint64_t call_count = atomic_load(&mha->forward_calls);
    uint32_t new_entropy = (old_entropy * call_count + entropy_scaled) / (call_count + 1);
    atomic_store(&mha->avg_entropy_scaled, new_entropy);

    /* WHAT: Update statistics
     * WHY:  Track system behavior
     */
    atomic_fetch_add(&mha->forward_calls, 1);

    /* WHAT: Free temporary buffers
     * WHY:  Prevent memory leaks
     */
    nimcp_free(head_outputs);
    nimcp_free(attention_weights);

    return true;
}

bool multihead_attention_set_gate(multihead_attention_t mha, float gate_signal)
{
    /* WHAT: Validate inputs
     * WHY:  Early return on invalid input
     */
    if (!mha) {
        return false;
    }

    /* WHAT: Clamp gate to valid range
     * WHY:  Prevent invalid values
     */
    if (gate_signal < 0.0f) {
        gate_signal = 0.0f;
    }
    if (gate_signal > 1.0f) {
        gate_signal = 1.0f;
    }

    /* WHAT: Store as fixed-point atomic
     * WHY:  Lock-free thread-safe update
     */
    const uint32_t gate_scaled = (uint32_t)(gate_signal * 1000.0f);
    atomic_store(&mha->gate_signal, gate_scaled);

    return true;
}

bool multihead_attention_get_stats(multihead_attention_t mha, attention_stats_t* stats)
{
    /* WHAT: Validate inputs
     * WHY:  Early return on invalid input
     */
    if (!mha || !stats) {
        return false;
    }

    /* WHAT: Read atomic counters
     * WHY:  Lock-free thread-safe read
     */
    stats->forward_calls = atomic_load(&mha->forward_calls);
    stats->avg_gate_activation = (float)atomic_load(&mha->gate_signal) / 1000.0f;
    stats->active_heads = mha->config.num_heads;
    stats->avg_attention_entropy = (float)atomic_load(&mha->avg_entropy_scaled) / 1000.0f;
    stats->computation_time_ms = 0.0f;  // TODO: Add timing

    return true;
}

void multihead_attention_reset_stats(multihead_attention_t mha)
{
    /* WHAT: Guard clause for NULL
     * WHY:  Early return if nothing to reset
     */
    if (!mha) {
        return;
    }

    /* WHAT: Reset all atomic counters
     * WHY:  Clear accumulated statistics
     */
    atomic_store(&mha->forward_calls, 0);
    atomic_store(&mha->avg_entropy_scaled, 0);
}

//=============================================================================
// Utility Functions
//=============================================================================

float attention_compute_entropy(const float* attention_weights, uint32_t sequence_length)
{
    /* WHAT: Guard clause for NULL
     * WHY:  Early return prevents crash
     */
    if (!attention_weights || sequence_length == 0) {
        return 0.0f;
    }

    /* WHAT: Compute entropy: H = -sum(p * log(p))
     * WHY:  Measure attention focus (lower = more focused)
     */
    float entropy = 0.0f;

    for (uint32_t i = 0; i < sequence_length; i++) {
        const float p = attention_weights[i];

        /* WHAT: Skip zero probabilities
         * WHY:  Avoid log(0) = -inf
         */
        if (p > 1e-10f) {
            entropy -= p * logf(p);
        }
    }

    return entropy;
}

bool attention_validate_config(const multihead_attention_config_t* config)
{
    /* WHAT: Guard clause for NULL
     * WHY:  Early return if no config
     */
    if (!config) {
        return false;
    }

    /* WHAT: Check required fields are non-zero
     * WHY:  Prevent division by zero and invalid allocations
     */
    if (config->num_heads == 0) {
        return false;
    }

    if (config->input_dim == 0) {
        return false;
    }

    if (config->output_dim == 0) {
        return false;
    }

    if (config->sequence_length == 0) {
        return false;
    }

    /* WHAT: Check input_dim is divisible by num_heads
     * WHY:  Each head needs equal dimension
     */
    if (config->input_dim % config->num_heads != 0) {
        return false;
    }

    return true;
}
