/**
 * @file nimcp_attention.h
 * @brief Biology-based multihead attention mechanism
 *
 * WHAT: Implements cortical column-inspired parallel attention streams with thalamic gating
 * WHY:  Enables selective focus on relevant input features across multiple representation spaces
 * HOW:  Multiple attention heads (cortical columns) process input in parallel,
 *       modulated by thalamic gating for top-down control
 *
 * BIOLOGICAL INSPIRATION:
 * - Cortical Columns: Each attention head acts as a specialized processing column
 * - Thalamic Gating: Gate mechanism controls information flow (like thalamic relay)
 * - Salience Weighting: Biologically-inspired attention based on feature importance
 * - Parallel Streams: Multiple heads process different aspects simultaneously
 *
 * DESIGN PATTERNS:
 * - Strategy Pattern: Pluggable attention computation strategies
 * - Factory Pattern: Create attention heads with different configurations
 * - Composite Pattern: Combine multiple heads into unified system
 * - SRP: Each component has single, well-defined responsibility
 *
 * PERFORMANCE:
 * - SIMD-friendly memory layout (cache-aligned structures)
 * - Thread-local buffers for zero-contention computation
 * - Atomic statistics for lock-free monitoring
 * - Early returns to minimize computation
 */

#ifndef NIMCP_ATTENTION_H
#define NIMCP_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations
//=============================================================================

typedef struct attention_head_struct* attention_head_t;
typedef struct multihead_attention_struct* multihead_attention_t;

//=============================================================================
// Configuration Structures
//=============================================================================

/**
 * WHAT: Configuration for single attention head (cortical column)
 * WHY:  Each head specializes in different feature representations
 */
typedef struct {
    uint32_t input_dim;           // Input feature dimension
    uint32_t output_dim;          // Output dimension per head
    uint32_t key_dim;             // Key/query dimension (typically input_dim / num_heads)
    uint32_t value_dim;           // Value dimension (typically same as key_dim)
    float temperature;            // Softmax temperature (biological gain control)
    float dropout_rate;           // Dropout for regularization (0.0 = none)
} attention_head_config_t;

/**
 * WHAT: Configuration for multihead attention system
 * WHY:  Coordinates multiple parallel attention streams with gating
 */
typedef struct {
    uint32_t num_heads;           // Number of attention heads (cortical columns)
    uint32_t input_dim;           // Input feature dimension
    uint32_t output_dim;          // Final output dimension
    uint32_t sequence_length;     // Maximum sequence length
    bool use_thalamic_gate;       // Enable thalamic gating mechanism
    bool use_salience_weighting;  // Enable salience-based attention
    float gate_bias;              // Thalamic gate bias (default opening)
} multihead_attention_config_t;

/**
 * WHAT: Statistics for monitoring attention system
 * WHY:  Track performance and behavior for debugging and optimization
 */
typedef struct {
    uint64_t forward_calls;       // Number of forward passes
    float avg_attention_entropy;  // Average entropy of attention weights
    float avg_gate_activation;    // Average thalamic gate opening
    uint32_t active_heads;        // Number of currently active heads
    float computation_time_ms;    // Average forward pass time
} attention_stats_t;

//=============================================================================
// Attention Head API (Single Cortical Column)
//=============================================================================

/**
 * WHAT: Create single attention head (cortical column)
 * WHY:  Each head specializes in different representational space
 * @param config Head configuration
 * @return Attention head handle or NULL on failure
 * PERFORMANCE: ~10μs creation time, thread-safe
 */
attention_head_t attention_head_create(const attention_head_config_t* config);

/**
 * WHAT: Destroy attention head and free resources
 * WHY:  Clean up allocated memory
 * @param head Attention head to destroy
 * THREAD_SAFETY: Must not be called concurrently
 */
void attention_head_destroy(attention_head_t head);

/**
 * WHAT: Compute attention for single head
 * WHY:  Projects input to Q/K/V, computes attention weights, applies to values
 * @param head Attention head
 * @param query Query vectors [sequence_length × input_dim]
 * @param key Key vectors [sequence_length × input_dim]
 * @param value Value vectors [sequence_length × input_dim]
 * @param sequence_length Length of sequences
 * @param output Output buffer [sequence_length × output_dim]
 * @param attention_weights Output attention weights [sequence_length × sequence_length] (optional)
 * @return true on success, false on error
 * PERFORMANCE: ~50μs for 128-dim, 32-length sequence
 * THREAD_SAFETY: Not thread-safe, use separate heads per thread
 */
bool attention_head_forward(attention_head_t head,
                           const float* query,
                           const float* key,
                           const float* value,
                           uint32_t sequence_length,
                           float* output,
                           float* attention_weights);

//=============================================================================
// Multihead Attention API (Cortical Column System)
//=============================================================================

/**
 * WHAT: Create multihead attention system
 * WHY:  Coordinate multiple parallel attention streams with gating
 * @param config System configuration
 * @return Multihead attention handle or NULL on failure
 * PERFORMANCE: ~100μs creation time, thread-safe
 */
multihead_attention_t multihead_attention_create(const multihead_attention_config_t* config);

/**
 * WHAT: Destroy multihead attention system
 * WHY:  Clean up all heads and resources
 * @param mha Multihead attention system
 * THREAD_SAFETY: Must not be called concurrently
 */
void multihead_attention_destroy(multihead_attention_t mha);

/**
 * WHAT: Forward pass through multihead attention
 * WHY:  Process input through all heads in parallel, apply gating, aggregate outputs
 * @param mha Multihead attention system
 * @param input Input sequence [sequence_length × input_dim]
 * @param sequence_length Length of input sequence
 * @param salience Salience scores per token (optional) [sequence_length]
 * @param output Output sequence [sequence_length × output_dim]
 * @return true on success, false on error
 * PERFORMANCE: ~200μs for 8 heads, 128-dim, 32-length sequence
 * THREAD_SAFETY: Not thread-safe, use separate systems per thread
 */
bool multihead_attention_forward(multihead_attention_t mha,
                                const float* input,
                                uint32_t sequence_length,
                                const float* salience,
                                float* output);

/**
 * WHAT: Update thalamic gate state
 * WHY:  Control top-down attention modulation (executive control)
 * @param mha Multihead attention system
 * @param gate_signal Gate control signal [0.0-1.0]
 * @return true on success
 * PERFORMANCE: ~1μs, lock-free
 */
bool multihead_attention_set_gate(multihead_attention_t mha, float gate_signal);

/**
 * WHAT: Get attention statistics
 * WHY:  Monitor system behavior and performance
 * @param mha Multihead attention system
 * @param stats Output statistics structure
 * @return true on success
 * THREAD_SAFETY: Thread-safe read operation
 */
bool multihead_attention_get_stats(multihead_attention_t mha, attention_stats_t* stats);

/**
 * WHAT: Reset attention statistics
 * WHY:  Clear accumulated statistics for new measurement period
 * @param mha Multihead attention system
 */
void multihead_attention_reset_stats(multihead_attention_t mha);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * WHAT: Compute attention entropy
 * WHY:  Measure how focused vs diffuse attention is
 * @param attention_weights Attention weight matrix [seq_len × seq_len]
 * @param sequence_length Length of sequence
 * @return Entropy value (lower = more focused, higher = more diffuse)
 * PERFORMANCE: ~5μs for 32-length sequence
 */
float attention_compute_entropy(const float* attention_weights, uint32_t sequence_length);

/**
 * WHAT: Validate attention configuration
 * WHY:  Catch configuration errors early
 * @param config Configuration to validate
 * @return true if valid, false otherwise
 */
bool attention_validate_config(const multihead_attention_config_t* config);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_ATTENTION_H
