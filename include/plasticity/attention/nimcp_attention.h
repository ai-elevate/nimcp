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
#include "utils/encoding/nimcp_positional_encoding.h"

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

    /* WHAT: Positional encoding configuration (Strategy Pattern)
     * WHY:  Allow runtime selection of RoPE vs ALiBi vs none
     * HOW:  Encoder created based on pe_type, applied in forward pass
     */
    bool use_positional_encoding; // Enable positional encoding
    nimcp_pos_encoding_type_t pe_type;  // PE type: NIMCP_POS_ROTARY or NIMCP_POS_ALIBI
    float rope_base;              // RoPE frequency base (default: 10000.0)
    float alibi_slope_base;       // ALiBi slope base for geometric sequence

    /* WHAT: Quantum attention acceleration configuration
     * WHY:  Enable O(√N) speedup for attention head selection
     * HOW:  Grover-inspired amplitude amplification for head selection
     */
    bool enable_quantum_attention; // Enable quantum-accelerated attention (default: true)
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
 * @param pos_encoder Positional encoder for RoPE/ALiBi (optional, NULL = no PE)
 * @param head_idx Head index for ALiBi bias (only used if pos_encoder is ALiBi type)
 * @return true on success, false on error
 * PERFORMANCE: ~50μs for 128-dim, 32-length sequence (without PE)
 * THREAD_SAFETY: Not thread-safe, use separate heads per thread
 */
bool attention_head_forward(attention_head_t head,
                           const float* query,
                           const float* key,
                           const float* value,
                           uint32_t sequence_length,
                           float* output,
                           float* attention_weights,
                           nimcp_pos_encoder_t* pos_encoder,
                           uint32_t head_idx);

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

/**
 * WHAT: Get average attention strength across all heads
 * WHY:  Enable attention-gated working memory (attended items more salient)
 * HOW:  Average gate activation across heads, bounded [0.0-1.0]
 *
 * @param mha Multihead attention system
 * @return Average attention strength [0.0-1.0], or 0.0 if NULL
 *
 * COMPLEXITY: O(1)
 * THREAD_SAFETY: Thread-safe read operation
 *
 * BIOLOGICAL BASIS:
 * - Attended stimuli have enhanced cortical representation
 * - Attention gates what enters working memory (PFC)
 * - Inattentional blindness: unattended items don't reach awareness
 *
 * USAGE:
 * ```c
 * float attention_strength = multihead_attention_get_strength(brain->multihead_attention);
 * float salience = base_salience + (attention_strength * 0.3f);  // Boost by up to 30%
 * working_memory_add(brain->working_memory, features, num_features, salience);
 * ```
 */
float multihead_attention_get_strength(multihead_attention_t mha);

//=============================================================================
// Positional Encoding Integration (Strategy Pattern)
//=============================================================================

/**
 * WHAT: Set positional encoding type for attention system
 * WHY:  Allow runtime switching between RoPE and ALiBi strategies
 * HOW:  Destroy old encoder, create new one with specified type
 *
 * @param mha Multihead attention system
 * @param pe_type Encoding type (NIMCP_POS_ROTARY or NIMCP_POS_ALIBI)
 * @return true on success, false on error
 *
 * PERFORMANCE: ~100μs (encoder creation overhead)
 * THREAD_SAFETY: Not thread-safe, call before forward passes
 *
 * BIOLOGICAL BASIS:
 * - Different brain regions use different spatial coding schemes
 * - Place cells: absolute position (analogous to learned PE)
 * - Grid cells: relative position (analogous to RoPE)
 * - Distance coding in parietal cortex (analogous to ALiBi)
 *
 * USAGE:
 * ```c
 * // Use RoPE for long-range dependencies
 * multihead_attention_set_pe_type(mha, NIMCP_POS_ROTARY);
 *
 * // Use ALiBi for efficiency on very long sequences
 * multihead_attention_set_pe_type(mha, NIMCP_POS_ALIBI);
 * ```
 */
bool multihead_attention_set_pe_type(multihead_attention_t mha,
                                     nimcp_pos_encoding_type_t pe_type);

/**
 * WHAT: Apply RoPE to query/key projections (internal helper)
 * WHY:  Inject relative position information into attention computation
 * HOW:  Rotate Q/K by position-dependent angles before scoring
 *
 * @param mha Multihead attention system (must have NIMCP_POS_ROTARY encoder)
 * @param query_proj Query projections [seq_len × key_dim]
 * @param key_proj Key projections [seq_len × key_dim]
 * @param seq_length Sequence length
 * @param key_dim Key dimension
 * @param query_out Rotated query output [seq_len × key_dim]
 * @param key_out Rotated key output [seq_len × key_dim]
 * @return true on success, false on error
 *
 * PERFORMANCE: ~O(seq_len × key_dim)
 * COMPLEXITY: Applied per-head in attention_head_forward()
 *
 * THEORETICAL BASIS:
 * - Su et al. (2021): RoFormer - Enhanced Transformer with Rotary Position Embedding
 * - Encodes relative position via rotation: q^T k = f(q, m) f(k, n)^T
 * - Rotation angle proportional to position difference (m - n)
 * - Maintains dot product structure while encoding position
 */
bool multihead_attention_apply_rope(multihead_attention_t mha,
                                   const float* query_proj,
                                   const float* key_proj,
                                   uint32_t seq_length,
                                   uint32_t key_dim,
                                   float* query_out,
                                   float* key_out);

/**
 * WHAT: Get ALiBi attention bias matrix
 * WHY:  Add position-dependent bias to attention scores
 * HOW:  Compute linear distance penalties: bias[i][j] = -slope * |i - j|
 *
 * @param mha Multihead attention system (must have NIMCP_POS_ALIBI encoder)
 * @param seq_length Sequence length
 * @param bias_out Output bias matrix [num_heads × seq_length × seq_length]
 * @return true on success, false on error
 *
 * PERFORMANCE: ~O(num_heads × seq_length^2)
 * USAGE: Add to attention scores before softmax
 *
 * THEORETICAL BASIS:
 * - Press et al. (2022): Train Short, Test Long - ALiBi
 * - Linear bias encourages nearby tokens: attention[i][j] += bias[i][j]
 * - Different slopes per head: slope[h] = 2^(-8(h+1)/num_heads)
 * - Extrapolates to longer sequences than training length
 *
 * BIOLOGICAL ANALOGY:
 * - Distance-dependent synaptic strength in cortex
 * - Lateral inhibition: nearby neurons compete, distant less so
 * - Temporal discounting in working memory
 */
bool multihead_attention_get_alibi_bias(multihead_attention_t mha,
                                       uint32_t seq_length,
                                       float* bias_out);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_ATTENTION_H
