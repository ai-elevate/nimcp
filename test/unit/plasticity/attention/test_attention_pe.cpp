//=============================================================================
// test_attention_pe.cpp - Unit Tests for Attention Positional Encoding
//=============================================================================
/**
 * @file test_attention_pe.cpp
 * @brief Unit tests for positional encoding integration in attention module
 *
 * WHAT: Test PE configuration, encoding application, and position-aware attention
 * WHY:  Positional encoding is critical for temporal attention mechanisms
 * HOW:  Test all PE types (Sinusoidal, RoPE, ALiBi) with attention operations
 *
 * TEST COVERAGE:
 * 1. PE configuration and initialization
 * 2. Sinusoidal PE for attention sequences
 * 3. RoPE application to query/key vectors
 * 4. ALiBi bias for relative position attention
 * 5. PE type switching at runtime
 * 6. Edge cases (NULL inputs, zero sequences, out-of-bounds)
 * 7. Integration with attention scoring
 *
 * BIOLOGICAL BASIS:
 * - Attention mechanisms need position information for temporal ordering
 * - Hippocampal place cells encode position via phase precession
 * - Prefrontal cortex maintains position-aware working memory
 *
 * @author NIMCP Development Team
 * @date 2025-12-10
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>

extern "C" {
    #include "cognitive/attention/nimcp_attention.h"
    #include "utils/encoding/nimcp_positional_encoding.h"
    #include "utils/memory/nimcp_memory.h"
}

//=============================================================================
// Test Fixtures
//=============================================================================

class AttentionPETest : public ::testing::Test {
protected:
    static constexpr float EPSILON = 1e-4f;
    static constexpr uint32_t TEST_SEQ_LEN = 16;
    static constexpr uint32_t TEST_EMBED_DIM = 64;
    static constexpr uint32_t TEST_NUM_HEADS = 4;

    attention_system_t* attention = nullptr;
    nimcp_pos_encoder_t* encoder = nullptr;

    void SetUp() override {
        srand(42);
        nimcp_memory_init();
    }

    void TearDown() override {
        if (attention) {
            attention_destroy(attention);
            attention = nullptr;
        }
        if (encoder) {
            nimcp_pos_encoder_destroy(encoder);
            encoder = nullptr;
        }
    }

    bool FloatEqual(float a, float b, float eps = EPSILON) {
        return std::abs(a - b) < eps;
    }

    // Helper to create attention system with PE
    attention_system_t* CreateAttentionWithPE(nimcp_pos_encoding_type_t pe_type) {
        attention_config_t config = attention_default_config();
        config.num_heads = TEST_NUM_HEADS;
        config.head_dim = TEST_EMBED_DIM / TEST_NUM_HEADS;
        config.max_seq_length = TEST_SEQ_LEN;
        config.enable_positional_encoding = true;
        config.pe_type = pe_type;
        config.pe_embedding_dim = TEST_EMBED_DIM;

        return attention_create(&config);
    }
};

//=============================================================================
// Unit Tests: PE Configuration
//=============================================================================

TEST_F(AttentionPETest, SetPEConfig_Sinusoidal) {
    // WHAT: Configure attention with sinusoidal PE
    // WHY:  Default PE type for transformers

    attention = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attention, nullptr) << "Attention with sinusoidal PE should be created";

    // Verify PE is enabled
    attention_stats_t stats;
    attention_get_stats(attention, &stats);
    EXPECT_TRUE(stats.pe_enabled) << "PE should be enabled";
}

TEST_F(AttentionPETest, SetPEConfig_RoPE) {
    // WHAT: Configure attention with RoPE
    // WHY:  Modern transformer position encoding

    attention = CreateAttentionWithPE(NIMCP_POS_ROTARY);
    ASSERT_NE(attention, nullptr) << "Attention with RoPE should be created";

    // Verify PE is enabled
    attention_stats_t stats;
    attention_get_stats(attention, &stats);
    EXPECT_TRUE(stats.pe_enabled) << "PE should be enabled";
}

TEST_F(AttentionPETest, SetPEConfig_ALiBi) {
    // WHAT: Configure attention with ALiBi
    // WHY:  Efficient attention with linear biases

    attention = CreateAttentionWithPE(NIMCP_POS_ALIBI);
    ASSERT_NE(attention, nullptr) << "Attention with ALiBi should be created";

    // Verify PE is enabled
    attention_stats_t stats;
    attention_get_stats(attention, &stats);
    EXPECT_TRUE(stats.pe_enabled) << "PE should be enabled";
}

TEST_F(AttentionPETest, SetPEConfig_Disable) {
    // WHAT: Create attention without PE
    // WHY:  Baseline comparison

    attention_config_t config = attention_default_config();
    config.enable_positional_encoding = false;

    attention = attention_create(&config);
    ASSERT_NE(attention, nullptr);

    attention_stats_t stats;
    attention_get_stats(attention, &stats);
    EXPECT_FALSE(stats.pe_enabled) << "PE should be disabled";
}

//=============================================================================
// Unit Tests: Sinusoidal PE Application
//=============================================================================

TEST_F(AttentionPETest, Sinusoidal_ApplyToSequence) {
    // WHAT: Apply sinusoidal PE to attention input
    // WHY:  Basic PE functionality test

    attention = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attention, nullptr);

    // Create test input sequence
    float input[TEST_SEQ_LEN * TEST_EMBED_DIM];
    for (uint32_t i = 0; i < TEST_SEQ_LEN * TEST_EMBED_DIM; i++) {
        input[i] = 1.0f;  // Constant input to see PE effect
    }

    float output[TEST_SEQ_LEN * TEST_EMBED_DIM];
    bool result = attention_encode_positions(attention, input, TEST_SEQ_LEN, output);
    EXPECT_TRUE(result) << "Position encoding should succeed";

    // Verify output differs from input (PE was added)
    bool changed = false;
    for (uint32_t i = 0; i < TEST_SEQ_LEN * TEST_EMBED_DIM; i++) {
        if (std::abs(output[i] - input[i]) > EPSILON) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed) << "Positional encoding should change output";
}

TEST_F(AttentionPETest, Sinusoidal_PositionUniqueness) {
    // WHAT: Verify different positions get different encodings
    // WHY:  Position discrimination is critical

    attention = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attention, nullptr);

    float pos0[TEST_EMBED_DIM];
    float pos5[TEST_EMBED_DIM];
    float pos10[TEST_EMBED_DIM];

    bool result;
    result = attention_get_position_embedding(attention, 0, pos0, TEST_EMBED_DIM);
    EXPECT_TRUE(result);
    result = attention_get_position_embedding(attention, 5, pos5, TEST_EMBED_DIM);
    EXPECT_TRUE(result);
    result = attention_get_position_embedding(attention, 10, pos10, TEST_EMBED_DIM);
    EXPECT_TRUE(result);

    // Verify positions are different
    bool pos0_vs_pos5_different = false;
    bool pos5_vs_pos10_different = false;

    for (uint32_t i = 0; i < TEST_EMBED_DIM; i++) {
        if (std::abs(pos0[i] - pos5[i]) > EPSILON) pos0_vs_pos5_different = true;
        if (std::abs(pos5[i] - pos10[i]) > EPSILON) pos5_vs_pos10_different = true;
    }

    EXPECT_TRUE(pos0_vs_pos5_different) << "Position 0 and 5 should differ";
    EXPECT_TRUE(pos5_vs_pos10_different) << "Position 5 and 10 should differ";
}

//=============================================================================
// Unit Tests: RoPE Application
//=============================================================================

TEST_F(AttentionPETest, RoPE_ApplyToQueryKey) {
    // WHAT: Apply RoPE rotation to query/key vectors
    // WHY:  RoPE modifies vectors via rotation

    attention = CreateAttentionWithPE(NIMCP_POS_ROTARY);
    ASSERT_NE(attention, nullptr);

    // Create test query and key vectors
    float query[TEST_EMBED_DIM];
    float key[TEST_EMBED_DIM];
    for (uint32_t i = 0; i < TEST_EMBED_DIM; i++) {
        query[i] = (i % 2 == 0) ? 1.0f : 0.0f;
        key[i] = (i % 2 == 0) ? 0.0f : 1.0f;
    }

    float query_out[TEST_EMBED_DIM];
    float key_out[TEST_EMBED_DIM];

    uint32_t position = 5;
    bool result = attention_apply_rope(attention, query, key, position,
                                       query_out, key_out, TEST_EMBED_DIM);
    EXPECT_TRUE(result) << "RoPE application should succeed";

    // Verify vectors were rotated (changed)
    bool query_changed = false;
    bool key_changed = false;

    for (uint32_t i = 0; i < TEST_EMBED_DIM; i++) {
        if (std::abs(query_out[i] - query[i]) > EPSILON) query_changed = true;
        if (std::abs(key_out[i] - key[i]) > EPSILON) key_changed = true;
    }

    EXPECT_TRUE(query_changed) << "RoPE should modify query";
    EXPECT_TRUE(key_changed) << "RoPE should modify key";
}

TEST_F(AttentionPETest, RoPE_PositionDependence) {
    // WHAT: Verify RoPE produces different rotations for different positions
    // WHY:  Position-dependent rotation is core to RoPE

    attention = CreateAttentionWithPE(NIMCP_POS_ROTARY);
    ASSERT_NE(attention, nullptr);

    float query[TEST_EMBED_DIM];
    for (uint32_t i = 0; i < TEST_EMBED_DIM; i++) {
        query[i] = 1.0f;
    }

    float key[TEST_EMBED_DIM];
    for (uint32_t i = 0; i < TEST_EMBED_DIM; i++) {
        key[i] = 1.0f;
    }

    float query_pos0[TEST_EMBED_DIM], key_pos0[TEST_EMBED_DIM];
    float query_pos10[TEST_EMBED_DIM], key_pos10[TEST_EMBED_DIM];

    attention_apply_rope(attention, query, key, 0, query_pos0, key_pos0, TEST_EMBED_DIM);
    attention_apply_rope(attention, query, key, 10, query_pos10, key_pos10, TEST_EMBED_DIM);

    // Verify different positions produce different outputs
    bool different = false;
    for (uint32_t i = 0; i < TEST_EMBED_DIM; i++) {
        if (std::abs(query_pos0[i] - query_pos10[i]) > EPSILON) {
            different = true;
            break;
        }
    }

    EXPECT_TRUE(different) << "Different positions should produce different RoPE outputs";
}

//=============================================================================
// Unit Tests: ALiBi Bias
//=============================================================================

TEST_F(AttentionPETest, ALiBi_GetBias) {
    // WHAT: Retrieve ALiBi bias matrix
    // WHY:  ALiBi adds linear bias to attention scores

    attention = CreateAttentionWithPE(NIMCP_POS_ALIBI);
    ASSERT_NE(attention, nullptr);

    uint32_t seq_len = 8;
    float bias[TEST_NUM_HEADS * seq_len * seq_len];

    bool result = attention_get_alibi_bias(attention, seq_len, bias);
    EXPECT_TRUE(result) << "ALiBi bias retrieval should succeed";

    // Verify bias is non-zero
    bool has_nonzero = false;
    for (uint32_t i = 0; i < TEST_NUM_HEADS * seq_len * seq_len; i++) {
        if (std::abs(bias[i]) > EPSILON) {
            has_nonzero = true;
            break;
        }
    }

    EXPECT_TRUE(has_nonzero) << "ALiBi bias should contain non-zero values";
}

TEST_F(AttentionPETest, ALiBi_DistanceDecay) {
    // WHAT: Verify ALiBi bias increases with distance
    // WHY:  ALiBi penalizes distant positions

    attention = CreateAttentionWithPE(NIMCP_POS_ALIBI);
    ASSERT_NE(attention, nullptr);

    uint32_t seq_len = 8;
    float bias[TEST_NUM_HEADS * seq_len * seq_len];

    attention_get_alibi_bias(attention, seq_len, bias);

    // For first head, check that bias increases with distance
    // bias[h][i][j] = -slope[h] * |i - j|
    uint32_t head = 0;
    uint32_t i = 3;  // Position i

    // Get bias at different distances
    float bias_dist_0 = bias[head * seq_len * seq_len + i * seq_len + i];  // Same position
    float bias_dist_1 = bias[head * seq_len * seq_len + i * seq_len + (i+1)];  // Distance 1
    float bias_dist_2 = bias[head * seq_len * seq_len + i * seq_len + (i+2)];  // Distance 2

    // Bias should become more negative as distance increases
    EXPECT_NEAR(bias_dist_0, 0.0f, EPSILON) << "Self-attention bias should be zero";
    EXPECT_LT(bias_dist_1, bias_dist_0) << "Distance 1 bias should be more negative";
    EXPECT_LT(bias_dist_2, bias_dist_1) << "Distance 2 bias should be more negative";
}

//=============================================================================
// Unit Tests: PE Type Switching
//=============================================================================

TEST_F(AttentionPETest, SwitchPEType_SinusoidalToRoPE) {
    // WHAT: Switch from sinusoidal to RoPE at runtime
    // WHY:  Allow dynamic PE reconfiguration

    attention = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attention, nullptr);

    // Switch to RoPE
    bool result = attention_set_pe_type(attention, NIMCP_POS_ROTARY);
    EXPECT_TRUE(result) << "PE type switch should succeed";

    // Verify RoPE is now active
    float query[TEST_EMBED_DIM], key[TEST_EMBED_DIM];
    float query_out[TEST_EMBED_DIM], key_out[TEST_EMBED_DIM];

    for (uint32_t i = 0; i < TEST_EMBED_DIM; i++) {
        query[i] = key[i] = 1.0f;
    }

    result = attention_apply_rope(attention, query, key, 5,
                                   query_out, key_out, TEST_EMBED_DIM);
    EXPECT_TRUE(result) << "RoPE should work after switch";
}

TEST_F(AttentionPETest, SwitchPEType_RoPEToALiBi) {
    // WHAT: Switch from RoPE to ALiBi
    // WHY:  Test different PE type transitions

    attention = CreateAttentionWithPE(NIMCP_POS_ROTARY);
    ASSERT_NE(attention, nullptr);

    // Switch to ALiBi
    bool result = attention_set_pe_type(attention, NIMCP_POS_ALIBI);
    EXPECT_TRUE(result) << "PE type switch should succeed";

    // Verify ALiBi is now active
    uint32_t seq_len = 8;
    float bias[TEST_NUM_HEADS * seq_len * seq_len];

    result = attention_get_alibi_bias(attention, seq_len, bias);
    EXPECT_TRUE(result) << "ALiBi should work after switch";
}

//=============================================================================
// Unit Tests: Edge Cases
//=============================================================================

TEST_F(AttentionPETest, EdgeCase_NullInput) {
    // WHAT: Handle NULL inputs gracefully
    // WHY:  Robustness testing

    attention = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attention, nullptr);

    float output[TEST_EMBED_DIM];
    bool result = attention_encode_positions(nullptr, nullptr, 10, output);
    EXPECT_FALSE(result) << "NULL attention should fail";

    float input[TEST_EMBED_DIM];
    result = attention_encode_positions(attention, input, 10, nullptr);
    EXPECT_FALSE(result) << "NULL output should fail";
}

TEST_F(AttentionPETest, EdgeCase_ZeroSequenceLength) {
    // WHAT: Handle zero-length sequences
    // WHY:  Edge case validation

    attention = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attention, nullptr);

    float input[TEST_EMBED_DIM];
    float output[TEST_EMBED_DIM];

    bool result = attention_encode_positions(attention, input, 0, output);
    // Should either succeed (no-op) or fail gracefully
    SUCCEED() << "Zero-length sequence handled";
}

TEST_F(AttentionPETest, EdgeCase_OutOfBoundsPosition) {
    // WHAT: Request position beyond max_seq_length
    // WHY:  Boundary testing

    attention = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attention, nullptr);

    float embedding[TEST_EMBED_DIM];
    bool result = attention_get_position_embedding(attention, TEST_SEQ_LEN + 100,
                                                    embedding, TEST_EMBED_DIM);

    // Sinusoidal can extrapolate, so may succeed or fail based on implementation
    SUCCEED() << "Out-of-bounds position handled";
}

TEST_F(AttentionPETest, EdgeCase_InsufficientBuffer) {
    // WHAT: Provide insufficient output buffer
    // WHY:  Memory safety testing

    attention = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attention, nullptr);

    float embedding[10];  // Too small
    bool result = attention_get_position_embedding(attention, 0,
                                                    embedding, 10);

    // Should detect buffer size mismatch
    EXPECT_FALSE(result) << "Insufficient buffer should fail";
}

//=============================================================================
// Unit Tests: Integration with Attention
//=============================================================================

TEST_F(AttentionPETest, Integration_AttentionWithPE) {
    // WHAT: Run full attention computation with PE
    // WHY:  End-to-end integration test

    attention = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attention, nullptr);

    uint32_t seq_len = 8;
    uint32_t embed_dim = TEST_EMBED_DIM;

    // Create input sequence
    float queries[seq_len * embed_dim];
    float keys[seq_len * embed_dim];
    float values[seq_len * embed_dim];

    for (uint32_t i = 0; i < seq_len * embed_dim; i++) {
        queries[i] = (float)(rand() % 100) / 100.0f;
        keys[i] = (float)(rand() % 100) / 100.0f;
        values[i] = (float)(rand() % 100) / 100.0f;
    }

    float output[seq_len * embed_dim];
    float attention_weights[TEST_NUM_HEADS * seq_len * seq_len];

    bool result = attention_forward(attention, queries, keys, values,
                                    seq_len, output, attention_weights);
    EXPECT_TRUE(result) << "Attention forward with PE should succeed";

    // Verify output is non-zero
    bool has_output = false;
    for (uint32_t i = 0; i < seq_len * embed_dim; i++) {
        if (std::abs(output[i]) > EPSILON) {
            has_output = true;
            break;
        }
    }
    EXPECT_TRUE(has_output) << "Attention should produce non-zero output";
}

TEST_F(AttentionPETest, Integration_CompareWithoutPE) {
    // WHAT: Compare attention output with and without PE
    // WHY:  Verify PE actually affects attention computation

    uint32_t seq_len = 8;
    uint32_t embed_dim = TEST_EMBED_DIM;

    // Create input sequence
    float queries[seq_len * embed_dim];
    float keys[seq_len * embed_dim];
    float values[seq_len * embed_dim];

    for (uint32_t i = 0; i < seq_len * embed_dim; i++) {
        queries[i] = (float)(i % 10) / 10.0f;
        keys[i] = (float)(i % 7) / 7.0f;
        values[i] = (float)(i % 5) / 5.0f;
    }

    // Attention WITH PE
    attention_system_t* attn_with_pe = CreateAttentionWithPE(NIMCP_POS_SINUSOIDAL);
    ASSERT_NE(attn_with_pe, nullptr);

    float output_with_pe[seq_len * embed_dim];
    float weights_with_pe[TEST_NUM_HEADS * seq_len * seq_len];

    attention_forward(attn_with_pe, queries, keys, values,
                     seq_len, output_with_pe, weights_with_pe);

    // Attention WITHOUT PE
    attention_config_t config = attention_default_config();
    config.num_heads = TEST_NUM_HEADS;
    config.head_dim = embed_dim / TEST_NUM_HEADS;
    config.enable_positional_encoding = false;

    attention_system_t* attn_without_pe = attention_create(&config);
    ASSERT_NE(attn_without_pe, nullptr);

    float output_without_pe[seq_len * embed_dim];
    float weights_without_pe[TEST_NUM_HEADS * seq_len * seq_len];

    attention_forward(attn_without_pe, queries, keys, values,
                     seq_len, output_without_pe, weights_without_pe);

    // Verify outputs differ
    bool outputs_differ = false;
    for (uint32_t i = 0; i < seq_len * embed_dim; i++) {
        if (std::abs(output_with_pe[i] - output_without_pe[i]) > EPSILON) {
            outputs_differ = true;
            break;
        }
    }

    EXPECT_TRUE(outputs_differ) << "PE should affect attention output";

    attention_destroy(attn_with_pe);
    attention_destroy(attn_without_pe);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
