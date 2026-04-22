/**
 * @file test_cortex_cnn_thalamic.cpp
 * @brief Unit tests for the cortex CNN thalamic adapter
 *
 * TEST COVERAGE:
 * - cortex_cnn_attach_thalamic_router(proc, NULL): no channel, forward works.
 * - Attach with a real router: internal channel is non-NULL (via test-only
 *   accessor), destination 0 pre-registered, source_id == cortex type.
 * - Double attach destroys + recreates cleanly (still valid, still dest 0).
 * - Detach (passing NULL after attach) tears down the channel cleanly.
 * - Destroy with attached channel does not leak the borrowed router.
 * - Three tunables round-trip through setter/getter with nonzero-coerce.
 * - NULL proc: attach returns NIMCP_ERROR_NULL_POINTER.
 *
 * @date 2026-04-22
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <cstdint>

extern "C" {
#include "training/nimcp_cortex_cnn.h"
#include "core/thalamic/nimcp_thalamic_channel.h"
#include "middleware/routing/nimcp_thalamic_router.h"
#include "utils/error/nimcp_error_codes.h"

/* Test-only accessor — defined in nimcp_cortex_cnn.c next to the public
 * attach API. Reaches into the opaque processor struct so white-box
 * tests can verify the channel field without a public getter. */
thalamic_channel_t* cortex_cnn_test_get_thalamic_channel(
    const cortex_cnn_processor_t* proc);
}

/* ------------------------------------------------------------------------- */
/* Fixture                                                                   */
/* ------------------------------------------------------------------------- */

class CortexCnnThalamicTest : public ::testing::Test {
protected:
    thalamic_router_t* router = nullptr;

    void SetUp() override {
        thalamic_router_config_t cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        cfg.max_queue_size            = 64;
        cfg.max_destinations          = 4;
        cfg.enable_attention_gating   = true;
        cfg.enable_priority_routing   = true;
        cfg.enable_statistics         = true;
        cfg.min_attention_threshold   = 0.0f;
        cfg.enable_learning           = true;
        cfg.enable_second_messengers  = false;
        cfg.num_neurons               = 16;
        cfg.enable_quantum_routing    = false;
        router = thalamic_router_create(&cfg);
        ASSERT_NE(nullptr, router);

        /* Reset tunables to known-on state before every test. */
        cortex_cnn_tune_set_thalamic_enabled(1.0f);
        cortex_cnn_tune_set_thalamic_featuremap_gain_on(1.0f);
        cortex_cnn_tune_set_thalamic_burst_dropout_reduce_on(1.0f);
    }

    void TearDown() override {
        if (router) {
            thalamic_router_destroy(router);
            router = nullptr;
        }
    }
};

/* ------------------------------------------------------------------------- */
/* Attach / detach lifecycle                                                 */
/* ------------------------------------------------------------------------- */

TEST_F(CortexCnnThalamicTest, AttachNullRouterLeavesChannelNull) {
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_AUDIO, 64);
    ASSERT_NE(nullptr, proc);

    nimcp_error_t rc = cortex_cnn_attach_thalamic_router(
        proc, static_cast<struct thalamic_router*>(nullptr));
    EXPECT_EQ(NIMCP_SUCCESS, rc);
    EXPECT_EQ(nullptr, cortex_cnn_test_get_thalamic_channel(proc));

    cortex_cnn_destroy(proc);
}

TEST_F(CortexCnnThalamicTest, AttachValidRouterCreatesChannel) {
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_AUDIO, 64);
    ASSERT_NE(nullptr, proc);

    nimcp_error_t rc = cortex_cnn_attach_thalamic_router(
        proc, reinterpret_cast<struct thalamic_router*>(router));
    EXPECT_EQ(NIMCP_SUCCESS, rc);

    thalamic_channel_t* ch = cortex_cnn_test_get_thalamic_channel(proc);
    ASSERT_NE(nullptr, ch);
    /* source_id must equal the cortex type so router learns per-cortex routes. */
    EXPECT_EQ(static_cast<uint32_t>(CORTEX_CNN_AUDIO), ch->source_id);
    /* Destination 0 is pre-registered. */
    EXPECT_GE(ch->n_destinations, 1u);
    EXPECT_EQ(0u, ch->destinations[0]);

    cortex_cnn_destroy(proc);
}

TEST_F(CortexCnnThalamicTest, AttachAllFourCortexTypesGetDistinctSourceIds) {
    const cortex_cnn_type_t types[4] = {
        CORTEX_CNN_VISUAL, CORTEX_CNN_AUDIO,
        CORTEX_CNN_SPEECH, CORTEX_CNN_SOMATO
    };
    for (int i = 0; i < 4; i++) {
        cortex_cnn_processor_t* proc = cortex_cnn_create(types[i], 64);
        ASSERT_NE(nullptr, proc) << "type=" << types[i];

        ASSERT_EQ(NIMCP_SUCCESS,
                  cortex_cnn_attach_thalamic_router(
                      proc, reinterpret_cast<struct thalamic_router*>(router)));
        thalamic_channel_t* ch = cortex_cnn_test_get_thalamic_channel(proc);
        ASSERT_NE(nullptr, ch) << "type=" << types[i];
        EXPECT_EQ(static_cast<uint32_t>(types[i]), ch->source_id);

        cortex_cnn_destroy(proc);
    }
}

TEST_F(CortexCnnThalamicTest, AttachNullProcReturnsError) {
    nimcp_error_t rc = cortex_cnn_attach_thalamic_router(
        nullptr, reinterpret_cast<struct thalamic_router*>(router));
    EXPECT_EQ(NIMCP_ERROR_NULL_POINTER, rc);
}

TEST_F(CortexCnnThalamicTest, DoubleAttachReplacesChannel) {
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_AUDIO, 64);
    ASSERT_NE(nullptr, proc);

    ASSERT_EQ(NIMCP_SUCCESS,
              cortex_cnn_attach_thalamic_router(
                  proc, reinterpret_cast<struct thalamic_router*>(router)));
    thalamic_channel_t* first = cortex_cnn_test_get_thalamic_channel(proc);
    ASSERT_NE(nullptr, first);

    /* Second attach — we cannot reliably pointer-compare because calloc
     * can reuse a just-freed block. Instead we verify the resulting
     * channel is valid and destination 0 is registered, which implies
     * the old one was torn down and a new one constructed. A leak check
     * under ASan / valgrind catches the double-alloc-without-free case. */
    ASSERT_EQ(NIMCP_SUCCESS,
              cortex_cnn_attach_thalamic_router(
                  proc, reinterpret_cast<struct thalamic_router*>(router)));
    thalamic_channel_t* second = cortex_cnn_test_get_thalamic_channel(proc);
    ASSERT_NE(nullptr, second);
    EXPECT_EQ(static_cast<uint32_t>(CORTEX_CNN_AUDIO), second->source_id);
    EXPECT_GE(second->n_destinations, 1u);
    EXPECT_EQ(0u, second->destinations[0]);

    cortex_cnn_destroy(proc);
}

TEST_F(CortexCnnThalamicTest, AttachThenDetachClearsChannel) {
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_SPEECH, 32);
    ASSERT_NE(nullptr, proc);

    ASSERT_EQ(NIMCP_SUCCESS,
              cortex_cnn_attach_thalamic_router(
                  proc, reinterpret_cast<struct thalamic_router*>(router)));
    ASSERT_NE(nullptr, cortex_cnn_test_get_thalamic_channel(proc));

    ASSERT_EQ(NIMCP_SUCCESS,
              cortex_cnn_attach_thalamic_router(
                  proc, static_cast<struct thalamic_router*>(nullptr)));
    EXPECT_EQ(nullptr, cortex_cnn_test_get_thalamic_channel(proc));

    cortex_cnn_destroy(proc);
}

TEST_F(CortexCnnThalamicTest, DestroyWithAttachedChannelDoesNotLeakRouter) {
    /* Lifecycle: destroying a processor that owns a channel must destroy
     * the channel but NOT the borrowed router. Verified by the fact that
     * TearDown() can still destroy the router cleanly. */
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_AUDIO, 64);
    ASSERT_NE(nullptr, proc);
    ASSERT_EQ(NIMCP_SUCCESS,
              cortex_cnn_attach_thalamic_router(
                  proc, reinterpret_cast<struct thalamic_router*>(router)));
    cortex_cnn_destroy(proc);
    /* router still alive, TearDown will destroy it. */
    SUCCEED();
}

/* ------------------------------------------------------------------------- */
/* Tunable round-trip                                                        */
/* ------------------------------------------------------------------------- */

TEST_F(CortexCnnThalamicTest, TunableEnabledRoundTrip) {
    cortex_cnn_tune_set_thalamic_enabled(0.0f);
    EXPECT_EQ(0.0f, cortex_cnn_tune_get_thalamic_enabled());
    cortex_cnn_tune_set_thalamic_enabled(1.0f);
    EXPECT_EQ(1.0f, cortex_cnn_tune_get_thalamic_enabled());
    /* Any nonzero coerces to 1.0 (SNN / LNN convention). */
    cortex_cnn_tune_set_thalamic_enabled(42.0f);
    EXPECT_EQ(1.0f, cortex_cnn_tune_get_thalamic_enabled());
    cortex_cnn_tune_set_thalamic_enabled(-3.14f);
    EXPECT_EQ(1.0f, cortex_cnn_tune_get_thalamic_enabled());
    cortex_cnn_tune_set_thalamic_enabled(0.0f);
    EXPECT_EQ(0.0f, cortex_cnn_tune_get_thalamic_enabled());
}

TEST_F(CortexCnnThalamicTest, TunableFeaturemapGainOnRoundTrip) {
    cortex_cnn_tune_set_thalamic_featuremap_gain_on(0.0f);
    EXPECT_EQ(0.0f, cortex_cnn_tune_get_thalamic_featuremap_gain_on());
    cortex_cnn_tune_set_thalamic_featuremap_gain_on(99.9f);
    EXPECT_EQ(1.0f, cortex_cnn_tune_get_thalamic_featuremap_gain_on());
    cortex_cnn_tune_set_thalamic_featuremap_gain_on(0.0f);
    EXPECT_EQ(0.0f, cortex_cnn_tune_get_thalamic_featuremap_gain_on());
}

TEST_F(CortexCnnThalamicTest, TunableBurstDropoutReduceOnRoundTrip) {
    cortex_cnn_tune_set_thalamic_burst_dropout_reduce_on(0.0f);
    EXPECT_EQ(0.0f, cortex_cnn_tune_get_thalamic_burst_dropout_reduce_on());
    cortex_cnn_tune_set_thalamic_burst_dropout_reduce_on(2.0f);
    EXPECT_EQ(1.0f, cortex_cnn_tune_get_thalamic_burst_dropout_reduce_on());
    cortex_cnn_tune_set_thalamic_burst_dropout_reduce_on(0.0f);
    EXPECT_EQ(0.0f, cortex_cnn_tune_get_thalamic_burst_dropout_reduce_on());
}

/* ------------------------------------------------------------------------- */
/* Forward behaviour under null attach                                       */
/* ------------------------------------------------------------------------- */

TEST_F(CortexCnnThalamicTest, ForwardWithoutAttachIsSafe) {
    /* A processor with NO thalamic channel attached must still produce
     * a valid embedding — the gate block is a pure no-op. */
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_AUDIO, 32);
    ASSERT_NE(nullptr, proc);
    ASSERT_EQ(nullptr, cortex_cnn_test_get_thalamic_channel(proc));

    /* 128-sample mel row to match default audio shape. */
    float mel[128];
    for (int i = 0; i < 128; i++) mel[i] = 0.25f * std::sin(0.1f * i);

    const float* emb = cortex_cnn_forward_audio(proc, mel, 128);
    ASSERT_NE(nullptr, emb);
    for (uint32_t i = 0; i < 32; i++) {
        EXPECT_TRUE(std::isfinite(emb[i]));
    }

    cortex_cnn_destroy(proc);
}

TEST_F(CortexCnnThalamicTest, ForwardWithEnabledZeroIsSafe) {
    /* Attach a channel, then disable the master toggle. The forward
     * path must not consult the channel at all — no crash, no NaN. */
    cortex_cnn_processor_t* proc = cortex_cnn_create(CORTEX_CNN_AUDIO, 32);
    ASSERT_NE(nullptr, proc);
    ASSERT_EQ(NIMCP_SUCCESS,
              cortex_cnn_attach_thalamic_router(
                  proc, reinterpret_cast<struct thalamic_router*>(router)));

    cortex_cnn_tune_set_thalamic_enabled(0.0f);

    float mel[128];
    for (int i = 0; i < 128; i++) mel[i] = 0.25f * std::sin(0.1f * i);
    const float* emb = cortex_cnn_forward_audio(proc, mel, 128);
    ASSERT_NE(nullptr, emb);
    for (uint32_t i = 0; i < 32; i++) {
        EXPECT_TRUE(std::isfinite(emb[i]));
    }

    cortex_cnn_destroy(proc);
}

/* ------------------------------------------------------------------------- */
/* main                                                                      */
/* ------------------------------------------------------------------------- */

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
