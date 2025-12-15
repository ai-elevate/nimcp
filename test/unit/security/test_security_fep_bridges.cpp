/**
 * @file test_security_fep_bridges.cpp
 * @brief Unit tests for all Security-FEP Bridge modules
 *
 * WHAT: Comprehensive tests for security-FEP bidirectional integrations
 * WHY:  Ensure threat detection and anomaly handling integrate with FEP predictions
 * HOW:  Test lifecycle, effects, and security-specific FEP modulations
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "security/nimcp_anomaly_detector_fep_bridge.h"
#include "security/nimcp_blood_brain_barrier_fep_bridge.h"
#include "security/nimcp_pattern_db_fep_bridge.h"
#include "security/nimcp_rate_limiter_fep_bridge.h"
#include "security/nimcp_security_fep_bridge.h"
#include "cognitive/free_energy/nimcp_free_energy.h"
}

class SecurityFepBridgesTestBase : public ::testing::Test {
protected:
    fep_system_t* fep = nullptr;

    void SetUp() override {
        fep_config_t fep_config;
        fep_default_config(&fep_config);
        fep = fep_create(&fep_config, 8, 4);
        ASSERT_NE(fep, nullptr);
    }

    void TearDown() override {
        if (fep) {
            fep_destroy(fep);
            fep = nullptr;
        }
    }
};

/* ============================================================================
 * Anomaly Detector FEP Bridge Tests
 * ============================================================================ */

class AnomalyDetectorFepBridgeTest : public SecurityFepBridgesTestBase {
protected:
    anomaly_detector_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SecurityFepBridgesTestBase::SetUp();
        anomaly_detector_fep_config_t config;
        anomaly_detector_fep_default_config(&config);
        bridge = anomaly_detector_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            anomaly_detector_fep_destroy(bridge);
            bridge = nullptr;
        }
        SecurityFepBridgesTestBase::TearDown();
    }
};

TEST_F(AnomalyDetectorFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(AnomalyDetectorFepBridgeTest, CreateWithNullConfig) {
    anomaly_detector_fep_bridge_t* br = anomaly_detector_fep_create(nullptr, fep);
    EXPECT_EQ(br, nullptr);
}

TEST_F(AnomalyDetectorFepBridgeTest, CreateWithNullFep) {
    anomaly_detector_fep_config_t config;
    anomaly_detector_fep_default_config(&config);
    anomaly_detector_fep_bridge_t* br = anomaly_detector_fep_create(&config, nullptr);
    EXPECT_EQ(br, nullptr);
}

TEST_F(AnomalyDetectorFepBridgeTest, DestroyNull) {
    anomaly_detector_fep_destroy(nullptr);
}

TEST_F(AnomalyDetectorFepBridgeTest, DefaultConfig) {
    anomaly_detector_fep_config_t config;
    int ret = anomaly_detector_fep_default_config(&config);
    EXPECT_EQ(ret, 0);
    EXPECT_GT(config.anomaly_surprise_threshold, 0.0f);
    EXPECT_TRUE(config.enable_predictive_detection);
}

TEST_F(AnomalyDetectorFepBridgeTest, Update) {
    int ret = anomaly_detector_fep_update(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(AnomalyDetectorFepBridgeTest, GetEffects) {
    anomaly_detector_fep_update(bridge);
    anomaly_detector_fep_effects_t effects;
    int ret = anomaly_detector_fep_get_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);
}

TEST_F(AnomalyDetectorFepBridgeTest, BioAsync) {
    EXPECT_FALSE(anomaly_detector_fep_is_bio_async_connected(bridge));
    anomaly_detector_fep_connect_bio_async(bridge);
    EXPECT_TRUE(anomaly_detector_fep_is_bio_async_connected(bridge));
    anomaly_detector_fep_disconnect_bio_async(bridge);
    EXPECT_FALSE(anomaly_detector_fep_is_bio_async_connected(bridge));
}

/* ============================================================================
 * Blood Brain Barrier FEP Bridge Tests
 * ============================================================================ */

class BloodBrainBarrierFepBridgeTest : public SecurityFepBridgesTestBase {
protected:
    blood_brain_barrier_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SecurityFepBridgesTestBase::SetUp();
        blood_brain_barrier_fep_config_t config;
        blood_brain_barrier_fep_default_config(&config);
        bridge = blood_brain_barrier_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            blood_brain_barrier_fep_destroy(bridge);
            bridge = nullptr;
        }
        SecurityFepBridgesTestBase::TearDown();
    }
};

TEST_F(BloodBrainBarrierFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(BloodBrainBarrierFepBridgeTest, CreateWithNullConfig) {
    blood_brain_barrier_fep_bridge_t* br = blood_brain_barrier_fep_create(nullptr, fep);
    EXPECT_EQ(br, nullptr);
}

TEST_F(BloodBrainBarrierFepBridgeTest, CreateWithNullFep) {
    blood_brain_barrier_fep_config_t config;
    blood_brain_barrier_fep_default_config(&config);
    blood_brain_barrier_fep_bridge_t* br = blood_brain_barrier_fep_create(&config, nullptr);
    EXPECT_EQ(br, nullptr);
}

TEST_F(BloodBrainBarrierFepBridgeTest, DestroyNull) {
    blood_brain_barrier_fep_destroy(nullptr);
}

TEST_F(BloodBrainBarrierFepBridgeTest, DefaultConfig) {
    blood_brain_barrier_fep_config_t config;
    int ret = blood_brain_barrier_fep_default_config(&config);
    EXPECT_EQ(ret, 0);
    EXPECT_GT(config.permeability_precision_coupling, 0.0f);
}

TEST_F(BloodBrainBarrierFepBridgeTest, Update) {
    int ret = blood_brain_barrier_fep_update(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(BloodBrainBarrierFepBridgeTest, GetEffects) {
    blood_brain_barrier_fep_update(bridge);
    blood_brain_barrier_fep_effects_t effects;
    int ret = blood_brain_barrier_fep_get_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);
}

TEST_F(BloodBrainBarrierFepBridgeTest, BioAsync) {
    EXPECT_FALSE(blood_brain_barrier_fep_is_bio_async_connected(bridge));
    blood_brain_barrier_fep_connect_bio_async(bridge);
    EXPECT_TRUE(blood_brain_barrier_fep_is_bio_async_connected(bridge));
}

/* ============================================================================
 * Pattern DB FEP Bridge Tests
 * ============================================================================ */

class PatternDbFepBridgeTest : public SecurityFepBridgesTestBase {
protected:
    pattern_db_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SecurityFepBridgesTestBase::SetUp();
        pattern_db_fep_config_t config;
        pattern_db_fep_default_config(&config);
        bridge = pattern_db_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            pattern_db_fep_destroy(bridge);
            bridge = nullptr;
        }
        SecurityFepBridgesTestBase::TearDown();
    }
};

TEST_F(PatternDbFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(PatternDbFepBridgeTest, CreateWithNullConfig) {
    pattern_db_fep_bridge_t* br = pattern_db_fep_create(nullptr, fep);
    EXPECT_EQ(br, nullptr);
}

TEST_F(PatternDbFepBridgeTest, CreateWithNullFep) {
    pattern_db_fep_config_t config;
    pattern_db_fep_default_config(&config);
    pattern_db_fep_bridge_t* br = pattern_db_fep_create(&config, nullptr);
    EXPECT_EQ(br, nullptr);
}

TEST_F(PatternDbFepBridgeTest, Update) {
    int ret = pattern_db_fep_update(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(PatternDbFepBridgeTest, GetEffects) {
    pattern_db_fep_update(bridge);
    pattern_db_fep_effects_t effects;
    int ret = pattern_db_fep_get_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);
}

/* ============================================================================
 * Rate Limiter FEP Bridge Tests
 * ============================================================================ */

class RateLimiterFepBridgeTest : public SecurityFepBridgesTestBase {
protected:
    rate_limiter_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SecurityFepBridgesTestBase::SetUp();
        rate_limiter_fep_config_t config;
        rate_limiter_fep_default_config(&config);
        bridge = rate_limiter_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            rate_limiter_fep_destroy(bridge);
            bridge = nullptr;
        }
        SecurityFepBridgesTestBase::TearDown();
    }
};

TEST_F(RateLimiterFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(RateLimiterFepBridgeTest, CreateWithNullConfig) {
    rate_limiter_fep_bridge_t* br = rate_limiter_fep_create(nullptr, fep);
    EXPECT_EQ(br, nullptr);
}

TEST_F(RateLimiterFepBridgeTest, CreateWithNullFep) {
    rate_limiter_fep_config_t config;
    rate_limiter_fep_default_config(&config);
    rate_limiter_fep_bridge_t* br = rate_limiter_fep_create(&config, nullptr);
    EXPECT_EQ(br, nullptr);
}

TEST_F(RateLimiterFepBridgeTest, Update) {
    int ret = rate_limiter_fep_update(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(RateLimiterFepBridgeTest, GetEffects) {
    rate_limiter_fep_update(bridge);
    rate_limiter_fep_effects_t effects;
    int ret = rate_limiter_fep_get_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);
}

/* ============================================================================
 * Security FEP Bridge Tests
 * ============================================================================ */

class SecurityFepBridgeTest : public SecurityFepBridgesTestBase {
protected:
    security_fep_bridge_t* bridge = nullptr;

    void SetUp() override {
        SecurityFepBridgesTestBase::SetUp();
        security_fep_config_t config;
        security_fep_default_config(&config);
        bridge = security_fep_create(&config, fep);
    }

    void TearDown() override {
        if (bridge) {
            security_fep_destroy(bridge);
            bridge = nullptr;
        }
        SecurityFepBridgesTestBase::TearDown();
    }
};

TEST_F(SecurityFepBridgeTest, CreateDestroy) {
    ASSERT_NE(bridge, nullptr);
}

TEST_F(SecurityFepBridgeTest, CreateWithNullConfig) {
    security_fep_bridge_t* br = security_fep_create(nullptr, fep);
    EXPECT_EQ(br, nullptr);
}

TEST_F(SecurityFepBridgeTest, CreateWithNullFep) {
    security_fep_config_t config;
    security_fep_default_config(&config);
    security_fep_bridge_t* br = security_fep_create(&config, nullptr);
    EXPECT_EQ(br, nullptr);
}

TEST_F(SecurityFepBridgeTest, Update) {
    int ret = security_fep_update(bridge);
    EXPECT_EQ(ret, 0);
}

TEST_F(SecurityFepBridgeTest, GetEffects) {
    security_fep_update(bridge);
    security_fep_effects_t effects;
    int ret = security_fep_get_effects(bridge, &effects);
    EXPECT_EQ(ret, 0);
}

TEST_F(SecurityFepBridgeTest, GetStats) {
    security_fep_stats_t stats;
    int ret = security_fep_get_stats(bridge, &stats);
    EXPECT_EQ(ret, 0);
}

TEST_F(SecurityFepBridgeTest, BioAsync) {
    EXPECT_FALSE(security_fep_is_bio_async_connected(bridge));
    security_fep_connect_bio_async(bridge);
    EXPECT_TRUE(security_fep_is_bio_async_connected(bridge));
    security_fep_disconnect_bio_async(bridge);
    EXPECT_FALSE(security_fep_is_bio_async_connected(bridge));
}
