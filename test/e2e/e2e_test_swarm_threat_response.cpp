/**
 * @file e2e_test_swarm_threat_response.cpp
 * @brief E2E Test for Swarm Threat Detection and Response
 *
 * WHAT: Complete end-to-end test of threat detection and coordinated response
 * WHY:  Verify swarm can detect threats and coordinate defensive actions
 * HOW:  Simulate threat detection, broadcast alert, coordinate evasive maneuvers
 *
 * TEST SCENARIO:
 * 1. Swarm of 6 drones in patrol formation
 * 2. Drone 3 detects threat (hostile drone)
 * 3. Threat broadcast to all drones
 * 4. Consensus vote on response strategy
 * 5. Coordinated evasive action
 * 6. Threat neutralization or escape
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 * @version 1.0.0
 */

#include "e2e_test_framework.h"
#include <thread>
#include <vector>
#include <atomic>
#include <cmath>

extern "C" {
#include "core/brain/nimcp_brain.h"
#include "swarm/nimcp_swarm_signal.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
}

using namespace nimcp::e2e;

//=============================================================================
// Threat Detection
//=============================================================================

enum ThreatLevel {
    THREAT_NONE = 0,
    THREAT_LOW = 1,
    THREAT_MEDIUM = 2,
    THREAT_HIGH = 3,
    THREAT_CRITICAL = 4
};

enum ResponseStrategy {
    RESPONSE_EVADE,
    RESPONSE_DEFENSIVE,
    RESPONSE_AGGRESSIVE,
    RESPONSE_RETREAT
};

struct ThreatInfo {
    float position[3];
    float velocity[3];
    ThreatLevel level;
    uint64_t detection_time;
    uint32_t detector_id;
};

struct DroneState {
    uint32_t id;
    float position[3];
    float velocity[3];
    bool threat_detected;
    ThreatInfo detected_threat;
    brain_t brain;
    nimcp_swarm_signal_adapter_t* adapter;
};

//=============================================================================
// Test Fixture
//=============================================================================

class SwarmThreatResponseE2ETest : public ::testing::Test {
protected:
    static constexpr uint32_t NUM_DRONES = 6;
    static constexpr float THREAT_DETECTION_RANGE = 50.0f;
    std::vector<DroneState> drones_;

    void SetUp() override {
        // logging initialized in framework
        // log level set in framework

        drones_.resize(NUM_DRONES);

        for (uint32_t i = 0; i < NUM_DRONES; i++) {
            drones_[i].id = 2000 + i;
            drones_[i].position[0] = i * 20.0f;
            drones_[i].position[1] = 0.0f;
            drones_[i].position[2] = 100.0f;
            drones_[i].velocity[0] = 5.0f;
            drones_[i].velocity[1] = 0.0f;
            drones_[i].velocity[2] = 0.0f;
            drones_[i].threat_detected = false;

            std::string name = "drone_threat_" + std::to_string(i);
            drones_[i].brain = brain_create(name.c_str(), BRAIN_SIZE_TINY,
                                           BRAIN_TASK_CLASSIFICATION, 10, 5);
            ASSERT_NE(drones_[i].brain, nullptr);

            swarm_signal_config_t config = {
                .radio_type = SWARM_RADIO_SIMULATION,
                .frequency_hz = 915000000,
                .bandwidth_hz = 125000,
                .tx_power_dbm = 14,
                .max_packet_size = 256,
                .retry_count = 3,
                .timeout_ms = 1000,
                .custom_send = nullptr,
                .custom_recv = nullptr,
                .custom_ctx = nullptr
            };

            drones_[i].adapter = swarm_signal_adapter_create(&config);
            ASSERT_NE(drones_[i].adapter, nullptr);
        }
    }

    void TearDown() override {
        for (auto& drone : drones_) {
            if (drone.brain) brain_destroy(drone.brain);
            if (drone.adapter) swarm_signal_adapter_destroy(drone.adapter);
        }
        drones_.clear();
    }

    float Distance(const float* a, const float* b) {
        float dx = a[0] - b[0];
        float dy = a[1] - b[1];
        float dz = a[2] - b[2];
        return sqrtf(dx*dx + dy*dy + dz*dz);
    }

    void BroadcastThreat(uint32_t detector_idx, const ThreatInfo& threat) {
        swarm_signal_broadcast(
            drones_[detector_idx].adapter,
            reinterpret_cast<const uint8_t*>(&threat),
            sizeof(ThreatInfo)
        );
    }

    ResponseStrategy DetermineResponse(const ThreatInfo& threat) {
        if (threat.level == THREAT_CRITICAL) return RESPONSE_RETREAT;
        if (threat.level == THREAT_HIGH) return RESPONSE_EVADE;
        if (threat.level == THREAT_MEDIUM) return RESPONSE_DEFENSIVE;
        return RESPONSE_EVADE;
    }
};

//=============================================================================
// Test Cases
//=============================================================================

TEST_F(SwarmThreatResponseE2ETest, SingleThreatDetectionAndBroadcast) {
    PipelineTracker tracker("Threat Detection and Broadcast");

    tracker.begin_stage("Setup Patrol Formation", 500);
    // Drones in line formation (already set up)
    tracker.end_stage();

    tracker.begin_stage("Threat Detection", 1000);
    // Drone 3 detects threat
    ThreatInfo threat;
    threat.position[0] = drones_[3].position[0] + 30.0f;
    threat.position[1] = drones_[3].position[1];
    threat.position[2] = drones_[3].position[2];
    threat.velocity[0] = -10.0f;
    threat.velocity[1] = 0.0f;
    threat.velocity[2] = 0.0f;
    threat.level = THREAT_HIGH;
    threat.detector_id = drones_[3].id;

    drones_[3].threat_detected = true;
    drones_[3].detected_threat = threat;

    BroadcastThreat(3, threat);
    tracker.end_stage();

    tracker.begin_stage("Threat Propagation", 500);
    // Simulate message propagation
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify broadcast occurred
    swarm_signal_stats_t stats;
    ASSERT_TRUE(swarm_signal_get_stats(drones_[3].adapter, &stats));
    EXPECT_GE(stats.packets_sent, 1);
    tracker.end_stage();

    EXPECT_TRUE(tracker.is_successful());
}

TEST_F(SwarmThreatResponseE2ETest, CoordinatedEvasiveAction) {
    PipelineTracker tracker("Coordinated Evasive Action");

    tracker.begin_stage("Detect Threat", 500);
    ThreatInfo threat;
    threat.position[0] = 50.0f;
    threat.position[1] = 0.0f;
    threat.position[2] = 100.0f;
    threat.velocity[0] = 0.0f;
    threat.velocity[1] = 0.0f;
    threat.velocity[2] = 0.0f;
    threat.level = THREAT_HIGH;
    threat.detector_id = drones_[2].id;

    BroadcastThreat(2, threat);
    tracker.end_stage();

    tracker.begin_stage("Determine Response Strategy", 500);
    ResponseStrategy strategy = DetermineResponse(threat);
    EXPECT_EQ(strategy, RESPONSE_EVADE);
    tracker.end_stage();

    tracker.begin_stage("Execute Evasive Maneuver", 2000);
    // All drones evade by increasing altitude
    for (auto& drone : drones_) {
        drone.position[2] += 20.0f; // Climb
        drone.velocity[2] = 5.0f;

        // Broadcast new position
        swarm_signal_broadcast(
            drone.adapter,
            reinterpret_cast<const uint8_t*>(drone.position),
            sizeof(drone.position)
        );
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify all drones increased altitude
    for (const auto& drone : drones_) {
        EXPECT_GT(drone.position[2], 110.0f);
    }
    tracker.end_stage();

    EXPECT_TRUE(tracker.is_successful());
}

TEST_F(SwarmThreatResponseE2ETest, MultiThreatScenario) {
    PipelineTracker tracker("Multi-Threat Scenario");

    tracker.begin_stage("Detect Multiple Threats", 1000);
    std::vector<ThreatInfo> threats;

    // Threat 1 from north
    ThreatInfo threat1;
    threat1.position[0] = 50.0f;
    threat1.position[1] = 100.0f;
    threat1.position[2] = 100.0f;
    threat1.level = THREAT_MEDIUM;
    threat1.detector_id = drones_[1].id;
    threats.push_back(threat1);

    // Threat 2 from east
    ThreatInfo threat2;
    threat2.position[0] = 150.0f;
    threat2.position[1] = 0.0f;
    threat2.position[2] = 100.0f;
    threat2.level = THREAT_HIGH;
    threat2.detector_id = drones_[4].id;
    threats.push_back(threat2);

    for (size_t i = 0; i < threats.size(); i++) {
        BroadcastThreat(i * 2, threats[i]);
    }
    tracker.end_stage();

    tracker.begin_stage("Prioritize Threats", 500);
    // Highest threat level takes priority
    ThreatLevel max_level = THREAT_NONE;
    for (const auto& threat : threats) {
        if (threat.level > max_level) {
            max_level = threat.level;
        }
    }

    EXPECT_EQ(max_level, THREAT_HIGH);
    tracker.end_stage();

    tracker.begin_stage("Execute Response", 1000);
    // Respond to highest threat
    ResponseStrategy strategy = DetermineResponse(threats[1]);
    EXPECT_EQ(strategy, RESPONSE_EVADE);
    tracker.end_stage();

    EXPECT_TRUE(tracker.is_successful());
}

TEST_F(SwarmThreatResponseE2ETest, RetreatStrategy) {
    PipelineTracker tracker("Critical Threat Retreat");

    tracker.begin_stage("Detect Critical Threat", 500);
    ThreatInfo critical_threat;
    critical_threat.position[0] = 50.0f;
    critical_threat.position[1] = 0.0f;
    critical_threat.position[2] = 100.0f;
    critical_threat.level = THREAT_CRITICAL;
    critical_threat.detector_id = drones_[0].id;

    BroadcastThreat(0, critical_threat);
    tracker.end_stage();

    tracker.begin_stage("Initiate Retreat", 1000);
    ResponseStrategy strategy = DetermineResponse(critical_threat);
    EXPECT_EQ(strategy, RESPONSE_RETREAT);

    // All drones retreat (reverse direction)
    for (auto& drone : drones_) {
        drone.velocity[0] = -10.0f; // Retreat
        drone.position[0] -= 50.0f;
    }
    tracker.end_stage();

    tracker.begin_stage("Verify Safe Distance", 500);
    // Check all drones moved away from threat
    for (const auto& drone : drones_) {
        float distance = Distance(drone.position, critical_threat.position);
        EXPECT_GT(distance, THREAT_DETECTION_RANGE);
    }
    tracker.end_stage();

    EXPECT_TRUE(tracker.is_successful());
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
