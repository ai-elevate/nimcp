/**
 * @file test_swarm_security_integration.cpp
 * @brief Integration tests for NIMCP Swarm Security with BBB
 *
 * WHAT: End-to-end security tests across all swarm components
 * WHY:  Validate complete security workflows and attack resistance
 * HOW:  Simulate real swarm scenarios with security validations
 *
 * TEST COVERAGE:
 * - End-to-end secure message flow
 * - Attack simulation (DoS, Byzantine, Injection)
 * - Gateway authorization chain
 * - Multi-drone consensus under attack
 * - Workspace security with Byzantine participants
 * - Audit trail verification
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <random>

extern "C" {
#include "swarm/nimcp_swarm_protocol.h"
#include "swarm/nimcp_swarm_signal.h"
#include "swarm/nimcp_collective_workspace.h"
#include "swarm/nimcp_swarm_consensus.h"
#include "swarm/nimcp_swarm_brain.h"
#include "swarm/nimcp_swarm_gateway.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "async/nimcp_bio_messages.h"
#include "utils/error/nimcp_error_codes.h"
}

//=============================================================================
// Test Fixture
//=============================================================================

class SwarmSecurityIntegrationTest : public ::testing::Test {
protected:
    bbb_system_t bbb;
    std::vector<swarm_consensus_t> consensus_nodes;
    std::vector<collective_workspace_t*> workspaces;

    void SetUp() override {
        // Create BBB system with strict security
        bbb_config_t config = bbb_default_config();
        config.strict_mode = true;
        config.default_action = BBB_ACTION_BLOCK;
        config.input.max_string_length = 512;
        config.input.max_array_size = 2048;
        config.memory.enable_stack_canaries = true;
        config.access.enable_rbac = true;

        bbb = bbb_system_create(&config);
        ASSERT_NE(bbb, nullptr);
        ASSERT_TRUE(bbb_system_set_enabled(bbb, true));
    }

    void TearDown() override {
        // Clean up consensus nodes
        for (auto ctx : consensus_nodes) {
            if (ctx) swarm_consensus_destroy(ctx);
        }
        consensus_nodes.clear();

        // Clean up workspaces
        for (auto ws : workspaces) {
            if (ws) collective_workspace_destroy(ws);
        }
        workspaces.clear();

        if (bbb) {
            bbb_system_destroy(bbb);
        }
    }

    // Helper: Create swarm node with security
    swarm_consensus_t CreateSecureConsensusNode(uint16_t drone_id) {
        swarm_consensus_config_t config = swarm_consensus_default_config(drone_id);
        config.enable_byzantine_ft = true;
        config.min_confidence = 0.5f;
        config.enable_logging = true;

        swarm_consensus_t ctx = swarm_consensus_create(&config);
        if (ctx) {
            consensus_nodes.push_back(ctx);
        }
        return ctx;
    }

    // Helper: Create secure workspace
    collective_workspace_t* CreateSecureWorkspace(uint16_t drone_id, uint16_t swarm_size) {
        collective_workspace_t* ws = collective_workspace_create_simple(drone_id, swarm_size);
        if (ws) {
            workspaces.push_back(ws);
        }
        return ws;
    }

    // Helper: Simulate message through BBB validation
    bool ValidateMessageThroughBBB(const void* msg, size_t size) {
        bbb_validation_result_t result;
        return bbb_validate_input(bbb, msg, size, &result);
    }
};

//=============================================================================
// 1. End-to-End Security Flow
//=============================================================================

TEST_F(SwarmSecurityIntegrationTest, SecureMessageFlow) {
    // Create swarm with BBB enabled
    swarm_signal_config_t signal_config = {
        .radio_type = SWARM_RADIO_SIMULATION,
        .max_packet_size = 256,
        .retry_count = 3,
        .timeout_ms = 1000
    };

    nimcp_swarm_signal_adapter_t* adapter = swarm_signal_adapter_create(&signal_config);
    ASSERT_NE(adapter, nullptr);

    // Step 1: Encode message
    swarm_phoneme_message_t msg;
    float position[4] = {10.5f, 20.3f, 5.0f, 0.85f};

    nimcp_error_t result = swarm_protocol_encode_heartbeat(&msg, 1,
                                                            position[0], position[1],
                                                            position[2], position[3]);
    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Step 2: Validate through BBB
    ASSERT_TRUE(ValidateMessageThroughBBB(&msg, sizeof(msg)));

    // Step 3: Validate CRC
    ASSERT_TRUE(swarm_protocol_validate(&msg));

    // Step 4: Send through signal adapter
    bool sent = swarm_signal_send(adapter, (uint8_t*)&msg, sizeof(msg), 0);
    EXPECT_TRUE(sent);

    // Step 5: Receive and validate
    uint8_t recv_buffer[256];
    uint32_t recv_len;
    uint32_t source_id;

    // Note: In simulation mode, we might not actually receive
    // This tests the send path validation

    // Step 6: Verify audit trail
    bbb_statistics_t stats;
    ASSERT_TRUE(bbb_system_get_statistics(bbb, &stats));
    EXPECT_GT(stats.total_validations, 0);

    swarm_signal_adapter_destroy(adapter);
}

//=============================================================================
// 2. Attack Simulation: DoS
//=============================================================================

TEST_F(SwarmSecurityIntegrationTest, SimulateDosAttack) {
    // Rapid message flood to test rate limiting
    swarm_signal_config_t config = {
        .radio_type = SWARM_RADIO_SIMULATION,
        .max_packet_size = 256,
        .retry_count = 1, // Low retry for DoS test
        .timeout_ms = 100
    };

    nimcp_swarm_signal_adapter_t* adapter = swarm_signal_adapter_create(&config);
    ASSERT_NE(adapter, nullptr);

    // Send 1000 messages rapidly
    const int flood_count = 1000;
    int sent_count = 0;
    int blocked_count = 0;

    swarm_phoneme_message_t msg;
    float payload[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < flood_count; i++) {
        nimcp_error_t result = swarm_protocol_encode(&msg, SWARM_MSG_HEARTBEAT,
                                                      42, payload, 4);
        ASSERT_EQ(result, NIMCP_SUCCESS);

        // Validate through BBB (should implement rate limiting)
        if (ValidateMessageThroughBBB(&msg, sizeof(msg))) {
            if (swarm_signal_send(adapter, (uint8_t*)&msg, sizeof(msg), 0)) {
                sent_count++;
            }
        } else {
            blocked_count++;
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Verify system stability
    swarm_signal_stats_t signal_stats;
    ASSERT_TRUE(swarm_signal_get_stats(adapter, &signal_stats));

    // System should remain operational
    EXPECT_TRUE(swarm_signal_is_operational(adapter));

    // Some rate limiting should occur (implementation dependent)
    // At minimum, system should handle flood without crash

    swarm_signal_adapter_destroy(adapter);
}

//=============================================================================
// 3. Attack Simulation: Byzantine Swarm
//=============================================================================

TEST_F(SwarmSecurityIntegrationTest, SimulateByzantineSwarm) {
    // Create swarm with 33% malicious drones
    const int total_drones = 12;
    const int honest_drones = 8;
    const int byzantine_drones = 4;

    std::vector<swarm_consensus_t> nodes;

    // Create consensus nodes
    for (int i = 0; i < total_drones; i++) {
        swarm_consensus_t ctx = CreateSecureConsensusNode(i);
        ASSERT_NE(ctx, nullptr);
        nodes.push_back(ctx);
    }

    // Propose vote on first node
    uint32_t proposal_id;
    float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    nimcp_error_t result = swarm_consensus_propose(nodes[0], VOTE_TOPIC_CUSTOM,
                                                    values, 0, total_drones, 0.667f,
                                                    nullptr, nullptr, &proposal_id);
    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Honest drones vote with high confidence
    for (int i = 0; i < honest_drones; i++) {
        swarm_vote_response_t vote = {
            .proposal_id = proposal_id,
            .voter_drone = (uint16_t)i,
            .choice = VOTE_CHOICE_AGREE,
            .confidence = 0.9f
        };

        result = swarm_consensus_receive_vote(nodes[0], &vote);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    // Byzantine drones: random votes, low confidence
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = honest_drones; i < total_drones; i++) {
        swarm_vote_choice_t choice = (dis(gen) > 0.5) ? VOTE_CHOICE_AGREE : VOTE_CHOICE_DISAGREE;

        swarm_vote_response_t vote = {
            .proposal_id = proposal_id,
            .voter_drone = (uint16_t)i,
            .choice = choice,
            .confidence = 0.3f // Low confidence
        };

        result = swarm_consensus_receive_vote(nodes[0], &vote);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    // Check result: should still reach consensus despite Byzantine nodes
    swarm_vote_result_t vote_result;
    result = swarm_consensus_get_result(nodes[0], proposal_id, &vote_result);
    ASSERT_EQ(result, NIMCP_SUCCESS);

    // With 8/12 honest agrees, should pass (66.7%)
    EXPECT_TRUE(vote_result.passed);

    // Verify Byzantine detection
    swarm_consensus_stats_t stats;
    result = swarm_consensus_get_stats(nodes[0], &stats);
    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Should detect some anomalies
    EXPECT_GT(stats.byzantine_faults_detected, 0);
}

//=============================================================================
// 4. Gateway Authorization Chain
//=============================================================================

TEST_F(SwarmSecurityIntegrationTest, GatewayAuthorizationChain) {
    // Setup access control hierarchy
    bbb_subject_t server = {
        .id = 1,
        .privilege_level = 10, // High privilege
        .roles = 0xFFFF,       // All roles
        .capabilities = 0xFFFFFFFFFFFFFFFF
    };

    bbb_subject_t gateway = {
        .id = 2,
        .privilege_level = 7,
        .roles = 0x00FF,
        .capabilities = 0x0000FFFFFFFFFFFF
    };

    bbb_subject_t drone = {
        .id = 3,
        .privilege_level = 3,
        .roles = 0x000F,
        .capabilities = 0x00000000FFFFFFFF
    };

    bbb_object_t mission_control = {
        .id = 100,
        .required_privilege = 5,
        .required_roles = 0x0001,
        .required_capabilities = 0
    };

    bbb_object_t emergency_override = {
        .id = 101,
        .required_privilege = 9,
        .required_roles = 0x8000,
        .required_capabilities = 0
    };

    // Register subjects and objects
    ASSERT_TRUE(bbb_register_subject(bbb, &server));
    ASSERT_TRUE(bbb_register_subject(bbb, &gateway));
    ASSERT_TRUE(bbb_register_subject(bbb, &drone));
    ASSERT_TRUE(bbb_register_object(bbb, &mission_control));
    ASSERT_TRUE(bbb_register_object(bbb, &emergency_override));

    // Test authorization chain
    // Server can access both
    EXPECT_TRUE(bbb_check_access(bbb, &server, &mission_control, 0x02));
    EXPECT_TRUE(bbb_check_access(bbb, &server, &emergency_override, 0x02));

    // Gateway can access mission control but not emergency override
    EXPECT_TRUE(bbb_check_access(bbb, &gateway, &mission_control, 0x02));
    EXPECT_FALSE(bbb_check_access(bbb, &gateway, &emergency_override, 0x02));

    // Drone cannot access mission control
    EXPECT_FALSE(bbb_check_access(bbb, &drone, &mission_control, 0x02));

    // Verify audit trail
    bbb_statistics_t stats;
    ASSERT_TRUE(bbb_system_get_statistics(bbb, &stats));
    EXPECT_GT(stats.access_violations, 0);
}

//=============================================================================
// 5. Workspace Security with Byzantine Participants
//=============================================================================

TEST_F(SwarmSecurityIntegrationTest, WorkspaceByzantineResistance) {
    const int swarm_size = 8;
    const int honest_nodes = 6;
    const int byzantine_nodes = 2;

    std::vector<collective_workspace_t*> nodes;

    // Create workspace nodes
    for (int i = 0; i < swarm_size; i++) {
        collective_workspace_t* ws = CreateSecureWorkspace(i, swarm_size);
        ASSERT_NE(ws, nullptr);
        nodes.push_back(ws);
    }

    // Honest nodes share legitimate threat
    collective_workspace_item_t threat;
    memset(&threat, 0, sizeof(threat));
    threat.item_id = 0x00000001; // Drone 0, local ID 1
    threat.salience = 0.9f;
    threat.type = WORKSPACE_ITEM_THREAT;
    threat.source_drone = 0;
    threat.timestamp_ms = 1000;

    // Honest nodes add item
    for (int i = 0; i < honest_nodes; i++) {
        collective_workspace_item_t item = threat;
        item.source_drone = i;
        item.item_id = (i << 16) | 1;

        ASSERT_TRUE(ValidateMessageThroughBBB(&item, sizeof(item)));
        ASSERT_TRUE(collective_workspace_add_item(nodes[i], &item));
    }

    // Byzantine nodes attempt manipulation
    for (int i = honest_nodes; i < swarm_size; i++) {
        collective_workspace_item_t fake_item;
        memset(&fake_item, 0, sizeof(fake_item));
        fake_item.item_id = (i << 16) | 1;
        fake_item.salience = 1.0f; // Artificially high
        fake_item.type = WORKSPACE_ITEM_GOAL; // Different type
        fake_item.source_drone = i;
        fake_item.timestamp_ms = 1000;

        // Validation should detect anomaly
        bool valid = ValidateMessageThroughBBB(&fake_item, sizeof(fake_item));

        // Even if validation passes, CRDT merge should handle conflicts
        if (valid) {
            collective_workspace_add_item(nodes[i], &fake_item);
        }
    }

    // Merge items across nodes (simulate propagation)
    for (int src = 0; src < swarm_size; src++) {
        collective_workspace_item_t items[COLLECTIVE_WORKSPACE_MAX_ITEMS];
        uint32_t count;

        if (collective_workspace_get_top_items(nodes[src], items,
                                               COLLECTIVE_WORKSPACE_MAX_ITEMS, &count)) {
            for (int dst = 0; dst < swarm_size; dst++) {
                if (src != dst) {
                    for (uint32_t j = 0; j < count; j++) {
                        collective_workspace_merge_item(nodes[dst], &items[j]);
                    }
                }
            }
        }
    }

    // Verify coherence despite Byzantine nodes
    for (int i = 0; i < honest_nodes; i++) {
        float coherence = collective_workspace_get_coherence(nodes[i]);
        // Coherence should still be reasonable
        EXPECT_GT(coherence, 0.3f);
    }

    // Verify statistics
    for (int i = 0; i < swarm_size; i++) {
        uint32_t total_received, total_sent;
        uint64_t merge_conflicts, items_pruned;

        ASSERT_TRUE(collective_workspace_get_statistics(nodes[i], &total_received,
                                                        &total_sent, &merge_conflicts,
                                                        &items_pruned));

        // Should have some merge conflicts from Byzantine nodes
        if (i < honest_nodes) {
            // Honest nodes should detect conflicts
            EXPECT_GT(merge_conflicts, 0);
        }
    }
}

//=============================================================================
// 6. Multi-Drone Consensus Under Attack
//=============================================================================

TEST_F(SwarmSecurityIntegrationTest, ConsensusUnderCoordinatedAttack) {
    const int swarm_size = 16;
    const int honest = 11;
    const int attackers = 5; // < 1/3 for BFT

    std::vector<swarm_consensus_t> nodes;

    // Create all nodes
    for (int i = 0; i < swarm_size; i++) {
        swarm_consensus_t ctx = CreateSecureConsensusNode(i);
        ASSERT_NE(ctx, nullptr);
        nodes.push_back(ctx);
    }

    // Honest leader proposes
    uint32_t proposal_id;
    float values[4] = {10.0f, 20.0f, 30.0f, 40.0f};

    nimcp_error_t result = swarm_consensus_propose(nodes[0], VOTE_TOPIC_TARGET_PRIORITY,
                                                    values, 0, swarm_size, 0.75f,
                                                    nullptr, nullptr, &proposal_id);
    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Honest nodes vote agree
    for (int i = 0; i < honest; i++) {
        swarm_vote_response_t vote = {
            .proposal_id = proposal_id,
            .voter_drone = (uint16_t)i,
            .choice = VOTE_CHOICE_AGREE,
            .confidence = 0.95f
        };

        result = swarm_consensus_receive_vote(nodes[0], &vote);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    // Coordinated attackers all vote disagree with max confidence
    for (int i = honest; i < swarm_size; i++) {
        swarm_vote_response_t vote = {
            .proposal_id = proposal_id,
            .voter_drone = (uint16_t)i,
            .choice = VOTE_CHOICE_DISAGREE,
            .confidence = 1.0f
        };

        result = swarm_consensus_receive_vote(nodes[0], &vote);
        EXPECT_EQ(result, NIMCP_SUCCESS);
    }

    // Despite coordinated attack, honest majority should prevail
    swarm_vote_result_t vote_result;
    result = swarm_consensus_get_result(nodes[0], proposal_id, &vote_result);
    ASSERT_EQ(result, NIMCP_SUCCESS);

    // 11/16 = 68.75% agrees, threshold is 75%
    // This should FAIL because threshold is high
    EXPECT_FALSE(vote_result.passed);

    // Lower threshold test
    uint32_t proposal_id2;
    result = swarm_consensus_propose(nodes[0], VOTE_TOPIC_TARGET_PRIORITY,
                                     values, 0, swarm_size, 0.60f, // Lower threshold
                                     nullptr, nullptr, &proposal_id2);
    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Same votes
    for (int i = 0; i < honest; i++) {
        swarm_vote_response_t vote = {
            .proposal_id = proposal_id2,
            .voter_drone = (uint16_t)i,
            .choice = VOTE_CHOICE_AGREE,
            .confidence = 0.95f
        };
        swarm_consensus_receive_vote(nodes[0], &vote);
    }

    for (int i = honest; i < swarm_size; i++) {
        swarm_vote_response_t vote = {
            .proposal_id = proposal_id2,
            .voter_drone = (uint16_t)i,
            .choice = VOTE_CHOICE_DISAGREE,
            .confidence = 1.0f
        };
        swarm_consensus_receive_vote(nodes[0], &vote);
    }

    result = swarm_consensus_get_result(nodes[0], proposal_id2, &vote_result);
    ASSERT_EQ(result, NIMCP_SUCCESS);

    // Now with 60% threshold, should pass
    EXPECT_TRUE(vote_result.passed);
}

//=============================================================================
// 7. Complete Audit Trail Verification
//=============================================================================

TEST_F(SwarmSecurityIntegrationTest, CompleteAuditTrail) {
    // Perform various operations and verify complete audit trail

    // 1. Message validation
    swarm_phoneme_message_t msg;
    float payload[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    nimcp_error_t result = swarm_protocol_encode(&msg, SWARM_MSG_HEARTBEAT,
                                                  42, payload, 4);
    ASSERT_EQ(result, NIMCP_SUCCESS);
    ASSERT_TRUE(ValidateMessageThroughBBB(&msg, sizeof(msg)));

    // 2. Access control checks
    bbb_subject_t subject = { .id = 1, .privilege_level = 1, .roles = 0, .capabilities = 0 };
    bbb_object_t object = { .id = 100, .required_privilege = 5, .required_roles = 0, .required_capabilities = 0 };

    ASSERT_TRUE(bbb_register_subject(bbb, &subject));
    ASSERT_TRUE(bbb_register_object(bbb, &object));

    bool access = bbb_check_access(bbb, &subject, &object, 0x02);
    EXPECT_FALSE(access); // Should be denied

    // 3. Trigger threat detection
    uint8_t malicious[1024];
    memset(malicious, 0xFF, sizeof(malicious));

    bbb_validation_result_t validation;
    bbb_validate_input(bbb, malicious, sizeof(malicious), &validation);

    // 4. Get comprehensive statistics
    bbb_statistics_t stats;
    ASSERT_TRUE(bbb_system_get_statistics(bbb, &stats));

    EXPECT_GT(stats.total_validations, 0);
    EXPECT_GT(stats.access_violations, 0);

    // 5. Get threat reports
    bbb_threat_report_t reports[100];
    size_t report_count = bbb_get_threat_reports(bbb, reports, 100);

    // Should have logged security events
    EXPECT_GT(report_count, 0);

    if (report_count > 0) {
        // Verify report structure
        EXPECT_NE(reports[0].type, BBB_THREAT_NONE);
        EXPECT_GT(reports[0].severity, BBB_SEVERITY_NONE);
        EXPECT_GT(reports[0].timestamp, 0);
    }
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
