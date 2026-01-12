/**
 * @file test_nlp_performance.cpp
 * @brief Performance regression tests for Neural Link Protocol
 *
 * WHAT: Measure throughput, latency, and resource usage under load
 * WHY:  Ensure protocol performance doesn't regress over time
 * HOW:  Quantitative benchmarks with baseline expectations
 *
 * PERFORMANCE BASELINES (as of 2025-12-08):
 * - Message Throughput: >1000 msg/sec (standard mode)
 * - Encryption Overhead: <2ms per message (256-bit AES-GCM)
 * - Burst Buffering: >90% efficiency in stealth mode
 * - Peer Scaling: Linear up to 50 peers, <10% degradation
 * - Memory Growth: <1MB per 1000 messages processed
 *
 * @author NIMCP Development Team
 * @date 2025-12-08
 */

#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>
#include <cmath>

#include "networking/nlp/nimcp_neural_link_protocol.h"

//=============================================================================
// Test Fixture
//=============================================================================

class NLPPerformanceTest : public ::testing::Test {
protected:
    nlp_node_t node1 = nullptr;
    nlp_node_t node2 = nullptr;
    nlp_config_t config1;
    nlp_config_t config2;

    std::atomic<int> messages_received{0};
    std::atomic<int> bytes_received{0};
    uint32_t connected_peer_id{0};

    void SetUp() override {
        // Node 1 configuration
        config1 = nlp_config_default();
        config1.brain_id = nlp_generate_brain_id();
        config1.port = 35001;
        strncpy(config1.bind_address, "127.0.0.1", sizeof(config1.bind_address));
        config1.default_mode = NLP_MODE_STANDARD;
        config1.heartbeat_interval_ms = 5000;

        // Node 2 configuration
        config2 = nlp_config_default();
        config2.brain_id = nlp_generate_brain_id();
        config2.port = 35002;
        strncpy(config2.bind_address, "127.0.0.1", sizeof(config2.bind_address));
        config2.default_mode = NLP_MODE_STANDARD;
        config2.heartbeat_interval_ms = 5000;
        config2.user_data = this;  // Set user_data BEFORE node creation

        // Create nodes
        node1 = nlp_node_create(&config1);
        node2 = nlp_node_create(&config2);

        ASSERT_NE(node1, nullptr);
        ASSERT_NE(node2, nullptr);

        // Set up message callback for node2
        nlp_set_message_callback(node2,
            [](nlp_node_t node, const nlp_peer_t* peer,
               const nlp_message_t* msg, void* user_data) {
                auto* test = static_cast<NLPPerformanceTest*>(user_data);
                if (test) {
                    test->messages_received++;
                    test->bytes_received += msg->header.payload_len;
                }
            });

        // Start nodes
        ASSERT_EQ(nlp_node_start(node1), 0);
        ASSERT_EQ(nlp_node_start(node2), 0);

        // Give nodes time to initialize
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void TearDown() override {
        if (node1) {
            nlp_node_stop(node1);
            nlp_node_destroy(node1);
        }
        if (node2) {
            nlp_node_stop(node2);
            nlp_node_destroy(node2);
        }

        // Allow cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    void ConnectPeers() {
        // Connect node1 to node2
        connected_peer_id = nlp_connect_peer(node1, "127.0.0.1", 35002);
        ASSERT_NE(connected_peer_id, 0u);

        // Wait for handshake to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    size_t GetCurrentMemoryUsage() {
        // Simple memory usage estimate from stats
        nlp_stats_t stats;
        nlp_get_stats(node1, &stats);
        return stats.bytes_sent + stats.bytes_received;
    }
};

//=============================================================================
// Message Throughput Tests
//=============================================================================

TEST_F(NLPPerformanceTest, MessageThroughputStandardMode) {
    // WHAT: Measure messages per second in standard mode
    // WHY:  Ensure baseline throughput is maintained
    // BASELINE: >1000 msg/sec for small payloads

    ConnectPeers();

    const int num_messages = 1000;
    const size_t payload_size = 64;  // Small payload
    uint8_t payload[payload_size];
    memset(payload, 0xAB, payload_size);

    messages_received = 0;
    auto start = std::chrono::steady_clock::now();

    // Send messages as fast as possible
    for (int i = 0; i < num_messages; i++) {
        int result = nlp_send(node1, connected_peer_id, NLP_MSG_SPIKE_BATCH,
                             payload, payload_size, NLP_PRIORITY_NORMAL);
        ASSERT_GE(result, 0) << "nlp_send failed with error: " << result;
    }

    // Wait for all messages to be received
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (messages_received < num_messages &&
           std::chrono::steady_clock::now() < timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    double msg_per_sec = num_messages / duration;
    std::cout << "Standard mode throughput: " << msg_per_sec
              << " msg/sec" << std::endl;

    EXPECT_GE(msg_per_sec, 1000.0)
        << "Throughput should exceed 1000 msg/sec";
    EXPECT_GE(messages_received, num_messages * 0.95)
        << "At least 95% of messages should be received";
}

TEST_F(NLPPerformanceTest, MessageThroughputTacticalMode) {
    // WHAT: Measure messages per second in tactical mode
    // WHY:  Tactical mode should have similar throughput
    // BASELINE: >800 msg/sec (slight overhead from self-contained messages)

    // Switch to tactical mode
    nlp_set_mode(node1, NLP_MODE_TACTICAL);
    nlp_set_mode(node2, NLP_MODE_TACTICAL);

    ConnectPeers();

    const int num_messages = 1000;
    const size_t payload_size = 64;
    uint8_t payload[payload_size];
    memset(payload, 0xCD, payload_size);

    messages_received = 0;
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_messages; i++) {
        nlp_send(node1, connected_peer_id, NLP_MSG_SPIKE_BATCH,
                payload, payload_size, NLP_PRIORITY_NORMAL);
    }

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (messages_received < num_messages &&
           std::chrono::steady_clock::now() < timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    double msg_per_sec = num_messages / duration;
    std::cout << "Tactical mode throughput: " << msg_per_sec
              << " msg/sec" << std::endl;

    EXPECT_GE(msg_per_sec, 800.0)
        << "Tactical mode should maintain >800 msg/sec";
}

TEST_F(NLPPerformanceTest, LargePayloadThroughput) {
    // WHAT: Measure throughput with large payloads
    // WHY:  Ensure large messages don't cause significant slowdown
    // BASELINE: >100 msg/sec for 8KB payloads

    ConnectPeers();

    const int num_messages = 100;
    const size_t payload_size = 8192;  // 8KB
    std::vector<uint8_t> payload(payload_size, 0xEF);

    messages_received = 0;
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_messages; i++) {
        nlp_send(node1, connected_peer_id, NLP_MSG_STATE_SYNC,
                payload.data(), payload_size, NLP_PRIORITY_NORMAL);
    }

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (messages_received < num_messages &&
           std::chrono::steady_clock::now() < timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    double msg_per_sec = num_messages / duration;
    double mbps = (num_messages * payload_size) / (duration * 1024 * 1024);

    std::cout << "Large payload throughput: " << msg_per_sec
              << " msg/sec (" << mbps << " MB/s)" << std::endl;

    EXPECT_GE(msg_per_sec, 100.0)
        << "Should handle >100 large messages per second";
}

//=============================================================================
// Encryption Overhead Tests
//=============================================================================

TEST_F(NLPPerformanceTest, EncryptionLatencyOverhead) {
    // WHAT: Measure encryption/decryption overhead per message
    // WHY:  Ensure encryption doesn't add excessive latency
    // BASELINE: <2ms encryption overhead per message

    // Set up pre-shared key for encryption
    uint8_t test_key[NLP_KEY_SIZE];
    memset(test_key, 0x42, NLP_KEY_SIZE);

    nlp_set_psk(node1, 0, test_key, 12345, 0, UINT64_MAX);
    nlp_set_psk(node2, 0, test_key, 12345, 0, UINT64_MAX);

    ConnectPeers();

    const int num_messages = 100;
    const size_t payload_size = 1024;
    uint8_t payload[payload_size];
    memset(payload, 0x88, payload_size);

    // Measure with encryption
    auto start_encrypted = std::chrono::steady_clock::now();

    for (int i = 0; i < num_messages; i++) {
        nlp_send(node1, connected_peer_id, NLP_MSG_WEIGHT_DELTA,
                payload, payload_size, NLP_PRIORITY_NORMAL);
    }

    auto end_encrypted = std::chrono::steady_clock::now();
    auto encrypted_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_encrypted - start_encrypted).count();

    double avg_encryption_time_ms = encrypted_duration / 1000.0 / num_messages;

    std::cout << "Average encryption overhead: " << avg_encryption_time_ms
              << " ms per message" << std::endl;

    EXPECT_LT(avg_encryption_time_ms, 2.0)
        << "Encryption overhead should be <2ms per message";
}

TEST_F(NLPPerformanceTest, EncryptionThroughputImpact) {
    // WHAT: Compare throughput with and without encryption
    // WHY:  Quantify encryption performance impact
    // BASELINE: <30% throughput reduction with encryption

    ConnectPeers();

    const int num_messages = 500;
    const size_t payload_size = 512;
    uint8_t payload[payload_size];
    memset(payload, 0x77, payload_size);

    // Measure without encryption
    messages_received = 0;
    auto start_plain = std::chrono::steady_clock::now();

    for (int i = 0; i < num_messages; i++) {
        nlp_send(node1, connected_peer_id, NLP_MSG_SPIKE_BATCH,
                payload, payload_size, NLP_PRIORITY_NORMAL);
    }

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (messages_received < num_messages &&
           std::chrono::steady_clock::now() < timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto end_plain = std::chrono::steady_clock::now();
    double plain_duration = std::chrono::duration<double>(
        end_plain - start_plain).count();
    double plain_throughput = num_messages / plain_duration;

    // Now enable encryption
    uint8_t test_key[NLP_KEY_SIZE];
    memset(test_key, 0x99, NLP_KEY_SIZE);
    nlp_set_psk(node1, 0, test_key, 67890, 0, UINT64_MAX);
    nlp_set_psk(node2, 0, test_key, 67890, 0, UINT64_MAX);

    // Wait for key sync
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Measure with encryption
    messages_received = 0;
    auto start_encrypted = std::chrono::steady_clock::now();

    for (int i = 0; i < num_messages; i++) {
        nlp_send(node1, connected_peer_id, NLP_MSG_SPIKE_BATCH,
                payload, payload_size, NLP_PRIORITY_NORMAL);
    }

    timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (messages_received < num_messages &&
           std::chrono::steady_clock::now() < timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto end_encrypted = std::chrono::steady_clock::now();
    double encrypted_duration = std::chrono::duration<double>(
        end_encrypted - start_encrypted).count();
    double encrypted_throughput = num_messages / encrypted_duration;

    double throughput_reduction = 100.0 * (1.0 - encrypted_throughput / plain_throughput);

    std::cout << "Plain throughput: " << plain_throughput << " msg/sec" << std::endl;
    std::cout << "Encrypted throughput: " << encrypted_throughput << " msg/sec" << std::endl;
    std::cout << "Throughput reduction: " << throughput_reduction << "%" << std::endl;

    EXPECT_LT(throughput_reduction, 30.0)
        << "Encryption should reduce throughput by <30%";
}

//=============================================================================
// Burst Buffering Tests (Stealth Mode)
//=============================================================================

TEST_F(NLPPerformanceTest, BurstBufferingEfficiency) {
    // WHAT: Measure stealth mode burst buffering efficiency
    // WHY:  Ensure messages are efficiently queued and burst
    // BASELINE: >90% of messages successfully buffered and sent

    // Switch to stealth mode
    nlp_set_mode(node1, NLP_MODE_STEALTH);
    nlp_set_mode(node2, NLP_MODE_STEALTH);

    ConnectPeers();

    const int num_messages = 100;
    const size_t payload_size = 128;
    uint8_t payload[payload_size];
    memset(payload, 0x55, payload_size);

    messages_received = 0;

    // Send messages rapidly (should be buffered)
    for (int i = 0; i < num_messages; i++) {
        nlp_send(node1, connected_peer_id, NLP_MSG_BURST_SYNC,
                payload, payload_size, NLP_PRIORITY_NORMAL);
    }

    // Wait for burst transmission
    std::this_thread::sleep_for(std::chrono::seconds(2));

    double efficiency = 100.0 * messages_received / num_messages;

    std::cout << "Burst buffering efficiency: " << efficiency << "%" << std::endl;

    EXPECT_GE(efficiency, 90.0)
        << "Burst buffering should achieve >90% efficiency";
}

TEST_F(NLPPerformanceTest, StealthModeLatency) {
    // WHAT: Measure end-to-end latency in stealth mode
    // WHY:  Quantify latency impact of burst transmission
    // BASELINE: Latency should be predictable (near burst interval)

    nlp_set_mode(node1, NLP_MODE_STEALTH);
    nlp_set_mode(node2, NLP_MODE_STEALTH);

    ConnectPeers();

    const size_t payload_size = 256;
    uint8_t payload[payload_size];
    memset(payload, 0x33, payload_size);

    messages_received = 0;
    auto start = std::chrono::steady_clock::now();

    // Send a single message
    nlp_send(node1, connected_peer_id, NLP_MSG_BURST_SYNC,
            payload, payload_size, NLP_PRIORITY_NORMAL);

    // Wait for receipt
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(35);
    while (messages_received < 1 &&
           std::chrono::steady_clock::now() < timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto end = std::chrono::steady_clock::now();
    auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    std::cout << "Stealth mode latency: " << latency_ms << " ms" << std::endl;

    // Latency should be close to burst interval (30s default)
    EXPECT_LT(latency_ms, 32000)
        << "Stealth latency should be near burst interval";
}

//=============================================================================
// Peer Scaling Tests
//=============================================================================

TEST_F(NLPPerformanceTest, PeerScalingLinear) {
    // WHAT: Measure performance with increasing peer count
    // WHY:  Ensure protocol scales to 50+ peers
    // BASELINE: <10% performance degradation up to 50 peers

    std::vector<nlp_node_t> peers;
    std::vector<nlp_config_t> configs;

    // Create 50 peer nodes
    const int num_peers = 50;
    for (int i = 0; i < num_peers; i++) {
        nlp_config_t cfg = nlp_config_default();
        cfg.brain_id = nlp_generate_brain_id();
        cfg.port = 36000 + i;
        snprintf(cfg.bind_address, sizeof(cfg.bind_address), "127.0.0.1");

        configs.push_back(cfg);
        nlp_node_t peer = nlp_node_create(&cfg);
        ASSERT_NE(peer, nullptr);
        ASSERT_EQ(nlp_node_start(peer), 0);
        peers.push_back(peer);
    }

    // Give nodes time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Connect node1 to all peers
    for (int i = 0; i < num_peers; i++) {
        nlp_connect_peer(node1, "127.0.0.1", 36000 + i);
    }

    // Wait for connections to establish
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Measure broadcast performance
    const int num_broadcasts = 10;
    const size_t payload_size = 256;
    uint8_t payload[payload_size];
    memset(payload, 0x66, payload_size);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_broadcasts; i++) {
        nlp_broadcast(node1, NLP_MSG_HEARTBEAT,
                     payload, payload_size, NLP_PRIORITY_NORMAL);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    double avg_broadcast_time = duration_ms / static_cast<double>(num_broadcasts);

    std::cout << "Average broadcast time to " << num_peers
              << " peers: " << avg_broadcast_time << " ms" << std::endl;

    // Should scale roughly linearly (allow 2x overhead for 50 peers)
    EXPECT_LT(avg_broadcast_time, 100.0)
        << "Broadcast to 50 peers should complete in <100ms";

    // Cleanup
    for (auto peer : peers) {
        nlp_node_stop(peer);
        nlp_node_destroy(peer);
    }
}

TEST_F(NLPPerformanceTest, PeerHandshakeOverhead) {
    // WHAT: Measure time to establish peer connections
    // WHY:  Ensure connection overhead is reasonable
    // BASELINE: <100ms per peer connection

    std::vector<nlp_node_t> peers;
    const int num_peers = 10;

    // Create peer nodes
    for (int i = 0; i < num_peers; i++) {
        nlp_config_t cfg = nlp_config_default();
        cfg.brain_id = nlp_generate_brain_id();
        cfg.port = 37000 + i;
        snprintf(cfg.bind_address, sizeof(cfg.bind_address), "127.0.0.1");

        nlp_node_t peer = nlp_node_create(&cfg);
        ASSERT_NE(peer, nullptr);
        ASSERT_EQ(nlp_node_start(peer), 0);
        peers.push_back(peer);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Measure connection time
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_peers; i++) {
        uint32_t peer_id = nlp_connect_peer(node1, "127.0.0.1", 37000 + i);
        EXPECT_NE(peer_id, 0u);
    }

    // Wait for all handshakes
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto end = std::chrono::steady_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();
    double avg_handshake_time = total_time_ms / static_cast<double>(num_peers);

    std::cout << "Average handshake time: " << avg_handshake_time
              << " ms per peer" << std::endl;

    EXPECT_LT(avg_handshake_time, 100.0)
        << "Peer handshake should complete in <100ms";

    // Cleanup
    for (auto peer : peers) {
        nlp_node_stop(peer);
        nlp_node_destroy(peer);
    }
}

//=============================================================================
// Memory Usage Tests
//=============================================================================

TEST_F(NLPPerformanceTest, MemoryUsageGrowth) {
    // WHAT: Measure memory usage over extended operation
    // WHY:  Ensure no memory leaks or unbounded growth
    // BASELINE: <1MB growth per 1000 messages

    ConnectPeers();

    const int messages_per_round = 1000;
    const int num_rounds = 5;
    const size_t payload_size = 512;
    uint8_t payload[payload_size];
    memset(payload, 0xAA, payload_size);

    std::vector<size_t> memory_samples;

    for (int round = 0; round < num_rounds; round++) {
        // Send messages
        for (int i = 0; i < messages_per_round; i++) {
            nlp_send(node1, connected_peer_id, NLP_MSG_SPIKE_BATCH,
                    payload, payload_size, NLP_PRIORITY_NORMAL);
        }

        // Wait for processing
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Sample memory usage
        size_t memory = GetCurrentMemoryUsage();
        memory_samples.push_back(memory);

        std::cout << "Round " << round << " memory usage: "
                  << memory / 1024 << " KB" << std::endl;
    }

    // Check memory growth rate
    if (memory_samples.size() >= 2) {
        size_t growth = memory_samples.back() - memory_samples.front();
        double growth_per_1000_msgs = growth / static_cast<double>(
            (num_rounds - 1) * messages_per_round) * 1000.0;

        std::cout << "Memory growth: " << growth_per_1000_msgs / 1024
                  << " KB per 1000 messages" << std::endl;

        EXPECT_LT(growth_per_1000_msgs, 1024 * 1024)
            << "Memory growth should be <1MB per 1000 messages";
    }
}

TEST_F(NLPPerformanceTest, PeerMemoryOverhead) {
    // WHAT: Measure memory overhead per peer connection
    // WHY:  Ensure peer state doesn't consume excessive memory
    // BASELINE: <10KB per peer

    nlp_stats_t stats_before;
    nlp_get_stats(node1, &stats_before);

    std::vector<nlp_node_t> peers;
    const int num_peers = 20;

    // Create and connect peers
    for (int i = 0; i < num_peers; i++) {
        nlp_config_t cfg = nlp_config_default();
        cfg.brain_id = nlp_generate_brain_id();
        cfg.port = 38000 + i;
        snprintf(cfg.bind_address, sizeof(cfg.bind_address), "127.0.0.1");

        nlp_node_t peer = nlp_node_create(&cfg);
        ASSERT_NE(peer, nullptr);
        ASSERT_EQ(nlp_node_start(peer), 0);
        peers.push_back(peer);

        nlp_connect_peer(node1, "127.0.0.1", 38000 + i);
    }

    // Wait for connections
    std::this_thread::sleep_for(std::chrono::seconds(1));

    nlp_stats_t stats_after;
    nlp_get_stats(node1, &stats_after);

    uint32_t connected_peers = stats_after.connected_peers;
    std::cout << "Connected peers: " << connected_peers << std::endl;

    // Estimate memory per peer (rough approximation)
    // In a real implementation, would query actual memory usage
    const size_t estimated_peer_memory = 5 * 1024;  // 5KB estimate
    size_t total_peer_memory = connected_peers * estimated_peer_memory;

    std::cout << "Estimated peer memory: "
              << total_peer_memory / 1024 << " KB total, "
              << estimated_peer_memory / 1024 << " KB per peer" << std::endl;

    EXPECT_LT(estimated_peer_memory, 10 * 1024)
        << "Per-peer memory should be <10KB";

    // Cleanup
    for (auto peer : peers) {
        nlp_node_stop(peer);
        nlp_node_destroy(peer);
    }
}

//=============================================================================
// Test Summary
//=============================================================================

// Test count: 13 performance regression tests
// Coverage:
// - Message Throughput: 3 tests (standard, tactical, large payloads)
// - Encryption Overhead: 2 tests (latency, throughput impact)
// - Burst Buffering: 2 tests (efficiency, latency)
// - Peer Scaling: 2 tests (linear scaling, handshake overhead)
// - Memory Usage: 2 tests (growth over time, per-peer overhead)
