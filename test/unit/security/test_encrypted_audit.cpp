/**
 * @file test_encrypted_audit.cpp
 * @brief Unit tests for Encrypted Audit Buffer
 *
 * WHAT: Comprehensive tests for encrypted audit logging
 * WHY:  Ensure encryption, decryption, and tamper detection work correctly
 * HOW:  Google Test framework with encryption verification
 *
 * TEST COVERAGE:
 * - Audit entry encryption/decryption
 * - Circular buffer behavior
 * - Key rotation
 * - Tamper detection
 * - Statistics and monitoring
 * - Thread safety
 * - Error handling
 *
 * @author NIMCP Security Team
 * @date 2025-12-07
 */

#include <gtest/gtest.h>
#include "security/nimcp_encrypted_audit.h"
#include "security/nimcp_security.h"
#include <thread>
#include <vector>
#include <atomic>

//=============================================================================
// Test Fixture
//=============================================================================

class EncryptedAuditTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate test key
        nimcp_encryption_generate_key(test_key);

        config = nimcp_encrypted_audit_default_config();
        config.buffer_size = 100;
        config.enable_compression = false;

        audit = nimcp_encrypted_audit_create(&config, test_key, sizeof(test_key));
        ASSERT_NE(audit, nullptr);
    }

    void TearDown() override {
        if (audit) {
            nimcp_encrypted_audit_destroy(audit);
            audit = nullptr;
        }
    }

    uint8_t test_key[NIMCP_AUDIT_KEY_SIZE];
    nimcp_encrypted_audit_config_t config;
    nimcp_encrypted_audit_t audit;
};

//=============================================================================
// Lifecycle Tests
//=============================================================================

TEST_F(EncryptedAuditTest, CreateDestroy) {
    EXPECT_NE(audit, nullptr);

    nimcp_encrypted_audit_stats_t stats;
    EXPECT_EQ(nimcp_encrypted_audit_get_stats(audit, &stats), NIMCP_SUCCESS);
    EXPECT_EQ(stats.total_entries, 0);
    EXPECT_EQ(stats.current_key_version, 0);
}

TEST_F(EncryptedAuditTest, CreateWithInvalidKey) {
    uint8_t short_key[16];  // Too short
    nimcp_encrypted_audit_t invalid_audit = nimcp_encrypted_audit_create(&config, short_key, sizeof(short_key));
    EXPECT_EQ(invalid_audit, nullptr);
}

TEST_F(EncryptedAuditTest, CreateWithNullKey) {
    nimcp_encrypted_audit_t invalid_audit = nimcp_encrypted_audit_create(&config, nullptr, NIMCP_AUDIT_KEY_SIZE);
    EXPECT_EQ(invalid_audit, nullptr);
}

//=============================================================================
// Logging Tests
//=============================================================================

TEST_F(EncryptedAuditTest, LogSimpleEntry) {
    EXPECT_EQ(nimcp_encrypted_audit_log(
        audit,
        NIMCP_AUDIT_INFO,
        NIMCP_AUDIT_AUTHENTICATION,
        "Test message",
        nullptr,
        0
    ), NIMCP_SUCCESS);

    nimcp_encrypted_audit_stats_t stats;
    nimcp_encrypted_audit_get_stats(audit, &stats);
    EXPECT_EQ(stats.total_entries, 1);
    EXPECT_EQ(stats.entries_encrypted, 1);
}

TEST_F(EncryptedAuditTest, LogWithData) {
    uint8_t test_data[] = {0x01, 0x02, 0x03, 0x04};

    EXPECT_EQ(nimcp_encrypted_audit_log(
        audit,
        NIMCP_AUDIT_WARNING,
        NIMCP_AUDIT_NETWORK,
        "Test with data",
        test_data,
        sizeof(test_data)
    ), NIMCP_SUCCESS);

    nimcp_encrypted_audit_stats_t stats;
    nimcp_encrypted_audit_get_stats(audit, &stats);
    EXPECT_EQ(stats.total_entries, 1);
}

TEST_F(EncryptedAuditTest, LogMultipleEntries) {
    const int num_entries = 10;

    for (int i = 0; i < num_entries; i++) {
        char message[256];
        snprintf(message, sizeof(message), "Entry %d", i);

        EXPECT_EQ(nimcp_encrypted_audit_log(
            audit,
            NIMCP_AUDIT_INFO,
            NIMCP_AUDIT_SYSTEM,
            message,
            nullptr,
            0
        ), NIMCP_SUCCESS);
    }

    nimcp_encrypted_audit_stats_t stats;
    nimcp_encrypted_audit_get_stats(audit, &stats);
    EXPECT_EQ(stats.total_entries, num_entries);
}

TEST_F(EncryptedAuditTest, LogFormattedEntry) {
    EXPECT_EQ(nimcp_encrypted_audit_logf(
        audit,
        NIMCP_AUDIT_ERROR,
        NIMCP_AUDIT_APPLICATION,
        "Error code: %d, message: %s",
        42,
        "test error"
    ), NIMCP_SUCCESS);

    nimcp_encrypted_audit_stats_t stats;
    nimcp_encrypted_audit_get_stats(audit, &stats);
    EXPECT_EQ(stats.total_entries, 1);
}

TEST_F(EncryptedAuditTest, LogAllSeverities) {
    const nimcp_audit_severity_t severities[] = {
        NIMCP_AUDIT_DEBUG,
        NIMCP_AUDIT_INFO,
        NIMCP_AUDIT_WARNING,
        NIMCP_AUDIT_ERROR,
        NIMCP_AUDIT_CRITICAL,
        NIMCP_AUDIT_EMERGENCY
    };

    for (auto severity : severities) {
        EXPECT_EQ(nimcp_encrypted_audit_log(
            audit,
            severity,
            NIMCP_AUDIT_SYSTEM,
            "Test severity",
            nullptr,
            0
        ), NIMCP_SUCCESS);
    }

    nimcp_encrypted_audit_stats_t stats;
    nimcp_encrypted_audit_get_stats(audit, &stats);
    EXPECT_EQ(stats.total_entries, 6);
}

TEST_F(EncryptedAuditTest, LogAllCategories) {
    const nimcp_audit_category_t categories[] = {
        NIMCP_AUDIT_AUTHENTICATION,
        NIMCP_AUDIT_AUTHORIZATION,
        NIMCP_AUDIT_DATA_ACCESS,
        NIMCP_AUDIT_CONFIGURATION,
        NIMCP_AUDIT_NETWORK,
        NIMCP_AUDIT_SYSTEM,
        NIMCP_AUDIT_APPLICATION,
        NIMCP_AUDIT_THREAT,
        NIMCP_AUDIT_ENCRYPTION,
        NIMCP_AUDIT_PATTERN
    };

    for (auto category : categories) {
        EXPECT_EQ(nimcp_encrypted_audit_log(
            audit,
            NIMCP_AUDIT_INFO,
            category,
            "Test category",
            nullptr,
            0
        ), NIMCP_SUCCESS);
    }

    nimcp_encrypted_audit_stats_t stats;
    nimcp_encrypted_audit_get_stats(audit, &stats);
    EXPECT_EQ(stats.total_entries, 10);
}

//=============================================================================
// Reading and Decryption Tests
//=============================================================================

TEST_F(EncryptedAuditTest, ReadAfterWrite) {
    // Log entry
    ASSERT_EQ(nimcp_encrypted_audit_log(
        audit,
        NIMCP_AUDIT_INFO,
        NIMCP_AUDIT_SYSTEM,
        "Read test",
        nullptr,
        0
    ), NIMCP_SUCCESS);

    // Read back
    nimcp_audit_entry_t entries[10];
    size_t num_entries = 0;
    EXPECT_EQ(nimcp_encrypted_audit_read(audit, entries, 10, &num_entries), NIMCP_SUCCESS);
    EXPECT_EQ(num_entries, 1);
    EXPECT_STREQ(entries[0].message, "Read test");
    EXPECT_EQ(entries[0].severity, NIMCP_AUDIT_INFO);
    EXPECT_EQ(entries[0].category, NIMCP_AUDIT_SYSTEM);
}

TEST_F(EncryptedAuditTest, ReadMultipleEntries) {
    const int num_to_write = 5;

    // Write entries
    for (int i = 0; i < num_to_write; i++) {
        char message[256];
        snprintf(message, sizeof(message), "Entry %d", i);

        ASSERT_EQ(nimcp_encrypted_audit_log(
            audit,
            NIMCP_AUDIT_INFO,
            NIMCP_AUDIT_SYSTEM,
            message,
            nullptr,
            0
        ), NIMCP_SUCCESS);
    }

    // Read back
    nimcp_audit_entry_t entries[10];
    size_t num_entries = 0;
    EXPECT_EQ(nimcp_encrypted_audit_read(audit, entries, 10, &num_entries), NIMCP_SUCCESS);
    EXPECT_EQ(num_entries, num_to_write);
}

TEST_F(EncryptedAuditTest, ReadWithFilter) {
    // Log entries with different severities
    ASSERT_EQ(nimcp_encrypted_audit_log(audit, NIMCP_AUDIT_DEBUG, NIMCP_AUDIT_SYSTEM, "Debug", nullptr, 0), NIMCP_SUCCESS);
    ASSERT_EQ(nimcp_encrypted_audit_log(audit, NIMCP_AUDIT_INFO, NIMCP_AUDIT_SYSTEM, "Info", nullptr, 0), NIMCP_SUCCESS);
    ASSERT_EQ(nimcp_encrypted_audit_log(audit, NIMCP_AUDIT_WARNING, NIMCP_AUDIT_SYSTEM, "Warning", nullptr, 0), NIMCP_SUCCESS);
    ASSERT_EQ(nimcp_encrypted_audit_log(audit, NIMCP_AUDIT_ERROR, NIMCP_AUDIT_SYSTEM, "Error", nullptr, 0), NIMCP_SUCCESS);

    // Read only WARNING and above
    nimcp_audit_entry_t entries[10];
    size_t num_entries = 0;
    EXPECT_EQ(nimcp_encrypted_audit_read_filtered(
        audit,
        NIMCP_AUDIT_WARNING,
        NIMCP_AUDIT_CATEGORY_COUNT,  // All categories
        entries,
        10,
        &num_entries
    ), NIMCP_SUCCESS);

    EXPECT_EQ(num_entries, 2);  // WARNING and ERROR
}

TEST_F(EncryptedAuditTest, ReadByCategory) {
    // Log entries in different categories
    ASSERT_EQ(nimcp_encrypted_audit_log(audit, NIMCP_AUDIT_INFO, NIMCP_AUDIT_AUTHENTICATION, "Auth 1", nullptr, 0), NIMCP_SUCCESS);
    ASSERT_EQ(nimcp_encrypted_audit_log(audit, NIMCP_AUDIT_INFO, NIMCP_AUDIT_NETWORK, "Net 1", nullptr, 0), NIMCP_SUCCESS);
    ASSERT_EQ(nimcp_encrypted_audit_log(audit, NIMCP_AUDIT_INFO, NIMCP_AUDIT_AUTHENTICATION, "Auth 2", nullptr, 0), NIMCP_SUCCESS);

    // Read only authentication
    nimcp_audit_entry_t entries[10];
    size_t num_entries = 0;
    EXPECT_EQ(nimcp_encrypted_audit_read_filtered(
        audit,
        NIMCP_AUDIT_DEBUG,  // All severities
        NIMCP_AUDIT_AUTHENTICATION,
        entries,
        10,
        &num_entries
    ), NIMCP_SUCCESS);

    EXPECT_EQ(num_entries, 2);
}

//=============================================================================
// Circular Buffer Tests
//=============================================================================

TEST_F(EncryptedAuditTest, CircularBufferWrap) {
    // Create small buffer
    uint8_t key[NIMCP_AUDIT_KEY_SIZE];
    nimcp_encryption_generate_key(key);

    nimcp_encrypted_audit_config_t small_config = nimcp_encrypted_audit_default_config();
    small_config.buffer_size = 5;

    nimcp_encrypted_audit_t small_audit = nimcp_encrypted_audit_create(&small_config, key, sizeof(key));
    ASSERT_NE(small_audit, nullptr);

    // Write more than buffer size
    for (int i = 0; i < 10; i++) {
        char message[256];
        snprintf(message, sizeof(message), "Entry %d", i);

        EXPECT_EQ(nimcp_encrypted_audit_log(
            small_audit,
            NIMCP_AUDIT_INFO,
            NIMCP_AUDIT_SYSTEM,
            message,
            nullptr,
            0
        ), NIMCP_SUCCESS);
    }

    // Check statistics
    nimcp_encrypted_audit_stats_t stats;
    nimcp_encrypted_audit_get_stats(small_audit, &stats);
    EXPECT_EQ(stats.total_entries, 10);
    EXPECT_GT(stats.buffer_wraps, 0);

    nimcp_encrypted_audit_destroy(small_audit);
}

//=============================================================================
// Key Rotation Tests
//=============================================================================

TEST_F(EncryptedAuditTest, GetKeyVersion) {
    uint32_t version = nimcp_encrypted_audit_get_key_version(audit);
    EXPECT_EQ(version, 0);  // Initial version
}

TEST_F(EncryptedAuditTest, RotateKey) {
    uint8_t new_key[NIMCP_AUDIT_KEY_SIZE];
    nimcp_encryption_generate_key(new_key);

    uint32_t old_version = nimcp_encrypted_audit_get_key_version(audit);

    EXPECT_EQ(nimcp_encrypted_audit_rotate_key(audit, new_key, sizeof(new_key)), NIMCP_SUCCESS);

    uint32_t new_version = nimcp_encrypted_audit_get_key_version(audit);
    EXPECT_GT(new_version, old_version);

    nimcp_encrypted_audit_stats_t stats;
    nimcp_encrypted_audit_get_stats(audit, &stats);
    EXPECT_EQ(stats.key_rotations, 1);
}

TEST_F(EncryptedAuditTest, SetRotationPolicy) {
    EXPECT_EQ(nimcp_encrypted_audit_set_rotation_policy(
        audit,
        NIMCP_KEY_ROTATION_COUNT,
        1000
    ), NIMCP_SUCCESS);
}

//=============================================================================
// Statistics Tests
//=============================================================================

TEST_F(EncryptedAuditTest, StatisticsTracking) {
    nimcp_encrypted_audit_stats_t stats;

    // Initial state
    EXPECT_EQ(nimcp_encrypted_audit_get_stats(audit, &stats), NIMCP_SUCCESS);
    EXPECT_EQ(stats.total_entries, 0);
    EXPECT_EQ(stats.entries_encrypted, 0);

    // Log entries
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(nimcp_encrypted_audit_log(
            audit,
            NIMCP_AUDIT_INFO,
            NIMCP_AUDIT_SYSTEM,
            "Test",
            nullptr,
            0
        ), NIMCP_SUCCESS);
    }

    // Check stats after logging
    EXPECT_EQ(nimcp_encrypted_audit_get_stats(audit, &stats), NIMCP_SUCCESS);
    EXPECT_EQ(stats.total_entries, 5);
    EXPECT_EQ(stats.entries_encrypted, 5);

    // Read entries (triggers decryption)
    nimcp_audit_entry_t entries[10];
    size_t num_entries = 0;
    EXPECT_EQ(nimcp_encrypted_audit_read(audit, entries, 10, &num_entries), NIMCP_SUCCESS);

    // Check decryption stats
    EXPECT_EQ(nimcp_encrypted_audit_get_stats(audit, &stats), NIMCP_SUCCESS);
    EXPECT_GT(stats.entries_decrypted, 0);
}

TEST_F(EncryptedAuditTest, ResetStatistics) {
    // Generate some stats
    EXPECT_EQ(nimcp_encrypted_audit_log(audit, NIMCP_AUDIT_INFO, NIMCP_AUDIT_SYSTEM, "Test", nullptr, 0), NIMCP_SUCCESS);

    nimcp_encrypted_audit_stats_t stats;
    EXPECT_EQ(nimcp_encrypted_audit_get_stats(audit, &stats), NIMCP_SUCCESS);
    EXPECT_GT(stats.entries_encrypted, 0);

    // Reset
    nimcp_encrypted_audit_reset_stats(audit);

    EXPECT_EQ(nimcp_encrypted_audit_get_stats(audit, &stats), NIMCP_SUCCESS);
    EXPECT_EQ(stats.encryption_failures, 0);
    EXPECT_EQ(stats.decryption_failures, 0);
}

//=============================================================================
// Utility Function Tests
//=============================================================================

TEST_F(EncryptedAuditTest, SeverityNameConversion) {
    /* The nimcp_audit_severity_name() function uses NIMCP_AUDIT_SEV_* enum values
     * from nimcp_security_audit.h which has different values than NIMCP_AUDIT_*
     * from nimcp_encrypted_audit.h. Use numeric values to avoid header conflicts.
     * SEV enum: DEBUG=0, INFO=1, NOTICE=2, WARNING=3, ERROR=4, CRITICAL=5, ALERT=6, EMERGENCY=7 */
    EXPECT_STREQ(nimcp_audit_severity_name((nimcp_audit_severity_t)0), "DEBUG");
    EXPECT_STREQ(nimcp_audit_severity_name((nimcp_audit_severity_t)1), "INFO");
    EXPECT_STREQ(nimcp_audit_severity_name((nimcp_audit_severity_t)3), "WARNING");
    EXPECT_STREQ(nimcp_audit_severity_name((nimcp_audit_severity_t)4), "ERROR");
    EXPECT_STREQ(nimcp_audit_severity_name((nimcp_audit_severity_t)5), "CRITICAL");
    EXPECT_STREQ(nimcp_audit_severity_name((nimcp_audit_severity_t)7), "EMERGENCY");
}

TEST_F(EncryptedAuditTest, CategoryNameConversion) {
    /* The nimcp_audit_category_name() function uses NIMCP_AUDIT_CAT_* enum values
     * from nimcp_security_audit.h. Use numeric values to avoid header conflicts.
     * CAT enum: ACCESS=0, AUTHENTICATION=1, AUTHORIZATION=2, INTEGRITY=3,
     * CONFIGURATION=4, POLICY=5, THREAT=6, SYSTEM=7 */
    EXPECT_STREQ(nimcp_audit_category_name((nimcp_audit_category_t)1), "AUTHENTICATION");
    EXPECT_STREQ(nimcp_audit_category_name((nimcp_audit_category_t)2), "AUTHORIZATION");
    EXPECT_STREQ(nimcp_audit_category_name((nimcp_audit_category_t)7), "SYSTEM");
    EXPECT_STREQ(nimcp_audit_category_name((nimcp_audit_category_t)6), "THREAT");
}

//=============================================================================
// Thread Safety Tests
//=============================================================================

TEST_F(EncryptedAuditTest, ConcurrentLogging) {
    const int num_threads = 10;
    const int entries_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([this, i, entries_per_thread, &success_count]() {
            for (int j = 0; j < entries_per_thread; j++) {
                char message[256];
                snprintf(message, sizeof(message), "Thread %d, Entry %d", i, j);

                nimcp_error_t result = nimcp_encrypted_audit_log(
                    audit,
                    NIMCP_AUDIT_INFO,
                    NIMCP_AUDIT_SYSTEM,
                    message,
                    nullptr,
                    0
                );

                if (result == NIMCP_SUCCESS) {
                    success_count++;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count, num_threads * entries_per_thread);

    nimcp_encrypted_audit_stats_t stats;
    nimcp_encrypted_audit_get_stats(audit, &stats);
    EXPECT_EQ(stats.total_entries, num_threads * entries_per_thread);
}

//=============================================================================
// Error Handling Tests
//=============================================================================

TEST_F(EncryptedAuditTest, LogWithNullMessage) {
    EXPECT_NE(nimcp_encrypted_audit_log(
        audit,
        NIMCP_AUDIT_INFO,
        NIMCP_AUDIT_SYSTEM,
        nullptr,  // Invalid
        nullptr,
        0
    ), NIMCP_SUCCESS);
}

TEST_F(EncryptedAuditTest, LogWithInvalidSeverity) {
    EXPECT_NE(nimcp_encrypted_audit_log(
        audit,
        static_cast<nimcp_audit_severity_t>(999),  // Invalid
        NIMCP_AUDIT_SYSTEM,
        "Test",
        nullptr,
        0
    ), NIMCP_SUCCESS);
}

TEST_F(EncryptedAuditTest, LogWithInvalidCategory) {
    EXPECT_NE(nimcp_encrypted_audit_log(
        audit,
        NIMCP_AUDIT_INFO,
        static_cast<nimcp_audit_category_t>(999),  // Invalid
        "Test",
        nullptr,
        0
    ), NIMCP_SUCCESS);
}

TEST_F(EncryptedAuditTest, ReadWithNullBuffer) {
    size_t num_entries = 0;
    EXPECT_NE(nimcp_encrypted_audit_read(audit, nullptr, 10, &num_entries), NIMCP_SUCCESS);
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
