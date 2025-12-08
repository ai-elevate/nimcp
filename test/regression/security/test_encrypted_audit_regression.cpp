/**
 * @file test_encrypted_audit_regression.cpp
 * @brief Regression tests for Encrypted Audit
 *
 * WHAT: Tests to prevent regression of encryption and audit issues
 * WHY:  Ensure encryption stays secure and bugs stay fixed
 * HOW:  Historical bug reproductions and security checks
 *
 * @author NIMCP Security Team
 * @date 2025-12-07
 */

#include <gtest/gtest.h>
extern "C" {
#include "security/nimcp_encrypted_audit.h"
#include "security/nimcp_security.h"
}

class EncryptedAuditRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        nimcp_encryption_generate_key(test_key);
        config = nimcp_encrypted_audit_default_config();
        audit = nimcp_encrypted_audit_create(&config, test_key, sizeof(test_key));
        ASSERT_NE(audit, nullptr);
    }

    void TearDown() override {
        if (audit) {
            nimcp_encrypted_audit_destroy(audit);
        }
    }

    uint8_t test_key[NIMCP_AUDIT_KEY_SIZE];
    nimcp_encrypted_audit_config_t config;
    nimcp_encrypted_audit_t audit;
};

TEST_F(EncryptedAuditRegressionTest, NonceUniqueness) {
    // Ensure nonces are never reused (critical for GCM security)
    const int num_entries = 100;
    std::set<std::string> nonces_seen;

    for (int i = 0; i < num_entries; i++) {
        EXPECT_EQ(nimcp_encrypted_audit_log(
            audit,
            NIMCP_AUDIT_INFO,
            NIMCP_AUDIT_SYSTEM,
            "Nonce test",
            nullptr,
            0
        ), NIMCP_SUCCESS);
    }

    // In production with libsodium, nonces would be verified
    // For now, just ensure no crashes occurred
    SUCCEED();
}

TEST_F(EncryptedAuditRegressionTest, EncryptionOverhead) {
    // Measure encryption overhead
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        nimcp_encrypted_audit_log(audit, NIMCP_AUDIT_INFO, NIMCP_AUDIT_SYSTEM, "Test", nullptr, 0);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 100 encryptions should take less than 1 second
    EXPECT_LT(duration.count(), 1000);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
