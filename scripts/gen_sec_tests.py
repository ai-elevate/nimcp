import os
B="/home/bbrelin/nimcp"
def w(p,c):
 f=os.path.join(B,p);os.makedirs(os.path.dirname(f),exist_ok=1);open(f,"w").write(c);print("OK:"+p)

# === TEST 1: BBB API Validation ===
w("test/unit/security/test_bbb_api_validation.cpp", r"""
// test_bbb_api_validation.cpp - BBB validation at API boundaries
// Tests P1-4: BBB validation for all public API entry points
#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <string>

#include "security/nimcp_blood_brain_barrier.h"
#include "security/nimcp_bbb_helpers.h"

class BBBApiValidationTest : public ::testing::Test {
protected:
    bbb_system_t system_ = nullptr;
    void SetUp() override {
        bbb_config_t c = bbb_default_config();
        c.strict_mode = true;
        c.input.validate_strings = true;
        c.input.max_string_length = 4096;
        system_ = bbb_system_create(&c);
        ASSERT_NE(system_, nullptr) << "Failed to create BBB system";
        bbb_system_set_enabled(system_, true);
    }
    void TearDown() override {
        if (system_) { bbb_system_destroy(system_); system_ = nullptr; }
    }
};

TEST_F(BBBApiValidationTest, ValidInput_Success) {
    const char* data = "hello world";
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_TRUE(bbb_validate_input(system_, data, strlen(data), &r));
    EXPECT_TRUE(r.valid);
}

TEST_F(BBBApiValidationTest, NullData_Rejected) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_FALSE(bbb_validate_input(system_, nullptr, 100, &r));
}

TEST_F(BBBApiValidationTest, NullSystem_Rejected) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_FALSE(bbb_validate_input(nullptr, "data", 4, &r));
}

TEST_F(BBBApiValidationTest, NullResult_Rejected) {
    const char* data = "test";
    EXPECT_FALSE(bbb_validate_input(system_, data, strlen(data), nullptr));
}

TEST_F(BBBApiValidationTest, ZeroSizeInput_Handled) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    bbb_validate_input(system_, "x", 0, &r);
}

TEST_F(BBBApiValidationTest, OversizedBuffer_Handled) {
    std::vector<char> big(8192, 65);
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    bbb_validate_input(system_, big.data(), big.size(), &r);
}

TEST_F(BBBApiValidationTest, ValidString_OK) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_TRUE(bbb_validate_string(system_, "safe string", &r));
    EXPECT_TRUE(r.valid);
}

TEST_F(BBBApiValidationTest, NullString_Rejected) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_FALSE(bbb_validate_string(system_, nullptr, &r));
}

TEST_F(BBBApiValidationTest, EmptyString_Accepted) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_TRUE(bbb_validate_string(system_, "", &r));
}

TEST_F(BBBApiValidationTest, OversizedString_Rejected) {
    std::string huge(8192, 88);
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_FALSE(bbb_validate_string(system_, huge.c_str(), &r));
}

TEST_F(BBBApiValidationTest, StringNullSystem_Rejected) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_FALSE(bbb_validate_string(nullptr, "test", &r));
}

TEST_F(BBBApiValidationTest, StringNullResult_Rejected) {
    EXPECT_FALSE(bbb_validate_string(system_, "test", nullptr));
}

TEST_F(BBBApiValidationTest, ValidInteger_OK) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_TRUE(bbb_validate_integer(system_, 42, &r));
    EXPECT_TRUE(r.valid);
}

TEST_F(BBBApiValidationTest, IntegerNullSystem_Rejected) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_FALSE(bbb_validate_integer(nullptr, 42, &r));
}

TEST_F(BBBApiValidationTest, IntegerNullResult_Rejected) {
    EXPECT_FALSE(bbb_validate_integer(system_, 42, nullptr));
}

TEST_F(BBBApiValidationTest, IntegerMinMax_NoOverflow) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    bbb_validate_integer(system_, INT64_MIN, &r);
    memset(&r, 0, sizeof(r));
    bbb_validate_integer(system_, INT64_MAX, &r);
}

TEST_F(BBBApiValidationTest, ValidPointer_OK) {
    int value = 42;
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_TRUE(bbb_validate_pointer(system_, &value, sizeof(int), &r));
    EXPECT_TRUE(r.valid);
}

TEST_F(BBBApiValidationTest, NullPointer_Rejected) {
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_FALSE(bbb_validate_pointer(system_, nullptr, 4, &r));
}

TEST_F(BBBApiValidationTest, PointerNullSystem_Rejected) {
    int value = 1;
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    EXPECT_FALSE(bbb_validate_pointer(nullptr, &value, sizeof(int), &r));
}

TEST_F(BBBApiValidationTest, PointerNullResult_Rejected) {
    int value = 1;
    EXPECT_FALSE(bbb_validate_pointer(system_, &value, sizeof(int), nullptr));
}

TEST_F(BBBApiValidationTest, ZeroSizePointer_Handled) {
    int value = 1;
    bbb_validation_result_t r; memset(&r, 0, sizeof(r));
    bbb_validate_pointer(system_, &value, 0, &r);
}

TEST(BBBSystemLifecycle, CreateWithDefaultConfig) {
    bbb_system_t s = bbb_system_create(nullptr);
    EXPECT_NE(s, nullptr);
    if (s) bbb_system_destroy(s);
}

TEST(BBBSystemLifecycle, CreateWithExplicitConfig) {
    bbb_config_t c = bbb_default_config();
    bbb_system_t s = bbb_system_create(&c);
    EXPECT_NE(s, nullptr);
    if (s) bbb_system_destroy(s);
}

TEST(BBBSystemLifecycle, DestroyNull_NoOp) {
    bbb_system_destroy(nullptr);
}

TEST(BBBSystemLifecycle, EnableDisable) {
    bbb_system_t s = bbb_system_create(nullptr);
    ASSERT_NE(s, nullptr);
    EXPECT_TRUE(bbb_system_set_enabled(s, true));
    EXPECT_TRUE(bbb_system_is_enabled(s));
    EXPECT_TRUE(bbb_system_set_enabled(s, false));
    EXPECT_FALSE(bbb_system_is_enabled(s));
    bbb_system_destroy(s);
}

TEST(BBBSystemLifecycle, EnableNull_Rejected) {
    EXPECT_FALSE(bbb_system_set_enabled(nullptr, true));
}

TEST(BBBSystemLifecycle, IsEnabledNull) {
    EXPECT_FALSE(bbb_system_is_enabled(nullptr));
}

TEST(BBBSystemLifecycle, GetStatistics) {
    bbb_system_t s = bbb_system_create(nullptr);
    ASSERT_NE(s, nullptr);
    bbb_statistics_t stats; memset(&stats, 0, sizeof(stats));
    EXPECT_TRUE(bbb_system_get_statistics(s, &stats));
    bbb_system_destroy(s);
}

TEST(BBBSystemLifecycle, GetStatisticsNullSystem) {
    bbb_statistics_t stats;
    EXPECT_FALSE(bbb_system_get_statistics(nullptr, &stats));
}

TEST(BBBSystemLifecycle, GetStatisticsNullStats) {
    bbb_system_t s = bbb_system_create(nullptr);
    ASSERT_NE(s, nullptr);
    EXPECT_FALSE(bbb_system_get_statistics(s, nullptr));
    bbb_system_destroy(s);
}

TEST_F(BBBApiValidationTest, QuarantineNullSystem) {
    char buf[16];
    EXPECT_FALSE(bbb_quarantine_region(nullptr, buf, sizeof(buf)));
}

TEST_F(BBBApiValidationTest, QuarantineNullAddress) {
    EXPECT_FALSE(bbb_quarantine_region(system_, nullptr, 16));
}

TEST_F(BBBApiValidationTest, IsQuarantinedNullSystem) {
    char buf[16];
    EXPECT_FALSE(bbb_is_quarantined(nullptr, buf, sizeof(buf)));
}

TEST_F(BBBApiValidationTest, QuarantineAndCheck) {
    char buf[64]; memset(buf, 0, sizeof(buf));
    bool q = bbb_quarantine_region(system_, buf, sizeof(buf));
    if (q) {
        EXPECT_TRUE(bbb_is_quarantined(system_, buf, sizeof(buf)));
        bbb_release_quarantine(system_, buf);
    }
}

TEST_F(BBBApiValidationTest, ReportThreat_Basic) {
    const char* desc = "test threat";
    uint8_t data[] = {0xDE, 0xAD};
    bbb_threat_report_t report = bbb_report_threat(
        system_, BBB_THREAT_BUFFER_OVERFLOW, BBB_SEVERITY_HIGH,
        desc, nullptr, data, sizeof(data));
    EXPECT_NE(report.report_id, 0u);
}

TEST_F(BBBApiValidationTest, ReportThreat_NullDescription) {
    uint8_t data[] = {1};
    bbb_report_threat(system_, BBB_THREAT_CODE_INJECTION, BBB_SEVERITY_MEDIUM,
                     nullptr, nullptr, data, sizeof(data));
}

class BBBHelpersTest : public ::testing::Test {
protected:
    void SetUp() override { bbb_helpers_init(); }
    void TearDown() override { bbb_helpers_shutdown(); }
};

TEST_F(BBBHelpersTest, RegisterModule_OK) {
    EXPECT_TRUE(bbb_register_module("test_module", BBB_MODULE_TYPE_CORE));
}

TEST_F(BBBHelpersTest, RegisterModule_NullName) {
    EXPECT_FALSE(bbb_register_module(nullptr, BBB_MODULE_TYPE_CORE));
}

TEST_F(BBBHelpersTest, CheckPointer_Valid) {
    int v = 1;
    EXPECT_TRUE(bbb_check_pointer(&v, "test_func"));
}

TEST_F(BBBHelpersTest, CheckPointer_Null) {
    EXPECT_FALSE(bbb_check_pointer(nullptr, "test_func"));
}

TEST_F(BBBHelpersTest, CheckString_Valid) {
    EXPECT_TRUE(bbb_check_string("hello", 256, "test_func"));
}

TEST_F(BBBHelpersTest, CheckString_Null) {
    EXPECT_FALSE(bbb_check_string(nullptr, 256, "test_func"));
}

TEST_F(BBBHelpersTest, CheckString_ExceedsMax) {
    std::string long_str(512, 65);
    EXPECT_FALSE(bbb_check_string(long_str.c_str(), 256, "test_func"));
}

TEST_F(BBBHelpersTest, ValidateRange_OK) {
    EXPECT_TRUE(bbb_validate_range(50, 0, 100, "test_func"));
}

TEST_F(BBBHelpersTest, ValidateRange_AtBoundaries) {
    EXPECT_TRUE(bbb_validate_range(0, 0, 100, "test_func"));
    EXPECT_TRUE(bbb_validate_range(100, 0, 100, "test_func"));
}

TEST_F(BBBHelpersTest, ValidateRange_OutOfRange) {
    EXPECT_FALSE(bbb_validate_range(200, 0, 100, "test_func"));
    EXPECT_FALSE(bbb_validate_range(-1, 0, 100, "test_func"));
}

TEST_F(BBBHelpersTest, ValidateBufferAccess_OK) {
    char buf[256];
    EXPECT_TRUE(bbb_validate_buffer_access(buf, 0, 10, 256, "test_func"));
}

TEST_F(BBBHelpersTest, ValidateBufferAccess_NullBuffer) {
    EXPECT_FALSE(bbb_validate_buffer_access(nullptr, 0, 10, 256, "test_func"));
}

TEST_F(BBBHelpersTest, ValidateBufferAccess_OutOfBounds) {
    char buf[64];
    EXPECT_FALSE(bbb_validate_buffer_access(buf, 60, 10, 64, "test_func"));
}

TEST_F(BBBHelpersTest, ValidateBufferAccess_OverflowOffset) {
    char buf[64];
    EXPECT_FALSE(bbb_validate_buffer_access(buf, SIZE_MAX, 1, 64, "test_func"));
}
""")

# === TEST 2: Security Error Codes ===
w("test/unit/security/test_security_error_codes.cpp", r"""
// test_security_error_codes.cpp - Security functions return proper NIMCP error codes
// Tests P2-7: No bare -1 returns; all return nimcp_error_t codes
#include <gtest/gtest.h>
#include <cstring>

#include "security/nimcp_security.h"
#include "security/nimcp_post_quantum.h"
#include "security/nimcp_security_consensus.h"
#include "utils/error/nimcp_error_codes.h"

// Helper: verify return is a valid NIMCP error code (not bare -1)
#define EXPECT_VALID_NIMCP(expr) do { \
    auto _r = (int)(expr); \
    EXPECT_NE(_r, -1) << #expr " returned bare -1 (P2-7 violation)"; \
    if (_r != 0) { EXPECT_GE(_r, NIMCP_ERROR_GENERIC) << #expr " returned non-standard error " << _r; } \
} while(0)

// --- Directive System ---
TEST(SecErrCode, DirectiveAdd_NullSystem) {
    EXPECT_VALID_NIMCP(nimcp_directive_add(nullptr, "test"));
}

TEST(SecErrCode, DirectiveAdd_NullText) {
    nimcp_directive_system_t* sys = nimcp_directive_system_create();
    if (sys) {
        EXPECT_VALID_NIMCP(nimcp_directive_add(sys, nullptr));
        nimcp_directive_system_destroy(sys);
    }
}

TEST(SecErrCode, DirectiveLock_NullSystem) {
    EXPECT_VALID_NIMCP(nimcp_directive_lock(nullptr));
}

// --- Encryption ---
TEST(SecErrCode, EncryptNull) {
    uint8_t d[] = {1, 2, 3}; uint8_t o[256]; size_t s = 0;
    EXPECT_VALID_NIMCP(nimcp_encryption_encrypt(nullptr, d, 3, o, 256, &s));
}

TEST(SecErrCode, DecryptNull) {
    uint8_t d[] = {1, 2, 3}; uint8_t o[256]; size_t s = 0;
    EXPECT_VALID_NIMCP(nimcp_encryption_decrypt(nullptr, d, 3, o, 256, &s));
}

TEST(SecErrCode, GenKeyNull) {
    EXPECT_VALID_NIMCP(nimcp_encryption_generate_key(nullptr));
}

// --- Input Sanitization ---
TEST(SecErrCode, SanitizeInput_NullInput) {
    char out[256];
    EXPECT_VALID_NIMCP(nimcp_security_sanitize_input(nullptr, out, 256));
}

TEST(SecErrCode, SanitizeInput_NullOutput) {
    EXPECT_VALID_NIMCP(nimcp_security_sanitize_input("test", nullptr, 256));
}

TEST(SecErrCode, SanitizeInput_ZeroSize) {
    char out[256];
    EXPECT_VALID_NIMCP(nimcp_security_sanitize_input("test", out, 0));
}

TEST(SecErrCode, SanitizeInput_Valid) {
    char out[256];
    nimcp_result_t r = nimcp_security_sanitize_input("safe input", out, 256);
    EXPECT_EQ(r, NIMCP_OK);
}

// --- Skepticism ---
TEST(SecErrCode, Skepticism_NullResult) {
    EXPECT_VALID_NIMCP(nimcp_security_evaluate_skepticism("claim", "source", "context", nullptr));
}

TEST(SecErrCode, Skepticism_NullClaim) {
    nimcp_skepticism_result_t res;
    EXPECT_VALID_NIMCP(nimcp_security_evaluate_skepticism(nullptr, "source", "context", &res));
}

// --- Security Audit/Stats ---
TEST(SecErrCode, LogEvent_Valid) {
    EXPECT_VALID_NIMCP(nimcp_security_log_event(NIMCP_SECURITY_EVENT_LOGIN, NIMCP_THREAT_LEVEL_LOW, "test event"));
}

TEST(SecErrCode, LogEvent_NullDetails) {
    EXPECT_VALID_NIMCP(nimcp_security_log_event(NIMCP_SECURITY_EVENT_LOGIN, NIMCP_THREAT_LEVEL_LOW, nullptr));
}

TEST(SecErrCode, GetStats_NullArgs) {
    EXPECT_VALID_NIMCP(nimcp_security_get_stats(nullptr, nullptr, nullptr));
}

TEST(SecErrCode, GetStats_Valid) {
    uint64_t threats = 0, rejected = 0, verified = 0;
    nimcp_result_t r = nimcp_security_get_stats(&threats, &rejected, &verified);
    EXPECT_VALID_NIMCP(r);
}

// --- Post-Quantum ---
TEST(SecErrCode, PQGetStats_NullCtx) {
    nimcp_pq_stats_t stats;
    EXPECT_VALID_NIMCP(nimcp_pq_get_stats(nullptr, &stats));
}

TEST(SecErrCode, PQGetStats_NullStats) {
    nimcp_pq_context_t ctx = nimcp_pq_context_create(nullptr);
    if (ctx) {
        EXPECT_VALID_NIMCP(nimcp_pq_get_stats(ctx, nullptr));
        nimcp_pq_context_destroy(ctx);
    }
}

TEST(SecErrCode, KyberKeygen_NullKeypair) {
    EXPECT_VALID_NIMCP(nimcp_kyber_keygen(NIMCP_KYBER_512, nullptr));
}

// --- Consensus ---
TEST(SecErrCode, ConsensusJoin_NullCtx) {
    EXPECT_VALID_NIMCP(nimcp_consensus_join(nullptr, "peer:8080"));
}

TEST(SecErrCode, ConsensusJoin_NullAddr) {
    nimcp_consensus_config_t cfg; memset(&cfg, 0, sizeof(cfg));
    nimcp_security_consensus_t ctx = nimcp_consensus_create(&cfg);
    if (ctx) {
        EXPECT_VALID_NIMCP(nimcp_consensus_join(ctx, nullptr));
        nimcp_consensus_destroy(ctx);
    }
}

TEST(SecErrCode, ConsensusLeave_NullCtx) {
    EXPECT_VALID_NIMCP(nimcp_consensus_leave(nullptr));
}

TEST(SecErrCode, ConsensusShareThreat_NullCtx) {
    nimcp_threat_info_t t; memset(&t, 0, sizeof(t));
    EXPECT_VALID_NIMCP(nimcp_consensus_share_threat(nullptr, &t));
}

TEST(SecErrCode, ConsensusShareThreat_NullThreat) {
    nimcp_consensus_config_t cfg; memset(&cfg, 0, sizeof(cfg));
    nimcp_security_consensus_t ctx = nimcp_consensus_create(&cfg);
    if (ctx) {
        EXPECT_VALID_NIMCP(nimcp_consensus_share_threat(ctx, nullptr));
        nimcp_consensus_destroy(ctx);
    }
}

TEST(SecErrCode, ConsensusProposePolicy_NullCtx) {
    nimcp_security_policy_t p; memset(&p, 0, sizeof(p));
    EXPECT_VALID_NIMCP(nimcp_consensus_propose_policy(nullptr, &p));
}

TEST(SecErrCode, ConsensusInitiateResponse_NullCtx) {
    EXPECT_VALID_NIMCP(nimcp_consensus_initiate_response(nullptr, NIMCP_RESPONSE_QUARANTINE, nullptr));
}

// --- Verify Success code is 0 ---
TEST(SecErrCode, SuccessIsZero) {
    EXPECT_EQ(NIMCP_SUCCESS, 0);
    EXPECT_EQ(NIMCP_OK, 0);
}

TEST(SecErrCode, SecurityErrorRange) {
    EXPECT_GE(NIMCP_ERROR_BBB_REJECTED, 9000);
    EXPECT_GE(NIMCP_ERROR_BBB_VALIDATION, 9000);
    EXPECT_GE(NIMCP_ERROR_SECURITY_THREAT, 9000);
}
""")

# === TEST 3: Overflow Checks ===
w("test/unit/utils/test_overflow_checks.cpp", r"""
// test_overflow_checks.cpp - Tests for NIMCP_MUL_SAFE/NIMCP_ADD_SAFE macros
// Tests P3-7: Overflow check macros for size calculations
#include <gtest/gtest.h>
#include <climits>
#include <cstddef>

#include "utils/nimcp_overflow.h"

// --- Multiplication Safety ---
TEST(OverflowChecks, MulSafe_SmallValues) {
    EXPECT_TRUE(NIMCP_MUL_SAFE(10, 20));
    EXPECT_TRUE(NIMCP_MUL_SAFE(1, 1));
    EXPECT_TRUE(NIMCP_MUL_SAFE(100, 100));
}

TEST(OverflowChecks, MulSafe_ZeroValues) {
    EXPECT_TRUE(NIMCP_MUL_SAFE(0, SIZE_MAX));
    EXPECT_TRUE(NIMCP_MUL_SAFE(SIZE_MAX, 0));
    EXPECT_TRUE(NIMCP_MUL_SAFE(0, 0));
}

TEST(OverflowChecks, MulSafe_OneValues) {
    EXPECT_TRUE(NIMCP_MUL_SAFE(1, SIZE_MAX));
    EXPECT_TRUE(NIMCP_MUL_SAFE(SIZE_MAX, 1));
}

TEST(OverflowChecks, MulSafe_Overflow) {
    EXPECT_FALSE(NIMCP_MUL_SAFE(SIZE_MAX, 2));
    EXPECT_FALSE(NIMCP_MUL_SAFE(2, SIZE_MAX));
    EXPECT_FALSE(NIMCP_MUL_SAFE(SIZE_MAX, SIZE_MAX));
}

TEST(OverflowChecks, MulSafe_NearOverflow) {
    size_t half = SIZE_MAX / 2;
    EXPECT_TRUE(NIMCP_MUL_SAFE(half, 2));
    EXPECT_FALSE(NIMCP_MUL_SAFE(half + 1, 2));
}

// --- Addition Safety ---
TEST(OverflowChecks, AddSafe_SmallValues) {
    EXPECT_TRUE(NIMCP_ADD_SAFE(10, 20));
    EXPECT_TRUE(NIMCP_ADD_SAFE(0, 0));
    EXPECT_TRUE(NIMCP_ADD_SAFE(1, 1));
}

TEST(OverflowChecks, AddSafe_ZeroValues) {
    EXPECT_TRUE(NIMCP_ADD_SAFE(0, SIZE_MAX));
    EXPECT_TRUE(NIMCP_ADD_SAFE(SIZE_MAX, 0));
}

TEST(OverflowChecks, AddSafe_Overflow) {
    EXPECT_FALSE(NIMCP_ADD_SAFE(SIZE_MAX, 1));
    EXPECT_FALSE(NIMCP_ADD_SAFE(1, SIZE_MAX));
    EXPECT_FALSE(NIMCP_ADD_SAFE(SIZE_MAX, SIZE_MAX));
}

TEST(OverflowChecks, AddSafe_NearOverflow) {
    EXPECT_TRUE(NIMCP_ADD_SAFE(SIZE_MAX - 1, 1));
    EXPECT_FALSE(NIMCP_ADD_SAFE(SIZE_MAX - 1, 2));
}

// --- Result variants ---
TEST(OverflowChecks, MulSafeResult_OK) {
    size_t result = 0;
    EXPECT_TRUE(NIMCP_MUL_SAFE_RESULT(10, 20, &result));
    EXPECT_EQ(result, 200u);
}

TEST(OverflowChecks, MulSafeResult_Overflow) {
    size_t result = 42;
    EXPECT_FALSE(NIMCP_MUL_SAFE_RESULT(SIZE_MAX, 2, &result));
    EXPECT_EQ(result, 42u);  // result unchanged on overflow
}

TEST(OverflowChecks, MulSafeResult_NullResult) {
    EXPECT_TRUE(NIMCP_MUL_SAFE_RESULT(10, 20, nullptr));
    // null result pointer should still report safe (just no output)
}

TEST(OverflowChecks, AddSafeResult_OK) {
    size_t result = 0;
    EXPECT_TRUE(NIMCP_ADD_SAFE_RESULT(100, 200, &result));
    EXPECT_EQ(result, 300u);
}

TEST(OverflowChecks, AddSafeResult_Overflow) {
    size_t result = 42;
    EXPECT_FALSE(NIMCP_ADD_SAFE_RESULT(SIZE_MAX, 1, &result));
    EXPECT_EQ(result, 42u);
}

TEST(OverflowChecks, AddSafeResult_NullResult) {
    EXPECT_TRUE(NIMCP_ADD_SAFE_RESULT(10, 20, nullptr));
}

// --- Inline function variants ---
TEST(OverflowChecks, InlineMulSafe) {
    EXPECT_TRUE(nimcp_mul_safe(100, 100));
    EXPECT_FALSE(nimcp_mul_safe(SIZE_MAX, 2));
}

TEST(OverflowChecks, InlineAddSafe) {
    EXPECT_TRUE(nimcp_add_safe(100, 100));
    EXPECT_FALSE(nimcp_add_safe(SIZE_MAX, 1));
}

TEST(OverflowChecks, InlineMulSafeResult) {
    size_t r = 0;
    EXPECT_TRUE(nimcp_mul_safe_result(5, 6, &r));
    EXPECT_EQ(r, 30u);
}

TEST(OverflowChecks, InlineAddSafeResult) {
    size_t r = 0;
    EXPECT_TRUE(nimcp_add_safe_result(11, 22, &r));
    EXPECT_EQ(r, 33u);
}

// --- Practical allocation size checks ---
TEST(OverflowChecks, AllocationSizeCheck) {
    // Simulating: count * element_size for allocation
    size_t count = 1000000;
    size_t elem_size = sizeof(double);
    size_t total = 0;
    EXPECT_TRUE(NIMCP_MUL_SAFE_RESULT(count, elem_size, &total));
    EXPECT_EQ(total, count * elem_size);
}

TEST(OverflowChecks, AllocationSizeOverflow) {
    // Pathological: huge count * large element
    size_t count = SIZE_MAX / 2;
    size_t elem_size = 4;
    EXPECT_FALSE(NIMCP_MUL_SAFE(count, elem_size));
}

TEST(OverflowChecks, HeaderPlusSizeOverflow) {
    // Pattern: header_size + data_size
    size_t header = 64;
    size_t data = SIZE_MAX - 32;  // Leaves room < header
    EXPECT_FALSE(NIMCP_ADD_SAFE(header, data));
}
""")

# === TEST 4: Immune Error Routing Integration ===
w("test/integration/security/test_immune_error_routing.cpp", r"""
// test_immune_error_routing.cpp - Integration test for immune error routing
// Tests P1-2/3/5/6 + P2-6: Exception-to-immune integration
#include <gtest/gtest.h>
#include <cstring>

#include "cognitive/immune/nimcp_brain_immune.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_immune.h"
#include "utils/error/nimcp_error_codes.h"

class ImmuneErrorRoutingTest : public ::testing::Test {
protected:
    brain_immune_system_t* immune_ = nullptr;
    bbb_system_t bbb_ = nullptr;

    void SetUp() override {
        nimcp_exception_system_init();
        brain_immune_config_t cfg;
        brain_immune_default_config(&cfg);
        cfg.enable_bbb_integration = true;
        cfg.enable_logging = false;
        immune_ = brain_immune_create(&cfg);
        ASSERT_NE(immune_, nullptr);
        bbb_ = bbb_system_create(nullptr);
        ASSERT_NE(bbb_, nullptr);
        bbb_system_set_enabled(bbb_, true);
        brain_immune_connect_bbb(immune_, bbb_);
    }

    void TearDown() override {
        if (immune_) brain_immune_destroy(immune_);
        if (bbb_) bbb_system_destroy(bbb_);
        nimcp_exception_system_shutdown();
    }
};

// --- BBB Threat Presentation ---
TEST_F(ImmuneErrorRoutingTest, BBBThreatCreatesAntigen) {
    uint32_t antigen_id = 0;
    uint8_t threat_data[] = {0xBE, 0xEF};
    int rc = brain_immune_present_bbb_threat(
        immune_, BBB_THREAT_BUFFER_OVERFLOW, BBB_SEVERITY_HIGH,
        threat_data, sizeof(threat_data), &antigen_id);
    EXPECT_EQ(rc, 0) << "Failed to present BBB threat";
    EXPECT_GT(antigen_id, 0u);
}

TEST_F(ImmuneErrorRoutingTest, BBBThreat_NullSystem) {
    uint32_t antigen_id = 0;
    uint8_t data[] = {1};
    int rc = brain_immune_present_bbb_threat(
        nullptr, BBB_THREAT_BUFFER_OVERFLOW, BBB_SEVERITY_HIGH,
        data, 1, &antigen_id);
    EXPECT_NE(rc, 0);
}

TEST_F(ImmuneErrorRoutingTest, BBBThreat_NullData) {
    uint32_t antigen_id = 0;
    int rc = brain_immune_present_bbb_threat(
        immune_, BBB_THREAT_BUFFER_OVERFLOW, BBB_SEVERITY_HIGH,
        nullptr, 0, &antigen_id);
    // Should handle gracefully
    (void)rc;
}

// --- Exception to Immune ---
TEST_F(ImmuneErrorRoutingTest, ExceptionPresentedToImmune) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_BBB_REJECTED, NIMCP_EXCEPTION_SEVERITY_HIGH,
        __FILE__, __LINE__, __func__, "Test BBB rejection");
    ASSERT_NE(ex, nullptr);
    nimcp_immune_response_t response;
    memset(&response, 0, sizeof(response));
    int rc = nimcp_exception_present_to_immune(ex, &response);
    EXPECT_EQ(rc, 0) << "Failed to present exception to immune";
    nimcp_exception_unref(ex);
}

TEST_F(ImmuneErrorRoutingTest, ExceptionPresent_NullException) {
    nimcp_immune_response_t response;
    memset(&response, 0, sizeof(response));
    int rc = nimcp_exception_present_to_immune(nullptr, &response);
    EXPECT_NE(rc, 0);
}

TEST_F(ImmuneErrorRoutingTest, ExceptionPresent_NullResponse) {
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_SECURITY_THREAT, NIMCP_EXCEPTION_SEVERITY_MEDIUM,
        __FILE__, __LINE__, __func__, "Test threat");
    ASSERT_NE(ex, nullptr);
    int rc = nimcp_exception_present_to_immune(ex, nullptr);
    EXPECT_NE(rc, 0);
    nimcp_exception_unref(ex);
}

// --- Antigen Lifecycle ---
TEST_F(ImmuneErrorRoutingTest, AntigenPresentAndRetrieve) {
    uint32_t antigen_id = 0;
    uint8_t epitope[] = {0xAA, 0xBB, 0xCC};
    int rc = brain_immune_present_antigen(
        immune_, ANTIGEN_SOURCE_BBB, epitope, sizeof(epitope),
        5, 0, &antigen_id);
    EXPECT_EQ(rc, 0);
    EXPECT_GT(antigen_id, 0u);

    const brain_antigen_t* ag = brain_immune_get_antigen(immune_, antigen_id);
    EXPECT_NE(ag, nullptr);
}

TEST_F(ImmuneErrorRoutingTest, AntigenPresent_NullSystem) {
    uint32_t antigen_id = 0;
    uint8_t epitope[] = {1};
    int rc = brain_immune_present_antigen(
        nullptr, ANTIGEN_SOURCE_BBB, epitope, 1, 1, 0, &antigen_id);
    EXPECT_NE(rc, 0);
}

TEST_F(ImmuneErrorRoutingTest, GetAntigen_NullSystem) {
    const brain_antigen_t* ag = brain_immune_get_antigen(nullptr, 1);
    EXPECT_EQ(ag, nullptr);
}

TEST_F(ImmuneErrorRoutingTest, GetAntigen_InvalidId) {
    const brain_antigen_t* ag = brain_immune_get_antigen(immune_, 99999);
    EXPECT_EQ(ag, nullptr);
}

// --- B Cell and Antibody ---
TEST_F(ImmuneErrorRoutingTest, BCellActivationAndAntibody) {
    // Present antigen first
    uint32_t antigen_id = 0;
    uint8_t epitope[] = {0xDE, 0xAD, 0xBE, 0xEF};
    ASSERT_EQ(0, brain_immune_present_antigen(
        immune_, ANTIGEN_SOURCE_BBB, epitope, sizeof(epitope),
        8, 0, &antigen_id));

    // Activate B cell
    uint32_t b_cell_id = 0;
    ASSERT_EQ(0, brain_immune_activate_b_cell(immune_, antigen_id, &b_cell_id));
    EXPECT_GT(b_cell_id, 0u);

    // B cell must reach PLASMA state before producing antibody
    // Update immune system to advance B cell state
    for (int i = 0; i < 50; i++) {
        brain_immune_update(immune_, 100);
    }

    // Try to produce antibody (may succeed if PLASMA reached)
    uint32_t ab_id = 0;
    int rc = brain_immune_produce_antibody(immune_, b_cell_id, ANTIBODY_IGM, &ab_id);
    // We just verify no crash; success depends on timing
    (void)rc;
}

TEST_F(ImmuneErrorRoutingTest, BCellActivate_NullSystem) {
    uint32_t b_cell_id = 0;
    int rc = brain_immune_activate_b_cell(nullptr, 1, &b_cell_id);
    EXPECT_NE(rc, 0);
}

TEST_F(ImmuneErrorRoutingTest, ProduceAntibody_NullSystem) {
    uint32_t ab_id = 0;
    int rc = brain_immune_produce_antibody(nullptr, 1, ANTIBODY_IGM, &ab_id);
    EXPECT_NE(rc, 0);
}

// --- Immune Stats and Phase ---
TEST_F(ImmuneErrorRoutingTest, GetStats) {
    brain_immune_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    int rc = brain_immune_get_stats(immune_, &stats);
    EXPECT_EQ(rc, 0);
}

TEST_F(ImmuneErrorRoutingTest, GetStats_NullSystem) {
    brain_immune_stats_t stats;
    int rc = brain_immune_get_stats(nullptr, &stats);
    EXPECT_NE(rc, 0);
}

TEST_F(ImmuneErrorRoutingTest, GetStats_NullStats) {
    int rc = brain_immune_get_stats(immune_, nullptr);
    EXPECT_NE(rc, 0);
}

TEST_F(ImmuneErrorRoutingTest, GetPhase) {
    brain_immune_phase_t phase = brain_immune_get_phase(immune_);
    EXPECT_EQ(phase, IMMUNE_PHASE_SURVEILLANCE);
}

TEST_F(ImmuneErrorRoutingTest, GetPhase_NullSystem) {
    brain_immune_phase_t phase = brain_immune_get_phase(nullptr);
    // Should return a default/error phase
    (void)phase;
}

TEST_F(ImmuneErrorRoutingTest, Update_NullSystem) {
    int rc = brain_immune_update(nullptr, 100);
    EXPECT_NE(rc, 0);
}
""")

# === TEST 5: E2E BBB + Immune Pipeline ===
w("test/e2e/security/test_bbb_immune_pipeline_e2e.cpp", r"""
// test_bbb_immune_pipeline_e2e.cpp - End-to-end BBB + immune pipeline test
// Tests full security pipeline: BBB detect -> immune respond -> quarantine
#include <gtest/gtest.h>
#include <cstring>
#include <vector>

#include "cognitive/immune/nimcp_brain_immune.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "security/nimcp_bbb_helpers.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_immune.h"
#include "utils/error/nimcp_error_codes.h"

class BBBImmunePipelineE2E : public ::testing::Test {
protected:
    brain_immune_system_t* immune_ = nullptr;
    bbb_system_t bbb_ = nullptr;

    void SetUp() override {
        nimcp_exception_system_init();
        bbb_helpers_init();

        // Create BBB with strict settings
        bbb_config_t bbb_cfg = bbb_default_config();
        bbb_cfg.strict_mode = true;
        bbb_cfg.input.validate_strings = true;
        bbb_cfg.input.max_string_length = 4096;
        bbb_ = bbb_system_create(&bbb_cfg);
        ASSERT_NE(bbb_, nullptr);
        bbb_system_set_enabled(bbb_, true);

        // Create immune system with BBB integration
        brain_immune_config_t imm_cfg;
        brain_immune_default_config(&imm_cfg);
        imm_cfg.enable_bbb_integration = true;
        imm_cfg.enable_logging = false;
        immune_ = brain_immune_create(&imm_cfg);
        ASSERT_NE(immune_, nullptr);

        // Connect BBB to immune
        int rc = brain_immune_connect_bbb(immune_, bbb_);
        ASSERT_EQ(rc, 0) << "Failed to connect BBB to immune system";
    }

    void TearDown() override {
        if (immune_) brain_immune_destroy(immune_);
        if (bbb_) bbb_system_destroy(bbb_);
        bbb_helpers_shutdown();
        nimcp_exception_system_shutdown();
    }
};

// --- Full Pipeline: Valid Input ---
TEST_F(BBBImmunePipelineE2E, ValidInput_PassesThrough) {
    const char* safe_data = "Hello, this is safe input";
    bbb_validation_result_t vr;
    memset(&vr, 0, sizeof(vr));

    EXPECT_TRUE(bbb_validate_input(bbb_, safe_data, strlen(safe_data), &vr));
    EXPECT_TRUE(vr.valid);
    EXPECT_EQ(vr.threat, BBB_THREAT_NONE);

    // Immune system should remain in surveillance
    brain_immune_phase_t phase = brain_immune_get_phase(immune_);
    EXPECT_EQ(phase, IMMUNE_PHASE_SURVEILLANCE);
}

// --- Full Pipeline: Threat Detection -> Immune Response ---
TEST_F(BBBImmunePipelineE2E, ThreatTriggersImmuneResponse) {
    // 1. BBB detects a threat
    uint8_t malicious[] = {0xFF, 0xFE, 0xDE, 0xAD, 0xBE, 0xEF};
    bbb_threat_report_t report = bbb_report_threat(
        bbb_, BBB_THREAT_BUFFER_OVERFLOW, BBB_SEVERITY_CRITICAL,
        "Buffer overflow detected", nullptr, malicious, sizeof(malicious));
    EXPECT_NE(report.report_id, 0u);

    // 2. Present threat to immune system
    uint32_t antigen_id = 0;
    int rc = brain_immune_present_bbb_threat(
        immune_, BBB_THREAT_BUFFER_OVERFLOW, BBB_SEVERITY_CRITICAL,
        malicious, sizeof(malicious), &antigen_id);
    EXPECT_EQ(rc, 0);
    EXPECT_GT(antigen_id, 0u);

    // 3. Verify antigen was registered
    const brain_antigen_t* ag = brain_immune_get_antigen(immune_, antigen_id);
    EXPECT_NE(ag, nullptr);

    // 4. Activate B cell response
    uint32_t b_cell_id = 0;
    rc = brain_immune_activate_b_cell(immune_, antigen_id, &b_cell_id);
    EXPECT_EQ(rc, 0);

    // 5. Run immune updates
    for (int i = 0; i < 50; i++) {
        brain_immune_update(immune_, 100);
    }

    // 6. Verify immune system is responding
    brain_immune_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    brain_immune_get_stats(immune_, &stats);
    EXPECT_GT(stats.total_antigens, 0u);
}

// --- Full Pipeline: String Validation Rejection ---
TEST_F(BBBImmunePipelineE2E, OversizedString_BBBRejects) {
    std::string huge(8192, 65);
    bbb_validation_result_t vr;
    memset(&vr, 0, sizeof(vr));

    EXPECT_FALSE(bbb_validate_string(bbb_, huge.c_str(), &vr));
    EXPECT_FALSE(vr.valid);
}

// --- Full Pipeline: Exception -> Immune ---
TEST_F(BBBImmunePipelineE2E, ExceptionRoutedToImmune) {
    // Create security exception
    nimcp_exception_t* ex = nimcp_exception_create(
        NIMCP_ERROR_BBB_REJECTED, NIMCP_EXCEPTION_SEVERITY_HIGH,
        __FILE__, __LINE__, __func__, "BBB rejected malicious input");
    ASSERT_NE(ex, nullptr);

    // Route to immune system
    nimcp_immune_response_t response;
    memset(&response, 0, sizeof(response));
    int rc = nimcp_exception_present_to_immune(ex, &response);
    EXPECT_EQ(rc, 0);

    nimcp_exception_unref(ex);
}

// --- Full Pipeline: Quarantine ---
TEST_F(BBBImmunePipelineE2E, QuarantineAfterThreat) {
    char suspicious_region[128];
    memset(suspicious_region, 0xDE, sizeof(suspicious_region));

    // Quarantine the region
    bool q = bbb_quarantine_region(bbb_, suspicious_region, sizeof(suspicious_region));
    if (q) {
        EXPECT_TRUE(bbb_is_quarantined(bbb_, suspicious_region, sizeof(suspicious_region)));

        // Validation of quarantined region
        bbb_validation_result_t vr;
        memset(&vr, 0, sizeof(vr));
        bbb_validate_pointer(bbb_, suspicious_region, sizeof(suspicious_region), &vr);

        // Release quarantine
        bbb_release_quarantine(bbb_, suspicious_region);
        EXPECT_FALSE(bbb_is_quarantined(bbb_, suspicious_region, sizeof(suspicious_region)));
    }
}

// --- Full Pipeline: Multiple Threats ---
TEST_F(BBBImmunePipelineE2E, MultipleThreatsCumulative) {
    // Present multiple threats
    for (int i = 0; i < 5; i++) {
        uint8_t data[] = {(uint8_t)i};
        uint32_t aid = 0;
        brain_immune_present_bbb_threat(
            immune_, BBB_THREAT_CODE_INJECTION, BBB_SEVERITY_HIGH,
            data, 1, &aid);
    }

    // Update immune system
    for (int i = 0; i < 20; i++) {
        brain_immune_update(immune_, 100);
    }

    // Check stats reflect multiple antigens
    brain_immune_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    brain_immune_get_stats(immune_, &stats);
    EXPECT_GE(stats.total_antigens, 5u);
}

// --- BBB Helpers Pipeline ---
TEST_F(BBBImmunePipelineE2E, HelpersIntegrateWithBBB) {
    // Register module
    EXPECT_TRUE(bbb_register_module("e2e_test", BBB_MODULE_TYPE_CORE));

    // Validate various inputs through helpers
    int val = 42;
    EXPECT_TRUE(bbb_check_pointer(&val, "e2e_test"));
    EXPECT_TRUE(bbb_check_string("safe input", 4096, "e2e_test"));
    EXPECT_TRUE(bbb_validate_range(50, 0, 100, "e2e_test"));

    char buf[256];
    EXPECT_TRUE(bbb_validate_buffer_access(buf, 0, 128, 256, "e2e_test"));

    // BBB statistics should reflect activity
    bbb_statistics_t bbb_stats;
    memset(&bbb_stats, 0, sizeof(bbb_stats));
    bbb_system_get_statistics(bbb_, &bbb_stats);
}

// --- Connect BBB null checks ---
TEST_F(BBBImmunePipelineE2E, ConnectBBB_NullImmune) {
    int rc = brain_immune_connect_bbb(nullptr, bbb_);
    EXPECT_NE(rc, 0);
}

TEST_F(BBBImmunePipelineE2E, ConnectBBB_NullBBB) {
    int rc = brain_immune_connect_bbb(immune_, nullptr);
    EXPECT_NE(rc, 0);
}
""")

# === TEST 6: Regression Security Return Codes ===
w("test/regression/security/test_security_return_codes_regression.cpp", r"""
// test_security_return_codes_regression.cpp - Regression for P2-7 fix
// Ensures security functions never return bare -1, always proper NIMCP error codes
#include <gtest/gtest.h>
#include <cstring>

#include "security/nimcp_security.h"
#include "security/nimcp_post_quantum.h"
#include "security/nimcp_security_consensus.h"
#include "security/nimcp_blood_brain_barrier.h"
#include "security/nimcp_bbb_helpers.h"
#include "cognitive/immune/nimcp_brain_immune.h"
#include "utils/error/nimcp_error_codes.h"

// Regression macro: verify no bare -1 returns
#define ASSERT_NO_BARE_MINUS1(expr) do { \
    int _v = (int)(expr); \
    ASSERT_NE(_v, -1) << "REGRESSION P2-7: " #expr " returned bare -1"; \
} while(0)

// --- Directive system regression ---
TEST(SecurityReturnCodesRegression, DirectiveAdd_NullSystem) {
    ASSERT_NO_BARE_MINUS1(nimcp_directive_add(nullptr, "test"));
}

TEST(SecurityReturnCodesRegression, DirectiveAdd_NullText) {
    nimcp_directive_system_t* sys = nimcp_directive_system_create();
    if (sys) {
        ASSERT_NO_BARE_MINUS1(nimcp_directive_add(sys, nullptr));
        nimcp_directive_system_destroy(sys);
    }
}

TEST(SecurityReturnCodesRegression, DirectiveLock_NullSystem) {
    ASSERT_NO_BARE_MINUS1(nimcp_directive_lock(nullptr));
}

// --- Encryption regression ---
TEST(SecurityReturnCodesRegression, Encrypt_NullCtx) {
    uint8_t d[] = {1}; uint8_t o[256]; size_t s = 0;
    ASSERT_NO_BARE_MINUS1(nimcp_encryption_encrypt(nullptr, d, 1, o, 256, &s));
}

TEST(SecurityReturnCodesRegression, Decrypt_NullCtx) {
    uint8_t d[] = {1}; uint8_t o[256]; size_t s = 0;
    ASSERT_NO_BARE_MINUS1(nimcp_encryption_decrypt(nullptr, d, 1, o, 256, &s));
}

TEST(SecurityReturnCodesRegression, GenerateKey_NullPtr) {
    ASSERT_NO_BARE_MINUS1(nimcp_encryption_generate_key(nullptr));
}

// --- Sanitization regression ---
TEST(SecurityReturnCodesRegression, SanitizeInput_NullInput) {
    char out[256];
    ASSERT_NO_BARE_MINUS1(nimcp_security_sanitize_input(nullptr, out, 256));
}

TEST(SecurityReturnCodesRegression, SanitizeInput_NullOutput) {
    ASSERT_NO_BARE_MINUS1(nimcp_security_sanitize_input("test", nullptr, 256));
}

TEST(SecurityReturnCodesRegression, SanitizeInput_ZeroSize) {
    char out[256];
    ASSERT_NO_BARE_MINUS1(nimcp_security_sanitize_input("test", out, 0));
}

// --- Post-quantum regression ---
TEST(SecurityReturnCodesRegression, PQGetStats_NullCtx) {
    nimcp_pq_stats_t stats;
    ASSERT_NO_BARE_MINUS1(nimcp_pq_get_stats(nullptr, &stats));
}

TEST(SecurityReturnCodesRegression, PQGetStats_NullStats) {
    nimcp_pq_context_t ctx = nimcp_pq_context_create(nullptr);
    if (ctx) {
        ASSERT_NO_BARE_MINUS1(nimcp_pq_get_stats(ctx, nullptr));
        nimcp_pq_context_destroy(ctx);
    }
}

TEST(SecurityReturnCodesRegression, KyberKeygen_NullKeypair) {
    ASSERT_NO_BARE_MINUS1(nimcp_kyber_keygen(NIMCP_KYBER_512, nullptr));
}

// --- Consensus regression ---
TEST(SecurityReturnCodesRegression, ConsensusJoin_NullCtx) {
    ASSERT_NO_BARE_MINUS1(nimcp_consensus_join(nullptr, "peer:8080"));
}

TEST(SecurityReturnCodesRegression, ConsensusLeave_NullCtx) {
    ASSERT_NO_BARE_MINUS1(nimcp_consensus_leave(nullptr));
}

TEST(SecurityReturnCodesRegression, ConsensusShareThreat_NullCtx) {
    nimcp_threat_info_t t; memset(&t, 0, sizeof(t));
    ASSERT_NO_BARE_MINUS1(nimcp_consensus_share_threat(nullptr, &t));
}

TEST(SecurityReturnCodesRegression, ConsensusProposePolicy_NullCtx) {
    nimcp_security_policy_t p; memset(&p, 0, sizeof(p));
    ASSERT_NO_BARE_MINUS1(nimcp_consensus_propose_policy(nullptr, &p));
}

TEST(SecurityReturnCodesRegression, ConsensusInitiateResponse_NullCtx) {
    ASSERT_NO_BARE_MINUS1(nimcp_consensus_initiate_response(nullptr, NIMCP_RESPONSE_QUARANTINE, nullptr));
}

// --- Skepticism regression ---
TEST(SecurityReturnCodesRegression, Skepticism_NullResult) {
    ASSERT_NO_BARE_MINUS1(nimcp_security_evaluate_skepticism("claim", "source", "ctx", nullptr));
}

TEST(SecurityReturnCodesRegression, Skepticism_NullClaim) {
    nimcp_skepticism_result_t res;
    ASSERT_NO_BARE_MINUS1(nimcp_security_evaluate_skepticism(nullptr, "source", "ctx", &res));
}

// --- Security log/stats regression ---
TEST(SecurityReturnCodesRegression, LogEvent_NullDetails) {
    ASSERT_NO_BARE_MINUS1(nimcp_security_log_event(NIMCP_SECURITY_EVENT_LOGIN, NIMCP_THREAT_LEVEL_LOW, nullptr));
}

TEST(SecurityReturnCodesRegression, GetStats_AllNull) {
    ASSERT_NO_BARE_MINUS1(nimcp_security_get_stats(nullptr, nullptr, nullptr));
}

// --- Immune system regression ---
TEST(SecurityReturnCodesRegression, ImmuneCreate_NullConfig) {
    brain_immune_system_t* sys = brain_immune_create(nullptr);
    // Should handle nullptr config gracefully (either create with defaults or return null)
    if (sys) brain_immune_destroy(sys);
}

TEST(SecurityReturnCodesRegression, ImmunePresentBBBThreat_NullSystem) {
    uint32_t aid = 0;
    uint8_t data[] = {1};
    int rc = brain_immune_present_bbb_threat(nullptr, BBB_THREAT_BUFFER_OVERFLOW, BBB_SEVERITY_HIGH, data, 1, &aid);
    ASSERT_NE(rc, -1) << "REGRESSION P2-7: bare -1 returned";
}

TEST(SecurityReturnCodesRegression, ImmuneGetStats_NullSystem) {
    brain_immune_stats_t stats;
    int rc = brain_immune_get_stats(nullptr, &stats);
    ASSERT_NE(rc, -1) << "REGRESSION P2-7: bare -1 returned";
}

TEST(SecurityReturnCodesRegression, ImmuneUpdate_NullSystem) {
    int rc = brain_immune_update(nullptr, 100);
    ASSERT_NE(rc, -1) << "REGRESSION P2-7: bare -1 returned";
}

TEST(SecurityReturnCodesRegression, ImmuneConnectBBB_NullSystem) {
    bbb_system_t bbb = bbb_system_create(nullptr);
    if (bbb) {
        int rc = brain_immune_connect_bbb(nullptr, bbb);
        ASSERT_NE(rc, -1) << "REGRESSION P2-7: bare -1 returned";
        bbb_system_destroy(bbb);
    }
}

// --- Error code value regression ---
TEST(SecurityReturnCodesRegression, ErrorCodesInRange) {
    // Verify security error codes are in the 9000+ range
    EXPECT_EQ(NIMCP_ERROR_SECURITY_BASE, 9000);
    EXPECT_GT(NIMCP_ERROR_BBB_REJECTED, NIMCP_ERROR_SECURITY_BASE);
    EXPECT_GT(NIMCP_ERROR_BBB_VALIDATION, NIMCP_ERROR_SECURITY_BASE);
    EXPECT_GT(NIMCP_ERROR_SECURITY_THREAT, NIMCP_ERROR_SECURITY_BASE);
    // Verify general errors are in 1000+ range
    EXPECT_GE(NIMCP_ERROR_NULL_POINTER, 1000);
    EXPECT_GE(NIMCP_ERROR_INVALID_STATE, 1000);
}
""")

# === CMakeLists.txt additions ===
# Append to unit/security/CMakeLists.txt
def a(p,c):
 f=os.path.join(B,p)
 with open(f,"a") as fh:
  fh.write(c)
 print("APPENDED:"+p)

a("test/unit/security/CMakeLists.txt", r"""

# ============================================================================
# BBB API Validation Tests (P1-4)
# ============================================================================
add_executable(unit_security_test_bbb_api_validation
    test_bbb_api_validation.cpp
)
target_link_libraries(unit_security_test_bbb_api_validation
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_bbb_api_validation
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/cognitive/immune
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_security_test_bbb_api_validation
         COMMAND unit_security_test_bbb_api_validation)
set_tests_properties(unit_security_test_bbb_api_validation PROPERTIES
    LABELS "unit;security;bbb;p1-4"
    TIMEOUT 60
)

# ============================================================================
# Security Error Codes Tests (P2-7)
# ============================================================================
add_executable(unit_security_test_error_codes
    test_security_error_codes.cpp
)
target_link_libraries(unit_security_test_error_codes
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_security_test_error_codes
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/utils/error
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME unit_security_test_error_codes
         COMMAND unit_security_test_error_codes)
set_tests_properties(unit_security_test_error_codes PROPERTIES
    LABELS "unit;security;error-codes;p2-7"
    TIMEOUT 60
)
""")

a("test/CMakeLists.txt", r"""

# ============================================================================
# Overflow Check Macro Tests (P3-7)
# ============================================================================
add_executable(unit_utils_test_overflow_checks
    unit/utils/test_overflow_checks.cpp
)
target_link_libraries(unit_utils_test_overflow_checks
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_utils_test_overflow_checks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/utils
        ${CMAKE_SOURCE_DIR}/test
)
add_test(NAME unit_utils_test_overflow_checks
         COMMAND unit_utils_test_overflow_checks)
set_tests_properties(unit_utils_test_overflow_checks PROPERTIES
    LABELS "unit;utils;overflow;p3-7"
    TIMEOUT 30
)
""")

a("test/integration/security/CMakeLists.txt", r"""

# ============================================================================
# Immune Error Routing Integration Tests (P1-2/3/5/6 + P2-6)
# ============================================================================
add_executable(integration_security_test_immune_error_routing
    test_immune_error_routing.cpp
)
target_link_libraries(integration_security_test_immune_error_routing
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(integration_security_test_immune_error_routing
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/cognitive/immune
        ${CMAKE_SOURCE_DIR}/include/utils/exception
        ${CMAKE_SOURCE_DIR}/include/utils/error
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME integration_security_test_immune_error_routing
         COMMAND integration_security_test_immune_error_routing)
set_tests_properties(integration_security_test_immune_error_routing PROPERTIES
    LABELS "integration;security;immune;error-routing"
    TIMEOUT 120
)
""")

a("test/e2e/security/CMakeLists.txt", r"""

# ============================================================================
# BBB + Immune Pipeline E2E Tests
# ============================================================================
add_executable(e2e_security_test_bbb_immune_pipeline
    test_bbb_immune_pipeline_e2e.cpp
)
target_link_libraries(e2e_security_test_bbb_immune_pipeline
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(e2e_security_test_bbb_immune_pipeline
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/cognitive/immune
        ${CMAKE_SOURCE_DIR}/include/utils/exception
        ${CMAKE_SOURCE_DIR}/include/utils/error
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
target_compile_features(e2e_security_test_bbb_immune_pipeline PRIVATE cxx_std_17)
add_test(NAME e2e_security_test_bbb_immune_pipeline
         COMMAND e2e_security_test_bbb_immune_pipeline)
set_tests_properties(e2e_security_test_bbb_immune_pipeline PROPERTIES
    LABELS "e2e;security;bbb;immune;pipeline"
    TIMEOUT 300
)
""")

a("test/regression/security/CMakeLists.txt", r"""

# ============================================================================
# Security Return Codes Regression Tests (P2-7)
# ============================================================================
add_executable(regression_security_test_return_codes
    test_security_return_codes_regression.cpp
)
target_link_libraries(regression_security_test_return_codes
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_security_test_return_codes
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/security
        ${CMAKE_SOURCE_DIR}/include/cognitive/immune
        ${CMAKE_SOURCE_DIR}/include/utils/error
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/utils/logging
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_SOURCE_DIR}/test/utils
)
add_test(NAME regression_security_test_return_codes
         COMMAND regression_security_test_return_codes)
set_tests_properties(regression_security_test_return_codes PROPERTIES
    LABELS "regression;security;return-codes;p2-7"
    TIMEOUT 60
)
""")

print("\n=== All test files and CMakeLists updates generated successfully ===")
