# ============================================================================
# Platform Tier Unit Tests
# ============================================================================

# Test 1: Platform Tier Detection
add_test_binary(
    test_platform_tier
    ${CMAKE_CURRENT_SOURCE_DIR}/test_platform_tier.cpp
    unit
)
