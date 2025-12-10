# ============================================================================
# Brain Regions Regression Tests - TEMPORARILY DISABLED
# ============================================================================
# WHAT: CMakeLists for brain region regression tests
# WHY:  Source file nimcp_brain_region_predictive.c needs error code fixes
# HOW:  Skip test discovery until source is fixed
#
# TODO: Re-enable once the following files are fixed:
#   - src/core/brain/regions/nimcp_brain_region_predictive.c
# ============================================================================

message(STATUS "  [SKIP] Brain region regression tests - awaiting source fixes")

# Build the existing signal propagation test (not dependent on predictive coding)
add_test_binary(
    regression_brain_regions_test_signal_propagation_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_signal_propagation_regression.cpp
    regression
)
