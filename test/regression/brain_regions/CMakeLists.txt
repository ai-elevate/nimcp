# ============================================================================
# Brain Regions Regression Tests
# ============================================================================
# WHAT: CMakeLists for brain region regression tests
# WHY:  Verify backward compatibility and API stability for brain regions
# HOW:  Test region initialization, API stability, signal propagation
# ============================================================================

# Signal Propagation Test
add_test_binary(
    regression_brain_regions_test_signal_propagation_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_signal_propagation_regression.cpp
    regression
)

# Prediction Accuracy Test - DISABLED: requires brain_region_predictive_* functions not implemented
# add_test_binary(
#     regression_brain_regions_test_prediction_accuracy
#     ${CMAKE_CURRENT_SOURCE_DIR}/test_prediction_accuracy.cpp
#     regression
# )

# Cerebellum Backward Compatibility Test
add_test_binary(
    regression_brain_regions_test_cerebellum_backward_compat
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cerebellum_backward_compat.cpp
    regression
)

# Hippocampus Backward Compatibility Test
add_test_binary(
    regression_brain_regions_test_hippocampus_backward_compat
    ${CMAKE_CURRENT_SOURCE_DIR}/test_hippocampus_backward_compat.cpp
    regression
)

# Prefrontal Backward Compatibility Test
add_test_binary(
    regression_brain_regions_test_prefrontal_backward_compat
    ${CMAKE_CURRENT_SOURCE_DIR}/test_prefrontal_backward_compat.cpp
    regression
)

# Region API Stability Test
add_test_binary(
    regression_brain_regions_test_region_api_stability
    ${CMAKE_CURRENT_SOURCE_DIR}/test_region_api_stability.cpp
    regression
)

# Sensory Processing Regression Tests (Phase 6: Somatosensory, Olfactory, Gustatory)
add_test_binary(
    regression_brain_regions_test_sensory_processing_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sensory_processing_regression.cpp
    regression
)
