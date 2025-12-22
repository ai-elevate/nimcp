# ============================================================================
# Subcortical Brain Structures Unit Tests
# ============================================================================
# Tests for basal ganglia action selection system
#
# Test Coverage:
# - Striatum (D1/D2 pathways, dopamine modulation)
# - Globus Pallidus (GPe/GPi, tonic inhibition)
# - Substantia Nigra (SNc dopamine, SNr output)
# - Subthalamic Nucleus (hyperdirect pathway, emergency stop)
# - Integrated Basal Ganglia (action selection, habits)

# Test: Basal Ganglia Comprehensive Tests
add_test_binary(
    unit_core_brain_subcortical_basal_ganglia_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_basal_ganglia.cpp
    unit
)

# Test: Thalamus Comprehensive Tests
add_test_binary(
    unit_core_brain_subcortical_thalamus_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_thalamus.cpp
    unit
)

# Test: Amygdala Comprehensive Tests
add_test_binary(
    unit_core_brain_subcortical_amygdala_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_amygdala.cpp
    unit
)

# Test: Amygdala-Training Bridge Tests
add_test_binary(
    unit_core_brain_subcortical_amygdala_training_bridge_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_amygdala_training_bridge.cpp
    unit
)

# Test: Amygdala-Attention Bridge Tests
add_test_binary(
    unit_core_brain_subcortical_amygdala_attention_bridge_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_amygdala_attention_bridge.cpp
    unit
)

# Test: Amygdala-Autobiographical Memory Bridge Tests
add_test_binary(
    unit_core_brain_subcortical_amygdala_autobio_bridge_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_amygdala_autobio_bridge.cpp
    unit
)

# Test: Amygdala-Stress/Wellbeing Bridge Tests
add_test_binary(
    unit_core_brain_subcortical_amygdala_stress_bridge_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_amygdala_stress_bridge.cpp
    unit
)
