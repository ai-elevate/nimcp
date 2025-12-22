# ============================================================================
# Hemispheric Brain Unit Tests
# ============================================================================
# Tests for bilateral brain architecture with two hemispheres and corpus callosum
#
# Test Coverage:
# - Hemisphere lifecycle (create, destroy)
# - Hemisphere processing (update, infer, train)
# - Corpus callosum (communication, bandwidth)
# - Hemispheric brain (coordination, modes)
# - Lateralization (dominance, plasticity)
# - Split-brain mode

# Test: Hemispheric Brain Comprehensive Tests
add_test_binary(
    unit_core_brain_hemispheric_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test_hemispheric_brain.cpp
    unit
)
