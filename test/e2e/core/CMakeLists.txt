# ============================================================================
# NIMCP Core Layer End-to-End Tests
# ============================================================================
# WHAT: Build configuration for core layer E2E tests
# WHY:  Test complete core pipelines including neural substrate
# HOW:  Include subdirectories for each core module
# ============================================================================

message(STATUS "")
message(STATUS "-- Configuring Core Layer E2E Tests")

# Include Neural Substrate E2E tests
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/neural_substrate/CMakeLists.txt)
    add_subdirectory(neural_substrate)
endif()
