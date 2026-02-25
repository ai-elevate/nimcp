
# =============================================================================
# Brain KG Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/brain/CMakeLists.txt)
    message(STATUS "Adding Brain KG Regression tests...")
    add_subdirectory(brain)
endif()

# =============================================================================
# Exception Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/exception/CMakeLists.txt)
    message(STATUS "Adding Exception Regression tests...")
    add_subdirectory(exception)
endif()

# =============================================================================
# Bio-Async Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bio_async/CMakeLists.txt)
    message(STATUS "Adding Bio-Async Regression tests...")
    add_subdirectory(bio_async)
endif()

# =============================================================================
# Fault Tolerance Regression Tests (Phase 8)
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fault_tolerance/CMakeLists.txt)
    message(STATUS "Adding Fault Tolerance Regression tests...")
    add_subdirectory(fault_tolerance)
endif()

# =============================================================================
# Statistics Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/statistics/CMakeLists.txt)
    message(STATUS "Adding Statistics Regression tests...")
    add_subdirectory(statistics)
endif()

# =============================================================================
# GPU Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/CMakeLists.txt)
    message(STATUS "Adding GPU Regression tests...")
    add_subdirectory(gpu)
endif()

# =============================================================================
# Genius Profiles Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/genius/CMakeLists.txt)
    message(STATUS "Adding Genius Profiles Regression tests...")
    add_subdirectory(genius)
endif()

# =============================================================================
# Cross-Module Walkthrough Fixes Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cross_module/CMakeLists.txt)
    message(STATUS "Adding Cross-Module Walkthrough Fixes Regression tests...")
    add_subdirectory(cross_module)
endif()

# =============================================================================
# Neuralnet Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core/neuralnet/CMakeLists.txt)
    message(STATUS "Adding Neuralnet Regression tests...")
    add_subdirectory(core/neuralnet)
endif()

# =============================================================================
# Reasoning-Generation Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/reasoning_generation/CMakeLists.txt)
    message(STATUS "Adding Reasoning-Generation Regression tests...")
    add_subdirectory(reasoning_generation)
endif()

# =============================================================================
# Training Integration Regression Tests
# =============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/training/CMakeLists.txt)
    message(STATUS "Adding Training Integration Regression tests...")
    add_subdirectory(training)
endif()
