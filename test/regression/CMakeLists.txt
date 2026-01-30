
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
