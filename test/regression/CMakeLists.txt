
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

