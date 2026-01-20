
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

