# ============================================================================
# Physics Integration Tests CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for physics module integration tests
# WHY:  Verify physics modules integrate correctly with NIMCP systems
# HOW:  Include subdirectories for each physics submodule
# ============================================================================

# Include subdirectories
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/biophysics)
    add_subdirectory(biophysics)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bridges)
    add_subdirectory(bridges)
endif()
