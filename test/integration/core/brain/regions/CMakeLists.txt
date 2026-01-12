# Brain Regions Integration Tests
# Tests for various brain region integrations

# Add region subdirectories
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/broca")
    add_subdirectory(broca)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hypothalamus")
    add_subdirectory(hypothalamus)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/locus_coeruleus")
    add_subdirectory(locus_coeruleus)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hippocampus")
    add_subdirectory(hippocampus)
endif()
