# ============================================================================
# SNN (Spiking Neural Network) Module CMakeLists.txt
# ============================================================================
# WHAT: Build configuration for the SNN module
# WHY:  Compile SNN library and link with existing infrastructure
# HOW:  Build static library, link to nimcp core and dependencies
# ============================================================================

# SNN source files
set(SNN_SOURCES
    nimcp_snn_config.c
    nimcp_snn_network.c
    nimcp_snn_encoding.c
    nimcp_snn_training.c
    nimcp_snn_bio_async.c
    nimcp_snn_immune.c
)

# Create SNN static library
add_library(nimcp_snn STATIC ${SNN_SOURCES})

# Include directories
target_include_directories(nimcp_snn
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/snn
)

# Link with core nimcp library (provides neuron_t, synapse_t, etc.)
target_link_libraries(nimcp_snn
    PUBLIC
        nimcp
    PRIVATE
        Threads::Threads
        m
)

# Compiler features
target_compile_features(nimcp_snn PRIVATE c_std_11)

# Compiler warnings
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nimcp_snn PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
endif()

# Export for use by other targets
set_target_properties(nimcp_snn PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
)

message(STATUS "Configured SNN module with ${CMAKE_C_COMPILER_ID}")
