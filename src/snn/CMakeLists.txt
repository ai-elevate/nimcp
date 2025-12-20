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

# SNN Bridge source files
set(SNN_BRIDGE_SOURCES
    bridges/nimcp_snn_visual_bridge.c
    bridges/nimcp_snn_audio_bridge.c
    bridges/nimcp_snn_speech_bridge.c
    bridges/nimcp_snn_attention_bridge.c
    bridges/nimcp_snn_memory_bridge.c
    bridges/nimcp_snn_routing_bridge.c
    bridges/nimcp_snn_stdp_bridge.c
    bridges/nimcp_snn_bcm_bridge.c
    bridges/nimcp_snn_homeostatic_bridge.c
    bridges/nimcp_snn_scaling_bridge.c
    bridges/nimcp_snn_buffer_bridge.c
    bridges/nimcp_snn_cortical_bridge.c
    bridges/nimcp_snn_hippocampus_bridge.c
    bridges/nimcp_snn_population_bridge.c
    bridges/nimcp_snn_reasoning_bridge.c
    bridges/nimcp_snn_executive_bridge.c
    bridges/nimcp_snn_thalamic_bridge.c
    bridges/nimcp_snn_prefrontal_bridge.c
)

# Create SNN static library with bridges
add_library(nimcp_snn STATIC ${SNN_SOURCES} ${SNN_BRIDGE_SOURCES})

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
