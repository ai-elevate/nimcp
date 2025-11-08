# Core library sources - New hierarchical structure
set(NIMCP_CORE_SOURCES
    # Public API facade (unified interface for all bindings)
    ${CMAKE_CURRENT_SOURCE_DIR}/../api/nimcp.c

    # Core brain components
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/nimcp_brain.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuralnet/nimcp_neuralnet.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_types/nimcp_neuron_types.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain_regions/nimcp_brain_regions.c

    # Multi-modal integration (Phase 8: Unified Architecture)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/integration/nimcp_multimodal_integration.c

    # Neuron models - Plugin architecture
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_models/nimcp_neuron_model.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_models/nimcp_izhikevich.c

    # Synapse computation - NIMCP 2.7 MAJOR FEATURE
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/synapse_compute/nimcp_synapse_compute.c

    # Fractal Topology - NIMCP 2.7 (scale-free networks)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/topology/nimcp_fractal_topology.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/topology/nimcp_network_builder.c

    # NLP Integration - NIMCP 2.7 (combines synapses + attention + neuromodulation)
    ${CMAKE_CURRENT_SOURCE_DIR}/../nlp/nimcp_nlp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../nlp/nimcp_spike_nlp.c

    # Cognitive functions
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/ethics/nimcp_ethics.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/knowledge/nimcp_knowledge.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/curiosity/nimcp_curiosity.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/introspection/nimcp_introspection.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/wellbeing/nimcp_wellbeing.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/consolidation/nimcp_consolidation.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/salience/nimcp_salience.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/logic/nimcp_symbolic_logic.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cognitive/nimcp_hierarchical.c

    # Plasticity mechanisms
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/adaptive/nimcp_adaptive.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/bcm/nimcp_bcm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/neuromodulators/nimcp_neuromodulators.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/neuromodulators/nimcp_neuromod_pink_noise.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/attention/nimcp_attention.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/stp/nimcp_stp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/noise/nimcp_pink_noise.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/stdp/nimcp_stdp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/eligibility/nimcp_eligibility_trace.c

    # Glial cells
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/astrocytes/nimcp_astrocytes.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/oligodendrocytes/nimcp_oligodendrocytes.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/microglia/nimcp_microglia.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/integration/nimcp_glial_integration.c

    # Networking
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/p2p/nimcp_p2pnode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/protocol/nimcp_protocol.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/events/nimcp_events.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/replication/nimcp_replication.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/distributed/nimcp_distributed_cognition.c

    # I/O
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/dataio/nimcp_dataio.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/stream/nimcp_stream.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/serialization/nimcp_serialization.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/serialization/nimcp_network_serialization.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/serialization/nimcp_encryption.c

    # Security
    ${CMAKE_CURRENT_SOURCE_DIR}/../security/nimcp_security.c

    # Utils - Memory & Thread
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/memory/nimcp_memory.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/thread/nimcp_thread.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/thread/nimcp_thread_pool.c

    # Utils - Logging
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/logging/nimcp_logging.c

    # Utils - Containers
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_vector.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_hash_table.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_btree.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_queue.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_graph.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_min_heap.c

    # Utils - Other
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/time/nimcp_time.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/validation/nimcp_validate.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/queue_manager/nimcp_queue_manager.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/json/nimcp_json.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/config/nimcp_config.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/metrics/nimcp_metrics.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cache/nimcp_cache.c

    # Utils - Spectral Analysis
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/spectral/nimcp_fft.c

    # Brain Oscillations
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain_oscillations/nimcp_brain_oscillations.c

    # Perception
    ${CMAKE_CURRENT_SOURCE_DIR}/perception/nimcp_visual_cortex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/perception/nimcp_audio_cortex.c

    # GPU Support (CPU fallback when CUDA unavailable)
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/execution/nimcp_execution_mode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/spike_event/nimcp_spike_event.c

    # GPU CUDA kernels (only compiled if CUDA is available)
    $<$<BOOL:${CUDAToolkit_FOUND}>:${CMAKE_CURRENT_SOURCE_DIR}/../gpu/neuron/nimcp_gpu_kernels.cu>
    $<$<BOOL:${CUDAToolkit_FOUND}>:${CMAKE_CURRENT_SOURCE_DIR}/../gpu/synapse_compute/nimcp_synapse_compute_gpu.cu>

    # Note: CUDA compile flags set below per-file

    # Python bindings
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_module.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_types.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_metrics_py.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_topology_py.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_pink_noise_py.c
)

# Core library
add_library(nimcp SHARED
    ${NIMCP_CORE_SOURCES}
)

# Update include directories to point to new structure
target_include_directories(nimcp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:include/nimcp>
    PRIVATE
        ${NIMCP_LIB_DIR}
        ${Python3_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/gpu
)

# Add CUDA include directories if available
if(CUDAToolkit_FOUND)
    target_include_directories(nimcp PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    # Set CUDA properties
    set_target_properties(nimcp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 14
        CUDA_ARCHITECTURES "75;80;86;89"  # Turing, Ampere, Ada Lovelace
    )

    # NVCC doesn't support some GCC flags - set per-file compile options
    # Reset incompatible flags and add only nvcc-compatible ones
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/neuron/nimcp_gpu_kernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/synapse_compute/nimcp_synapse_compute_gpu.cu
        PROPERTIES
        COMPILE_OPTIONS "--expt-relaxed-constexpr;-Xcompiler=-fPIC,-fstack-protector-strong,-Wformat,-Wformat-security,-fno-common"
        COMPILE_FLAGS ""
    )
endif()

# Add libsodium include directories if available
if(SODIUM_FOUND)
    target_include_directories(nimcp PRIVATE ${SODIUM_INCLUDE_DIR})
endif()

target_link_libraries(nimcp
    PUBLIC
        Python3::Python
        lz4
        jansson
)

# Add CUDA libraries if available
if(CUDAToolkit_FOUND)
    target_link_libraries(nimcp PUBLIC CUDA::cudart)
endif()

# Add libsodium library if available
if(SODIUM_FOUND)
    target_link_libraries(nimcp PUBLIC ${SODIUM_LIBRARY})
endif()

# Add testing API support
target_compile_definitions(nimcp PRIVATE NIMCP_TESTING)

# Explicitly add -fPIC for shared library with thread-local storage
target_compile_options(nimcp PRIVATE -fPIC)

# Enable Position Independent Code and set library properties
set_target_properties(nimcp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ENABLE_EXPORTS ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 2
    OUTPUT_NAME nimcp
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# Installation rules
install(TARGETS nimcp
    EXPORT nimcp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
