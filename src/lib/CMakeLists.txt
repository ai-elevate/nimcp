# Core library sources
set(NIMCP_CORE_SOURCES
    nimcp_neuralnet.c
    nimcp_p2pnode.c
    nimcp_protocol.c
    nimcp_events.c
    nimcp_adaptive.c
    nimcp_brain.c
    nimcp_ethics.c
    nimcp_curiosity.c
    nimcp_knowledge.c
    nimcp_replication.c                              # Distributed brain replication
    nimcp_dataio.c                                   # External dataset I/O
    nimcp_stream.c                                   # Continuous input streaming
    nimcp_salience.c                                 # Attention/salience evaluation
    nimcp_introspection.c                            # Brain state introspection
    nimcp_consolidation.c                            # Memory consolidation (sleep)
    nimcp_attention.c                                # Biology-based multihead attention (cortical columns + thalamic gating)
    nimcp_security.c                                 # Security framework (directives, encryption, validation)
    nimcp_neuromodulators.c                          # Neuromodulator system (dopamine, serotonin, ACh, NE)
    nimcp_bcm.c                                      # BCM (Bienenstock-Cooper-Munro) sliding threshold plasticity
    nimcp_astrocytes.c                               # Astrocyte glial cells (calcium dynamics, synaptic modulation, homeostasis)
    nimcp_oligodendrocytes.c                         # Oligodendrocyte glial cells (axon myelination, conduction velocity)
    nimcp_microglia.c                                # Microglia glial cells (synaptic surveillance, pruning)
    nimcp_glial_integration.c                        # Glial-neuron integration layer (tripartite synapses, myelination, pruning)
    utils/nimcp_memory.c                             # Memory tracking and allocation
    utils/nimcp_thread.c                             # Thread utilities and wrappers
    utils/nimcp_thread_pool.c                        # Thread pool for batch processing
    utils/nimcp_vector.c                             # Vector math operations
    utils/nimcp_time.c                               # Time and timing utilities
    utils/nimcp_hash_table.c                         # Generic hash table implementation
    logging/nimcp_logging.c                          # Logging system
    utils/nimcp_btree.c                              # B-tree implementation
    utils/nimcp_queue.c                              # Queue implementation
    utils/nimcp_graph.c                              # Graph data structure
    utils/nimcp_min_heap.c                           # Min-heap for O(log n) priority queue operations
    utils/nimcp_serialization.c                      # Serialization utilities
    utils/nimcp_validate.c                           # Input validation
    utils/nimcp_queue_manager.c                      # Queue manager with thread pool
    utils/nimcp_json.c                               # JSON wrapper for jansson
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_module.c   # Add the Python module source
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_types.c    # Add the Python types
)

# Core library
add_library(nimcp_core SHARED
    ${NIMCP_CORE_SOURCES}
)

target_include_directories(nimcp_core
    PUBLIC
        $<BUILD_INTERFACE:${NIMCP_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include/nimcp>
    PRIVATE
        ${NIMCP_LIB_DIR}
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(nimcp_core
    PUBLIC
        Python3::Python
        lz4
        jansson
)

# Enable Position Independent Code and set library properties
set_target_properties(nimcp_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ENABLE_EXPORTS ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 2
    OUTPUT_NAME nimcp_core
)

# Installation rules
install(TARGETS nimcp_core
    EXPORT nimcp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
