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
    utils/nimcp_memory.c                             # Memory tracking and allocation
    utils/nimcp_thread.c                             # Thread utilities and wrappers
    utils/nimcp_thread_pool.c                        # Thread pool for batch processing
    utils/nimcp_vector.c                             # Vector math operations
    utils/nimcp_time.c                               # Time and timing utilities
    utils/nimcp_hash_table.c                         # Generic hash table implementation
    logging/nimcp_logging.c                          # Logging system
    # utils/nimcp_queue.c                            # Queue (needs nimcp_common.h)
    # utils/nimcp_queue_manager.c                    # Queue manager (needs dependencies)
    # utils/nimcp_btree.c                            # B-tree (needs logging macros)
    # utils/nimcp_graph.c                            # Graph (needs dependencies)
    # utils/nimcp_json.c                             # JSON (needs logging macros)
    # utils/nimcp_serialization.c                    # Serialization (needs common)
    # utils/nimcp_validate.c                         # Validate (needs logging macros)
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
