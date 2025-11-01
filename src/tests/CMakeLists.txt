# Define test sources
set(NIMCP_TEST_SOURCES
    test_module.cpp
    test_neuralnet_create.cpp
    test_neuralnet_learning.cpp
    test_adaptive.cpp           # Adaptive threshold spiking tests
    test_events.cpp             # Event packet integration tests
    test_p2pnode.cpp
    test_protocol.cpp
    test_replication.cpp        # Brain replication tests
    test_dataio.cpp             # Data I/O tests
    test_stream.cpp             # Streaming API tests
    test_salience.cpp           # Salience/attention tests
    test_introspection.cpp      # Introspection API tests
    test_consolidation.cpp      # Memory consolidation tests
    test_attention.cpp          # Biology-based multihead attention tests (cortical columns + thalamic gating)
    test_thread_utils.cpp       # Thread utilities and pool tests
    test_queue_utils.cpp        # Queue and queue manager tests
    test_queue_manager.cpp      # Queue manager tests
    test_logging.cpp            # Logging infrastructure tests
    test_knowledge.cpp          # Knowledge acquisition system tests
    test_curiosity.cpp          # Curiosity-driven learning tests
    test_ethics.cpp             # Ethics and Golden Rule tests
    test_time.cpp               # Time utilities tests
    test_serialization.cpp      # Binary serialization tests
    test_json.cpp               # JSON wrapper tests
    test_memory.cpp             # Memory tracking tests
    test_vector.cpp             # Vector math utilities tests
    test_hash_table.cpp         # Hash table data structure tests
    test_btree.cpp              # B-tree data structure tests
    test_validate.cpp           # Input validation utilities tests
    test_graph.cpp              # Graph data structure tests
    test_security.cpp           # Security framework tests (directives, encryption, validation)
    test_neuromodulators.cpp    # Neuromodulator system tests (dopamine, serotonin, ACh, NE)
    test_bcm.cpp                # BCM sliding threshold plasticity tests
    test_error_injection.cpp    # Error injection and failure path testing
    test_stress.cpp             # Stress and load testing
    test_performance_optimizations.cpp  # TDD tests for algorithmic optimizations (bidirectional synapses, min-heap)

    test_integration_networking.cpp  # P2P networking integration tests
    test_integration_cognitive.cpp   # Multi-module cognitive integration tests
    test_integration_e2e.cpp    # End-to-end integration tests across modules
    test_lint.cpp               # Lint and code quality tests
    test_memory_leaks.cpp       # Memory leak detection tests
)

# Create test executable
add_executable(nimcp_tests ${NIMCP_TEST_SOURCES})

target_include_directories(nimcp_tests
    PRIVATE
        ${NIMCP_INCLUDE_DIR}
        ${NIMCP_TEST_DIR}
        ${NIMCP_LIB_DIR}
        ${GTEST_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

# Add the Python module source directly to the test executable
target_sources(nimcp_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_module.c
)

target_link_libraries(nimcp_tests
    PRIVATE
        nimcp_core
        GTest::GTest
        GTest::Main
        Python3::Python
        pthread
)

# Enable Position Independent Code
set_target_properties(nimcp_tests PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Enable testing
enable_testing()

# Add test
add_test(NAME nimcp_unit_tests 
         COMMAND nimcp_tests)

set_tests_properties(nimcp_unit_tests PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python"
)

# Coverage configuration for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(nimcp_tests PRIVATE --coverage)
        target_link_options(nimcp_tests PRIVATE --coverage)
    endif()
endif()
