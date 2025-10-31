# Define test sources
set(NIMCP_TEST_SOURCES
    test_module.cpp
    test_neuralnet_create.cpp
    test_neuralnet_learning.cpp
    test_p2pnode.cpp
    test_protocol.cpp
    test_replication.cpp        # Brain replication tests
    test_dataio.cpp             # Data I/O tests
    test_stream.cpp             # Streaming API tests
    test_salience.cpp           # Salience/attention tests
    test_introspection.cpp      # Introspection API tests
    test_consolidation.cpp      # Memory consolidation tests
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
