# Unit tests for STDP refactored module

add_executable(test_stdp_refactored
    test_stdp_refactored.cpp
)

target_include_directories(test_stdp_refactored PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/plasticity
    ${CMAKE_SOURCE_DIR}/include/async
    ${CMAKE_SOURCE_DIR}/include/utils
    ${CMAKE_SOURCE_DIR}/include/security
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(test_stdp_refactored
    nimcp_stdp
    nimcp_async
    nimcp_logging
    nimcp_config
    nimcp_security
    nimcp_memory
    ${GTEST_LIBRARIES}
    pthread
)

add_test(NAME STDPRefactoredTests
         COMMAND test_stdp_refactored)

# Set test properties
set_tests_properties(STDPRefactoredTests PROPERTIES
    TIMEOUT 30
    LABELS "unit;plasticity;stdp;refactored"
)
