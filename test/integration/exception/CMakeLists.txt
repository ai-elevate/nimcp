# =============================================================================
# Exception Integration Tests
# =============================================================================

# Signal Exception Integration Tests
add_executable(integration_exception_test_signal_exception_integration
    test_signal_exception_integration.cpp
)

target_link_libraries(integration_exception_test_signal_exception_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        pthread
        m
)

target_compile_features(integration_exception_test_signal_exception_integration PRIVATE cxx_std_17)

add_test(
    NAME integration_exception_test_signal_exception_integration
    COMMAND integration_exception_test_signal_exception_integration
)

set_tests_properties(integration_exception_test_signal_exception_integration
    PROPERTIES
    LABELS "integration;exception;signal"
    TIMEOUT 120
)

# =============================================================================
# Exception Recovery Integration Tests
# =============================================================================

add_executable(integration_exception_test_exception_recovery_integration
    test_exception_recovery_integration.cpp
)

target_link_libraries(integration_exception_test_exception_recovery_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(integration_exception_test_exception_recovery_integration PRIVATE cxx_std_17)

add_test(
    NAME integration_exception_test_exception_recovery_integration
    COMMAND integration_exception_test_exception_recovery_integration
)

set_tests_properties(integration_exception_test_exception_recovery_integration
    PROPERTIES
    LABELS "integration;exception;recovery"
    TIMEOUT 120
)

# =============================================================================
# Exception Handler Chain Integration Tests
# =============================================================================

add_executable(integration_exception_test_handler_chain_integration
    test_exception_handler_chain_integration.cpp
)

target_link_libraries(integration_exception_test_handler_chain_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(integration_exception_test_handler_chain_integration PRIVATE cxx_std_17)

add_test(
    NAME integration_exception_test_handler_chain_integration
    COMMAND integration_exception_test_handler_chain_integration
)

set_tests_properties(integration_exception_test_handler_chain_integration
    PROPERTIES
    LABELS "integration;exception;handler_chain"
    TIMEOUT 120
)

# =============================================================================
# Exception Propagation Integration Tests
# =============================================================================

add_executable(integration_exception_test_exception_propagation_integration
    test_exception_propagation_integration.cpp
)

target_link_libraries(integration_exception_test_exception_propagation_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(integration_exception_test_exception_propagation_integration PRIVATE cxx_std_17)

add_test(
    NAME integration_exception_test_exception_propagation_integration
    COMMAND integration_exception_test_exception_propagation_integration
)

set_tests_properties(integration_exception_test_exception_propagation_integration
    PROPERTIES
    LABELS "integration;exception;propagation"
    TIMEOUT 120
)

# =============================================================================
# Exception-Immune Integration Tests
# =============================================================================

add_executable(integration_exception_test_exception_immune_integration
    test_exception_immune_integration.cpp
)

target_link_libraries(integration_exception_test_exception_immune_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_compile_features(integration_exception_test_exception_immune_integration PRIVATE cxx_std_17)

add_test(
    NAME integration_exception_test_exception_immune_integration
    COMMAND integration_exception_test_exception_immune_integration
)

set_tests_properties(integration_exception_test_exception_immune_integration
    PROPERTIES
    LABELS "integration;exception;immune"
    TIMEOUT 120
)
