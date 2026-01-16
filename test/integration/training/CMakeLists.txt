# CNN Training Integration Tests
# Tests for CNN training pipeline integration with other subsystems

# CNN Backprop Integration Tests
add_executable(integration_training_test_cnn_backprop_integration
    test_cnn_backprop_integration.cpp
)

# Training Symbolic Logic Hub Bridge Integration Tests
add_executable(integration_training_symbolic_logic_hub
    test_training_symbolic_logic_hub_integration.cpp
)

target_include_directories(integration_training_symbolic_logic_hub PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_training_symbolic_logic_hub PRIVATE cxx_std_17)

target_link_libraries(integration_training_symbolic_logic_hub
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

add_test(NAME integration_training_symbolic_logic_hub
         COMMAND integration_training_symbolic_logic_hub)

set_tests_properties(integration_training_symbolic_logic_hub
    PROPERTIES
        LABELS "integration;training;symbolic-logic;hub"
        TIMEOUT 300)

target_include_directories(integration_training_test_cnn_backprop_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_training_test_cnn_backprop_integration PRIVATE cxx_std_20)

target_link_libraries(integration_training_test_cnn_backprop_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME integration_training_test_cnn_backprop_integration
         COMMAND integration_training_test_cnn_backprop_integration)

set_tests_properties(integration_training_test_cnn_backprop_integration
    PROPERTIES
        LABELS "integration;training;cnn;backprop;gradient"
        TIMEOUT 300)

# CNN-Cortex Bridge Integration Tests
add_executable(integration_training_test_cnn_cortex_integration
    test_cnn_cortex_integration.cpp
)

target_include_directories(integration_training_test_cnn_cortex_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_training_test_cnn_cortex_integration PRIVATE cxx_std_20)

target_link_libraries(integration_training_test_cnn_cortex_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME integration_training_test_cnn_cortex_integration
         COMMAND integration_training_test_cnn_cortex_integration)

set_tests_properties(integration_training_test_cnn_cortex_integration
    PROPERTIES
        LABELS "integration;training;cnn;cortex;bridge"
        TIMEOUT 300)

# Training Exception Handling Integration Tests
add_subdirectory(exception)
