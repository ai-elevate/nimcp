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

# Training Dispatch Integration Tests (C)
add_executable(integration_training_dispatch
    test_training_dispatch_integration.c
)

target_include_directories(integration_training_dispatch PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_options(integration_training_dispatch PRIVATE
    -std=c11
    -Wall
    -Wextra
)

target_link_libraries(integration_training_dispatch
    PRIVATE
        nimcp
        pthread
        m
)

add_test(NAME integration_training_dispatch
         COMMAND integration_training_dispatch)

set_tests_properties(integration_training_dispatch
    PROPERTIES
        LABELS "integration;training;dispatch;snn;lnn;cnn;adaptive"
        TIMEOUT 300)

# Bio Learning Integration Tests (Engram, Mammillary, Neuromodulators, Glial, PR Memory)
add_executable(integration_training_bio_learning
    test_bio_learning_integration.cpp
)

target_include_directories(integration_training_bio_learning PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_training_bio_learning PRIVATE cxx_std_17)

target_link_libraries(integration_training_bio_learning
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

add_test(NAME integration_training_bio_learning
         COMMAND integration_training_bio_learning)

set_tests_properties(integration_training_bio_learning
    PROPERTIES
        LABELS "integration;training;bio;engram;mammillary;neuromodulator;glial"
        TIMEOUT 300)

# Multi-Network Ensemble Training Integration Tests (LNN + CNN + Adaptive)
add_executable(integration_training_multi_network
    test_multi_network_training.cpp
)

target_include_directories(integration_training_multi_network PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_training_multi_network PRIVATE cxx_std_17)

target_link_libraries(integration_training_multi_network
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

add_test(NAME integration_training_multi_network
         COMMAND integration_training_multi_network)

set_tests_properties(integration_training_multi_network
    PROPERTIES
        LABELS "integration;training;multi-network;lnn;cnn;ensemble"
        TIMEOUT 300)

# VAE Training Integration Tests
add_executable(integration_training_vae
    test_vae_training_integration.cpp
)

target_include_directories(integration_training_vae PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_training_vae PRIVATE cxx_std_17)

target_link_libraries(integration_training_vae
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

add_test(NAME integration_training_vae
         COMMAND integration_training_vae)

set_tests_properties(integration_training_vae
    PROPERTIES
        LABELS "integration;training;vae;autoencoder;fep"
        TIMEOUT 300)

# Attention Training Integration Tests
add_executable(integration_training_attention
    test_attention_training_integration.cpp
)

target_include_directories(integration_training_attention PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_training_attention PRIVATE cxx_std_17)

target_link_libraries(integration_training_attention
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

add_test(NAME integration_training_attention
         COMMAND integration_training_attention)

set_tests_properties(integration_training_attention
    PROPERTIES
        LABELS "integration;training;attention;plasticity;multihead"
        TIMEOUT 300)

# Training Exception Handling Integration Tests
add_subdirectory(exception)
