# Training Module Integration Tests
# Phase TM-1: Security + Memory Pool Integration

add_executable(integration_middleware_training_test_training_module_integration
    test_training_module_integration.cpp
)

target_include_directories(integration_middleware_training_test_training_module_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_middleware_training_test_training_module_integration PRIVATE cxx_std_20)

target_link_libraries(integration_middleware_training_test_training_module_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
)

add_test(NAME integration_middleware_training_test_training_module_integration
         COMMAND integration_middleware_training_test_training_module_integration)

set_tests_properties(integration_middleware_training_test_training_module_integration
    PROPERTIES
        LABELS "integration;middleware;training"
        TIMEOUT 60)

# Loss Functions Integration Tests (Phase TM-2)
add_executable(integration_middleware_training_test_loss_functions_integration
    test_loss_functions_integration.cpp
)

target_include_directories(integration_middleware_training_test_loss_functions_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_middleware_training_test_loss_functions_integration PRIVATE cxx_std_20)

target_link_libraries(integration_middleware_training_test_loss_functions_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME integration_middleware_training_test_loss_functions_integration
         COMMAND integration_middleware_training_test_loss_functions_integration)

set_tests_properties(integration_middleware_training_test_loss_functions_integration
    PROPERTIES
        LABELS "integration;middleware;training;loss"
        TIMEOUT 120)

# Optimizers Integration Tests (Phase TM-2)
add_executable(integration_middleware_training_test_optimizers_integration
    test_optimizers_integration.cpp
)

target_include_directories(integration_middleware_training_test_optimizers_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_middleware_training_test_optimizers_integration PRIVATE cxx_std_20)

target_link_libraries(integration_middleware_training_test_optimizers_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME integration_middleware_training_test_optimizers_integration
         COMMAND integration_middleware_training_test_optimizers_integration)

set_tests_properties(integration_middleware_training_test_optimizers_integration
    PROPERTIES
        LABELS "integration;middleware;training;optimizer"
        TIMEOUT 120)

# Brain Training Integration Tests (Phase TM-3)
add_executable(integration_middleware_training_test_brain_training_integration
    test_brain_training_integration.cpp
)

target_include_directories(integration_middleware_training_test_brain_training_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_middleware_training_test_brain_training_integration PRIVATE cxx_std_20)

target_link_libraries(integration_middleware_training_test_brain_training_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME integration_middleware_training_test_brain_training_integration
         COMMAND integration_middleware_training_test_brain_training_integration)

set_tests_properties(integration_middleware_training_test_brain_training_integration
    PROPERTIES
        LABELS "integration;middleware;training;brain"
        TIMEOUT 180)

# Training-Plasticity Bridge Integration Tests (Phase TPB-1)
add_executable(integration_middleware_training_test_plasticity_bridge_integration
    test_training_plasticity_bridge_integration.cpp
)

target_include_directories(integration_middleware_training_test_plasticity_bridge_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_middleware_training_test_plasticity_bridge_integration PRIVATE cxx_std_20)

target_link_libraries(integration_middleware_training_test_plasticity_bridge_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME integration_middleware_training_test_plasticity_bridge_integration
         COMMAND integration_middleware_training_test_plasticity_bridge_integration)

set_tests_properties(integration_middleware_training_test_plasticity_bridge_integration
    PROPERTIES
        LABELS "integration;middleware;training;plasticity_bridge"
        TIMEOUT 240)

# Event-Driven Plasticity Integration Tests (Phase EDP-1)
add_executable(integration_middleware_training_test_event_driven_plasticity_integration
    test_event_driven_plasticity_integration.cpp
)

target_include_directories(integration_middleware_training_test_event_driven_plasticity_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(integration_middleware_training_test_event_driven_plasticity_integration PRIVATE cxx_std_20)

target_link_libraries(integration_middleware_training_test_event_driven_plasticity_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME integration_middleware_training_test_event_driven_plasticity_integration
         COMMAND integration_middleware_training_test_event_driven_plasticity_integration)

set_tests_properties(integration_middleware_training_test_event_driven_plasticity_integration
    PROPERTIES
        LABELS "integration;middleware;training;event_driven_plasticity"
        TIMEOUT 240)
