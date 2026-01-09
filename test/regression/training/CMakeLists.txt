# CNN Training Regression Tests
# Tests for CNN backpropagation numerical stability and gradient correctness

# CNN Backprop Regression Tests
add_executable(regression_training_test_cnn_backprop_regression
    test_cnn_backprop_regression.cpp
)

target_include_directories(regression_training_test_cnn_backprop_regression PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(regression_training_test_cnn_backprop_regression PRIVATE cxx_std_20)

target_link_libraries(regression_training_test_cnn_backprop_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME regression_training_test_cnn_backprop_regression
         COMMAND regression_training_test_cnn_backprop_regression)

set_tests_properties(regression_training_test_cnn_backprop_regression
    PROPERTIES
        LABELS "regression;training;cnn;backprop;gradient;stability"
        TIMEOUT 300)

# CNN-Cortex Bridge Regression Tests
add_executable(regression_training_test_cnn_cortex_bridge_regression
    test_cnn_cortex_bridge_regression.cpp
)

target_include_directories(regression_training_test_cnn_cortex_bridge_regression PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(regression_training_test_cnn_cortex_bridge_regression PRIVATE cxx_std_20)

target_link_libraries(regression_training_test_cnn_cortex_bridge_regression
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME regression_training_test_cnn_cortex_bridge_regression
         COMMAND regression_training_test_cnn_cortex_bridge_regression)

set_tests_properties(regression_training_test_cnn_cortex_bridge_regression
    PROPERTIES
        LABELS "regression;training;cnn;cortex;bridge"
        TIMEOUT 300)

# Training Symbolic Logic Hub Bridge Regression Tests
add_executable(regression_training_symbolic_logic_hub
    test_training_symbolic_logic_hub_regression.cpp
)

target_include_directories(regression_training_symbolic_logic_hub PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(regression_training_symbolic_logic_hub PRIVATE cxx_std_17)

target_link_libraries(regression_training_symbolic_logic_hub
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

add_test(NAME regression_training_symbolic_logic_hub
         COMMAND regression_training_symbolic_logic_hub)

set_tests_properties(regression_training_symbolic_logic_hub
    PROPERTIES
        LABELS "regression;training;symbolic-logic;hub;stability"
        TIMEOUT 300)
