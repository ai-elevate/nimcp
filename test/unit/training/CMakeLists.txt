# CNN Training Unit Tests
# Tests for CNN layer construction, forward/backward pass, gradient computation

# CNN Backprop Unit Tests
add_executable(unit_training_test_cnn_backprop
    test_cnn_backprop.cpp
)

target_include_directories(unit_training_test_cnn_backprop PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(unit_training_test_cnn_backprop PRIVATE cxx_std_20)

target_link_libraries(unit_training_test_cnn_backprop
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME unit_training_test_cnn_backprop
         COMMAND unit_training_test_cnn_backprop)

set_tests_properties(unit_training_test_cnn_backprop
    PROPERTIES
        LABELS "unit;training;cnn;backprop;gradient"
        TIMEOUT 120)

# CNN-Cortex Bridge Unit Tests
add_executable(unit_training_test_cnn_cortex_bridge
    test_cnn_cortex_bridge.cpp
)

target_include_directories(unit_training_test_cnn_cortex_bridge PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(unit_training_test_cnn_cortex_bridge PRIVATE cxx_std_20)

target_link_libraries(unit_training_test_cnn_cortex_bridge
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME unit_training_test_cnn_cortex_bridge
         COMMAND unit_training_test_cnn_cortex_bridge)

set_tests_properties(unit_training_test_cnn_cortex_bridge
    PROPERTIES
        LABELS "unit;training;cnn;cortex;bridge"
        TIMEOUT 120)
