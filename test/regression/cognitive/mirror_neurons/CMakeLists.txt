# Cognitive Mirror Neurons Performance Regression Tests

# Mirror Substrate Performance Regression Tests
add_executable(regression_cognitive_mirror_neurons_substrate
    test_mirror_substrate_regression.cpp
)
target_link_libraries(regression_cognitive_mirror_neurons_substrate
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(regression_cognitive_mirror_neurons_substrate
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/utils/memory
        ${CMAKE_SOURCE_DIR}/src/cognitive/mirror_neurons
)
add_test(NAME regression_cognitive_mirror_neurons_substrate
         COMMAND regression_cognitive_mirror_neurons_substrate)
set_tests_properties(regression_cognitive_mirror_neurons_substrate PROPERTIES LABELS "regression")
