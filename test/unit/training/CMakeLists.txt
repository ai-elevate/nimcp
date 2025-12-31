# SNN and CNN Training Unit Tests
# Tests for SNN/CNN layer construction, forward/backward pass, gradient computation

# SNN Backprop Unit Tests
# DISABLED: snn_surrogate_t and snn_surrogate_method_t have duplicate enum values
# Needs API refactoring to unify snn/nimcp_snn_types.h and training/nimcp_snn_backprop.h
message(STATUS "unit_training_test_snn_backprop: DISABLED (duplicate enum definitions in headers)")

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

# Auto Architecture Unit Tests
# DISABLED: auto_arch_stats_t/lnn_config_t incomplete type issues
message(STATUS "unit_training_test_auto_architecture: DISABLED (incomplete type issues)")
#[[
add_executable(unit_training_test_auto_architecture
    test_auto_architecture.cpp
)

target_include_directories(unit_training_test_auto_architecture PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(unit_training_test_auto_architecture PRIVATE cxx_std_20)

target_link_libraries(unit_training_test_auto_architecture
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME unit_training_test_auto_architecture
         COMMAND unit_training_test_auto_architecture)

set_tests_properties(unit_training_test_auto_architecture
    PROPERTIES
        LABELS "unit;training;auto;architecture"
        TIMEOUT 180)
#]]

# Auto Architecture NAS Unit Tests
add_executable(unit_training_test_auto_architecture_nas
    test_auto_architecture_nas.cpp
)

target_include_directories(unit_training_test_auto_architecture_nas PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(unit_training_test_auto_architecture_nas PRIVATE cxx_std_20)

target_link_libraries(unit_training_test_auto_architecture_nas
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME unit_training_test_auto_architecture_nas
         COMMAND unit_training_test_auto_architecture_nas)

set_tests_properties(unit_training_test_auto_architecture_nas
    PROPERTIES
        LABELS "unit;training;auto;architecture;nas;evolutionary;rl;darts;pruning;pareto"
        TIMEOUT 300)

# Meta-Learning and HPO Unit Tests
add_executable(unit_training_test_meta_hpo
    test_training_enhancements_meta_hpo.cpp
)

target_include_directories(unit_training_test_meta_hpo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(unit_training_test_meta_hpo PRIVATE cxx_std_20)

target_link_libraries(unit_training_test_meta_hpo
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME unit_training_test_meta_hpo
         COMMAND unit_training_test_meta_hpo)

set_tests_properties(unit_training_test_meta_hpo
    PROPERTIES
        LABELS "unit;training;meta-learning;hpo;hyperparameter"
        TIMEOUT 180)

# Knowledge Distillation and QAT Unit Tests
add_executable(unit_training_test_kd_qat
    test_training_enhancements_kd_qat.cpp
)

target_include_directories(unit_training_test_kd_qat PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(unit_training_test_kd_qat PRIVATE cxx_std_20)

target_link_libraries(unit_training_test_kd_qat
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME unit_training_test_kd_qat
         COMMAND unit_training_test_kd_qat)

set_tests_properties(unit_training_test_kd_qat
    PROPERTIES
        LABELS "unit;training;kd;qat;distillation;quantization"
        TIMEOUT 120)

# Continual Learning and Multi-Task Learning Unit Tests
add_executable(unit_training_test_cl_mtl
    test_training_enhancements_cl_mtl.cpp
)

target_include_directories(unit_training_test_cl_mtl PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(unit_training_test_cl_mtl PRIVATE cxx_std_20)

target_link_libraries(unit_training_test_cl_mtl
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME unit_training_test_cl_mtl
         COMMAND unit_training_test_cl_mtl)

set_tests_properties(unit_training_test_cl_mtl
    PROPERTIES
        LABELS "unit;training;continual-learning;multi-task;cl;mtl"
        TIMEOUT 180)

# Mixed Precision (AMP) and Distributed Training Unit Tests
add_executable(unit_training_test_amp_distributed
    test_training_enhancements_amp_dist.cpp
)

target_include_directories(unit_training_test_amp_distributed PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(unit_training_test_amp_distributed PRIVATE cxx_std_20)

target_link_libraries(unit_training_test_amp_distributed
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME unit_training_test_amp_distributed
         COMMAND unit_training_test_amp_distributed)

set_tests_properties(unit_training_test_amp_distributed
    PROPERTIES
        LABELS "unit;training;amp;mixed-precision;distributed"
        TIMEOUT 180)

# Adversarial Training and Gradient Scaling Unit Tests
add_executable(unit_training_test_adv_gs
    test_training_enhancements_adv_gs.cpp
)

target_include_directories(unit_training_test_adv_gs PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(unit_training_test_adv_gs PRIVATE cxx_std_20)

target_link_libraries(unit_training_test_adv_gs
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
        pthread
        m
)

add_test(NAME unit_training_test_adv_gs
         COMMAND unit_training_test_adv_gs)

set_tests_properties(unit_training_test_adv_gs
    PROPERTIES
        LABELS "unit;training;adversarial;gradient-scaling;adv;gs"
        TIMEOUT 180)
