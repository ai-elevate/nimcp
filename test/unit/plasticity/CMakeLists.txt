# Unit tests for plasticity modules with bio-async integration

# Attention bio-async tests
add_executable(test_attention_bioasync
    attention/test_attention_bioasync.cpp
)
target_link_libraries(test_attention_bioasync
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(test_attention_bioasync PRIVATE cxx_std_17)
gtest_discover_tests(test_attention_bioasync
    PROPERTIES LABELS "unit;plasticity;attention")

# Dendritic bio-async tests
add_executable(test_dendritic_bioasync
    dendritic/test_dendritic_bioasync.cpp
)
target_link_libraries(test_dendritic_bioasync
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(test_dendritic_bioasync PRIVATE cxx_std_17)
gtest_discover_tests(test_dendritic_bioasync
    PROPERTIES LABELS "unit;plasticity;dendritic")

# Adaptive bio-async tests
add_executable(test_adaptive_bioasync
    adaptive/test_adaptive_bioasync.cpp
)
target_link_libraries(test_adaptive_bioasync
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(test_adaptive_bioasync PRIVATE cxx_std_17)
gtest_discover_tests(test_adaptive_bioasync
    PROPERTIES LABELS "unit;plasticity;adaptive")

# Predictive coding bio-async tests
add_executable(test_predictive_bioasync
    predictive/test_predictive_bioasync.cpp
)
target_link_libraries(test_predictive_bioasync
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(test_predictive_bioasync PRIVATE cxx_std_17)
gtest_discover_tests(test_predictive_bioasync
    PROPERTIES LABELS "unit;plasticity;predictive")

# Second messenger cascade tests
add_executable(unit_plasticity_second_messengers_test
    second_messengers/test_second_messengers.cpp
)
target_link_libraries(unit_plasticity_second_messengers_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(unit_plasticity_second_messengers_test PRIVATE cxx_std_17)
gtest_discover_tests(unit_plasticity_second_messengers_test
    PROPERTIES LABELS "unit;plasticity;second_messengers")

# Second messenger module integration unit tests
add_executable(unit_plasticity_second_messengers_modules_test
    second_messengers/test_second_messengers_modules.cpp
)
target_link_libraries(unit_plasticity_second_messengers_modules_test
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(unit_plasticity_second_messengers_modules_test PRIVATE cxx_std_17)
gtest_discover_tests(unit_plasticity_second_messengers_modules_test
    PROPERTIES LABELS "unit;plasticity;second_messengers;modules")

# Neuromodulator-immune integration tests
add_executable(unit_plasticity_neuromodulator_immune
    immune/test_neuromodulator_immune.cpp
)
target_link_libraries(unit_plasticity_neuromodulator_immune
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(unit_plasticity_neuromodulator_immune PRIVATE cxx_std_17)
gtest_discover_tests(unit_plasticity_neuromodulator_immune
    PROPERTIES LABELS "unit;plasticity;immune;neuromodulator")

# Attention subdirectory (PE tests)
add_subdirectory(attention)
