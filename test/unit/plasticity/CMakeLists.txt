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

# ============================================================================
# Plasticity FEP Bridge Tests
# ============================================================================
# WHAT: Unit tests for Plasticity-FEP bidirectional integration bridges
# WHY:  Ensure synaptic plasticity integrates with FEP precision and predictions
# HOW:  Test lifecycle, effects, and bio-async for each bridge type
# ============================================================================

add_executable(unit_plasticity_test_fep_bridges test_plasticity_fep_bridges.cpp)
target_link_libraries(unit_plasticity_test_fep_bridges
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(unit_plasticity_test_fep_bridges PRIVATE cxx_std_17)
gtest_discover_tests(unit_plasticity_test_fep_bridges
    PROPERTIES LABELS "unit;plasticity;fep;bridge"
    TIMEOUT 60
)

# ============================================================================
# Plasticity Coordinator Tests
# ============================================================================
# WHAT: Unit tests for Plasticity Coordinator module
# WHY:  Central coordinator managing 8 plasticity mechanisms with conflict resolution
# HOW:  Test lifecycle, registration, updates, state management, conflict resolution
# ============================================================================

add_executable(unit_plasticity_test_coordinator test_plasticity_coordinator.cpp)
target_link_libraries(unit_plasticity_test_coordinator
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)
target_compile_features(unit_plasticity_test_coordinator PRIVATE cxx_std_17)
gtest_discover_tests(unit_plasticity_test_coordinator
    PROPERTIES LABELS "unit;plasticity;coordinator"
    TIMEOUT 60
)

# Attention subdirectory (PE tests)
add_subdirectory(attention)

# ============================================================================
# Sleep Bridge Integration Tests
# ============================================================================
# WHAT: Unit tests for Sleep-Plasticity integration bridges
# WHY:  Ensure plasticity mechanisms correctly modulate based on sleep state
# HOW:  Test helper functions, default config, biological validity
# ============================================================================

# Neuromodulators Sleep Bridge Tests
add_executable(unit_plasticity_neuromodulators_sleep_bridge
    neuromodulators/sleep/test_neuromodulators_sleep_bridge.cpp
)
target_link_libraries(unit_plasticity_neuromodulators_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_plasticity_neuromodulators_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_plasticity_neuromodulators_sleep_bridge
    PROPERTIES LABELS "unit;plasticity;neuromodulators;sleep")

# STDP Sleep Bridge Tests
add_executable(unit_plasticity_stdp_sleep_bridge
    stdp/sleep/test_stdp_sleep_bridge.cpp
)
target_link_libraries(unit_plasticity_stdp_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_plasticity_stdp_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_plasticity_stdp_sleep_bridge
    PROPERTIES LABELS "unit;plasticity;stdp;sleep")

# BCM Sleep Bridge Tests
add_executable(unit_plasticity_bcm_sleep_bridge
    bcm/sleep/test_bcm_sleep_bridge.cpp
)
target_link_libraries(unit_plasticity_bcm_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_plasticity_bcm_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_plasticity_bcm_sleep_bridge
    PROPERTIES LABELS "unit;plasticity;bcm;sleep")

# Homeostatic Sleep Bridge Tests
add_executable(unit_plasticity_homeostatic_sleep_bridge
    homeostatic/sleep/test_homeostatic_sleep_bridge.cpp
)
target_link_libraries(unit_plasticity_homeostatic_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_plasticity_homeostatic_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_plasticity_homeostatic_sleep_bridge
    PROPERTIES LABELS "unit;plasticity;homeostatic;sleep")
