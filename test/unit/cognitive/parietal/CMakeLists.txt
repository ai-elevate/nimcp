# Unit tests for Parietal Lobe module (Phase 7.2)
# Mathematical/Scientific Reasoning with full system integration

#=============================================================================
# Number Sense Tests
#=============================================================================

add_executable(unit_parietal_number_sense
    test_number_sense.cpp
)

target_link_libraries(unit_parietal_number_sense
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_number_sense
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_number_sense PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_number_sense
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;number-sense"
        TIMEOUT 90
)

#=============================================================================
# Spatial Reasoning Tests
#=============================================================================

add_executable(unit_parietal_spatial_reasoning
    test_spatial_reasoning.cpp
)

target_link_libraries(unit_parietal_spatial_reasoning
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_spatial_reasoning
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_spatial_reasoning PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_spatial_reasoning
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;spatial"
        TIMEOUT 90
)

#=============================================================================
# Mathematical Intuition Tests
#=============================================================================

add_executable(unit_parietal_mathematical_intuition
    test_mathematical_intuition.cpp
)

target_link_libraries(unit_parietal_mathematical_intuition
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_mathematical_intuition
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_mathematical_intuition PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_mathematical_intuition
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;math-intuition"
        TIMEOUT 90
)

#=============================================================================
# Scientific Reasoning Tests
#=============================================================================

add_executable(unit_parietal_scientific_reasoning
    test_scientific_reasoning.cpp
)

target_link_libraries(unit_parietal_scientific_reasoning
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_scientific_reasoning
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_scientific_reasoning PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_scientific_reasoning
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;scientific"
        TIMEOUT 90
)

#=============================================================================
# Equation Manipulation Tests
#=============================================================================

add_executable(unit_parietal_equation_manipulation
    test_equation_manipulation.cpp
)

target_link_libraries(unit_parietal_equation_manipulation
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_equation_manipulation
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_equation_manipulation PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_equation_manipulation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;equation"
        TIMEOUT 90
)

#=============================================================================
# Chemistry Tests
#=============================================================================

add_executable(unit_parietal_chemistry
    test_chemistry.cpp
)

target_link_libraries(unit_parietal_chemistry
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_chemistry
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_chemistry PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_chemistry
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;chemistry"
        TIMEOUT 90
)

#=============================================================================
# Biology Tests
#=============================================================================

add_executable(unit_parietal_biology
    test_biology.cpp
)

target_link_libraries(unit_parietal_biology
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_biology
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_biology PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_biology
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;biology"
        TIMEOUT 90
)

#=============================================================================
# Software Engineering Tests
#=============================================================================

add_executable(unit_parietal_software_engineering
    test_software_engineering.cpp
)

target_link_libraries(unit_parietal_software_engineering
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_software_engineering
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_software_engineering PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_software_engineering
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;software-engineering"
        TIMEOUT 90
)

#=============================================================================
# Parietal Orchestrator Tests
#=============================================================================

add_executable(unit_parietal_orchestrator
    test_parietal.cpp
)

target_link_libraries(unit_parietal_orchestrator
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_orchestrator
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_orchestrator PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_orchestrator
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;orchestrator"
        TIMEOUT 90
)

#=============================================================================
# Phase 6: Intuitive Reasoning & Discovery Tests
#=============================================================================

#-----------------------------------------------------------------------------
# Phase 6.1: Intuitive Reasoning
#-----------------------------------------------------------------------------

add_executable(unit_parietal_intuitive_reasoning
    test_intuitive_reasoning.cpp
)

target_link_libraries(unit_parietal_intuitive_reasoning
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_intuitive_reasoning
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_intuitive_reasoning PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_intuitive_reasoning
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;intuitive-reasoning;phase6"
        TIMEOUT 90
)

#-----------------------------------------------------------------------------
# Phase 6.2: Analogical Reasoning
#-----------------------------------------------------------------------------

add_executable(unit_parietal_analogical_reasoning
    test_analogical_reasoning.cpp
)

target_link_libraries(unit_parietal_analogical_reasoning
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_analogical_reasoning
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_analogical_reasoning PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_analogical_reasoning
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;analogical-reasoning;phase6"
        TIMEOUT 90
)

#-----------------------------------------------------------------------------
# Phase 6.3: Insight Discovery
#-----------------------------------------------------------------------------

add_executable(unit_parietal_insight_discovery
    test_insight_discovery.cpp
)

target_link_libraries(unit_parietal_insight_discovery
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_insight_discovery
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_insight_discovery PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_insight_discovery
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;insight-discovery;phase6"
        TIMEOUT 90
)

#-----------------------------------------------------------------------------
# Phase 6.4: Hypothesis Generation
#-----------------------------------------------------------------------------

add_executable(unit_parietal_hypothesis_generation
    test_hypothesis_generation.cpp
)

target_link_libraries(unit_parietal_hypothesis_generation
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_hypothesis_generation
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_hypothesis_generation PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_hypothesis_generation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;hypothesis-generation;phase6"
        TIMEOUT 90
)

#-----------------------------------------------------------------------------
# Phase 6.5: Conceptual Blending
#-----------------------------------------------------------------------------

add_executable(unit_parietal_conceptual_blending
    test_conceptual_blending.cpp
)

target_link_libraries(unit_parietal_conceptual_blending
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_conceptual_blending
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_conceptual_blending PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_conceptual_blending
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;conceptual-blending;phase6"
        TIMEOUT 90
)

#-----------------------------------------------------------------------------
# Phase 6.6: Counterfactual Reasoning
#-----------------------------------------------------------------------------

add_executable(unit_parietal_counterfactual
    test_counterfactual.cpp
)

target_link_libraries(unit_parietal_counterfactual
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_counterfactual
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_counterfactual PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_counterfactual
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;counterfactual;phase6"
        TIMEOUT 90
)

#-----------------------------------------------------------------------------
# Phase 6.7: Meta-Reasoning
#-----------------------------------------------------------------------------

add_executable(unit_parietal_meta_reasoning
    test_meta_reasoning.cpp
)

target_link_libraries(unit_parietal_meta_reasoning
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_meta_reasoning
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_meta_reasoning PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_meta_reasoning
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;meta-reasoning;phase6"
        TIMEOUT 90
)

#-----------------------------------------------------------------------------
# Phase 6.0: Intuition Integrations
#-----------------------------------------------------------------------------

add_executable(unit_parietal_intuition_integrations
    test_intuition_integrations.cpp
)

target_link_libraries(unit_parietal_intuition_integrations
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_intuition_integrations
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_intuition_integrations PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_intuition_integrations
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;intuition-integrations;phase6"
        TIMEOUT 120
)

#=============================================================================
# Parietal FEP Orchestrator Bridge Tests
#=============================================================================

add_executable(unit_parietal_fep_bridge
    test_parietal_fep_bridge.cpp
)

target_link_libraries(unit_parietal_fep_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_fep_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_fep_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_fep_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;fep-bridge"
        TIMEOUT 90
)

#=============================================================================
# Code Generation Engine Tests (Self-Repair Integration)
#=============================================================================

add_executable(unit_parietal_code_generation
    test_code_generation.cpp
)

target_link_libraries(unit_parietal_code_generation
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_code_generation
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_code_generation PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_code_generation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;code-generation;self-repair"
        TIMEOUT 120
)

#=============================================================================
# Code Generation Bio-Async Tests
#=============================================================================

add_executable(unit_parietal_code_generation_bio_async
    test_code_generation_bio_async.cpp
)

target_link_libraries(unit_parietal_code_generation_bio_async
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_code_generation_bio_async
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_code_generation_bio_async PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_code_generation_bio_async
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;code-generation;bio-async"
        TIMEOUT 60
)

#=============================================================================
# Parietal Training Bridge Tests
#=============================================================================

add_executable(unit_parietal_training_bridge
    test_parietal_training_bridge.cpp
)

target_link_libraries(unit_parietal_training_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_parietal_training_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_parietal_training_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_parietal_training_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;parietal;training-bridge"
        TIMEOUT 120
)
