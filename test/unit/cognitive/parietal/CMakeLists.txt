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
