# ============================================================================
# Glial Unit Tests
# ============================================================================

# Unit test for enhanced oligodendrocytes
add_executable(unit_glial_oligodendrocytes_enhanced
    oligodendrocytes/test_oligodendrocytes_enhanced.cpp
)

target_link_libraries(unit_glial_oligodendrocytes_enhanced
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)

target_compile_features(unit_glial_oligodendrocytes_enhanced PRIVATE cxx_std_17)

target_include_directories(unit_glial_oligodendrocytes_enhanced
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_glial_oligodendrocytes_enhanced
         COMMAND unit_glial_oligodendrocytes_enhanced)

set_tests_properties(unit_glial_oligodendrocytes_enhanced
    PROPERTIES
        TIMEOUT 60
        LABELS "unit;glial;oligodendrocyte"
)

# Unit test for enhanced microglia
add_executable(unit_glial_microglia_enhanced
    microglia/test_microglia_enhanced.cpp
)

target_link_libraries(unit_glial_microglia_enhanced
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)

target_compile_features(unit_glial_microglia_enhanced PRIVATE cxx_std_17)

target_include_directories(unit_glial_microglia_enhanced
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_glial_microglia_enhanced
         COMMAND unit_glial_microglia_enhanced)

set_tests_properties(unit_glial_microglia_enhanced
    PROPERTIES
        TIMEOUT 60
        LABELS "unit;glial;microglia"
)

# ============================================================================
# Unit test for myelin sheath
# ============================================================================
add_executable(unit_glial_myelin_sheath
    myelin_sheath/test_myelin_sheath.cpp
)

target_link_libraries(unit_glial_myelin_sheath
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)

target_compile_features(unit_glial_myelin_sheath PRIVATE cxx_std_17)

target_include_directories(unit_glial_myelin_sheath
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_glial_myelin_sheath
         COMMAND unit_glial_myelin_sheath)

set_tests_properties(unit_glial_myelin_sheath
    PROPERTIES
        TIMEOUT 120
        LABELS "unit;glial;myelin_sheath"
)

# ============================================================================
# Unit test for myelin math (biophysics mathematical models)
# ============================================================================
add_executable(unit_glial_myelin_math
    myelin_math/test_myelin_math.cpp
)

target_link_libraries(unit_glial_myelin_math
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

target_compile_features(unit_glial_myelin_math PRIVATE cxx_std_17)

target_include_directories(unit_glial_myelin_math
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME unit_glial_myelin_math
         COMMAND unit_glial_myelin_math)

set_tests_properties(unit_glial_myelin_math
    PROPERTIES
        TIMEOUT 180
        LABELS "unit;glial;myelin;biophysics;math"
)
