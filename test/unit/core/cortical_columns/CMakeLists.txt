# CMakeLists.txt for cortical column unit tests

# Test executable for cortical column module
add_executable(test_cortical_column
    test_cortical_column.cpp
)

# Link libraries
target_link_libraries(test_cortical_column
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

# C++17 standard
target_compile_features(test_cortical_column PRIVATE cxx_std_17)

# Include directories
target_include_directories(test_cortical_column
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core/cortical_columns
)

# Add to test suite
add_test(
    NAME test_cortical_column
    COMMAND test_cortical_column
)

# Set test properties
set_tests_properties(test_cortical_column PROPERTIES
    TIMEOUT 60
    LABELS "unit;core;cortical_columns"
)

# Test executable for columnar connectivity module
add_executable(test_columnar_connectivity
    test_columnar_connectivity.cpp
)

# Link libraries
target_link_libraries(test_columnar_connectivity
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)

target_compile_features(test_columnar_connectivity PRIVATE cxx_std_17)

# Include directories
target_include_directories(test_columnar_connectivity
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Add to test suite
add_test(NAME test_columnar_connectivity
         COMMAND test_columnar_connectivity)

# Set test properties
set_tests_properties(test_columnar_connectivity
    PROPERTIES
        TIMEOUT 120
        LABELS "unit;core;cortical_columns;connectivity"
)

# Test executable for feature hypercolumns module
add_executable(unit_core_cortical_columns_test_feature_hypercolumns
    test_feature_hypercolumns.cpp
    ${CMAKE_SOURCE_DIR}/src/core/cortical_columns/nimcp_feature_hypercolumns.c
)

# Link libraries
target_link_libraries(unit_core_cortical_columns_test_feature_hypercolumns
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
        m
)

target_compile_features(unit_core_cortical_columns_test_feature_hypercolumns PRIVATE cxx_std_17)

# Include directories
target_include_directories(unit_core_cortical_columns_test_feature_hypercolumns
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Add to test suite
add_test(NAME unit_core_cortical_columns_test_feature_hypercolumns
         COMMAND unit_core_cortical_columns_test_feature_hypercolumns)

# Set test properties
set_tests_properties(unit_core_cortical_columns_test_feature_hypercolumns
    PROPERTIES
        TIMEOUT 120
        LABELS "unit;core;cortical_columns;feature_hypercolumns"
)

# Test executable for cortical layers module
add_executable(test_cortical_layers
    test_cortical_layers.cpp
    ${CMAKE_SOURCE_DIR}/src/core/cortical_columns/nimcp_cortical_layers.c
)

# Link libraries
target_link_libraries(test_cortical_layers
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)

# C++20 for designated initializers
target_compile_features(test_cortical_layers PRIVATE cxx_std_20)

# Include directories
target_include_directories(test_cortical_layers
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Add to test suite
add_test(NAME test_cortical_layers
         COMMAND test_cortical_layers)

# Set test properties
set_tests_properties(test_cortical_layers
    PROPERTIES
        TIMEOUT 120
        LABELS "unit;core;cortical_columns;layers"
)
