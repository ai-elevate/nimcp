#=============================================================================
# Signal Handler Unit Tests
#=============================================================================

# Create test executable
add_executable(test_signal_handler
    test_signal_handler.cpp
)

# Configure includes
target_include_directories(test_signal_handler PRIVATE
    ${NIMCP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Enable C++20 for designated initializers and modern features
target_compile_features(test_signal_handler PRIVATE cxx_std_20)

# Enable test-only APIs if needed
target_compile_definitions(test_signal_handler PRIVATE NIMCP_TESTING)

# Link libraries
target_link_libraries(test_signal_handler
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
        pthread
)

# Register with CTest
add_test(NAME unit_signal_handler
         COMMAND test_signal_handler)

# Set properties
set_target_properties(test_signal_handler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Enable coverage for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(test_signal_handler PRIVATE --coverage)
        target_link_options(test_signal_handler PRIVATE --coverage)
    endif()
endif()
