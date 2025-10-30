cmake_minimum_required(VERSION 3.10)
project(nimcp VERSION 2.5.0 LANGUAGES C CXX)

# Set C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# Find required packages
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(GTest REQUIRED)

# Define project-wide variables
set(NIMCP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
set(NIMCP_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/lib)
set(NIMCP_PYTHON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/python)
set(NIMCP_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tests)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(examples)

# Install rules for headers
include(GNUInstallDirs)
install(DIRECTORY ${NIMCP_INCLUDE_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nimcp
        FILES_MATCHING PATTERN "*.h")

# Export targets
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/nimcpConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
