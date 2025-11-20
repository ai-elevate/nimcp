cmake_minimum_required(VERSION 3.15)
project(NIMCP VERSION 2.6.2 LANGUAGES C CXX)

# Set prefix path for locally installed packages
set(CMAKE_PREFIX_PATH $ENV{HOME}/.local ${CMAKE_PREFIX_PATH})

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python3 for bindings
find_package(Python3 COMPONENTS Interpreter Development)

# Find GTest for testing
find_package(GTest)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
