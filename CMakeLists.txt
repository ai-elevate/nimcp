cmake_minimum_required(VERSION 3.15)
project(NIMCP VERSION 2.6.2 LANGUAGES C CXX)

# Set prefix path for locally installed packages
set(CMAKE_PREFIX_PATH $ENV{HOME}/.local ${CMAKE_PREFIX_PATH})

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AVX2 SIMD optimizations on x86_64 platforms
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)")
    # Check if compiler supports AVX2
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "Enabling AVX2 SIMD optimizations")
        add_compile_options(-mavx2 -mfma)
    else()
        message(STATUS "AVX2 not supported by compiler")
    endif()
else()
    message(STATUS "Non-x86_64 platform: AVX2 disabled")
endif()

# Find Python3 for bindings
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find NumPy include directory
if(Python3_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE Python3_NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(Python3_NumPy_INCLUDE_DIRS)
        message(STATUS "NumPy include directory: ${Python3_NumPy_INCLUDE_DIRS}")
    else()
        message(WARNING "NumPy not found - Python bindings may be limited")
    endif()
endif()

# Find GTest for testing
find_package(GTest)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
