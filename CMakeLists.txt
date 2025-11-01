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

#==============================================================================
# Security and Hardening Options
#==============================================================================

# Sanitizer options (mutually exclusive: ASAN vs TSAN)
option(ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer for race condition detection" OFF)
option(ENABLE_HARDENING "Enable security hardening compiler flags" ON)

# Validate sanitizer options (ASAN and TSAN are mutually exclusive)
if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer are mutually exclusive. Choose one.")
endif()

# AddressSanitizer - detects memory errors (use-after-free, buffer overflows, etc.)
if(ENABLE_ASAN)
    set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
    message(STATUS "AddressSanitizer: ENABLED")
endif()

# UndefinedBehaviorSanitizer - detects undefined behavior
if(ENABLE_UBSAN)
    set(UBSAN_FLAGS "-fsanitize=undefined -fno-sanitize-recover=undefined")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${UBSAN_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UBSAN_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=undefined")
    message(STATUS "UndefinedBehaviorSanitizer: ENABLED")
endif()

# ThreadSanitizer - detects data races (mutually exclusive with ASAN)
if(ENABLE_TSAN)
    set(TSAN_FLAGS "-fsanitize=thread -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TSAN_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TSAN_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
    message(STATUS "ThreadSanitizer: ENABLED")
endif()

# Security hardening flags (enabled by default in Release builds)
if(ENABLE_HARDENING)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        # Fortify source - compile-time and runtime buffer overflow detection
        add_definitions(-D_FORTIFY_SOURCE=2)

        # Stack protection
        set(HARDENING_FLAGS
            -fstack-protector-strong      # Stack canary protection
            -fPIE                          # Position independent executable
            -Wformat                       # Format string vulnerability detection
            -Wformat-security              # Additional format string checks
            -Werror=format-security        # Make format violations errors
            -fno-common                    # Prevent common symbol vulnerabilities
        )

        add_compile_options(${HARDENING_FLAGS})

        # Link-time hardening
        if(NOT APPLE)  # Linux-specific hardening
            add_link_options(
                -pie                       # Enable PIE
                -Wl,-z,relro               # Read-only relocations
                -Wl,-z,now                 # Immediate binding (full RELRO)
                -Wl,-z,noexecstack         # Non-executable stack
            )
        endif()

        message(STATUS "Security hardening: ENABLED")
        message(STATUS "  - FORTIFY_SOURCE=2 (buffer overflow detection)")
        message(STATUS "  - Stack protector strong")
        message(STATUS "  - Position Independent Executable (PIE)")
        message(STATUS "  - Full RELRO (Read-only relocations)")
    else()
        message(STATUS "Security hardening: PARTIAL (Debug build)")
    endif()
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

# Generate and install pkg-config file
configure_file(nimcp.pc.in nimcp.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/nimcp.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# Export targets
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NIMCPConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/NIMCPConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NIMCPConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NIMCP
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NIMCPConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NIMCPConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NIMCP
)

install(EXPORT nimcp-targets
    FILE NIMCPTargets.cmake
    NAMESPACE NIMCP::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NIMCP
)

# Configuration summary
message(STATUS "")
message(STATUS "===============================================================================")
message(STATUS "NIMCP ${PROJECT_VERSION} - Configuration Summary")
message(STATUS "===============================================================================")
message(STATUS "Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Security Features:")
message(STATUS "  AddressSanitizer (ASAN): ${ENABLE_ASAN}")
message(STATUS "  UndefinedBehaviorSanitizer (UBSAN): ${ENABLE_UBSAN}")
message(STATUS "  ThreadSanitizer (TSAN): ${ENABLE_TSAN}")
message(STATUS "  Security Hardening: ${ENABLE_HARDENING}")
message(STATUS "")
message(STATUS "Installation:")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Library: ${CMAKE_INSTALL_LIBDIR}/libnimcp_core.so")
message(STATUS "  Headers: ${CMAKE_INSTALL_INCLUDEDIR}/nimcp/")
message(STATUS "  Pkg-config: ${CMAKE_INSTALL_LIBDIR}/pkgconfig/nimcp.pc")
message(STATUS "===============================================================================")
message(STATUS "")
