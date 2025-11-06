#==============================================================================
# Fuzzing Infrastructure for NIMCP
#==============================================================================
# Provides fuzzing targets for critical APIs using libFuzzer or AFL++
#
# Build with:
#   cmake -DENABLE_FUZZING=ON ..
#   make
#
# Run fuzzers:
#   ./fuzz_neuralnet -max_total_time=300
#   ./fuzz_protocol -max_total_time=300
#==============================================================================

cmake_minimum_required(VERSION 3.10)

# Only build if fuzzing is enabled
if(NOT ENABLE_FUZZING)
    message(STATUS "Fuzzing disabled (set -DENABLE_FUZZING=ON to enable)")
    return()
endif()

message(STATUS "Fuzzing infrastructure: ENABLED")

# Check for libFuzzer support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsanitize=fuzzer" HAS_LIBFUZZER)

if(NOT HAS_LIBFUZZER)
    message(WARNING "libFuzzer not supported by compiler, fuzzing targets disabled")
    return()
endif()

# Common fuzzer flags
set(FUZZER_FLAGS
    -fsanitize=fuzzer
    -fsanitize=address
    -fsanitize=undefined
    -g
    -O1  # Optimize for fast fuzzing
)

#==============================================================================
# Fuzz Target 1: Neural Network API
#==============================================================================

add_executable(fuzz_neuralnet fuzz_neuralnet.cpp)

target_include_directories(fuzz_neuralnet
    PRIVATE
        ${NIMCP_INCLUDE_DIR}
        ${NIMCP_LIB_DIR}
)

target_compile_options(fuzz_neuralnet PRIVATE ${FUZZER_FLAGS})
target_link_options(fuzz_neuralnet PRIVATE ${FUZZER_FLAGS})

target_link_libraries(fuzz_neuralnet
    PRIVATE
        nimcp
)

#==============================================================================
# Fuzz Target 2: Protocol Serialization
#==============================================================================

add_executable(fuzz_protocol fuzz_protocol.cpp)

target_include_directories(fuzz_protocol
    PRIVATE
        ${NIMCP_INCLUDE_DIR}
        ${NIMCP_LIB_DIR}
)

target_compile_options(fuzz_protocol PRIVATE ${FUZZER_FLAGS})
target_link_options(fuzz_protocol PRIVATE ${FUZZER_FLAGS})

target_link_libraries(fuzz_protocol
    PRIVATE
        nimcp
)

#==============================================================================
# Fuzz Target 3: Queue Manager
#==============================================================================

add_executable(fuzz_queue_manager fuzz_queue_manager.cpp)

target_include_directories(fuzz_queue_manager
    PRIVATE
        ${NIMCP_INCLUDE_DIR}
        ${NIMCP_LIB_DIR}
)

target_compile_options(fuzz_queue_manager PRIVATE ${FUZZER_FLAGS})
target_link_options(fuzz_queue_manager PRIVATE ${FUZZER_FLAGS})

target_link_libraries(fuzz_queue_manager
    PRIVATE
        nimcp
)

#==============================================================================
# Fuzz Target 4: Validation Functions
#==============================================================================

add_executable(fuzz_validate fuzz_validate.cpp)

target_include_directories(fuzz_validate
    PRIVATE
        ${NIMCP_INCLUDE_DIR}
        ${NIMCP_LIB_DIR}
)

target_compile_options(fuzz_validate PRIVATE ${FUZZER_FLAGS})
target_link_options(fuzz_validate PRIVATE ${FUZZER_FLAGS})

target_link_libraries(fuzz_validate
    PRIVATE
        nimcp
)

message(STATUS "Fuzzing targets:")
message(STATUS "  - fuzz_neuralnet")
message(STATUS "  - fuzz_protocol")
message(STATUS "  - fuzz_queue_manager")
message(STATUS "  - fuzz_validate")
