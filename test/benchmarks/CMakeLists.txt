#=============================================================================
# CMakeLists.txt - Complex Math Benchmarks
#=============================================================================

cmake_minimum_required(VERSION 3.18)

# Benchmark executables
set(BENCHMARK_SOURCES
    benchmark_complex_vs_real_oscillations.c
    benchmark_complex_pac_detection.c
    benchmark_complex_fft.c
)

# Find optional FFTW library
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW3F fftw3f)
endif()

# Create benchmark executables
foreach(BENCH_SOURCE ${BENCHMARK_SOURCES})
    # Get executable name from source file
    get_filename_component(BENCH_NAME ${BENCH_SOURCE} NAME_WE)
    set(BENCH_TARGET "bench_${BENCH_NAME}")

    # Create executable
    add_executable(${BENCH_TARGET} ${BENCH_SOURCE})

    # Link libraries
    # Note: Link complex math object directly to avoid unresolved symbols from main library
    target_link_libraries(${BENCH_TARGET}
        PRIVATE
            ${CMAKE_BINARY_DIR}/src/lib/CMakeFiles/nimcp.dir/__/utils/math/nimcp_complex_math.c.o
            m
            pthread
    )

    # Include directories
    target_include_directories(${BENCH_TARGET}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
    )

    # Compile options for benchmarks (optimization)
    target_compile_options(${BENCH_TARGET}
        PRIVATE
            -O3
            -march=native
            -Wall
            -Wextra
    )

    # Link to FFTW if available (for FFT benchmark)
    if(FFTW3F_FOUND AND BENCH_NAME STREQUAL "benchmark_complex_fft")
        target_link_libraries(${BENCH_TARGET} PRIVATE ${FFTW3F_LIBRARIES})
        target_include_directories(${BENCH_TARGET} PRIVATE ${FFTW3F_INCLUDE_DIRS})
        target_compile_definitions(${BENCH_TARGET} PRIVATE HAVE_FFTW)
        message(STATUS "FFTW3F found - enabling FFTW comparison in FFT benchmark")
    endif()

    # Add to benchmarks target
    add_custom_target(run_${BENCH_TARGET}
        COMMAND ${BENCH_TARGET}
        DEPENDS ${BENCH_TARGET}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running ${BENCH_NAME}"
    )
endforeach()

# Add master benchmark target to run all
add_custom_target(run_all_benchmarks
    DEPENDS
        run_bench_benchmark_complex_vs_real_oscillations
        run_bench_benchmark_complex_pac_detection
        run_bench_benchmark_complex_fft
    COMMENT "Running all complex math benchmarks"
)

# Generate benchmark report
add_custom_target(benchmark_report
    COMMAND ${CMAKE_COMMAND} -E echo "Running benchmarks and generating report..."
    COMMAND bash -c "cd ${CMAKE_BINARY_DIR}/test/benchmarks && ./bench_benchmark_complex_vs_real_oscillations > oscillations_results.txt 2>&1 || true"
    COMMAND bash -c "cd ${CMAKE_BINARY_DIR}/test/benchmarks && ./bench_benchmark_complex_pac_detection > pac_results.txt 2>&1 || true"
    COMMAND bash -c "cd ${CMAKE_BINARY_DIR}/test/benchmarks && ./bench_benchmark_complex_fft > fft_results.txt 2>&1 || true"
    COMMAND ${CMAKE_COMMAND} -E echo "Benchmark results saved to test/benchmarks/*_results.txt"
    DEPENDS
        bench_benchmark_complex_vs_real_oscillations
        bench_benchmark_complex_pac_detection
        bench_benchmark_complex_fft
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating benchmark report"
)

# Installation (optional)
install(TARGETS
    bench_benchmark_complex_vs_real_oscillations
    bench_benchmark_complex_pac_detection
    bench_benchmark_complex_fft
    RUNTIME DESTINATION bin/benchmarks
)

# Print configuration summary
message(STATUS "Benchmark configuration:")
message(STATUS "  Optimization: -O3 -march=native")
message(STATUS "  FFTW3F: ${FFTW3F_FOUND}")
