# ============================================================================
# NIMCP Go Bindings - CGo Build Integration
# ============================================================================
# WHAT: CMake target for Go CGo bindings (links against libnimcp.so)
# WHY:  Provides Go language bindings via CGo FFI
# HOW:  Builds Go package using go build CLI
# ============================================================================

find_program(GO_EXECUTABLE go REQUIRED)

add_custom_target(nimcp_go ALL
    COMMAND ${CMAKE_COMMAND} -E env
            "CGO_LDFLAGS=-L${CMAKE_BINARY_DIR}/lib -lnimcp -Wl,-rpath,${CMAKE_BINARY_DIR}/lib"
            "CGO_CFLAGS=-I${CMAKE_SOURCE_DIR}/include"
            ${GO_EXECUTABLE} build -v ./...
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS nimcp
    COMMENT "Building NIMCP Go bindings"
)
