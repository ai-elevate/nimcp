# ============================================================================
# NIMCP C# Bindings - .NET P/Invoke Library
# ============================================================================
# WHAT: CMake target for C# bindings (P/Invoke against libnimcp.so)
# WHY:  Provides C# language bindings via Platform Invoke
# HOW:  Builds .NET class library using dotnet CLI
# ============================================================================

find_program(DOTNET_EXECUTABLE dotnet REQUIRED)

add_custom_target(nimcp_csharp ALL
    COMMAND ${DOTNET_EXECUTABLE} build -c Release
            ${CMAKE_CURRENT_SOURCE_DIR}/NIMCP.csproj
            -o ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS nimcp
    COMMENT "Building NIMCP C# bindings"
)
