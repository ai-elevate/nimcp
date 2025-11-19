# Middleware Integration Tests
add_executable(unit_middleware_integration
    test_middleware_integration.cpp
)
target_link_libraries(unit_middleware_integration
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_integration
         COMMAND unit_middleware_integration)
add_subdirectory(buffering)
add_subdirectory(encoding)
add_subdirectory(features)
add_subdirectory(events)
add_subdirectory(normalization)
add_subdirectory(patterns)
add_subdirectory(pipeline)
add_subdirectory(routing)
