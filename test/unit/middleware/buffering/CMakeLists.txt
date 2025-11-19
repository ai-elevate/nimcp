# Middleware Buffering Unit Tests
# Test Circular Buffer
add_executable(unit_middleware_buffering_circular_buffer
    test_circular_buffer.cpp
)
target_link_libraries(unit_middleware_buffering_circular_buffer
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_buffering_circular_buffer
         COMMAND unit_middleware_buffering_circular_buffer)
# Test Integration Buffer
add_executable(unit_middleware_buffering_integration_buffer
    test_integration_buffer.cpp
)
target_link_libraries(unit_middleware_buffering_integration_buffer
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_buffering_integration_buffer
         COMMAND unit_middleware_buffering_integration_buffer)
# Test Sliding Window
add_executable(unit_middleware_buffering_sliding_window
    test_sliding_window.cpp
)
target_link_libraries(unit_middleware_buffering_sliding_window
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_buffering_sliding_window
         COMMAND unit_middleware_buffering_sliding_window)
# Test Temporal Accumulator
add_executable(unit_middleware_buffering_temporal_accumulator
    test_temporal_accumulator.cpp
)
target_link_libraries(unit_middleware_buffering_temporal_accumulator
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_buffering_temporal_accumulator
         COMMAND unit_middleware_buffering_temporal_accumulator)
