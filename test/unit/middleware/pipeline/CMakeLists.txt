# Middleware Pipeline Unit Tests
# Test Middleware Context
add_executable(unit_middleware_pipeline_middleware_context
    test_middleware_context.cpp
)
target_link_libraries(unit_middleware_pipeline_middleware_context
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_pipeline_middleware_context
         COMMAND unit_middleware_pipeline_middleware_context)

# Test Middleware Pipeline
add_executable(unit_middleware_pipeline_middleware_pipeline
    test_middleware_pipeline.cpp
)
target_link_libraries(unit_middleware_pipeline_middleware_pipeline
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_pipeline_middleware_pipeline
         COMMAND unit_middleware_pipeline_middleware_pipeline)

# Test Encoding Stage
add_executable(unit_middleware_pipeline_encoding_stage
    test_encoding_stage.cpp
)
target_link_libraries(unit_middleware_pipeline_encoding_stage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_pipeline_encoding_stage
         COMMAND unit_middleware_pipeline_encoding_stage)

# Test Extraction Stage
add_executable(unit_middleware_pipeline_extraction_stage
    test_extraction_stage.cpp
)
target_link_libraries(unit_middleware_pipeline_extraction_stage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_pipeline_extraction_stage
         COMMAND unit_middleware_pipeline_extraction_stage)

# Test Detection Stage
add_executable(unit_middleware_pipeline_detection_stage
    test_detection_stage.cpp
)
target_link_libraries(unit_middleware_pipeline_detection_stage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_pipeline_detection_stage
         COMMAND unit_middleware_pipeline_detection_stage)

# Test Routing Stage
add_executable(unit_middleware_pipeline_routing_stage
    test_routing_stage.cpp
)
target_link_libraries(unit_middleware_pipeline_routing_stage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_pipeline_routing_stage
         COMMAND unit_middleware_pipeline_routing_stage)

# Test Normalization Stage
add_executable(unit_middleware_pipeline_normalization_stage
    test_normalization_stage.cpp
)
target_link_libraries(unit_middleware_pipeline_normalization_stage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_pipeline_normalization_stage
         COMMAND unit_middleware_pipeline_normalization_stage)

# Test Buffering Stage
add_executable(unit_middleware_pipeline_buffering_stage
    test_buffering_stage.cpp
)
target_link_libraries(unit_middleware_pipeline_buffering_stage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_pipeline_buffering_stage
         COMMAND unit_middleware_pipeline_buffering_stage)

# Test Events Stage
add_executable(unit_middleware_pipeline_events_stage
    test_events_stage.cpp
)
target_link_libraries(unit_middleware_pipeline_events_stage
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_pipeline_events_stage
         COMMAND unit_middleware_pipeline_events_stage)
