# Middleware Encoding Unit Tests
# Test Rate Coding
add_executable(unit_middleware_encoding_rate_coding
    test_rate_coding.cpp
)
target_link_libraries(unit_middleware_encoding_rate_coding
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_encoding_rate_coding
         COMMAND unit_middleware_encoding_rate_coding)

# Test Temporal Coding
add_executable(unit_middleware_encoding_temporal_coding
    test_temporal_coding.cpp
)
target_link_libraries(unit_middleware_encoding_temporal_coding
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_encoding_temporal_coding
         COMMAND unit_middleware_encoding_temporal_coding)

# Test Population Coding (Phase 2)
add_executable(unit_middleware_encoding_population_coding
    test_population_coding.cpp
)
target_link_libraries(unit_middleware_encoding_population_coding
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_encoding_population_coding
         COMMAND unit_middleware_encoding_population_coding)
