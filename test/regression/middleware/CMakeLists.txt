# Middleware Regression Tests
add_subdirectory(buffering)
add_subdirectory(encoding)
add_subdirectory(events)
add_subdirectory(normalization)
add_subdirectory(patterns)
add_subdirectory(pipeline)
add_subdirectory(routing)

# Phase 2 Regression Tests
# TODO: Fix outdated API usage in test_phase2_regression.cpp
# - Replace rate_coder_t with rate_coding_encoder_t
# - Replace spike_record_t with spike_train_t
# - Update all function signatures to match current API
# Temporarily disabled until Phase 3 implementation is complete
#add_executable(regression_middleware_middleware_test_phase2_regression
#    test_phase2_regression.cpp
#)
#target_link_libraries(regression_middleware_middleware_test_phase2_regression
#    PRIVATE
#        GTest::GTest
#        GTest::Main
#        nimcp_middleware
#        nimcp
#    )
#add_test(NAME regression_middleware_middleware_test_phase2_regression
#         COMMAND regression_middleware_middleware_test_phase2_regression)
