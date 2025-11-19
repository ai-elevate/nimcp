# Middleware Features Unit Tests

# Test Feature Extractor (Phase 2)
add_executable(unit_middleware_features_feature_extractor
    test_feature_extractor.cpp
)
target_link_libraries(unit_middleware_features_feature_extractor
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_features_feature_extractor
         COMMAND unit_middleware_features_feature_extractor)
