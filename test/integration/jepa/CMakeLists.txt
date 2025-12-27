# ============================================================================
# JEPA Integration Tests
# ============================================================================
# WHAT: Build configuration for JEPA module integration tests
# WHY:  Test JEPA system integration with FEP, multimodal fusion, and context
# HOW:  Individual executables for focused testing of each integration area

# -----------------------------------------------------------------------------
# JEPA-FEP Integration Tests
# -----------------------------------------------------------------------------
add_executable(integration_jepa_fep
    test_jepa_fep_integration.cpp
)
target_link_libraries(integration_jepa_fep PRIVATE
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)
target_include_directories(integration_jepa_fep PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(integration_jepa_fep PRIVATE cxx_std_17)
target_compile_definitions(integration_jepa_fep PRIVATE NIMCP_TESTING)
gtest_discover_tests(integration_jepa_fep
    PROPERTIES LABELS "integration;cognitive;jepa;fep"
    TIMEOUT 180
)

# -----------------------------------------------------------------------------
# JEPA Visual-Speech Integration Tests
# -----------------------------------------------------------------------------
add_executable(integration_jepa_visual_speech
    test_jepa_visual_speech.cpp
)
target_link_libraries(integration_jepa_visual_speech PRIVATE
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)
target_include_directories(integration_jepa_visual_speech PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(integration_jepa_visual_speech PRIVATE cxx_std_17)
target_compile_definitions(integration_jepa_visual_speech PRIVATE NIMCP_TESTING)
gtest_discover_tests(integration_jepa_visual_speech
    PROPERTIES LABELS "integration;perception;jepa;multimodal"
    TIMEOUT 180
)

# -----------------------------------------------------------------------------
# JEPA Context Encoding Integration Tests
# -----------------------------------------------------------------------------
add_executable(integration_jepa_context
    test_jepa_context_encoding.cpp
)
target_link_libraries(integration_jepa_context PRIVATE
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)
target_include_directories(integration_jepa_context PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(integration_jepa_context PRIVATE cxx_std_17)
target_compile_definitions(integration_jepa_context PRIVATE NIMCP_TESTING)
gtest_discover_tests(integration_jepa_context
    PROPERTIES LABELS "integration;cognitive;jepa;context"
    TIMEOUT 180
)

# -----------------------------------------------------------------------------
# JEPA End-to-End Tests
# -----------------------------------------------------------------------------
add_executable(integration_jepa_e2e
    test_jepa_e2e.cpp
)
target_link_libraries(integration_jepa_e2e PRIVATE
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)
target_include_directories(integration_jepa_e2e PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(integration_jepa_e2e PRIVATE cxx_std_17)
target_compile_definitions(integration_jepa_e2e PRIVATE NIMCP_TESTING)
gtest_discover_tests(integration_jepa_e2e
    PROPERTIES LABELS "integration;cognitive;jepa;e2e"
    TIMEOUT 300
)

# -----------------------------------------------------------------------------
# Combined JEPA Integration Test (for full pipeline testing)
# -----------------------------------------------------------------------------
add_executable(integration_jepa_full
    test_jepa_fep_integration.cpp
    test_jepa_visual_speech.cpp
    test_jepa_context_encoding.cpp
    test_jepa_e2e.cpp
)
target_link_libraries(integration_jepa_full PRIVATE
    nimcp
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    m
)
target_include_directories(integration_jepa_full PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(integration_jepa_full PRIVATE cxx_std_17)
target_compile_definitions(integration_jepa_full PRIVATE NIMCP_TESTING)
gtest_discover_tests(integration_jepa_full
    PROPERTIES LABELS "integration;cognitive;jepa;full"
    TIMEOUT 300
)
