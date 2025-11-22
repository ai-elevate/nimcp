# Middleware Patterns Unit Tests
# Test Oscillation Detector
add_executable(unit_middleware_patterns_oscillation_detector
    test_oscillation_detector.cpp
)
target_link_libraries(unit_middleware_patterns_oscillation_detector
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_patterns_oscillation_detector
         COMMAND unit_middleware_patterns_oscillation_detector)
# Test Pattern Library
add_executable(unit_middleware_patterns_pattern_library
    test_pattern_library.cpp
)
target_link_libraries(unit_middleware_patterns_pattern_library
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_patterns_pattern_library
         COMMAND unit_middleware_patterns_pattern_library)
# Test Sequence Detector
add_executable(unit_middleware_patterns_sequence_detector
    test_sequence_detector.cpp
)
target_link_libraries(unit_middleware_patterns_sequence_detector
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_patterns_sequence_detector
         COMMAND unit_middleware_patterns_sequence_detector)
# Test Synchrony Detector
add_executable(unit_middleware_patterns_synchrony_detector
    test_synchrony_detector.cpp
)
target_link_libraries(unit_middleware_patterns_synchrony_detector
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_patterns_synchrony_detector
         COMMAND unit_middleware_patterns_synchrony_detector)
# Test Oscillation Detector with Phasor Support
add_executable(unit_middleware_patterns_oscillation_detector_phasor
    test_oscillation_detector_phasor.cpp
)
target_link_libraries(unit_middleware_patterns_oscillation_detector_phasor
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_patterns_oscillation_detector_phasor
         COMMAND unit_middleware_patterns_oscillation_detector_phasor)
