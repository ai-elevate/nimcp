# Middleware Subsystems
add_library(nimcp_middleware STATIC
    # Buffering
    buffering/nimcp_circular_buffer.c
    buffering/nimcp_sliding_window.c
    buffering/nimcp_temporal_accumulator.c
    buffering/nimcp_integration_buffer.c

    # Normalization
    normalization/nimcp_zscore_normalizer.c
    normalization/nimcp_min_max_normalizer.c
    normalization/nimcp_adaptive_normalizer.c
    normalization/nimcp_homeostatic_normalizer.c

    # Encoding (Phase 1 & Phase 2)
    encoding/nimcp_rate_coding.c
    encoding/nimcp_temporal_coding.c
    encoding/nimcp_population_coding.c

    # Feature Extraction (Phase 2)
    features/nimcp_feature_extractor.c

    # Brain Integration
    brain_integration.c

    # Events
    events/nimcp_event_bus.c
    events/nimcp_event_queue.c
    events/nimcp_event_subscriber.c
    events/nimcp_event_types.c

    # Pattern Recognition
    patterns/nimcp_synchrony_detector.c
    patterns/nimcp_sequence_detector.c
    patterns/nimcp_oscillation_detector.c
    patterns/nimcp_pattern_library.c

    # Thalamic Routing
    routing/nimcp_thalamic_router.c
    routing/nimcp_attention_gate.c
    routing/nimcp_routing_table.c
)

target_include_directories(nimcp_middleware PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(nimcp_middleware PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(nimcp_middleware
    PRIVATE
        m
)

install(TARGETS nimcp_middleware
    EXPORT nimcp-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Install middleware headers from include/middleware
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/middleware
    DESTINATION include/nimcp
    FILES_MATCHING PATTERN "*.h"
)
