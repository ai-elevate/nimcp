# Middleware Events Unit Tests
# Test Event Bus
add_executable(unit_middleware_events_event_bus
    test_event_bus.cpp
)
target_link_libraries(unit_middleware_events_event_bus
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_events_event_bus
         COMMAND unit_middleware_events_event_bus)
# Test Event Queue
add_executable(unit_middleware_events_event_queue
    test_event_queue.cpp
)
target_link_libraries(unit_middleware_events_event_queue
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_events_event_queue
         COMMAND unit_middleware_events_event_queue)
# Test Event Subscriber
add_executable(unit_middleware_events_event_subscriber
    test_event_subscriber.cpp
)
target_link_libraries(unit_middleware_events_event_subscriber
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_events_event_subscriber
         COMMAND unit_middleware_events_event_subscriber)
# Test Event Types
add_executable(unit_middleware_events_event_types
    test_event_types.cpp
)
target_link_libraries(unit_middleware_events_event_types
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp_middleware
        nimcp
    )
add_test(NAME unit_middleware_events_event_types
         COMMAND unit_middleware_events_event_types)
