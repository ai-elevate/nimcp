# ============================================================================
# Brain Factory Unit Tests
# ============================================================================

# Test 1: brain_create() function
add_executable(unit_core_brain_factory_test_brain_create
    test_brain_create.cpp
)
target_link_libraries(unit_core_brain_factory_test_brain_create
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_core_brain_factory_test_brain_create
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME unit_core_brain_factory_test_brain_create
         COMMAND unit_core_brain_factory_test_brain_create)
set_tests_properties(unit_core_brain_factory_test_brain_create PROPERTIES LABELS "unit" TIMEOUT 90)

# Test 2: brain_create_custom() function
add_executable(unit_core_brain_factory_test_brain_create_custom
    test_brain_create_custom.cpp
)
target_link_libraries(unit_core_brain_factory_test_brain_create_custom
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_core_brain_factory_test_brain_create_custom
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME unit_core_brain_factory_test_brain_create_custom
         COMMAND unit_core_brain_factory_test_brain_create_custom)
set_tests_properties(unit_core_brain_factory_test_brain_create_custom PROPERTIES LABELS "unit" TIMEOUT 90)

# Test 3: create_personality() helper function
add_executable(unit_core_brain_factory_test_personality
    test_personality.cpp
)
target_link_libraries(unit_core_brain_factory_test_personality
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_core_brain_factory_test_personality
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME unit_core_brain_factory_test_personality
         COMMAND unit_core_brain_factory_test_personality)
set_tests_properties(unit_core_brain_factory_test_personality PROPERTIES LABELS "unit")

# Test 4: Coordinator init functions (6 coordinators)
add_executable(unit_core_brain_factory_init_coordinators
    init/test_brain_init_coordinators.cpp
)
target_link_libraries(unit_core_brain_factory_init_coordinators
    PRIVATE
        GTest::GTest
        GTest::Main
        nimcp
)
target_include_directories(unit_core_brain_factory_init_coordinators
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME unit_core_brain_factory_init_coordinators
         COMMAND unit_core_brain_factory_init_coordinators)
set_tests_properties(unit_core_brain_factory_init_coordinators PROPERTIES LABELS "unit")
