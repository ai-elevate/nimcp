# Example programs

# NIMCP 2.0 Example
add_executable(event_demo event_demo.c)
target_link_libraries(event_demo nimcp)
target_include_directories(event_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.5 Brain API Example
add_executable(brain_demo brain_demo.c)
target_link_libraries(brain_demo nimcp)
target_include_directories(brain_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.5 Ethics Engine Example
add_executable(ethics_demo ethics_demo.c)
target_link_libraries(ethics_demo nimcp)
target_include_directories(ethics_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.5 Infant Learning Demo (conceptual)
add_executable(infant_demo infant_learning_demo.c)
target_link_libraries(infant_demo nimcp)
target_include_directories(infant_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.5 Integrated Learning Demo (FULLY FUNCTIONAL)
add_executable(integrated_demo integrated_learning_demo.c)
target_link_libraries(integrated_demo nimcp m)
target_include_directories(integrated_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.6 Brain Probe Demo
add_executable(brain_probe_demo brain_probe_demo.c)
target_link_libraries(brain_probe_demo nimcp)
target_include_directories(brain_probe_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# Install examples
install(TARGETS event_demo brain_demo ethics_demo infant_demo integrated_demo brain_probe_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
)
