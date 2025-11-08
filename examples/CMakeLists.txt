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

# NIMCP 2.6 Izhikevich Neuron Model Demo
add_executable(izhikevich_demo izhikevich_demo.c)
target_link_libraries(izhikevich_demo nimcp m)
target_include_directories(izhikevich_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.7 Programmable Synapses Demo (MAJOR FEATURE)
add_executable(programmable_synapses_demo programmable_synapses_demo.c)
target_link_libraries(programmable_synapses_demo nimcp m)
target_include_directories(programmable_synapses_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.7 NLP Integration Test (Synapses + Attention + Neuromodulation)
add_executable(nlp_integration_test nlp_integration_test.c)
target_link_libraries(nlp_integration_test nimcp m)
target_include_directories(nlp_integration_test PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.7 Fractal Network Demo (Scale-Free Topology + Pink Noise)
add_executable(fractal_network_demo fractal_network_demo.c)
target_link_libraries(fractal_network_demo nimcp m)
target_include_directories(fractal_network_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.7 Phase 2 - NLP with Fractal Networks
add_executable(nlp_integration_demo nlp_integration_demo.c)
target_link_libraries(nlp_integration_demo nimcp m)
target_include_directories(nlp_integration_demo PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.7 Phase 2 - Pink Noise Weight Initialization Test
add_executable(test_pink_weights test_pink_weights.c)
target_link_libraries(test_pink_weights nimcp m)
target_include_directories(test_pink_weights PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.7 Phase 3 - Integration Test
add_executable(test_phase3 test_phase3.c)
target_link_libraries(test_phase3 nimcp m)
target_include_directories(test_phase3 PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.7 Phase 4 - STDP + Eligibility Traces Test (Simplified)
add_executable(test_phase4 test_phase4_simple.c)
target_link_libraries(test_phase4 nimcp m)
target_include_directories(test_phase4 PRIVATE ${NIMCP_INCLUDE_DIR})

# NIMCP 2.7 Phase 4 - Full System Integration Test (ALL subsystems)
add_executable(full_system_integration_test full_system_integration_test.c)
target_link_libraries(full_system_integration_test nimcp m)
target_include_directories(full_system_integration_test PRIVATE ${NIMCP_INCLUDE_DIR})

# GPU Integration Test - Requires GPU neuron API (coming in Phase 2.7)
# add_executable(gpu_integration_test gpu_integration_test.c)
# target_link_libraries(gpu_integration_test nimcp)
# target_include_directories(gpu_integration_test PRIVATE ${NIMCP_INCLUDE_DIR})

# Install examples
install(TARGETS event_demo brain_demo ethics_demo infant_demo integrated_demo brain_probe_demo izhikevich_demo programmable_synapses_demo nlp_integration_test fractal_network_demo nlp_integration_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
)
