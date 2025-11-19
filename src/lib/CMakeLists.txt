# Core library sources - New hierarchical structure
set(NIMCP_CORE_SOURCES
    # Public API facade (unified interface for all bindings)
    ${CMAKE_CURRENT_SOURCE_DIR}/../api/nimcp.c

    # Core brain components
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/nimcp_brain.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/nimcp_brain_resize.c  # Phase 2.8: Dynamic Brain Resizing
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/nimcp_distributed_cow.c
    # ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/nimcp_pretrained.c  # TODO: Needs libcjson-dev

    # Brain modular architecture (Phase 11: SRP Modularization) - 7 extracted modules
    # ENABLED: Type conflicts resolved, modules ready for integration
    # Modules extracted and ready in src/core/brain/{factory,learning,inference,persistence,distributed,strategy,biological}/
    # See BRAIN_MODULARIZATION_PLAN.md for integration roadmap
    # Brain factory (Phase 11: SRP Split - 3,220 lines into 3 modules)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/factory/nimcp_brain_factory.c                  # Main factory orchestration (685 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/factory/init/nimcp_brain_init.c               # Initialization and subsystems (2,523 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/factory/validation/nimcp_brain_validation.c   # Validation and caching (219 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/learning/nimcp_brain_learning.c     # Training, backpropagation, batch processing (947 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/inference/nimcp_brain_inference.c   # Prediction, forward propagation, results (502 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/persistence/nimcp_brain_persistence.c # Save/load, snapshots, checkpoints (1,187 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/distributed/nimcp_brain_distributed.c # Copy-on-write, P2P sync, consensus (561 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/strategy/nimcp_brain_strategy.c     # Strategy pattern, monitoring, optimization (770 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/cognitive/nimcp_brain_cognitive.c   # Cognitive systems subsystem (2,150 lines extracted)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/biological/nimcp_brain_biological.c # Glial, neuromodulators, pink noise, multimodal (~1,500 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/accessors/nimcp_brain_accessors.c   # Brain accessor functions (388 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/pretrained/nimcp_brain_pretrained.c # Pretrained model management (510 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/information/nimcp_brain_shannon.c   # Shannon information theory (461 lines)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/analysis/nimcp_brain_topology.c     # Topology and community detection (462 lines)

    # Brain processing modules (Phase 9.1: SRP Refactoring) - TODO: Enable when refactoring complete
    # ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/processing/sensory_extractor.c
    # ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/processing/multimodal_integrator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain/processing/cognitive_processor.c  # Cognitive output processing

    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuralnet/nimcp_neuralnet.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_types/nimcp_neuron_types.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_types/nimcp_neural_logic.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain_regions/nimcp_brain_regions.c

    # Multi-modal integration (Phase 8: Unified Architecture)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/integration/nimcp_multimodal_integration.c

    # Neuron models - Plugin architecture
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_models/nimcp_neuron_model.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_models/nimcp_izhikevich.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_models/nimcp_two_compartment.c

    # Synapse computation - NIMCP 2.7 MAJOR FEATURE
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/synapse_compute/nimcp_synapse_compute.c

    # Synapse type system - NIMCP 2.8.7 (Phase 8.7) - BIOLOGICAL DIVERSITY
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/synapse_types/nimcp_synapse_types.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuralnet/nimcp_synapse_embeddings.c

    # Fractal Topology - NIMCP 2.7 (scale-free networks)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/topology/nimcp_fractal_topology.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/topology/nimcp_network_builder.c

    # NLP Integration - NIMCP 2.7 (combines synapses + attention + neuromodulation)
    ${CMAKE_CURRENT_SOURCE_DIR}/../nlp/nimcp_nlp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../nlp/nimcp_spike_nlp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../nlp/nimcp_multimodal_nlp_bridge.c

    # Cognitive functions
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/ethics/nimcp_ethics.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/knowledge/nimcp_knowledge.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/curiosity/nimcp_curiosity.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/introspection/nimcp_introspection.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/wellbeing/nimcp_wellbeing.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/consolidation/nimcp_consolidation.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/salience/nimcp_salience.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/logic/nimcp_symbolic_logic.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/epistemic/nimcp_epistemic_filter.c  # Phase 9.2: Bias prevention
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/working_memory/nimcp_working_memory.c  # Phase 10.1: Working Memory
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/emotional_tagging/nimcp_emotional_tagging.c  # Phase 10.2: Emotional Tagging
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/emotions/nimcp_emotional_system.c  # Phase 10.2: Integrated Emotional System
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/executive/nimcp_executive.c  # Phase 10.3: Executive Functions
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/sleep_wake/nimcp_sleep_wake.c  # Phase 10.4: Sleep/Wake Cycle
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/mental_health/nimcp_mental_health.c  # Phase 10.5: Mental Health Monitoring
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/emotion_recognition/nimcp_emotion_recognition_simple.c  # Phase 11: Part I.1: Emotion Recognition (Simple)
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/empathetic_response/nimcp_empathetic_response.c  # Phase 11: Part I.2: Empathetic Response
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/autobiographical_memory/nimcp_autobiographical_memory.c  # Phase 12: Autobiographical Memory
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/self_model/nimcp_self_model.c  # Phase 12: Explicit Self-Model
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/personality/nimcp_personality.c  # Phase 12: Personality, Gender, and Sexual Identity
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/grief/nimcp_grief_and_loss.c  # Phase E1: Grief, Loss, and Death Understanding
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/joy/nimcp_joy_euphoria.c  # Phase E2: Joy, Euphoria, and Value-Aligned Success
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/remorse/nimcp_remorse_regret.c  # Phase E3: Remorse, Regret, and Evaluative Emotions
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/social/nimcp_love_loyalty_friendship.c  # Phase E4: Love, Loyalty, and Friendship
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/shadow/nimcp_shadow_emotions.c  # Phase E5: Shadow Emotions (Jealousy, Envy, Obsession, Hubris, Greed, Narcissism)
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/bias/nimcp_bias_detection.c  # Phase E6: Bias Detection (Racial, LGBTQ+, Gender, Misogyny, Age, Disability, Religious, Socioeconomic)
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/self_awareness/nimcp_self_awareness_extended.c  # Phase 12: Advanced Self-Awareness (metacognition, agency, safety)
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/theory_of_mind/nimcp_theory_of_mind.c  # Phase 10.6: Theory of Mind
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/explanations/nimcp_explanations.c  # Phase 10.7: Natural Explanations
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/meta_learning/nimcp_meta_learning.c  # Phase 10.8: Meta-Learning
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/predictive/nimcp_predictive.c  # Phase 10.9: Predictive Processing
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/mirror_neurons/nimcp_mirror_neurons.c  # Phase 10.11: Mirror Neurons
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/global_workspace/nimcp_global_workspace.c  # Global Workspace Architecture (Global Workspace Theory)
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/analysis/nimcp_network_analysis.c  # Network topology analysis and community detection
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/memory/nimcp_engram.c  # Phase M1: Memory Engrams (explicit memory traces)
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/memory/nimcp_systems_consolidation.c  # Phase M2: Systems Consolidation (hippocampus → cortex transfer)
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/memory/nimcp_wm_transfer.c  # Phase M3: Working Memory Transfer (WM → engram encoding)
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/memory/nimcp_semantic_memory.c  # Phase M4: Semantic Memory Network (concept network with spreading activation)
    ${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/nimcp_fractal_cognitive.c  # Fractal cognitive processing
    ${CMAKE_CURRENT_SOURCE_DIR}/cognitive/nimcp_hierarchical.c

    # Plasticity mechanisms
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/adaptive/nimcp_adaptive.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/bcm/nimcp_bcm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/neuromodulators/nimcp_neuromodulators.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/neuromodulators/nimcp_neuromod_pink_noise.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/neuromodulators/nimcp_spatial_neuromod.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/neuromodulators/nimcp_receptor_subtypes.c  # Phase C2.2 Enhancement #1: Receptor subtypes
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/neuromodulators/nimcp_phasic_tonic.c  # Phase C2.2 Enhancement #2: Phasic-tonic dynamics
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/neuromodulators/nimcp_vesicle_packaging.c  # Phase C2.3 Enhancement #3: Synaptic vesicle packaging
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/neuromodulators/nimcp_metabolic_pathways.c  # Phase C2.4 Enhancement #4: Metabolic pathways
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/attention/nimcp_attention.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/stp/nimcp_stp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/noise/nimcp_pink_noise.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/stdp/nimcp_stdp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../plasticity/eligibility/nimcp_eligibility_trace.c

    # Glial cells
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/astrocytes/nimcp_astrocytes.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/astrocytes/nimcp_astrocyte_calcium.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/astrocyte_types/nimcp_astrocyte_types.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/oligodendrocytes/nimcp_oligodendrocytes.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/microglia/nimcp_microglia.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../glial/integration/nimcp_glial_integration.c

    # Networking
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/p2p/nimcp_p2pnode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/protocol/nimcp_protocol.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/events/nimcp_events.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/replication/nimcp_replication.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../networking/distributed/nimcp_distributed_cognition.c

    # I/O
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/dataio/nimcp_dataio.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/stream/nimcp_stream.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/serialization/nimcp_serialization.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/serialization/nimcp_network_serialization.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../io/serialization/nimcp_encryption.c

    # Security
    ${CMAKE_CURRENT_SOURCE_DIR}/../security/nimcp_security.c

    # Utils - Memory & Thread
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/memory/nimcp_memory.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/memory/nimcp_memory_guards.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/thread/nimcp_thread.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/thread/nimcp_thread_pool.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/thread/nimcp_deadlock_detector.c

    # Utils - Platform Abstraction (Phase 2: Cross-platform support, SRP-compliant)
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/platform/nimcp_platform.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/platform/nimcp_platform_mutex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/platform/nimcp_platform_thread.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/platform/nimcp_platform_cond.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/platform/nimcp_platform_rwlock.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/platform/nimcp_platform_once.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/platform/nimcp_platform_time.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/platform/nimcp_system_resources.c  # Phase 2.8: Hardware resource detection

    # Utils - Logging
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/logging/nimcp_logging.c

    # Utils - Signal Handling (Runtime Safety)
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/signal/nimcp_signal_handler.c

    # Utils - Fault Tolerance & Recovery (Production Resilience - Complete System)
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/fault_tolerance/nimcp_recovery.c                      # Recovery strategy execution
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/fault_tolerance/nimcp_diagnostics.c                   # Error detection & classification
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/fault_tolerance/nimcp_checkpoint.c                    # State persistence & checkpointing
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/fault_tolerance/nimcp_health_monitor.c                # Runtime health monitoring
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/fault_tolerance/nimcp_brain_recovery_integration.c    # Brain-driven intelligent recovery
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/fault_tolerance/nimcp_runtime_adaptation.c            # Runtime parameter adaptation

    # Utils - Error Codes (Structured Error System)
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/error/nimcp_error_codes.c

    # Utils - Spatial Indexing (KD-Tree)
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/spatial/nimcp_kdtree.c

    # Utils - Dynamic Configuration (Runtime Safety)
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/config/nimcp_dynamic_config.c

    # Utils - Containers
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_vector.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_hash_table.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_btree.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_queue.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_graph.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/containers/nimcp_min_heap.c

    # Utils - Algorithms (Graph metrics for brain topology validation)
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/topology/nimcp_community_detection.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/algorithms/nimcp_centrality.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/algorithms/nimcp_graph_metrics.c

    # Utils - Other
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/time/nimcp_time.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/validation/nimcp_validate.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/queue_manager/nimcp_queue_manager.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/json/nimcp_json.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/config/nimcp_config.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/metrics/nimcp_metrics.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cache/nimcp_cache.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/numerical/nimcp_integration.c

    # Utils - Spectral Analysis
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/spectral/nimcp_fft.c

    # Utils - Quantum-Inspired Algorithms (Phase C: Quantum-Inspired Algorithms)
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/quantum/nimcp_quantum_walk.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/quantum/nimcp_quantum_shannon.c  # Phase C4.1: Quantum Walk + Shannon Information Theory

    # Utils - Geometry (Phase B1.1: Hyperbolic Knowledge Graph Embeddings)
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/geometry/nimcp_hyperbolic.c

    # Utils - Tensor Networks (Phase C3.1: MPS Compression)
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/tensor_networks/nimcp_mps.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/tensor_networks/nimcp_svd_lapack.c  # LAPACK SVD for tensor compression

    # Information Theory (Phase C4: Shannon Information Theory)
    ${CMAKE_CURRENT_SOURCE_DIR}/../information/nimcp_shannon.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../information/nimcp_cross_modal.c  # Phase C4.7: Cross-Modal Information Flow

    # Optimization - Quantum Annealing (Phase 11 Enhancement C1.1)
    ${CMAKE_CURRENT_SOURCE_DIR}/../optimization/quantum_annealing/nimcp_quantum_annealing.c

    # Brain Oscillations
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/brain_oscillations/nimcp_brain_oscillations.c

    # Perception
    ${CMAKE_CURRENT_SOURCE_DIR}/perception/nimcp_visual_cortex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/perception/nimcp_audio_cortex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/perception/nimcp_speech_cortex.c

    # GPU Support (CPU fallback when CUDA unavailable)
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/execution/nimcp_execution_mode.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/spike_event/nimcp_spike_event.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/neuron/nimcp_gpu_neuron.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/nimcp_multigpu.c

    # GPU CUDA kernels (only compiled if CUDA is available)
    $<$<BOOL:${CUDAToolkit_FOUND}>:${CMAKE_CURRENT_SOURCE_DIR}/../gpu/neuron/nimcp_gpu_kernels.cu>
    $<$<BOOL:${CUDAToolkit_FOUND}>:${CMAKE_CURRENT_SOURCE_DIR}/../gpu/synapse_compute/nimcp_synapse_compute_gpu.cu>
    $<$<BOOL:${CUDAToolkit_FOUND}>:${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_types/nimcp_neural_logic_kernels.cu>

    # Note: CUDA compile flags set below per-file

    # Python bindings
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_module.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_types.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_metrics_py.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_topology_py.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_pink_noise_py.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_community_py.c

    # Middleware - Buffering
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/buffering/nimcp_circular_buffer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/buffering/nimcp_sliding_window.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/buffering/nimcp_temporal_accumulator.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/buffering/nimcp_integration_buffer.c

    # Middleware - Normalization
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/normalization/nimcp_zscore_normalizer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/normalization/nimcp_min_max_normalizer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/normalization/nimcp_adaptive_normalizer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/normalization/nimcp_homeostatic_normalizer.c

    # Middleware - Encoding
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/encoding/nimcp_rate_coding.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/encoding/nimcp_temporal_coding.c

    # Middleware - Events
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/events/nimcp_event_bus.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/events/nimcp_event_queue.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/events/nimcp_event_subscriber.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/events/nimcp_event_types.c

    # Middleware - Patterns
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/patterns/nimcp_synchrony_detector.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/patterns/nimcp_sequence_detector.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/patterns/nimcp_oscillation_detector.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/patterns/nimcp_pattern_library.c

    # Middleware - Routing
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/routing/nimcp_thalamic_router.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/routing/nimcp_attention_gate.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/routing/nimcp_routing_table.c

    # Middleware - Pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/pipeline/nimcp_middleware_context.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/pipeline/nimcp_middleware_pipeline.c

    # Middleware - Integration
    ${CMAKE_CURRENT_SOURCE_DIR}/../middleware/brain_integration.c
)

# Core library
add_library(nimcp SHARED
    ${NIMCP_CORE_SOURCES}
)

# Update include directories to point to new structure
target_include_directories(nimcp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/nimcp>
    PRIVATE
        ${NIMCP_LIB_DIR}
        ${Python3_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/gpu
)

# Add CUDA include directories if available
if(CUDAToolkit_FOUND)
    target_include_directories(nimcp PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

    # Set CUDA properties
    set_target_properties(nimcp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 14
        CUDA_ARCHITECTURES "75;80;86;89"  # Turing, Ampere, Ada Lovelace
    )

    # NVCC doesn't support some GCC flags - set per-file compile options
    # Reset incompatible flags and add only nvcc-compatible ones
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/neuron/nimcp_gpu_kernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../gpu/synapse_compute/nimcp_synapse_compute_gpu.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/neuron_types/nimcp_neural_logic_kernels.cu
        PROPERTIES
        COMPILE_OPTIONS "--expt-relaxed-constexpr;-Xcompiler=-fPIC,-fstack-protector-strong,-Wformat,-Wformat-security,-fno-common"
        COMPILE_FLAGS ""
    )
endif()

# Add libsodium include directories if available
if(SODIUM_FOUND)
    target_include_directories(nimcp PRIVATE ${SODIUM_INCLUDE_DIR})
endif()

target_link_libraries(nimcp
    PUBLIC
        nimcp_middleware
        Python3::Python
        lz4
        jansson
        cjson
)

# Add LAPACK/BLAS libraries if available
if(LAPACK_BLAS_FOUND)
    target_link_libraries(nimcp PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

# Add CUDA libraries if available
if(CUDAToolkit_FOUND)
    target_link_libraries(nimcp PUBLIC CUDA::cudart)
endif()

# Add libsodium library if available
if(SODIUM_FOUND)
    target_link_libraries(nimcp PUBLIC ${SODIUM_LIBRARY})
endif()

# Add testing API support
target_compile_definitions(nimcp PRIVATE NIMCP_TESTING)

# Explicitly add -fPIC for shared library with thread-local storage
target_compile_options(nimcp PRIVATE -fPIC)

# Enable Position Independent Code and set library properties
set_target_properties(nimcp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ENABLE_EXPORTS ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 2
    OUTPUT_NAME nimcp
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# Installation rules
install(TARGETS nimcp
    EXPORT nimcp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
