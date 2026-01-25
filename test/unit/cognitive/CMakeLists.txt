# Unit tests for cognitive module bio-async integration

# Cognitive bio-async integration tests
add_executable(unit_cognitive_test_bio_async
    test_cognitive_bio_async.cpp
)

target_link_libraries(unit_cognitive_test_bio_async
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_bio_async
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_bio_async PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_bio_async
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;bio-async"
        TIMEOUT 90
)

# Cognitive bio-async registration tests
add_executable(unit_cognitive_test_bio_async_registration
    test_cognitive_bio_async_registration.cpp
)
target_link_libraries(unit_cognitive_test_bio_async_registration
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(unit_cognitive_test_bio_async_registration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)
gtest_discover_tests(unit_cognitive_test_bio_async_registration)

# Cognitive Meta-Controller Tests
add_executable(unit_cognitive_test_meta_controller
    test_cognitive_meta_controller.cpp
)

target_link_libraries(unit_cognitive_test_meta_controller
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_meta_controller
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_meta_controller PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_meta_controller
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;meta_controller"
        TIMEOUT 120
)

# Emotion-Attention Integration Tests
# DISABLED: Source file nimcp_emotion_attention.c not yet implemented
message(STATUS "  [SKIP] test_emotion_attention.cpp - source file not implemented")

# Emotion-Consolidation Integration Tests
# DISABLED: Source file nimcp_emotion_consolidation.c not yet implemented
message(STATUS "  [SKIP] test_emotion_consolidation.cpp - source file not implemented")

# Working Memory subdirectory (PE tests)
add_subdirectory(working_memory)

# Wellbeing tests
add_subdirectory(wellbeing)

# Mental Health tests
add_subdirectory(mental_health)

# Working Memory - Immune Integration Tests
add_executable(unit_cognitive_test_working_memory_immune
    test_working_memory_immune_integration.cpp
)

target_link_libraries(unit_cognitive_test_working_memory_immune
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_working_memory_immune
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_working_memory_immune PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_working_memory_immune
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;working_memory;immune"
        TIMEOUT 120
)

# Brain Immune System tests - Added via test/CMakeLists.txt to avoid duplicate
# add_subdirectory(immune)

# Predictive-Immune Integration tests
add_subdirectory(predictive_immune)

# Free Energy Principle Enhancement tests
add_subdirectory(free_energy)

# Autobiographical Memory tests
add_subdirectory(autobiographical_memory)

# Global Workspace - Immune Integration Tests
add_executable(unit_cognitive_test_global_workspace_immune
    global_workspace/test_global_workspace_immune_integration.cpp
)

target_link_libraries(unit_cognitive_test_global_workspace_immune
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_global_workspace_immune
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_global_workspace_immune PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_global_workspace_immune
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;global_workspace;immune"
        TIMEOUT 120
)

# Curiosity - Immune Integration Tests
add_executable(unit_cognitive_test_curiosity_immune
    curiosity/test_curiosity_immune_integration.cpp
)

target_link_libraries(unit_cognitive_test_curiosity_immune
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_curiosity_immune
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_curiosity_immune PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_curiosity_immune
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;curiosity;immune"
        TIMEOUT 120
)

# Curiosity - FEP Bridge Tests
add_executable(unit_cognitive_test_curiosity_fep_bridge
    curiosity/test_curiosity_fep_bridge.cpp
)

target_link_libraries(unit_cognitive_test_curiosity_fep_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_curiosity_fep_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_curiosity_fep_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_curiosity_fep_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;curiosity;fep"
        TIMEOUT 120
)

# ============================================================================
# Enhanced Curiosity Tests (10 Enhancements)
# ============================================================================
# WHAT: Unit tests for Enhanced Curiosity System
# WHY:  Validate 10 curiosity enhancements (boredom, interest decay, types,
#       anxiety balance, social, meta, contagion, surprise, fatigue, counterfactual)
# HOW:  Test each enhancement plus integration, thread safety, edge cases
# ============================================================================

add_executable(unit_cognitive_test_curiosity_enhanced
    curiosity/test_curiosity_enhanced.cpp
)

target_link_libraries(unit_cognitive_test_curiosity_enhanced
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_curiosity_enhanced
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_curiosity_enhanced PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_curiosity_enhanced
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;curiosity;enhanced"
        TIMEOUT 180
)

# ============================================================================
# Curiosity SNN Bridge Tests
# ============================================================================
# WHAT: Unit tests for Curiosity-SNN bidirectional integration
# WHY:  Verify curiosity dimension encoding, novelty detection, exploration drive
# HOW:  Test lifecycle, encoding, simulation, detection, callbacks
# ============================================================================

add_executable(unit_cognitive_test_curiosity_snn_bridge
    curiosity/test_curiosity_snn_bridge.cpp
)

target_link_libraries(unit_cognitive_test_curiosity_snn_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_curiosity_snn_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_curiosity_snn_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_curiosity_snn_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;curiosity;snn"
        TIMEOUT 120
)

# ============================================================================
# Curiosity Plasticity Bridge Tests
# ============================================================================
# WHAT: Unit tests for Curiosity-Plasticity bidirectional integration
# WHY:  Verify learning, STDP, synapse protection, reward modulation
# HOW:  Test lifecycle, synapse management, learning, consolidation
# ============================================================================

add_executable(unit_cognitive_test_curiosity_plasticity_bridge
    curiosity/test_curiosity_plasticity_bridge.cpp
)

target_link_libraries(unit_cognitive_test_curiosity_plasticity_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_curiosity_plasticity_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(unit_cognitive_test_curiosity_plasticity_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_curiosity_plasticity_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;curiosity;plasticity"
        TIMEOUT 120
)

# ============================================================================
# Cognitive FEP Bridge Tests (Comprehensive)
# ============================================================================
# WHAT: Unit tests for Cognitive-FEP bidirectional integration bridges
# WHY:  Ensure cognitive systems integrate with FEP predictions and precision
# HOW:  Test lifecycle, effects, and bio-async for consolidation, hierarchical,
#       meta-learning, personality, wellbeing, sleep-wake, and brain immune
# ============================================================================

add_executable(unit_cognitive_test_fep_bridges
    test_cognitive_fep_bridges.cpp
)

target_link_libraries(unit_cognitive_test_fep_bridges
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_fep_bridges
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_fep_bridges PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_fep_bridges
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;fep;bridge"
        TIMEOUT 120
)

# ============================================================================
# Sleep Bridge Integration Tests
# ============================================================================
# WHAT: Unit tests for Sleep-Cognitive integration bridges
# WHY:  Ensure cognitive functions correctly modulate based on sleep state
# HOW:  Test helper functions, default config, biological validity
# ============================================================================

# Attention Sleep Bridge Tests
add_executable(unit_cognitive_attention_sleep_bridge
    attention/sleep/test_attention_sleep_bridge.cpp
)
target_link_libraries(unit_cognitive_attention_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_cognitive_attention_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_cognitive_attention_sleep_bridge
    PROPERTIES LABELS "unit;cognitive;attention;sleep")

# Working Memory Sleep Bridge Tests
add_executable(unit_cognitive_working_memory_sleep_bridge
    working_memory/sleep/test_working_memory_sleep_bridge.cpp
)
target_link_libraries(unit_cognitive_working_memory_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_cognitive_working_memory_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_cognitive_working_memory_sleep_bridge
    PROPERTIES LABELS "unit;cognitive;working_memory;sleep")

# Executive Sleep Bridge Tests
add_executable(unit_cognitive_executive_sleep_bridge
    executive/sleep/test_executive_sleep_bridge.cpp
)
target_link_libraries(unit_cognitive_executive_sleep_bridge
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_compile_features(unit_cognitive_executive_sleep_bridge PRIVATE cxx_std_17)
gtest_discover_tests(unit_cognitive_executive_sleep_bridge
    PROPERTIES LABELS "unit;cognitive;executive;sleep")

# ============================================================================
# Substrate Bridge Tests (Phase 6)
# ============================================================================
# WHAT: Unit tests for Neural Substrate integration bridges
# WHY:  Ensure cognitive systems correctly modulate based on metabolic state
# HOW:  Test ATP, temperature, fatigue effects on cognitive functions
# ============================================================================

# Attention tests (includes substrate bridge)
add_subdirectory(attention)

# Emotion tests (includes substrate bridge)
add_subdirectory(emotion)

# Emotions tests (emotional system thread safety)
add_subdirectory(emotions)

# Ethics tests (SNN/plasticity bridges)
add_subdirectory(ethics)

# Executive tests (includes substrate bridge)
add_subdirectory(executive)

# Note: introspection and reasoning directories are added from test/CMakeLists.txt
# Their substrate subdirectories are included via those parent CMakeLists

# Memory tests (includes substrate bridge)
add_subdirectory(memory)

# Theory of Mind tests (includes substrate bridge)
add_subdirectory(tom)

# Theory of Mind SNN/Plasticity bridge tests
add_subdirectory(theory_of_mind)

# Parietal Lobe tests (Phase 7.2 - Mathematical/Scientific Reasoning)
add_subdirectory(parietal)

# Meta Learning SNN/Plasticity bridge tests
add_subdirectory(meta_learning)

# Note: introspection and mirror_neurons subdirectories are added from
# test/CMakeLists.txt to avoid duplicate target issues

# Self Model SNN/Plasticity bridge tests
add_subdirectory(self_model)

# ============================================================================
# Self-Awareness Coordinator Tests (Phase 1: Self-Awareness Unification)
# ============================================================================
# WHAT: Unit tests for the Self-Awareness Coordinator and Feedback System
# WHY:  Ensure unified self-awareness components work correctly
# HOW:  Test coordinator lifecycle, feedback loops, coherence, phi monitoring
# ============================================================================

add_executable(unit_cognitive_self_awareness_coordinator
    test_self_awareness_coordinator.cpp
)
target_link_libraries(unit_cognitive_self_awareness_coordinator
    PRIVATE nimcp GTest::gtest GTest::gtest_main Threads::Threads m
)
target_include_directories(unit_cognitive_self_awareness_coordinator
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)
target_compile_features(unit_cognitive_self_awareness_coordinator PRIVATE cxx_std_17)
gtest_discover_tests(unit_cognitive_self_awareness_coordinator
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;self-awareness;coordinator"
        TIMEOUT 60
)

# ============================================================================
# JEPA Latent Space Tests
# ============================================================================
# WHAT: Unit tests for JEPA Latent Space Representation Module
# WHY:  Latent space is core to JEPA prediction/comparison - must be correct
# HOW:  Test creation, similarity, interpolation, normalization, pooling
# ============================================================================

add_executable(unit_cognitive_jepa_latent
    jepa/test_jepa_latent.cpp
)

target_link_libraries(unit_cognitive_jepa_latent
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_jepa_latent
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_jepa_latent PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_jepa_latent
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;jepa;latent"
        TIMEOUT 120
)

# ============================================================================
# JEPA Masking Tests
# ============================================================================
# WHAT: Unit tests for JEPA Masking Module
# WHY:  Validate all masking strategies (random, block, attention, curriculum,
#       tube, causal) and mask operations for self-supervised learning
# HOW:  Test lifecycle, all strategies, mask operations, edge cases
# ============================================================================

add_executable(unit_cognitive_jepa_masking
    jepa/test_jepa_masking.cpp
)

target_link_libraries(unit_cognitive_jepa_masking
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_jepa_masking
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_jepa_masking PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_jepa_masking
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;jepa;masking"
        TIMEOUT 120
)

# ============================================================================
# JEPA Multimodal Tests
# ============================================================================
# WHAT: Unit tests for Multimodal JEPA module
# WHY:  Validate joint embedding space for visual and speech modalities,
#       cross-modal prediction, alignment training, and fusion strategies
# HOW:  Test creation, encoding, fusion (all types), similarity, prediction,
#       alignment training, batch management, edge cases
# ============================================================================

add_executable(unit_cognitive_jepa_multimodal
    jepa/test_jepa_multimodal.cpp
)

target_link_libraries(unit_cognitive_jepa_multimodal
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_jepa_multimodal
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_jepa_multimodal PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_jepa_multimodal
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;jepa;multimodal"
        TIMEOUT 180
)

# ============================================================================
# JEPA Predictor Tests
# ============================================================================
# WHAT: Unit tests for JEPA Predictor module
# WHY:  Predictor is core to JEPA - predicts embeddings in latent space
# HOW:  Test creation, MLP forward pass, error computation, training, weights
# ============================================================================

add_executable(unit_cognitive_test_jepa_predictor
    jepa/test_jepa_predictor.cpp
)

target_link_libraries(unit_cognitive_test_jepa_predictor
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_jepa_predictor
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_jepa_predictor PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_jepa_predictor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;jepa;predictor"
        TIMEOUT 120
)

# ============================================================================
# JEPA Weights Tests
# ============================================================================
# WHAT: Unit tests for JEPA Weight Loading and Management Module
# WHY:  Weight loading/saving is critical for transfer learning from V-JEPA 2
# HOW:  Test file format (NJWT magic), load/save, round-trip, validation
# ============================================================================

add_executable(unit_cognitive_jepa_weights
    jepa/test_jepa_weights.cpp
)

target_link_libraries(unit_cognitive_jepa_weights
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_jepa_weights
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_definitions(unit_cognitive_jepa_weights PRIVATE NIMCP_TESTING)
target_compile_features(unit_cognitive_jepa_weights PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_jepa_weights
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;jepa;weights"
        TIMEOUT 120
)

# ============================================================================
# JEPA Context Tests
# ============================================================================
# WHAT: Unit tests for JEPA Context Encoder module
# WHY:  Context-conditioned encoding enables task-specific representations
# HOW:  Test creation, context setting, encoding with FiLM/cross-attention,
#       bio-async integration, statistics tracking
# ============================================================================

add_executable(unit_cognitive_test_jepa_context
    jepa/test_jepa_context.cpp
)

target_link_libraries(unit_cognitive_test_jepa_context
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_jepa_context
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_definitions(unit_cognitive_test_jepa_context PRIVATE NIMCP_TESTING)
target_compile_features(unit_cognitive_test_jepa_context PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_jepa_context
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;jepa;context"
        TIMEOUT 180
)

# ============================================================================
# Game Theory Tests
# ============================================================================
# WHAT: Unit tests for Game Theory module
# WHY:  Validate auction mechanisms, bargaining solutions, credit assignment
# HOW:  Test auctions (second-price, VCG), Nash bargaining, Shapley values
# ============================================================================

# Game Theory - Auction Tests
add_executable(unit_cognitive_test_game_theory_auction
    game_theory/test_auction.cpp
)

target_link_libraries(unit_cognitive_test_game_theory_auction
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_game_theory_auction
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_game_theory_auction PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_game_theory_auction
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;auction"
        TIMEOUT 120
)

# Game Theory - Bargaining Tests
add_executable(unit_cognitive_test_game_theory_bargaining
    game_theory/test_bargaining.cpp
)

target_link_libraries(unit_cognitive_test_game_theory_bargaining
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_game_theory_bargaining
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_game_theory_bargaining PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_game_theory_bargaining
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;bargaining"
        TIMEOUT 120
)

# Game Theory - Credit Assignment Tests
add_executable(unit_cognitive_test_game_theory_credit
    game_theory/test_credit_assignment.cpp
)

target_link_libraries(unit_cognitive_test_game_theory_credit
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_game_theory_credit
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_game_theory_credit PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_game_theory_credit
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;credit"
        TIMEOUT 120
)

# Game Theory - Integration Tests
add_executable(unit_cognitive_test_game_theory_integration
    game_theory/test_gt_integration.cpp
)

target_link_libraries(unit_cognitive_test_game_theory_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_game_theory_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_game_theory_integration PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_game_theory_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;integration"
        TIMEOUT 120
)

# ============================================================================
# Advanced Game Theory Module Tests
# ============================================================================
# WHAT: Unit tests for advanced game theory modules
# WHY:  Validate equilibrium computation, learning, coalition formation,
#       repeated games, fairness metrics, mechanism design, extended auctions,
#       and spatial/evolutionary games
# ============================================================================

# Game Theory - Equilibrium Tests
add_executable(unit_cognitive_test_gt_equilibrium
    game_theory/test_gt_equilibrium.cpp
)

target_link_libraries(unit_cognitive_test_gt_equilibrium
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_gt_equilibrium
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_gt_equilibrium PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_gt_equilibrium
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;equilibrium"
        TIMEOUT 180
)

# Game Theory - Learning Tests
add_executable(unit_cognitive_test_gt_learning
    game_theory/test_gt_learning.cpp
)

target_link_libraries(unit_cognitive_test_gt_learning
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_gt_learning
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_gt_learning PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_gt_learning
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;learning"
        TIMEOUT 180
)

# Game Theory - Coalition Tests
add_executable(unit_cognitive_test_gt_coalition
    game_theory/test_gt_coalition.cpp
)

target_link_libraries(unit_cognitive_test_gt_coalition
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_gt_coalition
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_gt_coalition PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_gt_coalition
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;coalition"
        TIMEOUT 180
)

# Game Theory - Repeated Games Tests
add_executable(unit_cognitive_test_gt_repeated
    game_theory/test_gt_repeated.cpp
)

target_link_libraries(unit_cognitive_test_gt_repeated
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_gt_repeated
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_gt_repeated PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_gt_repeated
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;repeated"
        TIMEOUT 180
)

# Game Theory - Fairness Tests
add_executable(unit_cognitive_test_gt_fairness
    game_theory/test_gt_fairness.cpp
)

target_link_libraries(unit_cognitive_test_gt_fairness
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_gt_fairness
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_gt_fairness PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_gt_fairness
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;fairness"
        TIMEOUT 180
)

# Game Theory - Mechanism Design Tests
add_executable(unit_cognitive_test_gt_mechanism
    game_theory/test_gt_mechanism.cpp
)

target_link_libraries(unit_cognitive_test_gt_mechanism
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_gt_mechanism
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_gt_mechanism PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_gt_mechanism
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;mechanism"
        TIMEOUT 180
)

# Game Theory - Extended Auction Tests
add_executable(unit_cognitive_test_gt_auction_ext
    game_theory/test_gt_auction_ext.cpp
)

target_link_libraries(unit_cognitive_test_gt_auction_ext
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_gt_auction_ext
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_gt_auction_ext PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_gt_auction_ext
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;auction_ext"
        TIMEOUT 180
)

# Game Theory - Spatial/Evolutionary Games Tests
add_executable(unit_cognitive_test_gt_spatial
    game_theory/test_gt_spatial.cpp
)

target_link_libraries(unit_cognitive_test_gt_spatial
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_gt_spatial
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_gt_spatial PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_gt_spatial
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;spatial"
        TIMEOUT 180
)

# Game Theory - FEP Bridge Tests
add_executable(unit_cognitive_test_gt_fep_bridge
    game_theory/test_game_theory_fep_bridge.cpp
)

target_link_libraries(unit_cognitive_test_gt_fep_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_gt_fep_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_gt_fep_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_gt_fep_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory;fep"
        TIMEOUT 180
)

# ============================================================================
# Phase 7 Substrate/Thalamic Bridge Tests
# ============================================================================
# WHAT: Unit tests for Phase 7 cognitive bridges
# WHY:  Validate substrate and thalamic bridges for additional cognitive modules
#       (consolidation, emotion_tensor, empathetic_response, fault_tolerance,
#        fractal_cognitive, free_energy, game_theory, brain_immune, social,
#        theory_of_mind, predictive_immune, self_awareness_extended)
# HOW:  Test lifecycle, effects computation, attention gating, signal routing
# ============================================================================

add_executable(unit_cognitive_test_phase7_bridges
    test_phase7_bridges.cpp
)

target_link_libraries(unit_cognitive_test_phase7_bridges
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_phase7_bridges
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_phase7_bridges PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_phase7_bridges
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;bridges;phase7"
        TIMEOUT 120
)

# ============================================================================
# Cognitive Thalamic Bridges Tests
# ============================================================================
add_executable(unit_cognitive_test_cognitive_thalamic_bridges
    test_cognitive_thalamic_bridges.cpp
)

target_link_libraries(unit_cognitive_test_cognitive_thalamic_bridges
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_cognitive_thalamic_bridges
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_cognitive_thalamic_bridges PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_cognitive_thalamic_bridges
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;thalamic;bridges"
        TIMEOUT 120
)

# ============================================================================
# Consolidation Tests
# ============================================================================
add_executable(unit_cognitive_test_consolidation
    test_consolidation.cpp
)

target_link_libraries(unit_cognitive_test_consolidation
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_consolidation
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_consolidation PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_consolidation
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;consolidation"
        TIMEOUT 120
)

# ============================================================================
# Emotion Tensor Tests
# ============================================================================
add_executable(unit_cognitive_test_emotion_tensor
    test_emotion_tensor.cpp
)

target_link_libraries(unit_cognitive_test_emotion_tensor
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_emotion_tensor
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_emotion_tensor PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_emotion_tensor
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;emotion;tensor"
        TIMEOUT 120
)

# ============================================================================
# Empathetic Response Tests
# ============================================================================
add_executable(unit_cognitive_test_empathetic_response
    test_empathetic_response.cpp
)

target_link_libraries(unit_cognitive_test_empathetic_response
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_empathetic_response
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_empathetic_response PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_empathetic_response
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;empathetic;response"
        TIMEOUT 120
)

# ============================================================================
# Fault Tolerance Tests
# ============================================================================
add_executable(unit_cognitive_test_fault_tolerance
    test_fault_tolerance.cpp
)

target_link_libraries(unit_cognitive_test_fault_tolerance
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_fault_tolerance
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_fault_tolerance PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_fault_tolerance
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;fault_tolerance"
        TIMEOUT 120
)

# ============================================================================
# Fractal Cognitive Tests
# ============================================================================
add_executable(unit_cognitive_test_fractal_cognitive
    test_fractal_cognitive.cpp
)

target_link_libraries(unit_cognitive_test_fractal_cognitive
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_fractal_cognitive
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_fractal_cognitive PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_fractal_cognitive
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;fractal"
        TIMEOUT 120
)

# ============================================================================
# Free Energy Tests
# ============================================================================
add_executable(unit_cognitive_test_free_energy
    test_free_energy.cpp
)

target_link_libraries(unit_cognitive_test_free_energy
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_free_energy
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_free_energy PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_free_energy
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;free_energy"
        TIMEOUT 120
)

# ============================================================================
# Game Theory Main Tests
# ============================================================================
add_executable(unit_cognitive_test_game_theory
    test_game_theory.cpp
)

target_link_libraries(unit_cognitive_test_game_theory
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_game_theory
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_game_theory PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_game_theory
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;game_theory"
        TIMEOUT 120
)

# ============================================================================
# Predictive Immune Tests - DISABLED: requires nimcp_predictive_immune.c which is disabled
# ============================================================================
# add_executable(unit_cognitive_test_predictive_immune
#     test_predictive_immune.cpp
# )
#
# target_link_libraries(unit_cognitive_test_predictive_immune
#     PRIVATE
#         nimcp
#         GTest::gtest
#         GTest::gtest_main
#         Threads::Threads
#         m
# )
#
# target_include_directories(unit_cognitive_test_predictive_immune
#     PRIVATE
#         ${PROJECT_SOURCE_DIR}/include
#         ${PROJECT_SOURCE_DIR}/src
#         ${PROJECT_SOURCE_DIR}/test
# )
#
# target_compile_features(unit_cognitive_test_predictive_immune PRIVATE cxx_std_17)
#
# gtest_discover_tests(unit_cognitive_test_predictive_immune
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#     PROPERTIES
#         LABELS "unit;cognitive;predictive;immune"
#         TIMEOUT 120
# )

# ============================================================================
# Self-Awareness Extended Tests
# ============================================================================
add_executable(unit_cognitive_test_self_awareness_extended
    test_self_awareness_extended.cpp
)

target_link_libraries(unit_cognitive_test_self_awareness_extended
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_self_awareness_extended
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_self_awareness_extended PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_self_awareness_extended
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;self_awareness;extended"
        TIMEOUT 120
)

# ============================================================================
# Collective Cognition Tests
# ============================================================================
add_subdirectory(collective_cognition)

# ============================================================================
# Recursive Cognition Tests (RLM-inspired)
# ============================================================================
add_subdirectory(recursive)

# ============================================================================
# Social Cognition Tests
# ============================================================================
add_subdirectory(social)

# ============================================================================
# Social Tests
# ============================================================================
add_executable(unit_cognitive_test_social
    test_social.cpp
)

target_link_libraries(unit_cognitive_test_social
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_social
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_social PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_social
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;social"
        TIMEOUT 120
)

# ============================================================================
# Imagination Engine Tests
# ============================================================================
add_executable(unit_cognitive_test_imagination_engine
    imagination/test_imagination_engine.cpp
)

target_link_libraries(unit_cognitive_test_imagination_engine
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_imagination_engine
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_imagination_engine PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_imagination_engine
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;imagination"
        TIMEOUT 180
)

# ============================================================================
# Imagination FEP Bridge Tests
# ============================================================================
# WHAT: Unit tests for Imagination-FEP bidirectional integration
# WHY:  Verify free energy computation from simulation divergence and
#       counterfactual costs, prediction accuracy tracking
# HOW:  Test lifecycle, registration, metrics, callbacks, FEP update cycle
# ============================================================================

add_executable(unit_cognitive_test_imagination_fep_bridge
    imagination/test_imagination_fep_bridge.cpp
)

target_link_libraries(unit_cognitive_test_imagination_fep_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_imagination_fep_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_test_imagination_fep_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_imagination_fep_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;imagination;fep"
        TIMEOUT 120
)

# ============================================================================
# Omnidirectional Inference Tests (Phase 4)
# ============================================================================
# WHAT: Unit tests for Omnidirectional Inference components
# WHY:  Validate bidirectional JEPA, Hopfield memory, predictive hierarchy,
#       and temporal replay modules
# ============================================================================

# JEPA Bidirectional Tests
add_executable(unit_cognitive_test_jepa_bidirectional
    jepa/test_jepa_bidirectional.cpp
)

target_link_libraries(unit_cognitive_test_jepa_bidirectional
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_jepa_bidirectional
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_jepa_bidirectional PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_jepa_bidirectional
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;jepa;bidirectional;omni"
        TIMEOUT 120
)

# Hopfield Memory Tests
add_executable(unit_cognitive_test_hopfield_memory
    memory/test_hopfield_memory.cpp
)

target_link_libraries(unit_cognitive_test_hopfield_memory
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_hopfield_memory
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_hopfield_memory PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_hopfield_memory
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;memory;hopfield;omni"
        TIMEOUT 120
)

# Predictive Hierarchy Tests
add_executable(unit_cognitive_test_predictive_hierarchy
    predictive/test_predictive_hierarchy.cpp
)

target_link_libraries(unit_cognitive_test_predictive_hierarchy
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_predictive_hierarchy
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_predictive_hierarchy PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_predictive_hierarchy
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;predictive;hierarchy;omni"
        TIMEOUT 120
)

# Temporal Replay Tests
add_executable(unit_cognitive_test_temporal_replay
    memory/test_temporal_replay.cpp
)

target_link_libraries(unit_cognitive_test_temporal_replay
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_temporal_replay
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_temporal_replay PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_temporal_replay
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;memory;replay;omni"
        TIMEOUT 120
)

# Global Workspace - SNN Bridge Unit Tests
add_executable(unit_cognitive_gw_snn_bridge
    global_workspace/test_gw_snn_bridge.cpp
)

target_link_libraries(unit_cognitive_gw_snn_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_gw_snn_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_gw_snn_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_gw_snn_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;global_workspace;snn"
        TIMEOUT 120
)

# Global Workspace - Plasticity Bridge Unit Tests
add_executable(unit_cognitive_gw_plasticity_bridge
    global_workspace/test_gw_plasticity_bridge.cpp
)

target_link_libraries(unit_cognitive_gw_plasticity_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_gw_plasticity_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test
)

target_compile_features(unit_cognitive_gw_plasticity_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_gw_plasticity_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;global_workspace;plasticity"
        TIMEOUT 120
)

# ============================================================================
# Epistemic SNN/Plasticity Bridge Tests
# ============================================================================
add_subdirectory(epistemic)

# ============================================================================
# Cognitive Bias SNN/Plasticity Bridge Tests
# ============================================================================
add_subdirectory(bias)

# ============================================================================
# Salience SNN/Plasticity Bridge Tests
# ============================================================================
add_subdirectory(salience)

# ============================================================================
# Cognitive Integration Bridge Tests (Emotion-Executive, GW-Cognitive)
# ============================================================================
add_subdirectory(integration)

# ============================================================================
# JEPA FEP Bridge Tests
# ============================================================================
# WHAT: Unit tests for JEPA-FEP bidirectional integration
# WHY:  Verify free energy computation from embedding prediction error,
#       representation quality, and collapse detection
# HOW:  Test lifecycle, registration, metrics, callbacks, FEP update cycle
# ============================================================================

add_executable(unit_cognitive_test_jepa_fep_bridge
    jepa/test_jepa_fep_bridge.cpp
)

target_link_libraries(unit_cognitive_test_jepa_fep_bridge
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_jepa_fep_bridge
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_jepa_fep_bridge PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_jepa_fep_bridge
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;jepa;fep"
        TIMEOUT 120
)

# ============================================================================
# Symbolic Logic Integration Tests
# ============================================================================
# WHAT: Unit tests for symbolic logic cognitive layer integration
# WHY:  Verify FEP orchestrator and cognitive hub integration
# HOW:  Test hub bridge, FEP registration, inter-module communication
# ============================================================================

add_executable(unit_cognitive_test_symbolic_logic_integration
    symbolic_logic/test_symbolic_logic_integration.cpp
)

target_link_libraries(unit_cognitive_test_symbolic_logic_integration
    PRIVATE
        nimcp
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        m
)

target_include_directories(unit_cognitive_test_symbolic_logic_integration
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/test/utils
)

target_compile_features(unit_cognitive_test_symbolic_logic_integration PRIVATE cxx_std_17)

gtest_discover_tests(unit_cognitive_test_symbolic_logic_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        LABELS "unit;cognitive;symbolic_logic;integration"
        TIMEOUT 120
)

# ============================================================================
# Extrapolation / World Model Tests
# ============================================================================
# WHAT: Unit tests for Multi-Modal World Model
# WHY:  Verify modality processing, fusion, prediction, entity tracking
# HOW:  Test all modalities, fusion types, prediction modes, entity CRUD
# ============================================================================

add_subdirectory(extrapolation)

# ============================================================================
# Exception Handling Tests
# ============================================================================
# WHAT: Unit tests for Cognitive Exception Handling
# WHY:  Verify exception creation, dispatch, and immune integration for cognitive errors
# HOW:  Test error code mapping, handler chains, recovery strategies
# ============================================================================

add_subdirectory(exception)

# ============================================================================
# Omni World Model Tests
# ============================================================================
# WHAT: Unit tests for Omnidirectional Generative World Model
# WHY:  Verify counterfactual reasoning, mental simulation, RSSM, MDN,
#       experience replay, symlog transformation, dynamics prediction
# HOW:  Test all APIs: lifecycle, state, RSSM, dynamics, latent encoding,
#       MDN, counterfactual, rollouts, learning, experience replay, bio-async
# ============================================================================

add_subdirectory(omni)

# ============================================================================
# Cognitive Health Integration Tests (Phase 8)
# ============================================================================
# WHAT: Unit tests for cognitive health integration
# WHY:  Validate collective health, RCOG health, meta-health, cognitive bridge
# HOW:  Test lifecycle, API functions, intelligent handling, statistics
# ============================================================================

add_subdirectory(health)

# ============================================================================
# Neuro-Symbolic Mathematics Tests (Mathesis-Inspired)
# ============================================================================
# WHAT: Unit tests for genius-level mathematics system
# WHY:  Validate energy consistency, hypergraph, quantum MCTS, orchestrator
# HOW:  Test all APIs: lifecycle, operations, integration with other modules
# ============================================================================

add_subdirectory(neuro_symbolic)

# ============================================================================
# Predictive Bridge Exception Tests
# ============================================================================
# WHAT: Unit tests for predictive bridge exception handling
# WHY:  Verify NIMCP_THROW_TO_IMMUNE exception handling for all predictive bridges
# HOW:  Test NULL pointer, allocation failure, and bio-async checks
# ============================================================================

add_subdirectory(predictive)
