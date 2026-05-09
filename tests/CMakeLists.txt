# ============================================================================
# Standalone-C unit tests for the SNN-language stack
# ============================================================================
# WHAT: Wires the 21 .c smoke tests under tests/unit/ into ctest so they run
#       under `ctest -R "test_lang_|test_bigram|test_snn_lang|test_bulk_lex"`.
# WHY:  Until now these tests existed as ad-hoc files compiled by hand from
#       the doc-comment header. Each one was self-contained (printf + exit,
#       no GTest dep) which made them fast and portable, but invisible to CI.
#       Wiring them in lets every CMake build verify the SNN-language bridge
#       contracts in one shot.
# HOW:  Each test is a single .c file that links against libnimcp + libm +
#       pthread. We define a helper `add_nimcp_standalone_test()` that
#       creates the executable, sets RPATH so it can find libnimcp.so.2 at
#       runtime, registers it with `add_test()`, and tags it under the
#       "lang_smoke" ctest label.
#
#       The tests are NOT linked against GTest — they use printf+exit-code
#       reporting (exit 0 == pass). ctest reads the exit code so we get
#       the same PASS/FAIL signal as a GTest binary.
#
#       The 21 .c files are listed by name (not glob) so a missing file
#       fails configuration with a clear "test source not found" error
#       rather than silently disappearing.
# ============================================================================

set(NIMCP_STANDALONE_LANG_TESTS
    test_lang_bridge_cosine_norm
    test_lang_bridge_sampling
    test_lang_bridge_glove_blend
    test_lang_bridge_autoregressive
    test_lang_bridge_next_token
    test_lang_bridge_spike_routing
    test_lang_bridge_riemannian
    test_bigram_spectrum
    test_lang_bridge_hyperbolic_glove
    test_lang_bridge_quantum_sampling
    test_lang_bridge_quantum_shannon
    test_lang_bridge_beam_eos_repeat
    test_lang_bridge_config_persistence
    test_lang_phantom_wiring_audit
    test_lang_anaphora
    test_lang_negation_discourse_polysemy
    test_lang_bridge_get_config
    test_lang_bridge_latency
    test_lang_bridge_rng_seed
    test_snn_lang_bridge_stdp
    test_bulk_lexicon
    test_lang_multiturn_persistence
    test_lang_lgss_gates
    test_lang_trigram_learning
    test_lang_immune_inspection
    test_lang_spectrum_tick
    test_lang_da_modulation
    test_lang_reconsolidation
    test_lang_topic_shift
    test_lang_speech_acts
    test_lang_length_control
    test_lang_stream_produce
    test_lang_sentence_segmentation
    test_lang_tom_wiring
    test_lang_decode_bench
    test_lang_config_persist
    test_lang_consumer_events
    test_lang_subword_oov
    test_lang_coref
)

# Helper: register a single standalone-C test.
#
#   target_name : ctest target name (also the executable name)
#   source_file : absolute path to the .c file
#
# Each test:
#   - links nimcp + m + pthread (the same triplet the doc-comments specify)
#   - sets BUILD_RPATH and INSTALL_RPATH to ${CMAKE_BINARY_DIR}/lib so the
#     executable can locate libnimcp.so.2 at run time without LD_LIBRARY_PATH.
#     This matches the user-facing build pattern documented in the test file
#     headers ("-Wl,-rpath,/home/bbrelin/nimcp/build/lib").
#   - uses include path ${CMAKE_SOURCE_DIR}/include so headers like
#     "snn/bridges/nimcp_snn_language_bridge.h" resolve from the worktree.
#   - registers with add_test() under the "lang_smoke" label.
function(add_nimcp_standalone_test target_name source_file)
    if(NOT EXISTS "${source_file}")
        message(WARNING "Standalone test source not found: ${source_file}")
        return()
    endif()

    add_executable(${target_name} "${source_file}")

    # The .c tests poke at private headers under include/.
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/src"
    )

    # Link against the in-tree libnimcp + libm + libpthread.
    # Threads::Threads is set up by the top-level CMakeLists; if not, fall
    # back to plain "pthread" so this file is self-contained.
    if(TARGET Threads::Threads)
        target_link_libraries(${target_name} PRIVATE nimcp m Threads::Threads)
    else()
        target_link_libraries(${target_name} PRIVATE nimcp m pthread)
    endif()

    # RPATH so the test binary can dlopen libnimcp.so.2 at runtime without
    # the caller exporting LD_LIBRARY_PATH. Two settings — BUILD_RPATH for
    # in-place ctest runs, INSTALL_RPATH for installed-test layouts.
    set_target_properties(${target_name} PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH   "${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib"
    )

    # Make sure the lib is built before the test (some generators get this
    # implicitly via target_link_libraries; explicit add_dependencies is a
    # belt-and-suspenders for older Make/Ninja generators).
    add_dependencies(${target_name} nimcp)

    # Register with ctest. Run from the binary dir so any temp files the
    # test writes (test_bulk_lexicon writes a /tmp .bin) land in a known
    # place. Tag with a label so `ctest -L lang_smoke` selects only these.
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES
        LABELS "lang_smoke"
        # 60s is plenty — every standalone test is a few-call smoke,
        # not a stress run.
        TIMEOUT 60
    )
endfunction()

# Walk the list and register each one. Skip any source file that is missing
# from the worktree (e.g. test_bulk_lexicon.c if the loader hasn't been
# checked in on this branch — per the task spec, that one is allowed to be
# absent).
foreach(_t IN LISTS NIMCP_STANDALONE_LANG_TESTS)
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/unit/${_t}.c")
    if(EXISTS "${_src}")
        add_nimcp_standalone_test(${_t} "${_src}")
    else()
        message(STATUS "Standalone test source not found, skipping: ${_src}")
    endif()
endforeach()
