# Changelog

All notable changes to the NIMCP (Neuromorphic Infant Machine Cognitive Platform) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — full-lang-walkthrough campaign (2026-05-07)
A 13-commit campaign covering Tier-A behaviour wiring, Tier-B production
ergonomics, Tier-C threading + cognitive integration, plus immune (IM-3)
and cycle-coordinator (CC-1) work. Every behaviour-changing feature
ships behind a default-OFF runtime flag; legacy callers are bit-identical
to pre-campaign behaviour until they opt in.

**Tier A — comprehend/produce semantics**
- **TA-1** (`4ac5e4c28`) Multi-turn state persistence: discourse turn ring
  + anaphora referent ring + bigram-spectrum count matrix all round-trip
  via `grounded_language_save_multiturn_state` / `_load_multiturn_state`.
- **TA-2** (`81b8c1cb2`) LGSS gates on comprehend + bridge produce. Input
  utterances + generated text are evaluated against the safety KB; deny
  results bump `lgss_inputs_blocked` / `lgss_outputs_blocked` and emit
  audit events. Closes the only previously-ungated user-facing pipeline.
- **TA-3** (`1f6cba089`) Dopamine-modulated STDP on the SNN language
  bridge. The pre-existing `enable_da_modulation` config knob was a
  statue — `apply_stdp` now reads neuromodulator dopamine once per pass
  and applies `weight_change *= 1 + DA × gain` to every binding update.
- **TA-4** (`e1ae72b02`) Trigram next-token training. Extends PA-4 bigram
  training to (w_t, w_{t+1}) → w_{t+2} updates at half the bigram lr.
- **TA-5** (`0bc697e8b`) Reconsolidation on contradiction. Negation-marked
  content words now decay their lexicon-entry binding strengths by 5%
  (tunable, clamped [0, 0.5]). Repeated contradictions across turns
  erode bad bindings; re-assertions recover via normal reinforcement.

**Tier B — production ergonomics**
- **TB-6** (`7f7ca8b3b`) Sentence-boundary segmentation in comprehend.
  Multi-sentence input recursively processes each sentence so discourse
  pushes one turn per sentence + anaphora resolves across sentences +
  bigram learning never bridges a `.`/`!`/`?` boundary.
- **TB-7** (`569088e3b`) Length control on bridge produce. New
  `min_produce_words` (suppresses EOS until reached) +
  `max_produce_words` (hard cap) config knobs with validated setter.
- **TB-8** (`6945afa86`) Streaming produce — per-token callback fires
  during the produce loop. Returning non-zero aborts cleanly; accumulated
  text stays in `result->text`.
- **TB-9** (`9f4ef98a0`) Speech-act intent classification. Rule-based
  classifier labels every comprehend with one of: assertion / question /
  imperative / greeting / exclamation. New `gl_speech_act_t` enum +
  `result->speech_act` field + 5 per-class stats counters.
- **TB-10** (`45ed33825`) Topic-shift detection in discourse. Cosine
  similarity between latest turn vs mean of prior K turns; below
  threshold flags a boundary, bumps `topic_shifts_detected`, exposes
  `last_topic_shift_score` + `last_was_topic_shift` query API.

**Tier C — threading + cognitive integration**
- **TC-12** (`e30215e26`) Per-gl side-state for anaphora + spectrum.
  Killed two global side-maps + one shared mutex that serialized every
  resolve / spectrum-tick across all brains in the process. State now
  lives directly on `struct grounded_language` with a per-instance
  lazy-initialized mutex. Removes the silent 4-brain map cap.
- **TC-13** (`ba15a2a60`) Theory-of-mind subscriber actually calls
  `tom_observe()` instead of just logging at DEBUG. COMPREHENDED +
  PRODUCED gl events flow into a `tom_observation_t` (semantic_vec →
  action_vector, valence/arousal → coarse `tom_emotion_t`). New public
  counters `nimcp_gl_tom_observations_pushed` / `_dropped`.

**Cycle + immune additions**
- **CC-1** (`77e4627d4`) Periodic bigram-spectrum FFT refresh on the
  language tick (~1Hz, gated by min-delta-events). Replaces the pre-CC1
  pattern where metrics only refreshed on external probe.
- **IM-3** (`21605370f`) Tier-3 immune content inspection on
  `grounded_language_comprehend`. Five rule-based heuristics
  (NaN/Inf, statistical outlier via Welford running stats, repetition
  spam, lexicon collision, negation cascade) produce a continuous
  inflammation level that damps confidence + skips engram encode +
  registers an antigen above 0.5.

12 new unit tests under `tests/unit/test_lang_*.c` registered in
`NIMCP_STANDALONE_LANG_TESTS`; all 12 pass cleanly. The 4 pre-existing
lang_smoke teardown crashes (`test_lang_bridge_spike_routing`, `_latency`,
`test_snn_lang_bridge_stdp`, `test_bulk_lexicon`) are unrelated and
unchanged by this campaign — they print PASS then SIGABRT on shutdown.

### Added
- **Conductance-based SNN synapses (CB migration)**: behind a runtime flag
  (`snn_tune("conductance_enabled", 1.0)`, default OFF). Solves the
  dead↔runaway oscillation observed on the pod by giving the membrane
  equation a saturating `(E_rev - V)` driving force instead of the
  unbounded current-based summation. New header-only module
  `include/snn/nimcp_snn_membrane.h` exposes three pure-function helpers
  (`snn_membrane_compute_dv`, `snn_membrane_decay_one`,
  `snn_membrane_deposit_synapse`) used by both the lightweight CSR and
  legacy NEURON_T branches in the SNN hot loop. The CB hot path forces a
  CPU fallback (the GPU kernel remains current-based; GPU port deferred).
  See `docs/claude/cb-phase0-design.md` for the full migration design.
- **CB tunables** exposed via `snn_tune` socket: `conductance_enabled`,
  `cb_weights_rescaled` (sticky), `e_exc_mv` (default 0), `e_inh_mv`
  (default −80), `tau_exc_ms` (default 2), `tau_inh_ms` (default 8). All
  persist in `snn_tune.json` automatically.
- **Weight rescaling admin command**:
  `snn_rescale_weights_for_conductance(network, factor)` (Python:
  `brain.snn_rescale_for_conductance()`) — one-shot scan over every CSR
  population, multiplies entries[] and weights[] mirror, syncs to GPU if
  resident, sets the `cb_weights_rescaled` sticky flag for idempotence.
  Default factor is `1.0/50.0` to compensate for the average ~50 mV
  driving force at rest.
- **Per-neuron CB conductance state**: new `g_exc[]`, `g_inh[]` float
  arrays in `snn_population_t`. Allocated in `snn_population_create_internal`,
  freed in destroy. Allocation failure non-fatal — CB mode silently no-ops
  if either array is NULL.
- **Test suite — 46 new tests across 4 categories** (unit/integration/
  regression/e2e under `test/{unit,integration,regression}/snn/` and
  `test/e2e/test_snn_runaway_suppression_e2e.cpp`). Covers CB math,
  saturation, decay, deposit routing, rescale idempotence, OFF-mode
  bit-identity, and a 500-step recurrent E2E.

### Changed
- SNN GPU fast path is bypassed when `conductance_enabled = 1.0` (the
  GPU kernel does not implement CB; CPU fallback handles all CB runs).
  Zero effect when CB is OFF.

### Notes
- Hierarchical SNN tiers all use lightweight CSR populations; legacy
  NEURON_T pops are not exercised in production. The rescale admin
  command only touches CSR weights — mixed brains with legacy CB pops
  would see un-rescaled legacy weights at ~50× too strong (out of scope
  per design doc).

## [2.7.0] - 2026-04-23

Large multi-day campaign activating previously-dormant biological subsystems
at the hot-path level. Before this release, many helpers existed but were
never wired into brain init or the forward pass — the "statue" pattern.

### Added
- **Neural substrate + thalamic router wire-up (F6/F7)**:
  `nimcp_brain_factory_init_substrate_thalamic_subsystem()` creates a shared
  `neural_substrate_t` (ATP / O₂ / glucose / temperature / ion-balance) and
  `thalamic_router_t` during brain-init Wave 3, and
  `nimcp_brain_attach_substrate_thalamic()` publishes them to every
  SNN / LNN / cortex-CNN network. The Phase 1-4 adapters that consume
  substrate scalars are no longer dormant.
- **Glial network activation (G1–G8)**:
  - `nimcp_brain_factory_init_glial_subsystem()` now creates astrocyte,
    oligodendrocyte, and microglia networks from
    `config.num_astrocytes / num_oligodendrocytes / num_microglia`
    (defaults `neurons/5`, `/7`, `/10`).
  - `nimcp_brain_attach_glial()` populates spatial lookup tables; idempotent.
  - `neural_network_set_glial_integration()` called during init so the
    forward-pass hot-path gate (`network->glial_integration`) actually sees
    the integration pointer. Regression test
    `BrainGlialInitTest.NeuralNetworkReceivesGlialPointer` guards against
    recurrence of the same "statue" pattern fixed by the substrate campaign.
  - Hot-path (`nimcp_neuralnet.c:compute_input_for_neuron`) applies:
    astrocyte modulation `[0.8, 1.2]`, oligodendrocyte myelin boost
    `1.0 + 0.5 × myelination_factor`, microglia pruning (zero transmission
    when `should_prune_synapse` is true, gated by
    `config.enable_microglia_pruning`).
- **SNN lightweight CSR wiring for cognitive bridges (G8)**: all 4 bridges
  (attention, mirror-neurons, emotion, working-memory) now use
  `snn_network_add_population_lightweight()` exclusively, plus a
  `snn_network_finalize_connections()` call after all connections. A typed
  return-value guard (`g8_add_lw_pop`) in every bridge prevents silent
  `0xFFFFFFFF` pop-id corruption on allocation failure. H1 I/O shunts
  (dense `input_pop` → bridge's primary lightweight pop, and bridge's
  output pop → dense `output_pop`) restore connectivity from
  `snn_network_set_inputs` / `get_outputs` into the bridge's custom
  lightweight graph.
- **Thalamic backpressure API**: `thalamic_router_queue_usage(router) → [0,1]`
  and `thalamic_router_is_under_pressure(router) → bool` (true at ≥80%).
  Producers can throttle low-priority signals without blocking.
- **Glial config flags**: `enable_microglia_pruning`,
  `enable_glial_synaptic_modulation`, `enable_glial_myelin_conduction` on
  `brain_config_t`.
- **synapse_id bijection (G7)**: widened from `uint32 (pre*10000+post)` to
  `uint64 ((pre<<32)|post)`; bijective over `(uint32, uint32)` pairs, zero
  collisions at any brain scale. Propagated through `glial_integration_t`,
  `monitored_synapse_t`, and `microglia_*_synapse` API (public signature
  change: uint32 → uint64).
- **Brain-wide monitoring probe** (`/root/brain_probe.py`): reads daemon
  RPCs (`stats`, `snn_pop_stats`, `substrate_get_health`,
  `get_network_metrics`, `snn_tune_get`) + log-derived signals; emits
  structured single-line health snapshot with `ALERT[...]` and (optional)
  `HEAL[...]` tags. Detects dead populations, gradient explosion/vanishing,
  mode collapse, spike collapse, queue overflow, fatal signatures.

### Changed
- `THALAMIC_MAX_QUEUE_SIZE` bumped 1000 → **16384** after the 2026-04-23
  silent-death postmortem (5 concurrent multimodal producers saturated
  the 1000-entry ring in ~7 s). ~786 KB per router; one router per brain.
- `enqueue_signal` queue-full now logs rate-limited `WARN` (every 1024th
  drop) instead of raising `NIMCP_THROW_TO_IMMUNE`. Queue-full is
  backpressure, not a fault; the old throws spammed the exception
  rate-limiter (5000/sec cap) and masked real errors.
- `CUDA_ARCHITECTURES` is now conditional on `CUDAToolkit_VERSION >= 12.8` —
  pod with RTX 5090 (Blackwell / compute_120) + CUDA 12.8 builds with
  `75;80;86;89;120`; local dev with CUDA 12.0 falls back to `75;80;86;89`.
- Sensory-input submission now logs **all four modalities symmetrically**
  (VISUAL / AUDIO / SPEECH / SOMATO client tags) — previously only visual
  logged, creating the false appearance that audio / speech / somato streams
  were silent. Empirical confirmation: daemon `total_requests` delta
  corroborates ~225 audio/speech/somato submit_sensory calls per minute
  under multimodal training load.

### Fixed
- **SIGPIPE silent-death class** (`nimcp_init_internal`): daemon now
  installs `SIG_IGN` for SIGPIPE at startup. Prior to this, any `write()` /
  `send()` on a Unix socket whose reader had hung up terminated the
  daemon with no core, no stack, no SIGSEGV signature — matching the
  2026-04-23 pod postmortem exactly. The exception-system crash handler
  covered SEGV / ABRT / BUS / FPE / ILL / TERM but not PIPE.
- **Dead-code "statue" bridge cleanup (F9)**: deleted the pre-Phase-1-4
  glial / substrate bridges superseded by the new adapter path
  (commit `2ecc5e81b`, ~13.5k LOC removed, 4 bridges).
- **SNN auxiliary-pop allocation exhaustion (G8)**: four cognitive bridges
  (`wm_inhibition`, `emotion_va`, mirror-neurons' `hidden` and `output`)
  were calling dense `snn_network_add_population` and exhausting the
  bridge's internal `neural_network_t` slot budget. All 17 bridge
  pop-creation sites now use the lightweight CSR path.
- **Thalamic queue overflow cascade**: combined with the 16384 bump and
  the WARN demotion, post-deploy logs show `qfull=0` under the same
  multimodal load that previously overflowed 4× in 7 s.
- **`snn_network_step_sparse` lightweight safety (M1 from G8 walkthrough)**:
  previously dereferenced `pop->neuron_ids[n]` unconditionally, which
  segfaults for lightweight pops whose neuron_ids array is empty.
  Now gated by `if (pop->lightweight)` and reads from
  `pop->external_current[n]` instead.
- **SNN bridge I/O routing (H1 from G8 walkthrough)**: the dense
  `input_pop` / `output_pop` created by `snn_network_create()` with a
  feedforward config were disconnected from each bridge's lightweight
  custom pops. `snn_network_set_inputs` / `get_outputs` therefore missed
  the bridge's actual graph. Added input / output shunts per bridge to
  restore the flow.

### Testing
- 9 new unit tests for glial brain init (`test_brain_glial_init.cpp`)
  covering creation, accessors, idempotent attach, destroy safety, NULL
  tolerance, C1 regression ("pointer reaches neural network"), and
  modulation-disabled identity behavior.
- 5 new integration tests for glial hot path (`test_glial_hot_path.cpp`)
  covering forward-pass behavior, unassigned-synapse safety, and G7
  bijection across old-collision points.
- All 14 new tests pass locally; existing regression coverage (substrate,
  SNN BPTT, etc.) still green.

### Deployment
- Cold-start deployed to RTX 5090 (Blackwell) pod:
  - `libnimcp.so` rebuilt with `compute_120` architecture support
  - Python bindings (`nimcp.cpython-312-x86_64-linux-gnu.so`) reinstalled
    after `brain_t` / `monitored_synapse_t` struct-layout changes
  - Daemon `--fresh` cold start; pre-G8 corrupt checkpoints quarantined
  - Post-deploy observed: **45/48 SNN populations active** (was 4/48
    pre-G8), all four multimodal streams logging, spike rate climbing
    under homeostat control from the 0.03 Hz quiet-start budget toward
    the 5 Hz target, `qfull=0`, `exh=0`, `fatal=0`.

### Deferred
- `brain_t` god-struct SOLID refactor (deferred until master stabilizes).
- Per-region substrate sensitivity (interoception-bus scope, task G8
  in the roadmap).
- `snn_network_get_population_rate` caller-side pop-id offset confusion
  (M3 from G8 walkthrough — pre-existing, not caused by G8).

## [2.6.1] - 2025-11-04

### Security
- Replaced all unsafe `strcpy()` calls with `strncpy()` + explicit null termination for buffer overflow protection
  - src/cognitive/knowledge/nimcp_knowledge.c:1675
  - src/utils/thread/nimcp_thread.c:2030
  - src/utils/json/nimcp_json.c:483
  - src/utils/json/nimcp_json.c:524
- Implemented defense-in-depth approach with explicit null byte termination after strncpy

### Fixed
- Memory corruption bug in Knowledge module that caused test segfaults
- Knowledge.LearnFromStory test now passes (previously segfaulted due to buffer overflow)

### Testing
- All 287 cognitive tests passing
- All 27 JSON tests passing
- Verified safe string handling patterns throughout codebase

## [2.6.0] - 2025-11-03

### Added
- FFT (Fast Fourier Transform) spectral analysis utilities
  - Cooley-Tukey radix-2 FFT algorithm (O(N log N) complexity)
  - Real-to-complex and complex-to-complex transforms
  - Window functions (Hann, Hamming, Blackman)
  - Power spectral density (PSD) computation
  - Brain wave band power extraction (Delta, Theta, Alpha, Beta, Gamma)
- Brain oscillation analysis module
  - Real-time neural oscillation detection
  - Cognitive state inference from brain waves
  - Phase-amplitude coupling (PAC) analysis
  - Network synchrony computation

### Testing
- 14/14 FFT tests passing
- All spectral analysis functionality validated

## [2.5.1] - 2025-11-03

### Added
- Knowledge B-tree indexing for efficient confidence-based queries (O(log n) range queries)
- `knowledge_get_by_confidence_range()` - Query knowledge items by confidence level
- `knowledge_get_all_ordered_by_confidence()` - Get all knowledge sorted by confidence
- `knowledge_add_item()` - Test helper API for direct knowledge insertion

### Fixed
- B-tree key extraction pattern using stable stored keys instead of thread-local buffers

### Testing
- 600+ tests passing
- 2/3 knowledge B-tree tests passing (1 performance issue under investigation)

## [2.5.0] - 2025-11-02

### Added
- Refactored visual and audio cortex modules to use NIMCP utility functions
- Consistent validation and logging patterns across perception modules
- Updated ~75+ validation points in visual and audio cortex

### Changed
- Replaced bare pointer checks with `nimcp_validate_pointer()` throughout perception modules
- Added descriptive `NIMCP_LOGGING_ERROR()` calls for better error tracking
- Improved error handling and diagnostics in perception layer

### Testing
- Build successful with no regressions
- All existing tests continue to pass

## Earlier Versions

See git history for changes in versions prior to 2.5.0.

---

**Note:** This changelog was started on 2025-11-04. Earlier project history is available in the git commit log.
