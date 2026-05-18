# Recurrent + Predictive Language Architecture

**Status**: design — implementation in slices
**Author**: 2026-05-18 thread
**Replaces**: `communication_cascade.c`'s sequential 15-stage pipeline (Levelt 1989 style)
**Motivation**: bridge-only training produces word-salad; cascade alone is sequential pipeline; real cortex is continuously recurrent with predictive coding. To achieve biologically-faithful coherent generation, replace the cascade with a recurrent dynamical system matching the published consensus models of language production (Dell 1986, Hickok & Poeppel 2007, Hagoort MUC 2005, Friston FEP).

## Target architecture

```
   ┌─────── prompt ───────┐
   │                      │
   ▼                      │
 [Wernicke] ◄───── arcuate fasciculus ─────► [Broca]
   │  ▲                                       ▲  │
   │  │ (predictions)                         │  │
   │  ▼                                       │  ▼
 [Temporal] ── lexical column ───► [Bridge] ──┴──► [Phonological]
   │            (competing                              │
   │             candidates)                            ▼
   │                                                [Motor]
   │           ┌─[OFC]──drive ─┐                      │
   ▼           ▼               │                      ▼
 [Working memory buffer] ◄─[PFC]─goal ◄──── self-feedback ──┐
   │                            ▲                            │
   ▼                            │                            │
 [Hippocampus]─episodic ────────┘                            │
   ▲                                                          │
   │                       ┌─[Cerebellum]─prediction ◄────────┘
   └─[Thalamus]─gating ────┘
```

All populations continuously active. Single iteration is a snapshot; the system iterates until activation patterns settle (or hard timeout).

## Computational model

Replace `communication_cascade_run` with `recurrent_language_run`:

```c
int recurrent_language_run(
    brain_t brain,
    const char* prompt,                  /* NULL = spontaneous */
    uint32_t max_iterations,             /* default 8 */
    float convergence_eps,               /* default 0.01 */
    production_cascade_state_t* state    /* same struct, reused */);
```

Each iteration:

```
1. Read-only snapshot of all population activations into local buffer.
2. For each population, compute next activation from snapshot + own state.
   - Drive: motivation_id, intensity from OFC state
   - Goal: speech_act from PFC + drive + prompt
   - Episodic: hippocampal retrieval given semantic_vector
   - Listener: sensory_integration adjustments
   - Content: blend of drive+goal+episodic+listener
   - Lexical: competing candidates (lateral inhibition)
   - Syntactic: Broca's CYK + grammar agreement
   - Self-comp: Wernicke re-parse own output
   - Phonological: syllable + prosody
   - Motor: articulation
   - Self-feedback: error monitoring
3. Compute predictions for each population (top-down):
   - Each population produces (current_estimate, prediction_for_inputs)
4. Compute prediction errors = current - prediction_from_downstream.
5. Apply FEP-driven plasticity weighted by prediction error.
6. Lateral updates: Wernicke ↔ Broca via arcuate fasciculus.
7. Thalamic gating: which populations get attention bandwidth.
8. Cerebellar correction: predict next-step pattern, drive online correction.
9. Convergence check: if max |activation_delta| < eps, exit; else iterate.
```

Final iteration's working-memory buffer is the utterance.

## Slices

Each slice is independently committable and brings observable behavior change.

### Slice 1: Iterative cascade loop with convergence

**What**: wrap the existing sequential cascade in an outer iteration loop. Same 15 stages run multiple times. After each pass, check if `self_match` and `utterance` have stabilized. Exit on convergence or max iterations.

**Why first**: scaffolds the iteration model without disturbing per-stage semantics. Lowest blast radius. Establishes that "settling dynamics" is the right metaphor before changing internal stages.

**API**:
```c
/* New function. Calls communication_cascade_run() up to max_iters times,
 * checking convergence between iterations. */
int communication_cascade_run_recurrent(
    brain_t brain, const char* prompt,
    uint32_t max_iters,        /* default 8 */
    float utterance_change_eps,/* default 0.0 = exact match */
    float self_match_eps,      /* default 0.01 */
    production_cascade_state_t* out_state);
```

**Wiring**: new flag `cascade_recurrent_enabled` defaults OFF; respond_via_cascade routes through recurrent path when on.

**Files**: `nimcp_communication_cascade.c`, `nimcp.h`, `nimcp_part_core.c`, `nimcp_python.c`, `brain_daemon.py`, `brain_client.py`.

### Slice 2: Bidirectional Wernicke ↔ Broca per iteration

**What**: between iterations, feed Broca's emerging utterance back into Wernicke's parser. Wernicke produces a refined parse that Broca uses on the next iteration. Implements the arcuate fasciculus as a real bidirectional information channel, not just "Wernicke runs once at start, Broca runs once at end".

**Why second**: this is the single most important biological-fidelity change. The arcuate fasciculus is the load-bearing white-matter tract for language; making it bidirectional in code matches the anatomy.

### Slice 3: FEP prediction-error hooks

**What**: each stage produces (current_estimate, prediction_of_its_inputs). Downstream stages compute prediction_error = own_input - predicted_input. Error signal drives plasticity (FEP module already exists in repo; wire it).

**Why third**: predictive coding is the mathematical heart of biological inference. Once the iteration scaffold exists, predictions are a natural addition.

### Slice 4: Lateral inhibition in lexical selection

**What**: replace `decode_spikes` top-K argmax with a competitive dynamic. Multiple word candidates excite themselves + inhibit each other; winner emerges from settling dynamics over ~50ms simulated time.

**Why fourth**: lexical competition is well-documented in psycholinguistics (cohort model, semantic interference). Need iteration scaffold first to run the competition.

### Slice 5: Phonological-loop working memory buffer

**What**: replace one-shot utterance construction with a buffer that's incrementally refined each iteration. Baddeley's phonological loop model.

### Slice 6: Thalamic gating

**What**: thalamus modulates which populations get processing bandwidth. Already wired into substrate (per `substrate-plan` memory) but not gating cascade flow.

### Slice 7: Cerebellar prediction-correction

**What**: cerebellum predicts next motor pattern; online correction signal feeds back into motor and prosody.

**Status**: implemented 2026-05-18.

**Existing cerebellum API used** (from `include/core/brain/regions/cerebellum/nimcp_cerebellum_adapter.h`, impl in `src/core/brain/regions/cerebellum/nimcp_cerebellum_adapter.c`):

- `cerebellum_predict_outcome(adapter, motor_cmd[N], num_dims, out_predicted[N], out_confidence)` — forward-model feed-forward. Internal weights are 8×8 (see `forward_model_t::weights[64]`); `num_dims` is clamped to 8 internally.
- `cerebellum_update_forward_model(adapter, motor_cmd[N], outcome[N], num_dims)` — gradient-descent update on the (cmd, outcome) pair.
- `cerebellum_broadcast_error(adapter, error_magnitude, error_type)` — climbing-fiber-style global error broadcast to every Purkinje cell; error_type 1 = timing (used by prosody), 2 = force (used by motor), following the convention already used by `world_model_cognitive_integration.c` and `medulla_cerebellum_bridge.c`.
- Adapter lifetime + lifecycle is owned by `brain->cerebellum` — created in `nimcp_brain_factory_init_cerebellum_subsystem`, destroyed in the matching teardown; no per-cascade resource ownership.

**Data flow** (default OFF; enabled via `nimcp_brain_set_cerebellar_correction_enabled(brain, true)`):

```
   cascade_stage_motor:
     build pre-stage 8D feature (drive, syntax, phonology, lexical)
     cerebellum_predict_outcome → predicted[8]      → state.cereb_motor_predicted
     (correction_pending: record bump diag, no text-mode effector yet)
     run stage body (text mode: no physical render)
     build post-stage 8D actual
     cerebellum_update_forward_model(pre, actual)
     cerebellum_broadcast_error(pe_norm, 2 /* force */)

   cascade_stage_prosody:
     build pre-stage 8D feature
     cerebellum_predict_outcome → predicted[8]      → state.cereb_prosody_predicted[0..2]
     if correction_pending:
       base_f0  = (1-strength)·base_f0  + strength·(80 + 320·pred[6])
       range_hz = (1-strength)·range_hz + strength·(      320·pred[7])
       state.cereb_correction_applied = true
     run stage body (FFT-style synthesis)
     build post-stage 8D actual (slots[6..7] = realised F0+range)
     cerebellum_update_forward_model(pre, actual)
     cerebellum_broadcast_error(pe_norm, 1 /* timing */)

   communication_cascade_run_recurrent (between iters):
     accum_pe = motor_pe_norm + prosody_pe_norm
     brain->cerebellar_correction_pending = (accum_pe > brain->cerebellar_pe_threshold)
     # pending consumed (and reset implicitly) by next iter's stages
   On exit: restore saved correction_pending.
```

**Files touched**:
- `include/core/brain/nimcp_brain_internal.h` — append-only 7 fields on `brain_struct` (`cerebellar_correction_enabled` + strength + pending + pe_threshold + 3 diag counters).
- `include/language/nimcp_communication_cascade.h` — append-only 7 fields on `production_cascade_state_t`; static_assert bumped 760 → 864.
- `include/nimcp.h` — three new public functions + a `nimcp_cerebellar_diag_t` snapshot struct.
- `src/language/nimcp_communication_cascade.c` — forward-declared the 3 cerebellum API entries (same pattern as `world_model_cognitive_integration.c`); added 7 helper statics (`cereb_clamp01`, `cereb_build_motor_features` / `_actual` / `_prosody_features`, `cereb_norm_diff`, `cereb_bump_*`); rewrote `cascade_stage_motor` (now does predict-stage-update-broadcast or falls back to the legacy skip-record when the flag is off); injected predict-bias-update around `cascade_stage_prosody`'s F0/range/synthesis; added between-iter correction-pending update + save/restore in `communication_cascade_run_recurrent`.
- `src/api/nimcp_part_core.c` — three thin C wrappers + a diag-snapshot wrapper.
- `src/bindings/python/nimcp_python.c` — four Python methods registered in the methods table.
- `scripts/brain_daemon.py` — three RPC handlers.
- `scripts/brain_client.py` — three `BrainProxy` wrappers.
- `tests/unit/test_lang_cerebellar_correction.c` — 5 subtests covering default-OFF, strength roundtrip, motor skip when OFF, on-but-no-cerebellum still skips, recurrent save/restore of `correction_pending`. Registered in `tests/CMakeLists.txt`.

**Default-OFF guarantee**: when the flag is off OR `brain->cerebellum` is NULL, `cascade_stage_motor` short-circuits to the original `cascade_record_skip("stage_motor: text mode...")` body and `cascade_stage_prosody` never enters its cerebellar block — the cascade is byte-identical to pre-Slice-7 master for callers who don't flip the flag.

**Known gaps / TODOs**:
- Text-mode motor stage's "actual" feature vector is currently built from the same upstream signals as the "predicted" — i.e. the cerebellum sees a degenerate (cmd ≈ outcome) pair, so the forward-model weights drift only marginally. Once a physical motor backend lands (TTS / articulator synthesizer), `cereb_build_motor_actual` should read the actual articulatory plan and the cerebellum will start learning real trajectory mappings.
- The cerebellum's forward-model uses a fixed 8×8 weight matrix; our 8D feature packing under-uses the slot count. A future slice could widen to 16+ slots once we add lexical-stress + intonation features.
- Prosody bias is applied only to `base_f0` + `range_hz` (the two scalars that gate the FFT-style contour synthesis). A future slice could bias the per-syllable `coef_decline`/`coef_rise`/`coef_front` coefficients directly.
- Deeper end-to-end tests that actually exercise `cerebellum_predict_outcome` need a non-minimal-init brain (FULL/FAST modes wire the cerebellum subsystem) and belong in `tests/integration` rather than the lang_smoke suite added here.

## Migration strategy

- Old `communication_cascade_run` stays intact. Recurrent path is opt-in via flag.
- Each slice can be rolled forward / back independently.
- Eval: after each slice, run `lang_eval.py` and compare diversity + coherence + self_match.
- Tests: per-slice unit tests in `tests/cascade_recurrent_*.c`.

## What this is NOT

- NOT a performance optimization. Each iteration takes longer than the current single-pass cascade. The goal is biological fidelity and learnable structure, not speedup.
- NOT a replacement for fixing the underlying training (echo + cascade self-train still needed). Architecture + training are orthogonal axes.
- NOT a guarantee of fluent output. The architecture becomes CAPABLE of coherent generation; learning the right weights still requires training cycles.

## Open questions

- Convergence criterion: stop on `self_match` stable, or on full population state stable? Probably the latter; needs measurement.
- Max iteration count: real cortex settles in ~200-400ms; at our ~50ms per stage that's 4-8 iterations. Start with 8, tune.
- How to combine FEP prediction error with existing STDP/echo/comprehend_stdp signals? Likely additive in the LR weighting.
- Lateral inhibition implementation: explicit recurrent network, or simulate via repeated softmax-with-temperature?
