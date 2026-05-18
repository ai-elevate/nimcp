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

### Slice 5: Phonological-loop working memory buffer (shipped 2026-05-18)

**What**: replace one-shot utterance construction with a buffer that's incrementally refined each iteration. Baddeley's phonological loop model.

**Implementation**:
- `phonological_loop` lives on the brain struct (`loop_*` fields, append-only). Holds: surface buffer + per-word phonemic traces + the words themselves + decay rate + max-word cap + last-refresh timestamp + enabled flag + per-loop mutex.
- Allocated once in `nimcp_brain_factory_init_broca_subsystem` (sibling of `speech_repair`), freed in the matching destroy. Initial capacity 256 bytes buffer + 16 trace slots; grows on demand via realloc, non-fatal on alloc failure.
- Default OFF — when `loop_enabled=false`, the recurrent cascade is byte-identical to Slice 1+2.

**Recurrent-loop integration**:
1. `communication_cascade_run_recurrent` calls `phonological_loop_clear(brain)` on entry (single-recurrent-run buffer; not persisted across calls).
2. At the start of each iteration after the first, `phonological_loop_decay(brain)` multiplies every trace by `(1 - decay_rate)` and evicts traces below 0.05.
3. Inside `cascade_stage_lexical`, AFTER the bridge produces, `phonological_loop_merge_words(brain, state->utterance)` tokenizes the produced text; existing words refresh trace to 1.0, new words append (capped at `loop_max_words=16`).
4. `cascade_stage_lexical` then SWAPS `state->utterance` to the buffer's surface form (`phonological_loop_render_active`, threshold 0.3). All downstream stages — `cascade_stage_syntactic`, `cascade_stage_self_comprehension`, speech-repair, prosody — see the buffered version.

**Public surface (all default-OFF)**:
- C API: `nimcp_brain_set_phonological_loop_enabled` / `_set_phonological_loop_decay` / `_clear_phonological_loop` / `_get_phonological_loop_state` (buffer + trace_count) / `_get_phonological_loop_diag` (full struct).
- Python: `Brain.set_phonological_loop_enabled` / `set_phonological_loop_decay` / `clear_phonological_loop` / `get_phonological_loop_state` / `get_phonological_loop_diag`.
- Daemon RPC: `set_phonological_loop_enabled` / `set_phonological_loop_decay` / `clear_phonological_loop` / `get_phonological_loop_state` / `get_phonological_loop_diag`.
- Client: `BrainProxy.set_phonological_loop_enabled` / `set_phonological_loop_decay` / `clear_phonological_loop` / `get_phonological_loop_state` / `get_phonological_loop_diag`.

**Files**:
- `include/core/brain/nimcp_brain_internal.h` (loop_* fields appended).
- `include/language/nimcp_phonological_loop.h` + `src/language/nimcp_phonological_loop.c` (new; all loop logic).
- `include/nimcp.h` + `src/api/nimcp_part_core.c` (public C API).
- `src/core/brain/factory/init/nimcp_brain_init_broca.c` (init / destroy wiring).
- `src/language/nimcp_communication_cascade.c` (recurrent loop + lexical stage swap).
- `src/bindings/python/nimcp_python.c`, `scripts/brain_daemon.py`, `scripts/brain_client.py`.

### Slice 6: Thalamic gating

**What**: thalamus modulates which populations get processing bandwidth. Already wired into substrate (per `substrate-plan` memory) but not gating cascade flow.

### Slice 7: Cerebellar prediction-correction

**What**: cerebellum predicts next motor pattern; online correction signal feeds back into motor and prosody.

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
