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

### Slice 3: FEP prediction-error hooks ✓ SHIPPED

**What**: each major cascade stage produces a trivial prediction of what its inputs SHOULD look like given the cognitive state, observes the actual input, and records the L2 norm of (actual - predicted) as a per-stage prediction-error scalar on `production_cascade_state_t`. The recurrent loop tracks the per-iteration `pe_total` trajectory and bumps `fep_precision` on high-surprise iterations so `cascade_stage_self_train` applies more plasticity to surprising inputs (Friston FEP: precision-weighted prediction error drives learning).

**Why third**: predictive coding is the mathematical heart of biological inference. Once the iteration scaffold exists (Slice 1) and the arcuate-feedback loop carries meaningful inter-iteration signal (Slice 2), predictions are a natural addition.

**Hooks landed** (all default-zero so the existing single-pass cascade behavior is bit-identical to pre-Slice-3):

| Stage | Predictor | PE scalar |
|-------|-----------|-----------|
| Content | Pre-arcuate blend of drives/episodic/listener/goal (steps 1..5) | `pe_content_norm` — magnitude of the arcuate-feedback correction applied this iter. Drops as the recurrent loop converges. |
| Lexical | "The bridge knows what it's saying" — the bridge's own `prod.semantic_vector` should equal `content_intent`. | `pe_lexical_norm` — `‖content_intent − prod.semantic_vector‖ / sqrt(dim)`. Fallback `1 − fluency` when the bridge doesn't fill `semantic_vector`. |
| Syntactic | "Broca accepts the bridge's word sequence." | `pe_syntactic_norm` — `1 − syntactic_validity`. Stays 0 when the stage was skipped (no observation → no surprise). |
| Self-comp | "My own utterance comprehends back to the original intent." | `pe_self_comp_norm` — `1 − self_match`. Full surprise (`1.0`) when even parsing my own output failed. |

`pe_total = sum(finite per-stage norms)` is the iteration's surprise budget. The recurrent loop pushes `pe_total` into a per-iter trace + summary scalars on the brain; consumers read them via `nimcp_brain_get_cascade_fep_metrics()` / `Brain.get_cascade_fep_metrics()` / RPC `get_cascade_fep_metrics`.

**FEP precision update** (recurrent loop only): `fep_precision_{iter+1} = clamp(1.0 + 0.5 × pe_total_iter, [0.5, 4.0])`. Stage_self_train multiplies its `lr_scale` by `precision` before invoking the bridge plasticity API, so a surprising iteration produces stronger learning on the next iter.

**Surface added** (all opt-in / append-only):

- `production_cascade_state_t` — 6 fields: `pe_content_norm`, `pe_lexical_norm`, `pe_syntactic_norm`, `pe_self_comp_norm`, `pe_total`, `fep_iteration`, `fep_precision`.
- `nimcp_cascade_diag_full_t` — same 6 fields mirrored.
- `brain_struct` — `fep_iterations_run`, `fep_pe_trace[64]`, `fep_pe_initial/terminal/min/max/mean`, `fep_pe_decay_rate`, `fep_converged`.
- New type `nimcp_cascade_fep_metrics_t` in `nimcp_communication_cascade.h`.
- C API: `nimcp_brain_get_cascade_fep_metrics()` in `nimcp.h`.
- Python: `Brain.get_cascade_fep_metrics()` returning dict; `Brain.produce_cascade()` adds 7 PE/FEP fields to its diag dict.
- Daemon RPC: `get_cascade_fep_metrics`.
- BrainProxy: `get_cascade_fep_metrics()`.

**What's NOT here** (deferred to walkthrough / later slices):

1. Predictors are still TRIVIAL (identity / fluency / validity / self_match). A real population-dynamic predictor at each stage — e.g. Broca's syntactic processor learning a model of what bridge-output it expects given the content_intent — lands in a later slice.
2. The PE signal is read-only on the bridge plasticity path; we precision-scale `lr_scale`, but we don't yet route the per-stage PE into stage-specific plasticity (e.g. Wernicke's parse-error → Wernicke STDP).
3. The `pe_content_norm` predictor records the arcuate-feedback magnitude — useful, but not the textbook "predict downstream input" semantics. A proper Content predictor would model what the lexical stage will produce given the current intent.

### Slice 4: Lateral inhibition in lexical selection

**What**: replace `decode_spikes` top-K argmax with a competitive dynamic. Multiple word candidates excite themselves + inhibit each other; winner emerges from settling dynamics over ~50ms simulated time.

**Why fourth**: lexical competition is well-documented in psycholinguistics (cohort model, semantic interference). Need iteration scaffold first to run the competition.

### Slice 5: Phonological-loop working memory buffer

**What**: replace one-shot utterance construction with a buffer that's incrementally refined each iteration. Baddeley's phonological loop model.

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
