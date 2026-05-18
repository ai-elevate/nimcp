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

### Slice 4: Lateral inhibition in lexical selection — SHIPPED

**What**: a new `snn_language_bridge_decode_with_lateral_inhibition()` wraps the standard cosine-`decode_spikes` argmax with a recurrent competition. Top-K candidates excite themselves AND inhibit each other across `T` micro-steps:

```
new_a[k] = sigmoid(a[k] * gain_self - sum_{j != k} a[j] * gain_inhibit)
```

After settling, the winner is re-ranked by post-competition activation rather than the one-shot cosine score. Implements the cohort model (Marslen-Wilson 1987) and interactive activation (McClelland 1981).

**Why fourth**: lexical competition is well-documented in psycholinguistics (cohort model, semantic interference). The iteration scaffold from Slices 1-2 ensures the bridge has the right granularity for this kind of within-step settling.

**Wiring**: default OFF via a new `enable_lateral_inhibition` bool on `snn_lang_config_t` (append-only — static_assert bumped 172 -> 188 bytes). When flipped on, the bridge's `produce()` loop transparently swaps `decode_spikes` for the lateral path on every per-word decode (and the same swap inside `produce_beam_search`). Hyperparameters runtime-tunable: `gain_self` (default 1.5), `gain_inhibit` (default 0.026 ~= 0.8/(K-1) for K=32), `micro_steps` (default 20).

**Stability**: activations are bounded to [0, 1] via sigmoid each step. NaN/Inf would force the function to fall back to cosine top-K with a one-shot warning. With the default gains, the leader saturates near ~0.62 and subordinates decay toward 0 within ~10-15 micro-steps.

**Cost**: K * T multiplies + K sigmoids per word selected. At K=32, T=20 that's ~640 ops + 20 sigmoid evals per emitted word — under 5us at -O2 on x86_64.

**Surface added — all opt-in, no legacy behavior change**:
```
C:      snn_language_bridge_decode_with_lateral_inhibition
        snn_language_bridge_set_lateral_inhibition_enabled
        snn_language_bridge_get_lateral_inhibition_enabled
        snn_language_bridge_set_lateral_inhibition_params
        snn_language_bridge_get_lateral_inhibition_params
API:    nimcp_brain_set_lateral_inhibition_enabled
        nimcp_brain_get_lateral_inhibition_enabled
        nimcp_brain_set_lateral_inhibition_params
        nimcp_brain_get_lateral_inhibition_params
Python: Brain.set_lateral_inhibition_enabled(bool)
        Brain.get_lateral_inhibition_enabled() -> bool
        Brain.set_lateral_inhibition_params(gain_self, gain_inhibit, micro_steps)
        Brain.get_lateral_inhibition_params() -> {gain_self, gain_inhibit, micro_steps}
Daemon: _cmd_set_lateral_inhibition_enabled
        _cmd_get_lateral_inhibition_enabled
        _cmd_set_lateral_inhibition_params
        _cmd_get_lateral_inhibition_params
Client: BrainProxy.set_lateral_inhibition_enabled
        BrainProxy.get_lateral_inhibition_enabled
        BrainProxy.set_lateral_inhibition_params
        BrainProxy.get_lateral_inhibition_params
```

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
