# NIMCP V2 — Implementation Plan (Rust Rewrite)

**Status:** planning
**Branch:** `v2` (forked from `master` at `fc952e587`)
**Target:** Rust, async-first, event-sourced, actor-model cognitive architecture

---

## 1. Why V2

V1 (this repo's `master`) taught us concrete lessons. The ones that justify a rewrite:

| V1 pain | V2 answer |
|---|---|
| 800-field `brain_struct`, cross-cutting access patterns | Actor model — every subsystem owns its state |
| Races on shared mutable state (today's `connected_dst` bug, pr_memory tick, many more) | Event-sourced; state mutations are messages |
| Cognitive recovery handler hangs 16s on corrupt heap | Crash-first design: cheap rollback via event replay |
| Saturated-checkpoint resume trap (quiet-start doesn't re-apply) | Quiet-start as a load-time weight transform, keyed on stats |
| Tight-bounds homeostatic oscillation bug (designed and refixed) | Feedback loops analyzed for phase margin before wiring |
| 60 cognitive modules, most unused or "statues" | Strict entry rule: prove value in isolation for 30+ days |
| 8 language bindings, maintenance drag | Python only via PyO3; others post-parity |
| 600+ commits before core training was stable | Toy-brain harness: 100-neuron variants, full-dynamics in seconds |
| C: NULL derefs, unchecked mallocs, missing locks | Rust: ownership, `Send`/`Sync` enforced at compile |

We keep V1's **biological grounding ambition** — homeostasis, multi-timescale memory, neuromodulators, multi-network ensemble. We change the implementation.

---

## 2. Principles

1. **No `unsafe` without a written justification in the PR.** The few FFI points (CUDA, PyO3) will have `unsafe`; nothing else.
2. **Event-sourced state.** Every mutation is an event appended to a log; current state is a materialized view. Checkpointing = truncate log + dump view. Recovery = replay tail.
3. **Actors own their state.** Inter-actor communication is typed messages. No `Arc<Mutex<T>>` passed between domains.
4. **One scheduler.** Tokio-based. Deterministic replay when `--deterministic` flag set.
5. **Toy before scale.** Every feature demonstrated on a 100-neuron toy brain first (runs in <5s) before being tested at 1.8M.
6. **Crash-first.** Use `minidump-writer` crate for crash dumps + `eyre` for rich backtraces from day one. No elaborate recovery; just log + rollback.
7. **Feedback control modeled.** Any feedback loop (homeostatic scaling, R-STDP rate, etc.) gets a Python notebook with phase-margin analysis before wiring.
8. **Quiet-start is a transform, not a code path.** Applied at any weight load based on observed saturation stats.
9. **Scope discipline.** Nothing enters core until 30+ days of isolation validation. No exceptions.

---

## 3. Architecture

### 3.1 Core abstractions

```rust
// Every state change is an event
pub trait Event: Serialize + DeserializeOwned + Send + 'static {
    fn apply(self, state: &mut Self::State);
    type State;
}

// Actors have a mailbox and process messages asynchronously
#[async_trait]
pub trait Actor: Send + 'static {
    type Msg: Send + 'static;
    async fn handle(&mut self, msg: Self::Msg, ctx: &mut Context) -> Result<(), Error>;
}

// Compute kernels are pure: (inputs, weights) -> outputs
pub trait Kernel {
    type Input;
    type Weights;
    type Output;
    fn forward(&self, input: &Self::Input, weights: &Self::Weights) -> Self::Output;
    fn backward(&self, input: &Self::Input, output_grad: &Self::Output,
                weights: &Self::Weights) -> (Self::Input, Self::Weights);
}
```

### 3.2 Top-level layout

```
nimcp-v2/
├── crates/
│   ├── core/              # traits, errors, ID types
│   ├── eventlog/          # persistent append-only log (sled or rkyv)
│   ├── scheduler/         # tokio-based actor runtime
│   ├── gpu/               # compute backend (cust or wgpu)
│   ├── networks/
│   │   ├── adaptive/      # MLP (the V1 "ANN")
│   │   ├── snn/           # spiking (LIF + CSR synapses)
│   │   └── lnn/           # liquid time-constant
│   ├── plasticity/        # STDP, R-STDP, homeostatic (each isolated, unit-testable)
│   ├── memory/            # Z-Ladder (port of V1 Phase E, simplified)
│   ├── checkpoint/        # state serialization via rkyv
│   └── brain/             # integration: brain = composition of actors
├── python/                # PyO3 bindings: nimcp-py
├── tests/                 # cross-crate integration
├── benches/               # criterion benchmarks vs V1
├── examples/
│   ├── toy_brain.rs       # 100 neurons, XOR learning
│   └── saturation.rs      # tests quiet-start recovery
└── Cargo.toml             # workspace root
```

### 3.3 Key invariants

- **No global state.** No `static` muts, no thread-local storage for domain data.
- **All I/O through actors.** No direct file/socket access from computation code.
- **Deterministic seeds.** Every RNG gets an explicit seed from config.
- **Actor death is isolated.** A panicking actor doesn't take down siblings; supervisor logs + restarts.

---

## 4. Phase plan

Estimated for one experienced Rust developer full-time. Numbers are realistic, not optimistic.

### Phase 0 — Foundation (3 weeks)
- Cargo workspace with above crate layout
- Core traits (`Event`, `Actor`, `Kernel`)
- Tokio-based scheduler with deterministic-replay mode
- Event log backed by `sled` (or `rkyv` + custom append file)
- PyO3 scaffolding — empty `nimcp-py` crate with `Brain::new()`
- CI: rustfmt, clippy, cargo test, cargo bench, coverage (`cargo-llvm-cov`)
- Crash pipeline: `minidump-writer` on panic, structured logs via `tracing`
- **Exit criteria:** empty actor runs, emits events, checkpoint + restore round-trips via event replay.

### Phase 1 — Minimum viable brain (6 weeks)
- Adaptive network on CPU: MLP forward + backward, SGD
- Event log captures every weight update
- Python API: `Brain::new(config)`, `brain.learn(x, y)`, `brain.predict(x)`
- Toy-brain test: 100 neurons learn XOR in <5 seconds
- Determinism: identical seed → identical weights, bit-for-bit, across runs
- **Exit criteria:** replaces `adaptive_network_learn` from V1 for MLP workloads on CPU. Benchmark: within 2× of V1 single-threaded.

### Phase 2 — GPU compute (5 weeks)
- Choose backend: **`cust`** (NVIDIA-specific, mature) over `wgpu` (portable but younger for HPC)
- GPU actor: submit job → future; batches across callers
- Port adaptive network forward/backward to CUDA kernels
- Benchmark parity: must match or beat V1 per-step throughput at 150K neurons
- **Exit criteria:** 150K-neuron MLP trains at V1 wall-clock or better.

### Phase 3 — SNN (8 weeks)
- LIF neurons as pure kernel
- CSR synapse format (port V1's lightweight approach — this one V1 got right)
- R-STDP plasticity as event source
- Homeostatic scaling with **load-time quiet-start transform**
- Saturation recovery: integration test that loads saturated weights + verifies recovery within 100 steps
- **Exit criteria:** 1.8M neuron SNN runs at V1 rate; recovery from saturation is automatic.

### Phase 4 — Multi-network ensemble (6 weeks)
- LNN (liquid time-constant) — CPU to start
- Actor per network: adaptive, SNN, LNN
- Shared loss aggregator actor
- Joint checkpoint atomic across networks
- **Exit criteria:** 3-network ensemble trains on multimodal input (image + text), loss converges, checkpoint round-trips.

### Phase 5 — One memory system (5 weeks)
- Z-Ladder port (V1 Phase E was well-designed; just translate)
- Landmark API with capacity eviction
- Checkpoint with full feature payload (unified_mem_manager equivalent by default)
- **Exit criteria:** landmark save/load preserves features; query returns expected hits.

### Phase 6 — Introspection (4 weeks)
- Read-only query actor that aggregates state across all brain actors
- Metrics: firing rates, loss, weight stats, memory audits
- Exposed via Python API
- **Exit criteria:** `brain.stats()` returns a comprehensive, documented dict.

### Phase 7 — Training harness + curriculum (5 weeks)
- Stages as actors
- Multimodal data loaders (image, audio, text)
- Optional: Claude teacher integration via HTTP actor
- **Exit criteria:** multi-stage training script runs end-to-end on a toy dataset.

### Phase 8 — Production hardening (3 weeks)
- Load testing (OOM under pressure, long-running stability)
- Performance profiling + optimization
- Documentation: public API, concepts, migration guide from V1
- **Exit criteria:** 24-hour training run on RunPod with no crashes, deterministic output.

**Total:** ~11 months single-developer to V1 core-feature parity.

---

## 4½. Safety Shell Track (parallel to main rewrite)

### Why

V1's **compute** code is fine. After today's `connected_dst` race fix, the ANN
forward/backward, SNN LIF + CSR synapses, plasticity rules, and cognitive
modules all run correctly. Every V1 crash we've triaged — recovery-handler
16 s hang, scattered checkpoint writes, pr_memory tick race, saturated-resume
trap — has lived in the **coordination layer**, not the math. Rewriting
kernels in Rust gives us nothing those kernels aren't already delivering.

So we take the V2 stability win (actor isolation, event-sourced state,
crash-cheap recovery, proper checkpointing) in **weeks instead of months** by
wrapping V1's libnimcp.so behind V2's coordinator. The curriculum investment —
4 developmental stages, `data/stimuli/`, `data/multimodal/` — is preserved
unchanged. The shell runs **parallel to Phases 2+ of the main rewrite**; as
each pure-Rust subsystem lands it slots in behind the bridge and the
curriculum doesn't notice.

### Architecture

```
Curriculum (immerse_athena.py, UNCHANGED)
    ↓ Unix socket, length-prefixed JSON
V2 Daemon (nimcp-daemon — new Rust binary)
    ↓ tokio actor, typed messages
V2 Brain Actor (nimcp-brain — V2's actor wrapper)
    ↓ RAII-safe FFI
nimcp-v1-bridge crate (new, bindgen against V1 headers)
    ↓ dynamic linking
libnimcp.so (V1, compiled by V1's CMake, UNCHANGED)
    ↓
CUDA runtime, GPU
```

### Phases

| Phase | Scope | Effort | Dependency |
|---|---|---|---|
| **SH-1** | `nimcp-v1-bridge` crate — bindgen + RAII wrappers for the 10-15 critical V1 functions | ~1 week | V1 `libnimcp.so` built |
| **SH-2** | `nimcp-daemon` binary — Unix socket server, V1-compatible JSON protocol, stub backend | ~1 week | — |
| **SH-3** | Wire daemon → bridge for the 11 critical-path commands (learn_vector, decide_full, submit_sensory, etc.) | ~1 week | SH-1, SH-2 |
| **SH-4** | End-to-end: curriculum runs Stage 0 against V2 daemon, loss trajectory matches V1. Proof of stability over 24 hours. | ~1 week | SH-3 |

**Total:** ~4 weeks single-developer. Compare to Phase 3's 8 weeks for a
pure-Rust SNN alone.

### What the shell wraps

The critical-path commands actually exercised by the curriculum:

- `ping`, `status`, `shutdown` — liveness + control
- `save`, `load` — checkpoint passthrough (V1 .bin + sidecars, unchanged)
- `learn_vector`, `learn_vector_batch` — the hot path
- `submit_sensory` — multimodal input
- `decide_full`, `predict` — inference
- `bg_update_reward` — R-STDP reward signal
- `get_neuron_count`, `get_stats` — introspection

The other 100+ V1 RPC commands are stubbed with
`{"error": "not yet wrapped in safety shell"}` and filled in on demand.

### What it does *not* include

- No rewrite of any V1 compute code
- No replacement of V1 sensory encoders (cortex_cnn, FNO, Phi-3 adapter)
- No replacement of V1 checkpoint format. The V2 coordinator adds rkyv
  sidecars for its own event log, but V1's `.bin` + existing sidecars keep
  working untouched.

### Relationship to the greenfield phases

The shell track runs **parallel** to Phases 2+ of the main rewrite. When a
pure-Rust subsystem completes, its output is swapped in behind the bridge:

- Phase 3 (pure Rust SNN) lands → bridge routes SNN calls to Rust, V1 still
  handles everything else
- Phase 4 (LNN), Phase 6 (introspection), etc. follow the same pattern
- Eventually `libnimcp.so` is no longer referenced and V2 is 100 % Rust

The curriculum never notices these swaps. Training capability is never lost
during the transition — which is the whole point.

### Risks specific to the safety shell

| Risk | Mitigation |
|---|---|
| V1 API drift (master-branch edits break bindgen output) | Pin V1 build to a specific commit hash; re-run bindgen only on intentional bumps |
| CUDA runtime skew (V1 built on 12.0, pod has 12.8) | `cudarc`-style dynamic loading (already handled by the main GPU plan) |
| V1 thread-safety under V2's scheduler | V1's brain mutex is correctly held when called from a single V2 actor thread; one actor, one V1 call site |
| FFI overhead | Length-prefixed JSON + base64 binary = identical wire format to V1's existing daemon. No worse than V1 today |

---

## 5. What to port vs rewrite vs drop

### Port (math is the math)
- Adaptive network forward/backward
- STDP / BCM / eligibility trace equations
- SNN LIF dynamics
- Z-Ladder structure + tier transitions
- Prime signature for content addressing
- Fluctuation-driven init formulas

### Rewrite (V1 abstraction was wrong)
- Brain struct → actor hierarchy
- bio_router → typed actor messages
- Signal handler architecture → panic hook + minidump
- Checkpoint format → rkyv (versioned, schema-evolution-safe)
- GPU weight cache → owned by GPU actor, not smeared across brain

### Drop for now — reintroduce later if value is demonstrated
- 60 cognitive modules (start with 0, add on value proof)
- 33 brain regions as first-class objects (actors subsume them)
- Swarm runtime, drone bridges, ROS 2, sim-to-real
- 8 language bindings (Python only until users demand more)
- HNN, FNO (late additions, not load-bearing)
- Brain-native language, emergent alien language
- Phi-3 adapter, Claude teacher tight integration
- 9-layer safety governance (simplify to 1 layer: input validation + output bounds)

---

## 5½. Development targets — dev-first, pod for scale validation

V2 is developed **primarily on the Hetzner dev server** (RTX 4000 SFF Ada,
sm_89, 20 GB VRAM, CUDA 12.0). RunPod (RTX 5090 Blackwell, sm_120, 32 GB,
CUDA 12.8) is used for full-scale validation at phase exits.

This shifts the inner-loop cost from "SSH + deploy + build on remote + test"
to "cargo test" — measured gain in developer throughput.

| Phase | GPU work | Dev? | Pod? |
|---|---|---|---|
| 0 foundation | none | ✓ | — |
| 1 MLP CPU | none | ✓ | — |
| 2 MLP GPU | offload to CUDA | ✓ (small, sm_89) | validate sm_120 |
| 3 SNN (1.8M) | big VRAM | **down-sized here**, full scale on pod | ✓ |
| 4 ensemble | big VRAM | down-sized here | ✓ |
| 5 memory | CPU | ✓ | — |
| 6 introspection | CPU | ✓ | — |
| 7 training | mixed | down-sized here | validate |
| 8 hardening | full scale | — | ✓ |

Ollama currently holds ~19.5 GB of dev server VRAM. Phases 3+ at full scale
will need Ollama paused during the run. For iteration, scale down to
~300 K neurons — fits in 500 MB.

`NIMCP_GPU_ARCH` env var controls build-time arch targeting:

| Value | Use case |
|---|---|
| `dev` (default) | sm_89 only — fastest compile, local dev |
| `pod` | sm_120 only — release build for pod deploy |
| `all` | sm_89 + sm_120 + PTX — CI / universal tarball |

## 6. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Rust CUDA ecosystem less mature than C++ | `cust` is production-ready; worst case, use `cxx` bridge to thin C++ CUDA layer |
| Scope creep (V1's original sin) | 30-day isolation rule, enforced in PR checklist |
| Learning curve if author isn't Rust-fluent | Pair-program with senior Rust eng OR accept 1.5× timeline |
| Python interop perf on large tensors | Measure early; use `numpy-rs` + zero-copy where possible |
| Checkpoint schema evolution | rkyv's versioning + explicit migration functions |
| GPU determinism for replay mode | `cuDnn` deterministic mode + CPU fallback for test harness |
| Feedback loop instability (V1's homeostatic oscillation) | Phase-margin notebook per loop, REQUIRED before merge |

---

## 7. Decision points that need resolution before Phase 0

1. **GPU crate: `cust` vs `cudarc` vs `cxx` bridge?** Recommend `cust` — most mature, Rust-idiomatic, but NVIDIA-only.
2. **Event log backend: `sled` vs custom rkyv-on-append-file?** Recommend custom — sled is KV-oriented, overkill for append-only.
3. **Async runtime: `tokio` vs `async-std` vs `smol`?** Recommend `tokio` — ecosystem mass.
4. **Serialization: `rkyv` vs `bincode` vs `postcard`?** Recommend `rkyv` — zero-copy, schema versioning.
5. **RNG: `rand` vs `rand_chacha`?** Use `rand_chacha` everywhere — deterministic seeded output.
6. **Linear algebra: `ndarray` vs `nalgebra` vs in-house?** `ndarray` for n-dim tensor, `nalgebra` for linear algebra kernels, custom for hot paths.

---

## 8. What success looks like

At end of Phase 8:
- Bit-for-bit reproducible runs
- 24+ hour training with zero crashes
- Matches V1 throughput on 1.8M-neuron benchmark
- Clean Python API, fully documented
- No `unsafe` outside FFI boundaries
- Every feedback loop has a phase-margin analysis
- Toy-brain regressions run in CI in <30 seconds
- Migration guide for V1 users

At that point V2 is a viable replacement for V1's core. Ecosystem features (swarm, drones, edge, brain-native language, etc.) can be added in a second year if value is demonstrated.

---

## 9. First commit

```bash
cd nimcp
git checkout v2
# Remove all V1 content (optional — or keep as reference under v1/)
# Create cargo workspace
cargo new --lib crates/core
cargo new --lib crates/eventlog
# ... etc per §3.2
# Create python/ with maturin config
# Create Cargo.toml workspace root
```

Author: nimcp maintainers
Last updated: 2026-04-21
