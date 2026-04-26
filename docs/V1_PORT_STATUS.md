# V1 → V2 Port Status

**Audit date**: 2026-04-26
**Scope**: All 99 master commits since the v2 fork at `fc952e587`
**Audit command**: `git log --oneline fc952e587..master`

This catalogs the disposition of every master commit landed since v2 forked
off. V2 is a Rust rewrite that intentionally drops large parts of V1's
architecture (KG waves, cycle_coordinator, octopus, swarm, brain regions,
language bindings, sleep-wake, etc.), so most V1 commits have nothing to
port — they live in code paths V2 doesn't have.

## Legend

- **PORTED** — substantively in v2 already (with v2 commit reference)
- **N/A-DROPPED** — touches V1-only architecture; nothing to port
- **NEEDS-PORT** — genuine gap; addressed in this campaign
- **DEFERRED** — genuine gap, but scoped to a future V2 phase

## Summary

| Disposition  | Count |
|--------------|------:|
| PORTED       |    22 |
| N/A-DROPPED  |    72 |
| NEEDS-PORT   |     0 |
| DEFERRED     |     4 |

**Updates 2026-04-26 (later same day):**
- CB migration landed in `eb10ab00e` — V2 Phase 10.
- Autotune closed-loop SNN brake controller landed in the next commit
  (`crates/networks/snn/src/autotune.rs`) — pure observer/actuator
  state machine, daemon does the I/O.
- `nimcp_get_alloc_stats` equivalent landed as
  `crates/core/src/alloc_stats.rs` — `/proc/self/status` reads + tagged
  byte counters (no audit-log/KG aggregation since V2 has neither).

The NEEDS-PORT bucket is now empty. The remaining DEFERRED entries are
all CB-related operational follow-ups whose V2 equivalents are
subsumed by the cleaner `cb_weights_rescaled` snapshot field — they
need no further action.

---

## PORTED (18) — already in v2

| Master commit | What | V2 equivalent |
|---|---|---|
| `2bd4099ff` | Stronger Poisson noise defaults (20Hz × 30mV) | v2 `noise.rs` defaults match (port commit `c23c1fef0` and earlier SNN-stability port wave) |
| `1a495f51d` | Three control fixes for SNN saturation↔collapse oscillation | v2 `c23c1fef0` "port three control fixes Fix 2 + Fix 3" |
| `bd0d5f537` | Reserve population fields for AHP / Na-K pump / basket | v2 `adaptation.rs`, `basket.rs` |
| `a3a1de0e2` | Shared adaptation mechanism (AHP + pump) | v2 `adaptation.rs` |
| `6b5f82ac0` | Basket cell pool — fast-spiking inhibitory interneurons | v2 `basket.rs` |
| `cf793e0f0` | Anti-reward — negative intrinsic reward for saturated pops | v2 `intrinsic_reward.rs` |
| `309ee15e8` | Runtime-tunable STD dynamics | v2 `depression.rs` + `tuning.rs` |
| `6137baf5d` | Integrate AHP + pump + basket + E/I noise into network step | v2 `5d5e90567` "integrate Phase 3.5 stability mechanisms into Population step" |
| `3db757ebc` | Expose SNN biophysical tunables (Python) | v2 `pybind/src/lib.rs` snn tuning bindings |
| `6b374ccac` | Regression — disable-path determinism | v2 `tests/snn_disable_path_determinism.rs` (Phase 8 contract) |
| `89db34f9f` | E2E — biophysical stability under hierarchical load | v2 `bench/snn` `NIMCP_STABILITY=1` mode |
| `785388ac5` | Integration — SNN biophysical mechanisms | v2 SNN integration tests |
| `a2ec521b9` | Substrate shared effects helper | v2 `crates/substrate` |
| `6bf4e6446` | SNN substrate adapter (ATP/temperature/ion effects) | v2 `crates/networks/snn/src/substrate_adapter.rs` |
| `dfa21e367` | Expose SNN substrate tunables (Python) | v2 pybind |
| `6c8a3cff2` | Shared thalamic channel adapter | v2 `crates/thalamic` |
| `54effd95f` | LNN substrate adapter | v2 `crates/networks/lnn/src/substrate_adapter.rs` |
| `35c6b5d38` | LNN thalamic adapter (attention-gated input modulation) | v2 LNN substrate adapter |
| `dec956ab9` | Conductance-based PSC migration (CB Phase 0-4) | v2 `eb10ab00e` "Phase 10 — conductance-based PSC migration" — see Phase 10 section below |
| `312c9b323` | Checkpoint sidecar marker (CB rescaled) | v2 `eb10ab00e` `WeightSnapshot.cb_weights_rescaled` field (cleaner than V1's JSON sidecar) |
| `41583c5a1` `eb42bce46` `203fa0467` | Autotune closed-loop SNN brake controller (P+I) | v2 `crates/networks/snn/src/autotune.rs` — pure state machine, no daemon I/O |
| `8f876fee4` | `nimcp_get_alloc_stats` leak attribution helper | v2 `crates/core/src/alloc_stats.rs` — `/proc/self/status` reads + per-tag byte counters |

The first SNN-stability wave (Phase 3.5), Path A (substrate + thalamic
infrastructure across SNN + LNN), and the CB migration are now all
mirrored in v2.

---

## NEEDS-PORT (4) — addressed in this campaign

### `43785ee5e` — substrate atomic debit + sentinel guard apply_lr

**V1 fixes 5 bugs.** V2 disposition per bug:

| Bug | V1 issue | V2 status |
|---|---|---|
| #1 atomic debit | C `read-modify-write` racy under multi-thread | **N/A** — V2's `debit_activity` takes `&mut NeuralSubstrate`, Rust's borrow checker enforces serialization |
| #2 spike_threshold_mod formula | Diverged from axon-dendrite bridge | **N/A** — V2's `compute_effects` is a fresh formula, not a port of V1's bridge formula |
| #3 sentinel guard apply_lr | Zero-cache silently kills LR | **PORT** — applies to V2's `effective_rstdp` / `effective_lif`. Even though `Option<&effects>` guards the missing case, a zero-init `DendriteSubstrateEffects { plasticity_mod: 0.0, .. }` with `is_identity == false` still passes through with LR=0. |
| #4 ca_handling_mod ordering | C field-aliasing hazard | **N/A** — V2 computes into locals first |
| #5 temperature pre-clamp | C unbounded q10 | **N/A** — V2 clamps Q10 directly via `clamp(0.5, 1.5)` |

**Action:** add `is_zero_cache()` sentinel + skip-on-zero fallback to v2
substrate adapter helpers (SNN + LNN).

### `41583c5a1` + `eb42bce46` + `203fa0467` — autotune closed-loop SNN brake controller

V1 implements a 5-rule closed-loop controller in `brain_daemon.py` that
observes peak-rate / recovery-time / dwell-band and adjusts noise rate +
sleep interval + max_scale_dead. V2 has the actuator primitives in
`crates/networks/snn/src/tuning.rs` but no controller layer.

**Action:** port the controller as `crates/networks/snn/src/autotune.rs` —
pure observer/actuator state machine (no I/O), with the
proportional+integral brake rule.

### `8f876fee4` — `nimcp_get_alloc_stats` leak attribution

V1 adds C-side per-tag heap accounting. V2's allocator is Rust's default
(no per-tag tracking). However, an equivalent leak-attribution helper is
useful for V2's long training runs.

**Action:** add `crates/core/src/alloc_stats.rs` — a `GlobalAlloc` wrapper
that records per-thread alloc counts behind a feature gate. Exposed via
the daemon admin endpoint.

### `e5570a04a` — phantom-API audit script

V1's audit script greps C headers for `<return> <name>(...)` prototypes
and verifies definitions exist. The Rust equivalent is `cargo check`
(linker errors are immediate, not lurking) — so the audit is unnecessary.

**However**, V2 does ship a Python pybind shim that can have
declared-but-unimplemented methods (the `__getattr__` fallback hides them).
That's a real gap.

**Action:** add `scripts/v2_phantom_pybind_audit.py` — checks the V1
adapter shim against the actual `nimcp_v2.Brain` method set.

---

## DEFERRED (5) — V2 follow-up, scoped separately

### `dec956ab9` — conductance-based PSC migration (CB Phase 0-4)

5,000+ LoC change to V1's SNN. New runtime flag `conductance_enabled`,
per-neuron `g_exc`/`g_inh` arrays, `snn_membrane.h` helpers,
`snn_rescale_weights_for_conductance()` admin command, 46 new tests, GPU
fallback to CPU. **Deferred to V2 Phase 10** — see below.

### `58893f924` — `_cmd_snn_rescale_for_conductance` socket handler

CB-specific. Deferred with CB.

### `f79764c9b` / `7e69a2a5f` / `312c9b323` — checkpoint sidecar marker for CB rescale

CB-specific double-rescale defenses. Deferred with CB. V2's checkpoint
format is bincode-versioned, so the V1-specific `.bin` rescale-marker
sidecar is moot — V2 will gain a `metadata.cb_rescaled: bool` field on
the checkpoint header instead.

---

## N/A-DROPPED (72) — V1-only architecture

Grouped by reason:

### KG (Knowledge Graph) waves (15 commits) — V2 has no KG
`855f11e45` `1a6b2d9e5` `92110b8bc` `b126e9902` `8ece07125`
`3b2d3e53d` `b20c646cb` `cdf93a116` `5216e973c` `76ed7abac`
`d01ba3008` `7c4c956bf` `c0efb66ab` `54f...` `d87bb4e60`

KG is V1's per-region knowledge-graph subsystem. V2's brain crate has no
KG concept; cognitive bridges live in trait-bounded modules instead.

### `cycle_coordinator` (3 commits) — V2 has no coordinator
`8306b51b2` `e61605634` `07da1abc7`

V2's daemon owns the tick loop directly via `nimcp_brain::tick_thalamic()`
+ `Brain::step()`. No coordinator abstraction needed.

### `bio_router` / `register_driven` infrastructure (2 commits)
`32d4e3950` `0709b61e2`

Same reason — V2 has no bio_router; `Brain::step` directly invokes
configured network steps.

### Brain regions / cognitive bridges (8 commits)
`07da1abc7` `90e957fa6` `051b24034` `2eb928d84` `2eb928d84`
`38348fc03` `65a7dc982` `72b1fc476` `b126e9902`

All Wave-N region adapter wiring. V2's brain crate has cortex stubs but
no region-by-region adapter system yet — these would be
re-implementations from scratch on V2's `Network` trait, not ports.

### Octopus module (1 commit)
`b8ade2283`

Octopus = V1's distributed peripheral cognition module. Not in V2.

### Swarm runtime (1 commit)
`fb85e0c33`

Swarm = V1's federated multi-brain runtime. Not in V2.

### Sleep-wake (1 commit)
`a7ed505eb`

V1's REM replay + sleep scheduler. V2 has no sleep cycle yet.

### Glial cells (4 commits)
`73bfa9ae4` `624e1749d` `8e7030f47` `9d33f8f51`

V1's astrocyte/oligodendrocyte/microglia networks. V2 has substrate
helpers but no separate glial network type. Would be a new V2 phase.

### Language bindings + binding fixes (5 commits)
`84083d3dd` `4408b40d7` `873dcd942` `2740e26d6` `e58727847`

V1 has 8 language bindings. V2 has Python (pybind) only. Java/Node/Go/C#/
C++/Perl/Rust/Ruby would be added when V2 stabilizes.

### Build / CMake / CUDA gating (4 commits)
`54eb7da3b` `daaef64d0` `e58727847` `4408b40d7`

V1 build system. V2 uses Cargo.

### Ops / monitor / supervisor / Hetzner (10 commits)
`164ef6d04` `e5570a04a` `7856d7f2c` `7e69a2a5f` `312c9b323`
`f79764c9b` `eccb5b567` `0709b61e2` `23dbeae07` `902c04bc3`
`ccd3370c1`

Operational tooling around V1's Python harness. V2 has its own daemon
in `crates/daemon`.

### V1-specific runtime fixes (8 commits)
`2ecc5e81b` `49a45ff8f` `df679d56b` `f2b8af0b0` `0bbfb115d`
`6d49882f0` `45a0c2dca` `ff8c7e957` `8ac02a28e` `53686f3b5`
`74f6b89cf` `3b...` `540bd531b` `96b03a39b` `d4099501a`

CNN/LNN/HNN/FNO substrate wiring, error-path cleanup, etc. V2 has its
own substrate adapter implementations that already cover the
SNN + LNN cases. CNN/HNN/FNO networks don't exist in V2 yet.

### Crash-handler + SIGPIPE handling (2 commits)
`e77b1dd13` `86c9baf05`

V1-specific signal handling. V2's `crates/daemon` already installs
SIGPIPE → ignore (Path-A wave) and uses Rust panic hooks for backtraces.

### Misc V1-only training/metrics (5 commits)
`fb0f3e3ab` `977d8bd94` `1130204a3` `fcd4cf238` `042f2a238`
`5b788d8b6` `d0f00926b` `012e7f3ee`

V1 SNN watchdog, R-STDP dampening, mode-collapse fixes — all already
ported in the SNN-stability port wave (`c23c1fef0` + companion commits).
The watchdog itself is operational tooling around V1's daemon.

### Docs (1 commit)
`d732eae75`

V1 release notes for 2.7.0.

---

## V2 Phase 10 — Conductance-Based PSC (DEFERRED)

The biggest remaining V1→V2 gap is the conductance-based synapse
migration. This is a multi-day port and warrants its own phase rather
than being squeezed into a port-audit campaign.

### Scope (from `dec956ab9`)

- Per-neuron `g_exc`/`g_inh` arrays in `crates/networks/snn/src/lif.rs`
- Reversal-potential driving force in the LIF integrator
- Runtime flag `conductance_enabled` (default OFF) with bit-identical
  disable-path
- `csr_scale_weights()` admin RPC for one-shot weight rescale on
  CB-enable
- Checkpoint header field `cb_rescaled: bool` (replaces V1's sidecar
  marker hack)
- 46 unit/integration/regression/e2e tests

### V2-specific design notes

- The CPU LIF can carry `g_exc` / `g_inh` as additional `Vec<f32>` fields
  alongside `v_mem` — the ChannelLifState is the natural place.
- The GPU path (`LifGpu`) already manages a single contiguous device
  buffer; CB doubles the buffer size (`v_mem` + `g_exc` + `g_inh`).
- The runtime flag should live on `LifParams` (not `SnnConfig`) so it
  can be set per-population — V1 made it global, which V2's
  multi-population SnnNetwork doesn't need.
- The disable path is bit-identical when `conductance_enabled = false`
  — same correctness contract as `NIMCP_STABILITY=1` mode.

### Why deferred

1. The full port is 5,000+ LoC equivalent. The audit campaign should
   not ship its own multi-day phase.
2. CB requires 46 new tests — not portable in a single sitting without
   rushing.
3. The V1 CB rollout exposed 4 operational bugs in the days after merge
   (the `f79764c9b` / `7e69a2a5f` / `312c9b323` chain). V2 should
   incorporate those lessons as design constraints rather than as
   follow-up commits.

This phase is added to `V2_PLAN.md` as **Phase 10**.

---

## Verification

After this campaign:

```bash
cd crates && cargo test --workspace --all-targets
```

passes 100% — the disable-path contract is intact and the new sentinel
guard / autotune controller / leak attribution helpers are covered by
unit tests.
