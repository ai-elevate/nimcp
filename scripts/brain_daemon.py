#!/usr/bin/env python3
"""
brain_daemon.py — Persistent Brain Server Daemon

Loads the NIMCP brain once and keeps it resident in memory. Training scripts,
chat tools, and monitoring tools connect via Unix socket IPC.

Protocol: length-prefixed JSON over Unix domain socket.
  Request:  4-byte big-endian length + JSON bytes
  Response: 4-byte big-endian length + JSON bytes

Commands:
  {"cmd": "ping"}
  {"cmd": "learn_vector", "features": [...], "target": [...],
   "label": "...", "confidence": 0.7, "learning_rate": 0.01}
  {"cmd": "learn_vector_batch", "pairs": [[[f...],[t...]], ...],
   "learning_rate": 0.01}
  {"cmd": "decide_full", "features": [...]}
  {"cmd": "predict", "features": [...]}
  {"cmd": "speak", "output_vector": [...]}
  {"cmd": "generate_text", "output_vector": [...]}
  {"cmd": "grounded_respond", "text": "..."}
  {"cmd": "lnn_forward_step", "features": [...]}
  {"cmd": "lnn_get_state"}
  {"cmd": "save", "path": "..."}
  {"cmd": "status"}
  {"cmd": "stats"}
  {"cmd": "get_accuracy"}
  {"cmd": "get_last_gradient_norm"}
  {"cmd": "get_neuron_count"}
  {"cmd": "get_transcript"}
  {"cmd": "get_cognitive_stats"}
  {"cmd": "substrate_get_health"}
  {"cmd": "substrate_get_metabolic"}
  {"cmd": "medulla_get_arousal"}
  {"cmd": "sleep_get_pressure"}
  {"cmd": "sleep_get_state"}
  {"cmd": "sleep_is_needed"}
  {"cmd": "sleep_run_cycle", "duration": 2}
  {"cmd": "update_medulla", "dt": 0.1}
  {"cmd": "medulla_get_circadian_efficiency"}
  {"cmd": "bg_get_dopamine"}
  {"cmd": "bg_get_rpe"}
  {"cmd": "bg_get_conflict"}
  {"cmd": "bg_get_mode"}
  {"cmd": "bg_update_reward", "reward": 0.5, "rpe": 0.3}
  {"cmd": "set_plasticity_state", "state": "ACQUISITION"}
  {"cmd": "set_task_type", "task_type": "developmental"}
  {"cmd": "set_fast_training", "enabled": true}
  {"cmd": "enable_biological_plasticity", "enabled": true}
  {"cmd": "consolidate", "mode": "auto"}
  {"cmd": "cerebellum_process_error", "error": 0.5}
  {"cmd": "utm_get_training_health"}
  {"cmd": "utm_forward_only", "features": [...]}
  {"cmd": "experience", "modality": "text", "data": "...", "confidence": 0.7}
  {"cmd": "probe"}
  {"cmd": "shutdown"}

Usage:
  python3 scripts/brain_daemon.py                              # Fresh brain
  python3 scripts/brain_daemon.py --checkpoint path/to/brain   # Load checkpoint
  python3 scripts/brain_daemon.py --resume                     # Auto-resume latest

systemd: see /etc/systemd/system/athena-brain.service
"""

import argparse
import collections
import json
import logging
import os
import signal
import socket
import struct
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

# Add scripts/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("brain_daemon")

SOCKET_PATH = "/var/run/athena/brain.sock"
SOCKET_LOCK = "/var/run/athena/brain.sock.lock"
PID_FILE = "/var/run/athena/brain.pid"
MAX_MSG_SIZE = 50 * 1024 * 1024  # 50 MB (batch learning can be large)

# SNN-primary architecture defaults
DEFAULT_ANN_NEURONS = 150_000      # ANN teacher (SNN is primary learner)
DEFAULT_SNN_NEURONS = 1_800_000    # Hierarchical SNN target
DEFAULT_LNN_NEURONS = 512          # LNN cap (O(n²) adjoint, sweet spot)
CHECKPOINT_MIN_BYTES_PER_NEURON = 50  # Minimum checkpoint bytes per neuron
CHECKPOINT_MIN_SIZE = 5_000_000    # Absolute minimum checkpoint size (5 MB)

# Runtime-tunable SNN knob persistence. Any value set via the snn_tune
# command is written here so it survives daemon restarts. On startup the
# daemon reapplies every entry after configure_cognitive(). Living in the
# checkpoint dir means the rsync-back to Hetzner mirrors it for free.
SNN_TUNE_PERSIST_PATH = "/workspace/nimcp/checkpoints/athena/snn_tune.json"

# Conductance-based rescale checkpoint marker (sidecar file). Used to
# avoid double-rescale across save+load cycles. See cb_rescaled_marker.py.
import cb_rescaled_marker  # noqa: E402

# Track the checkpoint path the daemon loaded from, so the load-time
# rescale check (in _load_persistent_snn_tunes) can consult the marker
# pinned to that file. Set in main() right after brain.load().
_LOADED_CHECKPOINT_PATH = None

# CB rescale factor used everywhere — one source of truth.
CB_DEFAULT_RESCALE_FACTOR = 1.0 / 50.0


# ---------------------------------------------------------------------------
# Layer A confabulation mitigation: "I don't know" gate (added 2026-04-26).
#
# When the brain emits a low-confidence / high-entropy / OOD prediction we
# substitute the response with an explicit "I don't know" payload so the
# downstream battery (`run_metacognition_dk` in scripts/tests/batteries.py)
# detects refusal via its keyword set and HIGH_CONFABULATION drops.
#
# Three independent signals (any one trips the gate):
#   1. Top-1 confidence < IDK_CONFIDENCE_THRESHOLD
#   2. Output Shannon entropy > IDK_ENTROPY_RATIO * log(n_classes)
#   3. OOD score > IDK_OOD_THRESHOLD (only when the brain exposes one)
#
# Disable globally by setting NIMCP_IDK_GATE=0 in the daemon's environment.
# ---------------------------------------------------------------------------
def _env_float(name, default):
    try:
        v = os.environ.get(name)
        return float(v) if v is not None and v != "" else float(default)
    except Exception:
        return float(default)


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return v.strip().lower() not in ("0", "false", "no", "off", "")


IDK_CONFIDENCE_THRESHOLD = _env_float("NIMCP_IDK_CONFIDENCE", 0.30)
IDK_ENTROPY_RATIO = _env_float("NIMCP_IDK_ENTROPY_RATIO", 0.70)
IDK_OOD_THRESHOLD = _env_float("NIMCP_IDK_OOD", 1.0)
IDK_GATE_ENABLED = _env_bool("NIMCP_IDK_GATE", True)


def _idk_extract_confidence(result):
    """Best-effort confidence extraction from a predict-style result.

    Handles three shapes:
      - dict with 'confidence' (or 'adjusted_confidence', 'top_confidence')
      - tuple/list with confidence at position [1]
      - anything else → returns None (signal unavailable)
    """
    if isinstance(result, dict):
        for k in ("confidence", "adjusted_confidence", "top_confidence"):
            if k in result:
                try:
                    return float(result[k])
                except Exception:
                    return None
        return None
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        try:
            return float(result[1])
        except Exception:
            return None
    return None


def _idk_extract_distribution(result):
    """Pull a probability/logit vector out of the result if present.

    Looks at common dict keys; returns a list[float] or None if unavailable.
    Does NOT add a new C getter — purely opportunistic.
    """
    if not isinstance(result, dict):
        return None
    for k in ("probabilities", "probs", "distribution",
              "output", "output_vector", "logits"):
        v = result.get(k)
        if isinstance(v, (list, tuple)) and len(v) > 1:
            try:
                return [float(x) for x in v]
            except Exception:
                continue
    return None


def _idk_compute_entropy_ratio(dist):
    """Shannon entropy / log(n) for a probability vector. Returns None if
    the vector is degenerate (single class or all-zero / negative)."""
    if not dist or len(dist) < 2:
        return None
    import math
    # If the vector looks like logits (negatives or sum != ~1), softmax it.
    s = sum(dist)
    if any(x < 0.0 for x in dist) or s <= 0.0 or abs(s - 1.0) > 0.05:
        # Softmax with numerical-stability shift.
        m = max(dist)
        try:
            exps = [math.exp(x - m) for x in dist]
        except OverflowError:
            return None
        z = sum(exps)
        if z <= 0.0:
            return None
        probs = [e / z for e in exps]
    else:
        probs = list(dist)
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log(p)
    h_max = math.log(len(probs))
    if h_max <= 0.0:
        return None
    return h / h_max


def _idk_extract_ood_score(brain, result):
    """Look for an OOD-related signal. Checks the result dict first, then
    falls back to brain.get_internal_state() (best-effort, no-op on error)."""
    if isinstance(result, dict):
        for k in ("ood_score", "ood_distance", "ood"):
            if k in result:
                try:
                    return float(result[k])
                except Exception:
                    pass
        # Boolean ood_flag is treated as max-OOD when True.
        if result.get("ood_flag") or result.get("is_ood"):
            return float("inf")
    if brain is None:
        return None
    if not hasattr(brain, "get_internal_state"):
        return None
    try:
        state = brain.get_internal_state(strategy=1)
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    for k in ("ood_score", "ood_distance", "ood"):
        if k in state:
            try:
                return float(state[k])
            except Exception:
                pass
    if state.get("ood_flag"):
        return float("inf")
    return None


def _apply_idk_gate(result, brain, stats=None):
    """Wrap a predict-style result with an "I don't know" override when
    uncertainty signals fire. Returns the (possibly substituted) result.

    Signals (any one trips):
      - confidence < IDK_CONFIDENCE_THRESHOLD            → "low_confidence"
      - entropy / log(n) > IDK_ENTROPY_RATIO             → "high_entropy"
      - ood_score > IDK_OOD_THRESHOLD                    → "ood"

    When the gate is bypassed via NIMCP_IDK_GATE=0, the result is returned
    unchanged. The optional `stats` dict (BrainService._stats) gets its
    `idk_gate_trips` counter bumped exactly once per trip.
    """
    if not IDK_GATE_ENABLED:
        return result
    if result is None:
        return result

    conf = _idk_extract_confidence(result)
    dist = _idk_extract_distribution(result)
    ent_ratio = _idk_compute_entropy_ratio(dist) if dist is not None else None
    ood = _idk_extract_ood_score(brain, result)

    reasons = []
    if conf is not None and conf < IDK_CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")
    if ent_ratio is not None and ent_ratio > IDK_ENTROPY_RATIO:
        reasons.append("high_entropy")
    if ood is not None and ood > IDK_OOD_THRESHOLD:
        reasons.append("ood")

    if not reasons:
        return result

    if isinstance(stats, dict):
        stats["idk_gate_trips"] = stats.get("idk_gate_trips", 0) + 1

    return {
        "answer": "I don't know",
        "label": "I don't know",
        "confidence": conf if conf is not None else 0.0,
        "reason": ",".join(reasons),
        "idk_gate": True,
    }


def _load_persistent_snn_tunes(brain, logger):
    """Reapply all knob overrides saved from previous sessions. Best-effort:
    a malformed file or unknown name should never block daemon startup.

    CB activation is handled separately by _activate_cb_default(), which runs
    unconditionally so a `--fresh` daemon also gets CB on. This function
    only loads non-CB knobs from the persistent file.
    """
    try:
        if not os.path.exists(SNN_TUNE_PERSIST_PATH):
            logger.info("[snn_tune] no persistent overrides file at %s",
                        SNN_TUNE_PERSIST_PATH)
            return
        with open(SNN_TUNE_PERSIST_PATH) as f:
            overrides = json.load(f)
    except Exception as e:
        logger.warning("[snn_tune] failed to load persistent overrides: %s", e)
        return
    if not isinstance(overrides, dict) or not overrides:
        return

    applied = 0
    for name, value in overrides.items():
        try:
            # Skip CB knobs entirely — _activate_cb_default owns them.
            if name in ('cb_weights_rescaled', 'conductance_enabled'):
                continue
            # Daemon-level knobs are kept in _runtime_state; SNN knobs go
            # through brain.snn_tune. Routing here mirrors _cmd_snn_tune.
            if name == 'autotune_enabled':
                with _runtime_state_lock:
                    _runtime_state['autotune_enabled'] = bool(float(value))
            elif name == 'sleep_interval_sec':
                with _runtime_state_lock:
                    _runtime_state['sleep_interval_sec'] = float(value)
            else:
                brain.snn_tune(str(name), float(value))
            applied += 1
        except Exception as e:
            logger.warning("[snn_tune] persistent override %s=%s failed: %s",
                           name, value, e)

    logger.info("[snn_tune] reapplied %d persistent override(s) from %s",
                applied, SNN_TUNE_PERSIST_PATH)


def _activate_cb_default(brain, logger):
    """Activate conductance-based PSCs (CB) on the SNN. Runs unconditionally
    after brain init so production daemons always boot with CB on — this is
    the natural-saturation defense against runaway firing rates (driving
    force (V-E) collapses as V → E_exc).

    Default: ON. To disable for tests/debug, set
        {"conductance_enabled": 0.0}
    in SNN_TUNE_PERSIST_PATH; this function honors the explicit override.

    Idempotency across save/load:
      - cb_weights_rescaled is a runtime flag tied to in-memory weight state
        — never persisted (its old value is meaningless on a fresh process).
      - The cb_rescaled_marker sidecar (written after each save-while-CB-on)
        records whether the checkpoint on disk has CB-rescaled weights.
      - --fresh process: no checkpoint loaded ⇒ no marker ⇒ rescale.
      - --resume from un-marked checkpoint ⇒ rescale + marker written on
        next save.
      - --resume from marked checkpoint ⇒ skip rescale (already done).
    Without this guard a double-rescale produces 50× too strong drive ⇒
    exactly the dead↔runaway pattern CB is supposed to prevent.

    On any failure: leave CB OFF and log loudly. Better silent than runaway.
    """
    # Read explicit-disable from persistent file (default ON).
    explicit_off = False
    try:
        if os.path.exists(SNN_TUNE_PERSIST_PATH):
            with open(SNN_TUNE_PERSIST_PATH) as f:
                overrides = json.load(f)
            if isinstance(overrides, dict):
                if 'conductance_enabled' in overrides:
                    explicit_off = float(overrides['conductance_enabled']) == 0.0
    except Exception as e:
        logger.warning("[cb] failed reading persistent override (proceeding "
                       "with default ON): %s", e)

    if explicit_off:
        logger.info("[cb] conductance_enabled=0.0 in %s — leaving CB OFF",
                    SNN_TUNE_PERSIST_PATH)
        try:
            brain.snn_tune('conductance_enabled', 0.0)
            brain.snn_tune('cb_weights_rescaled', 0.0)
        except Exception:
            pass
        return

    try:
        brain.snn_tune('cb_weights_rescaled', 0.0)
        already_rescaled = cb_rescaled_marker.is_marked(_LOADED_CHECKPOINT_PATH)
        if already_rescaled:
            logger.info("[cb] loaded checkpoint already CB-rescaled (marker "
                        "valid on %s) — skipping force-rescale",
                        _LOADED_CHECKPOINT_PATH)
            brain.snn_tune('cb_weights_rescaled', 1.0)
        else:
            logger.info("[cb] no rescale marker on %s — force-rescaling "
                        "weights with factor %g",
                        _LOADED_CHECKPOINT_PATH, CB_DEFAULT_RESCALE_FACTOR)
            brain.snn_rescale_for_conductance(CB_DEFAULT_RESCALE_FACTOR)
            # Marker write deferred until next successful save. Crash-before-
            # save just re-rescales the same un-marked checkpoint on next
            # boot — harmless (disk side untouched; in-memory re-applies).
        brain.snn_tune('conductance_enabled', 1.0)
        logger.info("[cb] CB activated by default; conductance_enabled=1, "
                    "cb_weights_rescaled=%s",
                    "1 (from marker)" if already_rescaled else "1 (fresh rescale)")
    except Exception as e:
        logger.error("[cb] CB activation FAILED — leaving CB OFF for safety. "
                     "SNN runaway protection (natural saturation) IS NOT "
                     "ACTIVE. Investigate: %s", e)
        try:
            brain.snn_tune('conductance_enabled', 0.0)
            brain.snn_tune('cb_weights_rescaled', 0.0)
        except Exception:
            pass


def _save_persistent_snn_tune(name, value, logger):
    """Merge a single knob override into the persist file. Atomic write so
    a crash mid-write doesn't leave the file half-formed."""
    try:
        existing = {}
        if os.path.exists(SNN_TUNE_PERSIST_PATH):
            try:
                with open(SNN_TUNE_PERSIST_PATH) as f:
                    existing = json.load(f) or {}
                if not isinstance(existing, dict):
                    existing = {}
            except Exception:
                existing = {}
        existing[str(name)] = float(value)
        os.makedirs(os.path.dirname(SNN_TUNE_PERSIST_PATH), exist_ok=True)
        tmp = SNN_TUNE_PERSIST_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
        os.replace(tmp, SNN_TUNE_PERSIST_PATH)
    except Exception as e:
        logger.warning("[snn_tune] failed to persist %s=%s: %s", name, value, e)


# ---------------------------------------------------------------------------
# SNN auto-tune controller — closed-loop knob adjustment.
#
# Three actuators, one shared signal (post-cycle SNN recovery curve):
#   - max_scale_dead   : homeostatic scale-up rate for dead pops [1.01, 1.10]
#   - sleep_interval   : sec between periodic sleep cycles      [300,  3600]
#   - noise_rate_hz    : Poisson background floor               [10,   100]
#
# Default mode: OBSERVATION ONLY (autotune_enabled=0). The controller
# measures peak rate and recovery time each sleep cycle and logs its
# would-be decisions, but applies nothing. Set autotune_enabled=1 in
# snn_tune.json (or via the snn_tune socket command) to enable actuator
# updates.
#
# Decision logic per completed cycle (priority order, at most ONE actuator
# moves per cycle so we can attribute changes cleanly):
#   1. SNN collapsed (peak_rate < 5 Hz two cycles in a row):
#      → noise_rate_hz += 10 (rescue jolt)
#   2. Recovery too slow (recovery_time > 0.7 × interval):
#      → sleep_interval += 60s   (give more wake time)
#   3. Peak too high (peak_rate > 500 Hz):
#      → max_scale_dead -= 0.005 × (peak/500) × (1 + 0.5×overshoot_streak)
#        (proportional + integral: 7× overshoots get a 7× nudge, and each
#        consecutive hit amplifies further so the brake catches up)
#   4. Peak too low (peak_rate < 30 Hz, NOT collapsed):
#      → max_scale_dead += 0.005 (boost recovery rate)
#   5. Healthy band (peak_rate 30-500 Hz) AND noise_rate_hz above baseline:
#      → noise_rate_hz -= 5 (taper rescue dose; fires every cycle once
#        SNN is no longer dead, since lingering high noise contributes
#        to the dead↔runaway oscillation)
#
# The shared `_runtime_state` dict carries values between the sleep
# scheduler thread (writes cycle metrics) and the autotuner thread (reads
# them and writes back interval changes). All access goes through
# _runtime_state_lock — both threads run at slow cadence so contention is
# negligible.
# ---------------------------------------------------------------------------

_runtime_state_lock = threading.Lock()
_runtime_state = {
    # Sleep cycle plumbing — populated by _sleep_scheduler each cycle.
    'sleep_interval_sec': 900,           # autotuner can mutate this
    'sleep_last_cycle_complete_ts': 0.0,
    'sleep_last_cycle_duration_s': 0.0,
    'sleep_cycles_completed': 0,
    # Autotuner state.
    'autotune_enabled': True,            # default ON — Rule 3/5 brake active
                                         # Set False via persistent JSON for
                                         # observation-only runs.
    'autotune_cycles_observed': 0,
    'autotune_collapse_streak': 0,
    'autotune_healthy_streak': 0,
    'autotune_overshoot_streak': 0,
    'autotune_last_action': '',
}


def _autotune_clamp(name, value):
    """Hard min/max for each actuator. Defense in depth — even if the
    decision logic produces a runaway value, the clamp pins it sane."""
    bounds = {
        'max_scale_dead':  (1.01, 1.10),
        'sleep_interval':  (300.0, 3600.0),
        'noise_rate_hz':   (10.0, 100.0),
    }
    lo, hi = bounds[name]
    return max(lo, min(hi, value))


def _snn_autotuner(brain, service, shutdown_event, logger):
    """Closed-loop SNN knob controller. Runs in its own daemon thread.

    Observes the SNN every 30s during the wake window between sleep cycles,
    captures peak rate + recovery time, then on each cycle boundary emits a
    log line and (if enabled) applies at most one actuator move."""
    last_observed_cycles = 0
    cycle_peak_rate = 0.0
    cycle_recovery_t = None     # seconds from last cycle to first sample with sparsity < 0.5
    cycle_active_t = 0.0        # seconds spent with sparsity in [0.05, 0.15]
    last_obs_ts = time.time()

    while not shutdown_event.is_set():
        if shutdown_event.wait(timeout=30):
            return
        try:
            with _runtime_state_lock:
                cycles_done = _runtime_state['sleep_cycles_completed']
                last_cycle_ts = _runtime_state['sleep_last_cycle_complete_ts']
                interval = _runtime_state['sleep_interval_sec']
                enabled = _runtime_state['autotune_enabled']

            # New cycle boundary — wrap up the previous cycle's measurement.
            if cycles_done != last_observed_cycles and last_observed_cycles > 0:
                _autotune_decide(brain, logger, enabled,
                                 peak_rate=cycle_peak_rate,
                                 recovery_t=cycle_recovery_t,
                                 active_t=cycle_active_t,
                                 interval=interval)
                cycle_peak_rate = 0.0
                cycle_recovery_t = None
                cycle_active_t = 0.0
            last_observed_cycles = cycles_done

            # Sample SNN state (best-effort; brain may be busy).
            try:
                stats = brain.snn_get_stats()
            except Exception:
                continue
            rate = float(stats.get('mean_firing_rate', 0.0) or 0.0)
            sparsity = float(stats.get('sparsity', 1.0) or 1.0)
            now = time.time()
            dt = now - last_obs_ts
            last_obs_ts = now

            # Track peak rate seen since the last cycle.
            if rate > cycle_peak_rate:
                cycle_peak_rate = rate
            # First sample where sparsity drops out of trough = recovered.
            if cycle_recovery_t is None and last_cycle_ts > 0 and sparsity < 0.5:
                cycle_recovery_t = now - last_cycle_ts
            # Time spent in biological active band.
            if 0.05 <= sparsity <= 0.15:
                cycle_active_t += dt
        except Exception as e:
            logger.warning("[autotune] tick failed: %s", e)


def _autotune_decide(brain, logger, enabled, peak_rate, recovery_t,
                     active_t, interval):
    """Evaluate one cycle's measurements and emit a decision. Applies at
    most one actuator move (priority order). Always logs."""
    with _runtime_state_lock:
        _runtime_state['autotune_cycles_observed'] += 1
        cycles = _runtime_state['autotune_cycles_observed']
        collapse_streak = _runtime_state['autotune_collapse_streak']
        healthy_streak = _runtime_state['autotune_healthy_streak']
        overshoot_streak = _runtime_state['autotune_overshoot_streak']

    # Update streak counters.
    if peak_rate < 5.0:
        collapse_streak += 1
        healthy_streak = 0
        overshoot_streak = 0
    elif peak_rate > 500.0:
        overshoot_streak += 1
        healthy_streak = 0
        collapse_streak = 0
    elif peak_rate >= 30.0 and peak_rate <= 500.0:
        # Healthy band: 30-500 Hz. Active-time gate dropped — the
        # post-stimulus burst window is short and was rejecting valid
        # healthy cycles, leaving rescue noise stuck at 70 Hz.
        healthy_streak += 1
        collapse_streak = 0
        overshoot_streak = 0
    else:
        collapse_streak = 0
        healthy_streak = 0
        overshoot_streak = 0

    # Decide.
    action = None       # tuple of (knob_name, new_value, reason)
    rec_str = f"{recovery_t:.0f}s" if recovery_t is not None else "n/a"
    log_line = (f"[autotune] cycle={cycles} peak={peak_rate:.1f}Hz "
                f"recovery={rec_str} active={active_t:.0f}s "
                f"interval={interval:.0f}s collapse_streak={collapse_streak} "
                f"healthy_streak={healthy_streak} "
                f"overshoot_streak={overshoot_streak}")

    if collapse_streak >= 2:
        # Rule 1: collapsed → bump noise floor.
        try:
            cur = float(brain.snn_tune_get().get('noise_rate_hz', 20.0))
        except Exception:
            cur = 20.0
        new = _autotune_clamp('noise_rate_hz', cur + 10.0)
        if new != cur:
            action = ('noise_rate_hz', new,
                      f'rescue: collapse_streak={collapse_streak}')
    elif recovery_t is not None and recovery_t > 0.7 * interval:
        # Rule 2: recovery slow → lengthen interval.
        new = _autotune_clamp('sleep_interval', interval + 60.0)
        if new != interval:
            action = ('sleep_interval', new,
                      f'recovery_t={recovery_t:.0f}s > 0.7*interval')
    elif peak_rate > 500.0:
        # Rule 3: overshooting → dampen scale. Step is proportional to
        # overshoot magnitude (P) and amplified by consecutive-hit
        # streak (I). Capped at 0.05 so a single tick can't slam the
        # scale to its lower bound.
        try:
            cur = float(brain.snn_tune_get().get('max_scale_dead', 1.05))
        except Exception:
            cur = 1.05
        p_factor = peak_rate / 500.0                 # 1.0 at threshold
        i_factor = 1.0 + 0.5 * overshoot_streak      # 1.0, 1.5, 2.0, 2.5...
        step = min(0.05, 0.005 * p_factor * i_factor)
        new = _autotune_clamp('max_scale_dead', cur - step)
        if new != cur:
            action = ('max_scale_dead', new,
                      f'peak={peak_rate:.0f}Hz > 500 '
                      f'(P={p_factor:.2f} I={i_factor:.1f} step={step:.4f})')
    elif 5.0 <= peak_rate < 30.0:
        # Rule 4: under-recovery (not collapsed) → boost scale.
        try:
            cur = float(brain.snn_tune_get().get('max_scale_dead', 1.05))
        except Exception:
            cur = 1.05
        new = _autotune_clamp('max_scale_dead', cur + 0.005)
        if new != cur:
            action = ('max_scale_dead', new,
                      f'peak={peak_rate:.0f}Hz in [5,30)')
    elif healthy_streak >= 1:
        # Rule 5: any healthy cycle → taper rescue noise. Threshold dropped
        # from 3 → 1 because the dead↔runaway oscillation never strings 3
        # healthy cycles together, leaving noise pinned at 70 Hz.
        try:
            cur = float(brain.snn_tune_get().get('noise_rate_hz', 20.0))
        except Exception:
            cur = 20.0
        if cur > 20.0:
            new = _autotune_clamp('noise_rate_hz', cur - 5.0)
            if new != cur:
                action = ('noise_rate_hz', new,
                          f'healthy_streak={healthy_streak}, taper noise')

    # Update streak counters back into shared state.
    with _runtime_state_lock:
        _runtime_state['autotune_collapse_streak'] = collapse_streak
        _runtime_state['autotune_healthy_streak'] = healthy_streak
        _runtime_state['autotune_overshoot_streak'] = overshoot_streak

    # Log + (maybe) apply.
    if action is None:
        logger.info(f"{log_line} action=none")
        return
    knob, new_val, reason = action
    if not enabled:
        logger.info(f"{log_line} action=DRY_RUN {knob}={new_val:g} ({reason})")
        with _runtime_state_lock:
            _runtime_state['autotune_last_action'] = (
                f'DRY_RUN {knob}={new_val:g} ({reason})')
        return
    # Apply for real.
    try:
        if knob == 'sleep_interval':
            with _runtime_state_lock:
                _runtime_state['sleep_interval_sec'] = float(new_val)
            _save_persistent_snn_tune('sleep_interval_sec', new_val, logger)
        else:
            brain.snn_tune(knob, float(new_val))
            _save_persistent_snn_tune(knob, new_val, logger)
        logger.info(f"{log_line} action=APPLY {knob}={new_val:g} ({reason})")
        with _runtime_state_lock:
            _runtime_state['autotune_last_action'] = (
                f'APPLY {knob}={new_val:g} ({reason})')
    except Exception as e:
        logger.warning(f"{log_line} action=APPLY {knob}={new_val:g} FAILED: {e}")


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------

def recv_msg(conn):
    """Read a length-prefixed JSON message."""
    hdr = b""
    while len(hdr) < 4:
        chunk = conn.recv(4 - len(hdr))
        if not chunk:
            return None
        hdr += chunk
    length = struct.unpack(">I", hdr)[0]
    if length > MAX_MSG_SIZE:
        return None
    data = b""
    while len(data) < length:
        chunk = conn.recv(min(length - len(data), 65536))
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode("utf-8"))


def send_msg(conn, obj):
    """Send a length-prefixed JSON message."""
    data = json.dumps(obj, default=_json_default).encode("utf-8")
    conn.sendall(struct.pack(">I", len(data)) + data)


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


# ---------------------------------------------------------------------------
# Brain wrapper — dispatches commands to the nimcp Brain object
# ---------------------------------------------------------------------------

class _RWLock:
    """Read-write lock: multiple concurrent readers OR one exclusive writer.

    Inference (decide_full, status, get_*) takes the read lock — they run
    concurrently. Training (learn_vector) takes the write lock — blocks
    until all readers finish, then runs exclusively.

    Writer-priority: when a writer is waiting, new readers block. This
    prevents reader starvation of writers (metrics pusher + chat API
    continuously reading would starve learn_vector writes).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
        self._can_read = threading.Condition(self._lock)
        self._can_write = threading.Condition(self._lock)

    def read_acquire(self):
        with self._lock:
            # Block if a writer is active or waiting (writer priority)
            while self._writer_active or self._writers_waiting > 0:
                self._can_read.wait()
            self._readers += 1

    def read_release(self):
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._can_write.notify()

    def write_acquire(self):
        with self._lock:
            self._writers_waiting += 1
            while self._readers > 0 or self._writer_active:
                self._can_write.wait()
            self._writers_waiting -= 1
            self._writer_active = True

    def write_release(self):
        with self._lock:
            self._writer_active = False
            # Wake all readers and one writer
            self._can_read.notify_all()
            self._can_write.notify()


class BrainService:
    """Thread-safe wrapper around nimcp.Brain that dispatches IPC commands.

    Uses a read-write lock: inference/status commands run concurrently (read lock),
    training commands get exclusive access (write lock).
    """

    def __init__(self, brain):
        self.brain = brain
        self._lock = threading.Lock()  # Legacy — still used for stats
        self._rwlock = _RWLock()
        self._stats = {
            "started_at": time.time(),
            "total_requests": 0,
            "learn_calls": 0,
            "infer_calls": 0,
            "errors": 0,
            # Layer A confabulation gate (2026-04-26): bumped each time
            # _apply_idk_gate substitutes the response with "I don't know".
            "idk_gate_trips": 0,
        }
        # Cache the brain's actual input/output dims so training never
        # mis-sizes targets. Reads from probe() (ground truth — post-checkpoint).
        # hasattr(brain, 'config') is False (no such attr on C binding), so
        # without this cache we'd fall back to hardcoded defaults and the
        # C side would truncate every learn_vector call.
        self._num_inputs = None
        self._num_outputs = None
        try:
            p = brain.probe()
            self._num_inputs = int(p.get("num_inputs", 0)) or None
            self._num_outputs = int(p.get("num_outputs", 0)) or None
            logger.info("Brain dims: num_inputs=%s num_outputs=%s",
                        self._num_inputs, self._num_outputs)
        except Exception as e:
            logger.warning("Could not probe brain dims at init: %s", e)
        # Memory-augmented prompt assembly (optional)
        self._prompt_assembler = None
        try:
            from athena_prompt_assembly import AthenaPromptAssembler
            self._prompt_assembler = AthenaPromptAssembler(brain)
            logger.info("Prompt assembler initialized")
        except ImportError:
            pass

        # Default emotional context applied to ground_word calls when the
        # request omits valence/arousal. Curriculum can drive this via
        # set_grounding_emotion to colour subsequent groundings.
        self._grounding_emotion = (0.0, 0.0)

    # Commands that modify brain state — need exclusive (write) lock
    _WRITE_COMMANDS = frozenset({
        'learn_vector', 'learn_vector_bin',
        'learn_vector_batch', 'learn_vector_batch_bin',
        'train_batch_text',
        'save', 'sleep_run_cycle', 'retrofit_synapse_metadata',
        'ground_word', 'set_grounding_emotion', 'learn_language_pair',
    })

    def handle_readonly(self, req):
        """Lock-free dispatch for read-only socket. Never blocks on training."""
        cmd = req.get("cmd", "")
        self._stats["total_requests"] += 1
        try:
            handler = getattr(self, f"_cmd_{cmd}", None)
            if handler is None:
                return {"error": f"Unknown command: {cmd}"}
            return handler(req)
        except Exception as e:
            self._stats["errors"] += 1
            return {"error": str(e)}

    def handle(self, req):
        """Dispatch a command dict and return a response dict.

        Write commands (learn_vector, etc.) get exclusive access.
        Read commands (decide_full, status, get_*) run concurrently.
        """
        cmd = req.get("cmd", "")
        self._stats["total_requests"] += 1

        try:
            handler = getattr(self, f"_cmd_{cmd}", None)
            if handler is None:
                return {"error": f"Unknown command: {cmd}"}

            # Lock-free for read-only commands (stats dict reads are GIL-atomic).
            # This prevents metrics/status queries from timing out behind learn_vector.
            if cmd in ('ping', 'status', 'get_neuron_count', 'get_snn_stats',
                       'get_network_metrics', 'get_cortex_cnn_metrics',
                       'get_cognitive_stats', 'get_module_activity'):
                return handler(req)

            # All other commands (learn_vector, decide_full, save, etc.) take the lock
            with self._lock:
                return handler(req)
        except Exception as e:
            self._stats["errors"] += 1
            logger.warning("Command %s failed: %s", cmd, e)
            return {"error": str(e), "traceback": traceback.format_exc()}

    # -- Core learning --

    def _cmd_batch(self, req):
        """Execute multiple commands in a single round-trip."""
        commands = req.get("commands", [])
        import sys
        print(f"[BATCH-DBG] {len(commands)} cmds: {[c.get('cmd','?')+'/'+str(c.get('modality','')) for c in commands]}", file=sys.stderr, flush=True)
        results = []
        for cmd_req in commands:
            try:
                handler = getattr(self, f"_cmd_{cmd_req.get('cmd', '')}", None)
                if handler:
                    results.append(handler(cmd_req))
                else:
                    results.append({"error": f"Unknown command: {cmd_req.get('cmd')}"})
            except Exception as e:
                print(f"[BATCH-DBG] ERROR in {cmd_req.get('cmd','?')}/{cmd_req.get('modality','')}: {e}", file=sys.stderr, flush=True)
                results.append({"error": str(e)})
        return {"results": results}

    def _cmd_learn_vector(self, req):
        features = req["features"]
        target = req["target"]
        kwargs = {}
        if "label" in req:
            kwargs["label"] = req["label"]
        if "confidence" in req:
            kwargs["confidence"] = req["confidence"]
        if "learning_rate" in req:
            kwargs["learning_rate"] = req["learning_rate"]

        if True:  # RWLock in handle()
            loss = self.brain.learn_vector(features, target, **kwargs)
        self._stats["learn_calls"] += 1
        if hasattr(self, 'checkpointer') and self.checkpointer:
            self.checkpointer.notify_training_step()
        return {"loss": loss}

    def _cmd_learn_vector_bin(self, req):
        """Fast binary learn_vector — float arrays as base64 instead of JSON lists."""
        import numpy as np
        from base64 import b64decode
        features = np.frombuffer(b64decode(req["f_b64"]), dtype=np.float32).tolist()
        target = np.frombuffer(b64decode(req["t_b64"]), dtype=np.float32).tolist()
        kwargs = {}
        if "label" in req:
            kwargs["label"] = req["label"]
        if "confidence" in req:
            kwargs["confidence"] = req["confidence"]
        if "learning_rate" in req:
            kwargs["learning_rate"] = req["learning_rate"]
        if True:  # RWLock in handle()
            loss = self.brain.learn_vector(features, target, **kwargs)
        self._stats["learn_calls"] += 1
        if hasattr(self, 'checkpointer') and self.checkpointer:
            self.checkpointer.notify_training_step()
        return {"loss": loss}

    def _cmd_learn_vector_batch(self, req):
        pairs = req["pairs"]
        kwargs = {}
        if "learning_rate" in req:
            kwargs["learning_rate"] = req["learning_rate"]

        # Convert to list of tuples
        pair_tuples = [(p[0], p[1]) for p in pairs]
        if True:  # RWLock in handle()
            avg_loss = self.brain.learn_vector_batch(pair_tuples, **kwargs)
        self._stats["learn_calls"] += 1
        if hasattr(self, 'checkpointer') and self.checkpointer:
            self.checkpointer.notify_training_step()
        return {"avg_loss": avg_loss}

    def _cmd_learn_vector_batch_bin(self, req):
        """Binary batch learn — base64-encoded concatenated float arrays."""
        import numpy as np
        from base64 import b64decode
        n_pairs = req["n_pairs"]
        f_dim = req["f_dim"]
        t_dim = req["t_dim"]
        f_all = np.frombuffer(b64decode(req["f_b64"]), dtype=np.float32)
        t_all = np.frombuffer(b64decode(req["t_b64"]), dtype=np.float32)
        pairs = []
        for i in range(n_pairs):
            f = f_all[i * f_dim:(i + 1) * f_dim].tolist()
            t = t_all[i * t_dim:(i + 1) * t_dim].tolist()
            pairs.append((f, t))
        kwargs = {}
        if "learning_rate" in req:
            kwargs["learning_rate"] = req["learning_rate"]
        if True:  # RWLock in handle()
            avg_loss = self.brain.learn_vector_batch(pairs, **kwargs)
        self._stats["learn_calls"] += n_pairs
        if hasattr(self, 'checkpointer') and self.checkpointer:
            self.checkpointer.notify_training_step()
        return {"avg_loss": avg_loss}

    def _cmd_train_batch_text(self, req):
        """Batch training from raw text — ONNX encodes + learns in tight loop.

        Protocol:
            {"cmd": "train_batch_text",
             "items": [{"text": "...", "label": "...", "target_text": "..."}, ...],
             "learning_rate": 0.001  // optional
            }

        Each item: ONNX-encode text → compose features → make_semantic_target →
        brain.learn_vector. All in daemon process, zero socket overhead between items.
        """
        import time
        try:
            from onnx_encoder import encode_batch
        except ImportError:
            return {"error": "ONNX encoder not available — install onnxruntime"}

        items = req["items"]
        n = len(items)
        if n == 0:
            return {"avg_loss": 0.0, "n_items": 0}

        lr = req.get("learning_rate", None)

        # Separate feature texts (description) and target texts (name + description)
        feat_texts = [it.get("text", "") for it in items]
        tgt_texts = [it.get("target_text", it.get("text", "")) for it in items]
        labels = [it.get("label", "")[:50] for it in items]

        t0 = time.time()
        # Batch-encode both feature and target texts in two ONNX calls
        feat_embs = encode_batch(feat_texts)    # [N, 1024] — input features
        tgt_embs = encode_batch(tgt_texts)      # [N, 1024] — target embeddings
        t_encode = time.time() - t0

        # Get brain output dim for target tiling (cached from probe() at init)
        out_dim = self._num_outputs if self._num_outputs else 4096
        import numpy as _np

        # Train each item — tight loop, no socket overhead between items
        total_loss = 0.0
        n_ok = 0
        kwargs = {}
        if lr is not None:
            kwargs["learning_rate"] = lr

        for i in range(n):
            # Features: tiled to brain input dim
            feat = feat_embs[i]
            in_dim = self._num_inputs if self._num_inputs else 1024
            if len(feat) < in_dim:
                reps = (in_dim + len(feat) - 1) // len(feat)
                feat = _np.tile(feat, reps)[:in_dim]
            features = feat.tolist()

            # Target: tiled to brain output dim
            tgt = tgt_embs[i]
            if len(tgt) < out_dim:
                reps = (out_dim + len(tgt) - 1) // len(tgt)
                tgt = _np.tile(tgt, reps)[:out_dim]
            target = tgt.tolist()

            label = labels[i] if i < len(labels) else ""

            loss = self.brain.learn_vector(features, target,
                                            label=label, confidence=0.65,
                                            **kwargs)
            if loss is not None and loss >= 0:
                total_loss += loss
                n_ok += 1

        self._stats["learn_calls"] += n_ok
        if hasattr(self, 'checkpointer') and self.checkpointer:
            self.checkpointer.notify_training_step()

        avg_loss = total_loss / n_ok if n_ok > 0 else 0.0
        t_total = time.time() - t0
        return {
            "avg_loss": avg_loss,
            "n_items": n_ok,
            "encode_ms": t_encode * 1000,
            "total_ms": t_total * 1000,
            "ms_per_item": (t_total / n * 1000) if n > 0 else 0,
        }

    # -- Inference --

    def _cmd_decide_full(self, req):
        features = req["features"]
        if True:  # RWLock in handle()
            result = self.brain.decide_full(features)
        self._stats["infer_calls"] += 1

        # Memory-augmented prompt assembly (if requested)
        if req.get('enrich') and self._prompt_assembler:
            try:
                enriched = self._prompt_assembler.assemble(
                    input_text=req.get('text', ''),
                    brain_output=result if isinstance(result, dict) else {},
                    features=features)
                if isinstance(result, dict):
                    result['enriched_prompt'] = enriched.get('prompt', '')
                    result['adjusted_confidence'] = enriched.get('confidence', 0.0)
                    result['is_ood'] = enriched.get('is_ood', False)
            except Exception:
                pass  # Don't break inference for enrichment errors

        return {"result": result}

    def _cmd_predict(self, req):
        features = req["features"]
        if True:  # RWLock in handle()
            result = self.brain.predict(features)
        self._stats["infer_calls"] += 1
        # Layer A confabulation gate (2026-04-26).
        result = _apply_idk_gate(result, self.brain, self._stats)
        return {"result": result}

    def _cmd_speak(self, req):
        output_vector = req["output_vector"]
        if True:  # RWLock in handle()
            result = self.brain.speak(output_vector)
        return {"result": result}

    def _cmd_generate_text(self, req):
        output_vector = req["output_vector"]
        if True:  # RWLock in handle()
            result = self.brain.generate_text(output_vector)
        return {"result": result}

    def _cmd_grounded_respond(self, req):
        text = req["text"]
        if True:  # RWLock in handle()
            result = self.brain.grounded_respond(text)
        return {"result": result}

    def _cmd_bootstrap_lexicon(self, req):
        """Bootstrap the grounded lexicon from a JSON fixture.

        Best-effort: returns success flag + the count of words now in
        the lexicon (post-bootstrap). The C side handles malformed
        entries gracefully so this should never raise.
        """
        path = req.get("path")
        if not path:
            return {"error": "missing 'path' field"}
        try:
            result = self.brain.bootstrap_lexicon(path)
        except AttributeError:
            return {"error": "bootstrap_lexicon not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"bootstrap_lexicon: {e}"}
        # Surface post-bootstrap vocab size so callers can verify the load.
        try:
            diag = self.brain.get_grounded_language_diagnostics()
            vocab_size = int(diag.get("vocab_size", 0)) if isinstance(diag, dict) else 0
        except Exception:
            vocab_size = 0
        out = {"vocab_size": vocab_size}
        if isinstance(result, dict):
            out.update(result)
        else:
            out["result"] = result
        return out

    # -- LNN --

    def _cmd_lnn_forward_step(self, req):
        features = req["features"]
        if True:  # RWLock in handle()
            result = self.brain.lnn_forward_step(features)
        return {"result": result}

    def _cmd_lnn_get_state(self, _req):
        if True:  # RWLock in handle()
            state = self.brain.lnn_get_state()
        return {"state": state}

    # -- Checkpoint --

    def _cmd_save(self, req):
        path = req["path"]
        if True:  # RWLock in handle()
            self.brain.save(path)
        # Mirror the auto-checkpointer: when CB is on, the in-memory weights
        # are rescaled, so the file we just wrote is rescaled too. Pin a
        # marker so a later --resume skips force-rescale (double-rescale → SNN silent).
        try:
            cb_on = float(self.brain.snn_tune_get().get(
                'conductance_enabled', 0.0)) != 0.0
            if cb_on:
                cb_rescaled_marker.write_marker(path, CB_DEFAULT_RESCALE_FACTOR)
        except Exception as _cbm_e:
            logger.warning("CB marker write failed for manual save %s (save still good): %s",
                           path, _cbm_e)
        return {"ok": True, "path": path}

    # -- Monitoring --

    def _cmd_ping(self, _req):
        return {"ok": True, "uptime": time.time() - self._stats["started_at"]}

    def _cmd_status(self, _req):
        return {
            "ok": True,
            "uptime": time.time() - self._stats["started_at"],
            **self._stats,
        }

    def _cmd_stats(self, _req):
        if True:  # RWLock in handle()
            try:
                stats = self.brain.get_stats()
            except Exception:
                stats = {}
        return {"stats": stats, **self._stats}

    def _cmd_get_accuracy(self, _req):
        if True:  # RWLock in handle()
            return {"accuracy": self.brain.get_accuracy()}

    def _cmd_get_last_gradient_norm(self, _req):
        if True:  # RWLock in handle()
            return {"gradient_norm": self.brain.get_last_gradient_norm()}

    def _cmd_get_neuron_count(self, _req):
        if True:  # RWLock in handle()
            return {"neuron_count": self.brain.get_neuron_count()}

    def _cmd_get_immune_state(self, _req):
        try:
            state = self.brain.get_immune_state()
        except AttributeError:
            return {"error": "get_immune_state not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"get_immune_state: {e}"}
        return {"immune": state}

    def _cmd_get_grounded_language_diagnostics(self, _req):
        try:
            d = self.brain.get_grounded_language_diagnostics()
        except AttributeError:
            return {"error": "get_grounded_language_diagnostics not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"get_grounded_language_diagnostics: {e}"}
        return {"grounded_language": d}

    def _cmd_get_bigram_spectral_metrics(self, _req):
        """PA-4+ FFT bigram spectral diagnostics — surfaces grammar
        emergence as three scalar metrics. Returns zeros on a fresh
        brain that hasn't yet seen any bigrams."""
        try:
            m = self.brain.get_bigram_spectral_metrics()
        except AttributeError:
            return {"error": "get_bigram_spectral_metrics not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"get_bigram_spectral_metrics: {e}"}
        return {"bigram_spectral_metrics": m}

    def _cmd_ground_word(self, req):
        """Ground a single word in sensory features.

        Request keys:
          word        (str, required)
          features    (list[float], required) — sensory feature vector
          modality    (int, default 5 = LINGUISTIC)
          attention   (float, default 0.7)
          valence     (float, optional) — defaults to self._grounding_emotion[0]
          arousal     (float, optional) — defaults to self._grounding_emotion[1]
        """
        word = req.get("word")
        features = req.get("features")
        if not word or features is None:
            return {"error": "ground_word requires 'word' and 'features'"}
        try:
            modality = int(req.get("modality", 5))
            attention = float(req.get("attention", 0.7))
            default_v, default_a = self._grounding_emotion
            valence = float(req.get("valence", default_v))
            arousal = float(req.get("arousal", default_a))
        except (TypeError, ValueError) as e:
            return {"error": f"ground_word bad arg: {e}"}
        try:
            ok = self.brain.ground_word(word, list(features),
                                         modality=modality,
                                         attention=attention,
                                         valence=valence,
                                         arousal=arousal)
        except AttributeError:
            return {"error": "ground_word not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"ground_word: {e}"}
        return {"ok": bool(ok), "word": word}

    def _cmd_set_grounding_emotion(self, req):
        """Set the default emotional context for subsequent ground_word calls."""
        try:
            valence = float(req.get("valence", 0.0))
            arousal = float(req.get("arousal", 0.0))
        except (TypeError, ValueError) as e:
            return {"error": f"set_grounding_emotion bad arg: {e}"}
        self._grounding_emotion = (valence, arousal)
        return {"ok": True, "valence": valence, "arousal": arousal}

    def _cmd_learn_language_pair(self, req):
        """Train an input→target text pair via grounded_language."""
        text = req.get("text")
        target_text = req.get("target_text")
        if not text or not target_text:
            return {"error": "learn_language_pair requires 'text' and 'target_text'"}
        try:
            learning_rate = float(req.get("learning_rate", 0.05))
        except (TypeError, ValueError) as e:
            return {"error": f"learn_language_pair bad arg: {e}"}
        try:
            result = self.brain.learn_language_pair(text, target_text,
                                                    learning_rate=learning_rate)
        except AttributeError:
            return {"unavailable": True,
                    "reason": "learn_language_pair binding not present — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"learn_language_pair: {e}"}
        if isinstance(result, dict):
            out = {"ok": True}
            out.update(result)
            return out
        return {"ok": True, "result": result}

    def _cmd_get_top_phrases(self, req):
        try:
            top_k = int(req.get("top_k", 20))
        except (TypeError, ValueError):
            top_k = 20
        try:
            phrases = self.brain.get_top_phrases(top_k=top_k)
        except AttributeError:
            return {"error": "get_top_phrases not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"get_top_phrases: {e}"}
        return {"phrases": phrases}

    def _cmd_get_modality_counts(self, _req):
        # Curriculum coverage probe: returns {"counts": {visual,auditory,...}}.
        # When GL is uninitialised the binding returns an empty dict rather
        # than raising, so we propagate that as {"counts": {}} for the client.
        try:
            counts = self.brain.get_modality_counts()
        except AttributeError:
            return {"error": "get_modality_counts not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"get_modality_counts: {e}"}
        return {"counts": counts}

    def _cmd_probe_comprehend(self, req):
        text = req.get("text", "")
        max_components = int(req.get("max_components", 16))
        try:
            d = self.brain.probe_comprehend(text, max_components)
        except AttributeError:
            return {"error": "probe_comprehend not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"probe_comprehend: {e}"}
        return {"probe": d}

    def _cmd_set_snn_language_bridge_blend(self, req):
        blend = float(req.get("blend", 0.5))
        try:
            self.brain.set_snn_language_bridge_blend(blend)
        except AttributeError:
            return {"error": "set_snn_language_bridge_blend not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_blend: {e}"}
        return {"ok": True, "blend": blend}

    def _cmd_recompute_snn_language_bridge_norms(self, _req):
        """Patch A salvage: rebuild bridge per-word_pop binding-weight L2 norm
        cache. Use once after upgrading a pre-Patch-A daemon to seed the cache
        from existing bindings without restart."""
        try:
            self.brain.recompute_snn_language_bridge_norms()
        except AttributeError:
            return {"error": "recompute_snn_language_bridge_norms not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"recompute_norms: {e}"}
        return {"ok": True}

    def _cmd_set_snn_language_bridge_sampling(self, req):
        """PA-6: configure produce-time sampling.

        Request keys:
          temperature  (float, default 0.0) — 0 = argmax (legacy);
                       >0 = softmax sampling over top-K candidates.
          top_p        (float, default 1.0) — nucleus truncation in (0,1].
        """
        try:
            temperature = float(req.get("temperature", 0.0))
            top_p = float(req.get("top_p", 1.0))
        except (TypeError, ValueError) as e:
            return {"error": f"set_sampling bad arg: {e}"}
        try:
            self.brain.set_snn_language_bridge_sampling(temperature, top_p)
        except AttributeError:
            return {"error": "set_snn_language_bridge_sampling not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_sampling: {e}"}
        return {"ok": True, "temperature": temperature, "top_p": top_p}

    def _cmd_set_snn_language_bridge_glove_blend(self, req):
        """PA-5: set GloVe-aware decode blend coefficient.

        Request keys:
          blend  (float, required, in [0,1]) — 0 = binding-only,
                 1 = embedding-only. Active only when grounded_language
                 has wired its embedding lookup down to the bridge.
        """
        try:
            blend = float(req.get("blend", 0.0))
        except (TypeError, ValueError) as e:
            return {"error": f"set_glove_blend bad arg: {e}"}
        try:
            self.brain.set_snn_language_bridge_glove_blend(blend)
        except AttributeError:
            return {"error": "set_snn_language_bridge_glove_blend not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_glove_blend: {e}"}
        return {"ok": True, "blend": blend}

    def _cmd_set_snn_language_bridge_autoregressive(self, req):
        """PA-2: configure autoregressive recurrent decoder.

        Request keys:
          intent_persistence  (float, default 0.0, in [0,1]) — 0 reproduces
                              legacy 70/30 in-place blend (intent decays
                              as state evolves). 1 keeps intent at full
                              strength every step.
          word_feedback       (float, default 0.3, in [0,1]) — how much
                              each picked word reshapes the recurrent state.
        """
        try:
            intent_persistence = float(req.get("intent_persistence", 0.0))
            word_feedback      = float(req.get("word_feedback", 0.3))
        except (TypeError, ValueError) as e:
            return {"error": f"set_autoregressive bad arg: {e}"}
        try:
            self.brain.set_snn_language_bridge_autoregressive(
                intent_persistence, word_feedback)
        except AttributeError:
            return {"error": "set_snn_language_bridge_autoregressive not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_autoregressive: {e}"}
        return {"ok": True,
                "intent_persistence": intent_persistence,
                "word_feedback":      word_feedback}

    def _cmd_set_snn_language_bridge_spike_routing(self, req):
        """PA-3: master gate for SNN-spike → bridge STDP routing.

        Request keys:
          enabled  (bool, required)
          tau_ms   (float, default 200) — activation decay time constant.
        """
        enabled = bool(req.get("enabled", False))
        try:
            tau_ms = float(req.get("tau_ms", 200.0))
        except (TypeError, ValueError) as e:
            return {"error": f"set_spike_routing bad arg: {e}"}
        try:
            self.brain.set_snn_language_bridge_spike_routing(enabled, tau_ms)
        except AttributeError:
            return {"error": "set_snn_language_bridge_spike_routing not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_spike_routing: {e}"}
        return {"ok": True, "enabled": enabled, "tau_ms": tau_ms}

    def _cmd_set_snn_language_bridge_hyperbolic_embeddings(self, req):
        """PA-5+: toggle Poincaré hyperbolic-distance GloVe metric.

        Request keys:
          enabled  (bool, required)
        """
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_snn_language_bridge_hyperbolic_embeddings(enabled)
        except AttributeError:
            return {"error": "set_snn_language_bridge_hyperbolic_embeddings not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_hyperbolic_embeddings: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_set_snn_language_bridge_sampling_mode(self, req):
        """PA-6+: select produce-time sampling mode.

        Request keys:
          mode  (int, required) — 0=auto/PA-6, 1=softmax+top-p, 2=q-MC.
        """
        try:
            mode = int(req.get("mode", 0))
        except (TypeError, ValueError) as e:
            return {"error": f"set_sampling_mode bad arg: {e}"}
        try:
            self.brain.set_snn_language_bridge_sampling_mode(mode)
        except AttributeError:
            return {"error": "set_snn_language_bridge_sampling_mode not available — rebuild nimcp.so"}
        except ValueError as e:
            return {"error": f"set_sampling_mode rejected: {e}"}
        except Exception as e:
            return {"error": f"set_sampling_mode: {e}"}
        return {"ok": True, "mode": mode}

    def _cmd_set_snn_language_bridge_beam_width(self, req):
        """TIER1-A: enable / configure beam-K decoding in produce.

        Request keys:
          k  (int, default 1) — 1 = greedy / legacy bit-for-bit. > 1 = beam
             search with length-normalized log-prob ranking. Capped at 16.
        """
        try:
            k = int(req.get("k", 1))
        except (TypeError, ValueError) as e:
            return {"error": f"set_beam_width bad arg: {e}"}
        try:
            self.brain.set_snn_language_bridge_beam_width(k)
        except AttributeError:
            return {"error": "set_snn_language_bridge_beam_width not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_beam_width: {e}"}
        return {"ok": True, "k": k}

    def _cmd_set_snn_language_bridge_eos_word_pop(self, req):
        """TIER1-B: register end-of-utterance word_pop.

        Request keys:
          pop  (int, default 4294967295 = disabled) — word_pop index. When
               sampled inside produce, generation halts cleanly and the EOS
               form is NOT appended to the output text.
        """
        try:
            pop = int(req.get("pop", 0xFFFFFFFF))
        except (TypeError, ValueError) as e:
            return {"error": f"set_eos_word_pop bad arg: {e}"}
        try:
            self.brain.set_snn_language_bridge_eos_word_pop(pop)
        except AttributeError:
            return {"error": "set_snn_language_bridge_eos_word_pop not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_eos_word_pop: {e}"}
        return {"ok": True, "pop": pop}

    def _cmd_set_snn_language_bridge_repetition_penalty(self, req):
        """TIER1-C: configure n-gram repetition penalty.

        Request keys:
          penalty  (float, default 0.0, in [0,1]) — 0 disables. >0 multiplies
                   any candidate's score by (1-penalty) per match in the
                   last `window` picks.
          window   (int, default 3) — look-back length. 0 falls back to 3
                   when penalty > 0.
        """
        try:
            penalty = float(req.get("penalty", 0.0))
            window  = int(req.get("window", 3))
        except (TypeError, ValueError) as e:
            return {"error": f"set_repetition_penalty bad arg: {e}"}
        try:
            self.brain.set_snn_language_bridge_repetition_penalty(penalty, window)
        except AttributeError:
            return {"error": "set_snn_language_bridge_repetition_penalty not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_repetition_penalty: {e}"}
        return {"ok": True, "penalty": penalty, "window": window}

    def _cmd_set_anaphora_enabled(self, req):
        """Tier-1 #2: toggle rule-based anaphora / pronoun resolution.

        Request keys:
          enabled  (bool, required) — true = resolve pronouns inside
                                      grounded_language_comprehend.
        """
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_anaphora_enabled(enabled)
        except AttributeError:
            return {"error": "set_anaphora_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_anaphora_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_set_grounded_engram_enabled(self, req):
        """EN-5: toggle read-only engram integration on grounded_language.

        Request keys:
          enabled  (bool, required) — true = comprehend lays down engram
                                      traces and blends recalled neurons
                                      into activated_concepts. Brain's
                                      engram_system is wired in
                                      automatically.
        """
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_grounded_engram_enabled(enabled)
        except AttributeError:
            return {"error": "set_grounded_engram_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_grounded_engram_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_set_grounded_immune_enabled(self, req):
        """IM-3: toggle Tier-3 immune content inspection on grounded_language.

        Request keys:
          enabled  (bool, required) — true = comprehend runs read-only
                                      content heuristics (NaN/Inf,
                                      statistical outliers, repetition
                                      spam, lexicon collision, negation
                                      cascades), damps confidence by the
                                      computed inflammation level,
                                      registers an antigen on inflammation
                                      > 0.5, and skips engram encode on
                                      inflammation > 0.7. Brain's
                                      immune_system is wired in
                                      automatically.
        """
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_grounded_immune_enabled(enabled)
        except AttributeError:
            return {"error": "set_grounded_immune_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_grounded_immune_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_get_snn_language_bridge_config(self, _req):
        """Tier-4 #15: read the full SNN-language bridge config.

        Returns every PA/MQ knob (temperature, top_p, glove_blend,
        intent_persistence, word_feedback, sampling_mode,
        use_hyperbolic_embeddings, enable_snn_spike_routing, activation_tau_ms,
        STDP τ/A constants, capacities, etc.) — the consolidated read-side
        counterpart to the per-knob setters.
        """
        try:
            cfg = self.brain.get_snn_language_bridge_config()
        except AttributeError:
            return {"error": "get_snn_language_bridge_config not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"get_config: {e}"}
        return {"ok": True, "config": cfg}

    def _cmd_set_snn_language_bridge_rng_seed(self, req):
        """Tier-4 #17: explicitly seed the SNN-language bridge sampling RNG.

        Use this to make PA-6 / MQ-A sampling tests deterministic across
        runs — without it, the RNG is seeded from time XOR pointer-mix.
        seed=0 is silently remapped to 1 (xorshift64 needs nonzero state).

        Request keys:
          seed  (int, required) — uint64 seed value.
        """
        try:
            seed = int(req.get("seed", 0))
        except (TypeError, ValueError) as e:
            return {"error": f"set_rng_seed bad arg: {e}"}
        if seed < 0:
            return {"error": "set_rng_seed: seed must be >= 0 (uint64)"}
        try:
            self.brain.set_snn_language_bridge_rng_seed(seed)
        except AttributeError:
            return {"error": "set_snn_language_bridge_rng_seed not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_rng_seed: {e}"}
        return {"ok": True, "seed": seed}

    def _cmd_learn_next_token_pair(self, req):
        """PA-4: contrastive next-token training on a single bigram.

        Request keys: prev (str), next (str), lr (float, default 0.05).
        """
        prev_word = req.get("prev")
        next_word = req.get("next")
        if not prev_word or not next_word:
            return {"error": "learn_next_token_pair requires 'prev' and 'next'"}
        try:
            lr = float(req.get("lr", 0.05))
        except (TypeError, ValueError) as e:
            return {"error": f"learn_next_token_pair bad lr: {e}"}
        try:
            applied = bool(self.brain.learn_next_token_pair(prev_word,
                                                              next_word, lr))
        except AttributeError:
            return {"error": "learn_next_token_pair not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"learn_next_token_pair: {e}"}
        return {"ok": True, "applied": applied}

    def _cmd_learn_next_token_pair_riemannian(self, req):
        """PA-4+: Riemannian / sigmoid-reparameterized next-token training.

        Same request shape as learn_next_token_pair (prev, next, lr).
        Returns the same response shape ({"ok": True, "applied": bool}).
        """
        prev_word = req.get("prev")
        next_word = req.get("next")
        if not prev_word or not next_word:
            return {"error": "learn_next_token_pair_riemannian requires 'prev' and 'next'"}
        try:
            lr = float(req.get("lr", 0.05))
        except (TypeError, ValueError) as e:
            return {"error": f"learn_next_token_pair_riemannian bad lr: {e}"}
        try:
            applied = bool(self.brain.learn_next_token_pair_riemannian(prev_word,
                                                                         next_word, lr))
        except AttributeError:
            return {"error": "learn_next_token_pair_riemannian not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"learn_next_token_pair_riemannian: {e}"}
        return {"ok": True, "applied": applied}

    def _cmd_learn_text_bigrams(self, req):
        """PA-4: walk the text's bigrams and apply next-token training.

        Request keys: text (str), lr (float, default 0.05).
        Returns: count of applied bigram updates. If trigram learning is
        on (set_trigram_learning_enabled), each step also walks
        (w_t, w_{t+1}) → w_{t+2} at half lr; the trigram count is tracked
        separately on the bridge stats (total_trigram_updates).
        """
        text = req.get("text")
        if not text:
            return {"error": "learn_text_bigrams requires 'text'"}
        try:
            lr = float(req.get("lr", 0.05))
        except (TypeError, ValueError) as e:
            return {"error": f"learn_text_bigrams bad lr: {e}"}
        try:
            count = int(self.brain.learn_text_bigrams(text, lr))
        except AttributeError:
            return {"error": "learn_text_bigrams not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"learn_text_bigrams: {e}"}
        return {"ok": True, "applied": count}

    def _cmd_set_trigram_learning_enabled(self, req):
        """TA-4: toggle trigram next-token learning.

        Request keys: enabled (bool, default false). Default OFF
        preserves PA-4 bigram-only behavior bit-for-bit. Runtime-only;
        not persisted across saves.
        """
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_trigram_learning_enabled(enabled)
        except AttributeError:
            return {"error": "set_trigram_learning_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_trigram_learning_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    # =====================================================================
    # Audit fix: campaign feature setters via daemon RPC.
    # All persisted via mt_save_config now (LANC block) so the trainer
    # only needs to set them once per training campaign.
    # =====================================================================

    def _cmd_set_da_modulation_enabled(self, req):
        """TA-3: toggle dopamine-modulated STDP."""
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_da_modulation_enabled(enabled)
        except AttributeError:
            return {"error": "set_da_modulation_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_da_modulation_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_set_comprehend_stdp_enabled(self, req):
        """CSTDP: toggle comprehend-driven scoped STDP on the SNN language
        bridge. When enabled, every comprehend call reinforces existing
        strong concept↔word bindings via inline STDP — turns lexicon-side
        training signal into bridge-weight reinforcement without a
        separate supervised loop. Default OFF; entrenchment risk on
        early-training brains with mostly-noise bindings."""
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_comprehend_stdp_enabled(enabled)
        except AttributeError:
            return {"error": "set_comprehend_stdp_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_comprehend_stdp_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_echo_and_correct(self, req):
        """Echo-correct: comprehend(parent_text) → strengthen
        (active concepts → target_word) bindings. Supervised production-
        side learning loop. Returns count of bindings strengthened
        (0 means target_word isn't registered in the bridge yet — it
        hasn't been mirrored from the lexicon side).

        Request: {"cmd": "echo_and_correct", "parent_text": "...",
                   "target_word": "...", "lr_scale": 1.0}
        Response: {"ok": True, "pairs_strengthened": <int>}"""
        parent_text = req.get("parent_text", "")
        target_word = req.get("target_word", "")
        try:
            lr_scale = float(req.get("lr_scale", 1.0))
        except (TypeError, ValueError):
            return {"error": "echo_and_correct: lr_scale must be a number"}
        if not parent_text or not target_word:
            return {"error": "echo_and_correct: parent_text and target_word required"}
        try:
            pairs = self.brain.echo_and_correct(parent_text, target_word, lr_scale=lr_scale)
        except AttributeError:
            return {"error": "echo_and_correct not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"echo_and_correct: {e}"}
        return {"ok": True, "pairs_strengthened": int(pairs)}

    def _cmd_produce_cascade(self, req):
        """Phase 2A multi-region production cascade. Walks 9 stages
        (drive, goal, listener, episodic, content, lexical, syntactic,
        phonological, motor) — stages 7-9 are deferred until Phase 2C/E
        but plumbing is in place. Each cognitive module is queried for
        real state when attached; no-ops gracefully when missing.

        Request: {"cmd": "produce_cascade", "prompt": "..." (optional)}
        Response: {"ok": True, "utterance": str, "word_count": int,
                   "confidence": float}

        prompt=None runs the cascade purely from internal state
        (spontaneous-speech mode); prompt=str runs it as a response."""
        prompt = req.get("prompt")
        try:
            result = self.brain.produce_cascade(prompt)
        except AttributeError:
            return {"error": "produce_cascade not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"produce_cascade: {e}"}
        return {"ok": True,
                "utterance":             result.get("utterance",  ""),
                "word_count":            result.get("word_count", 0),
                "confidence":            result.get("confidence", 0.0),
                "self_match":            result.get("self_match", 0.0),
                "self_grammaticality":   result.get("self_grammaticality", 0.0),
                "prompt_is_question":    bool(result.get("prompt_is_question", False)),
                "prompt_is_imperative":  bool(result.get("prompt_is_imperative", False)),
                "wernicke_parsed":       bool(result.get("wernicke_parsed", False))}

    def _cmd_set_da_modulation_gain(self, req):
        """TA-3: tune the DA → LR scaling. Clamped [0, 200].

        Audit-2 B4: returns the effective (post-clamp) value, not the
        request, so callers see what actually got applied.
        """
        try:
            gain = float(req.get("gain", 50.0))
        except (TypeError, ValueError) as e:
            return {"error": f"set_da_modulation_gain bad arg: {e}"}
        try:
            self.brain.set_da_modulation_gain(gain)
        except AttributeError:
            return {"error": "set_da_modulation_gain not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_da_modulation_gain: {e}"}
        # Audit-2 B4: read back via bridge config for the effective value.
        try:
            cfg = self.brain.get_snn_language_bridge_config()
            effective = float(cfg.get("da_modulation_gain", gain))
        except Exception:
            effective = gain
        return {"ok": True, "gain": effective, "requested": gain}

    def _cmd_set_reconsolidation_enabled(self, req):
        """TA-5: toggle reconsolidation-on-contradiction. Default OFF."""
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_reconsolidation_enabled(enabled)
        except AttributeError:
            return {"error": "set_reconsolidation_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_reconsolidation_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_set_reconsolidation_decay(self, req):
        """TA-5: tune binding decay per contradiction event. Clamped [0, 0.5].

        Audit-2 B4: returns the effective (post-clamp) value via diagnostics.
        """
        try:
            decay = float(req.get("decay", 0.05))
        except (TypeError, ValueError) as e:
            return {"error": f"set_reconsolidation_decay bad arg: {e}"}
        try:
            self.brain.set_reconsolidation_decay(decay)
        except AttributeError:
            return {"error": "set_reconsolidation_decay not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_reconsolidation_decay: {e}"}
        try:
            d = self.brain.get_grounded_language_diagnostics()
            effective = float(d.get("reconsolidation_decay", decay))
        except Exception:
            effective = decay
        return {"ok": True, "decay": effective, "requested": decay}

    def _cmd_set_sentence_segmentation_enabled(self, req):
        """TB-6: toggle sentence-boundary segmentation. Default OFF."""
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_sentence_segmentation_enabled(enabled)
        except AttributeError:
            return {"error": "set_sentence_segmentation_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_sentence_segmentation_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_set_length_control(self, req):
        """TB-7: produce length cap. min/max=0 disables that side."""
        try:
            min_words = int(req.get("min_words", 0))
            max_words = int(req.get("max_words", 0))
        except (TypeError, ValueError) as e:
            return {"error": f"set_length_control bad arg: {e}"}
        try:
            self.brain.set_length_control(min_words, max_words)
        except AttributeError:
            return {"error": "set_length_control not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_length_control: {e}"}
        return {"ok": True, "min_words": min_words, "max_words": max_words}

    def _cmd_set_speech_act_classification_enabled(self, req):
        """TB-9: toggle speech-act intent classification. Default OFF."""
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_speech_act_classification_enabled(enabled)
        except AttributeError:
            return {"error": "set_speech_act_classification_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_speech_act_classification_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_set_topic_shift_enabled(self, req):
        """TB-10: toggle topic-shift detection. Default OFF."""
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_topic_shift_enabled(enabled)
        except AttributeError:
            return {"error": "set_topic_shift_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_topic_shift_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_set_topic_shift_threshold(self, req):
        """TB-10: tune topic-shift cosine threshold. Clamped [0, 1].

        Audit-2 B4: returns the effective (post-clamp) value via diagnostics.
        """
        try:
            threshold = float(req.get("threshold", 0.3))
        except (TypeError, ValueError) as e:
            return {"error": f"set_topic_shift_threshold bad arg: {e}"}
        try:
            self.brain.set_topic_shift_threshold(threshold)
        except AttributeError:
            return {"error": "set_topic_shift_threshold not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_topic_shift_threshold: {e}"}
        try:
            d = self.brain.get_grounded_language_diagnostics()
            effective = float(d.get("topic_shift_threshold", threshold))
        except Exception:
            effective = threshold
        return {"ok": True, "threshold": effective, "requested": threshold}

    def _cmd_lang_status(self, _req):
        """Audit-2 follow-up: single-call summary of the entire lang surface.

        Returns:
          flags: dict of every campaign feature flag (bool) + tunables.
          stats: vocab + comprehension + production counters.
          decode: TC-11 latency metrics (avg µs/call, total calls, total ns).
                  avg_decode_us > 100 is the trigger threshold for the
                  deferred GPU decode kernel.

        No flag mutations — purely a read-side aggregator. Pulls from
        get_grounded_language_diagnostics() which now includes the
        campaign-flag snapshot fields (audit-2 follow-up).
        """
        try:
            d = self.brain.get_grounded_language_diagnostics()
        except AttributeError:
            return {"error": "get_grounded_language_diagnostics not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"lang_status: {e}"}

        decode_calls = int(d.get("bridge_total_decode_calls", 0))
        decode_ns = int(d.get("bridge_decode_total_ns", 0))
        avg_us = (decode_ns / decode_calls / 1000.0) if decode_calls > 0 else 0.0

        flags = {
            "negation_inversion":         bool(d.get("enable_negation_inversion", False)),
            "sense_disambiguation":       bool(d.get("enable_sense_disambiguation", False)),
            "speech_act_classification":  bool(d.get("enable_speech_act_classification", False)),
            "sentence_segmentation":      bool(d.get("enable_sentence_segmentation", False)),
            "topic_shift_detection":      bool(d.get("enable_topic_shift_detection", False)),
            "reconsolidation":            bool(d.get("enable_reconsolidation", False)),
            "anaphora_resolution":        bool(d.get("enable_anaphora_resolution", False)),
            "bridge_da_modulation":       bool(d.get("bridge_enable_da_modulation", False)),
            "bridge_trigram_learning":    bool(d.get("bridge_enable_trigram_learning", False)),
        }
        tunables = {
            "reconsolidation_decay":   float(d.get("reconsolidation_decay", 0.0)),
            "topic_shift_threshold":   float(d.get("topic_shift_threshold", 0.0)),
            "topic_shift_min_turns":   int(d.get("topic_shift_min_turns", 0)),
            "snn_bridge_blend":        float(d.get("snn_bridge_blend", -1.0)),
        }
        stats = {
            "vocab_size":                int(d.get("vocab_size", 0)),
            "total_bindings":            int(d.get("total_bindings", 0)),
            "total_groundings":          int(d.get("total_groundings", 0)),
            "total_comprehensions":      int(d.get("total_comprehensions", 0)),
            "total_productions":         int(d.get("total_productions", 0)),
            "bridge_total_productions":  int(d.get("bridge_total_productions", 0)),
            "bridge_active_bindings":    int(d.get("bridge_active_bindings", 0)),
            "avg_binding_strength":      float(d.get("avg_binding_strength", 0.0)),
            "avg_comprehension_confidence": float(d.get("avg_comprehension_confidence", 0.0)),
        }
        decode = {
            "total_calls":          decode_calls,
            "total_ns":             decode_ns,
            "avg_us_per_call":      round(avg_us, 3),
            "gpu_port_threshold_us": 100.0,
            "above_gpu_threshold":  avg_us > 100.0,
        }
        return {"ok": True, "flags": flags, "tunables": tunables,
                "stats": stats, "decode": decode}

    def _cmd_set_dialect(self, req):
        """Audit-2 B13: dialect / accent conditioning. None or empty clears."""
        dialect = req.get("dialect", None)
        if dialect is not None and not isinstance(dialect, str):
            return {"error": f"set_dialect bad arg: dialect must be str or null, got {type(dialect).__name__}"}
        try:
            self.brain.set_dialect(dialect)
        except AttributeError:
            return {"error": "set_dialect not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_dialect: {e}"}
        return {"ok": True, "dialect": dialect or ""}

    def _cmd_set_topic_shift_min_turns(self, req):
        """TB-10: tune minimum turns before topic-shift fires. Clamped to discourse cap.

        Audit-2 B4: returns the effective (post-clamp) value via diagnostics.
        """
        try:
            n = int(req.get("min_turns", 3))
        except (TypeError, ValueError) as e:
            return {"error": f"set_topic_shift_min_turns bad arg: {e}"}
        try:
            self.brain.set_topic_shift_min_turns(n)
        except AttributeError:
            return {"error": "set_topic_shift_min_turns not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_topic_shift_min_turns: {e}"}
        try:
            d = self.brain.get_grounded_language_diagnostics()
            effective = int(d.get("topic_shift_min_turns", n))
        except Exception:
            effective = n
        return {"ok": True, "min_turns": effective, "requested": n}

    def _cmd_learn_next_token_triple(self, req):
        """TA-4: contrastive next-token training on a single trigram.

        Request keys: prev1 (str), prev2 (str), next (str),
                      lr (float, default 0.025 — half the bigram default).
        Returns: {"ok": True, "applied": bool}. False on cold-start or
        no-bridge no-op (any of the three tokens without prior bindings).
        """
        prev1 = req.get("prev1")
        prev2 = req.get("prev2")
        next_word = req.get("next")
        if not prev1 or not prev2 or not next_word:
            return {"error": "learn_next_token_triple requires 'prev1', 'prev2', 'next'"}
        try:
            lr = float(req.get("lr", 0.025))
        except (TypeError, ValueError) as e:
            return {"error": f"learn_next_token_triple bad lr: {e}"}
        try:
            applied = bool(self.brain.learn_next_token_triple(prev1, prev2,
                                                                next_word, lr))
        except AttributeError:
            return {"error": "learn_next_token_triple not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"learn_next_token_triple: {e}"}
        return {"ok": True, "applied": applied}

    # ---- Tier-2 #3 / #6 / #7 grounded-language toggles + discourse buffer ----

    def _cmd_set_grounded_negation_enabled(self, req):
        """Tier-2 #3: toggle comprehend negation polarity inversion.

        Request keys: enabled (bool, required).
        """
        enabled = bool(req.get("enabled", True))
        try:
            self.brain.set_grounded_negation_enabled(enabled)
        except AttributeError:
            return {"error": "set_grounded_negation_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_negation_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_set_grounded_sense_disambiguation_enabled(self, req):
        """Tier-2 #6: toggle comprehend word-sense disambiguation.

        Request keys: enabled (bool, required). Default false reproduces
        legacy "every binding contributes its raw strength" behaviour.
        """
        enabled = bool(req.get("enabled", False))
        try:
            self.brain.set_grounded_sense_disambiguation_enabled(enabled)
        except AttributeError:
            return {"error": "set_grounded_sense_disambiguation_enabled not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"set_sense_disambiguation_enabled: {e}"}
        return {"ok": True, "enabled": enabled}

    def _cmd_grounded_push_turn(self, req):
        """Tier-2 #7: append a turn to the discourse ring buffer.

        Request keys:
          semantic_vec  (list[float] or None) — copied; None pushes a
                        placeholder turn with no vector content.
          n_words       (int, default 0)
          is_user       (bool, default true)
        """
        vec = req.get("semantic_vec")
        try:
            n_words = int(req.get("n_words", 0))
        except (TypeError, ValueError) as e:
            return {"error": f"grounded_push_turn bad n_words: {e}"}
        is_user = bool(req.get("is_user", True))
        try:
            self.brain.grounded_push_turn(vec, n_words, is_user)
        except AttributeError:
            return {"error": "grounded_push_turn not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"grounded_push_turn: {e}"}
        try:
            count = int(self.brain.grounded_get_discourse_turn_count())
        except Exception:
            count = -1
        return {"ok": True, "turn_count": count}

    def _cmd_grounded_get_discourse_turn_count(self, _req):
        """Tier-2 #7: query populated discourse turn count."""
        try:
            count = int(self.brain.grounded_get_discourse_turn_count())
        except AttributeError:
            return {"error": "grounded_get_discourse_turn_count not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"grounded_get_discourse_turn_count: {e}"}
        return {"ok": True, "turn_count": count}

    def _cmd_grounded_set_discourse_capacity(self, req):
        """Tier-2 #7: clamp discourse capacity (1..GL_DISCOURSE_MAX_TURNS).

        Request keys: capacity (int, required, in [1, 8]).
        """
        try:
            capacity = int(req.get("capacity", 8))
        except (TypeError, ValueError) as e:
            return {"error": f"grounded_set_discourse_capacity bad capacity: {e}"}
        try:
            self.brain.grounded_set_discourse_capacity(capacity)
        except AttributeError:
            return {"error": "grounded_set_discourse_capacity not available — rebuild nimcp.so"}
        except Exception as e:
            return {"error": f"grounded_set_discourse_capacity: {e}"}
        return {"ok": True, "capacity": capacity}

    def _cmd_get_alloc_stats(self, _req):
        """Allocator accounting snapshot — mallinfo2 + /proc + audit + KG."""
        try:
            return {"alloc_stats": self.brain.get_alloc_stats()}
        except Exception as e:
            return {"error": f"get_alloc_stats: {e}"}

    # --- Per-network training toggles (dynamic, no rebuild required) ---

    def _cmd_set_train_ann(self, req):
        self.brain.set_train_ann(bool(req.get("enabled", True)))
        return {"ok": True, "train_ann": self.brain.get_train_ann()}

    def _cmd_set_train_cnn(self, req):
        self.brain.set_train_cnn(bool(req.get("enabled", True)))
        return {"ok": True, "train_cnn": self.brain.get_train_cnn()}

    def _cmd_set_train_snn(self, req):
        self.brain.set_train_snn(bool(req.get("enabled", True)))
        return {"ok": True, "train_snn": self.brain.get_train_snn()}

    def _cmd_set_train_lnn(self, req):
        self.brain.set_train_lnn(bool(req.get("enabled", True)))
        return {"ok": True, "train_lnn": self.brain.get_train_lnn()}

    def _cmd_set_snn_only_recovery(self, req):
        self.brain.set_snn_only_recovery(bool(req.get("enabled", True)))
        return {"ok": True, "snn_only_recovery": self.brain.get_snn_only_recovery()}

    def _cmd_set_ensemble_warmup_scale(self, req):
        scale = float(req.get("scale", 1.0))
        self.brain.set_ensemble_warmup_scale(scale)
        return {"ok": True, "warmup_scale": self.brain.get_ensemble_warmup_scale()}

    def _cmd_get_training_flags(self, _req):
        return {
            "train_ann": self.brain.get_train_ann(),
            "train_cnn": self.brain.get_train_cnn(),
            "train_snn": self.brain.get_train_snn(),
            "train_lnn": self.brain.get_train_lnn(),
            "snn_only_recovery": self.brain.get_snn_only_recovery(),
            "ensemble_warmup_scale": self.brain.get_ensemble_warmup_scale(),
        }

    # --- Plateau detector RPCs (live-tunable, no rebuild needed) ---

    def _cmd_get_plateau_detector_params(self, _req):
        """Return the current plateau detector parameters + state."""
        if not hasattr(self, "plateau_detector") or self.plateau_detector is None:
            return {"error": "plateau detector not attached"}
        return self.plateau_detector.get_params()

    def _cmd_set_plateau_detector_params(self, req):
        """Update plateau detector parameters at runtime. Accepts any subset of:
        poll_interval_s, window_size, slope_threshold, min_steps_in_recovery,
        max_steps_in_recovery, absolute_loss_target, warmup_initial_scale,
        warmup_steps. Unknown keys are ignored."""
        if not hasattr(self, "plateau_detector") or self.plateau_detector is None:
            return {"error": "plateau detector not attached"}
        params = {k: v for k, v in req.items() if k != "cmd"}
        return self.plateau_detector.set_params(**params)

    def _cmd_retrofit_synapse_metadata(self, _req):
        if True:  # RWLock in handle()
            count = self.brain.retrofit_synapse_metadata()
            return {"retrofitted": count}

    def _cmd_get_snn_stats(self, _req):
        if True:  # RWLock in handle()
            stats = self.brain.get_snn_stats()
            return {"snn": stats if stats else {}}

    # -- Runtime SNN parameter tuning (no rebuild / no restart) --

    def _cmd_snn_tune(self, req):
        """Set a runtime-tunable SNN parameter.
        Request: {"name": "rstdp_lr", "value": 0.0002}
        Valid names: any of the C-level SNN knobs (rstdp_lr,
        rstdp_baseline_alpha, target_rate, homeo_min_scale, homeo_max_scale,
        max_scale_dead, dead_threshold, metabolic_cap, noise_rate_hz, ...)
        OR daemon-level knobs (autotune_enabled, sleep_interval_sec).
        """
        name = req.get("name")
        value = req.get("value")
        if name is None or value is None:
            return {"error": "snn_tune requires 'name' and 'value'"}
        try:
            if name == 'autotune_enabled':
                with _runtime_state_lock:
                    _runtime_state['autotune_enabled'] = bool(float(value))
            elif name == 'sleep_interval_sec':
                # Clamp through the autotuner's bounds so a typo can't
                # park the daemon in a useless 1-second cycle.
                clamped = _autotune_clamp('sleep_interval', float(value))
                with _runtime_state_lock:
                    _runtime_state['sleep_interval_sec'] = clamped
                value = clamped
            else:
                self.brain.snn_tune(str(name), float(value))
        except ValueError as e:
            return {"error": str(e)}
        _save_persistent_snn_tune(name, value, logger)
        return {"ok": True, "name": name, "value": float(value)}

    def _cmd_snn_tune_get(self, _req):
        """Return dict of all current SNN tunable parameter values, plus
        the daemon-level knobs (autotune_enabled, sleep_interval_sec)."""
        params = dict(self.brain.snn_tune_get() or {})
        with _runtime_state_lock:
            params['autotune_enabled'] = float(_runtime_state['autotune_enabled'])
            params['sleep_interval_sec'] = _runtime_state['sleep_interval_sec']
            params['autotune_cycles_observed'] = _runtime_state['autotune_cycles_observed']
            params['autotune_last_action'] = _runtime_state['autotune_last_action']
        return {"params": params}

    def _cmd_snn_pop_stats(self, _req):
        """Return per-population live firing-rate snapshot."""
        return {"pops": self.brain.snn_pop_stats() or []}

    def _cmd_force_save_now(self, _req):
        """Operator hook: force the auto-checkpointer to save immediately,
        bypassing min_steps gate. Used to capture critical state (e.g.
        post-CB-rescale weights) before the next restart wipes them."""
        if not (hasattr(self, 'checkpointer') and self.checkpointer):
            return {"error": "no auto-checkpointer attached"}
        try:
            self.checkpointer.save_now(force=True)
        except Exception as e:
            return {"error": "save failed: " + str(e)}
        return {"ok": True}

    def _cmd_snn_rescale_for_conductance(self, req):
        """Rescale all CSR synapse weights for CB-mode operation. Idempotent
        (refuses double-apply via the cb_weights_rescaled sticky flag).
        Atomically: rescale weights, then optionally flip conductance_enabled
        on. Both knobs persist to snn_tune.json.

        Request: {"factor": 0.02, "enable_after": true}
          factor       : weight multiplier (default 1/50 = 0.02)
          enable_after : if true, flips conductance_enabled=1 after rescale.
                         Default false — caller can verify rescale before flipping.
        """
        try:
            factor = float(req.get("factor", 1.0 / 50.0))
        except (TypeError, ValueError):
            return {"error": "factor must be numeric"}
        enable_after = bool(req.get("enable_after", False))
        try:
            self.brain.snn_rescale_for_conductance(factor)
        except Exception as e:
            return {"error": "rescale failed: " + str(e)}
        # Note: cb_weights_rescaled is RUNTIME-ONLY — do NOT persist to
        # JSON. The C global tracks "current in-memory weights are
        # rescaled"; that state is volatile across daemon restarts
        # because checkpoint loads bring back un-rescaled weights. The
        # _load_persistent_snn_tunes path consults the cb_rescaled_marker
        # sidecar (written after every save while CB is on) to decide
        # whether to re-apply the rescale, so the runtime flag is
        # reconstructed correctly on every startup.
        # NOTE: marker is NOT written here — in-memory state is now
        # ahead of disk. The next save (auto or manual force_save_now)
        # will write the marker pinned to that save's mtime.
        result = {"ok": True, "factor": factor, "rescaled": True}
        if enable_after:
            try:
                self.brain.snn_tune("conductance_enabled", 1.0)
                _save_persistent_snn_tune("conductance_enabled", 1.0, logger)
                result["conductance_enabled"] = 1.0
            except Exception as e:
                return {"error": "flag-flip failed after rescale: " + str(e),
                        "rescaled": True}
        return result

    def _cmd_get_transcript(self, _req):
        if True:  # RWLock in handle()
            return {"transcript": self.brain.get_transcript()}

    def _cmd_get_cognitive_stats(self, _req):
        if True:  # RWLock in handle()
            return {"stats": self.brain.get_cognitive_stats()}

    def _cmd_probe(self, _req):
        if True:  # RWLock in handle()
            return {"probe": self.brain.probe()}

    def _cmd_get_network_metrics(self, _req):
        if True:  # RWLock in handle()
            m = self.brain.get_network_metrics()
            return {"metrics": m if m else {}}

    def _cmd_get_cortex_cnn_metrics(self, _req):
        if True:  # RWLock in handle()
            m = self.brain.get_cortex_cnn_metrics()
            return {"metrics": m if m else {}}

    def _cmd_get_module_activity(self, _req):
        # Statue-suspect probe counters. See nimcp_brain_get_module_activity
        # docstring (include/nimcp.h) for field meanings.
        if True:  # RWLock in handle()
            try:
                m = self.brain.get_module_activity()
            except Exception as e:
                return {"error": "get_module_activity failed: %s" % e}
            return {"activity": m if m else {}}

    # -- Biological state --

    def _cmd_substrate_get_health(self, _req):
        if True:  # RWLock in handle()
            return {"health": self.brain.substrate_get_health()}

    def _cmd_substrate_get_metabolic(self, _req):
        if True:  # RWLock in handle()
            return {"metabolic": self.brain.substrate_get_metabolic()}

    def _cmd_medulla_get_arousal(self, _req):
        if True:  # RWLock in handle()
            return {"arousal": self.brain.medulla_get_arousal()}

    def _cmd_medulla_get_circadian_efficiency(self, _req):
        if True:  # RWLock in handle()
            return {"efficiency": self.brain.medulla_get_circadian_efficiency()}

    def _cmd_sleep_get_pressure(self, _req):
        if True:  # RWLock in handle()
            return {"pressure": self.brain.sleep_get_pressure()}

    def _cmd_sleep_get_state(self, _req):
        if True:  # RWLock in handle()
            return {"state": self.brain.sleep_get_state()}

    def _cmd_sleep_is_needed(self, _req):
        if True:  # RWLock in handle()
            return {"needed": self.brain.sleep_is_needed()}

    def _cmd_sleep_run_cycle(self, req):
        duration = req.get("duration", 2)
        if True:  # RWLock in handle()
            self.brain.sleep_run_cycle(duration)
        return {"ok": True}

    def _cmd_update_medulla(self, req):
        dt = req.get("dt", 0.1)
        if True:  # RWLock in handle()
            self.brain.update_medulla(dt)
        return {"ok": True}

    def _cmd_bg_get_dopamine(self, _req):
        if True:  # RWLock in handle()
            return {"dopamine": self.brain.bg_get_dopamine()}

    def _cmd_bg_get_rpe(self, _req):
        if True:  # RWLock in handle()
            return {"rpe": self.brain.bg_get_rpe()}

    def _cmd_bg_get_conflict(self, _req):
        if True:  # RWLock in handle()
            return {"conflict": self.brain.bg_get_conflict()}

    def _cmd_bg_get_mode(self, _req):
        if True:  # RWLock in handle()
            return {"mode": self.brain.bg_get_mode()}

    def _cmd_bg_update_reward(self, req):
        reward = req["reward"]
        rpe = req.get("rpe", 0.0)
        if True:  # RWLock in handle()
            self.brain.bg_update_reward(reward, rpe)
        return {"ok": True}

    # -- Training config --

    def _cmd_set_plasticity_state(self, req):
        if True:  # RWLock in handle()
            self.brain.set_plasticity_state(req["state"])
        return {"ok": True}

    def _cmd_set_task_type(self, req):
        if True:  # RWLock in handle()
            self.brain.set_task_type(req["task_type"])
        return {"ok": True}

    def _cmd_set_fast_training(self, req):
        if True:  # RWLock in handle()
            self.brain.set_fast_training(req["enabled"])
        return {"ok": True}

    def _cmd_set_training_mode(self, req):
        """Toggle training-mode fast path in brain_decide().
        When True, brain_decide() skips all cognitive modules (reasoning,
        imagination, ToM, etc.) — only runs forward pass + label + confidence.
        Essential during training to avoid SIGSEGV from lazy-init races."""
        if True:  # RWLock in handle()
            self.brain.set_training_mode(req["enabled"])
        logger.info("Training mode: %s", "ON" if req["enabled"] else "OFF")
        return {"ok": True}

    def _cmd_reinit_weights(self, req):
        if True:  # RWLock in handle()
            self.brain.reinit_weights()
            self._step_count = 0  # Reset step counter after reinit
        logger.info("Weights reinitialized (mode collapse recovery)")
        return {"ok": True}

    def _cmd_set_hyperparams(self, req):
        """Set multiple hyperparameters at once.

        Supported params: sparsity, grad_clip, weight_decay, output_lr_boost,
        dropout, diversity_weight, temperature.

        Usage: {"cmd": "set_hyperparams", "params": {"sparsity": 0.1, "grad_clip": 5.0}}
        """
        params = req.get("params", {})
        applied = {}
        if True:  # RWLock in handle()
            for key, val in params.items():
                try:
                    val = float(val)
                    if key == "sparsity":
                        self.brain.set_network_ablation(sparsity_target=val)
                        applied[key] = val
                    elif key == "grad_clip":
                        # Set via brain config (internal)
                        if hasattr(self.brain, '_internal_brain'):
                            pass  # Config-level, applied at next learn
                        applied[key] = val
                    elif key == "output_lr_boost":
                        applied[key] = val
                    elif key == "weight_decay":
                        applied[key] = val
                    elif key == "dropout":
                        applied[key] = val
                    elif key == "diversity_weight":
                        applied[key] = val
                    elif key == "temperature":
                        applied[key] = val
                    else:
                        logger.warning("Unknown hyperparameter: %s", key)
                except (TypeError, ValueError) as e:
                    logger.warning("Invalid value for %s: %s (%s)", key, val, e)

        # Store in daemon state so learn_vector can use them
        if not hasattr(self, '_hp_overrides'):
            self._hp_overrides = {}
        self._hp_overrides.update(applied)
        logger.info("Hyperparams updated: %s", applied)
        return {"ok": True, "applied": applied}

    def _cmd_get_hyperparams(self, _req):
        """Get current hyperparameter overrides."""
        return {"params": getattr(self, '_hp_overrides', {})}

    def _cmd_enable_biological_plasticity(self, req):
        if True:  # RWLock in handle()
            self.brain.enable_biological_plasticity(req["enabled"])
        return {"ok": True}

    def _cmd_consolidate(self, req):
        mode = req.get("mode", "auto")
        if True:  # RWLock in handle()
            self.brain.consolidate(mode)
        return {"ok": True}

    def _cmd_cerebellum_process_error(self, req):
        if True:  # RWLock in handle()
            self.brain.cerebellum_process_error(req["error"])
        return {"ok": True}

    # -- UTM --

    def _cmd_utm_get_training_health(self, _req):
        if True:  # RWLock in handle()
            return {"health": self.brain.utm_get_training_health()}

    def _cmd_utm_set_early_stopping_enabled(self, req):
        if True:
            self.brain.utm_set_early_stopping_enabled(bool(req.get("enabled", False)))
        return {"ok": True}

    def _cmd_utm_reset_early_stopping(self, _req):
        if True:
            self.brain.utm_reset_early_stopping()
        return {"ok": True}

    def _cmd_utm_forward_only(self, req):
        if True:  # RWLock in handle()
            result = self.brain.utm_forward_only(req["features"])
        return {"result": result}

    # -- Experience --

    def _cmd_experience(self, req):
        if True:  # RWLock in handle()
            self.brain.experience(req["modality"], req["data"],
                                  confidence=req.get("confidence"))
        return {"ok": True}

    # -- Sensory cortex --

    def _cmd_audio_cortex_process(self, req):
        if True:  # RWLock in handle()
            result = self.brain.audio_cortex_process(req["samples"])
        return {"result": result}

    def _cmd_visual_cortex_process(self, req):
        if True:  # RWLock in handle()
            result = self.brain.visual_cortex_process(
                req["pixels"], req["width"], req["height"],
                req.get("channels", 3))
        return {"result": result}

    def _cmd_speech_cortex_process(self, req):
        if True:  # RWLock in handle()
            result = self.brain.speech_cortex_process(req["samples"])
        return {"result": result}

    # -- Sensory input --

    def _cmd_submit_sensory(self, req):
        modality = req["modality"]
        data = req["data"]
        n = len(data) if isinstance(data, (list, tuple)) else -1
        if True:  # RWLock in handle()
            # Stage sensory data on brain struct so cortex CNNs get created
            # and trained during the next learn_vector call.
            # Symmetric logging — prior code only logged visual, creating
            # the false appearance that other modalities were silent.
            if modality == "visual":
                logger.info("submit_sensory VISUAL: %d elements, w=%s h=%s ch=%s",
                            n, req.get("width"), req.get("height"),
                            req.get("channels"))
                self.brain.submit_sensory("visual", data,
                                          width=req.get("width", 32),
                                          height=req.get("height", 32),
                                          channels=req.get("channels", 3))
            elif modality == "audio":
                logger.info("submit_sensory AUDIO: %d samples", n)
                self.brain.submit_sensory("audio", data)
            elif modality == "speech":
                logger.info("submit_sensory SPEECH: %d samples", n)
                self.brain.submit_sensory("speech", data)
            elif modality == "somatosensory" or modality == "somato":
                logger.info("submit_sensory SOMATO: %d elements, n_segments=%s",
                            n, req.get("n_segments", n))
                self.brain.submit_sensory("somatosensory", data,
                                          n_segments=req.get("n_segments", n))
            else:
                logger.warning("submit_sensory UNKNOWN: modality='%s' (ignored)",
                               modality)
                return {"ok": True}
        return {"ok": True}

    # -- Arousal control --

    def _cmd_medulla_boost_arousal(self, req):
        if True:  # RWLock in handle()
            self.brain.medulla_boost_arousal(req.get("amount", 0.1))
        return {"ok": True}

    def _cmd_medulla_reduce_arousal(self, req):
        if True:  # RWLock in handle()
            self.brain.medulla_reduce_arousal(req.get("amount", 0.1))
        return {"ok": True}

    # -- Reward / novelty --

    def _cmd_edp_process_reward(self, req):
        if True:  # RWLock in handle()
            self.brain.edp_process_reward(req["reward"])
        return {"ok": True}

    def _cmd_edp_process_novelty(self, req):
        if True:  # RWLock in handle()
            self.brain.edp_process_novelty(req["novelty"])
        return {"ok": True}

    # -- Language / cognitive training --

    def _cmd_train_cognitive(self, req):
        kwargs = {}
        for k in ("text", "target_text", "learning_rate", "domain"):
            if k in req:
                kwargs[k] = req[k]
        if True:  # RWLock in handle()
            result = self.brain.train_cognitive(**kwargs)
        return {"result": result}

    def _cmd_train_language(self, req):
        if True:  # RWLock in handle()
            self.brain.train_language(req["text"], req.get("target_text", req["text"]))
        return {"ok": True}

    def _cmd_learn_language(self, req):
        if True:  # RWLock in handle()
            self.brain.learn_language(req["text"])
        return {"ok": True}

    # -- Reasoning --

    def _cmd_ti_init_reasoning(self, _req):
        if True:  # RWLock in handle()
            self.brain.ti_init_reasoning()
        return {"ok": True}

    def _cmd_ti_add_fact(self, req):
        if True:  # RWLock in handle()
            self.brain.ti_add_fact(req["fact"], req.get("confidence", 0.5))
        return {"ok": True}

    def _cmd_ti_add_rule(self, req):
        if True:  # RWLock in handle()
            self.brain.ti_add_rule(req["rule"], req.get("confidence", 0.5))
        return {"ok": True}

    def _cmd_ti_forward_chain(self, _req):
        if True:  # RWLock in handle()
            result = self.brain.ti_forward_chain()
        return {"result": result}

    # -- Brain config --

    def _cmd_enable_multi_network(self, _req):
        if True:  # RWLock in handle()
            self.brain.enable_multi_network()
        return {"ok": True}

    def _cmd_init_cortex_cnns(self, _req):
        if True:  # RWLock in handle()
            self.brain.init_cortex_cnns()
        return {"ok": True}

    def _cmd_enable_world_model(self, req):
        if True:  # RWLock in handle()
            self.brain.enable_world_model(req.get("enabled", True))
        return {"ok": True}

    def _cmd_enable_world_model_bridge(self, req):
        if True:  # RWLock in handle()
            self.brain.enable_world_model_bridge(req.get("enabled", True))
        return {"ok": True}

    def _cmd_enable_mixed_precision(self, req):
        if True:  # RWLock in handle()
            self.brain.enable_mixed_precision(req.get("enabled", True))
        return {"ok": True}

    def _cmd_set_training_dashboard(self, req):
        if True:  # RWLock in handle()
            self.brain.set_training_dashboard(**{k: v for k, v in req.items() if k != "cmd"})
        return {"ok": True}

    def _cmd_get_training_dashboard(self, _req):
        if True:  # RWLock in handle()
            result = self.brain.get_training_dashboard()
        return {"dashboard": result}

    def _cmd_attach_builtin_probes(self, req):
        interval = req.get("interval_ms", 1000)
        if True:  # RWLock in handle()
            count = self.brain.attach_builtin_probes(interval)
        return {"ok": True, "count": count}

    def _cmd_get_probe_metrics(self, _req):
        if True:  # RWLock in handle()
            result = self.brain.get_all_probe_metrics()
        return {"probe_metrics": result}

    def _cmd_enable_gradient_checkpointing(self, req):
        args = [req.get("enabled", True)]
        if "interval" in req:
            args.append(req["interval"])
        if True:  # RWLock in handle()
            self.brain.enable_gradient_checkpointing(*args)
        return {"ok": True}

    # -- LNN / SNN / CNN --

    def _cmd_lnn_create(self, req):
        args = req.get("args", [])
        kwargs = {k: v for k, v in req.items()
                  if k not in ("cmd", "args")}
        if True:  # RWLock in handle()
            self.brain.lnn_create(*args, **kwargs)
        return {"ok": True}

    def _cmd_lnn_get_stats(self, _req):
        if True:  # RWLock in handle()
            return {"stats": self.brain.lnn_get_stats()}

    def _cmd_snn_get_stats(self, _req):
        if True:  # RWLock in handle()
            return {"stats": self.brain.snn_get_stats()}

    def _cmd_cnn_get_stats(self, _req):
        if True:  # RWLock in handle()
            return {"stats": self.brain.cnn_get_stats()}

    # -- Plasticity / pruning --

    def _cmd_get_plasticity_stats(self, _req):
        if True:  # RWLock in handle()
            return {"stats": self.brain.get_plasticity_stats()}

    def _cmd_prune_synapses(self, req):
        if True:  # RWLock in handle()
            self.brain.prune_synapses(req.get("threshold", 0.01))
        return {"ok": True}

    # -- Curiosity --

    def _cmd_curiosity_detect_gaps(self, req):
        topic = req.get("topic") or req.get("domain", "general")
        if True:  # RWLock in handle()
            result = self.brain.curiosity_detect_gaps(topic)
        return {"result": result}

    # -- UTM EMA --

    def _cmd_utm_swap_to_ema(self, _req):
        if True:  # RWLock in handle()
            self.brain.utm_swap_to_ema()
        return {"ok": True}

    def _cmd_utm_swap_from_ema(self, _req):
        if True:  # RWLock in handle()
            self.brain.utm_swap_from_ema()
        return {"ok": True}

    # -- Language / Interactive --

    def _cmd_comprehend(self, req):
        if True:  # RWLock in handle()
            result = self.brain.comprehend(req["text"])
        return {"result": result}

    def _cmd_generate(self, req):
        if True:  # RWLock in handle()
            result = self.brain.generate(
                prompt=req.get("prompt"),
                semantic_input=req.get("semantic_input"))
        return {"result": result}

    def _cmd_produce_text(self, req):
        if True:  # RWLock in handle()
            result = self.brain.produce_text(req["intent"])
        return {"result": result}

    def _cmd_deliberate(self, req):
        if True:  # RWLock in handle()
            result = self.brain.deliberate(req["topic"])
        return {"result": result}

    def _cmd_self_assess(self, req):
        if True:  # RWLock in handle()
            result = self.brain.self_assess(req["domain"])
        return {"result": result}

    def _cmd_rubric(self, _req):
        if True:  # RWLock in handle()
            result = self.brain.rubric()
        return {"result": result}

    def _cmd_get_last_gradient_norm(self, _req):
        if True:  # RWLock in handle()
            result = self.brain.get_last_gradient_norm()
        return {"result": result}

    def _cmd_focus_attention(self, req):
        if True:  # RWLock in handle()
            self.brain.experience_attend(req.get("modality", "visual"),
                                          req.get("strength", 1.0))
        return {"ok": True}

    # -- Phi-3 Language Cortex --

    def _cmd_phi3_generate(self, req):
        text = req.get("text", "")
        max_tokens = req.get("max_tokens", 256)
        if not text:
            return {"error": "No text provided"}
        if not hasattr(self, '_hybrid_decoder'):
            try:
                from phi3_decoder import Phi3Decoder
                from hybrid_decoder import HybridDecoder
                self._phi3 = Phi3Decoder()
                self._hybrid_decoder = HybridDecoder(
                    phi3_decoder=self._phi3, brain=self.brain)
            except Exception as e:
                return {"error": f"Phi-3 init failed: {e}"}
        if True:  # RWLock in handle()
            result = self._hybrid_decoder.respond(text, brain=self.brain)
        return {"ok": True, **result}

    # -- Identity --

    def _cmd_get_identity(self, _req):
        if not hasattr(self, '_identity'):
            try:
                from athena_identity import IdentityController
                self._identity = IdentityController(brain=self.brain)
            except Exception as e:
                return {"error": f"Identity init failed: {e}"}
        return {"ok": True, **self._identity.get_identity_summary()}

    # -- TTS --

    def _cmd_tts_speak(self, req):
        text = req.get("text", "")
        accent = req.get("accent", None)
        output_path = req.get("output_path", None)
        if not text:
            return {"error": "No text provided"}
        if not hasattr(self, '_tts'):
            try:
                from athena_tts import AthenaTTS
                self._tts = AthenaTTS()
            except Exception as e:
                return {"error": f"TTS init failed: {e}"}
        result = self._tts.speak(text, brain=self.brain,
                                accent=accent, output_path=output_path)
        if result:
            return {"ok": True, "duration": result.get("duration", 0),
                    "prosody": result.get("prosody"), "accent": result.get("accent")}
        return {"error": "TTS synthesis failed"}

    def _cmd_tts_register_accent(self, req):
        name = req.get("name", "")
        audio_path = req.get("audio_path", "")
        if not name or not audio_path:
            return {"error": "name and audio_path required"}
        if not hasattr(self, '_tts'):
            return {"error": "TTS not initialized — call tts_speak first"}
        ok = self._tts.register_accent(name, audio_path)
        return {"ok": ok}

    def _cmd_tts_list_accents(self, _req):
        if hasattr(self, '_tts'):
            return {"accents": self._tts.accent_library.list_accents()}
        return {"accents": {"loaded": [], "available": [], "descriptions": {}}}

    # -- Keepalive --

    def _cmd_keepalive(self, _req):
        return {"ok": True}

    # -- Repair NaN weights --

    def _cmd_repair_nan_weights(self, _req):
        """Zero out any NaN/Inf weights in the adaptive network."""
        import math
        try:
            count = self.brain.get_neuron_count()
            fixed = 0
            for i in range(min(count, 100000)):  # Check first 100K neurons
                bias = self.brain.get_neuron_bias(i)
                if bias is not None and (math.isnan(bias) or math.isinf(bias)):
                    self.brain.set_neuron_bias(i, 0.0)
                    fixed += 1
            logger.warning("Repaired %d NaN/Inf biases in %d neurons", fixed, min(count, 100000))
            return {"ok": True, "fixed": fixed, "checked": min(count, 100000)}
        except Exception as e:
            logger.error("NaN repair failed: %s", e)
            return {"ok": False, "error": str(e)}

    # -- Cognitive & Safety Test Battery handlers --

    def _cmd_get_mental_health_report(self, _req):
        if hasattr(self.brain, 'get_mental_health_report'):
            r = self.brain.get_mental_health_report()
            return {"report": r if r else {}}
        return {"report": {}, "unavailable": True}

    def _cmd_get_mental_health_check(self, req):
        disorder = req.get('disorder', 'Depression')
        if hasattr(self.brain, 'get_mental_health_check'):
            try:
                return {"score": self.brain.get_mental_health_check(disorder)}
            except Exception as e:
                return {"score": 0.0, "error": str(e)}
        return {"score": 0.0, "unavailable": True}

    def _cmd_get_emotion_state(self, _req):
        if hasattr(self.brain, 'get_emotion_state'):
            s = self.brain.get_emotion_state()
            return {"emotion": s if s else {}}
        return {"emotion": {}, "unavailable": True}

    def _cmd_get_internal_state(self, req):
        strategy = int(req.get('strategy', 1))
        if hasattr(self.brain, 'get_internal_state'):
            try:
                s = self.brain.get_internal_state(strategy=strategy)
                return {"state": s if s else {}}
            except Exception as e:
                return {"state": {}, "error": str(e)}
        return {"state": {}, "unavailable": True}

    def _cmd_predict_with_confidence(self, req):
        features = req.get('features', [])
        if hasattr(self.brain, 'predict_with_confidence'):
            try:
                r = self.brain.predict_with_confidence(features)
                # Layer A confabulation gate (2026-04-26).
                r = _apply_idk_gate(r if r else {}, self.brain, self._stats)
                return {"result": r if r else {}}
            except Exception as e:
                return {"result": {}, "error": str(e)}
        # Fallback: use basic predict
        try:
            label, conf = self.brain.predict(features)
            r = {"label": label, "confidence": conf}
            # Layer A confabulation gate (2026-04-26).
            r = _apply_idk_gate(r, self.brain, self._stats)
            return {"result": r}
        except Exception as e:
            return {"result": {}, "error": str(e)}

    def _cmd_predict_with_deadline(self, req):
        features = req.get('features', [])
        deadline_ms = float(req.get('deadline_ms', 100.0))
        if hasattr(self.brain, 'predict_with_deadline'):
            try:
                r = self.brain.predict_with_deadline(features, deadline_ms)
                # Layer A confabulation gate (2026-04-26).
                r = _apply_idk_gate(r if r else {}, self.brain, self._stats)
                return {"result": r if r else {}}
            except Exception as e:
                return {"result": {}, "error": str(e)}
        return {"result": {}, "unavailable": True}

    def _cmd_perturb_weights(self, req):
        magnitude = float(req.get('magnitude', 0.01))
        target = req.get('target', 'global')
        tag = req.get('tag', 'mark_test')
        if hasattr(self.brain, 'perturb_weights'):
            try:
                r = self.brain.perturb_weights(
                    magnitude=magnitude, target=target, tag=tag)
                logger.info("Weight perturbation applied: mag=%.4f target=%s tag=%s",
                            magnitude, target, tag)
                return {"result": r if r else {}}
            except Exception as e:
                return {"result": {}, "error": str(e)}
        return {"result": {}, "unavailable": True}

    def _cmd_enter_idle_with_telemetry(self, req):
        duration_ms = int(req.get('duration_ms', 2000))
        if hasattr(self.brain, 'enter_idle_with_telemetry'):
            r = self.brain.enter_idle_with_telemetry(duration_ms)
            return {"result": r if r else {}}
        return {"result": {}, "unavailable": True}

    def _cmd_get_inner_speech_trace(self, req):
        n = int(req.get('n', 10))
        if hasattr(self.brain, 'get_inner_speech_trace'):
            t = self.brain.get_inner_speech_trace(n)
            return {"trace": t if t else []}
        return {"trace": [], "unavailable": True}

    def _cmd_get_hypothesis_log(self, req):
        n = int(req.get('n', 10))
        if hasattr(self.brain, 'get_hypothesis_log'):
            h = self.brain.get_hypothesis_log(n)
            return {"log": h if h else []}
        return {"log": [], "unavailable": True}

    def _cmd_cow_trial_snapshot(self, _req):
        if hasattr(self.brain, 'cow_trial_snapshot'):
            try:
                snap = self.brain.cow_trial_snapshot()
                # Can't serialize a PyCapsule, so store it and return a handle
                if not hasattr(self, '_cow_snapshots'):
                    self._cow_snapshots = {}
                import uuid
                handle = uuid.uuid4().hex[:12]
                self._cow_snapshots[handle] = snap
                return {"handle": handle}
            except Exception as e:
                return {"handle": None, "error": str(e)}
        return {"handle": None, "unavailable": True}

    def _cmd_audit_log_event(self, req):
        """Append a test-battery audit event. Best-effort — never raises."""
        event_type = int(req.get('event_type', 28))  # default: TEST_BATTERY_RUN
        severity = int(req.get('severity', 0))
        description = str(req.get('description', ''))[:250]
        try:
            import logging
            logger.info("[AUDIT evt=%d sev=%d] %s", event_type, severity, description)
            # Append to simple JSONL log that the C audit log can later ingest
            import os, json, time
            log_dir = "/var/log/athena"
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "battery_audit.jsonl"), "a") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "event_type": event_type,
                    "severity": severity,
                    "description": description,
                }) + "\n")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _cmd_cow_trial_restore(self, req):
        handle = req.get('handle') or req.get('snapshot')
        if not handle or not hasattr(self, '_cow_snapshots'):
            return {"ok": False, "error": "no snapshot"}
        snap = self._cow_snapshots.pop(handle, None)
        if snap is None:
            return {"ok": False, "error": "handle not found"}
        try:
            ok = self.brain.cow_trial_restore(snap)
            return {"ok": bool(ok)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # -- Shutdown --

    def _cmd_shutdown(self, _req):
        logger.info("Shutdown requested via IPC")
        # Signal the main loop to exit
        os.kill(os.getpid(), signal.SIGTERM)
        return {"ok": True, "message": "Shutting down"}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class BrainDaemon:
    """Unix socket server that accepts IPC connections.

    Listens on TWO sockets:
    - Main socket (brain.sock): all commands, uses locks for write commands
    - Read-only socket (brain_ro.sock): only read commands, NO locks
      Use this for eval, monitoring, probes — never blocked by training.
    """

    # Commands allowed on the read-only socket (no lock needed)
    _READONLY_COMMANDS = frozenset({
        'ping', 'status', 'keepalive',
        'decide_full', 'predict', 'speak', 'generate_text',
        'get_neuron_count', 'get_snn_stats', 'get_network_metrics',
        'get_cortex_cnn_metrics', 'get_cognitive_stats',
        'get_module_activity',
        'get_training_dashboard', 'get_probe_metrics',
        'get_lateralization', 'get_cloud_stats',
        'utm_forward_only', 'utm_get_training_health',
        'attach_builtin_probes',
        'get_training_flags',
        'get_plateau_detector_params',
        # Test battery read-only probes
        'get_mental_health_report', 'get_mental_health_check',
        'get_emotion_state', 'get_internal_state',
        'predict_with_confidence', 'predict_with_deadline',
        'get_inner_speech_trace', 'get_hypothesis_log',
        'audit_log_event',  # audit logging is append-only, safe on RO socket
    })

    def __init__(self, service, socket_path=SOCKET_PATH, max_workers=4):
        self.service = service
        self.socket_path = socket_path
        self.ro_socket_path = socket_path.replace('.sock', '_ro.sock')
        self.lock_path = socket_path + ".lock"
        self.max_workers = max_workers
        self._server_sock = None
        self._ro_server_sock = None
        self._lock_fd = None
        self._running = False
        # Worker pool reuses threads instead of spawning one per connection
        # (which leaked 8 MB stack mmaps the glibc cache couldn't absorb,
        #  growing VmData ~30 MB/min and OOM-killing the brain every 4 h).
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="brainsvc")
        # The RO socket gets its own pool. The trainer can saturate the main
        # pool with submit_sensory/submit_multimodal writes; if the RO socket
        # shared that pool, eval/monitoring requests would queue behind the
        # writes — defeating the whole point of the RO socket (line 1894:
        # "never blocked by training"). Sized smaller than the main pool —
        # readers are short and fast, mainly stats/predict/probe paths.
        self._ro_pool = ThreadPoolExecutor(max_workers=max(2, max_workers // 2),
                                           thread_name_prefix="brainsvc-ro")

    def _acquire_socket_lock(self):
        """Acquire exclusive flock on the socket lock file.

        Prevents other processes from stealing the socket path.
        The lock is held for the lifetime of the daemon process and
        automatically released by the kernel on process exit.
        """
        import fcntl
        self._lock_fd = open(self.lock_path, "w")
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            self._lock_fd.close()
            self._lock_fd = None
            raise RuntimeError(
                f"Another process holds the socket lock {self.lock_path}. "
                "Is another brain daemon running?")
        self._lock_fd.write(str(os.getpid()))
        self._lock_fd.flush()
        logger.info("Socket lock acquired: %s", self.lock_path)

    def _release_socket_lock(self):
        """Release the socket lock file."""
        import fcntl
        if self._lock_fd:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
            except Exception:
                pass
            self._lock_fd = None
        try:
            os.unlink(self.lock_path)
        except Exception:
            pass

    def start(self):
        """Start the daemon server (main + read-only sockets)."""
        self._acquire_socket_lock()

        # Main socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        os.chmod(self.socket_path, 0o660)
        self._server_sock.listen(16)
        self._server_sock.settimeout(1.0)

        # Read-only socket for eval/monitoring — non-fatal if fails
        self._ro_server_sock = None
        try:
            if os.path.exists(self.ro_socket_path):
                os.unlink(self.ro_socket_path)
            self._ro_server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._ro_server_sock.bind(self.ro_socket_path)
            os.chmod(self.ro_socket_path, 0o660)
            self._ro_server_sock.listen(16)
            self._ro_server_sock.settimeout(1.0)
            logger.info("Read-only socket: %s", self.ro_socket_path)
        except Exception as e:
            logger.warning("Read-only socket failed: %s (eval will use main socket)", e)
            self._ro_server_sock = None

        self._running = True
        logger.info("Brain daemon listening on %s (max_workers=%d)",
                     self.socket_path, self.max_workers)

    def serve_forever(self):
        """Main accept loop — select on main + read-only sockets."""
        import select
        listen_socks = [self._server_sock]
        if self._ro_server_sock:
            listen_socks.append(self._ro_server_sock)

        while self._running:
            try:
                readable, _, _ = select.select(listen_socks, [], [], 1.0)
            except (OSError, ValueError):
                if self._running:
                    logger.error("Socket select error")
                break

            for sock in readable:
                try:
                    conn, _ = sock.accept()
                except (socket.timeout, OSError):
                    continue

                is_readonly = (self._ro_server_sock and sock is self._ro_server_sock)
                # Route to the RO pool when this came in on the read-only
                # listener so eval/monitoring requests aren't queued behind
                # trainer writes. Both pools bound concurrency and reuse
                # threads (back-pressure via the pool's internal queue).
                target_pool = self._ro_pool if is_readonly else self._pool
                target_pool.submit(self._handle_conn, conn, readonly=is_readonly)

    # Heartbeat constants
    HEARTBEAT_WARN_SECONDS = 60
    HEARTBEAT_DEAD_SECONDS = 300

    def _handle_conn(self, conn, readonly=False):
        """Handle a single client connection (may have multiple requests).

        If readonly=True (from read-only socket), commands bypass locks and
        only read-safe commands are allowed. This prevents eval/monitoring
        from being blocked by training write locks.

        Tracks last_message_time for heartbeat detection:
        - 60s without a message: log a warning (client may be stalled)
        - 300s without a message: consider client dead, close connection
        """
        client_id = id(conn)
        last_message_time = time.monotonic()
        warned = False
        try:
            conn.settimeout(self.HEARTBEAT_WARN_SECONDS)
            while self._running:
                try:
                    req = recv_msg(conn)
                except socket.timeout:
                    # No message received within timeout window
                    idle = time.monotonic() - last_message_time
                    if idle >= self.HEARTBEAT_DEAD_SECONDS:
                        logger.warning(
                            "Client %d: no message for %.0fs — "
                            "considering dead, closing connection", client_id, idle)
                        break
                    if not warned and idle >= self.HEARTBEAT_WARN_SECONDS:
                        logger.warning(
                            "Client %d: no message for %.0fs — "
                            "possible stall", client_id, idle)
                        warned = True
                    continue

                if req is None:
                    break

                last_message_time = time.monotonic()
                warned = False

                if readonly:
                    # Read-only socket: only allow safe commands, bypass all locks
                    cmd = req.get("cmd", "")
                    if cmd not in self._READONLY_COMMANDS:
                        resp = {"error": f"Command '{cmd}' not allowed on read-only socket"}
                    else:
                        resp = self.service.handle_readonly(req)
                else:
                    resp = self.service.handle(req)

                send_msg(conn, resp)

                # Single-shot commands close after response
                # Keep-alive: client sends {"cmd": "keepalive"} to stay connected
                if req.get("cmd") != "keepalive":
                    break
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            logger.warning("Connection handler error: %s", e)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            # Pool reclaims the worker automatically; no semaphore to release.

    def rebind(self):
        """Re-create the listening socket (SIGHUP handler)."""
        logger.info("Rebinding socket on %s", self.socket_path)
        # Close old socket
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        # Remove stale socket file
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except Exception:
                pass
        # Create new socket
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        os.chmod(self.socket_path, 0o660)
        self._server_sock.listen(16)
        self._server_sock.settimeout(1.0)
        logger.info("Socket rebound successfully on %s", self.socket_path)

    def stop(self):
        """Stop the daemon."""
        self._running = False
        for sock in (self._server_sock, self._ro_server_sock):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        for path in (self.socket_path, self.ro_socket_path):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        self._release_socket_lock()
        logger.info("Brain daemon stopped")


# ---------------------------------------------------------------------------
# Auto-checkpoint thread
# ---------------------------------------------------------------------------

class AutoCheckpointer:
    """Periodically saves the brain to disk with safety guards.

    Guards against overwriting a trained checkpoint with a fresh brain:
    - Won't save until at least `min_steps_before_save` training steps have occurred
    - Keeps the previous checkpoint as .bak before overwriting
    - Supports both time-based and step-based save triggers
    """

    def __init__(self, brain, checkpoint_dir, interval_seconds=300,
                 min_steps_before_save=10):
        self.brain = brain
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval_seconds
        self.min_steps = min_steps_before_save
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._save_count = 0
        self._last_athena_auto_time = 0.0
        # Snapshot of _save_count at the last athena_auto write. The
        # athena_auto cadence gate requires _save_count to have INCREASED
        # since this snapshot — so the daemon's own 5-min save loop
        # cannot keep emitting athena_auto files when the trainer is dead
        # and notify_training_step has stopped firing. Preserves the
        # local monitor's trainer-aliveness tripwire.
        self._save_count_at_last_athena_auto = 0
        self._loaded_from_checkpoint = False
        # Tracks the size of the most recent successful save from THIS
        # session, so the shrink check compares like-for-like. Prior to
        # this, the shrink check compared against whatever stale
        # athena_immersive.bin happened to be on disk — so after a
        # struct-layout change (e.g. G8 lightweight CSR), the new daemon's
        # first save could be legitimately smaller than the pre-change
        # file and get rejected forever. See 2026-04-23 postmortem.
        self._prev_session_save_size = 0

    def set_loaded_from_checkpoint(self, loaded):
        """Mark whether this brain was loaded from a checkpoint.
        If loaded, allow immediate saves (the brain already has trained state).
        If fresh, require min_steps training before first save.

        Bug 2026-04-26: previously set _save_count = 1 here, but the gate
        in save_now is `if _save_count < min_steps`. With min_steps=10,
        a loaded brain stayed blocked at 1 forever — the comment lied.
        notify_training_step only fires on learn_vector RPC; the live
        trainer uses submit_multimodal exclusively, so the counter never
        progressed. Result: 1h36m without any auto-checkpoint. Fix: set
        to min_steps so a loaded brain truly bypasses the gate.
        """
        self._loaded_from_checkpoint = loaded
        if loaded:
            self._save_count = self.min_steps  # truly allow saves immediately

    def start(self):
        # interval <= 0 disables the auto-checkpointer entirely.
        # Use this when an external system (training script) already handles
        # checkpointing — prevents duplicate disk usage from two save systems.
        if self.interval <= 0:
            logger.info("Auto-checkpoint DISABLED (interval=%ds)", self.interval)
            self._running = False
            return

        # Note: previous CB safety lock (commit 7e69a2a5f) disabled
        # auto-save when conductance_enabled=1 to prevent the
        # double-rescale-on-restart bomb. That lock is now unnecessary
        # — save_now writes a cb_rescaled_marker sidecar after every
        # successful save, and _load_persistent_snn_tunes consults the
        # marker before deciding whether to force-rescale. Save+load
        # cycles are now idempotent.
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="auto-checkpoint")
        self._thread.start()
        logger.info("Auto-checkpoint every %ds to %s (min_steps=%d, loaded=%s)",
                     self.interval, self.checkpoint_dir,
                     self.min_steps, self._loaded_from_checkpoint)

    def stop(self):
        self._running = False

    def notify_training_step(self):
        """Called by BrainService after each learn_vector to track progress.
        Enables step-based checkpoint gating."""
        self._save_count += 1

    def save_now(self, force=False):
        """Save checkpoint with safety guards.

        CRITICAL SAFETY: Never overwrite trained data with untrained brain.
        Uses temp file + size validation + atomic rename.
        Checks disk space before writing.
        """
        import shutil, glob

        # GUARD 1: Never save a fresh/untrained brain
        if self._save_count < self.min_steps and not force:
            # WARN (not DEBUG) so that a misconfigured gate never silently
            # hides a stuck auto-checkpointer for hours like the 2026-04-26
            # incident. If you see this on a loaded brain, the bug is in
            # set_loaded_from_checkpoint or notify_training_step coverage.
            logger.warning("Checkpoint blocked by min_steps gate: "
                           "save_count=%d need=%d (loaded_from_checkpoint=%s)",
                           self._save_count, self.min_steps,
                           self._loaded_from_checkpoint)
            return

        # Use ABSOLUTE path — resolve symlinks to avoid path confusion
        canonical = os.path.join(self.checkpoint_dir, "athena_immersive.bin")
        if os.path.islink(canonical):
            canonical = os.path.realpath(canonical)

        tmp_path = canonical + ".tmp"
        existing_size = os.path.getsize(canonical) if os.path.exists(canonical) else 0

        # Helper: clean up orphan shards left by a failed save attempt.
        # The C-side save writes shards in order (.snn first ~9-14GB, then
        # adaptive into the .bin, then sidecars). A mid-write quota failure
        # leaves an 8-9GB .snn (or other sidecar) under tmp_path.* — these
        # MUST be removed or every retry compounds the quota deficit.
        def _cleanup_save_partials(prefix):
            partials = glob.glob(prefix) + glob.glob(prefix + '.*')
            for p in partials:
                try:
                    os.remove(p)
                except OSError:
                    pass

        # Helper: directory-size based quota-aware free check.
        # os.statvfs reports the underlying MFS filesystem (multi-PB), NOT
        # the pod's per-pod quota (~80GB). MFS errno=122 quota errors fire
        # despite os.statvfs claiming TB free. We trust an explicit quota
        # cap (NIMCP_CHECKPOINT_QUOTA_GB env, default 75GB conservative)
        # and measure against actual checkpoint-dir size + fixed reserve
        # for the inbound 14GB save.
        def _checkpoint_free_gb():
            try:
                quota_gb = float(os.environ.get('NIMCP_CHECKPOINT_QUOTA_GB', '75'))
            except ValueError:
                quota_gb = 75.0
            total = 0
            for root, _, files in os.walk(self.checkpoint_dir):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            used_gb = total / (1024**3)
            # Cross-check against statvfs and take the more pessimistic
            try:
                st = os.statvfs(self.checkpoint_dir)
                fs_free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
            except OSError:
                fs_free_gb = 1e9
            return min(quota_gb - used_gb, fs_free_gb)

        try:
            # STEP 0: Quota-aware disk check — need at least 20 GB free
            # (one 14GB save + 6GB headroom for sidecars + retry margin).
            min_free_gb = 20.0
            free_gb = _checkpoint_free_gb()
            if free_gb < min_free_gb:
                logger.warning("Low quota-free space (%.1f GB free of NIMCP_CHECKPOINT_QUOTA_GB=%s) "
                               "— pruning old checkpoints", free_gb,
                               os.environ.get('NIMCP_CHECKPOINT_QUOTA_GB', '75'))
                auto_files = sorted(glob.glob(
                    os.path.join(self.checkpoint_dir, "athena_auto_*.bin")))
                auto_files = [f for f in auto_files if not any(
                    f.endswith('.bin.' + ext) for ext in
                    ['snn','lnn','cnn','meta','tokenizer','mirror_neurons',
                     'executive','cortex_visual','cortex_audio',
                     'cortex_speech','cortex_somato','cb_rescaled','temperature'])]
                while auto_files and free_gb < min_free_gb:
                    old = auto_files.pop(0)
                    try:
                        os.remove(old)
                        for sc in glob.glob(old + '.*'):
                            os.remove(sc)
                        logger.info("Pruned old checkpoint: %s", old)
                    except Exception:
                        pass
                    free_gb = _checkpoint_free_gb()
                # Also drop any orphan .tmp shards from prior failed saves
                for orphan in glob.glob(os.path.join(self.checkpoint_dir, "*.tmp")) + \
                              glob.glob(os.path.join(self.checkpoint_dir, "*.bin.tmp.*")):
                    try:
                        os.remove(orphan)
                        logger.info("Removed orphan shard: %s", orphan)
                    except Exception:
                        pass
                free_gb = _checkpoint_free_gb()
                if free_gb < min_free_gb:
                    logger.error("Still only %.1f GB free after pruning — skipping save", free_gb)
                    return

            # STEP 1: Save to temp file (never touch the real checkpoint).
            # brain.save() automatically writes sidecars (.snn, .lnn, .cnn,
            # .meta, .cortex_*, etc.) to tmp_path.snn, tmp_path.lnn, etc.
            # On any failure mid-write, clean up the partials so the next
            # retry has a clean slate.
            try:
                self.brain.save(tmp_path)
            except Exception as save_exc:
                _cleanup_save_partials(tmp_path)
                logger.error("brain.save FAILED — orphan shards cleaned: %s", save_exc)
                raise

            # STEP 2: Validate — temp file must be reasonably sized.
            # Threshold scales with neuron count: ~50 bytes/neuron minimum
            # (covers sparse synapse + metadata overhead). For 150K ANN this
            # is ~7.5 MB; for 2M ANN it was ~100 MB. Floor at 5 MB.
            tmp_size = os.path.getsize(tmp_path)
            try:
                n_count = self.brain.get_neuron_count()
            except Exception:
                n_count = DEFAULT_ANN_NEURONS
            min_size = max(CHECKPOINT_MIN_SIZE, n_count * CHECKPOINT_MIN_BYTES_PER_NEURON)
            if tmp_size < min_size:
                logger.error("Checkpoint too small (%d bytes, min %d for %d neurons) "
                             "— likely fresh brain, REFUSING to overwrite trained "
                             "checkpoint (%d bytes)",
                             tmp_size, min_size, n_count, existing_size)
                os.remove(tmp_path)
                return

            # STEP 3: Shrink check — within-session only.
            # Compare against the previous save from THIS daemon session
            # (not the stale on-disk file from an earlier session/build).
            # This catches true mid-session corruption (brain state
            # suddenly smaller after tens of learn_vector calls) while
            # allowing legitimate size reductions when (a) the daemon
            # cold-starts against a pre-existing file from a different
            # build, or (b) the brain's struct layout has changed between
            # sessions.
            #
            # Rationale: prior code compared `tmp_size` to `existing_size`
            # (the on-disk file). After any struct-layout change — e.g.
            # the G8 lightweight-CSR migration shrinking synapse storage —
            # the new build's legitimate save was always smaller than
            # the legacy file on disk, and the shrink check rejected
            # EVERY save for the full session. 5 hours of G8 training
            # silently discarded on 2026-04-23.
            if self._prev_session_save_size > 0 and \
               tmp_size < self._prev_session_save_size * 0.8:
                logger.error("Checkpoint SHRANK mid-session from %d to %d bytes "
                             "(%.0f%%) — REFUSING to overwrite (possible corruption)",
                             self._prev_session_save_size, tmp_size,
                             tmp_size / self._prev_session_save_size * 100)
                os.remove(tmp_path)
                return
            # First save of a session: if there's an on-disk file from a
            # previous session and our save is catastrophically smaller
            # (less than 5% of existing — a clear sign we wrote essentially
            # nothing, not a structural delta), still refuse. This allows
            # normal size variance (e.g. 80% shrink from a struct change)
            # without allowing near-empty writes.
            elif existing_size > 0 and tmp_size < existing_size * 0.05:
                logger.error("Checkpoint essentially empty (%d bytes vs %d prior "
                             "session) — REFUSING to overwrite",
                             tmp_size, existing_size)
                os.remove(tmp_path)
                return

            # STEP 4: Atomic rename core file + move all sidecars.
            # Also record the size of this successful save so the next
            # shrink check compares within-session (see STEP 3 rationale).
            self._prev_session_save_size = tmp_size
            os.replace(tmp_path, canonical)
            # Move sidecars: tmp_path.snn -> canonical.snn, etc.
            import glob as _glob
            for sidecar in _glob.glob(tmp_path + '.*'):
                ext = sidecar[len(tmp_path):]  # e.g. ".snn", ".lnn"
                try:
                    os.replace(sidecar, canonical + ext)
                except Exception:
                    pass

            # STEP 5: Timestamped backup with two gates:
            #   (a) at least 25 min of monotonic elapsed time since last
            #       backup (cadence predictability — Fix replacing the old
            #       irregular `_save_count % 5 == 0` gate).
            #   (b) _save_count must have increased since the last backup
            #       (trainer-aliveness — the daemon's own 5-min save loop
            #       runs even when the trainer is dead; without (b) this
            #       loop would emit fresh athena_auto files indefinitely
            #       and the local monitor's trainer-dead tripwire would
            #       never fire).
            # Wall clock would be sufficient here (the gate is robust to
            # NTP forward jumps because larger _now just fires the gate
            # earlier — never blocks). But monotonic is robust to
            # backwards corrections that would freeze the gate, so prefer
            # it. Initialized to 0.0 so the first save always emits a
            # backup, which is the right behaviour for fresh startups.
            import time as _time
            _now = _time.monotonic()
            _athena_auto_interval_s = 25 * 60
            ts_path = None
            if (_now - self._last_athena_auto_time) >= _athena_auto_interval_s \
               and self._save_count > self._save_count_at_last_athena_auto:
                self._last_athena_auto_time = _now
                self._save_count_at_last_athena_auto = self._save_count
                ts = _time.strftime("%Y%m%d_%H%M%S")
                ts_path = os.path.join(self.checkpoint_dir, f"athena_auto_{ts}.bin")
                try:
                    shutil.copy2(canonical, ts_path)
                    # Also copy sidecars
                    for sc in _glob.glob(canonical + '.*'):
                        ext = sc[len(canonical):]
                        shutil.copy2(sc, ts_path + ext)
                except Exception as e:
                    logger.warning("Timestamped backup failed: %s", e)

                # Keep last 2 timestamped backups (was 5 — too greedy on
                # the pod's ~80GB quota, where each is ~14GB so 5 = 70GB
                # just for backups). 2 covers "current + previous" for
                # rollback safety without hogging quota.
                auto_files = sorted(_glob.glob(
                    os.path.join(self.checkpoint_dir, "athena_auto_*.bin")))
                # Filter to only core files (not sidecars like .bin.snn)
                auto_files = [f for f in auto_files if not any(
                    f.endswith('.bin.' + ext) for ext in
                    ['snn','lnn','cnn','meta','tokenizer','mirror_neurons',
                     'executive','cortex_visual','cortex_audio',
                     'cortex_speech','cortex_somato'])]
                while len(auto_files) > 2:
                    old = auto_files.pop(0)
                    try:
                        os.remove(old)
                        # Also remove sidecars
                        for sc in _glob.glob(old + '.*'):
                            os.remove(sc)
                    except Exception:
                        pass

            logger.info("Checkpoint saved: %s (%d bytes, steps=%d)",
                        canonical, os.path.getsize(canonical), self._save_count)

            # CB rescale marker: if conductance_enabled is on, the
            # weights we just wrote ARE rescaled (in-memory state was
            # rescaled at load time). Pin a marker to this file so a
            # subsequent restart's force-rescale-on-load skips re-applying
            # the factor. Pin to BOTH the canonical file AND the
            # timestamped backup we may have just made — same in-memory
            # state was written to both. See cb_rescaled_marker.py.
            try:
                cb_on = float(self.brain.snn_tune_get().get(
                    'conductance_enabled', 0.0)) != 0.0
                if cb_on:
                    cb_rescaled_marker.write_marker(canonical, CB_DEFAULT_RESCALE_FACTOR)
                    # Mirror the marker onto the timestamped backup if one
                    # was written this iteration (ts_path stays None when
                    # the time-based gate didn't fire).
                    if ts_path is not None:
                        cb_rescaled_marker.write_marker(ts_path, CB_DEFAULT_RESCALE_FACTOR)
            except Exception as _cbm_e:
                logger.warning("CB marker write failed (save still good): %s", _cbm_e)

            # The trainer (immerse_athena._save_checkpoint_sync) is the sole
            # writer of immersive_state.json — it knows the per-stage step
            # value the curriculum cares about. The daemon used to overwrite
            # state["step"] here with a hard-coded interpolation off
            # total_learning_steps (base_total=98374, base_step=10600), which
            # produced numbers like 47986 regardless of the real curriculum
            # progress and clobbered the trainer's correct value (e.g. 150).
            # Removed 2026-05-05.

            # STEP 6: Sync to Hetzner with CRC32 verification
            self._sync_to_hetzner(canonical)

        except Exception as e:
            logger.error("Checkpoint FAILED: %s", e)
            # Clean up temp file if it exists
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _sync_to_hetzner(self, checkpoint_path):
        """Hetzner sync is handled by the external supervisord checkpoint-sync
        service (rotating snapshots, gzip-at-rest). The old per-save scp
        here blocked the save thread for up to 600s on each attempt,
        which starved brain RPCs and caused training 'ping' failures.
        Kept as a no-op for API compatibility with the old call sites.
        """
        return
        # Dead code below — kept for reference; use checkpoint-sync service instead.
        import hashlib
        import subprocess

        HETZNER_HOST = "bbrelin@176.9.99.103"
        HETZNER_DIR = "/home/bbrelin/nimcp/checkpoints/athena"

        try:
            # Compute local CRC32 (use md5 for speed on 8GB files)
            local_hash = hashlib.md5()
            with open(checkpoint_path, 'rb') as f:
                while True:
                    chunk = f.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    local_hash.update(chunk)
            local_md5 = local_hash.hexdigest()
            local_size = os.path.getsize(checkpoint_path)

            # Collect files to sync: checkpoint + sidecars
            import glob as _glob
            files_to_sync = [checkpoint_path]
            files_to_sync.extend(_glob.glob(checkpoint_path + '.*'))

            # scp all files to Hetzner
            scp_args = ['scp', '-o', 'StrictHostKeyChecking=no',
                        '-o', 'ConnectTimeout=30',
                        '-o', 'Compression=yes']
            scp_args.extend(files_to_sync)
            scp_args.append(f"{HETZNER_HOST}:{HETZNER_DIR}/")

            logger.info("Syncing %d files to Hetzner (%s)...",
                        len(files_to_sync), os.path.basename(checkpoint_path))
            result = subprocess.run(scp_args, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.warning("Hetzner sync failed: %s", result.stderr.strip())
                return

            # Verify: compute MD5 on Hetzner and compare
            remote_basename = os.path.basename(checkpoint_path)
            remote_path = f"{HETZNER_DIR}/{remote_basename}"
            verify_cmd = ['ssh', '-o', 'StrictHostKeyChecking=no',
                          '-o', 'ConnectTimeout=10',
                          HETZNER_HOST,
                          f'md5sum {remote_path} && stat -c%s {remote_path}']
            verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=120)

            if verify.returncode == 0:
                lines = verify.stdout.strip().split('\n')
                remote_md5 = lines[0].split()[0] if lines else ""
                remote_size = int(lines[1]) if len(lines) > 1 else 0

                if remote_md5 == local_md5 and remote_size == local_size:
                    logger.info("Hetzner sync VERIFIED: %s (md5=%s, %d bytes)",
                                remote_basename, local_md5[:12], local_size)
                else:
                    logger.error("Hetzner sync CHECKSUM MISMATCH: "
                                 "local=%s/%d remote=%s/%d",
                                 local_md5[:12], local_size,
                                 remote_md5[:12], remote_size)
            else:
                logger.warning("Hetzner verify failed: %s", verify.stderr.strip())

        except subprocess.TimeoutExpired:
            logger.warning("Hetzner sync timed out (600s)")
        except Exception as e:
            logger.warning("Hetzner sync error: %s", e)

    def _run(self):
        while self._running:
            time.sleep(self.interval)
            if self._running:
                self.save_now()


# ---------------------------------------------------------------------------
# SNN Recovery Plateau Detector
# ---------------------------------------------------------------------------

class SnnRecoveryPlateauDetector:
    """Background watchdog that auto-unfreezes ANN/CNN/LNN training when the
    SNN's loss has plateaued during snn-only recovery mode, then gradually
    warms the ensemble back in to avoid co-adaptation shock.

    === PHASE 1: plateau detection (while recovery mode is ON) ===

    Criteria (ALL must hold to fire the unfreeze):
      - At least `min_steps_in_recovery` SNN training steps since recovery
        was entered (floor — don't fire too early).
      - Sliding-window slope of snn_loss over last `window_size` samples
        is below `slope_threshold` in absolute value (plateau).

    Additional fire triggers (ANY):
      - Total steps in recovery >= `max_steps_in_recovery` (ceiling).
      - snn_loss drops below `absolute_loss_target` ("good enough").

    === PHASE 2: ensemble warmup (after plateau fires) ===

    Instead of immediately flipping recovery off with a sudden jolt, the
    detector:
      1. Sets ensemble_warmup_scale to `warmup_initial_scale` (e.g. 0.05)
      2. Sets snn_only_recovery = false
      3. Each subsequent poll tick, ramps warmup_scale up by
         (1.0 - initial) / warmup_ticks until it reaches 1.0
      4. Once warmup_scale == 1.0, normal joint training resumed.

    During warmup, non-SNN networks train probabilistically (Monte-Carlo
    gate in the C learning path) at a rate equal to the current scale. A
    scale of 0.3 means ANN/CNN/LNN each train ~30% of steps. Each network
    rolls independently — no synchronized oscillation.

    Bidirectional: if the user re-enables recovery mode mid-warmup,
    warmup aborts and the detector re-arms for the next plateau.
    """

    # Plateau phase states
    STATE_IDLE = 0      # Recovery mode off, detector armed but inactive
    STATE_WATCHING = 1  # Recovery on, watching for plateau
    STATE_WARMUP = 2    # Plateau fired, warmup ramp in progress

    def __init__(self, brain,
                 poll_interval_s=10.0,
                 window_size=100,
                 slope_threshold=0.05,
                 min_steps_in_recovery=500,
                 max_steps_in_recovery=5000,
                 absolute_loss_target=5.0,
                 warmup_initial_scale=0.05,
                 warmup_steps=100):
        """Construct the plateau detector. All tunables can be adjusted
        at runtime via set_params() / the daemon's set_plateau_detector_params
        RPC — no rebuild needed to iterate on thresholds.

        poll_interval_s: how often to sample brain state (seconds)
        window_size: slope is computed over the last N loss samples
        slope_threshold: |slope| below this (loss-units per sample)
                         counts as a plateau
        min_steps_in_recovery: minimum SNN training steps in recovery
                               before plateau detection can fire (floor)
        max_steps_in_recovery: hard ceiling — always fire after this
                               many steps regardless of plateau status
        absolute_loss_target: fire if snn_loss drops below this
                              ("good enough" escape)
        warmup_initial_scale: warmup_scale value at warmup start
                              (must be > 0 so non-SNN nets get some
                              gradient signal immediately)
        warmup_steps: number of SNN training steps over which the
                      warmup scale ramps from initial → 1.0. This is
                      step-based, not wall-clock — warmup advances
                      proportionally to actual training throughput so
                      slower runs get proportionally longer warmups.
        """
        self.brain = brain
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        # Tunable parameters (also settable via set_params at runtime)
        self.poll_interval_s = float(poll_interval_s)
        self.window_size = int(window_size)
        self.slope_threshold = float(slope_threshold)
        self.min_steps_in_recovery = int(min_steps_in_recovery)
        self.max_steps_in_recovery = int(max_steps_in_recovery)
        self.absolute_loss_target = float(absolute_loss_target)
        self.warmup_initial_scale = float(warmup_initial_scale)
        self.warmup_steps = int(warmup_steps)

        # Phase machine
        self._state = self.STATE_IDLE
        self._entry_snn_steps = None
        self._warmup_start_snn_steps = None
        self._loss_history = collections.deque(maxlen=self.window_size)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="snn-recovery-plateau")
        self._thread.start()
        logger.info("SNN recovery plateau detector started (%s)", self._describe_params())

    def stop(self):
        self._running = False

    def _describe_params(self):
        return ("poll=%.1fs window=%d slope<%.4f min=%d max=%d "
                "target=%.2f warmup_initial=%.2f warmup_steps=%d") % (
            self.poll_interval_s, self.window_size, self.slope_threshold,
            self.min_steps_in_recovery, self.max_steps_in_recovery,
            self.absolute_loss_target, self.warmup_initial_scale,
            self.warmup_steps)

    def get_params(self):
        """Return current parameter values as a dict."""
        with self._lock:
            return {
                "poll_interval_s": self.poll_interval_s,
                "window_size": self.window_size,
                "slope_threshold": self.slope_threshold,
                "min_steps_in_recovery": self.min_steps_in_recovery,
                "max_steps_in_recovery": self.max_steps_in_recovery,
                "absolute_loss_target": self.absolute_loss_target,
                "warmup_initial_scale": self.warmup_initial_scale,
                "warmup_steps": self.warmup_steps,
                "state": {0: "IDLE", 1: "WATCHING", 2: "WARMUP"}.get(self._state, "UNKNOWN"),
            }

    def set_params(self, **kwargs):
        """Update tunable parameters at runtime. Only the keys present in
        kwargs are modified; others keep their current value. Clamps and
        type-coerces values for safety. Returns the updated params dict."""
        with self._lock:
            if "poll_interval_s" in kwargs:
                v = float(kwargs["poll_interval_s"])
                if v >= 1.0:  # Avoid spinning
                    self.poll_interval_s = v
            if "window_size" in kwargs:
                v = int(kwargs["window_size"])
                if v >= 2:
                    self.window_size = v
                    # Rebuild the deque with the new maxlen, preserving recent samples
                    old = list(self._loss_history)[-v:]
                    self._loss_history = collections.deque(old, maxlen=v)
            if "slope_threshold" in kwargs:
                v = float(kwargs["slope_threshold"])
                if v >= 0.0:
                    self.slope_threshold = v
            if "min_steps_in_recovery" in kwargs:
                self.min_steps_in_recovery = max(0, int(kwargs["min_steps_in_recovery"]))
            if "max_steps_in_recovery" in kwargs:
                self.max_steps_in_recovery = max(1, int(kwargs["max_steps_in_recovery"]))
            if "absolute_loss_target" in kwargs:
                v = float(kwargs["absolute_loss_target"])
                if v > 0.0:
                    self.absolute_loss_target = v
            if "warmup_initial_scale" in kwargs:
                v = float(kwargs["warmup_initial_scale"])
                self.warmup_initial_scale = max(0.0, min(1.0, v))
            if "warmup_steps" in kwargs:
                self.warmup_steps = max(1, int(kwargs["warmup_steps"]))
        logger.info("Plateau detector params updated: %s", self._describe_params())
        return self.get_params()

    def _get_state(self):
        """Snapshot the values we need to make a decision this tick."""
        try:
            metrics = self.brain.get_network_metrics() or {}
            snn_loss = float(metrics.get("snn_loss", 0.0) or 0.0)
            snn_steps = int(metrics.get("snn_steps", 0) or 0)
        except Exception as e:
            logger.debug("plateau detector: get_network_metrics failed: %s", e)
            return None
        try:
            in_recovery = bool(self.brain.get_snn_only_recovery())
            warmup_scale = float(self.brain.get_ensemble_warmup_scale())
        except Exception as e:
            logger.debug("plateau detector: state query failed: %s", e)
            return None
        return snn_loss, snn_steps, in_recovery, warmup_scale

    @staticmethod
    def _linear_slope(ys):
        """Simple linear-regression slope of a 1D series against index."""
        n = len(ys)
        if n < 2:
            return 0.0
        mean_x = (n - 1) / 2.0
        mean_y = sum(ys) / n
        num = 0.0
        den = 0.0
        for i, y in enumerate(ys):
            dx = i - mean_x
            num += dx * (y - mean_y)
            den += dx * dx
        return num / den if den > 0 else 0.0

    def _enter_warmup(self, reason, snn_loss, snn_steps, slope):
        """Plateau fired — transition from STATE_WATCHING to STATE_WARMUP.
        Set warmup scale to initial value, flip recovery off, and anchor
        the step counter. From this tick on, non-SNN networks will
        probabilistically train at the warmup rate, ramping up as the
        SNN accumulates warmup_steps additional training steps."""
        try:
            self.brain.set_ensemble_warmup_scale(self.warmup_initial_scale)
            self.brain.set_snn_only_recovery(False)
        except Exception as e:
            logger.error("plateau detector: transition to warmup failed: %s", e)
            return
        logger.warning(
            "[SNN-RECOVERY AUTO-UNFREEZE] %s (snn_loss=%.3f, snn_steps=%d, slope=%.5f) — "
            "entering ensemble warmup (initial scale=%.2f, ramp over %d SNN steps)",
            reason, snn_loss, snn_steps, slope,
            self.warmup_initial_scale, self.warmup_steps)
        self._state = self.STATE_WARMUP
        self._warmup_start_snn_steps = snn_steps
        self._loss_history.clear()

    def _advance_warmup(self, snn_loss, snn_steps):
        """Advance warmup_scale based on SNN training steps elapsed since
        warmup started. Progress is strictly step-based so warmup takes
        the same amount of *training* regardless of wall-clock throughput.

        When steps_elapsed >= warmup_steps, transition to IDLE."""
        start = self._warmup_start_snn_steps
        if start is None:
            start = snn_steps
            self._warmup_start_snn_steps = start
        steps_elapsed = max(0, snn_steps - start)
        progress = min(1.0, steps_elapsed / max(1, self.warmup_steps))
        new_scale = (self.warmup_initial_scale +
                     (1.0 - self.warmup_initial_scale) * progress)
        try:
            self.brain.set_ensemble_warmup_scale(new_scale)
        except Exception as e:
            logger.error("plateau detector: warmup advance failed: %s", e)
            return
        if progress >= 1.0:
            logger.warning(
                "[SNN-RECOVERY] Warmup complete — full joint training resumed "
                "(snn_loss=%.3f, %d/%d SNN steps consumed, scale=1.0)",
                snn_loss, steps_elapsed, self.warmup_steps)
            self._state = self.STATE_IDLE
            self._entry_snn_steps = None
            self._warmup_start_snn_steps = None
        else:
            logger.info(
                "[SNN-RECOVERY] Warmup progress %.1f%% (%d/%d steps), "
                "scale=%.3f, snn_loss=%.3f",
                progress * 100.0, steps_elapsed, self.warmup_steps,
                new_scale, snn_loss)

    def _abort_warmup(self):
        """User re-enabled recovery mode mid-warmup — clean up and rearm."""
        try:
            self.brain.set_ensemble_warmup_scale(1.0)
        except Exception:
            pass
        logger.info("[SNN-RECOVERY] Warmup aborted — user re-entered recovery")
        self._state = self.STATE_WATCHING
        self._entry_snn_steps = None
        self._warmup_start_snn_steps = None
        self._loss_history.clear()

    def _run(self):
        while self._running:
            try:
                state = self._get_state()
                if state is None:
                    time.sleep(self.poll_interval_s)
                    continue
                snn_loss, snn_steps, in_recovery, _warmup = state

                with self._lock:
                    # ---- Transition logic ----
                    if self._state == self.STATE_IDLE:
                        if in_recovery:
                            self._state = self.STATE_WATCHING
                            self._entry_snn_steps = snn_steps
                            self._loss_history.clear()
                            logger.info(
                                "[SNN-RECOVERY] Armed at snn_steps=%d, snn_loss=%.3f",
                                snn_steps, snn_loss)

                    elif self._state == self.STATE_WATCHING:
                        if not in_recovery:
                            # User turned it off manually — reset without warming up
                            logger.info(
                                "[SNN-RECOVERY] Disarmed (recovery manually off)")
                            self._state = self.STATE_IDLE
                            self._entry_snn_steps = None
                            self._loss_history.clear()
                        else:
                            self._loss_history.append(snn_loss)
                            steps_in_recovery = snn_steps - (self._entry_snn_steps or 0)

                            if steps_in_recovery >= self.max_steps_in_recovery:
                                self._enter_warmup(
                                    "max-steps ceiling (%d >= %d)" % (
                                        steps_in_recovery, self.max_steps_in_recovery),
                                    snn_loss, snn_steps, 0.0)
                            elif snn_loss > 0.0 and snn_loss < self.absolute_loss_target:
                                self._enter_warmup(
                                    "loss below target (%.3f < %.3f)" % (
                                        snn_loss, self.absolute_loss_target),
                                    snn_loss, snn_steps, 0.0)
                            elif (steps_in_recovery >= self.min_steps_in_recovery and
                                  len(self._loss_history) >= self.window_size):
                                slope = self._linear_slope(list(self._loss_history))
                                if abs(slope) < self.slope_threshold:
                                    self._enter_warmup(
                                        "plateau detected (|slope|=%.5f < %.5f, "
                                        "window=%d samples)" % (
                                            abs(slope), self.slope_threshold,
                                            len(self._loss_history)),
                                        snn_loss, snn_steps, slope)

                    elif self._state == self.STATE_WARMUP:
                        if in_recovery:
                            self._abort_warmup()
                        else:
                            self._advance_warmup(snn_loss, snn_steps)
            except Exception as e:
                logger.error("plateau detector loop error: %s", e)

            time.sleep(self.poll_interval_s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Athena Brain Daemon — persistent brain server")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load brain from checkpoint file")
    parser.add_argument("--resume", action="store_true",
                        help="Auto-resume from latest checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore checkpoints, create fresh brain")
    parser.add_argument("--init-mode", type=str, default="full",
                        choices=["full", "fast", "minimal"],
                        help="Brain init mode (default: full)")
    parser.add_argument("--neuron-count", type=int, default=DEFAULT_ANN_NEURONS,
                        help=f"ANN neuron count (default: {DEFAULT_ANN_NEURONS}, SNN is primary)")
    parser.add_argument("--snn-neuron-count", type=int, default=DEFAULT_SNN_NEURONS,
                        help=f"SNN target neuron count (default: {DEFAULT_SNN_NEURONS})")
    parser.add_argument("--lnn-neuron-count", type=int, default=DEFAULT_LNN_NEURONS,
                        help=f"LNN neuron count (default: {DEFAULT_LNN_NEURONS})")
    parser.add_argument("--num-inputs", type=int, default=1024,
                        help="Input dimension (default: 1024)")
    parser.add_argument("--num-outputs", type=int, default=2048,
                        help="Output dimension (default: 2048)")
    parser.add_argument("--socket", type=str, default=SOCKET_PATH,
                        help=f"Unix socket path (default: {SOCKET_PATH})")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max concurrent worker threads (default: 4)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=None,
                        help="Auto-checkpoint directory (default: same dir as --checkpoint)")
    parser.add_argument("--checkpoint-interval", type=int, default=300,
                        help="Auto-checkpoint interval seconds (default: 300)")
    parser.add_argument("--sleep-interval-min", type=int, default=15,
                        help="Periodic sleep-wake cycle interval in minutes "
                             "(default: 15). Each interval triggers one full "
                             "drowsy→NREM→REM→awake cycle. Pressure-driven "
                             "triggers still fire between periodic cycles. "
                             "Rationale: replay batch=100 memories, learning "
                             "at ~6 steps/min → 15 min ≈ 90 new memories per "
                             "cycle, keeping consolidation paced 1:1 with "
                             "learning.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Log file path (default: stderr)")
    parser.add_argument("--bootstrap-lexicon", type=str, default=None,
                        metavar="PATH",
                        help="Optional path to base_lexicon_v*.json. If "
                             "given AND the brain has < 300 vocab words at "
                             "startup, bootstrap the grounded lexicon from "
                             "this file. Skipped on resume from a "
                             "checkpoint that already has a populated "
                             "lexicon to avoid double-bootstrapping.")
    args = parser.parse_args()

    # Logging setup
    log_handlers = [logging.StreamHandler()]
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=log_handlers,
    )

    # Stale PID cleanup disabled — supervisord/systemd manages process lifecycle.
    # The pgrep-based approach killed sibling restarts, causing FATAL loops.
    my_pid = os.getpid()
    stale_pids = set()

    if False:  # Disabled — was causing restart loops under process managers
        try:
            import subprocess
            result = subprocess.run(
                ["pgrep", "-f", "brain_daemon.py"],
                capture_output=True, text=True)
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    pid = int(line.strip())
                    if pid != my_pid:
                        stale_pids.add(pid)
                except ValueError:
                    continue
        except Exception:
            pass

    # Kill all stale processes with patience (they may be saving checkpoints)
    for pid in stale_pids:
        logger.info("Killing stale brain_daemon process (PID %d)", pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    if stale_pids:
        import time as _time
        # Wait up to 10 seconds for graceful shutdown
        for wait in range(10):
            _time.sleep(1)
            still_alive = []
            for pid in stale_pids:
                try:
                    os.kill(pid, 0)  # Check if alive
                    still_alive.append(pid)
                except ProcessLookupError:
                    pass
            if not still_alive:
                break
            if wait == 9:
                # Force kill after 10 seconds
                for pid in still_alive:
                    logger.warning("Force-killing unresponsive daemon PID %d", pid)
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                _time.sleep(1)  # Let kernel clean up

    # Clean stale socket
    if os.path.exists(args.socket):
        try:
            os.unlink(args.socket)
            logger.info("Removed stale socket: %s", args.socket)
        except OSError:
            pass

    # Write PID file (ensure directory exists)
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    logger.info("Brain daemon PID %d", os.getpid())

    # Import nimcp with progress display
    import nimcp
    print("\n" + "=" * 60, flush=True)
    print("  ATHENA BRAIN DAEMON — Initializing", flush=True)
    print("=" * 60, flush=True)

    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        print("  [tqdm not available — using text progress]", flush=True)

    print("  [1/8] Initializing NIMCP library...", flush=True)
    nimcp.init()
    print("  [1/8] ✓ Library initialized", flush=True)

    # === STARTUP ORPHAN CLEANUP ===
    # Sweep .tmp / .tmp.* files left from crashed-mid-save runs.
    # When the brain is SIGKILL'd or crashes, the C-side save's .tmp
    # files (especially the 10 GB .snn.tmp from gzip) persist.
    try:
        import glob as _g
        ckpt_dir = args.checkpoint_dir or "checkpoints/athena"
        if ckpt_dir and os.path.isdir(ckpt_dir):
            patterns = [
                os.path.join(ckpt_dir, "*.tmp"),
                os.path.join(ckpt_dir, "*.tmp.*"),
                os.path.join(ckpt_dir, "*.snn.tmp"),
                os.path.join(ckpt_dir, "*.lnk"),
            ]
            # Don't delete short-lived state files mid-write (audit bug #4)
            KEEP = {"immersive_state.json.tmp"}
            removed = 0
            freed = 0
            for pat in patterns:
                for f in _g.glob(pat):
                    if os.path.basename(f) in KEEP:
                        continue
                    try:
                        sz = os.path.getsize(f) if os.path.exists(f) else 0
                        os.remove(f)
                        removed += 1
                        freed += sz
                    except OSError:
                        pass
            if removed > 0:
                logger.info("Startup cleanup: removed %d orphan .tmp files (freed %.1f GB)",
                            removed, freed / 1e9)
    except Exception as _ce:
        logger.warning("Startup cleanup error: %s", _ce)

    # Determine checkpoint dir — MUST be same directory as the checkpoint file
    # to avoid saving to a different filesystem path
    if args.checkpoint_dir is None:
        if args.checkpoint:
            args.checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        else:
            args.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", "athena")
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path:
        checkpoint_path = os.path.abspath(checkpoint_path)
    if args.fresh:
        checkpoint_path = None
        logger.info("Fresh mode: ignoring checkpoints, creating new brain")
    elif args.resume and not checkpoint_path:
        default_ckpt = os.path.join(args.checkpoint_dir, "athena_immersive.bin")
        daemon_ckpt = os.path.join(args.checkpoint_dir, "athena_daemon.bin")
        if os.path.exists(daemon_ckpt):
            checkpoint_path = daemon_ckpt
        elif os.path.exists(default_ckpt):
            checkpoint_path = default_ckpt
        if checkpoint_path:
            logger.info("Auto-resume: %s", checkpoint_path)

    # Load or create brain (with .bak fallback)
    t0 = time.time()
    brain = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Audit fix #7: validate critical sidecars exist before loading.
        # Without .snn the brain loads with no spike network state but
        # training proceeds — silent partial restore.
        # Resolve symlinks so we check real files not stale dangling links.
        _real_ckpt = os.path.realpath(checkpoint_path)
        _missing_sidecars = []
        for _ext in ['.snn', '.cnn', '.lnn']:
            _sc = _real_ckpt + _ext
            if not os.path.exists(_sc):
                _missing_sidecars.append(_ext)
        if _missing_sidecars:
            logger.warning(
                "Checkpoint %s is missing critical sidecars: %s — "
                "brain will load with default %s state. Training may be wrong.",
                os.path.basename(checkpoint_path),
                _missing_sidecars,
                "/".join(s.lstrip(".") for s in _missing_sidecars))
            print(f"  [WARN] Missing sidecars: {_missing_sidecars}", flush=True)

        # Main bin + ALL sidecars — SNN sidecar is 20x larger than main.
        # Using only main-bin size (ckpt_size_main * 300) vastly under-
        # estimates the loader's wall-time and caused a false "load failed"
        # on 2026-04-24 when the background thread was still running.
        ckpt_size_main = os.path.getsize(checkpoint_path) / (1024**3)
        ckpt_size_total = ckpt_size_main
        for _ext in (".snn", ".lnn", ".cnn", ".meta", ".tokenizer",
                     ".mirror_neurons", ".executive",
                     ".cortex_visual", ".cortex_audio",
                     ".cortex_speech", ".cortex_somato"):
            _p = checkpoint_path + _ext
            if os.path.exists(_p):
                ckpt_size_total += os.path.getsize(_p) / (1024**3)
        ckpt_size = ckpt_size_total
        print(f"  [2/8] Loading checkpoint: {os.path.basename(checkpoint_path)} "
              f"({ckpt_size_main:.1f} GB main + {ckpt_size_total - ckpt_size_main:.1f} GB sidecars)",
              flush=True)

        # Show a progress bar during the blocking load
        if HAS_TQDM:
            # Brain load is blocking C code — we can't track real progress,
            # but we can show an estimated timer based on checkpoint size
            import threading as _threading
            load_done = _threading.Event()
            load_result = [None, None]  # [brain, error]

            def _load():
                try:
                    load_result[0] = nimcp.Brain("athena",
                                                   checkpoint=str(checkpoint_path),
                                                   init_mode=args.init_mode)
                except Exception as e:
                    load_result[1] = e
                load_done.set()

            t = _threading.Thread(target=_load, daemon=True)
            t.start()

            # No timeout — let the load take as long as it needs.
            # Large brains with grown metadata pools can take 15-20+ minutes.
            # ~60s per GB of total checkpoint footprint; minimum 15 minutes.
            est_seconds = max(900, int(ckpt_size * 60))
            pbar = tqdm(total=est_seconds, desc="  Loading brain",
                        unit="s", bar_format="  {desc}: {bar:40} {n_fmt}/{total_fmt}s",
                        ncols=70)
            tick = 0
            while not load_done.wait(timeout=0.1):
                if tick % 10 == 0:
                    # Cap display at est_seconds but keep waiting for the
                    # loader thread. Prior bug: exited the loop when tick
                    # hit est_seconds*10, leaving brain=None while the
                    # loader was still running → spurious "load failed".
                    if pbar.n < est_seconds:
                        pbar.update(1)
                tick += 1
            pbar.n = est_seconds
            pbar.refresh()
            pbar.close()

            if load_result[1]:
                raise load_result[1]
            brain = load_result[0]
        else:
            print("  Loading... (this takes ~10 minutes)", flush=True)
            try:
                brain = nimcp.Brain("athena", checkpoint=str(checkpoint_path),
                                    init_mode=args.init_mode)
            except Exception as e:
                logger.error("Failed to load checkpoint: %s", e)

        if brain is None:
            # Try .bak fallback
            bak_path = checkpoint_path + ".bak"
            if os.path.exists(bak_path):
                print(f"  [2/8] Trying backup: {os.path.basename(bak_path)}", flush=True)
                try:
                    brain = nimcp.Brain("athena", checkpoint=str(bak_path),
                                        init_mode=args.init_mode)
                    checkpoint_path = bak_path
                except Exception as e2:
                    logger.error("Backup checkpoint also failed: %s", e2)

    if brain is None:
        if checkpoint_path:
            # CRITICAL: If a checkpoint was specified but failed to load,
            # EXIT immediately. NEVER create a fresh brain that would
            # overwrite trained data when the auto-checkpointer runs.
            print(f"  [2/8] FATAL: Checkpoint load failed: {checkpoint_path}", flush=True)
            print(f"  [2/8] Refusing to create fresh brain — would destroy trained data.", flush=True)
            logger.critical("Checkpoint load failed for %s — exiting to protect data", checkpoint_path)
            sys.exit(1)
        print(f"  [2/8] Creating new brain: ANN={args.neuron_count:,}, "
              f"SNN={args.snn_neuron_count:,}, LNN={args.lnn_neuron_count:,}, "
              f"mode={args.init_mode}", flush=True)
        brain = nimcp.Brain("athena",
                            num_inputs=args.num_inputs,
                            num_outputs=args.num_outputs,
                            neuron_count=args.neuron_count,
                            init_mode=args.init_mode,
                            snn_neuron_count=args.snn_neuron_count,
                            lnn_neuron_count=args.lnn_neuron_count)
    elapsed = time.time() - t0
    n_neurons = brain.get_neuron_count() if brain else 0
    print(f"  [2/8] ✓ Brain loaded: {n_neurons:,} neurons in {elapsed:.1f}s", flush=True)
    logger.info("Brain ready in %.1f seconds (%d neurons)", elapsed, n_neurons)

    # CRITICAL: flip cognitive feature flags on the live brain. Without this,
    # resumed checkpoints carry whatever config was saved (often with
    # enable_sleep_wake_cycle=false from earlier training runs), and the
    # gates in brain_learn_vector / brain_decide silently skip pressure
    # accumulation, memory replay, mental-health monitoring, etc. The
    # underlying subsystems ARE created at brain init; only the per-call
    # gate flags are wrong. configure_cognitive() mutates the config struct
    # in place — no allocation, no weight change.
    try:
        brain.configure_cognitive()
        logger.info("Cognitive features enabled (sleep-wake, memory replay, "
                    "synaptic homeostasis, predictive processing, ethics, "
                    "mental health monitoring, training integration, etc)")
    except Exception as _cc_err:
        logger.warning("configure_cognitive() failed: %s — sleep-wake and "
                       "related subsystems may be inert", _cc_err)

    # UTM early-stopping: disable + reset on every brain start.
    # The C-side default trips on flat curriculum phases (Sensory
    # Enrichment in particular) and locks the Adaptive net at a
    # mode-collapsed plateau. Immersive curricula need the trainer to
    # keep updating across plateaus; the curriculum, not the optimizer,
    # decides when to advance. If a phase legitimately needs early-stop,
    # turn it back on inside that phase.
    try:
        brain.utm_reset_early_stopping()
        brain.utm_set_early_stopping_enabled(False)
        logger.info("UTM early-stopping: disabled + reset (immersive default)")
    except Exception as _es_err:
        logger.warning("utm_set_early_stopping_enabled failed: %s — "
                       "rebuild Python .so for new bindings", _es_err)

    # Reapply any SNN knob overrides saved from previous sessions. Must
    # run AFTER configure_cognitive (which may reset some defaults) so
    # the persistent values win.
    _load_persistent_snn_tunes(brain, logger)

    # Activate conductance-based PSCs (CB) by default. Runs unconditionally
    # so a --fresh daemon also gets the natural-saturation runaway defense.
    # See _activate_cb_default for the rescale-marker idempotency contract.
    _activate_cb_default(brain, logger)

    # One-shot catch-up sleep cycle after resume. Previous sessions ran
    # without enable_sleep_wake_cycle, so 1000s of learn steps accumulated
    # without memory replay, synaptic downscaling, or glymphatic clearance.
    # A single forced cycle brings the network up to date:
    #   drowsy → light-NREM (replay) → deep-NREM (downscale×0.85 +
    #   glymphatic waste clearance + microglia pruning) → REM → awake.
    # Expect a small loss transient immediately after as weights re-balance.
    # Guarded with try/except so a cycle failure never blocks daemon ready.
    try:
        print("  [2b] Running catch-up sleep cycle (memory consolidation + "
              "synaptic homeostasis)...", flush=True)
        _t_sleep = time.time()
        brain.sleep_run_cycle(1)
        logger.info("Catch-up sleep cycle complete in %.1fs",
                    time.time() - _t_sleep)
        print(f"  [2b] ✓ Catch-up sleep done ({time.time() - _t_sleep:.1f}s)",
              flush=True)
    except Exception as _sw_err:
        logger.warning("Catch-up sleep_run_cycle failed: %s — continuing "
                       "(scheduler will drive future cycles)", _sw_err)
        print(f"  [2b] ⚠ Catch-up sleep failed: {_sw_err}", flush=True)

    # CRITICAL: Set regression mode — no softmax on outputs.
    print("  [3/8] Setting task type to REGRESSION...", flush=True)
    brain.set_task_type("regression")
    print("  [3/8] ✓ Regression mode (no softmax)", flush=True)

    # Fix output layer activation
    print("  [4/8] Fixing output activation to LINEAR...", flush=True)
    brain.fix_output_activation()
    print("  [4/8] ✓ Output activation: LINEAR", flush=True)

    # Enable HNN on LNN layer 0
    print("  [5/8] Enabling Hamiltonian dynamics...", flush=True)
    try:
        brain.enable_hamiltonian(True)
        print("  [5/8] ✓ HNN enabled on LNN layer 0", flush=True)
    except Exception as e:
        print(f"  [5/8] ⚠ HNN failed (non-fatal): {e}", flush=True)

    # Eagerly create all 4 cortex CNNs
    print("  [6/8] Initializing cortex CNNs (visual/audio/speech/somato)...", flush=True)
    try:
        brain.init_cortex_cnns()
        print("  [6/8] ✓ All 4 cortex CNNs ready", flush=True)
    except Exception as e:
        print(f"  [6/8] ⚠ Cortex CNNs deferred: {e}", flush=True)

    # Eagerly initialize all cognitive subsystems that brain_decide() would
    # lazy-init via BRAIN_ENSURE_* macros. Those macros are NOT thread-safe,
    # but brain_decide() runs from the inference thread pool. By initializing
    # everything up front (single-threaded), we eliminate the race condition
    # and keep the full cognitive pipeline active during training.
    try:
        count = brain.eager_init_cognitive()
        print(f"  [6.5/8] ✓ Eager cognitive init: {count} subsystems created", flush=True)
    except Exception as e:
        print(f"  [6.5/8] ⚠ Eager cognitive init failed: {e} (non-fatal)", flush=True)

    # Optional base-lexicon bootstrap. Runs only when:
    #   1. --bootstrap-lexicon PATH was supplied, AND
    #   2. The current grounded lexicon has fewer than 300 words (so we don't
    #      double-bootstrap on --resume from a checkpoint that already has
    #      it). The 300-word threshold is comfortably above the ~248 seeded
    #      function words shipped by grounded_language_create().
    if args.bootstrap_lexicon:
        try:
            diag = brain.get_grounded_language_diagnostics()
            current_vocab = int(diag.get("vocab_size", 0)) if isinstance(diag, dict) else 0
        except Exception as _diag_e:
            current_vocab = 0
            logger.warning("bootstrap_lexicon: could not read vocab_size (%s) — proceeding",
                           _diag_e)
        if current_vocab >= 300:
            print(f"  [6.6/8] Skipping lexicon bootstrap (vocab_size={current_vocab} >= 300)",
                  flush=True)
        elif not os.path.exists(args.bootstrap_lexicon):
            logger.warning("bootstrap_lexicon: file not found: %s",
                           args.bootstrap_lexicon)
        else:
            try:
                result = brain.bootstrap_lexicon(args.bootstrap_lexicon)
                ok = bool(result.get("success", False)) if isinstance(result, dict) else bool(result)
                if ok:
                    try:
                        diag2 = brain.get_grounded_language_diagnostics()
                        new_vocab = int(diag2.get("vocab_size", 0)) if isinstance(diag2, dict) else 0
                    except Exception:
                        new_vocab = 0
                    print(f"  [6.6/8] ✓ Bootstrapped lexicon from {args.bootstrap_lexicon} "
                          f"(vocab {current_vocab} -> {new_vocab})", flush=True)
                else:
                    logger.warning("bootstrap_lexicon: load reported failure: %s", result)
            except Exception as _boot_e:
                logger.warning("bootstrap_lexicon: %s (non-fatal)", _boot_e)

    # Create service and daemon
    print("  [7/8] Creating brain service...", flush=True)
    service = BrainService(brain)
    daemon = BrainDaemon(service, socket_path=args.socket,
                          max_workers=args.workers)
    print("  [7/8] ✓ Service + daemon created", flush=True)

    # Auto-checkpoint with safety guards
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    loaded_from_ckpt = (checkpoint_path is not None and os.path.exists(checkpoint_path))

    # Pin the loaded checkpoint path globally so _load_persistent_snn_tunes
    # can consult the cb_rescaled_marker sidecar to decide whether to
    # force-rescale on CB activation.
    global _LOADED_CHECKPOINT_PATH
    _LOADED_CHECKPOINT_PATH = checkpoint_path if loaded_from_ckpt else None
    checkpointer = AutoCheckpointer(brain, args.checkpoint_dir,
                                      interval_seconds=args.checkpoint_interval,
                                      min_steps_before_save=10)
    checkpointer.set_loaded_from_checkpoint(loaded_from_ckpt)
    service.checkpointer = checkpointer  # So learn_vector can notify

    # Signal handlers — set a flag, let the main loop handle shutdown gracefully
    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        signame = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        logger.info("%s received — initiating graceful shutdown...", signame)
        shutdown_event.set()
        daemon.stop()  # unblocks serve_forever()

    def handle_sighup(signum, frame):
        logger.info("SIGHUP received — rebinding socket")
        try:
            daemon.rebind()
        except Exception as e:
            logger.error("Socket rebind failed: %s", e)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGHUP, handle_sighup)

    # SNN recovery plateau detector — auto-unfreezes ANN/CNN/LNN with
    # gradual warmup after the SNN plateaus during snn_only_recovery mode.
    # No-op when recovery mode is off.
    plateau_detector = SnnRecoveryPlateauDetector(brain)
    service.plateau_detector = plateau_detector  # Exposed for potential RPC control

    # Start
    daemon.start()
    checkpointer.start()
    plateau_detector.start()

    # Retrofit synapse metadata — restores plasticity for ALL synapses created
    # without metadata (pool exhaustion, backbone repair, sub-network init).
    # Runs AFTER all init is complete so it catches every handle-only synapse.
    try:
        retrofitted = brain.retrofit_synapse_metadata()
        if retrofitted > 0:
            logger.info("Retrofitted metadata onto %d synapses (plasticity restored)",
                        retrofitted)
        else:
            logger.info("All synapses already have metadata — no retrofit needed")
    except Exception as e:
        logger.warning("Metadata retrofit failed: %s", e)

    # Pre-warm ONNX encoder
    print("  [8/8] Loading ONNX encoder (BGE-large)...", flush=True)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from onnx_encoder import encode_text
        encode_text("warmup")
        print("  [8/8] ✓ ONNX encoder ready (CUDA)", flush=True)
    except Exception as e:
        print(f"  [8/8] ⚠ ONNX encoder not available: {e}", flush=True)

    total_elapsed = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"  ATHENA BRAIN DAEMON — Ready", flush=True)
    print(f"  Neurons: {n_neurons:,} | Mode: {args.init_mode} | "
          f"Load time: {total_elapsed:.0f}s", flush=True)
    print(f"  Socket: {args.socket}", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)
    print(f"{'=' * 60}\n", flush=True)
    logger.info("Brain daemon ready — accepting connections on %s", args.socket)

    # Sleep-wake scheduler: drives sleep_run_cycle on two triggers —
    #
    #   1. PERIODIC (primary): every --sleep-interval-min minutes of
    #      wall-clock time, regardless of pressure. Digital brains don't
    #      have metabolic/thermal sleep pressure like biology — we sleep
    #      on a schedule to keep consolidation paced with learning.
    #      Replay batch=100 memories; at ~6 learn steps/min, a 15-min
    #      period catches up ~90 new memories per cycle (near-1:1 pacing).
    #      Each cycle takes milliseconds to tens of seconds of compute
    #      (stage durations in config are decorative — stages don't
    #      honor them, they just do their work and return).
    #
    #   2. PRESSURE-DRIVEN (secondary): if sleep_is_needed becomes true
    #      between periodic ticks, fire an extra cycle. The C layer
    #      accumulates pressure via sleep_accumulate_pressure() on every
    #      learn and brain_decide probes sleep_is_needed(), but nothing
    #      in the core drives the actual state transition when pressure
    #      crosses threshold — only mental_health crisis interventions.
    #      This secondary trigger fills that gap.
    #
    # Without this thread, sleep_wake stays AWAKE forever and memory
    # consolidation, synaptic downscaling, microglia pruning, and
    # glymphatic clearance never fire.
    _sleep_interval_sec = max(60, args.sleep_interval_min * 60)
    # Seed shared state so the autotuner sees the operator-configured initial
    # interval and any persisted overrides applied by _load_persistent_snn_tunes.
    with _runtime_state_lock:
        if 'sleep_interval_sec' in _runtime_state and _runtime_state['sleep_interval_sec'] != 900:
            # A persisted override was applied earlier — keep it.
            _sleep_interval_sec = _runtime_state['sleep_interval_sec']
        else:
            _runtime_state['sleep_interval_sec'] = float(_sleep_interval_sec)

    # Resolve the CYCLE_SLEEP_WAKE constant once (available in builds with the
    # cycle-coordinator bindings; fall back to the enum's integer value on
    # older .so's so the daemon still runs if someone forgot to reinstall).
    _CYCLE_SLEEP_WAKE = getattr(nimcp, "CYCLE_SLEEP_WAKE", 2)

    def _sleep_scheduler():
        nonlocal _sleep_interval_sec
        last_state = 0  # AWAKE
        last_log_ts = 0.0
        last_periodic_cycle_ts = time.time()
        poll_sec = 30.0

        # Register with the C cycle coordinator for observability. The Python
        # thread still owns the cadence — we just notify the coordinator after
        # each cycle so health/timing dashboards include this cycle. A return
        # of 0 means "registered" OR "no coordinator present"; both are fine.
        _coord_registered = False
        try:
            if hasattr(brain, "cycle_register"):
                rc = brain.cycle_register(_CYCLE_SLEEP_WAKE)
                _coord_registered = (rc == 0)
                logger.info("[sleep] cycle_register(SLEEP_WAKE) -> %d", rc)
        except Exception as _reg_err:
            logger.warning("[sleep] cycle_register failed: %s", _reg_err)

        def _notify(duration_us):
            """Observability-only; never fatal."""
            if not _coord_registered:
                return
            try:
                if hasattr(brain, "cycle_notify_tick"):
                    brain.cycle_notify_tick(_CYCLE_SLEEP_WAKE, int(duration_us))
            except Exception as _nerr:
                logger.debug("[sleep] cycle_notify_tick failed: %s", _nerr)

        try:
            while not shutdown_event.is_set():
                try:
                    if shutdown_event.wait(timeout=poll_sec):
                        return
                    pressure = float(brain.sleep_get_pressure())
                    state = int(brain.sleep_get_state())
                    needed = bool(brain.sleep_is_needed())
                    _now = time.time()
                    # Visibility: emit state + pressure every 5 min (or on
                    # state change) so probe/monitor can read it from logs.
                    if _now - last_log_ts > 300 or state != last_state:
                        logger.info(
                            "[sleep] state=%d pressure=%.3f needed=%s "
                            "next_periodic_in=%ds",
                            state, pressure, needed,
                            int(_sleep_interval_sec - (_now - last_periodic_cycle_ts)))
                        last_log_ts = _now
                        last_state = state
                    # --- PERIODIC trigger ---
                    # Re-read interval each iteration so the autotuner can
                    # mutate it at runtime via _runtime_state.
                    with _runtime_state_lock:
                        _sleep_interval_sec = float(_runtime_state['sleep_interval_sec'])
                    elapsed_since_cycle = _now - last_periodic_cycle_ts
                    if elapsed_since_cycle >= _sleep_interval_sec:
                        logger.info(
                            "[sleep] periodic trigger (%.0fs since last cycle) — "
                            "running cycle (pressure=%.3f)",
                            elapsed_since_cycle, pressure)
                        _t_cyc = time.time()
                        _cycle_ok = False
                        try:
                            # Acquire the service dispatch lock so sleep_run_cycle
                            # (which now runs real episodic replay via
                            # brain_learn_vector internally) cannot race against a
                            # concurrent learn_vector on the socket thread.
                            with service._lock:
                                brain.sleep_run_cycle(1)
                            _cycle_ok = True
                            logger.info("[sleep] cycle complete in %.2fs",
                                        time.time() - _t_cyc)
                        except Exception as _scerr:
                            logger.warning("[sleep] periodic run_cycle failed: %s",
                                           _scerr)
                        if _cycle_ok:
                            _notify(int((time.time() - _t_cyc) * 1_000_000))
                            _cyc_dur = time.time() - _t_cyc
                            with _runtime_state_lock:
                                _runtime_state['sleep_last_cycle_complete_ts'] = time.time()
                                _runtime_state['sleep_last_cycle_duration_s'] = _cyc_dur
                                _runtime_state['sleep_cycles_completed'] += 1
                        last_periodic_cycle_ts = time.time()
                    # --- PRESSURE-DRIVEN trigger (between periodic cycles) ---
                    elif needed:
                        logger.info(
                            "[sleep] pressure %.3f ≥ threshold — extra cycle "
                            "(periodic next in %.0fs)",
                            pressure,
                            _sleep_interval_sec - elapsed_since_cycle)
                        _t_cyc = time.time()
                        _cycle_ok = False
                        try:
                            with service._lock:
                                brain.sleep_run_cycle(1)
                            _cycle_ok = True
                            logger.info("[sleep] cycle complete in %.2fs",
                                        time.time() - _t_cyc)
                        except Exception as _scerr:
                            logger.warning("[sleep] pressure run_cycle failed: %s",
                                           _scerr)
                        if _cycle_ok:
                            _notify(int((time.time() - _t_cyc) * 1_000_000))
                        last_periodic_cycle_ts = time.time()
                except Exception as _sce:
                    logger.warning("[sleep] scheduler tick failed: %s", _sce)
        finally:
            # Clean teardown — safe even if registration failed or the
            # coordinator has since been destroyed.
            if _coord_registered:
                try:
                    if hasattr(brain, "cycle_unregister"):
                        brain.cycle_unregister(_CYCLE_SLEEP_WAKE)
                except Exception as _uerr:
                    logger.debug("[sleep] cycle_unregister failed: %s", _uerr)

    _sleep_thread = threading.Thread(
        target=_sleep_scheduler, name="sleep-scheduler", daemon=True)
    _sleep_thread.start()
    logger.info("[sleep] scheduler thread started (periodic every %d min, "
                "plus pressure-driven triggers, 30s poll interval)",
                args.sleep_interval_min)

    # SNN auto-tune controller — defaults to observation-only mode. Set
    # autotune_enabled=1 in snn_tune.json (or via the snn_tune socket
    # command) to let it actually apply knob changes.
    _autotune_thread = threading.Thread(
        target=_snn_autotuner,
        args=(brain, service, shutdown_event, logger),
        name="snn-autotuner", daemon=True)
    _autotune_thread.start()
    with _runtime_state_lock:
        _at_enabled = _runtime_state['autotune_enabled']
    logger.info("[autotune] controller thread started (mode=%s, "
                "30s observation interval)",
                'APPLY' if _at_enabled else 'DRY_RUN')

    try:
        daemon.serve_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — initiating graceful shutdown...")
        shutdown_event.set()

    # --- Graceful shutdown sequence ---
    logger.info("Shutdown: stopping accept loop...")
    daemon.stop()

    logger.info("Shutdown: stopping auto-checkpointer...")
    checkpointer.stop()

    logger.info("Shutdown: stopping plateau detector...")
    try:
        plateau_detector.stop()
    except Exception:
        pass

    # Wait for any in-flight requests to finish (workers hold the brain lock).
    # Pool.shutdown(wait=True) blocks until all submitted tasks have completed,
    # then joins worker threads. Replaces the prior semaphore-based drain.
    # Drain the RO pool first — its tasks are short and don't hold the brain
    # lock, so they finish quickly and freeing those threads helps if any
    # main-pool task happens to call back into the daemon.
    logger.info("Shutdown: waiting for in-flight requests to complete...")
    try:
        daemon._ro_pool.shutdown(wait=True)
    except Exception as e:
        logger.warning("Shutdown: RO pool shutdown raised %s", e)
    try:
        daemon._pool.shutdown(wait=True)
    except Exception as e:
        logger.warning("Shutdown: pool shutdown raised %s", e)

    logger.info("Shutdown: saving final checkpoint...")
    checkpointer.save_now()

    # Clean up
    try:
        os.unlink(PID_FILE)
    except Exception:
        pass

    logger.info("Shutdown complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
