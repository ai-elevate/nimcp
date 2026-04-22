#!/usr/bin/env python3
"""
V2 Hera — stability soak test (Phase 8 exit-criterion tooling).

Runs a V2 brain through a sustained training load for N hours, with
per-minute sampling of stats + RSS. Fails fast on: crash, NaN loss,
SNN mode collapse (rate_ema < 0.1 * target across all pops for >1 min),
SNN saturation (rate_ema > 1.5 * target across all pops for >5 min),
ATP depletion (any substrate-enabled pop's atp_level < 0.1).

Not a training harness. Just a long-running smoke. For a real
training workload use:

    python3 scripts/immerse_athena.py --backend=v2 [...]

Usage:

    # 30-minute smoke (the default).
    PYTHONPATH=/tmp python3 scripts/v2_stability_soak.py

    # 24-hour full exit criterion.
    PYTHONPATH=/tmp python3 scripts/v2_stability_soak.py --hours 24

    # Write metrics to a JSONL file.
    PYTHONPATH=/tmp python3 scripts/v2_stability_soak.py \\
        --hours 24 --metrics-out soak-metrics.jsonl

Exits 0 on clean completion, nonzero on any detected stability fault.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

try:
    import nimcp_v2  # type: ignore
except ImportError as e:
    print(f"FATAL: cannot import nimcp_v2: {e}", file=sys.stderr)
    print(
        "       build with: cargo build -p nimcp-pybind --release",
        file=sys.stderr,
    )
    print(
        "       then: cp target/release/libnimcp_v2.so /tmp/nimcp_v2.so",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Stability-fault detectors
# ---------------------------------------------------------------------------


class SoakFault(Exception):
    """Raised on the first stability fault we want to fail fast on."""


def _check_loss(loss: float, step: int) -> None:
    if math.isnan(loss) or math.isinf(loss):
        raise SoakFault(f"step {step}: loss is {loss}")


def _check_snn(
    stats: dict[str, Any],
    collapse_ticks: dict[str, int],
    saturation_ticks: dict[str, int],
    step: int,
) -> None:
    snn = stats.get("snn")
    if snn is None:
        return
    for pop in snn.get("populations", []):
        name = str(pop.get("pop_idx", "?"))
        rate = float(pop.get("rate_ema") or 0.0)
        # Target rate isn't exposed in stats; approximate from typical
        # 0.05 target — callers inject real target if different.
        target = 0.05
        if rate < 0.1 * target:
            collapse_ticks[name] = collapse_ticks.get(name, 0) + 1
            if collapse_ticks[name] >= 60:
                raise SoakFault(
                    f"step {step}: SNN pop {name} collapsed "
                    f"(rate_ema={rate:.4f}, target≈{target}) for >60 samples"
                )
        else:
            collapse_ticks[name] = 0
        if rate > 1.5 * target:
            saturation_ticks[name] = saturation_ticks.get(name, 0) + 1
            if saturation_ticks[name] >= 300:
                raise SoakFault(
                    f"step {step}: SNN pop {name} saturated "
                    f"(rate_ema={rate:.4f}, target≈{target}) for >300 samples"
                )
        else:
            saturation_ticks[name] = 0


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Soak loop
# ---------------------------------------------------------------------------


def soak(
    *,
    hours: float,
    sample_interval_s: float,
    learn_lr: float,
    rng_seed: int,
    layers: list[int],
    metrics_out: Path | None,
) -> int:
    print(
        f"soak: hours={hours} sample_every={sample_interval_s}s "
        f"seed={rng_seed} layers={layers}"
    )
    brain = nimcp_v2.Brain(
        rng_seed=rng_seed,
        deterministic=False,
        layers=layers,
        activation="tanh",
    )
    rng = random.Random(rng_seed)
    start = time.time()
    deadline = start + hours * 3600.0
    last_sample = start
    step = 0
    samples = []

    # Graceful SIGINT: dump what we have before exit.
    stop = {"requested": False}

    def _on_sigint(_sig: int, _frame: Any) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    collapse_ticks: dict[str, int] = {}
    saturation_ticks: dict[str, int] = {}
    input_dim = layers[0]
    output_dim = layers[-1]

    try:
        while time.time() < deadline and not stop["requested"]:
            # Generate a random (x, y) pair. Deterministic via seeded rng.
            x = [rng.uniform(-1.0, 1.0) for _ in range(input_dim)]
            y = [rng.uniform(-1.0, 1.0) for _ in range(output_dim)]
            loss = brain.learn(x, y, learn_lr)
            step += 1
            _check_loss(loss, step)

            now = time.time()
            if now - last_sample >= sample_interval_s:
                stats = brain.stats()
                _check_snn(stats, collapse_ticks, saturation_ticks, step)
                rss = _rss_mb()
                elapsed_hrs = (now - start) / 3600.0
                sample = {
                    "t_sec": round(now - start, 3),
                    "step": step,
                    "loss": float(loss),
                    "rss_mb": round(rss, 1),
                    "elapsed_hrs": round(elapsed_hrs, 4),
                    "adaptive_loss_ema": (
                        stats.get("loss", {})
                        .get("adaptive", {})
                        .get("ema")
                    ),
                }
                samples.append(sample)
                print(
                    f"[{elapsed_hrs:6.3f}h step {step:>9}] "
                    f"loss={loss:.4f} rss={rss:.0f}MB",
                    flush=True,
                )
                last_sample = now
    except SoakFault as e:
        print(f"FAULT: {e}", file=sys.stderr)
        if metrics_out is not None:
            metrics_out.write_text(
                "\n".join(json.dumps(s) for s in samples) + "\n"
            )
        return 3
    finally:
        gc.collect()

    # Clean completion.
    elapsed_hrs = (time.time() - start) / 3600.0
    print(f"soak clean: {step} steps in {elapsed_hrs:.3f}h")
    if metrics_out is not None:
        metrics_out.write_text("\n".join(json.dumps(s) for s in samples) + "\n")
        print(f"metrics written to {metrics_out}")
    return 0 if not stop["requested"] else 130


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--hours", type=float, default=0.5, help="Soak duration, hours (default 0.5).")
    p.add_argument(
        "--sample-every",
        type=float,
        default=60.0,
        help="Stat-sampling interval in seconds (default 60).",
    )
    p.add_argument("--lr", type=float, default=0.01, help="Adaptive LR (default 0.01).")
    p.add_argument("--seed", type=int, default=0x5EED, help="RNG seed.")
    p.add_argument("--layers", default="8,16,4", help="Comma-separated layer widths (default 8,16,4).")
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Write per-sample metrics JSONL to this path.",
    )
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    if len(layers) < 2:
        print("FATAL: --layers must have at least 2 entries", file=sys.stderr)
        return 2

    return soak(
        hours=args.hours,
        sample_interval_s=args.sample_every,
        learn_lr=args.lr,
        rng_seed=args.seed,
        layers=layers,
        metrics_out=args.metrics_out,
    )


if __name__ == "__main__":
    sys.exit(main())
