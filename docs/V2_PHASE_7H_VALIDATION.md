# V2 Phase 7h — Live Daemon Validation Runbook

**Goal**: prove the V2 daemon survives a continuous ≥1hr training load
on real pod hardware without crashing, leaking memory, or corrupting
its observability surface. This is the gate before V2 is considered
production-ready alongside V1.

**Status**: harness written + committed; live execution **deferred** —
the RunPod box (`74.2.96.55:17653`) was unreachable at the time the
phase was scheduled (connection refused). Operator runs the harness
once the pod is back up.

## Prerequisites

| Item                                     | Where                                  |
|------------------------------------------|----------------------------------------|
| `nimcp-v2-daemon` binary built `--release` | `/opt/nimcp-v2/bin/nimcp-v2-daemon`   |
| systemd unit installed + enabled         | `/etc/systemd/system/nimcp-v2-daemon.service` |
| `BrainConfig` JSON                       | `/etc/nimcp-v2/config.json`            |
| State dir                                | `/var/lib/nimcp-v2` (owned by `nimcp:nimcp`) |
| Log dir                                  | `/var/log/nimcp-v2` (owned by `nimcp:nimcp`) |
| Tools                                    | `socat`, `python3`, `journalctl`, `systemctl` |

## Pre-flight (one-shot)

```bash
# Copy unit + harness to the pod.
scp deployment/v2/nimcp-v2-daemon.service root@pod:/etc/systemd/system/
scp scripts/v2_phase7h_validation.sh root@pod:/opt/nimcp-v2/bin/

# On the pod:
sudo useradd --system --no-create-home --shell /sbin/nologin nimcp || true
sudo install -d -m 0750 -o nimcp -g nimcp /var/lib/nimcp-v2 /var/log/nimcp-v2
sudo install -d -m 0755 /etc/nimcp-v2

# Minimal BrainConfig — adjust layers + backend to match your pod profile.
sudo tee /etc/nimcp-v2/config.json <<'EOF'
{
  "rng_seed": 24009,
  "deterministic": false,
  "state_dir": "/var/lib/nimcp-v2",
  "adaptive": { "layers": [64, 32, 10], "rng_seed": 24009, "activation": "Tanh" },
  "snn": null,
  "lnn": null,
  "memory": null,
  "backend": "gpu"
}
EOF

sudo systemctl daemon-reload
sudo systemctl enable nimcp-v2-daemon.service
```

## Run the validation

```bash
# On the pod — defaults to 3600s (1hr). Override for longer soaks.
sudo VALIDATION_DURATION_SEC=3600 \
     /opt/nimcp-v2/bin/v2_phase7h_validation.sh
```

Exit code:

| Code | Meaning                                                       |
|-----:|---------------------------------------------------------------|
| 0    | All acceptance checks passed — Phase 7h SHIP                  |
| 2    | Pre-check failed (binary / config / socket dir)               |
| 3    | Daemon failed to start within 30 s of `systemctl start`       |
| 4    | Protocol error during the drive loop                          |
| 5    | One or more acceptance checks failed (see artifact dir)       |

## Acceptance checks

The harness asserts six things, all evaluated at the end of the window:

| Check                       | Requirement                                                 |
|-----------------------------|-------------------------------------------------------------|
| `no_restarts`               | `systemctl show -p NRestarts` is `0`                        |
| `no_sigsegv_or_panic`       | Journal has no `SIGSEGV` / `panicked at` / `panic:` lines   |
| `metrics_advanced`          | `metrics.json` mtime advanced >60 times during the window   |
| `rss_no_serious_leak`       | End RSS ≤ 1.5× start RSS (lenient — catches gross leaks)    |
| `stats_log_nonempty`        | At least one successful `stats()` snapshot was captured     |
| `service_still_active`      | `systemctl is-active` returns 0 at end of run               |

## Artifacts captured

Every run drops a timestamped dir under
`/var/log/nimcp-v2/phase7h-<UTC>/`:

| File                | What                                                      |
|---------------------|-----------------------------------------------------------|
| `run.log`           | Wall-clock log of the harness                             |
| `journal.log`       | `journalctl -u nimcp-v2-daemon` for the run window        |
| `stats.jsonl`       | One stats snapshot per `STATS_INTERVAL_SEC` (default 30s) |
| `learn.log`         | Stdout of the per-second `learn()` socket calls           |
| `ckpt.log`          | Stdout of the every-5-min `save()` calls                  |
| `metrics_history.tsv` | Per-5s stamp of `metrics.json` mtime (for `metrics_advanced` check) |
| `ckpt-<unix-ts>.bin`| Periodic checkpoints (manual sanity-check possible)       |

## What success looks like

After 1 hr the harness should print:

```
== Phase 7h VALIDATION PASS — V2 daemon survived 3600s under load ==
```

and the artifact dir should contain a stats.jsonl whose final entry
shows >= 3600 successful `learn()` calls + several hundred `predict()`
/ `stats()` reads. RSS in the journal should be flat (or growing
linearly with checkpoint size if you didn't disable the auto-snapshot).

## Failure triage cheatsheet

| Symptom                             | Likely cause + next step                                 |
|-------------------------------------|----------------------------------------------------------|
| Exit 2 `daemon binary missing`      | `cargo build --release --features cuda -p nimcp-daemon` then re-deploy |
| Exit 3 `socket did not appear`      | `sudo journalctl -u nimcp-v2-daemon -n 100` — usually permission/path |
| Exit 4 during drive loop            | Check `learn.log` / `stats.log`; protocol mismatch or daemon crash mid-call |
| Exit 5 `no_restarts` failed         | `systemctl status nimcp-v2-daemon` — count + reason in unit state |
| Exit 5 `rss_no_serious_leak` failed | Compare `start_rss_kb` / `end_rss_kb` in `run.log`; profile with `heaptrack` next run |
| Exit 5 `metrics_advanced` failed    | Daemon got into a tight `Mutex` hot-loop; check journal for warnings around `metrics_writer` |

## Tuning the run

| Env var                       | Default                             | Use case                           |
|-------------------------------|-------------------------------------|------------------------------------|
| `VALIDATION_DURATION_SEC`     | `3600`                              | Bump to 7200 / 86400 for longer soaks |
| `LEARN_INTERVAL_SEC`          | `1`                                 | Drop to 0.1 to stress the request rate |
| `STATS_INTERVAL_SEC`          | `30`                                | Drop to 5 to detect protocol drift faster |
| `CHECKPOINT_INTERVAL_SEC`     | `300`                               | Drop to 60 to test save-path more often |
| `ARTIFACT_DIR`                | `/var/log/nimcp-v2/phase7h-<UTC>/`  | Pin to inspect prior runs                |

## What this run does NOT cover

- **Real workload mix.** The harness drives one synthetic feature
  vector + one synthetic target. Real-world drift patterns may
  expose issues this run won't.
- **GPU-specific stress.** When `backend=gpu` the kernels run, but
  the harness doesn't verify GPU utilisation, VRAM growth, or
  PCIe-bus pressure. Use `nvidia-smi dmon` in a separate terminal.
- **Recovery from kill -9.** This is a clean-shutdown soak. A
  hard-kill recovery harness is a separate Phase (8b candidate).

## Scheduling

Once the pod is back up, this is a one-shot operator run — not
recurring. After a clean PASS, the next time you'd re-run is on
upgrades to the daemon binary or the BrainConfig.
