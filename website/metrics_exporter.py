#!/usr/bin/env python3
"""Exports brain metrics to a JSON file for the website to consume.

Runs as a daemon, queries the brain daemon every 10 seconds, writes
metrics.json to the website directory (served as static file by nginx).
"""
import json
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

METRICS_FILE = os.path.join(os.path.dirname(__file__), 'metrics.json')
INTERVAL = 15  # seconds (allow time for slow daemon queries)


def _safe_query(b, cmd, timeout=10):
    """Query daemon with per-call timeout, return None on failure."""
    import socket as _socket
    sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(b.socket_path)
        import json, struct
        data = json.dumps({'cmd': cmd}).encode('utf-8')
        sock.sendall(struct.pack('>I', len(data)) + data)
        hdr = b''
        while len(hdr) < 4:
            chunk = sock.recv(4 - len(hdr))
            if not chunk: return None
            hdr += chunk
        length = struct.unpack('>I', hdr)[0]
        body = b''
        while len(body) < length:
            chunk = sock.recv(min(length - len(body), 65536))
            if not chunk: return None
            body += chunk
        return json.loads(body.decode('utf-8'))
    except Exception:
        return None
    finally:
        sock.close()


def collect_metrics():
    """Query brain daemon and return metrics dict.

    Each query is independent — a slow or failed query doesn't block others.
    Uses _send_once (no retry) to avoid blocking for minutes on a dead daemon.
    """
    from brain_client import BrainProxy
    b = BrainProxy(timeout=10)
    metrics = {'timestamp': time.time(), 'ok': False}

    # Status — should be fast, but daemon under load can be slow
    r = _safe_query(b, 'status', timeout=15)
    if r and not r.get('error'):
        metrics['uptime'] = r.get('uptime', 0)
        metrics['learn_calls'] = r.get('learn_calls', 0)
        metrics['infer_calls'] = r.get('infer_calls', 0)
        metrics['errors'] = r.get('errors', 0)
        metrics['ok'] = True
    else:
        return metrics  # Daemon down — return partial (still written to file)

    # Neuron count — fast
    r = _safe_query(b, 'get_neuron_count', timeout=5)
    if r:
        metrics['neuron_count'] = r.get('neuron_count', 0)

    # Network metrics — can be slow (4-5s)
    r = _safe_query(b, 'get_network_metrics', timeout=10)
    if r:
        m = r.get('metrics', r)
        metrics['ann_loss'] = m.get('ann_loss', 0)
        metrics['cnn_loss'] = m.get('cnn_loss', 0)
        metrics['snn_loss'] = m.get('snn_loss', 0)
        metrics['lnn_loss'] = m.get('lnn_loss', 0)
        metrics['ann_steps'] = m.get('ann_steps', 0)

    # SNN stats — fast
    r = _safe_query(b, 'get_snn_stats', timeout=5)
    if r:
        s = r.get('snn', r)
        metrics['snn_spikes'] = s.get('total_spikes', 0)
        metrics['snn_rate_hz'] = s.get('mean_firing_rate_hz', 0)
        metrics['snn_sparsity'] = s.get('sparsity', 0)
        metrics['snn_silent'] = s.get('silent_neurons', 0)
        metrics['snn_populations'] = s.get('n_populations', 0)

    # Cortex CNN — can be slow (4-5s)
    r = _safe_query(b, 'get_cortex_cnn_metrics', timeout=10)
    if r:
        m = r.get('metrics', r)
        cortex = {}
        for name in ['visual', 'audio', 'speech', 'somato']:
            if name in m:
                c = m[name]
                cortex[name] = {
                    'loss': c.get('last_loss', 0),
                    'ema_loss': c.get('ema_loss', 0),
                    'fwd': c.get('forward_steps', 0),
                    'bwd': c.get('backward_steps', 0),
                }
        metrics['cortex'] = cortex

    # Training log — local file, always fast
    try:
        log_path = os.path.join(os.path.dirname(__file__), '..', 'training.log')
        if os.path.exists(log_path):
            import re
            # Read last 10KB only (avoid reading entire log)
            with open(log_path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 10240))
                tail = f.read().decode('utf-8', errors='replace')
            for line in reversed(tail.splitlines()):
                m = re.match(r'.*\[(\d{4,})\]\s+loss=([0-9.]+)', line)
                if m:
                    metrics['current_step'] = int(m.group(1))
                    metrics['step_loss'] = float(m.group(2))
                    break
    except Exception:
        pass

    return metrics


def main():
    print(f"Metrics exporter starting — writing to {METRICS_FILE} every {INTERVAL}s",
          flush=True)
    consecutive_failures = 0
    while True:
        t0 = time.time()
        try:
            metrics = collect_metrics()
            # Atomic write: write to temp then rename
            tmp = METRICS_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(metrics, f)
            os.rename(tmp, METRICS_FILE)
            elapsed = time.time() - t0
            if consecutive_failures > 0:
                print(f"Recovered after {consecutive_failures} failures "
                      f"(cycle took {elapsed:.1f}s)", flush=True)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures <= 3 or consecutive_failures % 10 == 0:
                print(f"Export error ({consecutive_failures}): {e}", flush=True)
        # Sleep remaining time (account for query duration)
        elapsed = time.time() - t0
        sleep_time = max(1, INTERVAL - elapsed)
        time.sleep(sleep_time)


if __name__ == '__main__':
    main()
