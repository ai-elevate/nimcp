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
INTERVAL = 10  # seconds


def collect_metrics():
    """Query brain daemon and return metrics dict."""
    from brain_client import BrainProxy
    b = BrainProxy(timeout=10)
    metrics = {'timestamp': time.time(), 'ok': False}

    try:
        r = b._send({'cmd': 'status'})
        metrics['uptime'] = r.get('uptime', 0)
        metrics['learn_calls'] = r.get('learn_calls', 0)
        metrics['infer_calls'] = r.get('infer_calls', 0)
        metrics['errors'] = r.get('errors', 0)
        metrics['ok'] = True
    except Exception:
        return metrics

    try:
        r = b._send({'cmd': 'get_neuron_count'})
        metrics['neuron_count'] = r.get('neuron_count', 0)
    except Exception:
        pass

    try:
        r = b._send({'cmd': 'get_network_metrics'})
        m = r.get('metrics', r)
        metrics['ann_loss'] = m.get('ann_loss', 0)
        metrics['cnn_loss'] = m.get('cnn_loss', 0)
        metrics['snn_loss'] = m.get('snn_loss', 0)
        metrics['lnn_loss'] = m.get('lnn_loss', 0)
        metrics['ann_steps'] = m.get('ann_steps', 0)
    except Exception:
        pass

    try:
        r = b._send({'cmd': 'get_snn_stats'})
        s = r.get('snn', r)
        metrics['snn_spikes'] = s.get('total_spikes', 0)
        metrics['snn_rate_hz'] = s.get('mean_firing_rate_hz', 0)
        metrics['snn_sparsity'] = s.get('sparsity', 0)
        metrics['snn_silent'] = s.get('silent_neurons', 0)
        metrics['snn_populations'] = s.get('n_populations', 0)
    except Exception:
        pass

    try:
        r = b._send({'cmd': 'get_cortex_cnn_metrics'})
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
    except Exception:
        pass

    # Read latest step from training log
    try:
        log_path = os.path.join(os.path.dirname(__file__), '..', 'training.log')
        if os.path.exists(log_path):
            import re
            with open(log_path) as f:
                lines = f.readlines()
            for line in reversed(lines):
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
    while True:
        try:
            metrics = collect_metrics()
            # Atomic write: write to temp then rename
            tmp = METRICS_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(metrics, f)
            os.rename(tmp, METRICS_FILE)
        except Exception as e:
            print(f"Export error: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
