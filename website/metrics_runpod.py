#!/usr/bin/env python3
"""Metrics exporter: queries brain daemon via TCP, writes metrics.json."""
import json, struct, socket, time, os, subprocess

BRAIN_HOST = '127.0.0.1'
BRAIN_PORT = 9900
METRICS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics.json')
STEP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.step_cache.json')
INTERVAL = 60


def query(cmd, timeout=15):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((BRAIN_HOST, BRAIN_PORT))
        data = json.dumps({'cmd': cmd}).encode()
        s.sendall(struct.pack('>I', len(data)) + data)
        hdr = b''
        while len(hdr) < 4:
            c = s.recv(4 - len(hdr))
            if not c:
                return None
            hdr += c
        length = struct.unpack('>I', hdr)[0]
        body = b''
        while len(body) < length:
            c = s.recv(min(length - len(body), 65536))
            if not c:
                return None
            body += c
        return json.loads(body)
    except Exception as e:
        print(f'[metrics] Query {cmd} failed: {e}', flush=True)
        return None
    finally:
        s.close()


def fetch_step_cache():
    """Fetch immersive_state.json from RunPod in background."""
    try:
        subprocess.Popen(
            ['ssh', 'root@74.2.96.55', '-p', '11008',
             '-i', os.path.expanduser('~/.ssh/id_ed25519_runpod'),
             '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
             'cat /workspace/nimcp/checkpoints/athena/immersive_state.json 2>/dev/null'],
            stdout=open(STEP_CACHE, 'w'),
            stderr=subprocess.DEVNULL)
    except Exception:
        pass


def collect():
    metrics = {'timestamp': time.time(), 'ok': False}

    r = query('status', 30)
    if not r or r.get('error'):
        return metrics

    metrics['uptime'] = r.get('uptime', 0)
    metrics['learn_calls'] = r.get('learn_calls', 0)
    metrics['infer_calls'] = r.get('infer_calls', 0)
    metrics['errors'] = r.get('errors', 0)
    metrics['ok'] = True

    r = query('get_neuron_count', 15)
    if r:
        metrics['neuron_count'] = r.get('neuron_count', 0)

    r = query('get_network_metrics', 25)
    if r:
        m = r.get('metrics', r)
        metrics['ann_loss'] = m.get('ann_loss', 0)
        metrics['cnn_loss'] = m.get('cnn_loss', 0)
        metrics['snn_loss'] = m.get('snn_loss', 0)
        metrics['lnn_loss'] = m.get('lnn_loss', 0)
        metrics['ann_steps'] = m.get('ann_steps', 0)
        # FNO metrics
        metrics['fno_loss'] = m.get('fno_audio_ema_loss', 0)
        metrics['fno_steps'] = m.get('fno_audio_steps', 0)
        metrics['fno_params'] = m.get('fno_audio_params', 0)
        # HNN metrics
        metrics['hnn_active'] = m.get('hnn_active', False)
        metrics['hnn_energy'] = m.get('hnn_energy', 0)
        metrics['hnn_energy_deviation'] = m.get('hnn_energy_deviation', 0)

    r = query('get_snn_stats', 15)
    if r:
        s = r.get('snn', r)
        metrics['snn_spikes'] = s.get('total_spikes', 0)
        metrics['snn_rate_hz'] = s.get('mean_firing_rate_hz', 0)
        metrics['snn_sparsity'] = s.get('sparsity', 0)

    # Read cached step info
    try:
        with open(STEP_CACHE) as f:
            sc = json.loads(f.read())
        metrics['current_step'] = sc.get('step', 0)
        metrics['current_stage'] = sc.get('stage', 0)
    except Exception:
        pass

    # Stage 2 dashboard metrics — read from training log tail
    stage = metrics.get('current_stage', 0)
    if stage >= 2:
        try:
            # Parse last progress line from training log
            result = subprocess.run(
                ['ssh', 'root@74.2.96.55', '-p', '11008',
                 '-i', os.path.expanduser('~/.ssh/id_ed25519_runpod'),
                 '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
                 'grep -E "Stage 2.*fact_ratio|COLLAPSE|World Model|Warm Start" '
                 '/workspace/nimcp/training.log | tail -5'],
                capture_output=True, text=True, timeout=10)
            s2 = {}
            for line in result.stdout.strip().split('\n'):
                if 'fact_ratio=' in line:
                    import re
                    fr = re.search(r'fact_ratio=(\d+)%', line)
                    if fr: s2['fact_ratio'] = int(fr.group(1)) / 100.0
                if 'COLLAPSE' in line.upper():
                    s2['collapse_events'] = s2.get('collapse_events', 0) + 1
                if 'Warm Start' in line and 'Complete' in line:
                    s2['warm_start_complete'] = True
                if 'World Model' in line:
                    import re
                    wm = re.search(r'(\d+) transitions', line)
                    if wm: s2['wm_steps'] = int(wm.group(1))
            s2['active_engines'] = 57  # we know this from the build
            metrics['stage2'] = s2
        except Exception:
            pass

    return metrics


if __name__ == '__main__':
    print(f'[metrics] Starting — writing to {METRICS_FILE}', flush=True)
    _last_learn_calls = 0
    _last_learn_time = time.time()
    _stall_warned = False

    while True:
        try:
            m = collect()

            # Detect stalled training: if learn_calls hasn't increased in 5 min
            current_learns = m.get('learn_calls', 0)
            now = time.time()
            if current_learns > _last_learn_calls:
                _last_learn_calls = current_learns
                _last_learn_time = now
                _stall_warned = False
                m['training_active'] = True
            elif m.get('ok') and (now - _last_learn_time) > 300:
                m['training_active'] = False
                if not _stall_warned:
                    print(f'[metrics] WARNING: Training stalled — no learns in '
                          f'{int(now - _last_learn_time)}s', flush=True)
                    _stall_warned = True
            else:
                m['training_active'] = True  # Within grace period

            tmp = METRICS_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(m, f)
            os.rename(tmp, METRICS_FILE)
            print(f'[metrics] ok={m["ok"]} learn={m.get("learn_calls","?")} '
                  f'step={m.get("current_step","?")} active={m.get("training_active","?")}',
                  flush=True)
        except Exception as e:
            print(f'[metrics] Error: {e}', flush=True)

        # Fetch step count in background
        fetch_step_cache()

        time.sleep(INTERVAL)
