#!/usr/bin/env python3
"""
brain_client.py — Client library for the Brain Daemon

Provides a BrainProxy class that has the same API as nimcp.Brain but
forwards all calls over the Unix socket to the brain_daemon.py process.

Usage:
    from brain_client import BrainProxy, is_daemon_running

    if is_daemon_running():
        brain = BrainProxy()
    else:
        import nimcp
        brain = nimcp.Brain("athena", ...)

    # Same API either way:
    loss = brain.learn_vector(features, target, label="dog")
    result = brain.decide_full(features)
"""

import json
import os
import socket
import struct
import time

SOCKET_PATH = "/var/run/athena/brain.sock"
DEFAULT_TIMEOUT = 60.0


def is_daemon_running(socket_path=SOCKET_PATH):
    """Check if the brain daemon is running and responsive."""
    if not os.path.exists(socket_path):
        return False
    try:
        proxy = BrainProxy(socket_path=socket_path, timeout=5.0)
        resp = proxy._send({"cmd": "ping"})
        return resp.get("ok", False)
    except Exception:
        return False


class BrainProxy:
    """Drop-in replacement for nimcp.Brain that proxies to the daemon.

    All methods match the nimcp.Brain API. The daemon handles thread safety.
    """

    def __init__(self, socket_path=SOCKET_PATH, timeout=DEFAULT_TIMEOUT):
        self.socket_path = socket_path
        self.timeout = timeout

    def _send(self, req):
        """Send a command and return the response."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
            data = json.dumps(req, default=_json_default).encode("utf-8")
            sock.sendall(struct.pack(">I", len(data)) + data)

            # Receive response
            hdr = b""
            while len(hdr) < 4:
                chunk = sock.recv(4 - len(hdr))
                if not chunk:
                    raise ConnectionError("Daemon closed connection")
                hdr += chunk
            length = struct.unpack(">I", hdr)[0]
            body = b""
            while len(body) < length:
                chunk = sock.recv(min(length - len(body), 65536))
                if not chunk:
                    raise ConnectionError("Daemon closed connection")
                body += chunk
            resp = json.loads(body.decode("utf-8"))
            if "error" in resp and resp["error"]:
                raise RuntimeError(f"Brain daemon error: {resp['error']}")
            return resp
        finally:
            sock.close()

    # -- Core learning --

    def learn_vector(self, features, target, label=None, confidence=None,
                     learning_rate=None):
        req = {"cmd": "learn_vector",
               "features": _to_list(features),
               "target": _to_list(target)}
        if label is not None:
            req["label"] = label
        if confidence is not None:
            req["confidence"] = float(confidence)
        if learning_rate is not None:
            req["learning_rate"] = float(learning_rate)
        resp = self._send(req)
        return resp.get("loss", 0.0)

    def learn_vector_batch(self, pairs, learning_rate=None):
        req = {"cmd": "learn_vector_batch",
               "pairs": [[_to_list(f), _to_list(t)] for f, t in pairs]}
        if learning_rate is not None:
            req["learning_rate"] = float(learning_rate)
        resp = self._send(req)
        return resp.get("avg_loss", 0.0)

    # -- Inference --

    def decide_full(self, features):
        resp = self._send({"cmd": "decide_full",
                           "features": _to_list(features)})
        return resp.get("result", {})

    def predict(self, features):
        resp = self._send({"cmd": "predict",
                           "features": _to_list(features)})
        return resp.get("result", {})

    def speak(self, output_vector):
        resp = self._send({"cmd": "speak",
                           "output_vector": _to_list(output_vector)})
        return resp.get("result", {})

    def generate_text(self, output_vector):
        resp = self._send({"cmd": "generate_text",
                           "output_vector": _to_list(output_vector)})
        return resp.get("result", {})

    def grounded_respond(self, text):
        resp = self._send({"cmd": "grounded_respond", "text": text})
        return resp.get("result", {})

    # -- LNN --

    def lnn_forward_step(self, features):
        resp = self._send({"cmd": "lnn_forward_step",
                           "features": _to_list(features)})
        return resp.get("result")

    def lnn_get_state(self):
        resp = self._send({"cmd": "lnn_get_state"})
        return resp.get("state")

    # -- Checkpoint --

    def save(self, path):
        self._send({"cmd": "save", "path": path})

    # -- Monitoring --

    def get_stats(self):
        resp = self._send({"cmd": "stats"})
        return resp.get("stats", {})

    def get_accuracy(self):
        resp = self._send({"cmd": "get_accuracy"})
        return resp.get("accuracy", 0.0)

    def get_last_gradient_norm(self):
        resp = self._send({"cmd": "get_last_gradient_norm"})
        return resp.get("gradient_norm", 0.0)

    def get_neuron_count(self):
        resp = self._send({"cmd": "get_neuron_count"})
        return resp.get("neuron_count", 0)

    def get_transcript(self):
        resp = self._send({"cmd": "get_transcript"})
        return resp.get("transcript")

    def get_cognitive_stats(self):
        resp = self._send({"cmd": "get_cognitive_stats"})
        return resp.get("stats", {})

    def probe(self):
        resp = self._send({"cmd": "probe"})
        return resp.get("probe", {})

    def get_network_metrics(self):
        resp = self._send({"cmd": "get_network_metrics"})
        return resp.get("metrics", {})

    def get_cortex_cnn_metrics(self):
        resp = self._send({"cmd": "get_cortex_cnn_metrics"})
        return resp.get("metrics", {})

    # -- Phi-3 Language Cortex --

    def phi3_generate(self, text, max_tokens=256):
        """Generate a response using Phi-3 conditioned on brain state."""
        resp = self._send({"cmd": "phi3_generate", "text": text,
                           "max_tokens": max_tokens})
        return resp

    # -- Identity --

    def get_identity(self):
        """Get Athena's current identity state (personality, voice, accent)."""
        return self._send({"cmd": "get_identity"})

    # -- TTS --

    def tts_speak(self, text, accent=None, output_path=None):
        """Synthesize speech with brain-state modulation."""
        resp = self._send({"cmd": "tts_speak", "text": text,
                           "accent": accent, "output_path": output_path})
        return resp

    def tts_register_accent(self, name, audio_path):
        """Register a new accent from reference audio."""
        return self._send({"cmd": "tts_register_accent", "name": name,
                           "audio_path": audio_path})

    def tts_list_accents(self):
        """List available accents."""
        return self._send({"cmd": "tts_list_accents"})

    # -- Biological state --

    def substrate_get_health(self):
        resp = self._send({"cmd": "substrate_get_health"})
        return resp.get("health", "UNKNOWN")

    def substrate_get_metabolic(self):
        resp = self._send({"cmd": "substrate_get_metabolic"})
        return resp.get("metabolic", {})

    def medulla_get_arousal(self):
        resp = self._send({"cmd": "medulla_get_arousal"})
        return resp.get("arousal", 0.5)

    def medulla_get_circadian_efficiency(self):
        resp = self._send({"cmd": "medulla_get_circadian_efficiency"})
        return resp.get("efficiency", 1.0)

    def sleep_get_pressure(self):
        resp = self._send({"cmd": "sleep_get_pressure"})
        return resp.get("pressure", 0.0)

    def sleep_get_state(self):
        resp = self._send({"cmd": "sleep_get_state"})
        return resp.get("state", 0)

    def sleep_is_needed(self):
        resp = self._send({"cmd": "sleep_is_needed"})
        return resp.get("needed", False)

    def sleep_run_cycle(self, duration=2):
        self._send({"cmd": "sleep_run_cycle", "duration": duration})

    def update_medulla(self, dt=0.1):
        self._send({"cmd": "update_medulla", "dt": dt})

    def bg_get_dopamine(self):
        resp = self._send({"cmd": "bg_get_dopamine"})
        return resp.get("dopamine", 0.0)

    def bg_get_rpe(self):
        resp = self._send({"cmd": "bg_get_rpe"})
        return resp.get("rpe", 0.0)

    def bg_get_conflict(self):
        resp = self._send({"cmd": "bg_get_conflict"})
        return resp.get("conflict", 0.0)

    def bg_get_mode(self):
        resp = self._send({"cmd": "bg_get_mode"})
        return resp.get("mode", 0)

    def bg_update_reward(self, reward, rpe=0.0):
        self._send({"cmd": "bg_update_reward", "reward": reward, "rpe": rpe})

    # -- Training config --

    def set_plasticity_state(self, state):
        self._send({"cmd": "set_plasticity_state", "state": state})

    def set_task_type(self, task_type):
        self._send({"cmd": "set_task_type", "task_type": task_type})

    def set_fast_training(self, enabled):
        self._send({"cmd": "set_fast_training", "enabled": enabled})

    def reinit_weights(self):
        self._send({"cmd": "reinit_weights"})

    def enable_biological_plasticity(self, enabled):
        self._send({"cmd": "enable_biological_plasticity", "enabled": enabled})

    def consolidate(self, mode="auto"):
        self._send({"cmd": "consolidate", "mode": mode})

    def cerebellum_process_error(self, error):
        self._send({"cmd": "cerebellum_process_error", "error": error})

    # -- UTM --

    def utm_get_training_health(self):
        resp = self._send({"cmd": "utm_get_training_health"})
        return resp.get("health", {})

    def utm_forward_only(self, features):
        resp = self._send({"cmd": "utm_forward_only",
                           "features": _to_list(features)})
        return resp.get("result")

    # -- Experience --

    def experience(self, modality, data, confidence=None):
        req = {"cmd": "experience", "modality": modality, "data": data}
        if confidence is not None:
            req["confidence"] = confidence
        self._send(req)

    # -- Sensory cortex --

    def audio_cortex_process(self, samples):
        resp = self._send({"cmd": "audio_cortex_process",
                           "samples": _to_list(samples)})
        return resp.get("result")

    def visual_cortex_process(self, pixels, width, height, channels=3):
        resp = self._send({"cmd": "visual_cortex_process",
                           "pixels": _to_list(pixels),
                           "width": width, "height": height,
                           "channels": channels})
        return resp.get("result")

    def speech_cortex_process(self, samples):
        resp = self._send({"cmd": "speech_cortex_process",
                           "samples": _to_list(samples)})
        return resp.get("result")

    # -- Sensory input --

    def submit_sensory(self, modality, data, **kwargs):
        req = {"cmd": "submit_sensory", "modality": modality,
               "data": _to_list(data)}
        req.update(kwargs)
        self._send(req)

    # -- Arousal control --

    def medulla_boost_arousal(self, amount=0.1):
        self._send({"cmd": "medulla_boost_arousal", "amount": amount})

    def medulla_reduce_arousal(self, amount=0.1):
        self._send({"cmd": "medulla_reduce_arousal", "amount": amount})

    # -- Reward / novelty --

    def edp_process_reward(self, reward):
        self._send({"cmd": "edp_process_reward", "reward": reward})

    def edp_process_novelty(self, novelty):
        self._send({"cmd": "edp_process_novelty", "novelty": novelty})

    # -- Language / cognitive training --

    def train_cognitive(self, **kwargs):
        req = {"cmd": "train_cognitive"}
        req.update(kwargs)
        resp = self._send(req)
        return resp.get("result")

    def train_language(self, text, target_text=None):
        self._send({"cmd": "train_language", "text": text,
                     "target_text": target_text or text})

    def learn_language(self, text):
        self._send({"cmd": "learn_language", "text": text})

    # -- Reasoning --

    def ti_init_reasoning(self):
        self._send({"cmd": "ti_init_reasoning"})

    def ti_add_fact(self, fact, confidence=0.5):
        self._send({"cmd": "ti_add_fact", "fact": fact,
                     "confidence": confidence})

    def ti_add_rule(self, rule, confidence=0.5):
        self._send({"cmd": "ti_add_rule", "rule": rule,
                     "confidence": confidence})

    def ti_forward_chain(self):
        resp = self._send({"cmd": "ti_forward_chain"})
        return resp.get("result")

    # -- Brain config --

    def enable_multi_network(self):
        self._send({"cmd": "enable_multi_network"})

    def init_cortex_cnns(self):
        self._send({"cmd": "init_cortex_cnns"})

    def enable_world_model(self, enabled=True):
        self._send({"cmd": "enable_world_model", "enabled": enabled})

    def enable_mixed_precision(self, enabled=True):
        self._send({"cmd": "enable_mixed_precision", "enabled": enabled})

    def enable_gradient_checkpointing(self, enabled=True, interval=None):
        req = {"cmd": "enable_gradient_checkpointing", "enabled": enabled}
        if interval is not None:
            req["interval"] = interval
        self._send(req)

    # -- LNN / SNN / CNN --

    def lnn_create(self, *args, **kwargs):
        req = {"cmd": "lnn_create"}
        if args:
            req["args"] = list(args)
        req.update(kwargs)
        self._send(req)

    def lnn_get_stats(self):
        resp = self._send({"cmd": "lnn_get_stats"})
        return resp.get("stats", {})

    def snn_get_stats(self):
        resp = self._send({"cmd": "snn_get_stats"})
        return resp.get("stats", {})

    def cnn_get_stats(self):
        resp = self._send({"cmd": "cnn_get_stats"})
        return resp.get("stats", {})

    # -- Plasticity / pruning --

    def get_plasticity_stats(self):
        resp = self._send({"cmd": "get_plasticity_stats"})
        return resp.get("stats", {})

    def prune_synapses(self, threshold=0.01):
        self._send({"cmd": "prune_synapses", "threshold": threshold})

    # -- Curiosity --

    def curiosity_detect_gaps(self, domain=None):
        req = {"cmd": "curiosity_detect_gaps"}
        if domain is not None:
            req["domain"] = domain
        resp = self._send(req)
        return resp.get("result")

    # -- UTM EMA --

    def utm_swap_to_ema(self):
        self._send({"cmd": "utm_swap_to_ema"})

    def utm_swap_from_ema(self):
        self._send({"cmd": "utm_swap_from_ema"})

    # -- Language / interactive --

    def comprehend(self, text):
        resp = self._send({"cmd": "comprehend", "text": text})
        return resp.get("result", {})

    def generate(self, prompt=None, semantic_input=None):
        req = {"cmd": "generate"}
        if prompt is not None:
            req["prompt"] = prompt
        if semantic_input is not None:
            req["semantic_input"] = _to_list(semantic_input)
        resp = self._send(req)
        return resp.get("result", {})

    def produce_text(self, intent):
        resp = self._send({"cmd": "produce_text", "intent": _to_list(intent)})
        return resp.get("result", {})

    def deliberate(self, topic):
        resp = self._send({"cmd": "deliberate", "topic": topic})
        return resp.get("result", {})

    def self_assess(self, domain):
        resp = self._send({"cmd": "self_assess", "domain": domain})
        return resp.get("result", {})

    def curiosity_detect_gaps(self, topic):
        resp = self._send({"cmd": "curiosity_detect_gaps", "topic": topic})
        return resp.get("result", {})

    def rubric(self):
        resp = self._send({"cmd": "rubric"})
        return resp.get("result", {})

    def get_last_gradient_norm(self):
        resp = self._send({"cmd": "get_last_gradient_norm"})
        return resp.get("result", 0.0)

    def focus_attention(self, modality):
        self._send({"cmd": "focus_attention", "modality": modality})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(obj):
    """Convert numpy arrays, bytes, and other iterables to plain lists."""
    if obj is None:
        return None
    if isinstance(obj, (bytes, bytearray)):
        return list(obj)  # bytes → list of ints [0-255]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return obj


def _json_default(obj):
    """Handle numpy types."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)
