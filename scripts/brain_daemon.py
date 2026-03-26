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

# Add scripts/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("brain_daemon")

SOCKET_PATH = "/var/run/athena/brain.sock"
SOCKET_LOCK = "/var/run/athena/brain.sock.lock"
PID_FILE = "/var/run/athena/brain.pid"
MAX_MSG_SIZE = 50 * 1024 * 1024  # 50 MB (batch learning can be large)

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

class BrainService:
    """Thread-safe wrapper around nimcp.Brain that dispatches IPC commands."""

    def __init__(self, brain):
        self.brain = brain
        self._lock = threading.Lock()
        self._stats = {
            "started_at": time.time(),
            "total_requests": 0,
            "learn_calls": 0,
            "infer_calls": 0,
            "errors": 0,
        }
        # Memory-augmented prompt assembly (optional)
        self._prompt_assembler = None
        try:
            from athena_prompt_assembly import AthenaPromptAssembler
            self._prompt_assembler = AthenaPromptAssembler(brain)
            logger.info("Prompt assembler initialized")
        except ImportError:
            pass

    def handle(self, req):
        """Dispatch a command dict and return a response dict."""
        cmd = req.get("cmd", "")
        self._stats["total_requests"] += 1

        try:
            handler = getattr(self, f"_cmd_{cmd}", None)
            if handler is None:
                return {"error": f"Unknown command: {cmd}"}
            return handler(req)
        except Exception as e:
            self._stats["errors"] += 1
            logger.warning("Command %s failed: %s", cmd, e)
            return {"error": str(e), "traceback": traceback.format_exc()}

    # -- Core learning --

    def _cmd_batch(self, req):
        """Execute multiple commands in a single round-trip."""
        commands = req.get("commands", [])
        results = []
        for cmd_req in commands:
            try:
                handler = getattr(self, f"_cmd_{cmd_req.get('cmd', '')}", None)
                if handler:
                    results.append(handler(cmd_req))
                else:
                    results.append({"error": f"Unknown command: {cmd_req.get('cmd')}"})
            except Exception as e:
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

        with self._lock:
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
        with self._lock:
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
        with self._lock:
            avg_loss = self.brain.learn_vector_batch(pair_tuples, **kwargs)
        self._stats["learn_calls"] += 1
        if hasattr(self, 'checkpointer') and self.checkpointer:
            self.checkpointer.notify_training_step()
        return {"avg_loss": avg_loss}

    # -- Inference --

    def _cmd_decide_full(self, req):
        features = req["features"]
        with self._lock:
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
        with self._lock:
            result = self.brain.predict(features)
        self._stats["infer_calls"] += 1
        return {"result": result}

    def _cmd_speak(self, req):
        output_vector = req["output_vector"]
        with self._lock:
            result = self.brain.speak(output_vector)
        return {"result": result}

    def _cmd_generate_text(self, req):
        output_vector = req["output_vector"]
        with self._lock:
            result = self.brain.generate_text(output_vector)
        return {"result": result}

    def _cmd_grounded_respond(self, req):
        text = req["text"]
        with self._lock:
            result = self.brain.grounded_respond(text)
        return {"result": result}

    # -- LNN --

    def _cmd_lnn_forward_step(self, req):
        features = req["features"]
        with self._lock:
            result = self.brain.lnn_forward_step(features)
        return {"result": result}

    def _cmd_lnn_get_state(self, _req):
        with self._lock:
            state = self.brain.lnn_get_state()
        return {"state": state}

    # -- Checkpoint --

    def _cmd_save(self, req):
        path = req["path"]
        with self._lock:
            self.brain.save(path)
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
        with self._lock:
            try:
                stats = self.brain.get_stats()
            except Exception:
                stats = {}
        return {"stats": stats, **self._stats}

    def _cmd_get_accuracy(self, _req):
        with self._lock:
            return {"accuracy": self.brain.get_accuracy()}

    def _cmd_get_last_gradient_norm(self, _req):
        with self._lock:
            return {"gradient_norm": self.brain.get_last_gradient_norm()}

    def _cmd_get_neuron_count(self, _req):
        with self._lock:
            return {"neuron_count": self.brain.get_neuron_count()}

    def _cmd_retrofit_synapse_metadata(self, _req):
        with self._lock:
            count = self.brain.retrofit_synapse_metadata()
            return {"retrofitted": count}

    def _cmd_get_snn_stats(self, _req):
        with self._lock:
            stats = self.brain.get_snn_stats()
            return {"snn": stats if stats else {}}

    def _cmd_get_transcript(self, _req):
        with self._lock:
            return {"transcript": self.brain.get_transcript()}

    def _cmd_get_cognitive_stats(self, _req):
        with self._lock:
            return {"stats": self.brain.get_cognitive_stats()}

    def _cmd_probe(self, _req):
        with self._lock:
            return {"probe": self.brain.probe()}

    def _cmd_get_network_metrics(self, _req):
        with self._lock:
            m = self.brain.get_network_metrics()
            return {"metrics": m if m else {}}

    def _cmd_get_cortex_cnn_metrics(self, _req):
        with self._lock:
            m = self.brain.get_cortex_cnn_metrics()
            return {"metrics": m if m else {}}

    # -- Biological state --

    def _cmd_substrate_get_health(self, _req):
        with self._lock:
            return {"health": self.brain.substrate_get_health()}

    def _cmd_substrate_get_metabolic(self, _req):
        with self._lock:
            return {"metabolic": self.brain.substrate_get_metabolic()}

    def _cmd_medulla_get_arousal(self, _req):
        with self._lock:
            return {"arousal": self.brain.medulla_get_arousal()}

    def _cmd_medulla_get_circadian_efficiency(self, _req):
        with self._lock:
            return {"efficiency": self.brain.medulla_get_circadian_efficiency()}

    def _cmd_sleep_get_pressure(self, _req):
        with self._lock:
            return {"pressure": self.brain.sleep_get_pressure()}

    def _cmd_sleep_get_state(self, _req):
        with self._lock:
            return {"state": self.brain.sleep_get_state()}

    def _cmd_sleep_is_needed(self, _req):
        with self._lock:
            return {"needed": self.brain.sleep_is_needed()}

    def _cmd_sleep_run_cycle(self, req):
        duration = req.get("duration", 2)
        with self._lock:
            self.brain.sleep_run_cycle(duration)
        return {"ok": True}

    def _cmd_update_medulla(self, req):
        dt = req.get("dt", 0.1)
        with self._lock:
            self.brain.update_medulla(dt)
        return {"ok": True}

    def _cmd_bg_get_dopamine(self, _req):
        with self._lock:
            return {"dopamine": self.brain.bg_get_dopamine()}

    def _cmd_bg_get_rpe(self, _req):
        with self._lock:
            return {"rpe": self.brain.bg_get_rpe()}

    def _cmd_bg_get_conflict(self, _req):
        with self._lock:
            return {"conflict": self.brain.bg_get_conflict()}

    def _cmd_bg_get_mode(self, _req):
        with self._lock:
            return {"mode": self.brain.bg_get_mode()}

    def _cmd_bg_update_reward(self, req):
        reward = req["reward"]
        rpe = req.get("rpe", 0.0)
        with self._lock:
            self.brain.bg_update_reward(reward, rpe)
        return {"ok": True}

    # -- Training config --

    def _cmd_set_plasticity_state(self, req):
        with self._lock:
            self.brain.set_plasticity_state(req["state"])
        return {"ok": True}

    def _cmd_set_task_type(self, req):
        with self._lock:
            self.brain.set_task_type(req["task_type"])
        return {"ok": True}

    def _cmd_set_fast_training(self, req):
        with self._lock:
            self.brain.set_fast_training(req["enabled"])
        return {"ok": True}

    def _cmd_reinit_weights(self, req):
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self.brain.enable_biological_plasticity(req["enabled"])
        return {"ok": True}

    def _cmd_consolidate(self, req):
        mode = req.get("mode", "auto")
        with self._lock:
            self.brain.consolidate(mode)
        return {"ok": True}

    def _cmd_cerebellum_process_error(self, req):
        with self._lock:
            self.brain.cerebellum_process_error(req["error"])
        return {"ok": True}

    # -- UTM --

    def _cmd_utm_get_training_health(self, _req):
        with self._lock:
            return {"health": self.brain.utm_get_training_health()}

    def _cmd_utm_forward_only(self, req):
        with self._lock:
            result = self.brain.utm_forward_only(req["features"])
        return {"result": result}

    # -- Experience --

    def _cmd_experience(self, req):
        with self._lock:
            self.brain.experience(req["modality"], req["data"],
                                  confidence=req.get("confidence"))
        return {"ok": True}

    # -- Sensory cortex --

    def _cmd_audio_cortex_process(self, req):
        with self._lock:
            result = self.brain.audio_cortex_process(req["samples"])
        return {"result": result}

    def _cmd_visual_cortex_process(self, req):
        with self._lock:
            result = self.brain.visual_cortex_process(
                req["pixels"], req["width"], req["height"],
                req.get("channels", 3))
        return {"result": result}

    def _cmd_speech_cortex_process(self, req):
        with self._lock:
            result = self.brain.speech_cortex_process(req["samples"])
        return {"result": result}

    # -- Sensory input --

    def _cmd_submit_sensory(self, req):
        modality = req["modality"]
        data = req["data"]
        with self._lock:
            # Stage sensory data on brain struct so cortex CNNs get created
            # and trained during the next learn_vector call.
            if modality == "visual":
                self.brain.submit_sensory("visual", data,
                                          width=req.get("width", 32),
                                          height=req.get("height", 32),
                                          channels=req.get("channels", 3))
            elif modality == "audio":
                self.brain.submit_sensory("audio", data)
            elif modality == "speech":
                self.brain.submit_sensory("speech", data)
            elif modality == "somatosensory" or modality == "somato":
                self.brain.submit_sensory("somatosensory", data,
                                          n_segments=req.get("n_segments", len(data)))
            else:
                return {"ok": True}
        return {"ok": True}

    # -- Arousal control --

    def _cmd_medulla_boost_arousal(self, req):
        with self._lock:
            self.brain.medulla_boost_arousal(req.get("amount", 0.1))
        return {"ok": True}

    def _cmd_medulla_reduce_arousal(self, req):
        with self._lock:
            self.brain.medulla_reduce_arousal(req.get("amount", 0.1))
        return {"ok": True}

    # -- Reward / novelty --

    def _cmd_edp_process_reward(self, req):
        with self._lock:
            self.brain.edp_process_reward(req["reward"])
        return {"ok": True}

    def _cmd_edp_process_novelty(self, req):
        with self._lock:
            self.brain.edp_process_novelty(req["novelty"])
        return {"ok": True}

    # -- Language / cognitive training --

    def _cmd_train_cognitive(self, req):
        kwargs = {}
        for k in ("text", "target_text", "learning_rate", "domain"):
            if k in req:
                kwargs[k] = req[k]
        with self._lock:
            result = self.brain.train_cognitive(**kwargs)
        return {"result": result}

    def _cmd_train_language(self, req):
        with self._lock:
            self.brain.train_language(req["text"], req.get("target_text", req["text"]))
        return {"ok": True}

    def _cmd_learn_language(self, req):
        with self._lock:
            self.brain.learn_language(req["text"])
        return {"ok": True}

    # -- Reasoning --

    def _cmd_ti_init_reasoning(self, _req):
        with self._lock:
            self.brain.ti_init_reasoning()
        return {"ok": True}

    def _cmd_ti_add_fact(self, req):
        with self._lock:
            self.brain.ti_add_fact(req["fact"], req.get("confidence", 0.5))
        return {"ok": True}

    def _cmd_ti_add_rule(self, req):
        with self._lock:
            self.brain.ti_add_rule(req["rule"], req.get("confidence", 0.5))
        return {"ok": True}

    def _cmd_ti_forward_chain(self, _req):
        with self._lock:
            result = self.brain.ti_forward_chain()
        return {"result": result}

    # -- Brain config --

    def _cmd_enable_multi_network(self, _req):
        with self._lock:
            self.brain.enable_multi_network()
        return {"ok": True}

    def _cmd_init_cortex_cnns(self, _req):
        with self._lock:
            self.brain.init_cortex_cnns()
        return {"ok": True}

    def _cmd_enable_world_model(self, req):
        with self._lock:
            self.brain.enable_world_model(req.get("enabled", True))
        return {"ok": True}

    def _cmd_enable_mixed_precision(self, req):
        with self._lock:
            self.brain.enable_mixed_precision(req.get("enabled", True))
        return {"ok": True}

    def _cmd_enable_gradient_checkpointing(self, req):
        args = [req.get("enabled", True)]
        if "interval" in req:
            args.append(req["interval"])
        with self._lock:
            self.brain.enable_gradient_checkpointing(*args)
        return {"ok": True}

    # -- LNN / SNN / CNN --

    def _cmd_lnn_create(self, req):
        args = req.get("args", [])
        kwargs = {k: v for k, v in req.items()
                  if k not in ("cmd", "args")}
        with self._lock:
            self.brain.lnn_create(*args, **kwargs)
        return {"ok": True}

    def _cmd_lnn_get_stats(self, _req):
        with self._lock:
            return {"stats": self.brain.lnn_get_stats()}

    def _cmd_snn_get_stats(self, _req):
        with self._lock:
            return {"stats": self.brain.snn_get_stats()}

    def _cmd_cnn_get_stats(self, _req):
        with self._lock:
            return {"stats": self.brain.cnn_get_stats()}

    # -- Plasticity / pruning --

    def _cmd_get_plasticity_stats(self, _req):
        with self._lock:
            return {"stats": self.brain.get_plasticity_stats()}

    def _cmd_prune_synapses(self, req):
        with self._lock:
            self.brain.prune_synapses(req.get("threshold", 0.01))
        return {"ok": True}

    # -- Curiosity --

    def _cmd_curiosity_detect_gaps(self, req):
        topic = req.get("topic") or req.get("domain", "general")
        with self._lock:
            result = self.brain.curiosity_detect_gaps(topic)
        return {"result": result}

    # -- UTM EMA --

    def _cmd_utm_swap_to_ema(self, _req):
        with self._lock:
            self.brain.utm_swap_to_ema()
        return {"ok": True}

    def _cmd_utm_swap_from_ema(self, _req):
        with self._lock:
            self.brain.utm_swap_from_ema()
        return {"ok": True}

    # -- Language / Interactive --

    def _cmd_comprehend(self, req):
        with self._lock:
            result = self.brain.comprehend(req["text"])
        return {"result": result}

    def _cmd_generate(self, req):
        with self._lock:
            result = self.brain.generate(
                prompt=req.get("prompt"),
                semantic_input=req.get("semantic_input"))
        return {"result": result}

    def _cmd_produce_text(self, req):
        with self._lock:
            result = self.brain.produce_text(req["intent"])
        return {"result": result}

    def _cmd_deliberate(self, req):
        with self._lock:
            result = self.brain.deliberate(req["topic"])
        return {"result": result}

    def _cmd_self_assess(self, req):
        with self._lock:
            result = self.brain.self_assess(req["domain"])
        return {"result": result}

    def _cmd_rubric(self, _req):
        with self._lock:
            result = self.brain.rubric()
        return {"result": result}

    def _cmd_get_last_gradient_norm(self, _req):
        with self._lock:
            result = self.brain.get_last_gradient_norm()
        return {"result": result}

    def _cmd_focus_attention(self, req):
        with self._lock:
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
        with self._lock:
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
    """Unix socket server that accepts IPC connections."""

    def __init__(self, service, socket_path=SOCKET_PATH, max_workers=4):
        self.service = service
        self.socket_path = socket_path
        self.lock_path = socket_path + ".lock"
        self.max_workers = max_workers
        self._server_sock = None
        self._lock_fd = None
        self._running = False
        self._worker_semaphore = threading.Semaphore(max_workers)

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
        """Start the daemon server."""
        # Acquire exclusive lock before touching the socket
        self._acquire_socket_lock()

        # Clean up stale socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        os.chmod(self.socket_path, 0o660)
        self._server_sock.listen(16)
        self._server_sock.settimeout(1.0)
        self._running = True

        logger.info("Brain daemon listening on %s (max_workers=%d)",
                     self.socket_path, self.max_workers)

    def serve_forever(self):
        """Main accept loop."""
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.error("Socket accept error")
                break

            # Handle in a worker thread
            self._worker_semaphore.acquire()
            t = threading.Thread(target=self._handle_conn, args=(conn,),
                                 daemon=True)
            t.start()

    # Heartbeat constants
    HEARTBEAT_WARN_SECONDS = 60
    HEARTBEAT_DEAD_SECONDS = 300

    def _handle_conn(self, conn):
        """Handle a single client connection (may have multiple requests).

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
            self._worker_semaphore.release()

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
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
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
        self._loaded_from_checkpoint = False

    def set_loaded_from_checkpoint(self, loaded):
        """Mark whether this brain was loaded from a checkpoint.
        If loaded, allow immediate saves (the brain already has trained state).
        If fresh, require min_steps training before first save."""
        self._loaded_from_checkpoint = loaded
        if loaded:
            self._save_count = 1  # Allow saves immediately

    def start(self):
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

        Won't save a fresh untrained brain unless force=True.
        Rotates previous checkpoint to .bak before overwriting.
        """
        if not force and not self._loaded_from_checkpoint:
            if self._save_count < self.min_steps:
                logger.debug("Checkpoint skipped: only %d steps (need %d)",
                             self._save_count, self.min_steps)
                return

        path = os.path.join(self.checkpoint_dir, "athena_daemon.bin")
        bak_path = path + ".bak"

        try:
            with self._lock:
                # Rotate: current → .bak (so we always have a fallback)
                if os.path.exists(path):
                    try:
                        # Copy instead of rename so .bak is a full independent copy
                        import shutil
                        shutil.copy2(path, bak_path)
                    except Exception as e:
                        logger.warning("Backup rotation failed: %s", e)

                self.brain.save(path)

            logger.info("Checkpoint saved: %s (steps=%d)", path, self._save_count)
        except Exception as e:
            logger.error("Checkpoint failed: %s — .bak preserved at %s", e, bak_path)

    def _run(self):
        while self._running:
            time.sleep(self.interval)
            if self._running:
                self.save_now()


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
    parser.add_argument("--neuron-count", type=int, default=2_000_000,
                        help="Neuron count (default: 2000000)")
    parser.add_argument("--num-inputs", type=int, default=1024,
                        help="Input dimension (default: 1024)")
    parser.add_argument("--num-outputs", type=int, default=2048,
                        help="Output dimension (default: 2048)")
    parser.add_argument("--socket", type=str, default=SOCKET_PATH,
                        help=f"Unix socket path (default: {SOCKET_PATH})")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max concurrent worker threads (default: 4)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/home/bbrelin/nimcp/checkpoints/athena",
                        help="Auto-checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=300,
                        help="Auto-checkpoint interval seconds (default: 300)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Log file path (default: stderr)")
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

    # Kill stale daemon processes before starting.
    # First check PID file, then pgrep as fallback.
    # Wait up to 10 seconds for graceful shutdown (checkpoint may be in progress).
    my_pid = os.getpid()
    stale_pids = set()

    # Check PID file first (most reliable)
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                old_pid = int(f.read().strip())
            if old_pid != my_pid:
                stale_pids.add(old_pid)
        except (ValueError, OSError):
            pass

    # Also check pgrep (catches processes without PID file)
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

    # Import nimcp
    import nimcp
    nimcp.init()

    # Determine checkpoint path
    checkpoint_path = args.checkpoint
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
        logger.info("Loading brain from checkpoint: %s", checkpoint_path)
        try:
            brain = nimcp.Brain("athena", checkpoint=str(checkpoint_path),
                                init_mode=args.init_mode)
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            # Try .bak fallback
            bak_path = checkpoint_path + ".bak"
            if os.path.exists(bak_path):
                logger.info("Trying backup checkpoint: %s", bak_path)
                try:
                    brain = nimcp.Brain("athena", checkpoint=str(bak_path),
                                        init_mode=args.init_mode)
                    checkpoint_path = bak_path  # So loaded_from_ckpt is correct
                except Exception as e2:
                    logger.error("Backup checkpoint also failed: %s", e2)

    if brain is None:
        logger.info("Creating new brain: %d neurons, mode=%s",
                     args.neuron_count, args.init_mode)
        brain = nimcp.Brain("athena",
                            num_inputs=args.num_inputs,
                            num_outputs=args.num_outputs,
                            neuron_count=args.neuron_count,
                            init_mode=args.init_mode)
    elapsed = time.time() - t0
    logger.info("Brain ready in %.1f seconds", elapsed)

    # CRITICAL: Set regression mode — no softmax on outputs.
    # Without this, classification mode applies softmax to 4096 outputs,
    # which exponentially suppresses all but ~200 top neurons → mode collapse.
    brain.set_task_type("regression")
    logger.info("Task type set to REGRESSION (no softmax)")

    # Fix output layer activation — checkpoints from older code have TANH,
    # which bounds output to [-1,1] and causes gradient vanishing for regression.
    brain.fix_output_activation()
    logger.info("Output layer activation set to LINEAR")

    # Create service and daemon
    service = BrainService(brain)
    daemon = BrainDaemon(service, socket_path=args.socket,
                          max_workers=args.workers)

    # Auto-checkpoint with safety guards
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    loaded_from_ckpt = (checkpoint_path is not None and os.path.exists(checkpoint_path))
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

    # Start
    daemon.start()
    checkpointer.start()

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

    logger.info("Brain daemon ready — accepting connections on %s", args.socket)
    print(f"Brain daemon ready on {args.socket} (PID {os.getpid()})",
          flush=True)

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

    # Wait for any in-flight requests to finish (workers hold the brain lock)
    logger.info("Shutdown: waiting for in-flight requests to complete...")
    for _ in range(args.workers):
        # Acquire all worker permits = all workers have finished
        if not daemon._worker_semaphore.acquire(timeout=30):
            logger.warning("Shutdown: timed out waiting for a worker thread")
            break
    # Release them back (cleanup)
    for _ in range(args.workers):
        try:
            daemon._worker_semaphore.release()
        except ValueError:
            break

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
