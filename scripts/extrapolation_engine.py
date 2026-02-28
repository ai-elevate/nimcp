#!/usr/bin/env python3
"""
NIMCP Generative Extrapolation Engine
=======================================

WHAT: Generate novel content by extrapolating from trained knowledge using
      the FULL NIMCP cognitive stack — not just forward propagation.

WHY:  A brain that can only classify is half a brain. Real intelligence
      requires the ability to create, imagine, and extrapolate beyond
      training data. This engine uses every available NIMCP module to
      produce genuinely novel outputs grounded in learned knowledge.

HOW:  Multi-stage cognitive generation pipeline:
      1. SEED     — Encode input prompt/context into feature space
      2. RECALL   — Retrieve relevant knowledge (engrams, working memory)
      3. REASON   — Forward/backward chain + causal inference on seed
      4. IMAGINE  — Explore latent space via temperature-controlled sampling
      5. EVALUATE — Self-assess, curiosity-check, uncertainty-quantify
      6. REFINE   — Mesh network consensus + global workspace broadcast
      7. OUTPUT   — Decode refined representation into structured result

MODULES USED:
  Core:           decide_full, infer, predict_in_domain
  Memory:         working_memory_*, engram recall (via decide)
  Reasoning:      ti_forward_chain, ti_backward_chain, ti_reason, ti_add_fact
  Workspace:      workspace_compete, workspace_read, workspace_subscribe
  Neuromodulators: release_dopamine, bg_update_reward, medulla_boost_arousal
  Introspection:  self_assess, get_uncertainty, curiosity_detect_gaps
  Oscillations:   get_phase_coherence, get_pac_modulation
  Mesh Network:   ti_mesh_*, multi-brain consensus
  Sensory:        audio_cortex_process, visual_cortex_process, speech_cortex_process
  Safety:         lgss_check_content
  Consolidation:  consolidate (post-generation memory integration)
  Basal Ganglia:  bg_get_conflict, bg_get_dopamine, bg_get_rpe
  Medulla:        medulla_get_arousal, medulla_get_circadian_efficiency
"""

import math
import random
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class GenerationMode(Enum):
    """How to generate content."""
    EXTRAPOLATE = "extrapolate"      # Extend learned patterns
    INTERPOLATE = "interpolate"      # Blend between domains
    ANALOGIZE   = "analogize"        # Map structure from one domain to another
    SYNTHESIZE  = "synthesize"       # Combine multiple knowledge sources
    HYPOTHESIZE = "hypothesize"      # Reason about unseen scenarios


@dataclass
class ExtrapolationConfig:
    """Configuration for the extrapolation engine."""
    temperature: float = 0.7         # Output diversity (0.1=focused, 1.0=creative)
    top_k: int = 10                  # Top-k candidates to consider
    num_candidates: int = 5          # Number of candidates to generate
    max_reasoning_steps: int = 20    # Max forward/backward chaining iterations
    mesh_consensus_timeout: float = 2.0  # Seconds to wait for mesh consensus
    arousal_boost: float = 0.3       # Boost arousal during generation (creativity)
    dopamine_reward: float = 0.5     # Dopamine for novel discovery
    curiosity_weight: float = 0.4    # How much curiosity drives exploration
    confidence_threshold: float = 0.3  # Min confidence for output acceptance
    safety_check: bool = True        # Run LGSS safety filter on output
    use_mesh: bool = True            # Use mesh network for consensus
    use_oscillations: bool = True    # Use phase coherence for binding
    consolidate_after: bool = False  # Consolidate after generation
    add_to_kb: bool = False          # Add prompt as fact to knowledge base
    boost_arousal: bool = True       # Boost arousal during generation
    domain_context: str = ""         # Primary domain context
    cross_domains: List[str] = field(default_factory=list)  # Secondary domains


@dataclass
class GenerationResult:
    """Result of a generation/extrapolation operation."""
    content: Dict[str, Any]          # Generated output
    confidence: float                # Overall confidence [0, 1]
    novelty: float                   # How novel vs. memorized [0, 1]
    coherence: float                 # Internal consistency [0, 1]
    reasoning_trace: List[str]       # Step-by-step reasoning
    knowledge_sources: List[str]     # Domains/facts that contributed
    uncertainty: Dict[str, float]    # Per-component uncertainty
    generation_time_ms: float        # Wall-clock time
    mode: str                        # Generation mode used
    cognitive_state: Dict[str, Any]  # Brain state during generation

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to a plain dictionary."""
        return {
            "content": self.content,
            "confidence": self.confidence,
            "novelty": self.novelty,
            "coherence": self.coherence,
            "reasoning_trace": self.reasoning_trace,
            "knowledge_sources": self.knowledge_sources,
            "uncertainty": self.uncertainty,
            "generation_time_ms": self.generation_time_ms,
            "mode": self.mode,
            "cognitive_state": self.cognitive_state,
        }


# ---------------------------------------------------------------------------
# Extrapolation Engine
# ---------------------------------------------------------------------------

class ExtrapolationEngine:
    """
    Generative extrapolation engine using the full NIMCP cognitive stack.

    Usage:
        engine = ExtrapolationEngine(brain)
        result = engine.generate(
            prompt="What would happen if photosynthesis worked with infrared light?",
            mode=GenerationMode.HYPOTHESIZE,
            config=ExtrapolationConfig(temperature=0.8, domain_context="biology")
        )
    """

    def __init__(self, brain, feature_encoder=None):
        """
        Args:
            brain: NIMCP Brain instance (or ThreadSafeBrain wrapper)
            feature_encoder: Optional callable(text, dim) -> list[float].
                             Defaults to instructor_agent._text_to_features.
        """
        self.brain = brain
        self._lock = threading.RLock()

        # Feature encoder
        if feature_encoder is not None:
            self._encode = feature_encoder
        else:
            try:
                from instructor_agent import _text_to_features
                self._encode = _text_to_features
            except ImportError:
                self._encode = self._fallback_encode

        # Detect input dimensions from brain
        try:
            probe = self.brain.probe()
            self._num_inputs = probe.get("num_inputs", 1024)
            self._num_outputs = probe.get("num_outputs", 256)
        except Exception:
            self._num_inputs = 1024
            self._num_outputs = 256

        # Generation history for novelty tracking
        self._generation_history: List[np.ndarray] = []
        self._max_history = 100

        # Domain knowledge cache
        self._domain_knowledge: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, mode: GenerationMode = GenerationMode.EXTRAPOLATE,
                 config: Optional[ExtrapolationConfig] = None) -> GenerationResult:
        """Generate novel content from a prompt using the full cognitive stack.

        Args:
            prompt: Natural language prompt or structured query
            mode: Generation mode (extrapolate, interpolate, analogize, etc.)
            config: Generation configuration

        Returns:
            GenerationResult with content, confidence, reasoning trace, etc.
        """
        if config is None:
            config = ExtrapolationConfig()

        start_time = time.time()
        reasoning_trace = []
        knowledge_sources = []

        # NOTE: The lock is held for the entire 7-stage pipeline to ensure
        # generation history consistency.  Candidates produced by _stage_imagine
        # are scored against _generation_history in _compute_novelty; releasing
        # the lock between stages would allow a concurrent generate() call to
        # interleave history entries and corrupt novelty scores.  This means
        # generation is serialized per-engine — callers needing parallelism
        # should use separate ExtrapolationEngine instances.
        with self._lock:
            # Stage 1: SEED — Encode prompt into feature space
            features = self._stage_seed(prompt, config)
            reasoning_trace.append(f"Encoded prompt to {len(features)}-dim feature vector")

            # Stage 2: RECALL — Retrieve relevant knowledge
            recalled = self._stage_recall(features, config, reasoning_trace)
            knowledge_sources.extend(recalled.get("sources", []))

            # Stage 3: REASON — Chain reasoning on the seed
            reasoned = self._stage_reason(
                features, prompt, mode, config, reasoning_trace)

            # Stage 4: IMAGINE — Generate candidates via temperature sampling
            candidates = self._stage_imagine(
                features, reasoned, mode, config, reasoning_trace)

            # Stage 5: EVALUATE — Score candidates using introspection
            scored = self._stage_evaluate(
                candidates, features, config, reasoning_trace)

            # Stage 6: REFINE — Mesh consensus + workspace broadcast
            refined = self._stage_refine(
                scored, config, reasoning_trace)

            # Stage 7: OUTPUT — Assemble final result
            result = self._stage_output(
                refined, features, mode, config, reasoning_trace,
                knowledge_sources, start_time)

            # Optional: consolidate generated knowledge
            if config.consolidate_after:
                self._post_generation_consolidate(result, config)

        return result

    def generate_in_domain(self, prompt: str, domain: str,
                           temperature: float = 0.7) -> GenerationResult:
        """Convenience: generate within a specific domain context."""
        config = ExtrapolationConfig(
            temperature=temperature,
            domain_context=domain,
        )
        return self.generate(prompt, GenerationMode.EXTRAPOLATE, config)

    def hypothesize(self, question: str, domains: List[str] = None,
                    temperature: float = 0.8) -> GenerationResult:
        """Generate a hypothesis about a question using cross-domain reasoning."""
        config = ExtrapolationConfig(
            temperature=temperature,
            domain_context=domains[0] if domains else "",
            cross_domains=domains[1:] if domains and len(domains) > 1 else [],
            max_reasoning_steps=30,
        )
        return self.generate(question, GenerationMode.HYPOTHESIZE, config)

    def blend_domains(self, domain_a: str, domain_b: str,
                      prompt: str = "", temperature: float = 0.6) -> GenerationResult:
        """Interpolate between two domains to find cross-domain insights."""
        config = ExtrapolationConfig(
            temperature=temperature,
            domain_context=domain_a,
            cross_domains=[domain_b],
            curiosity_weight=0.6,
        )
        return self.generate(
            prompt or f"What connects {domain_a} and {domain_b}?",
            GenerationMode.INTERPOLATE, config)

    # ------------------------------------------------------------------
    # Stage 1: SEED — Encode into feature space
    # ------------------------------------------------------------------

    def _stage_seed(self, prompt: str, config: ExtrapolationConfig) -> list:
        """Encode prompt into the brain's feature space."""
        features = self._encode(prompt, self._num_inputs)

        # Boost arousal for creative generation
        if config.boost_arousal:
            try:
                self.brain.medulla_boost_arousal(config.arousal_boost)
            except (AttributeError, Exception):
                pass

        # Prime working memory with the prompt context
        try:
            salience = 0.8 + 0.2 * config.temperature
            self.brain.working_memory_add(features[:64], salience)
        except (AttributeError, Exception):
            pass

        return features

    # ------------------------------------------------------------------
    # Stage 2: RECALL — Retrieve relevant knowledge
    # ------------------------------------------------------------------

    def _stage_recall(self, features: list, config: ExtrapolationConfig,
                      trace: list) -> Dict:
        """Use brain's memory systems to recall relevant knowledge."""
        sources = []
        recalled_outputs = []

        # 2a: Full cognitive decision (triggers engram recall, memory completion)
        try:
            decision = self.brain.decide_full(features)
            if decision:
                label = decision.get("label", "")
                conf = decision.get("confidence", 0.0)
                output_vec = decision.get("output_vector", [])
                explanation = decision.get("explanation", "")
                recalled_outputs.append({
                    "type": "cognitive_decision",
                    "label": label,
                    "confidence": conf,
                    "output_vector": output_vec,
                    "explanation": explanation,
                })
                if label:
                    sources.append(f"decision:{label}")
                trace.append(f"Cognitive recall: '{label}' (conf={conf:.2f}) — {explanation}")
        except (AttributeError, Exception) as e:
            trace.append(f"Cognitive recall failed: {e}")

        # 2b: Domain-scoped predictions for context domains
        domains = [config.domain_context] + config.cross_domains
        for domain in domains:
            if not domain:
                continue
            try:
                pred, conf = self.brain.predict_in_domain(features, f"{domain}:")
                if pred and conf > 0.1:
                    recalled_outputs.append({
                        "type": "domain_prediction",
                        "domain": domain,
                        "label": pred,
                        "confidence": conf,
                    })
                    sources.append(f"domain:{domain}:{pred}")
                    trace.append(f"Domain recall [{domain}]: '{pred}' (conf={conf:.2f})")
            except (AttributeError, Exception):
                pass

        # 2c: Query knowledge base
        try:
            facts_found = self.brain.ti_query_knowledge(
                config.domain_context or "general")
            if facts_found > 0:
                sources.append(f"kb:{config.domain_context}:{facts_found}_facts")
                trace.append(f"Knowledge base: {facts_found} relevant facts retrieved")
        except (AttributeError, Exception):
            pass

        # 2d: Read working memory
        try:
            wm_stats = self.brain.working_memory_stats()
            if wm_stats and wm_stats[0] > 0:
                trace.append(f"Working memory: {wm_stats[0]}/{wm_stats[1]} items active")
        except (AttributeError, Exception):
            pass

        # 2e: Curiosity — detect knowledge gaps
        try:
            gaps = self.brain.curiosity_detect_gaps(config.domain_context or "general")
            if gaps:
                gap_list = gaps.get("questions", [])
                if gap_list:
                    trace.append(f"Curiosity gaps: {len(gap_list)} gaps detected")
                    sources.append(f"curiosity:{len(gap_list)}_gaps")
        except (AttributeError, Exception):
            pass

        return {
            "outputs": recalled_outputs,
            "sources": sources,
        }

    # ------------------------------------------------------------------
    # Stage 3: REASON — Forward/backward chain reasoning
    # ------------------------------------------------------------------

    def _stage_reason(self, features: list, prompt: str,
                      mode: GenerationMode, config: ExtrapolationConfig,
                      trace: list) -> Dict:
        """Apply reasoning modules to the seed."""
        reasoning_results = {}

        # 3a: Optionally add prompt as a fact for reasoning engine
        if config.add_to_kb:
            try:
                self.brain.ti_add_fact(prompt[:200], 0.9)
                trace.append("Seeded reasoning engine with prompt")
            except (AttributeError, Exception):
                pass

        # 3b: Forward chaining — derive new facts from existing knowledge
        try:
            inferences = self.brain.ti_forward_chain(config.max_reasoning_steps)
            reasoning_results["forward_chain"] = inferences
            if inferences > 0:
                trace.append(f"Forward chaining: {inferences} new inferences derived")
        except (AttributeError, Exception):
            pass

        # 3c: Backward chaining — try to prove the prompt as a goal
        if mode in (GenerationMode.HYPOTHESIZE, GenerationMode.ANALOGIZE):
            try:
                goal_conf = self.brain.ti_backward_chain(prompt[:200])
                reasoning_results["backward_chain"] = goal_conf
                trace.append(f"Backward chaining: goal confidence={goal_conf:.3f}")
            except (AttributeError, Exception):
                pass

        # 3d: General reasoning query
        try:
            reason_conf = self.brain.ti_reason(prompt[:200])
            reasoning_results["reason_confidence"] = reason_conf
            trace.append(f"Reasoning query: confidence={reason_conf:.3f}")
        except (AttributeError, Exception):
            pass

        # 3e: Check cognitive capacity and urgency
        try:
            capacity = self.brain.ti_get_cognitive_capacity()
            reasoning_results["cognitive_capacity"] = capacity
            if capacity < 0.3:
                trace.append(f"WARNING: Low cognitive capacity ({capacity:.2f})")
        except (AttributeError, Exception):
            pass

        # 3f: Basal ganglia conflict detection (competing hypotheses)
        try:
            conflict = self.brain.bg_get_conflict()
            reasoning_results["bg_conflict"] = conflict
            if conflict > 0.5:
                trace.append(f"High conflict detected ({conflict:.2f}) — "
                             "competing hypotheses")
        except (AttributeError, Exception):
            pass

        return reasoning_results

    # ------------------------------------------------------------------
    # Stage 4: IMAGINE — Temperature-controlled candidate generation
    # ------------------------------------------------------------------

    def _stage_imagine(self, features: list, reasoned: Dict,
                       mode: GenerationMode, config: ExtrapolationConfig,
                       trace: list) -> List[Dict]:
        """Generate candidate outputs via latent space exploration."""
        candidates = []
        feat_arr = np.array(features, dtype=np.float32)

        # Base: raw network inference for output distribution
        try:
            raw_output = self.brain.infer(features, self._num_outputs)
            if raw_output:
                base_output = np.array(raw_output, dtype=np.float32)
            else:
                base_output = np.zeros(self._num_outputs, dtype=np.float32)
        except (AttributeError, Exception):
            base_output = np.zeros(self._num_outputs, dtype=np.float32)

        for i in range(config.num_candidates):
            # Temperature-scaled perturbation of input features
            if config.temperature > 0:
                noise = np.random.randn(len(features)).astype(np.float32)
                noise *= config.temperature * 0.1
                perturbed = (feat_arr + noise).tolist()
            else:
                perturbed = features

            # Mode-specific generation strategies
            if mode == GenerationMode.INTERPOLATE and config.cross_domains:
                candidate = self._imagine_interpolate(
                    perturbed, config, i, trace)
            elif mode == GenerationMode.ANALOGIZE:
                candidate = self._imagine_analogize(
                    perturbed, base_output, config, i, trace)
            elif mode == GenerationMode.HYPOTHESIZE:
                candidate = self._imagine_hypothesize(
                    perturbed, reasoned, config, i, trace)
            elif mode == GenerationMode.SYNTHESIZE:
                candidate = self._imagine_synthesize(
                    perturbed, base_output, config, i, trace)
            else:  # EXTRAPOLATE
                candidate = self._imagine_extrapolate(
                    perturbed, base_output, config, i, trace)

            if candidate:
                candidates.append(candidate)

        trace.append(f"Generated {len(candidates)} candidates (T={config.temperature})")

        # Release dopamine for novel discovery
        if candidates:
            try:
                avg_novelty = np.mean([c.get("novelty", 0.5) for c in candidates])
                self.brain.release_dopamine(
                    config.dopamine_reward * avg_novelty, 0.3)
            except (AttributeError, Exception):
                pass

        return candidates

    def _imagine_extrapolate(self, features: list, base_output: np.ndarray,
                             config: ExtrapolationConfig, idx: int,
                             trace: list) -> Optional[Dict]:
        """Extrapolate by extending the output distribution beyond observed patterns."""
        try:
            decision = self.brain.decide_full(features)
            if not decision:
                return None

            output_vec = decision.get("output_vector", [])
            if not output_vec:
                output_vec = base_output.tolist()

            output_arr = np.array(output_vec, dtype=np.float32)

            # Temperature-scaled softmax sampling
            scaled = self._temperature_softmax(output_arr, config.temperature)

            # Top-k selection
            top_indices = np.argsort(scaled)[-config.top_k:][::-1]
            top_values = scaled[top_indices]

            return {
                "type": "extrapolation",
                "label": decision.get("label", ""),
                "confidence": decision.get("confidence", 0.0),
                "explanation": decision.get("explanation", ""),
                "output_distribution": scaled.tolist(),
                "top_k_indices": top_indices.tolist(),
                "top_k_values": top_values.tolist(),
                "active_neurons": decision.get("num_active_neurons", 0),
                "sparsity": decision.get("sparsity", 0.0),
                "novelty": self._compute_novelty(output_arr),
                "candidate_idx": idx,
            }
        except Exception:
            return None

    def _imagine_interpolate(self, features: list,
                             config: ExtrapolationConfig, idx: int,
                             trace: list) -> Optional[Dict]:
        """Interpolate between domain centroids for cross-domain insights."""
        try:
            # Get primary domain prediction
            pred1, conf1 = self.brain.predict_in_domain(
                features, f"{config.domain_context}:")

            # Get cross-domain predictions
            cross_preds = []
            for cd in config.cross_domains:
                try:
                    pred, conf = self.brain.predict_in_domain(features, f"{cd}:")
                    cross_preds.append({"domain": cd, "label": pred, "confidence": conf})
                except Exception:
                    pass

            # Try to get domain centroid for true interpolation
            centroid = None
            if hasattr(self.brain, '_domain_centroids'):
                # Use brain's domain centroids if available
                for d in config.cross_domains:
                    c = getattr(self.brain, '_domain_centroids', {}).get(d)
                    if c is not None:
                        centroid = c
                        break

            if centroid is not None and len(centroid) == len(features):
                feat_arr = np.array(features, dtype=np.float32)
                cent_arr = np.array(centroid, dtype=np.float32)
                alpha = random.uniform(0.3, 0.7)
                blended = (feat_arr * alpha + cent_arr * (1 - alpha)).tolist()
            else:
                # Fallback: perturb features with noise
                feat_arr = np.array(features, dtype=np.float32)
                noise = np.random.randn(len(features)).astype(np.float32) * 0.05
                blended = (feat_arr + noise).tolist()

            blended_decision = self.brain.decide_full(blended)
            if not blended_decision:
                return None

            return {
                "type": "interpolation",
                "primary": {"domain": config.domain_context, "label": pred1, "confidence": conf1},
                "cross_domain": cross_preds,
                "blended_label": blended_decision.get("label", ""),
                "blended_confidence": blended_decision.get("confidence", 0.0),
                "blend_alpha": alpha,
                "explanation": blended_decision.get("explanation", ""),
                "novelty": self._compute_novelty(
                    np.array(blended_decision.get("output_vector",
                             [0.0] * self._num_outputs), dtype=np.float32)),
                "candidate_idx": idx,
            }
        except Exception:
            return None

    def _imagine_analogize(self, features: list, base_output: np.ndarray,
                           config: ExtrapolationConfig, idx: int,
                           trace: list) -> Optional[Dict]:
        """Map structural relationships from one domain to another."""
        try:
            # Get uncertainty — high uncertainty areas are where analogies help most
            uncertainty = self.brain.get_uncertainty(features)
            epistemic = uncertainty.get("epistemic", 0.5) if uncertainty else 0.5

            # Use reasoning to find structural mappings
            reason_conf = 0.0
            try:
                reason_conf = self.brain.ti_reason(
                    f"analogical mapping from {config.domain_context}")
            except Exception:
                pass

            decision = self.brain.decide_full(features)
            if not decision:
                return None

            return {
                "type": "analogy",
                "source_domain": config.domain_context,
                "target_domains": config.cross_domains,
                "label": decision.get("label", ""),
                "confidence": decision.get("confidence", 0.0),
                "epistemic_uncertainty": epistemic,
                "reasoning_confidence": reason_conf,
                "explanation": decision.get("explanation", ""),
                "novelty": self._compute_novelty(
                    np.array(decision.get("output_vector",
                             [0.0] * self._num_outputs), dtype=np.float32)),
                "candidate_idx": idx,
            }
        except Exception:
            return None

    def _imagine_hypothesize(self, features: list, reasoned: Dict,
                             config: ExtrapolationConfig, idx: int,
                             trace: list) -> Optional[Dict]:
        """Generate hypotheses using causal reasoning + reasoning engine."""
        try:
            decision = self.brain.decide_full(features)
            if not decision:
                return None

            # Combine decision confidence with reasoning confidence
            reason_conf = reasoned.get("reason_confidence", 0.0)
            backward_conf = reasoned.get("backward_chain", 0.0)
            forward_inferences = reasoned.get("forward_chain", 0)

            # Hypothesis strength: weighted combination
            hypothesis_strength = (
                0.4 * decision.get("confidence", 0.0) +
                0.3 * reason_conf +
                0.2 * backward_conf +
                0.1 * min(forward_inferences / 10.0, 1.0)
            )

            return {
                "type": "hypothesis",
                "label": decision.get("label", ""),
                "confidence": decision.get("confidence", 0.0),
                "hypothesis_strength": hypothesis_strength,
                "reasoning_confidence": reason_conf,
                "backward_chain_confidence": backward_conf,
                "forward_chain_inferences": forward_inferences,
                "bg_conflict": reasoned.get("bg_conflict", 0.0),
                "cognitive_capacity": reasoned.get("cognitive_capacity", 1.0),
                "explanation": decision.get("explanation", ""),
                "novelty": self._compute_novelty(
                    np.array(decision.get("output_vector",
                             [0.0] * self._num_outputs), dtype=np.float32)),
                "candidate_idx": idx,
            }
        except Exception:
            return None

    def _imagine_synthesize(self, features: list, base_output: np.ndarray,
                            config: ExtrapolationConfig, idx: int,
                            trace: list) -> Optional[Dict]:
        """Synthesize from multiple knowledge sources simultaneously."""
        try:
            # Collect predictions from all specified domains
            domain_predictions = {}
            all_domains = [config.domain_context] + config.cross_domains
            for domain in all_domains:
                if not domain:
                    continue
                try:
                    pred, conf = self.brain.predict_in_domain(features, f"{domain}:")
                    domain_predictions[domain] = {"label": pred, "confidence": conf}
                except Exception:
                    pass

            decision = self.brain.decide_full(features)
            if not decision:
                return None

            return {
                "type": "synthesis",
                "domain_predictions": domain_predictions,
                "synthesized_label": decision.get("label", ""),
                "synthesized_confidence": decision.get("confidence", 0.0),
                "num_contributing_domains": len(domain_predictions),
                "explanation": decision.get("explanation", ""),
                "novelty": self._compute_novelty(
                    np.array(decision.get("output_vector",
                             [0.0] * self._num_outputs), dtype=np.float32)),
                "candidate_idx": idx,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Stage 5: EVALUATE — Score candidates
    # ------------------------------------------------------------------

    def _stage_evaluate(self, candidates: List[Dict], features: list,
                        config: ExtrapolationConfig,
                        trace: list) -> List[Dict]:
        """Evaluate and score candidates using introspection + uncertainty."""
        scored = []

        # Pre-compute uncertainty once (same features for all candidates)
        unc = None
        try:
            unc = self.brain.get_uncertainty(features)
        except (AttributeError, Exception):
            pass

        for candidate in candidates:
            score = 0.0
            uncertainty_scores = {}

            # 5a: Confidence from the prediction itself
            conf = candidate.get("confidence", 0.0)
            score += 0.3 * conf

            # 5b: Novelty bonus (we WANT novel outputs)
            novelty = candidate.get("novelty", 0.5)
            score += 0.25 * novelty

            # 5c: Uncertainty quantification (using pre-computed unc)
            if unc:
                epistemic = unc.get("epistemic", 0.5)
                aleatoric = unc.get("aleatoric", 0.5)
                uncertainty_scores["epistemic"] = epistemic
                uncertainty_scores["aleatoric"] = aleatoric
                # Moderate epistemic uncertainty is GOOD for generation
                # (too low = memorized, too high = random)
                optimal_uncertainty = 1.0 - abs(epistemic - 0.4) / 0.6
                score += 0.2 * optimal_uncertainty

            # 5d: Self-assessment
            try:
                self_assess = self.brain.self_assess(
                    config.domain_context or "general")
                if self_assess:
                    self_conf = self_assess.get("confidence", 0.5)
                    score += 0.15 * self_conf
                    uncertainty_scores["self_assessment"] = self_conf
            except (AttributeError, Exception):
                pass

            # 5e: Phase coherence (binding quality)
            if config.use_oscillations:
                try:
                    pac = self.brain.get_pac_modulation(6.0, 40.0)
                    if pac > 0:
                        score += 0.1 * pac
                        uncertainty_scores["phase_amplitude_coupling"] = pac
                except (AttributeError, Exception):
                    pass

            candidate["score"] = score
            candidate["uncertainty"] = uncertainty_scores
            scored.append(candidate)

        # Sort by score (highest first)
        scored.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        if scored:
            trace.append(f"Top candidate score: {scored[0].get('score', 0):.3f} "
                         f"(novelty={scored[0].get('novelty', 0):.2f})")

        return scored

    # ------------------------------------------------------------------
    # Stage 6: REFINE — Mesh consensus + workspace broadcast
    # ------------------------------------------------------------------

    def _stage_refine(self, candidates: List[Dict],
                      config: ExtrapolationConfig,
                      trace: list) -> List[Dict]:
        """Refine candidates via mesh network consensus and global workspace."""
        if not candidates:
            return candidates

        best = candidates[0]

        # 6a: Mesh network consensus
        if config.use_mesh:
            try:
                mesh_available = self.brain.ti_mesh_is_available()
                if mesh_available:
                    participants = self.brain.ti_mesh_get_participant_count()
                    coherence = self.brain.ti_mesh_get_coherence()
                    best["mesh_participants"] = participants
                    best["mesh_coherence"] = coherence
                    # Mesh coherence boosts confidence
                    if coherence > 0.5:
                        best["score"] = best.get("score", 0) * (1 + 0.1 * coherence)
                    trace.append(f"Mesh: {participants} participants, "
                                 f"coherence={coherence:.2f}")
            except (AttributeError, Exception):
                pass

        # 6b: Global workspace broadcast — compete for conscious access
        try:
            strength = best.get("score", 0.5)
            self.brain.workspace_compete("extrapolation_engine",
                                          best.get("label", "unknown"),
                                          strength)
            # Read back — check if we won the competition
            has_broadcast = self.brain.workspace_has_broadcast()
            if has_broadcast:
                broadcast = self.brain.workspace_read(256)
                if broadcast:
                    content, dim, source = broadcast
                    best["workspace_broadcast"] = True
                    best["workspace_source"] = source
                    trace.append(f"Global workspace: won broadcast (source={source})")
            else:
                best["workspace_broadcast"] = False
                trace.append("Global workspace: another module won broadcast")
        except (AttributeError, Exception):
            pass

        # 6c: Basal ganglia reward prediction error
        try:
            rpe = self.brain.bg_get_rpe()
            dopamine = self.brain.bg_get_dopamine()
            best["reward_prediction_error"] = rpe
            best["dopamine_level"] = dopamine
            if rpe > 0:
                trace.append(f"Positive RPE ({rpe:.3f}) — novel and better than expected")
        except (AttributeError, Exception):
            pass

        # 6d: Safety check
        if config.safety_check:
            try:
                label = str(best.get("label", ""))
                if label:
                    safety = self.brain.lgss_check_content(label)
                    if safety and not safety.get("is_safe", True):
                        best["safety_flag"] = safety.get("reason", "unsafe")
                        trace.append(f"Safety flag: {safety.get('reason', 'unsafe')}")
            except (AttributeError, Exception):
                pass

        return candidates

    # ------------------------------------------------------------------
    # Stage 7: OUTPUT — Assemble final result
    # ------------------------------------------------------------------

    def _stage_output(self, candidates: List[Dict], features: list,
                      mode: GenerationMode, config: ExtrapolationConfig,
                      trace: list, sources: List[str],
                      start_time: float) -> GenerationResult:
        """Assemble the final GenerationResult."""
        elapsed_ms = (time.time() - start_time) * 1000

        if not candidates:
            return GenerationResult(
                content={"error": "No candidates generated"},
                confidence=0.0, novelty=0.0, coherence=0.0,
                reasoning_trace=trace, knowledge_sources=sources,
                uncertainty={}, generation_time_ms=elapsed_ms,
                mode=mode.value, cognitive_state={},
            )

        best = candidates[0]

        # Gather cognitive state snapshot
        cognitive_state = self._capture_cognitive_state()

        # Compute overall coherence (agreement among top candidates)
        coherence = self._compute_coherence(candidates)

        # Build structured content
        content = {
            "primary_output": best.get("label", ""),
            "explanation": best.get("explanation", ""),
            "type": best.get("type", mode.value),
            "top_k_candidates": [
                {
                    "label": self._extract_label(c),
                    "score": round(c.get("score", 0), 4),
                    "novelty": round(c.get("novelty", 0), 4),
                    "type": c.get("type", ""),
                }
                for c in candidates[:config.top_k]
            ],
        }

        # Add type-specific fields
        if best.get("type") == "hypothesis":
            content["hypothesis_strength"] = best.get("hypothesis_strength", 0)
            content["forward_chain_inferences"] = best.get("forward_chain_inferences", 0)
        elif best.get("type") == "interpolation":
            content["blend_alpha"] = best.get("blend_alpha", 0)
            content["cross_domain"] = best.get("cross_domain", [])
        elif best.get("type") == "synthesis":
            content["domain_predictions"] = best.get("domain_predictions", {})
            content["contributing_domains"] = best.get("num_contributing_domains", 0)

        # Mesh network info
        if "mesh_coherence" in best:
            content["mesh_coherence"] = best["mesh_coherence"]
            content["mesh_participants"] = best.get("mesh_participants", 0)

        return GenerationResult(
            content=content,
            confidence=best.get("confidence", 0.0),
            novelty=best.get("novelty", 0.0),
            coherence=coherence,
            reasoning_trace=trace,
            knowledge_sources=sources,
            uncertainty=best.get("uncertainty", {}),
            generation_time_ms=round(elapsed_ms, 2),
            mode=mode.value,
            cognitive_state=cognitive_state,
        )

    # ------------------------------------------------------------------
    # Post-Generation: Consolidation
    # ------------------------------------------------------------------

    def _post_generation_consolidate(self, result: GenerationResult,
                                     config: ExtrapolationConfig):
        """Integrate generated knowledge back into brain."""
        try:
            # Add generated content as new fact
            content_str = str(result.content.get("primary_output", ""))[:200]
            self.brain.ti_add_fact(content_str, result.confidence)

            # Reward signal for successful generation
            if result.confidence > config.confidence_threshold:
                self.brain.bg_update_reward(result.confidence, 0.5)

            # Light consolidation
            self.brain.consolidate(mode="light")
        except (AttributeError, Exception):
            pass

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def _temperature_softmax(self, logits: np.ndarray,
                             temperature: float) -> np.ndarray:
        """Temperature-scaled softmax for diverse sampling."""
        if temperature <= 0:
            # Argmax (deterministic)
            result = np.zeros_like(logits)
            result[np.argmax(logits)] = 1.0
            return result
        scaled = logits / temperature
        scaled -= np.max(scaled)  # Numerical stability
        exp = np.exp(scaled)
        return exp / (np.sum(exp) + 1e-10)

    def _compute_novelty(self, output: np.ndarray) -> float:
        """Compute novelty score: how different is this from previous generations."""
        if not self._generation_history:
            self._generation_history.append(output.copy())
            return 1.0  # First generation is maximally novel

        # Cosine distance to all previous outputs
        norm = np.linalg.norm(output)
        if norm < 1e-8:
            return 0.0

        max_sim = -1.0
        for prev in self._generation_history:
            prev_norm = np.linalg.norm(prev)
            if prev_norm < 1e-8:
                continue
            sim = float(np.dot(output, prev) / (norm * prev_norm))
            max_sim = max(max_sim, sim)

        # Track this output
        self._generation_history.append(output.copy())
        if len(self._generation_history) > self._max_history:
            self._generation_history.pop(0)

        # Novelty = 1 - max similarity to previous
        return max(0.0, 1.0 - max_sim) if max_sim >= 0.0 else 1.0

    @staticmethod
    def _extract_label(candidate: Dict) -> str:
        """Extract the best label from a candidate regardless of type."""
        return (candidate.get("label")
                or candidate.get("blended_label")
                or candidate.get("synthesized_label")
                or "")

    def _compute_coherence(self, candidates: List[Dict]) -> float:
        """Measure agreement among top candidates."""
        if len(candidates) < 2:
            return 1.0
        labels = [self._extract_label(c) for c in candidates[:5]]
        labels = [l for l in labels if l]
        if not labels:
            return 0.0
        # Fraction that agree with the top candidate
        top = labels[0]
        agreement = sum(1 for l in labels if l == top)
        return agreement / len(labels)

    def _capture_cognitive_state(self) -> Dict[str, Any]:
        """Snapshot the brain's cognitive state during generation."""
        state = {}
        try:
            state["arousal"] = self.brain.medulla_get_arousal()
        except (AttributeError, Exception):
            pass
        try:
            state["dopamine"] = self.brain.bg_get_dopamine()
        except (AttributeError, Exception):
            pass
        try:
            state["circadian_efficiency"] = self.brain.medulla_get_circadian_efficiency()
        except (AttributeError, Exception):
            pass
        try:
            state["cognitive_capacity"] = self.brain.ti_get_cognitive_capacity()
        except (AttributeError, Exception):
            pass
        try:
            state["stress_level"] = self.brain.ti_get_stress_level()
        except (AttributeError, Exception):
            pass
        try:
            state["urgency_mode"] = self.brain.ti_get_urgency_mode()
        except (AttributeError, Exception):
            pass
        try:
            state["bg_mode"] = self.brain.bg_get_mode()
        except (AttributeError, Exception):
            pass
        try:
            state["rpe"] = self.brain.bg_get_rpe()
        except (AttributeError, Exception):
            pass
        try:
            if self.brain.ti_mesh_is_available():
                state["mesh_coherence"] = self.brain.ti_mesh_get_coherence()
                state["mesh_participants"] = self.brain.ti_mesh_get_participant_count()
        except (AttributeError, Exception):
            pass
        try:
            pac = self.brain.get_pac_modulation(6.0, 40.0)
            state["theta_gamma_coupling"] = pac
        except (AttributeError, Exception):
            pass
        return state

    def _fallback_encode(self, text: str, num_inputs: int) -> list:
        """Minimal fallback encoder when instructor_agent is not available."""
        features = [0.0] * num_inputs
        if not text:
            return features
        text_lower = text[:2000].lower()
        for i, ch in enumerate(text_lower):
            bucket = (ord(ch) + i * 7) % num_inputs  # Position-dependent hashing
            features[bucket] += 1.0
        total = sum(features) or 1.0
        return [f / total for f in features]


# ---------------------------------------------------------------------------
# Convenience: Module-level helpers
# ---------------------------------------------------------------------------

def create_engine(brain, **kwargs) -> ExtrapolationEngine:
    """Create an ExtrapolationEngine from a brain instance."""
    return ExtrapolationEngine(brain, **kwargs)


def extrapolate(brain, prompt: str, domain: str = "",
                temperature: float = 0.7,
                engine: Optional[ExtrapolationEngine] = None) -> GenerationResult:
    """One-shot extrapolation convenience function.

    Pass an existing *engine* to avoid creating a disposable instance each call.
    """
    if engine is None:
        engine = ExtrapolationEngine(brain)
    return engine.generate_in_domain(prompt, domain, temperature)


def hypothesize(brain, question: str,
                domains: List[str] = None,
                engine: Optional[ExtrapolationEngine] = None) -> GenerationResult:
    """One-shot hypothesis generation convenience function.

    Pass an existing *engine* to avoid creating a disposable instance each call.
    """
    if engine is None:
        engine = ExtrapolationEngine(brain)
    return engine.hypothesize(question, domains)
