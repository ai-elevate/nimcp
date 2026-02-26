#!/usr/bin/env python3
"""
Cognitive Orchestrator — Layer 3 of Athena's Active Learning System
====================================================================

WHAT: Coordinate the brain's cognitive modules during training
WHY:  The brain has curiosity, introspection, self-model, consolidation,
      imagination, and executive modules — training should engage them all,
      just as a child uses curiosity to explore, introspection to self-check,
      and consolidation (sleep) to integrate knowledge
HOW:  Calls existing C-level cognitive modules through Python bindings,
      with Python-level fallbacks for modules without bindings yet

Module Roles:
  Curiosity       → Detect knowledge gaps, generate exploration questions
  Introspection   → Metacognitive monitoring ("am I actually learning?")
  Self-Model      → Track capabilities/limitations per domain
  Consolidation   → "Sleep" — replay and strengthen important patterns
  Executive       → Task scheduling — which domain to study next
  Imagination     → Generate hypothetical scenarios (Stage 3+)
"""

import logging
import random
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question templates (fallback when C curiosity binding unavailable)
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES = {
    "science": [
        "What is the mechanism behind {topic}?",
        "How does {topic} relate to thermodynamics?",
        "What experiments demonstrate {topic}?",
        "What are the key equations governing {topic}?",
    ],
    "math": [
        "What are the fundamental theorems related to {topic}?",
        "How is {topic} applied in real-world problems?",
        "What are common misconceptions about {topic}?",
        "What proofs exist for {topic}?",
    ],
    "history": [
        "What caused {topic}?",
        "What were the consequences of {topic}?",
        "Who were the key figures in {topic}?",
        "How did {topic} influence later events?",
    ],
    "literature": [
        "What themes are explored in {topic}?",
        "What literary techniques are used in {topic}?",
        "How does {topic} reflect its historical context?",
        "What critical interpretations exist for {topic}?",
    ],
    "philosophy": [
        "What are the main arguments regarding {topic}?",
        "What counterarguments exist for {topic}?",
        "How does {topic} relate to ethics?",
        "What is the historical development of {topic}?",
    ],
    "technology": [
        "How does {topic} work at a technical level?",
        "What are the trade-offs in {topic}?",
        "What alternatives exist to {topic}?",
        "What are current research frontiers in {topic}?",
    ],
    "general": [
        "What are the key concepts in {topic}?",
        "What are common misconceptions about {topic}?",
        "How does {topic} connect to other fields?",
        "What are the practical applications of {topic}?",
    ],
}


class CognitiveOrchestrator:
    """
    Coordinate cognitive modules during active learning.

    Each method tries the C binding first, falls back to Python heuristics.
    """

    def __init__(self, brain):
        self.brain = brain
        self._has_curiosity = hasattr(brain, 'curiosity_detect_gaps')
        self._has_consolidation = hasattr(brain, 'consolidate')
        self._has_uncertainty = hasattr(brain, 'get_uncertainty')
        self._has_self_assess = hasattr(brain, 'self_assess')
        self._has_bg = hasattr(brain, 'bg_update_reward')
        self._has_medulla = hasattr(brain, 'medulla_get_arousal')
        self._has_logic = hasattr(brain, 'ti_add_fact')
        self._has_reasoning = hasattr(brain, 'ti_init_reasoning')
        self._learning_history: List[dict] = []

    # ------------------------------------------------------------------
    # Curiosity: Knowledge gap detection + question generation
    # ------------------------------------------------------------------

    def generate_questions(self, topic: str, domain: str = "general",
                           max_questions: int = 4) -> List[str]:
        """
        Use curiosity module to generate exploration questions.

        Tries C binding first, falls back to template-based generation.
        """
        # Try C-level curiosity
        if self._has_curiosity:
            try:
                gaps = self.brain.curiosity_detect_gaps(topic)
                if gaps and isinstance(gaps, dict):
                    questions = gaps.get("questions", [])
                    if questions:
                        return questions[:max_questions]
            except Exception as e:
                logger.debug(f"C curiosity binding error: {e}, using templates")

        # Python fallback: template-based questions
        templates = QUESTION_TEMPLATES.get(domain, QUESTION_TEMPLATES["general"])
        questions = [t.format(topic=topic) for t in templates]
        random.shuffle(questions)
        return questions[:max_questions]

    def detect_knowledge_gaps(self, domains: Dict[str, float]) -> List[str]:
        """
        Identify domains where the brain has the most to learn.

        Args:
            domains: Dict of domain_name → mastery_score (0-1)

        Returns:
            List of domain names sorted by learning potential (lowest mastery first)
        """
        # Try C-level curiosity for each domain
        gap_scores = {}
        for domain, mastery in domains.items():
            # Gap = inverse of mastery, weighted by curiosity
            gap_scores[domain] = 1.0 - mastery

            if self._has_curiosity:
                try:
                    gaps = self.brain.curiosity_detect_gaps(domain)
                    if isinstance(gaps, dict):
                        curiosity = gaps.get("curiosity_intensity", 0.5)
                        gap_scores[domain] *= (0.5 + curiosity)
                except Exception:
                    pass

        # Sort by gap score (highest gap = most to learn)
        sorted_domains = sorted(gap_scores.keys(),
                                key=lambda d: -gap_scores[d])
        return sorted_domains

    # ------------------------------------------------------------------
    # Introspection: Metacognitive monitoring
    # ------------------------------------------------------------------

    def assess_learning_progress(self) -> dict:
        """
        Introspection: metacognitive check on learning.

        Uses brain.probe() for core metrics + optional uncertainty binding.
        """
        result = {
            "uncertainty": 0.0,
            "active_neurons": 0,
            "sparsity": 0.0,
            "learning_rate_effective": 0.0,
            "total_learning_steps": 0,
            "accuracy": 0.0,
        }

        # Core metrics from probe
        try:
            probe = self.brain.probe()
            result["active_neurons"] = probe.get("num_neurons", 0)
            result["sparsity"] = probe.get("avg_sparsity", 0.0)
            result["learning_rate_effective"] = probe.get("current_learning_rate", 0.0)
            result["total_learning_steps"] = probe.get("total_learning_steps", 0)
            result["accuracy"] = probe.get("accuracy", 0.0)
        except Exception as e:
            logger.debug(f"Probe failed: {e}")

        # Optional: epistemic uncertainty
        if self._has_uncertainty:
            try:
                unc = self.brain.get_uncertainty()
                if isinstance(unc, dict):
                    result["uncertainty"] = unc.get("total", 0.0)
                    result["epistemic"] = unc.get("epistemic", 0.0)
                    result["aleatoric"] = unc.get("aleatoric", 0.0)
            except Exception as e:
                logger.debug(f"Uncertainty binding error: {e}")

        return result

    def is_learning_stalled(self, recent_accuracies: List[float],
                            window: int = 10, threshold: float = 0.01) -> bool:
        """
        Check if learning has stalled (accuracy plateau).

        Returns True if accuracy improvement over last `window` measurements
        is less than `threshold`.
        """
        if len(recent_accuracies) < window:
            return False

        recent = recent_accuracies[-window:]
        improvement = max(recent) - min(recent)
        return improvement < threshold

    # ------------------------------------------------------------------
    # Self-Model: Track capabilities per domain
    # ------------------------------------------------------------------

    def self_assess(self, domain: str) -> dict:
        """
        Self-model: assess brain's capabilities in a domain.

        Returns assessment dict with confidence, strengths, weaknesses.
        """
        if self._has_self_assess:
            try:
                assessment = self.brain.self_assess(domain)
                if isinstance(assessment, dict):
                    return assessment
            except Exception as e:
                logger.debug(f"Self-assess binding error: {e}")

        # Python fallback: use probe + accuracy
        try:
            probe = self.brain.probe()
            return {
                "domain": domain,
                "accuracy": probe.get("accuracy", 0.0),
                "learning_steps": probe.get("total_learning_steps", 0),
                "sparsity": probe.get("avg_sparsity", 0.0),
                "assessment": "learning",
            }
        except Exception:
            return {"domain": domain, "assessment": "unknown"}

    # ------------------------------------------------------------------
    # Consolidation: Memory integration ("sleep")
    # ------------------------------------------------------------------

    def cache_communities(self):
        """
        Pre-compute and cache community structure for consolidation replay.

        Runs Louvain community detection + hub identification on the brain's
        neural network. Results are cached and used automatically by
        consolidate() when mode is "auto" or "full".

        Returns:
            dict with num_communities, num_hubs, modularity, num_neurons
            or None if unavailable.
        """
        try:
            result = self.brain.cache_communities()
            logger.info(f"Community cache built: {result}")
            return result
        except Exception as e:
            logger.warning(f"cache_communities error: {e}")
            return None

    def invalidate_community_cache(self):
        """Invalidate cached community structure, forcing recomputation."""
        try:
            self.brain.invalidate_community_cache()
            logger.info("Community cache invalidated")
        except Exception as e:
            logger.warning(f"invalidate_community_cache error: {e}")

    def consolidate(self, mode="auto"):
        """
        Trigger memory consolidation ('sleep' between phases).

        Args:
            mode: "auto" (scale-aware), "light" (replay only, ~ms),
                  "full" (original 10-cycle)

        Uses cached community structure if available for community-aware
        replay prioritization (hub boost + cross-community boost).

        Calls C-level brain_consolidate_memory() with mode-appropriate config.
        """
        if self._has_consolidation:
            try:
                result = self.brain.consolidate(mode=mode)
                logger.info(f"Consolidation complete (mode={mode}): {result}")
                return result
            except Exception as e:
                logger.warning(f"C consolidation error: {e}")

        # No fallback — consolidation is a C-level operation
        logger.info("Consolidation skipped (binding not available)")
        return None

    # ------------------------------------------------------------------
    # Basal Ganglia: Reward processing and habit formation
    # ------------------------------------------------------------------

    def update_reward(self, accuracy: float, expected_accuracy: float):
        """
        Update basal ganglia reward signal based on prediction accuracy.

        Computes reward prediction error (RPE) from actual vs expected accuracy.
        Drives dopamine modulation of learning rate and action selection.
        """
        if not self._has_bg:
            return
        try:
            self.brain.bg_update_reward(accuracy, expected_accuracy)
        except Exception as e:
            logger.warning(f"bg_update_reward error: {e}")

    def get_conflict(self) -> float:
        """
        Get basal ganglia conflict signal (0.0 = no conflict, 1.0 = high).

        High conflict indicates competing action representations — may
        benefit from slower, more deliberate processing.
        """
        if not self._has_bg:
            return 0.0
        try:
            return float(self.brain.bg_get_conflict())
        except Exception as e:
            logger.warning(f"bg_get_conflict error: {e}")
            return 0.0

    def get_operating_mode(self) -> str:
        """
        Get basal ganglia operating mode.

        Returns one of: "goal_directed", "habitual", "exploratory", "suppressed".
        """
        if not self._has_bg:
            return "goal_directed"
        try:
            mode_id = int(self.brain.bg_get_mode())
            modes = {0: "goal_directed", 1: "habitual",
                     2: "exploratory", 3: "suppressed"}
            return modes.get(mode_id, "goal_directed")
        except Exception as e:
            logger.warning(f"bg_get_mode error: {e}")
            return "goal_directed"

    def check_habit(self, domain: str) -> Optional[int]:
        """
        Check if a domain has formed a habitual action mapping.

        Returns the action_id if the domain is habitual, or None if
        the domain is still in goal-directed mode.
        """
        if not self._has_bg:
            return None
        try:
            action_id = int(self.brain.bg_check_habit(domain))
            return action_id if action_id >= 0 else None
        except Exception as e:
            logger.warning(f"bg_check_habit error: {e}")
            return None

    def register_habit(self, domain: str, action_id: int = 0):
        """Register a domain-action pair as a candidate habit."""
        if not self._has_bg:
            return
        try:
            self.brain.bg_register_habit(domain, action_id)
        except Exception as e:
            logger.warning(f"bg_register_habit error: {e}")

    def strengthen_habit(self, habit_id: int, success: bool = True):
        """Strengthen or weaken a habit based on outcome."""
        if not self._has_bg:
            return
        try:
            self.brain.bg_strengthen_habit(habit_id, success)
        except Exception as e:
            logger.warning(f"bg_strengthen_habit error: {e}")

    # ------------------------------------------------------------------
    # Medulla: Arousal and circadian modulation
    # ------------------------------------------------------------------

    def get_arousal(self) -> float:
        """
        Get current arousal level from medulla (0.0 = drowsy, 1.0 = alert).

        Arousal modulates learning rate and attention — low arousal suggests
        a consolidation break ("recess"), high arousal supports intensive learning.
        """
        if not self._has_medulla:
            return 0.5
        try:
            return float(self.brain.medulla_get_arousal())
        except Exception as e:
            logger.warning(f"medulla_get_arousal error: {e}")
            return 0.5

    def boost_arousal(self, delta: float = 0.1):
        """
        Boost arousal level (e.g. after recess or when switching domains).

        Simulates the alerting effect of novelty or rest.
        """
        if not self._has_medulla:
            return
        try:
            self.brain.medulla_boost_arousal(delta)
        except Exception as e:
            logger.warning(f"medulla_boost_arousal error: {e}")

    def get_circadian_efficiency(self) -> float:
        """
        Get circadian-modulated learning efficiency (0.0-1.0).

        Maps the brain's internal circadian phase to an efficiency multiplier.
        Training during low-efficiency periods benefits from reduced LR or
        consolidation breaks.
        """
        if not self._has_medulla:
            return 1.0
        try:
            return float(self.brain.medulla_get_circadian_efficiency())
        except Exception as e:
            logger.warning(f"medulla_get_circadian_efficiency error: {e}")
            return 1.0

    # ------------------------------------------------------------------
    # Symbolic Logic: Knowledge base and reasoning
    # ------------------------------------------------------------------

    def add_logical_fact(self, fact: str, salience: float = 0.8):
        """
        Add a symbolic fact to the brain's knowledge base.

        Facts are first-order logic predicates (e.g. "Observable(x) & Repeatable(x)").
        Salience controls how strongly the fact influences reasoning chains.
        """
        if not self._has_logic:
            return
        try:
            self.brain.ti_add_fact(fact, salience)
        except Exception as e:
            logger.warning(f"ti_add_fact error: {e}")

    def add_logical_rule(self, rule: str, priority: float = 0.5):
        """
        Add an inference rule to the knowledge base.

        Rules define entailment relationships for forward/backward chaining.
        """
        if not self._has_logic:
            return
        try:
            self.brain.ti_add_rule(rule, priority)
        except Exception as e:
            logger.warning(f"ti_add_rule error: {e}")

    def forward_chain(self, max_iterations: int = 100) -> int:
        """
        Run forward chaining on the knowledge base.

        Derives new facts from existing facts + rules until no new facts
        are produced or max_iterations is reached.

        Returns:
            Number of new facts derived, or -1 on error.
        """
        if not self._has_logic:
            return 0
        try:
            return int(self.brain.ti_forward_chain(max_iterations))
        except Exception as e:
            logger.warning(f"ti_forward_chain error: {e}")
            return -1

    def backward_chain(self, goal: str) -> float:
        """
        Run backward chaining to verify a goal.

        Attempts to prove the goal from existing facts and rules.

        Returns:
            Confidence score (0.0-1.0) that the goal is supported,
            or -1.0 on error.
        """
        if not self._has_logic:
            return -1.0
        try:
            return float(self.brain.ti_backward_chain(goal))
        except Exception as e:
            logger.warning(f"ti_backward_chain error: {e}")
            return -1.0

    # ------------------------------------------------------------------
    # Training Integration: Adaptive LR and reasoning-augmented learning
    # ------------------------------------------------------------------

    def init_reasoning(self):
        """
        Initialize the reasoning engine.

        Must be called before reason_about() or compute_adaptive_lr().
        Safe to call multiple times (idempotent).
        """
        if not self._has_reasoning:
            return
        try:
            self.brain.ti_init_reasoning()
        except Exception as e:
            logger.warning(f"ti_init_reasoning error: {e}")

    def reason_about(self, query: str) -> float:
        """
        Run a reasoning chain on a query string.

        Returns:
            Confidence score (0.0-1.0), or -1.0 on error.
        """
        if not self._has_reasoning:
            return -1.0
        try:
            return float(self.brain.ti_reason(query))
        except Exception as e:
            logger.warning(f"ti_reason error: {e}")
            return -1.0

    def compute_adaptive_lr(self, base_lr: float) -> float:
        """
        Compute BG + medulla adaptive learning rate.

        Modulates base_lr based on dopamine level, arousal, circadian phase,
        and reward prediction error. Returns adjusted LR.

        Prefers the unified pipeline (arousal inverted-U, circadian, RPE,
        instability, inflammation, Portia, stress, cognitive capacity).
        Falls back to the old 3-factor formula if unified is unavailable.
        """
        if not self._has_reasoning:
            return base_lr
        # Try unified pipeline first (8-factor composition)
        try:
            return float(self.brain.ti_compute_unified_lr(base_lr))
        except (AttributeError, Exception):
            pass
        # Fall back to old 3-factor formula
        try:
            return float(self.brain.ti_compute_adaptive_lr(base_lr))
        except Exception as e:
            logger.warning(f"ti_compute_adaptive_lr error: {e}")
            return base_lr

    def get_modulation_state(self) -> Optional[dict]:
        """
        Get full modulation state from all brain subsystems.

        Returns dict with individual factors (arousal_factor, inflammation_factor,
        portia_compute_budget, etc.) and composed finals (final_lr_factor,
        final_batch_factor, final_clip_factor, should_pause).
        """
        try:
            return self.brain.ti_compute_modulation_state()
        except (AttributeError, Exception):
            return None

    def should_pause_training(self) -> bool:
        """Check if any brain subsystem signals training should pause."""
        state = self.get_modulation_state()
        if state is None:
            return False
        return bool(state.get("should_pause", False))

    def compute_decision_cycle(self, loss_current: float, loss_previous: float,
                               grad_norm: float, grad_norm_previous: float,
                               loss_volatility: float, gradient_variance: float,
                               current_lr: float, current_batch: float) -> Optional[dict]:
        """
        Run full training decision cycle: observe -> diagnose -> simulate -> decide.

        Combines Layer 1 (convergent evidence accumulation), Layer 2 (causal DAG
        simulation), and Layer 3 (abductive diagnosis) into one call.

        Returns dict with consensus_action, lr_factor, batch_factor, grad_clip_factor,
        urgency, converged, primary_diagnosis, causal_explanation, recommend_pause, etc.
        Returns None if the binding is unavailable.
        """
        try:
            return self.brain.ti_compute_decision_cycle(
                loss_current, loss_previous,
                grad_norm, grad_norm_previous,
                loss_volatility, gradient_variance,
                current_lr, current_batch)
        except (AttributeError, Exception) as e:
            logger.debug(f"decision_cycle unavailable: {e}")
            return None

    def get_last_gradient_norm(self) -> float:
        """Get gradient L2 norm from the most recent learn() call."""
        try:
            return float(self.brain.get_last_gradient_norm())
        except (AttributeError, Exception):
            return -1.0

    def post_batch_update(self, accuracy: float, expected: float,
                          domain: str):
        """
        Post-batch integration: update BG reward, medulla arousal, and
        training integration state after a training batch.
        """
        if not self._has_reasoning:
            return
        try:
            self.brain.ti_post_batch_update(accuracy, expected, domain)
        except Exception as e:
            logger.warning(f"ti_post_batch_update error: {e}")

    # ------------------------------------------------------------------
    # Executive: Task planning and scheduling
    # ------------------------------------------------------------------

    def plan_next_task(self, domain_mastery: Dict[str, float],
                       available_domains: List[str]) -> str:
        """
        Executive function: decide what to study next.

        Strategy: Balance between struggling domains (remediation)
        and curious domains (exploration). Prioritize:
        1. Domains below 0.3 mastery (critical gaps)
        2. Domains with highest curiosity scores
        3. Round-robin for remaining
        """
        if not available_domains:
            return "general"

        # Categorize domains
        critical = [d for d in available_domains
                    if domain_mastery.get(d, 0) < 0.3]
        mid_range = [d for d in available_domains
                     if 0.3 <= domain_mastery.get(d, 0) < 0.7]
        advanced = [d for d in available_domains
                    if domain_mastery.get(d, 0) >= 0.7]

        # Priority 1: Critical gaps (60% chance)
        if critical and random.random() < 0.6:
            return random.choice(critical)

        # Priority 2: Mid-range with curiosity boost (30% chance)
        if mid_range and random.random() < 0.75:
            # Weight by inverse mastery (lower mastery = more attention)
            weights = [1.0 - domain_mastery.get(d, 0.5) for d in mid_range]
            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]
                return random.choices(mid_range, weights=weights, k=1)[0]
            return random.choice(mid_range)

        # Priority 3: Advanced domains (for Stage 3-4 activities)
        if advanced:
            return random.choice(advanced)

        return random.choice(available_domains)

    def get_curiosity_priorities(self, domains: Dict[str, float]) -> List[str]:
        """
        Rank domains by curiosity/knowledge-gap score.

        Combines mastery gap with curiosity module (if available).
        """
        return self.detect_knowledge_gaps(domains)

    # ------------------------------------------------------------------
    # Learning history tracking
    # ------------------------------------------------------------------

    def record_batch(self, domain: str, accuracy: float, loss: float,
                     batch_size: int):
        """Record a training batch result for trend analysis."""
        self._learning_history.append({
            "timestamp": time.time(),
            "domain": domain,
            "accuracy": accuracy,
            "loss": loss,
            "batch_size": batch_size,
        })
        # Keep last 1000 entries
        if len(self._learning_history) > 1000:
            self._learning_history = self._learning_history[-1000:]

    def get_recent_trend(self, domain: str = None,
                         n: int = 20) -> Dict[str, float]:
        """Get recent accuracy/loss trend for a domain."""
        history = self._learning_history
        if domain:
            history = [h for h in history if h["domain"] == domain]

        recent = history[-n:]
        if not recent:
            return {"avg_accuracy": 0.0, "avg_loss": 0.0, "n": 0}

        return {
            "avg_accuracy": sum(h["accuracy"] for h in recent) / len(recent),
            "avg_loss": sum(h["loss"] for h in recent) / len(recent),
            "n": len(recent),
        }
