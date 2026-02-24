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

    def consolidate(self, mode="auto"):
        """
        Trigger memory consolidation ('sleep' between phases).

        Args:
            mode: "auto" (scale-aware), "light" (replay only, ~ms),
                  "full" (original 10-cycle)

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
