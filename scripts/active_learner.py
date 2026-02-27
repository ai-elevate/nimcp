#!/usr/bin/env python3
"""
Active Learner — Layer 2 of Athena's Active Learning System
=============================================================

WHAT: Progressive 4-stage active learning (worksheets → explain → research → create)
WHY:  Passive training saturates; active learning (researching, creating, getting
      graded) drives deeper understanding — modeled after child development stages
HOW:  Stage routing based on domain mastery, with web research (safety-gated),
      content generation, and self-grading via heuristic scoring

Stages:
  0 — Worksheets  (mastery < 0.4): Answer questions, get graded
  1 — Explain     (mastery 0.4-0.6): Answer + explain reasoning
  2 — Research    (mastery 0.6-0.8): Web research + synthesis
  3 — Create      (mastery > 0.8): Generate original content, self-grade
"""

import hashlib
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ActiveLearner:
    """
    Progressive active learning with 4 developmental stages.

    Coordinates SocraticTrainer (Layer 1), SafetyGate (Layer 4),
    and CognitiveOrchestrator (Layer 3) to implement active learning.
    """

    def __init__(self, brain, socratic, safety_gate, cognitive, config=None):
        """
        Args:
            brain: NIMCP brain handle
            socratic: SocraticTrainer instance (Layer 1)
            safety_gate: SafetyGate instance (Layer 4)
            cognitive: CognitiveOrchestrator instance (Layer 3)
            config: Optional configuration dict
        """
        self.brain = brain
        self.socratic = socratic
        self.safety = safety_gate
        self.cognitive = cognitive
        self.stage_thresholds = (0.4, 0.6, 0.8)
        self._stage_stats = {i: {"attempts": 0, "successes": 0} for i in range(4)}

    # ------------------------------------------------------------------
    # Stage routing
    # ------------------------------------------------------------------

    def get_stage(self, domain_mastery: float) -> int:
        """
        Determine active learning stage based on domain mastery.

        0=worksheets (<0.4), 1=explain (0.4-0.6),
        2=research (0.6-0.8), 3=create (>0.8)
        """
        if domain_mastery < self.stage_thresholds[0]:
            return 0
        elif domain_mastery < self.stage_thresholds[1]:
            return 1
        elif domain_mastery < self.stage_thresholds[2]:
            return 2
        else:
            return 3

    def train_active(self, features: List[float], label: str,
                     domain: str = "general",
                     metadata: Optional[dict] = None) -> dict:
        """
        Route to appropriate stage based on domain mastery.

        Stages 0-1 always run (they just use Socratic training).
        Stages 2-3 require additional prerequisites (web access, etc.).
        """
        mastery = self.socratic.mastery.mastery(domain)
        stage = self.get_stage(mastery)

        if stage == 0:
            return self._stage_worksheets(features, label, domain, metadata)
        elif stage == 1:
            return self._stage_explain(features, label, domain, metadata)
        elif stage == 2:
            return self._stage_explain(features, label, domain, metadata)
        else:
            return self._stage_explain(features, label, domain, metadata)

    # ------------------------------------------------------------------
    # Stage 0: Worksheets — basic Socratic training
    # ------------------------------------------------------------------

    def _stage_worksheets(self, features, label, domain, metadata) -> dict:
        """Stage 0: Pure Socratic predict→teach loop."""
        result = self.socratic.train_example(features, label, domain, metadata)
        result["stage"] = 0
        result["stage_name"] = "worksheets"
        self._record_stage(0, result["correct"])
        return result

    # ------------------------------------------------------------------
    # Stage 1: Explain Your Work — answer + evaluate explanation quality
    # ------------------------------------------------------------------

    def _stage_explain(self, features, label, domain, metadata) -> dict:
        """Stage 1: Socratic training + explanation quality scoring."""
        # First do the Socratic training
        result = self.socratic.train_example(features, label, domain, metadata)

        # Try to get an explanation from decide_full
        explanation = ""
        try:
            decision = self.brain.decide_full(features)
            if isinstance(decision, dict):
                explanation = decision.get("explanation", "")
        except Exception:
            pass

        # Grade the explanation
        explanation_score = self.grade_explanation(explanation, label, domain)

        # Combined score: correctness + explanation quality + calibration
        correctness = 1.0 if result["correct"] else 0.0
        calibration = 1.0 - abs(result.get("pred_confidence", 0.5) - correctness)
        quality_score = (0.5 * correctness +
                         0.3 * explanation_score +
                         0.2 * calibration)

        # Additional learning pass with quality-scaled confidence
        if quality_score < 0.8:
            self.brain.learn(features, label, quality_score)

        result["stage"] = 1
        result["stage_name"] = "explain"
        result["explanation"] = explanation[:200] if explanation else ""
        result["explanation_score"] = explanation_score
        result["quality_score"] = quality_score
        self._record_stage(1, result["correct"])
        return result

    # ------------------------------------------------------------------
    # Stage 2: Research — web research + synthesis (called from pipeline)
    # ------------------------------------------------------------------

    def research_topic(self, topic: str, domain: str,
                       fetch_fn=None) -> List[dict]:
        """
        Stage 3: Web research with safety gating.

        Args:
            topic: Topic to research
            domain: Training domain
            fetch_fn: Optional callable(query) -> list of {url, text} dicts
                      If None, research is simulated with template questions.

        Returns:
            List of processed results.
        """
        # 1. Curiosity generates questions
        questions = self.cognitive.generate_questions(topic, domain)

        processed = []
        for q in questions:
            # 2. Safety-gate each search query
            if not self.safety.check_search_query(q, domain):
                continue

            if fetch_fn is None:
                # No web access — just record the question for future use
                processed.append({
                    "question": q,
                    "status": "deferred",
                    "reason": "no fetch function provided",
                })
                continue

            # 3. Fetch search results
            try:
                results = fetch_fn(q)
            except Exception as e:
                logger.warning(f"Fetch error for '{q}': {e}")
                continue

            for result in results:
                text = result.get("text", "")
                if not text:
                    continue

                # 4. Filter fetched content through LGSS
                if not self.safety.filter_content(text):
                    continue

                # 5. Learn from safe content (reduced confidence — external source)
                features = self._encode_text(text)
                self.brain.learn(features, str(hash(domain) % 32), 0.7)

                processed.append({
                    "question": q,
                    "status": "learned",
                    "text_length": len(text),
                    "url": result.get("url", ""),
                })

        return processed

    # ------------------------------------------------------------------
    # Stage 3: Create and Grade — content generation + self-grading
    # ------------------------------------------------------------------

    def create_and_grade(self, prompt_features: List[float],
                         domain: str,
                         reference_label: str) -> dict:
        """
        Stage 4: Generate content, grade it, learn from feedback.

        Args:
            prompt_features: Input features to prompt generation
            domain: Training domain
            reference_label: Expected/reference label

        Returns:
            Dict with generated content, grade, and feedback.
        """
        # Generate using decide_full (engages imagination/creative modules)
        generated = ""
        decision = {}
        try:
            decision = self.brain.decide_full(prompt_features)
            if isinstance(decision, dict):
                generated = decision.get("explanation", "")
                pred_label = decision.get("label", "")
            else:
                pred_label = str(decision)
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return {"stage": 3, "stage_name": "create", "error": str(e)}

        # Safety check generated content
        if generated and not self.safety.filter_generated_content(generated):
            return {
                "stage": 3,
                "stage_name": "create",
                "filtered": True,
                "reason": "content failed safety filter",
            }

        # Grade the content
        grade = self.grade_content(generated, reference_label, domain,
                                   pred_label)

        # Feedback loop: learn from grade
        self.brain.learn(prompt_features, reference_label, grade)

        self._record_stage(3, grade > 0.5)

        return {
            "stage": 3,
            "stage_name": "create",
            "predicted_label": pred_label,
            "reference_label": reference_label,
            "generated_length": len(generated),
            "grade": grade,
        }

    # ------------------------------------------------------------------
    # Grading functions
    # ------------------------------------------------------------------

    def grade_explanation(self, explanation: str, label: str,
                         domain: str) -> float:
        """
        Score explanation quality using measurable heuristics.

        Metrics:
        - Length: Explanation should be substantive (>10 chars)
        - Keyword relevance: Contains domain-relevant terms
        - Non-repetitive: Unique word ratio
        """
        if not explanation or len(explanation.strip()) < 5:
            return 0.0

        score = 0.0
        words = explanation.lower().split()

        # Length score (0-0.3): Substantive explanation
        if len(words) >= 3:
            score += 0.1
        if len(words) >= 10:
            score += 0.1
        if len(words) >= 20:
            score += 0.1

        # Keyword relevance (0-0.4): Domain-specific terms present
        from safety_gate import DOMAIN_KEYWORDS
        keywords = DOMAIN_KEYWORDS.get(domain, DOMAIN_KEYWORDS.get("general", []))
        explanation_lower = explanation.lower()
        keyword_matches = sum(1 for kw in keywords if kw in explanation_lower)
        relevance = min(keyword_matches / max(len(keywords), 1), 1.0)
        score += 0.4 * relevance

        # Non-repetitive (0-0.3): Unique word ratio
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += 0.3 * unique_ratio

        return min(score, 1.0)

    def grade_content(self, content: str, reference_label: str,
                      domain: str, predicted_label: str = "") -> float:
        """
        Score generated content using measurable heuristics.

        Metrics:
        - Factual accuracy: predicted label matches reference (0 or 1)
        - Coherence: Content length + unique word ratio
        - Novelty: Output differs from most common prediction
        """
        score = 0.0

        # Factual accuracy (0-0.5): Does predicted label match reference?
        if predicted_label == reference_label:
            score += 0.5

        # Coherence (0-0.3): Content quality
        if content:
            words = content.split()
            if len(words) >= 3:
                score += 0.1
            if len(words) >= 10:
                score += 0.1
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                score += 0.1 * unique_ratio

        # Novelty (0-0.2): Non-trivial output
        if content and len(content.strip()) > 0:
            score += 0.1
            # Bonus for longer, substantive content
            if len(content) > 50:
                score += 0.1

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Creativity Testing — generate novel outputs, measure originality
    # ------------------------------------------------------------------

    # Cross-domain prompt seeds: combine two domains to test creative transfer
    CROSS_DOMAIN_PROMPTS = [
        ("science", "art", "How would you visually represent quantum entanglement?"),
        ("math", "music", "What is the relationship between fibonacci sequences and harmony?"),
        ("history", "technology", "How did ancient civilizations solve the problems we use computers for?"),
        ("philosophy", "medicine", "What are the ethical implications of consciousness in artificial brains?"),
        ("economics", "psychology", "How does loss aversion shape market behavior?"),
        ("literature", "science", "How does metaphor function like a scientific model?"),
        ("geography", "art", "How does landscape shape artistic expression?"),
        ("music", "math", "What mathematical structures underlie musical scales?"),
        ("law", "philosophy", "When does legal reasoning require moral intuition?"),
        ("psychology", "technology", "How does interface design exploit cognitive biases?"),
    ]

    def creativity_exam(self, num_inputs: int, num_trials: int = 10) -> dict:
        """
        Test the brain's ability to create something that hasn't existed before.

        Creativity is measured along 4 dimensions:
        - Novelty: How different is the output from seen training patterns?
        - Coherence: Is the output internally consistent (not random noise)?
        - Surprise: How unexpected is the output (high prediction error)?
        - Cross-domain: Does the output integrate concepts from multiple domains?

        Args:
            num_inputs: Feature vector size for the brain
            num_trials: Number of creativity prompts to test

        Returns:
            Dict with per-trial scores and overall creativity metrics.
        """
        trials = []
        prompts = self.CROSS_DOMAIN_PROMPTS[:num_trials]

        for domain_a, domain_b, prompt_text in prompts:
            trial = self._run_creativity_trial(
                domain_a, domain_b, prompt_text, num_inputs
            )
            trials.append(trial)

        # Aggregate creativity scores
        if not trials:
            return {"creativity_score": 0.0, "trials": []}

        avg_novelty = sum(t["novelty"] for t in trials) / len(trials)
        avg_coherence = sum(t["coherence"] for t in trials) / len(trials)
        avg_surprise = sum(t["surprise"] for t in trials) / len(trials)
        avg_cross = sum(t["cross_domain"] for t in trials) / len(trials)

        # Overall creativity: weighted combination
        # Novelty matters most (can't be creative by repeating),
        # coherence ensures it's not just noise
        creativity_score = (
            0.35 * avg_novelty +
            0.30 * avg_coherence +
            0.20 * avg_surprise +
            0.15 * avg_cross
        )

        return {
            "creativity_score": creativity_score,
            "novelty": avg_novelty,
            "coherence": avg_coherence,
            "surprise": avg_surprise,
            "cross_domain": avg_cross,
            "num_trials": len(trials),
            "trials": trials,
        }

    def _run_creativity_trial(self, domain_a: str, domain_b: str,
                               prompt_text: str, num_inputs: int) -> dict:
        """Run a single creativity trial with a cross-domain prompt."""
        # Encode the prompt into features, blending two domain signatures
        features_a = self._encode_text(f"{domain_a} {prompt_text}", num_inputs)
        features_b = self._encode_text(f"{domain_b} {prompt_text}", num_inputs)

        # Blend: 50/50 mix of both domain encodings
        blended = [(a + b) / 2.0 for a, b in zip(features_a, features_b)]

        # Add controlled noise to push the brain outside its comfort zone
        import random as rng
        noise_level = 0.1
        creative_input = [
            v + rng.gauss(0, noise_level) for v in blended
        ]
        # Clamp to [0, 1]
        creative_input = [max(0.0, min(1.0, v)) for v in creative_input]

        # Get brain's creative output
        pred_label = ""
        pred_confidence = 0.0
        output_vector = []
        explanation = ""
        try:
            decision = self.brain.decide_full(creative_input)
            if isinstance(decision, dict):
                pred_label = decision.get("label", "")
                pred_confidence = decision.get("confidence", 0.0)
                output_vector = decision.get("output_vector", [])
                explanation = decision.get("explanation", "")
            else:
                pred_label, pred_confidence = self.brain.predict(creative_input)
        except Exception:
            pred_label, pred_confidence = self.brain.predict(creative_input)

        # --- Score Novelty ---
        # Compare output to what domain_a and domain_b alone would produce
        novelty = self._score_novelty(
            creative_input, blended, output_vector, pred_label
        )

        # --- Score Coherence ---
        # Output should be internally consistent, not random noise
        coherence = self._score_coherence(output_vector, explanation)

        # --- Score Surprise ---
        # How uncertain was the brain about its own output?
        # High uncertainty + coherent output = creative insight
        surprise = self._score_surprise(pred_confidence, output_vector)

        # --- Score Cross-Domain Integration ---
        # Does the output reference/activate patterns from both domains?
        cross_domain = self._score_cross_domain(
            domain_a, domain_b, explanation, output_vector, features_a, features_b
        )

        return {
            "prompt": prompt_text,
            "domain_a": domain_a,
            "domain_b": domain_b,
            "predicted_label": pred_label,
            "confidence": pred_confidence,
            "novelty": novelty,
            "coherence": coherence,
            "surprise": surprise,
            "cross_domain": cross_domain,
            "explanation_length": len(explanation),
        }

    def _score_novelty(self, creative_input: list, blended: list,
                        output_vector: list, pred_label: str) -> float:
        """
        Novelty: How different is the output from simply interpolating inputs?

        A truly novel output diverges from what a linear combination would predict.
        We measure the distance between the actual output and the blended input.
        """
        if not output_vector:
            # No output vector — use prediction as proxy
            # Novel if the label is unexpected (not just "output_0")
            if pred_label and not pred_label.startswith("output_"):
                return 0.6  # Learned label = some knowledge, moderate novelty
            return 0.2  # Default output = low novelty

        # Cosine distance between output and blended input
        # (high distance = novel output, not just echoing input)
        if len(output_vector) == 0:
            return 0.0

        # Normalize output to same length as input for comparison
        out_sample = output_vector[:len(blended)] if len(output_vector) >= len(blended) else output_vector
        inp_sample = blended[:len(out_sample)]

        dot = sum(a * b for a, b in zip(out_sample, inp_sample))
        mag_a = sum(a * a for a in out_sample) ** 0.5
        mag_b = sum(b * b for b in inp_sample) ** 0.5

        if mag_a < 1e-8 or mag_b < 1e-8:
            return 0.5  # One is near zero — uncertain

        cosine_sim = dot / (mag_a * mag_b)
        # Low similarity = high novelty; transform to [0, 1]
        novelty = 1.0 - max(0.0, min(1.0, cosine_sim))

        # Bonus: output has structure (not all same value)
        if len(out_sample) > 1:
            mean_out = sum(out_sample) / len(out_sample)
            variance = sum((v - mean_out) ** 2 for v in out_sample) / len(out_sample)
            if variance > 0.01:
                novelty = min(1.0, novelty + 0.1)  # Structured output bonus

        return novelty

    def _score_coherence(self, output_vector: list, explanation: str) -> float:
        """
        Coherence: Is the output internally consistent or random noise?

        Coherent outputs have structure — not uniform, not maximally random.
        """
        score = 0.0

        if output_vector and len(output_vector) > 1:
            values = output_vector
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)

            # Moderate variance = coherent (not all same, not random)
            # Very low variance → just echoing → 0.3
            # Moderate variance → structured → 0.8
            # Very high variance → possibly noise → 0.4
            if variance < 0.001:
                score += 0.3  # Nearly uniform
            elif variance < 0.1:
                score += 0.8  # Structured
            elif variance < 0.5:
                score += 0.6  # Moderate structure
            else:
                score += 0.4  # High variance — possibly noisy

            # Check for dominant output (one clear winner = decisive)
            max_val = max(values)
            if max_val > 0:
                dominance = max_val / (sum(abs(v) for v in values) / len(values) + 1e-8)
                if dominance > 2.0:
                    score = min(1.0, score + 0.1)

        if explanation:
            words = explanation.split()
            if len(words) >= 3:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio > 0.5:
                    score = min(1.0, score + 0.1)

        return min(score, 1.0)

    def _score_surprise(self, confidence: float, output_vector: list) -> float:
        """
        Surprise: How unexpected was this output?

        High surprise = brain was uncertain but still produced something coherent.
        This mirrors the "aha!" moment — unexpected yet meaningful.
        """
        # Low confidence on a coherent output = high surprise (creative insight)
        # High confidence = expected output = low surprise
        surprise = 1.0 - confidence

        # Bonus: output has entropy (not one-hot)
        if output_vector and len(output_vector) > 1:
            # Softmax-like normalization for entropy
            max_v = max(output_vector)
            shifted = [v - max_v for v in output_vector]
            import math
            exp_vals = [math.exp(min(v, 20)) for v in shifted]  # Clamp to avoid overflow
            total = sum(exp_vals) + 1e-10
            probs = [e / total for e in exp_vals]

            # Shannon entropy
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            max_entropy = math.log(len(probs) + 1e-10)
            normalized_entropy = entropy / (max_entropy + 1e-10)

            # Moderate entropy = creative (not one-hot, not uniform)
            if 0.3 < normalized_entropy < 0.8:
                surprise = min(1.0, surprise + 0.15)

        return min(max(surprise, 0.0), 1.0)

    def _score_cross_domain(self, domain_a: str, domain_b: str,
                             explanation: str, output_vector: list,
                             features_a: list, features_b: list) -> float:
        """
        Cross-domain integration: Does the output reflect both input domains?

        True creativity often comes from combining concepts across fields.
        """
        score = 0.0

        # Check if explanation mentions concepts from both domains
        if explanation:
            from safety_gate import DOMAIN_KEYWORDS
            keywords_a = DOMAIN_KEYWORDS.get(domain_a, [])
            keywords_b = DOMAIN_KEYWORDS.get(domain_b, [])
            exp_lower = explanation.lower()

            hits_a = sum(1 for kw in keywords_a if kw in exp_lower)
            hits_b = sum(1 for kw in keywords_b if kw in exp_lower)

            # Both domains represented = good cross-domain integration
            if hits_a > 0 and hits_b > 0:
                score += 0.6
            elif hits_a > 0 or hits_b > 0:
                score += 0.3

        # Check output vector: correlates with BOTH domain inputs
        if output_vector and features_a and features_b:
            out_sample = output_vector[:min(len(output_vector), len(features_a))]
            fa_sample = features_a[:len(out_sample)]
            fb_sample = features_b[:len(out_sample)]

            # Correlation with domain A
            corr_a = self._correlation(out_sample, fa_sample)
            # Correlation with domain B
            corr_b = self._correlation(out_sample, fb_sample)

            # Both positive correlations = integrating both domains
            if corr_a > 0.1 and corr_b > 0.1:
                score = min(1.0, score + 0.3)
            elif corr_a > 0.1 or corr_b > 0.1:
                score = min(1.0, score + 0.15)

        # Minimum score: the brain at least attempted
        return max(score, 0.1)

    @staticmethod
    def _correlation(a: list, b: list) -> float:
        """Pearson correlation coefficient between two vectors."""
        n = min(len(a), len(b))
        if n < 2:
            return 0.0
        mean_a = sum(a[:n]) / n
        mean_b = sum(b[:n]) / n
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
        std_a = (sum((a[i] - mean_a) ** 2 for i in range(n)) / n) ** 0.5
        std_b = (sum((b[i] - mean_b) ** 2 for i in range(n)) / n) ** 0.5
        if std_a < 1e-8 or std_b < 1e-8:
            return 0.0
        return cov / (std_a * std_b)

    # ------------------------------------------------------------------
    # Text encoding helper
    # ------------------------------------------------------------------

    def _encode_text(self, text: str, num_features: int = 128) -> List[float]:
        """Encode text into feature vector (same as StreamingDatasetProcessor)."""
        features = [0.0] * num_features
        text_lower = text.lower().strip()
        if not text_lower:
            return features

        for ch in text_lower:
            features[ord(ch) % num_features] += 1.0

        for i in range(len(text_lower) - 1):
            bigram = text_lower[i:i + 2]
            h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            features[h % num_features] += 0.5

        words = text_lower.split()
        for wi, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for offset in range(3):
                features[(h + offset * 31) % num_features] += (wi + 1) * 0.1

        mx = max(features) if features else 1.0
        if mx > 0:
            features = [v / mx for v in features]
        return features

    # ------------------------------------------------------------------
    # Stage tracking
    # ------------------------------------------------------------------

    def _record_stage(self, stage: int, success: bool):
        self._stage_stats[stage]["attempts"] += 1
        if success:
            self._stage_stats[stage]["successes"] += 1

    def get_stage_stats(self) -> Dict[int, dict]:
        """Get success rates per stage."""
        result = {}
        stage_names = {0: "worksheets", 1: "explain", 2: "research", 3: "create"}
        for stage, stats in self._stage_stats.items():
            attempts = stats["attempts"]
            successes = stats["successes"]
            result[stage] = {
                "name": stage_names[stage],
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / max(attempts, 1),
            }
        return result
