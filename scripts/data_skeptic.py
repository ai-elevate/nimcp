#!/usr/bin/env python3
"""
DataSkeptic — Epistemic Quality Grading for Training Data
==========================================================

WHAT: Grade every training example on 7 dimensions before learning
WHY:  Athena NEVER assumes training data is accurate. Every example is
      graded before learning. Data is NEVER censored or rejected — Athena
      learns from ALL data, but with calibrated skepticism.
HOW:  7-dimension grading (accuracy, source_reliability, evidence_quality,
      coherence, bias, relevance, ethics) → composite score → confidence
      modifier that scales learning rate.

Ethics Scale:
  Malignant  (0–3.33): Hate speech, propaganda, incitement — still learned
                        with extreme skepticism so brain understands evil
  Neutral    (3.34–6.66): Factual, objective, neither prosocial nor harmful
  Beneficial (6.67–10.0): Educational, prosocial, constructive

Integration: Python DataSkeptic (pre-filter) + C epistemic filter + ethics
engine + LGSS training guard + brain immune system (internal).
"""

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# DataGrade — 7-dimension quality score
# ---------------------------------------------------------------------------

@dataclass
class DataGrade:
    """Quality grade for a single training example."""
    accuracy: float = 5.0           # 0-10: Factual correctness
    source_reliability: float = 5.0 # 0-10: Source trustworthiness
    evidence_quality: float = 5.0   # 0-10: Supporting evidence strength
    coherence: float = 5.0          # 0-10: Internal consistency
    bias: float = 5.0               # 0-10: Freedom from bias (10=unbiased)
    relevance: float = 5.0          # 0-10: Domain relevance / educational value
    ethics: float = 5.0             # 0-10: Ethical score with labeled bands

    @property
    def ethics_label(self) -> str:
        """Malignant (0-3.33), Neutral (3.34-6.66), Beneficial (6.67-10)"""
        if self.ethics <= 3.33:
            return "malignant"
        if self.ethics <= 6.66:
            return "neutral"
        return "beneficial"

    @property
    def composite(self) -> float:
        """Weighted composite score 0-10."""
        weights = [0.25, 0.10, 0.15, 0.15, 0.10, 0.10, 0.15]
        scores = [self.accuracy, self.source_reliability, self.evidence_quality,
                  self.coherence, self.bias, self.relevance, self.ethics]
        return sum(w * s for w, s in zip(weights, scores))

    @property
    def confidence_modifier(self) -> float:
        """Scale learning confidence: 0.1 (highly skeptical) to 1.0 (trusted).
        ALL data is learned — nothing rejected — confidence reflects quality."""
        return max(0.1, min(1.0, self.composite / 10.0))

    @property
    def is_malignant(self) -> bool:
        """Flag for malignant content — still learned, but flagged."""
        return self.ethics_label == "malignant"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ethics_label"] = self.ethics_label
        d["composite"] = round(self.composite, 3)
        d["confidence_modifier"] = round(self.confidence_modifier, 3)
        d["is_malignant"] = self.is_malignant
        return d


# ---------------------------------------------------------------------------
# Known patterns
# ---------------------------------------------------------------------------

# Known high-quality academic sources
_ACADEMIC_SOURCES = frozenset({
    "mmlu", "squad", "arc", "hellaswag", "winogrande", "commonsense_qa",
    "openwebtext", "wikitext", "fineweb", "gutenberg", "booksum",
    "hh_rlhf", "social_chemistry", "prosocial_dialog", "cais/mmlu",
    "deepmind/code_contests", "timit", "cifar", "fashion_mnist",
})

# Known malignant content patterns (regex → penalty)
_MALIGNANT_PATTERNS = [
    (re.compile(r"\bmein\s+kampf\b", re.I), 2.5),
    (re.compile(r"\bturner\s+diaries?\b", re.I), 2.5),
    (re.compile(r"\bprotocols?\s+of\s+.*?elders?\b", re.I), 2.5),
    (re.compile(r"\banarchist\s+cookbook\b", re.I), 2.0),
    (re.compile(r"\bwhite\s+genocide\b", re.I), 2.0),
    (re.compile(r"\bgreat\s+replacement\b", re.I), 1.5),
    (re.compile(r"\brace\s+war\b", re.I), 2.0),
    (re.compile(r"\bfinal\s+solution\b", re.I), 1.0),
    (re.compile(r"\bethnic\s+cleansing\b", re.I), 1.5),
]

# Dehumanizing language indicators
_DEHUMANIZING = re.compile(
    r"\b(subhuman|vermin|cockroach|infestation|exterminate|purge)\b", re.I
)

# Manipulation/propaganda techniques
_MANIPULATION = re.compile(
    r"\b(wake\s+up\s+sheeple|they\s+don.t\s+want\s+you\s+to\s+know|"
    r"the\s+truth\s+they\s+hide|false\s+flag|crisis\s+actor)\b", re.I
)

# Evidence indicators
_EVIDENCE_STRONG = re.compile(
    r"\b(doi:|arxiv:|pmid:|according\s+to\s+(?:the|a)\s+study|"
    r"peer[\s-]?reviewed|meta[\s-]?analysis|systematic\s+review)\b", re.I
)
_EVIDENCE_MODERATE = re.compile(
    r"\b(research\s+shows|studies\s+suggest|evidence\s+indicates|"
    r"experts?\s+say|professor|university|institute)\b", re.I
)
_EVIDENCE_ANECDOTAL = re.compile(
    r"\b(my\s+friend\s+said|i\s+heard|someone\s+told\s+me|"
    r"everybody\s+knows|it.s\s+common\s+knowledge|trust\s+me)\b", re.I
)

# Bias indicators
_EMOTIONAL_MANIPULATION = re.compile(
    r"\b(outrageous|disgusting|shocking|unbelievable|horrifying|"
    r"destroy|annihilate|obliterate)\b", re.I
)
_FALLACIES = re.compile(
    r"\b(ad\s+hominem|straw\s+?man|slippery\s+slope|"
    r"appeal\s+to\s+(?:authority|emotion|fear)|red\s+herring|"
    r"false\s+(?:dilemma|equivalence)|loaded\s+question)\b", re.I
)

# Hedging language (reduces absolute-claim penalty)
_HEDGING = re.compile(
    r"\b(studies\s+show|according\s+to|research\s+suggests|"
    r"may|might|possibly|likely|approximately|roughly|"
    r"it\s+appears|evidence\s+suggests)\b", re.I
)

# Absolute claims (penalized when unhedged)
_ABSOLUTE_CLAIMS = re.compile(
    r"\b(always|never|proven\s+fact|undeniable|irrefutable|"
    r"100\s*%|guaranteed|absolutely|without\s+exception)\b", re.I
)

# Known false claims
_FALSE_CLAIMS = re.compile(
    r"\b(flat\s+earth|holocaust\s+(?:hoax|didn.t|never)|"
    r"moon\s+landing\s+(?:fake|hoax|staged)|"
    r"vaccines?\s+cause\s+autism|5g\s+(?:causes?|spreads?))\b", re.I
)

# Domain relevance keywords per domain
_DOMAIN_KEYWORDS: Dict[str, list] = {
    "language": ["grammar", "syntax", "semantics", "word", "sentence", "text", "language"],
    "literature": ["novel", "poem", "story", "author", "character", "narrative", "literary"],
    "programming": ["code", "function", "algorithm", "class", "variable", "software", "api"],
    "ethics": ["moral", "ethical", "right", "wrong", "justice", "fairness", "virtue"],
    "philosophy": ["epistemology", "ontology", "logic", "argument", "reason", "philosophy"],
    "history": ["century", "war", "empire", "revolution", "treaty", "historical", "era"],
    "psychology": ["behavior", "cognition", "emotion", "memory", "personality", "mental"],
    "anthropology": ["culture", "society", "ritual", "kinship", "tribe", "anthropology"],
    "social_sciences": ["law", "governance", "policy", "institution", "social", "public"],
    "medicine": ["patient", "disease", "treatment", "diagnosis", "clinical", "symptom"],
    "biology": ["cell", "gene", "organism", "evolution", "species", "protein", "dna"],
    "chemistry": ["molecule", "reaction", "compound", "element", "bond", "chemical"],
    "physics": ["force", "energy", "quantum", "particle", "wave", "relativity", "mass"],
    "humanities": ["civilization", "culture", "art", "human", "belief", "geography"],
    "finance": ["market", "stock", "investment", "portfolio", "risk", "revenue", "profit"],
    "cryptocurrency": ["bitcoin", "blockchain", "crypto", "token", "defi", "wallet", "mining"],
    "sociology": ["social", "inequality", "class", "gender", "institution", "norm"],
    "paleontology": ["fossil", "dinosaur", "extinction", "evolution", "prehistoric", "species"],
    "economics": ["gdp", "inflation", "supply", "demand", "fiscal", "monetary", "market"],
    "earth_sciences": ["geology", "climate", "tectonic", "atmosphere", "ocean", "geography"],
    "audio": ["sound", "frequency", "audio", "noise", "signal", "acoustic", "music"],
    "visual": ["image", "pixel", "object", "shape", "color", "visual", "scene"],
    "speech": ["phoneme", "syllable", "speech", "pronunciation", "prosody", "voice"],
    "legal": ["statute", "precedent", "jurisdiction", "plaintiff", "defendant",
              "tort", "contract", "constitution", "amendment", "judicial"],
}


# ---------------------------------------------------------------------------
# DataSkeptic — grades every training example
# ---------------------------------------------------------------------------

class DataSkeptic:
    """
    Grade every training example on 7 dimensions.
    ALL data proceeds to learning — grades calibrate confidence only.
    """

    def __init__(self, brain=None):
        self._brain = brain
        self._source_track: Dict[str, list] = {}  # source → [accuracy scores]

    def grade(self, text: str, domain: str, source_name: str = "") -> DataGrade:
        """Grade a single training example across 7 dimensions."""
        if not text or not text.strip():
            return DataGrade(accuracy=1.0, coherence=1.0, relevance=1.0)

        accuracy = self._score_accuracy(text)
        source_reliability = self._score_source(source_name)
        evidence_quality = self._score_evidence(text)
        coherence = self._score_coherence(text)
        bias = self._score_bias(text)
        relevance = self._score_relevance(text, domain)
        ethics = self._score_ethics(text)

        grade = DataGrade(
            accuracy=_clamp(accuracy),
            source_reliability=_clamp(source_reliability),
            evidence_quality=_clamp(evidence_quality),
            coherence=_clamp(coherence),
            bias=_clamp(bias),
            relevance=_clamp(relevance),
            ethics=_clamp(ethics),
        )
        return grade

    # --- Scoring sub-systems ---

    def _score_accuracy(self, text: str) -> float:
        """Heuristic accuracy scoring."""
        score = 6.0  # default moderate

        # Known false claims → heavy penalty
        if _FALSE_CLAIMS.search(text):
            score -= 4.0

        # Hedging language → slight boost (epistemic humility)
        hedges = len(_HEDGING.findall(text))
        score += min(hedges * 0.3, 1.5)

        # Absolute claims without evidence → penalty
        absolutes = len(_ABSOLUTE_CLAIMS.findall(text))
        if absolutes > 0 and hedges == 0:
            score -= min(absolutes * 0.5, 2.0)

        return score

    def _score_source(self, source_name: str) -> float:
        """Source reliability scoring."""
        if not source_name:
            return 5.0

        name_lower = source_name.lower()

        # Check known academic sources
        for s in _ACADEMIC_SOURCES:
            if s in name_lower:
                return 8.0

        # Check historical reliability
        if name_lower in self._source_track:
            hist = self._source_track[name_lower]
            if hist:
                return min(9.0, 4.0 + 5.0 * (sum(hist) / len(hist)))

        return 5.0

    def update_source_reliability(self, source_name: str, accuracy: float):
        """Update source reliability tracking based on observed accuracy."""
        if not source_name:
            return
        key = source_name.lower()
        if key not in self._source_track:
            self._source_track[key] = []
        self._source_track[key].append(accuracy)
        # Keep window bounded
        if len(self._source_track[key]) > 200:
            self._source_track[key] = self._source_track[key][-200:]

    def _score_evidence(self, text: str) -> float:
        """Evidence quality scoring."""
        score = 5.0

        if _EVIDENCE_STRONG.search(text):
            score += 3.0
        elif _EVIDENCE_MODERATE.search(text):
            score += 1.5

        if _EVIDENCE_ANECDOTAL.search(text):
            score -= 2.0

        # Short texts with assertions but no evidence
        words = text.split()
        if len(words) < 20 and _ABSOLUTE_CLAIMS.search(text):
            score -= 1.0

        return score

    def _score_coherence(self, text: str) -> float:
        """Internal consistency and quality scoring."""
        score = 7.0  # start optimistic

        words = text.split()
        num_words = len(words)

        # Too short → low substance
        if num_words < 5:
            score -= 3.0
        elif num_words < 15:
            score -= 1.0

        # Repetition ratio
        if num_words > 10:
            unique_ratio = len(set(w.lower() for w in words)) / num_words
            if unique_ratio < 0.2:
                score -= 3.0
            elif unique_ratio < 0.4:
                score -= 1.5

        # Check for garbled text (high non-alpha ratio)
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = max(len(text), 1)
        if total_chars > 20 and alpha_chars / total_chars < 0.4:
            score -= 2.0

        return score

    def _score_bias(self, text: str) -> float:
        """Bias freedom scoring (10=unbiased, 0=heavily biased)."""
        score = 7.0

        # Emotional manipulation
        manip_count = len(_EMOTIONAL_MANIPULATION.findall(text))
        score -= min(manip_count * 0.5, 3.0)

        # Logical fallacies
        if _FALLACIES.search(text):
            score -= 2.0

        # One-sided absolute claims
        if _ABSOLUTE_CLAIMS.search(text) and not _HEDGING.search(text):
            score -= 1.0

        return score

    def _score_relevance(self, text: str, domain: str) -> float:
        """Domain relevance scoring."""
        keywords = _DOMAIN_KEYWORDS.get(domain, [])
        if not keywords:
            return 5.0  # unknown domain

        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        # Scale: 0 matches → 3.0, 1 → 5.0, 2 → 6.5, 3+ → 8.0+
        if matches == 0:
            return 3.0
        if matches == 1:
            return 5.0
        if matches == 2:
            return 6.5
        return min(10.0, 6.5 + (matches - 2) * 0.5)

    def _score_ethics(self, text: str) -> float:
        """Ethics scoring — malignant/neutral/beneficial bands."""
        score = 6.5  # default neutral

        # Check known malignant works
        penalty = 0.0
        for pattern, pen in _MALIGNANT_PATTERNS:
            if pattern.search(text):
                penalty += pen
        if penalty > 0:
            score = max(0.0, 3.33 - penalty)
            return score

        # Dehumanizing language
        if _DEHUMANIZING.search(text):
            score = min(score, 2.0)

        # Manipulation/propaganda
        if _MANIPULATION.search(text):
            score -= 2.5

        # Educational/prosocial markers → boost
        if re.search(r"\b(learn|understand|discover|teach|explain|"
                     r"research|study|evidence|knowledge)\b", text, re.I):
            score += 1.0

        # Scientific/constructive
        if _EVIDENCE_STRONG.search(text):
            score += 0.5

        return score


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp(val: float, lo: float = 0.0, hi: float = 10.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, val))
