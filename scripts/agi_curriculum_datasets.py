#!/usr/bin/env python3
"""
NIMCP AGI Curriculum Dataset Configuration
Comprehensive streaming datasets for training a biologically-inspired AGI from infancy.

Philosophy: Train like raising a child through developmental stages:
- Infant (0-2): Sensory patterns, basic associations, simple language
- Child (2-7): Language acquisition, basic concepts, social learning
- Adolescent (7-18): Abstract reasoning, formal education, ethics
- Adult (18+): Specialized knowledge, synthesis, wisdom

Domain Coverage:
- Sciences: Physics, Chemistry, Biology, Mathematics
- Humanities: Philosophy, Literature, History, Psychology
- Technical: Software Engineering, Computer Science
- Reasoning: Logic, Common Sense, Spatial, Causal
- Social: Ethics, Emotional Intelligence, Theory of Mind
- World Knowledge: Encyclopedia, Facts, Current Events
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum, auto
from pathlib import Path

#=============================================================================
# Dataset Configuration Types
#=============================================================================

class DatasetSource(Enum):
    """Where datasets can be streamed from"""
    HUGGINGFACE = "huggingface"      # Hugging Face datasets (primary)
    KAGGLE = "kaggle"                # Kaggle datasets
    DIRECT_URL = "url"               # Direct download URL
    LOCAL = "local"                  # Local preprocessed files
    API = "api"                      # API-based streaming


class DevelopmentStage(Enum):
    """Developmental stages for curriculum learning"""
    INFANT = "infant"           # 0-2 years equivalent
    CHILD = "child"             # 2-7 years equivalent
    ADOLESCENT = "adolescent"   # 7-18 years equivalent
    ADULT = "adult"             # 18+ years equivalent
    CONTINUOUS = "continuous"   # Ongoing learning at any stage


class DomainCategory(Enum):
    """High-level domain categories"""
    # Sciences
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MATHEMATICS = "mathematics"

    # Humanities
    PHILOSOPHY = "philosophy"
    LITERATURE = "literature"
    HISTORY = "history"
    PSYCHOLOGY = "psychology"

    # Technical
    COMPUTER_SCIENCE = "computer_science"
    SOFTWARE_ENGINEERING = "software_engineering"

    # Reasoning & Cognition
    LOGIC = "logic"
    COMMON_SENSE = "common_sense"
    SPATIAL_REASONING = "spatial_reasoning"
    CAUSAL_REASONING = "causal_reasoning"

    # Social & Ethical
    ETHICS = "ethics"
    SOCIAL_INTELLIGENCE = "social_intelligence"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"

    # World Knowledge
    ENCYCLOPEDIA = "encyclopedia"
    LANGUAGE = "language"
    MULTIMODAL = "multimodal"

    # Finance & Economics (NEW)
    FINANCE = "finance"
    ECONOMICS = "economics"

    # Earth & Climate Sciences (NEW)
    CLIMATE = "climate"
    WEATHER = "weather"
    EARTH_SCIENCE = "earth_science"

    # Academic & Research (NEW)
    ACADEMIC_RESEARCH = "academic_research"


@dataclass
class StreamingDataset:
    """Configuration for a streaming dataset"""
    name: str                                   # Human-readable name
    source: DatasetSource                       # Where to stream from
    identifier: str                             # HF dataset name, URL, etc.
    domain: DomainCategory                      # Primary domain
    stages: List[DevelopmentStage]              # Which stages to use this in
    difficulty: float                           # 0.0 (easy) to 1.0 (hard)
    priority: int                               # 1 (highest) to 5 (lowest)

    # Optional configuration
    subset: Optional[str] = None                # Dataset subset/config
    split: str = "train"                        # train/validation/test
    streaming: bool = True                      # Enable streaming
    max_samples: Optional[int] = None           # Limit samples (None = all)

    # Processing
    text_field: str = "text"                    # Field containing text
    label_field: Optional[str] = None           # Field containing labels
    transform_fn: Optional[str] = None          # Custom transform function name

    # Metadata
    description: str = ""
    license: str = "unknown"
    size_estimate: str = "unknown"              # Approximate size
    url: str = ""                               # Documentation URL


#=============================================================================
# COMPREHENSIVE DATASET CATALOG
#=============================================================================

AGI_CURRICULUM_DATASETS: List[StreamingDataset] = [

    #=========================================================================
    # WORLD KNOWLEDGE - Foundation for everything
    #=========================================================================

    StreamingDataset(
        name="Wikipedia (English)",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.ENCYCLOPEDIA,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=1,
        text_field="text",
        description="65M+ articles covering all human knowledge",
        license="CC-BY-SA",
        size_estimate="21GB",
        url="https://huggingface.co/datasets/wikimedia/wikipedia"
    ),

    StreamingDataset(
        name="Wikipedia Simple English",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.simple",
        domain=DomainCategory.ENCYCLOPEDIA,
        stages=[DevelopmentStage.INFANT, DevelopmentStage.CHILD],
        difficulty=0.2,
        priority=1,
        text_field="text",
        description="Simplified Wikipedia for early learning",
        license="CC-BY-SA",
        size_estimate="200MB"
    ),

    #=========================================================================
    # SCIENCES - Physics, Chemistry, Biology, Math
    #=========================================================================

    # Physics
    StreamingDataset(
        name="ScienceQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="derek-thomas/ScienceQA",
        domain=DomainCategory.PHYSICS,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=1,
        text_field="question",
        label_field="answer",
        description="21K multimodal science questions (physics, chemistry, biology)",
        license="CC-BY-NC-SA",
        size_estimate="500MB",
        url="https://huggingface.co/datasets/derek-thomas/ScienceQA"
    ),

    StreamingDataset(
        name="ScienceOlympiad",
        source=DatasetSource.HUGGINGFACE,
        identifier="ByteDance-Seed/ScienceOlympiad",
        domain=DomainCategory.PHYSICS,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.9,
        priority=2,
        description="Competition-level physics and chemistry problems",
        license="MIT",
        url="https://huggingface.co/datasets/ByteDance-Seed/ScienceOlympiad"
    ),

    StreamingDataset(
        name="MMMU Science",
        source=DatasetSource.HUGGINGFACE,
        identifier="MMMU/MMMU",
        domain=DomainCategory.PHYSICS,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=2,
        description="Multimodal university-level science (Physics, Chemistry, Biology)",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/MMMU/MMMU"
    ),

    # Mathematics
    StreamingDataset(
        name="GSM8K",
        source=DatasetSource.HUGGINGFACE,
        identifier="openai/gsm8k",
        subset="main",
        domain=DomainCategory.MATHEMATICS,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=1,
        text_field="question",
        label_field="answer",
        description="8.5K grade school math problems with reasoning steps",
        license="MIT",
        size_estimate="10MB",
        url="https://huggingface.co/datasets/openai/gsm8k"
    ),

    StreamingDataset(
        name="MATH Competition",
        source=DatasetSource.HUGGINGFACE,
        identifier="hendrycks/competition_math",
        domain=DomainCategory.MATHEMATICS,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=1,
        text_field="problem",
        label_field="solution",
        description="12.5K competition math problems (AMC, AIME, Olympiad)",
        license="MIT",
        size_estimate="50MB",
        url="https://huggingface.co/datasets/hendrycks/competition_math"
    ),

    StreamingDataset(
        name="DeepMind Math",
        source=DatasetSource.HUGGINGFACE,
        identifier="deepmind/math_dataset",
        domain=DomainCategory.MATHEMATICS,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.5,
        priority=2,
        description="Procedurally generated math problems at school level",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/deepmind/math_dataset"
    ),

    StreamingDataset(
        name="Big-Math",
        source=DatasetSource.HUGGINGFACE,
        identifier="SynthLabsAI/Big-Math-RL-Verified",
        domain=DomainCategory.MATHEMATICS,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        description="250K math problems for reinforcement learning",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified"
    ),

    StreamingDataset(
        name="NuminaMath",
        source=DatasetSource.HUGGINGFACE,
        identifier="AI-MO/NuminaMath-CoT",
        domain=DomainCategory.MATHEMATICS,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.75,
        priority=2,
        description="860K math problem-solution pairs with chain-of-thought",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/AI-MO/NuminaMath-CoT"
    ),

    # Biology
    StreamingDataset(
        name="PubMedQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="bigbio/pubmed_qa",
        domain=DomainCategory.BIOLOGY,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        description="Biomedical question answering from PubMed abstracts",
        license="MIT",
        url="https://huggingface.co/datasets/bigbio/pubmed_qa"
    ),

    StreamingDataset(
        name="MedQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="bigbio/med_qa",
        domain=DomainCategory.BIOLOGY,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=3,
        description="Medical licensing exam questions (USMLE)",
        license="MIT"
    ),

    #=========================================================================
    # REASONING & COMMON SENSE
    #=========================================================================

    StreamingDataset(
        name="ConceptNet",
        source=DatasetSource.HUGGINGFACE,
        identifier="lmind/conceptnet5",
        domain=DomainCategory.COMMON_SENSE,
        stages=[DevelopmentStage.INFANT, DevelopmentStage.CHILD],
        difficulty=0.3,
        priority=1,
        description="3.4M commonsense knowledge tuples",
        license="CC-BY-4.0",
        url="https://conceptnet.io/"
    ),

    StreamingDataset(
        name="ATOMIC 2020",
        source=DatasetSource.HUGGINGFACE,
        identifier="allenai/atomic",
        domain=DomainCategory.COMMON_SENSE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.5,
        priority=1,
        description="1.3M social commonsense if-then knowledge",
        license="CC-BY-4.0"
    ),

    StreamingDataset(
        name="ARC Challenge",
        source=DatasetSource.HUGGINGFACE,
        identifier="allenai/ai2_arc",
        subset="ARC-Challenge",
        domain=DomainCategory.LOGIC,
        stages=[DevelopmentStage.ADOLESCENT],
        difficulty=0.6,
        priority=1,
        text_field="question",
        label_field="answerKey",
        description="7.7K grade-school science questions requiring reasoning",
        license="CC-BY-SA-4.0",
        url="https://huggingface.co/datasets/allenai/ai2_arc"
    ),

    StreamingDataset(
        name="OpenBookQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="allenai/openbookqa",
        domain=DomainCategory.LOGIC,
        stages=[DevelopmentStage.ADOLESCENT],
        difficulty=0.6,
        priority=1,
        description="Open book exam style questions requiring multi-step reasoning",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/allenai/openbookqa"
    ),

    StreamingDataset(
        name="CommonsenseQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="tau/commonsense_qa",
        domain=DomainCategory.COMMON_SENSE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.5,
        priority=1,
        description="12K questions requiring commonsense reasoning",
        license="MIT"
    ),

    StreamingDataset(
        name="StrategyQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="wics/strategy-qa",
        domain=DomainCategory.CAUSAL_REASONING,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        description="Questions requiring implicit multi-step reasoning",
        license="MIT"
    ),

    StreamingDataset(
        name="WinoGrande",
        source=DatasetSource.HUGGINGFACE,
        identifier="allenai/winogrande",
        domain=DomainCategory.COMMON_SENSE,
        stages=[DevelopmentStage.ADOLESCENT],
        difficulty=0.6,
        priority=2,
        description="44K coreference resolution problems",
        license="Apache-2.0"
    ),

    StreamingDataset(
        name="PIQA (Physical)",
        source=DatasetSource.HUGGINGFACE,
        identifier="ybisk/piqa",
        domain=DomainCategory.SPATIAL_REASONING,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.5,
        priority=2,
        description="Physical intuition and interaction reasoning",
        license="CC-BY-4.0"
    ),

    StreamingDataset(
        name="SocialIQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="allenai/social_i_qa",
        domain=DomainCategory.SOCIAL_INTELLIGENCE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.5,
        priority=1,
        description="38K questions about social situations and emotional intelligence",
        license="CC-BY-4.0"
    ),

    #=========================================================================
    # PHILOSOPHY & ETHICS
    #=========================================================================

    StreamingDataset(
        name="Ethics Dataset",
        source=DatasetSource.HUGGINGFACE,
        identifier="hendrycks/ethics",
        domain=DomainCategory.ETHICS,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=1,
        description="130K ethics scenarios covering justice, deontology, virtue, utilitarianism",
        license="MIT"
    ),

    StreamingDataset(
        name="Moral Stories",
        source=DatasetSource.HUGGINGFACE,
        identifier="demelin/moral_stories",
        domain=DomainCategory.ETHICS,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=2,
        description="12K stories with moral lessons",
        license="CC-BY-4.0"
    ),

    StreamingDataset(
        name="Philosophy Papers (PhilPapers)",
        source=DatasetSource.HUGGINGFACE,
        identifier="AiresPucwortsPlayers/philosophy_papers",
        domain=DomainCategory.PHILOSOPHY,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.9,
        priority=3,
        description="Academic philosophy papers and abstracts",
        license="Various"
    ),

    #=========================================================================
    # LITERATURE & HUMANITIES
    #=========================================================================

    StreamingDataset(
        name="Project Gutenberg",
        source=DatasetSource.HUGGINGFACE,
        identifier="manu/project_gutenberg",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=1,
        description="70K+ public domain books - classical literature",
        license="Public Domain",
        url="https://www.gutenberg.org/"
    ),

    StreamingDataset(
        name="BookCorpus",
        source=DatasetSource.HUGGINGFACE,
        identifier="bookcorpus/bookcorpus",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="11K unpublished books - diverse fiction and non-fiction",
        license="Varies"
    ),

    StreamingDataset(
        name="Children's Book Test",
        source=DatasetSource.HUGGINGFACE,
        identifier="cbt",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.INFANT, DevelopmentStage.CHILD],
        difficulty=0.2,
        priority=1,
        description="Children's books for language learning",
        license="Public Domain"
    ),

    StreamingDataset(
        name="TinyStories",
        source=DatasetSource.HUGGINGFACE,
        identifier="roneneldan/TinyStories",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.INFANT, DevelopmentStage.CHILD],
        difficulty=0.1,
        priority=1,
        description="2.1M synthetic short stories for children (simple vocab)",
        license="MIT",
        url="https://huggingface.co/datasets/roneneldan/TinyStories"
    ),

    #=========================================================================
    # HISTORY
    #=========================================================================

    StreamingDataset(
        name="Wikipedia History Articles",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.HISTORY,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        transform_fn="filter_history_articles",
        description="History-related Wikipedia articles",
        license="CC-BY-SA"
    ),

    #=========================================================================
    # SOFTWARE ENGINEERING & COMPUTER SCIENCE
    #=========================================================================

    StreamingDataset(
        name="The Stack",
        source=DatasetSource.HUGGINGFACE,
        identifier="bigcode/the-stack",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        description="6TB of permissively licensed source code",
        license="Various OSI-approved",
        size_estimate="6TB",
        url="https://huggingface.co/datasets/bigcode/the-stack"
    ),

    StreamingDataset(
        name="Stack Overflow QA",
        source=DatasetSource.HUGGINGFACE,
        identifier="koutch/stackoverflow_python",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="Stack Overflow programming Q&A",
        license="CC-BY-SA"
    ),

    StreamingDataset(
        name="CodeSearchNet",
        source=DatasetSource.HUGGINGFACE,
        identifier="code_search_net",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=3,
        description="2M code-comment pairs for understanding code",
        license="Various",
        url="https://github.com/github/CodeSearchNet"
    ),

    StreamingDataset(
        name="TACO (Algorithmic Code)",
        source=DatasetSource.HUGGINGFACE,
        identifier="BAAI/TACO",
        domain=DomainCategory.COMPUTER_SCIENCE,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=3,
        description="26K programming competition problems",
        license="MIT",
        url="https://github.com/FlagOpen/TACO"
    ),

    #=========================================================================
    # LANGUAGE & COMMUNICATION
    #=========================================================================

    StreamingDataset(
        name="C4 (Common Crawl)",
        source=DatasetSource.HUGGINGFACE,
        identifier="allenai/c4",
        domain=DomainCategory.LANGUAGE,
        stages=[DevelopmentStage.CONTINUOUS],
        difficulty=0.4,
        priority=1,
        description="750GB cleaned web text for language understanding",
        license="ODC-BY",
        size_estimate="750GB"
    ),

    StreamingDataset(
        name="FineWeb",
        source=DatasetSource.HUGGINGFACE,
        identifier="HuggingFaceFW/fineweb",
        domain=DomainCategory.LANGUAGE,
        stages=[DevelopmentStage.CONTINUOUS],
        difficulty=0.4,
        priority=1,
        description="15T tokens of high-quality web text",
        license="ODC-BY",
        size_estimate="45TB",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb"
    ),

    StreamingDataset(
        name="RedPajama",
        source=DatasetSource.HUGGINGFACE,
        identifier="togethercomputer/RedPajama-Data-1T",
        domain=DomainCategory.LANGUAGE,
        stages=[DevelopmentStage.CONTINUOUS],
        difficulty=0.4,
        priority=2,
        description="1.2T tokens balanced across domains",
        license="Apache-2.0",
        size_estimate="5TB"
    ),

    #=========================================================================
    # EMOTIONAL INTELLIGENCE & PSYCHOLOGY
    #=========================================================================

    StreamingDataset(
        name="Emotions Dataset",
        source=DatasetSource.HUGGINGFACE,
        identifier="boltuix/emotions-dataset",
        domain=DomainCategory.EMOTIONAL_INTELLIGENCE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=2,
        description="131K text entries labeled with 13 emotions",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/boltuix/emotions-dataset"
    ),

    StreamingDataset(
        name="EmpatheticDialogues",
        source=DatasetSource.HUGGINGFACE,
        identifier="empathetic_dialogues",
        domain=DomainCategory.EMOTIONAL_INTELLIGENCE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=2,
        description="25K conversations grounded in emotional situations",
        license="CC-BY-NC-4.0"
    ),

    StreamingDataset(
        name="DailyDialog",
        source=DatasetSource.HUGGINGFACE,
        identifier="daily_dialog",
        domain=DomainCategory.SOCIAL_INTELLIGENCE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.3,
        priority=2,
        description="13K dialogues covering daily communication topics",
        license="CC-BY-NC-SA-4.0"
    ),

    #=========================================================================
    # MULTIMODAL (Future expansion)
    #=========================================================================

    StreamingDataset(
        name="Conceptual Captions",
        source=DatasetSource.HUGGINGFACE,
        identifier="conceptual_captions",
        domain=DomainCategory.MULTIMODAL,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=3,
        description="3.3M image-caption pairs for visual understanding",
        license="Custom"
    ),

    #=========================================================================
    # WIKIPEDIA - Domain-Specific Filtered Views
    #=========================================================================

    StreamingDataset(
        name="Wikipedia Philosophy",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.PHILOSOPHY,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        transform_fn="filter_philosophy_articles",
        description="Philosophy-related Wikipedia articles (epistemology, ethics, metaphysics)",
        license="CC-BY-SA"
    ),

    StreamingDataset(
        name="Wikipedia Physics",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.PHYSICS,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        transform_fn="filter_physics_articles",
        description="Physics-related Wikipedia articles (mechanics, thermodynamics, quantum)",
        license="CC-BY-SA"
    ),

    StreamingDataset(
        name="Wikipedia Chemistry",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.CHEMISTRY,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        transform_fn="filter_chemistry_articles",
        description="Chemistry-related Wikipedia articles (elements, compounds, reactions)",
        license="CC-BY-SA"
    ),

    StreamingDataset(
        name="Wikipedia Biology",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.BIOLOGY,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        transform_fn="filter_biology_articles",
        description="Biology-related Wikipedia articles (genetics, evolution, anatomy)",
        license="CC-BY-SA"
    ),

    StreamingDataset(
        name="Wikipedia Psychology",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.PSYCHOLOGY,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        transform_fn="filter_psychology_articles",
        description="Psychology-related Wikipedia articles (cognition, behavior, mental health)",
        license="CC-BY-SA"
    ),

    StreamingDataset(
        name="Wikipedia Mathematics",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.MATHEMATICS,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        transform_fn="filter_mathematics_articles",
        description="Mathematics-related Wikipedia articles (algebra, calculus, topology)",
        license="CC-BY-SA"
    ),

    StreamingDataset(
        name="Wikipedia Computer Science",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.COMPUTER_SCIENCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        transform_fn="filter_cs_articles",
        description="CS-related Wikipedia articles (algorithms, data structures, systems)",
        license="CC-BY-SA"
    ),

    StreamingDataset(
        name="Wikipedia Literature",
        source=DatasetSource.HUGGINGFACE,
        identifier="wikimedia/wikipedia",
        subset="20231101.en",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        transform_fn="filter_literature_articles",
        description="Literature-related Wikipedia articles (authors, works, literary movements)",
        license="CC-BY-SA"
    ),

    #=========================================================================
    # KAGGLE DATASETS - Philosophy, Psychology, Emotions
    #=========================================================================

    StreamingDataset(
        name="History of Philosophy (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="kouroshalizadeh/history-of-philosophy",
        domain=DomainCategory.PHILOSOPHY,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        text_field="sentence_str",
        description="300K+ sentences from major philosophical works spanning 2500 years",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/kouroshalizadeh/history-of-philosophy"
    ),

    StreamingDataset(
        name="Philosophical Texts (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="christopherlemke/philosophical-texts",
        domain=DomainCategory.PHILOSOPHY,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=3,
        description="Collection of classical philosophical texts from major philosophers",
        license="Public Domain",
        url="https://www.kaggle.com/datasets/christopherlemke/philosophical-texts"
    ),

    StreamingDataset(
        name="Emotions for NLP (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="praveengovi/emotions-dataset-for-nlp",
        domain=DomainCategory.EMOTIONAL_INTELLIGENCE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=2,
        text_field="text",
        label_field="emotion",
        description="34K text samples labeled with 6 emotion categories",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp"
    ),

    StreamingDataset(
        name="Human Emotions Dataset (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="muhammadhananasghar/human-emotions-datasethes",
        domain=DomainCategory.EMOTIONAL_INTELLIGENCE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=3,
        description="Human emotions dataset for sentiment and emotion analysis",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes"
    ),

    StreamingDataset(
        name="Social Media Mental Health (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="souvikahmed071/social-media-and-mental-health",
        domain=DomainCategory.PSYCHOLOGY,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=3,
        description="Social media data for understanding mental health patterns",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health"
    ),

    StreamingDataset(
        name="GoEmotions",
        source=DatasetSource.HUGGINGFACE,
        identifier="google-research-datasets/go_emotions",
        domain=DomainCategory.EMOTIONAL_INTELLIGENCE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=2,
        text_field="text",
        label_field="labels",
        description="58K Reddit comments labeled with 27 emotion categories",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/google-research-datasets/go_emotions"
    ),

    #=========================================================================
    # ADDITIONAL SCIENCE DATASETS
    #=========================================================================

    StreamingDataset(
        name="SciQ",
        source=DatasetSource.HUGGINGFACE,
        identifier="allenai/sciq",
        domain=DomainCategory.PHYSICS,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=2,
        text_field="question",
        label_field="correct_answer",
        description="13.7K crowdsourced science exam questions",
        license="CC-BY-NC-3.0",
        url="https://huggingface.co/datasets/allenai/sciq"
    ),

    StreamingDataset(
        name="QASC (Science Composition)",
        source=DatasetSource.HUGGINGFACE,
        identifier="allenai/qasc",
        domain=DomainCategory.PHYSICS,
        stages=[DevelopmentStage.ADOLESCENT],
        difficulty=0.6,
        priority=2,
        description="10K questions requiring composition of science facts",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/allenai/qasc"
    ),

    #=========================================================================
    # ADDITIONAL REASONING DATASETS
    #=========================================================================

    StreamingDataset(
        name="BoolQ",
        source=DatasetSource.HUGGINGFACE,
        identifier="google/boolq",
        domain=DomainCategory.LOGIC,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=2,
        text_field="question",
        description="16K yes/no questions requiring reasoning over passages",
        license="CC-BY-SA-3.0"
    ),

    StreamingDataset(
        name="HellaSwag",
        source=DatasetSource.HUGGINGFACE,
        identifier="Rowan/hellaswag",
        domain=DomainCategory.COMMON_SENSE,
        stages=[DevelopmentStage.ADOLESCENT],
        difficulty=0.6,
        priority=2,
        description="70K commonsense inference completions",
        license="MIT",
        url="https://huggingface.co/datasets/Rowan/hellaswag"
    ),

    StreamingDataset(
        name="CosmosQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="cosmos_qa",
        domain=DomainCategory.CAUSAL_REASONING,
        stages=[DevelopmentStage.ADOLESCENT],
        difficulty=0.6,
        priority=2,
        description="35K reading comprehension questions requiring contextual reasoning",
        license="CC-BY-4.0"
    ),

    StreamingDataset(
        name="DROP (Numerical Reasoning)",
        source=DatasetSource.HUGGINGFACE,
        identifier="ucinlp/drop",
        domain=DomainCategory.MATHEMATICS,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        description="96K questions requiring discrete reasoning over passages",
        license="CC-BY-SA-4.0",
        url="https://huggingface.co/datasets/ucinlp/drop"
    ),

    #=========================================================================
    # WORLD KNOWLEDGE & FACTS
    #=========================================================================

    StreamingDataset(
        name="Natural Questions",
        source=DatasetSource.HUGGINGFACE,
        identifier="google-research-datasets/natural_questions",
        domain=DomainCategory.ENCYCLOPEDIA,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="300K+ real Google search questions with Wikipedia answers",
        license="CC-BY-SA-3.0",
        url="https://huggingface.co/datasets/google-research-datasets/natural_questions"
    ),

    StreamingDataset(
        name="TriviaQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="trivia_qa",
        subset="unfiltered",
        domain=DomainCategory.ENCYCLOPEDIA,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="95K trivia question-answer pairs with evidence documents",
        license="Apache-2.0"
    ),

    StreamingDataset(
        name="HotpotQA",
        source=DatasetSource.HUGGINGFACE,
        identifier="hotpot_qa",
        subset="fullwiki",
        domain=DomainCategory.ENCYCLOPEDIA,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        description="113K multi-hop questions requiring reasoning across documents",
        license="CC-BY-SA-4.0"
    ),

    StreamingDataset(
        name="WebQuestions",
        source=DatasetSource.HUGGINGFACE,
        identifier="web_questions",
        domain=DomainCategory.ENCYCLOPEDIA,
        stages=[DevelopmentStage.ADOLESCENT],
        difficulty=0.5,
        priority=3,
        description="6K questions answerable using Freebase",
        license="CC-BY-4.0"
    ),

    #=========================================================================
    # FINANCE & ECONOMICS - Time Series, Markets, Trading
    #=========================================================================

    # Investopedia (Financial Education)
    StreamingDataset(
        name="Investopedia Embeddings",
        source=DatasetSource.HUGGINGFACE,
        identifier="FinLang/investopedia-embedding-dataset",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=1,
        text_field="Question",
        description="Financial education data from Investopedia with Q&A pairs",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/FinLang/investopedia-embedding-dataset"
    ),

    StreamingDataset(
        name="Investopedia Articles",
        source=DatasetSource.HUGGINGFACE,
        identifier="ksrepo/investopedia-dataset",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="4.7K Investopedia articles on finance and investing",
        license="Various",
        url="https://huggingface.co/datasets/ksrepo/investopedia-dataset"
    ),

    # Financial News Time Series
    StreamingDataset(
        name="FNSPID (Financial News Time Series)",
        source=DatasetSource.HUGGINGFACE,
        identifier="Zihan1004/FNSPID",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=1,
        description="10M+ financial news entries with S&P500 time series data",
        license="MIT",
        url="https://huggingface.co/datasets/Zihan1004/FNSPID"
    ),

    StreamingDataset(
        name="Financial News Multi-Source",
        source=DatasetSource.HUGGINGFACE,
        identifier="Brianferrell787/financial-news-multisource",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=1,
        description="57M+ rows of financial news from 1990-2025 across 24 subsets",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/Brianferrell787/financial-news-multisource"
    ),

    StreamingDataset(
        name="NIFTY (News-Informed Financial Trends)",
        source=DatasetSource.HUGGINGFACE,
        identifier="raeidsaqur/NIFTY",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        description="News-informed financial trend yield dataset for forecasting",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/raeidsaqur/NIFTY"
    ),

    StreamingDataset(
        name="S&P500 News Time Series",
        source=DatasetSource.HUGGINGFACE,
        identifier="KrossKinetic/SP500-Financial-News-Articles-Time-Series",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        description="S&P500 financial news articles with time series alignment",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/KrossKinetic/SP500-Financial-News-Articles-Time-Series"
    ),

    StreamingDataset(
        name="Financial Sentiment",
        source=DatasetSource.HUGGINGFACE,
        identifier="chiapudding/kaggle-financial-sentiment",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        text_field="text",
        label_field="sentiment",
        description="Financial news sentiment analysis dataset",
        license="CC-BY-4.0"
    ),

    StreamingDataset(
        name="Finance-Alpaca",
        source=DatasetSource.HUGGINGFACE,
        identifier="gbharti/finance-alpaca",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="68K finance Q&A pairs for instruction tuning",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/gbharti/finance-alpaca"
    ),

    StreamingDataset(
        name="Multimodal Financial Forecasting",
        source=DatasetSource.HUGGINGFACE,
        identifier="Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=3,
        description="S&P500 and HS300 multimodal data (text, images, tables, time series)",
        license="MIT",
        url="https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting"
    ),

    # Kaggle Financial Datasets
    StreamingDataset(
        name="S&P 500 Stock Data (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="camnugent/sandp500",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        description="Historical S&P 500 stock prices and fundamentals",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/camnugent/sandp500"
    ),

    StreamingDataset(
        name="S&P 500 Historical Data (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="henryhan117/sp-500-historical-data",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        description="Long-term S&P 500 historical price data",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/henryhan117/sp-500-historical-data"
    ),

    StreamingDataset(
        name="Bitcoin Historical Data (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="mczielinski/bitcoin-historical-data",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="Bitcoin minute-by-minute historical price data",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data"
    ),

    StreamingDataset(
        name="Bitcoin and S&P 500 Prices (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="linhanphm/bitcoin-and-s-and-p-500-historical-prices",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="Combined Bitcoin and S&P 500 historical prices for correlation analysis",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/linhanphm/bitcoin-and-s-and-p-500-historical-prices"
    ),

    StreamingDataset(
        name="Bitcoin, Gold, Oil, S&P 500 (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="prasertk/bitcon-gold-oil-sp-500",
        domain=DomainCategory.FINANCE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="Daily prices of Bitcoin, Gold, Oil, and S&P 500 (2010-2022)",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/prasertk/bitcon-gold-oil-sp-500"
    ),

    #=========================================================================
    # CLIMATE & WEATHER - Earth Science, Temperature, Environment
    #=========================================================================

    # HuggingFace Climate Datasets
    StreamingDataset(
        name="Climate Policy Radar",
        source=DatasetSource.HUGGINGFACE,
        identifier="ClimatePolicyRadar/all-document-text-data",
        domain=DomainCategory.CLIMATE,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=1,
        description="Full text of climate laws, policies, and NDCs from all countries",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/ClimatePolicyRadar/all-document-text-data"
    ),

    StreamingDataset(
        name="ClimateBERT Detection",
        source=DatasetSource.HUGGINGFACE,
        identifier="climatebert/climate_detection",
        domain=DomainCategory.CLIMATE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="Climate-related text detection dataset",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/climatebert/climate_detection"
    ),

    StreamingDataset(
        name="ClimateBERT Sentiment",
        source=DatasetSource.HUGGINGFACE,
        identifier="climatebert/climate_sentiment",
        domain=DomainCategory.CLIMATE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        text_field="text",
        label_field="label",
        description="Climate sentiment analysis dataset",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/climatebert/climate_sentiment"
    ),

    StreamingDataset(
        name="Climate FEVER",
        source=DatasetSource.HUGGINGFACE,
        identifier="tdiggelm/climate_fever",
        domain=DomainCategory.CLIMATE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        description="Climate fact verification dataset (SUPPORTS/REFUTES/NOT_ENOUGH_INFO)",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/tdiggelm/climate_fever"
    ),

    # Kaggle Climate Datasets
    StreamingDataset(
        name="Berkeley Earth Surface Temperature (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="berkeleyearth/climate-change-earth-surface-temperature-data",
        domain=DomainCategory.CLIMATE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=1,
        description="Global surface temperatures since 1750 from Berkeley Earth",
        license="CC-BY-NC-SA-4.0",
        url="https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data"
    ),

    StreamingDataset(
        name="Global Warming 195 Countries (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="ankushpanday1/global-warming-dataset-195-countries-1900-2023",
        domain=DomainCategory.CLIMATE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=1,
        description="Environmental and socioeconomic indicators for 195 countries (1900-2023)",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/ankushpanday1/global-warming-dataset-195-countries-1900-2023"
    ),

    StreamingDataset(
        name="World Bank Climate Data (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="theworldbank/world-bank-climate-change-data",
        domain=DomainCategory.CLIMATE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="World Bank climate change indicators and statistics",
        license="CC-BY-4.0",
        url="https://www.kaggle.com/datasets/theworldbank/world-bank-climate-change-data"
    ),

    StreamingDataset(
        name="Global Earth Temperatures (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="joebeachcapital/global-earth-temperatures",
        domain=DomainCategory.CLIMATE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="High-resolution global monthly temperature averages (1850-2022)",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/joebeachcapital/global-earth-temperatures"
    ),

    StreamingDataset(
        name="Climate Change Dataset 2020-2024 (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="atifmasih/climate-change-dataset2020-2024",
        domain=DomainCategory.CLIMATE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.4,
        priority=2,
        description="Recent climate change data from 2020-2024",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/atifmasih/climate-change-dataset2020-2024"
    ),

    # Weather Datasets
    StreamingDataset(
        name="Weather Data 2024 (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="sheemazain/weather-data-2024",
        domain=DomainCategory.WEATHER,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=2,
        description="Weather observations and forecasts for 2024",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/sheemazain/weather-data-2024"
    ),

    StreamingDataset(
        name="AWS NOAA Weather Data (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="goswamimurari/aws-open-source-weather-transaction-and-metadata",
        domain=DomainCategory.WEATHER,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="NOAA public weather data from AWS S3 with daily granularity",
        license="Public Domain",
        url="https://www.kaggle.com/datasets/goswamimurari/aws-open-source-weather-transaction-and-metadata"
    ),

    #=========================================================================
    # ARXIV - Academic Research Papers
    #=========================================================================

    StreamingDataset(
        name="arXiv Papers (Cornell)",
        source=DatasetSource.KAGGLE,
        identifier="Cornell-University/arxiv",
        domain=DomainCategory.ACADEMIC_RESEARCH,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=1,
        text_field="abstract",
        description="1.7M+ scientific papers from arXiv.org, updated weekly",
        license="CC0-1.0",
        size_estimate="1.1TB",
        url="https://www.kaggle.com/datasets/Cornell-University/arxiv"
    ),

    StreamingDataset(
        name="ML-ArXiv Papers",
        source=DatasetSource.HUGGINGFACE,
        identifier="CShorten/ML-ArXiv-Papers",
        domain=DomainCategory.ACADEMIC_RESEARCH,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=1,
        description="100K+ machine learning papers from arXiv",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers"
    ),

    StreamingDataset(
        name="arXiv Dataset (Community)",
        source=DatasetSource.HUGGINGFACE,
        identifier="arxiv-community/arxiv_dataset",
        domain=DomainCategory.ACADEMIC_RESEARCH,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=2,
        text_field="abstract",
        description="1.7M+ arXiv paper metadata and abstracts",
        license="CC0-1.0",
        url="https://huggingface.co/datasets/arxiv-community/arxiv_dataset"
    ),

    StreamingDataset(
        name="arXiv Abstracts 2021",
        source=DatasetSource.HUGGINGFACE,
        identifier="gfissore/arxiv-abstracts-2021",
        domain=DomainCategory.ACADEMIC_RESEARCH,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=2,
        text_field="abstract",
        description="2M+ arXiv paper abstracts through 2021",
        license="CC0-1.0",
        url="https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021"
    ),

    StreamingDataset(
        name="AI-ArXiv Papers",
        source=DatasetSource.HUGGINGFACE,
        identifier="jamescalam/ai-arxiv",
        domain=DomainCategory.ACADEMIC_RESEARCH,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=2,
        description="AI and LLM-focused papers from arXiv",
        license="Apache-2.0",
        url="https://huggingface.co/datasets/jamescalam/ai-arxiv"
    ),

    StreamingDataset(
        name="arXiv CS Papers",
        source=DatasetSource.HUGGINGFACE,
        identifier="gfissore/arxiv-cs-abstracts",
        domain=DomainCategory.COMPUTER_SCIENCE,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=2,
        text_field="abstract",
        description="Computer Science papers from arXiv",
        license="CC0-1.0",
        url="https://huggingface.co/datasets/gfissore/arxiv-cs-abstracts"
    ),

    StreamingDataset(
        name="arXiv Math Papers",
        source=DatasetSource.HUGGINGFACE,
        identifier="ccdv/arxiv-classification",
        subset="math",
        domain=DomainCategory.MATHEMATICS,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.9,
        priority=2,
        text_field="abstract",
        description="Mathematics papers from arXiv for classification",
        license="CC-BY-4.0",
        url="https://huggingface.co/datasets/ccdv/arxiv-classification"
    ),

    #=========================================================================
    # ADDITIONAL PROJECT GUTENBERG - Extended Literature
    #=========================================================================

    StreamingDataset(
        name="Gutenberg Multilingual (LAION)",
        source=DatasetSource.HUGGINGFACE,
        identifier="laion/Project-Gutenberg",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=1,
        description="Project Gutenberg books in 10 languages with metadata",
        license="Public Domain",
        url="https://huggingface.co/datasets/laion/Project-Gutenberg"
    ),

    StreamingDataset(
        name="Gutenberg English (48K Books)",
        source=DatasetSource.HUGGINGFACE,
        identifier="sedthh/gutenberg_english",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=1,
        description="48K English books from Project Gutenberg with cleaned text",
        license="Public Domain",
        url="https://huggingface.co/datasets/sedthh/gutenberg_english"
    ),

    StreamingDataset(
        name="Gutenberg Poetry Corpus",
        source=DatasetSource.HUGGINGFACE,
        identifier="biglam/gutenberg-poetry-corpus",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.4,
        priority=2,
        description="3M+ lines of poetry from Project Gutenberg (Shakespeare, Dickinson, Whitman, etc.)",
        license="Public Domain",
        url="https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus"
    ),

    StreamingDataset(
        name="PG-19 (Benchmark)",
        source=DatasetSource.HUGGINGFACE,
        identifier="pg19",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="Pre-1919 Project Gutenberg books - long-range language modeling benchmark",
        license="Public Domain",
        url="https://huggingface.co/datasets/pg19"
    ),

    StreamingDataset(
        name="Gutenberg Books 2025 (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="lokeshparab/gutenberg-books-and-metadata-2025",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=1,
        description="75K+ Project Gutenberg books with full text and comprehensive metadata",
        license="Public Domain",
        url="https://www.kaggle.com/datasets/lokeshparab/gutenberg-books-and-metadata-2025"
    ),

    StreamingDataset(
        name="Gutenberg Dialogues",
        source=DatasetSource.HUGGINGFACE,
        identifier="HuggingFaceTB/gutenberg-dialogs",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.CHILD, DevelopmentStage.ADOLESCENT],
        difficulty=0.4,
        priority=2,
        description="Extracted dialogues from Project Gutenberg books for conversation modeling",
        license="Public Domain",
        url="https://huggingface.co/datasets/HuggingFaceTB/gutenberg-dialogs"
    ),

    StreamingDataset(
        name="Standard eBooks",
        source=DatasetSource.HUGGINGFACE,
        identifier="tdoehmen/standardebooks",
        domain=DomainCategory.LITERATURE,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="High-quality curated public domain ebooks from Standard eBooks project",
        license="Public Domain",
        url="https://huggingface.co/datasets/tdoehmen/standardebooks"
    ),

    #=========================================================================
    # GITHUB - Source Code, Commits, Issues, Pull Requests
    #=========================================================================

    StreamingDataset(
        name="GitHub Code (CodeParrot)",
        source=DatasetSource.HUGGINGFACE,
        identifier="codeparrot/github-code",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=1,
        description="115M code files from GitHub in 32 programming languages (1TB)",
        license="Various OSI-approved",
        size_estimate="1TB",
        url="https://huggingface.co/datasets/codeparrot/github-code"
    ),

    StreamingDataset(
        name="GitHub Code 2025",
        source=DatasetSource.HUGGINGFACE,
        identifier="nick007x/github-code-2025",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=1,
        description="1.5M+ curated repositories representing quality and innovation in 2025",
        license="Various OSI-approved",
        url="https://huggingface.co/datasets/nick007x/github-code-2025"
    ),

    StreamingDataset(
        name="GitHub Code 2025 (2+ Stars)",
        source=DatasetSource.HUGGINGFACE,
        identifier="utter-project/github-code-2025-above-2-stars",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        description="2025 GitHub repositories with 2+ stars for quality filtering",
        license="Various OSI-approved",
        url="https://huggingface.co/datasets/utter-project/github-code-2025-above-2-stars"
    ),

    StreamingDataset(
        name="The Stack v2",
        source=DatasetSource.HUGGINGFACE,
        identifier="bigcode/the-stack-v2",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=1,
        description="Updated Stack dataset - permissively licensed source code for StarCoder2",
        license="Various OSI-approved",
        size_estimate="67TB",
        url="https://huggingface.co/datasets/bigcode/the-stack-v2"
    ),

    StreamingDataset(
        name="The Stack (Deduplicated)",
        source=DatasetSource.HUGGINGFACE,
        identifier="bigcode/the-stack-dedup",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=1,
        description="Near-deduplicated version of The Stack (3TB)",
        license="Various OSI-approved",
        size_estimate="3TB",
        url="https://huggingface.co/datasets/bigcode/the-stack-dedup"
    ),

    StreamingDataset(
        name="GitHub Issues (BigCode)",
        source=DatasetSource.HUGGINGFACE,
        identifier="bigcode/the-stack-github-issues",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=1,
        text_field="content",
        description="30M+ GitHub issues and pull request conversations",
        license="Various OSI-approved",
        size_estimate="54GB",
        url="https://huggingface.co/datasets/bigcode/the-stack-github-issues"
    ),

    StreamingDataset(
        name="CommitPack (Git Commits)",
        source=DatasetSource.HUGGINGFACE,
        identifier="bigcode/commitpack",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.7,
        priority=1,
        description="4TB of GitHub commits with diffs and commit messages",
        license="Various OSI-approved",
        size_estimate="4TB",
        url="https://huggingface.co/datasets/bigcode/commitpack"
    ),

    StreamingDataset(
        name="Git Commit Messages",
        source=DatasetSource.HUGGINGFACE,
        identifier="Tavernari/git-commit-message-dt",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="Git commit message dataset for learning conventional commits",
        license="MIT",
        url="https://huggingface.co/datasets/Tavernari/git-commit-message-dt"
    ),

    StreamingDataset(
        name="Coding Agent GitHub 2025",
        source=DatasetSource.HUGGINGFACE,
        identifier="DeepNLP/Coding-Agent-Github-2025-Feb",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=2,
        description="Coding agent repositories from GitHub including ICML/ICLR 2024 projects",
        license="Various",
        url="https://huggingface.co/datasets/DeepNLP/Coding-Agent-Github-2025-Feb"
    ),

    StreamingDataset(
        name="LiveCodeBench",
        source=DatasetSource.HUGGINGFACE,
        identifier="livecodebench/code_generation_lite",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.8,
        priority=2,
        description="880 coding problems benchmark (2023-2025), continuously updated",
        license="MIT",
        url="https://huggingface.co/datasets/livecodebench/code_generation_lite"
    ),

    StreamingDataset(
        name="GitHub Repositories 2020 (Kaggle)",
        source=DatasetSource.KAGGLE,
        identifier="vatsalparsaniya/github-repositories-analysis",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADOLESCENT, DevelopmentStage.ADULT],
        difficulty=0.5,
        priority=2,
        description="1200+ top starred GitHub repositories by domain",
        license="CC0-1.0",
        url="https://www.kaggle.com/datasets/vatsalparsaniya/github-repositories-analysis"
    ),

    StreamingDataset(
        name="GitHub Repos (Kaggle Official)",
        source=DatasetSource.KAGGLE,
        identifier="github/github-repos",
        domain=DomainCategory.SOFTWARE_ENGINEERING,
        stages=[DevelopmentStage.ADULT],
        difficulty=0.6,
        priority=2,
        description="Official GitHub repositories dataset on Kaggle",
        license="Various",
        url="https://www.kaggle.com/datasets/github/github-repos"
    ),
]


#=============================================================================
# Curriculum Organization
#=============================================================================

def get_datasets_by_stage(stage: DevelopmentStage) -> List[StreamingDataset]:
    """Get all datasets appropriate for a developmental stage"""
    return [d for d in AGI_CURRICULUM_DATASETS
            if stage in d.stages or DevelopmentStage.CONTINUOUS in d.stages]


def get_datasets_by_domain(domain: DomainCategory) -> List[StreamingDataset]:
    """Get all datasets in a domain"""
    return [d for d in AGI_CURRICULUM_DATASETS if d.domain == domain]


def get_datasets_by_priority(max_priority: int = 2) -> List[StreamingDataset]:
    """Get high-priority datasets"""
    return sorted([d for d in AGI_CURRICULUM_DATASETS if d.priority <= max_priority],
                  key=lambda x: (x.priority, x.difficulty))


def get_curriculum_schedule() -> Dict[str, Any]:
    """
    Generate a curriculum schedule organized by developmental stage.
    Returns recommended training order with sample counts.
    """
    schedule = {
        "infant": {
            "description": "Basic sensory patterns, simple language, foundational concepts",
            "age_equivalent": "0-2 years",
            "duration_samples": 500_000,
            "datasets": [],
            "domains_focus": ["simple_language", "patterns", "basic_concepts"],
            "learning_rate": 0.01,
            "consolidation_frequency": 500,
        },
        "child": {
            "description": "Language acquisition, social learning, basic reasoning",
            "age_equivalent": "2-7 years",
            "duration_samples": 2_000_000,
            "datasets": [],
            "domains_focus": ["language", "social", "basic_math", "stories"],
            "learning_rate": 0.008,
            "consolidation_frequency": 1000,
        },
        "adolescent": {
            "description": "Abstract reasoning, formal education, ethics",
            "age_equivalent": "7-18 years",
            "duration_samples": 10_000_000,
            "datasets": [],
            "domains_focus": ["science", "math", "literature", "ethics", "history"],
            "learning_rate": 0.005,
            "consolidation_frequency": 2000,
        },
        "adult": {
            "description": "Specialized knowledge, synthesis, continuous learning",
            "age_equivalent": "18+ years",
            "duration_samples": 50_000_000,
            "datasets": [],
            "domains_focus": ["all_domains", "specialization", "research"],
            "learning_rate": 0.002,
            "consolidation_frequency": 5000,
        },
    }

    # Populate datasets for each stage
    for stage_name, stage_config in schedule.items():
        stage_enum = DevelopmentStage(stage_name)
        stage_datasets = get_datasets_by_stage(stage_enum)
        # Sort by priority then difficulty
        stage_datasets.sort(key=lambda x: (x.priority, x.difficulty))
        stage_config["datasets"] = [
            {
                "name": d.name,
                "identifier": d.identifier,
                "domain": d.domain.value,
                "difficulty": d.difficulty,
                "priority": d.priority,
            }
            for d in stage_datasets
        ]

    return schedule


#=============================================================================
# Statistics and Summary
#=============================================================================

def print_curriculum_summary():
    """Print a summary of the AGI curriculum"""
    print("=" * 80)
    print("NIMCP AGI CURRICULUM - COMPREHENSIVE DATASET CATALOG")
    print("=" * 80)

    print(f"\nTotal Datasets: {len(AGI_CURRICULUM_DATASETS)}")

    # By domain
    print("\n--- DATASETS BY DOMAIN ---")
    for domain in DomainCategory:
        datasets = get_datasets_by_domain(domain)
        if datasets:
            print(f"\n{domain.value.upper()} ({len(datasets)} datasets):")
            for d in sorted(datasets, key=lambda x: x.priority):
                stages = ", ".join(s.value for s in d.stages)
                print(f"  [{d.priority}] {d.name} (difficulty: {d.difficulty}) - {stages}")

    # By stage
    print("\n--- DATASETS BY DEVELOPMENTAL STAGE ---")
    for stage in DevelopmentStage:
        if stage == DevelopmentStage.CONTINUOUS:
            continue
        datasets = get_datasets_by_stage(stage)
        print(f"\n{stage.value.upper()} ({len(datasets)} datasets):")
        for d in sorted(datasets, key=lambda x: (x.priority, x.difficulty))[:10]:
            print(f"  [{d.priority}] {d.name} ({d.domain.value})")
        if len(datasets) > 10:
            print(f"  ... and {len(datasets) - 10} more")

    # Priority datasets
    print("\n--- HIGH PRIORITY DATASETS (for quick start) ---")
    priority_datasets = get_datasets_by_priority(1)
    for d in priority_datasets:
        print(f"  * {d.name} - {d.domain.value}")

    print("\n" + "=" * 80)


def export_to_json(filepath: str):
    """Export curriculum to JSON for use by training scripts"""
    import json

    schedule = get_curriculum_schedule()

    # Add full dataset details
    full_config = {
        "version": "1.0.0",
        "description": "NIMCP AGI Curriculum - Comprehensive streaming datasets",
        "total_datasets": len(AGI_CURRICULUM_DATASETS),
        "curriculum_schedule": schedule,
        "all_datasets": [
            {
                "name": d.name,
                "source": d.source.value,
                "identifier": d.identifier,
                "subset": d.subset,
                "domain": d.domain.value,
                "stages": [s.value for s in d.stages],
                "difficulty": d.difficulty,
                "priority": d.priority,
                "streaming": d.streaming,
                "text_field": d.text_field,
                "label_field": d.label_field,
                "description": d.description,
                "license": d.license,
                "size_estimate": d.size_estimate,
                "url": d.url,
            }
            for d in AGI_CURRICULUM_DATASETS
        ]
    }

    with open(filepath, 'w') as f:
        json.dump(full_config, f, indent=2)

    print(f"Exported curriculum to {filepath}")


#=============================================================================
# Wikipedia Domain Filters
#=============================================================================

# Keywords for filtering Wikipedia articles by domain
WIKIPEDIA_DOMAIN_KEYWORDS = {
    "philosophy": [
        "philosophy", "philosopher", "epistemology", "metaphysics", "ethics",
        "ontology", "logic", "phenomenology", "existentialism", "rationalism",
        "empiricism", "idealism", "materialism", "dualism", "pragmatism",
        "stoicism", "platonism", "aristotelian", "kantian", "hegelian",
        "nietzsche", "socrates", "plato", "aristotle", "descartes", "hume",
        "locke", "leibniz", "spinoza", "wittgenstein", "heidegger", "sartre",
        "deontology", "utilitarianism", "virtue ethics", "moral", "aesthetics"
    ],
    "physics": [
        "physics", "quantum", "relativity", "thermodynamics", "mechanics",
        "electromagnetism", "particle", "wave", "energy", "force", "momentum",
        "entropy", "electron", "photon", "quark", "boson", "fermion",
        "nuclear", "atomic", "gravitational", "magnetic", "electric",
        "acceleration", "velocity", "newton", "einstein", "feynman",
        "planck", "heisenberg", "schrodinger", "maxwell", "boltzmann"
    ],
    "chemistry": [
        "chemistry", "chemical", "molecule", "atom", "element", "compound",
        "reaction", "bond", "organic", "inorganic", "polymer", "catalyst",
        "acid", "base", "oxidation", "reduction", "synthesis", "periodic",
        "electron", "ion", "covalent", "ionic", "hydrogen", "carbon",
        "oxygen", "nitrogen", "metal", "solution", "equilibrium"
    ],
    "biology": [
        "biology", "biological", "cell", "gene", "dna", "rna", "protein",
        "evolution", "species", "organism", "ecology", "ecosystem",
        "genetics", "genome", "chromosome", "mutation", "natural selection",
        "darwin", "mendel", "anatomy", "physiology", "neuroscience",
        "biochemistry", "microbiology", "immunology", "botany", "zoology"
    ],
    "psychology": [
        "psychology", "psychological", "cognition", "cognitive", "behavior",
        "behavioral", "mental", "mind", "consciousness", "perception",
        "memory", "learning", "emotion", "motivation", "personality",
        "freud", "jung", "piaget", "skinner", "pavlov", "neuroscience",
        "psychiatry", "psychotherapy", "anxiety", "depression", "disorder"
    ],
    "mathematics": [
        "mathematics", "mathematical", "theorem", "proof", "algebra",
        "geometry", "calculus", "topology", "number theory", "set theory",
        "statistics", "probability", "equation", "function", "variable",
        "integral", "derivative", "matrix", "vector", "group theory",
        "euler", "gauss", "riemann", "hilbert", "godel", "turing"
    ],
    "computer_science": [
        "computer science", "algorithm", "data structure", "programming",
        "software", "computation", "complexity", "machine learning",
        "artificial intelligence", "neural network", "database", "compiler",
        "operating system", "network", "cryptography", "turing machine",
        "boolean", "binary", "recursion", "iteration", "object-oriented"
    ],
    "literature": [
        "literature", "literary", "novel", "poetry", "drama", "fiction",
        "author", "writer", "playwright", "narrative", "prose", "verse",
        "shakespeare", "dickens", "austen", "tolstoy", "dostoevsky",
        "hemingway", "twain", "modernism", "romanticism", "realism",
        "symbolism", "tragedy", "comedy", "epic", "sonnet"
    ],
    "history": [
        "history", "historical", "ancient", "medieval", "renaissance",
        "revolution", "war", "empire", "civilization", "dynasty",
        "archaeology", "century", "era", "period", "kingdom"
    ]
}


def create_wikipedia_filter(domain: str):
    """
    Create a filter function for Wikipedia articles by domain.

    Args:
        domain: Domain name (physics, chemistry, biology, etc.)

    Returns:
        Filter function that returns True if article matches domain
    """
    keywords = WIKIPEDIA_DOMAIN_KEYWORDS.get(domain, [])

    def filter_fn(article):
        """Filter Wikipedia article by domain keywords"""
        if not keywords:
            return True

        title = article.get('title', '').lower()
        text = article.get('text', '')[:5000].lower()  # Check first 5000 chars

        # Check title first (higher weight)
        for keyword in keywords:
            if keyword in title:
                return True

        # Check text content
        matches = sum(1 for kw in keywords if kw in text)
        return matches >= 2  # Require at least 2 keyword matches

    return filter_fn


# Pre-created filter functions for use in transform_fn field
def filter_philosophy_articles(article):
    return create_wikipedia_filter("philosophy")(article)

def filter_physics_articles(article):
    return create_wikipedia_filter("physics")(article)

def filter_chemistry_articles(article):
    return create_wikipedia_filter("chemistry")(article)

def filter_biology_articles(article):
    return create_wikipedia_filter("biology")(article)

def filter_psychology_articles(article):
    return create_wikipedia_filter("psychology")(article)

def filter_mathematics_articles(article):
    return create_wikipedia_filter("mathematics")(article)

def filter_cs_articles(article):
    return create_wikipedia_filter("computer_science")(article)

def filter_literature_articles(article):
    return create_wikipedia_filter("literature")(article)

def filter_history_articles(article):
    return create_wikipedia_filter("history")(article)


#=============================================================================
# Kaggle Data Loading
#=============================================================================

def load_kaggle_dataset(dataset_id: str, download_path: str = None):
    """
    Load a Kaggle dataset.

    Requires kaggle CLI to be installed and configured:
    - pip install kaggle
    - Place kaggle.json in ~/.kaggle/

    Args:
        dataset_id: Kaggle dataset identifier (e.g., "kouroshalizadeh/history-of-philosophy")
        download_path: Where to download the dataset (default: ~/.kaggle/datasets/)

    Returns:
        Path to downloaded dataset directory
    """
    import subprocess
    from pathlib import Path

    if download_path is None:
        download_path = Path.home() / ".kaggle" / "datasets" / dataset_id.replace("/", "_")

    download_path = Path(download_path)

    if not download_path.exists():
        download_path.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", dataset_id,
                "-p", str(download_path),
                "--unzip"
            ], check=True)
            print(f"Downloaded Kaggle dataset: {dataset_id}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download Kaggle dataset {dataset_id}: {e}")
            return None
        except FileNotFoundError:
            print("Kaggle CLI not found. Install with: pip install kaggle")
            return None

    return download_path


def stream_kaggle_dataset(dataset: StreamingDataset, batch_size: int = 100):
    """
    Stream examples from a Kaggle dataset.

    Args:
        dataset: StreamingDataset configuration
        batch_size: Number of examples to yield at a time

    Yields:
        Batches of examples from the dataset
    """
    import csv
    import json

    dataset_path = load_kaggle_dataset(dataset.identifier)
    if dataset_path is None:
        return

    # Find data files
    for ext in ['.csv', '.json', '.jsonl', '.txt']:
        files = list(dataset_path.glob(f"*{ext}"))
        if files:
            break

    for filepath in files:
        try:
            if filepath.suffix == '.csv':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f)
                    batch = []
                    for row in reader:
                        text = row.get(dataset.text_field, '')
                        label = row.get(dataset.label_field, '') if dataset.label_field else None
                        if text:
                            batch.append({'text': text, 'label': label, 'source': 'kaggle'})
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                    if batch:
                        yield batch

            elif filepath.suffix in ['.json', '.jsonl']:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    batch = []
                    for line in f:
                        try:
                            row = json.loads(line) if filepath.suffix == '.jsonl' else None
                            if row is None:
                                data = json.load(f)
                                rows = data if isinstance(data, list) else [data]
                                for row in rows:
                                    text = row.get(dataset.text_field, '')
                                    label = row.get(dataset.label_field, '') if dataset.label_field else None
                                    if text:
                                        batch.append({'text': text, 'label': label, 'source': 'kaggle'})
                                        if len(batch) >= batch_size:
                                            yield batch
                                            batch = []
                                break
                            text = row.get(dataset.text_field, '')
                            label = row.get(dataset.label_field, '') if dataset.label_field else None
                            if text:
                                batch.append({'text': text, 'label': label, 'source': 'kaggle'})
                                if len(batch) >= batch_size:
                                    yield batch
                                    batch = []
                        except json.JSONDecodeError:
                            continue
                    if batch:
                        yield batch

            elif filepath.suffix == '.txt':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    batch = []
                    for line in f:
                        line = line.strip()
                        if line:
                            batch.append({'text': line, 'label': None, 'source': 'kaggle'})
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                    if batch:
                        yield batch

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue


#=============================================================================
# Unified Data Streaming Interface
#=============================================================================

def stream_dataset(dataset: StreamingDataset, batch_size: int = 100):
    """
    Stream examples from any dataset source.

    Args:
        dataset: StreamingDataset configuration
        batch_size: Number of examples per batch

    Yields:
        Batches of examples {'text': str, 'label': any, 'source': str}
    """
    if dataset.source == DatasetSource.HUGGINGFACE:
        try:
            from datasets import load_dataset
            hf_dataset = load_dataset(
                dataset.identifier,
                dataset.subset,
                split=dataset.split,
                streaming=dataset.streaming
            )

            # Get filter function if specified
            filter_fn = None
            if dataset.transform_fn:
                filter_fn = globals().get(dataset.transform_fn)

            batch = []
            for example in hf_dataset:
                # Apply filter if present
                if filter_fn and not filter_fn(example):
                    continue

                text = example.get(dataset.text_field, '')
                label = example.get(dataset.label_field) if dataset.label_field else None

                if text:
                    batch.append({'text': text, 'label': label, 'source': 'huggingface'})
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                if dataset.max_samples and len(batch) >= dataset.max_samples:
                    break

            if batch:
                yield batch

        except ImportError:
            print("Hugging Face datasets not installed. Run: pip install datasets")
        except Exception as e:
            print(f"Error streaming HuggingFace dataset {dataset.identifier}: {e}")

    elif dataset.source == DatasetSource.KAGGLE:
        yield from stream_kaggle_dataset(dataset, batch_size)

    elif dataset.source == DatasetSource.LOCAL:
        # Load from local preprocessed files
        local_path = Path(dataset.identifier)
        if local_path.exists():
            yield from stream_kaggle_dataset(dataset, batch_size)  # Reuse file loading logic

    else:
        print(f"Unsupported dataset source: {dataset.source}")


def get_datasets_by_source(source: DatasetSource) -> List[StreamingDataset]:
    """Get all datasets from a specific source"""
    return [d for d in AGI_CURRICULUM_DATASETS if d.source == source]


#=============================================================================
# Main
#=============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NIMCP AGI Curriculum Dataset Configuration"
    )
    parser.add_argument("--summary", action="store_true",
                       help="Print curriculum summary")
    parser.add_argument("--export", type=str,
                       help="Export curriculum to JSON file")
    parser.add_argument("--stage", type=str,
                       choices=["infant", "child", "adolescent", "adult"],
                       help="List datasets for specific stage")
    parser.add_argument("--domain", type=str,
                       help="List datasets for specific domain")

    args = parser.parse_args()

    if args.export:
        export_to_json(args.export)
    elif args.stage:
        stage = DevelopmentStage(args.stage)
        datasets = get_datasets_by_stage(stage)
        print(f"\nDatasets for {args.stage} stage ({len(datasets)} total):\n")
        for d in sorted(datasets, key=lambda x: (x.priority, x.difficulty)):
            print(f"[{d.priority}] {d.name}")
            print(f"    Domain: {d.domain.value}")
            print(f"    Identifier: {d.identifier}")
            print(f"    Difficulty: {d.difficulty}")
            print(f"    Description: {d.description}")
            print()
    elif args.domain:
        try:
            domain = DomainCategory(args.domain)
            datasets = get_datasets_by_domain(domain)
            print(f"\nDatasets for {args.domain} domain ({len(datasets)} total):\n")
            for d in datasets:
                print(f"  - {d.name}: {d.description}")
        except ValueError:
            print(f"Unknown domain: {args.domain}")
            print(f"Available: {[d.value for d in DomainCategory]}")
    else:
        print_curriculum_summary()
