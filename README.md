# NIMCP — Neuromorphic Infant Machine Cognitive Platform

*Exploring whether brain-like architecture produces inherently more interpretable, controllable, and alignable AI systems.*

## Motivation

Most AI safety work focuses on aligning systems after they're built — RLHF, constitutional AI, interpretability probes applied to opaque transformer internals. NIMCP asks a different question: **can architectural choices make alignment structurally easier?**

Biological brains evolved self-regulation long before they evolved language. Homeostatic plasticity prevents runaway activation. Immune-like systems detect and suppress anomalous neural patterns. Developmental stages gate capability acquisition. These aren't bolted-on safety measures — they're load-bearing architecture. NIMCP implements these mechanisms as first-class components of a neuromorphic cognitive platform, written in C with CUDA acceleration.

This is exploratory research, not a solved problem. We don't claim biological fidelity guarantees alignment. But we believe the structural properties of brain-like architectures — self-regulation, interpretability through modularity, developmental gating — deserve serious investigation as complements to post-hoc alignment techniques.

## Safety-Relevant Architecture

| Mechanism | Safety Property | Code |
|-----------|----------------|------|
| Brain immune system | Anomaly detection — identifies and suppresses aberrant activation patterns | `src/cognitive/immune/` |
| Ethics module | Value alignment by design — ethical constraints in the decision pipeline | `src/cognitive/ethics/` |
| Introspection (IIT Phi) | Interpretability — quantified self-monitoring of internal states | `src/cognitive/introspection/` |
| Developmental stages | Capability control — abilities gated by developmental maturity | `scripts/immerse_athena.py` |
| Theory of Mind | Perspective modeling — reasoning about other agents' beliefs and goals | `src/cognitive/theory_of_mind/` |
| Homeostatic plasticity | Self-regulating learning — prevents runaway weight growth | `src/plasticity/homeostatic/` |

## What This Is

- A research platform exploring neuromorphic approaches to AI safety
- ~2,500 C source files, 83 CUDA kernels, Python bindings
- Multi-layer diamond architecture (up to 1.5M neurons on a single GPU)
- 30+ cognitive modules inspired by neuroscience literature
- Actively under development — built primarily by one developer with significant Claude assistance

## What This Is Not

- Not production AI software
- Not a claim to have solved alignment
- Not a neuroscience simulation — we draw inspiration from biology without claiming biological accuracy
- Training is ongoing; results are preliminary
- Performance benchmarks have not been independently validated

## Technical Overview

NIMCP implements a spiking neural network with biologically-inspired cognitive modules:

- **Neuron models**: Leaky Integrate-and-Fire, Izhikevich, with diverse neurotransmitter systems (AMPA, NMDA, GABA, dopamine, serotonin, acetylcholine)
- **Learning**: STDP, eligibility traces, homeostatic plasticity, Oja's rule
- **Architecture**: Multi-layer diamond topology with depth-scaled layers. Small brains get 3 layers, medium get 5, large get 7.
- **Cognitive modules**: Working memory, ethical reasoning, theory of mind, emotional processing, epistemic filtering, mirror neurons, and more
- **GPU acceleration**: 83 CUDA kernels for forward/backward passes, sparse matrix operations, and synaptic updates
- **Multi-modal processing**: Visual cortex (CNN-like), auditory cortex (FFT-based), language processing
- **Decision pipeline**: `decide_full()` routes decisions through ethics checking, introspection, and immune monitoring

```
nimcp/
├── src/
│   ├── core/              # Neural network engine, brain API
│   ├── cognitive/         # 30+ cognitive modules
│   │   ├── ethics/        # Ethical reasoning constraints
│   │   ├── immune/        # Anomaly detection system
│   │   ├── introspection/ # IIT Phi self-monitoring
│   │   ├── theory_of_mind/# Perspective modeling
│   │   └── ...
│   ├── plasticity/        # Learning mechanisms (STDP, homeostatic)
│   ├── gpu/               # CUDA kernels
│   ├── python/            # Python C extension
│   └── bindings/          # Python, Go, Rust, Java, Node.js, Ruby, C#
├── test/                  # Unit and integration tests
├── scripts/               # Training and utility scripts
├── examples/              # Demo programs
└── docs/                  # Documentation
```

## Getting Started

```bash
# Dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake python3-dev libjansson-dev liblz4-dev

# Optional: CUDA for GPU acceleration
sudo apt-get install nvidia-cuda-toolkit

# Build
git clone https://github.com/redmage123/nimcp.git
cd nimcp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

For detailed usage, see:
- [External API Guide](docs/EXTERNAL_API_GUIDE.md) — public C and Python APIs
- [Quick Start](docs/QUICKSTART.md) — first program walkthrough
- [Training Pipeline](docs/TRAINING_PIPELINE.md) — what data flows into the brain during training

## Current Status

**Version 2.6.3** (March 2026)

What works:
- Brain creation, training, and inference through C and Python APIs
- GPU-accelerated forward and backward passes
- Multi-layer diamond architecture with automatic depth scaling
- FAST initialization mode (14s for 1.5M neurons, down from 10+ minutes)
- Developmental training pipeline with curriculum-based learning
- Immune system anomaly detection active during inference

What's in progress:
- Training accuracy on benchmark datasets (currently 23-35% across domains, targeting 95%)
- Cross-modal integration between visual/auditory/language pipelines
- Scaling validation beyond single-GPU configurations

What hasn't been validated:
- Whether the ethics module meaningfully constrains behavior at scale
- Whether developmental staging produces more robust value learning than direct training
- Independent performance benchmarks

## For Researchers

If you're interested in the safety-relevant components, start here:

| Research Question | Entry Point | What to Look For |
|-------------------|-------------|------------------|
| How does the immune system detect anomalies? | `src/cognitive/immune/` | B-cell activation, antigen pattern matching |
| How are ethical constraints enforced? | `src/cognitive/ethics/` | Integration with `decide_full()` pipeline |
| What does introspection actually measure? | `src/cognitive/introspection/` | IIT Phi computation, self-monitoring hooks |
| How does developmental gating work? | `scripts/immerse_athena.py` | Stage progression, capability unlocking |
| Can you inspect internal representations? | `src/cognitive/introspection/` | State export, activation tracing |

See also [SAFETY.md](SAFETY.md) for a deeper discussion of architectural safety properties and open questions.

## Development

This project was built by [Braun Brelin](mailto:braun.brelin@ai-elevate.ai) with substantial assistance from Claude (Anthropic). Claude contributed to architecture design, code generation, debugging, and documentation across 40+ development sessions. This collaboration is documented transparently because we believe human-AI co-development is worth studying openly.

## License

License TBD. This project is currently shared for research review purposes. Please contact [Braun Brelin](mailto:braun.brelin@ai-elevate.ai) regarding usage.
