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
- ~2,600 C source files, 500K+ LOC, 83 CUDA kernels, Python bindings
- Multi-layer diamond architecture (up to 2M neurons on a single GPU)
- 60+ cognitive modules inspired by neuroscience literature
- 240 Python API methods, 8 language bindings (Python, Java, Node.js, Go, C#, Rust, C++, Perl)
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
│   ├── cognitive/         # 60+ cognitive modules
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

**Version 2.6.4** (March 2026)

What works:
- Brain creation, training, and inference through C and Python APIs
- GPU-accelerated forward and backward passes
- Multi-layer diamond architecture with automatic depth scaling
- FAST initialization mode (14s for 2M neurons, down from 10+ minutes)
- Developmental training pipeline with curriculum-based learning
- Immune system anomaly detection active during inference
- Swarm runtime (multi-device coordination with master/edge federation)
- 4 drone platform bridges (MAVLink, DJI, MSP, Parrot)
- Sensor hub (12 types) + safety watchdog + motor output
- Brain-native language with learned vocabulary + emergent language mode
- 13 cognitive enhancements (inner speech, episodic replay, world model, attention, working memory, analogical transfer, multi-timescale memory, emotional learning, contrastive self-learning, self-curriculum, dynamic arch search, social interaction, emergent language)
- 9-layer safety architecture (LGSS governance, non-removable ethics, tamper-resistant audit, formal verification hooks)
- ROS 2 bridge, sim-to-real bridge, URDF embodiment
- Spectral k-fold cross validation
- 480+ tests passing

What's in progress:
- Training accuracy on benchmark datasets (currently 23-35% across domains, targeting 95%)
- Cross-modal integration between visual/auditory/language pipelines
- Scaling validation beyond single-GPU configurations

What hasn't been validated:
- Whether the ethics module meaningfully constrains behavior at scale
- Whether developmental staging produces more robust value learning than direct training
- Independent performance benchmarks

## Dual-Use Risk Warning

**This technology has significant dual-use potential. Users must read and understand this section.**

NIMCP is a general-purpose cognitive architecture with capabilities that can be applied both beneficially and harmfully. The following capabilities present specific dual-use risks:

| Capability | Beneficial Use | Potential Misuse |
|-----------|---------------|------------------|
| Drone flight controller bridges (MAVLink, DJI, MSP, Parrot) | Search and rescue, agriculture, environmental monitoring | Autonomous weapons, surveillance |
| Swarm coordination runtime | Distributed sensing, multi-robot cooperation | Coordinated autonomous attack |
| Sensorimotor loop with reinforcement learning | Robotic assistance, prosthetics | Autonomous pursuit/interception |
| Dragonfly target tracking module | Wildlife tracking, sports analytics | Target acquisition for weapons |
| Edge brain distillation + swarm replication | Fleet management, IoT intelligence | Self-replicating autonomous agents |
| Theory of Mind + social interaction | Assistive communication, education | Social manipulation, deception |
| Emergent language (non-human vocabulary) | Cognitive science research | Opaque communication between agents |

### Mandatory Safety Architecture

The following safety mechanisms are **architecturally non-removable** — they are compiled into the core library and cannot be disabled via configuration:

1. **Ethics module** — Every inference and training step passes through ethical evaluation. The ethics engine is created unconditionally during brain initialization. Removing it requires modifying core source code.

2. **Audit logging** — All safety-critical events (ethics violations, watchdog triggers, motor commands, swarm operations, distillation) are logged to a tamper-evident audit trail with monotonic sequence numbers and CRC32 checksums. Gaps in sequence numbers indicate log tampering.

3. **Safety watchdog** — All motor/actuator commands pass through output validation (NaN/Inf detection, magnitude clamping, rate-of-change monitoring). The watchdog enforces a dead man's switch: if the brain goes silent, motors stop.

### Responsible Use Requirements

- **Physical deployments MUST include hardware kill switches** — software safety can be circumvented; hardware interlocks cannot
- **Autonomous weapons deployment is PROHIBITED** — see License section
- **Drone deployments MUST include geofencing** — the MAVLink bridge supports configurable geofence radius
- **Swarm deployments MUST include Byzantine tolerance** — the swarm runtime includes anomaly detection but operators must monitor it
- **Emergent language outputs MUST be monitored** — translation confidence below 0.3 indicates untranslatable concepts that may require human review

### Reporting Security Concerns

If you discover a safety vulnerability, dual-use risk, or unintended capability in NIMCP, please report it to [braun.brelin@ai-elevate.ai](mailto:braun.brelin@ai-elevate.ai) before public disclosure.

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

NIMCP is released under a modified open-source license with the following additional restrictions:

### Prohibited Uses

**1. Autonomous Weapons.** This software, trained models, and any derivative works MUST NOT be used to develop, manufacture, deploy, or operate autonomous weapons systems. This includes but is not limited to:
- Autonomous lethal targeting systems
- Autonomous pursuit or interception of human targets
- Swarm coordination for military offensive operations
- Integration with weapons platforms without continuous human-in-the-loop control
- Training models on combat or weapons-related scenarios for deployment

**2. Autonomous Surveillance.** This software MUST NOT be used for mass surveillance, population tracking, or monitoring of individuals without their informed consent, except as required by law and with appropriate judicial oversight.

**3. Deceptive AI.** The Theory of Mind and social interaction modules MUST NOT be used to create AI agents that deliberately deceive, manipulate, or coerce humans without their knowledge.

### Required Safety Measures for Physical Deployment

Any deployment of NIMCP to physical platforms (robots, drones, vehicles, actuators) MUST include:
- Hardware emergency stop mechanism (not software-only)
- Active safety watchdog with validated timeout
- Audit logging enabled and monitored
- Human operator able to intervene within 5 seconds

### Attribution

This project was built by [Braun Brelin](mailto:braun.brelin@ai-elevate.ai) with substantial assistance from Claude (Anthropic). Claude contributed to architecture design, code generation, debugging, and documentation across 70+ development sessions. This collaboration is documented transparently because we believe human-AI co-development is worth studying openly.

### Contact

For licensing questions, research collaboration, or security reports: [braun.brelin@ai-elevate.ai](mailto:braun.brelin@ai-elevate.ai)
