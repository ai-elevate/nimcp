# NIMCP — Neuro-Inspired Modular Control Protocol

*A biologically-grounded cognitive architecture that learns from experience, develops through stages, and speaks its own language — running on a single GPU.*

## Motivation

Every major AI system today is the same architecture: a transformer predicting the next token. They're powerful at language but they don't learn after training, don't have memories, don't have bodies, and don't develop over time.

NIMCP asks a different question: **what if we built AI the way biology builds minds?** Not by scaling text prediction, but by implementing the mechanisms that produce cognition — synaptic plasticity, neuromodulation, sleep consolidation, developmental stages, embodied interaction with the physical world.

Safety isn't bolted on after the fact. It's structural. The ethics module can't be turned off. The governance system's rules can only get stricter. The audit log can't be tampered with. These aren't features — they're load-bearing architecture.

## What Makes This Different

| | LLMs (GPT, Claude) | NIMCP |
|---|---|---|
| **Learning** | Trained once, then frozen | Continuous learning from experience |
| **Memory** | Context window only | Episodic + semantic + autobiographical (SQLite-backed) |
| **Body** | None | Sensor hub, motor output, drone/robot deployment |
| **Language** | Statistical text prediction | Emergent from neural patterns (alien vocabulary) |
| **Safety** | RLHF post-hoc | 9 architectural layers, non-removable |
| **Development** | Instant (no growth) | 4-stage infant-to-reasoning curriculum |
| **Hardware** | Datacenter cluster | Single consumer GPU |
| **Self-awareness** | None | Introspection (IIT Phi) + metacognition |

## Architecture

### The Brain

A 1-million neuron brain with 6 neural network types running in parallel:

- **Adaptive** — main feedforward/backprop network
- **SNN** (Spiking) — temporal pattern encoding via spike timing
- **LNN** (Liquid) — temporal dynamics with Hamiltonian conservation
- **CNN** (Convolutional) — visual and audio cortex processing
- **FNO** (Fourier Neural Operator) — spectral/frequency domain
- **HNN** (Hamiltonian) — energy-conserving dynamics

Multi-layer diamond topology (3/5/7 layers by size), 320 synapses per neuron, GPU-accelerated via 83 CUDA kernels.

### Biological Plasticity

Real neuroscience learning mechanisms: STDP, BCM, eligibility traces, homeostatic plasticity, 6 neuromodulators (dopamine, serotonin, acetylcholine, norepinephrine, GABA, glutamate), and episodic replay during simulated sleep.

### 60+ Cognitive Modules

| Category | Modules |
|----------|---------|
| **Social** | Theory of Mind, mirror neurons, collective cognition, multi-agent social interaction |
| **Reasoning** | Recursive cognition (RCOG), imagination engine, analogical transfer, predictive world model |
| **Ethics** | Ethics engine (non-removable), LGSS governance (9 layers) |
| **Memory** | Engram system, working memory scratchpad, multi-timescale memory, episodic replay |
| **Self-awareness** | Introspection (IIT Phi), metacognition, inner speech loop, dynamic architecture search |
| **Learning** | Emotional learning modulation, contrastive self-learning, self-generated curriculum |
| **Perception** | Visual cortex, audio cortex, speech cortex, somatosensory, sensor hub (12 types) |
| **Motor** | Motor output translation, safety watchdog, sensorimotor closed-loop controller |
| **Biological** | Dragonfly target tracking, Portia resource adaptation, brain immune system |
| **Language** | Brain-native language production, BPE tokenizer, emergent alien language, phonological loop |

### Brain-Native Language

Instead of depending on an external LLM, the brain develops its own language:

- **Native mode**: Learned projection matrices map between 4096-dim thought space and 256-dim token space. Autoregressive decoding with nucleus sampling.
- **Emergent mode**: No English seed words. Vocabulary discovered from neural activation clusters. Produces symbols like `ξ₀ ◊₃ α₁ △₇` — each representing a concept-cluster in the brain's thought space. Some translate to human concepts. Others don't.

### Edge/Robot Deployment

| Component | What It Does |
|-----------|-------------|
| Sensor hub | 12 sensor types (LIDAR, IMU, GPS, cameras, force/torque, bumpers) |
| Safety watchdog | NaN/magnitude/rate validation, dead man's switch, emergency stop |
| Motor output | Brain→actuator translation with 4 presets (twist, quadrotor, differential, arm) |
| Drone bridges | MAVLink (PX4/ArduPilot), DJI OSDK, Betaflight MSP, Parrot Olympe |
| ROS 2 bridge | Robot Operating System integration (stub mode without ROS 2) |
| Sim-to-real | Cart-pole physics, domain randomization, curiosity-driven RL |
| URDF embodiment | Robot body model, forward kinematics, body schema |
| Sensorimotor loop | Closed-loop: sensor → brain → motor → environment → sensor |

### Swarm Runtime

Multiple brains coordinating across devices:

- Master/edge architecture with UDP multicast discovery
- Federated gradient aggregation (FedAvg, FedProx, FedMedian)
- Byzantine fault tolerance (gradient anomaly detection, quarantine)
- Delta weight pushes with LZ4 compression
- Gossip learning (peer-to-peer weight sharing)
- Theory of Mind via real multi-agent interaction

### Safety Architecture (9 Layers)

Safety is architecturally embedded. The ethics module and audit log **cannot be disabled** via configuration.

| Layer | What It Does | Removable? |
|-------|-------------|------------|
| 1. LGSS Input Validator | Catches adversarial/corrupted inputs | No |
| 2. Ethics Module | Evaluates every inference and training step | **No** |
| 3. LGSS Action Interceptor | Governance rules on all outputs | No |
| 4. Safety Watchdog | Heartbeat-based dead man's switch | No |
| 5. LGSS Motor Gate | Validates physical output commands | No |
| 6. LGSS Training Guard | Prevents adversarial training data | No |
| 7. LGSS Reward Alignment | Prevents reward hacking | No |
| 8. Audit Log | Tamper-evident trail (CRC32 + sequence numbers) | **No** |
| 9. LGSS Enhanced | Monotonic tightening, formal verification, multi-stakeholder governance | No |

Additional safety features:
- **Monotonic safety**: Rules can only get stricter, never more permissive
- **Formal verification hooks**: Export rules as SMT-LIB v2 for Z3/CVC5 proof
- **Cross-layer verification**: Detects if ethics module is rubber-stamping
- **Violation escalation**: Graduated response (warn → clamp → block → estop)
- **MILITARY_PROHIBITED context**: One-way lock that cannot be reversed

## Scale

| Metric | Value |
|--------|-------|
| Source files | ~2,600 C files, ~2,500 headers |
| Lines of code | ~500,000+ |
| CUDA kernels | 83 |
| Cognitive modules | 60+ |
| Brain regions | 33+ |
| Network types | 6 |
| Python API methods | 240 |
| Language bindings | 8 (Python, Java, Node.js, Go, C#, Rust, Perl, C++) |
| Tests | 480+ passing |
| Development sessions | 73 |

## Getting Started

```bash
# Dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake python3-dev libjansson-dev liblz4-dev libsqlite3-dev libsodium-dev

# Optional: CUDA for GPU acceleration
sudo apt-get install nvidia-cuda-toolkit

# Build
git clone https://github.com/redmage123/nimcp.git
cd nimcp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
make unit_edge_test_sensor_hub unit_cognitive_test_enhancements -j4
./test/unit/edge/unit_edge_test_sensor_hub
./test/unit/cognitive/unit_cognitive_test_enhancements

# Start training
cd .. && python3 scripts/immerse_athena.py --stage 0
```

For detailed usage:
- [External API Guide](docs/EXTERNAL_API_GUIDE.md) — public C and Python APIs
- [CLAUDE.md](CLAUDE.md) — development reference and conventions
- [docs/claude/](docs/claude/) — modular architecture documentation

## Current Status

**Version 2.6.4** (March 2026)

Training status: **Stage 1** (Association — learning to name objects). Loss: 0.03-3,400. SNN: 15-17 Hz (healthy). 4,750+ unique training descriptions across 13 cognitive domains.

| Stage | Description | Status |
|-------|-------------|--------|
| Stage 0 | Sensory Awakening — raw perception | Completed |
| Stage 1 | Association — "Look! That's a ___!" | In progress |
| Stage 2 | Babbling — feedback and correction | Queued |
| Stage 3 | Reasoning — conversation and thought | Queued |

## Dual-Use Risk Warning

**This technology has significant dual-use potential. Users must read and understand this section.**

| Capability | Beneficial Use | Potential Misuse |
|-----------|---------------|------------------|
| Drone flight controller bridges | Search and rescue, agriculture | Autonomous weapons, surveillance |
| Swarm coordination runtime | Distributed sensing, multi-robot cooperation | Coordinated autonomous attack |
| Sensorimotor loop with RL | Robotic assistance, prosthetics | Autonomous pursuit/interception |
| Dragonfly target tracking | Wildlife tracking, sports analytics | Target acquisition for weapons |
| Edge distillation + swarm | Fleet management, IoT intelligence | Self-replicating autonomous agents |
| Theory of Mind + social | Assistive communication, education | Social manipulation, deception |
| Emergent language | Cognitive science research | Opaque communication between agents |

### Reporting Security Concerns

If you discover a safety vulnerability, dual-use risk, or unintended capability, please report it to [braun.brelin@ai-elevate.ai](mailto:braun.brelin@ai-elevate.ai) before public disclosure.

## For Researchers

| Research Question | Entry Point |
|-------------------|-------------|
| How does architectural safety compare to post-hoc alignment? | `src/security/lgss/`, `src/cognitive/ethics/` |
| Can developmental staging produce robust value learning? | `scripts/immerse_athena.py` |
| What does emergent non-human language look like? | `src/cognitive/language/nimcp_emergent_language.c` |
| How does embodied learning differ from text-based? | `src/edge/nimcp_sensorimotor.c` |
| Can biological plasticity prevent catastrophic forgetting? | `src/plasticity/` |
| Does Theory of Mind require social interaction? | `src/cognitive/social/nimcp_social_interaction.c` |

## License

NIMCP is released under a modified open-source license with the following restrictions:

### Prohibited Uses

**1. Autonomous Weapons.** This software MUST NOT be used to develop, deploy, or operate autonomous weapons systems — including lethal targeting, autonomous pursuit of human targets, swarm coordination for military offensive operations, or integration with weapons platforms without continuous human-in-the-loop control.

**2. Autonomous Surveillance.** This software MUST NOT be used for mass surveillance or monitoring of individuals without informed consent, except as required by law with judicial oversight.

**3. Deceptive AI.** The Theory of Mind and social interaction modules MUST NOT be used to create AI agents that deliberately deceive or manipulate humans without their knowledge.

### Required Safety for Physical Deployment

Any deployment to robots, drones, vehicles, or actuators MUST include:
- Hardware emergency stop mechanism (not software-only)
- Active safety watchdog with validated timeout
- Audit logging enabled and monitored
- Human operator able to intervene within 5 seconds

## Development

Built by [Braun Brelin](mailto:braun.brelin@ai-elevate.ai) with substantial assistance from Claude (Anthropic) across 73 development sessions. Claude contributed to architecture design, code generation, debugging, testing, and documentation. This collaboration is documented transparently because we believe human-AI co-development deserves open scrutiny.

## Contact

For licensing, research collaboration, or security reports: [braun.brelin@ai-elevate.ai](mailto:braun.brelin@ai-elevate.ai)
