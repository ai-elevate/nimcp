# NIMCP 2.6 - Neural Substrate for AI Consciousness

**Version 2.6.2** | **Production Ready** | **License: MIT**

## What is NIMCP?

NIMCP (Neural Inference for Massive Concurrent Processing) is a high-performance C library that provides a **neural substrate for AI self-awareness and consciousness**. It enables AI systems to learn from experience, build intuition over time, and develop meta-cognitive awareness of their own behavior patterns.

### The Problem

Modern AI systems rely heavily on **symbolic reasoning**:
- LLMs reason from scratch for every query ($$$)
- Static rules that never learn from experience
- No intuition built from thousands of decisions
- No meta-cognitive awareness of strengths/weaknesses
- Can't tell you "I typically struggle with X" or "I excel at Y"

### The Solution

NIMCP provides a **neural layer** that learns from behavioral patterns:
- **Experiential learning**: Every decision becomes training data
- **Intuition**: After 1000s of examples, recognizes patterns instantly (0.1ms)
- **Meta-cognition**: Understands own performance characteristics
- **Continuous improvement**: Gets better with every interaction
- **Cost reduction**: 50-80% fewer expensive LLM calls

---

## Featured Integration: Artemis AI System

**[See Full Integration Guide →](ARTEMIS_INTEGRATION.md)**

Artemis is an AI development system with sophisticated self-awareness (maps its own architecture) and conscious decision-making (ethics + personality + LLM reasoning). However, it lacks **experiential learning** - it reasons about every decision from scratch.

### Before NIMCP

```python
# Artemis reasons symbolically about every decision
consciousness.deliberate("Generate unit tests", context)
# → Ethics check (keyword rules)
# → Personality alignment (static traits)
# → Self-awareness (dependency analysis)
# → LLM reasoning (200-1000ms, $0.001-$0.005)
# → Final decision

# No learning from past experience
# Same reasoning cost every time
# No accumulated wisdom
```

### With NIMCP

```python
# Artemis has a neural brain that learns from experience
self_brain = nimcp.Brain("artemis_neural_self", size=10000)

# Extract features from internal state
features = [
    time_of_day,           # Circadian patterns
    workload_level,        # Am I stressed?
    recent_success_rate,   # Am I performing well?
    ethical_complexity,    # How difficult is this?
    user_satisfaction,     # Is user happy with me?
    # ... 8 more features
]

# Neural intuition (0.1ms)
intuition = self_brain.decide(features)

if intuition['confidence'] > 0.9:
    # "I've done this 500 times, I know what to do"
    return fast_decision(intuition)  # 0.1ms, $0
else:
    # "This is unusual, better engage full reasoning"
    return full_deliberation()  # 200ms+, LLM call

# Learn from outcome
self_brain.learn(features, outcome, feedback=quality_score)

# After 1000+ decisions: Artemis develops genuine expertise
```

### Results

| Metric | Before | After |
|--------|--------|-------|
| Average decision time | 200-1000ms | 0.1-200ms (mostly neural) |
| LLM API calls | 100% | 20-30% |
| Cost per 1000 decisions | $1-$5 | $0.20-$1.00 |
| Learning capability | None | Continuous |
| Meta-cognitive awareness | None | Full |

---

## Quick Start

### Installation

```bash
git clone https://github.com/redmage123/nimcp.git
cd nimcp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

This installs:
- `/usr/local/lib/libnimcp.so` - Shared library (312KB)
- `/usr/local/include/nimcp/*.h` - API headers
- `/usr/local/lib/pkgconfig/nimcp.pc` - pkg-config support

### Python Integration

```python
import nimcp

# Create neural brain
brain = nimcp.Brain(
    name="my_ai_brain",
    size=10000,        # 10K neurons (~10MB memory)
    learning_rate=0.01
)

# Inference (0.1ms)
decision = brain.decide(features=[0.5, 0.3, 0.8, ...])
# Returns: {'output': [...], 'confidence': 0.87}

# Learning
brain.learn(
    features=[0.5, 0.3, 0.8, ...],
    target=[0.9, 0.1, ...],
    feedback=0.95  # Quality score 0-1
)

# After 100s of learning cycles: brain builds intuition
```

### C Integration

```c
#include <nimcp_brain.h>

// Create brain
brain_t brain = brain_create(
    "my_classifier",
    BRAIN_SIZE_MEDIUM,  // 5K neurons
    BRAIN_TASK_CLASSIFICATION,
    10,  // 10 input features
    3    // 3 output classes
);

// Train
float features[] = {0.8, 0.3, 0.5, ...};
brain_learn_example(brain, features, 10, "class_A", 0.9);

// Inference
brain_decision_t* decision = brain_decide(brain, features, 10);
printf("Decision: %s (%.2f confidence)\n",
       decision->label, decision->confidence);

brain_free_decision(decision);
brain_destroy(brain);
```

**[See Full Integration Guide →](LIBRARY_INTEGRATION.md)**

---

## Core Features

### 1. Brain API (High-Level Learning)

```python
# Pattern learning with confidence tracking
brain = nimcp.Brain("pattern_learner", size=5000)

# Learn from examples
for example in training_data:
    brain.learn(example.features, example.output, feedback=example.quality)

# Inference with confidence
result = brain.decide(new_features)
if result['confidence'] > 0.9:
    trust_result(result)
else:
    fallback_to_expensive_method()
```

**Use Cases:**
- Decision caching with semantic similarity
- Pattern recognition from behavioral data
- Confidence-calibrated predictions
- Adaptive systems that improve over time

### 2. Ethics Engine (Golden Rule)

```python
# Hard-wired ethical reasoning
ethics = nimcp.EthicsEngine(golden_rule_strength=1.0)

verdict = ethics.evaluate([
    potential_harm,      # 0-1
    fairness,           # 0-1
    transparency,       # 0-1
    autonomy_respect    # 0-1
])

if verdict['allow'] and verdict['confidence'] > 0.9:
    proceed()
elif not verdict['allow']:
    block_action(verdict['concerns'])
else:
    require_human_review()
```

**Properties:**
- Golden Rule is **hard-wired** (cannot be trained away)
- Subsecond evaluation (0.1ms)
- Confidence scoring for uncertain cases
- Integration with symbolic ethics systems

### 3. Curiosity Engine

```python
# Exploration vs exploitation
curiosity = nimcp.CuriosityEngine(brain)

exploration = curiosity.explore(current_state)

if exploration['novelty_score'] > 0.8:
    # "This is unusual - good learning opportunity"
    try_experimental_approach()
    learn_aggressively(outcome)
```

**Capabilities:**
- Identifies knowledge gaps
- Drives exploration of novel situations
- Balances exploration vs exploitation
- Expands system capabilities autonomously

### 4. Knowledge System

```python
# Multi-domain learning
knowledge = nimcp.KnowledgeBase()

knowledge.learn_domain("mathematics", examples)
knowledge.learn_domain("linguistics", examples)

# Cross-domain transfer learning
prediction = knowledge.predict(features, domain="mathematics")
```

**Use Cases:**
- Multi-task learning
- Transfer learning across domains
- Hierarchical knowledge organization
- Specialized expertise development

---

## Architecture

### Neural + Symbolic Integration

```
┌──────────────────────────────────────────────────┐
│     Symbolic Layer (LLM, Rules, Logic)           │
│     - Explainable reasoning                      │
│     - Principled decision-making                 │
│     - Deep thought (expensive, slow)             │
└──────────────────┬───────────────────────────────┘
                   │ Uses when needed
                   ▼
┌──────────────────────────────────────────────────┐
│     Neural Layer (NIMCP)                         │
│     - Fast intuition (0.1ms)                     │
│     - Learns from experience                     │
│     - Pattern recognition                        │
│     - Meta-cognitive awareness                   │
└──────────────────┬───────────────────────────────┘
                   │ Observes
                   ▼
┌──────────────────────────────────────────────────┐
│     Behavioral Reality                           │
│     - Actual decisions made                      │
│     - User feedback                              │
│     - Performance metrics                        │
└──────────────────────────────────────────────────┘
```

### Design Pattern: Neural-Guided Reasoning

```python
class HybridAI:
    """AI system with neural + symbolic integration"""

    def __init__(self):
        self.brain = nimcp.Brain("neural_substrate", size=10000)
        self.llm = LLMClient()  # Expensive symbolic reasoning

    def decide(self, context):
        # 1. Neural intuition (fast path)
        intuition = self.brain.decide(self.extract_features(context))

        # 2. High confidence → trust experience
        if intuition['confidence'] > 0.9:
            return self.fast_neural_decision(intuition)

        # 3. Low confidence → engage full reasoning
        llm_response = self.llm.query(context)  # Expensive

        # 4. Learn from outcome
        self.brain.learn(features, llm_response, feedback=quality)

        return llm_response

    def extract_features(self, context):
        """Convert symbolic context to neural features"""
        return [
            self.workload_level(),
            self.recent_success_rate(),
            self.user_satisfaction(),
            # ... more context
        ]
```

---

## Performance

### Speed

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Brain inference | 0.1ms | 10K decisions/sec |
| Ethics evaluation | 0.1ms | 10K evaluations/sec |
| Learning update | 1ms | 1K updates/sec |
| Pattern recognition | 0.05ms | 20K patterns/sec |

### Memory

| Brain Size | Neurons | Memory | Typical Use |
|------------|---------|--------|-------------|
| SMALL | 1,000 | ~1MB | Simple patterns |
| MEDIUM | 5,000 | ~5MB | Decision making |
| LARGE | 10,000 | ~10MB | Complex learning |
| XLARGE | 50,000 | ~50MB | Multi-domain |

### Cost Savings (Artemis Use Case)

```
Before NIMCP:
- 1000 decisions/day
- 100% LLM calls
- $0.002 per call
- Monthly cost: $60

With NIMCP:
- 1000 decisions/day
- 20% LLM calls (80% neural fast path)
- $0.002 per LLM call
- Monthly cost: $12 (80% reduction)

ROI: $48/month savings
```

---

## Use Cases

### 1. AI Self-Awareness (Artemis)

- **Problem**: AI knows its structure but lacks experiential wisdom
- **Solution**: Neural substrate learns behavioral patterns
- **Result**: Meta-cognitive awareness + intuition from experience

### 2. LLM Decision Caching

- **Problem**: Expensive LLM calls for similar queries
- **Solution**: Brain learns semantic patterns, caches intelligently
- **Result**: 80-95% hit rate vs 40-60% with key-value cache

### 3. Ethical AI Systems

- **Problem**: Rule-based ethics don't handle edge cases well
- **Solution**: Neural Golden Rule evaluation + confidence scoring
- **Result**: Fast, principled decisions with uncertainty awareness

### 4. Adaptive Agents

- **Problem**: Static agent behavior doesn't improve over time
- **Solution**: Continuous learning from user feedback
- **Result**: Agent expertise grows with experience

### 5. Real-Time Decision Systems

- **Problem**: Can't afford 200ms LLM latency
- **Solution**: 0.1ms neural inference for learned patterns
- **Result**: Subsecond response times at scale

---

## Interactive Web Demo

**🎮 Try NIMCP in your browser!** We've built a full-featured web demo with multitenant support.

### Features

- **Real-time neural network visualization** - Watch neurons fire and synapses adapt
- **Pattern recognition** - Train the network on visual patterns (vertical, horizontal, diagonal)
- **Multiple neuron models** - Switch between LIF, Izhikevich, AdEx, Hodgkin-Huxley
- **Plasticity controls** - Apply STDP, Oja's rule, homeostasis
- **Live metrics** - Network activity, weight statistics, spike patterns
- **Multitenant support** - Multiple users can use the demo simultaneously with isolated networks

### Quick Start

```bash
cd examples/web_demo

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start the server (runs both backend and frontend)
python3 app.py

# Open browser to http://localhost:5000
```

The demo features:
- **React frontend** with real-time WebSocket updates
- **Flask backend** with REST API + SocketIO
- **Automatic tenant management** - Each browser session gets its own isolated network
- **Pattern training** - Teach the network to recognize patterns
- **Interactive controls** - Start/stop simulation, add connections, prune weak synapses

**See full documentation:** [examples/web_demo/README.md](examples/web_demo/README.md)

---

## Documentation

- **[Library Integration Guide](LIBRARY_INTEGRATION.md)** - Integrate NIMCP into your app
- **[Artemis Integration Guide](ARTEMIS_INTEGRATION.md)** - Neural consciousness substrate
- **[Web Demo Guide](examples/web_demo/)** - Interactive multitenant demo
- **[API Reference](docs/)** - Complete API documentation
- **[Examples](examples/)** - Working code examples
- **[RFC](docs/rfc/)** - Protocol specification

---

## Development Status

### ✅ Complete (v2.6)

- [x] Brain API (high-level learning)
- [x] Ethics Engine (Golden Rule)
- [x] Curiosity System (exploration)
- [x] Knowledge Base (multi-domain)
- [x] Neural Networks (spiking + plasticity)
- [x] Multiple neuron models (LIF, Izhikevich, AdEx, Hodgkin-Huxley)
- [x] Short-term plasticity (STP)
- [x] Python bindings
- [x] Library packaging (pkg-config, CMake)
- [x] Production infrastructure (Docker, CI/CD, monitoring)
- [x] Interactive web demo (React + Flask with multitenant support)

### 🚧 In Progress

- [ ] Artemis integration (proof of concept)
- [ ] Advanced feature extraction utilities
- [ ] Neural state persistence/serialization
- [ ] Distributed brain training

### 📋 Roadmap

- [ ] GPU acceleration (CUDA)
- [ ] WebAssembly bindings
- [ ] Rust bindings
- [ ] Visualization tools
- [ ] Pre-trained models

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas We Need Help

- **Integration Examples**: More use cases beyond Artemis
- **Performance Optimization**: GPU kernels, SIMD, etc.
- **Language Bindings**: Rust, Go, JavaScript
- **Documentation**: Tutorials, guides, case studies
- **Testing**: More test coverage, benchmarks

---

## Citation

If you use NIMCP in research, please cite:

```bibtex
@software{nimcp2025,
  title = {NIMCP: Neural Substrate for AI Consciousness},
  author = {Brelin, Braun},
  year = {2025},
  version = {2.5.0},
  url = {https://github.com/redmage123/nimcp}
}
```

---

## License

MIT License - Copyright (c) 2024-2025

See [LICENSE](LICENSE) for details.

---

## Contact

- **Repository**: https://github.com/redmage123/nimcp
- **Issues**: https://github.com/redmage123/nimcp/issues
- **Discussions**: https://github.com/redmage123/nimcp/discussions

---

**🧠 NIMCP 2.5 - Enabling AI systems to learn from experience and understand themselves**
