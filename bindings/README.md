# NIMCP Brain API - Multi-Language Bindings

FFI bindings for NIMCP Brain API across multiple programming languages.

## Overview

NIMCP Brain provides a lightweight, fast neural learning framework inspired by adaptive threshold spiking. These bindings enable integration with applications written in various languages through a clean, idiomatic API.

## Supported Languages

| Language | Status | Location | Package Manager |
|----------|--------|----------|-----------------|
| **C** | ✅ Core API | `../src/include/nimcp_brain.h` | System libraries |
| **Python** | ✅ Complete | `python/` | pip |
| **C++** | ✅ Complete | `cpp/` | Header-only |
| **TypeScript/JavaScript** | ✅ Complete | `typescript/` | npm |
| **Rust** | 🏗️ Planned | `rust/` | cargo |
| **Go** | 🏗️ Planned | `golang/` | go modules |
| **Java** | 🏗️ Planned | `java/` | maven/gradle |
| **C#** | 🏗️ Planned | `csharp/` | NuGet |

## Quick Start by Language

### Python

```bash
cd python
pip install -r requirements.txt  # ffi, numpy
python example.py
```

```python
from nimcp_brain import Brain, BrainSize, BrainTask

# Create brain
brain = Brain("my_classifier", BrainSize.SMALL, BrainTask.CLASSIFICATION, 4, 3)

# Learn
brain.learn_example([0.8, 0.3, 0.5, 0.6], "allow", confidence=0.9)

# Decide
decision = brain.decide([0.8, 0.3, 0.5, 0.6])
print(f"{decision.label}: {decision.confidence:.2f}")

# Save
brain.save("brain.nimcp")
```

### C++

```bash
cd cpp
g++ -std=c++17 example.cpp -I../../src/include -L../../build/src/lib -lnimcp_core -o example
LD_LIBRARY_PATH=../../build/src/lib ./example
```

```cpp
#include "nimcp_brain.hpp"

using namespace nimcp;

// Create brain
Brain brain("my_classifier", BrainSize::SMALL, BrainTask::CLASSIFICATION, 4, 3);

// Learn
brain.learn_example({0.8f, 0.3f, 0.5f, 0.6f}, "allow", 0.9f);

// Decide
auto decision = brain.decide({0.8f, 0.3f, 0.5f, 0.6f});
std::cout << decision.label << ": " << decision.confidence << "\n";

// Save
brain.save("brain.nimcp");
```

### TypeScript/JavaScript

```bash
cd typescript
npm install
npm run example
```

```typescript
import { Brain, BrainSize, BrainTask } from '@nimcp/brain';

// Create brain
const brain = new Brain("my_classifier", BrainSize.SMALL, BrainTask.CLASSIFICATION, 4, 3);

// Learn
brain.learnExample([0.8, 0.3, 0.5, 0.6], "allow", 0.9);

// Decide
const decision = brain.decide([0.8, 0.3, 0.5, 0.6]);
console.log(`${decision.label}: ${decision.confidence.toFixed(2)}`);

// Save
brain.save("brain.nimcp");
```

## API Overview

All language bindings provide these core features:

### Brain Creation

```
Brain(name, size, task, num_inputs, num_outputs)
```

**Sizes**: TINY (100 neurons), SMALL (1K), MEDIUM (10K), LARGE (100K)

**Tasks**: CLASSIFICATION, REGRESSION, PATTERN_MATCHING, SEQUENCE, ASSOCIATION

### Learning

```
learn_example(features, label, confidence) -> loss
learn_batch(examples) -> avg_loss
learn_from_llm(features, teacher_fn) -> loss
```

### Inference

```
decide(features) -> Decision {
    label: string
    confidence: float
    num_active_neurons: int
    sparsity: float
    inference_time_us: int
    explanation: string
}
```

### Persistence

```
save(filepath)
load(filepath) -> Brain
```

### Monitoring

```
get_stats() -> Stats {
    num_neurons: int
    total_inferences: int
    avg_inference_time_us: float
    memory_bytes: int
    ...
}

get_top_neurons(top_n) -> [(neuron_id, importance), ...]
```

### Optimization

```
prune(threshold) -> num_pruned
optimize_for_inference()
```

## Use Cases

### 1. LLM Decision Caching

Cache expensive LLM decisions locally for 100-1000x speedup:

```python
# Expensive LLM decision
def llm_ethics_check(situation):
    response = openai.call(f"Is this ethical? {situation}")
    return response.decision, response.confidence

# Train brain to mimic LLM
brain = Brain("ethics_cache", BrainSize.SMALL, BrainTask.CLASSIFICATION, 10, 3)

for situation in training_situations:
    features = extract_features(situation)
    label, confidence = llm_ethics_check(situation)  # Slow: ~1000ms
    brain.learn_example(features, label, confidence)

# Fast local decisions
decision = brain.decide(new_situation_features)  # Fast: ~0.5ms
```

### 2. Pattern Recognition

Learn patterns from examples:

```cpp
Brain brain("anomaly_detector", BrainSize::MEDIUM, BrainTask::PATTERN_MATCHING, 50, 1);

// Learn normal patterns
for (const auto& normal_pattern : normal_data) {
    brain.learn_example(normal_pattern, "normal", 1.0f);
}

// Detect anomalies
auto decision = brain.decide(test_pattern);
if (decision.confidence < 0.5) {
    std::cout << "Anomaly detected!\n";
}
```

### 3. Real-time Classification

Fast classification with interpretability:

```typescript
const brain = createClassifier("spam_filter", 100, 2);

// Train on labeled emails
for (const [features, label] of training_emails) {
    brain.learnExample(features, label, 1.0);
}

// Classify new email
const decision = brain.decide(email_features);
console.log(`Classification: ${decision.label}`);
console.log(`Active neurons: ${decision.numActiveNeurons}`);
console.log(`Explanation: ${decision.explanation}`);
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Inference Time** | 0.1-5ms (size dependent) |
| **Training Speed** | ~1000 examples/sec |
| **Memory Usage** | 1MB (TINY) to 500MB (LARGE) |
| **Sparsity** | 70-90% neurons inactive |
| **Accuracy** | ~85-95% (after distillation) |

## Comparison with Alternatives

| Feature | NIMCP Brain | PyTorch/TensorFlow | LLMs |
|---------|-------------|-------------------|------|
| **Inference Speed** | <1ms | 10-100ms | 500-2000ms |
| **Model Size** | 1-500MB | 10MB-10GB | 7GB-100GB+ |
| **Training** | Online, incremental | Batch, offline | API-based |
| **Interpretability** | Built-in | Limited | None |
| **Hardware** | CPU-friendly | GPU-preferred | GPU required |
| **Cost per inference** | $0 | $0 | $0.001-0.01 |

## Architecture

```
Application (Python/C++/TS/Rust/Go/Java/C#)
    ↓ FFI Bindings
nimcp_brain.h (High-level API)
    ↓
nimcp_adaptive.h (Adaptive Spiking)
    ↓
nimcp_neuralnet.h (Neural Network Core)
    ↓
STDP, Hebbian, Oja's Learning Rules
```

## Building from Source

```bash
# Build core library
cd ../
mkdir build && cd build
cmake ..
make

# Verify library
ls -lh src/lib/libnimcp_core.so

# Test Python bindings
cd ../bindings/python
python example.py

# Test C++ bindings
cd ../cpp
g++ -std=c++17 example.cpp -I../../src/include -L../../build/src/lib -lnimcp_core -o example
LD_LIBRARY_PATH=../../build/src/lib ./example

# Test TypeScript bindings
cd ../typescript
npm install
npm run example
```

## FFI Design Principles

All bindings follow these principles:

1. **Opaque Handles**: `brain_t` is an opaque pointer across all languages
2. **Simple Types**: Use primitive types (int, float, char*) for FFI compatibility
3. **Error Handling**: Thread-local error strings via `brain_get_last_error()`
4. **Memory Management**: Explicit `destroy`/`free` functions for resource cleanup
5. **Idiomatic APIs**: Language-specific wrappers provide native idioms (RAII in C++, context managers in Python)

## Contributing

To add bindings for a new language:

1. Create directory: `bindings/<language>/`
2. Implement FFI wrapper for `nimcp_brain.h`
3. Create idiomatic API wrapper
4. Add example usage
5. Update this README
6. Add to CI/CD pipeline

## License

Same as NIMCP core (see main LICENSE file).

## Support

- **Issues**: https://github.com/your-org/nimcp/issues
- **Docs**: https://docs.nimcp.org
- **Examples**: See `examples/` in each binding directory

## Roadmap

### Q2 2025
- ✅ Python bindings (complete)
- ✅ C++ bindings (complete)
- ✅ TypeScript/JavaScript bindings (complete)
- 🏗️ Rust bindings (in progress)
- 🏗️ Go bindings (planned)

### Q3 2025
- Java bindings (JNI)
- C# bindings (.NET)
- WebAssembly support
- PyPI/npm package releases

### Q4 2025
- GPU acceleration support
- Distributed training
- Model compression tools
- Production deployment guides

## Citation

If you use NIMCP Brain in research, please cite:

```bibtex
@software{nimcp_brain,
  title = {NIMCP Brain: Lightweight Neural Learning Framework},
  author = {NIMCP Contributors},
  year = {2025},
  url = {https://github.com/your-org/nimcp}
}
```
