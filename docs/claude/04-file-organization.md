# File Organization

## Project Root: `/home/bbrelin/nimcp`

```
include/
├── core/
│   ├── brain/            # Brain init, factory, inference, lifecycle, resize
│   │   ├── regions/      # Language production, predictive regions
│   │   └── hemispheric/  # Bilateral brain (hemispheres, corpus callosum, lateralization)
│   ├── neuralnet/        # Neural network, sparse synapses, neuron/synapse access
│   ├── synapse_compute/  # Synapse computation kernels
│   ├── synapse_types/    # Synapse type definitions
│   ├── cortical_columns/ # Cortical column models
│   ├── brain_regions/    # Brain region definitions
│   ├── brain_oscillations/ # Neural oscillations
│   ├── neuron_types/     # Neuron type definitions
│   ├── axon/             # Axon models
│   └── dendrite/         # Dendrite models
├── cognitive/            # 60+ cognitive modules
│   ├── introspection/    # Consciousness metrics, temporal patterns, ensemble uncertainty
│   ├── immune/           # Brain immune system (B/T cells, antibodies, cytokines)
│   ├── common/           # Shared utilities (metabolic modulation, clamp functions)
│   ├── global_workspace/ # Global workspace theory
│   └── ...               # Working memory, emotion, attention, reasoning, ethics, etc.
├── lnn/                  # 17 headers — Liquid Neural Networks (LTC neurons, ODE solvers, wiring)
├── snn/                  # 50 headers — Spiking Neural Networks, 42 cross-modal bridges
├── plasticity/           # 164 headers — largest subsystem (STDP, BCM, homeostatic, etc.)
│   └── attention/        # Multihead attention with positional encoding
├── perception/           # Visual, audio, speech, somatosensory cortex
├── gpu/                  # 80+ CUDA modules (kernels, training, stream pool)
├── language/             # NLP, speech, grounded language, SNN-language bridge
├── glial/                # Astrocytes, oligodendrocytes, microglia
├── physics/              # Biophysics models
├── chemistry/            # Neurochemistry
├── biology/              # Biological models
├── security/             # Blood-brain barrier (BBB), LGSS, constant-time ops
├── middleware/           # Training pipeline, event handling, sequence detector, population coding
│   └── integration/      # Shannon monitor, routing
├── information/          # Shannon entropy, cross-modal analysis
├── swarm/                # Swarm signal, swarm immune
└── utils/
    ├── memory/           # Memory allocator, unified memory manager
    ├── tensor/           # Tensor library
    ├── encoding/         # Positional encoding
    ├── threading/        # Thread pool, mutex, atomics
    ├── logging/          # Logger, log levels
    ├── metrics/          # Performance metrics
    └── fault_tolerance/  # BFT, recovery cache, checkpointing

src/                      # Mirrors include/ structure
├── api/                  # Public API implementation (nimcp_part_core, nimcp_part_io)
├── core/brain/           # Brain factory, inference, lifecycle, resize, multimodal
│   └── factory/          # Brain initialization, config, parallel init
├── core/neuralnet/       # Neural network, sparse synapses, embeddings
├── core/synapse_compute/ # Synapse computation
├── core/synapse_types/   # Synapse types
├── bindings/python/      # Python C extension
├── gpu/                  # CUDA kernels (.cu files)
│   └── training/         # GPU training bridge
├── plasticity/           # Plasticity implementations
└── ...                   # Other modules mirror include/

scripts/
├── immerse_athena.py     # Immersive developmental training
├── talk_to_athena.py     # Interactive conversation with trained brain
└── ...                   # Monitoring, utilities

test/
├── unit/                 # Unit tests by module
├── integration/          # Module interaction tests
├── regression/           # Stability tests
└── e2e/                  # End-to-end pipeline tests

docs/                     # 823+ markdown files
├── claude/               # Claude-specific documentation (this directory)
│   └── modules/          # Per-module documentation
├── INDEX.md              # Master documentation index
└── EXTERNAL_API_GUIDE.md # External API reference

build/                    # CMake build directory (out-of-source)
data/                     # Training data, datasets
```

## Key Size Notes

- `include/plasticity/` is the largest subsystem at 164 headers
- `include/gpu/` has 80+ CUDA module headers
- `include/cognitive/` has 60+ cognitive module headers
- `include/snn/` has 50 headers including 42 cross-modal bridge headers
- `include/lnn/` has 17 headers for Liquid Neural Networks
- `docs/` contains 823+ markdown documentation files
