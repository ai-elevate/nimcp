# NIMCP Documentation Index

**Version**: 2.6.2
**Last Updated**: 2025-12-31

This is the master index for all NIMCP documentation. For developer workflow and coding standards, see [docs/claude/](claude/) which serves as the primary developer reference.

---

## Quick Links

| Purpose | Document |
|---------|----------|
| **Getting Started** | [QUICKSTART.md](QUICKSTART.md) |
| **External API Guide** | [EXTERNAL_API_GUIDE.md](EXTERNAL_API_GUIDE.md) |
| **Build Instructions** | [build/BUILD_INSTRUCTIONS.md](build/BUILD_INSTRUCTIONS.md) |
| **API Reference** | [api/API_REFERENCE.md](api/API_REFERENCE.md) |
| **Coding Standards** | [claude/02-coding-standards.md](claude/02-coding-standards.md) |

---

## Documentation Structure

### Core Developer Reference (`docs/claude/`)

The `docs/claude/` directory is the **primary developer reference** for working with NIMCP:

| File | Description |
|------|-------------|
| [00-overview.md](claude/00-overview.md) | Project vision, motivation, and philosophy |
| [01-build-test.md](claude/01-build-test.md) | Build and test commands |
| [02-coding-standards.md](claude/02-coding-standards.md) | Coding standards and protocols |
| [03-api-patterns.md](claude/03-api-patterns.md) | Key API patterns |
| [04-file-organization.md](claude/04-file-organization.md) | File and directory organization |
| [05-resource-optimization.md](claude/05-resource-optimization.md) | Resource optimization strategies |
| [06-error-codes.md](claude/06-error-codes.md) | Error codes reference |
| [07-common-issues.md](claude/07-common-issues.md) | Common issues and solutions |

#### Module Documentation (`docs/claude/modules/`)

| Module | File |
|--------|------|
| Hemispheric Brain | [modules/hemispheric-brain.md](claude/modules/hemispheric-brain.md) |
| Pink Noise Bridges | [modules/pink-noise.md](claude/modules/pink-noise.md) |
| Brain Immune System | [modules/brain-immune.md](claude/modules/brain-immune.md) |
| Training-Immune Integration | [modules/training-immune.md](claude/modules/training-immune.md) |
| Cross-Bridge Integration | [modules/cross-bridge.md](claude/modules/cross-bridge.md) |
| Liquid Neural Networks | [modules/lnn.md](claude/modules/lnn.md) |
| Bio-Async Integration | [modules/bio-async.md](claude/modules/bio-async.md) |
| Introspection | [modules/introspection.md](claude/modules/introspection.md) |
| Positional Encoding | [modules/positional-encoding.md](claude/modules/positional-encoding.md) |
| Tensor Integration | [modules/tensor.md](claude/modules/tensor.md) |
| Metabolic Modulation | [modules/metabolic-modulation.md](claude/modules/metabolic-modulation.md) |
| Recursive Cognition | [modules/recursive-cognition.md](claude/modules/recursive-cognition.md) |

---

### Build System (`docs/build/`)

| Document | Description |
|----------|-------------|
| [BUILD_INSTRUCTIONS.md](build/BUILD_INSTRUCTIONS.md) | Complete build instructions |
| [BUILD_SECURITY.md](build/BUILD_SECURITY.md) | Build security guidelines |
| [COMPILATION_STATUS.md](build/COMPILATION_STATUS.md) | Compilation status and known issues |

---

### API Documentation (`docs/api/`)

| Document | Description |
|----------|-------------|
| [API_REFERENCE.md](api/API_REFERENCE.md) | Full API reference |
| [QUICK_REFERENCE.md](api/QUICK_REFERENCE.md) | Quick API reference card |

---

### Plans (`docs/plans/`)

| Document | Description |
|----------|-------------|
| [TRAINING_ARCHITECTURE_ENHANCEMENTS.md](plans/TRAINING_ARCHITECTURE_ENHANCEMENTS.md) | World model & ToM training enhancements |

---

### Architecture (`docs/architecture/`)

| Document | Description |
|----------|-------------|
| [GPU_P2P_ARCHITECTURE.md](architecture/GPU_P2P_ARCHITECTURE.md) | GPU peer-to-peer architecture |
| [REFACTORING_PLAN.md](architecture/REFACTORING_PLAN.md) | Architecture refactoring plans |

---

### Implementation Status (`docs/implementation/`)

| Document | Description |
|----------|-------------|
| [IMPLEMENTATION_STATUS.md](implementation/IMPLEMENTATION_STATUS.md) | Current implementation status |
| [IMPLEMENTATION_SUMMARY.md](implementation/IMPLEMENTATION_SUMMARY.md) | Implementation summary |
| [NIMCP_2.5_STATUS.md](implementation/NIMCP_2.5_STATUS.md) | NIMCP 2.5 status |
| [README_2.5.md](implementation/README_2.5.md) | NIMCP 2.5 release notes |

---

### Integration Guides (`docs/integration/`)

| Document | Description |
|----------|-------------|
| [ARTEMIS_INTEGRATION.md](integration/ARTEMIS_INTEGRATION.md) | Integration with Artemis AI |
| [LIBRARY_INTEGRATION.md](integration/LIBRARY_INTEGRATION.md) | Library integration guide |

---

### Security (`docs/security/`)

| Document | Description |
|----------|-------------|
| [SECURITY.md](security/SECURITY.md) | Security overview |
| [SECURITY_AUDIT.md](security/SECURITY_AUDIT.md) | Security audit results |

---

### Ethics (`docs/ethics/`)

| Document | Description |
|----------|-------------|
| [ETHICAL_GUIDELINES.md](ethics/ETHICAL_GUIDELINES.md) | Ethical guidelines |
| [ETHICS_STATUS.md](ethics/ETHICS_STATUS.md) | Ethics implementation status |
| [ETHICS_QUICK_REFERENCE.md](ethics/ETHICS_QUICK_REFERENCE.md) | Ethics quick reference |

---

### Testing (`docs/testing/`)

| Document | Description |
|----------|-------------|
| [README_PARALLEL_TESTS.md](testing/README_PARALLEL_TESTS.md) | Parallel testing guide |
| [TESTING_QUICK_REFERENCE.md](TESTING_QUICK_REFERENCE.md) | Testing quick reference |

---

### Performance (`docs/performance/`)

| Document | Description |
|----------|-------------|
| [PERFORMANCE_OPTIMIZATIONS.md](performance/PERFORMANCE_OPTIMIZATIONS.md) | Performance optimization guide |

---

### Deployment (`docs/deployment/`)

| Document | Description |
|----------|-------------|
| [DEPLOYMENT.md](deployment/DEPLOYMENT.md) | Deployment guide |
| [SLA.md](deployment/SLA.md) | Service level agreement |

---

### Bindings (`docs/bindings/`)

| Document | Description |
|----------|-------------|
| [REFERENCE_IMPLEMENTATIONS.md](bindings/REFERENCE_IMPLEMENTATIONS.md) | Reference implementations |

---

### Development (`docs/development/`)

| Document | Description |
|----------|-------------|
| [LINTING.md](development/LINTING.md) | Linting setup and configuration |
| [LINT_AND_CI_SETUP.md](development/LINT_AND_CI_SETUP.md) | CI/CD setup |

---

### Archive (`docs/archive/`)

Historical documentation including session summaries and completed implementation notes. These are preserved for reference but are not part of active documentation.

---

## Key Concepts

### Brain Architecture

NIMCP implements a biologically-inspired neural network with:

- **Hemispheric processing** - Left/right brain specialization
- **Cortical regions** - Frontal, temporal, parietal, occipital lobes
- **Subcortical structures** - Limbic system, basal ganglia
- **Glial cells** - Astrocytes, microglia, oligodendrocytes
- **Neurotransmitter systems** - Dopamine, serotonin, acetylcholine, etc.

### Learning Systems

- **STDP** - Spike-timing dependent plasticity
- **Hebbian learning** - "Neurons that fire together, wire together"
- **Homeostatic plasticity** - Network stability mechanisms
- **Meta-plasticity** - Learning to learn

### Integration Points

- **Bio-Async** - Asynchronous biological processing
- **Middleware** - Message routing and protocol handling
- **Cognitive modules** - Higher-level cognitive functions

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.6.2 | 2025-12-31 | Documentation cleanup and reorganization |
| 2.6.1 | 2025-12-30 | Occipital lobe integration |
| 2.6.0 | 2025-12-24 | Major refactoring and API updates |
| 2.5.0 | 2025-11-01 | Brain API and ethics engine |
| 2.0.0 | 2025-10-01 | RFC 2.0 protocol implementation |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## Navigation

- **Root**: [CLAUDE.md](../CLAUDE.md) - Project memory and quick reference
- **Tutorials**: [QUICKSTART.md](QUICKSTART.md), [GETTING_STARTED.md](GETTING_STARTED.md)
- **Reference**: [api/API_REFERENCE.md](api/API_REFERENCE.md)
- **Standards**: [claude/02-coding-standards.md](claude/02-coding-standards.md)
