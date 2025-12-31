# NIMCP Project Memory

**Version**: 2.6.2
**Last Updated**: 2025-12-31

> **Documentation is modularized.** See `docs/claude/` for detailed documentation.
> **Master Index**: See `docs/INDEX.md` for complete documentation navigation.

---

## Quick Reference

| Topic | File |
|-------|------|
| **External API Guide** | [EXTERNAL_API_GUIDE.md](docs/EXTERNAL_API_GUIDE.md) |
| Project Vision & Motivation | [00-overview.md](docs/claude/00-overview.md) |
| Build & Test Commands | [01-build-test.md](docs/claude/01-build-test.md) |
| Coding Standards & Protocols | [02-coding-standards.md](docs/claude/02-coding-standards.md) |
| Key API Patterns | [03-api-patterns.md](docs/claude/03-api-patterns.md) |
| File Organization | [04-file-organization.md](docs/claude/04-file-organization.md) |
| Resource Optimization | [05-resource-optimization.md](docs/claude/05-resource-optimization.md) |
| Error Codes | [06-error-codes.md](docs/claude/06-error-codes.md) |
| Common Issues | [07-common-issues.md](docs/claude/07-common-issues.md) |

## Module Documentation

| Module | File |
|--------|------|
| Hemispheric Brain | [modules/hemispheric-brain.md](docs/claude/modules/hemispheric-brain.md) |
| Pink Noise Bridges | [modules/pink-noise.md](docs/claude/modules/pink-noise.md) |
| Brain Immune System | [modules/brain-immune.md](docs/claude/modules/brain-immune.md) |
| Training-Immune Integration | [modules/training-immune.md](docs/claude/modules/training-immune.md) |
| Cross-Bridge Integration | [modules/cross-bridge.md](docs/claude/modules/cross-bridge.md) |
| Liquid Neural Networks | [modules/lnn.md](docs/claude/modules/lnn.md) |
| Bio-Async Integration | [modules/bio-async.md](docs/claude/modules/bio-async.md) |
| Introspection | [modules/introspection.md](docs/claude/modules/introspection.md) |
| Positional Encoding | [modules/positional-encoding.md](docs/claude/modules/positional-encoding.md) |
| Tensor Integration | [modules/tensor.md](docs/claude/modules/tensor.md) |
| Metabolic Modulation | [modules/metabolic-modulation.md](docs/claude/modules/metabolic-modulation.md) |

---

## CRITICAL WARNING - PAST VIOLATIONS

**Claude has repeatedly ignored the MANDATORY Test Writing Protocol.**

**Before Claude writes ANY test code, the user MUST require Claude to:**
1. Show the complete header file content that was read
2. List the exact function signatures to be used
3. Only then approve test code writing

**See [02-coding-standards.md](docs/claude/02-coding-standards.md) for full protocol.**

---

## Essential Commands

```bash
# Build
cd /home/bbrelin/nimcp/build && cmake .. && make nimcp -j4

# Git
git add -A && git commit --no-verify -m "message" && git push
```

## Critical GOTCHAs

### Tensor API
- `nimcp_tensor_sum()` returns `nimcp_tensor_t*`, not scalar
- `nimcp_tensor_create(dims, rank, dtype)` requires 3 args (rank, not ndims)

### Mutex API (use thread layer, not platform layer)
- `nimcp_mutex_create(attr)` - allocate and init, returns `nimcp_mutex_t*`
- `nimcp_mutex_init(mutex, attr)` - init existing struct
- `mutex_attr_t` supports MUTEX_TYPE_NORMAL/RECURSIVE/ERRORCHECK
- **Deadlock prevention**: Never call public mutex-locking functions from within locked code - create `*_unlocked()` helpers

### Platform Tiers
- `PLATFORM_TIER_FULL/MEDIUM/CONSTRAINED/MINIMAL`

### Brain Immune System
- B cells must be in PLASMA state to produce antibodies
- State progression: NAIVE -> ACTIVATED -> PLASMA

### Return Value Conventions
- FEP bridges return `0` for success, `-1` for errors (not NIMCP_OK/NIMCP_ERROR_*)
- Metabolic modulation: `metabolic_compute_effects()` returns `0` for success, `-1` for errors
- Standard NIMCP functions return `nimcp_error_t` codes

### Occipital Lobe Integration (New in 2.6.1)
- Visual processing requires proper cortical bridge initialization
- Audiovisual bridge must be configured before cross-modal processing
- Cognitive bridge connects occipital to higher cognitive functions

### Path Changes (2.6.2)
- Project path: `/home/bbrelin/nimcp` (NOT `/home/bbrelin/repos/nimcp`)
