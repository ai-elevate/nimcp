# Function Distribution Map

This document maps all functions from the original `nimcp.c` to their new module locations.

## Core Functions (remain in nimcp.c)

### Initialization & Shutdown
- `nimcp_init()` - Initialize NIMCP library
- `nimcp_shutdown()` - Shutdown NIMCP library

### Version Information
- `nimcp_version()` - Get version string
- `nimcp_version_int()` - Get version as integer

### Error Handling
- `set_error()` - Internal error setter (exported to modules)
- `nimcp_get_error()` - Get last error message

## nimcp_api_brain.c (Brain Lifecycle & Management)

### Brain Creation & Destruction
- `nimcp_brain_create()` - Create brain instance
- `nimcp_brain_destroy()` - Destroy brain instance
- `nimcp_brain_create_from_config()` - Create from config file

### File I/O
- `nimcp_brain_save()` - Save brain to file
- `nimcp_brain_load()` - Load brain from file

### Named Snapshots
- `nimcp_brain_snapshot_save()` - Save named snapshot
- `nimcp_brain_snapshot_restore()` - Restore from snapshot
- `nimcp_brain_snapshot_list()` - List snapshots
- `nimcp_brain_snapshot_delete()` - Delete snapshot

### Copy-on-Write (COW) Operations
- `nimcp_brain_clone_cow()` - Clone with COW
- `nimcp_brain_snapshot_cow()` - Create COW snapshot
- `nimcp_brain_restore_cow()` - Restore from COW snapshot
- `nimcp_brain_snapshot_destroy_cow()` - Destroy COW snapshot

### Brain Probing & Metrics
- `nimcp_brain_probe()` - Get comprehensive brain stats
- `nimcp_brain_get_neuron_count()` - Get neuron count
- `nimcp_brain_get_utilization_metrics()` - Get utilization

### Dynamic Resizing
- `nimcp_brain_resize()` - Resize to specific neuron count
- `nimcp_brain_auto_resize()` - Auto-resize based on load

## nimcp_api_inference.c (Inference & Learning)

### Inference Operations
- `nimcp_brain_predict()` - Predict with label and confidence
- `nimcp_brain_infer()` - Get raw output vector

### Learning Operations
- `nimcp_brain_learn_example()` - Learn from labeled example

## nimcp_api_network.c (Neural Network)

### Network Management
- `nimcp_network_create()` - Create neural network
- `nimcp_network_destroy()` - Destroy neural network

### Network Operations
- `nimcp_network_forward()` - Forward propagation
- `nimcp_network_train()` - Network training (placeholder)

## nimcp_api_cognitive.c (Cognitive Systems)

### Working Memory
- `nimcp_brain_working_memory_add()` - Add to working memory
- `nimcp_brain_working_memory_get()` - Get from working memory
- `nimcp_brain_working_memory_stats()` - Get WM statistics
- `nimcp_brain_working_memory_refresh()` - Refresh item (rehearsal)

### Global Workspace Theory
- `nimcp_brain_workspace_compete()` - Compete for broadcast
- `nimcp_brain_workspace_read()` - Read broadcast content
- `nimcp_brain_workspace_subscribe()` - Subscribe module
- `nimcp_brain_workspace_unsubscribe()` - Unsubscribe module
- `nimcp_brain_workspace_has_broadcast()` - Check broadcast status
- `nimcp_brain_workspace_stats()` - Get workspace statistics

### Ethics Engine
- `nimcp_ethics_create()` - Create ethics engine
- `nimcp_ethics_destroy()` - Destroy ethics engine
- `nimcp_ethics_check()` - Evaluate ethical score

### Knowledge System
- `nimcp_knowledge_create()` - Create knowledge system
- `nimcp_knowledge_destroy()` - Destroy knowledge system
- `nimcp_knowledge_add_fact()` - Add fact to KB
- `nimcp_knowledge_query()` - Query knowledge base

## nimcp_api_oscillation.c (Complex Oscillations)

### Oscillation Control
- `nimcp_enable_complex_oscillations()` - Enable/disable
- `nimcp_is_complex_oscillations_enabled()` - Check status

### Phasor Operations
- `nimcp_get_oscillation_phasor()` - Get neuron phasor

### Phase Analysis
- `nimcp_get_phase_coherence()` - Compute coherence
- `nimcp_get_pac_modulation()` - Phase-amplitude coupling

## nimcp_api_training.c (Training Pipeline)

### Training Configuration
- `nimcp_training_config_default()` - Default config
- `nimcp_brain_configure_training()` - Configure pipeline

### Training Execution
- `nimcp_brain_train_step()` - Single training step
- `nimcp_brain_train_batch()` - Batch training
- `nimcp_brain_get_training_stats()` - Get statistics
- `nimcp_brain_step_scheduler()` - Step LR scheduler

### Training Callbacks
- `nimcp_callback_config_default()` - Default callback config
- `nimcp_brain_enable_callbacks()` - Enable callback system
- `nimcp_brain_disable_callbacks()` - Disable callbacks
- `nimcp_brain_register_callback()` - Register callback
- `nimcp_brain_unregister_callback()` - Unregister callback
- `nimcp_brain_get_callback_stats()` - Callback statistics

## Summary Statistics

| Module | Functions | Lines | Responsibility |
|--------|-----------|-------|----------------|
| nimcp.c (core) | 4 | ~250 | Init, shutdown, error handling |
| nimcp_api_brain.c | 18 | 659 | Brain lifecycle & snapshots |
| nimcp_api_inference.c | 3 | 218 | Prediction & learning |
| nimcp_api_network.c | 4 | 138 | Standalone networks |
| nimcp_api_cognitive.c | 18 | 675 | Cognitive systems |
| nimcp_api_oscillation.c | 5 | 295 | Complex oscillations |
| nimcp_api_training.c | 12 | 929 | Training pipeline |
| **TOTAL** | **64** | **3164** | **Complete API** |

## Module Dependencies

```
External Dependencies per Module:

nimcp_api_brain.c:
  - core/brain/nimcp_brain.h
  - core/brain/nimcp_brain_internal.h
  - utils/config/nimcp_config.h
  - utils/cache/nimcp_cache.h
  - security/nimcp_blood_brain_barrier.h

nimcp_api_inference.c:
  - core/brain/nimcp_brain.h
  - core/brain/nimcp_brain_internal.h
  - security/nimcp_blood_brain_barrier.h

nimcp_api_network.c:
  - core/neuralnet/nimcp_neuralnet.h

nimcp_api_cognitive.c:
  - cognitive/nimcp_working_memory.h
  - cognitive/global_workspace/nimcp_global_workspace.h
  - cognitive/ethics/nimcp_ethics.h
  - cognitive/knowledge/nimcp_knowledge.h

nimcp_api_oscillation.c:
  - core/brain/oscillations/nimcp_brain_complex_oscillations.h

nimcp_api_training.c:
  - middleware/training/nimcp_brain_training_integration.h
  - middleware/training/nimcp_loss_functions.h
  - middleware/training/nimcp_optimizers.h
  - middleware/training/nimcp_lr_scheduler.h
  - middleware/training/nimcp_training_callbacks.h
  - plasticity/adaptive/nimcp_adaptive.h
```

All modules share:
- async/nimcp_bio_async.h
- utils/logging/nimcp_logging.h
- utils/memory/nimcp_unified_memory.h
- utils/error/nimcp_error_codes.h
