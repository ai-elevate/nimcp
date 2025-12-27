# Common Issues & Solutions

## CMake Subdirectory Conflicts

If `add_subdirectory()` causes duplicate target errors, check if the subdirectory is already added from `test/CMakeLists.txt`.

## Missing Test Files

CMakeLists may reference non-existent test files. Check with `ls` before adding tests.

## Bio-async Message

"Bio-async router not available, skipping registration" is normal/expected in tests.

## LNN Wiring

**GOTCHA**: Use `lnn_wiring_is_connected()` (inline alias) or `lnn_wiring_has_edge()` to check connectivity.

## Brain Immune B Cells

**GOTCHA**: B cells must be in PLASMA state to produce antibodies. Use `brain_immune_t_help_b()` to transition from ACTIVATED → PLASMA.

## Manual CMakeLists.txt Edits

Some modules require manual CMakeLists.txt edits. See module-specific documentation:
- `TRAINING_IMMUNE_BUILD_INSTRUCTIONS.md` for training-immune module

## Inconsistent Function Prefixes (Lower Priority)

The following function prefixes deviate from the `nimcp_*` naming convention but are kept for backward compatibility and API stability. Refactoring would require major version bump:

### `shannon_*` functions
- Location: `include/information/nimcp_shannon.h`, `include/information/immune/nimcp_shannon_immune_bridge.h`, `include/cognitive/global_workspace/nimcp_global_workspace_shannon.h`, `include/middleware/integration/nimcp_shannon_monitor.h`
- Ideal prefix: `nimcp_shannon_*`
- Examples: `shannon_channel_capacity()`, `shannon_entropy()`, `shannon_mutual_information()`

### `cross_modal_*` functions
- Location: `include/information/nimcp_cross_modal.h`, `include/information/immune/nimcp_cross_modal_immune_bridge.h`
- Ideal prefix: `nimcp_cross_modal_*`
- Examples: `cross_modal_analyze_channel()`, `cross_modal_compute_synergy()`

### `neural_network_*` functions
- Location: `include/core/neuralnet/nimcp_neuralnet.h`
- Ideal prefix: `nimcp_neural_network_*`
- Examples: `neural_network_create()`, `neural_network_forward()`, `neural_network_destroy()`

**Note**: These are documented for future major version refactoring. No action required for current development.
