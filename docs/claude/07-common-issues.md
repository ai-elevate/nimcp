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
