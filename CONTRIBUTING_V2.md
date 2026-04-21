# Contributing to NIMCP V2

Conventions every crate must follow. Scope: the Rust workspace at the root of
this repo on the `v2` branch.

## 1. Language + toolchain

- Rust 2024 edition (per `Cargo.toml` workspace.package.edition)
- MSRV: 1.85
- `rustfmt` + `clippy` clean before any commit:
  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets --no-default-features -- -D warnings
  cargo test --workspace --no-default-features
  ```

## 2. `unsafe`

- Workspace lint: `unsafe_code = "warn"`.
- Allowed only in:
  - `crates/gpu` (CUDA FFI)
  - `pybind` (PyO3 expands to `unsafe`)
- Every `unsafe` block needs a `SAFETY:` comment explaining the invariant.

## 3. Errors

- Domain crates: use `thiserror` to define a crate-local error enum.
- Top-level integration (`brain`, `pybind`, examples): use `eyre` / `anyhow`.
- Return `Result<T, Error>` — no panics except in tests and genuinely
  impossible states.

## 4. Tracing

- Use `tracing` everywhere, never `println!` (except in examples).
- Log levels:
  - `ERROR`: unrecoverable; caller loses work.
  - `WARN`: recovered, but surprising.
  - `INFO`: one-time lifecycle events (actor start/stop, checkpoint saved).
  - `DEBUG`: per-step detail, turned off in production.
  - `TRACE`: per-event, never in hot paths without a feature gate.

## 5. Determinism

- Any random source must take a seed. Use `rand_chacha::ChaCha20Rng`.
- No `std::time::Instant` for logic decisions — only for diagnostics.
  Logic uses virtual time from the scheduler when `deterministic = true`.

## 6. Naming

- Types: `PascalCase`
- Traits: `PascalCase`, no `I` prefix
- Functions / methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Crate names: `nimcp-<domain>` (hyphen); crate identifier: `nimcp_<domain>`
- IDs: newtype over `u64` (see `nimcp_core::ids`) — never bare `u64`.

## 7. Testing

- Unit tests in `#[cfg(test)] mod tests` next to the code they test.
- Integration tests in `tests/` at the workspace root.
- Property tests via `proptest` for anything with algebraic invariants.
- Toy tests: every feature must work at a 100-scale variant before the
  real-scale version is merged.
- No flaky tests. If a test relies on wall-clock, use virtual time or
  ignore it.

## 8. Public API

- Everything `pub` must have a doc comment.
- Every crate root `lib.rs` must have a module-level `//! ...` doc block
  summarizing purpose + scope.
- Public types implement `Debug` unless there's a reason not to.
- Public types that wrap state implement `Clone` only if cloning is cheap.

## 9. Dependencies

- Add to `[workspace.dependencies]` in the root `Cargo.toml` when 2+ crates
  need it. Otherwise, put it in the crate's own `Cargo.toml`.
- Version pinning: minor version (e.g. `"1.42"`). Patch version floats.
- No `*` versions. No git dependencies (except for unreleased fixes in
  deps we own).

## 10. Commits on the `v2` branch

- Commit messages: `type(crate): summary` — e.g. `feat(scheduler): deterministic replay mode`.
- One logical change per commit.
- Tests for every change (exceptions: pure refactors with existing coverage).
- Every commit passes `cargo build && cargo test` on its own (bisect-friendly).

## 11. Cross-crate coordination

- Only `core` is allowed to define traits used by other crates.
- If you need a type in crate B that you created in crate A, it belongs in `core`.
- Depend on peer crates via path in `[workspace.dependencies]`, never a
  version number.

## 12. Performance

- Hot paths documented with a `// HOT PATH:` comment.
- Benchmarks for hot paths in `benches/` using `criterion`.
- Target: within 2× V1 on equivalent workloads for the same phase.

## 13. Phase discipline

A feature enters a crate only if its phase (per `docs/V2_PLAN.md`) has
started. Post-phase drift is forbidden — if you find yourself needing a
Phase 3 feature during Phase 1, stop and escalate rather than sneak it in.
