//! NIMCP V2 — Python bindings.
//!
//! Exposes a minimal surface. Expanded per-phase:
//!
//! - Phase 0: `Brain()` constructor (no-op)
//! - Phase 1: `.learn(x, y)`, `.predict(x)`, `.save(path)`, `.load(path)`
//! - Phase 3: SNN config options
//! - Phase 5: memory/landmark API
//!
//! Built with `maturin develop` for editable installs.

#![allow(unsafe_code)] // pyo3 macros expand to unsafe
#![allow(unsafe_op_in_unsafe_fn)] // pyo3 macros again

use nimcp_brain::{Brain, BrainConfig};
use pyo3::prelude::*;

/// Python-visible brain handle.
#[pyclass(name = "Brain", module = "nimcp_v2")]
struct PyBrain {
    inner: Brain,
}

#[pymethods]
impl PyBrain {
    /// Create a new brain with default config.
    ///
    /// Args:
    ///     rng_seed (int): seed for deterministic init. Default: 0x5EED.
    ///     deterministic (bool): run in deterministic mode. Default: False.
    #[new]
    #[pyo3(signature = (rng_seed = 0x5EED, deterministic = false))]
    fn new(rng_seed: u64, deterministic: bool) -> PyResult<Self> {
        let cfg = BrainConfig {
            rng_seed,
            deterministic,
            ..Default::default()
        };
        let inner = Brain::new(cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("brain init: {e}")))?;
        Ok(Self { inner })
    }

    /// String representation for Python repr.
    fn __repr__(&self) -> String {
        format!(
            "<Brain seed={} deterministic={}>",
            self.inner.config().rng_seed,
            self.inner.config().deterministic,
        )
    }
}

/// Python module initializer.
#[pymodule]
fn nimcp_v2(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBrain>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
