//! NIMCP V2 — Python bindings.
//!
//! Phase 1 surface:
//!
//! - `Brain(rng_seed, deterministic, layers, activation)` — construct.
//! - `brain.learn(x, y, lr) -> float` — one SGD step against MSE.
//! - `brain.predict(x) -> list[float]` — forward pass.
//! - `brain.save(path)` / `brain.load(path)` — rkyv checkpoint round-trip.
//!
//! Phase 6b surface (introspection):
//!
//! - `brain.stats() -> dict` — nested dict with one section per configured
//!   subsystem (`adaptive`, `snn`, `lnn`, `memory`). Sections for
//!   subsystems the brain wasn't built with are `None`.
//! - `brain.stats_json() -> str` — same payload as `stats()` but as the
//!   raw JSON string, for callers that want to log / pipe / save.
//!
//! Later phases add SNN config, memory/landmark API, streaming sensory I/O.
//!
//! Built with `maturin develop` for editable installs.

#![allow(unsafe_code)] // pyo3 macros expand to unsafe
#![allow(unsafe_op_in_unsafe_fn)] // pyo3 macros
// PyO3 extractors produce Rust-owned values (String, Vec<T>) that we
// consume once and drop. clippy::needless_pass_by_value is a false
// positive for bindings.
#![allow(clippy::needless_pass_by_value)]

use ndarray::Array1;
use nimcp_adaptive::{Activation, AdaptiveConfig};
use nimcp_brain::{Brain, BrainConfig};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

/// Python-visible brain handle.
#[pyclass(name = "Brain", module = "nimcp_v2")]
struct PyBrain {
    inner: Brain,
}

fn parse_activation(s: &str) -> PyResult<Activation> {
    match s.to_ascii_lowercase().as_str() {
        "relu" => Ok(Activation::Relu),
        "tanh" => Ok(Activation::Tanh),
        other => Err(PyValueError::new_err(format!(
            "unknown activation '{other}', expected 'relu' or 'tanh'"
        ))),
    }
}

fn shape_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}

fn rt_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

#[pymethods]
impl PyBrain {
    /// Create a new brain.
    ///
    /// Args:
    ///     rng_seed (int): seed for deterministic init. Default: 0x5EED.
    ///     deterministic (bool): run in deterministic mode. Default: False.
    ///     layers (list[int]): MLP widths including input + output. Default: [64, 32, 10].
    ///     activation (str): 'relu' or 'tanh' for hidden activations. Default: 'relu'.
    #[new]
    #[pyo3(signature = (
        rng_seed = 0x5EED,
        deterministic = false,
        layers = None,
        activation = "relu".to_string(),
    ))]
    fn new(
        rng_seed: u64,
        deterministic: bool,
        layers: Option<Vec<usize>>,
        activation: String,
    ) -> PyResult<Self> {
        let act = parse_activation(&activation)?;
        let adaptive = AdaptiveConfig {
            layers: layers.unwrap_or_else(|| vec![64, 32, 10]),
            rng_seed,
            activation: act,
        };
        if adaptive.layers.len() < 2 {
            return Err(PyValueError::new_err(
                "layers must have at least 2 entries (input + output)",
            ));
        }
        let cfg = BrainConfig {
            rng_seed,
            deterministic,
            adaptive,
            ..Default::default()
        };
        let inner = Brain::new(cfg).map_err(rt_err)?;
        Ok(Self { inner })
    }

    /// One gradient step; returns pre-update MSE loss.
    ///
    /// Args:
    ///     features (list[float]): input vector, len == layers[0].
    ///     target   (list[float]): target vector, len == layers[-1].
    ///     lr       (float, optional): learning rate. Default: 0.01.
    #[pyo3(signature = (features, target, lr = 0.01))]
    fn learn(&mut self, features: Vec<f32>, target: Vec<f32>, lr: f32) -> PyResult<f32> {
        let cfg = self.inner.config();
        let expected_in = cfg.adaptive.layers[0];
        let expected_out = cfg.adaptive.layers[cfg.adaptive.layers.len() - 1];
        if features.len() != expected_in {
            return Err(shape_err(format!(
                "features len {} != input layer {}",
                features.len(),
                expected_in
            )));
        }
        if target.len() != expected_out {
            return Err(shape_err(format!(
                "target len {} != output layer {}",
                target.len(),
                expected_out
            )));
        }
        let x = Array1::from_vec(features);
        let y = Array1::from_vec(target);
        Ok(self.inner.learn(&x, &y, lr))
    }

    /// Forward pass.
    ///
    /// Args:
    ///     features (list[float]): input vector, len == layers[0].
    /// Returns:
    ///     list[float]: output of the (linear) final layer.
    fn predict(&self, features: Vec<f32>) -> PyResult<Vec<f32>> {
        let cfg = self.inner.config();
        let expected_in = cfg.adaptive.layers[0];
        if features.len() != expected_in {
            return Err(shape_err(format!(
                "features len {} != input layer {}",
                features.len(),
                expected_in
            )));
        }
        let x = Array1::from_vec(features);
        Ok(self.inner.predict(&x).to_vec())
    }

    /// Save the brain's weights to `path`. Phase 1: rkyv over adaptive net.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(&path).map_err(rt_err)
    }

    /// Reload weights from a previous `save`. Layer shapes must match the
    /// brain's current config.
    fn load(&mut self, path: PathBuf) -> PyResult<()> {
        self.inner.load(&path).map_err(rt_err)
    }

    /// Tuple of (input_dim, output_dim) read from config.
    fn shape(&self) -> (usize, usize) {
        let layers = &self.inner.config().adaptive.layers;
        (layers[0], layers[layers.len() - 1])
    }

    /// Comprehensive read-only stats snapshot across every configured
    /// subsystem. Returns a nested Python dict with sections
    /// `rng_seed`, `adaptive`, `snn`, `lnn`, `memory`. Sections for
    /// subsystems not configured on this brain are `None`.
    ///
    /// Cheap — linear in parameter / node count — but not free. Callers
    /// polling on a tight loop should throttle.
    ///
    /// Returns:
    ///     dict: see `nimcp_brain::stats::BrainStats` for the schema.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json = self.inner.stats_json().map_err(rt_err)?;
        let json_mod = py.import("json")?;
        json_mod.call_method1("loads", (json,))
    }

    /// Same payload as `stats()` but returned as a raw JSON string.
    /// Faster than `stats()` when the caller is going to write it to
    /// disk / a socket / a log without ever touching the structure.
    fn stats_json(&self) -> PyResult<String> {
        self.inner.stats_json().map_err(rt_err)
    }

    /// String representation for Python repr.
    fn __repr__(&self) -> String {
        let cfg = self.inner.config();
        format!(
            "<Brain seed={} deterministic={} layers={:?} activation={:?}>",
            cfg.rng_seed, cfg.deterministic, cfg.adaptive.layers, cfg.adaptive.activation
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
