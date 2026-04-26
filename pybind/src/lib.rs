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
//! Phase 7b surface (full ensemble + memory + ensemble checkpoint):
//!
//! - `Brain.from_json(config_json: str) -> Brain` — alternative constructor
//!   accepting a full `BrainConfig` as JSON. Use this to build a brain
//!   with SNN, LNN, and/or memory configured. Hand-writing the JSON is
//!   straightforward; the adapter shim builds it from harness configs.
//! - `brain.snn_step(drive, reward, dt_ms) -> int` — one SNN integration
//!   step. `drive` is `list[list[float]]` — one current vector per
//!   population. Returns total spikes this step.
//! - `brain.lnn_reset()` — clear LNN transient state.
//! - `brain.lnn_forward_step(input) -> list[float]` — one sample.
//! - `brain.lnn_forward_sequence(inputs) -> list[list[float]]` — multi-sample.
//! - `brain.lnn_train_step_mse(inputs, targets, lr, grad_clip) -> (loss, grad_norm)`
//! - `brain.memory_insert(id, features, t_ms, salience=0.0)`
//! - `brain.memory_mark_landmark(id, reason)`
//! - `brain.memory_query_all(query, k) -> list[dict]`
//! - `brain.memory_query_landmarks(query, k) -> list[dict]`
//! - `brain.memory_consolidate(dt_seconds) -> (promotions, demotions, evictions)`
//! - `brain.save_ensemble(dir)` / `brain.load_ensemble(dir)` — atomic
//!   directory checkpoint covering every configured subsystem.
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
use nimcp_brain::{Backend, Brain, BrainConfig};
use nimcp_lnn::TrainParams;
use nimcp_memory::MemoryNode;
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

fn parse_backend(s: &str) -> PyResult<Backend> {
    match s.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Backend::Cpu),
        "gpu" => Ok(Backend::Gpu),
        other => Err(PyValueError::new_err(format!(
            "unknown backend '{other}', expected 'cpu' or 'gpu'"
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
    ///     backend (str): execution backend, 'cpu' (default) or 'gpu'.
    ///         When 'gpu', SNN's `use_gpu_forward` is forced on and
    ///         LNN's `enable_gpu()` is called automatically. On a CPU-only
    ///         build (no `--features cuda`) 'gpu' degrades to CPU with a
    ///         warning rather than failing.
    #[new]
    #[pyo3(signature = (
        rng_seed = 0x5EED,
        deterministic = false,
        layers = None,
        activation = "relu".to_string(),
        backend = "cpu".to_string(),
    ))]
    fn new(
        rng_seed: u64,
        deterministic: bool,
        layers: Option<Vec<usize>>,
        activation: String,
        backend: String,
    ) -> PyResult<Self> {
        let act = parse_activation(&activation)?;
        let backend_enum = parse_backend(&backend)?;
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
            backend: backend_enum,
            ..Default::default()
        };
        let inner = Brain::new(cfg).map_err(rt_err)?;
        Ok(Self { inner })
    }

    /// Currently-active backend, as a string. One of `"cpu"` / `"gpu"`.
    /// Mirrors the `backend` arg passed to the constructor.
    #[getter]
    fn backend(&self) -> &'static str {
        match self.inner.config().backend {
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
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

    // -------------------------------------------------------------------------
    // Phase 7b — full-ensemble constructor + SNN / LNN / memory surface.
    // -------------------------------------------------------------------------

    /// Alternative constructor accepting a full `BrainConfig` as JSON.
    /// The Python caller builds a dict mirroring the `BrainConfig`
    /// struct and hands `json.dumps(cfg)`; use this path to enable
    /// SNN / LNN / memory on the brain.
    ///
    /// Args:
    ///     config_json (str): JSON serialization of `BrainConfig`.
    /// Returns:
    ///     Brain
    ///
    /// Raises:
    ///     ValueError: JSON didn't decode to a `BrainConfig`.
    ///     RuntimeError: Brain construction failed (shape / config errors).
    #[staticmethod]
    fn from_json(config_json: &str) -> PyResult<Self> {
        let cfg: BrainConfig = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("BrainConfig JSON decode: {e}")))?;
        let inner = Brain::new(cfg).map_err(rt_err)?;
        Ok(Self { inner })
    }

    /// One SNN integration step.
    ///
    /// Args:
    ///     drive  (list[list[float]]): external `I_syn` current per population.
    ///         Outer list length must equal the number of populations; each
    ///         inner list is either empty (no external drive for that
    ///         population) or length-equal to the population size.
    ///     reward (float): global reward signal for R-STDP.
    ///     dt_ms  (float): integration step in milliseconds.
    /// Returns:
    ///     int: total spikes emitted this step.
    fn snn_step(&mut self, drive: Vec<Vec<f32>>, reward: f32, dt_ms: f32) -> PyResult<u32> {
        let slices: Vec<&[f32]> = drive.iter().map(Vec::as_slice).collect();
        self.inner.snn_step(&slices, reward, dt_ms).map_err(rt_err)
    }

    /// Reset the LNN's transient state to zeros. No-op on brains without LNN.
    fn lnn_reset(&mut self) {
        self.inner.lnn_reset();
    }

    /// Step the LNN forward one sample; returns readout.
    /// State carries across calls; use `lnn_reset()` to start a fresh sequence.
    fn lnn_forward_step(&mut self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let x = Array1::from_vec(input);
        self.inner
            .lnn_forward_step(&x)
            .map(|out| out.to_vec())
            .map_err(rt_err)
    }

    /// Run the LNN over a full sequence (resets state first).
    /// Returns per-step readouts.
    fn lnn_forward_sequence(&mut self, inputs: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let arrays: Vec<Array1<f32>> = inputs.into_iter().map(Array1::from_vec).collect();
        self.inner
            .lnn_forward_sequence(&arrays)
            .map(|outs| outs.into_iter().map(|a| a.to_vec()).collect())
            .map_err(rt_err)
    }

    /// One LNN training step (MSE over sequence). Returns `(loss, grad_norm)`.
    #[pyo3(signature = (inputs, targets, lr = 0.01, grad_clip = 1.0))]
    fn lnn_train_step_mse(
        &mut self,
        inputs: Vec<Vec<f32>>,
        targets: Vec<Vec<f32>>,
        lr: f32,
        grad_clip: f32,
    ) -> PyResult<(f32, f32)> {
        if inputs.len() != targets.len() {
            return Err(shape_err(format!(
                "len(inputs)={} != len(targets)={}",
                inputs.len(),
                targets.len()
            )));
        }
        let inputs_a: Vec<Array1<f32>> = inputs.into_iter().map(Array1::from_vec).collect();
        let targets_a: Vec<Array1<f32>> = targets.into_iter().map(Array1::from_vec).collect();
        let params = TrainParams { lr, grad_clip };
        self.inner
            .lnn_train_step_mse(&inputs_a, &targets_a, &params)
            .map_err(rt_err)
    }

    /// Insert a new memory node into the Z-Ladder.
    ///
    /// Args:
    ///     id (int): caller-assigned unique ID.
    ///     features (list[float]): payload vector; round-trips
    ///         bit-for-bit through checkpoints.
    ///     t_ms (int): insertion time in milliseconds (used for age).
    ///     salience (float, optional): initial salience in [0, 1].
    ///         Higher values make Z0→Z1 promotion more likely.
    #[pyo3(signature = (id, features, t_ms, salience = 0.0))]
    fn memory_insert(
        &mut self,
        id: u64,
        features: Vec<f32>,
        t_ms: u64,
        salience: f32,
    ) -> PyResult<()> {
        let mut node = MemoryNode::new(id, features, t_ms);
        node.salience = salience.clamp(0.0, 1.0);
        self.inner.memory_insert(node).map_err(rt_err)
    }

    /// Promote a node to Z3 and protect it from demotion indefinitely.
    fn memory_mark_landmark(&mut self, id: u64, reason: &str) -> PyResult<()> {
        self.inner.memory_mark_landmark(id, reason).map_err(rt_err)
    }

    /// Top-`k` cosine similarity matches across every tier.
    ///
    /// Returns:
    ///     list[dict]: each entry is `{node_id, similarity, tier}`,
    ///     sorted descending by similarity.
    fn memory_query_all<'py>(
        &self,
        py: Python<'py>,
        query: Vec<f32>,
        k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let hits = self.inner.memory_query_all(&query, k).map_err(rt_err)?;
        let json = serde_json::to_string(&hits)
            .map_err(|e| PyRuntimeError::new_err(format!("QueryHit encode: {e}")))?;
        py.import("json")?.call_method1("loads", (json,))
    }

    /// Top-`k` cosine similarity matches across the landmark subset only.
    fn memory_query_landmarks<'py>(
        &self,
        py: Python<'py>,
        query: Vec<f32>,
        k: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let hits = self
            .inner
            .memory_query_landmarks(&query, k)
            .map_err(rt_err)?;
        let json = serde_json::to_string(&hits)
            .map_err(|e| PyRuntimeError::new_err(format!("QueryHit encode: {e}")))?;
        py.import("json")?.call_method1("loads", (json,))
    }

    /// Run one consolidation tick: age-decay, promote / demote / evict.
    ///
    /// Args:
    ///     dt_seconds (float): wall-clock time since last consolidate.
    /// Returns:
    ///     tuple[int, int, int]: (total_promotions, total_demotions, total_evictions)
    ///     applied by this tick.
    ///
    /// Raises:
    ///     RuntimeError: no memory subsystem on this brain.
    fn memory_consolidate(&mut self, dt_seconds: f32) -> PyResult<(usize, usize, usize)> {
        let mem = self
            .inner
            .memory_mut()
            .ok_or_else(|| rt_err("memory not configured on this brain"))?;
        Ok(mem.consolidate(dt_seconds))
    }

    /// Atomic ensemble checkpoint. Writes `<dir>/{adaptive.rkyv,
    /// snn.json, lnn.json, memory.json, manifest.json}` via a temp dir
    /// + rename, so the old state survives any write failure.
    fn save_ensemble(&self, dir: PathBuf) -> PyResult<()> {
        self.inner.save_ensemble(&dir).map_err(rt_err)
    }

    /// Restore a full ensemble from a directory previously written by
    /// `save_ensemble()`. Shape mismatches (layer widths, SNN pop sizes,
    /// memory capacities) surface as `RuntimeError`.
    fn load_ensemble(&mut self, dir: PathBuf) -> PyResult<()> {
        self.inner.load_ensemble(&dir).map_err(rt_err)
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
