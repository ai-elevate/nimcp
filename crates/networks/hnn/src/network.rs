//! Top-level [`HnnNetwork`] — owns the MLP, integrator timestep, and
//! the live `(q, p)` state.
//!
//! Forward dynamics: each call to [`HnnNetwork::step`] advances the
//! state by one symplectic Euler step using the MLP-defined
//! Hamiltonian. The MLP gradients are computed by reverse-mode autodiff
//! inside [`HamiltonianMlp::evaluate`].

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::integrator::symplectic_euler_step;
use crate::mlp::HamiltonianMlp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnnConfig {
    /// Number of canonical coordinates (q) and corresponding momenta (p).
    /// Total state vector length is `2 * dof`.
    pub dof: usize,
    /// Hidden widths of the Hamiltonian MLP. Empty → linear in `[q;p]`.
    pub hidden_layers: Vec<usize>,
    /// Integration timestep in dimensionless units.
    pub dt: f32,
    pub rng_seed: u64,
}

impl Default for HnnConfig {
    fn default() -> Self {
        Self {
            dof: 1,
            hidden_layers: vec![32, 32],
            dt: 0.01,
            rng_seed: 0xA1A1,
        }
    }
}

#[derive(Debug, Error)]
pub enum HnnError {
    #[error("hnn: dof must be > 0")]
    ZeroDof,
    #[error("hnn: dt must be > 0")]
    NonPositiveDt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnnNetwork {
    pub config: HnnConfig,
    pub mlp: HamiltonianMlp,
    /// Current generalised coordinates.
    pub q: Array1<f32>,
    /// Current canonical momenta.
    pub p: Array1<f32>,
}

impl HnnNetwork {
    pub fn new(config: HnnConfig) -> Result<Self, HnnError> {
        if config.dof == 0 {
            return Err(HnnError::ZeroDof);
        }
        if !(config.dt > 0.0) {
            return Err(HnnError::NonPositiveDt);
        }
        let mlp = HamiltonianMlp::new(config.dof, &config.hidden_layers, config.rng_seed);
        let q = Array1::<f32>::zeros(config.dof);
        let p = Array1::<f32>::zeros(config.dof);
        Ok(Self { config, mlp, q, p })
    }

    /// Reset state to zeros. Caller can mutate `q` / `p` afterward via
    /// [`HnnNetwork::set_state`].
    pub fn reset(&mut self) {
        self.q.fill(0.0);
        self.p.fill(0.0);
    }

    pub fn set_state(&mut self, q: Array1<f32>, p: Array1<f32>) {
        assert_eq!(q.len(), self.config.dof, "set_state: q dof mismatch");
        assert_eq!(p.len(), self.config.dof, "set_state: p dof mismatch");
        self.q = q;
        self.p = p;
    }

    /// Current Hamiltonian value (single forward through the MLP, no
    /// state mutation).
    pub fn energy(&self) -> f32 {
        let (h, _, _) = self.mlp.evaluate(&self.q, &self.p);
        h
    }

    /// Advance one symplectic Euler step using the MLP-defined `H`.
    /// Returns the energy at the *start* of the step (so the caller can
    /// log a series without an extra forward pass).
    pub fn step(&mut self) -> f32 {
        let mlp = &self.mlp;
        symplectic_euler_step(&mut self.q, &mut self.p, self.config.dt, |q, p| {
            mlp.evaluate(q, p)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_steps_without_panic_and_state_changes() {
        let cfg = HnnConfig {
            dof: 2,
            hidden_layers: vec![16, 16],
            dt: 0.005,
            rng_seed: 0xC0FE,
        };
        let mut net = HnnNetwork::new(cfg).unwrap();
        net.set_state(
            Array1::from_vec(vec![0.5, -0.3]),
            Array1::from_vec(vec![0.1, 0.2]),
        );
        let q0 = net.q.clone();
        let p0 = net.p.clone();
        for _ in 0..10 {
            net.step();
        }
        assert_ne!(net.q, q0, "q should evolve under stepping");
        assert_ne!(net.p, p0, "p should evolve under stepping");
        for v in net.q.iter().chain(net.p.iter()) {
            assert!(v.is_finite(), "step produced non-finite state: {v}");
        }
    }

    #[test]
    fn network_serde_round_trip() {
        let cfg = HnnConfig {
            dof: 1,
            hidden_layers: vec![8],
            dt: 0.01,
            rng_seed: 0xABCD,
        };
        let net = HnnNetwork::new(cfg).unwrap();
        let json = serde_json::to_string(&net).unwrap();
        let restored: HnnNetwork = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.config.dof, net.config.dof);
        assert_eq!(restored.q, net.q);
        // MLP weights match.
        for (wa, wb) in net.mlp.weights.iter().zip(restored.mlp.weights.iter()) {
            assert_eq!(wa, wb);
        }
    }
}
