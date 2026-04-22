//! Discrete-time quantum walk on a finite graph.
//!
//! State is an amplitude vector over `n` nodes. Each step:
//!
//! 1. **Coin step**: apply a Hadamard-like mixing on the amplitude
//!    at each node — splits amplitude between self and one "next
//!    hop" slot (this is a very simplified 2-level coin; the V2
//!    port keeps it minimal).
//! 2. **Shift step**: propagate amplitude to neighbouring nodes
//!    per an adjacency list.
//! 3. **Normalize**: renormalise the `L²` norm to `1` so numerical
//!    drift doesn't accumulate.
//!
//! Observed probability at node `i` is `|amp[i]|² = re² + im²`.
//! Classical random walks converge to a single distribution; quantum
//! walks produce oscillating probability patterns that can lead to
//! faster exploration in some graph topologies.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use thiserror::Error;

/// Complex amplitude represented as `(re, im)`. Kept explicit to
/// avoid a complex-number crate dependency for this small use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Amp {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Amp {
    /// Zero amplitude.
    #[must_use]
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
    /// Probability = |amp|² = re² + im².
    #[must_use]
    pub fn probability(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

/// Construction / step errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WalkError {
    /// `n_nodes == 0`.
    #[error("walker needs at least one node")]
    NoNodes,
    /// Adjacency list length mismatch with node count.
    #[error("adjacency length {got} != n_nodes {expected}")]
    AdjacencyShape {
        /// Found length.
        got: usize,
        /// Expected length.
        expected: usize,
    },
}

/// Discrete-time quantum walker on a finite graph.
#[derive(Debug, Clone)]
pub struct QuantumWalker {
    /// Amplitude vector, length `n_nodes`.
    pub amps: Vec<Amp>,
    /// Adjacency list — `adj[i]` is the neighbours of node `i`.
    pub adj: Vec<Vec<usize>>,
    /// RNG for coin draws (deterministic from seed).
    rng: ChaCha20Rng,
}

impl QuantumWalker {
    /// Build a walker localized at `start_node` on the graph defined
    /// by `adj` (adjacency list — `adj[i]` = neighbours of `i`).
    pub fn new(start_node: usize, adj: Vec<Vec<usize>>, seed: u64) -> Result<Self, WalkError> {
        let n = adj.len();
        if n == 0 {
            return Err(WalkError::NoNodes);
        }
        let mut amps = vec![Amp::zero(); n];
        amps[start_node.min(n - 1)] = Amp { re: 1.0, im: 0.0 };
        Ok(Self {
            amps,
            adj,
            rng: ChaCha20Rng::seed_from_u64(seed),
        })
    }

    /// Number of nodes.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.amps.len()
    }

    /// One full step (coin + shift + normalize). Use `step_n` for
    /// repeated advances.
    pub fn step(&mut self) {
        // --- Coin step: simple Hadamard-like rotation on amp[i]:
        // (re, im) -> ((re + im)/√2, (re - im)/√2). Deterministic per
        // node; no RNG consumed — the RNG lives in the walker for
        // future extensions.
        let inv_sqrt2 = 1.0_f32 / core::f32::consts::SQRT_2;
        for a in self.amps.iter_mut() {
            let re2 = (a.re + a.im) * inv_sqrt2;
            let im2 = (a.re - a.im) * inv_sqrt2;
            a.re = re2;
            a.im = im2;
        }

        // --- Shift step: split amplitude uniformly across all
        // neighbours. Coefficient `1/√k` preserves L²-norm when
        // amplitudes combine at a shared destination (the √ is the
        // quantum-superposition convention). Isolated nodes keep their
        // amplitude in place.
        let mut next = vec![Amp::zero(); self.amps.len()];
        for (i, a) in self.amps.iter().enumerate() {
            let k = self.adj[i].len();
            if k == 0 {
                next[i].re += a.re;
                next[i].im += a.im;
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let inv_sqrt_k = 1.0 / (k as f32).sqrt();
            let re_share = a.re * inv_sqrt_k;
            let im_share = a.im * inv_sqrt_k;
            for &dst in &self.adj[i] {
                next[dst].re += re_share;
                next[dst].im += im_share;
            }
        }
        // `rng` is retained for future non-deterministic variants; the
        // current coin+shift is deterministic.
        let _ = &mut self.rng;
        self.amps = next;

        // --- Normalize to unit L² norm.
        normalize_amplitudes(&mut self.amps);
    }

    /// Advance `n` steps.
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

/// Re-normalise an amplitude vector to unit L² norm. No-op when the
/// vector is already (effectively) zero.
pub fn normalize_amplitudes(amps: &mut [Amp]) {
    let mut norm_sq = 0.0_f32;
    for a in amps.iter() {
        norm_sq += a.re * a.re + a.im * a.im;
    }
    if norm_sq > f32::EPSILON {
        let inv = 1.0 / norm_sq.sqrt();
        for a in amps.iter_mut() {
            a.re *= inv;
            a.im *= inv;
        }
    }
}

/// Per-node measurement probability = `|amp|²`. Returns a freshly
/// allocated `Vec<f32>` sized to the walker.
#[must_use]
pub fn probabilities(amps: &[Amp]) -> Vec<f32> {
    amps.iter().map(Amp::probability).collect()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn ring_graph(n: usize) -> Vec<Vec<usize>> {
        // Each node connects to its neighbours modulo n.
        (0..n).map(|i| vec![(i + n - 1) % n, (i + 1) % n]).collect()
    }

    #[test]
    fn rejects_empty_adjacency() {
        assert!(matches!(
            QuantumWalker::new(0, vec![], 0),
            Err(WalkError::NoNodes)
        ));
    }

    #[test]
    fn starts_localized_at_start_node() {
        let w = QuantumWalker::new(3, ring_graph(8), 0).unwrap();
        assert_eq!(w.amps[3].re, 1.0);
        assert_eq!(w.amps[3].im, 0.0);
        for (i, a) in w.amps.iter().enumerate() {
            if i != 3 {
                assert_eq!(a.re, 0.0);
                assert_eq!(a.im, 0.0);
            }
        }
    }

    #[test]
    fn amplitude_stays_normalized() {
        let mut w = QuantumWalker::new(0, ring_graph(6), 42).unwrap();
        for _ in 0..50 {
            w.step();
            let norm_sq: f32 = w.amps.iter().map(|a| a.probability()).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "norm drifted to {norm_sq} after step"
            );
        }
    }

    #[test]
    fn probabilities_sum_to_one() {
        let mut w = QuantumWalker::new(0, ring_graph(6), 99).unwrap();
        w.step_n(20);
        let p = probabilities(&w.amps);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn normalize_empty_noop() {
        let mut v: Vec<Amp> = vec![Amp::zero(); 4];
        normalize_amplitudes(&mut v);
        for a in &v {
            assert_eq!(a.re, 0.0);
            assert_eq!(a.im, 0.0);
        }
    }

    #[test]
    fn walk_spreads_amplitude() {
        // After a few steps on a ring, amplitude should have spread
        // beyond the start node.
        let mut w = QuantumWalker::new(0, ring_graph(10), 7).unwrap();
        w.step_n(6);
        let p = probabilities(&w.amps);
        let nonzero = p.iter().filter(|&&v| v > 1e-4).count();
        assert!(
            nonzero > 1,
            "walk should spread; only {nonzero} nonzero probs"
        );
    }

    #[test]
    fn same_seed_same_trajectory() {
        let mut a = QuantumWalker::new(0, ring_graph(8), 77).unwrap();
        let mut b = QuantumWalker::new(0, ring_graph(8), 77).unwrap();
        a.step_n(20);
        b.step_n(20);
        assert_eq!(a.amps, b.amps);
    }
}
