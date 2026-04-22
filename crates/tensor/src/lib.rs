//! NIMCP V2 — tensor helpers.
//!
//! Port of the V1 `utils/tensor/nimcp_tensor.h` convenience surface
//! onto ndarray. V2 already has ndarray as the canonical tensor
//! substrate; this crate provides V1-style factory / manipulation
//! helpers so ports from V1 code don't need to rewrite every
//! `nimcp_tensor_zeros` → `Array2::zeros` call.
//!
//! # Conventions
//!
//! - `DType` is an enum — V1 had `NIMCP_TENSOR_F32`, `NIMCP_TENSOR_F64`,
//!   `NIMCP_TENSOR_I32`. V2 surfaces the three via three parallel
//!   factory families: `f32_*`, `f64_*`, `i32_*`.
//! - Shape is always `&[usize]` (`ndarray::IxDyn`) so callers don't
//!   have to commit to 1D/2D/3D at the helper level.
//! - Factories that take an RNG seed use `ChaCha20Rng` — matches the
//!   V2 determinism convention and the SNN / LNN init pattern.
//! - Operations with two inputs (add/mul/dot/…) are ndarray's own
//!   operator overloads; no need to port them. What this crate
//!   ports is the **construction / reshape / I/O helpers** V1
//!   bundled into its tensor type.

#![forbid(unsafe_code)]

use ndarray::{Array, ArrayD, IxDyn};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Data type for a [`TensorView`] — matches V1's `nimcp_dtype_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 32-bit signed int.
    I32,
}

/// Construction / manipulation errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TensorError {
    /// Shape product overflows `usize`.
    #[error("shape overflow")]
    ShapeOverflow,
    /// Reshape target doesn't have the same number of elements as source.
    #[error("reshape size mismatch: from {from} to {to}")]
    ReshapeSize {
        /// Source element count.
        from: usize,
        /// Target element count.
        to: usize,
    },
    /// Empty shape (zero dimensions).
    #[error("empty shape")]
    EmptyShape,
}

// =========================================================================
// F32 factories
// =========================================================================

/// All-zeros f32 tensor of the given shape.
#[must_use]
pub fn f32_zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::zeros(IxDyn(shape))
}

/// All-ones f32 tensor.
#[must_use]
pub fn f32_ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0)
}

/// All-`value` f32 tensor.
#[must_use]
pub fn f32_full(shape: &[usize], value: f32) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), value)
}

/// Identity matrix (2D) of size `n × n`.
#[must_use]
pub fn f32_eye(n: usize) -> ArrayD<f32> {
    let mut out = f32_zeros(&[n, n]);
    for i in 0..n {
        out[[i, i].as_slice()] = 1.0;
    }
    out
}

/// Uniform `[0, 1)` random tensor from a seeded RNG — deterministic.
#[must_use]
pub fn f32_rand_seeded(shape: &[usize], seed: u64) -> ArrayD<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    ArrayD::from_shape_fn(IxDyn(shape), |_| rng.random::<f32>())
}

/// Standard-normal random tensor from a seeded RNG. Uses the
/// Box-Muller transform to stay in-crate (no extra `rand_distr` dep).
#[must_use]
pub fn f32_randn_seeded(shape: &[usize], seed: u64) -> ArrayD<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    ArrayD::from_shape_fn(IxDyn(shape), |_| box_muller_f32(&mut rng))
}

/// Linearly spaced values `[start, end)` step `step`, shape `(n,)` where
/// `n = floor((end − start) / step)`.
#[must_use]
pub fn f32_arange(start: f32, end: f32, step: f32) -> ArrayD<f32> {
    if step <= 0.0 || end <= start {
        return f32_zeros(&[0]);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = ((end - start) / step).floor() as usize;
    ArrayD::from_shape_fn(IxDyn(&[n]), |idx| {
        #[allow(clippy::cast_precision_loss)]
        let i = idx[0] as f32;
        start + i * step
    })
}

/// `n` evenly-spaced values from `start` to `end` inclusive, shape `(n,)`.
#[must_use]
pub fn f32_linspace(start: f32, end: f32, n: usize) -> ArrayD<f32> {
    if n == 0 {
        return f32_zeros(&[0]);
    }
    if n == 1 {
        return ArrayD::from_shape_fn(IxDyn(&[1]), |_| start);
    }
    #[allow(clippy::cast_precision_loss)]
    let step = (end - start) / ((n - 1) as f32);
    ArrayD::from_shape_fn(IxDyn(&[n]), |idx| {
        #[allow(clippy::cast_precision_loss)]
        let i = idx[0] as f32;
        start + i * step
    })
}

// =========================================================================
// F64 factories (subset — most callers use f32)
// =========================================================================

/// All-zeros f64 tensor.
#[must_use]
pub fn f64_zeros(shape: &[usize]) -> ArrayD<f64> {
    ArrayD::zeros(IxDyn(shape))
}

/// All-ones f64 tensor.
#[must_use]
pub fn f64_ones(shape: &[usize]) -> ArrayD<f64> {
    ArrayD::from_elem(IxDyn(shape), 1.0)
}

// =========================================================================
// I32 factories (subset)
// =========================================================================

/// All-zeros i32 tensor.
#[must_use]
pub fn i32_zeros(shape: &[usize]) -> ArrayD<i32> {
    ArrayD::zeros(IxDyn(shape))
}

/// Range `[0, n)` — V1's default integer `arange`.
#[must_use]
pub fn i32_arange(n: usize) -> ArrayD<i32> {
    ArrayD::from_shape_fn(IxDyn(&[n]), |idx| {
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let v = idx[0] as i32;
        v
    })
}

// =========================================================================
// Reshape / misc
// =========================================================================

/// Reshape preserving total element count. Returns
/// [`TensorError::ReshapeSize`] on mismatch.
pub fn reshape<A: Clone>(a: &ArrayD<A>, new_shape: &[usize]) -> Result<ArrayD<A>, TensorError> {
    let from: usize = a.len();
    let to: usize = new_shape.iter().product();
    if from != to {
        return Err(TensorError::ReshapeSize { from, to });
    }
    Array::from_shape_vec(IxDyn(new_shape), a.iter().cloned().collect())
        .map_err(|_| TensorError::ReshapeSize { from, to })
}

/// Number of elements total. Identical to `a.len()`; surfaced here
/// as a V1-style helper so callers porting from C don't need to know.
#[must_use]
pub fn numel<A>(a: &ArrayD<A>) -> usize {
    a.len()
}

/// Total bytes the tensor's data buffer occupies, not counting
/// ndarray overhead.
#[must_use]
pub fn nbytes<A>(a: &ArrayD<A>) -> usize {
    a.len() * core::mem::size_of::<A>()
}

// -------------------------------------------------------------------------
// Private — Box-Muller transform for standard-normal samples.
// -------------------------------------------------------------------------

fn box_muller_f32(rng: &mut ChaCha20Rng) -> f32 {
    // Generate two independent uniforms in (0, 1] and transform.
    let u1: f32 = rng.random::<f32>().max(1e-9);
    let u2: f32 = rng.random::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * core::f32::consts::PI * u2;
    r * theta.cos()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_right_shape() {
        let a = f32_zeros(&[3, 4]);
        assert_eq!(a.shape(), &[3, 4]);
        for &v in a.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn ones_has_right_shape_and_values() {
        let a = f32_ones(&[5]);
        assert_eq!(a.shape(), &[5]);
        for &v in a.iter() {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn full_sets_value() {
        let a = f32_full(&[2, 2], 3.14);
        for &v in a.iter() {
            assert_eq!(v, 3.14);
        }
    }

    #[test]
    fn eye_is_diagonal() {
        let a = f32_eye(4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(a[[i, j].as_slice()], expected);
            }
        }
    }

    #[test]
    fn rand_seeded_is_deterministic() {
        let a = f32_rand_seeded(&[4, 4], 42);
        let b = f32_rand_seeded(&[4, 4], 42);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn rand_seeded_in_unit_interval() {
        let a = f32_rand_seeded(&[100], 7);
        for &v in a.iter() {
            assert!((0.0..1.0).contains(&v), "uniform value {v} out of [0, 1)");
        }
    }

    #[test]
    fn randn_seeded_has_zero_mean_approx() {
        let a = f32_randn_seeded(&[10_000], 7);
        let mean: f32 = a.iter().sum::<f32>() / a.len() as f32;
        // Standard normal, 10K samples → mean should be close to zero.
        assert!(mean.abs() < 0.05, "mean {mean} too far from 0");
    }

    #[test]
    fn arange_length_matches() {
        let a = f32_arange(0.0, 1.0, 0.1);
        assert_eq!(a.shape(), &[10]);
        assert_eq!(a[[0].as_slice()], 0.0);
    }

    #[test]
    fn arange_empty_for_degenerate_input() {
        assert_eq!(f32_arange(1.0, 0.0, 0.1).len(), 0);
        assert_eq!(f32_arange(0.0, 1.0, -0.1).len(), 0);
    }

    #[test]
    fn linspace_endpoints_match() {
        let a = f32_linspace(0.0, 10.0, 11);
        assert_eq!(a[[0].as_slice()], 0.0);
        assert_eq!(a[[10].as_slice()], 10.0);
    }

    #[test]
    fn linspace_single_point() {
        let a = f32_linspace(5.0, 10.0, 1);
        assert_eq!(a.shape(), &[1]);
        assert_eq!(a[[0].as_slice()], 5.0);
    }

    #[test]
    fn reshape_preserves_elements() {
        let a = f32_arange(0.0, 12.0, 1.0);
        let r = reshape(&a, &[3, 4]).unwrap();
        assert_eq!(r.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                #[allow(clippy::cast_precision_loss)]
                let expected = (i * 4 + j) as f32;
                assert_eq!(r[[i, j].as_slice()], expected);
            }
        }
    }

    #[test]
    fn reshape_rejects_mismatch() {
        let a = f32_arange(0.0, 6.0, 1.0);
        assert!(matches!(
            reshape(&a, &[2, 2]),
            Err(TensorError::ReshapeSize { from: 6, to: 4 })
        ));
    }

    #[test]
    fn numel_nbytes() {
        let a = f32_zeros(&[3, 4]);
        assert_eq!(numel(&a), 12);
        assert_eq!(nbytes(&a), 12 * core::mem::size_of::<f32>());
    }

    #[test]
    fn f64_zeros_works() {
        let a = f64_zeros(&[2, 3]);
        assert_eq!(a.shape(), &[2, 3]);
    }

    #[test]
    fn i32_arange_range() {
        let a = i32_arange(5);
        let v: Vec<i32> = a.iter().copied().collect();
        assert_eq!(v, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn dtype_values() {
        assert_ne!(DType::F32, DType::F64);
        assert_ne!(DType::F32, DType::I32);
    }
}
