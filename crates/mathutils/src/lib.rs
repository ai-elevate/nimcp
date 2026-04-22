//! NIMCP V2 — numerically stable math helpers.
//!
//! Port of V1's `utils/math/nimcp_math_helpers.h`. Every function here
//! is pure, allocation-free (except `softmax` which returns a `Vec`),
//! and guards against the common float-precision pitfalls: `log(0)`,
//! `exp(large_x)` overflow, `sigmoid` saturation, softmax underflow.
//!
//! # Why a dedicated crate
//!
//! These helpers are consumed by multiple V2 crates (adaptive, SNN,
//! LNN, intrinsic reward, …). Centralising here avoids each crate
//! rolling its own `safe_log` / `sigmoid` and getting the numerics
//! slightly wrong.
//!
//! # Invariants every helper upholds
//!
//! - Finite input → finite output (no NaN, no ±∞).
//! - Monotone where the mathematical function is monotone.
//! - Identity on known fixed points (e.g. `sigmoid(0) = 0.5`).
//! - Zero-allocation on the hot path unless explicitly documented.

#![forbid(unsafe_code)]

/// `log(x)` epsilon floor — `log(x)` is computed as `log(max(x, EPS))`.
/// Matches V1's `1e-30` floor.
pub const LOG_EPS: f32 = 1.0e-30;

/// `exp(x)` argument cap — `exp(x)` is computed as `exp(min(x, MAX_EXP))`.
/// At `f32`, `exp(88.7)` overflows; we cap at `80` for safety.
pub const MAX_EXP: f32 = 80.0;

/// Clamp `x` to `[min_val, max_val]`. Identity when `x` is already in range.
/// Panics in debug builds if `min_val > max_val` (caller bug).
#[must_use]
#[inline]
pub fn clamp(x: f32, min_val: f32, max_val: f32) -> f32 {
    debug_assert!(min_val <= max_val, "clamp: min > max");
    x.clamp(min_val, max_val)
}

/// Clamp to `[0, 1]`.
#[must_use]
#[inline]
pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// `log(max(x, LOG_EPS))` — prevents `-inf` on zero / negative inputs.
#[must_use]
#[inline]
pub fn safe_log(x: f32) -> f32 {
    x.max(LOG_EPS).ln()
}

/// `exp(min(x, MAX_EXP))` — prevents `+inf` for large arguments.
#[must_use]
#[inline]
pub fn safe_exp(x: f32) -> f32 {
    x.min(MAX_EXP).exp()
}

/// Numerically stable logistic sigmoid: `1 / (1 + exp(-x))`.
/// For `x >= 0`: directly; for `x < 0`: uses the algebraic equivalent
/// `exp(x) / (1 + exp(x))` to avoid overflow of `exp(-x)`.
#[must_use]
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Numerically stable `log(sum(exp(xs)))`: subtract the max first.
/// Returns `f32::NEG_INFINITY` on empty input — matches the
/// mathematical convention that `log(sum_{∅}) = log(0)`.
#[must_use]
pub fn log_sum_exp(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return f32::NEG_INFINITY;
    }
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return m; // all -inf → stays -inf; any +inf wins.
    }
    let mut sum = 0.0_f32;
    for &x in xs {
        sum += (x - m).exp();
    }
    m + sum.ln()
}

/// Softmax in-place. `xs` is replaced by `exp(xs − max) / sum(exp(xs − max))`.
/// Zero-allocation; single pass for max, one for exp+sum, one for divide.
/// No-op on empty input.
pub fn softmax_inplace(xs: &mut [f32]) {
    if xs.is_empty() {
        return;
    }
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in xs.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in xs.iter_mut() {
            *x *= inv;
        }
    }
}

/// Softmax returning a fresh `Vec` — use the in-place variant on hot paths.
#[must_use]
pub fn softmax(xs: &[f32]) -> Vec<f32> {
    let mut out = xs.to_vec();
    softmax_inplace(&mut out);
    out
}

/// Shannon entropy of a probability distribution, in **nats** (natural log).
/// Clamps `p_i` to `[LOG_EPS, 1]` to avoid `-∞` terms. Caller is
/// responsible for ensuring the input approximates a distribution
/// (sums to ~1, all non-negative).
///
/// For a uniform distribution over `n` bins returns `ln(n)`.
#[must_use]
pub fn entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0_f32;
    for &p in probs {
        if p > 0.0 {
            let p = p.max(LOG_EPS);
            h -= p * p.ln();
        }
    }
    h
}

/// L2 norm of a vector.
#[must_use]
pub fn l2_norm(xs: &[f32]) -> f32 {
    xs.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity of two equal-length vectors. Returns `0.0` on
/// length mismatch or either vector zero (caller convention — treat
/// degenerate cases as "no match" rather than propagating an error).
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// KL divergence `D(p || q)` in nats. Both `p` and `q` must be
/// probability-like. Any `p_i = 0` contributes zero; any `q_i = 0`
/// with `p_i > 0` is clamped to `LOG_EPS` (soft floor).
///
/// Returns `0.0` on length mismatch (soft-fail; use `l2_norm(p - q)`
/// or inspect lengths beforehand if you need an error).
#[must_use]
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    if p.len() != q.len() || p.is_empty() {
        return 0.0;
    }
    let mut kl = 0.0_f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            let qi_safe = qi.max(LOG_EPS);
            kl += pi * (pi.ln() - qi_safe.ln());
        }
    }
    kl
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn clamp_identity_in_range() {
        assert_eq!(clamp(0.5, 0.0, 1.0), 0.5);
    }

    #[test]
    fn clamp_saturates() {
        assert_eq!(clamp(2.0, 0.0, 1.0), 1.0);
        assert_eq!(clamp(-1.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn clamp01_shortcut() {
        assert_eq!(clamp01(-0.5), 0.0);
        assert_eq!(clamp01(1.5), 1.0);
        assert_eq!(clamp01(0.5), 0.5);
    }

    #[test]
    fn safe_log_floors_at_eps() {
        assert!(safe_log(0.0).is_finite());
        assert!(safe_log(-1.0).is_finite());
        // log(1) = 0 — normal path still works.
        assert!((safe_log(1.0)).abs() < 1e-6);
    }

    #[test]
    fn safe_exp_caps_large() {
        assert!(safe_exp(100.0).is_finite());
        assert_eq!(safe_exp(100.0), MAX_EXP.exp());
    }

    #[test]
    fn sigmoid_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_saturation_finite() {
        assert!(sigmoid(-1000.0).is_finite());
        assert!(sigmoid(1000.0).is_finite());
        assert!(sigmoid(-1000.0) < 1e-6);
        assert!((sigmoid(1000.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_symmetry_around_zero() {
        let p = sigmoid(2.0);
        let n = sigmoid(-2.0);
        assert!((p + n - 1.0).abs() < 1e-6);
    }

    #[test]
    fn log_sum_exp_stable_for_large_values() {
        // log(exp(1000) + exp(1001)) ≈ 1001 + log(1 + 1/e)
        let x = log_sum_exp(&[1000.0, 1001.0]);
        let expected = 1001.0 + (1.0_f32 + (-1.0_f32).exp()).ln();
        assert!((x - expected).abs() < 1e-3);
    }

    #[test]
    fn log_sum_exp_empty_is_neg_inf() {
        assert_eq!(log_sum_exp(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_stable_for_large_values() {
        // Without max subtraction this overflows.
        let s = softmax(&[1000.0, 1000.0]);
        for &v in &s {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_inplace_matches_allocating() {
        let input = vec![0.3_f32, -0.5, 1.2, 0.1];
        let mut inplace = input.clone();
        softmax_inplace(&mut inplace);
        let alloc = softmax(&input);
        for (a, b) in inplace.iter().zip(alloc.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn entropy_uniform_is_log_n() {
        let n = 4;
        let probs = vec![0.25_f32; n];
        #[allow(clippy::cast_precision_loss)]
        let expected = (n as f32).ln();
        assert!((entropy(&probs) - expected).abs() < 1e-5);
    }

    #[test]
    fn entropy_point_mass_is_zero() {
        let probs = [1.0_f32, 0.0, 0.0];
        assert!(entropy(&probs).abs() < 1e-5);
    }

    #[test]
    fn l2_norm_unit() {
        assert!((l2_norm(&[0.6, 0.8]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_parallel_is_one() {
        assert!((cosine_similarity(&[3.0, 4.0], &[6.0, 8.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_length_mismatch_returns_zero() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn kl_divergence_self_is_zero() {
        let p = [0.3, 0.7];
        assert!(kl_divergence(&p, &p).abs() < 1e-6);
    }

    #[test]
    fn kl_divergence_finite_with_zero_q() {
        // p_i > 0 with q_i = 0 should give a finite (large) answer.
        assert!(kl_divergence(&[0.5, 0.5], &[1.0, 0.0]).is_finite());
    }

    #[test]
    fn kl_divergence_length_mismatch_returns_zero() {
        assert_eq!(kl_divergence(&[0.5, 0.5], &[1.0]), 0.0);
    }
}
