//! Box-counting fractal dimension.
//!
//! For a set of 2D points in `[0, 1]²`, the box-counting dimension
//! estimates the exponent `D` such that the number of occupied boxes
//! at side length `ε` scales as `N(ε) ~ ε^(-D)`. Equivalently, plot
//! `log(N)` vs `log(1/ε)` and `D` is the slope.
//!
//! V1's `src/core/topology/nimcp_fractal_topology.c` uses this to
//! measure connectivity fractality of graph node layouts. V2 re-uses
//! the same algorithm — pure data in / scalar out, no state.
//!
//! # Input contract
//!
//! - `points` is `&[(f32, f32)]`, assumed to be in `[0, 1] × [0, 1]`.
//!   Points outside that box are clipped to the unit square for
//!   counting purposes (they still occupy the boundary cells).
//! - `min_order` / `max_order` define the `2^k` grid side lengths to
//!   sample. V1 defaults: `min=2` (4×4 grid), `max=8` (256×256).

use std::collections::HashSet;

/// Box-counting dimension `D ≈ slope of log(N(ε)) vs log(1/ε)`.
///
/// Returns `0.0` for empty input or `min_order >= max_order` (not
/// enough data points to fit).
#[must_use]
pub fn box_counting_dimension(points: &[(f32, f32)], min_order: u32, max_order: u32) -> f32 {
    if points.is_empty() || min_order >= max_order {
        return 0.0;
    }

    // Collect (log_inv_eps, log_n) pairs.
    let mut xs: Vec<f32> = Vec::with_capacity((max_order - min_order) as usize);
    let mut ys: Vec<f32> = Vec::with_capacity((max_order - min_order) as usize);

    for k in min_order..=max_order {
        let side: u64 = 1u64 << k;
        #[allow(clippy::cast_precision_loss)]
        let side_f = side as f32;
        let mut occupied: HashSet<(u64, u64)> = HashSet::new();
        for &(x, y) in points {
            // Clip into unit square so edge points still register.
            let xc = x.clamp(0.0, 1.0 - f32::EPSILON);
            let yc = y.clamp(0.0, 1.0 - f32::EPSILON);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bx = (xc * side_f) as u64;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let by = (yc * side_f) as u64;
            occupied.insert((bx, by));
        }
        let n = occupied.len();
        if n == 0 {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let log_inv_eps = (side as f32).ln();
        #[allow(clippy::cast_precision_loss)]
        let log_n = (n as f32).ln();
        xs.push(log_inv_eps);
        ys.push(log_n);
    }

    if xs.len() < 2 {
        return 0.0;
    }
    // Least-squares slope: (N·Σxy − Σx·Σy) / (N·Σx² − (Σx)²).
    #[allow(clippy::cast_precision_loss)]
    let n_f = xs.len() as f32;
    let sx: f32 = xs.iter().sum();
    let sy: f32 = ys.iter().sum();
    let sxy: f32 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    let sxx: f32 = xs.iter().map(|&x| x * x).sum();
    let denom = n_f * sxx - sx * sx;
    if denom.abs() < 1e-9 {
        return 0.0;
    }
    (n_f * sxy - sx * sy) / denom
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_zero() {
        assert_eq!(box_counting_dimension(&[], 2, 6), 0.0);
    }

    #[test]
    fn invalid_range_returns_zero() {
        let pts = vec![(0.5, 0.5)];
        assert_eq!(box_counting_dimension(&pts, 5, 3), 0.0);
    }

    #[test]
    fn single_point_is_dimensionless() {
        // A single point → N=1 at every scale → slope = 0.
        let d = box_counting_dimension(&[(0.5, 0.5)], 2, 6);
        assert!(d.abs() < 0.1, "single point should give D≈0, got {d}");
    }

    #[test]
    fn filled_square_has_dim_near_two() {
        // A dense 2D grid of points should have D ≈ 2 (fills the plane).
        let mut pts: Vec<(f32, f32)> = Vec::new();
        for i in 0..50 {
            for j in 0..50 {
                pts.push((i as f32 / 50.0, j as f32 / 50.0));
            }
        }
        let d = box_counting_dimension(&pts, 2, 5);
        assert!(
            (d - 2.0).abs() < 0.3,
            "filled square should give D≈2.0, got {d}"
        );
    }

    #[test]
    fn diagonal_line_has_dim_near_one() {
        // Points along y=x should have D ≈ 1 (1D set embedded in 2D).
        let pts: Vec<(f32, f32)> = (0..100)
            .map(|i| {
                let t = i as f32 / 100.0;
                (t, t)
            })
            .collect();
        let d = box_counting_dimension(&pts, 2, 5);
        assert!(
            (d - 1.0).abs() < 0.3,
            "diagonal line should give D≈1.0, got {d}"
        );
    }
}
