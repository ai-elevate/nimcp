//! 2D Hilbert-curve index ↔ `(x, y)` round trip.
//!
//! Given an `order` (side length `n = 2^order`), the Hilbert curve
//! visits every `(x, y)` in `[0, n) × [0, n)` exactly once. Adjacent
//! indices on the curve correspond to adjacent cells in the 2D grid,
//! which makes the index a **locality-preserving** 1D address —
//! useful for memory layout, range-based nearest-neighbour queries,
//! image tiling.
//!
//! Algorithms match V1's reference (Wikipedia / Hacker's Delight 3rd
//! ed.). Uses successive quadrant reflection / rotation; no
//! lookup tables.

/// Convert a 1D Hilbert index `d` into 2D coordinates `(x, y)` on a
/// `n × n` grid where `n = 2^order`.
///
/// Returns `(0, 0)` if `d >= n²` (out-of-range guard).
#[must_use]
pub fn hilbert_index_to_xy(order: u32, d: u64) -> (u64, u64) {
    let n: u64 = 1u64.checked_shl(order).unwrap_or(1);
    if d >= n.saturating_mul(n) {
        return (0, 0);
    }
    let mut x: u64 = 0;
    let mut y: u64 = 0;
    let mut t = d;
    let mut s: u64 = 1;
    while s < n {
        let rx = ((t / 2) & 1) as u8;
        let ry = ((t ^ (rx as u64)) & 1) as u8;
        // Rotate quadrant appropriately.
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            core::mem::swap(&mut x, &mut y);
        }
        x += s * rx as u64;
        y += s * ry as u64;
        t /= 4;
        s *= 2;
    }
    (x, y)
}

/// Inverse: `(x, y)` → Hilbert index `d` on `n × n` grid with
/// `n = 2^order`. Returns `0` if `(x, y)` falls outside the grid.
#[must_use]
pub fn hilbert_xy_to_index(order: u32, mut x: u64, mut y: u64) -> u64 {
    let n: u64 = 1u64.checked_shl(order).unwrap_or(1);
    if x >= n || y >= n {
        return 0;
    }
    let mut d: u64 = 0;
    let mut s: u64 = n / 2;
    // Reverse uses the full grid size `n - 1` in the rotation, unlike
    // the forward version which uses the current scale `s - 1`. This
    // asymmetry is correct — see Wikipedia "Hilbert curve", functions
    // d2xy (forward, `rot(s, ...)`) and xy2d (reverse, `rot(n, ...)`).
    while s > 0 {
        let rx: u8 = if (x & s) > 0 { 1 } else { 0 };
        let ry: u8 = if (y & s) > 0 { 1 } else { 0 };
        d += s * s * u64::from((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 {
                x = (n - 1).wrapping_sub(x);
                y = (n - 1).wrapping_sub(y);
            }
            core::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_order_3() {
        // 8 × 8 grid = 64 cells. Every index ↔ (x, y) must round-trip.
        let order = 3;
        let n = 1u64 << order;
        for d in 0..(n * n) {
            let (x, y) = hilbert_index_to_xy(order, d);
            let d2 = hilbert_xy_to_index(order, x, y);
            assert_eq!(d, d2, "round-trip failed at d={d}: ({x},{y}) -> {d2}");
        }
    }

    #[test]
    fn bijection_order_2() {
        // 4 × 4 = 16 cells, each index maps to a unique coordinate.
        let order = 2;
        let mut seen = std::collections::HashSet::new();
        for d in 0..16 {
            let (x, y) = hilbert_index_to_xy(order, d);
            assert!(seen.insert((x, y)), "duplicate coord at d={d}");
        }
    }

    #[test]
    fn locality_adjacent_indices_are_adjacent_cells() {
        // Consecutive indices should map to cells that differ by 1 in
        // exactly one coordinate (Manhattan distance = 1) — the
        // Hilbert curve's defining property.
        let order = 4; // 16 × 16
        let n = 1u64 << order;
        for d in 0..(n * n - 1) {
            let (x1, y1) = hilbert_index_to_xy(order, d);
            let (x2, y2) = hilbert_index_to_xy(order, d + 1);
            let dx = x1.abs_diff(x2);
            let dy = y1.abs_diff(y2);
            assert_eq!(dx + dy, 1, "indices {d} and {}+1 not adjacent", d);
        }
    }

    #[test]
    fn out_of_range_index_returns_origin() {
        assert_eq!(hilbert_index_to_xy(2, 1000), (0, 0));
    }

    #[test]
    fn out_of_range_coords_return_zero() {
        assert_eq!(hilbert_xy_to_index(2, 99, 99), 0);
    }

    #[test]
    fn order_zero_degenerate_grid() {
        // order=0 means n=1 — single cell. Hilbert index 0 maps to (0,0).
        assert_eq!(hilbert_index_to_xy(0, 0), (0, 0));
        assert_eq!(hilbert_xy_to_index(0, 0, 0), 0);
    }
}
