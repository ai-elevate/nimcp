//! Hilbert-curve locality addressing for memory nodes.
//!
//! Thin, validating wrapper around [`nimcp_fractal::hilbert`] used as
//! an **alternative address primitive** for memory-node IDs. Given a
//! 2D grid of side `n = 2^order`, the Hilbert curve produces a 1D
//! index that preserves locality: two IDs that are close on the curve
//! correspond to cells that are close in the 2D grid. This property
//! is useful for range queries over memory nodes whose IDs carry
//! spatial / topological meaning.
//!
//! These helpers are additive and pure — they do **not** mutate
//! [`crate::ZLadder`] or any other memory state. Callers decide when
//! to use Hilbert addresses vs. opaque IDs.
//!
//! # Out-of-range semantics
//!
//! To match the upstream `nimcp_fractal::hilbert` contract, inputs
//! that fall outside the `order`-sized grid return a sentinel:
//!
//! - [`hilbert_address`] returns `0` when `(x, y)` falls outside the
//!   grid.
//! - [`hilbert_coord`] returns `(0, 0)` when `addr` exceeds the grid
//!   capacity.
//! - [`hilbert_manhattan_distance`] clamps both addresses before
//!   comparing, so out-of-range inputs collapse to the origin pair
//!   and the result is `0`.

use nimcp_fractal::hilbert::{hilbert_index_to_xy, hilbert_xy_to_index};

/// Convert a 2D grid coordinate to a 1D Hilbert-locality address.
///
/// Thin re-export around [`nimcp_fractal::hilbert::hilbert_xy_to_index`]
/// that validates the input is inside an `order`-sized grid
/// (side = `2^order`). Returns `0` for out-of-range `(x, y)` — this
/// matches the upstream sentinel convention.
#[must_use]
pub fn hilbert_address(order: u32, x: u64, y: u64) -> u64 {
    hilbert_xy_to_index(order, x, y)
}

/// Inverse of [`hilbert_address`]: recover `(x, y)` from a 1D Hilbert
/// address on a grid of side `2^order`. Returns `(0, 0)` if `addr`
/// falls outside the grid.
#[must_use]
pub fn hilbert_coord(order: u32, addr: u64) -> (u64, u64) {
    hilbert_index_to_xy(order, addr)
}

/// Manhattan distance between the coordinates that two Hilbert
/// addresses decode to, on the same `order`-sized grid.
///
/// Useful as a cheap **proximity proxy** for range queries over
/// memory-node IDs that were assigned Hilbert addresses: the curve's
/// locality property means small Manhattan distances usually
/// correspond to small address differences, but the reverse is not
/// guaranteed, so callers that need an exact neighbourhood should
/// compute this explicitly rather than relying on address arithmetic.
#[must_use]
pub fn hilbert_manhattan_distance(order: u32, a: u64, b: u64) -> u64 {
    let (ax, ay) = hilbert_coord(order, a);
    let (bx, by) = hilbert_coord(order, b);
    ax.abs_diff(bx) + ay.abs_diff(by)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_coords_order_3() {
        // 8 × 8 = 64 cells; every cell must round-trip through the
        // address / coord pair.
        let order = 3;
        let n = 1u64 << order;
        for x in 0..n {
            for y in 0..n {
                let addr = hilbert_address(order, x, y);
                let (x2, y2) = hilbert_coord(order, addr);
                assert_eq!(
                    (x, y),
                    (x2, y2),
                    "round-trip failed at ({x},{y}) -> {addr} -> ({x2},{y2})"
                );
            }
        }
    }

    #[test]
    fn adjacent_addresses_have_unit_manhattan_distance() {
        // Defining property of the Hilbert curve: addresses `d` and
        // `d+1` map to grid cells that are orthogonal neighbours, so
        // Manhattan distance = 1.
        let order = 4;
        let n = 1u64 << order;
        for d in 0..(n * n - 1) {
            let dist = hilbert_manhattan_distance(order, d, d + 1);
            assert_eq!(dist, 1, "addresses {d} and {}+1 not adjacent", d);
        }
    }

    #[test]
    fn out_of_range_coord_returns_zero_address() {
        // Matches upstream sentinel: anything outside the grid → 0.
        let order = 2; // 4 × 4 grid
        assert_eq!(hilbert_address(order, 99, 99), 0);
        assert_eq!(hilbert_address(order, 4, 0), 0);
        assert_eq!(hilbert_address(order, 0, 4), 0);
    }

    #[test]
    fn out_of_range_address_returns_origin_coord() {
        // 16 cells exist on a 4 × 4 grid; addr=1000 is far past.
        let order = 2;
        assert_eq!(hilbert_coord(order, 1000), (0, 0));
    }

    #[test]
    fn self_distance_is_zero() {
        let order = 3;
        let addr = hilbert_address(order, 5, 2);
        assert_eq!(hilbert_manhattan_distance(order, addr, addr), 0);
    }

    #[test]
    fn symmetric_distance() {
        // Manhattan distance is symmetric by definition.
        let order = 3;
        let a = hilbert_address(order, 1, 1);
        let b = hilbert_address(order, 5, 6);
        assert_eq!(
            hilbert_manhattan_distance(order, a, b),
            hilbert_manhattan_distance(order, b, a),
        );
    }
}
