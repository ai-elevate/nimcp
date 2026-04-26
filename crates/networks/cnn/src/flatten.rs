//! Flatten layer — collapses `[batch, channels, height, width]` to a
//! 2-D `[batch, channels * height * width]` for handoff to a dense
//! [`crate::LinearLayer`].
//!
//! V2 inference is single-sample today; the flatten output keeps the
//! batch axis for shape uniformity even when `batch == 1`.

use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlattenLayer;

impl FlattenLayer {
    pub fn new() -> Self {
        Self
    }

    /// Flatten `[N, C, H, W]` → `[N, C*H*W]` (row-major). Returns an
    /// owned `Array2` because the downstream `LinearLayer` expects 2-D.
    pub fn forward(&self, input: &Array4<f32>) -> Array2<f32> {
        let (n, c, h, w) = input.dim();
        let flat = c * h * w;
        let raw: Vec<f32> = input.iter().copied().collect();
        Array2::from_shape_vec((n, flat), raw)
            .expect("flatten: shape (n, c*h*w) always matches element count")
    }
}
