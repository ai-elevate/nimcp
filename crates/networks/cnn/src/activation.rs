//! ReLU activation layer — element-wise `max(0, x)`.
//!
//! Stateless. Operates on any 4-D `[batch, channels, height, width]`
//! tensor in place via `mapv_inplace`.

use ndarray::Array4;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReluLayer;

impl ReluLayer {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }
}
