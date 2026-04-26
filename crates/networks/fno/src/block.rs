//! FNO block — `tanh(SpectralConv1d(x) + LinearMix(x))`.
//!
//! Both branches share the same input/output channel count so they can
//! be summed element-wise; this is the standard FNO building block.

use ndarray::Array3;
use serde::{Deserialize, Serialize};

use crate::linear_mix::LinearMixLayer;
use crate::spectral::SpectralConv1dLayer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnoBlock {
    pub spectral: SpectralConv1dLayer,
    pub linear_mix: LinearMixLayer,
}

impl FnoBlock {
    pub fn new(channels: usize, modes: usize, seed: u64) -> Self {
        let s_seed = seed;
        let l_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        Self {
            spectral: SpectralConv1dLayer::new(channels, channels, modes, s_seed),
            linear_mix: LinearMixLayer::new(channels, channels, l_seed),
        }
    }

    /// `input: [N, channels, L]` → `[N, channels, L]`.
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let mut out = self.spectral.forward(input);
        let mix = self.linear_mix.forward(input);
        for (o, m) in out.iter_mut().zip(mix.iter()) {
            *o = (*o + *m).tanh();
        }
        out
    }
}
