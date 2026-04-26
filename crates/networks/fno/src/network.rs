//! Top-level [`FnoNetwork`] — input projection + N FNO blocks + output
//! projection.
//!
//! ```text
//!   x_in     [N, in_channels, L]                            (real)
//!   x_proj = LinearMix(x_in)              → [N, hidden, L]
//!   for block in blocks:
//!       x_proj = FnoBlock(x_proj)          → [N, hidden, L]
//!   x_out  = LinearMix(x_proj)            → [N, out_channels, L]
//! ```
//!
//! All projections are `LinearMixLayer` (1×1 convolutions) so the FFT
//! length is fixed for the duration of a forward pass. Channel counts
//! can change at the projections; spatial length is invariant.

use ndarray::Array3;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::block::FnoBlock;
use crate::linear_mix::LinearMixLayer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnoConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    /// Hidden channel width, used by every block + projection layer.
    pub hidden_channels: usize,
    /// Number of stacked [`FnoBlock`]s.
    pub n_blocks: usize,
    /// Number of low-frequency modes retained per block.
    pub modes: usize,
    pub rng_seed: u64,
}

#[derive(Debug, Error)]
pub enum FnoError {
    #[error("fno: zero blocks")]
    EmptyBlocks,
    #[error("fno: zero modes")]
    ZeroModes,
    #[error("fno: zero channels")]
    ZeroChannels,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnoNetwork {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_channels: usize,
    pub modes: usize,
    pub input_proj: LinearMixLayer,
    pub blocks: Vec<FnoBlock>,
    pub output_proj: LinearMixLayer,
}

impl FnoNetwork {
    pub fn new(config: FnoConfig) -> Result<Self, FnoError> {
        if config.n_blocks == 0 {
            return Err(FnoError::EmptyBlocks);
        }
        if config.modes == 0 {
            return Err(FnoError::ZeroModes);
        }
        if config.in_channels == 0 || config.out_channels == 0 || config.hidden_channels == 0 {
            return Err(FnoError::ZeroChannels);
        }
        let mix_seed = |idx: u64| {
            config
                .rng_seed
                .wrapping_add(idx)
                .wrapping_mul(0xD123_4567_89AB_CDEF)
        };

        let input_proj = LinearMixLayer::new(config.in_channels, config.hidden_channels, mix_seed(0));
        let blocks: Vec<FnoBlock> = (0..config.n_blocks)
            .map(|i| FnoBlock::new(config.hidden_channels, config.modes, mix_seed((i + 1) as u64)))
            .collect();
        let output_proj = LinearMixLayer::new(
            config.hidden_channels,
            config.out_channels,
            mix_seed((config.n_blocks + 1) as u64),
        );

        Ok(Self {
            in_channels: config.in_channels,
            out_channels: config.out_channels,
            hidden_channels: config.hidden_channels,
            modes: config.modes,
            input_proj,
            blocks,
            output_proj,
        })
    }

    /// `input: [N, in_channels, L]` → `[N, out_channels, L]`.
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let mut h = self.input_proj.forward(input);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.output_proj.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn fno_forward_shape_and_finite() {
        let cfg = FnoConfig {
            in_channels: 2,
            out_channels: 3,
            hidden_channels: 8,
            n_blocks: 2,
            modes: 4,
            rng_seed: 0xF00,
        };
        let net = FnoNetwork::new(cfg).unwrap();
        let length = 16;
        let input = Array3::from_shape_fn((1, 2, length), |(_, c, l)| {
            ((c + 1) as f32 * l as f32 * 0.1).sin()
        });
        let out = net.forward(&input);
        assert_eq!(out.dim(), (1, 3, length));
        for v in out.iter() {
            assert!(v.is_finite(), "fno produced non-finite: {v}");
        }
    }

    #[test]
    fn fno_resolution_independent_shape() {
        // Same network, two input lengths — both produce L-length output.
        let cfg = FnoConfig {
            in_channels: 1,
            out_channels: 1,
            hidden_channels: 4,
            n_blocks: 1,
            modes: 3,
            rng_seed: 0x1234,
        };
        let net = FnoNetwork::new(cfg).unwrap();
        for length in [16, 32, 64] {
            let input = Array3::from_shape_fn((1, 1, length), |(_, _, l)| l as f32 * 0.05);
            let out = net.forward(&input);
            assert_eq!(out.dim(), (1, 1, length), "fno length not preserved at {length}");
        }
    }

    #[test]
    fn fno_serde_round_trip() {
        let cfg = FnoConfig {
            in_channels: 1,
            out_channels: 2,
            hidden_channels: 6,
            n_blocks: 2,
            modes: 4,
            rng_seed: 0xABCD,
        };
        let net = FnoNetwork::new(cfg).unwrap();
        let json = serde_json::to_string(&net).unwrap();
        let restored: FnoNetwork = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.in_channels, net.in_channels);
        assert_eq!(restored.blocks.len(), net.blocks.len());

        let length = 16;
        let input = Array3::from_shape_fn((1, 1, length), |(_, _, l)| l as f32 * 0.07);
        let a = net.forward(&input);
        let b = restored.forward(&input);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "forward drift after serde: {x} vs {y}");
        }
    }
}
