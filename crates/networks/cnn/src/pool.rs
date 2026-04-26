//! 2-D max-pool — non-overlapping spatial downsample.
//!
//! Stateless; pool size + stride are config. No padding (floor-division
//! shape rule). Operates on `[batch, channels, height, width]`.

use ndarray::Array4;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool2dLayer {
    pub kh: usize,
    pub kw: usize,
    pub stride_h: usize,
    pub stride_w: usize,
}

impl MaxPool2dLayer {
    pub fn new(kh: usize, kw: usize, stride_h: usize, stride_w: usize) -> Self {
        assert!(kh > 0 && kw > 0, "pool: zero kernel");
        assert!(stride_h > 0 && stride_w > 0, "pool: zero stride");
        Self {
            kh,
            kw,
            stride_h,
            stride_w,
        }
    }

    /// Square pool with matching stride — the most common case.
    pub fn square(k: usize) -> Self {
        Self::new(k, k, k, k)
    }

    /// Output spatial extent for a given input H/W. Uses floor-division
    /// (no padding), matching standard PyTorch defaults.
    pub fn output_hw(&self, h_in: usize, w_in: usize) -> (usize, usize) {
        let h_out = (h_in.saturating_sub(self.kh)) / self.stride_h + 1;
        let w_out = (w_in.saturating_sub(self.kw)) / self.stride_w + 1;
        (h_out, w_out)
    }

    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let (n, c, h_in, w_in) = input.dim();
        assert!(
            h_in >= self.kh && w_in >= self.kw,
            "pool: input smaller than kernel ({h_in}x{w_in} vs {}x{})",
            self.kh,
            self.kw
        );
        let (h_out, w_out) = self.output_hw(h_in, w_in);
        let mut out = Array4::<f32>::zeros((n, c, h_out, w_out));

        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let h0 = oh * self.stride_h;
                        let w0 = ow * self.stride_w;
                        let mut m = f32::NEG_INFINITY;
                        for di in 0..self.kh {
                            for dj in 0..self.kw {
                                let v = input[[ni, ci, h0 + di, w0 + dj]];
                                if v > m {
                                    m = v;
                                }
                            }
                        }
                        out[[ni, ci, oh, ow]] = m;
                    }
                }
            }
        }
        out
    }
}
