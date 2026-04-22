//! Config types + error codes.
//!
//! Defaults port V1's Phase E biologically-inspired parameters (see
//! `include/cognitive/memory/core/nimcp_z_ladder.h`). Tunable per
//! deployment; the V2 sanity tests pin the tight-loop values so we
//! notice unintended drift.

use serde::{Deserialize, Serialize};

use crate::node::Tier;

/// Error codes returned by [`ZLadder`](crate::ZLadder) ops. Rust
/// flavour of V1's `z_ladder_error_t` — one variant per distinct
/// failure mode.
#[derive(Debug, thiserror::Error)]
pub enum ZLadderError {
    /// The requested node isn't in any tier.
    #[error("node {0} not found")]
    NotFound(u64),
    /// A node with this ID is already present.
    #[error("node {0} already present")]
    AlreadyExists(u64),
    /// Target tier is full and eviction was disabled (`no_evict`).
    #[error("tier {0:?} at capacity")]
    Capacity(Tier),
    /// Landmark set is full.
    #[error("landmark capacity {0} reached")]
    LandmarkCapacity(usize),
    /// A tier index out of bounds or malformed config.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Per-tier configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TierConfig {
    /// Max nodes in this tier (`0` = unlimited — only meaningful for `Z3`).
    pub capacity: usize,
    /// Strength decay per second. `exp(-decay_rate · dt)` applied every
    /// `ZLadder::apply_decay`.
    pub decay_rate: f32,
    /// Minimum strength required to promote to the next tier.
    pub promotion_strength: f32,
    /// Below this strength the node demotes (or is evicted from `Z0`).
    pub demotion_threshold: f32,
    /// Minimum age (ms since insert) required before promotion.
    pub min_age_ms: u64,
    /// Minimum access count for promotion.
    pub min_access: u32,
    /// Minimum salience for promotion.
    pub min_salience: f32,
    /// Minimum consolidation score for promotion.
    pub min_consolidation: f32,
}

/// V1-equivalent defaults for each tier.
pub fn default_tier_config(tier: Tier) -> TierConfig {
    match tier {
        // Z0 — working memory. Fast decay, low promotion bar.
        Tier::Z0 => TierConfig {
            capacity: 9, // Miller's 7±2
            decay_rate: 0.07,
            promotion_strength: 0.5,
            demotion_threshold: 0.15,
            min_age_ms: 100, // shortened from V1's 5s so tests are fast
            min_access: 3,
            min_salience: 0.3,
            min_consolidation: 0.0,
        },
        // Z1 — short-term. ~2 hr half-life; requires consolidation > 0.5.
        Tier::Z1 => TierConfig {
            capacity: 100,
            decay_rate: 0.0001,
            promotion_strength: 0.5,
            demotion_threshold: 0.15,
            min_age_ms: 500, // shortened from 1 hour
            min_access: 5,
            min_salience: 0.0,
            min_consolidation: 0.5,
        },
        // Z2 — long-term. ~19 hr half-life; tighter promotion bar.
        Tier::Z2 => TierConfig {
            capacity: 10_000,
            decay_rate: 0.00001,
            promotion_strength: 0.8,
            demotion_threshold: 0.15,
            min_age_ms: 1_000, // shortened from 1 day
            min_access: 10,
            min_salience: 0.0,
            min_consolidation: 0.8,
        },
        // Z3 — permanent. No decay; never promotes; never auto-demotes.
        Tier::Z3 => TierConfig {
            capacity: 0,
            decay_rate: 0.0,
            promotion_strength: 1.0,
            demotion_threshold: 0.0,
            min_age_ms: u64::MAX,
            min_access: u32::MAX,
            min_salience: 1.0,
            min_consolidation: 1.0,
        },
    }
}

/// Top-level [`ZLadder`](crate::ZLadder) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ZLadderConfig {
    /// Per-tier configuration (indexed by `Tier as usize`).
    pub tiers: [TierConfig; 4],
    /// Maximum number of landmarks.
    pub max_landmarks: usize,
}

impl Default for ZLadderConfig {
    /// All-V1 defaults.
    fn default() -> Self {
        Self {
            tiers: [
                default_tier_config(Tier::Z0),
                default_tier_config(Tier::Z1),
                default_tier_config(Tier::Z2),
                default_tier_config(Tier::Z3),
            ],
            max_landmarks: 256,
        }
    }
}

impl ZLadderConfig {
    /// Per-tier accessor.
    #[must_use]
    pub fn tier(&self, tier: Tier) -> &TierConfig {
        &self.tiers[tier.index()]
    }

    /// Validate a config. Rejects configs that would deadlock the
    /// consolidation loop — e.g. `demotion_threshold >= promotion_strength`
    /// in any tier except `Z3` (which never promotes anyway).
    pub fn validate(&self) -> Result<(), ZLadderError> {
        for (i, t) in self.tiers.iter().enumerate() {
            if i < 3 && t.demotion_threshold >= t.promotion_strength {
                return Err(ZLadderError::InvalidConfig(format!(
                    "tier {i}: demotion_threshold ({}) >= promotion_strength ({})",
                    t.demotion_threshold, t.promotion_strength
                )));
            }
        }
        Ok(())
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn z3_default_never_decays_nor_promotes() {
        let t = default_tier_config(Tier::Z3);
        assert_eq!(t.decay_rate, 0.0);
        assert_eq!(t.promotion_strength, 1.0);
    }

    #[test]
    fn default_config_validates() {
        ZLadderConfig::default().validate().unwrap();
    }

    #[test]
    fn invalid_tier_config_rejected() {
        let mut cfg = ZLadderConfig::default();
        cfg.tiers[0].demotion_threshold = 0.9;
        cfg.tiers[0].promotion_strength = 0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn tier_accessor_by_enum_works() {
        let cfg = ZLadderConfig::default();
        assert_eq!(cfg.tier(Tier::Z0).capacity, 9);
        assert_eq!(cfg.tier(Tier::Z1).capacity, 100);
        assert_eq!(cfg.tier(Tier::Z2).capacity, 10_000);
        assert_eq!(cfg.tier(Tier::Z3).capacity, 0);
    }
}
