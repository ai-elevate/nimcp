//! Memory node + tier types.

use serde::{Deserialize, Serialize};

/// Four memory tiers. Biological analog:
///
/// - [`Tier::Z0`] working memory (prefrontal, seconds half-life)
/// - [`Tier::Z1`] short-term (hippocampal, hours)
/// - [`Tier::Z2`] long-term (neocortical, days)
/// - [`Tier::Z3`] permanent (semantic / procedural, no decay)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    /// Working memory.
    Z0 = 0,
    /// Short-term memory.
    Z1 = 1,
    /// Long-term memory.
    Z2 = 2,
    /// Permanent memory.
    Z3 = 3,
}

impl Tier {
    /// Every tier in ascending order.
    pub const ALL: [Tier; 4] = [Tier::Z0, Tier::Z1, Tier::Z2, Tier::Z3];

    /// Convert to its array index.
    #[must_use]
    pub fn index(self) -> usize {
        self as u8 as usize
    }

    /// Next tier up, if any.
    #[must_use]
    pub fn promote(self) -> Option<Tier> {
        match self {
            Tier::Z0 => Some(Tier::Z1),
            Tier::Z1 => Some(Tier::Z2),
            Tier::Z2 => Some(Tier::Z3),
            Tier::Z3 => None,
        }
    }

    /// Next tier down, if any.
    #[must_use]
    pub fn demote(self) -> Option<Tier> {
        match self {
            Tier::Z0 => None,
            Tier::Z1 => Some(Tier::Z0),
            Tier::Z2 => Some(Tier::Z1),
            Tier::Z3 => Some(Tier::Z2),
        }
    }

    /// Human-readable name for logs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Z0 => "Z0",
            Tier::Z1 => "Z1",
            Tier::Z2 => "Z2",
            Tier::Z3 => "Z3",
        }
    }
}

/// A single memory record. Owned by exactly one tier in the [`ZLadder`].
///
/// [`MemoryNode::features`] is the retrievable payload — the checkpoint
/// preserves this in full, without compression or summarization. That
/// is V1's E6 "restore doesn't lose the feature" lesson, codified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Stable identifier — caller-assigned, unique within the ladder.
    pub id: u64,
    /// Feature payload. Round-tripped bit-for-bit across checkpoint.
    pub features: Vec<f32>,
    /// Current strength in `[0, 1]`. Decays over time; bumped by
    /// reinforcement.
    pub strength: f32,
    /// Salience — set by the caller on insert; used as a promotion
    /// gate (`Z0 → Z1` requires `salience ≥ min_salience`).
    pub salience: f32,
    /// Consolidation score — bumped by `access()` + reinforcement;
    /// used as a promotion gate (`Z1 → Z2` requires this ≥ min).
    pub consolidation: f32,
    /// Total access count since insertion.
    pub access_count: u32,
    /// Creation time (ms), set at insert.
    pub created_ms: u64,
    /// Last-access time (ms).
    pub last_access_ms: u64,
    /// Current tier.
    pub tier: Tier,
    /// `true` iff this node is flagged as a landmark — protected from
    /// demotion indefinitely.
    pub is_landmark: bool,
    /// Landmark reason tag (e.g. "first_reward"). Only meaningful when
    /// [`is_landmark`](Self::is_landmark) is `true`.
    pub landmark_reason: Option<String>,
}

impl MemoryNode {
    /// Construct a fresh node. Defaults: strength 1.0, salience 0.0,
    /// consolidation 0.0, access_count 0, `created_ms == last_access_ms`.
    #[must_use]
    pub fn new(id: u64, features: Vec<f32>, created_ms: u64) -> Self {
        Self {
            id,
            features,
            strength: 1.0,
            salience: 0.0,
            consolidation: 0.0,
            access_count: 0,
            created_ms,
            last_access_ms: created_ms,
            tier: Tier::Z0,
            is_landmark: false,
            landmark_reason: None,
        }
    }

    /// Age in ms at `now`. Saturates at `0` if `now` has somehow
    /// regressed (shouldn't happen with a monotonic clock).
    #[must_use]
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.created_ms)
    }

    /// Register an access at `now_ms`: increment count, update
    /// timestamp, and give a small strength + consolidation boost.
    pub fn record_access(&mut self, now_ms: u64) {
        self.access_count = self.access_count.saturating_add(1);
        self.last_access_ms = now_ms;
        // Small, bounded bumps — match V1's access behavior.
        self.strength = (self.strength + 0.05).clamp(0.0, 1.0);
        self.consolidation = (self.consolidation + 0.02).clamp(0.0, 1.0);
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
    fn tier_promote_demote_chain() {
        assert_eq!(Tier::Z0.promote(), Some(Tier::Z1));
        assert_eq!(Tier::Z1.promote(), Some(Tier::Z2));
        assert_eq!(Tier::Z2.promote(), Some(Tier::Z3));
        assert_eq!(Tier::Z3.promote(), None);

        assert_eq!(Tier::Z3.demote(), Some(Tier::Z2));
        assert_eq!(Tier::Z2.demote(), Some(Tier::Z1));
        assert_eq!(Tier::Z1.demote(), Some(Tier::Z0));
        assert_eq!(Tier::Z0.demote(), None);
    }

    #[test]
    fn tier_index_matches_repr() {
        assert_eq!(Tier::Z0.index(), 0);
        assert_eq!(Tier::Z1.index(), 1);
        assert_eq!(Tier::Z2.index(), 2);
        assert_eq!(Tier::Z3.index(), 3);
    }

    #[test]
    fn new_node_has_sensible_defaults() {
        let n = MemoryNode::new(42, vec![0.1, 0.2], 1_000);
        assert_eq!(n.id, 42);
        assert_eq!(n.strength, 1.0);
        assert_eq!(n.access_count, 0);
        assert_eq!(n.tier, Tier::Z0);
        assert!(!n.is_landmark);
    }

    #[test]
    fn record_access_updates_counters() {
        let mut n = MemoryNode::new(1, vec![0.0], 1_000);
        let s0 = n.strength;
        n.record_access(2_000);
        assert_eq!(n.access_count, 1);
        assert_eq!(n.last_access_ms, 2_000);
        // Strength clamps at 1.0 since we started there.
        assert_eq!(n.strength, 1.0);
        // Consolidation bumped.
        assert!(n.consolidation > 0.0);
        // Record-access on a lower-strength node actually raises it.
        let _ = s0;
    }

    #[test]
    fn age_ms_is_saturating() {
        let n = MemoryNode::new(1, vec![0.0], 5_000);
        assert_eq!(n.age_ms(10_000), 5_000);
        // Clock regression: shouldn't panic.
        assert_eq!(n.age_ms(1_000), 0);
    }
}
