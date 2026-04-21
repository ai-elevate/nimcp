//! The [`ZLadder`] struct + basic ops (Phase 5a) + consolidation and
//! landmarks (Phase 5b).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::{ZLadderConfig, ZLadderError};
use crate::node::{MemoryNode, Tier};

/// Statistics returned by [`ZLadder::stats`]. Cheap to compute — the
/// ladder keeps counters up to date on every op.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ZLadderStats {
    /// Nodes currently in each tier.
    pub tier_counts: [usize; 4],
    /// Total nodes across every tier.
    pub total_nodes: usize,
    /// Landmarks currently tracked.
    pub landmark_count: usize,
    /// Cumulative promotions per transition (`Z0→Z1`, `Z1→Z2`, `Z2→Z3`).
    pub promotions: [u64; 3],
    /// Cumulative demotions per transition (`Z1→Z0`, `Z2→Z1`, `Z3→Z2`).
    pub demotions: [u64; 3],
    /// Cumulative evictions per tier.
    pub evictions: [u64; 4],
    /// Number of full [`ZLadder::consolidate`] passes executed.
    pub consolidate_cycles: u64,
}

/// Four-tier Z-Ladder. One [`HashMap`] per tier — `id → MemoryNode`.
///
/// Every mutating op touches exactly one tier (or moves between exactly
/// two), so the ladder is `Send` and lives inside the memory actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZLadder {
    config: ZLadderConfig,
    tiers: [HashMap<u64, MemoryNode>; 4],
    /// Landmarks are tracked by id → reason so we can iterate and
    /// audit without scanning every tier. Capacity is bounded by
    /// `config.max_landmarks`.
    landmarks: HashMap<u64, String>,
    /// Monotonic simulated clock in ms. Caller drives it with
    /// [`ZLadder::advance_time`] (or passes explicit `now` to ops).
    clock_ms: u64,
    stats: ZLadderStats,
}

impl ZLadder {
    /// Build an empty ladder. Returns `InvalidConfig` if `config` fails
    /// [`ZLadderConfig::validate`].
    pub fn new(config: ZLadderConfig) -> Result<Self, ZLadderError> {
        config.validate()?;
        Ok(Self {
            config,
            tiers: [
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            ],
            landmarks: HashMap::new(),
            clock_ms: 0,
            stats: ZLadderStats::default(),
        })
    }

    /// Build with defaults. Convenience wrapper around `new(default())`.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ZLadderConfig::default()).expect("default config is valid")
    }

    /// Borrow the config used at construction.
    #[must_use]
    pub fn config(&self) -> &ZLadderConfig {
        &self.config
    }

    /// Current simulated clock, ms.
    #[must_use]
    pub fn clock_ms(&self) -> u64 {
        self.clock_ms
    }

    /// Advance the simulated clock. `delta_ms = 0` is a no-op.
    pub fn advance_time(&mut self, delta_ms: u64) {
        self.clock_ms = self.clock_ms.saturating_add(delta_ms);
    }

    /// Total nodes across all tiers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tiers.iter().map(HashMap::len).sum()
    }

    /// Nodes in one tier.
    #[must_use]
    pub fn len_tier(&self, tier: Tier) -> usize {
        self.tiers[tier.index()].len()
    }

    /// `true` iff the whole ladder is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tiers.iter().all(HashMap::is_empty)
    }

    // ---------------------------------------------------------------------
    // Insert / find / remove
    // ---------------------------------------------------------------------

    /// Insert `node` into its requested tier (defaults to whatever
    /// `node.tier` is set to). Evicts the weakest node if the tier is
    /// at capacity. Returns `AlreadyExists` if an entry with this ID
    /// is already in *any* tier.
    pub fn insert(&mut self, mut node: MemoryNode) -> Result<(), ZLadderError> {
        if self.find(node.id).is_some() {
            return Err(ZLadderError::AlreadyExists(node.id));
        }

        let tier = node.tier;
        let cap = self.config.tier(tier).capacity;
        // Non-Z3 tiers with `cap > 0` evict the weakest node on overflow.
        if cap > 0 && self.tiers[tier.index()].len() >= cap {
            self.evict_weakest_in(tier);
        }

        // Make sure the node reflects the tier we're inserting it into.
        node.tier = tier;
        self.tiers[tier.index()].insert(node.id, node);
        self.stats.tier_counts[tier.index()] = self.tiers[tier.index()].len();
        self.stats.total_nodes = self.len();
        Ok(())
    }

    /// Look up a node by ID across all tiers. Read-only — does *not*
    /// record an access (use [`ZLadder::access`] for that).
    #[must_use]
    pub fn find(&self, id: u64) -> Option<&MemoryNode> {
        for tier in &self.tiers {
            if let Some(n) = tier.get(&id) {
                return Some(n);
            }
        }
        None
    }

    /// Mutable lookup. Same rules as [`find`](Self::find).
    pub fn find_mut(&mut self, id: u64) -> Option<&mut MemoryNode> {
        for tier in &mut self.tiers {
            if let Some(n) = tier.get_mut(&id) {
                return Some(n);
            }
        }
        None
    }

    /// Locate which tier holds a node, if any.
    #[must_use]
    pub fn tier_of(&self, id: u64) -> Option<Tier> {
        Tier::ALL
            .into_iter()
            .find(|t| self.tiers[t.index()].contains_key(&id))
    }

    /// Remove a node from wherever it lives. Also drops it from the
    /// landmark set if present. Returns the removed node so callers can
    /// inspect it if useful.
    pub fn remove(&mut self, id: u64) -> Option<MemoryNode> {
        for t in Tier::ALL {
            if let Some(n) = self.tiers[t.index()].remove(&id) {
                self.stats.tier_counts[t.index()] = self.tiers[t.index()].len();
                self.stats.total_nodes = self.len();
                self.landmarks.remove(&id);
                self.stats.landmark_count = self.landmarks.len();
                return Some(n);
            }
        }
        None
    }

    /// Record an access on `id`. Returns `true` iff the node was
    /// present. Updates access_count, last-access timestamp, gives a
    /// small strength + consolidation bump.
    pub fn access(&mut self, id: u64) -> bool {
        let now = self.clock_ms;
        match self.find_mut(id) {
            Some(n) => {
                n.record_access(now);
                true
            }
            None => false,
        }
    }

    /// Bump a node's strength by `amount`, clamped to `[0, 1]`. Returns
    /// the new strength or `NotFound`.
    pub fn reinforce(&mut self, id: u64, amount: f32) -> Result<f32, ZLadderError> {
        match self.find_mut(id) {
            Some(n) => {
                n.strength = (n.strength + amount).clamp(0.0, 1.0);
                Ok(n.strength)
            }
            None => Err(ZLadderError::NotFound(id)),
        }
    }

    /// Set the salience of a node. Useful when the caller learns
    /// something new about a prior insertion that didn't know its own
    /// salience yet (e.g. a reward arrives after a perception).
    pub fn set_salience(&mut self, id: u64, salience: f32) -> Result<(), ZLadderError> {
        let n = self.find_mut(id).ok_or(ZLadderError::NotFound(id))?;
        n.salience = salience.clamp(0.0, 1.0);
        Ok(())
    }

    /// Explicit setter for the consolidation score.
    pub fn set_consolidation(&mut self, id: u64, consol: f32) -> Result<(), ZLadderError> {
        let n = self.find_mut(id).ok_or(ZLadderError::NotFound(id))?;
        n.consolidation = consol.clamp(0.0, 1.0);
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Decay
    // ---------------------------------------------------------------------

    /// Apply one tick of exponential decay to every tier. Uses each
    /// tier's configured `decay_rate` and the elapsed `dt_seconds`.
    pub fn apply_decay(&mut self, dt_seconds: f32) {
        if dt_seconds <= 0.0 || !dt_seconds.is_finite() {
            return;
        }
        for (t_idx, tier_map) in self.tiers.iter_mut().enumerate() {
            let rate = self.config.tiers[t_idx].decay_rate;
            if rate <= 0.0 {
                continue;
            }
            let factor = (-rate * dt_seconds).exp();
            for n in tier_map.values_mut() {
                n.strength *= factor;
            }
        }
    }

    // ---------------------------------------------------------------------
    // Capacity eviction
    // ---------------------------------------------------------------------

    /// Evict the weakest node in a tier (lowest strength wins). Skips
    /// landmarks — a landmark is protected even during capacity
    /// pressure. Returns the evicted node's ID or `None` if the tier is
    /// empty (or every node in it is a landmark).
    fn evict_weakest_in(&mut self, tier: Tier) -> Option<u64> {
        // Pick the weakest non-landmark node.
        let weakest_id = {
            let map = &self.tiers[tier.index()];
            map.iter()
                .filter(|(id, _)| !self.landmarks.contains_key(id))
                .min_by(|(_, a), (_, b)| {
                    a.strength
                        .partial_cmp(&b.strength)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _)| *id)
        };
        if let Some(id) = weakest_id {
            self.tiers[tier.index()].remove(&id);
            self.stats.tier_counts[tier.index()] = self.tiers[tier.index()].len();
            self.stats.total_nodes = self.len();
            self.stats.evictions[tier.index()] =
                self.stats.evictions[tier.index()].saturating_add(1);
            self.landmarks.remove(&id);
            self.stats.landmark_count = self.landmarks.len();
            return Some(id);
        }
        None
    }

    // ---------------------------------------------------------------------
    // Promote / demote
    // ---------------------------------------------------------------------

    /// `true` iff `node` meets the current tier's promotion criteria.
    #[must_use]
    pub fn should_promote(&self, node: &MemoryNode) -> bool {
        let cfg = self.config.tier(node.tier);
        if node.tier == Tier::Z3 {
            return false;
        }
        let age_ok = node.age_ms(self.clock_ms) >= cfg.min_age_ms;
        let strength_ok = node.strength >= cfg.promotion_strength;
        let access_ok = node.access_count >= cfg.min_access;
        let salience_ok = node.salience >= cfg.min_salience;
        let consol_ok = node.consolidation >= cfg.min_consolidation;
        age_ok && strength_ok && access_ok && salience_ok && consol_ok
    }

    /// `true` iff `node` should be demoted (or evicted, from `Z0`).
    /// Landmarks are **never** demoted — that's the elephant-matriarch
    /// guarantee.
    #[must_use]
    pub fn should_demote(&self, node: &MemoryNode) -> bool {
        if node.is_landmark {
            return false;
        }
        let cfg = self.config.tier(node.tier);
        node.strength < cfg.demotion_threshold
    }

    /// Force-promote a node by one tier. Returns the new tier or an
    /// error if already at `Z3` (or not found).
    pub fn promote(&mut self, id: u64) -> Result<Tier, ZLadderError> {
        let from = self.tier_of(id).ok_or(ZLadderError::NotFound(id))?;
        let to = from.promote().ok_or(ZLadderError::Capacity(from))?;
        self.move_node(id, from, to)?;

        // Stats: per-transition counter.
        let idx = from.index();
        if idx < 3 {
            self.stats.promotions[idx] = self.stats.promotions[idx].saturating_add(1);
        }
        Ok(to)
    }

    /// Force-demote a node. `Z0` demotion is an eviction. Landmarks
    /// cannot be demoted — returns `NotFound` (semantic: the landmark
    /// isn't "findable" as a demote target).
    pub fn demote(&mut self, id: u64) -> Result<Option<Tier>, ZLadderError> {
        let from = self.tier_of(id).ok_or(ZLadderError::NotFound(id))?;
        if self.tiers[from.index()]
            .get(&id)
            .is_some_and(|n| n.is_landmark)
        {
            return Err(ZLadderError::NotFound(id));
        }
        match from.demote() {
            Some(to) => {
                self.move_node(id, from, to)?;
                let idx = to.index(); // demotion counters are indexed by the destination
                // (Z1→Z0 lands at index 0, etc.)
                if idx < 3 {
                    self.stats.demotions[idx] = self.stats.demotions[idx].saturating_add(1);
                }
                Ok(Some(to))
            }
            None => {
                // Already at Z0 → evict.
                self.remove(id);
                self.stats.evictions[Tier::Z0.index()] =
                    self.stats.evictions[Tier::Z0.index()].saturating_add(1);
                Ok(None)
            }
        }
    }

    /// Move a node between tiers. Internal helper — does not run the
    /// promotion / demotion gates; callers gate first.
    fn move_node(&mut self, id: u64, from: Tier, to: Tier) -> Result<(), ZLadderError> {
        let mut node = self.tiers[from.index()]
            .remove(&id)
            .ok_or(ZLadderError::NotFound(id))?;
        node.tier = to;
        // Capacity on the destination — evict weakest if full.
        let cap = self.config.tier(to).capacity;
        if cap > 0 && self.tiers[to.index()].len() >= cap {
            self.evict_weakest_in(to);
        }
        self.tiers[to.index()].insert(id, node);
        // Refresh stats.
        self.stats.tier_counts[from.index()] = self.tiers[from.index()].len();
        self.stats.tier_counts[to.index()] = self.tiers[to.index()].len();
        self.stats.total_nodes = self.len();
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Consolidation — the "pump one step of the ladder" op
    // ---------------------------------------------------------------------

    /// Full consolidation pass:
    ///
    /// 1. Apply decay over `dt_seconds`.
    /// 2. Scan every tier (Z3 included), demote / evict anything whose
    ///    strength has fallen below its tier's threshold, skipping
    ///    landmarks.
    /// 3. Scan `Z0..=Z2`, promote anything that now meets its tier's
    ///    promotion gate.
    /// 4. Trim any tier still over capacity by evicting weakest
    ///    (same no-evict-landmarks rule).
    ///
    /// Returns `(promoted, demoted, evicted)` counts for this cycle.
    pub fn consolidate(&mut self, dt_seconds: f32) -> (usize, usize, usize) {
        self.apply_decay(dt_seconds);

        // Collect IDs to demote; mutate outside the borrow.
        let demote_ids: Vec<u64> = self
            .tiers
            .iter()
            .flat_map(|m| m.values())
            .filter(|n| self.should_demote(n))
            .map(|n| n.id)
            .collect();
        let mut demoted = 0;
        let mut evicted = 0;
        for id in demote_ids {
            match self.demote(id) {
                Ok(Some(_)) => demoted += 1,
                Ok(None) => evicted += 1, // dropped off Z0
                Err(_) => {}              // landmark — skipped
            }
        }

        // Promote in ascending order (Z0 first, then Z1, Z2). Do
        // multiple passes if we just promoted a node that now meets
        // the next tier's gate — V1 called this "sleep consolidation"
        // which is a nice-to-have; one pass is fine for Phase 5a.
        let mut promoted = 0;
        let promote_ids: Vec<u64> = self
            .tiers
            .iter()
            .take(3)
            .flat_map(|m| m.values())
            .filter(|n| self.should_promote(n))
            .map(|n| n.id)
            .collect();
        for id in promote_ids {
            if self.promote(id).is_ok() {
                promoted += 1;
            }
        }

        // Trim any tier over capacity (can happen after a cascade of
        // promotes land in Z1/Z2 simultaneously).
        for t in [Tier::Z0, Tier::Z1, Tier::Z2] {
            let cap = self.config.tier(t).capacity;
            while cap > 0 && self.tiers[t.index()].len() > cap {
                if self.evict_weakest_in(t).is_some() {
                    evicted += 1;
                } else {
                    break;
                }
            }
        }

        self.stats.consolidate_cycles = self.stats.consolidate_cycles.saturating_add(1);
        (promoted, demoted, evicted)
    }

    // ---------------------------------------------------------------------
    // Landmarks (Phase 5b)
    // ---------------------------------------------------------------------

    /// Mark a node as a landmark — elevate to `Z3` (if not already
    /// there) and protect from demotion. Idempotent.
    ///
    /// Returns `LandmarkCapacity` if the set is already at
    /// `config.max_landmarks`; `NotFound` if the node isn't anywhere.
    pub fn mark_landmark(&mut self, id: u64, reason: &str) -> Result<(), ZLadderError> {
        use std::collections::hash_map::Entry;
        // Already a landmark → just refresh the reason.
        if let Entry::Occupied(mut e) = self.landmarks.entry(id) {
            e.insert(reason.to_string());
            if let Some(n) = self.find_mut(id) {
                n.landmark_reason = Some(reason.to_string());
            }
            return Ok(());
        }
        if self.landmarks.len() >= self.config.max_landmarks {
            return Err(ZLadderError::LandmarkCapacity(self.config.max_landmarks));
        }
        let tier = self.tier_of(id).ok_or(ZLadderError::NotFound(id))?;
        // Promote straight to Z3 if not already there.
        if tier != Tier::Z3 {
            self.move_node(id, tier, Tier::Z3)?;
        }
        if let Some(n) = self.tiers[Tier::Z3.index()].get_mut(&id) {
            n.is_landmark = true;
            n.landmark_reason = Some(reason.to_string());
        }
        self.landmarks.insert(id, reason.to_string());
        self.stats.landmark_count = self.landmarks.len();
        Ok(())
    }

    /// Un-mark a landmark. Node stays at `Z3` but is no longer
    /// protected; normal demotion rules resume.
    pub fn unmark_landmark(&mut self, id: u64) -> Result<(), ZLadderError> {
        if self.landmarks.remove(&id).is_none() {
            return Err(ZLadderError::NotFound(id));
        }
        if let Some(n) = self.find_mut(id) {
            n.is_landmark = false;
            n.landmark_reason = None;
        }
        self.stats.landmark_count = self.landmarks.len();
        Ok(())
    }

    /// `true` iff the given ID is a currently-tracked landmark.
    #[must_use]
    pub fn is_landmark(&self, id: u64) -> bool {
        self.landmarks.contains_key(&id)
    }

    /// Iterate landmark (id, reason) pairs.
    pub fn landmarks(&self) -> impl Iterator<Item = (u64, &str)> + '_ {
        self.landmarks.iter().map(|(id, r)| (*id, r.as_str()))
    }

    /// Current landmark count.
    #[must_use]
    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }

    /// Prune landmarks whose node has been removed from the ladder
    /// (e.g. via `remove()` without unmarking). Returns the number of
    /// stale entries reclaimed. Normally a no-op — `remove()` already
    /// cleans up — but useful after bulk edits.
    pub fn prune_stale_landmarks(&mut self) -> usize {
        let stale: Vec<u64> = self
            .landmarks
            .keys()
            .filter(|id| self.tier_of(**id).is_none())
            .copied()
            .collect();
        let n = stale.len();
        for id in stale {
            self.landmarks.remove(&id);
        }
        self.stats.landmark_count = self.landmarks.len();
        n
    }

    /// Walk every node in every tier. Handy for debugging; the
    /// similarity queries in [`crate::query`] use a variant of this.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &MemoryNode> + '_ {
        self.tiers.iter().flat_map(HashMap::values)
    }

    // ---------------------------------------------------------------------
    // Stats
    // ---------------------------------------------------------------------

    /// Return the current stats snapshot.
    #[must_use]
    pub fn stats(&self) -> ZLadderStats {
        let mut s = self.stats;
        for (i, t) in self.tiers.iter().enumerate() {
            s.tier_counts[i] = t.len();
        }
        s.total_nodes = self.len();
        s.landmark_count = self.landmarks.len();
        s
    }
}

// -------------------------------------------------------------------------
// Tests (tiers + basic ops + decay)
// -------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn fresh() -> ZLadder {
        ZLadder::with_defaults()
    }

    fn node(id: u64, features: Vec<f32>, now: u64) -> MemoryNode {
        MemoryNode::new(id, features, now)
    }

    #[test]
    fn new_ladder_is_empty() {
        let l = fresh();
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn insert_find_remove_round_trip() {
        let mut l = fresh();
        l.insert(node(1, vec![0.1, 0.2], 0)).unwrap();
        assert_eq!(l.len(), 1);
        let n = l.find(1).unwrap();
        assert_eq!(n.id, 1);
        assert_eq!(n.features, vec![0.1, 0.2]);
        let removed = l.remove(1).unwrap();
        assert_eq!(removed.id, 1);
        assert!(l.is_empty());
    }

    #[test]
    fn duplicate_insert_rejected() {
        let mut l = fresh();
        l.insert(node(1, vec![], 0)).unwrap();
        let err = l.insert(node(1, vec![], 0)).unwrap_err();
        assert!(matches!(err, ZLadderError::AlreadyExists(1)));
    }

    #[test]
    fn access_updates_counters() {
        let mut l = fresh();
        l.insert(node(1, vec![0.0], 0)).unwrap();
        l.advance_time(100);
        assert!(l.access(1));
        let n = l.find(1).unwrap();
        assert_eq!(n.access_count, 1);
        assert_eq!(n.last_access_ms, 100);
    }

    #[test]
    fn reinforce_bumps_strength_clamped() {
        let mut l = fresh();
        l.insert(node(1, vec![0.0], 0)).unwrap();
        let s = l.reinforce(1, 0.25).unwrap();
        // New node starts at strength 1.0, so clamp to 1.0.
        assert_eq!(s, 1.0);

        // Lower strength manually, reinforce, observe partial bump.
        l.find_mut(1).unwrap().strength = 0.2;
        let s2 = l.reinforce(1, 0.3).unwrap();
        assert!((s2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tier_of_returns_correct_tier() {
        let mut l = fresh();
        l.insert(node(1, vec![], 0)).unwrap();
        assert_eq!(l.tier_of(1), Some(Tier::Z0));
        assert_eq!(l.tier_of(999), None);
    }

    #[test]
    fn z0_capacity_evicts_weakest() {
        let mut l = fresh();
        let cap = l.config().tier(Tier::Z0).capacity;
        // Fill Z0 to capacity with ascending strength.
        for i in 0..cap {
            let mut n = node(i as u64, vec![i as f32], 0);
            n.strength = 0.2 + (i as f32) * 0.05;
            l.insert(n).unwrap();
        }
        assert_eq!(l.len_tier(Tier::Z0), cap);

        // Insert one more → the weakest (id 0, strength 0.2) evicts.
        let mut new_node = node(100, vec![100.0], 0);
        new_node.strength = 1.0;
        l.insert(new_node).unwrap();
        assert_eq!(l.len_tier(Tier::Z0), cap);
        assert!(l.find(0).is_none(), "weakest node 0 was not evicted");
        assert!(l.find(100).is_some(), "new node was not inserted");
    }

    #[test]
    fn apply_decay_reduces_strength() {
        let mut l = fresh();
        let mut n = node(1, vec![], 0);
        n.strength = 1.0;
        l.insert(n).unwrap();
        // Z0 decay_rate = 0.07; after 10 seconds, factor = exp(-0.7) ≈ 0.497.
        l.apply_decay(10.0);
        let s = l.find(1).unwrap().strength;
        assert!((s - 0.4966).abs() < 1e-3, "unexpected decay: {s}");
    }

    #[test]
    fn decay_does_not_touch_z3() {
        let mut l = fresh();
        let mut n = node(1, vec![], 0);
        n.tier = Tier::Z3;
        n.strength = 0.5;
        l.insert(n).unwrap();
        l.apply_decay(100.0);
        assert_eq!(l.find(1).unwrap().strength, 0.5);
    }

    #[test]
    fn promote_moves_between_tiers_and_bumps_stats() {
        let mut l = fresh();
        l.insert(node(1, vec![], 0)).unwrap();
        let to = l.promote(1).unwrap();
        assert_eq!(to, Tier::Z1);
        assert_eq!(l.tier_of(1), Some(Tier::Z1));
        assert_eq!(l.stats().promotions[0], 1);
    }

    #[test]
    fn demote_from_z0_evicts() {
        let mut l = fresh();
        l.insert(node(1, vec![], 0)).unwrap();
        let result = l.demote(1).unwrap();
        assert_eq!(result, None, "Z0 demotion should evict");
        assert!(l.find(1).is_none());
    }

    #[test]
    fn promote_from_z3_errors() {
        let mut l = fresh();
        let mut n = node(1, vec![], 0);
        n.tier = Tier::Z3;
        l.insert(n).unwrap();
        let err = l.promote(1).unwrap_err();
        assert!(matches!(err, ZLadderError::Capacity(Tier::Z3)));
    }

    #[test]
    fn should_promote_respects_all_gates() {
        let mut l = fresh();
        let mut n = node(1, vec![], 0);
        // Fails on age (default min 100 ms).
        n.strength = 1.0;
        n.access_count = 5;
        n.salience = 0.5;
        assert!(!l.should_promote(&n));
        // Advance clock and retry — now passes.
        l.clock_ms = 1_000;
        assert!(l.should_promote(&n));
        // But drop strength below threshold → fail again.
        n.strength = 0.1;
        assert!(!l.should_promote(&n));
    }

    #[test]
    fn landmark_protected_from_demotion() {
        let mut l = fresh();
        let mut n = node(1, vec![0.1; 4], 0);
        n.strength = 0.01; // below Z0 demotion threshold
        l.insert(n).unwrap();
        l.mark_landmark(1, "test").unwrap();
        // Landmark now lives at Z3.
        assert_eq!(l.tier_of(1), Some(Tier::Z3));
        // should_demote returns false for landmarks even at very low strength.
        let n = l.find(1).unwrap();
        assert!(!l.should_demote(n));
    }

    #[test]
    fn mark_landmark_rejects_missing_id() {
        let mut l = fresh();
        let err = l.mark_landmark(999, "nope").unwrap_err();
        assert!(matches!(err, ZLadderError::NotFound(999)));
    }

    #[test]
    fn landmark_capacity_enforced() {
        let cfg = ZLadderConfig {
            max_landmarks: 2,
            ..ZLadderConfig::default()
        };
        let mut l = ZLadder::new(cfg).unwrap();
        for i in 0..3 {
            l.insert(node(i, vec![], 0)).unwrap();
        }
        l.mark_landmark(0, "a").unwrap();
        l.mark_landmark(1, "b").unwrap();
        let err = l.mark_landmark(2, "c").unwrap_err();
        assert!(matches!(err, ZLadderError::LandmarkCapacity(2)));
    }

    #[test]
    fn unmark_landmark_clears_flags() {
        // Landmarks live at Z3; Z3 has demotion_threshold 0.0 by design,
        // so `should_demote` can't meaningfully flip back to true after
        // unmark on this tier. The contract unmark actually owes the
        // caller is: (1) remove from landmark set, (2) clear the node's
        // `is_landmark` flag + reason. Verify those explicitly.
        let mut l = fresh();
        let mut n = node(1, vec![], 0);
        n.strength = 0.01;
        l.insert(n).unwrap();
        l.mark_landmark(1, "r").unwrap();
        assert!(l.is_landmark(1));
        assert!(l.find(1).unwrap().is_landmark);
        assert_eq!(l.find(1).unwrap().landmark_reason.as_deref(), Some("r"),);

        l.unmark_landmark(1).unwrap();
        assert!(!l.is_landmark(1));
        assert!(!l.find(1).unwrap().is_landmark);
        assert!(l.find(1).unwrap().landmark_reason.is_none());
    }

    #[test]
    fn consolidate_promotes_eligible_nodes() {
        let mut l = fresh();
        let mut n = node(1, vec![0.1; 3], 0);
        n.strength = 0.9;
        n.access_count = 10;
        n.salience = 0.9;
        n.consolidation = 0.9;
        l.insert(n).unwrap();
        l.advance_time(2_000);
        let (prom, _, _) = l.consolidate(0.1);
        // Should have promoted from Z0 all the way through the chain
        // given very high scores. At minimum Z0→Z1.
        assert!(prom >= 1, "expected at least one promotion");
        assert_ne!(l.tier_of(1), Some(Tier::Z0));
    }

    #[test]
    fn consolidate_demotes_weak_nodes() {
        let mut l = fresh();
        let mut n = node(1, vec![], 0);
        n.strength = 0.05; // below Z0 threshold
        l.insert(n).unwrap();
        let (_, _, evicted) = l.consolidate(0.0);
        // Z0 demotion is an eviction.
        assert_eq!(evicted, 1);
        assert!(l.find(1).is_none());
    }

    #[test]
    fn prune_stale_landmarks_noop_when_clean() {
        let mut l = fresh();
        l.insert(node(1, vec![], 0)).unwrap();
        l.mark_landmark(1, "r").unwrap();
        assert_eq!(l.prune_stale_landmarks(), 0);
    }
}
