//! Per-source thalamic channel.
//!
//! A channel is the adapter-facing handle: it knows which destinations
//! the source talks to, what the current attention gates look like,
//! and which relay mode the source is operating in. Adapters construct
//! one per network instance and hold it across steps.

use serde::{Deserialize, Serialize};

/// Maximum destinations per source. Matches V1's `THALAMIC_MAX_DESTINATIONS`.
/// Fixed cap keeps [`ThalamicChannel`] small enough to be `Copy` for
/// cheap hand-off between actor messages.
pub const THALAMIC_MAX_DESTINATIONS: usize = 16;

/// How the router relays signals from this source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelayMode {
    /// High-amplitude, priority-skip-queue — used for salient signals
    /// that need to reach the destination without buffering delay.
    Burst,
    /// Normal relay through the priority queue. (V1 default.)
    #[default]
    Tonic,
    /// Router decides per-submit based on payload amplitude.
    Adaptive,
}

/// One submit event — used by adapters to hand signals to the router
/// without coupling to the router's internal scheduling.
#[derive(Debug, Clone)]
pub struct ThalamicSubmit {
    /// Source ID (network or sub-region).
    pub source_id: u32,
    /// Destination ID.
    pub dest_id: u32,
    /// Payload — any shape the downstream consumer understands.
    pub payload: Vec<f32>,
    /// Priority in `[0, 255]`. Burst mode ignores queues; tonic mode
    /// uses priority.
    pub priority: u8,
}

/// Per-source channel: `source_id` + destinations + per-destination
/// attention weights + relay mode.
///
/// Constructed by `ThalamicRouter::open_channel`; adapters hold this
/// across steps and call `get_gate` / `submit` on it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThalamicChannel {
    /// Stable source identifier (per-network, per-adapter).
    pub source_id: u32,
    /// Number of valid entries in `destinations` / `attention_weights`.
    pub n_destinations: u32,
    /// Destination IDs — only the first `n_destinations` are meaningful.
    pub destinations: [u32; THALAMIC_MAX_DESTINATIONS],
    /// Attention gate per destination, in `[0, 1]`. `1.0` = full-strength
    /// relay; `0.0` = gated off.
    pub attention_weights: [f32; THALAMIC_MAX_DESTINATIONS],
    /// Current relay mode.
    pub mode: RelayMode,
    /// Counter of submits this step (reset by `tick_stats`).
    pub submits_this_step: u32,
}

impl ThalamicChannel {
    /// Construct a fresh channel with `n_dest` destinations, all gates
    /// at `1.0` (no modulation), tonic mode. Returns `None` if
    /// `n_dest > THALAMIC_MAX_DESTINATIONS`.
    #[must_use]
    pub fn new(source_id: u32, destinations: &[u32]) -> Option<Self> {
        if destinations.len() > THALAMIC_MAX_DESTINATIONS {
            return None;
        }
        let mut ch = Self {
            source_id,
            n_destinations: destinations.len() as u32,
            destinations: [0; THALAMIC_MAX_DESTINATIONS],
            attention_weights: [1.0; THALAMIC_MAX_DESTINATIONS],
            mode: RelayMode::default(),
            submits_this_step: 0,
        };
        for (slot, &dest) in ch.destinations.iter_mut().zip(destinations.iter()) {
            *slot = dest;
        }
        Some(ch)
    }

    /// Look up the current attention gate for a specific destination.
    /// Returns `1.0` (no modulation) when `dest_id` isn't in this
    /// channel — lets callers treat unknown routes as full-strength
    /// rather than zeroed.
    #[must_use]
    pub fn get_gate(&self, dest_id: u32) -> f32 {
        for i in 0..self.n_destinations as usize {
            if self.destinations[i] == dest_id {
                return self.attention_weights[i].clamp(0.0, 1.0);
            }
        }
        1.0
    }

    /// Set the attention gate for a destination. Returns `true` if the
    /// destination was found and updated.
    pub fn set_gate(&mut self, dest_id: u32, gate: f32) -> bool {
        for i in 0..self.n_destinations as usize {
            if self.destinations[i] == dest_id {
                self.attention_weights[i] = gate.clamp(0.0, 1.0);
                return true;
            }
        }
        false
    }

    /// Increment the per-step submit counter — adapters call this once
    /// per actual `submit()` so the router's Hebbian update has a
    /// count to work with.
    pub fn record_submit(&mut self) {
        self.submits_this_step = self.submits_this_step.saturating_add(1);
    }

    /// Zero the submit counter — called by the router at the end of
    /// each tick.
    pub fn tick_stats(&mut self) {
        self.submits_this_step = 0;
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn new_caps_at_max_destinations() {
        let too_many: Vec<u32> = (0..32).collect();
        assert!(ThalamicChannel::new(0, &too_many).is_none());
    }

    #[test]
    fn new_initializes_gates_to_one() {
        let ch = ThalamicChannel::new(7, &[1, 2, 3]).unwrap();
        assert_eq!(ch.source_id, 7);
        assert_eq!(ch.n_destinations, 3);
        assert_eq!(ch.get_gate(1), 1.0);
        assert_eq!(ch.get_gate(2), 1.0);
        assert_eq!(ch.get_gate(3), 1.0);
    }

    #[test]
    fn get_gate_unknown_destination_returns_one() {
        let ch = ThalamicChannel::new(0, &[1]).unwrap();
        assert_eq!(ch.get_gate(999), 1.0);
    }

    #[test]
    fn set_gate_returns_false_for_unknown() {
        let mut ch = ThalamicChannel::new(0, &[1]).unwrap();
        assert!(!ch.set_gate(999, 0.5));
    }

    #[test]
    fn set_gate_clamps_out_of_range() {
        let mut ch = ThalamicChannel::new(0, &[1]).unwrap();
        assert!(ch.set_gate(1, 2.0));
        assert_eq!(ch.get_gate(1), 1.0);
        assert!(ch.set_gate(1, -0.5));
        assert_eq!(ch.get_gate(1), 0.0);
    }

    #[test]
    fn record_submit_and_tick_stats() {
        let mut ch = ThalamicChannel::new(0, &[1]).unwrap();
        ch.record_submit();
        ch.record_submit();
        assert_eq!(ch.submits_this_step, 2);
        ch.tick_stats();
        assert_eq!(ch.submits_this_step, 0);
    }

    #[test]
    fn default_mode_is_tonic() {
        let ch = ThalamicChannel::new(0, &[1]).unwrap();
        assert_eq!(ch.mode, RelayMode::Tonic);
    }
}
