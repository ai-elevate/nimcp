//! Similarity-based retrieval: "what memory is most like this feature
//! vector?" V1 used a per-node prime signature; V2 uses plain cosine
//! similarity on the feature vector. Equivalent in spirit: a lookup
//! key that survives across re-embeddings of the same content.
//!
//! Two query scopes:
//!
//! - [`ZLadder::query_landmarks_by_similarity`] — the elephant-matriarch
//!   path. Scans only the landmark set.
//! - [`ZLadder::query_all_tiers`] — general oracle retrieval across
//!   every node in every tier.

use serde::{Deserialize, Serialize};

use crate::ladder::ZLadder;
use crate::node::{MemoryNode, Tier};

/// One result entry returned by the similarity queries. Sorted in
/// descending order of `similarity`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QueryHit {
    /// Matching node's ID.
    pub node_id: u64,
    /// Cosine similarity in `[-1, +1]`. `1.0` = identical direction,
    /// `0.0` = orthogonal, `-1.0` = opposite.
    pub similarity: f32,
    /// Which tier the node currently lives in.
    pub tier: Tier,
}

/// Cosine similarity between two vectors. Returns `0.0` if either
/// vector is all zeros or the lengths don't match — callers treat
/// such hits as "no match" rather than propagating an error.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

impl ZLadder {
    /// Query the **landmark** subset for the top-`k` cosine matches of
    /// `query`. Landmarks live exclusively at `Z3` by construction
    /// (see [`ZLadder::mark_landmark`]), so all returned hits are at
    /// `Tier::Z3`.
    ///
    /// Returned `Vec<QueryHit>` is sorted descending by similarity and
    /// clipped to at most `k` entries.
    #[must_use]
    pub fn query_landmarks_by_similarity(&self, query: &[f32], k: usize) -> Vec<QueryHit> {
        let mut hits: Vec<QueryHit> = self
            .landmarks()
            .filter_map(|(id, _)| self.find(id).map(|n| (id, n)))
            .map(|(id, node)| QueryHit {
                node_id: id,
                similarity: cosine(query, &node.features),
                tier: node.tier,
            })
            .collect();
        hits.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
    }

    /// Query **every** node in every tier for the top-`k` cosine
    /// matches of `query`. Intended for general retrieval when the
    /// landmark set doesn't have enough high-confidence matches.
    #[must_use]
    pub fn query_all_tiers(&self, query: &[f32], k: usize) -> Vec<QueryHit> {
        let mut hits: Vec<QueryHit> = self
            .iter_nodes()
            .map(|n: &MemoryNode| QueryHit {
                node_id: n.id,
                similarity: cosine(query, &n.features),
                tier: n.tier,
            })
            .collect();
        hits.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::node::MemoryNode;

    #[test]
    fn cosine_identical_vectors_is_one() {
        let s = cosine(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((s - 1.0).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn cosine_opposite_vectors_is_minus_one() {
        let s = cosine(&[1.0, 0.0], &[-1.0, 0.0]);
        assert!((s - (-1.0)).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let s = cosine(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn cosine_mismatched_dim_returns_zero() {
        let s = cosine(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let s = cosine(&[0.0, 0.0], &[1.0, 2.0]);
        assert_eq!(s, 0.0);
    }

    fn node_at(id: u64, features: Vec<f32>) -> MemoryNode {
        MemoryNode::new(id, features, 0)
    }

    #[test]
    fn query_all_tiers_returns_top_k_in_order() {
        let mut l = ZLadder::with_defaults();
        // Features: closer to [1, 0] = higher similarity.
        l.insert(node_at(1, vec![1.0, 0.0])).unwrap();
        l.insert(node_at(2, vec![0.9, 0.1])).unwrap();
        l.insert(node_at(3, vec![0.0, 1.0])).unwrap();

        let hits = l.query_all_tiers(&[1.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].node_id, 1, "identical match first");
        assert_eq!(hits[1].node_id, 2, "second closest second");
    }

    #[test]
    fn query_landmarks_only_scans_landmark_set() {
        let mut l = ZLadder::with_defaults();
        l.insert(node_at(1, vec![1.0, 0.0])).unwrap();
        l.insert(node_at(2, vec![0.9, 0.1])).unwrap();
        l.mark_landmark(1, "identity").unwrap();
        // Node 2 is a closer match on similarity, but not a landmark.
        let hits = l.query_landmarks_by_similarity(&[0.9, 0.1], 5);
        assert_eq!(hits.len(), 1, "only landmark 1 should appear");
        assert_eq!(hits[0].node_id, 1);
    }

    #[test]
    fn query_empty_ladder_returns_empty() {
        let l = ZLadder::with_defaults();
        assert!(l.query_all_tiers(&[0.1, 0.2], 5).is_empty());
        assert!(l.query_landmarks_by_similarity(&[0.1], 5).is_empty());
    }

    #[test]
    fn query_truncates_to_k() {
        let mut l = ZLadder::with_defaults();
        for i in 0..5 {
            l.insert(node_at(i, vec![i as f32, 0.0])).unwrap();
        }
        let hits = l.query_all_tiers(&[1.0, 0.0], 3);
        assert_eq!(hits.len(), 3);
    }
}
