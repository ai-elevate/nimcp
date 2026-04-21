//! Phase 5c integration tests — Z-Ladder wired into [`Brain`] +
//! the V2_PLAN Phase 5 exit criterion:
//!
//! > landmark save/load preserves features; query returns expected hits.
//!
//! Two tests:
//!
//! - [`memory_landmarks_survive_ensemble_round_trip`] — the exit
//!   criterion proper. Insert a mix of regular + landmark nodes,
//!   query by cosine similarity, checkpoint the ensemble, reboot a
//!   fresh brain, load, confirm every hit (and every node's feature
//!   vector) is bit-identical.
//! - [`memory_queries_without_landmarks_work`] — sanity: the
//!   general oracle path returns correct top-k when no nodes are
//!   landmarked.

use nimcp_brain::{Brain, BrainConfig};
use nimcp_memory::{MemoryNode, Tier, ZLadderConfig};

fn brain_with_memory(seed: u64) -> Brain {
    let cfg = BrainConfig {
        rng_seed: seed,
        deterministic: true,
        memory: Some(ZLadderConfig::default()),
        ..Default::default()
    };
    Brain::new(cfg).expect("brain with memory")
}

fn node_with(id: u64, feats: Vec<f32>, tier: Tier) -> MemoryNode {
    let mut n = MemoryNode::new(id, feats, 0);
    n.tier = tier;
    n.strength = 1.0;
    n
}

#[tokio::test]
async fn memory_queries_without_landmarks_work() {
    let mut brain = brain_with_memory(7);
    // Three nodes at Z0; one very close to the query vector.
    brain
        .memory_insert(node_with(1, vec![1.0, 0.0, 0.0], Tier::Z0))
        .unwrap();
    brain
        .memory_insert(node_with(2, vec![0.95, 0.1, 0.0], Tier::Z0))
        .unwrap();
    brain
        .memory_insert(node_with(3, vec![0.0, 1.0, 0.0], Tier::Z0))
        .unwrap();

    let hits = brain.memory_query_all(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].node_id, 1, "exact match first");
    assert_eq!(hits[1].node_id, 2, "closest non-exact second");
    assert!(hits[0].similarity > hits[1].similarity);
}

/// V2_PLAN Phase 5 exit criterion.
#[tokio::test]
async fn memory_landmarks_survive_ensemble_round_trip() {
    let cfg = BrainConfig {
        rng_seed: 42,
        deterministic: true,
        memory: Some(ZLadderConfig::default()),
        ..Default::default()
    };
    let mut a = Brain::new(cfg.clone()).unwrap();

    // Four regular nodes scattered across tiers (to exercise the tier
    // breakdown in the snapshot) and two landmarked nodes at Z3.
    a.memory_insert(node_with(1, vec![1.0, 0.0, 0.0, 0.0], Tier::Z0))
        .unwrap();
    a.memory_insert(node_with(2, vec![0.0, 1.0, 0.0, 0.0], Tier::Z1))
        .unwrap();
    a.memory_insert(node_with(3, vec![0.0, 0.0, 1.0, 0.0], Tier::Z2))
        .unwrap();
    a.memory_insert(node_with(4, vec![0.5, 0.5, 0.0, 0.0], Tier::Z0))
        .unwrap();
    a.memory_insert(node_with(10, vec![1.0, 1.0, 0.0, 0.0], Tier::Z3))
        .unwrap();
    a.memory_insert(node_with(11, vec![1.0, 0.0, 0.0, 1.0], Tier::Z3))
        .unwrap();

    // Elevate the two Z3 nodes to landmarks with distinct reasons.
    a.memory_mark_landmark(10, "identity").unwrap();
    a.memory_mark_landmark(11, "first_reward").unwrap();

    // Pre-save: capture baseline query results.
    let query = vec![0.9, 0.1, 0.0, 0.0];
    let all_before = a.memory_query_all(&query, 6).unwrap();
    let landmarks_before = a.memory_query_landmarks(&query, 6).unwrap();
    assert_eq!(landmarks_before.len(), 2, "two landmarks present");
    assert!(
        landmarks_before.iter().all(|h| matches!(h.tier, Tier::Z3)),
        "landmarks must be at Z3"
    );
    // Capture every feature vector so we can verify bit-identical restore.
    let mem_before = a.memory().unwrap();
    let feats_before: Vec<(u64, Vec<f32>)> = mem_before
        .iter_nodes()
        .map(|n| (n.id, n.features.clone()))
        .collect();
    let landmark_count_before = mem_before.landmark_count();

    // Checkpoint the full ensemble.
    let tmp = tempfile::tempdir().unwrap();
    let ensemble = tmp.path().join("brain-phase5");
    a.save_ensemble(&ensemble).unwrap();
    assert!(ensemble.join("memory.json").exists());

    // Reboot with the same config; load.
    let mut b = Brain::new(cfg).unwrap();
    b.load_ensemble(&ensemble).unwrap();

    // Landmark count preserved.
    let mem_after = b.memory().expect("memory present");
    assert_eq!(
        mem_after.landmark_count(),
        landmark_count_before,
        "landmark count changed across round trip"
    );
    assert!(mem_after.is_landmark(10));
    assert!(mem_after.is_landmark(11));

    // **Feature payload round-trips bit-for-bit** — this is the V1 E6
    // "restore doesn't lose the feature" rule.
    for (id, feats) in &feats_before {
        let node = mem_after.find(*id).expect("node missing after load");
        assert_eq!(
            &node.features, feats,
            "feature drift on node {id}: {:?} vs {:?}",
            node.features, feats
        );
    }

    // Queries return the same hits.
    let all_after = b.memory_query_all(&query, 6).unwrap();
    assert_eq!(all_before.len(), all_after.len(), "all-tier hit count");
    for (ha, hb) in all_before.iter().zip(all_after.iter()) {
        assert_eq!(ha.node_id, hb.node_id, "hit order drifted");
        assert!(
            (ha.similarity - hb.similarity).abs() < 1e-6,
            "similarity drift on node {}: {} vs {}",
            ha.node_id,
            ha.similarity,
            hb.similarity
        );
    }

    let landmarks_after = b.memory_query_landmarks(&query, 6).unwrap();
    assert_eq!(landmarks_before.len(), landmarks_after.len());
    for (ha, hb) in landmarks_before.iter().zip(landmarks_after.iter()) {
        assert_eq!(ha.node_id, hb.node_id, "landmark order drifted");
        assert!((ha.similarity - hb.similarity).abs() < 1e-6);
    }
}

#[tokio::test]
async fn memory_refuses_to_load_mismatched_capacity() {
    // Save with default caps, load into a brain whose memory has a
    // different max_landmarks — must be rejected.
    let save_cfg = BrainConfig {
        memory: Some(ZLadderConfig::default()),
        ..Default::default()
    };
    let a = Brain::new(save_cfg).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("mismatch");
    a.save_ensemble(&dir).unwrap();

    let tight_cfg = ZLadderConfig {
        max_landmarks: 8, // was 256
        ..ZLadderConfig::default()
    };
    let reload_cfg = BrainConfig {
        memory: Some(tight_cfg),
        ..Default::default()
    };
    let mut b = Brain::new(reload_cfg).unwrap();
    let err = b.load_ensemble(&dir).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("memory snapshot capacities"),
        "expected capacity-mismatch error, got {msg}",
    );
}
