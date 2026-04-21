//! Command dispatch + [`BrainBackend`] trait.
//!
//! The daemon does not call the V2 brain directly — it calls the
//! [`BrainBackend`] trait. That indirection lets us:
//!
//! - Compile + test the daemon without any dependency on V2's brain
//!   (or V1's C library). The in-tree [`StubBackend`] satisfies the
//!   trait with trivial responses.
//! - Plug in a future `V1Backend` (wired behind the `v1` cargo feature)
//!   that forwards to the V1 C brain via `nimcp-v1-bridge` — that crate
//!   does not exist yet and landing it is parallel work.
//! - Plug in a native V2 backend once the V2 brain exposes the matching
//!   API surface (Phase 2+).
//!
//! The critical-path commands are the ones V1's curriculum driver calls
//! frequently. The JSON shapes below are byte-compatible with V1's
//! `scripts/brain_daemon.py::BrainService::_cmd_*`.

use async_trait::async_trait;
use serde_json::{Value, json};
use std::path::PathBuf;

/// Sensory modality submitted via the `submit_sensory` command.
///
/// V1 accepts string modalities; the backend treats unknown strings as
/// no-ops. We preserve the raw string here so backends can pass it
/// through without loss.
#[derive(Debug, Clone)]
pub struct SensoryInput {
    /// "visual" | "audio" | "speech" | "somatosensory" | ...
    pub modality: String,
    /// Opaque payload — the backend is responsible for shape checks.
    pub data: Value,
    /// Optional shape hints (width/height/channels/n_segments).
    pub shape: Value,
}

/// Arguments for a single-sample learn step.
#[derive(Debug, Clone)]
pub struct LearnVectorArgs {
    /// Input features.
    pub features: Vec<f32>,
    /// Target output.
    pub target: Vec<f32>,
    /// Optional label (e.g., class name) for supervised signals.
    pub label: Option<String>,
    /// Optional confidence (0..=1) weighting the learn step.
    pub confidence: Option<f32>,
    /// Optional per-call learning rate override.
    pub learning_rate: Option<f32>,
}

/// Result of `decide_full` — matches the V1 response shape closely
/// enough that the curriculum driver's dict lookups still work.
#[derive(Debug, Clone)]
pub struct DecideFull {
    /// The brain's output vector (e.g. logits).
    pub output_vector: Vec<f32>,
    /// Optional top-k predictions, label + confidence pairs.
    pub predictions: Vec<(String, f32)>,
}

/// Summary of brain state produced by `status`. Freeform map so
/// backends can add fields without a trait change.
#[derive(Debug, Clone, Default)]
pub struct StatusReport {
    /// Per-key status fields. Flattened into the JSON response.
    pub fields: serde_json::Map<String, Value>,
}

/// The narrow trait every backend implements.
///
/// Methods are async because a real backend will talk to an actor
/// mailbox or to a C brain via a worker thread — both can block. The
/// server loop serializes access via `tokio::sync::Mutex<dyn BrainBackend>`,
/// so backends can assume no concurrent method calls.
#[async_trait]
pub trait BrainBackend: Send + Sync {
    /// Cheap liveness check — returns immediately.
    async fn ping(&mut self) -> Result<(), BackendError>;

    /// One gradient step; returns the resulting loss.
    async fn learn_vector(&mut self, args: LearnVectorArgs) -> Result<f32, BackendError>;

    /// Forward pass + top-k predictions.
    async fn decide_full(&mut self, features: Vec<f32>) -> Result<DecideFull, BackendError>;

    /// Stage sensory input for consumption by the next learn call.
    async fn submit_sensory(&mut self, input: SensoryInput) -> Result<(), BackendError>;

    /// Fire-and-forget reward / RPE update. Return fast; the real work
    /// may complete after the response is sent.
    async fn bg_update_reward(&mut self, reward: f32, rpe: Option<f32>)
    -> Result<(), BackendError>;

    /// Snapshot current brain state. Freeform key-value map.
    async fn status(&mut self) -> Result<StatusReport, BackendError>;

    /// Freeform stats — combined with `status` in the V1 response shape.
    async fn get_stats(&mut self) -> Result<StatusReport, BackendError>;

    /// Atomic checkpoint to `path`.
    async fn save(&mut self, path: PathBuf) -> Result<(), BackendError>;

    /// Reload brain state from `path`. V1's daemon doesn't implement this
    /// (the daemon loads once at startup) but the V2 daemon will support
    /// hot-reloading.
    async fn load(&mut self, path: PathBuf) -> Result<(), BackendError>;

    /// Total live neuron count — useful as a sanity check after `load`.
    async fn get_neuron_count(&mut self) -> Result<u32, BackendError>;
}

/// Error surface for backend calls. Kept deliberately coarse — the
/// daemon forwards the message verbatim to the client as the
/// `"error"` field, matching V1's behavior.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The request was malformed for this backend (shape mismatch etc).
    #[error("bad request: {0}")]
    BadRequest(String),

    /// The backend itself failed (C library error, actor panic, etc.).
    #[error("backend: {0}")]
    Backend(String),

    /// The backend is stopping and can no longer accept requests.
    #[error("backend is shutting down")]
    ShuttingDown,
}

// ---------------------------------------------------------------------------
// StubBackend — in-tree implementation for tests + bootstrap.
// ---------------------------------------------------------------------------

/// A backend that records what it was asked and returns trivial values.
///
/// Used by the daemon's `--backend stub` mode and by the crate's
/// integration tests. Not wired to any real brain.
#[derive(Debug, Default)]
pub struct StubBackend {
    /// Number of `learn_vector` calls.
    pub learn_calls: u64,
    /// Number of `decide_full` calls.
    pub infer_calls: u64,
    /// Sum of `reward` arguments seen via `bg_update_reward`.
    pub reward_total: f64,
    /// Last `submit_sensory` modality seen (if any).
    pub last_modality: Option<String>,
    /// Last path passed to `save`.
    pub last_save_path: Option<PathBuf>,
    /// Last path passed to `load`.
    pub last_load_path: Option<PathBuf>,
}

#[async_trait]
impl BrainBackend for StubBackend {
    async fn ping(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    async fn learn_vector(&mut self, args: LearnVectorArgs) -> Result<f32, BackendError> {
        // Minimal sanity: non-empty feature and target vectors.
        if args.features.is_empty() || args.target.is_empty() {
            return Err(BackendError::BadRequest(
                "features and target must be non-empty".into(),
            ));
        }
        self.learn_calls += 1;
        // Stub loss is 0.0 — tests assert on the shape, not the value.
        Ok(0.0)
    }

    async fn decide_full(&mut self, features: Vec<f32>) -> Result<DecideFull, BackendError> {
        if features.is_empty() {
            return Err(BackendError::BadRequest(
                "features must be non-empty".into(),
            ));
        }
        self.infer_calls += 1;
        Ok(DecideFull {
            output_vector: vec![0.0; features.len().min(16)],
            predictions: Vec::new(),
        })
    }

    async fn submit_sensory(&mut self, input: SensoryInput) -> Result<(), BackendError> {
        self.last_modality = Some(input.modality);
        Ok(())
    }

    async fn bg_update_reward(
        &mut self,
        reward: f32,
        _rpe: Option<f32>,
    ) -> Result<(), BackendError> {
        self.reward_total += reward as f64;
        Ok(())
    }

    async fn status(&mut self) -> Result<StatusReport, BackendError> {
        let mut fields = serde_json::Map::new();
        fields.insert("learn_calls".into(), json!(self.learn_calls));
        fields.insert("infer_calls".into(), json!(self.infer_calls));
        fields.insert("reward_total".into(), json!(self.reward_total));
        fields.insert("backend".into(), json!("stub"));
        Ok(StatusReport { fields })
    }

    async fn get_stats(&mut self) -> Result<StatusReport, BackendError> {
        // Stub has no deeper stats beyond what status reports.
        self.status().await
    }

    async fn save(&mut self, path: PathBuf) -> Result<(), BackendError> {
        self.last_save_path = Some(path);
        Ok(())
    }

    async fn load(&mut self, path: PathBuf) -> Result<(), BackendError> {
        self.last_load_path = Some(path);
        Ok(())
    }

    async fn get_neuron_count(&mut self) -> Result<u32, BackendError> {
        // Report a plausible but fixed count so test assertions can match.
        Ok(0)
    }
}

// ---------------------------------------------------------------------------
// Dispatch.
// ---------------------------------------------------------------------------

/// Dispatch one parsed request to the backend and build the response.
///
/// Unknown commands map to `{"error": "unknown command: <name>"}`, matching
/// V1's `BrainService.handle_readonly` / `handle` behavior. Backend errors
/// are serialized as `{"error": "<msg>"}` so V1's `BrainProxy._send_once`
/// can raise them as `RuntimeError`s unchanged.
pub async fn handle_request(backend: &mut dyn BrainBackend, req: Value) -> Value {
    let cmd = match req.get("cmd").and_then(Value::as_str) {
        Some(c) => c.to_string(),
        None => return json!({"error": "missing 'cmd' field"}),
    };

    // A small helper for producing {"error": "..."} bodies. Kept local so
    // the dispatch arms stay readable.
    fn err_response<E: std::fmt::Display>(e: E) -> Value {
        json!({ "error": e.to_string() })
    }

    match cmd.as_str() {
        "ping" => match backend.ping().await {
            Ok(()) => json!({"ok": true}),
            Err(e) => err_response(e),
        },

        "learn_vector" => {
            let args = match parse_learn_vector(&req) {
                Ok(a) => a,
                Err(e) => return err_response(e),
            };
            match backend.learn_vector(args).await {
                Ok(loss) => json!({"loss": loss}),
                Err(e) => err_response(e),
            }
        }

        "decide_full" => {
            let features = match parse_f32_array(&req, "features") {
                Ok(v) => v,
                Err(e) => return err_response(e),
            };
            match backend.decide_full(features).await {
                Ok(result) => json!({
                    "result": {
                        "output_vector": result.output_vector,
                        "predictions": result.predictions
                            .into_iter()
                            .map(|(lab, conf)| json!({"label": lab, "confidence": conf}))
                            .collect::<Vec<_>>(),
                    }
                }),
                Err(e) => err_response(e),
            }
        }

        "submit_sensory" => {
            let modality = match req.get("modality").and_then(Value::as_str) {
                Some(s) => s.to_string(),
                None => return err_response("missing 'modality'"),
            };
            let data = req.get("data").cloned().unwrap_or(Value::Null);
            // Everything else on the request (width/height/channels/n_segments)
            // is forwarded as the "shape" bag. Backends are free to ignore it.
            let mut shape = serde_json::Map::new();
            if let Some(obj) = req.as_object() {
                for (k, v) in obj {
                    if matches!(k.as_str(), "cmd" | "modality" | "data") {
                        continue;
                    }
                    shape.insert(k.clone(), v.clone());
                }
            }
            let input = SensoryInput {
                modality,
                data,
                shape: Value::Object(shape),
            };
            match backend.submit_sensory(input).await {
                Ok(()) => json!({"ok": true}),
                Err(e) => err_response(e),
            }
        }

        "bg_update_reward" => {
            let reward = match req.get("reward").and_then(Value::as_f64) {
                Some(v) => v as f32,
                None => return err_response("missing or non-numeric 'reward'"),
            };
            let rpe = req.get("rpe").and_then(Value::as_f64).map(|v| v as f32);
            // Fire-and-forget semantics: we still `.await` because the stub
            // returns immediately. A real backend can spawn the actual
            // update onto a worker and return before it finishes.
            match backend.bg_update_reward(reward, rpe).await {
                Ok(()) => json!({"ok": true}),
                Err(e) => err_response(e),
            }
        }

        "status" => match backend.status().await {
            Ok(report) => {
                let mut out = serde_json::Map::new();
                out.insert("ok".into(), json!(true));
                for (k, v) in report.fields {
                    out.insert(k, v);
                }
                Value::Object(out)
            }
            Err(e) => err_response(e),
        },

        "get_stats" => match backend.get_stats().await {
            Ok(report) => json!({"ok": true, "stats": Value::Object(report.fields)}),
            Err(e) => err_response(e),
        },

        "save" => {
            let path = match req.get("path").and_then(Value::as_str) {
                Some(s) => PathBuf::from(s),
                None => return err_response("missing 'path'"),
            };
            match backend.save(path.clone()).await {
                Ok(()) => json!({"ok": true, "path": path.display().to_string()}),
                Err(e) => err_response(e),
            }
        }

        "load" => {
            let path = match req.get("path").and_then(Value::as_str) {
                Some(s) => PathBuf::from(s),
                None => return err_response("missing 'path'"),
            };
            match backend.load(path.clone()).await {
                Ok(()) => json!({"ok": true, "path": path.display().to_string()}),
                Err(e) => err_response(e),
            }
        }

        "get_neuron_count" => match backend.get_neuron_count().await {
            Ok(n) => json!({"neuron_count": n}),
            Err(e) => err_response(e),
        },

        // `shutdown` is handled by the server loop — by the time we see
        // it here, the server has already arranged for graceful exit.
        // We still return `{"ok": true}` so the client gets a response.
        "shutdown" => json!({"ok": true, "message": "shutting down"}),

        other => json!({
            "error": format!("unknown command: {other}")
        }),
    }
}

// ---- request parsing helpers ----

fn parse_f32_array(req: &Value, key: &str) -> Result<Vec<f32>, String> {
    let arr = req
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or non-array '{key}'"))?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, v) in arr.iter().enumerate() {
        let f = v
            .as_f64()
            .ok_or_else(|| format!("'{key}'[{i}] is not a number"))?;
        out.push(f as f32);
    }
    Ok(out)
}

fn parse_learn_vector(req: &Value) -> Result<LearnVectorArgs, String> {
    let features = parse_f32_array(req, "features")?;
    let target = parse_f32_array(req, "target")?;
    let label = req.get("label").and_then(Value::as_str).map(str::to_owned);
    let confidence = req
        .get("confidence")
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    let learning_rate = req
        .get("learning_rate")
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    Ok(LearnVectorArgs {
        features,
        target,
        label,
        confidence,
        learning_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ping_returns_ok_true() {
        let mut b = StubBackend::default();
        let resp = handle_request(&mut b, json!({"cmd": "ping"})).await;
        assert_eq!(resp, json!({"ok": true}));
    }

    #[tokio::test]
    async fn learn_vector_returns_loss_field() {
        let mut b = StubBackend::default();
        let resp = handle_request(
            &mut b,
            json!({
                "cmd": "learn_vector",
                "features": [0.1, 0.2, 0.3],
                "target": [1.0, 0.0],
                "label": "a",
                "confidence": 0.9,
                "learning_rate": 0.001
            }),
        )
        .await;
        assert!(resp.get("loss").is_some(), "missing loss: {resp}");
        assert_eq!(b.learn_calls, 1);
    }

    #[tokio::test]
    async fn learn_vector_rejects_missing_features() {
        let mut b = StubBackend::default();
        let resp = handle_request(&mut b, json!({"cmd": "learn_vector", "target": [1.0]})).await;
        assert!(resp.get("error").is_some(), "expected error: {resp}");
    }

    #[tokio::test]
    async fn decide_full_wraps_result_in_result_field() {
        let mut b = StubBackend::default();
        let resp = handle_request(
            &mut b,
            json!({"cmd": "decide_full", "features": [0.1, 0.2]}),
        )
        .await;
        assert!(resp.get("result").is_some(), "missing result: {resp}");
        assert_eq!(b.infer_calls, 1);
    }

    #[tokio::test]
    async fn submit_sensory_forwards_shape_fields() {
        let mut b = StubBackend::default();
        let resp = handle_request(
            &mut b,
            json!({
                "cmd": "submit_sensory",
                "modality": "visual",
                "data": [0.0, 1.0],
                "width": 32, "height": 32, "channels": 3
            }),
        )
        .await;
        assert_eq!(resp, json!({"ok": true}));
        assert_eq!(b.last_modality.as_deref(), Some("visual"));
    }

    #[tokio::test]
    async fn bg_update_reward_is_fire_and_forget() {
        let mut b = StubBackend::default();
        let resp = handle_request(
            &mut b,
            json!({"cmd": "bg_update_reward", "reward": 0.5, "rpe": 0.3}),
        )
        .await;
        assert_eq!(resp, json!({"ok": true}));
        assert!((b.reward_total - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn save_and_load_echo_path() {
        let mut b = StubBackend::default();
        let save = handle_request(&mut b, json!({"cmd": "save", "path": "/tmp/ckpt"})).await;
        assert_eq!(save["ok"], json!(true));
        assert_eq!(b.last_save_path, Some(PathBuf::from("/tmp/ckpt")));

        let load = handle_request(&mut b, json!({"cmd": "load", "path": "/tmp/ckpt"})).await;
        assert_eq!(load["ok"], json!(true));
        assert_eq!(b.last_load_path, Some(PathBuf::from("/tmp/ckpt")));
    }

    #[tokio::test]
    async fn get_neuron_count_shape() {
        let mut b = StubBackend::default();
        let resp = handle_request(&mut b, json!({"cmd": "get_neuron_count"})).await;
        assert!(resp.get("neuron_count").is_some());
    }

    #[tokio::test]
    async fn unknown_command_yields_error() {
        let mut b = StubBackend::default();
        let resp = handle_request(&mut b, json!({"cmd": "no_such_command"})).await;
        assert!(resp.get("error").is_some());
    }

    #[tokio::test]
    async fn missing_cmd_is_an_error() {
        let mut b = StubBackend::default();
        let resp = handle_request(&mut b, json!({"features": [1.0]})).await;
        assert!(resp.get("error").is_some());
    }

    #[tokio::test]
    async fn status_flattens_report_fields() {
        let mut b = StubBackend::default();
        let resp = handle_request(&mut b, json!({"cmd": "status"})).await;
        assert_eq!(resp["ok"], json!(true));
        assert_eq!(resp["backend"], json!("stub"));
    }
}
