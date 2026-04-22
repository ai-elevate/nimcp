//! Wire protocol for the NIMCP V2 daemon.
//!
//! Every request / response is one line of JSON (newline-delimited). Requests
//! carry a `cmd` discriminant; responses carry a `status` discriminant
//! (`ok` with a `data` payload, or `err` with a `message`). The tagged-
//! enum layout is stable across language boundaries — Python clients can
//! build the same JSON by hand if they don't want to pull in a daemon
//! SDK.

use serde::{Deserialize, Serialize};

/// One request from a client to the daemon.
///
/// The JSON wire form is `{"cmd": "<name>", ...fields}`. Unknown fields
/// in a valid variant are ignored (serde default).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    /// Liveness check — responds with `{ "pong": true }`.
    Ping,

    /// One training step on the adaptive MLP.
    Learn {
        /// Input feature vector (length must match the first configured layer).
        features: Vec<f32>,
        /// Target vector (length must match the last configured layer).
        target: Vec<f32>,
        /// Learning rate for this step.
        lr: f32,
    },

    /// One forward pass on the adaptive MLP.
    Predict {
        /// Input feature vector.
        features: Vec<f32>,
    },

    /// Return the full [`nimcp_brain::stats::BrainStats`] JSON blob.
    Stats,

    /// Save the full ensemble (adaptive + optional SNN/LNN/memory) to `dir`.
    SaveEnsemble {
        /// Target directory. Parent must exist; the directory itself is
        /// written atomically via `<dir>.tmp/` + rename.
        dir: String,
    },

    /// Restore from a prior [`Request::SaveEnsemble`] directory.
    LoadEnsemble {
        /// Source directory previously written by `SaveEnsemble`.
        dir: String,
    },

    /// One SNN integration step. `drive[p]` is the external current vector
    /// for population `p` (length must match `n_neurons`, else silently
    /// ignored by the SNN).
    SnnStep {
        /// Per-population external drive.
        drive: Vec<Vec<f32>>,
        /// Reward signal for R-STDP.
        reward: f32,
        /// Integration timestep, ms.
        dt_ms: f32,
    },

    /// Return every population's firing-rate EMA, spike count, neuron
    /// count, and optional DFA pink-alpha. Matches the field names the
    /// SNN watchdog expects.
    SnnPopStats,

    /// Return a snapshot of the live-tunable parameters. V2 has no
    /// canonical live-tuning surface yet — the daemon returns the
    /// config-time values of well-known knobs so the existing watchdog
    /// can reason about them.
    SnnTuneGet,

    /// Best-effort set of a named tunable. Unknown names are rejected.
    SnnTune {
        /// Parameter name (e.g. `"rstdp_lr"`, `"target_rate"`).
        name: String,
        /// New value.
        value: f32,
    },

    /// Best-effort quench: zero the first `n` neurons' membrane voltages
    /// across every population. V2's SNN does not yet expose a public
    /// `v_mem_mut` so this returns a `note` and a zero `zeroed` count
    /// rather than erroring — keeps the watchdog RPC stable.
    SnnForceQuench {
        /// Number of neurons to quench per population.
        n: u32,
    },

    /// Shut the daemon down cleanly.
    Shutdown,
}

/// One response from the daemon to a client. `Ok` carries an arbitrary
/// JSON payload keyed by the request variant. `Err` carries a human-
/// readable message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Response {
    /// Successful response.
    Ok {
        /// Variant-specific JSON body.
        data: serde_json::Value,
    },
    /// Error response.
    Err {
        /// Operator-readable error text.
        message: String,
    },
}

impl Response {
    /// Build an `Ok` response from any serializable value. Serialization
    /// failure degrades to an `Err` so the transport layer always sends
    /// something.
    pub fn ok<T: Serialize>(data: T) -> Self {
        match serde_json::to_value(data) {
            Ok(v) => Self::Ok { data: v },
            Err(e) => Self::Err {
                message: format!("serialize response: {e}"),
            },
        }
    }

    /// Build an `Err` response from any `Display` value.
    pub fn err(msg: impl std::fmt::Display) -> Self {
        Self::Err {
            message: msg.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_round_trip() {
        let r = Request::Ping;
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(s, "{\"cmd\":\"ping\"}");
        let back: Request = serde_json::from_str(&s).unwrap();
        assert!(matches!(back, Request::Ping));
    }

    #[test]
    fn learn_round_trip() {
        let r = Request::Learn {
            features: vec![0.1, 0.2],
            target: vec![1.0],
            lr: 0.05,
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: Request = serde_json::from_str(&s).unwrap();
        match back {
            Request::Learn {
                features,
                target,
                lr,
            } => {
                assert_eq!(features, vec![0.1, 0.2]);
                assert_eq!(target, vec![1.0]);
                assert!((lr - 0.05).abs() < 1e-6);
            }
            _ => panic!("expected Learn"),
        }
    }

    #[test]
    fn snn_tune_round_trip() {
        let r = Request::SnnTune {
            name: "rstdp_lr".into(),
            value: 1e-4,
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"cmd\":\"snn_tune\""));
        let back: Request = serde_json::from_str(&s).unwrap();
        match back {
            Request::SnnTune { name, value } => {
                assert_eq!(name, "rstdp_lr");
                assert!((value - 1e-4).abs() < 1e-10);
            }
            _ => panic!("expected SnnTune"),
        }
    }

    #[test]
    fn snn_step_round_trip() {
        let r = Request::SnnStep {
            drive: vec![vec![1.0, 2.0], vec![]],
            reward: 0.5,
            dt_ms: 1.0,
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: Request = serde_json::from_str(&s).unwrap();
        match back {
            Request::SnnStep {
                drive,
                reward,
                dt_ms,
            } => {
                assert_eq!(drive.len(), 2);
                assert_eq!(drive[0], vec![1.0, 2.0]);
                assert!(drive[1].is_empty());
                assert!((reward - 0.5).abs() < 1e-6);
                assert!((dt_ms - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected SnnStep"),
        }
    }

    #[test]
    fn save_load_ensemble_round_trip() {
        let r = Request::SaveEnsemble {
            dir: "/tmp/nimcp-test".into(),
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: Request = serde_json::from_str(&s).unwrap();
        match back {
            Request::SaveEnsemble { dir } => assert_eq!(dir, "/tmp/nimcp-test"),
            _ => panic!("expected SaveEnsemble"),
        }
    }

    #[test]
    fn shutdown_round_trip() {
        let r = Request::Shutdown;
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(s, "{\"cmd\":\"shutdown\"}");
    }

    #[test]
    fn response_ok_builds_value() {
        let r = Response::ok(serde_json::json!({"pong": true}));
        let s = serde_json::to_string(&r).unwrap();
        // Tagged as ok with a data object.
        assert!(s.contains("\"status\":\"ok\""));
        assert!(s.contains("\"pong\":true"));
    }

    #[test]
    fn response_err_builds_message() {
        let r = Response::err("bad request");
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"status\":\"err\""));
        assert!(s.contains("\"message\":\"bad request\""));
    }

    #[test]
    fn unknown_variant_rejected() {
        let s = "{\"cmd\":\"totally_not_a_real_command\"}";
        let result: Result<Request, _> = serde_json::from_str(s);
        assert!(result.is_err());
    }

    /// Sanity — the response enum also round-trips through JSON so we
    /// don't rely on serde's default behaviour being symmetrical.
    #[test]
    fn response_round_trip() {
        let r = Response::Ok {
            data: serde_json::json!({"x": 1}),
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: Response = serde_json::from_str(&s).unwrap();
        match back {
            Response::Ok { data } => assert_eq!(data["x"], 1),
            Response::Err { .. } => panic!("expected Ok"),
        }
    }
}
