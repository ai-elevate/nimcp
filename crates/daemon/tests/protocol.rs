//! End-to-end tests: spin up the daemon on a temp socket, talk to it,
//! verify the wire protocol is byte-compatible with V1.

use std::path::{Path, PathBuf};
use std::time::Duration;

use nimcp_daemon::{StubBackend, read_frame, serve, write_frame};
use serde_json::json;
use tempfile::tempdir;
use tokio::net::UnixStream;

/// Build a short socket path inside `dir`. Unix sockets on Linux have
/// a ~108-byte `sun_path` limit; tempdir alone is usually safe but a
/// very deep CI path could push us over. Keep the leaf short.
fn sock(dir: &tempfile::TempDir) -> PathBuf {
    dir.path().join("t.sock")
}

async fn wait_for_socket(path: &Path) {
    for _ in 0..200 {
        if path.exists() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("socket {} did not appear within 2s", path.display());
}

/// One-shot RPC: connect, send one frame, read one frame, disconnect.
async fn rpc(path: &Path, req: serde_json::Value) -> serde_json::Value {
    let mut s = UnixStream::connect(path).await.expect("connect");
    write_frame(&mut s, &req).await.expect("write frame");
    read_frame(&mut s).await.expect("read frame")
}

struct Daemon {
    path: PathBuf,
    task: tokio::task::JoinHandle<()>,
}

impl Daemon {
    async fn start(dir: &tempfile::TempDir) -> Self {
        let path = sock(dir);
        let path_for_server = path.clone();
        let task = tokio::spawn(async move {
            serve(path_for_server, Box::<StubBackend>::default())
                .await
                .expect("server ran");
        });
        wait_for_socket(&path).await;
        Self { path, task }
    }

    async fn stop(self) {
        // Send the shutdown command to ask the server to exit cleanly.
        let _ = rpc(&self.path, json!({"cmd": "shutdown"})).await;
        self.task.await.expect("server task");
    }
}

#[tokio::test]
async fn ping_returns_ok_true() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;

    let resp = rpc(&d.path, json!({"cmd": "ping"})).await;
    assert_eq!(resp, json!({"ok": true}));

    d.stop().await;
}

#[tokio::test]
async fn learn_vector_returns_loss() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;

    let resp = rpc(
        &d.path,
        json!({
            "cmd": "learn_vector",
            "features": [0.1, 0.2, 0.3, -0.4],
            "target": [1.0, 0.0],
            "label": "cat",
            "confidence": 0.9
        }),
    )
    .await;

    // Stub returns 0.0; we assert the SHAPE, not the value.
    assert!(resp.get("loss").is_some(), "missing loss field: {resp}");
    assert!(resp.get("error").is_none(), "unexpected error: {resp}");

    d.stop().await;
}

#[tokio::test]
async fn decide_full_returns_result_wrapper() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;

    let resp = rpc(
        &d.path,
        json!({"cmd": "decide_full", "features": [0.1, 0.2, 0.3]}),
    )
    .await;

    let result = resp.get("result").expect("missing result wrapper");
    assert!(result.get("output_vector").is_some(), "no output_vector");
    assert!(result.get("predictions").is_some(), "no predictions");

    d.stop().await;
}

#[tokio::test]
async fn submit_sensory_accepts_visual_with_shape() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;

    let resp = rpc(
        &d.path,
        json!({
            "cmd": "submit_sensory",
            "modality": "visual",
            "data": [0.0, 1.0, 0.5],
            "width": 32, "height": 32, "channels": 3
        }),
    )
    .await;
    assert_eq!(resp, json!({"ok": true}));

    d.stop().await;
}

#[tokio::test]
async fn bg_update_reward_returns_fast() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;

    let resp = rpc(
        &d.path,
        json!({"cmd": "bg_update_reward", "reward": 0.5, "rpe": 0.3}),
    )
    .await;
    assert_eq!(resp, json!({"ok": true}));

    d.stop().await;
}

#[tokio::test]
async fn unknown_command_yields_error_field() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;

    let resp = rpc(&d.path, json!({"cmd": "does_not_exist"})).await;
    assert!(resp.get("error").is_some(), "expected error: {resp}");
    // V1's BrainProxy._send_once raises on any "error" key; keep the
    // format identical.
    let err = resp["error"].as_str().expect("error is a string");
    assert!(err.contains("does_not_exist"), "error mentions cmd: {err}");

    d.stop().await;
}

#[tokio::test]
async fn get_neuron_count_returns_number() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;

    let resp = rpc(&d.path, json!({"cmd": "get_neuron_count"})).await;
    let n = resp
        .get("neuron_count")
        .and_then(|v| v.as_u64())
        .expect("neuron_count is a number");
    // Stub advertises 0 — we only check the type.
    let _ = n;

    d.stop().await;
}

#[tokio::test]
async fn shutdown_removes_socket_file() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;
    let path = d.path.clone();
    assert!(path.exists());

    d.stop().await;

    assert!(
        !path.exists(),
        "socket file should be removed after shutdown at {}",
        path.display()
    );
}

#[tokio::test]
async fn save_and_load_echo_path() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;

    let save = rpc(
        &d.path,
        json!({"cmd": "save", "path": "/tmp/my-checkpoint"}),
    )
    .await;
    assert_eq!(save["ok"], json!(true));

    let load = rpc(
        &d.path,
        json!({"cmd": "load", "path": "/tmp/my-checkpoint"}),
    )
    .await;
    assert_eq!(load["ok"], json!(true));

    d.stop().await;
}

#[tokio::test]
async fn status_and_get_stats_respond() {
    let dir = tempdir().unwrap();
    let d = Daemon::start(&dir).await;

    let status = rpc(&d.path, json!({"cmd": "status"})).await;
    assert_eq!(status["ok"], json!(true));

    let stats = rpc(&d.path, json!({"cmd": "get_stats"})).await;
    assert_eq!(stats["ok"], json!(true));
    assert!(stats.get("stats").is_some(), "stats wrapper present");

    d.stop().await;
}
