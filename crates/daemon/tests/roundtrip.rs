//! End-to-end roundtrip: spawn the daemon as a subprocess, connect, and
//! exercise every request type. Everything runs under a 10 s timeout so
//! a hung test fails loudly.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// Wall-clock budget for the whole test. Whatever we're doing, if we
/// need more than this the daemon is hung.
const TIMEOUT: Duration = Duration::from_secs(10);

/// Cargo target dir discovery — tests may run from the package or the
/// workspace root, so pick whichever candidate exists.
fn daemon_binary() -> PathBuf {
    // Prefer `CARGO_BIN_EXE_*` when cargo test sets it for the bin.
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_nimcp-v2-daemon") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return pb;
        }
    }
    // Fall back to known layouts.
    let candidates = [
        "../../target/debug/nimcp-v2-daemon",
        "../../target/release/nimcp-v2-daemon",
        "target/debug/nimcp-v2-daemon",
        "target/release/nimcp-v2-daemon",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    panic!(
        "could not locate nimcp-v2-daemon binary; checked CARGO_BIN_EXE_* and {:?}",
        candidates
    );
}

struct DaemonProc {
    child: Child,
    socket: PathBuf,
    metrics_path: PathBuf,
    _tmp: tempfile::TempDir,
}

impl DaemonProc {
    fn spawn() -> Self {
        let tmp = tempfile::tempdir().expect("tempdir");
        let socket = tmp.path().join("daemon.sock");
        let metrics_path = tmp.path().join("metrics.json");
        let state_dir = tmp.path().join("state");
        let training_log = tmp.path().join("training.log");

        let bin = daemon_binary();
        let child = Command::new(&bin)
            .arg("--socket")
            .arg(&socket)
            .arg("--state-dir")
            .arg(&state_dir)
            .arg("--metrics-path")
            .arg(&metrics_path)
            .arg("--training-log")
            .arg(&training_log)
            .env("NIMCP_METRICS_INTERVAL_SEC", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn daemon");

        // Wait for the socket to appear.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !socket.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(25));
        }
        assert!(socket.exists(), "daemon failed to bind socket in 5s");

        Self {
            child,
            socket,
            metrics_path,
            _tmp: tmp,
        }
    }
}

impl Drop for DaemonProc {
    fn drop(&mut self) {
        // Force-kill in case the test bailed before sending Shutdown.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct DaemonClient {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl DaemonClient {
    fn connect(socket: &PathBuf) -> Self {
        let stream = UnixStream::connect(socket).expect("connect socket");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let writer = stream.try_clone().unwrap();
        let reader = BufReader::new(stream);
        Self { reader, writer }
    }

    fn call(&mut self, req: &Value) -> Value {
        let line = serde_json::to_string(req).unwrap();
        writeln!(self.writer, "{line}").expect("write req");
        self.writer.flush().unwrap();
        let mut buf = String::new();
        self.reader.read_line(&mut buf).expect("read resp");
        serde_json::from_str(&buf).expect("decode resp")
    }
}

/// Run the full happy-path scenario. A panic here fails the test.
fn run_scenario() {
    let proc = DaemonProc::spawn();
    let mut client = DaemonClient::connect(&proc.socket);

    // 1. Ping.
    let pong = client.call(&json!({"cmd": "ping"}));
    assert_eq!(pong["status"], "ok", "ping resp: {pong}");
    assert_eq!(pong["data"]["pong"], true);

    // 2. Stats — default brain has layers = [1], so a Learn needs 1-dim
    // features/target. Verify shape via stats first.
    let stats = client.call(&json!({"cmd": "stats"}));
    assert_eq!(stats["status"], "ok", "stats resp: {stats}");
    let first_layer = stats["data"]["adaptive"]["layer_widths"][0]
        .as_u64()
        .expect("first layer width");
    let last_layer = stats["data"]["adaptive"]["layer_widths"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .as_u64()
        .unwrap();

    // 3. Learn with correctly-shaped vectors.
    let feats: Vec<f32> = vec![0.5_f32; first_layer as usize];
    let targs: Vec<f32> = vec![0.0_f32; last_layer as usize];
    let learn_resp = client.call(&json!({
        "cmd": "learn",
        "features": feats,
        "target": targs,
        "lr": 0.01,
    }));
    assert_eq!(learn_resp["status"], "ok", "learn resp: {learn_resp}");
    assert!(learn_resp["data"]["loss"].is_number(), "learn has no loss");

    // 4. Predict.
    let predict_resp = client.call(&json!({
        "cmd": "predict",
        "features": feats,
    }));
    assert_eq!(predict_resp["status"], "ok", "predict resp: {predict_resp}");
    assert_eq!(
        predict_resp["data"]["output"].as_array().unwrap().len(),
        last_layer as usize
    );

    // 5. Stats post-learn sees learn count = 1.
    let stats2 = client.call(&json!({"cmd": "stats"}));
    let loss_count = stats2["data"]["loss"]["adaptive"]["count"]
        .as_u64()
        .unwrap_or(0);
    assert!(loss_count >= 1, "expected loss count >= 1, got {loss_count}");

    // 6. SNN requests should return structured errors (no SNN on default brain).
    let snn_pop = client.call(&json!({"cmd": "snn_pop_stats"}));
    assert_eq!(snn_pop["status"], "err");

    let snn_tune_get = client.call(&json!({"cmd": "snn_tune_get"}));
    assert_eq!(snn_tune_get["status"], "ok", "tune_get: {snn_tune_get}");
    assert!(snn_tune_get["data"]["params"]["rstdp_lr"].is_number());

    // 7. Unknown tunable rejected.
    let bad = client.call(&json!({"cmd": "snn_tune", "name": "nonsense", "value": 1.0}));
    assert_eq!(bad["status"], "err", "unknown tune resp: {bad}");

    // 8. Known tunable accepted.
    let ok = client.call(&json!({"cmd": "snn_tune", "name": "rstdp_lr", "value": 0.002}));
    assert_eq!(ok["status"], "ok", "tune ok: {ok}");

    // 9. Wait for the metrics writer to produce at least one file.
    //    The writer skips the first tick — wait up to 5s.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !proc.metrics_path.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(100));
    }
    assert!(
        proc.metrics_path.exists(),
        "metrics file {:?} not written within 5s",
        proc.metrics_path
    );
    let raw = std::fs::read_to_string(&proc.metrics_path).unwrap();
    let metrics: Value = serde_json::from_str(&raw).expect("metrics.json is valid JSON");
    for key in [
        "timestamp",
        "ok",
        "uptime",
        "learn_calls",
        "infer_calls",
        "errors",
        "neuron_count",
        "ann_loss",
        "snn_loss",
        "lnn_loss",
        "training_active",
        "ann_steps",
        "snn_spikes",
        "snn_rate_hz",
        "snn_sparsity",
    ] {
        assert!(
            metrics.get(key).is_some(),
            "metrics.json missing `{key}`: {metrics}"
        );
    }
    assert_eq!(metrics["ok"], true);
    assert!(metrics["learn_calls"].as_u64().unwrap() >= 1);

    // 10. Shutdown — daemon should exit with code 0 in a couple seconds.
    let bye = client.call(&json!({"cmd": "shutdown"}));
    assert_eq!(bye["status"], "ok", "shutdown resp: {bye}");
    drop(client);

    // Give the process up to 5s to exit.
    let mut proc = proc;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match proc.child.try_wait() {
            Ok(Some(status)) => {
                assert!(status.success(), "daemon exit status: {status:?}");
                break;
            }
            Ok(None) => {
                if Instant::now() > deadline {
                    panic!("daemon did not exit within 5s of shutdown");
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => panic!("wait failed: {e}"),
        }
    }
}

#[test]
fn daemon_roundtrip_all_commands_under_timeout() {
    // Run the scenario on a dedicated thread so we can hard-fail on
    // the global TIMEOUT. The scenario spawns + reaps the daemon itself.
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let h = thread::spawn(move || {
        run_scenario();
        tx.send(()).ok();
    });

    match rx.recv_timeout(TIMEOUT) {
        Ok(()) => {
            h.join().expect("scenario thread panicked");
        }
        Err(_) => panic!(
            "daemon roundtrip exceeded {}s timeout — likely hang",
            TIMEOUT.as_secs()
        ),
    }
}
