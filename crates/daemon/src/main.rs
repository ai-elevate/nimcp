//! `nimcp-daemon` — Unix-socket RPC server that speaks V1's protocol.
//!
//! Run this instead of V1's `scripts/brain_daemon.py` to let V1's
//! curriculum driver talk to a V2 backend. Today the only backend is
//! [`nimcp_daemon::StubBackend`], which returns trivial values — real
//! backends land behind cargo features as they come online.
//!
//! ```bash
//! cargo run -p nimcp-daemon --no-default-features -- \
//!     --socket-path /tmp/brain.sock --backend stub --log-level info
//! ```

#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use eyre::{Result, WrapErr};
use tracing::{info, warn};

use nimcp_daemon::{BrainBackend, StubBackend, serve};

/// Backend selector. Real (V1-bridged / native V2) backends will live
/// behind cargo features; for now only the stub is linked in.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendKind {
    /// In-memory stub — trivial responses, used for dev + tests.
    Stub,
    /// V1 C brain via `nimcp-v1-bridge`. Requires the `v1` feature.
    V1,
}

/// Command-line arguments.
#[derive(Debug, Parser)]
#[command(name = "nimcp-daemon", about = "NIMCP V2 Unix-socket daemon", version)]
struct Cli {
    /// Path for the Unix domain socket. V1 uses `/var/run/athena/brain.sock`;
    /// for non-root runs, point this somewhere under `/tmp` or `$XDG_RUNTIME_DIR`.
    #[arg(long, default_value = "/var/run/athena/brain.sock")]
    socket_path: PathBuf,

    /// Backend selector.
    #[arg(long, value_enum, default_value_t = BackendKind::Stub)]
    backend: BackendKind,

    /// Log level (`error` | `warn` | `info` | `debug` | `trace`). Env var
    /// `RUST_LOG` overrides this if set.
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    info!(
        socket = %cli.socket_path.display(),
        backend = ?cli.backend,
        "starting nimcp-daemon"
    );

    let backend = build_backend(cli.backend)?;

    // Spawn a SIGTERM / SIGINT handler that triggers a graceful shutdown.
    // The server loop itself honors `ServerHandle::shutdown`; we can't get
    // a handle back until `serve` returns, so instead we race the server
    // future with a signal future and drop the listener on signal.
    //
    // `tokio::select!` is the right primitive here because we want
    // whichever finishes first to cancel the other.
    let server = serve(cli.socket_path.clone(), backend);

    tokio::select! {
        res = server => {
            res.wrap_err("server loop failed")?;
        }
        () = wait_for_shutdown_signal() => {
            info!("shutdown signal received — exiting");
            // Best-effort cleanup: if the server was still running, its
            // tokio task is aborted when `main` returns. The socket file
            // may linger in that case; log a warning so it's visible.
            if cli.socket_path.exists() {
                warn!(
                    path = %cli.socket_path.display(),
                    "signal-driven exit; socket file may remain until next startup"
                );
            }
        }
    }

    Ok(())
}

fn init_tracing(level: &str) {
    use tracing_subscriber::{EnvFilter, fmt};
    // RUST_LOG wins if set; otherwise use the CLI flag.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));
    fmt().with_env_filter(filter).init();
}

fn build_backend(kind: BackendKind) -> Result<Box<dyn BrainBackend>> {
    match kind {
        BackendKind::Stub => Ok(Box::<StubBackend>::default()),
        BackendKind::V1 => {
            #[cfg(feature = "v1")]
            {
                // TODO: wire nimcp-v1-bridge here once it exists. Until
                // then, the `v1` feature compiles but this arm just
                // returns a stub so integration tests can still run.
                tracing::warn!("v1 feature enabled but bridge not implemented; using stub");
                Ok(Box::<StubBackend>::default())
            }
            #[cfg(not(feature = "v1"))]
            {
                eyre::bail!(
                    "backend=v1 requires building with `--features v1` (nimcp-v1-bridge not yet wired)"
                );
            }
        }
    }
}

async fn wait_for_shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};

    // If either signal stream fails to install, we still honor the other.
    // If both fail (vanishingly rare — PID 1 in a locked-down container,
    // maybe) we just park forever; the tokio runtime will exit once the
    // server future does.
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => Some(s),
        Err(e) => {
            warn!(error = %e, "couldn't install SIGTERM handler");
            None
        }
    };
    let mut int = match signal(SignalKind::interrupt()) {
        Ok(s) => Some(s),
        Err(e) => {
            warn!(error = %e, "couldn't install SIGINT handler");
            None
        }
    };

    match (term.as_mut(), int.as_mut()) {
        (Some(t), Some(i)) => {
            tokio::select! {
                _ = t.recv() => {},
                _ = i.recv() => {},
            }
        }
        (Some(t), None) => {
            let _ = t.recv().await;
        }
        (None, Some(i)) => {
            let _ = i.recv().await;
        }
        (None, None) => {
            std::future::pending::<()>().await;
        }
    }
}
