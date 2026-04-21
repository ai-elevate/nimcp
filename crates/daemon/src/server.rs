//! Server loop: accept Unix-socket connections, dispatch frames, shut
//! down cleanly.
//!
//! The server is intentionally small:
//!
//! 1. Create the parent dir for `socket_path` and remove any stale file.
//! 2. Bind a `UnixListener`.
//! 3. In a loop, `accept()`; for every connection, spawn a task that
//!    reads frames and dispatches them.
//! 4. On shutdown (either a client sent `{"cmd": "shutdown"}` or an
//!    external signal fired), stop accepting, let in-flight tasks
//!    finish, and remove the socket file.
//!
//! Concurrency model: one `tokio::sync::Mutex<dyn BrainBackend>` shared
//! across all connections. The V1 backend is single-writer (a C brain
//! object is not thread-safe without the read-write lock V1 maintains),
//! and V2 backends will wrap an actor whose mailbox is single-consumer
//! — so sequential dispatch is the right default. It trades throughput
//! for simplicity; an optimization can come later once the rest of the
//! V2 pipeline is real.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinSet;
use tracing::{debug, error, info, warn};

use crate::commands::{BrainBackend, handle_request};
use crate::protocol::{ProtocolError, read_frame, write_frame};

/// Opaque handle returned by [`serve`].
///
/// Dropping the handle does not stop the server — call
/// [`ServerHandle::shutdown`] or send the process a SIGTERM.
#[derive(Debug, Clone)]
pub struct ServerHandle {
    shutdown: Arc<Notify>,
    socket_path: PathBuf,
}

impl ServerHandle {
    /// Request a graceful shutdown. The server stops accepting new
    /// connections, drains in-flight requests, and removes the socket
    /// file before [`serve`] returns.
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }

    /// Path the server is listening on.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

/// Errors bubbled out of [`serve`].
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// Failed to create / clean the socket file or its parent directory.
    #[error("socket setup at {path:?}: {source}")]
    SocketSetup {
        /// Path we tried to bind.
        path: PathBuf,
        /// Underlying io error.
        source: std::io::Error,
    },

    /// Failed to bind the `UnixListener`.
    #[error("bind {path:?}: {source}")]
    Bind {
        /// Path we tried to bind.
        path: PathBuf,
        /// Underlying io error.
        source: std::io::Error,
    },
}

/// Run the server until an external signal / `ServerHandle::shutdown` /
/// a `{"cmd":"shutdown"}` request arrives.
///
/// This function takes ownership of the backend so there's exactly one
/// instance across the server's lifetime. Wrapped in an `Arc<Mutex<..>>`
/// internally so connection tasks can dispatch concurrently at the TCP
/// layer while still serializing backend calls.
pub async fn serve(
    socket_path: PathBuf,
    backend: Box<dyn BrainBackend>,
) -> Result<(), ServerError> {
    prepare_socket_path(&socket_path)?;

    let listener = UnixListener::bind(&socket_path).map_err(|e| ServerError::Bind {
        path: socket_path.clone(),
        source: e,
    })?;
    info!(path = %socket_path.display(), "nimcp-daemon listening");

    let shutdown = Arc::new(Notify::new());
    let handle = ServerHandle {
        shutdown: shutdown.clone(),
        socket_path: socket_path.clone(),
    };

    let backend: Arc<Mutex<Box<dyn BrainBackend>>> = Arc::new(Mutex::new(backend));
    let connections = Arc::new(Mutex::new(JoinSet::new()));

    run_accept_loop(listener, backend, connections.clone(), handle).await;

    // Accept loop exited — drain in-flight connections so we don't yank
    // the socket out from under them.
    drain_connections(connections).await;

    // Best-effort cleanup of the socket file. If this fails, warn but
    // still return success: the server itself finished normally.
    if let Err(e) = std::fs::remove_file(&socket_path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            warn!(path = %socket_path.display(), error = %e,
                  "failed to remove socket file on shutdown");
        }
    }

    Ok(())
}

/// Create the parent dir if it doesn't exist, and remove any stale
/// socket file at `path`. Refuses to clobber something that is *not*
/// a socket — that's almost certainly an operator mistake.
fn prepare_socket_path(path: &Path) -> Result<(), ServerError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| ServerError::SocketSetup {
                path: path.to_path_buf(),
                source: e,
            })?;
        }
    }

    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            use std::os::unix::fs::FileTypeExt;
            if !meta.file_type().is_socket() {
                return Err(ServerError::SocketSetup {
                    path: path.to_path_buf(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        "path exists and is not a Unix socket — refusing to unlink",
                    ),
                });
            }
            std::fs::remove_file(path).map_err(|e| ServerError::SocketSetup {
                path: path.to_path_buf(),
                source: e,
            })?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Fresh path — nothing to clean up.
        }
        Err(e) => {
            return Err(ServerError::SocketSetup {
                path: path.to_path_buf(),
                source: e,
            });
        }
    }
    Ok(())
}

async fn run_accept_loop(
    listener: UnixListener,
    backend: Arc<Mutex<Box<dyn BrainBackend>>>,
    connections: Arc<Mutex<JoinSet<()>>>,
    handle: ServerHandle,
) {
    loop {
        tokio::select! {
            // Shutdown — either external (SIGTERM/.shutdown()) or internal
            // (a client sent `{"cmd":"shutdown"}`, which also fires the Notify).
            () = handle.shutdown.notified() => {
                info!("shutdown signal received, stopping accept loop");
                return;
            }
            res = listener.accept() => {
                match res {
                    Ok((stream, _addr)) => {
                        debug!("accepted new connection");
                        let backend = backend.clone();
                        let handle = handle.clone();
                        let mut set = connections.lock().await;
                        set.spawn(async move {
                            if let Err(e) = handle_connection(stream, backend, handle).await {
                                // Dropped connections are normal; log at debug.
                                debug!(error = %e, "connection ended");
                            }
                        });
                    }
                    Err(e) => {
                        // Per-error-class behavior: transient errors log
                        // and continue; any other error brings down the
                        // loop so the caller can decide to restart.
                        match e.kind() {
                            std::io::ErrorKind::ConnectionAborted
                            | std::io::ErrorKind::Interrupted => {
                                warn!(error = %e, "transient accept error, continuing");
                            }
                            _ => {
                                error!(error = %e, "fatal accept error, stopping loop");
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}

async fn drain_connections(connections: Arc<Mutex<JoinSet<()>>>) {
    let mut set = connections.lock().await;
    while let Some(joined) = set.join_next().await {
        if let Err(e) = joined {
            warn!(error = %e, "connection task panicked during drain");
        }
    }
}

/// Handle one connection until the peer closes it, the connection
/// goes idle long enough, or shutdown is requested.
///
/// V1's protocol: after a response, the server closes the socket unless
/// the request was `{"cmd": "keepalive"}`. We mirror that so V1's
/// `BrainProxy._send_once` (which opens a new socket per call) works
/// unchanged, and a future keep-alive client can still stay connected.
async fn handle_connection(
    stream: UnixStream,
    backend: Arc<Mutex<Box<dyn BrainBackend>>>,
    handle: ServerHandle,
) -> Result<(), ConnectionError> {
    let (reader, writer) = stream.into_split();
    let mut reader = tokio::io::BufReader::new(reader);
    let mut writer = tokio::io::BufWriter::new(writer);

    loop {
        let req = tokio::select! {
            () = handle.shutdown.notified() => {
                debug!("shutdown during read — closing connection");
                return Ok(());
            }
            frame = read_frame(&mut reader) => match frame {
                Ok(value) => value,
                Err(ProtocolError::PeerClosed) => return Ok(()),
                Err(e) => return Err(ConnectionError::Protocol(e)),
            }
        };

        // Stash whether this is a shutdown/keepalive before we move `req`.
        let cmd = req.get("cmd").and_then(|v| v.as_str()).map(str::to_owned);
        let is_shutdown = cmd.as_deref() == Some("shutdown");
        let is_keepalive = cmd.as_deref() == Some("keepalive");

        // Keepalive is a cheap noop — don't touch the backend.
        let resp = if is_keepalive {
            serde_json::json!({"ok": true})
        } else {
            let mut guard = backend.lock().await;
            handle_request(guard.as_mut(), req).await
        };

        write_frame(&mut writer, &resp)
            .await
            .map_err(ConnectionError::Protocol)?;

        if is_shutdown {
            // Tell the accept loop to wind down *after* we flushed the
            // response. The client sees its `{"ok": true}` and we then
            // stop accepting.
            handle.shutdown.notify_waiters();
            return Ok(());
        }

        if !is_keepalive {
            // V1 semantics: one request, one response, close.
            return Ok(());
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum ConnectionError {
    #[error("protocol: {0}")]
    Protocol(#[from] ProtocolError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::StubBackend;
    use tempfile::tempdir;
    use tokio::io::AsyncReadExt;

    /// Build a unique socket path inside a temp dir. Unix sockets have a
    /// hard path-length limit (~108 bytes on Linux); keep names short.
    fn sock_path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("d.sock")
    }

    async fn call(socket: &Path, req: serde_json::Value) -> serde_json::Value {
        let mut s = UnixStream::connect(socket).await.expect("connect");
        write_frame(&mut s, &req).await.expect("write");
        read_frame(&mut s).await.expect("read")
    }

    #[tokio::test]
    async fn ping_round_trip() {
        let dir = tempdir().unwrap();
        let path = sock_path(&dir);
        let path_for_server = path.clone();
        let server = tokio::spawn(async move {
            serve(path_for_server, Box::<StubBackend>::default())
                .await
                .unwrap();
        });

        wait_for_socket(&path).await;
        let resp = call(&path, serde_json::json!({"cmd": "ping"})).await;
        assert_eq!(resp, serde_json::json!({"ok": true}));

        // Send shutdown so the server task returns cleanly.
        let _ = call(&path, serde_json::json!({"cmd": "shutdown"})).await;
        server.await.unwrap();
    }

    #[tokio::test]
    async fn refuses_to_clobber_regular_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("not-a-socket");
        std::fs::write(&path, b"hi").unwrap();
        let err = serve(path.clone(), Box::<StubBackend>::default())
            .await
            .expect_err("should refuse to bind");
        match err {
            ServerError::SocketSetup { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    async fn wait_for_socket(path: &Path) {
        use std::time::Duration;
        for _ in 0..100 {
            if path.exists() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("socket did not appear at {}", path.display());
    }

    async fn read_until_eof(stream: &mut UnixStream) -> Vec<u8> {
        let mut buf = Vec::new();
        let _ = stream.read_to_end(&mut buf).await;
        buf
    }

    #[tokio::test]
    async fn connection_closes_after_single_response() {
        let dir = tempdir().unwrap();
        let path = sock_path(&dir);
        let path_for_server = path.clone();
        let server = tokio::spawn(async move {
            serve(path_for_server, Box::<StubBackend>::default())
                .await
                .unwrap();
        });
        wait_for_socket(&path).await;

        let mut s = UnixStream::connect(&path).await.unwrap();
        write_frame(&mut s, &serde_json::json!({"cmd": "ping"}))
            .await
            .unwrap();
        let _ = read_frame(&mut s).await.unwrap();

        // After the response the server should close the connection.
        let leftovers = read_until_eof(&mut s).await;
        assert!(leftovers.is_empty());

        // Cleanly shut the server down.
        let _ = call(&path, serde_json::json!({"cmd": "shutdown"})).await;
        server.await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_command_unlinks_socket_file() {
        let dir = tempdir().unwrap();
        let path = sock_path(&dir);
        let path_for_server = path.clone();
        let server = tokio::spawn(async move {
            serve(path_for_server, Box::<StubBackend>::default())
                .await
                .unwrap();
        });
        wait_for_socket(&path).await;
        assert!(path.exists(), "socket should exist while server is up");

        let _ = call(&path, serde_json::json!({"cmd": "shutdown"})).await;
        server.await.unwrap();

        assert!(
            !path.exists(),
            "socket file should be removed after shutdown"
        );
    }
}
