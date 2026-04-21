//! NIMCP V2 — V1-protocol-compatible Unix-socket daemon.
//!
//! This crate provides a length-prefixed JSON RPC server over a Unix domain
//! socket that is byte-compatible with V1's `scripts/brain_daemon.py`.
//! The intent is that V1's curriculum driver (`scripts/immerse_athena.py`)
//! and the `BrainProxy` client in `scripts/brain_client.py` can talk to a
//! V2 daemon without modification.
//!
//! # Wire protocol
//!
//! Each frame is a 4-byte big-endian `u32` length prefix followed by a
//! UTF-8 JSON body. The maximum body size is 50 MB (matching V1).
//! Requests have a `"cmd"` field; responses are either
//! `{"ok": true, ...}` / command-specific success dicts, or
//! `{"error": "..."}` on failure.
//!
//! # Architecture
//!
//! [`commands::BrainBackend`] is the narrow trait every backend implements.
//! A [`commands::StubBackend`] in-tree satisfies the trait for testing and
//! development without any dependency on the V1 C library. A real
//! V1-bridged backend will land behind the `v1` feature flag (wired from
//! `src/main.rs`) once `nimcp-v1-bridge` is available.
//!
//! [`protocol`] exposes `read_frame` / `write_frame` used by the server
//! loop in `src/main.rs`.

#![forbid(unsafe_code)]

pub mod commands;
pub mod protocol;
pub mod server;

pub use commands::{BrainBackend, StubBackend, handle_request};
pub use protocol::{MAX_MESSAGE_BYTES, ProtocolError, read_frame, write_frame};
pub use server::{ServerHandle, serve};
