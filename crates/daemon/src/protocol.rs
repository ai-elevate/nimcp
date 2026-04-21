//! Length-prefixed JSON frame codec — byte-compatible with V1's
//! `scripts/brain_daemon.py` helpers `recv_msg` / `send_msg`.
//!
//! Frame layout:
//!
//! ```text
//! +---------+---------+---------+---------+----------------------------+
//! |   len[3]|   len[2]|   len[1]|   len[0]|  JSON body (UTF-8, len B)  |
//! +---------+---------+---------+---------+----------------------------+
//!  \----- big-endian u32 -----/
//! ```
//!
//! `len` is the byte length of the JSON body, **not** including the
//! 4-byte header. The V1 Python implementation rejects messages whose
//! header advertises > 50 MB ([`MAX_MESSAGE_BYTES`]); we match that cap.

use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Maximum JSON body size accepted by the frame decoder.
///
/// V1 uses `50 * 1024 * 1024`. Batch-training commands can push this
/// close to the limit; larger bodies are refused rather than truncated.
pub const MAX_MESSAGE_BYTES: usize = 50 * 1024 * 1024;

/// Errors returned by the frame codec.
#[derive(Debug, Error)]
pub enum ProtocolError {
    /// The peer closed the connection before a full frame arrived.
    #[error("peer closed connection")]
    PeerClosed,

    /// The advertised length exceeds [`MAX_MESSAGE_BYTES`].
    #[error("frame too large: {size} bytes (limit {limit})")]
    FrameTooLarge {
        /// Advertised body size.
        size: usize,
        /// Hard cap we enforce.
        limit: usize,
    },

    /// The frame body was not valid UTF-8 JSON.
    #[error("invalid JSON body: {0}")]
    InvalidJson(#[from] serde_json::Error),

    /// Underlying I/O failure (peer reset, write failure, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Read one length-prefixed JSON frame from `stream`.
///
/// Returns the decoded `serde_json::Value` on success. On clean EOF
/// before any bytes of the length prefix, returns [`ProtocolError::PeerClosed`]
/// — the server loop uses that as a signal to close the connection.
pub async fn read_frame<R>(stream: &mut R) -> Result<serde_json::Value, ProtocolError>
where
    R: AsyncReadExt + Unpin,
{
    // ---- header ----
    let mut hdr = [0u8; 4];
    read_exact_or_eof(stream, &mut hdr).await?;
    let len = u32::from_be_bytes(hdr) as usize;

    if len > MAX_MESSAGE_BYTES {
        return Err(ProtocolError::FrameTooLarge {
            size: len,
            limit: MAX_MESSAGE_BYTES,
        });
    }

    // ---- body ----
    // Zero-length bodies are unusual but legal; serde will reject them as
    // invalid JSON with a clear error.
    let mut body = vec![0u8; len];
    if len > 0 {
        read_exact_or_eof(stream, &mut body).await?;
    }

    let value = serde_json::from_slice(&body)?;
    Ok(value)
}

/// Write one length-prefixed JSON frame to `stream`.
///
/// Serializes `value` to UTF-8 JSON, prepends the 4-byte big-endian length,
/// and writes the whole thing in a single `write_all` call so partial
/// writes don't leave the stream in a half-framed state.
pub async fn write_frame<W>(stream: &mut W, value: &serde_json::Value) -> Result<(), ProtocolError>
where
    W: AsyncWriteExt + Unpin,
{
    let body = serde_json::to_vec(value)?;
    if body.len() > MAX_MESSAGE_BYTES {
        return Err(ProtocolError::FrameTooLarge {
            size: body.len(),
            limit: MAX_MESSAGE_BYTES,
        });
    }

    // Build a single buffer so the 4-byte header and the body are flushed
    // together; this keeps the write atomic from the peer's point of view.
    let mut frame = Vec::with_capacity(4 + body.len());
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(&body);

    stream.write_all(&frame).await?;
    stream.flush().await?;
    Ok(())
}

/// Like `AsyncReadExt::read_exact` but maps `UnexpectedEof` into a
/// semantic `PeerClosed` error. We need to distinguish "peer gracefully
/// closed" (normal end-of-stream) from "half-frame arrived then socket
/// died" — both are io::ErrorKind::UnexpectedEof at the read layer, but
/// the server loop treats them identically so we merge them here.
async fn read_exact_or_eof<R>(stream: &mut R, buf: &mut [u8]) -> Result<(), ProtocolError>
where
    R: AsyncReadExt + Unpin,
{
    match stream.read_exact(buf).await {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Err(ProtocolError::PeerClosed),
        Err(e) => Err(ProtocolError::Io(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::duplex;

    #[tokio::test]
    async fn round_trip_small_object() {
        let (mut a, mut b) = duplex(64 * 1024);
        let input = json!({"cmd": "ping", "seq": 42});
        let expected = input.clone();
        let writer = tokio::spawn(async move {
            write_frame(&mut a, &input).await.unwrap();
        });
        let got = read_frame(&mut b).await.unwrap();
        writer.await.unwrap();
        assert_eq!(got, expected);
    }

    #[tokio::test]
    async fn round_trip_nested_and_arrays() {
        let (mut a, mut b) = duplex(64 * 1024);
        let input = json!({
            "cmd": "learn_vector",
            "features": [0.1, 0.2, 0.3, -0.4],
            "target": [1.0, 0.0],
            "meta": {"label": "cat", "confidence": 0.9}
        });
        let expected = input.clone();
        tokio::spawn(async move {
            write_frame(&mut a, &input).await.unwrap();
        });
        let got = read_frame(&mut b).await.unwrap();
        assert_eq!(got, expected);
    }

    #[tokio::test]
    async fn header_is_big_endian() {
        let (mut a, mut b) = duplex(64 * 1024);
        // Body = "hi" as a bare string => 4 bytes including the quotes.
        let input = json!("hi");
        tokio::spawn(async move {
            write_frame(&mut a, &input).await.unwrap();
        });
        let mut hdr = [0u8; 4];
        b.read_exact(&mut hdr).await.unwrap();
        // Length of `"hi"` as JSON == 4 ("\"hi\"").
        assert_eq!(u32::from_be_bytes(hdr), 4);
        assert_eq!(hdr, [0, 0, 0, 4]);
    }

    #[tokio::test]
    async fn rejects_oversize_frame() {
        let (mut a, mut b) = duplex(64);
        // Craft a header that advertises > MAX_MESSAGE_BYTES without
        // actually sending that much data — decoder must reject on the
        // header alone.
        let bogus_len: u32 = (MAX_MESSAGE_BYTES as u32).saturating_add(1);
        tokio::spawn(async move {
            a.write_all(&bogus_len.to_be_bytes()).await.unwrap();
            a.flush().await.unwrap();
            // Hold the write side open so the reader doesn't flip to EOF
            // before it sees the oversize header.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            drop(a);
        });
        let err = read_frame(&mut b).await.unwrap_err();
        assert!(matches!(err, ProtocolError::FrameTooLarge { .. }));
    }

    #[tokio::test]
    async fn clean_eof_reports_peer_closed() {
        let (a, mut b) = duplex(64);
        drop(a);
        let err = read_frame(&mut b).await.unwrap_err();
        assert!(matches!(err, ProtocolError::PeerClosed));
    }

    #[tokio::test]
    async fn invalid_json_body_is_an_error() {
        let (mut a, mut b) = duplex(64);
        // Length = 3, body = `{{{` — not valid JSON.
        tokio::spawn(async move {
            a.write_all(&3u32.to_be_bytes()).await.unwrap();
            a.write_all(b"{{{").await.unwrap();
            a.flush().await.unwrap();
        });
        let err = read_frame(&mut b).await.unwrap_err();
        assert!(matches!(err, ProtocolError::InvalidJson(_)));
    }
}
