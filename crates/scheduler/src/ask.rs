//! Typed request/response (ask) pattern.
//!
//! The raw [`Actor`](crate::Actor) trait is fire-and-forget: `handle()`
//! returns `Result<()>` with no reply channel. For the common "send a
//! request and await a typed reply" pattern we layer a generic envelope
//! on top of it.
//!
//! Usage:
//! 1. The actor's `Msg` type is [`AskEnvelope<Req, Reply>`].
//! 2. Inside `handle()`, the actor destructures the envelope, computes the
//!    reply, and calls `envelope.reply(reply)`.
//! 3. Callers use [`AskExt::ask`](crate::AskExt::ask) on the handle to send
//!    a request and await the reply.
//!
//! The reply channel is a `tokio::sync::oneshot`. If the actor drops the
//! envelope without replying (e.g., a buggy handler) the receiver gets
//! [`AskError::ActorDropped`] instead of hanging forever.

use tokio::sync::oneshot;

/// Error returned when an `ask()` call could not get a reply.
#[derive(Debug, thiserror::Error)]
pub enum AskError {
    /// The mailbox was closed before we could deliver the request.
    #[error("mailbox closed before ask could be delivered")]
    MailboxClosed,
    /// The actor received the request but dropped the reply channel without
    /// answering. Typically a handler bug.
    #[error("actor dropped the reply channel without responding")]
    ActorDropped,
}

/// Envelope carrying a typed request plus a one-shot reply channel.
///
/// Actors that want to support `ask()` use this as their `Msg`. The actor
/// takes ownership of the envelope, reads `req`, computes a `Reply`, and
/// calls [`AskEnvelope::reply`].
#[derive(Debug)]
pub struct AskEnvelope<Req, Reply> {
    /// The caller's request.
    pub req: Req,
    /// One-shot channel back to the caller. Consumed by `reply()`.
    reply_tx: oneshot::Sender<Reply>,
}

impl<Req, Reply> AskEnvelope<Req, Reply> {
    /// Build an envelope and its matching receiver. Normally constructed
    /// via [`AskExt::ask`](crate::AskExt::ask); exposed here so users who
    /// want to construct it manually (tests, custom routing) can.
    pub fn new(req: Req) -> (Self, oneshot::Receiver<Reply>) {
        let (tx, rx) = oneshot::channel();
        (Self { req, reply_tx: tx }, rx)
    }

    /// Send the reply back to the caller.
    ///
    /// Returns `Err(reply)` if the caller already dropped the receiver
    /// (cancelled its future). This is normal, not an error — we hand the
    /// value back so the actor can log it if interesting.
    pub fn reply(self, reply: Reply) -> Result<(), Reply> {
        self.reply_tx.send(reply)
    }

    /// Borrow the request without consuming the envelope.
    pub fn request(&self) -> &Req {
        &self.req
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn envelope_round_trips_reply() {
        let (env, rx) = AskEnvelope::<u32, u32>::new(42);
        assert_eq!(env.req, 42);
        env.reply(84).unwrap();
        assert_eq!(rx.await.unwrap(), 84);
    }

    #[tokio::test]
    async fn dropped_envelope_errors_on_receiver() {
        let (env, rx) = AskEnvelope::<u32, u32>::new(1);
        drop(env);
        assert!(rx.await.is_err());
    }
}
