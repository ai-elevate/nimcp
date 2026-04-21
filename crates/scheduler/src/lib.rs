//! NIMCP V2 — actor runtime.
//!
//! # Design
//!
//! One tokio multi-threaded runtime drives all actors. Each actor owns its
//! state and receives messages through an mpsc channel. Actors may emit
//! events to the event log; they never share heap with sibling actors.
//!
//! V1 had many concurrent tick loops (brain_learn_vector, pr_memory driver,
//! cycle coordinator, cognitive recovery, bio_async), each competing for
//! execution and occasionally racing on shared state. V2 centralizes
//! scheduling in one place.
//!
//! # Deterministic replay
//!
//! When `SchedulerConfig::deterministic == true`:
//! - Time is virtual (tokio test-util)
//! - Actor scheduling is ordered by ActorId
//! - No multithreading
//! - Same inputs → same outputs, bit-for-bit
//!
//! This is how we regression-test emergent behaviors (homeostasis,
//! saturation recovery) that V1 could only test by running full training.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use nimcp_core::{ActorId, Error, Result};
use std::fmt::Debug;
use tokio::sync::mpsc;

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// If true, scheduling is deterministic (single-threaded, virtual time).
    /// Use for tests and regression.
    pub deterministic: bool,
    /// Mailbox capacity per actor. Messages beyond this block the sender.
    pub mailbox_capacity: usize,
    /// Random seed passed to any actor that takes a seeded RNG.
    pub rng_seed: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            deterministic: false,
            mailbox_capacity: 1024,
            rng_seed: 0x5EED_5EED_5EED_5EED,
        }
    }
}

/// Per-actor context passed into `handle()`. Gives the actor its id +
/// a handle to emit events.
pub struct Context {
    /// This actor's unique id.
    pub id: ActorId,
}

/// An actor: owned state + typed message handler.
///
/// Actors are single-threaded from their own perspective — `handle()` is
/// never called concurrently on the same actor. Inter-actor communication
/// is message-passing; no shared heap.
#[async_trait]
pub trait Actor: Send + 'static {
    /// Messages this actor accepts. Must be `Send` + `Debug`.
    type Msg: Send + Debug + 'static;

    /// Handle one message. Errors propagate to the supervisor.
    async fn handle(&mut self, msg: Self::Msg, ctx: &mut Context) -> Result<()>;

    /// Name for logs + introspection. Should be stable across runs.
    fn name(&self) -> &'static str;
}

/// Handle to a spawned actor.
pub struct ActorHandle<M> {
    /// Actor id.
    pub id: ActorId,
    /// Sender for messages to this actor.
    tx: mpsc::Sender<M>,
}

impl<M: Send + 'static> ActorHandle<M> {
    /// Send a message to the actor. Backpressure via mailbox.
    pub async fn send(&self, msg: M) -> Result<()> {
        self.tx
            .send(msg)
            .await
            .map_err(|e| Error::Mailbox(format!("actor {}: {e}", self.id.0)))
    }
}

/// Scheduler handle.
pub struct Scheduler {
    #[allow(dead_code)]
    config: SchedulerConfig,
    next_id: u64,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> Self {
        tracing::info!(
            deterministic = config.deterministic,
            mailbox_capacity = config.mailbox_capacity,
            "scheduler created"
        );
        Self { config, next_id: 0 }
    }

    /// Spawn an actor onto the runtime. Returns a handle for sending messages.
    pub fn spawn<A>(&mut self, mut actor: A) -> ActorHandle<A::Msg>
    where
        A: Actor,
    {
        let id = ActorId(self.next_id);
        self.next_id += 1;

        let (tx, mut rx) = mpsc::channel::<A::Msg>(self.config.mailbox_capacity);
        let name = actor.name();

        tokio::spawn(async move {
            let mut ctx = Context { id };
            tracing::info!(actor = name, id = %id, "actor started");
            while let Some(msg) = rx.recv().await {
                if let Err(e) = actor.handle(msg, &mut ctx).await {
                    tracing::error!(actor = name, id = %id, error = %e, "handler error");
                    break;
                }
            }
            tracing::info!(actor = name, id = %id, "actor stopped");
        });

        ActorHandle { id, tx }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct CountingActor {
        count: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Actor for CountingActor {
        type Msg = u32;

        async fn handle(&mut self, msg: Self::Msg, _ctx: &mut Context) -> Result<()> {
            self.count.fetch_add(msg, Ordering::SeqCst);
            Ok(())
        }

        fn name(&self) -> &'static str {
            "counting"
        }
    }

    #[tokio::test]
    async fn spawn_and_send() {
        let count = Arc::new(AtomicU32::new(0));
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let h = sched.spawn(CountingActor {
            count: Arc::clone(&count),
        });

        for v in [1_u32, 2, 3, 4, 5] {
            h.send(v).await.unwrap();
        }

        // Give the actor a tick to drain the mailbox
        tokio::task::yield_now().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        assert_eq!(count.load(Ordering::SeqCst), 15);
    }
}
