//! NIMCP V2 — actor runtime.
//!
//! # Design
//!
//! One tokio runtime drives all actors. Each actor owns its state and
//! receives messages through an mpsc channel. Actors may emit events to
//! the event log; they never share heap with sibling actors.
//!
//! V1 had many concurrent tick loops (brain_learn_vector, pr_memory
//! driver, cycle coordinator, cognitive recovery, bio_async), each
//! competing for execution and occasionally racing on shared state. V2
//! centralizes scheduling in one place.
//!
//! # Features
//!
//! - **Supervision** — when `handle()` returns `Err` or panics, the
//!   scheduler restarts the actor, up to [`SchedulerConfig::max_restarts`]
//!   within [`SchedulerConfig::restart_window`]. Beyond that, the actor is
//!   declared dead and its mailbox drained.
//! - **Graceful shutdown** — [`Scheduler::shutdown`] closes every mailbox,
//!   waits for in-flight handlers to finish (bounded by a timeout), and
//!   returns only when all actor tasks have exited.
//! - **Deterministic replay** — with [`SchedulerConfig::deterministic`]
//!   true, the caller opts into a single-threaded runtime and virtual time
//!   ([`tokio::time::pause`]). Within a run, message delivery order
//!   per-actor is the send order (mpsc is FIFO); same seed + same message
//!   sequence produces the same observable state bit-for-bit.
//! - **Typed ask pattern** — [`AskExt::ask`] round-trips a request to an
//!   actor and awaits its reply via a oneshot channel.
//! - **Broadcast topics** — [`Scheduler::topic`] hands out
//!   [`TopicHandle`]s that publish a value to every subscriber, built on
//!   `tokio::sync::broadcast`.
//! - **Metrics** — [`Scheduler::metrics`] snapshots per-actor mailbox
//!   depth, messages handled, restart count, and a coarse handler-duration
//!   histogram.
//!
//! # Deterministic replay
//!
//! When `SchedulerConfig::deterministic == true`:
//! - Callers drive the scheduler on a single-threaded tokio runtime (see
//!   [`Scheduler::build_deterministic_runtime`]).
//! - [`tokio::time::pause`] gives you virtual time; advance it via
//!   [`tokio::time::advance`].
//! - Across actors, no implicit ordering guarantee — tests that need it
//!   must serialize their sends via a single sender actor or via
//!   [`AskExt::ask`], which naturally orders.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use nimcp_core::{ActorId, Error, Result};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, broadcast, mpsc};

mod ask;
mod metrics;

pub use ask::{AskEnvelope, AskError};
pub use metrics::{ActorMetrics, HISTOGRAM_BUCKETS_US};

use metrics::MetricsCell;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the scheduler.
///
/// `Default` gives sensible production values:
/// - non-deterministic (multi-threaded runtime),
/// - mailbox capacity 1024,
/// - 5 restarts per 60s window,
/// - 5s shutdown timeout.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// If true, scheduling is deterministic (single-threaded, virtual time).
    /// Intended for tests and regression harnesses.
    pub deterministic: bool,
    /// Mailbox capacity per actor. Senders block (async) when full.
    pub mailbox_capacity: usize,
    /// Random seed surfaced to actors that need one. The scheduler itself
    /// does not randomise anything.
    pub rng_seed: u64,
    /// Max number of restarts the supervisor will attempt within
    /// `restart_window` before declaring an actor dead.
    pub max_restarts: u32,
    /// Rolling window for counting restarts.
    pub restart_window: Duration,
    /// Upper bound on how long [`Scheduler::shutdown`] waits for in-flight
    /// handlers before returning. Handlers past this point are abandoned
    /// (their task is dropped).
    pub shutdown_timeout: Duration,
    /// Capacity of every broadcast topic. Subscribers that lag beyond this
    /// receive [`broadcast::error::RecvError::Lagged`] on their next
    /// `recv`.
    pub topic_capacity: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            deterministic: false,
            mailbox_capacity: 1024,
            rng_seed: 0x5EED_5EED_5EED_5EED,
            max_restarts: 5,
            restart_window: Duration::from_secs(60),
            shutdown_timeout: Duration::from_secs(5),
            topic_capacity: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// Context + Actor trait (trait kept compatible with the existing stub)
// ---------------------------------------------------------------------------

/// Per-actor context passed into `handle()`. Gives the actor its id,
/// the configured RNG seed, and a cancellation flag it can check for
/// cooperative shutdown.
pub struct Context {
    /// This actor's unique id.
    pub id: ActorId,
    /// Seed derived from [`SchedulerConfig::rng_seed`] plus the actor id.
    /// Stable across restarts.
    pub rng_seed: u64,
    /// Flipped to `true` when [`Scheduler::shutdown`] is called. Long-
    /// running handlers should poll this to exit early.
    shutdown: Arc<AtomicBool>,
}

impl Context {
    /// Returns `true` once the scheduler has requested shutdown. Use this
    /// to break out of long-running loops inside a handler.
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }
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

// ---------------------------------------------------------------------------
// ActorHandle
// ---------------------------------------------------------------------------

/// Handle to a spawned actor. Clone cheaply — shares the underlying
/// channel.
pub struct ActorHandle<M> {
    /// Actor id.
    pub id: ActorId,
    /// Sender for messages to this actor.
    tx: mpsc::Sender<M>,
    /// Shared mailbox-depth counter (kept in sync with the metrics cell).
    mailbox_depth: Arc<std::sync::atomic::AtomicU64>,
}

impl<M> Clone for ActorHandle<M> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            tx: self.tx.clone(),
            mailbox_depth: Arc::clone(&self.mailbox_depth),
        }
    }
}

impl<M: Send + 'static> ActorHandle<M> {
    /// Send a message to the actor. Async — back-pressures when the mailbox
    /// is full.
    pub async fn send(&self, msg: M) -> Result<()> {
        self.mailbox_depth.fetch_add(1, Ordering::Relaxed);
        let result = self
            .tx
            .send(msg)
            .await
            .map_err(|e| Error::Mailbox(format!("actor {}: {e}", self.id.0)));
        if result.is_err() {
            self.mailbox_depth.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    /// Try to send without blocking. Fails if the mailbox is full or closed.
    pub fn try_send(&self, msg: M) -> Result<()> {
        match self.tx.try_send(msg) {
            Ok(()) => {
                self.mailbox_depth.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => Err(Error::Mailbox(format!("actor {}: {e}", self.id.0))),
        }
    }

    /// `true` iff the mailbox has not been closed.
    pub fn is_alive(&self) -> bool {
        !self.tx.is_closed()
    }
}

/// Extension trait providing the typed ask pattern on handles whose
/// message type is an [`AskEnvelope`].
#[async_trait]
pub trait AskExt<Req, Reply>
where
    Req: Send + 'static,
    Reply: Send + 'static,
{
    /// Send `req` and await the actor's reply.
    async fn ask(&self, req: Req) -> std::result::Result<Reply, AskError>;
}

#[async_trait]
impl<Req, Reply> AskExt<Req, Reply> for ActorHandle<AskEnvelope<Req, Reply>>
where
    Req: Send + 'static,
    Reply: Send + 'static,
{
    async fn ask(&self, req: Req) -> std::result::Result<Reply, AskError> {
        let (env, rx) = AskEnvelope::new(req);
        self.send(env).await.map_err(|_| AskError::MailboxClosed)?;
        rx.await.map_err(|_| AskError::ActorDropped)
    }
}

// ---------------------------------------------------------------------------
// Broadcast topics
// ---------------------------------------------------------------------------

/// A typed broadcast topic. Every subscriber receives each publish.
///
/// Built on `tokio::sync::broadcast` — subscribers that fall further behind
/// than [`SchedulerConfig::topic_capacity`] see their next `recv` return
/// `Err(Lagged(n))`. That is a subscriber bug, not a scheduler bug.
pub struct TopicHandle<M: Clone + Send + 'static> {
    tx: broadcast::Sender<M>,
}

impl<M: Clone + Send + 'static> Clone for TopicHandle<M> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<M: Clone + Send + 'static> TopicHandle<M> {
    /// Publish `msg` to every current subscriber. Returns the number of
    /// live subscribers that saw it, or `0` if there are none.
    pub fn publish(&self, msg: M) -> usize {
        // tokio's broadcast::Sender::send returns Err only when there are
        // no active receivers; that is normal here.
        self.tx.send(msg).unwrap_or(0)
    }

    /// Subscribe to this topic. Returns a receiver the caller can drain
    /// from inside an actor (or directly in a test).
    pub fn subscribe(&self) -> broadcast::Receiver<M> {
        self.tx.subscribe()
    }

    /// Current subscriber count. Useful for tests; don't use for load
    /// decisions — it races.
    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

// ---------------------------------------------------------------------------
// Internal supervisor state
// ---------------------------------------------------------------------------

/// Everything the scheduler needs to remember about an actor so it can
/// shut it down, inspect its metrics, and restart it if it dies.
struct ActorRecord {
    id: ActorId,
    name: &'static str,
    /// The actor loop task.
    join: tokio::task::JoinHandle<()>,
    /// Sentinel sender whose drop closes the mailbox from the scheduler
    /// side. `Option` so `shutdown()` can `take()` it.
    close_sentinel: Option<mpsc::Sender<()>>,
    /// Per-actor metrics (shared with the actor loop).
    metrics: MetricsCell,
}

/// State shared between the scheduler and every actor loop.
struct SharedState {
    shutdown: Arc<AtomicBool>,
    config: SchedulerConfig,
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// The scheduler: owns all actor tasks, exposes spawn/metrics/shutdown.
pub struct Scheduler {
    config: SchedulerConfig,
    next_id: u64,
    records: HashMap<ActorId, ActorRecord>,
    shared: Arc<SharedState>,
    /// Type-erased topic map. Actor code reaches for a concrete
    /// `TopicHandle<M>` via [`Scheduler::topic`], keyed by
    /// `(topic_name, TypeId<M>)`.
    topics: HashMap<(String, std::any::TypeId), Box<dyn std::any::Any + Send + Sync>>,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> Self {
        tracing::info!(
            deterministic = config.deterministic,
            mailbox_capacity = config.mailbox_capacity,
            max_restarts = config.max_restarts,
            "scheduler created"
        );
        let shared = Arc::new(SharedState {
            shutdown: Arc::new(AtomicBool::new(false)),
            config: config.clone(),
        });
        Self {
            config,
            next_id: 0,
            records: HashMap::new(),
            shared,
            topics: HashMap::new(),
        }
    }

    /// Build a single-threaded tokio runtime configured for deterministic
    /// replay. Callers use this instead of `#[tokio::main]` when they need
    /// virtual-time + single-worker semantics.
    ///
    /// # Example
    /// ```no_run
    /// use nimcp_scheduler::{Scheduler, SchedulerConfig};
    /// let rt = Scheduler::build_deterministic_runtime();
    /// rt.block_on(async {
    ///     let mut sched = Scheduler::new(SchedulerConfig {
    ///         deterministic: true, ..Default::default()
    ///     });
    ///     sched.shutdown().await;
    /// });
    /// ```
    pub fn build_deterministic_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .start_paused(true)
            .build()
            .expect("single-threaded deterministic runtime should always build")
    }

    /// Spawn an actor by value. If the actor panics or returns `Err`, the
    /// supervisor logs the error and declares the actor dead (no restart
    /// possible — we don't have a factory). For supervised-with-restart,
    /// use [`Scheduler::spawn_supervised`].
    pub fn spawn<A>(&mut self, actor: A) -> ActorHandle<A::Msg>
    where
        A: Actor,
    {
        // One-shot factory: first call returns the provided actor, any
        // subsequent call (i.e. an attempted restart) is an error.
        let mut slot = Some(actor);
        self.spawn_supervised(move || {
            slot.take().expect(
                "spawn() actor consumed — use spawn_supervised() to enable restart",
            )
        })
    }

    /// Spawn an actor under the supervisor. `factory` is called once to
    /// construct the initial instance, and again on every restart. The
    /// same [`ActorId`] and mailbox persist across restarts; messages
    /// queued before a crash survive and are delivered to the new
    /// incarnation.
    pub fn spawn_supervised<A, F>(&mut self, mut factory: F) -> ActorHandle<A::Msg>
    where
        A: Actor,
        F: FnMut() -> A + Send + 'static,
    {
        let id = ActorId(self.next_id);
        self.next_id += 1;

        let (tx, rx) = mpsc::channel::<A::Msg>(self.config.mailbox_capacity);

        // Close-sentinel channel: when the scheduler drops its sender,
        // the actor loop's select sees rx close and enters drain mode.
        let (close_tx, close_rx) = mpsc::channel::<()>(1);

        // Construct the first incarnation. We do this here (rather than
        // inside the spawned task) so we can pull `name()` for the record
        // synchronously.
        let first_actor = factory();
        let name = first_actor.name();

        let metrics = MetricsCell::new(name);
        let handle = ActorHandle {
            id,
            tx,
            mailbox_depth: Arc::clone(&metrics.mailbox_depth),
        };

        let shared = Arc::clone(&self.shared);
        let metrics_for_loop = metrics.clone();

        let join = tokio::spawn(supervise_loop(
            id,
            name,
            first_actor,
            factory,
            rx,
            close_rx,
            shared,
            metrics_for_loop,
        ));

        self.records.insert(
            id,
            ActorRecord {
                id,
                name,
                join,
                close_sentinel: Some(close_tx),
                metrics,
            },
        );

        tracing::info!(actor = name, id = %id, "actor spawned");
        handle
    }

    /// Get or create a typed broadcast topic keyed by `name`. The message
    /// type `M` is part of the key — two topics with the same name but
    /// different types are distinct.
    pub fn topic<M: Clone + Send + 'static>(&mut self, name: &str) -> TopicHandle<M> {
        let key = (name.to_string(), std::any::TypeId::of::<M>());
        if let Some(existing) = self.topics.get(&key)
            && let Some(h) = existing.downcast_ref::<TopicHandle<M>>()
        {
            return h.clone();
        }
        let (tx, _rx) = broadcast::channel::<M>(self.config.topic_capacity);
        let handle = TopicHandle { tx };
        self.topics.insert(key, Box::new(handle.clone()));
        handle
    }

    /// Publish `msg` to a topic by name. Equivalent to
    /// `self.topic::<M>(name).publish(msg)`.
    pub fn broadcast_topic<M>(&mut self, name: &str, msg: M) -> usize
    where
        M: Clone + Send + 'static,
    {
        self.topic::<M>(name).publish(msg)
    }

    /// Take a snapshot of every actor's metrics. Ordered by ActorId.
    pub fn metrics(&self) -> Vec<ActorMetrics> {
        let mut records: Vec<_> = self.records.values().collect();
        records.sort_by_key(|r| r.id.0);
        records.iter().map(|r| r.metrics.snapshot()).collect()
    }

    /// Metrics for one actor by id, or `None` if the actor has been
    /// shut down.
    pub fn actor_metrics(&self, id: ActorId) -> Option<ActorMetrics> {
        self.records.get(&id).map(|r| r.metrics.snapshot())
    }

    /// Number of live actors.
    pub fn actor_count(&self) -> usize {
        self.records.len()
    }

    /// Gracefully stop every actor.
    ///
    /// 1. Sets the shared shutdown flag so long-running handlers can
    ///    voluntarily exit via `ctx.is_shutting_down()`.
    /// 2. Drops every mailbox sender held by the scheduler. The actor
    ///    loop's select sees its receiver close; it finishes any in-flight
    ///    message, drains the queue, and exits.
    /// 3. Awaits every actor's `JoinHandle`, bounded by
    ///    [`SchedulerConfig::shutdown_timeout`]. If a handler wedges past
    ///    that, we log and move on; the task will be dropped at scheduler
    ///    drop time.
    pub async fn shutdown(&mut self) {
        tracing::info!("scheduler shutdown requested");
        self.shared.shutdown.store(true, Ordering::SeqCst);

        // Close every mailbox by dropping our retained close-sentinel.
        // The mpsc sender the user holds is separate; we want to drain
        // already-queued messages, so we do NOT close rx from this side
        // directly. The sentinel exists so the actor loop can observe a
        // shutdown intent even if the user's handles are all cloned and
        // leaked somewhere.
        for record in self.records.values_mut() {
            record.close_sentinel.take();
        }

        // Drain the records map so we own every ActorRecord by value.
        let records: Vec<_> = self.records.drain().map(|(_, r)| r).collect();
        let timeout = self.shared.config.shutdown_timeout;

        for record in records {
            match tokio::time::timeout(timeout, record.join).await {
                Ok(Ok(())) => {
                    tracing::info!(
                        actor = record.name,
                        id = %record.id,
                        "actor stopped cleanly"
                    );
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        actor = record.name,
                        id = %record.id,
                        error = %e,
                        "actor task ended with join error"
                    );
                }
                Err(_elapsed) => {
                    tracing::error!(
                        actor = record.name,
                        id = %record.id,
                        "actor did not stop within shutdown_timeout — abandoning"
                    );
                }
            }
        }

        tracing::info!("scheduler shutdown complete");
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        // If the user forgot to call shutdown(), at least drop the
        // close-sentinels and flag shutdown so loops exit. We cannot
        // await JoinHandles from Drop.
        self.shared.shutdown.store(true, Ordering::SeqCst);
        for record in self.records.values_mut() {
            record.close_sentinel.take();
        }
    }
}

// ---------------------------------------------------------------------------
// The supervisor loop.
// ---------------------------------------------------------------------------

/// Spawned once per actor. Runs the message-processing loop, catches
/// `handle()` errors and panics, and restarts the actor within the
/// configured budget.
#[allow(clippy::too_many_arguments)]
async fn supervise_loop<A, F>(
    id: ActorId,
    name: &'static str,
    first_actor: A,
    mut factory: F,
    rx: mpsc::Receiver<A::Msg>,
    mut close_rx: mpsc::Receiver<()>,
    shared: Arc<SharedState>,
    metrics: MetricsCell,
) where
    A: Actor,
    F: FnMut() -> A + Send + 'static,
{
    // Wrap rx so multiple incarnations of the actor share the same
    // receiver. The mutex is only ever held by the current incarnation's
    // loop, so contention is zero in practice.
    let rx = Arc::new(Mutex::new(rx));
    let actor_seed = derive_actor_seed(shared.config.rng_seed, id);

    // Sliding-window restart bookkeeping.
    let mut restart_times: Vec<Instant> = Vec::new();
    let mut current_actor: Option<A> = Some(first_actor);

    loop {
        let actor = match current_actor.take() {
            Some(a) => a,
            None => factory(),
        };

        let outcome = run_actor_once(
            id,
            name,
            actor,
            Arc::clone(&rx),
            &mut close_rx,
            actor_seed,
            Arc::clone(&shared.shutdown),
            metrics.clone(),
        )
        .await;

        match outcome {
            ActorRunOutcome::MailboxClosed => {
                tracing::info!(actor = name, id = %id, "mailbox closed; actor stopping");
                break;
            }
            ActorRunOutcome::HandlerError(e) => {
                metrics.errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    actor = name,
                    id = %id,
                    error = %e,
                    "actor handler returned Err; considering restart"
                );
            }
            ActorRunOutcome::HandlerPanic => {
                metrics.errors.fetch_add(1, Ordering::Relaxed);
                tracing::error!(
                    actor = name,
                    id = %id,
                    "actor handler panicked; considering restart"
                );
            }
        }

        // Restart-budget check.
        let now = Instant::now();
        let window = shared.config.restart_window;
        restart_times.retain(|t| now.duration_since(*t) <= window);
        if (restart_times.len() as u32) >= shared.config.max_restarts {
            tracing::error!(
                actor = name,
                id = %id,
                "actor exceeded {} restarts in {:?}; declaring dead",
                shared.config.max_restarts,
                window
            );
            // Close the mailbox so senders get a clean error instead of
            // blocking forever. We do this by dropping our receiver —
            // since current_actor is None and we're about to break, the
            // rx Arc's sole holder is us; it drops when the function
            // returns, which closes the channel from the receive side.
            break;
        }
        restart_times.push(now);
        metrics.restarts.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            actor = name,
            id = %id,
            restart_count = restart_times.len(),
            "supervisor restarting actor"
        );
        // current_actor stays None — factory() will construct on next iter.
    }

    tracing::info!(actor = name, id = %id, "actor loop exited");
}

/// Outcome of one actor incarnation. Controls whether the supervisor
/// restarts.
enum ActorRunOutcome {
    /// Mailbox has been closed, either via scheduler shutdown or because
    /// every `ActorHandle` was dropped. No restart.
    MailboxClosed,
    /// Handler returned `Err`. Restart if budget permits.
    HandlerError(Error),
    /// Handler panicked. Restart if budget permits.
    HandlerPanic,
}

/// Process messages until the mailbox closes, the handler errors, the
/// scheduler shuts down, or the handler panics. Uses
/// `FutureExt::catch_unwind` on the handle future so panics are
/// recoverable by the supervisor (requires `panic = "unwind"`, which the
/// workspace sets for both release and test).
#[allow(clippy::too_many_arguments)]
async fn run_actor_once<A>(
    id: ActorId,
    name: &'static str,
    mut actor: A,
    rx: Arc<Mutex<mpsc::Receiver<A::Msg>>>,
    close_rx: &mut mpsc::Receiver<()>,
    rng_seed: u64,
    shutdown: Arc<AtomicBool>,
    metrics: MetricsCell,
) -> ActorRunOutcome
where
    A: Actor,
{
    use futures::FutureExt;

    let mut ctx = Context {
        id,
        rng_seed,
        shutdown: Arc::clone(&shutdown),
    };

    loop {
        let msg = {
            // Hold the rx lock only long enough to recv one message. The
            // `close_rx.recv()` branch fires when the scheduler drops the
            // close sentinel — we then drain the mailbox and exit.
            let mut rx_guard = rx.lock().await;
            tokio::select! {
                maybe = rx_guard.recv() => match maybe {
                    Some(m) => m,
                    None => return ActorRunOutcome::MailboxClosed,
                },
                _ = close_rx.recv() => {
                    // Drain anything still queued before exiting — that
                    // is the definition of "graceful shutdown drains
                    // in-flight messages".
                    while let Ok(pending) = rx_guard.try_recv() {
                        metrics.mailbox_depth.fetch_sub(1, Ordering::Relaxed);
                        let start = std::time::Instant::now();
                        let handled = std::panic::AssertUnwindSafe(
                            actor.handle(pending, &mut ctx),
                        )
                        .catch_unwind()
                        .await;
                        metrics.record_handle_duration(start.elapsed().as_micros() as u64);
                        match handled {
                            Ok(Ok(())) => {
                                metrics.messages_handled.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(Err(e)) => return ActorRunOutcome::HandlerError(e),
                            Err(_) => return ActorRunOutcome::HandlerPanic,
                        }
                    }
                    return ActorRunOutcome::MailboxClosed;
                }
            }
        };

        metrics.mailbox_depth.fetch_sub(1, Ordering::Relaxed);
        let start = std::time::Instant::now();
        let handled = std::panic::AssertUnwindSafe(actor.handle(msg, &mut ctx))
            .catch_unwind()
            .await;
        metrics.record_handle_duration(start.elapsed().as_micros() as u64);
        match handled {
            Ok(Ok(())) => {
                metrics.messages_handled.fetch_add(1, Ordering::Relaxed);
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    actor = name,
                    id = %id,
                    error = %e,
                    "handler returned Err"
                );
                return ActorRunOutcome::HandlerError(e);
            }
            Err(_) => {
                tracing::error!(actor = name, id = %id, "handler panicked");
                return ActorRunOutcome::HandlerPanic;
            }
        }
    }
}

/// Derive a stable per-actor seed from the scheduler seed + actor id.
/// Splitmix64-style mix so different seeds yield uncorrelated per-actor
/// seeds, and no dependency on the `rand` crate is needed here.
fn derive_actor_seed(scheduler_seed: u64, id: ActorId) -> u64 {
    let mut x = scheduler_seed.wrapping_add(id.0.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

// Re-export the oneshot module used by `AskEnvelope` for callers that
// want to hand-construct an envelope without pulling in a direct tokio
// dependency.
pub use tokio::sync::oneshot as reply_channel;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Minimal actor: accumulates a u32 counter.
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

        sched.shutdown().await;
        assert_eq!(count.load(Ordering::SeqCst), 15);
    }

    // --- Supervision / restart -------------------------------------------

    struct PanickyActor {
        msg_index: u32,
        observed: Arc<Mutex<Vec<u32>>>,
    }

    #[async_trait]
    impl Actor for PanickyActor {
        type Msg = u32;

        async fn handle(&mut self, msg: Self::Msg, _ctx: &mut Context) -> Result<()> {
            self.msg_index += 1;
            self.observed.lock().await.push(msg);
            if self.msg_index == 3 {
                panic!("boom on msg {msg}");
            }
            Ok(())
        }

        fn name(&self) -> &'static str {
            "panicky"
        }
    }

    #[tokio::test]
    async fn supervisor_restarts_panicking_actor() {
        let observed = Arc::new(Mutex::new(Vec::<u32>::new()));
        let observed_clone = Arc::clone(&observed);
        let mut sched = Scheduler::new(SchedulerConfig {
            max_restarts: 3,
            restart_window: Duration::from_secs(10),
            ..Default::default()
        });
        let h = sched.spawn_supervised(move || PanickyActor {
            msg_index: 0,
            observed: Arc::clone(&observed_clone),
        });

        for v in [1_u32, 2, 3, 4, 5] {
            h.send(v).await.unwrap();
        }

        // Let the supervisor restart + drain.
        for _ in 0..200 {
            let seen = observed.lock().await.len();
            if seen >= 5 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        sched.shutdown().await;

        let seen = observed.lock().await.clone();
        assert_eq!(seen, vec![1, 2, 3, 4, 5], "all messages should land");
    }

    #[tokio::test]
    async fn supervisor_gives_up_after_max_restarts() {
        struct AlwaysPanics;
        #[async_trait]
        impl Actor for AlwaysPanics {
            type Msg = ();
            async fn handle(&mut self, _msg: (), _ctx: &mut Context) -> Result<()> {
                panic!("always");
            }
            fn name(&self) -> &'static str {
                "always-panics"
            }
        }

        let mut sched = Scheduler::new(SchedulerConfig {
            max_restarts: 2,
            restart_window: Duration::from_secs(10),
            shutdown_timeout: Duration::from_millis(500),
            ..Default::default()
        });
        let h = sched.spawn_supervised(|| AlwaysPanics);
        for _ in 0..5 {
            let _ = h.send(()).await;
        }
        // Wait long enough for 2 restarts (= max) to be consumed.
        for _ in 0..200 {
            if !h.is_alive() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        sched.shutdown().await;
        assert!(
            !h.is_alive(),
            "actor should be dead after exceeding restart budget"
        );
    }

    // --- Graceful shutdown ------------------------------------------------

    struct SlowActor {
        handled: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Actor for SlowActor {
        type Msg = ();
        async fn handle(&mut self, _msg: (), _ctx: &mut Context) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(100)).await;
            self.handled.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn name(&self) -> &'static str {
            "slow"
        }
    }

    #[tokio::test]
    async fn shutdown_drains_in_flight_messages() {
        let handled = Arc::new(AtomicU32::new(0));
        let mut sched = Scheduler::new(SchedulerConfig {
            shutdown_timeout: Duration::from_secs(5),
            ..Default::default()
        });
        let h = sched.spawn(SlowActor {
            handled: Arc::clone(&handled),
        });

        for _ in 0..10 {
            h.send(()).await.unwrap();
        }

        sched.shutdown().await;
        assert_eq!(handled.load(Ordering::SeqCst), 10);
    }

    // --- Ask pattern ------------------------------------------------------

    struct EchoActor;

    #[async_trait]
    impl Actor for EchoActor {
        type Msg = AskEnvelope<String, String>;
        async fn handle(&mut self, env: Self::Msg, _ctx: &mut Context) -> Result<()> {
            let reply = env.req.clone();
            let _ = env.reply(reply);
            Ok(())
        }
        fn name(&self) -> &'static str {
            "echo"
        }
    }

    #[tokio::test]
    async fn ask_round_trips_through_echo_actor() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let h = sched.spawn(EchoActor);
        let reply: String = h.ask("hello".to_string()).await.unwrap();
        assert_eq!(reply, "hello");
        sched.shutdown().await;
    }

    // --- Broadcast --------------------------------------------------------

    struct TopicListener {
        rx: Option<broadcast::Receiver<u32>>,
        latest: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Actor for TopicListener {
        type Msg = ();
        async fn handle(&mut self, _msg: (), _ctx: &mut Context) -> Result<()> {
            if let Some(rx) = self.rx.as_mut()
                && let Ok(v) = rx.try_recv()
            {
                self.latest.store(v, Ordering::SeqCst);
            }
            Ok(())
        }
        fn name(&self) -> &'static str {
            "topic-listener"
        }
    }

    #[tokio::test]
    async fn broadcast_reaches_every_subscriber() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let topic: TopicHandle<u32> = sched.topic::<u32>("ticks");

        let a = Arc::new(AtomicU32::new(0));
        let b = Arc::new(AtomicU32::new(0));
        let c = Arc::new(AtomicU32::new(0));
        let ha = sched.spawn(TopicListener {
            rx: Some(topic.subscribe()),
            latest: Arc::clone(&a),
        });
        let hb = sched.spawn(TopicListener {
            rx: Some(topic.subscribe()),
            latest: Arc::clone(&b),
        });
        let hc = sched.spawn(TopicListener {
            rx: Some(topic.subscribe()),
            latest: Arc::clone(&c),
        });
        assert_eq!(topic.receiver_count(), 3);

        let n = topic.publish(42);
        assert_eq!(n, 3);

        // Poke each listener so it reads from its broadcast receiver.
        ha.send(()).await.unwrap();
        hb.send(()).await.unwrap();
        hc.send(()).await.unwrap();

        sched.shutdown().await;
        assert_eq!(a.load(Ordering::SeqCst), 42);
        assert_eq!(b.load(Ordering::SeqCst), 42);
        assert_eq!(c.load(Ordering::SeqCst), 42);
    }

    // --- Metrics ----------------------------------------------------------

    #[tokio::test]
    async fn metrics_count_messages_handled() {
        let count = Arc::new(AtomicU32::new(0));
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let h = sched.spawn(CountingActor {
            count: Arc::clone(&count),
        });

        for _ in 0..100 {
            h.send(1).await.unwrap();
        }

        // Wait until all 100 have been handled.
        for _ in 0..400 {
            let snap = sched.actor_metrics(h.id).unwrap();
            if snap.messages_handled == 100 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let snap = sched.actor_metrics(h.id).unwrap();
        assert_eq!(snap.messages_handled, 100);
        assert_eq!(snap.name, "counting");
        sched.shutdown().await;
    }

    #[tokio::test]
    async fn metrics_histogram_populates() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let h = sched.spawn(SlowActor {
            handled: Arc::new(AtomicU32::new(0)),
        });
        h.send(()).await.unwrap();
        // Wait for handler to finish (100ms sleep).
        for _ in 0..100 {
            let snap = sched.actor_metrics(h.id).unwrap();
            if snap.messages_handled == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let snap = sched.actor_metrics(h.id).unwrap();
        assert_eq!(snap.messages_handled, 1);
        let total: u64 = snap.handle_duration_us.iter().sum();
        assert_eq!(total, 1, "exactly one handle should be recorded");
        sched.shutdown().await;
    }

    // --- Deterministic replay --------------------------------------------
    //
    // Two schedulers with the same seed + same message sequence must
    // produce the same counter state. Uses the deterministic runtime so
    // task scheduling is single-threaded.

    fn run_one(seed: u64, msgs: &[u32]) -> u32 {
        let rt = Scheduler::build_deterministic_runtime();
        rt.block_on(async move {
            let mut sched = Scheduler::new(SchedulerConfig {
                deterministic: true,
                rng_seed: seed,
                ..Default::default()
            });
            let count = Arc::new(AtomicU32::new(0));
            let h = sched.spawn(CountingActor {
                count: Arc::clone(&count),
            });
            for v in msgs {
                h.send(*v).await.unwrap();
            }
            sched.shutdown().await;
            count.load(Ordering::SeqCst)
        })
    }

    #[test]
    fn deterministic_replay_matches() {
        let msgs: Vec<u32> = (1..=50).collect();
        let a = run_one(0xDEAD_BEEF, &msgs);
        let b = run_one(0xDEAD_BEEF, &msgs);
        assert_eq!(a, b);
        assert_eq!(a, (1..=50).sum::<u32>());
    }

    // --- Seed derivation --------------------------------------------------

    #[test]
    fn actor_seed_is_deterministic_and_per_actor() {
        let s1_a1 = derive_actor_seed(0xABCD_1234, ActorId(1));
        let s1_a1_again = derive_actor_seed(0xABCD_1234, ActorId(1));
        let s1_a2 = derive_actor_seed(0xABCD_1234, ActorId(2));
        let s2_a1 = derive_actor_seed(0xDEAD_BEEF, ActorId(1));
        assert_eq!(s1_a1, s1_a1_again);
        assert_ne!(s1_a1, s1_a2);
        assert_ne!(s1_a1, s2_a1);
    }
}
