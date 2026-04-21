//! Strongly-typed ID newtypes.
//!
//! Rust won't let you confuse an ActorId for an EventId — a V1 class of
//! bug (passing raw uint64_t IDs around) eliminated at compile time.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Generic 64-bit ID. Use the named wrappers below for domain types.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id(pub u64);

impl fmt::Debug for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id({})", self.0)
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a live actor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorId(pub u64);

impl fmt::Debug for ActorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActorId({})", self.0)
    }
}

impl fmt::Display for ActorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "actor:{}", self.0)
    }
}

/// Monotonic event sequence number. Assigned by the event log.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EventId(pub u64);

impl fmt::Debug for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EventId({})", self.0)
    }
}

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ev:{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_distinct_types() {
        // This is a compile-time guarantee: can't cross-assign ActorId to
        // EventId even though both wrap u64.
        let a: ActorId = ActorId(1);
        let e: EventId = EventId(1);
        assert_eq!(a.0, 1);
        assert_eq!(e.0, 1);
        // The following would not compile:
        // let _: EventId = a;
    }

    #[test]
    fn event_ids_are_orderable() {
        assert!(EventId(1) < EventId(2));
    }
}
