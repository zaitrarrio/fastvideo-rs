//! Data channels (design §5.6, §5.7; WP-04).
//!
//! Clients create the channels and we accept them (`ondatachannel`
//! semantics): fal WMA opens `control` (fal §8.1), Reactor opens `data` and
//! `control` (reactor §5.2). Channels are reliable and ordered, the
//! `RTCDataChannel` default the clients use. A [`ChannelPolicy`] says which
//! labels a protocol accepts; any other label is closed on open.
//!
//! Messages written before a channel opens are queued, at most
//! [`MAX_QUEUED_MESSAGES`] per peer (the JS WMA client's own limit, fal
//! §8.2), and flushed in order on open.

use bytes::Bytes;

/// fal WMA control channel (JSON text).
pub const FAL_CONTROL: &str = "control";
/// Reactor data channel.
pub const REACTOR_DATA: &str = "data";
/// Reactor control channel.
pub const REACTOR_CONTROL: &str = "control";

/// Per-peer cap on messages queued before their channel opens.
pub const MAX_QUEUED_MESSAGES: usize = 64;

/// Which client-created channel labels a session accepts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ChannelPolicy {
    /// Accept every label.
    #[default]
    Any,
    /// Accept only these labels; others are closed when they open.
    Only(Vec<String>),
}

impl ChannelPolicy {
    pub fn fal() -> Self {
        ChannelPolicy::Only(vec![FAL_CONTROL.into()])
    }

    pub fn reactor() -> Self {
        ChannelPolicy::Only(vec![REACTOR_DATA.into(), REACTOR_CONTROL.into()])
    }

    /// WHIP carries no data channels.
    pub fn none() -> Self {
        ChannelPolicy::Only(Vec::new())
    }

    pub fn accepts(&self, label: &str) -> bool {
        match self {
            ChannelPolicy::Any => true,
            ChannelPolicy::Only(v) => v.iter().any(|l| l == label),
        }
    }
}

/// One data-channel message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMessage {
    pub label: String,
    /// `false`: a text (UTF-8) message; `true`: binary.
    pub binary: bool,
    pub data: Bytes,
}

impl ChannelMessage {
    pub fn text(label: impl Into<String>, text: impl Into<String>) -> Self {
        ChannelMessage {
            label: label.into(),
            binary: false,
            data: Bytes::from(text.into()),
        }
    }

    pub fn binary(label: impl Into<String>, data: impl Into<Bytes>) -> Self {
        ChannelMessage {
            label: label.into(),
            binary: true,
            data: data.into(),
        }
    }

    /// The payload as text, when it is a valid UTF-8 text message.
    pub fn as_text(&self) -> Option<&str> {
        if self.binary {
            return None;
        }
        std::str::from_utf8(&self.data).ok()
    }
}

/// Bounded FIFO of messages waiting for their channel to open.
#[derive(Debug, Default)]
pub struct PendingMessages {
    queue: std::collections::VecDeque<ChannelMessage>,
}

impl PendingMessages {
    /// Queue a message; `Err` returns it when the cap is reached.
    pub fn push(&mut self, m: ChannelMessage) -> Result<(), ChannelMessage> {
        if self.queue.len() >= MAX_QUEUED_MESSAGES {
            return Err(m);
        }
        self.queue.push_back(m);
        Ok(())
    }

    /// Remove and return every queued message for `label`, in order.
    pub fn take_label(&mut self, label: &str) -> Vec<ChannelMessage> {
        let (take, keep): (Vec<_>, Vec<_>) = self.queue.drain(..).partition(|m| m.label == label);
        self.queue = keep.into();
        take
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policies() {
        assert!(ChannelPolicy::fal().accepts("control"));
        assert!(!ChannelPolicy::fal().accepts("data"));
        assert!(ChannelPolicy::reactor().accepts("data"));
        assert!(ChannelPolicy::reactor().accepts("control"));
        assert!(!ChannelPolicy::none().accepts("control"));
        assert!(ChannelPolicy::Any.accepts("anything"));
    }

    #[test]
    fn pending_queue_is_bounded_and_ordered() {
        let mut q = PendingMessages::default();
        for i in 0..MAX_QUEUED_MESSAGES {
            let label = if i % 2 == 0 { "a" } else { "b" };
            q.push(ChannelMessage::text(label, i.to_string())).unwrap();
        }
        assert!(q.push(ChannelMessage::text("a", "overflow")).is_err());
        let a = q.take_label("a");
        assert_eq!(a.len(), MAX_QUEUED_MESSAGES / 2);
        assert_eq!(a[0].as_text(), Some("0"));
        assert_eq!(a[1].as_text(), Some("2"));
        assert_eq!(q.len(), MAX_QUEUED_MESSAGES / 2);
        assert!(q.take_label("a").is_empty());
    }

    #[test]
    fn text_and_binary() {
        assert_eq!(ChannelMessage::text("c", "{}").as_text(), Some("{}"));
        assert_eq!(ChannelMessage::binary("c", vec![0xff]).as_text(), None);
    }
}
