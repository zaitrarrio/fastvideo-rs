//! MiniMax rate limits (research §1.7): 300 creates per minute and 30
//! in-flight tasks per API key. Both answer 429 `rate_limit_error (1002)`.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use fastvideo_protocol::{ApiError, KeyId};
use time::{Duration, OffsetDateTime};

/// A sliding one-minute window per key.
#[derive(Debug)]
pub(crate) struct RateLimiter {
    rpm: u32,
    windows: Mutex<HashMap<Option<KeyId>, VecDeque<OffsetDateTime>>>,
}

impl RateLimiter {
    pub(crate) fn new(rpm: u32) -> Self {
        Self { rpm, windows: Mutex::new(HashMap::new()) }
    }

    /// Records one request at `now`, or refuses it when the key already made
    /// `rpm` requests in the last minute.
    pub(crate) fn hit(&self, owner: Option<&KeyId>, now: OffsetDateTime) -> Result<(), ApiError> {
        if self.rpm == 0 {
            return Ok(());
        }
        let mut w = self.windows.lock().unwrap_or_else(|p| p.into_inner());
        let q = w.entry(owner.cloned()).or_default();
        while q.front().is_some_and(|t| now - *t >= Duration::MINUTE) {
            q.pop_front();
        }
        if q.len() as u32 >= self.rpm {
            let retry = q
                .front()
                .map(|t| (Duration::MINUTE - (now - *t)).whole_seconds().max(1) as u32)
                .unwrap_or(1);
            return Err(ApiError::rate_limited("please retry later").with_retry_after(retry));
        }
        q.push_back(now);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_protocol::ErrorKind;

    #[test]
    fn window_slides() {
        let l = RateLimiter::new(2);
        let t0 = OffsetDateTime::UNIX_EPOCH;
        let k = KeyId("a".into());
        assert!(l.hit(Some(&k), t0).is_ok());
        assert!(l.hit(Some(&k), t0 + Duration::seconds(10)).is_ok());
        let e = l.hit(Some(&k), t0 + Duration::seconds(20)).unwrap_err();
        assert_eq!(e.kind, ErrorKind::RateLimited);
        assert_eq!(e.retry_after_s, Some(40));
        // Another key has its own window.
        assert!(l.hit(Some(&KeyId("b".into())), t0 + Duration::seconds(20)).is_ok());
        // The first request leaves the window after a minute.
        assert!(l.hit(Some(&k), t0 + Duration::seconds(60)).is_ok());
        assert!(RateLimiter::new(0).hit(None, t0).is_ok());
    }
}
