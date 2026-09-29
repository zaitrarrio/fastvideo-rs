//! Client uploads (reactor §3.5): the presigned-URL protocol a `FileRef`
//! command parameter (the avatar image, a voice file) is sent through.
//!
//! - `POST /sessions/{sid}/uploads` `{name, size, mime_type, upload_id?}` →
//!   **201** `{presigned_id, presigned_url, path}`;
//! - `PUT <presigned_url>` with the raw bytes; the declared size is enforced.
//!   The byte route is `PUT /sessions/{sid}/uploads/{id}` (RT's
//!   `/uploads/{id}` is serve-kit's route in fv-serve);
//! - a command then references the upload by id (`Command.uploads[param]`,
//!   or `{upload_id, …}` inline as the argument), and the model side waits up
//!   to [`RESOLVE_TIMEOUT`] for the bytes (`runner.py:_UPLOAD_RESOLVE_TIMEOUT_SECONDS`).
//!
//! Uploads live in a per-runtime directory and are cleared when the session
//! ends (RT clears uploads on CLOSING).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use tokio::sync::Notify;

use crate::session::Refusal;

/// How long a command waits for an upload's bytes.
pub const RESOLVE_TIMEOUT: Duration = Duration::from_secs(10);
/// Largest accepted upload.
pub const MAX_UPLOAD_BYTES: u64 = 64 * 1024 * 1024;
/// Uploads a session may hold.
pub const MAX_UPLOADS: usize = 64;

/// A completed upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Upload {
    pub id: String,
    pub name: String,
    pub mime_type: String,
    pub size: u64,
    pub path: PathBuf,
}

#[derive(Debug)]
struct Entry {
    name: String,
    mime_type: String,
    size: u64,
    done: bool,
}

/// The runtime's upload store.
#[derive(Debug)]
pub struct Uploads {
    dir: PathBuf,
    entries: Mutex<HashMap<String, Entry>>,
    notify: Notify,
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

impl Uploads {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir, entries: Mutex::new(HashMap::new()), notify: Notify::new() }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Entry>> {
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Registers an upload; returns its id.
    pub fn create(&self, name: &str, size: u64, mime_type: &str, id: Option<&str>) -> Result<String, Refusal> {
        if size == 0 || size > MAX_UPLOAD_BYTES {
            return Err(Refusal::new(422, format!("size must be 1..={MAX_UPLOAD_BYTES} bytes")));
        }
        let id = match id {
            Some(i) if valid_id(i) => i.to_owned(),
            Some(_) => return Err(Refusal::new(422, "upload_id: 1-64 of [A-Za-z0-9_-]")),
            None => uuid::Uuid::new_v4().simple().to_string(),
        };
        let mut m = self.lock();
        if m.len() >= MAX_UPLOADS && !m.contains_key(&id) {
            return Err(Refusal::new(429, format!("at most {MAX_UPLOADS} uploads per session")));
        }
        m.insert(
            id.clone(),
            Entry { name: name.chars().take(255).collect(), mime_type: mime_type.chars().take(127).collect(), size, done: false },
        );
        Ok(id)
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(id)
    }

    /// Stores the bytes of a registered upload (exactly its declared size).
    pub fn put(&self, id: &str, bytes: &[u8]) -> Result<(), Refusal> {
        let size = match self.lock().get(id) {
            Some(e) => e.size,
            None => return Err(Refusal::new(404, "no such upload")),
        };
        if bytes.len() as u64 != size {
            return Err(Refusal::new(400, format!("expected {size} bytes, got {}", bytes.len())));
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| Refusal::new(500, format!("upload store: {e}")))?;
        let tmp = self.dir.join(format!("{id}.part"));
        std::fs::write(&tmp, bytes)
            .and_then(|()| std::fs::rename(&tmp, self.path(id)))
            .map_err(|e| Refusal::new(500, format!("upload store: {e}")))?;
        if let Some(e) = self.lock().get_mut(id) {
            e.done = true;
        }
        self.notify.notify_waiters();
        Ok(())
    }

    /// The completed upload `id`, waiting up to `timeout` for its bytes.
    pub async fn resolve(&self, id: &str, timeout: Duration) -> Option<Upload> {
        let end = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.notify.notified();
            {
                let m = self.lock();
                match m.get(id) {
                    Some(e) if e.done => {
                        return Some(Upload {
                            id: id.to_owned(),
                            name: e.name.clone(),
                            mime_type: e.mime_type.clone(),
                            size: e.size,
                            path: self.path(id),
                        })
                    }
                    Some(_) => {}
                    None => return None,
                }
            }
            if tokio::time::timeout_at(end, notified).await.is_err() {
                return None;
            }
        }
    }

    /// Drops every upload (session end).
    pub fn clear(&self) {
        let ids: Vec<String> = self.lock().drain().map(|(k, _)| k).collect();
        for id in ids {
            let _ = std::fs::remove_file(self.path(&id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn create_put_resolve_clear() {
        let dir = std::env::temp_dir().join(format!("fv-reactor-uploads-test-{}", uuid::Uuid::new_v4().simple()));
        let u = Uploads::new(dir.clone());
        assert_eq!(u.create("a.png", 0, "image/png", None).unwrap_err().status, 422);
        assert_eq!(u.create("a.png", 3, "image/png", Some("../x")).unwrap_err().status, 422);
        let id = u.create("a.png", 3, "image/png", Some("img1")).unwrap();
        assert_eq!(id, "img1");
        assert_eq!(u.put("nope", b"abc").unwrap_err().status, 404);
        assert_eq!(u.put(&id, b"ab").unwrap_err().status, 400);
        // A waiter sees the bytes as soon as they land.
        let waiter = {
            let u = &u;
            async move { u.resolve("img1", Duration::from_secs(5)).await }
        };
        let (got, ()) = tokio::join!(waiter, async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            u.put("img1", b"abc").unwrap();
        });
        let got = got.expect("resolved");
        assert_eq!((got.name.as_str(), got.size), ("a.png", 3));
        assert_eq!(std::fs::read(&got.path).unwrap(), b"abc");
        assert!(u.resolve("missing", Duration::from_millis(10)).await.is_none());
        let pending = u.create("b", 1, "x", None).unwrap();
        assert!(u.resolve(&pending, Duration::from_millis(20)).await.is_none());
        u.clear();
        assert!(!got.path.exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
