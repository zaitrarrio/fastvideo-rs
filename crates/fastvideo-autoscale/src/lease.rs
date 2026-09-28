//! Leader election: one lease row per controller name. Only the holder
//! applies decisions; other gateway replicas decide (for their status page)
//! but never call a provider's write API.
//!
//! The D1 row is taken with one conditional upsert, so two replicas racing
//! for an expired lease cannot both win:
//!
//! ```sql
//! INSERT INTO fv_autoscale_lease (name, holder, expires_at, updated_at) VALUES (?, ?, ?, ?)
//! ON CONFLICT(name) DO UPDATE SET holder = excluded.holder, expires_at = excluded.expires_at,
//!   updated_at = excluded.updated_at
//! WHERE fv_autoscale_lease.holder = excluded.holder OR fv_autoscale_lease.expires_at < excluded.updated_at
//! ```

use std::collections::BTreeMap;
use std::sync::Mutex;

use fastvideo_serve_kit::d1::{D1Client, Stmt};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
#[error("lease: {0}")]
pub struct LeaseError(pub String);

#[async_trait::async_trait]
pub trait Lease: Send + Sync {
    /// Takes or renews the lease until `now_s + ttl_s`; true when `holder` has it.
    async fn acquire(&self, name: &str, holder: &str, now_s: f64, ttl_s: f64) -> Result<bool, LeaseError>;
    /// Gives it up (a shutting-down replica), if held.
    async fn release(&self, name: &str, holder: &str) -> Result<(), LeaseError>;
}

/// Process-local lease; share one `Arc` between controllers to model
/// several replicas.
#[derive(Default)]
pub struct MemoryLease {
    rows: Mutex<BTreeMap<String, (String, f64)>>,
}

#[async_trait::async_trait]
impl Lease for MemoryLease {
    async fn acquire(&self, name: &str, holder: &str, now_s: f64, ttl_s: f64) -> Result<bool, LeaseError> {
        let mut g = self.rows.lock().unwrap_or_else(|p| p.into_inner());
        match g.get(name) {
            Some((h, exp)) if h != holder && *exp >= now_s => Ok(false),
            _ => {
                g.insert(name.to_owned(), (holder.to_owned(), now_s + ttl_s));
                Ok(true)
            }
        }
    }
    async fn release(&self, name: &str, holder: &str) -> Result<(), LeaseError> {
        let mut g = self.rows.lock().unwrap_or_else(|p| p.into_inner());
        if g.get(name).is_some_and(|(h, _)| h == holder) {
            g.remove(name);
        }
        Ok(())
    }
}

/// The lease row in Cloudflare D1 (the gateway's `fv-jobs` database).
pub struct D1Lease {
    client: D1Client,
    ready: tokio::sync::OnceCell<()>,
}

pub const LEASE_TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS fv_autoscale_lease (\
name TEXT PRIMARY KEY, holder TEXT NOT NULL, expires_at REAL NOT NULL, updated_at REAL NOT NULL)";

impl D1Lease {
    pub fn new(client: D1Client) -> Self {
        Self { client, ready: tokio::sync::OnceCell::new() }
    }

    async fn ensure_table(&self) -> Result<(), LeaseError> {
        self.ready
            .get_or_try_init(|| async {
                self.client.query(Stmt::raw(LEASE_TABLE_SQL)).await.map(|_| ()).map_err(|e| LeaseError(e.to_string()))
            })
            .await
            .map(|_| ())
    }
}

#[async_trait::async_trait]
impl Lease for D1Lease {
    async fn acquire(&self, name: &str, holder: &str, now_s: f64, ttl_s: f64) -> Result<bool, LeaseError> {
        self.ensure_table().await?;
        let upsert = Stmt::new(
            "INSERT INTO fv_autoscale_lease (name, holder, expires_at, updated_at) VALUES (?, ?, ?, ?) \
             ON CONFLICT(name) DO UPDATE SET holder = excluded.holder, expires_at = excluded.expires_at, \
             updated_at = excluded.updated_at \
             WHERE fv_autoscale_lease.holder = excluded.holder OR fv_autoscale_lease.expires_at < excluded.updated_at",
            vec![json!(name), json!(holder), json!(now_s + ttl_s), json!(now_s)],
        );
        let select = Stmt::new("SELECT holder FROM fv_autoscale_lease WHERE name = ?", vec![json!(name)]);
        let out = self.client.batch(vec![upsert, select]).await.map_err(|e| LeaseError(e.to_string()))?;
        let got = out
            .get(1)
            .and_then(|r| r.rows.first())
            .and_then(|row| row.get("holder"))
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        Ok(got.as_deref() == Some(holder))
    }

    async fn release(&self, name: &str, holder: &str) -> Result<(), LeaseError> {
        self.ensure_table().await?;
        self.client
            .query(Stmt::new(
                "DELETE FROM fv_autoscale_lease WHERE name = ? AND holder = ?",
                vec![json!(name), json!(holder)],
            ))
            .await
            .map(|_| ())
            .map_err(|e| LeaseError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fastvideo_serve_kit::d1::mock::MockD1;
    use fastvideo_serve_kit::d1::RetryPolicy;

    use super::*;

    async fn exercise(l: &dyn Lease) {
        assert!(l.acquire("as", "a", 100.0, 60.0).await.unwrap());
        assert!(!l.acquire("as", "b", 110.0, 60.0).await.unwrap(), "b waits while a holds it");
        assert!(l.acquire("as", "a", 150.0, 60.0).await.unwrap(), "a renews to 210");
        assert!(!l.acquire("as", "b", 200.0, 60.0).await.unwrap());
        assert!(l.acquire("as", "b", 211.0, 60.0).await.unwrap(), "expired: b takes over");
        assert!(!l.acquire("as", "a", 212.0, 60.0).await.unwrap());
        assert!(l.acquire("other", "a", 212.0, 60.0).await.unwrap(), "names are independent");
        l.release("as", "a").await.unwrap();
        assert!(!l.acquire("as", "a", 213.0, 60.0).await.unwrap(), "release by a non-holder is a no-op");
        l.release("as", "b").await.unwrap();
        assert!(l.acquire("as", "a", 214.0, 60.0).await.unwrap());
    }

    #[tokio::test]
    async fn memory_lease() {
        exercise(&MemoryLease::default()).await;
    }

    #[tokio::test]
    async fn d1_lease_over_the_mock() {
        let m = MockD1::new();
        let client = D1Client::new(Arc::new(m.clone())).with_retry(RetryPolicy::immediate(3));
        exercise(&D1Lease::new(client)).await;
        assert!(m.statements().iter().any(|s| s.contains("ON CONFLICT(name)")));
    }
}
