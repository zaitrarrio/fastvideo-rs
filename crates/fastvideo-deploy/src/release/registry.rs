//! The release history and the deployments ledger in D1
//! (`scripts/serve/lib/registry.sh`; schema `deploy/d1/registry.sql`): the
//! same statements, the same columns in the same order, so fv-deploy,
//! `release.sh`, the deploy scripts and fv-control (which reads
//! `deployments` and `releases`, `control/src/releases.ts`) see one ledger.
//!
//! Deployments are written best effort (a D1 failure is a warning, never a
//! failed deploy); `FV_REGISTRY=0` turns them off.

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Map, Value};

use super::d1::{D1Error, D1};
use crate::boot::env::BootEnv;
use crate::boot::sys::Clock;
use crate::lifecycle::Recorder;
use crate::provider::{Instance, InstanceId, Launch, ProviderKind};

/// Ordered `key -> value` fields: a jq object, whose key order decides the
/// column order of the statements (`registry.sh` builds them with jq).
pub type Fields = Vec<(String, Value)>;

/// jq's `left + right`: keys of `left` first (a key of both keeps its place
/// and takes `right`'s value), then `right`'s new keys in order.
pub fn merge(left: Fields, right: Fields) -> Fields {
    let mut out = left;
    for (k, v) in right {
        match out.iter_mut().find(|(x, _)| *x == k) {
            Some(slot) => slot.1 = v,
            None => out.push((k, v)),
        }
    }
    out
}

fn get<'f>(f: &'f Fields, k: &str) -> Option<&'f Value> {
    f.iter().find(|(x, _)| x == k).map(|(_, v)| v)
}

/// A field's string value, "" when absent or not a string (jq's `.k // empty`).
fn get_str<'f>(f: &'f Fields, k: &str) -> &'f str {
    get(f, k).and_then(Value::as_str).unwrap_or("")
}

/// A release's image set, `key -> repo@sha256:…`, in its stored order (the
/// `digests` column is that JSON object as text; the order matters to the
/// scripts' readers, which take the first matching key).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Digests(pub Vec<(String, String)>);

impl<'de> Deserialize<'de> for Digests {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Digests;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object of image references")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> Result<Digests, A::Error> {
                let mut v = Vec::new();
                while let Some((k, x)) = m.next_entry::<String, String>()? {
                    v.push((k, x));
                }
                Ok(Digests(v))
            }
        }
        d.deserialize_map(V)
    }
}

impl Digests {
    /// The `digests` column's text.
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("digests {text:?}: {e}"))
    }

    pub fn get(&self, k: &str) -> Option<&str> {
        self.0.iter().find(|(x, _)| x == k).map(|(_, v)| v.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The keys sorted (jq's `keys[]`, the order the scripts retag in).
    pub fn sorted_keys(&self) -> Vec<&str> {
        let mut k: Vec<&str> = self.0.iter().map(|(x, _)| x.as_str()).collect();
        k.sort_unstable();
        k
    }

    /// Compact JSON in the stored order (`jq -c`, `tojson`).
    pub fn to_json(&self) -> String {
        let parts: Vec<String> = self
            .0
            .iter()
            .map(|(k, v)| format!("{}:{}", Value::from(k.as_str()), Value::from(v.as_str())))
            .collect();
        format!("{{{}}}", parts.join(","))
    }

    /// Compact JSON with sorted keys (`jq -S -c`: what "the same image set" compares).
    pub fn canonical(&self) -> String {
        let m: Map<String, Value> = self
            .0
            .iter()
            .map(|(k, v)| (k.clone(), Value::from(v.as_str())))
            .collect();
        Value::Object(m).to_string()
    }
}

/// Who did it (`fv_actor`): `FV_DEPLOYED_BY`, else `ci:<workflow>#<run>[ by
/// <actor>]` in GitHub Actions, `agent:<user>` under Claude Code, else
/// `user:<user>@<host>`.
pub fn actor(env: &BootEnv) -> String {
    if let Some(a) = env.get("FV_DEPLOYED_BY") {
        return a.to_owned();
    }
    if env.get("GITHUB_ACTIONS") == Some("true") {
        let mut s = format!(
            "ci:{}#{}",
            env.get("GITHUB_WORKFLOW").unwrap_or("?"),
            env.get("GITHUB_RUN_ID").unwrap_or("?")
        );
        if let Some(a) = env.get("GITHUB_ACTOR") {
            s.push_str(&format!(" by {a}"));
        }
        return s;
    }
    let user = env
        .get("USER")
        .or_else(|| env.get("LOGNAME"))
        .unwrap_or("?");
    if env.get("CLAUDECODE").is_some() {
        return format!("agent:{user}");
    }
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|h| h.trim().to_owned())
        .filter(|h| !h.is_empty())
        .or_else(|| env.get("HOSTNAME").map(str::to_owned))
        .unwrap_or_else(|| "?".into());
    let short = host.split('.').next().unwrap_or("?");
    format!("user:{user}@{short}")
}

// The statements, as registry.sh / release.sh spell them.
pub const SQL_RELEASE_CURRENT: &str =
    "SELECT * FROM releases WHERE channel = ? ORDER BY id DESC LIMIT 1";
pub const SQL_RELEASE_HEADS: &str = "SELECT r.* FROM releases r JOIN (SELECT channel, MAX(id) AS id FROM releases GROUP BY channel) h ON r.id = h.id ORDER BY r.channel";
pub const SQL_RELEASE_INSERT: &str = "INSERT INTO releases (channel, git_sha, digests, action, promoted_at, promoted_by, notes, run_url, templates_updated, source_release) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id";
pub const SQL_RELEASE_FOR_DIGEST: &str =
    "SELECT id, channel, git_sha, digests FROM releases WHERE digests LIKE ? ORDER BY id DESC LIMIT 50";
pub const SQL_RELEASE_BY_ID: &str = "SELECT * FROM releases WHERE id = ? AND channel = ?";
pub const SQL_RELEASE_PREVIOUS: &str = "SELECT * FROM releases WHERE channel = ? AND id < ? AND git_sha != ? AND rolled_back_at IS NULL ORDER BY id DESC LIMIT 1";
pub const SQL_RELEASE_ROLLED_BACK: &str = "UPDATE releases SET rolled_back_at = ? WHERE channel = ? AND git_sha != ? AND rolled_back_at IS NULL AND id > ?";
pub const SQL_DEPLOYMENTS_LIVE: &str = "SELECT * FROM deployments WHERE deleted_at IS NULL";
pub const SQL_DEPLOYMENT_READY_AT: &str =
    "UPDATE deployments SET ready_at = COALESCE(ready_at, ?) WHERE id = ?";
pub const SQL_DEPLOYMENT_DELETED: &str = "UPDATE deployments SET status = ?, deleted_at = COALESCE(deleted_at, ?), updated_at = ? WHERE id = ?";

/// The columns a deployments write may set from `key=value` arguments
/// (`_fv_kv_json`).
pub const DEPLOYMENT_FIELDS: [&str; 13] = [
    "name",
    "pool",
    "variant",
    "image",
    "digest",
    "git_sha",
    "channel",
    "region",
    "dc",
    "gpu",
    "cost_per_hr",
    "status",
    "meta",
];

/// `key=value` arguments as fields (`_fv_kv_json`): unknown keys, empty
/// values, a `cost_per_hr` that is not a plain number and a `meta` that is
/// not a JSON object are dropped; `cost_per_hr` becomes a number, the rest
/// stay strings (`meta` its JSON text).
pub fn kv(pairs: &[(&str, &str)]) -> Fields {
    let mut out = Fields::new();
    for (k, v) in pairs {
        if !DEPLOYMENT_FIELDS.contains(k) || v.is_empty() {
            continue;
        }
        let val = match *k {
            "cost_per_hr" => {
                if !v.chars().all(|c| c.is_ascii_digit() || c == '.') {
                    continue;
                }
                match serde_json::from_str::<Value>(v) {
                    Ok(n @ Value::Number(_)) => n,
                    _ => continue,
                }
            }
            "meta" => match serde_json::from_str::<Value>(v) {
                Ok(Value::Object(_)) => Value::from(*v),
                _ => continue,
            },
            _ => Value::from(*v),
        };
        out = merge(out, vec![((*k).to_owned(), val)]);
    }
    out
}

/// What a release row says about a digest (`fv_release_for_digest`).
#[derive(Clone, Debug, PartialEq)]
pub struct DigestHit {
    pub git_sha: Value,
    /// The image key of the newest release naming it (`null` when none matches).
    pub key: Value,
    /// The channels whose current release names it.
    pub channels: Vec<String>,
}

/// One row to insert into `releases` (`fv_release_insert`).
#[derive(Clone, Debug, PartialEq)]
pub struct ReleaseRow {
    pub channel: String,
    pub git_sha: String,
    pub digests: Digests,
    pub action: String,
    pub notes: Option<String>,
    pub run_url: Option<String>,
    pub templates_updated: bool,
    pub source_release: Option<i64>,
}

/// The ledger: D1 plus who writes (`<actor>/<script>` in `created_by`).
pub struct Registry<'a> {
    pub d1: &'a D1<'a>,
    pub clock: &'a dyn Clock,
    /// `fv_actor`.
    pub actor: String,
    /// The program word of `created_by` (`fv-deploy`; the scripts write
    /// their own name).
    pub program: String,
    /// `FV_RELEASE_CHANNEL`: the channel a deployment follows, when set.
    pub release_channel: Option<String>,
    /// `FV_REGISTRY` is not `0`: deployments are written.
    pub deployments_on: bool,
}

/// `program` in `created_by`.
pub const PROGRAM: &str = "fv-deploy";

impl<'a> Registry<'a> {
    pub fn new(d1: &'a D1<'a>, clock: &'a dyn Clock, env: &BootEnv) -> Self {
        Self {
            d1,
            clock,
            actor: actor(env),
            program: PROGRAM.into(),
            release_channel: env.get("FV_RELEASE_CHANNEL").map(str::to_owned),
            deployments_on: env.raw("FV_REGISTRY") != Some("0"),
        }
    }

    /// Unix milliseconds, whole seconds (`$(date +%s) * 1000`).
    pub fn now_ms(&self) -> i64 {
        (self.clock.now().floor() as i64) * 1000
    }

    // ------------------------------------------------------------- releases

    /// The channel's current release (`fv_release_current`).
    pub fn current(&self, channel: &str) -> Result<Option<Value>, D1Error> {
        Ok(self
            .d1
            .query(SQL_RELEASE_CURRENT, &[json!(channel)])?
            .into_iter()
            .next())
    }

    /// The current release of every channel (`fv_release_heads`).
    pub fn heads(&self) -> Result<Vec<Value>, D1Error> {
        self.d1.query(SQL_RELEASE_HEADS, &[])
    }

    /// Inserts a release; its id (`fv_release_insert`).
    pub fn insert(&self, r: &ReleaseRow) -> Result<i64, D1Error> {
        let opt = |s: &Option<String>| s.as_deref().map_or(Value::Null, Value::from);
        let params = [
            json!(r.channel),
            json!(r.git_sha),
            json!(r.digests.to_json()),
            json!(r.action),
            json!(self.now_ms()),
            json!(self.actor),
            opt(&r.notes),
            opt(&r.run_url),
            json!(i64::from(r.templates_updated)),
            r.source_release.map_or(Value::Null, Value::from),
        ];
        let rows = self.d1.query(SQL_RELEASE_INSERT, &params)?;
        rows.first()
            .and_then(|r| r.get("id"))
            .and_then(Value::as_i64)
            .ok_or_else(|| D1Error("the insert returned no id".into()))
    }

    /// The release naming a digest (`fv_release_for_digest`).
    pub fn for_digest(&self, digest: &str) -> Result<Option<DigestHit>, D1Error> {
        let rows = self
            .d1
            .query(SQL_RELEASE_FOR_DIGEST, &[json!(format!("%{digest}%"))])?;
        let heads = self.heads().unwrap_or_default();
        let Some(r) = rows.first() else {
            return Ok(None);
        };
        let suffix = format!("@{digest}");
        let names = |row: &Value| -> Digests {
            row.get("digests")
                .and_then(Value::as_str)
                .and_then(|t| Digests::parse(t).ok())
                .unwrap_or_default()
        };
        let key = names(r)
            .0
            .into_iter()
            .find(|(_, v)| v.ends_with(&suffix))
            .map_or(Value::Null, |(k, _)| Value::from(k));
        let channels = heads
            .iter()
            .filter(|h| names(h).0.iter().any(|(_, v)| v.ends_with(&suffix)))
            .filter_map(|h| h.get("channel").and_then(Value::as_str).map(str::to_owned))
            .collect();
        Ok(Some(DigestHit {
            git_sha: r.get("git_sha").cloned().unwrap_or(Value::Null),
            key,
            channels,
        }))
    }

    // ---------------------------------------------------------- deployments

    /// Whether deployments are written (`_fv_registry_on`), the schema
    /// applied first.
    fn on(&self) -> Result<bool, D1Error> {
        if !self.deployments_on {
            return Ok(false);
        }
        self.d1.ensure_schema()?;
        Ok(true)
    }

    /// Fills `digest` from the image and `git_sha` / `variant` / `channel`
    /// from the releases (`_fv_enrich`).
    pub fn enrich(&self, mut f: Fields) -> Fields {
        let img = get_str(&f, "image").to_owned();
        if get_str(&f, "digest").is_empty() && img.contains("@sha256:") {
            if let Some((_, d)) = img.rsplit_once('@') {
                f = merge(f, vec![("digest".into(), Value::from(d))]);
            }
        }
        let digest = get_str(&f, "digest").to_owned();
        if !digest.is_empty()
            && (get_str(&f, "git_sha").is_empty() || get_str(&f, "variant").is_empty())
        {
            if let Ok(Some(hit)) = self.for_digest(&digest) {
                let mut front: Fields =
                    vec![("git_sha".into(), hit.git_sha), ("variant".into(), hit.key)];
                if let Some(c) = hit.channels.first() {
                    front.push(("channel".into(), Value::from(c.as_str())));
                }
                f = merge(front, f);
            }
        }
        if let Some(c) = &self.release_channel {
            f = merge(vec![("channel".into(), Value::from(c.as_str()))], f);
        }
        f
    }

    /// The upsert of a new row (`_fv_deploy_upsert`): its statement and params.
    pub fn upsert_statement(
        &self,
        kind: &str,
        rid: &str,
        f: Fields,
        created_by: &str,
    ) -> (String, Vec<Value>) {
        let now = self.now_ms();
        let base: Fields = vec![
            ("id".into(), Value::from(format!("{kind}:{rid}"))),
            ("kind".into(), Value::from(kind)),
            ("runpod_id".into(), Value::from(rid)),
            ("created_at".into(), Value::from(now)),
            ("updated_at".into(), Value::from(now)),
            ("created_by".into(), Value::from(created_by)),
            ("status".into(), Value::from("creating")),
        ];
        let f = merge(base, f);
        let cols: Vec<&str> = f.iter().map(|(k, _)| k.as_str()).collect();
        let vals = vec!["?"; cols.len()].join(", ");
        let sets: Vec<String> = cols
            .iter()
            .filter(|k| !["id", "kind", "runpod_id", "created_at", "created_by"].contains(k))
            .map(|k| format!("{k} = excluded.{k}"))
            .collect();
        let sql = format!(
            "INSERT INTO deployments ({}) VALUES ({vals}) ON CONFLICT (id) DO UPDATE SET {}",
            cols.join(", "),
            sets.join(", ")
        );
        (sql, f.into_iter().map(|(_, v)| v).collect())
    }

    /// `fv_deploy_created`: a row `<kind>:<id>` (status `creating` unless the
    /// fields say otherwise). `who` overrides the actor (`reconcile`).
    pub fn created(
        &self,
        kind: &str,
        rid: &str,
        pairs: &[(&str, &str)],
        who: Option<&str>,
    ) -> Result<(), D1Error> {
        if !self.on()? {
            return Ok(());
        }
        let f = self.enrich(kv(pairs));
        let by = format!("{}/{}", who.unwrap_or(&self.actor), self.program);
        let (sql, params) = self.upsert_statement(kind, rid, f, &by);
        self.d1.query(&sql, &params).map(|_| ())
    }

    /// `fv_deploy_update`.
    pub fn update(&self, kind: &str, rid: &str, pairs: &[(&str, &str)]) -> Result<(), D1Error> {
        if !self.on()? {
            return Ok(());
        }
        let f = merge(
            kv(pairs),
            vec![("updated_at".into(), Value::from(self.now_ms()))],
        );
        let sets: Vec<String> = f.iter().map(|(k, _)| format!("{k} = ?")).collect();
        let mut params: Vec<Value> = f.into_iter().map(|(_, v)| v).collect();
        params.push(Value::from(format!("{kind}:{rid}")));
        self.d1
            .query(
                &format!("UPDATE deployments SET {} WHERE id = ?", sets.join(", ")),
                &params,
            )
            .map(|_| ())
    }

    /// `fv_deploy_ready`: status `ready`, then `ready_at` once.
    pub fn ready(&self, kind: &str, rid: &str, pairs: &[(&str, &str)]) -> Result<(), D1Error> {
        if !self.on()? {
            return Ok(());
        }
        let mut all = vec![("status", "ready")];
        all.extend_from_slice(pairs);
        self.update(kind, rid, &all)?;
        self.d1
            .query(
                SQL_DEPLOYMENT_READY_AT,
                &[json!(self.now_ms()), json!(format!("{kind}:{rid}"))],
            )
            .map(|_| ())
    }

    /// `fv_deploy_deleted`: `deleted_at` once, status `deleted` or `gone`.
    pub fn deleted(&self, kind: &str, rid: &str, status: &str) -> Result<(), D1Error> {
        if !self.on()? {
            return Ok(());
        }
        let now = self.now_ms();
        self.d1
            .query(
                SQL_DEPLOYMENT_DELETED,
                &[
                    json!(status),
                    json!(now),
                    json!(now),
                    json!(format!("{kind}:{rid}")),
                ],
            )
            .map(|_| ())
    }

    /// The live rows (`deleted_at IS NULL`).
    pub fn live_rows(&self) -> Result<Vec<Value>, D1Error> {
        self.d1.query(SQL_DEPLOYMENTS_LIVE, &[])
    }
}

// ------------------------------------------------------------------ up / down

/// The deployments kind and Runpod id of an fv-deploy instance; `None` for
/// Vast and GCP (the table holds Runpod resources: `reconcile` matches its
/// rows against the Runpod account and would mark any other one gone).
pub fn deployment_of(kind: ProviderKind, id: &InstanceId) -> Option<(&'static str, String)> {
    match kind {
        ProviderKind::RunpodPod => Some(("pod", id.as_str().to_owned())),
        ProviderKind::RunpodServerless => Some((
            "endpoint",
            id.as_str()
                .split_once('/')
                .map_or(id.as_str(), |(e, _)| e)
                .to_owned(),
        )),
        ProviderKind::Vast | ProviderKind::Gcp => None,
    }
}

/// The `key=value` fields `fv-deploy up` writes for a created instance:
/// what `runpod-pod.sh` / `runpod-endpoint.sh` write (name, image, gpu,
/// billed price, variant, data centre, the config in `meta`; an endpoint's
/// template, type and volume in `meta`), plus the region and the launch's
/// guards in `meta` (a guard `null` shown as "not checked").
pub fn created_fields(launch: &Launch, inst: &Instance) -> Vec<(&'static str, String)> {
    let config = launch
        .source
        .as_ref()
        .map(|s| s.config.clone())
        .unwrap_or_default();
    let variant = launch
        .source
        .as_ref()
        .map_or_else(|| "cuda".to_owned(), |s| s.image.clone());
    let dc = inst
        .data_center
        .clone()
        .or_else(|| launch.placement.data_centers.first().cloned())
        .unwrap_or_default();
    let mut meta = Map::new();
    let mut f: Vec<(&'static str, String)> =
        vec![("name", inst.name.clone()), ("image", launch.image.clone())];
    match launch.provider {
        ProviderKind::RunpodServerless => {
            let tpl = inst
                .id
                .as_str()
                .split_once('/')
                .map(|(_, t)| t.to_owned())
                .unwrap_or_default();
            f.push(("variant", variant));
            f.push(("dc", dc));
            f.push(("gpu", launch.gpu.types.join(",")));
            meta.insert("type".into(), json!("queue"));
            meta.insert("template".into(), json!(tpl));
            meta.insert("shared_template".into(), json!(false));
            if let Some(v) = &launch.placement.volume {
                meta.insert("volume".into(), json!(v.id));
            }
        }
        _ => {
            f.push((
                "gpu",
                inst.gpu
                    .clone()
                    .unwrap_or_else(|| launch.gpu.types.join(",")),
            ));
            if let Some(p) = inst.usd_per_hr {
                f.push(("cost_per_hr", p.to_string()));
            }
            f.push(("variant", variant));
            f.push(("dc", dc));
        }
    }
    if let Some(r) = &launch.placement.region {
        f.push(("region", r.clone()));
    }
    meta.insert("config".into(), json!(config));
    meta.insert("guards".into(), launch.guards.ledger_view());
    f.push(("meta", Value::Object(meta).to_string()));
    f
}

/// The [`Recorder`] of `fv-deploy up` / `down`: the D1 `deployments` rows of
/// Runpod pods and endpoints (Vast and GCP are not recorded there).
pub struct D1Recorder<'a> {
    pub registry: &'a Registry<'a>,
}

impl Recorder for D1Recorder<'_> {
    fn created(&self, launch: &Launch, inst: &Instance) -> Result<(), String> {
        let Some((kind, rid)) = deployment_of(inst.provider, &inst.id) else {
            return Ok(());
        };
        let f = created_fields(launch, inst);
        let pairs: Vec<(&str, &str)> = f.iter().map(|(k, v)| (*k, v.as_str())).collect();
        self.registry
            .created(kind, &rid, &pairs, None)
            .map_err(|e| e.to_string())
    }

    fn ready(&self, inst: &Instance) -> Result<(), String> {
        let Some((kind, rid)) = deployment_of(inst.provider, &inst.id) else {
            return Ok(());
        };
        self.registry
            .ready(kind, &rid, &[])
            .map_err(|e| e.to_string())
    }

    fn deleted(&self, kind: ProviderKind, id: &InstanceId, status: &str) -> Result<(), String> {
        let Some((k, rid)) = deployment_of(kind, id) else {
            return Ok(());
        };
        self.registry
            .deleted(k, &rid, status)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_is_jq_object_addition() {
        let l: Fields = vec![("a".into(), json!(1)), ("b".into(), json!(2))];
        let r: Fields = vec![("c".into(), json!(3)), ("a".into(), json!(9))];
        assert_eq!(
            merge(l, r),
            vec![
                ("a".to_owned(), json!(9)),
                ("b".to_owned(), json!(2)),
                ("c".to_owned(), json!(3))
            ]
        );
    }

    #[test]
    fn kv_drops_what_registry_sh_drops() {
        let f = kv(&[
            ("name", "n"),
            ("pool", ""),
            ("bogus", "x"),
            ("cost_per_hr", "1.84"),
            ("meta", "[1]"),
            ("gpu", "G"),
        ]);
        assert_eq!(
            f,
            vec![
                ("name".to_owned(), json!("n")),
                ("cost_per_hr".to_owned(), json!(1.84)),
                ("gpu".to_owned(), json!("G"))
            ]
        );
        assert!(kv(&[("cost_per_hr", "-1")]).is_empty());
    }

    #[test]
    fn digests_keep_their_order() {
        let d = Digests::parse(r#"{"debug":"r@sha256:1","cuda":"r@sha256:2","cpu":"r@sha256:3"}"#)
            .unwrap_or_default();
        assert_eq!(
            d.to_json(),
            r#"{"debug":"r@sha256:1","cuda":"r@sha256:2","cpu":"r@sha256:3"}"#
        );
        assert_eq!(d.sorted_keys(), vec!["cpu", "cuda", "debug"]);
        assert_eq!(
            d.canonical(),
            r#"{"cpu":"r@sha256:3","cuda":"r@sha256:2","debug":"r@sha256:1"}"#
        );
        assert!(Digests::parse("[]").is_err());
    }

    #[test]
    fn actor_follows_fv_actor() {
        let e = BootEnv::from_pairs([("FV_DEPLOYED_BY", "x")]);
        assert_eq!(actor(&e), "x");
        let e = BootEnv::from_pairs([
            ("GITHUB_ACTIONS", "true"),
            ("GITHUB_WORKFLOW", "release"),
            ("GITHUB_RUN_ID", "7"),
            ("GITHUB_ACTOR", "me"),
        ]);
        assert_eq!(actor(&e), "ci:release#7 by me");
        let e = BootEnv::from_pairs([("CLAUDECODE", "1"), ("USER", "u")]);
        assert_eq!(actor(&e), "agent:u");
        assert!(actor(&BootEnv::from_pairs([("USER", "u")])).starts_with("user:u@"));
    }
}
