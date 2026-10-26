//! `fv-deploy reconcile [--dry-run] [--adopt] [--fix]` (`release.sh
//! reconcile`): the account's pods, endpoints and templates against the D1
//! `deployments` ledger that the deploy scripts, fv-deploy and fv-control
//! share. A row whose resource is gone is marked `gone`; a live fv-serve
//! resource without a row is flagged `unknown` (`--adopt` records it); a row
//! whose image changed under it is flagged `drifted` (`--fix` records the
//! live image); a serve template whose image is not its channel's is
//! flagged. Only named fields of Runpod objects are shown, never their env.

use serde_json::{json, Value};

use super::registry::{Digests, Registry};
use super::{dshort, Ctx, ReleaseError};
use crate::boot::sys::Method;
use crate::launch::SERVE_REPO;
use crate::provider::runpod::{serve_template_name, RunpodApi};

/// The flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub dry_run: bool,
    pub adopt: bool,
    pub fix: bool,
}

/// What the account holds (REST v1 lists).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Account {
    pub pods: Vec<Value>,
    pub endpoints: Vec<Value>,
    pub templates: Vec<Value>,
}

/// `fetch_runpod`.
pub fn fetch(api: &RunpodApi<'_>) -> Result<Account, ReleaseError> {
    let list = |p: &str| -> Result<Vec<Value>, ReleaseError> {
        Ok(api
            .rest(Method::Get, p, None, &format!("Runpod: {}", &p[1..]))?
            .as_array()
            .cloned()
            .unwrap_or_default())
    };
    Ok(Account {
        pods: list("/pods")?,
        endpoints: list("/endpoints")?,
        templates: list("/templates")?,
    })
}

/// One live fv-serve resource (`live_resources`).
#[derive(Clone, Debug, PartialEq)]
pub struct Live {
    /// `pod`, `gateway` or `endpoint`.
    pub kind: &'static str,
    pub id: String,
    pub name: Option<String>,
    pub image: Option<String>,
    pub template: Option<String>,
    /// `desiredStatus`, `ENDPOINT` for an endpoint.
    pub status: Option<String>,
    pub gpu: Option<String>,
    pub dc: Option<String>,
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_owned)
}

fn joined(v: &Value, k: &str) -> String {
    v.get(k)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default()
}

/// A pod counts when it runs an image of the serve repository or is named
/// `fv-serve-*`, `fv-cluster-*` or `fv-gw-*`; an endpoint when its template's
/// image does or it is named `fv-*`.
pub fn live_resources(a: &Account, repo: &str) -> Vec<Live> {
    let fv_pod = |n: &str| {
        ["fv-serve-", "fv-cluster-", "fv-gw-"]
            .iter()
            .any(|p| n.starts_with(p))
    };
    let mut out = Vec::new();
    for p in &a.pods {
        let name = s(p, "name").unwrap_or_default();
        let image = s(p, "imageName").unwrap_or_default();
        if !(image.starts_with(repo) || fv_pod(&name)) {
            continue;
        }
        out.push(Live {
            kind: if name.contains("-gw-") || name.contains("gateway") {
                "gateway"
            } else {
                "pod"
            },
            id: s(p, "id").unwrap_or_default(),
            name: s(p, "name"),
            image: s(p, "imageName"),
            template: None,
            status: s(p, "desiredStatus"),
            gpu: p
                .pointer("/machine/gpuDisplayName")
                .or_else(|| p.pointer("/gpu/displayName"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            dc: p
                .pointer("/machine/dataCenterId")
                .or_else(|| p.get("dataCenterId"))
                .and_then(Value::as_str)
                .map(str::to_owned),
        });
    }
    for e in &a.endpoints {
        let tid = s(e, "templateId");
        let tp = tid.as_deref().and_then(|t| {
            a.templates
                .iter()
                .find(|x| x.get("id").and_then(Value::as_str) == Some(t))
        });
        let image = tp.and_then(|t| s(t, "imageName"));
        let name = s(e, "name").unwrap_or_default();
        if !(image.as_deref().unwrap_or("").starts_with(repo) || name.starts_with("fv-")) {
            continue;
        }
        out.push(Live {
            kind: "endpoint",
            id: s(e, "id").unwrap_or_default(),
            name: s(e, "name"),
            image,
            template: tid,
            status: Some("ENDPOINT".into()),
            gpu: Some(joined(e, "gpuTypeIds")),
            dc: Some(joined(e, "dataCenterIds")),
        });
    }
    out
}

/// A JSON value as jq's string interpolation shows it.
fn shown(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => "null".into(),
        Some(x) => x.to_string(),
    }
}

/// `reconcile`; the summary counts (gone, unknown, drifted).
pub fn reconcile(ctx: &Ctx<'_>, o: &Options) -> Result<(usize, usize, usize), ReleaseError> {
    let d1 = ctx.d1().ok_or_else(|| {
        super::refused("D1 is the release history: set FV_CF_API_TOKEN (or CLOUDFLARE_API_TOKEN)")
    })?;
    d1.ensure_schema()
        .map_err(|e| super::refused(format!("D1 schema: {e}")))?;
    let reg = Registry::new(&d1, ctx.clock, ctx.env);
    let api = ctx.runpod()?;
    let acct = fetch(&api)?;
    let repo = ctx.env.get("FV_SERVE_REPO").unwrap_or(SERVE_REPO);
    let live = live_resources(&acct, repo);
    let ids: Vec<&str> = acct
        .pods
        .iter()
        .chain(&acct.endpoints)
        .filter_map(|x| x.get("id").and_then(Value::as_str))
        .collect();
    let rows = reg.live_rows()?;
    let (mut gone, mut unknown, mut drift) = (0, 0, 0);
    for r in &rows {
        let rid = r.get("runpod_id").and_then(Value::as_str).unwrap_or("");
        if ids.contains(&rid) {
            continue;
        }
        gone += 1;
        let kind = r.get("kind").and_then(Value::as_str).unwrap_or("");
        let name = r.get("name").and_then(Value::as_str).unwrap_or("");
        ctx.print(&format!(
            "gone      {kind} {rid} {name}  (created by {})",
            shown(r.get("created_by"))
        ));
        if !o.dry_run {
            if let Err(e) = reg.deleted(kind, rid, "gone") {
                ctx.say(&format!("registry: could not mark {kind} {rid} gone: {e}"));
            }
        }
    }
    for l in &live {
        let cur = rows
            .iter()
            .find(|r| r.get("runpod_id").and_then(Value::as_str) == Some(l.id.as_str()));
        let image = l.image.clone().unwrap_or_default();
        match cur {
            None => {
                unknown += 1;
                ctx.print(&format!(
                    "unknown   {} {} {} {image}",
                    l.kind,
                    l.id,
                    l.name.as_deref().unwrap_or("")
                ));
                if o.adopt && !o.dry_run {
                    let status = match l.status.as_deref() {
                        Some("RUNNING" | "ENDPOINT") => "ready",
                        _ => "creating",
                    };
                    let mut meta = json!({"adopted": true}).to_string();
                    if let Some(t) = &l.template {
                        meta = format!(
                            "{{\"adopted\":true,\"template\":{}}}",
                            Value::from(t.as_str())
                        );
                    }
                    let pairs = [
                        ("name", l.name.as_deref().unwrap_or("")),
                        ("image", image.as_str()),
                        ("gpu", l.gpu.as_deref().unwrap_or("")),
                        ("dc", l.dc.as_deref().unwrap_or("")),
                        ("status", status),
                        ("meta", meta.as_str()),
                    ];
                    if let Err(e) = reg.created(l.kind, &l.id, &pairs, Some("reconcile")) {
                        ctx.say(&format!(
                            "registry: could not record {} {}: {e}",
                            l.kind, l.id
                        ));
                    }
                }
            }
            Some(cur) => {
                let recorded = cur.get("image").and_then(Value::as_str).unwrap_or("");
                if !image.is_empty() && recorded != image {
                    drift += 1;
                    ctx.print(&format!(
                        "drifted   {} {} recorded {} live {image}",
                        l.kind,
                        l.id,
                        if recorded.is_empty() { "-" } else { recorded }
                    ));
                    if o.fix && !o.dry_run {
                        let digest = image.split('@').nth(1).unwrap_or("");
                        if let Err(e) = reg.update(
                            l.kind,
                            &l.id,
                            &[("image", image.as_str()), ("digest", digest)],
                        ) {
                            ctx.say(&format!(
                                "registry: could not update {} {}: {e}",
                                l.kind, l.id
                            ));
                        }
                    }
                }
            }
        }
    }
    // The serve templates against the channel they follow.
    let tc = ctx.template_channel();
    let heads = reg.heads()?;
    let want = heads
        .iter()
        .find(|h| h.get("channel").and_then(Value::as_str) == Some(tc.as_str()))
        .and_then(|h| h.get("digests").and_then(Value::as_str))
        .and_then(|t| Digests::parse(t).ok())
        .unwrap_or_default();
    for img in &ctx.images {
        for &f in img.flavours() {
            let name = serve_template_name(&img.id, f);
            let cur = acct
                .templates
                .iter()
                .find(|t| t.get("name").and_then(Value::as_str) == Some(name.as_str()))
                .and_then(|t| t.get("imageName").and_then(Value::as_str))
                .filter(|s| !s.is_empty());
            let Some(cur) = cur else {
                ctx.print(&format!("template  {name} missing"));
                continue;
            };
            if !want.is_empty() && want.get(&img.id).unwrap_or("") != cur {
                drift += 1;
                ctx.print(&format!(
                    "template  {name} {} != {tc} {}",
                    dshort(cur),
                    dshort(want.get(&img.id).unwrap_or("-"))
                ));
            }
        }
    }
    ctx.print(&format!(
        "reconcile: {gone} gone{}, {unknown} unknown{}, {drift} drifted",
        if o.dry_run {
            " (not marked: dry run)"
        } else {
            ""
        },
        if o.adopt && !o.dry_run {
            " (adopted)"
        } else {
            ""
        }
    ));
    Ok((gone, unknown, drift))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_fv_resources_are_live() {
        let a = Account {
            pods: vec![
                json!({"id": "p1", "name": "fv-serve-x", "imageName": "other/img", "desiredStatus": "RUNNING"}),
                json!({"id": "p2", "name": "someone", "imageName": "daydream/scope"}),
                json!({"id": "p3", "name": "fv-gw-1", "imageName": "r/serve@sha256:1"}),
            ],
            endpoints: vec![
                json!({"id": "e1", "name": "x", "templateId": "t1", "gpuTypeIds": ["A", "B"]}),
                json!({"id": "e2", "name": "y", "templateId": "t2"}),
            ],
            templates: vec![
                json!({"id": "t1", "name": "fv-serve-cuda-sls", "imageName": "r/serve@sha256:2"}),
                json!({"id": "t2", "name": "other", "imageName": "z"}),
            ],
        };
        let l = live_resources(&a, "r/serve");
        let got: Vec<(&str, &str)> = l.iter().map(|x| (x.kind, x.id.as_str())).collect();
        assert_eq!(
            got,
            vec![("pod", "p1"), ("gateway", "p3"), ("endpoint", "e1")]
        );
        assert_eq!(l[2].gpu.as_deref(), Some("A,B"));
        assert_eq!(l[2].image.as_deref(), Some("r/serve@sha256:2"));
    }
}
