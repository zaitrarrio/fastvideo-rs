//! `fv-deploy release promote | rollback | resolve` (`release.sh`): a
//! release is a build's image set pointed at by a channel's tags
//! (`:<channel>`, `:<image>-<channel>`, and for `latest` the plain
//! `:<image>`), the Runpod templates when the channel is theirs, and one row
//! in D1's `releases` (the history `rollback` reads).
//!
//! Where it runs, as `release.sh`: in GitHub Actions (or with `--local`) it
//! retags with `docker buildx imagetools create` (a GHCR login with
//! `packages:write`), calls Runpod and D1 here; elsewhere it dispatches
//! `.github/workflows/release.yml` (`GH_TOKEN` / `GITHUB_TOKEN` with
//! `actions:write`) unless `--local`. `--dry-run` prints the plan and changes
//! nothing.

use std::time::Duration;

use serde_json::{json, Value};

use super::ghcr::{key_channel_tags, Ghcr};
use super::registry::{
    Digests, Registry, ReleaseRow, SQL_RELEASE_BY_ID, SQL_RELEASE_PREVIOUS, SQL_RELEASE_ROLLED_BACK,
};
use super::templates;
use super::{check_channel, dshort, refused, release_keys, short, Ctx, ReleaseError};
use crate::boot::sys::{Method, Request};
use crate::provider::runpod::serve_template_name;

/// The GitHub API (`FV_GITHUB_API`).
pub const GITHUB_API: &str = "https://api.github.com";
/// The repository whose release workflow is dispatched (`FV_GITHUB_REPO`).
pub const GITHUB_REPO: &str = "zaitrarrio/fastvideo-rs";

/// Where a promote / rollback runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Here: retag, templates, D1.
    Local,
    /// Dispatch `release.yml`, which runs it in GitHub Actions.
    Dispatch,
}

/// The flags of `release.sh` that promote and rollback take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub dry_run: bool,
    /// `--local` / `--dispatch`; `None`: local in GitHub Actions, else dispatch.
    pub mode: Option<Mode>,
    pub notes: Option<String>,
    /// `--no-templates` clears it.
    pub templates: bool,
    pub allow_partial: bool,
    pub force: bool,
    /// `rollback --to <release id>`.
    pub to: Option<i64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            dry_run: false,
            mode: None,
            notes: None,
            templates: true,
            allow_partial: false,
            force: false,
            to: None,
        }
    }
}

impl Options {
    fn mode(&self, ctx: &Ctx<'_>) -> Mode {
        self.mode
            .unwrap_or(if ctx.env.get("GITHUB_ACTIONS") == Some("true") {
                Mode::Local
            } else {
                Mode::Dispatch
            })
    }
}

/// A build's image set (`resolve`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    /// The full sha when an image's revision label gives it, else the short one.
    pub sha: String,
    pub short: String,
    pub digests: Digests,
}

impl Resolved {
    pub fn to_json(&self) -> Value {
        let d: Value = serde_json::from_str(&self.digests.to_json()).unwrap_or(Value::Null);
        json!({"sha": self.sha, "short": self.short, "digests": d})
    }
}

fn is_hex_sha(s: &str, min: usize) -> bool {
    (min..=40).contains(&s.len())
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

fn ghcr<'a>(ctx: &Ctx<'a>) -> Result<Ghcr<'a>, ReleaseError> {
    let g = Ghcr::from_env(ctx.http, ctx.env);
    g.token()
        .map_err(|_| refused(format!("cannot read {}", g.repo)))?;
    Ok(g)
}

/// `<sha|digest|tag>` → the build's image set: a 7–40 hex sha, else a digest
/// or tag whose image's revision label names the build; every image of the
/// set must carry that revision.
pub fn resolve(ctx: &Ctx<'_>, target: &str) -> Result<Resolved, ReleaseError> {
    let g = ghcr(ctx)?;
    let s = if is_hex_sha(target, 7) {
        short(target).to_owned()
    } else {
        let d = if let Some((_, hex)) = target.rsplit_once("sha256:") {
            format!("sha256:{hex}")
        } else {
            let tag = target.rsplit_once(':').map_or(target, |(_, t)| t);
            g.digest(tag)?
                .ok_or_else(|| refused(format!("no tag {tag} in {}", g.repo)))?
        };
        let rev = g
            .revision(&d)
            .map_err(|_| refused(format!("cannot read the labels of {d}")))?
            .unwrap_or_default();
        if !is_hex_sha(&rev, 7) {
            return Err(refused(format!("{target}: no git revision label")));
        }
        short(&rev).to_owned()
    };
    let keys = release_keys(&ctx.images);
    let imgs = g.set_for_sha(&s, &keys)?;
    if imgs.is_empty() {
        return Err(refused(format!(
            "no image tagged sha-{s} or <image>-sha-{s} in {}",
            g.repo
        )));
    }
    let mut rev = String::new();
    for k in imgs.sorted_keys() {
        let want = imgs.get(k).unwrap_or("");
        let digest = want.rsplit_once('@').map_or(want, |(_, d)| d);
        let got = g.revision(digest).ok().flatten().unwrap_or_default();
        if !(got.is_empty() || got == "unknown" || got.starts_with(&s)) {
            return Err(refused(format!("{k} ({want}) is labelled {got}, not {s}")));
        }
        if rev.is_empty() && got.len() == 40 && is_hex_sha(&got, 40) {
            rev = got;
        }
    }
    Ok(Resolved {
        sha: if rev.is_empty() { s.clone() } else { rev },
        short: s,
        digests: imgs,
    })
}

/// The release keys the set has no image for (`missing_keys`).
pub fn missing_keys(ctx: &Ctx<'_>, d: &Digests) -> Vec<String> {
    release_keys(&ctx.images)
        .into_iter()
        .filter(|k| d.get(k).is_none())
        .collect()
}

fn need_d1<'a>(ctx: &Ctx<'a>) -> Result<super::d1::D1<'a>, ReleaseError> {
    let d1 = ctx.d1().ok_or_else(|| {
        refused("D1 is the release history: set FV_CF_API_TOKEN (or CLOUDFLARE_API_TOKEN)")
    })?;
    d1.ensure_schema()
        .map_err(|e| refused(format!("D1 schema: {e}")))?;
    Ok(d1)
}

/// `release.yml`'s dispatch body (`dispatch`).
pub fn dispatch_body(
    ctx: &Ctx<'_>,
    action: &str,
    target: &str,
    channel: &str,
    o: &Options,
) -> Value {
    json!({
        "ref": ctx.env.get("FV_RELEASE_REF").unwrap_or("main"),
        "inputs": {
            "action": action,
            "target": target,
            "channel": channel,
            "notes": o.notes.clone().unwrap_or_default(),
            "to": o.to.map(|t| t.to_string()).unwrap_or_default(),
            "templates": if o.templates { "true" } else { "false" },
            "allow_partial": if o.allow_partial { "true" } else { "false" },
        }
    })
}

fn dispatch(
    ctx: &Ctx<'_>,
    action: &str,
    target: &str,
    channel: &str,
    o: &Options,
) -> Result<(), ReleaseError> {
    let token = ctx
        .env
        .get("GH_TOKEN")
        .or_else(|| ctx.env.get("GITHUB_TOKEN"))
        .ok_or_else(|| refused("dispatching the release workflow needs GH_TOKEN (actions:write); or run with --local"))?;
    let api = ctx
        .env
        .get("FV_GITHUB_API")
        .unwrap_or(GITHUB_API)
        .trim_end_matches('/');
    let repo = ctx.env.get("FV_GITHUB_REPO").unwrap_or(GITHUB_REPO);
    let req = Request::new(
        Method::Post,
        format!("{api}/repos/{repo}/actions/workflows/release.yml/dispatches"),
        Duration::from_secs(30),
    )
    .bearer(token)
    .header("accept", "application/vnd.github+json")
    .json(dispatch_body(ctx, action, target, channel, o).to_string());
    let code = match ctx.http.send(&req) {
        Ok(r) => r.status.to_string(),
        Err(_) => "000".into(),
    };
    if code != "204" {
        return Err(refused(format!("workflow dispatch answered HTTP {code}")));
    }
    ctx.print(&format!(
        "dispatched release.yml: {action} {target} -> {channel}"
    ));
    ctx.print(&format!(
        "https://github.com/{repo}/actions/workflows/release.yml"
    ));
    Ok(())
}

/// `promote <sha|digest|tag> [channel]`.
pub fn promote(
    ctx: &Ctx<'_>,
    target: &str,
    channel: &str,
    o: &Options,
) -> Result<(), ReleaseError> {
    check_channel(channel, &ctx.images)?;
    if o.mode(ctx) == Mode::Dispatch && !o.dry_run {
        return dispatch(ctx, "promote", target, channel, o);
    }
    let d1 = if o.dry_run {
        ctx.d1()
    } else {
        Some(need_d1(ctx)?)
    };
    let rel = resolve(ctx, target)?;
    let miss = missing_keys(ctx, &rel.digests);
    if !miss.is_empty() {
        if !o.allow_partial {
            return Err(refused(format!(
                "build {} has no image for: {} (--allow-partial promotes the rest)",
                rel.short,
                miss.join(" ")
            )));
        }
        ctx.say(&format!(
            "WARNING: no image for {}; those keep their current :{channel}",
            miss.join(" ")
        ));
    }
    if let Some(d1) = &d1 {
        let reg = Registry::new(d1, ctx.clock, ctx.env);
        if let Ok(Some(cur)) = reg.current(channel) {
            ctx.say(&format!(
                "current {channel}: release {} = {} ({})",
                cur.get("id").cloned().unwrap_or(Value::Null),
                short(cur.get("git_sha").and_then(Value::as_str).unwrap_or("")),
                cur.get("action").and_then(Value::as_str).unwrap_or("")
            ));
            let cur_digests = cur
                .get("digests")
                .and_then(Value::as_str)
                .and_then(|t| Digests::parse(t).ok())
                .unwrap_or_default();
            if cur_digests.canonical() == rel.digests.canonical() && !o.force {
                ctx.print(&format!(
                    "{channel} is already {} (--force re-applies)",
                    rel.short
                ));
                return Ok(());
            }
        }
    }
    apply(
        ctx,
        d1.as_ref(),
        channel,
        &rel.digests,
        "promote",
        &rel.sha,
        None,
        o,
    )
}

/// `rollback [channel]`: re-promote the channel's previous release (or
/// `--to <id>`); refused when that release lacks an image the channel has,
/// unless `--allow-partial`.
pub fn rollback(ctx: &Ctx<'_>, channel: &str, o: &Options) -> Result<(), ReleaseError> {
    check_channel(channel, &ctx.images)?;
    if o.mode(ctx) == Mode::Dispatch && !o.dry_run {
        return dispatch(ctx, "rollback", "", channel, o);
    }
    let d1 = need_d1(ctx)?;
    let reg = Registry::new(&d1, ctx.clock, ctx.env);
    let cur = reg
        .current(channel)?
        .ok_or_else(|| refused(format!("no release recorded for {channel}")))?;
    let field = |r: &Value, k: &str| r.get(k).cloned().unwrap_or(Value::Null);
    let target = match o.to {
        Some(to) => d1
            .query(SQL_RELEASE_BY_ID, &[json!(to), json!(channel)])?
            .into_iter()
            .next()
            .ok_or_else(|| refused(format!("no release {to} in {channel}")))?,
        None => d1
            .query(
                SQL_RELEASE_PREVIOUS,
                &[field(&cur, "channel"), field(&cur, "id"), field(&cur, "git_sha")],
            )?
            .into_iter()
            .next()
            .ok_or_else(|| {
                refused(format!(
                    "{channel} has no earlier release to roll back to (history: release.sh history --channel {channel})"
                ))
            })?,
    };
    let id = |r: &Value| r.get("id").and_then(Value::as_i64).unwrap_or(0);
    let sha = |r: &Value| {
        r.get("git_sha")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned()
    };
    ctx.say(&format!(
        "rollback {channel}: release {} ({}) -> release {} ({})",
        id(&cur),
        short(&sha(&cur)),
        id(&target),
        short(&sha(&target))
    ));
    let digests = target
        .get("digests")
        .and_then(Value::as_str)
        .map(Digests::parse)
        .transpose()
        .map_err(refused)?
        .unwrap_or_default();
    let miss = missing_keys(ctx, &digests);
    if !miss.is_empty() {
        let stay: String = miss
            .iter()
            .map(|k| format!(" :{}", key_channel_tags(k, channel)[0]))
            .collect();
        if !o.allow_partial {
            return Err(refused(format!(
                "release {} has no image for: {}; a rollback would leave{stay} (and its templates) on the current build (an older release predates those images: roll the deployments back instead, docs/serve/images.md; --allow-partial moves the rest)",
                id(&target),
                miss.join(" ")
            )));
        }
        ctx.say(&format!(
            "WARNING: release {} has no image for {}: {} and its templates stay on the current build",
            id(&target),
            miss.join(" "),
            stay.trim_start()
        ));
    }
    // The recorded digests must still exist (GHCR retention).
    let g = ghcr(ctx)?;
    for k in digests.sorted_keys() {
        let r = digests.get(k).unwrap_or("");
        let d = r.rsplit_once('@').map_or(r, |(_, d)| d);
        if g.digest(d)?.is_none() {
            return Err(refused(format!("{k}: {r} is gone from the registry")));
        }
    }
    apply(
        ctx,
        Some(&d1),
        channel,
        &digests,
        "rollback",
        &sha(&target),
        Some(id(&target)),
        o,
    )
}

/// The shared half of promote and rollback (`apply`): the channel tags, the
/// templates when the channel is theirs, the release row.
#[allow(clippy::too_many_arguments)]
fn apply(
    ctx: &Ctx<'_>,
    d1: Option<&super::d1::D1<'_>>,
    channel: &str,
    digests: &Digests,
    action: &str,
    sha: &str,
    source: Option<i64>,
    o: &Options,
) -> Result<(), ReleaseError> {
    ctx.say(&format!(
        "{action} {} -> {channel}",
        sha.get(..12).unwrap_or(sha)
    ));
    let docker = ctx.env.get("FV_DOCKER").unwrap_or("docker");
    let g = Ghcr::from_env(ctx.http, ctx.env);
    for k in digests.sorted_keys() {
        let r = digests.get(k).unwrap_or("");
        for tag in key_channel_tags(k, channel) {
            if o.dry_run {
                ctx.print(&format!("  retag  :{tag} -> {r}"));
            } else {
                g.retag(ctx.spawner, docker, r, &tag).map_err(|e| {
                    refused(format!(
                        "retag :{tag} failed (nothing recorded; rerun to finish): {e}"
                    ))
                })?;
                ctx.say(&format!("  :{tag} -> {}", dshort(r)));
            }
        }
    }
    let tc = ctx.template_channel();
    let mut templates_updated = false;
    if o.templates && channel == tc {
        let api = ctx.runpod().ok();
        for img in ctx.images.clone() {
            let Some(r) = digests.get(&img.id) else {
                continue;
            };
            if o.dry_run {
                for &f in img.flavours() {
                    let cur = api
                        .as_ref()
                        .and_then(|a| templates::current_image(a, &img.id, f))
                        .unwrap_or_default();
                    let name = serve_template_name(&img.id, f);
                    if cur == r {
                        ctx.print(&format!(
                            "  template {name}: image already {}; env gains FV_RELEASE_CHANNEL={channel}",
                            dshort(r)
                        ));
                    } else {
                        ctx.print(&format!(
                            "  template {name}: {} -> {r}",
                            if cur.is_empty() { "?" } else { cur.as_str() }
                        ));
                    }
                }
            } else {
                let api = api.as_ref().ok_or_else(|| {
                    refused(format!(
                        "RUNPOD_API_KEY missing (the templates follow {tc}; --no-templates skips them)"
                    ))
                })?;
                templates::sync(ctx, api, &img.id, r, Some(channel))
                    .map_err(|e| refused(format!("template sync {} failed: {e}", img.id)))?;
            }
        }
        templates_updated = true;
    } else if channel == tc {
        ctx.say("  templates left as they are (--no-templates)");
    } else {
        ctx.say(&format!(
            "  templates follow {tc}, not {channel}: unchanged"
        ));
    }
    if o.dry_run {
        ctx.print("(dry run: nothing changed)");
        return Ok(());
    }
    let d1 = d1.ok_or_else(|| {
        refused("D1 is the release history: set FV_CF_API_TOKEN (or CLOUDFLARE_API_TOKEN)")
    })?;
    let reg = Registry::new(d1, ctx.clock, ctx.env);
    if action == "rollback" {
        if let Some(src) = source {
            if let Err(e) = d1.query(
                SQL_RELEASE_ROLLED_BACK,
                &[json!(reg.now_ms()), json!(channel), json!(sha), json!(src)],
            ) {
                ctx.say(&format!(
                    "WARNING: could not mark the rolled-back releases: {e}"
                ));
            }
        }
    }
    let run_url = ctx.env.get("GITHUB_SERVER_URL").map(|s| {
        format!(
            "{s}/{}/actions/runs/{}",
            ctx.env.get("GITHUB_REPOSITORY").unwrap_or(""),
            ctx.env.get("GITHUB_RUN_ID").unwrap_or("")
        )
    });
    let id = reg
        .insert(&ReleaseRow {
            channel: channel.to_owned(),
            git_sha: sha.to_owned(),
            digests: digests.clone(),
            action: action.to_owned(),
            notes: o.notes.clone().filter(|n| !n.is_empty()),
            run_url,
            templates_updated,
            source_release: source,
        })
        .map_err(|e| {
            refused(format!(
                "the images moved but the release was not recorded in D1: {e}"
            ))
        })?;
    ctx.print(&format!(
        "release {id}: {channel} = {} ({action})",
        short(sha)
    ));
    if let Some(p) = ctx.env.get("GITHUB_STEP_SUMMARY") {
        let mut s = format!(
            "### {action}: `{channel}` = `{}` (release {id})\n\n| image | digest |\n|---|---|\n",
            short(sha)
        );
        for (k, v) in &digests.0 {
            s.push_str(&format!("| {k} | `{v}` |\n"));
        }
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
        {
            let _ = f.write_all(s.as_bytes());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shas_are_7_to_40_lower_hex() {
        assert!(is_hex_sha("abcdef0", 7));
        assert!(!is_hex_sha("abcdef", 7));
        assert!(!is_hex_sha("ABCDEF0", 7));
        assert!(!is_hex_sha(&"a".repeat(41), 7));
        assert!(!is_hex_sha("latest", 7));
    }
}
