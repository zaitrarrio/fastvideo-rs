//! `fv-deploy templates sync | sync-all | list | id | plan`: the Runpod serve
//! templates (`scripts/serve/runpod-templates.sh`). One per image and
//! flavour, `fv-serve-<image>-sls` (serverless queue worker) and
//! `fv-serve-<image>-pod` (HTTP pod; the CPU image has only this one),
//! booting a digest-pinned image with no worker config (a launch sends one,
//! ADR-0011). They follow one release channel (`FV_TEMPLATE_CHANNEL`,
//! default `stable`): `release promote` / `rollback` sync them. Updating a
//! template's image rolls every endpoint that uses it (Runpod rolling
//! release). Templates carry no secret value: the D1 / R2 / webhook values
//! are Runpod secret references.

use serde_json::Value;

use super::{refused, Ctx, ReleaseError};
use crate::boot::sys::Method;
use crate::provider::runpod::{
    serve_template_body, serve_template_launch, serve_template_name, Flavour, RunpodApi,
};

/// `RUNPOD_REGISTRY_AUTH_ID`: a Runpod container registry auth (a private
/// registry; the GHCR package is public, so none is set).
pub const REGISTRY_AUTH_VAR: &str = "RUNPOD_REGISTRY_AUTH_ID";

/// The REST v1 body of one serve template (`payload`), `create` with the
/// fields only a create takes.
pub fn body(
    ctx: &Ctx<'_>,
    image_id: &str,
    f: Flavour,
    image: &str,
    channel: Option<&str>,
    create: bool,
) -> Result<Value, ReleaseError> {
    let img = ctx
        .image(image_id)
        .ok_or_else(|| unknown_image(ctx, image_id))?;
    let l = serve_template_launch(image_id, f, img.cpu, image, channel);
    Ok(serve_template_body(
        &l,
        image_id,
        f,
        img.cpu,
        ctx.env.get(REGISTRY_AUTH_VAR),
        create,
    ))
}

fn unknown_image(ctx: &Ctx<'_>, id: &str) -> ReleaseError {
    let known: Vec<&str> = ctx.images.iter().map(|i| i.id.as_str()).collect();
    refused(format!(
        "unknown serve image {id} (known: {})",
        known.join(" ")
    ))
}

/// Every template of the account (`GET /templates`).
pub fn all(api: &RunpodApi<'_>) -> Result<Vec<Value>, ReleaseError> {
    let v = api.rest(Method::Get, "/templates", None, "GET templates")?;
    Ok(v.as_array().cloned().unwrap_or_default())
}

/// The template named `name` (`fv_template_get`).
pub fn by_name(api: &RunpodApi<'_>, name: &str) -> Result<Option<Value>, ReleaseError> {
    Ok(all(api)?
        .into_iter()
        .find(|t| t.get("name").and_then(Value::as_str) == Some(name)))
}

fn env_str<'v>(t: &'v Value, k: &str) -> &'v str {
    t.pointer(&format!("/env/{k}"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// `sync <image> <ref>` (`sync_one`): create or update the image's templates
/// to boot `image` (a digest reference) on `channel` (`FV_RELEASE_CHANNEL`);
/// a template already on that image, channel and identity is left alone.
pub fn sync(
    ctx: &Ctx<'_>,
    api: &RunpodApi<'_>,
    image_id: &str,
    image: &str,
    channel: Option<&str>,
) -> Result<(), ReleaseError> {
    let img = ctx
        .image(image_id)
        .ok_or_else(|| unknown_image(ctx, image_id))?
        .clone();
    if !image.contains("@sha256:") {
        return Err(refused(format!(
            "{image_id}: pass a digest reference, not {image}"
        )));
    }
    for &f in img.flavours() {
        let name = serve_template_name(image_id, f);
        match by_name(api, &name)? {
            Some(cur) => {
                let id = cur
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let cur_channel = env_str(&cur, "FV_RELEASE_CHANNEL");
                let want_channel = channel.filter(|c| !c.is_empty()).unwrap_or(cur_channel);
                if cur.get("imageName").and_then(Value::as_str) == Some(image)
                    && cur_channel == want_channel
                    && env_str(&cur, "FV_IMAGE_REF") == image
                {
                    ctx.say(&format!("{name} ({id}): already {image}"));
                    continue;
                }
                let b = body(ctx, image_id, f, image, channel, false)?;
                api.rest(
                    Method::Patch,
                    &format!("/templates/{id}"),
                    Some(&b),
                    "PATCH template",
                )
                .map_err(|e| refused(format!("{name}: update failed: {e}")))?;
                ctx.say(&format!("{name} ({id}): updated to {image}"));
            }
            None => {
                let b = body(ctx, image_id, f, image, channel, true)?;
                let r = api
                    .rest(Method::Post, "/templates", Some(&b), "POST templates")
                    .map_err(|e| refused(format!("{name}: create failed: {e}")))?;
                ctx.say(&format!(
                    "{name} ({}): created for {image}",
                    r.get("id").and_then(Value::as_str).unwrap_or("?")
                ));
            }
        }
    }
    Ok(())
}

/// `sync-all <file>`: `sync` for every `<image> <ref> …` line (blank lines and
/// `#` comments skipped).
pub fn sync_all(
    ctx: &Ctx<'_>,
    api: &RunpodApi<'_>,
    text: &str,
    channel: Option<&str>,
) -> Result<(), ReleaseError> {
    for line in text.lines() {
        let mut w = line.split_whitespace();
        let (Some(v), img) = (w.next(), w.next()) else {
            continue;
        };
        if v.starts_with('#') {
            continue;
        }
        sync(ctx, api, v, img.unwrap_or(""), channel)?;
    }
    Ok(())
}

/// `list`: the serve templates (`fv-serve-*-sls|pod`) as `id\tname\timage`.
pub fn list(api: &RunpodApi<'_>) -> Result<Vec<String>, ReleaseError> {
    Ok(all(api)?
        .iter()
        .filter_map(|t| {
            let name = t.get("name").and_then(Value::as_str)?;
            (name.starts_with("fv-serve-") && (name.ends_with("-sls") || name.ends_with("-pod")))
                .then(|| {
                    format!(
                        "{}\t{name}\t{}",
                        t.get("id").and_then(Value::as_str).unwrap_or(""),
                        t.get("imageName").and_then(Value::as_str).unwrap_or("")
                    )
                })
        })
        .collect())
}

/// The image a serve template boots now (`template_image`), `None` when it
/// does not exist or cannot be read.
pub fn current_image(api: &RunpodApi<'_>, image_id: &str, f: Flavour) -> Option<String> {
    by_name(api, &serve_template_name(image_id, f))
        .ok()
        .flatten()
        .and_then(|t| {
            t.get("imageName")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}
