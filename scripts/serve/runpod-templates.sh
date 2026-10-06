#!/usr/bin/env bash
# Publish the per-variant serve images to Runpod as templates (docs/serve/images.md).
#
#   runpod-templates.sh sync <variant> <image>   create or update fv-serve-<variant>-sls
#                                                (serverless queue worker) and
#                                                fv-serve-<variant>-pod (HTTP pod) to boot
#                                                <image> (a digest: ghcr.io/…@sha256:…)
#   runpod-templates.sh sync-all <file>          the same for every "<variant> <image> …" line
#   runpod-templates.sh list                     the fv-serve-* templates (id, name, image)
#   runpod-templates.sh id <variant> <sls|pod>   print one template id
#   runpod-templates.sh plan <variant> <image>   print the payloads (no API call)
#
# The templates follow one release channel (FV_TEMPLATE_CHANNEL, default
# `stable`; docs/serve/releases.md): `release.sh promote … stable` and
# `rollback` sync them. CI's main builds sync them only when the repository
# variable FV_TEMPLATE_CHANNEL is `latest`. runpod-endpoint.sh and
# runpod-pod.sh boot the image the template names. Updating a template's
# image rolls every endpoint that uses it to the new image (Runpod rolling
# release). Each template's env names its image (FV_IMAGE_REF,
# FV_IMAGE_DIGEST) and, when FV_RELEASE_CHANNEL is set, the channel, so
# fv-serve's /health reports them.
#
# Templates carry no secret values: the D1/R2/webhook values are Runpod secret
# references ({{ RUNPOD_SECRET_fv_* }}), resolved by Runpod at boot. A private
# registry needs RUNPOD_REGISTRY_AUTH_ID (a Runpod container registry auth id,
# `POST /containerregistryauth`); the GHCR package is public, so none is set.
#
# Env: RUNPOD_API_KEY; RUNPOD_REGISTRY_AUTH_ID (optional); FV_RELEASE_CHANNEL
# (optional: the channel being synced).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"
# shellcheck source-path=SCRIPTDIR source=variants.sh
source "$HERE/variants.sh"

REST="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
SECRET_ENV_JSON='{
  "FV_CF_ACCOUNT_ID": "{{ RUNPOD_SECRET_fv_cf_account_id }}",
  "FV_CF_API_TOKEN": "{{ RUNPOD_SECRET_fv_cf_api_token }}",
  "FV_D1_DATABASE_ID": "{{ RUNPOD_SECRET_fv_d1_database_id }}",
  "FV_R2_BUCKET": "{{ RUNPOD_SECRET_fv_r2_bucket }}",
  "FV_R2_ENDPOINT": "{{ RUNPOD_SECRET_fv_r2_endpoint }}",
  "FV_R2_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
  "FV_R2_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
  "FV_WEBHOOK_ED25519_KEY": "{{ RUNPOD_SECRET_fv_webhook_ed25519_key }}"
}'

rest() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$REST$2"
}

# payload <variant> <sls|pod> <image> [create] -> template JSON. The image's
# entrypoint (fv-entry) and baked FV_CONFIG start fv-serve; no start command.
# `create` adds the fields the update API does not take (isServerless, category).
payload() {
  local v="$1" flavour="$2" image="$3" mode="${4:-}"
  jq -n --arg name "$(fv_template_name "$v" "$flavour")" --arg image "$image" --arg flavour "$flavour" \
    --arg v "$v" --arg auth "${RUNPOD_REGISTRY_AUTH_ID:-}" --arg mode "$mode" --argjson secrets "$SECRET_ENV_JSON" \
    --arg ch "${FV_RELEASE_CHANNEL:-}" '
    ({FV_IMAGE_REF: $image} + (if ($image | contains("@sha256:")) then {FV_IMAGE_DIGEST: ($image | split("@")[1])} else {} end)
     + (if $ch == "" then {} else {FV_RELEASE_CHANNEL: $ch} end)) as $ident
    | {name: $name, imageName: $image, volumeInGb: 0,
     readme: ("fv-serve " + $v + " (" + $flavour + "), published by CI; docs/serve/images.md")}
    + (if $flavour == "sls" then {
         containerDiskInGb: 20,
         env: ($secrets + $ident + {FV_SERVE_MODE: "runpod-queue", FV_AUTH_MODE: "trust-gateway", FV_STATE_DIR: "/fvstate",
                           FV_WEIGHTS: "/runpod-volume/weights", FV_CACHE_DIR: "/fvstate/cache", RUST_LOG: "info"})}
       else {
         containerDiskInGb: 30, volumeMountPath: "/workspace",
         ports: (if $v == "cpu" then ["8000/http"] else ["8000/http", "70000/tcp"] end),
         env: ($secrets + $ident + {FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", RUST_LOG: "info"}
               + (if $v == "cpu" then {} else {FV_WEIGHTS: "/workspace/weights", FV_SERVE_FORWARD: "1"} end))}
       end)
    + (if $auth == "" then {} else {containerRegistryAuthId: $auth} end)
    + (if $mode == "create" then {isServerless: ($flavour == "sls"), category: (if $v == "cpu" then "CPU" else "NVIDIA" end)} else {} end)'
}

sync_one() {
  local v="$1" image="$2" flavour name cur id resp
  fv_variant_config "$v" >/dev/null || die "unknown variant $v (known: $FV_VARIANTS)"
  [[ "$image" == *@sha256:* ]] || die "$v: pass a digest reference, not $image"
  for flavour in $(fv_variant_flavours "$v"); do
    name="$(fv_template_name "$v" "$flavour")"
    cur="$(fv_template_get "$name")"
    if [[ -n "$cur" ]]; then
      id="$(jq -r .id <<<"$cur")"
      if [[ "$(jq -r .imageName <<<"$cur")" == "$image" && "$(jq -r '.env.FV_RELEASE_CHANNEL // ""' <<<"$cur")" == "${FV_RELEASE_CHANNEL:-$(jq -r '.env.FV_RELEASE_CHANNEL // ""' <<<"$cur")}" \
        && "$(jq -r '.env.FV_IMAGE_REF // ""' <<<"$cur")" == "$image" ]]; then
        log "$name ($id): already $image"; continue
      fi
      resp="$(rest PATCH "/templates/$id" "$(payload "$v" "$flavour" "$image")")" || die "$name: update failed: $(head -c 300 <<<"$resp")"
      log "$name ($id): updated to $image"
    else
      resp="$(rest POST /templates "$(payload "$v" "$flavour" "$image" create)")" || die "$name: create failed: $(head -c 300 <<<"$resp")"
      log "$name ($(jq -r '.id // "?"' <<<"$resp")): created for $image"
    fi
  done
}

case "${1:-}" in
  sync) shift; : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"; sync_one "${1:?variant}" "${2:?image}" ;;
  sync-all)
    shift; : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    while read -r v img _; do [[ -z "$v" || "$v" == \#* ]] || sync_one "$v" "$img"; done <"${1:?file}" ;;
  list)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    rest GET /templates | jq -r '.[] | select(.name | startswith("fv-serve-")) | select(.name | test("-(sls|pod)$")) | "\(.id)\t\(.name)\t\(.imageName)"' ;;
  id)
    shift; : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    fv_template_get "$(fv_template_name "${1:?variant}" "${2:?sls|pod}")" | jq -r '.id // empty' ;;
  plan)
    shift
    for f in $(fv_variant_flavours "${1:?variant}"); do payload "$1" "$f" "${2:?image}" create; done ;;
  *) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
