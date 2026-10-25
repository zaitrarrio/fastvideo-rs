# shellcheck shell=bash
# The serve images and their Runpod templates (docs/serve/images.md): the one CUDA image every preset runs and the
# CPU image (owner decision 2026-10-25, docs/adr/0011-deploy-section.md; the file keeps its old name, its callers
# source it). Sourced by runpod-templates.sh, runpod-endpoint.sh, runpod-pod.sh, release.sh and lib/ghcr.sh.
# BEGIN GENERATED from deploy/deploy.toml by control/gen-configs.mjs; do not edit
#
#   image  compute  carries (FV_WORKER_CONFIG)  GHCR tag        Runpod templates          note
#   cuda   gpu      runpod-fake.toml            …:cuda[-sha-…]  fv-serve-cuda-sls / -pod
#   cpu    cpu      runpod-fake.toml            …:cpu[-sha-…]   fv-serve-cpu-pod          CPU only, fake engine; gateway until 2026-10-06
#
# No image bakes a worker config: a launch sends one (FV_WORKER_TOML_B64, or FV_WORKER_CONFIG naming a file the
# image carries; fv_config_env_json below), and an image started without one refuses to boot.

FV_IMAGES="cuda cpu"
FV_CUDA_IMAGE="cuda"
# END GENERATED
FV_SERVE_REPO="${FV_SERVE_REPO:-ghcr.io/zaitrarrio/fastvideo-rs-serve}"
# The config every serve image carries (deploy/deploy.toml SMOKE_CONFIG): the one a launch may name in the image.
FV_IMAGE_CONFIG=/etc/fv/runpod-fake.toml

# fv_image_flavours <image> -> "sls pod" (cpu: "pod")
fv_image_flavours() { [[ "$1" == cpu ]] && echo pod || echo "sls pod"; }

# fv_config_env_json <config> -> the env that sends a worker config at boot: {"FV_WORKER_CONFIG": "/etc/fv/runpod-fake.toml"}
# for the file the images carry, else {"FV_WORKER_TOML_B64": …} with the text of <config> (a local file, or a
# configs/serve file name, also as /etc/fv/<name> the way the old variant images named it). Fails for a config it
# cannot find: the images bake none, and an image started without one refuses to boot.
fv_config_env_json() {
  local c="$1" f
  if [[ "$c" == "$FV_IMAGE_CONFIG" ]]; then
    jq -nc --arg c "$c" '{FV_WORKER_CONFIG: $c}'
    return
  fi
  if [[ -f "$c" && "$c" != /etc/fv/* ]]; then f="$c"
  else f="${FV_ROOT:?source scripts/gpu/lib.sh first}/configs/serve/$(basename "$c")"
  fi
  [[ -f "$f" ]] || { echo "variants: no worker config $c (configs/serve/$(basename "$c") does not exist)" >&2; return 1; }
  jq -nc --arg b "$(base64 -w0 <"$f")" '{FV_WORKER_TOML_B64: $b}'
}

fv_template_name() { echo "fv-serve-$1-$2"; }

# fv_template_get <name> -> the template JSON (REST v1), or nothing
fv_template_get() {
  curl -sS --fail-with-body -H "Authorization: Bearer $RUNPOD_API_KEY" "${RUNPOD_API_BASE:-https://rest.runpod.io/v1}/templates" \
    | jq -c --arg n "$1" '[.[]? | select(.name == $n)][0] // empty'
}

# fv_image_ref <image> <flavour> -> the image the image's Runpod template boots (what CI or a release registered),
# else the GHCR tag :<image> (the `latest` channel).
fv_image_ref() {
  local t img=""
  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    t="$(fv_template_get "$(fv_template_name "$1" "$2")" 2>/dev/null || true)"
    img="$(jq -r '.imageName // empty' <<<"${t:-{\}}" 2>/dev/null || true)"
  fi
  echo "${img:-$FV_SERVE_REPO:$1}"
}

# fv_image_env_json <image> -> the env that tells fv-serve which image it runs
# ({FV_IMAGE_REF, FV_IMAGE_DIGEST, FV_RELEASE_CHANNEL when set}; /health
# reports them, docs/serve/releases.md).
fv_image_env_json() {
  jq -nc --arg image "$1" --arg ch "${FV_RELEASE_CHANNEL:-}" '{FV_IMAGE_REF: $image}
    + (if ($image | contains("@sha256:")) then {FV_IMAGE_DIGEST: ($image | split("@")[1])} else {} end)
    + (if $ch == "" then {} else {FV_RELEASE_CHANNEL: $ch} end)'
}
