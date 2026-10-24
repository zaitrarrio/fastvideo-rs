# shellcheck shell=bash
# Per-variant serve images and their Runpod templates (docs/serve/images.md).
# Sourced by runpod-templates.sh, runpod-endpoint.sh and runpod-pod.sh.
# BEGIN GENERATED from deploy/deploy.toml by control/gen-configs.mjs; do not edit
#
#   variant   config (baked as FV_CONFIG)               GHCR tag            Runpod templates              note
#   h3-turbo  runpod.toml                               …:h3-turbo[-sha-…]  fv-serve-h3-turbo-sls / -pod
#   h3-max    runpod-h3-max.toml                        …:h3-max[-sha-…]    fv-serve-h3-max-sls / -pod
#   ltx       runpod-ltx.toml                           …:ltx[-sha-…]       fv-serve-ltx-sls / -pod
#   wan       runpod-wan.toml                           …:wan[-sha-…]       fv-serve-wan-sls / -pod
#   wan5b     runpod-wan5b.toml (+ runpod-wan14b.toml)  …:wan5b[-sha-…]     fv-serve-wan5b-sls / -pod
#   sfwan     runpod-sfwan.toml                         …:sfwan[-sha-…]     fv-serve-sfwan-sls / -pod
#   cpu       runpod-fake.toml                          …:cpu[-sha-…]       fv-serve-cpu-pod              CPU only, fake engine; gateway until 2026-10-06
#   debug     every config (legacy image)               …:latest / :sha-…   (none)                        the old all-in-one tags; since 2026-10-06 the runtime image + fv-serve, on the shared layers
#
# Pool presets (fv-control's POOL_PRESETS, control/src/presets.ts) beyond the standard pools: more worker
# configs on the same images. A config the variant's image does not carry rides inline (FV_WORKER_TOML_B64).
#
#   preset                      variant  config                              how           title
#   ltx-pro                     ltx      runpod-ltx-pro.toml                 inline        LTX pro (dense)
#   ltx-a2v                     ltx      runpod-ltx-a2v.toml                 inline        LTX guided audio-to-video
#   ltx-ref2v                   ltx      runpod-ltx-ref2v.toml               inline        LTX reference-to-video
#   h3-ref2v                    h3-max   runpod-h3-ref2v.toml                inline        H3 reference-to-video
#   fasth3-trim                 h3-max   runpod-h3-trim.toml                 inline        FastH3 Trim 8-step
#   fasth3-trim-nvfp4           h3-max   runpod-h3-trim-nvfp4.toml           inline        FastH3 Trim 8-step NVFP4
#   fasth3-v2-nvfp4             h3-max   runpod-h3-v2-nvfp4.toml             inline        H3 NVFP4 tier: FastH3 V2 NVFP4
#   fasth3-4step-nvfp4          h3-max   runpod-h3-turbo-nvfp4.toml          inline        H3 NVFP4 tier: FastH3 4-step NVFP4
#   fasth3-v2-nvfp4-preview     h3-max   runpod-h3-v2-nvfp4-preview.toml     inline        H3 NVFP4 tier, fast preview (TAEH3)
#   fasth3-4step-nvfp4-preview  h3-max   runpod-h3-turbo-nvfp4-preview.toml  inline        H3 NVFP4 4-step, fast preview (TAEH3)
#   h3-turbo-preview            h3-max   runpod-h3-turbo-preview.toml        inline        H3 turbo, fast preview (TAEH3)
#   fasth3-trim-nvfp4-fast      h3-max   runpod-h3-trim-nvfp4-fast.toml      inline        FastH3 Trim NVFP4 + FP4 attention (fastest, opt-in)
#   h3-larry-turbo              h3-max   runpod-h3-larry-turbo.toml          inline        MiniMax-H3 Turbo LoRA (larry)
#   h3-lightx2v-turbo           h3-max   runpod-h3-lightx2v-turbo.toml       inline        MiniMax-H3 Turbo LoRA (lightx2v)
#   vdn-h3                      h3-max   runpod-vdn-h3.toml                  inline        VDN-H3 (Video DeltaNet)
#   vdn-h3-8step                h3-max   runpod-vdn-h3-8step.toml            inline        VDN-H3 8-step (Video DeltaNet, DMD)
#   h3-lightx2v-turbo-v01       h3-max   runpod-h3-lightx2v-turbo-v01.toml   inline        MiniMax-H3 Turbo LoRA (lightx2v v0.1, the Space's)
#   fastwan21                   wan      runpod-wan.toml                     in the image  FastWan 2.1 1.3B
#   sfwan                       sfwan    runpod-sfwan.toml                   in the image  SF-Wan causal streaming
#   longlive                    sfwan    runpod-longlive.toml                inline        LongLive-1.3B causal (NON-COMMERCIAL)
#   h3-family                   h3-max   runpod-family-h3.toml               inline        H3 family: every H3 variant on one pod
#   fasth3-family               h3-max   runpod-family-fasth3.toml           inline        FastH3 family: every FastH3 release on one pod
#   ltx-family                  ltx      runpod-family-ltx.toml              inline        LTX family: turbo, pro, guided A2V and Ref2V on one pod
#   streaming-family            sfwan    runpod-family-streaming.toml        inline        Streaming: every causal streaming model on one pod (sfwan, longlive; LongLive weights NON-COMMERCIAL)

FV_VARIANTS="h3-turbo h3-max ltx wan wan5b sfwan cpu"
# <preset>:<variant>:<config> for every non-standard preset (the table above).
FV_POOL_PRESETS="ltx-pro:ltx:runpod-ltx-pro.toml ltx-a2v:ltx:runpod-ltx-a2v.toml ltx-ref2v:ltx:runpod-ltx-ref2v.toml h3-ref2v:h3-max:runpod-h3-ref2v.toml fasth3-trim:h3-max:runpod-h3-trim.toml fasth3-trim-nvfp4:h3-max:runpod-h3-trim-nvfp4.toml fasth3-v2-nvfp4:h3-max:runpod-h3-v2-nvfp4.toml fasth3-4step-nvfp4:h3-max:runpod-h3-turbo-nvfp4.toml fasth3-v2-nvfp4-preview:h3-max:runpod-h3-v2-nvfp4-preview.toml fasth3-4step-nvfp4-preview:h3-max:runpod-h3-turbo-nvfp4-preview.toml h3-turbo-preview:h3-max:runpod-h3-turbo-preview.toml fasth3-trim-nvfp4-fast:h3-max:runpod-h3-trim-nvfp4-fast.toml h3-larry-turbo:h3-max:runpod-h3-larry-turbo.toml h3-lightx2v-turbo:h3-max:runpod-h3-lightx2v-turbo.toml vdn-h3:h3-max:runpod-vdn-h3.toml vdn-h3-8step:h3-max:runpod-vdn-h3-8step.toml h3-lightx2v-turbo-v01:h3-max:runpod-h3-lightx2v-turbo-v01.toml fastwan21:wan:runpod-wan.toml sfwan:sfwan:runpod-sfwan.toml longlive:sfwan:runpod-longlive.toml h3-family:h3-max:runpod-family-h3.toml fasth3-family:h3-max:runpod-family-fasth3.toml ltx-family:ltx:runpod-family-ltx.toml streaming-family:sfwan:runpod-family-streaming.toml"

# fv_variant_config <variant> -> config file name
fv_variant_config() {
  case "$1" in
    h3-turbo) echo runpod.toml ;;
    h3-max) echo runpod-h3-max.toml ;;
    ltx) echo runpod-ltx.toml ;;
    wan) echo runpod-wan.toml ;;
    wan5b) echo runpod-wan5b.toml ;;
    sfwan) echo runpod-sfwan.toml ;;
    cpu) echo runpod-fake.toml ;;
    *) return 1 ;;
  esac
}
# END GENERATED
FV_SERVE_REPO="${FV_SERVE_REPO:-ghcr.io/zaitrarrio/fastvideo-rs-serve}"

# fv_preset <preset> -> "<variant> <config>" (a pool preset), or status 1
fv_preset() {
  local x
  for x in $FV_POOL_PRESETS; do
    [[ "${x%%:*}" == "$1" ]] || continue
    x="${x#*:}"
    echo "${x%%:*} ${x#*:}"
    return 0
  done
  return 1
}

# fv_variant_for_config </etc/fv/….toml> -> variant, or nothing (fake, ref2v, …)
fv_variant_for_config() {
  local base v
  base="$(basename "$1")"
  for v in $FV_VARIANTS; do
    [[ "$(fv_variant_config "$v")" == "$base" ]] && { echo "$v"; return 0; }
  done
  return 0
}

# fv_variant_flavours <variant> -> "sls pod" (cpu: "pod")
fv_variant_flavours() { [[ "$1" == cpu ]] && echo pod || echo "sls pod"; }

fv_template_name() { echo "fv-serve-$1-$2"; }

# fv_template_get <name> -> the template JSON (REST v1), or nothing
fv_template_get() {
  curl -sS --fail-with-body -H "Authorization: Bearer $RUNPOD_API_KEY" "${RUNPOD_API_BASE:-https://rest.runpod.io/v1}/templates" \
    | jq -c --arg n "$1" '[.[]? | select(.name == $n)][0] // empty'
}

# fv_variant_image <variant> <flavour> -> the image the variant's Runpod
# template boots (what CI registered), else the GHCR variant tag.
fv_variant_image() {
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

# fv_default_image -> the all-in-one image deploys boot when nothing names
# one: :$FV_DEFAULT_CHANNEL (default stable), or :latest while that channel
# tag does not exist yet (before the first `release.sh promote`).
fv_default_image() {
  local ch="${FV_DEFAULT_CHANNEL:-stable}" repo tok code
  repo="${FV_SERVE_REPO#*/}"
  tok="$(curl -sS --max-time 15 "${FV_REGISTRY_API:-https://ghcr.io}/token?scope=repository:$repo:pull" 2>/dev/null | jq -r '.token // empty' 2>/dev/null || true)"
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -I -H @<(printf 'Authorization: Bearer %s\n' "$tok") \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json' \
    "${FV_REGISTRY_API:-https://ghcr.io}/v2/$repo/manifests/$ch" 2>/dev/null || true)"
  if [[ "$code" == 200 ]]; then echo "$FV_SERVE_REPO:$ch"
  else
    echo "variants: no :$ch tag yet; using :latest" >&2
    echo "$FV_SERVE_REPO:latest"
  fi
}
