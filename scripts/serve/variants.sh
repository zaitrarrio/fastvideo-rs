# shellcheck shell=bash
# Per-variant serve images and their Runpod templates (docs/serve/images.md).
# Sourced by runpod-templates.sh, runpod-endpoint.sh and runpod-pod.sh.
#
#   variant    config (baked as FV_CONFIG)   GHCR tag                          Runpod templates
#   h3-turbo   runpod.toml                   fastvideo-rs-serve:h3-turbo[-sha-…]  fv-serve-h3-turbo-sls / -pod
#   h3-max     runpod-h3-max.toml            …:h3-max                           fv-serve-h3-max-sls / -pod
#   ltx        runpod-ltx.toml               …:ltx                              fv-serve-ltx-sls / -pod
#   wan        runpod-wan.toml               …:wan                              fv-serve-wan-sls / -pod
#   wan5b      runpod-wan5b.toml             …:wan5b                            fv-serve-wan5b-sls / -pod
#   sfwan      runpod-sfwan.toml             …:sfwan                            fv-serve-sfwan-sls / -pod
#   gateway    gateway.toml                  …:gateway (CPU only)               fv-serve-gateway-pod
#   debug      every config (legacy image)   …:latest / :sha-…                  (none)

FV_VARIANTS="h3-turbo h3-max ltx wan wan5b sfwan gateway"
FV_SERVE_REPO="${FV_SERVE_REPO:-ghcr.io/zaitrarrio/fastvideo-rs-serve}"

# fv_variant_config <variant> -> config file name
fv_variant_config() {
  case "$1" in
    h3-turbo) echo runpod.toml ;;
    h3-max) echo runpod-h3-max.toml ;;
    ltx) echo runpod-ltx.toml ;;
    wan) echo runpod-wan.toml ;;
    wan5b) echo runpod-wan5b.toml ;;
    sfwan) echo runpod-sfwan.toml ;;
    gateway) echo gateway.toml ;;
    *) return 1 ;;
  esac
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

# fv_variant_flavours <variant> -> "sls pod" (gateway: "pod")
fv_variant_flavours() { [[ "$1" == gateway ]] && echo pod || echo "sls pod"; }

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
