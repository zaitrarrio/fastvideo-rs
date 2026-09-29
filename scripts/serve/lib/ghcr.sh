# shellcheck shell=bash
# The fv-serve images in GHCR (docs/serve/images.md, docs/serve/releases.md):
# tags, digests, labels, retagging. Source after scripts/gpu/lib.sh and
# scripts/serve/variants.sh.
#
# Release keys: `debug` (the all-in-one image: :sha-<s>, :latest, :<channel>)
# and every variant of $FV_VARIANTS (:<v>-sha-<s>, :<v>, :<v>-<channel>).
#
#   fv_ghcr_init                     anonymous pull token (public package)
#   fv_ghcr_digest <tag>             the tag's digest (sha256:…), or fail
#   fv_ghcr_labels <digest>          the image config's labels (JSON)
#   fv_ghcr_set_for_sha <short sha>  {"debug": "<repo>@sha256:…", "h3-turbo": …} of the tags that exist
#   fv_key_sha_tag <key> <short>     sha-<s> | <v>-sha-<s>
#   fv_key_channel_tags <key> <ch>   the tags a promotion to <ch> moves
#   fv_ghcr_retag <ref@digest> <tag> docker buildx imagetools create (no rebuild), then
#                                    checks the tag resolves to the same digest
#
# Env: FV_SERVE_REPO (variants.sh); FV_REGISTRY_API (default https://ghcr.io;
# the tests use a mock); FV_DOCKER (default docker).

FV_REGISTRY_API="${FV_REGISTRY_API:-https://ghcr.io}"
FV_RELEASE_KEYS="debug $FV_VARIANTS"
_FV_MANIFEST_ACCEPT='Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json'

fv_repo_path() { echo "${FV_SERVE_REPO#*/}"; }

fv_ghcr_init() {
  [[ -n "${_FV_REG_TOKEN:-}" ]] && return 0
  _FV_REG_TOKEN="$(curl -sS --max-time 30 "$FV_REGISTRY_API/token?scope=repository:$(fv_repo_path):pull" | jq -r '.token // empty' 2>/dev/null)"
  [[ -n "$_FV_REG_TOKEN" ]] || { log "GHCR: no pull token for $FV_SERVE_REPO"; return 1; }
  export _FV_REG_TOKEN
}

_fv_reg() {
  fv_ghcr_init || return 1
  curl -sS --max-time 60 -H @<(printf 'Authorization: Bearer %s\n' "$_FV_REG_TOKEN") "$@"
}

# fv_ghcr_digest <tag|digest> -> sha256:… (fails when the tag does not exist)
fv_ghcr_digest() {
  local d
  d="$(_fv_reg -I -H "$_FV_MANIFEST_ACCEPT" "$FV_REGISTRY_API/v2/$(fv_repo_path)/manifests/$1" \
    | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')"
  [[ "$d" == sha256:* ]] || return 1
  echo "$d"
}

fv_ghcr_manifest() { _fv_reg -H "$_FV_MANIFEST_ACCEPT" "$FV_REGISTRY_API/v2/$(fv_repo_path)/manifests/$1"; }

# fv_ghcr_labels <digest> -> {"org.opencontainers.image.revision": …, …}
fv_ghcr_labels() {
  local m cfg
  m="$(fv_ghcr_manifest "$1")" || return 1
  if jq -e '.manifests' >/dev/null 2>&1 <<<"$m"; then
    m="$(fv_ghcr_manifest "$(jq -r '[.manifests[] | select(.platform.architecture == "amd64")][0].digest' <<<"$m")")" || return 1
  fi
  cfg="$(jq -r '.config.digest // empty' <<<"$m" 2>/dev/null)"
  [[ -n "$cfg" ]] || return 1
  _fv_reg -L "$FV_REGISTRY_API/v2/$(fv_repo_path)/blobs/$cfg" | jq -c '.config.Labels // {}'
}

fv_key_sha_tag() { if [[ "$1" == debug ]]; then echo "sha-$2"; else echo "$1-sha-$2"; fi; }

# The tags a promotion to <channel> points at the image: debug -> :<channel>;
# a variant -> :<v>-<channel>, and for `latest` also the legacy :<v>.
fv_key_channel_tags() {
  if [[ "$1" == debug ]]; then echo "$2"
  elif [[ "$2" == latest ]]; then echo "$1-latest $1"
  else echo "$1-$2"
  fi
}

# fv_ghcr_set_for_sha <short sha> -> JSON map key -> repo@digest (existing tags only)
fv_ghcr_set_for_sha() {
  local s="$1" k d out='{}'
  for k in $FV_RELEASE_KEYS; do
    d="$(fv_ghcr_digest "$(fv_key_sha_tag "$k" "$s")" 2>/dev/null)" || continue
    out="$(jq -c --arg k "$k" --arg r "$FV_SERVE_REPO@$d" '. + {($k): $r}' <<<"$out")"
  done
  echo "$out"
}

# fv_ghcr_retag <repo@sha256:…> <tag>
fv_ghcr_retag() {
  local src="$1" tag="$2" want got
  want="${src##*@}"
  "${FV_DOCKER:-docker}" buildx imagetools create --tag "$FV_SERVE_REPO:$tag" "$src" >&2 \
    || { log "retag $tag: imagetools create failed"; return 1; }
  got="$(fv_ghcr_digest "$tag")" || { log "retag $tag: the tag does not resolve"; return 1; }
  [[ "$got" == "$want" ]] || { log "retag $tag: resolves to $got, not $want"; return 1; }
}
