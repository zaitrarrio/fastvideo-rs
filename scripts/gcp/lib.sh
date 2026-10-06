#!/usr/bin/env bash
# Shared helpers for scripts/gcp/*.sh (docs/serve/deploy-gcp.md). Source, don't
# execute. Compute Engine and Cloud Storage through their REST APIs with a
# token from auth.sh (no gcloud).
#
# FV_GCP_DRY_RUN=1: every mutating or reading API call prints
# "DRY-RUN <METHOD> <url>" and its JSON body (secrets redacted) on stdout and
# returns a canned response; no token is minted and nothing is called.
# shellcheck shell=bash disable=SC2034

GCP_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$GCP_LIB_DIR/../gpu/lib.sh"
# shellcheck source-path=SCRIPTDIR source=auth.sh
source "$GCP_LIB_DIR/auth.sh"

DRY="${FV_GCP_DRY_RUN:-0}"
# API bases: overridable for scripts/gcp/tests (a local mock of the APIs).
COMPUTE="${FV_GCP_COMPUTE_API:-https://compute.googleapis.com/compute/v1}"
STORAGE="${FV_GCP_STORAGE_API:-https://storage.googleapis.com/storage/v1}"
SECRETS_API="${FV_GCP_SECRETS_API:-https://secretmanager.googleapis.com/v1}"

if [[ "$DRY" == 1 ]]; then
  PROJECT="${GCP_PROJECT:-$( (gcp_project 2>/dev/null) || true)}"
  PROJECT="${PROJECT:-dry-run-project}"
else
  PROJECT="${GCP_PROJECT:-}"
fi
# EU by default (the deployment is EU-only since 2026-10-06, CLAUDE.md, and the
# family Durable Objects live in `weur`): europe-west4-b offers G4 (RTX PRO
# 6000), A3 High (H100) and G2 (L4) (docs/serve/deploy-gcp.md, Regions).
REGION="${GCP_REGION:-europe-west4}"
ZONE="${GCP_ZONE:-${REGION}-b}"
[[ "$ZONE" == "$REGION"-* ]] || REGION="${ZONE%-*}"
NETWORK="${FV_GCP_NETWORK:-default}"

# Our resources carry these labels; nothing without fv-owner=fastvideo-rs is
# ever modified or deleted by these scripts.
FV_OWNER_LABEL="fastvideo-rs"
GCP_LEDGER="${FV_GCP_LEDGER:-$FV_ROOT/artifacts/gcp/ledger.tsv}"
GCP_OUT="${FV_GCP_OUT:-$FV_ROOT/artifacts/gcp}"
GCP_FAMILIES_TSV="$GCP_LIB_DIR/families.tsv"

# gcp_family <family> <column>: a field of scripts/gcp/families.tsv ('-' is
# printed empty); fails for an unknown family. Columns: config runpod_twin
# dispatch_family verify_cells weight_trees machine fal_app minimax_model
# served reactor_mode.
gcp_family() {
  local col
  case "$2" in
    config) col=2 ;; runpod_twin) col=3 ;; dispatch_family) col=4 ;; verify_cells) col=5 ;;
    weight_trees) col=6 ;; machine) col=7 ;; fal_app) col=8 ;; minimax_model) col=9 ;;
    served) col=10 ;; reactor_mode) col=11 ;; *) return 2 ;;
  esac
  awk -F'\t' -v f="$1" -v c="$col" '$1 == f { v = $c; found = 1; if (v == "-") v = ""; print v; exit } END { exit !found }' "$GCP_FAMILIES_TSV"
}
gcp_families() { awk -F'\t' '!/^#/ && NF > 1 { print $1 }' "$GCP_FAMILIES_TSV"; }

# The Ubuntu accelerator image ships the NVIDIA R580 driver (>= 580.95.05, the
# minimum Compute Engine lists for G4, A3 High and G2 with CUDA 13). DLVM
# images are refused on G2, and COS needs manual driver mounts (no NVIDIA
# container toolkit), so this is the one image for every machine type.
GCP_IMAGE="${FV_GCP_IMAGE:-projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accelerator-2404-amd64-with-nvidia-580}"

gcp_require_project() {
  if [[ -z "$PROJECT" ]]; then
    PROJECT="$(gcp_project)" || die "set GCP_SA_KEY_JSON (and GCP_PROJECT if the key's project is not the target)"
  fi
  [[ -n "$PROJECT" ]] || die "GCP_PROJECT is not set and the key has no project_id"
}

gcp_ledger() { mkdir -p "$(dirname "$GCP_LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$GCP_LEDGER"; }

# Redact secret values in a JSON body for display: metadata items whose key
# is a secret, and any string field named like a secret.
gcp_redact() {
  jq '
    def secretkey: test("^fv-secret-[A-Z]");
    walk(
      if type == "object" and has("key") and has("value") and ((.key|type) == "string") and (.key|secretkey)
      then .value = "<redacted \(.value|tostring|length) chars>"
      elif type == "object" and has("payload") and ((.payload|type) == "object") and (.payload|has("data"))
      then .payload.data = "<redacted>"
      else . end)
    | (if (.metadata.items? // null) != null then
        .metadata.items |= map(if .key == "startup-script" or (.key|startswith("fv-script-")) or .key == "fv-config" or .key == "fv-mmaudio-py" or .key == "fv-manifest" or .key == "fv-sha256" or .key == "fv-trees"
          then .value = "<\(.value|length) chars: \(.value|split("\n")|.[0][0:60])...>" else . end)
      else . end)
  ' 2>/dev/null || cat
}

# gce <METHOD> <url> [json body]: one REST call; prints the response JSON.
# Non-2xx: prints the error JSON and returns 1.
gce() {
  local method="$1" url="$2" body="${3:-}" out code
  if [[ "$DRY" == 1 ]]; then
    printf 'DRY-RUN %s %s\n' "$method" "$url" >&2
    if [[ -n "$body" ]]; then gcp_redact <<<"$body" >&2; fi
    _gcp_dry_response "$method" "$url"
    return 0
  fi
  out="$(mktemp)"
  if [[ -n "$body" ]]; then
    code="$(gcp_curl_auth -sS --max-time 120 -o "$out" -w '%{http_code}' -X "$method" \
      -H 'content-type: application/json' --data-binary @- "$url" <<<"$body")" || { rm -f "$out"; return 1; }
  else
    code="$(gcp_curl_auth -sS --max-time 120 -o "$out" -w '%{http_code}' -X "$method" "$url")" || { rm -f "$out"; return 1; }
  fi
  cat "$out"; rm -f "$out"
  [[ "$code" == 2* ]]
}

# Canned responses (stdout) so dry runs walk the whole flow; the payload
# itself went to stderr.
_gcp_dry_response() {
  local method="$1" url="$2"
  case "$method $url" in
    "GET "*"/serialPort"*) jq -nc '{contents: "FV-GCP STARTED\nFV-GCP VERIFY ok\nFV-ENCODE-BENCH {\"dry_run\":true}\n", next: "0"}' ;;
    "GET "*"/instances/"*) jq -nc '{status:"RUNNING", labels:{"fv-owner":"fastvideo-rs"}, networkInterfaces:[{accessConfigs:[{natIP:"203.0.113.10"}]}], metadata:{fingerprint:"dry", items:[]}}' ;;
    "GET "*"/disks/"*) jq -nc '{status:"READY", labels:{"fv-owner":"fastvideo-rs"}, sizeGb:"600", users:[]}' ;;
    "GET "*"/firewalls/"*) jq -nc '{description:"fv-owner=fastvideo-rs"}' ;;
    "GET "*"/images/"*) jq -nc '{name:"dry-image", status:"READY", labels:{"fv-owner":"fastvideo-rs"}, diskSizeGb:"600"}' ;;
    "GET "*) jq -nc '{items: []}' ;;
    *) jq -nc '{name:"dry-run-op", status:"DONE"}' ;;
  esac
}

# gce_wait <operation json>: wait for a zonal/regional/global operation to
# finish; fails with its error.
gce_wait() {
  local op="$1" link name st
  [[ "$DRY" == 1 ]] && return 0
  link="$(jq -r '.selfLink // empty' <<<"$op")"
  name="$(jq -r '.name // empty' <<<"$op")"
  [[ -n "$link" ]] || { log "not an operation: $(head -c 300 <<<"$op")"; return 1; }
  for _ in $(seq 1 90); do
    st="$(gce POST "$link/wait")" || { log "operation $name: $(head -c 300 <<<"$st")"; return 1; }
    if [[ "$(jq -r .status <<<"$st")" == DONE ]]; then
      if jq -e '.error' >/dev/null <<<"$st"; then
        log "operation $name failed: $(jq -c '.error.errors' <<<"$st")"
        return 1
      fi
      return 0
    fi
  done
  log "operation $name still running after the wait budget"
  return 1
}

# Resource labels for everything we create. VMs also carry fv-deadline
# (unix seconds): `vm.sh reap` deletes ours past it (label-based reaper).
gcp_labels() {
  local kind="$1" run="${2:-}" extra="${3:-{\}}"
  jq -nc --arg o "$FV_OWNER_LABEL" --arg k "$kind" --arg r "$run" --argjson x "$extra" \
    '{"fv-owner":$o, "fv-kind":$k} + (if $r == "" then {} else {"fv-run":$r} end) + $x'
}

# gcp_is_ours <resource json>: true when labelled fv-owner=fastvideo-rs (or,
# for firewall rules, which have no labels, when the description says so).
gcp_is_ours() {
  jq -e --arg o "$FV_OWNER_LABEL" '(.labels["fv-owner"] // "") == $o or ((.description // "") | test("fv-owner=" + $o))' >/dev/null <<<"$1"
}

# ghcr tag -> digest reference (public packages; anonymous pull token).
gcp_resolve_digest() {
  local ref="$1" repo tag tok digest
  if [[ "$ref" == *@sha256:* ]]; then echo "$ref"; return; fi
  [[ "$ref" == ghcr.io/* ]] || die "cannot pin $ref: only ghcr.io tags are resolved; pass a digest"
  repo="${ref#ghcr.io/}"; tag="${repo##*:}"; repo="${repo%:*}"
  [[ "$tag" != "$repo" ]] || tag=latest
  tok="$(curl -sS --max-time 20 "https://ghcr.io/token?scope=repository:$repo:pull" | jq -r '.token // empty' 2>/dev/null || true)"
  digest="$(curl -sS --max-time 20 -I -H "Authorization: Bearer $tok" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json' \
    "https://ghcr.io/v2/$repo/manifests/$tag" 2>/dev/null | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}' || true)"
  if [[ "$digest" != sha256:* ]]; then
    [[ "$DRY" == 1 ]] && { echo "ghcr.io/$repo@sha256:<resolved-at-run-time>"; return; }
    die "could not resolve $ref to a digest (private package, or no such tag)"
  fi
  echo "ghcr.io/$repo@$digest"
}

# --- prices -------------------------------------------------------------------
# On-demand and Spot $/hr for the whole machine (GPU + vCPU + RAM), Linux, no
# disks, from the Cloud Billing catalog as published by gcloud-compute.com
# (pages last updated 2026-10-04, read 2026-10-06; the official
# cloud.google.com/compute/gpus-pricing page renders prices client-side and
# could not be read). Spot prices move; treat them as estimates.
# machine<TAB>region<TAB>ondemand<TAB>spot
GCP_PRICES="g4-standard-48	us-central1	4.4999	1.7716
g4-standard-48	europe-west4	4.9499	2.2091
g4-standard-48	europe-north1	4.9499	2.11
g4-standard-48	europe-west1	4.9499	2.35
g4-standard-96	us-central1	8.9999	3.5433
a3-highgpu-1g	us-central1	11.0612	6.6203
a3-highgpu-1g	europe-west4	14.0676	7.6891
a3-highgpu-1g	europe-west1	12.17	7.6891
g2-standard-8	us-central1	0.8536	0.5121
g2-standard-8	europe-west4	0.8972	0.5383
g2-standard-8	europe-west1	0.9399	0.547
g2-standard-16	us-central1	1.1472	0.6882
g2-standard-16	europe-west4	1.2058	0.7234
g2-standard-24	us-central1	2.0008	1.2003
c3-standard-8	us-central1	0.4032	0.1519
c3-standard-8	europe-west4	0.4234	0.1678
c3-standard-22	us-central1	1.1088	0.4176
e2-standard-8	us-central1	0.2680	0.1608"

# gcp_price <machine> <STANDARD|SPOT|FLEX_START> [region]: $/hr (the region's
# row, else the highest listed row for that machine: conservative).
gcp_price() {
  local m="$1" model="$2" region="${3:-$REGION}" col=3
  [[ "$model" == STANDARD ]] || col=4
  awk -F'\t' -v m="$m" -v r="$region" -v c="$col" '
    $1 == m && $2 == r { print $c; found = 1; exit }
    $1 == m { if ($c + 0 > max + 0) max = $c }
    END { if (!found && max != "") print max }' <<<"$GCP_PRICES"
}

# Disk $/GiB-hour and throughput $/(MiB/s)-hour (us-central1 list prices,
# cloud.google.com/compute/disks-image-pricing, fetched 2026-09-28; Hyperdisk
# ML re-checked 2026-10-06). European rates are UNVERIFIED (expect ~10% more).
GCP_HDML_GIB_HR=0.000109589
GCP_HDML_MIBS_HR=0.000164384
GCP_HDB_GIB_HR=0.000109589
GCP_PDB_GIB_HR=0.000136986
GCP_IMAGE_GIB_HR=0.000068493
GCP_GCS_GIB_HR=0.000027397

# The machine type's boot disk type: G4 and A3 boot from Hyperdisk Balanced
# (G4 supports nothing else), G2 from pd-balanced (no Hyperdisk Balanced).
gcp_boot_disk_type() {
  case "$1" in
    g4-* | a3-* | a4-* | c4-*) echo hyperdisk-balanced ;;
    *) echo pd-balanced ;;
  esac
}

# Machine type -> provisioning model: a3-highgpu-{1,2,4}g only exist as Spot or
# Flex-start (Compute Engine docs); everything else follows FV_GCP_SPOT.
gcp_provisioning() {
  local m="$1"
  if [[ -n "${FV_GCP_PROVISIONING:-}" ]]; then echo "$FV_GCP_PROVISIONING"; return; fi
  case "$m" in
    a3-highgpu-1g | a3-highgpu-2g | a3-highgpu-4g) echo SPOT ;;
    *) if [[ "${FV_GCP_SPOT:-0}" == 1 ]]; then echo SPOT; else echo STANDARD; fi ;;
  esac
}

# The client's public addresses as /32 CIDRs (the HTTPS proxy's egress and
# the direct HTTP egress can differ); FV_GCP_SOURCE_CIDR wins.
gcp_source_cidrs() {
  if [[ -n "${FV_GCP_SOURCE_CIDR:-}" ]]; then
    jq -Rc 'split("[, ]+"; null) | map(select(length > 0))' <<<"$FV_GCP_SOURCE_CIDR"
    return
  fi
  [[ "$DRY" == 1 ]] && { echo '["198.51.100.7/32"]'; return; }
  local a b
  a="$(curl -sS --max-time 10 https://api.ipify.org 2>/dev/null || true)"
  b="$(curl -sS --max-time 10 http://api.ipify.org 2>/dev/null || true)"
  printf '%s\n%s\n' "$a" "$b" | grep -E '^[0-9]+(\.[0-9]+){3}$' | sort -u | sed 's#$#/32#' | jq -Rsc 'split("\n") | map(select(length > 0))'
}
