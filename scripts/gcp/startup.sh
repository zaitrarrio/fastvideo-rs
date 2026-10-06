#!/bin/bash
# Compute Engine startup script for fastvideo-rs VMs (scripts/gcp/vm.sh and
# weights.sh send it as the `startup-script` metadata item; it runs as root on
# the Ubuntu accelerator image). docs/serve/deploy-gcp.md.
#
# Role (metadata `fv-role`):
#   serve     NVIDIA driver check (>= 580), NVENC library, Docker + NVIDIA
#             container toolkit; mount the weight disk read-only at
#             /mnt/fvw; verify-weights.sh for `fv-verify-cells`; run
#             `fv-image` with --gpus all on the host network with the
#             `fv-config` TOML; mirror the container log to the serial
#             console (that is where the admin-token banner is read); then,
#             when `fv-encode-bench` = 1, wait for `fv-encode-bench-src` and
#             run the NVENC vs x264 comparison on that clip.
#   populate  (CPU VM) format + mount the work disk read-write, download the
#             `fv-trees` rows from the Hugging Face Hub at their revisions,
#             the auxiliary/ files (pinned URL + SHA-256) and MMAudio, run
#             verify-weights.sh, and optionally copy the tree to GCS.
#   quantize  (GPU VM) mount the work disk read-write and write the derived
#             text_encoder_fp8 trees with `fv-gpucheck quantize-text-encoder`
#             from `fv-image`, then verify-weights.sh text-fp8.
#
# Progress lines start with "FV-GCP " and go to the serial console (port 1),
# which vm.sh ssh-free-logs and the runners read through the Compute API.
# Secrets are read from Secret Manager or instance metadata into a mode-600
# env file and never printed.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

MD=http://metadata.google.internal/computeMetadata/v1
md() { curl -sf --max-time 10 -H 'Metadata-Flavor: Google' "$MD/$1"; }
attr() { md "instance/attributes/$1"; }
say() {
  local line="FV-GCP $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >/dev/ttyS0 2>/dev/null || true
}
fail() { say "FAILED $*"; exit 1; }

ROLE="$(attr fv-role || echo serve)"
IMAGE="$(attr fv-image || true)"
STAMP=/var/lib/fv-gcp
mkdir -p "$STAMP" /mnt/fvw
if [[ -f "$STAMP/started" && "$ROLE" == serve ]]; then
  say "restart detected (boot $(cat /proc/sys/kernel/random/boot_id)); the container has --restart no: not restarting"
  exit 0
fi
date +%s >"$STAMP/started"
say "role=$ROLE start $(date -u +%FT%TZ) machine=$(md instance/machine-type | awk -F/ '{print $NF}') zone=$(md instance/zone | awk -F/ '{print $NF}')"

apt_quiet() { apt-get -o DPkg::Lock::Timeout=600 -qq "$@" >/var/log/fv-apt.log 2>&1; }

gpu_prep() {
  command -v nvidia-smi >/dev/null || fail "no nvidia-smi: the image has no NVIDIA driver (FV_GCP_IMAGE)"
  local drv gpu major
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  gpu="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  major="${drv%%.*}"
  say "DRIVER $drv GPU $gpu"
  (( major >= 580 )) || fail "driver $drv < 580: CUDA 13 needs R580+"
  # NVENC: libnvidia-encode ships with the full driver packages, not with
  # the headless/compute ones. Install the matching one when missing.
  if ! ldconfig -p | grep -q 'libnvidia-encode\.so\.1'; then
    local pkg
    pkg="$(dpkg-query -W -f='${Package}\n' 'libnvidia-compute-*' 2>/dev/null | head -1)"
    pkg="${pkg/compute/encode}"
    say "installing ${pkg:-libnvidia-encode} (NVENC)"
    apt_quiet update
    if [[ -n "$pkg" ]] && apt_quiet install -y "$pkg"; then :; else say "WARNING: could not install libnvidia-encode; NVENC checks will fail"; fi
  fi
  ldconfig -p | grep -q 'libnvidia-encode\.so\.1' && say "NVENC library present"
}

docker_prep() {
  if ! command -v docker >/dev/null; then
    say "installing docker.io"
    apt_quiet update
    apt_quiet install -y docker.io || fail "docker install"
  fi
  if [[ "${1:-gpu}" == gpu ]] && ! command -v nvidia-ctk >/dev/null; then
    say "installing nvidia-container-toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      >/etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt_quiet update
    apt_quiet install -y nvidia-container-toolkit || fail "nvidia-container-toolkit install"
    nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1
  fi
  systemctl restart docker
  local t0=$SECONDS
  [[ -n "$IMAGE" ]] && { docker pull -q "$IMAGE" >/dev/null || fail "docker pull $IMAGE"; }
  say "IMAGE pulled in $((SECONDS - t0))s: $IMAGE"
}

# mount_disk <device name> <ro|rw>
mount_disk() {
  local dev="/dev/disk/by-id/google-$1" mode="$2"
  for _ in $(seq 60); do [[ -e "$dev" ]] && break; sleep 2; done
  [[ -e "$dev" ]] || fail "disk $1 is not attached"
  if [[ "$mode" == ro ]]; then
    mount -o ro,noload "$dev" /mnt/fvw || fail "mount $1 read-only"
  else
    blkid "$dev" >/dev/null 2>&1 || { say "formatting $1 (ext4)"; mkfs.ext4 -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard -L fvweights "$dev"; }
    mount -o discard,defaults "$dev" /mnt/fvw || fail "mount $1 read-write"
  fi
  say "DISK $1 mounted $mode at /mnt/fvw ($(df -h /mnt/fvw | awk 'NR==2{print $3" used of "$2}'))"
}

# verify <cells...>: verify-weights.sh from the metadata copy (populate) or
# from the image (serve/quantize).
verify_cells() {
  local out rc
  if [[ -n "$IMAGE" ]] && command -v docker >/dev/null; then
    out="$(docker run --rm --entrypoint bash -e FV_WEIGHTS=/workspace/weights -v /mnt/fvw:/workspace:ro "$IMAGE" \
      /opt/fastvideo-rs/scripts/gpu/verify-weights.sh "$@" 2>&1)"; rc=$?
  else
    out="$(FV_WEIGHTS=/mnt/fvw/weights bash /opt/fv/scripts/verify-weights.sh "$@" 2>&1)"; rc=$?
  fi
  printf '%s\n' "$out" | sed 's/^/FV-GCP verify: /' >/dev/ttyS0 2>/dev/null || true
  return $rc
}

# Secrets into a mode-600 env file: Secret Manager (lower-case ids, the
# Runpod secret spelling) or metadata items fv-secret-<NAME>.
write_secret_env() {
  local file="$1" mode names name val tok project id
  mode="$(attr fv-secrets-mode || echo none)"
  names="$(attr fv-secret-names || true)"
  [[ -n "$names" && "$mode" != none ]] || { say "secrets: none"; return 0; }
  project="$(md project/project-id)"
  local n=0 missing=()
  for name in $names; do
    val=""
    if [[ "$mode" == secret-manager ]]; then
      tok="$(md instance/service-accounts/default/token | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
      id="$(tr '[:upper:]' '[:lower:]' <<<"$name")"
      val="$(curl -sf --max-time 20 -K <(printf 'header = "Authorization: Bearer %s"\n' "$tok") \
        "https://secretmanager.googleapis.com/v1/projects/$project/secrets/$id/versions/latest:access" \
        | python3 -c 'import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode(), end="")' 2>/dev/null)" || val=""
    else
      val="$(attr "fv-secret-$name" 2>/dev/null)" || val=""
    fi
    if [[ -n "$val" ]]; then
      printf '%s=%s\n' "$name" "$val" >>"$file"; n=$((n + 1))
    else
      missing+=("$name")
    fi
  done
  say "secrets: $n loaded from $mode${missing[*]:+; missing: ${missing[*]}}"
}

# ---------------------------------------------------------------- serve
role_serve() {
  gpu_prep
  docker_prep gpu
  mount_disk fv-weights ro
  local cells
  cells="$(attr fv-verify-cells || true)"
  if [[ -n "$cells" ]]; then
    # shellcheck disable=SC2086 # cell list
    if verify_cells $cells; then say "VERIFY ok: $cells"; else fail "VERIFY weights incomplete: $cells (see verify lines)"; fi
  fi
  install -d -m 700 /etc/fv-gcp /var/lib/fvstate
  attr fv-config >/etc/fv-gcp/serve.toml || fail "no fv-config metadata"
  local env=/etc/fv-gcp/env ip
  ( umask 077; : >"$env" )
  ip="$(md instance/network-interfaces/0/access-configs/0/external-ip)"
  attr fv-env | python3 -c 'import json,sys
for k, v in json.load(sys.stdin).items(): print(f"{k}={v}")' >>"$env"
  {
    echo "FV_PUBLIC_IP=$ip"
    echo "FV_PUBLIC_BASE_URL=http://$ip:8000"
    echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video"
  } >>"$env"
  write_secret_env "$env"
  local t0=$SECONDS
  docker run -d --name fv-serve --restart no --gpus all --network host \
    --env-file "$env" \
    -v /mnt/fvw:/workspace:ro -v /etc/fv-gcp/serve.toml:/etc/fv-gcp/serve.toml:ro -v /var/lib/fvstate:/fvstate \
    "$IMAGE" --config /etc/fv-gcp/serve.toml >/dev/null || fail "docker run"
  # The container log (including the one-time admin-token banner) goes to
  # the serial console; read it with `vm.sh ssh-free-logs <vm> --banner`.
  systemd-run --unit fv-serial-log --collect bash -c 'docker logs -f fv-serve 2>&1 | sed -u "s/^/fv-serve: /" >/dev/ttyS0' >/dev/null
  say "STARTED fv-serve in $((SECONDS - t0))s public=http://$ip:8000"
  for _ in $(seq 1 1440); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/ping)" == 200 ]]; then
      say "READY after $(( $(date +%s) - $(cat "$STAMP/started") ))s from boot"; break
    fi
    docker inspect -f '{{.State.Running}}' fv-serve 2>/dev/null | grep -q true || fail "fv-serve exited: $(docker logs --tail 5 fv-serve 2>&1 | tr '\n' ' ')"
    sleep 5
  done
  if [[ "$(attr fv-encode-bench || echo 0)" == 1 ]]; then encode_bench_loop; fi
}

# NVENC vs x264 on the same clip: decode it once, then encode with the
# production file settings (h264_nvenc p5/hq/VBR cq19, fastvideo-media
# FfmpegH264::Nvenc::file_args) and with libx264 veryfast crf19 (the CPU
# test encoder) and medium crf19 (a CPU quality reference). Reports wall
# time, fps, bytes and PSNR/SSIM against the decoded source.
encode_bench_loop() {
  local src="" etag="0"
  say "ENCODE-BENCH waiting for fv-encode-bench-src"
  for _ in $(seq 1 720); do
    src="$(curl -sf --max-time 70 -H 'Metadata-Flavor: Google' "$MD/instance/attributes/fv-encode-bench-src?wait_for_change=true&timeout_sec=60&last_etag=$etag" || true)"
    [[ -n "$src" ]] && break
    sleep 1
  done
  [[ -n "$src" ]] || { say "ENCODE-BENCH no source clip"; return 0; }
  local d=/var/lib/fvstate/bench
  mkdir -p "$d"
  curl -sf --max-time 300 -o "$d/src.mp4" "$src" || { say "ENCODE-BENCH download failed"; return 0; }
  docker run --rm --gpus all --entrypoint bash -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video -v "$d:/b" "$IMAGE" -c '
    set -u
    cd /b
    probe() { ffprobe -v error -select_streams v:0 -show_entries "stream=$2" -of default=nw=1:nk=1 "$1"; }
    ffmpeg -v error -y -i src.mp4 -an -f rawvideo -pix_fmt yuv420p src.yuv
    W="$(probe src.mp4 width)"; H="$(probe src.mp4 height)"; R="$(probe src.mp4 r_frame_rate)"
    N=$(( $(stat -c %s src.yuv) / (W * H * 3 / 2) ))
    enc() { # name, ffmpeg video args
      local name="$1"; shift
      local t0 t1
      t0=$(date +%s.%N)
      ffmpeg -v error -y -f rawvideo -pix_fmt yuv420p -s "${W}x${H}" -r "$R" -i src.yuv "$@" "$name.mp4" || { echo "{\"encoder\":\"$name\",\"error\":\"encode failed\"}"; return; }
      t1=$(date +%s.%N)
      local q psnr ssim
      q="$(ffmpeg -v info -i "$name.mp4" -f rawvideo -pix_fmt yuv420p -s "${W}x${H}" -r "$R" -i src.yuv -lavfi "[0:v][1:v]psnr;[0:v][1:v]ssim" -f null - 2>&1)"
      psnr="$(grep -o "average:[0-9.inf]*" <<<"$q" | head -1 | cut -d: -f2)"
      ssim="$(grep -o "All:[0-9.]*" <<<"$q" | head -1 | cut -d: -f2)"
      printf "{\"encoder\":\"%s\",\"wall_s\":%s,\"fps\":%s,\"bytes\":%s,\"psnr_db\":\"%s\",\"ssim\":\"%s\"}\n" "$name" \
        "$(awk -v a="$t0" -v b="$t1" "BEGIN{printf \"%.3f\", b-a}")" \
        "$(awk -v a="$t0" -v b="$t1" -v n="$N" "BEGIN{printf \"%.1f\", n/(b-a)}")" "$(stat -c %s "$name.mp4")" "$psnr" "$ssim"
    }
    {
      echo "{\"source\":{\"width\":$W,\"height\":$H,\"frames\":\"$N\",\"rate\":\"$R\"}}"
      enc nvenc_p5_hq_cq19 -c:v h264_nvenc -preset p5 -tune hq -profile:v high -rc vbr -cq 19 -b:v 0 -bf 0 -pix_fmt yuv420p
      enc x264_veryfast_crf19 -c:v libx264 -preset veryfast -crf 19 -pix_fmt yuv420p
      enc x264_medium_crf19 -c:v libx264 -preset medium -crf 19 -pix_fmt yuv420p
    } > bench.jsonl
    rm -f src.yuv
  ' >/dev/null 2>"$d/bench.err"
  if [[ -s "$d/bench.jsonl" ]]; then
    say "ENCODE-BENCH $(python3 -c 'import json,sys; print(json.dumps([json.loads(l) for l in open(sys.argv[1]) if l.strip()]))' "$d/bench.jsonl")"
  else
    say "ENCODE-BENCH failed: $(tail -c 300 "$d/bench.err" | tr '\n' ' ')"
  fi
}

# ---------------------------------------------------------------- populate
role_populate() {
  mount_disk fv-work rw
  mkdir -p /opt/fv/scripts /mnt/fvw/weights
  attr fv-script-verify-weights >/opt/fv/scripts/verify-weights.sh
  attr fv-script-verify-safetensors >/opt/fv/scripts/verify-safetensors.sh
  attr fv-manifest >/opt/fv/scripts/weights-manifest.tsv
  attr fv-mmaudio-py >/opt/fv/scripts/fetch-mmaudio.py
  say "installing python venv + huggingface_hub"
  apt_quiet update
  apt_quiet install -y python3-venv python3-pip jq || fail "apt python3-venv"
  python3 -m venv /opt/fv/venv
  /opt/fv/venv/bin/pip install -q "huggingface_hub[hf_xet]>=0.34" safetensors numpy >/var/log/fv-pip.log 2>&1 || fail "pip huggingface_hub"
  local hf_token
  hf_token="$(attr fv-secret-HF_TOKEN 2>/dev/null || true)"
  [[ -n "$hf_token" ]] && export HF_TOKEN="$hf_token"
  export HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=0 HF_HOME=/mnt/fvw/.hf
  local rc=0 dest repo rev globs t0 incl sha
  # fv-trees: dest<TAB>repo<TAB>revision<TAB>space-separated globs
  while IFS=$'\t' read -r dest repo rev globs; do
    [[ -n "$dest" && "$dest" != \#* ]] || continue
    if [[ -f "/mnt/fvw/weights/$dest/.complete" ]]; then say "TREE $dest already complete"; continue; fi
    t0=$SECONDS
    incl=()
    for g in $globs model_index.json; do incl+=(--include "$g"); done
    say "TREE $dest <- $repo@$rev"
    if /opt/fv/venv/bin/hf download "$repo" --revision "$rev" "${incl[@]}" --local-dir "/mnt/fvw/weights/$dest" --max-workers 8 >"/var/log/fv-hf-$dest.log" 2>&1; then
      sha="$(curl -sf --max-time 20 ${HF_TOKEN:+-K <(printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN")} "https://huggingface.co/api/models/$repo/revision/$rev" | jq -r '.sha // empty')"
      printf '%s\t%s\t%s\n' "$repo" "$rev" "${sha:-unknown}" >"/mnt/fvw/weights/$dest/.revision"
      echo "$((SECONDS - t0))" >"/mnt/fvw/weights/$dest/.complete"
      say "TREE $dest ok in $((SECONDS - t0))s $(du -sb "/mnt/fvw/weights/$dest" | cut -f1) bytes sha=${sha:-unknown}"
    else
      say "TREE $dest FAILED: $(tail -3 "/var/log/fv-hf-$dest.log" | tr '\n' ' ')"; rc=1
    fi
  done < <(attr fv-trees)
  # auxiliary/: pinned URLs + SHA-256 (weights-manifest.tsv rows).
  local rel url meta want size p
  while IFS=$'\t' read -r rel url meta; do
    url="${url#url:}"; want="${meta#sha256:}"; want="${want%% *}"; size="${meta##*size:}"
    p="/mnt/fvw/weights/$rel"; mkdir -p "$(dirname "$p")"
    if [[ -f "$p" && "$(sha256sum "$p" | cut -d' ' -f1)" == "$want" ]]; then continue; fi
    if curl -sfL --max-time 600 -o "$p.part" "$url" \
      && [[ "$(sha256sum "$p.part" | cut -d' ' -f1)" == "$want" && "$(stat -c %s "$p.part")" == "$size" ]]; then
      mv "$p.part" "$p"
    else
      say "AUX $rel FAILED"; rc=1
    fi
  done < <(grep -E '^auxiliary/' /opt/fv/scripts/weights-manifest.tsv)
  for d in /mnt/fvw/weights/auxiliary/*/; do [[ -d "$d" ]] && date +%s >"$d/.complete"; done
  # MMAudio: three repos + safetensors conversion (fetch-mmaudio.py; CPU torch).
  if [[ "$(attr fv-mmaudio || echo 1)" == 1 && ! -f /mnt/fvw/weights/mmaudio-44k-v2/.complete ]]; then
    say "MMAUDIO fetch + convert"
    /opt/fv/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu >>/var/log/fv-pip.log 2>&1
    mkdir -p /var/log/fv-mmaudio
    if MMAUDIO_ROOT=/mnt/fvw/weights/mmaudio-44k-v2 FETCH_SRV=/var/log/fv-mmaudio /opt/fv/venv/bin/python /opt/fv/scripts/fetch-mmaudio.py >/var/log/fv-mmaudio/run.log 2>&1; then
      say "MMAUDIO ok"
    else
      say "MMAUDIO FAILED: $(tail -3 /var/log/fv-mmaudio/log.txt 2>/dev/null | tr '\n' ' ')"; rc=1
    fi
  fi
  local cells
  cells="$(attr fv-verify-cells || true)"
  # shellcheck disable=SC2086 # cell list
  if [[ -n "$cells" ]] && verify_cells $cells; then say "VERIFY ok: $cells"; else say "VERIFY incomplete: $cells"; rc=1; fi
  local bucket
  bucket="$(attr fv-gcs-bucket || true)"
  if [[ -n "$bucket" ]]; then
    if command -v gcloud >/dev/null; then
      local t1=$SECONDS
      if gcloud storage rsync --recursive --quiet /mnt/fvw/weights "gs://$bucket/weights" >/var/log/fv-gcs.log 2>&1; then
        say "GCS copy ok in $((SECONDS - t1))s -> gs://$bucket/weights"
      else
        say "GCS copy FAILED: $(tail -2 /var/log/fv-gcs.log | tr '\n' ' ')"; rc=1
      fi
    else
      say "GCS copy skipped: no gcloud on this image"
    fi
  fi
  du -sb /mnt/fvw/weights/* 2>/dev/null | sed 's/^/FV-GCP SIZE /' >/dev/ttyS0
  sync; umount /mnt/fvw
  say "POPULATE DONE rc=$rc"
}

# ---------------------------------------------------------------- quantize
role_quantize() {
  gpu_prep
  docker_prep gpu
  mount_disk fv-work rw
  local rc=0 fam root spec
  for spec in h3:h3-base ltx2-gemma4:ltx25; do
    fam="${spec%%:*}"; root="${spec#*:}"
    if [[ -f "/mnt/fvw/weights/$root/text_encoder_fp8/manifest.json" ]]; then say "QUANT $root exists"; continue; fi
    local t0=$SECONDS
    if docker run --rm --gpus all --entrypoint /opt/fastvideo-rs/target/release/fv-gpucheck -v /mnt/fvw:/workspace "$IMAGE" \
        --out /tmp/q quantize-text-encoder --family "$fam" --root "/workspace/weights/$root" >"/var/log/fv-quant-$root.log" 2>&1; then
      say "QUANT $root ok in $((SECONDS - t0))s"
    else
      say "QUANT $root FAILED: $(tail -3 "/var/log/fv-quant-$root.log" | tr '\n' ' ')"; rc=1
    fi
  done
  if verify_cells text-fp8; then say "VERIFY ok: text-fp8"; else say "VERIFY incomplete: text-fp8"; rc=1; fi
  sync; umount /mnt/fvw
  say "QUANTIZE DONE rc=$rc"
}

case "$ROLE" in
  serve) role_serve ;;
  populate) role_populate ;;
  quantize) role_quantize ;;
  *) fail "unknown fv-role $ROLE" ;;
esac
