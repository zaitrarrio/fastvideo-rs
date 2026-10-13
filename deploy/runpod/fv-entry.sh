#!/bin/sh
# Entrypoint of the per-variant fv-serve images (docs/serve/images.md).
# FV_VARIANT and FV_CONFIG are baked per variant; fv-serve reads FV_CONFIG.
#
# 1. Weights: the volume's HF-cache trees link absolutely into
#    /workspace/weights (where pods mount the volume). Serverless workers mount
#    it at /runpod-volume, so link /workspace/weights there when it is free,
#    and default FV_WEIGHTS to whichever root exists.
# 2. Preflight: list the model trees the variant's config names that are
#    missing under FV_WEIGHTS (a warning; fv-serve reports the error itself).
# 3. The prompt upsampler's local sidecar when FV_UPSAMPLER_LOCAL=1 (fv-upsampler,
#    docs/serve/upsampler-local.md): llama-server starts while the preflight
#    runs; its FV_UPSAMPLER_BASE_URL / FV_UPSAMPLER_MODEL are exported once it
#    answers. If it fails, fv-serve starts anyway (upsampling falls back).
# 4. exec fv-serve with the arguments (e.g. `--config …` to override).
set -eu
if [ -d /runpod-volume/weights ] && [ ! -e /workspace/weights ]; then
  mkdir -p /workspace && ln -s /runpod-volume/weights /workspace/weights
fi
if [ -z "${FV_WEIGHTS:-}" ]; then
  if [ -d /runpod-volume/weights ]; then FV_WEIGHTS=/runpod-volume/weights
  elif [ -d /workspace/weights ]; then FV_WEIGHTS=/workspace/weights
  fi
  [ -n "${FV_WEIGHTS:-}" ] && export FV_WEIGHTS
fi
UPS=/opt/fastvideo-rs/bin/fv-upsampler
if [ "${FV_UPSAMPLER_LOCAL:-0}" = 1 ]; then
  if [ -x "$UPS" ]; then "$UPS" start || true
  else echo "fv-entry: FV_UPSAMPLER_LOCAL=1 but this image has no llama-server sidecar: upsampling falls back" >&2
  fi
fi
cfg="${FV_CONFIG:-}"
if [ -n "$cfg" ] && [ -f "$cfg" ] && [ -n "${FV_WEIGHTS:-}" ]; then
  # shellcheck disable=SC2016 # a literal ${FV_WEIGHTS} in the config
  sed -n 's/^weights *= *"\${FV_WEIGHTS}\/\([^"/]*\).*/\1/p' "$cfg" | while read -r t; do
    [ -e "$FV_WEIGHTS/$t" ] || echo "fv-entry: variant ${FV_VARIANT:-?}: weights tree $FV_WEIGHTS/$t is missing" >&2
  done
fi
if [ "${FV_UPSAMPLER_LOCAL:-0}" = 1 ] && [ -x "$UPS" ]; then
  ups_env="$("$UPS" wait || true)"
  eval "$ups_env"
fi
exec /opt/fastvideo-rs/bin/fv-serve "$@"
