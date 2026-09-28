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
# 3. exec fv-serve with the arguments (e.g. `--config …` to override).
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
cfg="${FV_CONFIG:-}"
if [ -n "$cfg" ] && [ -f "$cfg" ] && [ -n "${FV_WEIGHTS:-}" ]; then
  # shellcheck disable=SC2016 # a literal ${FV_WEIGHTS} in the config
  sed -n 's/^weights *= *"\${FV_WEIGHTS}\/\([^"/]*\).*/\1/p' "$cfg" | while read -r t; do
    [ -e "$FV_WEIGHTS/$t" ] || echo "fv-entry: variant ${FV_VARIANT:-?}: weights tree $FV_WEIGHTS/$t is missing" >&2
  done
fi
exec /opt/fastvideo-rs/bin/fv-serve "$@"
