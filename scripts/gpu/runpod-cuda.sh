#!/usr/bin/env bash
# The Runpod CUDA host filter (allowedCudaVersions) for the GPU launchers. Source, don't execute.
# shellcheck shell=bash
#
# The serve images ship CUDA 13 runtime libraries, which run on any host whose
# driver offers CUDA 13.0 or newer (driver >= 580). So the filter means "at
# least 13.0", never "exactly 13.0". Runpod labels hosts by their driver's CUDA
# version; on 2026-10-14 (GraphQL lowestPrice by allowedCudaVersions) the Secure
# RTX PRO 6000 hosts were all "13.2", H100 "13.0", H200 "13.0" and "13.2". A
# "13.0"-only filter therefore hid every RTX PRO 6000 ("no capacity", or a pod
# that never starts).
#
# REST v1 (POST /pods, /endpoints) refused "13.1" and "13.2" (its enum
# stopped at "13.0", checked live 2026-10-14), so a pod could not list every
# compatible label: fv_runpod_cuda prints nothing (no filter) until REST names
# them all. REST v2 (api.runpod.io/v2, the API the launchers speak since v1
# retired) declares `gpu.allowedCudaVersions` as any `major.minor` with no
# enum, but that has not been checked live: FV_RUNPOD_REST_CUDA stays v1's
# list until it is. A serverless endpoint takes `gpu.minCudaVersion`
# "$FV_MIN_CUDA" instead, which Runpod applies as a minimum (v2 rejects a
# floor beside a non-empty exact set).
#
# The same constants live in control/src/enums.ts (MIN_CUDA, IMAGE_CUDA,
# CUDA_VERSIONS); control/test/unit/cuda-filter.test.ts keeps them equal. When
# Runpod shows a new label, add it to FV_IMAGE_CUDA (and to IMAGE_CUDA); when
# REST accepts it, to FV_RUNPOD_REST_CUDA (and CUDA_VERSIONS).
#
# Env: RUNPOD_ALLOWED_CUDA overrides the default in each launcher (space list;
# empty = no filter).

# shellcheck disable=SC2034 # read by the scripts that source this file
FV_MIN_CUDA="13.0"
FV_IMAGE_CUDA="13.0 13.1 13.2"
FV_RUNPOD_REST_CUDA="13.0 12.9 12.8 12.7 12.6 12.5 12.4 12.3 12.2 12.1 12.0 11.8"

# The default gpu.allowedCudaVersions (space list) for a REST GPU pod or
# endpoint: every FV_IMAGE_CUDA label when REST can name them all, else
# nothing (a partial list hides the newest hosts).
fv_runpod_cuda() {
  local v
  for v in $FV_IMAGE_CUDA; do
    [[ " $FV_RUNPOD_REST_CUDA " == *" $v "* ]] || return 0
  done
  printf '%s\n' "$FV_IMAGE_CUDA"
}
