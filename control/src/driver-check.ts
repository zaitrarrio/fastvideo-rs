// The host's NVIDIA driver against what the serve images need, checked at boot before anything else
// (docs/ops/r2-bench-2026-10-12.md finding 2).
//
// The images ship the CUDA 13.4 runtime libraries (scripts/gpu/cuda-13-runtime.pins) and fv-serve binds the driver
// API through cudarc's CUDA 13 bindings: on a host whose libcuda.so.1 is older than CUDA 13.0 (driver < 580) the first
// call panics with `libcuda.so: undefined symbol: cuCtxGetDevice_v2`, and the container crash-loops. Measured good:
// H100 580.126.09, RTX PRO 6000 595.91.07 (docs/serve/images.md).
//
// The check (POSIX sh, so the /bin/sh serverless boot can run it too): the driver version (nvidia-smi) and the
// symbol in the libcuda the loader would open. A host without either (a CPU worker, no nvidia-smi) passes. On
// failure it prints one line starting `driver_too_old:`, reports it to fv-control when the boot has a report URL
// (serverless r2: FV_SLS_REPORT_URL + FV_WEIGHTS_TOKEN; pods: FV_ENDPOINT_REPORT_URL + FV_ENDPOINT_REPORT_TOKEN), and
// exits 1 after FV_DRIVER_FAIL_SLEEP_S (default 20): no fetch and no fv-serve start on that host. fv-control's boot
// diagnosis (podlogs.ts) treats the line as fatal for the pod and places a new one; FV_SKIP_DRIVER_CHECK=1 skips it.
// The report names the pod by FV_POD_ID, else FV_DRV_POD (the Runpod boots set it from RUNPOD_POD_ID). FV_LIBCUDA
// names the libcuda to check (tests; default: the one ldconfig knows).

/** The least driver branch the images run on (CUDA 13.0's). */
export const MIN_DRIVER_MAJOR = 580;
/** A driver-API symbol of CUDA 13.0 that cudarc binds and older drivers lack (the 2026-10-12 panic). */
export const CUDA13_SYMBOL = "cuCtxGetDevice_v2";
/**
 * The default Runpod host filter (allowedCudaVersions) for GPU workers outside the EU volume's data centre: r2
 * serverless endpoints (any DC) and r2 pods placed outside EUR-IS-1. Runpod's REST enum stops at "13.0"
 * (rest.runpod.io/v1/openapi.json, 2026-10-12), the image's minimum. Caveat: "13.0" alone hid the EUR-IS-1 RTX PRO
 * 6000 hosts with driver 595.x on 2026-10-07 (they seem to carry a newer label the enum cannot name), so the filter
 * can leave the newest hosts out; allowed_cuda [] widens it, and the boot's driver check (above) still refuses an old
 * host before its fetch.
 */
export const R2_DEFAULT_CUDA = ["13.0"] as const;
/** The line the check prints (and podlogs.ts matches). */
export const DRIVER_TOO_OLD = "driver_too_old";

export const DRIVER_CHECK_SH = `# ---- the host's driver: the image needs CUDA 13 (driver >= ${MIN_DRIVER_MAJOR}); fail fast instead of a fv-serve panic
fv_driver_why() {
  dv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
  lc="\${FV_LIBCUDA:-$(ldconfig -p 2>/dev/null | awk '/libcuda\\.so\\.1 /{print $NF; exit}')}"
  [ -n "$lc" ] || for c in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1 /usr/local/nvidia/lib64/libcuda.so.1; do [ -e "$c" ] && { lc="$c"; break; }; done
  if [ -n "$dv" ] && [ "\${dv%%.*}" -lt ${MIN_DRIVER_MAJOR} ] 2>/dev/null; then echo "${DRIVER_TOO_OLD}: host driver $dv < ${MIN_DRIVER_MAJOR} (the image needs CUDA 13)"; return 1; fi
  if [ -n "$lc" ] && ! grep -aq ${CUDA13_SYMBOL} "$lc" 2>/dev/null; then echo "${DRIVER_TOO_OLD}: $lc (driver \${dv:-unknown}) has no ${CUDA13_SYMBOL} (the image needs a CUDA 13 driver, >= ${MIN_DRIVER_MAJOR})"; return 1; fi
  echo "driver \${dv:-unknown} ok"
}
if [ "\${FV_SKIP_DRIVER_CHECK:-0}" != 1 ]; then
  fv_drv="$(fv_driver_why)" || {
    echo "[fv-boot] $(date -u +%H:%M:%S) FAILED: $fv_drv" >&2
    fv_ru="\${FV_SLS_REPORT_URL:-\${FV_ENDPOINT_REPORT_URL:-}}"; fv_rt="\${FV_WEIGHTS_TOKEN:-\${FV_ENDPOINT_REPORT_TOKEN:-}}"
    fv_rp="\${FV_POD_ID:-\${FV_DRV_POD:-worker}}"; [ -n "\${FV_SLS_REPORT_URL:-}" ] && fv_rp="$fv_rp@$(date +%s)"
    [ -n "$fv_ru" ] && command -v curl >/dev/null 2>&1 && curl -fsS --connect-timeout 5 --max-time 10 -o /dev/null -X POST -H "authorization: Bearer $fv_rt" -H "content-type: application/json" \\
      --data "{\\"pod\\":\\"$fv_rp\\",\\"phase\\":\\"failed\\",\\"url\\":\\"\\",\\"detail\\":\\"$(printf '%s' "$fv_drv" | tr -d '"\\\\' | cut -c1-280)\\"}" "$fv_ru" >/dev/null 2>&1
    sleep "\${FV_DRIVER_FAIL_SLEEP_S:-20}"; exit 1
  }
  echo "[fv-boot] $(date -u +%H:%M:%S) $fv_drv" >&2
fi
`;
