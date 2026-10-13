// The prompt upsampler's local sidecar (docs/serve/upsampler-local.md; owner decisions 2026-10-13): llama.cpp's
// llama-server with a local Qwen3-VL-4B-Instruct GGUF and its vision projector (so the director's opening expansion
// sees configure.image_url and end images) next to fv-serve, on the same GPU. A pool, standalone pod or
// serverless endpoint turns it on with `upsampler_local: true` (default off); its pods then get FV_UPSAMPLER_LOCAL=1.
//
// The serve images carry llama-server and its launcher (/opt/fastvideo-rs/bin/fv-upsampler, deploy/runpod/
// fv-upsampler.sh). The boots below start it early (its fetch and load overlap the weights) and wait for it right
// before fv-serve starts: on success the launcher prints FV_UPSAMPLER_BASE_URL (http://127.0.0.1:18080/v1),
// FV_UPSAMPLER_MODEL and FV_UPSAMPLER_VISION=1 (image_url content parts work), which fv-serve's upsampler reads; on failure fv-serve starts anyway and upsampling falls back
// (logged, and reported as a boot step). The GGUF tree comes from the R2 weights store whatever the pod's weights
// source (it is in R2 only, never on a volume): the pod asks POST /ingest/v1/weights-plan?set=upsampler with its usual
// token (weights-plan.ts) and fetches with r2-fetch.py, onto its own disk.
import type { Env } from "./env";
import { r2Cfg, r2PodEnv, r2Trees } from "./r2weights";

/** The GGUF tree in the R2 weights store (scripts/gpu/weights-r2.tsv): Qwen's official Qwen3-VL-4B-Instruct GGUF,
 * Q8_0 language model + F16 mmproj. (The text-only qwen3-4b-instruct-gguf, published first the same day, stays in
 * R2, unused: the store is add-only.) */
export const UPSAMPLER_TREE = "qwen3-vl-4b-instruct-gguf";
/** The model name llama-server answers to (its --alias; FV_UPSAMPLER_MODEL). */
export const UPSAMPLER_MODEL = "qwen3-vl-4b-instruct";
/** The plan request's `set` for the sidecar's trees. */
export const UPSAMPLER_SET = "upsampler";

/** Why upsampler_local cannot work with this fv-control (no R2 weights store, or the tree not published); null if it can. */
export function upsamplerIssue(env: Env): string | null {
  const { off } = r2Cfg(env);
  if (off) return `upsampler_local fetches ${UPSAMPLER_TREE} from the R2 weights store: ${off}`;
  if (!r2Trees(env)[UPSAMPLER_TREE]) return `upsampler_local: ${UPSAMPLER_TREE} is not in the R2 weights store (scripts/gpu/weights-r2.tsv)`;
  return null;
}

/** The sidecar tree's bytes (0 when unpublished): added to the container disk of a pod that runs it. */
export const upsamplerBytes = (env: Env) => r2Trees(env)[UPSAMPLER_TREE]?.bytes ?? 0;
/** Container disk the sidecar adds (GB): its GGUF tree × 1.15, rounded up. */
export const upsamplerDiskGb = (env: Env) => Math.ceil((upsamplerBytes(env) / 1e9) * 1.15);

/** The env a pod with upsampler_local gets: the switch, the tree's pinned revision (a copy on the disk at another
 * revision is fetched again) and, for the fetch, the plan URL and the fetcher (the pod's own token authenticates the
 * plan request: FV_ENDPOINT_REPORT_TOKEN on cluster pods, FV_WEIGHTS_TOKEN on serverless workers). */
export function upsamplerPodEnv(env: Env): Record<string, string> {
  const t = r2Trees(env)[UPSAMPLER_TREE];
  return { FV_UPSAMPLER_LOCAL: "1", ...(t ? { FV_UPSAMPLER_REV: t.rev } : {}), FV_UPSAMPLER_MODEL: UPSAMPLER_MODEL, ...r2PodEnv(env) };
}

const UPS = "/opt/fastvideo-rs/bin/fv-upsampler";
/** Boot snippet: start the sidecar in the background (a no-op unless FV_UPSAMPLER_LOCAL=1). */
export const UPSAMPLER_START_SH = `# ---- the prompt upsampler's local sidecar (FV_UPSAMPLER_LOCAL=1, docs/serve/upsampler-local.md): fetched and loaded
# in the background while the weights come in; awaited just before fv-serve starts.
if [ "\${FV_UPSAMPLER_LOCAL:-0}" = 1 ]; then
  if [ -x ${UPS} ]; then ${UPS} start || true
  else echo "[fv-boot] $(date -u +%H:%M:%S) upsampler: FAILED: this image has no llama-server sidecar (${UPS}): fv-serve starts without it; upsampling falls back" >&2
  fi
fi
`;
/** Boot snippet: wait for the sidecar, export FV_UPSAMPLER_BASE_URL / FV_UPSAMPLER_MODEL for fv-serve, and report the
 * step (serverless: rep step; GMI / Brev: report upsampler). Never fails the boot. */
export const UPSAMPLER_WAIT_SH = `if [ "\${FV_UPSAMPLER_LOCAL:-0}" = 1 ] && [ -x ${UPS} ]; then
  ups_log="\${FV_STATE_DIR:-/fvstate}/upsampler-wait.log"
  ups_env="$(${UPS} wait 2>"$ups_log" || true)"
  cat "$ups_log" >&2 || true
  eval "$ups_env"
  ups_line="$(tail -n 1 "$ups_log" 2>/dev/null | sed 's/^\\[fv-boot\\] [0-9:]* //' | tr -d '"\\\\' | cut -c1-280)"
  ups_s="$(printf '%s' "$ups_line" | sed -n 's/^step upsampler: \\([0-9.]*\\)s.*/\\1/p')"
  if command -v rep >/dev/null 2>&1; then
    if [ -n "$ups_s" ]; then rep step "upsampler \${ups_s}s (\${ups_line#*s (}" || true; else rep step "upsampler_failed 0s ($ups_line)" || true; fi
  fi
  if command -v report >/dev/null 2>&1; then report upsampler "" "$ups_line" || true; fi
fi
`;
