// The worker config every launch sends (docs/adr/0011-deploy-section.md, "One image with boot-time config", owner
// decision 2026-10-25). No serve image bakes a config, and an image started without one refuses to boot (fv-boot,
// crates/fastvideo-deploy boot/config.rs), so every cluster pool, standalone pod, GMI / Brev pod and serverless
// endpoint sends its config inline (FV_WORKER_TOML_B64): its own `config_toml`, or the text of the named `config`
// (a configs/serve file fv-control carries, cluster/worker-configs.ts). The name is resolved at launch, so a pool or
// endpoint picks up an edited config on its next start.
import { WORKER_CONFIGS } from "./cluster/worker-configs";

/**
 * The largest FV_WORKER_TOML_B64 a launch sends (bytes of base64). Runpod documents no limit on one env value (only
 * 50 variables per pod, docs.runpod.io/pods/templates/environment-variables, and a 1 MiB request body on the v2
 * API); this is a conservative guard, about three times the largest committed config (runpod-family-h3.toml, 5600
 * bytes encoded). crates/fastvideo-deploy launch.rs MAX_WORKER_TOML_B64 is the same (a unit test compares them).
 */
export const MAX_WORKER_TOML_B64 = 16384;
/** The largest inline config_toml the schemas accept (UTF-8 bytes whose base64 fits MAX_WORKER_TOML_B64). */
export const MAX_WORKER_TOML_BYTES = (MAX_WORKER_TOML_B64 / 4) * 3;

/** The file name of a named config (`/etc/fv/runpod-ltx.toml` and `runpod-ltx.toml` name the same). */
export const configFile = (config: string): string => config.slice(config.lastIndexOf("/") + 1);
/** The text of a named config fv-control carries, or undefined. */
export const namedConfig = (config?: string): string | undefined => (config ? WORKER_CONFIGS[configFile(config)] : undefined);
/** The configs a `config` may name (configs/serve/runpod*.toml). */
export const CONFIG_NAMES: string[] = Object.keys(WORKER_CONFIGS);

/** UTF-8 base64 (what FV_WORKER_TOML_B64 carries). */
export function b64utf8(s: string): string {
  const u = new TextEncoder().encode(s);
  let bin = "";
  for (const x of u) bin += String.fromCharCode(x);
  return btoa(bin);
}

/** Why a config cannot be sent, or null: none given, a name fv-control does not carry, or over the env size guard. */
export function workerConfigProblem(x: { config?: string; config_toml?: string }): string | null {
  if (x.config_toml) {
    const n = b64utf8(x.config_toml).length;
    return n > MAX_WORKER_TOML_B64 ? `the config is ${n} bytes as FV_WORKER_TOML_B64, over the ${MAX_WORKER_TOML_B64}-byte guard on one Runpod env value` : null;
  }
  if (!x.config) return "no worker config: config (a configs/serve file name) or config_toml (the images bake none)";
  if (!namedConfig(x.config)) return `${x.config} is not a worker config fv-control carries (configs/serve: ${CONFIG_NAMES.join(", ")}); an inline config goes in config_toml`;
  return null;
}

/** The config's text: the inline one, else the named one's. Throws when there is none (validation refuses that first). */
export function workerConfigText(x: { config?: string; config_toml?: string }): string {
  const why = workerConfigProblem(x);
  if (why) throw new Error(why);
  return x.config_toml || namedConfig(x.config)!;
}

/** The env that carries a launch's config: FV_WORKER_TOML_B64. Throws as workerConfigText. */
export function workerConfigEnv(x: { config?: string; config_toml?: string }): { FV_WORKER_TOML_B64: string } {
  return { FV_WORKER_TOML_B64: b64utf8(workerConfigText(x)) };
}
