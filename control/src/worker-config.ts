// The worker config every launch sends (docs/adr/0011-deploy-section.md, "One image with boot-time config", owner
// decision 2026-10-25). No serve image bakes a config, and an image started without one refuses to boot (fv-boot,
// crates/fastvideo-deploy boot/config.rs), so every cluster pool, standalone pod, GMI / Brev pod and serverless
// endpoint sends its config inline (FV_WORKER_TOML_B64): its own `config_toml`, or the text of the named `config`
// (a configs/serve file fv-control carries, cluster/worker-configs.ts). The name is resolved at launch, so a pool or
// endpoint picks up an edited config on its next start.
import { WORKER_CONFIGS } from "./cluster/worker-configs";
import { HttpError } from "./util";

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

/** Why an inline config is too large to send, or null. */
export function inlineSizeProblem(toml: string): string | null {
  const n = b64utf8(toml).length;
  return n > MAX_WORKER_TOML_B64 ? `the config is ${n} bytes as FV_WORKER_TOML_B64, over the ${MAX_WORKER_TOML_B64}-byte guard on one Runpod env value` : null;
}

/** Why an inline config may not say `[cluster]` yet (stage 1 of the rename, docs/adr/0011-deploy-section.md
 * "Renaming a worker-config section"). */
export const CLUSTER_TABLE_PROBLEM =
  "the config sets a [cluster] table: images built before plan F3b refuse it (fv-serve: unknown field `cluster`), and a pool can run one (a pinned sha, the stable channel); until every image reads [cluster] (docs/adr/0011-deploy-section.md, \"Renaming a worker-config section\", stage 2), write it under its old name [gateway], which every fv-serve reads";

/** Whether a worker config's TOML sets the top-level `cluster` table, by any spelling: a `[cluster]` or
 * `[cluster.x]` header, an array of tables, a quoted name, or a top-level `cluster = …` / `cluster.x = …` key before
 * the first header. A line reader (no TOML parser in fv-control): a multi-line string whose line looks like one of
 * these counts too, which only refuses a config that could be written another way. */
export function setsClusterTable(toml: string): boolean {
  const name = String.raw`(?:cluster|"cluster"|'cluster')`;
  const header = new RegExp(String.raw`^\s*\[\[?\s*${name}\s*[\].]`);
  const key = new RegExp(String.raw`^\s*${name}\s*[=.]`);
  let topLevel = true;
  for (const line of toml.split(/\r?\n/)) {
    if (header.test(line)) return true;
    if (/^\s*\[/.test(line)) topLevel = false;
    else if (topLevel && key.test(line)) return true;
  }
  return false;
}

/** Why a config cannot be sent, or null: none given, a name fv-control does not carry, over the env size guard, or
 * an inline one that says `[cluster]` before every image reads it. */
export function workerConfigProblem(x: { config?: string; config_toml?: string }): string | null {
  if (x.config_toml) return inlineSizeProblem(x.config_toml) ?? (setsClusterTable(x.config_toml) ? CLUSTER_TABLE_PROBLEM : null);
  if (!x.config) return "no worker config: config (a configs/serve file name) or config_toml (the images bake none)";
  if (!namedConfig(x.config)) return `${x.config} is not a worker config fv-control carries (configs/serve: ${CONFIG_NAMES.join(", ")}); an inline config goes in config_toml`;
  return null;
}

/** The config's text: the inline one, else the named one's. Throws a 409 when there is none (validation refuses that
 * first; a spec stored by an older fv-control can still carry one, which poolConfigIssues reports before a launch). */
export function workerConfigText(x: { config?: string; config_toml?: string }): string {
  const why = workerConfigProblem(x);
  if (why) throw new HttpError(409, `config: ${why}`, { issues: [{ path: [x.config_toml ? "config_toml" : "config"], message: why }] });
  return x.config_toml || namedConfig(x.config)!;
}

/** The engine a config runs (`[engine] backend`, fv-serve's default `cuda` when unset), or null when there is no config
 * fv-control can send. A small line reader: the worker configs keep `[engine]` a plain table. */
export function workerConfigEngine(x: { config?: string; config_toml?: string }): string | null {
  if (workerConfigProblem(x)) return null;
  const text = x.config_toml || namedConfig(x.config)!;
  let inEngine = false;
  for (const raw of text.split("\n")) {
    const l = raw.replace(/#.*$/, "").trim();
    if (l.startsWith("[")) {
      inEngine = l === "[engine]";
      continue;
    }
    const m = inEngine ? /^backend\s*=\s*["']([^"']*)["']/.exec(l) : null;
    if (m) return m[1]!;
  }
  return "cuda";
}

/** Why a CPU worker cannot run this config, or null: the CPU image has no CUDA, so CPU workers run the fake engine
 * only (as before the one image, when CPU workers ran the cpu variant and its fake config). */
export function cpuConfigProblem(x: { compute?: string; config?: string; config_toml?: string }): string | null {
  if (x.compute !== "CPU") return null;
  const e = workerConfigEngine(x);
  return e && e !== "fake" ? `CPU workers run the fake engine (the CPU image has no CUDA); this config's engine is ${e}: compute GPU, or a fake-engine config (runpod-fake.toml)` : null;
}

type PoolLike = { config?: unknown; config_toml?: unknown; compute?: unknown; provider?: unknown };
/** Why each pool of a (stored) cluster spec could not be launched as it is: a config fv-control cannot send (a path
 * outside configs/serve such as /workspace/x.toml, a name it does not carry, an inline config over the env size
 * guard: an older fv-control accepted those) or a CUDA config on CPU workers. A stored spec loads anyway (migrateSpec);
 * these surface in checkSpec (the editor and the deploy plan) and refuse the ops that create pods (assertPoolConfigs). */
export function poolConfigIssues(pools: unknown): { path: (string | number)[]; message: string }[] {
  const out: { path: (string | number)[]; message: string }[] = [];
  (Array.isArray(pools) ? pools : []).forEach((p: PoolLike, i) => {
    if (!p || typeof p !== "object") return;
    const x = { config: typeof p.config === "string" ? p.config : undefined, config_toml: typeof p.config_toml === "string" ? p.config_toml : undefined };
    const why = workerConfigProblem(x);
    if (why) out.push({ path: ["pools", i, x.config_toml ? "config_toml" : "config"], message: why });
    else {
      const cpu = p.provider && p.provider !== "runpod" ? null : cpuConfigProblem({ ...x, compute: typeof p.compute === "string" ? p.compute : undefined });
      if (cpu) out.push({ path: ["pools", i, "compute"], message: cpu });
    }
  });
  return out;
}
/** Refuses (409, typed issues) an op that would create pods from a stored spec whose pool configs cannot be sent. */
export function assertPoolConfigs(spec: { pools?: unknown }): void {
  const issues = poolConfigIssues(spec.pools);
  if (issues.length) throw new HttpError(409, `${issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ")}. Edit the cluster spec first.`, { issues });
}

/** The env that carries a launch's config: FV_WORKER_TOML_B64. Throws as workerConfigText. */
export function workerConfigEnv(x: { config?: string; config_toml?: string }): { FV_WORKER_TOML_B64: string } {
  return { FV_WORKER_TOML_B64: b64utf8(workerConfigText(x)) };
}
