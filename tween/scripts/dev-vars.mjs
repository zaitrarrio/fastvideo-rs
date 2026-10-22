// Writes .dev.vars (read by `wrangler dev`) from .env, keeping only the
// Worker's own variables. Empty values are left out so the Worker's defaults
// (the fake engine's model ids, no sign-in) apply.
import { existsSync, readFileSync, writeFileSync } from "node:fs";

const WORKER_VARS = [
  "TWEEN_ENGINE_URL",
  "TWEEN_ENGINE_KEY",
  "TWEEN_ACCESS_TOKEN",
  "TWEEN_LTX_DRAFT_MODEL",
  "TWEEN_LTX_FINAL_MODEL",
  "TWEEN_H3_DRAFT_MODEL",
  "TWEEN_H3_FINAL_MODEL",
  "TWEEN_H3_REFERENCE_MODEL",
];

if (!existsSync(".env")) {
  console.error("no .env: copy .env.example first (task tween:env)");
  process.exit(1);
}

const values = new Map();
for (const raw of readFileSync(".env", "utf8").split("\n")) {
  const line = raw.trim();
  if (!line || line.startsWith("#")) continue;
  const eq = line.indexOf("=");
  if (eq < 0) continue;
  let v = line.slice(eq + 1).trim();
  if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
  values.set(line.slice(0, eq).trim(), v);
}

const out = WORKER_VARS.filter((k) => values.get(k)).map((k) => `${k}=${JSON.stringify(values.get(k))}`);
writeFileSync(".dev.vars", `# Generated from .env by scripts/dev-vars.mjs; edit .env instead.\n${out.join("\n")}\n`);
console.log(`.dev.vars: ${out.length} variable(s)`);
