// Packs the wasm-bindgen output of fastvideo-edge into a Worker module,
// the same steps as worker-build 0.8.7 (main.rs: generate_handlers,
// add_export_wrappers, bundle, fix_wasm_import, remove_unused_files), so no
// Rust tool has to be compiled here:
//
//   node pack.mjs <dir> <esbuild module dir>
//
// <dir> holds index.js + index_bg.wasm from
// `wasm-bindgen --target module --out-name index --no-typescript
//  --experimental-reset-state-function --force-enable-abort-handler`;
// afterwards it holds the bundled index.js, index_bg.wasm and
// worker/shim.mjs.
import { readFileSync, writeFileSync, rmSync, mkdirSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const [dir, esbuildDir] = process.argv.slice(2);
if (!dir || !esbuildDir) {
  console.error("usage: node pack.mjs <dir> <esbuild module dir>");
  process.exit(2);
}
const here = dirname(fileURLToPath(import.meta.url));
const index = readFileSync(join(dir, "index.js"), "utf8");

const SYSTEM = new Set(["__wbg_reset_state", "__worker_init_state"]);
const funcs = [];
const classes = [];
for (const m of index.matchAll(/(?:^|[\s;}])export (async function|function|class) ([A-Za-z0-9_$]+)/g)) {
  const [, kind, name] = m;
  if (kind === "class") classes.push(name);
  else if (!SYSTEM.has(name)) funcs.push(name);
}
for (const m of index.matchAll(/(?:^|[\s;}])export \{[^}]* as ([A-Za-z0-9_$]+)\s*\}/g)) {
  if (!SYSTEM.has(m[1])) funcs.push(m[1]);
}

let handlers = "";
for (const f of funcs) {
  if (["fetch", "queue", "scheduled", "email", "connect"].includes(f)) {
    handlers += `Entrypoint.prototype.${f} = function ${f} (arg) {\n  return exports.${f}.call(this, arg, this.env, this.ctx);\n}\n`;
  } else {
    handlers += `Entrypoint.prototype.${f} = exports.${f};\n`;
  }
}
let shim = readFileSync(join(here, "shim.js"), "utf8").replace("$HANDLERS", () => handlers);
for (const c of classes) {
  shim += `export const ${c} = new Proxy(exports.${c}, classProxyHooks);\n`;
}
writeFileSync(join(dir, "shim.js"), shim);

const esbuild = await import(pathToFileURL(join(esbuildDir, "lib", "main.js")).href);
await esbuild.build({
  absWorkingDir: dir,
  entryPoints: ["./shim.js"],
  outfile: join(dir, "index.js"),
  external: ["./index_bg.wasm", "cloudflare:*"],
  format: "esm",
  bundle: true,
  minify: true,
  allowOverwrite: true,
  logLevel: "warning",
});

const bundled = readFileSync(join(dir, "index.js"), "utf8").replaceAll("import source ", "import ");
writeFileSync(join(dir, "index.js"), bundled);
rmSync(join(dir, "shim.js"));
if (existsSync(join(dir, "snippets"))) rmSync(join(dir, "snippets"), { recursive: true });
mkdirSync(join(dir, "worker"), { recursive: true });
writeFileSync(
  join(dir, "worker", "shim.mjs"),
  "// Use index.js directly, this file provided for backwards compat\n// with former shim.mjs only.\nexport * from '../index.js';\nexport { default } from '../index.js';\n",
);
console.log(`packed: handlers [${funcs.join(", ")}], classes [${classes.join(", ")}]`);
