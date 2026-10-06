// Bundles ui/ (esbuild, minified): public/editor.js (+ editor.css), the
// editors and the cluster configuration page; public/logs.js (+ logs.css),
// the log explorer. wrangler runs it before dev and deploy ([build] in
// wrangler.toml); app.js loads each bundle on the pages that need it.
import { build } from "esbuild";
const opts = { bundle: true, minify: true, format: "iife", target: "es2020", metafile: true, legalComments: "none", logLevel: "warning" };
for (const [entry, out] of [["ui/index.ts", "public/editor.js"], ["ui/logs/index.ts", "public/logs.js"]]) {
  const r = await build({ ...opts, entryPoints: [entry], outfile: out });
  for (const [f, o] of Object.entries(r.metafile.outputs)) console.log(`${f} ${(o.bytes / 1024).toFixed(0)} KiB`);
}
