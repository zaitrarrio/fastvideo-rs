// Bundles ui/ into public/editor.js (esbuild, minified). wrangler runs it
// before dev and deploy ([build] in wrangler.toml).
import { build } from "esbuild";
const r = await build({ entryPoints: ["ui/index.ts"], bundle: true, minify: true, format: "iife", target: "es2020", outfile: "public/editor.js", metafile: true, legalComments: "none", logLevel: "warning" });
const bytes = Object.values(r.metafile.outputs)[0].bytes;
console.log(`public/editor.js ${(bytes / 1024).toFixed(0)} KiB`);
