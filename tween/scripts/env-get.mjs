// Prints one value from .env (or the process environment), else a default:
//   node scripts/env-get.mjs TWEEN_ENGINE_KEY sk-local
// The Taskfile uses it because an included Taskfile cannot declare dotenv.
import { existsSync, readFileSync } from "node:fs";

const [name, fallback = ""] = process.argv.slice(2);
if (!name) {
  console.error("usage: env-get.mjs NAME [default]");
  process.exit(2);
}

let value = process.env[name];
if (!value && existsSync(".env")) {
  for (const raw of readFileSync(".env", "utf8").split("\n")) {
    const line = raw.trim();
    if (!line.startsWith(`${name}=`)) continue;
    let v = line.slice(name.length + 1).trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
    value = v;
  }
}
process.stdout.write(value || fallback);
