// Runs a command with the mock engine up, then stops the mock whatever the
// command's outcome, and exits with the command's status:
//   node scripts/with-mock.mjs -- npx vitest run test/e2e
//   node scripts/with-mock.mjs -- npx wrangler dev --port 8787 --var TWEEN_ENGINE_URL:http://127.0.0.1:8010
// The command gets TWEEN_E2E_ENGINE_URL / TWEEN_E2E_ENGINE_KEY for the mock.
// (go-task's shell does not run EXIT traps, so a backgrounded `&` mock would outlive the task.)
import { spawn } from "node:child_process";

const sep = process.argv.indexOf("--");
const cmd = sep >= 0 ? process.argv.slice(sep + 1) : [];
if (cmd.length === 0) {
  console.error("usage: with-mock.mjs -- <command> [args...]");
  process.exit(2);
}

const port = process.env.MOCK_ENGINE_PORT ?? "8010";
const key = process.env.MOCK_ENGINE_KEY ?? "sk-local";
const url = `http://127.0.0.1:${port}`;

const mock = spawn(process.execPath, ["--experimental-strip-types", "--no-warnings", "scripts/mock-engine.ts"], {
  stdio: ["ignore", "inherit", "inherit"],
  env: { ...process.env, MOCK_ENGINE_PORT: port, MOCK_ENGINE_KEY: key },
});
let mockExited = false;
mock.on("exit", () => (mockExited = true));

const stopMock = () => {
  if (!mockExited) mock.kill("SIGTERM");
};
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => {
    stopMock();
    process.exit(130);
  });
}

async function ready() {
  for (let i = 0; i < 80; i++) {
    if (mockExited) return false;
    try {
      const r = await fetch(`${url}/v1/models`, { headers: { authorization: `Bearer ${key}` } });
      if (r.ok) return true;
    } catch {
      // not listening yet
    }
    await new Promise((res) => setTimeout(res, 250));
  }
  return false;
}

if (!(await ready())) {
  console.error(`with-mock: the mock engine did not come up on ${url}`);
  stopMock();
  process.exit(1);
}

const child = spawn(cmd[0], cmd.slice(1), {
  stdio: "inherit",
  shell: process.platform === "win32",
  env: { ...process.env, TWEEN_E2E_ENGINE_URL: url, TWEEN_E2E_ENGINE_KEY: key },
});
child.on("exit", (code, signal) => {
  stopMock();
  process.exit(code ?? (signal ? 1 : 0));
});
