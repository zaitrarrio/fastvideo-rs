// Waits until a URL answers 2xx, from the host (the serve image is not
// assumed to carry curl or wget):  node scripts/wait-http.mjs URL [seconds]
const [url, seconds = "60"] = process.argv.slice(2);
if (!url) {
  console.error("usage: wait-http.mjs URL [seconds]");
  process.exit(2);
}
const deadline = Date.now() + Number(seconds) * 1000;
let last = "no answer";
while (Date.now() < deadline) {
  try {
    const r = await fetch(url);
    if (r.ok) {
      console.log(`ready: ${url}`);
      process.exit(0);
    }
    last = `HTTP ${r.status}`;
  } catch (e) {
    last = e instanceof Error ? e.message : String(e);
  }
  await new Promise((res) => setTimeout(res, 500));
}
console.error(`timed out after ${seconds}s waiting for ${url} (${last})`);
process.exit(1);
