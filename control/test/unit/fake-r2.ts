// A fake R2 for the weights-store tests (r2weights.test.ts, serverless-r2.test.ts): path-style objects of bucket
// fv-weights served only through presigned GETs, checked with node:crypto (an implementation independent of src's),
// with Range; `routes` adds fv-control's side (the plan, the fetcher, the reports). Trees with their manifests.
import { createHash, createHmac } from "node:crypto";
import { createServer, type Server } from "node:http";
import type { R2Manifest } from "../../src/r2weights";
export const ACCESS = "r2-access-key-id";
export const SECRET = "r2-SECRET-access-key-0123456789";
export const sha = (b: Buffer | string) => createHash("sha256").update(b).digest("hex");

/** A tree, its manifest (part size 1 MiB) and the R2_TREES pin. */
export function makeTree(dest: string, rev: string, files: Record<string, Buffer>, part = 1 << 20) {
  const m: R2Manifest = {
    format: "fv-weights-manifest/1",
    dest,
    rev,
    part_size: part,
    total_bytes: Object.values(files).reduce((s, b) => s + b.length, 0),
    files: Object.entries(files).map(([path, b]) => ({ path, size: b.length, sha256: sha(b), parts: Array.from({ length: Math.ceil(b.length / part) }, (_, i) => sha(b.subarray(i * part, (i + 1) * part))) })),
  };
  const raw = Buffer.from(JSON.stringify(m));
  return { m, raw, pin: { rev, files: m.files.length, bytes: m.total_bytes, manifest_sha256: sha(raw), source: "test" } };
}
export const rnd = (n: number, seed: number) => {
  const out = Buffer.alloc(n);
  let h = createHash("sha256").update(String(seed)).digest();
  for (let o = 0; o < n; o += 32) {
    h = createHash("sha256").update(h).digest();
    h.copy(out, o);
  }
  return out;
};

/** A fake R2: path-style objects, presigned GETs checked with node:crypto (an implementation independent of src's). */
export function fakeR2() {
  const objects = new Map<string, Buffer>(); // "<bucket>/<key>"
  const gets: { key: string; range: string | null }[] = [];
  let bad = 0;
  const routes: Record<string, (req: any, res: any) => void> = {};
  const verify = (u: URL, host: string) => {
    const q = u.searchParams;
    const amz = q.get("X-Amz-Date") || "";
    const when = Date.UTC(+amz.slice(0, 4), +amz.slice(4, 6) - 1, +amz.slice(6, 8), +amz.slice(9, 11), +amz.slice(11, 13), +amz.slice(13, 15));
    if (Date.now() > when + Number(q.get("X-Amz-Expires")) * 1000) return false;
    const [ak, date, region] = (q.get("X-Amz-Credential") || "").split("/");
    const pairs = [...q.entries()].filter(([k]) => k !== "X-Amz-Signature").map(([k, v]) => [encodeURIComponent(k), encodeURIComponent(v)]).sort((a, b) => (a[0]! < b[0]! ? -1 : 1));
    const canonical = ["GET", u.pathname, pairs.map(([k, v]) => `${k}=${v}`).join("&"), `host:${host}\n`, "host", "UNSIGNED-PAYLOAD"].join("\n");
    const toSign = ["AWS4-HMAC-SHA256", amz, `${date}/${region}/s3/aws4_request`, sha(canonical)].join("\n");
    let k: Buffer = createHmac("sha256", "AWS4" + SECRET).update(date!).digest();
    for (const p of [region!, "s3", "aws4_request"]) k = createHmac("sha256", k).update(p).digest();
    return ak === ACCESS && createHmac("sha256", k).update(toSign).digest("hex") === q.get("X-Amz-Signature");
  };
  const srv: Server = createServer((req, res) => {
    const u = new URL(req.url!, "http://x");
    if (routes[u.pathname]) return routes[u.pathname]!(req, res);
    if (!verify(u, req.headers.host!)) {
      bad++;
      return res.writeHead(403).end("<Error><Code>AccessDenied</Code></Error>");
    }
    const key = decodeURIComponent(u.pathname.slice(1));
    const obj = objects.get(key);
    if (!obj) return res.writeHead(404).end("<Error><Code>NoSuchKey</Code></Error>");
    const m = /bytes=(\d+)-(\d+)/.exec(req.headers.range || "");
    gets.push({ key, range: req.headers.range || null });
    if (!m) return res.writeHead(200, { "content-length": obj.length }).end(obj);
    const a = +m[1]!, b = Math.min(+m[2]!, obj.length - 1);
    res.writeHead(206, { "content-length": b - a + 1, "content-range": `bytes ${a}-${b}/${obj.length}` }).end(obj.subarray(a, b + 1));
  });
  return {
    objects,
    gets,
    routes,
    bad: () => bad,
    async start() {
      await new Promise<void>((r) => srv.listen(0, "127.0.0.1", () => r()));
      return `http://127.0.0.1:${(srv.address() as any).port}`;
    },
    stop: () => new Promise<void>((r) => srv.close(() => r())),
  };
}
