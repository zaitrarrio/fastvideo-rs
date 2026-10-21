// Panel images and rendered clips in R2.

export const panelKey = (boardId: string, imageId: string) => `boards/${boardId}/panels/${imageId}`;
export const clipKey = (boardId: string, renderId: string) => `boards/${boardId}/clips/${renderId}.mp4`;

/** Base64 without spreading the whole buffer into one call (large images). */
export function base64(bytes: Uint8Array): string {
  let s = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}

export const dataUri = (bytes: Uint8Array, mime: string) => `data:${mime};base64,${base64(bytes)}`;

/** A single `bytes=a-b` / `bytes=a-` / `bytes=-n` range; null for none or multi-range. */
export function parseRange(header: string | null): R2Range | null {
  if (!header) return null;
  const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!m) return null;
  const [, a, b] = m;
  if (a === "" && b === "") return null;
  if (a === "") return { suffix: Number(b) };
  const offset = Number(a);
  if (b === "") return { offset };
  const end = Number(b);
  if (end < offset) return null;
  return { offset, length: end - offset + 1 };
}

/** Serves an R2 object, honouring a single byte range so video elements can seek. */
export async function serveObject(bucket: R2Bucket, key: string, req: Request, fallbackType: string): Promise<Response> {
  const range = parseRange(req.headers.get("range"));
  const obj = await bucket.get(key, range ? { range } : undefined);
  if (!obj) return new Response("not found", { status: 404 });
  const headers = new Headers({
    "content-type": obj.httpMetadata?.contentType || fallbackType,
    "accept-ranges": "bytes",
    "cache-control": "private, max-age=3600",
    etag: obj.httpEtag,
  });
  const body = (obj as R2ObjectBody).body;
  if (range && obj.range) {
    const r = obj.range as { offset?: number; length?: number; suffix?: number };
    const offset = r.suffix !== undefined ? obj.size - r.suffix : (r.offset ?? 0);
    const length = r.suffix !== undefined ? r.suffix : (r.length ?? obj.size - offset);
    headers.set("content-range", `bytes ${offset}-${offset + length - 1}/${obj.size}`);
    headers.set("content-length", String(length));
    return new Response(body, { status: 206, headers });
  }
  headers.set("content-length", String(obj.size));
  return new Response(body, { status: 200, headers });
}
