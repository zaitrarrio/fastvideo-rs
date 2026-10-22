// In-memory R2Bucket covering what Tween calls: put, get (with a range).
type Range = { offset?: number; length?: number; suffix?: number };

export function memoryR2() {
  const objects = new Map<string, { bytes: Uint8Array; contentType?: string }>();
  const bucket = {
    async put(key: string, value: ArrayBuffer | Uint8Array | string, opts?: { httpMetadata?: { contentType?: string } }) {
      const bytes =
        typeof value === "string" ? new TextEncoder().encode(value) : value instanceof Uint8Array ? value : new Uint8Array(value);
      objects.set(key, { bytes: new Uint8Array(bytes), contentType: opts?.httpMetadata?.contentType });
      return { key, size: bytes.length };
    },
    async get(key: string, opts?: { range?: Range }) {
      const o = objects.get(key);
      if (!o) return null;
      const size = o.bytes.length;
      let start = 0;
      let end = size;
      const range = opts?.range;
      if (range) {
        if (range.suffix !== undefined) start = Math.max(0, size - range.suffix);
        else {
          start = range.offset ?? 0;
          end = range.length !== undefined ? Math.min(size, start + range.length) : size;
        }
      }
      const slice = o.bytes.slice(start, end);
      return {
        key,
        size,
        httpEtag: `"${key.length}-${size}"`,
        httpMetadata: { contentType: o.contentType },
        range,
        body: new Response(slice).body,
        arrayBuffer: async () => slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength),
      };
    },
  };
  return { bucket: bucket as unknown as R2Bucket, objects };
}
