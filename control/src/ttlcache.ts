// A short in-memory cache (per isolate) for slow upstream listings: a value
// lives `ttlMs`, and concurrent callers of a missing or expired key share one
// in-flight call (the launch form's provider details and its validator ask
// for the same list at once). A failed call is not cached.
interface Entry {
  at: number;
  v?: unknown;
  has: boolean;
  p?: Promise<unknown>;
}
const store = new Map<string, Entry>();

export function ttlCached<T>(key: string, ttlMs: number, fn: () => Promise<T>): Promise<T> {
  const e = store.get(key);
  if (e?.has && Date.now() - e.at < ttlMs) return Promise.resolve(e.v as T);
  if (e?.p) return e.p as Promise<T>;
  const p = fn().then(
    (v) => {
      store.set(key, { at: Date.now(), v, has: true });
      return v;
    },
    (err) => {
      store.delete(key);
      throw err;
    },
  );
  store.set(key, { at: e?.at ?? 0, v: e?.v, has: false, p });
  return p;
}

/** Tests: forget the cached values (those whose key starts with `prefix`, or all). */
export function clearTtlCache(prefix = ""): void {
  for (const k of [...store.keys()]) if (k.startsWith(prefix)) store.delete(k);
}
