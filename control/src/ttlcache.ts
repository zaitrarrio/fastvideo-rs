// A short in-memory cache (per isolate) for slow upstream listings: a value
// lives `ttlMs`, and concurrent callers of a missing or expired key share one
// in-flight call (the launch form's provider details and its validator ask
// for the same list at once). A failed call is not cached. `refresh` skips a
// stored value (a manual refresh) but still joins a call already in flight.
interface Entry {
  at: number;
  v?: unknown;
  has: boolean;
  p?: Promise<unknown>;
}
const store = new Map<string, Entry>();

export function ttlCached<T>(key: string, ttlMs: number, fn: () => Promise<T>, o: { refresh?: boolean } = {}): Promise<T> {
  const e = store.get(key);
  if (!o.refresh && e?.has && Date.now() - e.at < ttlMs) return Promise.resolve(e.v as T);
  if (e?.p) return e.p as Promise<T>;
  const p = fn().then(
    (v) => {
      store.set(key, { at: Date.now(), v, has: true });
      return v;
    },
    (err) => {
      // A failed refresh keeps the value it would have replaced (callers may still serve it).
      if (e?.has) store.set(key, { at: e.at, v: e.v, has: true });
      else store.delete(key);
      throw err;
    },
  );
  store.set(key, { at: e?.at ?? 0, v: e?.v, has: false, p });
  return p;
}

/** When the value under `key` was fetched (ms), or null when none is stored. */
export function ttlCachedAt(key: string): number | null {
  const e = store.get(key);
  return e?.has ? e.at : null;
}

/** Tests: forget the cached values (those whose key starts with `prefix`, or all). */
export function clearTtlCache(prefix = ""): void {
  for (const k of [...store.keys()]) if (k.startsWith(prefix)) store.delete(k);
}
