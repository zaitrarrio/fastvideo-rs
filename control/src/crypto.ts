// WebCrypto helpers (Workers and Node 22 both have X25519, AES-GCM/CTR,
// PBKDF2, HMAC, SHA-2). No secret is ever logged here.

const te = new TextEncoder();
const td = new TextDecoder();

export function b64(bytes: ArrayBuffer | Uint8Array): string {
  const u = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let s = "";
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
  return btoa(s);
}
export function unb64(s: string): Uint8Array<ArrayBuffer> {
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/").replace(/\s+/g, ""));
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}
export function b64url(bytes: ArrayBuffer | Uint8Array): string {
  return b64(bytes).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
export function hex(bytes: ArrayBuffer | Uint8Array): string {
  const u = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return Array.from(u, (b) => b.toString(16).padStart(2, "0")).join("");
}
export function randomBytes(n: number): Uint8Array<ArrayBuffer> {
  const u = new Uint8Array(n);
  crypto.getRandomValues(u);
  return u;
}
/** A random token: `<prefix><hex>` (32 bytes by default, as the cluster script's `openssl rand -hex 32`). */
export function randomToken(prefix = "", bytes = 32): string {
  return prefix + hex(randomBytes(bytes));
}
export async function sha256Hex(s: string | Uint8Array<ArrayBuffer>): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", typeof s === "string" ? te.encode(s) : s));
}
/** Constant-time string comparison. */
export function safeEqual(a: string, b: string): boolean {
  const x = te.encode(a);
  const y = te.encode(b);
  let d = x.length ^ y.length;
  const n = Math.max(x.length, y.length);
  for (let i = 0; i < n; i++) d |= (x[i] ?? 0) ^ (y[i] ?? 0);
  return d === 0;
}

async function hmacKey(secret: string | Uint8Array<ArrayBuffer>): Promise<CryptoKey> {
  const raw = typeof secret === "string" ? te.encode(secret) : secret;
  return crypto.subtle.importKey("raw", raw, { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}
export async function hmacB64url(secret: string, msg: string): Promise<string> {
  return b64url(await crypto.subtle.sign("HMAC", await hmacKey(secret), te.encode(msg)));
}

// --- Owner passphrase: PBKDF2-SHA256 with a random salt and a pepper
// (SESSION_SECRET) mixed in with HMAC first, so a leaked D1/secret listing
// of the hash alone cannot be brute-forced offline. Workers cap PBKDF2 at
// 100 000 iterations; the passphrase is generated with >= 128 bits of
// entropy (docs/control/README.md), which is what really protects it.
export const PBKDF2_ITER = 100_000;

async function pbkdf2(pass: string, pepper: string, salt: Uint8Array<ArrayBuffer>, iter: number): Promise<Uint8Array<ArrayBuffer>> {
  const peppered = new Uint8Array(await crypto.subtle.sign("HMAC", await hmacKey(pepper), te.encode(pass)));
  const key = await crypto.subtle.importKey("raw", peppered, "PBKDF2", false, ["deriveBits"]);
  return new Uint8Array(await crypto.subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt, iterations: iter }, key, 256));
}
export async function hashPassphrase(pass: string, pepper: string, iter = PBKDF2_ITER): Promise<string> {
  const salt = randomBytes(16);
  return `pbkdf2-sha256$${iter}$${b64(salt)}$${b64(await pbkdf2(pass, pepper, salt, iter))}`;
}
export async function verifyPassphrase(pass: string, pepper: string, stored: string): Promise<boolean> {
  const [alg, iterS, saltS, hashS] = stored.trim().split("$");
  if (alg !== "pbkdf2-sha256" || !iterS || !saltS || !hashS) return false;
  const iter = Number(iterS);
  if (!Number.isInteger(iter) || iter < 10_000 || iter > 100_000) return false;
  const got = await pbkdf2(pass, pepper, unb64(saltS), iter);
  return safeEqual(b64(got), hashS);
}

// --- Sealing values at rest in D1 (cluster secrets, secret env values):
// AES-256-GCM under CONTROL_KEK, `v1.<iv b64>.<ct b64>`, with the row's
// identity as associated data so a sealed value cannot be moved to another row.
async function kek(kekB64: string): Promise<CryptoKey> {
  const raw = unb64(kekB64);
  if (raw.length < 32) throw new Error("CONTROL_KEK must be at least 32 bytes (base64)");
  return crypto.subtle.importKey("raw", raw.subarray(0, 32), "AES-GCM", false, ["encrypt", "decrypt"]);
}
export async function seal(kekB64: string, plain: string, aad: string): Promise<string> {
  const iv = randomBytes(12);
  const ct = await crypto.subtle.encrypt({ name: "AES-GCM", iv, additionalData: te.encode(aad) }, await kek(kekB64), te.encode(plain));
  return `v1.${b64(iv)}.${b64(ct)}`;
}
export async function unseal(kekB64: string, sealed: string, aad: string): Promise<string> {
  const [v, ivS, ctS] = sealed.split(".");
  if (v !== "v1" || !ivS || !ctS) throw new Error("not a sealed value");
  const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv: unb64(ivS), additionalData: te.encode(aad) }, await kek(kekB64), unb64(ctS));
  return td.decode(pt);
}
