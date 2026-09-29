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

// --- The gateway's sealed admin token (docs/serve/gateway.md §9,
// crates/fastvideo-serve/src/admin_token.rs): s = X25519(key, epk);
// k = SHA-512("fv-admin-token-v1" | s | epk | recipient public key);
// token = AES-256-CTR(k[0..32], iv), tag = HMAC-SHA256(k[32..64], iv | ct).
export interface X25519Pair {
  publicRaw: string; // base64, 32 bytes: FV_ADMIN_TOKEN_RECIPIENT
  privatePkcs8: string; // base64 (sealed in D1)
}
export async function x25519Generate(): Promise<X25519Pair> {
  const kp = (await crypto.subtle.generateKey({ name: "X25519" } as any, true, ["deriveBits"])) as CryptoKeyPair;
  const pub = (await crypto.subtle.exportKey("raw", kp.publicKey)) as ArrayBuffer;
  const priv = (await crypto.subtle.exportKey("pkcs8", kp.privateKey)) as ArrayBuffer;
  return { publicRaw: b64(pub), privatePkcs8: b64(priv) };
}
/** The raw public key of a PKCS#8 X25519 private key (the script's .pem, as DER). */
export async function x25519PublicOf(privatePkcs8: string): Promise<string> {
  const k = await crypto.subtle.importKey("pkcs8", unb64(privatePkcs8), { name: "X25519" } as any, true, ["deriveBits"]);
  const jwk = (await crypto.subtle.exportKey("jwk", k)) as JsonWebKey;
  return b64(unb64(jwk.x!));
}
export interface SealedToken {
  alg: string;
  epk: string;
  iv: string;
  ct: string;
  tag: string;
}
export async function openSealedToken(sealed: SealedToken, privatePkcs8: string, publicRaw: string): Promise<string> {
  if (sealed.alg !== "X25519-SHA512-AES256CTR-HMACSHA256") throw new Error("unknown sealed-token alg");
  const epk = unb64(sealed.epk);
  const iv = unb64(sealed.iv);
  const ct = unb64(sealed.ct);
  const priv = await crypto.subtle.importKey("pkcs8", unb64(privatePkcs8), { name: "X25519" } as any, false, ["deriveBits"]);
  const peer = await crypto.subtle.importKey("raw", epk, { name: "X25519" } as any, false, []);
  const shared = new Uint8Array(await crypto.subtle.deriveBits({ name: "X25519", public: peer } as any, priv, 256));
  const rpk = unb64(publicRaw);
  const label = te.encode("fv-admin-token-v1");
  const input = new Uint8Array(label.length + shared.length + epk.length + rpk.length);
  input.set(label, 0);
  input.set(shared, label.length);
  input.set(epk, label.length + shared.length);
  input.set(rpk, label.length + shared.length + epk.length);
  const k = new Uint8Array(await crypto.subtle.digest("SHA-512", input));
  const macKey = await hmacKey(k.subarray(32, 64));
  const ivct = new Uint8Array(iv.length + ct.length);
  ivct.set(iv, 0);
  ivct.set(ct, iv.length);
  const ok = await crypto.subtle.verify("HMAC", macKey, unb64(sealed.tag), ivct);
  if (!ok) throw new Error("sealed token: bad tag");
  const aes = await crypto.subtle.importKey("raw", k.subarray(0, 32), "AES-CTR", false, ["decrypt"]);
  const pt = await crypto.subtle.decrypt({ name: "AES-CTR", counter: iv, length: 128 }, aes, ct);
  return td.decode(pt);
}
/** The same sealing, for tests and the mock gateway. */
export async function sealTokenFor(plain: string, recipientRaw: string): Promise<SealedToken> {
  const eph = (await crypto.subtle.generateKey({ name: "X25519" } as any, true, ["deriveBits"])) as CryptoKeyPair;
  const epk = new Uint8Array((await crypto.subtle.exportKey("raw", eph.publicKey)) as ArrayBuffer);
  const rpk = unb64(recipientRaw);
  const peer = await crypto.subtle.importKey("raw", rpk, { name: "X25519" } as any, false, []);
  const shared = new Uint8Array(await crypto.subtle.deriveBits({ name: "X25519", public: peer } as any, eph.privateKey, 256));
  const label = te.encode("fv-admin-token-v1");
  const input = new Uint8Array([...label, ...shared, ...epk, ...rpk]);
  const k = new Uint8Array(await crypto.subtle.digest("SHA-512", input));
  const iv = randomBytes(16);
  const aes = await crypto.subtle.importKey("raw", k.subarray(0, 32), "AES-CTR", false, ["encrypt"]);
  const ct = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-CTR", counter: iv, length: 128 }, aes, te.encode(plain)));
  const tag = await crypto.subtle.sign("HMAC", await hmacKey(k.subarray(32, 64)), new Uint8Array([...iv, ...ct]));
  return { alg: "X25519-SHA512-AES256CTR-HMACSHA256", epk: b64(epk), iv: b64(iv), ct: b64(ct), tag: b64(tag) };
}
