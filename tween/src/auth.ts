// Optional shared-token gate. Tween holds the engine key, so an open
// deployment lets anyone spend GPU time: staging sets TWEEN_ACCESS_TOKEN.
// Signing in exchanges the token for an HttpOnly cookie so <img> and <video>
// requests are authorised without putting the token in URLs.

export const COOKIE = "tween_session";
const MAX_AGE_S = 60 * 60 * 24 * 30;

async function sha256Hex(s: string): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Cookie value for a token: a digest, so the cookie never carries the token. */
export const sessionValue = (token: string) => sha256Hex(`tween-session:v1:${token}`);

function equal(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export function readCookie(header: string | null, name: string): string | null {
  if (!header) return null;
  for (const part of header.split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k === name) return v.join("=");
  }
  return null;
}

export async function isAuthorised(token: string | undefined, req: Request): Promise<boolean> {
  if (!token) return true;
  const got = readCookie(req.headers.get("cookie"), COOKIE);
  return got !== null && equal(got, await sessionValue(token));
}

export async function tokenMatches(token: string, given: unknown): Promise<boolean> {
  if (typeof given !== "string" || !given) return false;
  return equal(await sessionValue(given), await sessionValue(token));
}

export function sessionCookie(value: string, secure: boolean): string {
  return `${COOKIE}=${value}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${MAX_AGE_S}${secure ? "; Secure" : ""}`;
}

export const clearedCookie = (secure: boolean) =>
  `${COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0${secure ? "; Secure" : ""}`;
