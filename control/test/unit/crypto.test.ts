import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  hashPassphrase,
  openSealedToken,
  safeEqual,
  seal,
  sealTokenFor,
  unseal,
  verifyPassphrase,
  x25519Generate,
  x25519PublicOf,
  b64,
  randomBytes,
} from "../../src/crypto";

const KEK = b64(randomBytes(32));
const repo = new URL("../../../", import.meta.url).pathname;

describe("sealing at rest", () => {
  it("round-trips and binds the row identity", async () => {
    const s = await seal(KEK, "hunter2", "env:account::X");
    expect(s.startsWith("v1.")).toBe(true);
    expect(s).not.toContain("hunter2");
    expect(await unseal(KEK, s, "env:account::X")).toBe("hunter2");
    await expect(unseal(KEK, s, "env:account::Y")).rejects.toThrow();
    await expect(unseal(b64(randomBytes(32)), s, "env:account::X")).rejects.toThrow();
  });
});

describe("owner passphrase", () => {
  it("verifies the right passphrase only, and needs the pepper", async () => {
    const h = await hashPassphrase("correct horse battery staple", "pepper", 20000);
    expect(h).toMatch(/^pbkdf2-sha256\$20000\$/);
    expect(await verifyPassphrase("correct horse battery staple", "pepper", h)).toBe(true);
    expect(await verifyPassphrase("correct horse battery stapl", "pepper", h)).toBe(false);
    expect(await verifyPassphrase("correct horse battery staple", "other", h)).toBe(false);
    expect(await verifyPassphrase("x", "pepper", "md5$1$a$b")).toBe(false);
  });
  it("safeEqual", () => {
    expect(safeEqual("abc", "abc")).toBe(true);
    expect(safeEqual("abc", "abd")).toBe(false);
    expect(safeEqual("abc", "abcd")).toBe(false);
    expect(safeEqual("", "")).toBe(true);
  });
});

describe("the gateway's sealed admin token (gateway.md §9)", () => {
  it("round-trips and rejects a tampered tag", async () => {
    const kp = await x25519Generate();
    expect(await x25519PublicOf(kp.privatePkcs8)).toBe(kp.publicRaw);
    const sealed = await sealTokenFor("fvadm_secret123", kp.publicRaw);
    expect(await openSealedToken(sealed, kp.privatePkcs8, kp.publicRaw)).toBe("fvadm_secret123");
    const bad = { ...sealed, ct: b64(randomBytes(15)) };
    await expect(openSealedToken(bad, kp.privatePkcs8, kp.publicRaw)).rejects.toThrow(/tag/);
  });

  it("interoperates with runpod-cluster.sh's openssl open_sealed and its key files", async () => {
    const dir = mkdtempSync(join(tmpdir(), "fvc-"));
    // A key made by openssl as the script does; the controller imports its PKCS#8.
    execFileSync("openssl", ["genpkey", "-algorithm", "X25519", "-out", join(dir, "k.pem")]);
    const pem = readFileSync(join(dir, "k.pem"), "utf8");
    const pkcs8 = pem.replace(/-----[^-]+-----/g, "").replace(/\s+/g, "");
    const scriptPub = execFileSync("bash", ["-c", `openssl pkey -in ${dir}/k.pem -pubout -outform DER | tail -c 32 | base64 -w0`]).toString();
    expect(await x25519PublicOf(pkcs8)).toBe(scriptPub);
    // Sealed here, opened by the script's function.
    const sealed = await sealTokenFor("fvadm_from_ts", scriptPub);
    const src = readFileSync(join(repo, "scripts/serve/runpod-cluster.sh"), "utf8");
    const fn = /\nopen_sealed\(\) \{[\s\S]*?\n\}\n/.exec(src)![0];
    writeFileSync(join(dir, "sealed.json"), JSON.stringify(sealed));
    const out = execFileSync("bash", ["-c", `set -euo pipefail; ADMIN_KEY=${dir}/k.pem
${fn}
open_sealed < ${dir}/sealed.json`]).toString();
    expect(out).toBe("fvadm_from_ts");
  });
});
