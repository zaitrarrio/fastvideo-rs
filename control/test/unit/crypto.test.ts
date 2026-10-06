import { describe, expect, it } from "vitest";
import {
  hashPassphrase,
  safeEqual,
  seal,
  unseal,
  verifyPassphrase,
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
