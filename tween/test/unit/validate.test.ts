import { describe, expect, it } from "vitest";
import { parseBoardInput, parseBoardPatch, parseImageUpload, parseRenderRequest, parseShotInput } from "../../src/validate";

describe("parseBoardInput", () => {
  it("accepts a title and a known aspect ratio, trimming the title", () => {
    expect(parseBoardInput({ title: "  Heist  ", aspect_ratio: "9:16" })).toEqual({ ok: true, value: { title: "Heist", aspect_ratio: "9:16" } });
  });
  it("defaults the aspect to 16:9", () => {
    expect(parseBoardInput({ title: "x" })).toEqual({ ok: true, value: { title: "x", aspect_ratio: "16:9" } });
  });
  it("refuses an empty title, a long title and an unknown aspect", () => {
    expect(parseBoardInput({ title: " " }).ok).toBe(false);
    expect(parseBoardInput({ title: "x".repeat(121) }).ok).toBe(false);
    expect(parseBoardInput({ title: "x", aspect_ratio: "2:1" }).ok).toBe(false);
    expect(parseBoardInput(null).ok).toBe(false);
  });
  it("patches only what is given", () => {
    expect(parseBoardPatch({ aspect_ratio: "1:1" })).toEqual({ ok: true, value: { aspect_ratio: "1:1" } });
    expect(parseBoardPatch({}).ok).toBe(false);
  });
});

describe("parseShotInput", () => {
  it("accepts full settings", () => {
    expect(parseShotInput({ prompt: "pan left", seconds: 8, mode: "reference", seed: 12, reference_panel_ids: ["pn_a"] })).toEqual({
      ok: true,
      value: { prompt: "pan left", seconds: 8, mode: "reference", seed: 12, reference_panel_ids: ["pn_a"] },
    });
  });
  it("fills defaults and keeps a null seed", () => {
    expect(parseShotInput({})).toEqual({
      ok: true,
      value: { prompt: "", seconds: 6, mode: "keyframes", seed: null, reference_panel_ids: [] },
    });
  });
  it("refuses fractional or out-of-range seconds, bad seeds, bad modes and too many references", () => {
    expect(parseShotInput({ seconds: 6.5 }).ok).toBe(false);
    expect(parseShotInput({ seconds: 0 }).ok).toBe(false);
    expect(parseShotInput({ seconds: 31 }).ok).toBe(false);
    expect(parseShotInput({ seed: -1 }).ok).toBe(false);
    expect(parseShotInput({ seed: 2 ** 31 }).ok).toBe(false);
    expect(parseShotInput({ mode: "mid" }).ok).toBe(false);
    expect(parseShotInput({ reference_panel_ids: Array.from({ length: 9 }, (_, i) => `p${i}`) }).ok).toBe(false);
    expect(parseShotInput({ prompt: "x".repeat(2001) }).ok).toBe(false);
  });
});

describe("parseRenderRequest", () => {
  it("defaults to a draft bake-off on both engines", () => {
    expect(parseRenderRequest({})).toEqual({ ok: true, value: { engines: ["ltx", "h3"], tier: "draft", seed: null } });
  });
  it("accepts one engine, a final tier and an explicit seed, de-duplicating engines", () => {
    expect(parseRenderRequest({ engines: ["h3", "h3"], tier: "final", seed: 5 })).toEqual({
      ok: true,
      value: { engines: ["h3"], tier: "final", seed: 5 },
    });
  });
  it("refuses unknown engines, an empty list and an unknown tier", () => {
    expect(parseRenderRequest({ engines: ["wan"] }).ok).toBe(false);
    expect(parseRenderRequest({ engines: [] }).ok).toBe(false);
    expect(parseRenderRequest({ tier: "max" }).ok).toBe(false);
  });
});

describe("parseImageUpload", () => {
  const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0]);
  const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 0]);
  const webp = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50]);

  it("sniffs PNG, JPEG and WebP regardless of the declared type", () => {
    expect(parseImageUpload(png)).toEqual({ ok: true, value: { mime: "image/png", bytes: png.length } });
    expect(parseImageUpload(jpeg)).toMatchObject({ ok: true, value: { mime: "image/jpeg" } });
    expect(parseImageUpload(webp)).toMatchObject({ ok: true, value: { mime: "image/webp" } });
  });
  it("refuses other content, empty bodies and oversized images", () => {
    expect(parseImageUpload(new TextEncoder().encode("<svg/>")).ok).toBe(false);
    expect(parseImageUpload(new Uint8Array()).ok).toBe(false);
    const big = new Uint8Array(10 * 1024 * 1024 + 1);
    big.set(png);
    expect(parseImageUpload(big).ok).toBe(false);
  });
});
