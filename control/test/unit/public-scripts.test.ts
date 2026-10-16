// The UI's classic scripts share one global scope and run in index.html order:
// a page function referenced at load (app.js's route table) must already exist.
// Loads every <script> of public/index.html in order in one context, with the DOM stubbed.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { runInContext, createContext } from "node:vm";
import { describe, expect, it } from "vitest";

const pub = join(__dirname, "../../public");
const stub: any = new Proxy(function () {}, {
  get: (_t, k) => (k === Symbol.toPrimitive ? () => "" : stub),
  apply: () => stub,
  construct: () => stub,
});

describe("public/index.html scripts", () => {
  it("load in order without a load-time error", () => {
    const html = readFileSync(join(pub, "index.html"), "utf8");
    const scripts = [...html.matchAll(/<script src="\/([^"]+)"/g)].map((m) => m[1]!);
    expect(scripts.length).toBeGreaterThan(1);
    const ctx: any = {
      console,
      document: stub,
      localStorage: stub,
      navigator: stub,
      history: stub,
      location: { hash: "" },
      addEventListener: () => {},
      setTimeout,
      queueMicrotask,
      performance,
      URLSearchParams,
      Node: function () {},
      fetch: () => new Promise(() => {}),
      matchMedia: () => ({ matches: false, addEventListener() {} }),
    };
    ctx.window = ctx;
    createContext(ctx);
    for (const f of scripts) expect(() => runInContext(readFileSync(join(pub, f), "utf8"), ctx, { filename: f }), f).not.toThrow();
  });
});
