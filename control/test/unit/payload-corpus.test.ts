// The payload corpus (plan C3c, docs/adr/0018-fv-deploy.md §7.2): fv-control's Runpod request bodies over
// payload-corpus.ts's inputs equal fixtures/payload-corpus.json, a snapshot first written from main at 2258aec2, the
// builders before fv-control built its launches as fv-deploy `Launch` values. The comparison is independent of the
// Launch code: the snapshot came from the old hand-written builders.
//
// `cases`: every case's canonical body (sorted keys) as the first 16 hex digits of its SHA-256; `shown`: one case of
// each kind in full, long strings by length and hash, for review. A builder that throws is a case too.
//
// Regenerate (only for an intended payload change; say why in the PR):
//   cd control && FV_WRITE_PAYLOAD_CORPUS=1 npx vitest run test/unit/payload-corpus.test.ts
// To compare against another revision's builders, check out its control/src/{cluster,serverless}/payloads.ts (and
// what they import) and run the same command, then diff the fixture.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { corpus, digest, SHOWN, shown } from "./payload-corpus";

const FILE = join(__dirname, "fixtures/payload-corpus.json");

describe("fv-control's Runpod bodies equal the snapshot of main's builders (fixtures/payload-corpus.json)", () => {
  const bodies = corpus();
  if (process.env.FV_WRITE_PAYLOAD_CORPUS === "1") {
    it("writes the snapshot", () => {
      const cases = Object.fromEntries(Object.keys(bodies).sort().map((k) => [k, digest(bodies[k])]));
      const show = Object.fromEntries(SHOWN.map((k) => [k, shown(bodies[k])]));
      writeFileSync(FILE, JSON.stringify({ note: "control/test/unit/payload-corpus.test.ts: regenerate only for an intended payload change", shown: show, cases }, null, 1) + "\n");
    });
    return;
  }
  const snap = JSON.parse(readFileSync(FILE, "utf8")) as { shown: Record<string, unknown>; cases: Record<string, string> };
  it("covers what it says: every preset and template, each weights source, control plane and serverless mode, stored specs", () => {
    const keys = Object.keys(snap.cases);
    expect(keys.length).toBeGreaterThan(900);
    for (const p of ["pod/tpl-", "pod/sa-", "/edge/", "/edge-nobucket/", "/direct/", "/backstop/", "-r2/", "-r2-dcs/", "-hub/", "-none/", "pod/stored-legacy/", "sls/stored-variant/", "sls/stored-nocompute-", "-lb/v2", "custom-cpu"])
      expect(keys.some((k) => k.includes(p)), p).toBe(true);
    expect(Object.keys(bodies).sort()).toEqual(keys.sort());
  });
  it("the shown cases are equal in full", () => {
    for (const k of SHOWN) expect(shown(bodies[k]), k).toStrictEqual(snap.shown[k]);
  });
  it("every case is equal", () => {
    const differ = Object.keys(snap.cases).filter((k) => digest(bodies[k]) !== snap.cases[k]);
    expect(differ, `${differ.length} bodies differ from the snapshot, e.g. ${differ[0]}: ${String(JSON.stringify(differ.length ? shown(bodies[differ[0]!]) : null)).slice(0, 2000)}`).toEqual([]);
  });
});
