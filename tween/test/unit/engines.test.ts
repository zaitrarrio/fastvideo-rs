import { describe, expect, it } from "vitest";
import { engineConfigs, modelFor } from "../../src/engines";

describe("engineConfigs", () => {
  it("defaults to the fake engine's model ids", () => {
    const [ltx, h3] = engineConfigs({});
    expect(ltx).toMatchObject({ id: "ltx", draft_model: "fake-ltx-turbo", final_model: "fake-ltx-pro", reference_model: null });
    expect(h3).toMatchObject({ id: "h3", draft_model: "fake-h3-turbo", final_model: "fake-h3-max", reference_model: "fake-h3-max" });
  });

  it("takes model ids from vars and treats an empty reference id as none", () => {
    const [ltx, h3] = engineConfigs({
      TWEEN_LTX_DRAFT_MODEL: "ltx25-distill-sol",
      TWEEN_LTX_FINAL_MODEL: "ltx25-distill-dense",
      TWEEN_H3_DRAFT_MODEL: "fasth3",
      TWEEN_H3_FINAL_MODEL: "sol-h3",
      TWEEN_H3_REFERENCE_MODEL: "",
    });
    expect(ltx!.draft_model).toBe("ltx25-distill-sol");
    expect(ltx!.final_model).toBe("ltx25-distill-dense");
    expect(h3!.final_model).toBe("sol-h3");
    expect(h3!.reference_model).toBeNull();
  });

  it("publishes advisory clip-length ranges", () => {
    const [ltx, h3] = engineConfigs({});
    expect(h3!.seconds).toEqual({ min: 5, max: 15 });
    expect(ltx!.seconds.min).toBeLessThan(ltx!.seconds.max);
  });
});

describe("modelFor", () => {
  const [ltx, h3] = engineConfigs({});

  it("picks the tier's model for keyframe shots", () => {
    expect(modelFor(ltx!, "draft", "keyframes")).toEqual({ ok: true, model: "fake-ltx-turbo" });
    expect(modelFor(h3!, "final", "keyframes")).toEqual({ ok: true, model: "fake-h3-max" });
  });

  it("uses the reference model for reference shots on H3", () => {
    expect(modelFor(h3!, "draft", "reference")).toEqual({ ok: true, model: "fake-h3-max" });
  });

  it("refuses reference shots on LTX and on H3 without a reference model", () => {
    const r = modelFor(ltx!, "draft", "reference");
    expect(r.ok).toBe(false);
    const [, bare] = engineConfigs({ TWEEN_H3_REFERENCE_MODEL: "" });
    expect(modelFor(bare!, "draft", "reference").ok).toBe(false);
  });
});
