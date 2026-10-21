// Engine catalog: which fv-serve model ids Tween uses for each engine and
// tier (ADR-0004 decision 3). Ids come from Worker vars; the defaults are
// the fake engine's (configs/serve/fake.toml), so local development needs
// no GPU.
import type { EngineId, ShotMode, Tier } from "./types";

export interface EngineVars {
  TWEEN_LTX_DRAFT_MODEL?: string;
  TWEEN_LTX_FINAL_MODEL?: string;
  TWEEN_H3_DRAFT_MODEL?: string;
  TWEEN_H3_FINAL_MODEL?: string;
  TWEEN_H3_REFERENCE_MODEL?: string;
}

export interface EngineConfig {
  id: EngineId;
  label: string;
  family: "ltx2" | "h3";
  draft_model: string;
  final_model: string;
  /** Model for `reference` shots; null when the engine cannot run them. */
  reference_model: string | null;
  /**
   * Advisory clip length in whole seconds, for the UI. The engine is the
   * authority: over-length LTX clips run shorter, H3 refuses outside 5..15.
   */
  seconds: { min: number; max: number };
  native_audio: boolean;
}

const pick = (v: string | undefined, fallback: string) => (v === undefined ? fallback : v.trim() || fallback);
const optional = (v: string | undefined, fallback: string) => {
  if (v === undefined) return fallback;
  const t = v.trim();
  return t ? t : null;
};

export function engineConfigs(vars: EngineVars): EngineConfig[] {
  return [
    {
      id: "ltx",
      label: "LTX-2.5",
      family: "ltx2",
      draft_model: pick(vars.TWEEN_LTX_DRAFT_MODEL, "fake-ltx-turbo"),
      final_model: pick(vars.TWEEN_LTX_FINAL_MODEL, "fake-ltx-pro"),
      // `task` is H3-only on /v1/videos, so LTX cannot take a two-panel
      // reference shot (ADR-0004 decision 2).
      reference_model: null,
      seconds: { min: 6, max: 20 },
      native_audio: true,
    },
    {
      id: "h3",
      label: "MiniMax H3",
      family: "h3",
      draft_model: pick(vars.TWEEN_H3_DRAFT_MODEL, "fake-h3-turbo"),
      final_model: pick(vars.TWEEN_H3_FINAL_MODEL, "fake-h3-max"),
      reference_model: optional(vars.TWEEN_H3_REFERENCE_MODEL, "fake-h3-max"),
      seconds: { min: 5, max: 15 },
      native_audio: true,
    },
  ];
}

export function engineById(configs: readonly EngineConfig[], id: string): EngineConfig | null {
  return configs.find((e) => e.id === id) ?? null;
}

export type ModelChoice = { ok: true; model: string } | { ok: false; reason: string };

export function modelFor(engine: EngineConfig, tier: Tier, mode: ShotMode): ModelChoice {
  if (mode === "reference") {
    if (engine.family !== "h3") {
      return { ok: false, reason: `${engine.label} cannot run reference shots over /v1/videos; use H3` };
    }
    if (!engine.reference_model) {
      return { ok: false, reason: `no reference model is configured for ${engine.label}` };
    }
    return { ok: true, model: engine.reference_model };
  }
  return { ok: true, model: tier === "final" ? engine.final_model : engine.draft_model };
}
