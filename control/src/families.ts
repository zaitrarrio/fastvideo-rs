// Helpers over the generated family-manifest tables (families-gen.ts, from
// families/*.toml by gen-configs.mjs): what a serve config's `recipe` or model
// name resolves to, as the engine resolves it, and the variants' GPU needs.
// Imports only the generated leaf, so every module may import this one.
import { MANIFEST_FAMILIES, MANIFEST_VARIANTS, MODEL_NAMES, type ManifestVariant } from "./families-gen";

/** The model families the manifests serve (h3, ltx2, wan): the `family` of a pool model. */
export const MODEL_FAMILY_IDS: string[] = MANIFEST_FAMILIES.map((f) => f.id);

export type Variant = ManifestVariant & { id: string };
/** The variant a name resolves to (caps.rs find / parse_tier_alias): a tier alias (`h3-turbo`), a variant id, a
 * served name or a recipe name; undefined for a name the manifests do not know. */
export function variantOf(name: string): Variant | undefined {
  const n = MODEL_NAMES[name];
  if (!n) return undefined;
  const v = MANIFEST_VARIANTS[n.variant];
  return v ? { id: n.variant, ...v } : undefined;
}
/** variantOf, or an error naming the sources. */
export function mustVariant(name: string): Variant {
  const v = variantOf(name);
  if (!v) throw new Error(`${name} is not a name the family manifests know (families/*.toml: a tier alias, variant id, served name or recipe name)`);
  return v;
}
/** The GPU memory (GiB) the largest of these variants needs (families/*.toml gpu.min_vram_gib). */
export function minVramGib(variants: readonly Variant[]): number {
  if (!variants.length) throw new Error("minVramGib: no variants");
  return Math.max(...variants.map((v) => v.gpu.min_vram_gib));
}
