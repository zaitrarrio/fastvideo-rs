// Helpers over the generated family-manifest tables (families-gen.ts, from
// families/*.toml by gen-configs.mjs): what a serve config's `recipe` or model
// name resolves to, as the engine resolves it, and the variants' GPU needs.
// Imports only the generated leaf, so every module may import this one.
import { ENGINE_RECIPE_ALIASES, MANIFEST_FAMILIES, MANIFEST_VARIANTS, MODEL_NAMES, type ManifestVariant } from "./families-gen";

/** The model families the manifests serve (h3, ltx2, wan): the `family` of a pool model. */
export const MODEL_FAMILY_IDS: string[] = MANIFEST_FAMILIES.map((f) => f.id);

export type Variant = ManifestVariant & { id: string };
const byId = (id: string | undefined): Variant | undefined => {
  const v = id ? MANIFEST_VARIANTS[id] : undefined;
  return v && id ? { id, ...v } : undefined;
};
/** The variant a name resolves to (caps.rs `find` / `parse_tier_alias`): a tier alias (`h3-turbo`), a variant id or
 * a served name; undefined for a name the manifests do not know (a `recipe_name` is not a config name). */
export function variantOf(name: string): Variant | undefined {
  return byId(MODEL_NAMES[name]?.variant);
}
/** The variant a serve config's `recipe` resolves to: variantOf, else an engine recipe alias (ENGINE_RECIPE_ALIASES,
 * the caps.rs `model_from_config` arms a config here uses). The engine's contract names (`4step-vsa`, `8step`,
 * `sol-h3`, `trim-8step`, an empty recipe) are neither: plan A3. */
export function recipeVariant(recipe: string): Variant | undefined {
  return variantOf(recipe) ?? byId(ENGINE_RECIPE_ALIASES[recipe]);
}
/** recipeVariant, or an error naming the sources. */
export function mustVariant(name: string): Variant {
  const v = recipeVariant(name);
  if (!v) throw new Error(`${name} is not a name the family manifests know (families/*.toml: a tier alias, variant id, served name) nor an engine recipe alias`);
  return v;
}
/** The GPU memory (GiB) the largest of these variants needs (families/*.toml gpu.min_vram_gib). */
export function minVramGib(variants: readonly Variant[]): number {
  if (!variants.length) throw new Error("minVramGib: no variants");
  return Math.max(...variants.map((v) => v.gpu.min_vram_gib));
}
