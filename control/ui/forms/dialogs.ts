// Typed dialogs for the operations that used to ask with prompt(): extend a
// deadline, scale a pool, roll to a target, mint a user API key. Each is a
// form bound to its schema (extend, scale, roll, mint-key).
import { formDialog, type Api } from "../fields";
import { loadDyn, loadSchemas, schemaRemote } from "./common";

export async function askExtend(api: Api, title: string, minutes = 30): Promise<number | null> {
  const schemas = await loadSchemas(api);
  const r = await formDialog({ title, intro: "Moves the deadline (the backstop that deletes the pods) this much later.", schemaName: "extend", remote: schemaRemote(api, "extend"), schema: schemas.extend!, api, value: { minutes }, fields: [[["minutes"], { label: "Extend by" }]], submitLabel: "Extend" });
  return r ? r.minutes : null;
}
export async function askScale(api: Api, title: string, pool: string, count: number, pools: string[]): Promise<{ pool: string; count: number } | null> {
  const schemas = await loadSchemas(api);
  return formDialog({ title, intro: "Workers are drained before they are deleted.", schemaName: "scale", remote: schemaRemote(api, "scale"), schema: schemas.scale!, api, value: { pool, count }, fields: [[["pool"], { label: "Pool", options: pools.map((p) => ({ value: p })), allowUnset: false }], [["count"], { label: "Workers" }]], submitLabel: "Scale" });
}
export async function askRoll(api: Api, title: string, target: string, pools: string[], cluster?: string): Promise<{ target: string; pools?: string[] } | null> {
  const [schemas, dyn] = await Promise.all([loadSchemas(api), loadDyn(api, cluster)]);
  const opts = [...(dyn.channels || []).map((c: any) => ({ value: c.id, detail: c.sha ? `at ${c.sha}` : "" })), ...(dyn.shas || []).map((s: string) => ({ value: s, detail: "commit" }))];
  return formDialog({
    title,
    intro: "New workers come up beside the old ones, then the old ones drain.",
    schemaName: "roll", remote: schemaRemote(api, "roll"),
    schema: schemas.roll!,
    dyn,
    api,
    value: { target, pools },
    fields: [
      [["target"], { label: "Target", kind: "combo", options: opts, help: "A channel, a commit with images, or an image reference." }],
      [["pools"], { label: "Pools", kind: "chips", options: pools.map((p) => ({ value: p })) }],
    ],
    submitLabel: "Roll",
  });
}
export async function askKeyName(api: Api, title: string): Promise<string | null> {
  const schemas = await loadSchemas(api);
  const r = await formDialog({ title, schemaName: "mint-key", remote: schemaRemote(api, "mint-key"), schema: schemas["mint-key"]!, api, value: { name: "laptop" }, fields: [[["name"], { label: "Key name" }]], submitLabel: "Mint key" });
  return r ? r.name : null;
}
