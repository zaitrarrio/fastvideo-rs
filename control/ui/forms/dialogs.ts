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

// ---- cancelling jobs (docs/control/serverless.md "Cancel and purge", docs/control/README.md "Jobs")
/** Cancel one job of a serverless endpoint: a Runpod job id (pasted) or an invoke's, and optionally the fv-serve job it created. */
export async function askSlsCancel(api: Api, endpoint: string, job = ""): Promise<{ job: string; fv_job?: string; fv_api?: string; stop_fv_job?: boolean } | null> {
  const schemas = await loadSchemas(api);
  return formDialog({
    title: `Cancel a job of ${endpoint}`,
    intro: "Runpod drops a queued job; a running one is stopped by its worker (job-stop). A job that already reached a worker also gets fv-serve's cancel as a queue job (kind http) when fv-control knows the fv-serve job.",
    schemaName: "serverless-cancel",
    remote: schemaRemote(api, "serverless-cancel"),
    schema: schemas["serverless-cancel"]!,
    api,
    value: { job, stop_fv_job: true },
    fields: [
      [["job"], { label: "Runpod job id", help: "Any job of this endpoint, also one fv-control did not submit (or an invoke's number)." }],
      [["fv_job"], { label: "fv-serve job id", help: "Only for a job submitted elsewhere: fv-control finds it in the output of its own invokes." }],
      [["fv_api"], { label: "Its API" }],
      [["stop_fv_job"], { label: "Also cancel the fv-serve job" }],
    ],
    submitLabel: "Cancel job",
  });
}
/** Purge a serverless endpoint's queue: shows the queued count; the endpoint's name typed to confirm. */
export async function askPurge(api: Api, endpoint: string, queued: number, running: number): Promise<{ confirm: string; expected?: number } | null> {
  const schemas = await loadSchemas(api);
  return formDialog({
    title: `Purge the queue of ${endpoint}`,
    intro: `${queued} job${queued === 1 ? "" : "s"} queued, ${running} running. A purge drops every queued job; running ones are not touched (cancel them one by one). Type the endpoint's name to confirm.`,
    schemaName: "serverless-purge",
    remote: schemaRemote(api, "serverless-purge"),
    schema: schemas["serverless-purge"]!,
    api,
    value: { confirm: "", expected: queued },
    fields: [[["confirm"], { label: `Type ${endpoint}` }]],
    submitLabel: `Purge ${queued} queued`,
  });
}
/** Cancel one fv-serve job of a cluster or standalone pod by any of its ids. */
export async function askJobCancel(api: Api, cluster?: string): Promise<{ job: string; cluster?: string } | null> {
  const [schemas, dyn] = await Promise.all([loadSchemas(api), loadDyn(api)]);
  return formDialog({
    title: "Cancel a job by id",
    intro: "Any id an API gave out (fvjob_…, a fal request id, a MiniMax task id, video_gen_…) or fv-serve's internal uuid. fv-control finds the job in the jobs D1 and sends the cancel to the worker that holds it.",
    schemaName: "job-cancel",
    remote: schemaRemote(api, "job-cancel"),
    schema: schemas["job-cancel"]!,
    dyn,
    api,
    value: cluster ? { job: "", cluster } : { job: "" },
    fields: [
      [["job"], { label: "Job id" }],
      [["cluster"], { label: "Cluster or pod", help: "Only when the id matches jobs of two clusters." }],
    ],
    submitLabel: "Cancel job",
  });
}
/** Cancel every queued job of a cluster or standalone pod (one pool, optionally). */
export async function askCancelQueued(api: Api, name: string, queued: number, pools: string[]): Promise<{ pool?: string; max?: number } | null> {
  const schemas = await loadSchemas(api);
  return formDialog({
    title: `Cancel the queued jobs of ${name}`,
    intro: `${queued} job${queued === 1 ? "" : "s"} queued. Each is cancelled at its worker (or, queued at the edge, through a front). Running jobs are not touched.`,
    schemaName: "jobs-cancel-queued",
    remote: schemaRemote(api, "jobs-cancel-queued"),
    schema: schemas["jobs-cancel-queued"]!,
    api,
    value: { max: Math.min(Math.max(queued, 1), 200) },
    fields: [
      [["pool"], { label: "Pool", options: pools.map((p) => ({ value: p })), help: "Empty: every pool." }],
      [["max"], { label: "At most" }],
    ],
    submitLabel: `Cancel ${queued} queued`,
  });
}
