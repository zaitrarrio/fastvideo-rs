// Runpod API: REST v2 (api.runpod.io/v2: pods, templates, serverless, billing,
// the catalog), GraphQL (balance, runtime metrics, prices) and the console's
// log endpoint (hapi.runpod.net/v1/pod/<id>/logs: not in the public docs, but
// it answers with the API key; docs/control/README.md "Logs"). REST v1
// (rest.runpod.io/v1) is retired: its host no longer resolves and Runpod's API
// overview says it is "deprecated and no longer available". The API key only
// ever goes in an Authorization header. Nothing here returns a pod's env:
// Runpod's GET /pods/<id> includes it in clear, so only named fields are
// picked.
import { runpodErrorText } from "./runpoderr";
import { defaults, type Env } from "./env";
import { fetchWithTimeout, HttpError, scrub } from "./util";

/** A pod as GraphQL `myself { pods }` reports it (the per-minute collector; REST v2 names these `status`, `cost`,
 * `image`: `runpod.pod`). */
export interface RunpodPod {
  id: string;
  name: string;
  desiredStatus: string;
  costPerHr: number;
  imageName?: string;
  lastStartedAt?: string;
  gpuCount?: number;
  vcpuCount?: number;
  memoryInGb?: number;
  machine?: { gpuDisplayName?: string; dataCenterId?: string } | null;
  runtime?: {
    uptimeInSeconds?: number;
    gpus?: { id: string; gpuUtilPercent?: number; memoryUtilPercent?: number }[];
    container?: { cpuPercent?: number; memoryPercent?: number };
  } | null;
}

/** A pod as REST v2 `GET /pods/<id>` (and the create answer) reports it: the named fields fv-control reads. */
export interface RestPod {
  id: string;
  name: string;
  /** PROVISIONING | STARTING | RUNNING | EXITED | ERROR | TERMINATED */
  status: string;
  /** $/hr of the pod's compute (reported in every status: a stopped pod is not billed for it). */
  cost: number;
  image?: string;
  dataCenterId?: string | null;
}
/** The named fields of a REST v2 pod body (never its env). */
export const restPod = (p: any): RestPod => ({ id: String(p.id), name: String(p.name ?? ""), status: String(p.status ?? ""), cost: Number(p.cost ?? 0), image: typeof p.image === "string" ? p.image : undefined, dataCenterId: p.dataCenterId ?? null });

async function call(env: Env, url: string, init: RequestInit & { timeoutMs?: number }): Promise<any> {
  const r = await fetchWithTimeout(url, {
    ...init,
    headers: { authorization: `Bearer ${env.RUNPOD_API_KEY}`, "content-type": "application/json", ...(init.headers || {}) },
  });
  const text = await r.text();
  let j: any = null;
  try {
    j = text ? JSON.parse(text) : null;
  } catch {
    j = { raw: text.slice(0, 300) };
  }
  if (!r.ok) {
    const msg = typeof j === "object" && j ? runpodErrorText(j, typeof init.body === "string" ? init.body : null) : text.slice(0, 2000);
    throw new HttpError(r.status >= 500 ? 502 : r.status, `runpod: ${scrub(env, String(msg))}`, { upstream: r.status });
  }
  return j;
}

export const runpod = {
  /** One REST v2 call: `path` under `defaults.runpodRest` (`/pods`, `/templates`, `/serverless`, `/billing/…`, `/catalog/…`). */
  rest(env: Env, method: string, path: string, body?: unknown) {
    return call(env, `${defaults.runpodRest(env)}${path}`, { method, body: body === undefined ? undefined : JSON.stringify(body), timeoutMs: 60000 });
  },
  async gql<T = any>(env: Env, query: string, variables?: Record<string, unknown>): Promise<T> {
    const j = await call(env, defaults.runpodGraphql(env), { method: "POST", body: JSON.stringify({ query, variables }) });
    if (j?.errors?.length) throw new HttpError(502, `runpod graphql: ${scrub(env, j.errors.map((e: any) => e.message).join("; "))}`);
    return j.data as T;
  },
  async account(env: Env): Promise<{ balance: number; spendPerHr: number; spendLimit?: number }> {
    const d = await runpod.gql<any>(env, "{ myself { clientBalance currentSpendPerHr spendLimit } }");
    return { balance: Number(d.myself.clientBalance), spendPerHr: Number(d.myself.currentSpendPerHr), spendLimit: d.myself.spendLimit };
  },
  /** Every pod of the account with its runtime metrics (one GraphQL call). */
  async pods(env: Env): Promise<{ balance: number; spendPerHr: number; pods: RunpodPod[] }> {
    const d = await runpod.gql<any>(
      env,
      `{ myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr imageName lastStartedAt gpuCount vcpuCount memoryInGb
          machine { gpuDisplayName dataCenterId }
          runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } container { cpuPercent memoryPercent } } } } }`,
    );
    return { balance: Number(d.myself.clientBalance), spendPerHr: Number(d.myself.currentSpendPerHr), pods: d.myself.pods || [] };
  },
  /** A pod's public fields (REST v2 `GET /pods/<id>`; never its env). null when it is gone. */
  async pod(env: Env, id: string): Promise<RestPod | null> {
    try {
      const p = await runpod.rest(env, "GET", `/pods/${encodeURIComponent(id)}`);
      if (!p || !p.id) return null;
      return restPod(p);
    } catch (e) {
      if (e instanceof HttpError && (e.status === 404 || e.status === 400)) return null;
      throw e;
    }
  },
  /** One pod's state and container uptime (null when it is gone): the boot diagnosis while an `up` waits. */
  async podRuntime(env: Env, id: string): Promise<{ desiredStatus: string; uptimeS: number | null } | null> {
    const d = await runpod.gql<any>(env, "query($id: String!) { pod(input: {podId: $id}) { id desiredStatus runtime { uptimeInSeconds } } }", { id });
    const p = d?.pod;
    if (!p?.id) return null;
    return { desiredStatus: String(p.desiredStatus || ""), uptimeS: typeof p.runtime?.uptimeInSeconds === "number" ? p.runtime.uptimeInSeconds : null };
  },
  /** `POST /pods` (a CreatePodRequest: src/launch.ts runpodPodBody, cluster/payloads.ts, buildpods.ts). Answers the Pod
   * (201), or a QueuedPod (202) when the request queued for capacity, which fv-control never asks for. */
  create(env: Env, payload: unknown) {
    return runpod.rest(env, "POST", "/pods", payload);
  },
  /** `PATCH /pods/<id>` (an UpdatePodRequest: `env`, `image`, …). */
  patch(env: Env, id: string, body: unknown) {
    return runpod.rest(env, "PATCH", `/pods/${encodeURIComponent(id)}`, body);
  },
  async remove(env: Env, id: string): Promise<boolean> {
    try {
      await runpod.rest(env, "DELETE", `/pods/${encodeURIComponent(id)}`);
      return true;
    } catch (e) {
      if (e instanceof HttpError && e.status === 404) return true;
      throw e;
    }
  },
  /** A pod state transition: REST v2 `POST /pods/<id>/action {action}` (v1's `/stop` and `/start` routes). */
  action(env: Env, id: string, action: "start" | "stop" | "restart" | "terminate") {
    return runpod.rest(env, "POST", `/pods/${encodeURIComponent(id)}/action`, { action });
  },
  stop(env: Env, id: string) {
    return runpod.action(env, id, "stop");
  },
  start(env: Env, id: string) {
    return runpod.action(env, id, "start");
  },
  /** The container and system log tail Runpod keeps for a pod. */
  async logs(env: Env, id: string): Promise<{ container: string[]; system: string[] }> {
    const j = await call(env, `${defaults.runpodHapi(env)}/pod/${encodeURIComponent(id)}/logs`, { method: "GET" });
    return { container: Array.isArray(j?.container) ? j.container : [], system: Array.isArray(j?.system) ? j.system : [] };
  },
  /** On-demand secure-cloud $/hr of one GPU of a type (null when unknown). */
  async gpuPrice(env: Env, gpuId: string): Promise<{ price: number | null; stock?: string }> {
    const d = await runpod.gql<any>(env, "query($id: String) { gpuTypes(input: {id: $id}) { id securePrice lowestPrice(input: {gpuCount: 1}) { stockStatus } } }", { id: gpuId });
    const g = d?.gpuTypes?.[0];
    return { price: g?.securePrice ?? null, stock: g?.lowestPrice?.stockStatus };
  },
};

/** CPU pod $/hr per vCPU (estimates; the create answer's `cost` is checked against the cap). */
export const CPU_DPH_PER_VCPU: Record<string, number> = { cpu3c: 0.03, cpu3g: 0.04, cpu3m: 0.05, cpu5c: 0.035, cpu5g: 0.045, cpu5m: 0.06 };
export const cpuPrice = (flavor: string, vcpu: number) => (CPU_DPH_PER_VCPU[flavor] ?? 0.06) * vcpu;
