// Which fv-serve API owns a job id, and the route that cancels it there
// (crates/fastvideo-serve/src/router.rs). A job's id is its API's own id:
// native ids resolve only at /fv/v1/jobs/{id}, a fal request id only under
// its app, and so on (the comment on GET /fv/v1/jobs in native.rs); the
// jobs D1 row says which API (`protocol`). A cancel stops a queued job at
// once and a running one at its next denoise step; on a finished job the
// native, OpenAI and MiniMax routes DELETE the job instead, so fv-control
// never sends them to a job it knows has finished.

/** fv-serve's ProtocolId names (fastvideo-protocol request.rs). */
export type FvApi = "native" | "openai_videos" | "fastwan" | "minimax_v2" | "fal" | "fal_director" | "ltx_v1" | "ltx_v2" | "reactor";

/** The public route that cancels job `id` of `api` (`model`: a fal job's endpoint id, e.g. minimax/h3-max/text-to-video), or null when the API has none. */
export function cancelRoute(api: string, id: string, model?: string | null): { method: "DELETE" | "PUT"; path: string } | null {
  const e = encodeURIComponent(id);
  switch (api) {
    case "native":
      return { method: "DELETE", path: `/fv/v1/jobs/${e}` };
    case "openai_videos":
      return { method: "DELETE", path: `/v1/videos/${e}` };
    case "fastwan":
      return { method: "DELETE", path: `/video/${e}` };
    case "minimax_v2":
      return { method: "DELETE", path: `/v2/video_generation/${e}` };
    case "fal":
    case "fal_director": {
      // PUT /{app}[/{sub}]/requests/{id}/cancel: the full endpoint id is a registered prefix.
      const app = String(model || "").replace(/^\/+|\/+$/g, "");
      if (!app || !/^[A-Za-z0-9._~/-]{1,200}$/.test(app)) return null;
      return { method: "PUT", path: `/${app}/requests/${e}/cancel` };
    }
    default:
      return null; // ltx_v1 / ltx_v2 (no cancel route), reactor (sessions: stop_session)
  }
}
/** Why an API has no cancel route (for the error text). */
export function noCancelReason(api: string): string {
  if (api === "ltx_v1" || api === "ltx_v2") return "the LTX API has no cancel route";
  if (api === "reactor") return "a Reactor session is stopped with its own stop_session, not cancelled";
  if (api === "fal" || api === "fal_director") return "a fal job's cancel route needs its app (the job's model)";
  return `unknown API ${api}`;
}

/** The API a submit path belongs to (a serverless `kind: http` job's `path`). */
export function apiOfSubmit(path: string): FvApi | null {
  const p = String(path || "").split("?")[0]!.replace(/\/+$/, "");
  if (p === "/fv/v1/jobs") return "native";
  if (p === "/v1/videos" || p === "/v1/videos/generations") return "openai_videos";
  if (p === "/generate") return "fastwan";
  if (p === "/v2/video_generation") return "minimax_v2";
  return null;
}
/** The `cancel_path` a serverless http job gets by default, so a Runpod cancel (the worker's job-stop) also stops the fv-serve job it created and waits on (crates/fastvideo-deploy/src/dispatch.rs). */
export function defaultCancelPath(path: string): string | null {
  const api = apiOfSubmit(path);
  const r = api ? cancelRoute(api, "{id}") : null;
  return r ? r.path.replace(encodeURIComponent("{id}"), "{id}") : null;
}

const TERMINAL_WORDS = new Set(["succeeded", "failed", "cancelled", "canceled", "completed", "success", "fail", "error", "expired"]);
/** A status word an fv-serve reply carries (native / OpenAI `status`, MiniMax `status`, fal `status`), lower-cased. */
export function statusWord(body: any): string {
  const s = body?.status ?? body?.data?.status;
  return typeof s === "string" ? s.toLowerCase() : "";
}
export const isTerminalWord = (w: string) => TERMINAL_WORDS.has(w.toLowerCase());

/**
 * The fv-serve job a serverless `kind: http` job created, from its input and
 * Runpod's output: the submit reply (`body` without wait, `submit` with
 * wait), or the poll path of the progress output while it waits. `done`:
 * the output already shows the fv-serve job finished.
 */
export function fvJobOf(input: any, output: any): { id: string; api: FvApi; done: boolean } | null {
  const path = typeof input?.path === "string" ? input.path : "";
  const method = String(input?.method || "POST").toUpperCase();
  if (input?.kind && input.kind !== "http") return null;
  if (method !== "POST") return null;
  const api = apiOfSubmit(path);
  if (!api) return null;
  const idOf = (v: any): string | null => {
    for (const k of ["id", "task_id", "request_id", "job_id", "prompt_id"]) {
      const x = v?.[k];
      if (typeof x === "string" && x) return x;
      if (typeof x === "number") return String(x);
    }
    return null;
  };
  const o = typeof output === "string" ? safeJson(output) : output;
  // With wait: {status, body (the last poll), submit (the submit reply), poll_path}; without: {status, body}.
  const submit = o?.submit ?? o?.body;
  let id = idOf(submit);
  if (!id && typeof o?.poll_path === "string") {
    // Progress while waiting: {state, poll_path: "/fv/v1/jobs/<id>" | "/v2/query/video_generation?task_id=<id>"}.
    const m = /[?&]task_id=([^&]+)|\/([^/?]+)$/.exec(o.poll_path);
    id = m ? decodeURIComponent(m[1] || m[2] || "") || null : null;
  }
  if (!id) return null;
  const word = o?.submit ? statusWord(o.body) : o?.state ? String(o.state).toLowerCase() : statusWord(o?.body);
  return { id, api, done: !!word && isTerminalWord(word) };
}
function safeJson(s: string): any {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}
