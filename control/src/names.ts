// Name checks (GET /api/names/:kind?name=…): the pattern, the reserved
// names and uniqueness against what exists, the same rules every create
// path enforces (docs/control/config-validation.md "Names").
//   cluster   clusters and standalone pods (one namespace: D1 clusters.name UNIQUE)
//   endpoint  serverless endpoints (unique among the live ones: serverless_endpoints_live_name)
//   token     API tokens (unique among the active ones: not revoked, not expired)
import type { Env } from "./env";
import { nameProblem, NAME_RULE, TOKEN_NAME_RE, TOKEN_NAME_RULE } from "./enums";
import { HttpError, now } from "./util";

export const NAME_KINDS = ["cluster", "endpoint", "token"] as const;
export type CheckedName = (typeof NAME_KINDS)[number];

export interface NameCheck {
  ok: boolean;
  /** Why not (pattern, reserved, taken), or null. */
  problem: string | null;
  /** The rule, for the form to show before typing. */
  rule: string;
  taken: boolean;
}

export async function nameTaken(env: Env, kind: CheckedName, name: string): Promise<string | null> {
  if (kind === "cluster") {
    const r = await env.DB.prepare("SELECT source FROM clusters WHERE name = ?").bind(name).first<{ source: string }>();
    return r ? `${r.source === "standalone" ? "a standalone pod" : "a cluster"} named ${name} exists` : null;
  }
  if (kind === "endpoint") {
    const r = await env.DB.prepare("SELECT id FROM serverless_endpoints WHERE name = ? AND deleted_at IS NULL").bind(name).first<{ id: string }>();
    return r ? `a live endpoint named ${name} exists (fvc-${name})` : null;
  }
  const r = await env.DB.prepare("SELECT id FROM api_tokens WHERE name = ? AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at > ?)").bind(name, now()).first<{ id: string }>();
  return r ? `an active token named ${name} exists (revoke it first)` : null;
}

export async function checkName(env: Env, kind: string, name: string): Promise<NameCheck> {
  if (!(NAME_KINDS as readonly string[]).includes(kind)) throw new HttpError(404, `name kinds: ${NAME_KINDS.join(", ")}`);
  const k = kind as CheckedName;
  const rule = k === "token" ? TOKEN_NAME_RULE : NAME_RULE;
  const shape = k === "token" ? (TOKEN_NAME_RE.test(name) ? null : TOKEN_NAME_RULE) : nameProblem(k, name);
  if (shape) return { ok: false, problem: shape, rule, taken: false };
  const taken = await nameTaken(env, k, name);
  return { ok: !taken, problem: taken, rule, taken: !!taken };
}

/** Throws 409 when the name is taken (create paths). */
export async function assertNameFree(env: Env, kind: CheckedName, name: string, path: (string | number)[] = ["name"]): Promise<void> {
  const taken = await nameTaken(env, kind, name);
  if (taken) throw new HttpError(409, `${path.join(".")}: ${taken}`, { issues: [{ path, message: taken }] });
}
