// Runpod's refusals made readable. REST v2 answers with an ErrorResponse
// {"title": "Bad Request", "status": 400, "detail": "…", "errors": ["$:
// additional properties 'bogus' not allowed", …]}; REST v1 answered a schema
// violation with {"error": …, "problems": ["At /endpoints/properties/
// gpuTypeIds/items/enum: value must be one of 'AMD Instinct MI300X OAM', …"]}:
// dozens of allowed values before anything useful, which the 300- and
// 500-character cuts used to lose. GraphQL, the queue API and hapi still
// answer {"error": …} / {"message": …}. This puts the field, the value we sent
// and the closest allowed values first, then the rest of the list.
import { closeMatches } from "./enums";

const ENUM_RE = /^At\s+(\S+?):\s*value must be one of\s+(.*)$/s;
/** One `problems` / `errors` entry → "gpuTypeIds: sent "RTX 6000 PRO" (did you mean …); allowed: …" (an entry in
 * another form is returned as is). */
export function readableProblem(p: string, sent?: unknown): string {
  const m = ENUM_RE.exec(String(p).trim());
  if (!m) return String(p);
  const path = m[1]!.replace(/\/(properties|items|enum)\b/g, "").replace(/^\//, "").split("/").filter(Boolean);
  const field = path[path.length - 1] || m[1]!;
  const allowed = [...m[2]!.matchAll(/'([^']*)'/g)].map((x) => x[1]!);
  const val = sent && typeof sent === "object" ? (sent as Record<string, unknown>)[field] : undefined;
  const values = (Array.isArray(val) ? val : val !== undefined ? [val] : []).map(String);
  const bad = values.filter((v) => !allowed.includes(v));
  const parts = bad.map((v) => {
    const near = closeMatches(v, allowed);
    return `${JSON.stringify(v)}${near.length ? ` (did you mean ${near.map((x) => JSON.stringify(x)).join(" or ")}?)` : ""}`;
  });
  return `${field}: ${bad.length ? `Runpod refused ${parts.join(", ")}` : "a value Runpod does not accept"}; allowed (${allowed.length}): ${allowed.join(", ")}`;
}
/** The message of a Runpod error body: its validation failures (v2 `errors`, v1 `problems`) made readable first,
 * then `detail` / `error` / `message` (v2's `title` when nothing else says more). `sent`: the JSON we posted. */
export function runpodErrorText(body: unknown, sentJson?: string | null): string {
  let sent: unknown;
  try {
    sent = sentJson ? JSON.parse(sentJson) : undefined;
  } catch {
    sent = undefined;
  }
  if (body && typeof body === "object") {
    const b = body as Record<string, any>;
    const list = Array.isArray(b.errors) ? b.errors : Array.isArray(b.problems) ? b.problems : [];
    const problems = list.map((p: unknown) => readableProblem(typeof p === "string" ? p : JSON.stringify(p), sent));
    const head = typeof b.detail === "string" ? b.detail : typeof b.error === "string" ? b.error : typeof b.message === "string" ? b.message : typeof b.title === "string" ? b.title : "";
    if (problems.length) return `${problems.join("; ")}${head && !/^At /.test(head) ? ` (${head})` : ""}`;
    if (head) return head;
    return JSON.stringify(b).slice(0, 2000);
  }
  return String(body ?? "").slice(0, 2000);
}
