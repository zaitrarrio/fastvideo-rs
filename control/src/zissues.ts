// zod issues as fv-control reports them: a path and a message. A record key
// that fails its schema (an env name, a data centre) reports the key's own
// message ("set by the controller") at that key, not zod's "Invalid key in record".
import type { z } from "zod";

export interface Issue {
  path: (string | number)[];
  message: string;
}
export function zIssues(e: z.ZodError): Issue[] {
  const out: Issue[] = [];
  const norm = (p: PropertyKey[]) => p.map((x) => (typeof x === "symbol" ? String(x) : x)) as (string | number)[];
  for (const i of e.issues as any[]) {
    if (i.code === "invalid_key" && Array.isArray(i.issues) && i.issues.length) for (const n of i.issues) out.push({ path: norm(i.path), message: n.message });
    else out.push({ path: norm(i.path), message: i.message });
  }
  return out;
}
