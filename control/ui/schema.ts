// A small JSON Schema (2020-12 subset) toolkit for the editors: what zod's
// toJSONSchema emits for fv-control's documents. Validation runs in the
// browser as you type; the server validates again with zod (refinements
// such as "exactly one of channel, sha, ref" come back from there).
export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
export type Schema = { [k: string]: any };
export type Path = (string | number)[];
export interface Issue {
  path: Path;
  message: string;
}

export function resolve(s: Schema | undefined, root: Schema): Schema | undefined {
  let cur = s;
  for (let i = 0; cur && cur.$ref && i < 10; i++) {
    const ref: string = cur.$ref;
    if (!ref.startsWith("#/")) return cur;
    cur = ref
      .slice(2)
      .split("/")
      .reduce<any>((o, k) => (o ? o[k.replace(/~1/g, "/").replace(/~0/g, "~")] : undefined), root);
  }
  return cur;
}

const typeOf = (v: unknown): string => (v === null ? "null" : Array.isArray(v) ? "array" : typeof v === "number" ? (Number.isInteger(v) ? "integer" : "number") : typeof v);
function typeMatches(t: string | string[], v: unknown): boolean {
  const ts = Array.isArray(t) ? t : [t];
  const vt = typeOf(v);
  return ts.some((x) => x === vt || (x === "number" && vt === "integer"));
}

export function validate(schema: Schema | undefined, v: unknown, root: Schema = schema || {}, path: Path = []): Issue[] {
  const s = resolve(schema, root);
  if (!s || s === (true as any)) return [];
  const out: Issue[] = [];
  const push = (message: string, p: Path = path) => out.push({ path: p, message });
  for (const k of ["anyOf", "oneOf"] as const) {
    if (Array.isArray(s[k])) {
      const branches = s[k].map((b: Schema) => validate(b, v, root, path));
      const ok = branches.filter((b: Issue[]) => b.length === 0).length;
      if (k === "anyOf" ? ok === 0 : ok !== 1) {
        // Report the branch whose type matches, else a summary.
        const typed = s[k].findIndex((b: Schema) => { const r = resolve(b, root); return r?.type && typeMatches(r.type, v); });
        if (typed >= 0) out.push(...branches[typed]);
        else push(`expected ${s[k].map((b: Schema) => resolve(b, root)?.type || "value").join(" or ")}`);
      }
    }
  }
  if (Array.isArray(s.allOf)) for (const b of s.allOf) out.push(...validate(b, v, root, path));
  if (s.type && !typeMatches(s.type, v)) {
    push(`expected ${Array.isArray(s.type) ? s.type.join(" or ") : s.type}, got ${typeOf(v)}`);
    return out;
  }
  if (s.const !== undefined && JSON.stringify(s.const) !== JSON.stringify(v)) push(`must be ${JSON.stringify(s.const)}`);
  if (Array.isArray(s.enum) && !s.enum.some((e: unknown) => JSON.stringify(e) === JSON.stringify(v))) push(`one of: ${s.enum.map((e: unknown) => JSON.stringify(e)).join(", ")}`);
  if (typeof v === "string") {
    if (s.minLength !== undefined && v.length < s.minLength) push(s.minLength === 1 ? "must not be empty" : `at least ${s.minLength} characters`);
    if (s.maxLength !== undefined && v.length > s.maxLength) push(`at most ${s.maxLength} characters`);
    if (s.pattern && !new RegExp(s.pattern).test(v)) push(`does not match ${s.pattern}`);
  }
  if (typeof v === "number") {
    if (s.minimum !== undefined && v < s.minimum) push(`must be ≥ ${s.minimum}`);
    if (s.maximum !== undefined && v > s.maximum && s.maximum < 9e15) push(`must be ≤ ${s.maximum}`);
    if (s.exclusiveMinimum !== undefined && v <= s.exclusiveMinimum) push(`must be > ${s.exclusiveMinimum}`);
    if (s.exclusiveMaximum !== undefined && v >= s.exclusiveMaximum) push(`must be < ${s.exclusiveMaximum}`);
  }
  if (Array.isArray(v)) {
    if (s.minItems !== undefined && v.length < s.minItems) push(`at least ${s.minItems} item(s)`);
    if (s.maxItems !== undefined && v.length > s.maxItems) push(`at most ${s.maxItems} item(s)`);
    if (s.items) v.forEach((x, i) => out.push(...validate(s.items, x, root, [...path, i])));
  }
  if (v && typeof v === "object" && !Array.isArray(v)) {
    const o = v as Record<string, unknown>;
    for (const r of s.required || []) if (!(r in o)) push(`missing required property "${r}"`);
    for (const [k, x] of Object.entries(o)) {
      const ps = s.properties?.[k];
      if (ps) out.push(...validate(ps, x, root, [...path, k]));
      else {
        if (s.propertyNames) for (const i of validate(s.propertyNames, k, root, [...path, k])) out.push({ ...i, message: `key ${JSON.stringify(k)}: ${i.message}` });
        if (s.additionalProperties === false) push(`unknown property "${k}"`, [...path, k]);
        else if (s.additionalProperties && typeof s.additionalProperties === "object") out.push(...validate(s.additionalProperties, x, root, [...path, k]));
      }
    }
  }
  return out;
}

/** The subschema that governs `path` (merging nullable anyOf branches; the first object branch otherwise). */
export function schemaAt(schema: Schema, path: Path, root: Schema = schema): Schema | undefined {
  let s: Schema | undefined = resolve(schema, root);
  for (const seg of path) {
    s = unwrap(s, root);
    if (!s) return undefined;
    if (typeof seg === "number") s = resolve(s.items, root);
    else s = resolve(s.properties?.[seg] ?? (typeof s.additionalProperties === "object" ? s.additionalProperties : undefined), root);
  }
  return s;
}
/** anyOf [X, null] -> X (keeps the outer description). */
export function unwrap(s: Schema | undefined, root: Schema): Schema | undefined {
  s = resolve(s, root);
  if (!s) return s;
  const alts = s.anyOf || s.oneOf;
  if (Array.isArray(alts)) {
    const nn = alts.map((b: Schema) => resolve(b, root)).filter((b: Schema | undefined) => b && b.type !== "null");
    if (nn.length === 1) {
      // The outer node's own hints (x-dynamic, x-unit, x-rule …) apply to the non-null branch.
      const { anyOf: _a, oneOf: _o, ...outer } = s;
      return { ...nn[0], ...outer, description: s.description ?? nn[0]!.description, nullable: true };
    }
  }
  return s;
}
export function pathString(p: Path): string {
  return "$" + p.map((x) => (typeof x === "number" ? `[${x}]` : /^[A-Za-z_][A-Za-z0-9_]*$/.test(x) ? `.${x}` : `[${JSON.stringify(x)}]`)).join("");
}
export function getAt(v: any, p: Path): any {
  return p.reduce((o, k) => (o == null ? undefined : o[k as any]), v);
}
export function setAt(v: any, p: Path, x: any): any {
  if (!p.length) return x;
  const [h, ...t] = p;
  const base = Array.isArray(v) ? [...v] : { ...(v || {}) };
  (base as any)[h as any] = setAt((v || {})[h as any], t, x);
  return base;
}
export function defaultFor(s: Schema | undefined, root: Schema): Json {
  const u = unwrap(s, root);
  if (!u) return null;
  if (u.default !== undefined) return u.default;
  if (Array.isArray(u.enum)) return u.enum[0];
  const t = Array.isArray(u.type) ? u.type[0] : u.type;
  if (t === "object") {
    const o: Record<string, Json> = {};
    for (const r of u.required || []) o[r] = defaultFor(u.properties?.[r], root);
    return o;
  }
  return t === "array" ? [] : t === "string" ? "" : t === "number" || t === "integer" ? (u.minimum ?? 0) : t === "boolean" ? false : null;
}
