// Schema-bound form controls (docs/control/config-validation.md): every
// control is made from the field's JSON Schema (GET /api/schemas, the zod
// schemas of src/schemas.ts), so the UI cannot offer what the server
// refuses. An enum is a select or a segmented control, a live list
// (x-dynamic) a select or ordered chips with stale values flagged, a number
// an <input type=number> with the schema's min / max / step and its unit,
// a name its rule before typing and a uniqueness check as you type. The
// form model checks every field locally on each edit, asks the server's
// validator for the cross-field rules (debounced), lists every problem with
// a link to its field, and keeps Submit disabled until the form is valid.
import { dynOptions, type Dynamic } from "./editor";
import { getAt, resolve, schemaAt, unwrap, validate, type Issue, type Json, type Path, type Schema } from "./schema";
import "./fields.css";

export type Api = (path: string, opts?: { method?: string; body?: unknown }) => Promise<any>;
export type ControlKind = "select" | "segmented" | "toggle" | "number" | "text" | "combo" | "textarea" | "secret" | "chips" | "list" | "table" | "record" | "object";
/** Live lists that are the whole choice (a select); the others only suggest (a text input with a datalist). */
export const CLOSED_SOURCES = new Set(["channels", "pool_presets", "pools", "clusters", "data_centers", "volumes", "releases", "regions", "variants", "gpu_types", "cpu_flavors", "model_ids", "families", "recipes", "fake_models"]);

const typeOf = (u: Schema | undefined): string | undefined => (Array.isArray(u?.type) ? u!.type.find((t: string) => t !== "null") : u?.type);
/** The closed set of values a schema allows (enum, const, or an anyOf of consts), or null. */
export function enumValues(u: Schema | undefined): Json[] | null {
  if (!u) return null;
  if (Array.isArray(u.enum)) return u.enum;
  if (u.const !== undefined) return [u.const];
  return null;
}
/** Which control a schema node gets. An enum is never free text: the coverage test (test/unit/config-validation.test.ts) walks every schema with this. */
export function controlKind(s: Schema | undefined, root: Schema): ControlKind {
  const u = unwrap(s, root) || {};
  const vals = enumValues(u);
  if (vals) return vals.length <= 4 && !u["x-dynamic"] && vals.every((v) => typeof v === "string") ? "segmented" : "select";
  const t = typeOf(u);
  if (t === "boolean") return "toggle";
  if (t === "number" || t === "integer") return "number";
  if (t === "string") {
    if (u["x-secret"]) return "secret";
    if (u["x-dynamic"] && CLOSED_SOURCES.has(u["x-dynamic"])) return "select";
    if (u["x-ui"] === "textarea" || (u.maxLength > 1000 && !u.pattern)) return "textarea";
    if (u["x-dynamic"]) return "combo";
    return "text";
  }
  if (t === "array") {
    const it = unwrap(u.items, root) || {};
    if (enumValues(it) || (it["x-dynamic"] && CLOSED_SOURCES.has(it["x-dynamic"]))) return "chips";
    if (typeOf(it) === "object") return "table";
    return "list";
  }
  if (t === "object") return !u.properties && u.additionalProperties && typeof u.additionalProperties === "object" ? "record" : "object";
  // anyOf of non-null alternatives (e.g. "" or an image): text with the union's check.
  if (Array.isArray(u.anyOf)) return "text";
  return "text";
}
/** The rule a field shows before anything is typed: x-rule, else what the schema says (range, length, unit). */
export function ruleOf(s: Schema | undefined, root: Schema): string {
  const u = unwrap(s, root) || {};
  if (u["x-rule"]) return u["x-rule"];
  const unit = u["x-unit"] ? ` ${u["x-unit"]}` : "";
  const t = typeOf(u);
  if ((t === "number" || t === "integer") && !enumValues(u)) {
    const lo = u.minimum ?? (u.exclusiveMinimum !== undefined ? `>${u.exclusiveMinimum}` : undefined);
    const hi = u.maximum !== undefined && u.maximum < 9e15 ? u.maximum : undefined;
    const kind = t === "integer" ? "whole number" : "number";
    if (lo !== undefined && hi !== undefined) return `${kind}, ${lo}–${hi}${unit}`;
    if (lo !== undefined) return `${kind}, at least ${lo}${unit}`;
    if (hi !== undefined) return `${kind}, at most ${hi}${unit}`;
    return kind + unit;
  }
  if (t === "string" && (u.minLength !== undefined || u.maxLength !== undefined) && !enumValues(u)) return `${u.minLength ? `${u.minLength}–` : "up to "}${u.maxLength ?? "∞"} characters`;
  return "";
}

let uid = 0;
type Kid = Node | string | null | undefined | false;
function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, any> = {}, ...kids: (Kid | Kid[])[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (k === "class") e.className = v;
    else if (k === "value") (e as any).value = v;
    else if (k === "checked") (e as any).checked = !!v;
    else e.setAttribute(k, v === true ? "" : String(v));
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k);
  return e;
}
const key = (p: Path) => p.join(".");
const clone = <T>(v: T): T => (v === undefined ? v : JSON.parse(JSON.stringify(v)));
function setIn(root: any, p: Path, x: any): any {
  if (!p.length) return x;
  const base = root && typeof root === "object" ? root : typeof p[0] === "number" ? [] : {};
  const [h0, ...t] = p;
  const next: any = Array.isArray(base) ? [...base] : { ...base };
  const v = setIn(base[h0 as any], t, x);
  if (v === undefined) {
    if (Array.isArray(next) && typeof h0 === "number") next.splice(h0, 1);
    else delete next[h0 as any];
  } else next[h0 as any] = v;
  return next;
}

export interface Option {
  value: Json;
  label?: string;
  detail?: string;
  /** Not offered now (out of stock, deleted): still selectable when it is the current value, flagged. */
  stale?: boolean;
}
export interface FieldOpts {
  label?: string;
  help?: string | Node;
  kind?: ControlKind;
  wide?: boolean;
  placeholder?: string;
  /** Check the name's uniqueness as you type (GET /api/names/<kind>). */
  nameCheck?: "cluster" | "endpoint" | "token";
  /** Options instead of the schema's / the live list's. */
  options?: Option[];
  /** Number shown = stored / scale (e.g. seconds shown as minutes: 60). */
  scale?: number;
  unit?: string;
  /** Label of the "unset" choice of an optional select / number (e.g. "policy default"). */
  unsetLabel?: string;
  /** Allow choosing "unset" (default: when the field is optional). */
  allowUnset?: boolean;
  /** For record (env) fields: secrets allowed, and the value shape. */
  env?: EnvOpts;
  /** Called after the value changes (redraw dependent fields). */
  onSet?: (v: any) => void;
  step?: number | "any";
}
export interface FormOptions {
  /** The schema's name (GET /api/schemas/<name>), recorded on every control (data-schema) for the coverage check. */
  schemaName: string;
  schema: Schema;
  dyn?: Dynamic;
  api: Api;
  value: any;
  /** The server's validator for the cross-field rules: issues by path (debounced, after each change). */
  remote?: (v: any) => Promise<{ issues: Issue[]; warnings?: Issue[] }>;
  onChange?: (v: any) => void;
  /** Disabled until the form is valid. */
  submit?: HTMLButtonElement;
}
interface Field {
  path: Path;
  box: HTMLElement;
  err: HTMLElement;
  opts: FieldOpts;
  control: HTMLElement;
  local: string[];
  async: string[];
  remote: string[];
  warn: string[];
  sync: () => void;
}
export interface Form {
  el: HTMLElement;
  summary: HTMLElement;
  field(path: Path, opts?: FieldOpts): HTMLElement;
  value(): any;
  get(path: Path): any;
  set(path: Path, v: any, silent?: boolean): void;
  ok(): boolean;
  /** Every current problem (local, uniqueness, server), path-anchored. */
  problems(): Issue[];
  /** Re-run every check (e.g. after fields were redrawn). */
  check(): void;
  /** Forget fields whose element left the DOM (after a redraw). */
  prune(): void;
  ready(): Promise<boolean>;
}

/** A form bound to a schema. Fields are added with form.field(path); the caller lays them out. */
export function createForm(o: FormOptions): Form {
  const root = o.schema;
  let value = clone(o.value) ?? {};
  const fields = new Map<string, Field>();
  let remoteIssues: Issue[] = [];
  let remoteWarnings: Issue[] = [];
  let remotePending = !!o.remote;
  let remoteSeq = 0;
  let remoteTimer: any;
  const summary = el("div", { class: "ff-summary", role: "status", "aria-live": "polite" });
  const formEl = el("div", { class: "ff", "data-schema-form": o.schemaName });

  const requiredAt = (p: Path): boolean => {
    if (!p.length) return false;
    const parent = unwrap(schemaAt(root, p.slice(0, -1)), root);
    const last = p[p.length - 1];
    return typeof last === "string" && Array.isArray(parent?.required) && parent!.required.includes(last);
  };
  function localCheck(f: Field) {
    const s = schemaAt(root, f.path);
    const v = getAt(value, f.path);
    f.local = [];
    if (v === undefined || v === "") {
      if (requiredAt(f.path) && v === undefined) f.local.push("required");
      else if (v === "" && unwrap(s, root)?.minLength) f.local.push("required");
      return;
    }
    for (const i of validate(s, v, root, [])) f.local.push(i.path.length ? `${i.path.join(".")}: ${i.message}` : i.message);
  }
  function paint() {
    // Server issues go to the field with the longest matching path; the rest to the summary only.
    for (const f of fields.values()) (f.remote = []), (f.warn = []);
    const place = (i: Issue, warn: boolean) => {
      for (let n = i.path.length; n >= 0; n--) {
        const f = fields.get(key(i.path.slice(0, n)));
        if (f && f.box.isConnected) {
          const msg = (n < i.path.length ? `${i.path.slice(n).join(".")}: ` : "") + i.message;
          (warn ? f.warn : f.remote).push(msg);
          return true;
        }
      }
      return false;
    };
    const unplaced: Issue[] = [];
    for (const i of remoteIssues) if (!place(i, false)) unplaced.push(i);
    for (const i of remoteWarnings) place(i, true);
    for (const f of fields.values()) {
      const errs = [...new Set([...f.local, ...f.async, ...f.remote])];
      f.err.textContent = errs.join("; ");
      f.err.hidden = !errs.length;
      const w = f.box.querySelector<HTMLElement>(":scope > .ff-warn");
      if (w) {
        w.textContent = f.warn.join("; ");
        w.hidden = !f.warn.length;
      }
      f.box.classList.toggle("has-err", errs.length > 0);
      for (const c of f.box.querySelectorAll<HTMLElement>("[data-ctl]")) c.setAttribute("aria-invalid", String(errs.length > 0));
    }
    const probs = problems();
    summary.replaceChildren();
    if (probs.length) {
      summary.append(
        el("p", { class: "small" }, el("span", { class: "badge critical" }, `${probs.length} problem${probs.length === 1 ? "" : "s"}`), " fix these first:"),
        el(
          "ul",
          { class: "ff-issues" },
          ...probs.map((i) =>
            el(
              "li",
              {},
              el("a", { href: "#", "data-path": key(i.path), onclick: (ev: Event) => (ev.preventDefault(), focus(i.path)) }, i.path.length ? key(i.path) : "form"),
              " ",
              i.message,
            ),
          ),
        ),
      );
      if (unplaced.length) summary.dataset.unplaced = String(unplaced.length);
    } else if (remotePending) summary.append(el("p", { class: "small muted" }, "checking…"));
    else summary.append(el("p", { class: "small" }, el("span", { class: "badge good" }, "valid"), " the server's validator accepts it"));
    if (o.submit) {
      o.submit.disabled = !ok();
      o.submit.title = ok() ? "" : probs.length ? `${probs.length} problem(s): see the list` : "checking…";
    }
  }
  function problems(): Issue[] {
    const out: Issue[] = [];
    const seen = new Set<string>();
    const add = (i: Issue) => {
      const k = `${key(i.path)}|${i.message}`;
      if (!seen.has(k)) seen.add(k), out.push(i);
    };
    for (const f of fields.values()) if (f.box.isConnected) for (const m of [...f.local, ...f.async]) add({ path: f.path, message: m });
    for (const i of remoteIssues) add(i);
    return out;
  }
  const ok = () => !remotePending && problems().length === 0;
  function focus(p: Path) {
    for (let n = p.length; n >= 0; n--) {
      const f = fields.get(key(p.slice(0, n)));
      if (f && f.box.isConnected) {
        f.box.scrollIntoView({ block: "center", behavior: "smooth" });
        (f.box.querySelector<HTMLElement>("[data-ctl]") || f.box.querySelector<HTMLElement>("input,select,textarea,button"))?.focus({ preventScroll: true });
        return;
      }
    }
  }
  let readyWaiters: ((ok: boolean) => void)[] = [];
  function runRemote() {
    if (!o.remote) return;
    clearTimeout(remoteTimer);
    remotePending = true;
    const my = ++remoteSeq;
    remoteTimer = setTimeout(async () => {
      try {
        const r = await o.remote!(clone(value));
        if (my !== remoteSeq) return;
        remoteIssues = r.issues || [];
        remoteWarnings = r.warnings || [];
      } catch (e) {
        if (my !== remoteSeq) return;
        remoteIssues = [{ path: [], message: (e as Error).message }];
      }
      remotePending = false;
      paint();
      const w = readyWaiters;
      readyWaiters = [];
      for (const f of w) f(ok());
    }, 300);
  }
  function changed(f?: Field) {
    if (f) localCheck(f);
    o.onChange?.(value);
    paint();
    runRemote();
  }
  function set(p: Path, v: any, silent = false) {
    value = setIn(value, p, v);
    const f = fields.get(key(p));
    if (f && silent) f.sync();
    if (!silent) {
      for (const g of fields.values()) if (key(g.path).startsWith(key(p)) || key(p).startsWith(key(g.path))) localCheck(g);
      changed(f);
      f?.opts.onSet?.(v);
    } else changed();
  }

  // ---------------------------------------------------------------- controls
  function optionsFor(s: Schema | undefined, opts: FieldOpts): Option[] {
    if (opts.options) return opts.options;
    const u = unwrap(s, root) || {};
    const it = typeOf(u) === "array" ? unwrap(u.items, root) || {} : u;
    const src = it["x-dynamic"];
    const vals = enumValues(it);
    const live = src ? dynOptions(o.dyn, src) : [];
    if (vals) {
      // An enum with a live list: the live list's order and details; enum values it lacks are offered but marked.
      if (live.length) {
        const out: Option[] = live.filter((x) => vals.some((v) => String(v) === x.value)).map((x) => ({ value: vals.find((v) => String(v) === x.value)!, detail: [x.detail, x.info].filter(Boolean).join(" · ") }));
        for (const v of vals) if (!out.some((x) => x.value === v)) out.push({ value: v, detail: src === "gpu_types" ? "not in Runpod's live list" : "", stale: src === "gpu_types" });
        return out;
      }
      return vals.map((v) => ({ value: v }));
    }
    return live.map((x) => ({ value: x.value, detail: [x.detail, x.info].filter(Boolean).join(" · ") }));
  }
  const label = (x: Option) => `${x.label ?? String(x.value)}${x.detail ? ` — ${x.detail}` : ""}`;
  function control(f: Field, kind: ControlKind, s: Schema | undefined): { ctl: HTMLElement; sync: () => void } {
    const u = unwrap(s, root) || {};
    const opts = f.opts;
    const cur = () => getAt(value, f.path);
    const attrs = (extra: Record<string, any> = {}) => ({ "data-ctl": "", "data-schema": o.schemaName, "data-path": key(f.path), "data-kind": kind, "aria-label": opts.label || key(f.path), "aria-describedby": `${f.err.id} ${f.box.dataset.ruleId || ""}`.trim(), ...extra });
    const unsetOk = opts.allowUnset ?? !requiredAt(f.path);
    if (kind === "select") {
      const sel = el("select", attrs());
      const draw = () => {
        const all = optionsFor(s, opts);
        const v = cur();
        const has = all.some((x) => JSON.stringify(x.value) === JSON.stringify(v));
        sel.replaceChildren(
          ...(unsetOk || v === undefined ? [el("option", { value: "" }, unsetOk ? opts.unsetLabel || "(default)" : "— choose —")] : []),
          ...all.map((x) => el("option", { value: JSON.stringify(x.value), class: x.stale ? "ff-stale" : "" }, label(x) + (x.stale ? " (unavailable now)" : ""))),
          ...(v !== undefined && v !== null && !has ? [el("option", { value: JSON.stringify(v), class: "ff-stale" }, `${String(v)} (not available: pick another)`)] : []),
        );
        sel.value = v === undefined ? "" : v === null && !all.some((x) => x.value === null) ? "" : JSON.stringify(v);
        f.async = f.async.filter((m) => !m.startsWith("not available"));
        if (v !== undefined && v !== null && !has && all.length) f.async.push(`not available: ${String(v)} is not offered (any more)`);
        else if (has && all.find((x) => JSON.stringify(x.value) === JSON.stringify(v))?.stale) {
          /* in the schema, not in the live list: a warning, not an error */
        }
      };
      draw();
      sel.addEventListener("change", () => {
        f.async = f.async.filter((m) => !m.startsWith("not available"));
        set(f.path, sel.value === "" ? (u.nullable && !unsetOk ? null : undefined) : JSON.parse(sel.value));
      });
      return { ctl: sel, sync: draw };
    }
    if (kind === "segmented") {
      const box = el("div", { class: "cf-seg ff-seg", role: "radiogroup", ...attrs() });
      const draw = () => {
        const all = optionsFor(s, opts);
        box.replaceChildren(
          ...all.map((x, i) => {
            const on = JSON.stringify(cur()) === JSON.stringify(x.value);
            return el(
              "button",
              {
                type: "button",
                role: "radio",
                "aria-checked": String(on),
                tabindex: on || (cur() === undefined && i === 0) ? "0" : "-1",
                title: x.detail || "",
                onclick: () => (set(f.path, x.value), draw()),
                onkeydown: (ev: KeyboardEvent) => {
                  const d = ev.key === "ArrowRight" || ev.key === "ArrowDown" ? 1 : ev.key === "ArrowLeft" || ev.key === "ArrowUp" ? -1 : 0;
                  if (!d) return;
                  ev.preventDefault();
                  const n = all[(i + d + all.length) % all.length]!;
                  set(f.path, n.value);
                  draw();
                  (box.querySelector('[aria-checked="true"]') as HTMLElement | null)?.focus();
                },
              },
              x.label ?? String(x.value),
            );
          }),
        );
      };
      draw();
      return { ctl: box, sync: draw };
    }
    if (kind === "toggle") {
      const cb = el("input", { type: "checkbox", ...attrs(), checked: !!cur(), onchange: () => set(f.path, cb.checked) });
      return { ctl: el("label", { class: "cf-toggle" }, cb, " ", opts.placeholder || "on"), sync: () => (cb.checked = !!cur()) };
    }
    if (kind === "number") {
      const sc = opts.scale ?? 1;
      const vals = enumValues(u);
      const lo = u.minimum !== undefined ? u.minimum / sc : undefined;
      const hi = u.maximum !== undefined && u.maximum < 9e15 ? u.maximum / sc : undefined;
      const step = opts.step ?? (typeOf(u) === "integer" && sc === 1 ? 1 : "any");
      const inp = el("input", { type: "number", inputmode: "decimal", min: lo, max: hi, step, placeholder: opts.placeholder || (unsetOk ? opts.unsetLabel || "" : ""), ...attrs() });
      const sync = () => {
        const v = cur();
        inp.value = v === undefined || v === null ? "" : String(Math.round((v / sc) * 1000) / 1000);
      };
      sync();
      inp.addEventListener("input", () => {
        if (inp.value === "") return set(f.path, u.nullable ? null : undefined);
        const n = Number(inp.value);
        set(f.path, Number.isFinite(n) ? Math.round(n * sc * 1000) / 1000 : (inp.value as any));
      });
      const unit = opts.unit ?? u["x-unit"];
      void vals;
      return { ctl: unit ? el("span", { class: "ff-num" }, inp, el("span", { class: "ff-unit muted small" }, sc === 60 && unit === "s" ? "min" : unit)) : inp, sync };
    }
    if (kind === "textarea" || kind === "text" || kind === "combo" || kind === "secret") {
      const ta = kind === "textarea";
      const inp: HTMLInputElement | HTMLTextAreaElement = ta
        ? el("textarea", { rows: 8, spellcheck: "false", ...attrs() })
        : el("input", { type: kind === "secret" ? "password" : "text", autocomplete: kind === "secret" ? "new-password" : "off", spellcheck: "false", maxlength: u.maxLength, placeholder: opts.placeholder || "", ...attrs() });
      if (!ta && u.pattern) inp.setAttribute("pattern", u.pattern);
      const sync = () => (inp.value = cur() ?? "");
      sync();
      inp.addEventListener("input", () => set(f.path, inp.value === "" ? (requiredAt(f.path) ? "" : undefined) : inp.value));
      if (kind === "combo") {
        const dl = el("datalist", { id: `ffdl${++uid}` }, ...optionsFor(s, opts).map((x) => el("option", { value: String(x.value) }, x.detail || "")));
        inp.setAttribute("list", dl.id);
        return { ctl: el("span", { class: "ff-combo" }, inp, dl), sync };
      }
      return { ctl: inp, sync };
    }
    if (kind === "chips" || kind === "list") {
      const box = el("div", { class: "cf-multi fv-chips", role: kind === "chips" ? "group" : undefined, ...attrs() });
      if (kind === "list") {
        const inp = el("input", { type: "text", placeholder: opts.placeholder || "comma-separated", ...attrs() });
        const sync = () => (inp.value = (cur() || []).join(", "));
        sync();
        inp.addEventListener("input", () => {
          const xs = inp.value.split(/[,\n]/).map((x) => x.trim()).filter(Boolean);
          set(f.path, xs.length ? xs : requiredAt(f.path) ? [] : undefined);
        });
        return { ctl: inp, sync };
      }
      const draw = () => {
        const all = optionsFor(s, opts);
        const v: Json[] = cur() || [];
        const items = [...all];
        for (const x of v) if (!items.some((y) => JSON.stringify(y.value) === JSON.stringify(x))) items.push({ value: x, detail: "not available", stale: true });
        f.async = f.async.filter((m) => !m.startsWith("not available"));
        const gone = v.filter((x) => !all.some((y) => JSON.stringify(y.value) === JSON.stringify(x)));
        if (gone.length && all.length) f.async.push(`not available: ${gone.join(", ")}`);
        box.replaceChildren(
          ...items.map((x) => {
            const i = v.findIndex((y) => JSON.stringify(y) === JSON.stringify(x.value));
            return el(
              "button",
              {
                type: "button",
                class: `fv-chip${i >= 0 ? " on" : ""}${x.stale ? " ff-stale" : ""}`,
                "aria-pressed": String(i >= 0),
                title: x.detail || "",
                onclick: () => {
                  const next = i >= 0 ? v.filter((_, j) => j !== i) : [...v, x.value];
                  set(f.path, next.length ? next : requiredAt(f.path) ? [] : undefined);
                  draw();
                },
              },
              i >= 0 && v.length > 1 ? `${i + 1}. ` : "",
              x.label ?? String(x.value),
              x.detail ? el("span", { class: "muted" }, ` ${x.detail}`) : null,
            );
          }),
        );
      };
      draw();
      return { ctl: box, sync: draw };
    }
    if (kind === "record") {
      const ed = envEditor({
        value: cur() || {},
        dyn: o.dyn,
        attrs: attrs(),
        ...(opts.env || {}),
        onChange: (v, errs) => {
          f.async = errs;
          set(f.path, Object.keys(v).length ? v : requiredAt(f.path) ? {} : undefined);
        },
      });
      return { ctl: ed.el, sync: () => ed.update(cur() || {}) };
    }
    return { ctl: el("span", { class: "muted small" }, "(edit in the JSON tab)"), sync: () => {} };
  }

  function field(path: Path, opts: FieldOpts = {}): HTMLElement {
    const s = schemaAt(root, path);
    const u = unwrap(s, root) || {};
    const kind = opts.kind || controlKind(s, root);
    const id = `ff${++uid}`;
    const rule = ruleOf(s, root);
    const err = el("div", { class: "cf-err", role: "alert", id: `${id}-err`, hidden: true });
    const box = el("div", { class: `cf-field ff-field${opts.wide || kind === "chips" || kind === "record" || kind === "textarea" ? " wide" : ""}`, "data-path": key(path), "data-schema": o.schemaName, "data-kind": kind });
    if (rule) box.dataset.ruleId = `${id}-rule`;
    const f: Field = { path, box, err, opts, control: box, local: [], async: [], remote: [], warn: [], sync: () => {} };
    const { ctl, sync } = control(f, kind, s);
    f.control = ctl;
    f.sync = sync;
    const lab = el("label", { for: ctl.matches("input,select,textarea") ? (ctl.id ||= `${id}-c`) : undefined }, opts.label || String(path[path.length - 1] ?? ""), requiredAt(path) ? el("span", { class: "ff-req", "aria-hidden": "true" }, " *") : null);
    const parts: (Node | null)[] = [
      lab,
      rule ? el("div", { class: "ff-rule small muted", id: `${id}-rule` }, rule) : null,
      ctl,
      err,
      el("div", { class: "ff-warn small", hidden: true }),
      opts.help !== undefined ? el("div", { class: "cf-help" }, opts.help) : u.description ? el("div", { class: "cf-help" }, u.description) : null,
    ];
    box.append(...(parts.filter(Boolean) as Node[]));
    fields.set(key(path), f);
    localCheck(f);
    if (opts.nameCheck) {
      let t: any;
      let seq = 0;
      const check = () => {
        clearTimeout(t);
        const v = getAt(value, path);
        f.async = f.async.filter((m) => !m.startsWith("taken:"));
        if (typeof v !== "string" || !v || f.local.length) return paint();
        const my = ++seq;
        t = setTimeout(async () => {
          try {
            const r = await o.api(`/api/names/${opts.nameCheck}?name=${encodeURIComponent(v)}`);
            if (my !== seq) return;
            f.async = f.async.filter((m) => !m.startsWith("taken:"));
            if (r.taken) f.async.push(`taken: ${r.problem}`);
            paint();
          } catch {
            /* the server's validator still checks */
          }
        }, 250);
      };
      const prev = opts.onSet;
      opts.onSet = (v) => (check(), prev?.(v));
      check();
    }
    queueMicrotask(paint);
    return box;
  }
  formEl.append(summary);
  if (o.submit) o.submit.disabled = true;
  queueMicrotask(() => {
    paint();
    runRemote();
  });
  return {
    el: formEl,
    summary,
    field,
    value: () => clone(value),
    get: (p) => getAt(value, p),
    set,
    ok,
    problems,
    check() {
      for (const f of fields.values()) localCheck(f);
      paint();
      runRemote();
    },
    prune() {
      for (const [k, f] of fields) if (!f.box.isConnected) fields.delete(k);
      paint();
    },
    ready: () => (remotePending ? new Promise<boolean>((r) => readyWaiters.push(r)) : Promise.resolve(ok())),
  };
}

// ---------------------------------------------------------------- env editor
export interface EnvOpts {
  /** "plain": {KEY: "v"}; "launch": {KEY: "v" | {value, secret}}; "doc": {KEY: {value, secret, set?}} (the env documents). */
  shape?: "plain" | "launch" | "doc";
  secrets?: boolean;
  /** Extra reserved keys (the serverless template's). */
  reserved?: string[];
}
export interface EnvEditor {
  el: HTMLElement;
  update(v: Record<string, any>): void;
}
type Row = { key: string; value: string; secret: boolean; stored: boolean; set?: string };
const ENV_KEY_RE = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;
/**
 * Key / value rows: the key's rule before typing, reserved and duplicate keys
 * refused as you type, a typed value control for the engine keys fv-serve
 * knows (a select for enums, 0 / 1 for switches), secrets masked and write-only.
 */
export function envEditor(o: EnvOpts & { value: Record<string, any>; dyn?: Dynamic; attrs?: Record<string, any>; onChange: (v: Record<string, any>, errors: string[]) => void }): EnvEditor {
  const shape = o.shape || "plain";
  const types: Record<string, { kind: string; values?: string[] }> = o.dyn?.env_types || {};
  const reserved = new Set<string>([...(o.dyn?.reserved_env_keys || []), ...(o.reserved || [])]);
  let rows: Row[] = [];
  const fromValue = (v: Record<string, any>) =>
    (rows = Object.entries(v || {}).map(([k, x]) =>
      typeof x === "string" ? { key: k, value: x, secret: false, stored: false } : shape === "doc" ? { key: k, value: x.value ?? "", secret: !!x.secret, stored: x.secret && x.value === null && x.set === undefined, set: x.set } : { key: k, value: x.value ?? "", secret: !!x.secret, stored: false },
    ));
  fromValue(o.value);
  const box = el("div", { class: "ff-env", ...(o.attrs || {}), "data-kind": "record" });
  const listId = `ff-envkeys${++uid}`;
  const tbody = el("tbody");
  const keyProblem = (r: Row, i: number): string | null => {
    if (!r.key) return "a name is required";
    if (!ENV_KEY_RE.test(r.key)) return "a letter or '_', then letters, digits or '_' (max 128)";
    if (reserved.has(r.key)) return `${r.key} is set by the controller: it cannot be overridden`;
    if (rows.findIndex((x) => x.key === r.key) !== i) return `${r.key} twice`;
    return null;
  };
  const valueProblem = (r: Row): string | null => {
    const t = types[r.key];
    if (!t || r.secret) return null;
    const v = r.value.trim().toLowerCase();
    if (t.kind === "bool") return ["0", "1", "true", "false", "on", "off"].includes(v) ? null : "a switch: 0 or 1";
    if (t.kind === "enum") return t.values!.includes(v) ? null : `one of ${t.values!.join(", ")}`;
    if (t.kind === "path") return /^\/\S*$/.test(r.value) ? null : "an absolute path";
    return null;
  };
  const emit = () => {
    const out: Record<string, any> = {};
    const errs: string[] = [];
    rows.forEach((r, i) => {
      const kp = keyProblem(r, i);
      const vp = valueProblem(r);
      if (kp) errs.push(`${r.key || `row ${i + 1}`}: ${kp}`);
      if (vp) errs.push(`${r.key}: ${vp}`);
      if (!r.key) return;
      if (shape === "plain") out[r.key] = r.value;
      else if (shape === "launch") out[r.key] = r.secret ? { value: r.value, secret: true } : r.value;
      else out[r.key] = r.secret ? { value: null, secret: true, ...(r.set !== undefined ? { set: r.set } : {}) } : { value: r.value, secret: false };
      if (shape === "doc" && r.secret && r.set === undefined && !r.stored) errs.push(`${r.key}: a new secret needs its value`);
    });
    o.onChange(out, errs);
  };
  function draw() {
    tbody.replaceChildren(
      ...rows.map((r, i) => {
        const kp = keyProblem(r, i);
        const t = types[r.key];
        const err = el("div", { class: "cf-err", role: "alert" });
        const refresh = () => {
          const k = keyProblem(r, i);
          const v = valueProblem(r);
          err.textContent = [k, v].filter(Boolean).join("; ");
          tr.classList.toggle("has-err", !!(k || v));
        };
        const keyIn = el("input", { type: "text", value: r.key, spellcheck: "false", "aria-label": "variable name", placeholder: "NAME", pattern: ENV_KEY_RE.source, list: listId, class: "mono" });
        keyIn.addEventListener("input", () => {
          r.key = keyIn.value.trim();
          refresh();
          emit();
        });
        keyIn.addEventListener("change", () => draw());
        let val: HTMLElement;
        if (r.secret) {
          const pw = el("input", { type: "password", autocomplete: "new-password", "aria-label": `${r.key || "variable"} value (secret)`, placeholder: r.stored ? "•••••••• stored (type to replace)" : "write-only: the value", value: shape === "doc" ? r.set ?? "" : r.value });
          pw.addEventListener("input", () => {
            if (shape === "doc") r.set = pw.value === "" ? undefined : pw.value;
            else r.value = pw.value;
            refresh();
            emit();
          });
          val = pw;
        } else if (t && (t.kind === "enum" || t.kind === "bool")) {
          const vals = t.kind === "bool" ? ["0", "1"] : t.values!;
          const sel = el("select", { "aria-label": `${r.key} value` }, el("option", { value: "" }, "— choose —"), ...vals.map((v) => el("option", { value: v }, v)), ...(r.value && !vals.includes(r.value.toLowerCase()) ? [el("option", { value: r.value }, `${r.value} (not valid)`)] : []));
          sel.value = r.value;
          sel.addEventListener("change", () => {
            r.value = sel.value;
            refresh();
            emit();
          });
          val = sel;
        } else {
          const inp = el("input", { type: "text", value: r.value, spellcheck: "false", "aria-label": `${r.key || "variable"} value` });
          inp.addEventListener("input", () => {
            r.value = inp.value;
            refresh();
            emit();
          });
          val = inp;
        }
        const tr = el(
          "tr",
          { "data-key": r.key, class: kp ? "has-err" : "" },
          el("td", {}, keyIn),
          el("td", { class: "fv-grow" }, val, err, t ? el("div", { class: "cf-help" }, (o.dyn?.env_keys || []).find((x: any) => x.id === r.key)?.detail || "") : null),
          o.secrets !== false && shape !== "plain"
            ? el(
                "td",
                {},
                el(
                  "label",
                  { class: "fv-inline cf-toggle" },
                  el("input", {
                    type: "checkbox",
                    checked: r.secret,
                    "aria-label": `${r.key || "variable"} is a secret`,
                    onchange: (ev: any) => {
                      r.secret = ev.target.checked;
                      if (shape === "doc") {
                        if (r.secret) (r.set = r.value || undefined), (r.value = ""), (r.stored = false);
                        else (r.value = r.set ?? ""), (r.set = undefined);
                      }
                      emit();
                      draw();
                    },
                  }),
                  "secret",
                ),
              )
            : null,
          el("td", {}, el("button", { type: "button", class: "ghost danger", "aria-label": `remove ${r.key || "row"}`, onclick: () => (rows.splice(i, 1), emit(), draw()) }, "✕")),
        );
        refresh();
        return tr;
      }),
    );
  }
  const keyOpts = el("datalist", { id: listId }, ...dynOptions(o.dyn, "env_keys").map((x) => el("option", { value: x.value }, x.detail || "")));
  box.append(
    el("div", { class: "ff-rule small muted" }, "Names: a letter or '_', then letters, digits or '_'. Not a controller key", o.reserved?.length ? " or a serverless template key" : "", ". Known engine keys get a typed value."),
    el("div", { class: "tablewrap" }, el("table", { class: "ff-envtable" }, el("thead", {}, el("tr", {}, el("th", {}, "name"), el("th", {}, "value"), o.secrets !== false && shape !== "plain" ? el("th", {}, "") : null, el("th", {}, ""))), tbody)),
    el("button", { type: "button", class: "small", onclick: () => (rows.push({ key: "", value: "", secret: false, stored: false }), draw(), (tbody.lastElementChild?.querySelector("input") as HTMLElement | null)?.focus(), emit()) }, "+ variable"),
    keyOpts,
  );
  draw();
  return {
    el: box,
    update(v) {
      fromValue(v);
      draw();
    },
  };
}

// ---------------------------------------------------------------- a dialog form
/** A modal with schema-bound fields (extend, scale, roll, mint key …): resolves to the value, or null when cancelled. */
export function formDialog(o: { title: string; intro?: string; schemaName: string; schema: Schema; dyn?: Dynamic; api: Api; value: any; fields: [Path, FieldOpts?][]; submitLabel: string; remote?: FormOptions["remote"] }): Promise<any | null> {
  return new Promise((resolveP) => {
    const dlg = el("dialog", { class: "ff-dialog", "aria-label": o.title });
    const go = el("button", { type: "submit", class: "primary" }, o.submitLabel);
    const f = createForm({ schemaName: o.schemaName, schema: o.schema, dyn: o.dyn, api: o.api, value: o.value, submit: go, remote: o.remote });
    let done = false;
    const finish = (v: any) => {
      if (done) return;
      done = true;
      dlg.close();
      resolveP(v);
    };
    const formTag = el(
      "form",
      { method: "dialog", onsubmit: (ev: Event) => (ev.preventDefault(), f.ok() && finish(f.value())) },
      el("h2", {}, o.title),
      o.intro ? el("p", { class: "muted small" }, o.intro) : null,
      el("div", { class: "cf-fields" }, ...o.fields.map(([p, fo]) => f.field(p, fo))),
      f.summary,
      el("div", { class: "row", style: "margin-top:10px" }, go, el("button", { type: "button", onclick: () => finish(null) }, "Cancel")),
    );
    f.el.append(formTag);
    dlg.append(f.el);
    dlg.addEventListener("close", () => {
      finish(null);
      dlg.remove();
    });
    document.body.append(dlg);
    dlg.showModal();
    (dlg.querySelector("[data-ctl]") as HTMLElement | null)?.focus();
  });
}

/** Every control of the schema-bound forms on the page that has no schema entry, or renders an enum as free text (the UI coverage check). */
export function auditForms(doc: ParentNode, schemas: Record<string, Schema>): string[] {
  const out: string[] = [];
  for (const form of doc.querySelectorAll<HTMLElement>("[data-schema-form]")) {
    const name = form.dataset.schemaForm!;
    const root = schemas[name];
    if (!root) {
      out.push(`${name}: no such schema`);
      continue;
    }
    for (const c of form.querySelectorAll<HTMLElement>("input, select, textarea")) {
      if (c.closest(".ff-env")) continue; // env rows: their own key / value rules (the record's schema)
      if (c.closest("[data-schema-ignore]")) continue; // page controls that are not part of the document (a template picker)
      const host = c.closest<HTMLElement>("[data-path]");
      if (!host || !form.contains(host)) {
        out.push(`${name}: a control (${c.outerHTML.slice(0, 80)}) has no schema path`);
        continue;
      }
      const p = host.dataset.path ? host.dataset.path.split(".").map((x) => (/^\d+$/.test(x) ? Number(x) : x)) : [];
      const s = schemaAt(root, p);
      if (!s) {
        out.push(`${name}: ${host.dataset.path} is not in the schema`);
        continue;
      }
      const u = unwrap(s, root) || {};
      const free = c.tagName === "INPUT" && ["text", "search", ""].includes((c as HTMLInputElement).type) && !c.closest(".fv-chips");
      if (free && (enumValues(u) || enumValues(unwrap(u.items, root)))) out.push(`${name}: ${host.dataset.path} is an enum rendered as free text`);
    }
  }
  return out;
}
export { resolve };
