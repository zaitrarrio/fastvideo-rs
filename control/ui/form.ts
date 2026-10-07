// A form generated from a JSON Schema, kept in sync with the raw JSON tab:
// objects become fieldsets, enums selects, live values (x-dynamic)
// datalists or chip pickers, booleans toggles, arrays of objects tables
// (a row per item, the item's full form behind "more"), env sets a
// key / value / secret table with write-only secret inputs.
import type { Dynamic } from "./editor";
import { dynOptions } from "./editor";
import { CLOSED_SOURCES, controlKind, envEditor, ruleOf } from "./fields";
import { resolve, unwrap, validate, type Json, type Path, type Schema } from "./schema";

type Change = (v: any) => void;
let uid = 0;
function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, any> = {}, ...kids: (Node | string | null | undefined | false)[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "class") e.className = v;
    else if (k === "value") (e as any).value = v;
    else if (k === "checked") (e as any).checked = !!v;
    else e.setAttribute(k, v === true ? "" : String(v));
  }
  for (const k of kids) if (k !== null && k !== undefined && k !== false) e.append(k);
  return e;
}
const typeOf = (u: Schema | undefined) => (Array.isArray(u?.type) ? u!.type.find((t: string) => t !== "null") : u?.type);
const isPrimitive = (u: Schema | undefined) => !!u && (u.enum || ["string", "number", "integer", "boolean"].includes(typeOf(u)));

export interface FormOptions {
  schema: Schema;
  value: any;
  dyn?: Dynamic;
  onChange: Change;
}
export interface FormHandle {
  update(v: any): void;
}

export function renderForm(host: HTMLElement, o: FormOptions): FormHandle {
  let value = o.value;
  const root = o.schema;
  const set = (v: any, rerender = false) => {
    value = v;
    o.onChange(v);
    if (rerender) draw();
  };
  function draw() {
    host.replaceChildren(node(root, value, [], (v, rr) => set(v, rr), true));
  }
  function node(s: Schema | undefined, v: any, path: Path, change: (v: any, rerender?: boolean) => void, top = false): HTMLElement {
    const u = unwrap(s, root) || {};
    const ty = typeOf(u);
    if (ty === "object" && u.additionalProperties && typeof u.additionalProperties === "object" && !u.properties) return envTable(u, v || {}, change, path);
    if (ty === "object") return objectForm(u, v || {}, path, change, top);
    if (ty === "array") {
      const it = unwrap(u.items, root);
      if (typeOf(it) === "object") return tableForm(u, it!, Array.isArray(v) ? v : [], path, change);
      return listField(u, it, Array.isArray(v) ? v : [], change);
    }
    return field(u, v, change);
  }
  function objectForm(u: Schema, v: Record<string, any>, path: Path, change: (v: any, rr?: boolean) => void, top: boolean) {
    const box = el("div", { class: top ? "fv-form" : "fv-form nested" });
    const req = new Set<string>(u.required || []);
    for (const [k, ps] of Object.entries<Schema>(u.properties || {})) {
      const pu = unwrap(ps, root) || {};
      const id = `f${++uid}`;
      const err = el("div", { class: "cf-err", role: "alert" });
      const onChange = (x: any, rr = false) => {
        const next = { ...v };
        if (x === undefined) delete next[k];
        else next[k] = x;
        v = next;
        // This field's own rules, as you type (the server's validator adds the cross-field ones).
        err.textContent = x === undefined ? (req.has(k) ? "required" : "") : validate(ps, x, root).map((i) => (i.path.length ? `${i.path.join(".")}: ` : "") + i.message).join("; ");
        row.classList.toggle("has-err", !!err.textContent);
        change(next, rr);
      };
      const ctl = node(ps, v[k], [...path, k], onChange);
      const complex = typeOf(pu) === "object" || typeOf(pu) === "array";
      const lab = el("label", { for: id, title: pu.description || "" }, k, req.has(k) ? "" : el("span", { class: "muted" }, " (optional)"));
      const firstInput = ctl.matches("input,select,textarea") ? ctl : ctl.querySelector("input,select,textarea");
      if (firstInput && !firstInput.id) firstInput.id = id;
      const rule = ruleOf(ps, root);
      const row = el("div", { class: `fv-row${complex ? " wide" : ""}`, "data-path": [...path, k].join("."), "data-kind": controlKind(ps, root) }, lab, rule ? el("div", { class: "fv-help ff-rule" }, rule) : null, ctl, err, pu.description ? el("div", { class: "fv-help" }, pu.description) : null);
      box.append(row);
    }
    return box;
  }
  function field(u: Schema, v: any, change: (v: any, rr?: boolean) => void): HTMLElement {
    const ty = typeOf(u);
    const optional = (x: any) => (x === "" || x === undefined ? (u.nullable ? null : undefined) : x);
    if (u.enum) {
      const sel = el("select", { onchange: () => change(optional(sel.value === "" ? undefined : JSON.parse(sel.value))) }, el("option", { value: "" }, "—"), ...u.enum.map((e: Json) => el("option", { value: JSON.stringify(e), selected: JSON.stringify(e) === JSON.stringify(v) }, String(e))));
      return sel;
    }
    if (ty === "boolean") {
      const cb = el("input", { type: "checkbox", checked: !!v, onchange: () => change(cb.checked) });
      return el("span", { class: "fv-toggle" }, cb);
    }
    if (ty === "number" || ty === "integer") {
      const inp = el("input", { type: "number", step: ty === "integer" ? 1 : "any", value: v ?? "", min: u.minimum, max: u.maximum !== undefined && u.maximum < 9e15 ? u.maximum : undefined, oninput: () => change(optional(inp.value === "" ? "" : Number(inp.value))) });
      return u["x-unit"] ? el("span", { class: "ff-num" }, inp, el("span", { class: "ff-unit muted small" }, u["x-unit"])) : inp;
    }
    if (u["x-secret"]) {
      const inp = el("input", { type: "password", autocomplete: "new-password", placeholder: "write-only: type a new value", value: "", oninput: () => change(inp.value === "" ? undefined : inp.value) });
      return inp;
    }
    const src = u["x-dynamic"];
    if (src && CLOSED_SOURCES.has(src) && ty === "string") {
      // A live list that is the whole choice: a select; a value no longer offered stays, flagged.
      const opts = dynOptions(o.dyn, src);
      const sel = el("select", { onchange: () => change(optional(sel.value)) }, el("option", { value: "" }, "—"), ...opts.map((x) => el("option", { value: x.value, selected: x.value === v }, [x.value, x.detail].filter(Boolean).join(" — "))), ...(v && !opts.some((x) => x.value === v) ? [el("option", { value: v, selected: true, class: "ff-stale" }, `${v} (not available)`)] : []));
      return sel;
    }
    if (u["x-ui"] === "textarea" || (u.maxLength && u.maxLength > 1000 && !u.pattern)) {
      const ta = el("textarea", { rows: 6, value: v ?? "", oninput: () => change(optional(ta.value)) });
      return ta;
    }
    const inp = el("input", { type: "text", value: v ?? "", pattern: u.pattern, maxlength: u.maxLength, spellcheck: "false", oninput: () => change(optional(inp.value)) });
    if (src) {
      const dl = el("datalist", { id: `dl${++uid}` }, ...dynOptions(o.dyn, src).map((x) => el("option", { value: x.value }, [x.detail, x.info].filter(Boolean).join(" · "))));
      inp.setAttribute("list", dl.id);
      return el("span", { class: "fv-dyn" }, inp, dl);
    }
    return inp;
  }
  function listField(u: Schema, it: Schema | undefined, v: any[], change: (v: any, rr?: boolean) => void): HTMLElement {
    const src = it?.["x-dynamic"];
    const opts = it?.enum ? it.enum.map((e: Json) => ({ value: String(e) })) : src ? dynOptions(o.dyn, src) : null;
    if (opts && opts.length) {
      // Ordered chips: click to add or remove; the order of selection is kept (placement order).
      const box = el("div", { class: "fv-chips" });
      const all = [...new Set([...v.map(String), ...opts.map((x: any) => x.value)])];
      for (const val of all) {
        const on = v.includes(val);
        const meta = opts.find((x: any) => x.value === val);
        box.append(
          el(
            "button",
            { type: "button", class: `fv-chip${on ? " on" : ""}`, "aria-pressed": on ? "true" : "false", title: meta?.info || "", onclick: () => change(on ? (v.filter((x) => x !== val).length || !u.nullable ? v.filter((x) => x !== val) : undefined) : [...v, val], true) },
            on ? `${v.indexOf(val) + 1}. ` : "",
            val,
            meta?.detail ? el("span", { class: "muted" }, ` ${meta.detail}`) : "",
          ),
        );
      }
      return box;
    }
    const inp = el("input", { type: "text", value: v.join(", "), placeholder: "comma-separated", oninput: () => { const xs = inp.value.split(",").map((x) => x.trim()).filter(Boolean); change(xs.length ? xs : undefined); } });
    return inp;
  }
  function tableForm(u: Schema, it: Schema, v: any[], path: Path, change: (v: any, rr?: boolean) => void): HTMLElement {
    const props = Object.entries<Schema>(it.properties || {});
    const req = new Set<string>(it.required || []);
    const cols = props.filter(([k, ps]) => isPrimitive(unwrap(ps, root)) && (req.has(k) || props.length <= 4)).slice(0, 6);
    const wrap = el("div", { class: "fv-table" });
    const tbody = el("tbody");
    v.forEach((row, i) => {
      const setRow = (r: any, rr = false) => {
        const next = [...v];
        next[i] = r;
        v = next;
        change(next, rr);
      };
      const more = el("tr", { class: "fv-more", hidden: true }, el("td", { colspan: cols.length + 1 }, objectForm(it, row, [...path, i], (r, rr) => setRow(r, rr), false)));
      tbody.append(
        el(
          "tr",
          { "data-path": [...path, i].join(".") },
          ...cols.map(([k, ps]) => el("td", {}, field(unwrap(ps, root) || {}, row?.[k], (x) => setRow(x === undefined ? (({ [k]: _, ...rest }) => rest)(row) : { ...row, [k]: x }, false)))),
          el(
            "td",
            { class: "fv-actions" },
            props.length > cols.length ? el("button", { type: "button", class: "ghost", onclick: () => (more.hidden = !more.hidden), "aria-label": "more fields" }, "more") : null,
            el("button", { type: "button", class: "ghost danger", onclick: () => change(v.filter((_, j) => j !== i), true), "aria-label": "remove row" }, "✕"),
          ),
        ),
        more,
      );
    });
    const minRow = () => {
      const o2: any = {};
      for (const r of it.required || []) {
        const ps = unwrap(it.properties?.[r], root);
        const t2 = typeOf(ps);
        o2[r] = ps?.enum ? ps.enum[0] : t2 === "number" || t2 === "integer" ? ps?.minimum ?? 0 : t2 === "boolean" ? false : t2 === "array" ? [] : t2 === "object" ? {} : "";
      }
      return o2;
    };
    wrap.append(
      el("div", { class: "tablewrap" }, el("table", {}, el("thead", {}, el("tr", {}, ...cols.map(([k, ps]) => el("th", { title: unwrap(ps, root)?.description || "" }, k)), el("th", {}, ""))), tbody)),
      el("button", { type: "button", onclick: () => change([...v, minRow()], true) }, "+ add"),
    );
    return wrap;
  }
  function envTable(_u: Schema, v: Record<string, any>, change: (v: any, rr?: boolean) => void, path: Path): HTMLElement {
    // Key rule and reserved keys as you type, typed values for the engine keys, secrets write-only (ui/fields.ts envEditor).
    const err = el("div", { class: "cf-err", role: "alert" });
    const ed = envEditor({
      value: v,
      dyn: o.dyn,
      shape: "doc",
      secrets: true,
      onChange: (next, errs) => {
        err.textContent = errs.join("; ");
        change(next, false);
      },
    });
    return el("div", { class: "fv-table", "data-path": path.join(".") }, ed.el, err);
  }
  draw();
  return {
    update(v: any) {
      value = v;
      draw();
    },
  };
}
export { resolve };
