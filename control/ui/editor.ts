// The smart JSON editor: CodeMirror 6 with schema validation (inline
// diagnostics), hover docs, completion of keys, enum values and live values
// (GPU types with price and stock, channels, variants, env keys, …),
// folding, search, format, and light / dark through the page's CSS tokens.
import { autocompletion, closeBrackets, closeBracketsKeymap, completionKeymap, type Completion, type CompletionContext, type CompletionResult } from "@codemirror/autocomplete";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { json } from "@codemirror/lang-json";
import { bracketMatching, foldGutter, foldKeymap, HighlightStyle, indentOnInput, syntaxHighlighting, syntaxTree } from "@codemirror/language";
import { forceLinting, linter, lintGutter, type Diagnostic } from "@codemirror/lint";
import { highlightSelectionMatches, search, searchKeymap } from "@codemirror/search";
import { EditorState, type Extension } from "@codemirror/state";
import { EditorView, highlightActiveLine, hoverTooltip, keymap, lineNumbers, placeholder } from "@codemirror/view";
import type { SyntaxNode } from "@lezer/common";
import { tags as t } from "@lezer/highlight";
import { schemaAt, unwrap, validate, type Issue, type Path, type Schema } from "./schema";

export type Dynamic = Record<string, any>;
export interface DynOption {
  value: string;
  detail?: string;
  info?: string;
}
/** The options of an x-dynamic source. */
export function dynOptions(dyn: Dynamic | undefined, src: string): DynOption[] {
  const d = dyn?.[src];
  if (!Array.isArray(d)) return [];
  if (src === "gpu_types")
    return d.map((g: any) => ({ value: g.id, detail: `${g.secure_price != null ? `$${g.secure_price}/hr` : "no price"}${g.stock ? ` · ${g.stock} stock` : ""}`, info: `${g.display}${g.memory_gb ? `, ${g.memory_gb} GB` : ""}` }));
  if (src === "channels") return d.map((c: any) => ({ value: c.id, detail: c.sha ? `at ${c.sha}` : "", info: Object.keys(c.digests || {}).length ? `${Object.keys(c.digests).length} image digests` : undefined }));
  if (src === "regions") return d.map((r: any) => ({ value: r.id, detail: `${r.dc} · volume ${r.volume}` }));
  return d.map((x: any) => (typeof x === "string" ? { value: x } : { value: String(x.id ?? x.value), detail: x.detail }));
}

const VALUE_NODES = new Set(["Object", "Array", "String", "Number", "True", "False", "Null"]);
const unq = (s: string) => {
  try {
    return JSON.parse(s);
  } catch {
    return s.replace(/^"|"$/g, "");
  }
};
/** The JSON path of a syntax node (a PropertyName yields its key's path, with `key: true`). */
export function pathOfNode(state: EditorState, node: SyntaxNode): { path: Path; key: boolean } {
  const path: Path = [];
  let key = false;
  let n: SyntaxNode | null = node;
  while (n && !VALUE_NODES.has(n.name) && n.name !== "PropertyName" && n.parent) n = n.parent;
  if (n?.name === "PropertyName") key = true;
  while (n && n.parent) {
    const p: SyntaxNode = n.parent;
    if (p.name === "Property") {
      const nm = p.getChild("PropertyName");
      if (nm) path.unshift(unq(state.sliceDoc(nm.from, nm.to)));
    } else if (p.name === "Array" && VALUE_NODES.has(n.name)) {
      let i = 0;
      for (let c = p.firstChild; c && c.from < n.from; c = c.nextSibling) if (VALUE_NODES.has(c.name)) i++;
      path.unshift(i);
    }
    n = p;
  }
  return { path, key };
}
/** Where a path lives in the document (the key of the last property when `key`), for diagnostics. */
export function rangeOfPath(state: EditorState, path: Path, key = false): { from: number; to: number } {
  const top = syntaxTree(state).topNode;
  let n: SyntaxNode | null = top.firstChild;
  while (n && !VALUE_NODES.has(n.name)) n = n.nextSibling;
  if (!n) return { from: 0, to: Math.min(1, state.doc.length) };
  for (let i = 0; i < path.length; i++) {
    const seg = path[i];
    let next: SyntaxNode | null = null;
    if (n.name === "Object" && typeof seg === "string") {
      for (let p: SyntaxNode | null = n.firstChild; p; p = p.nextSibling) {
        if (p.name !== "Property") continue;
        const nm = p.getChild("PropertyName");
        if (nm && unq(state.sliceDoc(nm.from, nm.to)) === seg) {
          if (i === path.length - 1 && key) return { from: nm.from, to: nm.to };
          next = p.lastChild && VALUE_NODES.has(p.lastChild.name) ? p.lastChild : nm;
          break;
        }
      }
    } else if (n.name === "Array" && typeof seg === "number") {
      let k = 0;
      for (let c: SyntaxNode | null = n.firstChild; c; c = c.nextSibling) if (VALUE_NODES.has(c.name) && k++ === seg) { next = c; break; }
    }
    if (!next) break; // a missing key: point at its container
    n = next;
  }
  // Objects and arrays: underline only the opening bracket line, not the whole block.
  if (n.name === "Object" || n.name === "Array") return { from: n.from, to: n.from + 1 };
  return { from: n.from, to: n.to };
}

export interface EditorOptions {
  parent: HTMLElement;
  doc: string;
  schema?: Schema;
  dynamic?: Dynamic;
  readOnly?: boolean;
  onChange?: (text: string) => void;
  /** Extra issues (from the server's validation). */
  extraIssues?: () => Issue[];
  placeholder?: string;
}
export interface JsonEditor {
  view: EditorView;
  get(): string;
  set(text: string): void;
  format(): boolean;
  relint(): void;
  setSchema(s: Schema | undefined, dyn?: Dynamic): void;
  destroy(): void;
}

const theme = EditorView.theme({
  "&": { fontSize: "13px", backgroundColor: "var(--surface-1)", color: "var(--text-primary)", border: "1px solid var(--border)", borderRadius: "8px" },
  "&.cm-focused": { outline: "2px solid var(--accent)", outlineOffset: "-1px" },
  ".cm-scroller": { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", lineHeight: "1.5", maxHeight: "65vh" },
  ".cm-content": { caretColor: "var(--text-primary)", padding: "6px 0" },
  ".cm-gutters": { backgroundColor: "var(--surface-2)", color: "var(--text-muted)", border: "none", borderRadius: "8px 0 0 8px" },
  ".cm-activeLine": { backgroundColor: "color-mix(in srgb, var(--accent) 7%, transparent)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--text-primary)" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection": { backgroundColor: "color-mix(in srgb, var(--accent) 28%, transparent) !important" },
  ".cm-tooltip": { backgroundColor: "var(--surface-1)", color: "var(--text-primary)", border: "1px solid var(--border)", borderRadius: "8px", maxWidth: "420px" },
  ".cm-tooltip-autocomplete ul li[aria-selected]": { backgroundColor: "var(--accent)", color: "#fff" },
  ".cm-completionDetail": { color: "var(--text-muted)", fontStyle: "normal", marginLeft: "8px" },
  ".cm-panels": { backgroundColor: "var(--surface-2)", color: "var(--text-primary)" },
  ".cm-panel input, .cm-panel button": { font: "inherit", fontSize: "12px" },
  ".cm-hover": { padding: "6px 9px", fontSize: "12.5px", lineHeight: "1.45" },
  ".cm-hover b": { fontWeight: "600" },
  ".cm-hover code": { fontSize: "11.5px" },
  ".cm-diagnostic": { fontSize: "12.5px" },
});
const highlight = HighlightStyle.define([
  { tag: t.propertyName, color: "var(--series-1)" },
  { tag: t.string, color: "var(--series-6)" },
  { tag: t.number, color: "var(--series-2)" },
  { tag: [t.bool, t.null], color: "var(--series-7)" },
  { tag: t.punctuation, color: "var(--text-muted)" },
]);

function lintSource(opts: { schema: () => Schema | undefined; extra?: () => Issue[] }) {
  return (view: EditorView): Diagnostic[] => {
    const text = view.state.doc.toString();
    if (!text.trim()) return [];
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch (e) {
      const m = /position (\d+)/.exec((e as Error).message);
      const pos = m ? Math.min(Number(m[1]), text.length) : text.length;
      return [{ from: Math.max(0, pos - 1), to: Math.min(text.length, pos + 1), severity: "error", message: `JSON: ${(e as Error).message.replace(/^JSON\.parse: /, "")}`, source: "json" }];
    }
    const s = opts.schema();
    const issues = [...(s ? validate(s, value) : []), ...(opts.extra?.() || [])];
    const seen = new Set<string>();
    return issues
      .filter((i) => !seen.has(i.path.join(".") + i.message) && seen.add(i.path.join(".") + i.message))
      .map((i) => {
        const r = rangeOfPath(view.state, i.path, /unknown property|^key /.test(i.message));
        return { from: r.from, to: r.to, severity: "error" as const, message: `${i.path.length ? i.path.join(".") + ": " : ""}${i.message}`, source: "schema" };
      });
  };
}

function describe(s: Schema | undefined, root: Schema, dyn: Dynamic | undefined, value?: unknown): HTMLElement | null {
  const u = unwrap(s, root);
  if (!u) return null;
  const el = document.createElement("div");
  el.className = "cm-hover";
  const add = (tag: string, text: string) => {
    const x = document.createElement(tag);
    x.textContent = text;
    el.append(x);
    return x;
  };
  const type = u.enum ? "enum" : Array.isArray(u.type) ? u.type.join(" | ") : u.type || "any";
  add("b", `${type}${u.nullable ? " | null" : ""}`);
  if (u.description) add("div", u.description);
  if (u.enum) add("div", `one of ${u.enum.map((e: unknown) => JSON.stringify(e)).join(", ")}`);
  const range = [u.minimum !== undefined ? `≥ ${u.minimum}` : "", u.maximum !== undefined && u.maximum < 9e15 ? `≤ ${u.maximum}` : "", u.pattern ? `pattern ${u.pattern}` : ""].filter(Boolean).join(", ");
  if (range) add("div", range).style.color = "var(--text-muted)";
  const src = u["x-dynamic"] || u.items?.["x-dynamic"];
  if (src && typeof value === "string") {
    const o = dynOptions(dyn, src).find((x) => x.value === value);
    if (o) add("div", `${o.value}: ${[o.detail, o.info].filter(Boolean).join(" · ")}`);
    else if (dynOptions(dyn, src).length) add("div", `not a known ${src.replace(/_/g, " ")} value`).style.color = "var(--serious)";
  }
  return el;
}

function completionSource(get: () => { schema?: Schema; dyn?: Dynamic }) {
  return (ctx: CompletionContext): CompletionResult | null => {
    const { schema, dyn } = get();
    if (!schema) return null;
    const word = ctx.matchBefore(/"?[^"\s,:{}\[\]]*"?/);
    if (!word) return null;
    if (word.from === word.to && !ctx.explicit) return null;
    const text = ctx.state.sliceDoc(0, word.from);
    const prev = text.replace(/\s+$/, "").slice(-1);
    // The innermost object / array around the cursor.
    let n: SyntaxNode | null = syntaxTree(ctx.state).resolveInner(word.from, -1);
    while (n && n.name !== "Object" && n.name !== "Array") n = n.parent;
    if (!n) return null;
    const container = pathOfNode(ctx.state, n).path;
    let options: Completion[] = [];
    if (n.name === "Object" && (prev === "{" || prev === ",")) {
      // A key.
      const s = unwrap(schemaAt(schema, container), schema);
      const present = new Set<string>();
      for (let p = n.firstChild; p; p = p.nextSibling) if (p.name === "Property") { const nm = p.getChild("PropertyName"); if (nm && !(nm.from <= word.from && nm.to >= word.to)) present.add(unq(ctx.state.sliceDoc(nm.from, nm.to))); }
      const req = new Set<string>(s?.required || []);
      for (const [k, ps] of Object.entries<Schema>(s?.properties || {})) {
        if (present.has(k)) continue;
        const u = unwrap(ps, schema);
        options.push({ label: k, type: "property", detail: `${u?.type || (u?.enum ? "enum" : "")}${req.has(k) ? " · required" : ""}`, info: u?.description, apply: `"${k}": `, boost: req.has(k) ? 1 : 0 });
      }
      const pn = s?.propertyNames && unwrap(s.propertyNames, schema);
      if (pn?.["x-dynamic"]) for (const o of dynOptions(dyn, pn["x-dynamic"])) if (!present.has(o.value)) options.push({ label: o.value, type: "property", detail: o.detail || "in use", apply: `"${o.value}": ` });
    } else {
      // A value: after "key": in an object, or an item of an array.
      let path: Path;
      if (n.name === "Object") {
        const m = /"((?:[^"\\]|\\.)*)"\s*:\s*$/.exec(text);
        if (!m) return null;
        path = [...container, unq(`"${m[1]}"`)];
      } else path = [...container, 0];
      const s = unwrap(schemaAt(schema, path), schema);
      if (!s) return null;
      const lit = (v: unknown, extra: Partial<Completion> = {}) => ({ label: JSON.stringify(v), type: "enum", apply: JSON.stringify(v), ...extra });
      for (const e of s.enum || []) options.push(lit(e));
      const src = s["x-dynamic"];
      if (src) for (const o of dynOptions(dyn, src)) options.push(lit(o.value, { detail: o.detail, info: o.info, type: "constant" }));
      const ty = Array.isArray(s.type) ? s.type : [s.type];
      if (ty.includes("boolean")) options.push(lit(true), lit(false));
      if (s.nullable || ty.includes("null")) options.push(lit(null));
      if (ty.includes("object")) options.push({ label: "{…}", type: "keyword", apply: "{}", detail: "object" });
      if (ty.includes("array")) {
        const it = unwrap(s.items, schema);
        if (it?.["x-dynamic"]) for (const o of dynOptions(dyn, it["x-dynamic"])) options.push({ label: `[${JSON.stringify(o.value)}]`, type: "constant", apply: `[${JSON.stringify(o.value)}]`, detail: o.detail });
        options.push({ label: "[…]", type: "keyword", apply: "[]", detail: "array" });
      }
    }
    if (!options.length) return null;
    return { from: word.from, to: word.to, options, validFor: /^"?[^"\s,:{}\[\]]*"?$/ };
  };
}

export function createJsonEditor(o: EditorOptions): JsonEditor {
  let schema = o.schema;
  let dyn = o.dynamic;
  const exts: Extension[] = [
    lineNumbers(),
    foldGutter(),
    lintGutter(),
    history(),
    indentOnInput(),
    bracketMatching(),
    closeBrackets(),
    highlightActiveLine(),
    highlightSelectionMatches(),
    search({ top: true }),
    json(),
    syntaxHighlighting(highlight),
    theme,
    EditorView.lineWrapping,
    keymap.of([...closeBracketsKeymap, ...defaultKeymap, ...searchKeymap, ...historyKeymap, ...foldKeymap, ...completionKeymap, indentWithTab]),
    linter(lintSource({ schema: () => schema, extra: o.extraIssues }), { delay: 300 }),
    autocompletion({ override: [completionSource(() => ({ schema, dyn }))], activateOnTyping: true, icons: false }),
    hoverTooltip((view, pos) => {
      if (!schema) return null;
      const node = syntaxTree(view.state).resolveInner(pos, 1);
      const { path } = pathOfNode(view.state, node);
      const s = schemaAt(schema, path);
      let value: unknown;
      if (node.name === "String") value = unq(view.state.sliceDoc(node.from, node.to));
      const dom = describe(s, schema, dyn, value);
      if (!dom) return null;
      const label = document.createElement("div");
      label.textContent = path.length ? path.join(".") : "(document)";
      label.style.cssText = "color:var(--text-muted);font-size:11px;margin-bottom:2px";
      dom.prepend(label);
      return { pos: node.from, end: node.to, above: true, create: () => ({ dom }) };
    }),
    EditorState.readOnly.of(!!o.readOnly),
    EditorView.updateListener.of((u) => {
      if (u.docChanged) o.onChange?.(u.state.doc.toString());
    }),
  ];
  if (o.placeholder) exts.push(placeholder(o.placeholder));
  const view = new EditorView({ parent: o.parent, state: EditorState.create({ doc: o.doc, extensions: exts }) });
  return {
    view,
    get: () => view.state.doc.toString(),
    set(text: string) {
      if (text === view.state.doc.toString()) return;
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
    },
    format() {
      try {
        const pretty = JSON.stringify(JSON.parse(view.state.doc.toString()), null, 2);
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: pretty } });
        return true;
      } catch {
        return false;
      }
    },
    relint: () => forceLinting(view),
    setSchema(s, d) {
      schema = s;
      if (d) dyn = d;
      forceLinting(view);
    },
    destroy: () => view.destroy(),
  };
}
