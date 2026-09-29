// A line diff (LCS) of two pretty-printed JSON documents, and a collapsible
// JSON tree viewer with search and copy-path / copy-value.
import { pathString, type Path } from "./schema";

export interface DiffLine {
  op: " " | "+" | "-";
  text: string;
}
export function diffLines(a: string, b: string): DiffLine[] {
  const x = a.split("\n");
  const y = b.split("\n");
  // Trim the common head and tail first (most edits are small).
  let s = 0;
  while (s < x.length && s < y.length && x[s] === y[s]) s++;
  let e = 0;
  while (e < x.length - s && e < y.length - s && x[x.length - 1 - e] === y[y.length - 1 - e]) e++;
  const xm = x.slice(s, x.length - e);
  const ym = y.slice(s, y.length - e);
  const n = xm.length;
  const m = ym.length;
  const out: DiffLine[] = x.slice(0, s).map((t) => ({ op: " ", text: t }));
  if (n * m > 4_000_000) {
    out.push(...xm.map((t) => ({ op: "-" as const, text: t })), ...ym.map((t) => ({ op: "+" as const, text: t })));
  } else {
    const L = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i]![j] = xm[i] === ym[j] ? L[i + 1]![j + 1]! + 1 : Math.max(L[i + 1]![j]!, L[i]![j + 1]!);
    let i = 0;
    let j = 0;
    while (i < n && j < m) {
      if (xm[i] === ym[j]) out.push({ op: " ", text: xm[i++]! }), j++;
      else if (L[i + 1]![j]! >= L[i]![j + 1]!) out.push({ op: "-", text: xm[i++]! });
      else out.push({ op: "+", text: ym[j++]! });
    }
    while (i < n) out.push({ op: "-", text: xm[i++]! });
    while (j < m) out.push({ op: "+", text: ym[j++]! });
  }
  out.push(...x.slice(x.length - e).map((t) => ({ op: " " as const, text: t })));
  return out;
}
/** A diff as DOM, unchanged runs longer than 6 lines collapsed. */
export function renderDiff(a: unknown, b: unknown): HTMLElement {
  const lines = diffLines(JSON.stringify(a, null, 2) ?? "", JSON.stringify(b, null, 2) ?? "");
  const box = document.createElement("div");
  box.className = "fv-diff";
  const changed = lines.some((l) => l.op !== " ");
  if (!changed) {
    box.textContent = "No changes.";
    box.classList.add("muted");
    return box;
  }
  let run: DiffLine[] = [];
  const flush = (last: boolean) => {
    const keep = 3;
    if (run.length > keep * 2 + 1) {
      for (const l of run.slice(0, box.childElementCount ? keep : 0)) box.append(line(l));
      const more = document.createElement("div");
      more.className = "fv-diff-skip";
      more.textContent = `⋯ ${run.length - (box.childElementCount ? keep : 0) - (last ? 0 : keep)} unchanged lines`;
      box.append(more);
      if (!last) for (const l of run.slice(-keep)) box.append(line(l));
    } else for (const l of run) box.append(line(l));
    run = [];
  };
  const line = (l: DiffLine) => {
    const d = document.createElement("div");
    d.className = l.op === "+" ? "add" : l.op === "-" ? "del" : "ctx";
    d.textContent = `${l.op} ${l.text}`;
    return d;
  };
  for (const l of lines) {
    if (l.op === " ") run.push(l);
    else {
      flush(false);
      box.append(line(l));
    }
  }
  flush(true);
  return box;
}

/** A read-only JSON tree: collapsible, searchable, copy path / value. */
export function renderTree(value: unknown, opts: { open?: number; title?: string } = {}): HTMLElement {
  const box = document.createElement("div");
  box.className = "fv-tree";
  const bar = document.createElement("div");
  bar.className = "row";
  const q = document.createElement("input");
  q.type = "search";
  q.placeholder = "search keys and values";
  q.setAttribute("aria-label", "search");
  const status = document.createElement("span");
  status.className = "muted small";
  const expand = document.createElement("button");
  expand.type = "button";
  expand.textContent = "expand all";
  bar.append(q, expand, status);
  const body = document.createElement("div");
  box.append(bar, body);
  const copy = async (text: string, what: string) => {
    try {
      await navigator.clipboard.writeText(text);
      status.textContent = `copied ${what}`;
    } catch {
      status.textContent = text.length < 200 ? text : `${what}: clipboard unavailable`;
    }
  };
  const openDepth = opts.open ?? 1;
  function nodeFor(key: string | number | null, v: unknown, path: Path, depth: number): HTMLElement {
    const isObj = v !== null && typeof v === "object";
    const row = document.createElement("div");
    row.className = "fv-tnode";
    row.dataset.path = pathString(path);
    const head = document.createElement(isObj ? "summary" : "div");
    head.className = "fv-thead";
    if (key !== null) {
      const k = document.createElement("span");
      k.className = "fv-tkey";
      k.textContent = typeof key === "number" ? `[${key}]` : key;
      head.append(k, document.createTextNode(": "));
    }
    const val = document.createElement("span");
    val.className = `fv-tval t-${v === null ? "null" : Array.isArray(v) ? "array" : typeof v}`;
    val.textContent = isObj ? (Array.isArray(v) ? `[${v.length}]` : `{${Object.keys(v as object).length}}`) : JSON.stringify(v);
    head.append(val);
    const btnP = document.createElement("button");
    btnP.type = "button";
    btnP.className = "fv-copy";
    btnP.textContent = "path";
    btnP.title = "copy path";
    btnP.addEventListener("click", (e) => { e.preventDefault(); copy(pathString(path), "path"); });
    const btnV = document.createElement("button");
    btnV.type = "button";
    btnV.className = "fv-copy";
    btnV.textContent = "value";
    btnV.title = "copy value";
    btnV.addEventListener("click", (e) => { e.preventDefault(); copy(typeof v === "string" ? v : JSON.stringify(v, null, 2), "value"); });
    head.append(btnP, btnV);
    if (!isObj) {
      row.append(head);
      return row;
    }
    const det = document.createElement("details");
    det.open = depth < openDepth;
    det.append(head);
    const kids = document.createElement("div");
    kids.className = "fv-tkids";
    // Children render lazily on first open (large snapshots stay fast).
    const fill = () => {
      if (kids.childElementCount) return;
      const entries: [string | number, unknown][] = Array.isArray(v) ? v.map((x, i) => [i, x]) : Object.entries(v as object);
      for (const [k, x] of entries) kids.append(nodeFor(k, x, [...path, k], depth + 1));
    };
    if (det.open) fill();
    det.addEventListener("toggle", () => det.open && fill());
    det.append(kids);
    row.append(det);
    return row;
  }
  body.append(nodeFor(opts.title ?? null, value, [], 0));
  expand.addEventListener("click", () => {
    const openAll = (el: Element) => el.querySelectorAll("details").forEach((d) => { (d as HTMLDetailsElement).open = true; d.dispatchEvent(new Event("toggle")); });
    for (let i = 0; i < 8; i++) openAll(body);
  });
  q.addEventListener("input", () => {
    const needle = q.value.trim().toLowerCase();
    let hits = 0;
    if (needle) {
      for (let i = 0; i < 8; i++) body.querySelectorAll("details").forEach((d) => { if (!(d as HTMLDetailsElement).open) { (d as HTMLDetailsElement).open = true; d.dispatchEvent(new Event("toggle")); } });
    }
    body.querySelectorAll<HTMLElement>(".fv-thead").forEach((h) => {
      const m = !!needle && (h.textContent || "").toLowerCase().includes(needle);
      h.classList.toggle("hit", m);
      if (m) hits++;
    });
    status.textContent = needle ? `${hits} match(es)` : "";
    body.querySelector(".fv-thead.hit")?.scrollIntoView({ block: "nearest" });
  });
  return box;
}
