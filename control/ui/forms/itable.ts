// The launch screen's instance-type table (owner 2026-10-13): every type the
// picked provider lists (GET /api/v1/providers/<p>/instance-types), sortable,
// searchable, filterable and paginated; a click (or Enter / Space) on a
// launchable row selects it. The selection is kept across sort, filter and
// page; the table loads when its provider is picked and on Refresh (which
// reads the provider again: ?refresh=1). Its controls are page controls, not
// fields of the launch document (data-schema-ignore).
// Runpod (owner 2026-10-13): which rows are launchable follows the form's
// weights source (each row's `reasons` per source, recomputed here, no
// refetch), and the table is an ordered multi-pick: a click picks the primary,
// "+" on a row (or Shift+click / Shift+Enter) adds a fallback, and the
// "Selected GPUs" strip shows the order with move / remove buttons.
import { h } from "../dom";
import type { Api } from "../fields";
import "./itable.css";

export interface ITRow {
  id: string;
  name: string;
  spec: string;
  available: boolean | null;
  stock: string | null;
  usd_per_hr: number | null;
  usd_per_gpu_hr: number | null;
  gpu_model: string | null;
  model_family: string | null;
  gpu_count: number;
  gpu: string;
  gpu_family: string | null;
  vram_gb: number | null;
  vram_total_gb: number | null;
  ram_gb: number | null;
  vcpu: number | null;
  storage_type: string | null;
  storage_kind: "fixed" | "elastic" | null;
  storage_gb: number | null;
  storage_min_gb: number | null;
  storage_max_gb: number | null;
  region: string | null;
  cloud: string | null;
  stoppable: boolean | null;
  deploy_s: number | null;
  supported: boolean;
  reason: string | null;
  launch: Record<string, unknown> | null;
  /** Runpod: the data centre has an fv weights volume (null: no data centre). */
  volume_dc?: boolean | null;
  /** Runpod: why not launchable with each weights source (null: launchable with it). */
  reasons?: Record<string, string | null>;
}
/** One entry of the multi-pick's "Selected GPUs" strip (in order). */
export interface ITPick {
  key: string;
  label: string;
  detail?: string;
}
export interface ITView {
  provider: string;
  enabled: boolean;
  reason: string | null;
  at: number | null;
  cached: boolean;
  refresh_limited: number | null;
  live_ok: boolean;
  error: string | null;
  rows: ITRow[];
  counts: { total: number; supported: number; available: number };
}

interface Col {
  key: string;
  label: string;
  num?: boolean;
  only?: string[];
  sort: (r: ITRow) => string | number | null;
  cell: (r: ITRow) => Node | string;
  title?: string;
}
const money = (v: number | null) => (v === null ? "-" : `$${v < 10 ? v.toFixed(2) : v.toFixed(1)}`);
const gbs = (v: number | null) => (v === null ? "-" : `${v % 1 ? v.toFixed(1) : v} GB`);
export const storageSize = (r: ITRow) =>
  r.storage_gb !== null ? `${r.storage_kind === "elastic" ? "default " : ""}${gbs(r.storage_gb)}${r.storage_min_gb !== null || r.storage_max_gb !== null ? ` (${r.storage_min_gb ?? "?"}–${r.storage_max_gb ?? "?"} GB)` : ""}` : r.storage_min_gb !== null || r.storage_max_gb !== null ? `${r.storage_min_gb ?? "?"}–${r.storage_max_gb ?? "?"} GB` : "-";
const availCell = (r: ITRow) =>
  h("span", { class: `it-av ${r.available === true ? "yes" : r.available === false ? "no" : "unk"}` }, r.available === true ? `yes${r.stock ? ` (${r.stock})` : ""}` : r.available === false ? "no" : "?");

const COLS: Col[] = [
  { key: "name", label: "Name", sort: (r) => r.name, cell: (r) => h("span", { class: "mono" }, r.name, r.id !== r.name && r.region ? h("span", { class: "muted small" }, ` @ ${r.region}`) : "") },
  { key: "spec", label: "Spec", sort: (r) => r.spec, cell: (r) => r.spec },
  { key: "available", label: "Available", sort: (r) => (r.available === true ? 2 + ({ High: 0.3, Medium: 0.2, Low: 0.1 }[r.stock || ""] ?? 0) : r.available === false ? 0 : 1), cell: availCell },
  { key: "usd_per_hr", label: "Cost $/hr", num: true, sort: (r) => r.usd_per_hr, cell: (r) => money(r.usd_per_hr) },
  { key: "usd_per_gpu_hr", label: "$/GPU-hr", num: true, sort: (r) => r.usd_per_gpu_hr, cell: (r) => money(r.usd_per_gpu_hr) },
  { key: "gpu_model", label: "Model", sort: (r) => r.gpu_model, cell: (r) => r.gpu_model ?? "-" },
  { key: "model_family", label: "Model family", sort: (r) => r.model_family, cell: (r) => r.model_family ?? "-", title: "The GPU's architecture" },
  { key: "ram_gb", label: "RAM", num: true, sort: (r) => r.ram_gb, cell: (r) => gbs(r.ram_gb), title: "System memory (Runpod: the least a one-GPU pod gets)" },
  { key: "vcpu", label: "vCPU", num: true, sort: (r) => r.vcpu, cell: (r) => (r.vcpu === null ? "-" : String(r.vcpu)) },
  { key: "gpu", label: "GPU", sort: (r) => `${r.gpu_model ?? "~"} ${String(r.gpu_count).padStart(3, "0")}`, cell: (r) => r.gpu },
  { key: "gpu_family", label: "GPU family", sort: (r) => r.gpu_family, cell: (r) => r.gpu_family ?? "-", title: "The GPU's segment: datacenter, workstation, consumer" },
  { key: "vram_gb", label: "VRAM", num: true, sort: (r) => r.vram_total_gb ?? r.vram_gb, cell: (r) => (r.vram_gb === null ? "-" : r.gpu_count > 1 ? `${r.vram_gb} GB ×${r.gpu_count} = ${r.vram_total_gb} GB` : gbs(r.vram_gb)) },
  { key: "storage_type", label: "Storage type", sort: (r) => r.storage_type, cell: (r) => r.storage_type ?? "-" },
  { key: "storage_size", label: "Storage size", num: true, sort: (r) => r.storage_gb ?? r.storage_max_gb, cell: storageSize },
  { key: "region", label: "Region", sort: (r) => r.region, cell: (r) => h("span", { title: r.cloud || "" }, r.region ?? "-", r.cloud ? h("span", { class: "muted small" }, ` ${r.cloud}`) : "") },
  { key: "stoppable", label: "Stoppable", only: ["brev"], sort: (r) => (r.stoppable === null ? null : r.stoppable ? 1 : 0), cell: (r) => (r.stoppable === null ? "-" : r.stoppable ? "yes" : "no") },
  { key: "deploy_s", label: "Deploy", num: true, only: ["brev"], sort: (r) => r.deploy_s, cell: (r) => (r.deploy_s === null ? "-" : `~${Math.max(1, Math.round(r.deploy_s / 60))} min`) },
  { key: "supported", label: "Launchable", sort: (r) => (r.supported ? 1 : 0), cell: (r) => (r.supported ? h("span", { class: "it-ok" }, "yes") : h("span", { class: "it-no", title: r.reason || "" }, "no")) },
];
export const PAGE_SIZES = [10, 25, 50, 100];

export interface InstanceTable {
  el: HTMLElement;
  provider: string;
  /** Loads the table (refresh: the provider is read again, not the cache). Cancels a load in flight. */
  load(refresh?: boolean): Promise<void>;
  /** Cancels a load in flight (the provider was switched). */
  abort(): void;
  /** Re-reads which row the form's value selects (after the form changed). */
  sync(): void;
  view(): ITView | null;
}

/** A row's launchability under a weights source (its `reasons` entry), else the server's (the volume's) answer. */
export function launchable(r: ITRow, weights?: string | null): { supported: boolean; reason: string | null } {
  if (weights && r.reasons && Object.hasOwn(r.reasons, weights)) return { supported: r.reasons[weights] === null, reason: r.reasons[weights] ?? null };
  return { supported: r.supported, reason: r.reason };
}

export function instanceTable(o: {
  api: Api;
  provider: string;
  isSelected: (r: ITRow) => boolean;
  /** append: add the row as a fallback (the multi-pick) instead of making it the only pick. */
  onSelect: (r: ITRow | null, how?: { append: boolean }) => void;
  noun?: string;
  /** The form's weights source: the rows' launchable flags follow it (Runpod). */
  weights?: () => string | null | undefined;
  /** The ordered multi-pick (Runpod): the selected entries in order, and how to reorder / remove one. */
  multi?: { items: () => ITPick[]; order: (r: ITRow) => number | null; move: (i: number, d: -1 | 1) => void; remove: (i: number) => void };
  /** Called after each load (the form redraws what the listing feeds: the data centres' options). */
  onLoad?: () => void;
}): InstanceTable {
  const p = o.provider;
  const noun = o.noun || "instance type";
  let view: ITView | null = null;
  let error: string | null = null;
  let loading = false;
  let ac: AbortController | null = null;
  let selectedId: string | null = null;
  let sort: { key: string; dir: 1 | -1 } | null = null;
  let page = 0;
  let pageSize = 25;
  const f = { q: "", available: "", family: "", gpuFamily: "", region: "", minVram: "", maxCost: "", stoppable: "", launchable: "" };
  const ok = (r: ITRow) => launchable(r, o.weights?.());
  const cols = COLS.filter((c) => !c.only || c.only.includes(p)).map((c) =>
    c.key === "supported" ? { ...c, sort: (r: ITRow) => (ok(r).supported ? 1 : 0), cell: (r: ITRow) => (ok(r).supported ? h("span", { class: "it-ok" }, "yes") : h("span", { class: "it-no", title: ok(r).reason || "" }, "no")) } : c,
  );
  if (o.multi)
    cols.push({
      key: "pick",
      label: "Fallback",
      title: "Add the row's GPU type (and data centre) after the ones picked: a fallback, tried in order",
      sort: (r) => o.multi!.order(r),
      cell: (r) =>
        ok(r).supported
          ? h(
              "button",
              {
                type: "button",
                class: "ghost small it-add",
                "data-it-add": r.id,
                "aria-label": `add ${r.name}${r.region ? ` in ${r.region}` : ""} as a fallback`,
                title: "add as a fallback (or Shift+click / Shift+Enter on the row)",
                onclick: (e: Event) => (e.stopPropagation(), choose(r, true)),
                onkeydown: (e: KeyboardEvent) => e.stopPropagation(),
              },
              "+ fallback",
            )
          : "",
    });
  const strip = h("div", { class: "it-strip small", "data-it-strip": "", role: "group", "aria-label": "Selected GPUs (in order)" });

  const el = h("div", { class: "it", "data-instance-table": p, "data-schema-ignore": "" });
  const status = h("div", { class: "it-status small muted", role: "status", "aria-live": "polite" });
  const picked = h("div", { class: "it-picked small", "data-it-selected": "" });
  const tableHost = h("div", { class: "tablewrap it-wrap" });
  const pager = h("div", { class: "it-pager row small" });
  const refreshBtn = h("button", { type: "button", class: "it-refresh", "data-it-refresh": "", onclick: () => void load(true), title: "Read the provider's list again (not the cache)" }, "Refresh");

  const ctl = (k: keyof typeof f, label: string, input: HTMLInputElement | HTMLSelectElement) => {
    input.setAttribute("data-it-filter", k);
    input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => {
      f[k] = input.value;
      page = 0;
      render();
    });
    return h("label", { class: "it-f" }, label, input);
  };
  const sel = (opts: [string, string][]) => h("select", {}, ...opts.map(([v, t]) => h("option", { value: v }, t)));
  const filters = h("div", { class: "it-filters" });
  const search = h("input", { type: "search", placeholder: `search ${noun}s…`, "aria-label": `Search ${noun}s`, "data-it-search": "" });
  search.addEventListener("input", () => {
    f.q = search.value;
    page = 0;
    render();
  });
  // Filters whose options come from the rows (rebuilt after each load).
  const famSel = sel([["", "any"]]);
  const gfamSel = sel([["", "any"]]);
  const regionSel = sel([["", "any"]]);
  function drawFilters() {
    const opts = (s: HTMLSelectElement, vals: (string | null)[], cur: string) => {
      const u = [...new Set(vals.filter((v): v is string => !!v))].sort();
      s.replaceChildren(h("option", { value: "" }, "any"), ...u.map((v) => h("option", { value: v }, v)));
      s.value = u.includes(cur) ? cur : "";
    };
    const rows = view?.rows || [];
    opts(famSel, rows.map((r) => r.model_family), f.family);
    opts(gfamSel, rows.map((r) => r.gpu_family), f.gpuFamily);
    opts(regionSel, rows.map((r) => r.region), f.region);
    f.family = famSel.value;
    f.gpuFamily = gfamSel.value;
    f.region = regionSel.value;
  }
  const num = (ph: string) => h("input", { type: "number", min: "0", step: "any", placeholder: ph, class: "it-num" });
  filters.append(
    h("label", { class: "it-f it-search" }, "Search", search),
    ctl("available", "Available", sel([["", "any"], ["yes", "yes"], ["no", "no / unknown"]])),
    ctl("launchable", "Launchable", sel([["", "any"], ["yes", "yes"], ["no", "no"]])),
    ctl("family", "Model family", famSel),
    ctl("gpuFamily", "GPU family", gfamSel),
    ctl("region", "Region", regionSel),
    ctl("minVram", "Min VRAM (GB)", num("any")),
    ctl("maxCost", "Max $/hr", num("any")),
    ...(p === "brev" ? [ctl("stoppable", "Stoppable", sel([["", "any"], ["yes", "yes"], ["no", "no"]]))] : []),
  );

  function rowsShown(): ITRow[] {
    const q = f.q.trim().toLowerCase();
    const minV = f.minVram === "" ? null : Number(f.minVram);
    const maxC = f.maxCost === "" ? null : Number(f.maxCost);
    let rows = (view?.rows || []).filter(
      (r) =>
        (!q || [r.name, r.spec, r.gpu_model, r.model_family, r.gpu_family, r.region, r.cloud, r.storage_type, ok(r).reason].some((x) => x && x.toLowerCase().includes(q))) &&
        (!f.available || (f.available === "yes" ? r.available === true : r.available !== true)) &&
        (!f.launchable || (f.launchable === "yes") === ok(r).supported) &&
        (!f.family || r.model_family === f.family) &&
        (!f.gpuFamily || r.gpu_family === f.gpuFamily) &&
        (!f.region || r.region === f.region) &&
        (minV === null || !Number.isFinite(minV) || (r.vram_total_gb ?? r.vram_gb ?? -1) >= minV) &&
        (maxC === null || !Number.isFinite(maxC) || (r.usd_per_hr !== null && r.usd_per_hr <= maxC)) &&
        (!f.stoppable || (f.stoppable === "yes") === (r.stoppable === true)),
    );
    if (sort) {
      const c = cols.find((x) => x.key === sort!.key)!;
      const d = sort.dir;
      rows = rows
        .map((r, i) => ({ r, i, v: c.sort(r) }))
        .sort((a, b) => {
          // Unknown values last whichever the direction; ties keep the server's order.
          if (a.v === null || a.v === undefined) return b.v === null || b.v === undefined ? a.i - b.i : 1;
          if (b.v === null || b.v === undefined) return -1;
          const x = typeof a.v === "number" && typeof b.v === "number" ? a.v - b.v : String(a.v).localeCompare(String(b.v), undefined, { numeric: true });
          return x ? x * d : a.i - b.i;
        })
        .map((x) => x.r);
    }
    return rows;
  }
  const selectedRow = () => (view?.rows || []).find((r) => r.id === selectedId) || null;
  function choose(r: ITRow, append = false) {
    if (!ok(r).supported) return;
    if (!append || selectedId === null) selectedId = r.id;
    o.onSelect(r, { append });
    render();
  }
  function drawStrip() {
    if (!o.multi) return;
    const items = o.multi.items();
    const btn = (label: string, aria: string, attr: string, on: () => void, off = false) => h("button", { type: "button", class: "ghost small", "aria-label": aria, title: aria, [attr]: "", disabled: off, onclick: () => (on(), render()) }, label);
    strip.replaceChildren(
      h("b", {}, "Selected GPUs "),
      ...(items.length
        ? [
            h(
              "ol",
              { class: "it-picks" },
              ...items.map((x, i) =>
                h(
                  "li",
                  { "data-it-pick": x.key },
                  h("span", { class: "it-ord" }, String(i + 1)),
                  " ",
                  h("span", { class: "mono" }, x.label),
                  x.detail ? h("span", { class: "muted" }, ` ${x.detail}`) : "",
                  " ",
                  btn("↑", `move ${x.label} up`, "data-it-up", () => o.multi!.move(i, -1), i === 0),
                  btn("↓", `move ${x.label} down`, "data-it-down", () => o.multi!.move(i, 1), i === items.length - 1),
                  btn("×", `remove ${x.label}`, "data-it-remove", () => o.multi!.remove(i)),
                ),
              ),
            ),
          ]
        : [h("span", { class: "muted" }, `none: click a launchable row (the primary), then "+ fallback" on others`)]),
    );
  }
  function render() {
    // The selected row: kept while it still matches the form; else the first row the form's value selects.
    const cur = selectedRow();
    const ord = (r: ITRow) => o.multi?.order(r) ?? 0;
    const primary = () => (view?.rows || []).filter((r) => ok(r).supported && o.isSelected(r)).sort((a, b) => ord(a) - ord(b))[0]?.id ?? null;
    if (!cur || !o.isSelected(cur) || !ok(cur).supported || ord(cur) > 1) selectedId = primary();
    drawStrip();
    const s = selectedRow();
    picked.replaceChildren(
      ...(s
        ? [h("b", {}, "Selected: "), h("span", { class: "mono" }, s.name), ` · ${s.spec} · ${money(s.usd_per_hr)}/hr${s.region ? ` · ${s.region}` : ""} `, h("button", { type: "button", class: "ghost small", "data-it-clear": "", onclick: () => ((selectedId = null), o.onSelect(null), render()) }, "Clear")]
        : [h("span", { class: "muted" }, view ? `No ${noun} selected: click a launchable row.` : "")]),
    );
    if (!view) {
      tableHost.replaceChildren(h("div", { class: "it-empty muted small", "data-it-state": loading ? "loading" : error ? "error" : "idle" }, loading ? `loading ${p}'s ${noun}s…` : error ? `could not load ${p}'s ${noun}s: ${error}` : ""));
      pager.replaceChildren();
      return;
    }
    const all = rowsShown();
    const pages = Math.max(1, Math.ceil(all.length / pageSize));
    page = Math.min(page, pages - 1);
    const shown = all.slice(page * pageSize, page * pageSize + pageSize);
    const th = (c: Col) => {
      const dir = sort?.key === c.key ? sort.dir : 0;
      return h(
        "th",
        { class: c.num ? "num" : "", "aria-sort": dir === 1 ? "ascending" : dir === -1 ? "descending" : "none", "data-col": c.key, title: c.title },
        h(
          "button",
          {
            type: "button",
            class: "it-sort",
            onclick: () => {
              sort = !sort || sort.key !== c.key ? { key: c.key, dir: 1 } : sort.dir === 1 ? { key: c.key, dir: -1 } : null;
              page = 0;
              render();
              (tableHost.querySelector(`th[data-col="${c.key}"] button`) as HTMLElement | null)?.focus();
            },
          },
          c.label,
          h("span", { class: "it-arrow", "aria-hidden": "true" }, dir === 1 ? " ▲" : dir === -1 ? " ▼" : ""),
        ),
      );
    };
    const tr = (r: ITRow) => {
      const on = r.id === selectedId;
      const can = ok(r).supported;
      const n = o.multi && can && o.isSelected(r) ? o.multi.order(r) : null;
      const row = h(
        "tr",
        {
          class: `${can ? "it-can" : "it-cannot"}${on ? " it-on" : ""}${n ? " it-picked" : ""}`,
          "data-row": r.id,
          "aria-selected": String(on || !!n),
          ...(n ? { "data-it-order": String(n) } : {}),
          "aria-disabled": can ? undefined : "true",
          tabindex: can ? "0" : "-1",
          title: can ? `select ${r.name}${o.multi ? " (Shift: add as a fallback)" : ""}` : `not launchable: ${ok(r).reason}`,
          onclick: (e: MouseEvent) => choose(r, !!o.multi && e.shiftKey),
          onkeydown: (e: KeyboardEvent) => {
            if (e.key === "Enter" || e.key === " ") (e.preventDefault(), choose(r, !!o.multi && e.shiftKey));
            else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
              e.preventDefault();
              const sib = (e.key === "ArrowDown" ? row.nextElementSibling : row.previousElementSibling) as HTMLElement | null;
              sib?.focus();
            }
          },
        },
        ...cols.map((c) => h("td", { class: c.num ? "num" : c.key === "spec" || c.key === "storage_type" ? "wrap" : "" }, c.key === "name" && n ? h("span", { class: "it-ord", title: `pick ${n}` }, String(n)) : "", c.cell(r))),
      );
      return row;
    };
    tableHost.replaceChildren(
      all.length
        ? h(
            "table",
            { class: "it-table", role: "grid", "aria-label": `${p} ${noun}s`, "aria-rowcount": String(all.length + 1), "aria-busy": loading ? "true" : "false" },
            h("thead", {}, h("tr", {}, ...cols.map(th))),
            h("tbody", {}, ...shown.map(tr)),
          )
        : h("div", { class: "it-empty muted small", "data-it-state": "empty" }, view.rows.length ? `No ${noun} matches the search and filters.` : `${p} lists no ${noun}s.`),
    );
    const sizeSel = h("select", { "aria-label": "Rows per page", "data-it-pagesize": "" }, ...PAGE_SIZES.map((n) => h("option", { value: String(n), selected: n === pageSize }, `${n} / page`)));
    sizeSel.value = String(pageSize);
    sizeSel.addEventListener("change", () => {
      pageSize = Number(sizeSel.value) || 25;
      page = 0;
      render();
    });
    pager.replaceChildren(
      h("button", { type: "button", "data-it-prev": "", disabled: page === 0, onclick: () => (page--, render()), "aria-label": "Previous page" }, "‹ Prev"),
      h("span", { "data-it-page": "" }, `page ${page + 1} of ${pages}`),
      h("button", { type: "button", "data-it-next": "", disabled: page >= pages - 1, onclick: () => (page++, render()), "aria-label": "Next page" }, "Next ›"),
      sizeSel,
      h("span", { class: "muted", "data-it-count": "" }, `${all.length ? page * pageSize + 1 : 0}–${page * pageSize + shown.length} of ${all.length}${all.length !== view.rows.length ? ` (filtered from ${view.rows.length})` : ""} · ${view.rows.filter((r) => ok(r).supported).length} launchable · ${view.counts.available} available`),
    );
  }
  function statusLine() {
    const bits: (Node | string)[] = [];
    if (loading) bits.push(h("span", { class: "spin", "aria-hidden": "true" }), view ? ` refreshing ${p}…` : ` loading ${p}…`);
    else if (view) {
      bits.push(view.at ? `updated ${new Date(view.at).toLocaleTimeString()}${view.cached ? " (cached)" : ""}` : "not loaded from the provider");
      if (view.refresh_limited) bits.push(` · refreshed recently: the cached list (again in ${view.refresh_limited} s)`);
      if (!view.enabled && view.reason) bits.push(` · ${view.reason}`);
      if (view.error) bits.push(h("span", { class: "it-err" }, ` · ${view.error}`));
    } else if (error) bits.push(h("span", { class: "it-err" }, `could not load: ${error}`));
    status.replaceChildren(...bits);
    status.toggleAttribute("data-loading", loading);
    if (view?.at) status.setAttribute("data-at", String(view.at));
  }
  async function load(refresh = false) {
    ac?.abort();
    const my = (ac = new AbortController());
    loading = true;
    error = null;
    refreshBtn.disabled = true;
    statusLine();
    render();
    try {
      const v: ITView = await o.api(`/api/v1/providers/${encodeURIComponent(p)}/instance-types${refresh ? "?refresh=1" : ""}`, { signal: my.signal });
      if (my.signal.aborted) return;
      view = v;
      drawFilters();
      queueMicrotask(() => o.onLoad?.());
    } catch (e) {
      if (my.signal.aborted) return;
      error = (e as Error).message;
    }
    if (ac === my) ac = null;
    loading = false;
    refreshBtn.disabled = false;
    statusLine();
    render();
  }
  el.append(h("div", { class: "it-head row" }, h("b", { class: "it-title" }, `${p === "gmi" ? "GMI Cloud products" : p === "brev" ? "NVIDIA Brev instance types" : "Runpod GPU types × data centres"}`), refreshBtn, status), filters, picked, ...(o.multi ? [strip] : []), tableHost, pager);
  statusLine();
  render();
  return {
    el,
    provider: p,
    load,
    abort() {
      if (!ac) return;
      ac.abort();
      ac = null;
      loading = false;
      refreshBtn.disabled = false;
      statusLine();
      render();
    },
    sync: () => render(),
    view: () => view,
  };
}
