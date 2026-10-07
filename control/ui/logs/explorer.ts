// The log explorer (Logs page): every log fv-control keeps in one virtualized
// list over /api/logs/query, with the filters in the URL. See
// docs/control/README.md "Logs" for the keyboard shortcuts and the API.
import { badge, copyText, download, h, store, type Api } from "../dom";
import {
  buildRows,
  colorSlot,
  DEFAULT_VIEW,
  describeRange,
  encodeCursor,
  formatLine,
  iso,
  LEVELS,
  matcher,
  matchRanges,
  mergeLines,
  nextMatch,
  parseJump,
  podKey,
  RANGES,
  searchable,
  selectedRange,
  serverParams,
  shortTime,
  SOURCE_LABEL,
  SOURCES,
  viewFromParams,
  viewToParams,
  type Level,
  type Row,
  type Source,
  type View,
  type XLine,
} from "./model";

export interface MountOptions {
  api: Api;
  toast: (m: string) => void;
  /** Registers a function the router calls when the page is left. */
  onCleanup: (fn: () => void) => void;
}
const ROW_H = 22;
const PAGE = 300;
const MAX_LINES = 50_000;
const OVERSCAN = 30;
const LIVE_MS = 2000;

interface Bookmark {
  uid: string | null;
  ts: number | null;
  level?: string;
  title: string;
  hash: string;
  saved_at: number;
}

export async function mountLogs(host: HTMLElement, o: MountOptions) {
  const { api, toast } = o;
  let view: View = viewFromParams(new URLSearchParams(location.hash.split("?")[1] || ""));
  // ---- data
  let buf: XLine[] = [];
  const seen = new Set<string>();
  let rows: Row[] = [];
  let olderCursor: string | null = null;
  let newerCursor: string | null = null;
  let atHead = true;
  let partial: string | null = null;
  let seq = 0;
  let loadingOlder = false;
  let loadingNewer = false;
  let tail: string | null = null;
  let paused = false;
  let pending: XLine[] = [];
  let unseen = 0;
  let anchor = view.sel;
  let focus = view.sel;
  const extra = new Set<string>();
  let detailOpen = !!view.sel;
  let pins: XLine[] = (() => {
    try {
      return JSON.parse(sessionStorage.getItem("fvc-log-pins") || "[]");
    } catch {
      return [];
    }
  })();
  let bookmarks: Bookmark[] = store.get("fvc-log-bookmarks", []);
  let lastError = "";
  /** The view is at the newest edge: new lines and resizes keep it there. */
  let stuck = true;
  let findTest: ((l: XLine) => boolean) | null = null;

  // ---- pickers' data (loaded once)
  const [clustersR, podsR] = await Promise.all([api("/api/clusters").catch(() => ({ clusters: [] })), api("/api/pods?all=1").catch(() => ({ pods: [] }))]);
  const clusters: { id: string; name: string; pools: string[] }[] = (clustersR.clusters || []).map((c: any) => ({ id: c.id, name: c.name, pools: (c.spec?.pools || []).map((p: any) => p.id) }));
  const podsInfo = new Map<string, { name: string; owner: string; cluster_id: string | null }>();
  for (const p of podsR.pods || []) podsInfo.set(p.pod_id, { name: p.name || p.pod_id, owner: p.owner, cluster_id: p.cluster_id ?? null });
  let facetPods: any[] = [];

  // ---- DOM
  const el = {
    search: h("input", { type: "search", "data-path": "q", class: "lx-search", placeholder: "Search messages and fields  ( / )", "aria-label": "Search", value: view.q, spellcheck: "false" }),
    re: h("button", { type: "button", class: "lx-tog", title: "Regular expression", "aria-pressed": String(view.re) }, ".*"),
    cs: h("button", { type: "button", class: "lx-tog", title: "Case sensitive", "aria-pressed": String(view.cs) }, "Aa"),
    mode: h("select", { "data-schema-ignore": "", "aria-label": "Search mode", title: "Filter: only matching lines (server-side). Find: every line, matches highlighted; n / N jump." }, h("option", { value: "filter" }, "filter"), h("option", { value: "find" }, "find")),
    cluster: h("select", { "data-path": "cluster", "aria-label": "Cluster" }),
    pool: h("select", { "data-path": "pool", "aria-label": "Pool" }),
    pod: h("select", { "data-path": "pod", "aria-label": "Pod" }),
    sources: h("div", { class: "lx-chips", role: "group", "aria-label": "Sources" }),
    levels: h("div", { class: "lx-chips", role: "group", "aria-label": "Levels" }),
    ranges: h("div", { class: "lx-chips", role: "group", "aria-label": "Time range" }),
    from: h("input", { "data-path": "from", class: "lx-time", placeholder: "from (UTC)", "aria-label": "From", title: "UTC: 2026-10-06 12:00, 12:00 (today), unix ms, or relative: 15m, 2h, 7d" }),
    to: h("input", { "data-path": "to", class: "lx-time", placeholder: "to (now)", "aria-label": "To", title: "UTC; empty: now (live tail possible)" }),
    order: h("select", { "data-path": "order", "aria-label": "Time order" }, h("option", { value: "asc" }, "oldest first"), h("option", { value: "desc" }, "newest first")),
    sort: h("select", { "data-schema-ignore": "", "aria-label": "Sort" }, h("option", { value: "time" }, "sort: time"), h("option", { value: "level" }, "sort: level"), h("option", { value: "source" }, "sort: source"), h("option", { value: "pod" }, "sort: pod")),
    group: h("select", { "data-schema-ignore": "", "aria-label": "Group" }, h("option", { value: "none" }, "no grouping"), h("option", { value: "pod" }, "group: pod"), h("option", { value: "level" }, "group: level")),
    live: h("button", { type: "button", class: "lx-live", title: "Live tail (l); pause / resume: space" }),
    jump: h("input", { "data-schema-ignore": "", class: "lx-time", placeholder: "jump to time ( t )", "aria-label": "Jump to time", title: "UTC: 2026-10-06 12:00, 12:00 (today), or 10m (ago)" }),
    ctx: h("input", { type: "number", "data-path": "ctx", min: "0", max: "200", step: "1", class: "lx-ctx", "aria-label": "Context lines", title: "Lines of context before and after (c)", value: String(view.ctx) }),
    hist: h("div", { class: "lx-hist", "aria-label": "Lines over time" }),
    pins: h("div", { class: "lx-pins" }),
    list: h("div", { class: "lx-list", tabindex: "0", role: "listbox", "aria-label": "Log lines", "aria-multiselectable": "true" }),
    spacer: h("div", { class: "lx-spacer" }),
    win: h("div", { class: "lx-win" }),
    detail: h("aside", { class: "lx-detail", "aria-label": "Line details" }),
    status: h("div", { class: "lx-status small", role: "status" }),
    newPill: h("button", { type: "button", class: "lx-newpill", hidden: true }),
    bmMenu: h("details", { class: "lx-menu" }),
    dlMenu: h("details", { class: "lx-menu" }),
    help: h("dialog", { class: "lx-help" }),
  };
  el.spacer.append(el.win);
  el.list.append(el.spacer);
  const main = h("div", { class: "lx-main" }, h("div", { class: "lx-listwrap" }, el.list, el.newPill), el.detail);
  const bar1 = h("div", { class: "lx-bar" }, el.sources, h("span", { class: "lx-sep" }), el.cluster, el.pool, el.pod, h("button", { type: "button", title: "Live tail of the chosen pod only (f on a line follows its pod)", onclick: () => followPod(view.pod) }, "Follow pod"));
  const bar2 = h("div", { class: "lx-bar" }, h("div", { class: "lx-searchbox" }, el.search, el.re, el.cs), el.mode, el.levels, h("span", { class: "lx-sep" }), el.ranges, el.from, h("span", { class: "muted" }, "→"), el.to);
  const bar3 = h(
    "div",
    { class: "lx-bar" },
    el.order,
    el.sort,
    el.group,
    h("label", { class: "lx-inline" }, "context ±", el.ctx),
    el.jump,
    el.live,
    h("span", { class: "spacer" }),
    el.bmMenu,
    el.dlMenu,
    h("button", { type: "button", title: "Copy a link to this view", onclick: () => copyText(location.href).then(() => toast("link copied")) }, "Copy link"),
    h("button", { type: "button", title: "Keyboard shortcuts (?)", onclick: () => el.help.showModal() }, "?"),
  );
  host.replaceChildren(h("div", { class: "lx", "data-schema-form": "log-query" }, h("div", { class: "lx-head" }, h("h1", {}, "Logs"), el.status), bar1, bar2, bar3, el.hist, el.pins, main), el.help);
  el.help.append(
    h("h2", {}, "Keyboard"),
    h(
      "dl",
      { class: "kv" },
      ...(
        [
          ["j / k, ↓ / ↑", "next / previous line (shift: extend the selection)"],
          ["g / G", "first / last line"],
          ["/", "search"],
          ["n / N", "next / previous match (no search: next warn or error)"],
          ["Enter", "details of the line"],
          ["c", "context around the line"],
          ["p / b", "pin / bookmark the line"],
          ["f", "follow the line's pod (live)"],
          ["t", "jump to a time"],
          ["l / space", "live tail on / off; pause and resume"],
          ["Ctrl+C", "copy the selected lines"],
          ["Esc", "close details, clear the selection"],
          ["mouse", "click: select; shift-click or drag: a range; ctrl-click: add a line"],
        ] as const
      ).flatMap(([k, v]) => [h("dt", {}, h("kbd", {}, k)), h("dd", {}, v)]),
    ),
    h("div", { class: "row", style: "margin-top:10px" }, h("button", { type: "button", onclick: () => el.help.close() }, "Close")),
  );

  // ---- controls
  function drawControls() {
    el.sources.replaceChildren(
      ...SOURCES.map((s) => {
        const needsPod = s === "runpod" || s === "archive";
        const on = view.src.includes(s);
        return h(
          "button",
          { type: "button", class: `lx-chip src-${s}`, "aria-pressed": String(on), disabled: needsPod && !view.pod, title: needsPod && !view.pod ? "choose a pod first" : s === "archive" ? "R2 archive: lines older than the 24 h tail (one pod)" : "" },
          SOURCE_LABEL[s],
        );
      }),
    );
    [...el.sources.children].forEach((b, i) =>
      b.addEventListener("click", () => {
        const s = SOURCES[i]!;
        const src = view.src.includes(s) ? view.src.filter((x) => x !== s) : [...view.src, s];
        if (!src.length) return toast("at least one source");
        update({ src });
      }),
    );
    el.levels.replaceChildren(
      ...LEVELS.map((l) => {
        const b = h("button", { type: "button", class: `lx-chip lvchip-${l}`, "aria-pressed": String(view.lv.includes(l)), title: "click: toggle; shift-click: this level and above" }, l);
        b.addEventListener("click", (ev) => {
          let lv: Level[];
          if ((ev as MouseEvent).shiftKey) lv = LEVELS.slice(LEVELS.indexOf(l)) as Level[];
          else lv = view.lv.includes(l) ? view.lv.filter((x) => x !== l) : (LEVELS.filter((x) => view.lv.includes(x) || x === l) as Level[]);
          if (!lv.length) return toast("at least one level");
          update({ lv });
        });
        return b;
      }),
    );
    el.ranges.replaceChildren(
      ...RANGES.map((r) => {
        const b = h("button", { type: "button", class: "lx-chip", "aria-pressed": String(view.from === r && !view.to) }, r);
        b.addEventListener("click", () => update({ from: r, to: "", at: null }));
        return b;
      }),
    );
    el.from.value = RANGES.includes(view.from) ? "" : view.from;
    el.to.value = view.to;
    el.mode.value = view.mode;
    el.order.value = view.order;
    el.sort.value = view.sort;
    el.group.value = view.group;
    el.re.setAttribute("aria-pressed", String(view.re));
    el.cs.setAttribute("aria-pressed", String(view.cs));
    el.cluster.replaceChildren(h("option", { value: "" }, "all clusters"), ...clusters.map((c) => h("option", { value: c.id }, c.name)));
    el.cluster.value = view.cluster;
    const pools = view.cluster ? clusters.find((c) => c.id === view.cluster)?.pools || [] : [...new Set(clusters.flatMap((c) => c.pools))].sort();
    el.pool.replaceChildren(h("option", { value: "" }, "all pools"), ...pools.map((p) => h("option", { value: p }, p)));
    if (view.pool && !pools.includes(view.pool)) el.pool.append(h("option", { value: view.pool }, view.pool));
    el.pool.value = view.pool;
    drawPodPicker();
    el.live.textContent = view.live ? (paused ? `▶ Resume${pending.length ? ` (${pending.length})` : ""}` : "❚❚ Pause") : "● Live";
    el.live.classList.toggle("on", view.live && !paused);
    el.live.title = view.to ? "Live tail needs an open time range (no 'to')" : "Live tail (l); pause / resume: space";
    drawMenus();
  }
  function drawPodPicker() {
    const opts = new Map<string, string>();
    for (const f of facetPods) {
      if (view.cluster && f.cluster_id !== view.cluster) continue;
      if (view.pool && f.pool !== view.pool) continue;
      const nm = f.name || podsInfo.get(f.pod_id)?.name || f.pod_id;
      opts.set(f.pod_id, `${nm}${f.pool ? ` · ${f.pool}` : ""} · ${f.n} lines${f.errors ? ` · ${f.errors} err` : ""}`);
    }
    for (const [id, p] of podsInfo) {
      if (opts.has(id) || (view.cluster && p.cluster_id !== view.cluster)) continue;
      if (view.pool) continue;
      opts.set(id, `${p.name} (${p.owner})`);
    }
    if (view.pod && !opts.has(view.pod)) opts.set(view.pod, view.pod);
    el.pod.replaceChildren(h("option", { value: "" }, "all pods"), ...[...opts].map(([id, label]) => h("option", { value: id }, label)));
    el.pod.value = view.pod;
  }
  function drawMenus() {
    el.bmMenu.replaceChildren(
      h("summary", { class: "btn" }, `Bookmarks${bookmarks.length ? ` (${bookmarks.length})` : ""}`),
      h(
        "div",
        { class: "lx-menubody" },
        h("button", { type: "button", onclick: () => addBookmark(null) }, "Bookmark this view"),
        ...(bookmarks.length
          ? bookmarks.map((b, i) =>
              h(
                "div",
                { class: "lx-bm" },
                h("a", { href: `#/logs?${b.hash}`, title: b.hash }, b.uid ? h("span", { class: `lx-lv lv-${b.level}` }, (b.level || "").toUpperCase()) : badge("view"), " ", b.title),
                h("button", { type: "button", class: "ghost", title: "Remove", onclick: () => ((bookmarks = bookmarks.filter((_, j) => j !== i)), store.set("fvc-log-bookmarks", bookmarks), drawMenus()) }, "×"),
              ),
            )
          : [h("p", { class: "muted small" }, "No bookmarks: b on a line, or bookmark the view.")]),
      ),
    );
    const sp = serverParams(view);
    el.dlMenu.replaceChildren(
      h("summary", { class: "btn" }, "Download"),
      h(
        "div",
        { class: "lx-menubody" },
        h("a", { class: "btn", href: `/api/logs/export?${sp}&format=ndjson`, download: "" }, "Filtered result (NDJSON)"),
        h("a", { class: "btn", href: `/api/logs/export?${sp}&format=txt`, download: "" }, "Filtered result (text)"),
        h("button", { type: "button", onclick: () => download(`fv-logs-loaded.ndjson`, buf.map((l) => JSON.stringify(l)).join("\n") + "\n", "application/x-ndjson") }, `Loaded lines (${buf.length})`),
        h("button", { type: "button", onclick: () => { const t = selectionText(); if (!t) return toast("nothing selected"); download("fv-logs-selection.log", t + "\n"); } }, "Selection (text)"),
        view.pod && (view.src.includes("pod") || view.src.includes("archive")) ? h("a", { class: "btn", href: `/api/logs/download?pod=${encodeURIComponent(view.pod)}&day=${new Date().toISOString().slice(0, 10)}` }, "Pod archive, today (raw)") : null,
      ),
    );
  }
  const onChange = (e: HTMLElement, fn: () => void) => e.addEventListener("change", fn);
  onChange(el.cluster, () => update({ cluster: el.cluster.value, pool: "", pod: "" }));
  onChange(el.pool, () => update({ pool: el.pool.value, pod: "" }));
  onChange(el.pod, () => update({ pod: el.pod.value, src: el.pod.value ? view.src : view.src.filter((s) => s !== "runpod" && s !== "archive") }));
  onChange(el.mode, () => update({ mode: el.mode.value as View["mode"] }));
  onChange(el.order, () => update({ order: el.order.value as View["order"] }, false));
  onChange(el.sort, () => update({ sort: el.sort.value as View["sort"] }, false));
  onChange(el.group, () => update({ group: el.group.value as View["group"] }, false));
  onChange(el.ctx, () => update({ ctx: Math.min(200, Math.max(0, Number(el.ctx.value) || 0)) }, false));
  const timeInput = (inp: HTMLInputElement, k: "from" | "to") =>
    inp.addEventListener("keydown", (ev) => {
      if (ev.key !== "Enter") return;
      const v = inp.value.trim();
      if (!v) return update({ [k]: k === "from" ? DEFAULT_VIEW.from : "" } as Partial<View>);
      if (/^\d+(m|h|d)$/.test(v)) return update({ [k]: v } as Partial<View>);
      const t = parseJump(v);
      if (t === null) return toast(`${k}: not a time`);
      update({ [k]: new Date(t).toISOString() } as Partial<View>);
    });
  timeInput(el.from, "from");
  timeInput(el.to, "to");
  // As you type: a time the server would refuse (log-query schema: from / to) and a regex that does not compile show at once.
  const timeOk = (v: string) => !v.trim() || /^\d+(m|h|d)$/.test(v.trim()) || parseJump(v) !== null;
  const mark = (inp: HTMLInputElement, msg: string | null) => {
    inp.setAttribute("aria-invalid", String(!!msg));
    inp.classList.toggle("bad", !!msg);
    const tip = inp.nextElementSibling?.classList.contains("lx-err") ? (inp.nextElementSibling as HTMLElement) : null;
    if (msg && !tip) inp.after(h("span", { class: "lx-err small", role: "alert" }, msg));
    else if (tip) msg ? (tip.textContent = msg) : tip.remove();
  };
  for (const [inp, what] of [[el.from, "from"], [el.to, "to"], [el.jump, "jump"]] as const)
    inp.addEventListener("input", () => mark(inp, timeOk(inp.value) ? null : `${what}: UTC 2026-10-06 12:00, 12:00, unix ms, or 15m / 2h / 7d`));
  const reCheck = () => {
    let msg: string | null = null;
    if (view.re && el.search.value) {
      if (el.search.value.length > 300) msg = "regex: at most 300 characters";
      else
        try {
          new RegExp(el.search.value);
        } catch (e) {
          msg = `regex: ${(e as Error).message.replace(/^Invalid regular expression: /, "")}`;
        }
    }
    el.search.setAttribute("aria-invalid", String(!!msg));
    el.search.title = msg || "";
    const box = el.search.parentElement!;
    const tip = box.nextElementSibling?.classList.contains("lx-err") ? (box.nextElementSibling as HTMLElement) : null;
    if (msg && !tip) box.after(h("span", { class: "lx-err small", role: "alert" }, msg));
    else if (tip) msg ? (tip.textContent = msg) : tip.remove();
    return !msg;
  };
  el.search.addEventListener("input", reCheck);
  el.re.addEventListener("click", () => setTimeout(reCheck));
  el.jump.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter") return;
    const t = parseJump(el.jump.value);
    if (t === null) return toast("jump: not a time");
    jumpTo(t);
  });
  let searchT: any;
  el.search.addEventListener("input", () => {
    clearTimeout(searchT);
    searchT = setTimeout(() => (view.re && !reCheck() ? undefined : update({ q: el.search.value }, view.mode === "filter")), view.mode === "filter" ? 450 : 120);
  });
  el.search.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      clearTimeout(searchT);
      if (el.search.value !== view.q) update({ q: el.search.value }, view.mode === "filter");
      else step(ev.shiftKey ? -1 : 1);
    }
    if (ev.key === "Escape") el.list.focus();
  });
  el.re.addEventListener("click", () => update({ re: !view.re }, view.mode === "filter" && !!view.q));
  el.cs.addEventListener("click", () => update({ cs: !view.cs }, view.mode === "filter" && !!view.q));
  el.live.addEventListener("click", () => toggleLive());
  el.newPill.addEventListener("click", () => scrollToHead());

  /** Applies a view change: the URL, the controls, and a reload when the server's filters changed. */
  function update(p: Partial<View>, reload = true) {
    const before = serverParams(view).toString();
    view = { ...view, ...p };
    if (view.src.includes("runpod") && !view.pod) view.src = view.src.filter((s) => s !== "runpod");
    if (view.src.includes("archive") && !view.pod) view.src = view.src.filter((s) => s !== "archive");
    if (view.to) view.live = false;
    syncUrl();
    drawControls();
    compileFind();
    if (reload && (serverParams(view).toString() !== before || p.at !== undefined)) load();
    else rebuild(true);
  }
  function syncUrl() {
    const qs = viewToParams({ ...view, sel: focus }).toString();
    const want = `#/logs${qs ? `?${qs}` : ""}`;
    if (location.hash !== want) history.replaceState(null, "", want);
  }
  function compileFind() {
    const m = matcher(view.q, view.re, view.cs);
    findTest = m ? (l) => m(searchable(l)) : null;
    el.search.classList.toggle("bad", !!view.q && view.re && !m);
  }

  // ---- loading
  const q = (extra: Record<string, string>) => {
    const p = serverParams(view);
    for (const [k, v] of Object.entries(extra)) p.set(k, v);
    return api(`/api/logs/query?${p}`);
  };
  async function load() {
    const my = ++seq;
    buf = [];
    seen.clear();
    olderCursor = newerCursor = null;
    partial = null;
    pending = [];
    unseen = 0;
    tail = null;
    lastError = "";
    status("loading…");
    rebuild(false);
    loadFacets();
    try {
      if (view.at !== null) {
        const c = encodeCursor(view.at, "~");
        const [older, newer] = await Promise.all([q({ order: "desc", limit: String(PAGE), cursor: c }), q({ order: "asc", limit: String(Math.round(PAGE / 2)), cursor: c })]);
        if (my !== seq) return;
        add([...older.lines, ...newer.lines]);
        olderCursor = older.next;
        newerCursor = newer.next;
        partial = older.partial ? `search stopped at ${iso(older.searched_until)}` : null;
        atHead = !newer.next && !view.to;
      } else {
        const r = await q({ order: "desc", limit: String(PAGE) });
        if (my !== seq) return;
        add(r.lines);
        olderCursor = r.next;
        partial = r.partial ? `search stopped at ${iso(r.searched_until)}` : null;
        atHead = !view.to;
      }
      rebuild(false);
      if (view.sel && rows.some((r) => r.kind === "line" && r.line.uid === view.sel)) {
        anchor = focus = view.sel;
        scrollToUid(view.sel, "center");
      } else if (view.at !== null) {
        const i = rows.findIndex((r) => r.kind === "line" && (view.order === "asc" ? r.line.ts >= view.at! : r.line.ts <= view.at!));
        if (i >= 0) scrollToIndex(i, "center");
      } else if (view.sort === "time" && view.group === "none") scrollToHead();
      else (el.list.scrollTop = 0), render();
      drawDetail();
      if (view.live) startTail();
    } catch (e) {
      if (my !== seq) return;
      lastError = (e as Error).message;
    }
    status();
  }
  function add(lines: XLine[]) {
    buf = mergeLines(buf, lines, seen);
    if (buf.length > MAX_LINES) {
      const drop = buf.length - MAX_LINES;
      // Live tail at the head: drop the oldest; the older edge continues from the new oldest line.
      for (const l of buf.slice(0, drop)) seen.delete(l.uid);
      buf = buf.slice(drop);
      olderCursor = encodeCursor(buf[0]!.ts, buf[0]!.uid);
    }
  }
  async function loadOlder() {
    if (!olderCursor || loadingOlder) return;
    loadingOlder = true;
    const my = seq;
    status("loading older lines…");
    try {
      const r = await q({ order: "desc", limit: String(PAGE), cursor: olderCursor });
      if (my !== seq) return;
      keepPosition(() => add(r.lines), view.order === "asc");
      olderCursor = r.next;
      partial = r.partial ? `search stopped at ${iso(r.searched_until)}` : null;
    } catch (e) {
      lastError = (e as Error).message;
    } finally {
      loadingOlder = false;
      status();
    }
  }
  async function loadNewer() {
    if (!newerCursor || loadingNewer) return;
    loadingNewer = true;
    const my = seq;
    try {
      const r = await q({ order: "asc", limit: String(PAGE), cursor: newerCursor });
      if (my !== seq) return;
      keepPosition(() => add(r.lines), view.order === "desc");
      newerCursor = r.next;
      if (!r.next && !view.to) {
        atHead = true;
        if (view.live) startTail();
      }
    } catch (e) {
      lastError = (e as Error).message;
    } finally {
      loadingNewer = false;
      status();
    }
  }
  /** Runs a buffer change and keeps the lines on screen where they were (rows added above shift the scroll). */
  function keepPosition(fn: () => void, addsAbove: boolean) {
    const firstVisible = rowAt(el.list.scrollTop);
    const uid = firstVisible?.kind === "line" ? firstVisible.line.uid : null;
    const offset = el.list.scrollTop % ROW_H;
    fn();
    rebuild(false);
    if (addsAbove && uid) {
      const i = indexOf(uid);
      if (i >= 0) el.list.scrollTop = i * ROW_H + offset;
    }
    render();
  }

  // ---- live tail
  let liveTimer: any = null;
  function startTail() {
    stopTail();
    if (!view.live || !atHead || view.to) return;
    const my = seq;
    const tick = async () => {
      if (my !== seq || !view.live) return;
      try {
        const p = serverParams(view);
        if (tail) p.set("tail", tail);
        const r = await api(`/api/logs/live?${p}`);
        if (my !== seq) return;
        tail = r.tail;
        if (r.lines.length) {
          if (paused) {
            pending.push(...r.lines);
            drawControls();
          } else applyLive(r.lines);
        }
        lastError = "";
      } catch (e) {
        lastError = `live: ${(e as Error).message}`;
      }
      status();
    };
    tick();
    liveTimer = setInterval(tick, LIVE_MS);
  }
  function stopTail() {
    if (liveTimer) clearInterval(liveTimer);
    liveTimer = null;
  }
  o.onCleanup(() => {
    stopTail();
    seq++;
  });
  function atHeadEdge() {
    const max = el.list.scrollHeight - el.list.clientHeight;
    return view.order === "asc" ? max - el.list.scrollTop < ROW_H * 2 : el.list.scrollTop < ROW_H * 2;
  }
  function applyLive(lines: XLine[]) {
    const stick = stuck;
    const n = buf.length;
    keepPosition(() => add(lines), view.order === "desc");
    const added = buf.length - n;
    if (stick) scrollToHead();
    else if (added > 0) {
      unseen += added;
      el.newPill.hidden = false;
      el.newPill.textContent = `${unseen} new line${unseen === 1 ? "" : "s"} ${view.order === "asc" ? "↓" : "↑"}`;
    }
  }
  function toggleLive() {
    if (view.to) return toast("live tail needs an open range: clear 'to'");
    if (!view.live) {
      paused = false;
      // Away from now (a jump): reload at the head first; load() starts the tail.
      if (!atHead) return update({ live: true, at: null, sel: "" });
      update({ live: true }, false);
      startTail();
    } else if (!paused) {
      paused = true;
      drawControls();
    } else {
      paused = false;
      const p = pending;
      pending = [];
      if (p.length) applyLive(p);
      drawControls();
    }
    status();
  }
  function stopLive() {
    paused = false;
    pending = [];
    stopTail();
    update({ live: false }, false);
  }
  function followPod(pod: string | null) {
    if (!pod) return toast("choose a pod (or press f on one of its lines)");
    paused = false;
    update({ pod, src: view.src.includes("pod") ? view.src : [...view.src, "pod"], live: true, at: null, to: "", sort: "time", group: "none", from: view.from || "1h" });
  }

  // ---- facets: the histogram and the pod picker's counts
  let facetT: any;
  async function loadFacets() {
    clearTimeout(facetT);
    const my = seq;
    try {
      const p = serverParams(view);
      const f = await api(`/api/logs/facets?${p}&buckets=90`);
      if (my !== seq) return;
      facetPods = f.pods || [];
      drawPodPicker();
      drawHist(f);
    } catch {
      el.hist.replaceChildren();
    }
    if (view.live) facetT = setTimeout(loadFacets, 30_000);
  }
  o.onCleanup(() => clearTimeout(facetT));
  function drawHist(f: { since: number; until: number; step_ms: number; histogram: { t: number; counts: Record<string, number> }[] }) {
    const W = 900;
    const H = 46;
    const n = Math.max(1, Math.ceil((f.until - f.since) / f.step_ms));
    const bw = W / n;
    const max = Math.max(1, ...f.histogram.map((b) => Object.values(b.counts).reduce((s, x) => s + x, 0)));
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "none");
    const rect = (x: number, y: number, w: number, hh: number, cls: string) => {
      const r = document.createElementNS(ns, "rect");
      r.setAttribute("x", String(x));
      r.setAttribute("y", String(y));
      r.setAttribute("width", String(Math.max(0.5, w)));
      r.setAttribute("height", String(Math.max(0, hh)));
      r.setAttribute("class", cls);
      return r;
    };
    // The loaded window.
    if (buf.length) {
      const x0 = ((buf[0]!.ts - f.since) / (f.until - f.since)) * W;
      const x1 = ((buf[buf.length - 1]!.ts - f.since) / (f.until - f.since)) * W;
      svg.append(rect(Math.max(0, x0), 0, Math.max(2, x1 - x0), H, "lx-hist-win"));
    }
    for (const b of f.histogram) {
      const i = Math.round((b.t - f.since) / f.step_ms);
      const err = b.counts.error || 0;
      const warn = b.counts.warn || 0;
      const rest = Object.entries(b.counts).reduce((s, [k, v]) => (k === "error" || k === "warn" ? s : s + v), 0);
      let y = H;
      for (const [v, cls] of [[rest, "lx-hist-info"], [warn, "lx-hist-warn"], [err, "lx-hist-error"]] as const) {
        const hh = (v / max) * (H - 4);
        if (v) svg.append(rect(i * bw + 0.5, y - hh, bw - 1, hh, cls));
        y -= hh;
      }
    }
    const tip = h("div", { class: "lx-hist-tip small muted" }, `${iso(f.since).slice(0, 16)} → ${iso(f.until).slice(0, 16)} UTC · pod lines per ${Math.round(f.step_ms / 60000) || "<1"} min · click to jump`);
    svg.addEventListener("click", (ev) => {
      const r = svg.getBoundingClientRect();
      jumpTo(Math.round(f.since + ((ev.clientX - r.left) / r.width) * (f.until - f.since)));
    });
    svg.addEventListener("mousemove", (ev) => {
      const r = svg.getBoundingClientRect();
      const t = f.since + ((ev.clientX - r.left) / r.width) * (f.until - f.since);
      const b = f.histogram.find((x) => t >= x.t && t < x.t + f.step_ms);
      tip.textContent = `${iso(t).slice(0, 19)} UTC${b ? ` · ${Object.entries(b.counts).map(([k, v]) => `${v} ${k}`).join(", ")}` : " · no lines"} · click to jump`;
    });
    el.hist.replaceChildren(svg, tip);
  }

  // ---- rows and rendering
  function rebuild(keep: boolean) {
    const top = keep ? rowAt(el.list.scrollTop) : null;
    rows = buildRows(buf, view);
    el.spacer.style.height = `${rows.length * ROW_H}px`;
    if (keep && top?.kind === "line") {
      const i = indexOf(top.line.uid);
      if (i >= 0) el.list.scrollTop = i * ROW_H;
    }
    render();
    drawPins();
  }
  const rowAt = (y: number) => rows[Math.floor(y / ROW_H)];
  const indexOf = (uid: string) => rows.findIndex((r) => r.kind === "line" && r.line.uid === uid);
  let raf = 0;
  const schedule = () => {
    if (!raf) raf = requestAnimationFrame(() => ((raf = 0), render()));
  };
  function selected(): Set<string> {
    const s = new Set(anchor ? selectedRange(rows, anchor, focus) : []);
    for (const u of extra) s.add(u);
    return s;
  }
  function render() {
    const top = el.list.scrollTop;
    const hgt = el.list.clientHeight || 600;
    const first = Math.max(0, Math.floor(top / ROW_H) - OVERSCAN);
    const last = Math.min(rows.length, Math.ceil((top + hgt) / ROW_H) + OVERSCAN);
    const sel = selected();
    const pinned = new Set(pins.map((p) => p.uid));
    const marked = new Set(bookmarks.flatMap((b) => (b.uid ? [b.uid] : [])));
    const frag = document.createDocumentFragment();
    for (let i = first; i < last; i++) frag.append(rowEl(rows[i]!, i, sel, pinned, marked));
    el.win.replaceChildren(frag);
    if (!rows.length) el.win.replaceChildren(h("div", { class: "lx-empty muted" }, lastError ? "" : seq && !buf.length ? "No lines match. Widen the time range, the levels or the sources." : ""));
    // Near an edge with more to load: fetch it.
    if (rows.length && top < ROW_H * 40) (view.order === "asc" ? loadOlder : loadNewer)();
    if (rows.length && top + hgt > rows.length * ROW_H - ROW_H * 40) (view.order === "asc" ? loadNewer : loadOlder)();
    if (unseen && atHeadEdge()) {
      unseen = 0;
      el.newPill.hidden = true;
    }
  }
  function rowEl(r: Row, i: number, sel: Set<string>, pinned: Set<string>, marked: Set<string>): HTMLElement {
    const y = `top:${i * ROW_H}px`;
    if (r.kind === "group")
      return h("div", { class: `lx-row lx-group${view.group === "pod" ? ` pod-c${colorSlot(r.key)}` : ` lv-${r.key}`}`, style: y, "data-i": i }, h("b", {}, r.label), h("span", { class: "muted" }, ` ${r.count} line${r.count === 1 ? "" : "s"}`));
    const l = r.line;
    const isFocus = l.uid === focus;
    const isMatch = !!(view.q && findTest && view.mode === "find" && findTest(l));
    const cls = ["lx-row", `lv-${l.level}`, sel.has(l.uid) && "sel", isFocus && "focus", pinned.has(l.uid) && "pinned", isMatch && "match", `pod-c${colorSlot(podKey(l))}`].filter(Boolean).join(" ");
    const who = l.pod_id ? (l.pool ? `${l.pool} ${l.pod_id.slice(0, 6)}` : l.pod_name || l.pod_id) : l.source === "op" ? `op ${String(l.fields?.op_kind ?? "")}` : l.source;
    const msg = h("span", { class: "lx-msg" });
    highlight(msg, "", l.msg);
    if (l.fields) highlight(msg, " ", compactFields(l.fields), "lx-fields");
    return h(
      "div",
      { class: cls, style: y, "data-i": i, role: "option", "aria-selected": String(sel.has(l.uid)) },
      h("span", { class: "lx-mark" }, marked.has(l.uid) ? "★" : pinned.has(l.uid) ? "📌" : ""),
      h("span", { class: "lx-t", title: iso(l.ts) + " UTC" }, shortTime(l.ts)),
      h("span", { class: `lx-lv lv-${l.level}` }, l.level.toUpperCase()),
      h("span", { class: `lx-who src-${l.source}`, title: [l.source, l.cluster, l.pool, l.pod_id].filter(Boolean).join(" · ") }, who),
      msg,
    );
  }
  function compactFields(f: Record<string, unknown>): string {
    const s = Object.entries(f)
      .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
      .join(" ");
    return s.length > 600 ? s.slice(0, 600) + "…" : s;
  }
  function highlight(into: HTMLElement, prefix: string, text: string, cls?: string) {
    const span = cls ? h("span", { class: cls }) : into;
    if (prefix) span.append(prefix);
    const ranges = view.q ? matchRanges(text, view.q, view.re, view.cs, 20) : [];
    let at = 0;
    for (const [a, b] of ranges) {
      if (a > at) span.append(text.slice(at, a));
      span.append(h("mark", {}, text.slice(a, b)));
      at = b;
    }
    if (at < text.length) span.append(text.slice(at));
    if (cls) into.append(span);
  }
  el.list.addEventListener("scroll", () => ((stuck = atHeadEdge() && view.sort === "time" && view.group === "none"), schedule()), { passive: true });
  const ro = new ResizeObserver(() => schedule());
  ro.observe(el.list);
  /** The list and the details fill the window below the controls (the page itself does not scroll). */
  const fit = () => {
    const top = el.list.getBoundingClientRect().top + window.scrollY;
    const hgt = Math.max(320, window.innerHeight - top - 14);
    const want = window.innerWidth > 900 ? `${hgt}px` : "";
    if (el.list.style.height === want) return;
    el.list.style.height = el.detail.style.height = want;
    if (stuck) scrollToHead();
  };
  const onResize = () => (fit(), schedule());
  window.addEventListener("resize", onResize);
  const ro2 = new ResizeObserver(() => fit());
  ro2.observe(host.querySelector(".lx-head")!.parentElement!);
  o.onCleanup(() => (ro.disconnect(), ro2.disconnect(), window.removeEventListener("resize", onResize)));

  function scrollToIndex(i: number, where: "center" | "nearest" = "nearest") {
    const y = i * ROW_H;
    const hgt = el.list.clientHeight;
    if (where === "center") el.list.scrollTop = Math.max(0, y - hgt / 2);
    else if (y < el.list.scrollTop) el.list.scrollTop = y;
    else if (y + ROW_H > el.list.scrollTop + hgt) el.list.scrollTop = y + ROW_H - hgt;
    render();
  }
  function scrollToUid(uid: string, where: "center" | "nearest" = "nearest") {
    const i = indexOf(uid);
    if (i >= 0) scrollToIndex(i, where);
  }
  function scrollToHead() {
    el.list.scrollTop = view.order === "asc" ? el.list.scrollHeight : 0;
    stuck = view.sort === "time" && view.group === "none";
    unseen = 0;
    el.newPill.hidden = true;
    render();
  }

  // ---- selection (click, shift-click, ctrl-click, drag)
  let drag: { from: string; moved: boolean } | null = null;
  const uidAtEvent = (ev: MouseEvent): string | null => {
    const hit = (ev.target as HTMLElement | null)?.closest?.(".lx-row") as HTMLElement | null;
    if (hit && el.win.contains(hit)) {
      const row = rows[Number(hit.dataset.i)];
      return row?.kind === "line" ? row.line.uid : null;
    }
    const r = el.list.getBoundingClientRect();
    const row = rows[Math.floor((ev.clientY - r.top + el.list.scrollTop) / ROW_H)];
    return row?.kind === "line" ? row.line.uid : null;
  };
  el.list.addEventListener("mousedown", (ev) => {
    if (ev.button !== 0) return;
    const uid = uidAtEvent(ev);
    if (!uid) return;
    ev.preventDefault();
    el.list.focus();
    if (ev.shiftKey && anchor) focus = uid;
    else if (ev.ctrlKey || ev.metaKey) {
      extra.has(uid) ? extra.delete(uid) : extra.add(uid);
      focus = uid;
      if (!anchor) anchor = uid;
    } else {
      anchor = focus = uid;
      extra.clear();
      drag = { from: uid, moved: false };
    }
    selectionChanged();
  });
  let dragScroll: any = null;
  const onMove = (ev: MouseEvent) => {
    if (!drag) return;
    const r = el.list.getBoundingClientRect();
    clearInterval(dragScroll);
    if (ev.clientY < r.top + 10 || ev.clientY > r.bottom - 10) {
      const dir = ev.clientY < r.top + 10 ? -1 : 1;
      dragScroll = setInterval(() => ((el.list.scrollTop += dir * ROW_H), render()), 40);
    }
    const uid = uidAtEvent(ev);
    if (uid && uid !== focus) {
      focus = uid;
      drag.moved = true;
      selectionChanged(false);
    }
  };
  const onUp = () => {
    clearInterval(dragScroll);
    if (drag && !drag.moved) {
      detailOpen = true;
      drawDetail();
    }
    drag = null;
  };
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
  o.onCleanup(() => (document.removeEventListener("mousemove", onMove), document.removeEventListener("mouseup", onUp), clearInterval(dragScroll)));
  function selectionChanged(detail = true) {
    syncUrl();
    render();
    if (detail && detailOpen) drawDetail();
    status();
  }
  function lineOf(uid: string): XLine | undefined {
    const r = rows[indexOf(uid)];
    return r?.kind === "line" ? r.line : buf.find((l) => l.uid === uid) || pins.find((p) => p.uid === uid);
  }
  function selectionText(): string {
    const s = selected();
    return rows
      .filter((r): r is Extract<Row, { kind: "line" }> => r.kind === "line" && s.has(r.line.uid))
      .map((r) => formatLine(r.line))
      .join("\n");
  }
  const onCopy = (ev: ClipboardEvent) => {
    const active = document.activeElement;
    if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
    const ws = window.getSelection();
    if (ws && !ws.isCollapsed && ws.toString().trim()) return; // the user selected text: the browser copies it
    const t = selectionText();
    if (!t || !host.isConnected) return;
    ev.clipboardData?.setData("text/plain", t);
    ev.preventDefault();
    toast(`copied ${t.split("\n").length} line(s)`);
  };
  document.addEventListener("copy", onCopy);
  o.onCleanup(() => document.removeEventListener("copy", onCopy));

  // ---- navigation
  function moveFocus(d: number, extend: boolean) {
    if (!rows.length) return;
    let i = focus ? indexOf(focus) : -1;
    if (i < 0) i = d > 0 ? -1 : rows.length;
    do i += d;
    while (i >= 0 && i < rows.length && rows[i]!.kind !== "line");
    if (i < 0 || i >= rows.length) return;
    const uid = (rows[i] as Extract<Row, { kind: "line" }>).line.uid;
    focus = uid;
    if (!extend) {
      anchor = uid;
      extra.clear();
    }
    scrollToIndex(i);
    selectionChanged();
  }
  function edge(last: boolean) {
    if (!rows.length) return;
    const k = last ? [...rows].reverse().find((x) => x.kind === "line") : rows.find((x) => x.kind === "line");
    if (k?.kind === "line") (anchor = focus = k.line.uid), extra.clear();
    scrollToIndex(last ? rows.length - 1 : 0);
    selectionChanged();
  }
  /** n / N: the next match of the search; without one, the next warn or error. */
  function step(dir: 1 | -1) {
    const test = view.q && findTest ? findTest : (l: XLine) => l.level === "warn" || l.level === "error";
    const from = focus ? indexOf(focus) : dir > 0 ? -1 : rows.length;
    const i = nextMatch(rows, from, dir, test);
    if (i < 0) return toast(view.q ? "no match in the loaded lines" : "no warn or error in the loaded lines");
    const r = rows[i] as Extract<Row, { kind: "line" }>;
    anchor = focus = r.line.uid;
    extra.clear();
    scrollToIndex(i, "center");
    selectionChanged();
  }
  function jumpTo(t: number) {
    anchor = focus = "";
    stopTail();
    update({ at: t, sel: "" });
  }
  const onKey = (ev: KeyboardEvent) => {
    if (!host.isConnected || el.help.open) return;
    const t = ev.target as HTMLElement;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
    if (typing) {
      if (ev.key === "Escape") (t as HTMLInputElement).blur();
      return;
    }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const k = ev.key;
    const act: Record<string, () => void> = {
      j: () => moveFocus(1, ev.shiftKey),
      J: () => moveFocus(1, true),
      ArrowDown: () => moveFocus(1, ev.shiftKey),
      k: () => moveFocus(-1, ev.shiftKey),
      K: () => moveFocus(-1, true),
      ArrowUp: () => moveFocus(-1, ev.shiftKey),
      g: () => edge(false),
      Home: () => edge(false),
      G: () => edge(true),
      End: () => edge(true),
      "/": () => (el.search.focus(), el.search.select()),
      n: () => step(1),
      N: () => step(-1),
      Enter: () => ((detailOpen = !detailOpen), drawDetail()),
      Escape: () => {
        if (detailOpen) detailOpen = false;
        else (anchor = focus = ""), extra.clear();
        drawDetail();
        selectionChanged(false);
      },
      p: () => focus && togglePin(focus),
      b: () => focus && addBookmark(focus),
      c: () => focus && ((detailOpen = true), drawDetail(true)),
      f: () => {
        const l = focus ? lineOf(focus) : null;
        followPod(l?.pod_id ?? null);
      },
      t: () => (el.jump.focus(), el.jump.select()),
      l: () => (view.live ? stopLive() : toggleLive()),
      " ": () => (view.live ? toggleLive() : undefined),
      "?": () => el.help.showModal(),
    };
    const fn = act[k];
    if (!fn) return;
    ev.preventDefault();
    fn();
  };
  document.addEventListener("keydown", onKey);
  o.onCleanup(() => document.removeEventListener("keydown", onKey));

  // ---- pins and bookmarks
  function togglePin(uid: string) {
    const l = lineOf(uid);
    if (!l) return;
    pins = pins.some((p) => p.uid === uid) ? pins.filter((p) => p.uid !== uid) : [...pins, l].slice(-50);
    try {
      sessionStorage.setItem("fvc-log-pins", JSON.stringify(pins));
    } catch {
      /* storage off */
    }
    drawPins();
    render();
    if (detailOpen) drawDetail();
  }
  function drawPins() {
    if (!pins.length) return el.pins.replaceChildren();
    el.pins.replaceChildren(
      h("div", { class: "lx-pins-head small muted" }, `Pinned (${pins.length})`, h("button", { type: "button", class: "ghost small", onclick: () => ((pins = []), sessionStorage.removeItem("fvc-log-pins"), drawPins(), render()) }, "clear")),
      ...pins.map((p) =>
        h(
          "div",
          { class: `lx-pin lv-${p.level}` },
          h("a", { href: "#", onclick: (ev: Event) => (ev.preventDefault(), reveal(p)) }, h("span", { class: "lx-t" }, iso(p.ts).slice(5, 23)), " ", h("span", { class: `lx-lv lv-${p.level}` }, p.level.toUpperCase()), " ", h("span", { class: "lx-who" }, p.pod_name || p.pod_id || p.source), " ", p.msg.slice(0, 200)),
          h("button", { type: "button", class: "ghost", title: "Unpin", onclick: () => togglePin(p.uid) }, "×"),
        ),
      ),
    );
  }
  /** Selects a line: in the buffer if it is there, else loads around its time. */
  function reveal(l: XLine) {
    if (indexOf(l.uid) >= 0) {
      anchor = focus = l.uid;
      extra.clear();
      scrollToUid(l.uid, "center");
      detailOpen = true;
      drawDetail();
      selectionChanged();
      return;
    }
    anchor = focus = l.uid;
    view.sel = l.uid;
    update({ at: l.ts, sel: l.uid, live: false });
  }
  function addBookmark(uid: string | null) {
    const l = uid ? lineOf(uid) : null;
    const v = l ? { ...view, at: l.ts, sel: l.uid, live: false } : { ...view, sel: "" };
    const hash = viewToParams(v).toString();
    const title = l ? `${iso(l.ts).slice(5, 19)} ${l.pod_name || l.pod_id || l.source}: ${l.msg.slice(0, 80)}` : `${describeRange(view.from, view.to)}${view.q ? ` · "${view.q}"` : ""}${view.pod ? ` · ${view.pod}` : view.cluster ? ` · ${clusters.find((c) => c.id === view.cluster)?.name || view.cluster}` : ""}`;
    if (bookmarks.some((b) => b.hash === hash)) return toast("already bookmarked");
    bookmarks = [{ uid: l?.uid ?? null, ts: l?.ts ?? null, level: l?.level, title, hash, saved_at: Date.now() }, ...bookmarks].slice(0, 100);
    store.set("fvc-log-bookmarks", bookmarks);
    drawMenus();
    render();
    toast(l ? "line bookmarked" : "view bookmarked");
  }

  // ---- details
  async function drawDetail(withContext = false) {
    const l = focus ? lineOf(focus) : null;
    el.detail.classList.toggle("open", detailOpen && !!l);
    main.classList.toggle("with-detail", detailOpen && !!l);
    if (!detailOpen || !l) return el.detail.replaceChildren();
    const isPinned = pins.some((p) => p.uid === l.uid);
    const ctxBox = h("div", { class: "lx-ctx-box" });
    const kv = (k: string, v: Node | string | null) => (v === null || v === "" ? [] : [h("dt", {}, k), h("dd", {}, v)]);
    const fieldRows = Object.entries(l.fields || {}).map(([k, v]) =>
      h("tr", {}, h("td", { class: "lx-fk" }, k), h("td", { class: "lx-fv" }, typeof v === "string" ? v : h("pre", {}, JSON.stringify(v, null, 2))), h("td", {}, h("button", { type: "button", class: "ghost small", title: "Show only lines with this value", onclick: () => update({ q: typeof v === "string" ? v : JSON.stringify(v), re: false, mode: "filter" }) }, "filter"))),
    );
    const sel = selected();
    el.detail.replaceChildren(
      ...([
      h(
        "div",
        { class: "lx-detail-head" },
        h("span", { class: `lx-lv lv-${l.level}` }, l.level.toUpperCase()),
        h("b", {}, iso(l.ts), " UTC"),
        h("span", { class: "spacer" }),
        h("button", { type: "button", class: "ghost", title: "Close (Esc)", onclick: () => ((detailOpen = false), drawDetail()) }, "×"),
      ),
      sel.size > 1 ? h("p", { class: "small" }, badge(`${sel.size} lines selected`), " ", h("button", { type: "button", class: "small", onclick: () => copyText(selectionText()).then(() => toast(`copied ${sel.size} lines`)) }, "Copy selection")) : null,
      h("pre", { class: "lx-detail-msg" }, l.msg),
      h(
        "dl",
        { class: "kv" },
        ...kv("source", SOURCE_LABEL[l.source]),
        ...kv("cluster", l.cluster ? h("a", { href: "#", onclick: (ev: Event) => (ev.preventDefault(), update({ cluster: l.cluster_id || "", pool: "", pod: "" })) }, l.cluster) : l.cluster_id),
        ...kv("pool", l.pool),
        ...kv("pod", l.pod_id ? h("span", {}, h("a", { href: `#/pod/${l.pod_id}` }, l.pod_name || l.pod_id), " · ", h("a", { href: "#", onclick: (ev: Event) => (ev.preventDefault(), update({ pod: l.pod_id! })) }, "only this pod"), " · ", h("a", { href: "#", onclick: (ev: Event) => (ev.preventDefault(), followPod(l.pod_id)) }, "follow")) : null),
        ...kv("operation", l.source === "op" ? h("a", { href: "#", onclick: (ev: Event) => (ev.preventDefault(), update({ op: String(l.fields?.op_id), src: ["op"] })) }, `${l.fields?.op_kind} ${l.fields?.op_id} (${l.fields?.op_status})`) : null),
        ...kv("target", l.target),
        ...kv("uid", h("code", {}, l.uid)),
      ),
      fieldRows.length ? h("table", { class: "lx-ftable" }, h("tbody", {}, ...fieldRows)) : null,
      h(
        "div",
        { class: "row lx-detail-actions" },
        h("button", { type: "button", onclick: () => copyText(formatLine(l)).then(() => toast("line copied")) }, "Copy line"),
        h("button", { type: "button", onclick: () => copyText(JSON.stringify(l, null, 2)).then(() => toast("JSON copied")) }, "Copy JSON"),
        h("button", { type: "button", onclick: () => togglePin(l.uid) }, isPinned ? "Unpin" : "Pin"),
        h("button", { type: "button", onclick: () => addBookmark(l.uid) }, "Bookmark"),
        h("button", { type: "button", onclick: () => copyText(location.origin + location.pathname + "#/logs?" + viewToParams({ ...view, at: l.ts, sel: l.uid, live: false }).toString()).then(() => toast("link copied")) }, "Link"),
        ["pod", "op", "audit"].includes(l.source) ? h("button", { type: "button", onclick: () => loadContext(l, ctxBox) }, `Context ±${view.ctx}`) : null,
      ),
      ctxBox,
      ] as (HTMLElement | null)[]).filter((x): x is HTMLElement => x !== null),
    );
    if (withContext) loadContext(l, ctxBox);
  }
  async function loadContext(l: XLine, box: HTMLElement) {
    if (!["pod", "op", "audit"].includes(l.source)) return;
    box.replaceChildren(h("p", { class: "muted small" }, "loading context…"));
    try {
      const r = await api(`/api/logs/context?uid=${encodeURIComponent(l.uid)}&before=${view.ctx}&after=${view.ctx}`);
      box.replaceChildren(
        h("h3", {}, `Context: ${view.ctx} before and after`, l.source === "pod" ? " (same pod, every level)" : l.source === "op" ? " (same operation)" : " (audit log)"),
        h(
          "div",
          { class: "lx-ctx" },
          ...r.lines.map((c: XLine) =>
            h(
              "div",
              { class: `lx-ctx-line lv-${c.level}${c.uid === l.uid ? " anchor" : ""}`, title: "Show in the list", onclick: () => reveal(c) },
              h("span", { class: "lx-t" }, shortTime(c.ts)),
              " ",
              h("span", { class: `lx-lv lv-${c.level}` }, c.level.toUpperCase()),
              " ",
              c.msg,
            ),
          ),
        ),
        h("button", { type: "button", class: "small", onclick: () => update({ pod: l.pod_id || view.pod, src: l.source === "pod" ? ["pod"] : view.src, op: l.source === "op" ? String(l.fields?.op_id || "") : view.op, lv: [...LEVELS], q: "", at: l.ts, sel: l.uid, live: false }) }, "Open this stretch in the list"),
      );
    } catch (e) {
      box.replaceChildren(h("p", { class: "small" }, (e as Error).message));
    }
  }

  // ---- status line
  function status(msg?: string) {
    const parts: (Node | string)[] = [];
    if (msg) parts.push(msg);
    else {
      parts.push(`${buf.length.toLocaleString()} line${buf.length === 1 ? "" : "s"} loaded · ${describeRange(view.from, view.to)}`);
      const s = selected().size;
      if (s > 1) parts.push(` · ${s} selected`);
      if (view.q && view.mode === "find" && findTest) parts.push(` · ${buf.filter(findTest).length} matches`);
      if (olderCursor) parts.push(" · ", h("a", { href: "#", onclick: (ev: Event) => (ev.preventDefault(), loadOlder()) }, "load older"));
      if (newerCursor) parts.push(" · ", h("a", { href: "#", onclick: (ev: Event) => (ev.preventDefault(), loadNewer()) }, "load newer"));
      if (!atHead && !view.to) parts.push(" · ", h("a", { href: "#", onclick: (ev: Event) => (ev.preventDefault(), update({ at: null, sel: "" })) }, "back to now"));
      if (partial) parts.push(" · ", badge(partial, "warn"));
      if (view.live) parts.push(" · ", paused ? badge(`paused${pending.length ? `: ${pending.length} new` : ""}`, "warn") : badge(view.pod ? `following ${podsInfo.get(view.pod)?.name || view.pod}` : "live", "good"));
    }
    if (lastError) parts.push(" · ", h("span", { class: "lx-err" }, lastError));
    el.status.replaceChildren(...parts);
  }

  compileFind();
  drawControls();
  status();
  await load();
}
