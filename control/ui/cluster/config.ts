// The cluster configuration page (#/clusters/new, #/cluster/<id>/config):
// every spec field as a form (or the raw JSON), checked as you type by the
// API's own validator (POST /api/clusters/validate, inline errors at each
// field), a price and Runpod stock preview per pool (POST
// /api/clusters/<id|new>/price), the diff against the saved spec before
// saving, Start / Stop / Scale / Extend with the operation's live log, and
// the effective env of every pod.
import { badge, copyText, h, kids, type Api } from "../dom";
import { createJsonEditor, type Dynamic, type JsonEditor } from "../editor";
import { askExtend } from "../forms/dialogs";
import { schemaAt, unwrap, type Issue, type Schema } from "../schema";
import { renderDiff } from "../view";
import "./config.css";

export interface ConfigOptions {
  api: Api;
  toast: (m: string) => void;
  onCleanup: (fn: () => void) => void;
  /** An existing cluster; none: a new one. */
  id?: string;
  template?: string;
  clone?: string;
}
type Path = (string | number)[];
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v));
const fmt$ = (v: number | null | undefined, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : `$${Number(v).toFixed(d)}`);
const until = (ms: number | null) => {
  if (!ms) return "–";
  const m = Math.round((ms - Date.now()) / 60000);
  return m < 0 ? `${-m} min ago` : m < 120 ? `in ${m} min` : `in ${(m / 60).toFixed(1)} h`;
};
const dur = (s: number) => (s < 3600 ? `${Math.round(s / 60)} min` : `${Math.floor(s / 3600)} h${s % 3600 ? ` ${Math.round((s % 3600) / 60)} min` : ""}`);
const STOCK_KIND: Record<string, string> = { ok: "good", low: "warn", none: "critical", unknown: "" };
const statusKind = (st: string) => (st === "running" || st === "done" ? "good" : st === "starting" || st === "stopping" ? "warn" : st === "failed" ? "critical" : "");

function getAt(v: any, p: Path): any {
  return p.reduce((o, k) => (o == null ? undefined : o[k as any]), v);
}
function setAt(root: any, p: Path, x: any) {
  let o = root;
  for (let i = 0; i < p.length - 1; i++) {
    if (o[p[i]!] == null) o[p[i]!] = typeof p[i + 1] === "number" ? [] : {};
    o = o[p[i]!];
  }
  const last = p[p.length - 1]!;
  if (x === undefined) {
    if (Array.isArray(o) && typeof last === "number") o.splice(last, 1);
    else delete o[last];
  } else o[last] = x;
}

export async function openClusterConfig(host: HTMLElement, o: ConfigOptions) {
  const { api, toast } = o;
  host.replaceChildren(h("p", { class: "muted" }, "loading…"));
  const [schemas, dyn, tpl] = await Promise.all([api("/api/schemas").then((r) => r.schemas as Record<string, Schema>), api(`/api/schemas/dynamic${o.id ? `?cluster=${encodeURIComponent(o.id)}` : ""}`).catch(() => ({}) as Dynamic), api("/api/templates")]);
  const presets: { id: string; title: string; description: string; weights: string[]; licence?: string; pool: any }[] = tpl.pool_presets || [];
  let saved: any = null;
  let version = 0;
  let cl: any = null; // the cluster's view (GET /api/clusters/<id>)
  let spec: any;
  if (o.id) {
    const [d, c] = await Promise.all([api(`/api/docs/cluster-spec/${encodeURIComponent(o.id)}`), api(`/api/clusters/${encodeURIComponent(o.id)}`)]);
    saved = d.doc;
    version = d.version;
    cl = c;
    spec = clone(saved);
  } else if (o.clone) {
    const c = await api(`/api/clusters/${encodeURIComponent(o.clone)}`);
    spec = { ...clone(c.cluster.spec), name: `${c.cluster.name}-copy`.slice(0, 31).replace(/-+$/, "") };
  } else {
    const t = o.template && tpl[o.template] ? o.template : "tiny-cpu";
    spec = { ...clone(tpl[t]), name: "" };
  }
  const id = o.id ? cl.cluster.id : null;
  let issues: Issue[] = [];
  let warnings: Issue[] = [];
  let price: any = null;
  let priceErr = "";
  let priceBusy = false;
  let mode: "form" | "json" = "form";
  let jsonOk = true;
  const openPools = new Set<number>();
  let watchOp: string | null = cl?.op?.id ?? null;

  // ---- skeleton
  const el = {
    title: h("h1", {}),
    sub: h("p", { class: "muted small cf-sub" }),
    form: h("div", { class: "cf-form", "data-schema-form": "cluster-spec" }),
    json: h("div", { class: "cf-json", hidden: true }),
    tabs: h("div", { class: "tabs", role: "tablist" }),
    issues: h("div", { class: "cf-issues" }),
    price: h("div", { class: "cf-price" }),
    changes: h("div", { class: "cf-changes" }),
    ops: h("div", { class: "cf-ops" }),
    oplog: h("div", { class: "cf-oplog" }),
    env: h("div", { class: "cf-env" }),
    save: h("button", { type: "button", class: "primary" }, o.id ? "Review & save" : "Review & define"),
    dirty: h("span", { class: "muted small" }),
  };
  const side = h(
    "aside",
    { class: "cf-side" },
    h("section", { class: "card cf-card" }, h("h2", {}, o.id ? "Cluster" : "Define"), el.ops, h("div", { class: "row", style: "margin-top:8px" }, el.save, el.dirty)),
    h("section", { class: "card cf-card" }, h("h2", {}, "Checks"), el.issues),
    h("section", { class: "card cf-card" }, h("h2", {}, "Price & stock"), el.price),
    h("section", { class: "card cf-card" }, h("h2", {}, o.id ? "Changes (saved → yours)" : "Spec"), el.changes),
    el.oplog,
  );
  host.replaceChildren(
    h("div", { class: "cf" }, h("div", { class: "cf-head" }, h("a", { href: o.id ? `#/cluster/${id}` : "#/clusters", class: "small" }, o.id ? `← ${cl.cluster.name}` : "← Clusters"), el.title, el.sub), h("div", { class: "cf-grid" }, h("div", { class: "cf-main" }, el.tabs, el.form, el.json, o.id ? h("section", { class: "card cf-section", id: "cf-env" }, h("h2", {}, "Effective env per pod"), el.env) : null), side)),
  );
  el.save.addEventListener("click", () => review());

  // ---- tabs
  let editor: JsonEditor | null = null;
  const tabBtn = (m: "form" | "json", label: string) => h("button", { type: "button", class: m === mode ? "on" : "", onclick: () => switchTo(m) }, label);
  function drawTabs() {
    el.tabs.replaceChildren(tabBtn("form", "Form"), tabBtn("json", "JSON"));
  }
  function switchTo(m: "form" | "json") {
    if (m === "form" && !jsonOk) return toast("fix the JSON first: the form needs a valid document");
    mode = m;
    drawTabs();
    el.form.hidden = m !== "form";
    el.json.hidden = m !== "json";
    if (m === "json") {
      if (!editor)
        editor = createJsonEditor({
          parent: el.json,
          doc: pretty(spec),
          schema: schemas["cluster-spec"],
          dynamic: dyn,
          extraIssues: () => issues,
          onChange: (t) => {
            try {
              spec = JSON.parse(t);
              jsonOk = true;
            } catch (e) {
              jsonOk = false;
              issues = [{ path: [], message: `not valid JSON: ${(e as Error).message}` }];
              drawIssues();
              el.save.disabled = true;
              return;
            }
            changed(false);
          },
        });
      else editor.set(pretty(spec));
    } else drawForm();
  }

  // ---- fields
  const fieldBox = (path: Path, label: string, control: Node, help?: string | Node | null, wide = false) =>
    h("div", { class: `cf-field${wide ? " wide" : ""}`, "data-path": path.join(".") }, h("label", {}, label), control, help ? h("div", { class: "cf-help" }, help) : null, h("div", { class: "cf-err", role: "alert" }));
  const describe = (path: Path) => {
    // The schema's description of a field (the same text as the JSON tab's hover docs).
    let s: any = schemas["cluster-spec"];
    const root = s;
    const res = (x: any) => (x && x.$ref ? x.$ref.slice(2).split("/").reduce((a: any, k: string) => a?.[k], root) : x);
    for (const k of path) {
      s = res(s);
      if (!s) return "";
      if (typeof k === "number") s = res(s.items);
      else s = s.properties?.[k] ?? (s.anyOf || []).map(res).find((x: any) => x?.properties?.[k])?.properties?.[k];
    }
    s = res(s);
    return s?.description || (s?.anyOf || []).map(res).find((x: any) => x?.description)?.description || "";
  };
  function text(path: Path, o2: { placeholder?: string; list?: string; optional?: boolean; disabled?: boolean; mono?: boolean } = {}) {
    const sch = unwrap(schemaAt(schemas["cluster-spec"]!, path), schemas["cluster-spec"]!) || {};
    const inp = h("input", { type: "text", value: getAt(spec, path) ?? "", placeholder: o2.placeholder, list: o2.list, disabled: o2.disabled, spellcheck: "false", class: o2.mono ? "mono" : "", "aria-label": path.join("."), pattern: sch.pattern, maxlength: sch.maxLength });
    inp.addEventListener("input", () => {
      setAt(spec, path, inp.value === "" && o2.optional ? undefined : inp.value);
      changed();
    });
    return inp;
  }
  function num(path: Path, o2: { min?: number; max?: number; step?: number; optional?: boolean; scale?: number; placeholder?: string } = {}) {
    const sc = o2.scale ?? 1;
    const cur = getAt(spec, path);
    // The range comes from the schema (the server's own limits); the caller's step is a convenience.
    const sch = unwrap(schemaAt(schemas["cluster-spec"]!, path), schemas["cluster-spec"]!) || {};
    const lo = sch.minimum !== undefined ? sch.minimum / sc : o2.min;
    const hi = sch.maximum !== undefined && sch.maximum < 9e15 ? sch.maximum / sc : o2.max;
    const inp = h("input", { type: "number", value: cur === undefined || cur === null ? "" : String(cur / sc), min: lo, max: hi, step: o2.step ?? (sch.type === "integer" && sc === 1 ? 1 : "any"), placeholder: o2.placeholder, "aria-label": path.join("."), inputmode: "decimal" });
    inp.addEventListener("input", () => {
      if (inp.value === "") setAt(spec, path, o2.optional ? undefined : null);
      else setAt(spec, path, Math.round(Number(inp.value) * sc * 1000) / 1000);
      changed();
    });
    return inp;
  }
  /** A schema enum's values at a path (the coverage test: an enum is a select, never free text). */
  const enumOf = (path: Path): any[] => (unwrap(schemaAt(schemas["cluster-spec"]!, path), schemas["cluster-spec"]!)?.enum as any[]) || [];
  function select(path: Path, opts: [any, string][], o2: { optional?: boolean; onSet?: () => void; num?: boolean } = {}) {
    const cur = getAt(spec, path);
    const sel = h("select", { "aria-label": path.join(".") }, ...(o2.optional ? [h("option", { value: "" }, "(default)")] : []), ...opts.map(([v, l]) => h("option", { value: String(v) }, l)));
    // A value no longer offered (a channel that is gone, a variant renamed) stays visible, flagged, and fails the check.
    if (cur !== undefined && !opts.some(([v]) => v === cur)) sel.append(h("option", { value: String(cur), class: "ff-stale" }, `${cur} (not available: pick another)`));
    sel.value = cur === undefined ? "" : String(cur);
    sel.addEventListener("change", () => {
      setAt(spec, path, sel.value === "" && o2.optional ? undefined : o2.num ? Number(sel.value) : sel.value);
      changed();
      o2.onSet?.();
    });
    return sel;
  }
  function toggle(path: Path, label: string) {
    const cb = h("input", { type: "checkbox", checked: !!getAt(spec, path) });
    cb.addEventListener("change", () => {
      setAt(spec, path, cb.checked);
      changed();
    });
    return h("label", { class: "cf-toggle" }, cb, " ", label);
  }
  function segmented(path: Path, opts: [string, string, string?][], onSet?: () => void) {
    const box = h("div", { class: "cf-seg", role: "radiogroup" });
    const draw = () =>
      box.replaceChildren(
        ...opts.map(([v, l, title]) =>
          h(
            "button",
            {
              type: "button",
              role: "radio",
              "aria-checked": String(getAt(spec, path) === v),
              title,
              onclick: () => {
                setAt(spec, path, v);
                draw();
                changed();
                onSet?.();
              },
            },
            l,
          ),
        ),
      );
    draw();
    return box;
  }
  /** A multi-choice list (regions, GPU types, CPU flavors) kept in the order chosen (the order Runpod is tried in). */
  function multi(path: Path, opts: { id: string; label: string; detail?: string; kind?: string }[], o2: { optional?: boolean } = {}) {
    const box = h("div", { class: "cf-multi" });
    const draw = () => {
      const cur: string[] = getAt(spec, path) || [];
      const all = [...opts];
      for (const c of cur) if (!all.some((x) => x.id === c)) all.push({ id: c, label: c, detail: "unknown here" });
      box.replaceChildren(
        ...all.map((x) => {
          const i = cur.indexOf(x.id);
          return h(
            "button",
            {
              type: "button",
              class: `fv-chip${i >= 0 ? " on" : ""}`,
              "aria-pressed": String(i >= 0),
              title: x.detail || "",
              onclick: () => {
                const next = i >= 0 ? cur.filter((c) => c !== x.id) : [...cur, x.id];
                setAt(spec, path, next.length || !o2.optional ? next : undefined);
                draw();
                changed();
              },
            },
            i >= 0 && cur.length > 1 ? `${i + 1}. ` : "",
            x.label,
            x.detail ? h("span", { class: "muted" }, ` ${x.detail}`) : null,
          );
        }),
      );
    };
    draw();
    return box;
  }
  const datalist = (idd: string, items: { id: string; detail?: string }[]) => h("datalist", { id: idd }, ...items.map((x) => h("option", { value: x.id }, x.detail || "")));
  const section = (title: string, sub: string | null, ...kids: (Node | null)[]) => h("section", { class: "card cf-section" }, h("h2", {}, title), sub ? h("p", { class: "muted small" }, sub) : null, h("div", { class: "cf-fields" }, ...kids));

  // ---- the form
  const gpuOpts = () => (dyn.gpu_types || []).map((g: any) => ({ id: g.id, label: g.display || g.id, detail: `${g.secure_price != null ? `$${g.secure_price}/hr` : "no price"}${g.stock ? ` · ${g.stock}` : ""}` }));
  function drawForm() {
    const s = spec;
    const lists = h(
      "div",
      { hidden: true },
      datalist("cf-channels", (dyn.channels || []).map((c: any) => ({ id: c.id, detail: c.sha ? `at ${c.sha}` : "" }))),
      datalist("cf-shas", (dyn.shas || []).map((x: string) => ({ id: x }))),
      datalist("cf-config-paths", (dyn.config_paths || []).map((x: string) => ({ id: x }))),
    );
    const imgKind = s.image?.channel !== undefined ? "channel" : s.image?.sha !== undefined ? "sha" : s.image?.ref !== undefined ? "ref" : "channel";
    const imgInput = imgKind === "channel" ? select(["image", "channel"], (dyn.channels || []).map((c: any) => [c.id, `${c.id}${c.sha ? ` (at ${c.sha})` : ""}`])) : imgKind === "sha" ? text(["image", "sha"], { list: "cf-shas", placeholder: "git sha (7-40 hex)", mono: true }) : text(["image", "ref"], { placeholder: "ghcr.io/owner/repo:tag or @sha256:…", mono: true });
    const imgHelp =
      imgKind === "channel"
        ? `Per-variant images <variant>-${s.image?.channel || "<channel>"}${(dyn.channels || []).find((c: any) => c.id === s.image?.channel)?.sha ? `, now at ${(dyn.channels || []).find((c: any) => c.id === s.image?.channel).sha}` : ""}; resolved to digests at Start.`
        : imgKind === "sha"
          ? `Per-variant images <variant>-sha-${(s.image?.sha || "<sha>").slice(0, 7)}.`
          : "One all-in-one image for every pod.";
    const regionOpts = (dyn.regions || []).map((r: any) => ({ id: r.id, label: r.id, detail: `${r.dc} · volume ${r.volume} · ${(r.gpus || []).map((g: string) => g.replace(/^NVIDIA /, "").replace(/ Blackwell Server Edition$/, "")).join(", ")}` }));
    el.form.replaceChildren(
      lists,
      section(
        "Basics",
        null,
        fieldBox(["name"], "Name", text(["name"], { disabled: !!o.id, placeholder: "lower-case, e.g. ltx-eu" }), o.id ? "Fixed once defined: Clone to use another name." : describe(["name"])),
        !o.id
          ? fieldBox(
              [],
              "Start from",
              (() => {
                const sel = h("select", { "aria-label": "Template", "data-schema-ignore": "" }, h("option", { value: "" }, "(keep the current pools)"), ...(tpl.templates || []).map((t: any) => h("option", { value: t.id }, `${t.id}: ${t.title}`)));
                sel.addEventListener("change", () => {
                  if (!sel.value) return;
                  const name = spec.name;
                  spec = { ...clone(tpl[sel.value]), name };
                  openPools.clear();
                  drawForm();
                  changed();
                });
                return sel;
              })(),
              "A template replaces every field but the name.",
            )
          : null,
        fieldBox(["control_plane"], "Control plane", segmented(["control_plane"], [["edge", "edge", "The edge Worker is the only entry point"], ["direct", "direct", "Clients call each worker"]], () => drawForm()), describe(["control_plane"]), true),
        fieldBox(["auth"], "Client auth", segmented(["auth"], [["keys", "API keys"], ["none", "none"]]), s.control_plane === "direct" ? describe(["auth"]) : "An edge cluster uses the edge's own setting; this applies to direct clusters."),
      ),
      section(
        "Image",
        null,
        fieldBox(
          ["image"],
          "Source",
          (() => {
            const seg = h("div", { class: "cf-seg", role: "radiogroup" });
            for (const k of ["channel", "sha", "ref"] as const)
              seg.append(
                h(
                  "button",
                  {
                    type: "button",
                    role: "radio",
                    "aria-checked": String(k === imgKind),
                    onclick: () => {
                      const prev = spec.image || {};
                      spec.image = { [k]: prev[k] ?? (k === "channel" ? "stable" : "") };
                      drawForm();
                      changed();
                    },
                  },
                  k === "channel" ? "release channel" : k === "sha" ? "git commit" : "image ref",
                ),
              );
            return seg;
          })(),
        ),
        fieldBox(["image", imgKind], imgKind === "channel" ? "Channel" : imgKind === "sha" ? "Commit" : "Image reference", imgInput, imgHelp, true),
      ),
      section(
        "Placement",
        "Regions are tried in the order chosen; each brings its weights volume and GPU types.",
        fieldBox(["regions"], "Regions", multi(["regions"], regionOpts), describe(["regions"]), true),
      ),
      poolsSection(),
      section(
        "Deadline & money",
        "Backstops: the deadline deletes every pod; the floors refuse a start or stop the cluster.",
        fieldBox(
          ["cap_s"],
          "Deadline after Start (minutes)",
          h("div", { class: "row" }, num(["cap_s"], { min: 5, max: 10080, step: 5, scale: 60 }), ...[30, 60, 90, 120, 240].map((m) => h("button", { type: "button", class: "small", onclick: () => ((spec.cap_s = m * 60), drawForm(), changed()) }, m < 60 ? `${m}m` : `${m / 60}h`))),
          `${describe(["cap_s"])} Now: ${typeof s.cap_s === "number" ? dur(s.cap_s) : "–"}.`,
          true,
        ),
        fieldBox(["max_gpu_dph"], "Max GPU $/hr per pod", num(["max_gpu_dph"], { min: 0.1, max: 50, step: 0.05 }), describe(["max_gpu_dph"])),
        fieldBox(["min_start"], "Min balance to start ($)", num(["min_start"], { min: 8, step: 1 }), describe(["min_start"])),
        fieldBox(["balance_floor"], "Balance floor ($)", num(["balance_floor"], { min: 8, step: 1 }), describe(["balance_floor"])),
        fieldBox(["min_balance"], "Worker watchdog floor ($)", num(["min_balance"], { min: 8, step: 0.25 }), describe(["min_balance"])),
        fieldBox(
          ["auto_stop_idle_min"],
          "Idle auto-stop (minutes)",
          (() => {
            const cur = s.auto_stop_idle_min;
            const cb = h("input", { type: "checkbox", checked: cur === null || cur === undefined });
            const n = num(["auto_stop_idle_min"], { min: 5, max: 1440, step: 5, placeholder: "60" });
            n.disabled = cb.checked;
            cb.addEventListener("change", () => {
              spec.auto_stop_idle_min = cb.checked ? null : 60;
              drawForm();
              changed();
            });
            return h("div", { class: "row" }, h("label", { class: "cf-toggle" }, cb, " account policy"), n);
          })(),
          describe(["auto_stop_idle_min"]),
        ),
      ),
      section(
        "Logs",
        null,
        fieldBox(["log_shipping"], "Shipping", toggle(["log_shipping"], "pods ship their logs to fv-control (searchable on the Logs page)"), describe(["log_shipping"])),
        fieldBox(["log_level"], "Most verbose level shipped", select(["log_level"], enumOf(["log_level"]).map((l) => [l, l]), { optional: true }), "Default: info."),
      ),
    );
    markIssues();
  }

  function poolsSection(): HTMLElement {
    const box = h("div", { class: "cf-pools" });
    const st = new Map<string, any>((price?.availability?.pools || []).map((p: any) => [p.pool, p]));
    const dph = new Map<string, number>();
    for (const p of price?.pods || []) if (p.pool) dph.set(p.pool, p.dph);
    const running: Record<string, any[]> = cl?.cluster?.state?.workers || {};
    spec.pools.forEach((p: any, i: number) => box.append(poolCard(p, i, st.get(p.id), dph.get(p.id), running[p.id]?.length ?? 0)));
    const sel = h("select", { "aria-label": "Pool preset" }, h("option", { value: "" }, "add a pool…"), ...presets.map((p) => h("option", { value: p.id }, `${p.id}: ${p.title}${p.licence ? " (licence!)" : ""}`)), h("option", { value: "__blank" }, "blank GPU pool"), h("option", { value: "__cpu" }, "blank CPU pool (fake engine)"));
    sel.addEventListener("change", () => {
      const v = sel.value;
      if (!v) return;
      let pool: any;
      if (v === "__blank") pool = { id: uniqueId("pool"), variant: "ltx", count: 1, compute: "GPU", config: "/etc/fv/runpod-ltx.toml", models: [{ id: "ltx25-distill-sol", family: "ltx2", recipe: "ltx-turbo" }] };
      else if (v === "__cpu") pool = { id: uniqueId("fake"), variant: "cpu", count: 1, compute: "CPU", config_toml: (tpl["tiny-cpu"]?.pools?.[0]?.config_toml as string) || "", cpu_flavors: ["cpu3c"], vcpu: 2, volume: false, fake_models: ["fake-wan"] };
      else {
        const pr = presets.find((x) => x.id === v)!;
        if (pr.licence && !confirm(`${pr.title}: ${pr.licence}. Add it anyway?`)) return void (sel.value = "");
        pool = clone(pr.pool);
        if (spec.pools.some((x: any) => x.id === pool.id)) pool.id = uniqueId(pool.id);
      }
      spec.pools.push(pool);
      openPools.add(spec.pools.length - 1);
      drawForm();
      changed();
    });
    const sum = spec.pools.reduce((a: number, p: any) => a + (Number(p.count) || 0), 0);
    return h("section", { class: "card cf-section", "data-path": "pools" }, h("div", { class: "row" }, h("h2", { style: "margin:0" }, `Pools`), h("span", { class: "muted small" }, `${spec.pools.length} pool${spec.pools.length === 1 ? "" : "s"}, ${sum} worker${sum === 1 ? "" : "s"}`), h("span", { class: "spacer" }), sel), h("div", { class: "cf-err cf-err-block", "data-path": "pools" }), box);
  }
  const uniqueId = (base: string) => {
    let n = 2;
    let x = base;
    while (spec.pools.some((p: any) => p.id === x)) x = `${base}-${n++}`;
    return x;
  };

  /** "Scale to N" on a pool of a running cluster whose count differs from its workers (a scale operation). */
  function scaleBits(i: number): Node[] {
    const p = spec.pools[i];
    const runningN = cl?.cluster?.state?.workers?.[p?.id]?.length ?? 0;
    const ok = !!p && !!cl && cl.pods?.length > 0 && !!saved?.pools?.some((x: any) => x.id === p.id) && Number.isInteger(Number(p.count)) && Number(p.count) !== runningN && !cl.op;
    if (!ok) return [];
    return [h("button", { type: "button", class: "small", title: "Scale the running pool to this count (a scale operation)", onclick: () => runOp("scale", { pool: p.id, count: Number(p.count) }, `Scale ${p.id} from ${runningN} to ${p.count} worker(s)?${Number(p.count) < runningN ? " Workers are drained, then deleted." : ""}`) }, `Scale to ${p.count}`)];
  }
  const paintScale = () => spec.pools.forEach((_: any, i: number) => host.querySelector(`.cf-pool[data-path="pools.${i}"] .cf-scaleslot`)?.replaceChildren(...scaleBits(i)));
  function stockBits(stock: any, dphPerPod: number | undefined): Node[] {
    return kids(
      stock ? h("span", { class: `badge ${STOCK_KIND[stock.status] || ""}`, title: stock.hint }, stock.compute === "CPU" ? "CPU" : `stock: ${stock.status === "ok" ? "available" : stock.status}`) : null,
      dphPerPod !== undefined ? h("span", { class: "small muted", title: "estimated $/hr per pod (the most expensive GPU it may land on)" }, `${fmt$(dphPerPod)}/hr`) : null,
    ) as Node[];
  }
  const stockHint = (stock: any): Node[] => (stock && stock.status !== "ok" && stock.compute !== "CPU" ? [h("div", { class: `cf-stockhint ${STOCK_KIND[stock.status]}` }, stock.hint)] : []);
  /** The pools' stock badges and $/hr after a price answer, in place (focus stays in the field being edited). */
  function paintStock() {
    const st = new Map<string, any>((price?.availability?.pools || []).map((x: any) => [x.pool, x]));
    const dph = new Map<string, number>();
    for (const x of price?.pods || []) if (x.pool) dph.set(x.pool, x.dph);
    spec.pools.forEach((p: any, i: number) => {
      const card = host.querySelector<HTMLElement>(`.cf-pool[data-path="pools.${i}"]`);
      card?.querySelector(".cf-stockslot")?.replaceChildren(...stockBits(st.get(p.id), dph.get(p.id)));
      card?.querySelector(".cf-hintslot")?.replaceChildren(...stockHint(st.get(p.id)));
    });
  }
  function poolCard(p: any, i: number, stock: any, dphPerPod: number | undefined, runningN: number): HTMLElement {
    const P = (k: string): Path => ["pools", i, k];
    const open = openPools.has(i);
    const errs = issues.filter((x) => x.path[0] === "pools" && x.path[1] === i).length;
    const counter = (() => {
      const n = num(P("count"), { min: 0, max: 8, step: 1 });
      n.classList.add("cf-count");
      const bump = (d: number) => {
        spec.pools[i].count = Math.min(8, Math.max(0, (Number(spec.pools[i].count) || 0) + d));
        n.value = String(spec.pools[i].count);
        changed();
      };
      return h("span", { class: "cf-stepper", "data-path": P("count").join(".") }, h("button", { type: "button", "aria-label": "fewer", onclick: () => bump(-1) }, "−"), n, h("button", { type: "button", "aria-label": "more", onclick: () => bump(1) }, "+"));
    })();
    const move = (d: number) => {
      const j = i + d;
      if (j < 0 || j >= spec.pools.length) return;
      [spec.pools[i], spec.pools[j]] = [spec.pools[j], spec.pools[i]];
      const a = openPools.has(i);
      const b = openPools.has(j);
      openPools.delete(i), openPools.delete(j);
      if (a) openPools.add(j);
      if (b) openPools.add(i);
      drawForm();
      changed();
    };
    const head = h(
      "div",
      { class: "cf-pool-head" },
      h("button", { type: "button", class: "ghost cf-caret", "aria-expanded": String(open), onclick: () => (open ? openPools.delete(i) : openPools.add(i), drawForm()) }, open ? "▾" : "▸"),
      h("b", { class: "mono" }, p.id || "(no id)"),
      badge(`${p.variant || "?"} · ${p.compute || "GPU"}`),
      counter,
      h("span", { class: "muted small" }, "workers"),
      cl?.pods?.length ? h("span", { class: "small muted", title: "running now" }, `(${runningN} running)`) : null,
      h("span", { class: "cf-scaleslot" }, ...scaleBits(i)),
      h("span", { class: "cf-stockslot row" }, ...stockBits(stock, dphPerPod)),
      errs ? badge(`${errs} problem${errs === 1 ? "" : "s"}`, "critical") : null,
      h("span", { class: "spacer" }),
      h("button", { type: "button", class: "ghost small", title: "Move up", disabled: i === 0, onclick: () => move(-1) }, "↑"),
      h("button", { type: "button", class: "ghost small", title: "Move down", disabled: i === spec.pools.length - 1, onclick: () => move(1) }, "↓"),
      h("button", { type: "button", class: "ghost small", title: "Duplicate", onclick: () => { const c = clone(p); c.id = uniqueId(p.id); spec.pools.splice(i + 1, 0, c); openPools.add(i + 1); drawForm(); changed(); } }, "⧉"),
      h("button", { type: "button", class: "ghost small danger", title: "Remove the pool", onclick: () => { if (!confirm(`Remove pool ${p.id}?${runningN ? ` Its ${runningN} running worker(s) stay until scaled to 0.` : ""}`)) return; spec.pools.splice(i, 1); openPools.clear(); drawForm(); changed(); } }, "✕"),
    );
    const card = h("div", { class: `cf-pool${errs ? " has-err" : ""}`, "data-path": `pools.${i}` }, head);
    card.append(h("div", { class: "cf-hintslot" }, ...stockHint(stock)));
    if (!open) return card;
    const isCpu = p.compute === "CPU";
    const cfgKind = p.config_toml !== undefined ? "toml" : "file";
    const models = h(
      "div",
      { class: "cf-models" },
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, h("th", {}, "model id"), h("th", {}, "family"), h("th", {}, "recipe"), h("th", {}, ""))),
        h(
          "tbody",
          {},
          ...(p.models || []).map((_: any, j: number) =>
            h(
              "tr",
              { "data-path": `pools.${i}.models.${j}` },
              h(
                "td",
                { "data-path": `pools.${i}.models.${j}.id` },
                // A catalog model: choosing it fills its family and recipe (both checked against it by the server).
                select(["pools", i, "models", j, "id"], enumOf(["pools", 0, "models", 0, "id"]).map((m) => [m, m]), {
                  onSet: () => {
                    const m = (dyn.catalog_models || []).find((x: any) => x.id === spec.pools[i].models[j].id);
                    if (m) Object.assign(spec.pools[i].models[j], { family: m.family, recipe: m.recipe });
                    drawForm();
                    changed();
                  },
                }),
              ),
              h("td", { "data-path": `pools.${i}.models.${j}.family` }, select(["pools", i, "models", j, "family"], enumOf(["pools", 0, "models", 0, "family"]).map((x) => [x, x]))),
              h(
                "td",
                { "data-path": `pools.${i}.models.${j}.recipe` },
                select(
                  ["pools", i, "models", j, "recipe"],
                  enumOf(["pools", 0, "models", 0, "recipe"])
                    .filter((r) => !p.models[j]?.family || (dyn.recipes || []).find((x: any) => x.id === r)?.detail?.startsWith(p.models[j].family) !== false)
                    .map((r) => [r, r]),
                ),
              ),
              h("td", {}, h("button", { type: "button", class: "ghost small", title: "Remove", onclick: () => { spec.pools[i].models.splice(j, 1); if (!spec.pools[i].models.length) delete spec.pools[i].models; drawForm(); changed(); } }, "✕")),
            ),
          ),
        ),
      ),
      h("button", { type: "button", class: "small", onclick: () => { const m = (dyn.catalog_models || [])[0] || { id: "fasth3", family: "h3", recipe: "h3-turbo" }; (spec.pools[i].models ||= []).push({ ...m }); drawForm(); changed(); } }, "+ model"),
    );
    const fakeModels = multi(P("fake_models"), enumOf(["pools", 0, "fake_models", 0]).map((x) => ({ id: x, label: x })), { optional: true });
    const cfg = (() => {
      const seg = h("div", { class: "cf-seg", role: "radiogroup" });
      for (const [k, l] of [["file", "file in the image"], ["toml", "inline TOML"]] as const)
        seg.append(
          h(
            "button",
            {
              type: "button",
              role: "radio",
              "aria-checked": String(k === cfgKind),
              onclick: () => {
                const q = spec.pools[i];
                if (k === "file") {
                  delete q.config_toml;
                  q.config ||= "/etc/fv/runpod.toml";
                } else {
                  delete q.config;
                  q.config_toml ??= "";
                }
                drawForm();
                changed();
              },
            },
            l,
          ),
        );
      let ctl: HTMLElement;
      if (cfgKind === "file") ctl = text(P("config"), { placeholder: "/etc/fv/runpod.toml", mono: true, list: "cf-config-paths" });
      else {
        const ta = h("textarea", { class: "cf-toml", spellcheck: "false", rows: "10", "aria-label": `pools.${i}.config_toml` }, p.config_toml || "");
        ta.addEventListener("input", () => {
          spec.pools[i].config_toml = ta.value;
          changed();
        });
        ctl = ta;
      }
      return h("div", {}, seg, ctl);
    })();
    const volSel = (() => {
      const sel = h("select", { "aria-label": `pools.${i}.volume` }, h("option", { value: "" }, `default (${isCpu ? "off" : "on"})`), h("option", { value: "1" }, "mount the weights volume"), h("option", { value: "0" }, "no volume"));
      sel.value = p.volume === undefined ? "" : p.volume ? "1" : "0";
      sel.addEventListener("change", () => {
        setAt(spec, P("volume"), sel.value === "" ? undefined : sel.value === "1");
        changed();
      });
      return sel;
    })();
    const body = h(
      "div",
      { class: "cf-fields cf-pool-body" },
      fieldBox(P("id"), "Pool id", text(P("id"), { mono: true }), describe(["pools", 0, "id"])),
      fieldBox(P("variant"), "Image variant", select(P("variant"), (dyn.variants || []).map((v: any) => [v.id, `${v.id}${v.detail ? ` — ${v.detail}` : ""}`]), { onSet: () => drawForm() }), "The CI-built image this pool runs."),
      fieldBox(P("compute"), "Compute", segmented(P("compute"), [["GPU", "GPU"], ["CPU", "CPU"]], () => drawForm())),
      isCpu
        ? fieldBox(P("cpu_flavors"), "CPU flavors (in order)", multi(P("cpu_flavors"), (dyn.cpu_flavors || []).map((f: any) => ({ id: f.id, label: f.id, detail: `$${f.dph_per_vcpu}/vCPU·hr` })), { optional: true }), describe(["pools", 0, "cpu_flavors"]), true)
        : fieldBox(P("gpu_types"), "GPU types (in order; none: the regions')", multi(P("gpu_types"), gpuOpts(), { optional: true }), describe(["pools", 0, "gpu_types"]), true),
      fieldBox(P("regions"), "Regions (none: the cluster's)", multi(P("regions"), (dyn.regions || []).map((r: any) => ({ id: r.id, label: r.id, detail: r.dc })), { optional: true }), null, true),
      isCpu ? fieldBox(P("vcpu"), "vCPUs", select(P("vcpu") as Path, enumOf(["pools", 0, "vcpu"]).map((v) => [v, `${v} vCPU`]), { optional: true, num: true }), "A Runpod CPU instance size (default 2).") : null,
      fieldBox(P("container_disk_gb"), "Container disk (GB)", num(P("container_disk_gb"), { min: 5, max: 500, step: 5, optional: true, placeholder: isCpu ? "10" : "40" })),
      fieldBox(P("volume"), "Weights volume", volSel, "Mounted at /workspace."),
      fieldBox(P("image"), "Image override", text(P("image"), { optional: true, mono: true, placeholder: "(the variant's image)" }), describe(["pools", 0, "image"])),
      spec.control_plane !== "direct" ? fieldBox(P("family"), "Edge family", select(P("family"), enumOf(["pools", 0, "family"]).map((v) => [v, v]), { optional: true }), describe(["pools", 0, "family"])) : null,
      fieldBox(P("config"), "Worker config", cfg, "A path inside the image, or a whole worker config sent inline (FV_WORKER_TOML_B64).", true),
      fieldBox(P("models"), "Models", models, describe(["pools", 0, "models"]), true),
      fieldBox(P("fake_models"), "Fake-engine models", fakeModels, "The fake engine's models (CPU pools, tests).", true),
      fieldBox(P("max_queued"), "Max queued", num(P("max_queued"), { min: 0, max: 10000, step: 1, optional: true }), describe(["pools", 0, "max_queued"])),
      fieldBox(P("job_timeout_s"), "Job timeout (s)", num(P("job_timeout_s"), { min: 10, max: 86400, step: 10, optional: true })),
      fieldBox(P("stale_after_s"), "Stale after (s)", num(P("stale_after_s"), { min: 10, max: 86400, step: 10, optional: true }), describe(["pools", 0, "stale_after_s"])),
    );
    const pre = presets.find((x) => x.id === p.id);
    if (pre) card.append(h("p", { class: "muted small cf-preset" }, `Preset ${pre.id}: ${pre.description} Weights: ${pre.weights.join(", ")}.`, pre.licence ? h("span", {}, " ", badge("licence", "critical"), " ", pre.licence) : null));
    card.append(body);
    return card;
  }

  // ---- issues at their fields
  function markIssues() {
    for (const f of host.querySelectorAll<HTMLElement>(".cf-err")) f.textContent = "";
    for (const f of host.querySelectorAll<HTMLElement>(".has-err[data-path]")) if (!f.classList.contains("cf-pool")) f.classList.remove("has-err");
    const unplaced: Issue[] = [];
    for (const i of issues) {
      let placed = false;
      for (let n = i.path.length; n >= 0 && !placed; n--) {
        const key = i.path.slice(0, n).join(".");
        const box = host.querySelector<HTMLElement>(`.cf-form [data-path="${CSS.escape(key)}"]`);
        if (!box) continue;
        const errEl = box.classList.contains("cf-err") ? box : box.querySelector<HTMLElement>(":scope > .cf-err") || box.closest(".cf-field")?.querySelector<HTMLElement>(":scope > .cf-err");
        box.classList.add("has-err");
        if (errEl) {
          errEl.textContent = (errEl.textContent ? errEl.textContent + "; " : "") + (n < i.path.length ? `${i.path.slice(n).join(".")}: ` : "") + i.message;
          placed = true;
        }
      }
      if (!placed) unplaced.push(i);
    }
    return unplaced;
  }
  function drawIssues() {
    markIssues();
    const item = (i: Issue, kind: "err" | "warn") =>
      h(
        "li",
        { class: `cf-issue ${kind}`, onclick: () => focusPath(i.path) },
        h("code", {}, i.path.length ? i.path.join(".") : "spec"),
        " ",
        i.message,
      );
    el.issues.replaceChildren(...kids(
      issues.length ? h("p", { class: "small" }, badge(`${issues.length} problem${issues.length === 1 ? "" : "s"}`, "critical"), " the API would refuse this spec") : h("p", { class: "small" }, badge("valid", "good"), " the API's validator accepts it"),
      issues.length ? h("ul", { class: "cf-issuelist" }, ...issues.map((i) => item(i, "err"))) : null,
      warnings.length ? h("ul", { class: "cf-issuelist" }, ...warnings.map((i) => item(i, "warn"))) : null,
    ));
  }
  function focusPath(path: Path) {
    if (mode === "json") return editor?.view.focus();
    if (path[0] === "pools" && typeof path[1] === "number" && !openPools.has(path[1])) {
      openPools.add(path[1]);
      drawForm();
    }
    for (let n = path.length; n >= 0; n--) {
      const box = host.querySelector<HTMLElement>(`.cf-form [data-path="${CSS.escape(path.slice(0, n).join("."))}"]`);
      if (!box) continue;
      box.scrollIntoView({ block: "center", behavior: "smooth" });
      box.querySelector<HTMLElement>("input, select, textarea, button")?.focus({ preventScroll: true });
      return;
    }
  }

  // ---- change handling: validate (fast), price (slow), diff
  let vt: any;
  let pt: any;
  let vseq = 0;
  function changed(redrawJson = true) {
    if (redrawJson && editor && mode === "json") editor.set(pretty(spec));
    const dirty = !saved || pretty(spec) !== pretty(saved);
    el.dirty.textContent = saved ? (dirty ? "● unsaved changes" : "saved") : "";
    if (mode === "form") paintScale();
    drawChanges();
    clearTimeout(vt);
    vt = setTimeout(runValidate, 300);
  }
  async function runValidate() {
    const my = ++vseq;
    try {
      const r = await api("/api/clusters/validate", { method: "POST", body: { spec, ...(id ? { id } : {}) } });
      if (my !== vseq) return;
      issues = r.issues || [];
      warnings = r.warnings || [];
    } catch (e) {
      issues = [{ path: [], message: (e as Error).message }];
    }
    drawIssues();
    editor?.relint();
    el.save.disabled = issues.length > 0 || (!!saved && pretty(spec) === pretty(saved));
    // The price needs a valid spec; it asks Runpod, so it waits for a pause in typing.
    clearTimeout(pt);
    if (!issues.length) pt = setTimeout(runPrice, 900);
    // The pool cards show the problems' counts.
    if (mode === "form") for (const c of host.querySelectorAll<HTMLElement>(".cf-pool")) c.classList.toggle("has-err", issues.some((x) => x.path[0] === "pools" && `pools.${x.path[1]}` === c.dataset.path));
  }
  let pseq = 0;
  async function runPrice() {
    const my = ++pseq;
    priceBusy = true;
    drawPrice();
    try {
      const r = await api(`/api/clusters/${id ?? "new"}/price`, { method: "POST", body: { spec } });
      if (my !== pseq) return;
      price = r;
      priceErr = "";
    } catch (e) {
      if (my !== pseq) return;
      priceErr = (e as Error).message;
    }
    priceBusy = false;
    drawPrice();
    if (mode === "form") paintStock();
  }
  function drawPrice() {
    if (!price && priceBusy) return el.price.replaceChildren(h("p", { class: "muted small" }, "asking Runpod for prices and stock…"));
    if (!price) return el.price.replaceChildren(h("p", { class: "muted small" }, priceErr || (issues.length ? "Fix the problems to see a price." : "…")));
    const p = price;
    const a = p.availability;
    el.price.replaceChildren(
      h(
        "div",
        { class: `cf-pricebox${priceBusy ? " stale" : ""}` },
        h("div", { class: "cf-big" }, `${fmt$(p.cluster_dph)}/hr`, h("span", { class: "muted small" }, ` · ${fmt$(p.cluster_dph * p.hours)} for ${p.hours.toFixed(1)} h`)),
        h("dl", { class: "kv" }, h("dt", {}, "balance"), h("dd", {}, `${fmt$(p.balance)} (account ${fmt$(p.account_spend_per_hr)}/hr)`), h("dt", {}, "at the deadline"), h("dd", {}, `${fmt$(p.projected_balance)} (floor ${fmt$(p.floor, 0)})`)),
        p.ok ? h("p", { class: "small" }, badge("within the floor", "good")) : h("div", {}, ...(p.reasons || []).map((r: string) => h("p", { class: "small" }, badge("Start would be refused", "critical"), " ", r))),
        a
          ? h(
              "div",
              { class: "cf-stock" },
              h("h3", {}, "Runpod stock"),
              ...(a.warnings || []).map((w: string) => h("p", { class: "small cf-stockwarn" }, badge("competition", "warn"), " ", w)),
              h(
                "table",
                {},
                h(
                  "tbody",
                  {},
                  ...a.pools.map((x: any) =>
                    h(
                      "tr",
                      { title: x.hint },
                      h("td", { class: "mono" }, x.pool),
                      h("td", {}, `×${x.count}`),
                      h("td", {}, h("span", { class: `badge ${STOCK_KIND[x.status] || ""}` }, x.compute === "CPU" ? "CPU" : x.status === "ok" ? "available" : x.status)),
                      h("td", { class: "small muted wrap" }, x.compute === "CPU" ? "" : x.placements.map((pl: any) => `${pl.dc}: ${pl.stock ?? "none"}${pl.max_available !== null ? ` (${pl.max_available} free)` : ""}`).join("; ")),
                    ),
                  ),
                ),
              ),
              h("p", { class: "muted small" }, `As Runpod reports it (secure cloud, 1 GPU), ${new Date(a.at).toISOString().slice(11, 16)} UTC; stock moves by the minute.`),
            )
          : h("p", { class: "muted small" }, "Stock: Runpod did not answer."),
        h("div", { class: "row" }, h("button", { type: "button", class: "small", onclick: () => runPrice() }, "Refresh")),
      ),
    );
  }
  function drawChanges() {
    if (!saved) {
      el.changes.replaceChildren(h("p", { class: "muted small" }, "A new definition: Define saves it; nothing starts until Start."), h("button", { type: "button", class: "small", onclick: () => copyText(pretty(spec)).then(() => toast("spec copied")) }, "Copy JSON"));
      return;
    }
    el.changes.replaceChildren(renderDiff(saved, spec), h("p", { class: "muted small" }, `v${version} · `, h("a", { href: `#/cluster/${id}` }, "history and restore"), " on the cluster page"));
  }

  // ---- review and save
  function review() {
    if (issues.length) return toast("fix the problems first");
    const dlg = h("dialog", { class: "cf-review" });
    const plan = h("div", { class: "fv-plan" }, saved ? h("span", { class: "muted small" }, "planning…") : null);
    const go = h("button", { type: "button", class: "primary" }, saved ? `Save (v${version} → v${version + 1})` : "Define");
    dlg.append(...kids(
      h("h2", {}, saved ? `Save ${spec.name}` : `Define ${spec.name || "(no name)"}`),
      h("h3", {}, saved ? "Changes (saved → yours)" : "The new spec"),
      saved ? renderDiff(saved, spec) : h("pre", { class: "fv-diff" }, pretty(spec)),
      saved ? h("h3", {}, "What it does") : null,
      plan,
      h("div", { class: "row", style: "margin-top:10px" }, go, h("button", { type: "button", onclick: () => dlg.close() }, "Cancel")),
    ));
    go.addEventListener("click", async () => {
      go.disabled = true;
      try {
        if (!saved) {
          const r = await api("/api/clusters", { method: "POST", body: { spec } });
          dlg.close();
          toast(`defined ${r.cluster.name}`);
          location.hash = `#/cluster/${r.cluster.id}/config`;
          return;
        }
        const r = await api(`/api/docs/cluster-spec/${id}`, { method: "PUT", body: { doc: spec, version } });
        saved = r.doc;
        version = r.version;
        spec = clone(saved);
        dlg.close();
        toast(`saved (v${version})`);
        await refreshCluster();
        drawForm();
        changed();
      } catch (e) {
        const msg = (e as Error).message;
        go.disabled = false;
        if (/changed since you loaded/.test(msg)) {
          plan.replaceChildren(h("p", { class: "fv-conflict" }, "Someone saved this cluster after you loaded it. Yours was not saved."), h("button", { type: "button", class: "danger", onclick: () => location.reload() }, "Reload (discard my edit)"));
        } else toast(`save failed: ${msg}`);
      }
    });
    dlg.addEventListener("close", () => dlg.remove());
    document.body.append(dlg);
    dlg.showModal();
    if (saved)
      api(`/api/docs/cluster-spec/${id}/plan`, { method: "POST", body: { doc: spec } })
        .then((p) => {
          const pr = p.projection || {};
          plan.replaceChildren(
            h("ul", {}, ...(p.actions || []).map((a: any) => h("li", { class: `plan-${a.action}` }, h("b", {}, a.action), ` ${a.target}: ${a.detail}`))),
            ...(p.warnings || []).map((w: string) => h("p", { class: "small muted" }, `note: ${w}`)),
            h("p", { class: "small" }, `$/hr ${fmt$(p.dph_now)} now → ~${fmt$(pr.cluster_dph)} after.`),
            h("p", { class: "small muted" }, "Saving changes the definition only; a running cluster gets it through Scale, Roll or a restart."),
          );
        })
        .catch((e) => plan.replaceChildren(h("p", { class: "small" }, (e as Error).message)));
  }

  // ---- the cluster: status, operations, live log
  async function refreshCluster() {
    if (!id) return;
    cl = await api(`/api/clusters/${id}`);
    if (cl.op?.id) watchOp = cl.op.id;
    drawOps();
  }
  async function runOp(path: string, body: any, question: string) {
    if (!confirm(question)) return;
    try {
      const r = await api(`/api/clusters/${id}/${path}`, { method: "POST", body });
      watchOp = r.operation || null;
      toast(`${path}: started`);
      await refreshCluster();
      pollOp();
    } catch (e) {
      toast(`${path}: ${(e as Error).message}`);
    }
  }
  function drawOps() {
    if (!id) {
      el.ops.replaceChildren(h("p", { class: "muted small" }, "Define saves the spec; then Start runs the price check and creates the pods."));
      return;
    }
    const c = cl.cluster;
    const running = (cl.pods || []).length > 0;
    const op = cl.op;
    const dirty = pretty(spec) !== pretty(saved);
    el.title.replaceChildren(c.name, " ", badge(c.status, statusKind(c.status)));
    el.sub.textContent = `${c.spec.control_plane === "direct" ? "direct" : "edge"} · ${(cl.pods || []).length} pod(s) · deadline ${c.deadline ? `${new Date(c.deadline).toISOString().slice(0, 16)}Z (${until(c.deadline)})` : "–"}${op ? ` · running: ${op.kind} (${op.phase})` : ""}`;
    const b = (label: string, fn: () => void, cls = "", dis = false) => h("button", { type: "button", class: cls, disabled: dis || (!!op && label !== "Cancel operation"), onclick: fn }, label);
    el.ops.replaceChildren(...kids(
      h("div", { class: "row" }, badge(c.status, statusKind(c.status)), op ? badge(`${op.kind}: ${op.phase}`, "warn") : null),
      h(
        "div",
        { class: "row", style: "margin-top:8px" },
        !running ? b("Start", () => runOp("start", {}, `Start ${c.name}? Price check: ${price ? `${fmt$(price.cluster_dph)}/hr, ${price.ok ? "within the floor" : "REFUSED: " + price.reasons.join("; ")}` : "runs first"}.${dirty ? "\n\nYour unsaved edits are NOT used: Start uses the saved spec." : ""}`), "primary", dirty) : null,
        running ? b("Stop", () => runOp("stop", {}, `Stop ${c.name}: delete every pod?`), "danger") : null,
        running ? b("Extend…", async () => { const m = await askExtend(api, `Extend ${c.name}`); if (m) runOp("extend", { minutes: m }, `Extend ${c.name} by ${m} min?`); }) : null,
        op ? b("Cancel operation", () => runOp("cancel", {}, "Cancel the running operation? Pods it created stay (the backstops hold them).")) : null,
        b("Clone", () => (location.hash = `#/clusters/new?clone=${id}`)),
        !running ? b("Delete", async () => { if (!confirm(`Delete the definition of ${c.name}? (audited)`)) return; try { await api(`/api/clusters/${id}`, { method: "DELETE" }); toast("deleted"); location.hash = "#/clusters"; } catch (e) { toast((e as Error).message); } }, "danger") : null,
      ),
      dirty && !running ? h("p", { class: "muted small" }, "Save first: Start uses the saved spec.") : null,
    ));
  }
  let opTimer: any = null;
  async function pollOp() {
    clearTimeout(opTimer);
    if (!watchOp) return;
    try {
      const r = await api(`/api/ops/${watchOp}`);
      const op = r.operation;
      const live = op.status === "running";
      el.oplog.replaceChildren(
        h(
          "section",
          { class: "card cf-card" },
          h("h2", {}, `Operation: ${op.kind} `, badge(op.status, statusKind(op.status))),
          h("p", { class: "muted small" }, `${op.id} · by ${op.actor} · `, h("a", { href: `#/logs?src=op&op=${op.id}&lv=trace,debug,info,warn,error&from=7d` }, "open in the log explorer")),
          h("div", { class: "log cf-oplines" }, ...op.log.map((l: any) => h("div", { class: /no stock|WARNING|fail|error|refused/i.test(l.msg) ? "lv-warn" : "" }, `${new Date(l.at).toISOString().slice(11, 19)}  ${l.msg}`)), op.error ? h("div", { class: "lv-error" }, `error: ${op.error}`) : null),
        ),
      );
      const box = el.oplog.querySelector(".cf-oplines");
      if (box) box.scrollTop = box.scrollHeight;
      if (live) opTimer = setTimeout(pollOp, 1500);
      else {
        await refreshCluster();
        const a = document.activeElement as HTMLElement | null;
        if (mode === "form" && !(a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA"))) drawForm();
        loadEnv();
      }
    } catch {
      opTimer = setTimeout(pollOp, 4000);
    }
  }
  o.onCleanup(() => (clearTimeout(opTimer), clearTimeout(vt), clearTimeout(pt), editor?.destroy()));

  // ---- effective env
  async function loadEnv() {
    if (!id) return;
    try {
      const e = await api(`/api/clusters/${id}/env`);
      const table = (env: any[]) =>
        h(
          "table",
          { class: "cf-envtable" },
          h("thead", {}, h("tr", {}, h("th", {}, "name"), h("th", {}, "value"), h("th", {}, "source"))),
          h(
            "tbody",
            {},
            ...env.map((x: any) =>
              h(
                "tr",
                { class: x.overrides ? "env-override" : "" },
                h("td", {}, h("code", {}, x.key)),
                h("td", { class: "wrap" }, x.secret ? h("span", { class: "muted" }, "•••••••• secret") : h("code", {}, x.value)),
                h("td", {}, badge(x.source, x.source === "pod" ? "serious" : x.source === "pool" || x.source === "cluster" ? "warn" : x.source === "account" ? "good" : ""), x.overrides ? h("span", { class: "muted small" }, ` over ${x.overrides.join(", ")}`) : null),
              ),
            ),
          ),
        );
      const parts: Node[] = [h("p", { class: "muted small" }, "Resolution: pod > pool > cluster > account > the controller's own keys. ", h("a", { href: `#/env?cluster=${id}` }, "Edit env levels →"))];
      if (e.needs_restart?.length) parts.push(h("p", { class: "small" }, badge(`${e.needs_restart.length} pod(s) need a restart to get it`, "warn")));
      if (e.pods.length)
        for (const p of e.pods)
          parts.push(h("details", {}, h("summary", {}, `${p.pool || p.role} `, h("code", {}, p.pod_id), " ", p.needs_restart ? badge("needs restart", "warn") : badge("applied", "good"), h("span", { class: "muted small" }, ` ${p.env.length} variables`)), table(p.env)));
      else {
        parts.push(h("p", { class: "muted small" }, "No pods: what a new worker of each pool would get (the saved spec)."));
        for (const [k, env] of Object.entries<any[]>(e.preview || {})) parts.push(h("details", {}, h("summary", {}, `pool ${k}`, h("span", { class: "muted small" }, ` ${env.length} variables`)), table(env)));
      }
      el.env.replaceChildren(...parts);
    } catch (e) {
      el.env.replaceChildren(h("p", { class: "small" }, (e as Error).message));
    }
  }

  // ---- go
  if (!o.id) el.title.replaceChildren(o.clone ? "Clone a cluster" : "New cluster");
  drawTabs();
  drawForm();
  drawOps();
  drawPrice();
  changed();
  loadEnv();
  if (watchOp) pollOp();
}
