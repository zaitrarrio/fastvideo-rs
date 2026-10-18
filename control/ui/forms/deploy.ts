// One-click deploy (#/deploy; owner 2026-10-18): pick model families (or all of them), for each its pools (a preset
// of the family or one of its templates: h3, fasth3, streaming, …) and its instance types with the launch form's
// instance-type table (itable.ts: Runpod an ordered multi-pick with fallbacks, GMI / Brev one product), then a new
// cluster or an existing one. "Review" asks the server for the plan (POST /api/v1/deploy/plan: the cluster page's
// checks, nothing saved) and shows what will be created; "Deploy" applies exactly that (POST /api/v1/deploy/apply).
import { badge, h } from "../dom";
import type { Api } from "../fields";
import { loadDyn, loadProvider, mergeProvider } from "./common";
import { instanceTable, type InstanceTable, type ITRow } from "./itable";

interface FamilyOpt {
  id: string;
  detail: string;
  default: string;
  presets: { id: string; title: string; min_vram_gb: number; licence: string | null; models: string[]; members?: string[]; left_out?: { what: string; why: string }[] }[];
  /** A family template: ONE pool (one pod, one process) serving every variant (owner 2026-10-18). */
  templates: { id: string; title: string; pools: string[]; models?: string[]; left_out?: { what: string; why: string }[] }[];
}
/** The pools field's hint: a family pool's models and what it leaves out; several pools; or one preset. */
function poolsHint(tp: FamilyOpt["templates"][number] | null | undefined, pr: FamilyOpt["presets"][number] | undefined): string {
  const out = (xs?: { what: string; why: string }[]) => (xs?.length ? ` Not on it: ${xs.map((x) => `${x.what} (${x.why})`).join("; ")}.` : "");
  if (tp && tp.pools.length === 1) return `ONE pod (one GPU, one process) serving every variant: ${(tp.models || []).length} models (${(tp.models || []).join(", ")}); the first resident, the others loaded on their first request.${out(tp.left_out)}`;
  if (tp) return `${tp.pools.length} pools: ${tp.pools.join(", ")} (one GPU per pod each).`;
  if (pr?.members) return `ONE pod serving every variant: ${pr.models.length} models; the first resident, the others loaded on their first request.${out(pr.left_out)}`;
  return "One pool of this preset (its worker config and models).";
}
interface Catalog {
  families: FamilyOpt[];
  clusters: { id: string; name: string; status: string; control_plane: string; pools: string[]; running: boolean }[];
}
interface Fam {
  on: boolean;
  pools: string;
  count?: number;
  provider: string;
  weights_source?: string;
  hub_download_approved?: boolean;
  gpu_types: string[];
  data_centers?: string[];
  provider_gpu?: string;
  provider_region?: string;
  container_disk_gb?: number;
}

const fmt$ = (v: unknown) => (typeof v === "number" && isFinite(v) ? `$${v.toFixed(2)}` : "–");

export async function mountDeployForm(host: HTMLElement, o: { api: Api; toast: (m: string) => void }) {
  const { api } = o;
  const [cat, dyn] = await Promise.all([api("/api/v1/deploy/catalog") as Promise<Catalog>, loadDyn(api, "", { live: false })]);
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  const st = {
    target: (q.get("cluster") ? "existing" : "new") as "new" | "existing",
    name: "",
    cluster_id: q.get("cluster") || cat.clusters[0]?.id || "",
    control_plane: "edge",
    all: false,
    start: true,
    fams: new Map<string, Fam>(cat.families.map((f) => [f.id, { on: false, pools: f.default, provider: "runpod", gpu_types: [] }])),
  };
  const pre = q.get("family");
  if (pre && st.fams.has(pre)) st.fams.get(pre)!.on = true;
  const tpl = q.get("template");
  if (tpl) for (const f of cat.families) if (f.templates.some((t) => t.id === tpl)) Object.assign(st.fams.get(f.id)!, { on: true, pools: `template:${tpl}` });
  let plan: any = null;
  let planned = "";
  let busy = false;
  let result: any = null;

  // ---- the instance-type tables: one per family and provider, kept across redraws (their sort, filters and page).
  const tables = new Map<string, InstanceTable>();
  const presetWeights = (f: Fam): string => {
    const first = f.pools.startsWith("template:") ? cat.families.flatMap((x) => x.templates).find((t) => `template:${t.id}` === f.pools)?.pools[0] : f.pools;
    return (dyn.pool_presets || []).find((p: any) => p.id === first)?.weights_source || "volume";
  };
  const runpodWeights = (f: Fam) => f.weights_source || presetWeights(f);
  const freeDc = (f: Fam) => runpodWeights(f) !== "volume";
  const volumeDc = () => (dyn.regions || [])[0]?.dc ?? "EU-RO-1";
  const dcsTried = (f: Fam) => (freeDc(f) ? f.data_centers ?? [volumeDc()] : [volumeDc()]);
  const withDc = (dcs: string[], dc: string) => (dcs.includes(dc) ? dcs : [...dcs.filter((d) => d !== "any"), dc, ...(dcs.includes("any") ? ["any"] : [])]);
  function tableFor(fid: string, p: string): InstanceTable {
    const key = `${fid}|${p}`;
    let t = tables.get(key);
    if (t) return t;
    const fam = () => st.fams.get(fid)!;
    t = instanceTable({
      api,
      provider: p,
      noun: p === "gmi" ? "product" : p === "runpod" ? "GPU type" : "instance type",
      isSelected: (r: ITRow) => (p === "runpod" ? fam().gpu_types.includes(r.name) && !!r.region && dcsTried(fam()).includes(r.region) : fam().provider_gpu === r.name && (p !== "gmi" || !fam().provider_region || fam().provider_region === r.region)),
      ...(p === "runpod"
        ? {
            weights: () => runpodWeights(fam()),
            multi: {
              items: () => fam().gpu_types.map((g) => ({ key: g, label: g, detail: `in ${dcsTried(fam()).map((d) => (d === "any" ? "any data centre" : d)).join(", ")}` })),
              order: (r: ITRow) => fam().gpu_types.indexOf(r.name) + 1 || null,
              move: (i: number, d: -1 | 1) => {
                const g = [...fam().gpu_types];
                if (i + d < 0 || i + d >= g.length) return;
                [g[i], g[i + d]] = [g[i + d]!, g[i]!];
                fam().gpu_types = g;
                changed(true);
              },
              remove: (i: number) => {
                fam().gpu_types = fam().gpu_types.filter((_, j) => j !== i);
                changed(true);
              },
            },
          }
        : {}),
      onSelect: (r, how) => {
        const f = fam();
        if (p === "runpod") {
          if (!r) (f.gpu_types = []), freeDc(f) && delete f.data_centers;
          else if (how?.append) {
            if (!f.gpu_types.includes(r.name)) f.gpu_types = [...f.gpu_types, r.name];
            if (freeDc(f) && r.region) f.data_centers = withDc(f.data_centers ?? [], r.region);
          } else {
            f.gpu_types = [r.name];
            if (freeDc(f)) f.data_centers = r.region ? [r.region] : undefined;
          }
        } else {
          f.provider_gpu = r ? r.name : undefined;
          if (p === "gmi") f.provider_region = r ? (r.launch?.provider_region as string | undefined) : undefined;
          // A fixed-disk Brev type takes no disk size.
          if (p === "brev" && r?.storage_kind === "fixed") delete f.container_disk_gb;
        }
        changed(true);
      },
    });
    tables.set(key, t);
    return t;
  }
  function pickProvider(fid: string, p: string) {
    for (const [k, t] of tables) if (k.startsWith(`${fid}|`) && k !== `${fid}|${p}`) t.abort();
    void tableFor(fid, p).load(false);
    if (p !== "runpod")
      loadProvider(api, p)
        .then((d) => (mergeProvider(dyn, d), draw()))
        .catch(() => {});
  }

  // ---- the page
  const body = h("div", { class: "dp" });
  const planBox = h("div", { "data-deploy-plan": "" });
  /** A change: the plan shown no longer matches (Deploy waits for a new Review); `redraw` rebuilds the controls. */
  function changed(redraw = false) {
    if (planned && planned !== JSON.stringify(request())) {
      plan = null;
      planned = "";
      drawPlan();
    }
    if (redraw) draw();
  }
  function request() {
    const families = [...st.fams]
      .filter(([, f]) => f.on)
      .map(([family, f]) => {
        const other = f.provider !== "runpod";
        return {
          family,
          pools: f.pools,
          ...(f.count !== undefined ? { count: f.count } : {}),
          ...(other ? { provider: f.provider } : {}),
          ...(f.weights_source ? { weights_source: f.weights_source } : {}),
          ...(f.weights_source === "hub" && f.hub_download_approved ? { hub_download_approved: true } : {}),
          ...(!other && f.gpu_types.length ? { gpu_types: f.gpu_types } : {}),
          ...(!other && freeDc(f) && f.data_centers?.length ? { data_centers: f.data_centers } : {}),
          ...(other && f.provider_gpu ? { provider_gpu: f.provider_gpu } : {}),
          ...(f.provider === "gmi" && f.provider_region ? { provider_region: f.provider_region } : {}),
          ...(f.provider === "brev" && f.container_disk_gb !== undefined ? { container_disk_gb: f.container_disk_gb } : {}),
        };
      });
    return {
      ...(st.target === "new" ? { name: st.name.trim(), control_plane: st.control_plane } : { cluster_id: st.cluster_id }),
      ...(st.all ? { all_families: true } : {}),
      families,
    };
  }
  const field = (label: string, ctl: Node, help?: string, attrs: Record<string, string> = {}) => h("div", { class: "cf-field", ...attrs }, h("label", {}, label), ctl, help ? h("span", { class: "cf-help" }, help) : null);
  const sel = (opts: [string, string][], cur: string, on: (v: string) => void, aria: string) => {
    const s = h("select", { "aria-label": aria }, ...opts.map(([v, l]) => h("option", { value: v }, l)));
    s.value = cur;
    s.addEventListener("change", () => on(s.value));
    return s;
  };
  function targetCard() {
    const seg = h(
      "div",
      { class: "cf-seg", role: "radiogroup", "aria-label": "Target cluster" },
      ...(["new", "existing"] as const).map((k) =>
        h("button", { type: "button", role: "radio", "aria-checked": String(st.target === k), "data-deploy-target": k, disabled: k === "existing" && !cat.clusters.length, onclick: () => ((st.target = k), changed(true)) }, k === "new" ? "New cluster" : "Existing cluster"),
      ),
    );
    const nameIn = h("input", { type: "text", value: st.name, placeholder: "e.g. h3-all", "aria-label": "Cluster name", "data-deploy-name": "" }) as HTMLInputElement;
    nameIn.addEventListener("input", () => ((st.name = nameIn.value), changed()));
    const cl = cat.clusters.find((c) => c.id === st.cluster_id);
    return h(
      "section",
      { class: "card" },
      h("h2", {}, "1. Cluster"),
      h("div", { class: "cf-fields" }, field("Target", seg, "A new cluster gets one pool per family (a template: its pools); an existing one gets the pools added (ids made unique)."), ...(st.target === "new"
        ? [
            field("Name", nameIn, "Unique among clusters and standalone pods."),
            field("Control plane", sel([["edge", "edge: the edge Worker fronts it (one edge cluster at a time)"], ["direct", "direct: clients call each worker"]], st.control_plane, (v) => ((st.control_plane = v), changed()), "Control plane")),
          ]
        : [
            field(
              "Cluster",
              sel(cat.clusters.map((c) => [c.id, `${c.name} · ${c.status}${c.running ? " (running)" : ""} · ${c.pools.length} pool(s)`]), st.cluster_id, (v) => ((st.cluster_id = v), changed(true)), "Existing cluster"),
              cl ? `Its pools: ${cl.pools.join(", ") || "none"}.${cl.running ? " Running: the new pools are saved and get their workers through Scale." : " Stopped: Deploy starts it with every pool."}` : undefined,
            ),
          ]),
      ),
    );
  }
  function familiesCard() {
    const all = h("input", { type: "checkbox", checked: st.all, "data-deploy-all": "" }) as HTMLInputElement;
    all.addEventListener("change", () => {
      st.all = all.checked;
      for (const [fid, f] of st.fams) if (st.all && !f.on) (f.on = true), pickProvider(fid, f.provider);
      changed(true);
    });
    return h(
      "section",
      { class: "card" },
      h("h2", {}, "2. Model families"),
      h(
        "div",
        { class: "row", style: "flex-wrap:wrap;gap:14px" },
        h("label", { class: "cf-toggle" }, all, h("b", {}, "All families")),
        ...cat.families.map((f) => {
          const cb = h("input", { type: "checkbox", checked: st.fams.get(f.id)!.on, disabled: st.all, "data-deploy-family": f.id }) as HTMLInputElement;
          cb.addEventListener("change", () => {
            const x = st.fams.get(f.id)!;
            x.on = cb.checked;
            if (x.on) pickProvider(f.id, x.provider);
            changed(true);
          });
          return h("label", { class: "cf-toggle", title: f.detail }, cb, h("span", { class: "mono" }, f.id), h("span", { class: "muted small" }, ` ${f.detail}`));
        }),
      ),
    );
  }
  function familyCard(fo: FamilyOpt) {
    const f = st.fams.get(fo.id)!;
    const other = f.provider !== "runpod";
    const poolsSel = h("select", { "aria-label": `${fo.id} pools`, "data-deploy-pools": fo.id });
    poolsSel.append(
      h("optgroup", { label: "One pool (a preset)" }, ...fo.presets.map((p) => h("option", { value: p.id }, `${p.id}: ${p.title}${p.licence ? " (licence!)" : ""}${p.id === fo.default ? " (default)" : ""}`))),
      ...(fo.templates.length ? [h("optgroup", { label: "A family template (one pod, every variant)" }, ...fo.templates.map((t) => h("option", { value: `template:${t.id}` }, `template ${t.id}: ${t.title}`)))] : []),
    );
    poolsSel.value = f.pools;
    poolsSel.addEventListener("change", () => ((f.pools = poolsSel.value), changed(true)));
    const tp = f.pools.startsWith("template:") ? fo.templates.find((t) => `template:${t.id}` === f.pools) : null;
    const countIn = h("input", { type: "number", min: "0", max: "8", step: "1", value: f.count ?? "", placeholder: "1 (the preset's)", "aria-label": `${fo.id} workers per pool` }) as HTMLInputElement;
    countIn.addEventListener("input", () => ((f.count = countIn.value === "" ? undefined : Number(countIn.value)), changed()));
    const provOpts: [string, string][] = (dyn.providers || [{ id: "runpod", detail: "" }]).map((p: any) => [p.id, `${p.id}${p.detail ? ` — ${p.detail}` : ""}`]);
    const provSel = sel(provOpts, f.provider, (v) => {
      Object.assign(f, { provider: v, gpu_types: [], data_centers: undefined, provider_gpu: undefined, provider_region: undefined, container_disk_gb: undefined, weights_source: undefined });
      pickProvider(fo.id, v);
      changed(true);
    }, `${fo.id} provider`);
    provSel.setAttribute("data-deploy-provider", fo.id);
    const weights = other
      ? [["", "default: r2 when every tree is published, else none"], ["r2", "r2 — our R2 weights store"], ["hub", "hub — download at boot (owner approval)"], ["none", "none — fake engine"]]
      : [["", `default: ${presetWeights(f)} (the preset's)`], ["volume", "volume — the EU weights volume (EU-RO-1)"], ["r2", "r2 — our R2 weights store: any data centre"], ["hub", "hub — download at boot (owner approval): any data centre"], ["none", "none — fake engine: any data centre"]];
    const wSel = sel(weights as [string, string][], f.weights_source || "", (v) => {
      f.weights_source = v || undefined;
      if (!freeDc(f)) delete f.data_centers;
      changed(true);
    }, `${fo.id} weights`);
    const t = tableFor(fo.id, f.provider);
    const brevRow = f.provider === "brev" && f.provider_gpu ? t.view()?.rows.find((r) => r.name === f.provider_gpu) : null;
    const brevD = (dyn.provider_gpus || []).find((g: any) => g.provider === "brev" && g.id === f.provider_gpu)?.disk;
    const fixed = brevRow?.storage_kind === "fixed" ? brevRow.storage_gb : brevD?.kind === "fixed" ? brevD.gb : null;
    const diskIn = h("input", { type: "number", min: "5", max: "16384", step: "10", value: f.container_disk_gb ?? "", placeholder: fixed ? `${fixed} (fixed)` : "sized for the weights", "aria-label": `${fo.id} disk`, "data-deploy-disk": fo.id }) as HTMLInputElement;
    diskIn.addEventListener("input", () => ((f.container_disk_gb = diskIn.value === "" ? undefined : Number(diskIn.value)), changed()));
    const diskHelp = fixed
      ? `${f.provider_gpu} has a fixed ${fixed} GB disk (Brev does not size it): leave it unset.`
      : brevRow?.storage_kind === "elastic" || brevD?.kind === "elastic"
        ? `${f.provider_gpu}: an elastic disk of ${brevRow?.storage_min_gb ?? brevD?.min_gb ?? "?"}–${brevRow?.storage_max_gb ?? brevD?.max_gb ?? "?"} GB; at least the weights + 40 GB. Unset: sized for the weights.`
        : "The Brev VM's disk. Unset: fv-control sizes it for the weights (the trees × 1.3 + 40 GB, at least 200 GB).";
    const issuesHere = (plan?.issues || []).filter((i: any) => i.path[0] === "families" && request().families[i.path[1]]?.family === fo.id);
    return h(
      "section",
      { class: `card dp-family${issuesHere.length ? " has-err" : ""}`, "data-deploy-family-card": fo.id },
      h("h2", {}, h("span", { class: "mono" }, fo.id), " ", h("span", { class: "muted small" }, fo.detail)),
      h(
        "div",
        { class: "cf-fields" },
        field("Pools", poolsSel, poolsHint(tp, fo.presets.find((p) => p.id === f.pools)), { "data-deploy-pools-hint": fo.id }),
        field("Workers per pool", countIn),
        field("Provider", provSel, "Its instance types load below."),
        field("Weights", wSel, other ? "No Runpod volume on GMI / Brev." : "volume pins the pods to EU-RO-1; r2 / hub / none run in any data centre (the table's data centres)."),
        ...(f.weights_source === "hub"
          ? [
              field(
                "Hub download",
                (() => {
                  const cb = h("input", { type: "checkbox", checked: !!f.hub_download_approved }) as HTMLInputElement;
                  cb.addEventListener("change", () => ((f.hub_download_approved = cb.checked), changed()));
                  return h("label", { class: "cf-toggle" }, cb, "the owner approved this download");
                })(),
              ),
            ]
          : []),
        ...(f.provider === "brev" ? [field("Disk (GB, the Brev VM)", diskIn, diskHelp)] : []),
      ),
      t.el,
      h(
        "p",
        { class: "small", "data-deploy-picked": fo.id },
        h("b", {}, "Instance types: "),
        other ? f.provider_gpu || h("span", { class: "muted" }, "none picked (click a row)") : f.gpu_types.length ? `${f.gpu_types.join(" → ")}${freeDc(f) ? ` in ${dcsTried(f).join(" → ")}` : ` in ${volumeDc()}`}` : h("span", { class: "muted" }, "the region's default GPU types (click rows to choose, \"+ fallback\" for more)"),
      ),
      issuesHere.length ? h("ul", { class: "cf-err-block" }, ...issuesHere.map((i: any) => h("li", { class: "cf-err" }, i.message))) : null,
    );
  }
  function drawPlan() {
    if (result) {
      const c = result.cluster;
      planBox.replaceChildren(
        h(
          "section",
          { class: "card", "data-deploy-result": "" },
          h("h2", {}, "Deployed"),
          h("p", {}, badge("ok", "good"), " ", h("a", { href: `#/cluster/${c.id}` }, c.name), ` · pools ${result.pools.map((p: any) => p.id).join(", ")}`),
          result.operation ? h("p", { class: "small" }, "Operation ", h("code", {}, result.operation), " started: follow it on the cluster page.") : null,
          result.scale?.length ? h("p", { class: "small" }, `The cluster is running: the new pools are saved; Scale them on the cluster page (${result.scale.map((s: any) => `${s.pool} to ${s.count}`).join(", ")}).`) : null,
          h("div", { class: "row" }, h("a", { class: "btn primary", href: `#/cluster/${c.id}` }, "View cluster"), h("button", { type: "button", onclick: () => ((result = null), (plan = null), (planned = ""), drawPlan()) }, "Deploy more")),
        ),
      );
      return;
    }
    const review = h("button", { type: "button", class: plan ? "" : "primary", disabled: busy, "data-deploy-review": "", onclick: () => void doPlan() }, busy && !plan ? "Checking…" : "Review the plan");
    if (!plan) return void planBox.replaceChildren(h("section", { class: "card" }, h("h2", {}, "3. Review"), h("p", { class: "muted small" }, "Review shows every pool that will be created, where, and the price, before anything is saved."), h("div", { class: "row" }, review)));
    const famOf = (i: any) => (i.path[0] === "families" ? request().families[i.path[1]]?.family : null);
    const pr = plan.projection || {};
    const startCb = h("input", { type: "checkbox", checked: st.start, "data-deploy-start": "" }) as HTMLInputElement;
    startCb.addEventListener("change", () => (st.start = startCb.checked));
    const go = h("button", { type: "button", class: "primary", disabled: busy || !plan.ok, "data-deploy-apply": "", onclick: () => void doApply() }, plan.target.kind === "new" ? "Deploy: define the cluster" : `Deploy: add ${plan.pools.length} pool(s) to ${plan.target.name}`);
    planBox.replaceChildren(
      h(
        "section",
        { class: "card", "data-deploy-review-card": "" },
        h("h2", {}, "3. Review: what will be created"),
        h("p", {}, plan.ok ? badge("valid", "good") : badge(`${plan.issues.length} problem(s)`, "critical"), " ", plan.target.kind === "new" ? `New cluster ${plan.target.name}` : `Existing cluster ${plan.target.name} (${plan.target.status}${plan.target.running ? ", running" : ""})`),
        plan.issues.length ? h("ul", { "data-deploy-issues": "" }, ...plan.issues.map((i: any) => h("li", { class: "cf-err" }, `${famOf(i) ? `${famOf(i)}: ` : i.path.length ? `${i.path.join(".")}: ` : ""}${i.message}`))) : null,
        h(
          "div",
          { class: "tablewrap" },
          h(
            "table",
            { "data-deploy-pools-table": "" },
            h("thead", {}, h("tr", {}, ...["pool", "family", "preset", "workers", "where", "models"].map((x) => h("th", { class: x === "workers" ? "num" : "" }, x)))),
            h("tbody", {}, ...plan.pools.map((p: any) => h("tr", {}, h("td", { class: "mono" }, p.id), h("td", {}, p.family), h("td", {}, p.title), h("td", { class: "num" }, String(p.count)), h("td", { class: "wrap" }, p.where), h("td", { class: "wrap" }, p.models.join(", "))))),
          ),
        ),
        plan.actions?.length ? h("ul", { class: "small" }, ...plan.actions.map((a: any) => h("li", {}, h("b", {}, a.action), ` ${a.target}: ${a.detail}`))) : null,
        plan.ok ? h("p", { class: "small", "data-deploy-price": "" }, `Projected: ~${fmt$(pr.cluster_dph)}/hr for the cluster${pr.projected_balance !== undefined ? `; balance ${fmt$(pr.balance)} → ${fmt$(pr.projected_balance)} at the deadline (floor ${fmt$(pr.floor)})` : ""}.`) : null,
        ...(plan.warnings || []).slice(0, 12).map((w: any) => h("p", { class: "small muted" }, `note: ${famOf(w) ? `${famOf(w)}: ` : ""}${w.message}`)),
        h(
          "div",
          { class: "row", style: "margin-top:8px" },
          review,
          plan.target.running ? h("span", { class: "small muted" }, "Running: the pools are saved; Scale adds their workers.") : h("label", { class: "cf-toggle" }, startCb, plan.target.kind === "new" ? "start it now" : "start it now (with every pool)"),
          go,
        ),
      ),
    );
  }
  async function doPlan() {
    busy = true;
    drawPlan();
    const req = request();
    try {
      plan = await api("/api/v1/deploy/plan", { method: "POST", body: req });
      planned = JSON.stringify(req);
    } catch (e) {
      plan = { ok: false, issues: (e as any).issues || [{ path: [], message: (e as Error).message }], warnings: [], pools: [], actions: [], target: { kind: st.target, name: st.name } };
      planned = JSON.stringify(req);
    }
    busy = false;
    draw();
  }
  async function doApply() {
    if (!plan?.ok) return;
    const what = plan.target.kind === "new" ? `Define ${plan.target.name} with ${plan.pools.length} pool(s)${st.start ? " and start it" : ""}?` : `Add ${plan.pools.length} pool(s) to ${plan.target.name}${!plan.target.running && st.start ? " and start it" : ""}?`;
    if (!confirm(`${what}\n\n${plan.pools.map((p: any) => `${p.id} ×${p.count}: ${p.where}`).join("\n")}`)) return;
    busy = true;
    drawPlan();
    try {
      result = await api("/api/v1/deploy/apply", { method: "POST", body: { ...JSON.parse(planned), start: st.start, ...(plan.target.version !== null ? { version: plan.target.version } : {}) } });
      o.toast("deploy: ok");
      cat.clusters = (await api("/api/v1/deploy/catalog")).clusters;
    } catch (e) {
      o.toast(`deploy: ${(e as Error).message}`);
    }
    busy = false;
    draw();
  }
  function draw() {
    body.replaceChildren(targetCard(), familiesCard(), ...cat.families.filter((f) => st.fams.get(f.id)!.on).map(familyCard), planBox);
    drawPlan();
  }
  for (const [fid, f] of st.fams) if (f.on) pickProvider(fid, f.provider);
  draw();
  host.replaceChildren(body);
}
