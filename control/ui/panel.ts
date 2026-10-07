// The document editor panel: Form and JSON tabs kept in sync, live schema
// and server validation, review (diff, and a plan for cluster specs), save
// with the loaded version (409 on a stale edit), history with restore.
import { createJsonEditor, type Dynamic, type JsonEditor } from "./editor";
import { renderForm, type FormHandle } from "./form";
import type { Issue, Schema } from "./schema";
import { renderDiff } from "./view";

export type Api = (path: string, opts?: { method?: string; body?: unknown }) => Promise<any>;
export interface PanelOptions {
  kind: "cluster-spec" | "policies" | "attribution" | "env" | "build-pods" | "serverless-policy";
  id: string;
  title?: string;
  api: Api;
  cluster?: string; // for the dynamic enums (pools)
  onSaved?: (r: any) => void;
  toast?: (msg: string) => void;
}
const SECRET_MASK = "•••••••• (new value, write-only)";
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, any> = {}, ...kids: (Node | string | null | undefined | false)[]) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "class") e.className = v;
    else e.setAttribute(k, v === true ? "" : String(v));
  }
  for (const k of kids) if (k !== null && k !== undefined && k !== false) e.append(k);
  return e;
}

let schemaCache: Promise<Record<string, Schema>> | null = null;
const dynCache = new Map<string, Promise<Dynamic>>();

export async function openDocPanel(host: HTMLElement, o: PanelOptions) {
  const say = o.toast || (() => {});
  schemaCache ||= o.api("/api/schemas").then((r) => r.schemas);
  const dk = o.cluster || "";
  if (!dynCache.has(dk)) dynCache.set(dk, o.api(`/api/schemas/dynamic${dk ? `?cluster=${encodeURIComponent(dk)}` : ""}`).catch(() => ({})));
  const base = `/api/docs/${o.kind}/${encodeURIComponent(o.id)}`;
  const [schemas, dyn, loaded] = await Promise.all([schemaCache, dynCache.get(dk)!, o.api(base)]);
  const schema = schemas[loaded.schema] as Schema;
  let original: any = loaded.doc;
  let version: number = loaded.version;
  let current: any = structuredClone(original);
  let serverIssues: Issue[] = [];
  let jsonValid = true;
  let mode: "form" | "json" = "form";
  const secrets = new Map<string, string>(); // env: pending write-only values, never shown

  // env documents: the JSON tab shows a placeholder for a pending `set`.
  const forJson = (v: any) => {
    if (o.kind !== "env" || !v) return v;
    return Object.fromEntries(Object.entries<any>(v).map(([k, e]) => [k, e.set !== undefined ? { ...e, set: SECRET_MASK } : e]));
  };
  const fromJson = (v: any): { v: any; issues: Issue[] } => {
    if (o.kind !== "env" || !v || typeof v !== "object") return { v, issues: [] };
    const issues: Issue[] = [];
    const out: any = {};
    for (const [k, e] of Object.entries<any>(v)) {
      if (e && e.set !== undefined && e.set !== SECRET_MASK) issues.push({ path: [k, "set"], message: "type secret values in the Form tab (write-only field), not in the JSON" });
      out[k] = e && e.set === SECRET_MASK ? { ...e, set: secrets.get(k) } : e;
    }
    return { v: out, issues };
  };

  const status = h("div", { class: "fv-status small", role: "status" });
  const issuesBox = h("ul", { class: "fv-issues" });
  const verTag = h("span", { class: "badge" }, `v${version}`);
  const formHost = h("div", { class: "fv-formhost" });
  const jsonHost = h("div", { class: "fv-jsonhost", hidden: true });
  const review = h("div", { class: "fv-review", hidden: true });
  const hist = h("div", { class: "fv-history", hidden: true });
  const tabForm = h("button", { type: "button", class: "on", onclick: () => switchTo("form") }, "Form");
  const tabJson = h("button", { type: "button", onclick: () => switchTo("json") }, "JSON");
  const dirtyTag = h("span", { class: "muted small" });
  // Off while the document is invalid (the JSON does not parse, or the schema / the server refuse it).
  const reviewBtn = h("button", { type: "button", class: "primary", disabled: true, onclick: () => openReview() }, o.kind === "cluster-spec" ? "Review, plan & save" : "Review & save") as HTMLButtonElement;
  host.replaceChildren(
    h(
      "div",
      { class: "fv-panel", "data-kind": o.kind },
      h("div", { class: "row fv-bar" }, h("b", {}, o.title || `${o.kind} ${o.id}`), verTag, dirtyTag, h("span", { class: "spacer" }), h("div", { class: "tabs", role: "tablist" }, tabForm, tabJson)),
      formHost,
      jsonHost,
      issuesBox,
      h(
        "div",
        { class: "row fv-bar" },
        h("button", { type: "button", onclick: () => { if (mode === "json" && !editor.format()) say("not valid JSON"); } }, "Format"),
        h("button", { type: "button", onclick: () => runValidate(true) }, "Validate"),
        reviewBtn,
        h("button", { type: "button", onclick: () => openHistory() }, "History"),
        h("button", { type: "button", class: "ghost", onclick: () => reload() }, "Revert"),
        status,
      ),
      review,
      hist,
    ),
  );

  let form: FormHandle;
  const editor: JsonEditor = createJsonEditor({
    parent: jsonHost,
    doc: pretty(forJson(current)),
    schema,
    dynamic: dyn,
    extraIssues: () => serverIssues,
    onChange: (text) => {
      if (suppress) return;
      try {
        const r = fromJson(JSON.parse(text));
        current = r.v;
        jsonValid = r.issues.length === 0;
        serverIssues = r.issues;
      } catch {
        jsonValid = false;
      }
      changed();
    },
  });
  let suppress = false;
  form = renderForm(formHost, {
    schema,
    value: current,
    dyn,
    onChange: (v) => {
      current = v;
      if (o.kind === "env") for (const [k, e] of Object.entries<any>(v || {})) e.set !== undefined ? secrets.set(k, e.set) : secrets.delete(k);
      suppress = true;
      editor.set(pretty(forJson(current)));
      suppress = false;
      changed();
    },
  });

  function switchTo(m: "form" | "json") {
    if (m === "form" && !jsonValid) {
      say("fix the JSON first: the form needs a valid document");
      return;
    }
    mode = m;
    tabForm.classList.toggle("on", m === "form");
    tabJson.classList.toggle("on", m === "json");
    formHost.hidden = m !== "form";
    jsonHost.hidden = m !== "json";
    if (m === "form") form.update(current);
    else editor.view.requestMeasure();
  }
  let vt: any;
  function changed() {
    const dirty = pretty(current) !== pretty(original);
    dirtyTag.textContent = dirty ? "● unsaved changes" : "";
    review.hidden = true;
    clearTimeout(vt);
    vt = setTimeout(() => runValidate(false), 600);
  }
  function showIssues(list: Issue[]) {
    // Each problem at its field in the form too.
    for (const r of formHost.querySelectorAll<HTMLElement>(".fv-row.has-srv")) r.classList.remove("has-srv", "has-err");
    for (const e of formHost.querySelectorAll<HTMLElement>(".fv-srv")) e.remove();
    for (const i of list) {
      for (let n = i.path.length; n > 0; n--) {
        const row = formHost.querySelector<HTMLElement>(`[data-path="${CSS.escape(i.path.slice(0, n).join("."))}"]`);
        if (!row) continue;
        row.classList.add("has-srv", "has-err");
        row.append(h("div", { class: "cf-err fv-srv", role: "alert" }, (n < i.path.length ? `${i.path.slice(n).join(".")}: ` : "") + i.message));
        break;
      }
    }
    issuesBox.replaceChildren(
      ...list.map((i) =>
        h(
          "li",
          { class: "fv-issue", onclick: () => focusPath(i.path) },
          h("code", {}, i.path.length ? i.path.join(".") : "(document)"),
          " ",
          i.message,
        ),
      ),
    );
  }
  function focusPath(path: (string | number)[]) {
    if (mode === "form") {
      const el = formHost.querySelector<HTMLElement>(`[data-path="${CSS.escape(path.join("."))}"] input, [data-path="${CSS.escape(path.join("."))}"] select`);
      if (el) return el.focus();
      switchTo("json");
    }
    editor.view.focus();
  }
  async function runValidate(explicit: boolean): Promise<boolean> {
    if (!jsonValid) {
      status.textContent = "invalid JSON";
      reviewBtn.disabled = true;
      showIssues(serverIssues.length ? serverIssues : [{ path: [], message: "the JSON does not parse" }]);
      editor.relint();
      return false;
    }
    try {
      const r = await o.api(`${base}/validate`, { method: "POST", body: { doc: current } });
      serverIssues = r.issues || [];
      showIssues(serverIssues);
      editor.relint();
      status.textContent = r.ok ? "✓ valid" : `${serverIssues.length} problem(s)`;
      status.className = `fv-status small ${r.ok ? "ok" : "bad"}`;
      reviewBtn.disabled = !r.ok;
      reviewBtn.title = r.ok ? "" : `${serverIssues.length} problem(s): see the list`;
      if (explicit && r.ok) say("valid");
      return r.ok;
    } catch (e) {
      status.textContent = (e as Error).message;
      reviewBtn.disabled = true;
      return false;
    }
  }
  const masked = (v: any) => (o.kind === "env" && v ? Object.fromEntries(Object.entries<any>(v).map(([k, e]) => [k, e.set !== undefined ? { ...e, set: "•••••••• (new value)" } : e])) : v);
  async function openReview() {
    if (!(await runValidate(false))) {
      say("fix the problems first");
      return;
    }
    review.hidden = false;
    hist.hidden = true;
    review.replaceChildren(h("h3", {}, "Changes (current → proposed)"), renderDiff(original, masked(current)));
    if (o.kind === "cluster-spec") {
      const planBox = h("div", { class: "fv-plan" }, h("span", { class: "muted small" }, "planning…"));
      review.append(h("h3", {}, "Plan"), planBox);
      try {
        const p = await o.api(`${base}/plan`, { method: "POST", body: { doc: current } });
        const pr = p.projection || {};
        planBox.replaceChildren(
          h("ul", {}, ...(p.actions || []).map((a: any) => h("li", { class: `plan-${a.action}` }, h("b", {}, a.action), ` ${a.target}: ${a.detail}`))),
          ...(p.warnings || []).map((w: string) => h("p", { class: "small muted" }, `note: ${w}`)),
          h(
            "p",
            { class: "small" },
            `$/hr: ${p.dph_now?.toFixed?.(2) ?? "0.00"} now → ~${pr.cluster_dph?.toFixed?.(2) ?? "?"} after. `,
            pr.balance !== undefined ? `Balance $${pr.balance.toFixed(2)} → projected $${pr.projected_balance.toFixed(2)} after ${pr.hours.toFixed(1)} h (floor $${pr.floor}). ` : "",
            pr.ok === false ? h("span", { class: "badge critical" }, `over the floor: ${(pr.reasons || []).join("; ")}`) : pr.ok ? h("span", { class: "badge good" }, "within the floor") : "",
          ),
          h("p", { class: "small muted" }, "Saving changes the definition only; apply it to running pods with Scale, Roll or the env restart."),
        );
      } catch (e) {
        planBox.textContent = (e as Error).message;
      }
    }
    review.append(
      h(
        "div",
        { class: "row" },
        h("button", { type: "button", class: "primary", onclick: () => save() }, `Save (v${version} → v${version + 1})`),
        h("button", { type: "button", onclick: () => (review.hidden = true) }, "Cancel"),
      ),
    );
    review.scrollIntoView({ block: "nearest" });
  }
  async function save() {
    try {
      const r = await o.api(base, { method: "PUT", body: { doc: current, version } });
      original = r.doc;
      version = r.version;
      current = structuredClone(original);
      secrets.clear();
      verTag.textContent = `v${version}`;
      review.hidden = true;
      suppress = true;
      editor.set(pretty(forJson(current)));
      suppress = false;
      form.update(current);
      changed();
      say(r.skipped?.length ? `saved; secrets not restored: ${r.skipped.join(", ")}` : `saved (v${version})`);
      o.onSaved?.(r);
    } catch (e) {
      const msg = (e as Error).message;
      if (/changed since you loaded/.test(msg)) {
        review.replaceChildren(
          h("p", { class: "fv-conflict" }, "Someone saved this document after you loaded it. Your edit was not saved."),
          h("div", { class: "row" }, h("button", { type: "button", onclick: () => showServerDiff() }, "Compare with the saved version"), h("button", { type: "button", class: "danger", onclick: () => reload() }, "Reload (discard my edit)")),
        );
      } else say(`save failed: ${msg}`);
    }
  }
  async function showServerDiff() {
    const r = await o.api(base);
    review.append(h("h3", {}, `Saved v${r.version} → yours`), renderDiff(r.doc, masked(current)), h("p", { class: "small muted" }, "Rebase: reload, re-apply your edit, and save."));
  }
  async function reload() {
    const r = await o.api(base);
    original = r.doc;
    version = r.version;
    current = structuredClone(original);
    secrets.clear();
    jsonValid = true;
    serverIssues = [];
    verTag.textContent = `v${version}`;
    suppress = true;
    editor.set(pretty(forJson(current)));
    suppress = false;
    form.update(current);
    review.hidden = true;
    showIssues([]);
    changed();
  }
  async function openHistory() {
    hist.hidden = !hist.hidden;
    if (hist.hidden) return;
    review.hidden = true;
    const r = await o.api(`${base}/history`);
    const list = h("div", {});
    hist.replaceChildren(h("h3", {}, "History"), list);
    if (!r.history.length) list.append(h("p", { class: "muted small" }, "No saved versions yet (only saves through the editor are kept here; the audit log has every change)."));
    for (const e of r.history) {
      const det = h("details", {}, h("summary", {}, `${new Date(e.at).toISOString().replace("T", " ").slice(0, 19)}Z · ${e.actor} · ${e.action}${e.detail ? ` · ${e.detail}` : ""}`));
      det.addEventListener("toggle", () => {
        if (!det.open || det.childElementCount > 1) return;
        det.append(
          renderDiff(e.before, e.after),
          h(
            "div",
            { class: "row" },
            h("button", { type: "button", onclick: () => restore(e.audit_id, "after") }, "Restore this version"),
            h("button", { type: "button", class: "ghost", onclick: () => restore(e.audit_id, "before") }, "Restore the version before it"),
          ),
        );
      });
      list.append(det);
    }
  }
  async function restore(auditId: number, which: "after" | "before") {
    try {
      const r = await o.api(`${base}/restore`, { method: "POST", body: { audit_id: auditId, which, version } });
      say(r.skipped?.length ? `restored; secrets not restored: ${r.skipped.join(", ")}` : `restored (v${r.version})`);
      await reload();
      hist.hidden = true;
      o.onSaved?.(r);
    } catch (e) {
      say(`restore failed: ${(e as Error).message}`);
    }
  }
  changed();
  return { reload, get: () => current, editor };
}
