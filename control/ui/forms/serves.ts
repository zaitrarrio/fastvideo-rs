// What a serverless endpoint serves (src/serverless/serves.ts, derived from
// its preset or config before any worker boots) and the ready-made test
// invokes built from it (src/serverless/examples.ts): the "Serves" panel of
// the new-endpoint form (live, from POST /api/serverless/validate) and of the
// endpoint page, and the endpoint page's invoke picker (an example, a
// reference image / audio URL where the task needs one, the free-form JSON).
import { badge, h, type Api } from "../dom";

const IMG = "{{image_url}}";
const AUD = "{{audio_url}}";
const TASK_WORD: Record<string, string> = { t2v: "text-to-video", i2v: "image-to-video", keyframes: "first/last frame", ref2v: "reference-to-video", a2v: "audio-to-video", retake: "retake", extend: "extend", stream: "streams (causal)" };

/** The Serves panel: preset (given or inferred), models with tasks and tiers, API names, mounted APIs, fal apps, weights. */
export function servesPanel(serving: any, o: { check?: () => Promise<any> } = {}): HTMLElement {
  if (!serving) return h("div", { class: "small muted" }, "Serves: fix the spec to see what it serves.");
  const s = serving.serves || {};
  const models: any[] = s.models || [];
  const box = h("div", { class: "sv-serves small", "data-serves": "" });
  const presetLine = serving.preset
    ? h("span", {}, "preset ", h("b", {}, serving.preset), serving.preset_inferred ? h("span", { class: "muted" }, " (inferred: the variant and config match it)") : null)
    : h("span", {}, h("b", {}, "custom"), h("span", { class: "muted" }, " (no preset matches the variant and config)"));
  const src = s.source || {};
  const srcLine = src.kind === "inline" ? "inline config" : src.kind === "image" ? `${src.config} (in the image)` : src.kind === "image-default" ? `${src.config} (the image's default)` : `${src.config || "?"}: unknown to fv-control`;
  box.append(...([
    h("div", { class: "row", style: "flex-wrap:wrap;gap:12px" }, presetLine, h("span", { class: "muted" }, "config: ", srcLine), s.engine ? h("span", { class: "muted" }, `engine ${s.engine}${s.swap ? ", swap on (one model resident at a time)" : ""}`) : null),
    src.note ? h("p", { class: "muted" }, src.note) : null,
    models.length
      ? h(
          "table",
          { class: "tbl", style: "margin-top:6px" },
          h("thead", {}, h("tr", {}, ...["model", "recipe", "tier", "tasks", "resident", "weights"].map((x) => h("th", {}, x)))),
          h(
            "tbody",
            {},
            ...models.map((m) =>
              h(
                "tr",
                { "data-model": m.id },
                h("td", {}, h("code", {}, m.id)),
                h("td", {}, m.recipe || "–"),
                h("td", {}, m.tier || "–"),
                h("td", {}, (m.tasks || []).map((t: string) => TASK_WORD[t] || t).join(", ") || "–"),
                h("td", {}, m.resident ? "yes" : "on demand"),
                h("td", {}, (m.weights || []).join(", ") || "–"),
              ),
            ),
          ),
        )
      : h("p", { class: "muted" }, "No model derived."),
    (s.aliases || []).length
      ? h("p", {}, "API names: ", ...(s.aliases as any[]).flatMap((a, i) => [i ? ", " : "", h("code", {}, a.name), ` → ${a.model}`, a.via === "tier" ? h("span", { class: "muted" }, " (tier)") : ""]))
      : null,
    h("p", {}, "APIs: ", (s.protocols || []).join(", ") || "–", (s.fal_apps || []).length ? h("span", { class: "muted" }, ` · fal apps: ${s.fal_apps.join(", ")}`) : null),
  ].filter(Boolean) as HTMLElement[]));
  if (o.check) {
    const out = h("span", { class: "muted" });
    const btn = h("button", { type: "button", class: "ghost" }, "Check against a running worker");
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      out.replaceChildren("checking…");
      try {
        const r = await o.check!();
        out.replaceChildren(
          r.ok ? badge("worker matches", "good") : badge("mismatch", "critical"),
          " ",
          r.ok ? `serves ${r.served.join(", ")}` : `${r.missing.length ? `missing on the worker: ${r.missing.join(", ")}. ` : ""}${r.extra.length ? `served but not expected: ${r.extra.join(", ")}.` : ""}${r.note ? ` ${r.note}` : ""}`,
        );
      } catch (e) {
        out.replaceChildren((e as Error).message);
      } finally {
        btn.disabled = false;
      }
    });
    box.append(h("div", { class: "row", style: "margin-top:4px;gap:8px" }, btn, out));
  }
  return box;
}

/**
 * The endpoint page's test invoke: a picker of the examples it serves, the URL fields an example needs,
 * and the JSON (editable). `send` posts {input} (queue) or {method, path, body} (lb) to the invoke route.
 */
export function mountInvokePicker(host: HTMLElement, o: { api: Api; endpoint: { id: string; mode: string }; examples: any[]; onResult: (r: any) => void; send: (body: any) => Promise<void> }) {
  const ex: any[] = o.examples || [];
  const sel = h("select", { "aria-label": "Example request", style: "max-width:100%" }, ...ex.map((e, i) => h("option", { value: String(i) }, e.label)));
  const img = h("input", { type: "url", placeholder: "https://… an image the worker can fetch", "aria-label": "Reference image URL", style: "width:100%" });
  const aud = h("input", { type: "url", placeholder: "https://… an audio file the worker can fetch", "aria-label": "Audio URL", style: "width:100%" });
  const imgRow = h("label", { class: "small", style: "display:none" }, "Image URL (first frame / reference)", img);
  const audRow = h("label", { class: "small", style: "display:none" }, "Audio URL", aud);
  const ta = h("textarea", { rows: 8, style: "width:100%;font-family:var(--mono, monospace);font-size:12px", "aria-label": "Invoke input" });
  const hint = h("p", { class: "small muted" });
  const fill = (v: unknown) =>
    JSON.parse(
      JSON.stringify(v)
        .split(IMG)
        .join(JSON.stringify(img.value.trim()).slice(1, -1))
        .split(AUD)
        .join(JSON.stringify(aud.value.trim()).slice(1, -1)),
    );
  const cur = () => ex[Number(sel.value)];
  const show = () => {
    const e = cur();
    if (!e) return;
    imgRow.style.display = e.needs?.includes("image_url") ? "block" : "none";
    audRow.style.display = e.needs?.includes("audio_url") ? "block" : "none";
    const v = o.endpoint.mode === "lb" ? e.invoke : e.invoke.input;
    ta.value = JSON.stringify(fill(v), null, 1);
    hint.textContent = o.endpoint.mode === "lb" && e.poll ? `The load balancer answers with the job's id; poll it with GET ${e.poll}.` : e.needs?.length ? "Fill the URL: the worker fetches it." : "";
  };
  sel.addEventListener("change", show);
  img.addEventListener("input", show);
  aud.addEventListener("input", show);
  const go = h("button", { class: "primary", type: "button" }, "Send");
  go.addEventListener("click", async () => {
    let x;
    try {
      x = JSON.parse(ta.value);
    } catch (err) {
      return o.onResult({ error: `input: ${(err as Error).message}` });
    }
    const e = cur();
    if (e?.needs?.includes("image_url") && !img.value.trim() && ta.value.includes('""')) return o.onResult({ error: "this example needs an image URL" });
    go.disabled = true;
    try {
      await o.send(o.endpoint.mode === "lb" ? x : { input: x });
    } finally {
      go.disabled = false;
    }
  });
  host.replaceChildren(h("div", { class: "row", style: "margin-bottom:6px;flex-wrap:wrap" }, sel), imgRow, audRow, ta, hint, h("div", { class: "row", style: "margin-top:6px" }, go));
  show();
}
