// Settings forms: minting an API token (token-create) and the release
// dispatch (release-dispatch), each bound to its schema; the token's name is
// checked for uniqueness as you type, the release channel is chosen from the
// live channels and the target from the commits with images.
import { h } from "../dom";
import { createForm, type Api } from "../fields";
import { loadDyn, loadSchemas, schemaRemote } from "./common";

export async function mountTokenForm(host: HTMLElement, o: { api: Api; toast: (m: string) => void; onMinted: (r: any) => void }) {
  const { api } = o;
  const schemas = await loadSchemas(api);
  const mint = h("button", { type: "button", class: "primary" }, "Mint token");
  const f = createForm({ schemaName: "token-create", schema: schemas["token-create"]!, api, value: { name: "", scope: "read", ttl_days: 90 }, submit: mint, remote: schemaRemote(api, "token-create") });
  mint.addEventListener("click", async () => {
    if (!f.ok()) return;
    try {
      const r = await api("/api/tokens", { method: "POST", body: f.value() });
      o.toast("mint: ok");
      o.onMinted(r);
    } catch (e) {
      o.toast(`mint: ${(e as Error).message}`);
    }
  });
  f.el.append(
    h("div", { class: "cf-fields" }, f.field(["name"], { label: "Token name", nameCheck: "token", placeholder: "e.g. laptop-ci" }), f.field(["scope"], { label: "Scope" }), f.field(["ttl_days"], { label: "Expires after", unsetLabel: "90 days" })),
    f.summary,
    h("div", { class: "row", style: "margin-top:8px" }, mint),
  );
  host.replaceChildren(f.el);
}

export async function mountReleaseForm(host: HTMLElement, o: { api: Api; toast: (m: string) => void }) {
  const { api } = o;
  const [schemas, dyn] = await Promise.all([loadSchemas(api), loadDyn(api)]);
  const go = h("button", { type: "button", class: "primary" }, "Dispatch release.yml");
  const f = createForm({ schemaName: "release-dispatch", schema: schemas["release-dispatch"]!, dyn, api, value: { action: "promote", channel: "stable", dry_run: true }, submit: go, remote: schemaRemote(api, "release-dispatch") });
  const body = h("div", {});
  const draw = () => {
    const promote = f.get(["action"]) === "promote";
    body.replaceChildren(
      h(
        "div",
        { class: "cf-fields" },
        f.field(["action"], { label: "Action", onSet: (v) => (v === "promote" ? f.set(["to"], undefined, true) : f.set(["target"], undefined, true), draw()) }),
        f.field(["channel"], { label: "Channel", allowUnset: false }),
        promote ? f.field(["target"], { label: "Target", placeholder: "a git sha with images, a digest or a tag", help: "Suggestions: the commits that have serve images in GHCR and earlier releases." }) : f.field(["to"], { label: "Roll back to release", unsetLabel: "the previous release of the channel" }),
        f.field(["notes"], { label: "Notes", placeholder: "why" }),
        f.field(["dry_run"], { label: "Dry run", placeholder: "only show what would change" }),
        f.field(["templates"], { label: "Templates", placeholder: "update the Runpod templates too (default on)" }),
        ...(promote ? [f.field(["allow_partial"], { label: "Allow partial", placeholder: "even if some variants have no image" })] : []),
      ),
    );
    f.prune();
  };
  go.addEventListener("click", async () => {
    if (!f.ok()) return;
    const v = f.value();
    if (!confirm(`${v.action} ${v.action === "promote" ? `${v.target} to ` : ""}${v.channel}${v.dry_run ? " (dry run)" : ""}? This dispatches release.yml.`)) return;
    try {
      const r = await api("/api/github/release", { method: "POST", body: v });
      o.toast(`dispatched: see ${r.runs_url}`);
    } catch (e) {
      o.toast(`${v.action}: ${(e as Error).message}`);
    }
  });
  draw();
  f.el.append(body, f.summary, h("div", { class: "row", style: "margin-top:8px" }, go));
  host.replaceChildren(f.el);
}
