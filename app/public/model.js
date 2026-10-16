// fv-app model page: detail + playground. Talks to the engine directly;
// the key lives in localStorage and is sent only to the engine URL.
"use strict";

document.addEventListener("DOMContentLoaded", async () => {
  const id = new URLSearchParams(location.search).get("id");
  const [{ models }, { prices }] = await Promise.all([
    fetch("/api/v1/models").then((r) => r.json()),
    fetch("/api/v1/pricing").then((r) => r.json()),
  ]);
  const m = models.find((x) => x.id === id) || models[0];
  if (!m) return;
  document.title = `FV — ${m.title}`;
  const price = (prices.find((p) => p.model === m.id)?.tiers || [])[0] || null;

  const detail = document.getElementById("detail");
  detail.innerHTML = `
    <div class="eyebrow">${m.family.toUpperCase()}${m.tier ? " · " + m.tier.toUpperCase() : ""}</div>
    <h1 style="font-size:clamp(30px,4vw,48px);margin:0 0 12px;letter-spacing:-.02em">${m.title}</h1>
    <p class="muted" style="max-width:60ch">${m.description}</p>
    <div style="margin:14px 0">
      ${(m.tasks || []).map((t) => `<span class="badge">${t}</span>`).join("")}
      ${price ? `<span class="badge">$${price.usd_per_sec.toFixed(2)}/s</span>` : ""}
    </div>`;

  const urlEl = document.getElementById("pgUrl");
  const keyEl = document.getElementById("pgKey");
  const secsEl = document.getElementById("pgSecs");
  urlEl.value = localStorage.getItem("fv_engine_url") || "";
  keyEl.value = localStorage.getItem("fv_api_key") || "";

  const estEl = document.getElementById("pgEst");
  const updateEst = () => {
    if (price && estEl) {
      const secs = parseInt(secsEl.value, 10) || 5;
      estEl.textContent = `Est. cost: $${(secs * price.usd_per_sec).toFixed(2)} (${secs}s × $${price.usd_per_sec.toFixed(2)}/s)`;
    }
  };
  secsEl.onchange = updateEst;
  updateEst();

  document.getElementById("pgRun").onclick = async () => {
    const out = document.getElementById("pgOut");
    const vidBox = document.getElementById("pgVideo");
    const base = urlEl.value.replace(/\/+$/, "");
    const key = keyEl.value;
    const prompt = document.getElementById("pgPrompt").value.trim();
    if (!base || !key || !prompt) { out.textContent = "Engine URL, API key, and prompt are all required."; return; }
    localStorage.setItem("fv_engine_url", base);
    localStorage.setItem("fv_api_key", key);
    out.textContent = "Submitting…";
    vidBox.innerHTML = "";
    try {
      const r = await fetch(`${base}/v1/videos`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${key}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: m.api_model,
          prompt,
          seconds: document.getElementById("pgSecs").value,
          size: document.getElementById("pgSize").value,
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
      const jobId = j.id;
      out.textContent = `Job ${jobId} submitted — polling…`;
      for (let i = 0; i < 120; i++) {
        await new Promise((res) => setTimeout(res, 5000));
        const pr = await fetch(`${base}/v1/videos/${jobId}`, { headers: { "Authorization": `Bearer ${key}` } });
        const pj = await pr.json();
        const status = pj.status || pj.state;
        out.textContent = `Job ${jobId}: ${status}…`;
        if (status === "completed" || status === "succeeded") {
          const url = pj.url || pj.video_url || (pj.data && pj.data[0] && pj.data[0].url);
          if (url) {
            vidBox.innerHTML = "";
            const v = document.createElement("video");
            v.src = url; v.controls = true; v.style.width = "100%"; v.style.borderRadius = "12px"; v.style.marginTop = "12px";
            vidBox.appendChild(v);
          }
          const secs = parseInt(document.getElementById("pgSecs").value, 10) || 5;
          const cost = price ? ` · Cost: $${(secs * price.usd_per_sec).toFixed(2)}` : "";
          out.textContent = `Done in ${pj.inference_time_s ? Number(pj.inference_time_s).toFixed(1) + "s" : "—"}.${cost}`;
          break;
        }
        if (status === "failed" || status === "error") { out.textContent = `Failed: ${pj.error || "unknown"}`; break; }
      }
    } catch (e) {
      out.textContent = `Error: ${e.message}`;
    }
  };
});
