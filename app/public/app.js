// fv-app shared: catalog + pricing rendering. No framework, no build step.
"use strict";

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

async function renderGallery() {
  const grid = document.getElementById("gallery");
  if (!grid) return;
  const [{ models }, { prices, placeholder }] = await Promise.all([getJSON("/api/v1/models"), getJSON("/api/v1/pricing")]);
  const priceFor = (id) => {
    const p = prices.find((x) => x.model === id);
    if (!p || !p.tiers.length) return "";
    const t = p.tiers[0];
    return `$${t.usd_per_sec.toFixed(2)}/s`;
  };
  for (const m of models) {
    const card = el("div", "card");
    const vid = el("div", "vid");
    if (m.sample_video) {
      const v = document.createElement("video");
      v.src = m.sample_video; v.muted = true; v.loop = true; v.playsInline = true;
      v.onmouseenter = () => v.play().catch(() => {});
      v.onmouseleave = () => v.pause();
      vid.appendChild(v);
    } else {
      vid.innerHTML = `<span class="play"><span></span></span><span class="vlabel">${m.title} · sample coming soon</span>`;
    }
    card.appendChild(vid);
    const body = el("div", "");
    body.innerHTML = `
      <h3>${m.title}</h3>
      <div style="margin-bottom:8px">
        ${m.tier ? `<span class="badge hot">${m.tier}</span>` : ""}
        <span class="badge">${m.family}</span>
        ${priceFor(m.id) ? `<span class="badge">${priceFor(m.id)}</span>` : ""}
      </div>
      <p class="muted small">${m.tagline}</p>
      <a class="btn sm" href="/model.html?id=${encodeURIComponent(m.id)}">Try it</a>`;
    card.appendChild(body);
    grid.appendChild(card);
  }

  const pt = document.getElementById("priceTable");
  if (pt) {
    const t = el("table", "");
    t.innerHTML = `<tr><th>Model</th><th>Tier</th><th class="num">Price</th></tr>` +
      prices.flatMap((p) => p.tiers.map((tier) =>
        `<tr><td><code>${p.model}</code></td><td>${tier.label}</td><td class="num">$${tier.usd_per_sec.toFixed(2)}/s</td></tr>`
      )).join("");
    pt.appendChild(t);
    if (placeholder) {
      const n = el("p", "muted small", "Representative prices — final pricing coming soon.");
      pt.appendChild(n);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  renderGallery().catch((e) => console.error(e));
});
