// Tween UI: boards of panels, shots between them, LTX and H3 side by side,
// and the animatic. Plain modules, no build step; every user string goes in
// through textContent.

const $ = (sel, root = document) => root.querySelector(sel);
const view = $("#view");
const ENGINE_LABEL = { ltx: "LTX-2.5", h3: "H3" };
const POLL_MS = 1500;

const state = {
  config: null,
  board: null,
  panels: [],
  shots: [],
  renders: [],
  shotStart: null,
  pollTimer: null,
  lanes: new Map(),
};

// ---- helpers ---------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, "");
    else node.setAttribute(k, String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function api(path, { method = "GET", body, raw } = {}) {
  const init = { method, headers: {} };
  if (raw !== undefined) init.body = raw;
  else if (body !== undefined) {
    init.headers["content-type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const r = await fetch(path, init);
  const data = await r.json().catch(() => ({}));
  if (r.status === 401 && path !== "/api/session") {
    showSignIn();
    throw new ApiError(401, "sign in first");
  }
  if (!r.ok) throw new ApiError(r.status, data.error || `request failed (${r.status})`);
  return data;
}

let toastTimer;
function toast(message, kind = "info") {
  const t = $("#toast");
  t.textContent = message;
  t.dataset.kind = kind;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), kind === "error" ? 7000 : 3500);
}

const fail = (e) => {
  if (e instanceof ApiError && e.status === 401) return;
  toast(e.message || String(e), "error");
};

const fmt = {
  secs: (s) => (s === null || s === undefined ? "—" : `${Number(s).toFixed(s < 10 ? 2 : 1)} s`),
  factor: (f) => (f === null || f === undefined ? "—" : `${f.toFixed(f < 10 ? 2 : 1)}×`),
  clock: (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`,
  ago: (ms) => {
    const s = Math.max(0, (Date.now() - ms) / 1000);
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(ms).toLocaleDateString();
  },
};

const aspectCss = (a) => (a || "16:9").replace(":", " / ");
const isOpen = (r) => r.status === "queued" || r.status === "in_progress";
const panelById = (id) => state.panels.find((p) => p.id === id);
const currentShot = () => state.shots.find((s) => s.start_panel_id === state.shotStart) || null;
const rendersFor = (shot) =>
  state.renders.filter((r) => r.start_panel_id === shot.start_panel_id && r.end_panel_id === shot.end_panel_id);
const latest = (list) => list.reduce((a, b) => (!a || b.created_at > a.created_at ? b : a), null);

function setCrumbs(...parts) {
  const c = $("#crumbs");
  c.replaceChildren();
  parts.forEach((p, i) => {
    if (i) c.append(el("span", { class: "sep", text: "/" }));
    c.append(el("span", { text: p }));
  });
}

function emptyState(title, body) {
  const node = $("#tpl-empty").content.firstElementChild.cloneNode(true);
  $(".empty-title", node).textContent = title;
  $(".empty-body", node).textContent = body;
  return node;
}

// ---- routing ---------------------------------------------------------------

function navigate(path) {
  history.pushState(null, "", path);
  route();
}

document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-nav]");
  if (!a || e.metaKey || e.ctrlKey || e.shiftKey) return;
  e.preventDefault();
  navigate(a.getAttribute("data-nav"));
});
window.addEventListener("popstate", route);

async function route() {
  stopPolling();
  const m = /^\/b\/([^/]+)/.exec(location.pathname);
  try {
    const session = await api("/api/session");
    if (session.required && !session.authenticated) return showSignIn();
    if (!state.config) state.config = await api("/api/config");
    paintEnginePill();
    if (m) await openBoard(decodeURIComponent(m[1]));
    else await showHome();
  } catch (e) {
    fail(e);
  }
  view.focus({ preventScroll: true });
}

function paintEnginePill() {
  const pill = $("#engine-pill");
  pill.hidden = false;
  const ok = state.config.engine_configured;
  pill.dataset.state = ok ? "ok" : "off";
  pill.textContent = ok ? "engine connected" : "no engine configured";
}

// ---- sign in ---------------------------------------------------------------

function showSignIn() {
  stopPolling();
  setCrumbs("Sign in");
  const input = el("input", { type: "password", autocomplete: "current-password", required: true, "aria-label": "Access token" });
  const form = el(
    "form",
    {
      class: "signin card",
      onsubmit: async (e) => {
        e.preventDefault();
        try {
          await api("/api/session", { method: "POST", body: { token: input.value } });
          route();
        } catch (err) {
          fail(err);
        }
      },
    },
    el("h2", { text: "Sign in to Tween" }),
    el("p", { class: "muted", text: "This studio is private. Enter the access token you were given." }),
    el("label", { class: "field" }, el("span", { text: "Access token" }), input),
    el("button", { class: "btn primary", type: "submit", text: "Sign in" }),
  );
  view.replaceChildren(form);
  input.focus();
}

// ---- home ------------------------------------------------------------------

async function showHome() {
  setCrumbs("Boards");
  const { boards } = await api("/api/boards");
  const title = el("input", { type: "text", placeholder: "New board title", maxlength: 120, required: true, "aria-label": "Board title" });
  const aspect = el(
    "select",
    { "aria-label": "Aspect ratio" },
    state.config.aspect_ratios.map((a) => el("option", { value: a, text: a })),
  );
  const form = el(
    "form",
    {
      class: "new-board",
      onsubmit: async (e) => {
        e.preventDefault();
        try {
          const { board } = await api("/api/boards", { method: "POST", body: { title: title.value, aspect_ratio: aspect.value } });
          navigate(`/b/${board.id}`);
        } catch (err) {
          fail(err);
        }
      },
    },
    title,
    aspect,
    el("button", { class: "btn primary", type: "submit", text: "Create board" }),
  );
  const list = boards.length
    ? el(
        "div",
        { class: "boards" },
        boards.map((b) =>
          el(
            "a",
            { class: "board-card", href: `/b/${b.id}`, "data-nav": `/b/${b.id}` },
            el("p", { class: "eyebrow", text: b.aspect_ratio }),
            el("h3", { text: b.title }),
            el("span", { class: "muted", text: `Edited ${fmt.ago(b.updated_at)}` }),
          ),
        ),
      )
    : emptyState("No boards yet", "Create one, drop in your panels, and Tween renders the motion between each pair.");
  view.replaceChildren(
    el(
      "section",
      { class: "hero" },
      el("p", { class: "eyebrow", text: "Keyframe storyboards" }),
      el("h1", { text: "Draw the frames. Tween fills the motion." }),
      el("p", {
        text: "Each pair of adjacent panels becomes a shot. Render it on LTX and H3 side by side, keep the take you like, and play the board as an animatic.",
      }),
    ),
    form,
    list,
  );
}

// ---- board -----------------------------------------------------------------

async function loadBoard(id) {
  const data = await api(`/api/boards/${encodeURIComponent(id)}`);
  Object.assign(state, { board: data.board, panels: data.panels, shots: data.shots, renders: data.renders });
  if (!state.shots.some((s) => s.start_panel_id === state.shotStart)) state.shotStart = state.shots[0]?.start_panel_id ?? null;
}

async function openBoard(id) {
  state.shotStart = null;
  await loadBoard(id);
  paintBoard();
  startPolling();
}

async function reloadBoard() {
  await loadBoard(state.board.id);
  paintBoard();
  startPolling();
}

function paintBoard() {
  const b = state.board;
  setCrumbs("Boards", b.title);
  document.documentElement.style.setProperty("--aspect", aspectCss(b.aspect_ratio));
  state.lanes.clear();

  const titleInput = el("input", {
    class: "board-title",
    value: b.title,
    maxlength: 120,
    "aria-label": "Board title",
    onchange: async (e) => {
      try {
        const { board } = await api(`/api/boards/${b.id}`, { method: "PATCH", body: { title: e.target.value } });
        state.board = board;
        setCrumbs("Boards", board.title);
      } catch (err) {
        fail(err);
        e.target.value = state.board.title;
      }
    },
  });
  const aspect = el(
    "select",
    {
      "aria-label": "Aspect ratio",
      onchange: async (e) => {
        try {
          const { board } = await api(`/api/boards/${b.id}`, { method: "PATCH", body: { aspect_ratio: e.target.value } });
          state.board = board;
          document.documentElement.style.setProperty("--aspect", aspectCss(board.aspect_ratio));
        } catch (err) {
          fail(err);
        }
      },
    },
    state.config.aspect_ratios.map((a) => el("option", { value: a, text: a, selected: a === b.aspect_ratio })),
  );

  const reelSource = el(
    "select",
    { "aria-label": "Animatic source" },
    el("option", { value: "cut", text: "The cut" }),
    el("option", { value: "ltx", text: "LTX reel" }),
    el("option", { value: "h3", text: "H3 reel" }),
  );

  view.replaceChildren(
    el(
      "div",
      { class: "board-head" },
      el("div", { style: "flex:1;min-width:240px" }, el("p", { class: "eyebrow", text: "Board" }), titleInput),
      el("label", { class: "field" }, el("span", { text: "Aspect" }), aspect),
      el("label", { class: "field" }, el("span", { text: "Play" }), reelSource),
      el("button", {
        class: "btn primary",
        type: "button",
        text: "Play animatic",
        disabled: state.shots.length === 0,
        onclick: () => openReel(reelSource.value),
      }),
    ),
    el("div", { class: "strip-wrap" }, paintStrip()),
    paintShot(),
  );
}

function paintStrip() {
  const strip = el("div", { class: "strip", id: "strip" });
  state.panels.forEach((p, i) => {
    strip.append(paintPanel(p, i));
    const shot = state.shots.find((s) => s.start_panel_id === p.id);
    if (shot) strip.append(paintLink(shot));
  });
  strip.append(paintAddPanel());
  return strip;
}

let dragId = null;
function paintPanel(p, i) {
  const node = el(
    "figure",
    {
      class: "panel",
      draggable: "true",
      "data-id": p.id,
      ondragstart: (e) => {
        dragId = p.id;
        node.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData("text/plain", p.id);
      },
      ondragend: () => {
        dragId = null;
        node.classList.remove("dragging");
        document.querySelectorAll(".drop-before").forEach((n) => n.classList.remove("drop-before"));
      },
      ondragover: (e) => {
        if (!dragId || dragId === p.id) return;
        e.preventDefault();
        node.classList.add("drop-before");
      },
      ondragleave: () => node.classList.remove("drop-before"),
      ondrop: (e) => {
        e.preventDefault();
        node.classList.remove("drop-before");
        if (dragId && dragId !== p.id) movePanel(dragId, p.id);
      },
    },
    el("div", { class: "panel-frame" }, el("img", { src: p.image_url, alt: p.caption || `Panel ${i + 1}`, loading: "lazy", draggable: "false" })),
    el("span", { class: "panel-num", text: String(i + 1).padStart(2, "0") }),
    el("input", {
      class: "panel-caption",
      value: p.caption,
      placeholder: "Caption",
      maxlength: 280,
      "aria-label": `Caption for panel ${i + 1}`,
      onchange: async (e) => {
        try {
          await api(`/api/boards/${state.board.id}/panels/${p.id}`, { method: "PATCH", body: { caption: e.target.value } });
          p.caption = e.target.value;
        } catch (err) {
          fail(err);
        }
      },
    }),
    el("button", {
      class: "btn small ghost danger panel-remove",
      type: "button",
      text: "Remove",
      "aria-label": `Remove panel ${i + 1}`,
      onclick: async () => {
        if (!confirm("Remove this panel? Shots on either side are re-paired; past renders stay in the history.")) return;
        try {
          await api(`/api/boards/${state.board.id}/panels/${p.id}`, { method: "DELETE" });
          await reloadBoard();
        } catch (err) {
          fail(err);
        }
      },
    }),
  );
  return node;
}

async function movePanel(id, beforeId) {
  const ids = state.panels.map((p) => p.id).filter((x) => x !== id);
  ids.splice(ids.indexOf(beforeId), 0, id);
  try {
    await api(`/api/boards/${state.board.id}/panel-order`, { method: "PUT", body: { panel_ids: ids } });
    await reloadBoard();
  } catch (err) {
    fail(err);
  }
}

function linkStatus(shot) {
  const rs = rendersFor(shot);
  return state.config.engine_ids.map((engine) => {
    const r = latest(rs.filter((x) => x.engine === engine));
    return { engine, status: r ? r.status : "none" };
  });
}

function paintLink(shot) {
  return el(
    "button",
    {
      class: "link",
      type: "button",
      "data-start": shot.start_panel_id,
      "aria-current": shot.start_panel_id === state.shotStart ? "true" : "false",
      "aria-label": `Shot ${shot.index + 1}`,
      onclick: () => {
        state.shotStart = shot.start_panel_id;
        paintBoard();
      },
    },
    el("div", { class: "link-arrow", "aria-hidden": "true" }),
    el("span", { class: "link-prompt", text: shot.prompt || "No prompt yet" }),
    el(
      "span",
      { class: "link-meta" },
      `${shot.seconds}s`,
      linkStatus(shot).map(({ engine, status }) => el("i", { class: "dot", "data-status": status, title: `${ENGINE_LABEL[engine]}: ${status}` })),
    ),
  );
}

function paintAddPanel() {
  const input = el("input", { type: "file", accept: "image/png,image/jpeg,image/webp", multiple: true, onchange: (e) => uploadFiles([...e.target.files]) });
  const zone = el(
    "label",
    {
      class: "add-panel",
      ondragover: (e) => {
        if (dragId || !e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        zone.classList.add("over");
      },
      ondragleave: () => zone.classList.remove("over"),
      ondrop: (e) => {
        if (dragId) return;
        e.preventDefault();
        zone.classList.remove("over");
        uploadFiles([...e.dataTransfer.files]);
      },
    },
    input,
    el("span", {}, el("strong", { text: "Add panels" }), el("br"), el("span", { class: "hint", text: "Drop PNG, JPEG or WebP, or click to choose" })),
  );
  return zone;
}

async function uploadFiles(files) {
  const images = files.filter((f) => /^image\/(png|jpeg|webp)$/.test(f.type));
  if (!images.length) return toast("Choose PNG, JPEG or WebP images", "error");
  let done = 0;
  for (const f of images) {
    const form = new FormData();
    form.set("image", f);
    form.set("caption", f.name.replace(/\.[a-z0-9]+$/i, "").slice(0, 280));
    try {
      await api(`/api/boards/${state.board.id}/panels`, { method: "POST", raw: form });
      done++;
    } catch (err) {
      fail(err);
    }
  }
  if (done) toast(`Added ${done} panel${done === 1 ? "" : "s"}`);
  await reloadBoard();
}

// ---- shot editor -------------------------------------------------------------

function paintShot() {
  const shot = currentShot();
  if (!shot) {
    return emptyState(
      state.panels.length < 2 ? "Add at least two panels" : "Pick a shot",
      "A shot is the motion between two adjacent panels: the first is where it starts, the next is where it lands.",
    );
  }
  const first = panelById(shot.start_panel_id);
  const last = panelById(shot.end_panel_id);
  const engines = state.config.engines;
  const prompt = el("textarea", { maxlength: 2000, placeholder: "What happens between the two frames: action, camera, mood." }, shot.prompt);
  const seconds = el("input", { type: "number", min: 1, max: 30, step: 1, value: shot.seconds });
  const seed = el("input", { type: "number", min: 0, max: 2147483647, step: 1, placeholder: "random", value: shot.seed ?? "" });
  let mode = shot.mode;
  let refs = new Set(shot.reference_panel_ids);

  const modeSeg = el("div", { class: "seg", role: "group", "aria-label": "Shot mode" });
  const refsWrap = el("div", { class: "field" });
  const ranges = engines.map((e) => `${e.label} ${e.seconds.min}–${e.seconds.max} s`).join(" · ");
  const common = {
    min: Math.max(...engines.map((e) => e.seconds.min)),
    max: Math.min(...engines.map((e) => e.seconds.max)),
  };

  const paintMode = () => {
    modeSeg.replaceChildren(
      ...[
        ["keyframes", "First + last"],
        ["reference", "Reference"],
      ].map(([value, label]) =>
        el("button", {
          type: "button",
          "aria-pressed": mode === value ? "true" : "false",
          text: label,
          onclick: () => {
            mode = value;
            paintMode();
          },
        }),
      ),
    );
    refsWrap.hidden = mode !== "reference";
    refsWrap.replaceChildren(
      el("span", { text: "Extra reference panels" }),
      el(
        "div",
        { class: "refs" },
        state.panels
          .filter((p) => p.id !== shot.start_panel_id && p.id !== shot.end_panel_id)
          .map((p) =>
            el(
              "button",
              {
                class: "ref-chip",
                type: "button",
                "aria-pressed": refs.has(p.id) ? "true" : "false",
                onclick: (e) => {
                  refs.has(p.id) ? refs.delete(p.id) : refs.add(p.id);
                  e.currentTarget.setAttribute("aria-pressed", refs.has(p.id) ? "true" : "false");
                },
              },
              el("img", { src: p.image_url, alt: "" }),
              p.caption || "Panel",
            ),
          ),
      ),
      el("p", { class: "hint", text: "Reference shots run on H3 only. Both panels of the shot are always sent." }),
    );
    for (const btn of document.querySelectorAll("[data-render-engine]")) {
      const id = btn.getAttribute("data-render-engine");
      const can = mode === "keyframes" || engines.find((e) => e.id === id)?.reference_model;
      btn.disabled = !state.config.engine_configured || !can;
    }
    const bake = $("#render-bakeoff");
    if (bake) bake.disabled = !state.config.engine_configured || mode === "reference";
  };

  const settings = () => ({
    prompt: prompt.value,
    seconds: Number(seconds.value),
    mode,
    seed: seed.value === "" ? null : Number(seed.value),
    reference_panel_ids: [...refs],
  });

  const save = async () => {
    const { shot: saved } = await api(`/api/boards/${state.board.id}/shots/${shot.start_panel_id}`, { method: "PUT", body: settings() });
    Object.assign(shot, saved);
    const link = document.querySelector(`.link[data-start="${CSS.escape(shot.start_panel_id)}"] .link-prompt`);
    if (link) link.textContent = saved.prompt || "No prompt yet";
    return saved;
  };

  const render = async (engineIds, tier = "draft") => {
    if (!prompt.value.trim()) {
      toast("Write a prompt for this shot first: the engines refuse an empty one", "error");
      prompt.focus();
      return;
    }
    try {
      await save();
      const out = await api(`/api/boards/${state.board.id}/shots/${shot.start_panel_id}/renders`, {
        method: "POST",
        body: { engines: engineIds, tier },
      });
      state.renders.push(...out.renders);
      const refused = out.renders.filter((r) => r.status === "failed");
      if (refused.length) toast(`${refused.map((r) => ENGINE_LABEL[r.engine]).join(", ")} refused: ${refused[0].error}`, "error");
      updateLive();
      startPolling();
    } catch (err) {
      fail(err);
    }
  };

  const form = el(
    "form",
    {
      class: "card shot-form",
      onsubmit: async (e) => {
        e.preventDefault();
        try {
          await save();
          toast("Shot saved");
        } catch (err) {
          fail(err);
        }
      },
    },
    el(
      "div",
      {},
      el("p", { class: "eyebrow", text: `Shot ${shot.index + 1} of ${state.shots.length}` }),
      el(
        "div",
        { class: "ends" },
        el("div", { class: "panel-frame" }, el("img", { src: first.image_url, alt: "First frame" })),
        el("span", { class: "ends-arrow", "aria-hidden": "true", text: "→" }),
        el("div", { class: "panel-frame" }, el("img", { src: last.image_url, alt: "Last frame" })),
      ),
    ),
    el("label", { class: "field" }, el("span", { text: "Prompt (needed to render)" }), prompt),
    el(
      "div",
      { class: "pair" },
      el("label", { class: "field" }, el("span", { text: "Length (s)" }), seconds),
      el("label", { class: "field" }, el("span", { text: "Seed" }), seed),
    ),
    el("p", {
      class: "hint",
      text: `${ranges}. A bake-off fits both at ${common.min}–${common.max} s; the engine has the final say.`,
    }),
    el("div", { class: "field" }, el("span", { text: "Mode" }), modeSeg),
    refsWrap,
    el(
      "div",
      { class: "actions" },
      el("button", {
        class: "btn primary",
        type: "button",
        id: "render-bakeoff",
        text: "Bake-off: render on both",
        onclick: () => render(state.config.engine_ids),
      }),
      el(
        "div",
        { class: "row" },
        engines.map((e) =>
          el("button", {
            class: "btn",
            type: "button",
            "data-engine": e.id,
            "data-render-engine": e.id,
            text: `${e.label} only`,
            onclick: () => render([e.id]),
          }),
        ),
      ),
      el("button", { class: "btn ghost", type: "submit", text: "Save shot" }),
      state.config.engine_configured ? null : el("p", { class: "hint", text: "No engine is configured, so renders are off. Set TWEEN_ENGINE_URL." }),
    ),
  );

  queueMicrotask(paintMode);
  return el("section", { class: "shot" }, form, paintCompare(shot));
}

// ---- side by side ---------------------------------------------------------

function paintCompare(shot) {
  const wrap = el("div", { class: "compare", id: "compare" });
  for (const e of state.config.engines) {
    const lane = el("article", { class: "lane", "data-engine": e.id });
    state.lanes.set(e.id, { node: lane, sig: null });
    wrap.append(lane);
  }
  wrap.append(
    el(
      "div",
      { class: "compare-bar" },
      el("button", { class: "btn small", type: "button", text: "Play both in sync", onclick: playBoth }),
      el("span", { class: "hint", text: "Real-time factor = clip length / engine generation time. Above 1× is faster than real time." }),
    ),
    el("div", { class: "history card", id: "history" }),
  );
  queueMicrotask(updateLive);
  return wrap;
}

function playBoth() {
  const videos = [...document.querySelectorAll(".lane video")];
  if (!videos.length) return toast("Nothing rendered for this shot yet");
  for (const v of videos) {
    v.pause();
    v.currentTime = 0;
  }
  Promise.all(videos.map((v) => v.play().catch(() => {}))).catch(() => {});
}

function laneFor(engineId, shot) {
  const engine = state.config.engines.find((e) => e.id === engineId);
  const rs = rendersFor(shot).filter((r) => r.engine === engineId);
  const r = latest(rs);
  return { engine, r };
}

function paintLane(engine, r) {
  const head = el(
    "header",
    { class: "lane-head" },
    el("h3", { text: engine.label }),
    r ? el("span", { class: `tag${r.tier === "final" ? " final" : ""}`, text: r.tier }) : null,
    r?.selected ? el("span", { class: "tag cut", text: "in the cut" }) : null,
    el("span", { class: "spacer" }),
    r ? el("span", { class: "mono faint", text: r.model }) : null,
  );
  let screen;
  if (!r) screen = el("div", { class: "lane-screen" }, el("div", { class: "state", text: "Not rendered yet" }));
  else if (r.status === "completed" && r.video_url)
    screen = el("div", { class: "lane-screen" }, el("video", { src: r.video_url, controls: true, playsinline: true, preload: "metadata", loop: true }));
  else if (r.status === "failed")
    screen = el("div", { class: "lane-screen" }, el("div", { class: "state error" }, el("strong", { text: "Failed" }), el("span", { text: r.error || "the engine failed this job" })));
  else
    screen = el(
      "div",
      { class: "lane-screen" },
      el(
        "div",
        { class: "state" },
        el("span", { text: r.status === "queued" ? "Queued" : `Rendering ${r.progress}%` }),
        el("div", { class: "bar" }, el("i", { style: `width:${r.progress}%` })),
      ),
    );

  const f = r?.factors;
  const factorClass = f?.inference !== null && f?.inference !== undefined ? (f.inference > 1 ? "fast" : "slow") : "";
  const stats = el(
    "div",
    { class: "lane-stats" },
    el("div", { class: "stat" }, el("b", { class: factorClass, text: fmt.factor(f?.inference) }), el("span", { text: "real time" })),
    el("div", { class: "stat" }, el("b", { text: fmt.secs(r?.inference_time_s) }), el("span", { text: "engine" })),
    el("div", { class: "stat" }, el("b", { text: fmt.secs(r?.wall_time_s) }), el("span", { text: `wall ${fmt.factor(f?.wall)}` })),
  );
  const foot = el(
    "footer",
    { class: "lane-foot" },
    el("span", { class: "mono faint", text: r ? `seed ${r.seed} · ${r.seconds}s${r.size ? ` · ${r.size}` : ""}` : "" }),
    el("span", { class: "spacer" }),
    r?.status === "completed" && !r.selected
      ? el("button", { class: "btn small", type: "button", text: "Use in cut", onclick: () => selectRender(r) })
      : null,
    r?.status === "completed" && r.tier === "draft"
      ? el("button", { class: "btn small", type: "button", text: "Finalize", disabled: !state.config.engine_configured, onclick: () => finalize(r) })
      : null,
  );
  return [head, screen, stats, foot];
}

async function selectRender(r) {
  try {
    const { render } = await api(`/api/renders/${r.id}/select`, { method: "POST" });
    for (const x of state.renders) if (x.start_panel_id === render.start_panel_id) x.selected = x.id === render.id;
    updateLive(true);
    toast(`${ENGINE_LABEL[render.engine]} take is in the cut`);
  } catch (err) {
    fail(err);
  }
}

async function finalize(r) {
  try {
    const out = await api(`/api/renders/${r.id}/finalize`, { method: "POST" });
    state.renders.push(...out.renders);
    const bad = out.renders.find((x) => x.status === "failed");
    if (bad) toast(`Final render refused: ${bad.error}`, "error");
    else toast(`Final render queued on ${ENGINE_LABEL[r.engine]} with seed ${r.seed}`);
    updateLive();
    startPolling();
  } catch (err) {
    fail(err);
  }
}

function paintHistory(shot) {
  const box = $("#history");
  if (!box) return;
  const rs = [...rendersFor(shot)].sort((a, b) => b.created_at - a.created_at);
  if (!rs.length) {
    box.replaceChildren(el("p", { class: "muted", style: "margin:0", text: "No renders for this shot yet." }));
    return;
  }
  const shown = new Set(state.config.engine_ids.map((e) => latest(rs.filter((r) => r.engine === e))?.id));
  box.replaceChildren(
    el("h3", { text: "Takes", style: "margin-bottom:8px" }),
    el(
      "table",
      {},
      el("thead", {}, el("tr", {}, ["Engine", "Tier", "Status", "Real time", "Engine", "Seed", "When", ""].map((h) => el("th", { text: h })))),
      el(
        "tbody",
        {},
        rs.map((r) =>
          el(
            "tr",
            { "data-active": shown.has(r.id) ? "true" : "false" },
            el("td", { text: ENGINE_LABEL[r.engine] }),
            el("td", { text: r.tier }),
            r.status === "failed" ? el("td", { class: "err", text: r.error || "failed" }) : el("td", { text: isOpen(r) ? `${r.status.replace("_", " ")} ${r.progress}%` : r.selected ? "in the cut" : "done" }),
            el("td", { class: "mono", text: fmt.factor(r.factors?.inference) }),
            el("td", { class: "mono", text: fmt.secs(r.inference_time_s) }),
            el("td", { class: "mono", text: String(r.seed) }),
            el("td", { class: "faint", text: fmt.ago(r.created_at) }),
            el(
              "td",
              {},
              r.status === "completed" && !r.selected ? el("button", { class: "btn small ghost", type: "button", text: "Use", onclick: () => selectRender(r) }) : null,
            ),
          ),
        ),
      ),
    ),
  );
}

/** Updates lanes, link dots and history in place, so typing and playback survive polling. */
function updateLive(force = false) {
  const shot = currentShot();
  if (shot) {
    for (const [engineId, lane] of state.lanes) {
      const { engine, r } = laneFor(engineId, shot);
      const sig = r ? `${r.id}:${r.status}:${r.progress}:${r.selected}` : "none";
      if (!force && sig === lane.sig) continue;
      lane.sig = sig;
      lane.node.replaceChildren(...paintLane(engine, r));
    }
    paintHistory(shot);
  }
  for (const s of state.shots) {
    const link = document.querySelector(`.link[data-start="${CSS.escape(s.start_panel_id)}"]`);
    if (!link) continue;
    const dots = link.querySelectorAll(".dot");
    linkStatus(s).forEach(({ status }, i) => dots[i] && (dots[i].dataset.status = status));
  }
}

// ---- polling ---------------------------------------------------------------

function startPolling() {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(poll, POLL_MS);
}

function stopPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = null;
}

let polling = false;
async function poll() {
  if (polling || document.hidden) return;
  const open = state.renders.filter(isOpen);
  if (!open.length) return stopPolling();
  polling = true;
  try {
    const fresh = await Promise.all(open.map((r) => api(`/api/renders/${r.id}`).then((x) => x.render).catch(() => r)));
    for (const f of fresh) {
      const i = state.renders.findIndex((r) => r.id === f.id);
      if (i >= 0) state.renders[i] = f;
      if (f.status === "failed" && open.find((o) => o.id === f.id)?.status !== "failed") {
        toast(`${ENGINE_LABEL[f.engine]} failed: ${f.error}`, "error");
      }
    }
    updateLive();
  } finally {
    polling = false;
  }
}

// ---- animatic ----------------------------------------------------------------

const reel = {
  dialog: $("#reel"),
  video: $("#reel-video"),
  still: $("#reel-still"),
  gap: $("#reel-gap"),
  entries: [],
  index: 0,
  playing: false,
  gapTimer: null,
  gapStart: 0,
};

const entryLength = (e) => (e.render ? (e.render.clip_seconds ?? e.render.seconds) : e.shot.seconds);
const startOffset = (i) => reel.entries.slice(0, i).reduce((t, e) => t + entryLength(e), 0);
const totalLength = () => startOffset(reel.entries.length);

async function openReel(source) {
  try {
    const data = await api(`/api/boards/${state.board.id}/animatic?source=${encodeURIComponent(source)}`);
    reel.entries = data.entries;
    if (!reel.entries.length) return toast("Add at least two panels first");
    $("#reel-source").textContent = source === "cut" ? "The cut" : `${ENGINE_LABEL[source]} reel`;
    const missing = reel.entries.filter((e) => !e.render).length;
    $("#reel-title").textContent = missing
      ? `${state.board.title} · ${missing} of ${reel.entries.length} shots not rendered`
      : state.board.title;
    paintTimeline();
    reel.dialog.showModal();
    showEntry(0, false);
  } catch (err) {
    fail(err);
  }
}

function paintTimeline() {
  const total = totalLength();
  $("#reel-timeline").replaceChildren(
    ...reel.entries.map((e, i) =>
      el(
        "button",
        {
          class: "reel-seg",
          type: "button",
          "data-has": e.render ? "true" : "false",
          "data-engine": e.render?.engine,
          style: `flex:${entryLength(e) / total}`,
          title: `Shot ${i + 1}${e.render ? ` · ${ENGINE_LABEL[e.render.engine]} ${e.render.tier}` : " · not rendered"}`,
          onclick: () => showEntry(i, reel.playing),
        },
        el("i", {}),
        String(i + 1),
      ),
    ),
  );
}

function showEntry(i, autoplay) {
  clearInterval(reel.gapTimer);
  reel.index = i;
  const e = reel.entries[i];
  document.querySelectorAll(".reel-seg").forEach((s, k) => s.setAttribute("aria-current", k === i ? "true" : "false"));
  if (e.render) {
    reel.still.hidden = true;
    reel.gap.hidden = true;
    reel.video.hidden = false;
    reel.video.src = e.render.video_url;
    if (autoplay) reel.video.play().catch(() => {});
  } else {
    reel.video.pause();
    reel.video.hidden = true;
    reel.still.hidden = false;
    reel.gap.hidden = false;
    reel.still.src = panelById(e.shot.start_panel_id)?.image_url ?? "";
    reel.gapStart = performance.now();
    if (autoplay) runGap();
  }
  tickClock();
}

function runGap() {
  const e = reel.entries[reel.index];
  reel.gapStart = performance.now();
  reel.gapTimer = setInterval(() => {
    tickClock();
    if ((performance.now() - reel.gapStart) / 1000 >= e.shot.seconds) advance();
  }, 100);
}

function advance() {
  clearInterval(reel.gapTimer);
  if (reel.index + 1 < reel.entries.length) showEntry(reel.index + 1, true);
  else {
    reel.playing = false;
    $("#reel-play").textContent = "Play";
  }
}

function elapsedInEntry() {
  const e = reel.entries[reel.index];
  if (!e) return 0;
  return e.render ? reel.video.currentTime || 0 : reel.playing ? Math.min(e.shot.seconds, (performance.now() - reel.gapStart) / 1000) : 0;
}

function tickClock() {
  const e = reel.entries[reel.index];
  if (!e) return;
  const at = startOffset(reel.index) + elapsedInEntry();
  $("#reel-clock").textContent = `${fmt.clock(at)} / ${fmt.clock(totalLength())}`;
  const seg = document.querySelectorAll(".reel-seg i")[reel.index];
  if (seg) seg.style.width = `${Math.min(100, (elapsedInEntry() / entryLength(e)) * 100)}%`;
}

reel.video.addEventListener("ended", advance);
reel.video.addEventListener("timeupdate", tickClock);
$("#reel-play").addEventListener("click", () => {
  reel.playing = !reel.playing;
  $("#reel-play").textContent = reel.playing ? "Pause" : "Play";
  const e = reel.entries[reel.index];
  if (!e) return;
  if (reel.playing) {
    if (reel.index === reel.entries.length - 1 && elapsedInEntry() >= entryLength(e) - 0.05) return showEntry(0, true);
    if (e.render) reel.video.play().catch(() => {});
    else runGap();
  } else {
    reel.video.pause();
    clearInterval(reel.gapTimer);
  }
});
$("#reel-close").addEventListener("click", () => reel.dialog.close());
reel.dialog.addEventListener("close", () => {
  reel.video.pause();
  clearInterval(reel.gapTimer);
  reel.playing = false;
  $("#reel-play").textContent = "Play";
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.renders.some(isOpen)) startPolling();
});

route();
