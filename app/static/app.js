// Router + view controllers for the Intelligent PDF Parser console.
import { api } from "/static/api.js";
import {
  $, $$, esc, fmtBytes, fmtTime, statusBadge, toast,
  renderStats, renderPages, renderQA, renderLogLine, renderChatMessage,
} from "/static/ui.js";

const view = $("#view");
let pollTimer = null;          // active polling interval, cleared on navigation
let samplesCache = null;

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

function setHeader(title, crumb = "", actionsHtml = "") {
  $("#view-title").textContent = title;
  $("#view-crumb").innerHTML = crumb;
  $("#topbar-actions").innerHTML = actionsHtml;
}

function setActiveNav(route) {
  $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === route));
}

// --------------------------------------------------------------------------- //
// Health
// --------------------------------------------------------------------------- //
async function loadHealth() {
  const el = $("#health");
  try {
    const d = await api.health();
    el.innerHTML = `<span class="dot ok"></span> ${esc(d.status)} · v${esc(d.version)} · ${esc(d.env)}`;
  } catch {
    el.innerHTML = `<span class="dot bad"></span> offline`;
  }
}

async function getSamples() {
  if (samplesCache) return samplesCache;
  try { samplesCache = (await api.samples()).samples || []; }
  catch { samplesCache = []; }
  return samplesCache;
}

// --------------------------------------------------------------------------- //
// View: New Run
// --------------------------------------------------------------------------- //
async function viewNew() {
  setActiveNav("new");
  setHeader("New Run", "Parse a document and optionally generate analyst Q&amp;A");
  const [samples, config] = await Promise.all([getSamples(), api.config().catch(() => null)]);

  const sampleOpts = samples.length
    ? samples.map((s) => `<option value="${esc(s.name)}">${esc(s.name)} · ${fmtBytes(s.size_bytes)}</option>`).join("")
    : `<option value="">no samples on server</option>`;

  const prov = config?.providers || {};
  const provLine = config
    ? `LLM <strong>${esc(prov.llm?.name)}</strong> (${esc(prov.llm?.model)}) ${prov.llm?.available ? "✓ ready" : "— no key"} ·
       Vision <strong>${esc(prov.vision?.name)}</strong> (${esc(prov.vision?.model)}) ${prov.vision?.available ? "✓ ready" : "— no key"}`
    : "config unavailable";

  view.innerHTML = `
    <div class="grid-side">
      <div class="card">
        <div class="card-head"><h3>Source &amp; mode</h3></div>
        <div class="card-body">
          <label class="field"><span>Source</span>
            <div class="seg" id="src-seg">
              <button type="button" data-src="sample" aria-pressed="true">Sample</button>
              <button type="button" data-src="upload" aria-pressed="false">Upload</button>
            </div>
          </label>
          <label class="field" id="f-sample"><span>Sample PDF</span>
            <select id="sample-select">${sampleOpts}</select>
          </label>
          <label class="field hidden" id="f-upload"><span>PDF file</span>
            <input type="file" id="file-input" accept="application/pdf,.pdf" />
          </label>
          <label class="field"><span>Mode</span>
            <div class="seg" id="mode-seg">
              <button type="button" data-mode="parse" aria-pressed="false">Parse only</button>
              <button type="button" data-mode="process" aria-pressed="true">Parse + Q&amp;A</button>
            </div>
          </label>
          <label class="field" id="f-nq"><span>Analyst questions</span>
            <input type="number" id="num-questions" value="12" min="1" max="40" />
          </label>
          <button class="btn block" id="start-btn">Start run</button>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><h3>Active configuration</h3></div>
        <div class="card-body">
          <p class="muted" style="margin:0 0 0.8rem;font-size:0.86rem;line-height:1.6">${provLine}</p>
          <p class="faint" style="font-size:0.8rem;line-height:1.6;margin:0">
            Parser backend: <span class="mono">${esc(config?.parsing?.parser_backend || "—")}</span><br>
            Runs are saved to history with full logs and an interactive Q&amp;A chat.
          </p>
          <a href="#/settings" class="ghost" style="margin-top:0.9rem">Adjust settings →</a>
        </div>
      </div>
    </div>`;

  const state = { src: "sample", mode: "process" };
  $("#src-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    state.src = b.dataset.src;
    $$("#src-seg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    $("#f-sample").classList.toggle("hidden", state.src !== "sample");
    $("#f-upload").classList.toggle("hidden", state.src !== "upload");
  });
  $("#mode-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    state.mode = b.dataset.mode;
    $$("#mode-seg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    $("#f-nq").classList.toggle("hidden", state.mode !== "process");
  });

  $("#start-btn").addEventListener("click", async () => {
    const btn = $("#start-btn");
    const payload = { source: state.src, mode: state.mode };
    if (state.src === "sample") {
      payload.name = $("#sample-select").value;
      if (!payload.name) return toast("No sample selected", "bad");
    } else {
      payload.file = $("#file-input").files[0];
      if (!payload.file) return toast("Choose a PDF file first", "bad");
    }
    if (state.mode === "process") payload.numQuestions = Number($("#num-questions").value) || 12;
    btn.disabled = true;
    btn.innerHTML = `<span class="spin">↻</span> starting…`;
    try {
      const { run_id } = await api.startRun(payload);
      toast("Run started", "ok");
      location.hash = `#/run/${run_id}`;
    } catch (e) {
      toast(`Failed to start: ${e.message}`, "bad");
      btn.disabled = false;
      btn.textContent = "Start run";
    }
  });
}

// --------------------------------------------------------------------------- //
// View: History
// --------------------------------------------------------------------------- //
function historyRow(r) {
  const mode = r.mode === "process" ? "Parse + Q&A" : "Parse";
  return `<tr data-id="${esc(r.id)}">
    <td>${statusBadge(r.status)}</td>
    <td class="fname">${esc(r.filename)}</td>
    <td><span class="badge dim">${mode}</span></td>
    <td class="mono">${r.num_pages ?? "—"}</td>
    <td class="mono">${r.llm_provider ? esc(r.llm_provider) : "—"}</td>
    <td class="when">${fmtTime(r.created_at)}</td>
  </tr>`;
}

async function renderHistoryTable() {
  const tbody = $("#runs-tbody");
  if (!tbody) return false;
  try {
    const { runs, total } = await api.listRuns(100);
    $("#runs-count").textContent = `${total} run${total === 1 ? "" : "s"}`;
    if (!runs.length) {
      tbody.innerHTML = `<tr><td colspan="6"><div class="placeholder" style="border:0">No runs yet. Start one from <a href="#/new">New Run</a>.</div></td></tr>`;
      return false;
    }
    tbody.innerHTML = runs.map(historyRow).join("");
    return runs.some((r) => r.status === "running" || r.status === "queued");
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="6"><div class="err">${esc(e.message)}</div></td></tr>`;
    return false;
  }
}

async function viewHistory() {
  setActiveNav("history");
  setHeader("Run History", `<span id="runs-count">…</span>`,
    `<button class="ghost" id="refresh-runs">↻ Refresh</button>`);
  view.innerHTML = `
    <div class="card">
      <table class="table">
        <thead><tr>
          <th>Status</th><th>Document</th><th>Mode</th><th>Pages</th><th>LLM</th><th>When</th>
        </tr></thead>
        <tbody id="runs-tbody"><tr><td colspan="6"><div class="placeholder" style="border:0">loading…</div></td></tr></tbody>
      </table>
    </div>`;

  $("#runs-tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr) location.hash = `#/run/${tr.dataset.id}`;
  });
  $("#refresh-runs").addEventListener("click", renderHistoryTable);

  const hasActive = await renderHistoryTable();
  if (hasActive) {
    pollTimer = setInterval(async () => {
      const still = await renderHistoryTable();
      if (!still) stopPolling();
    }, 2500);
  }
}

// --------------------------------------------------------------------------- //
// View: Run detail
// --------------------------------------------------------------------------- //
const TABS = [
  { id: "output", label: "Output" },
  { id: "qa", label: "Analyst Q&A" },
  { id: "ask", label: "Ask the document" },
  { id: "logs", label: "Logs" },
];

async function viewRun(id) {
  setActiveNav("history");
  setHeader("Run", `<span class="mono">${esc(id.slice(0, 8))}</span>`,
    `<button class="ghost" id="back-btn">← History</button>
     <button class="ghost danger" id="del-btn">Delete</button>`);
  view.innerHTML = `<div class="placeholder">loading run…</div>`;

  $("#back-btn").addEventListener("click", () => { location.hash = "#/history"; });
  $("#del-btn").addEventListener("click", async () => {
    if (!confirm("Delete this run and its logs and chat?")) return;
    try { await api.deleteRun(id); toast("Run deleted", "ok"); location.hash = "#/history"; }
    catch (e) { toast(e.message, "bad"); }
  });

  let run;
  try { run = await api.getRun(id); }
  catch (e) { view.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }

  // While a run is in flight, open on Logs so progress is visible immediately.
  const active = run.status === "running" || run.status === "queued";
  const ctx = { id, run, tab: active ? "logs" : "output", logSeq: 0, chatLoaded: false, chatBusy: false };

  view.innerHTML = `
    <div id="run-summary"></div>
    <div class="tabs" id="run-tabs">
      ${TABS.map((t) => `<button data-tab="${t.id}">${t.label}${t.id === "qa" && run.qa ? `<span class="count">${run.qa.items?.length || 0}</span>` : ""}</button>`).join("")}
    </div>
    <div id="tab-body"></div>`;

  $("#run-tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]"); if (!b) return;
    ctx.tab = b.dataset.tab; renderTabs(ctx);
  });

  renderSummary(ctx);
  renderTabs(ctx);

  if (run.status === "running" || run.status === "queued") {
    pollTimer = setInterval(() => pollRun(ctx), 1300);
  }
}

function renderSummary(ctx) {
  const r = ctx.run;
  const head = `<div class="card"><div class="card-body">
    <div style="display:flex;align-items:center;gap:0.7rem;flex-wrap:wrap;margin-bottom:0.9rem">
      ${statusBadge(r.status)}
      <strong style="font-size:1.02rem">${esc(r.filename)}</strong>
      <span class="badge dim">${r.mode === "process" ? "Parse + Q&A" : "Parse"}</span>
      <span class="grow" style="flex:1"></span>
      <span class="mono faint" style="font-size:0.76rem">${esc((r.created_at || "").replace("T", " ").slice(0, 19))}</span>
    </div>
    ${r.error ? `<div class="err">${esc(r.error)}</div>` : renderStats(r)}
    ${(r.parse?.warnings || []).length ? `<div class="warn-box" style="margin-top:0.8rem">⚠ ${r.parse.warnings.map(esc).join("<br>")}</div>` : ""}
  </div></div>`;
  $("#run-summary").innerHTML = head;
}

function renderTabs(ctx) {
  $$("#run-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === ctx.tab));
  const body = $("#tab-body");
  const r = ctx.run;
  if (ctx.tab === "output") {
    body.innerHTML = `<div class="card"><div class="card-head"><h3>Parsed pages</h3></div>
      <div class="card-body">${renderPages(r.parse)}</div></div>`;
  } else if (ctx.tab === "qa") {
    body.innerHTML = `<div class="card"><div class="card-head"><h3>Analyst Q&amp;A</h3></div>
      <div class="card-body">${renderQA(r.qa)}</div></div>`;
  } else if (ctx.tab === "ask") {
    renderAskTab(ctx);
  } else if (ctx.tab === "logs") {
    renderLogsTab(ctx);
  }
}

// ---- Ask tab (interactive grounded chat) ----
function renderAskTab(ctx) {
  const ready = ctx.run.status === "completed" && ctx.run.has_markdown;
  const body = $("#tab-body");
  if (!ready) {
    body.innerHTML = `<div class="placeholder">Ask becomes available once the run has completed parsing.</div>`;
    return;
  }
  body.innerHTML = `<div class="card"><div class="card-head"><h3>Ask the document</h3>
      <span class="grow"></span><span class="faint mono" style="font-size:0.7rem">grounded · cites pages</span></div>
    <div class="card-body">
      <div class="suggestions" id="suggestions">
        ${["Summarize this document", "What are the key figures?", "Any risks or caveats mentioned?"]
          .map((q) => `<button class="ghost" data-q="${esc(q)}">${esc(q)}</button>`).join("")}
      </div>
      <div class="chat-wrap">
        <div class="chat-log" id="chat-log"><div class="placeholder" style="border:0">Ask anything about this document.</div></div>
        <div class="chat-input">
          <textarea id="chat-text" placeholder="Ask a question about this document…" rows="1"></textarea>
          <button class="btn" id="chat-send">Send</button>
        </div>
      </div>
    </div></div>`;

  const send = (q) => sendChat(ctx, q);
  $("#suggestions").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-q]"); if (b) send(b.dataset.q);
  });
  $("#chat-send").addEventListener("click", () => send($("#chat-text").value));
  $("#chat-text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("#chat-text").value); }
  });

  if (!ctx.chatLoaded) loadChat(ctx);
  else paintChat(ctx);
}

async function loadChat(ctx) {
  try {
    ctx.chat = (await api.chat(ctx.id)).messages || [];
    ctx.chatLoaded = true;
    paintChat(ctx);
  } catch (e) { toast(e.message, "bad"); }
}

function paintChat(ctx) {
  const log = $("#chat-log");
  if (!log) return;
  const msgs = ctx.chat || [];
  log.innerHTML = msgs.length
    ? msgs.map(renderChatMessage).join("") +
      (ctx.chatBusy ? `<div class="msg assistant"><div class="bubble"><span class="spin">↻</span> thinking…</div></div>` : "")
    : (ctx.chatBusy
        ? `<div class="msg assistant"><div class="bubble"><span class="spin">↻</span> thinking…</div></div>`
        : `<div class="placeholder" style="border:0">Ask anything about this document.</div>`);
  log.scrollTop = log.scrollHeight;
}

async function sendChat(ctx, raw) {
  const q = (raw || "").trim();
  if (!q || ctx.chatBusy) return;
  ctx.chat = ctx.chat || [];
  ctx.chat.push({ role: "user", content: q, sources: [] });
  ctx.chatBusy = true;
  $("#chat-text").value = "";
  paintChat(ctx);
  try {
    const msg = await api.ask(ctx.id, q);
    ctx.chat.push(msg);
  } catch (e) {
    ctx.chat.push({ role: "assistant", content: `⚠️ ${e.message}`, sources: [] });
  } finally {
    ctx.chatBusy = false;
    paintChat(ctx);
  }
}

// ---- Logs tab ----
async function renderLogsTab(ctx) {
  const body = $("#tab-body");
  body.innerHTML = `<div class="card"><div class="card-head"><h3>Run logs</h3>
      <span class="grow"></span><span id="log-status">${statusBadge(ctx.run.status)}</span></div>
    <div class="card-body"><div class="console" id="console"><div class="placeholder" style="border:0">loading logs…</div></div></div></div>`;
  try {
    const { logs } = await api.runLogs(ctx.id, 0);
    ctx.allLogs = logs;
    const c = $("#console");
    c.innerHTML = logs.length ? logs.map(renderLogLine).join("")
      : `<div class="placeholder" style="border:0">No log lines recorded.</div>`;
    c.scrollTop = c.scrollHeight;
  } catch (e) {
    $("#console").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

// ---- Live polling for an in-flight run ----
async function pollRun(ctx) {
  try {
    const { status, logs } = await api.runLogs(ctx.id, ctx.logSeq);
    if (logs.length) {
      ctx.logSeq = logs[logs.length - 1].seq;
      const c = $("#console");
      if (c) {
        if (c.querySelector(".placeholder")) c.innerHTML = "";
        c.insertAdjacentHTML("beforeend", logs.map(renderLogLine).join(""));
        c.scrollTop = c.scrollHeight;
      }
    }
    if (status !== ctx.run.status) {
      const sEl = $("#log-status"); if (sEl) sEl.innerHTML = statusBadge(status);
    }
    if (status === "completed" || status === "failed") {
      stopPolling();
      ctx.run = await api.getRun(ctx.id);
      renderSummary(ctx);
      // Refresh tab counts + current tab content now that results exist.
      $("#run-tabs").innerHTML = TABS.map((t) =>
        `<button data-tab="${t.id}" class="${t.id === ctx.tab ? "active" : ""}">${t.label}${t.id === "qa" && ctx.run.qa ? `<span class="count">${ctx.run.qa.items?.length || 0}</span>` : ""}</button>`).join("");
      if (ctx.tab !== "logs") renderTabs(ctx);
      toast(`Run ${status}`, status === "completed" ? "ok" : "bad");
    }
  } catch { /* transient; keep polling */ }
}

// --------------------------------------------------------------------------- //
// View: Settings
// --------------------------------------------------------------------------- //
function settingField(key, value, schema, overridden) {
  const s = schema[key] || { type: "str" };
  const tag = overridden.includes(key) ? `<span class="badge ok" style="margin-left:0.4rem">override</span>` : "";
  let control;
  if (s.type === "bool") {
    control = `<input type="checkbox" id="set-${key}" ${value ? "checked" : ""} />`;
  } else if (s.type === "enum") {
    const opts = [...(s.nullable ? [""] : []), ...s.options];
    control = `<select id="set-${key}">${opts.map((o) =>
      `<option value="${esc(o)}" ${String(value ?? "") === o ? "selected" : ""}>${o === "" ? "— default —" : esc(o)}</option>`).join("")}</select>`;
  } else if (s.type === "int" || s.type === "float") {
    const step = s.type === "float" ? "0.01" : "1";
    control = `<input type="number" id="set-${key}" value="${esc(value ?? "")}" step="${step}" ${s.min != null ? `min="${s.min}"` : ""} ${s.max != null ? `max="${s.max}"` : ""} />`;
  } else {
    control = `<input type="text" id="set-${key}" value="${esc(value ?? "")}" />`;
  }
  return `<label class="field"><span>${esc(key)}${tag}</span>${control}</label>`;
}

const SETTING_GROUPS = [
  { title: "Application", keys: ["log_level", "max_upload_mb", "default_num_questions"] },
  { title: "Providers", keys: ["llm_provider", "vision_provider"] },
  { title: "Parsing", keys: ["parser_backend", "force_vision_llm", "parser_confidence_threshold", "handwriting_scan_char_threshold", "handwriting_image_area_threshold", "handwriting_text_area_threshold", "vision_image_detail", "vision_render_dpi", "vision_max_concurrency"] },
  { title: "Models", keys: ["openai_model", "openai_vision_model", "anthropic_model", "anthropic_vision_model", "ollama_base_url", "ollama_model", "ollama_vision_model", "azure_openai_deployment", "azure_openai_vision_deployment", "azure_openai_api_version"] },
];

async function viewSettings() {
  setActiveNav("settings");
  setHeader("Settings", "Operational knobs (secrets stay in the environment)",
    `<button class="btn" id="save-settings">Save changes</button>`);
  view.innerHTML = `<div class="placeholder">loading settings…</div>`;

  let data;
  try { data = await api.getSettings(); }
  catch (e) { view.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }

  const { values, schema, overridden, keys_present, backends_available } = data;

  const keyChips = Object.entries(keys_present).map(([k, v]) =>
    `<span class="chip ${v ? "on" : "off"}"><span class="dot ${v ? "ok" : "bad"}"></span>${esc(k)}</span>`).join("");
  const backendChips = Object.entries(backends_available).map(([k, v]) =>
    `<span class="chip ${v ? "on" : "off"}"><span class="dot ${v ? "ok" : "bad"}"></span>${esc(k)}</span>`).join("");

  const groups = SETTING_GROUPS.map((g) => `
    <div class="card"><div class="card-head"><h3>${g.title}</h3></div>
      <div class="card-body grid-2">
        ${g.keys.filter((k) => k in values).map((k) => settingField(k, values[k], schema, overridden)).join("")}
      </div></div>`).join("");

  view.innerHTML = `
    <div class="card"><div class="card-head"><h3>Credentials &amp; backends</h3></div>
      <div class="card-body">
        <p class="faint" style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.5rem">API keys present</p>
        <div style="display:flex;flex-wrap:wrap;gap:0.45rem;margin-bottom:1rem">${keyChips}</div>
        <p class="faint" style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.5rem">Parser backends available</p>
        <div style="display:flex;flex-wrap:wrap;gap:0.45rem">${backendChips}</div>
      </div></div>
    ${groups}`;

  $("#save-settings").addEventListener("click", () => saveSettings(values, schema));
}

async function saveSettings(current, schema) {
  const changed = {};
  for (const key of Object.keys(current)) {
    const el = $(`#set-${key}`);
    if (!el) continue;
    const s = schema[key] || { type: "str" };
    let val;
    if (s.type === "bool") val = el.checked;
    else if (s.type === "int") val = el.value === "" ? null : parseInt(el.value, 10);
    else if (s.type === "float") val = el.value === "" ? null : parseFloat(el.value);
    else val = el.value === "" ? null : el.value;
    // Normalize for comparison; only send genuine changes.
    const cur = current[key] ?? null;
    if (JSON.stringify(val) !== JSON.stringify(cur)) changed[key] = val;
  }
  if (!Object.keys(changed).length) return toast("No changes to save");
  const btn = $("#save-settings");
  btn.disabled = true; btn.innerHTML = `<span class="spin">↻</span> saving…`;
  try {
    await api.updateSettings(changed);
    toast("Settings saved", "ok");
    viewSettings();
  } catch (e) {
    toast(e.message, "bad");
    btn.disabled = false; btn.textContent = "Save changes";
  }
}

// --------------------------------------------------------------------------- //
// Router
// --------------------------------------------------------------------------- //
function router() {
  stopPolling();
  const hash = location.hash || "#/new";
  const [, route, param] = hash.split("/");
  if (route === "history") viewHistory();
  else if (route === "run" && param) viewRun(param);
  else if (route === "settings") viewSettings();
  else viewNew();
}

window.addEventListener("hashchange", router);
loadHealth();
router();
