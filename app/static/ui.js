// DOM helpers, formatters, and HTML render fragments shared across views.

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];

export const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function fmtBytes(n) {
  if (n == null) return "—";
  return n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : (n / 1024).toFixed(0) + " KB";
}

export function fmtDuration(ms) {
  if (ms == null) return "—";
  if (ms < 1000) return Math.round(ms) + " ms";
  return (ms / 1000).toFixed(1) + " s";
}

export function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return Math.floor(diff / 60) + "m ago";
  if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export const pct = (c) => Math.round((c || 0) * 100);

const STATUS_KIND = { completed: "ok", failed: "bad", running: "run", queued: "dim" };

export function statusBadge(status) {
  const kind = STATUS_KIND[status] || "dim";
  const pulse = status === "running" || status === "queued";
  return `<span class="badge ${kind}"><span class="dot ${kind} ${pulse ? "pulse" : ""}"></span>${esc(status)}</span>`;
}

// Minimal, safe markdown for chat answers (escape first, then decorate).
export function mdLite(text) {
  let h = esc(text);
  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
  h = h.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  h = h.replace(/\n/g, "<br>");
  return h;
}

let toastSeq = 0;
export function toast(message, kind = "") {
  const wrap = $("#toasts");
  if (!wrap) return;
  const id = "t" + toastSeq++;
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.id = id;
  node.textContent = message;
  wrap.appendChild(node);
  setTimeout(() => node.remove(), 4200);
}

// ---- HTML fragments ------------------------------------------------------ //

export function renderStats(run) {
  const t = run.timings_ms || {};
  const bs = (run.parse && run.parse.backend_summary) || {};
  const backends = Object.entries(bs).map(([k, v]) => `${k}:${v}`).join("  ") || "—";
  return `<div class="stat-row">
    <div class="stat"><div class="n">${run.num_pages ?? "—"}</div><div class="l">pages</div></div>
    <div class="stat"><div class="n">${fmtDuration(run.duration_ms)}</div><div class="l">total</div></div>
    <div class="stat"><div class="n">${t.parse != null ? Math.round(t.parse) : "—"}</div><div class="l">parse ms</div></div>
    ${t.qa != null ? `<div class="stat"><div class="n">${Math.round(t.qa)}</div><div class="l">q&a ms</div></div>` : ""}
    <div class="stat"><div class="n" style="font-size:0.92rem;line-height:1.7">${esc(backends)}</div><div class="l">backends</div></div>
  </div>`;
}

export function renderPages(parse) {
  if (!parse || !parse.pages || !parse.pages.length)
    return `<div class="placeholder">No parsed pages.</div>`;
  return parse.pages.map((p) => {
    const hw = p.has_handwriting ? `<span class="badge hw">handwriting</span>` : "";
    const notes = (p.notes || []).length
      ? `<div class="notes">⚐ ${p.notes.map(esc).join(" · ")}</div>` : "";
    return `<article class="page">
      <div class="page-head">
        <span class="pno">p${p.page_number}</span>
        <span class="badge">${esc(p.backend || "—")}</span>
        <span class="badge dim">${esc(p.kind)}</span>${hw}
        <span class="conf"><span class="bar"><i style="width:${pct(p.confidence)}%"></i></span><span class="pct">${pct(p.confidence)}%</span></span>
      </div>
      <pre class="md">${esc(p.markdown) || "<span class='faint'>(empty)</span>"}</pre>
      ${notes}
    </article>`;
  }).join("");
}

// Web-sourced peer citations. Only http(s) links are rendered (defensive
// against javascript:/data: schemes), each opened in a new tab with noopener.
function renderCitations(citations) {
  if (!citations || !citations.length) return "";
  const links = citations
    .filter((c) => c && /^https?:\/\//i.test(c.url || ""))
    .map((c) => {
      const label = esc(c.title || c.url);
      const attrib = [c.peer, c.venue].filter(Boolean).map(esc).join(", ");
      const who = attrib ? `${attrib} — ` : "";
      const quote = c.quote ? `<span class="cite-quote">“${esc(c.quote.trim())}”</span>` : "";
      return `<li>📎 <a href="${esc(c.url)}" target="_blank" rel="noopener noreferrer">${who}${label}</a>${quote}</li>`;
    })
    .join("");
  return links ? `<ul class="qa-citations">${links}</ul>` : "";
}

export function renderQA(qa) {
  if (!qa || !qa.items || !qa.items.length)
    return `<div class="placeholder">No analyst Q&A for this run.<br>Run in <strong>Parse + Q&A</strong> mode to generate it.</div>`;
  const head = [qa.company, qa.period].filter(Boolean).map(esc).join(" — ");
  const items = qa.items.map((it) => `
    <article class="qa-item">
      <p class="qa-q">${esc(it.question)}</p>
      <p class="qa-a">${esc(it.answer)}</p>
      ${it.reasoning ? `<div class="qa-reason">💡 <em>Why it's asked:</em> ${esc(it.reasoning)}</div>` : ""}
      ${it.peer_context ? `<div class="qa-peer">🔍 <em>Peer context:</em> ${esc(it.peer_context)}</div>` : ""}
      ${it.peer_answer ? `<div class="qa-peer-answer">🌐 <em>How peers answered it:</em> ${esc(it.peer_answer)}</div>` : ""}
      ${renderCitations(it.peer_citations)}
      <div class="qa-meta">
        ${it.asker ? `<span class="badge">${esc(it.asker)}</span>` : ""}
        <span class="badge dim">${esc(it.category)}</span>
        <span class="badge dim">conf: ${esc(it.confidence)}</span>
        <span class="badge dim">pages: ${(it.source_pages || []).join(", ") || "—"}</span>
      </div>
      ${it.caveat ? `<div class="qa-caveat">⚠️ ${esc(it.caveat)}</div>` : ""}
    </article>`).join("");
  return (head ? `<p class="muted" style="margin:0 0 0.9rem">${head}</p>` : "") + items;
}

export function renderLogLine(l) {
  const kv = Object.entries(l.data || {})
    .map(([k, v]) => `<span class="kv">${esc(k)}=${esc(JSON.stringify(v))}</span>`)
    .join("");
  const ts = (l.ts || "").slice(11, 19);
  return `<div class="logline">
    <span class="ts">${esc(ts)}</span>
    <span class="lvl ${esc(l.level)}">${esc(l.level)}</span>
    <span class="ev">${esc(l.event)}${kv}</span>
  </div>`;
}

export function renderChatMessage(m) {
  const src = (m.sources && m.sources.length)
    ? `<span class="src">source page(s): ${m.sources.join(", ")}</span>` : "";
  return `<div class="msg ${esc(m.role)}">
    <div class="bubble">${m.role === "assistant" ? mdLite(m.content) : esc(m.content)}</div>
    ${src}
  </div>`;
}
