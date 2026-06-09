// Thin typed-ish wrappers over the backend REST API.

async function req(url, opts = {}) {
  const res = await fetch(url, opts);
  const ct = res.headers.get("content-type") || "";
  const body = ct.includes("application/json")
    ? await res.json().catch(() => ({}))
    : await res.text();
  if (!res.ok) {
    const detail = (body && body.detail) || (typeof body === "string" ? body : "") ||
      `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return body;
}

export const api = {
  health: () => req("/health"),
  config: () => req("/config"),
  samples: () => req("/samples"),

  listRuns: (limit = 50, offset = 0) =>
    req(`/runs?limit=${limit}&offset=${offset}`),
  getRun: (id) => req(`/runs/${id}`),
  runLogs: (id, after = 0) => req(`/runs/${id}/logs?after=${after}`),
  deleteRun: (id) => req(`/runs/${id}`, { method: "DELETE" }),

  startRun({ source, name, file, mode, numQuestions, peers }) {
    const fd = new FormData();
    fd.append("source", source);
    fd.append("mode", mode);
    if (source === "sample") fd.append("name", name);
    else fd.append("file", file);
    if (numQuestions != null) fd.append("num_questions", String(numQuestions));
    if (peers != null) fd.append("peers", String(peers));
    return req("/runs", { method: "POST", body: fd });
  },

  testAgent: () => req("/agent/test", { method: "POST" }),

  chat: (id) => req(`/runs/${id}/chat`),
  ask: (id, question) =>
    req(`/runs/${id}/ask`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    }),

  getSettings: () => req("/settings"),
  updateSettings: (values) =>
    req("/settings", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ values }),
    }),
};
