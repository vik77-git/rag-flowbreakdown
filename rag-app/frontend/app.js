/* RAG using LangChain — vanilla front end. Session state lives in sessionStorage,
   so it survives reloads and disappears when the tab closes. */
const $ = (id) => document.getElementById(id);

const SESSION_ID = (() => {
  let id = sessionStorage.getItem("rag_session");
  if (!id) { id = "s_" + Math.random().toString(36).slice(2, 12); sessionStorage.setItem("rag_session", id); }
  return id;
})();

const STAGES = [
  ["analyze", "Query analysis"], ["vector", "Vector search"], ["bm25", "BM25 search"],
  ["fusion", "Hybrid fusion"], ["rerank", "Reranking (20B)"], ["context", "Context builder"],
  ["generate", "Generation"],
];

const state = {
  docs: [],
  selected: new Set(JSON.parse(sessionStorage.getItem("rag_selected") || "[]")),
  chunks: [],
  answer: "",
  busy: false,
  defaultPrompt: "",
  activeDocument: null,
  previewToken: 0,
  streamNode: null,
};

/* ---------- helpers ---------- */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (n) => (n ?? 0).toLocaleString();
const ms = (n) => (n == null ? "—" : `${Number(n).toFixed(0)} ms`);

function toast(msg, isError) {
  const t = $("toast");
  t.textContent = msg; t.className = "toast" + (isError ? " err" : "");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), 4500);
}

async function api(path, options = {}) {
  const res = await fetch(path, { ...options, headers: { "X-Session-Id": SESSION_ID, ...(options.headers || {}) } });
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try { message = (await res.json()).message || message; } catch (_) {}
    throw new Error(message);
  }
  return res.status === 204 ? null : res.json();
}

/* ---------- markdown (small, safe: escapes first) ---------- */
function highlight(code) {
  return esc(code)
    .replace(/(#.*|\/\/.*)/g, '<span class="tok-com">$1</span>')
    .replace(/(&quot;[^&]*?&quot;|&#39;[^&]*?&#39;)/g, '<span class="tok-str">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="tok-num">$1</span>')
    .replace(/\b(def|class|return|if|else|elif|for|while|import|from|const|let|var|function|async|await|try|except|SELECT|FROM|WHERE)\b/g,
      '<span class="tok-kw">$1</span>');
}

function inline(text) {
  return esc(text)
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<!\w)\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[(\d+)\]/g, '<span class="cite" data-cite="$1">[$1]</span>');
}

function renderMarkdown(src) {
  const lines = src.split("\n");
  let html = "", inCode = false, code = [], list = null, table = null;

  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const closeTable = () => {
    if (table) {
      const [head, ...rows] = table;
      html += "<table><thead><tr>" + head.map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") + "</tbody></table>";
      table = null;
    }
  };
  const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (line.trim().startsWith("```")) {
      if (inCode) { html += `<pre><code>${highlight(code.join("\n"))}</code></pre>`; code = []; inCode = false; }
      else { closeList(); closeTable(); inCode = true; }
      continue;
    }
    if (inCode) { code.push(raw); continue; }

    if (/^\s*\|.*\|\s*$/.test(line)) {
      const row = cells(line);
      if (row.every((c) => /^:?-{2,}:?$/.test(c))) continue;
      closeList();
      (table = table || []).push(row);
      continue;
    }
    closeTable();

    if (!line.trim()) { closeList(); continue; }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) { closeList(); html += `<h3>${inline(heading[2])}</h3>`; continue; }
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    if (ol || ul) {
      const want = ol ? "ol" : "ul";
      if (list !== want) { closeList(); html += `<${want}>`; list = want; }
      html += `<li>${inline((ol || ul)[1])}</li>`;
      continue;
    }
    closeList();
    html += `<p>${inline(line)}</p>`;
  }
  if (inCode && code.length) html += `<pre><code>${highlight(code.join("\n"))}</code></pre>`;
  closeList(); closeTable();
  return html;
}

/* ---------- documents ---------- */
function renderDocs() {
  const list = $("docList");
  if (!state.docs.length) { list.innerHTML = '<li class="sub">No documents yet. Upload a PDF, DOCX or TXT file.</li>'; }
  else {
    list.innerHTML = state.docs.map((d) => `
      <li>
        <input type="checkbox" data-select="${esc(d.document_id)}" ${state.selected.has(d.document_id) ? "checked" : ""} />
        <span class="name" data-open="${esc(d.document_id)}" title="${esc(d.name)}">
          ${esc(d.name)}${d.is_sample ? " <span class='sub'>(sample)</span>" : ""}
          <div class="sub">${d.pages} page(s) · ${d.chunks} chunks · ${num(d.tokens)} tokens</div>
        </span>
        <button class="del" data-del="${esc(d.document_id)}" title="Remove">✕</button>
      </li>`).join("");
  }
  $("metaDocs").textContent = `${state.docs.length} document${state.docs.length === 1 ? "" : "s"}`;
  renderSuggestions();
}

function renderSuggestions() {
  const selected = state.docs.find((d) => state.selected.has(d.document_id));
  const suggestions = selected?.suggestions || [];
  $("examples").innerHTML = suggestions.length
    ? suggestions.map((q) => `<button type="button" data-example="${esc(q)}">${esc(q)}</button>`).join("")
    : '<span class="hint">Select a document to see suggested questions.</span>';
}

async function loadDocs() {
  state.docs = await api("/api/documents");
  for (const id of [...state.selected]) if (!state.docs.some((d) => d.document_id === id)) state.selected.delete(id);
  if (!state.selected.size && state.docs.length === 1) state.selected.add(state.docs[0].document_id);
  if (state.selected.size > 1) state.selected = new Set([[...state.selected][0]]);
  sessionStorage.setItem("rag_selected", JSON.stringify([...state.selected]));
  renderDocs();
  const selectedId = [...state.selected][0];
  if (selectedId) openDocument(selectedId);
}

const UPLOAD_STAGES = ["Uploading…", "Parsing…", "Analyzing structure…", "Chunking…", "Generating embeddings…", "Indexing…"];

async function uploadFile(file) {
  const status = $("uploadStatus");
  status.classList.remove("hidden");
  let i = 0;
  status.textContent = UPLOAD_STAGES[0];
  const ticker = setInterval(() => { i = Math.min(i + 1, UPLOAD_STAGES.length - 1); status.textContent = UPLOAD_STAGES[i]; }, 900);
  const body = new FormData(); body.append("file", file);
  try {
    const doc = await api("/api/documents/upload", { method: "POST", body });
    status.textContent = `Ready — ${doc.name} (${doc.chunks} chunks)`;
    state.selected = new Set([doc.document_id]);
    sessionStorage.setItem("rag_selected", JSON.stringify([...state.selected]));
    await loadDocs();
    openDocument(doc.document_id);
  } catch (err) {
    status.textContent = "";
    status.classList.add("hidden");
    toast(err.message, true);
  } finally {
    clearInterval(ticker);
    setTimeout(() => status.classList.add("hidden"), 4000);
  }
}

async function openDocument(id) {
  const requestToken = ++state.previewToken;
  const panel = document.querySelector(".preview");
  panel.classList.add("loading");
  $("previewBody").textContent = "Loading page…";
  try {
    const doc = await api(`/api/documents/${id}`);
    if (requestToken !== state.previewToken) return;
    state.activeDocument = doc;
    $("previewTitle").textContent = doc.name;
    $("previewMeta").textContent = `${doc.pages} page(s) · ${doc.chunks} chunks · ${doc.extension}`;
    const pageSel = $("pageSelect"), secSel = $("sectionSelect");
    pageSel.innerHTML = Array.from({ length: doc.pages }, (_, i) => `<option value="${i + 1}">Page ${i + 1}</option>`).join("");
    pageSel.classList.toggle("hidden", doc.pages < 2);
    secSel.innerHTML = '<option value="">Jump to section…</option>' + (doc.sections || []).map((s) => `<option value="${s.page}">${esc(s.title)}</option>`).join("");
    secSel.classList.toggle("hidden", !(doc.sections || []).length);
    const show = async (page) => {
      const pageToken = ++state.previewToken;
      panel.classList.add("loading");
      $("previewBody").textContent = "Loading page…";
      try {
        const hit = await api(`/api/documents/${id}/pages/${page}`);
        if (pageToken !== state.previewToken) return;
        $("previewBody").textContent = hit.text || "No extracted text.";
      } catch (err) { if (pageToken === state.previewToken) toast(err.message, true); }
      finally { if (pageToken === state.previewToken) panel.classList.remove("loading"); }
    };
    pageSel.onchange = () => show(pageSel.value);
    secSel.onchange = () => {
      if (!secSel.value) return;
      pageSel.value = secSel.value;
      show(secSel.value);
    };
    show(1);
  } catch (err) { panel.classList.remove("loading"); toast(err.message, true); }
}

/* ---------- workflow ---------- */
function resetStages() {
  $("stages").innerHTML = STAGES.map(([k, label]) =>
    `<li id="stage-${k}"><div class="t">${label}</div><div class="d">waiting</div></li>`).join("");
  $("stageNote").textContent = "";
}
function setStage(key, cls, detail) {
  const el = $(`stage-${key}`); if (!el) return;
  el.className = cls;
  if (detail != null) el.querySelector(".d").textContent = detail;
}

/* ---------- chunks ---------- */
function renderChunks() {
  const box = $("chunkList");
  if (!state.chunks.length) { box.innerHTML = '<p class="hint">No chunks retrieved.</p>'; $("chunkCount").textContent = ""; return; }
  const best = Math.max(...state.chunks.map((c) => c.rerank_score || 0));
  box.innerHTML = state.chunks.map((c, i) => `
    <details class="chunk ${c.selected ? "selected" : ""}" id="chunk-${esc(c.chunk_id)}" ${i < 2 ? "open" : ""}>
      <summary>
        <div class="title">${esc(c.document)} · Page ${c.page}</div>
        <div class="meta">§ ${esc(c.section)} · ${esc(c.chunk_id)} · ${num(c.token_count)} tokens ${c.selected ? "· in context" : ""}</div>
        <div class="meta">
          <span class="score ${(c.rerank_score || 0) >= best && best > 0 ? "top" : ""}">rerank ${c.rerank_score ?? "—"}</span>
          <span class="score">vector ${c.vector_score ?? "—"}${c.vector_rank ? ` (#${c.vector_rank})` : ""}</span>
          <span class="score">bm25 ${c.bm25_score ?? "—"}${c.bm25_rank ? ` (#${c.bm25_rank})` : ""}</span>
          <span class="score">fusion ${c.fusion_score ?? "—"}</span>
        </div>
      </summary>
      <pre>${esc(c.text)}</pre>
    </details>`).join("");
  $("chunkCount").textContent = `${state.chunks.filter((c) => c.selected).length} of ${state.chunks.length} used as context`;
}

/* ---------- usage ---------- */
function rows(target, data) {
  $(target).querySelector("tbody").innerHTML = data.map(([k, v, est]) =>
    `<tr><td>${esc(k)}${est ? ' <span class="est">(estimate)</span>' : ""}</td><td>${v}</td></tr>`).join("");
}
function renderUsage(u, l) {
  const est = !u.generator_usage_reported;
  rows("usageTable", [
    ["Query tokens", num(u.query_tokens), true],
    ["Retrieval context", num(u.context_tokens), true],
    [`Reranker input (${u.reranker_mode})`, num(u.reranker_input_tokens), u.reranker_mode !== "llm"],
    ["Reranker output", num(u.reranker_output_tokens), u.reranker_mode !== "llm"],
    ["Generator input", num(u.generator_input_tokens), est],
    ["Generator output", num(u.generator_output_tokens), est],
    ["Total", num(u.total_tokens), false],
  ]);
  rows("latencyTable", [
    ["Analysis", ms(l.analyze_ms)], ["Retrieval", ms(l.retrieval_ms)], ["Reranking", ms(l.rerank_ms)],
    ["Context", ms(l.context_ms)], ["Generation", ms(l.llm_ms)], ["Total", ms(l.total_ms)],
  ]);
}
function renderSession(s) {
  rows("sessionTable", [
    ["Queries", num(s.queries)], ["Total tokens", num(s.total_tokens)],
    ["Avg / query", num(Math.round(s.avg_tokens_per_query ?? (s.queries ? s.total_tokens / s.queries : 0)))],
    ["Input tokens", num(s.total_input_tokens)], ["Output tokens", num(s.total_output_tokens)],
  ]);
  $("metaSession").textContent = `session ${num(s.total_tokens)} tokens · ${num(s.queries)} queries`;
}

/* ---------- ask ---------- */
async function ask() {
  const query = $("queryInput").value.trim();
  if (query.length < 2) return toast("Type a question first.", true);
  if (state.busy) return;
  state.busy = true; $("askBtn").disabled = true; $("askBtn").textContent = "…";
  $("queryInput").value = ""; $("queryInput").focus();

  state.answer = ""; state.chunks = [];
  $("userMessage").textContent = query;
  $("userMessage").classList.remove("hidden");
  resetStages(); renderChunks();
  $("citations").innerHTML = "";
  $("answerBody").innerHTML = '<p class="hint">Thinking through the selected document…<span class="caret"></span></p>';
  setStage("analyze", "active", "analyzing");

  const payload = {
    query,
    document_ids: [...state.selected],
    model: $("modelSelect").value,
    system_prompt: $("systemPrompt").value.trim() || null,
  };

  try {
    const res = await fetch("/api/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-Id": SESSION_ID },
      body: JSON.stringify(payload),
    });
    if (!res.ok || !res.body) throw new Error(`Stream failed (${res.status})`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const evLine = part.split("\n").find((l) => l.startsWith("event:"));
        const dataLine = part.split("\n").find((l) => l.startsWith("data:"));
        if (!evLine || !dataLine) continue;
        try { handleEvent(evLine.slice(6).trim(), JSON.parse(dataLine.slice(5).trim())); }
        catch (_) { /* ignore malformed frame */ }
      }
    }
  } catch (err) {
    toast(err.message, true);
    setStage("generate", "failed", "failed");
  } finally {
    state.busy = false; $("askBtn").disabled = false; $("askBtn").textContent = "↑";
  }
}

function handleEvent(event, data) {
  switch (event) {
    case "query_analyzed":
      setStage("analyze", "done", `${data.analysis.intent} · depth ${data.analysis.retrieval_depth} · ${ms(data.duration_ms)}`);
      setStage("vector", "active", "searching");
      $("answerModel").textContent = data.model;
      $("metaModel").textContent = data.model;
      $("stageNote").textContent = data.analysis.multi_hop ? "multi-hop question detected" : "";
      break;
    case "vector_search_complete":
      setStage("vector", "done", `${data.count} candidates`); setStage("bm25", "active", "searching"); break;
    case "bm25_search_complete":
      setStage("bm25", "done", `${data.count} candidates`); setStage("fusion", "active", "merging"); break;
    case "hybrid_search_complete":
      setStage("fusion", "done", `${data.count} unique · ${ms(data.duration_ms)}`); setStage("rerank", "active", "scoring"); break;
    case "reranking_complete":
      setStage("rerank", "done", `${data.input} → ${data.output} · ${data.mode} · ${ms(data.duration_ms)}`);
      setStage("context", "active", "building"); break;
    case "context_built":
      setStage("context", "done", `${data.count} chunks · ${num(data.context_tokens)} tokens`);
      state.chunks = data.chunks || []; renderChunks();
      setStage("generate", "active", "streaming"); break;
    case "token":
      state.answer += data.text;
      if (!state.streamNode) {
        $("answerBody").textContent = "";
        state.streamNode = document.createElement("div");
        state.streamNode.className = "stream-text";
        $("answerBody").append(state.streamNode);
      }
      state.streamNode.textContent = state.answer;
      $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
      break;
    case "generation_complete":
      setStage("generate", "done", ms(data.duration_ms)); break;
    case "usage":
      renderUsage(data.usage, data.latency); renderSession(data.session); break;
    case "query_complete":
      state.answer = data.answer;
      state.streamNode = null;
      $("answerBody").innerHTML = renderMarkdown(state.answer);
      state.chunks = data.chunks || []; renderChunks();
      renderCitations(data.citations);
      sessionStorage.setItem("rag_last", JSON.stringify(data));
      break;
    case "error":
      toast(data.message || "The query failed.", true);
      setStage("generate", "failed", "failed");
      $("answerBody").innerHTML = `<p class="hint">${esc(data.message || "The query failed.")}</p>`;
      break;
  }
}

function renderCitations(citations) {
  $("citations").innerHTML = (citations || []).map((c) =>
    `<button data-cite="${c.index}">[${c.index}] ${esc(c.document)} · p${c.page} · ${esc(c.section)}</button>`).join("");
}

function jumpToCitation(index) {
  const last = JSON.parse(sessionStorage.getItem("rag_last") || "null");
  const cite = last?.citations?.find((c) => String(c.index) === String(index));
  if (!cite) return;
  const el = $(`chunk-${cite.chunk_id}`);
  if (el) {
    $("tabEvidence").checked = true;
    el.open = true;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

/* ---------- init ---------- */
async function init() {
  resetStages();
  renderSuggestions();
  rows("usageTable", [["Query tokens", "—"], ["Retrieval context", "—"], ["Generator input", "—"], ["Generator output", "—"], ["Total", "—"]]);
  rows("latencyTable", [["Analysis", "—"], ["Retrieval", "—"], ["Reranking", "—"], ["Generation", "—"], ["Total", "—"]]);

  try {
    const health = await api("/api/health");
    state.defaultPrompt = health.default_system_prompt || "";
    $("systemPrompt").placeholder = state.defaultPrompt;
    const sel = $("modelSelect");
    sel.innerHTML = '<option value="auto">Auto (20B, 120B for complex)</option>' +
      health.models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join("");
    $("modelHint").textContent = `${health.models[0]?.split("/").pop()}: fast, standard questions · ${health.models[1]?.split("/").pop()}: more reasoning capacity for complex questions.`;
    const badge = $("metaHealth");
    badge.textContent = health.groq_configured ? "Groq ready" : "Groq key missing";
    badge.className = "badge " + (health.groq_configured ? "ok" : "bad");
    if (!health.neural_embeddings) toast("Embedding model unavailable — using fallback embeddings.", true);
  } catch (err) { toast(err.message, true); }

  await loadDocs();
  try { renderSession(await api("/api/session/usage")); } catch (_) {}

  const last = JSON.parse(sessionStorage.getItem("rag_last") || "null");
  if (last) {
    state.answer = last.answer; state.chunks = last.chunks || [];
    $("answerBody").innerHTML = renderMarkdown(last.answer);
    $("answerModel").textContent = last.model; $("metaModel").textContent = last.model;
    renderChunks(); renderCitations(last.citations); renderUsage(last.usage, last.latency);
  }

  $("askBtn").onclick = ask;
  $("fileInput").onchange = (e) => { const f = e.target.files[0]; if (f) uploadFile(f); e.target.value = ""; };
  $("promptToggle").onclick = () => $("promptBox").classList.toggle("hidden");
  $("previewToggle").onclick = () => {
    document.querySelector(".preview").classList.toggle("collapsed");
    $("previewToggle").textContent = document.querySelector(".preview").classList.contains("collapsed") ? "⌄" : "⌃";
  };
  $("queryInput").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } };

  document.querySelector(".sidebar-nav").addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-item");
    if (!btn) return;
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n === btn));
    const target = btn.dataset.target;
    if (target === "evidence") { $("tabEvidence").checked = true; document.querySelector(".context-rail")?.scrollIntoView({ behavior: "smooth", block: "start" }); }
    else if (target === "documents") document.querySelector(".docs")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    else { document.querySelector(".workflow")?.scrollIntoView({ behavior: "smooth", block: "start" }); $("queryInput").focus(); }
  });

  document.addEventListener("click", async (e) => {
    const t = e.target.closest("[data-example],[data-open],[data-del],[data-cite]");
    if (!t) return;
    if (t.dataset.example) { $("queryInput").value = t.dataset.example; ask(); }
    else if (t.dataset.open) openDocument(t.dataset.open);
    else if (t.dataset.cite) jumpToCitation(t.dataset.cite);
    else if (t.dataset.del) {
      if (!confirm("Remove this document from the knowledge base?")) return;
      try { await api(`/api/documents/${t.dataset.del}`, { method: "DELETE" }); state.selected.delete(t.dataset.del); await loadDocs(); toast("Document removed."); }
      catch (err) { toast(err.message, true); }
    }
  });

  document.addEventListener("change", (e) => {
    const id = e.target.dataset?.select;
    if (!id) return;
    state.selected.clear();
    if (e.target.checked) state.selected.add(id);
    sessionStorage.setItem("rag_selected", JSON.stringify([...state.selected]));
    renderDocs();
    if (e.target.checked) openDocument(id);
  });
}

init();
