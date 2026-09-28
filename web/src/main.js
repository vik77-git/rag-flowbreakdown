/* UI layer for the browser-only RAG workspace. */
import "./styles.css";
import samplePolicy from "./assets/security_policy.txt?raw";
import { parseFile, parseText } from "./rag/parse.js";
import { chunkDocument, suggestionsFor, approxTokens } from "./rag/chunk.js";
import { HybridIndex } from "./rag/retrieve.js";
import * as groq from "./rag/groq.js";
import { run, DEFAULT_SYSTEM_PROMPT } from "./rag/pipeline.js";

const $ = (id) => document.getElementById(id);
const MEMORY_KEY = "rag_chat_memory";

const state = {
  docs: [],
  activeId: null,
  index: new HybridIndex(),
  busy: false,
  memory: [],
  session: { queries: 0, tokens: 0, input: 0, output: 0 },
  previewToken: 0,
};

const STAGES = [
  ["analyze", "Query analysis"],
  ["vector", "Vector search"],
  ["bm25", "BM25 search"],
  ["fusion", "RRF fusion"],
  ["rerank", "Reranking"],
  ["context", "Context build"],
  ["generate", "Answer generation"],
];

/* ---------- small helpers ---------- */
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ms = (n) => `${Math.round(n || 0)} ms`;

function toast(message, bad = false) {
  const el = $("toast");
  el.textContent = message;
  el.style.background = bad ? "var(--danger)" : "var(--foreground)";
  el.classList.remove("hidden");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.classList.add("hidden"), 3600);
}

function rows(tableId, data) {
  $(tableId).querySelector("tbody").innerHTML = data
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`)
    .join("");
}

function resetStages(note = "Idle — ask a question to run the pipeline.") {
  $("stages").innerHTML = STAGES.map(
    (s) => `<li class="stage" data-stage="${s[0]}"><strong>${s[1]}</strong><span>waiting</span></li>`
  ).join("");
  $("stageNote").textContent = note;
}

function setStage(key, text, status = "done") {
  const el = document.querySelector(`[data-stage="${key}"]`);
  if (!el) return;
  el.className = `stage ${status}`;
  el.querySelector("span").textContent = text;
}

/* ---------- markdown ---------- */
function renderMarkdown(src) {
  let text = esc(src);
  const blocks = [];
  text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  text = text.replace(/\[(\d+)\]/g, '<button class="cite" data-cite="$1">$1</button>');
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  const lines = text.split("\n");
  const out = [];
  let list = null;
  let table = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const closeTable = () => {
    if (table) {
      const [head, ...body] = table;
      out.push(
        `<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
          body.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") +
          `</tbody></table>`
      );
      table = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (/^\u0000\d+\u0000$/.test(line)) { closeList(); closeTable(); out.push(line); continue; }
    if (!line) { closeList(); closeTable(); continue; }
    if (/^\|.*\|$/.test(line)) {
      const cells = line.slice(1, -1).split("|").map((c) => c.trim());
      if (cells.every((c) => /^:?-{2,}:?$/.test(c))) continue;
      closeList();
      (table ||= []).push(cells);
      continue;
    }
    closeTable();
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) { closeList(); out.push(`<h${heading[1].length + 2}>${heading[2]}</h${heading[1].length + 2}>`); continue; }
    const ordered = line.match(/^\d+[.)]\s+(.*)$/);
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    if (ordered || bullet) {
      const want = ordered ? "ol" : "ul";
      if (list !== want) { closeList(); out.push(`<${want}>`); list = want; }
      out.push(`<li>${(ordered || bullet)[1]}</li>`);
      continue;
    }
    closeList();
    out.push(`<p>${line}</p>`);
  }
  closeList(); closeTable();
  return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

/* ---------- chat ---------- */
function intro() {
  const el = document.createElement("div");
  el.className = "assistant-intro";
  el.innerHTML =
    `<div class="assistant-mark">R</div><div><strong>Ready to investigate.</strong>` +
    `<p class="hint">Say hello, ask for a summary, or ask anything supported by the selected document.</p></div>`;
  return el;
}

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text;
  $("chatScroll").append(el);
  scrollChat();
}

function addAssistantMessage() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `<div class="body"></div><div class="citations"></div>`;
  $("chatScroll").append(el);
  scrollChat();
  return el;
}

const scrollChat = () => { $("chatScroll").scrollTop = $("chatScroll").scrollHeight; };

function clearChat(silent = false) {
  state.memory = [];
  localStorage.removeItem(MEMORY_KEY);
  $("chatScroll").replaceChildren(intro());
  $("chunkList").innerHTML = `<p class="hint">Run a query to inspect ranked chunks.</p>`;
  $("chunkCount").textContent = "";
  resetStages();
  placeholderMetrics();
  state.session = { queries: 0, tokens: 0, input: 0, output: 0 };
  renderSession();
  if (!silent) toast("Chat and memory cleared.");
}

function saveMemory() {
  try { localStorage.setItem(MEMORY_KEY, JSON.stringify(state.memory.slice(-12))); } catch (_) {}
}

/* ---------- documents ---------- */
function renderDocs() {
  $("docList").innerHTML = state.docs
    .map(
      (d) => `<li class="docitem ${d.id === state.activeId ? "active" : ""}" data-id="${d.id}">
        <div class="meta"><strong>${esc(d.name)}</strong><span>${d.pages.length} page(s) · ${d.chunks.length} chunks</span></div>
        <button data-remove="${d.id}" title="Remove">✕</button></li>`
    )
    .join("");
  $("metaDocs").textContent = `${state.docs.length} document${state.docs.length === 1 ? "" : "s"}`;
}

function activeDoc() {
  return state.docs.find((d) => d.id === state.activeId) || null;
}

function renderSuggestions() {
  const doc = activeDoc();
  const list = doc ? doc.suggestions : [];
  $("examples").innerHTML = list.map((q) => `<button type="button">${esc(q)}</button>`).join("");
}

async function addDocument(parsed) {
  const id = `doc_${Date.now().toString(36)}`;
  const { chunks, sections } = chunkDocument(parsed, id);
  const doc = {
    id,
    name: parsed.name,
    extension: parsed.extension,
    size_bytes: parsed.size_bytes,
    pages: parsed.pages,
    chunks,
    sections,
    suggestions: suggestionsFor(parsed.name, sections),
    tokens: chunks.reduce((a, c) => a + c.metadata.token_count, 0),
  };
  state.docs = [doc];
  state.activeId = id;
  state.index.build(chunks);
  renderDocs();
  renderSuggestions();
  openPreview();
  return doc;
}

function openPreview() {
  const doc = activeDoc();
  const token = ++state.previewToken;
  if (!doc) {
    $("previewTitle").textContent = "Document preview";
    $("previewMeta").textContent = "";
    $("previewBody").textContent = "Select a document to preview its extracted text.";
    $("pageSelect").classList.add("hidden");
    $("sectionSelect").classList.add("hidden");
    return;
  }
  $("previewTitle").textContent = doc.name;
  $("previewMeta").textContent = `${doc.pages.length} page(s) · ${doc.chunks.length} chunks · ~${doc.tokens} tokens`;

  const pageSelect = $("pageSelect");
  pageSelect.innerHTML = doc.pages.map((p) => `<option value="${p.page}">Page ${p.page}</option>`).join("");
  pageSelect.classList.remove("hidden");

  const sectionSelect = $("sectionSelect");
  sectionSelect.innerHTML =
    `<option value="">All sections</option>` +
    doc.sections.map((s, i) => `<option value="${i}">${esc(s.title)}</option>`).join("");
  sectionSelect.classList.toggle("hidden", !doc.sections.length);

  showPage(doc.pages[0].page, token);
}

function showPage(pageNumber, token = state.previewToken) {
  const doc = activeDoc();
  if (!doc || token !== state.previewToken) return;
  const page = doc.pages.find((p) => p.page === Number(pageNumber));
  $("previewBody").textContent = page ? page.text : "No text on this page.";
  $("pageSelect").value = String(pageNumber);
  $("previewBody").scrollTop = 0;
}

/* ---------- evidence ---------- */
function renderChunks(chunks) {
  $("chunkCount").textContent = `${chunks.filter((c) => c.selected).length} of ${chunks.length} used`;
  $("chunkList").innerHTML = chunks
    .map(
      (c, i) => `<details class="chunk ${c.selected ? "selected" : ""}" id="chunk-${i + 1}">
        <summary>${c.selected ? `[${chunks.filter((x, j) => x.selected && j <= i).length}] ` : ""}p${c.page} · ${esc(c.section)}</summary>
        <div class="scores">
          <span>vector ${c.vector_score ?? "—"}</span><span>bm25 ${c.bm25_score ?? "—"}</span>
          <span>rrf ${c.fusion_score ?? "—"}</span><span>rerank ${c.rerank_score ?? "—"}</span>
        </div>
        <p>${esc(c.text.slice(0, 900))}${c.text.length > 900 ? "…" : ""}</p>
      </details>`
    )
    .join("");
}

function renderCitations(container, citations) {
  container.innerHTML = citations
    .map((c) => `<button type="button" data-cite="${c.index}">[${c.index}] ${esc(c.document)} · p${c.page}</button>`)
    .join("");
}

function jumpToCitation(index) {
  $("tabEvidence").checked = true;
  const selected = [...document.querySelectorAll(".chunk.selected")];
  const el = selected[Number(index) - 1];
  if (el) { el.open = true; el.scrollIntoView({ behavior: "smooth", block: "center" }); }
}

/* ---------- metrics ---------- */
function placeholderMetrics() {
  rows("usageTable", [["Query tokens", "—"], ["Retrieval context", "—"], ["Generator input", "—"], ["Generator output", "—"], ["Total", "—"]]);
  rows("latencyTable", [["Analysis", "—"], ["Retrieval", "—"], ["Reranking", "—"], ["Generation", "—"], ["Total", "—"]]);
}

function renderSession() {
  rows("sessionTable", [
    ["Queries", state.session.queries],
    ["Input tokens", state.session.input],
    ["Output tokens", state.session.output],
    ["Total tokens", state.session.tokens],
  ]);
  $("metaSession").textContent = `session ${state.session.tokens} tokens`;
}

/* ---------- ask ---------- */
async function ask(question) {
  const query = (question ?? $("queryInput").value).trim();
  if (!query || state.busy) return;

  $("queryInput").value = "";
  $("queryInput").focus();
  state.busy = true;
  $("askBtn").disabled = true;
  $("askBtn").textContent = "…";

  const intro = $("chatScroll").querySelector(".assistant-intro");
  if (intro) intro.remove();

  addUserMessage(query);
  const node = addAssistantMessage();
  const body = node.querySelector(".body");
  body.innerHTML = `<span class="typing">Thinking…</span>`;

  resetStages("Running…");
  let answer = "";
  let streaming = false;

  try {
    for await (const { event, data } of run(query, {
      index: state.index,
      document: activeDoc() ? { name: activeDoc().name, pages: activeDoc().pages.length, chunks: activeDoc().chunks.length } : null,
      model: $("modelSelect").value,
      systemPrompt: $("systemPrompt").value,
      memory: state.memory,
    })) {
      if (event === "query_analyzed") {
        setStage("analyze", `${data.analysis.intent} · depth ${data.analysis.retrieval_depth}`);
        $("answerModel").textContent = data.model;
        $("metaModel").textContent = `model ${data.model}`;
      } else if (event === "vector_search_complete") setStage("vector", `${data.count} candidates`);
      else if (event === "bm25_search_complete") setStage("bm25", `${data.count} candidates`);
      else if (event === "hybrid_search_complete") setStage("fusion", `${data.count} fused · ${ms(data.duration_ms)}`);
      else if (event === "reranking_complete") setStage("rerank", `${data.output} kept · ${data.mode}`);
      else if (event === "context_built") { setStage("context", `${data.count} chunks · ${data.context_tokens} tokens`); renderChunks(data.chunks); }
      else if (event === "generation_started") setStage("generate", "streaming…", "active");
      else if (event === "token") {
        if (!streaming) { body.textContent = ""; streaming = true; }
        answer += data.text;
        body.textContent = answer;
        scrollChat();
      } else if (event === "generation_complete") setStage("generate", ms(data.duration_ms));
      else if (event === "error") {
        body.innerHTML = `<p class="msg error">${esc(data.message)}</p>`;
        const pending = document.querySelector(".stage.active");
        if (pending) { pending.className = "stage failed"; pending.querySelector("span").textContent = "stopped"; }
        $("stageNote").textContent = "Stopped.";
        toast(data.message, true);
        return;
      } else if (event === "query_complete") {
        setStage("generate", ms(data.latency.llm_ms));
        body.innerHTML = renderMarkdown(data.answer);
        renderCitations(node.querySelector(".citations"), data.citations);
        if (data.chunks?.length) renderChunks(data.chunks);
        rows("usageTable", [
          ["Query tokens", data.usage.query_tokens],
          ["Retrieval context", data.usage.context_tokens],
          ["Generator input", data.usage.generator_input_tokens],
          ["Generator output", data.usage.generator_output_tokens],
          ["Total", data.usage.total_tokens],
        ]);
        rows("latencyTable", [
          ["Analysis", ms(data.latency.analyze_ms)],
          ["Retrieval", ms(data.latency.retrieval_ms)],
          ["Reranking", ms(data.latency.rerank_ms)],
          ["Generation", ms(data.latency.llm_ms)],
          ["Total", ms(data.latency.total_ms)],
        ]);
        state.session.queries += 1;
        state.session.input += data.usage.generator_input_tokens + data.usage.reranker_input_tokens;
        state.session.output += data.usage.generator_output_tokens + data.usage.reranker_output_tokens;
        state.session.tokens += data.usage.total_tokens;
        renderSession();
        state.memory.push({ role: "user", content: query }, { role: "assistant", content: data.answer.slice(0, 1500) });
        saveMemory();
        $("stageNote").textContent = `Completed in ${ms(data.latency.total_ms)}`;
      }
    }
  } catch (err) {
    body.innerHTML = `<p class="msg error">${esc(err.message || "Something went wrong.")}</p>`;
    toast(err.message || "Something went wrong.", true);
  } finally {
    state.busy = false;
    $("askBtn").disabled = false;
    $("askBtn").textContent = "↑";
    scrollChat();
  }
}

/* ---------- init ---------- */
async function init() {
  resetStages();
  placeholderMetrics();
  renderSession();
  $("systemPrompt").value = DEFAULT_SYSTEM_PROMPT;
  $("chatScroll").replaceChildren(intro());

  for (const model of groq.MODELS) {
    const opt = document.createElement("option");
    opt.value = model;
    opt.textContent = model;
    $("modelSelect").append(opt);
  }

  const ok = groq.hasKey();
  $("metaHealth").textContent = ok ? "Groq connected" : "No API key";
  document.querySelector(".status-dot").classList.toggle("bad", !ok);
  $("modelHint").textContent = ok
    ? "Auto picks the 120B model for summaries and multi-hop questions."
    : "Set VITE_GROQ_API_KEY in your environment (or on Vercel) to enable answers.";

  try { state.memory = JSON.parse(localStorage.getItem(MEMORY_KEY) || "[]"); } catch (_) { state.memory = []; }

  await addDocument(await parseText("security_policy.txt", samplePolicy));

  /* events */
  $("askBtn").addEventListener("click", () => ask());
  $("queryInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });
  $("examples").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (btn) ask(btn.textContent);
  });
  $("clearChat").addEventListener("click", () => clearChat());
  $("promptToggle").addEventListener("click", () => $("promptBox").classList.toggle("hidden"));
  $("previewToggle").addEventListener("click", () => {
    const hidden = $("previewPanel").classList.toggle("hidden");
    $("previewToggle").textContent = hidden ? "⌄" : "⌃";
  });
  $("pageSelect").addEventListener("change", (e) => showPage(e.target.value));
  $("sectionSelect").addEventListener("change", (e) => {
    const doc = activeDoc();
    if (!doc || e.target.value === "") return;
    showPage(doc.sections[Number(e.target.value)].page);
  });
  $("docList").addEventListener("click", (e) => {
    const remove = e.target.closest("[data-remove]");
    if (remove) {
      state.docs = state.docs.filter((d) => d.id !== remove.dataset.remove);
      state.activeId = state.docs[0]?.id || null;
      state.index.build(state.docs[0]?.chunks || []);
      renderDocs(); renderSuggestions(); openPreview();
      return;
    }
    const item = e.target.closest(".docitem");
    if (item) { state.activeId = item.dataset.id; state.index.build(activeDoc().chunks); renderDocs(); renderSuggestions(); openPreview(); }
  });
  $("fileInput").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const status = $("uploadStatus");
    status.classList.remove("hidden");
    status.textContent = `Parsing ${file.name}…`;
    try {
      const doc = await addDocument(await parseFile(file));
      status.textContent = `${doc.name} indexed (${doc.chunks.length} chunks).`;
      clearChat(true);
      toast(`${doc.name} is ready.`);
    } catch (err) {
      status.textContent = err.message;
      toast(err.message, true);
    }
    e.target.value = "";
  });
  document.addEventListener("click", (e) => {
    const cite = e.target.closest("[data-cite]");
    if (cite) jumpToCitation(cite.dataset.cite);
  });
  const sidebar = document.querySelector(".sidebar");
  sidebar.querySelector(".sidebar-nav").addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-item");
    if (!btn) return;
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n === btn));
    const target = btn.dataset.target;
    if (target === "evidence") $("tabEvidence").checked = true;
    if (target === "documents") $("fileInput").closest(".docs").scrollIntoView({ behavior: "smooth", block: "nearest" });
    $(target === "documents" ? "workflow" : target)?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (target === "workspace" || target === "workflow") $("queryInput").focus();
    sidebar.classList.add("collapsed");
    sidebar.classList.remove("revealed");
  });
  sidebar.addEventListener("mouseenter", () => { if (sidebar.classList.contains("collapsed")) sidebar.classList.add("revealed"); });
  sidebar.addEventListener("mouseleave", () => sidebar.classList.remove("revealed"));
  sidebar.querySelector(".sidebar-brand").addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
    sidebar.classList.remove("revealed");
  });

  $("queryInput").focus();
}

init();
