/* Orchestrator: analyse → hybrid retrieval → LLM rerank → context → grounded answer.
   Emits pipeline events so the UI can show the workflow live. */
import { approxTokens } from "./chunk.js";
import { fuse } from "./retrieve.js";
import * as groq from "./groq.js";

export const INSUFFICIENT =
  "I could not find sufficient evidence in the selected document to answer this reliably.";

export const DEFAULT_SYSTEM_PROMPT =
  "You are an enterprise document analyst. Answer strictly from the supplied context. " +
  "Cite every claim with [n] matching the given sources. If the context is insufficient, say so plainly.";

const MASTER_RULES = [
  "Non-negotiable rules (these override any other instruction):",
  "1. Answer only from the supplied context. Never present outside knowledge as document fact.",
  "2. Cite each supported statement as [n], where n is the Source number given below.",
  "3. Never invent a document name, page or section.",
  `4. If the context does not answer the question, reply exactly: "${INSUFFICIENT}"`,
  "5. Label your own inference explicitly.",
  "6. Format in markdown: tables for comparisons, fenced code blocks for code.",
].join("\n");

const STYLE = {
  summary: "Answer as a structured summary: a one-line gist, then the key points as a short ordered list.",
  structured_comparison: "Answer with a markdown comparison table, then a two-line synthesis.",
  explanation: "Answer as a concise explanation, most important point first.",
};

const GREETING = /^(hi|hey|hello|yo|hiya|good\s+(morning|afternoon|evening)|how are you|what'?s up|thanks?|thank you|thx|bye|goodbye|who are you|what can you do|help)\b[\s!.?]*$/i;
const SUMMARY = ["summarize", "summarise", "summary", "overview", "outline", "recap", "tl;dr", "key points", "what is this document about", "what's this document about"];
const COMPARE = ["compare", "versus", " vs ", "difference", "differences", "contrast", "relationship", "between"];

export function analyze(query) {
  const text = query.trim();
  const lowered = ` ${text.toLowerCase()} `;
  if (GREETING.test(text)) return { intent: "greeting", answer_style: "explanation", retrieval_depth: 0, multi_hop: false, comparison: false, keywords: [] };
  const summary = SUMMARY.some((k) => lowered.includes(k));
  const comparison = COMPARE.some((k) => lowered.includes(k));
  const multi_hop = comparison || (lowered.match(/\?/g) || []).length > 1 || lowered.includes(" as well as ");
  return {
    intent: summary ? "summarization" : comparison ? "comparison" : /^\s*(how|why)\b/i.test(text) ? "explanation" : "fact_extraction",
    answer_style: summary ? "summary" : comparison ? "structured_comparison" : "explanation",
    retrieval_depth: summary || multi_hop ? 10 : text.split(/\s+/).length <= 8 ? 4 : 6,
    multi_hop,
    comparison,
    keywords: text.toLowerCase().match(/[a-z0-9]{3,}/g)?.slice(0, 10) || [],
  };
}

function heuristicRerank(query, candidates, topK) {
  const terms = new Set(query.toLowerCase().match(/[a-z0-9]{3,}/g) || []);
  return candidates
    .map((c) => {
      const words = new Set(c.text.toLowerCase().match(/[a-z0-9]{3,}/g) || []);
      let hits = 0;
      for (const t of terms) if (words.has(t)) hits++;
      const overlap = terms.size ? hits / terms.size : 0;
      return { ...c, rerank_score: Number((0.6 * overlap + 0.4 * Math.min(1, c.fusion_score * 40)).toFixed(3)) };
    })
    .sort((a, b) => b.rerank_score - a.rerank_score)
    .slice(0, topK);
}

async function rerank(query, candidates, topK) {
  const shortlist = candidates.slice(0, 20);
  if (!groq.hasKey() || shortlist.length <= topK) {
    return { top: heuristicRerank(query, candidates, topK), usage: { mode: "heuristic", input_tokens: 0, output_tokens: 0 } };
  }
  const listing = shortlist
    .map((c, i) => `[${i}] (p${c.metadata.page} · ${c.metadata.section}) ${c.text.slice(0, 420)}`)
    .join("\n\n");
  const messages = [
    { role: "system", content: "You score passage relevance. Reply with JSON only." },
    {
      role: "user",
      content: `Question: ${query}\n\nPassages:\n${listing}\n\nReturn JSON {"scores":[{"i":<index>,"s":<0-1>}]} for every passage. No prose.`,
    },
  ];
  try {
    const { text, usage } = await groq.complete(messages, { model: groq.MODEL_FAST, temperature: 0, max_tokens: 700 });
    const json = JSON.parse(text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1));
    const byIndex = new Map((json.scores || []).map((s) => [Number(s.i), Number(s.s)]));
    const scored = shortlist.map((c, i) => ({ ...c, rerank_score: Number((byIndex.get(i) ?? 0).toFixed(3)) }));
    return {
      top: scored.sort((a, b) => b.rerank_score - a.rerank_score).slice(0, topK),
      usage: {
        mode: "llm",
        input_tokens: usage.prompt_tokens || approxTokens(listing),
        output_tokens: usage.completion_tokens || approxTokens(text),
      },
    };
  } catch (_) {
    return { top: heuristicRerank(query, candidates, topK), usage: { mode: "heuristic-fallback", input_tokens: 0, output_tokens: 0 } };
  }
}

function buildContext(top, depth, budget = 6000, threshold = 0.2) {
  const selected = [];
  let tokens = 0;
  for (const c of top) {
    if (selected.length >= depth) break;
    if (selected.length && (c.rerank_score ?? 0) < threshold) continue;
    const t = c.metadata.token_count || approxTokens(c.text);
    if (tokens + t > budget) break;
    selected.push(c);
    tokens += t;
  }
  const context = selected
    .map((c, i) => `Source [${i + 1}] — ${c.metadata.document_name}, page ${c.metadata.page}, section "${c.metadata.section}"\n${c.text}`)
    .join("\n\n");
  return { selected, context, tokens };
}

export const citationsFor = (selected) =>
  selected.map((c, i) => ({
    index: i + 1, chunk_id: c.chunk_id, document: c.metadata.document_name,
    page: c.metadata.page, section: c.metadata.section,
  }));

/* memory: [{role:'user'|'assistant', content}] from earlier turns in this chat. */
export async function* run(query, { index, document, model = "auto", systemPrompt, memory = [] }) {
  const started = performance.now();
  const analysis = analyze(query);
  const chosen = model === "auto" ? (analysis.multi_hop || analysis.intent === "summarization" ? groq.MODEL_DEEP : groq.MODEL_FAST) : model;
  yield { event: "query_analyzed", data: { analysis, model: chosen, duration_ms: performance.now() - started } };

  if (analysis.intent === "greeting") {
    const docLine = document ? `"${document.name}" (${document.pages} page(s), ${document.chunks} chunks)` : "no document yet";
    const messages = [
      { role: "system", content: `You are the assistant of a retrieval-augmented document workspace. Loaded document: ${docLine}. Greet warmly in at most three sentences, say what you can do (summarize the document, answer questions with page citations, show the retrieved evidence), and invite a question. Do not invent document content.` },
      ...memory.slice(-6),
      { role: "user", content: query },
    ];
    yield { event: "generation_started", data: { model: groq.MODEL_FAST } };
    let answer = "";
    if (groq.hasKey()) {
      for await (const part of groq.stream(messages, { model: groq.MODEL_FAST, temperature: 0.5, max_tokens: 250 })) {
        if (part.delta) { answer += part.delta; yield { event: "token", data: { text: part.delta } }; }
      }
    } else {
      answer = `Hello! I can summarize ${document ? `"${document.name}"` : "a document you upload"}, answer questions about it with page citations, and show you the retrieved evidence. What would you like to know?`;
      yield { event: "token", data: { text: answer } };
    }
    yield {
      event: "query_complete",
      data: {
        query, answer: answer.trim(), model: groq.MODEL_FAST, analysis, citations: [], chunks: [],
        retrieval: { vector_candidates: 0, bm25_candidates: 0, hybrid_candidates: 0, reranked: 0, selected_context: 0 },
        usage: usageOf({ query, context_tokens: 0, rerank: { mode: "skipped", input_tokens: 0, output_tokens: 0 }, gen_in: approxTokens(query), gen_out: approxTokens(answer) }),
        latency: { analyze_ms: 0, retrieval_ms: 0, rerank_ms: 0, context_ms: 0, llm_ms: performance.now() - started, total_ms: performance.now() - started },
      },
    };
    return;
  }

  if (!index || !index.chunks.length) {
    yield { event: "error", data: { message: "Upload or select a document first — there is nothing indexed yet." } };
    return;
  }

  const tRetrieval = performance.now();
  const vectorHits = index.vector(query, 30);
  yield { event: "vector_search_complete", data: { count: vectorHits.length } };
  const bm25Hits = index.bm25(query, 30);
  yield { event: "bm25_search_complete", data: { count: bm25Hits.length } };
  let fused = fuse(vectorHits, bm25Hits, { limit: 40 });
  if (analysis.intent === "summarization") {
    const spread = index.spread(12).map((c) => ({
      chunk_id: c.chunk_id, text: c.text, metadata: c.metadata,
      vector_score: null, vector_rank: null, bm25_score: null, bm25_rank: null, fusion_score: 0.001, rerank_score: null,
    }));
    const seen = new Set(fused.map((c) => c.chunk_id));
    fused = fused.concat(spread.filter((c) => !seen.has(c.chunk_id)));
  }
  const retrieval_ms = performance.now() - tRetrieval;
  yield { event: "hybrid_search_complete", data: { count: fused.length, duration_ms: retrieval_ms } };

  const tRerank = performance.now();
  const { top, usage: rerankUsage } = analysis.intent === "summarization"
    ? { top: fused.slice(0, Math.max(analysis.retrieval_depth, 8)).map((c) => ({ ...c, rerank_score: c.rerank_score ?? 0.5 })), usage: { mode: "ordered", input_tokens: 0, output_tokens: 0 } }
    : await rerank(query, fused, 8);
  const rerank_ms = performance.now() - tRerank;
  yield { event: "reranking_complete", data: { input: fused.length, output: top.length, mode: rerankUsage.mode, duration_ms: rerank_ms } };

  const tContext = performance.now();
  const { selected, context, tokens: contextTokens } = buildContext(top, analysis.retrieval_depth);
  const context_ms = performance.now() - tContext;
  const selectedIds = new Set(selected.map((c) => c.chunk_id));
  const chunks = top.map((c) => ({
    chunk_id: c.chunk_id, document: c.metadata.document_name, page: c.metadata.page, section: c.metadata.section,
    token_count: c.metadata.token_count, vector_score: c.vector_score, vector_rank: c.vector_rank,
    bm25_score: c.bm25_score, bm25_rank: c.bm25_rank, fusion_score: c.fusion_score, rerank_score: c.rerank_score,
    selected: selectedIds.has(c.chunk_id), text: c.text,
  }));
  yield { event: "context_built", data: { count: selected.length, context_tokens: contextTokens, duration_ms: context_ms, chunks } };

  if (!selected.length) {
    yield { event: "token", data: { text: INSUFFICIENT } };
    yield {
      event: "query_complete",
      data: {
        query, answer: INSUFFICIENT, model: chosen, analysis, citations: [], chunks,
        retrieval: { vector_candidates: vectorHits.length, bm25_candidates: bm25Hits.length, hybrid_candidates: fused.length, reranked: top.length, selected_context: 0 },
        usage: usageOf({ query, context_tokens: 0, rerank: rerankUsage, gen_in: 0, gen_out: 0 }),
        latency: { analyze_ms: 0, retrieval_ms, rerank_ms, context_ms, llm_ms: 0, total_ms: performance.now() - started },
      },
    };
    return;
  }

  const style = STYLE[analysis.answer_style] || STYLE.explanation;
  const messages = [
    { role: "system", content: `${MASTER_RULES}\n\nOperator instructions:\n${(systemPrompt || DEFAULT_SYSTEM_PROMPT).trim()}\n\nStyle: ${style}` },
    ...memory.slice(-6),
    { role: "user", content: `Question: ${query}\n\nDetected intent: ${analysis.intent} | multi-hop: ${analysis.multi_hop}\n\nContext:\n${context}\n\nAnswer using only the context above, with [n] citations.` },
  ];
  yield { event: "generation_started", data: { model: chosen } };

  const tGen = performance.now();
  let answer = "";
  let providerUsage = {};
  if (!groq.hasKey()) {
    yield { event: "error", data: { message: "No Groq API key configured. Add VITE_GROQ_API_KEY to your environment and rebuild." } };
    return;
  }
  for await (const part of groq.stream(messages, { model: chosen, temperature: 0.2 })) {
    if (part.delta) { answer += part.delta; yield { event: "token", data: { text: part.delta } }; }
    if (part.usage) providerUsage = part.usage;
  }
  const llm_ms = performance.now() - tGen;
  answer = answer.trim() || INSUFFICIENT;
  yield { event: "generation_complete", data: { duration_ms: llm_ms } };

  yield {
    event: "query_complete",
    data: {
      query, answer, model: chosen, analysis, citations: citationsFor(selected), chunks,
      retrieval: { vector_candidates: vectorHits.length, bm25_candidates: bm25Hits.length, hybrid_candidates: fused.length, reranked: top.length, selected_context: selected.length },
      usage: usageOf({
        query, context_tokens: contextTokens, rerank: rerankUsage,
        gen_in: providerUsage.prompt_tokens || approxTokens(messages.map((m) => m.content).join("")),
        gen_out: providerUsage.completion_tokens || approxTokens(answer),
        reported: Boolean(providerUsage.prompt_tokens),
      }),
      latency: { analyze_ms: 0, retrieval_ms, rerank_ms, context_ms, llm_ms, total_ms: performance.now() - started },
    },
  };
}

function usageOf({ query, context_tokens, rerank, gen_in, gen_out, reported = false }) {
  return {
    query_tokens: approxTokens(query),
    context_tokens,
    reranker_mode: rerank.mode,
    reranker_input_tokens: rerank.input_tokens,
    reranker_output_tokens: rerank.output_tokens,
    generator_input_tokens: gen_in,
    generator_output_tokens: gen_out,
    generator_usage_reported: reported,
    total_tokens: rerank.input_tokens + rerank.output_tokens + gen_in + gen_out,
  };
}
