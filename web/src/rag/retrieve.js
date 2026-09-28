/* Hybrid retrieval, fully client-side:
   - lexical: BM25 (Okapi, k1=1.5, b=0.75)
   - vector: TF-IDF cosine similarity over the chunk vocabulary
   - fusion: reciprocal rank fusion */

const STOP = new Set("the a an of and or to in is are was were be been it its this that for on at by with as from into than then so such not no if we you they he she i our your their".split(" "));

export function tokenize(text) {
  return String(text).toLowerCase().match(/[a-z0-9]+/g)?.filter((t) => t.length > 1 && !STOP.has(t)) || [];
}

export class HybridIndex {
  constructor() { this.reset(); }

  reset() {
    this.chunks = [];
    this.docFreq = new Map();
    this.termFreqs = [];
    this.lengths = [];
    this.avgLength = 0;
  }

  build(chunks) {
    this.reset();
    this.chunks = chunks;
    for (const chunk of chunks) {
      const tokens = tokenize(chunk.text + " " + (chunk.metadata.section || ""));
      const tf = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
      for (const t of tf.keys()) this.docFreq.set(t, (this.docFreq.get(t) || 0) + 1);
      this.termFreqs.push(tf);
      this.lengths.push(tokens.length || 1);
    }
    this.avgLength = this.lengths.reduce((a, b) => a + b, 0) / (this.lengths.length || 1);
  }

  idf(term) {
    const n = this.chunks.length || 1;
    const df = this.docFreq.get(term) || 0;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  bm25(query, k = 30) {
    const terms = tokenize(query);
    const scored = this.chunks.map((chunk, i) => {
      const tf = this.termFreqs[i];
      let score = 0;
      for (const term of terms) {
        const f = tf.get(term);
        if (!f) continue;
        const norm = 1.5 + 1;
        score += this.idf(term) * ((f * norm) / (f + 1.5 * (1 - 0.75 + 0.75 * (this.lengths[i] / this.avgLength))));
      }
      return { chunk, score };
    });
    return scored.filter((s) => s.score > 0).sort((a, b) => b.score - a.score).slice(0, k)
      .map((s, rank) => ({ chunk: s.chunk, score: Number(s.score.toFixed(4)), rank: rank + 1 }));
  }

  vector(query, k = 30) {
    const qTf = new Map();
    for (const t of tokenize(query)) qTf.set(t, (qTf.get(t) || 0) + 1);
    const qVec = new Map();
    let qNorm = 0;
    for (const [t, f] of qTf) { const w = (1 + Math.log(f)) * this.idf(t); qVec.set(t, w); qNorm += w * w; }
    qNorm = Math.sqrt(qNorm) || 1;

    const scored = this.chunks.map((chunk, i) => {
      const tf = this.termFreqs[i];
      let dot = 0, norm = 0;
      for (const [t, f] of tf) {
        const w = (1 + Math.log(f)) * this.idf(t);
        norm += w * w;
        if (qVec.has(t)) dot += w * qVec.get(t);
      }
      const score = dot / ((Math.sqrt(norm) || 1) * qNorm);
      return { chunk, score };
    });
    return scored.filter((s) => s.score > 0).sort((a, b) => b.score - a.score).slice(0, k)
      .map((s, rank) => ({ chunk: s.chunk, score: Number(s.score.toFixed(4)), rank: rank + 1 }));
  }

  /* Ordered slice of the document — used for summarization, where ranking by
     keyword overlap is the wrong tool. */
  spread(limit = 10) {
    if (this.chunks.length <= limit) return this.chunks.slice();
    const step = this.chunks.length / limit;
    return Array.from({ length: limit }, (_, i) => this.chunks[Math.floor(i * step)]);
  }
}

export function fuse(vectorHits, bm25Hits, { k = 60, limit = 40 } = {}) {
  const map = new Map();
  const touch = (chunk) => {
    if (!map.has(chunk.chunk_id)) {
      map.set(chunk.chunk_id, {
        chunk_id: chunk.chunk_id, text: chunk.text, metadata: chunk.metadata,
        vector_score: null, vector_rank: null, bm25_score: null, bm25_rank: null,
        fusion_score: 0, rerank_score: null,
      });
    }
    return map.get(chunk.chunk_id);
  };
  for (const hit of vectorHits) {
    const c = touch(hit.chunk);
    c.vector_score = hit.score; c.vector_rank = hit.rank; c.fusion_score += 1 / (k + hit.rank);
  }
  for (const hit of bm25Hits) {
    const c = touch(hit.chunk);
    c.bm25_score = hit.score; c.bm25_rank = hit.rank; c.fusion_score += 1 / (k + hit.rank);
  }
  return [...map.values()]
    .map((c) => ({ ...c, fusion_score: Number(c.fusion_score.toFixed(5)) }))
    .sort((a, b) => b.fusion_score - a.fusion_score)
    .slice(0, limit);
}
