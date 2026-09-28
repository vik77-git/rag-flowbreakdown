/* Structure-aware chunking: detect headings, split into sentences, then grow
   chunks up to a character budget with one sentence of overlap. */

const HEADING = /^(?:(?:\d+(?:\.\d+)*)[.)]?\s+\S.*|#{1,4}\s+\S.*|[A-Z][A-Z0-9 &/,'-]{5,70})$/;

export const approxTokens = (text) => Math.max(1, Math.round(String(text).length / 4));

function sentences(text) {
  return text
    .split(/(?<=[.!?:;])\s+(?=[A-Z0-9"“(])|\n{2,}/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function isHeading(line) {
  const t = line.trim();
  return t.length > 2 && t.length < 90 && HEADING.test(t) && !/[.!?]$/.test(t.replace(/^\d+(\.\d+)*[.)]?\s*/, ""));
}

export function chunkDocument(parsed, documentId, { minChars = 200, maxChars = 1600, overlap = 1 } = {}) {
  const chunks = [];
  const sections = [];
  let section = "Introduction";

  for (const { page, text } of parsed.pages) {
    const blocks = [];
    let buffer = [];
    for (const rawLine of text.split("\n")) {
      const line = rawLine.trim();
      if (!line) { if (buffer.length) { blocks.push({ heading: section, body: buffer.join(" ") }); buffer = []; } continue; }
      if (isHeading(line)) {
        if (buffer.length) { blocks.push({ heading: section, body: buffer.join(" ") }); buffer = []; }
        section = line.replace(/^#{1,4}\s*/, "");
        if (!sections.some((s) => s.title.toLowerCase() === section.toLowerCase())) sections.push({ title: section, page });
        continue;
      }
      buffer.push(line);
    }
    if (buffer.length) blocks.push({ heading: section, body: buffer.join(" ") });

    for (const block of blocks) {
      const parts = sentences(block.body);
      let current = [];
      const flush = () => {
        const body = current.join(" ").trim();
        if (!body) return;
        chunks.push({
          chunk_id: `${documentId}_c${chunks.length}`,
          document_id: documentId,
          text: body,
          metadata: {
            document_id: documentId,
            document_name: parsed.name,
            page,
            section: block.heading,
            chunk_type: "text",
            token_count: approxTokens(body),
          },
        });
        current = overlap > 0 ? current.slice(-overlap) : [];
      };
      for (const sentence of parts) {
        const size = current.join(" ").length;
        if (size + sentence.length > maxChars && size >= minChars) flush();
        current.push(sentence);
      }
      if (current.join(" ").trim().length) {
        const tail = current.join(" ").trim();
        const last = chunks[chunks.length - 1];
        if (tail.length < minChars && last && last.metadata.page === page) {
          last.text += " " + tail;
          last.metadata.token_count = approxTokens(last.text);
        } else flush();
      }
    }
  }
  return { chunks, sections };
}

export function suggestionsFor(name, sections) {
  const subject = name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  const out = [`Summarize the key points in ${subject}.`];
  for (const s of sections.slice(0, 2)) out.push(`What does the document say about ${s.title}?`);
  if (sections.length > 1) out.push(`Compare ${sections[0].title} and ${sections[1].title}.`);
  return out.slice(0, 4);
}
