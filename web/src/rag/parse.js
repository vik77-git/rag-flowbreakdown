/* Document parsing in the browser: TXT/MD, PDF (pdf.js) and DOCX (mammoth). */

export const SUPPORTED = [".txt", ".md", ".pdf", ".docx"];

function clean(text) {
  return String(text || "")
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

async function parsePdf(file) {
  const pdfjs = await import("pdfjs-dist");
  const workerUrl = (await import("pdfjs-dist/build/pdf.worker.min.mjs?url")).default;
  pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
  const data = new Uint8Array(await file.arrayBuffer());
  const doc = await pdfjs.getDocument({ data }).promise;
  const pages = [];
  for (let n = 1; n <= doc.numPages; n++) {
    const page = await doc.getPage(n);
    const content = await page.getTextContent();
    let text = "";
    let lastY = null;
    for (const item of content.items) {
      const y = item.transform?.[5];
      if (lastY !== null && Math.abs(y - lastY) > 4) text += "\n";
      text += item.str + (item.hasEOL ? "\n" : " ");
      lastY = y;
    }
    pages.push({ page: n, text: clean(text) });
  }
  return pages;
}

async function parseDocx(file) {
  const mammoth = (await import("mammoth/mammoth.browser.js")).default;
  const { value } = await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() });
  return paginate(clean(value));
}

/* Plain text has no pages; split on ~3500 characters so the viewer stays responsive. */
function paginate(text, size = 3500) {
  if (text.length <= size) return [{ page: 1, text }];
  const pages = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(text.length, start + size);
    const brk = text.lastIndexOf("\n\n", end);
    if (brk > start + size * 0.5) end = brk;
    pages.push({ page: pages.length + 1, text: text.slice(start, end).trim() });
    start = end;
  }
  return pages;
}

export async function parseFile(file) {
  const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
  if (!SUPPORTED.includes(ext)) throw new Error(`Unsupported file type ${ext}. Use PDF, DOCX, TXT or MD.`);
  if (file.size > 25 * 1024 * 1024) throw new Error("File is larger than 25 MB.");
  let pages;
  if (ext === ".pdf") pages = await parsePdf(file);
  else if (ext === ".docx") pages = await parseDocx(file);
  else pages = paginate(clean(await file.text()));
  pages = pages.filter((p) => p.text);
  if (!pages.length) throw new Error("No extractable text found in this file.");
  return { name: file.name, extension: ext, size_bytes: file.size, pages };
}

export async function parseText(name, text) {
  return { name, extension: "." + (name.split(".").pop() || "txt").toLowerCase(), size_bytes: text.length, pages: paginate(clean(text)) };
}
