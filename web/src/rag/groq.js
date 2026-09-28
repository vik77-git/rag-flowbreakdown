/* Groq client for the browser. Keys come from Vite env vars (VITE_*), which are
   injected at build time — on Vercel set them in Project Settings → Environment Variables. */

const BASE = (import.meta.env.VITE_GROQ_BASE_URL || "https://api.groq.com/openai/v1").replace(/\/$/, "");
export const MODEL_FAST = import.meta.env.VITE_GROQ_MODEL_20B || "openai/gpt-oss-20b";
export const MODEL_DEEP = import.meta.env.VITE_GROQ_MODEL_120B || "openai/gpt-oss-120b";
export const MODELS = [MODEL_FAST, MODEL_DEEP];

const KEYS = [import.meta.env.VITE_GROQ_API_KEY, import.meta.env.VITE_GROQ_API_KEY_2]
  .map((k) => (k || "").trim())
  .filter(Boolean);

export const hasKey = () => KEYS.length > 0;

/* Round-robin rotation with a short cooldown after a rate limit. */
const cooldown = new Map();
let cursor = 0;
function nextKey() {
  if (!KEYS.length) throw new Error("No Groq API key configured. Set VITE_GROQ_API_KEY and redeploy.");
  const now = Date.now();
  for (let i = 0; i < KEYS.length; i++) {
    const key = KEYS[(cursor + i) % KEYS.length];
    if ((cooldown.get(key) || 0) <= now) {
      cursor = (cursor + i + 1) % KEYS.length;
      return key;
    }
  }
  return KEYS[cursor++ % KEYS.length];
}

async function request(body, key) {
  const res = await fetch(`${BASE}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify(body),
  });
  if (res.status === 429 || res.status === 401) {
    cooldown.set(key, Date.now() + 30000);
    const detail = await res.text().catch(() => "");
    throw new Error(res.status === 401 ? "Groq rejected the API key." : `Rate limited by Groq. ${detail.slice(0, 120)}`);
  }
  if (!res.ok) throw new Error(`Groq request failed (${res.status}).`);
  return res;
}

export async function complete(messages, { model = MODEL_FAST, temperature = 0.1, max_tokens = 1200 } = {}) {
  let lastError;
  for (let attempt = 0; attempt < Math.max(1, KEYS.length); attempt++) {
    const key = nextKey();
    try {
      const res = await request({ model, messages, temperature, max_tokens, stream: false }, key);
      const data = await res.json();
      return {
        text: data.choices?.[0]?.message?.content || "",
        usage: data.usage || {},
      };
    } catch (err) { lastError = err; }
  }
  throw lastError;
}

export async function* stream(messages, { model = MODEL_FAST, temperature = 0.2, max_tokens = 1600 } = {}) {
  const key = nextKey();
  const res = await request({ model, messages, temperature, max_tokens, stream: true }, key);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop();
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        const json = JSON.parse(payload);
        const delta = json.choices?.[0]?.delta?.content;
        if (delta) yield { delta };
        if (json.usage) yield { usage: json.usage };
      } catch (_) { /* ignore partial frame */ }
    }
  }
}
