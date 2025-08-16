// Client-side embeddings using Transformers.js (no server, no API keys)
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@3.0.0";

env.allowLocalModels = false; // force CDN
// Optional: tune wasm threads or cache dir via env if desired

let extractorPromise = null;

async function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  return extractorPromise;
}

export async function embedTexts(texts, onProgress) {
  const extractor = await getExtractor();
  const out = [];
  const total = texts.length;
  for (let i = 0; i < texts.length; i++) {
    const text = texts[i];
    // pooling and normalize handled by the pipeline options
    const emb = await extractor(text, { pooling: "mean", normalize: true });
    // emb is a Float32Array
    out.push(emb.data ? emb.data : emb);
    if (onProgress && i % 5 === 0) onProgress({ current: i + 1, total });
  }
  if (onProgress) onProgress({ current: total, total });
  return out;
}

export function cosineSim(a, b) {
  // a and b assumed L2-normalized -> cosine = dot
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}