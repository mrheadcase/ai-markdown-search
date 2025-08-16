import { chunkMarkdown } from "./chunker.js";
import { embedTexts, cosineSim } from "./embeddings.js";
import { generateAnswer } from "./rag.js";

const els = {
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  mdUrl: document.getElementById("md-url"),
  fetchUrl: document.getElementById("fetch-url"),
  buildIndex: document.getElementById("build-index"),
  status: document.getElementById("status"),
  query: document.getElementById("query"),
  search: document.getElementById("search"),
  useAI: document.getElementById("use-ai"),
  modelId: document.getElementById("model-id"),
  progress: document.getElementById("progress"),
  results: document.getElementById("results"),
};

let mdText = "";
let chunks = [];
let vectors = []; // Float32Array[]
let indexBuilt = false;

// Drag-drop UI
setupDragDrop();

els.fileInput.addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  mdText = await f.text();
  setStatus(`Loaded file: ${f.name} (${mdText.length} chars)`);
  els.buildIndex.disabled = false;
});

els.fetchUrl.addEventListener("click", async () => {
  const url = els.mdUrl.value.trim();
  if (!url) return;
  setStatus("Fetching Markdown...");
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    mdText = await res.text();
    setStatus(`Fetched ${url} (${mdText.length} chars)`);
    els.buildIndex.disabled = false;
  } catch (err) {
    setStatus("Fetch failed: " + err.message);
  }
});

els.buildIndex.addEventListener("click", async () => {
  if (!mdText.trim()) {
    setStatus("Please load a Markdown file first.");
    return;
  }
  indexBuilt = false;
  setStatus("Chunking...");
  chunks = chunkMarkdown(mdText, { maxWords: 350, overlapWords: 40 });
  setStatus(`Chunked into ${chunks.length} segments. Creating embeddings (first time will download a small model)...`);

  const texts = chunks.map(c => c.text);
  vectors = await embedTexts(texts, (p) => setProgress(`Embedding ${p.current}/${p.total}...`));
  setProgress("");
  indexBuilt = true;
  setStatus(`Index ready. ${chunks.length} chunks embedded. You can search now.`);
});

els.search.addEventListener("click", () => doSearch());
els.query.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch();
});

async function doSearch() {
  const q = els.query.value.trim();
  if (!q) return;
  if (!indexBuilt) {
    setStatus("Please build the index first.");
    return;
  }
  setProgress("Embedding query...");
  const [qVec] = await embedTexts([q]);
  setProgress("");

  // Rank by cosine sim (dot product since normalized)
  const scored = vectors.map((v, i) => ({ i, score: cosineSim(qVec, v) }));
  scored.sort((a, b) => b.score - a.score);
  const topK = 5;
  const top = scored.slice(0, topK).map(s => ({ ...chunks[s.i], score: s.score }));

  renderResults(top);

  if (els.useAI.checked) {
    const modelId = els.modelId.value;
    setProgress("Loading in-browser AI model... (first time can be large and slow)");
    const answer = await generateAnswer(modelId, q, top, (t) => setProgress(typeof t === "string" ? t : JSON.stringify(t)));
    setProgress("");
    renderAnswer(answer);
  }
}

function setStatus(msg) {
  els.status.textContent = msg;
}
function setProgress(msg) {
  els.progress.textContent = msg;
}

function renderResults(top) {
  els.results.innerHTML = "";
  const container = document.createElement("div");

  top.forEach((t, idx) => {
    const card = document.createElement("div");
    card.className = "result";
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `#${idx + 1} • score: ${t.score.toFixed(3)} • heading: ${t.heading || "N/A"}`;
    const content = document.createElement("div");
    content.className = "content";
    content.textContent = t.text;
    card.appendChild(meta);
    card.appendChild(content);
    container.appendChild(card);
  });

  els.results.appendChild(container);
}

function renderAnswer(text) {
  const box = document.createElement("div");
  box.className = "answer";
  box.textContent = text;
  els.results.prepend(box);
}

function setupDragDrop() {
  ["dragenter", "dragover"].forEach(ev => {
    els.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      els.dropZone.style.borderColor = "var(--accent)";
    });
  });
  ["dragleave", "drop"].forEach(ev => {
    els.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      els.dropZone.style.borderColor = "var(--border)";
    });
  });
  els.dropZone.addEventListener("drop", async (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    if (!f.name.endsWith(".md") && !f.name.endsWith(".markdown")) {
      setStatus("Please drop a Markdown (.md) file.");
      return;
    }
    mdText = await f.text();
    setStatus(`Loaded file: ${f.name} (${mdText.length} chars)`);
    els.buildIndex.disabled = false;
  });
}