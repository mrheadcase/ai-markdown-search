// AI Markdown Q&A — all in-browser
// Uses @xenova/transformers via CDN for sentence embeddings

const els = {
  dropzone: document.getElementById('dropzone'),
  input: document.getElementById('file-input'),
  hint: document.getElementById('hint'),
  progress: document.getElementById('progress'),
  progressBar: document.getElementById('progress-bar'),
  dots: {
    upload: document.getElementById('status-upload'),
    parse: document.getElementById('status-parse'),
    models: document.getElementById('status-models'),
    index: document.getElementById('status-index'),
  },
  qaCard: document.getElementById('qa-card'),
  question: document.getElementById('question'),
  ask: document.getElementById('ask-btn'),
  answer: document.getElementById('answer'),
  sources: document.getElementById('sources'),
};

const state = {
  chunks: [], // { id, text, headingPath, tokenCount }
  vectors: [], // Float32Array normalized
  embed: null, // function(texts)=>Promise<Float32Array[]>
  ready: false,
  fileName: null,
};

// UI helpers
function setHint(text) {
  els.hint.textContent = text;
}
function setProgressVisible(show) {
  els.progress.hidden = !show;
}
function setProgress(pct) {
  els.progressBar.style.width = Math.max(0, Math.min(100, pct)).toFixed(1) + '%';
}
function setDot(dotEl, state) {
  dotEl.classList.remove('pending', 'active', 'done', 'error');
  dotEl.classList.add(state);
}
function resetStatus() {
  Object.values(els.dots).forEach((d) => setDot(d, 'pending'));
  setProgressVisible(false);
  setProgress(0);
  setHint('No file loaded');
  els.answer.textContent = '';
  els.sources.innerHTML = '';
}

resetStatus();

// Dropzone behavior
['dragenter', 'dragover'].forEach((evt) => {
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.dropzone.classList.add('hover');
  });
});
['dragleave', 'drop'].forEach((evt) => {
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.dropzone.classList.remove('hover');
  });
});
els.dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});
els.input.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  resetStatus();
  state.fileName = file.name;
  setDot(els.dots.upload, 'active');
  setHint(`Uploading ${file.name} ...`);

  const text = await readFileText(file).catch((err) => {
    console.error(err);
    setDot(els.dots.upload, 'error');
    setHint('Failed to read file');
    return null;
  });
  if (!text) return;
  setDot(els.dots.upload, 'done');

  setDot(els.dots.parse, 'active');
  setHint('Parsing markdown ...');
  const chunks = parseAndChunkMarkdown(text);
  state.chunks = chunks;
  setDot(els.dots.parse, 'done');
  setHint(`Parsed ${chunks.length} chunks`);

  // Load embedding model
  try {
    setDot(els.dots.models, 'active');
    setHint('Downloading AI model (embeddings) ...');
    setProgressVisible(true);
    setProgress(5);

    // Lazy import transformers from CDN
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers');
    // Configure environment
    env.allowLocalModels = false; // always fetch from CDN
    // Reduce console noise
    env.backends.onnx.wasm.numThreads = Math.max(1, navigator?.hardwareConcurrency ? Math.min(4, navigator.hardwareConcurrency) : 1);

    const progress_callback = (data) => {
      // data: { status, loaded, total }
      if (data?.status?.toLowerCase?.().includes('loading')) {
        const pct = data.total ? (data.loaded / data.total) * 100 : 10;
        setProgress(Math.min(95, pct));
      }
    };

    const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { progress_callback });

    // Build embedding function with mean pooling and L2 normalize
    state.embed = async (texts) => {
      if (!Array.isArray(texts)) texts = [texts];
      const output = await extractor(texts, { pooling: 'mean', normalize: true });
      // output.data is Float32Array or array of them
      if (Array.isArray(output)) {
        return output.map((o) => o.data);
      } else {
        return [output.data];
      }
    };

    setDot(els.dots.models, 'done');
    setProgress(100);
    setProgressVisible(false);
  } catch (err) {
    console.error(err);
    setDot(els.dots.models, 'error');
    setHint('Failed to load AI model. Check your internet connection.');
    return;
  }

  // Create index
  try {
    setDot(els.dots.index, 'active');
    setHint('Indexing content ...');
    setProgressVisible(true);
    const total = state.chunks.length;
    const batchSize = 16;
    state.vectors = new Array(total);

    for (let i = 0; i < total; i += batchSize) {
      const batch = state.chunks.slice(i, i + batchSize).map((c) => c.text);
      // Embed batch
      const vecs = await state.embed(batch);
      vecs.forEach((v, j) => {
        state.vectors[i + j] = v; // already normalized
      });
      setProgress(Math.round(((i + batch.length) / total) * 100));
      await microDelay();
    }

    setDot(els.dots.index, 'done');
    setProgressVisible(false);
    setHint(`Ready. Indexed ${total} chunks from ${state.fileName}`);

    state.ready = true;
    els.qaCard.hidden = false;
    els.question.disabled = false;
    els.ask.disabled = false;
    els.question.focus();
  } catch (err) {
    console.error(err);
    setDot(els.dots.index, 'error');
    setHint('Failed to index content.');
  }
}

// Read file as text
function readFileText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => resolve(reader.result);
    reader.readAsText(file);
  });
}

// Simple markdown parsing and chunking
function parseAndChunkMarkdown(md) {
  const lines = md.split(/\r?\n/);
  const chunks = [];
  let path = [];
  let buf = [];
  let sectionId = 0;

  const flush = () => {
    const text = cleanText(buf.join('\n').trim());
    if (text) {
      chunks.push({
        id: sectionId++,
        text: limitText(text, 1200),
        headingPath: path.join(' > ') || 'Document',
        tokenCount: Math.min(1200, text.length / 4),
      });
    }
    buf = [];
  };

  for (const line of lines) {
    const m = /^(#{1,6})\s+(.*)$/.exec(line);
    if (m) {
      flush();
      const level = m[1].length;
      const title = m[2].trim();
      path = path.slice(0, level - 1);
      path[level - 1] = title;
    } else if (line.trim() === '' && buf.length) {
      // paragraph break — emit a chunk to keep chunks small
      flush();
    } else {
      buf.push(line);
    }
  }
  flush();

  // Merge tiny chunks with neighbors
  const minChars = 200;
  const merged = [];
  for (let i = 0; i < chunks.length; i++) {
    const cur = chunks[i];
    if (cur.text.length < minChars && i + 1 < chunks.length) {
      const next = chunks[i + 1];
      merged.push({
        id: cur.id,
        text: `${cur.text}\n\n${next.text}`.trim(),
        headingPath: next.headingPath || cur.headingPath,
        tokenCount: cur.tokenCount + next.tokenCount,
      });
      i++; // skip next
    } else {
      merged.push(cur);
    }
  }

  // Cap chunks to prevent huge downloads on extreme docs
  const maxChunks = 400;
  if (merged.length > maxChunks) {
    return merged.slice(0, maxChunks);
  }
  return merged;
}

function cleanText(t) {
  return t
    .replace(/```[\s\S]*?```/g, '[code block]')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/<[^>]+>/g, '')
    .replace(/[\t\r]+/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function limitText(t, maxLen) {
  if (t.length <= maxLen) return t;
  return t.slice(0, maxLen) + '…';
}

function microDelay() {
  return new Promise((r) => setTimeout(r, 0));
}

// Cosine similarity (vectors already normalized)
function cosine(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// Lightweight answer synthesis from top chunks
const STOPWORDS = new Set('a,an,the,and,or,of,to,in,on,for,with,as,by,at,from,is,are,was,were,be,been,being,it,its,that,this,these,those,which,what,who,whom,into,about,how,why,when,where,can,could,should,would,may,might,than,then,also,not,no,do,does,did,done,if,else,but,so,such,using,use,used,via,like,between,within,over,under,per,each'.split(',').map((x) => x.trim()));

function tokenize(s) {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w && !STOPWORDS.has(w));
}

function synthesizeAnswer(question, contexts) {
  const qTokens = tokenize(question);
  const sentences = contexts
    .flatMap((c) => c.text.split(/(?<=[.!?])\s+/).map((t) => ({ text: t.trim(), src: c.headingPath })))
    .filter((s) => s.text.length > 0);

  const scored = sentences
    .map((s) => ({
      ...s,
      score: scoreSentence(s.text, qTokens),
    }))
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  if (scored.length === 0) {
    const fallback = contexts[0]?.text || 'No relevant information found.';
    return limitText(fallback, 500);
  }
  const answer = scored.map((s) => s.text).join(' ');
  return limitText(answer, 600);
}

function scoreSentence(text, qTokens) {
  const t = tokenize(text);
  let score = 0;
  for (const q of qTokens) {
    if (t.includes(q)) score += 2;
    else {
      // partial match
      if (t.some((w) => w.startsWith(q) || q.startsWith(w))) score += 1;
    }
  }
  // Prefer slightly longer but not too long sentences
  const len = t.length;
  if (len > 4 && len < 60) score += 1;
  return score;
}

// Ask flow
els.ask.addEventListener('click', onAsk);
els.question.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') onAsk();
});

async function onAsk() {
  const q = (els.question.value || '').trim();
  if (!q || !state.ready || !state.embed) return;
  els.ask.disabled = true;
  els.answer.textContent = 'Thinking…';
  els.sources.innerHTML = '';
  try {
    const [qVec] = await state.embed(q);
    // Compute similarities
    const sims = state.vectors.map((v, i) => ({ i, s: cosine(qVec, v) }));
    sims.sort((a, b) => b.s - a.s);
    const topK = 4;
    const top = sims.slice(0, topK).map(({ i, s }) => ({ ...state.chunks[i], sim: s }));

    const answer = synthesizeAnswer(q, top);
    els.answer.textContent = answer;

    // Show sources
    els.sources.innerHTML = '';
    top.forEach((c) => {
      const pill = document.createElement('span');
      pill.className = 'source-pill';
      pill.textContent = `${c.headingPath} (${(c.sim * 100).toFixed(1)}%)`;
      pill.title = c.text;
      els.sources.appendChild(pill);
    });
  } catch (err) {
    console.error(err);
    els.answer.textContent = 'Sorry, something went wrong answering your question.';
  } finally {
    els.ask.disabled = false;
  }
}

// Expose a link to try the bundled sample.md (for convenience in dev)
(function attachSampleLoader(){
  let hintLink = document.getElementById('load-sample');
  if (!hintLink) {
    hintLink = document.createElement('a');
    hintLink.id = 'load-sample';
    hintLink.href = '#';
    hintLink.textContent = 'Load included sample.md';
    hintLink.style.marginLeft = '8px';
    const hint = document.getElementById('hint');
    const wrap = document.createElement('span');
    wrap.appendChild(document.createTextNode(' '));
    wrap.appendChild(hintLink);
    hint?.after(wrap);
  }

  hintLink?.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
      setDot(els.dots.upload, 'active');
      setHint('Fetching sample.md ...');
      const res = await fetch('../sample.md');
      const text = await res.text();
      setDot(els.dots.upload, 'done');
      setDot(els.dots.parse, 'active');
      const chunks = parseAndChunkMarkdown(text);
      state.chunks = chunks;
      state.fileName = 'sample.md';
      setDot(els.dots.parse, 'done');
      setHint(`Parsed ${chunks.length} chunks`);

      // Ensure model loaded if not yet
      if (!state.embed) {
        setDot(els.dots.models, 'active');
        setHint('Downloading AI model (embeddings) ...');
        setProgressVisible(true);
        const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers');
        env.allowLocalModels = false;
        env.backends.onnx.wasm.numThreads = Math.max(1, navigator?.hardwareConcurrency ? Math.min(4, navigator.hardwareConcurrency) : 1);
        const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { progress_callback: (d)=>{ if (d.total) setProgress(Math.min(95, (d.loaded/d.total)*100)); } });
        state.embed = async (texts) => {
          if (!Array.isArray(texts)) texts = [texts];
          const output = await extractor(texts, { pooling: 'mean', normalize: true });
          if (Array.isArray(output)) return output.map((o)=>o.data); else return [output.data];
        };
        setDot(els.dots.models, 'done');
        setProgress(100);
        setProgressVisible(false);
      }

      // Index as usual
      setDot(els.dots.index, 'active');
      setProgressVisible(true);
      const total = state.chunks.length;
      state.vectors = new Array(total);
      for (let i=0;i<total;i+=16){
        const batch = state.chunks.slice(i, i+16).map(c=>c.text);
        const vecs = await state.embed(batch);
        vecs.forEach((v,j)=>{ state.vectors[i+j]=v; });
        setProgress(Math.round(((i + batch.length) / total) * 100));
        await microDelay();
      }
      setDot(els.dots.index, 'done');
      setProgressVisible(false);
      setHint(`Ready. Indexed ${total} chunks from sample.md`);
      state.ready = true;
      els.qaCard.hidden = false;
      els.question.disabled = false;
      els.ask.disabled = false;
      els.question.focus();
    } catch (e) {
      console.error(e);
      setDot(els.dots.upload, 'error');
      setHint('Failed to load sample.md');
    }
  });
})();
