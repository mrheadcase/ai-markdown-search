# AI Search for Markdown (Local & Free)

This is a zero-backend, privacy-first way to search and chat with a Markdown file:
- Semantic search via local embeddings (Xenova/all-MiniLM-L6-v2 using Transformers.js).
- Optional AI-generated answers using WebGPU in your browser (web-llm + Phi-3-mini or Llama 3.1).
- No API keys. No servers. Free to host on GitHub Pages.

## Quick start

1. Push all files in this repository to the `main` branch (see file list in the root).
2. GitHub Actions will automatically run the "Deploy to GitHub Pages" workflow and publish the site.
3. After it succeeds, visit your Pages URL (also shown in the workflow summary):  
   https://mrheadcase.github.io/ai-markdown-search/

If the workflow does not publish automatically, go to Settings → Pages and set "Source" to "GitHub Actions", then re-run the workflow.

## Local development

- Serve `index.html` via a local HTTP server (not file://) to avoid CORS issues:
  - Python: `python3 -m http.server 8080` → open http://localhost:8080
  - Node: `npx http-server -p 8080` → open http://localhost:8080

## Notes and tips

- First run downloads the embedding model (~100MB, cached by the browser). Subsequent loads are faster.
- AI answer mode:
  - Requires a modern browser with WebGPU (Chrome, Edge, or recent Firefox Nightly).
  - Model download is large (hundreds of MB to a few GB depending on choice).
  - If you only need search, leave "Use in-browser AI answer" unchecked for speed.
- All data stays local in your browser. Nothing is uploaded.

## How it works

- Chunking: Markdown is split by headings and paragraphs into ~350-word overlapping chunks.
- Embeddings: Each chunk is embedded via Transformers.js (MiniLM). Query embedding is compared to all chunks via cosine similarity.
- RAG (optional): Top chunks are sent to an in-browser LLM via `@mlc-ai/web-llm` to synthesize an answer constrained to the provided context.

## Troubleshooting

- If fetching a Markdown URL fails, the remote server may block cross-origin (CORS). Download the file and drop it instead.
- If WebGPU initialization fails:
  - Update your browser and GPU drivers.
  - Try the smaller "Phi-3-mini-4k-instruct" model first.
  - Disable "Use in-browser AI answer" to use just semantic search.

## License

MIT