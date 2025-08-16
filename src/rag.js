// Optional in-browser LLM via web-llm (WebGPU). Big download; free; fully local.
let engine = null;
let currentModelId = null;

export async function ensureEngine(modelId, onProgress) {
  if (engine && currentModelId === modelId) return engine;

  try {
    const { CreateMLCEngine } = await import("https://esm.run/@mlc-ai/web-llm");
    engine = await CreateMLCEngine(modelId, {
      initProgressCallback: (p) => {
        if (onProgress) onProgress(p.text || JSON.stringify(p));
      },
    });
    currentModelId = modelId;
    return engine;
  } catch (err) {
    throw new Error("Failed to initialize web-llm. Ensure a modern browser with WebGPU and try again. " + err.message);
  }
}

export async function generateAnswer(modelId, question, topChunks, onProgress) {
  const eng = await ensureEngine(modelId, onProgress);
  const context = topChunks.map((c, i) => `Chunk ${i + 1} [heading: ${c.heading || "N/A"}]:\n${c.text}`).join("\n\n---\n\n");

  const sys = `You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say you don't know. Be concise.`;
  const user = `Question: ${question}

Context:
${context}`;

  // web-llm supports an OpenAI-like chat.completions.create in recent versions.
  if (eng.chat?.completions?.create) {
    const resp = await eng.chat.completions.create({
      messages: [
        { role: "system", content: sys },
        { role: "user", content: user },
      ],
      stream: true,
      temperature: 0.2,
      max_tokens: 512,
    });

    let answer = "";
    for await (const chunk of resp) {
      const delta = chunk.choices?.[0]?.delta?.content ?? "";
      answer += delta;
      if (onProgress) onProgress(delta);
    }
    return answer.trim();
  }

  // Fallback to a simpler generate API
  const prompt = `${sys}\n\n${user}\n\nAnswer:`;
  const res = await eng.generate(prompt, { maxTokens: 512, temperature: 0.2, stream: true });
  let answer = "";
  for await (const piece of res) {
    answer += piece;
    if (onProgress) onProgress(piece);
  }
  return answer.trim();
}