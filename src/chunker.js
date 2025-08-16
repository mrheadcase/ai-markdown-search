// Simple Markdown chunker: splits by headings and paragraphs, then merges to ~500 tokens-ish.
export function chunkMarkdown(mdText, opts = {}) {
  const maxWords = opts.maxWords ?? 350; // ~350-500 tokens
  const overlapWords = opts.overlapWords ?? 40;

  // Normalize line endings
  const md = mdText.replace(/\r\n/g, "\n");

  // Split by top-level blocks (headings or blank-line-separated paragraphs)
  const lines = md.split("\n");
  const blocks = [];
  let buf = [];
  let currentHeading = "";
  for (const line of lines) {
    const headingMatch = /^#{1,6}\s+(.*)$/.exec(line);
    if (headingMatch) {
      // flush previous
      if (buf.length) {
        blocks.push({ heading: currentHeading, text: buf.join("\n").trim() });
        buf = [];
      }
      currentHeading = headingMatch[1].trim();
      continue;
    }
    if (line.trim() === "" && buf.length) {
      blocks.push({ heading: currentHeading, text: buf.join("\n").trim() });
      buf = [];
    } else {
      buf.push(line);
    }
  }
  if (buf.length) blocks.push({ heading: currentHeading, text: buf.join("\n").trim() });

  // Merge blocks into overlapping chunks by word count
  const chunks = [];
  let acc = [];
  let accWords = 0;

  function wordsCount(s) {
    return (s.match(/\S+/g) || []).length;
  }

  for (const b of blocks) {
    const t = b.text.trim();
    if (!t) continue;
    const w = wordsCount(t);
    if (accWords + w <= maxWords) {
      acc.push(b);
      accWords += w;
    } else {
      if (acc.length) {
        const merged = mergeBlocks(acc);
        chunks.push(merged);
        // overlap
        const overlap = takeFromEnd(acc, overlapWords);
        acc = overlap;
        accWords = wordsCount(mergeBlocks(overlap).text);
      }
      acc.push(b);
      accWords += w;
      // if a single block is huge, split by sentences
      if (accWords > maxWords * 1.5) {
        const merged = mergeBlocks(acc);
        const splitChunks = splitBySentences(merged.text, maxWords, overlapWords, merged.heading);
        chunks.push(...splitChunks);
        acc = [];
        accWords = 0;
      }
    }
  }
  if (acc.length) chunks.push(mergeBlocks(acc));

  // assign ids
  return chunks.map((c, i) => ({ id: i, ...c }));
}

function mergeBlocks(blocks) {
  const heading = blocks.find(b => b.heading)?.heading || "";
  const text = blocks.map(b => b.text).join("\n\n").trim();
  return { heading, text };
}

function takeFromEnd(blocks, targetWords) {
  const out = [];
  let count = 0;
  for (let i = blocks.length - 1; i >= 0; i--) {
    const b = blocks[i];
    out.unshift(b);
    count += (b.text.match(/\S+/g) || []).length;
    if (count >= targetWords) break;
  }
  return out;
}

function splitBySentences(text, maxWords, overlapWords, heading) {
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks = [];
  let buf = [];
  let words = 0;
  for (const s of sentences) {
    const w = (s.match(/\S+/g) || []).length;
    if (words + w <= maxWords) {
      buf.push(s);
      words += w;
    } else {
      const chunkText = buf.join(" ").trim();
      if (chunkText) chunks.push({ heading, text: chunkText });
      // overlap
      const overlap = takeLastWords(buf.join(" "), overlapWords);
      buf = overlap ? [overlap] : [];
      words = (overlap.match(/\S+/g) || []).length;
      buf.push(s);
      words += w;
    }
  }
  const final = buf.join(" ").trim();
  if (final) chunks.push({ heading, text: final });
  return chunks;
}

function takeLastWords(text, n) {
  const arr = (text.match(/\S+/g) || []);
  if (!arr.length) return "";
  const take = arr.slice(Math.max(0, arr.length - n));
  return take.join(" ");
}