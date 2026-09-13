export function parseSummariseResponse(
  responseText: string,
  requestedNodeIds: string[]
): Record<string, string> {
  if (!responseText || typeof responseText !== 'string') {
    throw new Error('AI response was empty or invalid.');
  }

  let cleanedText = responseText.trim();

  // Strip Markdown code fences if present (e.g. ```json ... ``` or ``` ... ```)
  const fenceMatch = cleanedText.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenceMatch && fenceMatch[1]) {
    cleanedText = fenceMatch[1].trim();
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(cleanedText);
  } catch {
    const firstBrace = cleanedText.indexOf('{');
    const lastBrace = cleanedText.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      try {
        parsed = JSON.parse(cleanedText.substring(firstBrace, lastBrace + 1));
      } catch {
        throw new Error('AI response could not be parsed as JSON.');
      }
    } else {
      throw new Error('AI response could not be parsed as JSON.');
    }
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('AI response was not a valid JSON object mapping node IDs to summaries.');
  }

  const requestedSet = new Set(requestedNodeIds);
  let mapObj = parsed as Record<string, unknown>;

  // Check if mapObj has any requested keys directly, or unwrap inner object (e.g. { "summaries": { ... } })
  const hasDirectKeys = Object.keys(mapObj).some((k) => requestedSet.has(k));
  if (!hasDirectKeys) {
    for (const val of Object.values(mapObj)) {
      if (val && typeof val === 'object' && !Array.isArray(val)) {
        const subObj = val as Record<string, unknown>;
        if (Object.keys(subObj).some((k) => requestedSet.has(k))) {
          mapObj = subObj;
          break;
        }
      }
    }
  }

  const summaries: Record<string, string> = {};

  for (const [id, value] of Object.entries(mapObj)) {
    if (requestedSet.has(id) && typeof value === 'string') {
      const trimmed = value.trim();
      if (trimmed.length > 0) {
        const words = trimmed.split(/\s+/);
        summaries[id] = words.length > 5 ? words.slice(0, 5).join(' ') : trimmed;
      }
    }
  }

  if (Object.keys(summaries).length === 0) {
    throw new Error('AI summary response contained no valid node ID summary mappings.');
  }

  return summaries;
}

export function truncateExplanation(text: string, maxWords = 100): string {
  if (!text) return '';
  const trimmed = text.trim();
  const words = trimmed.split(/\s+/);
  if (words.length <= maxWords) return trimmed;

  const sentenceRegex = /[^.!?]+[.!?]+(\s+|$)/g;
  const matches = [...trimmed.matchAll(sentenceRegex)];

  let result = '';
  let wordCount = 0;

  for (const match of matches) {
    const sentence = match[0];
    const sentenceWords = sentence.trim().split(/\s+/).length;
    if (wordCount + sentenceWords > maxWords) {
      break;
    }
    result += sentence;
    wordCount += sentenceWords;
  }

  if (!result.trim()) {
    result = words.slice(0, maxWords).join(' ') + '...';
  } else {
    result = result.trim();
  }

  const fenceCount = (result.match(/```/g) || []).length;
  if (fenceCount % 2 !== 0) {
    result += '\n```';
  }

  return result;
}
