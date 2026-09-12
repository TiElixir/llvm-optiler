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
    throw new Error('AI response could not be parsed as JSON.');
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('AI response was not a valid JSON object mapping node IDs to summaries.');
  }

  const resultObj = parsed as Record<string, unknown>;
  const summaries: Record<string, string> = {};
  const requestedSet = new Set(requestedNodeIds);

  for (const [id, value] of Object.entries(resultObj)) {
    if (requestedSet.has(id) && typeof value === 'string' && value.trim().length > 0) {
      summaries[id] = value.trim();
    }
  }

  if (Object.keys(summaries).length === 0) {
    throw new Error('AI summary response contained no valid node ID summary mappings.');
  }

  return summaries;
}
