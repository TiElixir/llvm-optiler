import type { CompactNode, NodeContext } from './types';
import type { ChatMessage } from './aiClient';

export function buildSummarisePrompt(nodes: CompactNode[]): ChatMessage[] {
  const systemPrompt = `You are an MLIR compiler education assistant.

For every supplied graph node, produce a concise
plain-English description containing 1–5 words.

Return JSON only.

The JSON object must map each supplied node ID
to exactly one summary string.

Do not omit nodes.
Do not add nodes.
Do not include explanations.
Do not use Markdown.
Do not include additional JSON fields.

Prefer compiler/tensor semantics over merely
repeating the MLIR operation name.`;

  const userPayload = JSON.stringify({ nodes }, null, 2);

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPayload },
  ];
}

export function buildExplainPrompt(
  ctx: NodeContext,
  mlirCode: string,
  pythonCode: string
): ChatMessage[] {
  const systemPrompt = `You are an MLIR and Torch-MLIR compiler education assistant.

Explain the specified graph node concisely to a software engineer.

Requirements:
- Target length: 50–90 words. Hard maximum: 100 words.
- Single compact paragraph preferred.
- Explain what the node does, what its key inputs/outputs mean, and why it matters in the current graph.
- Do NOT use generic intro phrases like "This node represents...", "This operation is...", or "In MLIR...".
- Do NOT repeat the full MLIR operation signature or full lists of upstream/downstream nodes.
- Keep Markdown formatting minimal. Avoid headings unless essential.
- Do NOT invent compiler behavior not supported by the context.`;

  const upstreamFormatted =
    ctx.incomingEdges.length > 0
      ? ctx.incomingEdges.map((e) => `- ${e.source} -> ${e.target}`).join('\n')
      : 'None';

  const downstreamFormatted =
    ctx.outgoingEdges.length > 0
      ? ctx.outgoingEdges.map((e) => `- ${e.source} -> ${e.target}`).join('\n')
      : 'None';

  const userContent = `Node ID: ${ctx.id}
Node Label: ${ctx.label}
Node Type: ${ctx.type || 'operation'}
Raw Op: ${ctx.rawOp || 'N/A'}

Direct Upstream Operands:
${upstreamFormatted}

Direct Downstream Consumers:
${downstreamFormatted}

Current MLIR Snapshot:
\`\`\`mlir
${mlirCode || '(Empty snapshot)'}
\`\`\`

Current PyTorch Source:
\`\`\`python
${pythonCode || '(Empty source)'}
\`\`\``;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userContent },
  ];
}
