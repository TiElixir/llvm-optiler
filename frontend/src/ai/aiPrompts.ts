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

Explain the supplied operation to a programmer.

Explain:
1. what the operation does,
2. why it appears in the generated MLIR,
3. how it relates to the original PyTorch operation.

Use the supplied graph relationships, MLIR and
Python source as evidence.

Do not invent compiler behavior that is not
supported by the supplied context.

Write approximately 80–150 words.
Use plain English.

Markdown is allowed for formatting and code references.`;

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
