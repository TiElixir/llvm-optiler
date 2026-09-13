import type { CompactNode, NodeContext, GraphContext } from './types';
import type { ChatMessage } from './aiClient';

export function buildSummarisePrompt(nodes: CompactNode[]): ChatMessage[] {
  const systemPrompt = `You are an MLIR compiler education assistant.

For EVERY supplied graph node in the user payload, produce a concise plain-English summary containing 1–5 words.

Return JSON only in the following shape:
{
  "summaries": {
    "<node-id>": "<1-5 word summary>"
  }
}

Requirements:
- Provide a summary mapping for EVERY node ID listed in requested_node_ids.
- Do not omit any requested node ID.
- Do not invent extra node IDs.
- Keep each summary short (1–5 words).
- Do not use Markdown or fenced code blocks outside the JSON object.
- Prefer compiler/tensor semantics over repeating the MLIR operation name.`;

  const userPayload = JSON.stringify(
    {
      requested_node_ids: nodes.map((n) => n.id),
      nodes,
    },
    null,
    2
  );

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

export function buildChatPrompt(
  mlirCode: string,
  pythonCode: string,
  graphContext: GraphContext,
  chatHistory: ChatMessage[],
  latestUserMessage: string
): ChatMessage[] {
  const MAX_MLIR_CHARS = 32000;
  const cappedMlir =
    mlirCode.length > MAX_MLIR_CHARS
      ? mlirCode.slice(0, MAX_MLIR_CHARS) + '\n... [MLIR Snapshot truncated due to length]'
      : mlirCode;

  const graphSummary = JSON.stringify(graphContext, null, 2);

  const systemMessage: ChatMessage = {
    role: 'system',
    content: `You are an expert compiler and AI optimization assistant.

You are assisting a developer inspecting the currently displayed compiler Control Flow Graph, MLIR snapshot, and PyTorch source code.

AUTHORITATIVE CONTEXT:

Current PyTorch Source:
\`\`\`python
${pythonCode || '(Empty source)'}
\`\`\`

Current MLIR Snapshot:
\`\`\`mlir
${cappedMlir || '(Empty snapshot)'}
\`\`\`

Graph Topology Serialization (Node IDs, Labels, Types, and Edges):
\`\`\`json
${graphSummary}
\`\`\`

INSTRUCTIONS:
- Use the supplied MLIR, PyTorch source, and Graph Serialization as the authoritative context.
- Do NOT invent nodes, edges, operations, or source code not supported by the context.
- If the supplied context does not contain enough information to answer a question, explicitly state so.
- Explain compiler, MLIR, and tensor concepts in clear, direct language suitable for a software engineer.
- Keep answers concise unless the user explicitly asks for details or deep analysis.
- Use Markdown for formatting and code snippets. Syntax highlighting for code blocks is supported.`,
  };

  const userMessage: ChatMessage = {
    role: 'user',
    content: latestUserMessage,
  };

  return [systemMessage, ...chatHistory, userMessage];
}
