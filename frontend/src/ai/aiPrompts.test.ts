import { describe, it, expect } from 'vitest';
import { buildSummarisePrompt, buildExplainPrompt, buildChatPrompt } from './aiPrompts';
import type { CompactNode, NodeContext, GraphContext } from './types';
import type { ChatMessage } from './aiClient';

describe('aiPrompts', () => {
  it('builds Summarise prompt requesting 1–5 word JSON summaries for all node IDs', () => {
    const compactNodes: CompactNode[] = [
      { id: 'node_1', label: 'arith.addf', type: 'arithmetic' },
      { id: 'node_2', label: '%arg0', type: 'input' },
    ];

    const messages = buildSummarisePrompt(compactNodes);
    expect(messages).toHaveLength(2);
    expect(messages[0].role).toBe('system');
    expect(messages[0].content).toContain('1–5 words');
    expect(messages[0].content).toContain('Return JSON only');

    expect(messages[1].role).toBe('user');
    const parsedPayload = JSON.parse(messages[1].content);
    expect(parsedPayload.nodes).toEqual(compactNodes);
  });

  it('builds Explain prompt explicitly specifying 50–90 words target and 100 max constraint', () => {
    const ctx: NodeContext = {
      id: 'op_linalg',
      label: '%0 = linalg.generic',
      type: 'region container',
      rawOp: 'linalg.generic',
      incomingEdges: [{ source: '%arg0', target: 'op_linalg' }],
      outgoingEdges: [{ source: 'op_linalg', target: 'ret_1' }],
    };

    const messages = buildExplainPrompt(ctx, 'func.func @main(...)', 'import torch');
    expect(messages).toHaveLength(2);
    expect(messages[0].content).toContain('Target length: 50–90 words');
    expect(messages[0].content).toContain('Hard maximum: 100 words');
    expect(messages[0].content).toContain('Single compact paragraph preferred');

    const userContent = messages[1].content;
    expect(userContent).toContain('Node ID: op_linalg');
    expect(userContent).toContain('Node Label: %0 = linalg.generic');
    expect(userContent).toContain('Node Type: region container');
    expect(userContent).toContain('Direct Upstream Operands:');
    expect(userContent).toContain('- %arg0 -> op_linalg');
    expect(userContent).toContain('Direct Downstream Consumers:');
    expect(userContent).toContain('- op_linalg -> ret_1');
    expect(userContent).toContain('func.func @main(...)');
    expect(userContent).toContain('import torch');
  });

  it('builds Chat prompt with MLIR, PyTorch source, node IDs/labels/types/edges, and history', () => {
    const graphContext: GraphContext = {
      nodes: [
        { id: '%arg0', label: '%arg0', type: 'function argument' },
        { id: 'op_matmul', label: 'torch.matmul', type: 'operation (torch.matmul)' },
      ],
      edges: [{ source: '%arg0', target: 'op_matmul' }],
    };

    const history: ChatMessage[] = [
      { role: 'user', content: 'Hello assistant' },
      { role: 'assistant', content: 'Hello! How can I help with your graph?' },
    ];

    const messages = buildChatPrompt(
      'module { func.func @forward(%arg0: tensor<4x4xf32>) }',
      'import torch\nclass Model(nn.Module): pass',
      graphContext,
      history,
      'What does this graph represent?'
    );

    // 1 system message + 2 history messages + 1 user message = 4 messages total
    expect(messages).toHaveLength(4);

    const systemMsg = messages[0];
    expect(systemMsg.role).toBe('system');
    expect(systemMsg.content).toContain('func.func @forward');
    expect(systemMsg.content).toContain('import torch');
    expect(systemMsg.content).toContain('%arg0');
    expect(systemMsg.content).toContain('op_matmul');
    expect(systemMsg.content).toContain('torch.matmul');
    expect(systemMsg.content).toContain('"source": "%arg0"');
    expect(systemMsg.content).toContain('"target": "op_matmul"');

    // Check history preserved in sequence
    expect(messages[1]).toEqual(history[0]);
    expect(messages[2]).toEqual(history[1]);

    // Check latest user message appended
    expect(messages[3]).toEqual({
      role: 'user',
      content: 'What does this graph represent?',
    });
  });

  it('truncates excessively long MLIR without truncating Python source or graph topology', () => {
    const longMlir = 'a'.repeat(40000);
    const pythonCode = 'def forward(x): return x * 2';
    const graphContext: GraphContext = {
      nodes: [{ id: 'n1', label: 'op1', type: 'operation' }],
      edges: [],
    };

    const messages = buildChatPrompt(longMlir, pythonCode, graphContext, [], 'Explain');
    const systemContent = messages[0].content;

    expect(systemContent).toContain('... [MLIR Snapshot truncated due to length]');
    expect(systemContent).toContain(pythonCode);
    expect(systemContent).toContain('n1');
  });
});
