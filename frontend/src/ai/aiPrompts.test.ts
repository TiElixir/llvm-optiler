import { describe, it, expect } from 'vitest';
import { buildSummarisePrompt, buildExplainPrompt } from './aiPrompts';
import type { CompactNode, NodeContext } from './types';

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
});
