import { describe, it, expect } from 'vitest';
import { serializeGraphForAI, extractNodeContext, classifyNodeType } from './aiContext';
import type { Node, Edge } from '@xyflow/react';

describe('aiContext', () => {
  const nodes: Node[] = [
    {
      id: '%arg0',
      type: 'input',
      position: { x: 0, y: 0 },
      data: { label: 'x (PyTorch)', rawLabel: '%arg0', isFuncArg: true },
    },
    {
      id: 'op_1',
      type: 'regionOp',
      position: { x: 0, y: 0 },
      data: { label: 'y = linalg.generic', rawLabel: '%0 = linalg.generic', rawOp: 'linalg.generic' },
    },
    {
      id: 'block_arg_1',
      type: 'default',
      position: { x: 0, y: 0 },
      data: { label: '%in', rawLabel: '%in', isBlockArg: true },
    },
    {
      id: 'node_no_meta',
      type: 'default',
      position: { x: 0, y: 0 },
      data: { label: 'fallback_label' }, // missing rawLabel and rawOp
    },
  ];

  const edges: Edge[] = [
    { id: 'e1', source: '%arg0', target: 'op_1' },
    { id: 'e2', source: 'op_1', target: 'block_arg_1' },
  ];

  it('classifies node types correctly', () => {
    expect(classifyNodeType(nodes[0])).toBe('function argument');
    expect(classifyNodeType(nodes[1])).toBe('region container');
    expect(classifyNodeType(nodes[2])).toBe('block argument');
  });

  it('serializes graph context using rawLabel and node classification', () => {
    const serialized = serializeGraphForAI(nodes, edges);
    expect(serialized.nodes).toHaveLength(4);
    expect(serialized.nodes[0].label).toBe('%arg0');
    expect(serialized.nodes[0].type).toBe('function argument');
    expect(serialized.nodes[1].label).toBe('%0 = linalg.generic');
    expect(serialized.nodes[1].type).toBe('region container');
    expect(serialized.edges).toHaveLength(2);
  });

  it('falls back safely to label or node ID when optional metadata like rawLabel/rawOp is missing', () => {
    const serialized = serializeGraphForAI([nodes[3]], []);
    expect(serialized.nodes).toHaveLength(1);
    expect(serialized.nodes[0].id).toBe('node_no_meta');
    expect(serialized.nodes[0].label).toBe('fallback_label');
  });

  it('extracts node context with upstream and downstream edges', () => {
    const ctx = extractNodeContext('op_1', nodes, edges);
    expect(ctx).not.toBeNull();
    expect(ctx?.label).toBe('%0 = linalg.generic');
    expect(ctx?.type).toBe('region container');
    expect(ctx?.incomingEdges).toEqual([{ source: '%arg0', target: 'op_1' }]);
    expect(ctx?.outgoingEdges).toEqual([{ source: 'op_1', target: 'block_arg_1' }]);
  });
});
