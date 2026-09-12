import { describe, it, expect } from 'vitest';
import { serializeGraphForAI, extractNodeContext } from './aiContext';
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
  ];

  const edges: Edge[] = [
    { id: 'e1', source: '%arg0', target: 'op_1' },
  ];

  it('serializes graph context using rawLabel rather than display label', () => {
    const serialized = serializeGraphForAI(nodes, edges);
    expect(serialized.nodes).toHaveLength(2);
    expect(serialized.nodes[0].label).toBe('%arg0');
    expect(serialized.nodes[1].label).toBe('%0 = linalg.generic');
    expect(serialized.edges).toHaveLength(1);
    expect(serialized.edges[0]).toEqual({ source: '%arg0', target: 'op_1' });
  });

  it('extracts node context cleanly', () => {
    const ctx = extractNodeContext('op_1', nodes, edges);
    expect(ctx).not.toBeNull();
    expect(ctx?.label).toBe('%0 = linalg.generic');
    expect(ctx?.incomingEdges).toEqual([{ source: '%arg0', target: 'op_1' }]);
  });
});
