import type { Node, Edge } from '@xyflow/react';
import type { GraphContext, CompactNode, CompactEdge, NodeContext } from './types';

export function classifyNodeType(node: Node): string {
  if (node.data?.isBlockArg) return 'block argument';
  if (node.data?.isFuncArg || node.type === 'input') return 'function argument';
  if (node.data?.isYield) return 'yield';
  if (node.type === 'output' || String(node.data?.rawLabel || '').includes('return')) return 'return';
  if (node.type === 'arithmetic' || Boolean(node.data?.arithOp)) return 'arithmetic operation';
  if (node.type === 'regionOp') return 'region container';
  if (typeof node.data?.rawOp === 'string') return `operation (${node.data.rawOp})`;
  return node.type || 'operation';
}

export function serializeGraphForAI(nodes: Node[], edges: Edge[]): GraphContext {
  const compactNodes: CompactNode[] = nodes
    .filter((n) => !n.hidden && !n.data?.isConstantsGroup && !n.data?.isInputsGroup && !n.data?.isOutputsGroup)
    .map((n) => {
      const rawLabel = typeof n.data?.rawLabel === 'string' ? n.data.rawLabel : typeof n.data?.label === 'string' ? n.data.label : n.id;
      return {
        id: n.id,
        label: rawLabel,
        type: classifyNodeType(n),
        ...(typeof n.data?.rawOp === 'string' ? { rawOp: n.data.rawOp } : {}),
        ...(n.parentId ? { parentId: n.parentId } : {}),
      };
    });

  const compactEdges: CompactEdge[] = edges
    .filter((e) => !e.hidden)
    .map((e) => ({
      source: e.source,
      target: e.target,
    }));

  return {
    nodes: compactNodes,
    edges: compactEdges,
  };
}

export function extractNodeContext(nodeId: string, nodes: Node[], edges: Edge[]): NodeContext | null {
  const node = nodes.find((n) => n.id === nodeId);
  if (!node) return null;

  const rawLabel = typeof node.data?.rawLabel === 'string' ? node.data.rawLabel : typeof node.data?.label === 'string' ? node.data.label : node.id;
  const type = classifyNodeType(node);
  const incomingEdges = edges
    .filter((e) => e.target === nodeId && !e.hidden)
    .map((e) => ({ source: e.source, target: e.target }));
  const outgoingEdges = edges
    .filter((e) => e.source === nodeId && !e.hidden)
    .map((e) => ({ source: e.source, target: e.target }));

  return {
    id: node.id,
    label: rawLabel,
    type,
    rawOp: typeof node.data?.rawOp === 'string' ? node.data.rawOp : undefined,
    parentId: node.parentId,
    incomingEdges,
    outgoingEdges,
  };
}
