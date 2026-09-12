import type { AIConfig } from './aiConfig';
import type { ChatMessage } from './aiClient';

export type { AIConfig, ChatMessage };

export interface CompactNode {
  id: string;
  label: string;
  type?: string;
  rawOp?: string;
  parentId?: string;
}

export interface CompactEdge {
  source: string;
  target: string;
}

export interface GraphContext {
  nodes: CompactNode[];
  edges: CompactEdge[];
}

export interface NodeContext {
  id: string;
  label: string;
  type?: string;
  rawOp?: string;
  parentId?: string;
  incomingEdges: CompactEdge[];
  outgoingEdges: CompactEdge[];
}

export type NodeSummaryMap = Record<string, string>;

export interface ExplainState {
  explainedNodeId: string | null;
  explanation: string | null;
  explanationLoading: boolean;
  explanationError: string | null;
}
