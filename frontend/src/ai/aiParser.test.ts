import { describe, it, expect } from 'vitest';
import { parseSummariseResponse } from './aiParser';

describe('aiParser', () => {
  const requestedIds = ['node_1', 'node_2'];

  it('parses valid JSON response', () => {
    const raw = JSON.stringify({
      node_1: 'elementwise tensor addition',
      node_2: 'input tensor argument',
    });
    const parsed = parseSummariseResponse(raw, requestedIds);
    expect(parsed).toEqual({
      node_1: 'elementwise tensor addition',
      node_2: 'input tensor argument',
    });
  });

  it('parses Markdown code fenced JSON response', () => {
    const raw = "```json\n{\n  \"node_1\": \"adds two floats\",\n  \"node_2\": \"input parameter\"\n}\n```";
    const parsed = parseSummariseResponse(raw, requestedIds);
    expect(parsed).toEqual({
      node_1: 'adds two floats',
      node_2: 'input parameter',
    });
  });

  it('ignores unknown node IDs and retains requested IDs', () => {
    const raw = JSON.stringify({
      node_1: 'tensor operation',
      unknown_node: 'should be ignored',
    });
    const parsed = parseSummariseResponse(raw, requestedIds);
    expect(parsed).toEqual({
      node_1: 'tensor operation',
    });
  });

  it('throws error for malformed JSON', () => {
    expect(() => parseSummariseResponse('not json', requestedIds)).toThrow(
      'AI response could not be parsed as JSON.'
    );
  });

  it('throws error for non-object JSON', () => {
    expect(() => parseSummariseResponse('["item1", "item2"]', requestedIds)).toThrow(
      'AI response was not a valid JSON object'
    );
  });

  it('throws error when no matching node summaries are returned', () => {
    const raw = JSON.stringify({ unrelated_id: 'some text' });
    expect(() => parseSummariseResponse(raw, requestedIds)).toThrow(
      'AI summary response contained no valid node ID summary mappings.'
    );
  });
});
