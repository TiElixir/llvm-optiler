import type { AIConfig } from './aiConfig';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatCompletionRequest {
  config: AIConfig;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

export interface ChatCompletionResponse {
  id?: string;
  choices: Array<{
    message: {
      role: string;
      content: string;
    };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

export async function createChatCompletion(
  req: ChatCompletionRequest,
  signal?: AbortSignal
): Promise<string> {
  const { config, messages, temperature = 0.2, maxTokens } = req;
  const baseUrl = config.baseUrl.replace(/\/+$/, '');
  const endpoint = `${baseUrl}/chat/completions`;

  const payload: Record<string, unknown> = {
    model: config.model,
    messages,
    temperature,
  };

  if (typeof maxTokens === 'number') {
    payload.max_tokens = maxTokens;
  }

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${config.apiKey.trim()}`,
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    let errorDetail = '';
    try {
      const errorJson = await response.json();
      errorDetail = errorJson?.error?.message || errorJson?.detail || JSON.stringify(errorJson);
    } catch {
      errorDetail = await response.text();
    }
    throw new Error(`AI API request failed (${response.status}): ${errorDetail || response.statusText}`);
  }

  const data: ChatCompletionResponse = await response.json();
  const text = data.choices?.[0]?.message?.content;

  if (typeof text !== 'string') {
    throw new Error('AI API returned an invalid or empty choice response.');
  }

  return text;
}
