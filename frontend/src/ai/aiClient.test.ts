import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createChatCompletion } from './aiClient';
import type { AIConfig } from './aiConfig';

describe('aiClient', () => {
  const config: AIConfig = {
    baseUrl: 'https://api.openai.com/v1/',
    apiKey: 'sk-test12345',
    model: 'gpt-4o-mini',
  };

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('strips trailing slash from baseUrl and posts to /chat/completions with Bearer token', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { role: 'assistant', content: 'Test response' } }],
      }),
    } as Response);

    const result = await createChatCompletion({
      config,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(result).toBe('Test response');
    expect(fetchSpy).toHaveBeenCalledWith(
      'https://api.openai.com/v1/chat/completions',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer sk-test12345',
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [{ role: 'user', content: 'Hello' }],
          temperature: 0.2,
        }),
      })
    );
  });

  it('surfaces 401 Unauthorized errors cleanly', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      json: async () => ({ error: { message: 'Invalid API key provided' } }),
    } as Response);

    await expect(
      createChatCompletion({
        config,
        messages: [{ role: 'user', content: 'Hi' }],
      })
    ).rejects.toThrow('AI API request failed (401): Invalid API key provided');
  });

  it('surfaces 429 Rate Limit errors cleanly', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 429,
      statusText: 'Too Many Requests',
      json: async () => ({ error: { message: 'Rate limit exceeded' } }),
    } as Response);

    await expect(
      createChatCompletion({
        config,
        messages: [{ role: 'user', content: 'Hi' }],
      })
    ).rejects.toThrow('AI API request failed (429): Rate limit exceeded');
  });

  it('surfaces 500 Server errors cleanly', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      text: async () => 'Internal Error Details',
    } as Response);

    await expect(
      createChatCompletion({
        config,
        messages: [{ role: 'user', content: 'Hi' }],
      })
    ).rejects.toThrow('AI API request failed (500): Internal Error Details');
  });

  it('handles network failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));

    await expect(
      createChatCompletion({
        config,
        messages: [{ role: 'user', content: 'Hi' }],
      })
    ).rejects.toThrow('Failed to fetch');
  });

  it('throws error when choices or content is missing', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [] }),
    } as Response);

    await expect(
      createChatCompletion({
        config,
        messages: [{ role: 'user', content: 'Hi' }],
      })
    ).rejects.toThrow('AI API returned an invalid or empty choice response.');
  });
});
