import { describe, it, expect, beforeEach } from 'vitest';
import { loadAIConfig, saveAIConfig, clearAIConfig, isAIConfigValid, DEFAULT_AI_CONFIG, AI_CONFIG_KEY } from './aiConfig';

const mockStorage = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => { store[key] = value.toString(); },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
  };
})();

Object.defineProperty(globalThis, 'localStorage', {
  value: mockStorage,
  writable: true,
});

describe('aiConfig', () => {
  beforeEach(() => {
    mockStorage.clear();
  });

  it('loads default config when nothing stored', () => {
    const config = loadAIConfig();
    expect(config).toEqual(DEFAULT_AI_CONFIG);
  });

  it('saves and loads configuration in xcomp.ai.config', () => {
    const customConfig = {
      baseUrl: 'https://custom-endpoint.example.com/v1/',
      apiKey: 'test-key-123',
      model: 'gpt-4o',
    };
    saveAIConfig(customConfig);

    const loaded = loadAIConfig();
    expect(loaded.baseUrl).toBe('https://custom-endpoint.example.com/v1');
    expect(loaded.apiKey).toBe('test-key-123');
    expect(loaded.model).toBe('gpt-4o');

    expect(mockStorage.getItem(AI_CONFIG_KEY)).toBeDefined();
  });

  it('clears configuration', () => {
    saveAIConfig({ baseUrl: 'http://localhost:8000', apiKey: 'key', model: 'custom' });
    clearAIConfig();
    expect(loadAIConfig()).toEqual(DEFAULT_AI_CONFIG);
  });

  it('validates config correctly', () => {
    expect(isAIConfigValid({ baseUrl: '', apiKey: '123', model: 'gpt-4o' })).toBe(false);
    expect(isAIConfigValid({ baseUrl: 'http://a', apiKey: '', model: 'gpt-4o' })).toBe(false);
    expect(isAIConfigValid({ baseUrl: 'http://a', apiKey: '123', model: '' })).toBe(false);
    expect(isAIConfigValid({ baseUrl: 'http://a', apiKey: '123', model: 'gpt-4o' })).toBe(true);
  });
});
