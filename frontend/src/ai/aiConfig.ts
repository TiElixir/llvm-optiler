export interface AIConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

export const AI_CONFIG_KEY = 'xcomp.ai.config';

export const DEFAULT_AI_CONFIG: AIConfig = {
  baseUrl: 'https://api.openai.com/v1',
  apiKey: '',
  model: 'gpt-4o-mini',
};

export const MODEL_PRESETS = [
  { label: 'gpt-4o-mini', value: 'gpt-4o-mini' },
  { label: 'gpt-4o', value: 'gpt-4o' },
  { label: 'claude-3-5-haiku', value: 'claude-3-5-haiku' },
  { label: 'Custom...', value: 'custom' },
] as const;

function getStorage(): Storage | null {
  if (typeof window !== 'undefined' && window.localStorage) return window.localStorage;
  if (typeof localStorage !== 'undefined') return localStorage;
  return null;
}

export function loadAIConfig(): AIConfig {
  const storage = getStorage();
  if (!storage) return { ...DEFAULT_AI_CONFIG };
  try {
    const raw = storage.getItem(AI_CONFIG_KEY);
    if (!raw) return { ...DEFAULT_AI_CONFIG };
    const parsed = JSON.parse(raw);
    return {
      baseUrl: typeof parsed.baseUrl === 'string' && parsed.baseUrl.trim() ? parsed.baseUrl.trim() : DEFAULT_AI_CONFIG.baseUrl,
      apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey : DEFAULT_AI_CONFIG.apiKey,
      model: typeof parsed.model === 'string' && parsed.model.trim() ? parsed.model.trim() : DEFAULT_AI_CONFIG.model,
    };
  } catch {
    return { ...DEFAULT_AI_CONFIG };
  }
}

export function saveAIConfig(config: AIConfig): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    const cleaned: AIConfig = {
      baseUrl: config.baseUrl.trim().replace(/\/+$/, ''),
      apiKey: config.apiKey,
      model: config.model.trim(),
    };
    storage.setItem(AI_CONFIG_KEY, JSON.stringify(cleaned));
  } catch (err) {
    console.error('Failed to save AI configuration to localStorage:', err);
  }
}

export function clearAIConfig(): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    storage.removeItem(AI_CONFIG_KEY);
  } catch (err) {
    console.error('Failed to clear AI configuration from localStorage:', err);
  }
}

export function isAIConfigValid(config: AIConfig): boolean {
  return Boolean(config.baseUrl && config.baseUrl.trim() && config.apiKey && config.apiKey.trim() && config.model && config.model.trim());
}
