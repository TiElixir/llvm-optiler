import { useState, useEffect, useRef } from 'react';
import type { AIConfig } from '../ai/aiConfig';
import { DEFAULT_AI_CONFIG, MODEL_PRESETS } from '../ai/aiConfig';
import { Settings, X } from 'lucide-react';

interface AIProviderSettingsProps {
  isOpen: boolean;
  onClose: () => void;
  config: AIConfig;
  onSave: (newConfig: AIConfig) => void;
}

export function AIProviderSettings({ isOpen, onClose, config, onSave }: AIProviderSettingsProps) {
  const [baseUrl, setBaseUrl] = useState(config.baseUrl);
  const [apiKey, setApiKey] = useState(config.apiKey);
  const [modelPreset, setModelPreset] = useState<string>(() => {
    const matched = MODEL_PRESETS.find((p) => p.value === config.model);
    return matched ? matched.value : 'custom';
  });
  const [customModel, setCustomModel] = useState(
    MODEL_PRESETS.some((p) => p.value === config.model) ? '' : config.model
  );

  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
    }
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    const finalModel = modelPreset === 'custom' ? customModel.trim() : modelPreset;
    onSave({
      baseUrl: baseUrl.trim() || DEFAULT_AI_CONFIG.baseUrl,
      apiKey: apiKey,
      model: finalModel || DEFAULT_AI_CONFIG.model,
    });
    onClose();
  };

  return (
    <div
      ref={popoverRef}
      className="absolute right-0 top-11 z-50 w-80 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 shadow-xl text-xs text-[var(--text)] font-sans"
    >
      <div className="flex items-center justify-between border-b border-[var(--border)] pb-2 mb-3">
        <h3 className="font-semibold text-xs flex items-center gap-1.5 text-[var(--text)]">
          <Settings className="w-3.5 h-3.5 text-[var(--muted)]" />
          AI Provider Settings
        </h3>
        <button
          type="button"
          onClick={onClose}
          className="p-1 rounded text-[var(--muted)] hover:bg-[var(--control-hover)] hover:text-[var(--text)] transition-colors"
          title="Close"
          aria-label="Close"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <form onSubmit={handleSave} className="space-y-3">
        <div>
          <label className="block font-medium text-[11px] text-[var(--muted-strong)] mb-1">
            Provider Base URL
          </label>
          <input
            type="text"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.openai.com/v1"
            className="w-full rounded-md border border-[var(--control-border)] bg-[var(--control-bg)] px-2.5 py-1.5 text-xs text-[var(--text)] focus:border-[var(--button-bg)] focus:outline-none font-mono"
          />
        </div>

        <div>
          <label className="block font-medium text-[11px] text-[var(--muted-strong)] mb-1">
            API Key
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
            className="w-full rounded-md border border-[var(--control-border)] bg-[var(--control-bg)] px-2.5 py-1.5 text-xs text-[var(--text)] focus:border-[var(--button-bg)] focus:outline-none font-mono"
          />
        </div>

        <div>
          <label className="block font-medium text-[11px] text-[var(--muted-strong)] mb-1">
            Model
          </label>
          <select
            value={modelPreset}
            onChange={(e) => setModelPreset(e.target.value)}
            className="w-full rounded-md border border-[var(--control-border)] bg-[var(--control-bg)] px-2 py-1.5 text-xs text-[var(--text)] focus:border-[var(--button-bg)] focus:outline-none font-sans"
          >
            {MODEL_PRESETS.map((preset) => (
              <option key={preset.value} value={preset.value}>
                {preset.label}
              </option>
            ))}
          </select>

          {modelPreset === 'custom' && (
            <input
              type="text"
              value={customModel}
              onChange={(e) => setCustomModel(e.target.value)}
              placeholder="e.g. gpt-4o-mini"
              className="mt-1.5 w-full rounded-md border border-[var(--control-border)] bg-[var(--control-bg)] px-2.5 py-1.5 text-xs text-[var(--text)] focus:border-[var(--button-bg)] focus:outline-none font-mono"
            />
          )}
        </div>

        <div className="flex items-center justify-end gap-2 pt-2 border-t border-[var(--border)]">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-md border border-[var(--control-border)] bg-[var(--control-bg)] hover:bg-[var(--control-hover)] text-[var(--muted-strong)] font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="px-3 py-1.5 rounded-md bg-[var(--button-bg)] hover:bg-[var(--button-hover)] text-[var(--button-text)] font-medium transition-colors"
          >
            Save
          </button>
        </div>
      </form>
    </div>
  );
}
