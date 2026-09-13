interface SelectionBarProps {
  selectedCount: number;
  onClear: () => void;
}

export function SelectionBar({ selectedCount, onClear }: SelectionBarProps) {
  if (selectedCount <= 1) return null;

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 px-3.5 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface-translucent)] backdrop-blur-md shadow-md text-xs text-[var(--text)] font-sans">
      <span className="font-medium text-[var(--muted-strong)]">{selectedCount} selected</span>
      <button
        type="button"
        onClick={onClear}
        className="px-2 py-1 rounded bg-[var(--control-bg)] hover:bg-[var(--control-hover)] text-[var(--text)] font-medium transition-colors"
      >
        Clear selection
      </button>
    </div>
  );
}
