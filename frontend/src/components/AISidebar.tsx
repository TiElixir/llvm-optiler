import { Sparkles, MessageSquare, X, HelpCircle } from 'lucide-react';

interface AISidebarProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: 'explain' | 'chat';
  onTabChange: (tab: 'explain' | 'chat') => void;
  explainedNodeId: string | null;
  explainedNodeLabel?: string;
  explanation?: string | null;
  explanationLoading?: boolean;
  explanationError?: string | null;
}

export function AISidebar({
  isOpen,
  onClose,
  activeTab,
  onTabChange,
  explainedNodeId,
  explainedNodeLabel,
}: AISidebarProps) {
  if (!isOpen) return null;

  return (
    <div className="absolute inset-0 z-30 flex flex-col bg-[var(--surface)] border-r border-[var(--border)] h-full w-[320px] font-sans shadow-lg">
      {/* Header with Tabs & Close */}
      <div className="p-3 border-b border-[var(--border)] flex items-center justify-between bg-[var(--subtle-surface)]">
        <div className="flex items-center gap-1.5 bg-[var(--control-bg)] p-1 rounded-lg border border-[var(--control-border)]">
          <button
            type="button"
            onClick={() => onTabChange('explain')}
            className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'explain'
                ? 'bg-[var(--surface)] text-[var(--text)] shadow-sm'
                : 'text-[var(--muted)] hover:text-[var(--text)]'
            }`}
          >
            <HelpCircle className="w-3.5 h-3.5" />
            Explain
          </button>
          <button
            type="button"
            onClick={() => onTabChange('chat')}
            className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
              activeTab === 'chat'
                ? 'bg-[var(--surface)] text-[var(--text)] shadow-sm'
                : 'text-[var(--muted)] hover:text-[var(--text)]'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            Chat
          </button>
        </div>

        <button
          type="button"
          onClick={onClose}
          aria-label="Close AI Sidebar"
          title="Close AI Sidebar"
          className="p-1.5 rounded-md text-[var(--muted)] hover:bg-[var(--control-hover)] hover:text-[var(--text)] transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Main Tab Body */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'explain' && (
          <div className="space-y-4">
            <div>
              <h2 className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider mb-2">
                Node Analysis
              </h2>
              {explainedNodeId ? (
                <div className="p-3 rounded-lg border border-[var(--border)] bg-[var(--subtle-surface)] space-y-1.5">
                  <div className="text-[11px] text-[var(--muted)] font-mono">
                    ID: {explainedNodeId}
                  </div>
                  {explainedNodeLabel && (
                    <div className="text-xs font-mono font-medium text-[var(--text)] truncate">
                      {explainedNodeLabel}
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-xs text-[var(--muted)] italic">
                  Right-click any node in the graph and select <span className="font-semibold text-[var(--text)] font-sans">Explain</span> to inspect it.
                </p>
              )}
            </div>

            {/* Explanation Content Structural Container */}
            <div className="rounded-lg border border-[var(--border)] p-4 bg-[var(--surface)] space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-[var(--muted-strong)]">
                <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                <span>AI Explanation</span>
              </div>
              <p className="text-xs text-[var(--muted)] leading-relaxed">
                AI explanation engine will be connected in the next round. Configuration and UI shell are ready.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="flex flex-col h-full items-center justify-center text-center p-4 text-[var(--muted)]">
            <MessageSquare className="w-8 h-8 mb-2 text-[var(--muted)]" />
            <h3 className="text-xs font-medium text-[var(--text)] mb-1">Context-Aware Chat</h3>
            <p className="text-xs text-[var(--muted)]">
              Chat interface and context-aware graph queries will be connected in Round 2.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
