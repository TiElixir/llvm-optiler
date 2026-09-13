import { useState, useRef, useEffect } from 'react';
import { Sparkles, MessageSquare, X, HelpCircle, Loader2, AlertCircle, Trash2, Send, User, Bot } from 'lucide-react';
import { renderMarkdown } from '../ai/markdownRenderer';

export interface ChatMessageItem {
  role: 'user' | 'assistant';
  content: string;
}

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
  chatMessages: ChatMessageItem[];
  chatLoading: boolean;
  chatError: string | null;
  onSendChatMessage: (message: string) => void;
  onClearChat: () => void;
}

export function AISidebar({
  isOpen,
  onClose,
  activeTab,
  onTabChange,
  explainedNodeId,
  explainedNodeLabel,
  explanation,
  explanationLoading,
  explanationError,
  chatMessages,
  chatLoading,
  chatError,
  onSendChatMessage,
  onClearChat,
}: AISidebarProps) {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeTab === 'chat') {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, chatLoading, activeTab]);

  if (!isOpen) return null;

  const handleSend = () => {
    const trimmed = inputText.trim();
    if (!trimmed || chatLoading) return;
    onSendChatMessage(trimmed);
    setInputText('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

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
      {activeTab === 'explain' && (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
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

          {/* Explanation Content */}
          {explanationLoading ? (
            <div className="rounded-lg border border-[var(--border)] p-4 bg-[var(--surface)] flex flex-col items-center justify-center text-center gap-2">
              <Loader2 className="w-5 h-5 text-[var(--button-bg)] animate-spin" />
              <span className="text-xs text-[var(--muted)]">Generating AI explanation...</span>
            </div>
          ) : explanationError ? (
            <div className="rounded-lg border border-[var(--graph-output-border)] bg-[var(--graph-output-bg)] p-3 text-[var(--graph-output-zone-text)] text-xs space-y-1">
              <div className="flex items-center gap-1.5 font-medium">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>Explanation Error</span>
              </div>
              <p className="text-[11px] opacity-90 leading-relaxed break-words">{explanationError}</p>
            </div>
          ) : explanation ? (
            <div className="rounded-lg border border-[var(--border)] p-4 bg-[var(--surface)] space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-[var(--muted-strong)] border-b border-[var(--border)] pb-2">
                <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                <span>AI Explanation</span>
              </div>
              <div
                className="ai-explanation-markdown"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(explanation) }}
              />
            </div>
          ) : explainedNodeId ? (
            <div className="rounded-lg border border-[var(--border)] p-4 bg-[var(--surface)] text-xs text-[var(--muted)]">
              Preparing explanation request...
            </div>
          ) : null}
        </div>
      )}

      {activeTab === 'chat' && (
        <div className="flex-1 flex flex-col min-h-0">
          {/* Chat Sub-Header */}
          <div className="px-4 py-2 border-b border-[var(--border)] flex items-center justify-between bg-[var(--subtle-surface)]">
            <span className="text-xs font-medium text-[var(--muted-strong)]">Graph Assistant</span>
            {chatMessages.length > 0 && (
              <button
                type="button"
                onClick={onClearChat}
                title="Clear Conversation"
                className="flex items-center gap-1 text-[11px] font-medium text-[var(--muted)] hover:text-[var(--text)] transition-colors"
              >
                <Trash2 className="w-3 h-3" />
                Clear
              </button>
            )}
          </div>

          {/* Scrollable Conversation Area */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
            {chatMessages.length === 0 ? (
              <div className="flex flex-col items-center justify-center text-center p-4 text-[var(--muted)] my-auto space-y-3">
                <MessageSquare className="w-7 h-7 text-[var(--muted)] opacity-60" />
                <div>
                  <h3 className="text-xs font-semibold text-[var(--text)] mb-1">Compiler Chat Agent</h3>
                  <p className="text-[11px] text-[var(--muted)] leading-relaxed">
                    Ask context-aware questions about the current graph topology, MLIR passes, or PyTorch code.
                  </p>
                </div>
                <div className="space-y-1.5 w-full pt-2">
                  {[
                    'What does this graph represent?',
                    'Which operation performs matrix multiplication?',
                    'What happens after the matmul operation?',
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => onSendChatMessage(suggestion)}
                      className="w-full text-left p-2 rounded-md border border-[var(--border)] bg-[var(--subtle-surface)] hover:bg-[var(--control-hover)] text-[11px] text-[var(--muted-strong)] transition-colors leading-tight"
                    >
                      "{suggestion}"
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              chatMessages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex flex-col gap-1 ${
                    msg.role === 'user' ? 'items-end' : 'items-start'
                  }`}
                >
                  <div className="flex items-center gap-1 text-[10px] font-medium text-[var(--muted)] px-1">
                    {msg.role === 'user' ? (
                      <>
                        <span>You</span>
                        <User className="w-3 h-3 text-[var(--muted-strong)]" />
                      </>
                    ) : (
                      <>
                        <Bot className="w-3 h-3 text-amber-500" />
                        <span>Assistant</span>
                      </>
                    )}
                  </div>
                  <div
                    className={`max-w-[90%] p-2.5 rounded-lg text-xs ${
                      msg.role === 'user'
                        ? 'bg-[var(--button-bg)] text-[var(--button-text)] rounded-br-none font-sans'
                        : 'bg-[var(--subtle-surface)] border border-[var(--border)] text-[var(--text)] rounded-bl-none'
                    }`}
                  >
                    {msg.role === 'user' ? (
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    ) : (
                      <div
                        className="ai-explanation-markdown"
                        dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                      />
                    )}
                  </div>
                </div>
              ))
            )}

            {chatLoading && (
              <div className="flex items-center gap-2 p-2.5 rounded-lg border border-[var(--border)] bg-[var(--subtle-surface)] text-xs text-[var(--muted)]">
                <Loader2 className="w-3.5 h-3.5 text-amber-500 animate-spin" />
                <span>Assistant is thinking...</span>
              </div>
            )}

            {chatError && (
              <div className="p-3 rounded-lg border border-[var(--graph-output-border)] bg-[var(--graph-output-bg)] text-[var(--graph-output-zone-text)] text-xs space-y-1">
                <div className="flex items-center gap-1.5 font-medium">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>Chat Error</span>
                </div>
                <p className="text-[11px] opacity-90 leading-relaxed break-words">{chatError}</p>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Text Input Area */}
          <div className="p-3 border-t border-[var(--border)] bg-[var(--subtle-surface)] space-y-2">
            <div className="relative flex items-center">
              <textarea
                rows={2}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about the graph or MLIR... (Enter to send)"
                className="w-full rounded-md border border-[var(--control-border)] bg-[var(--surface)] p-2 pr-9 text-xs text-[var(--text)] placeholder:text-[var(--muted)] focus:border-[var(--button-bg)] focus:outline-none resize-none font-sans"
              />
              <button
                type="button"
                onClick={handleSend}
                disabled={!inputText.trim() || chatLoading}
                title="Send Message"
                aria-label="Send Message"
                className="absolute right-2 bottom-2.5 p-1 rounded bg-[var(--button-bg)] text-[var(--button-text)] hover:bg-[var(--button-hover)] transition-colors disabled:opacity-40"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="flex justify-between text-[10px] text-[var(--muted)] px-0.5">
              <span>Shift+Enter for newline</span>
              {chatMessages.length > 0 && (
                <button
                  type="button"
                  onClick={onClearChat}
                  className="hover:underline cursor-pointer"
                >
                  Clear Chat
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
