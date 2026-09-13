import { marked } from 'marked';
import DOMPurify from 'dompurify';

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function highlightCodeTokens(code: string, lang?: string): string {
  const language = (lang || '').toLowerCase();

  if (language === 'mlir' || language === 'llvm') {
    return code
      .replace(/(%[a-zA-Z0-9_$.-]+)/g, '<span class="token-ssa">$1</span>')
      .replace(/\b(func\.func|linalg\.generic|arith\.[a-z]+|tensor\.[a-z]+|math\.[a-z]+|return|yield|ins|outs)\b/g, '<span class="token-keyword">$1</span>')
      .replace(/(\/\/.*)/g, '<span class="token-comment">$1</span>');
  }

  if (language === 'python' || language === 'py') {
    return code
      .replace(/\b(def|class|import|from|return|if|else|for|in|with|as|pass)\b/g, '<span class="token-keyword">$1</span>')
      .replace(/(#.*)/g, '<span class="token-comment">$1</span>')
      .replace(/(".*?"|'.*?')/g, '<span class="token-string">$1</span>');
  }

  return code;
}

const renderer = new marked.Renderer();
renderer.code = ({ text, lang }) => {
  const safeCode = escapeHtml(text);
  const highlighted = highlightCodeTokens(safeCode, lang);
  return `<pre><code class="language-${lang || 'text'}">${highlighted}</code></pre>`;
};

marked.setOptions({
  renderer,
  gfm: true,
  breaks: true,
});

export function renderMarkdown(content: string): string {
  if (!content) return '';
  const rawHtml = marked.parse(content) as string;
  return DOMPurify.sanitize(rawHtml, {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li', 'blockquote', 'span', 'div', 'a'
    ],
    ALLOWED_ATTR: ['class', 'href', 'title', 'target', 'rel']
  });
}
