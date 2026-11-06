// Syntax highlighting for MLIR, LLVM IR and objdump assembly. Four classes at most:
// op (instruction/op names), ssa (values, labels), type, attr (symbols, metadata, immediates).
import { esc } from "./format.js";

export const flavourForStage = (stage) => (stage === "llvm_ir" ? "llvm" : stage === "asm" ? "asm" : "mlir");

const y = (re) => new RegExp(re.source, "y");
const R = {
  ws: y(/\s+/),
  string: y(/"(?:[^"\\]|\\.)*"/),
  mlirSsa: y(/%[\w$.#:-]+|\^[\w$.]+/),
  mlirAttr: y(/#[\w$.]+|@[\w$.]+/),
  mlirType: y(/(?:![A-Za-z_][\w.$]*(?:<[^>]*>)?|(?:tensor|memref|vector|complex|tuple)<[^>]*>|(?:[uis]?i\d+|f(?:16|32|64)|bf16|index|none)\b)/),
  dotted: y(/[a-z_]\w*(?:\.\w+)+/),
  kw: y(/(?:func|module|return|affine|scf|cf)\b/),
  word: y(/[\w$.-]+/),
  llSsa: y(/%[\w$.-]+|\^[\w$.]+/),
  llAttr: y(/@[\w$."-]+|!\w*(?:\([^)]*\))?/),
  llType: y(/(?:i\d+|ptr|void|float|double|half|bfloat|label)\b|<\d+ x [^>]+>|\[\d+ x [^\]]+\]/),
  llOp: y(/(?:ret|br|switch|unreachable|add|sub|mul|udiv|sdiv|urem|srem|fadd|fsub|fmul|fdiv|frem|fneg|and|or|xor|shl|lshr|ashr|alloca|load|store|getelementptr|trunc|zext|sext|fptrunc|fpext|fptoui|fptosi|uitofp|sitofp|ptrtoint|inttoptr|bitcast|icmp|fcmp|phi|select|call|extractvalue|insertvalue|extractelement|insertelement|shufflevector|freeze|define|declare|private|internal|constant|global|unnamed_addr|align|nsw|nuw|inbounds|noalias|nocapture|readonly|writeonly|local_unnamed_addr|dso_local|attributes|target|source_filename|to|poison|undef|null|zeroinitializer|fast|reassoc|nnan|ninf|nsz|contract|afn|arcp)\b/),
  llLabel: y(/[\w$.-]+:/),
  addr: y(/[0-9a-f]+:/),
  mnemonic: y(/[a-z][\w.]*/),
  reg: y(/%[a-z]+\d*[a-z]*/),
  imm: y(/\$-?(?:0x[0-9a-f]+|\d+)|-?0x[0-9a-f]+|<[^>]+>/),
};

const span = (cls, text) => `<span class="t-${cls}">${esc(text)}</span>`;

function take(re, line, i) {
  re.lastIndex = i;
  const m = re.exec(line);
  return m ? m[0] : null;
}

/** Tokenize one line; returns [text, class|null] pairs (exposed for tests). */
export function tokenizeLine(flavour, line) {
  const out = [];
  const push = (text, cls) => out.push([text, cls]);
  let i = 0, seenOp = false, m;
  const n = line.length;

  if (flavour === "asm") {
    if (/^\s*;/.test(line)) return [[line, "cmt"]];
    if (/^[0-9a-f]{8,16} <[^>]+>:/.test(line)) return [[line, "attr"]];
  }
  while (i < n) {
    if ((m = take(R.ws, line, i))) { push(m, null); i += m.length; continue; }
    let tok = null, cls = null;
    if (flavour === "mlir") {
      if (line.startsWith("//", i)) { tok = line.slice(i); cls = "cmt"; }
      else if ((m = take(R.string, line, i))) { tok = m; cls = "attr"; }
      else if ((m = take(R.mlirSsa, line, i))) { tok = m; cls = "ssa"; }
      else if ((m = take(R.mlirAttr, line, i))) { tok = m; cls = "attr"; }
      else if ((m = take(R.mlirType, line, i))) { tok = m; cls = "type"; }
      else if (line.startsWith("loc(", i)) { tok = line.slice(i); cls = "cmt"; }
      else if ((m = take(R.dotted, line, i))) { tok = m; cls = seenOp ? "attr" : "op"; seenOp = true; }
      else if (line[i] === "=") { tok = "="; seenOp = false; }
      else if ((m = take(R.kw, line, i))) { tok = m; cls = "op"; }
      else if ((m = take(R.word, line, i))) { tok = m; }
    } else if (flavour === "llvm") {
      if (line[i] === ";") { tok = line.slice(i); cls = "cmt"; }
      else if ((m = take(R.string, line, i))) { tok = m; cls = "attr"; }
      else if ((m = take(R.llSsa, line, i))) { tok = m; cls = "ssa"; }
      else if ((m = take(R.llAttr, line, i))) { tok = m; cls = "attr"; }
      else if ((m = take(R.llType, line, i))) { tok = m; cls = "type"; }
      else if ((m = take(R.llOp, line, i))) { tok = m; cls = "op"; }
      else if ((m = take(R.llLabel, line, i))) { tok = m; cls = "ssa"; }
      else if ((m = take(R.word, line, i))) { tok = m; }
    } else {
      if ((m = take(R.addr, line, i))) { tok = m; cls = "ssa"; }
      else if (!seenOp && (m = take(R.mnemonic, line, i))) { tok = m; cls = "op"; seenOp = true; }
      else if ((m = take(R.reg, line, i))) { tok = m; cls = "type"; }
      else if ((m = take(R.imm, line, i))) { tok = m; cls = "attr"; }
      else if (line[i] === "#") { tok = line.slice(i); cls = "cmt"; }
    }
    if (tok == null) tok = line[i];
    push(tok, cls);
    i += tok.length;
  }
  return out;
}

export function highlightLine(flavour, line) {
  return tokenizeLine(flavour, line).map(([t, c]) => (c ? span(c, t) : esc(t))).join("");
}

/** Whole text -> { gutter, html } for the two-column code view. */
export function renderCode(flavour, text, highlighted = []) {
  const lines = text.split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  const hl = new Set(highlighted);
  return {
    gutter: lines.map((_, i) => `<span class="${hl.has(i + 1) ? "hl" : ""}">${i + 1}</span>`).join(""),
    html: lines.map((l, i) => `<span class="l${hl.has(i + 1) ? " hl" : ""}" data-n="${i + 1}">${highlightLine(flavour, l) || " "}</span>`).join(""),
  };
}
