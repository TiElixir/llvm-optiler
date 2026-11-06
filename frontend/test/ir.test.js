import test from "node:test";
import assert from "node:assert/strict";
import { flavourForStage, highlightLine, renderCode, tokenizeLine } from "../js/ir.js";

const toks = (f, s) => tokenizeLine(f, s).filter(([, c]) => c);
const has = (t, text, cls) => assert.ok(t.some(([a, b]) => a === text && b === cls), `${text} -> ${cls} in ${JSON.stringify(t)}`);

test("mlir: ops, SSA values, types, attributes", () => {
  const t = toks("mlir", "    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%1 : tensor<256x256xf32>)");
  has(t, "%2", "ssa"); has(t, "linalg.matmul", "op"); has(t, "tensor<256x256xf32>", "type");
});

test("mlir: leading dialect.op is the op, trailing loc() is a comment", () => {
  const t = toks("mlir", "    affine.store %cst, %alloc[%arg3, %arg4] : memref<256x256xf32> loc(#loc1)");
  assert.deepEqual(t[0], ["affine.store", "op"]);
  has(t, "memref<256x256xf32>", "type");
  assert.equal(t[t.length - 1][1], "cmt");
});

test("mlir: labels, symbols, comments", () => {
  assert.deepEqual(toks("mlir", "  ^bb2:  // pred: ^bb1")[0], ["^bb2", "ssa"]);
  const t = toks("mlir", "  func.func @forward(%arg0: !llvm.ptr)");
  has(t, "@forward", "attr"); has(t, "!llvm.ptr", "type");
});

test("llvm: instructions, values, types, metadata, vectors", () => {
  const t = toks("llvm", "  %39 = call ptr @_mlir_memref_to_llvm_alloc(i64 262208), !dbg !26");
  has(t, "%39", "ssa"); has(t, "call", "op"); has(t, "ptr", "type"); has(t, "@_mlir_memref_to_llvm_alloc", "attr"); has(t, "!dbg", "attr");
  has(toks("llvm", "  %5 = fadd <8 x float> %3, %4"), "<8 x float>", "type");
  assert.equal(toks("llvm", "; ModuleID = 'x'")[0][1], "cmt");
});

test("asm: addresses, mnemonics, registers, source markers", () => {
  const t = toks("asm", "    11b7:      \tvmulss\t-0xc00(%r10), %xmm2, %xmm2");
  has(t, "11b7:", "ssa"); has(t, "vmulss", "op"); has(t, "%xmm2", "type");
  assert.equal(toks("asm", "; /tmp/anchor/llvm_dialect.mlir:96")[0][1], "cmt");
});

test("highlighted HTML is escaped and renderCode numbers lines", () => {
  assert.ok(highlightLine("mlir", "%x = a.b : tensor<2xf32>").includes("&lt;"));
  assert.ok(!highlightLine("llvm", "<script>").includes("<script>"));
  const c = renderCode("mlir", "a.b\nc.d\n");
  assert.ok(c.gutter.includes(">1<") && c.gutter.includes(">2<"));
  assert.ok(renderCode("mlir", "a\nb", [2]).html.includes('class="l hl" data-n="2"'));
  assert.ok(c.html.includes('data-n="2"'));
});

test("flavourForStage", () => {
  assert.equal(flavourForStage("linalg_tensor"), "mlir");
  assert.equal(flavourForStage("llvm_dialect"), "mlir");
  assert.equal(flavourForStage("llvm_ir"), "llvm");
  assert.equal(flavourForStage("asm"), "asm");
});
