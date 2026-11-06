#!/usr/bin/env bash
# Reproduce the project-local LLVM toolchain in toolchain/llvm (gitignored).
# Nothing on the system is touched. Needs: curl, zstd, tar, sha256sum. ~1.2 GB download.
set -euo pipefail
VER=23.1.5
SHA=c3107384835f788768f614db610e87fba7077f125ca950eb02a78bab0a308065   # published by GitHub for the .tar.zst asset
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DL="$ROOT/toolchain/dl"; OUT="$ROOT/toolchain/llvm"; TMP="$ROOT/toolchain/llvm-extract"
mkdir -p "$DL" "$TMP"
cd "$DL"
curl -L --fail --retry 3 -C - -o "LLVM-$VER-Linux-X64.tar.zst" \
  "https://github.com/llvm/llvm-project/releases/download/llvmorg-$VER/LLVM-$VER-Linux-X64.tar.zst"
echo "$SHA  LLVM-$VER-Linux-X64.tar.zst" | sha256sum -c -
cd "$TMP"
zstd -dc --long=30 "$DL/LLVM-$VER-Linux-X64.tar.zst" | tar -x --wildcards '*/bin/*' '*/lib/clang/*'
E="$TMP/LLVM-$VER-Linux-X64"
mkdir -p "$OUT/bin" "$OUT/lib"
for t in clang-23 opt llc llvm-objdump llvm-symbolizer llvm-mca llvm-nm llvm-readobj; do mv "$E/bin/$t" "$OUT/bin/"; done
ln -sf clang-23 "$OUT/bin/clang"; ln -sf llvm-readobj "$OUT/bin/llvm-readelf"
rm -rf "$OUT/lib/clang"; mv "$E/lib/clang" "$OUT/lib/"
rm -rf "$TMP" "$DL"
"$OUT/bin/opt" --version | head -3
