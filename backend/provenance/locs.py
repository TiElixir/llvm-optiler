"""Parse MLIR location text and resolve it to lines in the user's model source.

The Python bindings do not expose location kinds, but ``str(op.location)`` is stable:

    loc(unknown)
    loc("source/model.py":7:0)
    loc("name")  /  loc("name"(<loc>))
    loc(callsite(<loc> at <loc>))        callee at caller; the caller is the user's frame
    loc(fused[<loc>, <loc>])  /  loc(fused<"tag">[<loc>, ...])

Root identity is a line in ``source/model.py`` (PLAN.md section 7).
"""
from __future__ import annotations

import re

SOURCE_FILE = "source/model.py"

_TOKEN = re.compile(r'''
    (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<num>-?\d+)
  | (?P<word>[A-Za-z_][\w.$-]*)
  | (?P<punct>[()\[\]<>,:])
  | (?P<ws>\s+)
''', re.X)


class _Parser:
    def __init__(self, text: str):
        self.toks: list[tuple[str, str]] = []
        for m in _TOKEN.finditer(text):
            if m.lastgroup != "ws":
                self.toks.append((m.lastgroup or "", m.group()))
        self.i = 0

    def peek(self) -> tuple[str, str]:
        return self.toks[self.i] if self.i < len(self.toks) else ("eof", "")

    def take(self, value: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if value is not None and tok[1] != value:
            raise ValueError(f"expected {value!r}, got {tok!r}")
        self.i += 1
        return tok

    # returns a nested structure: ("file", path, line) | ("unknown",) | ("name", s, child|None)
    #                              | ("callsite", callee, caller) | ("fused", [children])
    def body(self):
        kind, val = self.peek()
        if kind == "str":
            self.take()
            s = _unquote(val)
            if self.peek()[1] == ":":  # "file":line:col [to ...]
                self.take(":")
                line = int(self.take()[1])
                if self.peek()[1] == ":":
                    self.take(":")
                    self.take()  # column
                if self.peek() == ("word", "to"):  # range: to [file:]line:col
                    self.take()
                    while self.peek()[0] in ("num", "str") or self.peek()[1] == ":":
                        self.take()
                return ("file", s, line)
            if self.peek()[1] == "(":
                self.take("(")
                child = self.body()
                self.take(")")
                return ("name", s, child)
            return ("name", s, None)
        if kind == "word" and val == "unknown":
            self.take()
            return ("unknown",)
        if kind == "word" and val == "callsite":
            self.take()
            self.take("(")
            callee = self.body()
            self.take("at") if self.peek()[1] == "at" else None
            caller = self.body()
            self.take(")")
            return ("callsite", callee, caller)
        if kind == "word" and val == "fused":
            self.take()
            if self.peek()[1] == "<":  # fused<"tag">
                depth = 0
                while True:
                    t = self.take()[1]
                    depth += (t == "<") - (t == ">")
                    if depth == 0:
                        break
            self.take("[")
            kids = [self.body()]
            while self.peek()[1] == ",":
                self.take(",")
                kids.append(self.body())
            self.take("]")
            return ("fused", kids)
        raise ValueError(f"unparsable location near {self.peek()!r}")


def _unquote(s: str) -> str:
    return bytes(s[1:-1], "utf-8").decode("unicode_escape")


def parse_loc(text: str):
    """Parse ``loc(...)`` text into a nested tuple; ``("unknown",)`` if it cannot be parsed."""
    t = text.strip()
    if t.startswith("loc(") and t.endswith(")"):
        t = t[4:-1]
    try:
        p = _Parser(t)
        node = p.body()
        return node
    except (ValueError, IndexError):
        return ("unknown",)


def user_lines(node, source: str = SOURCE_FILE) -> list[int]:
    """Lines of the user's source this location stands for (sorted, unique)."""
    out: set[int] = set()

    def walk(n) -> bool:
        kind = n[0]
        if kind == "file":
            if n[1] == source or n[1].endswith("/" + source):
                out.add(n[2])
                return True
            return False
        if kind == "callsite":  # the caller is the user's frame; fall back to the callee
            return walk(n[2]) or walk(n[1])
        if kind == "fused":
            hit = False
            for c in n[1]:
                hit = walk(c) or hit
            return hit
        if kind == "name":
            return walk(n[2]) if n[2] is not None else False
        return False

    walk(node)
    return sorted(out)


def lines_of(loc_text: str, source: str = SOURCE_FILE) -> list[int]:
    return user_lines(parse_loc(loc_text), source)
