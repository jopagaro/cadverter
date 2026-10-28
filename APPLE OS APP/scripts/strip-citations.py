#!/usr/bin/env python3
"""Strip citations from the documentation text of the Python libraries shipped in CADVERT.app.

App Review scans every file in the bundle, documentation included: a networkx docstring
citing a blog on an Iranian domain got a build rejected. The app never shows these texts,
so the shipped copy (never the source engine) drops, from docstrings and comments only:
  - numpydoc / Google-style "References" sections and reST citations (``.. [1] ...``),
  - any other line holding a URL or a DOI.
Code, string literals that code uses, and anything mentioning a licence or copyright are
left exactly as they are. Each package's long description is also dropped from its
METADATA; the headers stay, so importlib.metadata keeps working.

Usage: strip-citations.py <lib/python3.x dir>     (run with any Python >= the engine's)
"""
from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

URL = re.compile(r"(https?|ftp)://|\bdoi:\s*10\.|\bdoi\.org/", re.I)
URL_SPAN = re.compile(r"<?(https?|ftp)://[^\s<>`'\")\]]+>?", re.I)
LICENCE = re.compile(r"licen[cs]e|copyright|\(c\)|©|SPDX", re.I)
DASHES = re.compile(r"^\s*-{3,}\s*$")
SECTION = re.compile(r"^\s*[A-Z][A-Za-z ]*\s*$")
REST_CITATION = re.compile(r"^(\s*)\.\. \[[^\]]+\]")
TRIPLE = re.compile(r"^[rRuUbB]{0,2}(\"\"\"|''')")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _docstring_spans(tree: ast.AST, lines: list[str]) -> list[tuple[int, int]]:
    """(first, last) 0-based line indexes of every triple-quoted string that stands alone as a
    statement: docstrings, and the attribute docs written under fields. Python discards these."""
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        val = node.value
        if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
            continue
        first, last = val.lineno - 1, val.end_lineno - 1
        if first == last:
            continue
        head = lines[first][val.col_offset:]
        m = TRIPLE.match(head)
        # Only a single triple-quoted literal: its interior lines then hold no quotes or code.
        if not m or "".join(lines[first:last + 1]).count(m.group(1)) != 2:
            continue
        if LICENCE.search(val.value):
            continue
        spans.append((first, last))
    return spans


def _strip_docstring(lines: list[str], first: int, last: int, drop: set[int]) -> None:
    inner = list(range(first + 1, last))           # lines wholly inside the literal
    i = 0
    while i < len(inner):
        n = inner[i]
        text = lines[n]
        nxt = lines[inner[i + 1]] if i + 1 < len(inner) else ""
        if text.strip() == "References" and DASHES.match(nxt):      # numpydoc section
            j = i + 2
            while j < len(inner):
                after = lines[inner[j + 1]] if j + 1 < len(inner) else ""
                if SECTION.match(lines[inner[j]]) and DASHES.match(after):
                    break
                j += 1
            drop.update(inner[i:j]); i = j; continue
        if text.strip() == "References:":                           # Google style
            h, j = _indent(text), i + 1
            while j < len(inner) and (not lines[inner[j]].strip() or _indent(lines[inner[j]]) > h):
                j += 1
            drop.update(inner[i:j]); i = j; continue
        m = REST_CITATION.match(text)
        if m:                                                       # .. [1] entry + continuation
            h, j = len(m.group(1)), i + 1
            while j < len(inner) and lines[inner[j]].strip() and _indent(lines[inner[j]]) > h:
                j += 1
            drop.update(inner[i:j]); i = j; continue
        if URL.search(text):
            drop.add(n)
        i += 1
    for n in (first, last):                                         # lines carrying the quotes
        lines[n] = URL_SPAN.sub("", lines[n])


def _strip_comments(src: str, lines: list[str], drop: set[int]) -> None:
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return
    comments = [t for t in toks if t.type == tokenize.COMMENT]
    code_lines = {t.start[0] - 1 for t in toks if t.type not in (
        tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)}
    # Consecutive comment-only lines form a block; a block mentioning a licence is kept whole.
    blocks, cur = [], []
    for t in comments:
        n = t.start[0] - 1
        if n in code_lines:
            continue
        if cur and n != cur[-1] + 1:
            blocks.append(cur); cur = []
        cur.append(n)
    if cur:
        blocks.append(cur)
    protected = {n for b in blocks if LICENCE.search("".join(lines[n] for n in b)) for n in b}
    for t in comments:
        n, col = t.start[0] - 1, t.start[1]
        if n in protected or not URL.search(t.string) or LICENCE.search(t.string):
            continue
        if n in code_lines:                                         # inline: cut the comment
            eol = lines[n][len(lines[n].rstrip("\r\n")):]
            lines[n] = lines[n][:col].rstrip() + eol
        else:
            drop.add(n)


def strip_py(path: Path) -> bool:
    raw = path.read_bytes()
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        src = raw.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError):
        return False
    if not (URL.search(src) or "References" in src or ".. [" in src):
        return False
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return False
    lines = src.splitlines(keepends=True)
    drop: set[int] = set()
    for first, last in _docstring_spans(tree, lines):
        _strip_docstring(lines, first, last, drop)
    _strip_comments(src, lines, drop)
    new = "".join(l for n, l in enumerate(lines) if n not in drop)
    if new == src:
        return False
    try:
        ast.parse(new)
    except (SyntaxError, ValueError):
        return False                                                # never ship a broken file
    path.write_bytes(new.encode(encoding))
    return True


def strip_metadata(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    head, sep, body = text.partition("\n\n")
    kept, skipping = [], False
    for line in head.split("\n"):
        if line[:1] in (" ", "\t") and skipping:                    # folded continuation
            continue
        name = line.split(":", 1)[0].strip().lower()
        skipping = name in ("description", "home-page", "download-url", "project-url")
        if not skipping:
            kept.append(line)
    new = "\n".join(kept) + "\n"
    if new == text:
        return False
    path.write_text(new, encoding="utf-8", errors="surrogateescape")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    py = sum(strip_py(p) for p in root.rglob("*.py"))
    meta = sum(strip_metadata(p) for p in root.rglob("*.dist-info/METADATA"))
    print(f"strip-citations: {py} Python files and {meta} METADATA files cleaned under {root.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
