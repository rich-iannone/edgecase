"""Point every Graph docstring at its figure, with alt text from the registry.

    .venv/bin/python scripts/doc_graphics/sync_docstrings.py          # rewrite graph.py
    .venv/bin/python scripts/doc_graphics/sync_docstrings.py --check  # exit 1 if out of date
    .venv/bin/python scripts/doc_graphics/sync_docstrings.py --insert # also add missing lines

For each registered figure, finds the `![...](/assets/graphs/....light.svg){...}`
line in that method's docstring and replaces it with the canonical snippet.
Methods with a figure but no image line are reported; with --insert, the line
is added on its own, one blank line under the docstring's summary (the house
placement). Image lines with no registered figure are reported only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build import ROOT, load_figures, snippet  # noqa: E402

GRAPH_PY = ROOT / "python" / "edgecase" / "graph.py"
IMG = re.compile(r"^(\s*)!\[[^\]]*\]\(/assets/graphs/[^)]+\.light\.svg\)\{[^\n]*\}\s*$")
DEF = re.compile(r"^    def (\w+)\(")
OPEN = re.compile(r'^\s*r?"""\s*$')


def _insert(lines: list[str], method: str, snip: str) -> bool:
    """Insert `snip` under the summary of `method`'s docstring. True on success."""
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"    def {method}(")), None)
    if start is None:
        return False
    i = start
    while i < len(lines) and not OPEN.match(lines[i]):
        if i > start and DEF.match(lines[i]):
            return False
        i += 1
    j = i + 1  # first summary line
    if j >= len(lines) or not lines[j].strip():
        return False
    indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
    while j < len(lines) and lines[j].strip() and '"""' not in lines[j]:
        j += 1
    lines[j:j] = ["", indent + snip]
    return True


def main() -> int:
    check = "--check" in sys.argv
    insert = "--insert" in sys.argv
    figs = {f.method: f for f in load_figures()}
    for f in figs.values():
        if '"' in f.alt or "\\" in f.alt:
            print(f"error: alt text for {f.method} contains a double quote or backslash")
            return 1

    lines = GRAPH_PY.read_text().split("\n")
    current = None
    seen: set[str] = set()
    changed = 0
    orphans = []
    for i, line in enumerate(lines):
        m = DEF.match(line)
        if m:
            current = m.group(1)
            continue
        im = IMG.match(line)
        if not im:
            continue
        if current not in figs:
            orphans.append(f"{current} (line {i + 1})")
            continue
        if current in seen:
            continue
        seen.add(current)
        new = im.group(1) + snippet(figs[current])
        if line != new:
            lines[i] = new
            changed += 1

    missing = sorted(set(figs) - seen)
    inserted = 0
    for m in missing:
        if insert and not check and _insert(lines, m, snippet(figs[m])):
            inserted += 1
            changed += 1
        else:
            print(f"note: {m}() has a figure but no image line in its docstring")
    if inserted:
        print(f"inserted {inserted} new figure lines")
    for o in orphans:
        print(f"note: image line in {o} has no registered figure")
    if check:
        print(f"{changed} docstring figure lines out of date")
        return 1 if changed else 0
    if changed:
        GRAPH_PY.write_text("\n".join(lines))
    print(f"updated {changed} docstring figure lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
