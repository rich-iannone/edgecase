#!/usr/bin/env python3
"""Generate instructional SVG cartoons for edgecase docstrings.

Run: .venv/bin/python scripts/generate_doc_graphics.py [--lint] [--only ...] [--gallery PATH]
Outputs:
    assets/graphs/<method>.light.svg
    assets/graphs/<method>.dark.svg
    assets/graphs/README.md      (maintainer index; not linked from the site)

The design system and per-family recipes live in DOC_GRAPHICS_PLAN.md; the
code lives in scripts/doc_graphics/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "doc_graphics"))

from build import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
