#!/usr/bin/env python3
"""Minify the viewer's runtime, `static/viewer.js`, into `static/viewer.min.js`.

The package inlines the minified file in each viewer plot's HTML, and serves it to the app
frameworks, when its first line has the hash of `viewer.js` as it is; otherwise (as while
`viewer.js` is being changed) it uses `viewer.js` itself. A test fails while the minified file is
out of date, so run this after changing `viewer.js`. It needs node, for esbuild, which `npx`
fetches at the version pinned here.

Usage:
    python scripts/minify_viewer.py           # write static/viewer.min.js
    python scripts/minify_viewer.py --check   # exit 1 if it's out of date
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ESBUILD = "esbuild@0.28.2"
STATIC = Path(__file__).resolve().parent.parent / "python" / "edgecase" / "static"
SOURCE = STATIC / "viewer.js"
MINIFIED = STATIC / "viewer.min.js"
# The first line of the minified file; the package reads the hash back from it
BANNER = "// edgecase's viewer.js, minified by scripts/minify_viewer.py, sha256 {}\n"


def banner(source: bytes) -> str:
    return BANNER.format(hashlib.sha256(source).hexdigest())


def minified(source: Path) -> str:
    npx = shutil.which("npx")
    if npx is None:
        sys.exit("minify_viewer.py needs node's npx, for esbuild")
    run = subprocess.run(
        [npx, "--yes", ESBUILD, str(source), "--minify", "--format=esm", "--charset=ascii",
         "--legal-comments=none", "--log-level=warning"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if run.returncode != 0:
        sys.exit(run.stderr)
    return run.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if it's out of date")
    args = parser.parse_args()
    head = banner(SOURCE.read_bytes())
    if args.check:
        current = MINIFIED.exists() and MINIFIED.read_text(encoding="utf-8").startswith(head)
        print(f"{MINIFIED.name} is {'up to date' if current else 'out of date'}")
        return 0 if current else 1
    MINIFIED.write_text(head + minified(SOURCE), encoding="utf-8")
    print(
        f"wrote {MINIFIED.name}: {MINIFIED.stat().st_size:,} bytes from {SOURCE.stat().st_size:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
