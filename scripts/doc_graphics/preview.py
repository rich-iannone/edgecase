"""Render figures to PNG contact sheets for visual review (light | dark).

    .venv/bin/python scripts/doc_graphics/preview.py OUTDIR [method-or-family ...]
    .venv/bin/python scripts/doc_graphics/preview.py OUTDIR --legacy SLUG ...

Needs `rsvg-convert` (brew install librsvg) and Pillow. Writes
OUTDIR/sheet_NN.png, eight figures per sheet. `--legacy` renders the old,
pre-redesign SVGs from git HEAD for side-by-side comparison.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from build import ROOT, load_figures  # noqa: E402
from core import THEMES  # noqa: E402

PER_SHEET = 8


def _png(svg: str, path: str) -> None:
    Path(path).write_text(svg)
    subprocess.run(["rsvg-convert", "-w", "760", path, "-o", path[:-4] + ".png"], check=True)


def _sheet(out: str, items: list[tuple[str, list[str]]]) -> list[str]:
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    paths = []
    for k in range(0, len(items), PER_SHEET):
        chunk = items[k : k + PER_SHEET]
        rows = []
        for label, pngs in chunk:
            ims = [Image.open(p).convert("RGB") for p in pngs]
            rows.append((label, ims))
        width = 10 + sum(im.width + 10 for im in rows[0][1])
        height = sum(24 + max(im.height for im in ims) for _, ims in rows)
        sheet = Image.new("RGB", (max(width, 800), height), "#bbbbbb")
        d = ImageDraw.Draw(sheet)
        y = 0
        for label, ims in rows:
            d.text((10, y + 4), label, fill="black", font=font)
            x = 10
            for im in ims:
                sheet.paste(im, (x, y + 22))
                x += im.width + 10
            y += 24 + max(im.height for im in ims)
        p = f"{out}/sheet_{k // PER_SHEET:02d}.png"
        sheet.save(p)
        paths.append(p)
    return paths


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    out, *names = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    items = []
    if names and names[0] == "--legacy":
        for slug in names[1:]:
            svg = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:assets/graphs/{slug}.light.svg"],
                                 capture_output=True, text=True, check=True).stdout
            p = f"{out}/legacy-{slug}.svg"
            _png(svg, p)
            items.append((f"legacy / {slug}", [p[:-4] + ".png"]))
    else:
        want = set(names)
        for f in load_figures():
            if want and f.method not in want and f.family not in want:
                continue
            sc = f.scene()
            pngs = []
            for th in THEMES:
                p = f"{out}/{f.slug}.{th.suffix}.svg"
                _png(sc.render(th), p)
                pngs.append(p[:-4] + ".png")
            items.append((f"{f.family} / {f.method}", pngs))
    for p in _sheet(out, items):
        print(p)


if __name__ == "__main__":
    main()
