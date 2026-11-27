"""Write the spike's payloads: positions from the force layout `plot()` uses, edges and a color per node.

Each graph gives `data/<name>.json` (counts, palette, byte offsets) and `data/<name>.bin`:
float32 x, y per node, then uint32 source, target per edge, then uint8 color index per node.

    .venv/bin/python docs_plan/viewer_spike/export.py
"""

import gzip
import json
import random
import time
from pathlib import Path

import numpy as np

import edgecase as ec
from edgecase.colors import get_palette
from edgecase.svg_renderer import _force_directed_layout

OUT = Path(__file__).parent / "data"
PALETTE = get_palette("default")


def write(name, n, edges, colors, iterations, note):
    start = time.perf_counter()
    # The layout `plot()` draws: components laid out alone and packed, on a 1400 x 1000 frame
    raw = _force_directed_layout(list(range(n)), edges, 1400, 1000, iterations, seed=1)
    took = time.perf_counter() - start
    xy = np.array([raw[i] for i in range(n)], dtype=np.float32)
    xy[:, 1] *= -1  # SVG's y runs down, WebGL's up
    # Centered on the origin and scaled to [-1, 1] along the longer side, as the viewer expects
    xy -= (xy.min(axis=0) + xy.max(axis=0)) / 2
    xy /= np.abs(xy).max()
    # The edges in a random order, so that the viewer's first k of them are a fair sample
    order = np.random.default_rng(1).permutation(len(edges))
    ends = np.asarray(edges, dtype=np.uint32)[order].reshape(-1)
    color = np.asarray(colors, dtype=np.uint8)
    blob = xy.tobytes() + ends.tobytes() + color.tobytes()
    (OUT / f"{name}.bin").write_bytes(blob)
    header = {
        "name": name,
        "nodes": n,
        "edges": len(edges),
        "palette": PALETTE[: max(colors) + 1],
        "xy": [0, xy.nbytes],
        "ends": [xy.nbytes, ends.nbytes],
        "color": [xy.nbytes + ends.nbytes, color.nbytes],
        "layout": f"svg_renderer._force_directed_layout, {iterations} iterations, {took:.1f} s",
        "note": note,
    }
    (OUT / f"{name}.json").write_text(json.dumps(header, indent=1))
    zipped = len(gzip.compress(blob, 6))
    print(f"{name}: {n:,} nodes, {len(edges):,} edges, layout {took:.1f} s, "
          f"{len(blob) / 1e6:.1f} MB ({zipped / 1e6:.1f} MB gzipped)", flush=True)


def main():
    OUT.mkdir(exist_ok=True)

    g = ec.wiki_vote()
    ids = g.get_node_df()["id"].to_list()
    index = {v: i for i, v in enumerate(ids)}
    df = g.get_edge_df()
    edges = [(index[s], index[t]) for s, t in zip(df["source"], df["target"])]
    # Louvain needs an undirected graph; the colors are its communities, the largest first,
    # with the ones past the palette folded into its last color
    und = g.to_undirected().cmty_louvain()
    nd = und.get_node_df()
    cmty = dict(zip(nd["id"], nd["cmty_louvain"]))
    sizes = {}
    for c in cmty.values():
        sizes[c] = sizes.get(c, 0) + 1
    rank = {c: r for r, c in enumerate(sorted(sizes, key=lambda c: -sizes[c]))}
    top = len(PALETTE) - 1
    colors = [min(rank[cmty[v]], top) for v in ids]
    write("wiki_vote", len(ids), edges, colors, 300, "colored by Louvain on the undirected graph")

    rng = random.Random(1)
    n = 200_000
    edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(1_000_000)]
    colors = [i % 8 for i in range(n)]
    write("rand_1m", n, edges, colors, 100, "uniform random edges; colors by node index mod 8")


if __name__ == "__main__":
    main()
