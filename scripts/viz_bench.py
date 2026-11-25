#!/usr/bin/env python3
"""Measure what drawing a graph costs: time, peak memory and output size, per graph.

Each case runs in its own subprocess, so its peak memory is its own, under a time limit. A case
is a graph and a kind of work:

- `plot`: `plot()` with its defaults, without labels.
- `plot_fast`: `plot()` with straight edges and overlapping nodes allowed.
- `edge_df`: `get_edge_df()`.
- `load`: only building the graph (with a `gnm:` graph, this measures `add_gnm_graph()`).

A graph is a dataset loader name (`power_grid`), `rand:N:M` (a sparse random edge list sampled
here, so building it costs nothing much), or `gnm:N:M` (built by `add_gnm_graph()`).

The rows are written as JSON to `scripts/viz_bench.json`, the baseline. With `--compare`, a run
is printed against that baseline instead of replacing it, which is how a change shows its gain.

Usage:
    python scripts/viz_bench.py                       # the default cases, written as the baseline
    python scripts/viz_bench.py --compare             # the same cases, against the baseline
    python scripts/viz_bench.py plot power_grid pgp   # chosen cases (with --compare or --out)
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "viz_bench.json"

SMALL = ["karate_club", "les_miserables", "football", "jazz", "celegans"]
MEDIUM = ["power_grid", "ca_grqc", "internet", "gnutella", "pgp"]
DEFAULT_CASES = (
    [("plot", name) for name in SMALL + MEDIUM]
    + [("plot_fast", name) for name in MEDIUM + ["wiki_vote", "condmat", "rand:100000:200000"]]
    + [("edge_df", name) for name in ["wiki_vote", "physics", "rand:100000:200000"]]
    + [("load", "gnm:5000:10000")]
)


def build_graph(name: str):
    import edgecase as ec

    if name.startswith(("rand:", "gnm:")):
        source, n, m = name.split(":")
        n, m = int(n), int(m)
        if source == "gnm":
            return ec.Graph().add_gnm_graph(n=n, m=m, seed=1)
        rng = random.Random(1)
        pairs = set()
        while len(pairs) < m:
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        return ec.Graph.from_edges(sorted(pairs))
    return getattr(ec, name)()


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # bytes on macOS, kilobytes on Linux
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def run_case(kind: str, name: str) -> dict:
    """Do one case in this process and return its row."""
    start = time.perf_counter()
    graph = build_graph(name)
    load_s = time.perf_counter() - start

    out_mb = None
    start = time.perf_counter()
    if kind == "plot":
        out_mb = len(graph.plot(seed=1, show_labels=False)._repr_html_()) / 1e6
    elif kind == "plot_fast":
        plot = graph.plot(seed=1, show_labels=False, edge_routing="straight", node_overlap="allow")
        out_mb = len(plot._repr_html_()) / 1e6
    elif kind == "edge_df":
        graph.get_edge_df()
    elif kind != "load":
        raise ValueError(f"unknown kind {kind!r}")
    # for `load`, the work is building the graph
    work_s = load_s if kind == "load" else time.perf_counter() - start

    return {
        "kind": kind,
        "graph": name,
        "n": graph.count_nodes(),
        "m": graph.count_edges(),
        "load_s": round(load_s, 2),
        "work_s": round(work_s, 2),
        "peak_rss_mb": round(peak_rss_mb()),
        "out_mb": None if out_mb is None else round(out_mb, 2),
    }


def run_in_subprocess(kind: str, name: str, timeout: float) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, __file__, "--one", kind, name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"kind": kind, "graph": name, "error": f"timeout {timeout:g}s"}
    if result.returncode != 0:
        last = result.stderr.strip().splitlines()[-1:] or ["no output"]
        return {"kind": kind, "graph": name, "error": last[0]}
    return json.loads(result.stdout.strip().splitlines()[-1])


def ratio(new, old) -> str:
    if new is None or old is None:
        return ""
    if old == 0:
        return "" if new == 0 else "new"
    return f"{new / old:.2f}x"


def compare_line(row: dict, base: dict | None, compare: bool) -> str:
    label = f"{row['kind']:<10} {row['graph']:<22}"
    if "error" in row:
        return f"{label} {row['error']}"
    line = f"{label} {row['work_s']:>8.2f} s {row['peak_rss_mb']:>6} MB" + (
        f" {row['out_mb']:>8.2f} MB out" if row["out_mb"] is not None else " " * 16
    )
    if not compare:
        return line
    if base is None or "error" in base:
        return line + ("   (no baseline)" if base is None else f"   (baseline: {base['error']})")
    return line + (
        f"   time {ratio(row['work_s'], base['work_s']):>7}"
        f"   memory {ratio(row['peak_rss_mb'], base['peak_rss_mb']):>6}"
        f"   output {ratio(row['out_mb'], base['out_mb']):>6}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", nargs="?", help="plot, plot_fast, edge_df or load")
    parser.add_argument("graphs", nargs="*", help="dataset names, rand:N:M or gnm:N:M")
    parser.add_argument("--compare", action="store_true", help="print against the baseline")
    parser.add_argument("--out", type=Path, help="write the rows here instead")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per case")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        print(json.dumps(run_case(args.kind, args.graphs[0])))
        return 0

    cases = [(args.kind, g) for g in args.graphs] if args.kind else DEFAULT_CASES
    if args.kind and not (args.compare or args.out):
        parser.error("choosing cases needs --compare or --out, so the baseline stays whole")

    baseline = {}
    if args.compare and BASELINE.exists():
        rows = json.loads(BASELINE.read_text())["rows"]
        baseline = {(r["kind"], r["graph"]): r for r in rows}

    rows = []
    for kind, name in cases:
        row = run_in_subprocess(kind, name, args.timeout)
        rows.append(row)
        print(compare_line(row, baseline.get((kind, name)), args.compare), flush=True)

    out = args.out or (None if args.compare else BASELINE)
    if out is not None:
        import edgecase as ec

        meta = {
            "edgecase": ec.__version__,
            "python": sys.version.split()[0],
            "date": time.strftime("%Y-%m-%d"),
        }
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
