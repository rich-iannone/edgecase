"""Time conversions between NetworkX graphs and edgecase.

Compares the NetworkX backend's converter (`edgecase.nx_backend.from_networkx`, one
Python pass plus one bulk Rust call) with `edgecase.networkx.from_networkx` (which builds
a full `ec.Graph`, one Rust call per node and edge) and with copying the NetworkX graph.

Usage:
    python scripts/benchmark_nx_conversion.py [--edges 10000 1000000] [--out results.csv]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import subprocess
import sys
import time

import networkx as nx
from edgecase.networkx import from_networkx as ec_from_networkx
from edgecase.nx_backend import from_networkx, to_networkx


def best_time(func, repeat):
    """Fastest of `repeat` runs, in seconds (garbage collection off while timing)."""
    times = []
    for _ in range(repeat):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            func()
            times.append(time.perf_counter() - start)
        finally:
            gc.enable()
    return min(times)


def make_graph(num_edges, directed, seed=42):
    num_nodes = max(num_edges // 10, 10)
    G = nx.gnm_random_graph(num_nodes, num_edges, seed=seed, directed=directed)
    for i, (_, _, data) in enumerate(G.edges(data=True)):
        data["weight"] = (i % 97) / 7 + 1
    return G


def methods(G):
    def backend_unweighted():
        from_networkx(G).rust_graph()

    def backend_weighted():
        from_networkx(G, edge_attrs={"weight": 1}).rust_graph("weight")

    def backend_python_pass_only():
        from_networkx(G, edge_attrs={"weight": 1})

    def backend_all_attrs():
        from_networkx(
            G, preserve_edge_attrs=True, preserve_node_attrs=True, preserve_graph_attrs=True
        ).rust_graph("weight")

    converted = from_networkx(G, edge_attrs={"weight": 1})

    def backend_to_networkx():
        to_networkx(converted)

    def ec_graph():
        ec_from_networkx(G)

    def nx_copy():
        G.copy()

    return {
        "nx_backend.from_networkx (no attrs) + rust build": backend_unweighted,
        "nx_backend.from_networkx (weight) + rust build": backend_weighted,
        "nx_backend.from_networkx (weight), python pass only": backend_python_pass_only,
        "nx_backend.from_networkx (all attrs) + rust build": backend_all_attrs,
        "nx_backend.to_networkx (weight)": backend_to_networkx,
        "edgecase.networkx.from_networkx (ec.Graph)": ec_graph,
        "networkx G.copy() (reference)": nx_copy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=int, nargs="+", default=[10_000, 1_000_000])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", default=None, help="CSV file to append rows to")
    args = parser.parse_args()

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    rows = []
    for num_edges in args.edges:
        for directed in (False, True):
            G = make_graph(num_edges, directed)
            repeat = args.repeat if num_edges <= 100_000 else max(1, args.repeat - 1)
            baseline = None
            for method, func in methods(G).items():
                seconds = best_time(func, repeat)
                if method.startswith("edgecase.networkx"):
                    baseline = seconds
                rows.append(
                    {
                        "timestamp": stamp,
                        "commit": commit,
                        "graph": "DiGraph" if directed else "Graph",
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "method": method,
                        "seconds": f"{seconds:.6f}",
                    }
                )
                print(f"{rows[-1]['graph']:8} {num_edges:>9} {method:55} {seconds:9.4f} s")
                sys.stdout.flush()
            for row in rows[-len(methods(G)) :]:
                row["speedup_vs_ec_graph"] = f"{baseline / float(row['seconds']):.2f}"
    if args.out:
        with open(args.out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
