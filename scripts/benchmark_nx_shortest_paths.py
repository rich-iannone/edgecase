"""Time the NetworkX backend's shortest paths against NetworkX itself.

For each function and graph size, times NetworkX, edgecase including the conversion of the
NetworkX graph (what `nx.func(G, backend="edgecase")` costs on a graph seen for the first
time), and edgecase on a graph converted beforehand (warm: its search structure already
built). Each time is the fastest of a few runs.

Usage:
    python scripts/benchmark_nx_shortest_paths.py [--quick] [--out results.csv]
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
from edgecase.nx_backend import from_networkx


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


def make_graph(num_nodes, num_edges, directed=False, seed=42):
    G = nx.gnm_random_graph(num_nodes, num_edges, seed=seed, directed=directed)
    for i, (_, _, data) in enumerate(G.edges(data=True)):
        data["weight"] = (i % 97) / 7 + 1
    return G


def consume(result):
    """Exhaust a generator (all-pairs functions) without keeping its items."""
    if not isinstance(result, dict | list | tuple | int | float):
        for _ in result:
            pass


# (function, call(G), weighted, sizes as (nodes, edges), repeats)
CASES = [
    (
        "single_source_dijkstra_path_length",
        lambda G: nx.single_source_dijkstra_path_length(G, 0),
        True,
        [(10_000, 100_000), (100_000, 1_000_000)],
        5,
    ),
    (
        "single_source_shortest_path_length",
        lambda G: nx.single_source_shortest_path_length(G, 0),
        False,
        [(10_000, 100_000), (100_000, 1_000_000)],
        5,
    ),
    (
        "all_pairs_shortest_path_length",
        lambda G: consume(nx.all_pairs_shortest_path_length(G)),
        False,
        [(500, 5_000), (2_000, 20_000)],
        3,
    ),
    (
        "all_pairs_dijkstra_path_length",
        lambda G: consume(nx.all_pairs_dijkstra_path_length(G)),
        True,
        [(500, 5_000), (2_000, 20_000)],
        3,
    ),
    (
        "shortest_path (weighted, one pair)",
        lambda G: nx.shortest_path(G, 0, 1, weight="weight"),
        True,
        [(100_000, 1_000_000)],
        5,
    ),
    (
        "average_shortest_path_length",
        lambda G: nx.average_shortest_path_length(G),
        False,
        [(2_000, 20_000)],
        3,
    ),
]


def run(cases, quick):
    nx.config.cache_converted_graphs = False
    rows = []
    for name, call, weighted, sizes, repeat in cases:
        if quick:
            sizes = sizes[:1]
            repeat = min(repeat, 2)
        for num_nodes, num_edges in sizes:
            G = make_graph(num_nodes, num_edges)
            E = from_networkx(G, edge_attrs={"weight": 1} if weighted else None)
            call(E)  # builds and caches E's search structure
            with nx.config.backend_priority(algos=[]):
                t_nx = best_time(lambda: call(G), repeat)
            with nx.config.backend_priority(algos=["edgecase"]):
                t_conv = best_time(lambda: call(G), repeat)
            t_cached = best_time(lambda: call(E), repeat)
            row = {
                "function": name,
                "graph": "Graph",
                "nodes": num_nodes,
                "edges": num_edges,
                "networkx_s": round(t_nx, 6),
                "edgecase_with_conversion_s": round(t_conv, 6),
                "edgecase_cached_graph_s": round(t_cached, 6),
                "speedup_with_conversion": round(t_nx / t_conv, 1),
                "speedup_cached": round(t_nx / t_cached, 1),
            }
            print(row, flush=True)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smallest size only")
    parser.add_argument("--out", help="append rows to this CSV file")
    args = parser.parse_args()
    rows = run(CASES, args.quick)
    if args.out:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        fields = ["timestamp", "commit", *rows[0]]
        try:
            with open(args.out) as f:
                new_file = not f.read(1)
        except FileNotFoundError:
            new_file = True
        with open(args.out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                writer.writeheader()
            for row in rows:
                writer.writerow({"timestamp": timestamp, "commit": commit, **row})
    return 0


if __name__ == "__main__":
    sys.exit(main())
