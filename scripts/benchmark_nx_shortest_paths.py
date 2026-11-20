"""Time the NetworkX backend's shortest paths against NetworkX itself.

For each function and graph size, times NetworkX, edgecase including the conversion of the
NetworkX graph (what `nx.func(G, backend="edgecase")` costs on a graph seen for the first
time), and edgecase on a graph converted beforehand (warm: its search structure already
built). Each time is the fastest of a few runs.

Usage:
    python scripts/benchmark_nx_shortest_paths.py [--quick] [--only NAME...] [--out results.csv]
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


def make_grid(num_nodes, num_edges=None):
    """A square grid (large diameter: where `usebounds=True` helps), weighted like above."""
    side = round(num_nodes**0.5)
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    for i, (_, _, data) in enumerate(G.edges(data=True)):
        data["weight"] = (i % 97) / 7 + 1
    return G


def consume(result):
    """Exhaust a generator (all-pairs functions) without keeping its items."""
    if not isinstance(result, dict | list | tuple | int | float):
        for _ in result:
            pass


# (function, call(G), weighted, sizes as (nodes, edges), repeats[, graph maker])
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
    (
        "closeness_centrality",
        lambda G: nx.closeness_centrality(G),
        False,
        [(500, 5_000), (2_000, 20_000)],
        3,
    ),
    (
        "closeness_centrality (weighted)",
        lambda G: nx.closeness_centrality(G, distance="weight"),
        True,
        [(500, 5_000), (2_000, 20_000)],
        3,
    ),
    (
        "harmonic_centrality",
        lambda G: nx.harmonic_centrality(G),
        False,
        [(500, 5_000), (2_000, 20_000)],
        3,
    ),
    (
        "eccentricity",
        lambda G: nx.eccentricity(G),
        False,
        [(500, 5_000), (2_000, 20_000)],
        3,
    ),
    (
        "eigenvector_centrality",
        lambda G: nx.eigenvector_centrality(G, max_iter=1000),
        False,
        [(10_000, 100_000), (50_000, 500_000)],
        2,
    ),
    (
        "katz_centrality",
        lambda G: nx.katz_centrality(G, alpha=0.02),
        False,
        [(10_000, 100_000), (50_000, 500_000)],
        2,
    ),
    (
        "edge_betweenness_centrality",
        lambda G: nx.edge_betweenness_centrality(G),
        False,
        [(500, 5_000), (2_000, 20_000)],
        2,
    ),
    (
        "hits",
        lambda G: nx.hits(G, max_iter=1000),
        True,
        [(10_000, 100_000), (100_000, 1_000_000)],
        2,
    ),
    (
        "jaccard_coefficient (all non-edges)",
        lambda G: consume(nx.jaccard_coefficient(G)),
        False,
        [(500, 5_000), (1_500, 15_000)],
        2,
    ),
    (
        "load_centrality",
        lambda G: nx.load_centrality(G),
        False,
        [(500, 5_000), (2_000, 20_000)],
        2,
    ),
    (
        "diameter (usebounds)",
        lambda G: nx.diameter(G, usebounds=True),
        False,
        [(10_000, None), (90_000, None)],
        3,
        make_grid,
    ),
]


def run(cases, quick, out=None):
    nx.config.cache_converted_graphs = False
    commit = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    rows = []
    for name, call, weighted, sizes, repeat, *maker in cases:
        maker = maker[0] if maker else make_graph
        if quick:
            sizes = sizes[:1]
            repeat = min(repeat, 2)
        for num_nodes, num_edges in sizes:
            G = maker(num_nodes, num_edges)
            E = from_networkx(G, edge_attrs={"weight": 1} if weighted else None)
            call(E)  # builds and caches E's search structure
            with nx.config.backend_priority(algos=[]):
                t_nx = best_time(lambda: call(G), repeat)
            with nx.config.backend_priority(algos=["edgecase"]):
                t_conv = best_time(lambda: call(G), repeat)
            t_cached = best_time(lambda: call(E), repeat)
            row = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "commit": commit,
                "function": name,
                "graph": "grid" if maker is make_grid else "Graph",
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "networkx_s": round(t_nx, 6),
                "edgecase_with_conversion_s": round(t_conv, 6),
                "edgecase_cached_graph_s": round(t_cached, 6),
                "speedup_with_conversion": round(t_nx / t_conv, 1),
                "speedup_cached": round(t_nx / t_cached, 1),
            }
            print(row, flush=True)
            rows.append(row)
            if out:
                write_row(out, row)
    return rows


def write_row(path, row):
    """Append `row` to the CSV file at `path` (with a header if the file is new)."""
    try:
        with open(path) as f:
            new_file = not f.read(1)
    except FileNotFoundError:
        new_file = True
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smallest size only")
    parser.add_argument("--out", help="append rows to this CSV file")
    parser.add_argument("--only", nargs="*", help="run the cases whose name starts so")
    args = parser.parse_args()
    cases = CASES
    if args.only:
        cases = [case for case in CASES if case[0].startswith(tuple(args.only))]
    run(cases, args.quick, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
