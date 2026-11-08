#!/usr/bin/env python3
"""Time the README's benchmark table: EdgeCase against NetworkX and igraph.

The same simple undirected graph is built in each library from one edge list: the co-authorship
network `condmat()` (23,133 nodes and 93,439 edges once the listing of every collaboration in
both directions is merged), and the power grid network for exact betweenness, which is too slow
in NetworkX on the larger graph. Each task is timed as the best of `--repeat` runs (NetworkX of
at most three), with the results returned to Python.

The timings are written to `assets/readme/benchmarks.json`, which `scripts/readme_graphics.py`
draws as the benchmark chart. NetworkX and python-igraph must be installed.

Usage:
    python scripts/readme_benchmarks.py [--repeat 5]
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from pathlib import Path

import edgecase as ec
import igraph as ig
import networkx as nx

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme" / "benchmarks.json"


def best_of(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def simple_edges(graph: ec.Graph) -> list[tuple[str, str]]:
    """The edges as label pairs, without self-loops or repeated pairs."""
    label = {n: graph.get_label_from_id(n) for n in graph.get_node_ids(limit=None)}
    pairs = set()
    for _, source, target in graph.get_edges_as_tuples():
        if source != target:
            pairs.add(tuple(sorted((label[source], label[target]))))
    return sorted(pairs)


def build(edges):
    nodes = sorted({node for edge in edges for node in edge})
    index = {node: i for i, node in enumerate(nodes)}
    pairs = [(index[a], index[b]) for a, b in edges]
    g = ec.Graph.from_edges(edges)
    n = nx.Graph(edges)
    i = ig.Graph(n=len(nodes), edges=pairs)
    return g, n, i, nodes[0], pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    rows = []

    def run(task, graph_name, ec_fn, nx_fn, ig_fn):
        row = {
            "task": task,
            "graph": graph_name,
            "edgecase": best_of(ec_fn, args.repeat),
            "networkx": best_of(nx_fn, min(args.repeat, 3)),
            "igraph": best_of(ig_fn, args.repeat),
        }
        rows.append(row)
        print(
            f"{task:30s} edgecase {row['edgecase'] * 1e3:9.1f} ms   "
            f"networkx {row['networkx'] * 1e3:10.1f} ms   igraph {row['igraph'] * 1e3:9.1f} ms",
            flush=True,
        )

    edges = simple_edges(ec.condmat())
    g, n, i, first, pairs = build(edges)
    source = g.get_id_from_label(first)
    name = f"condmat ({g.count_nodes():,} nodes, {g.count_edges():,} edges)"
    print(name)
    run(
        "Build from an edge list",
        name,
        lambda: ec.Graph.from_edges(edges),
        lambda: nx.Graph(edges),
        lambda: ig.Graph(n=i.vcount(), edges=pairs),
    )
    run(
        "Connected components",
        name,
        lambda: g.weakly_connected_components(),
        lambda: list(nx.connected_components(n)),
        lambda: i.connected_components(),
    )
    run(
        "PageRank",
        name,
        lambda: g.pagerank(return_dict=True),
        lambda: nx.pagerank(n),
        lambda: i.pagerank(),
    )
    run(
        "Distances from one node",
        name,
        lambda: g.single_source_shortest_path_length(source, return_dict=True),
        lambda: nx.single_source_shortest_path_length(n, first),
        lambda: i.distances(source=[0]),
    )
    run(
        "Triangles per node",
        name,
        lambda: g.count_triangles(),
        lambda: nx.triangles(n),
        lambda: i.list_triangles(),
    )
    run(
        "Average clustering",
        name,
        lambda: g.average_clustering_coefficient(),
        lambda: nx.average_clustering(n),
        lambda: i.transitivity_avglocal_undirected(mode="zero"),
    )
    run(
        "k-core decomposition",
        name,
        lambda: g.coreness(return_dict=True),
        lambda: nx.core_number(n),
        lambda: i.coreness(),
    )
    run(
        "Louvain communities",
        name,
        lambda: g.cmty_louvain(return_dict=True),
        lambda: nx.community.louvain_communities(n, seed=1),
        lambda: i.community_multilevel(),
    )

    edges = simple_edges(ec.power_grid())
    g, n, i, _, _ = build(edges)
    name = f"power_grid ({g.count_nodes():,} nodes, {g.count_edges():,} edges)"
    print(name)
    run(
        "Exact betweenness",
        name,
        lambda: g.betweenness_centrality(return_dict=True),
        lambda: nx.betweenness_centrality(n),
        lambda: i.betweenness(),
    )

    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        cpu = ""
    meta = {
        "machine": cpu or platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "networkx": nx.__version__,
        "igraph": ig.__version__,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
