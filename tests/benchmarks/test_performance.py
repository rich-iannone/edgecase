# tests/benchmarks/test_performance.py
"""
Benchmark tests using pytest-benchmark
"""

import pytest
from edgecase import Graph
import polars as pl
import random


@pytest.fixture
def benchmark_graph_small():
    """1K node graph for benchmarking"""
    G = Graph()
    # Create nodes first
    for i in range(1000):
        G.add_node(f"n{i}")
    # Then create edges
    for i in range(1000):
        G.add_edge(f"n{i}", f"n{(i + 1) % 1000}")
    return G


@pytest.fixture
def benchmark_graph_medium():
    """10K node graph for benchmarking"""
    G = Graph()
    random.seed(42)

    # Create scale-free-like graph
    for i in range(10000):
        G.add_node(f"n{i}")

    for i in range(50000):
        src = f"n{random.randint(0, 9999)}"
        tgt = f"n{random.randint(0, 9999)}"
        G.add_edge(src, tgt)

    return G


# Graph creation performance tests


def test_add_1000_nodes(benchmark):
    """Benchmark adding 1000 nodes"""

    def add_nodes():
        G = Graph()
        for i in range(1000):
            G.add_node(f"node_{i}")
        return G

    G = benchmark(add_nodes)
    assert G.count_nodes() == 1000


def test_add_10000_edges(benchmark):
    """Benchmark adding 10000 edges"""

    def add_edges():
        G = Graph()
        # Create nodes first
        for i in range(1000):
            G.add_node(f"n{i}")
        # Then create edges
        for i in range(10000):
            G.add_edge(f"n{i % 1000}", f"n{(i + 1) % 1000}")
        return G

    G = benchmark(add_edges)
    assert G.count_edges() > 0


# Algorithm performance tests


def test_pagerank_1k_nodes(benchmark, benchmark_graph_small):
    """Benchmark PageRank on 1K node graph"""
    result = benchmark(lambda: benchmark_graph_small.pagerank(return_dict=True))
    assert len(result) == 1000


def test_shortest_path_1k_nodes(benchmark, benchmark_graph_small):
    """Benchmark shortest path on 1K node graph"""
    result = benchmark(benchmark_graph_small.shortest_path, "n0", "n500")
    assert result is not None


# Polars integration performance tests


def test_graph_from_df_1k_edges(benchmark):
    """Benchmark Graph constructor with DataFrame containing 1K edges"""
    df = pl.DataFrame(
        {
            "from": [f"n{i}" for i in range(1000)],
            "to": [f"n{(i + 1) % 1000}" for i in range(1000)],
            "weight": [1.0] * 1000,
        }
    )

    G = benchmark(lambda: Graph(edges_df=df, weighted=True))
    assert G.count_edges() == 1000


def test_graph_from_df_10k_edges(benchmark):
    """Benchmark Graph constructor with DataFrame containing 10K edges"""
    random.seed(42)

    df = pl.DataFrame(
        {
            "from": [f"n{random.randint(0, 1000)}" for _ in range(10000)],
            "to": [f"n{random.randint(0, 1000)}" for _ in range(10000)],
            "weight": [random.random() for _ in range(10000)],
        }
    )

    G = benchmark(lambda: Graph(edges_df=df, weighted=True))
    assert G.count_edges() > 0


# Parameterized benchmarks for scaling analysis
@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_pagerank_scaling(benchmark, size):
    """Test PageRank scaling with graph size"""
    G = Graph()
    random.seed(42)

    for i in range(size):
        G.add_node(f"n{i}")

    for i in range(size * 5):
        src = f"n{random.randint(0, size - 1)}"
        tgt = f"n{random.randint(0, size - 1)}"
        G.add_edge(src, tgt)

    result = benchmark(lambda: G.pagerank(return_dict=True))
    assert len(result) == size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])



def test_all_pairs_weighted_400_nodes(benchmark):
    """Benchmark weighted all-pairs Dijkstra on 400 nodes / 1600 edges.

    Edge weights come from an O(1) index; the old linear scan per relaxation made this
    roughly 20x slower (~0.9s vs ~0.05s per run).
    """
    rng = random.Random(0)
    G = Graph(weighted=True)
    for i in range(400):
        G.add_node(f"n{i}")
    for _ in range(1600):
        G.add_edge(f"n{rng.randrange(400)}", f"n{rng.randrange(400)}", weight=rng.uniform(1, 5))

    result = benchmark(G.all_pairs_shortest_path_length)
    assert len(result) == 400


def test_ada_star_search_80x80_grid(benchmark):
    """Benchmark ADA* across an 80x80 weighted grid (6,320 edges).

    Edge weights and predecessors come from the graph's edge index; scanning every edge for
    each weight and every node for each predecessor made this ~50x slower (~0.95s vs ~0.02s).
    """
    rng = random.Random(80)
    k = 80
    G = Graph(weighted=True)
    for r in range(k):
        for c in range(k):
            G.add_node(f"{r},{c}")
    for r in range(k):
        for c in range(k):
            if c + 1 < k:
                G.add_edge(f"{r},{c}", f"{r},{c + 1}", weight=rng.uniform(1, 5))
            if r + 1 < k:
                G.add_edge(f"{r},{c}", f"{r + 1},{c}", weight=rng.uniform(1, 5))

    path = benchmark(G.ada_star_search, "0,0", f"{k - 1},{k - 1}")
    assert path[0] == G.get_id_from_label("0,0")
    assert path[-1] == G.get_id_from_label(f"{k - 1},{k - 1}")
