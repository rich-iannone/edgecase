# tests/benchmarks/test_performance.py
"""
Benchmark tests using pytest-benchmark
"""

import pytest
from edgecase import Graph, shortest_path, pagerank, from_polars
import polars as pl
import random


@pytest.fixture
def benchmark_graph_small():
    """1K node graph for benchmarking"""
    G = Graph()
    for i in range(1000):
        G.add_edge(f"n{i}", f"n{(i+1) % 1000}")
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
    assert G.num_nodes() == 1000


def test_add_10000_edges(benchmark):
    """Benchmark adding 10000 edges"""
    def add_edges():
        G = Graph()
        for i in range(10000):
            G.add_edge(f"n{i % 1000}", f"n{(i+1) % 1000}")
        return G
    
    G = benchmark(add_edges)
    assert G.num_edges() > 0


# Algorithm performance tests

def test_pagerank_1k_nodes(benchmark, benchmark_graph_small):
    """Benchmark PageRank on 1K node graph"""
    result = benchmark(pagerank, benchmark_graph_small)
    assert len(result) == 1000


def test_pagerank_10k_nodes(benchmark, benchmark_graph_medium):
    """Benchmark PageRank on 10K node graph"""
    result = benchmark(pagerank, benchmark_graph_medium)
    assert len(result) == 10000


def test_shortest_path_1k_nodes(benchmark, benchmark_graph_small):
    """Benchmark shortest path on 1K node graph"""
    result = benchmark(shortest_path, benchmark_graph_small, "n0", "n500")
    assert result is not None


# Polars integration performance tests

def test_from_polars_1k_edges(benchmark):
    """Benchmark from_polars with 1K edges"""
    df = pl.DataFrame({
        "source": [f"n{i}" for i in range(1000)],
        "target": [f"n{(i+1) % 1000}" for i in range(1000)],
        "weight": [1.0] * 1000,
    })
    
    G = benchmark(from_polars, df, "source", "target", "weight")
    assert G.num_edges() == 1000


def test_from_polars_10k_edges(benchmark):
    """Benchmark from_polars with 10K edges"""
    random.seed(42)
    
    df = pl.DataFrame({
        "source": [f"n{random.randint(0, 1000)}" for _ in range(10000)],
        "target": [f"n{random.randint(0, 1000)}" for _ in range(10000)],
        "weight": [random.random() for _ in range(10000)],
    })
    
    G = benchmark(from_polars, df, "source", "target", "weight")
    assert G.num_edges() > 0


# Parameterized benchmarks for scaling analysis
@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_pagerank_scaling(benchmark, size):
    """Test PageRank scaling with graph size"""
    G = Graph()
    random.seed(42)
    
    for i in range(size):
        G.add_node(f"n{i}")
    
    for i in range(size * 5):
        src = f"n{random.randint(0, size-1)}"
        tgt = f"n{random.randint(0, size-1)}"
        G.add_edge(src, tgt)
    
    result = benchmark(pagerank, G)
    assert len(result) == size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
