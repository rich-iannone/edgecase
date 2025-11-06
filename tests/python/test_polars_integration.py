"""Tests for Polars integration."""

import pytest
import polars as pl
from edgecase import from_polars, to_polars, Graph


# from_polars tests

def test_from_polars_simple_graph() -> None:
    """Test creating a graph from a simple DataFrame."""
    df = pl.DataFrame({
        "from": ["A", "B", "C"],
        "to": ["B", "C", "A"],
    })
    
    graph = from_polars(df, source_col="from", target_col="to")
    
    assert graph.num_nodes() == 3
    assert graph.num_edges() == 3


def test_from_polars_with_weights() -> None:
    """Test creating a weighted graph from DataFrame."""
    df = pl.DataFrame({
        "source": ["A", "B"],
        "target": ["B", "C"],
        "weight": [1.5, 2.5],
    })
    
    graph = from_polars(
        df,
        source_col="source",
        target_col="target",
        weight_col="weight",
    )
    
    assert graph.is_weighted
    assert graph.num_edges() == 2


def test_from_polars_directed() -> None:
    """Test creating a directed graph."""
    df = pl.DataFrame({
        "from": ["A", "B"],
        "to": ["B", "C"],
    })
    
    graph = from_polars(df, source_col="from", target_col="to", directed=True)
    assert graph.is_directed


def test_from_polars_lazyframe() -> None:
    """Test creating graph from LazyFrame."""
    df = pl.DataFrame({
        "from": ["A", "B"],
        "to": ["B", "C"],
    })
    lazy_df = df.lazy()
    
    graph = from_polars(lazy_df, source_col="from", target_col="to")
    assert graph.num_nodes() > 0


def test_from_polars_empty_dataframe() -> None:
    """Test handling empty DataFrame."""
    df = pl.DataFrame({
        "from": [],
        "to": [],
    })
    
    graph = from_polars(df, source_col="from", target_col="to")
    assert graph.num_nodes() == 0
    assert graph.num_edges() == 0


def test_from_polars_self_loops() -> None:
    """Test graph with self-loops."""
    df = pl.DataFrame({
        "from": ["A", "A"],
        "to": ["A", "B"],
    })
    
    graph = from_polars(df, source_col="from", target_col="to")
    assert graph.num_edges() == 2


# to_polars tests

def test_to_polars_basic() -> None:
    """Test converting graph to DataFrame."""
    graph = Graph(directed=False)
    # Add nodes and edges
    # ... 
    
    df = to_polars(graph)
    assert isinstance(df, pl.DataFrame)


# Benchmark tests
@pytest.mark.benchmark
def test_from_polars_performance_large_graph(benchmark) -> None:
    """Benchmark creating graph from large DataFrame."""
    # Create large test data
    n = 10000
    df = pl.DataFrame({
        "from": [f"Node{i}" for i in range(n)],
        "to": [f"Node{(i+1) % n}" for i in range(n)],
    })
    
    def create_graph() -> Graph:
        return from_polars(df, source_col="from", target_col="to")
    
    result = benchmark(create_graph)
    assert result.num_nodes() == n
