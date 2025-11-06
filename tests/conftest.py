# tests/conftest.py
"""
Shared pytest fixtures for EdgeCase tests
"""

import sys
from pathlib import Path

# Add python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import pytest
from edgecase import Graph
import polars as pl
import random


@pytest.fixture
def empty_graph():
    """Empty graph for testing initialization"""
    return Graph()


@pytest.fixture
def simple_graph():
    """Simple 3-node graph: A → B → C"""
    G = Graph(directed=True)
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    return G


@pytest.fixture
def small_graph():
    """Small graph for quick tests (10 nodes, ring topology)"""
    G = Graph()
    for i in range(10):
        G.add_edge(f"node_{i}", f"node_{(i+1)%10}")
    return G


@pytest.fixture
def triangle_graph():
    """Triangle graph: A-B-C-A"""
    G = Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")
    return G


@pytest.fixture
def star_graph():
    """Star graph: center connected to 5 outer nodes"""
    G = Graph()
    for i in range(5):
        G.add_edge("center", f"outer_{i}")
    return G


@pytest.fixture(params=[True, False])
def directed_empty_graph(request):
    """Parameterized fixture for directed/undirected empty graphs"""
    return Graph(directed=request.param)


@pytest.fixture(params=[True, False])
def weighted_empty_graph(request):
    """Parameterized fixture for weighted/unweighted empty graphs"""
    return Graph(weighted=request.param)


@pytest.fixture
def sample_dataframe():
    """Sample Polars DataFrame for testing from_polars"""
    return pl.DataFrame({
        "source": ["A", "B", "C", "D"],
        "target": ["B", "C", "D", "A"],
        "weight": [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducible tests"""
    random.seed(42)
    yield 42


# Marker definitions
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "stress: marks stress tests (large graphs, high memory)"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks benchmark tests"
    )
