import sys
from pathlib import Path

# Add python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import pytest
from edgecase import Graph, Node, Edge
import polars as pl
import random


@pytest.fixture
def empty_graph():
    """Empty graph for testing initialization"""
    return Graph()


@pytest.fixture
def simple_graph():
    """Simple 3-node graph: A -> B -> C"""
    G = Graph(directed=True)
    G.add_node("A")
    G.add_node("B")
    G.add_node("C")
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    return G


@pytest.fixture
def small_graph():
    """Small graph for quick tests (10 nodes, ring topology)"""
    G = Graph()
    for i in range(10):
        G.add_node(f"node_{i}")
    for i in range(10):
        G.add_edge(f"node_{i}", f"node_{(i + 1) % 10}")
    return G


@pytest.fixture
def triangle_graph():
    """Triangle graph: A-B-C-A"""
    G = Graph()
    G.add_node("A")
    G.add_node("B")
    G.add_node("C")
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")
    return G


@pytest.fixture
def star_graph():
    """Star graph: center connected to 5 outer nodes"""
    G = Graph()
    G.add_node("center")
    for i in range(5):
        G.add_node(f"outer_{i}")
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
    """Sample Polars DataFrame for testing Graph constructor with edges_df"""
    return pl.DataFrame(
        {
            "source": ["A", "B", "C", "D"],
            "target": ["B", "C", "D", "A"],
            "weight": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducible tests"""
    random.seed(42)
    yield 42


@pytest.fixture
def simple_undirected_graph() -> Graph:
    """Fixture providing a simple undirected graph with explicit Node/Edge objects."""
    graph = Graph(directed=False)
    for node_id in ["A", "B", "C", "D"]:
        graph.add_node(Node(node_id, None))

    graph.add_edge(Edge(0, "A", "B", None, None))
    graph.add_edge(Edge(0, "B", "C", None, None))
    graph.add_edge(Edge(0, "C", "D", None, None))

    return graph


@pytest.fixture
def directed_graph() -> Graph:
    """Fixture providing a simple directed graph with explicit Node/Edge objects."""
    graph = Graph(directed=True)
    for node_id in ["A", "B", "C"]:
        graph.add_node(Node(node_id, None))

    graph.add_edge(Edge(0, "A", "B", None, None))
    graph.add_edge(Edge(0, "B", "C", None, None))

    return graph


@pytest.fixture
def weighted_graph() -> Graph:
    """Fixture providing a weighted graph with explicit Node/Edge objects."""
    graph = Graph(directed=False, weighted=True)
    for node_id in ["A", "B", "C"]:
        graph.add_node(Node(node_id, None))

    graph.add_edge(Edge(0, "A", "B", 1.5, None))
    graph.add_edge(Edge(0, "B", "C", 2.5, None))

    return graph


@pytest.fixture
def sample_transactions() -> pl.DataFrame:
    """Fixture providing sample transaction data."""
    return pl.DataFrame(
        {
            "from_account": ["A1", "A2", "A3", "A4"],
            "to_account": ["A2", "A3", "A4", "A1"],
            "amount": [1000.0, 2000.0, 1500.0, 3000.0],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        }
    )


@pytest.fixture
def sample_protein_interactions() -> pl.DataFrame:
    """Fixture providing sample protein interaction data."""
    return pl.DataFrame(
        {
            "protein_a": ["P53", "BRCA1", "ATM"],
            "protein_b": ["MDM2", "P53", "BRCA1"],
            "confidence": [0.95, 0.88, 0.76],
        }
    )


# Marker definitions
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "stress: marks stress tests (large graphs, high memory)")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
