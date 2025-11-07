# Shared fixtures and configuration for pytest

import pytest
import polars as pl
from edgecase import Graph, Node, Edge


@pytest.fixture
def simple_graph() -> Graph:
    """Fixture providing a simple undirected graph."""
    graph = Graph(directed=False)
    for node_id in ["A", "B", "C", "D"]:
        graph.add_node(Node(node_id, None))
    
    graph.add_edge(Edge(0, "A", "B", None, None))
    graph.add_edge(Edge(0, "B", "C", None, None))
    graph.add_edge(Edge(0, "C", "D", None, None))
    
    return graph


@pytest.fixture
def directed_graph() -> Graph:
    """Fixture providing a simple directed graph."""
    graph = Graph(directed=True)
    for node_id in ["A", "B", "C"]:
        graph.add_node(Node(node_id, None))
    
    graph.add_edge(Edge(0, "A", "B", None, None))
    graph.add_edge(Edge(0, "B", "C", None, None))
    
    return graph


@pytest.fixture
def weighted_graph() -> Graph:
    """Fixture providing a weighted graph."""
    graph = Graph(directed=False, weighted=True)
    for node_id in ["A", "B", "C"]:
        graph.add_node(Node(node_id, None))
    
    graph.add_edge(Edge(0, "A", "B", 1.5, None))  # id, source, target, weight, attributes
    graph.add_edge(Edge(0, "B", "C", 2.5, None))  # id, source, target, weight, attributes
    
    return graph


@pytest.fixture
def sample_transactions() -> pl.DataFrame:
    """Fixture providing sample transaction data."""
    return pl.DataFrame({
        "from_account": ["A1", "A2", "A3", "A4"],
        "to_account": ["A2", "A3", "A4", "A1"],
        "amount": [1000.0, 2000.0, 1500.0, 3000.0],
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    })


@pytest.fixture
def sample_protein_interactions() -> pl.DataFrame:
    """Fixture providing sample protein interaction data."""
    return pl.DataFrame({
        "protein_a": ["P53", "BRCA1", "ATM"],
        "protein_b": ["MDM2", "P53", "BRCA1"],
        "confidence": [0.95, 0.88, 0.76],
    })
