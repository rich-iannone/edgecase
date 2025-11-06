"""Tests for graph algorithms."""

import pytest
from edgecase import Graph, Node, Edge, pagerank, betweenness_centrality, shortest_path


# PageRank tests

def test_pagerank_single_node() -> None:
    """Test PageRank on graph with single node."""
    graph = Graph(directed=True)
    graph.add_node(Node("A", None))
    
    ranks = pagerank(graph)
    assert len(ranks) == 1
    assert abs(ranks["A"] - 1.0) < 1e-6


def test_pagerank_simple_chain() -> None:
    """Test PageRank on a simple chain."""
    graph = Graph(directed=True)
    for node_id in ["A", "B", "C"]:
        graph.add_node(Node(node_id, None))
    
    graph.add_edge(Edge("A", "B", None, None))
    graph.add_edge(Edge("B", "C", None, None))
    
    ranks = pagerank(graph)
    assert len(ranks) == 3


# Betweenness centrality tests

def test_betweenness_empty_graph() -> None:
    """Test betweenness centrality on empty graph."""
    graph = Graph(directed=False)
    centrality = betweenness_centrality(graph)
    assert len(centrality) == 0


def test_betweenness_star_graph() -> None:
    """Test betweenness on star graph (center should have high centrality)."""
    graph = Graph(directed=False)
    
    # Create star graph with center node
    graph.add_node(Node("center", None))
    for i in range(5):
        node_id = f"leaf{i}"
        graph.add_node(Node(node_id, None))
        graph.add_edge(Edge("center", node_id, None, None))
    
    centrality = betweenness_centrality(graph)
    # Center should have highest centrality (placeholder checks existence)
    assert "center" in centrality


# Shortest path tests

def test_shortest_path_direct_connection() -> None:
    """Test shortest path between directly connected nodes."""
    graph = Graph(directed=False)
    graph.add_node(Node("A", None))
    graph.add_node(Node("B", None))
    graph.add_edge(Edge("A", "B", None, None))
    
    path = shortest_path(graph, "A", "B")
    assert path == ["A", "B"]


def test_shortest_path_multiple_hops() -> None:
    """Test shortest path requiring multiple hops."""
    graph = Graph(directed=False)
    for node_id in ["A", "B", "C", "D"]:
        graph.add_node(Node(node_id, None))
    
    graph.add_edge(Edge("A", "B", None, None))
    graph.add_edge(Edge("B", "C", None, None))
    graph.add_edge(Edge("C", "D", None, None))
    
    path = shortest_path(graph, "A", "D")
    assert len(path) == 4
    assert path[0] == "A"
    assert path[-1] == "D"


def test_shortest_path_no_connection() -> None:
    """Test shortest path when no path exists."""
    graph = Graph(directed=False)
    graph.add_node(Node("A", None))
    graph.add_node(Node("B", None))
    
    with pytest.raises(Exception):  # Should raise error
        shortest_path(graph, "A", "B")


# Property-based algorithm tests
from hypothesis import given, strategies as st, assume

@given(
    st.integers(min_value=1, max_value=20),
    st.floats(min_value=0.1, max_value=0.99),
)
def test_pagerank_damping_factor_property(n_nodes: int, damping: float) -> None:
    """Property test: PageRank should work with various damping factors."""
    graph = Graph(directed=True)
    for i in range(n_nodes):
        graph.add_node(Node(f"Node{i}", None))
    
    ranks = pagerank(graph, damping=damping)
    
    # Sum of ranks should be approximately 1.0
    total_rank = sum(ranks.values())
    assert abs(total_rank - 1.0) < 0.01
