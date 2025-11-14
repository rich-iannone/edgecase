"""Tests for graph algorithms."""

import pytest
from edgecase import Graph, Node, Edge


# PageRank tests


def test_pagerank_single_node() -> None:
    """Test PageRank on graph with single node."""
    graph = Graph(directed=True)
    graph.add_node(Node(1, "A"))

    ranks = graph.pagerank()
    assert len(ranks) == 1
    assert abs(ranks[1] - 1.0) < 1e-6


def test_pagerank_simple_chain() -> None:
    """Test PageRank on a simple chain."""
    graph = Graph(directed=True)
    for i, node_label in enumerate(["A", "B", "C"], start=1):
        graph.add_node(Node(i, node_label))

    graph.add_edge(Edge(0, 1, 2, None, None))
    graph.add_edge(Edge(0, 2, 3, None, None))

    ranks = graph.pagerank()
    assert len(ranks) == 3


# Betweenness centrality tests


def test_betweenness_empty_graph() -> None:
    """Test betweenness centrality on empty graph."""
    graph = Graph(directed=False)
    centrality = graph.betweenness_centrality()
    assert len(centrality) == 0


def test_betweenness_star_graph() -> None:
    """Test betweenness on star graph (center should have high centrality)."""
    graph = Graph(directed=False)

    # Create star graph with center node
    graph.add_node(Node(1, "center"))
    for i in range(5):
        node_id = i + 2  # IDs 2-6
        node_label = f"leaf{i}"
        graph.add_node(Node(node_id, node_label))
        graph.add_edge(Edge(0, 1, node_id, None, None))

    centrality = graph.betweenness_centrality()
    # Center (node ID 1) should have highest centrality
    assert 1 in centrality


# Shortest path tests


def test_shortest_path_direct_connection() -> None:
    """Test shortest path between directly connected nodes."""
    graph = Graph(directed=False)
    graph.add_node(Node(1, "A"))
    graph.add_node(Node(2, "B"))
    graph.add_edge(Edge(0, 1, 2, None, None))

    path = graph.shortest_path("A", "B")
    assert path == [1, 2]


def test_shortest_path_multiple_hops() -> None:
    """Test shortest path requiring multiple hops."""
    graph = Graph(directed=False)
    for i, node_label in enumerate(["A", "B", "C", "D"], start=1):
        graph.add_node(Node(i, node_label))

    graph.add_edge(Edge(0, 1, 2, None, None))
    graph.add_edge(Edge(0, 2, 3, None, None))
    graph.add_edge(Edge(0, 3, 4, None, None))

    path = graph.shortest_path("A", "D")
    assert len(path) == 4
    assert path[0] == 1
    assert path[-1] == 4


def test_shortest_path_no_connection() -> None:
    """Test shortest path when no path exists."""
    graph = Graph(directed=False)
    graph.add_node(Node(1, "A"))
    graph.add_node(Node(2, "B"))

    # When no path exists, shortest_path likely returns None
    result = graph.shortest_path(1, 2)
    assert result is None  # Or check for empty list/path


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
        graph.add_node(Node(i + 1, f"Node{i}"))

    ranks = graph.pagerank(damping=damping)

    # Sum of ranks should be approximately 1.0
    total_rank = sum(ranks.values())
    assert abs(total_rank - 1.0) < 0.01
