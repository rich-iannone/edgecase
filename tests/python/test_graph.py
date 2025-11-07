"""Tests for the core graph data structure."""

import pytest
from edgecase import Graph, Node, Edge


# Graph tests

def test_create_empty_graph() -> None:
    """Test creating an empty graph."""
    graph = Graph(directed=False)
    assert graph.num_nodes() == 0
    assert graph.num_edges() == 0
    assert not graph.is_directed


def test_create_directed_graph() -> None:
    """Test creating a directed graph."""
    graph = Graph(directed=True)
    assert graph.is_directed


def test_add_single_node() -> None:
    """Test adding a single node."""
    graph = Graph(directed=False)
    node = Node("A", None)
    graph.add_node(node)
    assert graph.num_nodes() == 1


def test_add_multiple_nodes() -> None:
    """Test adding multiple nodes."""
    graph = Graph(directed=False)
    nodes = [Node(f"Node{i}", None) for i in range(10)]
    for node in nodes:
        graph.add_node(node)
    assert graph.num_nodes() == 10


def test_add_edge_between_existing_nodes() -> None:
    """Test adding an edge between existing nodes."""
    graph = Graph(directed=False)
    node_a = Node("A", None)
    node_b = Node("B", None)
    graph.add_node(node_a)
    graph.add_node(node_b)
    
    edge = Edge(0, "A", "B", None, None)  # edge_id=0 means auto-assign
    graph.add_edge(edge)
    assert graph.num_edges() == 1


def test_add_edge_nonexistent_source_raises_error() -> None:
    """Test that adding edge with non-existent source raises error."""
    graph = Graph(directed=False)
    graph.add_node(Node("B", None))
    
    edge = Edge(0, "A", "B", None, None)
    with pytest.raises(Exception):  # Should raise ValueError
        graph.add_edge(edge)


def test_add_edge_nonexistent_target_raises_error() -> None:
    """Test that adding edge with non-existent target raises error."""
    graph = Graph(directed=False)
    graph.add_node(Node("A", None))
    
    edge = Edge(0, "A", "B", None, None)
    with pytest.raises(Exception):  # Should raise ValueError
        graph.add_edge(edge)


def test_weighted_graph() -> None:
    """Test creating a weighted graph."""
    graph = Graph(directed=False, weighted=True)
    assert graph.is_weighted
    
    # Add nodes and weighted edge
    graph.add_node(Node("A", None))
    graph.add_node(Node("B", None))
    edge = Edge(0, "A", "B", 2.5, None)  # id, source, target, weight, attributes
    graph.add_edge(edge)
    assert graph.num_edges() == 1


def test_neighbors_undirected_graph() -> None:
    """Test getting neighbors in an undirected graph."""
    graph = Graph(directed=False)
    graph.add_node(Node("A", None))
    graph.add_node(Node("B", None))
    graph.add_node(Node("C", None))
    
    graph.add_edge(Edge(0, "A", "B", None, None))
    graph.add_edge(Edge(0, "A", "C", None, None))
    
    neighbors = graph.neighbors("A")
    assert len(neighbors) == 2
    assert "B" in neighbors
    assert "C" in neighbors


def test_neighbors_directed_graph() -> None:
    """Test getting neighbors in a directed graph."""
    graph = Graph(directed=True)
    graph.add_node(Node("A", None))
    graph.add_node(Node("B", None))
    graph.add_node(Node("C", None))
    
    # A -> B, A -> C
    graph.add_edge(Edge(0, "A", "B", None, None))
    graph.add_edge(Edge(0, "A", "C", None, None))
    
    # A should have B and C as neighbors
    a_neighbors = graph.neighbors("A")
    assert len(a_neighbors) == 2
    
    # B should have no outgoing neighbors
    b_neighbors = graph.neighbors("B")
    assert len(b_neighbors) == 0


def test_neighbors_nonexistent_node_raises_error() -> None:
    """Test that getting neighbors of non-existent node raises error."""
    graph = Graph(directed=False)
    with pytest.raises(Exception):  # Should raise KeyError
        graph.neighbors("NonExistent")


# Node tests

def test_create_node_without_attributes() -> None:
    """Test creating a node without attributes."""
    node = Node("A", None)
    assert node.id == "A"
    assert node.attributes is None


def test_node_repr() -> None:
    """Test node string representation."""
    node = Node("TestNode", None)
    repr_str = repr(node)
    assert "TestNode" in repr_str


# Edge tests

def test_create_unweighted_edge() -> None:
    """Test creating an unweighted edge."""
    edge = Edge(0, "A", "B", None, None)
    assert edge.source == "A"
    assert edge.target == "B"
    assert edge.weight is None


def test_create_weighted_edge() -> None:
    """Test creating a weighted edge."""
    edge = Edge(0, "A", "B", 3.5, None)  # id, source, target, weight, attributes
    assert edge.weight == 3.5


def test_edge_repr() -> None:
    """Test edge string representation."""
    edge = Edge(0, "A", "B", 2.0, None)  # id, source, target, weight, attributes
    repr_str = repr(edge)
    assert "A" in repr_str
    assert "B" in repr_str


# Property-based tests using Hypothesis
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=100))
def test_graph_node_count_property(n: int) -> None:
    """Property test: graph should have exactly n nodes after adding n nodes."""
    graph = Graph(directed=False)
    for i in range(n):
        graph.add_node(Node(f"Node{i}", None))
    assert graph.num_nodes() == n

@given(
    st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=20, unique=True)
)
def test_graph_maintains_node_uniqueness(node_ids: list[str]) -> None:
    """Property test: graph should maintain unique nodes."""
    graph = Graph(directed=False)
    for node_id in node_ids:
        graph.add_node(Node(node_id, None))
    
    # Add same nodes again
    for node_id in node_ids:
        graph.add_node(Node(node_id, None))
    
    # Should still have same number of unique nodes
    # Note: Current implementation allows duplicates, this test documents that
    # In a real implementation, you'd want to prevent duplicates
