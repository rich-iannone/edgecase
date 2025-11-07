# tests/unit/test_graph_basics.py
"""
Unit tests for basic graph operations
"""

import pytest
from edgecase import Graph


# Graph creation tests

def test_empty_graph(empty_graph):
    """Empty graph should have 0 nodes and 0 edges"""
    assert empty_graph.num_nodes() == 0
    assert empty_graph.num_edges() == 0


def test_directed_graph_creation():
    """Test creating directed graph"""
    G = Graph(directed=True)
    assert G.directed


def test_undirected_graph_creation():
    """Test creating undirected graph"""
    G = Graph(directed=False)
    assert not G.directed


def test_weighted_graph_creation():
    """Test creating weighted graph"""
    G = Graph(weighted=True)
    assert G.weighted


# Node operation tests

def test_add_single_node(empty_graph):
    """Adding node should increase count"""
    empty_graph.add_node("A")
    assert empty_graph.num_nodes() == 1


def test_add_duplicate_node(empty_graph):
    """Adding duplicate node should not increase count"""
    empty_graph.add_node("A")
    empty_graph.add_node("A")
    assert empty_graph.num_nodes() == 1


def test_add_multiple_nodes(empty_graph):
    """Adding multiple nodes"""
    for i in range(10):
        empty_graph.add_node(f"node_{i}")
    assert empty_graph.num_nodes() == 10


def test_nodes_returns_list(simple_graph):
    """nodes() should return list of node IDs"""
    nodes = simple_graph.nodes()
    assert isinstance(nodes, list)
    assert set(nodes) == {"A", "B", "C"}


# Edge operation tests

def test_add_edge_creates_nodes(empty_graph):
    """Adding edge should implicitly create nodes"""
    empty_graph.add_edge("A", "B")
    assert empty_graph.num_nodes() == 2
    assert empty_graph.num_edges() == 1


def test_add_weighted_edge():
    """Adding weighted edge"""
    G = Graph(weighted=True)
    G.add_edge("A", "B", weight=5.0)
    assert G.num_edges() == 1


def test_add_duplicate_edge_undirected():
    """Adding duplicate edge in undirected graph"""
    G = Graph(directed=False)
    G.add_edge("A", "B")
    G.add_edge("B", "A")  # Same edge in undirected graph
    # Implementation dependent - could be 1 or 2
    assert G.num_edges() in [1, 2]


def test_add_duplicate_edge_directed():
    """Adding duplicate edge in directed graph"""
    G = Graph(directed=True)
    G.add_edge("A", "B")
    G.add_edge("A", "B")  # Second edge between same nodes
    # With edge IDs, both edges are preserved (multi-edge support)
    assert G.num_edges() == 2
    # Each edge gets a unique ID
    edge_ids = G.edge_ids()
    assert len(edge_ids) == 2
    assert edge_ids[0] != edge_ids[1]


def test_edges_returns_list(simple_graph):
    """edges() should return list of edges"""
    edges = simple_graph.edges()
    assert isinstance(edges, list)
    assert len(edges) == 2


# Neighbor query tests

def test_neighbors_simple(simple_graph):
    """Test basic neighbor query"""
    neighbors = simple_graph.neighbors("A")
    assert "B" in neighbors


def test_neighbors_nonexistent_node(empty_graph):
    """Querying neighbors of nonexistent node should return empty or raise"""
    try:
        neighbors = empty_graph.neighbors("X")
        assert neighbors == [] or neighbors is None
    except (KeyError, ValueError):
        pass  # Also acceptable


def test_undirected_neighbors_symmetric():
    """In undirected graph, neighbors should be symmetric"""
    G = Graph(directed=False)
    G.add_edge("A", "B")
    
    assert "B" in G.neighbors("A")
    assert "A" in G.neighbors("B")


def test_directed_neighbors_asymmetric():
    """In directed graph, neighbors should be asymmetric"""
    G = Graph(directed=True)
    G.add_edge("A", "B")
    
    assert "B" in G.neighbors("A")
    # B might not have A as neighbor (depends on API)


# Graph property tests

def test_num_nodes_matches_added(small_graph):
    """num_nodes() should match actual node count"""
    nodes = small_graph.nodes()
    assert small_graph.num_nodes() == len(nodes)


def test_num_edges_matches_added(small_graph):
    """num_edges() should match actual edge count"""
    edges = small_graph.edges()
    assert small_graph.num_edges() == len(edges)


def test_empty_graph_properties(empty_graph):
    """Empty graph should have zero counts"""
    assert empty_graph.num_nodes() == 0
    assert empty_graph.num_edges() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
