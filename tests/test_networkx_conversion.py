"""
Tests for NetworkX interoperability.

Tests bidirectional conversion between EdgeCase and NetworkX graphs,
ensuring nodes, edges, and attributes are preserved correctly.
"""

import pytest
import sys


# Check if NetworkX is available
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

pytestmark = pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX is not installed")


from edgecase import Graph
from edgecase.networkx import to_networkx, from_networkx


class TestToNetworkX:
    """Tests for converting EdgeCase Graph to NetworkX."""

    def test_empty_graph_undirected(self):
        """Convert empty undirected graph."""
        G = Graph(directed=False)
        nx_graph = to_networkx(G)

        assert isinstance(nx_graph, nx.Graph)
        assert not isinstance(nx_graph, nx.DiGraph)
        assert nx_graph.number_of_nodes() == 0
        assert nx_graph.number_of_edges() == 0

    def test_empty_graph_directed(self):
        """Convert empty directed graph."""
        G = Graph(directed=True)
        nx_graph = to_networkx(G)

        assert isinstance(nx_graph, nx.DiGraph)
        assert nx_graph.number_of_nodes() == 0
        assert nx_graph.number_of_edges() == 0

    def test_nodes_without_attributes(self):
        """Convert graph with nodes but no attributes."""
        G = Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        nx_graph = to_networkx(G)

        assert nx_graph.number_of_nodes() == 3
        assert set(nx_graph.nodes()) == {"A", "B", "C"}

    def test_nodes_with_attributes(self):
        """Convert graph with node attributes."""
        G = Graph(directed=False)
        G.add_node("A", color="red", value=10, active=True)
        G.add_node("B", color="blue", value=20, active=False)

        nx_graph = to_networkx(G)

        assert nx_graph.nodes["A"]["color"] == "red"
        assert nx_graph.nodes["A"]["value"] == 10
        assert nx_graph.nodes["A"]["active"] is True
        assert nx_graph.nodes["B"]["color"] == "blue"
        assert nx_graph.nodes["B"]["value"] == 20
        assert nx_graph.nodes["B"]["active"] is False

    def test_edges_undirected(self):
        """Convert undirected graph with edges."""
        G = Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        nx_graph = to_networkx(G)

        assert nx_graph.number_of_edges() == 2
        assert nx_graph.has_edge("A", "B")
        assert nx_graph.has_edge("B", "C")
        # Undirected edges work both ways
        assert nx_graph.has_edge("B", "A")
        assert nx_graph.has_edge("C", "B")

    def test_edges_directed(self):
        """Convert directed graph with edges."""
        G = Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        nx_graph = to_networkx(G)

        assert nx_graph.number_of_edges() == 2
        assert nx_graph.has_edge("A", "B")
        assert nx_graph.has_edge("B", "C")
        # Directed edges only work one way
        assert not nx_graph.has_edge("B", "A")
        assert not nx_graph.has_edge("C", "B")

    def test_edges_with_weights(self):
        """Convert graph with weighted edges."""
        G = Graph(directed=True, weighted=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B", weight=1.5)
        G.add_edge("B", "C", weight=2.7)

        nx_graph = to_networkx(G)

        assert nx_graph["A"]["B"]["weight"] == 1.5
        assert nx_graph["B"]["C"]["weight"] == 2.7

    def test_edges_with_attributes(self):
        """Convert graph with edge attributes."""
        G = Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B", weight=1.0, label="edge1", type="friend")
        G.add_edge("B", "C", weight=2.0, label="edge2", type="colleague")

        nx_graph = to_networkx(G)

        assert nx_graph["A"]["B"]["weight"] == 1.0
        assert nx_graph["A"]["B"]["label"] == "edge1"
        assert nx_graph["A"]["B"]["type"] == "friend"
        assert nx_graph["B"]["C"]["weight"] == 2.0
        assert nx_graph["B"]["C"]["label"] == "edge2"
        assert nx_graph["B"]["C"]["type"] == "colleague"

    def test_complete_graph(self):
        """Convert a complete graph with nodes and edges."""
        G = Graph(directed=True)
        G.add_node("A", description="Node A", value=1)
        G.add_node("B", description="Node B", value=2)
        G.add_node("C", description="Node C", value=3)
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=2.0)
        G.add_edge("A", "C", weight=3.0)

        nx_graph = to_networkx(G)

        # Check nodes
        assert nx_graph.number_of_nodes() == 3
        assert nx_graph.nodes["A"]["description"] == "Node A"
        assert nx_graph.nodes["B"]["value"] == 2

        # Check edges
        assert nx_graph.number_of_edges() == 3
        assert nx_graph["A"]["B"]["weight"] == 1.0
        assert nx_graph["B"]["C"]["weight"] == 2.0
        assert nx_graph["A"]["C"]["weight"] == 3.0

    def test_multigraph_conversion(self):
        """Convert graph with multiple edges using multigraph=True."""
        G = Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B", weight=1.0, label="edge1")
        G.add_edge("A", "B", weight=2.0, label="edge2")

        nx_graph = to_networkx(G, multigraph=True)

        assert isinstance(nx_graph, nx.MultiGraph)
        assert nx_graph.number_of_nodes() == 2
        assert nx_graph.number_of_edges() == 2

        # Check both edges exist
        edges = list(nx_graph["A"]["B"].values())
        assert len(edges) == 2
        weights = [e["weight"] for e in edges]
        labels = [e["label"] for e in edges]
        assert sorted(weights) == [1.0, 2.0]
        assert sorted(labels) == ["edge1", "edge2"]

    def test_multigraph_warning(self):
        """Test that warning is issued for multiple edges without multigraph=True."""
        G = Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("A", "B", weight=2.0)

        with pytest.warns(UserWarning, match="multiple edges"):
            nx_graph = to_networkx(G, multigraph=False)

        # Should only have one edge preserved (the last one)
        assert nx_graph.number_of_edges() == 1


class TestFromNetworkX:
    """Tests for converting NetworkX graph to EdgeCase Graph."""

    def test_empty_graph_undirected(self):
        """Convert empty undirected NetworkX graph."""
        nx_graph = nx.Graph()
        G = from_networkx(nx_graph)

        assert not G.is_directed
        assert G.num_nodes() == 0
        assert G.num_edges() == 0

    def test_empty_graph_directed(self):
        """Convert empty directed NetworkX graph."""
        nx_graph = nx.DiGraph()
        G = from_networkx(nx_graph)

        assert G.is_directed
        assert G.num_nodes() == 0
        assert G.num_edges() == 0

    def test_nodes_without_attributes(self):
        """Convert NetworkX graph with nodes but no attributes."""
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(["A", "B", "C"])

        G = from_networkx(nx_graph)

        assert G.num_nodes() == 3
        node_labels = [G.get_label_from_id(nid) for nid in G.node_ids(limit=None)]
        assert set(node_labels) == {"A", "B", "C"}

    def test_nodes_with_attributes(self):
        """Convert NetworkX graph with node attributes."""
        nx_graph = nx.Graph()
        nx_graph.add_node("A", color="red", value=10)
        nx_graph.add_node("B", color="blue", value=20)

        G = from_networkx(nx_graph)

        a_id = G.get_id_from_label("A")
        b_id = G.get_id_from_label("B")
        assert G.get_node(a_id).attributes["color"] == "red"
        assert G.get_node(a_id).attributes["value"] == 10
        assert G.get_node(b_id).attributes["color"] == "blue"
        assert G.get_node(b_id).attributes["value"] == 20

    def test_edges_undirected(self):
        """Convert undirected NetworkX graph with edges."""
        nx_graph = nx.Graph()
        nx_graph.add_edge("A", "B")
        nx_graph.add_edge("B", "C")

        G = from_networkx(nx_graph)

        assert not G.is_directed
        assert G.num_edges() == 2
        # Check both directions work (undirected)
        assert len(G.find_edge_ids("A", "B")) > 0 or len(G.find_edge_ids("B", "A")) > 0
        assert len(G.find_edge_ids("B", "C")) > 0 or len(G.find_edge_ids("C", "B")) > 0

    def test_edges_directed(self):
        """Convert directed NetworkX graph with edges."""
        nx_graph = nx.DiGraph()
        nx_graph.add_edge("A", "B")
        nx_graph.add_edge("B", "C")

        G = from_networkx(nx_graph)

        assert G.is_directed
        assert G.num_edges() == 2
        assert len(G.find_edge_ids("A", "B")) == 1
        assert len(G.find_edge_ids("B", "C")) == 1
        # Reverse edges should not exist
        assert len(G.find_edge_ids("B", "A")) == 0
        assert len(G.find_edge_ids("C", "B")) == 0

    def test_edges_with_weights(self):
        """Convert NetworkX graph with weighted edges."""
        nx_graph = nx.DiGraph()
        nx_graph.add_edge("A", "B", weight=1.5)
        nx_graph.add_edge("B", "C", weight=2.7)

        G = from_networkx(nx_graph)

        assert G.is_weighted
        edge_ids_ab = G.find_edge_ids("A", "B")
        assert len(edge_ids_ab) == 1
        assert G.get_edge(edge_ids_ab[0]).weight == 1.5

        edge_ids_bc = G.find_edge_ids("B", "C")
        assert len(edge_ids_bc) == 1
        assert G.get_edge(edge_ids_bc[0]).weight == 2.7

    def test_edges_with_attributes(self):
        """Convert NetworkX graph with edge attributes."""
        nx_graph = nx.Graph()
        nx_graph.add_edge("A", "B", weight=1.0, label="edge1", type="friend")
        nx_graph.add_edge("B", "C", weight=2.0, label="edge2", type="colleague")

        G = from_networkx(nx_graph)

        edge_ids_ab = G.find_edge_ids("A", "B")
        if len(edge_ids_ab) == 0:
            edge_ids_ab = G.find_edge_ids("B", "A")
        assert len(edge_ids_ab) == 1
        edge_ab = G.get_edge(edge_ids_ab[0])
        assert edge_ab.weight == 1.0
        assert edge_ab.attributes and edge_ab.attributes.get("label") == "edge1"
        assert edge_ab.attributes and edge_ab.attributes.get("type") == "friend"

    def test_multigraph_conversion(self):
        """Convert NetworkX MultiGraph with multiple edges."""
        nx_graph = nx.MultiGraph()
        nx_graph.add_edge("A", "B", weight=1.0, label="edge1")
        nx_graph.add_edge("A", "B", weight=2.0, label="edge2")

        G = from_networkx(nx_graph)

        assert G.num_nodes() == 2
        assert G.num_edges() == 2

        # Check both edges exist
        edge_ids = G.find_edge_ids("A", "B")
        if len(edge_ids) == 0:
            edge_ids = G.find_edge_ids("B", "A")
        assert len(edge_ids) == 2

    def test_weighted_auto_detection(self):
        """Test automatic detection of weighted graphs."""
        # Graph with weights
        nx_graph = nx.Graph()
        nx_graph.add_edge("A", "B", weight=1.5)
        G = from_networkx(nx_graph)
        assert G.is_weighted

        # Graph without weights
        nx_graph = nx.Graph()
        nx_graph.add_edge("A", "B")
        G = from_networkx(nx_graph)
        assert not G.is_weighted

    def test_numeric_node_ids(self):
        """Test conversion with numeric node IDs."""
        nx_graph = nx.Graph()
        nx_graph.add_edge(1, 2, weight=1.0)
        nx_graph.add_edge(2, 3, weight=2.0)

        G = from_networkx(nx_graph)

        # Node IDs should be converted to strings
        node_labels = [G.get_label_from_id(nid) for nid in G.node_ids(limit=None)]
        assert "1" in node_labels
        assert "2" in node_labels
        assert "3" in node_labels
        assert G.num_edges() == 2


class TestRoundTrip:
    """Tests for round-trip conversion between EdgeCase and NetworkX."""

    def test_undirected_graph_roundtrip(self):
        """Test EdgeCase -> NetworkX -> EdgeCase preserves graph."""
        G1 = Graph(directed=False)
        G1.add_node("A", color="red")
        G1.add_node("B", color="blue")
        G1.add_edge("A", "B", weight=1.5, label="test")

        nx_graph = to_networkx(G1)
        G2 = from_networkx(nx_graph)

        assert G2.num_nodes() == G1.num_nodes()
        assert G2.num_edges() == G1.num_edges()
        assert G2.is_directed == G1.is_directed
        a_id = G2.get_id_from_label("A")
        b_id = G2.get_id_from_label("B")
        assert G2.get_node(a_id).attributes["color"] == "red"
        assert G2.get_node(b_id).attributes["color"] == "blue"

    def test_directed_graph_roundtrip(self):
        """Test EdgeCase -> NetworkX -> EdgeCase preserves directed graph."""
        G1 = Graph(directed=True)
        G1.add_node("X", value=10)
        G1.add_node("Y", value=20)
        G1.add_edge("X", "Y", weight=2.5)

        nx_graph = to_networkx(G1)
        G2 = from_networkx(nx_graph)

        assert G2.num_nodes() == G1.num_nodes()
        assert G2.num_edges() == G1.num_edges()
        assert G2.is_directed == G1.is_directed
        assert len(G2.find_edge_ids("X", "Y")) == 1
        assert len(G2.find_edge_ids("Y", "X")) == 0

    def test_networkx_roundtrip(self):
        """Test NetworkX -> EdgeCase -> NetworkX preserves graph."""
        nx1 = nx.DiGraph()
        nx1.add_node("A", description="Node A")
        nx1.add_node("B", description="Node B")
        nx1.add_edge("A", "B", weight=1.0, type="test")

        G = from_networkx(nx1)
        nx2 = to_networkx(G)

        assert nx2.number_of_nodes() == nx1.number_of_nodes()
        assert nx2.number_of_edges() == nx1.number_of_edges()
        assert nx2.is_directed() == nx1.is_directed()
        assert nx2.nodes["A"]["description"] == "Node A"
        assert nx2.nodes["B"]["description"] == "Node B"
        assert nx2["A"]["B"]["weight"] == 1.0
        assert nx2["A"]["B"]["type"] == "test"

    def test_complex_graph_roundtrip(self):
        """Test round-trip with complex graph structure."""
        G1 = Graph(directed=True)
        # Add multiple nodes with various attributes
        for i in range(5):
            G1.add_node(f"N{i}", value=i * 10, description=f"Node {i}")

        # Add edges with attributes
        G1.add_edge("N0", "N1", weight=1.0, type="A")
        G1.add_edge("N1", "N2", weight=2.0, type="B")
        G1.add_edge("N2", "N3", weight=3.0, type="A")
        G1.add_edge("N3", "N4", weight=4.0, type="C")
        G1.add_edge("N0", "N4", weight=5.0, type="A")

        # Round trip
        nx_graph = to_networkx(G1)
        G2 = from_networkx(nx_graph)

        assert G2.num_nodes() == 5
        assert G2.num_edges() == 5
        assert G2.is_directed

        # Check a sample of attributes
        n0_id = G2.get_id_from_label("N0")
        n3_id = G2.get_id_from_label("N3")
        assert G2.get_node(n0_id).attributes["value"] == 0
        assert G2.get_node(n3_id).attributes["description"] == "Node 3"

        edge_ids = G2.find_edge_ids("N2", "N3")
        assert len(edge_ids) == 1
        assert G2.get_edge(edge_ids[0]).attributes.get("type") == "A"


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_missing_networkx_import(self):
        """Test that ImportError is raised when NetworkX is not available."""
        # This test is tricky since we need NetworkX to run tests
        # We can at least verify the error message exists
        pass  # Skip for now

    def test_self_loops(self):
        """Test conversion of graphs with self-loops."""
        G = Graph(directed=True)
        G.add_node("A")
        G.add_edge("A", "A", weight=1.0)  # Self-loop

        nx_graph = to_networkx(G)

        assert nx.number_of_selfloops(nx_graph) == 1
        assert nx_graph.has_edge("A", "A")

        # Round trip
        G2 = from_networkx(nx_graph)
        assert G2.num_edges() == 1
