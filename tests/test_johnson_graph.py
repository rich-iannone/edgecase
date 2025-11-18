"""Tests for Johnson graph generator."""

import pytest
from edgecase import Graph


class TestJohnsonGraph:
    """Test suite for Johnson graph J(n,k) generation."""

    def test_johnson_4_2_basic_properties(self):
        """Test J(4,2) has correct number of nodes and edges."""
        graph = Graph(directed=False).add_johnson_graph(n=4, k=2)

        # C(4,2) = 6 nodes
        assert graph.num_nodes() == 6

        # Each node has degree k(n-k) = 2*2 = 4
        # Total edges = (6 * 4) / 2 = 12
        assert graph.num_edges() == 12

    def test_johnson_4_2_node_labels(self):
        """Test J(4,2) has correct node labels (2-subsets)."""
        graph = Graph(directed=False).add_johnson_graph(n=4, k=2)

        node_labels = {graph.get_label_from_id(nid) for nid in graph.get_node_ids(limit=None)}
        expected = {"0,1", "0,2", "0,3", "1,2", "1,3", "2,3"}

        assert node_labels == expected

    def test_johnson_4_2_regularity(self):
        """Test J(4,2) is k(n-k)-regular."""
        graph = Graph(directed=False).add_johnson_graph(n=4, k=2)

        expected_degree = 2 * (4 - 2)  # k * (n-k) = 4
        degrees = [graph.get_degree(nid) for nid in graph.get_node_ids(limit=None)]

        assert all(d == expected_degree for d in degrees)

    def test_johnson_4_2_adjacency(self):
        """Test J(4,2) adjacency rule: edges when intersection size = k-1."""
        graph = Graph(directed=False).add_johnson_graph(n=4, k=2)

        # Node "0,1" should be adjacent to nodes that share exactly 1 element
        node_01_id = graph.get_id_from_label("0,1")
        neighbors = graph.neighbors(node_01_id)
        neighbor_labels = {graph.get_label_from_id(nid) for nid in neighbors}

        # Neighbors: share 0 or 1 with {0,1}
        expected_neighbors = {"0,2", "0,3", "1,2", "1,3"}
        assert neighbor_labels == expected_neighbors

    def test_johnson_5_2_properties(self):
        """Test J(5,2) basic properties."""
        graph = Graph(directed=False).add_johnson_graph(n=5, k=2)

        # C(5,2) = 10 nodes
        assert graph.num_nodes() == 10

        # Degree = k(n-k) = 2*3 = 6
        # Total edges = (10 * 6) / 2 = 30
        assert graph.num_edges() == 30

        degrees = [graph.get_degree(nid) for nid in graph.get_node_ids(limit=None)]
        assert all(d == 6 for d in degrees)

    def test_johnson_5_1_singletons(self):
        """Test J(5,1) - graph of 1-subsets (singletons)."""
        graph = Graph(directed=False).add_johnson_graph(n=5, k=1)

        # C(5,1) = 5 nodes
        assert graph.num_nodes() == 5

        # Singletons never intersect in k-1=0 elements
        # Actually they do: {0} and {1} intersect in 0 elements
        # So all pairs are adjacent
        # Degree = 1*(5-1) = 4 (complete graph K5)
        assert graph.num_edges() == 10  # C(5,2) = 10 edges

        degrees = [graph.get_degree(nid) for nid in graph.get_node_ids(limit=None)]
        assert all(d == 4 for d in degrees)

    def test_johnson_edge_cases(self):
        """Test edge cases: k=0, k=n."""
        # J(5,0) - single empty set, no edges
        g1 = Graph(directed=False).add_johnson_graph(n=5, k=0)
        assert g1.num_nodes() == 1
        assert g1.num_edges() == 0

        # J(5,5) - single full set, no edges
        g2 = Graph(directed=False).add_johnson_graph(n=5, k=5)
        assert g2.num_nodes() == 1
        assert g2.num_edges() == 0

    def test_johnson_invalid_parameters(self):
        """Test that k > n raises ValueError."""
        graph = Graph(directed=False)

        with pytest.raises(ValueError, match="k must be between 0 and n"):
            graph.add_johnson_graph(n=3, k=5)

    def test_johnson_method_chaining(self):
        """Test that add_johnson_graph returns self for chaining."""
        graph = Graph(directed=False)
        result = graph.add_johnson_graph(n=4, k=2)

        assert result is graph
        assert graph.num_nodes() == 6

    def test_johnson_connectivity(self):
        """Test that Johnson graphs are connected for 1 <= k < n."""
        # J(4,2) should be connected
        graph = Graph(directed=False).add_johnson_graph(n=4, k=2)
        assert graph.count_weakly_connected_components() == 1

        # J(5,2) should be connected
        graph2 = Graph(directed=False).add_johnson_graph(n=5, k=2)
        assert graph2.count_weakly_connected_components() == 1

    def test_johnson_undirected(self):
        """Test that Johnson graphs are undirected."""
        graph = Graph(directed=False).add_johnson_graph(n=4, k=2)
        # is_directed is a property, not a method
        assert graph.is_directed == False

    def test_johnson_symmetric_adjacency(self):
        """Test that edges are symmetric (u-v implies v-u)."""
        graph = Graph(directed=False).add_johnson_graph(n=4, k=2)

        # For each edge, check that reverse edge exists
        # edges_as_tuples() returns tuples of (edge_id, source_id, target_id)
        for edge_tuple in graph.edges_as_tuples():
            edge_id, source, target = edge_tuple

            # Should find edge from target to source (symmetric)
            reverse_neighbors = graph.neighbors(target)
            assert source in reverse_neighbors
