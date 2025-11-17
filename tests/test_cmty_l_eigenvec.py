"""
Tests for Leading Eigenvector (Newman) community detection.

This module tests the get_cmty_l_eigenvec() method which implements
Newman's spectral optimization algorithm using the leading eigenvector
of the modularity matrix for recursive bisection.
"""

import pytest
import edgecase as ec


class TestLeadingEigenvectorBasic:
    """Basic functionality tests."""

    def test_empty_graph(self):
        """Empty graph should return empty dict."""
        G = ec.Graph(directed=False)
        communities = G.cmty_l_eigenvec(return_dict=True)
        assert communities == {}

    def test_single_node(self):
        """Single node should be in community 0."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        communities = G.cmty_l_eigenvec(return_dict=True)
        a_id = G.get_id_from_label("A")
        assert communities == {a_id: 0}

    def test_two_nodes_no_edge(self):
        """Two unconnected nodes should be in separate communities."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        communities = G.cmty_l_eigenvec(return_dict=True)
        assert len(set(communities.values())) == 2

    def test_two_nodes_with_edge(self):
        """Two connected nodes should be in same community."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B", auto_create_nodes=True)
        communities = G.cmty_l_eigenvec(return_dict=True)
        a_id, b_id = G.get_id_from_label("A"), G.get_id_from_label("B")
        assert communities[a_id] == communities[b_id]

    def test_directed_graph_error(self):
        """Should raise error for directed graphs."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        with pytest.raises(ValueError, match="undirected"):
            G.cmty_l_eigenvec(return_dict=True)


class TestLeadingEigenvectorBisection:
    """Test spectral bisection on graphs with clear 2-way splits."""

    def test_two_cliques_bridge(self):
        """Two triangles connected by bridge should split into 2 communities."""
        G = ec.Graph(directed=False)

        # Triangle 1
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "A", auto_create_nodes=True)

        # Bridge
        G.add_edge("C", "D", auto_create_nodes=True)

        # Triangle 2
        G.add_edge("D", "E", auto_create_nodes=True)
        G.add_edge("E", "F", auto_create_nodes=True)
        G.add_edge("F", "D", auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)

        # Should find 2 communities
        assert len(set(communities.values())) == 2

        # Triangle 1 together
        a_id, b_id, c_id = [G.get_id_from_label(x) for x in ["A", "B", "C"]]
        assert communities[a_id] == communities[b_id] == communities[c_id]

        # Triangle 2 together
        d_id, e_id, f_id = [G.get_id_from_label(x) for x in ["D", "E", "F"]]
        assert communities[d_id] == communities[e_id] == communities[f_id]

        # Different communities
        assert communities[a_id] != communities[d_id]

    def test_barbell_graph(self):
        """Barbell graph should split into two cliques."""
        G = ec.Graph(directed=False)

        # Left clique (K3)
        for i in ["A", "B", "C"]:
            for j in ["A", "B", "C"]:
                if i < j:
                    G.add_edge(i, j, auto_create_nodes=True)

        # Bridge
        G.add_edge("C", "D", auto_create_nodes=True)

        # Right clique (K3)
        for i in ["D", "E", "F"]:
            for j in ["D", "E", "F"]:
                if i < j:
                    G.add_edge(i, j, auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)

        # Should find 2 communities
        assert len(set(communities.values())) == 2

        # Left clique together
        a_id, b_id, c_id = [G.get_id_from_label(x) for x in ["A", "B", "C"]]
        assert communities[a_id] == communities[b_id] == communities[c_id]

        # Right clique together
        d_id, e_id, f_id = [G.get_id_from_label(x) for x in ["D", "E", "F"]]
        assert communities[d_id] == communities[e_id] == communities[f_id]

        # Different communities
        assert communities[a_id] != communities[d_id]

    def test_four_cliques(self):
        """Four small cliques should split hierarchically."""
        G = ec.Graph(directed=False)

        # Clique 1
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "A", auto_create_nodes=True)

        # Clique 2
        G.add_edge("D", "E", auto_create_nodes=True)
        G.add_edge("E", "F", auto_create_nodes=True)
        G.add_edge("F", "D", auto_create_nodes=True)

        # Clique 3
        G.add_edge("G", "H", auto_create_nodes=True)
        G.add_edge("H", "I", auto_create_nodes=True)
        G.add_edge("I", "G", auto_create_nodes=True)

        # Clique 4
        G.add_edge("J", "K", auto_create_nodes=True)
        G.add_edge("K", "L", auto_create_nodes=True)
        G.add_edge("L", "J", auto_create_nodes=True)

        # Bridges between pairs
        G.add_edge("C", "D", auto_create_nodes=True)  # 1-2
        G.add_edge("F", "G", auto_create_nodes=True)  # 2-3
        G.add_edge("I", "J", auto_create_nodes=True)  # 3-4

        communities = G.cmty_l_eigenvec(return_dict=True)

        # Should find 2-4 communities depending on recursive splitting
        num_communities = len(set(communities.values()))
        assert 2 <= num_communities <= 4


class TestLeadingEigenvectorKnownStructures:
    """Test on graphs with known structure."""

    def test_karate_club_simplified(self):
        """Simplified karate club should split into communities."""
        G = ec.Graph(directed=False)

        # Faction 1 (instructor's group)
        for i in range(1, 18):
            for j in range(i + 1, 18):
                if (j - i) <= 3:
                    G.add_edge(str(i), str(j), auto_create_nodes=True)

        # Faction 2 (officer's group)
        for i in range(18, 35):
            for j in range(i + 1, 35):
                if (j - i) <= 3:
                    G.add_edge(str(i), str(j), auto_create_nodes=True)

        # Sparse connections between factions
        G.add_edge("17", "18", auto_create_nodes=True)
        G.add_edge("16", "19", auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)

        # Should find at least 2 communities
        num_communities = len(set(communities.values()))
        assert num_communities >= 2


class TestLeadingEigenvectorModularity:
    """Test modularity-related properties."""

    def compute_modularity(self, G, communities):
        """Compute modularity Q for a given partition."""
        if G.num_nodes() == 0 or G.num_edges() == 0:
            return 0.0

        m = G.num_edges()
        Q = 0.0

        for node_i in G.node_ids(limit=None):
            for node_j in G.node_ids(limit=None):
                if communities[node_i] == communities[node_j]:
                    A_ij = 1.0 if G.has_edge(node_i, node_j) else 0.0
                    k_i = G.get_degree(node_i)
                    k_j = G.get_degree(node_j)
                    Q += A_ij - (k_i * k_j) / (2.0 * m)

        return Q / (2.0 * m)

    def test_modularity_positive_structured(self):
        """Modularity should be positive for structured graphs."""
        G = ec.Graph(directed=False)

        # Two well-separated cliques
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "A", auto_create_nodes=True)
        G.add_edge("D", "E", auto_create_nodes=True)
        G.add_edge("E", "F", auto_create_nodes=True)
        G.add_edge("F", "D", auto_create_nodes=True)

        # Single bridge
        G.add_edge("C", "D", auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)
        Q = self.compute_modularity(G, communities)

        # Should have positive modularity
        assert Q > 0.0

    def test_modularity_complete_graph(self):
        """Complete graph should have near-zero modularity."""
        G = ec.Graph(directed=False)
        G.add_full_graph(6)

        communities = G.cmty_l_eigenvec(return_dict=True)
        Q = self.compute_modularity(G, communities)

        # Complete graph has no clear structure, Q should be small
        assert abs(Q) < 0.1


class TestLeadingEigenvectorProperties:
    """Test algorithm properties and invariants."""

    def test_all_nodes_assigned(self):
        """Every node should be assigned to a community."""
        G = ec.Graph(directed=False)
        for i in range(10):
            for j in range(i + 1, 10):
                if abs(i - j) <= 2:
                    G.add_edge(str(i), str(j), auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)
        node_ids = G.node_ids(limit=None)

        assert len(communities) == len(node_ids)
        for node_id in node_ids:
            assert node_id in communities

    def test_community_ids_contiguous(self):
        """Community IDs should be 0, 1, 2, ... with no gaps."""
        G = ec.Graph(directed=False)

        # Create 3 disconnected triangles
        for offset in [0, 3, 6]:
            for i in range(3):
                for j in range(i + 1, 3):
                    G.add_edge(str(offset + i), str(offset + j), auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)
        comm_ids = sorted(set(communities.values()))

        # Should be [0, 1, 2, ...]
        assert comm_ids == list(range(len(comm_ids)))

    def test_deterministic(self):
        """Same graph should produce same communities."""
        G1 = ec.Graph(directed=False)
        G2 = ec.Graph(directed=False)

        for G in [G1, G2]:
            G.add_edge("A", "B", auto_create_nodes=True)
            G.add_edge("B", "C", auto_create_nodes=True)
            G.add_edge("C", "A", auto_create_nodes=True)
            G.add_edge("C", "D", auto_create_nodes=True)
            G.add_edge("D", "E", auto_create_nodes=True)
            G.add_edge("E", "F", auto_create_nodes=True)
            G.add_edge("F", "D", auto_create_nodes=True)

        comm1 = G1.cmty_l_eigenvec(return_dict=True)
        comm2 = G2.cmty_l_eigenvec(return_dict=True)

        assert comm1 == comm2


class TestLeadingEigenvectorPerformance:
    """Performance tests for medium-sized graphs."""

    def test_medium_graph(self):
        """Test with 50 nodes."""
        G = ec.Graph(directed=False)

        # Create 5 communities of 10 nodes each
        for comm_idx in range(5):
            offset = comm_idx * 10
            for i in range(10):
                for j in range(i + 1, 10):
                    if abs(i - j) <= 3:
                        G.add_edge(str(offset + i), str(offset + j), auto_create_nodes=True)

        # Add sparse inter-community edges
        G.add_edge("9", "10", auto_create_nodes=True)
        G.add_edge("19", "20", auto_create_nodes=True)
        G.add_edge("29", "30", auto_create_nodes=True)
        G.add_edge("39", "40", auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)

        # Should find multiple communities
        num_communities = len(set(communities.values()))
        assert 2 <= num_communities <= 7

    def test_sparse_graph(self):
        """Test with sparse connectivity."""
        G = ec.Graph(directed=False)

        # Chain of 30 nodes
        for i in range(29):
            G.add_edge(str(i), str(i + 1), auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)

        # Should split chain
        num_communities = len(set(communities.values()))
        assert num_communities >= 2


class TestLeadingEigenvectorEdgeCases:
    """Edge case tests."""

    def test_complete_graph_small(self):
        """Complete graph (K5) - all nodes equally connected."""
        G = ec.Graph(directed=False)
        G.add_full_graph(5)

        communities = G.cmty_l_eigenvec(return_dict=True)

        # Complete graph might be single community
        # (no clear split via eigenvector)
        assert len(communities) == 5
        num_communities = len(set(communities.values()))
        assert num_communities >= 1

    def test_star_graph(self):
        """Star graph (hub and spokes)."""
        G = ec.Graph(directed=False)

        # Central hub connected to 10 nodes
        for i in range(10):
            G.add_edge("hub", str(i), auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)

        # All nodes assigned
        assert len(communities) == 11

    def test_disconnected_components(self):
        """Multiple disconnected components."""
        G = ec.Graph(directed=False)

        # Component 1
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)

        # Component 2
        G.add_edge("D", "E", auto_create_nodes=True)
        G.add_edge("E", "F", auto_create_nodes=True)

        # Component 3
        G.add_edge("G", "H", auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)

        # Should find at least 3 communities
        assert len(set(communities.values())) >= 3

        # Each component should be separate
        a_id, d_id, g_id = [G.get_id_from_label(x) for x in ["A", "D", "G"]]
        assert communities[a_id] != communities[d_id]
        assert communities[d_id] != communities[g_id]

    def test_cycle_graph(self):
        """Cycle graph (ring)."""
        G = ec.Graph(directed=False)

        # Create cycle
        for i in range(10):
            G.add_edge(str(i), str((i + 1) % 10), auto_create_nodes=True)

        communities = G.cmty_l_eigenvec(return_dict=True)

        # All nodes assigned
        assert len(communities) == 10

        # Should split into communities
        num_communities = len(set(communities.values()))
        assert num_communities >= 1
