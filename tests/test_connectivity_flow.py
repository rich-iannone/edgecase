"""Tests for connectivity and flow metrics."""

import pytest
import edgecase as ec


class TestReciprocity:
    """Test reciprocity (fraction of reciprocal edges) calculation."""

    def test_reciprocity_fully_reciprocal(self):
        """Test reciprocity when all edges are reciprocal."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("B", "A")  # Reciprocal
        G.add_edge("B", "C")
        G.add_edge("C", "B")  # Reciprocal

        reciprocity = G.get_reciprocity()
        assert reciprocity == 1.0  # All edges are reciprocal

    def test_reciprocity_no_reciprocal(self):
        """Test reciprocity when no edges are reciprocal."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        reciprocity = G.get_reciprocity()
        assert reciprocity == 0.0  # No reciprocal edges

    def test_reciprocity_partial(self):
        """Test reciprocity with some reciprocal edges."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("B", "A")  # Reciprocal (2 edges)
        G.add_edge("A", "C")  # Not reciprocal (1 edge)

        reciprocity = G.get_reciprocity()
        # 2 out of 3 edges are part of reciprocal pairs
        assert abs(reciprocity - 2.0 / 3.0) < 0.001

    def test_reciprocity_self_loops_ignored(self):
        """Test that self-loops don't affect reciprocity."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")
        G.add_edge("B", "A")  # Reciprocal
        G.add_edge("A", "A")  # Self-loop (counted as edge but not reciprocal)

        reciprocity = G.get_reciprocity()
        # 2 out of 3 edges are reciprocal (self-loop doesn't have reverse)
        assert abs(reciprocity - 2.0 / 3.0) < 0.001

    def test_reciprocity_empty_graph(self):
        """Test reciprocity on empty graph."""
        G = ec.Graph(directed=True)

        reciprocity = G.get_reciprocity()
        assert reciprocity == 0.0

    def test_reciprocity_single_edge(self):
        """Test reciprocity with single edge."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")

        reciprocity = G.get_reciprocity()
        assert reciprocity == 0.0

    def test_reciprocity_cycle(self):
        """Test reciprocity on directed cycle."""
        G = ec.Graph(directed=True)
        G.add_cycle_graph(4)  # 0→1→2→3→0

        reciprocity = G.get_reciprocity()
        assert reciprocity == 0.0  # Cycle has no reciprocal edges

    def test_reciprocity_undirected_error(self):
        """Test that reciprocity raises error for undirected graphs."""
        G = ec.Graph(directed=False)
        G.add_path_graph(3)

        with pytest.raises(ValueError, match="requires a directed graph"):
            G.get_reciprocity()

    def test_reciprocity_social_network(self):
        """Test reciprocity on social network example."""
        G = ec.Graph(directed=True)
        # Create a social network with follows
        for node in ["Alice", "Bob", "Charlie", "David"]:
            G.add_node(node)

        # Mutual follows
        G.add_edge("Alice", "Bob")
        G.add_edge("Bob", "Alice")

        # One-way follows
        G.add_edge("Alice", "Charlie")
        G.add_edge("Bob", "David")

        reciprocity = G.get_reciprocity()
        # 2 reciprocal out of 4 total edges = 0.5
        assert reciprocity == 0.5


class TestMinCutBetween:
    """Test minimum cut between nodes calculation."""

    def test_min_cut_path_graph(self):
        """Test min cut on path graph."""
        G = ec.Graph()
        G.add_path_graph(4)  # Creates nodes 1-2-3-4

        # Minimum cut between ends is 1 (cut the path)
        min_cut = G.get_min_cut_between(1, 4)
        assert min_cut == 1

    def test_min_cut_parallel_edges(self):
        """Test min cut with parallel edges."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "B", auto_create_nodes=True)  # Parallel edge

        # Need to cut 2 edges to disconnect
        a_id, b_id = G.get_id_from_label("A"), G.get_id_from_label("B")
        min_cut = G.get_min_cut_between(a_id, b_id)
        assert min_cut == 2

    def test_min_cut_star_graph(self):
        """Test min cut on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)  # Hub at 1, leaves at 2-6

        # Cut between hub and leaf needs 1 edge
        min_cut = G.get_min_cut_between(1, 2)
        assert min_cut == 1

    def test_min_cut_complete_graph(self):
        """Test min cut on complete graph."""
        G = ec.Graph()
        G.add_full_graph(4)

        # In K4, min cut between any two nodes
        min_cut = G.get_min_cut_between(1, 4)
        assert min_cut == 3  # Three edge-disjoint paths

    def test_min_cut_directed_graph(self):
        """Test min cut on directed graph."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)

        # From A to C, there are 2 edge-disjoint paths
        a_id, c_id = G.get_id_from_label("A"), G.get_id_from_label("C")
        min_cut = G.get_min_cut_between(a_id, c_id)
        assert min_cut == 2

    def test_min_cut_bottleneck(self):
        """Test min cut identifies bottleneck."""
        G = ec.Graph()
        # Create two cliques connected by single edge
        G.add_full_graph(3)  # Nodes 1, 2, 3
        G.add_node("4")
        G.add_node("5")
        G.add_node("6")
        G.add_edge("3", "4")  # Bottleneck
        G.add_edge("4", "5")
        G.add_edge("4", "6")
        G.add_edge("5", "6")

        # Min cut between cliques is 1 (the bottleneck)
        min_cut = G.get_min_cut_between("1", "6")
        assert min_cut == 1

    def test_min_cut_same_node_error(self):
        """Test that same source and sink raises error."""
        G = ec.Graph()
        G.add_node("A")

        with pytest.raises(ValueError, match="must be different"):
            G.get_min_cut_between("A", "A")

    def test_min_cut_nonexistent_source(self):
        """Test error when source doesn't exist."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")

        with pytest.raises(ValueError, match="not found in graph"):
            G.get_min_cut_between("X", "B")

    def test_min_cut_nonexistent_sink(self):
        """Test error when sink doesn't exist."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")

        with pytest.raises(ValueError, match="not found in graph"):
            G.get_min_cut_between("A", "X")

    def test_min_cut_disconnected(self):
        """Test min cut when nodes are disconnected."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        # No edges - disconnected

        min_cut = G.get_min_cut_between("A", "B")
        assert min_cut == 0  # Already disconnected

    def test_min_cut_cycle(self):
        """Test min cut on cycle graph."""
        G = ec.Graph()
        G.add_cycle_graph(6)

        # In a cycle, min cut is 2 (cut in two places)
        min_cut = G.get_min_cut_between("1", "4")
        assert min_cut == 2

    def test_min_cut_grid(self):
        """Test min cut on grid graph."""
        G = ec.Graph()
        G.add_grid_2d(3, 3)

        # In 3x3 grid, corner to opposite corner (node 1 to node 9)
        min_cut = G.get_min_cut_between("1", "9")
        # Should have multiple paths
        assert min_cut >= 2


class TestConnectivityFlowIntegration:
    """Test interactions between connectivity and flow metrics."""

    def test_reciprocity_vs_edge_count(self):
        """Test reciprocity relationship with edge count."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("B", "A")
        G.add_edge("B", "C")

        reciprocity = G.get_reciprocity()
        edge_count = G.count_edges()

        # Should have consistent relationship
        assert reciprocity * edge_count == 2.0  # 2 reciprocal edges

    def test_min_cut_symmetric(self):
        """Test that min cut is symmetric for undirected graphs."""
        G = ec.Graph()
        G.add_path_graph(5)

        cut_ab = G.get_min_cut_between("1", "5")
        cut_ba = G.get_min_cut_between("5", "1")

        assert cut_ab == cut_ba

    def test_min_cut_vs_connectivity(self):
        """Test min cut relationship with connectivity."""
        G = ec.Graph()
        G.add_cycle_graph(5)

        # Cycle is 2-edge-connected
        min_cut = G.get_min_cut_between("1", "3")
        assert min_cut == 2

    def test_reciprocity_complete_bidirectional(self):
        """Test reciprocity on complete bidirectional graph."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        # Add all edges in both directions
        for src in ["A", "B", "C"]:
            for dst in ["A", "B", "C"]:
                if src != dst:
                    G.add_edge(src, dst)

        reciprocity = G.get_reciprocity()
        assert reciprocity == 1.0  # All edges reciprocal


class TestConnectivityFlowEdgeCases:
    """Test edge cases for connectivity and flow metrics."""

    def test_reciprocity_multi_edges(self):
        """Test reciprocity with multiple edges same direction."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")
        G.add_edge("A", "B")  # Duplicate
        G.add_edge("B", "A")

        reciprocity = G.get_reciprocity()
        # 3 total edges: 2 A→B and 1 B→A. Only 2 have reciprocal counterparts
        assert abs(reciprocity - 2 / 3) < 0.001

    def test_min_cut_single_node(self):
        """Test min cut error with single node graph."""
        G = ec.Graph()
        G.add_node("A")

        with pytest.raises(ValueError):
            G.get_min_cut_between("A", "A")

    def test_reciprocity_asymmetric_network(self):
        """Test reciprocity on asymmetric directed network."""
        G = ec.Graph(directed=True)
        # Web page links: pages link out but few link back
        for i in range(5):
            G.add_node(str(i))

        # Create a hub-and-spoke pattern (low reciprocity)
        for i in range(1, 5):
            G.add_edge("0", str(i))  # Hub links to all
        G.add_edge("1", "0")  # Only one links back

        reciprocity = G.get_reciprocity()
        # 2 reciprocal out of 5 edges = 0.4
        assert reciprocity == 0.4
