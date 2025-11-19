"""Tests for edge analysis functions."""

import pytest
import edgecase as ec


class TestLoopEdges:
    """Test self-loop counting."""

    def test_count_all_loops_empty(self):
        """Empty graph has no loops."""
        G = ec.Graph()
        assert G.count_all_loops() == 0

    def test_count_all_loops_no_loops(self):
        """Graph with no self-loops."""
        G = ec.Graph()
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)
        assert G.count_all_loops() == 0

    def test_count_all_loops_single(self):
        """Graph with one self-loop."""
        G = ec.Graph()
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "A", auto_create_nodes=True)  # Self-loop
        G.add_edge("B", "C", auto_create_nodes=True)
        assert G.count_all_loops() == 1

    def test_count_all_loops_multiple(self):
        """Graph with multiple self-loops."""
        G = ec.Graph()
        G.add_edge("A", "A", auto_create_nodes=True)  # Loop 1
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "B", auto_create_nodes=True)  # Loop 2
        G.add_edge("C", "C", auto_create_nodes=True)  # Loop 3
        assert G.count_all_loops() == 3

    def test_count_all_loops_only_loops(self):
        """Graph with only self-loops."""
        G = ec.Graph()
        G.add_edge("A", "A", auto_create_nodes=True)
        G.add_edge("B", "B", auto_create_nodes=True)
        G.add_edge("C", "C", auto_create_nodes=True)
        assert G.count_all_loops() == 3

    def test_count_all_loops_alias(self):
        """count_all_loops() is an alias for count_all_loops()."""
        G = ec.Graph()
        G.add_edge("A", "A", auto_create_nodes=True)
        G.add_edge("B", "B", auto_create_nodes=True)
        assert G.count_all_loops() == G.count_all_loops() == 2


class TestMultiEdges:
    """Test multi-edge counting."""

    def test_count_multi_edges_empty(self):
        """Empty graph has no multi-edges."""
        G = ec.Graph()
        assert G.count_multi_edges() == 0

    def test_count_multi_edges_simple(self):
        """Simple graph (no parallel edges)."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", id=1, auto_create_nodes=True)
        G.add_edge("B", "C", id=2, auto_create_nodes=True)
        G.add_edge("C", "D", id=3, auto_create_nodes=True)
        assert G.count_multi_edges() == 0

    def test_count_multi_edges_single_duplicate(self):
        """Two edges between same pair."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", id=1, auto_create_nodes=True)
        G.add_edge("A", "B", id=2, auto_create_nodes=True)  # Duplicate
        G.add_edge("B", "C", id=3, auto_create_nodes=True)
        assert G.count_multi_edges() == 1  # 1 duplicate

    def test_count_multi_edges_triple(self):
        """Three edges between same pair."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", id=1, auto_create_nodes=True)
        G.add_edge("A", "B", id=2, auto_create_nodes=True)  # Duplicate 1
        G.add_edge("A", "B", id=3, auto_create_nodes=True)  # Duplicate 2
        assert G.count_multi_edges() == 2  # 2 duplicates

    def test_count_multi_edges_multiple_pairs(self):
        """Multiple pairs with duplicates."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", id=1, auto_create_nodes=True)
        G.add_edge("A", "B", id=2, auto_create_nodes=True)  # Duplicate
        G.add_edge("B", "C", id=3, auto_create_nodes=True)
        G.add_edge("B", "C", id=4, auto_create_nodes=True)  # Duplicate
        G.add_edge("B", "C", id=5, auto_create_nodes=True)  # Another duplicate
        G.add_edge("C", "D", id=6, auto_create_nodes=True)
        assert G.count_multi_edges() == 3  # 1 + 2 duplicates

    def test_count_multi_edges_directed_different_directions(self):
        """Edges in different directions are not multi-edges."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", id=1, auto_create_nodes=True)
        G.add_edge("B", "A", id=2, auto_create_nodes=True)  # Different direction
        assert G.count_multi_edges() == 0

    def test_count_multi_edges_undirected(self):
        """Multi-edges in undirected graph."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B", id=1, auto_create_nodes=True)
        G.add_edge("A", "B", id=2, auto_create_nodes=True)  # Duplicate
        assert G.count_multi_edges() == 1


class TestMutualNodePairs:
    """Test bidirectional edge counting."""

    def test_count_mutual_empty(self):
        """Empty graph has no mutual pairs."""
        G = ec.Graph(directed=True)
        assert G.count_mutual_node_pairs() == 0

    def test_count_mutual_undirected_error(self):
        """Raises error for undirected graphs."""
        G = ec.Graph(directed=False)
        with pytest.raises(ValueError, match="only defined for directed"):
            G.count_mutual_node_pairs()

    def test_count_mutual_no_mutual(self):
        """Graph with no bidirectional edges."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)
        assert G.count_mutual_node_pairs() == 0

    def test_count_mutual_single_pair(self):
        """One mutual pair."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)  # Mutual with A
        G.add_edge("B", "C", auto_create_nodes=True)  # One-way only
        assert G.count_mutual_node_pairs() == 1

    def test_count_mutual_multiple_pairs(self):
        """Multiple mutual pairs."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)  # Mutual pair 1
        G.add_edge("C", "D", auto_create_nodes=True)
        G.add_edge("D", "C", auto_create_nodes=True)  # Mutual pair 2
        G.add_edge("E", "F", auto_create_nodes=True)  # One-way
        assert G.count_mutual_node_pairs() == 2

    def test_count_mutual_excludes_self_loops(self):
        """Self-loops are not counted as mutual pairs."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "A", auto_create_nodes=True)  # Self-loop
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)  # Mutual
        assert G.count_mutual_node_pairs() == 1

    def test_count_mutual_complex_network(self):
        """Complex network with mixed edge types."""
        G = ec.Graph(directed=True)
        # Mutual pairs
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "B", auto_create_nodes=True)
        # One-way edges
        G.add_edge("C", "D", auto_create_nodes=True)
        G.add_edge("D", "E", auto_create_nodes=True)
        # Self-loop
        G.add_edge("E", "E", auto_create_nodes=True)
        assert G.count_mutual_node_pairs() == 2

    def test_count_mutual_fully_connected(self):
        """Fully connected directed graph (complete digraph)."""
        G = ec.Graph(directed=True)
        nodes = ["A", "B", "C"]
        for i, src in enumerate(nodes):
            for j, tgt in enumerate(nodes):
                if i != j:  # No self-loops
                    G.add_edge(src, tgt, auto_create_nodes=True)
        # 3 nodes = 3 pairs: (A,B), (B,C), (A,C)
        assert G.count_mutual_node_pairs() == 3


class TestAsymmetricNodePairs:
    """Test one-way edge counting."""

    def test_count_asymmetric_empty(self):
        """Empty graph has no asymmetric pairs."""
        G = ec.Graph(directed=True)
        assert G.count_asymmetric_node_pairs() == 0

    def test_count_asymmetric_undirected_error(self):
        """Raises error for undirected graphs."""
        G = ec.Graph(directed=False)
        with pytest.raises(ValueError, match="only defined for directed"):
            G.count_asymmetric_node_pairs()

    def test_count_asymmetric_all_one_way(self):
        """All edges are one-way."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)
        assert G.count_asymmetric_node_pairs() == 3

    def test_count_asymmetric_no_asymmetric(self):
        """All edges are mutual (no asymmetric pairs)."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)
        G.add_edge("D", "C", auto_create_nodes=True)
        assert G.count_asymmetric_node_pairs() == 0

    def test_count_asymmetric_mixed(self):
        """Mix of mutual and asymmetric pairs."""
        G = ec.Graph(directed=True)
        # Mutual pair
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)
        # Asymmetric pairs
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("D", "E", auto_create_nodes=True)
        assert G.count_asymmetric_node_pairs() == 2

    def test_count_asymmetric_excludes_self_loops(self):
        """Self-loops are not counted."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "A", auto_create_nodes=True)  # Self-loop
        G.add_edge("A", "B", auto_create_nodes=True)  # Asymmetric
        assert G.count_asymmetric_node_pairs() == 1

    def test_count_asymmetric_complex_network(self):
        """Complex network with various edge patterns."""
        G = ec.Graph(directed=True)
        # Mutual pairs
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)
        # Asymmetric pairs
        G.add_edge("B", "C", auto_create_nodes=True)  # One-way
        G.add_edge("C", "D", auto_create_nodes=True)  # One-way
        G.add_edge("E", "F", auto_create_nodes=True)  # One-way
        # Self-loop
        G.add_edge("F", "F", auto_create_nodes=True)
        assert G.count_asymmetric_node_pairs() == 3

    def test_count_asymmetric_partial_mutual(self):
        """Some mutual, some one-way from same nodes."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)  # Mutual
        G.add_edge("A", "C", auto_create_nodes=True)  # Asymmetric from A
        G.add_edge("B", "D", auto_create_nodes=True)  # Asymmetric from B
        assert G.count_asymmetric_node_pairs() == 2


class TestEdgeAnalysisIntegration:
    """Integration tests combining multiple edge analysis functions."""

    def test_complete_edge_analysis(self):
        """Comprehensive edge analysis on complex graph."""
        G = ec.Graph(directed=True)

        # Add various edge types
        G.add_edge("A", "B", id=1, auto_create_nodes=True)
        G.add_edge("B", "A", id=2, auto_create_nodes=True)  # Mutual pair
        G.add_edge("A", "B", id=3, auto_create_nodes=True)  # Multi-edge
        G.add_edge("C", "D", id=4, auto_create_nodes=True)  # Asymmetric
        G.add_edge("E", "E", id=5, auto_create_nodes=True)  # Self-loop
        G.add_edge("F", "G", id=6, auto_create_nodes=True)  # Asymmetric

        assert G.count_all_loops() == 1
        assert G.count_multi_edges() == 1
        assert G.count_mutual_node_pairs() == 1
        assert G.count_asymmetric_node_pairs() == 2

    def test_social_network_analysis(self):
        """Analyze a social network graph."""
        G = ec.Graph(directed=True)

        # Mutual friendships
        G.add_edge("Alice", "Bob", auto_create_nodes=True)
        G.add_edge("Bob", "Alice", auto_create_nodes=True)
        G.add_edge("Bob", "Carol", auto_create_nodes=True)
        G.add_edge("Carol", "Bob", auto_create_nodes=True)

        # One-way follows
        G.add_edge("Alice", "Dave", auto_create_nodes=True)
        G.add_edge("Carol", "Dave", auto_create_nodes=True)
        G.add_edge("Dave", "Eve", auto_create_nodes=True)

        # No self-loops in social network
        assert G.count_all_loops() == 0
        # No multi-edges
        assert G.count_multi_edges() == 0
        # 2 mutual friendships
        assert G.count_mutual_node_pairs() == 2
        # 3 one-way follows
        assert G.count_asymmetric_node_pairs() == 3

    def test_citation_network(self):
        """Analyze citation network (directed, no mutual citations typically)."""
        G = ec.Graph(directed=True)

        # Paper citations (typically one-way, chronological)
        G.add_edge("Paper2023", "Paper2020", auto_create_nodes=True)
        G.add_edge("Paper2023", "Paper2019", auto_create_nodes=True)
        G.add_edge("Paper2022", "Paper2020", auto_create_nodes=True)
        G.add_edge("Paper2022", "Paper2019", auto_create_nodes=True)
        G.add_edge("Paper2021", "Paper2019", auto_create_nodes=True)

        # No self-citations
        assert G.count_all_loops() == 0
        # Multiple citations to same paper (not multi-edges, different sources)
        assert G.count_multi_edges() == 0
        # No mutual citations (no time travel!)
        assert G.count_mutual_node_pairs() == 0
        # All citations are one-way
        assert G.count_asymmetric_node_pairs() == 5

    def test_multigraph_scenario(self):
        """Test multi-edge scenario with multiple relationships."""
        G = ec.Graph(directed=True)

        # Multiple types of edges between same nodes
        G.add_edge("A", "B", id=1, attributes={"type": "friend"}, auto_create_nodes=True)
        G.add_edge("A", "B", id=2, attributes={"type": "colleague"}, auto_create_nodes=True)
        G.add_edge("A", "B", id=3, attributes={"type": "neighbor"}, auto_create_nodes=True)
        G.add_edge("B", "A", id=4, attributes={"type": "friend"}, auto_create_nodes=True)

        assert G.count_multi_edges() == 2  # 2 extra edges A->B
        assert G.count_mutual_node_pairs() == 1  # A and B are mutual

    def test_comparison_with_simple_graph(self):
        """Compare simple vs complex graph properties."""
        G_simple = ec.Graph(directed=True)
        G_simple.add_edge("A", "B", auto_create_nodes=True)
        G_simple.add_edge("B", "C", auto_create_nodes=True)
        G_simple.add_edge("C", "D", auto_create_nodes=True)

        G_complex = ec.Graph(directed=True)
        G_complex.add_edge("A", "B", auto_create_nodes=True)
        G_complex.add_edge("B", "A", auto_create_nodes=True)
        G_complex.add_edge("A", "B", id=2, auto_create_nodes=True)  # Multi
        G_complex.add_edge("C", "C", auto_create_nodes=True)  # Loop

        # Simple graph
        assert G_simple.count_all_loops() == 0
        assert G_simple.count_multi_edges() == 0
        assert G_simple.count_mutual_node_pairs() == 0
        assert G_simple.count_asymmetric_node_pairs() == 3

        # Complex graph
        assert G_complex.count_all_loops() == 1
        assert G_complex.count_multi_edges() == 1
        assert G_complex.count_mutual_node_pairs() == 1
        assert G_complex.count_asymmetric_node_pairs() == 0


class TestEdgeAnalysisEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_node_no_edges(self):
        """Single node with no edges."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        assert G.count_all_loops() == 0
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 0
        assert G.count_asymmetric_node_pairs() == 0

    def test_single_self_loop_only(self):
        """Single node with only self-loop."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "A", auto_create_nodes=True)
        assert G.count_all_loops() == 1
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 0
        assert G.count_asymmetric_node_pairs() == 0

    def test_two_nodes_both_directions(self):
        """Two nodes, fully connected both ways."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "A", auto_create_nodes=True)
        assert G.count_all_loops() == 0
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 1
        assert G.count_asymmetric_node_pairs() == 0

    def test_star_graph_directed(self):
        """Star topology (hub with spokes)."""
        G = ec.Graph(directed=True)
        hub = "Hub"
        for i in range(5):
            G.add_edge(hub, f"Node{i}", auto_create_nodes=True)

        assert G.count_all_loops() == 0
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 0
        assert G.count_asymmetric_node_pairs() == 5

    def test_cycle_directed(self):
        """Directed cycle."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("B", "C", auto_create_nodes=True)
        G.add_edge("C", "D", auto_create_nodes=True)
        G.add_edge("D", "A", auto_create_nodes=True)  # Complete cycle

        assert G.count_all_loops() == 0
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 0
        assert G.count_asymmetric_node_pairs() == 4
