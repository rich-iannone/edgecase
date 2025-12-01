import pytest
from edgecase import Graph


class TestKatzBasic:
    """Basic Katz centrality tests."""

    def test_empty_graph(self):
        """Empty graph returns empty scores."""
        g = Graph()
        result = g.katz_centrality(
            return_dict=True,
        )
        assert result == {}

    def test_single_node(self):
        """Single node gets normalized score of 1.0."""
        g = Graph()
        g.add_node("A")
        result = g.katz_centrality(
            return_dict=True,
        )
        a_id = g.get_id_from_label("A")
        assert a_id in result
        assert abs(result[a_id] - 1.0) < 1e-6

    def test_two_nodes_directed(self):
        """Directed edge: receiver has higher Katz."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        result = g.katz_centrality(return_dict=True, alpha=0.1)
        # B receives from A, so B should have higher centrality
        assert result[g.get_id_from_label("B")] > result[g.get_id_from_label("A")]

    def test_two_nodes_undirected(self):
        """Undirected edge: symmetric Katz scores."""
        g = Graph(directed=False)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        result = g.katz_centrality(return_dict=True, alpha=0.1)
        # Should be symmetric
        assert abs(result[g.get_id_from_label("A")] - result[g.get_id_from_label("B")]) < 1e-6

    def test_path_graph(self):
        """Path A->B->C: C has highest Katz."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        result = g.katz_centrality(return_dict=True, alpha=0.1)
        # C receives from B (direct) and A (via B)
        # B receives from A (direct)
        # A receives nothing
        assert result[g.get_id_from_label("C")] > result[g.get_id_from_label("B")]
        assert result[g.get_id_from_label("B")] > result[g.get_id_from_label("A")]

    def test_star_directed(self):
        """Directed star: center has highest Katz if edges point to it."""
        g = Graph(directed=True)
        g.add_node("hub")
        for i in range(3):
            leaf = f"leaf_{i}"
            g.add_node(leaf)
            g.add_edge(leaf, "hub")  # Leaves point TO hub

        result = g.katz_centrality(return_dict=True, alpha=0.1)
        # Hub receives from all leaves
        hub_id = g.get_id_from_label("hub")
        for i in range(3):
            leaf_id = g.get_id_from_label(f"leaf_{i}")
            assert result[hub_id] > result[leaf_id]

    def test_cycle_directed(self):
        """Directed cycle: all nodes equal Katz."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        result = g.katz_centrality(return_dict=True, alpha=0.1)
        # Symmetric cycle - all equal
        assert abs(result[g.get_id_from_label("A")] - result[g.get_id_from_label("B")]) < 1e-6
        assert abs(result[g.get_id_from_label("B")] - result[g.get_id_from_label("C")]) < 1e-6


class TestKatzParameters:
    """Test Katz parameter effects."""

    def test_alpha_effect(self):
        """Higher alpha increases influence of longer paths."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        low_alpha = g.katz_centrality(return_dict=True, alpha=0.05)
        high_alpha = g.katz_centrality(return_dict=True, alpha=0.15)

        # D receives via long path - should benefit more from high alpha
        d_id = g.get_id_from_label("D")
        assert high_alpha[d_id] > low_alpha[d_id]

    def test_beta_effect(self):
        """Beta adds uniform bias to all nodes."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        low_beta = g.katz_centrality(return_dict=True, alpha=0.1, beta=0.5)
        high_beta = g.katz_centrality(return_dict=True, alpha=0.1, beta=2.0)

        # Both betas should maintain same normalized scores after normalization
        # The key is that beta affects the absolute values before normalization
        # After normalization, the relative ordering is preserved
        a_id, b_id = g.get_id_from_label("A"), g.get_id_from_label("B")
        assert low_beta[b_id] > low_beta[a_id]
        assert high_beta[b_id] > high_beta[a_id]

    def test_convergence_tolerance(self):
        """Lower tolerance means more iterations."""
        g = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        loose = g.katz_centrality(return_dict=True, tol=1e-3)
        tight = g.katz_centrality(return_dict=True, tol=1e-8)

        # Both should give similar results
        for label in ["A", "B", "C"]:
            node_id = g.get_id_from_label(label)
            assert abs(loose[node_id] - tight[node_id]) < 1e-2

    def test_max_iter_limit(self):
        """Katz completes even with low max_iter."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        result = g.katz_centrality(return_dict=True, max_iter=10)
        assert len(result) == 2
        assert result[g.get_id_from_label("B")] > result[g.get_id_from_label("A")]


class TestKatzNormalization:
    """Test Katz score normalization."""

    def test_scores_sum_to_one(self):
        """Katz scores are normalized to sum to 1.0."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        result = g.katz_centrality(
            return_dict=True,
        )
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_positive(self):
        """All Katz scores should be positive."""
        g = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        result = g.katz_centrality(
            return_dict=True,
        )
        for score in result.values():
            assert score > 0


class TestKatzEdgeCases:
    """Test edge cases."""

    def test_disconnected_graph(self):
        """Disconnected components handled correctly."""
        g = Graph(directed=True)
        # Component 1: A->B
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")
        # Component 2: C->D
        g.add_node("C")
        g.add_node("D")
        g.add_edge("C", "D")

        result = g.katz_centrality(
            return_dict=True,
        )
        assert len(result) == 4
        # Within each component, receiver should have higher score
        assert result[g.get_id_from_label("B")] > result[g.get_id_from_label("A")]
        assert result[g.get_id_from_label("D")] > result[g.get_id_from_label("C")]

    def test_isolated_node(self):
        """Isolated nodes get base beta value."""
        g = Graph(directed=True)
        g.add_node("isolated")
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        result = g.katz_centrality(
            return_dict=True,
        )
        # Isolated node and A (no incoming edges) should have equal scores
        # B should have highest (receives from A)
        assert result[g.get_id_from_label("isolated")] == result[g.get_id_from_label("A")]
        assert result[g.get_id_from_label("B")] > result[g.get_id_from_label("isolated")]

    def test_self_loop(self):
        """Self-loop contributes to own Katz."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "A")  # Self-loop

        result = g.katz_centrality(return_dict=True, alpha=0.1)
        # A receives from itself
        assert result[g.get_id_from_label("A")] > result[g.get_id_from_label("B")]


class TestKatzComparison:
    """Compare Katz with other centrality measures."""

    def test_vs_alpha_centrality_undirected(self):
        """On undirected graphs, Katz ≈ alpha centrality."""
        g = Graph(directed=False)
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        katz = g.katz_centrality(return_dict=True, alpha=0.1, beta=1.0)
        alpha_c = g.alpha_centrality(return_dict=True, alpha=0.1)

        # Both now return node IDs as keys
        # Should be very similar for undirected graphs
        for label in ["A", "B", "C"]:
            node_id = g.get_id_from_label(label)
            # Allow some difference due to different formulations
            assert abs(katz[node_id] - alpha_c[node_id]) < 0.1

    def test_prestige_ordering(self):
        """Katz correctly identifies high-prestige nodes."""
        g = Graph(directed=True)
        # Create a "prestige" structure
        g.add_node("leader")
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "leader")
        g.add_edge("B", "leader")
        g.add_edge("C", "leader")

        katz = g.katz_centrality(
            return_dict=True,
        )
        # Leader receives from all others
        assert katz[g.get_id_from_label("leader")] > katz[g.get_id_from_label("A")]
        assert katz[g.get_id_from_label("leader")] > katz[g.get_id_from_label("B")]
        assert katz[g.get_id_from_label("leader")] > katz[g.get_id_from_label("C")]


class TestKatzLargeGraph:
    """Test Katz on larger graphs."""

    def test_large_path(self):
        """Katz on a long path graph."""
        g = Graph(directed=True)
        n = 50
        nodes = [f"n{i}" for i in range(n)]
        for node in nodes:
            g.add_node(node)
        for i in range(n - 1):
            g.add_edge(nodes[i], nodes[i + 1])

        result = g.katz_centrality(return_dict=True, alpha=0.05)

        # Get node IDs
        node_ids = [g.get_id_from_label(label) for label in nodes]

        # End of path should have highest Katz
        assert result[node_ids[-1]] == max(result.values())
        # Start should have lowest
        assert result[node_ids[0]] == min(result.values())

    def test_complete_graph(self):
        """Complete graph: all nodes equal Katz (symmetric)."""
        g = Graph(directed=False)
        nodes = [f"n{i}" for i in range(10)]
        for node in nodes:
            g.add_node(node)
        # Connect all pairs
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                g.add_edge(nodes[i], nodes[j])

        result = g.katz_centrality(
            return_dict=True,
        )
        scores = list(result.values())
        # All should be equal (within tolerance)
        assert all(abs(s - scores[0]) < 1e-6 for s in scores)
