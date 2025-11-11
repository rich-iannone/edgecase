import pytest
from edgecase import Graph


class TestKatzBasic:
    """Basic Katz centrality tests."""

    def test_empty_graph(self):
        """Empty graph returns empty scores."""
        g = Graph()
        result = g.get_katz_centrality()
        assert result == {}

    def test_single_node(self):
        """Single node gets normalized score of 1.0."""
        g = Graph()
        g.add_node("A")
        result = g.get_katz_centrality()
        assert "A" in result
        assert abs(result["A"] - 1.0) < 1e-6

    def test_two_nodes_directed(self):
        """Directed edge: receiver has higher Katz."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        result = g.get_katz_centrality(alpha=0.1)
        # B receives from A, so B should have higher centrality
        assert result["B"] > result["A"]

    def test_two_nodes_undirected(self):
        """Undirected edge: symmetric Katz scores."""
        g = Graph(directed=False)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        result = g.get_katz_centrality(alpha=0.1)
        # Should be symmetric
        assert abs(result["A"] - result["B"]) < 1e-6

    def test_path_graph(self):
        """Path A→B→C: C has highest Katz."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        result = g.get_katz_centrality(alpha=0.1)
        # C receives from B (direct) and A (via B)
        # B receives from A (direct)
        # A receives nothing
        assert result["C"] > result["B"]
        assert result["B"] > result["A"]

    def test_star_directed(self):
        """Directed star: center has highest Katz if edges point to it."""
        g = Graph(directed=True)
        g.add_node("hub")
        for i in range(3):
            leaf = f"leaf_{i}"
            g.add_node(leaf)
            g.add_edge(leaf, "hub")  # Leaves point TO hub

        result = g.get_katz_centrality(alpha=0.1)
        # Hub receives from all leaves
        for i in range(3):
            assert result["hub"] > result[f"leaf_{i}"]

    def test_cycle_directed(self):
        """Directed cycle: all nodes equal Katz."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        result = g.get_katz_centrality(alpha=0.1)
        # Symmetric cycle - all equal
        assert abs(result["A"] - result["B"]) < 1e-6
        assert abs(result["B"] - result["C"]) < 1e-6


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

        low_alpha = g.get_katz_centrality(alpha=0.05)
        high_alpha = g.get_katz_centrality(alpha=0.15)

        # D receives via long path - should benefit more from high alpha
        assert high_alpha["D"] > low_alpha["D"]

    def test_beta_effect(self):
        """Beta adds uniform bias to all nodes."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        low_beta = g.get_katz_centrality(alpha=0.1, beta=0.5)
        high_beta = g.get_katz_centrality(alpha=0.1, beta=2.0)

        # Both betas should maintain same normalized scores after normalization
        # The key is that beta affects the absolute values before normalization
        # After normalization, the relative ordering is preserved
        assert low_beta["B"] > low_beta["A"]
        assert high_beta["B"] > high_beta["A"]

    def test_convergence_tolerance(self):
        """Lower tolerance means more iterations."""
        g = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        loose = g.get_katz_centrality(tol=1e-3)
        tight = g.get_katz_centrality(tol=1e-8)

        # Both should give similar results
        for node in ["A", "B", "C"]:
            assert abs(loose[node] - tight[node]) < 1e-2

    def test_max_iter_limit(self):
        """Katz completes even with low max_iter."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        result = g.get_katz_centrality(max_iter=10)
        assert len(result) == 2
        assert result["B"] > result["A"]


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

        result = g.get_katz_centrality()
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_positive(self):
        """All Katz scores should be positive."""
        g = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        result = g.get_katz_centrality()
        for score in result.values():
            assert score > 0


class TestKatzEdgeCases:
    """Test edge cases."""

    def test_disconnected_graph(self):
        """Disconnected components handled correctly."""
        g = Graph(directed=True)
        # Component 1: A→B
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")
        # Component 2: C→D
        g.add_node("C")
        g.add_node("D")
        g.add_edge("C", "D")

        result = g.get_katz_centrality()
        assert len(result) == 4
        # Within each component, receiver should have higher score
        assert result["B"] > result["A"]
        assert result["D"] > result["C"]

    def test_isolated_node(self):
        """Isolated nodes get base beta value."""
        g = Graph(directed=True)
        g.add_node("isolated")
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        result = g.get_katz_centrality()
        # Isolated node and A (no incoming edges) should have equal scores
        # B should have highest (receives from A)
        assert result["isolated"] == result["A"]
        assert result["B"] > result["isolated"]

    def test_self_loop(self):
        """Self-loop contributes to own Katz."""
        g = Graph(directed=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "A")  # Self-loop

        result = g.get_katz_centrality(alpha=0.1)
        # A receives from itself
        assert result["A"] > result["B"]


class TestKatzComparison:
    """Compare Katz with other centrality measures."""

    def test_vs_alpha_centrality_undirected(self):
        """On undirected graphs, Katz ≈ alpha centrality."""
        g = Graph(directed=False)
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        katz = g.get_katz_centrality(alpha=0.1, beta=1.0)
        alpha_c = g.get_alpha_centrality(alpha=0.1)

        # alpha_centrality returns numeric IDs, convert to labels
        alpha_c_labeled = {g.label_from_id(nid): score for nid, score in alpha_c.items()}

        # Should be very similar for undirected graphs
        for node in ["A", "B", "C"]:
            # Allow some difference due to different formulations
            assert abs(katz[node] - alpha_c_labeled[node]) < 0.1

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

        katz = g.get_katz_centrality()
        # Leader receives from all others
        assert katz["leader"] > katz["A"]
        assert katz["leader"] > katz["B"]
        assert katz["leader"] > katz["C"]


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

        result = g.get_katz_centrality(alpha=0.05)

        # End of path should have highest Katz
        assert result[nodes[-1]] == max(result.values())
        # Start should have lowest
        assert result[nodes[0]] == min(result.values())

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

        result = g.get_katz_centrality()
        scores = list(result.values())
        # All should be equal (within tolerance)
        assert all(abs(s - scores[0]) < 1e-6 for s in scores)
