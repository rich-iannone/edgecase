import math
import pytest

from edgecase import Graph


def _build_graph(edges, directed=False):
    """Helper to build a graph from edges."""
    g = Graph(directed=directed)

    # Extract unique nodes
    nodes = set()
    for src, tgt in edges:
        nodes.add(src)
        nodes.add(tgt)

    # Add all nodes
    for node in nodes:
        g.add_node(node)

    # Add edges
    for src, tgt in edges:
        g.add_edge(src, tgt)

    return g


class TestDomiRankBasic:
    """Basic DomiRank functionality tests."""

    def test_empty_graph(self):
        """Empty graph returns empty scores."""
        g = Graph()
        result = g.get_domirank_centrality()
        assert result == {}

    def test_single_node(self):
        """Single node returns nearly zero dominance."""
        g = Graph()
        g.add_node("A")
        result = g.get_domirank_centrality()
        assert "A" in result

        # Single node should have zero dominance (no competition)
        # Allow small numerical error
        assert abs(result["A"]) < 1e-6

    def test_path_graph(self):
        """Path graph P3: A-B-C."""
        g = _build_graph([("A", "B"), ("B", "C")])

        result = g.get_domirank_centrality()

        # B (center) should have highest dominance
        assert result["B"] > result["A"]
        assert result["B"] > result["C"]

        # End nodes A and C should have similar (negative) dominance
        assert abs(result["A"] - result["C"]) < 1e-6

    def test_cycle_graph(self):
        """Triangle cycle: all nodes equal."""
        g = _build_graph([("A", "B"), ("B", "C"), ("C", "A")])

        result = g.get_domirank_centrality()

        # All nodes equivalent in symmetric cycle
        assert abs(result["A"] - result["B"]) < 1e-6
        assert abs(result["B"] - result["C"]) < 1e-6

    def test_star_graph(self):
        """Star graph: hub dominates."""
        g = _build_graph([("hub", "A"), ("hub", "B"), ("hub", "C"), ("hub", "D")])

        result = g.get_domirank_centrality()

        # Hub should have highest dominance
        assert result["hub"] > result["A"]
        assert result["hub"] > result["B"]
        assert result["hub"] > result["C"]
        assert result["hub"] > result["D"]

        # All leaves should have similar (negative) dominance
        leaf_scores = [result["A"], result["B"], result["C"], result["D"]]
        assert all(abs(s - leaf_scores[0]) < 1e-6 for s in leaf_scores)

    def test_complete_graph(self):
        """Complete graph K5: all nodes equal."""
        nodes = ["A", "B", "C", "D", "E"]
        edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
        g = _build_graph(edges)

        result = g.get_domirank_centrality()

        # All nodes should have same dominance (symmetric)
        scores = list(result.values())
        assert all(abs(s - scores[0]) < 1e-6 for s in scores)


class TestDomiRankParameters:
    """Test DomiRank parameter handling."""

    def test_manual_sigma(self):
        """Manual sigma specification."""
        g = _build_graph([("A", "B"), ("B", "C")])

        result_auto = g.get_domirank_centrality()
        result_manual = g.get_domirank_centrality(sigma=0.5)

        # Results should differ
        assert result_auto != result_manual

    def test_return_sigma(self):
        """Test return_sigma parameter."""
        g = _build_graph([("A", "B"), ("B", "C")])

        # Without return_sigma
        result = g.get_domirank_centrality()
        assert isinstance(result, dict)

        # With return_sigma
        result, sigma = g.get_domirank_centrality(return_sigma=True)
        assert isinstance(result, dict)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_alpha_parameter(self):
        """Test alpha parameter affects auto-computed sigma."""
        g = _build_graph([("A", "B"), ("B", "C")])

        _, sigma_low = g.get_domirank_centrality(alpha=0.5, return_sigma=True)
        _, sigma_high = g.get_domirank_centrality(alpha=0.95, return_sigma=True)

        # Higher alpha should produce higher sigma
        assert sigma_high > sigma_low

    def test_tolerance_parameter(self):
        """Test tolerance parameter."""
        g = _build_graph([("A", "B"), ("B", "C")])

        # Should complete without error
        result_tight = g.get_domirank_centrality(tolerance=1e-8)
        result_loose = g.get_domirank_centrality(tolerance=1e-3)

        # Both should produce reasonable results
        assert len(result_tight) == 3
        assert len(result_loose) == 3

    def test_max_iter_parameter(self):
        """Test max_iter parameter."""
        g = _build_graph([("A", "B"), ("B", "C")])

        # Should complete even with low max_iter (will warn about convergence)
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = g.get_domirank_centrality(max_iter=100)
        assert len(result) == 3

    def test_patience_parameter(self):
        """Test patience parameter."""
        g = _build_graph([("A", "B"), ("B", "C")])

        # Should complete with different patience values
        result1 = g.get_domirank_centrality(patience=5)
        result2 = g.get_domirank_centrality(patience=20)

        assert len(result1) == 3
        assert len(result2) == 3

    def test_invalid_method(self):
        """Test invalid method raises error."""
        g = _build_graph([("A", "B")])

        with pytest.raises(ValueError, match="Only method='iterative'"):
            g.get_domirank_centrality(method="analytical")


class TestDomiRankDirected:
    """Test DomiRank on directed graphs."""

    def test_directed_path(self):
        """Directed path: A→B→C."""
        g = _build_graph([("A", "B"), ("B", "C")], directed=True)

        result = g.get_domirank_centrality()

        # A (out-degree 1) vs C (in-degree 1)
        # B has both in and out connections
        assert len(result) == 3

    def test_directed_cycle(self):
        """Directed cycle: A→B→C→A."""
        g = _build_graph([("A", "B"), ("B", "C"), ("C", "A")], directed=True)

        result = g.get_domirank_centrality()

        # All nodes should have equal dominance (symmetric cycle)
        assert abs(result["A"] - result["B"]) < 1e-6
        assert abs(result["B"] - result["C"]) < 1e-6

    def test_directed_star(self):
        """Directed star: hub with all outgoing edges."""
        g = _build_graph([("hub", "A"), ("hub", "B"), ("hub", "C")], directed=True)

        result = g.get_domirank_centrality()

        # Hub has out-degree but no in-degree
        # Leaves have in-degree but no out-degree
        assert len(result) == 4


class TestDomiRankDisconnected:
    """Test DomiRank on disconnected graphs."""

    def test_two_components(self):
        """Two separate path components."""
        g = Graph(directed=False)

        # Component 1: A-B
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        # Component 2: C-D-E
        g.add_node("C")
        g.add_node("D")
        g.add_node("E")
        g.add_edge("C", "D")
        g.add_edge("D", "E")

        result = g.get_domirank_centrality()

        # Should compute for all nodes
        assert len(result) == 5

        # Within component 2, D should dominate
        assert result["D"] > result["C"]
        assert result["D"] > result["E"]

    def test_isolated_nodes(self):
        """Graph with isolated nodes."""
        g = Graph(directed=False)
        g.add_node("isolated")
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        result = g.get_domirank_centrality()

        # Isolated node should have zero dominance
        assert abs(result["isolated"]) < 1e-6


class TestDomiRankNumerical:
    """Test numerical properties of DomiRank."""

    def test_sigma_zero_limit(self):
        """As σ→0, DomiRank should approach degree centrality."""
        g = _build_graph([("A", "B"), ("B", "C"), ("B", "D")])

        result = g.get_domirank_centrality(sigma=0.01)

        # B should have highest score (highest degree)
        assert result["B"] > result["A"]
        assert result["B"] > result["C"]
        assert result["B"] > result["D"]

        # Low-degree nodes should have similar low scores
        low_degree_nodes = [result["A"], result["C"], result["D"]]
        assert all(s < result["B"] for s in low_degree_nodes)

    def test_higher_sigma(self):
        """Higher σ still identifies structural properties."""
        g = _build_graph([("A", "B"), ("B", "C"), ("B", "D")])

        # Use a moderate sigma value (auto-computed would be safer)
        result = g.get_domirank_centrality(sigma=0.5)

        # B should still dominate (highest degree)
        assert result["B"] > result["A"]
        assert result["B"] > result["C"]
        assert result["B"] > result["D"]

    def test_score_properties(self):
        """DomiRank scores preserve network structure."""
        g = _build_graph([("A", "B"), ("B", "C"), ("C", "D")])

        result = g.get_domirank_centrality()

        # Center nodes should have higher dominance
        assert result["B"] > result["A"]
        assert result["C"] > result["D"]

        # Symmetric path endpoints should have similar scores
        assert abs(result["A"] - result["D"]) < 1e-2

    def test_negative_scores(self):
        """Some nodes can have negative dominance (submissive)."""
        g = _build_graph([("hub", "A"), ("hub", "B"), ("hub", "C")])

        result = g.get_domirank_centrality()

        # Leaves should have negative scores (dominated by hub)
        assert result["A"] < 0
        assert result["B"] < 0
        assert result["C"] < 0

        # Hub should have positive score
        assert result["hub"] > 0


class TestDomiRankEdgeCases:
    """Test edge cases and error handling."""

    def test_two_nodes(self):
        """Two-node graph."""
        g = _build_graph([("A", "B")])

        result = g.get_domirank_centrality()

        # Both nodes have degree 1, should be symmetric
        assert abs(result["A"] - result["B"]) < 1e-6

    def test_self_loop(self):
        """Graph with self-loop."""
        g = Graph(directed=False)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "A")  # Self-loop
        g.add_edge("A", "B")

        result = g.get_domirank_centrality()

        # Should handle self-loop
        assert len(result) == 2

    def test_convergence_warning(self):
        """Test that convergence warning is raised if needed."""
        g = _build_graph([("A", "B"), ("B", "C")])

        # Force non-convergence with very low max_iter
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = g.get_domirank_centrality(max_iter=5, patience=1)

        # Should still return results
        assert len(result) == 3

    def test_large_graph_performance(self):
        """Test DomiRank scales to larger graphs."""
        g = Graph(directed=False)

        # Create a grid-like structure (10x10 = 100 nodes)
        for i in range(10):
            for j in range(10):
                node = f"n{i}_{j}"
                g.add_node(node)
                if i > 0:
                    g.add_edge(node, f"n{i - 1}_{j}")
                if j > 0:
                    g.add_edge(node, f"n{i}_{j - 1}")

        result = g.get_domirank_centrality()

        # Should complete successfully
        assert len(result) == 100

        # Interior nodes should have higher dominance than corners
        center = result["n5_5"]
        corner = result["n0_0"]
        assert center > corner


class TestDomiRankComparison:
    """Compare DomiRank with other centrality measures."""

    def test_vs_degree_centrality(self):
        """Compare DomiRank with degree centrality."""
        g = _build_graph([("A", "B"), ("B", "C"), ("B", "D"), ("C", "E")])

        domirank = g.get_domirank_centrality(sigma=0.01)  # Low sigma ~ degree
        degrees = {node: g.get_degree(node) for node in ["A", "B", "C", "D", "E"]}

        # Rankings should be similar at low sigma
        dr_ranking = sorted(domirank.keys(), key=lambda k: domirank[k], reverse=True)
        deg_ranking = sorted(degrees.keys(), key=lambda k: degrees[k], reverse=True)

        # B should be top in both (highest degree)
        assert dr_ranking[0] == deg_ranking[0] == "B"
