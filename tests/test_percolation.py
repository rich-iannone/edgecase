"""
Tests for percolation algorithms.

Tests cover:
- percolation_centrality: Brandes-style betweenness weighted by percolation states
- node_percolation: Random node removal simulation
- edge_percolation: Random edge removal simulation
- largest_component_size: Size of largest connected component
- percolation_threshold_analysis: Systematic robustness analysis
"""

import pytest
from edgecase import Graph


class TestPercolationCentrality:
    """Tests for percolation_centrality method."""

    def test_empty_graph(self):
        """Empty graph should return empty dict."""
        graph = Graph(directed=False)
        result = graph.percolation_centrality(return_dict=True)
        assert result == {}

    def test_single_node(self):
        """Single node should have zero centrality."""
        graph = Graph(directed=False)
        graph.add_node("A")
        result = graph.percolation_centrality(return_dict=True)
        assert result["A"] == 0.0

    def test_path_graph_default_states(self):
        """Path graph with default states (all 1.0)."""
        graph = Graph(directed=False)
        graph.add_path_graph(5)  # nodes 1-2-3-4-5
        result = graph.percolation_centrality(return_dict=True)

        # Middle nodes should have higher centrality than endpoints
        # Node 3 is in the center (labels are strings)
        assert result["3"] > result["1"]
        assert result["3"] > result["5"]

    def test_path_graph_custom_states(self):
        """Path graph with custom percolation states."""
        graph = Graph(directed=False)
        graph.add_path_graph(5)

        # Custom states: endpoints have higher percolation (labels are strings)
        states = {"1": 1.0, "2": 0.5, "3": 0.5, "4": 0.5, "5": 1.0}
        result = graph.percolation_centrality(states=states, return_dict=True)

        # All results should be non-negative
        for centrality in result.values():
            assert centrality >= 0.0

    def test_invalid_states(self):
        """States outside [0, 1] should raise error."""
        graph = Graph(directed=False)
        graph.add_path_graph(3)

        # State > 1.0 (labels are strings)
        with pytest.raises(ValueError):
            graph.percolation_centrality(states={"1": 1.5}, return_dict=True)

    def test_star_graph(self):
        """Star graph: center should have highest centrality."""
        graph = Graph(directed=False)
        graph.add_star_graph(5)  # center is node 1
        result = graph.percolation_centrality(return_dict=True)

        # Center node ("1") should have highest centrality (labels are strings)
        center_centrality = result["1"]
        for node, centrality in result.items():
            if node != "1":
                assert center_centrality >= centrality

    def test_complete_graph(self):
        """Complete graph: all nodes should have equal centrality."""
        graph = Graph(directed=False)
        graph.add_full_graph(5)
        result = graph.percolation_centrality(return_dict=True)

        # All nodes should have same centrality (within tolerance)
        values = list(result.values())
        for v in values:
            assert abs(v - values[0]) < 1e-10

    def test_store_as_attribute(self):
        """Test storing result as node attribute."""
        graph = Graph(directed=False)
        graph.add_path_graph(3)
        graph.percolation_centrality(attr_name="pc")

        # Check attribute exists on all nodes using get_node_attrs
        pc_values = graph.get_node_attrs("pc")
        # Should have values for all 3 nodes
        assert len(pc_values) == 3
        # All values should be non-negative
        for node_id, value in pc_values.items():
            assert value >= 0


class TestNodePercolation:
    """Tests for node_percolation method."""

    def test_q_zero(self):
        """q=0 means no removal."""
        graph = Graph(directed=False)
        graph.add_full_graph(10)
        result = graph.node_percolation(q=0.0, seed=42)

        assert result.count_nodes() == 10

    def test_q_one(self):
        """q=1 means all nodes removed."""
        graph = Graph(directed=False)
        graph.add_full_graph(10)
        result = graph.node_percolation(q=1.0, seed=42)

        assert result.count_nodes() == 0

    def test_partial_removal(self):
        """Partial removal should reduce node count."""
        graph = Graph(directed=False)
        graph.add_full_graph(10)
        result = graph.node_percolation(q=0.3, seed=42)

        # Should have approximately 7 nodes (10 - 3)
        assert result.count_nodes() == 7

    def test_reproducibility(self):
        """Same seed should give same result."""
        graph = Graph(directed=False)
        graph.add_full_graph(20)

        result1 = graph.node_percolation(q=0.5, seed=123)
        result2 = graph.node_percolation(q=0.5, seed=123)

        assert result1.count_nodes() == result2.count_nodes()
        ids1 = set(result1.get_node_ids())
        ids2 = set(result2.get_node_ids())
        assert ids1 == ids2

    def test_different_seeds(self):
        """Different seeds should (likely) give different results."""
        graph = Graph(directed=False)
        graph.add_full_graph(20)

        result1 = graph.node_percolation(q=0.5, seed=123)
        result2 = graph.node_percolation(q=0.5, seed=456)

        ids1 = set(result1.get_node_ids())
        ids2 = set(result2.get_node_ids())
        # Very unlikely to be the same
        assert ids1 != ids2

    def test_invalid_q(self):
        """Invalid q values should raise error."""
        graph = Graph(directed=False)
        graph.add_full_graph(5)

        with pytest.raises(ValueError):
            graph.node_percolation(q=1.5, seed=42)

        with pytest.raises(ValueError):
            graph.node_percolation(q=-0.1, seed=42)

    def test_edges_removed_with_nodes(self):
        """Edges incident to removed nodes should be removed."""
        graph = Graph(directed=False)
        graph.add_full_graph(4)  # K4 has 6 edges

        result = graph.node_percolation(q=0.5, seed=42)

        # With 2 nodes remaining, should have at most 1 edge (if connected)
        remaining_nodes = result.count_nodes()
        max_possible_edges = remaining_nodes * (remaining_nodes - 1) // 2
        assert result.count_edges() <= max_possible_edges


class TestEdgePercolation:
    """Tests for edge_percolation method."""

    def test_q_zero(self):
        """q=0 means no edge removal."""
        graph = Graph(directed=False)
        graph.add_full_graph(5)  # K5 has 10 edges
        result = graph.edge_percolation(q=0.0, seed=42)

        assert result.count_edges() == 10

    def test_q_one(self):
        """q=1 means all edges removed."""
        graph = Graph(directed=False)
        graph.add_full_graph(5)
        result = graph.edge_percolation(q=1.0, seed=42)

        assert result.count_edges() == 0
        # But all nodes should remain
        assert result.count_nodes() == 5

    def test_partial_removal(self):
        """Partial removal should reduce edge count."""
        graph = Graph(directed=False)
        graph.add_full_graph(5)  # 10 edges
        result = graph.edge_percolation(q=0.5, seed=42)

        # Should have approximately 5 edges
        assert result.count_edges() == 5

    def test_nodes_preserved(self):
        """All nodes should be preserved in edge percolation."""
        graph = Graph(directed=False)
        graph.add_full_graph(10)
        result = graph.edge_percolation(q=0.8, seed=42)

        # All 10 nodes should remain
        assert result.count_nodes() == 10

    def test_reproducibility(self):
        """Same seed should give same result."""
        graph = Graph(directed=False)
        graph.add_full_graph(10)

        result1 = graph.edge_percolation(q=0.5, seed=789)
        result2 = graph.edge_percolation(q=0.5, seed=789)

        assert result1.count_edges() == result2.count_edges()

    def test_invalid_q(self):
        """Invalid q values should raise error."""
        graph = Graph(directed=False)
        graph.add_full_graph(5)

        with pytest.raises(ValueError):
            graph.edge_percolation(q=1.5, seed=42)

        with pytest.raises(ValueError):
            graph.edge_percolation(q=-0.1, seed=42)


class TestLargestComponentSize:
    """Tests for largest_component_size method."""

    def test_empty_graph(self):
        """Empty graph should return 0."""
        graph = Graph(directed=False)
        assert graph.largest_component_size() == 0

    def test_connected_graph(self):
        """Connected graph: component size equals node count."""
        graph = Graph(directed=False)
        graph.add_full_graph(7)
        assert graph.largest_component_size() == 7

    def test_disconnected_graph(self):
        """Should return size of largest component."""
        graph = Graph(directed=False)
        # Create two separate triangles
        graph.add_cycle_graph(3)  # nodes 1, 2, 3
        # Add isolated nodes
        graph.add_node("X")
        graph.add_node("Y")
        graph.add_edge("X", "Y")

        # Largest component has 3 nodes
        assert graph.largest_component_size() == 3

    def test_single_node(self):
        """Single node is a component of size 1."""
        graph = Graph(directed=False)
        graph.add_node("A")
        assert graph.largest_component_size() == 1

    def test_after_node_percolation(self):
        """Test component size after percolation."""
        graph = Graph(directed=False)
        graph.add_full_graph(10)

        percolated = graph.node_percolation(q=0.5, seed=42)
        lcs = percolated.largest_component_size()

        # Component size should be at most the number of remaining nodes
        assert lcs <= percolated.count_nodes()
        assert lcs >= 1  # At least one node if graph is non-empty


class TestPercolationThresholdAnalysis:
    """Tests for percolation_threshold_analysis method."""

    def test_basic_analysis(self):
        """Basic analysis should return correct number of data points."""
        graph = Graph(directed=False)
        graph.add_full_graph(20)

        results = graph.percolation_threshold_analysis(steps=10, trials=3, seed=42)

        # Should have 11 data points (0%, 10%, ..., 100%)
        assert len(results) == 11

    def test_first_point(self):
        """First point (q=0) should have relative GCC size near 1.0."""
        graph = Graph(directed=False)
        graph.add_full_graph(20)

        results = graph.percolation_threshold_analysis(steps=5, trials=3, seed=42)

        # First point: q=0, should have GCC relative size = 1.0
        q, gcc_size = results[0]
        assert q == 0.0
        assert abs(gcc_size - 1.0) < 0.01

    def test_last_point(self):
        """Last point (q=1) should have relative GCC size = 0."""
        graph = Graph(directed=False)
        graph.add_full_graph(20)

        results = graph.percolation_threshold_analysis(steps=5, trials=3, seed=42)

        # Last point: q=1, no nodes, GCC should be 0
        q, gcc_size = results[-1]
        assert q == 1.0
        assert gcc_size == 0.0

    def test_monotonic_decrease(self):
        """GCC size should generally decrease as q increases."""
        graph = Graph(directed=False)
        graph.add_full_graph(50)

        results = graph.percolation_threshold_analysis(steps=10, trials=10, seed=42)

        # With enough trials, should show general decrease
        # (may not be strictly monotonic due to randomness)
        first_gcc = results[0][1]
        last_gcc = results[-1][1]
        assert first_gcc > last_gcc

    def test_reproducibility(self):
        """Same seed should give same results."""
        graph = Graph(directed=False)
        graph.add_full_graph(30)

        results1 = graph.percolation_threshold_analysis(steps=5, trials=3, seed=123)
        results2 = graph.percolation_threshold_analysis(steps=5, trials=3, seed=123)

        for (q1, gcc1), (q2, gcc2) in zip(results1, results2):
            assert q1 == q2
            assert abs(gcc1 - gcc2) < 1e-10


class TestDirectedGraphPercolation:
    """Tests for percolation on directed graphs."""

    def test_node_percolation_directed(self):
        """Node percolation should work on directed graphs."""
        graph = Graph(directed=True)
        graph.add_path_graph(10)
        result = graph.node_percolation(q=0.3, seed=42)

        assert result.count_nodes() == 7

    def test_edge_percolation_directed(self):
        """Edge percolation should work on directed graphs."""
        graph = Graph(directed=True)
        graph.add_path_graph(10)  # 9 directed edges
        result = graph.edge_percolation(q=0.5, seed=42)

        # Should have approximately half the edges (4-5 for 9 edges)
        assert 3 <= result.count_edges() <= 6

    def test_largest_component_directed(self):
        """Largest component on directed graph (weak connectivity)."""
        graph = Graph(directed=True)
        graph.add_path_graph(5)  # 1->2->3->4->5

        # Should still be weakly connected
        assert graph.largest_component_size() == 5


class TestRobustnessScenarios:
    """Test realistic robustness analysis scenarios."""

    def test_star_graph_fragility(self):
        """Star graphs are fragile - removing center disconnects all."""
        graph = Graph(directed=False)
        graph.add_star_graph(10)  # center is node 1

        # If we remove 50% of nodes, center is very likely to be removed
        # and the graph fragments
        results = []
        for _ in range(10):
            percolated = graph.node_percolation(q=0.5)
            lcs = percolated.largest_component_size()
            remaining = percolated.count_nodes()
            if remaining > 0:
                results.append(lcs / remaining)

        # Average should be low due to fragmentation
        avg_relative_gcc = sum(results) / len(results) if results else 0
        assert avg_relative_gcc < 0.8

    def test_complete_graph_robustness(self):
        """Complete graphs are robust - remain connected longer."""
        graph = Graph(directed=False)
        graph.add_full_graph(10)

        results = graph.percolation_threshold_analysis(steps=10, trials=5, seed=42)

        # Should maintain high connectivity until high q
        # At q=0.3, should still have nearly full connectivity
        q_30 = [gcc for q, gcc in results if abs(q - 0.3) < 0.01][0]
        assert q_30 > 0.9  # Still mostly connected
