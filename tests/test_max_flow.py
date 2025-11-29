import pytest
from edgecase import Graph


def create_flow_graph(directed=True):
    """Helper to create a graph for flow tests using string labels."""
    return Graph(directed=directed)


class TestMaxFlowBasic:
    """Basic max flow tests."""

    def test_simple_graph(self):
        """Test max flow on a simple graph (verified against NetworkX)."""
        g = create_flow_graph()
        # Using string labels with auto_create_nodes
        g.add_edge("0", "1", capacity=10.0, auto_create_nodes=True)
        g.add_edge("0", "2", capacity=10.0, auto_create_nodes=True)
        g.add_edge("1", "3", capacity=4.0, auto_create_nodes=True)
        g.add_edge("1", "2", capacity=2.0, auto_create_nodes=True)
        g.add_edge("2", "4", capacity=8.0, auto_create_nodes=True)
        g.add_edge("3", "4", capacity=10.0, auto_create_nodes=True)
        g.add_edge("3", "5", capacity=10.0, auto_create_nodes=True)
        g.add_edge("4", "5", capacity=10.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="0", sink="5")
        # Verified with NetworkX: max flow is 12.0
        assert flow_value == 12.0

        flow_dict = g.max_flow(source="0", sink="5")
        assert isinstance(flow_dict, dict)
        # All flows should be non-negative
        for edge_id, flow in flow_dict.items():
            assert flow >= 0.0

    def test_single_path(self):
        """Test max flow on a single path."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=5.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=3.0, auto_create_nodes=True)
        g.add_edge("C", "D", capacity=7.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="A", sink="D")
        assert flow_value == 3.0  # Bottleneck is edge B->C

    def test_parallel_paths(self):
        """Test max flow with parallel paths."""
        g = create_flow_graph()
        g.add_edge("S", "A", capacity=5.0, auto_create_nodes=True)
        g.add_edge("S", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("A", "T", capacity=5.0, auto_create_nodes=True)
        g.add_edge("B", "T", capacity=10.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="S", sink="T")
        assert flow_value == 15.0


class TestMaxFlowEdgeCases:
    """Edge cases for max flow."""

    def test_same_source_sink(self):
        """Test when source equals sink (should raise error)."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)

        # Our implementation raises an error for source==sink
        with pytest.raises(ValueError, match="Source and sink must be different"):
            g.max_flow_value(source="A", sink="A")

    def test_no_path(self):
        """Test when there is no path from source to sink."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("C", "D", capacity=10.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="A", sink="D")
        assert flow_value == 0.0

    def test_single_edge(self):
        """Test with a single edge."""
        g = create_flow_graph()
        g.add_edge("S", "T", capacity=5.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="S", sink="T")
        assert flow_value == 5.0

    def test_zero_capacity_edge(self):
        """Test with zero capacity edge."""
        g = create_flow_graph()
        g.add_edge("S", "T", capacity=0.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="S", sink="T")
        assert flow_value == 0.0

    def test_nonexistent_source(self):
        """Test with non-existent source node."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)

        with pytest.raises(Exception):
            g.max_flow_value(source="X", sink="B")

    def test_nonexistent_sink(self):
        """Test with non-existent sink node."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)

        with pytest.raises(Exception):
            g.max_flow_value(source="A", sink="X")


class TestMaxFlowUndirected:
    """Test max flow on undirected graphs."""

    def test_undirected_simple(self):
        """Test max flow on undirected graph."""
        g = create_flow_graph(directed=False)
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=5.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="A", sink="C")
        assert flow_value == 5.0

    def test_undirected_multiple_paths(self):
        """Test max flow with multiple paths in undirected graph."""
        g = create_flow_graph(directed=False)
        g.add_edge("S", "A", capacity=5.0, auto_create_nodes=True)
        g.add_edge("S", "B", capacity=5.0, auto_create_nodes=True)
        g.add_edge("A", "T", capacity=5.0, auto_create_nodes=True)
        g.add_edge("B", "T", capacity=5.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="S", sink="T")
        assert flow_value == 10.0


class TestMaxFlowCapacity:
    """Test different capacity scenarios."""

    def test_default_capacity(self):
        """Test max flow with default capacity (1.0)."""
        g = create_flow_graph()
        g.add_edge("A", "B", auto_create_nodes=True)  # No explicit capacity
        g.add_edge("B", "C", auto_create_nodes=True)

        flow_value = g.max_flow_value(source="A", sink="C")
        assert flow_value == 1.0

    def test_mixed_capacities(self):
        """Test with mix of explicit and default capacities."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=100.0, auto_create_nodes=True)
        g.add_edge("B", "C", auto_create_nodes=True)  # Default capacity 1.0

        flow_value = g.max_flow_value(source="A", sink="C")
        assert flow_value == 1.0  # Bottleneck is default capacity

    def test_large_capacity(self):
        """Test with large capacities."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=1e12, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=1e12, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="A", sink="C")
        assert flow_value == 1e12

    def test_fractional_capacity(self):
        """Test with fractional capacities."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=2.5, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=1.7, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="A", sink="C")
        assert abs(flow_value - 1.7) < 1e-10


class TestMaxFlowFlowDict:
    """Test the flow dictionary returned by max_flow()."""

    def test_flow_dict_structure(self):
        """Test structure of the flow dict."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=5.0, auto_create_nodes=True)

        flow_dict = g.max_flow(source="A", sink="C")

        assert isinstance(flow_dict, dict)
        # Should have exactly 2 entries (both edges carry flow)
        assert len(flow_dict) == 2
        # All keys should be integers (edge IDs)
        for edge_id in flow_dict.keys():
            assert isinstance(edge_id, int)
        # All values should be floats
        for flow in flow_dict.values():
            assert isinstance(flow, float)

    def test_flow_dict_values_non_negative(self):
        """Test that all flow values are non-negative."""
        g = create_flow_graph()
        g.add_edge("0", "1", capacity=10.0, auto_create_nodes=True)
        g.add_edge("0", "2", capacity=10.0, auto_create_nodes=True)
        g.add_edge("1", "3", capacity=4.0, auto_create_nodes=True)
        g.add_edge("1", "2", capacity=2.0, auto_create_nodes=True)
        g.add_edge("2", "4", capacity=8.0, auto_create_nodes=True)
        g.add_edge("3", "4", capacity=10.0, auto_create_nodes=True)
        g.add_edge("3", "5", capacity=10.0, auto_create_nodes=True)
        g.add_edge("4", "5", capacity=10.0, auto_create_nodes=True)

        flow_dict = g.max_flow(source="0", sink="5")

        for edge_id, flow in flow_dict.items():
            assert flow >= 0.0, f"Edge {edge_id} has negative flow {flow}"

    def test_flow_dict_respects_capacity(self):
        """Test that flow doesn't exceed capacity."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=5.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=3.0, auto_create_nodes=True)

        flow_dict = g.max_flow(source="A", sink="C")

        # All flows should be at most 3.0 (the bottleneck)
        for edge_id, flow in flow_dict.items():
            assert flow <= 5.0  # Max capacity

    def test_flow_value_matches_flow_dict_sum(self):
        """Test that flow_value equals sum of flows through the network."""
        g = create_flow_graph()
        g.add_edge("S", "A", capacity=10.0, auto_create_nodes=True)
        g.add_edge("S", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("A", "T", capacity=5.0, auto_create_nodes=True)
        g.add_edge("B", "T", capacity=5.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="S", sink="T")
        flow_dict = g.max_flow(source="S", sink="T")

        # The flow value should equal the total flow
        assert flow_value == 10.0  # 5 through each path

        # The sum of flows on edges leaving source should equal flow_value
        # Edge IDs are 1-based: edge 1 is S->A, edge 2 is S->B
        total_from_source = flow_dict.get(1, 0.0) + flow_dict.get(2, 0.0)
        assert abs(flow_value - total_from_source) < 1e-10


class TestMaxFlowComplexGraphs:
    """Test on complex graph topologies."""

    def test_diamond_graph(self):
        """Test on a diamond-shaped graph."""
        g = create_flow_graph()
        g.add_edge("S", "A", capacity=3.0, auto_create_nodes=True)
        g.add_edge("S", "B", capacity=3.0, auto_create_nodes=True)
        g.add_edge("A", "T", capacity=2.0, auto_create_nodes=True)
        g.add_edge("B", "T", capacity=2.0, auto_create_nodes=True)
        g.add_edge("A", "B", capacity=1.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="S", sink="T")
        assert flow_value == 4.0  # Max possible through the diamond

    def test_grid_graph(self):
        """Test on a small grid graph."""
        g = create_flow_graph()
        # 3x3 grid - using string node labels
        # n0 - n1 - n2
        # |    |    |
        # n3 - n4 - n5
        # |    |    |
        # n6 - n7 - n8
        edges = [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n0", "n3"),
            ("n1", "n4"),
            ("n2", "n5"),
            ("n3", "n4"),
            ("n4", "n5"),
            ("n3", "n6"),
            ("n4", "n7"),
            ("n5", "n8"),
            ("n6", "n7"),
            ("n7", "n8"),
        ]
        for u, v in edges:
            g.add_edge(u, v, capacity=1.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="n0", sink="n8")
        assert flow_value == 2.0  # Two vertex-disjoint paths

    def test_bipartite_graph(self):
        """Test on a bipartite graph."""
        g = create_flow_graph()
        # Source S, left vertices L1,L2,L3, right vertices R1,R2,R3, sink T
        # Source to left
        g.add_edge("S", "L1", capacity=1.0, auto_create_nodes=True)
        g.add_edge("S", "L2", capacity=1.0, auto_create_nodes=True)
        g.add_edge("S", "L3", capacity=1.0, auto_create_nodes=True)
        # Left to right (some connections)
        g.add_edge("L1", "R1", capacity=1.0, auto_create_nodes=True)
        g.add_edge("L1", "R2", capacity=1.0, auto_create_nodes=True)
        g.add_edge("L2", "R1", capacity=1.0, auto_create_nodes=True)
        g.add_edge("L2", "R3", capacity=1.0, auto_create_nodes=True)
        g.add_edge("L3", "R2", capacity=1.0, auto_create_nodes=True)
        g.add_edge("L3", "R3", capacity=1.0, auto_create_nodes=True)
        # Right to sink
        g.add_edge("R1", "T", capacity=1.0, auto_create_nodes=True)
        g.add_edge("R2", "T", capacity=1.0, auto_create_nodes=True)
        g.add_edge("R3", "T", capacity=1.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="S", sink="T")
        assert flow_value == 3.0  # Maximum matching is 3


class TestMaxFlowConsistency:
    """Test consistency between max_flow and max_flow_value."""

    def test_value_consistency(self):
        """Test that max_flow and max_flow_value give consistent results."""
        g = create_flow_graph()
        g.add_edge("0", "1", capacity=10.0, auto_create_nodes=True)
        g.add_edge("0", "2", capacity=10.0, auto_create_nodes=True)
        g.add_edge("1", "3", capacity=4.0, auto_create_nodes=True)
        g.add_edge("1", "2", capacity=2.0, auto_create_nodes=True)
        g.add_edge("2", "4", capacity=8.0, auto_create_nodes=True)
        g.add_edge("3", "4", capacity=10.0, auto_create_nodes=True)
        g.add_edge("3", "5", capacity=10.0, auto_create_nodes=True)
        g.add_edge("4", "5", capacity=10.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="0", sink="5")
        flow_dict = g.max_flow(source="0", sink="5")

        # The flow value should be 12 (verified with NetworkX)
        assert flow_value == 12.0

        # Flow dict should have positive flows
        assert len(flow_dict) > 0
        assert all(f > 0 for f in flow_dict.values())

    def test_multiple_calls_same_result(self):
        """Test that multiple calls give the same result."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=5.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=3.0, auto_create_nodes=True)
        g.add_edge("A", "C", capacity=2.0, auto_create_nodes=True)

        v1 = g.max_flow_value(source="A", sink="C")
        v2 = g.max_flow_value(source="A", sink="C")
        assert v1 == v2


class TestMaxFlowSpecialCases:
    """Special case tests."""

    def test_self_loop(self):
        """Test graph with self-loop."""
        g = create_flow_graph()
        g.add_edge("A", "A", capacity=10.0, auto_create_nodes=True)  # Self-loop
        g.add_edge("A", "B", capacity=5.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="A", sink="B")
        assert flow_value == 5.0

    def test_cycle_in_graph(self):
        """Test graph with cycles."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=10.0, auto_create_nodes=True)
        g.add_edge("C", "A", capacity=10.0, auto_create_nodes=True)  # Creates cycle
        g.add_edge("B", "D", capacity=5.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="A", sink="D")
        assert flow_value == 5.0

    def test_multiple_edges_between_same_nodes(self):
        """Test with multiple edges between same node pair."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=3.0, auto_create_nodes=True)
        g.add_edge("A", "B", capacity=4.0, auto_create_nodes=True)  # Second edge
        g.add_edge("B", "C", capacity=10.0, auto_create_nodes=True)

        flow_value = g.max_flow_value(source="A", sink="C")
        assert flow_value == 7.0  # Both edges can carry flow


class TestMaxFlowErrors:
    """Test error handling."""

    def test_negative_capacity_raises_error(self):
        """Test that negative capacity raises an error."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=-5.0, auto_create_nodes=True)

        with pytest.raises(ValueError, match="Negative capacity"):
            g.max_flow_value(source="A", sink="B")

    def test_empty_graph_with_nodes(self):
        """Test on graph with nodes but no edges."""
        g = create_flow_graph()
        g.add_node(label="A")
        g.add_node(label="B")

        flow_value = g.max_flow_value(source="A", sink="B")
        assert flow_value == 0.0
