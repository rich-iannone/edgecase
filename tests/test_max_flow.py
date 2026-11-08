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

        flow_value = g.maximum_flow_value(source="0", target="5")

        # Verified with NetworkX: max flow is 12.0
        assert flow_value == 12.0

        flow_dict = g.maximum_flow(source="0", target="5")

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

        flow_value = g.maximum_flow_value(source="A", target="D")

        assert flow_value == 3.0  # Bottleneck is edge B->C

    def test_parallel_paths(self):
        """Test max flow with parallel paths."""
        g = create_flow_graph()
        g.add_edge("S", "A", capacity=5.0, auto_create_nodes=True)
        g.add_edge("S", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("A", "T", capacity=5.0, auto_create_nodes=True)
        g.add_edge("B", "T", capacity=10.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="S", target="T")

        assert flow_value == 15.0


class TestMaxFlowEdgeCases:
    """Edge cases for max flow."""

    def test_same_source_sink(self):
        """Test when source equals sink (should raise error)."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)

        # Our implementation raises an error for source==sink
        with pytest.raises(ValueError, match="Source and target must be different"):
            g.maximum_flow_value(source="A", target="A")

    def test_no_path(self):
        """Test when there is no path from source to sink."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("C", "D", capacity=10.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="A", target="D")

        assert flow_value == 0.0

    def test_single_edge(self):
        """Test with a single edge."""
        g = create_flow_graph()
        g.add_edge("S", "T", capacity=5.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="S", target="T")

        assert flow_value == 5.0

    def test_zero_capacity_edge(self):
        """Test with zero capacity edge."""
        g = create_flow_graph()
        g.add_edge("S", "T", capacity=0.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="S", target="T")

        assert flow_value == 0.0

    def test_nonexistent_source(self):
        """Test with non-existent source node."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)

        with pytest.raises(Exception):
            g.maximum_flow_value(source="X", target="B")

    def test_nonexistent_sink(self):
        """Test with non-existent sink node."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)

        with pytest.raises(Exception):
            g.maximum_flow_value(source="A", target="X")


class TestMaxFlowUndirected:
    """Test max flow on undirected graphs."""

    def test_undirected_simple(self):
        """Test max flow on undirected graph."""
        g = create_flow_graph(directed=False)
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=5.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="A", target="C")

        assert flow_value == 5.0

    def test_undirected_multiple_paths(self):
        """Test max flow with multiple paths in undirected graph."""
        g = create_flow_graph(directed=False)
        g.add_edge("S", "A", capacity=5.0, auto_create_nodes=True)
        g.add_edge("S", "B", capacity=5.0, auto_create_nodes=True)
        g.add_edge("A", "T", capacity=5.0, auto_create_nodes=True)
        g.add_edge("B", "T", capacity=5.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="S", target="T")

        assert flow_value == 10.0


class TestMaxFlowCapacity:
    """Test different capacity scenarios."""

    def test_default_capacity(self):
        """Test max flow with default capacity (1.0)."""
        g = create_flow_graph()
        g.add_edge("A", "B", auto_create_nodes=True)  # No explicit capacity
        g.add_edge("B", "C", auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="A", target="C")

        assert flow_value == 1.0

    def test_mixed_capacities(self):
        """Test with mix of explicit and default capacities."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=100.0, auto_create_nodes=True)
        g.add_edge("B", "C", auto_create_nodes=True)  # Default capacity 1.0

        flow_value = g.maximum_flow_value(source="A", target="C")

        assert flow_value == 1.0  # Bottleneck is default capacity

    def test_large_capacity(self):
        """Test with large capacities."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=1e12, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=1e12, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="A", target="C")

        assert flow_value == 1e12

    def test_fractional_capacity(self):
        """Test with fractional capacities."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=2.5, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=1.7, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="A", target="C")

        assert abs(flow_value - 1.7) < 1e-10


class TestMaxFlowNamedCapacity:
    """Regression: a named capacity attribute that an edge lacked silently became 1.0."""

    def test_missing_named_attribute_raises(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", bw=5.0, auto_create_nodes=True)
        g.add_edge("B", "C", auto_create_nodes=True)
        with pytest.raises(ValueError, match="no numeric 'bw' attribute"):
            g.maximum_flow_value(source="A", target="C", capacity="bw")
        with pytest.raises(ValueError, match="no numeric 'bw' attribute"):
            g.maximum_flow(source="A", target="C", capacity="bw")
        with pytest.raises(ValueError, match="no numeric 'bw' attribute"):
            g.minimum_cut("A", "C", capacity="bw")

    def test_non_numeric_named_attribute_raises(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", bw="fast", auto_create_nodes=True)
        with pytest.raises(ValueError, match="no numeric 'bw' attribute"):
            g.maximum_flow_value(source="A", target="B", capacity="bw")

    def test_named_weight_uses_edge_weight(self):
        g = Graph(directed=True, weighted=True)
        g.add_edge("A", "B", weight=3.0, auto_create_nodes=True)
        g.add_edge("B", "C", weight=2.0, auto_create_nodes=True)
        assert g.maximum_flow_value(source="A", target="C", capacity="weight") == 2.0

    def test_named_cost_missing_raises(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", price=1.0, auto_create_nodes=True)
        g.add_edge("B", "C", auto_create_nodes=True)
        with pytest.raises(ValueError, match="no numeric 'price' attribute"):
            g.minimum_cost_flow("A", "C", cost="price")

    def test_default_capacity_still_falls_back(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", capacity=4.0, auto_create_nodes=True)
        g.add_edge("B", "C", auto_create_nodes=True)
        assert g.maximum_flow_value(source="A", target="C") == 1.0


class TestMaxFlowFlowDict:
    """Test the flow dictionary returned by maximum_flow()."""

    def test_flow_dict_structure(self):
        """Test structure of the flow dict."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=5.0, auto_create_nodes=True)

        flow_dict = g.maximum_flow(source="A", target="C")

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

        flow_dict = g.maximum_flow(source="0", target="5")

        for edge_id, flow in flow_dict.items():
            assert flow >= 0.0, f"Edge {edge_id} has negative flow {flow}"

    def test_flow_dict_respects_capacity(self):
        """Test that flow doesn't exceed capacity."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=5.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=3.0, auto_create_nodes=True)

        flow_dict = g.maximum_flow(source="A", target="C")

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

        flow_value = g.maximum_flow_value(source="S", target="T")
        flow_dict = g.maximum_flow(source="S", target="T")

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

        flow_value = g.maximum_flow_value(source="S", target="T")

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

        flow_value = g.maximum_flow_value(source="n0", target="n8")

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

        flow_value = g.maximum_flow_value(source="S", target="T")

        assert flow_value == 3.0  # Maximum matching is 3


class TestMaxFlowConsistency:
    """Test consistency between maximum_flow and maximum_flow_value."""

    def test_value_consistency(self):
        """Test that maximum_flow and maximum_flow_value give consistent results."""
        g = create_flow_graph()
        g.add_edge("0", "1", capacity=10.0, auto_create_nodes=True)
        g.add_edge("0", "2", capacity=10.0, auto_create_nodes=True)
        g.add_edge("1", "3", capacity=4.0, auto_create_nodes=True)
        g.add_edge("1", "2", capacity=2.0, auto_create_nodes=True)
        g.add_edge("2", "4", capacity=8.0, auto_create_nodes=True)
        g.add_edge("3", "4", capacity=10.0, auto_create_nodes=True)
        g.add_edge("3", "5", capacity=10.0, auto_create_nodes=True)
        g.add_edge("4", "5", capacity=10.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="0", target="5")
        flow_dict = g.maximum_flow(source="0", target="5")

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

        v1 = g.maximum_flow_value(source="A", target="C")
        v2 = g.maximum_flow_value(source="A", target="C")

        assert v1 == v2


class TestMaxFlowSpecialCases:
    """Special case tests."""

    def test_self_loop(self):
        """Test graph with self-loop."""
        g = create_flow_graph()
        g.add_edge("A", "A", capacity=10.0, auto_create_nodes=True)  # Self-loop
        g.add_edge("A", "B", capacity=5.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="A", target="B")

        assert flow_value == 5.0

    def test_cycle_in_graph(self):
        """Test graph with cycles."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=10.0, auto_create_nodes=True)
        g.add_edge("B", "C", capacity=10.0, auto_create_nodes=True)
        g.add_edge("C", "A", capacity=10.0, auto_create_nodes=True)  # Creates cycle
        g.add_edge("B", "D", capacity=5.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="A", target="D")

        assert flow_value == 5.0

    def test_multiple_edges_between_same_nodes(self):
        """Test with multiple edges between same node pair."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=3.0, auto_create_nodes=True)
        g.add_edge("A", "B", capacity=4.0, auto_create_nodes=True)  # Second edge
        g.add_edge("B", "C", capacity=10.0, auto_create_nodes=True)

        flow_value = g.maximum_flow_value(source="A", target="C")

        assert flow_value == 7.0  # Both edges can carry flow


class TestMaxFlowErrors:
    """Test error handling."""

    def test_negative_capacity_raises_error(self):
        """Test that negative capacity raises an error."""
        g = create_flow_graph()
        g.add_edge("A", "B", capacity=-5.0, auto_create_nodes=True)

        with pytest.raises(ValueError, match="Negative capacity"):
            g.maximum_flow_value(source="A", target="B")

    def test_empty_graph_with_nodes(self):
        """Test on graph with nodes but no edges."""
        g = create_flow_graph()
        g.add_node(label="A")
        g.add_node(label="B")

        flow_value = g.maximum_flow_value(source="A", target="B")

        assert flow_value == 0.0


class TestMaxFlowMultiEdges:
    """Parallel and antiparallel edges each keep their own capacity and flow."""

    def test_parallel_directed_edges_report_own_flow(self):
        """Two parallel edges with capacities 2 and 3 carry 2 and 3, not 5 each."""
        g = Graph(directed=True)
        g.add_node(label="S")
        g.add_node(label="T")
        g.add_edge("S", "T", capacity=2)
        g.add_edge("S", "T", capacity=3)
        small, large = g.find_edge_ids("S", "T")

        assert g.maximum_flow("S", "T") == {small: 2.0, large: 3.0}
        assert g.maximum_flow_value("S", "T") == 5.0

    def test_parallel_undirected_edges_report_own_flow(self):
        """Parallel undirected edges (stored either way round) split the flow by capacity."""
        g = Graph()
        g.add_node(label="S")
        g.add_node(label="T")
        g.add_edge("S", "T", capacity=2)
        g.add_edge("T", "S", capacity=3)
        first, second = g.find_edge_ids("S", "T")  # both orientations, in creation order

        assert g.maximum_flow("S", "T") == {first: 2.0, second: 3.0}

    def test_flow_dict_respects_capacity_and_conservation(self):
        """On random directed multigraphs the per-edge flows form a valid maximum flow."""
        nx = pytest.importorskip("networkx")
        import random

        rng = random.Random(5)
        for _ in range(60):
            n = rng.randint(3, 8)
            g = Graph(directed=True)
            for i in range(n):
                g.add_node(label=str(i))
            reference = nx.DiGraph()
            reference.add_nodes_from(range(n))
            for _ in range(rng.randint(n, 3 * n)):
                u, v = rng.sample(range(n), 2)
                cap = rng.randint(1, 6)
                g.add_edge(str(u), str(v), capacity=cap)
                prior = reference[u][v]["capacity"] if reference.has_edge(u, v) else 0
                reference.add_edge(u, v, capacity=prior + cap)

            expected = nx.maximum_flow_value(reference, 0, n - 1)
            flow = g.maximum_flow("0", str(n - 1))
            assert g.maximum_flow_value("0", str(n - 1)) == pytest.approx(expected)

            net = {node_id: 0.0 for node_id in g.get_node_ids()}
            for edge_id, amount in flow.items():
                edge = g.get_edge(edge_id)
                assert amount <= edge.attributes["capacity"] + 1e-9
                net[edge.source] -= amount
                net[edge.target] += amount
            source_id = g.get_id_from_label("0")
            sink_id = g.get_id_from_label(str(n - 1))
            assert net[sink_id] == pytest.approx(expected)
            assert all(
                net[node_id] == pytest.approx(0.0)
                for node_id in net
                if node_id not in (source_id, sink_id)
            )
