import pytest
import edgecase as ec


class TestMinCostFlowBasic:
    """Basic functionality tests for min_cost_flow."""

    def test_simple_path(self):
        """Simple path should send flow along the only path."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", weight=10.0, cost=5.0)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T")

        assert flow == 10.0
        assert cost == 50.0  # 10 units * 5 cost
        assert len(flow_dict) == 1

    def test_two_paths_prefer_cheaper(self):
        """Should prefer cheaper path."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="T")
            .add_edge("S", "A", weight=10.0, cost=1.0)  # Cheap path
            .add_edge("S", "B", weight=10.0, cost=5.0)  # Expensive path
            .add_edge("A", "T", weight=10.0, cost=1.0)
            .add_edge("B", "T", weight=10.0, cost=1.0)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T")

        assert flow == 20.0  # Uses both paths

        # Cheap path: 10 * (1+1) = 20
        # Expensive path: 10 * (5+1) = 60
        # Total: 80
        assert cost == 80.0

    def test_diamond_network(self):
        """Diamond shaped network with different costs."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="T")
            .add_edge("S", "A", weight=5.0, cost=1.0)
            .add_edge("S", "B", weight=5.0, cost=2.0)
            .add_edge("A", "T", weight=5.0, cost=1.0)
            .add_edge("B", "T", weight=5.0, cost=2.0)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T")

        assert flow == 10.0  # Both paths used

        # S->A->T: 5 * 2 = 10
        # S->B->T: 5 * 4 = 20
        assert cost == 30.0


class TestMinCostFlowWithDemand:
    """Tests for min_cost_flow with specific demand."""

    def test_exact_demand(self):
        """Should send exactly the demanded flow."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", weight=100.0, cost=5.0)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T", demand=25.0)

        assert flow == 25.0
        assert cost == 125.0  # 25 * 5

    def test_demand_uses_cheapest_paths_first(self):
        """Partial demand should use cheapest paths first."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="T")
            .add_edge("S", "A", weight=10.0, cost=1.0)  # Cheap
            .add_edge("S", "B", weight=10.0, cost=10.0)  # Expensive
            .add_edge("A", "T", weight=10.0, cost=1.0)
            .add_edge("B", "T", weight=10.0, cost=10.0)
        )

        # Only need 5 units - should use only cheap path
        flow, cost, flow_dict = graph.min_cost_flow("S", "T", demand=5.0)

        assert flow == 5.0
        assert cost == 10.0  # 5 * (1 + 1)

    def test_unsatisfiable_demand_raises(self):
        """Should raise error if demand cannot be satisfied."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", weight=10.0, cost=1.0)
        )

        with pytest.raises(ValueError, match="Cannot satisfy demand"):
            graph.min_cost_flow("S", "T", demand=100.0)

    def test_zero_demand(self):
        """Zero demand should return zero flow and cost."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", weight=10.0, cost=5.0)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T", demand=0.0)

        assert flow == 0.0
        assert cost == 0.0
        assert len(flow_dict) == 0


class TestMinCostFlowCustomAttributes:
    """Tests for custom capacity and cost attributes."""

    def test_custom_capacity_attribute(self):
        """Should use custom capacity attribute."""
        graph = (
            ec.Graph(directed=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", bandwidth=50)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T", capacity="bandwidth")

        assert flow == 50.0

    def test_custom_cost_attribute(self):
        """Should use custom cost attribute."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", weight=10.0, price=7.0)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T", cost="price")

        assert flow == 10.0
        assert cost == 70.0  # 10 * 7

    def test_both_custom_attributes(self):
        """Should use both custom capacity and cost attributes."""
        graph = (
            ec.Graph(directed=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", bandwidth=20, price=3)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T", capacity="bandwidth", cost="price")

        assert flow == 20.0
        assert cost == 60.0  # 20 * 3


class TestMinCostFlowErrors:
    """Tests for error handling."""

    def test_undirected_graph_raises(self):
        """Should raise error for undirected graphs."""
        graph = ec.Graph(directed=False).add_node(label="S").add_node(label="T").add_edge("S", "T")

        with pytest.raises(ValueError, match="directed"):
            graph.min_cost_flow("S", "T")

    def test_invalid_source_raises(self):
        """Should raise error for non-existent source."""
        graph = ec.Graph(directed=True).add_node(label="S").add_node(label="T")

        with pytest.raises(ValueError):
            graph.min_cost_flow("X", "T")

    def test_invalid_sink_raises(self):
        """Should raise error for non-existent sink."""
        graph = ec.Graph(directed=True).add_node(label="S").add_node(label="T")

        with pytest.raises(ValueError):
            graph.min_cost_flow("S", "X")

    def test_same_source_sink_raises(self):
        """Should raise error if source equals sink."""
        graph = ec.Graph(directed=True).add_node(label="S")

        with pytest.raises(ValueError, match="different"):
            graph.min_cost_flow("S", "S")

    def test_negative_demand_raises(self):
        """Should raise error for negative demand."""
        graph = ec.Graph(directed=True).add_node(label="S").add_node(label="T").add_edge("S", "T")

        with pytest.raises(ValueError, match="non-negative"):
            graph.min_cost_flow("S", "T", demand=-5.0)


class TestMinCostFlowEdgeCases:
    """Edge case tests."""

    def test_no_path(self):
        """No path should return zero flow."""
        graph = (
            ec.Graph(directed=True).add_node(label="S").add_node(label="T")
            # No edge connecting them
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T")

        assert flow == 0.0
        assert cost == 0.0
        assert len(flow_dict) == 0

    def test_self_loops_ignored(self):
        """Self-loops should be ignored."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "S", weight=100.0, cost=0.0)  # Self-loop
            .add_edge("S", "T", weight=10.0, cost=5.0)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T")

        assert flow == 10.0
        assert cost == 50.0

    def test_disconnected_sink(self):
        """Disconnected sink should return zero flow."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="A")
            .add_node(label="T")
            .add_edge("S", "A", weight=10.0, cost=1.0)
            # T is not reachable
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T")

        assert flow == 0.0
        assert cost == 0.0


class TestMinCostFlowIntegration:
    """Integration tests with more complex scenarios."""

    def test_transportation_problem(self):
        """Classic transportation problem."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="Source")
            .add_node(label="W1")
            .add_node(label="W2")
            .add_node(label="S1")
            .add_node(label="S2")
            .add_node(label="Sink")
            # Supply constraints
            .add_edge("Source", "W1", weight=30.0, cost=0.0)
            .add_edge("Source", "W2", weight=40.0, cost=0.0)
            # Shipping costs
            .add_edge("W1", "S1", weight=100.0, cost=8.0)
            .add_edge("W1", "S2", weight=100.0, cost=5.0)
            .add_edge("W2", "S1", weight=100.0, cost=4.0)
            .add_edge("W2", "S2", weight=100.0, cost=7.0)
            # Demand constraints
            .add_edge("S1", "Sink", weight=35.0, cost=0.0)
            .add_edge("S2", "Sink", weight=35.0, cost=0.0)
        )

        flow, cost, flow_dict = graph.min_cost_flow("Source", "Sink")

        # Max possible flow is min(supply, demand) = min(70, 70) = 70
        assert flow == 70.0

        # Algorithm should find minimum cost assignment
        assert cost > 0  # Some cost incurred

    def test_with_node_ids(self):
        """Test using node IDs instead of labels."""
        graph = ec.Graph(directed=True, weighted=True)
        s_id = graph.add_node(label="S").get_id_from_label("S")
        t_id = graph.add_node(label="T").get_id_from_label("T")
        graph.add_edge("S", "T", weight=10.0, cost=5.0)

        flow, cost, flow_dict = graph.min_cost_flow(s_id, t_id)

        assert flow == 10.0
        assert cost == 50.0

    def test_parallel_edges(self):
        """Parallel edges should have capacities summed."""
        graph = ec.Graph(directed=True, weighted=True)
        graph.add_node(label="S")
        graph.add_node(label="T")
        graph.add_edge("S", "T", weight=10.0, cost=5.0)
        graph.add_edge("S", "T", weight=10.0, cost=3.0)  # Cheaper parallel edge

        flow, cost, flow_dict = graph.min_cost_flow("S", "T")

        # Total capacity is 20
        assert flow == 20.0

        # Uses cheaper edge cost (3) for the flow
        assert cost == 60.0  # 20 * 3

    def test_flow_dict_has_positive_flows(self):
        """Flow dict should only contain edges with positive flow."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="T")
            .add_edge("S", "A", weight=10.0, cost=1.0)
            .add_edge("S", "B", weight=0.0, cost=1.0)  # Zero capacity
            .add_edge("A", "T", weight=10.0, cost=1.0)
            .add_edge("B", "T", weight=10.0, cost=1.0)
        )

        flow, cost, flow_dict = graph.min_cost_flow("S", "T")

        # Only S->A and A->T should have flow
        assert len(flow_dict) == 2

        for edge_id, edge_flow in flow_dict.items():
            assert edge_flow > 0
