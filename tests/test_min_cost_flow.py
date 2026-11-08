import pytest
import edgecase as ec


class TestMinCostFlowBasic:
    """Basic functionality tests for minimum_cost_flow."""

    def test_simple_path(self):
        """Simple path should send flow along the only path."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", weight=10.0, cost=5.0)
        )

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T")

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

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T")

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

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T")

        assert flow == 10.0  # Both paths used

        # S->A->T: 5 * 2 = 10
        # S->B->T: 5 * 4 = 20
        assert cost == 30.0


class TestMinCostFlowWithDemand:
    """Tests for minimum_cost_flow with specific demand."""

    def test_exact_demand(self):
        """Should send exactly the demanded flow."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", weight=100.0, cost=5.0)
        )

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T", demand=25.0)

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
        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T", demand=5.0)

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
            graph.minimum_cost_flow("S", "T", demand=100.0)

    def test_zero_demand(self):
        """Zero demand should return zero flow and cost."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", weight=10.0, cost=5.0)
        )

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T", demand=0.0)

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

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T", capacity="bandwidth")

        assert flow == 50.0

    def test_custom_cost_attribute(self):
        """Should use custom cost attribute."""
        graph = (
            ec.Graph(directed=True, weighted=True)
            .add_node(label="S")
            .add_node(label="T")
            .add_edge("S", "T", weight=10.0, price=7.0)
        )

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T", cost="price")

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

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T", capacity="bandwidth", cost="price")

        assert flow == 20.0
        assert cost == 60.0  # 20 * 3


class TestMinCostFlowErrors:
    """Tests for error handling."""

    def test_undirected_graph_raises(self):
        """Should raise error for undirected graphs."""
        graph = ec.Graph(directed=False).add_node(label="S").add_node(label="T").add_edge("S", "T")

        with pytest.raises(ValueError, match="directed"):
            graph.minimum_cost_flow("S", "T")

    def test_invalid_source_raises(self):
        """Should raise error for non-existent source."""
        graph = ec.Graph(directed=True).add_node(label="S").add_node(label="T")

        with pytest.raises(ValueError):
            graph.minimum_cost_flow("X", "T")

    def test_invalid_sink_raises(self):
        """Should raise error for non-existent sink."""
        graph = ec.Graph(directed=True).add_node(label="S").add_node(label="T")

        with pytest.raises(ValueError):
            graph.minimum_cost_flow("S", "X")

    def test_same_source_sink_raises(self):
        """Should raise error if source equals sink."""
        graph = ec.Graph(directed=True).add_node(label="S")

        with pytest.raises(ValueError, match="different"):
            graph.minimum_cost_flow("S", "S")

    def test_negative_demand_raises(self):
        """Should raise error for negative demand."""
        graph = ec.Graph(directed=True).add_node(label="S").add_node(label="T").add_edge("S", "T")

        with pytest.raises(ValueError, match="non-negative"):
            graph.minimum_cost_flow("S", "T", demand=-5.0)


class TestMinCostFlowEdgeCases:
    """Edge case tests."""

    def test_no_path(self):
        """No path should return zero flow."""
        graph = (
            ec.Graph(directed=True).add_node(label="S").add_node(label="T")
            # No edge connecting them
        )

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T")

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

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T")

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

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T")

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

        flow, cost, flow_dict = graph.minimum_cost_flow("Source", "Sink")

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

        flow, cost, flow_dict = graph.minimum_cost_flow(s_id, t_id)

        assert flow == 10.0
        assert cost == 50.0

    def test_parallel_edges(self):
        """Parallel edges each keep their own capacity and cost."""
        graph = ec.Graph(directed=True, weighted=True)
        graph.add_node(label="S")
        graph.add_node(label="T")
        graph.add_edge("S", "T", weight=10.0, cost=5.0)
        graph.add_edge("S", "T", weight=10.0, cost=3.0)  # Cheaper parallel edge
        expensive_id, cheap_id = graph.find_edge_ids("S", "T")

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T")

        # Total capacity is 20
        assert flow == 20.0

        # Each edge fills to its own capacity at its own cost: 10 * 3 + 10 * 5
        assert cost == 80.0
        assert flow_dict == {cheap_id: 10.0, expensive_id: 10.0}

    def test_parallel_edges_prefer_cheaper_edge(self):
        """With spare capacity, flow goes on the cheaper of two parallel edges only."""
        graph = ec.Graph(directed=True, weighted=True)
        graph.add_node(label="S")
        graph.add_node(label="T")
        graph.add_edge("S", "T", weight=10.0, cost=5.0)
        graph.add_edge("S", "T", weight=10.0, cost=3.0)
        _, cheap_id = graph.find_edge_ids("S", "T")

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T", demand=6.0)

        assert (flow, cost) == (6.0, 18.0)
        assert flow_dict == {cheap_id: 6.0}

    def test_antiparallel_edges(self):
        """A real B -> A edge doesn't share capacity or cost with A -> B's cancellation arc."""
        graph = ec.Graph(directed=True)
        for label in ["S", "A", "B", "T"]:
            graph.add_node(label=label)
        graph.add_edge("S", "A", capacity=2, cost=1)
        graph.add_edge("S", "B", capacity=2, cost=1)
        graph.add_edge("A", "B", capacity=2, cost=1)
        graph.add_edge("B", "A", capacity=2, cost=1)
        graph.add_edge("A", "T", capacity=1, cost=1)
        graph.add_edge("B", "T", capacity=3, cost=1)

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T")

        # Max flow is 4: S->A->T (1), S->B->T (2), S->A->B->T (1)
        assert flow == 4.0
        assert cost == 9.0
        b_to_a = graph.find_edge_ids("B", "A")[0]
        assert b_to_a not in flow_dict

    def test_negative_cost_cycle_raises(self):
        """A reachable negative-cost cycle makes the cheapest flow undefined."""
        graph = ec.Graph(directed=True)
        for label in ["S", "A", "B", "T"]:
            graph.add_node(label=label)
        graph.add_edge("S", "A", capacity=1, cost=1)
        graph.add_edge("A", "B", capacity=5, cost=-3)
        graph.add_edge("B", "A", capacity=5, cost=1)
        graph.add_edge("A", "T", capacity=1, cost=1)

        with pytest.raises(ValueError, match="negative-cost cycle"):
            graph.minimum_cost_flow("S", "T")

    def test_matches_networkx_on_multigraphs(self):
        """Flow value, cost, and a feasible per-edge flow match NetworkX's network simplex."""
        nx = pytest.importorskip("networkx")
        import random

        rng = random.Random(11)
        for _ in range(60):
            n = rng.randint(3, 7)
            reference = nx.MultiDiGraph()
            reference.add_nodes_from(range(n))
            graph = ec.Graph(directed=True)
            for i in range(n):
                graph.add_node(label=str(i))
            for _ in range(rng.randint(n, 3 * n)):
                u, v = rng.sample(range(n), 2)
                cap, unit_cost = rng.randint(1, 6), rng.randint(0, 9)
                reference.add_edge(u, v, capacity=cap, weight=unit_cost)
                graph.add_edge(str(u), str(v), capacity=cap, cost=unit_cost)

            collapsed = nx.DiGraph()
            collapsed.add_nodes_from(range(n))
            for u, v, data in reference.edges(data=True):
                prior = collapsed[u][v]["capacity"] if collapsed.has_edge(u, v) else 0
                collapsed.add_edge(u, v, capacity=prior + data["capacity"])
            max_value = nx.maximum_flow_value(collapsed, 0, n - 1)
            if max_value == 0:
                continue

            nx.set_node_attributes(reference, 0, "demand")
            reference.nodes[0]["demand"] = -max_value
            reference.nodes[n - 1]["demand"] = max_value
            expected_cost, _ = nx.network_simplex(reference)

            flow, cost, flow_dict = graph.minimum_cost_flow("0", str(n - 1))
            assert flow == pytest.approx(max_value)
            assert cost == pytest.approx(expected_cost)

            net = {node_id: 0.0 for node_id in graph.get_node_ids()}
            edge_cost_sum = 0.0
            for edge_id, edge_flow in flow_dict.items():
                edge = graph.get_edge(edge_id)
                assert edge_flow <= edge.attributes["capacity"] + 1e-9
                net[edge.source] -= edge_flow
                net[edge.target] += edge_flow
                edge_cost_sum += edge_flow * edge.attributes["cost"]
            assert edge_cost_sum == pytest.approx(cost)
            source_id, sink_id = graph.get_id_from_label("0"), graph.get_id_from_label(str(n - 1))
            assert net[sink_id] == pytest.approx(max_value)
            assert all(
                net[node_id] == pytest.approx(0.0)
                for node_id in net
                if node_id not in (source_id, sink_id)
            )

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

        flow, cost, flow_dict = graph.minimum_cost_flow("S", "T")

        # Only S->A and A->T should have flow
        assert len(flow_dict) == 2

        for edge_id, edge_flow in flow_dict.items():
            assert edge_flow > 0
