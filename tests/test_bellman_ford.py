import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph


def get_path_labels(graph, path_ids):
    """Helper to convert path of node IDs to labels for readability."""
    return [graph.get_label_from_id(nid) for nid in path_ids]


class TestBellmanFordBasic:
    """Basic Bellman-Ford functionality tests."""

    def test_simple_path(self):
        """Test shortest path in simple weighted graph."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
        )

        path = graph.bellman_ford_shortest_path("A", "C", return_list=True)
        assert path == [1, 2, 3]  # A, B, C

    def test_negative_weights(self):
        """Test with negative edge weights."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=4.0)
            .add_edge("A", "C", weight=2.0)
            .add_edge("C", "B", weight=-3.0)  # Negative weight
            .add_edge("B", "D", weight=1.0)
        )

        path = graph.bellman_ford_shortest_path("A", "D", return_list=True)
        # Should go A -> C -> B -> D to take advantage of negative weight
        assert path == [1, 3, 2, 4]  # A, C, B, D

    def test_direct_connection(self):
        """Test direct edge between nodes."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B", weight=5.0)
        )

        path = graph.bellman_ford_shortest_path("A", "B", return_list=True)
        assert path == [1, 2]  # A, B

    def test_no_path(self):
        """Test when no path exists."""
        graph = (
            Graph(directed=True).add_node(label="A").add_node(label="B").add_node(label="isolated")
        )

        result = graph.bellman_ford_shortest_path("A", "isolated", return_list=True)
        assert result is None


class TestBellmanFordNegativeCycles:
    """Tests for negative cycle detection."""

    def test_negative_cycle_detection(self):
        """Test that negative cycles are detected."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=-3.0)
            .add_edge("C", "A", weight=1.0)  # Creates negative cycle: total = -1
        )

        with pytest.raises(ValueError, match="negative weight cycle"):
            graph.bellman_ford_shortest_path("A", "B")

    def test_negative_cycle_self_loop(self):
        """Test negative self-loop detection."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "A", weight=-1.0)  # Negative self-loop
            .add_edge("A", "B", weight=1.0)
        )

        with pytest.raises(ValueError, match="negative weight cycle"):
            graph.bellman_ford_shortest_path("A", "B")


class TestBellmanFordComplexGraphs:
    """Tests with more complex graph structures."""

    def test_multiple_paths_chooses_shortest(self):
        """Test that algorithm finds truly shortest path."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=10.0)
            .add_edge("A", "C", weight=1.0)
            .add_edge("C", "D", weight=1.0)
            .add_edge("D", "B", weight=1.0)
        )

        path = graph.bellman_ford_shortest_path("A", "B", return_list=True)
        # Should go A -> C -> D -> B (total: 3) instead of A -> B (10)
        assert path == [1, 3, 4, 2]  # A, C, D, B

    def test_star_graph(self):
        """Test with star topology."""
        graph = (
            Graph(directed=True)
            .add_node(label="hub")
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("hub", "A", weight=1.0)
            .add_edge("hub", "B", weight=2.0)
            .add_edge("hub", "C", weight=3.0)
        )

        path = graph.bellman_ford_shortest_path("hub", "C", return_list=True)
        assert path == [1, 4]  # hub, C

    def test_mixed_positive_negative_weights(self):
        """Test graph with both positive and negative weights."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B", weight=5.0)
            .add_edge("B", "C", weight=-2.0)
            .add_edge("C", "D", weight=3.0)
            .add_edge("A", "D", weight=10.0)  # Alternative longer path
        )

        path = graph.bellman_ford_shortest_path("A", "D", return_list=True)
        # Should be A -> B -> C -> D = 5 + (-2) + 3 = 6
        assert path == [1, 2, 3, 4]  # A, B, C, D


class TestBellmanFordEdgeCases:
    """Edge case tests."""

    def test_single_node(self):
        """Test path from node to itself."""
        graph = Graph(directed=True).add_node(label="A")

        path = graph.bellman_ford_shortest_path("A", "A", return_list=True)
        assert path == [1]  # A

    def test_zero_weight_edges(self):
        """Test with zero-weight edges."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=0.0)
            .add_edge("B", "C", weight=0.0)
        )

        path = graph.bellman_ford_shortest_path("A", "C", return_list=True)
        assert path == [1, 2, 3]  # A, B, C

    def test_large_negative_weights(self):
        """Test with large negative weights (no cycle)."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=100.0)
            .add_edge("B", "C", weight=-50.0)
        )

        path = graph.bellman_ford_shortest_path("A", "C", return_list=True)
        assert path == [1, 2, 3]  # A, B, C


class TestBellmanFordUndirected:
    """Test Bellman-Ford on undirected graphs."""

    def test_undirected_graph(self):
        """Test on undirected graph."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
        )

        path = graph.bellman_ford_shortest_path("A", "C", return_list=True)
        assert path == [1, 2, 3]  # A, B, C

    def test_undirected_negative_weights(self):
        """Test undirected graph with negative weights."""
        # Note: Negative weights in undirected graphs can easily create cycles
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=5.0)
            .add_edge("B", "C", weight=3.0)
        )

        path = graph.bellman_ford_shortest_path("A", "C", return_list=True)
        assert path == [1, 2, 3]  # A, B, C


class TestBellmanFordDualBehavior:
    """Test the dual-behavior interface (return list or store as attributes)."""

    def test_return_list_behavior(self):
        """Test return_list=True returns path as list."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
        )

        path = graph.bellman_ford_shortest_path("A", "C", return_list=True)
        assert isinstance(path, list)
        assert path == [1, 2, 3]

    def test_store_as_attributes_behavior(self):
        """Test return_list=False stores positions as node attributes."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
        )

        result = graph.bellman_ford_shortest_path("A", "C", return_list=False)

        # Should return self for chaining
        assert result is graph

        # Check that attributes were set
        ndf = graph.get_node_df()
        path_order_col = ndf.filter(ndf["bellman_ford_path_order"].is_not_null())
        assert len(path_order_col) == 3  # A, B, C on path

        # Verify order: A=0, B=1, C=2
        a_order = ndf.filter(ndf["id"] == 1).select("bellman_ford_path_order")[0, 0]
        b_order = ndf.filter(ndf["id"] == 2).select("bellman_ford_path_order")[0, 0]
        c_order = ndf.filter(ndf["id"] == 3).select("bellman_ford_path_order")[0, 0]
        assert a_order == 0
        assert b_order == 1
        assert c_order == 2

    def test_custom_attr_name(self):
        """Test custom attribute name."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=1.0)
            .add_edge("B", "C", weight=2.0)
        )

        graph.bellman_ford_shortest_path("A", "C", attr_name="my_path", return_list=False)

        ndf = graph.get_node_df()
        assert "my_path" in ndf.columns
        assert "bellman_ford_path_order" not in ndf.columns

    def test_method_chaining(self):
        """Test that return_list=False enables method chaining."""
        graph = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B", weight=-1.0)
            .add_edge("B", "C", weight=2.0)
            .bellman_ford_shortest_path("A", "C", return_list=False)
        )

        # Should be able to continue chaining
        assert isinstance(graph, Graph)
        ndf = graph.get_node_df()
        assert "bellman_ford_path_order" in ndf.columns


class TestNegativeCycles:
    """negative_edge_cycle and find_negative_cycle, validated against NetworkX."""

    def test_against_networkx(self):
        for seed in range(150):
            rnd = random.Random(seed)
            directed = seed % 3 != 0
            G = nx.gnp_random_graph(rnd.randint(1, 9), 0.35, seed=seed, directed=directed)
            for u, v in G.edges:
                low = -3 if directed else (-1 if seed % 4 == 0 else 1)
                G[u][v]["weight"] = rnd.choice([rnd.randint(1, 5), rnd.randint(low, 5)])
            g = ec.Graph(directed=directed, weighted=True).add_nodes([str(n) for n in G])
            for u, v, w in G.edges(data="weight"):
                g.add_edge(str(u), str(v), weight=w)
            assert g.negative_edge_cycle() == nx.negative_edge_cycle(G)
            for source in G:
                reach = {source} | nx.descendants(G, source)
                expected = nx.negative_edge_cycle(G.subgraph(reach).copy())
                if not expected:
                    with pytest.raises(ValueError, match="no negative cycle"):
                        g.find_negative_cycle(str(source))
                    continue
                cycle = [int(g.get_label_from_id(n)) for n in g.find_negative_cycle(str(source))]
                assert cycle[0] == cycle[-1]
                assert sum(G[a][b]["weight"] for a, b in zip(cycle, cycle[1:])) < 0

    def test_examples(self):
        fx = ec.Graph(directed=True, weighted=True).add_nodes(["USD", "EUR", "GBP"])
        fx.add_edges([("USD", "EUR", -0.10), ("EUR", "GBP", -0.15), ("GBP", "USD", 0.20)])
        assert fx.negative_edge_cycle()
        cycle = [fx.get_label_from_id(n) for n in fx.find_negative_cycle("USD")]
        assert cycle == ["USD", "EUR", "GBP", "USD"]
        assert not fx.negative_edge_cycle(weighted=False)
        undirected = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=-1.0)
        assert len(undirected.find_negative_cycle("A")) == 3
