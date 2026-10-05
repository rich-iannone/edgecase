import math
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph


class TestBridges:
    """Tests for bridges method."""

    def test_empty_graph_no_bridges(self):
        """Empty graph has no bridges."""
        graph = Graph()

        bridges = graph.bridges()

        assert bridges == []

    def test_single_node_no_bridges(self):
        """Single node with no edges has no bridges."""
        graph = Graph()
        graph.add_node(0)

        bridges = graph.bridges()

        assert bridges == []

    def test_single_edge_is_bridge(self):
        """Single edge is a bridge."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        n1 = graph.get_id_from_label("A")
        n2 = graph.get_id_from_label("B")
        graph.add_edge(n1, n2)

        bridges = graph.bridges()

        assert len(bridges) == 1

    def test_path_all_edges_are_bridges(self):
        """In a path graph, all edges are bridges."""
        graph = Graph()
        graph.add_path_graph(5)

        bridges = graph.bridges()

        assert len(bridges) == 4  # Path has 4 edges

    def test_cycle_no_bridges(self):
        """Cycle graph has no bridges."""
        graph = Graph()
        graph.add_cycle_graph(5)

        bridges = graph.bridges()

        assert len(bridges) == 0

    def test_tree_all_edges_are_bridges(self):
        """In a tree, all edges are bridges."""
        graph = Graph()

        # Create a simple tree manually
        graph.add_path_graph(5)  # A path is a tree

        bridges = graph.bridges()
        num_edges = len(graph.get_edges_as_tuples())

        assert len(bridges) == num_edges
        assert num_edges == 4  # Path with 5 nodes has 4 edges

    def test_cycle_with_tail(self):
        """Cycle with a tail has one bridge (the tail edge)."""
        graph = Graph()
        graph.add_cycle_graph(4)
        nodes = sorted(graph.get_node_ids())

        # Add a tail using a new node
        graph.add_node(label="tail")
        tail = graph.get_id_from_label("tail")
        graph.add_edge(nodes[-1], tail)

        bridges = graph.bridges()

        assert len(bridges) == 1

    def test_star_graph_all_edges_are_bridges(self):
        """Star graph has all edges as bridges."""
        graph = Graph()
        graph.add_star_graph(5)

        bridges = graph.bridges()
        num_edges = len(graph.get_edges_as_tuples())

        assert len(bridges) == num_edges

    def test_directed_simple_bridge(self):
        """Test bridge detection in directed graph."""
        graph = Graph(directed=True)
        graph.add_path_graph(3)

        bridges = graph.bridges()

        # Both edges are bridges (no cycles)
        assert len(bridges) == 2

    def test_directed_cycle_no_bridge(self):
        """Directed cycle has no bridges."""
        graph = Graph(directed=True)
        graph.add_cycle_graph(5)

        bridges = graph.bridges()

        # A directed cycle has no bridges (all edges are in a cycle)
        assert len(bridges) == 0

    def test_bridge_edge_ids(self):
        """Verify that returned values are valid edge IDs."""
        graph = Graph()
        graph.add_path_graph(5)

        bridges = graph.bridges()
        all_edge_ids = {edge_id for edge_id, _, _ in graph.get_edges_as_tuples()}

        # All bridge IDs should be valid edge IDs
        for bridge_id in bridges:
            assert bridge_id in all_edge_ids


class TestHasBridges:
    """Tests for has_bridges method."""

    def test_empty_graph_has_no_bridges(self):
        """Empty graph has no bridges."""
        graph = Graph()

        assert not graph.has_bridges()

    def test_single_edge_has_bridge(self):
        """Single edge is a bridge."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        n1 = graph.get_id_from_label("A")
        n2 = graph.get_id_from_label("B")
        graph.add_edge(n1, n2)

        assert graph.has_bridges()

    def test_cycle_has_no_bridges(self):
        """Cycle has no bridges."""
        graph = Graph()
        graph.add_cycle_graph(5)

        assert not graph.has_bridges()

    def test_path_has_bridges(self):
        """Path has bridges."""
        graph = Graph()
        graph.add_path_graph(5)

        assert graph.has_bridges()

    def test_consistency_with_bridges_method(self):
        """has_bridges should be consistent with bridges."""
        # Test on various graphs
        test_cases = [
            ("cycle", lambda g: g.add_cycle_graph(5)),
            ("path", lambda g: g.add_path_graph(5)),
            ("star", lambda g: g.add_star_graph(5)),
        ]

        for name, builder in test_cases:
            graph = Graph()
            builder(graph)

            has_bridges = graph.has_bridges()
            bridges_list = graph.bridges()

            assert has_bridges == (len(bridges_list) > 0), (
                f"Inconsistency for {name}: has_bridges={has_bridges}, len(bridges)={len(bridges_list)}"
            )


class TestBridgesIntegration:
    """Integration tests for bridge detection."""

    def test_bridges_with_multiple_components(self):
        """Test bridges in disconnected graph."""
        graph = Graph()
        graph.add_path_graph(3)  # Component 1
        graph.add_cycle_graph(3)  # Component 2

        bridges = graph.bridges()

        # The path has 2 bridges, cycle has 0
        assert len(bridges) == 2

    def test_bridge_removal_increases_components(self):
        """Verify that removing a bridge increases connected components."""
        graph = Graph()
        graph.add_path_graph(3)

        # Get initial component count
        initial_components = graph.count_weakly_connected_components()

        # Find and remove a bridge
        bridges = graph.bridges()

        assert len(bridges) > 0

        bridge_id = bridges[0]
        graph.delete_edge_by_id(bridge_id)

        # Component count should increase
        final_components = graph.count_weakly_connected_components()

        assert final_components == initial_components + 1

    def test_no_bridges_after_adding_redundant_edge(self):
        """Adding edge to make cycle removes bridges."""
        graph = Graph()

        # Start with a path (has bridges)
        graph.add_path_graph(4)  # 0-1-2-3

        assert graph.has_bridges()

        # Complete the cycle
        node_ids = sorted(graph.get_node_ids())
        first_node = node_ids[0]
        last_node = node_ids[-1]
        graph.add_edge(first_node, last_node)

        # Now it's a cycle (no bridges)
        bridges = graph.bridges()

        assert len(bridges) == 0

    def test_large_graph_performance(self):
        """Test bridge detection on a larger graph."""
        graph = Graph()

        # Long path: 100 nodes
        graph.add_path_graph(100)

        bridges = graph.bridges()

        # Should have 99 bridges (all edges in a path)
        assert len(bridges) == 99

    def test_multigraph_bridges(self):
        """Test bridges in a multigraph (multiple edges between nodes)."""
        graph = Graph()

        # Create nodes and edges
        graph.add_path_graph(3)
        nodes = sorted(graph.get_node_ids())

        # Add parallel edge between first two nodes
        graph.add_edge(nodes[0], nodes[1])

        bridges = graph.bridges()

        # The parallel edge means the first connection is not a bridge,
        # but the last edge still is. However, our algorithm may
        # still count both original edges as bridges due to how it works.
        # For a path with parallel edges, we expect at least one bridge
        assert len(bridges) >= 1

    def test_self_loop_not_bridge(self):
        """Self loops are not bridges."""
        graph = Graph()

        # Simple path with a self-loop
        graph.add_path_graph(3)
        node_ids = sorted(graph.get_node_ids())
        middle_node = node_ids[1]
        graph.add_edge(middle_node, middle_node)  # Self loop

        bridges = graph.bridges()

        # Should have 2 bridges (the path edges), not the self-loop
        assert len(bridges) == 2

        # Verify no self-loop is in bridges
        for bridge_id in bridges:
            result = graph.get_node_ids_by_edge_id(bridge_id)
            if result:
                u, v = result
                assert u != v


class TestBridgesParallelEdges:
    """Regression: bridges were found per node pair, so a doubled edge counted as a bridge."""

    def test_doubled_edge_is_not_a_bridge(self):
        g = Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("A", "B"), ("B", "C")])
        assert g.bridges() == g.find_edge_ids("B", "C")
        doubled = Graph().add_nodes(["A", "B"]).add_edges([("A", "B"), ("A", "B")])
        assert doubled.bridges() == []
        assert not doubled.has_bridges()
        reversed_pair = Graph().add_nodes(["A", "B"]).add_edges([("A", "B"), ("B", "A")])
        assert reversed_pair.bridges() == []
        assert not reversed_pair.has_bridges()

    def test_self_loop_is_not_a_bridge(self):
        g = Graph().add_nodes(["A", "B"]).add_edges([("A", "B"), ("B", "B")])
        assert g.bridges() == g.find_edge_ids("A", "B")

    def test_matches_networkx_and_handles_long_paths(self):
        import random

        import networkx as nx

        for seed in range(30):
            G = nx.gnp_random_graph(20, random.Random(seed).choice([0.08, 0.12, 0.2]), seed=seed)
            g = Graph().add_nodes([str(n) for n in G])
            if G.number_of_edges():
                g.add_edges([(str(u), str(v)) for u, v in G.edges])
            ours = sorted(
                tuple(sorted(int(g.get_label_from_id(x)) for x in g.get_node_ids_by_edge_id(e)))
                for e in g.bridges()
            )
            assert ours == sorted(tuple(sorted(e)) for e in nx.bridges(G))
        # iterative, so no recursion limit on long paths
        assert len(Graph().add_path_graph(20000).bridges()) == 19999


# Local bridges, validated against NetworkX.


def _random_undirected():
    for seed in range(25):
        rnd = random.Random(seed)
        G = nx.gnp_random_graph(18, rnd.choice([0.1, 0.15, 0.25]), seed=seed)
        G.add_nodes_from(range(18))
        for u, v in G.edges:
            G[u][v]["weight"] = rnd.choice([0.5, 1, 2, 3])
        yield G


def _build(G: nx.Graph, weighted: bool = False) -> ec.Graph:
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G.nodes])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


def _lab(g: ec.Graph, nid: int) -> int:
    return int(g.get_label_from_id(nid))


class TestLocalBridges:
    @pytest.mark.parametrize("weighted", [False, True])
    def test_match_networkx(self, weighted):
        for G in _random_undirected():
            g = _build(G, weighted)
            ours = {}
            for e, span in g.local_bridges():
                key = tuple(sorted(_lab(g, x) for x in g.get_node_ids_by_edge_id(e)))
                ours[key] = span
            ref = {
                tuple(sorted((u, v))): span
                for u, v, span in nx.local_bridges(
                    G, with_span=True, weight="weight" if weighted else None
                )
            }
            assert ours.keys() == ref.keys()
            for k, span in ref.items():
                assert ours[k] == span or (math.isinf(ours[k]) and math.isinf(span))

    def test_example_and_types(self):
        g = ec.Graph().add_nodes(list("ABCDEFGH"))
        g.add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E"), ("E", "F"), ("F", "D"),
             ("C", "D"), ("A", "G"), ("G", "F"), ("E", "H")]
        )  # fmt: skip
        spans = dict(g.local_bridges())
        cd = g.find_edge_ids("C", "D")[0]
        eh = g.find_edge_ids("E", "H")[0]
        assert spans[cd] == 4 and isinstance(spans[cd], int)
        assert math.isinf(spans[eh])
        assert g.local_bridges(with_span=False) == sorted(spans)

    def test_steps(self):
        g = ec.Graph().add_nodes(list("ABCDEF"))
        g.add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "C"), ("E", "F")]
        )
        g.local_bridges()
        assert g.steps[-1].delta.edges_selected == [("E", "F")]
