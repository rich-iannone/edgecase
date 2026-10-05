import itertools
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase import Graph
from edgecase.steps import KIND_QUERY


class TestIsMatching:
    """Tests for is_matching() method."""

    def test_valid_matching_simple(self):
        """Test valid matching with disjoint edges."""
        graph = Graph()
        for i in range(6):
            graph.add_node(label=str(i))

        # Create edges: 0-1, 2-3, 4-5
        graph.add_edge("0", "1")
        graph.add_edge("2", "3")
        graph.add_edge("4", "5")

        # Get edge IDs
        edge_ids = graph.get_edge_ids()

        assert graph.is_matching(edge_ids)

    def test_invalid_matching_shared_node(self):
        """Test invalid matching with shared endpoint."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        # B is shared between edges
        edge_ids = graph.get_edge_ids()

        assert not graph.is_matching(edge_ids)

    def test_empty_matching(self):
        """Test empty matching is valid."""
        graph = Graph()
        graph.add_path_graph(5)

        assert graph.is_matching([])

    def test_single_edge_matching(self):
        """Test single edge is valid matching."""
        graph = Graph()
        graph.add_path_graph(3)

        edge_ids = graph.get_edge_ids()

        assert graph.is_matching([edge_ids[0]])

    def test_perfect_matching(self):
        """Test perfect matching in complete graph."""
        graph = Graph()
        graph.add_full_graph(4)

        # Get nodes
        nodes = list(graph.get_node_ids())

        # Create perfect matching: pick 2 non-adjacent edges
        # In K4, find edges that don't share endpoints
        all_edges = graph.get_edge_ids()
        edges = []
        for i, e1 in enumerate(all_edges):
            edge1 = graph.get_edge(e1)
            for e2 in all_edges[i + 1 :]:
                edge2 = graph.get_edge(e2)
                # Check if edges don't share endpoints
                if (
                    edge1.source != edge2.source
                    and edge1.source != edge2.target
                    and edge1.target != edge2.source
                    and edge1.target != edge2.target
                ):
                    edges = [e1, e2]
                    break
            if edges:
                break

        assert len(edges) == 2
        assert graph.is_matching(edges)

    def test_matching_directed_graph(self):
        """Test matching in directed graph."""
        graph = Graph(directed=True)
        for i in range(4):
            graph.add_node(label=str(i))

        graph.add_edge("0", "1")
        graph.add_edge("2", "3")

        edge_ids = graph.get_edge_ids()

        assert graph.is_matching(edge_ids)


class TestMaximalMatching:
    """Tests for maximal_matching() method."""

    def test_path_graph(self):
        """Test maximal matching in path graph."""
        graph = Graph()
        graph.add_path_graph(6)  # 1-2-3-4-5-6

        matching = graph.maximal_matching()

        # Should match 3 edges
        assert len(matching) >= 2  # At least 2 edges
        assert graph.is_matching(matching)

    def test_complete_graph(self):
        """Test maximal matching in complete graph."""
        graph = Graph()
        graph.add_full_graph(8)

        matching = graph.maximal_matching()

        # Perfect matching: 4 edges covering all 8 nodes
        assert len(matching) == 4
        assert graph.is_matching(matching)

    def test_star_graph(self):
        """Test maximal matching in star graph."""
        graph = Graph()
        graph.add_star_graph(5)  # 1 center, 5 leaves

        matching = graph.maximal_matching()

        # Can only match 1 edge (center is matched once)
        assert len(matching) == 1
        assert graph.is_matching(matching)

    def test_cycle_graph(self):
        """Test maximal matching in cycle graph."""
        graph = Graph()
        graph.add_cycle_graph(6)

        matching = graph.maximal_matching()

        # Should match 3 edges
        assert len(matching) == 3
        assert graph.is_matching(matching)

    def test_disconnected_components(self):
        """Test maximal matching with multiple components."""
        graph = Graph()
        graph.add_path_graph(4)

        # Add second component
        offset = graph.count_nodes()
        for i in range(4):
            graph.add_node(label=f"comp2_{i}")

        nodes = list(graph.get_node_ids())
        graph.add_edge(nodes[offset], nodes[offset + 1])
        graph.add_edge(nodes[offset + 1], nodes[offset + 2])
        graph.add_edge(nodes[offset + 2], nodes[offset + 3])

        matching = graph.maximal_matching()

        # Should match edges from both components
        assert len(matching) >= 2
        assert graph.is_matching(matching)

    def test_empty_graph(self):
        """Test maximal matching in graph with no edges."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")

        matching = graph.maximal_matching()

        assert len(matching) == 0

    def test_single_edge(self):
        """Test maximal matching with single edge."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")

        matching = graph.maximal_matching()

        assert len(matching) == 1
        assert graph.is_matching(matching)


class TestMaxWeightMatching:
    """Tests for max_weight_matching() method."""

    def test_weighted_graph_simple(self):
        """Test maximum weight matching with different weights."""
        graph = Graph(weighted=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_node(label="D")

        # Heavy edge
        graph.add_edge("A", "B", weight=10.0)
        # Light edges
        graph.add_edge("B", "C", weight=1.0)
        graph.add_edge("C", "D", weight=1.0)

        matching = graph.max_weight_matching()

        # Should include heavy edge (first edge ID = 1)
        edge_ids = graph.get_edge_ids()
        heavy_edge_id = edge_ids[0]  # A-B edge

        assert heavy_edge_id in matching
        assert graph.is_matching(matching)

    def test_weighted_graph_balanced(self):
        """Test when weights are similar."""
        graph = Graph(weighted=True)
        for i in range(6):
            graph.add_node(label=str(i))

        # Create path with equal weights
        for i in range(5):
            graph.add_edge(str(i), str(i + 1), weight=1.0)

        matching = graph.max_weight_matching()

        # Should match 2-3 edges
        assert len(matching) >= 2
        assert graph.is_matching(matching)

    def test_unweighted_graph(self):
        """Test max weight matching on unweighted graph."""
        graph = Graph()
        graph.add_full_graph(6)

        matching = graph.max_weight_matching()

        # Perfect matching: 3 edges
        assert len(matching) == 3
        assert graph.is_matching(matching)

    def test_triangle_heavy_edge(self):
        """In a triangle only one edge fits, so the heaviest one wins."""
        graph = Graph(weighted=True)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")

        graph.add_edge("A", "B", weight=5.0)
        graph.add_edge("B", "C", weight=3.0)
        graph.add_edge("A", "C", weight=2.0)

        matching = graph.max_weight_matching()

        # The heaviest edge, A-B, is the best single edge
        edge_ids = graph.get_edge_ids()
        heavy_edge_id = edge_ids[0]  # A-B edge

        assert heavy_edge_id in matching
        assert len(matching) == 1
        assert graph.is_matching(matching)

    def test_weighted_perfect_matching(self):
        """Test perfect matching with weights."""
        graph = Graph(weighted=True)
        for i in range(4):
            graph.add_node(label=str(i))

        # Create weighted complete graph
        weights = {
            ("0", "1"): 5.0,
            ("0", "2"): 2.0,
            ("0", "3"): 1.0,
            ("1", "2"): 3.0,
            ("1", "3"): 4.0,
            ("2", "3"): 6.0,
        }

        for (src, tgt), weight in weights.items():
            graph.add_edge(src, tgt, weight=weight)

        matching = graph.max_weight_matching()

        # Should get 2 edges (perfect matching)
        assert len(matching) == 2
        assert graph.is_matching(matching)

        # Calculate total weight
        total_weight = sum(graph.get_edge(eid).weight for eid in matching)

        assert total_weight >= 9.0  # At least decent solution

    def test_grid_graph(self):
        """Test max weight matching on grid."""
        graph = Graph()
        graph.add_grid_2d(rows=3, cols=3)

        matching = graph.max_weight_matching()

        # Should get reasonable matching
        assert len(matching) >= 3
        assert graph.is_matching(matching)

    def test_compare_with_maximal(self):
        """Test that max_weight_matching finds valid maximal matching."""
        graph = Graph()
        graph.add_path_graph(10)

        matching1 = graph.maximal_matching()
        matching2 = graph.max_weight_matching()

        # Both should be valid matchings
        assert graph.is_matching(matching1)
        assert graph.is_matching(matching2)

        # Should have same cardinality for unweighted
        assert len(matching1) == len(matching2)

    def test_directed_graph_weighted(self):
        """Test max weight matching in directed graph."""
        graph = Graph(directed=True, weighted=True)
        for i in range(4):
            graph.add_node(label=str(i))

        graph.add_edge("0", "1", weight=5.0)
        graph.add_edge("1", "2", weight=2.0)
        graph.add_edge("2", "3", weight=3.0)

        matching = graph.max_weight_matching()

        assert len(matching) >= 1
        assert graph.is_matching(matching)


class TestMatchingEdgeCases:
    """Test edge cases for matching algorithms."""

    def test_self_loop(self):
        """Test matching with self-loop."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "A")  # Self-loop
        graph.add_edge("A", "B")

        matching = graph.maximal_matching()

        assert graph.is_matching(matching)

    def test_multi_edges(self):
        """Test matching with multi-edges."""
        graph = Graph()
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_edge("A", "B")
        graph.add_edge("A", "B")  # Duplicate edge

        # Can't use both edges (share endpoints)
        edge_ids = graph.get_edge_ids()

        assert not graph.is_matching(edge_ids)

    def test_large_complete_graph(self):
        """Test matching on larger complete graph."""
        graph = Graph()
        graph.add_full_graph(20)

        matching = graph.maximal_matching()

        # Perfect matching: 10 edges
        assert len(matching) == 10
        assert graph.is_matching(matching)

    def test_matching_preserves_graph(self):
        """Test that finding matching doesn't modify graph."""
        graph = Graph()
        graph.add_path_graph(6)

        original_edges = graph.count_edges()
        original_nodes = graph.count_nodes()

        _ = graph.maximal_matching()
        _ = graph.max_weight_matching()

        assert graph.count_edges() == original_edges
        assert graph.count_nodes() == original_nodes


class TestMatchingIntegration:
    """Integration tests combining matching with other graph operations."""

    def test_matching_after_node_removal(self):
        """Test matching after removing nodes."""
        graph = Graph()
        graph.add_full_graph(6)

        # Remove a node
        node_ids = list(graph.get_node_ids())
        graph.delete_node(node_ids[0])

        matching = graph.maximal_matching()
        assert graph.is_matching(matching)

    def test_matching_with_node_attributes(self):
        """Test matching preserves node attributes."""
        graph = Graph()
        for i in range(4):
            graph.add_node(label=str(i), color=f"color_{i}")

        graph.add_edge("0", "1")
        graph.add_edge("2", "3")

        matching = graph.maximal_matching()

        # Verify attributes still exist
        for node_id in graph.get_node_ids():
            node = graph.get_node(node_id)
            assert "color" in node.attributes

    def test_weighted_matching_different_weights(self):
        """Test that weights affect matching selection."""
        graph = Graph(weighted=True)
        for i in range(6):
            graph.add_node(label=str(i))

        # Create two paths with different weights
        graph.add_edge("0", "1", weight=1.0)
        graph.add_edge("1", "2", weight=1.0)
        graph.add_edge("2", "3", weight=1.0)

        graph.add_edge("3", "4", weight=10.0)  # Heavy edge
        graph.add_edge("4", "5", weight=1.0)

        matching = graph.max_weight_matching()

        # Should include the heavy edge
        heavy_edge = None
        for edge_id in graph.get_edge_ids():
            edge = graph.get_edge(edge_id)
            if edge.weight == 10.0:
                heavy_edge = edge.id
                break

        assert heavy_edge in matching


# Exact matchings and covers, validated against NetworkX (blossom, min-weight, Hopcroft-Karp,
# assignment) and by brute force or König's theorem (covers).


def _build(G, weighted=True):
    g = ec.Graph(weighted=weighted)
    g.add_nodes([str(n) for n in sorted(G)])
    for u, v, d in G.edges(data=True):
        g.add_edge(str(u), str(v), weight=d.get("weight", 1.0))
    return g


def _general_cases(count=150):
    for seed in range(count):
        rnd = random.Random(seed)
        G = nx.gnp_random_graph(rnd.randint(1, 14), rnd.choice([0.15, 0.3, 0.6]), seed=seed)
        kind = seed % 3
        for u, v in G.edges:
            G[u][v]["weight"] = [rnd.randint(1, 10), rnd.uniform(-2, 10), rnd.randint(-5, 20)][kind]
        yield G


def _bipartite_cases(count=120):
    for seed in range(count):
        rnd = random.Random(seed)
        a, b = rnd.randint(1, 8), rnd.randint(1, 8)
        B = nx.bipartite.random_graph(a, b, rnd.choice([0.2, 0.4, 0.8]), seed=seed)
        for u, v in B.edges:
            B[u][v]["weight"] = rnd.choice([rnd.randint(1, 9), rnd.uniform(0, 5)])
        yield B, a


def _total(g, eids):
    return sum(g.get_edge(e).weight for e in eids)


def _ids(g, pairs):
    return [g.find_edge_ids(u, v)[0] for u, v in pairs]


class TestExactMaxWeightMatching:
    def test_against_networkx(self):
        for G in _general_cases():
            g = _build(G)
            for mc in (False, True):
                ours = g.max_weight_matching(maxcardinality=mc)
                ref = nx.max_weight_matching(G, maxcardinality=mc)
                assert g.is_matching(ours)
                assert _total(g, ours) == pytest.approx(
                    sum(G[u][v]["weight"] for u, v in ref), abs=1e-7
                )
                if mc:
                    assert len(ours) == len(ref)

    def test_unweighted_is_maximum_cardinality(self):
        for G in _general_cases(60):
            g = _build(G, weighted=False)
            assert len(g.max_weight_matching()) == len(nx.max_weight_matching(G, True))

    def test_beats_greedy_and_weighted_flag(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B", "C", "D"])
        g.add_edges([("A", "B", 4.0), ("B", "C", 5.0), ("C", "D", 4.0)])
        assert _total(g, g.max_weight_matching()) == 8.0
        assert _total(g, g.max_weight_matching(method="greedy")) == 5.0
        assert len(g.max_weight_matching(weighted=False)) == 2
        with pytest.raises(ValueError, match="method"):
            g.max_weight_matching(method="hungarian")

    def test_parallel_edges_self_loops_and_direction(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 1.0), ("B", "A", 3.0), ("C", "C", 9.0)])
        assert g.max_weight_matching() == g.find_edge_ids("B", "A")
        assert ec.Graph().max_weight_matching() == []


class TestMinWeightMatching:
    def test_against_networkx(self):
        for G in _general_cases(100):
            g = _build(G)
            ours, ref = g.min_weight_matching(), nx.min_weight_matching(G)
            assert len(ours) == len(ref)
            assert _total(g, ours) == pytest.approx(
                sum(G[u][v]["weight"] for u, v in ref), abs=1e-7
            )


class TestMatchingPredicates:
    def test_against_networkx(self):
        for G in _general_cases(80):
            g = _build(G, weighted=False)
            edges = list(G.edges)
            rnd = random.Random(len(edges))
            for size in range(min(4, len(edges)) + 1):
                pick = rnd.sample(edges, size)
                eids = _ids(g, [(str(u), str(v)) for u, v in pick])
                assert g.is_maximal_matching(eids) == nx.is_maximal_matching(G, pick)
                assert g.is_perfect_matching(eids) == nx.is_perfect_matching(G, pick)

    def test_examples_and_invalid_input(self):
        g = ec.Graph().add_nodes(["A", "B", "C", "D"])
        g.add_edges([("A", "B"), ("B", "C"), ("C", "D")])
        ab, bc, cd = _ids(g, [("A", "B"), ("B", "C"), ("C", "D")])
        assert g.is_maximal_matching([bc]) and not g.is_maximal_matching([ab])
        assert g.is_perfect_matching([ab, cd]) and not g.is_perfect_matching([bc])
        assert not g.is_maximal_matching([ab, bc]) and not g.is_perfect_matching([999])
        loop = ec.Graph().add_nodes(["A"]).add_edge("A", "A")
        assert loop.is_maximal_matching([]) and not loop.is_maximal_matching(loop.get_edge_ids())


class TestBipartiteMatching:
    def test_hopcroft_karp_and_konig(self):
        for B, a in _bipartite_cases():
            g = _build(B, weighted=False)
            top = [str(i) for i in range(a)]
            matching = g.bipartite_maximum_matching(top_nodes=top)
            ref = nx.bipartite.hopcroft_karp_matching(B, top_nodes=range(a))
            assert g.is_matching(matching) and len(matching) == len(ref) // 2
            assert len(g.bipartite_maximum_matching()) == len(matching)  # automatic sides
            cover = g.bipartite_vertex_cover(top_nodes=top)
            assert g.is_vertex_cover(cover) and len(cover) == len(matching)

    def test_assignment_against_networkx(self):
        for B, a in _bipartite_cases():
            g = _build(B)
            top = [str(i) for i in range(a)]
            try:
                ref = nx.bipartite.minimum_weight_full_matching(B, top_nodes=range(a))
            except ValueError:
                with pytest.raises(ValueError, match="no full matching"):
                    g.bipartite_min_weight_full_matching(top_nodes=top)
                continue
            ours = g.bipartite_min_weight_full_matching(top_nodes=top)
            assert len(ours) == min(a, B.number_of_nodes() - a)
            assert _total(g, ours) == pytest.approx(
                sum(B[u][v]["weight"] for u, v in ref.items() if u < a), abs=1e-7
            )

    def test_errors(self):
        tri = ec.Graph().add_nodes(["A", "B", "C"])
        tri.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        for method in (tri.bipartite_maximum_matching, tri.bipartite_vertex_cover):
            with pytest.raises(ValueError, match="not bipartite"):
                method()
        path = ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("B", "C")])
        with pytest.raises(ValueError, match="same side"):
            path.bipartite_maximum_matching(top_nodes=["A", "B"])


class TestCovers:
    def test_min_edge_cover_against_networkx(self):
        for G in _general_cases(100):
            if not len(G) or list(nx.isolates(G)):
                continue
            g = _build(G, weighted=False)
            cover = g.min_edge_cover()
            assert g.is_edge_cover(cover)
            assert len(cover) == len({frozenset(e) for e in nx.min_edge_cover(G)})
            assert nx.is_edge_cover(
                G,
                [
                    tuple(int(g.get_label_from_id(n)) for n in g.get_node_ids_by_edge_id(e))
                    for e in cover
                ],
            )

    def test_min_edge_cover_edge_cases(self):
        assert ec.Graph().min_edge_cover() == []
        with pytest.raises(ValueError, match="isolated"):
            ec.Graph().add_nodes(["A", "B", "C"]).add_edge("A", "B").min_edge_cover()
        loop = ec.Graph().add_nodes(["A"]).add_edge("A", "A")
        assert loop.min_edge_cover() == loop.get_edge_ids()

    def test_weighted_vertex_cover_is_a_2_approximation(self):
        for seed in range(60):
            rnd = random.Random(seed)
            G = nx.gnp_random_graph(9, 0.35, seed=seed)
            g = _build(G, weighted=False)
            cost = {g.get_id_from_label(str(v)): float(rnd.randint(1, 6)) for v in G}
            g.set_node_attrs("cost", cost)
            cover = g.min_weighted_vertex_cover(weight_attr="cost")
            assert g.is_vertex_cover(cover)
            best = min(
                sum(cost[n] for n in subset)
                for k in range(len(cost) + 1)
                for subset in itertools.combinations(cost, k)
                if g.is_vertex_cover(list(subset))
            )
            assert sum(cost[n] for n in cover) <= 2 * best + 1e-9

    def test_vertex_cover_predicate_and_errors(self):
        g = ec.Graph().add_nodes(["hub", "a", "b"]).add_edges([("hub", "a"), ("hub", "b")])
        assert g.is_vertex_cover(["hub"]) and not g.is_vertex_cover(["a"])
        assert g.is_edge_cover(g.get_edge_ids()) and not g.is_edge_cover([999])
        g.set_node_attrs("cost", {g.get_id_from_label("a"): -1.0})
        with pytest.raises(ValueError, match="non-negative"):
            g.min_weighted_vertex_cover(weight_attr="cost")


class TestMatchingStepHooks:
    def test_steps(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B", "C", "D"])
        g.add_edges([("A", "B", 4.0), ("B", "C", 5.0), ("C", "D", 4.0)])
        g.max_weight_matching()
        step = g.steps[-1]
        assert step.kind == KIND_QUERY and step.code == "G.max_weight_matching()"
        assert step.delta.edges_selected == [("A", "B"), ("C", "D")]
        assert step.narration == "Matching of 2 edges with total weight 8."
        b = ec.Graph().add_nodes(list("ABCDE"))
        b.add_edges([("A", "D"), ("A", "E"), ("B", "E"), ("C", "E")])
        b.bipartite_vertex_cover()
        assert b.steps[-1].delta.nodes_selected == ["A", "E"]
        b.min_edge_cover()
        assert len(b.steps[-1].delta.edges_selected) == 3  # 5 nodes - matching of 2
