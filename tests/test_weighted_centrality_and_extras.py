"""Tests for weighted shortest-path centralities and the second batch of NetworkX-parity methods.

- weights in betweenness_centrality, edge_betweenness_centrality, closeness_centrality and
  harmonic_centrality (`weighted=None` follows the graph, like shortest_path())
- bidirectional_dijkstra, all_topological_sorts, partition_quality
- cmty_fast_greedy rewrite (lazy-bound CNM) and its resolution= parameter
"""

import itertools
import math
import random
import time

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_QUERY


def _build(G: nx.Graph, weighted: bool) -> ec.Graph:
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G.nodes])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


def _by_label(g: ec.Graph, scores: dict) -> dict:
    return {int(g.get_label_from_id(k)): v for k, v in scores.items()}


def _weighted_cases():
    for seed in range(15):
        rnd = random.Random(seed)
        for directed in (False, True):
            G = nx.gnp_random_graph(15, 0.22, seed=seed, directed=directed)
            for u, v in G.edges:
                G[u][v]["weight"] = rnd.choice([0.5, 1, 1.5, 2, 3])
            yield G, _build(G, True)


def _triangle_detour() -> ec.Graph:
    """A-C is heavy, so the cheapest A-C route detours through B."""
    return (
        ec.Graph(weighted=True)
        .add_nodes(["A", "B", "C"])
        .add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 5.0)])
    )


# --- weighted centralities ----------------------------------------------------------


class TestWeightedCentralitiesAgainstNetworkX:
    def test_betweenness(self):
        for G, g in _weighted_cases():
            ours = _by_label(g, g.betweenness_centrality(return_dict=True))
            ref = nx.betweenness_centrality(G, weight="weight")
            assert ours == pytest.approx(ref, abs=1e-12)

    @pytest.mark.parametrize("normalized", [True, False])
    def test_edge_betweenness(self, normalized):
        for G, g in _weighted_cases():
            scores = g.edge_betweenness_centrality(normalized=normalized, return_dict=True)
            ours = {}
            for eid, value in scores.items():
                s, t = g.get_node_ids_by_edge_id(eid)
                ours[(int(g.get_label_from_id(s)), int(g.get_label_from_id(t)))] = value
            ref = nx.edge_betweenness_centrality(G, normalized=normalized, weight="weight")
            for (u, v), val in ref.items():
                got = ours.get((u, v))
                if got is None and not G.is_directed():
                    got = ours[(v, u)]
                assert got == pytest.approx(val, abs=1e-12)

    @pytest.mark.parametrize("wf_improved", [True, False])
    def test_closeness(self, wf_improved):
        for G, g in _weighted_cases():
            mode = "in" if G.is_directed() else "out"  # NetworkX uses incoming distances
            ours = _by_label(
                g, g.closeness_centrality(mode=mode, wf_improved=wf_improved, return_dict=True)
            )
            ref = nx.closeness_centrality(G, distance="weight", wf_improved=wf_improved)
            assert ours == pytest.approx(ref, abs=1e-12)

    def test_harmonic(self):
        for G, g in _weighted_cases():
            ours = _by_label(g, g.harmonic_centrality(return_dict=True))
            # NetworkX sums over incoming distances; mode="out" matches it on the reverse
            ref = nx.harmonic_centrality(G.reverse() if G.is_directed() else G, distance="weight")
            assert ours == pytest.approx(ref, abs=1e-12)


class TestWeightedFlag:
    def test_follows_graph_by_default_and_can_be_overridden(self):
        g = _triangle_detour()
        b = g.get_id_from_label("B")
        assert g.betweenness_centrality(return_dict=True)[b] == 1.0
        assert g.betweenness_centrality(weighted=False, return_dict=True)[b] == 0.0

    def test_weighted_true_on_unweighted_graph_uses_stored_weights(self):
        g = (
            ec.Graph()
            .add_nodes(["A", "B", "C"])
            .add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 5.0)])
        )
        b = g.get_id_from_label("B")
        assert g.betweenness_centrality(return_dict=True)[b] == 0.0
        assert g.betweenness_centrality(weighted=True, return_dict=True)[b] == 1.0

    def test_consistent_with_shortest_path_length(self):
        g = _triangle_detour()
        close = g.closeness_centrality(return_dict=True)
        a = g.get_id_from_label("A")
        total = g.shortest_path_length("A", "B") + g.shortest_path_length("A", "C")
        assert close[a] == pytest.approx(2 / total)

    def test_unweighted_results_unchanged_on_weighted_graph_when_disabled(self):
        G = nx.gnp_random_graph(20, 0.2, seed=3)
        for u, v in G.edges:
            G[u][v]["weight"] = 2.5
        weighted, plain = _build(G, True), _build(G, False)
        assert weighted.betweenness_centrality(
            weighted=False, return_dict=True
        ) == plain.betweenness_centrality(return_dict=True)
        assert weighted.closeness_centrality(
            weighted=False, return_dict=True
        ) == plain.closeness_centrality(return_dict=True)

    def test_parallel_edges_only_cheapest_carries_paths(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"])
        g.add_edges([("A", "B", 1.0), ("A", "B", 3.0)])
        cheap, dear = g.find_edge_ids("A", "B")
        raw = g.edge_betweenness_centrality(normalized=False, return_dict=True)
        assert raw[cheap] == pytest.approx(1.0) and raw[dear] == 0.0

    def test_harmonic_cutoff_is_a_distance(self):
        g = _triangle_detour()
        h = g.harmonic_centrality(cutoff=1, return_dict=True)
        a = g.get_id_from_label("A")
        assert h[a] == pytest.approx(1.0)  # only B (distance 1) is within the cutoff

    def test_negative_weights_rejected(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=-1.0)
        for method in (
            "betweenness_centrality",
            "edge_betweenness_centrality",
            "closeness_centrality",
            "harmonic_centrality",
        ):
            with pytest.raises(ValueError, match="non-negative"):
                getattr(g, method)(return_dict=True)

    def test_harmonic_weights_argument_points_to_weighted(self):
        with pytest.raises(ValueError, match="weighted"):
            _triangle_detour().harmonic_centrality(weights="weight")

    def test_explicit_flag_recorded_in_computations(self):
        g = _triangle_detour().betweenness_centrality(weighted=False).closeness_centrality()
        assert ("betweenness_centrality", {"weighted": False}, "betweenness_centrality") in (
            g._computations
        )
        assert ("closeness_centrality", {}, "closeness_centrality") in g._computations


# --- bidirectional Dijkstra ------------------------------------------------------------


class TestBidirectionalDijkstra:
    def test_matches_networkx(self):
        for G, g in _weighted_cases():
            for weighted in (True, False):
                gg = g if weighted else _build(G, False)
                for a, b in itertools.islice(itertools.permutations(G.nodes, 2), 0, 200, 7):
                    res = gg.bidirectional_dijkstra(str(a), str(b))
                    if not nx.has_path(G, a, b):
                        assert res is None
                        continue
                    length, path = res
                    ref = nx.shortest_path_length(G, a, b, weight="weight" if weighted else None)
                    labels = [int(gg.get_label_from_id(n)) for n in path]
                    cost = sum(
                        G[x][y]["weight"] if weighted else 1 for x, y in zip(labels, labels[1:])
                    )
                    assert length == pytest.approx(ref) and cost == pytest.approx(ref)
                    assert labels[0] == a and labels[-1] == b

    def test_types_and_trivial_cases(self):
        g = _triangle_detour()
        length, path = g.bidirectional_dijkstra("A", "C")
        assert length == 2.0 and [g.get_label_from_id(n) for n in path] == ["A", "B", "C"]
        hops = ec.Graph().add_nodes(["A", "B"]).add_edge("A", "B")
        assert hops.bidirectional_dijkstra("A", "B") == (1, [1, 2])
        assert isinstance(hops.bidirectional_dijkstra("A", "B")[0], int)
        assert hops.bidirectional_dijkstra("A", "A") == (0, [1])
        lonely = ec.Graph().add_nodes(["A", "B"])
        assert lonely.bidirectional_dijkstra("A", "B") is None


# --- all topological sorts -------------------------------------------------------------


class TestAllTopologicalSorts:
    def test_matches_networkx(self):
        for seed in range(20):
            D = nx.gnp_random_graph(7, 0.3, seed=seed, directed=True)
            D = nx.DiGraph([(u, v) for u, v in D.edges if u < v])
            D.add_nodes_from(range(7))
            g = ec.Graph(directed=True).add_nodes([str(n) for n in range(7)])
            if D.number_of_edges():
                g.add_edges([(str(u), str(v)) for u, v in D.edges])
            ours = [[int(g.get_label_from_id(n)) for n in s] for s in g.all_topological_sorts()]
            assert ours == sorted(ours)  # lexicographic by ID (= label order here)
            assert sorted(ours) == sorted(nx.all_topological_sorts(D))

    def test_counts_cap_and_errors(self):
        free = ec.Graph(directed=True).add_nodes(list("ABCD"))
        assert len(free.all_topological_sorts()) == 24
        assert len(free.all_topological_sorts(max_sorts=5)) == 5
        chain = ec.Graph(directed=True).add_nodes(list("ABC")).add_edges([("A", "B"), ("B", "C")])
        assert chain.all_topological_sorts() == [[1, 2, 3]]
        assert ec.Graph(directed=True).all_topological_sorts() == [[]]
        with pytest.raises(ValueError, match="max_sorts"):
            free.all_topological_sorts(max_sorts=0)
        cyc = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edges([("A", "B"), ("B", "A")])
        with pytest.raises(ValueError, match="cycle"):
            cyc.all_topological_sorts()
        with pytest.raises(ValueError, match="directed"):
            ec.Graph().add_nodes(["A"]).all_topological_sorts()

    def test_every_ordering_is_valid(self):
        g = ec.Graph(directed=True).add_nodes(list("ABCDE"))
        g.add_edges([("A", "C"), ("B", "C"), ("C", "D"), ("B", "E")])
        orders = g.all_topological_sorts()
        assert len(orders) == len({tuple(o) for o in orders})
        for order in orders:
            pos = {n: i for i, n in enumerate(order)}
            assert all(pos[s] < pos[t] for _, s, t in g.get_edges_as_tuples())


# --- partition quality ---------------------------------------------------------------


class TestPartitionQuality:
    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        for seed in range(15):
            rnd = random.Random(seed)
            G = nx.gnp_random_graph(14, 0.25, seed=seed, directed=directed)
            g = ec.Graph(directed=directed).add_nodes([str(n) for n in G.nodes])
            if G.number_of_edges():
                g.add_edges([(str(u), str(v)) for u, v in G.edges])
            part = {n: rnd.randrange(3) for n in G.nodes}
            groups = [x for x in ({n for n in G if part[n] == k} for k in range(3)) if x]
            ours = g.partition_quality({str(n): c for n, c in part.items()})
            assert ours == pytest.approx(nx.community.partition_quality(G, groups))

    def test_two_triangles(self):
        g = ec.Graph().add_nodes(list("ABCDEF"))
        g.add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
        coverage, performance = g.partition_quality([["A", "B", "C"], ["D", "E", "F"]])
        assert coverage == pytest.approx(6 / 7)
        assert performance == pytest.approx(14 / 15)  # only the C-D pair is misclassified
        assert g.partition_quality([list("ABCDEF")]) == pytest.approx((1.0, 7 / 15))

    def test_edge_cases_and_errors(self):
        assert ec.Graph().add_nodes(["A", "B"]).partition_quality([["A"], ["B"]]) == (0.0, 1.0)
        assert ec.Graph().add_nodes(["A"]).partition_quality([["A"]]) == (0.0, 0.0)
        g = ec.Graph().add_nodes(["A", "B"]).add_edge("A", "B")
        with pytest.raises(ValueError, match="missing"):
            g.partition_quality([["A"]])


# --- fast greedy -----------------------------------------------------------------------


class TestFastGreedy:
    @staticmethod
    def _from_nx(G):
        g = ec.Graph().add_nodes([str(n) for n in G.nodes])
        g.add_edges([(str(u), str(v)) for u, v in G.edges])
        return g

    @pytest.mark.parametrize("resolution", [1.0, 0.5, 1.5])
    def test_matches_networkx_modularity(self, resolution):
        graphs = [
            nx.Graph(nx.karate_club_graph().edges()),
            nx.powerlaw_cluster_graph(400, 3, 0.3, seed=1),
            nx.planted_partition_graph(5, 20, 0.4, 0.02, seed=3),
        ]
        for G in graphs:
            g = self._from_nx(G)
            m = g.cmty_fast_greedy(resolution=resolution, return_dict=True)
            ref = nx.community.greedy_modularity_communities(G, resolution=resolution)
            ref_q = nx.community.modularity(G, ref, resolution=resolution)
            # Greedy merge order can differ on ties; require at least NetworkX's quality
            assert g.modularity(m, resolution=resolution) >= ref_q - 2e-3

    def test_resolution_controls_granularity(self):
        karate = ec.karate_club()
        counts = [
            len(set(karate.cmty_fast_greedy(resolution=r, return_dict=True).values()))
            for r in (0.5, 1.0, 1.5)
        ]
        assert counts[0] <= counts[1] < counts[2]

    def test_large_graph_is_fast(self):
        g = self._from_nx(nx.powerlaw_cluster_graph(20000, 3, 0.3, seed=1))
        start = time.perf_counter()
        m = g.cmty_fast_greedy(return_dict=True)
        assert time.perf_counter() - start < 5.0  # ~0.1 s in practice
        assert g.modularity(m) > 0.45

    def test_uses_weights_and_resolution_recorded(self):
        g = ec.Graph(weighted=True).add_nodes(list("ABCD"))
        for u, v in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]:
            g.add_edge(u, v, weight=10.0 if (u, v) in {("B", "C"), ("D", "A")} else 1.0)
        m = {g.get_label_from_id(n): c for n, c in g.cmty_fast_greedy(return_dict=True).items()}
        assert m["B"] == m["C"] and m["D"] == m["A"] and m["A"] != m["B"]
        g.cmty_fast_greedy(resolution=1.2)
        assert ("cmty_fast_greedy", {"resolution": 1.2}, "cmty_fast_greedy") in g._computations


# --- step hooks ------------------------------------------------------------------------


class TestStepHooks:
    def test_bidirectional_dijkstra_highlights_path(self):
        g = _triangle_detour()
        g.bidirectional_dijkstra("A", "C")
        step = g.steps[-1]
        assert step.kind == KIND_QUERY
        assert step.delta.nodes_selected == ["A", "B", "C"]
        assert step.delta.edges_selected == [("A", "B"), ("B", "C")]
        assert "length 2" in step.narration

    def test_all_topological_sorts_step(self):
        g = ec.Graph(directed=True).add_nodes(list("ABC")).add_edges([("A", "C"), ("B", "C")])
        g.all_topological_sorts()
        step = g.steps[-1]
        assert step.narration == "2 topological orderings; first: A → B → C."
        assert step.delta.node_groups == {"A": 0, "B": 1, "C": 2}

    def test_partition_quality_step(self):
        g = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("C", "D")])
        g.partition_quality([["A", "B"], ["C", "D"]])
        step = g.steps[-1]
        assert step.delta.node_groups == {"A": 0, "B": 0, "C": 1, "D": 1}
        assert step.narration == "2 communities: coverage 1.000, performance 1.000."

    def test_weighted_closeness_step_code(self):
        g = _triangle_detour().closeness_centrality(weighted=False)
        assert g.steps[-1].code == (
            "G.closeness_centrality(weighted=False, attr_name='closeness_centrality')"
        )
