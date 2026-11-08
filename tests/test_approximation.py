"""Tests for approximation and optimization methods: traveling_salesman, metric_closure,
tree_decomposition, densest_subgraph and maximum_cut.

Validated against NetworkX (closure distances, greedy tours, treewidth heuristics, Greedy++)
and by brute force on small graphs (optimal tours, densest subgraphs, maximum cuts).
"""

import itertools
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_QUERY, KIND_TRANSFORM
from networkx.algorithms import approximation as nx_approx


def _build(G, weighted=True):
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in sorted(G)])
    for u, v, d in G.edges(data=True):
        g.add_edge(str(u), str(v), weight=d.get("weight", 1.0))
    return g


def _lab(g, nid):
    return int(g.get_label_from_id(nid))


def _connected_weighted(count=80, directed_every=4):
    for seed in range(count):
        rnd = random.Random(seed)
        directed = seed % directed_every == directed_every - 1
        G = nx.gnp_random_graph(rnd.randint(2, 7), 0.6, seed=seed, directed=directed)
        connected = nx.is_strongly_connected(G) if directed else nx.is_connected(G)
        if not connected:
            continue
        for u, v in G.edges:
            G[u][v]["weight"] = rnd.uniform(1, 10)
        yield G


def _five_sites():
    g = ec.Graph(weighted=True).add_nodes(["A", "B", "C", "D", "E"])
    g.add_edges(
        [
            ("A", "B", 2.0),
            ("B", "C", 2.0),
            ("C", "D", 3.0),
            ("D", "E", 2.0),
            ("E", "A", 3.0),
            ("A", "C", 4.0),
            ("B", "E", 3.0),
        ]
    )
    return g


class TestTravelingSalesman:
    def test_tours_are_valid_and_bounded(self):
        for G in _connected_weighted():
            g = _build(G)
            dist = dict(nx.all_pairs_dijkstra_path_length(G))
            n = len(G)
            opt = min(
                sum(dist[t[i]][t[i + 1]] for i in range(n))
                for p in itertools.permutations(range(1, n))
                for t in [(0, *p, 0)]
            )
            methods = ["greedy", "simulated_annealing", "threshold_accepting"]
            if not G.is_directed():
                methods.append("christofides")
            for method in methods:
                walk = [_lab(g, x) for x in g.traveling_salesman(method=method, seed=3)]
                assert walk[0] == walk[-1] and set(walk) == set(G)
                assert all(G.has_edge(a, b) for a, b in zip(walk, walk[1:]))
                cost = sum(G[a][b]["weight"] for a, b in zip(walk, walk[1:]))
                assert cost >= opt - 1e-9
                if method == "christofides":
                    assert cost <= 1.5 * opt + 1e-9

    def test_greedy_matches_networkx(self):
        for G in _connected_weighted():
            g = _build(G)
            walk = [_lab(g, x) for x in g.traveling_salesman(method="greedy")]
            ref = nx_approx.traveling_salesman_problem(
                G, method=lambda GG, weight: nx_approx.greedy_tsp(GG, weight=weight, source=0)
            )
            assert sum(G[a][b]["weight"] for a, b in zip(walk, walk[1:])) == pytest.approx(
                sum(G[a][b]["weight"] for a, b in zip(ref, ref[1:]))
            )

    def test_local_search_never_worse_than_greedy_and_seeded(self):
        g = ec.Graph(weighted=True)
        rnd = random.Random(4)
        pts = {str(i): (rnd.random(), rnd.random()) for i in range(12)}
        g.add_nodes(list(pts))
        for (a, pa), (b, pb) in itertools.combinations(pts.items(), 2):
            g.add_edge(a, b, weight=((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5)

        def cost(walk):
            return sum(g.get_edge(g.find_edge_ids(a, b)[0]).weight for a, b in zip(walk, walk[1:]))

        greedy = cost(g.traveling_salesman(method="greedy"))
        for method in ("simulated_annealing", "threshold_accepting"):
            for move in ("1-1", "1-0"):
                tour = g.traveling_salesman(method=method, seed=5, move=move, temperature=0.05)
                assert cost(tour) <= greedy + 1e-12
                assert tour == g.traveling_salesman(
                    method=method, seed=5, move=move, temperature=0.05
                )

    def test_subset_path_and_small_cases(self):
        path = ec.Graph(weighted=True).add_nodes(["A", "B", "C", "D"])
        path.add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("C", "D", 1.0)])
        labels = [path.get_label_from_id(n) for n in path.traveling_salesman(nodes=["A", "D"])]
        assert labels == ["A", "B", "C", "D", "C", "B", "A"]
        route = [
            path.get_label_from_id(n)
            for n in path.traveling_salesman(nodes=["A", "D"], cycle=False)
        ]
        assert route in (["A", "B", "C", "D"], ["D", "C", "B", "A"])
        assert path.traveling_salesman(nodes=["B"]) == [path.get_id_from_label("B")] * 2
        g = _five_sites()
        assert [g.get_label_from_id(n) for n in g.traveling_salesman()] == list("ABCDEA")

    def test_errors(self):
        split = ec.Graph().add_nodes(["A", "B", "C"]).add_edge("A", "B")
        with pytest.raises(ValueError, match="not connected"):
            split.traveling_salesman()
        d = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edges([("A", "B"), ("B", "A")])
        with pytest.raises(ValueError, match="undirected"):
            d.traveling_salesman(method="christofides")
        assert len(d.traveling_salesman(seed=1)) == 3
        with pytest.raises(ValueError, match="unknown method"):
            _five_sites().traveling_salesman(method="ant_colony")
        with pytest.raises(ValueError, match="move"):
            _five_sites().traveling_salesman(method="greedy", move="2-opt")


class TestMetricClosure:
    def test_against_networkx(self):
        for G in _connected_weighted(60):
            g = _build(G)
            closure = g.metric_closure()
            n = len(G)
            assert closure.count_edges() == (n * (n - 1) if G.is_directed() else n * (n - 1) // 2)
            dist = dict(nx.all_pairs_dijkstra_path_length(G))
            for _, s, t in closure.get_edges_as_tuples():
                e = closure.find_edge_ids(s, t)[0]
                assert closure.get_edge(e).weight == pytest.approx(dist[_lab(g, s)][_lab(g, t)])
            assert sorted(closure.get_node_ids(limit=None)) == sorted(g.get_node_ids(limit=None))

    def test_hops_and_errors(self):
        path = ec.Graph().add_path_graph(4)
        assert max(path.metric_closure().get_edge_attrs_by_id("weight").values()) == 3
        with pytest.raises(ValueError, match="connected"):
            ec.Graph().add_nodes(["A", "B"]).metric_closure()


class TestTreeDecomposition:
    @staticmethod
    def _valid(G, bags, tree_edges):
        T = nx.Graph()
        T.add_nodes_from(range(len(bags)))
        T.add_edges_from(tree_edges)
        if not nx.is_tree(T):
            return False
        if any(not any({u, v} <= b for b in bags) for u, v in G.edges if u != v):
            return False
        return all(
            nx.is_connected(T.subgraph([i for i, b in enumerate(bags) if v in b])) for v in G
        )

    def test_against_networkx(self):
        for seed in range(100):
            rnd = random.Random(seed)
            G = nx.gnp_random_graph(rnd.randint(1, 13), rnd.choice([0.15, 0.3, 0.5]), seed=seed)
            g = _build(G, weighted=False)
            for heuristic, ref in (
                ("min_fill_in", nx_approx.treewidth_min_fill_in),
                ("min_degree", nx_approx.treewidth_min_degree),
            ):
                width, tree = g.tree_decomposition(heuristic)
                assert width == ref(G)[0]
                ids = sorted(tree.get_node_ids(limit=None))
                bag_attr = tree.get_node_attrs("bag")
                bags = [{_lab(g, x) for x in bag_attr[i]} for i in ids]
                pos = {i: k for k, i in enumerate(ids)}
                edges = [(pos[s], pos[t]) for _, s, t in tree.get_edges_as_tuples()]
                assert self._valid(G, bags, edges)
                assert width == max(len(b) for b in bags) - 1

    def test_examples_and_errors(self):
        assert ec.Graph().add_full_graph(5).tree_decomposition()[0] == 4
        assert ec.Graph().add_star_graph(6).tree_decomposition()[0] == 1
        width, tree = ec.Graph().tree_decomposition()
        assert width == -1 and tree.count_nodes() == 1
        with pytest.raises(ValueError, match="heuristic"):
            ec.Graph().tree_decomposition("min_width")


class TestDensestSubgraph:
    @pytest.mark.skipif(not hasattr(nx_approx, "densest_subgraph"), reason="NetworkX 3.5+")
    def test_exact_against_brute_force_and_greedy_plus_plus(self):
        for seed in range(100):
            rnd = random.Random(seed)
            n = rnd.randint(1, 10)
            G = nx.gnp_random_graph(n, rnd.choice([0.2, 0.4, 0.7]), seed=seed)
            g = _build(G, weighted=False)
            best = max(
                (
                    G.subgraph(c).number_of_edges() / len(c)
                    for k in range(1, n + 1)
                    for c in itertools.combinations(G, k)
                ),
                default=0.0,
            )
            density, nodes = g.densest_subgraph()
            assert density == pytest.approx(best)
            if nodes:
                sub = G.subgraph([_lab(g, x) for x in nodes])
                assert sub.number_of_edges() / len(nodes) == pytest.approx(density)
            for iterations in (1, 4):
                approx, _ = g.densest_subgraph("greedy++", iterations)
                ref, _ = nx_approx.densest_subgraph(G, iterations, method="greedy++")
                assert approx == pytest.approx(ref)
                assert best / 2 - 1e-12 <= approx <= best + 1e-12

    def test_ignores_direction_loops_and_parallel_edges(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B"), ("B", "A"), ("B", "C"), ("C", "A"), ("A", "A")])
        assert g.densest_subgraph() == (1.0, sorted(g.get_node_ids(limit=None)))
        assert ec.Graph().add_nodes(["A"]).densest_subgraph() == (0.0, [])
        with pytest.raises(ValueError, match="iterations"):
            g.densest_subgraph("greedy++", 0)


class TestMaxCut:
    def test_local_optimum_and_bounds(self):
        for seed in range(100):
            rnd = random.Random(seed)
            n = rnd.randint(1, 10)
            G = nx.gnp_random_graph(n, 0.4, seed=seed)
            for u, v in G.edges:
                G[u][v]["weight"] = rnd.randint(1, 5)
            g = _build(G)
            best = max(
                nx.cut_size(G, set(c), weight="weight")
                for k in range(n + 1)
                for c in itertools.combinations(G, k)
            )
            value, (S, T) = g.maximum_cut(seed=seed)
            side = {_lab(g, x) for x in S}
            assert set(S).isdisjoint(T) and len(S) + len(T) == n
            assert value == pytest.approx(nx.cut_size(G, side, weight="weight"))
            assert G.size(weight="weight") / 2 - 1e-9 <= value <= best + 1e-9
            assert all(nx.cut_size(G, side ^ {v}, weight="weight") <= value + 1e-9 for v in G)

    def test_randomized_initial_and_errors(self):
        ring = ec.Graph().add_cycle_graph(6)
        assert ring.maximum_cut(seed=1)[0] == 6
        value, (S, T) = ring.maximum_cut(method="randomized", seed=2, p=1.0)
        assert value == 0 and not T
        start = ring.get_node_ids(limit=None)[:1]
        assert ring.maximum_cut(initial=start, seed=3)[0] == 6
        with pytest.raises(ValueError, match="method"):
            ring.maximum_cut(method="goemans_williamson")
        with pytest.raises(ValueError, match="between 0 and 1"):
            ring.maximum_cut(method="randomized", p=1.5)
        inf = ec.Graph(weighted=True).add_nodes(list("ABCD"))
        inf.add_edges([("A", "B", float("inf")), ("B", "C", 1.0), ("C", "D", float("inf"))])
        with pytest.raises(ValueError, match="finite"):  # the local search used to hang
            inf.maximum_cut(seed=1)


class TestApproximationStepHooks:
    def test_steps(self):
        g = _five_sites()
        g.traveling_salesman()
        step = g.steps[-1]
        assert step.kind == KIND_QUERY
        assert step.narration == "Tour of cost 12 through 5 nodes (christofides)."
        assert len(step.delta.edges_selected) == 5
        ring = ec.Graph().add_nodes(list("ABCDEF"))
        ring.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "A")])
        ring.maximum_cut(seed=1)
        step = ring.steps[-1]
        assert (
            len(set(step.delta.node_groups.values())) == 2 and len(step.delta.edges_selected) == 6
        )
        width, tree = ring.tree_decomposition()
        assert ring.steps[-1].kind == KIND_TRANSFORM
        assert (
            ring.steps[-1].narration
            == f"Tree decomposition of width {width} with {tree.count_nodes()} bags."
        )
        closure = ring.metric_closure()
        assert ring.steps[-1].kind == KIND_TRANSFORM and len(closure.steps) == 1
        ring.densest_subgraph()
        assert ring.steps[-1].narration == "Densest subgraph: 6 nodes, density 1."
