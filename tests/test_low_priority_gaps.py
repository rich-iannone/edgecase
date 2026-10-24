"""Tests for Goldberg-Radzik shortest paths, Panther similarity, random paths, ISMAGS symmetry
handling (symmetric enumeration, automorphism cosets, largest common subgraphs) and Asadpour's
asymmetric TSP, with NetworkX parity where NetworkX gives the same answer."""

import itertools
import math
import random

import edgecase as ec
import networkx as nx
import numpy as np
import pytest
from edgecase.steps import KIND_METRIC, KIND_QUERY


def _labels(graph, ids):
    return [graph.get_label_from_id(n) for n in ids]


def _random_ec_nx(rng, n, p, directed=False):
    """The same random graph in EdgeCase (labels '0'..) and NetworkX (nodes 0..)."""
    G = nx.gnp_random_graph(n, p, seed=rng.randrange(10**6), directed=directed)
    g = ec.Graph(directed=directed).add_nodes([str(i) for i in range(n)])
    for u, v in G.edges:
        g.add_edge(str(u), str(v))
    return g, G


# =============================================================================
# goldberg_radzik
# =============================================================================


def _potential_graph(rng, n, m, directed=True):
    """Random graph with some negative weights but no negative cycles (reduced costs)."""
    pot = [rng.uniform(-10, 10) for _ in range(n)]
    g = ec.Graph(directed=directed, weighted=True).add_nodes([str(i) for i in range(n)])
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for _ in range(m):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v or G.has_edge(u, v):
            continue
        w = round(rng.uniform(0, 5) + pot[u] - pot[v], 3)
        g.add_edges([(str(u), str(v), w)])
        G.add_edge(u, v, weight=w)
    return g, G


class TestGoldbergRadzik:
    def test_docstring_example(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["A", "B", "C", "D"])
        g.add_edges([("A", "B", 4.0), ("A", "C", 2.0), ("C", "B", -3.0), ("B", "D", 1.0)])
        pred, dist = g.goldberg_radzik("A")
        named = {
            g.get_label_from_id(n): (
                None if pred[n] is None else g.get_label_from_id(pred[n]),
                dist[n],
            )
            for n in dist
        }
        assert named == {"A": (None, 0.0), "B": ("C", -1.0), "C": ("A", 2.0), "D": ("B", 0.0)}

    @pytest.mark.parametrize("seed", range(25))
    def test_networkx_parity(self, seed):
        rng = random.Random(seed)
        n = rng.randint(1, 12)
        g, G = _potential_graph(rng, n, rng.randint(0, 3 * n))
        source = rng.randrange(n)
        pred, dist = g.goldberg_radzik(str(source))
        nx_pred, nx_dist = nx.goldberg_radzik(G, source)
        ours = {int(g.get_label_from_id(k)): v for k, v in dist.items()}
        assert set(ours) == set(nx_dist)
        for node, d in nx_dist.items():
            assert ours[node] == pytest.approx(d, abs=1e-9)
        # the predecessors form a shortest-path tree (ties may pick other predecessors)
        for node, p in pred.items():
            if p is None:
                assert int(g.get_label_from_id(node)) == source
                continue
            u, v = int(g.get_label_from_id(p)), int(g.get_label_from_id(node))
            assert ours[u] + G[u][v]["weight"] == pytest.approx(ours[v], abs=1e-9)

    def test_unweighted_counts_hops(self):
        g = ec.Graph(directed=True).add_path_graph(4)
        _, dist = g.goldberg_radzik(1)
        assert dist == {1: 0, 2: 1, 3: 2, 4: 3}

    def test_negative_cycle_raises(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["a", "b", "c"])
        g.add_edges([("a", "b", 1.0), ("b", "c", -2.0), ("c", "b", 1.0)])
        with pytest.raises(ValueError, match="egative cycle"):
            g.goldberg_radzik("a")
        G = nx.DiGraph(
            [("a", "b", {"weight": 1}), ("b", "c", {"weight": -2}), ("c", "b", {"weight": 1})]
        )
        with pytest.raises(nx.NetworkXUnbounded):
            nx.goldberg_radzik(G, "a")

    def test_negative_self_loop_and_undirected_negative_edge(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["a"])
        g.add_edges([("a", "a", -1.0)])
        with pytest.raises(ValueError):
            g.goldberg_radzik("a")
        u = ec.Graph(weighted=True).add_nodes(["a", "b"])
        u.add_edges([("a", "b", -1.0)])
        with pytest.raises(ValueError):
            u.goldberg_radzik("a")

    def test_unreachable_nodes_left_out_and_steps(self):
        g = ec.Graph(directed=True).add_path_graph(3).add_nodes(["lonely"])
        pred, dist = g.goldberg_radzik(2)
        assert set(dist) == {2, 3} and pred == {2: None, 3: 2}
        assert g.steps[-1].kind == KIND_METRIC

    def test_missing_source(self):
        with pytest.raises(Exception):
            ec.Graph().add_path_graph(3).goldberg_radzik("nope")


# =============================================================================
# panther_similarity and generate_random_paths
# =============================================================================


TWO_TRIANGLES = [("a", "b"), ("b", "c"), ("c", "a"), ("c", "x"), ("x", "y"), ("y", "z"), ("z", "x")]


def _two_triangles():
    g = ec.Graph().add_nodes(["a", "b", "c", "x", "y", "z"])
    g.add_edges(TWO_TRIANGLES)
    return g


class TestGenerateRandomPaths:
    def test_shape_and_walk_validity(self):
        g = _two_triangles()
        paths = g.generate_random_paths(50, path_length=4, seed=3)
        assert len(paths) == 50
        pairs = {frozenset((s, t)) for _, s, t in g.get_edges_as_tuples()}
        for p in paths:
            assert len(p) == 5
            for a, b in zip(p, p[1:]):
                assert frozenset((a, b)) in pairs

    def test_seed_reproducible_and_source(self):
        g = ec.Graph().add_cycle_graph(6)
        assert g.generate_random_paths(5, seed=2) == g.generate_random_paths(5, seed=2)
        assert all(p[0] == 4 for p in g.generate_random_paths(10, source=4, seed=1))

    def test_directed_follows_arcs(self):
        g = ec.Graph(directed=True).add_cycle_graph(5)
        for p in g.generate_random_paths(10, path_length=6, seed=0):
            for a, b in zip(p, p[1:]):
                assert b == a % 5 + 1

    def test_weighted_steps(self):
        # from b, the heavy edge to c is taken almost always
        g = ec.Graph(weighted=True).add_nodes(["a", "b", "c"])
        g.add_edges([("a", "b", 0.001), ("b", "c", 1000.0)])
        b, c = g.get_node_ids_by_edge_id(g.find_edge_ids("b", "c")[0])
        steps = [p[1] for p in g.generate_random_paths(200, path_length=1, seed=5, source="b")]
        assert steps.count(c) > 190

    def test_random_starts_skip_nodes_without_out_edges(self):
        # As in NetworkX #8578, walks without a source start only where a step is possible
        g = ec.Graph().add_path_graph(2).add_nodes(["lonely"])
        paths = g.generate_random_paths(20, path_length=2, seed=42)
        assert len(paths) == 20 and all(len(p) == 3 for p in paths)
        assert all(p[0] in (1, 2) for p in paths)
        # In a directed graph a sink is not a start either, but a walk can still reach one
        d = ec.Graph(directed=True).add_path_graph(3)
        assert all(p[0] != 3 for p in d.generate_random_paths(30, path_length=1, seed=0))
        with pytest.raises(ValueError, match="no out-edges"):
            d.generate_random_paths(30, path_length=2, seed=0)
        with pytest.raises(ValueError, match="no random path can start"):
            ec.Graph().add_nodes(["a", "b"]).generate_random_paths(1, seed=0)
        assert ec.Graph().add_nodes(["a"]).generate_random_paths(0) == []

    def test_dead_end_and_bad_weight_raise(self):
        g = ec.Graph(directed=True).add_path_graph(2)
        with pytest.raises(ValueError):
            g.generate_random_paths(5, path_length=3, source=2, seed=0)
        h = ec.Graph(weighted=True).add_nodes(["a", "b"])
        h.add_edges([("a", "b", -1.0)])
        with pytest.raises(ValueError):
            h.generate_random_paths(5, seed=0)


class TestPantherSimilarity:
    def test_docstring_example(self):
        g = ec.Graph().add_nodes(["a", "b", "c", "x", "y", "z"])
        g.add_edges(
            [("a", "b"), ("b", "c"), ("c", "a"), ("c", "x"), ("x", "y"), ("y", "z"), ("z", "x")]
        )
        scores = g.panther_similarity("a", k=3, seed=1)
        named = {g.get_label_from_id(n): round(s, 3) for n, s in scores.items()}
        assert named == {"c": 0.577, "b": 0.462, "x": 0.346}

    def test_matches_sampled_paths(self):
        # the scores count the sampled paths holding both nodes
        g = _two_triangles()
        # R = ceil((c / eps^2)(log2 C(T + 1, 2) + 1 + ln(1 / delta))), as in NetworkX #8578
        size = math.ceil(0.5 * 7 * (math.log2(math.comb(6, 2)) + 1 + math.log(1 / 0.1)))
        for seed in range(5):
            counts = {}
            for p in g.generate_random_paths(size, path_length=5, seed=seed):
                if 1 in p:
                    for v in set(p) - {1}:
                        counts[v] = counts.get(v, 0) + 1
            scores = g.panther_similarity("a", k=5, seed=seed)
            for v in set(scores) | set(counts):
                assert scores.get(v, 0.0) == pytest.approx(counts.get(v, 0) / size)

    def test_ranking_and_bounds(self):
        g = _two_triangles()
        scores = g.panther_similarity("a", k=4, seed=0)
        assert len(scores) == 4 and 1 not in scores
        values = list(scores.values())
        assert values == sorted(values, reverse=True)
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_statistical_parity_with_networkx(self):
        g = _two_triangles()
        G = nx.Graph(TWO_TRIANGLES)
        ours, theirs = {}, {}
        for seed in range(150):
            for node, s in g.panther_similarity("a", k=5, seed=seed).items():
                ours.setdefault(g.get_label_from_id(node), []).append(s)
            random.seed(seed)
            np.random.seed(seed)
            for node, s in nx.panther_similarity(G, "a", k=6).items():
                theirs.setdefault(node, []).append(s)
        for node in "bcxyz":
            assert np.mean(ours.get(node, [0])) == pytest.approx(np.mean(theirs[node]), abs=0.03)

    def test_k_too_large_warns(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.warns(UserWarning, match="k is 10"):
            scores = g.panther_similarity(1, k=10, seed=0)
        assert len(scores) == 2

    def test_invalid_arguments(self):
        g = _two_triangles()
        with pytest.raises(ValueError):
            g.panther_similarity("a", k=0)
        with pytest.raises(ValueError):
            g.panther_similarity("a", path_length=1)
        with pytest.raises(ValueError):
            g.panther_similarity("a", c=float("nan"))
        with pytest.raises(ValueError):
            g.panther_similarity("a", delta=0.0)
        iso = ec.Graph().add_path_graph(2).add_nodes(["alone"])
        with pytest.raises(ValueError):
            iso.panther_similarity("alone")

    def test_records_step(self):
        g = _two_triangles()
        g.panther_similarity("a", k=2, seed=0)
        assert g.steps[-1].kind == KIND_METRIC


# =============================================================================
# ISMAGS: symmetric enumeration, cosets, largest common subgraphs
# =============================================================================


def _automorphisms(g):
    return g.subgraph_isomorphisms(g, induced=True)


PATTERNS = [
    lambda: ec.Graph().add_cycle_graph(3),
    lambda: ec.Graph().add_path_graph(3),
    lambda: ec.Graph().add_cycle_graph(4),
    lambda: ec.Graph().add_star_graph(3),
    lambda: ec.Graph().add_path_graph(4),
    lambda: ec.Graph().add_full_graph(4),
]


class TestSymmetricEnumeration:
    def test_docstring_example(self):
        k4 = ec.Graph().add_full_graph(4)
        triangle = ec.Graph().add_cycle_graph(3)
        assert len(k4.subgraph_isomorphisms(triangle)) == 24
        assert len(k4.subgraph_isomorphisms(triangle, symmetry=True)) == 4

    def test_petersen(self):
        p = ec.Graph().add_named_graph("petersen")
        assert len(p.subgraph_isomorphisms(p, induced=True)) == 120
        assert len(p.subgraph_isomorphisms(p, induced=True, symmetry=True)) == 1

    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize("induced", [True, False])
    def test_one_per_class(self, seed, induced):
        rng = random.Random(seed)
        g, _ = _random_ec_nx(rng, 8, 0.5)
        pattern = PATTERNS[seed % len(PATTERNS)]()
        full = g.subgraph_isomorphisms(pattern, induced=induced)
        sym = g.subgraph_isomorphisms(pattern, induced=induced, symmetry=True)
        n_aut = (
            len(_automorphisms(pattern)) if induced else len(pattern.subgraph_isomorphisms(pattern))
        )
        assert len(full) == len(sym) * n_aut

        # two mappings are symmetric copies exactly when they cover the same edges
        def image(m):
            edges = frozenset(frozenset((m[s], m[t])) for _, s, t in pattern.get_edges_as_tuples())
            return frozenset(m.values()), edges

        sym_images = [image(m) for m in sym]
        assert len(set(sym_images)) == len(sym_images)
        assert set(sym_images) == {image(m) for m in full}
        full_set = {tuple(sorted(m.items())) for m in full}
        assert all(tuple(sorted(m.items())) in full_set for m in sym)

    @pytest.mark.parametrize("seed", range(10))
    def test_networkx_ismags_counts(self, seed):
        rng = random.Random(50 + seed)
        g, G = _random_ec_nx(rng, 9, 0.45)
        pattern = PATTERNS[seed % len(PATTERNS)]()
        P = nx.Graph()
        P.add_nodes_from(pattern.get_node_ids())
        P.add_edges_from((s, t) for _, s, t in pattern.get_edges_as_tuples())
        ismags = nx.isomorphism.ISMAGS(G, P)
        for symmetry in (True, False):
            theirs = list(ismags.find_isomorphisms(symmetry=symmetry))
            ours = g.subgraph_isomorphisms(pattern, induced=True, symmetry=symmetry)
            assert len(ours) == len(theirs)
        ours = {
            frozenset((k, int(g.get_label_from_id(v))) for k, v in m.items())
            for m in g.subgraph_isomorphisms(pattern, induced=True)
        }
        theirs = {
            frozenset((v, k) for k, v in m.items())
            for m in ismags.find_isomorphisms(symmetry=False)
        }
        assert ours == theirs

    def test_directed_and_colored(self):
        g = ec.Graph(directed=True).add_cycle_graph(6)
        pattern = ec.Graph(directed=True).add_cycle_graph(3)
        assert g.subgraph_isomorphisms(pattern, symmetry=True) == []
        tri = ec.Graph(directed=True).add_cycle_graph(3)
        assert len(tri.subgraph_isomorphisms(tri, symmetry=True)) == 1
        assert len(tri.subgraph_isomorphisms(tri)) == 3
        colored = ec.Graph().add_cycle_graph(4)
        colored.set_node_attrs("c", {1: "r", 2: "b", 3: "r", 4: "b"})
        auts = colored.subgraph_isomorphisms(colored, node_attr="c")
        assert len(auts) == 4
        assert len(colored.subgraph_isomorphisms(colored, node_attr="c", symmetry=True)) == 1


class TestAutomorphismCosets:
    def test_docstring_example(self):
        assert ec.Graph().add_cycle_graph(4).automorphism_cosets() == {
            1: [1, 2, 3, 4],
            2: [2, 4],
            3: [3],
            4: [4],
        }

    @pytest.mark.parametrize(
        "make",
        PATTERNS
        + [
            lambda: ec.Graph().add_named_graph("petersen"),
            lambda: ec.Graph().add_hypercube_graph(3),
            lambda: ec.Graph(directed=True).add_cycle_graph(5),
            lambda: ec.Graph().add_path_graph(3).add_nodes(["iso1", "iso2"]),
        ],
    )
    def test_cosets_from_automorphisms(self, make):
        g = make()
        auts = _automorphisms(g)
        cosets = g.automorphism_cosets()
        ids = sorted(g.get_node_ids(limit=None))
        assert list(cosets) == ids
        assert math.prod(len(c) for c in cosets.values()) == len(auts)
        for v in ids:
            expected = sorted({m[v] for m in auts if all(m[u] == u for u in ids if u < v)})
            assert cosets[v][0] == v
            assert sorted(cosets[v]) == expected
            assert cosets[v][1:] == sorted(cosets[v][1:])

    def test_petersen_group_order(self):
        # NetworkX's analyze_symmetry misses coset members here (its cosets multiply to 80)
        p = ec.Graph().add_named_graph("petersen")
        assert math.prod(len(c) for c in p.automorphism_cosets().values()) == 120

    @pytest.mark.parametrize("seed", range(8))
    def test_random_graphs(self, seed):
        rng = random.Random(seed)
        g, _ = _random_ec_nx(rng, 7, 0.4)
        assert math.prod(len(c) for c in g.automorphism_cosets().values()) == len(_automorphisms(g))

    def test_attributes_and_steps(self):
        g = ec.Graph().add_path_graph(3)
        assert g.automorphism_cosets() == {1: [1, 3], 2: [2], 3: [3]}
        g.set_node_attrs("kind", {1: "x", 2: "y", 3: "z"})
        assert g.automorphism_cosets(node_attr="kind") == {1: [1], 2: [2], 3: [3]}
        assert g.steps[-1].kind == KIND_QUERY
        assert ec.Graph().automorphism_cosets() == {}


class TestLargestCommonSubgraph:
    def test_docstring_example(self):
        square = ec.Graph().add_nodes(["a", "b", "c", "d"])
        square.add_edges([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
        paw = ec.Graph().add_nodes(["x", "y", "z", "w"])
        paw.add_edges([("x", "y"), ("y", "z"), ("z", "x"), ("z", "w")])
        found = square.largest_common_subgraph(paw)
        assert len(found) == 8
        assert len(square.largest_common_subgraph(paw, symmetry=False)) == 16
        first = {paw.get_label_from_id(k): square.get_label_from_id(v) for k, v in found[0].items()}
        assert first == {"x": "b", "z": "a", "w": "d"}

    def test_networkx_docs_examples(self):
        # path of 4 nodes against a star with 3 leaves: the common part is a 3-node path
        path = ec.Graph().add_path_graph(4)
        star = ec.Graph().add_star_graph(3)
        found = path.largest_common_subgraph(star)
        assert {len(m) for m in found} == {3}
        G1, G2 = nx.path_graph(4), nx.star_graph(3)
        theirs = list(nx.isomorphism.ISMAGS(G1, G2).largest_common_subgraph())
        assert len(found) == len(theirs)

    @pytest.mark.parametrize("seed", range(15))
    def test_networkx_parity(self, seed):
        rng = random.Random(200 + seed)
        g, G = _random_ec_nx(rng, rng.randint(4, 7), 0.5)
        h, H = _random_ec_nx(rng, rng.randint(3, 5), 0.5)
        ismags = nx.isomorphism.ISMAGS(G, H)
        theirs = {
            frozenset((v, k) for k, v in m.items())
            for m in ismags.largest_common_subgraph(symmetry=False)
        }
        ours = {
            frozenset(
                (int(h.get_label_from_id(k)), int(g.get_label_from_id(v))) for k, v in m.items()
            )
            for m in g.largest_common_subgraph(h, symmetry=False)
        }
        assert ours == theirs
        sym = g.largest_common_subgraph(h)
        assert len(sym) == len(list(ismags.largest_common_subgraph(symmetry=True)))
        # every symmetric result is a genuine common induced subgraph of the largest size
        sizes = {len(m) for m in g.largest_common_subgraph(h, symmetry=False)}
        assert {len(m) for m in sym} <= sizes

    def test_contained_graph_matches_whole(self):
        g = ec.Graph().add_full_graph(5)
        k3 = ec.Graph().add_full_graph(3)
        found = g.largest_common_subgraph(k3)
        assert len(found) == 10 and all(len(m) == 3 for m in found)

    def test_edge_cases(self):
        g = ec.Graph().add_path_graph(3)
        assert g.largest_common_subgraph(ec.Graph()) == [{}]
        assert ec.Graph().largest_common_subgraph(g) == []
        with pytest.raises(ValueError):
            g.largest_common_subgraph(ec.Graph(directed=True).add_path_graph(2))

    def test_attributes(self):
        g = ec.Graph().add_path_graph(3)
        g.set_node_attrs("c", {1: "r", 2: "r", 3: "b"})
        h = ec.Graph().add_path_graph(2)
        h.set_node_attrs("c", {1: "b", 2: "b"})
        found = g.largest_common_subgraph(h, node_attr="c")
        assert {len(m) for m in found} == {1}
        assert all(list(m.values()) == [3] for m in found)


# =============================================================================
# Asadpour ATSP
# =============================================================================


def _complete_digraph(rng, n, metric=True):
    G = nx.complete_graph(n, create_using=nx.DiGraph)
    for u, v in G.edges:
        G[u][v]["weight"] = rng.randint(1, 20)
    if metric:
        dist = dict(nx.all_pairs_dijkstra_path_length(G))
        for u, v in G.edges:
            G[u][v]["weight"] = dist[u][v]
    g = ec.Graph(directed=True, weighted=True).add_nodes([str(i) for i in range(n)])
    g.add_edges([(str(u), str(v), float(d["weight"])) for u, v, d in G.edges(data=True)])
    return g, G


def _cost(G, tour):
    return sum(G[a][b]["weight"] for a, b in zip(tour, tour[1:]))


def _optimum(G):
    n = len(G)
    return min(_cost(G, (0, *p, 0)) for p in itertools.permutations(range(1, n)))


class TestAsadpour:
    def test_networkx_docs_example(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["0", "1", "2"])
        g.add_edges(
            [
                ("0", "1", 2.0),
                ("1", "2", 2.0),
                ("2", "0", 2.0),
                ("0", "2", 1.0),
                ("2", "1", 1.0),
                ("1", "0", 1.0),
            ]
        )
        assert _labels(g, g.traveling_salesman(method="asadpour")) == ["0", "2", "1", "0"]

    def test_docstring_example(self):
        oneway = ec.Graph(directed=True, weighted=True).add_nodes(["A", "B", "C"])
        oneway.add_edges(
            [
                ("A", "B", 2.0),
                ("B", "C", 2.0),
                ("C", "A", 2.0),
                ("A", "C", 1.0),
                ("C", "B", 1.0),
                ("B", "A", 1.0),
            ]
        )
        assert _labels(oneway, oneway.traveling_salesman(method="asadpour")) == ["A", "C", "B", "A"]

    @pytest.mark.parametrize("seed", range(30))
    def test_against_brute_force(self, seed):
        rng = random.Random(seed)
        n = rng.randint(3, 8)
        g, G = _complete_digraph(rng, n)
        walk = g.traveling_salesman(method="asadpour", seed=seed)
        tour = [int(x) for x in _labels(g, walk)]
        assert tour[0] == tour[-1] and sorted(tour[:-1]) == list(range(n))
        opt = _optimum(G)
        cost = _cost(G, tour)
        assert opt - 1e-9 <= cost <= 2 * opt

    @pytest.mark.parametrize("seed", range(4))
    def test_networkx_small_instances(self, seed):
        # NetworkX's ascent is slow, so only tiny instances; both are near-optimal here
        rng = random.Random(70 + seed)
        g, G = _complete_digraph(rng, 4)
        theirs = nx.approximation.asadpour_atsp(G.copy(), seed=seed)
        ours = [int(x) for x in _labels(g, g.traveling_salesman(method="asadpour", seed=seed))]
        assert _cost(G, ours) <= _cost(G, theirs) + 1e-9 or _cost(G, ours) <= 1.5 * _optimum(G)

    def test_seed_reproducible(self):
        rng = random.Random(9)
        g, _ = _complete_digraph(rng, 12)
        assert g.traveling_salesman(method="asadpour", seed=4) == g.traveling_salesman(
            method="asadpour", seed=4
        )

    def test_larger_instance_beats_lower_bound_check(self):
        rng = random.Random(11)
        g, G = _complete_digraph(rng, 20)
        tour = [int(x) for x in _labels(g, g.traveling_salesman(method="asadpour", seed=0))]
        assert sorted(tour[:-1]) == list(range(20))
        greedy = [int(x) for x in _labels(g, g.traveling_salesman(method="greedy"))]
        assert _cost(G, tour) <= 1.5 * _cost(G, greedy)

    def test_sparse_graph_uses_metric_closure(self):
        # a directed ring with one shortcut; the tour must follow the ring
        g = ec.Graph(directed=True, weighted=True).add_nodes(list("abcde"))
        g.add_edges(
            [
                ("a", "b", 1.0),
                ("b", "c", 1.0),
                ("c", "d", 1.0),
                ("d", "e", 1.0),
                ("e", "a", 1.0),
                ("a", "c", 5.0),
            ]
        )
        walk = _labels(g, g.traveling_salesman(method="asadpour", seed=1))
        assert walk == ["a", "b", "c", "d", "e", "a"]

    def test_stops_cycle_false_and_undirected(self):
        rng = random.Random(3)
        g, G = _complete_digraph(rng, 7)
        stops = ["1", "4", "6"]
        walk = g.traveling_salesman(nodes=stops, method="asadpour", seed=2)
        assert walk[0] == walk[-1] and set(_labels(g, walk)) >= set(stops)
        path = g.traveling_salesman(method="asadpour", cycle=False, seed=2)
        assert len(path) == 7 and len(set(path)) == 7
        u = ec.Graph(weighted=True).add_cycle_graph(6)
        walk = u.traveling_salesman(method="asadpour", seed=0)
        assert sorted(walk[:-1]) == list(range(1, 7))

    def test_two_nodes_and_unreachable(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["a", "b"])
        g.add_edges([("a", "b", 1.0), ("b", "a", 3.0)])
        assert _labels(g, g.traveling_salesman(method="asadpour")) == ["a", "b", "a"]
        h = ec.Graph(directed=True).add_path_graph(3)
        with pytest.raises(ValueError, match="reachable"):
            h.traveling_salesman(method="asadpour")
