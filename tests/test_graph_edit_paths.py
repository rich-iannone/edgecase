"""Tests for the exact graph edit distance: graph_edit_distance(exact=True),
optimal_edit_paths(), optimize_graph_edit_distance() and optimize_edit_paths()."""

import itertools
import random
from collections import Counter

import pytest

import edgecase as ec

nx = pytest.importorskip("networkx")


def make_pair(n, edges, directed=False, multigraph=False, node_attrs=None, edge_attrs=None):
    """The same graph as an EdgeCase graph (labels "0".."n-1") and a NetworkX graph."""
    g = ec.Graph(directed=directed)
    labels = [str(i) for i in range(n)]
    attrs = node_attrs or {}
    g.add_nodes([(label, attrs.get(i, {})) for i, label in enumerate(labels)])
    if multigraph:
        h = nx.MultiDiGraph() if directed else nx.MultiGraph()
    else:
        h = nx.DiGraph() if directed else nx.Graph()
    h.add_nodes_from((label, dict(attrs.get(i, {}))) for i, label in enumerate(labels))
    for k, (u, v) in enumerate(edges):
        data = (edge_attrs or {}).get(k, {})
        g.add_edge(str(u), str(v), **data)
        h.add_edge(str(u), str(v), **data)
    return g, h


def random_edges(rng, n, p, directed=False, loops=False):
    pairs = itertools.permutations(range(n), 2) if directed else itertools.combinations(range(n), 2)
    edges = [e for e in pairs if rng.random() < p]
    if loops:
        edges += [(i, i) for i in range(n) if rng.random() < 0.3]
    return edges


def label_path(g, h, node_path):
    """A node path of IDs, as label pairs."""
    return tuple(
        (
            None if a is None else g.get_label_from_id(a),
            None if b is None else h.get_label_from_id(b),
        )
        for a, b in node_path
    )


def brute_force_ged(g, h):
    """Exact unit-cost GED by enumerating every partial injection of g's nodes into h's."""
    ids1, ids2 = sorted(g.get_node_ids(limit=None)), sorted(h.get_node_ids(limit=None))
    directed = g.is_directed()

    def edge_counts(graph):
        counts = Counter()
        for _, s, t, _ in graph._graph._weighted_edges():
            counts[(s, t) if directed else frozenset((s, t))] += 1
        return counts

    e1, e2 = edge_counts(g), edge_counts(h)
    total_edges = sum(e1.values()) + sum(e2.values())
    best = None
    targets = ids2 + [None] * len(ids1)
    for image in set(itertools.permutations(targets, len(ids1))):
        phi = dict(zip(ids1, image))
        mapped = sum(v is not None for v in image)
        subst = 0
        for key, count in e1.items():
            ends = tuple(key) if not directed else key
            if len(ends) == 1:
                ends = (ends[0], ends[0])
            a, b = phi[ends[0]], phi[ends[1]]
            if a is None or b is None:
                continue
            other = (a, b) if directed else frozenset((a, b))
            subst += min(count, e2.get(other, 0))
        cost = (len(ids1) - mapped) + (len(ids2) - mapped) + total_edges - 2 * subst
        best = cost if best is None else min(best, cost)
    return float(best)


def check_edit_path(g, h, node_path, edge_path):
    """Every node and edge appears exactly once, and substituted edges join mapped nodes."""
    assert sorted(a for a, _ in node_path if a is not None) == sorted(g.get_node_ids(limit=None))
    assert sorted(b for _, b in node_path if b is not None) == sorted(h.get_node_ids(limit=None))
    assert sorted(e for e, _ in edge_path if e is not None) == sorted(g.get_edge_ids(limit=None))
    assert sorted(f for _, f in edge_path if f is not None) == sorted(h.get_edge_ids(limit=None))
    phi = dict((a, b) for a, b in node_path if a is not None)
    for e, f in edge_path:
        if e is None or f is None:
            continue
        s1, t1 = g.get_node_ids_by_edge_id(e)
        s2, t2 = h.get_node_ids_by_edge_id(f)
        image = (phi[s1], phi[t1])
        if g.is_directed():
            assert image == (s2, t2)
        else:
            assert set(image) == {s2, t2}


# =============================================================================
# Agreement with NetworkX
# =============================================================================


class TestAgainstNetworkX:
    @pytest.mark.parametrize("seed", range(12))
    def test_random_undirected(self, seed):
        rng = random.Random(seed)
        n1, n2 = rng.randint(1, 6), rng.randint(1, 6)
        g, gx = make_pair(n1, random_edges(rng, n1, 0.5))
        h, hx = make_pair(n2, random_edges(rng, n2, 0.5))
        expected = nx.graph_edit_distance(gx, hx)
        assert g.graph_edit_distance(h, exact=True)["distance"] == expected

    @pytest.mark.parametrize("seed", range(8))
    def test_random_directed(self, seed):
        rng = random.Random(100 + seed)
        n1, n2 = rng.randint(1, 5), rng.randint(1, 5)
        g, gx = make_pair(n1, random_edges(rng, n1, 0.4, directed=True), directed=True)
        h, hx = make_pair(n2, random_edges(rng, n2, 0.4, directed=True), directed=True)
        expected = nx.graph_edit_distance(gx, hx)
        assert g.graph_edit_distance(h, exact=True)["distance"] == expected

    def test_multigraph(self):
        g, gx = make_pair(3, [(0, 1), (0, 1), (1, 2)], multigraph=True)
        h, hx = make_pair(3, [(0, 1), (1, 2), (1, 2), (1, 2)], multigraph=True)
        expected = nx.graph_edit_distance(gx, hx)
        assert g.graph_edit_distance(h, exact=True)["distance"] == expected == 1.0

    def test_networkx_docstring_examples(self):
        cycle, wheel = ec.Graph().add_cycle_graph(6), ec.Graph().add_wheel_graph(7)
        assert cycle.graph_edit_distance(wheel, exact=True)["distance"] == 7.0
        assert list(cycle.optimize_graph_edit_distance(wheel))[-1] == 7.0
        c4, w5 = ec.Graph().add_cycle_graph(4), ec.Graph().add_wheel_graph(5)
        paths, cost = c4.optimal_edit_paths(w5)
        assert (len(paths), cost) == (40, 5.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_optimal_edit_paths_same_node_paths(self, seed):
        rng = random.Random(200 + seed)
        n1, n2 = rng.randint(1, 4), rng.randint(1, 4)
        g, gx = make_pair(n1, random_edges(rng, n1, 0.5))
        h, hx = make_pair(n2, random_edges(rng, n2, 0.5))
        paths, cost = g.optimal_edit_paths(h)
        nx_paths, nx_cost = nx.optimal_edit_paths(gx, hx)
        assert cost == nx_cost
        ours = {frozenset(label_path(g, h, node_path)) for node_path, _ in paths}
        theirs = {frozenset(node_path) for node_path, _ in nx_paths}
        assert ours == theirs
        assert len(paths) == len(nx_paths)
        for node_path, edge_path in paths:
            check_edit_path(g, h, node_path, edge_path)

    def test_node_match(self):
        attrs1 = {0: {"el": "C"}, 1: {"el": "O"}, 2: {"el": "C"}}
        attrs2 = {0: {"el": "C"}, 1: {"el": "C"}, 2: {"el": "N"}}
        g, gx = make_pair(3, [(0, 1), (1, 2)], node_attrs=attrs1)
        h, hx = make_pair(3, [(0, 1), (1, 2)], node_attrs=attrs2)

        def match(a, b):
            return a["el"] == b["el"]

        expected = nx.graph_edit_distance(gx, hx, node_match=match)
        result = g.graph_edit_distance(h, exact=True, node_match=match)
        assert result["distance"] == expected
        paths, cost = g.optimal_edit_paths(h, node_match=match)
        nx_paths, nx_cost = nx.optimal_edit_paths(gx, hx, node_match=match)
        assert cost == nx_cost
        assert {frozenset(label_path(g, h, p)) for p, _ in paths} == {
            frozenset(p) for p, _ in nx_paths
        }

    def test_edge_costs_from_weights(self):
        w1 = {0: {"weight": 1.0}, 1: {"weight": 5.0}}
        w2 = {0: {"weight": 2.0}, 1: {"weight": 4.0}}
        g, gx = make_pair(3, [(0, 1), (1, 2)], edge_attrs=w1)
        h, hx = make_pair(3, [(0, 1), (1, 2)], edge_attrs=w2)
        options = dict(
            edge_subst_cost=lambda a, b: abs(a["weight"] - b["weight"]),
            edge_del_cost=lambda a: a["weight"],
            edge_ins_cost=lambda b: b["weight"],
        )
        expected = nx.graph_edit_distance(gx, hx, **options)
        assert g.graph_edit_distance(h, exact=True, **options)["distance"] == expected == 2.0

    def test_edge_match(self):
        e1 = {0: {"kind": "a"}, 1: {"kind": "b"}}
        e2 = {0: {"kind": "b"}, 1: {"kind": "b"}}
        g, gx = make_pair(3, [(0, 1), (1, 2)], edge_attrs=e1)
        h, hx = make_pair(3, [(0, 1), (1, 2)], edge_attrs=e2)

        def match(a, b):
            return a["kind"] == b["kind"]

        expected = nx.graph_edit_distance(gx, hx, edge_match=match)
        assert g.graph_edit_distance(h, exact=True, edge_match=match)["distance"] == expected

    def test_node_cost_functions(self):
        attrs1 = {0: {"w": 3}, 1: {"w": 1}}
        attrs2 = {0: {"w": 1}, 1: {"w": 2}, 2: {"w": 4}}
        g, gx = make_pair(2, [(0, 1)], node_attrs=attrs1)
        h, hx = make_pair(3, [(0, 1), (1, 2)], node_attrs=attrs2)
        options = dict(
            node_subst_cost=lambda a, b: abs(a["w"] - b["w"]),
            node_del_cost=lambda a: a["w"],
            node_ins_cost=lambda b: b["w"],
        )
        expected = nx.graph_edit_distance(gx, hx, **options)
        assert g.graph_edit_distance(h, exact=True, **options)["distance"] == expected

    def test_roots(self):
        star = ec.Graph().add_star_graph(6)
        center = max(star.get_node_ids(), key=star.get_degree)
        leaf = min(star.get_node_ids(), key=star.get_degree)
        sx = nx.star_graph(5)
        assert star.graph_edit_distance(star, exact=True, roots=(center, center))["distance"] == (
            nx.graph_edit_distance(sx, sx, roots=(0, 0))
        )
        assert star.graph_edit_distance(star, exact=True, roots=(leaf, center))["distance"] == (
            nx.graph_edit_distance(sx, sx, roots=(1, 0))
        )
        path = next(star.optimize_edit_paths(star, roots=(leaf, center)))[0]
        assert path[0] == (leaf, center)

    def test_roots_by_label(self):
        g = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "B")])
        result = g.graph_edit_distance(g, exact=True, roots=("A", "B"))
        assert result["distance"] == 0.0
        assert result["node_substitutions"] == 2.0


# =============================================================================
# Correctness beyond NetworkX
# =============================================================================


class TestBruteForce:
    @pytest.mark.parametrize("seed", range(15))
    def test_undirected_with_loops(self, seed):
        rng = random.Random(300 + seed)
        n1, n2 = rng.randint(0, 4), rng.randint(0, 4)
        g, _ = make_pair(n1, random_edges(rng, n1, 0.5, loops=True))
        h, _ = make_pair(n2, random_edges(rng, n2, 0.5, loops=True))
        assert g.graph_edit_distance(h, exact=True)["distance"] == brute_force_ged(g, h)

    @pytest.mark.parametrize("seed", range(10))
    def test_directed_with_loops(self, seed):
        rng = random.Random(400 + seed)
        n1, n2 = rng.randint(0, 4), rng.randint(0, 4)
        g, _ = make_pair(n1, random_edges(rng, n1, 0.4, True, loops=True), directed=True)
        h, _ = make_pair(n2, random_edges(rng, n2, 0.4, True, loops=True), directed=True)
        assert g.graph_edit_distance(h, exact=True)["distance"] == brute_force_ged(g, h)

    def test_loop_cannot_stand_in_for_edge(self):
        # NetworkX reports 1.0 here by substituting the loop for the edge 1--2.
        g, _ = make_pair(2, [(0, 1), (0, 0)])
        h, _ = make_pair(3, [(0, 1), (1, 2)])
        assert g.graph_edit_distance(h, exact=True)["distance"] == 3.0

    def test_single_node_loops(self):
        g, _ = make_pair(1, [(0, 0)])
        h, _ = make_pair(1, [])
        assert g.graph_edit_distance(h, exact=True)["distance"] == 1.0
        assert g.graph_edit_distance(g, exact=True)["distance"] == 0.0

    @pytest.mark.parametrize("seed", range(6))
    def test_multigraph_brute_force(self, seed):
        rng = random.Random(500 + seed)
        n1, n2 = rng.randint(1, 3), rng.randint(1, 3)
        e1 = [(rng.randrange(n1), rng.randrange(n1)) for _ in range(rng.randint(0, 4))]
        e2 = [(rng.randrange(n2), rng.randrange(n2)) for _ in range(rng.randint(0, 4))]
        g, _ = make_pair(n1, e1, multigraph=True)
        h, _ = make_pair(n2, e2, multigraph=True)
        assert g.graph_edit_distance(h, exact=True)["distance"] == brute_force_ged(g, h)

    @pytest.mark.parametrize("seed", range(6))
    def test_paths_are_valid(self, seed):
        rng = random.Random(600 + seed)
        n1, n2 = rng.randint(1, 4), rng.randint(1, 4)
        g, _ = make_pair(n1, random_edges(rng, n1, 0.5, True, loops=True), directed=True)
        h, _ = make_pair(n2, random_edges(rng, n2, 0.5, True, loops=True), directed=True)
        paths, cost = g.optimal_edit_paths(h)
        assert cost == brute_force_ged(g, h)
        for node_path, edge_path in paths:
            check_edit_path(g, h, node_path, edge_path)


# =============================================================================
# Results and options
# =============================================================================


class TestResults:
    def test_counts(self):
        cycle, wheel = ec.Graph().add_cycle_graph(6), ec.Graph().add_wheel_graph(7)
        result = cycle.graph_edit_distance(wheel, exact=True)
        assert result == {
            "distance": 7.0,
            "node_insertions": 1.0,
            "node_deletions": 0.0,
            "node_substitutions": 6.0,
            "edge_insertions": 6.0,
            "edge_deletions": 0.0,
            "edge_substitutions": 6.0,
        }
        assert all(isinstance(v, float) for v in result.values())

    def test_reverse_swaps_counts(self):
        cycle, wheel = ec.Graph().add_cycle_graph(6), ec.Graph().add_wheel_graph(7)
        result = wheel.graph_edit_distance(cycle, exact=True)
        assert result["distance"] == 7.0
        assert result["node_deletions"] == 1.0
        assert result["edge_deletions"] == 6.0

    def test_node_and_edge_cost_defaults(self):
        g = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "B")])
        h = ec.Graph().add_nodes(["A"])
        result = g.graph_edit_distance(h, exact=True, node_cost=2.0, edge_cost=0.5)
        assert result["distance"] == 2.5
        assert g.graph_edit_distance(h, exact=True, node_del_cost=lambda a: 7)["distance"] == 8.0

    def test_empty_graphs(self):
        empty = ec.Graph()
        triangle = ec.Graph().add_cycle_graph(3)
        assert empty.graph_edit_distance(empty, exact=True)["distance"] == 0.0
        assert empty.graph_edit_distance(triangle, exact=True)["distance"] == 6.0
        assert triangle.graph_edit_distance(empty, exact=True)["distance"] == 6.0
        assert empty.optimal_edit_paths(empty) == ([([], [])], 0.0)

    def test_isomorphic_paths_are_isomorphisms(self):
        triangle = ec.Graph().add_cycle_graph(3)
        paths, cost = triangle.optimal_edit_paths(triangle)
        assert cost == 0.0
        assert len(paths) == 6
        mappings = [dict(node_path) for node_path, _ in paths]
        assert all(all(v is not None for v in m.values()) for m in mappings)

    def test_upper_bound(self):
        cycle, wheel = ec.Graph().add_cycle_graph(6), ec.Graph().add_wheel_graph(7)
        result = cycle.graph_edit_distance(wheel, exact=True, upper_bound=6)
        assert result == dict.fromkeys(result)
        assert result["distance"] is None
        assert cycle.graph_edit_distance(wheel, exact=True, upper_bound=7)["distance"] == 7.0
        assert cycle.optimal_edit_paths(wheel, upper_bound=6) == ([], None)
        assert list(cycle.optimize_graph_edit_distance(wheel, upper_bound=6)) == []

    def test_optimize_graph_edit_distance_decreases(self):
        rng = random.Random(7)
        g, _ = make_pair(6, random_edges(rng, 6, 0.5))
        h, _ = make_pair(6, random_edges(rng, 6, 0.5))
        costs = list(g.optimize_graph_edit_distance(h))
        assert costs
        assert all(a > b for a, b in zip(costs, costs[1:]))
        assert costs[-1] == g.graph_edit_distance(h, exact=True)["distance"]

    def test_optimize_edit_paths_non_strict(self):
        c4, w5 = ec.Graph().add_cycle_graph(4), ec.Graph().add_wheel_graph(5)
        found = list(c4.optimize_edit_paths(w5, strictly_decreasing=False))
        costs = [cost for _, _, cost in found]
        assert all(a >= b for a, b in zip(costs, costs[1:]))
        assert sum(cost == 5.0 for cost in costs) == 40
        strict = [cost for _, _, cost in c4.optimize_edit_paths(w5)]
        assert all(a > b for a, b in zip(strict, strict[1:]))
        assert strict[-1] == 5.0

    def test_timeout_returns_a_bound(self):
        rng = random.Random(3)
        g, _ = make_pair(9, random_edges(rng, 9, 0.5))
        h, _ = make_pair(9, random_edges(rng, 9, 0.5))
        result = g.graph_edit_distance(h, exact=True, timeout=0.05)
        assert result["distance"] is not None
        assert result["distance"] >= 0

    def test_label_mode_unchanged(self):
        g = ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("A", "C")])
        h = ec.Graph().add_nodes(["A", "B", "D"]).add_edges([("A", "B"), ("B", "D")])
        result = g.graph_edit_distance(h)
        assert set(result) == {
            "distance",
            "node_insertions",
            "node_deletions",
            "edge_insertions",
            "edge_deletions",
        }
        assert result["distance"] == 4.0
        assert g.graph_edit_distance(h, exact=True)["distance"] == 0.0

    def test_attribute_dicts_passed_to_callables(self):
        g = ec.Graph(weighted=True).add_nodes([("A", {"x": 1})])
        g.add_nodes([("B", {"x": 2})])
        g.add_edge("A", "B", weight=2.5, kind="k")
        seen_nodes, seen_edges = [], []
        g.graph_edit_distance(
            g,
            exact=True,
            node_match=lambda a, b: seen_nodes.append(a) or True,
            edge_match=lambda a, b: seen_edges.append(a) or True,
        )
        assert {"x": 1} in seen_nodes and {"x": 2} in seen_nodes
        assert seen_edges[0] == {"kind": "k", "weight": 2.5}


class TestValidation:
    def test_exact_options_require_exact(self):
        g = ec.Graph().add_cycle_graph(3)
        with pytest.raises(ValueError, match="node_match= requires exact=True"):
            g.graph_edit_distance(g, node_match=lambda a, b: True)
        with pytest.raises(ValueError, match="timeout= requires exact=True"):
            g.graph_edit_distance(g, timeout=1)

    def test_directedness_mismatch(self):
        g = ec.Graph().add_cycle_graph(3)
        d = ec.Graph(directed=True).add_cycle_graph(3)
        with pytest.raises(ValueError, match="directed"):
            g.graph_edit_distance(d, exact=True)
        with pytest.raises(ValueError, match="directed"):
            g.optimal_edit_paths(d)
        with pytest.raises(ValueError, match="directed"):
            g.optimize_graph_edit_distance(d)
        with pytest.raises(ValueError, match="directed"):
            g.optimize_edit_paths(d)

    def test_bad_costs(self):
        g = ec.Graph().add_cycle_graph(3)
        with pytest.raises(ValueError, match="non-negative"):
            g.graph_edit_distance(g, exact=True, node_del_cost=lambda a: -1)
        with pytest.raises(ValueError, match="must be a number"):
            g.optimal_edit_paths(g, edge_subst_cost=lambda a, b: "x")
        with pytest.raises(ValueError, match="non-negative"):
            g.graph_edit_distance(g, exact=True, edge_cost=float("nan"))

    def test_bad_timeout(self):
        g = ec.Graph().add_cycle_graph(3)
        with pytest.raises(ValueError, match="timeout"):
            g.graph_edit_distance(g, exact=True, timeout=0)

    def test_bad_root(self):
        g = ec.Graph().add_cycle_graph(3)
        with pytest.raises((ValueError, ec.NodeNotFoundError)):
            g.graph_edit_distance(g, exact=True, roots=(99, 1))

    def test_other_must_be_graph(self):
        g = ec.Graph().add_cycle_graph(3)
        with pytest.raises(TypeError):
            g.optimal_edit_paths("not a graph")
