"""Tests for `canonical_labeling()` (NetworkX PR #8953)."""

import itertools
import random

import edgecase as ec
import networkx as nx
import pytest

from .canonical_graphs import C240, PR_GRAPHS

# =============================================================================
# Helpers
# =============================================================================


def _graph(edges, nodes=(), directed=False, node_attrs=None, edge_attrs=None, order=None):
    """An EdgeCase graph on string labels; `order` permutes the node insertion order."""
    labels = list(dict.fromkeys([str(v) for v in nodes] + [str(x) for e in edges for x in e]))
    if order is not None:
        labels = [labels[i] for i in order]
    g = ec.Graph(directed=directed)
    g.add_nodes([(v, dict((node_attrs or {}).get(v) or {})) for v in labels])
    for i, (u, v) in enumerate(edges):
        g.add_edge(str(u), str(v), **dict((edge_attrs or {}).get(i, {})))
    return g


def _from_strings(lines):
    return _graph([tuple(line.split()) for line in lines])


def _relabeled(g, rng):
    """An isomorphic copy of `g`: new labels, nodes and edges added in a shuffled order."""
    ids = g.get_node_ids(limit=None)
    names = [f"n{i}" for i in range(len(ids))]
    rng.shuffle(names)
    new = dict(zip(ids, names))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    h = ec.Graph(directed=g.is_directed())
    h.add_nodes([(new[n], dict(g.get_node(n).attributes or {})) for n in shuffled])
    edges = [g.get_edge(e) for e in g.get_edge_ids(limit=None)]
    rng.shuffle(edges)
    for e in edges:
        h.add_edge(new[e.source], new[e.target], **dict(e.attributes or {}))
    return h


def _form(g, node_attr=None, edge_attr=None):
    """The graph rewritten with canonical numbers (with the attributes taken into account)."""
    number = g.canonical_labeling(node_attr=node_attr, edge_attr=edge_attr)
    assert sorted(number.values()) == list(range(g.count_nodes()))
    names = [node_attr] if isinstance(node_attr, str) else list(node_attr or [])
    columns = [g.get_node_attrs(name) for name in names]
    evals = g.get_edge_attrs_by_id(edge_attr) if edge_attr else {}
    nodes = sorted((number[n], repr([c.get(n) for c in columns])) for n in number)
    edges = []
    for e in g.get_edge_ids(limit=None):
        x = g.get_edge(e)
        a, b = number[x.source], number[x.target]
        if not g.is_directed() and a > b:
            a, b = b, a
        edges.append((a, b, repr(evals.get(e))))
    return nodes, sorted(edges)


def _random_graph(rng, n_max=7):
    n = rng.randint(1, n_max)
    directed = rng.random() < 0.4
    m = rng.randint(0, 12)
    edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(m)]
    node_attrs = (
        {str(v): {"c": rng.choice("xy")} for v in range(n)} if rng.random() < 0.3 else None
    )
    edge_attrs = {i: {"w": rng.randint(0, 1)} for i in range(m)} if rng.random() < 0.3 else None
    return _graph(edges, range(n), directed, node_attrs, edge_attrs)


def _attr_names(g):
    nodes = g.get_node_ids(limit=None)
    edges = g.get_edge_ids(limit=None)
    node_attr = "c" if nodes and "c" in (g.get_node(nodes[0]).attributes or {}) else None
    edge_attr = "w" if edges and "w" in (g.get_edge(edges[0]).attributes or {}) else None
    return node_attr, edge_attr


def _shrikhande():
    """Cayley graph on Z4 x Z4: strongly regular (16, 6, 2, 2), like the 4 x 4 rook's graph."""
    steps = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    return _graph(
        {
            tuple(sorted((f"{a}{b}", f"{(a + x) % 4}{(b + y) % 4}")))
            for a, b in itertools.product(range(4), repeat=2)
            for x, y in steps
        }
    )


def _rook():
    cells = list(itertools.product(range(4), repeat=2))
    return _graph(
        [
            (f"{a}{b}", f"{c}{d}")
            for (a, b), (c, d) in itertools.combinations(cells, 2)
            if (a == c) != (b == d)
        ]
    )


def _nx_graph(G):
    return _graph(list(G.edges()), list(G.nodes()), directed=G.is_directed())


# =============================================================================
# Invariance on the PR's graphs
# =============================================================================


class TestPRGraphs:
    @pytest.mark.parametrize("graph", PR_GRAPHS, ids=lambda f: f.__name__)
    def test_invariance(self, graph):
        g = _from_strings(graph())
        want = _form(g)
        rng = random.Random(42)
        for _ in range(3):
            assert _form(_relabeled(g, rng)) == want

    @pytest.mark.parametrize("graph", PR_GRAPHS, ids=lambda f: f.__name__)
    def test_is_a_numbering(self, graph):
        g = _from_strings(graph())
        number = g.canonical_labeling()
        assert list(number) == sorted(g.get_node_ids(limit=None))
        assert sorted(number.values()) == list(range(g.count_nodes()))

    def test_c240_with_node_attributes(self):
        # The PR's attribute example: a distinct value on one node, then on another
        g = _from_strings(C240())
        first = g.get_node_ids(limit=None)[0]
        g.set_node_attrs("foo", 0).set_node_attrs("bar", 0)
        g.set_node_attrs("foo", {first: 42})
        g.set_node_attrs("bar", {g.get_id_from_label("83"): 42})
        rng = random.Random(3)
        for attrs in ([], ["foo"], ["foo", "bar"]):
            want = _form(g, node_attr=attrs or None)
            assert _form(_relabeled(g, rng), node_attr=attrs or None) == want

    def test_c240_attribute_fixes_the_marked_node(self):
        # With the attribute, the marked node's number is the same in every relabeled copy
        g = _from_strings(C240())
        g.set_node_attrs("a", 0).set_node_attrs("a", 1, nodes=["83"])
        rng = random.Random(5)
        numbers = set()
        for _ in range(3):
            h = _relabeled(g, rng)
            marked = next(n for n, v in h.get_node_attrs("a").items() if v == 1)
            numbers.add(h.canonical_labeling(node_attr="a")[marked])
        assert len(numbers) == 1


# =============================================================================
# Canonical forms decide isomorphism
# =============================================================================


class TestAgainstIsomorphism:
    @pytest.mark.parametrize("seed", range(200))
    def test_random_small_graphs(self, seed):
        rng = random.Random(seed)
        g = _random_graph(rng)
        node_attr, edge_attr = _attr_names(g)
        want = _form(g, node_attr, edge_attr)
        # an isomorphic copy has the same form
        assert _form(_relabeled(g, rng), node_attr, edge_attr) == want
        # another graph has the same form exactly when it is isomorphic
        n, m = g.count_nodes(), g.count_edges()
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(m)]
        node_attrs = {str(v): g.get_node(g.get_id_from_label(str(v))).attributes for v in range(n)}
        edge_attrs = {i: {"w": rng.randint(0, 1)} for i in range(m)} if edge_attr else None
        h = _graph(edges, range(n), g.is_directed(), node_attrs, edge_attrs)
        same = _form(h, node_attr, edge_attr) == want
        assert same == g.is_isomorphic(h, node_attr=node_attr, edge_attr=edge_attr)

    @pytest.mark.parametrize(
        "G",
        [
            nx.petersen_graph(),
            nx.complete_graph(12),
            nx.hypercube_graph(5),
            nx.dodecahedral_graph(),
            nx.grid_2d_graph(8, 8),
            nx.balanced_tree(3, 4),
            nx.disjoint_union_all([nx.cycle_graph(3)] * 20),
            nx.random_regular_graph(3, 200, seed=1),
            nx.gnm_random_graph(300, 900, seed=2),
            nx.paley_graph(37).to_undirected(),
            nx.gn_graph(100, seed=3),
        ],
        ids=[
            "petersen", "K12", "Q5", "dodecahedron", "grid", "tree", "triangles", "3-regular",
            "gnm", "paley", "directed-tree",
        ],
    )
    def test_named_graphs(self, G):
        g = _nx_graph(G)
        rng = random.Random(7)
        assert _form(_relabeled(g, rng)) == _form(g)

    def test_hexagon_and_two_triangles(self):
        # same degrees, and color refinement cannot tell them apart
        hexagon = ec.Graph().add_cycle_graph(6)
        triangles = ec.Graph().add_cycle_graph(3).add_cycle_graph(3)
        assert _form(hexagon) != _form(triangles)

    def test_shrikhande_and_rook(self):
        # two strongly regular graphs with the same parameters
        shrikhande, rook = _shrikhande(), _rook()
        assert shrikhande.count_edges() == rook.count_edges() == 48
        assert not shrikhande.is_isomorphic(rook)
        assert _form(shrikhande) != _form(rook)
        rng = random.Random(1)
        assert _form(_relabeled(shrikhande, rng)) == _form(shrikhande)
        assert _form(_relabeled(rook, rng)) == _form(rook)


# =============================================================================
# Direction, multigraphs and attributes
# =============================================================================


class TestStructure:
    def test_direction_counts(self):
        out_star = _graph([("c", "a"), ("c", "b"), ("c", "d")], directed=True)
        in_star = _graph([("a", "c"), ("b", "c"), ("d", "c")], directed=True)
        assert _form(out_star) != _form(in_star)
        path = _graph([("a", "b"), ("b", "c")], directed=True)
        bent = _graph([("a", "b"), ("c", "b")], directed=True)
        assert _form(path) != _form(bent)

    def test_directed_cycle_and_reverse(self):
        g = _graph([(i, (i + 1) % 5) for i in range(5)], directed=True)
        h = _graph([((i + 1) % 5, i) for i in range(5)], directed=True)
        assert _form(g) == _form(h)

    def test_parallel_edges_count(self):
        one = _graph([("a", "b"), ("b", "c")])
        two = _graph([("a", "b"), ("a", "b"), ("b", "c")])
        assert _form(one) != _form(two)
        other_end = _graph([("a", "b"), ("b", "c"), ("b", "c")])
        assert _form(two) == _form(other_end)

    def test_self_loops_count(self):
        plain = _graph([("a", "b"), ("b", "c")])
        end_loop = _graph([("a", "b"), ("b", "c"), ("a", "a")])
        middle_loop = _graph([("a", "b"), ("b", "c"), ("b", "b")])
        assert len({str(_form(g)) for g in (plain, end_loop, middle_loop)}) == 3
        assert _form(end_loop) == _form(_graph([("a", "b"), ("b", "c"), ("c", "c")]))

    def test_edge_attributes(self):
        def path(weights):
            edges = [("a", "b"), ("b", "c"), ("c", "d")]
            return _graph(edges, edge_attrs={i: {"w": w} for i, w in enumerate(weights)})

        assert _form(path([1, 2, 1]), edge_attr="w") != _form(path([1, 1, 2]), edge_attr="w")
        assert _form(path([1, 1, 2]), edge_attr="w") == _form(path([2, 1, 1]), edge_attr="w")
        # ignored unless named
        assert _form(path([1, 2, 1])) == _form(path([1, 1, 2]))

    def test_parallel_edge_attributes(self):
        def g(first, second):
            edges = [("a", "b"), ("a", "b"), ("b", "c")]
            return _graph(edges, edge_attrs={0: {"w": first}, 1: {"w": second}, 2: {"w": 0}})

        assert _form(g(1, 2), edge_attr="w") == _form(g(2, 1), edge_attr="w")
        assert _form(g(1, 2), edge_attr="w") != _form(g(1, 1), edge_attr="w")

    def test_node_attribute_values_ranked_not_by_appearance(self):
        # The same colored path, with the values first met in opposite orders
        g = _graph([("a", "b"), ("b", "c")], node_attrs={"a": {"c": "x"}, "b": {"c": "y"},
                                                         "c": {"c": "x"}})
        h = _graph([("b", "a"), ("a", "c")], node_attrs={"b": {"c": "x"}, "a": {"c": "y"},
                                                         "c": {"c": "x"}})
        assert _form(g, "c") == _form(h, "c")
        middle_g = g.canonical_labeling(node_attr="c")[g.get_id_from_label("b")]
        middle_h = h.canonical_labeling(node_attr="c")[h.get_id_from_label("a")]
        assert middle_g == middle_h

    def test_several_attributes_missing_and_unhashable(self):
        # with both attributes the three nodes all differ ("b" has no "k", "c" has no "t",
        # and a missing value counts as None), so each gets the same number in every copy
        attrs = {"a": {"k": [1, 2], "t": 1}, "b": {"t": 1}, "c": {"k": [1, 2]}}
        g = _graph([("a", "b"), ("b", "c")], node_attrs=attrs)

        def by_values(graph):
            number = graph.canonical_labeling(node_attr=["k", "t"])
            nodes = {n: graph.get_node(n).attributes for n in number}
            return {repr(sorted(nodes[n].items())): k for n, k in number.items()}

        want = by_values(g)
        assert len(set(want.values())) == 3
        rng = random.Random(0)
        for _ in range(4):
            assert by_values(_relabeled(g, rng)) == want

    def test_uniform_partitions(self):
        for G in (nx.empty_graph(6), nx.complete_graph(6), nx.complete_bipartite_graph(3, 4),
                  nx.star_graph(5)):
            g = _nx_graph(G)
            assert _form(_relabeled(g, random.Random(2))) == _form(g)


# =============================================================================
# Edge cases and steps
# =============================================================================


class TestEdgeCases:
    def test_empty_graph(self):
        assert ec.Graph().canonical_labeling() == {}
        assert ec.Graph(directed=True).canonical_labeling() == {}

    def test_single_node(self):
        g = _graph([], ["x"])
        assert g.canonical_labeling() == {g.get_id_from_label("x"): 0}

    def test_isolated_nodes(self):
        g = _graph([("a", "b")], ["c", "d"])
        h = _graph([("d", "c")], ["a", "b"])
        assert _form(g) == _form(h)

    def test_deterministic(self):
        g = _from_strings(PR_GRAPHS[0]())
        assert g.canonical_labeling() == g.canonical_labeling()

    def test_does_not_modify_graph(self):
        g = _graph([("a", "b"), ("b", "c")], node_attrs={"a": {"c": 1}})
        before = (g.count_nodes(), g.count_edges(), g.get_node_attrs("c"))
        g.canonical_labeling(node_attr="c")
        assert (g.count_nodes(), g.count_edges(), g.get_node_attrs("c")) == before

    def test_steps(self):
        g = _graph([("a", "b"), ("b", "c")])
        number = g.canonical_labeling()
        step = g.steps[-1]
        assert step.code == "G.canonical_labeling()"
        assert step.result == number
        assert step.narration == "canonical_labeling(): numbered 3 nodes 0 to 2."
        one = _graph([], ["x"])
        one.canonical_labeling()
        assert one.steps[-1].narration == "canonical_labeling(): numbered 1 node."
        empty = ec.Graph()
        empty.canonical_labeling()
        assert empty.steps[-1].narration == "canonical_labeling(): the graph is empty."
