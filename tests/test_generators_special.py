"""Tests for the bipartite, duplication, internet, expander, and other special generators.

The deterministic constructions (and the random kernel graph, which shares NetworkX's random
stream) are checked edge for edge against NetworkX; the random models are checked for the
structure they guarantee, reproducibility, edge orientation, and argument validation.
"""

import random
from collections import Counter

import edgecase as ec
import networkx as nx
import pytest


def edge_pairs(g):
    """`(source, target)` pairs as 0-based indices, for a graph built from empty."""
    return [(s - 1, t - 1) for _, s, t in g.get_edges_as_tuples()]


def edge_multiset(g):
    return Counter(tuple(sorted(p)) for p in edge_pairs(g))


def nx_multiset(h):
    return Counter(tuple(sorted(e)) for e in h.edges())


def degrees_in_order(g):
    return [g.get_degree(v) for v in sorted(g.get_node_ids(limit=None))]


def node_attr(g, attr):
    values = g.get_node_attrs(attr)
    return [values.get(nid) for nid in sorted(g.get_node_ids(limit=None))]


def flat_kernel(c):
    return dict(kernel_integral=lambda u, w, z: c * (z - w), kernel_root=lambda u, w, r: w + r / c)


# Small calls for every generator, used for the shared checks below
CALLS = {
    "add_bipartite_configuration_model": dict(aseq=[2, 2, 1], bseq=[1, 2, 2], seed=1),
    "add_bipartite_havel_hakimi_graph": dict(aseq=[2, 2, 1], bseq=[1, 2, 2]),
    "add_bipartite_preferential_attachment_graph": dict(aseq=[2, 3, 1, 2], p=0.5, seed=1),
    "add_bipartite_gnp_graph": dict(n1=5, n2=6, p=0.4, seed=1),
    "add_bipartite_gnm_graph": dict(n1=5, n2=6, m=12, seed=1),
    "add_random_kernel_graph": dict(n=30, **flat_kernel(4), seed=1),
    "add_random_clustered_graph": dict(joint_degree_sequence=[(1, 1)] * 6, seed=1),
    "add_duplication_divergence_graph": dict(n=20, p=0.5, seed=1),
    "add_partial_duplication_graph": dict(n=20, n_initial=3, p=0.5, q=0.5, seed=1),
    "add_visibility_graph": dict(series=[3, 1, 4, 1, 5, 9, 2, 6]),
    "add_interval_graph": dict(intervals=[(0, 2), (1, 3), (4, 5), (2.5, 4.5)]),
    "add_margulis_gabber_galil_graph": dict(n=3),
    "add_chordal_cycle_graph": dict(p=11),
    "add_paley_graph": dict(p=13),
    "add_maybe_regular_expander": dict(n=12, d=4, seed=1),
    "add_random_regular_expander_graph": dict(n=12, d=4, seed=1),
    "add_sudoku_graph": dict(n=2),
    "add_prefix_tree": dict(paths=["ant", "and", "be"]),
    "add_atlas_graph": dict(index=100),
}

# Generators with no `type=` or `rel=` parameters
UNTYPED = {"add_random_internet_as_graph": dict(n=30, seed=1)}

SEEDED = sorted(m for m, kwargs in {**CALLS, **UNTYPED}.items() if "seed" in kwargs)

# Methods whose edges have no inherent direction: in a directed graph they point from the
# earlier-created node to the later one (or are loops)
FORWARD = [
    "add_bipartite_configuration_model",
    "add_bipartite_havel_hakimi_graph",
    "add_bipartite_preferential_attachment_graph",
    "add_bipartite_gnm_graph",
    "add_random_kernel_graph",
    "add_random_clustered_graph",
    "add_duplication_divergence_graph",
    "add_partial_duplication_graph",
    "add_visibility_graph",
    "add_interval_graph",
    "add_maybe_regular_expander",
    "add_random_regular_expander_graph",
    "add_sudoku_graph",
    "add_atlas_graph",
]


@pytest.mark.parametrize("method", sorted(CALLS))
def test_conventions(method):
    g = ec.Graph().add_node("existing")
    result = getattr(g, method)(type="t", rel="r", **CALLS[method])
    assert result is g
    assert [s.method for s in g._steps] == ["add_node", method]
    node_types = g.get_node_attrs("type")
    assert 1 not in node_types and set(node_types.values()) == {"t"}
    assert set(g.get_edge_attrs("rel").values()) == {"r"}
    assert g.get_degree(1) == 0
    assert "seed" not in g._steps[-1].code


def test_internet_as_conventions():
    g = ec.Graph().add_node("existing")
    assert g.add_random_internet_as_graph(n=30, seed=1) is g
    assert [s.method for s in g._steps] == ["add_node", "add_random_internet_as_graph"]
    assert g.get_degree(1) == 0 and g.count_nodes() == 31
    assert "seed" not in g._steps[-1].code


LABEL_FNS = {"add_margulis_gabber_galil_graph", "add_sudoku_graph"}


@pytest.mark.parametrize("method", sorted({**CALLS, **UNTYPED}))
def test_labels(method):
    kwargs = {**CALLS, **UNTYPED}[method]
    g = getattr(ec.Graph(), method)(**kwargs)
    if method not in LABEL_FNS:
        assert all(g.get_label_from_id(nid) == str(nid) for nid in g.get_node_ids())
    g = getattr(ec.Graph(), method)(label=False, **kwargs)
    assert all(g.get_label_from_id(nid) is None for nid in g.get_node_ids())


@pytest.mark.parametrize("method", sorted(LABEL_FNS))
def test_label_clash_adds_nothing(method):
    g = getattr(ec.Graph(), method)(**CALLS[method])
    before = (g.count_nodes(), g.count_edges(), len(g._steps))
    with pytest.raises(ValueError, match="already exist"):
        getattr(g, method)(**CALLS[method])
    assert (g.count_nodes(), g.count_edges(), len(g._steps)) == before
    getattr(g, method)(label=False, **CALLS[method])
    assert g.count_nodes() == 2 * before[0]


@pytest.mark.parametrize("method", SEEDED)
def test_reproducible(method):
    kwargs = {**CALLS, **UNTYPED}[method]
    for directed in (False, True):
        a = getattr(ec.Graph(directed=directed), method)(**kwargs)
        b = getattr(ec.Graph(directed=directed), method)(**kwargs)
        assert edge_pairs(a) == edge_pairs(b)


@pytest.mark.parametrize("method", FORWARD)
def test_symmetric_models_point_forward(method):
    g = getattr(ec.Graph(directed=True), method)(**CALLS[method])
    assert g.count_edges() > 0
    assert all(s <= t for s, t in edge_pairs(g))


@pytest.mark.parametrize("method", FORWARD)
def test_same_edges_with_or_without_direction(method):
    directed = getattr(ec.Graph(directed=True), method)(**CALLS[method])
    undirected = getattr(ec.Graph(), method)(**CALLS[method])
    assert edge_pairs(directed) == edge_pairs(undirected)


# ---------------------------------------------------------------------------
# Bipartite generators
# ---------------------------------------------------------------------------


def assert_bipartite_parts(g, n1, n2):
    parts = node_attr(g, "part")
    assert parts == [0] * n1 + [1] * n2
    for s, t in edge_pairs(g):
        assert parts[s] != parts[t]


class TestBipartiteConfiguration:
    @pytest.mark.parametrize("seed", range(5))
    def test_degrees(self, seed):
        aseq, bseq = [3, 1, 0, 2, 2], [2, 2, 2, 1, 1]
        g = ec.Graph().add_bipartite_configuration_model(aseq, bseq, seed=seed)
        assert degrees_in_order(g) == aseq + bseq
        assert_bipartite_parts(g, 5, 5)
        assert g.count_all_loops() == 0

    def test_part_attr(self):
        g = ec.Graph().add_bipartite_configuration_model([1], [1], part_attr="side")
        assert node_attr(g, "side") == [0, 1]
        g = ec.Graph().add_bipartite_configuration_model([1], [1], part_attr=None)
        assert g.get_node_attrs("part") == {}

    @pytest.mark.parametrize("aseq,bseq", [([2], [1]), ([-1, 2], [1]), ([1], [])])
    def test_invalid(self, aseq, bseq):
        with pytest.raises(ValueError):
            ec.Graph().add_bipartite_configuration_model(aseq, bseq)

    def test_empty(self):
        g = ec.Graph().add_bipartite_configuration_model([], [])
        assert g.count_nodes() == 0 and len(g._steps) == 1


NX_HH = {
    "largest": nx.bipartite.havel_hakimi_graph,
    "smallest": nx.bipartite.reverse_havel_hakimi_graph,
    "alternating": nx.bipartite.alternating_havel_hakimi_graph,
}


class TestBipartiteHavelHakimi:
    @pytest.mark.parametrize("variant", sorted(NX_HH))
    def test_matches_networkx(self, variant):
        rng = random.Random(5)
        compared = 0
        for _ in range(3000):
            aseq = [rng.randint(0, 4) for _ in range(rng.randint(1, 6))]
            bseq = [rng.randint(0, 4) for _ in range(rng.randint(1, 6))]
            if sum(aseq) != sum(bseq):
                continue
            h = NX_HH[variant](aseq, bseq)
            # NetworkX can return wrong degrees or parallel edges where we raise
            nx_valid = [d for _, d in sorted(h.degree())] == aseq + bseq
            nx_valid = nx_valid and h.number_of_edges() == nx.Graph(h).number_of_edges()
            try:
                g = ec.Graph().add_bipartite_havel_hakimi_graph(aseq, bseq, variant=variant)
            except ValueError:
                assert not nx_valid
                continue
            assert degrees_in_order(g) == aseq + bseq
            assert g.is_simple()
            assert_bipartite_parts(g, len(aseq), len(bseq))
            if nx_valid:
                assert sorted(map(tuple, map(sorted, edge_pairs(g)))) == sorted(
                    tuple(sorted(e)) for e in h.edges()
                )
                compared += 1
        assert compared > 20

    def test_largest_realizes_every_graphical_pair(self):
        rng = random.Random(1)
        for _ in range(200):
            g0 = ec.Graph().add_bipartite_gnp_graph(4, 5, p=0.5, seed=rng.randrange(10**6))
            degrees = degrees_in_order(g0)
            g = ec.Graph().add_bipartite_havel_hakimi_graph(degrees[:4], degrees[4:])
            assert degrees_in_order(g) == degrees

    def test_invalid(self):
        with pytest.raises(ValueError, match="realized"):
            ec.Graph().add_bipartite_havel_hakimi_graph([3, 1], [2, 2])
        with pytest.raises(ValueError):
            ec.Graph().add_bipartite_havel_hakimi_graph([1], [1], variant="random")
        with pytest.raises(ValueError):
            ec.Graph().add_bipartite_havel_hakimi_graph([2], [1])


class TestBipartitePreferentialAttachment:
    @pytest.mark.parametrize("p", [0.0, 0.3, 1.0])
    def test_structure(self, p):
        aseq = [3, 0, 2, 4, 1]
        g = ec.Graph().add_bipartite_preferential_attachment_graph(aseq, p=p, seed=3)
        parts = node_attr(g, "part")
        n2 = parts.count(1)
        assert_bipartite_parts(g, 5, n2)
        assert degrees_in_order(g)[:5] == aseq
        assert all(d > 0 for d in degrees_in_order(g)[5:])
        if p == 0:
            assert n2 == 1
        if p == 1:
            assert n2 == sum(aseq) and g.is_simple()

    def test_invalid(self):
        for kwargs in (dict(aseq=[1], p=1.5), dict(aseq=[-1], p=0.5)):
            with pytest.raises(ValueError):
                ec.Graph().add_bipartite_preferential_attachment_graph(**kwargs)


class TestBipartiteGnp:
    def test_extremes(self):
        assert ec.Graph().add_bipartite_gnp_graph(4, 5, p=0).count_edges() == 0
        full = ec.Graph().add_bipartite_gnp_graph(4, 5, p=1)
        assert full.count_edges() == 20 and full.is_simple()
        assert_bipartite_parts(full, 4, 5)
        both = ec.Graph(directed=True).add_bipartite_gnp_graph(4, 5, p=1)
        assert Counter(s < 4 for s, _ in edge_pairs(both)) == {True: 20, False: 20}

    def test_density(self):
        g = ec.Graph().add_bipartite_gnp_graph(60, 50, p=0.2, seed=1)
        assert abs(g.count_edges() - 600) < 100
        assert g.is_simple() and g.is_bipartite()

    def test_directed_second_pass_points_back(self):
        g = ec.Graph(directed=True).add_bipartite_gnp_graph(10, 12, p=0.3, seed=2)
        forward = {(s, t) for s, t in edge_pairs(g) if s < 10}
        back = {(s, t) for s, t in edge_pairs(g) if s >= 10}
        assert all(t >= 10 for _, t in forward) and all(t < 10 for _, t in back)
        assert forward and back

    @pytest.mark.parametrize("kwargs", [dict(n1=-1, n2=2, p=0.5), dict(n1=2, n2=2, p=-0.1)])
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            ec.Graph().add_bipartite_gnp_graph(**kwargs)


class TestBipartiteGnm:
    @pytest.mark.parametrize("n1,n2,m", [(4, 5, 0), (4, 5, 7), (4, 5, 20), (4, 5, 99), (1, 6, 3)])
    def test_edge_count(self, n1, n2, m):
        g = ec.Graph().add_bipartite_gnm_graph(n1, n2, m=m, seed=4)
        assert g.count_edges() == min(m, n1 * n2) and g.is_simple()
        assert_bipartite_parts(g, n1, n2)
        assert all(s < n1 <= t for s, t in edge_pairs(g))

    def test_uniform_over_pairs(self):
        counts = Counter()
        for seed in range(400):
            counts.update(edge_pairs(ec.Graph().add_bipartite_gnm_graph(2, 2, m=1, seed=seed)))
        assert len(counts) == 4 and min(counts.values()) > 60

    def test_invalid(self):
        with pytest.raises(ValueError):
            ec.Graph().add_bipartite_gnm_graph(2, 2, m=-1)


# ---------------------------------------------------------------------------
# Kernel, clustered, and duplication models
# ---------------------------------------------------------------------------


class TestRandomKernel:
    @pytest.mark.parametrize("seed", range(10))
    def test_matches_networkx_with_root(self, seed):
        kernel = flat_kernel(3)
        h = nx.random_kernel_graph(80, kernel["kernel_integral"], kernel["kernel_root"], seed=seed)
        g = ec.Graph().add_random_kernel_graph(80, **kernel, seed=seed)
        assert edge_multiset(g) == nx_multiset(h)

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_networkx_without_root(self, seed):
        pytest.importorskip("scipy")

        def integral(u, w, z):
            return 2 * (z**0.5 - w**0.5) / u**0.5

        h = nx.random_kernel_graph(60, integral, seed=seed)
        g = ec.Graph().add_random_kernel_graph(60, integral, seed=seed)
        assert edge_multiset(g) == nx_multiset(h)

    def test_simple(self):
        g = ec.Graph().add_random_kernel_graph(200, **flat_kernel(6), seed=1)
        assert g.is_simple() and g.count_nodes() == 200

    def test_small_and_invalid(self):
        assert ec.Graph().add_random_kernel_graph(0, **flat_kernel(1)).count_nodes() == 0
        assert ec.Graph().add_random_kernel_graph(1, **flat_kernel(1)).count_edges() == 0
        with pytest.raises(ValueError):
            ec.Graph().add_random_kernel_graph(-1, **flat_kernel(1))


class TestRandomClustered:
    @pytest.mark.parametrize("seed", range(5))
    def test_degrees(self, seed):
        joint = [(1, 0), (2, 1), (0, 2), (1, 1), (2, 1), (0, 1)]
        g = ec.Graph().add_random_clustered_graph(joint, seed=seed)
        assert degrees_in_order(g) == [s + 2 * t for s, t in joint]
        assert g.count_edges() == 6 // 2 + 3 * 6 // 3

    def test_triangles_raise_transitivity(self):
        g = ec.Graph().add_random_clustered_graph([(0, 3)] * 99, seed=1)
        assert g.transitivity() > 0.15

    @pytest.mark.parametrize("joint", [[(1, 0)], [(0, 1)], [(-1, 1), (1, 2)]])
    def test_invalid(self, joint):
        with pytest.raises(ValueError):
            ec.Graph().add_random_clustered_graph(joint)


class TestDuplicationDivergence:
    @pytest.mark.parametrize("p", [0.1, 0.5, 1.0])
    def test_connected(self, p):
        g = ec.Graph().add_duplication_divergence_graph(n=60, p=p, seed=2)
        assert g.count_nodes() == 60 and g.is_connected() and g.is_simple()

    def test_p_one_gives_star_like_bipartite(self):
        # with p = 1, copies keep every edge, so the graph stays complete bipartite
        g = ec.Graph().add_duplication_divergence_graph(n=12, p=1.0, seed=1)
        assert g.is_bipartite()

    def test_two_nodes_and_invalid(self):
        assert ec.Graph().add_duplication_divergence_graph(n=2, p=0).count_edges() == 1
        for kwargs in (dict(n=1, p=0.5), dict(n=5, p=0), dict(n=5, p=1.5)):
            with pytest.raises(ValueError):
                ec.Graph().add_duplication_divergence_graph(**kwargs)


class TestPartialDuplication:
    def test_initial_clique_and_growth(self):
        g = ec.Graph().add_partial_duplication_graph(n=40, n_initial=4, p=0.5, q=0.5, seed=1)
        assert g.count_nodes() == 40 and g.is_simple()
        assert {(i, j) for i in range(4) for j in range(i + 1, 4)} <= set(edge_pairs(g))

    def test_extremes(self):
        none = ec.Graph().add_partial_duplication_graph(n=10, n_initial=1, p=0, q=0, seed=1)
        assert none.count_edges() == 0
        tree = ec.Graph().add_partial_duplication_graph(n=30, n_initial=1, p=0, q=1, seed=1)
        assert tree.is_tree()
        full = ec.Graph().add_partial_duplication_graph(n=8, n_initial=2, p=1, q=1, seed=1)
        assert full.count_edges() == 28

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(n=3, n_initial=4, p=0.5, q=0.5),
            dict(n=3, n_initial=0, p=0.5, q=0.5),
            dict(n=3, n_initial=1, p=1.5, q=0.5),
            dict(n=3, n_initial=1, p=0.5, q=-1),
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            ec.Graph().add_partial_duplication_graph(**kwargs)


# ---------------------------------------------------------------------------
# Visibility, interval, and internet graphs
# ---------------------------------------------------------------------------


class TestVisibility:
    @pytest.mark.parametrize("seed", range(10))
    def test_matches_networkx(self, seed):
        rng = random.Random(seed)
        series = [rng.random() * 10 for _ in range(40)]
        g = ec.Graph().add_visibility_graph(series)
        assert set(edge_multiset(g)) == set(nx_multiset(nx.visibility_graph(series)))
        assert node_attr(g, "value") == series

    def test_exact_collinear_points_block(self):
        assert edge_pairs(ec.Graph().add_visibility_graph([1, 2, 3])) == [(0, 1), (1, 2)]
        assert edge_pairs(ec.Graph().add_visibility_graph([0, 3, 6, 9])) == [
            (0, 1),
            (1, 2),
            (2, 3),
        ]

    def test_contains_path_and_no_value_attr(self):
        g = ec.Graph().add_visibility_graph([5, 2, 8, 1, 1, 7], value_attr=None)
        assert {(i, i + 1) for i in range(5)} <= set(edge_pairs(g))
        assert g.get_node_attrs("value") == {}

    def test_invalid(self):
        with pytest.raises(ValueError):
            ec.Graph().add_visibility_graph([1, float("nan")])


class TestInterval:
    @pytest.mark.parametrize("seed", range(10))
    def test_matches_networkx(self, seed):
        rng = random.Random(seed)
        intervals = []
        while len(intervals) < 25:
            lo = rng.randint(0, 30)
            iv = (lo, lo + rng.randint(0, 6))
            if iv not in intervals:
                intervals.append(iv)
        g = ec.Graph().add_interval_graph(intervals)
        h = nx.interval_graph(intervals)
        expected = {tuple(sorted((intervals.index(a), intervals.index(b)))) for a, b in h.edges()}
        assert set(edge_multiset(g)) == expected
        assert node_attr(g, "interval") == intervals

    def test_closed_and_repeated(self):
        g = ec.Graph().add_interval_graph([(0, 1), (1, 2), (0, 1), (3, 3)])
        assert edge_pairs(g) == [(0, 1), (0, 2), (1, 2)]
        assert g.count_nodes() == 4 and g.count_all_loops() == 0

    def test_lists_and_tuples(self):
        g = ec.Graph().add_interval_graph([[0, 1], (2, 3)], interval_attr="span")
        assert node_attr(g, "span") == [(0, 1), (2, 3)]

    @pytest.mark.parametrize("bad", [[(1,)], [5], ["ab"], [(1, 2, 3)]])
    def test_type_errors(self, bad):
        with pytest.raises(TypeError):
            ec.Graph().add_interval_graph(bad)

    @pytest.mark.parametrize("bad", [[(2, 1)], [(float("nan"), 1)]])
    def test_value_errors(self, bad):
        with pytest.raises(ValueError):
            ec.Graph().add_interval_graph(bad)


class TestInternetAs:
    @pytest.mark.parametrize("n", [0, 1, 5, 7, 8, 30, 500])
    def test_exact_size(self, n):
        g = ec.Graph().add_random_internet_as_graph(n=n, seed=2)
        assert g.count_nodes() == n

    def test_structure(self):
        g = ec.Graph().add_random_internet_as_graph(n=600, seed=5)
        kinds = node_attr(g, "type")
        assert set(kinds) == {"T", "M", "CP", "C"}
        assert 4 <= kinds.count("T") <= 6
        assert g.is_connected() and g.is_simple()
        rels = g.get_edge_attrs("rel")
        tiers = [i for i, k in enumerate(kinds) if k == "T"]
        peer = {frozenset(p) for eid, *p in g.get_edges_as_tuples() if rels[eid] == "peer"}
        assert all(frozenset((a + 1, b + 1)) in peer for a in tiers for b in tiers if a < b)
        providers = {i: set() for i in range(600)}
        for eid, s, t in g.get_edges_as_tuples():
            if rels[eid] == "transit":
                providers[s - 1].add(t - 1)
        for i, kind in enumerate(kinds):
            assert (kind == "T") == (not providers[i])
            assert all(kinds[p] in ("T", "M") for p in providers[i])

    def test_invalid(self):
        with pytest.raises(ValueError):
            ec.Graph().add_random_internet_as_graph(n=-1)


# ---------------------------------------------------------------------------
# Expanders
# ---------------------------------------------------------------------------


class TestMargulisGabberGalil:
    @pytest.mark.parametrize("n", range(6))
    def test_matches_networkx(self, n):
        g = ec.Graph().add_margulis_gabber_galil_graph(n)
        h = nx.margulis_gabber_galil_graph(n)
        index = {(x, y): x * n + y for x in range(n) for y in range(n)}
        assert edge_multiset(g) == Counter(
            tuple(sorted((index[u], index[v]))) for u, v in h.edges()
        )
        assert set(degrees_in_order(g)) <= {8}

    def test_labels(self):
        g = ec.Graph().add_margulis_gabber_galil_graph(3)
        assert [g.get_label_from_id(v) for v in sorted(g.get_node_ids())][:4] == [
            "(0,0)",
            "(0,1)",
            "(0,2)",
            "(1,0)",
        ]


class TestChordalCycle:
    @pytest.mark.parametrize("p", [1, 2, 3, 5, 7, 11, 13, 101])
    def test_networkx_doubles_every_edge(self, p):
        g = ec.Graph().add_chordal_cycle_graph(p)
        ours = edge_multiset(g)
        if p > 1:
            doubled = Counter({e: c if e[0] == e[1] else 2 * c for e, c in ours.items()})
            assert doubled == nx_multiset(nx.chordal_cycle_graph(p))
        assert set(ours) == set(nx_multiset(nx.chordal_cycle_graph(p)))

    def test_three_regular(self):
        g = ec.Graph().add_chordal_cycle_graph(31)
        assert g.count_edges() == 31 + (31 - 3) // 2 + 3 and g.count_all_loops() == 3
        assert degrees_in_order(g) == [4 if x in (0, 1, 30) else 3 for x in range(31)]


class TestPaley:
    @pytest.mark.parametrize("p", [2, 3, 5, 7, 11, 13, 17])
    def test_matches_networkx(self, p):
        h = nx.paley_graph(p)
        g = ec.Graph(directed=True).add_paley_graph(p)
        assert sorted(edge_pairs(g)) == sorted(h.edges())
        u = ec.Graph().add_paley_graph(p)
        assert sorted(edge_pairs(u)) == sorted({tuple(sorted(e)) for e in h.edges()})
        assert all(s < t for s, t in edge_pairs(u))

    def test_strongly_regular_parameters(self):
        g = ec.Graph().add_paley_graph(13)
        assert set(degrees_in_order(g)) == {6}
        assert g.is_isomorphic(g.complement())


class TestExpanders:
    @pytest.mark.parametrize(
        "method", ["add_maybe_regular_expander", "add_random_regular_expander_graph"]
    )
    @pytest.mark.parametrize("n,d", [(5, 4), (11, 2), (30, 4), (60, 6)])
    def test_regular_simple(self, method, n, d):
        g = getattr(ec.Graph(), method)(n=n, d=d, max_tries=1000, seed=3)
        assert set(degrees_in_order(g)) == {d} and g.is_simple() and g.is_connected()

    def test_random_regular_expander_passes(self):
        g = ec.Graph().add_random_regular_expander_graph(n=80, d=4, seed=1)
        assert g.is_regular_expander()

    @pytest.mark.parametrize(
        "kwargs", [dict(n=0, d=2), dict(n=10, d=3), dict(n=10, d=0), dict(n=4, d=4)]
    )
    def test_invalid(self, kwargs):
        for method in ("add_maybe_regular_expander", "add_random_regular_expander_graph"):
            with pytest.raises(ValueError):
                getattr(ec.Graph(), method)(**kwargs)
        with pytest.raises(ValueError):
            ec.Graph().add_random_regular_expander_graph(n=10, d=2, epsilon=-1)


class TestIsRegularExpander:
    def graphs(self):
        yield ec.Graph().add_named_graph("petersen")
        yield ec.Graph().add_cycle_graph(10)
        yield ec.Graph().add_cycle_graph(9)
        yield ec.Graph().add_full_graph(6)
        yield ec.Graph().add_star_graph(5)
        yield ec.Graph().add_named_graph("cubical")
        for seed in range(4):
            yield ec.Graph().add_random_regular_graph(n=40, d=3, seed=seed)
            yield ec.Graph().add_random_regular_graph(n=12, d=4, seed=seed)

    def test_matches_networkx(self):
        pytest.importorskip("scipy")
        for g in self.graphs():
            h = nx.Graph(ec.to_networkx(g))
            for eps in (0.0, 0.5):
                assert g.is_regular_expander(epsilon=eps) == nx.is_regular_expander(h, epsilon=eps)

    def test_edge_cases(self):
        assert not ec.Graph().is_regular_expander()
        assert not ec.Graph().add_node("a").is_regular_expander()
        assert not ec.Graph().add_nodes(["a", "b", "c"]).is_regular_expander()
        assert ec.Graph().add_named_graph("petersen").is_regular_expander()

    def test_invalid(self):
        with pytest.raises(ValueError):
            ec.Graph(directed=True).add_cycle_graph(5).is_regular_expander()
        with pytest.raises(ValueError):
            ec.Graph().add_chordal_cycle_graph(7).is_regular_expander()
        with pytest.raises(ValueError):
            ec.Graph().add_cycle_graph(5).is_regular_expander(epsilon=-0.1)


# ---------------------------------------------------------------------------
# Sudoku, prefix trees, atlas, and nonisomorphic trees
# ---------------------------------------------------------------------------


class TestSudoku:
    @pytest.mark.parametrize("n", range(5))
    def test_matches_networkx(self, n):
        g = ec.Graph().add_sudoku_graph(n)
        assert edge_multiset(g) == nx_multiset(nx.sudoku_graph(n))
        assert g.count_nodes() == n**4
        if n:
            assert set(degrees_in_order(g)) == {3 * n * n - 2 * n - 1}

    def test_labels(self):
        g = ec.Graph().add_sudoku_graph(2)
        assert g.get_label_from_id(6) == "(1,1)"
        assert sorted(g.get_label_from_id(v) for v in g.get_neighbors("(0,0)")) == [
            "(0,1)",
            "(0,2)",
            "(0,3)",
            "(1,0)",
            "(1,1)",
            "(2,0)",
            "(3,0)",
        ]

    def test_invalid(self):
        with pytest.raises(ValueError):
            ec.Graph().add_sudoku_graph(-1)


def nx_prefix_index(node):
    """Our 0-based index for NetworkX's prefix tree node (root 0, NIL -1, others k + 1)."""
    return {0: 0, -1: 1}.get(node, node + 1)


class TestPrefixTree:
    @pytest.mark.parametrize("seed", range(20))
    def test_matches_networkx(self, seed):
        rng = random.Random(seed)
        paths = ["".join(rng.choice("abc") for _ in range(rng.randint(0, 4))) for _ in range(8)]
        g = ec.Graph(directed=True).add_prefix_tree(paths)
        h = nx.prefix_tree(paths)
        assert g.count_nodes() == h.number_of_nodes()
        assert sorted(edge_pairs(g)) == sorted(
            (nx_prefix_index(u), nx_prefix_index(v)) for u, v in h.edges()
        )
        sources = node_attr(g, "source")
        for node, source in h.nodes(data="source"):
            assert sources[nx_prefix_index(node)] == source

    def test_sequences_of_any_items(self):
        g = ec.Graph(directed=True).add_prefix_tree([(1, 2), (1, 3), ()], source_attr="item")
        assert node_attr(g, "item") == [None, "NIL", 1, 2, 3]
        assert sorted(edge_pairs(g)) == [(0, 1), (0, 2), (2, 3), (2, 4), (3, 1), (4, 1)]

    def test_empty(self):
        g = ec.Graph().add_prefix_tree([])
        assert g.count_nodes() == 2 and g.count_edges() == 0


class TestAtlas:
    def test_every_graph_matches_networkx(self):
        atlas = ec.Graph.atlas_graphs()
        expected = nx.graph_atlas_g()
        assert len(atlas) == len(expected) == 1253
        for i, (g, h) in enumerate(zip(atlas, expected)):
            assert g.get_graph_name() == f"G{i}"
            assert g.count_nodes() == h.number_of_nodes()
            assert edge_multiset(g) == nx_multiset(h)

    @pytest.mark.parametrize("index", [0, 1, 7, 100, 208, 1252])
    def test_single(self, index):
        g = ec.Graph().add_atlas_graph(index)
        assert edge_multiset(g) == nx_multiset(nx.graph_atlas(index))
        assert g._steps[-1].code == f"G.add_atlas_graph({index})"

    @pytest.mark.parametrize("index", [-1, 1253])
    def test_out_of_range(self, index):
        with pytest.raises(ValueError):
            ec.Graph().add_atlas_graph(index)

    @pytest.mark.parametrize("index", [1.0, "3", True])
    def test_not_an_integer(self, index):
        with pytest.raises(TypeError):
            ec.Graph().add_atlas_graph(index)


class TestNonisomorphicTrees:
    @pytest.mark.parametrize("order", range(2, 11))
    def test_matches_networkx(self, order):
        ours = [edge_pairs(t) for t in ec.Graph.iter_nonisomorphic_trees(order)]
        theirs = [sorted(t.edges()) for t in nx.nonisomorphic_trees(order)]
        assert [sorted(map(tuple, map(sorted, e))) for e in ours] == [
            sorted(map(tuple, map(sorted, e))) for e in theirs
        ]
        assert len(ours) == ec.Graph.count_nonisomorphic_trees(order)

    def test_trees_are_new_graphs(self):
        trees = list(ec.Graph.iter_nonisomorphic_trees(6))
        assert [s.method for s in trees[0]._steps] == ["iter_nonisomorphic_trees"]
        assert all(t.is_tree() and t.count_nodes() == 6 for t in trees)
        assert all(not a.is_isomorphic(b) for i, a in enumerate(trees) for b in trees[i + 1 :])

    def test_counts(self):
        expected = [1, 1, 2, 3, 6, 11, 23, 47, 106, 235, 551, 1301, 3159]
        assert [ec.Graph.count_nonisomorphic_trees(n) for n in range(2, 15)] == expected
        assert ec.Graph.count_nonisomorphic_trees(30) == nx.number_of_nonisomorphic_trees(30)

    @pytest.mark.parametrize("order", [-1, 0, 1])
    def test_invalid(self, order):
        with pytest.raises(ValueError):
            ec.Graph.count_nonisomorphic_trees(order)
        with pytest.raises(ValueError):
            next(ec.Graph.iter_nonisomorphic_trees(order))
