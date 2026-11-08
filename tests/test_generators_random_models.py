"""Tests for the random, degree-sequence, directed-growth, and geometric generators.

The deterministic degree-sequence constructions are checked edge for edge against NetworkX;
the random models are checked for the structure they guarantee, reproducibility, edge
orientation, and argument validation.
"""

import math

import edgecase as ec
import networkx as nx
import pytest


def edge_pairs(g):
    """`(source, target)` pairs as 0-based indices, for a graph built from empty."""
    return [(s - 1, t - 1) for _, s, t in g.get_edges_as_tuples()]


def undirected_edges(g):
    return {frozenset(p) for p in edge_pairs(g)}


def as_nx(g):
    return nx.Graph(ec.to_networkx(g))


def degrees_in_order(g):
    return [g.get_degree(v) for v in sorted(g.get_node_ids())]


def node_attr(g, attr):
    values = g.get_node_attrs(attr)
    return [values[nid] for nid in sorted(values)]


# Small calls for every generator, used for the shared checks below
CALLS = {
    "add_random_regular_graph": dict(n=10, d=3, seed=1),
    "add_powerlaw_cluster_graph": dict(n=20, m=2, p=0.5, seed=1),
    "add_newman_watts_strogatz_graph": dict(n=12, k=4, p=0.3, seed=1),
    "add_connected_watts_strogatz_graph": dict(n=12, k=4, p=0.3, seed=1),
    "add_extended_barabasi_albert_graph": dict(n=20, m=2, p=0.2, q=0.2, seed=1),
    "add_dual_barabasi_albert_graph": dict(n=20, m1=1, m2=3, p=0.5, seed=1),
    "add_random_lobster": dict(n=6, p1=0.5, p2=0.5, seed=1),
    "add_random_labeled_tree": dict(n=12, seed=1),
    "add_random_powerlaw_tree": dict(n=10, seed=1),
    "add_havel_hakimi_graph": dict(degree_sequence=[3, 3, 2, 2, 2]),
    "add_directed_configuration_model": dict(
        in_degree_sequence=[1, 2, 0, 2], out_degree_sequence=[2, 1, 2, 0], seed=1
    ),
    "add_directed_havel_hakimi_graph": dict(
        in_degree_sequence=[1, 2, 0, 2], out_degree_sequence=[2, 1, 2, 0]
    ),
    "add_random_degree_sequence_graph": dict(degree_sequence=[3, 3, 2, 2, 2], seed=1),
    "add_joint_degree_graph": dict(joint_degrees={1: {2: 2}, 2: {1: 2, 2: 2}}, seed=1),
    "add_gn_graph": dict(n=15, seed=1),
    "add_gnr_graph": dict(n=15, p=0.5, seed=1),
    "add_gnc_graph": dict(n=15, seed=1),
    "add_scale_free_graph": dict(n=15, seed=1),
    "add_random_k_out_graph": dict(n=10, k=2, alpha=1.0, seed=1),
    "add_waxman_graph": dict(n=15, beta=0.8, alpha=0.5, seed=1),
    "add_soft_random_geometric_graph": dict(n=20, radius=0.5, seed=1),
    "add_geographical_threshold_graph": dict(n=20, theta=10, seed=1),
    "add_navigable_small_world_graph": dict(n=4, seed=1),
}

SEEDED = sorted(m for m, kwargs in CALLS.items() if "seed" in kwargs)

# Models whose edges have no inherent direction (in a directed graph they point from the
# earlier-created node to the later one), and those that point from a new node to an older one
LOWER_TO_HIGHER = [
    "add_random_regular_graph",
    "add_newman_watts_strogatz_graph",
    "add_extended_barabasi_albert_graph",
    "add_random_lobster",
    "add_random_labeled_tree",
    "add_random_powerlaw_tree",
    "add_havel_hakimi_graph",
    "add_random_degree_sequence_graph",
    "add_joint_degree_graph",
    "add_waxman_graph",
    "add_soft_random_geometric_graph",
    "add_geographical_threshold_graph",
]
NEWER_TO_OLDER = ["add_powerlaw_cluster_graph", "add_gn_graph", "add_gnr_graph", "add_gnc_graph"]


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
    # the code string never carries the seed
    assert "seed" not in g._steps[-1].code


@pytest.mark.parametrize("method", sorted(CALLS))
def test_labels(method):
    g = getattr(ec.Graph(), method)(**CALLS[method])
    assert all(g.get_label_from_id(nid) == str(nid) for nid in g.get_node_ids())
    g = getattr(ec.Graph(), method)(label=False, **CALLS[method])
    assert all(g.get_label_from_id(nid) is None for nid in g.get_node_ids())


@pytest.mark.parametrize("method", SEEDED)
def test_reproducible(method):
    for directed in (False, True):
        a = getattr(ec.Graph(directed=directed), method)(**CALLS[method])
        b = getattr(ec.Graph(directed=directed), method)(**CALLS[method])
        assert edge_pairs(a) == edge_pairs(b)


@pytest.mark.parametrize("method", LOWER_TO_HIGHER)
def test_symmetric_models_point_forward(method):
    g = getattr(ec.Graph(directed=True), method)(**CALLS[method])
    assert g.count_edges() > 0
    assert all(s < t for s, t in edge_pairs(g))


@pytest.mark.parametrize("method", NEWER_TO_OLDER)
def test_growth_models_point_back(method):
    g = getattr(ec.Graph(directed=True), method)(**CALLS[method])
    assert g.count_edges() > 0
    assert all(s > t for s, t in edge_pairs(g))


SIMPLE = LOWER_TO_HIGHER + [
    "add_powerlaw_cluster_graph",
    "add_dual_barabasi_albert_graph",
    "add_connected_watts_strogatz_graph",
    "add_directed_havel_hakimi_graph",
]


@pytest.mark.parametrize("method", SIMPLE)
def test_simple(method):
    g = getattr(ec.Graph(directed=True), method)(**CALLS[method])
    pairs = edge_pairs(g)
    assert all(s != t for s, t in pairs)
    assert len(set(pairs)) == len(pairs)


@pytest.mark.parametrize("method", sorted(CALLS))
def test_same_edges_with_or_without_direction(method):
    directed = getattr(ec.Graph(directed=True), method)(**CALLS[method])
    undirected = getattr(ec.Graph(), method)(**CALLS[method])
    if method == "add_navigable_small_world_graph":
        # an undirected graph merges the two directions of a link
        assert undirected_edges(directed) == undirected_edges(undirected)
    else:
        assert sorted(map(sorted, edge_pairs(directed))) == sorted(
            map(sorted, edge_pairs(undirected))
        )


# ---------------------------------------------------------------------------
# Random graphs
# ---------------------------------------------------------------------------


class TestRandomRegular:
    @pytest.mark.parametrize("n,d", [(10, 3), (20, 4), (7, 6), (8, 0), (0, 0)])
    def test_regular(self, n, d):
        g = ec.Graph().add_random_regular_graph(n=n, d=d, seed=2)
        assert g.count_nodes() == n and g.count_edges() == n * d // 2
        assert set(degrees_in_order(g)) <= {d}
        assert g.count_multi_edges() == 0 and g.count_all_loops() == 0

    @pytest.mark.parametrize("n,d", [(5, 3), (4, 4), (-2, 1), (3, -1)])
    def test_invalid(self, n, d):
        with pytest.raises(ValueError):
            ec.Graph().add_random_regular_graph(n=n, d=d)


class TestPowerlawCluster:
    def test_edge_count_and_clustering(self):
        g = ec.Graph().add_powerlaw_cluster_graph(n=200, m=3, p=0.0, seed=4)
        assert g.count_edges() == (200 - 3) * 3
        low = g.average_clustering_coefficient()
        high = ec.Graph().add_powerlaw_cluster_graph(n=200, m=3, p=0.9, seed=4)
        assert high.count_edges() <= (200 - 3) * 3
        assert high.average_clustering_coefficient() > low

    def test_invalid(self):
        for kwargs in (dict(n=5, m=0, p=0.5), dict(n=3, m=4, p=0.5), dict(n=5, m=2, p=1.5)):
            with pytest.raises(ValueError):
                ec.Graph().add_powerlaw_cluster_graph(**kwargs)


class TestWattsStrogatz:
    def test_newman_keeps_the_lattice(self):
        g = ec.Graph().add_newman_watts_strogatz_graph(n=30, k=4, p=0.5, seed=3)
        lattice = {frozenset((i, (i + j) % 30)) for i in range(30) for j in (1, 2)}
        assert lattice <= undirected_edges(g)
        assert g.count_edges() > 60 and g.is_connected()

    def test_newman_odd_k_and_complete(self):
        g = ec.Graph().add_newman_watts_strogatz_graph(n=10, k=3, p=0, seed=1)
        assert g.count_edges() == 10
        g = ec.Graph().add_newman_watts_strogatz_graph(n=6, k=6, p=0.5, seed=1)
        assert g.count_edges() == 15

    def test_connected(self):
        for seed in range(10):
            g = ec.Graph().add_connected_watts_strogatz_graph(n=30, k=2, p=0.5, seed=seed)
            assert g.is_connected() and g.count_edges() == 30

    def test_first_draw_matches_smallworld(self):
        for seed in range(20):
            sw = ec.Graph(directed=True).add_smallworld_graph(n=20, k=4, p=0.3, seed=seed)
            if not sw.is_weakly_connected():
                continue
            cws = ec.Graph(directed=True).add_connected_watts_strogatz_graph(
                n=20, k=4, p=0.3, seed=seed
            )
            assert edge_pairs(cws) == edge_pairs(sw)

    def test_connected_gives_up(self):
        with pytest.raises(ValueError, match="no connected"):
            ec.Graph().add_connected_watts_strogatz_graph(n=5, k=0, p=0.5, tries=3)
        for kwargs in (dict(n=5, k=3, p=0.1), dict(n=4, k=4, p=0.1), dict(n=5, k=2, p=0.1, tries=0)):
            with pytest.raises(ValueError):
                ec.Graph().add_connected_watts_strogatz_graph(**kwargs)


class TestBarabasiAlbertVariants:
    def test_extended_without_extra_steps(self):
        g = ec.Graph().add_extended_barabasi_albert_graph(n=50, m=2, p=0, q=0, seed=1)
        assert g.count_edges() == 48 * 2

    def test_extended_adds_edges(self):
        base = ec.Graph().add_extended_barabasi_albert_graph(n=80, m=2, p=0, q=0, seed=1)
        dense = ec.Graph().add_extended_barabasi_albert_graph(n=80, m=2, p=0.5, q=0, seed=1)
        assert dense.count_edges() > base.count_edges()
        assert dense.count_multi_edges() == 0 and dense.count_all_loops() == 0

    def test_extended_invalid(self):
        for kwargs in (dict(n=5, m=5, p=0.1, q=0.1), dict(n=5, m=1, p=0.6, q=0.4)):
            with pytest.raises(ValueError):
                ec.Graph().add_extended_barabasi_albert_graph(**kwargs)

    def test_dual_edge_counts(self):
        g = ec.Graph().add_dual_barabasi_albert_graph(n=30, m1=2, m2=4, p=1.0, seed=1)
        assert g.count_edges() == 2 + (30 - 3) * 2
        g = ec.Graph().add_dual_barabasi_albert_graph(n=30, m1=2, m2=4, p=0.0, seed=1)
        assert g.count_edges() == 4 + (30 - 5) * 4

    def test_dual_orientation(self):
        g = ec.Graph(directed=True).add_dual_barabasi_albert_graph(n=30, m1=1, m2=3, p=0.5, seed=1)
        pairs = edge_pairs(g)
        star = [(0, leaf) for leaf in range(1, 4)]
        assert pairs[:3] == star
        assert all(s > t for s, t in pairs[3:])

    def test_dual_invalid(self):
        with pytest.raises(ValueError, match="m2"):
            ec.Graph().add_dual_barabasi_albert_graph(n=5, m1=1, m2=5, p=0.5)


class TestTrees:
    def test_lobster(self):
        sizes = set()
        for seed in range(20):
            g = ec.Graph().add_random_lobster(n=6, p1=0.6, p2=0.5, seed=seed)
            sizes.add(g.count_nodes())
            if g.count_nodes() == 0:
                continue
            h = as_nx(g)
            assert nx.is_tree(h)
            # removing the leaves twice leaves the backbone, a path
            for _ in range(2):
                leaves = [v for v in h if h.degree(v) <= 1]
                if len(leaves) < h.number_of_nodes():
                    h.remove_nodes_from(leaves)
            assert max(dict(h.degree()).values(), default=0) <= 2
        assert len(sizes) > 5

    def test_caterpillar_when_no_leaves(self):
        g = ec.Graph().add_random_lobster(n=8, p1=0.5, p2=0.0, seed=2)
        h = as_nx(g)
        h.remove_nodes_from([v for v in list(h) if h.degree(v) == 1])
        assert max(dict(h.degree()).values()) <= 2

    def test_lobster_invalid(self):
        with pytest.raises(ValueError, match="p1"):
            ec.Graph().add_random_lobster(n=5, p1=1.0, p2=0.5)

    def test_labeled_tree(self):
        for n in (1, 2, 3, 30):
            g = ec.Graph().add_random_labeled_tree(n=n, seed=5)
            assert g.count_nodes() == n and g.is_tree()
        with pytest.raises(ValueError):
            ec.Graph().add_random_labeled_tree(n=0)

    def test_labeled_tree_covers_all_trees(self):
        # Cayley: 4 ** 2 = 16 labeled trees on 4 nodes, each about equally likely
        counts = {}
        for seed in range(1600):
            key = frozenset(undirected_edges(ec.Graph().add_random_labeled_tree(n=4, seed=seed)))
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == 16
        assert min(counts.values()) > 50

    def test_powerlaw_tree(self):
        g = ec.Graph().add_random_powerlaw_tree(n=30, gamma=2.5, seed=1, tries=1000)
        h = as_nx(g)
        assert nx.is_tree(h)
        # a caterpillar: dropping the leaves leaves a path
        h.remove_nodes_from([v for v in list(h) if h.degree(v) == 1])
        assert max(dict(h.degree()).values()) <= 2
        assert ec.Graph().add_random_powerlaw_tree(n=1).count_nodes() == 1

    def test_powerlaw_tree_invalid(self):
        with pytest.raises(ValueError, match="gamma"):
            ec.Graph().add_random_powerlaw_tree(n=5, gamma=1.0)
        with pytest.raises(ValueError, match="tries"):
            ec.Graph().add_random_powerlaw_tree(n=200, tries=0, seed=1)


# ---------------------------------------------------------------------------
# Degree sequences
# ---------------------------------------------------------------------------

GRAPHICAL = [
    [3, 3, 2, 2, 2],
    [4, 4, 4, 4, 4],
    [1, 1, 1, 1],
    [5, 3, 3, 3, 2, 2, 1, 1],
    [2, 2, 2, 2, 2, 2],
]


class TestHavelHakimi:
    @pytest.mark.parametrize("seq", GRAPHICAL)
    def test_matches_networkx(self, seq):
        g = ec.Graph().add_havel_hakimi_graph(seq)
        assert undirected_edges(g) == {frozenset(e) for e in nx.havel_hakimi_graph(seq).edges()}

    def test_zero_degrees_keep_positions(self):
        seq = [0, 2, 2, 2, 0, 1, 1]
        g = ec.Graph().add_havel_hakimi_graph(seq)
        assert degrees_in_order(g) == seq

    @pytest.mark.parametrize("seq", [[3, 3, 1, 1], [1], [2, 0], [-1, 1]])
    def test_invalid(self, seq):
        with pytest.raises(ValueError):
            ec.Graph().add_havel_hakimi_graph(seq)

    def test_empty(self):
        assert ec.Graph().add_havel_hakimi_graph([]).count_nodes() == 0


DIGRAPHICAL = [
    ([0, 1, 2, 3], [3, 2, 1, 0]),
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([2, 2, 2, 1, 1], [1, 2, 2, 2, 1]),
    ([3, 0, 1, 2, 1], [1, 3, 2, 0, 1]),
]


class TestDirectedDegreeSequences:
    @pytest.mark.parametrize("ins,outs", DIGRAPHICAL)
    def test_havel_hakimi_matches_networkx(self, ins, outs):
        g = ec.Graph(directed=True).add_directed_havel_hakimi_graph(ins, outs)
        assert set(edge_pairs(g)) == set(nx.directed_havel_hakimi_graph(ins, outs).edges())
        assert g.count_edges() == sum(ins)

    def test_havel_hakimi_invalid(self):
        for ins, outs in (([1], [1]), ([2, 0], [1, 1]), ([0, 0, 3], [1, 1, 1]), ([-1], [-1])):
            with pytest.raises(ValueError):
                ec.Graph(directed=True).add_directed_havel_hakimi_graph(ins, outs)

    def test_havel_hakimi_undirected_keeps_both_directions(self):
        g = ec.Graph().add_directed_havel_hakimi_graph([1, 1], [1, 1])
        assert g.count_edges() == 2 and degrees_in_order(g) == [2, 2]

    @pytest.mark.parametrize("ins,outs", DIGRAPHICAL)
    def test_configuration_degrees(self, ins, outs):
        g = ec.Graph(directed=True).add_directed_configuration_model(ins, outs, seed=2)
        nodes = sorted(g.get_node_ids())
        assert [g.get_in_degree(v) for v in nodes] == ins
        assert [g.get_out_degree(v) for v in nodes] == outs
        u = ec.Graph().add_directed_configuration_model(ins, outs, seed=2)
        assert degrees_in_order(u) == [i + o for i, o in zip(ins, outs)]

    def test_configuration_pads_shorter_sequence(self):
        g = ec.Graph(directed=True).add_directed_configuration_model([1, 1, 1], [3], seed=1)
        assert g.count_nodes() == 3 and g.get_out_degree(1) == 3

    def test_configuration_invalid(self):
        with pytest.raises(ValueError, match="equal sums"):
            ec.Graph().add_directed_configuration_model([1, 1], [1])


class TestRandomDegreeSequence:
    @pytest.mark.parametrize("seq", GRAPHICAL)
    def test_exact_simple(self, seq):
        for seed in range(5):
            # small, dense sequences often get stuck, so allow many attempts
            g = ec.Graph().add_random_degree_sequence_graph(seq, tries=100, seed=seed)
            assert degrees_in_order(g) == seq
            assert g.count_multi_edges() == 0 and g.count_all_loops() == 0

    def test_varies_with_seed(self):
        seq = [3] * 12
        graphs = {
            frozenset(undirected_edges(ec.Graph().add_random_degree_sequence_graph(seq, seed=s)))
            for s in range(10)
        }
        assert len(graphs) > 1

    def test_invalid(self):
        with pytest.raises(ValueError, match="graphical"):
            ec.Graph().add_random_degree_sequence_graph([3, 3, 1, 1])
        with pytest.raises(ValueError, match="tries"):
            ec.Graph().add_random_degree_sequence_graph([1, 1], tries=0)


class TestJointDegree:
    def test_round_trip(self):
        original = ec.Graph().add_powerlaw_cluster_graph(n=60, m=2, p=0.4, seed=8)
        joint = original.degree_mixing_dict()
        for seed in range(3):
            g = ec.Graph().add_joint_degree_graph(joint, seed=seed)
            assert g.degree_mixing_dict() == joint
            assert g.count_nodes() == 60 and g.count_edges() == original.count_edges()
            assert g.count_multi_edges() == 0 and g.count_all_loops() == 0

    def test_nodes_ordered_by_degree(self):
        joint = {1: {3: 3}, 3: {1: 3}}
        g = ec.Graph().add_joint_degree_graph(joint, seed=1)
        assert degrees_in_order(g) == [1, 1, 1, 3]

    def test_networkx_joint_degrees(self):
        h = nx.karate_club_graph()
        joint = {k: dict(row) for k, row in nx.degree_mixing_dict(h).items()}
        assert nx.is_valid_joint_degree(joint)
        g = ec.Graph().add_joint_degree_graph(joint, seed=1)
        assert g.degree_mixing_dict() == joint
        assert sorted(degrees_in_order(g)) == sorted(d for _, d in h.degree())

    @pytest.mark.parametrize(
        "joint",
        [
            {1: {2: 1}, 2: {1: 2}},  # not symmetric
            {2: {2: 3}},  # odd diagonal
            {2: {3: 1}, 3: {2: 1}},  # row sums not multiples of the degree
            {3: {3: 6}},  # three edges between the two degree-3 nodes
            {1: {2: -1}},
            [1, 2],
        ],
    )
    def test_invalid(self, joint):
        with pytest.raises(ValueError):
            ec.Graph().add_joint_degree_graph(joint)


# ---------------------------------------------------------------------------
# Directed growth
# ---------------------------------------------------------------------------


class TestGrowingNetworks:
    @pytest.mark.parametrize("method", ["add_gn_graph", "add_gnr_graph"])
    def test_trees_toward_first_node(self, method):
        kwargs = dict(n=40, seed=3) if method == "add_gn_graph" else dict(n=40, p=0.4, seed=3)
        g = getattr(ec.Graph(directed=True), method)(**kwargs)
        assert g.count_edges() == 39 and g.is_weakly_connected()
        assert [g.get_out_degree(v) for v in sorted(g.get_node_ids())] == [0] + [1] * 39
        assert as_nx(ec.Graph().add_gn_graph(n=40, seed=3)).number_of_edges() == 39

    def test_gn_kernel(self):
        star = ec.Graph(directed=True).add_gn_graph(n=30, kernel=lambda d: 1e6**d, seed=1)
        assert max(star.get_in_degree(v) for v in star.get_node_ids()) >= 25
        with pytest.raises(ValueError, match="zero weight"):
            ec.Graph().add_gn_graph(n=5, kernel=lambda d: 0)
        with pytest.raises(ValueError, match="non-negative"):
            ec.Graph().add_gn_graph(n=5, kernel=lambda d: -1)
        assert ec.Graph().add_gn_graph(n=1).count_nodes() == 1

    def test_gnr_without_redirection_is_uniform(self):
        g = ec.Graph(directed=True).add_gnr_graph(n=50, p=0.0, seed=2)
        assert all(s > t for s, t in edge_pairs(g))
        with pytest.raises(ValueError):
            ec.Graph().add_gnr_graph(n=5, p=1.5)

    def test_gnc_is_transitively_closed(self):
        g = ec.Graph(directed=True).add_gnc_graph(n=40, seed=6)
        h = nx.DiGraph(edge_pairs(g))
        assert nx.is_directed_acyclic_graph(h)
        assert set(nx.transitive_closure_dag(h).edges()) == set(h.edges())

    def test_scale_free(self):
        g = ec.Graph(directed=True).add_scale_free_graph(n=100, seed=5)
        assert g.count_nodes() == 100 and g.count_edges() >= 99
        assert edge_pairs(g)[:3] == [(0, 1), (1, 2), (2, 0)]
        assert ec.Graph().add_scale_free_graph(n=3, seed=1).count_edges() == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(n=2),
            dict(n=10, alpha=0.5, beta=0.5, gamma=0.5),
            dict(n=10, alpha=0.0, beta=0.95, gamma=0.05),
            dict(n=10, delta_in=-1),
        ],
    )
    def test_scale_free_invalid(self, kwargs):
        with pytest.raises(ValueError):
            ec.Graph().add_scale_free_graph(**kwargs)

    def test_k_out(self):
        g = ec.Graph(directed=True).add_random_k_out_graph(n=30, k=4, alpha=0.5, seed=2)
        assert g.count_edges() == 120
        assert {g.get_out_degree(v) for v in g.get_node_ids()} == {4}
        g = ec.Graph(directed=True).add_random_k_out_graph(
            n=30, k=4, alpha=0.5, self_loops=False, seed=2
        )
        assert g.count_all_loops() == 0
        assert ec.Graph().add_random_k_out_graph(n=5, k=0, alpha=1.0).count_edges() == 0

    def test_k_out_invalid(self):
        with pytest.raises(ValueError, match="alpha"):
            ec.Graph().add_random_k_out_graph(n=5, k=1, alpha=0)
        with pytest.raises(ValueError, match="self_loops"):
            ec.Graph().add_random_k_out_graph(n=1, k=1, alpha=1, self_loops=False)


# ---------------------------------------------------------------------------
# Geometric graphs
# ---------------------------------------------------------------------------


class TestGeometric:
    def test_waxman_positions(self):
        g = ec.Graph().add_waxman_graph(n=30, domain=(2, -1, 5, 3), seed=4)
        pos = node_attr(g, "pos")
        assert len(pos) == 30
        assert all(2 <= x <= 5 and -1 <= y <= 3 for x, y in pos)
        assert ec.Graph().add_waxman_graph(n=5, pos_attr=None, seed=1).get_node_attrs("pos") == {}

    def test_waxman_density(self):
        sparse = ec.Graph().add_waxman_graph(n=60, alpha=0.05, seed=2).count_edges()
        dense = ec.Graph().add_waxman_graph(n=60, alpha=1.0, beta=1.0, seed=2).count_edges()
        assert sparse < dense
        assert ec.Graph().add_waxman_graph(n=40, L=2.0, seed=2).count_edges() > 0

    @pytest.mark.parametrize(
        "kwargs",
        [dict(beta=0), dict(beta=1.5), dict(alpha=0), dict(L=-1.0), dict(domain=(1, 0, 0, 1))],
    )
    def test_waxman_invalid(self, kwargs):
        with pytest.raises(ValueError):
            ec.Graph().add_waxman_graph(n=5, **kwargs)

    @pytest.mark.parametrize("p_norm", [1.0, 2.0, 3.0, math.inf])
    def test_soft_rgg_respects_radius(self, p_norm):
        g = ec.Graph().add_soft_random_geometric_graph(
            n=40, radius=0.3, p_norm=p_norm, p_dist=lambda d: 1.0, seed=7
        )
        pos = node_attr(g, "pos")
        edges = undirected_edges(g)

        def dist(a, b):
            diffs = [abs(x - y) for x, y in zip(a, b)]
            return max(diffs) if p_norm == math.inf else sum(d**p_norm for d in diffs) ** (
                1 / p_norm
            )

        within = {
            frozenset((i, j))
            for i in range(40)
            for j in range(i + 1, 40)
            if dist(pos[i], pos[j]) <= 0.3
        }
        assert edges == within

    def test_soft_rgg_thins_edges(self):
        hard = ec.Graph().add_soft_random_geometric_graph(
            n=60, radius=0.4, dim=3, p_dist=lambda d: 1.0, seed=3
        )
        soft = ec.Graph().add_soft_random_geometric_graph(n=60, radius=0.4, dim=3, seed=3)
        assert undirected_edges(soft) < undirected_edges(hard)
        assert len(node_attr(soft, "pos")[0]) == 3

    def test_geographical_threshold_rule(self):
        theta = 30
        g = ec.Graph().add_geographical_threshold_graph(n=40, theta=theta, seed=9)
        pos, weight = node_attr(g, "pos"), node_attr(g, "weight")
        expected = {
            frozenset((i, j))
            for i in range(40)
            for j in range(i + 1, 40)
            if (weight[i] + weight[j]) * math.dist(pos[i], pos[j]) ** -2 >= theta
        }
        assert undirected_edges(g) == expected
        assert all(w > 0 for w in weight)

    def test_geographical_threshold_custom(self):
        g = ec.Graph().add_geographical_threshold_graph(
            n=20, theta=0.0, p_dist=lambda d: 1.0, pos_attr=None, node_weight_attr="w", seed=1
        )
        assert g.count_edges() == 190
        assert g.get_node_attrs("pos") == {} and len(g.get_node_attrs("w")) == 20

    def test_geometric_invalid(self):
        with pytest.raises(ValueError, match="radius"):
            ec.Graph().add_soft_random_geometric_graph(n=5, radius=-0.1)
        with pytest.raises(ValueError, match="p_norm"):
            ec.Graph().add_soft_random_geometric_graph(n=5, radius=0.1, p_norm=0)
        with pytest.raises(ValueError, match="dim"):
            ec.Graph().add_geographical_threshold_graph(n=5, theta=1, dim=0)


class TestNavigableSmallWorld:
    def test_lattice_and_contacts(self):
        n = 6
        g = ec.Graph(directed=True).add_navigable_small_world_graph(n=n, q=2, seed=2)
        pos = node_attr(g, "pos")
        assert g.count_nodes() == n * n
        assert pos[: n + 1] == [[0, j] for j in range(n)] + [[1, 0]]
        pairs = edge_pairs(g)
        assert all(s != t for s, t in pairs) and len(set(pairs)) == len(pairs)
        lattice = {
            (u, v)
            for u in range(n * n)
            for v in range(n * n)
            if sum(abs(a - b) for a, b in zip(pos[u], pos[v])) == 1
        }
        assert lattice <= set(pairs)
        # every node has its q contacts, unless they repeat a link
        assert len(pairs) <= len(lattice) + 2 * n * n

    def test_undirected_merges_directions(self):
        g = ec.Graph().add_navigable_small_world_graph(n=5, seed=1)
        pairs = [frozenset(p) for p in edge_pairs(g)]
        assert len(set(pairs)) == len(pairs)
        assert g.count_edges() >= 2 * 5 * 4

    def test_dimensions_and_radius(self):
        g = ec.Graph().add_navigable_small_world_graph(n=3, p=2, q=0, dim=3, seed=1)
        assert g.count_nodes() == 27
        assert all(len(c) == 3 for c in node_attr(g, "pos"))

    @pytest.mark.parametrize(
        "kwargs", [dict(n=3, p=0), dict(n=3, r=-1.0), dict(n=3, dim=0), dict(n=1001)]
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            ec.Graph().add_navigable_small_world_graph(**kwargs)
