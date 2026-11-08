"""Tests for the classic deterministic generators, named graphs, and community benchmarks.

Structures are checked against NetworkX (edge for edge where the node numbering is meant to
match, by isomorphism otherwise); the random community generators are checked for their
planted structure, reproducibility, and argument validation.
"""

import edgecase as ec
import networkx as nx
import pytest
from edgecase.named_graphs import NAMED_GRAPHS, lcf_edges, mycielski_edges


def undirected_edges(g):
    """Edges as 0-based frozensets, assuming the graph was built from empty (IDs 1..n)."""
    return {frozenset((s - 1, t - 1)) for _, s, t in g.get_edges_as_tuples()}


def directed_edges(g):
    return {(s - 1, t - 1) for _, s, t in g.get_edges_as_tuples()}


def nx_edges(h):
    return {frozenset(e) for e in h.edges()}


def as_nx(g):
    return nx.Graph(ec.to_networkx(g))


def node_attr(g, attr):
    """`{0-based index: value}` in index order, for a graph built from empty."""
    values = g.get_node_attrs(attr)
    return {nid - 1: values[nid] for nid in sorted(values)}


# ---------------------------------------------------------------------------
# Classic generators: exact edge sets where the numbering follows NetworkX
# ---------------------------------------------------------------------------

EXACT = [
    ("complete_bipartite", lambda: ec.Graph().add_complete_bipartite_graph(3, 4),
     lambda: nx.complete_bipartite_graph(3, 4)),
    ("complete_multipartite", lambda: ec.Graph().add_complete_multipartite_graph([1, 2, 3]),
     lambda: nx.complete_multipartite_graph(1, 2, 3)),
    ("turan", lambda: ec.Graph().add_turan_graph(10, 3), lambda: nx.turan_graph(10, 3)),
    ("wheel", lambda: ec.Graph().add_wheel_graph(7), lambda: nx.wheel_graph(7)),
    ("ladder", lambda: ec.Graph().add_ladder_graph(5), lambda: nx.ladder_graph(5)),
    ("circulant", lambda: ec.Graph().add_circulant_graph(10, [1, 3, 5]),
     lambda: nx.circulant_graph(10, [1, 3, 5])),
    ("barbell", lambda: ec.Graph().add_barbell_graph(4, 3), lambda: nx.barbell_graph(4, 3)),
    ("barbell_no_bar", lambda: ec.Graph().add_barbell_graph(3, 0),
     lambda: nx.barbell_graph(3, 0)),
    ("lollipop", lambda: ec.Graph().add_lollipop_graph(4, 3), lambda: nx.lollipop_graph(4, 3)),
    ("tadpole", lambda: ec.Graph().add_tadpole_graph(5, 2), lambda: nx.tadpole_graph(5, 2)),
    ("kneser", lambda: ec.Graph().add_kneser_graph(6, 2), lambda: nx.kneser_graph(6, 2)),
    ("full_rary", lambda: ec.Graph().add_full_rary_tree(3, 20),
     lambda: nx.full_rary_tree(3, 20)),
    ("lcf", lambda: ec.Graph().add_lcf_graph(14, [5, -5], 7), lambda: nx.LCF_graph(14, [5, -5], 7)),
    ("caveman", lambda: ec.Graph().add_caveman_graph(3, 4), lambda: nx.caveman_graph(3, 4)),
    ("connected_caveman", lambda: ec.Graph().add_connected_caveman_graph(4, 5),
     lambda: nx.connected_caveman_graph(4, 5)),
    ("ring_of_cliques", lambda: ec.Graph().add_ring_of_cliques(5, 4),
     lambda: nx.ring_of_cliques(5, 4)),
]


@pytest.mark.parametrize("name,build,reference", EXACT, ids=[e[0] for e in EXACT])
def test_edges_match_networkx(name, build, reference):
    g, h = build(), reference()
    nodes = h.nodes()
    if not all(isinstance(v, int) for v in nodes):
        # Kneser labels nodes by frozenset; relabel in our subset order
        h = nx.convert_node_labels_to_integers(h, ordering="sorted")
    assert g.count_nodes() == h.number_of_nodes()
    assert g.count_edges() == h.number_of_edges()
    if name == "kneser":
        assert nx.is_isomorphic(as_nx(g), h)
    else:
        assert undirected_edges(g) == nx_edges(h)


ISOMORPHIC = [
    ("hypercube", lambda: ec.Graph().add_hypercube_graph(4), lambda: nx.hypercube_graph(4)),
    ("grid_3d", lambda: ec.Graph().add_grid_graph([2, 3, 4]), lambda: nx.grid_graph([4, 3, 2])),
    ("grid_periodic", lambda: ec.Graph().add_grid_graph([3, 4, 5], periodic=True),
     lambda: nx.grid_graph([5, 4, 3], periodic=True)),
    ("mycielski", lambda: ec.Graph().add_mycielski_graph(5), lambda: nx.mycielski_graph(5)),
    ("binomial", lambda: ec.Graph().add_binomial_tree(5), lambda: nx.binomial_tree(5)),
    ("windmill", lambda: ec.Graph().add_windmill_graph(3, 4), lambda: nx.windmill_graph(3, 4)),
]


@pytest.mark.parametrize("name,build,reference", ISOMORPHIC, ids=[e[0] for e in ISOMORPHIC])
def test_isomorphic_to_networkx(name, build, reference):
    assert nx.is_isomorphic(as_nx(build()), reference())


class TestPartiteGraphs:
    def test_bipartite_parts_and_direction(self):
        g = ec.Graph(directed=True).add_complete_bipartite_graph(2, 3)
        part = node_attr(g, "part")
        assert part == {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
        assert all(part[s] == 0 and part[t] == 1 for s, t in directed_edges(g))

    def test_part_attr_rename_and_none(self):
        g = ec.Graph().add_complete_bipartite_graph(2, 2, part_attr="side")
        assert set(node_attr(g, "side").values()) == {0, 1}
        bare = ec.Graph().add_complete_bipartite_graph(2, 2, part_attr=None)
        assert "part" not in bare.get_node_df().columns

    def test_empty_side_gives_isolated_nodes(self):
        g = ec.Graph().add_complete_bipartite_graph(3, 0)
        assert (g.count_nodes(), g.count_edges()) == (3, 0)

    def test_turan_part_sizes(self):
        g = ec.Graph().add_turan_graph(7, 3)
        counts = sorted(node_attr(g, "part").values())
        assert counts == [0, 0, 1, 1, 2, 2, 2]

    def test_turan_extremes(self):
        assert ec.Graph().add_turan_graph(5, 1).count_edges() == 0
        assert ec.Graph().add_turan_graph(5, 5).count_edges() == 10

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.add_complete_bipartite_graph(-1, 2),
            lambda g: g.add_complete_multipartite_graph([2, -1]),
            lambda g: g.add_turan_graph(0, 1),
            lambda g: g.add_turan_graph(3, 4),
            lambda g: g.add_turan_graph(3, 0),
        ],
    )
    def test_invalid(self, call):
        with pytest.raises(ValueError):
            call(ec.Graph())


class TestShapes:
    def test_wheel_hub_first_and_directed_spokes(self):
        g = ec.Graph(directed=True).add_wheel_graph(6)
        edges = directed_edges(g)
        assert {(0, i) for i in range(1, 6)} <= edges
        assert (g.get_out_degree(1), g.get_in_degree(1)) == (5, 0)
        # the rim is a directed cycle
        assert {(i, i % 5 + 1) for i in range(1, 6)} <= edges

    def test_ladder_is_grid(self):
        assert ec.Graph().add_ladder_graph(6).is_isomorphic(ec.Graph().add_grid_2d(2, 6))

    def test_ladder_single_rung(self):
        g = ec.Graph().add_ladder_graph(1)
        assert (g.count_nodes(), g.count_edges()) == (2, 1)

    def test_circulant_directed_keeps_each_offset(self):
        g = ec.Graph(directed=True).add_circulant_graph(6, [1, 5])
        assert g.count_edges() == 12
        assert directed_edges(g) == {(i, (i + o) % 6) for i in range(6) for o in (1, 5)}

    def test_circulant_undirected_dedupes(self):
        assert ec.Graph().add_circulant_graph(6, [1, 5]).count_edges() == 6
        assert ec.Graph().add_circulant_graph(6, [3]).count_edges() == 3

    def test_lollipop_stick_points_away(self):
        g = ec.Graph(directed=True).add_lollipop_graph(3, 2)
        assert {(2, 3), (3, 4)} <= directed_edges(g)

    def test_tadpole_zero_tail_is_cycle(self):
        g = ec.Graph().add_tadpole_graph(5, 0)
        assert g.is_isomorphic(ec.Graph().add_cycle_graph(5))

    def test_windmill_friendship(self):
        g = ec.Graph().add_windmill_graph(4, 3)
        h = as_nx(g)
        # every pair of nodes has exactly one common neighbor
        for u in h:
            for v in h:
                if u < v:
                    assert len(set(h[u]) & set(h[v])) == 1

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.add_wheel_graph(3),
            lambda g: g.add_ladder_graph(0),
            lambda g: g.add_circulant_graph(1, [1]),
            lambda g: g.add_circulant_graph(5, []),
            lambda g: g.add_circulant_graph(5, [5]),
            lambda g: g.add_barbell_graph(1, 2),
            lambda g: g.add_barbell_graph(3, -1),
            lambda g: g.add_lollipop_graph(1, 2),
            lambda g: g.add_lollipop_graph(3, -1),
            lambda g: g.add_tadpole_graph(2, 1),
            lambda g: g.add_tadpole_graph(3, -1),
            lambda g: g.add_windmill_graph(1, 3),
            lambda g: g.add_windmill_graph(3, 1),
        ],
    )
    def test_invalid(self, call):
        with pytest.raises(ValueError):
            call(ec.Graph())


class TestLattices:
    def test_hypercube_labels_and_distance(self):
        g = ec.Graph().add_hypercube_graph(4)
        assert g.get_label_from_id(1) == "0000"
        assert g.get_label_from_id(16) == "1111"
        assert g.shortest_path_length("0101", "1010") == 4

    def test_hypercube_directed_points_up(self):
        g = ec.Graph(directed=True).add_hypercube_graph(3)
        assert all(t > s and bin(s ^ t).count("1") == 1 for s, t in directed_edges(g))

    def test_grid_labels_row_major(self):
        g = ec.Graph().add_grid_graph([2, 3])
        labels = [g.get_label_from_id(nid) for nid in sorted(g.get_node_ids())]
        assert labels == ["(0,0)", "(0,1)", "(0,2)", "(1,0)", "(1,1)", "(1,2)"]

    def test_grid_matches_grid_2d(self):
        assert ec.Graph().add_grid_graph([3, 5]).is_isomorphic(ec.Graph().add_grid_2d(3, 5))

    def test_grid_periodic_per_axis(self):
        cylinder = ec.Graph().add_grid_graph([4, 3], periodic=[True, False])
        assert nx.is_isomorphic(
            as_nx(cylinder), nx.grid_graph([3, 4], periodic=[False, True])
        )

    def test_grid_periodic_short_axes_add_nothing(self):
        plain = ec.Graph().add_grid_graph([2, 2])
        wrapped = ec.Graph().add_grid_graph([2, 2], periodic=True)
        assert plain.count_edges() == wrapped.count_edges() == 4

    def test_grid_1d_periodic_is_cycle(self):
        assert ec.Graph().add_grid_graph([6], periodic=True).is_isomorphic(
            ec.Graph().add_cycle_graph(6)
        )

    @pytest.mark.parametrize("rows", [1, 2, 3, 4])
    @pytest.mark.parametrize("cols", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize(
        "method,reference",
        [
            ("add_triangular_lattice_graph", nx.triangular_lattice_graph),
            ("add_hexagonal_lattice_graph", nx.hexagonal_lattice_graph),
        ],
    )
    def test_lattice_matches_networkx(self, method, reference, rows, cols):
        g = getattr(ec.Graph(), method)(rows, cols)
        h = reference(rows, cols, with_positions=False)
        # NetworkX nodes are (x, y); EdgeCase labels are "(row,col)" = "(y,x)"
        labels = {f"({y},{x})": (x, y) for x, y in h.nodes()}
        assert sorted(g.get_label_from_id(nid) for nid in g.get_node_ids()) == sorted(labels)
        ours = {
            frozenset(labels[g.get_label_from_id(n)] for n in (s, t))
            for _, s, t in g.get_edges_as_tuples()
        }
        assert ours == nx_edges(h)
        assert g.count_edges() == h.number_of_edges()

    @pytest.mark.parametrize("rows,cols", [(1, 1), (2, 4), (3, 5), (4, 8)])
    def test_triangular_faces(self, rows, cols):
        g = ec.Graph().add_triangular_lattice_graph(rows, cols)
        assert len(g.list_triangles()) == rows * cols
        assert g.is_planar()

    def test_triangular_interior_degree_six(self):
        g = ec.Graph().add_triangular_lattice_graph(4, 8)
        assert g.get_degree("(2,2)") == 6

    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (3, 4), (4, 5)])
    def test_hexagonal_honeycomb(self, rows, cols):
        g = ec.Graph().add_hexagonal_lattice_graph(rows, cols)
        assert g.count_nodes() == 2 * (rows * cols + rows + cols)
        assert g.is_bipartite() and g.is_planar()
        assert set(g.get_node_df()["degree"]) <= {2, 3}
        assert g.girth() == 6
        # Euler: one bounded face per hexagon
        assert g.count_edges() - g.count_nodes() + 1 == rows * cols

    def test_lattice_edges_point_forward(self):
        for method in ("add_triangular_lattice_graph", "add_hexagonal_lattice_graph"):
            g = getattr(ec.Graph(directed=True), method)(3, 4)
            assert all(s < t for _, s, t in g.get_edges_as_tuples())

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.add_hypercube_graph(0),
            lambda g: g.add_grid_graph([]),
            lambda g: g.add_grid_graph([3, 0]),
            lambda g: g.add_grid_graph([3, 3], periodic=[True]),
            lambda g: g.add_triangular_lattice_graph(0, 3),
            lambda g: g.add_hexagonal_lattice_graph(3, 0),
        ],
    )
    def test_invalid(self, call):
        with pytest.raises(ValueError):
            call(ec.Graph())


class TestCombinatorial:
    def test_kneser_petersen(self):
        assert ec.Graph().add_kneser_graph(5, 2).is_isomorphic(
            ec.Graph().add_named_graph("petersen")
        )

    def test_kneser_labels(self):
        g = ec.Graph().add_kneser_graph(4, 2)
        labels = [g.get_label_from_id(nid) for nid in sorted(g.get_node_ids())]
        assert labels == ["0,1", "0,2", "0,3", "1,2", "1,3", "2,3"]

    def test_kneser_no_edges_when_n_below_2k(self):
        assert ec.Graph().add_kneser_graph(5, 3).count_edges() == 0

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_mycielski_sizes(self, n):
        size, edges = mycielski_edges(n)
        h = nx.mycielski_graph(n)
        assert size == h.number_of_nodes()
        assert len(edges) == h.number_of_edges()

    def test_mycielski_triangle_free(self):
        g = ec.Graph().add_mycielski_graph(5)
        assert g.list_triangles() == []

    def test_binomial_tree_parents(self):
        g = ec.Graph(directed=True).add_binomial_tree(3)
        assert directed_edges(g) == {
            (0, 1), (0, 2), (2, 3), (0, 4), (4, 5), (4, 6), (6, 7)
        }
        assert undirected_edges(g) == nx_edges(nx.binomial_tree(3))

    def test_binomial_tree_order_zero(self):
        g = ec.Graph().add_binomial_tree(0)
        assert (g.count_nodes(), g.count_edges()) == (1, 0)

    def test_full_rary_r1_is_path(self):
        assert ec.Graph().add_full_rary_tree(1, 5).is_isomorphic(ec.Graph().add_path_graph(5))

    def test_lcf_skips_duplicates(self):
        # shift 1 duplicates every cycle edge, so only the cycle remains
        assert lcf_edges(5, [1], 5) == [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]

    @pytest.mark.parametrize(
        "call",
        [
            lambda g: g.add_kneser_graph(0, 1),
            lambda g: g.add_kneser_graph(3, 4),
            lambda g: g.add_mycielski_graph(0),
            lambda g: g.add_binomial_tree(-1),
            lambda g: g.add_full_rary_tree(0, 3),
            lambda g: g.add_full_rary_tree(2, 0),
            lambda g: g.add_lcf_graph(2, [1]),
            lambda g: g.add_lcf_graph(6, [3], repeats=-1),
        ],
    )
    def test_invalid(self, call):
        with pytest.raises(ValueError):
            call(ec.Graph())


# ---------------------------------------------------------------------------
# Named graphs
# ---------------------------------------------------------------------------

NX_NAMED = {
    "bull": nx.bull_graph,
    "butterfly": lambda: nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)]),
    "chvatal": nx.chvatal_graph,
    "cubical": nx.cubical_graph,
    "desargues": nx.desargues_graph,
    "diamond": nx.diamond_graph,
    "dodecahedral": nx.dodecahedral_graph,
    "frucht": nx.frucht_graph,
    "heawood": nx.heawood_graph,
    "hoffman_singleton": nx.hoffman_singleton_graph,
    "house": nx.house_graph,
    "house_x": nx.house_x_graph,
    "icosahedral": nx.icosahedral_graph,
    "krackhardt_kite": nx.krackhardt_kite_graph,
    "moebius_kantor": nx.moebius_kantor_graph,
    "octahedral": nx.octahedral_graph,
    "pappus": nx.pappus_graph,
    "petersen": nx.petersen_graph,
    "sedgewick_maze": nx.sedgewick_maze_graph,
    "tetrahedral": nx.tetrahedral_graph,
    "truncated_cube": nx.truncated_cube_graph,
    "truncated_tetrahedron": nx.truncated_tetrahedron_graph,
    "tutte": nx.tutte_graph,
}

# graphs NetworkX doesn't provide, checked by order, size, regularity, and girth
INVARIANTS = {
    "franklin": (12, 18, 3, 4),
    "grotzsch": (11, 20, None, 4),
    "mcgee": (24, 36, 3, 7),
    "nauru": (24, 36, 3, 6),
    "shrikhande": (16, 48, 6, 3),
    "tutte_coxeter": (30, 45, 3, 8),
}


def test_registry_is_covered():
    assert set(NAMED_GRAPHS) == set(NX_NAMED) | set(INVARIANTS)


@pytest.mark.parametrize("name", sorted(NX_NAMED))
def test_named_graph_isomorphic_to_networkx(name):
    g = ec.Graph().add_named_graph(name)
    assert nx.is_isomorphic(as_nx(g), NX_NAMED[name]())


@pytest.mark.parametrize("name", sorted(INVARIANTS))
def test_named_graph_invariants(name):
    n, m, degree, girth = INVARIANTS[name]
    h = as_nx(ec.Graph().add_named_graph(name))
    assert (h.number_of_nodes(), h.number_of_edges(), nx.girth(h)) == (n, m, girth)
    if degree is not None:
        assert {d for _, d in h.degree()} == {degree}


def test_named_graph_registry_entries_consistent():
    for name, spec in NAMED_GRAPHS.items():
        assert spec.description
        assert all(0 <= a < b < spec.n for a, b in spec.edges), name
        assert len(set(spec.edges)) == len(spec.edges), name


def test_named_graph_name_normalization():
    for name in ("Krackhardt Kite", "krackhardt-kite", "  KRACKHARDT_KITE "):
        assert ec.Graph().add_named_graph(name).count_edges() == 18


def test_named_graph_unknown_lists_names():
    with pytest.raises(ValueError, match="petersen"):
        ec.Graph().add_named_graph("nope")


def test_named_graph_appends_and_records_one_step():
    g = ec.Graph().add_path_graph(3).add_named_graph("petersen", type="p", rel="r")
    assert g.count_nodes() == 13
    assert [s.method for s in g._steps] == ["add_path_graph", "add_named_graph"]
    assert set(g.get_node_attrs("type").values()) == {"p"}
    new_ids = set(range(4, 14))
    for _, s, t in g.get_edges_as_tuples():
        assert (s in new_ids) == (t in new_ids)


# ---------------------------------------------------------------------------
# Shared conventions across all new generators
# ---------------------------------------------------------------------------

ALL_CALLS = {
    "add_complete_bipartite_graph": dict(n1=2, n2=3),
    "add_complete_multipartite_graph": dict(sizes=[1, 2]),
    "add_turan_graph": dict(n=5, r=2),
    "add_wheel_graph": dict(n=5),
    "add_ladder_graph": dict(n=3),
    "add_circulant_graph": dict(n=5, offsets=[1, 2]),
    "add_barbell_graph": dict(m1=3, m2=1),
    "add_lollipop_graph": dict(m=3, n=2),
    "add_tadpole_graph": dict(m=3, n=2),
    "add_windmill_graph": dict(n=2, k=3),
    "add_hypercube_graph": dict(d=2),
    "add_grid_graph": dict(dims=[2, 2]),
    "add_triangular_lattice_graph": dict(rows=2, cols=2),
    "add_hexagonal_lattice_graph": dict(rows=2, cols=3),
    "add_kneser_graph": dict(n=4, k=1),
    "add_mycielski_graph": dict(n=3),
    "add_binomial_tree": dict(order=2),
    "add_full_rary_tree": dict(r=2, n=4),
    "add_lcf_graph": dict(n=6, shifts=[3], repeats=6),
    "add_generalized_petersen_graph": dict(n=5, k=2),
    "add_named_graph": dict(name="house"),
    "add_caveman_graph": dict(num_cliques=2, clique_size=3),
    "add_connected_caveman_graph": dict(num_cliques=2, clique_size=3),
    "add_ring_of_cliques": dict(num_cliques=2, clique_size=3),
    "add_relaxed_caveman_graph": dict(num_cliques=2, clique_size=3, p=0.3, seed=1),
    "add_planted_partition_graph": dict(num_groups=2, group_size=3, p_in=0.8, p_out=0.2, seed=1),
    "add_random_partition_graph": dict(sizes=[2, 3], p_in=0.8, p_out=0.2, seed=1),
    "add_gaussian_random_partition_graph": dict(
        n=10, mean_size=3, shape=2, p_in=0.8, p_out=0.2, seed=1
    ),
    "add_lfr_graph": dict(n=60, mu=0.2, average_degree=5, seed=1),
}


@pytest.mark.parametrize("method", sorted(ALL_CALLS))
def test_conventions(method):
    kwargs = ALL_CALLS[method]
    g = ec.Graph().add_node("existing")
    result = getattr(g, method)(type="t", rel="r", **kwargs)
    assert result is g
    assert [s.method for s in g._steps][-1] == method
    assert len(g._steps) == 2
    node_types = g.get_node_attrs("type")
    assert 1 not in node_types and set(node_types.values()) == {"t"}
    assert g.get_edge_attrs("rel") and set(g.get_edge_attrs("rel").values()) == {"r"}
    # nothing touches the existing node
    assert g.get_degree(1) == 0


@pytest.mark.parametrize("method", sorted(ALL_CALLS))
def test_unlabelled(method):
    g = getattr(ec.Graph(), method)(label=False, **ALL_CALLS[method])
    assert all(g.get_label_from_id(nid) is None for nid in g.get_node_ids())


@pytest.mark.parametrize(
    "method", ["add_hypercube_graph", "add_grid_graph", "add_triangular_lattice_graph",
               "add_hexagonal_lattice_graph", "add_kneser_graph"]
)
def test_fixed_labels_clash_before_adding(method):
    g = getattr(ec.Graph(), method)(**ALL_CALLS[method])
    before = g.count_nodes()
    with pytest.raises(ValueError, match="already exist"):
        getattr(g, method)(**ALL_CALLS[method])
    assert g.count_nodes() == before
    getattr(g, method)(label=False, **ALL_CALLS[method])
    assert g.count_nodes() == 2 * before


@pytest.mark.parametrize("method", sorted(ALL_CALLS))
def test_simple_graph(method):
    for directed in (False, True):
        g = getattr(ec.Graph(directed=directed), method)(**ALL_CALLS[method])
        pairs = [(s, t) for _, s, t in g.get_edges_as_tuples()]
        assert all(s != t for s, t in pairs)
        keys = pairs if directed else [frozenset(p) for p in pairs]
        assert len(set(keys)) == len(keys)


# ---------------------------------------------------------------------------
# Community benchmarks
# ---------------------------------------------------------------------------

COMMUNITY = [m for m in ALL_CALLS if m in {
    "add_caveman_graph", "add_connected_caveman_graph", "add_ring_of_cliques",
    "add_relaxed_caveman_graph", "add_planted_partition_graph", "add_random_partition_graph",
    "add_gaussian_random_partition_graph", "add_lfr_graph",
}]
RANDOM = [m for m in COMMUNITY if "seed" in ALL_CALLS[m]]


@pytest.mark.parametrize("method", COMMUNITY)
def test_community_attr(method):
    g = getattr(ec.Graph(), method)(**ALL_CALLS[method])
    groups = list(node_attr(g, "community").values())
    assert len(groups) == g.count_nodes()
    # 0-based and contiguous, created community by community
    assert groups == sorted(groups) and set(groups) == set(range(max(groups) + 1))
    renamed = getattr(ec.Graph(), method)(community_attr="block", **ALL_CALLS[method])
    assert list(node_attr(renamed, "block").values()) == groups
    bare = getattr(ec.Graph(), method)(community_attr=None, **ALL_CALLS[method])
    assert "community" not in bare.get_node_df().columns


@pytest.mark.parametrize("method", RANDOM)
def test_seed_reproducible(method):
    kwargs = ALL_CALLS[method]
    a = getattr(ec.Graph(), method)(**kwargs)
    b = getattr(ec.Graph(), method)(**kwargs)
    assert a.get_edges_as_tuples() == b.get_edges_as_tuples()


def test_relaxed_caveman_keeps_edge_count_and_p0_is_caveman():
    g = ec.Graph().add_relaxed_caveman_graph(6, 5, 0.5, seed=3)
    assert g.count_edges() == 6 * 10
    p0 = ec.Graph().add_relaxed_caveman_graph(3, 4, 0.0, seed=3)
    assert undirected_edges(p0) == nx_edges(nx.caveman_graph(3, 4))


def test_relaxed_caveman_rewiring_rate():
    g = ec.Graph().add_relaxed_caveman_graph(50, 6, 0.3, seed=11)
    coverage, _ = g.partition_quality("community")
    # an edge leaves its cave with probability about p * (1 - k/n)
    assert 0.65 < coverage < 0.76


def test_planted_partition_densities():
    g = ec.Graph().add_planted_partition_graph(4, 50, 0.3, 0.02, seed=2)
    groups = node_attr(g, "community")
    inside = sum(groups[s - 1] == groups[t - 1] for _, s, t in g.get_edges_as_tuples())
    outside = g.count_edges() - inside
    assert abs(inside / (4 * 50 * 49 / 2) - 0.3) < 0.03
    assert abs(outside / (6 * 50 * 50) - 0.02) < 0.005


def test_planted_partition_extremes():
    assert ec.Graph().add_planted_partition_graph(3, 4, 1.0, 0.0, seed=1).is_isomorphic(
        ec.Graph().add_caveman_graph(3, 4)
    )
    assert ec.Graph().add_planted_partition_graph(3, 4, 0.0, 0.0, seed=1).count_edges() == 0


def test_random_partition_directed_draws_ordered_pairs():
    g = ec.Graph(directed=True).add_random_partition_graph([30, 30], 1.0, 0.0, seed=1)
    assert g.count_edges() == 2 * 30 * 29


def test_random_partition_sizes():
    g = ec.Graph().add_random_partition_graph([3, 0, 5], 0.5, 0.1, seed=1)
    groups = list(node_attr(g, "community").values())
    assert groups == [0, 0, 0, 2, 2, 2, 2, 2]


def test_gaussian_partition_sizes_sum_to_n():
    g = ec.Graph().add_gaussian_random_partition_graph(500, 25, 5, 0.2, 0.01, seed=9)
    groups = list(node_attr(g, "community").values())
    assert len(groups) == 500
    sizes = [groups.count(c) for c in set(groups)]
    assert all(s >= 1 for s in sizes)
    assert 15 <= len(sizes) <= 25


def test_lfr_properties():
    g = ec.Graph().add_lfr_graph(1000, 0.3, 20, max_degree=50, seed=1)
    degrees = g.get_node_df()["degree"]
    assert abs(degrees.mean() - 20) < 1.5
    assert degrees.max() <= 50
    groups = node_attr(g, "community")
    edges = g.get_edges_as_tuples()
    external = sum(groups[s - 1] != groups[t - 1] for _, s, t in edges)
    assert abs(external / len(edges) - 0.3) < 0.03
    # each community is internally connected enough to be found
    assert g.modularity("community") > 0.5


def test_lfr_default_max_degree():
    g = ec.Graph().add_lfr_graph(500, 0.2, 10, seed=4)
    assert max(g.get_node_df()["degree"]) <= 25


def test_lfr_community_size_bounds():
    g = ec.Graph().add_lfr_graph(
        600, 0.2, 10, max_degree=30, min_community=30, max_community=60, seed=5
    )
    groups = list(node_attr(g, "community").values())
    sizes = [groups.count(c) for c in set(groups)]
    assert min(sizes) >= 30 and max(sizes) <= 60


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=100, mu=1.5, average_degree=5),
        dict(n=100, mu=0.2, average_degree=5, tau1=-1),
        dict(n=1, mu=0.2, average_degree=5),
        dict(n=100, mu=0.2, average_degree=5, max_degree=200),
        dict(n=100, mu=0.2, average_degree=50, max_degree=20),
        dict(n=100, mu=0.2, average_degree=5, min_community=50, max_community=20),
    ],
)
def test_lfr_invalid(kwargs):
    with pytest.raises(ValueError):
        ec.Graph().add_lfr_graph(seed=1, **kwargs)


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.add_caveman_graph(0, 3),
        lambda g: g.add_connected_caveman_graph(1, 4),
        lambda g: g.add_connected_caveman_graph(3, 2),
        lambda g: g.add_ring_of_cliques(1, 3),
        lambda g: g.add_ring_of_cliques(3, 1),
        lambda g: g.add_relaxed_caveman_graph(2, 3, 1.5),
        lambda g: g.add_planted_partition_graph(0, 3, 0.5, 0.1),
        lambda g: g.add_planted_partition_graph(2, 3, -0.1, 0.1),
        lambda g: g.add_random_partition_graph([2, -1], 0.5, 0.1),
        lambda g: g.add_random_partition_graph([2, 2], 0.5, 1.1),
        lambda g: g.add_gaussian_random_partition_graph(10, 0.5, 1, 0.5, 0.1),
        lambda g: g.add_gaussian_random_partition_graph(10, 3, 0, 0.5, 0.1),
        lambda g: g.add_gaussian_random_partition_graph(-1, 3, 1, 0.5, 0.1),
    ],
)
def test_community_invalid(call):
    g = ec.Graph()
    with pytest.raises(ValueError):
        call(g)
    assert g.count_nodes() == 0
