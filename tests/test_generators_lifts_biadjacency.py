"""Tests for the generators and constructors added in NetworkX 3.6/3.7: generalized Petersen
graphs, the Shrikhande graph, random k-lifts and `from_biadjacency_matrix()`.

The installed NetworkX may be older and lack them, so the references below re-implement
NetworkX 3.7's definitions or check invariants.
"""

import itertools
import math
from collections import Counter

import edgecase as ec
import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

# =============================================================================
# Helpers
# =============================================================================


def _as_nx(g, multi=False):
    H = (nx.MultiDiGraph if g.is_directed() else nx.MultiGraph)() if multi else nx.Graph()
    H.add_nodes_from(g.get_node_ids(limit=None))
    H.add_edges_from((s, t) for _, s, t in g.get_edges_as_tuples())
    return H


def _ref_generalized_petersen(n, k):
    """NetworkX 3.7's generalized_petersen_graph(n, k), as a set of edges on 0..2n-1."""
    edges = set()
    for i in range(n):
        edges.add(frozenset((i, (i + 1) % n)))
        edges.add(frozenset((i, n + i)))
        edges.add(frozenset((n + i, n + (i + k) % n)))
    return edges


# =============================================================================
# Generalized Petersen graphs
# =============================================================================


class TestGeneralizedPetersen:
    def test_matches_reference(self):
        for n in range(3, 16):
            for k in range(1, n // 2 + 1):
                g = ec.Graph().add_generalized_petersen_graph(n, k)
                ours = [frozenset((s - 1, t - 1)) for _, s, t in g.get_edges_as_tuples()]
                assert len(set(ours)) == len(ours)
                assert set(ours) == _ref_generalized_petersen(n, k), (n, k)
                assert g.count_nodes() == 2 * n

    @pytest.mark.parametrize(
        "n, k, name",
        [(5, 2, "petersen"), (8, 3, "moebius_kantor"), (10, 2, "dodecahedral"),
         (10, 3, "desargues")],
    )
    def test_named_members(self, n, k, name):
        g = ec.Graph().add_generalized_petersen_graph(n, k)
        assert nx.is_isomorphic(_as_nx(g), _as_nx(ec.Graph().add_named_graph(name)))

    def test_prism_and_half_k(self):
        # GP(n, 1) is the n-prism
        prism = ec.Graph().add_generalized_petersen_graph(6, 1)
        assert nx.is_isomorphic(_as_nx(prism), nx.circular_ladder_graph(6))
        # With k = n / 2 the inner edges pair up, so each appears once and the inner nodes
        # have degree 2
        g = ec.Graph().add_generalized_petersen_graph(8, 4)
        assert g.count_edges() == 8 + 8 + 4
        degrees = Counter(g.get_degree(nid) for nid in g.get_node_ids())
        assert degrees == {3: 8, 2: 8}

    def test_cubic_and_girth(self):
        for n, k in ((7, 2), (9, 4), (12, 5)):
            g = ec.Graph().add_generalized_petersen_graph(n, k)
            assert g.is_k_regular(3)
            assert g.count_edges() == 3 * n
        assert {
            k: ec.Graph().add_generalized_petersen_graph(10, k).girth() for k in (1, 2, 3)
        } == {1: 4, 2: 5, 3: 6}

    def test_errors(self):
        g = ec.Graph()
        with pytest.raises(ValueError, match="n must be at least 3"):
            g.add_generalized_petersen_graph(2, 1)
        for k in (0, 3, -1):
            with pytest.raises(ValueError, match="k must satisfy"):
                g.add_generalized_petersen_graph(5, k)
        assert g.count_nodes() == 0

    def test_step(self):
        g = ec.Graph().add_generalized_petersen_graph(5, 2)
        assert g.steps[-1].code == "G.add_generalized_petersen_graph(n=5, k=2)"
        assert len(g.steps) == 1


# =============================================================================
# Shrikhande graph
# =============================================================================


class TestShrikhande:
    def test_strongly_regular_16_6_2_2(self):
        H = _as_nx(ec.Graph().add_named_graph("shrikhande"))
        assert H.number_of_nodes() == 16 and H.number_of_edges() == 48
        assert {d for _, d in H.degree()} == {6}
        for u, v in itertools.combinations(H, 2):
            common = len(set(H[u]) & set(H[v]))
            assert common == 2, (u, v)

    def test_not_the_rook_graph(self):
        H = _as_nx(ec.Graph().add_named_graph("shrikhande"))
        rook = nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))
        assert not nx.is_isomorphic(H, rook)
        # The rook's graph has K4 cliques; the Shrikhande graph's largest cliques are triangles
        assert max(len(c) for c in nx.find_cliques(H)) == 3

    def test_networkx_numbering(self):
        # NetworkX 3.7 joins (x, y) to (x +- 1, y), (x, y +- 1) and (x +- 1, y +- 1) mod 4;
        # here (x, y) is node 4x + y + 1
        expected = set()
        for x, y in itertools.product(range(4), repeat=2):
            for dx, dy in ((1, 0), (0, 1), (1, 1)):
                a, b = 4 * x + y, 4 * ((x + dx) % 4) + (y + dy) % 4
                expected.add(frozenset((a + 1, b + 1)))
        g = ec.Graph().add_named_graph("shrikhande")
        assert {frozenset((s, t)) for _, s, t in g.get_edges_as_tuples()} == expected


# =============================================================================
# Random k-lifts
# =============================================================================


def _covering_map(base, lift, k):
    """Copy i of the j-th base node (in ID order) has ID j * k + i + 1."""
    ids = sorted(base.get_node_ids(limit=None))
    return {j * k + i + 1: ids[j] for j in range(len(ids)) for i in range(k)}


class TestRandomKLift:
    def test_sizes_and_degrees(self):
        base = ec.Graph().add_named_graph("petersen")
        for k in (1, 2, 3, 7):
            lift = base.random_k_lift(k, seed=k)
            assert lift.count_nodes() == 10 * k and lift.count_edges() == 15 * k
            assert lift.is_k_regular(3)

    def test_is_a_covering(self):
        base = ec.Graph().add_nodes(list("ABCDE"))
        base.add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("A", "E")])
        k = 4
        lift = base.random_k_lift(k, seed=3)
        proj = _covering_map(base, lift, k)
        base_pairs = Counter(frozenset((s, t)) for _, s, t in base.get_edges_as_tuples())
        lifted = Counter(
            frozenset((proj[s], proj[t])) for _, s, t in lift.get_edges_as_tuples()
        )
        assert lifted == Counter({e: k * c for e, c in base_pairs.items()})
        # Over each base edge, the lifted edges form a perfect matching of the copies
        for nid in lift.get_node_ids(limit=None):
            images = Counter(
                proj[t if s == nid else s]
                for _, s, t in lift.get_edges_as_tuples()
                if nid in (s, t)
            )
            v = proj[nid]
            assert images == Counter(
                t if s == v else s for _, s, t in base.get_edges_as_tuples() if v in (s, t)
            )

    def test_labels_and_attributes(self):
        base = ec.Graph(weighted=True).add_nodes(["a", "b"])
        base.add_edge("a", "b", weight=2.5)
        base.set_node_attrs("color", {base.get_id_from_label("a"): "red"})
        lift = base.random_k_lift(3, seed=0)
        assert lift.is_weighted()
        assert [lift.get_label_from_id(n) for n in sorted(lift.get_node_ids())] == [
            "(a, 0)", "(a, 1)", "(a, 2)", "(b, 0)", "(b, 1)", "(b, 2)"
        ]
        assert lift.get_node_attrs("color") == {1: "red", 2: "red", 3: "red"}
        assert lift.get_edge_df()["weight"].to_list() == [2.5] * 3
        unlabeled = ec.Graph().add_path_graph(2, label=False).random_k_lift(2, seed=0)
        assert unlabeled.get_label_from_id(4) == "(2, 1)"

    def test_directed(self):
        base = ec.Graph(directed=True).add_nodes(list("ABC"))
        base.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        lift = base.random_k_lift(5, seed=1)
        assert lift.is_directed()
        H = _as_nx(lift, multi=True)
        assert {d for _, d in H.in_degree()} == {1} and {d for _, d in H.out_degree()} == {1}
        # A lift of a directed 3-cycle is a union of directed cycles with lengths divisible by 3
        assert all(len(c) % 3 == 0 for c in nx.simple_cycles(nx.DiGraph(H)))

    def test_self_loops_and_parallel_edges(self):
        base = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "A"), ("A", "B"), ("A", "B")])
        k = 6
        lift = base.random_k_lift(k, seed=4)
        assert lift.count_edges() == 3 * k
        H = _as_nx(lift, multi=True)
        # A self-loop lifts to a 2-factor among A's copies (fixed points give loops), so every
        # copy of A has degree 4, and every copy of B has degree 2
        assert all(H.degree(n) == 4 for n in range(1, k + 1))
        assert all(H.degree(n) == 2 for n in range(k + 1, 2 * k + 1))

    def test_k_one_is_a_copy(self):
        base = ec.Graph().add_named_graph("house")
        lift = base.random_k_lift(1, seed=9)
        assert [(s, t) for _, s, t in lift.get_edges_as_tuples()] == [
            (s, t) for _, s, t in base.get_edges_as_tuples()
        ]

    def test_seed(self):
        base = ec.Graph().add_full_graph(5)
        edges = lambda g: [(s, t) for _, s, t in g.get_edges_as_tuples()]  # noqa: E731
        assert edges(base.random_k_lift(4, seed=11)) == edges(base.random_k_lift(4, seed=11))
        draws = {tuple(edges(base.random_k_lift(4, seed=s))) for s in range(10)}
        assert len(draws) > 1
        assert base.random_k_lift(4).count_edges() == 40

    def test_triangle_lifts(self):
        triangle = ec.Graph().add_cycle_graph(3)
        counts = {triangle.random_k_lift(2, seed=s).count_weakly_connected_components()
                  for s in range(40)}
        assert counts == {1, 2}

    def test_empty_and_errors(self):
        assert ec.Graph().random_k_lift(3).count_nodes() == 0
        nodes_only = ec.Graph().add_nodes(["A", "B"]).random_k_lift(3)
        assert nodes_only.count_nodes() == 6 and nodes_only.count_edges() == 0
        for k in (0, -2):
            with pytest.raises(ValueError, match="positive integer"):
                ec.Graph().add_full_graph(3).random_k_lift(k)

    def test_steps(self):
        base = ec.Graph().add_full_graph(4)
        lift = base.random_k_lift(2, seed=1)
        assert base.steps[-1].code == "G.random_k_lift(2, seed=1)"
        assert base.steps[-1].narration == "random_k_lift(2): 8 nodes and 12 edges."
        assert lift.steps[-1].code == "G.random_k_lift(2, seed=1)"
        assert base.random_k_lift(2) is not None
        assert base.steps[-1].code == "G.random_k_lift(2)"


# =============================================================================
# from_biadjacency_matrix
# =============================================================================


def _edges(g):
    return [
        (g.get_label_from_id(s), g.get_label_from_id(t)) for _, s, t in g.get_edges_as_tuples()
    ]


class TestFromBiadjacencyMatrix:
    def test_networkx_layout(self):
        # NetworkX: rows are nodes 0..r-1, columns r..r+c-1, edges in row-major order
        g = ec.from_biadjacency_matrix([[1, 0, 2], [0, 3, 1]])
        assert [g.get_label_from_id(n) for n in sorted(g.get_node_ids())] == list("01234")
        assert _edges(g) == [("0", "2"), ("0", "4"), ("1", "3"), ("1", "4")]
        assert g.get_edge_df()["weight"].to_list() == [1.0, 2.0, 3.0, 1.0]
        assert g.get_node_df()["bipartite"].to_list() == [0, 0, 1, 1, 1]
        assert not g.is_directed()

    def test_round_trip(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            r, c = rng.integers(1, 7, size=2)
            A = rng.choice([0.0, 0.0, 1.0, 2.5], size=(r, c))
            g = ec.from_biadjacency_matrix(A)
            # Isolated nodes and separate components can't be placed by bipartite_sets()
            rows, row_ids, col_ids = g.biadjacency_matrix(row_nodes=list(range(1, r + 1)))
            assert row_ids == list(range(1, r + 1))
            assert col_ids == list(range(r + 1, r + c + 1))
            np.testing.assert_array_equal(np.array(rows).reshape(r, c), A)

    def test_sparse_and_numpy_inputs(self):
        A = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0]])
        dense = ec.from_biadjacency_matrix(A)
        for fmt in ("csr", "csc", "coo", "lil"):
            sparse = ec.from_biadjacency_matrix(sp.coo_array(A).asformat(fmt))
            assert _edges(sparse) == _edges(dense)
        assert not dense.is_weighted()
        assert _edges(ec.from_biadjacency_matrix(A.tolist())) == _edges(dense)
        assert _edges(ec.from_biadjacency_matrix(A.astype(bool))) == _edges(dense)

    def test_sparse_duplicates_and_explicit_zeros(self):
        coo = sp.coo_array(([1.0, 2.0, 0.0], ([0, 0, 1], [1, 1, 0])), shape=(2, 2))
        g = ec.from_biadjacency_matrix(coo)
        assert _edges(g) == [("0", "3")]
        assert g.get_edge_df()["weight"].to_list() == [3.0]

    def test_labels(self):
        g = ec.from_biadjacency_matrix(
            [[1, 1, 0], [0, 1, 1]], row_labels=["ann", "bob"], column_labels=["x", "y", "z"]
        )
        assert _edges(g) == [("ann", "x"), ("ann", "y"), ("bob", "y"), ("bob", "z")]
        g = ec.from_biadjacency_matrix([[1]], row_labels=[10], graph_name="tiny")
        assert _edges(g) == [("10", "1")]
        with pytest.raises(ValueError, match="row_labels has 1 labels for 2 rows"):
            ec.from_biadjacency_matrix([[1], [1]], row_labels=["a"])
        with pytest.raises(ValueError, match="column_labels has 2 labels for 1 columns"):
            ec.from_biadjacency_matrix([[1], [1]], column_labels=["a", "b"])
        with pytest.raises(ValueError, match="more than once"):
            ec.from_biadjacency_matrix([[1], [1]], row_labels=["a", "b"], column_labels=["a"])
        with pytest.raises(ValueError, match="more than once"):
            # Default column labels continue the row numbering, so "2" collides
            ec.from_biadjacency_matrix([[1], [1]], row_labels=["2", "0"])

    def test_weighted_and_parallel_edges(self):
        A = [[2, 0], [1, 3]]
        assert ec.from_biadjacency_matrix(A).is_weighted()
        assert not ec.from_biadjacency_matrix(A, weighted=False).is_weighted()
        multi = ec.from_biadjacency_matrix(A, parallel_edges=True)
        assert multi.count_edges() == 6 and not multi.is_weighted()
        assert Counter(_edges(multi)) == {("0", "2"): 2, ("1", "2"): 1, ("1", "3"): 3}
        with pytest.raises(ValueError, match="whole number"):
            ec.from_biadjacency_matrix([[1.5]], parallel_edges=True)

    def test_shapes_and_empty(self):
        g = ec.from_biadjacency_matrix(np.zeros((3, 2)))
        assert g.count_nodes() == 5 and g.count_edges() == 0
        assert ec.from_biadjacency_matrix(np.zeros((0, 2))).count_nodes() == 2
        tall = ec.from_biadjacency_matrix(np.ones((5, 1)))
        assert tall.count_edges() == 5 and tall.get_degree(6) == 5

    def test_errors(self):
        with pytest.raises(ValueError, match="2-D"):
            ec.from_biadjacency_matrix([1, 2, 3])
        with pytest.raises(ValueError, match="non-finite"):
            ec.from_biadjacency_matrix([[1, math.inf]])
        with pytest.raises(ValueError, match="non-finite"):
            ec.from_biadjacency_matrix([[math.nan]])
        with pytest.raises(TypeError, match="real numbers"):
            ec.from_biadjacency_matrix([["a", "b"]])
        with pytest.raises(TypeError, match="real numbers"):
            ec.from_biadjacency_matrix(np.array([[1j]]))

    def test_step(self):
        g = ec.from_biadjacency_matrix([[1, 0], [1, 1]])
        assert len(g.steps) == 1
        assert g.steps[0].code.startswith("from_biadjacency_matrix(")
