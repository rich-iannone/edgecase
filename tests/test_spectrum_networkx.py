"""
Correctness tests for `adjacency_spectrum()` and `laplacian_spectrum()` against NetworkX.

Every graph in the corpus is built identically in NetworkX and EdgeCase. Reference
matrices are taken from NetworkX in the node order EdgeCase reports under `"nodes"`,
so these tests check eigenvalues, eigenvector/node alignment, and the matrix
conventions for weights, parallel edges, self-loops, and isolated nodes.
"""

import networkx as nx
import numpy as np
import pytest
from edgecase import Graph

TOL = 1e-8


# ============================================================================
# Graph corpus
# ============================================================================


def _with_isolates(G, count):
    G = G.copy()
    start = max(G.nodes) + 1
    G.add_nodes_from(range(start, start + count))
    return G


def _weighted(G, seed):
    rng = np.random.default_rng(seed)
    G = G.copy()
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(0.2, 3.0))
    return G


def _self_loops():
    G = nx.cycle_graph(5)
    G.add_edge(0, 0, weight=2.0)
    G.add_edge(3, 3)
    return G


def _multigraph():
    G = nx.MultiGraph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(0, 1), (0, 1), (1, 2), (2, 3), (2, 3), (2, 3), (3, 4)])
    G.add_edge(4, 0, weight=1.5)
    G.add_edge(4, 0, weight=0.5)
    return G


def _two_cliques():
    G = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(3))
    return G


UNDIRECTED = {
    "single_node": nx.empty_graph(1),
    "single_edge": nx.path_graph(2),
    "path_7": nx.path_graph(7),
    "cycle_8": nx.cycle_graph(8),
    "complete_6": nx.complete_graph(6),
    "star_6": nx.star_graph(6),
    "complete_bipartite_3_4": nx.complete_bipartite_graph(3, 4),
    "petersen": nx.petersen_graph(),
    "karate": nx.karate_club_graph(),
    "barbell_4_2": nx.barbell_graph(4, 2),
    "grid_4x3": nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 3)),
    "two_cliques": _two_cliques(),
    "only_isolates": nx.empty_graph(4),
    "path_with_isolates": _with_isolates(nx.path_graph(4), 2),
    "gnp_sparse": nx.gnp_random_graph(40, 0.06, seed=3),
    "gnp_dense_weighted": _weighted(nx.gnp_random_graph(30, 0.3, seed=7), seed=7),
    "watts_strogatz": nx.connected_watts_strogatz_graph(50, 4, 0.2, seed=2),
    "self_loops": _self_loops(),
    "multigraph": _multigraph(),
}


def _directed_self_loop():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (1, 1)])
    G[1][1]["weight"] = 2.5
    return G


def _directed_multigraph():
    G = nx.MultiDiGraph()
    G.add_edges_from([(0, 1), (0, 1), (1, 2), (2, 0), (2, 3), (3, 2)])
    return G


DIRECTED = {
    "single_node": nx.empty_graph(1, create_using=nx.DiGraph),
    "cycle_3": nx.cycle_graph(3, create_using=nx.DiGraph),
    "cycle_6": nx.cycle_graph(6, create_using=nx.DiGraph),
    "path_5": nx.path_graph(5, create_using=nx.DiGraph),
    "reciprocal_path_4": nx.DiGraph(nx.path_graph(4)),
    "star_out": nx.DiGraph([(0, i) for i in range(1, 6)]),
    "gnp": nx.gnp_random_graph(25, 0.12, seed=5, directed=True),
    "gnp_weighted": _weighted(nx.gnp_random_graph(20, 0.2, seed=11, directed=True), 11),
    "self_loop": _directed_self_loop(),
    "multidigraph": _directed_multigraph(),
    "with_isolates": _with_isolates(nx.cycle_graph(4, create_using=nx.DiGraph), 2),
}


def build(G):
    """Mirror a NetworkX graph in EdgeCase, one EdgeCase edge per NetworkX edge."""
    g = Graph(directed=G.is_directed())
    for node in G.nodes:
        g.add_node(str(node))
    edges = G.edges(data="weight", keys=False) if G.is_multigraph() else G.edges(data="weight")
    for u, v, w in edges:
        if w is None:
            g.add_edge(str(u), str(v))
        else:
            g.add_edge(str(u), str(v), weight=w)
    return g


def nx_nodelist(g, spec):
    """Translate EdgeCase's reported node order into NetworkX node keys."""
    return [int(g.get_label_from_id(node_id)) for node_id in spec["nodes"]]


def adjacency(G, nodelist):
    return nx.to_numpy_array(G, nodelist=nodelist, weight="weight")


def laplacian(G, nodelist, normalized):
    if normalized:
        return nx.normalized_laplacian_matrix(G, nodelist=nodelist, weight="weight").toarray()
    return nx.laplacian_matrix(G, nodelist=nodelist, weight="weight").toarray()


def assert_same_real_spectrum(actual, expected):
    np.testing.assert_allclose(sorted(actual), sorted(expected), atol=TOL)


def assert_same_complex_spectrum(actual, expected):
    """Compare complex eigenvalue multisets, matching each value to its nearest partner."""
    remaining = list(np.asarray(expected, dtype=complex))
    assert len(actual) == len(remaining)
    for value in actual:
        dists = [abs(value - other) for other in remaining]
        idx = int(np.argmin(dists))
        assert dists[idx] < 1e-6, f"{value} not found in reference spectrum"
        remaining.pop(idx)


def assert_eigenpairs(matrix, values, vectors):
    """Check M v = λ v for each pair and that the vectors are orthonormal."""
    V = np.array(vectors).T
    lam = np.array(values)
    assert V.shape == (matrix.shape[0], len(values))
    np.testing.assert_allclose(matrix @ V, V * lam, atol=1e-7)
    np.testing.assert_allclose(V.T @ V, np.eye(len(values)), atol=1e-7)


def _collapse_multigraph(G):
    """Sum parallel edge weights (networkx.algebraic_connectivity rejects multigraphs)."""
    H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, w in G.edges(data="weight", default=1.0):
        prev = H.get_edge_data(u, v, {"weight": 0.0})["weight"]
        H.add_edge(u, v, weight=prev + w)
    return H


undirected = pytest.mark.parametrize("name", list(UNDIRECTED))
directed = pytest.mark.parametrize("name", list(DIRECTED))


# ============================================================================
# Undirected graphs: values and vectors
# ============================================================================


class TestUndirectedAgainstNetworkX:
    @undirected
    def test_adjacency_values(self, name):
        G = UNDIRECTED[name]
        spec = build(G).adjacency_spectrum(which="all")
        assert_same_real_spectrum(spec["values"], nx.adjacency_spectrum(G).real)

    @undirected
    def test_adjacency_eigenpairs(self, name):
        G = UNDIRECTED[name]
        g = build(G)
        spec = g.adjacency_spectrum(which="all")
        A = adjacency(G, nx_nodelist(g, spec))
        assert_eigenpairs(A, spec["values"], spec["vectors"])

    @undirected
    def test_laplacian_values(self, name):
        G = UNDIRECTED[name]
        spec = build(G).laplacian_spectrum(which="all")
        assert_same_real_spectrum(spec["values"], nx.laplacian_spectrum(G))

    @undirected
    def test_laplacian_eigenpairs(self, name):
        G = UNDIRECTED[name]
        g = build(G)
        spec = g.laplacian_spectrum(which="all")
        L = laplacian(G, nx_nodelist(g, spec), normalized=False)
        assert_eigenpairs(L, spec["values"], spec["vectors"])

    @undirected
    def test_normalized_laplacian_values(self, name):
        G = UNDIRECTED[name]
        spec = build(G).laplacian_spectrum(which="all", normalized=True)
        assert_same_real_spectrum(spec["values"], nx.normalized_laplacian_spectrum(G))

    @undirected
    def test_normalized_laplacian_eigenpairs(self, name):
        G = UNDIRECTED[name]
        g = build(G)
        spec = g.laplacian_spectrum(which="all", normalized=True)
        L = laplacian(G, nx_nodelist(g, spec), normalized=True)
        assert_eigenpairs(L, spec["values"], spec["vectors"])

    @undirected
    def test_spectral_gap(self, name):
        G = UNDIRECTED[name]
        spec = build(G).adjacency_spectrum()
        ref = np.sort(nx.adjacency_spectrum(G).real)[::-1]
        if len(ref) < 2:
            assert spec["spectral_gap"] is None
        else:
            assert spec["spectral_gap"] == pytest.approx(ref[0] - ref[1], abs=TOL)

    @undirected
    def test_algebraic_connectivity(self, name):
        G = UNDIRECTED[name]
        spec = build(G).laplacian_spectrum()
        if len(G) < 2:
            assert spec["algebraic_connectivity"] is None
        elif not nx.is_connected(G):
            assert spec["algebraic_connectivity"] == pytest.approx(0.0, abs=TOL)
        else:
            H = _collapse_multigraph(G) if G.is_multigraph() else G
            expected = nx.algebraic_connectivity(H, weight="weight", method="lanczos", tol=1e-12)
            assert spec["algebraic_connectivity"] == pytest.approx(expected, abs=1e-6)


# ============================================================================
# Directed graphs: complex eigenvalues
# ============================================================================


class TestDirectedAgainstNetworkX:
    @directed
    def test_adjacency_values(self, name):
        G = DIRECTED[name]
        spec = build(G).adjacency_spectrum(which="all")
        assert all(isinstance(v, complex) for v in spec["values"])
        assert_same_complex_spectrum(spec["values"], nx.adjacency_spectrum(G))

    @directed
    def test_laplacian_values(self, name):
        G = DIRECTED[name]
        g = build(G)
        spec = g.laplacian_spectrum(which="all")
        L = laplacian(G, nx_nodelist(g, spec), normalized=False)
        assert_same_complex_spectrum(spec["values"], np.linalg.eigvals(L))

    @directed
    def test_normalized_laplacian_values(self, name):
        G = DIRECTED[name]
        g = build(G)
        spec = g.laplacian_spectrum(which="all", normalized=True)
        L = laplacian(G, nx_nodelist(g, spec), normalized=True)
        assert_same_complex_spectrum(spec["values"], np.linalg.eigvals(L))

    @directed
    def test_no_vectors_or_scalar_summaries(self, name):
        g = build(DIRECTED[name])
        spec = g.adjacency_spectrum(which="all")
        lap = g.laplacian_spectrum(which="all")
        assert spec["vectors"] is None
        assert spec["spectral_gap"] is None
        assert lap["vectors"] is None
        assert lap["algebraic_connectivity"] is None

    def test_directed_cycle_roots_of_unity(self):
        """The directed n-cycle's adjacency eigenvalues are the n-th roots of unity."""
        n = 5
        g = build(nx.cycle_graph(n, create_using=nx.DiGraph))
        values = g.adjacency_spectrum(which="all")["values"]
        roots = np.exp(2j * np.pi * np.arange(n) / n)
        assert_same_complex_spectrum(values, roots)
        # Sorted descending by real part: 1 comes first
        assert values[0] == pytest.approx(1.0 + 0j)

    def test_dag_adjacency_is_nilpotent(self):
        """A DAG's adjacency matrix is nilpotent, so every eigenvalue is zero."""
        g = build(nx.gn_graph(15, seed=4))
        for v in g.adjacency_spectrum(which="all")["values"]:
            assert abs(v) < 1e-8

    def test_directed_differs_from_undirected(self):
        """Direction must matter: a directed path is not its symmetrized version."""
        G = nx.path_graph(4, create_using=nx.DiGraph)
        directed_vals = build(G).adjacency_spectrum(which="all")["values"]
        undirected_vals = build(G.to_undirected()).adjacency_spectrum(which="all")["values"]
        assert max(abs(v) for v in directed_vals) < 1e-8
        assert max(undirected_vals) > 1.0


# ============================================================================
# Matrix conventions
# ============================================================================


class TestMatrixConventions:
    def test_parallel_edges_sum_weights(self):
        """Two parallel unit edges behave like one edge of weight 2."""
        g = Graph(directed=False)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_edge("a", "b")
        assert g.adjacency_spectrum(which="all")["values"] == pytest.approx([2.0, -2.0])
        assert g.laplacian_spectrum(which="all")["values"] == pytest.approx([0.0, 4.0], abs=TOL)

    def test_self_loops_cancel_in_laplacian(self):
        """Laplacian rows sum to zero, so λ = 0 survives self-loops."""
        g = build(_self_loops())
        values = g.laplacian_spectrum(which="all")["values"]
        assert values[0] == pytest.approx(0.0, abs=TOL)
        plain = build(nx.cycle_graph(5)).laplacian_spectrum(which="all")["values"]
        assert values == pytest.approx(plain, abs=TOL)

    def test_self_loop_counted_once_in_adjacency(self):
        """An undirected loop of weight w puts w (not 2w) on the diagonal."""
        g = Graph(directed=False)
        g.add_node("a")
        g.add_edge("a", "a", weight=3.0)
        assert g.adjacency_spectrum(which="all")["values"] == pytest.approx([3.0])

    def test_isolated_node_normalized_laplacian_zero(self):
        """Isolated nodes contribute a zero eigenvalue to the normalized Laplacian."""
        g = build(_with_isolates(nx.path_graph(2), 1))
        values = g.laplacian_spectrum(which="all", normalized=True)["values"]
        assert values == pytest.approx([0.0, 0.0, 2.0], abs=TOL)

    @undirected
    def test_zero_eigenvalue_multiplicity_is_component_count(self, name):
        G = UNDIRECTED[name]
        values = build(G).laplacian_spectrum(which="all")["values"]
        zeros = sum(1 for v in values if abs(v) < 1e-8)
        assert zeros == nx.number_connected_components(G)

    @undirected
    def test_adjacency_trace_is_loop_weight(self, name):
        G = UNDIRECTED[name]
        values = build(G).adjacency_spectrum(which="all")["values"]
        loop_weight = sum(w for u, v, w in G.edges(data="weight", default=1.0) if u == v)
        assert sum(values) == pytest.approx(loop_weight, abs=1e-7)

    @undirected
    def test_laplacian_trace_is_non_loop_degree_sum(self, name):
        G = UNDIRECTED[name]
        values = build(G).laplacian_spectrum(which="all")["values"]
        weight = sum(w for u, v, w in G.edges(data="weight", default=1.0) if u != v)
        assert sum(values) == pytest.approx(2 * weight, abs=1e-7)

    @undirected
    def test_normalized_laplacian_bounds(self, name):
        values = build(UNDIRECTED[name]).laplacian_spectrum(which="all", normalized=True)[
            "values"
        ]
        assert min(values) > -1e-9
        assert max(values) < 2.0 + 1e-9


# ============================================================================
# Node alignment
# ============================================================================


class TestNodeAlignment:
    def test_nodes_follow_creation_order(self):
        g = Graph(directed=False)
        labels = ["zeta", "alpha", "mid", "beta", "omega"]
        for label in labels:
            g.add_node(label)
        g.add_edge("zeta", "alpha")
        spec = g.adjacency_spectrum(which="all")
        assert [g.get_label_from_id(i) for i in spec["nodes"]] == labels
        lap = g.laplacian_spectrum(which="all")
        assert lap["nodes"] == spec["nodes"]

    def test_vectors_have_one_entry_per_node(self):
        g = build(nx.karate_club_graph())
        spec = g.adjacency_spectrum(which="largest", k=3)
        assert len(spec["nodes"]) == 34
        assert all(len(vec) == 34 for vec in spec["vectors"])

    def test_path_fiedler_vector_is_monotone(self):
        """On a path the Fiedler vector is monotone along the path in node order."""
        g = build(nx.path_graph(9))
        spec = g.laplacian_spectrum(which="smallest", k=2)
        labels = [int(g.get_label_from_id(i)) for i in spec["nodes"]]
        fiedler = dict(zip(labels, spec["vectors"][1]))
        seq = [fiedler[i] for i in range(9)]
        diffs = np.diff(seq)
        assert np.all(diffs < 0) or np.all(diffs > 0)

    def test_fiedler_sign_split_recovers_clusters(self):
        """Sign of the Fiedler vector separates the two halves of a barbell."""
        g = build(nx.barbell_graph(5, 0))
        spec = g.laplacian_spectrum(which="smallest", k=2)
        labels = [int(g.get_label_from_id(i)) for i in spec["nodes"]]
        positive = {lab for lab, v in zip(labels, spec["vectors"][1]) if v > 0}
        negative = {lab for lab, v in zip(labels, spec["vectors"][1]) if v < 0}
        assert {frozenset(positive), frozenset(negative)} == {
            frozenset(range(5)),
            frozenset(range(5, 10)),
        }

    def test_deleted_nodes_are_excluded(self):
        g = build(nx.path_graph(5))
        removed = g.get_id_from_label("2")
        g.delete_node("2")
        spec = g.laplacian_spectrum(which="all")
        assert removed not in spec["nodes"]
        assert len(spec["nodes"]) == 4
        # Removing the middle of P5 leaves two P2 components
        assert spec["values"] == pytest.approx([0.0, 0.0, 2.0, 2.0], abs=TOL)

    def test_eigenvector_sign_is_normalized(self):
        """The first non-negligible entry of each eigenvector is positive."""
        g = build(nx.karate_club_graph())
        for spec in (g.adjacency_spectrum(which="all"), g.laplacian_spectrum(which="all")):
            for vec in spec["vectors"]:
                first = next(v for v in vec if abs(v) > 1e-12)
                assert first > 0

    def test_results_are_deterministic(self):
        G = nx.gnp_random_graph(30, 0.2, seed=9)
        a = build(G).laplacian_spectrum(which="all")
        b = build(G).laplacian_spectrum(which="all")
        assert a["values"] == b["values"]
        assert a["vectors"] == b["vectors"]


# ============================================================================
# Selection with `which` and `k`
# ============================================================================


GRAPH_FOR_SELECTION = nx.gnp_random_graph(12, 0.3, seed=21)


class TestSelection:
    @pytest.mark.parametrize("is_directed", [False, True])
    @pytest.mark.parametrize("k", [0, 1, 3, 12, 50])
    def test_adjacency_largest_and_smallest(self, is_directed, k):
        G = GRAPH_FOR_SELECTION.to_directed() if is_directed else GRAPH_FOR_SELECTION
        if is_directed:
            G = nx.DiGraph([(u, v) for u, v in G.edges() if (u + v) % 3])
            G.add_nodes_from(range(12))
        g = build(G)
        full = g.adjacency_spectrum(which="all")["values"]
        kk = min(k, len(full))
        largest = g.adjacency_spectrum(which="largest", k=k)["values"]
        smallest = g.adjacency_spectrum(which="smallest", k=k)["values"]
        assert largest == full[:kk]
        assert smallest == (full[len(full) - kk :] if kk else [])

    @pytest.mark.parametrize("normalized", [False, True])
    @pytest.mark.parametrize("k", [0, 1, 3, 12, 50])
    def test_laplacian_largest_and_smallest(self, normalized, k):
        g = build(GRAPH_FOR_SELECTION)
        full = g.laplacian_spectrum(which="all", normalized=normalized)
        kk = min(k, len(full["values"]))
        smallest = g.laplacian_spectrum(which="smallest", k=k, normalized=normalized)
        largest = g.laplacian_spectrum(which="largest", k=k, normalized=normalized)
        assert smallest["values"] == full["values"][:kk]
        assert smallest["vectors"] == full["vectors"][:kk]
        tail = slice(len(full["values"]) - kk, None) if kk else slice(0, 0)
        assert largest["values"] == full["values"][tail]
        assert largest["vectors"] == full["vectors"][tail]

    def test_adjacency_values_descending(self):
        values = build(GRAPH_FOR_SELECTION).adjacency_spectrum(which="all")["values"]
        assert values == sorted(values, reverse=True)

    def test_laplacian_values_ascending(self):
        values = build(GRAPH_FOR_SELECTION).laplacian_spectrum(which="all")["values"]
        assert values == sorted(values)

    def test_directed_values_sorted_by_real_part(self):
        G = nx.gnp_random_graph(15, 0.2, seed=2, directed=True)
        g = build(G)
        adj = [v.real for v in g.adjacency_spectrum(which="all")["values"]]
        lap = [v.real for v in g.laplacian_spectrum(which="all")["values"]]
        assert adj == sorted(adj, reverse=True)
        assert lap == sorted(lap)

    @pytest.mark.parametrize("n, expected", [(1, 1), (4, 4), (6, 6), (10, 6)])
    def test_default_k(self, n, expected):
        g = build(nx.path_graph(n))
        assert len(g.adjacency_spectrum()["values"]) == expected
        assert len(g.laplacian_spectrum()["values"]) == expected

    def test_summaries_use_full_spectrum(self):
        """spectral_gap / algebraic_connectivity don't depend on which/k."""
        g = build(nx.karate_club_graph())
        full_adj = g.adjacency_spectrum(which="all")
        full_lap = g.laplacian_spectrum(which="all")
        for which in ("largest", "smallest"):
            assert g.adjacency_spectrum(which=which, k=1)["spectral_gap"] == full_adj["spectral_gap"]
            assert (
                g.laplacian_spectrum(which=which, k=1)["algebraic_connectivity"]
                == full_lap["algebraic_connectivity"]
            )

    @pytest.mark.parametrize("is_directed", [False, True])
    def test_invalid_which(self, is_directed):
        g = Graph(directed=is_directed)
        g.add_node("a")
        with pytest.raises(ValueError, match="which"):
            g.adjacency_spectrum(which="middle")
        with pytest.raises(ValueError, match="which"):
            g.laplacian_spectrum(which="middle")

    @pytest.mark.parametrize("is_directed", [False, True])
    def test_empty_graph_errors(self, is_directed):
        g = Graph(directed=is_directed)
        with pytest.raises(ValueError, match="empty graph"):
            g.adjacency_spectrum()
        with pytest.raises(ValueError, match="empty graph"):
            g.laplacian_spectrum()
