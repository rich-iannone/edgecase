"""
Tests for the sparse solver used by `adjacency_spectrum()` / `laplacian_spectrum()`.

For undirected graphs with at least 400 nodes, `which="largest"`/`"smallest"` with a
small `k` runs Chebyshev-filtered subspace iteration instead of a dense
eigendecomposition. Dense NumPy results are the reference: `scipy.sparse.linalg.eigsh`
misses copies of repeated eigenvalues on several of these graphs.
"""

import functools

import networkx as nx
import numpy as np
import pytest
import scipy.sparse.linalg as sla
from edgecase import Graph

TOL = 1e-8


def _isolates_and_path():
    G = nx.empty_graph(400)
    return nx.disjoint_union(G, nx.path_graph(400))


GRAPHS = {
    "watts_strogatz": lambda: nx.connected_watts_strogatz_graph(1200, 6, 0.1, seed=1),
    "barabasi_albert": lambda: nx.barabasi_albert_graph(1500, 3, seed=1),
    "powerlaw_cluster": lambda: nx.powerlaw_cluster_graph(1000, 3, 0.3, seed=2),
    "grid_30x30": lambda: nx.convert_node_labels_to_integers(nx.grid_2d_graph(30, 30)),
    "path_800": lambda: nx.path_graph(800),
    "star_800": lambda: nx.star_graph(800),
    "clique_union_60x10": lambda: nx.disjoint_union_all([nx.complete_graph(10)] * 60),
    "isolates_and_path": _isolates_and_path,
    "gnp_with_isolates": lambda: nx.gnp_random_graph(1200, 0.002, seed=4),
}

KINDS = ["adjacency", "laplacian", "normalized"]


@functools.lru_cache(maxsize=None)
def graph(name):
    G = GRAPHS[name]()
    g = Graph(directed=False)
    for node in G.nodes:
        g.add_node(str(node))
    for u, v in G.edges():
        g.add_edge(str(u), str(v))
    return G, g


def spectrum(g, kind, **kwargs):
    if kind == "adjacency":
        return g.adjacency_spectrum(**kwargs)
    return g.laplacian_spectrum(normalized=(kind == "normalized"), **kwargs)


def reference_matrix(G, g, spec, kind):
    nodelist = [int(g.get_label_from_id(i)) for i in spec["nodes"]]
    if kind == "adjacency":
        return nx.to_numpy_array(G, nodelist=nodelist)
    if kind == "laplacian":
        return nx.laplacian_matrix(G, nodelist=nodelist).toarray().astype(float)
    return nx.normalized_laplacian_matrix(G, nodelist=nodelist).toarray()


@functools.lru_cache(maxsize=None)
def dense_eigenvalues(name, kind):
    G, g = graph(name)
    spec = spectrum(g, kind, which="largest", k=1)
    return np.linalg.eigvalsh(reference_matrix(G, g, spec, kind))


def assert_eigenpairs(M, values, vectors, scale):
    V = np.array(vectors).T
    lam = np.array(values)
    residual = np.linalg.norm(M @ V - V * lam, axis=0).max()
    assert residual < 1e-9 * max(scale, 1.0)
    np.testing.assert_allclose(V.T @ V, np.eye(len(values)), atol=1e-9)


params = pytest.mark.parametrize("name, kind", [(name, kind) for name in GRAPHS for kind in KINDS])


class TestAgainstDense:
    @params
    @pytest.mark.parametrize("which", ["largest", "smallest"])
    def test_values_and_vectors(self, name, kind, which):
        G, g = graph(name)
        spec = spectrum(g, kind, which=which, k=6)
        ref = dense_eigenvalues(name, kind)
        expected = ref[-6:] if which == "largest" else ref[:6]
        scale = np.abs(ref).max()

        np.testing.assert_allclose(sorted(spec["values"]), expected, atol=TOL * max(scale, 1.0))
        M = reference_matrix(G, g, spec, kind)
        assert_eigenpairs(M, spec["values"], spec["vectors"], scale)

    @params
    def test_output_order(self, name, kind):
        _, g = graph(name)
        for which in ("largest", "smallest"):
            values = spectrum(g, kind, which=which, k=6)["values"]
            descending = kind == "adjacency"
            assert values == sorted(values, reverse=descending)

    @params
    @pytest.mark.parametrize("which", ["largest", "smallest"])
    def test_scalar_summaries(self, name, kind, which):
        _, g = graph(name)
        ref = dense_eigenvalues(name, kind)
        tol = TOL * max(np.abs(ref).max(), 1.0)
        spec = spectrum(g, kind, which=which, k=3)
        if kind == "adjacency":
            assert spec["spectral_gap"] == pytest.approx(ref[-1] - ref[-2], abs=tol)
        else:
            assert spec["algebraic_connectivity"] == pytest.approx(ref[1], abs=tol)

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_small_k_still_reports_summaries(self, k):
        _, g = graph("watts_strogatz")
        ref = dense_eigenvalues("watts_strogatz", "laplacian")
        spec = g.laplacian_spectrum(which="largest", k=k)
        assert len(spec["values"]) == len(spec["vectors"]) == k
        assert spec["algebraic_connectivity"] == pytest.approx(ref[1], abs=1e-8)

    def test_matches_dense_path_on_all(self):
        """`which="all"` (dense) and the sparse path agree on the extremal values."""
        _, g = graph("powerlaw_cluster")
        full = g.laplacian_spectrum(which="all")["values"]
        sparse = g.laplacian_spectrum(which="smallest", k=6)["values"]
        np.testing.assert_allclose(sparse, full[:6], atol=1e-8)


class TestRepeatedEigenvalues:
    """Degenerate spectra, where single-vector Lanczos can miss copies."""

    def test_clique_union_eigenvalue_multiplicities(self):
        # 60 x K_10: adjacency {9 (x60), -1 (x540)}, Laplacian {0 (x60), 10 (x540)}
        _, g = graph("clique_union_60x10")
        assert g.adjacency_spectrum(which="largest", k=8)["values"] == pytest.approx([9.0] * 8)
        assert g.adjacency_spectrum(which="smallest", k=8)["values"] == pytest.approx([-1.0] * 8)
        lap_small = g.laplacian_spectrum(which="smallest", k=8)["values"]
        assert lap_small == pytest.approx([0.0] * 8, abs=1e-9)
        lap_large = g.laplacian_spectrum(which="largest", k=8)["values"]
        assert lap_large == pytest.approx([10.0] * 8)

    def test_null_space_larger_than_block(self):
        # 400 isolates + a path: λ = 0 has multiplicity 401, next is ~1.5e-5
        _, g = graph("isolates_and_path")
        spec = g.laplacian_spectrum(which="smallest", k=6)
        assert spec["values"] == pytest.approx([0.0] * 6, abs=1e-9)
        assert spec["algebraic_connectivity"] == pytest.approx(0.0, abs=1e-9)

    def test_bipartite_components_normalized_top(self):
        # Every bipartite component with an edge contributes a normalized eigenvalue 2
        G, g = graph("gnp_with_isolates")
        bipartite = sum(
            1
            for comp in nx.connected_components(G)
            if len(comp) > 1 and nx.is_bipartite(G.subgraph(comp))
        )
        assert bipartite >= 6
        values = g.laplacian_spectrum(which="largest", k=6, normalized=True)["values"]
        assert values == pytest.approx([2.0] * 6)


class TestLargeGraphs:
    """Graphs far too large for the dense solver's O(n²) memory in reasonable time."""

    @pytest.fixture(scope="class")
    def big(self):
        G = nx.connected_watts_strogatz_graph(20000, 6, 0.1, seed=7)
        g = Graph(directed=False)
        for node in G.nodes:
            g.add_node(str(node))
        for u, v in G.edges():
            g.add_edge(str(u), str(v))
        return G, g

    @pytest.mark.parametrize(
        "kind, which",
        [("adjacency", "largest"), ("laplacian", "smallest"), ("normalized", "largest")],
    )
    def test_matches_eigsh(self, big, kind, which):
        G, g = big
        spec = spectrum(g, kind, which=which, k=4)
        nodelist = [int(g.get_label_from_id(i)) for i in spec["nodes"]]
        if kind == "adjacency":
            M = nx.to_scipy_sparse_array(G, nodelist=nodelist, format="csr").astype(float)
        elif kind == "laplacian":
            M = nx.laplacian_matrix(G, nodelist=nodelist).tocsr().astype(float)
        else:
            M = nx.normalized_laplacian_matrix(G, nodelist=nodelist).tocsr()

        mode = "LA" if which == "largest" else "SA"
        ref = np.sort(sla.eigsh(M, k=4, which=mode, tol=1e-12, return_eigenvectors=False))
        np.testing.assert_allclose(sorted(spec["values"]), ref, atol=1e-8)

        V = np.array(spec["vectors"]).T
        residual = np.linalg.norm(M @ V - V * np.array(spec["values"]), axis=0).max()
        assert residual < 1e-8


class TestDeterminism:
    def test_repeated_calls_identical(self):
        _, g = graph("barabasi_albert")
        a = g.laplacian_spectrum(which="smallest", k=5)
        b = g.laplacian_spectrum(which="smallest", k=5)
        assert a["values"] == b["values"]
        assert a["vectors"] == b["vectors"]

    def test_eigenvector_signs_normalized(self):
        _, g = graph("grid_30x30")
        for vec in g.adjacency_spectrum(which="largest", k=5)["vectors"]:
            first = next(v for v in vec if abs(v) > 1e-12)
            assert first > 0
