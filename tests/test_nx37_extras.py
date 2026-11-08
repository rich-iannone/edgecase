"""Tests for the parameters and extras added in NetworkX 3.6/3.7:
`subgraph_centrality(normalized=)`, `sample_weight=` on approximate current-flow betweenness,
the spectral and greedy node-swap modularity bipartitions, Panther++ vector similarity, the
magnetic Laplacian and spectrum, and GEXF `<meta>` description and keywords.

The installed NetworkX may be older and lack them, so the references below re-implement
NetworkX 3.7's definitions or use the values from its documentation.
"""

import cmath
import itertools
import math
import random
import warnings

import edgecase as ec
import networkx as nx
import numpy as np
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _from_nx(G, directed=None, weighted=False):
    """An EdgeCase graph with node i + 1 for the i-th node of sorted(G)."""
    directed = G.is_directed() if directed is None else directed
    g = ec.Graph(directed=directed, weighted=weighted)
    nodes = sorted(G)
    g.add_nodes([str(n) for n in nodes])
    for u, v, data in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=float(data.get("weight", 1.0)))
        else:
            g.add_edge(str(u), str(v))
    return g, {n: i + 1 for i, n in enumerate(nodes)}


def _modularity(g, sides):
    H = nx.Graph()
    H.add_nodes_from(g.get_node_ids(limit=None))
    H.add_edges_from((s, t) for _, s, t in g.get_edges_as_tuples())
    return nx.community.modularity(H, sides)


# =============================================================================
# subgraph_centrality(normalized=)
# =============================================================================

ESTRADA = [(1, 2), (1, 5), (1, 8), (2, 3), (2, 8), (3, 4), (3, 6), (4, 5), (4, 7), (5, 6),
           (6, 7), (7, 8)]


class TestSubgraphCentralityNormalized:
    def _graph(self):
        g = ec.Graph().add_nodes([str(i) for i in range(1, 9)])
        return g.add_edges([(str(u), str(v)) for u, v in ESTRADA])

    def test_networkx_doc_values(self):
        g = self._graph()
        plain = g.subgraph_centrality(return_dict=True)
        norm = g.subgraph_centrality(normalized=True, return_dict=True)
        assert [f"{plain[n]:0.2f}" for n in range(1, 9)] == [
            "3.90", "3.90", "3.64", "3.71", "3.64", "3.71", "3.64", "3.90"
        ]
        assert [f"{norm[n]:0.3f}" for n in range(1, 9)] == [
            "0.194", "0.194", "0.181", "0.184", "0.181", "0.184", "0.181", "0.194"
        ]

    def test_scaled_by_exp_of_largest_eigenvalue(self):
        H = nx.gnp_random_graph(30, 0.2, seed=4)
        g, idx = _from_nx(H)
        lam = max(np.linalg.eigvalsh(nx.to_numpy_array(H, sorted(H))))
        plain = g.subgraph_centrality(return_dict=True)
        norm = g.subgraph_centrality(normalized=True, return_dict=True)
        for n in H:
            assert norm[idx[n]] == pytest.approx(plain[idx[n]] / math.exp(lam), rel=1e-9)
        # The normalized values stay finite where the plain ones overflow
        big = ec.Graph().add_full_graph(800)
        values = big.subgraph_centrality(normalized=True, return_dict=True)
        assert all(math.isfinite(v) for v in values.values())
        assert values[1] == pytest.approx(1 / 800, rel=1e-6)

    def test_steps(self):
        g = self._graph()
        g.subgraph_centrality(normalized=True)
        assert g.steps[-1].code == (
            "G.subgraph_centrality(normalized=True, attr_name='subgraph_centrality')"
        )
        g.subgraph_centrality()
        assert g.steps[-1].code == "G.subgraph_centrality(attr_name='subgraph_centrality')"


# =============================================================================
# approximate_current_flow_betweenness_centrality(sample_weight=)
# =============================================================================


class TestSampleWeight:
    def test_more_samples_and_errors(self):
        g = ec.Graph().add_grid_graph([5, 5])
        exact = g.current_flow_betweenness_centrality(return_dict=True)
        errors = {}
        for w in (0.05, 1.0, 4.0):
            approx = g.approximate_current_flow_betweenness_centrality(
                epsilon=0.5, kmax=10**6, sample_weight=w, seed=3, return_dict=True
            )
            errors[w] = max(abs(approx[n] - exact[n]) for n in exact)
        assert errors[4.0] < errors[0.05]
        default = g.approximate_current_flow_betweenness_centrality(seed=3, return_dict=True)
        same = g.approximate_current_flow_betweenness_centrality(
            seed=3, sample_weight=1.0, return_dict=True
        )
        assert default == same
        for bad in (0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match="sample_weight"):
                g.approximate_current_flow_betweenness_centrality(sample_weight=bad)

    def test_kmax_and_tiny_weight(self):
        g = ec.Graph().add_cycle_graph(10)
        # k = ceil((cstar / eps)^2 ln n) = ceil(6.25 * 2.30) = 15 at eps 0.5 (cstar = 1.25)
        with pytest.raises(ValueError, match="kmax"):
            g.approximate_current_flow_betweenness_centrality(kmax=50, sample_weight=5)
        values = g.approximate_current_flow_betweenness_centrality(
            sample_weight=1e-9, seed=1, return_dict=True
        )
        assert len(values) == 10

    def test_steps(self):
        g = ec.Graph().add_cycle_graph(6)
        g.approximate_current_flow_betweenness_centrality(seed=1, sample_weight=2.0)
        assert "sample_weight=2.0" in g.steps[-1].code
        g.approximate_current_flow_betweenness_centrality(seed=1)
        assert "sample_weight" not in g.steps[-1].code


# =============================================================================
# spectral_modularity_bipartition
# =============================================================================


class TestSpectralModularityBipartition:
    def test_karate(self):
        g, idx = _from_nx(nx.karate_club_graph())
        left, right = g.spectral_modularity_bipartition()
        mr_hi = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21]
        assert left == {idx[n] for n in mr_hi}
        assert right == set(idx.values()) - left

    def test_matches_eigenvector_reference(self):
        for seed in range(15):
            H = nx.gnp_random_graph(25, 0.15, seed=seed)
            g, idx = _from_nx(H)
            nodes = sorted(H)
            B = nx.modularity_matrix(H, nodes)
            w, v = np.linalg.eigh(B)
            x = v[:, np.argmax(w)]
            first_nonzero = next(i for i in range(len(x)) if abs(x[i]) > 1e-12)
            if x[first_nonzero] > 0:
                x = -x
            expected = {idx[nodes[i]] for i in range(len(x)) if x[i] < -1e-12}
            left, right = g.spectral_modularity_bipartition()
            assert left == expected, seed
            assert left | right == set(idx.values()) and not left & right

    def test_weights(self):
        # Heavy edges within {1, 2, 3} and {4, 5, 6}; the unweighted graph is a 6-cycle with a
        # chord, which splits differently
        g = ec.Graph(weighted=True).add_nodes(list("ABCDEF"))
        heavy = [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")]
        g.add_edges([(u, v, {"weight": 10.0}) for u, v in heavy])
        g.add_edges([(u, v, {"weight": 0.1}) for u, v in [("C", "D"), ("F", "A")]])
        assert g.spectral_modularity_bipartition() == ({1, 2, 3}, {4, 5, 6})
        assert g.spectral_modularity_bipartition(weighted=False) in (
            ({1, 2, 3}, {4, 5, 6}), ({1, 2, 6}, {3, 4, 5}), ({1, 5, 6}, {2, 3, 4})
        )

    def test_edge_cases(self):
        assert ec.Graph().spectral_modularity_bipartition() == (set(), set())
        assert ec.Graph().add_nodes(["a", "b"]).spectral_modularity_bipartition() == (
            set(), {1, 2}
        )
        two = ec.Graph().add_nodes(["a", "b", "c", "d"]).add_edges([("a", "b"), ("c", "d")])
        assert two.spectral_modularity_bipartition() == ({1, 2}, {3, 4})
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_cycle_graph(4).spectral_modularity_bipartition()

    def test_step(self):
        g = ec.Graph().add_barbell_graph(3, 0)
        result = g.spectral_modularity_bipartition()
        step = g.steps[-1]
        assert step.code == "G.spectral_modularity_bipartition()"
        assert step.result == result
        assert sorted(step.delta.node_groups.values()) == [0, 0, 0, 1, 1, 1]
        assert len(step.delta.edges_selected) == 1


# =============================================================================
# greedy_node_swap_bipartition
# =============================================================================


def _nx37_greedy(G, init_split, max_iter):
    """NetworkX 3.7's greedy_node_swap_bipartition, with ties broken by the lowest node."""
    from copy import deepcopy

    best_split = deepcopy(init_split)
    best_mod = nx.community.modularity(G, best_split)
    max_split, max_mod = best_split, best_mod
    m = G.number_of_edges()
    degree = dict(G.degree)
    its = 0
    while max_mod >= best_mod and its < max_iter:
        best_split, best_mod = max_split, max_mod
        next_split, next_mod = deepcopy(max_split), max_mod
        nodes = set(G)
        while nodes:
            max_swap, max_node = -1, None
            left, right = next_split
            leftd = sum(degree[n] for n in left)
            rightd = sum(degree[n] for n in right)
            for n in sorted(nodes):
                if n in left:
                    inc, outc, ind, outd = left, right, leftd, rightd
                else:
                    inc, outc, ind, outd = right, left, rightd, leftd
                change = (
                    -len(G[n].keys() & inc) / m
                    + len(G[n].keys() & outc) / m
                    + degree[n] / (2 * m**2) * (ind - outd - degree[n])
                )
                if change > max_swap:
                    max_swap, max_node, src, dst = change, n, inc, outc
            src.remove(max_node)
            dst.add(max_node)
            next_mod += max_swap
            if next_mod > max_mod:
                max_split, max_mod = deepcopy(next_split), next_mod
            nodes.remove(max_node)
        its += 1
    return max_split, max_mod


class TestGreedyNodeSwapBipartition:
    def test_barbell(self):
        g = ec.Graph().add_barbell_graph(3, 0)
        for seed in range(10):
            left, right = g.greedy_node_swap_bipartition(seed=seed)
            assert sorted([sorted(left), sorted(right)]) == [[1, 2, 3], [4, 5, 6]]

    def test_matches_networkx_best_split(self):
        # NetworkX returns the split from before its last pass; its best split (max_split) is
        # what this method returns
        for seed in range(12):
            H = nx.gnp_random_graph(20, 0.2, seed=seed)
            H.remove_nodes_from(list(nx.isolates(H)))
            if H.number_of_edges() == 0:
                continue
            H = nx.convert_node_labels_to_integers(H, first_label=1)
            g, idx = _from_nx(H)
            assert all(idx[n] == n for n in H)
            rng = random.Random(seed)
            first = set(rng.sample(sorted(H), len(H) // 2))
            init = (first, set(H) - first)
            for max_iter in (1, 3, 10):
                split, mod = _nx37_greedy(H, (set(init[0]), set(init[1])), max_iter)
                left, right = g.greedy_node_swap_bipartition(
                    init_split=(sorted(init[0]), sorted(init[1])), max_iter=max_iter
                )
                # NetworkX sometimes accepts a pass whose gain is only rounding error, which
                # swaps the two sides, so compare them unordered
                assert {frozenset(left), frozenset(right)} == set(map(frozenset, split))
                assert _modularity(g, [left, right]) == pytest.approx(mod)

    def test_improves_on_start_and_karate(self):
        g, _ = _from_nx(nx.karate_club_graph())
        start = g.spectral_modularity_bipartition()
        left, right = g.greedy_node_swap_bipartition(init_split=start)
        assert _modularity(g, [left, right]) >= _modularity(g, list(start)) - 1e-12
        assert _modularity(g, [left, right]) == pytest.approx(0.3718, abs=1e-4)

    def test_weights_and_loops(self):
        g = ec.Graph(weighted=True).add_nodes(list("ABCD"))
        g.add_edges([("A", "B", {"weight": 5.0}), ("C", "D", {"weight": 5.0}),
                     ("B", "C", {"weight": 1.0}), ("A", "D", {"weight": 1.0})])
        left, right = g.greedy_node_swap_bipartition(init_split=([1, 3], [2, 4]))
        assert sorted([sorted(left), sorted(right)]) == [[1, 2], [3, 4]]
        left, right = g.greedy_node_swap_bipartition(
            init_split=([1, 3], [2, 4]), weighted=False
        )
        assert len(left) + len(right) == 4
        # A self-loop doesn't tie a node to either side
        loops = ec.Graph().add_barbell_graph(3, 0).add_edge(1, 1)
        left, right = loops.greedy_node_swap_bipartition(seed=2)
        assert sorted([sorted(left), sorted(right)]) == [[1, 2, 3], [4, 5, 6]]

    def test_init_split_labels_and_validation(self):
        g = ec.Graph().add_nodes(list("abcdef"))
        g.add_edges([("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d"),
                     ("c", "d")])
        left, right = g.greedy_node_swap_bipartition(init_split=(["a", "d", "e"], ["b", "c", "f"]))
        assert {frozenset(left), frozenset(right)} == {frozenset({1, 2, 3}), frozenset({4, 5, 6})}
        # max_iter=0 returns the starting split
        assert g.greedy_node_swap_bipartition(
            init_split=(["a", "d"], ["b", "c", "e", "f"]), max_iter=0
        ) == ({1, 4}, {2, 3, 5, 6})
        with pytest.raises(ValueError, match="init_split"):
            g.greedy_node_swap_bipartition(init_split=(["a", "b"], ["b", "c", "d", "e", "f"]))
        with pytest.raises(ValueError, match="init_split"):
            g.greedy_node_swap_bipartition(init_split=(["a"], ["b"]))
        with pytest.raises(ValueError, match="max_iter"):
            g.greedy_node_swap_bipartition(max_iter=-1)
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_cycle_graph(4).greedy_node_swap_bipartition()

    def test_seed_and_edgeless(self):
        g = ec.Graph().add_named_graph("petersen")
        assert g.greedy_node_swap_bipartition(seed=5) == g.greedy_node_swap_bipartition(seed=5)
        empty = ec.Graph().add_nodes(list("abcd"))
        left, right = empty.greedy_node_swap_bipartition(seed=1)
        assert len(left) == 2 and len(right) == 2
        assert ec.Graph().greedy_node_swap_bipartition() == (set(), set())

    def test_step(self):
        g = ec.Graph().add_barbell_graph(3, 0)
        result = g.greedy_node_swap_bipartition(seed=1)
        step = g.steps[-1]
        assert step.code == "G.greedy_node_swap_bipartition()"
        assert step.result == result
        assert step.narration.startswith("Greedy node-swap bipartition into 3 and 3 nodes")
        assert "modularity 0.3571" in step.narration


# =============================================================================
# panther_vector_similarity
# =============================================================================


class TestPantherVectorSimilarity:
    def test_star_leaves_alike(self):
        star = ec.Graph().add_star_graph(100, label=False)
        scores = star.panther_vector_similarity(2, k=10, seed=1)
        assert len(scores) == 10 and 1 not in scores and 2 not in scores
        # A leaf whose vector equals the source's scores 1 / (machine epsilon), as in NetworkX
        # #8578; the others are near
        top = 1 / np.finfo(float).eps
        assert max(scores.values()) == top
        assert all(v > 100 for v in scores.values())
        values = list(scores.values())
        assert values == sorted(values, reverse=True)
        # The center is unlike any leaf
        center = star.panther_vector_similarity(1, k=99, seed=1)
        assert max(center.values()) < 2

    def test_roles_across_communities(self):
        g = ec.Graph().add_nodes(["a", "b", "c", "x", "y", "z"])
        g.add_edges([("a", "b"), ("b", "c"), ("c", "a"), ("c", "x"), ("x", "y"), ("y", "z"),
                     ("z", "x")])
        scores = g.panther_vector_similarity("a", D=4, k=5, epsilon=0.03, seed=2)
        top = set(sorted(scores, key=scores.get, reverse=True)[:3])
        # b is a's twin; y and z mirror a and b in the other triangle; c and x are bridge ends
        assert top == {2, 5, 6}
        assert max(scores[3], scores[4]) < min(scores[v] for v in top) / 10

    def test_matches_reference_on_its_paths(self):
        # Without isolates the method samples the same paths as generate_random_paths(), so
        # the feature vectors can be rebuilt from them with NumPy
        H = nx.gnp_random_graph(15, 0.3, seed=7)
        H.remove_nodes_from(list(nx.isolates(H)))
        H = nx.convert_node_labels_to_integers(H, first_label=1)
        g, _ = _from_nx(H)
        n, eps, dims = g.count_nodes(), 0.05, 5
        size = math.ceil(0.5 / eps**2 * (math.log2(math.comb(6, 2)) + 1 + math.log(1 / 0.1)))
        for seed in range(3):
            paths = g.generate_random_paths(size, path_length=5, seed=seed)
            member = np.zeros((n, size))
            for j, path in enumerate(paths):
                member[[v - 1 for v in path], j] = 1
            S = member @ member.T / size
            theta = -np.sort(-S, axis=1)[:, :dims]
            for source in (1, 4):
                d = np.linalg.norm(theta - theta[source - 1], axis=1)
                expected = {
                    i + 1: 1 / max(d[i], np.finfo(float).eps)
                    for i in range(n)
                    if i + 1 != source
                }
                scores = g.panther_vector_similarity(
                    source, D=dims, k=n - 1, epsilon=eps, seed=seed
                )
                assert set(scores) == set(expected)
                for v, score in scores.items():
                    assert score == pytest.approx(expected[v], rel=1e-9)
                assert list(scores.values()) == sorted(scores.values(), reverse=True)

    def test_isolates_dropped_and_warning(self):
        g = ec.Graph().add_path_graph(4).add_node("lonely")
        with pytest.warns(UserWarning, match="only 3 other nodes"):
            scores = g.panther_vector_similarity(1, D=2, k=10, seed=1)
        assert set(scores) == {2, 3, 4}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g.panther_vector_similarity(1, D=2, k=3, seed=1)

    def test_errors(self):
        g = ec.Graph().add_path_graph(4).add_node("lonely")
        with pytest.raises(ValueError, match="D must be a positive integer"):
            g.panther_vector_similarity(1, D=0)
        with pytest.raises(ValueError, match="D=5"):
            g.panther_vector_similarity(1, D=5)
        with pytest.raises(ValueError, match="k must be"):
            g.panther_vector_similarity(1, D=2, k=0)
        with pytest.raises(ValueError):
            g.panther_vector_similarity("lonely", D=2)
        with pytest.raises(ValueError):
            g.panther_vector_similarity(99)

    def test_seed_and_step(self):
        g = ec.Graph().add_named_graph("petersen")
        a = g.panther_vector_similarity(1, D=4, k=4, epsilon=0.05, seed=3)
        assert a == g.panther_vector_similarity(1, D=4, k=4, epsilon=0.05, seed=3)
        step = g.steps[-1]
        assert step.code == "G.panther_vector_similarity('1', D=4, k=4)"
        assert step.result == a
        assert step.narration.startswith("Panther vector similarity: the 4 nodes")


# =============================================================================
# magnetic_laplacian_matrix and magnetic_spectrum
# =============================================================================


def _ref_magnetic(D, q, normalized, weight="weight"):
    """NetworkX 3.7's magnetic_laplacian_matrix, dense, on sorted(D)."""
    nodes = sorted(D)
    pos = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    for u, v, data in D.edges(data=True):
        if u != v:
            W[pos[u], pos[v]] += data.get(weight, 1) if weight else 1
    Ws = 0.5 * (W + W.T)
    H = np.zeros((n, n), dtype=complex)
    for j, k in itertools.product(range(n), repeat=2):
        if Ws[j, k]:
            delta = int(W[j, k] > 0) - int(W[k, j] > 0)
            H[j, k] = Ws[j, k] * cmath.exp(2j * math.pi * q * delta)
    deg = np.abs(H).sum(axis=1)
    if not normalized:
        return np.diag(deg) - H
    with np.errstate(divide="ignore"):
        inv = np.where(deg > 0, 1 / np.sqrt(deg), 0)
    return np.eye(n) - inv[:, None] * H * inv[None, :]


class TestMagnetic:
    def test_matches_reference(self):
        for seed in range(8):
            D = nx.gnp_random_graph(9, 0.35, directed=True, seed=seed)
            rng = random.Random(seed)
            for u, v in D.edges():
                D[u][v]["weight"] = rng.choice([0.5, 1.0, 2.0, 3.0])
            D.add_node(100)  # an isolated node
            g, _ = _from_nx(D, weighted=True)
            for q, norm, weighted in itertools.product((0, 0.1, 0.25, 0.5), (False, True),
                                                       (True, False)):
                matrix, ids = g.magnetic_laplacian_matrix(q=q, normalized=norm, weighted=weighted)
                assert ids == list(range(1, 11))
                ref = _ref_magnetic(D, q, norm, "weight" if weighted else None)
                np.testing.assert_allclose(np.array(matrix), ref, atol=1e-12)
                assert all(isinstance(z, complex) for row in matrix for z in row)
                spectrum = g.magnetic_spectrum(q=q, normalized=norm, weighted=weighted)
                np.testing.assert_allclose(spectrum, np.linalg.eigvalsh(ref), atol=1e-10)
                assert spectrum == sorted(spectrum)

    def test_q_zero_is_symmetrized_laplacian(self):
        D = nx.gnp_random_graph(12, 0.3, directed=True, seed=11)
        g, _ = _from_nx(D)
        matrix, _ = g.magnetic_laplacian_matrix(q=0)
        A = nx.to_numpy_array(D, sorted(D))
        Ws = 0.5 * (A + A.T)
        np.testing.assert_allclose(np.array(matrix), np.diag(Ws.sum(1)) - Ws, atol=1e-12)

    def test_hermitian_and_spectrum_bounds(self):
        D = nx.gnp_random_graph(15, 0.25, directed=True, seed=2)
        g, _ = _from_nx(D)
        M = np.array(g.magnetic_laplacian_matrix()[0])
        np.testing.assert_allclose(M, M.conj().T)
        assert min(g.magnetic_spectrum()) > -1e-10
        assert max(g.magnetic_spectrum(normalized=True)) < 2 + 1e-10

    def test_directed_cycle(self):
        triangle = ec.Graph(directed=True).add_cycle_graph(3)
        # Eigenvalues 1 - cos(2 pi (q + j / 3)) for the three Fourier modes
        expected = sorted(1 - math.cos(2 * math.pi * (0.25 + j / 3)) for j in range(3))
        assert triangle.magnetic_spectrum() == pytest.approx(expected)
        assert min(triangle.magnetic_spectrum(q=0)) == pytest.approx(0, abs=1e-12)
        # At q = 1/4 the phases around a 4-cycle make a full turn, so 0 is an eigenvalue again
        square = ec.Graph(directed=True).add_cycle_graph(4)
        assert min(square.magnetic_spectrum()) == pytest.approx(0, abs=1e-12)

    def test_two_way_edges_parallel_edges_and_loops(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["a", "b", "c"])
        g.add_edges([("a", "b", {"weight": 2.0}), ("b", "a", {"weight": 4.0}),
                     ("b", "c", {"weight": 1.0}), ("b", "c", {"weight": 1.0}),
                     ("c", "c", {"weight": 9.0})])
        matrix, _ = g.magnetic_laplacian_matrix()
        # a <-> b: no phase and the mean weight 3; b -> c twice: weight 2 halved, phase i
        assert matrix[0][1] == pytest.approx(-3) and matrix[1][0] == pytest.approx(-3)
        assert matrix[1][2] == pytest.approx(-1j) and matrix[2][1] == pytest.approx(1j)
        assert [matrix[i][i] for i in range(3)] == pytest.approx([3, 4, 1])

    def test_errors_and_steps(self):
        with pytest.raises(ValueError, match="directed graph"):
            ec.Graph().add_cycle_graph(3).magnetic_laplacian_matrix()
        with pytest.raises(ValueError, match="directed graph"):
            ec.Graph().add_cycle_graph(3).magnetic_spectrum()
        g = ec.Graph(directed=True).add_cycle_graph(3)
        for q in (-0.1, 0.6):
            with pytest.raises(ValueError, match="q must be between 0 and 0.5"):
                g.magnetic_laplacian_matrix(q=q)
            with pytest.raises(ValueError, match="q must be between 0 and 0.5"):
                g.magnetic_spectrum(q=q)
        assert ec.Graph(directed=True).magnetic_laplacian_matrix() == ([], [])
        n_steps = len(g.steps)
        g.magnetic_laplacian_matrix()
        assert g.steps[-1].code == "G.magnetic_laplacian_matrix()"
        assert g.steps[-1].narration == "magnetic_laplacian_matrix(): a 3 x 3 matrix."
        g.magnetic_spectrum()
        assert len(g.steps) == n_steps + 1


# =============================================================================
# GEXF <meta> description and keywords
# =============================================================================


class TestGexfMeta:
    def test_round_trip(self, tmp_path):
        g = ec.Graph().add_path_graph(3)
        path = tmp_path / "g.gexf"
        ec.to_gexf(g, path, description="A small <path> & more", keywords=["paths", "toy"])
        copy, meta = ec.from_gexf(path, return_meta=True)
        assert meta == {
            "creator": "EdgeCase",
            "description": "A small <path> & more",
            "keywords": "paths, toy",
        }
        assert copy.count_edges() == 2
        assert isinstance(ec.from_gexf(path), ec.Graph)
        ec.to_gexf(g, path, keywords="one, two", version="1.3")
        assert ec.from_gexf(path, return_meta=True)[1] == {
            "creator": "EdgeCase", "keywords": "one, two"
        }
        ec.to_gexf(g, path)
        assert ec.from_gexf(path, return_meta=True)[1] == {"creator": "EdgeCase"}

    def test_networkx_written_meta(self, tmp_path):
        path = tmp_path / "nx.gexf"
        path.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n'
            '  <meta lastmodifieddate="2026-10-23"><creator>NetworkX 3.7</creator>'
            "<description>roads</description><keywords>maps</keywords></meta>\n"
            '  <graph defaultedgetype="undirected" mode="static"><nodes>'
            '<node id="0" label="0" /><node id="1" label="1" /></nodes>'
            '<edges><edge source="0" target="1" id="0" /></edges></graph>\n'
            "</gexf>\n"
        )
        graph, meta = ec.from_gexf(path, return_meta=True)
        assert graph.count_edges() == 1
        assert meta == {
            "lastmodifieddate": "2026-10-23",
            "creator": "NetworkX 3.7",
            "description": "roads",
            "keywords": "maps",
        }
        no_meta = tmp_path / "bare.gexf"
        no_meta.write_text(path.read_text().replace(path.read_text().split("\n")[2], ""))
        assert ec.from_gexf(no_meta, return_meta=True)[1] == {}

    def test_readable_by_networkx(self, tmp_path):
        path = tmp_path / "g.gexf"
        ec.to_gexf(ec.Graph().add_cycle_graph(4), path, description="ring", keywords="k")
        G = nx.read_gexf(path)
        if "description" in G.graph:  # NetworkX 3.7 and later
            assert G.graph["description"] == "ring" and G.graph["keywords"] == "k"
        assert G.number_of_edges() == 4

    def test_type_errors(self, tmp_path):
        g = ec.Graph().add_path_graph(2)
        with pytest.raises(TypeError, match="keywords"):
            ec.to_gexf(g, tmp_path / "x.gexf", keywords=["a", 1])
        with pytest.raises(TypeError, match="description"):
            ec.to_gexf(g, tmp_path / "x.gexf", description=5)
