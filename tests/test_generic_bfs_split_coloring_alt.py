"""Tests for `generic_bfs_edges()` (NetworkX's node counterpart of PR #8950), split graphs
(NetworkX PR #8126), edge colouring (NetworkX PR #7397) and the ALT landmark heuristic for A*
(NetworkX PR #8767).
"""

import random

import edgecase as ec
import networkx as nx
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _random_graph(seed, directed=False, n_max=10, m_max=15):
    rng = random.Random(seed)
    n = rng.randrange(1, n_max)
    g = ec.Graph(directed=directed).add_nodes([str(i) for i in range(n)])
    edges = [(str(rng.randrange(n)), str(rng.randrange(n))) for _ in range(rng.randrange(m_max))]
    if edges:
        g.add_edges(edges)
    return g


def _nx_multi(g):
    """The graph as a NetworkX multigraph on EdgeCase's node IDs, edges in ID order."""
    G = (nx.MultiDiGraph if g.is_directed() else nx.MultiGraph)()
    G.add_nodes_from(sorted(g.get_node_ids(limit=None)))
    for _, s, t in sorted(g.get_edges_as_tuples()):
        G.add_edge(s, t)
    return G


# =============================================================================
# generic_bfs_edges
# =============================================================================


class TestGenericBfsEdges:
    def test_matches_networkx(self):
        key = lambda x: (x % 3, -x)  # noqa: E731
        for seed in range(150):
            g = _random_graph(seed, directed=seed % 2 == 0)
            G = _nx_multi(g)
            for source in G:
                for depth in (None, 0, 1, 2):
                    assert g.generic_bfs_edges(source, depth_limit=depth) == list(
                        nx.generic_bfs_edges(G, source, depth_limit=depth)
                    )

                    def order(v):
                        return sorted(G.neighbors(v), key=key)

                    assert g.generic_bfs_edges(source, order, depth) == list(
                        nx.generic_bfs_edges(G, source, lambda v: iter(order(v)), depth)
                    )

    def test_default_is_bfs_edges(self):
        for seed in range(40):
            g = _random_graph(seed, directed=seed % 2 == 1)
            for source in g.get_node_ids(limit=None):
                assert g.generic_bfs_edges(source) == g.bfs_edges(source)
                assert g.generic_bfs_edges(source, depth_limit=1) == g.bfs_edges(source, 1)

    def test_networkx_examples(self):
        g = ec.Graph().add_path_graph(7)
        assert g.generic_bfs_edges(1) == [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        assert g.generic_bfs_edges(3) == [(3, 2), (3, 4), (2, 1), (4, 5), (5, 6), (6, 7)]
        assert g.generic_bfs_edges(3, depth_limit=2) == [(3, 2), (3, 4), (2, 1), (4, 5)]
        star = ec.Graph().add_nodes([str(i) for i in range(6)])
        star.add_edges([("0", str(i)) for i in range(1, 6)])
        lab = star.get_label_from_id

        def odd_first(n):
            return sorted(star.neighbors(n), key=lambda x: int(lab(x)) % 2, reverse=True)

        found = [(lab(u), lab(v)) for u, v in star.generic_bfs_edges("0", neighbors=odd_first)]
        assert found == [("0", "1"), ("0", "3"), ("0", "5"), ("0", "2"), ("0", "4")]

    def test_callback_filters_and_takes_labels(self):
        g = ec.Graph().add_nodes(list("ABCD"))
        g.add_edges([("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")])
        lab = g.get_label_from_id
        # Never step from A straight to C; labels are accepted as well as IDs
        found = g.generic_bfs_edges(
            "A", lambda n: [lab(m) for m in g.neighbors(n) if {lab(n), lab(m)} != {"A", "C"}]
        )
        assert [(lab(u), lab(v)) for u, v in found] == [("A", "B"), ("B", "C"), ("C", "D")]
        # Links that are not edges of the graph are followed too
        chain = g.generic_bfs_edges("D", lambda n: [n - 1] if n > 1 else [])
        assert chain == [(4, 3), (3, 2), (2, 1)]

    def test_callback_called_once_per_reached_node(self):
        g = ec.Graph().add_full_graph(5)
        calls = []

        def spy(n):
            calls.append(n)
            return g.neighbors(n)

        g.generic_bfs_edges(1, spy)
        assert sorted(calls) == [1, 2, 3, 4, 5]

    def test_errors(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError):
            g.generic_bfs_edges("nobody")
        with pytest.raises(ValueError, match="depth_limit"):
            g.generic_bfs_edges(1, depth_limit=-1)
        with pytest.raises(ValueError):
            g.generic_bfs_edges(1, lambda n: [99])

    def test_steps(self):
        g = ec.Graph().add_path_graph(3)
        n_steps = len(g.steps)

        def forward(n):
            return g.neighbors(n)

        result = g.generic_bfs_edges(1, forward, depth_limit=5)
        assert len(g.steps) == n_steps + 1  # the callback's queries record nothing
        step = g.steps[-1]
        assert step.code == "G.generic_bfs_edges(1, neighbors=forward, depth_limit=5)"
        assert step.narration == "BFS tree from 1: 2 edges."
        assert result == [(1, 2), (2, 3)]
        g.generic_bfs_edges(2)
        assert g.steps[-1].code == "G.generic_bfs_edges(2)"

    def test_generic_edge_bfs_callback_records_nothing(self):
        g = ec.Graph().add_path_graph(3)
        n_steps = len(g.steps)
        g.generic_edge_bfs(1, edges=lambda n: [(n, m) for m in g.neighbors(n)])
        assert len(g.steps) == n_steps + 1


# =============================================================================
# is_split_graph / is_complete_split_graph
# =============================================================================


def _from_nx(G):
    """An undirected EdgeCase graph with the nodes of `G` as labels."""
    g = ec.Graph().add_nodes([str(v) for v in G])
    if G.number_of_edges():
        g.add_edges([(str(u), str(v)) for u, v in G.edges()])
    return g


def _pr_is_split(G):
    """NetworkX PR #8126's definition: chordal with a chordal complement."""
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.is_chordal(G) and nx.is_chordal(nx.complement(G))


def _is_complete_split_reference(G):
    """A clique, an independent set and every edge between them, by brute force."""
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = list(G)
    for mask in range(1 << len(nodes)):
        clique = {v for i, v in enumerate(nodes) if mask >> i & 1}
        rest = set(nodes) - clique
        if all(G.has_edge(u, v) for u in clique for v in clique if u != v) and all(
            G.has_edge(u, v) == (u in clique or v in clique) for u in G for v in rest if u != v
        ):
            return True
    return False


class TestSplitGraphs:
    def test_networkx_pr_examples(self):
        kite = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4)])
        split = [nx.complete_graph(4), nx.star_graph(3), kite, nx.path_graph(4)]
        assert [_from_nx(G).is_split_graph() for G in split] == [True] * 4
        assert not _from_nx(nx.cycle_graph(4)).is_split_graph()
        complete = [nx.star_graph(3), kite, nx.path_graph(4), nx.cycle_graph(4)]
        assert [_from_nx(G).is_complete_split_graph() for G in complete] == [
            True,
            False,
            False,
            False,
        ]
        assert _from_nx(nx.complete_graph(4)).is_complete_split_graph()

    def test_matches_references(self):
        for seed in range(400):
            g = _random_graph(seed, n_max=9, m_max=20)
            G = nx.Graph(_nx_multi(g))
            assert g.is_split_graph() == _pr_is_split(G)
            assert g.is_complete_split_graph() == _is_complete_split_reference(G)

    def test_classic_families(self):
        # C5 and the 2K2 are the forbidden induced subgraphs (with C4)
        assert not _from_nx(nx.cycle_graph(5)).is_split_graph()
        assert not _from_nx(nx.Graph([(0, 1), (2, 3)])).is_split_graph()
        assert _from_nx(nx.complete_multipartite_graph(4, 1, 1, 1)).is_complete_split_graph()
        assert not _from_nx(nx.complete_bipartite_graph(2, 2)).is_split_graph()
        assert _from_nx(nx.empty_graph(5)).is_complete_split_graph()

    def test_trivial_graphs(self):
        for n in range(3):
            g = ec.Graph().add_nodes([str(i) for i in range(n)])
            assert g.is_split_graph() and g.is_complete_split_graph()

    def test_self_loops_and_parallel_edges_ignored(self):
        g = ec.Graph().add_nodes(list("ABCD"))
        g.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
        assert not g.is_split_graph()
        g.add_edges([("A", "A"), ("A", "B"), ("C", "C")])
        assert not g.is_split_graph()
        star = ec.Graph().add_star_graph(4)
        star.add_edges([(1, 1), (1, 2), (3, 3)])
        assert star.is_split_graph() and star.is_complete_split_graph()

    def test_directed_raises(self):
        g = ec.Graph(directed=True).add_path_graph(3)
        with pytest.raises(ValueError, match="undirected"):
            g.is_split_graph()
        with pytest.raises(ValueError, match="undirected"):
            g.is_complete_split_graph()

    def test_steps(self):
        g = ec.Graph().add_star_graph(4)
        g.is_split_graph()
        assert g.steps[-1].code == "G.is_split_graph()"
        assert g.steps[-1].narration == "is_split_graph(): True."
        g.is_complete_split_graph()
        assert g.steps[-1].code == "G.is_complete_split_graph()"


# =============================================================================
# edge_coloring / is_valid_edge_coloring
# =============================================================================


def _check_edge_coloring(g):
    coloring = g.edge_coloring()
    assert sorted(coloring) == sorted(g.get_edge_ids())
    assert g.is_valid_edge_coloring(coloring)
    G = nx.Graph(_nx_multi(g))
    delta = max((d for _, d in G.degree()), default=0)
    assert len(set(coloring.values())) <= delta + 1
    assert set(coloring.values()) <= set(range(delta + 1))
    return coloring


class TestEdgeColoring:
    def test_networkx_pr_graphs(self):
        thirteen = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 12), (1, 2), (1, 3), (1, 11), (1, 5), (2, 3),
            (2, 4), (2, 5), (3, 4), (3, 5), (4, 5), (6, 7), (3, 8), (6, 9), (6, 10), (6, 11),
            (6, 12), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (8, 9), (8, 10), (5, 11),
            (8, 12), (2, 10), (9, 11), (9, 12), (10, 11), (10, 12), (11, 12),
        ]  # fmt: skip
        disconnected = [(0, 1), (0, 2), (0, 3), (1, 2), (4, 5), (4, 6), (6, 7)]
        for G in (
            nx.complete_graph(13),
            nx.kneser_graph(5, 2),
            nx.Graph(thirteen),
            nx.Graph(disconnected),
        ):
            _check_edge_coloring(_from_nx(G))

    def test_random_graphs(self):
        for seed in range(300):
            rng = random.Random(seed)
            n = rng.randrange(1, 14)
            G = nx.gnm_random_graph(n, rng.randrange(n * (n - 1) // 2 + 1), seed=seed)
            _check_edge_coloring(_from_nx(G))

    def test_class_one_and_two(self):
        # Bipartite graphs and even complete graphs need only delta colors; odd cliques need
        # delta + 1 (Vizing's bound is tight there), so any proper colouring uses delta + 1
        k5 = _check_edge_coloring(ec.Graph().add_full_graph(5))
        assert len(set(k5.values())) == 5
        petersen = _check_edge_coloring(_from_nx(nx.petersen_graph()))
        assert len(set(petersen.values())) == 4
        assert len(set(_check_edge_coloring(ec.Graph().add_path_graph(6)).values())) == 2

    def test_directed_ignores_direction(self):
        g = ec.Graph(directed=True).add_nodes(list("ABCD"))
        g.add_edges([("A", "B"), ("C", "B"), ("B", "D"), ("D", "A")])
        coloring = _check_edge_coloring(g)
        assert len(set(coloring.values())) == 3

    def test_empty_and_edgeless(self):
        assert ec.Graph().edge_coloring() == {}
        assert ec.Graph().add_nodes(["a", "b"]).edge_coloring() == {}
        assert ec.Graph().is_valid_edge_coloring({})

    def test_errors(self):
        loop = ec.Graph().add_nodes(["a"])
        loop.add_edge("a", "a")
        with pytest.raises(ValueError, match="self-loops"):
            loop.edge_coloring()
        multi = ec.Graph().add_nodes(["a", "b"])
        multi.add_edges([("a", "b"), ("a", "b")])
        with pytest.raises(ValueError, match="parallel"):
            multi.edge_coloring()
        anti = ec.Graph(directed=True).add_nodes(["a", "b"])
        anti.add_edges([("a", "b"), ("b", "a")])
        with pytest.raises(ValueError, match="parallel"):
            anti.edge_coloring()

    def test_is_valid_edge_coloring(self):
        g = ec.Graph().add_cycle_graph(3)
        e1, e2, e3 = sorted(g.get_edge_ids())
        assert g.is_valid_edge_coloring({e1: 0, e2: 1, e3: 2})
        assert not g.is_valid_edge_coloring({e1: 0, e2: 0, e3: 1})
        # Uncoloured edges are skipped and unknown keys ignored
        assert g.is_valid_edge_coloring({e1: 0, e2: 1})
        assert g.is_valid_edge_coloring({e1: 0, 999: 0})

    def test_records_no_steps(self):
        g = ec.Graph().add_cycle_graph(4)
        n_steps = len(g.steps)
        g.is_valid_edge_coloring(g.edge_coloring())
        assert len(g.steps) == n_steps


# =============================================================================
# alt_heuristic
# =============================================================================


def _weighted_random(seed, directed, n=25, m=120):
    G = nx.gnm_random_graph(n, m, seed=seed, directed=directed)
    g = ec.Graph(directed=directed, weighted=True).add_nodes([str(v) for v in G])
    rng = random.Random(seed)
    for u, v in G.edges():
        G[u][v]["weight"] = w = round(rng.uniform(0.1, 5), 3)
        g.add_edge(str(u), str(v), weight=w)
    return g, G


def _pr_alt(G, landmarks):
    """NetworkX PR #8767's heuristic for explicit landmarks."""
    inf = float("inf")
    dist_from = [nx.single_source_dijkstra_path_length(G, x) for x in landmarks]
    dist_to = (
        [nx.single_source_dijkstra_path_length(G.reverse(copy=False), x) for x in landmarks]
        if G.is_directed()
        else dist_from
    )

    def h(u, v):
        best = 0
        for d_from, d_to in zip(dist_from, dist_to):
            u_to, v_to = d_to.get(u, inf), d_to.get(v, inf)
            if u_to < inf and v_to < inf:
                best = max(best, u_to - v_to)
            u_from, v_from = d_from.get(u, inf), d_from.get(v, inf)
            if u_from < inf and v_from < inf:
                best = max(best, v_from - u_from)
        return best

    return h


def _true_distances(G):
    return dict(nx.all_pairs_dijkstra_path_length(G))


class TestAltHeuristic:
    def test_admissible_and_consistent(self):
        for seed in range(12):
            directed = seed % 2 == 0
            g, G = _weighted_random(seed, directed)
            h = g.alt_heuristic(k=1 + seed % 6, method=("farthest", "random")[seed % 3 == 0])
            d = _true_distances(G)
            for u in G:
                for v in G:
                    assert h(str(u), str(v)) <= d[u].get(v, float("inf")) + 1e-9
            for u, v, data in G.edges(data=True):
                pairs = [(u, v), (v, u)] if not directed else [(u, v)]
                for a, b in pairs:
                    for t in d[b]:  # targets the edge's head can reach
                        assert h(str(a), str(t)) <= data["weight"] + h(str(b), str(t)) + 1e-9

    def test_matches_networkx_pr_with_landmarks(self):
        for seed in range(10):
            g, G = _weighted_random(seed, directed=seed % 2 == 1, n=15, m=40)
            chosen = random.Random(seed).sample(list(G), 3)
            h = g.alt_heuristic(landmarks=[str(x) for x in chosen])
            ref = _pr_alt(G, chosen)
            for u in G:
                for v in G:
                    assert h(str(u), str(v)) == pytest.approx(ref(u, v))

    def test_astar_matches_dijkstra(self):
        for seed in range(6):
            g, G = _weighted_random(seed, directed=True, n=40, m=150)
            h = g.alt_heuristic(k=8, seed=seed)
            d = _true_distances(G)
            rng = random.Random(seed)
            for _ in range(30):
                s, t = rng.sample(list(G), 2)
                found = g.astar_path_length(str(s), str(t), heuristic=h)
                if t in d[s]:
                    assert found == pytest.approx(d[s][t])
                else:
                    assert found is None

    def test_networkx_example(self):
        g = ec.Graph().add_path_graph(5)
        h = g.alt_heuristic(k=2, seed=10)
        assert g.astar_path(1, 5, heuristic=h) == [1, 2, 3, 4, 5]
        assert h(1, 5) == 4 and isinstance(h(1, 5), int)

    def test_farthest_selection(self):
        g = ec.Graph().add_path_graph(9)
        for seed in range(10):
            first, *rest = g.alt_heuristic(k=4, seed=seed).landmarks
            expected = [first]
            while len(expected) < 4:  # farthest from its nearest landmark; ties to smallest ID
                gap = {v: min(abs(v - x) for x in expected) for v in range(1, 10)}
                expected.append(max((v for v in gap if v not in expected), key=gap.get))
            assert rest == expected[1:]

    def test_k_capped_and_seed_reproducible(self):
        g = ec.Graph().add_grid_2d(3, 3)
        assert sorted(g.alt_heuristic(k=50).landmarks) == sorted(g.get_node_ids(limit=None))
        for method in ("farthest", "random"):
            a = g.alt_heuristic(k=3, method=method, seed=4).landmarks
            assert a == g.alt_heuristic(k=3, method=method, seed=4).landmarks
            assert len(set(a)) == 3

    def test_disconnected(self):
        G = nx.disjoint_union(nx.path_graph(5), nx.cycle_graph(5))
        g = _from_nx(G)
        h = g.alt_heuristic(k=3, seed=0)
        # Unreachable nodes are picked early, so both components get a landmark
        labels = {g.get_label_from_id(x) for x in h.landmarks}
        assert labels & {"0", "1", "2", "3", "4"} and labels & {"5", "6", "7", "8", "9"}
        assert all(0 <= h(str(u), "7") < float("inf") for u in G)
        assert h("0", "7") == 0
        assert g.astar_path("0", "7", heuristic=h) is None

    def test_explicit_landmarks(self):
        g = ec.Graph().add_grid_2d(5, 5)
        ids = sorted(g.get_node_ids(limit=None))
        h = g.alt_heuristic(landmarks=[ids[0], ids[-1], ids[0]], k=0, method="nope")
        assert h.landmarks == [ids[0], ids[-1]]
        assert h(ids[0], ids[-1]) == 8
        assert h(ids[4], ids[20]) == 0  # the other two corners are invisible to these

    def test_unweighted_flag(self):
        g = ec.Graph(weighted=True).add_nodes(list("abc"))
        g.add_edges([("a", "b", 5.0), ("b", "c", 5.0)])
        assert g.alt_heuristic(landmarks=["a"])("a", "c") == 10.0
        hops = g.alt_heuristic(landmarks=["a"], weighted=False)
        assert hops("a", "c") == 2
        assert g.astar_path_length("a", "c", heuristic=hops, weighted=False) == 2

    def test_stale_and_unknown_nodes(self):
        g = ec.Graph().add_path_graph(3)
        h = g.alt_heuristic(k=2, seed=0)
        g.add_nodes(["late"])
        assert h("late", 1) == 0
        with pytest.raises(ValueError):
            h("nobody", 1)

    def test_errors(self):
        with pytest.raises(ValueError, match="non-empty"):
            ec.Graph().alt_heuristic()
        g = ec.Graph().add_path_graph(3)
        for k in (0, -1):
            with pytest.raises(ValueError, match="k="):
                g.alt_heuristic(k=k)
        with pytest.raises(ValueError, match="method="):
            g.alt_heuristic(method="invalid")
        with pytest.raises(ValueError, match="landmarks="):
            g.alt_heuristic(landmarks=[])
        with pytest.raises(ValueError):
            g.alt_heuristic(landmarks=[1, "not-a-node"])
        neg = ec.Graph(directed=True, weighted=True).add_nodes(["a", "b"])
        neg.add_edge("a", "b", weight=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            neg.alt_heuristic()

    def test_steps(self):
        g = ec.Graph().add_nodes(list("abc"))
        g.add_edges([("a", "b"), ("b", "c")])
        h = g.alt_heuristic(landmarks=["a", "c"])
        step = g.steps[-1]
        assert step.code == "G.alt_heuristic(k=2)"
        assert step.narration == "ALT heuristic with 2 landmarks: a, c."
        assert step.delta.nodes_selected == ["a", "c"]
        n_steps = len(g.steps)
        h("a", "c")
        assert len(g.steps) == n_steps
