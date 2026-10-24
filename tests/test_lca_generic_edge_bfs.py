"""Tests for the lowest-common-ancestor additions of NetworkX PR #7988
(`all_lowest_common_ancestors()`, `all_pairs_all_lowest_common_ancestors()`,
`is_lowest_common_ancestor()` and `key=`) and for `generic_edge_bfs()` from NetworkX PR #8950.

The installed NetworkX may lack them, so the references below follow the PRs' definitions.
"""

import itertools
import random

import edgecase as ec
import networkx as nx
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _random_dag(seed, n=9, p=0.3):
    rng = random.Random(seed)
    edges = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    order = list(range(n))
    rng.shuffle(order)  # so node IDs are not a topological order
    edges = [(order[u], order[v]) for u, v in edges]
    g = ec.Graph(directed=True).add_nodes([str(i) for i in range(n)])
    if edges:
        g.add_edges([(str(u), str(v)) for u, v in edges])
    G = nx.DiGraph(edges)
    G.add_nodes_from(range(n))
    return g, G


def _ref_all_lcas(G, u, v):
    """NetworkX PR #7988: common ancestors with no successor that is a common ancestor."""
    common = (nx.ancestors(G, u) | {u}) & (nx.ancestors(G, v) | {v})
    return sorted(x for x in common if not any(s in common for s in G.successors(x)))


def _diamonds():
    """ann and bob share both parents mum and dad; grandma is mum's parent."""
    g = ec.Graph(directed=True).add_nodes(["mum", "dad", "ann", "bob", "grandma", "eve"])
    g.add_edges(
        [("grandma", "mum"), ("mum", "ann"), ("mum", "bob"), ("dad", "ann"), ("dad", "bob")]
    )
    return g


# =============================================================================
# all_lowest_common_ancestors / all_pairs_all_lowest_common_ancestors
# =============================================================================


class TestAllLowestCommonAncestors:
    def test_matches_reference(self):
        for seed in range(40):
            g, G = _random_dag(seed)
            nid = {n: g.get_id_from_label(str(n)) for n in G}
            back = {v: k for k, v in nid.items()}
            for u, v in itertools.combinations_with_replacement(G, 2):
                ours = [back[x] for x in g.all_lowest_common_ancestors(nid[u], nid[v])]
                assert sorted(ours) == _ref_all_lcas(G, u, v), (seed, u, v)

    def test_networkx_examples(self):
        g = ec.Graph(directed=True).add_nodes([str(i) for i in range(5)])
        g.add_edges([("0", "1"), ("1", "2"), ("2", "3"), ("0", "4"), ("4", "3")])
        assert [g.get_label_from_id(n) for n in g.all_lowest_common_ancestors("2", "4")] == ["0"]
        h = ec.Graph(directed=True).add_nodes([str(i) for i in range(5)])
        h.add_edges([("0", "1"), ("0", "2"), ("1", "3"), ("2", "3"), ("1", "4"), ("2", "4")])
        assert [h.get_label_from_id(n) for n in h.all_lowest_common_ancestors("3", "4")] == [
            "1", "2"
        ]

    def test_self_ancestor_none_and_order(self):
        g = _diamonds()
        assert g.all_lowest_common_ancestors("ann", "bob") == [1, 2]
        assert g.all_lowest_common_ancestors("ann", "ann") == [3]
        assert g.all_lowest_common_ancestors("grandma", "ann") == [5]
        assert g.all_lowest_common_ancestors("ann", "eve") == []
        assert g.all_lowest_common_ancestors(3, 4) == [1, 2]

    def test_contains_the_single_lca(self):
        for seed in range(25):
            g, G = _random_dag(seed, n=8, p=0.35)
            for u, v in itertools.combinations(sorted(g.get_node_ids()), 2):
                one = g.lowest_common_ancestor(u, v)
                every = g.all_lowest_common_ancestors(u, v)
                assert (one is None) == (not every)
                assert one is None or one in every

    def test_all_pairs(self):
        g = _diamonds()
        found = g.all_pairs_all_lowest_common_ancestors([("ann", "bob"), ("ann", "eve"), (5, 3)])
        assert found == {(3, 4): [1, 2], (3, 6): [], (5, 3): [5]}
        everything = g.all_pairs_all_lowest_common_ancestors()
        ids = sorted(g.get_node_ids())
        assert list(everything) == [(u, v) for i, u in enumerate(ids) for v in ids[i:]]
        for (u, v), lcas in everything.items():
            assert lcas == g.all_lowest_common_ancestors(u, v)

    def test_all_pairs_networkx_examples(self):
        g = ec.Graph(directed=True).add_nodes(["0", "1", "2", "3"])
        g.add_edges([("0", "1"), ("0", "3"), ("1", "2")])
        lab = g.get_label_from_id
        found = {
            tuple(int(lab(n)) for n in pair): [int(lab(a)) for a in lcas]
            for pair, lcas in g.all_pairs_all_lowest_common_ancestors().items()
        }
        assert found == {
            (0, 0): [0], (0, 1): [0], (0, 3): [0], (0, 2): [0], (1, 1): [1], (1, 3): [0],
            (1, 2): [1], (3, 3): [3], (2, 3): [0], (2, 2): [2],
        }
        h = ec.Graph(directed=True).add_nodes(["0", "1", "2", "3"])
        h.add_edges([("0", "1"), ("2", "3")])
        assert h.all_pairs_all_lowest_common_ancestors([("1", "3")]) == {(2, 4): []}

    def test_errors(self):
        cyclic = ec.Graph(directed=True).add_cycle_graph(3)
        undirected = ec.Graph().add_path_graph(3)
        for g in (cyclic, undirected):
            with pytest.raises(ValueError, match="directed"):
                g.all_lowest_common_ancestors(1, 2)
            with pytest.raises(ValueError, match="directed"):
                g.all_pairs_all_lowest_common_ancestors()
            with pytest.raises(ValueError, match="directed"):
                g.is_lowest_common_ancestor(1, 2, 3)
        with pytest.raises(ValueError, match="no nodes"):
            ec.Graph(directed=True).all_pairs_all_lowest_common_ancestors()
        g = _diamonds()
        with pytest.raises(ValueError):
            g.all_lowest_common_ancestors("ann", "nobody")
        with pytest.raises(ValueError):
            g.all_pairs_all_lowest_common_ancestors([("ann", "nobody")])
        with pytest.raises(ValueError):
            g.is_lowest_common_ancestor("ann", "bob", "nobody")

    def test_steps(self):
        g = _diamonds()
        g.all_lowest_common_ancestors("ann", "bob")
        step = g.steps[-1]
        assert step.code == "G.all_lowest_common_ancestors('ann', 'bob')"
        assert step.result == [1, 2]
        assert step.narration == "Lowest common ancestors of ann and bob: mum, dad."
        g.all_lowest_common_ancestors("ann", "eve")
        assert g.steps[-1].narration == "ann and eve share no ancestor."
        g.all_pairs_all_lowest_common_ancestors([("ann", "bob"), ("ann", "eve")])
        assert g.steps[-1].narration == "Lowest common ancestors for 1 of 2 pairs."
        g.is_lowest_common_ancestor("ann", "bob", "grandma")
        assert g.steps[-1].narration == "grandma is not a lowest common ancestor of ann and bob."
        assert g.steps[-1].result is False


# =============================================================================
# is_lowest_common_ancestor
# =============================================================================


class TestIsLowestCommonAncestor:
    def test_networkx_examples(self):
        g = ec.Graph(directed=True).add_nodes(["0", "1", "2", "3"])
        g.add_edges([("0", "1"), ("1", "3"), ("0", "2"), ("2", "3")])
        assert g.is_lowest_common_ancestor("1", "2", "0")
        assert not g.is_lowest_common_ancestor("1", "2", "3")
        assert g.is_lowest_common_ancestor("3", "3", "3")
        assert not g.is_lowest_common_ancestor("1", "3", "0")
        assert g.is_lowest_common_ancestor("1", "3", "1")

    def test_matches_reference(self):
        for seed in range(20):
            g, G = _random_dag(seed, n=7, p=0.35)
            nid = {n: g.get_id_from_label(str(n)) for n in G}
            for u, v in itertools.combinations_with_replacement(G, 2):
                lcas = set(_ref_all_lcas(G, u, v))
                for x in G:
                    assert g.is_lowest_common_ancestor(nid[u], nid[v], nid[x]) == (x in lcas)


# =============================================================================
# key=
# =============================================================================


class TestLcaKey:
    def test_lowest_common_ancestor_key(self):
        g = _diamonds()
        lab = g.get_label_from_id
        # Without key=, the deepest LCA (mum, below grandma) is returned
        assert lab(g.lowest_common_ancestor("ann", "bob")) == "mum"
        assert lab(g.lowest_common_ancestor("ann", "bob", key=lab)) == "dad"
        assert lab(g.lowest_common_ancestor("ann", "bob", key=lambda n: -n)) == "dad"
        # Ties in key go to the lowest node ID
        assert g.lowest_common_ancestor("ann", "bob", key=lambda n: 0) == 1
        assert g.lowest_common_ancestor("ann", "eve", key=lab) is None
        assert g.lowest_common_ancestor("ann", "ann", key=lab) == 3
        with pytest.raises(ValueError, match="directed"):
            ec.Graph().add_path_graph(3).lowest_common_ancestor(1, 2, key=str)

    def test_all_pairs_key(self):
        g = _diamonds()
        lab = g.get_label_from_id
        pairs = [("ann", "bob"), ("ann", "eve"), ("mum", "ann")]
        assert g.all_pairs_lowest_common_ancestor(pairs) == {(3, 4): 1, (1, 3): 1}
        assert g.all_pairs_lowest_common_ancestor(pairs, key=lab) == {(3, 4): 2, (1, 3): 1}

    def test_key_picks_min_of_all(self):
        for seed in range(25):
            g, _ = _random_dag(seed, n=8, p=0.35)
            ids = sorted(g.get_node_ids())
            key = lambda n: (n * 7) % 5  # noqa: E731
            found = g.all_pairs_lowest_common_ancestor(key=key)
            every = g.all_pairs_all_lowest_common_ancestors()
            assert found == {p: min(lcas, key=key) for p, lcas in every.items() if lcas}
            for u, v in itertools.combinations(ids, 2):
                assert g.lowest_common_ancestor(u, v, key=key) == found.get((u, v))

    def test_steps(self):
        g = _diamonds()
        g.lowest_common_ancestor("ann", "bob", key=str)
        assert g.steps[-1].code == "G.lowest_common_ancestor('ann', 'bob', key=...)"
        g.all_pairs_lowest_common_ancestor(key=str)
        assert g.steps[-1].code == "G.all_pairs_lowest_common_ancestor(key=...)"


# =============================================================================
# generic_edge_bfs
# =============================================================================


def _nx_view(g):
    G = (nx.MultiDiGraph if g.is_directed() else nx.MultiGraph)()
    G.add_nodes_from(sorted(g.get_node_ids(limit=None)))
    for eid, s, t in sorted(g.get_edges_as_tuples()):
        G.add_edge(s, t, key=eid)
    return G


class TestGenericEdgeBfs:
    def test_default_is_edge_bfs(self):
        for seed in range(30):
            rng = random.Random(seed)
            directed = seed % 2 == 0
            g = ec.Graph(directed=directed).add_nodes([str(i) for i in range(7)])
            edges = [(str(rng.randrange(7)), str(rng.randrange(7))) for _ in range(11)]
            g.add_edges(edges)
            for source in (None, 1, [3, 1]):
                assert g.generic_edge_bfs(source) == g.edge_bfs(source)
            # ...which matches NetworkX's edge_bfs on the multigraph, keys as edge IDs
            assert g.generic_edge_bfs() == list(nx.edge_bfs(_nx_view(g)))

    def test_networkx_examples(self):
        g = ec.Graph().add_path_graph(4)
        assert [(u, v) for u, v, _ in g.generic_edge_bfs(1)] == [(1, 2), (2, 3), (3, 4)]
        h = ec.Graph().add_nodes(["0", "1", "2"]).add_edges([("0", "1"), ("0", "2"), ("1", "2")])
        a, c = h.get_id_from_label("0"), h.get_id_from_label("2")

        def custom(node):
            return [(s, t) for _, s, t in h.get_edges_as_tuples()
                    if node in (s, t) and {s, t} != {a, c}]

        assert h.generic_edge_bfs(a, edges=custom) == [(1, 2), (2, 3)]

    def test_pairs_deduplicated_by_endpoints(self):
        und = ec.Graph().add_cycle_graph(3)
        both_ways = lambda n: [(n, m) for m in (1, 2, 3) if m != n]  # noqa: E731
        assert und.generic_edge_bfs(1, edges=both_ways) == [(1, 2), (1, 3), (2, 3)]
        di = ec.Graph(directed=True).add_cycle_graph(3)
        assert di.generic_edge_bfs(1, edges=both_ways) == [
            (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)
        ]

    def test_parallel_edges_and_extra_items(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B"])
        g.add_edges([("A", "B"), ("A", "B"), ("B", "A")])

        def with_data(node):
            return [(s, t, e, "x") for e, s, t in sorted(g.get_edges_as_tuples()) if s == node]

        assert g.generic_edge_bfs("A", edges=with_data) == [
            (1, 2, 1, "x"), (1, 2, 2, "x"), (2, 1, 3, "x")
        ]

    def test_reverse_edges_from_callback(self):
        # A callback may give edges with the node at position 1, as for in-edges
        g = ec.Graph(directed=True).add_path_graph(4)

        def in_edges(node):
            return [(s, t, e) for e, s, t in g.get_edges_as_tuples() if t == node]

        assert g.generic_edge_bfs(4, edges=in_edges) == [(3, 4, 3), (2, 3, 2), (1, 2, 1)]

    def test_errors_and_empty(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError, match="not incident"):
            g.generic_edge_bfs(1, edges=lambda n: [(2, 3)])
        with pytest.raises(ValueError):
            g.generic_edge_bfs("nobody")
        assert ec.Graph().generic_edge_bfs() == []
        assert g.generic_edge_bfs(1, edges=lambda n: []) == []

    def test_step(self):
        g = ec.Graph().add_cycle_graph(3)
        result = g.generic_edge_bfs(1)
        step = g.steps[-1]
        assert step.code == "G.generic_edge_bfs()"
        assert step.result == result
        assert step.narration == "generic_edge_bfs(): 3 edges visited."
