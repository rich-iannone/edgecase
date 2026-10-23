"""Tests for cmty_semisync_label_propagation(), incremental_closeness_centrality(),
bfs_beam_edges(), dfs_labeled_edges() and maximal_extendability(), checked against NetworkX."""

import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_TRAVERSE
from networkx.algorithms import bipartite as nxb


def from_nx(G: nx.Graph) -> tuple[ec.Graph, dict]:
    """An edgecase graph with nodes 1..n for NetworkX's nodes (in order), and the node map."""
    index = {v: i + 1 for i, v in enumerate(G)}
    g = ec.Graph(directed=G.is_directed()).add_nodes([str(i + 1) for i in range(len(G))])
    g.add_edges([(str(index[u]), str(index[v])) for u, v in G.edges()])
    return g, index


def random_graphs(count: int, seed: int, directed: bool = False):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(1, 14)
        p = rng.choice([0.1, 0.2, 0.35, 0.5])
        yield nx.gnp_random_graph(n, p, seed=rng.randrange(2**31), directed=directed)


def random_multigraphs(count: int, seed: int, directed: bool = False):
    """Small random multigraphs with parallel edges and self-loops, built in both libraries
    from the same edge list (a NetworkX multigraph doesn't list its edges in insertion order,
    which sets the neighbor order). Yields `(G, g, index)` like `from_nx`."""
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(1, 8)
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 14))]
        G = nx.MultiDiGraph() if directed else nx.MultiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        g = ec.Graph(directed=directed).add_nodes([str(i + 1) for i in range(n)])
        g.add_edges([(str(u + 1), str(v + 1)) for u, v in edges])
        yield G, g, {v: v + 1 for v in range(n)}


def random_matchable_bipartite(rng: random.Random) -> nx.Graph:
    """A random bipartite graph with a perfect matching, its nodes shuffled."""
    k = rng.randint(1, 6)
    B = nx.Graph()
    B.add_nodes_from(range(2 * k))
    B.add_edges_from((i, k + i) for i in range(k))
    p = rng.choice([0.2, 0.5, 0.8])
    B.add_edges_from((i, k + j) for i in range(k) for j in range(k) if rng.random() < p)
    order = list(B)
    rng.shuffle(order)
    H = nx.Graph()
    H.add_nodes_from(order)
    H.add_edges_from(B.edges())
    return H


# =============================================================================
# cmty_semisync_label_propagation
# =============================================================================


class TestSemisyncLabelPropagation:
    def test_matches_networkx_on_random_graphs(self):
        for G in random_graphs(500, seed=1):
            g, index = from_nx(G)
            theirs = sorted(
                sorted(index[v] for v in c) for c in nx.community.label_propagation_communities(G)
            )
            membership = g.cmty_semisync_label_propagation(return_dict=True)
            groups: dict = {}
            for nid, c in membership.items():
                groups.setdefault(c, []).append(nid)
            assert sorted(sorted(c) for c in groups.values()) == theirs

    def test_matches_networkx_on_karate_club(self):
        theirs = nx.community.label_propagation_communities(nx.karate_club_graph())
        membership = ec.karate_club().cmty_semisync_label_propagation(return_dict=True)
        groups: dict = {}
        for nid, c in membership.items():
            groups.setdefault(c, set()).add(nid - 1)
        assert sorted(map(sorted, groups.values())) == sorted(map(sorted, theirs))

    def test_two_cliques(self):
        g = ec.Graph().add_full_graph(n=4).add_full_graph(n=4)
        g.add_edge(4, 5)
        assert g.cmty_semisync_label_propagation(return_dict=True) == {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
        }

    def test_numbered_by_lowest_id_and_deterministic(self):
        g = ec.Graph().add_gnm_graph(n=60, m=150, seed=4)
        first = g.cmty_semisync_label_propagation(return_dict=True)
        assert first == g.cmty_semisync_label_propagation(return_dict=True)
        seen = []
        for nid in sorted(first):
            if first[nid] not in seen:
                seen.append(first[nid])
        assert seen == list(range(len(seen)))

    def test_bipartite_graph_converges(self):
        # synchronous updates would oscillate here; the semi-synchronous rounds settle
        membership = (
            ec.Graph()
            .add_complete_bipartite_graph(5, 7)
            .cmty_semisync_label_propagation(return_dict=True)
        )
        assert len(membership) == 12

    def test_isolated_nodes_and_empty(self):
        g = ec.Graph().add_nodes(["a", "b"]).add_path_graph(3)
        membership = g.cmty_semisync_label_propagation(return_dict=True)
        assert membership[1] != membership[2]
        assert len({membership[3], membership[4], membership[5]}) == 1
        assert ec.Graph().cmty_semisync_label_propagation(return_dict=True) == {}

    def test_ignores_direction_weights_and_parallel_edges(self):
        g = ec.Graph().add_gnm_graph(n=30, m=60, seed=2)
        base = g.cmty_semisync_label_propagation(return_dict=True)
        d = ec.Graph(directed=True, weighted=True).add_nodes([str(i) for i in range(1, 31)])
        for _, s, t in g.get_edges_as_tuples():
            d.add_edge(t, s, weight=7.5)
        d.add_edge(*g.get_edges_as_tuples()[0][1:])
        assert d.cmty_semisync_label_propagation(return_dict=True) == base

    def test_stores_attribute_and_records_step(self):
        g = ec.Graph().add_full_graph(n=4).add_full_graph(n=4)
        assert g.cmty_semisync_label_propagation() is g
        stored = g.get_node_attrs("cmty_semisync_label_propagation")
        assert stored == g.cmty_semisync_label_propagation(return_dict=True)
        g.cmty_semisync_label_propagation(attr_name="lpa")
        assert g.steps[-1].code == "G.cmty_semisync_label_propagation(attr_name='lpa')"


# =============================================================================
# incremental_closeness_centrality
# =============================================================================


class TestIncrementalCloseness:
    @pytest.mark.parametrize("wf_improved", [True, False])
    def test_matches_networkx_and_full_recompute(self, wf_improved):
        rng = random.Random(5)
        checked = 0
        for G in random_graphs(400, seed=6):
            if len(G) < 2:
                continue
            insertion = rng.random() < 0.5 or G.number_of_edges() == 0
            a, b = rng.sample(list(G), 2) if insertion else rng.choice(list(G.edges()))
            g, index = from_nx(G)
            prev = g.closeness_centrality(weighted=False, wf_improved=wf_improved, return_dict=True)
            ours = g.incremental_closeness_centrality(
                (index[a], index[b]), prev_cc=prev, insertion=insertion, wf_improved=wf_improved
            )
            prev_nx = nx.closeness_centrality(G, wf_improved=wf_improved)
            theirs = nx.incremental_closeness_centrality(
                G, (a, b), prev_nx, insertion=insertion, wf_improved=wf_improved
            )
            assert ours == pytest.approx({index[k]: v for k, v in theirs.items()})
            # ... and the same as computing the changed graph from scratch
            H = G.copy()
            if insertion:
                H.add_edge(a, b)
            else:
                H.remove_edge(a, b)
            fresh = from_nx(H)[0].closeness_centrality(
                weighted=False, wf_improved=wf_improved, return_dict=True
            )
            assert ours == pytest.approx(fresh)
            checked += 1
        assert checked > 300

    def test_without_prev_computes_everything(self):
        g = ec.Graph().add_path_graph(6)
        result = g.incremental_closeness_centrality((1, 6))
        assert result == pytest.approx(
            ec.Graph().add_cycle_graph(6).closeness_centrality(return_dict=True)
        )

    def test_graph_is_unchanged(self):
        g = ec.Graph().add_path_graph(5)
        before = (g.count_edges(), len(g.steps))
        prev = g.closeness_centrality(weighted=False, return_dict=True)
        g.incremental_closeness_centrality((1, 5), prev_cc=prev)
        g.incremental_closeness_centrality((2, 3), prev_cc=prev, insertion=False)
        g.incremental_closeness_centrality((2, 3), insertion=False)
        assert (g.count_edges(), len(g.steps)) == before

    def test_labels_and_disconnecting_removal(self):
        g = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("B", "C"), ("C", "D")])
        prev = g.closeness_centrality(weighted=False, return_dict=True)
        result = g.incremental_closeness_centrality(("B", "C"), prev_cc=prev, insertion=False)
        # two separate pairs: each node reaches one other node, one hop away, out of three
        assert result == pytest.approx(dict.fromkeys([1, 2, 3, 4], 1 / 3))

    def test_parallel_edges(self):
        g = ec.Graph().add_path_graph(4)
        g.add_edge(2, 3)
        prev = g.closeness_centrality(weighted=False, return_dict=True)
        # removing one of the two copies changes nothing, and inserting a third neither
        assert g.incremental_closeness_centrality((2, 3), prev, insertion=False) == prev
        assert g.incremental_closeness_centrality((3, 2), prev) == prev

    def test_self_loop_insertion_changes_nothing(self):
        g = ec.Graph().add_path_graph(4)
        prev = g.closeness_centrality(weighted=False, return_dict=True)
        assert g.incremental_closeness_centrality((2, 2), prev) == prev

    def test_errors(self):
        g = ec.Graph().add_path_graph(4)
        with pytest.raises(ValueError, match="no edge"):
            g.incremental_closeness_centrality((1, 3), insertion=False)
        with pytest.raises(ValueError, match="prev_cc"):
            g.incremental_closeness_centrality((1, 3), prev_cc={1: 0.5})
        with pytest.raises(ValueError, match="pair"):
            g.incremental_closeness_centrality(1)
        with pytest.raises(ValueError):
            g.incremental_closeness_centrality((1, 99))
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_path_graph(3).incremental_closeness_centrality((1, 3))


# =============================================================================
# bfs_beam_edges
# =============================================================================


class TestBfsBeamEdges:
    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        rng = random.Random(7)
        for G in random_graphs(400, seed=8, directed=directed):
            g, index = from_nx(G)
            values = {v: rng.choice([0, 1, 2, 3.5]) for v in G}
            by_id = {index[v]: x for v, x in values.items()}
            width = rng.choice([None, 1, 2, 3])
            source = rng.choice(list(G))
            theirs = [
                (index[u], index[v]) for u, v in nx.bfs_beam_edges(G, source, values.get, width)
            ]
            assert g.bfs_beam_edges(index[source], by_id.get, width) == theirs

    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx_with_parallel_edges_and_loops(self, directed):
        rng = random.Random(17)
        for G, g, index in random_multigraphs(300, seed=18, directed=directed):
            values = {v: rng.choice([0, 1, 2]) for v in G}
            by_id = {index[v]: x for v, x in values.items()}
            width = rng.choice([None, 1, 2])
            source = rng.choice(list(G))
            theirs = [
                (index[u], index[v]) for u, v in nx.bfs_beam_edges(G, source, values.get, width)
            ]
            assert g.bfs_beam_edges(index[source], by_id.get, width) == theirs

    def test_value_forms_agree(self):
        g = ec.Graph().add_gnm_graph(n=40, m=90, seed=3)
        scores = {nid: (nid * 7) % 11 for nid in g.get_node_ids(limit=None)}
        g.set_node_attrs("s", scores)
        by_function = g.bfs_beam_edges(1, scores.get, width=2)
        assert g.bfs_beam_edges(1, "s", width=2) == by_function
        assert g.bfs_beam_edges(1, scores, width=2) == by_function

    def test_full_width_is_a_bfs(self):
        g = ec.Graph().add_gnm_graph(n=40, m=90, seed=9)
        edges = g.bfs_beam_edges(1, lambda nid: 0)
        assert edges == g.bfs_edges(1)

    def test_width_one_follows_a_single_path(self):
        g = ec.Graph().add_balanced_tree(2, 4)
        edges = g.bfs_beam_edges(1, lambda nid: -nid, width=1)
        # from each node, the lowest-ID neighbor that is not its parent is never ranked first,
        # because the parent has a lower ID: the beam is spent on the parent
        assert edges == [(1, 2)]
        edges = g.bfs_beam_edges(1, lambda nid: nid, width=1)
        assert [c for _, c in edges] == [3, 7, 15, 31]

    def test_missing_or_bad_scores(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError, match="no score"):
            g.bfs_beam_edges(1, {1: 1, 2: 2})
        with pytest.raises(ValueError, match="number"):
            g.bfs_beam_edges(1, lambda nid: "high")
        with pytest.raises(ValueError, match="NaN"):
            g.bfs_beam_edges(1, lambda nid: float("nan"))
        with pytest.raises(ValueError, match="which no node has"):
            g.bfs_beam_edges(1, "missing")
        with pytest.raises(TypeError):
            g.bfs_beam_edges(1, 5)

    def test_function_errors_propagate(self):
        def boom(nid):
            raise KeyError(nid)

        with pytest.raises(KeyError):
            ec.Graph().add_path_graph(3).bfs_beam_edges(1, boom)

    @pytest.mark.parametrize("width", [0, -1, 1.5, True, "2"])
    def test_bad_width(self, width):
        with pytest.raises(ValueError, match="width"):
            ec.Graph().add_path_graph(3).bfs_beam_edges(1, lambda nid: 0, width=width)

    def test_mode_and_bad_source(self):
        g = ec.Graph(directed=True).add_path_graph(3)
        assert g.bfs_beam_edges(3, lambda nid: 0) == []
        assert g.bfs_beam_edges(3, lambda nid: 0, mode="in") == [(3, 2), (2, 1)]
        with pytest.raises(ValueError):
            g.bfs_beam_edges(9, lambda nid: 0)
        with pytest.raises(ValueError, match="mode"):
            g.bfs_beam_edges(1, lambda nid: 0, mode="up")

    def test_records_traversal_step(self):
        g = ec.Graph().add_path_graph(4)
        before = len(g.steps)
        g.bfs_beam_edges(1, lambda nid: nid, width=1)
        assert len(g.steps) == before + 1
        step = g.steps[-1]
        assert step.kind == KIND_TRAVERSE and step.method == "bfs_beam_edges"
        assert step.code == "G.bfs_beam_edges(1, value=<lambda>, width=1)"


# =============================================================================
# dfs_labeled_edges
# =============================================================================


class TestDfsLabeledEdges:
    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        rng = random.Random(10)
        for G in random_graphs(400, seed=11, directed=directed):
            g, index = from_nx(G)
            for source in (None, rng.choice(list(G))):
                depth = rng.choice([None, 1, 2, 3])
                theirs = [
                    (index[u], index[v], kind)
                    for u, v, kind in nx.dfs_labeled_edges(G, source, depth_limit=depth)
                ]
                ours = g.dfs_labeled_edges(
                    None if source is None else index[source], depth_limit=depth
                )
                assert ours == theirs

    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx_with_parallel_edges_and_loops(self, directed):
        rng = random.Random(19)
        for G, g, index in random_multigraphs(300, seed=20, directed=directed):
            for source in (None, rng.choice(list(G))):
                depth = rng.choice([None, 1, 2])
                theirs = [
                    (index[u], index[v], kind)
                    for u, v, kind in nx.dfs_labeled_edges(G, source, depth_limit=depth)
                ]
                ours = g.dfs_labeled_edges(
                    None if source is None else index[source], depth_limit=depth
                )
                assert ours == theirs

    def test_forward_edges_are_dfs_edges(self):
        g = ec.Graph().add_gnm_graph(n=50, m=80, seed=12)
        for source, depth in [(None, None), (1, None), (5, 2), (None, 0), (3, 0)]:
            steps = g.dfs_labeled_edges(source, depth_limit=depth)
            forward = [(u, v) for u, v, kind in steps if kind == "forward" and u != v]
            assert forward == g.dfs_edges(source, depth_limit=depth)

    def test_depth_limit_zero_reports_only_the_root(self):
        g = ec.Graph().add_path_graph(3)
        assert g.dfs_labeled_edges(2, depth_limit=0) == [(2, 2, "forward"), (2, 2, "reverse")]

    def test_parallel_edges_and_self_loops_looked_at_once(self):
        g = ec.Graph().add_path_graph(2)
        g.add_edge(1, 2).add_edge(2, 2)
        assert g.dfs_labeled_edges(1) == [
            (1, 1, "forward"),
            (1, 2, "forward"),
            (2, 1, "nontree"),
            (2, 2, "nontree"),
            (1, 2, "reverse"),
            (1, 1, "reverse"),
        ]

    def test_mode_all_on_directed(self):
        g = ec.Graph(directed=True).add_nodes(list("abc"))
        g.add_edges([("a", "b"), ("c", "b")])
        triples = g.dfs_labeled_edges("a", mode="all")
        assert [(u, v) for u, v, kind in triples if kind == "forward"] == [(1, 1), (1, 2), (2, 3)]
        assert g.dfs_labeled_edges("a", mode="in") == [(1, 1, "forward"), (1, 1, "reverse")]

    def test_every_tree_is_bracketed(self):
        g = ec.Graph().add_path_graph(2).add_path_graph(2).add_nodes(["x"])
        roots = [u for u, v, kind in g.dfs_labeled_edges() if u == v and kind == "forward"]
        assert roots == [1, 3, 5]

    def test_errors(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError):
            g.dfs_labeled_edges(9)
        with pytest.raises(ValueError, match="depth_limit"):
            g.dfs_labeled_edges(1, depth_limit=-1)
        with pytest.raises(ValueError, match="mode"):
            g.dfs_labeled_edges(1, mode="sideways")

    def test_records_traversal_step(self):
        g = ec.Graph().add_path_graph(4)
        before = len(g.steps)
        g.dfs_labeled_edges(2, depth_limit=1)
        assert len(g.steps) == before + 1
        step = g.steps[-1]
        assert step.kind == KIND_TRAVERSE and step.method == "dfs_labeled_edges"
        assert step.code == "G.dfs_labeled_edges(2, depth_limit=1)"


# =============================================================================
# maximal_extendability
# =============================================================================


class TestMaximalExtendability:
    def test_matches_networkx(self):
        rng = random.Random(13)
        answered = zero = 0
        for _ in range(300):
            B = random_matchable_bipartite(rng)
            g, _ = from_nx(B)
            try:
                theirs = nxb.maximal_extendability(B)
            except nx.NetworkXError as err:
                if "strongly connected" in str(err):
                    assert g.maximal_extendability() == 0
                    zero += 1
                else:
                    with pytest.raises(ValueError):
                        g.maximal_extendability()
                continue
            assert g.maximal_extendability() == theirs
            answered += 1
        assert answered > 100 and zero > 20

    def test_by_brute_force(self):
        # straight from the definition: every set of k independent edges extends
        from itertools import combinations

        def extendable(G, k):
            for chosen in combinations(G.edges(), k):
                ends = [x for e in chosen for x in e]
                if len(set(ends)) < 2 * k:
                    continue
                rest = G.copy()
                rest.remove_nodes_from(ends)
                if len(nx.max_weight_matching(rest, maxcardinality=True)) * 2 != len(rest):
                    return False
            return True

        rng = random.Random(14)
        for _ in range(60):
            B = random_matchable_bipartite(rng)
            if not nx.is_connected(B):
                continue
            g, _ = from_nx(B)
            k = g.maximal_extendability()
            assert extendable(B, k)
            assert k + 1 > len(B) // 2 - 1 or not extendable(B, k + 1)

    @pytest.mark.parametrize(
        "make, expected",
        [
            (lambda: ec.Graph().add_complete_bipartite_graph(3, 3), 2),
            (lambda: ec.Graph().add_complete_bipartite_graph(5, 5), 4),
            (lambda: ec.Graph().add_cycle_graph(8), 1),
            (lambda: ec.Graph().add_named_graph("cubical"), 2),
            (lambda: ec.Graph().add_path_graph(4), 0),
            (lambda: ec.Graph().add_path_graph(2), 0),
        ],
    )
    def test_known_values(self, make, expected):
        assert make().maximal_extendability() == expected

    def test_parallel_edges_ignored(self):
        g = ec.Graph().add_cycle_graph(6)
        g.add_edge(1, 2).add_edge(1, 2)
        assert g.maximal_extendability() == 1

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: ec.Graph(), "no nodes"),
            (lambda: ec.Graph().add_path_graph(2).add_path_graph(2), "not connected"),
            (lambda: ec.Graph().add_cycle_graph(5), "not bipartite"),
            (lambda: ec.Graph().add_path_graph(2).add_edge(1, 1), "not bipartite"),
            (lambda: ec.Graph().add_path_graph(3), "perfect matching"),
            (lambda: ec.Graph().add_complete_bipartite_graph(1, 3), "perfect matching"),
            (lambda: ec.Graph(directed=True).add_cycle_graph(4), "undirected"),
        ],
    )
    def test_errors(self, make, match):
        with pytest.raises(ValueError, match=match):
            make().maximal_extendability()
