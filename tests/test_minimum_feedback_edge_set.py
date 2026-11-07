"""Tests for `minimum_feedback_edge_set()` (NetworkX PR #8521)."""

import itertools
import random

import edgecase as ec
import networkx as nx
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _graph(edges, nodes=(), directed=True, weighted=False):
    """An EdgeCase graph on string labels; edges are (u, v) or (u, v, weight)."""
    labels = list(dict.fromkeys([str(v) for v in nodes] + [str(x) for e in edges for x in e[:2]]))
    g = ec.Graph(directed=directed, weighted=weighted).add_nodes(labels)
    if edges:
        g.add_edges([(str(e[0]), str(e[1]), *e[2:]) for e in edges])
    return g


def _pairs(g, ids):
    """The removed edges as label pairs."""
    out = []
    for e in ids:
        s, t = g.get_node_ids_by_edge_id(e)
        out.append((g.get_label_from_id(s), g.get_label_from_id(t)))
    return out


def _edges(g, weighted):
    """{edge ID: (source, target, cost)} on the graph's node IDs."""
    out = {}
    for i in g.get_edge_ids(limit=None):
        e = g.get_edge(i)
        out[i] = (e.source, e.target, e.weight if weighted and e.weight is not None else 1.0)
    return out


def _acyclic(g, edges):
    """Whether the edges (source, target, cost) form a forest / DAG on g's nodes."""
    G = (nx.MultiDiGraph if g.is_directed() else nx.MultiGraph)()
    G.add_nodes_from(g.get_node_ids(limit=None))
    G.add_edges_from((s, t) for s, t, _ in edges)
    if g.is_directed():
        return nx.is_directed_acyclic_graph(G)
    return G.number_of_edges() == 0 or nx.is_forest(G)


def _check(g, weighted=None):
    """Assert the result breaks every cycle and is minimal; return (cost, optimum or None)."""
    use = g.is_weighted() if weighted is None else weighted
    found = g.minimum_feedback_edge_set(weighted=weighted)
    assert found == sorted(set(found))
    E = _edges(g, use)
    rest = [E[i] for i in E if i not in found]
    assert _acyclic(g, rest)
    for f in found:
        assert not _acyclic(g, rest + [E[f]])
    cost = sum(E[i][2] for i in found)
    best = None
    if len(E) <= 12:
        best = min(
            sum(E[i][2] for i in S)
            for k in range(len(E) + 1)
            for S in itertools.combinations(E, k)
            if _acyclic(g, [E[i] for i in E if i not in S])
        )
    return cost, best


def _random_graph(seed, directed, weighted, n_max=7, m_max=11):
    rng = random.Random(seed)
    n = rng.randint(1, n_max)
    g = ec.Graph(directed=directed, weighted=weighted).add_nodes([str(i) for i in range(n)])
    ids = sorted(g.get_node_ids(limit=None))
    for _ in range(rng.randint(0, m_max)):
        w = float(rng.randint(0, 5)) if weighted else None
        g.add_edge(rng.choice(ids), rng.choice(ids), weight=w)
    return g


# =============================================================================
# Known answers (ported from the PR)
# =============================================================================


class TestKnownAnswers:
    def test_paper_figure_2(self):
        g = _graph(
            [(2, 1, 4.0), (1, 5, 5.0), (5, 2, 6.0), (2, 4, 5.0), (4, 5, 7.0), (4, 3, 2.0),
             (3, 2, 6.0)],
            weighted=True,
        )
        assert sorted(_pairs(g, g.minimum_feedback_edge_set())) == [("4", "3"), ("5", "2")]

    def test_small_complete_digraph(self):
        G = nx.complete_graph(3, create_using=nx.DiGraph)
        g = _graph(G.edges())
        assert len(g.minimum_feedback_edge_set()) == 3  # one edge of each opposite pair
        assert _check(g) == (3.0, 3.0)

    def test_5_cycle(self):
        edges = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 2), (3, 4), (4, 2)]
        g = _graph(edges)
        assert _pairs(g, g.minimum_feedback_edge_set()) in ([("0", "1")], [("2", "0")])
        g = _graph([(*e, 10.0 if e == (2, 0) else 1.0) for e in edges], weighted=True)
        assert _pairs(g, g.minimum_feedback_edge_set()) == [("0", "1")]

    def test_ties(self):
        G = nx.cycle_graph(5, create_using=nx.DiGraph)
        assert len(_graph(G.edges()).minimum_feedback_edge_set()) == 1

    @pytest.mark.parametrize("directed", [True, False])
    def test_complete_graphs(self, directed):
        G = nx.complete_graph(10, create_using=nx.DiGraph if directed else nx.Graph)
        g = _graph(G.edges(), directed=directed)
        found = g.minimum_feedback_edge_set()
        # undirected: all but a spanning tree; directed: one edge of each opposite pair
        assert len(found) == (45 if directed else 45 - 9)
        _check(g)

    def test_docstring_examples(self):
        tri = _graph([("A", "B", 3.0), ("B", "C", 1.0), ("C", "A", 2.0)], directed=False,
                     weighted=True)
        assert _pairs(tri, tri.minimum_feedback_edge_set()) == [("B", "C")]
        deps = _graph([("A", "B"), ("B", "C"), ("C", "A"), ("B", "D"), ("D", "C"), ("D", "E"),
                       ("E", "C")])
        assert _pairs(deps, deps.minimum_feedback_edge_set()) == [("C", "A")]


# =============================================================================
# Exactness (undirected) and quality (directed) against brute force
# =============================================================================


class TestAgainstBruteForce:
    @pytest.mark.parametrize("seed", range(60))
    def test_undirected_is_optimal(self, seed):
        cost, best = _check(_random_graph(seed, directed=False, weighted=seed % 2 == 0))
        assert best is None or cost == pytest.approx(best)

    @pytest.mark.parametrize("seed", range(60))
    def test_directed_is_minimal_and_close(self, seed):
        g = _random_graph(seed, directed=True, weighted=seed % 2 == 0)
        cost, best = _check(g)
        if best is not None and best > 0:
            # the local-ratio bound: at most the optimum times the longest cycle length
            assert cost <= best * g.count_nodes() + 1e-9

    def test_directed_mostly_optimal(self):
        hits = total = 0
        for seed in range(80):
            cost, best = _check(_random_graph(1000 + seed, directed=True, weighted=True))
            if best is not None:
                total += 1
                hits += cost == pytest.approx(best)
        assert hits >= 0.9 * total


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    @pytest.mark.parametrize("directed", [True, False])
    def test_empty_and_acyclic(self, directed):
        assert ec.Graph(directed=directed).minimum_feedback_edge_set() == []
        assert _graph([(0, 1), (1, 2), (0, 2)], directed=directed).minimum_feedback_edge_set() == (
            [] if directed else [3]
        )

    @pytest.mark.parametrize("directed", [True, False])
    def test_self_loop_always_removed(self, directed):
        g = _graph([(0, 0), (0, 1)], directed=directed)
        assert _pairs(g, g.minimum_feedback_edge_set()) == [("0", "0")]

    def test_undirected_parallel_edges_keep_heaviest(self):
        g = _graph([(0, 1, 1.0), (0, 1, 5.0), (1, 0, 2.0)], directed=False, weighted=True)
        found = g.minimum_feedback_edge_set()
        assert sorted(g.get_edge(e).weight for e in found) == [1.0, 2.0]

    def test_directed_two_cycle(self):
        g = _graph([(0, 1, 3.0), (1, 0, 2.0)], weighted=True)
        assert _pairs(g, g.minimum_feedback_edge_set()) == [("1", "0")]

    def test_weighted_flag(self):
        # 0 -> 1 lies on both cycles: alone it suffices, but its weight 9 loses to 1 + 1
        g = _graph([(0, 1, 9.0), (1, 0, 1.0), (1, 2, 1.0), (2, 0, 1.0)], weighted=True)
        assert _pairs(g, g.minimum_feedback_edge_set(weighted=False)) == [("0", "1")]
        found = g.minimum_feedback_edge_set()
        assert sum(g.get_edge(e).weight for e in found) == pytest.approx(2.0)

    def test_zero_weight_edges_go_first(self):
        g = _graph([(0, 1, 0.0), (1, 2, 4.0), (2, 0, 4.0)], weighted=True)
        assert _pairs(g, g.minimum_feedback_edge_set()) == [("0", "1")]

    def test_larger_random_digraph(self):
        rng = random.Random(5)
        g = _graph([(rng.randrange(300), rng.randrange(300)) for _ in range(1200)], range(300))
        _check(g)


# =============================================================================
# Validation and steps
# =============================================================================


class TestValidationAndSteps:
    @pytest.mark.parametrize("directed", [True, False])
    @pytest.mark.parametrize("bad", [-1.0, float("nan")])
    def test_bad_weights(self, directed, bad):
        if bad != bad:
            with pytest.raises(ValueError, match="NaN"):  # rejected on entry
                _graph([(0, 1, bad), (1, 0, 1.0)], directed=directed, weighted=True)
            return
        g = _graph([(0, 1, bad), (1, 0, 1.0)], directed=directed, weighted=True)
        with pytest.raises(ValueError, match="non-negative"):
            g.minimum_feedback_edge_set()
        assert len(g.minimum_feedback_edge_set(weighted=False)) == 1

    def test_does_not_modify_graph(self):
        g = _graph([(0, 1), (1, 2), (2, 0)])
        before = sorted(g.get_edge_ids(limit=None))
        g.minimum_feedback_edge_set()
        assert sorted(g.get_edge_ids(limit=None)) == before

    def test_steps(self):
        g = _graph([("a", "b", 2.0), ("b", "c", 1.5), ("c", "a", 4.0), ("c", "b", 1.0)],
                   weighted=True)
        found = g.minimum_feedback_edge_set()
        step = g.steps[-1]
        assert step.code == "G.minimum_feedback_edge_set()"
        assert step.result == found
        assert sorted(step.delta.edges_selected) == sorted(_pairs(g, found))
        assert step.narration.startswith(f"Removing {len(found)} edge")
        _graph([(0, 1)]).minimum_feedback_edge_set()
        dag = _graph([(0, 1)])
        dag.minimum_feedback_edge_set()
        assert dag.steps[-1].narration == "The graph has no cycle; nothing to remove."
