"""Tests for `reachability_index()`, the pruned-landmark-labeling reachability index
(NetworkX PR #8718).
"""

import random

import edgecase as ec
import networkx as nx
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _digraph(edges, nodes=()):
    """A directed EdgeCase graph on string labels."""
    labels = list(dict.fromkeys([str(v) for v in nodes] + [str(x) for e in edges for x in e]))
    g = ec.Graph(directed=True).add_nodes(labels)
    if edges:
        g.add_edges([(str(u), str(v)) for u, v in edges])
    return g


def _from_nx(G):
    """The NetworkX graph `G` as an EdgeCase graph with string labels."""
    return _digraph(G.edges(), G.nodes()) if G.is_directed() else _undirected(G)


def _undirected(G):
    g = ec.Graph().add_nodes([str(v) for v in G])
    if G.number_of_edges():
        g.add_edges([(str(u), str(v)) for u, v in G.edges()])
    return g


def _mismatches(g):
    """All (s, t) pairs where the index disagrees with has_path()."""
    index = g.reachability_index()
    labels = [g.get_label_from_id(i) for i in sorted(g.get_node_ids(limit=None))]
    return [(s, t) for s in labels for t in labels if index(s, t) != g.has_path(s, t)]


# =============================================================================
# Known answers (ported from the PR)
# =============================================================================


class TestKnownAnswers:
    def test_simple_path(self):
        index = _digraph([(0, 1), (1, 2), (2, 3)]).reachability_index()
        assert index("0", "3") is True
        assert index("3", "0") is False
        assert index("1", "2") is True
        assert index("0", "0") is True

    def test_branching_dag(self):
        index = _digraph([(0, 1), (0, 2), (1, 3), (2, 3)]).reachability_index()
        assert index("0", "3") is True
        assert index("1", "2") is False
        assert index("3", "0") is False

    def test_cycle_is_mutually_reachable(self):
        index = _digraph([(0, 1), (1, 2), (2, 0), (2, 3)]).reachability_index()
        assert index("0", "2") is True
        assert index("2", "0") is True
        assert index("1", "3") is True
        assert index("3", "0") is False
        assert index.n_components == 2

    def test_ids_and_labels(self):
        g = _digraph([("a", "b"), ("b", "c")])
        index = g.reachability_index()
        a, c = (g._resolve_node_id(x) for x in "ac")
        assert index(a, c) is True
        assert index("c", a) is False
        assert index(a, "c") is True

    def test_docstring_example(self):
        g = _digraph([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")], nodes="ABCDE")
        index = g.reachability_index()
        assert (index("B", "A"), index("A", "D"), index("D", "A"), index("A", "E")) == (
            True,
            True,
            False,
            False,
        )
        assert index.n_components == 3


# =============================================================================
# Agreement with has_path()
# =============================================================================


class TestCorrectness:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("acyclic", [True, False])
    def test_exhaustive_vs_has_path(self, seed, acyclic):
        H = nx.gnp_random_graph(40, 0.07, seed=seed, directed=True)
        edges = [(u, v) for u, v in H.edges() if u < v or not acyclic]
        assert _mismatches(_digraph(edges, range(40))) == []

    @pytest.mark.parametrize("density", [0.02, 0.05, 0.15, 0.40])
    def test_exhaustive_varying_density(self, density):
        G = nx.gnp_random_graph(50, density, seed=11, directed=True)
        assert _mismatches(_from_nx(G)) == []

    @pytest.mark.parametrize("seed", range(30))
    def test_random_multigraphs_with_self_loops(self, seed):
        rng = random.Random(seed)
        n = rng.randrange(1, 25)
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randrange(3 * n))]
        assert _mismatches(_digraph(edges, range(n))) == []

    @pytest.mark.parametrize("seed", range(6))
    def test_undirected(self, seed):
        G = nx.gnp_random_graph(30, 0.06, seed=seed)
        g = _from_nx(G)
        assert _mismatches(g) == []
        assert g.reachability_index().n_components == nx.number_connected_components(G)

    def test_condensation_size(self):
        G = nx.gnp_random_graph(60, 0.04, seed=3, directed=True)
        index = _from_nx(G).reachability_index()
        assert index.n_components == nx.number_strongly_connected_components(G)


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    def test_empty_graph(self):
        index = ec.Graph(directed=True).reachability_index()
        assert index.n_components == 0
        assert index.label_size == 0

    def test_single_node(self):
        assert _digraph([], [0]).reachability_index()("0", "0") is True

    def test_self_loop(self):
        index = _digraph([(0, 0), (0, 1)]).reachability_index()
        assert index("0", "1") is True
        assert index("1", "0") is False
        assert index("0", "0") is True

    def test_isolated_nodes(self):
        index = _digraph([], range(5)).reachability_index()
        assert index("0", "0") is True
        assert index("0", "1") is False
        assert index.n_components == 5

    def test_disconnected_components(self):
        index = _digraph([(0, 1), (1, 2), (10, 11)]).reachability_index()
        assert index("0", "2") is True
        assert index("0", "10") is False
        assert index("10", "11") is True

    def test_single_scc(self):
        index = _digraph([(i, (i + 1) % 6) for i in range(6)]).reachability_index()
        assert all(index(str(s), str(t)) for s in range(6) for t in range(6))
        assert index.n_components == 1

    def test_complete_digraph(self):
        G = nx.complete_graph(5, create_using=nx.DiGraph)
        assert _mismatches(_from_nx(G)) == []

    def test_hub_first_keeps_labels_small(self):
        # 50 sources -> hub -> 50 sinks: the hub is processed first and covers all 2,500
        # source-to-sink pairs, so each leaf only needs itself and the hub as hubs.
        edges = [(f"s{i}", "hub") for i in range(50)] + [("hub", f"t{i}") for i in range(50)]
        index = _digraph(edges).reachability_index()
        assert index("s3", "t7") is True
        assert index("t7", "s3") is False
        assert index.label_size <= 2 + 3 * 100


# =============================================================================
# Validation and steps
# =============================================================================


class TestValidation:
    def test_missing_label_raises(self):
        index = _digraph([(0, 1)]).reachability_index()
        with pytest.raises(ValueError):
            index("99", "1")
        with pytest.raises(ValueError):
            index("0", "99")

    def test_missing_id_raises(self):
        index = _digraph([(0, 1)]).reachability_index()
        with pytest.raises(ValueError, match="not in the reachability index"):
            index(10**12, 10**12 + 1)

    def test_node_added_later_raises(self):
        g = _digraph([(0, 1)])
        index = g.reachability_index()
        g.add_node("new")
        with pytest.raises(ValueError, match="rebuild it"):
            index("0", "new")

    def test_snapshot_ignores_later_edges(self):
        g = _digraph([(0, 1)], [2])
        index = g.reachability_index()
        g.add_edge("1", "2")
        assert index("0", "2") is False
        assert g.reachability_index()("0", "2") is True

    def test_steps(self):
        g = _digraph([(0, 1), (1, 2), (2, 0), (2, 3)])
        index = g.reachability_index()
        step = g.steps[-1]
        assert step.code == "G.reachability_index()"
        assert step.narration == (
            f"Reachability index over 2 components with {index.label_size} hub entries."
        )
        n_steps = len(g.steps)
        index("0", "3")
        assert len(g.steps) == n_steps
