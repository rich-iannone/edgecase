"""Tests for directed Steiner tree (Steiner arborescence)."""

import pytest
import edgecase as ec


@pytest.fixture
def diamond_dag():
    """R -> A -> B, R -> C (expensive), A -> C (cheap), C -> D."""
    g = ec.Graph(directed=True, weighted=True)
    for n in ["R", "A", "B", "C", "D"]:
        g.add_node(label=n)
    g.add_edge("R", "A", weight=1.0)
    g.add_edge("A", "B", weight=2.0)
    g.add_edge("R", "C", weight=5.0)
    g.add_edge("A", "C", weight=1.0)
    g.add_edge("C", "D", weight=1.0)
    return g


@pytest.fixture
def chain():
    """Linear chain: 1 -> 2 -> 3 -> 4 -> 5."""
    g = ec.Graph(directed=True, weighted=True)
    for i in range(1, 6):
        g.add_node(label=str(i))
    for i in range(1, 5):
        g.add_edge(str(i), str(i + 1), weight=1.0)
    return g


@pytest.fixture
def branching():
    """Branching DAG: R -> A -> B, R -> A -> C, R -> D -> E, D -> F."""
    g = ec.Graph(directed=True, weighted=True)
    for n in ["R", "A", "B", "C", "D", "E", "F"]:
        g.add_node(label=n)
    g.add_edge("R", "A", weight=1.0)
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("A", "C", weight=1.0)
    g.add_edge("R", "D", weight=2.0)
    g.add_edge("D", "E", weight=1.0)
    g.add_edge("D", "F", weight=1.0)
    return g


class TestDirectedSteinerTree:
    def test_returns_subgraph(self, diamond_dag):
        tree = diamond_dag.directed_steiner_tree("R", ["B", "D"])
        assert isinstance(tree, ec.Graph)
        assert tree.is_directed()

    def test_returns_edge_list(self, diamond_dag):
        edges = diamond_dag.directed_steiner_tree("R", ["B", "D"], return_subgraph=False)
        assert isinstance(edges, list)
        assert all(len(e) == 3 for e in edges)

    def test_all_terminals_reachable(self, diamond_dag):
        tree = diamond_dag.directed_steiner_tree("R", ["B", "D"])
        edge_list = diamond_dag.directed_steiner_tree("R", ["B", "D"], return_subgraph=False)
        node_labels = set()
        for s, t, _ in edge_list:
            node_labels.add(s)
            node_labels.add(t)
        assert "B" in node_labels
        assert "D" in node_labels
        assert "R" in node_labels

    def test_prefers_shared_path(self, diamond_dag):
        """Should route through A->C (cost 2) rather than R->C (cost 5)."""
        edges = diamond_dag.directed_steiner_tree("R", ["B", "D"], return_subgraph=False)
        edge_set = {(s, t) for s, t, _ in edges}
        assert ("A", "C") in edge_set, "Should use cheaper A->C path"
        assert ("R", "C") not in edge_set, "Should not use expensive R->C"

    def test_chain_all_edges(self, chain):
        """Reaching terminals 3 and 5 from 1 requires the full chain."""
        edges = chain.directed_steiner_tree("1", ["3", "5"], return_subgraph=False)
        assert len(edges) == 4

    def test_chain_single_terminal(self, chain):
        edges = chain.directed_steiner_tree("1", ["3"], return_subgraph=False)
        edge_set = {(s, t) for s, t, _ in edges}
        assert ("1", "2") in edge_set
        assert ("2", "3") in edge_set
        assert len(edges) == 2

    def test_branching_two_branches(self, branching):
        """Reaching B and E requires both branches from R."""
        edges = branching.directed_steiner_tree("R", ["B", "E"], return_subgraph=False)
        node_labels = set()
        for s, t, _ in edges:
            node_labels.add(s)
            node_labels.add(t)
        assert "R" in node_labels
        assert "B" in node_labels
        assert "E" in node_labels

    def test_root_is_terminal(self, chain):
        """Root as terminal should be handled gracefully."""
        edges = chain.directed_steiner_tree("1", ["1", "3"], return_subgraph=False)
        edge_set = {(s, t) for s, t, _ in edges}
        assert ("1", "2") in edge_set
        assert ("2", "3") in edge_set

    def test_single_terminal(self, chain):
        edges = chain.directed_steiner_tree("1", ["2"], return_subgraph=False)
        assert len(edges) == 1
        assert edges[0][:2] == ("1", "2")

    def test_edge_weights_preserved(self, diamond_dag):
        edges = diamond_dag.directed_steiner_tree("R", ["B"], return_subgraph=False)
        weight_map = {(s, t): w for s, t, w in edges}
        assert weight_map[("R", "A")] == 1.0
        assert weight_map[("A", "B")] == 2.0

    def test_subgraph_is_directed(self, diamond_dag):
        tree = diamond_dag.directed_steiner_tree("R", ["B", "D"])
        assert tree.is_directed()

    def test_does_not_mutate_original(self, diamond_dag):
        n = diamond_dag.count_nodes()
        e = diamond_dag.count_edges()
        diamond_dag.directed_steiner_tree("R", ["B", "D"])
        assert diamond_dag.count_nodes() == n
        assert diamond_dag.count_edges() == e


class TestDirectedSteinerTreeErrors:
    def test_undirected_graph_rejected(self):
        g = ec.Graph(directed=False)
        g.add_node(label="A").add_node(label="B").add_edge("A", "B")
        with pytest.raises(Exception, match="directed"):
            g.directed_steiner_tree("A", ["B"])

    def test_missing_root(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        with pytest.raises(Exception, match="not found"):
            g.directed_steiner_tree("Z", ["A"])

    def test_missing_terminal(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A").add_node(label="B")
        g.add_edge("A", "B")
        with pytest.raises(Exception, match="not found"):
            g.directed_steiner_tree("A", ["Z"])

    def test_unreachable_terminal(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A").add_node(label="B").add_node(label="C")
        g.add_edge("A", "B")
        with pytest.raises(Exception, match="not reachable"):
            g.directed_steiner_tree("A", ["C"])

    def test_empty_terminals(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        with pytest.raises(Exception):
            g.directed_steiner_tree("A", [])


class TestDirectedSteinerTreeEdgeCases:
    def test_two_nodes(self):
        g = ec.Graph(directed=True, weighted=True)
        g.add_node(label="A").add_node(label="B")
        g.add_edge("A", "B", weight=3.0)
        edges = g.directed_steiner_tree("A", ["B"], return_subgraph=False)
        assert len(edges) == 1
        assert edges[0] == ("A", "B", 3.0)

    def test_duplicate_terminals(self):
        """Duplicate terminal labels should be deduplicated."""
        g = ec.Graph(directed=True, weighted=True)
        g.add_node(label="A").add_node(label="B")
        g.add_edge("A", "B", weight=1.0)
        edges = g.directed_steiner_tree("A", ["B", "B"], return_subgraph=False)
        assert len(edges) == 1

    def test_unweighted_graph(self):
        """Unweighted edges should default to weight 1.0."""
        g = ec.Graph(directed=True)
        g.add_node(label="A").add_node(label="B").add_node(label="C")
        g.add_edge("A", "B").add_edge("B", "C")
        edges = g.directed_steiner_tree("A", ["C"], return_subgraph=False)
        total_weight = sum(w for _, _, w in edges)
        assert total_weight == 2.0

    def test_parallel_paths_picks_cheaper(self):
        """With two paths to a terminal, should pick the cheaper one."""
        g = ec.Graph(directed=True, weighted=True)
        for n in ["R", "A", "B", "T"]:
            g.add_node(label=n)
        g.add_edge("R", "A", weight=1.0)
        g.add_edge("A", "T", weight=1.0)
        g.add_edge("R", "B", weight=10.0)
        g.add_edge("B", "T", weight=10.0)
        edges = g.directed_steiner_tree("R", ["T"], return_subgraph=False)
        total_weight = sum(w for _, _, w in edges)
        assert total_weight == 2.0

    def test_edge_reuse_reduces_cost(self):
        """Shared prefix should be reused, not duplicated."""
        g = ec.Graph(directed=True, weighted=True)
        for n in ["R", "M", "T1", "T2"]:
            g.add_node(label=n)
        g.add_edge("R", "M", weight=10.0)
        g.add_edge("M", "T1", weight=1.0)
        g.add_edge("M", "T2", weight=1.0)
        tree = g.directed_steiner_tree("R", ["T1", "T2"])
        assert tree.count_edges() == 3
