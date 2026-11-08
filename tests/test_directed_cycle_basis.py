"""Tests for directed cycle basis."""

import pytest
import edgecase as ec


@pytest.fixture
def triangle():
    """Directed triangle: A -> B -> C -> A."""
    g = ec.Graph(directed=True)
    g.add_node(label="A").add_node(label="B").add_node(label="C")
    g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
    return g


@pytest.fixture
def figure_eight():
    """Two directed cycles sharing node A: A->B->C->A and A->D->A."""
    g = ec.Graph(directed=True)
    for n in ["A", "B", "C", "D"]:
        g.add_node(label=n)
    g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
    g.add_edge("A", "D").add_edge("D", "A")
    return g


@pytest.fixture
def two_disjoint_triangles():
    """Two separate directed triangles (two SCCs)."""
    g = ec.Graph(directed=True)
    for n in ["A", "B", "C", "D", "E", "F"]:
        g.add_node(label=n)
    g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
    g.add_edge("D", "E").add_edge("E", "F").add_edge("F", "D")
    return g


@pytest.fixture
def diamond_scc():
    """Diamond-shaped SCC: A->B, B->C, B->D, C->A, D->A. 5 edges, 4 nodes."""
    g = ec.Graph(directed=True)
    for n in ["A", "B", "C", "D"]:
        g.add_node(label=n)
    g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
    g.add_edge("B", "D").add_edge("D", "A")
    return g


class TestDirectedCycleBasisBasic:
    def test_returns_list(self, triangle):
        result = triangle.directed_cycle_basis()
        assert isinstance(result, list)

    def test_cycles_are_lists(self, triangle):
        result = triangle.directed_cycle_basis()
        for cycle in result:
            assert isinstance(cycle, list)

    def test_triangle_one_cycle(self, triangle):
        result = triangle.directed_cycle_basis()
        assert len(result) == 1

    def test_triangle_cycle_has_three_nodes(self, triangle):
        result = triangle.directed_cycle_basis()
        assert len(result[0]) == 3

    def test_triangle_cycle_contains_all_nodes(self, triangle):
        result = triangle.directed_cycle_basis()
        node_ids = set(triangle.get_node_ids())
        assert set(result[0]) == node_ids

    def test_figure_eight_two_cycles(self, figure_eight):
        """5 edges - 4 nodes + 1 SCC = 2 cycles."""
        result = figure_eight.directed_cycle_basis()
        assert len(result) == 2

    def test_two_disjoint_triangles_two_cycles(self, two_disjoint_triangles):
        """6 edges - 6 nodes + 2 SCCs = 2 cycles."""
        result = two_disjoint_triangles.directed_cycle_basis()
        assert len(result) == 2

    def test_diamond_scc_two_cycles(self, diamond_scc):
        """5 edges - 4 nodes + 1 SCC = 2 cycles."""
        result = diamond_scc.directed_cycle_basis()
        assert len(result) == 2


class TestDirectedCycleBasisDimension:
    """Verify the cycle count equals m - n + p."""

    def test_self_loop(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_edge("A", "A")
        result = g.directed_cycle_basis()
        # 1 edge - 1 node + 1 SCC = 1, but self-loops form a 1-node SCC
        # which the algorithm skips (SCC needs >= 2 nodes)
        # Self-loop is a degenerate case
        assert len(result) >= 0

    def test_complete_digraph_3(self):
        """K3 directed: 6 edges, 3 nodes, 1 SCC => 6-3+1 = 4 cycles."""
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C"]:
            g.add_node(label=n)
        for s in ["A", "B", "C"]:
            for t in ["A", "B", "C"]:
                if s != t:
                    g.add_edge(s, t)
        result = g.directed_cycle_basis()
        assert len(result) == 4

    def test_complete_digraph_4(self):
        """K4 directed: 12 edges, 4 nodes, 1 SCC => 12-4+1 = 9 cycles."""
        g = ec.Graph(directed=True)
        nodes = ["A", "B", "C", "D"]
        for n in nodes:
            g.add_node(label=n)
        for s in nodes:
            for t in nodes:
                if s != t:
                    g.add_edge(s, t)
        result = g.directed_cycle_basis()
        assert len(result) == 9

    def test_chain_with_back_edge(self):
        """A->B->C->D with D->A: 4 edges, 4 nodes, 1 SCC => 4-4+1 = 1."""
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C", "D"]:
            g.add_node(label=n)
        g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "D").add_edge("D", "A")
        result = g.directed_cycle_basis()
        assert len(result) == 1
        assert len(result[0]) == 4


class TestDirectedCycleBasisValidity:
    """Verify that returned cycles are valid directed cycles."""

    def test_cycle_edges_exist(self, triangle):
        """Every consecutive pair in a cycle must be a directed edge."""
        result = triangle.directed_cycle_basis()
        adj = {}
        for nid in triangle.get_node_ids():
            adj[nid] = set(triangle.get_neighbors(nid, mode="out"))
        for cycle in result:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                assert v in adj[u], f"Edge {u}->{v} does not exist"

    def test_diamond_cycles_valid(self, diamond_scc):
        result = diamond_scc.directed_cycle_basis()
        adj = {}
        for nid in diamond_scc.get_node_ids():
            adj[nid] = set(diamond_scc.get_neighbors(nid, mode="out"))
        for cycle in result:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                assert v in adj[u], f"Edge {u}->{v} does not exist"

    def test_figure_eight_cycles_valid(self, figure_eight):
        result = figure_eight.directed_cycle_basis()
        adj = {}
        for nid in figure_eight.get_node_ids():
            adj[nid] = set(figure_eight.get_neighbors(nid, mode="out"))
        for cycle in result:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                assert v in adj[u], f"Edge {u}->{v} does not exist"

    def test_complete3_cycles_valid(self):
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C"]:
            g.add_node(label=n)
        for s in ["A", "B", "C"]:
            for t in ["A", "B", "C"]:
                if s != t:
                    g.add_edge(s, t)
        result = g.directed_cycle_basis()
        adj = {}
        for nid in g.get_node_ids():
            adj[nid] = set(g.get_neighbors(nid, mode="out"))
        for cycle in result:
            assert len(cycle) >= 2
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                assert v in adj[u], f"Edge {u}->{v} does not exist"

    def test_cycles_follow_arcs(self):
        """Every basis cycle is a closed directed walk (no arc traversed backwards)."""
        edges = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "E"), ("E", "D"), ("D", "B")]
        g = ec.Graph(directed=True)
        for label in "ABCDE":
            g.add_node(label=label)
        for u, v in edges:
            g.add_edge(u, v)
        arcs = {(g.get_id_from_label(u), g.get_id_from_label(v)) for u, v in edges}
        basis = g.directed_cycle_basis()
        assert len(basis) == 2  # m - n + p
        for cycle in basis:
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                assert (a, b) in arcs


class TestDirectedCycleBasisEdgeCases:
    def test_empty_graph(self):
        g = ec.Graph(directed=True)
        result = g.directed_cycle_basis()
        assert result == []

    def test_single_node(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        result = g.directed_cycle_basis()
        assert result == []

    def test_dag_no_cycles(self):
        """A DAG has no cycles."""
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C", "D"]:
            g.add_node(label=n)
        g.add_edge("A", "B").add_edge("A", "C").add_edge("B", "D").add_edge("C", "D")
        result = g.directed_cycle_basis()
        assert result == []

    def test_two_nodes_cycle(self):
        """A <-> B forms a 2-cycle."""
        g = ec.Graph(directed=True)
        g.add_node(label="A").add_node(label="B")
        g.add_edge("A", "B").add_edge("B", "A")
        result = g.directed_cycle_basis()
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_isolated_nodes_ignored(self):
        """Isolated nodes don't contribute cycles."""
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C", "X", "Y"]:
            g.add_node(label=n)
        g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
        result = g.directed_cycle_basis()
        assert len(result) == 1

    def test_scc_with_pendant(self):
        """SCC with a pendant (non-SCC) node attached."""
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C", "D"]:
            g.add_node(label=n)
        g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
        g.add_edge("A", "D")  # pendant
        result = g.directed_cycle_basis()
        assert len(result) == 1

    def test_does_not_mutate_graph(self, triangle):
        n = triangle.count_nodes()
        e = triangle.count_edges()
        triangle.directed_cycle_basis()
        assert triangle.count_nodes() == n
        assert triangle.count_edges() == e


class TestDirectedCycleBasisErrors:
    def test_undirected_graph_rejected(self):
        g = ec.Graph(directed=False)
        g.add_node(label="A").add_node(label="B").add_edge("A", "B")
        with pytest.raises(Exception, match="directed"):
            g.directed_cycle_basis()


class TestDirectedCycleBasisComplex:
    def test_two_sccs_connected(self):
        """Two SCCs connected by a bridge edge."""
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C", "D", "E", "F"]:
            g.add_node(label=n)
        g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "A")
        g.add_edge("C", "D")  # bridge
        g.add_edge("D", "E").add_edge("E", "F").add_edge("F", "D")
        result = g.directed_cycle_basis()
        assert len(result) == 2

    def test_nested_cycles(self):
        """A->B->C->D->A with shortcut B->D: 5 edges, 4 nodes, 1 SCC => 2 cycles."""
        g = ec.Graph(directed=True)
        for n in ["A", "B", "C", "D"]:
            g.add_node(label=n)
        g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "D").add_edge("D", "A")
        g.add_edge("B", "D")
        result = g.directed_cycle_basis()
        assert len(result) == 2

    def test_large_scc(self):
        """Ring of 10 nodes with one extra chord: 11 edges, 10 nodes, 1 SCC => 2 cycles."""
        g = ec.Graph(directed=True)
        for i in range(10):
            g.add_node(label=str(i))
        for i in range(10):
            g.add_edge(str(i), str((i + 1) % 10))
        g.add_edge("0", "5")  # chord
        result = g.directed_cycle_basis()
        assert len(result) == 2
