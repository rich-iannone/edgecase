import pytest
import edgecase as ec


class TestDensity:
    """Test density() method."""

    def test_density_empty_graph(self):
        """Empty graph has density 0."""
        G = ec.Graph()
        assert G.density() == 0.0

    def test_density_single_node(self):
        """Single node has density 0."""
        G = ec.Graph()
        G.add_node("A")
        assert G.density() == 0.0

    def test_density_complete_graph_undirected(self):
        """Complete undirected graph has density 1.0."""
        G = ec.Graph()
        G.add_full_graph(4)
        assert G.density() == 1.0

    def test_density_complete_graph_directed(self):
        """Complete directed graph has density 1.0."""
        G = ec.Graph(directed=True)
        nodes = ["A", "B", "C"]
        for node in nodes:
            G.add_node(node)
        # Add all possible directed edges (including both directions)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    G.add_edge(nodes[i], nodes[j])
        assert G.density() == 1.0

    def test_density_sparse_graph(self):
        """Sparse graph has low density."""
        G = ec.Graph()
        G.add_path_graph(10)  # 9 edges, 10 nodes
        # Undirected: 9 / (10*9/2) = 9 / 45 = 0.2
        assert abs(G.density() - 0.2) < 1e-10

    def test_density_partial_graph(self):
        """Partial graph has intermediate density."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")  # 1 edge out of 3 possible
        # 1 / 3 = 0.333...
        assert abs(G.density() - 1.0 / 3.0) < 1e-10


class TestIsConnected:
    """Test is_connected() method."""

    def test_connected_empty_graph(self):
        """Empty graph is considered connected."""
        G = ec.Graph()
        assert G.is_connected()

    def test_connected_single_node(self):
        """Single node is connected."""
        G = ec.Graph()
        G.add_node("A")
        assert G.is_connected()

    def test_connected_path_graph(self):
        """Path graph is connected."""
        G = ec.Graph()
        G.add_path_graph(5)
        assert G.is_connected()

    def test_connected_cycle_graph(self):
        """Cycle graph is connected."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        assert G.is_connected()

    def test_disconnected_with_isolated_node(self):
        """Graph with isolated node is not connected."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")
        G.add_node("C")  # Isolated
        assert not G.is_connected()

    def test_disconnected_two_components(self):
        """Graph with two separate components is not connected."""
        G = ec.Graph()
        for label in ["A", "B", "C", "D", "E", "F"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("D", "E")
        G.add_edge("E", "F")
        assert not G.is_connected()

    def test_weakly_connected_directed(self):
        """Directed graph checks weak connectivity."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        # Weakly connected (would be connected if undirected)
        assert G.is_connected()


class TestIsDAG:
    """Test is_dag() method."""

    def test_dag_empty_graph(self):
        """Empty directed graph is a DAG."""
        G = ec.Graph(directed=True)
        assert G.is_dag()

    def test_dag_single_node(self):
        """Single node is a DAG."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        assert G.is_dag()

    def test_dag_simple_path(self):
        """Simple directed path is a DAG."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        assert G.is_dag()

    def test_dag_tree_structure(self):
        """Tree structure is a DAG."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("B", "E")
        assert G.is_dag()

    def test_not_dag_simple_cycle(self):
        """Simple cycle is not a DAG."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")
        assert not G.is_dag()

    def test_not_dag_self_loop(self):
        """Self-loop makes it not a DAG."""
        G = ec.Graph(directed=True)
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "B")  # Self-loop
        assert not G.is_dag()

    def test_dag_undirected_raises_error(self):
        """Calling is_dag on undirected graph raises error."""
        G = ec.Graph(directed=False)
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        with pytest.raises(ValueError, match="only works on directed graphs"):
            G.is_dag()


class TestIsSimple:
    """Test is_simple() method."""

    def test_simple_empty_graph(self):
        """Empty graph is simple."""
        G = ec.Graph()
        assert G.is_simple()

    def test_simple_path_graph(self):
        """Path graph is simple."""
        G = ec.Graph()
        G.add_path_graph(5)
        assert G.is_simple()

    def test_not_simple_with_self_loop(self):
        """Graph with self-loop is not simple."""
        G = ec.Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "A")  # Self-loop
        assert not G.is_simple()

    def test_not_simple_with_multi_edges(self):
        """Graph with parallel edges is not simple."""
        G = ec.Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "B")  # Duplicate edge
        assert not G.is_simple()

    def test_simple_directed_graph(self):
        """Simple directed graph."""
        G = ec.Graph(directed=True)
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "A")  # Reverse edge is OK
        assert G.is_simple()


class TestIsBipartite:
    """Test is_bipartite() method."""

    def test_bipartite_empty_graph(self):
        """Empty graph is bipartite."""
        G = ec.Graph()
        assert G.is_bipartite()

    def test_bipartite_single_edge(self):
        """Single edge is bipartite."""
        G = ec.Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        assert G.is_bipartite()

    def test_bipartite_path_graph(self):
        """Path graph is bipartite."""
        G = ec.Graph()
        G.add_path_graph(5)
        assert G.is_bipartite()

    def test_bipartite_even_cycle(self):
        """Even cycle is bipartite."""
        G = ec.Graph()
        G.add_cycle_graph(4)
        assert G.is_bipartite()

    def test_not_bipartite_odd_cycle(self):
        """Odd cycle is not bipartite."""
        G = ec.Graph()
        G.add_cycle_graph(3)
        assert not G.is_bipartite()

    def test_not_bipartite_triangle(self):
        """Triangle is not bipartite."""
        G = ec.Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")
        assert not G.is_bipartite()

    def test_bipartite_star_graph(self):
        """Star graph is bipartite."""
        G = ec.Graph()
        G.add_star_graph(5)
        assert G.is_bipartite()


class TestDegreeDistribution:
    """Test degree_distribution() method."""

    def test_degree_dist_empty_graph(self):
        """Empty graph has empty distribution."""
        G = ec.Graph()
        assert G.degree_distribution() == {}

    def test_degree_dist_path_graph(self):
        """Path graph has 2 nodes with degree 1, rest with degree 2."""
        G = ec.Graph()
        G.add_path_graph(5)
        dist = G.degree_distribution()
        assert dist[1] == 2  # End nodes
        assert dist[2] == 3  # Middle nodes

    def test_degree_dist_star_graph(self):
        """Star graph has 1 hub with high degree, rest with degree 1."""
        G = ec.Graph()
        G.add_star_graph(6)  # 1 center + 5 spokes
        dist = G.degree_distribution()
        assert dist[1] == 5  # Spoke nodes
        assert dist[5] == 1  # Hub node

    def test_degree_dist_cycle_graph(self):
        """Cycle graph has all nodes with degree 2."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        dist = G.degree_distribution()
        assert dist[2] == 6
        assert len(dist) == 1  # Only one degree value


class TestDegreeDistributionDirected:
    """Regression: directed graphs were binned by out-degree, unlike get_degree()."""

    def test_directed_uses_total_degree(self):
        # A -> B -> C plus A -> C and a self-loop on C: totals A 2, B 2, C 4
        G = ec.Graph(directed=True).add_nodes(["A", "B", "C"])
        G.add_edges([("A", "B"), ("B", "C"), ("A", "C"), ("C", "C")])
        assert G.degree_distribution() == {2: 2, 4: 1}
        expected: dict[int, int] = {}
        for nid in G.get_node_ids():
            d = G.get_degree(nid)
            expected[d] = expected.get(d, 0) + 1
        assert G.degree_distribution() == expected

    def test_directed_path_has_no_zero_bin(self):
        # The sink used to land in degree 0
        G = ec.Graph(directed=True).add_path_graph(4)
        assert G.degree_distribution() == {1: 2, 2: 2}


class TestCountSelfloops:
    """Test count_all_loops() method."""

    def test_count_all_loops_empty(self):
        """Empty graph has 0 self-loops."""
        G = ec.Graph()
        assert G.count_all_loops() == 0

    def test_count_all_loops_none(self):
        """Graph with no self-loops."""
        G = ec.Graph()
        G.add_path_graph(4)
        assert G.count_all_loops() == 0

    def test_count_all_loops_single(self):
        """Graph with one self-loop."""
        G = ec.Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "A")
        assert G.count_all_loops() == 1

    def test_count_all_loops_multiple(self):
        """Graph with multiple self-loops."""
        G = ec.Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "A")
        G.add_edge("B", "B")
        G.add_edge("C", "C")
        G.add_edge("A", "B")
        assert G.count_all_loops() == 3


class TestAverageDegree:
    """Test average_degree() method."""

    def test_avg_degree_empty_graph(self):
        """Empty graph has average degree 0."""
        G = ec.Graph()
        assert G.average_degree() == 0.0

    def test_avg_degree_cycle_graph(self):
        """Cycle graph has average degree 2."""
        G = ec.Graph()
        G.add_cycle_graph(5)
        assert G.average_degree() == 2.0

    def test_avg_degree_star_graph(self):
        """Star graph average degree calculation."""
        G = ec.Graph()
        G.add_star_graph(5)  # 1 node with degree 4, 4 nodes with degree 1
        # (4 + 1 + 1 + 1 + 1) / 5 = 8 / 5 = 1.6
        assert abs(G.average_degree() - 1.6) < 1e-10

    def test_avg_degree_complete_graph(self):
        """Complete graph average degree."""
        G = ec.Graph()
        G.add_full_graph(5)  # Each node has degree 4
        assert G.average_degree() == 4.0

    def test_average_degree_directed_counts_in_and_out(self):
        """For directed graphs the average is of in-degree + out-degree (2E/N)."""
        G_ = ec.Graph(directed=True)
        for label in "ABC":
            G_.add_node(label=label)
        for u, v in [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C")]:
            G_.add_edge(u, v)
        assert G_.average_degree() == pytest.approx(2 * 4 / 3)


class TestStructuralPropertiesIntegration:
    """Integration tests combining multiple property checks."""

    def test_properties_on_generated_graphs(self):
        """Test properties on various generated graphs."""
        # Path graph
        G1 = ec.Graph().add_path_graph(10)
        assert G1.is_connected()
        assert G1.is_simple()
        assert G1.is_bipartite()
        assert G1.count_all_loops() == 0

        # Cycle graph
        G2 = ec.Graph().add_cycle_graph(6)
        assert G2.is_connected()
        assert G2.is_simple()
        assert G2.is_bipartite()  # Even cycle
        assert G2.average_degree() == 2.0

        # Complete graph
        G3 = ec.Graph().add_full_graph(4)
        assert G3.is_connected()
        assert G3.is_simple()
        assert not G3.is_bipartite()  # Contains triangles
        assert G3.density() == 1.0

    def test_dag_properties(self):
        """Test DAG-specific properties."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("A", "C")

        assert G.is_dag()
        assert G.is_simple()
        # Directed graph, so weakly connected
        assert G.is_connected()

    def test_disconnected_graph_properties(self):
        """Test properties on disconnected graph."""
        G = ec.Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")

        assert not G.is_connected()
        assert G.is_simple()
        assert G.density() < 0.5  # Sparse


class TestHasCycles:
    """Test has_cycles() method."""

    @staticmethod
    def _graph(edges, directed=False):
        g = ec.Graph(directed=directed)
        for label in dict.fromkeys(x for e in edges for x in e):
            g.add_node(label=label)
        for u, v in edges:
            g.add_edge(u, v)
        return g

    def test_undirected_path_and_tree_are_acyclic(self):
        """Walking an edge and straight back is not a cycle."""
        assert not self._graph([("A", "B"), ("B", "C")]).has_cycles()
        assert not self._graph([("A", "B"), ("A", "C"), ("C", "D"), ("C", "E")]).has_cycles()

    def test_undirected_cycles_detected(self):
        """A triangle, or two parallel edges, is a cycle."""
        assert self._graph([("A", "B"), ("B", "C"), ("C", "A")]).has_cycles()
        g = self._graph([("A", "B")])
        g.add_edge("A", "B")
        assert g.has_cycles()

    def test_directed(self):
        """Directed graphs: a path is acyclic, a directed triangle is not."""
        assert not self._graph([("A", "B"), ("B", "C")], directed=True).has_cycles()
        assert self._graph([("A", "B"), ("B", "C"), ("C", "A")], directed=True).has_cycles()
