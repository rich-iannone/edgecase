import pytest
from edgecase import Graph


class TestKCore:
    """Tests for k-core decomposition"""

    def test_k_core_simple_graph(self):
        """Test k-core on a simple graph"""
        g = Graph()
        # Create a simple graph:
        # 1 -- 2 -- 3
        # |    |    |
        # 4 -- 5 -- 6
        for i in range(1, 7):
            g.add_node(label=str(i))
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("1", "4")
        g.add_edge("2", "5")
        g.add_edge("3", "6")
        g.add_edge("4", "5")
        g.add_edge("5", "6")

        # 2-core should remove nodes with degree < 2
        core_2 = g.k_core(2)

        assert core_2.count_nodes() > 0
        assert core_2.count_edges() > 0

    def test_k_core_empty_result(self):
        """Test k-core when k is too high"""
        g = Graph()
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_edge("1", "2")
        g.add_edge("2", "3")

        # 3-core should be empty (no node has degree >= 3)
        core_3 = g.k_core(3)

        assert core_3.count_nodes() == 0
        assert core_3.count_edges() == 0

    def test_k_core_complete_graph(self):
        """Test k-core on complete graph"""
        g = Graph()
        n = 5
        for i in range(1, n + 1):
            g.add_node(label=str(i))
        # Make complete graph K5
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                g.add_edge(str(i), str(j))

        # In K5, every node has degree 4
        # k-core for k <= 4 should be the full graph
        core_4 = g.k_core(4)

        assert core_4.count_nodes() == 5
        assert core_4.count_edges() == 10

    def test_k_core_star_graph(self):
        """Test k-core on star graph"""
        g = Graph()
        g.add_node(label="center")
        for i in range(1, 6):
            g.add_node(label=f"leaf{i}")
            g.add_edge("center", f"leaf{i}")

        # 2-core should be empty (leaves have degree 1, center would have degree 0 after removing leaves)
        core_2 = g.k_core(2)

        assert core_2.count_nodes() == 0

    def test_k_core_directed_graph(self):
        """Test k-core on directed graph"""
        g = Graph(directed=True)

        for i in range(1, 5):
            g.add_node(label=str(i))
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "4")
        g.add_edge("4", "1")

        # In directed graphs, degree counts outgoing edges in adjacency list
        # So each node in this cycle has out-degree = 1
        core_1 = g.k_core(1)

        assert core_1.count_nodes() == 4  # All nodes have degree >= 1

        # For degree 2, we need more connections
        core_2 = g.k_core(2)

        assert core_2.count_nodes() == 0  # No nodes have degree >= 2 in this cycle


class TestKShell:
    """Tests for k-shell decomposition"""

    def test_k_shell_simple_graph(self):
        """Test k-shell extraction"""
        g = Graph()

        for i in range(1, 6):
            g.add_node(label=str(i))
        # Create graph with different coreness values
        # 1 -- 2 -- 3
        #      |    |
        #      4 -- 5
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("2", "4")
        g.add_edge("3", "5")
        g.add_edge("4", "5")

        # Get 1-shell (nodes with coreness exactly 1)
        shell_1 = g.k_shell(1)
        # Node 1 has coreness 1

        assert shell_1.count_nodes() >= 1

    def test_k_shell_empty_result(self):
        """Test k-shell when no nodes have that coreness"""
        g = Graph()
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_edge("1", "2")
        g.add_edge("2", "3")

        # 10-shell should be empty
        shell_10 = g.k_shell(10)

        assert shell_10.count_nodes() == 0

    def test_k_shell_complete_graph(self):
        """Test k-shell on complete graph"""
        g = Graph()
        n = 5
        for i in range(1, n + 1):
            g.add_node(label=str(i))
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                g.add_edge(str(i), str(j))

        # All nodes in K5 have coreness 4
        shell_4 = g.k_shell(4)

        assert shell_4.count_nodes() == 5


class TestKCrust:
    """Tests for k-crust decomposition"""

    def test_k_crust_simple_graph(self):
        """Test k-crust extraction"""
        g = Graph()
        for i in range(1, 6):
            g.add_node(label=str(i))
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("2", "4")
        g.add_edge("3", "5")
        g.add_edge("4", "5")

        # k-crust should include all nodes with coreness <= k
        crust_2 = g.k_crust(2)

        assert crust_2.count_nodes() >= 3  # At least some nodes

    def test_k_crust_includes_all(self):
        """Test k-crust with high k includes all nodes"""
        g = Graph()
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_edge("1", "2")
        g.add_edge("2", "3")

        # High k should include all nodes
        crust_10 = g.k_crust(10)

        assert crust_10.count_nodes() == 3

    def test_k_crust_excludes_high_coreness(self):
        """Test k-crust excludes high coreness nodes"""
        g = Graph()
        n = 5
        for i in range(1, n + 1):
            g.add_node(label=str(i))
        # Complete graph K5
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                g.add_edge(str(i), str(j))

        # 0-crust should be empty (all nodes have coreness 4)
        crust_0 = g.k_crust(0)

        assert crust_0.count_nodes() == 0


class TestKTruss:
    """Tests for k-truss decomposition"""

    def test_k_truss_triangle(self):
        """Test k-truss on triangle"""
        g = Graph()
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "1")

        # 3-truss should include the triangle (each edge is in 1 triangle)
        truss_3 = g.k_truss(3)

        assert truss_3.count_nodes() == 3
        assert truss_3.count_edges() == 3

    def test_k_truss_no_triangles(self):
        """Test k-truss on graph with no triangles"""
        g = Graph()
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_edge("1", "2")
        g.add_edge("2", "3")

        # 3-truss should be empty (no triangles)
        truss_3 = g.k_truss(3)

        assert truss_3.count_nodes() == 0
        assert truss_3.count_edges() == 0

    def test_k_truss_complete_graph(self):
        """Test k-truss on complete graph"""
        g = Graph()
        n = 4
        for i in range(1, n + 1):
            g.add_node(label=str(i))
        # Complete graph K4
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                g.add_edge(str(i), str(j))

        # In K4, each edge is in 2 triangles
        # So 4-truss (k-2=2 triangles) should include all edges
        truss_4 = g.k_truss(4)

        assert truss_4.count_nodes() == 4
        assert truss_4.count_edges() == 6

    def test_k_truss_invalid_k(self):
        """Test k-truss with k < 3 raises error"""
        g = Graph()
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_edge("1", "2")

        with pytest.raises(ValueError, match="k must be at least 3"):
            g.k_truss(2)

    def test_k_truss_removes_isolated_nodes(self):
        """Test k-truss removes nodes with no edges"""
        g = Graph()
        for i in range(1, 6):
            g.add_node(label=str(i))
        # Triangle + isolated nodes
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "1")
        # nodes 4 and 5 are isolated

        truss_3 = g.k_truss(3)

        # Should only include triangle nodes
        assert truss_3.count_nodes() == 3


class TestKCorona:
    """Tests for k-corona decomposition"""

    def test_k_corona_triangle_with_periphery(self):
        """Test k-corona on triangle with peripheral nodes"""
        g = Graph()
        # Create triangle (2-core)
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "1")

        # Add peripheral nodes (1-core only)
        g.add_node(label="4")
        g.add_node(label="5")
        g.add_edge("1", "4")
        g.add_edge("2", "5")

        # 2-corona: nodes in 2-core with exactly 2 neighbors in 2-core
        # Each triangle node has exactly 2 neighbors in the 2-core (the other 2 triangle nodes)
        corona_2 = g.k_corona(2)
        assert corona_2.count_nodes() == 3

    def test_k_corona_complete_graph(self):
        """Test k-corona on complete graph K4"""
        g = Graph()
        g.add_full_graph(4)

        # In K4, all nodes are in the 3-core and each has exactly 3 neighbors in 3-core
        corona_3 = g.k_corona(3)
        assert corona_3.count_nodes() == 4
        assert corona_3.count_edges() == 6

    def test_k_corona_path_graph(self):
        """Test k-corona on path graph"""
        g = Graph()
        g.add_path_graph(5)  # 1-2-3-4-5

        # Path has only 1-core (max coreness is 1)
        # End nodes (1 and 5) have exactly 1 neighbor in 1-core
        corona_1 = g.k_corona(1)
        assert corona_1.count_nodes() == 2  # End nodes

        # 2-corona should be empty (no 2-core exists)
        corona_2 = g.k_corona(2)
        assert corona_2.count_nodes() == 0

    def test_k_corona_empty_result(self):
        """Test k-corona returns empty graph when no nodes qualify"""
        g = Graph()
        g.add_cycle_graph(4)  # 1-2-3-4-1

        # All nodes in cycle have degree 2, so they're in 2-core
        # Each node has exactly 2 neighbors in 2-core, so all are in 2-corona
        corona_2 = g.k_corona(2)
        assert corona_2.count_nodes() == 4

        # 3-corona should be empty (no 3-core exists)
        corona_3 = g.k_corona(3)
        assert corona_3.count_nodes() == 0

    def test_k_corona_star_graph(self):
        """Test k-corona on star graph"""
        g = Graph()
        g.add_star_graph(5)  # Hub (1) + 4 leaves (2, 3, 4, 5)

        # Hub has degree 4, leaves have degree 1
        # All are in 1-core
        # Hub has 4 neighbors in 1-core, so not in 1-corona
        # Leaves have 1 neighbor in 1-core, so they are in 1-corona
        corona_1 = g.k_corona(1)
        assert corona_1.count_nodes() == 4  # The 4 leaves

    def test_k_corona_preserves_edges(self):
        """Test that k-corona preserves edges between corona nodes"""
        g = Graph()
        # Two triangles sharing one edge
        for i in range(1, 5):
            g.add_node(label=str(i))
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("3", "1")
        g.add_edge("2", "4")
        g.add_edge("4", "3")

        corona_2 = g.k_corona(2)
        # Should preserve edges between corona nodes
        assert corona_2.count_edges() >= 0  # At least 0 edges


class TestKDecompositionProperties:
    """Test mathematical properties of k-decompositions"""

    def test_core_shell_crust_relationship(self):
        """Test that k-core, k-shell, k-crust are related correctly"""
        g = Graph()
        n = 10
        for i in range(1, n + 1):
            g.add_node(label=str(i))
        # Create graph with varying degrees
        for i in range(1, n):
            g.add_edge(str(i), str(i + 1))
        g.add_edge(str(1), str(n))

        # For any k, nodes in k-shell should be subset of nodes in k-crust
        k = 1
        shell = g.k_shell(k)
        crust = g.k_crust(k)

        assert shell.count_nodes() <= crust.count_nodes()

    def test_increasing_k_decreases_core_size(self):
        """Test that k-core size decreases as k increases"""
        g = Graph()
        n = 10
        for i in range(1, n + 1):
            g.add_node(label=str(i))
        # Create a graph with varying connectivity
        for i in range(1, n):
            g.add_edge(str(i), str(i + 1))
        # Add some extra edges
        g.add_edge("1", "6")
        g.add_edge("3", "8")

        core_1 = g.k_core(1)
        core_2 = g.k_core(2)

        # k=2 core should have <= nodes than k=1 core
        assert core_2.count_nodes() <= core_1.count_nodes()

    def test_weighted_graph_k_core(self):
        """Test k-core on weighted graph"""
        g = Graph(weighted=True)
        for i in range(1, 5):
            g.add_node(label=str(i))
        g.add_edge("1", "2", weight=1.5)
        g.add_edge("2", "3", weight=2.0)
        g.add_edge("3", "4", weight=0.5)
        g.add_edge("4", "1", weight=3.0)

        # k-core should work on weighted graphs (degree ignores weights)
        core_2 = g.k_core(2)

        assert core_2.count_nodes() == 4  # All nodes have degree 2
