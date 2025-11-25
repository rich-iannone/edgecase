from edgecase import Graph


class TestIsPlanar:
    """Tests for is_planar() method."""

    def test_empty_graph_is_planar(self):
        """Empty graph is planar."""
        graph = Graph(directed=False)

        assert graph.is_planar()

    def test_single_node_is_planar(self):
        """Single node is planar."""
        graph = Graph(directed=False).add_node(label="A")

        assert graph.is_planar()

    def test_two_nodes_is_planar(self):
        """Two nodes with edge is planar."""
        graph = Graph(directed=False).add_node(label="A").add_node(label="B").add_edge("A", "B")

        assert graph.is_planar()

    def test_path_graph_is_planar(self):
        """Path graphs are planar."""
        graph = Graph(directed=False).add_path_graph(n=10)

        assert graph.is_planar()

    def test_cycle_graph_is_planar(self):
        """Cycle graphs are planar."""
        graph = Graph(directed=False).add_cycle_graph(n=8)

        assert graph.is_planar()

    def test_tree_is_planar(self):
        """Trees are always planar (e = n-1)."""
        tree = Graph(directed=False)
        tree.add_balanced_tree(k=3, h=3)

        assert tree.is_planar()

    def test_star_graph_is_planar(self):
        """Star graphs are planar."""
        star = Graph(directed=False)
        star.add_star_graph(n=10)

        assert star.is_planar()

    def test_grid_is_planar(self):
        """Grid graphs are planar."""
        grid = Graph(directed=False)
        grid.add_grid_2d(rows=5, cols=5)

        assert grid.is_planar()

    def test_k4_is_planar(self):
        """Complete graph K_4 is planar (6 edges ≤ 3*4-6 = 6)."""
        k4 = Graph(directed=False)
        k4.add_full_graph(n=4)

        assert k4.is_planar()

    def test_k5_not_planar(self):
        """Complete graph K_5 is not planar (10 edges > 3*5-6 = 9)."""
        k5 = Graph(directed=False)
        k5.add_full_graph(n=5)

        assert not k5.is_planar()

    def test_k6_not_planar(self):
        """Complete graph K_6 is not planar."""
        k6 = Graph(directed=False)
        k6.add_full_graph(n=6)

        assert not k6.is_planar()

    def test_bipartite_graph_k33_not_planar(self):
        """Complete bipartite K_{3,3} is not planar."""
        # K_{3,3} has 6 nodes, 9 edges
        # For graphs without triangles: e ≤ 2n-4 = 8
        # So 9 > 8, not planar
        k33 = (
            Graph(directed=False)
            .add_node(label="A1")
            .add_node(label="A2")
            .add_node(label="A3")
            .add_node(label="B1")
            .add_node(label="B2")
            .add_node(label="B3")
            .add_edge("A1", "B1")
            .add_edge("A1", "B2")
            .add_edge("A1", "B3")
            .add_edge("A2", "B1")
            .add_edge("A2", "B2")
            .add_edge("A2", "B3")
            .add_edge("A3", "B1")
            .add_edge("A3", "B2")
            .add_edge("A3", "B3")
        )

        assert not k33.is_planar()

    def test_triangle_is_planar(self):
        """Triangle is planar."""
        triangle = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        assert triangle.is_planar()


class TestIsIsomorphic:
    """Tests for is_isomorphic() method."""

    def test_empty_graphs_isomorphic(self):
        """Two empty graphs are isomorphic."""
        g1 = Graph(directed=False)
        g2 = Graph(directed=False)

        assert g1.is_isomorphic(g2)

    def test_single_node_graphs_isomorphic(self):
        """Two single-node graphs are isomorphic."""
        g1 = Graph(directed=False).add_node(label="A")
        g2 = Graph(directed=False).add_node(label="X")

        assert g1.is_isomorphic(g2)

    def test_same_path_graphs_isomorphic(self):
        """Two path graphs with same length are isomorphic."""
        g1 = Graph(directed=False).add_path_graph(n=5)
        g2 = Graph(directed=False).add_path_graph(n=5)

        assert g1.is_isomorphic(g2)

    def test_different_sized_graphs_not_isomorphic(self):
        """Graphs with different node counts are not isomorphic."""
        g1 = Graph(directed=False).add_path_graph(n=3)
        g2 = Graph(directed=False).add_path_graph(n=4)

        assert not g1.is_isomorphic(g2)

    def test_different_edge_counts_not_isomorphic(self):
        """Graphs with different edge counts are not isomorphic."""
        g1 = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
        )
        g2 = (
            Graph(directed=False)
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("X", "Y")
            .add_edge("Y", "Z")
        )

        assert not g1.is_isomorphic(g2)

    def test_different_directedness_not_isomorphic(self):
        """Directed and undirected graphs are not isomorphic."""
        g1 = Graph(directed=True).add_path_graph(n=3)
        g2 = Graph(directed=False).add_path_graph(n=3)

        assert not g1.is_isomorphic(g2)

    def test_path_and_star_not_isomorphic(self):
        """Path and star graphs with same size have different degree sequences."""
        path = Graph(directed=False).add_path_graph(n=5)
        star = Graph(directed=False).add_star_graph(n=5)

        assert not path.is_isomorphic(star)

    def test_same_complete_graphs_isomorphic(self):
        """Two complete graphs of same size are isomorphic."""
        k4_a = Graph(directed=False).add_full_graph(n=4)
        k4_b = Graph(directed=False).add_full_graph(n=4)

        assert k4_a.is_isomorphic(k4_b)

    def test_same_cycle_graphs_isomorphic(self):
        """Two cycle graphs of same size are isomorphic."""
        c6_a = Graph(directed=False).add_cycle_graph(n=6)
        c6_b = Graph(directed=False).add_cycle_graph(n=6)

        assert c6_a.is_isomorphic(c6_b)

    def test_triangles_with_different_labels_isomorphic(self):
        """Triangles with different labels are isomorphic."""
        g1 = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        g2 = (
            Graph(directed=False)
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("X", "Y")
            .add_edge("Y", "Z")
            .add_edge("Z", "X")
        )

        assert g1.is_isomorphic(g2)

    def test_different_degree_sequences_not_isomorphic(self):
        """Graphs with different degree sequences are not isomorphic."""
        # Graph 1: Star (degrees: 1,1,1,3)
        g1 = Graph(directed=False).add_star_graph(n=4)

        # Graph 2: Path (degrees: 1,2,2,1)
        g2 = Graph(directed=False).add_path_graph(n=4)

        assert not g1.is_isomorphic(g2)

    def test_isomorphic_with_itself(self):
        """Any graph is isomorphic with itself."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        assert graph.is_isomorphic(graph)

    def test_complex_graphs_same_structure(self):
        """Complex graphs with same structure are isomorphic."""
        # Both are "diamond" shapes: 4 nodes, 5 edges
        g1 = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
            .add_edge("B", "C")
        )
        g2 = (
            Graph(directed=False)
            .add_node(label="W")
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("W", "X")
            .add_edge("W", "Y")
            .add_edge("X", "Z")
            .add_edge("Y", "Z")
            .add_edge("X", "Y")
        )

        assert g1.is_isomorphic(g2)

    def test_directed_graphs_isomorphic(self):
        """Directed graphs can be isomorphic."""
        g1 = (
            Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )
        g2 = (
            Graph(directed=True)
            .add_node(label="X")
            .add_node(label="Y")
            .add_node(label="Z")
            .add_edge("X", "Y")
            .add_edge("Y", "Z")
        )

        assert g1.is_isomorphic(g2)


class TestIntegration:
    """Integration tests for both methods."""

    def test_planar_graphs_can_be_isomorphic(self):
        """Two planar graphs can be isomorphic."""
        g1 = Graph(directed=False).add_cycle_graph(n=4)
        g2 = Graph(directed=False).add_cycle_graph(n=4)

        assert g1.is_planar()
        assert g2.is_planar()
        assert g1.is_isomorphic(g2)

    def test_non_planar_graphs_can_be_isomorphic(self):
        """Two non-planar graphs can be isomorphic."""
        k5_a = Graph(directed=False).add_full_graph(n=5)
        k5_b = Graph(directed=False).add_full_graph(n=5)

        assert not k5_a.is_planar()
        assert not k5_b.is_planar()
        assert k5_a.is_isomorphic(k5_b)

    def test_trees_are_planar(self):
        """All trees are planar."""
        for n in [5, 10, 20]:
            tree = Graph(directed=False).add_path_graph(n=n)
            assert tree.is_planar()
            assert tree.is_tree()
