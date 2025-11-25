from edgecase import Graph


class TestIsTree:
    """Test the is_tree() method."""

    def test_empty_graph_is_tree(self):
        """An empty graph is considered a tree."""
        g = Graph(directed=False)

        assert g.is_tree()

    def test_single_node_is_tree(self):
        """A single node is a tree."""
        g = Graph(directed=False)
        g.add_node(label="A")

        assert g.is_tree()

    def test_path_is_tree(self):
        """A path graph is a tree."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        assert g.is_tree()

    def test_star_is_tree(self):
        """A star graph is a tree."""
        g = Graph(directed=False)
        g.add_star_graph(n=5)

        assert g.is_tree()

    def test_cycle_not_tree(self):
        """A graph with a cycle is not a tree."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        assert not g.is_tree()

    def test_disconnected_not_tree(self):
        """A disconnected graph is not a tree."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        # C is isolated

        assert not g.is_tree()

    def test_too_many_edges_not_tree(self):
        """A graph with n nodes and > n-1 edges is not a tree."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")  # Extra edge

        assert not g.is_tree()

    def test_too_few_edges_not_tree(self):
        """A graph with n nodes and < n-1 edges is not a tree."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        # Missing edge to D

        assert not g.is_tree()

    def test_directed_graph_not_tree(self):
        """Directed graphs should use is_arborescence() instead."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        assert not g.is_tree()

    def test_complete_binary_tree(self):
        """A complete binary tree is a tree."""
        g = Graph(directed=False)

        # Build binary tree manually
        g.add_node(label="1")
        g.add_node(label="2")
        g.add_node(label="3")
        g.add_node(label="4")
        g.add_node(label="5")
        g.add_node(label="6")
        g.add_node(label="7")
        g.add_edge("1", "2")
        g.add_edge("1", "3")
        g.add_edge("2", "4")
        g.add_edge("2", "5")
        g.add_edge("3", "6")
        g.add_edge("3", "7")

        assert g.is_tree()


class TestIsForest:
    """Test the is_forest() method."""

    def test_empty_graph_is_forest(self):
        """An empty graph is a forest."""
        g = Graph(directed=False)

        assert g.is_forest()

    def test_single_node_is_forest(self):
        """A single node is a forest."""
        g = Graph(directed=False)
        g.add_node(label="A")

        assert g.is_forest()

    def test_tree_is_forest(self):
        """A tree is a special case of forest."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        assert g.is_forest()
        assert g.is_tree()

    def test_two_trees_is_forest(self):
        """Two disjoint trees form a forest."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("C", "D")

        assert g.is_forest()
        assert not g.is_tree()  # Not connected

    def test_multiple_isolated_nodes_is_forest(self):
        """Multiple isolated nodes form a forest."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        assert g.is_forest()

    def test_forest_with_isolated_node(self):
        """Tree + isolated node is a forest."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        # D is isolated

        assert g.is_forest()

    def test_cycle_not_forest(self):
        """A graph with a cycle is not a forest."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        assert not g.is_forest()

    def test_forest_with_cycle_not_forest(self):
        """A forest with one component containing a cycle is not a forest."""
        g = Graph(directed=False)

        # Tree component
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # Cycle component
        g.add_node(label="X")
        g.add_node(label="Y")
        g.add_node(label="Z")
        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")
        g.add_edge("Z", "X")

        assert not g.is_forest()

    def test_directed_graph_not_forest(self):
        """Directed graphs return False (not implemented) with is_forest()."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        assert not g.is_forest()

    def test_edge_count_formula(self):
        """Forest satisfies e = n - k where k is component count."""
        g = Graph(directed=False)

        # 3 components: A-B, C-D-E, F
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_node(label="E")
        g.add_node(label="F")
        g.add_edge("A", "B")  # Component 1: 2 nodes, 1 edge
        g.add_edge("C", "D")  # Component 2: 3 nodes, 2 edges
        g.add_edge("D", "E")
        # Component 3: 1 node (F), 0 edges
        # Total: n=6, k=3, e=3, so e = n-k

        assert g.is_forest()


class TestIsArborescence:
    """Test is_arborescence() method."""

    def test_empty_graph_is_arborescence(self):
        """An empty graph is considered an arborescence."""
        g = Graph(directed=True)

        assert g.is_arborescence()

    def test_single_node_is_arborescence(self):
        """A single node is an arborescence."""
        g = Graph(directed=True)
        g.add_node(label="root")

        assert g.is_arborescence()

    def test_simple_arborescence(self):
        """Simple directed tree with root->child."""
        g = Graph(directed=True)
        g.add_node(label="root")
        g.add_node(label="A")
        g.add_edge("root", "A")

        assert g.is_arborescence()

    def test_binary_arborescence(self):
        """Binary tree rooted at one node."""
        g = Graph(directed=True)
        g.add_node(label="root")
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("root", "A")
        g.add_edge("root", "B")
        g.add_edge("A", "C")

        assert g.is_arborescence()

    def test_balanced_tree_is_arborescence(self):
        """A balanced tree is an arborescence."""
        g = Graph(directed=True)
        g.add_balanced_tree(k=2, h=3)

        assert g.is_arborescence()

    def test_no_root_not_arborescence(self):
        """A graph with no root (all nodes have in-degree >=1) is not an arborescence."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")  # Cycle, no root

        assert not g.is_arborescence()

    def test_multiple_roots_not_arborescence(self):
        """A graph with multiple roots is not an arborescence."""
        g = Graph(directed=True)
        g.add_node(label="root1")
        g.add_node(label="root2")
        g.add_node(label="A")
        g.add_edge("root1", "A")
        g.add_edge("root2", "A")

        assert not g.is_arborescence()  # A has in-degree 2

    def test_high_in_degree_not_arborescence(self):
        """Node with in-degree > 1 makes it not an arborescence."""
        g = Graph(directed=True)
        g.add_node(label="root")
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("root", "A")
        g.add_edge("root", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "C")  # C has in-degree 2

        assert not g.is_arborescence()

    def test_disconnected_not_arborescence(self):
        """A disconnected directed graph is not an arborescence."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        # C is isolated

        assert not g.is_arborescence()

    def test_too_many_edges_not_arborescence(self):
        """A graph with n nodes and ≥n edges is not an arborescence."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")  # Extra edge

        assert not g.is_arborescence()

    def test_undirected_not_arborescence(self):
        """Undirected graphs should use is_tree() instead."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        assert not g.is_arborescence()

    def test_deep_arborescence(self):
        """A deep chain is an arborescence."""
        g = Graph(directed=True)
        g.add_node(label="root")
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("root", "A")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        assert g.is_arborescence()


class TestTreeMethodsIntegration:
    """Integration tests for tree methods."""

    def test_tree_is_also_forest(self):
        """Any tree is also a forest."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")

        assert g.is_tree()
        assert g.is_forest()

    def test_forest_not_necessarily_tree(self):
        """A forest may not be a tree (if disconnected)."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "B")
        g.add_edge("C", "D")

        assert g.is_forest()
        assert not g.is_tree()

    def test_neither_tree_nor_forest(self):
        """A graph with cycles is neither a tree nor a forest."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        assert not g.is_tree()
        assert not g.is_forest()

    def test_arborescence_properties(self):
        """Verify arborescence structural properties."""
        g = Graph(directed=True)
        g.add_node(label="root")
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("root", "A")
        g.add_edge("root", "B")
        g.add_edge("A", "C")

        assert g.is_arborescence()
        assert g.count_nodes() == 4
        assert g.count_edges() == 3  # n-1 edges
        assert g.count_weakly_connected_components() == 1
