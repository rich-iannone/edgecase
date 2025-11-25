import pytest
from edgecase import Graph


class TestFindCliques:
    """Tests for find_cliques() method."""

    def test_empty_graph(self):
        """Empty graph returns empty clique."""
        graph = Graph(directed=False)
        cliques = graph.find_cliques()

        # Empty graph returns one clique: the empty set
        assert len(cliques) == 1
        assert cliques[0] == []

    def test_single_node(self):
        """Single node is a clique of size 1."""
        graph = Graph(directed=False).add_node(label="A")
        cliques = graph.find_cliques()

        assert len(cliques) == 1
        assert len(cliques[0]) == 1

    def test_triangle_clique(self):
        """Triangle forms a single clique of size 3."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        cliques = graph.find_cliques()

        assert len(cliques) == 1
        assert len(cliques[0]) == 3

    def test_path_graph_cliques(self):
        """Path graph has cliques of size 2 (edges)."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )
        cliques = graph.find_cliques()

        assert len(cliques) == 2  # Two edges: A-B and B-C

        for clique in cliques:
            assert len(clique) == 2

    def test_complete_graph(self):
        """Complete graph K_n has one clique of size n."""
        graph = Graph(directed=False)
        graph.add_full_graph(n=5)
        cliques = graph.find_cliques()

        assert len(cliques) == 1
        assert len(cliques[0]) == 5

    def test_multiple_disjoint_cliques(self):
        """Graph with multiple separate cliques."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
            # Triangle 1
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
            # Triangle 2
            .add_edge("D", "E")
            .add_edge("E", "F")
            .add_edge("F", "D")
        )
        cliques = graph.find_cliques()

        assert len(cliques) == 2

        for clique in cliques:
            assert len(clique) == 3

    def test_directed_graph_raises_error(self):
        """find_cliques() raises error on directed graphs."""
        graph = Graph(directed=True).add_node(label="A").add_node(label="B")

        with pytest.raises(ValueError, match="only works on undirected"):
            graph.find_cliques()

    def test_cliques_are_sorted(self):
        """Clique node IDs are sorted."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        cliques = graph.find_cliques()
        clique = cliques[0]

        assert clique == sorted(clique)

    def test_star_graph_cliques(self):
        """Star graph has cliques of size 2 (hub-leaf pairs)."""
        graph = Graph(directed=False)
        graph.add_star_graph(n=5)
        cliques = graph.find_cliques()

        # Star with n=5 has center + 4 leaves = 4 edges
        assert len(cliques) == 4

        for clique in cliques:
            assert len(clique) == 2


class TestMaxClique:
    """Tests for max_clique() method."""

    def test_empty_graph(self):
        """Empty graph returns empty clique."""
        graph = Graph(directed=False)
        max_clique = graph.max_clique()

        assert max_clique == []

    def test_single_node(self):
        """Single node is the maximum clique."""
        graph = Graph(directed=False).add_node(label="A")
        max_clique = graph.max_clique()

        assert len(max_clique) == 1

    def test_triangle_max_clique(self):
        """Triangle's max clique is size 3."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        max_clique = graph.max_clique()

        assert len(max_clique) == 3

    def test_path_graph_max_clique(self):
        """Path graph's max clique is size 2."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "D")
        )
        max_clique = graph.max_clique()

        assert len(max_clique) == 2

    def test_complete_graph_max_clique(self):
        """Complete graph K_n has max clique of size n."""
        graph = Graph(directed=False)
        graph.add_full_graph(n=6)
        max_clique = graph.max_clique()

        assert len(max_clique) == 6

    def test_directed_graph_raises_error(self):
        """max_clique() raises error on directed graphs."""
        graph = Graph(directed=True).add_node(label="A")

        with pytest.raises(ValueError, match="only works on undirected"):
            graph.max_clique()

    def test_max_clique_is_sorted(self):
        """Maximum clique node IDs are sorted."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        max_clique = graph.max_clique()

        assert max_clique == sorted(max_clique)

    def test_graph_with_multiple_max_cliques(self):
        """Graph with multiple maximum cliques returns one."""
        # Two triangles connected by a single edge
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_node(label="F")
            # Triangle 1
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
            # Bridge
            .add_edge("C", "D")
            # Triangle 2
            .add_edge("D", "E")
            .add_edge("E", "F")
            .add_edge("F", "D")
        )
        max_clique = graph.max_clique()

        assert len(max_clique) == 3  # One of the triangles


class TestCliqueNumber:
    """Tests for clique_number() method."""

    def test_empty_graph(self):
        """Empty graph has clique number 0."""
        graph = Graph(directed=False)

        assert graph.clique_number() == 0

    def test_single_node(self):
        """Single node has clique number 1."""
        graph = Graph(directed=False).add_node(label="A")

        assert graph.clique_number() == 1

    def test_triangle_clique_number(self):
        """Triangle has clique number 3."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        assert graph.clique_number() == 3

    def test_path_graph_clique_number(self):
        """Path graph has clique number 2."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
        )

        assert graph.clique_number() == 2

    def test_complete_graph_clique_number(self):
        """Complete graph K_n has clique number n."""
        graph = Graph(directed=False)
        graph.add_full_graph(n=7)

        assert graph.clique_number() == 7

    def test_star_graph_clique_number(self):
        """Star graph has clique number 2."""
        graph = Graph(directed=False)
        graph.add_star_graph(n=10)

        assert graph.clique_number() == 2

    def test_bipartite_clique_number(self):
        """Bipartite graph has clique number at most 2."""
        # Complete bipartite graph K_{2,3}
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_node(label="E")
            .add_edge("A", "C")
            .add_edge("A", "D")
            .add_edge("A", "E")
            .add_edge("B", "C")
            .add_edge("B", "D")
            .add_edge("B", "E")
        )

        assert graph.clique_number() == 2
        assert graph.is_bipartite()  # Verify it's bipartite

    def test_directed_graph_raises_error(self):
        """clique_number() raises error on directed graphs."""
        graph = Graph(directed=True).add_node(label="A")

        with pytest.raises(ValueError, match="only works on undirected"):
            graph.clique_number()

    def test_cycle_graph_clique_number(self):
        """Cycle graph has clique number 2."""
        graph = Graph(directed=False)
        graph.add_cycle_graph(n=8)

        assert graph.clique_number() == 2

    def test_four_clique(self):
        """Complete graph K_4 has clique number 4."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .add_edge("A", "D")
            .add_edge("B", "C")
            .add_edge("B", "D")
            .add_edge("C", "D")
        )

        assert graph.clique_number() == 4


class TestCliqueIntegration:
    """Integration tests across clique methods."""

    def test_clique_number_equals_max_clique_size(self):
        """clique_number() equals len(max_clique())."""
        graph = Graph(directed=False)
        graph.add_full_graph(n=5)

        max_clique = graph.max_clique()
        clique_number = graph.clique_number()

        assert clique_number == len(max_clique)

    def test_max_clique_in_find_cliques(self):
        """max_clique() result is one of find_cliques() results."""
        graph = (
            Graph(directed=False)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )

        all_cliques = graph.find_cliques()
        max_clique = graph.max_clique()

        # max_clique should be in all_cliques
        assert max_clique in all_cliques

    def test_consistency_across_methods(self):
        """All three methods give consistent results."""
        graph = Graph(directed=False)
        graph.add_cycle_graph(n=6)

        all_cliques = graph.find_cliques()
        max_clique = graph.max_clique()
        clique_number = graph.clique_number()

        # All cliques should have size <= clique_number
        for clique in all_cliques:
            assert len(clique) <= clique_number

        # max_clique should have size == clique_number
        assert len(max_clique) == clique_number

        # At least one clique should have size == clique_number
        max_sizes = [len(c) for c in all_cliques]

        assert max(max_sizes) == clique_number
