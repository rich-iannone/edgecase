import math
import pytest
import edgecase as ec
from edgecase.matrix_utils import from_adjacency_matrix, to_adjacency_matrix


class TestToAdjacencyMatrixBasics:
    """Basic functionality tests for to_adjacency_matrix."""

    def test_empty_graph(self):
        """Empty graph should return empty matrix."""
        g = ec.Graph()

        matrix, node_ids = to_adjacency_matrix(g)

        assert matrix == []
        assert node_ids == []

    def test_single_node(self):
        """Single node should return 1x1 matrix with zero."""
        g = ec.Graph().add_node(label="A")

        matrix, node_ids = to_adjacency_matrix(g)

        assert len(matrix) == 1
        assert len(matrix[0]) == 1
        assert matrix[0][0] == 0.0
        assert len(node_ids) == 1

    def test_two_nodes_no_edge(self):
        """Two disconnected nodes should have all zeros."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")

        matrix, node_ids = to_adjacency_matrix(g)

        assert len(matrix) == 2
        assert all(matrix[i][j] == 0.0 for i in range(2) for j in range(2))

    def test_two_nodes_with_edge(self):
        """Two connected nodes should have 1s for the edge."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        matrix, node_ids = to_adjacency_matrix(g)

        # Find indices
        idx_a = node_ids.index(g.get_id_from_label("A"))
        idx_b = node_ids.index(g.get_id_from_label("B"))

        # Both directions should be 1 for undirected
        assert matrix[idx_a][idx_b] == 1.0
        assert matrix[idx_b][idx_a] == 1.0

    def test_triangle(self):
        """Triangle should have correct adjacency."""
        g = ec.Graph().add_cycle_graph(3)
        matrix, node_ids = to_adjacency_matrix(g)

        assert len(matrix) == 3
        assert len(node_ids) == 3

        # Each node should have exactly 2 neighbors
        for i in range(3):
            row_sum = sum(matrix[i])

            assert row_sum == 2.0

    def test_path_graph(self):
        """Path graph should have correct structure."""
        g = ec.Graph().add_path_graph(4)
        matrix, node_ids = to_adjacency_matrix(g)

        # Find degree of each node from matrix
        degrees = [sum(row) for row in matrix]

        # End nodes have degree 1, middle nodes have degree 2
        assert sorted(degrees) == [1.0, 1.0, 2.0, 2.0]

    def test_complete_graph(self):
        """Complete graph should have all 1s except diagonal."""
        g = ec.Graph().add_full_graph(4)
        matrix, node_ids = to_adjacency_matrix(g)

        for i in range(4):
            for j in range(4):
                if i == j:
                    assert matrix[i][j] == 0.0
                else:
                    assert matrix[i][j] == 1.0


class TestToAdjacencyMatrixDirected:
    """Tests for directed graphs."""

    def test_directed_edge(self):
        """Directed edge should only appear in one direction."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        matrix, node_ids = to_adjacency_matrix(g)

        idx_a = node_ids.index(g.get_id_from_label("A"))
        idx_b = node_ids.index(g.get_id_from_label("B"))

        assert matrix[idx_a][idx_b] == 1.0  # A -> B exists
        assert matrix[idx_b][idx_a] == 0.0  # B -> A does not exist

    def test_directed_cycle(self):
        """Directed cycle should be asymmetric."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        matrix, node_ids = to_adjacency_matrix(g)

        # Matrix should not be symmetric
        is_symmetric = all(matrix[i][j] == matrix[j][i] for i in range(3) for j in range(3))

        assert not is_symmetric


class TestToAdjacencyMatrixWeighted:
    """Tests for weighted graphs."""

    def test_weighted_edges(self):
        """Weighted edges should appear in matrix."""
        g = ec.Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B", weight=2.5)
        matrix, node_ids = to_adjacency_matrix(g)

        idx_a = node_ids.index(g.get_id_from_label("A"))
        idx_b = node_ids.index(g.get_id_from_label("B"))

        assert matrix[idx_a][idx_b] == 2.5
        assert matrix[idx_b][idx_a] == 2.5  # Undirected

    def test_mixed_weights(self):
        """Multiple edges with different weights."""
        g = ec.Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B", weight=1.5)
        g.add_edge("B", "C", weight=2.5)
        matrix, node_ids = to_adjacency_matrix(g)

        idx_a = node_ids.index(g.get_id_from_label("A"))
        idx_b = node_ids.index(g.get_id_from_label("B"))
        idx_c = node_ids.index(g.get_id_from_label("C"))

        assert matrix[idx_a][idx_b] == 1.5
        assert matrix[idx_b][idx_c] == 2.5
        assert matrix[idx_a][idx_c] == 0.0  # No edge


class TestToAdjacencyMatrixNonedge:
    """Tests for the nonedge parameter."""

    def test_nonedge_infinity(self):
        """Non-edges can be represented as infinity."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        # No edge between A and B
        matrix, node_ids = to_adjacency_matrix(g, nonedge=math.inf)

        assert matrix[0][1] == math.inf
        assert matrix[1][0] == math.inf

        # Diagonal should also be infinity (no self-loops)
        assert matrix[0][0] == math.inf

    def test_nonedge_negative_one(self):
        """Non-edges can be -1."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        matrix, node_ids = to_adjacency_matrix(g, nonedge=-1.0)

        assert matrix[0][1] == -1.0


class TestFromAdjacencyMatrixBasics:
    """Basic functionality tests for from_adjacency_matrix."""

    def test_empty_matrix(self):
        """Empty matrix should create empty graph."""
        g = from_adjacency_matrix([])

        assert g.count_nodes() == 0
        assert g.count_edges() == 0

    def test_single_node(self):
        """1x1 zero matrix creates single isolated node."""
        g = from_adjacency_matrix([[0]])

        assert g.count_nodes() == 1
        assert g.count_edges() == 0

    def test_triangle(self):
        """Create a triangle from adjacency matrix."""
        matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

        g = from_adjacency_matrix(matrix)

        assert g.count_nodes() == 3
        assert g.count_edges() == 3

    def test_path(self):
        """Create a path from adjacency matrix."""
        matrix = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]

        g = from_adjacency_matrix(matrix)

        assert g.count_nodes() == 4
        assert g.count_edges() == 3


class TestFromAdjacencyMatrixLabels:
    """Tests for node labels."""

    def test_with_labels(self):
        """Create graph with node labels."""
        matrix = [[0, 1], [1, 0]]
        labels = ["Alice", "Bob"]
        g = from_adjacency_matrix(matrix, node_labels=labels)

        assert g.count_nodes() == 2

        # Check labels exist
        node_ids = list(g.get_node_ids())
        labels_found = [g.get_label_from_id(nid) for nid in node_ids]

        assert set(labels_found) == {"Alice", "Bob"}

    def test_label_mismatch_raises(self):
        """Wrong number of labels should raise error."""
        matrix = [[0, 1], [1, 0]]
        labels = ["Alice"]  # Only 1 label for 2x2 matrix

        with pytest.raises(ValueError, match="node_labels length"):
            from_adjacency_matrix(matrix, node_labels=labels)


class TestFromAdjacencyMatrixDirected:
    """Tests for directed graph creation."""

    def test_directed_from_asymmetric(self):
        """Asymmetric matrix creates directed edges."""
        matrix = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

        g = from_adjacency_matrix(matrix, directed=True)

        assert g.is_directed()
        assert g.count_edges() == 3

    def test_undirected_uses_upper_triangle(self):
        """Undirected mode only uses upper triangle."""
        # Asymmetric matrix, but undirected
        matrix = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]

        g = from_adjacency_matrix(matrix, directed=False)

        assert not g.is_directed()

        # Only upper triangle edges: (0,1) and (1,2)
        assert g.count_edges() == 2


class TestFromAdjacencyMatrixWeighted:
    """Tests for weighted graph creation."""

    def test_auto_detect_weighted(self):
        """Should auto-detect weighted from non-0/1 values."""
        matrix = [[0, 2.5], [2.5, 0]]

        g = from_adjacency_matrix(matrix)

        assert g.is_weighted()

    def test_force_unweighted(self):
        """Can force unweighted even with float values."""
        matrix = [[0, 2.5], [2.5, 0]]

        g = from_adjacency_matrix(matrix, weighted=False)

        assert not g.is_weighted()

    def test_force_weighted(self):
        """Can force weighted even with 0/1 values."""
        matrix = [[0, 1], [1, 0]]

        g = from_adjacency_matrix(matrix, weighted=True)

        assert g.is_weighted()


class TestFromAdjacencyMatrixValidation:
    """Tests for input validation."""

    def test_non_square_raises(self):
        """Non-square matrix should raise error."""
        matrix = [[0, 1, 0], [1, 0, 1]]  # 2x3 matrix

        with pytest.raises(ValueError, match="not square"):
            from_adjacency_matrix(matrix)

    def test_ragged_matrix_raises(self):
        """Ragged matrix should raise error."""
        matrix = [[0, 1], [1, 0, 1]]  # Second row has 3 elements

        with pytest.raises(ValueError, match="not square"):
            from_adjacency_matrix(matrix)


class TestRoundTrip:
    """Tests for round-trip conversion."""

    def test_undirected_unweighted_roundtrip(self):
        """Undirected unweighted graph should survive round-trip."""
        original = ec.Graph().add_full_graph(4)
        matrix, node_ids = to_adjacency_matrix(original)
        reconstructed = from_adjacency_matrix(matrix)

        assert original.count_nodes() == reconstructed.count_nodes()
        assert original.count_edges() == reconstructed.count_edges()

    def test_directed_roundtrip(self):
        """Directed graph should survive round-trip."""
        original = ec.Graph(directed=True)
        original.add_node(label="A")
        original.add_node(label="B")
        original.add_node(label="C")
        original.add_edge("A", "B")
        original.add_edge("B", "C")

        matrix, node_ids = to_adjacency_matrix(original)
        reconstructed = from_adjacency_matrix(matrix, directed=True)

        assert original.count_nodes() == reconstructed.count_nodes()
        assert original.count_edges() == reconstructed.count_edges()

    def test_weighted_roundtrip(self):
        """Weighted graph should preserve weights."""
        original = ec.Graph(weighted=True)
        original.add_node(label="A")
        original.add_node(label="B")
        original.add_edge("A", "B", weight=3.14)

        matrix, node_ids = to_adjacency_matrix(original)
        reconstructed = from_adjacency_matrix(matrix)

        assert reconstructed.is_weighted()

        # The weight should be preserved
        edges = list(reconstructed.get_edge_ids())
        edge = reconstructed.get_edge(edges[0])

        assert abs(edge.weight - 3.14) < 0.001


class TestLargerGraphs:
    """Tests with larger graphs."""

    def test_large_random_graph(self):
        """Test with a larger random graph."""
        g = ec.Graph().add_gnp_graph(50, 0.3, seed=23)

        matrix, node_ids = to_adjacency_matrix(g)

        assert len(matrix) == 50
        assert len(node_ids) == 50
        assert all(len(row) == 50 for row in matrix)

    def test_large_roundtrip(self):
        """Round-trip with larger graph."""
        original = ec.Graph().add_gnp_graph(30, 0.2, seed=123)

        matrix, node_ids = to_adjacency_matrix(original)
        reconstructed = from_adjacency_matrix(matrix)

        assert original.count_nodes() == reconstructed.count_nodes()
        assert original.count_edges() == reconstructed.count_edges()


class TestSelfLoops:
    """Tests involving self-loops."""

    def test_self_loop_in_matrix(self):
        """Diagonal entries represent self-loops."""
        matrix = [[1, 0], [0, 0]]  # Self-loop on node 0

        g = from_adjacency_matrix(matrix)

        assert g.count_nodes() == 2
        assert g.count_edges() == 1  # The self-loop

    def test_graph_with_self_loop_to_matrix(self):
        """Self-loops should appear on diagonal."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_edge("A", "A")  # Self-loop

        matrix, node_ids = to_adjacency_matrix(g)

        # Diagonal should be 1
        assert matrix[0][0] == 1.0


class TestModuleExports:
    """Tests that functions are properly exported."""

    def test_available_from_edgecase(self):
        """Functions should be available from main edgecase module."""
        assert hasattr(ec, "to_adjacency_matrix")
        assert hasattr(ec, "from_adjacency_matrix")

    def test_available_from_matrix_utils(self):
        """Functions should be available from matrix_utils module."""
        from edgecase import matrix_utils

        assert hasattr(matrix_utils, "to_adjacency_matrix")
        assert hasattr(matrix_utils, "from_adjacency_matrix")

    def test_ec_functions_work(self):
        """Functions from ec module should work correctly."""
        g = ec.Graph().add_path_graph(3)

        matrix, node_ids = ec.to_adjacency_matrix(g)

        assert len(matrix) == 3

        g2 = ec.from_adjacency_matrix([[0, 1], [1, 0]])

        assert g2.count_nodes() == 2
