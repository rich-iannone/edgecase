import pytest
from edgecase import Graph


class TestSimilarityBasics:
    """Basic similarity computation tests."""

    def test_jaccard_identical_neighborhoods(self):
        """Nodes with identical neighborhoods should have Jaccard similarity of 1.0."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        # A and B both connect to C and D
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")

        sim = g.pairwise_similarity(["A", "B"], method="jaccard")
        assert sim[0][1] == pytest.approx(1.0)
        assert sim[1][0] == pytest.approx(1.0)

    def test_jaccard_no_common_neighbors(self):
        """Nodes with no common neighbors should have Jaccard similarity of 0.0."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "D")

        sim = g.pairwise_similarity(["A", "B"], method="jaccard")
        assert sim[0][1] == pytest.approx(0.0)

    def test_jaccard_partial_overlap(self):
        """Test Jaccard with partial neighborhood overlap."""
        g = Graph()
        for node in ["A", "B", "C", "D", "E"]:
            g.add_node(node)
        # A neighbors: {C, D}
        # B neighbors: {C, E}
        # Intersection: {C}, Union: {C, D, E}
        # Jaccard = 1/3
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "E")

        sim = g.pairwise_similarity(["A", "B"], method="jaccard")
        assert sim[0][1] == pytest.approx(1.0 / 3.0)

    def test_dice_identical_neighborhoods(self):
        """Nodes with identical neighborhoods should have Dice similarity of 1.0."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")

        sim = g.pairwise_similarity(["A", "B"], method="dice")
        assert sim[0][1] == pytest.approx(1.0)

    def test_dice_no_common_neighbors(self):
        """Nodes with no common neighbors should have Dice similarity of 0.0."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "D")

        sim = g.pairwise_similarity(["A", "B"], method="dice")
        assert sim[0][1] == pytest.approx(0.0)

    def test_dice_partial_overlap(self):
        """Test Dice with partial neighborhood overlap."""
        g = Graph()
        for node in ["A", "B", "C", "D", "E"]:
            g.add_node(node)
        # A neighbors: {C, D}
        # B neighbors: {C, E}
        # Dice = 2 * 1 / (2 + 2) = 0.5
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "E")

        sim = g.pairwise_similarity(["A", "B"], method="dice")
        assert sim[0][1] == pytest.approx(0.5)

    def test_cosine_identical_neighborhoods(self):
        """Nodes with identical neighborhoods should have cosine similarity of 1.0."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")

        sim = g.pairwise_similarity(["A", "B"], method="cosine")
        assert sim[0][1] == pytest.approx(1.0)

    def test_cosine_no_common_neighbors(self):
        """Nodes with no common neighbors should have cosine similarity of 0.0."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "D")

        sim = g.pairwise_similarity(["A", "B"], method="cosine")
        assert sim[0][1] == pytest.approx(0.0)

    def test_cosine_partial_overlap(self):
        """Test cosine with partial neighborhood overlap."""
        g = Graph()
        for node in ["A", "B", "C", "D", "E"]:
            g.add_node(node)
        # A neighbors: {C, D}
        # B neighbors: {C, E}
        # Cosine = 1 / sqrt(2 * 2) = 0.5
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "E")

        sim = g.pairwise_similarity(["A", "B"], method="cosine")
        assert sim[0][1] == pytest.approx(0.5)

    def test_invlogweighted_identical_neighborhoods(self):
        """Test inverse log-weighted with identical neighborhoods."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")
        # Common neighbors: C and D
        # Each has degree 2
        # Score = 1/log(2) + 1/log(2) = 2/log(2) ≈ 2.885

        sim = g.pairwise_similarity(["A", "B"], method="invlogweighted")
        import math

        expected = 2.0 / math.log(2)
        assert sim[0][1] == pytest.approx(expected)

    def test_invlogweighted_no_common_neighbors(self):
        """Nodes with no common neighbors should have inverse log-weighted similarity of 0.0."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "D")

        sim = g.pairwise_similarity(["A", "B"], method="invlogweighted")
        assert sim[0][1] == pytest.approx(0.0)

    def test_invlogweighted_rare_neighbor(self):
        """Rare neighbors should contribute more to inverse log-weighted score."""
        g = Graph()
        for node in ["A", "B", "C", "D", "E", "F"]:
            g.add_node(node)
        # A and B share neighbor C (degree 2)
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        # C also connects to D, E, F (total degree 5)
        g.add_edge("C", "D")
        g.add_edge("C", "E")
        g.add_edge("C", "F")

        sim = g.pairwise_similarity(["A", "B"], method="invlogweighted")
        import math

        # C has degree 5
        expected = 1.0 / math.log(5)
        assert sim[0][1] == pytest.approx(expected)


class TestSimilarityDiagonal:
    """Test that diagonal elements (self-similarity) are always 1.0."""

    def test_jaccard_diagonal(self):
        """Jaccard self-similarity should be 1.0."""
        g = Graph()
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")

        sim = g.pairwise_similarity(["A", "B", "C"], method="jaccard")
        assert sim[0][0] == pytest.approx(1.0)
        assert sim[1][1] == pytest.approx(1.0)
        assert sim[2][2] == pytest.approx(1.0)

    def test_dice_diagonal(self):
        """Dice self-similarity should be 1.0."""
        g = Graph()
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")

        sim = g.pairwise_similarity(["A", "B", "C"], method="dice")
        assert sim[0][0] == pytest.approx(1.0)
        assert sim[1][1] == pytest.approx(1.0)
        assert sim[2][2] == pytest.approx(1.0)

    def test_cosine_diagonal(self):
        """Cosine self-similarity should be 1.0."""
        g = Graph()
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")

        sim = g.pairwise_similarity(["A", "B", "C"], method="cosine")
        assert sim[0][0] == pytest.approx(1.0)
        assert sim[1][1] == pytest.approx(1.0)
        assert sim[2][2] == pytest.approx(1.0)

    def test_invlogweighted_diagonal(self):
        """Inverse log-weighted self-similarity should be 1.0."""
        g = Graph()
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")

        sim = g.pairwise_similarity(["A", "B", "C"], method="invlogweighted")
        assert sim[0][0] == pytest.approx(1.0)
        assert sim[1][1] == pytest.approx(1.0)
        assert sim[2][2] == pytest.approx(1.0)


class TestSimilaritySymmetry:
    """Test that similarity matrices are symmetric."""

    def test_jaccard_symmetry(self):
        """Jaccard similarity should be symmetric."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        sim = g.pairwise_similarity(["A", "B", "C", "D"], method="jaccard")
        for i in range(4):
            for j in range(4):
                assert sim[i][j] == pytest.approx(sim[j][i])

    def test_dice_symmetry(self):
        """Dice similarity should be symmetric."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        sim = g.pairwise_similarity(["A", "B", "C", "D"], method="dice")
        for i in range(4):
            for j in range(4):
                assert sim[i][j] == pytest.approx(sim[j][i])

    def test_cosine_symmetry(self):
        """Cosine similarity should be symmetric."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        sim = g.pairwise_similarity(["A", "B", "C", "D"], method="cosine")
        for i in range(4):
            for j in range(4):
                assert sim[i][j] == pytest.approx(sim[j][i])

    def test_invlogweighted_symmetry(self):
        """Inverse log-weighted similarity should be symmetric."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        sim = g.pairwise_similarity(["A", "B", "C", "D"], method="invlogweighted")
        for i in range(4):
            for j in range(4):
                assert sim[i][j] == pytest.approx(sim[j][i])


class TestSimilarityDirectedGraphs:
    """Test similarity on directed graphs."""

    def test_jaccard_directed(self):
        """Test Jaccard similarity on directed graph."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("A", "D")

        sim = g.pairwise_similarity(["A", "B"], method="jaccard")
        # A neighbors: {C, D}, B neighbors: {C}
        # Intersection: {C}, Union: {C, D}
        # Jaccard = 1/2 = 0.5
        assert sim[0][1] == pytest.approx(0.5)

    def test_dice_directed(self):
        """Test Dice similarity on directed graph."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("A", "D")

        sim = g.pairwise_similarity(["A", "B"], method="dice")
        # A neighbors: {C, D}, B neighbors: {C}
        # Dice = 2 * 1 / (2 + 1) = 2/3
        assert sim[0][1] == pytest.approx(2.0 / 3.0)

    def test_cosine_directed(self):
        """Test cosine similarity on directed graph."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("A", "D")

        sim = g.pairwise_similarity(["A", "B"], method="cosine")
        # A neighbors: {C, D}, B neighbors: {C}
        # Cosine = 1 / sqrt(2 * 1) = 1/sqrt(2) ≈ 0.707
        import math

        assert sim[0][1] == pytest.approx(1.0 / math.sqrt(2))

    def test_invlogweighted_directed(self):
        """Test inverse log-weighted similarity on directed graph."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D", "E"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("C", "E")

        sim = g.pairwise_similarity(["A", "B"], method="invlogweighted")
        # Common neighbor: C (degree 2)
        import math

        expected = 1.0 / math.log(2)
        assert sim[0][1] == pytest.approx(expected)


class TestSimilarityEdgeCases:
    """Test edge cases and special scenarios."""

    def test_isolated_nodes(self):
        """Isolated nodes should have zero similarity with all other nodes."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "B")
        # C and D are isolated

        sim = g.pairwise_similarity(["A", "C", "D"], method="jaccard")
        assert sim[1][0] == pytest.approx(0.0)  # C-A
        assert sim[2][0] == pytest.approx(0.0)  # D-A
        assert sim[1][2] == pytest.approx(0.0)  # C-D

    def test_single_node(self):
        """Single node should have self-similarity of 1.0."""
        g = Graph()
        g.add_node("A")

        sim = g.pairwise_similarity(["A"], method="jaccard")
        assert len(sim) == 1
        assert len(sim[0]) == 1
        assert sim[0][0] == pytest.approx(1.0)

    def test_empty_node_list_default(self):
        """Empty node list should compute for all nodes."""
        g = Graph()
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")

        sim = g.pairwise_similarity([], method="jaccard")
        assert len(sim) == 3
        assert len(sim[0]) == 3

    def test_star_graph(self):
        """Test similarity in a star graph (one central node)."""
        g = Graph()
        for node in ["center", "A", "B", "C"]:
            g.add_node(node)
        g.add_edge("center", "A")
        g.add_edge("center", "B")
        g.add_edge("center", "C")

        # A, B, C all have only 'center' as neighbor
        # So they should all have Jaccard = 1.0 with each other
        sim = g.pairwise_similarity(["A", "B", "C"], method="jaccard")
        assert sim[0][1] == pytest.approx(1.0)
        assert sim[0][2] == pytest.approx(1.0)
        assert sim[1][2] == pytest.approx(1.0)

    def test_complete_graph(self):
        """Test similarity in a complete graph."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        # Make complete graph
        nodes = ["A", "B", "C", "D"]
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1 :]:
                g.add_edge(n1, n2)

        # All pairs (except diagonal) should have Jaccard = 2/3
        # Each node has 3 neighbors, any two nodes share 2 common neighbors
        # Jaccard = 2 / (3 + 3 - 2) = 2/4 = 0.5
        sim = g.pairwise_similarity(["A", "B", "C", "D"], method="jaccard")
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert sim[i][j] == pytest.approx(0.5)


class TestSimilarityErrors:
    """Test error handling."""

    def test_invalid_node(self):
        """Should raise error for invalid node."""
        g = Graph()
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")

        with pytest.raises(Exception, match="Node.*not found"):
            g.pairwise_similarity(["A", "X"], method="jaccard")

    def test_invalid_method(self):
        """Should raise error for invalid method."""
        g = Graph()
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")

        with pytest.raises(Exception, match="Unknown similarity method"):
            g.pairwise_similarity(["A", "B"], method="invalid")

    def test_empty_graph(self):
        """Empty graph should handle gracefully."""
        g = Graph()

        sim = g.pairwise_similarity([], method="jaccard")
        assert len(sim) == 0


class TestSimilarityMetricComparison:
    """Compare behavior of different metrics."""

    def test_dice_vs_jaccard_weighting(self):
        """Dice should give more weight to common neighbors than Jaccard."""
        g = Graph()
        for node in ["A", "B", "C", "D", "E"]:
            g.add_node(node)
        # A: {C, D}
        # B: {C, E}
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "E")

        sim_jac = g.pairwise_similarity(["A", "B"], method="jaccard")
        sim_dice = g.pairwise_similarity(["A", "B"], method="dice")

        # Jaccard = 1/3, Dice = 1/2
        assert sim_dice[0][1] > sim_jac[0][1]

    def test_all_metrics_zero_for_no_overlap(self):
        """All metrics should return 0 for no common neighbors."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("B", "D")

        for method in ["jaccard", "dice", "cosine", "invlogweighted"]:
            sim = g.pairwise_similarity(["A", "B"], method=method)
            assert sim[0][1] == pytest.approx(0.0)

    def test_all_metrics_high_for_identical(self):
        """All metrics should return high values for identical neighborhoods."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")

        for method in ["jaccard", "dice", "cosine"]:
            sim = g.pairwise_similarity(["A", "B"], method=method)
            assert sim[0][1] == pytest.approx(1.0)

        # Inverse log-weighted will be high but not necessarily 1.0
        sim_inv = g.pairwise_similarity(["A", "B"], method="invlogweighted")
        assert sim_inv[0][1] > 2.0  # Should be 2/log(2) ≈ 2.885


class TestSimilarityRealWorldScenarios:
    """Test realistic graph scenarios."""

    def test_triangle_graph(self):
        """Test similarity in a triangle (3-clique)."""
        g = Graph()
        for node in ["A", "B", "C"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        # Each node has 2 neighbors
        # Any two nodes share 1 common neighbor
        # Jaccard = 1 / (2 + 2 - 1) = 1/3
        sim = g.pairwise_similarity(["A", "B", "C"], method="jaccard")
        assert sim[0][1] == pytest.approx(1.0 / 3.0)
        assert sim[0][2] == pytest.approx(1.0 / 3.0)
        assert sim[1][2] == pytest.approx(1.0 / 3.0)

    def test_path_graph(self):
        """Test similarity in a path graph A-B-C-D."""
        g = Graph()
        for node in ["A", "B", "C", "D"]:
            g.add_node(node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")

        # A and C share B as common neighbor
        # A neighbors: {B}, C neighbors: {B, D}
        # Jaccard = 1 / 2 = 0.5
        sim = g.pairwise_similarity(["A", "C"], method="jaccard")
        assert sim[0][1] == pytest.approx(0.5)

        # A and D have no common neighbors
        sim = g.pairwise_similarity(["A", "D"], method="jaccard")
        assert sim[0][1] == pytest.approx(0.0)

    def test_bipartite_graph(self):
        """Test similarity in a bipartite graph."""
        g = Graph()
        # Left nodes: A, B
        # Right nodes: X, Y
        for node in ["A", "B", "X", "Y"]:
            g.add_node(node)
        g.add_edge("A", "X")
        g.add_edge("A", "Y")
        g.add_edge("B", "X")
        g.add_edge("B", "Y")

        # A and B have identical neighborhoods {X, Y}
        sim = g.pairwise_similarity(["A", "B"], method="jaccard")
        assert sim[0][1] == pytest.approx(1.0)

        # X and Y have identical neighborhoods {A, B}
        sim = g.pairwise_similarity(["X", "Y"], method="jaccard")
        assert sim[0][1] == pytest.approx(1.0)


class TestSimilarityNodeIDSupport:
    """Test that pairwise_similarity accepts both node IDs and labels."""

    def test_accepts_node_ids(self):
        """Test that node IDs work correctly."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        # A and B both connect to C and D
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")

        # Using node IDs (1 and 2 for A and B)
        sim = g.pairwise_similarity([1, 2], method="jaccard")
        assert sim[0][1] == pytest.approx(1.0)
        assert sim[1][0] == pytest.approx(1.0)

    def test_mixed_ids_and_labels(self):
        """Test that mixing node IDs and labels works."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("A", "C")
        g.add_edge("A", "D")
        g.add_edge("B", "C")
        g.add_edge("B", "D")

        # Mix node ID (1) and label ("B")
        sim = g.pairwise_similarity([1, "B"], method="jaccard")
        assert sim[0][1] == pytest.approx(1.0)

    def test_node_ids_without_labels(self):
        """Test that node IDs work when labels aren't set."""
        g = Graph()
        # Create nodes without explicit labels
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_node()
        # Add edges using generated IDs (1, 2, 3, 4)
        g.add_edge(1, 3)
        g.add_edge(1, 4)
        g.add_edge(2, 3)
        g.add_edge(2, 4)

        # Nodes 1 and 2 have identical neighborhoods
        sim = g.pairwise_similarity([1, 2], method="jaccard")
        assert sim[0][1] == pytest.approx(1.0)

    def test_invalid_node_id(self):
        """Test error handling for invalid node ID."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")

        with pytest.raises(ValueError, match="Node with ID 999 not found"):
            g.pairwise_similarity([1, 999], method="jaccard")

    def test_all_similarity_methods_with_ids(self):
        """Test that all similarity methods work with node IDs."""
        g = Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "C")
        g.add_edge("B", "C")

        for method in ["jaccard", "dice", "cosine", "invlogweighted"]:
            # Should not raise any errors
            sim = g.pairwise_similarity([1, 2], method=method)
            assert len(sim) == 2
            assert len(sim[0]) == 2
