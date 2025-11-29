import pytest
from edgecase import Graph


class TestCommonNeighbors:
    """Tests for common_neighbors method."""

    def test_common_neighbors_basic(self):
        """Test basic common neighbors calculation."""
        graph = Graph(directed=False)
        # Create triangle with an extra node
        # 1 -- 2
        # |    |
        # 3    4
        # 2 and 3 have 1 as common neighbor
        for _ in range(4):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 4)

        predictions = graph.common_neighbors([(2, 3)])
        assert len(predictions) == 1
        u, v, count = predictions[0]
        assert {u, v} == {2, 3}
        assert count == 1  # Node 1 is common neighbor

    def test_common_neighbors_no_common(self):
        """Test when nodes have no common neighbors."""
        graph = Graph(directed=False)
        for _ in range(4):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(3, 4)

        predictions = graph.common_neighbors([(2, 3)])
        assert len(predictions) == 1
        _, _, count = predictions[0]
        assert count == 0

    def test_common_neighbors_multiple_common(self):
        """Test with multiple common neighbors."""
        graph = Graph(directed=False)
        # Node 1 connected to 2,3,4
        # Node 5 connected to 2,3,4
        # Nodes 1 and 5 have 3 common neighbors: 2, 3, 4
        for _ in range(5):
            graph.add_node()
        for n in [2, 3, 4]:
            graph.add_edge(1, n)
            graph.add_edge(5, n)

        predictions = graph.common_neighbors([(1, 5)])
        assert len(predictions) == 1
        _, _, count = predictions[0]
        assert count == 3

    def test_common_neighbors_all_pairs(self):
        """Test with ebunch=None returns all non-adjacent pairs."""
        graph = Graph(directed=False)
        graph.add_path_graph(4)  # 1-2-3-4

        predictions = graph.common_neighbors()
        # Non-adjacent pairs: (1,3), (1,4), (2,4)
        assert len(predictions) == 3

    def test_common_neighbors_with_labels(self):
        """Test using node labels."""
        graph = Graph(directed=False)
        graph.add_node(label="Alice")
        graph.add_node(label="Bob")
        graph.add_node(label="Carol")
        graph.add_edge(source="Alice", target="Bob")
        graph.add_edge(source="Bob", target="Carol")

        predictions = graph.common_neighbors([("Alice", "Carol")])
        assert len(predictions) == 1
        _, _, count = predictions[0]
        assert count == 1  # Bob is common neighbor


class TestJaccardCoefficient:
    """Tests for jaccard_coefficient method."""

    def test_jaccard_identical_neighborhoods(self):
        """Test Jaccard when neighborhoods are identical."""
        graph = Graph(directed=False)
        # Nodes 1 and 2 both connected only to node 3
        for _ in range(3):
            graph.add_node()
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)

        predictions = graph.jaccard_coefficient([(1, 2)])
        _, _, coef = predictions[0]
        assert coef == 1.0  # |{3}| / |{3}| = 1

    def test_jaccard_disjoint_neighborhoods(self):
        """Test Jaccard when neighborhoods are disjoint."""
        graph = Graph(directed=False)
        for _ in range(4):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(3, 4)

        predictions = graph.jaccard_coefficient([(1, 3)])
        _, _, coef = predictions[0]
        assert coef == 0.0  # |{}| / |{2,4}| = 0

    def test_jaccard_partial_overlap(self):
        """Test Jaccard with partial neighborhood overlap."""
        graph = Graph(directed=False)
        # 1 connected to 3,4
        # 2 connected to 3,5
        # Intersection: {3}, Union: {3,4,5}
        # Jaccard = 1/3
        for _ in range(5):
            graph.add_node()
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)
        graph.add_edge(2, 3)
        graph.add_edge(2, 5)

        predictions = graph.jaccard_coefficient([(1, 2)])
        _, _, coef = predictions[0]
        assert abs(coef - 1 / 3) < 1e-10

    def test_jaccard_isolated_nodes(self):
        """Test Jaccard with isolated nodes (no neighbors)."""
        graph = Graph(directed=False)
        for _ in range(2):
            graph.add_node()

        predictions = graph.jaccard_coefficient([(1, 2)])
        _, _, coef = predictions[0]
        assert coef == 0.0  # Empty union


class TestAdamicAdarIndex:
    """Tests for adamic_adar_index method."""

    def test_adamic_adar_basic(self):
        """Test basic Adamic-Adar calculation."""
        graph = Graph(directed=False)
        # Nodes 1 and 3 share node 2 as common neighbor
        # Node 2 has degree 2
        for _ in range(3):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)

        predictions = graph.adamic_adar_index([(1, 3)])
        _, _, index = predictions[0]
        import math

        expected = 1 / math.log(2)  # 1 / log(degree of 2)
        assert abs(index - expected) < 1e-10

    def test_adamic_adar_high_degree_neighbor(self):
        """Test that high-degree common neighbors contribute less."""
        graph = Graph(directed=False)
        # Hub node 1 connected to all others
        for _ in range(10):
            graph.add_node()
        for i in range(2, 11):
            graph.add_edge(1, i)

        # Nodes 2 and 3 share node 1 (degree 9)
        predictions = graph.adamic_adar_index([(2, 3)])
        _, _, index = predictions[0]
        import math

        expected = 1 / math.log(9)
        assert abs(index - expected) < 1e-10

    def test_adamic_adar_multiple_common(self):
        """Test with multiple common neighbors of different degrees."""
        graph = Graph(directed=False)
        # Create network where 1 and 4 have common neighbors 2 and 3
        for _ in range(6):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 4)
        graph.add_edge(3, 4)
        # Add extra edge to node 2 to give it higher degree
        graph.add_edge(2, 5)

        predictions = graph.adamic_adar_index([(1, 4)])
        _, _, index = predictions[0]
        import math

        # Node 2 has degree 3, Node 3 has degree 2
        expected = 1 / math.log(3) + 1 / math.log(2)
        assert abs(index - expected) < 1e-10

    def test_adamic_adar_no_common(self):
        """Test Adamic-Adar with no common neighbors."""
        graph = Graph(directed=False)
        for _ in range(4):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(3, 4)

        predictions = graph.adamic_adar_index([(1, 3)])
        _, _, index = predictions[0]
        assert index == 0.0


class TestResourceAllocationIndex:
    """Tests for resource_allocation_index method."""

    def test_resource_allocation_basic(self):
        """Test basic resource allocation calculation."""
        graph = Graph(directed=False)
        # Nodes 1 and 3 share node 2 (degree 2)
        for _ in range(3):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)

        predictions = graph.resource_allocation_index([(1, 3)])
        _, _, index = predictions[0]
        expected = 1 / 2  # 1 / degree(2)
        assert abs(index - expected) < 1e-10

    def test_resource_allocation_high_degree(self):
        """Test that high-degree neighbors contribute less than Adamic-Adar."""
        graph = Graph(directed=False)
        # Hub with degree 9
        for _ in range(10):
            graph.add_node()
        for i in range(2, 11):
            graph.add_edge(1, i)

        predictions = graph.resource_allocation_index([(2, 3)])
        _, _, index = predictions[0]
        expected = 1 / 9  # More penalized than Adamic-Adar's 1/log(9)
        assert abs(index - expected) < 1e-10

    def test_resource_allocation_vs_adamic_adar(self):
        """Compare resource allocation and Adamic-Adar indices."""
        graph = Graph(directed=False)
        # Hub with high degree
        for _ in range(20):
            graph.add_node()
        for i in range(2, 21):
            graph.add_edge(1, i)

        ra_preds = graph.resource_allocation_index([(2, 3)])
        aa_preds = graph.adamic_adar_index([(2, 3)])

        _, _, ra_index = ra_preds[0]
        _, _, aa_index = aa_preds[0]

        # RA should penalize more: 1/19 < 1/log(19)
        assert ra_index < aa_index


class TestPreferentialAttachment:
    """Tests for preferential_attachment method."""

    def test_preferential_attachment_basic(self):
        """Test basic preferential attachment calculation."""
        graph = Graph(directed=False)
        # Node 1 has degree 2, Node 4 has degree 1
        for _ in range(4):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(3, 4)

        predictions = graph.preferential_attachment([(1, 4)])
        _, _, score = predictions[0]
        assert score == 2 * 1  # degree(1) * degree(4)

    def test_preferential_attachment_high_degree(self):
        """Test that high-degree nodes get higher scores."""
        graph = Graph(directed=False)
        # Hub node 1 with degree 9
        for _ in range(10):
            graph.add_node()
        for i in range(2, 11):
            graph.add_edge(1, i)

        predictions = graph.preferential_attachment([(2, 3)])
        _, _, score = predictions[0]
        assert score == 1 * 1  # Both peripheral nodes have degree 1

        predictions2 = graph.preferential_attachment([(1, 2)])
        # But 1 and 2 are already connected, so this shouldn't be in default ebunch

    def test_preferential_attachment_isolated(self):
        """Test with isolated node (degree 0)."""
        graph = Graph(directed=False)
        for _ in range(3):
            graph.add_node()
        graph.add_edge(1, 2)
        # Node 3 is isolated

        predictions = graph.preferential_attachment([(1, 3)])
        _, _, score = predictions[0]
        assert score == 1 * 0  # degree(1)=1, degree(3)=0

    def test_preferential_attachment_complete_graph(self):
        """Test on complete graph with one edge removed."""
        graph = Graph(directed=False)
        graph.add_full_graph(4)  # K4

        # Remove edge between 1 and 2
        edge_ids = graph.find_edge_ids(1, 2)
        graph.delete_edge(id=edge_ids[0])

        # All other nodes have degree 3, but 1 and 2 have degree 2
        predictions = graph.preferential_attachment([(1, 2)])
        _, _, score = predictions[0]
        assert score == 2 * 2


class TestLinkPredictionEdgeCases:
    """Edge case tests for link prediction methods."""

    def test_self_loop_not_included(self):
        """Test that self-loops are not included in default ebunch."""
        graph = Graph(directed=False)
        graph.add_node()
        graph.add_node()
        graph.add_edge(1, 2)

        predictions = graph.common_neighbors()
        # Should not include (1,1) or (2,2)
        for u, v, _ in predictions:
            assert u != v

    def test_empty_ebunch(self):
        """Test with empty ebunch list."""
        graph = Graph(directed=False)
        graph.add_path_graph(3)

        predictions = graph.common_neighbors([])
        assert predictions == []

    def test_single_node_graph(self):
        """Test with single node graph."""
        graph = Graph(directed=False)
        graph.add_node()

        predictions = graph.common_neighbors()
        assert predictions == []

    def test_directed_graph(self):
        """Test link prediction on directed graph."""
        graph = Graph(directed=True)
        for _ in range(3):
            graph.add_node()
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)

        # Should work on directed graphs
        predictions = graph.common_neighbors([(1, 3)])
        assert len(predictions) == 1

    def test_mixed_labels_and_ids(self):
        """Test mixing labels and IDs in ebunch."""
        graph = Graph(directed=False)
        graph.add_node(label="A")
        graph.add_node(label="B")
        graph.add_node(label="C")
        graph.add_edge(source="A", target="B")
        graph.add_edge(source="B", target="C")

        a_id = graph.get_id_from_label("A")

        # Mix ID and label
        predictions = graph.common_neighbors([(a_id, "C")])
        assert len(predictions) == 1


class TestLinkPredictionNetworkX:
    """Tests comparing with expected NetworkX-like behavior."""

    def test_path_graph_predictions(self):
        """Test on path graph with known structure."""
        graph = Graph(directed=False)
        graph.add_path_graph(5)  # 1-2-3-4-5

        # Check specific predictions
        cn = dict(((min(u, v), max(u, v)), c) for u, v, c in graph.common_neighbors())

        # 1 and 3 have common neighbor 2
        assert cn[(1, 3)] == 1
        # 1 and 4 have no common neighbors
        assert cn[(1, 4)] == 0
        # 1 and 5 have no common neighbors
        assert cn[(1, 5)] == 0
        # 2 and 4 have common neighbor 3
        assert cn[(2, 4)] == 1
        # 2 and 5 have no common neighbors
        assert cn[(2, 5)] == 0
        # 3 and 5 have common neighbor 4
        assert cn[(3, 5)] == 1

    def test_cycle_graph_predictions(self):
        """Test on cycle graph with known structure."""
        graph = Graph(directed=False)
        graph.add_cycle_graph(6)  # 1-2-3-4-5-6-1

        # In cycle, non-adjacent nodes 1 and 3 share neighbor 2
        # Node 1 also has neighbor 6
        predictions = graph.common_neighbors([(1, 3)])
        _, _, count = predictions[0]
        assert count == 1  # Just node 2

    def test_star_graph_predictions(self):
        """Test on star graph with known structure."""
        graph = Graph(directed=False)
        graph.add_star_graph(5)  # Hub 1 connected to 2,3,4,5

        # All peripheral nodes share hub as common neighbor
        predictions = graph.common_neighbors([(2, 3)])
        _, _, count = predictions[0]
        assert count == 1  # Hub node 1

        # Preferential attachment: peripheral nodes have degree 1
        pa_preds = graph.preferential_attachment([(2, 3)])
        _, _, score = pa_preds[0]
        assert score == 1  # 1 * 1
