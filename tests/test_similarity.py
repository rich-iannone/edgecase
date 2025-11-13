"""Tests for similarity metric methods."""
import pytest
import edgecase as ec


class TestDiceSimilarity:
    """Test get_dice_similarity() method."""
    
    def test_identical_neighborhoods(self):
        """Nodes with identical neighbors should have Dice = 1.0."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('B', 'X')
        G.add_edge('B', 'Y')
        
        sim = G.get_dice_similarity()
        assert sim['A']['B'] == pytest.approx(1.0)
        assert sim['B']['A'] == pytest.approx(1.0)
    
    def test_no_common_neighbors(self):
        """Nodes with no common neighbors should have Dice = 0.0."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_edge('A', 'X')
        G.add_edge('B', 'Y')
        
        sim = G.get_dice_similarity()
        assert sim['A']['B'] == pytest.approx(0.0)
    
    def test_self_similarity(self):
        """Self-similarity should always be 1.0."""
        G = ec.Graph()
        G.add_star_graph(5)
        
        sim = G.get_dice_similarity()
        for node_id in G.node_ids(limit=None):
            node_label = G.get_label_from_id(node_id)
            assert sim[node_label][node_label] == pytest.approx(1.0)
    
    def test_partial_overlap(self):
        """Test Dice coefficient with partial neighborhood overlap."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_node('Z')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('B', 'Y')
        G.add_edge('B', 'Z')
        
        sim = G.get_dice_similarity()
        # A has 2 neighbors, B has 2 neighbors, 1 in common
        # Dice = 2 * 1 / (2 + 2) = 0.5
        assert sim['A']['B'] == pytest.approx(0.5)
    
    def test_star_graph(self):
        """In star graph, leaves have no common neighbors."""
        G = ec.Graph()
        G.add_star_graph(4)  # Hub + 3 leaves
        
        sim = G.get_dice_similarity()
        # Leaves should have Dice = 0 with each other
        assert sim['1']['2'] == pytest.approx(0.0)
        assert sim['1']['3'] == pytest.approx(0.0)
    
    def test_complete_graph(self):
        """In complete graph K4, all non-self pairs share 2 neighbors."""
        G = ec.Graph()
        G.add_full_graph(4)
        
        sim = G.get_dice_similarity()
        # Each node has 3 neighbors, any pair shares 2 neighbors
        # Dice = 2 * 2 / (3 + 3) = 4/6 ≈ 0.667
        for node1 in ['1', '2', '3', '4']:
            for node2 in ['1', '2', '3', '4']:
                if node1 != node2:
                    assert sim[node1][node2] == pytest.approx(0.667, abs=0.01)


class TestJaccardSimilarity:
    """Test get_jaccard_similarity() method."""
    
    def test_identical_neighborhoods(self):
        """Nodes with identical neighbors should have Jaccard = 1.0."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('B', 'X')
        G.add_edge('B', 'Y')
        
        sim = G.get_jaccard_similarity()
        assert sim['A']['B'] == pytest.approx(1.0)
    
    def test_no_common_neighbors(self):
        """Nodes with no common neighbors should have Jaccard = 0.0."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_edge('A', 'X')
        G.add_edge('B', 'Y')
        
        sim = G.get_jaccard_similarity()
        assert sim['A']['B'] == pytest.approx(0.0)
    
    def test_self_similarity(self):
        """Self-similarity should always be 1.0."""
        G = ec.Graph()
        G.add_cycle_graph(5)
        
        sim = G.get_jaccard_similarity()
        for node_id in G.node_ids(limit=None):
            node_label = G.get_label_from_id(node_id)
            assert sim[node_label][node_label] == pytest.approx(1.0)
    
    def test_partial_overlap(self):
        """Test Jaccard index with partial neighborhood overlap."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_node('Z')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('B', 'Y')
        G.add_edge('B', 'Z')
        
        sim = G.get_jaccard_similarity()
        # A neighbors: {X, Y}, B neighbors: {Y, Z}
        # Intersection: {Y} (size 1)
        # Union: {X, Y, Z} (size 3)
        # Jaccard = 1/3 ≈ 0.333
        assert sim['A']['B'] == pytest.approx(0.333, abs=0.01)
    
    def test_jaccard_vs_dice(self):
        """Jaccard should always be ≤ Dice coefficient."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('W')
        G.add_node('X')
        G.add_node('Y')
        G.add_node('Z')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('A', 'Z')
        G.add_edge('B', 'Y')
        G.add_edge('B', 'Z')
        G.add_edge('B', 'W')
        
        dice = G.get_dice_similarity()
        jaccard = G.get_jaccard_similarity()
        
        assert jaccard['A']['B'] <= dice['A']['B']
    
    def test_star_graph(self):
        """In star graph, leaves have Jaccard = 0."""
        G = ec.Graph()
        G.add_star_graph(4)
        
        sim = G.get_jaccard_similarity()
        assert sim['1']['2'] == pytest.approx(0.0)


class TestCosineSimilarity:
    """Test get_cosine_similarity() method."""
    
    def test_identical_neighborhoods(self):
        """Nodes with identical neighbors should have Cosine = 1.0."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('B', 'X')
        G.add_edge('B', 'Y')
        
        sim = G.get_cosine_similarity()
        assert sim['A']['B'] == pytest.approx(1.0)
    
    def test_no_common_neighbors(self):
        """Nodes with no common neighbors should have Cosine = 0.0."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_edge('A', 'X')
        G.add_edge('B', 'Y')
        
        sim = G.get_cosine_similarity()
        assert sim['A']['B'] == pytest.approx(0.0)
    
    def test_self_similarity(self):
        """Self-similarity should always be 1.0."""
        G = ec.Graph()
        G.add_path_graph(5)
        
        sim = G.get_cosine_similarity()
        for node_id in G.node_ids(limit=None):
            node_label = G.get_label_from_id(node_id)
            assert sim[node_label][node_label] == pytest.approx(1.0)
    
    def test_partial_overlap(self):
        """Test Cosine similarity with partial neighborhood overlap."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('W')
        G.add_node('X')
        G.add_node('Y')
        G.add_node('Z')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('A', 'Z')
        G.add_edge('A', 'W')  # A has 4 neighbors
        G.add_edge('B', 'Y')
        G.add_edge('B', 'Z')  # B has 2 neighbors, 2 in common
        
        sim = G.get_cosine_similarity()
        # Cosine = 2 / (sqrt(4) * sqrt(2)) = 2 / (2 * 1.414) ≈ 0.707
        assert sim['A']['B'] == pytest.approx(0.707, abs=0.01)
    
    def test_cycle_graph(self):
        """In cycle, adjacent nodes share one neighbor."""
        G = ec.Graph()
        G.add_cycle_graph(5)
        
        sim = G.get_cosine_similarity()
        # Each node has 2 neighbors, adjacent nodes share 1 neighbor (the node 2-hops away)
        # Cosine = 1 / (sqrt(2) * sqrt(2)) = 1/2 = 0.5
        # From the actual graph: 1 is adjacent to 3 and 4
        assert sim['1']['3'] == pytest.approx(0.5)
        assert sim['1']['4'] == pytest.approx(0.5)
    
    def test_between_dice_and_jaccard(self):
        """Cosine typically falls between Dice and Jaccard."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_node('Z')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('B', 'Y')
        G.add_edge('B', 'Z')
        
        dice = G.get_dice_similarity()
        cosine = G.get_cosine_similarity()
        jaccard = G.get_jaccard_similarity()
        
        # For this graph: Dice > Cosine > Jaccard
        assert dice['A']['B'] >= cosine['A']['B'] >= jaccard['A']['B']


class TestOverlapSimilarity:
    """Test get_overlap_similarity() method."""
    
    def test_identical_neighborhoods(self):
        """Nodes with identical neighbors should have Overlap = 1.0."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('B', 'X')
        G.add_edge('B', 'Y')
        
        sim = G.get_overlap_similarity()
        assert sim['A']['B'] == pytest.approx(1.0)
    
    def test_no_common_neighbors(self):
        """Nodes with no common neighbors should have Overlap = 0.0."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_edge('A', 'X')
        G.add_edge('B', 'Y')
        
        sim = G.get_overlap_similarity()
        assert sim['A']['B'] == pytest.approx(0.0)
    
    def test_self_similarity(self):
        """Self-similarity should always be 1.0."""
        G = ec.Graph()
        G.add_full_graph(4)
        
        sim = G.get_overlap_similarity()
        for node_id in G.node_ids(limit=None):
            node_label = G.get_label_from_id(node_id)
            assert sim[node_label][node_label] == pytest.approx(1.0)
    
    def test_subset_relationship(self):
        """When one neighborhood is subset of another, Overlap = 1.0."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_node('Z')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('A', 'Z')
        G.add_edge('B', 'X')
        G.add_edge('B', 'Y')
        
        sim = G.get_overlap_similarity()
        # B's neighbors {X, Y} ⊂ A's neighbors {X, Y, Z}
        # Overlap = 2 / min(3, 2) = 2/2 = 1.0
        assert sim['A']['B'] == pytest.approx(1.0)
        assert sim['B']['A'] == pytest.approx(1.0)
    
    def test_hub_spoke(self):
        """Test overlap in more complex pattern."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('C')
        # Create triangle
        G.add_edge('A', 'B')
        G.add_edge('A', 'C')
        G.add_edge('B', 'C')
        # A and B both have C as neighbor
        # A neighbors: {B, C}, B neighbors: {A, C}
        # Common neighbors: {C}
        
        sim = G.get_overlap_similarity()
        # They share C, min size is 2
        # Overlap = 1 / min(2, 2) = 0.5
        assert sim['A']['B'] == pytest.approx(0.5)
    
    def test_asymmetric_sizes(self):
        """Overlap emphasizes smaller neighborhood."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('W')
        G.add_node('X')
        G.add_node('Y')
        G.add_node('Z')
        G.add_edge('A', 'X')  # A has 1 neighbor
        G.add_edge('B', 'X')
        G.add_edge('B', 'Y')
        G.add_edge('B', 'Z')
        G.add_edge('B', 'W')  # B has 4 neighbors
        
        sim = G.get_overlap_similarity()
        # They share 1 neighbor, min size is 1
        # Overlap = 1 / 1 = 1.0
        assert sim['A']['B'] == pytest.approx(1.0)
    
    def test_star_graph(self):
        """In star graph, leaves have Overlap = 0."""
        G = ec.Graph()
        G.add_star_graph(4)
        
        sim = G.get_overlap_similarity()
        assert sim['1']['2'] == pytest.approx(0.0)


class TestSimilaritySymmetry:
    """Test that all similarity metrics are symmetric."""
    
    def test_dice_symmetric(self):
        """Dice(A, B) should equal Dice(B, A)."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        G.add_node('X')
        G.add_node('Y')
        G.add_node('Z')
        G.add_edge('A', 'X')
        G.add_edge('A', 'Y')
        G.add_edge('B', 'Y')
        G.add_edge('B', 'Z')
        
        sim = G.get_dice_similarity()
        assert sim['A']['B'] == pytest.approx(sim['B']['A'])
    
    def test_jaccard_symmetric(self):
        """Jaccard(A, B) should equal Jaccard(B, A)."""
        G = ec.Graph()
        G.add_cycle_graph(6)
        
        sim = G.get_jaccard_similarity()
        for i in range(1, 7):
            for j in range(i + 1, 6):
                assert sim[str(i)][str(j)] == pytest.approx(sim[str(j)][str(i)])
    
    def test_cosine_symmetric(self):
        """Cosine(A, B) should equal Cosine(B, A)."""
        G = ec.Graph()
        G.add_full_graph(5)
        
        sim = G.get_cosine_similarity()
        nodes = G.node_ids(limit=None)
        for n1_id in nodes:
            for n2_id in nodes:
                n1 = G.get_label_from_id(n1_id)
                n2 = G.get_label_from_id(n2_id)
                assert sim[n1][n2] == pytest.approx(sim[n2][n1])
    
    def test_overlap_symmetric(self):
        """Overlap(A, B) should equal Overlap(B, A)."""
        G = ec.Graph()
        G.add_path_graph(7)
        
        sim = G.get_overlap_similarity()
        for i in range(1, 8):
            for j in range(i + 1, 7):
                assert sim[str(i)][str(j)] == pytest.approx(sim[str(j)][str(i)])


class TestSimilarityRange:
    """Test that all similarity metrics return values in [0, 1]."""
    
    def test_dice_range(self):
        """Dice coefficient should be in [0, 1]."""
        G = ec.Graph()
        G.add_pa_graph(20)
        
        sim = G.get_dice_similarity()
        for n1, scores in sim.items():
            for n2, score in scores.items():
                assert 0.0 <= score <= 1.0
    
    def test_jaccard_range(self):
        """Jaccard index should be in [0, 1]."""
        G = ec.Graph()
        G.add_gnp_graph(15, 0.3)
        
        sim = G.get_jaccard_similarity()
        for n1, scores in sim.items():
            for n2, score in scores.items():
                assert 0.0 <= score <= 1.0
    
    def test_cosine_range(self):
        """Cosine similarity should be in [0, 1]."""
        G = ec.Graph()
        G.add_smallworld_graph(10, 4, 0.3)
        
        sim = G.get_cosine_similarity()
        for n1, scores in sim.items():
            for n2, score in scores.items():
                assert 0.0 <= score <= 1.0
    
    def test_overlap_range(self):
        """Overlap coefficient should be in [0, 1]."""
        G = ec.Graph()
        G.add_balanced_tree(3, 4)  # r=3 (branching factor), h=4 (height)
        
        sim = G.get_overlap_similarity()
        for n1, scores in sim.items():
            for n2, score in scores.items():
                assert 0.0 <= score <= 1.0


class TestSimilarityEmptyGraph:
    """Test similarity metrics on edge cases."""
    
    def test_single_node(self):
        """Single node should have self-similarity = 1.0."""
        G = ec.Graph()
        G.add_node('A')
        
        dice = G.get_dice_similarity()
        jaccard = G.get_jaccard_similarity()
        cosine = G.get_cosine_similarity()
        overlap = G.get_overlap_similarity()
        
        assert dice['A']['A'] == pytest.approx(1.0)
        assert jaccard['A']['A'] == pytest.approx(1.0)
        assert cosine['A']['A'] == pytest.approx(1.0)
        assert overlap['A']['A'] == pytest.approx(1.0)
    
    def test_disconnected_nodes(self):
        """Disconnected nodes with no neighbors."""
        G = ec.Graph()
        G.add_node('A')
        G.add_node('B')
        
        dice = G.get_dice_similarity()
        jaccard = G.get_jaccard_similarity()
        cosine = G.get_cosine_similarity()
        overlap = G.get_overlap_similarity()
        
        # Nodes with no neighbors should have 0 similarity
        assert dice['A']['B'] == pytest.approx(0.0)
        assert jaccard['A']['B'] == pytest.approx(0.0)
        assert cosine['A']['B'] == pytest.approx(0.0)
        assert overlap['A']['B'] == pytest.approx(0.0)


class TestSimilarityReturnStructure:
    """Test that return structure is correct."""
    
    def test_nested_dict_structure(self):
        """All methods should return nested dict[str, dict[str, float]]."""
        G = ec.Graph()
        G.add_full_graph(4)
        
        for method in [G.get_dice_similarity, G.get_jaccard_similarity,
                      G.get_cosine_similarity, G.get_overlap_similarity]:
            sim = method()
            
            assert isinstance(sim, dict)
            for node1, scores in sim.items():
                assert isinstance(node1, str)
                assert isinstance(scores, dict)
                for node2, score in scores.items():
                    assert isinstance(node2, str)
                    assert isinstance(score, float)
    
    def test_all_pairs_present(self):
        """Result should include all node pairs."""
        G = ec.Graph()
        G.add_path_graph(5)
        # Convert node IDs to labels for comparison
        node_labels = {G.get_label_from_id(node_id) for node_id in G.node_ids(limit=None)}
        
        for method in [G.get_dice_similarity, G.get_jaccard_similarity,
                      G.get_cosine_similarity, G.get_overlap_similarity]:
            sim = method()
            
            # Check all node labels are keys
            assert set(sim.keys()) == node_labels
            
            # Check each node has scores for all nodes
            for node1, scores in sim.items():
                assert set(scores.keys()) == node_labels
