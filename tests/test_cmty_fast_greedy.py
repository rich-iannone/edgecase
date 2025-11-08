"""
Tests for Fast Greedy community detection algorithm.

Tests cover:
- Known community structures (Karate club, disconnected components)
- Edge cases (empty, single node, complete graphs)
- Modularity validation
- Algorithm correctness
"""

import pytest
import edgecase as ec


class TestFastGreedyBasic:
    """Basic functionality tests for Fast Greedy algorithm."""
    
    def test_empty_graph(self):
        """Test with empty graph."""
        G = ec.Graph(directed=False)
        communities = G.get_cmty_fast_greedy()
        assert communities == {}
    
    def test_single_node(self):
        """Test with single isolated node."""
        G = ec.Graph(directed=False)
        G.add_node('A')
        communities = G.get_cmty_fast_greedy()
        assert communities == {'A': 0}
        assert len(set(communities.values())) == 1
    
    def test_two_nodes_no_edge(self):
        """Test with two disconnected nodes."""
        G = ec.Graph(directed=False)
        G.add_node('A')
        G.add_node('B')
        communities = G.get_cmty_fast_greedy()
        
        # Should be in separate communities (no edges to merge)
        assert len(set(communities.values())) == 2
        assert communities['A'] != communities['B']
    
    def test_two_nodes_with_edge(self):
        """Test with two connected nodes."""
        G = ec.Graph(directed=False)
        G.add_edge('A', 'B')
        communities = G.get_cmty_fast_greedy()
        
        # Should merge into single community
        assert len(set(communities.values())) == 1
        assert communities['A'] == communities['B']
    
    def test_directed_graph_error(self):
        """Test that directed graphs raise error."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        
        with pytest.raises(ValueError, match="undirected"):
            G.get_cmty_fast_greedy()


class TestFastGreedyDisconnected:
    """Tests with disconnected components."""
    
    def test_two_triangles(self):
        """Test with two disconnected triangles."""
        G = ec.Graph(directed=False)
        
        # Triangle 1
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        
        # Triangle 2
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        communities = G.get_cmty_fast_greedy()
        
        # Should find 2 communities
        assert len(set(communities.values())) == 2
        
        # All nodes in triangle 1 should be in same community
        assert communities['A'] == communities['B'] == communities['C']
        
        # All nodes in triangle 2 should be in same community
        assert communities['D'] == communities['E'] == communities['F']
        
        # Triangles should be in different communities
        assert communities['A'] != communities['D']
    
    def test_three_components(self):
        """Test with three separate components."""
        G = ec.Graph(directed=False)
        
        # Component 1: Triangle
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        
        # Component 2: Line
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        
        # Component 3: Single edge
        G.add_edge('G', 'H')
        
        communities = G.get_cmty_fast_greedy()
        
        # Should find 3 communities
        assert len(set(communities.values())) == 3
        
        # Each component in separate community
        c1 = communities['A']
        c2 = communities['D']
        c3 = communities['G']
        
        assert c1 != c2 and c2 != c3 and c1 != c3


class TestFastGreedyComplete:
    """Tests with complete graphs (cliques)."""
    
    def test_complete_graph_small(self):
        """Test with complete graph (K5)."""
        G = ec.Graph(directed=False)
        G.add_full_graph(5)
        
        communities = G.get_cmty_fast_greedy()
        
        # Complete graph should be single community
        assert len(set(communities.values())) == 1
    
    def test_complete_graph_large(self):
        """Test with larger complete graph (K20)."""
        G = ec.Graph(directed=False)
        G.add_full_graph(20)
        
        communities = G.get_cmty_fast_greedy()
        
        # Should remain as single community
        assert len(set(communities.values())) == 1


class TestFastGreedyKnownStructures:
    """Tests with known community structures."""
    
    def test_barbell_graph(self):
        """Test with barbell graph (2 cliques connected by bridge)."""
        G = ec.Graph(directed=False)
        
        # Clique 1
        for i in range(4):
            for j in range(i + 1, 4):
                G.add_edge(str(i), str(j))
        
        # Bridge
        G.add_edge('3', '4')
        
        # Clique 2
        for i in range(4, 8):
            for j in range(i + 1, 8):
                G.add_edge(str(i), str(j))
        
        communities = G.get_cmty_fast_greedy()
        
        # Should find 2 communities (the two cliques)
        assert len(set(communities.values())) == 2
        
        # Nodes 0-3 should be in one community
        comm1 = communities['0']
        assert communities['1'] == comm1
        assert communities['2'] == comm1
        assert communities['3'] == comm1
        
        # Nodes 4-7 should be in another community
        comm2 = communities['4']
        assert communities['5'] == comm2
        assert communities['6'] == comm2
        assert communities['7'] == comm2
        
        # Communities should be different
        assert comm1 != comm2
    
    def test_simple_karate_club(self):
        """Test with simplified Karate club structure."""
        G = ec.Graph(directed=False)
        
        # Faction 1 (densely connected)
        faction1 = ['A', 'B', 'C', 'D', 'E']
        for i in range(len(faction1)):
            for j in range(i + 1, len(faction1)):
                G.add_edge(faction1[i], faction1[j])
        
        # Faction 2 (densely connected)
        faction2 = ['F', 'G', 'H', 'I', 'J']
        for i in range(len(faction2)):
            for j in range(i + 1, len(faction2)):
                G.add_edge(faction2[i], faction2[j])
        
        # Bridge between factions (weak connection)
        G.add_edge('E', 'F')
        
        communities = G.get_cmty_fast_greedy()
        
        # Should find 2 communities
        assert len(set(communities.values())) == 2
        
        # Check faction 1 nodes are together
        comm_f1 = communities['A']
        for node in faction1:
            assert communities[node] == comm_f1
        
        # Check faction 2 nodes are together
        comm_f2 = communities['F']
        for node in faction2:
            assert communities[node] == comm_f2
        
        # Factions should be different
        assert comm_f1 != comm_f2


class TestFastGreedyModularity:
    """Tests for modularity validation."""
    
    def compute_modularity(self, G, communities):
        """Helper: compute modularity score."""
        m = G.num_edges()
        if m == 0:
            return 0.0
        
        Q = 0.0
        for node_i in communities:
            for node_j in communities:
                if communities[node_i] == communities[node_j]:
                    # Same community
                    A_ij = 1.0 if G.has_edge(node_i, node_j) else 0.0
                    k_i = G.degree(node_i)
                    k_j = G.degree(node_j)
                    Q += A_ij - (k_i * k_j) / (2.0 * m)
        
        return Q / (2.0 * m)
    
    def test_modularity_positive(self):
        """Test that modularity is positive for structured graphs."""
        G = ec.Graph(directed=False)
        
        # Two well-separated cliques
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        communities = G.get_cmty_fast_greedy()
        Q = self.compute_modularity(G, communities)
        
        # Should have positive modularity
        assert Q > 0.0
    
    def test_modularity_complete_graph(self):
        """Test modularity for complete graph."""
        G = ec.Graph(directed=False)
        G.add_full_graph(5)
        
        communities = G.get_cmty_fast_greedy()
        Q = self.compute_modularity(G, communities)
        
        # Complete graph has zero modularity (one community)
        assert abs(Q) < 0.01  # Near zero


class TestFastGreedyProperties:
    """Tests for algorithm properties."""
    
    def test_all_nodes_assigned(self):
        """Test that all nodes get community assignments."""
        G = ec.Graph(directed=False)
        G.add_cycle_graph(10)
        
        communities = G.get_cmty_fast_greedy()
        
        # All nodes should be present
        assert len(communities) == G.num_nodes()
        assert set(communities.keys()) == set(G.nodes())
    
    def test_community_ids_contiguous(self):
        """Test that community IDs are 0-indexed and contiguous."""
        G = ec.Graph(directed=False)
        
        # Three separate triangles
        for base in [0, 10, 20]:
            for i in range(3):
                for j in range(i + 1, 3):
                    G.add_edge(str(base + i), str(base + j))
        
        communities = G.get_cmty_fast_greedy()
        community_ids = set(communities.values())
        
        # Should be 0, 1, 2 (or just fewer communities)
        assert community_ids == set(range(len(community_ids)))
    
    def test_deterministic(self):
        """Test that algorithm is deterministic."""
        G = ec.Graph(directed=False)
        G.add_star_graph(10)
        
        # Run twice
        comm1 = G.get_cmty_fast_greedy()
        comm2 = G.get_cmty_fast_greedy()
        
        # Should get same result
        assert comm1 == comm2


class TestFastGreedyPerformance:
    """Performance and scalability tests."""
    
    def test_medium_graph(self):
        """Test with medium-sized graph (100 nodes)."""
        G = ec.Graph(directed=False)
        G.add_pa_graph(n=100, m=3, seed=42)
        
        communities = G.get_cmty_fast_greedy()
        
        # Should complete successfully
        assert len(communities) == 100
        assert len(set(communities.values())) >= 1
    
    def test_sparse_graph(self):
        """Test with sparse graph."""
        G = ec.Graph(directed=False)
        G.add_gnp_graph(n=50, p=0.05, seed=42)
        
        communities = G.get_cmty_fast_greedy()
        
        # Should handle sparse graphs
        assert len(communities) == 50
    
    def test_dense_graph(self):
        """Test with dense graph."""
        G = ec.Graph(directed=False)
        G.add_gnp_graph(n=30, p=0.8, seed=42)
        
        communities = G.get_cmty_fast_greedy()
        
        # Should handle dense graphs
        assert len(communities) == 30


class TestFastGreedyEdgeCases:
    """Edge case tests."""
    
    def test_star_graph(self):
        """Test with star graph."""
        G = ec.Graph(directed=False)
        G.add_star_graph(10)
        
        communities = G.get_cmty_fast_greedy()
        
        # Star graph should be single community
        assert len(set(communities.values())) == 1
    
    def test_path_graph(self):
        """Test with path graph."""
        G = ec.Graph(directed=False)
        G.add_path_graph(10)
        
        communities = G.get_cmty_fast_greedy()
        
        # Path should likely be single community (no clear split)
        # But algorithm might split, so just check it runs
        assert len(communities) == 10
    
    def test_cycle_graph(self):
        """Test with cycle graph."""
        G = ec.Graph(directed=False)
        G.add_cycle_graph(10)
        
        communities = G.get_cmty_fast_greedy()
        
        # Cycle has no natural split point
        assert len(communities) == 10
