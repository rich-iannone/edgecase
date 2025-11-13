"""
Tests for Edge Betweenness (Girvan-Newman) community detection.

This module tests the get_cmty_edge_btwns() method which implements
the Girvan-Newman algorithm for community detection by iteratively
removing high betweenness edges.
"""

import pytest
import edgecase as ec


class TestEdgeBetweennessBasic:
    """Basic functionality tests."""
    
    def test_empty_graph(self):
        """Empty graph should return empty dict."""
        G = ec.Graph(directed=False)
        communities = G.get_cmty_edge_btwns()
        assert communities == {}
    
    def test_single_node(self):
        """Single node should be in community 0."""
        G = ec.Graph(directed=False)
        G.add_node('A')
        communities = G.get_cmty_edge_btwns()
        assert communities == {'A': 0}
    
    def test_two_nodes_no_edge(self):
        """Two unconnected nodes should be in separate communities."""
        G = ec.Graph(directed=False)
        G.add_node('A')
        G.add_node('B')
        communities = G.get_cmty_edge_btwns()
        assert len(set(communities.values())) == 2
    
    def test_two_nodes_with_edge(self):
        """Two connected nodes should be in same community."""
        G = ec.Graph(directed=False)
        G.add_edge('A', 'B')
        communities = G.get_cmty_edge_btwns()
        assert communities['A'] == communities['B']
    
    def test_directed_graph_error(self):
        """Should raise error for directed graphs."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        with pytest.raises(ValueError, match="undirected"):
            G.get_cmty_edge_btwns()


class TestEdgeBetweennessBridge:
    """Test graphs with clear bridge edges."""
    
    def test_two_cliques_one_bridge(self):
        """Two triangles connected by single edge should split into 2 communities."""
        G = ec.Graph(directed=False)
        
        # Triangle 1
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        
        # Bridge
        G.add_edge('C', 'D')
        
        # Triangle 2
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        communities = G.get_cmty_edge_btwns()
        
        # Should find 2 communities
        assert len(set(communities.values())) == 2
        
        # Triangle 1 should be together
        assert communities['A'] == communities['B'] == communities['C']
        
        # Triangle 2 should be together
        assert communities['D'] == communities['E'] == communities['F']
        
        # Triangles should be separate
        assert communities['A'] != communities['D']
    
    def test_barbell_graph(self):
        """Barbell graph should split into two communities."""
        G = ec.Graph(directed=False)
        
        # Left clique
        for i in ['A', 'B', 'C']:
            for j in ['A', 'B', 'C']:
                if i < j:
                    G.add_edge(i, j)
        
        # Bridge
        G.add_edge('C', 'D')
        
        # Right clique
        for i in ['D', 'E', 'F']:
            for j in ['D', 'E', 'F']:
                if i < j:
                    G.add_edge(i, j)
        
        communities = G.get_cmty_edge_btwns()
        
        # Should find 2 communities
        assert len(set(communities.values())) == 2
        
        # Left clique together
        assert communities['A'] == communities['B'] == communities['C']
        
        # Right clique together
        assert communities['D'] == communities['E'] == communities['F']
        
        # Different communities
        assert communities['A'] != communities['D']
    
    def test_chain_of_triangles(self):
        """Chain of 3 triangles should find 3 communities."""
        G = ec.Graph(directed=False)
        
        # Triangle 1
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        
        # Bridge to triangle 2
        G.add_edge('C', 'D')
        
        # Triangle 2
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        # Bridge to triangle 3
        G.add_edge('F', 'G')
        
        # Triangle 3
        G.add_edge('G', 'H')
        G.add_edge('H', 'I')
        G.add_edge('I', 'G')
        
        communities = G.get_cmty_edge_btwns()
        
        # Should find 3 communities
        assert len(set(communities.values())) == 3
        
        # Each triangle internal consistency
        assert communities['A'] == communities['B'] == communities['C']
        assert communities['D'] == communities['E'] == communities['F']
        assert communities['G'] == communities['H'] == communities['I']
        
        # Triangles are different
        assert communities['A'] != communities['D']
        assert communities['D'] != communities['G']
        assert communities['A'] != communities['G']


class TestEdgeBetweennessKnownStructures:
    """Test on graphs with known community structure."""
    
    def test_karate_club_simplified(self):
        """Simplified karate club should split into multiple communities."""
        G = ec.Graph(directed=False)
        
        # Faction 1 (instructor's group)
        for i in range(1, 18):
            for j in range(i+1, 18):
                if (j - i) <= 3:  # Dense connections within group
                    G.add_edge(str(i), str(j))
        
        # Faction 2 (officer's group)
        for i in range(18, 35):
            for j in range(i+1, 35):
                if (j - i) <= 3:  # Dense connections within group
                    G.add_edge(str(i), str(j))
        
        # Sparse connections between factions
        G.add_edge('17', '18')
        G.add_edge('16', '19')
        
        communities = G.get_cmty_edge_btwns()
        
        # Should find at least 2 communities
        num_communities = len(set(communities.values()))
        assert num_communities >= 2
        
        # Algorithm may find sub-communities within factions
        # This is valid behavior for hierarchical methods


class TestEdgeBetweennessModularity:
    """Test modularity-related properties."""
    
    def compute_modularity(self, G, communities):
        """Compute modularity Q for a given partition."""
        if G.num_nodes() == 0 or G.num_edges() == 0:
            return 0.0
        
        m = G.num_edges()
        Q = 0.0
        
        for node_i in G.node_ids(limit=None):
            for node_j in G.node_ids(limit=None):
                if communities[node_i] == communities[node_j]:
                    # A_ij
                    A_ij = 1.0 if G.has_edge(node_i, node_j) else 0.0
                    
                    # (k_i * k_j) / (2m)
                    k_i = G.degree(node_i)
                    k_j = G.degree(node_j)
                    Q += A_ij - (k_i * k_j) / (2.0 * m)
        
        return Q / (2.0 * m)
    
    def test_modularity_positive_structured(self):
        """Modularity should be positive for structured graphs."""
        G = ec.Graph(directed=False)
        
        # Two well-separated cliques
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        G.add_edge('F', 'D')
        
        # Single bridge
        G.add_edge('C', 'D')
        
        communities = G.get_cmty_edge_btwns()
        Q = self.compute_modularity(G, communities)
        
        # Should have positive modularity
        assert Q > 0.0
    
    def test_modularity_complete_graph(self):
        """Complete graph should have near-zero modularity."""
        G = ec.Graph(directed=False)
        G.add_full_graph(6)
        
        communities = G.get_cmty_edge_btwns()
        Q = self.compute_modularity(G, communities)
        
        # Complete graph has no clear structure, Q should be small
        assert abs(Q) < 0.1


class TestEdgeBetweennessProperties:
    """Test algorithm properties and invariants."""
    
    def test_all_nodes_assigned(self):
        """Every node should be assigned to a community."""
        G = ec.Graph(directed=False)
        for i in range(10):
            for j in range(i+1, 10):
                if abs(i - j) <= 2:
                    G.add_edge(str(i), str(j))
        
        communities = G.get_cmty_edge_btwns()
        node_ids = G.node_ids(limit=None)
        
        assert len(communities) == len(node_ids)
        for node_id in node_ids:
            assert node_id in communities
    
    def test_community_ids_contiguous(self):
        """Community IDs should be 0, 1, 2, ... with no gaps."""
        G = ec.Graph(directed=False)
        
        # Create 3 disconnected triangles
        for offset in [0, 3, 6]:
            for i in range(3):
                for j in range(i+1, 3):
                    G.add_edge(str(offset + i), str(offset + j))
        
        communities = G.get_cmty_edge_btwns()
        comm_ids = sorted(set(communities.values()))
        
        # Should be [0, 1, 2, ...]
        assert comm_ids == list(range(len(comm_ids)))
    
    def test_deterministic(self):
        """Same graph should produce same communities."""
        G1 = ec.Graph(directed=False)
        G2 = ec.Graph(directed=False)
        
        for G in [G1, G2]:
            G.add_edge('A', 'B')
            G.add_edge('B', 'C')
            G.add_edge('C', 'A')
            G.add_edge('C', 'D')
            G.add_edge('D', 'E')
            G.add_edge('E', 'F')
            G.add_edge('F', 'D')
        
        comm1 = G1.get_cmty_edge_btwns()
        comm2 = G2.get_cmty_edge_btwns()
        
        assert comm1 == comm2


class TestEdgeBetweennessPerformance:
    """Performance tests for larger graphs."""
    
    def test_medium_graph(self):
        """Test with 50 nodes."""
        G = ec.Graph(directed=False)
        
        # Create 5 communities of 10 nodes each
        for comm_idx in range(5):
            offset = comm_idx * 10
            for i in range(10):
                for j in range(i+1, 10):
                    if abs(i - j) <= 3:
                        G.add_edge(str(offset + i), str(offset + j))
        
        # Add sparse inter-community edges
        G.add_edge('9', '10')
        G.add_edge('19', '20')
        G.add_edge('29', '30')
        G.add_edge('39', '40')
        
        communities = G.get_cmty_edge_btwns()
        
        # Should find multiple communities
        num_communities = len(set(communities.values()))
        assert 3 <= num_communities <= 7  # Reasonable range
    
    def test_sparse_graph(self):
        """Test with sparse connectivity."""
        G = ec.Graph(directed=False)
        
        # Chain of 30 nodes
        for i in range(29):
            G.add_edge(str(i), str(i+1))
        
        communities = G.get_cmty_edge_btwns()
        
        # Should split chain into communities
        num_communities = len(set(communities.values()))
        assert num_communities >= 2


class TestEdgeBetweennessEdgeCases:
    """Edge case tests."""
    
    def test_star_graph(self):
        """Star graph (hub and spokes)."""
        G = ec.Graph(directed=False)
        
        # Central hub connected to 10 nodes
        for i in range(10):
            G.add_edge('hub', str(i))
        
        communities = G.get_cmty_edge_btwns()
        
        # All nodes assigned
        assert len(communities) == 11
    
    def test_cycle_graph(self):
        """Cycle graph (ring)."""
        G = ec.Graph(directed=False)
        
        # Create cycle
        for i in range(10):
            G.add_edge(str(i), str((i+1) % 10))
        
        communities = G.get_cmty_edge_btwns()
        
        # All nodes assigned
        assert len(communities) == 10
    
    def test_disconnected_components(self):
        """Multiple disconnected components."""
        G = ec.Graph(directed=False)
        
        # Component 1
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        
        # Component 2
        G.add_edge('D', 'E')
        G.add_edge('E', 'F')
        
        # Component 3
        G.add_edge('G', 'H')
        
        communities = G.get_cmty_edge_btwns()
        
        # Should find at least 3 communities
        assert len(set(communities.values())) >= 3
        
        # Each component should be separate
        assert communities['A'] != communities['D']
        assert communities['D'] != communities['G']
