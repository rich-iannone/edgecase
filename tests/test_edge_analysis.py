"""Tests for edge analysis functions."""

import pytest
import edgecase as ec


class TestLoopEdges:
    """Test self-loop counting."""
    
    def test_count_loop_edges_empty(self):
        """Empty graph has no loops."""
        G = ec.Graph()
        assert G.count_loop_edges() == 0
    
    def test_count_loop_edges_no_loops(self):
        """Graph with no self-loops."""
        G = ec.Graph()
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'D')
        assert G.count_loop_edges() == 0
    
    def test_count_loop_edges_single(self):
        """Graph with one self-loop."""
        G = ec.Graph()
        G.add_edge('A', 'B')
        G.add_edge('A', 'A')  # Self-loop
        G.add_edge('B', 'C')
        assert G.count_loop_edges() == 1
    
    def test_count_loop_edges_multiple(self):
        """Graph with multiple self-loops."""
        G = ec.Graph()
        G.add_edge('A', 'A')  # Loop 1
        G.add_edge('A', 'B')
        G.add_edge('B', 'B')  # Loop 2
        G.add_edge('C', 'C')  # Loop 3
        assert G.count_loop_edges() == 3
    
    def test_count_loop_edges_only_loops(self):
        """Graph with only self-loops."""
        G = ec.Graph()
        G.add_edge('A', 'A')
        G.add_edge('B', 'B')
        G.add_edge('C', 'C')
        assert G.count_loop_edges() == 3
    
    def test_count_selfloops_alias(self):
        """count_selfloops() is an alias for count_loop_edges()."""
        G = ec.Graph()
        G.add_edge('A', 'A')
        G.add_edge('B', 'B')
        assert G.count_selfloops() == G.count_loop_edges() == 2


class TestMultiEdges:
    """Test multi-edge counting."""
    
    def test_count_multi_edges_empty(self):
        """Empty graph has no multi-edges."""
        G = ec.Graph()
        assert G.count_multi_edges() == 0
    
    def test_count_multi_edges_simple(self):
        """Simple graph (no parallel edges)."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B', id=1)
        G.add_edge('B', 'C', id=2)
        G.add_edge('C', 'D', id=3)
        assert G.count_multi_edges() == 0
    
    def test_count_multi_edges_single_duplicate(self):
        """Two edges between same pair."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B', id=1)
        G.add_edge('A', 'B', id=2)  # Duplicate
        G.add_edge('B', 'C', id=3)
        assert G.count_multi_edges() == 1  # 1 duplicate
    
    def test_count_multi_edges_triple(self):
        """Three edges between same pair."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B', id=1)
        G.add_edge('A', 'B', id=2)  # Duplicate 1
        G.add_edge('A', 'B', id=3)  # Duplicate 2
        assert G.count_multi_edges() == 2  # 2 duplicates
    
    def test_count_multi_edges_multiple_pairs(self):
        """Multiple pairs with duplicates."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B', id=1)
        G.add_edge('A', 'B', id=2)  # Duplicate
        G.add_edge('B', 'C', id=3)
        G.add_edge('B', 'C', id=4)  # Duplicate
        G.add_edge('B', 'C', id=5)  # Another duplicate
        G.add_edge('C', 'D', id=6)
        assert G.count_multi_edges() == 3  # 1 + 2 duplicates
    
    def test_count_multi_edges_directed_different_directions(self):
        """Edges in different directions are not multi-edges."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B', id=1)
        G.add_edge('B', 'A', id=2)  # Different direction
        assert G.count_multi_edges() == 0
    
    def test_count_multi_edges_undirected(self):
        """Multi-edges in undirected graph."""
        G = ec.Graph(directed=False)
        G.add_edge('A', 'B', id=1)
        G.add_edge('A', 'B', id=2)  # Duplicate
        assert G.count_multi_edges() == 1


class TestMutualNodePairs:
    """Test bidirectional edge counting."""
    
    def test_count_mutual_empty(self):
        """Empty graph has no mutual pairs."""
        G = ec.Graph(directed=True)
        assert G.count_mutual_node_pairs() == 0
    
    def test_count_mutual_undirected_error(self):
        """Raises error for undirected graphs."""
        G = ec.Graph(directed=False)
        with pytest.raises(ValueError, match="only defined for directed"):
            G.count_mutual_node_pairs()
    
    def test_count_mutual_no_mutual(self):
        """Graph with no bidirectional edges."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'D')
        assert G.count_mutual_node_pairs() == 0
    
    def test_count_mutual_single_pair(self):
        """One mutual pair."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        G.add_edge('B', 'A')  # Mutual with A
        G.add_edge('B', 'C')  # One-way only
        assert G.count_mutual_node_pairs() == 1
    
    def test_count_mutual_multiple_pairs(self):
        """Multiple mutual pairs."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        G.add_edge('B', 'A')  # Mutual pair 1
        G.add_edge('C', 'D')
        G.add_edge('D', 'C')  # Mutual pair 2
        G.add_edge('E', 'F')  # One-way
        assert G.count_mutual_node_pairs() == 2
    
    def test_count_mutual_excludes_self_loops(self):
        """Self-loops are not counted as mutual pairs."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'A')  # Self-loop
        G.add_edge('A', 'B')
        G.add_edge('B', 'A')  # Mutual
        assert G.count_mutual_node_pairs() == 1
    
    def test_count_mutual_complex_network(self):
        """Complex network with mixed edge types."""
        G = ec.Graph(directed=True)
        # Mutual pairs
        G.add_edge('A', 'B')
        G.add_edge('B', 'A')
        G.add_edge('B', 'C')
        G.add_edge('C', 'B')
        # One-way edges
        G.add_edge('C', 'D')
        G.add_edge('D', 'E')
        # Self-loop
        G.add_edge('E', 'E')
        assert G.count_mutual_node_pairs() == 2
    
    def test_count_mutual_fully_connected(self):
        """Fully connected directed graph (complete digraph)."""
        G = ec.Graph(directed=True)
        nodes = ['A', 'B', 'C']
        for i, src in enumerate(nodes):
            for j, tgt in enumerate(nodes):
                if i != j:  # No self-loops
                    G.add_edge(src, tgt)
        # 3 nodes = 3 pairs: (A,B), (B,C), (A,C)
        assert G.count_mutual_node_pairs() == 3


class TestAsymmetricNodePairs:
    """Test one-way edge counting."""
    
    def test_count_asymmetric_empty(self):
        """Empty graph has no asymmetric pairs."""
        G = ec.Graph(directed=True)
        assert G.count_asymmetric_node_pairs() == 0
    
    def test_count_asymmetric_undirected_error(self):
        """Raises error for undirected graphs."""
        G = ec.Graph(directed=False)
        with pytest.raises(ValueError, match="only defined for directed"):
            G.count_asymmetric_node_pairs()
    
    def test_count_asymmetric_all_one_way(self):
        """All edges are one-way."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'D')
        assert G.count_asymmetric_node_pairs() == 3
    
    def test_count_asymmetric_no_asymmetric(self):
        """All edges are mutual (no asymmetric pairs)."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        G.add_edge('B', 'A')
        G.add_edge('C', 'D')
        G.add_edge('D', 'C')
        assert G.count_asymmetric_node_pairs() == 0
    
    def test_count_asymmetric_mixed(self):
        """Mix of mutual and asymmetric pairs."""
        G = ec.Graph(directed=True)
        # Mutual pair
        G.add_edge('A', 'B')
        G.add_edge('B', 'A')
        # Asymmetric pairs
        G.add_edge('A', 'C')
        G.add_edge('D', 'E')
        assert G.count_asymmetric_node_pairs() == 2
    
    def test_count_asymmetric_excludes_self_loops(self):
        """Self-loops are not counted."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'A')  # Self-loop
        G.add_edge('A', 'B')  # Asymmetric
        assert G.count_asymmetric_node_pairs() == 1
    
    def test_count_asymmetric_complex_network(self):
        """Complex network with various edge patterns."""
        G = ec.Graph(directed=True)
        # Mutual pairs
        G.add_edge('A', 'B')
        G.add_edge('B', 'A')
        # Asymmetric pairs
        G.add_edge('B', 'C')  # One-way
        G.add_edge('C', 'D')  # One-way
        G.add_edge('E', 'F')  # One-way
        # Self-loop
        G.add_edge('F', 'F')
        assert G.count_asymmetric_node_pairs() == 3
    
    def test_count_asymmetric_partial_mutual(self):
        """Some mutual, some one-way from same nodes."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        G.add_edge('B', 'A')  # Mutual
        G.add_edge('A', 'C')  # Asymmetric from A
        G.add_edge('B', 'D')  # Asymmetric from B
        assert G.count_asymmetric_node_pairs() == 2


class TestEdgeAnalysisIntegration:
    """Integration tests combining multiple edge analysis functions."""
    
    def test_complete_edge_analysis(self):
        """Comprehensive edge analysis on complex graph."""
        G = ec.Graph(directed=True)
        
        # Add various edge types
        G.add_edge('A', 'B', id=1)
        G.add_edge('B', 'A', id=2)  # Mutual pair
        G.add_edge('A', 'B', id=3)  # Multi-edge
        G.add_edge('C', 'D', id=4)  # Asymmetric
        G.add_edge('E', 'E', id=5)  # Self-loop
        G.add_edge('F', 'G', id=6)  # Asymmetric
        
        assert G.count_loop_edges() == 1
        assert G.count_multi_edges() == 1
        assert G.count_mutual_node_pairs() == 1
        assert G.count_asymmetric_node_pairs() == 2
    
    def test_social_network_analysis(self):
        """Analyze a social network graph."""
        G = ec.Graph(directed=True)
        
        # Mutual friendships
        G.add_edge('Alice', 'Bob')
        G.add_edge('Bob', 'Alice')
        G.add_edge('Bob', 'Carol')
        G.add_edge('Carol', 'Bob')
        
        # One-way follows
        G.add_edge('Alice', 'Dave')
        G.add_edge('Carol', 'Dave')
        G.add_edge('Dave', 'Eve')
        
        # No self-loops in social network
        assert G.count_loop_edges() == 0
        # No multi-edges
        assert G.count_multi_edges() == 0
        # 2 mutual friendships
        assert G.count_mutual_node_pairs() == 2
        # 3 one-way follows
        assert G.count_asymmetric_node_pairs() == 3
    
    def test_citation_network(self):
        """Analyze citation network (directed, no mutual citations typically)."""
        G = ec.Graph(directed=True)
        
        # Paper citations (typically one-way, chronological)
        G.add_edge('Paper2023', 'Paper2020')
        G.add_edge('Paper2023', 'Paper2019')
        G.add_edge('Paper2022', 'Paper2020')
        G.add_edge('Paper2022', 'Paper2019')
        G.add_edge('Paper2021', 'Paper2019')
        
        # No self-citations
        assert G.count_loop_edges() == 0
        # Multiple citations to same paper (not multi-edges, different sources)
        assert G.count_multi_edges() == 0
        # No mutual citations (no time travel!)
        assert G.count_mutual_node_pairs() == 0
        # All citations are one-way
        assert G.count_asymmetric_node_pairs() == 5
    
    def test_multigraph_scenario(self):
        """Test multi-edge scenario with multiple relationships."""
        G = ec.Graph(directed=True)
        
        # Multiple types of edges between same nodes
        G.add_edge('A', 'B', id=1, attributes={'type': 'friend'})
        G.add_edge('A', 'B', id=2, attributes={'type': 'colleague'})
        G.add_edge('A', 'B', id=3, attributes={'type': 'neighbor'})
        G.add_edge('B', 'A', id=4, attributes={'type': 'friend'})
        
        assert G.count_multi_edges() == 2  # 2 extra edges A->B
        assert G.count_mutual_node_pairs() == 1  # A and B are mutual
    
    def test_comparison_with_simple_graph(self):
        """Compare simple vs complex graph properties."""
        G_simple = ec.Graph(directed=True)
        G_simple.add_edge('A', 'B')
        G_simple.add_edge('B', 'C')
        G_simple.add_edge('C', 'D')
        
        G_complex = ec.Graph(directed=True)
        G_complex.add_edge('A', 'B')
        G_complex.add_edge('B', 'A')
        G_complex.add_edge('A', 'B', id=2)  # Multi
        G_complex.add_edge('C', 'C')  # Loop
        
        # Simple graph
        assert G_simple.count_loop_edges() == 0
        assert G_simple.count_multi_edges() == 0
        assert G_simple.count_mutual_node_pairs() == 0
        assert G_simple.count_asymmetric_node_pairs() == 3
        
        # Complex graph
        assert G_complex.count_loop_edges() == 1
        assert G_complex.count_multi_edges() == 1
        assert G_complex.count_mutual_node_pairs() == 1
        assert G_complex.count_asymmetric_node_pairs() == 0


class TestEdgeAnalysisEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_node_no_edges(self):
        """Single node with no edges."""
        G = ec.Graph(directed=True)
        G.add_node('A')
        assert G.count_loop_edges() == 0
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 0
        assert G.count_asymmetric_node_pairs() == 0
    
    def test_single_self_loop_only(self):
        """Single node with only self-loop."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'A')
        assert G.count_loop_edges() == 1
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 0
        assert G.count_asymmetric_node_pairs() == 0
    
    def test_two_nodes_both_directions(self):
        """Two nodes, fully connected both ways."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        G.add_edge('B', 'A')
        assert G.count_loop_edges() == 0
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 1
        assert G.count_asymmetric_node_pairs() == 0
    
    def test_star_graph_directed(self):
        """Star topology (hub with spokes)."""
        G = ec.Graph(directed=True)
        hub = 'Hub'
        for i in range(5):
            G.add_edge(hub, f'Node{i}')
        
        assert G.count_loop_edges() == 0
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 0
        assert G.count_asymmetric_node_pairs() == 5
    
    def test_cycle_directed(self):
        """Directed cycle."""
        G = ec.Graph(directed=True)
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'D')
        G.add_edge('D', 'A')  # Complete cycle
        
        assert G.count_loop_edges() == 0
        assert G.count_multi_edges() == 0
        assert G.count_mutual_node_pairs() == 0
        assert G.count_asymmetric_node_pairs() == 4
