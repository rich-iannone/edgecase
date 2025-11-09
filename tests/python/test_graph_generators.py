"""
Tests for graph generator methods.

These tests verify that all graph generator methods work correctly,
produce expected graph structures, and support method chaining.
"""

import pytest
import edgecase as ec


class TestDeterministicGenerators:
    """Test deterministic graph generators."""
    
    def test_add_path_graph(self):
        """Generate a path graph."""
        G = ec.Graph()
        G.add_path_graph(5)
        
        assert G.num_nodes() == 5
        assert G.num_edges() == 4
        
    def test_add_cycle_graph(self):
        """Generate a cycle graph."""
        G = ec.Graph()
        G.add_cycle_graph(4)
        
        assert G.num_nodes() == 4
        assert G.num_edges() == 4
        
    def test_add_star_graph(self):
        """Generate a star graph."""
        G = ec.Graph()
        G.add_star_graph(5)
        
        assert G.num_nodes() == 5
        assert G.num_edges() == 4
        
    def test_add_full_graph(self):
        """Generate a complete graph."""
        G = ec.Graph()
        G.add_full_graph(4)
        
        assert G.num_nodes() == 4
        assert G.num_edges() == 6  # n*(n-1)/2 for undirected
        
    def test_add_full_graph_directed(self):
        """Generate a complete directed graph."""
        G = ec.Graph(directed=True)
        G.add_full_graph(4)
        
        assert G.num_nodes() == 4
        assert G.num_edges() == 12  # n*(n-1) for directed
        
    def test_add_balanced_tree(self):
        """Generate a balanced tree."""
        G = ec.Graph(directed=True)
        G.add_balanced_tree(k=2, h=3)
        
        assert G.num_nodes() == 15  # 1 + 2 + 4 + 8
        assert G.num_edges() == 14  # n - 1 for tree
        
    def test_add_grid_2d(self):
        """Generate a 2D grid."""
        G = ec.Graph()
        G.add_grid_2d(3, 4)
        
        assert G.num_nodes() == 12
        # Grid edges: (rows-1)*cols + rows*(cols-1)
        assert G.num_edges() == 2 * 4 + 3 * 3
        
    def test_add_prism_graph(self):
        """Generate a prism graph."""
        G = ec.Graph()
        G.add_prism_graph(5)
        
        assert G.num_nodes() == 10  # Two cycles of 5
        assert G.num_edges() == 15  # 5 + 5 + 5


class TestRandomGenerators:
    """Test random graph generators."""
    
    def test_add_gnp_graph_basic(self):
        """Generate Erdős-Rényi G(n,p) graph."""
        G = ec.Graph()
        G.add_gnp_graph(50, 0.1, seed=42)
        
        assert G.num_nodes() == 50
        # With p=0.1, expect roughly 0.1 * 50*49/2 ≈ 122 edges
        assert 80 < G.num_edges() < 160
        
    def test_add_gnp_graph_reproducibility(self):
        """G(n,p) with same seed produces same graph."""
        G1 = ec.Graph()
        G1.add_gnp_graph(30, 0.2, seed=123)
        
        G2 = ec.Graph()
        G2.add_gnp_graph(30, 0.2, seed=123)
        
        assert G1.num_edges() == G2.num_edges()
        
    def test_add_gnp_graph_probability_bounds(self):
        """G(n,p) respects probability bounds."""
        # p=0 should create no edges
        G_empty = ec.Graph()
        G_empty.add_gnp_graph(20, 0.0, seed=42)
        assert G_empty.num_edges() == 0
        
        # p=1 should create complete graph
        G_complete = ec.Graph()
        G_complete.add_gnp_graph(10, 1.0, seed=42)
        assert G_complete.num_edges() == 45  # 10*9/2
        
    def test_add_gnm_graph_basic(self):
        """Generate Erdős-Rényi G(n,m) graph."""
        G = ec.Graph()
        G.add_gnm_graph(50, 100, seed=42)
        
        assert G.num_nodes() == 50
        assert G.num_edges() == 100
        
    def test_add_gnm_graph_directed(self):
        """G(n,m) works with directed graphs."""
        G = ec.Graph(directed=True)
        G.add_gnm_graph(20, 50, directed=True, seed=42)
        
        assert G.num_nodes() == 20
        assert G.num_edges() == 50
        
    def test_add_gnm_graph_max_edges(self):
        """G(n,m) rejects m > max possible edges."""
        G = ec.Graph()
        
        with pytest.raises(ValueError, match="Cannot create"):
            G.add_gnm_graph(10, 100, seed=42)  # max is 45
            
    def test_add_pa_graph_basic(self):
        """Generate Barabási-Albert graph."""
        G = ec.Graph()
        G.add_pa_graph(100, m=3, seed=42)
        
        assert G.num_nodes() == 100
        # Initial clique of m+1 nodes has m+1 choose 2 edges
        # Each additional node adds m edges
        # Total: (m+1)*m/2 + (n - m - 1) * m
        expected_edges = 4 * 3 // 2 + (100 - 4) * 3
        assert G.num_edges() == expected_edges
        
    def test_add_pa_graph_parameters(self):
        """PA graph validates parameters."""
        G = ec.Graph()
        
        with pytest.raises(ValueError, match="m must be at least 1"):
            G.add_pa_graph(50, m=0)
            
        with pytest.raises(ValueError, match="m must be less than n"):
            G.add_pa_graph(10, m=10)
            
    def test_add_smallworld_graph_basic(self):
        """Generate Watts-Strogatz small-world graph."""
        G = ec.Graph()
        G.add_smallworld_graph(100, k=6, p=0.1, seed=42)
        
        assert G.num_nodes() == 100
        # Should have roughly n*k/2 edges (some may be rewired)
        assert 280 < G.num_edges() <= 300
        
    def test_add_smallworld_graph_no_rewiring(self):
        """Small-world with p=0 creates ring lattice."""
        G = ec.Graph()
        G.add_smallworld_graph(20, k=4, p=0.0, seed=42)
        
        assert G.num_nodes() == 20
        assert G.num_edges() == 40  # 20 * 4 / 2
        
    def test_add_smallworld_graph_parameters(self):
        """Small-world validates parameters."""
        G = ec.Graph()
        
        with pytest.raises(ValueError, match="k must be even"):
            G.add_smallworld_graph(100, k=5, p=0.1)
            
        with pytest.raises(ValueError, match="k must be less than n"):
            G.add_smallworld_graph(10, k=20, p=0.1)


class TestMethodChaining:
    """Test that generators support method chaining."""
    
    def test_chain_multiple_generators(self):
        """Chain multiple graph generators."""
        G = ec.Graph() \
            .add_path_graph(5) \
            .add_cycle_graph(4) \
            .add_star_graph(3)
        
        assert G.num_nodes() == 12
        
    def test_chain_with_selection(self):
        """Chain generators with selection operations."""
        G = ec.Graph() \
            .add_full_graph(4) \
            .select_nodes_by_id([1, 2, 3, 4]) \
            .set_node_attrs_ws(group='A')
        
        assert G.num_nodes() == 4
        # get_node() takes a label, so use '1' string label
        assert G.get_node('1').attributes['group'] == 'A'
        
    def test_chain_random_and_deterministic(self):
        """Chain random and deterministic generators."""
        G = ec.Graph() \
            .add_path_graph(10) \
            .add_gnp_graph(20, 0.1, seed=42)
        
        assert G.num_nodes() == 30


class TestGraphProperties:
    """Test properties of generated graphs."""
    
    def test_path_graph_connectivity(self):
        """Path graph has correct connectivity."""
        G = ec.Graph().add_path_graph(5)
        
        # First and last nodes have degree 1
        # Middle nodes have degree 2
        degrees = [G.degree('1'), G.degree('2'), G.degree('3'), 
                  G.degree('4'), G.degree('5')]
        assert degrees == [1, 2, 2, 2, 1]
        
    def test_star_graph_hub(self):
        """Star graph has one hub node."""
        G = ec.Graph().add_star_graph(6)
        
        # First node should be the hub with degree 5
        assert G.degree('1') == 5
        
        # Other nodes should have degree 1
        for i in range(2, 7):
            assert G.degree(str(i)) == 1
            
    def test_cycle_graph_uniform_degree(self):
        """Cycle graph has uniform degree 2."""
        G = ec.Graph().add_cycle_graph(5)
        
        for i in range(1, 6):
            assert G.degree(str(i)) == 2
            
    def test_full_graph_uniform_degree(self):
        """Complete graph has uniform degree n-1."""
        G = ec.Graph().add_full_graph(5)
        
        for i in range(1, 6):
            assert G.degree(str(i)) == 4


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_minimum_sizes(self):
        """Test minimum valid sizes."""
        G1 = ec.Graph().add_path_graph(1)
        assert G1.num_nodes() == 1 and G1.num_edges() == 0
        
        G2 = ec.Graph().add_cycle_graph(3)
        assert G2.num_nodes() == 3 and G2.num_edges() == 3
        
        G3 = ec.Graph().add_star_graph(2)
        assert G3.num_nodes() == 2 and G3.num_edges() == 1
        
    def test_empty_gnp_graph(self):
        """G(n,p) with p=0 creates isolated nodes."""
        G = ec.Graph().add_gnp_graph(10, 0.0)
        
        assert G.num_nodes() == 10
        assert G.num_edges() == 0
        
    def test_gnm_zero_edges(self):
        """G(n,m) with m=0 creates isolated nodes."""
        G = ec.Graph().add_gnm_graph(10, 0)
        
        assert G.num_nodes() == 10
        assert G.num_edges() == 0


class TestPerformance:
    """Test performance with larger graphs."""
    
    def test_large_path(self):
        """Generate large path graph."""
        G = ec.Graph().add_path_graph(1000)
        
        assert G.num_nodes() == 1000
        assert G.num_edges() == 999
        
    def test_large_gnp(self):
        """Generate large random graph."""
        G = ec.Graph().add_gnp_graph(500, 0.01, seed=42)
        
        assert G.num_nodes() == 500
        # Should complete quickly
        
    def test_large_pa(self):
        """Generate large scale-free graph."""
        G = ec.Graph().add_pa_graph(1000, m=3, seed=42)
        
        assert G.num_nodes() == 1000
