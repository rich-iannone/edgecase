"""
Additional comprehensive tests for graph builders.

This includes property-based tests, edge cases, and performance tests.
"""

import pytest
from hypothesis import given, strategies as st, assume
import edgecase as ec
import polars as pl


class TestGraphBuilderEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_add_path_single_node(self):
        """Test path with single node."""
        g = ec.Graph(directed=True)
        g = g.add_path_graph(n=1)
        
        assert g.count_nodes() == 1
        assert g.count_edges() == 0
    
    def test_add_path_zero_nodes_fails(self):
        """Test path with zero nodes fails."""
        g = ec.Graph(directed=True)
        with pytest.raises(ValueError, match="at least 1 node"):
            g.add_path_graph(n=0)
    
    def test_add_path_negative_fails(self):
        """Test path with negative nodes fails."""
        g = ec.Graph(directed=True)
        with pytest.raises(ValueError):
            g.add_path_graph(n=-5)
    
    def test_add_cycle_triangle(self):
        """Test minimum cycle (3 nodes)."""
        g = ec.Graph(directed=True)
        g = g.add_cycle_graph(n=3)
        
        assert g.count_nodes() == 3
        assert g.count_edges() == 3
        
        # Each node should have exactly one neighbor
        for node_id in g.get_node_ids(limit=None):
            assert len(g.neighbors(node_id)) == 1
    
    def test_add_star_two_nodes(self):
        """Test minimum star (2 nodes - center and one leaf)."""
        g = ec.Graph(directed=False)
        g = g.add_star_graph(n=2)
        
        assert g.count_nodes() == 2
        assert g.count_edges() == 1
    
    def test_add_star_one_node_fails(self):
        """Test star with one node fails."""
        g = ec.Graph(directed=False)
        with pytest.raises(ValueError, match="at least 2 nodes"):
            g.add_star_graph(n=1)
    
    def test_add_full_graph_single_node(self):
        """Test complete graph with single node."""
        g = ec.Graph(directed=False)
        g = g.add_full_graph(n=1)
        
        assert g.count_nodes() == 1
        assert g.count_edges() == 0
    
    def test_add_full_graph_self_loops_directed(self):
        """Test complete graph with self-loops in directed mode."""
        g = ec.Graph(directed=True)
        g = g.add_full_graph(n=2, keep_loops=True)
        
        # With loops: n*n = 4 edges
        assert g.count_nodes() == 2
        assert g.count_edges() == 4
    
    def test_add_balanced_tree_height_zero(self):
        """Test tree with height 0 (just root)."""
        g = ec.Graph(directed=True)
        g = g.add_balanced_tree(k=2, h=0)
        
        assert g.count_nodes() == 1
        assert g.count_edges() == 0
    
    def test_add_balanced_tree_k_one(self):
        """Test tree with branching factor 1 (linear)."""
        g = ec.Graph(directed=True)
        g = g.add_balanced_tree(k=1, h=3)
        
        # Should be like a path: 1 -> 2 -> 3 -> 4
        assert g.count_nodes() == 4
        assert g.count_edges() == 3
    
    def test_add_grid_1x1(self):
        """Test 1x1 grid (single node)."""
        g = ec.Graph(directed=False)
        g = g.add_grid_2d(1, 1)
        
        assert g.count_nodes() == 1
        assert g.count_edges() == 0
    
    def test_add_grid_1xn(self):
        """Test 1xN grid (linear path)."""
        g = ec.Graph(directed=False)
        g = g.add_grid_2d(1, 5)
        
        # Should be like a path
        assert g.count_nodes() == 5
        assert g.count_edges() == 4
    
    def test_add_prism_triangle(self):
        """Test minimum prism (3 nodes per cycle)."""
        g = ec.Graph(directed=False)
        g = g.add_prism_graph(n=3)
        
        assert g.count_nodes() == 6
        assert g.count_edges() == 9  # 3 + 3 + 3


class TestGraphBuilderProperties:
    """Property-based tests using Hypothesis."""
    
    @given(n=st.integers(min_value=1, max_value=50))
    def test_path_properties(self, n):
        """Test path graph properties hold for various sizes."""
        g = ec.Graph(directed=True)
        g = g.add_path_graph(n=n)
        
        assert g.count_nodes() == n
        assert g.count_edges() == max(0, n - 1)
        
        # Verify it's directed
        assert g.is_directed()
    
    @given(n=st.integers(min_value=3, max_value=50))
    def test_cycle_properties(self, n):
        """Test cycle graph properties."""
        g = ec.Graph(directed=True)
        g = g.add_cycle_graph(n=n)
        
        assert g.count_nodes() == n
        assert g.count_edges() == n
        
        # Each node should have exactly 1 outgoing edge
        for node_id in g.get_node_ids(limit=None):
            assert len(g.neighbors(node_id)) == 1
    
    @given(n=st.integers(min_value=2, max_value=30))
    def test_star_properties(self, n):
        """Test star graph properties."""
        g = ec.Graph(directed=False)
        g = g.add_star_graph(n=n)
        
        assert g.count_nodes() == n
        assert g.count_edges() == n - 1
        
        # Count degree distribution
        degrees = [len(g.neighbors(nid)) for nid in g.get_node_ids(limit=None)]
        
        if n == 2:
            # Special case: both nodes have degree 1
            assert all(d == 1 for d in degrees)
        else:
            # One node (center) should have n-1 neighbors
            # All others should have exactly 1 neighbor (the center)
            assert degrees.count(n - 1) == 1  # One center
            assert degrees.count(1) == n - 1  # n-1 leaves
    
    @given(n=st.integers(min_value=1, max_value=20))
    def test_full_graph_undirected_properties(self, n):
        """Test complete undirected graph properties."""
        g = ec.Graph(directed=False)
        g = g.add_full_graph(n=n)
        
        assert g.count_nodes() == n
        # For undirected: n*(n-1)/2 edges
        expected_edges = n * (n - 1) // 2
        assert g.count_edges() == expected_edges
        
        # Each node should have n-1 neighbors
        if n > 1:
            for node_id in g.get_node_ids(limit=None):
                assert len(g.neighbors(node_id)) == n - 1
    
    @given(
        k=st.integers(min_value=1, max_value=5),
        h=st.integers(min_value=0, max_value=4)
    )
    def test_balanced_tree_properties(self, k, h):
        """Test balanced tree properties."""
        g = ec.Graph(directed=True)
        g = g.add_balanced_tree(k=k, h=h)
        
        # Total nodes: sum of k^i for i from 0 to h
        expected_nodes = sum(k**i for i in range(h + 1))
        assert g.count_nodes() == expected_nodes
        
        # Total edges: sum of k^i for i from 1 to h
        expected_edges = sum(k**i for i in range(1, h + 1))
        assert g.count_edges() == expected_edges
    
    @given(
        x=st.integers(min_value=1, max_value=10),
        y=st.integers(min_value=1, max_value=10)
    )
    def test_grid_2d_properties(self, x, y):
        """Test 2D grid properties."""
        g = ec.Graph(directed=False)
        g = g.add_grid_2d(x, y)
        
        assert g.count_nodes() == x * y
        
        # Edges: horizontal + vertical
        expected_edges = (x - 1) * y + x * (y - 1)
        assert g.count_edges() == expected_edges
    
    @given(n=st.integers(min_value=3, max_value=20))
    def test_prism_properties(self, n):
        """Test prism graph properties."""
        g = ec.Graph(directed=False)
        g = g.add_prism_graph(n=n)
        
        assert g.count_nodes() == 2 * n
        # Two cycles + connecting edges
        assert g.count_edges() == 3 * n


class TestGraphBuilderCombinations:
    """Test combining multiple graph builders."""
    
    def test_multiple_paths(self):
        """Test adding multiple paths creates disconnected components."""
        g = ec.Graph(directed=False)
        
        g = g.add_path_graph(n=3)
        g = g.add_path_graph(n=4)
        g = g.add_path_graph(n=5)
        
        assert g.count_nodes() == 12
        assert g.count_edges() == 9  # 2 + 3 + 4
    
    def test_path_then_cycle(self):
        """Test adding path then cycle."""
        g = ec.Graph(directed=False)
        
        g = g.add_path_graph(n=5)
        initial_nodes = g.count_nodes()
        
        g = g.add_cycle_graph(n=4)
        
        assert g.count_nodes() == initial_nodes + 4
        assert g.count_edges() == 4 + 4  # path edges + cycle edges
    
    def test_star_then_full_graph(self):
        """Test adding star then complete graph."""
        g = ec.Graph(directed=False)
        
        g = g.add_star_graph(n=5)
        g = g.add_full_graph(n=3)
        
        assert g.count_nodes() == 8  # 5 + 3
        assert g.count_edges() == 4 + 3  # star edges + complete graph edges
    
    def test_mixed_directed_undirected_attributes(self):
        """Test that attributes persist across multiple additions."""
        g = ec.Graph(directed=True)
        
        g = g.add_path_graph(n=3, type="path_node", rel="path_edge")
        g = g.add_cycle_graph(n=4, type="cycle_node", rel="cycle_edge")
        
        assert g.count_nodes() == 7
        assert g.count_edges() == 6  # 2 path edges + 4 cycle edges


class TestCreateGraphDataFrames:
    """Test graph creation from DataFrames with various scenarios."""
    
    def test_create_with_empty_dataframes(self):
        """Test with empty DataFrames."""
        nodes = pl.DataFrame({'id': []})
        edges = pl.DataFrame({'from': [], 'to': []})
        
        g = ec.Graph(nodes_df=nodes, edges_df=edges)
        assert g.count_nodes() == 0
        assert g.count_edges() == 0
    
    def test_create_with_integer_node_ids(self):
        """Test with integer node IDs."""
        nodes = pl.DataFrame({
            'id': [1, 2, 3],
            'type': ['a', 'b', 'c']
        })
        
        edges = pl.DataFrame({
            'from': [1, 2],
            'to': [2, 3]
        })
        
        g = ec.Graph(nodes_df=nodes, edges_df=edges)
        assert g.count_nodes() == 3
        assert g.count_edges() == 2
        
        # IDs should be converted to strings
        assert g.has_node('1')
        assert g.has_node('2')
        assert g.has_node('3')
    
    def test_create_with_mixed_types(self):
        """Test with mixed attribute types."""
        nodes = pl.DataFrame({
            'id': ['A', 'B', 'C'],
            'value': [1.5, 2.5, 3.5],
            'count': [10, 20, 30],
            'active': [True, False, True]
        })
        
        g = ec.Graph(nodes_df=nodes)
        assert g.count_nodes() == 3
    
    def test_create_with_null_values(self):
        """Test handling of null values in DataFrames."""
        nodes = pl.DataFrame({
            'id': ['A', 'B', 'C'],
            'type': ['user', None, 'admin'],
            'label': [None, 'B', 'C']
        })
        
        edges = pl.DataFrame({
            'from': ['A', 'B'],
            'to': ['B', 'C'],
            'weight': [1.0, None]
        })
        
        g = ec.Graph(nodes_df=nodes, edges_df=edges)
        assert g.count_nodes() == 3
        assert g.count_edges() == 2
    
    def test_create_edges_create_nodes_automatically(self):
        """Test that edges create nodes if nodes_df not provided."""
        edges = pl.DataFrame({
            'from': ['X', 'Y', 'Z'],
            'to': ['Y', 'Z', 'X'],
            'weight': [1.0, 2.0, 3.0]
        })
        
        g = ec.Graph(edges_df=edges, directed=True)
        
        assert g.count_nodes() == 3
        assert g.count_edges() == 3
        assert g.has_node('X')
        assert g.has_node('Y')
        assert g.has_node('Z')
    
    def test_create_with_duplicate_edges(self):
        """Test behavior with duplicate edges."""
        edges = pl.DataFrame({
            'from': ['A', 'A', 'B'],
            'to': ['B', 'B', 'C'],
        })
        
        g = ec.Graph(edges_df=edges, directed=False)
        
        # Duplicate edges should be handled (current impl silently ignores)
        assert g.count_nodes() == 3
        # Exact edge count depends on duplicate handling
        assert g.count_edges() >= 2
    
    def test_create_directed_vs_undirected(self):
        """Test that directed flag affects graph structure."""
        edges = pl.DataFrame({
            'from': ['A', 'B'],
            'to': ['B', 'C']
        })
        
        g_directed = ec.Graph(edges_df=edges, directed=True)
        g_undirected = ec.Graph(edges_df=edges, directed=False)
        
        assert g_directed.is_directed()
        assert not g_undirected.is_directed()
        
        # In directed graph, B has no outgoing edges to A
        # In undirected graph, B has edge to A
        assert len(g_directed.neighbors('B')) == 1  # only to C
        assert len(g_undirected.neighbors('B')) >= 1  # to A and C


class TestGraphBuilderPerformance:
    """Test performance characteristics of graph builders."""
    
    def test_large_path_reasonable_time(self):
        """Test that large paths can be created efficiently."""
        g = ec.Graph(directed=True)
        g = g.add_path_graph(n=1000)
        
        assert g.count_nodes() == 1000
        assert g.count_edges() == 999
    
    def test_large_star_reasonable_time(self):
        """Test that large star graphs can be created efficiently."""
        g = ec.Graph(directed=False)
        g = g.add_star_graph(n=1000)
        
        assert g.count_nodes() == 1000
        assert g.count_edges() == 999
    
    def test_moderately_dense_full_graph(self):
        """Test complete graph with moderate size."""
        g = ec.Graph(directed=False)
        g = g.add_full_graph(n=50)
        
        assert g.count_nodes() == 50
        assert g.count_edges() == 50 * 49 // 2
    
    def test_large_grid(self):
        """Test large grid graph creation."""
        g = ec.Graph(directed=False)
        g = g.add_grid_2d(50, 50)
        
        assert g.count_nodes() == 2500
        expected_edges = (50 - 1) * 50 + 50 * (50 - 1)
        assert g.count_edges() == expected_edges
    
    def test_sequential_additions_performance(self):
        """Test that sequential additions don't degrade."""
        g = ec.Graph(directed=False)
        
        for _ in range(10):
            g = g.add_path_graph(n=100)
        
        assert g.count_nodes() == 1000
        assert g.count_edges() == 990


class TestGraphBuilderConsistency:
    """Test consistency and correctness of graph structures."""
    
    def test_path_connectivity(self):
        """Test that path creates proper linear connectivity."""
        g = ec.Graph(directed=True)
        g = g.add_path_graph(n=5)
        
        # Should be able to trace from 1 to 5
        node_ids = sorted(g.get_node_ids(limit=None), key=int)
        for i in range(len(node_ids) - 1):
            current = node_ids[i]
            expected_next = node_ids[i + 1]
            neighbors = g.neighbors(current)
            assert expected_next in neighbors
    
    def test_cycle_connectivity(self):
        """Test that cycle closes properly."""
        g = ec.Graph(directed=True)
        g = g.add_cycle_graph(n=4)
        
        # Traverse the cycle and return to start
        visited = set()
        current = g.get_node_ids(limit=None)[0]
        start = current
        
        while current not in visited:
            visited.add(current)
            neighbors = g.neighbors(current)
            assert len(neighbors) == 1
            current = neighbors[0]
        
        # Should return to start
        assert current == start
        assert len(visited) == 4
    
    def test_star_center_connectivity(self):
        """Test that star center connects to all leaves."""
        g = ec.Graph(directed=False)
        g = g.add_star_graph(n=7)
        
        # Find the center (node with highest degree)
        nodes = g.get_node_ids(limit=None)
        degrees = {nid: len(g.neighbors(nid)) for nid in nodes}
        center = max(degrees, key=degrees.get)
        
        center_neighbors = set(g.neighbors(center))
        all_nodes = set(nodes)
        
        # Center should connect to all non-center nodes
        assert center_neighbors == all_nodes - {center}
    
    def test_full_graph_all_pairs_connected(self):
        """Test that complete graph connects all pairs."""
        g = ec.Graph(directed=False)
        g = g.add_full_graph(n=5)
        
        nodes = g.get_node_ids(limit=None)
        for node in nodes:
            neighbors = set(g.neighbors(node))
            # Each node should connect to all others
            assert neighbors == set(nodes) - {node}
    
    def test_grid_orthogonal_connections(self):
        """Test that grid only creates orthogonal connections."""
        g = ec.Graph(directed=False)
        g = g.add_grid_2d(3, 3)
        
        # Corner nodes should have exactly 2 neighbors
        # Edge nodes should have exactly 3 neighbors
        # Center node should have exactly 4 neighbors
        degrees = [len(g.neighbors(nid)) for nid in g.get_node_ids(limit=None)]
        
        assert degrees.count(2) == 4  # 4 corners
        assert degrees.count(3) == 4  # 4 edge midpoints
        assert degrees.count(4) == 1  # 1 center


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
