"""
Tests for graph building functions.
"""

import pytest
import edgecase as ec


class TestGraphBuilders:
    """Test graph construction functions."""
    
    def test_add_path_graph(self):
        """Test creating a path graph."""
        g = ec.Graph(directed=True).add_path_graph(n=5)
        
        assert g.num_nodes() == 5
        assert g.num_edges() == 4
        
        # Verify linear structure
        neighbors_1 = g.neighbors("1")
        assert len(neighbors_1) == 1
        # Neighbor could be returned as int or string
        assert "2" in neighbors_1 or 2 in neighbors_1
    
    def test_add_path_graph_with_attributes(self):
        """Test path graph with node types and edge relationships."""
        g = ec.Graph(directed=True).add_path_graph(n=3, type="step", rel="next")
        
        assert g.num_nodes() == 3
        assert g.num_edges() == 2
    
    def test_add_cycle_graph(self):
        """Test creating a cycle graph."""
        g = ec.Graph(directed=True).add_cycle_graph(n=4)
        
        assert g.num_nodes() == 4
        assert g.num_edges() == 4  # n edges for a cycle
        
        # Verify cycle structure - each node has exactly 1 outgoing edge
        for node_id in g.node_ids():
            assert len(g.neighbors(node_id)) == 1
    
    def test_add_cycle_graph_minimum(self):
        """Test cycle graph must have at least 3 nodes."""
        g = ec.Graph(directed=True)
        with pytest.raises(ValueError, match="at least 3 nodes"):
            g.add_cycle_graph(n=2)
    
    def test_add_star_graph(self):
        """Test creating a star graph."""
        g = ec.Graph(directed=False).add_star_graph(n=6)
        
        assert g.num_nodes() == 6
        assert g.num_edges() == 5  # n-1 edges
        
        # Center node (first added) should connect to all others
        center_id = "1"
        neighbors = g.neighbors(center_id)
        assert len(neighbors) == 5
    
    def test_add_full_graph(self):
        """Test creating a complete graph."""
        g = ec.Graph(directed=False).add_full_graph(n=4)
        
        assert g.num_nodes() == 4
        # For undirected complete graph: n*(n-1)/2 edges
        assert g.num_edges() == 6
    
    def test_add_full_graph_directed(self):
        """Test complete graph in directed mode."""
        g = ec.Graph(directed=True).add_full_graph(n=3)
        
        assert g.num_nodes() == 3
        # For directed: n*(n-1) edges (no self-loops by default)
        assert g.num_edges() == 6
    
    def test_add_full_graph_with_loops(self):
        """Test complete graph including self-loops."""
        g = ec.Graph(directed=True).add_full_graph(n=3, keep_loops=True)
        
        assert g.num_nodes() == 3
        # With self-loops: n*n edges
        assert g.num_edges() == 9
    
    def test_add_balanced_tree(self):
        """Test creating a balanced tree."""
        g = ec.Graph(directed=True).add_balanced_tree(k=2, h=2)
        
        # Binary tree of height 2: 1 + 2 + 4 = 7 nodes
        assert g.num_nodes() == 7
        # Edges: 2 + 4 = 6 edges
        assert g.num_edges() == 6
    
    def test_add_balanced_tree_ternary(self):
        """Test ternary tree."""
        g = ec.Graph(directed=True).add_balanced_tree(k=3, h=1)
        
        # Height 1 ternary tree: 1 + 3 = 4 nodes
        assert g.num_nodes() == 4
        assert g.num_edges() == 3
    
    def test_add_prism_graph(self):
        """Test creating a prism graph."""
        g = ec.Graph(directed=False).add_prism_graph(n=4)
        
        assert g.num_nodes() == 8  # 2 cycles of 4
        # 4 edges per cycle + 4 connecting edges
        assert g.num_edges() == 12
    
    def test_add_prism_graph_minimum(self):
        """Test prism must have at least 3 nodes per cycle."""
        g = ec.Graph(directed=False)
        with pytest.raises(ValueError, match="at least 3 nodes"):
            g.add_prism_graph(n=2)
    
    def test_add_grid_2d(self):
        """Test creating a 2D grid graph."""
        g = ec.Graph(directed=False).add_grid_2d(3, 3)
        
        assert g.num_nodes() == 9
        # For 3x3 grid: (3-1)*3 + 3*(3-1) = 6 + 6 = 12 edges
        assert g.num_edges() == 12
    
    def test_add_grid_2d_rectangular(self):
        """Test rectangular grid."""
        g = ec.Graph(directed=False).add_grid_2d(4, 2)
        
        assert g.num_nodes() == 8
        # (4-1)*2 + 4*(2-1) = 6 + 4 = 10 edges
        assert g.num_edges() == 10
    
    def test_sequential_additions(self):
        """Test adding multiple structures to the same graph."""
        # Using method chaining to add multiple structures
        g = (
            ec.Graph(directed=False)
            .add_path_graph(n=3)
            .add_cycle_graph(n=4)
            .add_star_graph(n=3)
        )
        
        assert g.num_nodes() == 10  # 3 + 4 + 3
        assert g.num_edges() == 8   # 2 + 4 + 2


class TestCreateGraph:
    """Test graph creation from DataFrames."""
    
    def test_create_empty_graph(self):
        """Test creating an empty graph."""
        g = ec.Graph.from_dataframes(directed=False)
        assert g.num_nodes() == 0
        assert g.num_edges() == 0
    
    def test_create_from_nodes_only(self):
        """Test creating graph from nodes DataFrame only."""
        import polars as pl
        
        nodes = pl.DataFrame({
            'id': ['A', 'B', 'C'],
            'type': ['user', 'user', 'admin'],
            'label': ['Alice', 'Bob', 'Charlie']
        })
        
        g = ec.Graph.from_dataframes(nodes_df=nodes)
        assert g.num_nodes() == 3
        assert g.num_edges() == 0
    
    def test_create_from_nodes_and_edges(self):
        """Test creating graph from both DataFrames."""
        import polars as pl
        
        nodes = pl.DataFrame({
            'id': ['A', 'B', 'C'],
            'type': ['user', 'user', 'user']
        })
        
        edges = pl.DataFrame({
            'from': ['A', 'B'],
            'to': ['B', 'C'],
            'rel': ['follows', 'follows']
        })
        
        g = ec.Graph.from_dataframes(nodes_df=nodes, edges_df=edges, directed=True)
        assert g.num_nodes() == 3
        assert g.num_edges() == 2
    
    def test_create_edges_without_nodes(self):
        """Test that edges automatically create missing nodes."""
        import polars as pl
        
        edges = pl.DataFrame({
            'from': ['A', 'B'],
            'to': ['B', 'C']
        })
        
        g = ec.Graph.from_dataframes(edges_df=edges)
        assert g.num_nodes() == 3  # A, B, C created automatically
        assert g.num_edges() == 2
    
    def test_create_graph_invalid_nodes_df(self):
        """Test error when nodes DataFrame missing 'id' column."""
        import polars as pl
        
        nodes = pl.DataFrame({
            'name': ['A', 'B'],  # Wrong column name
            'type': ['user', 'user']
        })
        
        with pytest.raises(ValueError, match="must have an 'id' column"):
            ec.Graph.from_dataframes(nodes_df=nodes)
    
    def test_create_graph_invalid_edges_df(self):
        """Test error when edges DataFrame missing required columns."""
        import polars as pl
        
        edges = pl.DataFrame({
            'source': ['A'],  # Wrong column names
            'target': ['B']
        })
        
        with pytest.raises(ValueError, match="must have 'from' and 'to' columns"):
            ec.Graph.from_dataframes(edges_df=edges)


class TestGraphBuildersIntegration:
    """Integration tests combining builders with graph operations."""
    
    def test_path_and_analyze(self):
        """Test creating path and analyzing it."""
        g = ec.Graph(directed=True).add_path_graph(n=10)
        
        # Path should have exactly 2 leaf nodes (endpoints)
        leaf_count = sum(1 for nid in g.node_ids() if len(g.neighbors(nid)) == 0)
        assert leaf_count == 1  # Only last node has no outgoing edges
    
    def test_star_centrality(self):
        """Test that center of star has highest degree."""
        g = ec.Graph(directed=False).add_star_graph(n=10)
        
        # Center node should have degree n-1
        center_degree = len(g.neighbors("1"))
        assert center_degree == 9
        
        # All other nodes should have degree 1
        for i in range(2, 11):
            assert len(g.neighbors(str(i))) == 1
    
    def test_grid_neighbors(self):
        """Test that grid nodes have correct neighbor counts."""
        g = ec.Graph(directed=False).add_grid_2d(3, 3)
        
        # Count nodes by degree
        degree_2 = 0  # Corners
        degree_3 = 0  # Edges
        degree_4 = 0  # Interior
        
        for nid in g.node_ids():
            degree = len(g.neighbors(nid))
            if degree == 2:
                degree_2 += 1
            elif degree == 3:
                degree_3 += 1
            elif degree == 4:
                degree_4 += 1
        
        assert degree_2 == 4  # 4 corners
        assert degree_3 == 4  # 4 edge midpoints
        assert degree_4 == 1  # 1 center node


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
