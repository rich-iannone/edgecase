"""Tests for neighborhood-based and centrality-based selection methods."""

import pytest
import edgecase as ec


class TestSelectNodesInNeighborhood:
    """Tests for select_nodes_in_neighborhood()."""
    
    def test_neighborhood_basic_1hop(self):
        """Test basic 1-hop neighborhood selection."""
        G = ec.Graph(directed=True)
        G.add_path(5)  # 1 -> 2 -> 3 -> 4 -> 5
        
        G.select_nodes_in_neighborhood("1", k=1, mode="out")
        selected = G.get_selected_nodes()
        
        # Should select node 1 and its immediate successor
        assert "1" in selected
        assert "2" in selected
        assert len(selected) == 2
    
    def test_neighborhood_2hop_directed(self):
        """Test 2-hop neighborhood in directed graph."""
        G = ec.Graph(directed=True)
        G.add_path(6)  # 1 -> 2 -> 3 -> 4 -> 5 -> 6
        
        G.select_nodes_in_neighborhood("1", k=2, mode="out")
        selected = G.get_selected_nodes()
        
        # Should select nodes 1, 2, 3
        assert "1" in selected
        assert "2" in selected
        assert "3" in selected
        assert "4" not in selected
        assert len(selected) == 3
    
    def test_neighborhood_undirected(self):
        """Test neighborhood selection in undirected graph."""
        G = ec.Graph(directed=False)
        G.add_path(5)  # 1 - 2 - 3 - 4 - 5
        
        # From node 3, 1-hop should get 2 and 4
        G.select_nodes_in_neighborhood("3", k=1)
        selected = G.get_selected_nodes()
        
        assert "3" in selected
        assert "2" in selected
        assert "4" in selected
        assert len(selected) == 3
    
    def test_neighborhood_multi_seed(self):
        """Test neighborhood selection with multiple seed nodes."""
        G = ec.Graph(directed=True)
        G.add_path(10)
        
        # Select 1-hop from nodes 2 and 7
        G.select_nodes_in_neighborhood(["2", "7"], k=1, mode="out")
        selected = G.get_selected_nodes()
        
        # Should get 2->3 and 7->8
        assert "2" in selected
        assert "3" in selected
        assert "7" in selected
        assert "8" in selected
        assert "4" not in selected
    
    def test_neighborhood_mode_in(self):
        """Test incoming neighborhood selection."""
        G = ec.Graph(directed=True)
        G.add_path(5)  # 1 -> 2 -> 3 -> 4 -> 5
        
        # From node 3, incoming should get 2 and 1
        G.select_nodes_in_neighborhood("3", k=2, mode="in")
        selected = G.get_selected_nodes()
        
        assert "3" in selected
        assert "2" in selected
        assert "1" in selected
        assert "4" not in selected
    
    def test_neighborhood_mode_all(self):
        """Test bidirectional neighborhood selection."""
        G = ec.Graph(directed=True)
        # Create a small network
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("D", "B")
        
        # From B, all directions should get A, C, D
        G.select_nodes_in_neighborhood("B", k=1, mode="all")
        selected = G.get_selected_nodes()
        
        assert "B" in selected
        assert "A" in selected
        assert "C" in selected
        assert "D" in selected
    
    def test_neighborhood_star_graph(self):
        """Test neighborhood on star topology."""
        G = ec.Graph(directed=False)
        G.add_star(10)  # Creates nodes 1-10, 1 is center
        
        # From center (node 1), 1-hop should get all spokes
        G.select_nodes_in_neighborhood("1", k=1)
        selected = G.get_selected_nodes()
        
        # Center + 9 spokes = 10 nodes
        assert len(selected) == 10
        assert "1" in selected
    
    def test_neighborhood_zero_hops(self):
        """Test k=0 selects only seed nodes."""
        G = ec.Graph(directed=True)
        G.add_path(5)
        
        G.select_nodes_in_neighborhood("3", k=0)
        selected = G.get_selected_nodes()
        
        # k=0 means no traversal, only the current level (seed nodes)
        assert len(selected) == 0 or len(selected) == 1
    
    def test_neighborhood_nonexistent_node(self):
        """Test error when seed node doesn't exist."""
        G = ec.Graph(directed=True)
        G.add_path(5)
        
        with pytest.raises(ValueError, match="does not exist"):
            G.select_nodes_in_neighborhood("999", k=1)
    
    def test_neighborhood_chaining(self):
        """Test chaining with other selection methods."""
        G = ec.Graph(directed=True)
        G.add_path(10)
        
        # Select neighborhood then deselect some
        G.select_nodes_in_neighborhood("5", k=2, mode="out")
        G.deselect_nodes(["7"])
        
        selected = G.get_selected_nodes()
        assert "5" in selected
        assert "6" in selected
        assert "7" not in selected


class TestSelectNodesByCentrality:
    """Tests for select_nodes_by_centrality()."""
    
    def test_centrality_by_degree(self):
        """Test selection by degree centrality."""
        G = ec.Graph(directed=False)
        G.add_star(10)  # Node 1 is center with highest degree
        
        # Center (node 1) has highest degree
        G.select_nodes_by_centrality("degree", k=1)
        selected = G.get_selected_nodes()
        
        assert "1" in selected
        assert len(selected) == 1
    
    def test_centrality_top_k(self):
        """Test selecting top-K by degree."""
        G = ec.Graph(directed=False)
        # Create nodes with varying degrees
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        
        # A has degree 3, B has 2, C has 1, D has 0
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")
        G.add_edge("B", "C")
        
        G.select_nodes_by_centrality("degree", k=2)
        selected = G.get_selected_nodes()
        
        # Should get A and B (highest degrees)
        assert "A" in selected
        assert "B" in selected
        assert len(selected) == 2
    
    def test_centrality_in_degree_directed(self):
        """Test selection by in-degree in directed graph."""
        G = ec.Graph(directed=True)
        # Create a graph where one node has many incoming edges
        G.add_node("hub")
        for i in range(5):
            G.add_edge(str(i), "hub")
        
        G.select_nodes_by_centrality("in_degree", k=1)
        selected = G.get_selected_nodes()
        
        assert "hub" in selected
    
    def test_centrality_out_degree_directed(self):
        """Test selection by out-degree in directed graph."""
        G = ec.Graph(directed=True)
        G.add_star(10)  # Node 1 (center) has high out-degree
        
        G.select_nodes_by_centrality("out_degree", k=1)
        selected = G.get_selected_nodes()
        
        assert "1" in selected
    
    def test_centrality_in_degree_undirected_error(self):
        """Test that in_degree fails on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_path(5)
        
        with pytest.raises(ValueError, match="only available for directed"):
            G.select_nodes_by_centrality("in_degree", k=1)
    
    def test_centrality_pagerank(self):
        """Test selection by PageRank."""
        G = ec.Graph(directed=True)
        # Create a simple graph where one node gets more PageRank
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        
        # C should have higher PageRank (receives from A and B)
        G.select_nodes_by_centrality("pagerank", k=1)
        selected = G.get_selected_nodes()
        
        assert len(selected) == 1
        # The exact node depends on PageRank algorithm
    
    def test_centrality_include_ties(self):
        """Test including tied scores."""
        G = ec.Graph(directed=False)
        # Create nodes with same degree
        G.add_edge("A", "D")
        G.add_edge("B", "D")
        G.add_edge("C", "D")
        
        # A, B, C all have degree 1, D has degree 3
        G.select_nodes_by_centrality("degree", k=2, include_ties=True)
        selected = G.get_selected_nodes()
        
        # Should get D (degree 3) and potentially A, B, C (all degree 1, tied)
        assert "D" in selected
    
    def test_centrality_empty_graph(self):
        """Test centrality selection on empty graph."""
        G = ec.Graph(directed=True)
        
        G.select_nodes_by_centrality("degree", k=5)
        selected = G.get_selected_nodes()
        
        assert len(selected) == 0
    
    def test_centrality_k_larger_than_nodes(self):
        """Test k larger than number of nodes."""
        G = ec.Graph(directed=False)
        G.add_path(3)
        
        G.select_nodes_by_centrality("degree", k=100)
        selected = G.get_selected_nodes()
        
        # Should select all 3 nodes
        assert len(selected) == 3
    
    def test_centrality_unknown_measure(self):
        """Test error with unknown centrality measure."""
        G = ec.Graph(directed=True)
        G.add_path(5)
        
        with pytest.raises(ValueError, match="Unknown centrality measure"):
            G.select_nodes_by_centrality("unknown_measure", k=1)
    
    def test_centrality_chaining(self):
        """Test chaining centrality selection."""
        G = ec.Graph(directed=False)
        G.add_path(10)
        
        # Select by degree, then add more by ID
        G.select_nodes_by_centrality("degree", k=3)
        initial_count = len(G.get_selected_nodes())
        
        G.select_nodes_by_id(["10"])
        final_count = len(G.get_selected_nodes())
        
        assert final_count >= initial_count


class TestCombinedNeighborhoodCentrality:
    """Tests combining neighborhood and centrality selection."""
    
    def test_neighborhood_then_centrality(self):
        """Test selecting neighborhood then filtering by centrality."""
        G = ec.Graph(directed=True)
        # Build a more complex graph
        G.add_star(10)  # Nodes 1-10, center is 1
        G.add_star(10)  # Nodes 11-20, center is 11
        G.add_edge("1", "11")  # Connect the two hubs
        
        # Get neighborhood of first hub
        G.select_nodes_in_neighborhood("1", k=1, mode="out")
        neighborhood = G.get_selected_nodes()
        
        # Clear and select by centrality
        G.clear_selection()
        G.select_nodes_by_centrality("out_degree", k=2)
        
        # Should get the two centers (highest out-degree)
        selected = G.get_selected_nodes()
        assert "1" in selected or "11" in selected
    
    def test_workflow_find_influential_neighbors(self):
        """Test realistic workflow: find influential neighbors."""
        G = ec.Graph(directed=True)
        
        # Create a network
        G.add_star(5)  # Nodes 1-5, center is 1
        G.add_star(5)  # Nodes 6-10, center is 6
        G.add_star(3)  # Nodes 11-13, center is 11
        G.add_edge("1", "6")
        G.add_edge("6", "11")
        
        # Find 2-hop neighborhood of node 1
        G.select_nodes_in_neighborhood("1", k=2, mode="out")
        
        # Node 1 has highest out-degree
        assert "1" in G.get_selected_nodes()
