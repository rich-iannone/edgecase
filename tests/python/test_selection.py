"""
Tests for node and edge selection methods.
"""

import pytest
from edgecase import Graph


class TestNodeSelection:
    """Test node selection operations."""
    
    def test_select_nodes_by_id_basic(self):
        """Basic node selection by ID."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Select specific nodes (use integer IDs)
        G.select_nodes_by_id([1, 3])  # A=1, C=3
        
        selected = G.get_selected_nodes()
        assert len(selected) == 2
        assert 1 in selected  # A
        assert 3 in selected  # C
        assert 2 not in selected  # B
        assert 4 not in selected  # D
    
    def test_select_nodes_by_id_chaining(self):
        """Chaining node selections."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Chain selections (use integer IDs)
        G.select_nodes_by_id([1, 2]).select_nodes_by_id([3, 4])  # A=1, B=2, C=3, D=4
        
        selected = G.get_selected_nodes()
        assert len(selected) == 4
        assert set(selected) == {1, 2, 3, 4}
    
    def test_select_nodes_by_id_error(self):
        """Selecting non-existent node raises error."""
        G = Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        
        with pytest.raises(KeyError, match="Node 99 does not exist"):
            G.select_nodes_by_id([99])  # Use non-existent integer ID
    
    def test_select_nodes_by_degree_total(self):
        """Select nodes by total degree."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")  # A has degree 3
        G.add_edge("B", "C")  # B has degree 2, C has degree 2
        
        # Select nodes with degree >= 3
        G.select_nodes_by_degree(min_degree=3)
        
        selected = G.get_selected_nodes()
        assert len(selected) == 1
        assert 1 in selected  # A
    
    def test_select_nodes_by_degree_range(self):
        """Select nodes within degree range."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E", "F"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("B", "D")
        G.add_edge("E", "F")  # E and F have degree 1
        
        # Select nodes with degree 2
        G.select_nodes_by_degree(min_degree=2, max_degree=2)
        
        selected = G.get_selected_nodes()
        assert len(selected) == 2
        assert 1 in selected  # A
        assert 3 in selected  # C
    
    def test_select_nodes_by_degree_directed_out(self):
        """Select nodes by out-degree in directed graph."""
        G = Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")  # A has out-degree 3
        G.add_edge("B", "C")  # B has out-degree 1
        
        # Select nodes with out-degree >= 2
        G.select_nodes_by_degree(min_degree=2, degree_type="out")
        
        selected = G.get_selected_nodes()
        assert len(selected) == 1
        assert 1 in selected  # A
    
    def test_select_nodes_by_degree_directed_in(self):
        """Select nodes by in-degree in directed graph."""
        G = Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "D")
        G.add_edge("B", "D")
        G.add_edge("C", "D")  # D has in-degree 3
        G.add_edge("B", "C")  # C has in-degree 1
        
        # Select nodes with in-degree >= 2
        G.select_nodes_by_degree(min_degree=2, degree_type="in")
        
        selected = G.get_selected_nodes()
        assert len(selected) == 1
        assert 4 in selected  # D
    
    def test_select_nodes_by_degree_invalid_type(self):
        """Invalid degree type raises error."""
        G = Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        
        with pytest.raises(ValueError, match="Invalid degree_type"):
            G.select_nodes_by_degree(min_degree=1, degree_type="invalid")
    
    def test_deselect_nodes(self):
        """Deselect specific nodes."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Select all, then deselect some (use integer IDs)
        G.select_nodes_by_id([1, 2, 3, 4])
        G.deselect_nodes([2, 4])  # B=2, D=4
        
        selected = G.get_selected_nodes()
        assert len(selected) == 2
        assert 1 in selected  # A
        assert 3 in selected  # C
    
    def test_clear_selection_nodes(self):
        """Clear node selection."""
        G = Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        G.select_nodes_by_id([1, 2])  # A=1, B=2
        assert G.count_selected_nodes() == 2
        
        G.clear_selection()
        assert G.count_selected_nodes() == 0
        assert len(G.get_selected_nodes()) == 0
    
    def test_invert_node_selection(self):
        """Invert node selection."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Select A and B (use integer IDs)
        G.select_nodes_by_id([1, 2])  # A=1, B=2
        
        # Invert - should select C and D
        G.invert_node_selection()
        
        selected = G.get_selected_nodes()
        assert len(selected) == 2
        assert 3 in selected  # C
        assert 4 in selected  # D
        assert 1 not in selected  # A
        assert 2 not in selected  # B
    
    def test_count_selected_nodes(self):
        """Count selected nodes."""
        G = Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        assert G.count_selected_nodes() == 0
        
        G.select_nodes_by_id([1, 2])  # A=1, B=2
        assert G.count_selected_nodes() == 2


class TestEdgeSelection:
    """Test edge selection operations."""
    
    def test_select_edges_by_id_basic(self):
        """Basic edge selection by ID."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        edge_ids = G.edge_ids()
        
        # Select first two edges
        G.select_edges_by_id([edge_ids[0], edge_ids[1]])
        
        selected = G.get_selected_edges()
        assert len(selected) == 2
        assert edge_ids[0] in selected
        assert edge_ids[1] in selected
        assert edge_ids[2] not in selected
    
    def test_select_edges_by_id_chaining(self):
        """Chaining edge selections."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        edge_ids = G.edge_ids()
        
        # Chain selections
        G.select_edges_by_id([edge_ids[0]]).select_edges_by_id([edge_ids[1]])
        
        selected = G.get_selected_edges()
        assert len(selected) == 2
    
    def test_select_edges_by_id_error(self):
        """Selecting non-existent edge raises error."""
        G = Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B")
        
        with pytest.raises(KeyError, match="Edge with ID 999 does not exist"):
            G.select_edges_by_id([999])
    
    def test_select_edges_by_node_id(self):
        """Select edges connected to specific nodes."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        G.add_edge("D", "E")
        
        # Select edges connected to B (should get A-B and B-C)
        G.select_edges_by_node_id([2])  # B=2
        
        selected_edges = G.get_selected_edges()
        assert len(selected_edges) == 2
        
        # Verify the correct edges are selected (edges() returns (id, source_id, target_id))
        edges = G.edges()
        selected_edge_info = [e for e in edges if e[0] in selected_edges]
        
        sources_targets = [(e[1], e[2]) for e in selected_edge_info]
        assert (1, 2) in sources_targets  # A-B
        assert (2, 3) in sources_targets  # B-C
    
    def test_select_edges_by_node_id_multiple_nodes(self):
        """Select edges connected to multiple nodes."""
        G = Graph()
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        G.add_edge("D", "E")
        
        # Select edges connected to A or D
        G.select_edges_by_node_id([1, 4])  # A=1, D=4
        
        selected_edges = G.get_selected_edges()
        assert len(selected_edges) == 3  # A-B, C-D, D-E
        
        edges = G.edges()
        selected_edge_info = [e for e in edges if e[0] in selected_edges]
        sources_targets = [(e[1], e[2]) for e in selected_edge_info]
        
        assert (1, 2) in sources_targets  # A-B
        assert (3, 4) in sources_targets  # C-D
        assert (4, 5) in sources_targets  # D-E
    
    def test_deselect_edges(self):
        """Deselect specific edges."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        edge_ids = G.edge_ids()
        
        # Select all, then deselect one
        G.select_edges_by_id(edge_ids)
        G.deselect_edges([edge_ids[1]])
        
        selected = G.get_selected_edges()
        assert len(selected) == 2
        assert edge_ids[0] in selected
        assert edge_ids[2] in selected
        assert edge_ids[1] not in selected
    
    def test_clear_selection_edges(self):
        """Clear edge selection."""
        G = Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        edge_ids = G.edge_ids()
        G.select_edges_by_id(edge_ids)
        assert G.count_selected_edges() == 2
        
        G.clear_selection()
        assert G.count_selected_edges() == 0
        assert len(G.get_selected_edges()) == 0
    
    def test_invert_edge_selection(self):
        """Invert edge selection."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        edge_ids = G.edge_ids()
        
        # Select first edge
        G.select_edges_by_id([edge_ids[0]])
        
        # Invert - should select other two
        G.invert_edge_selection()
        
        selected = G.get_selected_edges()
        assert len(selected) == 2
        assert edge_ids[1] in selected
        assert edge_ids[2] in selected
        assert edge_ids[0] not in selected
    
    def test_count_selected_edges(self):
        """Count selected edges."""
        G = Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        assert G.count_selected_edges() == 0
        
        edge_ids = G.edge_ids()
        G.select_edges_by_id([edge_ids[0]])
        assert G.count_selected_edges() == 1


class TestCombinedSelection:
    """Test combined node and edge selections."""
    
    def test_select_nodes_and_edges(self):
        """Select both nodes and edges."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Select some nodes (use integer IDs)
        G.select_nodes_by_id([1, 2])  # A=1, B=2
        assert G.count_selected_nodes() == 2
        
        # Select some edges
        edge_ids = G.edge_ids()
        G.select_edges_by_id([edge_ids[0]])
        assert G.count_selected_edges() == 1
        
        # Both selections should be independent
        assert G.count_selected_nodes() == 2
        assert G.count_selected_edges() == 1
    
    def test_clear_selection_clears_both(self):
        """Clear selection clears both nodes and edges."""
        G = Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        # Select nodes and edges (use integer IDs)
        G.select_nodes_by_id([1, 2])  # A=1, B=2
        edge_ids = G.edge_ids()
        G.select_edges_by_id([edge_ids[0]])
        
        assert G.count_selected_nodes() == 2
        assert G.count_selected_edges() == 1
        
        # Clear all
        G.clear_selection()
        
        assert G.count_selected_nodes() == 0
        assert G.count_selected_edges() == 0
    
    def test_complex_selection_workflow(self):
        """Complex selection workflow with chaining."""
        G = Graph(directed=True)
        
        # Build a simple network
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")
        G.add_edge("D", "E")
        
        # Select high out-degree nodes
        G.select_nodes_by_degree(min_degree=2, degree_type="out")
        selected_nodes = G.get_selected_nodes()
        
        # Should select A (out-degree 2) and B (out-degree 2) - IDs 1 and 2
        assert len(selected_nodes) == 2
        assert 1 in selected_nodes  # A
        assert 2 in selected_nodes  # B
        
        # Now select edges connected to these nodes
        G.select_edges_by_node_id(selected_nodes)
        selected_edges = G.get_selected_edges()
        
        # Should select all edges involving A or B
        assert G.count_selected_edges() == 4  # A-B, A-C, B-C, B-D


class TestSelectionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_selection_on_empty_graph(self):
        """Selection operations on empty graph."""
        G = Graph()
        
        # Should not crash
        G.select_nodes_by_degree(min_degree=1)
        assert G.count_selected_nodes() == 0
        
        G.invert_node_selection()
        assert G.count_selected_nodes() == 0
        
        G.clear_selection()
        assert G.count_selected_nodes() == 0
        assert G.count_selected_edges() == 0
    
    def test_deselect_unselected(self):
        """Deselecting nodes/edges that aren't selected."""
        G = Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        # Deselect without selecting first - should not error
        G.deselect_nodes([1])  # A=1
        G.deselect_edges([1])
        
        assert G.count_selected_nodes() == 0
        assert G.count_selected_edges() == 0
    
    def test_select_all_nodes(self):
        """Select all nodes by inverting empty selection."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Invert empty selection = select all
        G.invert_node_selection()
        
        assert G.count_selected_nodes() == 4
        selected = G.get_selected_nodes()
        assert set(selected) == {1, 2, 3, 4}  # Integer IDs
    
    def test_select_all_edges(self):
        """Select all edges by inverting empty selection."""
        G = Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        edge_ids = G.edge_ids()
        
        # Invert empty selection = select all
        G.invert_edge_selection()
        
        assert G.count_selected_edges() == 3
        selected = G.get_selected_edges()
        assert set(selected) == set(edge_ids)
    
    def test_method_chaining_returns_self(self):
        """Verify method chaining returns Graph object."""
        G = Graph()
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        # All selection methods should return self
        result = (G.select_nodes_by_id([1])  # A=1
                   .select_nodes_by_degree(min_degree=1)
                   .deselect_nodes([])
                   .invert_node_selection()
                   .clear_selection())
        
        assert result is G
