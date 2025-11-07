"""
Test suite for graph traversal operations.

Tests the 7 core traversal methods (trav_out, trav_in, trav_both,
trav_out_edge, trav_in_edge, trav_reverse_edge).
"""

import pytest
import edgecase as ec


class TestBasicTraversal:
    """Tests for trav_out, trav_in, trav_both."""
    
    def test_trav_out_directed(self):
        """Test outward traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        
        # Traverse from A to its neighbors
        G.select_nodes_by_id(["A"]).trav_out()
        nodes = sorted(G.get_selected_nodes())
        assert nodes == ["B", "C"]
        
    def test_trav_out_multi_hop(self):
        """Test chaining trav_out for multi-hop traversal."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Traverse 2 hops from A
        G.select_nodes_by_id(["A"]).trav_out().trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == ["C"]
        
    def test_trav_in_directed(self):
        """Test inward traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Traverse from C to predecessors
        G.select_nodes_by_id(["C"]).trav_in()
        nodes = sorted(G.get_selected_nodes())
        assert nodes == ["A", "B"]
        
    def test_trav_both_directed(self):
        """Test both-direction traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("C", "B")
        G.add_edge("B", "D")
        
        # Traverse from B to all neighbors
        G.select_nodes_by_id(["B"]).trav_both()
        nodes = sorted(G.get_selected_nodes())
        assert nodes == ["A", "C", "D"]
        
    def test_trav_out_undirected(self):
        """Test that undirected graphs work like trav_both."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        
        # In undirected, trav_out should get all neighbors
        G.select_nodes_by_id(["A"]).trav_out()
        nodes = sorted(G.get_selected_nodes())
        assert nodes == ["B", "C"]
        
    def test_trav_empty_selection(self):
        """Test traversal with no selection does nothing."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        
        # No selection
        G.trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == []
        
    def test_trav_no_neighbors(self):
        """Test traversal from node with no neighbors."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        G.add_node("B")
        
        # Node A has no outgoing edges
        G.select_nodes_by_id(["A"]).trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == []


class TestEdgeTraversal:
    """Tests for trav_out_edge, trav_in_edge."""
    
    def test_trav_out_edge_directed(self):
        """Test outward edge traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        
        # Traverse from A to its outgoing edges
        G.select_nodes_by_id(["A"]).trav_out_edge()
        
        # Should select 2 edges (A->B, A->C)
        edges = G.get_selected_edges()
        assert len(edges) == 2
        
        # Node selection should be cleared
        nodes = G.get_selected_nodes()
        assert nodes == []
        
    def test_trav_in_edge_directed(self):
        """Test inward edge traversal in directed graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Traverse from C to its incoming edges
        G.select_nodes_by_id(["C"]).trav_in_edge()
        
        # Should select 2 edges (A->C, B->C)
        edges = G.get_selected_edges()
        assert len(edges) == 2
        
        # Node selection should be cleared
        nodes = G.get_selected_nodes()
        assert nodes == []
        
    def test_trav_out_edge_undirected(self):
        """Test edge traversal in undirected graph."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        
        # In undirected, trav_out_edge should get all incident edges
        G.select_nodes_by_id(["A"]).trav_out_edge()
        edges = G.get_selected_edges()
        assert len(edges) == 2
        
    def test_trav_edge_multiple_nodes(self):
        """Test edge traversal from multiple selected nodes."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        
        # Select both A and B
        G.select_nodes_by_id(["A", "B"]).trav_out_edge()
        edges = G.get_selected_edges()
        assert len(edges) == 2


class TestReverseEdgeTraversal:
    """Tests for trav_reverse_edge."""
    
    def test_trav_reverse_edge_basic(self):
        """Test basic reverse edge traversal."""
        G = ec.Graph(directed=True)
        # Add bidirectional edges
        edge_id_1 = G.add_edge("A", "B")
        edge_id_2 = G.add_edge("B", "A")
        
        # Select A->B edge
        G.select_edges_by_id([edge_id_1]).trav_reverse_edge()
        
        # Should now select B->A edge
        selected = G.get_selected_edges()
        assert edge_id_2 in selected
        assert edge_id_1 not in selected
        
    def test_trav_reverse_edge_add_to_selection(self):
        """Test reverse edge with add_to_selection=True."""
        G = ec.Graph(directed=True)
        edge_id_1 = G.add_edge("A", "B")
        edge_id_2 = G.add_edge("B", "A")
        
        # Select A->B and add its reverse
        G.select_edges_by_id([edge_id_1]).trav_reverse_edge(add_to_selection=True)
        
        # Should have both edges selected
        selected = G.get_selected_edges()
        assert edge_id_1 in selected
        assert edge_id_2 in selected
        assert len(selected) == 2
        
    def test_trav_reverse_edge_no_reverse(self):
        """Test reverse edge when no reverse exists."""
        G = ec.Graph(directed=True)
        edge_id = G.add_edge("A", "B")
        
        # Select A->B (no B->A exists)
        G.select_edges_by_id([edge_id]).trav_reverse_edge()
        
        # Selection should be empty
        selected = G.get_selected_edges()
        assert selected == []
        
    def test_trav_reverse_edge_multiple(self):
        """Test reverse edge with multiple selected edges."""
        G = ec.Graph(directed=True)
        edge_id_1 = G.add_edge("A", "B")
        edge_id_2 = G.add_edge("B", "A")
        edge_id_3 = G.add_edge("C", "D")
        edge_id_4 = G.add_edge("D", "C")
        
        # Select A->B and C->D
        G.select_edges_by_id([edge_id_1, edge_id_3]).trav_reverse_edge()
        
        # Should select B->A and D->C
        selected = G.get_selected_edges()
        assert edge_id_2 in selected
        assert edge_id_4 in selected
        assert len(selected) == 2


class TestTraversalChaining:
    """Tests for chaining multiple traversal operations."""
    
    def test_chain_out_out(self):
        """Test chaining trav_out twice."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)  # Creates chain: 1->2->3->4->5
        
        # Traverse 2 hops from node 1
        G.select_nodes_by_id(["1"]).trav_out().trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == ["3"]
        
    def test_chain_out_in(self):
        """Test chaining trav_out then trav_in."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        # Traverse out from A (to B), then back in (to A)
        G.select_nodes_by_id(["A"]).trav_out().trav_in()
        nodes = G.get_selected_nodes()
        assert nodes == ["A"]
        
    def test_chain_node_to_edge_to_node(self):
        """Test chaining node->edge->node traversals."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        # Select A, get its outgoing edges, then traverse back to A
        G.select_nodes_by_id(["A"]).trav_out_edge()
        
        # Edge selection active now
        edges = G.get_selected_edges()
        assert len(edges) == 1


class TestTraversalEdgeCases:
    """Edge cases and error conditions."""
    
    def test_trav_single_node_graph(self):
        """Test traversal in single-node graph."""
        G = ec.Graph(directed=True)
        G.add_node("A")
        
        G.select_nodes_by_id(["A"]).trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == []
        
    def test_trav_self_loop(self):
        """Test traversal with self-loop."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "A")  # Self-loop
        
        G.select_nodes_by_id(["A"]).trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == ["A"]
        
    def test_trav_disconnected_graph(self):
        """Test traversal in disconnected graph."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("C", "D")
        
        # Traverse from A (should not reach C or D)
        G.select_nodes_by_id(["A"]).trav_out()
        nodes = G.get_selected_nodes()
        assert nodes == ["B"]
        assert "C" not in nodes
        assert "D" not in nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
