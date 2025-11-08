"""
Tests for set_edge_attrs() method.
"""

import pytest
import edgecase as ec


class TestSetEdgeAttrsBasic:
    """Test basic functionality of set_edge_attrs()."""
    
    def test_set_edge_attr_all_edges_single_value(self):
        """Set same value for all edges."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        G.set_edge_attrs("color", "red")
        
        result = G.get_edge_attrs("color")
        assert result == {"A->B": "red", "B->C": "red", "C->D": "red"}
    
    def test_set_edge_attr_specific_edges(self):
        """Set attribute for specific edges only."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        G.set_edge_attrs("color", "blue", edges=[("A", "B"), ("C", "D")])
        
        result = G.get_edge_attrs("color")
        assert result == {"A->B": "blue", "C->D": "blue"}
        # Middle edge should not have the attribute
        assert "B->C" not in result
    
    def test_set_edge_attr_from_to_filtering(self):
        """Set attribute using from/to node filtering."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("B", "D")
        
        # Set for edges from A to B or C
        G.set_edge_attrs("type", "important", from_nodes=["A"], to_nodes=["B", "C"])
        
        result = G.get_edge_attrs("type")
        assert result == {"A->B": "important", "A->C": "important"}
    
    def test_set_edge_attr_list_values(self):
        """Set different values using a list."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        edge_ids = G.edge_ids()
        G.set_edge_attrs("priority", [1, 2, 3], edges=[("A", "B"), ("B", "C"), ("C", "D")])
        
        result = G.get_edge_attrs("priority")
        assert len(result) == 3
        assert set(result.values()) == {1, 2, 3}
    
    def test_set_edge_attr_dict_values(self):
        """Set values using a dict mapping edge IDs to values."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        edge_ids = G.edge_ids()
        value_dict = {edge_ids[0]: "high", edge_ids[1]: "low"}
        
        G.set_edge_attrs("priority", value_dict)
        
        result = G.get_edge_attrs("priority")
        assert len(result) == 2
        assert set(result.values()) == {"high", "low"}


class TestSetEdgeAttrsWeight:
    """Test weight attribute handling (special case)."""
    
    def test_set_weight_all_edges(self):
        """Set weight for all edges."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        G.set_edge_attrs("weight", 5.0)
        
        result = G.get_edge_attrs("weight")
        assert result == {"A->B": 5.0, "B->C": 5.0}
    
    def test_set_weight_specific_edges(self):
        """Set weight for specific edges."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=2.0)
        G.add_edge("C", "D", weight=3.0)
        
        G.set_edge_attrs("weight", 10.0, edges=[("A", "B"), ("C", "D")])
        
        result = G.get_edge_attrs("weight")
        assert result["A->B"] == 10.0
        assert result["B->C"] == 2.0  # Unchanged
        assert result["C->D"] == 10.0
    
    def test_set_weight_list_values(self):
        """Set different weights using a list."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        G.set_edge_attrs("weight", [1.5, 2.5], edges=[("A", "B"), ("B", "C")])
        
        result = G.get_edge_attrs("weight")
        assert result["A->B"] == 1.5
        assert result["B->C"] == 2.5


class TestSetEdgeAttrsMultigraph:
    """Test with multiple edges between same nodes."""
    
    def test_set_attr_multigraph_all_parallel_edges(self):
        """Set attribute on all parallel edges."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", type="friend")
        G.add_edge("A", "B", type="colleague")
        G.add_edge("A", "B", type="family")
        
        # Should update all three edges
        G.set_edge_attrs("status", "active", edges=[("A", "B")])
        
        result = G.get_edge_attrs_by_id("status")
        # All three A->B edges should have status
        assert len(result) == 3
        assert all(v == "active" for v in result.values())
    
    def test_set_attr_by_edge_id_multigraph(self):
        """Set attribute on specific parallel edge by ID."""
        G = ec.Graph()
        G.add_edge("A", "B", type="friend")
        G.add_edge("A", "B", type="colleague")
        
        edge_ids = G.edge_ids()
        
        # Set attribute using dict with specific edge ID
        G.set_edge_attrs("priority", {edge_ids[0]: "high"})
        
        result = G.get_edge_attrs_by_id("priority")
        assert edge_ids[0] in result
        assert result[edge_ids[0]] == "high"
        # Second edge should not have priority
        assert edge_ids[1] not in result


class TestSetEdgeAttrsChaining:
    """Test method chaining."""
    
    def test_chaining_returns_self(self):
        """Verify set_edge_attrs returns self for chaining."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        
        result = G.set_edge_attrs("color", "red")
        assert result is G
    
    def test_chaining_multiple_operations(self):
        """Chain multiple set_edge_attrs calls."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        G.set_edge_attrs("color", "red").set_edge_attrs("weight", 2.0).set_edge_attrs("type", "link")
        
        assert G.get_edge_attrs("color") == {"A->B": "red", "B->C": "red"}
        assert G.get_edge_attrs("weight") == {"A->B": 2.0, "B->C": 2.0}
        assert G.get_edge_attrs("type") == {"A->B": "link", "B->C": "link"}


class TestSetEdgeAttrsDirected:
    """Test with directed graphs."""
    
    def test_directed_graph_from_to_filtering(self):
        """Test from/to filtering respects direction."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "A")  # Reverse edge
        G.add_edge("A", "C")
        
        # Only A->B should be updated
        G.set_edge_attrs("marked", True, from_nodes=["A"], to_nodes=["B"])
        
        result = G.get_edge_attrs("marked")
        assert result == {"A->B": True}
        # B->A should not be marked
        assert "B->A" not in result


class TestSetEdgeAttrsUndirected:
    """Test with undirected graphs."""
    
    def test_undirected_graph_basic(self):
        """Set attributes on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        G.set_edge_attrs("type", "connection")
        
        result = G.get_edge_attrs("type")
        assert result == {"A--B": "connection", "B--C": "connection"}


class TestSetEdgeAttrsErrors:
    """Test error handling."""
    
    def test_error_list_length_mismatch(self):
        """Raise error if list length doesn't match edges."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        with pytest.raises(ValueError, match="Length of values .* must match"):
            G.set_edge_attrs("color", ["red"], edges=[("A", "B"), ("B", "C")])
    
    def test_error_nonexistent_edge(self):
        """Handle nonexistent edges gracefully."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        
        # Trying to set attribute on nonexistent edge should just not find it
        # (find_edge_ids returns empty list)
        G.set_edge_attrs("color", "red", edges=[("X", "Y")])
        
        # Should not raise error, just have no effect
        result = G.get_edge_attrs("color")
        assert result == {}


class TestSetEdgeAttrsEmptyGraph:
    """Test edge cases with empty or minimal graphs."""
    
    def test_empty_graph(self):
        """Set attributes on empty graph should work but have no effect."""
        G = ec.Graph(directed=True)
        G.set_edge_attrs("color", "red")
        
        result = G.get_edge_attrs("color")
        assert result == {}
    
    def test_single_edge(self):
        """Set attribute on graph with single edge."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        
        G.set_edge_attrs("important", True)
        
        result = G.get_edge_attrs("important")
        assert result == {"A->B": True}


class TestSetEdgeAttrsOverwrite:
    """Test overwriting existing attributes."""
    
    def test_overwrite_existing_attr(self):
        """Overwrite existing attribute value."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", color="red")
        
        G.set_edge_attrs("color", "blue")
        
        result = G.get_edge_attrs("color")
        assert result == {"A->B": "blue"}
    
    def test_overwrite_weight(self):
        """Overwrite existing weight."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", weight=1.0)
        
        G.set_edge_attrs("weight", 5.0)
        
        result = G.get_edge_attrs("weight")
        assert result == {"A->B": 5.0}
    
    def test_partial_overwrite(self):
        """Overwrite some edges but not all."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B", status="old")
        G.add_edge("B", "C", status="old")
        G.add_edge("C", "D", status="old")
        
        G.set_edge_attrs("status", "new", edges=[("A", "B"), ("C", "D")])
        
        result = G.get_edge_attrs("status")
        assert result["A->B"] == "new"
        assert result["B->C"] == "old"  # Unchanged
        assert result["C->D"] == "new"
