"""Tests for query operations."""

import pytest
import polars as pl
import edgecase as ec
from edgecase.builders import add_path, add_star


class TestGetNodeAttrs:
    """Tests for get_node_attrs()."""
    
    def test_get_node_attrs_basic(self):
        """Test extracting node attributes."""
        G = ec.Graph()
        G.add_node("A", type="user", score=95)
        G.add_node("B", type="admin", score=88)
        G.add_node("C", type="user", score=92)
        
        scores = G.get_node_attrs("score")
        
        assert scores == {"A": 95, "B": 88, "C": 92}
    
    def test_get_node_attrs_specific_nodes(self):
        """Test extracting attrs from specific nodes."""
        G = ec.Graph()
        G.add_node("A", score=95)
        G.add_node("B", score=88)
        G.add_node("C", score=92)
        
        scores = G.get_node_attrs("score", nodes=["A", "C"])
        
        assert scores == {"A": 95, "C": 92}
        assert "B" not in scores
    
    def test_get_node_attrs_missing_attr(self):
        """Test when some nodes don't have the attribute."""
        G = ec.Graph()
        G.add_node("A", score=95)
        G.add_node("B")  # No score
        G.add_node("C", score=92)
        
        scores = G.get_node_attrs("score")
        
        assert scores == {"A": 95, "C": 92}
        assert "B" not in scores
    
    def test_get_node_attrs_empty_graph(self):
        """Test on empty graph."""
        G = ec.Graph()
        
        scores = G.get_node_attrs("score")
        
        assert scores == {}


class TestGetEdgeAttrs:
    """Tests for get_edge_attrs()."""
    
    def test_get_edge_attrs_basic(self):
        """Test extracting edge attributes."""
        G = ec.Graph()
        G.add_edge("A", "B", weight=1.5)
        G.add_edge("B", "C", weight=2.3)
        G.add_edge("C", "D", weight=0.8)
        
        weights = G.get_edge_attrs("weight")
        
        assert weights == {1: 1.5, 2: 2.3, 3: 0.8}
    
    def test_get_edge_attrs_specific_edges(self):
        """Test extracting attrs from specific edges."""
        G = ec.Graph()
        G.add_edge("A", "B", weight=1.5)
        G.add_edge("B", "C", weight=2.3)
        G.add_edge("C", "D", weight=0.8)
        
        weights = G.get_edge_attrs("weight", edges=[1, 3])
        
        assert weights == {1: 1.5, 3: 0.8}
        assert 2 not in weights


class TestGetNodeDF:
    """Tests for get_node_df()."""
    
    def test_get_node_df_basic(self):
        """Test getting nodes as DataFrame."""
        G = ec.Graph()
        G.add_node("A", type="user", score=95)
        G.add_node("B", type="admin", score=88)
        
        df = G.get_node_df()
        
        assert isinstance(df, pl.DataFrame)
        assert "id" in df.columns
        assert "type" in df.columns
        assert "score" in df.columns
        assert "degree" in df.columns
        assert len(df) == 2
    
    def test_get_node_df_without_degree(self):
        """Test DataFrame without degree column."""
        G = ec.Graph()
        G.add_node("A", score=95)
        G.add_node("B", score=88)
        
        df = G.get_node_df(include_degree=False)
        
        assert "degree" not in df.columns
        assert "id" in df.columns
    
    def test_get_node_df_empty_graph(self):
        """Test on empty graph."""
        G = ec.Graph()
        
        df = G.get_node_df()
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
    
    def test_get_node_df_with_edges(self):
        """Test that degree column is correct."""
        G = ec.Graph(directed=False)
        add_path(G, 5)
        
        df = G.get_node_df()
        
        # Path graph: nodes have degree 1 or 2
        degrees = df["degree"].to_list()
        assert degrees.count(1) == 2  # Endpoints
        assert degrees.count(2) == 3  # Middle nodes


class TestGetEdgeDF:
    """Tests for get_edge_df()."""
    
    def test_get_edge_df_basic(self):
        """Test getting edges as DataFrame."""
        G = ec.Graph()
        G.add_edge("A", "B", weight=1.5, type="friend")
        G.add_edge("B", "C", weight=2.3, type="colleague")
        
        df = G.get_edge_df()
        
        assert isinstance(df, pl.DataFrame)
        assert "id" in df.columns
        assert "source" in df.columns
        assert "target" in df.columns
        assert "weight" in df.columns
        assert "type" in df.columns
        assert len(df) == 2
    
    def test_get_edge_df_without_endpoints(self):
        """Test DataFrame without source/target columns."""
        G = ec.Graph()
        G.add_edge("A", "B", weight=1.5)
        
        df = G.get_edge_df(include_endpoints=False)
        
        assert "source" not in df.columns
        assert "target" not in df.columns
        assert "id" in df.columns
        assert "weight" in df.columns
    
    def test_get_edge_df_empty_graph(self):
        """Test on empty graph."""
        G = ec.Graph()
        
        df = G.get_edge_df()
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


class TestGetDFWithSelection:
    """Tests for get_node_df_ws() and get_edge_df_ws()."""
    
    def test_get_node_df_ws(self):
        """Test getting selected nodes as DataFrame."""
        G = ec.Graph()
        add_path(G, 5)
        
        # Select specific nodes
        G.select_nodes_by_id(["1", "3", "5"])
        df = G.get_node_df_ws()
        
        assert len(df) == 3
        assert set(df["id"].to_list()) == {"1", "3", "5"}
    
    def test_get_edge_df_ws(self):
        """Test getting selected edges as DataFrame."""
        G = ec.Graph()
        add_path(G, 5)
        
        # Select specific edges
        G.select_edges_by_id([1, 3])
        df = G.get_edge_df_ws()
        
        assert len(df) == 2
        assert set(df["id"].to_list()) == {1, 3}
    
    def test_get_node_df_ws_empty_selection(self):
        """Test with no selection."""
        G = ec.Graph()
        add_path(G, 5)
        
        df = G.get_node_df_ws()
        
        assert len(df) == 0
    
    def test_get_df_ws_workflow(self):
        """Test realistic workflow with selection and DataFrame."""
        G = ec.Graph(directed=False)
        add_star(G, 10)
        
        # Select high-degree nodes
        G.select_nodes_by_degree(min_degree=5)
        df = G.get_node_df_ws()
        
        # Should only get the center node
        assert len(df) == 1
        assert df["degree"].to_list()[0] == 9


class TestGetNeighbors:
    """Tests for get_neighbors()."""
    
    def test_get_neighbors_directed_out(self):
        """Test getting outgoing neighbors."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("D", "A")
        
        neighbors = G.get_neighbors("A", mode="out")
        
        assert set(neighbors) == {"B", "C"}
        assert "D" not in neighbors
    
    def test_get_neighbors_directed_in(self):
        """Test getting incoming neighbors."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("D", "A")
        
        neighbors = G.get_neighbors("A", mode="in")
        
        assert set(neighbors) == {"D"}
        assert "B" not in neighbors
        assert "C" not in neighbors
    
    def test_get_neighbors_directed_all(self):
        """Test getting all neighbors."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("D", "A")
        
        neighbors = G.get_neighbors("A", mode="all")
        
        assert set(neighbors) == {"B", "C", "D"}
    
    def test_get_neighbors_undirected(self):
        """Test on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("D", "A")
        
        neighbors = G.get_neighbors("A")
        
        assert set(neighbors) == {"B", "C", "D"}
    
    def test_get_neighbors_with_predicate(self):
        """Test filtering neighbors with predicate."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        G.add_node("B", type="user")
        G.add_node("C", type="admin")
        G.add_node("D", type="user")
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")
        
        # Filter for user type nodes
        def is_user(node_id):
            node = G.get_node(node_id)
            return node and node.attributes and node.attributes.get("type") == "user"
        
        neighbors = G.get_neighbors("A", predicate=is_user)
        
        assert set(neighbors) == {"B", "D"}
        assert "C" not in neighbors


class TestGetCommonNeighbors:
    """Tests for get_common_neighbors()."""
    
    def test_get_common_neighbors_basic(self):
        """Test finding common neighbors."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("A", "D")
        G.add_edge("B", "D")
        G.add_edge("A", "E")  # Only A connects to E
        
        common = G.get_common_neighbors("A", "B")
        
        assert set(common) == {"C", "D"}
        assert "E" not in common
    
    def test_get_common_neighbors_none(self):
        """Test when there are no common neighbors."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        
        common = G.get_common_neighbors("A", "B")
        
        assert common == []


class TestGetPredecessorsSuccessors:
    """Tests for get_predecessors() and get_successors()."""
    
    def test_get_predecessors(self):
        """Test getting predecessors."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        predecessors = G.get_predecessors("C")
        
        assert set(predecessors) == {"A", "B"}
    
    def test_get_successors(self):
        """Test getting successors."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")
        
        successors = G.get_successors("A")
        
        assert set(successors) == {"B", "C", "D"}
    
    def test_predecessors_undirected_error(self):
        """Test that predecessors fails on undirected graph."""
        G = ec.Graph(directed=False)
        add_path(G, 5)
        
        with pytest.raises(ValueError, match="only available for directed"):
            G.get_predecessors("3")
    
    def test_successors_undirected_error(self):
        """Test that successors fails on undirected graph."""
        G = ec.Graph(directed=False)
        add_path(G, 5)
        
        with pytest.raises(ValueError, match="only available for directed"):
            G.get_successors("3")


class TestQueryWorkflows:
    """Test realistic query workflows."""
    
    def test_workflow_analyze_with_polars(self):
        """Test workflow: select -> DataFrame -> Polars analysis."""
        G = ec.Graph(directed=False)
        
        # Build a network
        for i in range(20):
            G.add_node(str(i), score=i * 5, category="A" if i % 2 == 0 else "B")
        
        # Add edges
        for i in range(19):
            G.add_edge(str(i), str(i + 1))
        
        # Get as DataFrame and analyze
        df = G.get_node_df()
        
        # Use Polars for analysis
        high_score = df.filter(pl.col("score") > 50)
        assert len(high_score) > 0
        
        # Group by category
        by_category = df.group_by("category").agg(pl.col("score").mean())
        assert len(by_category) == 2
    
    def test_workflow_selected_subgraph_analysis(self):
        """Test analyzing a selected subgraph."""
        G = ec.Graph(directed=True)
        add_star(G, 10)
        
        # Select high-degree nodes and their neighbors
        G.select_nodes_by_degree(min_degree=5)
        G.select_nodes_in_neighborhood(G.get_selected_nodes(), k=1, mode="out")
        
        # Get selected subgraph as DataFrame
        df = G.get_node_df_ws()
        
        # Should have center + all spokes
        assert len(df) == 10
    
    def test_workflow_filter_extract_attributes(self):
        """Test filtering and extracting specific attributes."""
        G = ec.Graph()
        
        # Create nodes with different attributes
        G.add_node("A", priority="high", value=100)
        G.add_node("B", priority="low", value=50)
        G.add_node("C", priority="high", value=150)
        G.add_node("D", priority="medium", value=75)
        
        # Get all values
        values = G.get_node_attrs("value")
        assert sum(values.values()) == 375
        
        # Get only high priority node values
        high_priority_nodes = [nid for nid in G.node_ids()
                               if G.get_node(nid).attributes.get("priority") == "high"]
        high_values = G.get_node_attrs("value", nodes=high_priority_nodes)
        
        assert set(high_values.keys()) == {"A", "C"}
        assert sum(high_values.values()) == 250
