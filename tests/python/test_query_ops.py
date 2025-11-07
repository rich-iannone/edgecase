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


class TestGetNonNeighbors:
    """Tests for get_non_neighbors()."""
    
    def test_non_neighbors_path_graph(self):
        """Test non-neighbors on a path graph."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)  # 1 -> 2 -> 3 -> 4 -> 5
        
        # Node 2 connects to 1 and 3, so non-neighbors are 4, 5
        non_neighbors = G.get_non_neighbors("2")
        
        assert set(non_neighbors) == {"4", "5"}
        assert "1" not in non_neighbors
        assert "3" not in non_neighbors
        assert "2" not in non_neighbors  # Self excluded
    
    def test_non_neighbors_undirected(self):
        """Test non-neighbors on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_node("D")
        G.add_node("E")
        
        # A connects to B and C, so non-neighbors are D, E
        non_neighbors = G.get_non_neighbors("A")
        
        assert set(non_neighbors) == {"D", "E"}
    
    def test_non_neighbors_fully_connected(self):
        """Test when node is connected to all others."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")
        
        # A is connected to everyone
        non_neighbors = G.get_non_neighbors("A")
        
        assert non_neighbors == []
    
    def test_non_neighbors_isolated(self):
        """Test isolated node (everyone is a non-neighbor)."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        G.add_edge("B", "C")  # Only B-C connected
        
        # A is isolated, so all others are non-neighbors
        non_neighbors = G.get_non_neighbors("A")
        
        assert set(non_neighbors) == {"B", "C", "D"}
    
    def test_non_neighbors_directed_asymmetric(self):
        """Test directed graph with asymmetric connections."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")  # A -> B
        G.add_edge("C", "A")  # C -> A
        G.add_node("D")
        G.add_node("E")
        
        # A connects to B (out) and C (in), mode="all" by default
        non_neighbors = G.get_non_neighbors("A")
        
        assert set(non_neighbors) == {"D", "E"}
        assert "B" not in non_neighbors
        assert "C" not in non_neighbors


class TestGetSimilarNeighbors:
    """Tests for get_similar_neighbors()."""
    
    def test_similar_neighbors_absolute_tolerance(self):
        """Test with absolute tolerance."""
        G = ec.Graph()
        G.add_node("A", value=5.0)
        G.add_node("B", value=5.5)
        G.add_node("C", value=10.0)
        G.add_node("D", value=4.8)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")
        
        # A has value=5.0, tolerance ±1.0
        similar = G.get_similar_neighbors("A", "value", tol_abs=(1.0, 1.0))
        
        # B (5.5) and D (4.8) within ±1.0, C (10.0) outside
        assert set(similar) == {"B", "D"}
        assert "C" not in similar
    
    def test_similar_neighbors_percentage_tolerance(self):
        """Test with percentage tolerance."""
        G = ec.Graph()
        G.add_node("A", value=100.0)
        G.add_node("B", value=105.0)  # +5%
        G.add_node("C", value=95.0)   # -5%
        G.add_node("D", value=120.0)  # +20%
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")
        
        # Tolerance ±10%
        similar = G.get_similar_neighbors("A", "value", tol_pct=(10, 10))
        
        # B and C within ±10%, D outside
        assert set(similar) == {"B", "C"}
        assert "D" not in similar
    
    def test_similar_neighbors_exact_match(self):
        """Test exact match (no tolerance specified)."""
        G = ec.Graph()
        G.add_node("A", score=5)
        G.add_node("B", score=5)
        G.add_node("C", score=6)
        G.add_node("D", score=5)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")
        
        # No tolerance = exact match
        similar = G.get_similar_neighbors("A", "score")
        
        assert set(similar) == {"B", "D"}
        assert "C" not in similar
    
    def test_similar_neighbors_asymmetric_tolerance(self):
        """Test with different upper/lower tolerances."""
        G = ec.Graph()
        G.add_node("A", value=10.0)
        G.add_node("B", value=12.0)  # +2.0
        G.add_node("C", value=8.5)   # -1.5
        G.add_node("D", value=7.0)   # -3.0
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")
        
        # Allow +3.0 above, -2.0 below
        similar = G.get_similar_neighbors("A", "value", tol_abs=(2.0, 3.0))
        
        # B (12.0) within +3.0, C (8.5) within -2.0, D (7.0) outside -2.0
        assert set(similar) == {"B", "C"}
        assert "D" not in similar
    
    def test_similar_neighbors_missing_attribute(self):
        """Test when some neighbors lack the attribute."""
        G = ec.Graph()
        G.add_node("A", value=5.0)
        G.add_node("B", value=5.5)
        G.add_node("C")  # No value attribute
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        
        similar = G.get_similar_neighbors("A", "value", tol_abs=(1.0, 1.0))
        
        # Only B should be included, C ignored
        assert similar == ["B"]
    
    def test_similar_neighbors_non_numeric_error(self):
        """Test error when attribute is not numeric."""
        G = ec.Graph()
        G.add_node("A", category="X")
        G.add_node("B", category="Y")
        G.add_edge("A", "B")
        
        with pytest.raises(ValueError, match="must be numeric"):
            G.get_similar_neighbors("A", "category", tol_abs=(1, 1))
    
    def test_similar_neighbors_directed_graph(self):
        """Test on directed graph."""
        G = ec.Graph(directed=True)
        G.add_node("A", score=10)
        G.add_node("B", score=11)
        G.add_node("C", score=9)
        G.add_edge("A", "B")  # A -> B
        G.add_edge("C", "A")  # C -> A
        
        # By default uses mode="all", so both B and C are neighbors
        similar = G.get_similar_neighbors("A", "score", tol_abs=(2, 2))
        
        assert set(similar) == {"B", "C"}


class TestGetAllConnectedNodes:
    """Tests for get_all_connected_nodes()."""
    
    def test_connected_nodes_simple_component(self):
        """Test on a simple connected component."""
        G = ec.Graph()
        G.add_path_graph(5)  # 1-2-3-4-5
        
        # All nodes 2-5 are connected to 1
        connected = G.get_all_connected_nodes("1")
        
        assert set(connected) == {"2", "3", "4", "5"}
        assert "1" not in connected  # Source excluded
    
    def test_connected_nodes_disconnected_graph(self):
        """Test with disconnected components."""
        G = ec.Graph()
        G.add_path_graph(3)  # 1-2-3
        G.add_node("4")
        G.add_node("5")
        G.add_edge("4", "5")  # Separate component: 4-5
        
        # Only 2 and 3 are connected to 1
        connected1 = G.get_all_connected_nodes("1")
        assert set(connected1) == {"2", "3"}
        
        # Only 5 is connected to 4
        connected4 = G.get_all_connected_nodes("4")
        assert set(connected4) == {"5"}
    
    def test_connected_nodes_isolated(self):
        """Test isolated node."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("B", "C")
        
        # A is isolated, no connected nodes
        connected = G.get_all_connected_nodes("A")
        
        assert connected == []
    
    def test_connected_nodes_directed_as_undirected(self):
        """Test that directed graph is treated as undirected."""
        G = ec.Graph(directed=True)
        G.add_edge("1", "2")  # 1 -> 2
        G.add_edge("2", "3")  # 2 -> 3
        
        # Even though edges are directed, BFS treats as undirected
        connected_from_3 = G.get_all_connected_nodes("3")
        
        # Should find both 1 and 2 despite direction
        assert set(connected_from_3) == {"1", "2"}
    
    def test_connected_nodes_cyclic(self):
        """Test on graph with cycles."""
        G = ec.Graph()
        # Create cycle: A-B-C-D-A
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        G.add_edge("D", "A")
        
        # All others are connected to A
        connected = G.get_all_connected_nodes("A")
        
        assert set(connected) == {"B", "C", "D"}
    
    def test_connected_nodes_star_graph(self):
        """Test on star graph."""
        G = ec.Graph()
        add_star(G, 5)  # Center node "1" connected to 2-5
        
        # From center, all spokes are connected
        connected_center = G.get_all_connected_nodes("1")
        assert set(connected_center) == {"2", "3", "4", "5"}
        
        # From spoke, only center is connected (spokes not interconnected)
        connected_spoke = G.get_all_connected_nodes("2")
        assert set(connected_spoke) == {"1", "3", "4", "5"}
    
    def test_connected_nodes_large_component(self):
        """Test on larger connected component."""
        G = ec.Graph()
        # Create a grid-like structure
        for i in range(1, 11):
            for j in range(1, 11):
                node = f"{i}_{j}"
                G.add_node(node)
                if i > 1:
                    G.add_edge(f"{i-1}_{j}", node)
                if j > 1:
                    G.add_edge(f"{i}_{j-1}", node)
        
        # All 100 nodes are connected
        connected = G.get_all_connected_nodes("1_1")
        
        assert len(connected) == 99  # All except source
        assert "1_1" not in connected


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
