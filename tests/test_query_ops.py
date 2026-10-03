import pytest
import polars as pl
import edgecase as ec


class TestGetNodeAttrs:
    """Tests for get_node_attrs()."""

    def test_get_node_attrs_basic(self):
        """Test extracting node attributes."""
        G = ec.Graph()
        G.add_node("A", type="user", score=95)
        G.add_node("B", type="admin", score=88)
        G.add_node("C", type="user", score=92)

        scores = G.get_node_attrs("score")

        # Returns ID-keyed dict: {1: 95, 2: 88, 3: 92}
        assert scores == {1: 95, 2: 88, 3: 92}

    def test_get_node_attrs_specific_nodes(self):
        """Test extracting attrs from specific nodes."""
        G = ec.Graph()
        G.add_node("A", score=95)
        G.add_node("B", score=88)
        G.add_node("C", score=92)

        scores = G.get_node_attrs("score", nodes=[1, 3])

        # Returns ID-keyed dict
        assert scores == {1: 95, 3: 92}
        assert 2 not in scores

    def test_get_node_attrs_missing_attr(self):
        """Test when some nodes don't have the attribute."""
        G = ec.Graph()
        G.add_node("A", score=95)
        G.add_node("B")  # No score
        G.add_node("C", score=92)

        scores = G.get_node_attrs("score")

        # Returns ID-keyed dict, node 2 (B) has no score
        assert scores == {1: 95, 3: 92}
        assert 2 not in scores

    def test_get_node_attrs_empty_graph(self):
        """Test on empty graph."""
        G = ec.Graph()

        scores = G.get_node_attrs("score")

        assert scores == {}

    def test_get_node_attrs_accepts_labels(self):
        """Test that get_node_attrs accepts both IDs and labels."""
        G = ec.Graph()
        G.add_node("A", score=95)
        G.add_node("B", score=88)
        G.add_node("C", score=92)

        # Using node IDs
        scores_by_id = G.get_node_attrs("score", nodes=[1, 3])
        assert scores_by_id == {1: 95, 3: 92}

        # Using node labels
        scores_by_label = G.get_node_attrs("score", nodes=["A", "C"])
        assert scores_by_label == {1: 95, 3: 92}

        # Should return same results (both keyed by IDs)
        assert scores_by_id == scores_by_label

        # Mixed usage
        scores_mixed = G.get_node_attrs("score", nodes=[1, "C"])
        assert scores_mixed == {1: 95, 3: 92}


class TestSetNodeAttrs:
    """Tests for set_node_attrs()."""

    def test_set_node_attrs_accepts_labels_list(self):
        """Test that set_node_attrs accepts labels in the nodes parameter."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        # Using node IDs
        G.set_node_attrs("color", "blue", nodes=[1, 2])
        colors_by_id = G.get_node_attrs("color", nodes=[1, 2])
        assert colors_by_id == {1: "blue", 2: "blue"}

        # Try with labels
        G.set_node_attrs("color", "red", nodes=["A", "B"])
        colors_by_label = G.get_node_attrs("color", nodes=["A", "B"])
        assert colors_by_label == {1: "red", 2: "red"}

        # Mixed usage
        G.set_node_attrs("status", "active", nodes=[1, "C"])
        status = G.get_node_attrs("status")
        assert status == {1: "active", 3: "active"}

    def test_set_node_attrs_accepts_labels_dict(self):
        """Test that set_node_attrs accepts labels in the values dict."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        # Using node IDs in dict
        G.set_node_attrs("score", {1: 100, 2: 85, 3: 92})
        scores_by_id = G.get_node_attrs("score")
        assert scores_by_id == {1: 100, 2: 85, 3: 92}

        # Using labels in dict
        G.set_node_attrs("grade", {"A": "A+", "B": "B", "C": "A"})
        grades_by_label = G.get_node_attrs("grade")
        assert grades_by_label == {1: "A+", 2: "B", 3: "A"}

        # Mixed dict keys
        G.set_node_attrs("category", {1: "primary", "C": "secondary"})
        categories = G.get_node_attrs("category")
        assert categories == {1: "primary", 3: "secondary"}


class TestGetEdgeAttrs:
    """Tests for get_edge_attrs()."""

    def test_get_edge_attrs_basic(self):
        """Test extracting edge attributes."""
        G = ec.Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=1.5)
        G.add_edge("B", "C", weight=2.3)
        G.add_edge("C", "D", weight=0.8)

        weights = G.get_edge_attrs("weight")

        # Returns dict keyed by edge IDs
        edge_ids = list(G.get_edge_ids())
        assert len(weights) == 3
        assert weights[edge_ids[0]] == 1.5
        assert weights[edge_ids[1]] == 2.3
        assert weights[edge_ids[2]] == 0.8

    def test_get_edge_attrs_specific_edges(self):
        """Test extracting attrs from specific edges (if supported)."""
        G = ec.Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=1.5)
        G.add_edge("B", "C", weight=2.3)
        G.add_edge("C", "D", weight=0.8)

        # get_edge_attrs returns all edges, no filtering by specific edges
        weights = G.get_edge_attrs("weight")

        # Can filter manually after retrieval using edge IDs
        edge_ids = list(G.get_edge_ids())
        filtered_weights = {k: v for k, v in weights.items() if k in [edge_ids[0], edge_ids[2]]}
        assert len(filtered_weights) == 2
        assert filtered_weights[edge_ids[0]] == 1.5
        assert filtered_weights[edge_ids[2]] == 0.8

    @staticmethod
    def _directed_diamond():
        G_ = ec.Graph(directed=True)
        for label in "ABCD":
            G_.add_node(label=label)
        for u, v in [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D")]:
            G_.add_edge(u, v)
        G_.set_edge_attrs("w", 0)
        return G_

    def test_get_edge_attrs_sources_alone_filters(self):
        """sources= on its own keeps only edges leaving those nodes."""
        G_ = self._directed_diamond()
        got = G_.get_edge_attrs("w", sources=["B"])
        assert sorted(G_.get_edge(e).target for e in got) == sorted(
            [G_.get_id_from_label("C"), G_.get_id_from_label("D")]
        )

    def test_get_edge_attrs_targets_alone_filters(self):
        """targets= on its own keeps only edges entering those nodes."""
        G_ = self._directed_diamond()
        got = G_.get_edge_attrs("w", targets=["D"])
        assert sorted(G_.get_edge(e).source for e in got) == sorted(
            [G_.get_id_from_label("B"), G_.get_id_from_label("C")]
        )


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

    def test_get_node_df_always_includes_degree(self):
        """Test DataFrame always includes degree column."""
        G = ec.Graph()
        G.add_node("A", score=95)
        G.add_node("B", score=88)

        df = G.get_node_df()

        assert "degree" in df.columns
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
        G.add_path_graph(5)

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
        for label in ["A", "B", "C"]:
            G.add_node(label)
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

    def test_get_edge_df_always_includes_endpoints(self):
        """Test DataFrame always includes source/target columns."""
        G = ec.Graph()
        for label in ["A", "B"]:
            G.add_node(label)
        G.add_edge("A", "B", weight=1.5)

        df = G.get_edge_df()

        assert "source" in df.columns
        assert "target" in df.columns
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
        G.add_path_graph(5)

        # Select specific nodes (by integer IDs)
        G.select_nodes_by_id([1, 3, 5])
        df = G.get_node_df_ws()

        assert len(df) == 3
        assert set(df["id"].to_list()) == {1, 3, 5}

    def test_get_edge_df_ws(self):
        """Test getting selected edges as DataFrame."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Select specific edges
        G.select_edges_by_id([1, 3])
        df = G.get_edge_df_ws()

        assert len(df) == 2
        assert set(df["id"].to_list()) == {1, 3}

    def test_get_node_df_ws_empty_selection(self):
        """Test with no selection."""
        G = ec.Graph()
        G.add_path_graph(5)

        df = G.get_node_df_ws()

        assert len(df) == 0

    def test_get_df_ws_workflow(self):
        """Test realistic workflow with selection and DataFrame."""
        G = ec.Graph(directed=False)
        G.add_star_graph(10)

        # Select high-degree nodes
        G.select_nodes_by_degree(min_degree=5)
        df = G.get_node_df_ws()

        # Should only get the center node
        assert len(df) == 1
        assert df["degree"].to_list()[0] == 9


class TestGetNeighbors:
    """Tests for get_neighbors()."""

    def test_get_neighbors_accepts_node_id(self):
        """Test that get_neighbors accepts node ID (int) as input."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("A", "C")

        # Using node ID (int)
        neighbors = G.neighbors(1)  # A's ID is 1

        # Should return node IDs
        assert set(neighbors) == {2, 3}
        assert isinstance(neighbors[0], int)

    def test_get_neighbors_accepts_node_label(self):
        """Test that get_neighbors accepts node label (str) as input."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("A", "B")
        G.add_edge("A", "C")

        # Using node label (str)
        neighbors = G.neighbors("A")

        # Should return node IDs (not labels)
        assert set(neighbors) == {2, 3}
        assert isinstance(neighbors[0], int)

        # Verify we get the same results with ID vs label
        neighbors_by_id = G.neighbors(1)
        assert set(neighbors) == set(neighbors_by_id)

    def test_get_neighbors_directed_out(self):
        """Test getting outgoing neighbors."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("D", "A")

        # get_neighbors expects node_id (1 = A)
        neighbors = G.neighbors(1, mode="out")

        # Returns list of node IDs: [2, 3] (B, C)
        assert set(neighbors) == {2, 3}
        assert 4 not in neighbors

    def test_get_neighbors_directed_in(self):
        """Test getting incoming neighbors."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("D", "A")

        neighbors = G.neighbors(1, mode="in")

        # Returns list of node IDs: [4] (D)
        assert set(neighbors) == {4}
        assert 2 not in neighbors
        assert 3 not in neighbors

    def test_get_neighbors_directed_all(self):
        """Test getting all neighbors."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("D", "A")

        neighbors = G.neighbors(1, mode="all")

        # Returns list of node IDs: [2, 3, 4] (B, C, D)
        assert set(neighbors) == {2, 3, 4}

    def test_get_neighbors_undirected(self):
        """Test on undirected graph."""
        G = ec.Graph(directed=False)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("D", "A")

        neighbors = G.neighbors(1)

        # Returns list of node IDs: [2, 3, 4] (B, C, D)
        assert set(neighbors) == {2, 3, 4}

    def test_get_neighbors_with_filtering(self):
        """Test filtering neighbors using list comprehension."""
        G = ec.Graph(directed=False)
        G.add_node("A")
        G.add_node("B", type="user")
        G.add_node("C", type="admin")
        G.add_node("D", type="user")
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")

        # Filter for user type nodes using list comprehension
        def is_user(node_id):
            node = G.get_node(node_id)
            return node and node.attributes and node.attributes.get("type") == "user"

        neighbors = [n for n in G.neighbors(1) if is_user(n)]

        # Returns list of node IDs: [2, 4] (B, D)
        assert set(neighbors) == {2, 4}
        assert 3 not in neighbors


class TestGetCommonNeighbors:
    """Tests for get_common_neighbors()."""

    def test_get_common_neighbors_accepts_labels(self):
        """Test that get_common_neighbors accepts both IDs and labels."""
        G = ec.Graph(directed=False)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("A", "D")
        G.add_edge("B", "D")

        # Using node IDs
        common_by_ids = G.get_common_neighbors(1, 2)  # A=1, B=2
        assert set(common_by_ids) == {3, 4}

        # Using node labels
        common_by_labels = G.get_common_neighbors("A", "B")
        assert set(common_by_labels) == {3, 4}

        # Mixed: ID and label
        common_mixed = G.get_common_neighbors(1, "B")
        assert set(common_mixed) == {3, 4}

        # Should all return same results (node IDs)
        assert set(common_by_ids) == set(common_by_labels) == set(common_mixed)
        assert isinstance(common_by_labels[0], int)

    def test_get_common_neighbors_basic(self):
        """Test finding common neighbors."""
        G = ec.Graph(directed=False)
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("A", "D")
        G.add_edge("B", "D")
        G.add_edge("A", "E")  # Only A connects to E

        common = G.get_common_neighbors(1, 2)  # A=1, B=2

        # Returns list of node IDs: [3, 4] (C, D)
        assert set(common) == {3, 4}
        assert 5 not in common

    def test_get_common_neighbors_none(self):
        """Test when there are no common neighbors."""
        G = ec.Graph(directed=False)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "C")
        G.add_edge("B", "D")

        common = G.get_common_neighbors(1, 2)  # A=1, B=2

        assert common == []


class TestGetPredecessorsSuccessors:
    """Tests for get_predecessors() and get_successors()."""

    def test_get_predecessors_accepts_label(self):
        """Test that get_predecessors accepts both ID and label."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "C")
        G.add_edge("B", "C")

        # Using node ID
        predecessors_by_id = G.get_predecessors(3)  # C=3
        assert set(predecessors_by_id) == {1, 2}

        # Using node label
        predecessors_by_label = G.get_predecessors("C")
        assert set(predecessors_by_label) == {1, 2}

        # Should return same results
        assert set(predecessors_by_id) == set(predecessors_by_label)
        assert isinstance(predecessors_by_label[0], int)

    def test_get_successors_accepts_label(self):
        """Test that get_successors accepts both ID and label."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")

        # Using node ID
        successors_by_id = G.get_successors(1)  # A=1
        assert set(successors_by_id) == {2, 3}

        # Using node label
        successors_by_label = G.get_successors("A")
        assert set(successors_by_label) == {2, 3}

        # Should return same results
        assert set(successors_by_id) == set(successors_by_label)
        assert isinstance(successors_by_label[0], int)

    def test_get_predecessors(self):
        """Test getting predecessors."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        G.add_edge("C", "D")

        predecessors = G.get_predecessors(3)  # C=3

        # Returns list of node IDs: [1, 2] (A, B)
        assert set(predecessors) == {1, 2}

    def test_get_successors(self):
        """Test getting successors."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")

        successors = G.get_successors(1)  # A=1

        # Returns list of node IDs: [2, 3, 4] (B, C, D)
        assert set(successors) == {2, 3, 4}

    def test_predecessors_undirected_error(self):
        """Test that predecessors fails on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_path_graph(5)

        with pytest.raises(ValueError, match="only available for directed"):
            G.get_predecessors("3")

    def test_successors_undirected_error(self):
        """Test that successors fails on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_path_graph(5)

        with pytest.raises(ValueError, match="only available for directed"):
            G.get_successors("3")


class TestGetNonNeighbors:
    """Tests for get_non_neighbors()."""

    def test_non_neighbors_accepts_label(self):
        """Test that get_non_neighbors accepts both ID and label."""
        G = ec.Graph(directed=False)
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")

        # Using node ID
        non_neighbors_by_id = G.get_non_neighbors(1)  # A=1
        assert set(non_neighbors_by_id) == {4, 5}  # D and E

        # Using node label
        non_neighbors_by_label = G.get_non_neighbors("A")
        assert set(non_neighbors_by_label) == {4, 5}

        # Should return same results (node IDs)
        assert set(non_neighbors_by_id) == set(non_neighbors_by_label)
        assert isinstance(non_neighbors_by_label[0], int)

    def test_non_neighbors_path_graph(self):
        """Test non-neighbors on a path graph."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)  # 1 -> 2 -> 3 -> 4 -> 5

        # Node 2 connects to 1 and 3, so non-neighbors are 4, 5
        non_neighbors = G.get_non_neighbors(2)

        # Returns list of node IDs: [4, 5]
        assert set(non_neighbors) == {4, 5}
        assert 1 not in non_neighbors
        assert 3 not in non_neighbors
        assert 2 not in non_neighbors  # Self excluded

    def test_non_neighbors_undirected(self):
        """Test non-neighbors on undirected graph."""
        G = ec.Graph(directed=False)
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")

        # A connects to B and C, so non-neighbors are D, E
        non_neighbors = G.get_non_neighbors(1)  # A=1

        # Returns list of node IDs: [4, 5] (D, E)
        assert set(non_neighbors) == {4, 5}

    def test_non_neighbors_fully_connected(self):
        """Test when node is connected to all others."""
        G = ec.Graph(directed=False)
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")

        # A is connected to everyone
        non_neighbors = G.get_non_neighbors(1)  # A=1

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
        non_neighbors = G.get_non_neighbors(1)  # A=1

        # Returns list of node IDs: [2, 3, 4] (B, C, D)
        assert set(non_neighbors) == {2, 3, 4}

    def test_non_neighbors_directed_asymmetric(self):
        """Test directed graph with asymmetric connections."""
        G = ec.Graph(directed=True)
        for label in ["A", "B", "C", "D", "E"]:
            G.add_node(label)
        G.add_edge("A", "B")  # A -> B
        G.add_edge("C", "A")  # C -> A

        # A connects to B (out) and C (in), mode="all" by default
        non_neighbors = G.get_non_neighbors(1)  # A=1

        # Returns list of node IDs: [4, 5] (D, E)
        assert set(non_neighbors) == {4, 5}
        assert 2 not in non_neighbors
        assert 3 not in non_neighbors


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
        similar = G.get_similar_neighbors(1, "value", tol_abs=(1.0, 1.0))  # A=1

        # B (5.5) and D (4.8) within ±1.0, C (10.0) outside
        # Returns list of node IDs: [2, 4] (B, D)
        assert set(similar) == {2, 4}
        assert 3 not in similar

    def test_similar_neighbors_percentage_tolerance(self):
        """Test with percentage tolerance."""
        G = ec.Graph()
        G.add_node("A", value=100.0)
        G.add_node("B", value=105.0)  # +5%
        G.add_node("C", value=95.0)  # -5%
        G.add_node("D", value=120.0)  # +20%
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")

        # Tolerance ±10%
        similar = G.get_similar_neighbors(1, "value", tol_pct=(10, 10))  # A=1

        # B and C within ±10%, D outside
        # Returns list of node IDs: [2, 3] (B, C)
        assert set(similar) == {2, 3}
        assert 4 not in similar

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
        similar = G.get_similar_neighbors(1, "score")  # A=1

        # Returns list of node IDs: [2, 4] (B, D)
        assert set(similar) == {2, 4}
        assert 3 not in similar

    def test_similar_neighbors_asymmetric_tolerance(self):
        """Test with different upper/lower tolerances."""
        G = ec.Graph()
        G.add_node("A", value=10.0)
        G.add_node("B", value=12.0)  # +2.0
        G.add_node("C", value=8.5)  # -1.5
        G.add_node("D", value=7.0)  # -3.0
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("A", "D")

        # Allow +3.0 above, -2.0 below
        similar = G.get_similar_neighbors(1, "value", tol_abs=(2.0, 3.0))  # A=1

        # B (12.0) within +3.0, C (8.5) within -2.0, D (7.0) outside -2.0
        # Returns list of node IDs: [2, 3] (B, C)
        assert set(similar) == {2, 3}
        assert 4 not in similar

    def test_similar_neighbors_missing_attribute(self):
        """Test when some neighbors lack the attribute."""
        G = ec.Graph()
        G.add_node("A", value=5.0)
        G.add_node("B", value=5.5)
        G.add_node("C")  # No value attribute
        G.add_edge("A", "B")
        G.add_edge("A", "C")

        similar = G.get_similar_neighbors(1, "value", tol_abs=(1.0, 1.0))  # A=1

        # Only B should be included, C ignored
        # Returns list of node IDs: [2] (B)
        assert similar == [2]

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
        similar = G.get_similar_neighbors(1, "score", tol_abs=(2, 2))  # A=1

        # Returns list of node IDs: [2, 3] (B, C)
        assert set(similar) == {2, 3}


class TestGetAllConnectedNodes:
    """Tests for get_all_connected_nodes()."""

    def test_connected_nodes_accepts_label(self):
        """Test that get_all_connected_nodes accepts both ID and label."""
        G = ec.Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")

        # Using node ID
        connected_by_id = G.get_all_connected_nodes(1)  # A=1
        assert set(connected_by_id) == {2, 3, 4}  # B, C, D

        # Using node label
        connected_by_label = G.get_all_connected_nodes("A")
        assert set(connected_by_label) == {2, 3, 4}

        # Should return same results (node IDs)
        assert set(connected_by_id) == set(connected_by_label)
        assert isinstance(connected_by_label[0], int)

    def test_connected_nodes_simple_component(self):
        """Test on a simple connected component."""
        G = ec.Graph()
        G.add_path_graph(5)  # 1-2-3-4-5

        # All nodes 2-5 are connected to 1
        connected = G.get_all_connected_nodes(1)

        # Returns list of node IDs: [2, 3, 4, 5]
        assert set(connected) == {2, 3, 4, 5}
        assert 1 not in connected  # Source excluded

    def test_connected_nodes_disconnected_graph(self):
        """Test with disconnected components."""
        G = ec.Graph()
        G.add_path_graph(3)  # 1-2-3
        G.add_node("4")
        G.add_node("5")
        G.add_edge("4", "5")  # Separate component: 4-5

        # Only 2 and 3 are connected to 1
        connected1 = G.get_all_connected_nodes(1)
        assert set(connected1) == {2, 3}

        # Only 5 is connected to 4
        connected4 = G.get_all_connected_nodes(4)
        assert set(connected4) == {5}

    def test_connected_nodes_isolated(self):
        """Test isolated node."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_edge("B", "C")

        # A is isolated, no connected nodes
        connected = G.get_all_connected_nodes(1)  # A=1

        assert connected == []

    def test_connected_nodes_directed_as_undirected(self):
        """Test that directed graph is treated as undirected."""
        G = ec.Graph(directed=True)
        for label in ["1", "2", "3"]:
            G.add_node(label)
        G.add_edge("1", "2")  # 1 -> 2
        G.add_edge("2", "3")  # 2 -> 3

        # Even though edges are directed, BFS treats as undirected
        connected_from_3 = G.get_all_connected_nodes(3)

        # Should find both 1 and 2 despite direction
        # Returns list of node IDs: [1, 2]
        assert set(connected_from_3) == {1, 2}

    def test_connected_nodes_cyclic(self):
        """Test on graph with cycles."""
        G = ec.Graph()
        for label in ["A", "B", "C", "D"]:
            G.add_node(label)
        # Create cycle: A-B-C-D-A
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        G.add_edge("D", "A")

        # All others are connected to A
        connected = G.get_all_connected_nodes(1)  # A=1

        # Returns list of node IDs: [2, 3, 4] (B, C, D)
        assert set(connected) == {2, 3, 4}

    def test_connected_nodes_star_graph(self):
        """Test on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)  # Center node 1 connected to 2-5

        # From center, all spokes are connected
        connected_center = G.get_all_connected_nodes(1)
        assert set(connected_center) == {2, 3, 4, 5}

        # From spoke, all others are connected (including center and other spokes)
        connected_spoke = G.get_all_connected_nodes(2)
        assert set(connected_spoke) == {1, 3, 4, 5}

    def test_connected_nodes_large_component(self):
        """Test on larger connected component."""
        G = ec.Graph()
        # Create a grid-like structure
        for i in range(1, 11):
            for j in range(1, 11):
                node = f"{i}_{j}"
                G.add_node(node)
                if i > 1:
                    G.add_edge(f"{i - 1}_{j}", node)
                if j > 1:
                    G.add_edge(f"{i}_{j - 1}", node)

        # All 100 nodes are connected
        connected = G.get_all_connected_nodes(1)  # Node with label "1_1" = ID 1

        assert len(connected) == 99  # All except source
        assert 1 not in connected


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
        G.add_star_graph(10)

        # Select high-degree nodes and their neighbors
        G.select_nodes_by_degree(min_degree=5)
        G.select_nodes_in_neighborhood(G.get_selected_node_ids(), k=1, mode="out")

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

        # Get all values (returns ID-keyed dict)
        values = G.get_node_attrs("value")
        assert sum(values.values()) == 375

        # Get only high priority node values
        high_priority_nodes = [
            nid
            for nid in G.get_node_ids(limit=None)
            if G.get_node(nid).attributes.get("priority") == "high"
        ]
        high_values = G.get_node_attrs("value", nodes=high_priority_nodes)

        # Returns ID-keyed dict: {1: 100, 3: 150} (A, C)
        assert set(high_values.keys()) == {1, 3}
        assert sum(high_values.values()) == 250
