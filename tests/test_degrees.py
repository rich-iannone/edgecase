import pytest
import edgecase as ec


class TestDegreesBasic:
    """Basic tests for degrees() batch method."""

    def test_degrees_empty_graph(self):
        """Empty graph returns empty dict."""
        g = ec.Graph()
        result = g.degrees(return_dict=True)
        assert result == {}

    def test_degrees_single_node(self):
        """Single isolated node has degree 0."""
        g = ec.Graph()
        g.add_node(label="A")

        result = g.degrees(return_dict=True)
        a_id = g.get_id_from_label("A")
        assert result[a_id] == 0

    def test_degrees_simple_edge(self):
        """Two connected nodes each have degree 1."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        result = g.degrees(return_dict=True)
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")

        assert result[a_id] == 1
        assert result[b_id] == 1

    def test_degrees_star_graph(self):
        """Star graph: center has high degree, leaves have degree 1."""
        g = ec.Graph()
        g.add_node(label="center")
        for i in range(5):
            g.add_node(label=f"leaf{i}")
            g.add_edge("center", f"leaf{i}")

        result = g.degrees(return_dict=True)
        center_id = g.get_id_from_label("center")

        assert result[center_id] == 5
        for i in range(5):
            leaf_id = g.get_id_from_label(f"leaf{i}")
            assert result[leaf_id] == 1

    def test_degrees_complete_graph(self):
        """Complete graph: all nodes have degree n-1."""
        g = ec.Graph()
        n = 5
        for i in range(n):
            g.add_node(label=str(i))

        # Add all edges
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(str(i), str(j))

        result = g.degrees(return_dict=True)

        # All nodes should have degree n-1
        for i in range(n):
            node_id = g.get_id_from_label(str(i))
            assert result[node_id] == n - 1


class TestDegreesDirected:
    """Tests for degrees() on directed graphs."""

    def test_degrees_directed_total(self):
        """Directed graph degrees are in-degree + out-degree."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")

        # A -> B, A -> C, B -> C
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "C")

        result = g.degrees(return_dict=True)
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")

        # A: out=2, in=0, total=2
        assert result[a_id] == 2
        # B: out=1, in=1, total=2
        assert result[b_id] == 2
        # C: out=0, in=2, total=2
        assert result[c_id] == 2

    def test_degrees_directed_mutual(self):
        """Mutual directed edges count both ways."""
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")

        # Mutual connection
        g.add_edge("A", "B")
        g.add_edge("B", "A")

        result = g.degrees(return_dict=True)
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")

        # Both have in=1, out=1, total=2
        assert result[a_id] == 2
        assert result[b_id] == 2


class TestDegreesReturnDict:
    """Tests for return_dict parameter behavior."""

    def test_return_dict_true(self):
        """return_dict=True returns dictionary of node IDs to degrees."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        result = g.degrees(return_dict=True)

        assert isinstance(result, dict)
        assert len(result) == 2
        # Keys should be node IDs (integers)
        assert all(isinstance(k, int) for k in result.keys())
        # Values should be integers
        assert all(isinstance(v, int) for v in result.values())

    def test_return_dict_false_default(self):
        """return_dict=False (default) returns self for chaining."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # Test explicit False
        result = g.degrees(return_dict=False)
        assert result is g

        # Test default (should be False)
        g2 = ec.Graph()
        g2.add_node(label="X")
        result2 = g2.degrees()
        assert result2 is g2

    def test_return_dict_false_stores_attributes(self):
        """return_dict=False stores degrees as node attributes."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge("A", "B")
        g.add_edge("A", "C")

        g.degrees(return_dict=False)

        # Check attributes were stored
        attrs = g.get_node_attrs("degree")
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        c_id = g.get_id_from_label("C")

        assert attrs[a_id] == 2
        assert attrs[b_id] == 1
        assert attrs[c_id] == 1

    def test_return_dict_false_default_stores(self):
        """Default behavior (no return_dict) stores degrees as attributes."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # Call without return_dict parameter
        result = g.degrees()
        assert result is g

        # Check attributes were stored with default name
        attrs = g.get_node_attrs("degree")
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")

        assert attrs[a_id] == 1
        assert attrs[b_id] == 1


class TestDegreesAttrName:
    """Tests for attr_name parameter."""

    def test_attr_name_default(self):
        """Default attr_name is 'degree'."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        g.degrees(return_dict=False)

        ndf = g.get_node_df()
        assert "degree" in ndf.columns

    def test_attr_name_custom(self):
        """Custom attr_name creates column with that name."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        g.degrees(attr_name="custom_deg", return_dict=False)

        ndf = g.get_node_df()
        assert "custom_deg" in ndf.columns

        # Verify the custom column has the right values
        a_id = g.get_id_from_label("A")
        b_id = g.get_id_from_label("B")
        attrs = g.get_node_attrs("custom_deg")
        assert attrs[a_id] == 1
        assert attrs[b_id] == 1

    def test_attr_name_overwrites(self):
        """Calling degrees() multiple times overwrites the column."""
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # First call
        g.degrees(return_dict=False)
        attrs1 = g.get_node_attrs("degree")

        # Add more edges
        g.add_node(label="C")
        g.add_edge("A", "C")

        # Second call should update
        g.degrees(return_dict=False)
        attrs2 = g.get_node_attrs("degree")

        # A's degree should have increased
        a_id = g.get_id_from_label("A")
        assert attrs2[a_id] == 2
        assert attrs2[a_id] > attrs1[a_id]


class TestDegreesMethodChaining:
    """Tests for method chaining with degrees()."""

    def test_chain_with_other_methods(self):
        """degrees() can be chained with other methods."""
        g = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("A", "C")
            .degrees()  # Should use default return_dict=False
        )

        assert isinstance(g, ec.Graph)
        ndf = g.get_node_df()
        assert "degree" in ndf.columns

    def test_chain_multiple_computations(self):
        """Chain degrees() with other computation methods."""
        g = ec.karate_club().degrees().pagerank(return_dict=False)

        ndf = g.get_node_df()
        assert "degree" in ndf.columns
        assert "pagerank" in ndf.columns


class TestDegreesConsistency:
    """Tests for consistency with get_degree()."""

    def test_consistency_with_get_degree(self):
        """degrees() should match individual get_degree() calls."""
        g = ec.Graph()
        for i in range(10):
            g.add_node(label=str(i))

        # Add some edges
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        for src, dst in edges:
            g.add_edge(str(src), str(dst))

        batch_degrees = g.degrees(return_dict=True)

        # Verify each matches get_degree()
        for node_id in batch_degrees:
            individual_degree = g.get_degree(node_id)
            assert batch_degrees[node_id] == individual_degree

    def test_consistency_directed_graph(self):
        """degrees() consistency with get_degree() on directed graphs."""
        g = ec.Graph(directed=True)
        for i in range(5):
            g.add_node(label=str(i))

        # Create directed edges
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 0)]
        for src, dst in edges:
            g.add_edge(str(src), str(dst))

        batch_degrees = g.degrees(return_dict=True)

        # Verify each matches get_degree()
        for node_id in batch_degrees:
            individual_degree = g.get_degree(node_id)
            assert batch_degrees[node_id] == individual_degree


class TestDegreesRealWorld:
    """Tests with real-world graphs."""

    def test_karate_club(self):
        """Test degrees() on karate club graph."""
        g = ec.karate_club()
        result = g.degrees(return_dict=True)

        # Karate club has 33 nodes
        assert len(result) == 33

        # All degrees should be positive
        assert all(d > 0 for d in result.values())

        # Check a few known properties
        # Node 1 (Mr. Hi) should have one of the highest degrees
        degrees_list = list(result.values())
        max_degree = max(degrees_list)
        assert max_degree > 10  # Mr. Hi has degree 25

    def test_karate_club_chaining(self):
        """Test degrees() in a pipeline with karate club."""
        g = ec.karate_club().degrees()

        ndf = g.get_node_df()
        assert "degree" in ndf.columns
        assert len(ndf) == 33

        # Verify degrees were computed
        degree_col = ndf["degree"]
        assert all(d > 0 for d in degree_col)
