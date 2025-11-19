import pytest
import polars as pl
import edgecase as ec


class TestNodeDataFrameStructure:
    """Tests for mandatory node DataFrame structure."""

    def test_mandatory_columns_present(self):
        """Test that id, label, and type columns are always present."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user")
        g.add_node()  # No label or type

        df = g.get_node_df()

        # First 3 columns must be id, label, type
        assert list(df.columns[:3]) == ["id", "label", "type"]

    def test_column_order_fixed(self):
        """Test that id, label, type are always first 3 columns in that order."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user", age=30, city="NYC")
        g.add_node(label="Bob", score=100)

        df = g.get_node_df()

        # First 3 columns must always be in this exact order
        assert df.columns[0] == "id"
        assert df.columns[1] == "label"
        assert df.columns[2] == "type"

    def test_degree_column_always_last(self):
        """Test that degree column is always last."""
        g = ec.Graph()
        g.add_node(label="Alice", age=30)
        g.add_node(label="Bob", score=100, city="LA")

        df = g.get_node_df()

        # degree should be the last column
        assert df.columns[-1] == "degree"

    def test_sorted_by_id_ascending(self):
        """Test that rows are always sorted by id in ascending order."""
        g = ec.Graph()
        # Add nodes in random order
        g.add_node(label="Z")  # id=1
        g.add_node(label="A")  # id=2
        g.add_node(label="M")  # id=3
        g.add_node(label="B")  # id=4

        df = g.get_node_df()

        # IDs should be in ascending order
        ids = df["id"].to_list()
        assert ids == sorted(ids)
        assert ids == [1, 2, 3, 4]

    def test_null_label_not_omitted(self):
        """Test that missing labels show as null, not omitted."""
        g = ec.Graph()
        g.add_node(label="Alice")
        g.add_node()  # No label

        df = g.get_node_df()

        # label column must exist and have null for second node
        assert "label" in df.columns
        labels = df["label"].to_list()
        assert labels[0] == "Alice"
        assert labels[1] is None

    def test_null_type_not_omitted(self):
        """Test that missing types show as null, not omitted."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user")
        g.add_node(label="Bob")  # No type

        df = g.get_node_df()

        # type column must exist and have null for second node
        assert "type" in df.columns
        types = df["type"].to_list()
        assert types[0] == "user"
        assert types[1] is None

    def test_empty_graph_structure(self):
        """Test that empty graph returns empty DataFrame."""
        g = ec.Graph()
        df = g.get_node_df()

        # Empty graph returns empty DataFrame with no columns
        assert len(df) == 0
        # When nodes are added, structure will be correct
        g.add_node()
        df = g.get_node_df()
        assert list(df.columns[:3]) == ["id", "label", "type"]


class TestAddNodeWithType:
    """Tests for add_node() method with type parameter."""

    def test_add_node_with_type_parameter(self):
        """Test that add_node accepts type as a parameter."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user", age=30)

        df = g.get_node_df()

        assert df["type"][0] == "user"
        assert df["label"][0] == "Alice"
        assert df["age"][0] == 30

    def test_add_node_type_appears_in_third_column(self):
        """Test that type appears in the third column, not mixed with other attrs."""
        g = ec.Graph()
        g.add_node(label="Alice", type="admin", age=25, score=95)

        df = g.get_node_df()

        # Type must be in column 2 (0-indexed)
        assert df.columns[2] == "type"
        assert df["type"][0] == "admin"

    def test_add_node_without_type(self):
        """Test that add_node without type still creates type column with null."""
        g = ec.Graph()
        g.add_node(label="Alice", age=30)

        df = g.get_node_df()

        # type column should exist but be null
        assert "type" in df.columns
        assert df["type"][0] is None

    def test_mixed_nodes_with_and_without_type(self):
        """Test graph with some nodes having type and others not."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user")
        g.add_node(label="Bob")
        g.add_node(label="Charlie", type="admin")

        df = g.get_node_df()

        types = df["type"].to_list()
        assert types[0] == "user"
        assert types[1] is None
        assert types[2] == "admin"


class TestDataFrameInitialization:
    """Tests for initializing graphs from DataFrames with type column."""

    def test_init_with_type_column(self):
        """Test that DataFrame initialization recognizes type as first-class."""
        nodes_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "label": ["Alice", "Bob", "Charlie"],
                "type": ["user", "admin", "user"],
                "age": [30, 25, 35],
            }
        )

        g = ec.Graph(nodes_df=nodes_df)
        df = g.get_node_df()

        # Type should be preserved and in third column
        assert df.columns[2] == "type"
        assert df["type"].to_list() == ["user", "admin", "user"]

    def test_init_without_type_column(self):
        """Test DataFrame init without type column creates null type column."""
        nodes_df = pl.DataFrame({"id": [1, 2], "label": ["Alice", "Bob"], "age": [30, 25]})

        g = ec.Graph(nodes_df=nodes_df)
        df = g.get_node_df()

        # type column should exist but be all nulls
        assert "type" in df.columns
        assert all(t is None for t in df["type"].to_list())

    def test_init_with_mixed_types(self):
        """Test DataFrame init with some null types."""
        nodes_df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "label": ["Alice", "Bob", "Charlie"],
                "type": ["user", None, "admin"],
            }
        )

        g = ec.Graph(nodes_df=nodes_df)
        df = g.get_node_df()

        types = df["type"].to_list()
        assert types[0] == "user"
        assert types[1] is None
        assert types[2] == "admin"

    def test_init_preserves_id_sorting(self):
        """Test that DataFrame init produces sorted output regardless of input order."""
        nodes_df = pl.DataFrame(
            {"id": [30, 10, 20], "label": ["C", "A", "B"], "type": ["user", "admin", "user"]}
        )

        g = ec.Graph(nodes_df=nodes_df)
        df = g.get_node_df()

        # Note: EdgeCase reassigns IDs sequentially (1, 2, 3)
        # But output is still sorted by id
        ids = df["id"].to_list()
        assert ids == sorted(ids)
        # Labels should maintain their order from input
        assert df["label"].to_list() == ["C", "A", "B"]


class TestGetNodeDfWs:
    """Tests for get_node_df_ws() method with workspace selection."""

    def test_get_node_df_ws_maintains_structure(self):
        """Test that get_node_df_ws() maintains the same column structure."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user", age=30)
        g.add_node(label="Bob", type="admin", age=25)
        g.add_node(label="Charlie", age=35)

        # Select subset of nodes by ID
        node_ids = [g.get_id_from_label("Alice"), g.get_id_from_label("Charlie")]
        g.select_nodes_by_id(node_ids)
        df = g.get_node_df_ws()

        # Should have same structure: id, label, type first
        assert list(df.columns[:3]) == ["id", "label", "type"]
        assert df.columns[-1] == "degree"

    def test_get_node_df_ws_sorted_by_id(self):
        """Test that get_node_df_ws() returns sorted results."""
        g = ec.Graph()
        for i in range(10, 0, -1):  # Add in reverse order
            g.add_node(label=f"Node{i}", type="user")

        # Select all nodes using lambda
        g.select_nodes(lambda n: n.id > 0)
        df = g.get_node_df_ws()

        # Should be sorted
        ids = df["id"].to_list()
        assert ids == sorted(ids)

    def test_get_node_df_ws_preserves_type_column(self):
        """Test that get_node_df_ws() includes type column even if some nulls."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user")
        g.add_node(label="Bob")  # No type
        g.add_node(label="Charlie", type="admin")

        # Select all using lambda
        g.select_nodes(lambda n: n.id > 0)
        df = g.get_node_df_ws()

        assert "type" in df.columns
        assert df.columns[2] == "type"
        types = df["type"].to_list()
        assert types[0] == "user"
        assert types[1] is None
        assert types[2] == "admin"


class TestAttributeOrdering:
    """Tests for attribute column ordering after id, label, type."""

    def test_other_attributes_sorted_alphabetically(self):
        """Test that non-mandatory attributes appear sorted alphabetically."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user", zebra=1, apple=2, mango=3)

        df = g.get_node_df()

        # After id, label, type, attrs should be alphabetical, then degree
        # Expected: id, label, type, apple, mango, zebra, degree
        columns = df.columns
        assert columns[0] == "id"
        assert columns[1] == "label"
        assert columns[2] == "type"

        # Find attributes between type and degree
        attr_columns = [c for c in columns[3:-1]]
        assert attr_columns == sorted(attr_columns)

    def test_type_not_duplicated_in_attributes(self):
        """Test that type doesn't appear twice in DataFrame."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user", age=30)

        df = g.get_node_df()
        columns = list(df.columns)

        # type should appear exactly once
        assert columns.count("type") == 1
        assert columns[2] == "type"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_node_with_only_type_no_label(self):
        """Test node with type but no label."""
        g = ec.Graph()
        g.add_node(type="user", age=30)

        df = g.get_node_df()

        assert df["label"][0] is None
        assert df["type"][0] == "user"

    def test_large_number_of_nodes_maintains_structure(self):
        """Test that structure is maintained with many nodes."""
        g = ec.Graph()
        for i in range(1000):
            g.add_node(label=f"Node{i}", type="user" if i % 2 == 0 else None)

        df = g.get_node_df()

        # Structure should still be correct
        assert list(df.columns[:3]) == ["id", "label", "type"]
        assert df.columns[-1] == "degree"
        assert len(df) == 1000

        # Should be sorted
        ids = df["id"].to_list()
        assert ids == sorted(ids)

    def test_multiple_dataframe_calls_consistent(self):
        """Test that multiple calls to get_node_df() return consistent structure."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user")
        g.add_node(label="Bob", age=30)

        df1 = g.get_node_df()
        df2 = g.get_node_df()

        # Should be identical
        assert df1.columns == df2.columns
        assert df1.shape == df2.shape
        assert df1["id"].to_list() == df2["id"].to_list()

    def test_node_update_maintains_structure(self):
        """Test that adding nodes after getting DataFrame maintains structure."""
        g = ec.Graph()
        g.add_node(label="Alice", type="user")

        df1 = g.get_node_df()
        first_cols_1 = list(df1.columns[:3])

        # Add more nodes
        g.add_node(label="Bob", type="admin")
        g.add_node(label="Charlie")

        df2 = g.get_node_df()
        first_cols_2 = list(df2.columns[:3])

        # Structure should remain the same
        assert first_cols_1 == first_cols_2 == ["id", "label", "type"]
        assert df2.columns[-1] == "degree"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_add_node_without_type_still_works(self):
        """Test that old code without type parameter still works."""
        g = ec.Graph()
        # Old API style - no type parameter
        g.add_node(label="Alice", age=30, city="NYC")

        df = g.get_node_df()

        # Should work fine, just have null type
        assert df["label"][0] == "Alice"
        assert df["type"][0] is None
        assert df["age"][0] == 30

    def test_dataframe_without_type_still_works(self):
        """Test that DataFrames without type column still work."""
        nodes_df = pl.DataFrame({"id": [1, 2], "label": ["Alice", "Bob"], "age": [30, 25]})

        g = ec.Graph(nodes_df=nodes_df)
        df = g.get_node_df()

        # Should work, with null types
        assert "type" in df.columns
        assert all(t is None for t in df["type"].to_list())

    def test_mixed_old_and_new_api(self):
        """Test that mixing old API (no type) and new API (with type) works."""
        g = ec.Graph()

        # Old style
        g.add_node(label="Alice", age=30)

        # New style
        g.add_node(label="Bob", type="admin", age=25)

        df = g.get_node_df()

        types = df["type"].to_list()
        assert types[0] is None  # Old style
        assert types[1] == "admin"  # New style
