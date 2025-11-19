import pytest
import edgecase as ec


class TestAddEdgeSignature:
    """Test the add_edge() method signature with rel before weight."""

    def test_basic_edge_no_params(self):
        """Test basic edge with no rel or weight."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        df = g.get_edge_df()
        assert len(df) == 1
        assert df["source"][0] == 1
        assert df["target"][0] == 2
        assert df["rel"][0] is None
        assert "weight" not in df.columns

    def test_rel_only_positional(self):
        """Test edge with rel as positional argument."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "friend")  # rel as 3rd positional arg

        df = g.get_edge_df()
        assert df["rel"][0] == "friend"
        assert "weight" not in df.columns

    def test_rel_only_keyword(self):
        """Test edge with rel as keyword argument."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", rel="colleague")

        df = g.get_edge_df()
        assert df["rel"][0] == "colleague"
        assert "weight" not in df.columns

    def test_weight_only_positional(self):
        """Test edge with weight as positional argument (4th position)."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", None, 5.5)  # rel=None, weight=5.5

        edge_id = g.get_edge_ids()[0]
        edge = g.get_edge(edge_id)
        assert edge.weight == 5.5
        df = g.get_edge_df()
        assert df["rel"][0] is None

    def test_weight_only_keyword(self):
        """Test edge with weight as keyword argument."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", weight=3.7)

        edge_id = g.get_edge_ids()[0]
        edge = g.get_edge(edge_id)
        assert edge.weight == 3.7
        df = g.get_edge_df()
        assert df["rel"][0] is None

    def test_both_rel_and_weight_positional(self):
        """Test edge with both rel and weight as positional arguments."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "mentor", 10.0)

        df = g.get_edge_df()
        assert df["rel"][0] == "mentor"
        assert df["weight"][0] == 10.0

    def test_both_rel_and_weight_keyword(self):
        """Test edge with both rel and weight as keyword arguments."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", rel="collaborator", weight=7.2)

        df = g.get_edge_df()
        assert df["rel"][0] == "collaborator"
        assert df["weight"][0] == 7.2

    def test_both_rel_and_weight_mixed(self):
        """Test edge with rel positional and weight keyword."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "partner", weight=8.5)

        df = g.get_edge_df()
        assert df["rel"][0] == "partner"
        assert df["weight"][0] == 8.5

    def test_rel_with_additional_attributes_positional(self):
        """Test edge with rel (positional) and additional attributes."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "friend", since=2020, strength="strong")

        df = g.get_edge_df()
        assert df["rel"][0] == "friend"
        assert df["since"][0] == 2020
        assert df["strength"][0] == "strong"

    def test_rel_with_additional_attributes_keyword(self):
        """Test edge with rel (keyword) and additional attributes."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", rel="colleague", department="engineering", level=5)

        df = g.get_edge_df()
        assert df["rel"][0] == "colleague"
        assert df["department"][0] == "engineering"
        assert df["level"][0] == 5

    def test_all_params_positional(self):
        """Test edge with rel and weight positional, plus attributes."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "supervisor", 9.0, False, verified=True, year=2023)

        df = g.get_edge_df()
        assert df["rel"][0] == "supervisor"
        assert df["verified"][0] is True
        assert df["year"][0] == 2023
        assert df["weight"][0] == 9.0

    def test_all_params_keyword(self):
        """Test edge with all parameters as keywords."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge(
            source="A",
            target="B",
            rel="reports_to",
            weight=6.5,
            auto_create_nodes=False,
            status="active",
            duration=365,
        )

        df = g.get_edge_df()
        assert df["rel"][0] == "reports_to"
        assert df["status"][0] == "active"
        assert df["duration"][0] == 365
        assert df["weight"][0] == 6.5

    def test_auto_create_nodes_with_rel_positional(self):
        """Test auto_create_nodes with rel as positional argument."""
        g = ec.Graph()
        g.add_edge("X", "Y", "connects", auto_create_nodes=True)

        assert g.count_nodes() == 2
        assert g.count_edges() == 1
        df = g.get_edge_df()
        assert df["rel"][0] == "connects"

    def test_auto_create_nodes_with_rel_and_weight(self):
        """Test auto_create_nodes with both rel and weight."""
        g = ec.Graph(weighted=True)
        g.add_edge("P", "Q", "linked", 4.2, auto_create_nodes=True)

        assert g.count_nodes() == 2
        assert g.count_edges() == 1
        df = g.get_edge_df()
        assert df["rel"][0] == "linked"
        assert df["weight"][0] == 4.2

    def test_rel_with_special_characters(self):
        """Test rel with special characters."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", rel="friend_of_family")

        df = g.get_edge_df()
        assert df["rel"][0] == "friend_of_family"

    def test_rel_empty_string(self):
        """Test rel with empty string (should be stored, not null)."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", rel="")

        df = g.get_edge_df()
        assert df["rel"][0] == ""

    def test_rel_none_explicit(self):
        """Test explicitly passing None for rel."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", rel=None)

        df = g.get_edge_df()
        assert df["rel"][0] is None

    def test_weight_zero(self):
        """Test edge with weight of zero."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "zero_weight", 0.0)

        df = g.get_edge_df()
        assert df["weight"][0] == 0.0
        assert df["rel"][0] == "zero_weight"

    def test_weight_negative(self):
        """Test edge with negative weight."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "debt", -5.5)

        df = g.get_edge_df()
        assert df["weight"][0] == -5.5
        assert df["rel"][0] == "debt"

    def test_multiple_edges_different_rels(self):
        """Test multiple edges with different rel values."""
        g = ec.Graph()
        for i in range(5):
            g.add_node(f"N{i}")

        g.add_edge("N0", "N1", "friend")
        g.add_edge("N1", "N2", "colleague")
        g.add_edge("N2", "N3", "mentor")
        g.add_edge("N3", "N4")  # No rel

        df = g.get_edge_df()
        assert df["rel"][0] == "friend"
        assert df["rel"][1] == "colleague"
        assert df["rel"][2] == "mentor"
        assert df["rel"][3] is None

    def test_multiple_edges_same_rel(self):
        """Test multiple edges with same rel value."""
        g = ec.Graph()
        for i in range(4):
            g.add_node(f"N{i}")

        g.add_edge("N0", "N1", "friend")
        g.add_edge("N1", "N2", "friend")
        g.add_edge("N2", "N3", "friend")

        df = g.get_edge_df()
        assert all(df["rel"] == "friend")

    def test_rel_with_numeric_node_ids(self):
        """Test rel with numeric node IDs instead of labels."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge(1, 2, "numeric_nodes")

        df = g.get_edge_df()
        assert df["rel"][0] == "numeric_nodes"

    def test_chaining_with_rel(self):
        """Test method chaining with add_edge using rel."""
        g = ec.Graph()
        g.add_node("A").add_node("B").add_node("C")
        g.add_edge("A", "B", "first").add_edge("B", "C", "second")

        df = g.get_edge_df()
        assert len(df) == 2
        assert df["rel"][0] == "first"
        assert df["rel"][1] == "second"

    def test_rel_and_weight_with_long_attribute_list(self):
        """Test rel and weight with many additional attributes."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge(
            "A",
            "B",
            rel="complex",
            weight=7.8,
            attr1="value1",
            attr2=100,
            attr3=True,
            attr4=3.14,
            attr5="value5",
        )

        df = g.get_edge_df()
        assert df["rel"][0] == "complex"
        assert df["attr1"][0] == "value1"
        assert df["attr2"][0] == 100
        assert df["attr3"][0] is True
        assert df["attr4"][0] == 3.14
        assert df["attr5"][0] == "value5"
        assert df["weight"][0] == 7.8

    def test_rel_unicode(self):
        """Test rel with unicode characters."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", rel="朋友")  # "friend" in Chinese

        df = g.get_edge_df()
        assert df["rel"][0] == "朋友"

    def test_weight_without_weighted_graph(self):
        """Test that weight parameter is accepted but stored as attribute in unweighted graphs."""
        g = ec.Graph(weighted=False)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "unweighted", 999.9)

        df = g.get_edge_df()
        assert df["rel"][0] == "unweighted"
        # In unweighted graphs, weight is stored as a regular attribute
        assert df["weight"][0] == 999.9

    def test_dataframe_column_order_with_rel_and_weight(self):
        """Test that DataFrame maintains correct column order."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "ordered", 5.0, extra1="x", extra2="y")

        df = g.get_edge_df()
        # First 4 columns must be: id, source, target, rel
        assert list(df.columns[:4]) == ["id", "source", "target", "rel"]
        # weight should be after rel and before other attributes
        assert "weight" in df.columns
        # Other attributes should be alphabetically sorted after mandatory columns
        assert "extra1" in df.columns
        assert "extra2" in df.columns

    def test_get_edge_df_ws_preserves_rel(self):
        """Test that get_edge_df_ws() maintains rel column."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", "type1")
        g.add_edge("B", "C", "type2")

        g.select_edges_by_id([1])
        df_ws = g.get_edge_df_ws()

        assert list(df_ws.columns[:4]) == ["id", "source", "target", "rel"]
        assert df_ws["rel"][0] == "type1"

    def test_edge_exists_after_adding_with_rel(self):
        """Test that edge properly exists after adding with rel."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "exists")

        assert g.has_edge("A", "B")
        assert g.has_edge(1, 2)

    def test_get_edge_returns_rel(self):
        """Test that edge attributes include rel."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", "retrievable", extra="data")

        edge_id = g.get_edge_ids()[0]
        edge = g.get_edge(edge_id)
        assert edge.attributes["rel"] == "retrievable"
        assert edge.attributes["extra"] == "data"


class TestAddEdgeBackwardCompatibility:
    """Ensure backward compatibility with old code patterns."""

    def test_old_weight_first_pattern_fails_gracefully(self):
        """Test that old weight-first pattern would fail with new signature."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")

        # Old pattern: add_edge(source, target, weight)
        # New signature: add_edge(source, target, rel, weight)
        # If someone passes 5.0 as 3rd arg, it becomes rel (string expected)
        g.add_edge("A", "B", None, 5.0)  # Correct: rel=None, weight=5.0

        df = g.get_edge_df()
        assert df["weight"][0] == 5.0

    def test_keyword_weight_still_works(self):
        """Test that using weight= keyword still works (most compatible)."""
        g = ec.Graph(weighted=True)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", weight=7.5)  # Old code using keyword

        df = g.get_edge_df()
        assert df["weight"][0] == 7.5

    def test_no_params_still_works(self):
        """Test that basic edge creation without params still works."""
        g = ec.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")  # Old simple pattern

        assert g.has_edge("A", "B")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
