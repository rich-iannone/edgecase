import pytest
import polars as pl
from edgecase import Graph


# Graph constructor with edges_df tests


def test_basic_conversion(sample_dataframe):
    """Test basic DataFrame to graph conversion"""
    df = sample_dataframe.rename({"source": "from", "target": "to"})
    G = Graph(edges_df=df)

    assert G.count_nodes() == 4  # A, B, C, D
    assert G.count_edges() == 4


def test_with_weights(sample_dataframe):
    """Test conversion with weight column"""
    df = sample_dataframe.rename({"source": "from", "target": "to"})
    G = Graph(edges_df=df, weighted=True)

    assert G.is_weighted()
    assert G.count_edges() == 4


def test_custom_column_names():
    """Test with custom column names"""
    df = pl.DataFrame(
        {
            "from": ["A", "B"],
            "to": ["B", "C"],
        }
    )

    G = Graph(edges_df=df)

    assert G.count_nodes() == 3
    assert G.count_edges() == 2


def test_empty_dataframe():
    """Test with empty DataFrame"""
    df = pl.DataFrame(
        {
            "from": [],
            "to": [],
        }
    )

    G = Graph(edges_df=df)

    assert G.count_nodes() == 0
    assert G.count_edges() == 0


def test_self_loops():
    """Test handling self-loops"""
    df = pl.DataFrame(
        {
            "from": ["A", "A", "B"],
            "to": ["A", "B", "B"],
        }
    )

    G = Graph(edges_df=df)

    # Should handle self-loops
    assert G.count_nodes() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
