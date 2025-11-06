# tests/integration/test_polars_integration.py
"""
Integration tests for Polars interoperability
"""

import pytest
import polars as pl
from edgecase import from_polars, Graph


# from_polars tests

def test_basic_conversion(sample_dataframe):
    """Test basic DataFrame to graph conversion"""
    G = from_polars(
        sample_dataframe,
        source_col="source",
        target_col="target"
    )
    
    assert G.num_nodes() == 4  # A, B, C, D
    assert G.num_edges() == 4


def test_with_weights(sample_dataframe):
    """Test conversion with weight column"""
    G = from_polars(
        sample_dataframe,
        source_col="source",
        target_col="target",
        weight_col="weight"
    )
    
    assert G.is_weighted
    assert G.num_edges() == 4


def test_custom_column_names():
    """Test with custom column names"""
    df = pl.DataFrame({
        "from": ["A", "B"],
        "to": ["B", "C"],
    })
    
    G = from_polars(df, source_col="from", target_col="to")
    
    assert G.num_nodes() == 3
    assert G.num_edges() == 2


def test_empty_dataframe():
    """Test with empty DataFrame"""
    df = pl.DataFrame({
        "source": [],
        "target": [],
    })
    
    G = from_polars(df, source_col="source", target_col="target")
    
    assert G.num_nodes() == 0
    assert G.num_edges() == 0


def test_self_loops():
    """Test handling self-loops"""
    df = pl.DataFrame({
        "source": ["A", "A", "B"],
        "target": ["A", "B", "B"],
    })
    
    G = from_polars(df, source_col="source", target_col="target")
    
    # Should handle self-loops
    assert G.num_nodes() == 2


# to_polars tests

@pytest.mark.skip(reason="to_polars not yet implemented")
def test_to_polars_basic_conversion(simple_graph):
    """Test basic graph to DataFrame conversion"""
    from edgecase import to_polars
    
    df = to_polars(simple_graph)
    
    assert isinstance(df, pl.DataFrame)
    assert len(df) == simple_graph.num_edges()
    assert "source" in df.columns
    assert "target" in df.columns


@pytest.mark.skip(reason="to_polars not yet implemented")
def test_to_polars_roundtrip(sample_dataframe):
    """Test DataFrame → Graph → DataFrame preserves data"""
    from edgecase import to_polars
    
    G = from_polars(
        sample_dataframe,
        source_col="source",
        target_col="target",
        weight_col="weight"
    )
    
    df_out = to_polars(G)
    
    # Sort both for comparison
    df_in = sample_dataframe.sort("source", "target")
    df_out = df_out.sort("source", "target")
    
    assert df_in.equals(df_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
