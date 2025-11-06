"""Polars integration for graph construction and export."""

from typing import Optional, Union, List
import polars as pl

# Import Graph from the wrapped module to ensure consistent API
try:
    from edgecase._core_rust import Graph
    _USING_RUST = True
except ImportError:
    from edgecase._core_simple import Graph
    _USING_RUST = False


def from_polars(
    df: Union[pl.DataFrame, pl.LazyFrame],
    source_col: str,
    target_col: str,
    weight_col: Optional[str] = None,
    node_attrs: Optional[List[str]] = None,
    edge_attrs: Optional[List[str]] = None,
    directed: bool = False,
    temporal: bool = False,
) -> Graph:
    """
    Create a graph from a Polars DataFrame or LazyFrame.
    
    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        The input data containing edge information
    source_col : str
        Column name for source nodes
    target_col : str
        Column name for target nodes
    weight_col : str, optional
        Column name for edge weights
    node_attrs : list of str, optional
        Additional columns to use as node attributes
    edge_attrs : list of str, optional
        Additional columns to use as edge attributes
    directed : bool, default False
        Whether the graph is directed
    temporal : bool, default False
        Whether this is a temporal graph
        
    Returns
    -------
    Graph
        A graph constructed from the DataFrame
        
    Examples
    --------
    >>> import polars as pl
    >>> import edgecase as ec
    >>> df = pl.DataFrame({
    ...     "from": ["A", "B", "C"],
    ...     "to": ["B", "C", "A"],
    ...     "weight": [1.0, 2.0, 1.5]
    ... })
    >>> G = ec.from_polars(df, source_col="from", target_col="to", weight_col="weight")
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    # Create graph
    graph = Graph(directed=directed, weighted=weight_col is not None)
    
    # Extract unique nodes
    sources = df[source_col].unique()
    targets = df[target_col].unique()
    all_nodes = pl.concat([sources, targets]).unique()
    
    # Add nodes - both wrappers support add_node(str)
    for node_id in all_nodes:
        graph.add_node(str(node_id))
    
    # Add edges - both wrappers support add_edge(str, str, weight=...)
    for row in df.iter_rows(named=True):
        source = str(row[source_col])
        target = str(row[target_col])
        weight = float(row[weight_col]) if weight_col else None
        graph.add_edge(source, target, weight=weight)
    
    return graph


def to_polars(graph: Graph) -> pl.DataFrame:
    """
    Convert a graph to a Polars DataFrame.
    
    Parameters
    ----------
    graph : Graph
        The input graph
        
    Returns
    -------
    pl.DataFrame
        DataFrame with edge list representation
        
    Examples
    --------
    >>> import edgecase as ec
    >>> G = ec.Graph(directed=False)
    >>> # ... build graph ...
    >>> df = ec.to_polars(G)
    """
    # Placeholder implementation
    return pl.DataFrame({
        "source": [],
        "target": [],
        "weight": [],
    })
