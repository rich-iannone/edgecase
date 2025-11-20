import polars as pl

from edgecase.graph import Graph


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
    ```python
    import edgecase as ec
    G = ec.Graph(directed=False)
    # ... build graph ...
    df = ec.to_polars(G)
    ```
    """
    # Placeholder implementation
    return pl.DataFrame(
        {
            "source": [],
            "target": [],
            "weight": [],
        }
    )
