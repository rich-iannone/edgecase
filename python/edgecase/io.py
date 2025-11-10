from pathlib import Path
from typing import Optional, Union

import polars as pl

from edgecase.graph import Graph


def from_matrix_market(
    filepath: Union[str, Path],
    directed: Optional[bool] = None,
    weighted: Optional[bool] = None,
    graph_name: Optional[str] = None,
) -> Graph:
    """
    Create a graph from a Matrix Market coordinate format file.

    Matrix Market is a standard format for representing sparse matrices and networks.
    This function reads coordinate format files (.mtx) and automatically detects
    graph properties from the header line:
    - Symmetry: 'symmetric' → undirected, 'general' → directed
    - Data type: 'pattern' → unweighted, 'real'/'integer' → weighted

    The function automatically converts 1-indexed node IDs (Matrix Market standard)
    to 0-indexed node IDs used internally.

    Parameters
    ----------
    filepath : str or Path
        Path to the Matrix Market (.mtx) file
    directed : bool, optional
        Whether to create a directed graph. If None (default), automatically
        detected from file header ('symmetric' → False, 'general' → True)
    weighted : bool, optional
        Whether the graph is weighted. If None (default), automatically detected
        from file header ('pattern' → False, 'real'/'integer' → True)
    graph_name : str, optional
        Name for the graph

    Returns
    -------
    Graph
        A graph constructed from the Matrix Market file

    Examples
    --------
    ```python
    from edgecase.io import from_matrix_market

    # Auto-detect graph properties from file header
    G = from_matrix_market("soc-karate.mtx")
    print(f"Directed: {G.is_directed}, Weighted: {G.is_weighted}")
    # Directed: False, Weighted: False

    # Override auto-detection
    G = from_matrix_market("network.mtx", directed=True, weighted=False)

    # Load with a custom name
    G = from_matrix_market(
        "network.mtx",
        graph_name="My Network"
    )
    print(G.name)
    # My Network
    ```

    Notes
    -----
    - Matrix Market files use 1-based indexing, which is automatically converted
      to 0-based indexing
    - Only coordinate format (sparse) is supported
    - Header format: %%MatrixMarket matrix coordinate <type> <symmetry>
      * Symmetry: 'symmetric' → undirected, 'general' → directed
      * Type: 'pattern' → unweighted, 'real'/'integer' → weighted
    - Parameters `directed` and `weighted` override auto-detection when provided

    References
    ----------
    Matrix Market format specification:
    https://math.nist.gov/MatrixMarket/formats.html
    """
    filepath = Path(filepath)

    # Parse header to detect graph properties
    detected_directed = None
    detected_weighted = None

    with open(filepath, "r") as f:
        first_line = f.readline().strip()

        # Parse Matrix Market header: %%MatrixMarket matrix coordinate <type> <symmetry>
        if first_line.startswith("%%MatrixMarket"):
            parts = first_line.lower().split()

            # Check symmetry (last part)
            if len(parts) >= 5:
                symmetry = parts[4]
                if symmetry == "symmetric":
                    detected_directed = False
                elif symmetry in ("general", "skew-symmetric"):
                    detected_directed = True

            # Check data type (4th part)
            if len(parts) >= 4:
                data_type = parts[3]
                if data_type == "pattern":
                    detected_weighted = False
                elif data_type in ("real", "integer", "complex"):
                    detected_weighted = True

    # Use provided values or fall back to detected values or defaults
    use_directed = (
        directed
        if directed is not None
        else (detected_directed if detected_directed is not None else False)
    )
    use_weighted = (
        weighted
        if weighted is not None
        else (detected_weighted if detected_weighted is not None else False)
    )

    # Read the file and parse edges
    edges = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments
            if line.startswith("%"):
                continue

            parts = line.split()

            # Skip header line (rows cols nonzeros)
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                continue

            # Parse edge data
            if len(parts) >= 2:
                # Convert from 1-indexed to 0-indexed
                source = int(parts[0]) - 1
                target = int(parts[1]) - 1

                # Get weight if available and requested
                weight = None
                if use_weighted and len(parts) >= 3:
                    weight = float(parts[2])

                edges.append((source, target, weight))

    # Create edges DataFrame
    edge_data = {"from": [e[0] for e in edges], "to": [e[1] for e in edges]}

    if use_weighted:
        edge_data["weight"] = [e[2] if e[2] is not None else 1.0 for e in edges]

    edges_df = pl.DataFrame(edge_data)

    # Build and return graph
    # Note: nodes_df is None, so nodes will be auto-created from edges
    graph = Graph(
        directed=use_directed, weighted=use_weighted, edges_df=edges_df, graph_name=graph_name
    )

    return graph


__all__ = [
    "from_matrix_market",
]
