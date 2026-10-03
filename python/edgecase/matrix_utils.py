from typing import Optional, Sequence, Union

from edgecase import _core as _ec


def to_adjacency_matrix(graph, nonedge: float = 0.0) -> tuple[list[list[float]], list[int]]:
    """Convert an EdgeCase Graph to an adjacency matrix.

    Returns an adjacency matrix representation of the graph along with the node IDs in the order
    they appear in the matrix. The matrix value at position [i][j] represents the edge weight from
    node i to node j (or `1.0` for unweighted graphs), or the `nonedge=` value if no edge exists.

    Parameters
    ----------
    graph
        The EdgeCase Graph object to convert.
    nonedge
        Value to use for non-edges in the matrix. Common choices are: (1) `0.0` (tdefault) for a
        standard adjacency matrix, (2) `float('inf')` for distance/shortest path algorithms, and (3)
        `-1` for algorithms that need to distinguish non-edges.

    Returns
    -------
    tuple[list[list[float]], list[int]]
        A tuple containing: (1) a matrix as a 2D list where matrix[i][j] is the edge weight from
        node i to j, and (2) node_ids comprising a list of node IDs in the order they appear in the
        matrix.

    Examples
    --------
    ```python
    import edgecase as ec
    from edgecase.matrix_utils import to_adjacency_matrix

    # Simple undirected graph
    graph = ec.Graph()
    graph.add_node(label="A")
    graph.add_node(label="B")
    graph.add_node(label="C")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")

    matrix, node_ids = to_adjacency_matrix(graph)
    # matrix is symmetric for undirected graphs

    # Weighted graph
    graph = ec.Graph(weighted=True)
    graph.add_node(label="X")
    graph.add_node(label="Y")
    graph.add_node(label="Z")
    graph.add_edge("X", "Y", weight=2.5)
    graph.add_edge("Y", "Z", weight=1.0)

    matrix, node_ids = to_adjacency_matrix(graph)
    # matrix contains actual edge weights

    # Use infinity for non-edges (useful for shortest path)
    matrix, node_ids = to_adjacency_matrix(graph, nonedge=float('inf'))
    ```

    Notes
    -----
    - for undirected graphs, the matrix is symmetric
    - for directed graphs, matrix[i][j] represents edge from i to j
    - self-loops appear on the diagonal
    - the node ordering is determined by the internal node ID order

    """
    matrix, node_ids = _ec.to_adjacency_matrix(graph._graph, None, nonedge)
    return matrix, node_ids


def from_adjacency_matrix(
    matrix: Sequence[Sequence[float]],
    directed: bool = False,
    weighted: Optional[bool] = None,
    node_labels: Optional[Sequence[Union[str, int]]] = None,
):
    """Create an EdgeCase Graph from an adjacency matrix.

    Constructs a graph from a 2D matrix where non-zero values indicate edges. The value at
    matrix[i][j] becomes the edge weight from node i to node j.

    Parameters
    ----------
    matrix
        A square 2D matrix (list of lists, numpy array, etc.) where: (1) non-zero values indicate
        edges, (2) the value becomes the edge weight (for weighted graphs), and (3) matrix[i][j]
        represents an edge from node i to node j.
    directed
        If `True`, create a directed graph. If `False`, only the upper triangle is used for edge
        creation (lower triangle is ignored).
    weighted
        If `True`, use matrix values as edge weights. If `False`, all edges have weight `1.0`. If
        `None` (the default), auto-detect as weighted if any non-zero value is not equal to `1.0`.
    node_labels
        Labels for the nodes. If provided, must have length equal to the matrix dimension. If
        `None`, nodes are labeled `"0"`, `"1"`, `"2"`, etc.

    Returns
    -------
    Graph
        A new EdgeCase Graph constructed from the matrix.

    Raises
    ------
    ValueError
        If the matrix is not square, or if node_labels length doesn't match the matrix dimension.

    Examples
    --------
    ```python
    from edgecase.matrix_utils import from_adjacency_matrix

    # Simple triangle graph
    matrix = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    graph = from_adjacency_matrix(matrix)
    print(graph.count_edges())  # 3

    # With custom node labels
    graph = from_adjacency_matrix(matrix, node_labels=["A", "B", "C"])

    # Directed graph
    matrix = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]
    graph = from_adjacency_matrix(matrix, directed=True)

    # Weighted graph
    matrix = [
        [0.0, 2.5, 0.0],
        [2.5, 0.0, 1.5],
        [0.0, 1.5, 0.0]
    ]
    graph = from_adjacency_matrix(matrix, weighted=True)
    ```

    Notes
    -----
    - for undirected graphs, only the upper triangle of the matrix is used
    - zero values are treated as non-edges
    - the diagonal represents self-loops
    - works with numpy arrays (they are converted to lists internally)

    """
    from edgecase import Graph

    # Convert to list of lists if needed (e.g., numpy array)
    matrix_list = [list(row) for row in matrix]

    # Convert node_labels to list of strings if provided
    labels_list = None
    if node_labels is not None:
        labels_list = [str(label) for label in node_labels]

    # Call the Rust function (signature: matrix, node_labels, directed, weighted)
    rust_graph = _ec.from_adjacency_matrix(matrix_list, labels_list, directed, weighted)

    # Wrap in Python Graph
    graph = Graph._from_rust(rust_graph)

    return graph


__all__ = ["to_adjacency_matrix", "from_adjacency_matrix"]
