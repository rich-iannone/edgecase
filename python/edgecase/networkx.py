import warnings
from typing import Optional


def to_networkx(graph, multigraph: bool = False):
    """Convert an EdgeCase Graph to a NetworkX graph.

    Creates a NetworkX graph preserving directionality, node/edge attributes,
    and optionally multiple edges between nodes. Useful for validation against
    NetworkX algorithms and leveraging the NetworkX ecosystem.

    Parameters
    ---
    graph : Graph
        The EdgeCase Graph object to convert.
    multigraph : bool, default False
        If `True`, create a MultiGraph/MultiDiGraph preserving edge IDs and
        multiple edges. If `False`, create a Graph/DiGraph. When `False` and
        multiple edges exist, only the last edge is preserved.

    Returns
    ---
    networkx.Graph or networkx.DiGraph or networkx.MultiGraph or networkx.MultiDiGraph
        A NetworkX graph with the same structure and attributes.

    Raises
    ---
    ImportError
        If NetworkX is not installed.

    Examples
    --------
    ```python
    import edgecase as ec
    from edgecase.networkx import to_networkx

    graph = ec.Graph(directed=True)
    graph.add_node("A", color="red", value=10)
    graph.add_node("B", color="blue", value=20)
    graph.add_edge("A", "B", weight=1.5, label="edge1")

    nx_graph = to_networkx(graph)
    print(nx_graph.nodes["A"])  # {'color': 'red', 'value': 10}
    print(nx_graph["A"]["B"])   # {'weight': 1.5, 'label': 'edge1'}

    # For graphs with multiple edges, use multigraph=True
    graph.add_edge("A", "B", weight=2.0, label="edge2")
    nx_multi = to_networkx(graph, multigraph=True)
    print(len(nx_multi["A"]["B"]))  # 2 edges
    ```

    %family Interoperability
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for graph conversion. Install it with: pip install networkx"
        )

    # Determine the appropriate NetworkX graph type
    if multigraph:
        nx_graph = nx.MultiDiGraph() if graph.is_directed() else nx.MultiGraph()
    else:
        nx_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    # Add nodes with attributes
    for node_id in graph.get_node_ids(limit=None):
        node = graph.get_node(node_id)
        node_attrs = {}
        if node and node.attributes:
            # Convert PyDict to regular dict
            node_attrs = dict(node.attributes)

        # Use label as NetworkX node ID if it exists, otherwise use the numeric ID
        nx_node_id = graph.get_label_from_id(node_id) or node_id
        nx_graph.add_node(nx_node_id, **node_attrs)

    # Add edges with attributes
    edges = graph.get_edges_as_tuples()  # Returns list of (id, source, target) tuples

    if multigraph:
        # MultiGraph: preserve edge IDs and multiple edges
        for edge_id, source, target in edges:
            edge = graph.get_edge(edge_id)
            edge_attrs = {}
            if edge and edge.attributes:
                edge_attrs = dict(edge.attributes)

            # Add weight if it exists
            if edge and edge.weight is not None:
                edge_attrs["weight"] = edge.weight

            # Add edge ID as metadata
            edge_attrs["edgecase_edge_id"] = edge_id

            # Use labels if available, otherwise use IDs
            nx_source = graph.get_label_from_id(source) or source
            nx_target = graph.get_label_from_id(target) or target
            nx_graph.add_edge(nx_source, nx_target, key=edge_id, **edge_attrs)
    else:
        # Simple Graph: warn if multiple edges exist
        edge_counts = {}
        for edge_id, source, target in edges:
            edge_key = (source, target)
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1

        if any(count > 1 for count in edge_counts.values()):
            warnings.warn(
                "Graph contains multiple edges between some node pairs. "
                "Only the last edge will be preserved. Use multigraph=True "
                "to preserve all edges.",
                UserWarning,
            )

        for edge_id, source, target in edges:
            edge = graph.get_edge(edge_id)
            edge_attrs = {}
            if edge and edge.attributes:
                edge_attrs = dict(edge.attributes)

            # Add weight if it exists
            if edge and edge.weight is not None:
                edge_attrs["weight"] = edge.weight

            # Use labels if available, otherwise use IDs
            nx_source = graph.get_label_from_id(source) or source
            nx_target = graph.get_label_from_id(target) or target
            nx_graph.add_edge(nx_source, nx_target, **edge_attrs)

    return nx_graph


def from_networkx(nx_graph, weighted: Optional[bool] = None):
    """Convert a NetworkX graph to an EdgeCase Graph.

    Creates an EdgeCase graph from NetworkX, preserving directionality, node/edge
    attributes, and multiple edges (for MultiGraph/MultiDiGraph). Auto-detects
    weighted graphs unless explicitly specified.

    Parameters
    ---
    nx_graph : networkx.Graph or networkx.DiGraph or networkx.MultiGraph or networkx.MultiDiGraph
        The NetworkX graph to convert.
    weighted : bool, optional
        If `True`, create a weighted graph. If `None` (default), auto-detect
        by checking if any edge has a 'weight' attribute.

    Returns
    ---
    Graph
        An EdgeCase Graph object with the same structure and attributes.

    Raises
    ---
    ImportError
        If NetworkX is not installed.

    Examples
    --------
    ```python
    import networkx as nx
    from edgecase.networkx import from_networkx

    nx_graph = nx.DiGraph()
    nx_graph.add_node("A", color="red", value=10)
    nx_graph.add_node("B", color="blue", value=20)
    nx_graph.add_edge("A", "B", weight=1.5, label="edge1")

    graph = from_networkx(nx_graph)
    print(graph.count_nodes())  # 2
    print(graph.count_edges())  # 1
    node = graph.get_node(graph.get_id_from_label("A"))
    print(node.attributes["color"])  # 'red'

    # Works with MultiGraphs too
    nx_multi = nx.MultiGraph()
    nx_multi.add_edge("X", "Y", weight=1.0)
    nx_multi.add_edge("X", "Y", weight=2.0)
    graph = from_networkx(nx_multi)
    print(graph.count_edges())  # 2
    ```

    %family Interoperability
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for graph conversion. Install it with: pip install networkx"
        )

    # Import EdgeCase Graph
    from edgecase import Graph

    # Determine if graph is directed
    directed = nx_graph.is_directed()

    # Auto-detect if graph is weighted
    if weighted is None:
        weighted = False
        # Check if any edge has a weight attribute
        for u, v, data in nx_graph.edges(data=True):
            if "weight" in data:
                weighted = True
                break

    # Create EdgeCase graph
    graph = Graph(directed=directed, weighted=weighted)

    # Add nodes with attributes
    for node_id, node_attrs in nx_graph.nodes(data=True):
        # Convert node_id to string (EdgeCase uses string node IDs)
        node_id_str = str(node_id)
        graph.add_node(label=node_id_str, **node_attrs)

    # Add edges with attributes
    if nx_graph.is_multigraph():
        # MultiGraph: iterate over all edges with keys
        for u, v, key, edge_attrs in nx_graph.edges(keys=True, data=True):
            u_str = str(u)
            v_str = str(v)

            # Extract weight if present
            weight = edge_attrs.get("weight", None)

            # Remove weight from attrs dict to avoid duplication
            edge_attrs_copy = edge_attrs.copy()
            edge_attrs_copy.pop("weight", None)

            # If edge has an 'edgecase_edge_id', use it
            edge_id = edge_attrs_copy.pop("edgecase_edge_id", None)

            graph.add_edge(u_str, v_str, weight=weight, edge_id=edge_id, **edge_attrs_copy)
    else:
        # Simple Graph: iterate over edges normally
        for u, v, edge_attrs in nx_graph.edges(data=True):
            u_str = str(u)
            v_str = str(v)

            # Extract weight if present
            weight = edge_attrs.get("weight", None)

            # Remove weight from attrs dict to avoid duplication
            edge_attrs_copy = edge_attrs.copy()
            edge_attrs_copy.pop("weight", None)

            graph.add_edge(u_str, v_str, weight=weight, **edge_attrs_copy)

    return graph


__all__ = ["to_networkx", "from_networkx"]
