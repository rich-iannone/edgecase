from typing import Optional


def to_igraph(graph):
    """Convert an EdgeCase Graph to an igraph graph.

    Creates an igraph graph preserving directionality, node/edge attributes, and
    weights. Node names are stored as 'name' node attribute since igraph uses
    integer node indices internally.

    Vertices are added in node ID order, each named by its label (or, if it has none, a string of
    its ID). Because `name` holds the labels, a node attribute called `name` isn't written. Edges
    are added in ID order, with the EdgeCase edge IDs in the `edgecase_edge_id` edge attribute,
    and the graph name, if any, is stored as the graph attribute `name`.

    Parameters
    ----------
    graph
        The EdgeCase Graph object to convert.

    Returns
    -------
    igraph.Graph
        An igraph.Graph object with the same structure and attributes.

    Raises
    ------
    ImportError
        If igraph is not installed.

    Examples
    --------
    ```python
    import edgecase as ec
    from edgecase.igraph_interop import to_igraph

    graph = ec.Graph(directed=True)
    graph.add_node("A", color="red", value=10)
    graph.add_node("B", color="blue", value=20)
    graph.add_edge("A", "B", weight=1.5, label="edge1")

    ig_graph = to_igraph(graph)

    # Returns ['A', 'B']
    ig_graph.vs["name"]

    # Returns ['red', 'blue']
    ig_graph.vs["color"]

    # Returns [1.5]
    ig_graph.es["weight"]
    ```

    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError(
            "igraph is required for graph conversion. Install it with: pip install igraph"
        )

    from edgecase.io import _node_keys, _nodes_of

    # Create igraph graph with appropriate directionality
    ig_graph = ig.Graph(directed=graph.is_directed())
    if graph.get_graph_name():
        ig_graph["name"] = graph.get_graph_name()

    # Nodes in ID order, each named by its label (or, if unlabeled, a string of its ID)
    nodes = _nodes_of(graph)
    names = _node_keys(nodes)
    ig_graph.add_vertices(len(nodes))
    ig_graph.vs["name"] = [names[node_id] for node_id, _, _ in nodes]

    # Add node attributes; `name` holds the labels, so an attribute of that name isn't written
    node_attr_names = list(dict.fromkeys(k for _, _, attrs in nodes for k in attrs))
    for attr_name in node_attr_names:
        if attr_name != "name":
            ig_graph.vs[attr_name] = [attrs.get(attr_name) for _, _, attrs in nodes]

    # Create node ID to index mapping for edge creation
    node_to_idx = {node_id: idx for idx, (node_id, _, _) in enumerate(nodes)}

    # Edges in ID order, with their weights (or `weight` attributes) and their attributes
    edges = sorted(graph.get_edges_as_tuples())  # (id, source, target) tuples
    edge_list = []
    edge_weights = []
    edge_ids = []
    edge_attrs = []
    for edge_id, source, target in edges:
        edge = graph.get_edge(edge_id)
        attrs = dict(edge.attributes or {})
        weight = attrs.pop("weight", None)
        if edge.weight is not None:
            weight = edge.weight
        attrs.pop("edgecase_edge_id", None)
        edge_list.append((node_to_idx[source], node_to_idx[target]))
        edge_weights.append(weight)
        edge_ids.append(edge_id)
        edge_attrs.append(attrs)

    # Add all edges at once
    if edge_list:
        ig_graph.add_edges(edge_list)

        # Set edge attributes
        if any(w is not None for w in edge_weights):
            ig_graph.es["weight"] = edge_weights

        # Store EdgeCase edge IDs for round-trip conversion
        ig_graph.es["edgecase_edge_id"] = edge_ids

        # Set other edge attributes
        for attr_name in dict.fromkeys(k for attrs in edge_attrs for k in attrs):
            ig_graph.es[attr_name] = [attrs.get(attr_name) for attrs in edge_attrs]

    return ig_graph


def from_igraph(ig_graph, weighted: Optional[bool] = None):
    """Convert an igraph graph to an EdgeCase Graph.

    Creates an EdgeCase graph from igraph, preserving directionality, node/edge
    attributes, and weights. Node names are taken from 'name' node attribute if
    present, otherwise node indices are used.

    Every other vertex attribute is kept, including ones named `label` or `type`. If two vertices
    share a name, every node is labeled with its index instead (with a warning) and the names are
    kept in a `name` attribute. A numeric `weight` becomes the edge weight and every other edge
    attribute is kept, except the `edgecase_edge_id` that `to_igraph()` adds. A graph attribute
    `name` becomes the graph name. The graph is built as a single step.

    Parameters
    ----------
    ig_graph
        The igraph.Graph object to convert.
    weighted
        If `True`, create a weighted graph. If `None` (default), auto-detect
        by checking if any edge has a numeric 'weight' attribute. If `False`, edge
        weights are dropped.

    Returns
    -------
    Graph
        An EdgeCase Graph object with the same structure and attributes.

    Raises
    ------
    ImportError
        If igraph is not installed.

    Examples
    --------
    ```python
    import igraph as ig
    from edgecase.igraph_interop import from_igraph

    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(2)
    ig_graph.vs["name"] = ["A", "B"]
    ig_graph.vs["color"] = ["red", "blue"]
    ig_graph.add_edge(0, 1, weight=1.5, label="edge1")

    graph = from_igraph(ig_graph)

    # Returns 2
    graph.count_nodes()

    # Returns 1
    graph.count_edges()

    node = graph.get_node(graph.get_id_from_label("A"))

    # Returns 'red'
    node.attributes["color"]
    ```

    """
    try:
        import igraph  # noqa: F401
    except ImportError:
        raise ImportError(
            "igraph is required for graph conversion. Install it with: pip install igraph"
        )

    from edgecase.io import _build_graph, _code, _numeric_weight

    # Each vertex is labeled with its `name` if it has one, otherwise with its index; every other
    # attribute is kept (igraph fills `None` for vertices without it, and those are dropped)
    nodes = []
    for vertex in ig_graph.vs:
        attrs = vertex.attributes()
        name = attrs.pop("name", None)
        nodes.append((vertex.index, None if name is None else str(name), attrs))

    # A numeric `weight` becomes the edge weight; the `edgecase_edge_id` that `to_igraph()` adds is
    # dropped
    edges = []
    for edge in ig_graph.es:
        attrs = edge.attributes()
        attrs.pop("edgecase_edge_id", None)
        weight = _numeric_weight(attrs.get("weight"))
        if weight is not None:
            del attrs["weight"]
        edges.append((edge.source, edge.target, weight, attrs))

    # Auto-detect if graph is weighted
    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _ in edges)

    name = ig_graph["name"] if "name" in ig_graph.attributes() else None
    return _build_graph(
        nodes,
        edges,
        ig_graph.is_directed(),
        weighted,
        name if isinstance(name, str) and name else None,
        _code("from_igraph", ig_graph),
    )


__all__ = ["to_igraph", "from_igraph"]
