from typing import Optional


def to_igraph(graph):
    """Convert an EdgeCase Graph to an igraph graph.

    Creates an igraph graph preserving directionality, node/edge attributes, and
    weights. Node names are stored as 'name' node attribute since igraph uses
    integer node indices internally.

    Parameters
    ---
    graph : Graph
        The EdgeCase Graph object to convert.

    Returns
    ---
    igraph.Graph
        An igraph.Graph object with the same structure and attributes.

    Raises
    ---
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
    print(ig_graph.vs["name"])    # ['A', 'B']
    print(ig_graph.vs["color"])   # ['red', 'blue']
    print(ig_graph.es["weight"])  # [1.5]
    ```

    %family Interoperability
    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError(
            "igraph is required for graph conversion. Install it with: pip install igraph"
        )

    # Create igraph graph with appropriate directionality
    ig_graph = ig.Graph(directed=graph.is_directed())

    # Get all node IDs and create a mapping
    node_ids = list(graph.get_node_ids(limit=None))
    n_nodes = len(node_ids)

    # Add vertices
    ig_graph.add_vertices(n_nodes)

    # Set node labels as "name" attribute (converting IDs to labels)
    node_labels = [graph.get_label_from_id(node_id) for node_id in node_ids]
    ig_graph.vs["name"] = node_labels

    # Add node attributes
    # First, collect all unique attribute names
    all_node_attrs = set()
    for node_id in node_ids:
        node = graph.get_node(node_id)
        if node and node.attributes:
            all_node_attrs.update(node.attributes.keys())

    # For each attribute, collect values for all nodes
    for attr_name in all_node_attrs:
        attr_values = []
        for node_id in node_ids:
            node = graph.get_node(node_id)
            if node and node.attributes and attr_name in node.attributes:
                attr_values.append(node.attributes[attr_name])
            else:
                attr_values.append(None)
        ig_graph.vs[attr_name] = attr_values

    # Create node ID to index mapping for edge creation
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Add edges
    edges = graph.get_edges_as_tuples()  # Returns list of (id, source, target) tuples
    edge_list = []
    edge_weights = []
    edge_ids = []

    # Collect all edge attributes
    all_edge_attrs = set()
    for edge_id, source, target in edges:
        edge = graph.get_edge(edge_id)
        if edge and edge.attributes:
            all_edge_attrs.update(edge.attributes.keys())

    # Prepare edge attribute storage
    edge_attr_values = {attr: [] for attr in all_edge_attrs}

    # Add edges and collect attributes
    for edge_id, source, target in edges:
        # Convert node IDs to indices
        source_idx = node_to_idx[source]
        target_idx = node_to_idx[target]
        edge_list.append((source_idx, target_idx))

        # Get edge object
        edge = graph.get_edge(edge_id)

        # Store weight
        if edge and edge.weight is not None:
            edge_weights.append(edge.weight)
        else:
            edge_weights.append(None)

        # Store edge ID
        edge_ids.append(edge_id)

        # Store other attributes
        for attr_name in all_edge_attrs:
            if edge and edge.attributes and attr_name in edge.attributes:
                edge_attr_values[attr_name].append(edge.attributes[attr_name])
            else:
                edge_attr_values[attr_name].append(None)

    # Add all edges at once
    if edge_list:
        ig_graph.add_edges(edge_list)

        # Set edge attributes
        if any(w is not None for w in edge_weights):
            ig_graph.es["weight"] = edge_weights

        # Store EdgeCase edge IDs for round-trip conversion
        ig_graph.es["edgecase_edge_id"] = edge_ids

        # Set other edge attributes
        for attr_name, attr_vals in edge_attr_values.items():
            ig_graph.es[attr_name] = attr_vals

    return ig_graph


def from_igraph(ig_graph, weighted: Optional[bool] = None):
    """Convert an igraph graph to an EdgeCase Graph.

    Creates an EdgeCase graph from igraph, preserving directionality, node/edge
    attributes, and weights. Node names are taken from 'name' node attribute if
    present, otherwise node indices are used.

    Parameters
    ---
    ig_graph : igraph.Graph
        The igraph.Graph object to convert.
    weighted : bool, optional
        If `True`, create a weighted graph. If `None` (default), auto-detect
        by checking if edges have a 'weight' attribute.

    Returns
    ---
    Graph
        An EdgeCase Graph object with the same structure and attributes.

    Raises
    ---
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
    print(graph.count_nodes())  # 2
    print(graph.count_edges())  # 1
    node = graph.get_node(graph.get_id_from_label("A"))
    print(node.attributes["color"])  # 'red'
    ```

    %family Interoperability
    """
    try:
        import igraph  # noqa: F401
    except ImportError:
        raise ImportError(
            "igraph is required for graph conversion. Install it with: pip install igraph"
        )

    # Import EdgeCase Graph
    from edgecase import Graph

    # Determine if graph is directed
    directed = ig_graph.is_directed()

    # Auto-detect if graph is weighted
    if weighted is None:
        weighted = False
        if ig_graph.ecount() > 0 and "weight" in ig_graph.es.attributes():
            weighted = True

    # Create EdgeCase graph
    graph = Graph(directed=directed, weighted=weighted)

    # Determine node IDs
    # If 'name' attribute exists, use it; otherwise use node indices
    if "name" in ig_graph.vs.attributes():
        node_ids = ig_graph.vs["name"]
    else:
        node_ids = [str(i) for i in range(ig_graph.vcount())]

    # Get all node attribute names (excluding 'name' since we use it for node ID)
    node_attrs = [attr for attr in ig_graph.vs.attributes() if attr != "name"]

    # Add nodes with attributes
    for idx, node_id in enumerate(node_ids):
        # Convert node_id to string (EdgeCase uses string node IDs)
        node_id_str = str(node_id)

        # Collect node attributes
        node_attributes = {}
        for attr_name in node_attrs:
            attr_value = ig_graph.vs[idx][attr_name]
            if attr_value is not None:
                node_attributes[attr_name] = attr_value

        graph.add_node(label=node_id_str, **node_attributes)

    # Get all edge attribute names
    edge_attrs = ig_graph.es.attributes()

    # Add edges with attributes
    for edge in ig_graph.es:
        # Get source and target node IDs
        source_idx = edge.source
        target_idx = edge.target
        source_id = str(node_ids[source_idx])
        target_id = str(node_ids[target_idx])

        # Extract weight if present
        weight = None
        if "weight" in edge_attrs:
            weight = edge["weight"]

        # Extract edge ID if it was preserved from EdgeCase
        edge_id = None
        if "edgecase_edge_id" in edge_attrs:
            edge_id = edge["edgecase_edge_id"]

        # Collect other edge attributes
        edge_attr_dict = {}
        for attr_name in edge_attrs:
            if attr_name not in ["weight", "edgecase_edge_id"]:
                attr_value = edge[attr_name]
                if attr_value is not None:
                    edge_attr_dict[attr_name] = attr_value

        # Add edge to graph
        graph.add_edge(source_id, target_id, weight=weight, edge_id=edge_id, **edge_attr_dict)

    return graph


__all__ = ["to_igraph", "from_igraph"]
