"""
igraph interoperability module for EdgeCase.

This module provides bidirectional conversion between EdgeCase and igraph graph objects,
enabling validation of EdgeCase algorithms against igraph and providing interoperability
with the igraph ecosystem.

Functions:
    to_igraph: Convert an EdgeCase Graph to an igraph graph
    from_igraph: Convert an igraph graph to an EdgeCase Graph
"""

from typing import Optional


def to_igraph(graph):
    """Convert an EdgeCase Graph to an igraph graph.
    
    This function creates an igraph graph from an EdgeCase graph, preserving:
    - Graph directionality (directed/undirected)
    - Node attributes
    - Edge attributes (including weights)
    - Node names (stored as 'name' vertex attribute)
    
    Note: igraph uses integer vertex indices internally, but node names are preserved
    as the 'name' vertex attribute for compatibility.
    
    Args:
        graph: An EdgeCase Graph object
    
    Returns:
        An igraph.Graph object
        
    Raises:
        ImportError: If igraph is not installed
        
    Examples:
        >>> import edgecase as ec
        >>> from edgecase.igraph_interop import to_igraph
        >>> 
        >>> # Create an EdgeCase graph
        >>> G = ec.Graph(directed=True)
        >>> G.add_node("A", color="red", value=10)
        >>> G.add_node("B", color="blue", value=20)
        >>> G.add_edge("A", "B", weight=1.5, label="edge1")
        >>> 
        >>> # Convert to igraph
        >>> ig_graph = to_igraph(G)
        >>> print(ig_graph.vs["name"])  # ['A', 'B']
        >>> print(ig_graph.vs["color"])  # ['red', 'blue']
        >>> print(ig_graph.es["weight"])  # [1.5]
    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError(
            "igraph is required for graph conversion. "
            "Install it with: pip install igraph"
        )
    
    # Create igraph graph with appropriate directionality
    ig_graph = ig.Graph(directed=graph.is_directed)
    
    # Get all node IDs and create a mapping
    node_ids = list(graph.nodes())
    n_nodes = len(node_ids)
    
    # Add vertices
    ig_graph.add_vertices(n_nodes)
    
    # Set node labels as vertex "name" attribute (converting IDs to labels)
    node_labels = [graph.label_from_id(node_id) for node_id in node_ids]
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
    edges = graph.edges()  # Returns list of (id, source, target) tuples
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
    
    This function creates an EdgeCase graph from an igraph graph, preserving:
    - Graph directionality (directed/undirected)
    - Node attributes (vertex attributes)
    - Edge attributes (including weights)
    - Node names (from 'name' vertex attribute if present, otherwise use indices)
    
    Args:
        ig_graph: An igraph.Graph object
        weighted: If True, create a weighted graph. If None (default), auto-detect
                 by checking if edges have a 'weight' attribute.
    
    Returns:
        An EdgeCase Graph object
        
    Raises:
        ImportError: If igraph is not installed
        
    Examples:
        >>> import igraph as ig
        >>> from edgecase.igraph_interop import from_igraph
        >>> 
        >>> # Create an igraph graph
        >>> ig_graph = ig.Graph(directed=True)
        >>> ig_graph.add_vertices(2)
        >>> ig_graph.vs["name"] = ["A", "B"]
        >>> ig_graph.vs["color"] = ["red", "blue"]
        >>> ig_graph.add_edge(0, 1, weight=1.5, label="edge1")
        >>> 
        >>> # Convert to EdgeCase
        >>> G = from_igraph(ig_graph)
        >>> print(G.num_nodes())  # 2
        >>> print(G.num_edges())  # 1
        >>> print(G.get_node_attr("A", "color"))  # 'red'
    """
    try:
        import igraph  # noqa: F401
    except ImportError:
        raise ImportError(
            "igraph is required for graph conversion. "
            "Install it with: pip install igraph"
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
    # If 'name' attribute exists, use it; otherwise use vertex indices
    if "name" in ig_graph.vs.attributes():
        node_ids = ig_graph.vs["name"]
    else:
        node_ids = [str(i) for i in range(ig_graph.vcount())]
    
    # Get all vertex attribute names (excluding 'name' since we use it for node ID)
    vertex_attrs = [attr for attr in ig_graph.vs.attributes() if attr != "name"]
    
    # Add nodes with attributes
    for idx, node_id in enumerate(node_ids):
        # Convert node_id to string (EdgeCase uses string node IDs)
        node_id_str = str(node_id)
        
        # Collect vertex attributes
        node_attrs = {}
        for attr_name in vertex_attrs:
            attr_value = ig_graph.vs[idx][attr_name]
            if attr_value is not None:
                node_attrs[attr_name] = attr_value
        
        graph.add_node(node_id_str, **node_attrs)
    
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
