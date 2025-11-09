"""
NetworkX interoperability module for EdgeCase.

This module provides bidirectional conversion between EdgeCase and NetworkX graph objects,
enabling validation of EdgeCase algorithms against NetworkX and providing interoperability
with the NetworkX ecosystem.

Functions:
    to_networkx: Convert an EdgeCase Graph to a NetworkX graph
    from_networkx: Convert a NetworkX graph to an EdgeCase Graph
"""

from typing import Optional
import warnings


def to_networkx(graph, multigraph: bool = False):
    """Convert an EdgeCase Graph to a NetworkX graph.
    
    This function creates a NetworkX graph from an EdgeCase graph, preserving:
    - Graph directionality (directed/undirected)
    - Node attributes
    - Edge attributes (including weights)
    - Multiple edges between nodes (if multigraph=True)
    
    Args:
        graph: An EdgeCase Graph object
        multigraph: If True, create a MultiGraph/MultiDiGraph to preserve edge IDs
                   and multiple edges. If False (default), create a Graph/DiGraph.
                   Note: When False and multiple edges exist, only the last edge
                   is preserved.
    
    Returns:
        A NetworkX Graph, DiGraph, MultiGraph, or MultiDiGraph object
        
    Raises:
        ImportError: If NetworkX is not installed
        
    Examples:
        >>> import edgecase as ec
        >>> from edgecase.networkx import to_networkx
        >>> 
        >>> # Create an EdgeCase graph
        >>> G = ec.Graph(directed=True)
        >>> G.add_node("A", color="red", value=10)
        >>> G.add_node("B", color="blue", value=20)
        >>> G.add_edge("A", "B", weight=1.5, label="edge1")
        >>> 
        >>> # Convert to NetworkX
        >>> nx_graph = to_networkx(G)
        >>> print(nx_graph.nodes["A"])  # {'color': 'red', 'value': 10}
        >>> print(nx_graph["A"]["B"])  # {'weight': 1.5, 'label': 'edge1'}
        
        >>> # For graphs with multiple edges, use multigraph=True
        >>> G.add_edge("A", "B", weight=2.0, label="edge2")
        >>> nx_multi = to_networkx(G, multigraph=True)
        >>> len(nx_multi["A"]["B"])  # 2 edges
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for graph conversion. "
            "Install it with: pip install networkx"
        )
    
    # Determine the appropriate NetworkX graph type
    if multigraph:
        nx_graph = nx.MultiDiGraph() if graph.is_directed else nx.MultiGraph()
    else:
        nx_graph = nx.DiGraph() if graph.is_directed else nx.Graph()
    
    # Add nodes with attributes
    for node_id in graph.nodes():
        node = graph.get_node(node_id)
        node_attrs = {}
        if node and node.attributes:
            # Convert PyDict to regular dict
            node_attrs = dict(node.attributes)
        
        # Use label as NetworkX node ID if it exists, otherwise use the numeric ID
        nx_node_id = graph.label_from_id(node_id) or node_id
        nx_graph.add_node(nx_node_id, **node_attrs)
    
    # Add edges with attributes
    edges = graph.edges()  # Returns list of (id, source, target) tuples
    
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
            nx_source = graph.label_from_id(source) or source
            nx_target = graph.label_from_id(target) or target
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
                UserWarning
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
            nx_source = graph.label_from_id(source) or source
            nx_target = graph.label_from_id(target) or target
            nx_graph.add_edge(nx_source, nx_target, **edge_attrs)
    
    return nx_graph


def from_networkx(nx_graph, weighted: Optional[bool] = None):
    """Convert a NetworkX graph to an EdgeCase Graph.
    
    This function creates an EdgeCase graph from a NetworkX graph, preserving:
    - Graph directionality (directed/undirected)
    - Node attributes
    - Edge attributes (including weights)
    - Multiple edges (for MultiGraph/MultiDiGraph)
    
    Args:
        nx_graph: A NetworkX Graph, DiGraph, MultiGraph, or MultiDiGraph object
        weighted: If True, create a weighted graph. If None (default), auto-detect
                 by checking if any edge has a 'weight' attribute.
    
    Returns:
        An EdgeCase Graph object
        
    Raises:
        ImportError: If NetworkX is not installed
        
    Examples:
        >>> import networkx as nx
        >>> from edgecase.networkx import from_networkx
        >>> 
        >>> # Create a NetworkX graph
        >>> nx_graph = nx.DiGraph()
        >>> nx_graph.add_node("A", color="red", value=10)
        >>> nx_graph.add_node("B", color="blue", value=20)
        >>> nx_graph.add_edge("A", "B", weight=1.5, label="edge1")
        >>> 
        >>> # Convert to EdgeCase
        >>> G = from_networkx(nx_graph)
        >>> print(G.num_nodes())  # 2
        >>> print(G.num_edges())  # 1
        >>> print(G.get_node_attr("A", "color"))  # 'red'
        
        >>> # Works with MultiGraphs too
        >>> nx_multi = nx.MultiGraph()
        >>> nx_multi.add_edge("X", "Y", weight=1.0)
        >>> nx_multi.add_edge("X", "Y", weight=2.0)
        >>> G = from_networkx(nx_multi)
        >>> print(G.num_edges())  # 2
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for graph conversion. "
            "Install it with: pip install networkx"
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
            if 'weight' in data:
                weighted = True
                break
    
    # Create EdgeCase graph
    graph = Graph(directed=directed, weighted=weighted)
    
    # Add nodes with attributes
    for node_id, node_attrs in nx_graph.nodes(data=True):
        # Convert node_id to string (EdgeCase uses string node IDs)
        node_id_str = str(node_id)
        graph.add_node(node_id_str, **node_attrs)
    
    # Add edges with attributes
    if nx_graph.is_multigraph():
        # MultiGraph: iterate over all edges with keys
        for u, v, key, edge_attrs in nx_graph.edges(keys=True, data=True):
            u_str = str(u)
            v_str = str(v)
            
            # Extract weight if present
            weight = edge_attrs.get('weight', None)
            
            # Remove weight from attrs dict to avoid duplication
            edge_attrs_copy = edge_attrs.copy()
            edge_attrs_copy.pop('weight', None)
            
            # If edge has an 'edgecase_edge_id', use it
            edge_id = edge_attrs_copy.pop('edgecase_edge_id', None)
            
            graph.add_edge(u_str, v_str, weight=weight, edge_id=edge_id, **edge_attrs_copy)
    else:
        # Simple Graph: iterate over edges normally
        for u, v, edge_attrs in nx_graph.edges(data=True):
            u_str = str(u)
            v_str = str(v)
            
            # Extract weight if present
            weight = edge_attrs.get('weight', None)
            
            # Remove weight from attrs dict to avoid duplication
            edge_attrs_copy = edge_attrs.copy()
            edge_attrs_copy.pop('weight', None)
            
            graph.add_edge(u_str, v_str, weight=weight, **edge_attrs_copy)
    
    return graph


__all__ = ["to_networkx", "from_networkx"]
