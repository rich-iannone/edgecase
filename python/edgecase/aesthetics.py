"""
Aesthetic mapping functions for graph visualization.

This module provides functions to map node attributes to visual properties.

1. Compute node attributes (e.g., centrality, community)
2. Map attributes to visual properties (color, size, shape)
3. Render with those visual attributes

Example workflow:
    >>> G = ec.Graph()
    >>> G.add_path(10)
    >>> 
    >>> # Compute attributes
    >>> communities = G.get_cmty_louvain()
    >>> betweenness = G.get_betweenness()
    >>> 
    >>> # Map to visual properties
    >>> G.colorize_node_attrs('community', communities, palette='vibrant')
    >>> G.rescale_node_attrs('betweenness', betweenness, 
    ...                       attr_to='size', to_min=5, to_max=20)
    >>> 
    >>> # Render (widget automatically reads 'color' and 'size' attributes)
    >>> widget = ec.GraphWidget(G)
    >>> widget.show()
"""

from typing import Dict, List, Optional, Union, Any
from edgecase.widget import COLOR_PALETTES


def colorize_node_attrs(
    graph,
    node_attr_from: str = None,
    values: Dict[str, Any] = None,
    node_attr_to: str = "color",
    palette: Union[str, List[str]] = "default",
    alpha: Optional[int] = None,
    nodes: Optional[List[str]] = None,
) -> "Graph":
    """
    Map node attribute values to colors.
    
    Sets a color attribute on nodes based on categorical or continuous values.
    For categorical data, assigns distinct colors from a palette.
    For continuous data, interpolates between colors.
    
    Parameters
    ----------
    graph : Graph
        The graph to modify
    node_attr_from : str, optional
        Attribute name to read values from (e.g., 'community', 'type')
        If None, must provide values dict
    values : dict, optional
        Dictionary mapping node IDs to values
        If None, reads from node_attr_from attribute
    node_attr_to : str, default "color"
        Attribute name to write colors to (e.g., 'color', 'fillcolor')
    palette : str or list, default "default"
        Color palette name ('default', 'vibrant', 'pastel', 'dark', 'tableau')
        or list of hex color strings
    alpha : int, optional
        Alpha transparency (0-100), not yet implemented
    nodes : list, optional
        Subset of nodes to colorize (default: all nodes)
        
    Returns
    -------
    Graph
        Self for method chaining
        
    Examples
    --------
    >>> # Color by community
    >>> communities = G.get_cmty_louvain()
    >>> G.colorize_node_attrs('community', communities, palette='vibrant')
    
    >>> # Color by node type
    >>> G.colorize_node_attrs(node_attr_from='type', palette='tableau')
    
    >>> # Custom color list
    >>> G.colorize_node_attrs('status', values={'A': 'active', 'B': 'inactive'},
    ...                       palette=['#ff0000', '#00ff00'])
    """
    if nodes is None:
        nodes = graph.node_ids()
    
    # Get values to colorize
    if values is None:
        if node_attr_from is None:
            raise ValueError("Must provide either node_attr_from or values")
        values = graph.get_node_attrs(node_attr_from, nodes=nodes)
    
    if not values:
        return graph
    
    # Get unique values and assign colors
    unique_values = sorted(list(set(values.values())))
    
    # Get color palette
    if isinstance(palette, str):
        colors = COLOR_PALETTES.get(palette, COLOR_PALETTES['default'])
    else:
        colors = palette
    
    # Create color mapping
    n_values = len(unique_values)
    if n_values <= len(colors):
        color_list = colors[:n_values]
    else:
        # Cycle through colors if we have more values than colors
        color_list = (colors * (n_values // len(colors) + 1))[:n_values]
    
    color_map = dict(zip(unique_values, color_list))
    
    # Set colors on nodes
    for node_id, value in values.items():
        color = color_map.get(value, '#cccccc')
        graph._graph.set_node_attr(node_id, node_attr_to, color)
    
    return graph


def rescale_node_attrs(
    graph,
    node_attr_from: str = None,
    values: Dict[str, Union[int, float]] = None,
    node_attr_to: str = "size",
    to_lower_bound: float = 0.5,
    to_upper_bound: float = 1.0,
    nodes: Optional[List[str]] = None,
) -> "Graph":
    """
    Rescale node attribute values to a range.
    
    Normalizes numeric values to a specified range, useful for sizing nodes
    by centrality, degree, or other continuous metrics.
    
    Parameters
    ----------
    graph : Graph
        The graph to modify
    node_attr_from : str, optional
        Attribute name to read values from (e.g., 'betweenness', 'degree')
        If None, must provide values dict
    values : dict, optional
        Dictionary mapping node IDs to numeric values
        If None, reads from node_attr_from attribute
    node_attr_to : str, default "size"
        Attribute name to write rescaled values to
    to_lower_bound : float, default 0.5
        Minimum value in output range
    to_upper_bound : float, default 1.0
        Maximum value in output range
    nodes : list, optional
        Subset of nodes to rescale (default: all nodes)
        
    Returns
    -------
    Graph
        Self for method chaining
        
    Examples
    --------
    >>> # Size by betweenness centrality
    >>> betweenness = G.get_betweenness()
    >>> G.rescale_node_attrs('betweenness', betweenness, 
    ...                       attr_to='size', to_min=5, to_max=20)
    
    >>> # Scale by degree
    >>> degrees = {n: len(G.neighbors(n)) for n in G.nodes()}
    >>> G.rescale_node_attrs(values=degrees, attr_to='width', 
    ...                       to_min=1, to_max=10)
    """
    if nodes is None:
        nodes = graph.node_ids()
    
    # Get values to rescale
    if values is None:
        if node_attr_from is None:
            raise ValueError("Must provide either node_attr_from or values")
        values = graph.get_node_attrs(node_attr_from, nodes=nodes)
    
    if not values:
        return graph
    
    # Get min/max for normalization
    value_list = list(values.values())
    min_val = min(value_list)
    max_val = max(value_list)
    
    # Avoid division by zero
    value_range = max_val - min_val if max_val > min_val else 1.0
    output_range = to_upper_bound - to_lower_bound
    
    # Rescale and set on nodes
    for node_id, value in values.items():
        normalized = (value - min_val) / value_range
        rescaled = to_lower_bound + (normalized * output_range)
        graph._graph.set_node_attr(node_id, node_attr_to, rescaled)
    
    return graph


def set_node_attr_to_display(
    graph,
    attr: Optional[str] = None,
) -> "Graph":
    """
    Set which node attribute to display in tooltips/labels.
    
    Parameters
    ----------
    graph : Graph
        The graph to modify
    attr : str, optional
        Attribute name to display. If None, shows node ID.
        
    Returns
    -------
    Graph
        Self for method chaining
        
    Examples
    --------
    >>> G.set_node_attr_to_display('label')
    >>> G.set_node_attr_to_display(None)  # Reset to node IDs
    """
    graph._display_attr = attr
    return graph
