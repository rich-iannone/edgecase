import importlib.util
import warnings
from typing import Optional


def to_networkx(graph, multigraph: bool = False):
    """Convert an EdgeCase Graph to a NetworkX graph.

    Creates a NetworkX graph preserving directionality, node/edge attributes, and optionally
    multiple edges between nodes. Useful for validation against NetworkX algorithms and leveraging
    the NetworkX ecosystem.

    Nodes are added in ID order, each named by its label (or by its numeric ID if it has none; if
    some label is that number as a string, by an `n`-prefixed string instead), and edges in ID
    order. A weight is stored as the `weight` edge attribute and the graph name, if
    any, as `nx_graph.graph["name"]`.

    Parameters
    ----------
    graph
        The EdgeCase Graph object to convert.
    multigraph
        If `True`, create a MultiGraph/MultiDiGraph preserving edge IDs and multiple edges. If
        `False`, create a Graph/DiGraph. When `False` and multiple edges exist, only the last edge
        is preserved (with a warning).

    Returns
    -------
    networkx.Graph or networkx.DiGraph or networkx.MultiGraph or networkx.MultiDiGraph
        A NetworkX graph with the same structure and attributes.

    Raises
    ------
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

    # Returns {'color': 'red', 'value': 10}
    nx_graph.nodes["A"]

    # Returns {'weight': 1.5, 'label': 'edge1'}
    nx_graph["A"]["B"]

    # For graphs with multiple edges, use multigraph=True
    graph.add_edge("A", "B", weight=2.0, label="edge2")
    nx_multi = to_networkx(graph, multigraph=True)

    # Returns 2 (number of edges)
    len(nx_multi["A"]["B"])
    ```

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
    if graph.get_graph_name():
        nx_graph.graph["name"] = graph.get_graph_name()

    # Use label as NetworkX node ID if it exists, otherwise use the numeric ID (or, if a label is
    # that number as a string, a string that no label is, so `from_networkx()` can read it back)
    from edgecase.io import _node_keys

    node_ids = sorted(graph.get_node_ids(limit=None))
    labels = {node_id: graph.get_label_from_id(node_id) for node_id in node_ids}
    keys = _node_keys([(node_id, label, {}) for node_id, label in labels.items()])
    nx_ids = {}
    for node_id, label in labels.items():
        if label is None:
            label = node_id if keys[node_id] == str(node_id) else keys[node_id]
        nx_ids[node_id] = label

    # Add nodes with attributes (as `(node, attrs)` pairs, so no attribute name can collide
    # with an argument of `add_node()`)
    nx_graph.add_nodes_from(
        (nx_ids[node_id], dict(graph.get_node(node_id).attributes or {})) for node_id in node_ids
    )

    edges = sorted(graph.get_edges_as_tuples())  # (id, source, target) tuples, in ID order

    def edge_attrs_of(edge_id):
        edge = graph.get_edge(edge_id)
        edge_attrs = dict(edge.attributes or {})
        if edge.weight is not None:
            edge_attrs["weight"] = edge.weight
        return edge_attrs

    if multigraph:
        # MultiGraph: preserve edge IDs (as edge keys) and multiple edges
        nx_graph.add_edges_from(
            (
                nx_ids[source],
                nx_ids[target],
                edge_id,
                {**edge_attrs_of(edge_id), "edgecase_edge_id": edge_id},
            )
            for edge_id, source, target in edges
        )
    else:
        # Simple Graph: warn if multiple edges exist (either way round, when undirected)
        directed = graph.is_directed()
        pairs = [
            (source, target) if directed else (min(source, target), max(source, target))
            for _, source, target in edges
        ]
        if len(set(pairs)) < len(pairs):
            warnings.warn(
                "Graph contains multiple edges between some node pairs. "
                "Only the last edge will be preserved. Use multigraph=True "
                "to preserve all edges.",
                UserWarning,
            )

        nx_graph.add_edges_from(
            (nx_ids[source], nx_ids[target], edge_attrs_of(edge_id))
            for edge_id, source, target in edges
        )

    return nx_graph


def from_networkx(nx_graph, weighted: Optional[bool] = None):
    """Convert a NetworkX graph to an EdgeCase Graph.

    Creates an EdgeCase graph from NetworkX, preserving directionality, node/edge attributes, and
    multiple edges (for MultiGraph/MultiDiGraph). Auto-detects weighted graphs unless explicitly
    specified.

    Each node is labeled with `str(node)`. Every node attribute is kept as an attribute, including
    ones named `label` or `type`. A numeric `weight` becomes the edge weight and every other edge
    attribute is kept; the edge keys of a MultiGraph, and the `edgecase_edge_id` attribute that
    `to_networkx()` adds, are dropped. A `name` in `nx_graph.graph` becomes the graph name. The
    graph is built as a single step.

    Parameters
    ----------
    nx_graph
        The NetworkX graph to convert.
    weighted
        If `True`, create a weighted graph. If `None` (default), auto-detect by checking if any edge
        has a numeric 'weight' attribute. If `False`, edge weights are dropped.

    Returns
    -------
    Graph
        An EdgeCase Graph object with the same structure and attributes.

    Raises
    ------
    ImportError
        If NetworkX is not installed.
    ValueError
        If two nodes have the same `str()`, such as `1` and `"1"`.

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

    # Returns 2
    graph.count_nodes()

    # Returns 1
    graph.count_edges()

    node = graph.get_node(graph.get_id_from_label("A"))

    # Returns 'red'
    node.attributes["color"]

    # Works with MultiGraphs too
    nx_multi = nx.MultiGraph()
    nx_multi.add_edge("X", "Y", weight=1.0)
    nx_multi.add_edge("X", "Y", weight=2.0)
    graph = from_networkx(nx_multi)

    # Returns 2
    graph.count_edges()
    ```

    """
    # Only availability matters here: the conversion uses methods on `nx_graph` itself.
    if importlib.util.find_spec("networkx") is None:
        raise ImportError(
            "NetworkX is required for graph conversion. Install it with: pip install networkx"
        )

    from edgecase.io import _build_graph, _code, _numeric_weight

    # Nodes keep their NetworkX attributes as given (`type`, `label` and the like are plain
    # attributes too); `str(node)` becomes the label
    nodes = [(node, str(node), dict(node_attrs)) for node, node_attrs in nx_graph.nodes(data=True)]

    # A numeric `weight` becomes the edge weight; the edge key, and the `edgecase_edge_id` that
    # `to_networkx()` adds, are dropped
    edges = []
    for u, v, edge_attrs in nx_graph.edges(data=True):
        edge_attrs = dict(edge_attrs)
        edge_attrs.pop("edgecase_edge_id", None)
        weight = _numeric_weight(edge_attrs.get("weight"))
        if weight is not None:
            del edge_attrs["weight"]
        edges.append((u, v, weight, edge_attrs))

    # Auto-detect if graph is weighted
    if weighted is None:
        weighted = any(weight is not None for _, _, weight, _ in edges)

    name = nx_graph.graph.get("name")
    return _build_graph(
        nodes,
        edges,
        nx_graph.is_directed(),
        weighted,
        name if isinstance(name, str) and name else None,
        _code("from_networkx", nx_graph),
    )


__all__ = ["to_networkx", "from_networkx"]
