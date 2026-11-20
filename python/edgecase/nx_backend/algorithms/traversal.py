"""Breadth- and depth-first traversal, reachability, DAG ordering and isolated nodes.

The traversals run on edgecase's `bfs_search`/`dfs_labeled_search` kernels with
`adjacency_order=True`, which visit neighbors in NetworkX's order, so results come out in
the same order as NetworkX's. Calls with `sort_neighbors` (a Python callable) are left to
NetworkX, as is the deprecated `bfs_predecessors`.
"""

from __future__ import annotations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "ancestors",
    "bfs_edges",
    "bfs_layers",
    "bfs_successors",
    "bfs_tree",
    "descendants",
    "descendants_at_distance",
    "dfs_edges",
    "dfs_labeled_edges",
    "dfs_postorder_nodes",
    "dfs_predecessors",
    "dfs_preorder_nodes",
    "dfs_successors",
    "dfs_tree",
    "is_directed_acyclic_graph",
    "is_isolate",
    "isolates",
    "number_of_isolates",
    "topological_generations",
    "topological_sort",
]


def _node_error(G, n):
    import networkx as nx

    kind = "digraph" if G.is_directed() else "graph"
    return nx.NetworkXError(f"The node {n} is not in the {kind}.")


def _id_of(G, n):
    """The node's edgecase ID, raising NetworkX's error for a missing node."""
    try:
        return G.key_to_id[n]
    except KeyError:
        raise _node_error(G, n) from None


def _no_sort_neighbors(*args, sort_neighbors=None, **kwargs):
    if sort_neighbors is not None:
        return "sort_neighbors is a Python function"
    return True


def _bfs_can_run(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    if sort_neighbors is not None:
        return "sort_neighbors is a Python function"
    if reverse and G.is_directed():
        import networkx as nx

        config = getattr(nx.config.backends, "edgecase", None)
        if not getattr(config, "preserve_pred_order", True):
            return "predecessor order is not recorded (preserve_pred_order=False)"
    return True


def _bfs_successors_can_run(G, source, depth_limit=None, sort_neighbors=None):
    return _no_sort_neighbors(sort_neighbors=sort_neighbors)


# Breadth-first search


def _bfs_edge_ids(G, source, reverse=False, depth_limit=None):
    from edgecase._core import bfs_search

    source_id = _id_of(G, source)
    if depth_limit is not None and depth_limit <= 0:
        return []
    rg = G.reverse_rust_graph() if reverse and G.is_directed() else G.rust_graph()
    edges, _ = bfs_search(rg, [source_id], depth_limit, "out", adjacency_order=True)
    return edges


def _bfs_edges(G, source, reverse=False, depth_limit=None):
    nodes = G.nodes
    for u, v in _bfs_edge_ids(G, source, reverse, depth_limit):
        yield nodes[u - 1], nodes[v - 1]


@networkx_algorithm(linear_time=True, can_run=_bfs_can_run)
def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    yield from _bfs_edges(G, source, reverse, depth_limit)


@networkx_algorithm(linear_time=True, can_run=_bfs_can_run)
def bfs_tree(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    import networkx as nx

    T = nx.DiGraph()
    T.add_node(source)
    T.add_edges_from(_bfs_edges(G, source, reverse, depth_limit))
    return T


@networkx_algorithm(linear_time=True, can_run=_bfs_successors_can_run)
def bfs_successors(G, source, depth_limit=None, sort_neighbors=None):
    parent = source
    children = []
    for p, c in _bfs_edges(G, source, depth_limit=depth_limit):
        if p == parent:
            children.append(c)
            continue
        yield (parent, children)
        children = [c]
        parent = p
    yield (parent, children)


def _bfs_layers(G, sources):
    import networkx as nx

    from edgecase._core import bfs_search

    if sources in G:
        sources = [sources]
    current_layer = list(set(sources))  # NetworkX starts from a set of the sources
    for source in current_layer:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")
    if not current_layer:
        return
    key_to_id = G.key_to_id
    _, layers = bfs_search(
        G.rust_graph(), [key_to_id[s] for s in current_layer], None, "out", adjacency_order=True
    )
    nodes = G.nodes
    yield current_layer
    for layer in layers[1:]:
        yield [nodes[i - 1] for i in layer]


@networkx_algorithm(linear_time=True)
def bfs_layers(G, sources):
    yield from _bfs_layers(G, sources)


@networkx_algorithm(linear_time=True)
def descendants_at_distance(G, source, distance):
    import networkx as nx

    from edgecase._core import bfs_search

    if source not in G:
        raise nx.NetworkXError(f"The node {source} is not in the graph.")
    if distance < 0:
        return set()
    _, layers = bfs_search(
        G.rust_graph(), [G.key_to_id[source]], distance, "out", adjacency_order=True
    )
    if len(layers) <= distance:
        return set()
    nodes = G.nodes
    return {nodes[i - 1] for i in layers[distance]}


# Reachability


@networkx_algorithm(linear_time=True)
def descendants(G, source):
    from edgecase._core import descendants as rust_descendants

    nodes = G.nodes
    return {nodes[i - 1] for i in rust_descendants(G.rust_graph(), _id_of(G, source))}


@networkx_algorithm(linear_time=True)
def ancestors(G, source):
    from edgecase._core import ancestors as rust_ancestors

    nodes = G.nodes
    return {nodes[i - 1] for i in rust_ancestors(G.rust_graph(), _id_of(G, source))}


# Depth-first search


def _dfs_labeled(G, source, depth_limit):
    """NetworkX's labeled DFS steps as (u, v, label) with node keys."""
    from edgecase._core import dfs_labeled_search

    nodes = G.nodes
    if source is None:
        source_id = None
    else:
        source_id = G.key_to_id.get(source) if source in G else None
        if source_id is None:
            # NetworkX announces the start before looking up its neighbors
            yield source, source, "forward"
            raise _node_error(G, source)
    if depth_limit is not None and depth_limit < 1:
        depth_limit = 1  # NetworkX explores the start's neighbors even then
    for u, v, label in dfs_labeled_search(
        G.rust_graph(), source_id, depth_limit, "out", adjacency_order=True
    ):
        yield nodes[u - 1], nodes[v - 1], label


def _dfs_edges(G, source, depth_limit):
    if source is not None and source not in G:
        raise _node_error(G, source)
    for u, v, label in _dfs_labeled(G, source, depth_limit):
        if label == "forward" and u != v:
            yield u, v


@networkx_algorithm(linear_time=True, can_run=_no_sort_neighbors)
def dfs_labeled_edges(G, source=None, depth_limit=None, *, sort_neighbors=None):
    yield from _dfs_labeled(G, source, depth_limit)


@networkx_algorithm(linear_time=True, can_run=_no_sort_neighbors)
def dfs_edges(G, source=None, depth_limit=None, *, sort_neighbors=None):
    yield from _dfs_edges(G, source, depth_limit)


@networkx_algorithm(linear_time=True, can_run=_no_sort_neighbors)
def dfs_tree(G, source=None, depth_limit=None, *, sort_neighbors=None):
    import networkx as nx

    T = nx.DiGraph()
    if source is None:
        T.add_nodes_from(G.nodes)
    else:
        T.add_node(source)
    T.add_edges_from(_dfs_edges(G, source, depth_limit))
    return T


@networkx_algorithm(linear_time=True, can_run=_no_sort_neighbors)
def dfs_predecessors(G, source=None, depth_limit=None, *, sort_neighbors=None):
    return {t: s for s, t in _dfs_edges(G, source, depth_limit)}


@networkx_algorithm(linear_time=True, can_run=_no_sort_neighbors)
def dfs_successors(G, source=None, depth_limit=None, *, sort_neighbors=None):
    successors = {}
    for s, t in _dfs_edges(G, source, depth_limit):
        successors.setdefault(s, []).append(t)
    return successors


@networkx_algorithm(linear_time=True, can_run=_no_sort_neighbors)
def dfs_preorder_nodes(G, source=None, depth_limit=None, *, sort_neighbors=None):
    return (v for u, v, d in _dfs_labeled(G, source, depth_limit) if d == "forward")


@networkx_algorithm(linear_time=True, can_run=_no_sort_neighbors)
def dfs_postorder_nodes(G, source=None, depth_limit=None, *, sort_neighbors=None):
    return (v for u, v, d in _dfs_labeled(G, source, depth_limit) if d == "reverse")


# Directed acyclic graphs


def _generations(G):
    """(generations as lists of node keys, complete) in NetworkX's order."""
    import networkx as nx

    from edgecase._core import topological_generations_partial

    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")
    generations, complete = topological_generations_partial(G.rust_graph())
    nodes = G.nodes
    return [[nodes[i - 1] for i in generation] for generation in generations], complete


def _unfeasible():
    import networkx as nx

    return nx.NetworkXUnfeasible("Graph contains a cycle or graph changed during iteration")


@networkx_algorithm(linear_time=True)
def topological_generations(G):
    generations, complete = _generations(G)
    yield from generations
    if not complete:
        raise _unfeasible()


@networkx_algorithm(linear_time=True)
def topological_sort(G):
    generations, complete = _generations(G)
    for generation in generations:
        yield from generation
    if not complete:
        raise _unfeasible()


@networkx_algorithm(linear_time=True)
def is_directed_acyclic_graph(G):
    if not G.is_directed():
        return False
    from edgecase._core import topological_generations_partial

    return topological_generations_partial(G.rust_graph())[1]


# Isolated nodes


def _isolated_ids(G):
    connected = set(G.src)
    connected.update(G.dst)
    return [i for i in range(1, len(G.nodes) + 1) if i not in connected]


@networkx_algorithm(linear_time=True)
def isolates(G):
    nodes = G.nodes
    return (nodes[i - 1] for i in _isolated_ids(G))


@networkx_algorithm(linear_time=True)
def number_of_isolates(G):
    return len(_isolated_ids(G))


def _is_isolate_can_run(G, n):
    return True if n in G else "NetworkX's own error handling for a node not in the graph"


@networkx_algorithm(linear_time=True, can_run=_is_isolate_can_run)
def is_isolate(G, n):
    node_id = G.key_to_id[n]
    if node_id in G.src:
        return False
    return node_id not in G.dst
