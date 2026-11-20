"""The edgecase backend's graph type and its conversions to and from NetworkX graphs.

`EdgecaseGraph` stores a NetworkX graph's structure as flat arrays of *arcs*, the
`(node, neighbor)` entries of NetworkX's adjacency dict in its iteration order: for an
undirected graph every edge appears twice (once from each end; a self-loop once). Nodes
are numbered `1..n` in NetworkX node order (edgecase reserves node ID 0), so a node's
key is `nodes[id - 1]`.

Keeping NetworkX's adjacency order makes the conversion lossless for the attributes that
were converted: converting back gives a graph whose nodes, neighbors, predecessors and edge
keys iterate in the original order, which matters for every order-dependent result
(traversals, tie-breaking). A directed graph's predecessor order is recorded separately
(it can differ from the order of the arcs into each node); setting
`nx.config.backends.edgecase.preserve_pred_order = False` skips that, saving 10-15% of the
conversion time, and predecessors then come back in arc order.

The edgecase Rust graph is built from these arrays in one call, on first use, once per
edge attribute that is used as the weight; its neighbor lists keep NetworkX's order.
"""

from __future__ import annotations

import os
from itertools import chain, islice, repeat
from operator import methodcaller

__all__ = ["EdgecaseGraph", "from_networkx", "to_networkx"]


class _Missing:
    """Marks an attribute value absent from an edge or node (default `None`)."""

    __slots__ = ()

    def __repr__(self):
        return "MISSING"

    def __reduce__(self):
        return "MISSING"


MISSING = _Missing()

_values = methodcaller("values")


class EdgecaseGraph:
    """A graph converted from NetworkX, backed by edgecase's Rust core.

    Attributes
    ----------
    nodes : list
        The NetworkX node keys; the node with edgecase ID `i` is `nodes[i - 1]`.
    key_to_id : dict
        Node key -> edgecase node ID.
    src, dst : list of int
        Arc endpoints (edgecase node IDs) in NetworkX adjacency order.
    keys : list or None
        The edge key of each arc (multigraphs only).
    edge_values : dict
        Edge attribute name -> value per arc (`MISSING` where an edge lacks the attribute
        and its default is `None`), for the attributes the dispatcher asked for.
    edge_dicts : list of dict or None
        The full attribute dict of each arc, when all edge attributes were preserved.
    node_values, node_dicts
        The same for nodes, per node in node order.
    graph : dict
        Graph attributes (only when they were preserved).
    """

    __networkx_backend__ = "edgecase"

    def __init__(
        self,
        *,
        nodes,
        key_to_id,
        src,
        dst,
        directed,
        multigraph,
        keys=None,
        edge_values=None,
        edge_defaults=None,
        edge_dicts=None,
        node_values=None,
        node_defaults=None,
        node_dicts=None,
        graph=None,
        nx_class=None,
        digraph_storage=None,
        pred_order=None,
    ):
        self.nodes = nodes
        self.key_to_id = key_to_id
        self.src = src
        self.dst = dst
        self.directed = directed
        self.multigraph = multigraph
        self.keys = keys
        self.edge_values = {} if edge_values is None else edge_values
        # attribute -> default used when converting (`None`: absent values are MISSING)
        self.edge_defaults = {} if edge_defaults is None else edge_defaults
        self.edge_dicts = edge_dicts
        self.node_values = {} if node_values is None else node_values
        self.node_defaults = {} if node_defaults is None else node_defaults
        self.node_dicts = node_dicts
        self.graph = {} if graph is None else graph
        self.nx_class = nx_class
        # Whether arcs are stored per direction (as in `nx.DiGraph`), which can differ from
        # `directed` for a DiGraph subclass that reports itself undirected (PlanarEmbedding)
        self.digraph_storage = directed if digraph_storage is None else digraph_storage
        # (nodes, predecessor counts, predecessors) as node keys: each node's predecessors in
        # NetworkX's order (which can differ from the order of the arcs into it), if recorded
        self.pred_order = pred_order
        self._rust_graphs = {}
        self._edge_arcs = None
        # The dispatcher caches conversions of this graph (to NetworkX) here; the graph is
        # never mutated, so the cache stays valid
        self.__networkx_cache__ = {}

    # A small slice of the NetworkX graph API, for introspection and for NetworkX's test
    # helpers
    def is_directed(self):
        return self.directed

    def is_multigraph(self):
        return self.multigraph

    def __len__(self):
        return len(self.nodes)

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        if self.digraph_storage:
            return len(self.src)
        selfloops = sum(map(int.__eq__, self.src, self.dst))
        return (len(self.src) + selfloops) // 2

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, key):
        try:
            return key in self.key_to_id
        except TypeError:
            return False

    def __repr__(self):
        kind = ("Multi" if self.multigraph else "") + ("DiGraph" if self.directed else "Graph")
        return (
            f"<EdgecaseGraph ({kind}) with {self.number_of_nodes()} nodes and "
            f"{self.number_of_edges()} edges>"
        )

    # Arcs <-> edgecase edges
    @property
    def edge_arcs(self):
        """The arc behind each edgecase edge: edge ID `e` is arc `edge_arcs[e - 1]`."""
        if self._edge_arcs is None:
            if self.directed:
                self._edge_arcs = range(len(self.src))
            else:
                self._edge_arcs = [i for i, (s, d) in enumerate(zip(self.src, self.dst)) if s <= d]
        return self._edge_arcs

    def edge_values_of(self, attr, default=None):
        """The value of edge attribute `attr` per arc, with `default` where it is absent.

        Raises KeyError if the attribute was not converted.
        """
        values = self.edge_values.get(attr)
        if values is not None:
            if default is MISSING or self.edge_defaults.get(attr) is not None:
                return values  # (no value is MISSING)
            return [default if v is MISSING else v for v in values]
        if self.edge_dicts is not None:
            return list(map(methodcaller("get", attr, default), self.edge_dicts))
        if not self.src:
            return []
        raise KeyError(f"edge attribute {attr!r} was not converted to the edgecase graph")

    def node_values_of(self, attr, default=None):
        """The value of node attribute `attr` per node, with `default` where it is absent."""
        values = self.node_values.get(attr)
        if values is not None:
            if default is MISSING or self.node_defaults.get(attr) is not None:
                return values  # (no value is MISSING)
            return [default if v is MISSING else v for v in values]
        if self.node_dicts is not None:
            return list(map(methodcaller("get", attr, default), self.node_dicts))
        if not self.nodes:
            return []
        raise KeyError(f"node attribute {attr!r} was not converted to the edgecase graph")

    def rust_graph(self, weight=None, default=1.0):
        """The edgecase Rust graph (`edgecase._core.Graph`), built on first use.

        With `weight`, each edge's native weight is its value of that attribute (or
        `default` where it has none; `default=None` leaves those edges without a weight).

        Raises NotImplementedError when the values cannot be edgecase weights (NaN, `None`
        where a default applies, or not numbers), so NetworkX runs the call itself and gives
        its own answer or error.

        The Rust graph is a fixed function of this (immutable) graph, so it is kept here per
        `(weight, default)`; NetworkX caches the converted graph itself on the input graph.
        """
        cache_key = (weight, default)
        rg = self._rust_graphs.get(cache_key)
        if rg is None:
            from edgecase._core import Graph as RustGraph

            # (Rust reads each value as a float: ints, bools and anything with `__float__`)
            weights = None if weight is None else self.edge_values_of(weight, default)
            try:
                rg = RustGraph.from_edge_arrays(
                    len(self.nodes),
                    self.src,
                    self.dst,
                    weights,
                    directed=self.directed,
                    symmetric=not self.directed,
                    require_weights=weights is not None and default is not None,
                )
            except (TypeError, ValueError) as exc:
                if weights is None:
                    raise
                raise NotImplementedError(
                    f"edgecase cannot use edge attribute {weight!r} as weights: {exc}"
                ) from exc
            self._rust_graphs[cache_key] = rg
        return rg

    def reverse_rust_graph(self):
        """A directed Rust graph with every arc reversed, built on first use.

        Each node's successors in it are the node's predecessors in NetworkX's order when
        that was recorded (`pred_order`), else in arc order. Unweighted.
        """
        rg = self._rust_graphs.get("reverse")
        if rg is None:
            from edgecase._core import Graph as RustGraph

            if not self.directed:
                raise ValueError("only directed graphs have a reverse")
            if self.pred_order is not None:
                pred_nodes, counts, preds = self.pred_order
                get_id = self.key_to_id.__getitem__
                src = list(chain.from_iterable(map(repeat, map(get_id, pred_nodes), counts)))
                dst = list(map(get_id, preds))
                if self.multigraph:
                    # one arc per key, as for successors
                    src, dst = self._expand_pred_keys(src, dst)
            else:
                src, dst = self.dst, self.src
            rg = RustGraph.from_edge_arrays(len(self.nodes), src, dst, None, directed=True)
            self._rust_graphs["reverse"] = rg
        return rg

    def _expand_pred_keys(self, src, dst):
        """Repeat each (node, predecessor) pair once per parallel edge."""
        counts = {}
        for u, v in zip(self.src, self.dst):
            counts[(u, v)] = counts.get((u, v), 0) + 1
        multiplicity = [counts[(u, v)] for v, u in zip(src, dst)]
        return (
            list(chain.from_iterable(map(repeat, src, multiplicity))),
            list(chain.from_iterable(map(repeat, dst, multiplicity))),
        )

    # Results keyed by edgecase node IDs -> keyed by NetworkX node keys
    def key_of(self, node_id):
        return self.nodes[node_id - 1]

    def keys_of(self, node_ids):
        nodes = self.nodes
        return [nodes[i - 1] for i in node_ids]

    def dict_by_key(self, values_by_id):
        """`{node key: value}` from `{node ID: value}`, in node order."""
        nodes = self.nodes
        return {nodes[i - 1]: values_by_id[i] for i in range(1, len(nodes) + 1)}

    def to_networkx(self):
        return to_networkx(self)


def _overrides_adjacency(G, nx):
    """True if `G`'s class computes its adjacency instead of reading `G._adj`.

    (NetworkX's own `_AntiGraph`, for example, presents the complement of what it stores.)
    """
    base_classes = {nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph}
    for klass in type(G).__mro__:
        if klass in base_classes:
            return False
        names = vars(klass)
        if "__getitem__" in names or "adj" in names or "succ" in names:
            return True
    return False


def _preserve_pred_order(nx):
    """Whether conversions record the order of predecessors (see `from_networkx()`)."""
    if os.environ.get("EDGECASE_NX_ROUNDTRIP", "").lower() in {"1", "true", "yes"}:
        return True
    config = getattr(nx.config.backends, "edgecase", None)
    return bool(getattr(config, "preserve_pred_order", True))


def from_networkx(
    G,
    edge_attrs=None,
    node_attrs=None,
    preserve_edge_attrs=False,
    preserve_node_attrs=False,
    preserve_graph_attrs=False,
    *,
    name=None,
    graph_name=None,
):
    """Convert a NetworkX graph to an `EdgecaseGraph` (the backend's `convert_from_nx`).

    Parameters follow NetworkX's `convert_from_nx` protocol: `edge_attrs` and
    `node_attrs` map attribute names to defaults (a `None` default means "leave it out
    where absent"), and `preserve_*_attrs=True` keeps every attribute. The input graph is
    only read; attribute dicts are copied.
    """
    import networkx as nx

    if not isinstance(G, nx.Graph):
        if isinstance(G, nx.classes.reportviews.NodeView):
            H = nx.Graph()
            H.add_nodes_from(G.items())
            G = H
        else:
            raise TypeError(f"Bad type for graph argument {graph_name!r} in {name!r}: {type(G)}")

    directed = G.is_directed()
    multigraph = G.is_multigraph()
    digraph_storage = isinstance(G, nx.DiGraph)
    nx_class = type(G)
    if _overrides_adjacency(G, nx):
        # Read the graph the class presents, not the one it stores
        adj = G.succ if directed else G.adj
        pred = None
        nx_class = None
    else:
        adj = G._adj
        pred = G._pred if digraph_storage else None
    if nx_class in {nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph}:
        nx_class = None

    node_store = G._node
    nodes = list(node_store)
    key_to_id = dict(zip(nodes, range(1, len(nodes) + 1)))
    get_id = key_to_id.__getitem__

    if preserve_edge_attrs:
        edge_attrs = None
    getters = {}
    if edge_attrs:
        for attr, default in edge_attrs.items():
            getters[attr] = methodcaller("get", attr, MISSING if default is None else default)

    # One pass per array over the adjacency, all in C (`map`/`chain`) rather than a Python
    # loop per node: arcs are listed node by node, neighbor by neighbor (and key by key)
    nbr_maps = list(adj.values())
    if type(adj) is not dict:
        # Graph views (and the like): copy each node's neighbors, since `len()` of a filtered
        # view can disagree with what it yields (NetworkX's FilterMultiInner counts
        # neighbors without checking their edges)
        nbr_maps = list(map(dict, nbr_maps))
    src_per_pair = chain.from_iterable(map(repeat, map(get_id, adj), map(len, nbr_maps)))
    if not multigraph:
        src = list(src_per_pair)
        dst = list(map(get_id, chain.from_iterable(nbr_maps)))
        keys = None
        data_maps = nbr_maps
    else:
        keydicts = list(chain.from_iterable(map(_values, nbr_maps)))
        if type(adj) is not dict:
            keydicts = list(map(dict, keydicts))
        counts = list(map(len, keydicts))
        src = list(chain.from_iterable(map(repeat, src_per_pair, counts)))
        dst_per_pair = map(get_id, chain.from_iterable(nbr_maps))
        dst = list(chain.from_iterable(map(repeat, dst_per_pair, counts)))
        keys = list(chain.from_iterable(keydicts))
        data_maps = keydicts

    # Attribute dicts per arc: streamed for a single use, else listed once
    listed = None
    if len(getters) + bool(preserve_edge_attrs) > 1:
        listed = list(chain.from_iterable(map(_values, data_maps)))

    def arc_data():
        return listed if listed is not None else chain.from_iterable(map(_values, data_maps))

    edge_values = {attr: list(map(getter, arc_data())) for attr, getter in getters.items()}
    edge_dicts = list(map(dict, arc_data())) if preserve_edge_attrs else None

    # The order of each node's predecessors, which NetworkX keeps separately from the order
    # of the arcs into each node (one more pass over the edges, 10-15% of the time)
    pred_order = None
    if pred is not None and _preserve_pred_order(nx):
        pred_maps = list(pred.values())
        pred_order = (
            list(pred),
            list(map(len, pred_maps)),
            list(chain.from_iterable(pred_maps)),
        )

    node_values = {}
    node_dicts = None
    if preserve_node_attrs:
        node_dicts = list(map(dict, node_store.values()))
    elif node_attrs:
        datadicts = node_store.values()
        for attr, default in node_attrs.items():
            getter = methodcaller("get", attr, MISSING if default is None else default)
            node_values[attr] = list(map(getter, datadicts))

    return EdgecaseGraph(
        nodes=nodes,
        key_to_id=key_to_id,
        src=src,
        dst=dst,
        directed=directed,
        multigraph=multigraph,
        keys=keys,
        edge_values=edge_values,
        edge_defaults=dict(edge_attrs) if edge_attrs else None,
        edge_dicts=edge_dicts,
        node_values=node_values,
        node_defaults=dict(node_attrs) if node_values else None,
        node_dicts=node_dicts,
        graph=dict(G.graph) if preserve_graph_attrs else None,
        nx_class=nx_class,
        digraph_storage=digraph_storage,
        pred_order=pred_order,
    )


def _new_nx_graph(G, nx):
    """An empty NetworkX graph of the class `G` was converted from (or the plain class)."""
    if G.nx_class is not None:
        try:
            return G.nx_class()
        except Exception:
            pass
    if G.multigraph:
        return nx.MultiDiGraph() if G.digraph_storage else nx.MultiGraph()
    return nx.DiGraph() if G.digraph_storage else nx.Graph()


def to_networkx(G):
    """Convert an `EdgecaseGraph` back to a NetworkX graph (in the original order)."""
    import networkx as nx

    H = _new_nx_graph(G, nx)
    H.graph.update(G.graph)
    nodes = G.nodes

    # Node attribute dicts
    if G.node_dicts is not None:
        node_data = map(dict, G.node_dicts)
    elif G.node_values:
        columns = list(G.node_values.items())
        node_data = (
            {attr: values[i] for attr, values in columns if values[i] is not MISSING}
            for i in range(len(nodes))
        )
    else:
        node_data = (H.node_attr_dict_factory() for _ in nodes)
    H_node = H._node
    H_adj = H._adj
    inner = H.adjlist_inner_dict_factory
    for node, data in zip(nodes, node_data):
        H_node[node] = data
        H_adj[node] = inner()
    if G.digraph_storage:
        H_pred = H._pred
        for node in nodes:
            H_pred[node] = inner()

    # Edge attribute dicts, per arc
    num_arcs = len(G.src)
    if G.edge_dicts is not None:
        edge_data = map(dict, G.edge_dicts)
    elif G.edge_values:
        columns = list(G.edge_values.items())
        edge_data = (
            {attr: values[i] for attr, values in columns if values[i] is not MISSING}
            for i in range(num_arcs)
        )
    else:
        edge_data = (H.edge_attr_dict_factory() for _ in range(num_arcs))

    # Predecessors are filled in afterwards when their order is known
    fill_pred = G.digraph_storage and G.pred_order is None
    if G.multigraph:
        keys = G.keys
        for u_id, v_id, key, data in zip(G.src, G.dst, keys, edge_data):
            u = nodes[u_id - 1]
            v = nodes[v_id - 1]
            u_nbrs = H_adj[u]
            keydict = u_nbrs.get(v)
            if keydict is None:
                if G.digraph_storage:
                    keydict = u_nbrs[v] = H.edge_key_dict_factory()
                    if fill_pred:
                        H_pred[v][u] = keydict
                elif u in H_adj[v]:
                    # The other end of an undirected edge was seen first: share its keydict
                    keydict = u_nbrs[v] = H_adj[v][u]
                else:
                    keydict = u_nbrs[v] = H.edge_key_dict_factory()
            if key not in keydict:
                keydict[key] = data
    elif fill_pred:
        for u_id, v_id, data in zip(G.src, G.dst, edge_data):
            u = nodes[u_id - 1]
            v = nodes[v_id - 1]
            H_adj[u][v] = data
            H_pred[v][u] = data
    elif G.digraph_storage:
        for u_id, v_id, data in zip(G.src, G.dst, edge_data):
            H_adj[nodes[u_id - 1]][nodes[v_id - 1]] = data
    else:
        for u_id, v_id, data in zip(G.src, G.dst, edge_data):
            u = nodes[u_id - 1]
            v = nodes[v_id - 1]
            v_nbrs = H_adj[v]
            # The other end of an undirected edge shares one attribute dict
            H_adj[u][v] = v_nbrs[u] if u in v_nbrs else data

    if G.digraph_storage and not fill_pred:
        pred_nodes, pred_counts, pred_keys = G.pred_order
        preds = iter(pred_keys)
        for v, count in zip(pred_nodes, pred_counts):
            if count:
                v_pred = H_pred[v]
                for u in islice(preds, count):
                    v_pred[u] = H_adj[u][v]
    return H
