"""Degree correlations, assortativity, reciprocity, efficiency, Wiener index and rich clubs.

The assortativity family follows NetworkX's computations step by step on the converted
graph's arrays: degrees summed as NetworkX sums them, node pairs visited in the same order
(including the order of the Python sets NetworkX iterates), and the same NumPy/SciPy final
steps, so the results are NetworkX's. Efficiencies sum `1/d` in Rust in NetworkX's
breadth-first order.
"""

from __future__ import annotations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "attribute_assortativity_coefficient",
    "attribute_mixing_dict",
    "attribute_mixing_matrix",
    "average_degree_connectivity",
    "average_neighbor_degree",
    "degree_assortativity_coefficient",
    "degree_mixing_dict",
    "degree_mixing_matrix",
    "degree_pearson_correlation_coefficient",
    "efficiency",
    "global_efficiency",
    "local_efficiency",
    "node_attribute_xy",
    "node_degree_xy",
    "numeric_assortativity_coefficient",
    "overall_reciprocity",
    "reciprocity",
    "rich_club_coefficient",
    "wiener_index",
]


# Graph views on the converted arrays


class _View:
    """Per-node neighbor structure of an `EdgecaseGraph`, as NetworkX iterates it."""

    def __init__(self, G):
        self.G = G
        n = len(G.nodes)
        starts = [0] * (n + 2)
        for s in G.src:
            starts[s + 1] += 1
        for i in range(1, n + 2):
            starts[i] += starts[i - 1]
        self.starts = starts

    def arcs(self, u):
        """Arc indices of node ID `u` (each key of a multigraph's parallel edges)."""
        return range(self.starts[u], self.starts[u + 1])

    def succ(self, u):
        """Distinct successors (neighbors) of `u` with the first arc to each."""
        dst = self.G.dst
        out = []
        previous = None
        for k in self.arcs(u):
            if dst[k] != previous:
                out.append((dst[k], k))
                previous = dst[k]
        return out

    def pred(self, u):
        """Distinct predecessors of `u` (directed), in NetworkX's predecessor order."""
        preds = self.__dict__.get("_preds")
        if preds is None:
            preds = self._preds = self._build_preds()
        return preds[u]

    def _build_preds(self):
        G = self.G
        n = len(G.nodes)
        preds = [[] for _ in range(n + 1)]
        first_arc = {}
        for k, (s, d) in enumerate(zip(G.src, G.dst)):
            first_arc.setdefault((s, d), k)
        if G.pred_order is not None:
            key_to_id = G.key_to_id
            pred_nodes, counts, keys = G.pred_order
            k = 0
            for v, count in zip(pred_nodes, counts):
                vid = key_to_id[v]
                for u in keys[k : k + count]:
                    uid = key_to_id[u]
                    preds[vid].append((uid, first_arc[(uid, vid)]))
                k += count
        else:
            for (s, d), k in first_arc.items():
                preds[d].append((s, k))
        return preds


def _degrees(G, weight, kind):
    """Degrees per node ID (list indexed by ID - 1) as NetworkX's degree views give them.

    `kind`: "all" (`G.degree`), "out" or "in".
    """
    from edgecase.nx_backend.algorithms.community import _degrees as degree_lists

    weights = None if weight is None else G.edge_values_of(weight, 1)
    first, second = degree_lists(G, weights)
    if second is None:  # undirected
        return first
    if kind == "out":
        return first
    if kind == "in":
        return second
    return [o + i for o, i in zip(first, second)]


def _items(G, degrees, nbunch):
    """`(node, degree)` for `degree_view(nbunch)` (NetworkX's nbunch semantics)."""
    from edgecase.nx_backend.algorithms.cluster import nbunch_iter

    key_to_id = G.key_to_id
    return ((n, degrees[key_to_id[n] - 1]) for n in nbunch_iter(G, nbunch))


def _multigraph_weight(G, weight):
    """`G[u][v].get(weight, 1)` on a multigraph reads the keydict: 1 unless a key is `weight`."""
    if G.keys is not None and weight in set(G.keys):
        raise NotImplementedError("an edge key equals the weight attribute name")
    return 1


# Neighbor degrees


@networkx_algorithm(linear_time=True)
def average_neighbor_degree(G, source="out", target="out", nodes=None, weight=None):
    import networkx as nx

    directed = G.is_directed()
    kinds = {"in": "in", "out": "out", "in+out": "all"}
    if directed:
        if source not in kinds:
            raise nx.NetworkXError(f"source argument {source} must be 'in', 'out' or 'in+out'")
        if target not in kinds:
            raise nx.NetworkXError(f"target argument {target} must be 'in', 'out' or 'in+out'")
    elif source != "out" or target != "out":
        raise nx.NetworkXError("source and target arguments are only supported for directed graphs")
    source_kind = kinds[source] if directed else "all"
    target_kind = kinds[target] if directed else "all"
    t_deg = _degrees(G, None, target_kind)
    s_deg = _degrees(G, weight, source_kind)
    view = _View(G)
    use_succ = not directed or "out" in source
    use_pred = directed and "in" in source
    if weight is not None:
        if G.is_multigraph():
            factor = _multigraph_weight(G, weight)
            values = None
        else:
            values = G.edge_values_of(weight, 1)
    key_to_id = G.key_to_id
    avg = {}
    for n, deg in _items(G, s_deg, nodes):
        if deg == 0:
            avg[n] = 0.0
            continue
        nid = key_to_id[n]
        succ = view.succ(nid) if use_succ else ()
        pred = view.pred(nid) if use_pred else ()
        if weight is None:
            avg[n] = (sum(t_deg[v - 1] for v, _ in succ) + sum(t_deg[v - 1] for v, _ in pred)) / deg
        elif values is None:
            avg[n] = (
                sum(factor * t_deg[v - 1] for v, _ in succ)
                + sum(factor * t_deg[v - 1] for v, _ in pred)
            ) / deg
        else:
            avg[n] = (
                sum(values[k] * t_deg[v - 1] for v, k in succ)
                + sum(values[k] * t_deg[v - 1] for v, k in pred)
            ) / deg
    return avg


@networkx_algorithm(linear_time=True)
def average_degree_connectivity(G, source="in+out", target="in+out", nodes=None, weight=None):
    import networkx as nx

    directed = G.is_directed()
    kinds = {"out": "out", "in": "in", "in+out": "all"}
    if directed:
        if source not in kinds:
            raise nx.NetworkXError('source must be one of "in", "out", or "in+out"')
        if target not in kinds:
            raise nx.NetworkXError('target must be one of "in", "out", or "in+out"')
        reverse = source == "in"
    else:
        if source != "in+out" or target != "in+out":
            raise nx.NetworkXError(
                "source and target arguments are only supported for directed graphs"
            )
        reverse = False
    source_kind = kinds[source] if directed else "all"
    target_kind = kinds[target] if directed else "all"
    s_deg = _degrees(G, None, source_kind)
    s_deg_w = s_deg if weight is None else _degrees(G, weight, source_kind)
    t_deg = _degrees(G, None, target_kind)
    view = _View(G)
    key_to_id = G.key_to_id
    if weight is not None:
        if G.is_multigraph():
            factor = _multigraph_weight(G, weight)
            values = None
        else:
            values = G.edge_values_of(weight, 1)

    if nodes in G:
        source_nodes = [(nodes, s_deg[key_to_id[nodes] - 1])]
    else:
        source_nodes = _items(G, s_deg, nodes)
    dsum = {}
    dnorm = {}
    for n, k in source_nodes:
        nid = key_to_id[n]
        # G.neighbors (successors for a directed graph) or G.predecessors
        nbrs = view.pred(nid) if reverse else view.succ(nid)
        if weight is None:
            s = sum(t_deg[v - 1] for v, _ in nbrs)
        elif values is None:
            s = sum(factor * t_deg[v - 1] for v, _ in nbrs)
        else:
            s = sum(values[a] * t_deg[v - 1] for v, a in nbrs)
        dnorm[k] = dnorm.get(k, 0) + s_deg_w[nid - 1]
        dsum[k] = dsum.get(k, 0) + s
    return {k: avg if dnorm[k] == 0 else avg / dnorm[k] for k, avg in dsum.items()}


# Mixing and assortativity


def _node_degree_xy(G, x="out", y="in", weight=None, nodes=None):
    nodes = set(G.nodes) if nodes is None else set(nodes)
    directed = G.is_directed()
    xdeg = _degrees(G, weight, x if directed else "all")
    ydeg = _degrees(G, weight, y if directed else "all")
    view = _View(G)
    key_to_id, keys, dst = G.key_to_id, G.nodes, G.dst
    for u, degu in _items(G, xdeg, nodes):
        for k in view.arcs(key_to_id[u]):
            v = dst[k]
            if keys[v - 1] in nodes:
                yield degu, ydeg[v - 1]


@networkx_algorithm(linear_time=True)
def node_degree_xy(G, x="out", y="in", weight=None, nodes=None):
    return _node_degree_xy(G, x, y, weight, nodes)


def _mixing_dict(xy, normalized=False):
    d = {}
    psum = 0.0
    for x, y in xy:
        if x not in d:
            d[x] = {}
        if y not in d:
            d[y] = {}
        v = d[x].get(y, 0)
        d[x][y] = v + 1
        psum += 1
    if normalized:
        for _, jdict in d.items():
            for j in jdict:
                jdict[j] /= psum
    return d


@networkx_algorithm(linear_time=True)
def degree_mixing_dict(G, x="out", y="in", weight=None, nodes=None, normalized=False):
    return _mixing_dict(_node_degree_xy(G, x, y, weight, nodes), normalized=normalized)


def _degree_mixing_matrix(
    G, x="out", y="in", weight=None, nodes=None, normalized=True, mapping=None
):
    from networkx.utils import dict_to_numpy_array

    d = _mixing_dict(_node_degree_xy(G, x, y, weight, nodes))
    a = dict_to_numpy_array(d, mapping=mapping)
    if normalized:
        a = a / a.sum()
    return a


@networkx_algorithm(linear_time=True)
def degree_mixing_matrix(
    G, x="out", y="in", weight=None, nodes=None, normalized=True, mapping=None
):
    return _degree_mixing_matrix(G, x, y, weight, nodes, normalized, mapping)


def _numeric_ac(M, mapping):
    import numpy as np

    if M.sum() != 1.0:
        M = M / M.sum()
    x = np.array(list(mapping.keys()))
    y = x  # x and y have the same support
    idx = list(mapping.values())
    a = M.sum(axis=0)
    b = M.sum(axis=1)
    vara = (a[idx] * x**2).sum() - ((a[idx] * x).sum()) ** 2
    varb = (b[idx] * y**2).sum() - ((b[idx] * y).sum()) ** 2
    xy = np.outer(x, y)
    ab = np.outer(a[idx], b[idx])
    return float((xy * (M - ab)).sum() / np.sqrt(vara * varb))


def _attribute_ac(M):
    if M.sum() != 1.0:
        M = M / M.sum()
    s = (M @ M).sum()
    t = M.trace()
    r = (t - s) / (1 - s)
    return float(r)


@networkx_algorithm(linear_time=True)
def degree_assortativity_coefficient(G, x="out", y="in", weight=None, nodes=None):
    if nodes is None:
        nodes = G.nodes
    if G.is_directed():
        indeg = (
            {d for _, d in _items(G, _degrees(G, weight, "in"), nodes)} if "in" in (x, y) else set()
        )
        outdeg = (
            {d for _, d in _items(G, _degrees(G, weight, "out"), nodes)}
            if "out" in (x, y)
            else set()
        )
        degrees = set.union(indeg, outdeg)
    else:
        degrees = {d for _, d in _items(G, _degrees(G, weight, "all"), nodes)}
    mapping = {d: i for i, d in enumerate(degrees)}
    M = _degree_mixing_matrix(G, x=x, y=y, nodes=nodes, weight=weight, mapping=mapping)
    return _numeric_ac(M, mapping=mapping)


@networkx_algorithm(linear_time=True)
def degree_pearson_correlation_coefficient(G, x="out", y="in", weight=None, nodes=None):
    import scipy as sp

    xy = list(_node_degree_xy(G, x=x, y=y, nodes=nodes, weight=weight))
    if len(xy) < 2:
        return float("nan")
    x, y = zip(*xy)
    return float(sp.stats.pearsonr(x, y)[0])


def _node_attribute_xy(G, attribute, nodes=None):
    nodes = set(G.nodes) if nodes is None else set(nodes)
    values = G.node_values_of(attribute, None)
    keys, dst = G.nodes, G.dst
    view = _View(G)
    for uid, u in enumerate(keys, 1):
        if u not in nodes:
            continue
        uattr = values[uid - 1]
        for k in view.arcs(uid):
            yield (uattr, values[dst[k] - 1])


@networkx_algorithm(linear_time=True)
def node_attribute_xy(G, attribute, nodes=None):
    return _node_attribute_xy(G, attribute, nodes)


@networkx_algorithm(linear_time=True)
def attribute_mixing_dict(G, attribute, nodes=None, normalized=False):
    return _mixing_dict(_node_attribute_xy(G, attribute, nodes), normalized=normalized)


def _attribute_mixing_matrix(G, attribute, nodes=None, mapping=None, normalized=True):
    from networkx.utils import dict_to_numpy_array

    d = _mixing_dict(_node_attribute_xy(G, attribute, nodes))
    a = dict_to_numpy_array(d, mapping=mapping)
    if normalized:
        a = a / a.sum()
    return a


@networkx_algorithm(linear_time=True)
def attribute_mixing_matrix(G, attribute, nodes=None, mapping=None, normalized=True):
    return _attribute_mixing_matrix(G, attribute, nodes, mapping, normalized)


@networkx_algorithm(linear_time=True)
def attribute_assortativity_coefficient(G, attribute, nodes=None):
    return _attribute_ac(_attribute_mixing_matrix(G, attribute, nodes))


@networkx_algorithm(linear_time=True)
def numeric_assortativity_coefficient(G, attribute, nodes=None):
    from edgecase.nx_backend.graph import MISSING

    if nodes is None:
        nodes = G.nodes
    values = G.node_values_of(attribute, MISSING)
    key_to_id = G.key_to_id
    vals = set()
    for n in nodes:
        value = values[key_to_id[n] - 1]
        if value is MISSING:
            raise KeyError(attribute)
        vals.add(value)
    mapping = {d: i for i, d in enumerate(vals)}
    M = _attribute_mixing_matrix(G, attribute, nodes, mapping)
    return _numeric_ac(M, mapping)


# Reciprocity


@networkx_algorithm(linear_time=True)
def overall_reciprocity(G):
    import networkx as nx

    n_all_edge = G.number_of_edges()
    pairs = {(s, d) if s <= d else (d, s) for s, d in zip(G.src, G.dst)}
    n_overlap_edge = (n_all_edge - len(pairs)) * 2
    if n_all_edge == 0:
        raise nx.NetworkXError("Not defined for empty graphs")
    return n_overlap_edge / n_all_edge


@networkx_algorithm(linear_time=True)
def reciprocity(G, nodes=None):
    import networkx as nx

    from edgecase.nx_backend.algorithms.cluster import nbunch_iter

    if nodes is None:
        return overall_reciprocity(G)
    view = _View(G)
    key_to_id = G.key_to_id

    def node_reciprocity(node):
        nid = key_to_id[node]
        pred = {v for v, _ in view.pred(nid)}
        succ = {v for v, _ in view.succ(nid)}
        n_total = len(pred) + len(succ)
        if n_total == 0:
            return None
        return 2 * len(pred & succ) / n_total

    if nodes in G:
        result = node_reciprocity(nodes)
        if result is None:
            raise nx.NetworkXError("Not defined for isolated nodes.")
        return result
    return {node: node_reciprocity(node) for node in nbunch_iter(G, nodes)}


# Efficiency and Wiener index


@networkx_algorithm(linear_time=True)
def efficiency(G, u, v):
    import networkx as nx

    from edgecase.nx_backend.algorithms.shortest_paths import shortest_path_length

    try:
        eff = 1 / shortest_path_length(G, u, v, None, "dijkstra")
    except nx.NetworkXNoPath:
        eff = 0
    return eff


@networkx_algorithm
def global_efficiency(G):
    from edgecase._core import nx_global_efficiency_sum

    n = len(G)
    denom = n * (n - 1)
    if denom == 0:
        return 0
    return nx_global_efficiency_sum(G.rust_graph()) / denom


@networkx_algorithm
def local_efficiency(G):
    from edgecase._core import nx_local_efficiency_sums

    nodes = G.nodes
    view = _View(G)
    total = len(nodes)
    # Each neighborhood's nodes in the order NetworkX's subgraph view iterates them: the
    # Python set it builds (when it is under half the graph) or the graph's node order
    offsets = [0]
    members = []
    sizes = []
    key_to_id = G.key_to_id
    for nid in range(1, total + 1):
        nbr_ids = [v for v, _ in view.succ(nid)]
        node_set = {nodes[v - 1] for v in nbr_ids}
        if 2 * len(node_set) < total:
            members.extend(key_to_id[k] for k in node_set)
        else:
            members.extend(sorted(nbr_ids))
        offsets.append(len(members))
        sizes.append(len(node_set))
    sums = nx_local_efficiency_sums(G.rust_graph(), offsets, members)
    efficiencies = (s / (k * (k - 1)) if k * (k - 1) != 0 else 0 for s, k in zip(sums, sizes))
    return sum(efficiencies) / len(nodes)


@networkx_algorithm
def wiener_index(G, weight=None):
    from itertools import chain

    from edgecase.nx_backend.algorithms.components import is_connected, is_strongly_connected
    from edgecase.nx_backend.algorithms.shortest_paths import shortest_path_length

    connected = is_strongly_connected(G) if G.is_directed() else is_connected(G)
    if not connected:
        return float("inf")
    spl = shortest_path_length(G, None, None, weight, "dijkstra")
    total = sum(chain.from_iterable(nbrs.values() for node, nbrs in spl))
    return total if G.is_directed() else total / 2


# Rich club


def _rich_club_can_run(G, normalized=True, *args, **kwargs):
    if kwargs.get("normalized", normalized):
        return "the normalized rich-club coefficient randomizes the graph; left to NetworkX"
    return True


@networkx_algorithm(can_run=_rich_club_can_run, linear_time=True)
def rich_club_coefficient(G, normalized=True, Q=100, seed=None, *, n_samples=1):
    from itertools import accumulate

    if any(s == d for s, d in zip(G.src, G.dst)):
        raise Exception("rich_club_coefficient is not implemented for graphs with self loops.")
    degrees = _degrees(G, None, "all")
    # nx.degree_histogram
    counts = {}
    for d in degrees:
        counts[d] = counts.get(d, 0) + 1
    deghist = [counts.get(d, 0) for d in range(max(counts) + 1)] if counts else []
    total = sum(deghist)
    nks = (total - cs for cs in accumulate(deghist) if total - cs > 1)
    src, dst = G.src, G.dst
    edge_degrees = sorted(
        (sorted((degrees[src[k] - 1], degrees[dst[k] - 1])) for k in G.edge_arcs), reverse=True
    )
    ek = G.number_of_edges()
    if ek == 0:
        return {}
    k1, k2 = edge_degrees.pop()
    rc = {}
    for d, nk in enumerate(nks):
        while k1 <= d:
            if len(edge_degrees) == 0:
                ek = 0
                break
            k1, k2 = edge_degrees.pop()
            ek -= 1
        rc[d] = 2 * ek / (nk * (nk - 1))
    return rc
