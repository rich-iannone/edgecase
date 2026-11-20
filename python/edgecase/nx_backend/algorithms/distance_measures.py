"""Distance measures and distance-based centralities, from shortest-path searches.

NetworkX's `distance_measures` (eccentricity, diameter, radius, center, periphery, centroid
or barycenter, harmonic_diameter) and the closeness and harmonic centralities. Each runs one
search per node in Rust (`PathAdjacency`, in parallel on big graphs) and combines the
distances in NetworkX's order and arithmetic, so floating-point results are equal to the bit.

Breadth-first search counts edges; with a weight attribute the searches are Dijkstra's (as
in `shortest_paths`, a missing weight is 1). Precomputed eccentricities (`e=`) are used as
given; precomputed shortest paths (`sp=`), callable weights and centroid's `attr=` (which
writes node attributes) are left to NetworkX.
"""

from __future__ import annotations

import inspect
import math
from collections import Counter
from itertools import pairwise

import networkx as nx

from edgecase.nx_backend.algorithms.shortest_paths import (
    _adjacency,
    _bfs_lengths,
    _contradiction,
    _dijkstra_lengths,
    _position,
)
from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "barycenter",
    "center",
    "centroid",
    "closeness_centrality",
    "diameter",
    "eccentricity",
    "harmonic_centrality",
    "harmonic_diameter",
    "periphery",
    "radius",
]


def _register(weight_name, *refused):
    """Register a function, leaving calls to NetworkX that give a callable weight (the
    argument `weight_name`) or any of the arguments `refused` (not None).
    """

    def decorate(func):
        signature = inspect.signature(func)

        def can_run(*args, **kwargs):
            try:
                arguments = signature.bind(*args, **kwargs).arguments
            except TypeError:
                return True  # (the call raises the TypeError itself)
            if callable(arguments.get(weight_name)):
                return "a callable weight is not supported; give an edge attribute name"
            for name in refused:
                if arguments.get(name) is not None:
                    return f"{name}= is not supported by edgecase"
            return True

        return networkx_algorithm(func, can_run=can_run)

    return decorate


def _nbunch_iter(G, nbunch):
    """NetworkX's `G.nbunch_iter(nbunch)` (same order and errors)."""
    if nbunch is None:
        return iter(G.nodes)
    adj = G.key_to_id
    try:
        single = nbunch in adj
    except TypeError:
        single = False
    if single:
        return iter([nbunch])

    def bunch_iter(nlist):
        try:
            for n in nlist:
                if n in adj:
                    yield n
        except TypeError as err:
            exc, message = err, err.args[0]
            if "iter" in message:
                exc = nx.NetworkXError("nbunch is not a node or a sequence of nodes.")
            if "object is not iterable" in message:
                exc = nx.NetworkXError(f"Node {nbunch} is not in the graph.")
            if "hashable" in message:
                exc = nx.NetworkXError(f"Node {n} in sequence nbunch is not a valid node.")
            raise exc

    return bunch_iter(nbunch)


def _number(value, integral):
    """A distance from Rust as NetworkX would have it: an int for int weights (or hops),
    else a float, except that 0 (a source's own distance) is the int 0.
    """
    if integral or value == 0:
        return int(value)
    return value


def _stats(G, sources, weight, reverse=False):
    """`source_stats()` from `sources` (positions); `integral` says if distances are ints."""
    adj = _adjacency(G, weight)
    reached, sums, maxima, failed = adj.source_stats(sources, weight is not None, reverse)
    return reached, sums, maxima, failed, weight is None or adj.integral


# Distance measures


@_register("weight", "sp")
def eccentricity(G, v=None, sp=None, weight=None):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Cannot compute eccentricity of a null graph.")
    order = len(G)
    nodes = list(_nbunch_iter(G, v))
    sources = [G.key_to_id[n] - 1 for n in nodes]
    reached, _, maxima, failed, integral = _stats(G, sources, weight)
    e = {}
    for i, n in enumerate(nodes):
        if i == failed:
            raise _contradiction()
        if reached[i] != order:
            if G.is_directed():
                msg = "Found infinite path length because the digraph is not strongly connected"
            else:
                msg = "Found infinite path length because the graph is not connected"
            raise nx.NetworkXError(msg)
        e[n] = _number(maxima[i], integral)
    if v in G:
        return e[v]
    return e


def _degrees(G):
    """`dict(G.degree())` for an undirected graph (a self-loop counts twice)."""
    nodes = G.nodes
    counts = Counter(G.src)
    counts.update(s for s, d in zip(G.src, G.dst) if s == d)
    return {node: counts[i] for i, node in enumerate(nodes, 1)}


def _lengths(G, node, weight):
    """`nx.shortest_path_length(G, source=node, weight=weight)`."""
    position = G.key_to_id[node] - 1
    if weight is None:
        return _bfs_lengths(G, position, None)
    return _dijkstra_lengths(G, [position], None, weight)


def _extrema_bounding(G, compute="diameter", weight=None):
    """NetworkX's `_extrema_bounding` (usebounds=True), with edgecase's searches."""
    degrees = _degrees(G)
    minlowernode = max(degrees, key=degrees.get)
    maxuppernode = None  # (set in the first round, used from the second)
    N = len(degrees)
    high = False
    ecc_lower = dict.fromkeys(G, 0)
    ecc_upper = dict.fromkeys(G, math.inf)
    candidates = set(G)
    minlower = math.inf
    maxlower = 0
    minupper = math.inf
    maxupper = 0
    searched = set()
    repeats = 0
    while candidates:
        if high:
            current = maxuppernode
        else:
            current = minlowernode
        high = not high
        # Searching a node again changes no bound, so the next round picks the same pair of
        # nodes; when that happens twice in a row NetworkX loops forever (float rounding can
        # leave a searched node's lower bound above its upper bound, so it is never ruled
        # out). Finish with exact eccentricities instead.
        if current in searched:
            repeats += 1
            if repeats == 2:
                return _exact_extrema(G, compute, weight)
        else:
            repeats = 0
            searched.add(current)
        dist = _lengths(G, current, weight)
        if len(dist) != N:
            msg = "Cannot compute metric because graph is not connected."
            raise nx.NetworkXError(msg)
        current_ecc = max(dist.values())
        maxuppernode = None
        minlowernode = None
        for i in candidates:
            d = dist[i]
            ecc_lower[i] = low = max(ecc_lower[i], max(d, current_ecc - d))
            ecc_upper[i] = upp = min(ecc_upper[i], current_ecc + d)
            minlower = min(low, minlower)
            maxlower = max(low, maxlower)
            minupper = min(upp, minupper)
            maxupper = max(upp, maxupper)
        if compute == "diameter":
            ruled_out = {
                i for i in candidates if ecc_upper[i] <= maxlower and 2 * ecc_lower[i] >= maxupper
            }
        elif compute == "radius":
            ruled_out = {
                i
                for i in candidates
                if ecc_lower[i] >= minupper and ecc_upper[i] + 1 <= 2 * minlower
            }
        elif compute == "periphery":
            ruled_out = {
                i
                for i in candidates
                if ecc_upper[i] < maxlower and (maxlower == maxupper or ecc_lower[i] > maxupper)
            }
        elif compute == "center":
            ruled_out = {
                i
                for i in candidates
                if ecc_lower[i] > minupper
                and (minlower == minupper or ecc_upper[i] + 1 < 2 * minlower)
            }
        elif compute == "eccentricities":
            ruled_out = set()
        else:
            msg = "compute must be one of 'diameter', 'radius', 'periphery', 'center', 'eccentricities'"
            raise ValueError(msg)
        ruled_out.update(i for i in candidates if ecc_lower[i] == ecc_upper[i])
        candidates -= ruled_out
        for i in candidates:
            if (
                minlowernode is None
                or (ecc_lower[i] == ecc_lower[minlowernode] and degrees[i] > degrees[minlowernode])
                or ecc_lower[i] < ecc_lower[minlowernode]
            ):
                minlowernode = i
            if (
                maxuppernode is None
                or (ecc_upper[i] == ecc_upper[maxuppernode] and degrees[i] > degrees[maxuppernode])
                or ecc_upper[i] > ecc_upper[maxuppernode]
            ):
                maxuppernode = i
    if compute == "diameter":
        return maxlower
    if compute == "radius":
        return minupper
    if compute == "periphery":
        return [v for v in G if ecc_lower[v] == maxlower]
    if compute == "center":
        return [v for v in G if ecc_upper[v] == minupper]
    if compute == "eccentricities":
        return ecc_lower
    return None


def _exact_extrema(G, compute, weight):
    """`_extrema_bounding()`'s result from all eccentricities (for a connected graph)."""
    e = eccentricity(G, weight=weight)
    if compute == "diameter":
        return max(e.values())
    if compute == "radius":
        return min(e.values())
    if compute == "periphery":
        diameter = max(e.values())
        return [v for v in G if e[v] == diameter]
    if compute == "center":
        radius = min(e.values())
        return [v for v in G if e[v] == radius]
    return e


@_register("weight")
def diameter(G, e=None, usebounds=False, weight=None):
    if usebounds is True and e is None and not G.is_directed():
        return _extrema_bounding(G, compute="diameter", weight=weight)
    if e is None:
        e = eccentricity(G, weight=weight)
    return max(e.values())


@_register("weight")
def periphery(G, e=None, usebounds=False, weight=None):
    if usebounds is True and e is None and not G.is_directed():
        return _extrema_bounding(G, compute="periphery", weight=weight)
    if e is None:
        e = eccentricity(G, weight=weight)
    diameter = max(e.values())
    return [v for v in e if e[v] == diameter]


@_register("weight")
def radius(G, e=None, usebounds=False, weight=None):
    if usebounds is True and e is None and not G.is_directed():
        return _extrema_bounding(G, compute="radius", weight=weight)
    if e is None:
        e = eccentricity(G, weight=weight)
    return min(e.values())


@_register("weight")
def center(G, e=None, usebounds=False, weight=None):
    if usebounds is True and e is None and not G.is_directed():
        return _extrema_bounding(G, compute="center", weight=weight)
    if e is None and weight is None and not G.is_directed() and len(G) == 0:
        # (NetworkX first asks `is_tree`, which raises this; on a tree its answer equals
        # the general one below)
        raise nx.NetworkXPointlessConcept("G has no nodes.")
    if e is None:
        e = eccentricity(G, weight=weight)
    radius = min(e.values())
    return [v for v in e if e[v] == radius]


@_register("weight", "sp")
def harmonic_diameter(G, sp=None, *, weight=None):
    order = len(G)
    adj = _adjacency(G, weight)
    _, _, sum_invd, failed = adj.inverse_distance_sums(list(range(order)), weight is not None)
    if failed is not None:
        raise _contradiction()
    if sum_invd != 0:
        return order * (order - 1) / sum_invd
    if order > 1:
        return math.inf
    return math.nan


def _nx_str(G):
    """`str(G)` of the NetworkX graph `G` was converted from (without its name)."""
    if G.nx_class is not None:
        kind = G.nx_class.__name__
    else:
        kind = ("Multi" if G.multigraph else "") + ("DiGraph" if G.digraph_storage else "Graph")
    name = G.graph.get("name", "")
    named = f" named {name!r}" if name else ""
    return f"{kind}{named} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"


def _is_tree(G):
    """`nx.is_tree(G)` for an undirected graph with nodes."""
    if G.number_of_edges() != len(G) - 1:
        return False
    reached, *_ = _adjacency(G).source_stats([0])
    return reached[0] == len(G)


def _dfs_edges(nbrs, source):
    """NetworkX's `dfs_edges(G, source)` over adjacency lists of positions."""
    visited = {source}
    stack = [(source, iter(nbrs[source]))]
    depth_now = 1
    depth_limit = len(nbrs)
    while stack:
        parent, children = stack[-1]
        for child in children:
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now < depth_limit:
                    stack.append((child, iter(nbrs[child])))
                    depth_now += 1
                    break
        else:
            stack.pop()
            depth_now -= 1


def _tree_centroid(G):
    """NetworkX's `nx.tree.centroid(G)` (whose walk decides the order of two centroids)."""
    n = len(G)
    nbrs = [[] for _ in range(n)]
    for s, d in zip(G.src, G.dst):
        nbrs[s - 1].append(d - 1)
    root = 0  # (`arbitrary_element(G)`: the first node)
    sizes = {root: 1}
    stack = [root]
    for parent, child in _dfs_edges(nbrs, root):
        while stack[-1] != parent:
            descendant = stack.pop()
            sizes[stack[-1]] += sizes[descendant]
        stack.append(child)
        sizes[child] = 1
    for child, parent in pairwise(reversed(stack)):
        sizes[parent] += sizes[child]

    def heaviest_child(prev, root):
        return max((x for x in nbrs[root] if x != prev), key=sizes.get, default=None)

    prev = None
    hc = heaviest_child(prev, root)
    while max(n - sizes[root], sizes.get(hc, 0)) > n / 2:
        prev, root = root, hc
        hc = heaviest_child(prev, root)
    nodes = G.nodes
    return [nodes[root]] + [nodes[x] for x in nbrs[root] if x != prev and sizes[x] == n / 2]


@_register("weight", "attr", "sp")
def centroid(G, weight=None, attr=None, sp=None):
    if weight is None and not G.is_directed():
        if len(G) == 0:
            raise nx.NetworkXPointlessConcept("G has no nodes.")
        if _is_tree(G):
            return _tree_centroid(G)
    n = len(G)
    reached, sums, _, failed, integral = _stats(G, list(range(n)), weight)
    smallest, centroid_vertices = float("inf"), []
    for i, v in enumerate(G.nodes):
        if i == failed:
            raise _contradiction()
        if reached[i] < n:
            raise nx.NetworkXNoPath(
                f"Input graph {_nx_str(G)} is disconnected, so every induced subgraph has "
                "infinite barycentricity."
            )
        barycentricity = int(sums[i]) if integral else sums[i]
        if barycentricity < smallest:
            smallest = barycentricity
            centroid_vertices = [v]
        elif barycentricity == smallest:
            centroid_vertices.append(v)
    return centroid_vertices


barycenter = centroid


# Centrality


@_register("distance", "sp")
def closeness_centrality(G, u=None, distance=None, wf_improved=True, *, sp=None):
    directed = G.is_directed()
    if u is None:
        nodes = G.nodes
        sources = list(range(len(nodes)))
    else:
        position = _position(G, u)
        if position < 0:
            if distance is None:
                raise nx.NodeNotFound(f"Source {u} is not in G")
            raise nx.NodeNotFound(f"Node {u} not found in graph")
        nodes = [u]
        sources = [position]
    # A directed graph's closeness counts distances *to* each node
    reached, sums, _, failed, integral = _stats(G, sources, distance, reverse=directed)
    if failed is not None:
        raise _contradiction()
    len_G = len(G)
    closeness_dict = {}
    for n, count, totsp in zip(nodes, reached, sums):
        if integral:
            totsp = int(totsp)
        _closeness_centrality = 0.0
        if totsp > 0.0 and len_G > 1:
            _closeness_centrality = (count - 1.0) / totsp
            if wf_improved:
                s = (count - 1.0) / (len_G - 1)
                _closeness_centrality *= s
        closeness_dict[n] = _closeness_centrality
    if u is not None:
        return closeness_dict[u]
    return closeness_dict


@_register("distance")
def harmonic_centrality(G, nbunch=None, distance=None, sources=None):
    nbunch = set(_nbunch_iter(G, nbunch) if nbunch is not None else G.nodes)
    sources = set(_nbunch_iter(G, sources) if sources is not None else G.nodes)
    centrality = {u: 0 for u in nbunch}
    transposed = False
    if len(nbunch) < len(sources):
        transposed = True
        nbunch, sources = sources, nbunch
    key_to_id = G.key_to_id
    members = [False] * len(G)
    for u in nbunch:
        members[key_to_id[u] - 1] = True
    adj = _adjacency(G, distance)
    per_node, touched, _, failed = adj.inverse_distance_sums(
        [key_to_id[v] - 1 for v in sources],
        distance is not None,
        transposed and G.is_directed(),
        members,
        transposed,
    )
    if failed is not None:
        raise _contradiction()
    nodes = G.nodes
    for i, hit in enumerate(touched):
        if hit:
            centrality[nodes[i]] = per_node[i]
    return centrality
