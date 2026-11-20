"""Community detection and partition quality.

The randomized algorithms (`louvain_*`, `asyn_lpa_communities`) draw their random numbers
from the `seed` generator exactly as NetworkX does: edgecase's kernels run CPython's
Mersenne Twister on the generator's state (`Random.getstate()`), and the state is written
back afterwards, so results, and the generator's later output, are NetworkX's. NumPy-based
`seed` generators are left to NetworkX.
"""

from __future__ import annotations

import inspect
import random
from itertools import combinations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = [
    "asyn_lpa_communities",
    "greedy_modularity_communities",
    "is_partition",
    "label_propagation_communities",
    "leiden_communities",
    "leiden_partitions",
    "louvain_communities",
    "louvain_partitions",
    "modularity",
    "partition_quality",
]


# Python's random generator state <-> edgecase


def _rng_state(seed):
    version, internal, gauss = seed.getstate()
    return list(internal[:624]), internal[624], (version, gauss)


def _set_rng_state(seed, words, index, extra):
    version, gauss = extra
    seed.setstate((version, (*words, index), gauss))


def _python_random(seed):
    if type(seed) is not random.Random:
        return f"seed is a {type(seed).__name__}, not a random.Random"
    return True


# Partitions


def _is_partition(G, communities):
    if not isinstance(communities, list):
        communities = list(communities)
    nodes = {n for c in communities for n in c if n in G}
    return len(G) == len(nodes) == sum(len(c) for c in communities)


@networkx_algorithm(linear_time=True)
def is_partition(G, communities):
    return _is_partition(G, communities)


@networkx_algorithm
def modularity(G, communities, weight="weight", resolution=1):
    from networkx.algorithms.community.quality import NotAPartition

    from edgecase._core import modularity as rust_modularity

    if not isinstance(communities, list):
        communities = list(communities)
    if not _is_partition(G, communities):
        raise NotAPartition(G, communities)
    key_to_id = G.key_to_id
    membership = {key_to_id[v]: i for i, c in enumerate(communities) for v in c}
    rg = G.rust_graph() if weight is None else G.rust_graph(weight, default=1.0)
    q = rust_modularity(rg, membership, resolution)
    return 0 if q == 0 else q


@networkx_algorithm(linear_time=True)
def partition_quality(G, partition):
    # (NetworkX checks that `partition` is a partition before dispatching)
    community = [0] * (len(G.nodes) + 1)
    key_to_id = G.key_to_id
    for i, nodes in enumerate(partition):
        for node in nodes:
            community[key_to_id[node]] = i
    if not G.is_multigraph():
        possible_inter_community_edges = sum(
            len(p1) * len(p2) for p1, p2 in combinations(partition, 2)
        )
        if G.is_directed():
            possible_inter_community_edges *= 2
    else:
        possible_inter_community_edges = 0
    n = len(G)
    total_pairs = n * (n - 1)
    if not G.is_directed():
        total_pairs //= 2
    src, dst = G.src, G.dst
    intra_community_edges = sum(community[src[i]] == community[dst[i]] for i in G.edge_arcs)
    num_edges = len(G.edge_arcs)
    inter_community_non_edges = possible_inter_community_edges - (num_edges - intra_community_edges)
    coverage = intra_community_edges / num_edges
    if G.is_multigraph():
        performance = -1.0
    else:
        performance = (intra_community_edges + inter_community_non_edges) / total_pairs
    return coverage, performance


# Louvain


def _degrees(G, weights):
    """Weighted degrees per node (in node order) summed as NetworkX sums them.

    Undirected: `G.degree(weight)` (a self-loop counts twice); directed:
    `(G.out_degree(weight), G.in_degree(weight))`, the in-degrees in NetworkX's
    predecessor order when the conversion recorded it. `weights` is per arc (None: 1).
    """
    src, dst = G.src, G.dst
    if weights is None:
        weights = [1] * len(src)
    n = len(G.nodes)
    starts = [0] * (n + 2)
    for s in src:
        starts[s + 1] += 1
    for i in range(1, n + 2):
        starts[i] += starts[i - 1]
    if G.is_directed():
        out_degrees = [sum(weights[starts[u] : starts[u + 1]]) for u in range(1, n + 1)]
        in_weights = {}
        if G.pred_order is not None:
            by_pair = {}
            for s, d, w in zip(src, dst, weights):
                by_pair.setdefault((s, d), []).append(w)
            key_to_id = G.key_to_id
            pred_nodes, counts, preds = G.pred_order
            k = 0
            for v, count in zip(pred_nodes, counts):
                vid = key_to_id[v]
                ws = in_weights[vid] = []
                for u in preds[k : k + count]:
                    ws.extend(by_pair[(key_to_id[u], vid)])
                k += count
        else:
            for d, w in zip(dst, weights):
                in_weights.setdefault(d, []).append(w)
        in_degrees = [sum(in_weights.get(u, ())) for u in range(1, n + 1)]
        return out_degrees, in_degrees
    multigraph = G.is_multigraph()
    degrees = []
    for u in range(1, n + 1):
        deg = sum(weights[starts[u] : starts[u + 1]])
        loops = [weights[i] for i in range(starts[u], starts[u + 1]) if dst[i] == u]
        if loops:
            deg = deg + (sum(loops) if multigraph else loops[0])
        degrees.append(deg)
    return degrees, None


def _size(G, weights):
    """`G.size(weight)` with NetworkX's summation order (weights per arc)."""
    if weights is None:
        return G.number_of_edges()
    degrees, in_degrees = _degrees(G, weights)
    if in_degrees is not None:
        degrees = [d + i for d, i in zip(degrees, in_degrees)]
    return sum(degrees) / 2


def _louvain(G, weight, resolution, threshold, seed):
    from edgecase._core import NxLouvain

    nodes = G.nodes
    if G.number_of_edges() == 0:
        yield [{u} for u in nodes]
        return
    weights = None if weight is None else G.edge_values_of(weight, 1)
    arcs = G.edge_arcs
    src, dst = G.src, G.dst
    try:
        state = NxLouvain(
            len(nodes),
            [src[i] for i in arcs],
            [dst[i] for i in arcs],
            [1.0] * len(arcs) if weights is None else [weights[i] for i in arcs],
            G.is_directed(),
            float(_size(G, weights)),
            resolution,
        )
    except TypeError as exc:
        raise NotImplementedError(f"edge weights edgecase cannot use: {exc}") from exc

    def one_level():
        words, index, extra = _rng_state(seed)
        improvement, words, index = state.one_level(words, index)
        _set_rng_state(seed, words, index, extra)
        return improvement

    mod = state.singleton_modularity()
    one_level()
    improvement = True
    while improvement:
        yield [{nodes[i - 1] for i in community} for community in state.partition()]
        new_mod = state.inner_modularity()
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        state.aggregate()
        improvement = one_level()


def _seed_can_run(func):
    """`can_run` that checks the `seed` argument of `func`'s call."""
    signature = inspect.signature(func)

    def can_run(*args, **kwargs):
        try:
            seed = signature.bind(*args, **kwargs).arguments.get("seed")
        except TypeError:
            return True  # (the call raises the TypeError itself)
        return _python_random(random._inst if seed is None else seed)

    return can_run


def _seeded(func):
    return networkx_algorithm(func, can_run=_seed_can_run(func))


@_seeded
def louvain_partitions(G, weight="weight", resolution=1, threshold=0.0000001, seed=None):
    yield from _louvain(G, weight, resolution, threshold, seed)


@_seeded
def louvain_communities(
    G, weight="weight", resolution=1, threshold=0.0000001, max_level=None, seed=None
):
    from collections import deque
    from itertools import islice

    partitions = _louvain(G, weight, resolution, threshold, seed)
    if max_level is not None:
        if max_level <= 0:
            raise ValueError("max_level argument must be a positive integer or None")
        partitions = islice(partitions, max_level)
    final_partition = deque(partitions, maxlen=1)
    return final_partition.pop()


# Label propagation


def _groups(G, labels):
    """Nodes grouped by label, in order of each label's first node."""
    nodes = G.nodes
    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, set()).add(nodes[i])
    return groups


@_seeded
def asyn_lpa_communities(G, weight=None, seed=None):
    from edgecase._core import nx_asyn_lpa

    weights = None
    if weight is not None:
        weights = G.edge_values_of(weight, 1)
    words, index, extra = _rng_state(seed)
    try:
        labels, words, index = nx_asyn_lpa(len(G.nodes), G.src, G.dst, weights, words, index)
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(f"edgecase cannot run this graph: {exc}") from exc
    _set_rng_state(seed, words, index, extra)
    yield from _groups(G, labels).values()


def _not_multigraph(G):
    if G.is_multigraph():
        return "NetworkX colors a multigraph by degrees that count parallel edges"
    return True


@networkx_algorithm(can_run=_not_multigraph)
def label_propagation_communities(G):
    from edgecase._core import semi_synchronous_label_propagation

    ids, labels = semi_synchronous_label_propagation(G.rust_graph())
    label_of = dict(zip(ids, labels))
    return _groups(G, [label_of[i] for i in range(1, len(G.nodes) + 1)]).values()


# Greedy modularity (Clauset-Newman-Moore)


@networkx_algorithm
def greedy_modularity_communities(G, weight=None, resolution=1, cutoff=1, best_n=None):
    from edgecase._core import nx_greedy_modularity

    nodes = G.nodes
    n = len(nodes)
    if not G.number_of_edges():
        return [{v} for v in nodes]
    if (cutoff < 1) or (cutoff > n):
        raise ValueError(f"cutoff must be between 1 and {n}. Got {cutoff}.")
    if best_n is not None:
        if (best_n < 1) or (best_n > n):
            raise ValueError(f"best_n must be between 1 and {n}. Got {best_n}.")
        if best_n < cutoff:
            raise ValueError(f"Must have best_n >= cutoff. Got {best_n} < {cutoff}")
        if best_n == 1:
            return [set(nodes)]
    else:
        best_n = n
    try:
        # NetworkX breaks ties between equal gains by comparing nodes
        order = sorted(range(n), key=nodes.__getitem__)
    except TypeError as exc:
        raise NotImplementedError(f"nodes that cannot be sorted: {exc}") from exc
    rank = [0] * n
    for r, i in enumerate(order):
        rank[i] = r

    weights = None if weight is None else G.edge_values_of(weight, 1)
    m = _size(G, weights)
    q0 = 1 / m
    out_degrees, in_degrees = _degrees(G, weights)
    if in_degrees is None:
        a = b = [deg * q0 * 0.5 for deg in out_degrees]
    else:
        a = [deg * q0 for deg in out_degrees]
        b = [deg * q0 for deg in in_degrees]
    arcs = G.edge_arcs
    src, dst = G.src, G.dst
    try:
        merges, exhausted = nx_greedy_modularity(
            n,
            [src[i] for i in arcs],
            [dst[i] for i in arcs],
            [1.0] * len(arcs) if weights is None else [weights[i] for i in arcs],
            a,
            b,
            rank,
            q0,
            resolution,
            G.is_directed(),
            cutoff,
            best_n,
        )
    except TypeError as exc:
        raise NotImplementedError(f"edge weights edgecase cannot use: {exc}") from exc

    # The communities after the merges: each absorbed node's community joins its absorber's
    parent = list(range(n))
    for u, v in merges:
        parent[u] = v

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    members = {}
    for i in range(n):
        members.setdefault(root(i), []).append(nodes[i])
    # (in the order of the surviving nodes, as NetworkX's dict keeps them)
    communities = [frozenset(members[r]) for r in sorted(members)]
    if exhausted:
        communities = sorted(communities, key=len, reverse=True)
        while len(communities) > best_n:
            comm1, comm2, *rest = communities
            communities = [comm1 ^ comm2]
            communities.extend(rest)
        return communities
    return sorted(communities, key=len, reverse=True)


# Leiden


def _leiden_can_run(G, *args, metric="cpm", seed=None, **kwargs):
    if G.is_directed():
        return "edgecase's Leiden is for undirected graphs"
    if metric not in {"cpm", "modularity"}:
        return f"metric {metric!r} is left to NetworkX"
    return _python_random(random._inst if seed is None else seed)


def _leiden(G, weight, metric, resolution, seed, theta):
    """Leiden levels from edgecase's kernel (`community_levels(leiden=True)`).

    The partitions are valid Leiden partitions but not NetworkX's for the same seed:
    NetworkX's implementation visits nodes in the iteration order of Python sets and draws
    its random numbers accordingly. The kernel's own generator is seeded from `seed`.
    """
    from edgecase._core import community_levels

    nodes = G.nodes
    if G.number_of_edges() == 0:
        yield [{u} for u in nodes]
        return
    rg = G.rust_graph() if weight is None else G.rust_graph(weight, default=1.0)
    ids, levels = community_levels(
        rg, True, resolution, seed.getrandbits(64), objective=metric, theta=theta
    )
    for labels in levels:
        communities = {}
        for node_id, label in zip(ids, labels):
            communities.setdefault(label, set()).add(nodes[node_id - 1])
        yield list(communities.values())


@networkx_algorithm(can_run=_leiden_can_run)
def leiden_partitions(G, *, weight="weight", metric="cpm", resolution=1.0, seed=None, theta=0.01):
    yield from _leiden(G, weight, metric, resolution, seed, theta)


@networkx_algorithm(can_run=_leiden_can_run)
def leiden_communities(
    G,
    *,
    weight="weight",
    resolution=1.0,
    max_level=None,
    seed=None,
    metric="cpm",
    theta=0.01,
):
    from collections import deque
    from itertools import islice

    partitions = _leiden(G, weight, metric, resolution, seed, theta)
    if max_level is not None:
        if max_level <= 0:
            raise ValueError("max_level argument must be a positive integer or None")
        partitions = islice(partitions, max_level)
    final_partition = deque(partitions, maxlen=1)
    return final_partition.pop()
