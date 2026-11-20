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
    "is_partition",
    "label_propagation_communities",
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


def _size(G, weights):
    """`G.size(weight)` with NetworkX's summation order (weights per arc)."""
    if weights is None:
        return G.number_of_edges()
    src, dst = G.src, G.dst
    n = len(G.nodes)
    starts = [0] * (n + 2)
    for s in src:
        starts[s + 1] += 1
    for i in range(1, n + 2):
        starts[i] += starts[i - 1]
    degrees = []
    if G.is_directed():
        # predecessors' weights in NetworkX's predecessor order when it was recorded
        if G.pred_order is not None:
            by_pair = {}
            for s, d, w in zip(src, dst, weights):
                by_pair.setdefault((s, d), []).append(w)
            key_to_id = G.key_to_id
            pred_nodes, counts, preds = G.pred_order
            in_weights = {}
            k = 0
            for v, count in zip(pred_nodes, counts):
                vid = key_to_id[v]
                ws = []
                for u in preds[k : k + count]:
                    ws.extend(by_pair[(key_to_id[u], vid)])
                k += count
                in_weights[vid] = ws
        else:
            in_weights = {}
            for s, d, w in zip(src, dst, weights):
                in_weights.setdefault(d, []).append(w)
        for u in range(1, n + 1):
            out = weights[starts[u] : starts[u + 1]]
            degrees.append(sum(out) + sum(in_weights.get(u, ())))
    else:
        multigraph = G.is_multigraph()
        for u in range(1, n + 1):
            arcs = range(starts[u], starts[u + 1])
            deg = sum(weights[starts[u] : starts[u + 1]])
            loops = [weights[i] for i in arcs if dst[i] == u]
            if loops:
                deg = deg + (sum(loops) if multigraph else loops[0])
            degrees.append(deg)
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
