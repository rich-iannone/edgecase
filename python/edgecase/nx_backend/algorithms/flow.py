"""Maximum flows and minimum cuts.

edgecase's Edmonds-Karp (`max_flow_details`) computes the flow on NetworkX's residual
network conventions: zero-capacity, hidden (`None`) and self-loop edges dropped, missing
capacities infinite and replaced by NetworkX's large finite stand-in, an infinite-capacity
path reported as unbounded. The flow value and the minimum cut (the side that can still
reach the sink, which is the same for every maximum flow) are therefore NetworkX's; the flow
assignment itself is a valid maximum flow but may differ from the one NetworkX's chosen
algorithm finds. Flow functions given as `flow_func` (NetworkX's own ones) are accepted for
the value and the cut; their keyword arguments, custom flow functions, callable capacities
and multigraphs are left to NetworkX.
"""

from __future__ import annotations

from edgecase.nx_backend.registry import networkx_algorithm

__all__ = ["maximum_flow", "maximum_flow_value", "minimum_cut", "minimum_cut_value"]


def _flow_can_run(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    import networkx as nx

    if flowG.is_multigraph():
        return "NetworkX raises for multigraphs"
    if callable(capacity):
        return "a callable capacity is left to NetworkX"
    if kwargs:
        return "flow-function keyword arguments are left to NetworkX"
    flow = nx.algorithms.flow
    builtin = {
        flow.edmonds_karp,
        flow.shortest_augmenting_path,
        flow.preflow_push,
        flow.dinitz,
        flow.boykov_kolmogorov,
    }
    if flow_func is not None and flow_func not in builtin:
        return "a custom flow_func is left to NetworkX"
    return True


def _max_flow(G, s, t, capacity):
    """(value, signed flow per edge arc, sink side as node IDs) on NetworkX's conventions."""
    import networkx as nx

    from edgecase._core import Graph as RustGraph
    from edgecase._core import max_flow_details

    if s not in G:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if s == t:
        raise nx.NetworkXError("source and sink are the same node")
    inf = float("inf")
    src, dst = G.src, G.dst
    values = G.edge_values_of(capacity, inf) if src else []
    # NetworkX keeps edges whose capacity is not None and positive, and replaces infinite
    # capacities by 3 * (sum of the finite ones) or 1
    caps = [c if (u != v and c is not None and c > 0) else 0 for u, v, c in zip(src, dst, values)]
    finite = [c for k, c in zip(G.edge_arcs, (caps[k] for k in G.edge_arcs)) if c and c != inf]
    big = 3 * sum(finite) or 1
    # An infinite-capacity path from s to t makes the flow unbounded
    unbounded = [k for k, c in enumerate(caps) if c == inf]
    if unbounded:
        nbrs = {}
        for k in unbounded:
            nbrs.setdefault(src[k], []).append(dst[k])
            if not G.is_directed():
                nbrs.setdefault(dst[k], []).append(src[k])
        sid, tid = G.key_to_id[s], G.key_to_id[t]
        seen, stack = {sid}, [sid]
        while stack:
            for w in nbrs.get(stack.pop(), ()):
                if w == tid:
                    raise nx.NetworkXUnbounded("Infinite capacity path, flow unbounded above.")
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
    caps = [min(c, big) for c in caps]
    try:
        rg = RustGraph.from_edge_arrays(
            len(G.nodes),
            src,
            dst,
            caps,
            directed=G.is_directed(),
            symmetric=not G.is_directed(),
            require_weights=True,
        )
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(f"capacities edgecase cannot use: {exc}") from exc
    value, flows, _, sink_side = max_flow_details(rg, G.key_to_id[s], G.key_to_id[t])
    integral = all(isinstance(c, int) for c in caps)
    if integral:
        value = int(round(value))
    return value, flows, sink_side, integral


@networkx_algorithm(can_run=_flow_can_run)
def maximum_flow_value(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    return _max_flow(flowG, _s, _t, capacity)[0]


@networkx_algorithm(can_run=_flow_can_run)
def minimum_cut_value(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    return _max_flow(flowG, _s, _t, capacity)[0]


@networkx_algorithm(can_run=_flow_can_run)
def minimum_cut(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    value, _, sink_side, _ = _max_flow(flowG, _s, _t, capacity)
    nodes = flowG.nodes
    non_reachable = {nodes[i - 1] for i in sink_side}
    partition = (set(nodes) - non_reachable, non_reachable)
    return (value, partition)


@networkx_algorithm(can_run=_flow_can_run)
def maximum_flow(flowG, _s, _t, capacity="capacity", flow_func=None, **kwargs):
    G = flowG
    value, flows, _, integral = _max_flow(G, _s, _t, capacity)
    nodes, src, dst = G.nodes, G.src, G.dst
    arcs = G.edge_arcs
    positive = {}
    for edge_id, flow in flows:
        if integral:
            flow = int(round(flow))
        if flow == 0:
            continue
        k = arcs[edge_id - 1]
        u, v = src[k], dst[k]
        if flow < 0:
            u, v, flow = v, u, -flow
        positive[(u, v)] = positive.get((u, v), 0) + flow
    # build_flow_dict: every neighbor of every node, with the flow where it is positive
    flow_dict = {}
    k = 0
    for u in range(1, len(nodes) + 1):
        row = {}
        while k < len(src) and src[k] == u:
            v = dst[k]
            row[nodes[v - 1]] = positive.get((u, v), 0)
            k += 1
        flow_dict[nodes[u - 1]] = row
    return (value, flow_dict)
