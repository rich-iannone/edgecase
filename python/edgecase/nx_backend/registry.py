"""Registry of the NetworkX functions the edgecase backend implements.

An implementation is a plain function with NetworkX's name and signature (NetworkX calls
it with the arguments bound to its own signature, defaults applied and graphs already
converted to `EdgecaseGraph`). Decorate it with `@networkx_algorithm` to register it:

    @networkx_algorithm
    def number_connected_components(G):
        ...

`can_run=` takes a function with the same signature that returns True, or False or a
string saying why edgecase cannot run the call (NetworkX then uses another backend).
`should_run=` is the same for calls edgecase can run but should not, which NetworkX only
consults when it picks backends automatically (not with `backend="edgecase"`, and not in
its test mode). `linear_time=True` sets it to `prefer_converted_graph`: for functions as
cheap as converting the graph, run only when the graph needs no conversion.
"""

from __future__ import annotations

from functools import partial

__all__ = ["algorithms", "networkx_algorithm", "prefer_converted_graph"]

# NetworkX function name -> implementation
algorithms = {}


def _always(*args, **kwargs):
    return True


def prefer_converted_graph(G, *args, **kwargs):
    """`should_run` for linear-time functions: only on graphs that need no conversion.

    Converting a NetworkX graph costs about as much as one linear-time NetworkX call, so
    edgecase only pays off when `G` is already an edgecase graph or NetworkX has a
    conversion of it cached (`G.__networkx_cache__`, filled by an earlier edgecase call).
    """
    if getattr(G, "__networkx_backend__", None) == "edgecase":
        return True
    cache = getattr(G, "__networkx_cache__", None)
    if cache and cache.get("backends", {}).get("edgecase"):
        return True
    return (
        "a linear-time function on a graph not yet converted to edgecase: the conversion "
        "would cost about as much as running NetworkX"
    )


def networkx_algorithm(func=None, *, name=None, can_run=None, should_run=None, linear_time=False):
    """Register `func` as the edgecase implementation of the NetworkX function `name`."""
    if func is None:
        return partial(
            networkx_algorithm,
            name=name,
            can_run=can_run,
            should_run=should_run,
            linear_time=linear_time,
        )
    if linear_time and should_run is None:
        should_run = prefer_converted_graph
    name = func.__name__ if name is None else name
    if name in algorithms:
        raise ValueError(f"{name!r} is already registered")
    func.networkx_name = name
    func.can_run = _always if can_run is None else can_run
    func.should_run = _always if should_run is None else should_run
    algorithms[name] = func
    return func
