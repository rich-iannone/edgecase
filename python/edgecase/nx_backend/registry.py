"""Registry of the NetworkX functions the edgecase backend implements.

An implementation is a plain function with NetworkX's name and signature (NetworkX calls
it with the arguments bound to its own signature, defaults applied and graphs already
converted to `EdgecaseGraph`). Decorate it with `@networkx_algorithm` to register it:

    @networkx_algorithm
    def number_connected_components(G):
        ...

`can_run=` takes a function with the same signature that returns True, or False or a
string saying why edgecase cannot run the call (NetworkX then uses another backend).
"""

from __future__ import annotations

from functools import partial

__all__ = ["algorithms", "networkx_algorithm"]

# NetworkX function name -> implementation
algorithms = {}


def _always(*args, **kwargs):
    return True


def networkx_algorithm(func=None, *, name=None, can_run=None, should_run=None):
    """Register `func` as the edgecase implementation of the NetworkX function `name`."""
    if func is None:
        return partial(networkx_algorithm, name=name, can_run=can_run, should_run=should_run)
    name = func.__name__ if name is None else name
    if name in algorithms:
        raise ValueError(f"{name!r} is already registered")
    func.networkx_name = name
    func.can_run = _always if can_run is None else can_run
    func.should_run = _always if should_run is None else should_run
    algorithms[name] = func
    return func
