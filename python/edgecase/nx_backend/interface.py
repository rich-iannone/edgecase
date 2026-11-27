"""The object NetworkX loads as the edgecase backend (`networkx.backends` entry point)."""

from __future__ import annotations

import os

from edgecase.nx_backend import algorithms as _algorithms  # noqa: F401 (registers them)
from edgecase.nx_backend.graph import EdgecaseGraph, from_networkx, to_networkx
from edgecase.nx_backend.registry import algorithms

__all__ = ["BackendInterface", "backend_interface"]


def _roundtrip_enabled():
    return os.environ.get("EDGECASE_NX_ROUNDTRIP", "").lower() in {"1", "true", "yes"}


def _to_nx(obj):
    return to_networkx(obj) if isinstance(obj, EdgecaseGraph) else obj


def _dispatchable(name):
    """NetworkX's `_dispatchable` object for function `name` (or None)."""
    from networkx.utils.backends import _registered_algorithms

    func = _registered_algorithms.get(name)
    return None if func is None else getattr(func, "__wrapped__", func)


def _roundtrip_function(dispatchable):
    """Run NetworkX's own code on graphs converted to edgecase and back (tests only)."""
    signature = dispatchable.__signature__
    graphs = dispatchable.graphs
    list_graphs = dispatchable.list_graphs
    orig_func = dispatchable.orig_func

    def roundtrip(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        arguments = bound.arguments
        for gname in graphs:
            if gname not in arguments:
                continue
            if gname in list_graphs:
                arguments[gname] = [_to_nx(g) for g in arguments[gname]]
            else:
                arguments[gname] = _to_nx(arguments[gname])
        return orig_func(*bound.args, **bound.kwargs)

    roundtrip.__name__ = dispatchable.name
    roundtrip.__wrapped__ = orig_func
    return roundtrip


class BackendInterface:
    """NetworkX backend interface: conversions, test hooks and the implemented functions.

    The implemented functions are set as attributes from the registry (see
    `edgecase.nx_backend.registry`), so `hasattr(backend, name)` tells NetworkX what
    edgecase provides.
    """

    @staticmethod
    def convert_from_nx(
        graph,
        edge_attrs=None,
        node_attrs=None,
        preserve_edge_attrs=False,
        preserve_node_attrs=False,
        preserve_graph_attrs=False,
        name=None,
        graph_name=None,
    ):
        return from_networkx(
            graph,
            edge_attrs=edge_attrs,
            node_attrs=node_attrs,
            preserve_edge_attrs=preserve_edge_attrs,
            preserve_node_attrs=preserve_node_attrs,
            preserve_graph_attrs=preserve_graph_attrs,
            name=name,
            graph_name=graph_name,
        )

    @staticmethod
    def convert_to_nx(obj, *, name=None):
        if isinstance(obj, EdgecaseGraph):
            return to_networkx(obj)
        if isinstance(obj, tuple | list) and any(isinstance(x, EdgecaseGraph) for x in obj):
            return type(obj)(_to_nx(x) for x in obj)
        return obj

    @staticmethod
    def on_start_tests(items):
        from edgecase.nx_backend._testing import mark_tests

        mark_tests(items)

    def can_run(self, name, args, kwargs):
        func = algorithms.get(name)
        if func is not None:
            return func.can_run(*args, **kwargs)
        if _roundtrip_enabled():
            return _roundtrip_can_run(name, args, kwargs)
        return False

    def should_run(self, name, args, kwargs):
        func = algorithms.get(name)
        if func is not None:
            return func.should_run(*args, **kwargs)
        return True

    def __getattr__(self, name):
        # Round-trip test mode: claim every dispatchable function (see `_roundtrip_can_run`)
        if name.startswith("_") or not _roundtrip_enabled():
            raise AttributeError(name)
        dispatchable = _dispatchable(name)
        if dispatchable is None:
            raise AttributeError(name)
        func = _roundtrip_function(dispatchable)
        setattr(self, name, func)
        return func


def _roundtrip_can_run(name, args, kwargs):
    dispatchable = _dispatchable(name)
    if dispatchable is None or not dispatchable.graphs:
        return "no graph arguments to convert"
    if dispatchable._will_call_mutate_input(args, kwargs):
        return "mutates its input graph"
    return True


for _name, _func in algorithms.items():
    setattr(BackendInterface, _name, staticmethod(_func))

backend_interface = BackendInterface()
