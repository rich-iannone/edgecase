"""Markers for NetworkX's test suite when it runs with edgecase as the backend.

NetworkX calls `BackendInterface.on_start_tests(items)` after collecting its tests (with
`NETWORKX_TEST_BACKEND=edgecase`). Tests listed here get an xfail or skip marker with the
reason. Keys are `<path from the networkx package>::<test>` with the test's class (if any)
and name, optionally with its parameters, e.g.
`algorithms/tests/test_cycles.py::TestMinimumCycleBasis::test_weighted_diamond`.
"""

from __future__ import annotations

import os
from pathlib import Path

# Reasons
MUTATES_DURING_ITERATION = (
    "the test mutates the input graph while iterating over the result and expects the "
    "iteration to notice; a graph converted to edgecase cannot see later changes"
)

DIFFERENT_LEIDEN_PARTITION = (
    "expects NetworkX's exact Leiden partition for a seed; NetworkX visits nodes in the "
    "iteration order of Python sets and draws its random numbers accordingly, while "
    "edgecase's Leiden finds a different valid (connected, quality-improving) partition"
)
DIFFERENT_LEIDEN_LEVELS = (
    "counts NetworkX's Leiden levels (it yields the refined partition of every "
    "aggregation level); edgecase's Leiden yields only the levels at which its partition "
    "changes, so it can report fewer levels for the same final partition"
)

# test key -> reason
XFAIL = {
    "algorithms/community/tests/test_leiden.py::test_expected_stable_across_code_changes_cpm": (
        DIFFERENT_LEIDEN_PARTITION
    ),
    "algorithms/community/tests/test_leiden.py::test_cpm_node_weights_on_G": (
        DIFFERENT_LEIDEN_PARTITION
    ),
    "algorithms/community/tests/test_leiden.py::test_weights_matter": DIFFERENT_LEIDEN_LEVELS,
    # topological_sort is implemented by edgecase
    "algorithms/tests/test_dag.py::TestDAG::test_topological_sort6": MUTATES_DURING_ITERATION,
}
SKIP = {}

# Only in round-trip mode (EDGECASE_NX_ROUNDTRIP=1), where every dispatchable function
# converts its graphs
ROUNDTRIP_XFAIL = {}


def _test_keys(item):
    """The keys `item` may be listed under (with and without its parameters)."""
    import networkx as nx

    root = Path(nx.__file__).parent
    try:
        path = Path(item.path).relative_to(root).as_posix()
    except ValueError:
        path = Path(item.path).name
    names = item.nodeid.split("::")[1:]
    key = "::".join([path, *names])
    keys = [key]
    if "[" in key:
        keys.append(key[: key.index("[")])
    return keys


def mark_tests(items):
    xfail = dict(XFAIL)
    if os.environ.get("EDGECASE_NX_ROUNDTRIP", "").lower() in {"1", "true", "yes"}:
        xfail.update(ROUNDTRIP_XFAIL)
    if not xfail and not SKIP:
        return
    import pytest

    for item in items:
        for key in _test_keys(item):
            if key in xfail:
                item.add_marker(pytest.mark.xfail(reason=xfail[key], strict=False))
                break
            if key in SKIP:
                item.add_marker(pytest.mark.skip(reason=SKIP[key]))
                break
