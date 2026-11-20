"""The guard on the backend's model of CPython's set order.

NetworkX's `find_cliques`, several `greedy_color` strategies, `maximal_independent_set`
and `dominating_set` take nodes in the iteration and `pop()` order of Python sets, so that
order decides which cliques come first, which colors nodes get and which nodes are picked.
The backend reproduces it with a Rust model of CPython's set tables (`PySetModel` in
`src/algorithms/nx_cliques.rs`) fed with the node keys' hashes.

The model follows CPython's `Objects/setobject.c`. What it depends on (the table size,
probing and resize rules, the dummy and `pop()` bookkeeping, and the algorithms of `copy`,
`&`, `-`, `&=`, `-=` and `difference_update`) is the same in CPython 3.10 through 3.14.
Other interpreters and versions are not verified, so these calls are left to NetworkX
there. On first use a self-test also replays set operations on real sets (with int, string
and tuple keys) and leaves the calls to NetworkX if any order differs.

If exact order is not wanted, a deterministic Bron-Kerbosch with pivoting and plain node
orders for the greedy strategies would replace the model, at the cost of NetworkX's order.
"""

from __future__ import annotations

import functools
import platform
import random
import sys

__all__ = ["set_order_can_run", "set_order_reason"]

# The CPython versions whose `Objects/setobject.c` was checked against the model
VERIFIED_VERSIONS = ((3, 10), (3, 14))

_OPS = ["add"] * 6 + ["discard"] * 2 + ["pop", "copy", "and", "iand", "sub", "isub", "isub_iter"]
_SETS = 4


@functools.cache
def set_order_reason():
    """None if the set model gives this interpreter's set order, else why not."""
    if sys.implementation.name != "cpython":
        name = platform.python_implementation()
        return f"NetworkX's set order is modeled for CPython only, not {name}"
    version = sys.version_info[:2]
    low, high = VERIFIED_VERSIONS
    if not low <= version <= high:
        return "NetworkX's set order is not verified for Python {}.{}".format(*version)
    try:
        ok = _self_test()
    except Exception as exc:  # (a broken model must not break the call)
        return f"the set order self-test failed: {exc!r}"
    if not ok:
        return "the set order model does not match this interpreter's sets"
    return None


def set_order_can_run(*args, **kwargs):
    """`can_run` for a function whose result follows NetworkX's set order."""
    reason = set_order_reason()
    return True if reason is None else reason


def _replay(keys, ops):
    """`ops` (as `nx_pyset_model_check` takes them) on real sets."""
    real = [set() for _ in range(_SETS)]
    popped = []
    for op, t, a in ops:
        if op == "add":
            real[t].add(keys[a])
        elif op == "discard":
            real[t].discard(keys[a])
        elif op == "pop":
            popped.append(real[t].pop() if real[t] else None)
        elif op == "copy":
            real[t] = real[a].copy()
        elif op == "and":
            real[t] = real[t] & real[a]
        elif op == "iand":
            real[t] &= real[a]
        elif op == "sub":
            real[t] = real[t] - real[a]
        elif op == "isub":
            real[t] -= real[a]
        else:
            real[t].difference_update(list(real[a]))
    return real, popped


def _random_ops(rng, n_keys, n_ops):
    ops = []
    for _ in range(n_ops):
        op = rng.choice(_OPS)
        t = rng.randrange(_SETS)
        if op in ("add", "discard"):
            ops.append((op, t, rng.randrange(n_keys)))
        elif op == "pop":
            ops.append((op, t, 0))
        else:
            ops.append((op, t, rng.choice([s for s in range(_SETS) if s != t])))
    return ops


def _matches(keys, ops):
    from edgecase._core import nx_pyset_model_check

    index = {k: i for i, k in enumerate(keys)}
    real, popped = _replay(keys, ops)
    orders, model_popped = nx_pyset_model_check([hash(k) for k in keys], _SETS, ops)
    return orders == [[index[k] for k in s] for s in real] and model_popped == [
        None if k is None else index[k] for k in popped
    ]


def _self_test():
    rng = random.Random(8)  # (not the global generator: its state belongs to the caller)
    # Small and negative ints (hash(-1) == hash(-2)), ints colliding in the low bits and
    # with the small ones modulo 2**61 - 1, and strings and tuples (randomized hashes)
    keys = [
        *range(-2, 64),
        *(8 * i for i in range(8, 64)),
        *(2**61 + i for i in range(-3, 3)),
        *(f"v{i}" for i in range(96)),
        *((i, "e") for i in range(32)),
    ]
    for n_ops in (60, 400, 1500):
        if not _matches(keys, _random_ops(rng, len(keys), n_ops)):
            return False
    # Past 50000 members, tables grow by 2x instead of 4x (here from 78643 members on)
    keys = [*range(0, 158000, 2), *(f"w{i}" for i in range(500))]
    n = len(keys)
    ops = [("add", 0, (i * 7919) % n) for i in range(n)]
    ops += [("discard", 0, i) for i in range(0, n, 3)]
    ops += [("pop", 0, 0)] * 20 + [("copy", 1, 0)]
    ops += [("add", 2, i) for i in range(1, n, 5)]
    ops += [("isub", 1, 2), ("and", 3, 0), ("iand", 0, 2), ("isub_iter", 2, 0)]
    return _matches(keys, ops)
