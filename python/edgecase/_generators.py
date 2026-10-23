"""Pure-Python helpers for the generators that work on Python objects or bundled data.

These build plain index structures (node counts, `(i, j)` index pairs, per-node values)
that `Graph` methods turn into nodes and edges, like the Rust generators do:

- `nonisomorphic_tree_edges()` / `count_nonisomorphic_trees()`: free trees by level
  sequence (Wright, Richmond, Odlyzko and McKay 1986), as in NetworkX
- `prefix_tree_structure()`: the trie of a list of paths, numbered as NetworkX does
- `random_kernel_edges()`: the inhomogeneous random graph of Bollobás, Janson and Riordan,
  with NetworkX's sampling scheme and `random.Random` stream
- `atlas_entries()`: the 1253 graphs of "An Atlas of Graphs", from `static/atlas.dat.gz`
"""

from __future__ import annotations

import gzip
import math
import random
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path

ATLAS_FILE = Path(__file__).parent / "static" / "atlas.dat.gz"
NUM_ATLAS_GRAPHS = 1253


# --- nonisomorphic trees ------------------------------------------------------------


def _next_rooted_tree(predecessor: list[int], p: int | None = None) -> list[int] | None:
    """The level sequence after `predecessor` in reverse lexicographic order, or `None`."""
    if p is None:
        p = len(predecessor) - 1
        while predecessor[p] == 1:
            p -= 1
    if p == 0:
        return None
    q = p - 1
    while predecessor[q] != predecessor[p] - 1:
        q -= 1
    result = list(predecessor)
    for i in range(p, len(result)):
        result[i] = result[i - p + q]
    return result


def _split_tree(layout: list[int]) -> tuple[list[int], list[int]]:
    """The root's first subtree (levels shifted up) and the rest of the tree."""
    m = len(layout)
    one_found = False
    for i, level in enumerate(layout):
        if level == 1:
            if one_found:
                m = i
                break
            one_found = True
    left = [layout[i] - 1 for i in range(1, m)]
    rest = [0] + layout[m:]
    return left, rest


def _next_tree(candidate: list[int]) -> list[int] | None:
    """The first level sequence from `candidate` on that is a canonical free tree."""
    left, rest = _split_tree(candidate)
    left_height, rest_height = max(left), max(rest)
    valid = rest_height >= left_height
    if valid and rest_height == left_height:
        if len(left) > len(rest) or (len(left) == len(rest) and left > rest):
            valid = False
    if valid:
        return candidate
    p = len(left)
    new_candidate = _next_rooted_tree(candidate, p)
    if candidate[p] > 2:
        new_left, _ = _split_tree(new_candidate)
        suffix = list(range(1, max(new_left) + 2))
        new_candidate[-len(suffix) :] = suffix
    return new_candidate


def _layout_edges(layout: list[int]) -> list[tuple[int, int]]:
    """The `(parent, child)` pairs of a level sequence, children in sequence order."""
    edges = []
    stack: list[int] = []
    for i, level in enumerate(layout):
        while stack and layout[stack[-1]] >= level:
            stack.pop()
        if stack:
            edges.append((stack[-1], i))
        stack.append(i)
    return edges


def nonisomorphic_tree_edges(order: int) -> Iterator[list[tuple[int, int]]]:
    """Edge lists of every tree on `order >= 2` nodes up to isomorphism, as NetworkX."""
    # start at the path rooted at its center
    layout = list(range(order // 2 + 1)) + list(range(1, (order + 1) // 2))
    while layout is not None:
        layout = _next_tree(layout)
        if layout is not None:
            yield _layout_edges(layout)
            layout = _next_rooted_tree(layout)


@lru_cache(None)
def _rooted_trees(n: int) -> int:
    """Number of rooted unlabeled trees on `n` nodes (OEIS A000081)."""
    if n < 2:
        return n
    value = 0
    for j in range(1, n):
        for d in range(1, n):
            if j % d == 0:
                value += d * _rooted_trees(d) * _rooted_trees(n - j)
    return value // (n - 1)


def count_nonisomorphic_trees(order: int) -> int:
    """Number of free unlabeled trees on `order` nodes (OEIS A000055), by Otter's formula."""
    value = sum(_rooted_trees(k) * _rooted_trees(order - k) for k in range(order + 1))
    if order % 2 == 0:
        value -= _rooted_trees(order // 2)
    return _rooted_trees(order) - value // 2


# --- prefix tree --------------------------------------------------------------------


def prefix_tree_structure(paths) -> tuple[list, list[tuple[int, int]]]:
    """
    The trie of `paths`: each node's source value and the `(parent, child)` index pairs.

    Index 0 is the root (source `None`) and index 1 the NIL node (source `"NIL"`) that every
    path ends at; the others are numbered depth first, children in order of first
    appearance, so index `k + 1` is NetworkX's node `k`.
    """
    sources: list = [None, "NIL"]
    edges: list[tuple[int, int]] = []

    def children_of(parent: int, remaining) -> dict:
        children: dict = {}
        ends = False
        for path in remaining:
            path = list(path)
            if not path:
                # a repeated path ends here only once
                if not ends:
                    edges.append((parent, 1))
                    ends = True
                continue
            children.setdefault(path[0], []).append(path[1:])
        return children

    stack = [(0, iter(children_of(0, paths).items()))]
    while stack:
        parent, remaining_children = stack[-1]
        try:
            child, remaining_paths = next(remaining_children)
        except StopIteration:
            stack.pop()
            continue
        index = len(sources)
        sources.append(child)
        edges.append((parent, index))
        stack.append((index, iter(children_of(index, remaining_paths).items())))
    return sources, edges


# --- random kernel graph ------------------------------------------------------------


def _bisect_root(f: Callable[[float], float], lo: float, hi: float) -> float:
    """A root of `f` in `[lo, hi]`, where `f(lo) < 0 <= f(hi)`, to full float precision."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid <= lo or mid >= hi:
            break
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return hi


def random_kernel_edges(
    n: int,
    kernel_integral: Callable[[float, float, float], float],
    kernel_root: Callable[[float, float, float], float] | None,
    seed: int | None,
) -> list[tuple[int, int]]:
    """
    Edges of the random kernel graph, by NetworkX's `random_kernel_graph()` scheme.

    The random stream is `random.Random(seed)`, as NetworkX's `py_random_state` makes it, so
    with the same `kernel_root` the edges are NetworkX's. Without one, the root is found by
    bisection to full precision where NetworkX uses SciPy's `brentq`.
    """
    if kernel_root is None:

        def kernel_root(y: float, a: float, r: float) -> float:
            return _bisect_root(lambda b: kernel_integral(y, a, b) - r, a, 1.0)

    rng = random.Random(seed)
    edges = []
    i = j = 1
    while i < n:
        r = -math.log(1 - rng.random())
        if kernel_integral(i / n, j / n, 1) <= r:
            i = j = i + 1
        else:
            j = math.ceil(n * kernel_root(i / n, j / n, r))
            edges.append((i - 1, j - 1))
    return edges


# --- graph atlas --------------------------------------------------------------------


@lru_cache(None)
def atlas_entries() -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    """The node count and edge list of each atlas graph, in atlas order."""
    entries = []
    current: tuple[int, list[tuple[int, int]]] | None = None
    with gzip.open(ATLAS_FILE, "rt", encoding="ascii") as f:
        for line in f:
            if line.startswith("GRAPH"):
                if current is not None:
                    entries.append((current[0], tuple(current[1])))
                current = (0, [])
            elif line.startswith("NODES"):
                current = (int(line[6:]), current[1])
            elif line.strip():
                u, v = line.split()
                current[1].append((int(u), int(v)))
    entries.append((current[0], tuple(current[1])))
    assert len(entries) == NUM_ATLAS_GRAPHS, len(entries)
    return tuple(entries)
