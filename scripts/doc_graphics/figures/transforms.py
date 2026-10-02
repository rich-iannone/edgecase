"""Transforms: an input graph → the new graph the method derives from it.

Template: `transform`. Left: the input `g`; right: the returned graph, which
is always a new object (`g` is untouched). Wherever the derived graph keeps
the input's nodes, both panels share one layout so only the change moves.
Elements the method creates are `accent` on the right; elements it drops are
`accent` on the left and `removed` (dashed) on the right, as in mutations.
For extractions (`subgraph`, `edge_subgraph`) the selection you pass is
`soft` on the left and the extracted part is `accent` on the right. When the
output is built from different things (line graph, quotient), group colors
tie each output node to the input element it came from.
"""

from core import SIZE_NOTE, Scene, figure, place, transform

UNIT = 80  # one density for the whole family

# Four nodes on a wide square: the base for direction changes.
SQUARE = {"A": (0, 0), "B": (1.5, 0), "C": (1.5, 1.1), "D": (0, 1.1)}

# Six nodes, two rows: the base for the two extraction methods.
GRID = {"A": (0, 0), "B": (1.3, 0), "C": (2.6, 0), "D": (0, 1.2), "E": (1.3, 1.2), "F": (2.6, 1.2)}
# Insertion order fixes the edge ids 1..7 used by edge_subgraph.
GRID_EDGES = [("A", "B"), ("B", "C"), ("A", "D"), ("B", "E"), ("D", "E"), ("E", "F"), ("C", "F")]

# Four nodes in a row: the base for transitive closure / reduction.
CHAIN = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0)}
CHAIN_EDGES = [("A", "B"), ("B", "C"), ("C", "D")]
# The implied edges, as arcs: above for two-step pairs, below for B→D, highest for A→D.
CHAIN_ARCS = [("A", "C", -24), ("A", "D", -58), ("B", "D", 26)]


def _both(coords, left, right, unit: float = UNIT):
    return place(coords, left, unit=unit), place(coords, right, unit=unit)


@figure("reverse", "transforms", "Reverse edges",
        "A directed graph on A, B, C, D with arrows A to B, B to C, D to C, and A to D. "
        "After reverse(), the same four edges point the opposite way, B to A, C to B, "
        "C to D, and D to A, all drawn in the accent color.")
def reverse(s: Scene) -> None:
    left, right = transform(s, "g", "g.reverse()")
    lp, rp = _both(SQUARE, left, right)
    edges = [("A", "B"), ("B", "C"), ("D", "C"), ("A", "D")]
    s.graph(lp, edges, directed=True, prefix="l.")
    s.graph(rp, [(b, a, {"style": "accent"}) for a, b in edges], directed=True, prefix="r.")


@figure("to_directed", "transforms", "Convert to directed",
        "An undirected path A, B, C, D with three plain edges. After to_directed(), each "
        "undirected edge has become a pair of accent arrows, one in each direction, so the "
        "three edges become six.")
def to_directed(s: Scene) -> None:
    left, right = transform(s, "g", "g.to_directed()")
    lp, rp = _both(SQUARE, left, right)
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    s.graph(lp, edges, prefix="l.")
    arcs = []
    for a, b in edges:
        arcs += [(a, b, {"style": "accent", "bend": 9}), (b, a, {"style": "accent", "bend": 9})]
    s.graph(rp, arcs, directed=True, prefix="r.")


@figure("to_undirected", "transforms", "Convert to undirected",
        "A directed graph with a mutual pair of accent arrows between A and B, plus arrows "
        "A to D and D to C. After to_undirected(), the mutual pair collapses into a single "
        "accent edge A to B, and the other two edges simply lose their arrowheads.")
def to_undirected(s: Scene) -> None:
    left, right = transform(s, "g", "g.to_undirected()")
    lp, rp = _both(SQUARE, left, right)
    s.graph(lp, [("A", "B", {"style": "accent", "bend": 9}), ("B", "A", {"style": "accent", "bend": 9}),
                 ("A", "D"), ("D", "C")], directed=True, prefix="l.")
    s.graph(rp, [("A", "B", {"style": "accent"}), ("A", "D"), ("D", "C")], prefix="r.")


@figure("complement", "transforms", "Complement",
        "A four-node path A, B, C, D. In the complement every pair that was not connected "
        "becomes an accent edge (A to C, A to D, B to D), while the three original edges "
        "are dropped and shown dashed.")
def complement(s: Scene) -> None:
    left, right = transform(s, "g", "g.complement()")
    # D sits inside triangle A, B, C so that path + complement (= K4) has no crossings.
    coords = {"B": (1, 0), "A": (0, 1.55), "C": (2, 1.55), "D": (1, 1.0)}
    lp, rp = _both(coords, left, right)
    path = [("A", "B"), ("B", "C"), ("C", "D")]
    s.graph(lp, path, prefix="l.")
    s.graph(rp, [(a, b, {"style": "removed"}) for a, b in path]
            + [(a, b, {"style": "accent"}) for a, b in (("A", "C"), ("A", "D"), ("B", "D"))],
            prefix="r.")


@figure("subgraph", "transforms", "Induced subgraph",
        "A six-node graph with node ids 1 to 6; nodes 1, 2, 4, 5 are the ones requested. "
        "The induced subgraph keeps those four nodes and every edge among them (1-2, 1-4, "
        "2-5, 4-5), in the accent color; nodes 3 and 6 and their edges are dropped and "
        "shown dashed.")
def subgraph(s: Scene) -> None:
    left, right = transform(s, "g", "g.subgraph([1, 2, 4, 5])")
    lp, rp = _both(GRID, left, right)
    ids = {k: str(i) for i, k in enumerate("ABCDEF", 1)}  # node ids follow insertion order
    keep = set("ABDE")
    s.graph(lp, GRID_EDGES, styles={k: "soft" for k in keep}, labels=ids, prefix="l.")
    s.graph(rp, [(a, b, {"style": "accent" if {a, b} <= keep else "removed"}) for a, b in GRID_EDGES],
            styles={k: "accent" if k in keep else "removed" for k in GRID}, labels=ids, prefix="r.")


@figure("edge_subgraph", "transforms", "Edge-induced subgraph",
        "The same six-node graph with its edges numbered 1 to 7; edges 1, 4 and 5 are "
        "requested. The result keeps exactly those three accent edges and their endpoints "
        "A, B, D, E. Edge 3 between A and D is dropped even though both endpoints stay, "
        "which is what separates this from an induced subgraph.")
def edge_subgraph(s: Scene) -> None:
    left, right = transform(s, "g", "g.edge_subgraph([1, 4, 5])")
    lp, rp = _both(GRID, left, right)
    pick = {1, 4, 5}
    keep = set("ABDE")
    s.graph(lp, [(a, b, {"weight": i, "style": "soft" if i in pick else "default"})
                 for i, (a, b) in enumerate(GRID_EDGES, 1)], prefix="l.")
    s.graph(rp, [(a, b, {"weight": i, "style": "accent"} if i in pick else {"style": "removed"})
                 for i, (a, b) in enumerate(GRID_EDGES, 1)],
            styles={k: "accent" if k in keep else "removed" for k in GRID}, prefix="r.")


@figure("line_graph", "transforms", "Line graph",
        "A four-node graph whose edges AB, BC, CD and BD each have their own color. In the "
        "line graph each of those edges becomes a node of the same color, named after it, and two "
        "nodes are joined when their edges share an endpoint; AB and CD share none, so they "
        "stay unconnected.")
def line_graph(s: Scene) -> None:
    left, right = transform(s, "g", "g.line_graph()")
    color = {"AB": "accent", "BC": "g2", "CD": "g3", "BD": "g4"}
    lp = place({"A": (0, 0.6), "B": (1.3, 0.6), "C": (2.4, 0), "D": (2.4, 1.2)}, left, unit=UNIT)
    s.graph(lp, [(e[0], e[1], {"style": c, "weight": e, "weight_t": 0.5}) for e, c in color.items()],
            prefix="l.")
    rp = place({"AB": (0, 0.6), "BC": (1.2, 0), "BD": (1.2, 1.2), "CD": (2.4, 0.6)}, right, unit=UNIT)
    # Adjacent when the edges share an endpoint (checked against edgecase: 5 edges).
    s.graph(rp, [("AB", "BC"), ("AB", "BD"), ("BC", "CD"), ("BC", "BD"), ("CD", "BD")],
            styles=color, prefix="r.")


@figure("quotient_graph", "transforms", "Quotient graph",
        "A six-node graph partitioned into three blocks, shown as tinted regions: A, B, C "
        "in the accent color, D, E in teal, and F in violet. The quotient has one node per "
        "block, labeled block_0, block_1 and block_2; edges inside a block vanish and the "
        "several edges between two blocks merge into one.")
def quotient_graph(s: Scene) -> None:
    left, right = transform(s, "g", "g.quotient_graph(blocks)")
    grp = {**dict.fromkeys("ABC", "accent"), **dict.fromkeys("DE", "g2"), "F": "g3"}
    coords = {"A": (0, 0), "B": (0, 1.2), "C": (0.9, 0.6), "D": (1.9, 0), "E": (1.9, 1.2), "F": (2.8, 0.6)}
    edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "E"), ("D", "F"), ("E", "F")]
    s.graph(place(coords, left, unit=UNIT),
            [(a, b, {"style": grp[a]}) if grp[a] == grp[b] else (a, b) for a, b in edges],
            styles=grp, prefix="l.")
    for block in ("ABC", "DE", "F"):
        s.hull(["l." + k for k in block], grp[block[0]])
    # One node per block, labeled with its members; the returned label sits underneath.
    rp = place({"ABC": (0, 0), "DE": (1.4, 0), "F": (2.8, 0)}, right, unit=UNIT)
    s.graph(rp, [("ABC", "DE"), ("DE", "F")], styles={"ABC": "accent", "DE": "g2", "F": "g3"},
            prefix="r.")
    for i, (k, (x, y)) in enumerate(rp.items()):
        s.text(x, y + 30, f"block_{i}", size=SIZE_NOTE, opacity=0.7)


@figure("transitive_closure", "transforms", "Transitive closure",
        "A directed chain A to B to C to D. The transitive closure adds an accent arrow for "
        "every pair joined by a longer path: A to C, B to D, and A to D.")
def transitive_closure(s: Scene) -> None:
    left, right = transform(s, "g", "g.transitive_closure()")
    lp, rp = _both(CHAIN, left, right)
    s.graph(lp, CHAIN_EDGES, directed=True, prefix="l.")
    s.graph(rp, CHAIN_EDGES + [(a, b, {"style": "accent", "bend": k}) for a, b, k in CHAIN_ARCS],
            directed=True, prefix="r.")


@figure("transitive_reduction", "transforms", "Transitive reduction",
        "A DAG on A, B, C, D: the chain A to B to C to D plus three shortcut arrows A to C, "
        "B to D and A to D, in the accent color. The reduction drops the shortcuts, shown "
        "dashed, because the chain already reaches the same nodes.")
def transitive_reduction(s: Scene) -> None:
    left, right = transform(s, "g", "g.transitive_reduction()")
    lp, rp = _both(CHAIN, left, right)
    s.graph(lp, CHAIN_EDGES + [(a, b, {"style": "accent", "bend": k}) for a, b, k in CHAIN_ARCS],
            directed=True, prefix="l.")
    s.graph(rp, CHAIN_EDGES + [(a, b, {"style": "removed", "bend": k}) for a, b, k in CHAIN_ARCS],
            directed=True, prefix="r.")


@figure("transform_with_spanner", "transforms", "Spanner",
        "A six-node ladder with two diagonal shortcuts, A to E and B to F, in the accent "
        "color. The stretch-2 spanner drops both diagonals, shown dashed: each one is "
        "replaced by a two-step path, so no distance more than doubles.")
def transform_with_spanner(s: Scene) -> None:
    left, right = transform(s, "g", "g.transform_with_spanner(stretch=2)")
    lp, rp = _both(GRID, left, right)
    ladder = [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F"), ("A", "D"), ("B", "E"), ("C", "F")]
    diag = [("A", "E"), ("B", "F")]
    s.graph(lp, ladder + [(a, b, {"style": "accent"}) for a, b in diag], prefix="l.")
    s.graph(rp, ladder + [(a, b, {"style": "removed"}) for a, b in diag], prefix="r.")
