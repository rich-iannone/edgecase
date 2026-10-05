"""Transforms: an input graph → the new graph the method derives from it.

Template: `transform`. Left: the input `g`; right: the returned graph, which
is always a new object (`g` is untouched). Wherever the derived graph keeps
the input's nodes, both panels share one layout so only the change moves.
Elements the method creates are `accent` on the right; elements it drops are
`accent` on the left and `removed` (dashed) on the right, as in mutations.
For extractions (`subgraph`, `edge_subgraph`) the selection you pass is
`soft` on the left and the extracted part is `accent` on the right.
`ego_graph` is an extraction too: the ego is `soft` in both panels and, since
the result renumbers node ids, id pills show the old ids (ink) on the left
and the new ids (accent) on the right. When the output is built from
different things (line graph, quotient), group colors
tie each output node to the input element it came from. condensation
shades each strongly connected component with a hull in its group color
(on the communities family's component graph) and draws the condensed DAG
with one node per component in the same color, its returned label beneath.
eulerize uses the house from paths.py with predicates-style degree badges:
the odd nodes are `accent` on the left, and each duplicated edge is an `accent`
arc bowed beside the original on the right, where every badge is even.
"""

from core import SIZE_NOTE, Scene, figure, place, transform

from figures.paths import HOUSE, HOUSE_EDGES
from figures.predicates import _degrees

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


# The ego_graph docstring example: triangle B, C, D, with A off B and the
# chain E–F off C. Node ids are pills, since the result renumbers them.
EGO = {"A": (0, 0), "B": (1, 0), "D": (1, 1.2), "C": (2, 0.6), "E": (3, 0.6), "F": (3, -0.4)}
EGO_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("B", "D"), ("C", "E"), ("E", "F")]
EGO_ID = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
EGO_NEW_ID = {"B": 1, "C": 2, "D": 3, "E": 4}  # ids in the returned graph
EGO_PILL = {"A": "n", "B": "n", "C": "s", "D": "s", "E": "s", "F": "e"}


@figure("ego_graph", "transforms", "Ego graph",
        "A six-node graph with node ids in pills; the ego C is outlined. Its radius-1 ego "
        "graph keeps C, its neighbors B, D, and E, and every edge among them, including B-D, "
        "which does not touch C, all in the accent color; A and F are dropped and shown "
        "dashed. The new graph keeps the labels but renumbers the ids 1 to 4.")
def ego_graph(s: Scene) -> None:
    left, right = transform(s, "g", 'g.ego_graph("C")')
    lp, rp = _both(EGO, left, right)
    keep = set(EGO_NEW_ID)
    s.graph(lp, EGO_EDGES, styles={"C": "soft"}, prefix="l.")
    s.graph(rp, [(a, b, {"style": "accent" if {a, b} <= keep else "removed"})
                 for a, b in EGO_EDGES],
            styles={k: "soft" if k == "C" else "accent" if k in keep else "removed"
                    for k in EGO}, prefix="r.")
    for k, i in EGO_ID.items():
        s.badge("l." + k, str(i), where=EGO_PILL[k], color="ink")
    for k, i in EGO_NEW_ID.items():
        s.badge("r." + k, str(i), where=EGO_PILL[k])


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


# Bipartite input: the partition projected onto (which=0) on top, the other
# side underneath; each bottom node gets its own color so every projected
# edge can be traced to the shared neighbor that created it.
BIPARTITE = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0), "E": (1, 1.2), "F": (2, 1.2)}
BIPARTITE_EDGES = [("A", "E"), ("B", "E"), ("C", "E"), ("B", "F"), ("C", "F"), ("D", "F")]


@figure("bipartite_projection", "transforms", "Bipartite projection",
        "A bipartite graph with A, B, C, D on top and E, F below; E, in the accent color, is "
        "joined to A, B, C and F, in teal, to B, C, D. Projecting onto the top side joins two "
        "nodes when they share a neighbor below, weighted by how many they share: A-B and A-C "
        "share only E (accent), B-D and C-D share only F (teal), and B-C shares both, so it "
        "is a plain edge of weight 2.")
def bipartite_projection(s: Scene) -> None:
    left, right = transform(s, "g", "g.bipartite_projection(which=0)")
    side = {"E": "accent", "F": "g2"}
    s.graph(place(BIPARTITE, left, unit=UNIT),
            [(a, b, {"style": side[b]}) for a, b in BIPARTITE_EDGES], styles=side, prefix="l.")
    # edgecase: weights A-B 1, A-C 1, B-C 2, B-D 1, C-D 1; one-witness edges take
    # their witness's color, the two-witness edge stays ink.
    rp = place({"A": (0, 0.6), "B": (1, 0), "C": (1, 1.2), "D": (2, 0.6)}, right, unit=UNIT)
    s.graph(rp, [("A", "B", {"style": "accent", "weight": 1}),
                 ("A", "C", {"style": "accent", "weight": 1}),
                 ("B", "C", {"weight": 2}),
                 ("B", "D", {"style": "g2", "weight": 1}),
                 ("C", "D", {"style": "g2", "weight": 1})], prefix="r.")


# g.degree_preserving_rewire(n_swaps=1, seed=3) on the GRID ladder: the swap
# takes A-B and D-E and rewires them as A-E and B-D.
REWIRE_OUT = [("A", "B"), ("D", "E")]
REWIRE_IN = [("A", "E"), ("B", "D")]


@figure("degree_preserving_rewire", "transforms", "Degree-preserving rewire",
        "A six-node ladder with each node's degree in a pill: 2 for A, C, D, F and 3 for B "
        "and E. One swap takes the accent edges A-B and D-E and rewires them as the accent "
        "diagonals A-E and B-D, with the old edges shown dashed; every degree pill on the "
        "right matches the left.")
def degree_preserving_rewire(s: Scene) -> None:
    left, right = transform(s, "g", "g.degree_preserving_rewire(n_swaps=1)")
    lp, rp = _both(GRID, left, right)
    out = {frozenset(e) for e in REWIRE_OUT}
    s.graph(lp, [(a, b, {"style": "accent"}) if frozenset((a, b)) in out else (a, b)
                 for a, b in GRID_EDGES], prefix="l.")
    s.graph(rp, [(a, b, {"style": "removed"}) if frozenset((a, b)) in out else (a, b)
                 for a, b in GRID_EDGES] + [(a, b, {"style": "accent"}) for a, b in REWIRE_IN],
            prefix="r.")
    degree = {k: sum(k in e for e in GRID_EDGES) for k in GRID}
    for p in ("l.", "r."):
        for k in GRID:  # same pill spot on both sides: above the top row, below the bottom
            s.badge(p + k, str(degree[k]), where="n" if k in "ABC" else "s")


# Condensation reuses the directed component graph from communities.py: a cycle
# A -> B -> C -> A, a one-way edge C -> D, a reciprocal pair D <-> E, and F -> G.
COMP = {"A": (0, 0), "B": (0, 1.3), "C": (1, 0.65), "D": (2, 0.65), "E": (3, 0.65),
        "F": (3.9, 0.05), "G": (3.9, 1.25)}
COMP_EDGES = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "D"),
              ("F", "G")]
SCC_COLOR = {**dict.fromkeys("ABC", "accent"), **dict.fromkeys("DE", "g2"), "F": "g3",
             "G": "g4"}


@figure("condensation", "transforms", "Condensation",
        "Left: a directed graph whose strongly connected components are shaded: the cycle "
        "A, B, C (accent color), the two-way pair D, E (teal), and the single nodes F "
        "(violet) and G (gold). Right: the condensation has one node per component, labeled "
        "{A,B,C}, {D,E}, F and G, joined by the edges that ran between components, "
        "{A,B,C} to {D,E} and F to G; the result is acyclic.")
def condensation(s: Scene) -> None:
    left, right = transform(s, "g", "g.condensation()")
    bend = {("D", "E"): 7, ("E", "D"): 7}
    s.graph(place(COMP, left, unit=UNIT),
            [(a, b, {"style": SCC_COLOR[a] if SCC_COLOR[a] == SCC_COLOR[b] else "default",
                     "bend": bend.get((a, b), 0)}) for a, b in COMP_EDGES],
            directed=True, styles=SCC_COLOR, prefix="l.")
    for scc in ("ABC", "DE", "F", "G"):
        s.hull(["l." + k for k in scc], SCC_COLOR[scc[0]])
    # One node per component, in topological (= new id) order; member letters
    # inside, the returned label underneath.
    rp = place({"ABC": (0, 0), "DE": (1.1, 0), "F": (2.3, 0), "G": (3.4, 0)}, right,
               unit=UNIT * 0.9)
    s.graph(rp, [("ABC", "DE"), ("F", "G")], directed=True,
            styles={k: SCC_COLOR[k[0]] for k in rp}, prefix="r.")
    for k, lab in zip(rp, ("{A,B,C}", "{D,E}", "F", "G")):
        x, y = rp[k]
        s.text(x, y + 30, lab, size=SIZE_NOTE, opacity=0.7)


@figure("eulerize", "transforms", "Eulerize",
        "Left: a square with a roof; badges give each node's degree, and only the top corners "
        "B and C are odd (accent color). Right: eulerize() returns a new graph with the edge "
        "B-C duplicated (accent arc), the cheapest fix, so every degree is even and the graph "
        "has an Eulerian circuit.")
def eulerize(s: Scene) -> None:
    left, right = transform(s, "g", "g.eulerize()")
    lp, rp = _both(HOUSE, left, right)
    s.graph(lp, HOUSE_EDGES, prefix="l.")
    _degrees(s, "l.", HOUSE_EDGES, where={"A": "e"})  # beside the apex, not between its edges
    # the duplicate bows down into the square, away from the roof
    s.graph(rp, HOUSE_EDGES + [("B", "C", {"style": "accent", "bend": 10})], prefix="r.")
    _degrees(s, "r.", HOUSE_EDGES + [("B", "C")], where={"A": "e"})
