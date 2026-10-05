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
The small-world references sit beside degree_preserving_rewire and draw
graph_metrics' `SW` ring (four triangles) in both directions:
random_reference rewires it (seed 8) into a graph with almost no triangles,
and lattice_reference (seed 1) rewires that random graph back into SW. The
edges a reference drops are accent on the left; the edges it creates are
accent on the right (dropped edges are not repeated there, to keep the ring
legible). A small card under each graph gives its average clustering and
degree sequence, which never changes. Both assert their results.
"""

from core import R_SMALL, SIZE_NOTE, STROKE, Box, Scene, figure, place, ring, transform

from figures.graph_metrics import SW, SW_EDGES, sw_graph
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


# --- later additions: distance closure and tree decomposition ---------------
#
# Both figures rebuild their docstring example in edgecase and assert the
# result, so a library change breaks the build instead of the picture.
# metric_closure reuses CHAIN (and the closure arcs of transitive_closure),
# undirected and with unit weights, so the two closures look alike.


def _ec_edges(h) -> dict[frozenset, float]:
    """{frozenset(labels): weight} for every edge of an edgecase graph."""
    out = {}
    for eid in h.get_edge_ids(limit=None):
        e = h.get_edge(eid)
        out[frozenset((h.get_label_from_id(e.source), h.get_label_from_id(e.target)))] = e.weight
    return out


@figure("metric_closure", "transforms", "Metric closure",
        "Left: a path A, B, C, D whose three edges each have weight 1. Right: the metric "
        "closure joins every pair of nodes, weighted by the shortest-path distance between "
        "them. It keeps the three unit edges and adds the accent shortcuts A–C with weight "
        "2, B–D with weight 2, and A–D with weight 3.")
def metric_closure(s: Scene) -> None:
    import edgecase as ec

    left, right = transform(s, "g", "g.metric_closure()")
    g = ec.Graph(weighted=True).add_nodes(list(CHAIN))
    g.add_edges([(a, b, 1.0) for a, b in CHAIN_EDGES])
    dist = _ec_edges(g.metric_closure())
    new = {frozenset(("A", "C")): 2, frozenset(("B", "D")): 2, frozenset(("A", "D")): 3}
    assert dist == {frozenset(e): 1.0 for e in CHAIN_EDGES} | new
    lp, rp = _both(CHAIN, left, right)
    s.graph(lp, [(a, b, {"weight": 1}) for a, b in CHAIN_EDGES], prefix="l.")
    s.graph(rp, [(a, b, {"weight": 1}) for a, b in CHAIN_EDGES]
            + [(a, b, {"style": "accent", "bend": k, "weight": int(dist[frozenset((a, b))])})
               for a, b, k in CHAIN_ARCS], prefix="r.")


# tree_decomposition docstring: a 5-cycle has treewidth 2. The dashed chords
# B–E and C–E cut it into the three triangles that become the bags.
CYCLE5 = ring(list("ABCDE"), radius=1.0)
CYCLE5_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")]
BAG_UNIT = 28  # one scale for the three mini triangles inside the bags


@figure("tree_decomposition", "transforms", "Tree decomposition",
        "Left: a five-node cycle A to E; the dashed accent chords B–E and C–E split it into "
        "three triangles. Right: the returned tree of bags, each bag a shaded box holding "
        "one triangle: {A,B,E}, {B,C,E}, and {C,D,E}, joined in a chain; neighboring bags "
        "share two nodes, and every cycle edge lies inside a bag. The largest bag has 3 "
        "nodes, so the card gives the width 3 − 1 = 2.")
def tree_decomposition(s: Scene) -> None:
    import edgecase as ec

    left, right = transform(s, "g", "g.tree_decomposition()")
    g = ec.Graph().add_nodes(list(CYCLE5))
    g.add_edges(CYCLE5_EDGES)
    width, tree = g.tree_decomposition()
    labels = {tree.get_label_from_id(n) for n in tree.get_node_ids(limit=None)}
    bags = ["ABE", "BCE", "CDE"]  # drawn left to right along the chain
    assert width == 2 and labels == {"{" + ",".join(b) + "}" for b in bags}
    assert set(_ec_edges(tree)) == {frozenset(("{A,B,E}", "{B,C,E}")),
                                    frozenset(("{B,C,E}", "{C,D,E}"))}
    chords = [("B", "E"), ("C", "E")]
    s.graph(place(CYCLE5, left, unit=UNIT), CYCLE5_EDGES
            + [(a, b, {"style": "ghost"}) for a, b in chords], prefix="l.")
    # one box per bag, its triangle drawn with the cycle's own geometry; all
    # boxes share one size so the chain reads as a row of equal bags
    gb = Box(right.x0, right.y0, right.x1, right.y1 - 56)
    slot = gb.w / len(bags)
    placed = [place({k: CYCLE5[k] for k in bag},
                    Box(gb.x0 + i * slot, gb.y0, gb.x0 + (i + 1) * slot, gb.y1),
                    unit=BAG_UNIT, r=R_SMALL) for i, bag in enumerate(bags)]
    pad = R_SMALL + 6
    bw = max(max(x for x, _ in p.values()) - min(x for x, _ in p.values()) for p in placed)
    bh = max(max(y for _, y in p.values()) - min(y for _, y in p.values()) for p in placed)
    bw, bh = bw + 2 * pad, bh + 2 * pad
    rects = []
    for i, (bag, pos) in enumerate(zip(bags, placed)):
        cx = gb.x0 + (i + 0.5) * slot
        rects.append(s.rect(cx - bw / 2, gb.cy - bh / 2, bw, bh, rx=10, stroke="accent",
                            stroke_opacity=1, fill="tint", fill_color="accent",
                            fill_opacity=0.55, role="region"))
        inside = [(a, b) for a, b in CYCLE5_EDGES if a in bag and b in bag]
        s.graph(pos, inside + [(a, b, {"style": "ghost"}) for a, b in chords
                               if a in bag and b in bag], r=R_SMALL, prefix=f"b{i}.")
    for r0, r1 in zip(rects, rects[1:]):
        s.line(r0.x + r0.w, gb.cy, r1.x, gb.cy, width=STROKE, role="tree_edge")
    s.card_centered(Box(right.x0, right.y1 - 56, right.x1, right.y1), [
        (f"width  3 − 1 = {width}", "accent")], title="largest bag − 1")


# --- later additions: small-world references ----------------------------------


def _edge_set(g) -> set:
    return {frozenset((g.get_label_from_id(a), g.get_label_from_id(b)))
            for _, a, b in g.get_edges_as_tuples()}


def _degrees_of(g) -> str:
    return " ".join(str(g.get_degree(g.get_id_from_label(k))) for k in SW)


# random_reference(seed=8) of SW, as edgecase returns it; lattice_reference(seed=1)
# of this graph gives SW back.
SW_RANDOM = [("A", "B"), ("A", "C"), ("A", "E"), ("A", "H"), ("B", "E"), ("C", "D"), ("C", "E"),
             ("C", "F"), ("D", "G"), ("E", "G"), ("F", "G"), ("G", "H")]


def _reference(s: Scene, call: str, before, after) -> None:
    left, right = transform(s, "g", call)
    old, new = _edge_set(before), _edge_set(after)
    for box, p, g, edges, hot in ((left, "l.", before, old, old - new),
                                  (right, "r.", after, new, new - old)):
        gbox = Box(box.x0, box.y0, box.x0 + 180, box.y1)  # the card takes the rest
        pairs = sorted(tuple(sorted(e)) for e in edges)
        s.graph(place(SW, gbox, unit=UNIT, r=R_SMALL),
                [(a, b, {"style": "accent"}) if frozenset((a, b)) in hot else (a, b)
                 for a, b in pairs], r=R_SMALL, prefix=p)
        c = f"clustering {g.get_avg_clustering_coefficient():.3f}"
        lines = [(c, "accent"), "degrees kept"] if p == "r." else [c, "degree 4 or 2"]
        s.card(gbox.x1, box.cy - 28, box.x1 - gbox.x1, lines)


@figure("random_reference", "transforms", "Random reference graph",
        "Left: an eight-node ring with four chords that close four triangles, average "
        "clustering 0.667. Right: random_reference(seed=8) rewires it with degree-preserving "
        "swaps: the four accent edges on the left are replaced by the four accent edges on the "
        "right, every node keeps its degree (4 or 2), and the triangles mostly disappear, "
        "leaving an average clustering of 0.229.")
def random_reference(s: Scene) -> None:
    g = sw_graph()
    r = g.random_reference(seed=8)
    assert _edge_set(r) == {frozenset(e) for e in SW_RANDOM}
    assert _degrees_of(r) == _degrees_of(g) and r.is_connected()
    assert round(g.get_avg_clustering_coefficient(), 3) == 0.667
    assert round(r.get_avg_clustering_coefficient(), 3) == 0.229
    _reference(s, "g.random_reference(seed=8)", g, r)


@figure("lattice_reference", "transforms", "Lattice reference graph",
        "Left: a randomly wired eight-node graph with average clustering 0.229. Right: "
        "lattice_reference(seed=1) keeps every degree but only accepts swaps that shorten "
        "edges around the ring A to H: the four long accent edges on the left become the four "
        "short accent edges on the right, closing four triangles around the ring, and the "
        "average clustering rises to 0.667.")
def lattice_reference(s: Scene) -> None:
    g = sw_graph(SW_RANDOM)
    lat = g.lattice_reference(seed=1)
    assert _edge_set(lat) == {frozenset(e) for e in SW_EDGES}
    assert _degrees_of(lat) == _degrees_of(g)
    assert round(lat.get_avg_clustering_coefficient(), 3) == 0.667
    _reference(s, "g.lattice_reference(seed=1)", g, lat)
