"""Predicates: methods that answer True or False about the graph (or a candidate set in it).

Template: `contrast`. Left: a small graph for which the answer is True.
Right: a near-identical graph for which it is False, with the culprit (the
element that breaks the property) in `accent`. Keeping the two graphs nearly
the same makes the deciding difference obvious. An element that is missing
(the edge that would make the property hold) is drawn `ghost`.

Sibling predicates share a base layout so they can be compared directly:
the six-node tree (is_tree / is_forest / is_arborescence), the directed
triangle strip (is_dag / has_cycles), two triangles joined at C-D
(is_connected / is_strongly_connected / is_weakly_connected / has_bridges /
is_biconnected), the bowtie (is_eulerian / is_semieulerian), the house of
Santa Claus from paths.py (has_eulerian_path: dropping its roof leaves four odd
corners, so the roof is `removed` on the right), the hexagon (is_regular /
is_k_regular), and the 2 x 3 ladder (is_bipartite and every set-checking
method). Set checkers draw the candidate set `soft` on both sides and name it
in a note above the graph; the violation is `accent`. has_path keeps the
query endpoints `soft` on both sides: the path is `accent` on the left, the
reversed edge that cuts it off is `accent` on the right. The isomorphism
predicates split each panel into two small graphs (g and h, or pattern p and
host g), named in notes above them; could_be_isomorphic shows the compared
per-node property as badges. is_chordal, is_semiconnected and
is_attracting_component rebuild their docstring examples (square vs. square
with a diagonal, chain vs. fork, cycle vs. cycle with one edge reversed) and
assert the answers. Graph scale is the default `UNIT` throughout.
"""


from core import R_SMALL, SIZE_NOTE, Box, Scene, contrast, figure, place, ring

from figures.paths import HOUSE, SANTA_EDGES


def _hi(edges, accent, style="accent"):
    """Edge list with the pairs in `accent` drawn in `style`."""
    return [(a, b, {"style": style}) if (a, b) in accent else (a, b) for a, b in edges]


def _note(s: Scene, box: Box, text: str) -> Box:
    """Small note naming the candidate (set, path, coloring) above a panel's graph.

    Returns the box left for the graph below it.
    """
    s.text(box.cx, box.y0 + 8, text, size=SIZE_NOTE, opacity=0.6)
    return Box(box.x0, box.y0 + 22, box.x1, box.y1)


# --- trees ---------------------------------------------------------------------

TREE = {"A": (1, 0), "B": (0.25, 1), "C": (1.75, 1), "D": (-0.25, 2), "E": (0.75, 2),
        "F": (1.75, 2)}
TREE_EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]


@figure("is_tree", "predicates", "Is tree",
        "Left: six nodes connected with five edges and no cycle, so is_tree() returns True. "
        "Right: the same graph plus one extra edge between E and F, which closes the cycle "
        "A, B, E, F, C (shown in the accent color), so is_tree() returns False.")
def is_tree(s: Scene) -> None:
    left, right = contrast(s, "g.is_tree()", "g.is_tree()")
    s.graph(place(TREE, left), TREE_EDGES, prefix="l.")
    cyc = {("A", "B"), ("A", "C"), ("B", "E"), ("C", "F")}
    s.graph(place(TREE, right), _hi(TREE_EDGES, cyc) + [("E", "F", {"style": "accent"})],
            prefix="r.")


# The tree without its A-C edge: two trees, with the C-F tree nudged right so the
# gap between the components reads at a glance.
FOREST = {**TREE, "C": (2.25, 1), "F": (2.25, 2)}
FOREST_EDGES = [("A", "B"), ("B", "D"), ("B", "E"), ("C", "F")]


@figure("is_forest", "predicates", "Is forest",
        "Left: two separate trees, one on A, B, D, E and one on C and F; neither has a cycle, "
        "so is_forest() returns True. Right: the same two trees plus an edge between D and E, "
        "which closes the triangle B, D, E (shown in the accent color), so is_forest() "
        "returns False.")
def is_forest(s: Scene) -> None:
    left, right = contrast(s, "g.is_forest()", "g.is_forest()")
    s.graph(place(FOREST, left), FOREST_EDGES, prefix="l.")
    cyc = {("B", "D"), ("B", "E")}
    s.graph(place(FOREST, right), _hi(FOREST_EDGES, cyc) + [("D", "E", {"style": "accent"})],
            prefix="r.")


@figure("is_arborescence", "predicates", "Is arborescence",
        "Left: the six-node tree with every edge pointing away from the root A, so each other "
        "node has exactly one incoming edge and is_arborescence() returns True. Right: the "
        "same tree with the edge between C and F reversed (shown in the accent color); F now "
        "has no incoming edge, giving two roots, so is_arborescence() returns False.")
def is_arborescence(s: Scene) -> None:
    left, right = contrast(s, "g.is_arborescence()", "g.is_arborescence()")
    s.graph(place(TREE, left), TREE_EDGES, directed=True, prefix="l.")
    s.graph(place(TREE, right), TREE_EDGES[:-1] + [("F", "C", {"style": "accent"})],
            directed=True, prefix="r.")


# Two small arborescences side by side: A -> B, A -> C and D -> E. Adding D -> C
# gives C a second parent. Both answers are computed with edgecase.
BRANCH = {"A": (0.5, 0), "B": (0, 1.2), "C": (1, 1.2), "D": (2.2, 0), "E": (2.2, 1.2)}
BRANCH_ARCS = [("A", "B"), ("A", "C"), ("D", "E")]


def _is_branching(arcs) -> bool:
    import edgecase as ec

    g = ec.Graph(directed=True).add_nodes(list(BRANCH))
    g.add_edges(arcs)
    return g.is_branching()


@figure("is_branching", "predicates", "Is branching",
        "Left: a directed forest of two trees, A pointing to B and C, and D pointing to E; "
        "every node has at most one incoming edge, so is_branching() returns True. Right: "
        "the same graph plus an edge from D to C (shown in the accent color); C now has two "
        "parents, A and D, so is_branching() returns False.")
def is_branching(s: Scene) -> None:
    extra = ("D", "C")
    left, right = contrast(s, "g.is_branching()", "g.is_branching()",
                           str(_is_branching(BRANCH_ARCS)),
                           str(_is_branching(BRANCH_ARCS + [extra])))
    s.graph(place(BRANCH, left), BRANCH_ARCS, directed=True, prefix="l.")
    s.graph(place(BRANCH, right), BRANCH_ARCS + [(*extra, {"style": "accent"})],
            directed=True, styles={"C": "accent"}, prefix="r.")


# --- directed acyclicity -----------------------------------------------------------

# A strip of two triangles. Flipping B->C into C->B creates the cycle B->E->C->B.
STRIP = {"A": (0, 0), "B": (1.5, 0), "C": (3, 0), "D": (0.75, 1.4), "E": (2.25, 1.4)}
DAG_EDGES = [("A", "B"), ("A", "D"), ("D", "E"), ("B", "E"), ("E", "C")]


@figure("is_dag", "predicates", "Is DAG",
        "Left: a directed graph on five nodes where every edge leads rightward or downward, so "
        "no path returns to its start and is_dag() returns True. Right: the same graph with "
        "the edge between B and C reversed, which creates the cycle B, E, C, B (shown in the "
        "accent color), so is_dag() returns False.")
def is_dag(s: Scene) -> None:
    left, right = contrast(s, "g.is_dag()", "g.is_dag()")
    s.graph(place(STRIP, left), DAG_EDGES + [("B", "C")], directed=True, prefix="l.")
    s.graph(place(STRIP, right),
            _hi(DAG_EDGES, {("B", "E"), ("E", "C")}) + [("C", "B", {"style": "accent"})],
            directed=True, prefix="r.")


# Drawn directed on purpose: on undirected graphs has_cycles() currently reports
# True even for trees, so the figure sticks to the case the method gets right.
@figure("has_cycles", "predicates", "Has cycles",
        "Left: a directed graph on five nodes containing the cycle B, E, C, B (shown in the "
        "accent color), so has_cycles() returns True. Right: the same graph with the edge "
        "between B and C pointing the other way (shown in the accent color); no path returns "
        "to its start any more, so has_cycles() returns False.")
def has_cycles(s: Scene) -> None:
    left, right = contrast(s, "g.has_cycles()", "g.has_cycles()")
    s.graph(place(STRIP, left),
            _hi(DAG_EDGES, {("B", "E"), ("E", "C")}) + [("C", "B", {"style": "accent"})],
            directed=True, prefix="l.")
    s.graph(place(STRIP, right), DAG_EDGES + [("B", "C", {"style": "accent"})],
            directed=True, prefix="r.")


# --- connectivity ----------------------------------------------------------------

# Two triangles joined by the single edge C-D.
TWIN = {"A": (0, 0), "B": (0, 1.6), "C": (1, 0.8), "D": (2.4, 0.8), "E": (3.4, 0),
        "F": (3.4, 1.6)}
TWIN_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("D", "E"), ("D", "F"), ("E", "F")]
# The same triangles as directed cycles A->C->B->A and D->E->F->D.
TWIN_ARCS = [("A", "C"), ("C", "B"), ("B", "A"), ("D", "E"), ("E", "F"), ("F", "D")]


@figure("is_connected", "predicates", "Is connected",
        "Left: two triangles, A B C and D E F, joined by the edge between C and D, so every "
        "node can reach every other and is_connected() returns True. Right: the same graph "
        "without the C-D edge (drawn dashed in the accent color), which leaves two separate "
        "pieces, so is_connected() returns False.")
def is_connected(s: Scene) -> None:
    left, right = contrast(s, "g.is_connected()", "g.is_connected()")
    s.graph(place(TWIN, left), TWIN_EDGES + [("C", "D")], prefix="l.")
    s.graph(place(TWIN, right), TWIN_EDGES + [("C", "D", {"style": "ghost"})], prefix="r.")


@figure("is_strongly_connected", "predicates", "Is strongly connected",
        "Left: two directed triangles joined by edges in both directions between C and D, so "
        "every node can reach every other along the arrows and is_strongly_connected() returns "
        "True. Right: the same graph without the edge from D back to C (drawn dashed in the "
        "accent color); nothing in D, E, F can reach A, B, C, so is_strongly_connected() "
        "returns False.")
def is_strongly_connected(s: Scene) -> None:
    left, right = contrast(s, "g.is_strongly_connected()", "g.is_strongly_connected()")
    s.graph(place(TWIN, left), TWIN_ARCS + [("C", "D", {"bend": 9}), ("D", "C", {"bend": 9})],
            directed=True, prefix="l.")
    s.graph(place(TWIN, right),
            TWIN_ARCS + [("C", "D", {"bend": 9}), ("D", "C", {"bend": 9, "style": "ghost"})],
            directed=True, prefix="r.")


@figure("is_weakly_connected", "predicates", "Is weakly connected",
        "Left: two directed triangles joined by a single one-way edge from C to D; ignoring "
        "the arrows, everything is linked, so is_weakly_connected() returns True. Right: the "
        "same graph without the C-D edge (drawn dashed in the accent color), which leaves two "
        "separate pieces even with directions ignored, so is_weakly_connected() returns False.")
def is_weakly_connected(s: Scene) -> None:
    left, right = contrast(s, "g.is_weakly_connected()", "g.is_weakly_connected()")
    s.graph(place(TWIN, left), TWIN_ARCS + [("C", "D")], directed=True, prefix="l.")
    s.graph(place(TWIN, right), TWIN_ARCS + [("C", "D", {"style": "ghost"})],
            directed=True, prefix="r.")


# Later additions on the directed connectivity theme; each rebuilds its
# docstring example in edgecase and asserts the answer.


def _ec_digraph(nodes, edges):
    import edgecase as ec

    g = ec.Graph(directed=True).add_nodes(nodes)
    g.add_edges(edges)
    return g


SEMI = {"A": (0, 0.8), "B": (1.6, 0), "C": (1.6, 1.6)}


@figure("is_semiconnected", "predicates", "Is semiconnected",
        "Left: a chain A to B to C; for every pair one node reaches the other, so "
        "is_semiconnected() returns True. Right: a fork, A to B and A to C; B and C (shown "
        "in the accent color) cannot reach each other in either direction, so "
        "is_semiconnected() returns False.")
def is_semiconnected(s: Scene) -> None:
    chain, fork = [("A", "B"), ("B", "C")], [("A", "B"), ("A", "C")]
    assert _ec_digraph(list(SEMI), chain).is_semiconnected() is True
    assert _ec_digraph(list(SEMI), fork).is_semiconnected() is False
    left, right = contrast(s, "g.is_semiconnected()", "g.is_semiconnected()")
    s.graph(place(SEMI, left), chain, directed=True, prefix="l.")
    s.graph(place(SEMI, right), fork, directed=True,
            styles={"B": "accent", "C": "accent"}, prefix="r.")


ATTR = {"A": (0.8, 0), "B": (0, 1.4), "C": (1.6, 1.4)}


@figure("is_attracting_component", "predicates", "Is attracting component",
        "Left: a directed cycle A to B to C to A; a walk can go anywhere and never leave, so "
        "the whole graph is one attracting component and is_attracting_component() returns "
        "True. Right: the same triangle with the edge between C and A reversed to point from "
        "A to C (shown in the accent color); walks now end at C, the only attracting "
        "component, which leaves out A and B, so is_attracting_component() returns False.")
def is_attracting_component(s: Scene) -> None:
    cycle, dag = [("A", "B"), ("B", "C"), ("C", "A")], [("A", "B"), ("B", "C"), ("A", "C")]
    assert _ec_digraph(list(ATTR), cycle).is_attracting_component() is True
    g = _ec_digraph(list(ATTR), dag)
    assert g.is_attracting_component() is False
    assert [[g.get_label_from_id(n) for n in c] for c in g.attracting_components()] == [["C"]]
    left, right = contrast(s, "g.is_attracting_component()", "g.is_attracting_component()")
    s.graph(place(ATTR, left), cycle, directed=True, prefix="l.")
    s.graph(place(ATTR, right), _hi(dag, {("A", "C")}), directed=True, prefix="r.")



@figure("has_bridges", "predicates", "Has bridges",
        "Left: two triangles joined only by the edge between C and D, a bridge (shown in the "
        "accent color) whose removal splits the graph, so has_bridges() returns True. Right: "
        "the same graph plus an edge between B and F (shown in the accent color), which gives "
        "every edge a detour, so has_bridges() returns False.")
def has_bridges(s: Scene) -> None:
    left, right = contrast(s, "g.has_bridges()", "g.has_bridges()")
    s.graph(place(TWIN, left), TWIN_EDGES + [("C", "D", {"style": "accent"})], prefix="l.")
    s.graph(place(TWIN, right), TWIN_EDGES + [("C", "D"), ("B", "F", {"style": "accent"})],
            prefix="r.")


@figure("is_biconnected", "predicates", "Is biconnected",
        "Left: two triangles joined by the edges C-D and B-F, so every pair of nodes has two "
        "routes with no other node in common, and is_biconnected() returns True. Right: the same "
        "triangles joined by C-D and B-D instead; every route between the two sides now "
        "passes through D (shown in the accent color), so is_biconnected() returns False.")
def is_biconnected(s: Scene) -> None:
    left, right = contrast(s, "g.is_biconnected()", "g.is_biconnected()")
    # The True side is has_bridges' False side; the False side has no bridge
    # (B-D and C-D lie on triangle B, C, D), only the articulation point D.
    s.graph(place(TWIN, left), TWIN_EDGES + [("C", "D"), ("B", "F")], prefix="l.")
    # B-D bows below C to stay clear of it
    s.graph(place(TWIN, right), TWIN_EDGES + [("C", "D"), ("B", "D", {"bend": 14})],
            styles={"D": "accent"}, prefix="r.")


# --- the 2 x 3 ladder (bipartite + candidate-set checks) --------------------------

LADDER = {"A": (0, 0), "B": (1.4, 0), "C": (2.8, 0), "D": (0, 1.4), "E": (1.4, 1.4),
          "F": (2.8, 1.4)}
LADDER_EDGES = [("A", "B"), ("B", "C"), ("A", "D"), ("B", "E"), ("C", "F"), ("D", "E"),
                ("E", "F")]


@figure("is_bipartite", "predicates", "Is bipartite",
        "Left: a two-by-three ladder whose nodes split into two colors, A, C, E and B, D, F, "
        "with every edge joining different colors, so is_bipartite() returns True. Right: the "
        "same ladder plus a diagonal between A and E, which closes the odd triangle A, B, E "
        "(shown in the accent color), so is_bipartite() returns False.")
def is_bipartite(s: Scene) -> None:
    left, right = contrast(s, "g.is_bipartite()", "g.is_bipartite()")
    side = {k: ("accent" if k in "ACE" else "g2") for k in LADDER}
    s.graph(place(LADDER, left), LADDER_EDGES, styles=side, prefix="l.")
    tri = {("A", "B"), ("B", "E")}
    s.graph(place(LADDER, right), _hi(LADDER_EDGES, tri) + [("A", "E", {"style": "accent"})],
            prefix="r.")


@figure("is_independent_set", "predicates", "Is independent set",
        "The same two-by-three ladder on both sides, with the candidate nodes outlined. Left: "
        "A, C, E share no edge, so is_independent_set() returns True. Right: A, C, F, where "
        "the edge between C and F (shown in the accent color) joins two members, so "
        "is_independent_set() returns False.")
def is_independent_set(s: Scene) -> None:
    left, right = contrast(s, "g.is_independent_set(nodes)", "g.is_independent_set(nodes)")
    for box, chosen, bad in ((left, "ACE", set()), (right, "ACF", {("C", "F")})):
        gb = _note(s, box, f"nodes = {', '.join(chosen)}")
        s.graph(place(LADDER, gb), _hi(LADDER_EDGES, bad), styles={k: "soft" for k in chosen},
                prefix=f"{box.x0}.")


@figure("is_dominating_set", "predicates", "Is dominating set",
        "The same two-by-three ladder on both sides, with the candidate nodes outlined. Left: "
        "every node is B, E, or next to one of them, so is_dominating_set() returns True. "
        "Right: with B and D, node F (shown in the accent color) has no neighbor in the set, "
        "so is_dominating_set() returns False.")
def is_dominating_set(s: Scene) -> None:
    left, right = contrast(s, "g.is_dominating_set(nodes)", "g.is_dominating_set(nodes)")
    for box, chosen, bad in ((left, "BE", ""), (right, "BD", "F")):
        gb = _note(s, box, f"nodes = {', '.join(chosen)}")
        styles = {k: "soft" for k in chosen} | {k: "accent" for k in bad}
        s.graph(place(LADDER, gb), LADDER_EDGES, styles=styles, prefix=f"{box.x0}.")


@figure("is_matching", "predicates", "Is matching",
        "The same two-by-three ladder on both sides, with the candidate edges in the accent "
        "color. Left: A-D, B-C, and E-F touch six different nodes, so is_matching() returns "
        "True. Right: B-C and C-F both end at C (filled in the accent color), so is_matching() "
        "returns False.")
def is_matching(s: Scene) -> None:
    left, right = contrast(s, "g.is_matching(edges)", "g.is_matching(edges)")
    for box, chosen, bad in ((left, [("A", "D"), ("B", "C"), ("E", "F")], ""),
                             (right, [("A", "D"), ("B", "C"), ("C", "F")], "C")):
        gb = _note(s, box, "edges = " + ", ".join(f"{a}-{b}" for a, b in chosen))
        s.graph(place(LADDER, gb), _hi(LADDER_EDGES, set(chosen), "soft"),
                styles={k: "accent" for k in bad}, prefix=f"{box.x0}.")


# --- matchings and covers (docstring examples) -------------------------------------
#
# These rebuild each docstring example in edgecase and assert both answers, so
# the pictures cannot drift from the library. Candidate edges are `soft`; the
# culprit (an edge that could still be added, an uncovered node or edge) is
# `accent`.

P4 = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0)}
P4_EDGES = [("A", "B"), ("B", "C"), ("C", "D")]
HEX = ring(list("ABCDEF"), start_deg=-120)
HEX_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "A")]
STAR3 = {"a": (0, 0), "b": (2, 0), "hub": (1, 0.55), "c": (1, 1.5)}
STAR3_EDGES = [("hub", "a"), ("hub", "b"), ("hub", "c")]


def _ec(nodes, edges):
    import edgecase as ec

    g = ec.Graph().add_nodes(list(nodes))
    g.add_edges(edges)
    return g


def _eids(g, pairs) -> list[int]:
    return [g.find_edge_ids(a, b)[0] for a, b in pairs]


def _edge_pair(s: Scene, box: Box, coords, edges, cases, culprit_edges=(), culprit_nodes=""):
    """One side of an edge-set predicate: note, graph, candidate edges `soft`."""
    gb = _note(s, box, "edges = " + ", ".join(f"{a}-{b}" for a, b in cases))
    specs = []
    for a, b in edges:
        if (a, b) in cases:
            specs.append((a, b, {"style": "soft"}))
        elif (a, b) in culprit_edges:
            specs.append((a, b, {"style": "accent"}))
        else:
            specs.append((a, b))
    s.graph(place(coords, gb), specs, styles={k: "accent" for k in culprit_nodes},
            prefix=f"{box.x0}.")


@figure("is_maximal_matching", "predicates", "Is maximal matching",
        "The path A-B-C-D on both sides, with the candidate edges drawn as the input. Left: "
        "the middle edge B-C alone; both other edges touch B or C, so nothing can be added "
        "and is_maximal_matching() returns True. Right: A-B alone; the edge C-D (shown in "
        "the accent color with its free endpoints C and D) could still be added, so "
        "is_maximal_matching() returns False.")
def is_maximal_matching(s: Scene) -> None:
    g = _ec(P4, P4_EDGES)
    assert g.is_maximal_matching(_eids(g, [("B", "C")]))
    assert not g.is_maximal_matching(_eids(g, [("A", "B")]))
    left, right = contrast(s, "g.is_maximal_matching(edges)", "g.is_maximal_matching(edges)")
    _edge_pair(s, left, P4, P4_EDGES, [("B", "C")])
    _edge_pair(s, right, P4, P4_EDGES, [("A", "B")], {("C", "D")}, "CD")


@figure("is_perfect_matching", "predicates", "Is perfect matching",
        "A six-node ring A to F on both sides, with the candidate edges drawn as the input. "
        "Left: the alternate edges A-B, C-D, and E-F pair up every node, so "
        "is_perfect_matching() returns True. Right: only A-B and C-D, which leave E and F "
        "(filled in the accent color) unmatched, so is_perfect_matching() returns False.")
def is_perfect_matching(s: Scene) -> None:
    alt = [("A", "B"), ("C", "D"), ("E", "F")]
    g = _ec(HEX, HEX_EDGES)
    assert g.is_perfect_matching(_eids(g, alt))
    assert not g.is_perfect_matching(_eids(g, alt[:2]))
    left, right = contrast(s, "g.is_perfect_matching(edges)", "g.is_perfect_matching(edges)")
    _edge_pair(s, left, HEX, HEX_EDGES, alt)
    _edge_pair(s, right, HEX, HEX_EDGES, alt[:2], culprit_nodes="EF")


@figure("is_vertex_cover", "predicates", "Is vertex cover",
        "A star with center hub and leaves a, b, c on both sides, with the candidate nodes "
        "outlined. Left: hub touches every edge, so is_vertex_cover() returns True. Right: "
        "the leaf a alone leaves the edges hub-b and hub-c (shown in the accent color) "
        "untouched, so is_vertex_cover() returns False.")
def is_vertex_cover(s: Scene) -> None:
    g = _ec(STAR3, STAR3_EDGES)
    assert g.is_vertex_cover(["hub"]) and not g.is_vertex_cover(["a"])
    left, right = contrast(s, "g.is_vertex_cover(nodes)", "g.is_vertex_cover(nodes)")
    for box, chosen in ((left, ["hub"]), (right, ["a"])):
        gb = _note(s, box, f"nodes = {', '.join(chosen)}")
        bad = {e for e in STAR3_EDGES if not set(e) & set(chosen)}
        s.graph(place(STAR3, gb), _hi(STAR3_EDGES, bad), styles={k: "soft" for k in chosen},
                prefix=f"{box.x0}.")


@figure("is_edge_cover", "predicates", "Is edge cover",
        "The path A-B-C-D on both sides, with the candidate edges drawn as the input. Left: "
        "the outer edges A-B and C-D touch all four nodes, so is_edge_cover() returns True. "
        "Right: the middle edge B-C alone misses A and D (filled in the accent color), so "
        "is_edge_cover() returns False.")
def is_edge_cover(s: Scene) -> None:
    g = _ec(P4, P4_EDGES)
    assert g.is_edge_cover(_eids(g, [("A", "B"), ("C", "D")]))
    assert not g.is_edge_cover(_eids(g, [("B", "C")]))
    left, right = contrast(s, "g.is_edge_cover(edges)", "g.is_edge_cover(edges)")
    _edge_pair(s, left, P4, P4_EDGES, [("A", "B"), ("C", "D")])
    _edge_pair(s, right, P4, P4_EDGES, [("B", "C")], culprit_nodes="AD")


@figure("is_valid_coloring", "predicates", "Is valid coloring",
        "The same two-by-three ladder on both sides, each node filled with its color and "
        "badged with the color number. Left: no edge joins two nodes of the same color, so "
        "is_valid_coloring() returns True. Right: F is recolored to 2, the same color as its "
        "neighbor C, and that edge (shown in the accent color) makes is_valid_coloring() "
        "return False.")
def is_valid_coloring(s: Scene) -> None:
    left, right = contrast(s, "g.is_valid_coloring(colors)", "g.is_valid_coloring(colors)")
    group = ("g2", "g3", "g4")
    for box, colors, bad in ((left, [0, 1, 2, 2, 0, 1], set()),
                             (right, [0, 1, 2, 2, 0, 2], {("C", "F")})):
        p = f"{box.x0}."
        col = dict(zip(LADDER, colors))
        s.graph(place(LADDER, box), _hi(LADDER_EDGES, bad),
                styles={k: group[c] for k, c in col.items()}, prefix=p)
        for k, c in col.items():
            s.badge(p + k, str(c), color=group[c])


@figure("is_simple_path", "predicates", "Is simple path",
        "The same two-by-three ladder on both sides, with the candidate path's nodes outlined "
        "and numbered in order. Left: A, B, C, F follows existing edges without repeating a "
        "node, so is_simple_path() returns True. Right: A, B, C, E needs a C-E edge that does "
        "not exist (drawn dashed in the accent color), so is_simple_path() returns False.")
def is_simple_path(s: Scene) -> None:
    left, right = contrast(s, "g.is_simple_path(path)", "g.is_simple_path(path)")
    for box, path in ((left, "ABCF"), (right, "ABCE")):
        p = f"{box.x0}."
        gb = _note(s, box, f"path = {', '.join(path)}")
        hops = set(zip(path, path[1:]))
        edges = _hi(LADDER_EDGES, hops, "soft")
        edges += [(a, b, {"style": "ghost"}) for a, b in hops
                  if (a, b) not in LADDER_EDGES and (b, a) not in LADDER_EDGES]
        s.graph(place(LADDER, gb), edges, styles={k: "soft" for k in path}, prefix=p)
        for i, k in enumerate(path, 1):
            s.badge(p + k, str(i), filled=True)


# --- degree conditions -----------------------------------------------------------

# Two triangles sharing node C. Every degree is even.
BOWTIE = {"A": (0, 0), "B": (0, 1.8), "C": (1.2, 0.9), "D": (2.4, 0), "E": (2.4, 1.8)}
BOWTIE_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "E")]


def _degrees(s: Scene, prefix: str, edges, odd_style: str = "accent",
             where: dict | None = None) -> None:
    """Degree badge on every node of a panel; odd-degree nodes get `odd_style`.

    `where` optionally pins a node's badge to a compass direction.
    """
    deg = {}
    for a, b, *_ in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    for k, d in deg.items():
        if d % 2:
            s.nodes[prefix + k].style = odd_style
    for k, d in sorted(deg.items()):
        s.badge(prefix + k, str(d), color="accent" if d % 2 else "ink",
                where=(where or {}).get(k, "auto"))


@figure("is_eulerian", "predicates", "Is Eulerian",
        "Left: a bowtie of two triangles sharing node C; badges give each node's degree and "
        "all are even, so a closed trail can use every edge once and is_eulerian() returns "
        "True. Right: the same bowtie plus an edge between B and E, which makes B and E odd "
        "(degree 3, shown in the accent color), so is_eulerian() returns False.")
def is_eulerian(s: Scene) -> None:
    left, right = contrast(s, "g.is_eulerian()", "g.is_eulerian()")
    s.graph(place(BOWTIE, left), BOWTIE_EDGES, prefix="l.")
    _degrees(s, "l.", BOWTIE_EDGES)
    edges = BOWTIE_EDGES + [("B", "E")]
    s.graph(place(BOWTIE, right), edges[:-1] + [("B", "E", {"style": "accent"})], prefix="r.")
    _degrees(s, "r.", edges)


@figure("is_semieulerian", "predicates", "Is semi-Eulerian",
        "Left: a bowtie plus an edge between B and E; badges give each node's degree and "
        "exactly two nodes, B and E, are odd (outlined), so a trail from one to the other uses "
        "every edge once and is_semieulerian() returns True. Right: the same graph plus an edge "
        "between A and D, which leaves four odd nodes (shown in the accent color), so "
        "is_semieulerian() returns False.")
def is_semieulerian(s: Scene) -> None:
    left, right = contrast(s, "g.is_semieulerian()", "g.is_semieulerian()")
    edges = BOWTIE_EDGES + [("B", "E")]
    s.graph(place(BOWTIE, left), edges, prefix="l.")
    _degrees(s, "l.", edges, odd_style="soft")
    s.graph(place(BOWTIE, right), edges + [("A", "D", {"style": "accent"})], prefix="r.")
    _degrees(s, "r.", edges + [("A", "D")])


@figure("has_eulerian_path", "predicates", "Has Eulerian path",
        "Left: the house of Santa Claus, a square with both diagonals and a roof; badges give "
        "each node's degree and only the bottom corners D and E are odd (outlined), so one "
        "stroke can draw it and has_eulerian_path() returns True. Right: without the roof "
        "(dashed), all four corners have degree 3 (shown in the accent color), so "
        "has_eulerian_path() returns False.")
def has_eulerian_path(s: Scene) -> None:
    left, right = contrast(s, "g.has_eulerian_path()", "g.has_eulerian_path()")
    s.graph(place(HOUSE, left), SANTA_EDGES, prefix="l.")
    _degrees(s, "l.", SANTA_EDGES, odd_style="soft")
    kept = [e for e in SANTA_EDGES if "A" not in e]
    s.graph(place(HOUSE, right), kept + [(a, b, {"style": "removed"}) for a, b in SANTA_EDGES
                                         if "A" in (a, b)],
            styles={"A": "removed"}, prefix="r.")
    _degrees(s, "r.", kept)


# Flat-topped hexagon, A at the upper left, read clockwise.
HEX = ring(list("ABCDEF"), radius=1.1, start_deg=-120)
HEX_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "A")]
HEX_OUT = dict(zip("ABCDEF", ("nw", "ne", "e", "se", "sw", "w")))  # degree badges sit outside


@figure("is_regular", "predicates", "Is regular",
        "Left: a six-node cycle; the badges show every node has degree 2, so is_regular() "
        "returns True. Right: the same cycle plus a chord between C and F (shown in the accent "
        "color), which raises C and F to degree 3 while the rest stay at 2, so is_regular() "
        "returns False.")
def is_regular(s: Scene) -> None:
    left, right = contrast(s, "g.is_regular()", "g.is_regular()")
    s.graph(place(HEX, left), HEX_EDGES, prefix="l.")
    s.graph(place(HEX, right), HEX_EDGES + [("C", "F", {"style": "accent"})],
            styles={"C": "accent", "F": "accent"}, prefix="r.")
    for p, odd in (("l.", ""), ("r.", "CF")):
        for k in HEX:
            s.badge(p + k, "3" if k in odd else "2", where=HEX_OUT[k],
                    color="accent" if k in odd else "ink")


@figure("is_k_regular", "predicates", "Is k-regular",
        "The same six-node cycle on both sides, with badges showing every node has degree 2. "
        "Left: asking for k=2 matches every node, so is_k_regular(k=2) returns True. Right: "
        "asking for k=3 matches none of them (badges in the accent color), so "
        "is_k_regular(k=3) returns False even though the graph is regular.")
def is_k_regular(s: Scene) -> None:
    left, right = contrast(s, "g.is_k_regular(k=2)", "g.is_k_regular(k=3)")
    s.graph(place(HEX, left), HEX_EDGES, prefix="l.")
    s.graph(place(HEX, right), HEX_EDGES, prefix="r.")
    for p, c in (("l.", "ink"), ("r.", "accent")):
        for k in HEX:
            s.badge(p + k, "2", where=HEX_OUT[k], color=c)


# --- structure -------------------------------------------------------------------

ZIG = {"A": (1.2, 0), "B": (3.6, 0), "C": (0, 1.3), "D": (2.4, 1.3)}
ZIG_EDGES = [("A", "C"), ("A", "D"), ("C", "D"), ("B", "D")]


@figure("is_simple", "predicates", "Is simple",
        "Left: four nodes where each pair is joined by at most one edge and no edge loops back "
        "to its own node, so is_simple() returns True. Right: the same graph with a self-loop "
        "on C and a second, parallel edge between B and D (both shown in the accent color), "
        "so is_simple() returns False.")
def is_simple(s: Scene) -> None:
    left, right = contrast(s, "g.is_simple()", "g.is_simple()")
    s.graph(place(ZIG, left), ZIG_EDGES, prefix="l.")
    s.graph(place(ZIG, right),
            ZIG_EDGES + [("B", "D", {"bend": 18, "style": "accent"}),
                         ("C", "C", {"style": "accent"})], prefix="r.")


TRI = {"A": (1.2, 0), "B": (0, 1.4), "C": (2.4, 1.4)}
TRI_EDGES = [("A", "B"), ("A", "C"), ("B", "C")]


@figure("is_directed", "predicates", "Is directed",
        "Left: a triangle whose edges carry arrowheads, a graph created as directed, so "
        "is_directed() returns True. Right: the same triangle drawn with plain lines, an "
        "undirected graph, so is_directed() returns False.")
def is_directed(s: Scene) -> None:
    left, right = contrast(s, "g.is_directed()", "g.is_directed()")
    s.graph(place(TRI, left), TRI_EDGES, directed=True, prefix="l.")
    s.graph(place(TRI, right), TRI_EDGES, prefix="r.")


@figure("is_weighted", "predicates", "Is weighted",
        "Left: a triangle whose edges carry the weights 2.5, 4, and 1, a graph created as "
        "weighted, so is_weighted() returns True. Right: the same triangle with no weights, "
        "so is_weighted() returns False.")
def is_weighted(s: Scene) -> None:
    left, right = contrast(s, "g.is_weighted()", "g.is_weighted()")
    w = {("A", "B"): "2.5", ("A", "C"): "4", ("B", "C"): "1"}
    s.graph(place(TRI, left), [(a, b, {"weight": w[a, b]}) for a, b in TRI_EDGES], prefix="l.")
    s.graph(place(TRI, right), TRI_EDGES, prefix="r.")


def _crossings(s: Scene, prefix: str, edges) -> list[tuple[float, float]]:
    """Points where two straight edges of a panel cross (excluding shared endpoints)."""
    pts = []
    for i, (a, b) in enumerate(edges):
        for c, d in edges[i + 1:]:
            if {a, b} & {c, d}:
                continue
            p, q = s.nodes[prefix + a], s.nodes[prefix + b]
            r, t = s.nodes[prefix + c], s.nodes[prefix + d]
            den = (q.x - p.x) * (t.y - r.y) - (q.y - p.y) * (t.x - r.x)
            if abs(den) < 1e-9:
                continue
            u = ((r.x - p.x) * (t.y - r.y) - (r.y - p.y) * (t.x - r.x)) / den
            v = ((r.x - p.x) * (q.y - p.y) - (r.y - p.y) * (q.x - p.x)) / den
            if 0 < u < 1 and 0 < v < 1:
                pts.append((p.x + u * (q.x - p.x), p.y + u * (q.y - p.y)))
    return pts


@figure("is_planar", "predicates", "Is planar",
        "Left: the complete graph on four nodes drawn with D inside triangle A B C, so no "
        "edges cross; its 6 edges are within the 3n - 6 bound and is_planar() returns True. "
        "Right: the complete graph on five nodes, where the inner star's edges cross at the "
        "points marked in the accent color; its 10 edges exceed the bound of 9, so is_planar() "
        "returns False.")
def is_planar(s: Scene) -> None:
    left, right = contrast(s, "g.is_planar()", "g.is_planar()")
    k4 = {"A": (1, 0), "B": (0, 1.6), "C": (2, 1.6), "D": (1, 1.05)}
    k5 = ring(list("ABCDE"), radius=0.95)
    for box, coords, n, e, bound, ok in ((left, k4, 4, 6, 6, True), (right, k5, 5, 10, 9, False)):
        p = f"{box.x0}."
        gb = Box(box.x0, box.y0, box.x1, box.y1 - 22)  # leave a row for the edge bound
        edges = [(a, b) for i, a in enumerate(coords) for b in list(coords)[i + 1:]]
        s.graph(place(coords, gb), edges, prefix=p)
        s.text(box.cx, box.y1 - 8, f"{e} edges {'≤' if ok else '>'} 3n − 6 = {bound}",
               size=SIZE_NOTE, opacity=0.6 if ok else 1, color="ink" if ok else "accent")
        for x, y in _crossings(s, p, edges):
            s.path(f"M {x - 4:.1f} {y:.1f} a 4 4 0 1 0 8 0 a 4 4 0 1 0 -8 0", color="accent",
                   fill="color", width=1)


# is_chordal docstring: the square A, B, C, D, then with the diagonal A–C.
CHORD_SQ = {"A": (0, 0), "B": (1.5, 0), "C": (1.5, 1.5), "D": (0, 1.5)}
CHORD_SQ_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")]


@figure("is_chordal", "predicates", "Is chordal",
        "Left: a square A, B, C, D with the diagonal A to C, which splits it into two "
        "triangles, so every cycle of four or more nodes has a chord and is_chordal() "
        "returns True. Right: the square without the diagonal; the cycle A, B, C, D (shown "
        "in the accent color) has no chord, so is_chordal() returns False.")
def is_chordal(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(list(CHORD_SQ))
    g.add_edges(CHORD_SQ_EDGES)
    assert g.is_chordal() is False
    g.add_edge("A", "C")
    assert g.is_chordal() is True
    left, right = contrast(s, "g.is_chordal()", "g.is_chordal()")
    s.graph(place(CHORD_SQ, left), CHORD_SQ_EDGES + [("A", "C")], prefix="l.")
    s.graph(place(CHORD_SQ, right), _hi(CHORD_SQ_EDGES, set(CHORD_SQ_EDGES)),
            styles=dict.fromkeys(CHORD_SQ, "accent"), prefix="r.")



@figure("is_isomorphic", "predicates", "Is isomorphic",
        "Each side compares graph g with graph h. Left: g is a five-node cycle and h is the "
        "same cycle redrawn as a star, so is_isomorphic(h) returns True. Right: h is a "
        "triangle with a two-edge tail; it has the same node and edge counts, but its degree-3 "
        "node C and degree-1 node E (shown in the accent color) have no match in the cycle, so "
        "is_isomorphic(h) returns False.")
def is_isomorphic(s: Scene) -> None:
    left, right = contrast(s, "g.is_isomorphic(h)", "g.is_isomorphic(h)")
    c5 = ring(list("ABCDE"), radius=0.85)
    c5_edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")]
    # The same cycle with labels placed every second spot around the ring: a pentagram.
    star = dict(zip("ADBEC", ring(list("01234"), radius=0.85).values()))
    tail = {"A": (0, 0), "B": (1.3, 0), "C": (0.65, 0.85), "D": (0.1, 1.7), "E": (1.2, 1.7)}
    tail_edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")]
    for box, h, h_edges, odd in ((left, star, c5_edges, ""), (right, tail, tail_edges, "CE")):
        gbox = Box(box.x0, box.y0 + 22, box.cx, box.y1)
        hbox = Box(box.cx, box.y0 + 22, box.x1, box.y1)
        for sub, name in ((gbox, "g"), (hbox, "h")):
            s.text(sub.cx, box.y0 + 8, name, size=SIZE_NOTE, opacity=0.6)
        s.graph(place(c5, gbox, r=R_SMALL), c5_edges, r=R_SMALL, prefix=f"{box.x0}g.")
        s.graph(place(h, hbox, r=R_SMALL), h_edges, r=R_SMALL,
                styles={k: "accent" for k in odd}, prefix=f"{box.x0}h.")


# could_be_isomorphic docstring: a hexagon against two triangles (both 2-regular).
HEX6 = ring([str(i) for i in range(6)], radius=0.85, start_deg=-90)
HEX6_EDGES = [(str(i), str((i + 1) % 6)) for i in range(6)]
TWO_TRI = {"0": (0.5, 0), "1": (0, 0.85), "2": (1, 0.85),
           "3": (2.1, 0), "4": (1.6, 0.85), "5": (2.6, 0.85)}
TWO_TRI_EDGES = [("0", "1"), ("1", "2"), ("0", "2"), ("3", "4"), ("4", "5"), ("3", "5")]


@figure("could_be_isomorphic", "predicates", "Could be isomorphic",
        "Each side compares graph g, a six-node cycle, with graph h, two separate triangles; "
        "both have 6 nodes and 6 edges. Left: comparing degrees only, every node has degree 2 "
        "(badges), so could_be_isomorphic(h, properties=d) returns True. Right: the default "
        "properties also count triangles per node, 0 on the cycle and 1 on every triangle node "
        "(badges in the accent color), so could_be_isomorphic(h) returns False and the graphs "
        "are certainly not isomorphic.")
def could_be_isomorphic(s: Scene) -> None:
    import edgecase as ec

    hexagon = ec.Graph().add_cycle_graph(6)
    two = ec.Graph().add_cycle_graph(3).add_cycle_graph(3)
    assert hexagon.could_be_isomorphic(two, properties="d") is True
    assert hexagon.could_be_isomorphic(two) is False
    assert hexagon.could_be_isomorphic(two, properties="t") is False
    left, right = contrast(s, 'g.could_be_isomorphic(h, properties="d")',
                           "g.could_be_isomorphic(h)")
    for box, side in ((left, "d"), (right, "t")):
        gbox = Box(box.x0, box.y0 + 22, box.cx, box.y1)
        hbox = Box(box.cx, box.y0 + 22, box.x1, box.y1)
        tag = "degree" if side == "d" else "triangles"
        for sub, name in ((gbox, "g"), (hbox, "h")):
            s.text(sub.cx, box.y0 + 8, f"{name}: {tag}", size=SIZE_NOTE, opacity=0.6)
        gp, hp = f"{box.x0}g.", f"{box.x0}h."
        s.graph(place(HEX6, gbox, r=R_SMALL), HEX6_EDGES, r=R_SMALL, labels=False, prefix=gp)
        s.graph(place(TWO_TRI, hbox, r=R_SMALL), TWO_TRI_EDGES, r=R_SMALL, labels=False,
                styles={k: "accent" for k in TWO_TRI} if side == "t" else None, prefix=hp)
        for k, w in zip(HEX6, ("n", "ne", "se", "s", "sw", "nw")):
            s.badge(gp + k, "2" if side == "d" else "0", where=w, color="ink")
        for k in TWO_TRI:
            s.badge(hp + k, "2" if side == "d" else "1", where="n" if k in "03" else "s",
                    color="accent" if side == "t" else "ink")


# is_subgraph_isomorphic docstring: a 3-node path pattern inside a triangle. Both
# share one footprint; the pattern is the triangle with its base absent (dashed).
PEAK = {"A": (0, 1), "B": (1, 0), "C": (2, 1)}


@figure("is_subgraph_isomorphic", "predicates", "Is subgraph isomorphic",
        "Each side searches graph g, a triangle A, B, C, for pattern p, a path of three "
        "nodes. Left: with induced=False extra edges are allowed, so the path maps onto "
        "B–A and B–C (in the accent color) and is_subgraph_isomorphic(p, induced=False) "
        "returns True. Right: with the default induced=True the matched nodes may have no "
        "extra edge, but the edge A–C (in the accent color) closes every candidate path into "
        "the triangle, so is_subgraph_isomorphic(p) returns False.")
def is_subgraph_isomorphic(s: Scene) -> None:
    import edgecase as ec

    tri = ec.Graph().add_cycle_graph(3)
    path = ec.Graph().add_path_graph(3)
    assert tri.is_subgraph_isomorphic(path, induced=False) is True
    assert tri.is_subgraph_isomorphic(path) is False
    left, right = contrast(s, "g.is_subgraph_isomorphic(p, induced=False)",
                           "g.is_subgraph_isomorphic(p)")
    for box, induced in ((left, False), (right, True)):
        pbox = Box(box.x0, box.y0 + 22, box.x0 + box.w * 0.42, box.y1)
        gbox = Box(pbox.x1, box.y0 + 22, box.x1, box.y1)
        for sub, name in ((pbox, "p"), (gbox, "g")):
            s.text(sub.cx, box.y0 + 8, name, size=SIZE_NOTE, opacity=0.6)
        s.graph(place(PEAK, pbox, r=R_SMALL, unit=44), [("A", "B"), ("B", "C")],
                r=R_SMALL, labels=False, styles=dict.fromkeys(PEAK, "soft"),
                prefix=f"{box.x0}p.")
        path_style = {"style": "soft" if induced else "accent"}
        s.graph(place(PEAK, gbox, r=R_SMALL), [("A", "B", path_style), ("B", "C", path_style),
                                               ("A", "C", {"style": "accent" if induced
                                                           else "muted"})],
                r=R_SMALL, styles=dict.fromkeys(PEAK, "soft" if induced else "accent"),
                prefix=f"{box.x0}g.")


# --- reachability ------------------------------------------------------------------

# A to D runs A -> B -> C -> D (E is a side branch into C). Reversing C -> D
# leaves D with no incoming edge, so nothing reaches it.
REACH = {"A": (0, 0.5), "B": (1, 0), "C": (2, 0), "D": (3, 0.5), "E": (1.5, 1.1)}
REACH_EDGES = [("A", "B"), ("B", "C"), ("A", "E"), ("E", "C")]


@figure("has_path", "predicates", "Has path",
        "Left: a five-node directed graph in which the arrows lead from A through B and C to "
        "D (the path in the accent color), so a path from A to D exists and has_path() "
        "returns True. Right: the same graph with the edge between C and D reversed (in the "
        "accent color); no arrow enters D any more, so has_path() returns False.")
def has_path(s: Scene) -> None:
    left, right = contrast(s, 'g.has_path("A", "D")', 'g.has_path("A", "D")')
    ends = {"A": "soft", "D": "soft"}
    s.graph(place(REACH, left),
            _hi(REACH_EDGES, {("A", "B"), ("B", "C")}) + [("C", "D", {"style": "accent"})],
            directed=True, styles=ends, prefix="l.")
    s.graph(place(REACH, right), REACH_EDGES + [("D", "C", {"style": "accent"})],
            directed=True, styles=ends, prefix="r.")
