"""Substructures: a set of nodes or edges with a special property.

Template: `compute`. Left: the graph with the found structure (tree edges,
matched edges, independent/dominating/clique nodes) in `accent` and the rest
`default`, or `muted` where it would compete. Right: a card listing the
structure as edgecase returns it (and the total weight for weighted ones).

Methods that return *several* structures (all_spanning_trees, find_cliques,
find_maximal_independent_sets) leave the left graph plain and stack small
multiples on the right: one miniature of the same graph per result, the
result in `accent`, everything else `muted`, with the returned list beside it.

Sibling methods share a graph so they can be compared directly: one weighted
graph for both spanning trees; one five-node "bull" for matchings,
independent sets, and cliques; one seven-node graph for dominating sets.
Methods that return node or edge ids label nodes with their ids (edgecase
ids start at 1); label-based methods use letters.

The Steiner variants (prize-collecting, directed) reuse the steiner_tree
graph, with prizes as node badges. The motif methods use the dominating-set
graph and stack one small motif shape per class on the right (the
all_spanning_trees rows), with its count (and z-score) beside it.
`subgraph_isomorphisms` splits the input panel into the pattern (`soft`) and
the host, and stacks one miniature host per occurrence like find_cliques.
`isolates` returns ids, so its two-row graph is labeled with ids and the
edgeless nodes are `accent`.
`triadic_census` uses the five-node directed network from its docstring with a
type / count / triads table; two triads are highlighted in the graph and their
rows: the cycle 030C in accent, the 120U triad in g2. Its per-triad
companions follow it: triad_type shows two copies of its graph, the two
queried triples in accent and g2; triads_by_type and all_triads stack one
miniature per returned triple in a 2 x 2 grid, labeled with the type or ids.
The later clique variants rebuild their docstring examples in edgecase and
assert the results: enumerate_all_cliques stacks mini triangles in rows by
clique size; max_weight_clique shows node weights as ink badges;
node_clique_number / number_of_cliques draw the paw's two maximal cliques as
hulls with node-metrics heat and a sorted table; make_max_clique_graph is a
`transform` whose clique nodes keep their hull colors. The chordal methods
(chordal_graph_cliques, chordal_graph_treewidth, find_induced_nodes) share the
diamond-plus-pendant graph or the path A–D of their docstrings;
find_induced_nodes draws the hypothetical edge s–t as a `ghost` arc.
"""

from core import (
    R_SMALL,
    SIZE_CARD,
    SIZE_NOTE,
    Box,
    Scene,
    compute,
    figure,
    place,
    text_width,
    transform,
)

UNIT = 84  # family-wide grid step for the main graph
MINI_UNIT = 36  # grid step inside the small multiples (max that keeps rows apart)

# --- shared graphs ---------------------------------------------------------

# Weighted graph for minimum_spanning_tree / maximum_spanning_tree.
SPAN = {"A": (0, 0), "B": (1.4, 0), "C": (0, 1.2), "D": (1.4, 1.2), "E": (2.6, 0.6)}
SPAN_EDGES = [("A", "B", 4), ("A", "C", 1), ("A", "D", 5), ("B", "D", 2), ("C", "D", 3),
              ("B", "E", 6), ("D", "E", 7)]

# The bull: a triangle (2, 3, 5) with two horns (1, 4). Edge 1 is 2–3, so
# the greedy maximal matching takes it first and blocks everything else.
BULL = {"1": (0, 0), "2": (1, 0), "3": (2, 0), "4": (3, 0), "5": (1.5, 0.9)}
BULL_EDGES = [("2", "3"), ("1", "2"), ("3", "4"), ("2", "5"), ("3", "5")]

# Pendant 1, triangle (2, 3, 4), bridge 4–5, triangle (5, 6, 7).
DOM = {"1": (0, 0.6), "2": (1, 0), "3": (1, 1.2), "4": (2, 0.6), "5": (3, 0.6),
       "6": (4, 0), "7": (4, 1.2)}
DOM_EDGES = [("1", "2"), ("2", "3"), ("2", "4"), ("3", "4"), ("4", "5"), ("5", "6"),
             ("5", "7"), ("6", "7")]

# Triangle with a pendant, for all_spanning_trees (three trees).
PAW = {"A": (0.6, 0), "B": (0, 0.7), "C": (1.2, 0.7), "D": (2.4, 0.7)}
PAW_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")]


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _edges(edges, on: set, off: str = "muted", weights: dict | None = None) -> list:
    """Edge specs: those in `on` accent, the rest `off`; optional weights."""
    out = []
    for e in edges:
        a, b = e[0], e[1]
        kw = {"style": "accent" if _key(a, b) in on else off}
        if len(e) > 2:
            kw["weight"] = e[2]
        elif weights is not None:
            kw["weight"] = weights[(a, b)]
        if kw["style"] == "default":
            del kw["style"]
        out.append((a, b, kw))
    return out


def _multiples(s: Scene, right: Box, coords, edges, results: list[tuple[set, set, str]]) -> None:
    """Stack one miniature per result in `right`, each with its list beside it.

    `results` items are (accent nodes, accent edges as sorted pairs, text).
    """
    xs = [p[0] for p in coords.values()]
    gw = (max(xs) - min(xs)) * MINI_UNIT + 2 * R_SMALL
    tw = max(text_width(t, SIZE_CARD) for _, _, t in results)
    gap = 18
    x0 = right.cx - (gw + gap + tw) / 2
    row_h = right.h / len(results)
    for i, (nodes, on, label) in enumerate(results):
        cell = Box(x0, right.y0 + i * row_h, x0 + gw, right.y0 + (i + 1) * row_h)
        pos = place(coords, cell, unit=MINI_UNIT, r=R_SMALL, pad=0)
        s.graph(pos, _edges(edges, on, off="removed" if not nodes else "muted"),
                styles={k: "accent" if k in nodes or not nodes else "muted" for k in coords},
                r=R_SMALL, prefix=f"m{i}.")
        s.text(x0 + gw + gap, cell.cy, label, size=SIZE_CARD, anchor="start", color="accent",
               weight=600, role="card")


# --- spanning trees --------------------------------------------------------


@figure("minimum_spanning_tree", "substructures", "Minimum spanning tree",
        "A five-node weighted graph with edge weights 1 to 7. The minimum spanning tree "
        "edges A–C (1), B–D (2), C–D (3), and B–E (6) are in the accent color and connect "
        "every node; the other edges are faded. The card lists the four tree edges and "
        "their total weight, 1 + 2 + 3 + 6 = 12.")
def minimum_spanning_tree(s: Scene) -> None:
    left, right = compute(s, "g", "g.minimum_spanning_tree()")
    tree = {("A", "C"), ("B", "D"), ("C", "D"), ("B", "E")}
    s.graph(place(SPAN, left, unit=UNIT), _edges(SPAN_EDGES, tree))
    s.card_centered(right, ["A–C B–D C–D B–E", ("1 + 2 + 3 + 6 = 12", "accent")],
                    title="tree edges")


@figure("maximum_spanning_tree", "substructures", "Maximum spanning tree",
        "The same weighted graph as the minimum spanning tree figure. The maximum spanning "
        "tree keeps the heaviest edges that still form a tree: D–E (7), B–E (6), A–D (5), "
        "and C–D (3), in the accent color. The card lists them and their total weight, "
        "7 + 6 + 5 + 3 = 21.")
def maximum_spanning_tree(s: Scene) -> None:
    left, right = compute(s, "g", "g.maximum_spanning_tree()")
    tree = {("D", "E"), ("B", "E"), ("A", "D"), ("C", "D")}
    s.graph(place(SPAN, left, unit=UNIT), _edges(SPAN_EDGES, tree))
    s.card_centered(right, ["D–E B–E A–D C–D", ("7 + 6 + 5 + 3 = 21", "accent")],
                    title="tree edges")


@figure("all_spanning_trees", "substructures", "All spanning trees",
        "Left: a triangle A, B, C with a pendant node D attached to C. Right: its three "
        "spanning trees, one per row; each keeps edge C–D and drops one triangle edge "
        "(drawn dashed), with the tree edges A–C B–C C–D, A–B B–C C–D, and A–B A–C C–D "
        "listed beside them.")
def all_spanning_trees(s: Scene) -> None:
    left, right = compute(s, "g", "g.all_spanning_trees()")
    s.graph(place(PAW, left, unit=UNIT), PAW_EDGES)
    trees = [{("A", "C"), ("B", "C"), ("C", "D")}, {("A", "B"), ("B", "C"), ("C", "D")},
             {("A", "B"), ("A", "C"), ("C", "D")}]
    _multiples(s, right, PAW, PAW_EDGES,
               [(set(), t, " ".join(f"{a}–{b}" for a, b in sorted(t))) for t in trees])


@figure("steiner_tree", "substructures", "Steiner tree",
        "A weighted five-node graph with terminals B, C, and E outlined in the accent color. "
        "The Steiner tree joins them through the non-terminal node D, filled in the accent "
        "color, using edges B–D, D–C, and D–E; node A and the heavier outer edges are "
        "unused. The card lists the tree edges and the total weight 2 + 2 + 1 = 5.")
def steiner_tree(s: Scene) -> None:
    left, right = compute(s, "g", 'g.steiner_tree(["B", "C", "E"])')
    coords = {"A": (1.5, 0), "B": (0, 0.8), "C": (3, 0.8), "D": (1.5, 0.8), "E": (1.5, 1.6)}
    edges = [("A", "B", 3), ("A", "C", 3), ("B", "E", 4), ("C", "E", 4), ("B", "D", 2),
             ("C", "D", 2), ("D", "E", 1)]
    tree = {("B", "D"), ("C", "D"), ("D", "E")}
    s.graph(place(coords, left, unit=UNIT), _edges(edges, tree),
            styles={"B": "soft", "C": "soft", "E": "soft", "D": "accent", "A": "muted"})
    s.card_centered(right, ["B–D D–C D–E", ("2 + 2 + 1 = 5", "accent")], title="tree edges")


# --- matchings -------------------------------------------------------------


@figure("maximal_matching", "substructures", "Maximal matching",
        "A five-node graph: a triangle 2, 3, 5 with pendant nodes 1 and 4. The greedy "
        "maximal matching takes edge 2–3 first (accent); every other edge touches 2 or 3, "
        "so nothing more can be added, even though the larger matching 1–2, 3–4 exists. "
        "The card shows the returned edge id 1, the pair 2–3, and size 1.")
def maximal_matching(s: Scene) -> None:
    left, right = compute(s, "g", "g.maximal_matching()")
    s.graph(place(BULL, left, unit=UNIT), _edges(BULL_EDGES, {("2", "3")}),
            styles={"2": "accent", "3": "accent"})
    s.card_centered(right, [("edge ids  [1]", "accent"), "pairs     2–3",
                            "size      1 (maximum 2)"])


# max_weight_matching docstring: the heavy middle edge B–C is a trap.
TRAP = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0)}
TRAP_EDGES = [("A", "B", 4), ("B", "C", 5), ("C", "D", 4)]


@figure("max_weight_matching", "substructures", "Maximum weight matching",
        "A weighted path A–B (4), B–C (5), C–D (4). The exact blossom matching takes the two "
        "outer edges A–B and C–D, in the accent color with their endpoints, for a total "
        "weight of 4 + 4 = 8; the heavier middle edge B–C is faded. The card notes that the "
        "greedy method would lock in B–C first and stop at a total of 5.")
def max_weight_matching(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.max_weight_matching()")
    g = ec.Graph(weighted=True).add_nodes(list(TRAP))
    g.add_edges([(a, b, float(w)) for a, b, w in TRAP_EDGES])
    exact = _id_edges(g, g.max_weight_matching())
    greedy = _id_edges(g, g.max_weight_matching(method="greedy"))
    ex_on = {_key(a, b) for a, b, _ in exact}
    assert ex_on == {("A", "B"), ("C", "D")}
    assert [_key(a, b) for a, b, _ in greedy] == [("B", "C")]
    s.graph(place(TRAP, left, unit=UNIT), _edges(TRAP_EDGES, ex_on),
            styles={k: "accent" for k in TRAP})
    gw = [w for _, _, w in greedy]
    s.card_centered(right, [" ".join(f"{a}–{b}" for a, b in sorted(ex_on)),
                            (_total([w for _, _, w in exact]), "accent"),
                            f"greedy: {greedy[0][0]}–{greedy[0][1]} = {_num(gw[0])}"],
                    title="matched edges")


# --- independent sets ------------------------------------------------------


@figure("maximal_independent_set", "substructures", "Maximal independent set",
        "A five-node graph: a triangle 2, 3, 5 with pendant nodes 1 and 4. Nodes 2 and 4, "
        "in the accent color, share no edge, and every other node is adjacent to one of "
        "them, so no node can be added. The card shows the result [2, 4], size 2, smaller "
        "than the maximum of 3.")
def maximal_independent_set(s: Scene) -> None:
    left, right = compute(s, "g", "g.maximal_independent_set(seed=1)")
    s.graph(place(BULL, left, unit=UNIT), BULL_EDGES, styles={"2": "accent", "4": "accent"})
    s.card_centered(right, [("nodes  [2, 4]", "accent"), "size   2 (maximum 3)"])


@figure("maximum_independent_set", "substructures", "Maximum independent set",
        "The same five-node graph. Nodes 1, 4, and 5, in the accent color, share no edge, "
        "and no independent set here is larger. The card shows the result [1, 4, 5], size 3.")
def maximum_independent_set(s: Scene) -> None:
    left, right = compute(s, "g", "g.maximum_independent_set()")
    s.graph(place(BULL, left, unit=UNIT), BULL_EDGES,
            styles={"1": "accent", "4": "accent", "5": "accent"})
    s.card_centered(right, [("nodes  [1, 4, 5]", "accent"), "size   3"])


@figure("find_maximal_independent_sets", "substructures", "Find maximal independent sets",
        "Left: a five-node graph, a triangle 2, 3, 5 with pendant nodes 1 and 4. Right: "
        "three miniature copies, each highlighting one maximal independent set in the "
        "accent color, listed beside it: [1, 3], [1, 4, 5], and [2, 4].")
def find_maximal_independent_sets(s: Scene) -> None:
    left, right = compute(s, "g", "g.find_maximal_independent_sets()")
    s.graph(place(BULL, left, unit=UNIT), BULL_EDGES)
    sets = [["1", "3"], ["1", "4", "5"], ["2", "4"]]
    _multiples(s, right, BULL, BULL_EDGES,
               [(set(x), set(), "[" + ", ".join(x) + "]") for x in sets])


# --- dominating sets -------------------------------------------------------


@figure("dominating_set", "substructures", "Dominating set",
        "A seven-node graph: pendant 1, triangle 2, 3, 4, a bridge to triangle 5, 6, 7. "
        "Nodes 1, 3, and 5 form the dominating set (accent); an accent edge links each "
        "other node to a set member that dominates it. The card shows the result "
        "[1, 3, 5], size 3.")
def dominating_set(s: Scene) -> None:
    left, right = compute(s, "g", "g.dominating_set()")
    # one dominating edge per node outside the set
    witness = {("1", "2"), ("3", "4"), ("5", "6"), ("5", "7")}
    s.graph(place(DOM, left, unit=UNIT), _edges(DOM_EDGES, witness),
            styles={"1": "accent", "3": "accent", "5": "accent"})
    s.card_centered(right, [("nodes  [1, 3, 5]", "accent"), "size   3"])


@figure("min_weighted_dominating_set", "substructures", "Minimum weighted dominating set",
        "The same seven-node graph, with each node's weight w in a badge: the hubs 2 and 5 "
        "cost 4, every other node costs 1. The cheapest dominating set found is 1, 4, and 6 "
        "(accent), with an accent edge from each other node to the member that dominates "
        "it. The card shows [1, 4, 6] and total weight 1 + 1 + 1 = 3.")
def min_weighted_dominating_set(s: Scene) -> None:
    left, right = compute(s, "g", 'g.min_weighted_dominating_set("w")')
    witness = {("1", "2"), ("3", "4"), ("4", "5"), ("6", "7")}
    s.graph(place(DOM, left, unit=UNIT), _edges(DOM_EDGES, witness),
            styles={"1": "accent", "4": "accent", "6": "accent"})
    weights = {"1": 1, "2": 4, "3": 1, "4": 1, "5": 4, "6": 1, "7": 1}
    # badges above top-row nodes and below bottom-row ones, so the 4–5 bridge
    # stays clear (4 and 5 sit on the same row and would crowd it)
    where = {"1": "n", "2": "n", "3": "s", "4": "s", "5": "n", "6": "n", "7": "s"}
    for k, w in weights.items():
        s.badge(k, f"w={w}", where=where[k], color="ink")
    s.card_centered(right, [("nodes  [1, 4, 6]", "accent"), "weight 1 + 1 + 1 = 3"])


# --- cliques & triangles ---------------------------------------------------


@figure("find_cliques", "substructures", "Find maximal cliques",
        "Left: a five-node graph, a triangle 2, 3, 5 with pendant nodes 1 and 4. Right: "
        "three miniature copies, each highlighting one maximal clique in the accent color, "
        "listed beside it: [2, 3, 5], [3, 4], and [1, 2].")
def find_cliques(s: Scene) -> None:
    left, right = compute(s, "g", "g.find_cliques()")
    s.graph(place(BULL, left, unit=UNIT), BULL_EDGES)
    cliques = [["2", "3", "5"], ["3", "4"], ["1", "2"]]
    _multiples(s, right, BULL, BULL_EDGES,
               [(set(c), {_key(a, b) for a in c for b in c if a < b},
                 "[" + ", ".join(c) + "]") for c in cliques])


@figure("max_clique", "substructures", "Maximum clique",
        "The same five-node graph. The triangle 2, 3, 5, in the accent color, is the "
        "largest set of nodes that are all pairwise connected. The card shows the result "
        "[2, 3, 5], size 3.")
def max_clique(s: Scene) -> None:
    left, right = compute(s, "g", "g.max_clique()")
    s.graph(place(BULL, left, unit=UNIT),
            _edges(BULL_EDGES, {("2", "3"), ("2", "5"), ("3", "5")}, off="default"),
            styles={"2": "accent", "3": "accent", "5": "accent"})
    s.card_centered(right, [("nodes  [2, 3, 5]", "accent"), "size   3"])


@figure("list_triangles", "substructures", "List triangles",
        "A six-node graph with two triangles, 3, 5, 6 in the accent color and 2, 4, 5 in "
        "teal, sharing node 5, plus a four-node cycle 1, 2, 5, 3 that is not a triangle. "
        "The card lists the two triangles in matching colors, (3, 5, 6) and (2, 4, 5).")
def list_triangles(s: Scene) -> None:
    left, right = compute(s, "g", "g.list_triangles()")
    coords = {"1": (1.5, 0), "2": (0.75, 0.8), "3": (2.25, 0.8), "4": (0, 1.6),
              "5": (1.5, 1.6), "6": (3, 1.6)}
    t1, t2 = {("3", "5"), ("3", "6"), ("5", "6")}, {("2", "4"), ("2", "5"), ("4", "5")}
    edges = [("1", "2"), ("1", "3"), ("2", "4"), ("2", "5"), ("4", "5"), ("3", "5"),
             ("3", "6"), ("5", "6")]
    s.graph(place(coords, left, unit=UNIT),
            [(a, b, {"style": "accent"}) if (a, b) in t1 else
             (a, b, {"style": "g2"}) if (a, b) in t2 else (a, b) for a, b in edges],
            styles={"3": "accent", "6": "accent", "2": "g2", "4": "g2"})
    s.card_centered(right, [("(3, 5, 6)", "accent"), ("(2, 4, 5)", "g2")], title="triangles")


# --- later additions: clique variants ------------------------------------------
#
# Each builder rebuilds its docstring example in edgecase and asserts the result,
# so a library change breaks the build instead of leaving a stale picture.


def _ec_graph(nodes, edges):
    import edgecase as ec

    g = ec.Graph().add_nodes(nodes)
    g.add_edges(edges)
    return g


def _lab(g, ids) -> list[str]:
    return [g.get_label_from_id(n) for n in ids]


def _per_node(g, scores: dict) -> dict[str, int]:
    return {g.get_label_from_id(n): v for n, v in scores.items()}


# The triangle of enumerate_all_cliques (add_cycle_graph(3): ids and labels 1..3).
TRI3 = {"1": (0.5, 0), "2": (0, 0.62), "3": (1, 0.62)}
TRI3_EDGES = [("1", "2"), ("2", "3"), ("1", "3")]
TRI3_UNIT = 40  # three rows of mini triangles fill the result panel


@figure("enumerate_all_cliques", "substructures", "Enumerate all cliques",
        "Left: a triangle of nodes 1, 2, 3. Right: its seven cliques as miniature "
        "triangles in rows by size, each clique in the accent color: the three single "
        "nodes [1], [2], [3], the three edges [1, 2], [1, 3], [2, 3], and the whole "
        "triangle [1, 2, 3].")
def enumerate_all_cliques(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_cycle_graph(3)
    cliques = [_lab(g, c) for c in g.enumerate_all_cliques()]
    assert cliques == [["1"], ["2"], ["3"], ["1", "2"], ["1", "3"], ["2", "3"],
                       ["1", "2", "3"]]
    left, right = compute(s, "g", "g.enumerate_all_cliques()")
    s.graph(place({"1": (0.5, 0), "2": (0, 0.87), "3": (1, 0.87)}, left, unit=1.4 * UNIT),
            TRI3_EDGES)
    rows = [[c for c in cliques if len(c) == k] for k in (1, 2, 3)]
    gw = TRI3_UNIT + 2 * R_SMALL  # one mini triangle
    gap = 12
    lw = text_width("k=3", SIZE_NOTE) + gap
    x0 = right.cx - (lw + 3 * gw + 2 * gap) / 2
    row_h = right.h / 3
    i = 0
    for k, row in enumerate(rows):
        y0 = right.y0 + k * row_h
        s.text(x0, y0 + row_h / 2, f"k={k + 1}", size=SIZE_NOTE, anchor="start",
               opacity=0.6)
        for j, c in enumerate(row):
            cx0 = x0 + lw + j * (gw + gap)
            pos = place(TRI3, Box(cx0, y0, cx0 + gw, y0 + row_h), unit=TRI3_UNIT, r=R_SMALL,
                        pad=0)
            on = {_key(a, b) for a in c for b in c if a < b}
            s.graph(pos, _edges(TRI3_EDGES, on), r=R_SMALL, prefix=f"m{i}.",
                    styles={n: "accent" if n in c else "muted" for n in TRI3})
            i += 1


# max_weight_clique docstring: triangle A, B, C plus B–D, node weights in "w".
MWC = {"A": (0, 1.2), "B": (0.7, 0), "C": (1.4, 1.2), "D": (2.1, 0)}
MWC_EDGES = [("A", "B"), ("B", "C"), ("A", "C"), ("B", "D")]
MWC_W = {"A": 1, "B": 2, "C": 1, "D": 5}


@figure("max_weight_clique", "substructures", "Maximum weight clique",
        "Four nodes with weights w in badges: A 1, B 2, C 1, D 5. The triangle A, B, C "
        "has the most nodes but weighs only 1 + 2 + 1 = 4; the edge B to D, in the accent "
        "color, weighs 2 + 5 = 7. The card shows the result [B, D] with weight 7.")
def max_weight_clique(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes([(k, {"w": w}) for k, w in MWC_W.items()])
    g.add_edges(MWC_EDGES)
    nodes, weight = g.max_weight_clique(weight_attr="w")
    assert _lab(g, nodes) == ["B", "D"] and weight == 7
    left, right = compute(s, "g", 'g.max_weight_clique(weight_attr="w")')
    s.graph(place(MWC, left, unit=UNIT), _edges(MWC_EDGES, {("B", "D")}, off="default"),
            styles={"B": "accent", "D": "accent"})
    where = {"A": "s", "B": "n", "C": "s", "D": "n"}
    for k, w in MWC_W.items():
        s.badge(k, f"w={w}", where=where[k], color="ink")
    s.card_centered(right, [("[B, D]   2 + 5 = 7", "accent"), "A, B, C  1 + 2 + 1 = 4"],
                    title="heaviest clique")


# node_clique_number / number_of_cliques docstrings: the paw, triangle A, B, C
# plus the edge C–D. Its two maximal cliques are drawn as hulls (accent, teal).
PAW_CLQ = {"A": (0, 0), "B": (0, 1.2), "C": (1, 0.6), "D": (2.2, 0.6)}


def _clique_metric(s: Scene, call: str, header: str, scores: dict[str, int]) -> None:
    """Paw with its maximal cliques as hulls, nodes heat-shaded by score → table."""
    from figures.node_metrics import _rows

    left, right = compute(s, "g", call)
    top = max(scores.values())
    s.graph(place(PAW_CLQ, left, unit=UNIT),
            [("A", "B", {"style": "accent"}), ("B", "C", {"style": "accent"}),
             ("A", "C", {"style": "accent"}), ("C", "D", {"style": "g2"})],
            heat={k: v / top for k, v in scores.items()},
            styles={k: "accent" for k, v in scores.items() if v == top})
    s.hull(["A", "B", "C"], "accent")
    s.hull(["C", "D"], "g2")
    rows = [["node", header]] + _rows(scores)
    s.table_centered(right, rows, [72, 72], highlight={(1, 0): "accent", (1, 1): "accent"})


@figure("node_clique_number", "substructures", "Node clique number",
        "A triangle A, B, C with a tail edge C to D; its two maximal cliques are tinted "
        "regions, the triangle in the accent color and the edge C, D in teal. Each node is "
        "shaded by the size of the largest clique containing it: A, B and C are in the "
        "triangle, size 3, and D only in the edge, size 2, as the table lists.")
def node_clique_number(s: Scene) -> None:
    g = _ec_graph(list(PAW_CLQ), [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])
    scores = _per_node(g, g.node_clique_number())
    assert scores == {"A": 3, "B": 3, "C": 3, "D": 2}
    _clique_metric(s, "g.node_clique_number()", "largest", scores)


@figure("number_of_cliques", "substructures", "Number of cliques per node",
        "A triangle A, B, C with a tail edge C to D; its two maximal cliques are tinted "
        "regions, the triangle in the accent color and the edge C, D in teal. Each node is "
        "shaded by how many maximal cliques contain it: C sits in both, so it has 2 and the "
        "darkest shading, while A, B and D have 1 each, as the table lists.")
def number_of_cliques(s: Scene) -> None:
    g = _ec_graph(list(PAW_CLQ), [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")])
    scores = _per_node(g, g.number_of_cliques())
    assert scores == {"A": 1, "B": 1, "C": 2, "D": 1}
    _clique_metric(s, "g.number_of_cliques()", "cliques", scores)


# make_max_clique_graph docstring: triangles A, B, C and C, E, F sharing C,
# plus the tail C–D. Each maximal clique gets a group color in both panels.
BOWTIE = {"A": (0, 0), "B": (0, 1.2), "C": (1, 0.6), "E": (2, 0), "F": (2, 1.2),
          "D": (1, 1.75)}
BOWTIE_EDGES = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "E"), ("E", "F"), ("C", "F"),
                ("C", "D")]
BOWTIE_CLIQUES = [("ABC", "accent"), ("CD", "g2"), ("CEF", "g3")]


@figure("make_max_clique_graph", "substructures", "Maximal clique graph",
        "Left: two triangles A, B, C and C, E, F sharing node C, plus a tail C to D; the "
        "three maximal cliques are tinted regions in the accent color, teal and violet, "
        "all overlapping at C. Right: the clique graph has one node per clique, labeled "
        "{A,B,C}, {C,D} and {C,E,F} in the same colors, and an edge for every pair that "
        "shares a node: all three pairs share C, named on each edge, so 3 edges.")
def make_max_clique_graph(s: Scene) -> None:
    g = _ec_graph(list("ABCDEF"), BOWTIE_EDGES)
    cg = g.make_max_clique_graph()
    labels = sorted(cg.get_label_from_id(n) for n in cg.get_node_ids(limit=None))
    assert labels == ["{A,B,C}", "{C,D}", "{C,E,F}"] and cg.count_edges() == 3
    left, right = transform(s, "g", "g.make_max_clique_graph()")
    color = {k: c for members, c in BOWTIE_CLIQUES for k in members}
    # C is in all three cliques, so it stays ink inside the overlap
    s.graph(place(BOWTIE, left, unit=UNIT),
            [(a, b, {"style": color[b] if a == "C" else color[a]}) for a, b in BOWTIE_EDGES],
            styles={k: c for k, c in color.items() if k != "C"}, prefix="l.")
    for members, c in BOWTIE_CLIQUES:
        s.hull(["l." + k for k in members], c)
    rp = place({"ABC": (0, 0), "CEF": (2, 0), "CD": (1, 1.1)}, right, unit=UNIT * 1.1)
    # every pair of cliques overlaps in C, named on the edge
    s.graph(rp, [(a, b, {"weight": "C"}) for a, b in (("ABC", "CEF"), ("ABC", "CD"),
                                                       ("CD", "CEF"))],
            styles={m: c for m, c in BOWTIE_CLIQUES}, prefix="r.")
    for k, (x, y) in rp.items():
        lab = "{" + ",".join(k) + "}"
        s.text(x, y + (32 if k == "CD" else -30), lab, size=SIZE_NOTE, opacity=0.7)


# --- later additions: chordal graphs -------------------------------------------
#
# chordal_graph_cliques / chordal_graph_treewidth / find_induced_nodes use their
# docstring graphs: a diamond A, B, C, D with a pendant E, and the path A–D.

DIAMOND = {"A": (0, 0.6), "B": (1, 0), "C": (1, 1.2), "D": (2, 0.6), "E": (3.1, 0.6)}
DIAMOND_EDGES = [("A", "B"), ("B", "C"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]


def _diamond():
    g = _ec_graph(list(DIAMOND), DIAMOND_EDGES)
    cliques = [_lab(g, c) for c in g.chordal_graph_cliques()]
    assert cliques == [["A", "B", "C"], ["B", "C", "D"], ["D", "E"]]
    return g, cliques


@figure("chordal_graph_cliques", "substructures", "Chordal graph cliques",
        "A chordal graph: two triangles A, B, C and B, C, D sharing the edge B to C, plus a "
        "pendant E on D. Its maximal cliques are tinted regions, A, B, C in the accent "
        "color, B, C, D in teal and D, E in violet, overlapping where they share nodes. "
        "The card lists them in the same colors: [A, B, C], [B, C, D], [D, E].")
def chordal_graph_cliques(s: Scene) -> None:
    _, cliques = _diamond()
    left, right = compute(s, "g", "g.chordal_graph_cliques()")
    colors = ("accent", "g2", "g3")

    def style(a, b):
        hits = [c for clq, c in zip(cliques, colors) if a in clq and b in clq]
        return {"style": hits[0]} if len(hits) == 1 else {}

    # nodes in one clique take its color; B, C and D sit in two and stay ink
    s.graph(place(DIAMOND, left, unit=UNIT), [(a, b, style(a, b)) for a, b in DIAMOND_EDGES],
            styles={"A": "accent", "E": "g3"})
    for clq, c in zip(cliques, colors):
        s.hull(clq, c)
    s.card_centered(right, [("[" + ", ".join(c) + "]", col) for c, col in zip(cliques, colors)],
                    title="maximal cliques")


@figure("chordal_graph_treewidth", "substructures", "Chordal graph treewidth",
        "The same chordal graph: triangles A, B, C and B, C, D sharing an edge, plus a "
        "pendant E. Its largest cliques have three nodes; A, B, C is drawn in the accent "
        "color and B, C, D ties it. The card gives the treewidth as the largest clique "
        "size minus one, 3 − 1 = 2.")
def chordal_graph_treewidth(s: Scene) -> None:
    g, _ = _diamond()
    assert g.chordal_graph_treewidth() == 2
    left, right = compute(s, "g", "g.chordal_graph_treewidth()")
    s.graph(place(DIAMOND, left, unit=UNIT),
            _edges(DIAMOND_EDGES, {("A", "B"), ("B", "C"), ("A", "C")}, off="default"),
            styles={"A": "accent", "B": "accent", "C": "accent"})
    s.hull(["A", "B", "C"], "accent")
    s.card_centered(right, ["largest  A, B, C", "tie      B, C, D",
                            ("width    3 − 1 = 2", "accent")])


INDUCED = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0)}
INDUCED_EDGES = [("A", "B"), ("B", "C"), ("C", "D")]


@figure("find_induced_nodes", "substructures", "Find induced nodes",
        "A path A, B, C, D, a tree and so chordal, queried from A to D (both outlined). The "
        "dashed accent arc is the hypothetical edge A to D: it would close the cycle A, B, "
        "C, D, so every node on the induced path between them is returned. B and C are in "
        "the accent color and the card lists [A, B, C, D].")
def find_induced_nodes(s: Scene) -> None:
    g = _ec_graph(list(INDUCED), INDUCED_EDGES)
    assert _lab(g, g.find_induced_nodes("A", "D")) == ["A", "B", "C", "D"]
    left, right = compute(s, "g", 'g.find_induced_nodes("A", "D")')
    pos = place(INDUCED, left, unit=UNIT)
    pos = {k: (x, y + 22) for k, (x, y) in pos.items()}  # room above for the arc
    s.graph(pos, [(a, b, {"style": "accent"}) for a, b in INDUCED_EDGES]
            + [("A", "D", {"style": "ghost", "bend": -46})],
            styles={"A": "soft", "B": "accent", "C": "accent", "D": "soft"})
    s.card_centered(right, [("[A, B, C, D]", "accent"), "A–D would close", "a 4-cycle"],
                    title="induced nodes")



# --- later additions: Steiner variants -------------------------------------

# The steiner_tree graph: A on top, the hub D in the middle, E below.
STEINER = {"A": (1.5, 0), "B": (0, 0.8), "C": (3, 0.8), "D": (1.5, 0.8), "E": (1.5, 1.6)}
STEINER_EDGES = [("A", "B", 3), ("A", "C", 3), ("B", "E", 4), ("C", "E", 4), ("B", "D", 2),
                 ("C", "D", 2), ("D", "E", 1)]


@figure("prize_collecting_steiner_tree", "substructures", "Prize-collecting Steiner tree",
        "The weighted graph from the steiner_tree figure, with a prize p in a badge on "
        "nodes A (1), B (5), C (5), and E (4); edge weights are costs. The tree B–D, D–C, "
        "D–E (accent) collects B, C, and E through the prize-less node D, while A is left "
        "out and faded because its prize of 1 does not pay for any edge costing 3. The card "
        "lists the tree edges and its prize 14 against cost 5.")
def prize_collecting_steiner_tree(s: Scene) -> None:
    left, right = compute(s, "g", "g.prize_collecting_steiner_tree(prizes)")
    tree = {("B", "D"), ("C", "D"), ("D", "E")}
    s.graph(place(STEINER, left, unit=UNIT), _edges(STEINER_EDGES, tree),
            styles={"B": "soft", "C": "soft", "E": "soft", "D": "accent", "A": "muted"})
    prizes = {"A": 1, "B": 5, "C": 5, "E": 4}
    where = {"A": "e", "B": "n", "C": "n", "E": "e"}
    for k, v in prizes.items():
        s.badge(k, f"p={v}", where=where[k], color="ink")
    s.card_centered(right, ["B–D D–C D–E", ("prize 14, cost 5", "accent")], title="tree edges")


@figure("directed_steiner_tree", "substructures", "Directed Steiner tree",
        "A directed version of the steiner_tree graph, rooted at B with terminals C and E "
        "outlined. The edge between C and D now points from C to D, so C cannot be reached "
        "through D: the arborescence B→D→E→C (accent) reaches E through the non-terminal D "
        "and then C from E. The card lists the three edges and their total weight "
        "2 + 1 + 4 = 7.")
def directed_steiner_tree(s: Scene) -> None:
    left, right = compute(s, "g", 'g.directed_steiner_tree("B", ["C", "E"])')
    arcs = [("B", "A", 3), ("A", "C", 3), ("B", "E", 4), ("E", "C", 4), ("B", "D", 2),
            ("C", "D", 2), ("D", "E", 1)]
    tree = {("B", "D"), ("D", "E"), ("E", "C")}
    s.graph(place(STEINER, left, unit=UNIT),
            [(a, b, {"style": "accent" if (a, b) in tree else "muted", "weight": w})
             for a, b, w in arcs],
            directed=True,
            styles={"B": "soft", "C": "soft", "E": "soft", "D": "accent", "A": "muted"})
    s.card_centered(right, ["B→D D→E E→C", ("2 + 1 + 4 = 7", "accent")], title="tree edges")


# --- later additions: motifs -----------------------------------------------

# The two connected 3-node motifs, drawn on one footprint so they compare:
# the path is the triangle minus its base (drawn dashed, absent).
MOTIF = {"a": (0, 1), "b": (1, 0), "c": (2, 1)}


def _motif_rows(s: Scene, right: Box, rows: list[tuple[bool, str, list[str]]]) -> None:
    """One motif shape per row in `right`, its numbers beside it.

    `rows` items are (is_triangle, style of the row, text lines); the first
    line is in card type, any second one is a note under it.
    """
    gw = 2 * MINI_UNIT + 2 * R_SMALL
    tw = max(text_width(t, SIZE_CARD if j == 0 else SIZE_NOTE)
             for _, _, lines in rows for j, t in enumerate(lines))
    gap = 18
    x0 = right.cx - (gw + gap + tw) / 2
    row_h = right.h / len(rows)
    for i, (closed, style, lines) in enumerate(rows):
        cell = Box(x0, right.y0 + i * row_h, x0 + gw, right.y0 + (i + 1) * row_h)
        pos = place(MOTIF, cell, unit=MINI_UNIT, r=R_SMALL, pad=0)
        edge = {} if style == "default" else {"style": style}
        edges = [("a", "b", edge), ("b", "c", edge),
                 ("a", "c", edge if closed else {"style": "removed"})]
        s.graph(pos, edges, styles={k: style for k in MOTIF}, labels=False, r=R_SMALL,
                prefix=f"m{i}.")
        color = "accent" if style == "accent" else "ink"
        n = len(lines)
        for j, t in enumerate(lines):
            y = cell.cy + (j - (n - 1) / 2) * 19
            s.text(x0 + gw + gap, y, t, size=SIZE_CARD if j == 0 else SIZE_NOTE,
                   anchor="start", color=color, weight=600 if j == 0 else 400,
                   opacity=1 if j == 0 else 0.75, role="card")


@figure("motif_census", "substructures", "Motif census",
        "Left: the seven-node graph from the dominating set figures, a pendant 1, triangle "
        "2, 3, 4, a bridge 4–5, and triangle 5, 6, 7. Right: the two connected three-node "
        "motifs with their counts: 2 triangles and 6 induced paths, where a path is a "
        "triangle with its third edge missing (dashed).")
def motif_census(s: Scene) -> None:
    left, right = compute(s, "g", "g.motif_census(size=3)")
    s.graph(place(DOM, left, unit=UNIT), DOM_EDGES)
    _motif_rows(s, right, [(True, "accent", ["triangle  2"]), (False, "accent", ["path      6"])])


@figure("motif_significance", "substructures", "Motif significance",
        "The same seven-node graph. Against 100 degree-preserving rewirings (seed 1), its 2 "
        "triangles are more than expected, z = +1.77 with p = 0.15, shown in the accent "
        "color; its 6 induced paths are fewer than expected, z = -1.77 with p = 1.00, shown "
        "in plain ink.")
def motif_significance(s: Scene) -> None:
    left, right = compute(s, "g", "g.motif_significance(seed=1)")
    s.graph(place(DOM, left, unit=UNIT), DOM_EDGES)
    _motif_rows(s, right, [(True, "accent", ["triangle  2", "z +1.77  p 0.15"]),
                           (False, "default", ["path      6", "z −1.77  p 1.00"])])


# --- isolates --------------------------------------------------------------

# Two rows of four; nodes 4 and 6 touch no edge.
ISO = {str(i + 1): (i % 4, i // 4 * 1.1) for i in range(8)}
ISO_EDGES = [("1", "2"), ("2", "3"), ("1", "5"), ("3", "7"), ("7", "8")]


@figure("isolates", "substructures", "Isolates",
        "Eight nodes with ids 1 to 8, joined by five edges into one chain, except nodes 4 and "
        "6, which touch no edge. Those two isolates are drawn in the accent color, and the "
        "card returns their ids [4, 6].")
def isolates(s: Scene) -> None:
    left, right = compute(s, "g", "g.isolates()")
    s.graph(place(ISO, left, unit=UNIT), ISO_EDGES, styles={"4": "accent", "6": "accent"})
    s.card_centered(right, [("[4, 6]", "accent")], title="isolated node ids")


# --- later additions: triadic census ---------------------------------------

# The triadic_census docstring network: a mutual pair A ⇄ B both pointing at
# C, and the directed cycle C → D → E → C.
TRIAD = {"A": (0, 0), "B": (0, 1.3), "C": (1.2, 0.65), "D": (2.4, 0), "E": (2.4, 1.3)}
TRIAD_EDGES = [("A", "B"), ("B", "A"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"),
               ("E", "C")]
# g.triadic_census() non-zero types in standard order, with the triads behind them
TRIAD_ROWS = [("012", "ADE, BDE"), ("102", "ABD, ABE"), ("021U", "ACE, BCE"),
              ("021C", "ACD, BCD"), ("030C", "CDE"), ("120U", "ABC")]


@figure("triadic_census", "substructures", "Triadic census",
        "A directed graph on five nodes: A and B point at each other and both at C, and C, D, "
        "E form a directed cycle. Each of its 10 triads gets one of 16 types; the table lists "
        "the non-zero counts with the triads behind them: 2 each of 012, 102, 021U, and 021C, "
        "1 cyclic triad 030C (C, D, E, accent color), and 1 120U (A, B, C, teal). Every "
        "other type is 0.")
def triadic_census(s: Scene) -> None:
    left, right = compute(s, "g", "g.triadic_census()")
    color = {"ABC": "g2", "CDE": "accent"}

    def style(a: str, b: str) -> dict:
        kw = {"bend": 7} if (b, a) in TRIAD_EDGES else {}
        for tri, c in color.items():
            if {a, b} <= set(tri):
                kw["style"] = c
        return kw

    s.graph(place(TRIAD, left, unit=UNIT), [(a, b, style(a, b)) for a, b in TRIAD_EDGES],
            directed=True, styles={"A": "g2", "B": "g2", "D": "accent", "E": "accent"})
    rows = [["type", "count", "triads"]] + [[t, str(len(tr.split(","))), tr]
                                           for t, tr in TRIAD_ROWS] + [["total", "10", ""]]
    hl = {(i, j): color[tr] for i, (t, tr) in enumerate(TRIAD_ROWS, 1) if tr in color
          for j in range(3)}
    s.table_centered(right, rows, [50, 50, 76], highlight=hl)


# --- later additions: individual triads ----------------------------------------
#
# The per-triad companions of triadic_census rebuild their docstring examples
# in edgecase and assert the results. triad_type splits the input panel into
# two copies of the graph, one per queried triple (accent and g2, the rest
# muted), with the MAN codes on the card. triads_by_type and all_triads draw
# the graph plain and set one miniature per returned triple in a 2 x 2 grid
# on the right, the triple (and its arcs) in accent, the rest muted, with the
# type or id triple beneath it. all_triads returns ids, so its nodes carry ids.


def _ec_digraph(nodes, edges):
    import edgecase as ec

    g = ec.Graph(directed=True).add_nodes(list(nodes))
    if edges:
        g.add_edges(edges)
    return g


def _triad_edges(edges, on: set, style: str = "accent", off: str = "muted") -> list:
    """Directed edge specs: arcs inside the triple `on` in `style`; reciprocal arcs bowed."""
    out = []
    for a, b in edges:
        kw = {"style": style if {a, b} <= on else off}
        if (b, a) in edges:
            kw["bend"] = 7
        out.append((a, b, kw))
    return out


def _grid4(s: Scene, right: Box, coords, edges, results: list[tuple[set, str]]) -> None:
    """Four miniatures of one graph in a 2 x 2 grid, each highlighting a triple.

    Each grid row holds a miniature with its label just beneath it, the pair
    centered in the row so a clear gap separates the label from the next row.
    """
    unit = 34  # the largest that leaves a clear gap under each label
    xs, ys = [p[0] for p in coords.values()], [p[1] for p in coords.values()]
    gw = (max(xs) - min(xs)) * unit + 2 * R_SMALL
    gh = (max(ys) - min(ys)) * unit + 2 * R_SMALL
    block = gh + 4 + SIZE_CARD  # miniature, gap, label
    pitch = right.h / 2
    for i, (on, label) in enumerate(results):
        cx = right.x0 + right.w * (0.25 + 0.5 * (i % 2))
        top = right.y0 + (i // 2) * pitch + (pitch - block) / 2
        cell = Box(cx - gw / 2, top, cx + gw / 2, top + gh)
        s.graph(place(coords, cell, unit=unit, r=R_SMALL, pad=0), _triad_edges(edges, on),
                directed=True, styles={k: "accent" if k in on else "muted" for k in coords},
                r=R_SMALL, prefix=f"m{i}.")
        s.text(cx, top + gh + 4 + SIZE_CARD / 2, label, size=SIZE_CARD, color="accent",
               weight=600, role="card")


# triad_type docstring: a transitive triple A, B, C and the pair C <-> D.
TT = {"A": (0, 1), "B": (0.9, 0), "C": (1.8, 1), "D": (2.9, 1)}
TT_EDGES = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "D"), ("D", "C")]


@figure("triad_type", "substructures", "Triad type",
        "Two copies of a directed graph on A, B, C, D: A points to B and C, B points to C, and "
        "C and D point at each other. Left copy: the triple A, B, C in the accent color is a "
        "transitive triad of three one-way arcs, type 030T. Right copy: B, C, D in teal has "
        "one mutual pair, one one-way pair and one empty pair, type 111D. The card gives both "
        "MAN codes.")
def triad_type(s: Scene) -> None:
    g = _ec_digraph(TT, TT_EDGES)
    assert g.triad_type(["A", "B", "C"]) == "030T"
    assert g.triad_type(["B", "C", "D"]) == "111D"
    left, right = compute(s, "g", "g.triad_type(nodes)")
    for k, (box, tri, c) in enumerate(((Box(left.x0, left.y0, left.cx, left.y1), "ABC", "accent"),
                                       (Box(left.cx, left.y0, left.x1, left.y1), "BCD", "g2"))):
        s.text(box.cx, box.y0 + 8, f"nodes = [{', '.join(tri)}]", size=SIZE_NOTE, opacity=0.6)
        gb = Box(box.x0, box.y0 + 22, box.x1, box.y1)
        s.graph(place(TT, gb, unit=UNIT * 0.65), _triad_edges(TT_EDGES, set(tri), c),
                directed=True, styles={n: c if n in tri else "muted" for n in TT}, prefix=f"t{k}.")
    s.card_centered(right, [("[A, B, C]  030T", "accent"), "mutual 0, asym 3, null 0",
                            ("[B, C, D]  111D", "g2"), "mutual 1, asym 1, null 1"],
                    title="MAN code")


# triads_by_type docstring: the cycle A -> B -> C -> A with the tail C -> D.
TB = {"A": (0, 0), "B": (1.2, 0), "C": (0.6, 1), "D": (1.8, 1)}
TB_EDGES = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")]


@figure("triads_by_type", "substructures", "Triads by type",
        "Left: a directed graph with the cycle A to B to C to A and the extra arc C to D. "
        "Right: one miniature per triad, its three nodes and arcs in the accent color, "
        "labeled with the type it is grouped under: A, B, C is the cycle 030C; A, B, D is "
        "012; A, C, D is 021D (C points at both others); B, C, D is 021C, a two-step chain.")
def triads_by_type(s: Scene) -> None:
    g = _ec_digraph(TB, TB_EDGES)
    found = {code: ["".join(g.get_label_from_id(n) for n in t) for t in ts]
             for code, ts in g.triads_by_type().items()}
    assert found == {"030C": ["ABC"], "012": ["ABD"], "021D": ["ACD"], "021C": ["BCD"]}
    left, right = compute(s, "g", "g.triads_by_type()")
    s.graph(place(TB, left, unit=UNIT), TB_EDGES, directed=True)
    _grid4(s, right, TB, TB_EDGES, [(set(ts[0]), code) for code, ts in found.items()])


# all_triads docstring: four nodes, so C(4, 3) = 4 triples (ids 1..4).
QUAD = {"1": (0, 0), "2": (1, 0), "3": (0, 1), "4": (1, 1)}


@figure("all_triads", "substructures", "All triads",
        "Left: a directed graph with four nodes, ids 1 to 4, and no edges. Right: one "
        "miniature per triad, its three nodes in the accent color and the fourth faded, "
        "labeled with the id triple returned: (1, 2, 3), (1, 2, 4), (1, 3, 4), and "
        "(2, 3, 4), the 4 ways to choose 3 of 4 nodes.")
def all_triads(s: Scene) -> None:
    g = _ec_digraph(QUAD, [])
    found = g.all_triads()
    assert found == [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
    left, right = compute(s, "g", "g.all_triads()")
    s.graph(place(QUAD, left, unit=UNIT), [], directed=True)
    _grid4(s, right, QUAD, [], [({str(n) for n in t}, f"({', '.join(map(str, t))})")
                                for t in found])


# --- later additions: branchings and arborescences -------------------------
#
# The directed counterparts of the spanning trees. Each figure builds the
# docstring's example with edgecase and draws what the method returns: chosen
# edges `accent`, the rest `muted`, except an edge that beats the chosen
# in-edge of its target and was still dropped (the edge Edmonds' algorithm
# breaks a cycle at), which is `removed`. The card lists the chosen edges and
# their total weight.


def _ec_edges(h) -> list[tuple[str, str, float | None]]:
    """(source label, target label, weight) for every edge of an edgecase graph."""
    out = []
    for eid in h.get_edge_ids(limit=None):
        e = h.get_edge(eid)
        out.append((h.get_label_from_id(e.source), h.get_label_from_id(e.target), e.weight))
    return out


def _id_edges(g, ids) -> list[tuple[str, str, float | None]]:
    """(source label, target label, weight) for the edge ids a matching/cover returns."""
    return [(g.get_label_from_id(e.source), g.get_label_from_id(e.target), e.weight)
            for e in (g.get_edge(i) for i in ids)]


def _num(w: float) -> str:
    """A weight as drawn: integral, with a real minus sign."""
    v = int(w) if float(w).is_integer() else w
    return f"−{-v}" if v < 0 else f"{v}"


def _total(ws: list[float]) -> str:
    """'3 − 2 + 4 = 5' for the weights [3, -2, 4]."""
    out = _num(ws[0])
    for w in ws[1:]:
        out += f" − {_num(-w)}" if w < 0 else f" + {_num(w)}"
    return f"{out} = {_num(sum(ws))}"


def _branching(method: str, coords, arcs, *, maximize: bool):
    """Run `method` on the weighted digraph `arcs`; return (edge specs, chosen, weights)."""
    import edgecase as ec

    g = ec.Graph(directed=True, weighted=True).add_nodes(list(coords))
    g.add_edges([(a, b, float(w)) for a, b, w in arcs])
    chosen = _ec_edges(getattr(g, method)())
    parent_w = {b: w for _, b, w in chosen}
    on = {(a, b) for a, b, _ in chosen}
    better = (lambda x, y: x > y) if maximize else (lambda x, y: x < y)

    def style(a: str, b: str, w: float) -> str:
        if (a, b) in on:
            return "accent"
        if b in parent_w and better(w, parent_w[b]):
            return "removed"
        return "muted"

    specs = []
    for a, b, w in arcs:
        kw = {"style": style(a, b, w), "weight": _num(w)}
        if (b, a) in {(x, y) for x, y, _ in arcs}:
            kw["bend"] = 12
        specs.append((a, b, kw))
    return specs, chosen, [w for _, _, w in chosen]


def _arrows(chosen) -> str:
    return " ".join(f"{a}→{b}" for a, b, _ in chosen)


# maximum_branching docstring: the cycle B -> C -> D -> B with A feeding in.
MAXB = {"A": (0, 0.6), "B": (1.2, 0), "C": (2.4, 0.6), "D": (1.2, 1.2)}
MAXB_ARCS = [("A", "B", 3), ("B", "C", 5), ("C", "D", 4), ("D", "B", 6), ("A", "D", 1)]


@figure("maximum_branching", "substructures", "Maximum branching",
        "A weighted directed graph on four nodes: A points to B (3) and D (1), and B, C, D "
        "form the cycle B→C (5), C→D (4), D→B (6). The heaviest in-edges would close that "
        "cycle, so Edmonds' algorithm drops D→B (dashed) and keeps A→B, B→C, and C→D in "
        "the accent color, a single tree rooted at A; A→D is faded. The card lists the three "
        "edges and their total weight 3 + 5 + 4 = 12.")
def maximum_branching(s: Scene) -> None:
    left, right = compute(s, "g", "g.maximum_branching()")
    specs, chosen, ws = _branching("maximum_branching", MAXB, MAXB_ARCS, maximize=True)
    s.graph(place(MAXB, left, unit=UNIT), specs, directed=True)
    s.card_centered(right, [_arrows(chosen), (_total(ws), "accent")], title="branching edges")


# minimum_branching docstring: only negative weights can lower the total.
MINB = {"A": (0.7, 0), "B": (0, 1.2), "C": (1.4, 1.2)}
MINB_ARCS = [("A", "B", -2), ("C", "B", -3), ("A", "C", -1), ("B", "C", 4)]


@figure("minimum_branching", "substructures", "Minimum branching",
        "A weighted directed triangle: A→B (−2), A→C (−1), C→B (−3), and B→C (4). Only "
        "negative edges lower the total, and each node may keep one parent, so the minimum "
        "branching takes C→B and A→C, in the accent color, while A→B and B→C are faded. The "
        "card lists the two edges and their total weight −3 − 1 = −4.")
def minimum_branching(s: Scene) -> None:
    left, right = compute(s, "g", "g.minimum_branching()")
    specs, chosen, ws = _branching("minimum_branching", MINB, MINB_ARCS, maximize=False)
    s.graph(place(MINB, left, unit=UNIT), specs, directed=True)
    s.card_centered(right, [_arrows(chosen), (_total(ws), "accent")], title="branching edges")


# maximum_spanning_arborescence docstring: C has only negative in-edges.
MAXA = {"A": (0, 0), "B": (1.4, 0), "C": (1.4, 1.2), "D": (2.8, 1.2)}
MAXA_ARCS = [("A", "B", 3), ("B", "C", -2), ("A", "C", -5), ("C", "D", 4)]


@figure("maximum_spanning_arborescence", "substructures", "Maximum spanning arborescence",
        "A weighted directed graph: A→B (3), B→C (−2), A→C (−5), and C→D (4). Node C can only "
        "be reached through a negative edge, and a spanning arborescence must reach every "
        "node, so it takes the less costly B→C: the tree A→B, B→C, C→D in the accent color, "
        "total 3 − 2 + 4 = 5, with A→C faded. The card contrasts it with the maximum "
        "branching, which leaves C without a parent and keeps only A→B and C→D, total 7.")
def maximum_spanning_arborescence(s: Scene) -> None:
    left, right = compute(s, "g", "g.maximum_spanning_arborescence()")
    specs, chosen, ws = _branching("maximum_spanning_arborescence", MAXA, MAXA_ARCS,
                                   maximize=True)
    s.graph(place(MAXA, left, unit=UNIT), specs, directed=True)
    _, br, bws = _branching("maximum_branching", MAXA, MAXA_ARCS, maximize=True)
    s.card_centered(right, [_arrows(chosen), (_total(ws), "accent"),
                            f"branching {_total(bws).split(' = ')[1]}: {_arrows(br)}"],
                    title="arborescence edges")


# minimum_spanning_arborescence docstring: one-way links between relay stations.
MINA = {"A": (0, 0.6), "B": (1.3, 0), "C": (1.3, 1.2), "D": (2.6, 0.6)}
MINA_ARCS = [("A", "B", 4), ("A", "C", 1), ("C", "B", 2), ("B", "D", 3), ("C", "D", 6),
             ("D", "C", 1)]


@figure("minimum_spanning_arborescence", "substructures", "Minimum spanning arborescence",
        "One-way links between four relay stations, weighted by cost: A→B (4), A→C (1), "
        "C→B (2), B→D (3), C→D (6), and D→C (1). The cheapest tree reaching every station "
        "from the root A, outlined in the accent color, uses A→C, C→B, and B→D, in the "
        "accent color; the other links are faded. The card lists the three edges and their "
        "total cost 1 + 2 + 3 = 6.")
def minimum_spanning_arborescence(s: Scene) -> None:
    left, right = compute(s, "g", "g.minimum_spanning_arborescence()")
    specs, chosen, ws = _branching("minimum_spanning_arborescence", MINA, MINA_ARCS,
                                   maximize=False)
    root = next(k for k in MINA if k not in {b for _, b, _ in chosen})
    s.graph(place(MINA, left, unit=UNIT), specs, directed=True, styles={root: "soft"})
    s.card_centered(right, [_arrows(chosen), (_total(ws), "accent")],
                    title=f"arborescence edges, root {root}")


# --- later additions: counting and sampling spanning trees -----------------

SQUARE = {"A": (0, 0), "B": (1, 0), "C": (1, 1), "D": (0, 1)}
SQUARE_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
DIAG = ("A", "C")
DOT = 5  # unlabeled miniature node radius


@figure("number_of_spanning_trees", "substructures", "Number of spanning trees",
        "Left: a square A, B, C, D with the diagonal A–C in the accent color. Right: its 8 "
        "spanning trees as miniatures in two rows. The top row holds the 4 trees of the "
        "plain square, each dropping one side; the bottom row holds the 4 trees that use the "
        "diagonal. The heading gives the count, 8, and a note under the rows the sum 4 + 4 = 8.")
def number_of_spanning_trees(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.number_of_spanning_trees()")
    g = ec.Graph().add_nodes(list(SQUARE))
    g.add_edges(SQUARE_EDGES)
    before = g.number_of_spanning_trees()
    g.add_edge(*DIAG)
    total = g.number_of_spanning_trees()
    trees = [{_key(a, b) for a, b, _ in _ec_edges(t)} for t in g.all_spanning_trees()]
    rows = [[t for t in trees if DIAG not in t], [t for t in trees if DIAG in t]]
    assert len(rows[0]) == before and len(trees) == total

    s.graph(place(SQUARE, left, unit=UNIT),
            SQUARE_EDGES + [(*DIAG, {"style": "accent"})])

    side, gap, n = 24, 12, max(len(r) for r in rows)
    note_w = max(text_width(t, SIZE_NOTE) for t in ("no A–C", "with A–C"))
    row_w = n * (side + 2 * DOT) + (n - 1) * gap + 10 + note_w
    x0 = right.cx - row_w / 2
    s.text(right.cx, right.y0 + 16, f"{total} spanning trees", size=SIZE_CARD,
           color="accent", weight=600, role="card")
    all_edges = SQUARE_EDGES + [DIAG]
    for i, (row, note) in enumerate(zip(rows, ("no A–C", "with A–C"))):
        cy = right.y0 + 62 + i * 58
        for j, t in enumerate(row):
            cx0 = x0 + j * (side + 2 * DOT + gap) + DOT
            pos = {k: (cx0 + x * side, cy - side / 2 + y * side) for k, (x, y) in SQUARE.items()}
            s.graph(pos, [(a, b, {"style": "accent" if _key(a, b) in t else "removed"})
                          for a, b in all_edges],
                    styles={k: "accent" for k in SQUARE}, labels=False, r=DOT,
                    prefix=f"t{i}{j}.")
        s.text(x0 + row_w - note_w, cy, note, size=SIZE_NOTE, anchor="start", opacity=0.75)
    s.text(right.cx, right.y1 - 10, f"{before} + {total - before} = {total}", size=SIZE_NOTE,
           opacity=0.75)


GRID_N = 4
GRID_SEED = 7


@figure("random_spanning_tree", "substructures", "Random spanning tree",
        "Left: a 4 by 4 grid graph, 16 nodes and 24 edges. Right: the spanning tree drawn "
        "with seed 7, its 15 edges in the accent color and the 9 unused grid edges dashed. "
        "Every node is reached by exactly one route, so the tree reads as a maze.")
def random_spanning_tree(s: Scene) -> None:
    import edgecase as ec

    left, right = transform(s, "g", f"g.random_spanning_tree(seed={GRID_SEED})")
    grid = ec.Graph().add_grid_2d(GRID_N, GRID_N)

    def coord(label: str) -> tuple[float, float]:
        i, j = (int(v) for v in label.strip("()").split(","))
        return (j, i)

    coords = {lab: coord(lab) for lab in (grid.get_label_from_id(n)
                                          for n in grid.get_node_ids(limit=None))}
    edges = [(a, b) for a, b, _ in _ec_edges(grid)]
    tree = {_key(a, b) for a, b, _ in _ec_edges(grid.random_spanning_tree(seed=GRID_SEED))}
    assert len(tree) == len(coords) - 1
    r = 8
    s.graph(place(coords, left, unit=48, r=r), edges, labels=False, r=r, prefix="l.")
    s.graph(place(coords, right, unit=48, r=r),
            [(a, b, {"style": "accent" if _key(a, b) in tree else "removed"}) for a, b in edges],
            labels=False, r=r, prefix="r.")


# --- later additions: Prüfer sequences -------------------------------------

# A tree on ids 1..6 whose code [5, 1, 5, 6] shows a neighbor (1) turning
# into the next smallest leaf.
PRUFER = {"3": (0, 0), "1": (1, 0.7), "5": (2, 0), "2": (2, 1.2), "6": (3, 0.7), "4": (4, 0)}
PRUFER_EDGES = [("2", "5"), ("1", "3"), ("1", "5"), ("4", "6"), ("5", "6")]


def _prufer_encode(edges, order: list[str]) -> list[tuple[str, str]]:
    """(removed leaf, its neighbor) per step, removing the smallest leaf each time."""
    adj = {k: set() for k in order}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    steps = []
    for _ in range(len(order) - 2):
        leaf = next(k for k in order if k in adj and len(adj[k]) == 1)
        (nb,) = adj.pop(leaf)
        adj[nb].discard(leaf)
        steps.append((leaf, nb))
    return steps


@figure("to_prufer_sequence", "substructures", "To Prüfer sequence",
        "A tree on node ids 1 to 6: 3–1, 1–5, 5–2, 5–6, and 6–4. Repeatedly removing the "
        "smallest leaf and writing down its neighbor removes 2, 3, 1, and 4 in that order "
        "(filled order badges), along the four edges in the accent color; removing 3 turns "
        "1 into a leaf, so it goes next. The card shows the removed leaves and the returned "
        "sequence [5, 1, 5, 6].")
def to_prufer_sequence(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.to_prufer_sequence()")
    order = sorted(PRUFER, key=int)
    g = ec.Graph().add_nodes(order)  # labels "1".."6" get ids 1..6
    g.add_edges(PRUFER_EDGES)
    seq = [g.get_label_from_id(n) for n in g.to_prufer_sequence()]
    steps = _prufer_encode(PRUFER_EDGES, order)
    assert seq == [nb for _, nb in steps]
    removed = {_key(a, b) for a, b in steps}
    s.graph(place(PRUFER, left, unit=UNIT),
            [(a, b, {"style": "accent"}) if _key(a, b) in removed else (a, b)
             for a, b in PRUFER_EDGES])
    for i, (leaf, _) in enumerate(steps, 1):
        s.badge(leaf, str(i), filled=True)
    s.card_centered(right, ["leaves  " + "  ".join(leaf for leaf, _ in steps),
                            ("result  [" + ", ".join(seq) + "]", "accent")],
                    title="remove leaf, write neighbor")


PRUFER_NAMES = ["A", "B", "C", "D", "E", "F"]
PRUFER_SEQ = ["D", "D", "E", "E"]
NAMED = {"A": (0, 0), "B": (0, 1.2), "D": (1, 0.6), "E": (2.2, 0.6), "C": (3.2, 0),
         "F": (3.2, 1.2)}


def _prufer_decode(seq: list[str], order: list[str]) -> list[tuple[str, str]]:
    """Edges in the order the decoder adds them; the last joins the final two nodes."""
    deg = {k: 1 + seq.count(k) for k in order}
    out = []
    for x in seq:
        leaf = next(k for k in order if deg[k] == 1)
        out.append((leaf, x))
        deg[leaf] -= 1
        deg[x] -= 1
    out.append(tuple(k for k in order if deg[k] == 1))
    return out


@figure("from_prufer_sequence", "substructures", "From Prüfer sequence",
        "Left: a card with the node names A to F, smallest first, and the sequence D, D, E, "
        "E. Each entry joins the smallest remaining leaf to it: A–D, B–D, C–E, and D–E; the "
        "last two nodes left, E and F, are then joined. Right: the decoded six-node tree, "
        "with D and E, the nodes named in the sequence, and the four edges added for its "
        "entries in the accent color, and the final edge E–F plain.")
def from_prufer_sequence(s: Scene) -> None:
    import edgecase as ec

    left, right = transform(s, "names, seq", "Graph.from_prufer_sequence(seq, nodes=names)")
    t = ec.Graph.from_prufer_sequence(PRUFER_SEQ, nodes=PRUFER_NAMES)
    got = {_key(a, b) for a, b, _ in _ec_edges(t)}
    steps = _prufer_decode(PRUFER_SEQ, PRUFER_NAMES)
    assert got == {_key(a, b) for a, b in steps}
    s.card_centered(left, ["names  " + " ".join(PRUFER_NAMES),
                           ("seq    " + " ".join(PRUFER_SEQ), "accent"),
                           "joins  " + " ".join(f"{a}–{b}" for a, b in steps[:-1]),
                           f"last   {steps[-1][0]}–{steps[-1][1]}"])
    entries = {_key(a, b) for a, b in steps[:-1]}
    s.graph(place(NAMED, right, unit=UNIT),
            [(a, b, {"style": "accent"}) if _key(a, b) in entries else (a, b)
             for a, b in sorted(got)],
            styles={k: "accent" for k in set(PRUFER_SEQ)})


# --- later additions: exact matchings and covers ---------------------------
#
# Each figure rebuilds its docstring example in edgecase and asserts the
# returned structure, so a library change breaks the build instead of the
# picture. Bipartite graphs are drawn as two columns (left side / right side)
# or two rows (top / bottom); matched edges are `accent`, the rest `muted`.


def _labels(g, ids) -> list[str]:
    return [g.get_label_from_id(n) for n in ids]


# min_weight_matching docstring: four drivers paired into two teams.
TEAMS = {"A": (0, 0), "B": (1.6, 0), "C": (0, 1.2), "D": (1.6, 1.2)}
TEAMS_EDGES = [("A", "B", 2), ("C", "D", 3), ("A", "C", 1), ("B", "D", 6), ("A", "D", 4)]


@figure("min_weight_matching", "substructures", "Minimum weight matching",
        "Four drivers A, B, C, D with the extra driving time of each possible pair as edge "
        "weights: A–B (2), C–D (3), A–C (1), B–D (6), and A–D (4). Among the matchings that "
        "pair everyone, the cheapest is A–B and C–D, in the accent color, total 2 + 3 = 5; "
        "the other edges are faded. The card notes that taking the cheapest edge A–C (1) "
        "would force B–D (6), a total of 7.")
def min_weight_matching(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.min_weight_matching()")
    g = ec.Graph(weighted=True).add_nodes(list(TEAMS))
    g.add_edges([(a, b, float(w)) for a, b, w in TEAMS_EDGES])
    got = _id_edges(g, g.min_weight_matching())
    on = {_key(a, b) for a, b, _ in got}
    assert on == {("A", "B"), ("C", "D")}
    s.graph(place(TEAMS, left, unit=UNIT), _edges(TEAMS_EDGES, on),
            styles={k: "accent" for k in TEAMS})
    s.card_centered(right, [" ".join(f"{a}–{b}" for a, b in sorted(on)),
                            (_total([w for _, _, w in sorted(got)]), "accent"),
                            "A–C B–D  1 + 6 = 7"],
                    title="teams")


# bipartite_maximum_matching docstring: applicants (left) and jobs (right).
JOBS = {"Ann": (0, 0), "Bob": (0, 1), "Cy": (0, 2), "Dev": (2.4, 0), "Ops": (2.4, 1),
        "QA": (2.4, 2)}
JOBS_EDGES = [("Ann", "Dev"), ("Bob", "Dev"), ("Cy", "Ops"), ("Cy", "QA")]


@figure("bipartite_maximum_matching", "substructures", "Bipartite maximum matching",
        "Three applicants, Ann, Bob, and Cy, in a column on the left and three jobs, Dev, Ops, "
        "and QA, on the right. Ann and Bob can only do Dev, and Cy can do Ops or QA, so at "
        "most two pairs form: Ann–Dev and Cy–Ops, in the accent color with their endpoints. "
        "Bob and QA stay unmatched. The card lists the two pairs and the size 2.")
def bipartite_maximum_matching(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.bipartite_maximum_matching()")
    g = ec.Graph().add_nodes(list(JOBS))
    g.add_edges(JOBS_EDGES)
    got = _id_edges(g, g.bipartite_maximum_matching(top_nodes=["Ann", "Bob", "Cy"]))
    assert {(a, b) for a, b, _ in got} == {("Ann", "Dev"), ("Cy", "Ops")}
    assert len(g.bipartite_maximum_matching()) == 2
    on = {_key(a, b) for a, b, _ in got}
    s.graph(place(JOBS, left, unit=UNIT), _edges(JOBS_EDGES, on),
            styles={k: "accent" for a, b, _ in got for k in (a, b)})
    s.card_centered(right, [f"{a}–{b}" for a, b, _ in got] + [(f"size  {len(got)}", "accent")],
                    title="matched pairs")


# bipartite_min_weight_full_matching docstring: couriers and delivery times.
COURIERS = ["Ana", "Ben", "Cho"]
DROPS = ["d1", "d2", "d3"]
COST = {"Ana": (2, 3, 8), "Ben": (3, 7, 9), "Cho": (6, 4, 5)}


@figure("bipartite_min_weight_full_matching", "substructures",
        "Bipartite minimum weight full matching",
        "Three couriers, Ana, Ben, and Cho, in a column on the left, every one linked to three "
        "deliveries d1, d2, d3 on the right. The cheapest assignment, in the accent color, is "
        "Ana–d2 (3), Ben–d1 (3), and Cho–d3 (5); the other six links are faded. The table "
        "of travel times highlights those three cells and gives the total 3 + 3 + 5 = 11, "
        "even though Ana's nearest delivery is d1 (2).")
def bipartite_min_weight_full_matching(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.bipartite_min_weight_full_matching()")
    g = ec.Graph(weighted=True).add_nodes(COURIERS + DROPS)
    arcs = [(c, d, COST[c][j]) for c in COURIERS for j, d in enumerate(DROPS)]
    g.add_edges([(a, b, float(w)) for a, b, w in arcs])
    got = _id_edges(g, g.bipartite_min_weight_full_matching(top_nodes=COURIERS))
    pick = {a: b for a, b, _ in got}
    assert pick == {"Ana": "d2", "Ben": "d1", "Cho": "d3"}
    coords = {c: (0, i) for i, c in enumerate(COURIERS)} | {d: (2.4, i) for i, d in
                                                            enumerate(DROPS)}
    # nine crossing links leave no room for weight labels; the table carries them
    specs = [(a, b, {"style": "accent" if pick[a] == b else "muted"}) for a, b, _ in arcs]
    s.graph(place(coords, left, unit=UNIT), specs, styles={k: "accent" for k in coords})
    rows = [[""] + DROPS] + [[c] + [str(v) for v in COST[c]] for c in COURIERS]
    hi = {(i + 1, DROPS.index(pick[c]) + 1): "accent" for i, c in enumerate(COURIERS)}
    tb = s.table(right.cx - 88, right.cy - 52, rows, [56, 40, 40, 40], highlight=hi)
    s.text(right.cx, tb[3] + 20, _total([COST[c][DROPS.index(pick[c])] for c in COURIERS]),
           size=SIZE_CARD, color="accent", weight=600, role="card")


# bipartite_vertex_cover docstring: top row A, B, C; bottom row D, E.
KONIG = {"A": (0, 0), "B": (1.2, 0), "C": (2.4, 0), "D": (0.6, 1.3), "E": (1.8, 1.3)}
KONIG_EDGES = [("A", "D"), ("A", "E"), ("B", "E"), ("C", "E")]


@figure("bipartite_vertex_cover", "substructures", "Bipartite vertex cover",
        "A bipartite graph with A, B, C on the top row and D, E on the bottom row, and edges "
        "A–D, A–E, B–E, and C–E. The minimum vertex cover is A and E, in the accent color: "
        "every edge touches one of them. A maximum matching, A–D and B–E in a second color, "
        "has the same size, as König's theorem says. The card shows the cover [A, E] and "
        "the matching, both of size 2.")
def bipartite_vertex_cover(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.bipartite_vertex_cover()")
    g = ec.Graph().add_nodes(list(KONIG))
    g.add_edges(KONIG_EDGES)
    cover = _labels(g, g.bipartite_vertex_cover())
    match = {_key(a, b) for a, b, _ in _id_edges(g, g.bipartite_maximum_matching())}
    assert cover == ["A", "E"] and match == {("A", "D"), ("B", "E")}
    s.graph(place(KONIG, left, unit=UNIT),
            [(a, b, {"style": "g2"}) if (a, b) in match else (a, b) for a, b in KONIG_EDGES],
            styles={k: "accent" for k in cover})
    s.card_centered(right, [("cover     [" + ", ".join(cover) + "]", "accent"),
                            ("matching  " + " ".join(f"{a}–{b}" for a, b in sorted(match)),
                             "g2"),
                            f"size      {len(cover)} = {len(match)}"])


# min_weighted_vertex_cover docstring: a star whose hub costs 10.
HUBSTAR = {"a": (0, 0), "b": (2, 0), "hub": (1, 0.6), "c": (0, 1.2), "d": (2, 1.2)}


@figure("min_weighted_vertex_cover", "substructures", "Minimum weighted vertex cover",
        "A star: a hub joined to four leaves a, b, c, d, with node costs in badges, 10 for the "
        "hub and 1 for each leaf. Covering every edge with the hub alone costs 10, so the "
        "method takes the four leaves instead, in the accent color. The card shows the cover "
        "[a, b, c, d] and its cost 1 + 1 + 1 + 1 = 4, against 10 for the hub.")
def min_weighted_vertex_cover(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", 'g.min_weighted_vertex_cover("cost")')
    leaves = ["a", "b", "c", "d"]
    g = ec.Graph().add_nodes(["hub"] + leaves)
    g.add_edges([("hub", x) for x in leaves])
    g.set_node_attrs("cost", {g.get_id_from_label("hub"): 10.0})
    cover = _labels(g, g.min_weighted_vertex_cover(weight_attr="cost"))
    assert cover == leaves
    assert _labels(g, g.min_weighted_vertex_cover()) == ["hub"]
    s.graph(place(HUBSTAR, left, unit=UNIT), [("hub", x) for x in leaves],
            styles={k: "accent" for k in cover})
    s.badge("hub", "10", where="n", color="ink")
    for k in leaves:
        s.badge(k, "1", where="n" if HUBSTAR[k][1] == 0 else "s", color="ink")
    s.card_centered(right, [("cover  [" + ", ".join(cover) + "]", "accent"),
                            "cost   " + " + ".join("1" for _ in cover) + f" = {len(cover)}",
                            "hub    10"])


# min_edge_cover docstring: a five-node path.
PATH5 = {k: (i, 0) for i, k in enumerate("ABCDE")}
PATH5_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]


@figure("min_edge_cover", "substructures", "Minimum edge cover",
        "A path of five nodes, A to E. A maximum matching covers four nodes with two edges, "
        "and one more edge is needed for the fifth, so the minimum edge cover has three "
        "edges: A–B, B–C, and D–E, in the accent color; C–D is faded. Every node touches a "
        "chosen edge. The card lists the three edges and the count 5 − 2 = 3.")
def min_edge_cover(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.min_edge_cover()")
    g = ec.Graph().add_nodes(list(PATH5))
    g.add_edges(PATH5_EDGES)
    got = {_key(a, b) for a, b, _ in _id_edges(g, g.min_edge_cover())}
    nu = len(g.max_weight_matching())
    assert got == {("A", "B"), ("B", "C"), ("D", "E")} and len(got) == len(PATH5) - nu
    s.graph(place(PATH5, left, unit=UNIT), _edges(PATH5_EDGES, got),
            styles={k: "accent" for k in PATH5})
    s.card_centered(right, [("edges  " + " ".join(f"{a}–{b}" for a, b in sorted(got)),
                             "accent"),
                            f"size   {len(PATH5)} − {nu} = {len(got)}"])


# densest_subgraph docstring: a 4-clique of friends with a chain D–E–F.
DENSE = {"A": (0, 0.55), "B": (0.8, 0), "C": (0.8, 1.1), "D": (1.6, 0.55), "E": (2.6, 0.55),
         "F": (3.6, 0.55)}
DENSE_EDGES = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"),
               ("D", "E"), ("E", "F")]


@figure("densest_subgraph", "substructures", "Densest subgraph",
        "A clique of four nodes A, B, C, D, every pair joined, with a chain D–E–F hanging "
        "off D. The densest subgraph is the clique, in the accent color: 6 edges on 4 nodes, "
        "density 6 / 4 = 1.5. The chain dilutes it, since the whole graph has 8 edges on 6 "
        "nodes, density 8 / 6 = 1.33. The card shows the nodes [A, B, C, D] and both "
        "densities.")
def densest_subgraph(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.densest_subgraph()")
    g = ec.Graph().add_nodes(list(DENSE))
    g.add_edges(DENSE_EDGES)
    density, ids = g.densest_subgraph()
    nodes = _labels(g, ids)
    on = {_key(a, b) for a, b in DENSE_EDGES if a in nodes and b in nodes}
    assert nodes == ["A", "B", "C", "D"] and density == len(on) / len(nodes) == 1.5
    m, n = g.count_edges(), g.count_nodes()
    assert (m, n) == (8, 6)
    s.graph(place(DENSE, left, unit=UNIT), _edges(DENSE_EDGES, on, off="default"),
            styles={k: "accent" for k in nodes})
    s.card_centered(right, [("nodes    [" + ", ".join(nodes) + "]", "accent"),
                            (f"density  {len(on)} / {len(nodes)} = {density:g}", "accent"),
                            f"whole    {m} / {n} = {m / n:.2f}"],
                    title="edges / nodes")


# --- pattern occurrences ---------------------------------------------------

# subgraph_isomorphisms docstring: feed-forward loops (A -> B -> C plus A -> C)
# in a four-node directed network. The host zigzags so each occurrence is one
# of its two triangles; the pattern is drawn on the first triangle's footprint.
FFL = {"A": (0, 1), "B": (1, 0), "C": (2, 1)}
FFL_EDGES = [("A", "B"), ("B", "C"), ("A", "C")]
FFL_HOST = {"x": (0, 1), "y": (1, 0), "z": (2, 1), "w": (3, 0)}
FFL_HOST_EDGES = [("x", "y"), ("y", "z"), ("x", "z"), ("z", "w"), ("y", "w")]


@figure("subgraph_isomorphisms", "substructures", "Subgraph isomorphisms",
        "Left: pattern p, a feed-forward loop where A points to B and C and B points to C, "
        "beside graph g, a four-node directed network with edges x→y, y→z, x→z, z→w, and "
        "y→w. Right: two miniature copies of g, each highlighting one occurrence of the "
        "pattern in the accent color, listed beside it: A→x B→y C→z, and A→y B→z C→w.")
def subgraph_isomorphisms(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph(directed=True).add_nodes(list(FFL_HOST))
    g.add_edges(FFL_HOST_EDGES)
    p = ec.Graph(directed=True).add_nodes(list(FFL))
    p.add_edges(FFL_EDGES)
    found = [{p.get_label_from_id(a): g.get_label_from_id(b) for a, b in m.items()}
             for m in g.subgraph_isomorphisms(p)]
    assert found == [{"A": "x", "B": "y", "C": "z"}, {"A": "y", "B": "z", "C": "w"}]
    left, right = compute(s, "p", "g.subgraph_isomorphisms(p)")
    pbox = Box(left.x0, left.y0, left.x0 + 150, left.y1)
    gbox = Box(pbox.x1, left.y0, left.x1, left.y1)
    cap = next(t for t in s.texts if t.role == "caption" and t.text == "p")
    cap.x = pbox.cx
    s.caption(gbox.cx, "g", max_w=gbox.w)
    s.graph(place(FFL, pbox, unit=UNIT * 0.6), FFL_EDGES, directed=True,
            styles=dict.fromkeys(FFL, "soft"), prefix="p.")
    s.graph(place(FFL_HOST, gbox, unit=UNIT * 0.8), FFL_HOST_EDGES, directed=True,
            prefix="g.")
    texts = [" ".join(f"{a}→{b}" for a, b in m.items()) for m in found]
    xs = [c[0] for c in FFL_HOST.values()]
    gw = (max(xs) - min(xs)) * MINI_UNIT + 2 * R_SMALL
    tw = max(text_width(t, SIZE_CARD) for t in texts)
    gap = 18
    x0 = right.cx - (gw + gap + tw) / 2
    row_h = right.h / len(found)
    for i, (m, label) in enumerate(zip(found, texts)):
        cell = Box(x0, right.y0 + i * row_h, x0 + gw, right.y0 + (i + 1) * row_h)
        hit = set(m.values())
        on = {(m[a], m[b]) for a, b in FFL_EDGES}
        s.graph(place(FFL_HOST, cell, unit=MINI_UNIT, r=R_SMALL, pad=0),
                [(a, b, {"style": "accent" if (a, b) in on else "muted"})
                 for a, b in FFL_HOST_EDGES],
                directed=True, styles={k: "accent" if k in hit else "muted" for k in FFL_HOST},
                r=R_SMALL, prefix=f"m{i}.")
        s.text(x0 + gw + gap, cell.cy, label, size=SIZE_CARD, anchor="start", color="accent",
               weight=600, role="card")
