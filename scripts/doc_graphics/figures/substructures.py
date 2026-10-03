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
"""

from core import (R_SMALL, SIZE_CARD, SIZE_NOTE, Box, Scene, compute, figure,
                  place, text_width)

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
BULL_W = {("2", "3"): 3, ("1", "2"): 5, ("3", "4"): 4, ("2", "5"): 1, ("3", "5"): 2}

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


@figure("max_weight_matching", "substructures", "Maximum weight matching",
        "The same five-node graph with edge weights. Taking edges by weight, the matching "
        "picks 1–2 (5) and 3–4 (4), shown in the accent color with their endpoints. The "
        "card shows the returned edge ids 2 and 3, the pairs, and the total weight 5 + 4 = 9.")
def max_weight_matching(s: Scene) -> None:
    left, right = compute(s, "g", "g.max_weight_matching()")
    s.graph(place(BULL, left, unit=UNIT),
            _edges(BULL_EDGES, {("1", "2"), ("3", "4")}, weights=BULL_W),
            styles={k: "accent" for k in "1234"})
    s.card_centered(right, [("edge ids  [2, 3]", "accent"), "pairs     1–2, 3–4",
                            "weight    5 + 4 = 9"])


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
        "Nodes 1, 3, and 7 form the dominating set (accent); an accent edge links each "
        "other node to a set member that dominates it. The card shows the result "
        "[1, 3, 7], size 3.")
def dominating_set(s: Scene) -> None:
    left, right = compute(s, "g", "g.dominating_set()")
    # one dominating edge per node outside the set
    witness = {("1", "2"), ("3", "4"), ("5", "7"), ("6", "7")}
    s.graph(place(DOM, left, unit=UNIT), _edges(DOM_EDGES, witness),
            styles={"1": "accent", "3": "accent", "7": "accent"})
    s.card_centered(right, [("nodes  [1, 3, 7]", "accent"), "size   3"])


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
