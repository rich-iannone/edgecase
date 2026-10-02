"""Communities and partitions: every node assigned to a group.

Template: `compute`. Left: the graph with each group behind a soft hull and
its nodes in the group color (group 1 `accent`, then `g2`, `g3`, `g4`).
Edges inside a group take the group color; edges between groups stay ink.
Right: a card listing the groups in the same colors.

Shared layouts so siblings can be compared directly:

- `CMTY`: two dense clusters joined by a bridge, used by all three community
  detection methods (they agree on it; numbering is what edgecase returns).
- `COMP`: a directed graph whose strong and weak components differ, used by
  all five component methods.
- `CORE`: a K4 nucleus, a 2-shell ring and one pendant (nested cores), used by
  coreness, the k-core family, k-truss, and cohesive blocks. The k-core family
  returns subgraphs, so those figures use `transform`: kept elements `accent`
  inside a hull, dropped ones `removed`, same layout in both panels.
"""

from core import Box, Scene, compute, figure, place, transform

UNIT = 80  # one density for the whole family

GROUPS = ("accent", "g2", "g3", "g4")


def _partition(s: Scene, pos, edges, groups, *, directed=False, prefix="", hulls=True,
               edge_kw=None) -> None:
    """Draw a graph whose nodes are split into `groups` (list of node lists).

    Group i gets GROUPS[i]; edges with both ends in one group take its color.
    `edge_kw` adds per-edge options keyed by (a, b).
    """
    grp = {k: GROUPS[i] for i, g in enumerate(groups) for k in g}
    edge_kw = edge_kw or {}
    out = []
    for a, b in edges:
        kw = {"style": grp[a]} if grp.get(a) and grp.get(a) == grp.get(b) else {}
        kw.update(edge_kw.get((a, b), {}))
        out.append((a, b, kw))
    s.graph(pos, out, styles=grp, directed=directed, prefix=prefix)
    if hulls:
        for i, g in enumerate(groups):
            s.hull([prefix + k for k in g], GROUPS[i])


# --- community detection ---------------------------------------------------

CMTY = {"A": (0, 0), "B": (0, 1.2), "C": (1, 0.6), "D": (2.2, 0.6), "E": (3, 0),
        "F": (3, 1.2), "G": (3.8, 0.6)}
CMTY_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("D", "F"),
              ("E", "F"), ("E", "G"), ("F", "G")]
# All three methods return {D, E, F, G} -> 0 and {A, B, C} -> 1 on this graph.
CMTY_GROUPS = [list("DEFG"), list("ABC")]
CMTY_CARD = [("0 → D, E, F, G", "accent"), ("1 → A, B, C", "g2")]


def _cmty(s: Scene, call: str, **kw) -> None:
    left, right = compute(s, "g", call)
    _partition(s, place(CMTY, left, unit=UNIT), CMTY_EDGES, CMTY_GROUPS, **kw)
    s.card_centered(right, CMTY_CARD, title="community")


@figure("cmty_louvain", "communities", "Louvain communities",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the single edge C to D. Louvain puts each cluster in its own community, drawn as "
        "tinted regions in teal and the accent color. The card lists community 0 as D, E, F, G "
        "and community 1 as A, B, C.")
def cmty_louvain(s: Scene) -> None:
    _cmty(s, "g.cmty_louvain()")


@figure("cmty_walktrap", "communities", "Walktrap communities",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the single edge C to D. Short random walks stay trapped inside each cluster, so "
        "Walktrap returns the two clusters as communities, drawn as tinted regions in teal and "
        "the accent color. The card lists community 0 as D, E, F, G and community 1 as A, B, C.")
def cmty_walktrap(s: Scene) -> None:
    _cmty(s, "g.cmty_walktrap()")


@figure("cmty_edge_btwns", "communities", "Edge-betweenness communities",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the bridge C to D, drawn dashed because every path between the clusters crosses it, "
        "so it has the highest betweenness and is the first edge Girvan-Newman cuts. The "
        "remaining pieces are the communities: 0 is D, E, F, G and 1 is A, B, C.")
def cmty_edge_btwns(s: Scene) -> None:
    _cmty(s, "g.cmty_edge_btwns()", edge_kw={("C", "D"): {"style": "removed"}})


# --- connected components (directed) ---------------------------------------

COMP = {"A": (0, 0), "B": (0, 1.3), "C": (1, 0.65), "D": (2, 0.65), "E": (3, 0.65),
        "F": (3.9, 0.05), "G": (3.9, 1.25)}
# A -> B -> C -> A is a cycle; C -> D one way; D <-> E reciprocal; F -> G one way.
COMP_EDGES = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "D"),
              ("F", "G")]
COMP_BEND = {("D", "E"): {"bend": 7}, ("E", "D"): {"bend": 7}}
SCC = [list("ABC"), list("DE"), ["F"], ["G"]]
WCC = [list("ABCDE"), list("FG")]


def _comp(s: Scene, call: str, groups, **kw) -> Box:
    left, right = compute(s, "g", call)
    _partition(s, place(COMP, left, unit=UNIT), COMP_EDGES, groups, directed=True,
               edge_kw=COMP_BEND, **kw)
    return right


@figure("strongly_connected_components", "communities", "Strongly connected components",
        "A directed graph: a cycle A to B to C to A, a one-way edge C to D, a two-way pair D "
        "and E, and a separate one-way edge F to G. Following arrows, only the cycle and the "
        "two-way pair are mutually reachable, so there are four strong components, A, B, C in "
        "the accent color, D, E in teal, and the singletons F in violet and G in gold, listed "
        "on the card in the same colors.")
def strongly_connected_components(s: Scene) -> None:
    right = _comp(s, "g.strongly_connected_components()", SCC)
    s.card_centered(right, [("{A, B, C}", "accent"), ("{D, E}", "g2"), ("{F}", "g3"),
                            ("{G}", "g4")])


@figure("weakly_connected_components", "communities", "Weakly connected components",
        "A directed graph: a cycle A to B to C to A, a one-way edge C to D, a two-way pair D "
        "and E, and a separate edge F to G. Ignoring arrow direction leaves two pieces, A to E "
        "in the accent color and F, G in teal, which the card lists as the two weak components.")
def weakly_connected_components(s: Scene) -> None:
    right = _comp(s, "g.weakly_connected_components()", WCC)
    s.card_centered(right, [("{A, B, C, D, E}", "accent"), ("{F, G}", "g2")])


@figure("count_strongly_connected_components", "communities", "Count strong components",
        "The same directed graph split into its four strong components: the cycle A, B, C, "
        "the two-way pair D, E, and the singletons F and G, each in its own color. The card "
        "gives the component sizes 3, 2, 1, 1 and the count 4.")
def count_strongly_connected_components(s: Scene) -> None:
    right = _comp(s, "g.count_strongly_connected_components()", SCC)
    s.card_centered(right, ["sizes  3, 2, 1, 1", ("count  4", "accent")])


@figure("count_weakly_connected_components", "communities", "Count weak components",
        "The same directed graph with arrow directions ignored: A to E form one weak "
        "component in the accent color and F, G another in teal. The card gives the "
        "component sizes 5 and 2 and the count 2.")
def count_weakly_connected_components(s: Scene) -> None:
    right = _comp(s, "g.count_weakly_connected_components()", WCC)
    s.card_centered(right, ["sizes  5, 2", ("count  2", "accent")])


@figure("largest_component_size", "communities", "Largest component size",
        "A directed graph with two weak components: A to E, highlighted in the accent color, "
        "and the pair F, G in plain ink. The card gives the component sizes 5 and 2 and the "
        "largest, 5.")
def largest_component_size(s: Scene) -> None:
    left, right = compute(s, "g", "g.largest_component_size()")
    pos = place(COMP, left, unit=UNIT)
    big = set("ABCDE")
    s.graph(pos, [(a, b, {"style": "accent" if a in big else "default", **COMP_BEND.get((a, b), {})})
                  for a, b in COMP_EDGES],
            styles={k: "accent" for k in big}, directed=True)
    s.hull(sorted(big), "accent")
    s.hull(["F", "G"], "ink", opacity=0.06)
    s.card_centered(right, ["sizes    5, 2", ("largest  5", "accent")])


# --- coloring --------------------------------------------------------------

STRIP = {"A": (0, 0), "C": (1, 0), "E": (2, 0), "B": (0.5, 1), "D": (1.5, 1), "F": (2.5, 1)}
STRIP_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("C", "E"),
               ("D", "E"), ("D", "F"), ("E", "F")]


@figure("greedy_color", "communities", "Greedy coloring",
        "A strip of four triangles on six nodes, each node filled with the color of its class: "
        "C and F in the accent color, A and D in teal, B and E in violet. No edge joins two "
        "nodes of the same color, and the card lists the three classes 0, 1, 2.")
def greedy_color(s: Scene) -> None:
    left, right = compute(s, "g", "g.greedy_color()")
    classes = [list("CF"), list("AD"), list("BE")]  # edgecase, strategy="largest_first"
    # every edge joins two classes, so all edges stay ink; no hulls (classes aren't regions)
    _partition(s, place(STRIP, left, unit=UNIT), STRIP_EDGES, classes, hulls=False)
    s.card_centered(right, [("0 → C, F", "accent"), ("1 → A, D", "g2"), ("2 → B, E", "g3")],
                    title="color")


# --- cores, shells, trusses, cohesive blocks -------------------------------

# K4 {A, B, C, D} drawn planar (B inside triangle A, C, D); E closes triangle C, D, E;
# D-E-F-G is a 4-cycle with no chord; H hangs off G.
CORE = {"A": (0, 0.75), "B": (0.87, 0.75), "C": (1.3, 0), "D": (1.3, 1.5), "E": (2.2, 0.4),
        "F": (3.1, 0.55), "G": (2.45, 1.4), "H": (3.35, 1.5)}
CORE_EDGES = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"),
              ("C", "E"), ("D", "E"), ("E", "F"), ("F", "G"), ("D", "G"), ("G", "H")]


def _extract(s: Scene, call: str, keep: str, keep_edges=None) -> None:
    """Before -> after for a returned subgraph: kept nodes/edges accent, rest removed.

    `keep_edges` defaults to the subgraph induced by `keep`.
    """
    left, right = transform(s, "g", call)
    s.graph(place(CORE, left, unit=UNIT), CORE_EDGES, prefix="l.")
    if keep_edges is None:
        keep_edges = {(a, b) for a, b in CORE_EDGES if a in keep and b in keep}
    s.graph(place(CORE, right, unit=UNIT),
            [(a, b, {"style": "accent" if (a, b) in keep_edges else "removed"})
             for a, b in CORE_EDGES],
            styles={k: "accent" if k in keep else "removed" for k in CORE}, prefix="r.")
    s.hull(["r." + k for k in keep], "accent")


@figure("k_core", "communities", "k-core",
        "Left: an eight-node graph with a dense four-node cluster A, B, C, D, a ring E, F, G "
        "and a pendant H. Right: the 3-core, the part in which every node keeps at least three "
        "neighbors, is the cluster A, B, C, D in the accent color; E to H and their edges are "
        "peeled away and drawn dashed.")
def k_core(s: Scene) -> None:
    _extract(s, "g.k_core(3)", "ABCD")


@figure("k_shell", "communities", "k-shell",
        "Left: the eight-node graph with a dense cluster A to D, a ring E, F, G and a pendant H. "
        "Right: the 2-shell holds the nodes whose coreness is exactly 2, so E, F, G and the "
        "edges E to F and F to G are in the accent color, while the 3-core A to D and the "
        "pendant H (coreness 1) are dashed.")
def k_shell(s: Scene) -> None:
    _extract(s, "g.k_shell(2)", "EFG")


@figure("k_crust", "communities", "k-crust",
        "Left: the eight-node graph with a dense cluster A to D, a ring E, F, G and a pendant H. "
        "Right: the 2-crust keeps every node with coreness 2 or less, so E, F, G, H and the "
        "edges among them are in the accent color, and only the 3-core A to D is dashed.")
def k_crust(s: Scene) -> None:
    _extract(s, "g.k_crust(2)", "EFGH")


@figure("k_corona", "communities", "k-corona",
        "Left: the eight-node graph with a dense cluster A to D, a ring E, F, G and a pendant H. "
        "Right: the 2-corona keeps the 2-core nodes with exactly two neighbors in the 2-core, "
        "F and G with their edge in the accent color. E is in the 2-core too but has three "
        "such neighbors, so it is dashed along with the rest.")
def k_corona(s: Scene) -> None:
    _extract(s, "g.k_corona(2)", "FG")


@figure("k_truss", "communities", "k-truss",
        "Left: the eight-node graph with a dense cluster A to D, a ring E, F, G and a pendant H. "
        "Right: the 3-truss keeps only edges that lie in a triangle: the six edges of A to D "
        "plus C to E and D to E, in the accent color. The chordless loop through F and G and "
        "the pendant edge are in no triangle and are dashed, along with F, G, H.")
def k_truss(s: Scene) -> None:
    _extract(s, "g.k_truss(3)", "ABCDE")


@figure("coreness", "communities", "Coreness",
        "The eight-node graph with each node colored by its core number: the dense cluster "
        "A, B, C, D has coreness 3 (accent), the ring E, F, G coreness 2 (teal), and the "
        "pendant H coreness 1 (violet). The table lists the nodes at each coreness value.")
def coreness(s: Scene) -> None:
    left, right = compute(s, "g", "g.coreness()")
    shells = [list("ABCD"), list("EFG"), ["H"]]
    # node colors = shell; badges were dropped: B sits inside the K4 triangle with
    # no free side, and the color + table already give every node's number
    _partition(s, place(CORE, left, unit=UNIT), CORE_EDGES, shells)
    rows = [["coreness", "nodes"], ["3", "A B C D"], ["2", "E F G"], ["1", "H"]]
    s.table_centered(right, rows, [72, 84],
                     highlight={(i, j): GROUPS[i - 1] for i in (1, 2, 3) for j in (0, 1)})


@figure("find_cohesive_blocks", "communities", "Cohesive blocks",
        "The eight-node graph drawn with three nested regions. The whole graph is block 0 "
        "with cohesion 1, since removing G cuts off H; A to G is block 1 with cohesion 2 "
        "(teal); and the cluster A, B, C, D is block 2 with cohesion 3 (accent). The table "
        "lists each block's cohesion, parent block and nodes.")
def find_cohesive_blocks(s: Scene) -> None:
    left, right = compute(s, "g", "g.find_cohesive_blocks()")
    pos = place(CORE, left, unit=UNIT, pad=16)  # room for the outermost hull
    blk2, blk1 = list("ABCD"), list("ABCDEFG")
    grp = {**{k: "g2" for k in blk1}, **{k: "accent" for k in blk2}}
    # an edge takes the color of the innermost block holding both ends
    def style(a, b):
        return ({"style": "accent"} if a in blk2 and b in blk2 else
                {"style": "g2"} if a in blk1 and b in blk1 else {})
    s.graph(pos, [(a, b, style(a, b)) for a, b in CORE_EDGES], styles=grp)
    # nested: wider pad for each enclosing block so the borders stay apart
    s.hull(list(CORE), "ink", pad=21, opacity=0.06)
    s.hull(blk1, "g2", pad=14)
    s.hull(blk2, "accent", pad=7)
    rows = [["block", "k", "parent", "nodes"], ["0", "1", "–", "A … H"], ["1", "2", "0", "A … G"],
            ["2", "3", "1", "A B C D"]]
    s.table_centered(right, rows, [44, 26, 50, 66],
                     highlight={**{(2, j): "g2" for j in range(4)},
                                **{(3, j): "accent" for j in range(4)}})
