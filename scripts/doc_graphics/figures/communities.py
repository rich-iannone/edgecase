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

`modularity` scores a partition you pass in, so that partition is drawn in
g2 / g3 (input) and accent is kept for the returned Q; its card shows each
community's L_c/m − (D_c/2m)² term. `partition_quality` follows the same
input convention and its card shows the coverage and performance fractions. `cmty_k_clique` returns overlapping
communities: their hulls overlap, a node in several communities stays ink
inside the overlap, and nodes in no community are `muted`.
`attracting_components` follows the same rule (A, in no attracting component,
is `muted`); `node_connected_component` shows one component: the query node
`soft`, the rest of its component `accent` inside a hull, other nodes `muted`.

Later additions share `TWO`, the docstrings' two K4 groups joined by D–E:
`cmty_fluid`, `cmty_fast_label_propagation` and
`cmty_semisync_label_propagation` draw the usual partition;
`greedy_source_expansion` draws the source `soft`, the local community as an
accent hull, its boundary node (members with outside neighbors) `g2` and the
rest `muted`; `harmonic_function` and `local_and_global_consistency` follow
`label_propagation` (seed labels as badges, predicted classes as accent / g2).
`cmty_louvain_levels` / `cmty_leiden_levels` are `transform`s on a 2 x 4
ladder: level 0 on the left, level 1 on the right. `kernighan_lin_bisection`
follows connectivity's cut recipe (sides g2 / g3, cut edges accent), from the
starting split to the result; `lukes_partitioning` dashes its one cut edge.
`greedy_node_swap_bipartition` repeats the Kernighan-Lin figure from the same
starting split, with modularity in the note lines, and
`spectral_modularity_bipartition` draws the result alone on the same two
squares, sides as g2 / g3 hulls like `spectral_bisection`.

The partition scores follow `modularity` (the partition you pass in g2 / g3,
the score in accent): `constant_potts_model` on CMTY at resolution 0.5, its
card giving E_c − γ n_c² / 2 per community; `overlapping_modularity` on the
bowtie (two triangles sharing C), whose hulls overlap at C, which stays ink;
and `bipartite_modularity` on its docstring's customers and products, drawn
with node_metrics_local.two_rows, each term L_c / m − k_c d_c / m².
"""

from core import SIZE_NOTE, Box, Scene, compute, figure, place, transform

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
# methods that number communities from the lowest node ID (Louvain, Leiden, label
# propagation, fast greedy)
CMTY_GROUPS_LOW = [list("ABC"), list("DEFG")]
CMTY_CARD_LOW = [("0 → A, B, C", "accent"), ("1 → D, E, F, G", "g2")]


def _cmty(s: Scene, call: str, groups=CMTY_GROUPS, card=CMTY_CARD, **kw) -> None:
    left, right = compute(s, "g", call)
    _partition(s, place(CMTY, left, unit=UNIT), CMTY_EDGES, groups, **kw)
    s.card_centered(right, card, title="community")


@figure("cmty_louvain", "communities", "Louvain communities",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the single edge C to D. Louvain puts each cluster in its own community, drawn as "
        "tinted regions in the accent color and teal. The card lists community 0 as A, B, C "
        "and community 1 as D, E, F, G.")
def cmty_louvain(s: Scene) -> None:
    _cmty(s, "g.cmty_louvain()", CMTY_GROUPS_LOW, CMTY_CARD_LOW)


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
        "remaining pieces are the communities: 0 is A, B, C and 1 is D, E, F, G.")
def cmty_edge_btwns(s: Scene) -> None:
    # communities are numbered by smallest node id, so A, B, C is community 0
    left, right = compute(s, "g", "g.cmty_edge_btwns()")
    _partition(s, place(CMTY, left, unit=UNIT), CMTY_EDGES, [list("ABC"), list("DEFG")],
               edge_kw={("C", "D"): {"style": "removed"}})
    s.card_centered(right, [("0 → A, B, C", "accent"), ("1 → D, E, F, G", "g2")],
                    title="community")


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


# Later additions; each rebuilds its docstring example in edgecase and asserts
# the result. attracting_components draws the returned groups as hulls (in the
# order edgecase returns them); A, in no group, is `muted`.
ATTRACT = {"A": (0, 0.6), "B": (1.2, 0), "C": (2.4, 0), "D": (1.2, 1.2)}
ATTRACT_EDGES = [("A", "B"), ("B", "C"), ("C", "B"), ("A", "D")]


@figure("attracting_components", "communities", "Attracting components",
        "A directed graph: A points to B and to D, and B and C point to each other. A "
        "random walk that leaves A never comes back: it ends up circling B and C or stuck at "
        "D. Those are the attracting components, D in the accent color and B, C in teal, "
        "listed on the card as {D} and {B, C}; A, in neither, is faded.")
def attracting_components(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph(directed=True).add_nodes(list(ATTRACT))
    g.add_edges(ATTRACT_EDGES)
    comps = [sorted(g.get_label_from_id(n) for n in c) for c in g.attracting_components()]
    assert comps == [["D"], ["B", "C"]]
    left, right = compute(s, "g", "g.attracting_components()")
    _partition(s, place(ATTRACT, left, unit=UNIT), ATTRACT_EDGES, comps, directed=True,
               edge_kw={("B", "C"): {"bend": 7}, ("C", "B"): {"bend": 7}})
    s.nodes["A"].style = "muted"
    s.card_centered(right, [("{D}", "accent"), ("{B, C}", "g2")], title="attracting")


NCC = {"A": (0, 0), "B": (1, 0.6), "C": (2, 0), "D": (3.2, 0.2), "E": (3.2, 1.4)}
NCC_EDGES = [("A", "B"), ("B", "C"), ("D", "E")]


@figure("node_connected_component", "communities", "Connected component of a node",
        "Two separate pieces, the path A, B, C and the edge D to E. Starting from A "
        "(outlined), everything reachable lies in its component, the tinted region in the "
        "accent color holding A, B and C; D and E are faded. The card shows {A, B, C}.")
def node_connected_component(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(list(NCC))
    g.add_edges(NCC_EDGES)
    comp = sorted(g.get_label_from_id(n) for n in g.node_connected_component("A"))
    assert comp == ["A", "B", "C"]
    left, right = compute(s, "g", 'g.node_connected_component("A")')
    s.graph(place(NCC, left, unit=UNIT),
            [("A", "B", {"style": "accent"}), ("B", "C", {"style": "accent"}),
             ("D", "E", {"style": "muted"})],
            styles={"A": "soft", "B": "accent", "C": "accent", "D": "muted", "E": "muted"})
    s.hull(comp, "accent")
    s.card_centered(right, [("{A, B, C}", "accent")], title="component of A")


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


# --- more community detection on the shared CMTY graph ---------------------


@figure("cmty_fast_greedy", "communities", "Fast greedy communities",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the single edge C to D. Greedily merging the pairs that raise modularity most stops "
        "at the two clusters, drawn as tinted regions in the accent color and teal. The card "
        "lists community 0 as A, B, C and community 1 as D, E, F, G.")
def cmty_fast_greedy(s: Scene) -> None:
    _cmty(s, "g.cmty_fast_greedy()", CMTY_GROUPS_LOW, CMTY_CARD_LOW)


@figure("cmty_l_eigenvec", "communities", "Leading eigenvector communities",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the single edge C to D. The leading eigenvector of the modularity matrix splits the "
        "nodes by sign at that edge, and neither side splits further, giving two communities "
        "drawn in the accent color and teal: 0 is D, E, F, G and 1 is A, B, C.")
def cmty_l_eigenvec(s: Scene) -> None:
    _cmty(s, "g.cmty_l_eigenvec()")


@figure("cmty_girvan_newman", "communities", "Girvan-Newman communities",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the bridge C to D, drawn dashed because it has the highest betweenness and is cut "
        "first. It shares cmty_edge_btwns()'s implementation, so the result is identical: "
        "community 0 is A, B, C (accent) and community 1 is D, E, F, G (teal).")
def cmty_girvan_newman(s: Scene) -> None:
    # same implementation (and so the same partition and numbering) as cmty_edge_btwns()
    left, right = compute(s, "g", "g.cmty_girvan_newman()")
    _partition(s, place(CMTY, left, unit=UNIT), CMTY_EDGES, [list("ABC"), list("DEFG")],
               edge_kw={("C", "D"): {"style": "removed"}})
    s.card_centered(right, [("0 → A, B, C", "accent"), ("1 → D, E, F, G", "g2")],
                    title="community")


@figure("cmty_stochastic_block_model", "communities", "Stochastic block model",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the single edge C to D. A two-block model fitted with seed 7 puts each cluster in its "
        "own block, dense inside and sparse between, drawn in the accent color and teal. The "
        "card lists block 0 as D, E, F, G and block 1 as A, B, C.")
def cmty_stochastic_block_model(s: Scene) -> None:
    # EM from a random start: seed=7 reaches the two clusters (other seeds may not)
    left, right = compute(s, "g", "g.cmty_stochastic_block_model(seed=7)")
    _partition(s, place(CMTY, left, unit=UNIT), CMTY_EDGES, CMTY_GROUPS)
    s.card_centered(right, CMTY_CARD, title="block")


@figure("label_propagation", "communities", "Label propagation",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the edge C to D. Only A (id 1) and G (id 7) are given labels, 0 and 1, shown as "
        "badges; every other node adopts the label most common among its neighbors, so the "
        "triangle ends with label 0 (accent) and D, E, F, G with label 1 (teal).")
def label_propagation(s: Scene) -> None:
    left, right = compute(s, "g", "g.label_propagation({1: 0, 7: 1})")
    _partition(s, place(CMTY, left, unit=UNIT), CMTY_EDGES, [list("ABC"), list("DEFG")])
    # the seeds you pass in carry their fixed label as a badge
    s.badge("A", "0", where="w")
    s.badge("G", "1", where="e", color="g2")
    s.card_centered(right, [("0 → A, B, C", "accent"), ("1 → D, E, F, G", "g2")],
                    title="label")


# --- k-components (the shared CORE graph) -----------------------------------


@figure("k_components", "communities", "k-components",
        "The shared eight-node core graph: a dense cluster A, B, C, D, a ring through E, F, "
        "G, and a pendant H. The whole graph is the 1-component (gray); dropping H, which "
        "hangs on a single cut vertex, leaves A to G, the 2-component (teal); and the cluster "
        "A, B, C, D survives the loss of any two nodes, so it is the 3-component (accent). "
        "The table lists the components at each level k.")
def k_components(s: Scene) -> None:
    left, right = compute(s, "g", "g.k_components()")
    pos = place(CORE, left, unit=UNIT, pad=16)  # room for the outermost hull
    inner, middle = list("ABCD"), list("ABCDEFG")
    grp = {**{k: "g2" for k in middle}, **{k: "accent" for k in inner}}

    def style(a, b):  # innermost component holding both ends
        if a in inner and b in inner:
            return {"style": "accent"}
        return {"style": "g2"} if a in middle and b in middle else {}
    s.graph(pos, [(a, b, style(a, b)) for a, b in CORE_EDGES], styles=grp)
    s.hull(list(CORE), "ink", pad=21, opacity=0.06)
    s.hull(middle, "g2", pad=14)
    s.hull(inner, "accent", pad=7)
    rows = [["k", "components"], ["1", "[A … H]"], ["2", "[A … G]"], ["3", "[A B C D]"]]
    s.table_centered(right, rows, [30, 100],
                     highlight={**{(2, j): "g2" for j in (0, 1)},
                                **{(3, j): "accent" for j in (0, 1)}})


# --- bipartite sets (the ladder from is_bipartite) -------------------------

LADDER = {"A": (0, 0), "B": (1.4, 0), "C": (2.8, 0), "D": (0, 1.4), "E": (1.4, 1.4),
          "F": (2.8, 1.4)}
LADDER_EDGES = [("A", "B"), ("B", "C"), ("A", "D"), ("B", "E"), ("C", "F"), ("D", "E"),
                ("E", "F")]


@figure("bipartite_sets", "communities", "Bipartite sets",
        "A two-by-three ladder whose nodes alternate between two sides, A, C, E in the accent "
        "color and B, D, F in teal, with every edge joining the two sides. The card lists the "
        "two returned sets, 0 as A, C, E and 1 as B, D, F.")
def bipartite_sets(s: Scene) -> None:
    left, right = compute(s, "g", "g.bipartite_sets()")
    # no hulls: the sides interleave, so regions would overlap; color alone (as in
    # greedy_color and is_bipartite) shows the split
    _partition(s, place(LADDER, left, unit=UNIT), LADDER_EDGES, [list("ACE"), list("BDF")],
               hulls=False)
    s.card_centered(right, [("0 → A, C, E", "accent"), ("1 → B, D, F", "g2")], title="set")


# --- more community methods on the shared CMTY graph -----------------------


@figure("cmty_leiden", "communities", "Leiden communities",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the single edge C to D. Leiden returns the two clusters as communities, each one "
        "connected, drawn as tinted regions in the accent color and teal. The card lists "
        "community 0 as A, B, C and community 1 as D, E, F, G.")
def cmty_leiden(s: Scene) -> None:
    # same partition for every seed tried, so the caption leaves seed= out
    _cmty(s, "g.cmty_leiden()", CMTY_GROUPS_LOW, CMTY_CARD_LOW)


@figure("cmty_label_propagation", "communities", "Label propagation communities",
        "Two dense clusters, a triangle A, B, C and a four-node cluster D, E, F, G, joined by "
        "the single edge C to D. Starting from one label per node, each node repeatedly "
        "adopts its neighbors' most common label; the labels settle on the two clusters, "
        "drawn in the accent color and teal. The card lists community 0 as A, B, C and "
        "community 1 as D, E, F, G.")
def cmty_label_propagation(s: Scene) -> None:
    # same partition for every seed tried, so the caption leaves seed= out
    _cmty(s, "g.cmty_label_propagation()", CMTY_GROUPS_LOW, CMTY_CARD_LOW)


@figure("modularity", "communities", "Modularity",
        "The two-cluster graph with 9 edges, split into the partition you pass: A, B, C "
        "(teal) and D, E, F, G (violet). The card scores each community as its share of "
        "edges inside minus the share expected by chance, 3/9 − (7/18)² and "
        "5/9 − (11/18)², each 0.18, for a modularity Q of 0.36.")
def modularity(s: Scene) -> None:
    # the partition is an input here, so it takes g2/g3 and accent is kept for Q
    left, right = compute(s, 'g,  p = [list("ABC"), list("DEFG")]', "g.modularity(p)")
    pos = place(CMTY, left, unit=UNIT)
    grp = {**dict.fromkeys("ABC", "g2"), **dict.fromkeys("DEFG", "g3")}
    s.graph(pos, [(a, b, {"style": grp[a]}) if grp[a] == grp[b] else (a, b)
                  for a, b in CMTY_EDGES], styles=grp)
    s.hull(list("ABC"), "g2")
    s.hull(list("DEFG"), "g3")
    s.card_centered(right, [("ABC   3/9 − (7/18)² = 0.18", "g2"),
                            ("DEFG  5/9 − (11/18)² = 0.18", "g3"),
                            ("Q     0.36", "accent")], title="L/m − (D/2m)², m = 9")


@figure("partition_quality", "communities", "Partition quality",
        "The two-cluster graph split into the partition you pass: A, B, C (teal) and D, E, "
        "F, G (violet). 8 of the 9 edges lie inside a community, all but the bridge C to D, "
        "so coverage is 8/9 = 0.889. Of the 21 node pairs, 19 are classified correctly (8 "
        "joined pairs inside a community and 11 unjoined pairs across), so performance is "
        "19/21 = 0.905.")
def partition_quality(s: Scene) -> None:
    # same input convention as modularity: the partition is g2/g3, accent is the result
    left, right = compute(s, 'g,  p = [list("ABC"), list("DEFG")]', "g.partition_quality(p)")
    pos = place(CMTY, left, unit=UNIT)
    grp = {**dict.fromkeys("ABC", "g2"), **dict.fromkeys("DEFG", "g3")}
    s.graph(pos, [(a, b, {"style": grp[a]}) if grp[a] == grp[b] else (a, b)
                  for a, b in CMTY_EDGES], styles=grp)
    s.hull(list("ABC"), "g2")
    s.hull(list("DEFG"), "g3")
    s.card_centered(right, ["edges inside      8 of 9",
                            "pairs right  8 + 11 of 21",
                            ("coverage     8/9 = 0.889", "accent"),
                            ("performance 19/21 = 0.905", "accent")],
                    title="(coverage, performance)")


# --- overlapping communities -------------------------------------------------

# Triangles ABC and BCD share edge B-C, so they percolate into one 3-clique
# community; triangle DEF shares only node D with it, so it is a second
# community overlapping at D. G is in no triangle.
KCLQ = {"A": (0, 0.6), "B": (1, 0), "C": (1, 1.2), "D": (2, 0.6), "E": (3, 0), "F": (3, 1.2),
        "G": (4, 1.2)}
KCLQ_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("D", "E"),
              ("D", "F"), ("E", "F"), ("F", "G")]


@figure("cmty_k_clique", "communities", "k-clique communities",
        "Two triangles A, B, C and B, C, D share an edge, so they merge into one community "
        "(accent color); triangle D, E, F shares only node D with it and forms a second "
        "community (teal). The two tinted regions overlap at D, which belongs to both. The "
        "pendant G is in no triangle and in no community.")
def cmty_k_clique(s: Scene) -> None:
    left, right = compute(s, "g", "g.cmty_k_clique(k=3)")
    c1, c2 = set("ABCD"), set("DEF")

    def style(a, b):
        if a in c1 and b in c1:
            return {"style": "accent"}
        if a in c2 and b in c2:
            return {"style": "g2"}
        return {"style": "muted"}
    # D, in both communities, stays ink inside the overlap of the two regions
    styles = {**dict.fromkeys("ABC", "accent"), **dict.fromkeys("EF", "g2"), "G": "muted"}
    s.graph(place(KCLQ, left, unit=UNIT), [(a, b, style(a, b)) for a, b in KCLQ_EDGES],
            styles=styles)
    s.hull(sorted(c1), "accent")
    s.hull(sorted(c2), "g2")
    s.card_centered(right, [("[A, B, C, D]", "accent"), ("[D, E, F]", "g2"),
                            "D in both, G in none"], title="communities")


# --- later additions: more partitioners on two K4 groups ----------------------
#
# Every figure below rebuilds its docstring example in edgecase and asserts the
# result (seeded methods: the seeded result), so a library change breaks the
# build. TWO is the docstrings' graph: two K4 groups A-D and E-H joined by D-E,
# each drawn planar (C and F sit inside their group's outer triangle).
TWO = {"A": (0, 0), "B": (0, 1.2), "C": (0.42, 0.6), "D": (1.2, 0.6), "E": (2.2, 0.6),
       "F": (2.98, 0.6), "G": (3.4, 0), "H": (3.4, 1.2)}
TWO_EDGES = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"),
             ("E", "F"), ("E", "G"), ("E", "H"), ("F", "G"), ("F", "H"), ("G", "H"),
             ("D", "E")]
TWO_GROUPS = [list("ABCD"), list("EFGH")]
TWO_CARD = [("0 → A, B, C, D", "accent"), ("1 → E, F, G, H", "g2")]


def _ec_graph(nodes, edges):
    import edgecase as ec

    g = ec.Graph().add_nodes(list(nodes))
    g.add_edges(list(edges))
    return g


def _groups(g, membership: dict) -> list[list[str]]:
    """Node-to-community dict -> label lists, ordered by community number."""
    out: dict = {}
    for n, c in sorted(membership.items()):
        out.setdefault(c, []).append(g.get_label_from_id(n))
    return [sorted(v) for _, v in sorted(out.items())]


@figure("cmty_fluid", "communities", "Fluid communities",
        "Two dense four-node groups, A, B, C, D and E, F, G, H, joined by the single edge "
        "D to E. Asked for k = 2 communities with seed 3, two fluids start at random nodes "
        "and spread until each fills one group, drawn as tinted regions in the accent color "
        "and teal. The card lists community 0 as A, B, C, D and community 1 as E, F, G, H.")
def cmty_fluid(s: Scene) -> None:
    g = _ec_graph(TWO, TWO_EDGES)
    assert _groups(g, g.cmty_fluid(2, seed=3, return_dict=True)) == TWO_GROUPS
    left, right = compute(s, "g", "g.cmty_fluid(2, seed=3)")
    _partition(s, place(TWO, left, unit=UNIT), TWO_EDGES, TWO_GROUPS)
    s.card_centered(right, TWO_CARD, title="community")


@figure("cmty_fast_label_propagation", "communities", "Fast label propagation communities",
        "Two dense four-node groups, A, B, C, D and E, F, G, H, joined by the single edge "
        "D to E. Every node starts with its own label and adopts its neighbors' most common "
        "label, revisiting only nodes whose neighborhood changed; with seed 1 the labels "
        "settle on the two groups, drawn in the accent color and teal. The card lists "
        "community 0 as A, B, C, D and community 1 as E, F, G, H.")
def cmty_fast_label_propagation(s: Scene) -> None:
    g = _ec_graph(TWO, TWO_EDGES)
    assert _groups(g, g.cmty_fast_label_propagation(seed=1, return_dict=True)) == TWO_GROUPS
    left, right = compute(s, "g", "g.cmty_fast_label_propagation(seed=1)")
    _partition(s, place(TWO, left, unit=UNIT), TWO_EDGES, TWO_GROUPS)
    s.card_centered(right, TWO_CARD, title="community")


@figure("cmty_semisync_label_propagation", "communities",
        "Semi-synchronous label propagation communities",
        "Two dense four-node groups, A, B, C, D and E, F, G, H, joined by the single edge "
        "D to E. The nodes are colored so that no two neighbors share a color, and each "
        "color class adopts its neighbors' most common label at once; with no randomness, "
        "the labels settle on the two groups, drawn in the accent color and teal. The card "
        "lists community 0 as A, B, C, D and community 1 as E, F, G, H.")
def cmty_semisync_label_propagation(s: Scene) -> None:
    g = _ec_graph(TWO, TWO_EDGES)
    assert _groups(g, g.cmty_semisync_label_propagation(return_dict=True)) == TWO_GROUPS
    left, right = compute(s, "g", "g.cmty_semisync_label_propagation()")
    _partition(s, place(TWO, left, unit=UNIT), TWO_EDGES, TWO_GROUPS)
    s.card_centered(right, TWO_CARD, title="community")


@figure("greedy_source_expansion", "communities", "Greedy source expansion",
        "Two dense four-node groups, A, B, C, D and E, F, G, H, joined by the edge D to E. "
        "Growing a local community from the source A (outlined), the expansion takes A's "
        "whole group and then the bridge node E, and stops: the community A, B, C, D, E is "
        "the tinted region in the accent color. E, drawn in teal, is its boundary, the one "
        "member with neighbors outside (F, G, H, faded).")
def greedy_source_expansion(s: Scene) -> None:
    g = _ec_graph(TWO, TWO_EDGES)
    comm = sorted(g.get_label_from_id(n) for n in g.greedy_source_expansion("A"))
    assert comm == list("ABCDE")
    # Clauset's boundary: members with at least one neighbor outside the community
    nbrs = {k: {b if a == k else a for a, b in TWO_EDGES if k in (a, b)} for k in TWO}
    boundary = [k for k in comm if nbrs[k] - set(comm)]
    assert boundary == ["E"]
    left, right = compute(s, "g", 'g.greedy_source_expansion("A")')
    styles = {**dict.fromkeys(comm, "accent"), "A": "soft", "E": "g2",
              **dict.fromkeys("FGH", "muted")}

    def style(a, b):
        if a in comm and b in comm:
            return {"style": "accent"}
        return {} if "E" in (a, b) else {"style": "muted"}
    s.graph(place(TWO, left, unit=UNIT), [(a, b, style(a, b)) for a, b in TWO_EDGES],
            styles=styles)
    s.hull(comm, "accent")
    s.card_centered(right, [("{A, B, C, D, E}", "accent"), ("boundary  E", "g2"),
                            "source    A"], title="local community")


def _classify(s: Scene, method: str) -> None:
    """Two K4 groups with A labeled red, H labeled blue -> predicted label per node."""
    g = _ec_graph(TWO, TWO_EDGES)
    g.set_node_attrs("label", {g.get_id_from_label("A"): "red",
                               g.get_id_from_label("H"): "blue"})
    pred = {g.get_label_from_id(n): v for n, v in getattr(g, method)().items()}
    assert pred == {**dict.fromkeys("ABCD", "red"), **dict.fromkeys("EFGH", "blue")}
    left, right = compute(s, "g", f"g.{method}()")
    _partition(s, place(TWO, left, unit=UNIT), TWO_EDGES, TWO_GROUPS)
    # the two labels you pass in, as badges on their nodes (as in label_propagation)
    s.badge("A", "red", where="w")
    s.badge("H", "blue", where="e", color="g2")
    s.card_centered(right, [("red  → A, B, C, D", "accent"), ("blue → E, F, G, H", "g2")],
                    title="predicted label")


@figure("harmonic_function", "communities", "Harmonic function",
        "Two dense four-node groups, A, B, C, D and E, F, G, H, joined by the edge D to E. "
        "Only A is labeled red and H blue, shown as badges. Each unlabeled node takes the "
        "average of its neighbors' class scores, so A's group is predicted red (accent "
        "color) and H's group blue (teal), as the card lists.")
def harmonic_function(s: Scene) -> None:
    _classify(s, "harmonic_function")


@figure("local_and_global_consistency", "communities", "Local and global consistency",
        "Two dense four-node groups, A, B, C, D and E, F, G, H, joined by the edge D to E. "
        "Only A is labeled red and H blue, shown as badges. Spreading the labels with the "
        "normalized adjacency, each node mixing its neighbors' scores with its own start, "
        "predicts red (accent color) for A, B, C, D and blue (teal) for E, F, G, H, as the "
        "card lists; both seeds keep their labels.")
def local_and_global_consistency(s: Scene) -> None:
    _classify(s, "local_and_global_consistency")


# --- later additions: hierarchy levels ----------------------------------------
#
# A 2 x 4 ladder (A-D on top, E-H below). With seed 0, Louvain and Leiden both
# first pair the nodes along the rails (4 communities), then merge the pairs
# into the two halves of the ladder (2 communities). Same layout in both panels;
# level 0 uses accent / g2 / g3 / g4, level 1 accent / g2.
LADDER4 = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0), "E": (0, 1), "F": (1, 1),
           "G": (2, 1), "H": (3, 1)}
LADDER4_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F"), ("F", "G"), ("G", "H"),
                 ("A", "E"), ("B", "F"), ("C", "G"), ("D", "H")]
LEVEL0 = [list("AB"), list("CD"), list("EF"), list("GH")]
LEVEL1 = [list("ABEF"), list("CDGH")]


def _levels(s: Scene, method: str) -> None:
    g = _ec_graph(LADDER4, LADDER4_EDGES)
    levels = [_groups(g, lv) for lv in getattr(g, method)(seed=0)]
    assert levels == [LEVEL0, LEVEL1]
    left, right = transform(s, f"g.{method}(seed=0)[0]", f"g.{method}(seed=0)[1]")
    _partition(s, place(LADDER4, left, unit=UNIT), LADDER4_EDGES, LEVEL0, prefix="l.")
    _partition(s, place(LADDER4, right, unit=UNIT), LADDER4_EDGES, LEVEL1, prefix="r.")


@figure("cmty_louvain_levels", "communities", "Louvain community levels",
        "A two-by-four ladder, A to D on top and E to H below. Left, Louvain's first level "
        "(seed 0) pairs neighbors along the rails into four communities, A, B in the accent "
        "color, C, D in teal, E, F in violet and G, H in gold. Right, the next level merges "
        "those pairs into two communities, the left half A, B, E, F (accent) and the right "
        "half C, D, G, H (teal).")
def cmty_louvain_levels(s: Scene) -> None:
    _levels(s, "cmty_louvain_levels")


@figure("cmty_leiden_levels", "communities", "Leiden community levels",
        "A two-by-four ladder, A to D on top and E to H below. Left, Leiden's first level "
        "(seed 0) pairs neighbors along the rails into four communities, A, B in the accent "
        "color, C, D in teal, E, F in violet and G, H in gold. Right, after refinement and "
        "aggregation the next level merges them into two communities, A, B, E, F (accent) "
        "and C, D, G, H (teal).")
def cmty_leiden_levels(s: Scene) -> None:
    _levels(s, "cmty_leiden_levels")


# --- later additions: bisection and tree partitioning -------------------------
#
# kernighan_lin_bisection follows connectivity's cut recipe (as spectral_bisection
# does): the two sides are g2 / g3, the cut edges accent. It is a `transform`
# from the starting split you pass (sides as node colors, they interleave) to the
# improved one (sides as hulls), each panel with its cut size in a note line.
# lukes_partitioning draws its clusters like communities, with the one cut tree
# edge dashed (`removed`), as cmty_edge_btwns draws its cut bridge.
SQUARES = {"A": (0, 0), "B": (0, 1.1), "C": (1, 1.1), "D": (1, 0), "E": (2, 0), "F": (2, 1.1),
           "G": (3, 1.1), "H": (3, 0)}
SQUARES_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C"),
                 ("E", "F"), ("F", "G"), ("G", "H"), ("H", "E"), ("E", "G"),
                 ("D", "E"), ("C", "F")]
KL_START = (list("ABEF"), list("CDGH"))


def _sides(s: Scene, box: Box, sides, note: str, note_color, *, hulls: bool, prefix: str):
    side = {k: c for part, c in zip(sides, ("g2", "g3")) for k in part}
    s.text(box.cx, box.y0 + 8, note, size=SIZE_NOTE, color=note_color)
    gb = Box(box.x0, box.y0 + 22, box.x1, box.y1)
    s.graph(place(SQUARES, gb, unit=UNIT),
            [(a, b, {"style": side[a] if side[a] == side[b] else "accent"})
             for a, b in SQUARES_EDGES], styles=side, prefix=prefix)
    if hulls:
        for part, c in zip(sides, ("g2", "g3")):
            s.hull([prefix + k for k in part], c)


@figure("kernighan_lin_bisection", "communities", "Kernighan-Lin bisection",
        "Two squares with a diagonal, A, B, C, D and E, F, G, H, joined by the edges D to E "
        "and C to F. Left, the starting split you pass, A, B, E, F (teal) against C, D, G, H "
        "(violet), cuts 8 edges, in the accent color. Right, after swapping pairs of nodes "
        "the halves are A, B, C, D and E, F, G, H, and only the two accent edges D to E and "
        "C to F are cut, a cut size of 2.0.")
def kernighan_lin_bisection(s: Scene) -> None:
    g = _ec_graph(SQUARES, SQUARES_EDGES)
    start_cut = g.cut_size(KL_START[0])
    a, b = g.kernighan_lin_bisection(partition=KL_START)
    sides = [sorted(g.get_label_from_id(n) for n in part) for part in (a, b)]
    assert sides == TWO_GROUPS and start_cut == 8.0 and g.cut_size(list(a)) == 2.0
    left, right = transform(s, 'p = (list("ABEF"), list("CDGH"))',
                            "g.kernighan_lin_bisection(partition=p)")
    _sides(s, left, KL_START, "start: 8 edges cut", "ink", hulls=False, prefix="l.")
    _sides(s, right, sides, "result: cut size 2.0", "accent", hulls=True, prefix="r.")


# a gentle zigzag, with extra room at the cut C-D so the two hulls stay apart
PATH6 = {k: (0.8 * i + (0.35 if i >= 3 else 0), 0.4 * (i % 2)) for i, k in enumerate("ABCDEF")}
PATH6_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")]


@figure("lukes_partitioning", "communities", "Lukes partitioning",
        "A path of six nodes, A to F, cut into connected clusters of at most three nodes. "
        "The best split cuts only the middle edge C to D (dashed), keeping 4 of the 5 edges "
        "inside clusters: A, B, C in the accent color and D, E, F in teal, as the card "
        "lists.")
def lukes_partitioning(s: Scene) -> None:
    g = _ec_graph(PATH6, PATH6_EDGES)
    clusters = sorted(sorted(g.get_label_from_id(n) for n in c) for c in g.lukes_partitioning(3))
    assert clusters == [list("ABC"), list("DEF")]
    kept = sum(any(a in c and b in c for c in clusters) for a, b in PATH6_EDGES)
    assert kept == 4
    left, right = compute(s, "g", "g.lukes_partitioning(3)")
    _partition(s, place(PATH6, left, unit=UNIT), PATH6_EDGES, clusters,
               edge_kw={("C", "D"): {"style": "removed"}})
    s.card_centered(right, [("{A, B, C}", "accent"), ("{D, E, F}", "g2"),
                            "kept  4 of 5 edges"], title="clusters, max_size 3")


# --- later additions: modularity bipartitions -----------------------------------
#
# Both reuse SQUARES. greedy_node_swap_bipartition starts from KL_START, like
# kernighan_lin_bisection, and reports modularity instead of the cut size.


@figure("greedy_node_swap_bipartition", "communities", "Greedy node-swap bipartition",
        "Two squares with a diagonal, A, B, C, D and E, F, G, H, joined by the edges D to E "
        "and C to F. Left, the starting split you pass, A, B, E, F (teal) against C, D, G, H "
        "(violet), cuts 8 edges, in the accent color, and has modularity -0.167. Right, after "
        "moving the nodes that raise modularity most, the halves are A, B, C, D and E, F, G, "
        "H, with only D to E and C to F cut and a modularity of 0.333.")
def greedy_node_swap_bipartition(s: Scene) -> None:
    g = _ec_graph(SQUARES, SQUARES_EDGES)
    start_q = g.modularity([list(p) for p in KL_START])
    a, b = g.greedy_node_swap_bipartition(init_split=KL_START)
    sides = [sorted(g.get_label_from_id(n) for n in part) for part in (a, b)]
    q = g.modularity([list(a), list(b)])
    assert sides == TWO_GROUPS and (round(start_q, 3), round(q, 3)) == (-0.167, 0.333)
    left, right = transform(s, 'p = (list("ABEF"), list("CDGH"))',
                            "g.greedy_node_swap_bipartition(p)")
    _sides(s, left, KL_START, "start: modularity −0.167", "ink", hulls=False, prefix="l.")
    _sides(s, right, sides, "result: modularity 0.333", "accent", hulls=True, prefix="r.")


@figure("spectral_modularity_bipartition", "communities", "Spectral modularity bipartition",
        "Two squares with a diagonal, A, B, C, D and E, F, G, H, joined by the edges D to E "
        "and C to F. Splitting the nodes by the sign of their entries in the leading "
        "eigenvector of the modularity matrix gives A, B, C, D (teal) and E, F, G, H "
        "(violet), cutting only the two accent edges. The card lists both sides as id sets "
        "and the split's modularity, 0.333.")
def spectral_modularity_bipartition(s: Scene) -> None:
    g = _ec_graph(SQUARES, SQUARES_EDGES)
    a, b = g.spectral_modularity_bipartition()
    sides = [sorted(g.get_label_from_id(n) for n in part) for part in (a, b)]
    q = g.modularity([list(a), list(b)])
    assert sides == TWO_GROUPS and (sorted(a), sorted(b)) == ([1, 2, 3, 4], [5, 6, 7, 8])
    assert round(q, 3) == 0.333
    left, right = compute(s, "g", "g.spectral_modularity_bipartition()")
    _sides(s, left, sides, "sign of the leading eigenvector", "ink", hulls=True, prefix="")
    s.card_centered(right, [("{1, 2, 3, 4}  A, B, C, D", "g2"), ("{5, 6, 7, 8}  E, F, G, H", "g3"),
                            ("cut  D–E, C–F", "accent"), "modularity  0.333"],
                    title="(negative, non-negative side)")


# --- later additions: more partition scores --------------------------------------


@figure("constant_potts_model", "communities", "Constant Potts Model",
        "The two-cluster graph, split into the partition you pass: A, B, C (teal) and D, E, "
        "F, G (violet). At resolution 0.5 each community scores its edges minus 0.5 for "
        "every pair of its nodes, counted as n squared over 2: 3 - 0.5 x 9/2 = 0.75 and "
        "5 - 0.5 x 16/2 = 1.00. The card adds them to a CPM score of 1.75.")
def constant_potts_model(s: Scene) -> None:
    g = _ec_graph(CMTY, CMTY_EDGES)
    p = [list("ABC"), list("DEFG")]
    assert g.constant_potts_model(p, resolution=0.5) == 1.75
    left, right = compute(s, 'g,  p = [list("ABC"), list("DEFG")]',
                          "g.constant_potts_model(p, resolution=0.5)")
    pos = place(CMTY, left, unit=UNIT)
    grp = {**dict.fromkeys("ABC", "g2"), **dict.fromkeys("DEFG", "g3")}
    s.graph(pos, [(a, b, {"style": grp[a]}) if grp[a] == grp[b] else (a, b)
                  for a, b in CMTY_EDGES], styles=grp)
    s.hull(list("ABC"), "g2")
    s.hull(list("DEFG"), "g3")
    s.card_centered(right, [("ABC   3 − 0.5 · 9/2  = 0.75", "g2"),
                            ("DEFG  5 − 0.5 · 16/2 = 1.00", "g3"),
                            ("CPM   1.75", "accent")], title="E − γ n² / 2,  γ = 0.5")


# The is_cover / overlapping_modularity docstring bowtie: triangles A, B, C and C, D, E.
BOWTIE = {"A": (0, 0), "B": (0, 1.2), "C": (1, 0.6), "D": (2, 0), "E": (2, 1.2)}
BOWTIE_EDGES = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "D"), ("D", "E"), ("C", "E")]


@figure("overlapping_modularity", "communities", "Overlapping modularity",
        "Two triangles, A, B, C and C, D, E, sharing node C, with the cover you pass: A, B, C "
        "(teal) and C, D, E (violet), overlapping at C. C is in two communities, so its "
        "edges and degree count one half in each: each community scores 2/6 - (6/12)² = "
        "0.083, for an extended modularity of 0.167. Forcing C into one group gives a "
        "modularity of only 0.111.")
def overlapping_modularity(s: Scene) -> None:
    g = _ec_graph(BOWTIE, BOWTIE_EDGES)
    eq = g.overlapping_modularity([list("ABC"), list("CDE")])
    q = g.modularity([list("ABC"), list("DE")])
    assert (round(eq, 3), round(q, 3)) == (0.167, 0.111)
    left, right = compute(s, 'g,  c = [list("ABC"), list("CDE")]',
                          "g.overlapping_modularity(c)")
    grp = {**dict.fromkeys("AB", "g2"), **dict.fromkeys("DE", "g3")}
    side = {**grp, "C": "both"}

    def style(a, b):
        cols = {side[a], side[b]} - {"both"}
        return {"style": cols.pop()} if len(cols) == 1 else {}

    s.graph(place(BOWTIE, left, unit=UNIT), [(a, b, style(a, b)) for a, b in BOWTIE_EDGES],
            styles=grp)
    s.hull(list("ABC"), "g2")
    s.hull(list("CDE"), "g3")
    s.card_centered(right, [("ABC  2/6 − (6/12)² = 0.083", "g2"),
                            ("CDE  2/6 − (6/12)² = 0.083", "g3"),
                            ("EQ   0.167", "accent"), "C in one group: Q 0.111"],
                    title="C counts ½ in each community")


# bipartite_modularity docstring: customers and the products they buy.
CUSTOMERS = ["Ann", "Bob", "Cy", "Dee"]
PRODUCTS = ["tea", "jam", "nails", "glue"]
BUYS = [("Ann", "tea"), ("Ann", "jam"), ("Bob", "tea"), ("Bob", "jam"), ("Cy", "nails"),
        ("Cy", "glue"), ("Dee", "glue"), ("Bob", "glue")]
SPLIT = [["Ann", "Bob", "tea", "jam"], ["Cy", "Dee", "nails", "glue"]]


@figure("bipartite_modularity", "communities", "Bipartite modularity",
        "Customers Ann, Bob, Cy and Dee on top and the products tea, jam, nails and glue below, "
        "split into the partition you pass: Ann, Bob, tea and jam (teal) and Cy, Dee, nails "
        "and glue (violet). Only Bob to glue crosses. The null model only joins customers to "
        "products: each community scores its share of the 8 edges minus its customer "
        "degrees times its product degrees over 8², 4/8 - 5 x 4/64 and 3/8 - 3 x 4/64, "
        "0.1875 each, for a bipartite "
        "modularity of 0.375.")
def bipartite_modularity(s: Scene) -> None:
    from figures.node_metrics_local import two_rows

    g = _ec_graph(CUSTOMERS + PRODUCTS, BUYS)
    q = g.bipartite_modularity(SPLIT, top_nodes=CUSTOMERS)
    assert round(q, 4) == 0.375
    left, right = compute(s, "g,  p (the two hulls),  c = customers",
                          "g.bipartite_modularity(p, top_nodes=c)")
    grp = {**dict.fromkeys(SPLIT[0], "g2"), **dict.fromkeys(SPLIT[1], "g3")}
    two_rows(s, left, CUSTOMERS, PRODUCTS,
             [(a, b, {"style": grp[a]}) if grp[a] == grp[b] else (a, b) for a, b in BUYS],
             styles=grp)
    s.hull(SPLIT[0], "g2")
    s.hull(SPLIT[1], "g3")
    s.card_centered(right, [("AB  4/8 − 5 · 4/64 = 0.1875", "g2"),
                            ("CD  3/8 − 3 · 4/64 = 0.1875", "g3"),
                            ("Q_B  0.375", "accent")], title="L/m − k d / m²,  m = 8")


# edge_coloring docstring: a round robin of four players, one color per round.
K4 = {"A": (0, 0), "B": (1.6, 0), "C": (0, 1.6), "D": (1.6, 1.6)}
K4_EDGES = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]


@figure("edge_coloring", "communities", "Edge coloring",
        "Four players A, B, C, D, every pair joined by an edge (a game), each edge drawn in "
        "the color of its round and labeled with the round number. No two edges of one color "
        "share a player: round 0 (accent) is A-B and C-D, round 1 (teal) A-C and B-D, and "
        "round 2 (violet) A-D and B-C. The card lists the three rounds.")
def edge_coloring(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(list(K4))
    g.add_edges(K4_EDGES)
    lab = g.get_label_from_id
    colors = g.edge_coloring()
    by_pair = {(lab(a), lab(b)): colors[e] for e, a, b in g.get_edges_as_tuples()}
    rounds = {0: [("A", "B"), ("C", "D")], 1: [("A", "C"), ("B", "D")],
              2: [("A", "D"), ("B", "C")]}
    assert by_pair == {e: c for c, es in rounds.items() for e in es}
    left, right = compute(s, "g", "g.edge_coloring()")
    s.graph(place(K4, left, unit=UNIT),
            [(a, b, {"style": GROUPS[by_pair[(a, b)]], "weight": str(by_pair[(a, b)]),
                     "weight_t": 0.25 if (a, b) in rounds[2] else 0.5}) for a, b in K4_EDGES])
    s.card_centered(right, [(f"{c} → " + ", ".join(f"{a}-{b}" for a, b in es), GROUPS[c])
                            for c, es in rounds.items()], title="color (round)")
