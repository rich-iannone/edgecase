"""Connectivity: how hard the graph is to pull apart.

Template: `compute`. Left: the graph, with the cut or critical elements in
`accent` (edges of a minimum edge cut, nodes of a minimum node cut, bridges,
articulation points) and, for the pairwise versions, the two endpoints s/t
`soft`. Right: a card. Methods that return elements list them in accent;
methods that return a number show one cut that achieves it, then the value
in accent.

Most siblings share one base graph: two diamonds sharing the hub D. Its edge
cuts have size 2 (B–D, C–D) but its node cut has size 1 (D), so the edge and
node variants visibly disagree. Pairwise variants all use the pair B, E.
Exceptions: bridges/articulation points use two triangles strung on a path
(the base graph has no bridges); `max_flow` uses a small directed network;
`node_percolation` returns a Graph, so it is a `transform` (before → after).

Later additions follow the same rules: `max_flow_value` and `min_cost_flow`
reuse the `max_flow` network; `edge_percolation` mirrors `node_percolation`;
`network_robustness` and `percolation_threshold_analysis` return a curve, so
the card becomes a small line chart (accent line and dots, ink axes) beside
the base graph.

`biconnected_components` borrows the communities family's overlapping hulls on
the bridge graph: one hull and group color per block, the articulation points
ink inside the overlaps, an isolate `muted` (it is in no block).
`local_bridges` uses the two triangles joined both directly and through G,
with the local bridges in accent and an id / edge / span table.

The cut methods that return both sides (`minimum_cut`, `stoer_wagner`) share a
weighted graph, CUT: two triangles linked by B–D and C–E, plus G hanging off A
and C by light edges. Capacities are edge weight labels, the cut edges are in
accent, and the two sides are hulls in `g2` / `g3` (accent is taken by the
cut). Between A and F the cut is the 1 + 2 link; globally it is G's 1 + 1.
`gomory_hu_tree` returns a Graph, so it is a `transform`: the graph with the
A–F cut on the left, the tree (drawn on the same positions) on the right, with
the A … F tree path in ink, the rest muted, and its lightest edge in accent.

The set measures (`cut_size`, `volume`, `normalized_cut_size`, `conductance`,
`edge_expansion`, `mixing_expansion`, `node_expansion`, `boundary_expansion`,
`node_boundary`, `edge_boundary`) share ONE graph, SET, and ONE set S = A, B, C
so their numbers compare directly. S and the rest T are `g2` / `g3` hulls,
named in a note line in the same colors (S is the input; accent stays free for
the result). The crossing edges, counted neighbors, or boundary nodes are in
accent; degree badges appear where a volume enters the formula, edge-id labels
where ids are returned; the card shows the formula's numbers.

`spectral_bisection` returns two sides as well, so it follows the cut recipe
on graph_metrics' PAIR (two triangles joined by C–D, the graph fiedler_vector
shades): the negative side `g2` (fiedler_vector's negative color), the
non-negative side `g3`, the one cut edge in accent, and a card with both id
lists in their colors.
"""

from core import SIZE_NOTE, Box, Scene, compute, figure, place, transform

from figures.graph_metrics import PAIR, PAIR_EDGES

UNIT = 72  # one density for the whole family

# Base graph: two diamonds A-B-C-D and D-E-F-G sharing the hub D.
BASE = {"A": (0, 0.6), "B": (1, 0), "C": (1, 1.2), "D": (2, 0.6), "E": (3, 0), "F": (3, 1.2),
        "G": (4, 0.6)}
BASE_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("D", "E"), ("D", "F"),
              ("E", "F"), ("E", "G"), ("F", "G")]
EDGE_CUT = {("B", "D"), ("C", "D")}  # a minimum edge cut (size 2)
NODE_CUT = {"D"}  # the minimum node cut (size 1)

# Bridge graph: two triangles strung on the path C–D–E.
CHAIN = {"A": (0, 0), "B": (0, 1.2), "C": (1, 0.6), "D": (2, 0.6), "E": (3, 0.6), "F": (4, 0),
         "G": (4, 1.2)}
CHAIN_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("E", "G"),
               ("F", "G")]


def _base(s: Scene, box, *, edges: set = frozenset(), nodes: set = frozenset(),
          pair: bool = False) -> None:
    """Draw the base graph with the given cut edges/nodes in accent."""
    styles = {k: "accent" for k in nodes}
    if pair:
        styles.update(B="soft", E="soft")
    s.graph(place(BASE, box, unit=UNIT),
            [(a, b, {"style": "accent"}) if (a, b) in edges else (a, b) for a, b in BASE_EDGES],
            styles=styles)


# --- critical elements -----------------------------------------------------


@figure("get_articulation_points", "connectivity", "Articulation points",
        "Two triangles joined by the path C, D, E. The nodes C, D, and E are in the accent "
        "color: removing any one of them splits the graph. The card lists the articulation "
        "points C, D, E.")
def get_articulation_points(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_articulation_points()")
    s.graph(place(CHAIN, left, unit=UNIT), CHAIN_EDGES, styles={k: "accent" for k in "CDE"})
    s.card_centered(right, [("[C, D, E]", "accent")], title="articulation points")


@figure("bridges", "connectivity", "Bridges",
        "Two triangles joined by the path C, D, E. The edges C–D and D–E are in the accent "
        "color: each is the only link between its two sides, so removing it disconnects the "
        "graph. The card lists the bridges C–D and D–E.")
def bridges(s: Scene) -> None:
    left, right = compute(s, "g", "g.bridges()")
    bridge = {("C", "D"), ("D", "E")}
    s.graph(place(CHAIN, left, unit=UNIT),
            [(a, b, {"style": "accent"}) if (a, b) in bridge else (a, b) for a, b in CHAIN_EDGES])
    s.card_centered(right, [("[C–D, D–E]", "accent")], title="bridge edges")


# --- global connectivity numbers -------------------------------------------


@figure("get_edge_connectivity", "connectivity", "Edge connectivity",
        "Two diamonds that share the hub node D. The two accent edges B–D and C–D form a "
        "smallest edge cut: removing both disconnects the graph and no single edge does. "
        "The card shows that cut and the edge connectivity 2.")
def get_edge_connectivity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_edge_connectivity()")
    _base(s, left, edges=EDGE_CUT)
    s.card_centered(right, ["cut    B–D, C–D", ("value  2", "accent")])


@figure("get_node_connectivity", "connectivity", "Node connectivity",
        "Two diamonds that share the hub node D, drawn in the accent color. Removing D alone "
        "disconnects the graph, so the node connectivity is 1, smaller than the edge "
        "connectivity of 2. The card shows the cut D and the value 1.")
def get_node_connectivity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_node_connectivity()")
    _base(s, left, nodes=NODE_CUT)
    s.card_centered(right, ["cut    D", ("value  1", "accent")])


@figure("get_adhesion", "connectivity", "Adhesion",
        "Two diamonds that share the hub node D. The two accent edges B–D and C–D form a "
        "smallest edge cut, so the adhesion (edge connectivity) is 2. The card shows that "
        "cut and the value 2.")
def get_adhesion(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_adhesion()")
    _base(s, left, edges=EDGE_CUT)
    s.card_centered(right, ["cut    B–D, C–D", ("value  2", "accent")])


@figure("get_cohesion", "connectivity", "Cohesion",
        "Two triangles joined through the two middle nodes D and E, drawn in the accent "
        "color. No single node disconnects the graph, but removing D and E together does, "
        "so the cohesion (node connectivity) is 2. The card shows the cut D, E and the value 2.")
def get_cohesion(s: Scene) -> None:
    # A 2-connected example, so this alias doesn't just repeat the node
    # connectivity figure (which shows the base graph's single cut node).
    left, right = compute(s, "g", "g.get_cohesion()")
    coords = {"A": (0, 0.6), "B": (1, 0), "C": (1, 1.2), "D": (2, 0), "E": (2, 1.2), "F": (3, 0),
              "G": (3, 1.2), "H": (4, 0.6)}
    edges = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"), ("D", "F"),
             ("E", "G"), ("F", "G"), ("F", "H"), ("G", "H")]
    s.graph(place(coords, left, unit=UNIT), edges, styles={"D": "accent", "E": "accent"})
    s.card_centered(right, ["cut    D, E", ("value  2", "accent")])


# --- pairwise (s, t) versions: always B and E on the base graph -------------


@figure("get_min_cut_between", "connectivity", "Minimum cut between two nodes",
        "Two diamonds that share the hub node D, with B and E outlined as source and sink. "
        "The two accent edges B–D and C–D are a smallest set of edges whose removal "
        "separates B from E. The card shows that cut and its size 2.")
def get_min_cut_between(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_min_cut_between("B", "E")')
    _base(s, left, edges=EDGE_CUT, pair=True)
    s.card_centered(right, ["cut    B–D, C–D", ("value  2", "accent")])


@figure("minimum_edge_cut", "connectivity", "Minimum edge cut",
        "Two diamonds that share the hub node D, with B and E outlined as source and sink. "
        "The two accent edges B–D and C–D are the minimum edge cut separating B from E, and "
        "the card returns them as the set of pairs (B, D) and (C, D).")
def minimum_edge_cut(s: Scene) -> None:
    left, right = compute(s, "g", 'g.minimum_edge_cut("B", "E")')
    _base(s, left, edges=EDGE_CUT, pair=True)
    s.card_centered(right, [("{(B, D), (C, D)}", "accent")], title="edges")


@figure("minimum_node_cut", "connectivity", "Minimum node cut",
        "Two diamonds that share the hub node D, with B and E outlined as source and sink. "
        "The hub D is in the accent color: removing it alone separates B from E, so the "
        "card returns the set containing D.")
def minimum_node_cut(s: Scene) -> None:
    left, right = compute(s, "g", 'g.minimum_node_cut("B", "E")')
    _base(s, left, nodes=NODE_CUT, pair=True)
    s.card_centered(right, [("{D}", "accent")], title="nodes")


@figure("get_node_connectivity_between", "connectivity", "Node connectivity between two nodes",
        "Two diamonds that share the hub node D, with B and E outlined as source and target. "
        "Every path from B to E passes through the accent hub D, so one node removal "
        "separates them. The card shows the cut D and the value 1.")
def get_node_connectivity_between(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_node_connectivity_between("B","E")')
    _base(s, left, nodes=NODE_CUT, pair=True)
    s.card_centered(right, ["cut    D", ("value  1", "accent")])


# --- flow ------------------------------------------------------------------


@figure("max_flow", "connectivity", "Maximum flow",
        "A directed network from source S to sink T through A and B. Each edge is labeled "
        "flow over capacity, and every edge carries flow, drawn in the accent color with "
        "width growing with the flow. The card lists the flow per edge, which totals 5.")
def max_flow(s: Scene) -> None:
    left, right = compute(s, "g", 'g.max_flow("S", "T")')
    coords = {"S": (0, 0.8), "A": (1.5, 0), "B": (1.5, 1.6), "T": (3, 0.8)}
    # (from, to, capacity, flow) as returned by edgecase for this network
    net = [("S", "A", 4, 3), ("S", "B", 2, 2), ("A", "B", 2, 1), ("A", "T", 2, 2), ("B", "T", 3, 3)]
    s.graph(place(coords, left, unit=UNIT),
            [(a, b, {"style": "accent", "weight": f"{f}/{c}", "width": 1.25 + 0.75 * f})
             for a, b, c, f in net],
            directed=True, styles={"S": "soft", "T": "soft"})
    s.card_centered(right, [f"{a}→{b}  {f}" for a, b, _, f in net] + [("total 5", "accent")],
                    title="flow per edge")


# --- percolation (returns a Graph) -----------------------------------------


@figure("node_percolation", "connectivity", "Node percolation",
        "Before: the two diamonds sharing hub D. After removing 30 percent of the nodes at "
        "random (here D and F, drawn dashed with their edges), the surviving graph has split "
        "into the triangle A, B, C and the pair E, G.")
def node_percolation(s: Scene) -> None:
    left, right = transform(s, "g", "g.node_percolation(q=0.3, seed=25)")
    gone = {"D", "F"}  # what seed=25 removes (round(0.3 * 7) = 2 nodes)
    s.graph(place(BASE, left, unit=UNIT), BASE_EDGES, prefix="l.")
    s.graph(place(BASE, right, unit=UNIT),
            [(a, b, {"style": "removed"}) if gone & {a, b} else (a, b) for a, b in BASE_EDGES],
            styles={k: "removed" for k in gone}, prefix="r.")


# --- later additions: flow variants ----------------------------------------

# The max_flow network: (from, to, capacity).
FLOW = {"S": (0, 0.8), "A": (1.5, 0), "B": (1.5, 1.6), "T": (3, 0.8)}
FLOW_NET = [("S", "A", 4), ("S", "B", 2), ("A", "B", 2), ("A", "T", 2), ("B", "T", 3)]


@figure("max_flow_value", "connectivity", "Maximum flow value",
        "The directed network from the max_flow figure, each edge labeled with its capacity. "
        "The accent edges A→T and B→T form a minimum cut separating source S from sink T, "
        "with capacity 2 + 3. The card shows that cut and the returned maximum flow value 5.0.")
def max_flow_value(s: Scene) -> None:
    left, right = compute(s, "g", 'g.max_flow_value("S", "T")')
    cut = {("A", "T"), ("B", "T")}  # S-side {S, A, B}: the only cut of capacity 5
    s.graph(place(FLOW, left, unit=UNIT),
            [(a, b, {"style": "accent", "weight": c}) if (a, b) in cut else (a, b, {"weight": c})
             for a, b, c in FLOW_NET],
            directed=True, styles={"S": "soft", "T": "soft"})
    s.card_centered(right, ["cut    A→T, B→T", ("value  2 + 3 = 5.0", "accent")])


@figure("min_cost_flow", "connectivity", "Minimum cost flow",
        "The max_flow network with a per-unit cost on each edge, labeled flow/capacity "
        "then cost. Sending a demand of 4 from S to T at the lowest cost fills S→A, then "
        "splits at A between the cheap detour A→B→T and the direct A→T; the costly S→B "
        "stays unused and faded. The card lists flow times cost per edge and the totals: "
        "flow 4, cost 14.")
def min_cost_flow(s: Scene) -> None:
    left, right = compute(s, "g", 'g.min_cost_flow("S", "T", demand=4)')
    # (from, to, capacity, cost, flow) as returned by edgecase for demand=4
    net = [("S", "A", 4, 1, 4), ("S", "B", 2, 4, 0), ("A", "B", 2, 1, 2), ("A", "T", 2, 3, 2),
           ("B", "T", 3, 1, 2)]
    s.graph(place(FLOW, left, unit=UNIT),
            [(a, b, {"style": "accent", "weight": f"{f}/{c} ${k}", "width": 1.25 + 0.75 * f})
             if f else (a, b, {"style": "muted", "weight": f"{f}/{c} ${k}"})
             for a, b, c, k, f in net],
            directed=True, styles={"S": "soft", "T": "soft"})
    s.card_centered(right, [f"{a}→{b}  {f} × ${k}" for a, b, _, k, f in net if f]
                    + [("flow 4, cost 14", "accent")], title="flow × cost")


# --- later additions: percolation & robustness -----------------------------


@figure("edge_percolation", "connectivity", "Edge percolation",
        "Before: the two diamonds sharing hub D. After removing 30 percent of the edges at "
        "random (here B–D, C–D, and D–E, drawn dashed), every node survives but the graph "
        "has split into the triangle A, B, C and the four nodes D, E, F, G.")
def edge_percolation(s: Scene) -> None:
    left, right = transform(s, "g", "g.edge_percolation(q=0.3, seed=14)")
    gone = {("B", "D"), ("C", "D"), ("D", "E")}  # what seed=14 removes (3 of 10 edges)
    s.graph(place(BASE, left, unit=UNIT), BASE_EDGES, prefix="l.")
    s.graph(place(BASE, right, unit=UNIT),
            [(a, b, {"style": "removed"}) if (a, b) in gone else (a, b) for a, b in BASE_EDGES],
            prefix="r.")


# Plot area of the small curve charts, inside the compute result panel.
PLOT = Box(546, 50, 726, 150)  # y0 is the value 1, y1 the value 0


def _curve(s: Scene, pts: list[tuple[float, float]], *, title: str, xlabel: str,
           xticks: list[str], labels: dict[int, str] | None = None) -> None:
    """Line chart of (x in 0..1, y in 0..1): ink axes, accent line and dots.

    `xticks` sit under each point; `labels` (point index -> text) sit above
    each dot, or below it where the curve dips.
    """
    p = PLOT
    xy = [(p.x0 + x * p.w, p.y1 - y * p.h) for x, y in pts]
    s.line(p.x0 - 8, p.y1, p.x1 + 8, p.y1, opacity=0.5, width=1, role="axis")
    s.line(p.x0 - 8, p.y0, p.x1 + 8, p.y0, opacity=0.2, width=1, dash="3 4", role="axis")
    s.text(p.x0 - 14, p.y0, "1.0", size=SIZE_NOTE, anchor="end", opacity=0.6)
    s.text(p.x0 - 14, p.y1, "0", size=SIZE_NOTE, anchor="end", opacity=0.6)
    s.text(p.x0 - 8, p.y0 - 26, title, size=SIZE_NOTE, anchor="start", opacity=0.6)
    s.text(p.cx, p.y1 + 28, xlabel, size=SIZE_NOTE, opacity=0.6)
    s.path("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in xy), color="accent", width=2.25)
    for i, (x, y) in enumerate(xy):
        s.path(f"M {x - 3.5:.1f} {y:.1f} a 3.5 3.5 0 1 0 7 0 a 3.5 3.5 0 1 0 -7 0",
               color="accent", fill="color", width=1)
        s.text(x, p.y1 + 13, xticks[i], size=SIZE_NOTE, opacity=0.75)
        if labels and i in labels:
            # above the dot, or below it at a dip; paper-boxed so a steep
            # neighboring segment never runs through it
            dip = 0 < i < len(xy) - 1 and y > xy[i - 1][1] and y > xy[i + 1][1]
            s.text(x, y + (14 if dip else -14), labels[i], size=SIZE_NOTE, color="accent",
                   weight=600, boxed=True)


@figure("network_robustness", "connectivity", "Network robustness",
        "The two diamonds sharing hub D, with filled badges giving the degree-based removal "
        "order: D first (degree 4), then B, C, E, F, A, G. The chart plots the largest "
        "component as a fraction of the 7 nodes after each removal: 1.0, then 0.43 once D "
        "is gone, falling to 0.29, 0.14, and 0.")
def network_robustness(s: Scene) -> None:
    left, right = compute(s, "g", 'g.network_robustness("degree")')
    # highest degree first, ties by node id (edgecase's static order)
    order = ["D", "B", "C", "E", "F", "A", "G"]
    _base(s, left, nodes={"D"})
    where = {"A": "n", "B": "n", "C": "s", "D": "n", "E": "n", "F": "s", "G": "n"}
    for i, k in enumerate(order, 1):
        s.badge(k, str(i), where=where[k], filled=True)
    sizes = [7, 3, 3, 3, 2, 1, 1, 0]  # largest component after each removal
    _curve(s, [(i / 7, v / 7) for i, v in enumerate(sizes)], title="largest component / n",
           xlabel="nodes removed", xticks=["–"] + order,
           labels={2: "0.43", 6: "0.14"})  # on the flat runs, clear of the line


@figure("percolation_threshold_analysis", "connectivity", "Percolation threshold analysis",
        "The two diamonds sharing hub D. The chart plots, for removal fractions q of 0, "
        "0.25, 0.5, 0.75, and 1, the giant component's share of the surviving nodes, "
        "averaged over 10 random trials: 1.0, 0.82, 0.55, 0.75, 0. It dips at q = 0.5, "
        "where the graph tends to split, and is 0 once every node is gone.")
def percolation_threshold_analysis(s: Scene) -> None:
    left, right = compute(s, "g", "g.percolation_threshold_analysis(4)")
    _base(s, left)
    # one run of steps=4 with the default 10 trials (seed=1); with so few
    # nodes the 2 survivors at q = 0.75 are often adjacent, hence the rebound
    vals = [1.0, 0.82, 0.55, 0.75, 0.0]
    _curve(s, [(i / 4, v) for i, v in enumerate(vals)], title="giant component / survivors",
           xlabel="fraction removed q", xticks=["0", ".25", ".5", ".75", "1"],
           labels={i: f"{v:.2f}" for i, v in enumerate(vals) if 0 < i < 4})


# --- later additions: blocks and local bridges -----------------------------


@figure("biconnected_components", "connectivity", "Biconnected components",
        "Two triangles joined by the path C, D, E, plus an isolated node H. Each block has "
        "its own tinted region and color: A, B, C; the bridge C, D; the bridge D, E; and "
        "E, F, G. The articulation points C, D, and E sit where two regions overlap, because "
        "each belongs to two blocks; H belongs to none.")
def biconnected_components(s: Scene) -> None:
    left, right = compute(s, "g", "g.biconnected_components()")
    # Blocks follow the path's order, so their colors run left to right; an
    # articulation point (in two blocks) stays ink inside the overlap, as in
    # the communities family's overlapping cmty_k_clique figure.
    blocks = [("ABC", "accent"), ("CD", "g2"), ("DE", "g3"), ("EFG", "g4")]
    color = {e: c for b, c in blocks for e in CHAIN_EDGES if set(e) <= set(b)}
    coords = {**CHAIN, "H": (2, 1.5)}  # isolate tucked under the path
    s.graph(place(coords, left, unit=UNIT),
            [(a, b, {"style": color[(a, b)]}) for a, b in CHAIN_EDGES],
            styles={"A": "accent", "B": "accent", "F": "g4", "G": "g4", "H": "muted"})
    for b, c in blocks:
        s.hull(list(b), c)
    s.card_centered(right, [("[A, B, C]", "accent"), ("[C, D]", "g2"), ("[D, E]", "g3"),
                            ("[E, F, G]", "g4"), "H in none"], title="blocks")


# Two triangles joined directly by C–D and the long way round through G,
# plus the pendant edge E–H (the local_bridges docstring example).
SPAN = {"A": (0, 0.1), "B": (0, 1.3), "C": (1, 0.7), "D": (2.4, 0.7), "E": (3.4, 1.3),
        "F": (3.4, 0.1), "G": (1.7, -0.45), "H": (4.4, 1.3)}
SPAN_EDGES = [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E"), ("E", "F"), ("F", "D"),
              ("C", "D"), ("A", "G"), ("G", "F"), ("E", "H")]


@figure("local_bridges", "connectivity", "Local bridges",
        "Two triangles joined directly by the edge C–D and the long way round through G, "
        "plus a pendant node H. The four accent edges C–D, A–G, G–F, and E–H lie on no "
        "triangle. The table lists each with its edge id and span: without it, its ends "
        "are 4 hops apart, or never reconnect (inf) for the true bridge E–H.")
def local_bridges(s: Scene) -> None:
    left, right = compute(s, "g", "g.local_bridges()")
    local = {("C", "D"), ("A", "G"), ("G", "F"), ("E", "H")}
    s.graph(place(SPAN, left, unit=UNIT),
            [(a, b, {"style": "accent"}) if (a, b) in local else (a, b) for a, b in SPAN_EDGES])
    # (edge_id, span) as edgecase returns them; ids follow SPAN_EDGES order
    rows = [["id", "edge", "span"], ["7", "C–D", "4"], ["8", "A–G", "4"], ["9", "G–F", "4"],
            ["10", "E–H", "inf"]]
    s.table_centered(right, rows, [40, 60, 50],
                     highlight={(i, 2): "accent" for i in range(1, 5)})


# --- later additions: cuts with both sides ---------------------------------

# Weighted cut graph: triangles A, B, C and D, E, F linked by B–D and C–E,
# plus G tied to A and C by two light edges. Weights are the capacities.
CUT = {"A": (0, 0.55), "B": (1, 0), "C": (1, 1.1), "D": (2.3, 0), "E": (2.3, 1.1),
       "F": (3.3, 0.55), "G": (0.05, 1.65)}
CUT_EDGES = [("A", "B", 3), ("A", "C", 3), ("B", "C", 2), ("B", "D", 1), ("C", "E", 2),
             ("D", "E", 2), ("D", "F", 3), ("E", "F", 3), ("A", "G", 1), ("C", "G", 1)]
AF_CUT = {("B", "D"), ("C", "E")}  # minimum_cut("A", "F"): value 1 + 2 = 3.0


def _cut(s: Scene, box: Box, cut: set, *, sides: tuple[str, str] = (), pair: bool = False,
         prefix: str = "") -> None:
    """The weighted cut graph, `cut` edges in accent, sides as g2 / g3 hulls."""
    s.graph(place(CUT, box, unit=UNIT),
            [(a, b, {"style": "accent", "weight": w}) if (a, b) in cut else (a, b, {"weight": w})
             for a, b, w in CUT_EDGES],
            styles={"A": "soft", "F": "soft"} if pair else {}, prefix=prefix)
    for side, c in zip(sides, ("g2", "g3")):
        s.hull([prefix + k for k in side], c)


@figure("minimum_cut", "connectivity", "Minimum cut",
        "A weighted graph whose weights are capacities, with source A and sink F outlined. "
        "The cheapest way to separate them cuts the two accent edges B–D and C–E, of "
        "capacity 1 + 2 = 3.0. The two sides are shaded: A, B, C, G with the source (teal) "
        "and D, E, F with the sink (violet), as listed on the card.")
def minimum_cut(s: Scene) -> None:
    left, right = compute(s, "g", 'g.minimum_cut("A", "F")')
    _cut(s, left, AF_CUT, sides=("ABCG", "DEF"), pair=True)
    s.card_centered(right, [("value  1 + 2 = 3.0", "accent"), ("A side  A, B, C, G", "g2"),
                            ("F side  D, E, F", "g3")], title="(value, (source side, sink side))")


@figure("stoer_wagner", "connectivity", "Stoer-Wagner minimum cut",
        "The same weighted graph, with no source or sink. The cheapest split of the whole "
        "graph is not between the two triangles (capacity 3) but around G: its two light "
        "accent edges A–G and C–G cost 1 + 1 = 2.0. The card gives that value and the two "
        "sides, A to F (teal) and G alone (violet).")
def stoer_wagner(s: Scene) -> None:
    left, right = compute(s, "g", "g.stoer_wagner()")
    _cut(s, left, {("A", "G"), ("C", "G")}, sides=("ABCDEF", "G"))
    s.card_centered(right, [("value  1 + 1 = 2.0", "accent"), ("side a  A, B, C, D, E, F", "g2"),
                            ("side b  G", "g3")], title="(value, (side a, side b))")


# gomory_hu_tree() on the cut graph; every tree edge happens to be a graph edge.
GH_TREE = [("A", "B", 6), ("A", "C", 7), ("A", "G", 2), ("C", "E", 3), ("D", "E", 6),
           ("D", "F", 6)]
GH_PATH = {("A", "C"), ("C", "E"), ("D", "E"), ("D", "F")}  # tree path A ... F


@figure("gomory_hu_tree", "connectivity", "Gomory-Hu tree",
        "Left: the weighted cut graph, whose minimum cut between A and F is the accent pair "
        "of edges B–D and C–E, capacity 3. Right: its Gomory-Hu tree, six edges weighted "
        "with pairwise cut values. On the tree path from A to F (weights 7, 3, 6, 6) the "
        "lightest edge, C–E with weight 3, is in the accent color: it gives the same cut "
        "value 3.")
def gomory_hu_tree(s: Scene) -> None:
    left, right = transform(s, "g", "g.gomory_hu_tree()")
    _cut(s, left, AF_CUT, pair=True, prefix="l.")
    s.graph(place(CUT, right, unit=UNIT),
            [(a, b, {"weight": w, "style": "accent" if (a, b) == ("C", "E")
                     else "default" if (a, b) in GH_PATH else "muted"})
             for a, b, w in GH_TREE],
            styles={"A": "soft", "F": "soft"}, prefix="r.")


# --- later additions: one set S and its cut / boundary ---------------------

# Every set measure uses ONE graph and ONE set S, so the numbers compare
# directly: the triangle S = A, B, C leaks into D and E through three edges
# (B–D, C–D, C–E); the rest is D, E, F and the pendant G. Degrees: A 2, B 3,
# C 4, D 4, E 3, F 3, G 1 (m = 10), so vol(S) = 9 < vol(T) = 11 and
# |S| = 3 < |T| = 4, and the cut-, size- and boundary-based scores all differ.
SET = {"A": (0, 0.6), "B": (1, 0), "C": (1, 1.2), "D": (2.2, 0), "E": (2.2, 1.2),
       "F": (3.2, 0.6), "G": (4.2, 0.6)}
SET_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("C", "E"),
             ("D", "E"), ("D", "F"), ("E", "F"), ("F", "G")]
SET_S, SET_T = "ABC", "DEFG"
SET_CUT = {("B", "D"), ("C", "D"), ("C", "E")}  # edge ids 4, 5, 6
SET_DEG = {"A": 2, "B": 3, "C": 4, "D": 4, "E": 3, "F": 3, "G": 1}


def _set(s: Scene, call: str, *, cut: bool = True, nodes: str = "",
         badges: str = "", edge_ids: bool = False) -> Box:
    """The set graph: S and the rest as g2 / g3 hulls named in a note line.

    `cut` draws the crossing edges in accent, `nodes` fills nodes in accent,
    `badges` puts degree badges on those nodes, `edge_ids` labels the cut
    edges with their edge ids. Returns the result panel.
    """
    left, right = compute(s, "g", call)
    s.text(left.cx - 12, left.y0 + 8, "S = [A, B, C]", size=SIZE_NOTE, anchor="end",
           color="g2")
    s.text(left.cx + 12, left.y0 + 8, "T = rest (D, E, F, G)", size=SIZE_NOTE,
           anchor="start", color="g3")
    gb = Box(left.x0, left.y0 + 22, left.x1, left.y1)
    edges = []
    for i, (a, b) in enumerate(SET_EDGES):
        kw: dict = {}
        if cut and (a, b) in SET_CUT:
            kw["style"] = "accent"
            if edge_ids:
                kw["weight"] = str(i + 1)
        edges.append((a, b, kw))
    s.graph(place(SET, gb, unit=UNIT), edges, styles={k: "accent" for k in nodes})
    s.hull(list(SET_S), "g2")
    s.hull(list(SET_T), "g3")
    for k in badges:
        s.badge(k, str(SET_DEG[k]), where={"B": "n", "C": "s", "D": "n", "E": "s"}.get(k, "auto"))
    return right


@figure("cut_size", "connectivity", "Cut size",
        "A seven-node graph split into the set S = A, B, C (teal) and the rest D, E, F, G "
        "(violet). Three accent edges cross between them, B–D, C–D and C–E, so the cut "
        "size is 3.0.")
def cut_size(s: Scene) -> None:
    right = _set(s, "g.cut_size(S)")
    s.card_centered(right, ["crossing  B–D, C–D, C–E", ("cut_size  3.0", "accent")])


@figure("volume", "connectivity", "Volume",
        "The same graph and set S = A, B, C, with each member's degree in a badge: A 2, "
        "B 3, C 4. The volume of S is the sum of its degrees, 2 + 3 + 4 = 9.0, which counts "
        "the edge ends S holds, including its three edges to the rest.")
def volume(s: Scene) -> None:
    right = _set(s, "g.volume(S)", cut=False, badges="ABC")
    s.card_centered(right, ["degrees  2 + 3 + 4", ("volume   9.0", "accent")])


@figure("normalized_cut_size", "connectivity", "Normalized cut size",
        "The same graph and set S, with every degree in a badge. Three edges cross the "
        "cut (accent); S holds 9 edge ends and the rest 11, so the normalized cut is "
        "3 · (1/9 + 1/11) = 0.606.")
def normalized_cut_size(s: Scene) -> None:
    right = _set(s, "g.normalized_cut_size(S)", badges="ABCDEFG")
    s.card_centered(right, ["cut     3", "vol(S)  9    vol(T)  11",
                            ("3·(1/9 + 1/11)  0.606", "accent")],
                    title="cut · (1/vol(S) + 1/vol(T))")


@figure("conductance", "connectivity", "Conductance",
        "The same graph and set S, with every degree in a badge. Three edges cross the "
        "cut (accent), and the smaller side by volume is S with 9 edge ends against 11, "
        "so a third of S's edge ends leave it: conductance 3 / 9 = 0.333.")
def conductance(s: Scene) -> None:
    right = _set(s, "g.conductance(S)", badges="ABCDEFG")
    s.card_centered(right, ["cut       3", "min vol   min(9, 11) = 9", ("3 / 9     0.333", "accent")],
                    title="cut / min(vol(S), vol(T))")


@figure("edge_expansion", "connectivity", "Edge expansion",
        "The same graph and set S. Three edges cross the cut (accent) and the smaller "
        "side by node count is S, with 3 nodes against 4, so the edge expansion is "
        "3 / 3 = 1.0 crossing edge per member.")
def edge_expansion(s: Scene) -> None:
    right = _set(s, "g.edge_expansion(S)")
    s.card_centered(right, ["cut       3", "min size  min(3, 4) = 3", ("3 / 3     1.0", "accent")],
                    title="cut / min(|S|, |T|)")


@figure("mixing_expansion", "connectivity", "Mixing expansion",
        "The same graph and set S. Three of its ten edges cross the cut (accent), so 3 of "
        "the 20 edge ends in the graph lie on the cut: mixing expansion 3 / 20 = 0.15.")
def mixing_expansion(s: Scene) -> None:
    right = _set(s, "g.mixing_expansion(S)")
    s.card_centered(right, ["cut    3", "2m     2 · 10 = 20", ("3 / 20  0.15", "accent")],
                    title="cut / (2 · edges)")


@figure("node_expansion", "connectivity", "Node expansion",
        "The same graph and set S. Every neighbor of a member counts, members included: A, "
        "B and C are adjacent to each other, and D and E lie across the cut, all five in "
        "accent. Five neighbors over three members gives node expansion 5 / 3 = 1.667.")
def node_expansion(s: Scene) -> None:
    right = _set(s, "g.node_expansion(S)", cut=False, nodes="ABCDE")
    s.card_centered(right, ["neighbors  A, B, C, D, E", "|S|        3",
                            ("5 / 3      1.667", "accent")], title="|N(S)| / |S|, members too")


@figure("boundary_expansion", "connectivity", "Boundary expansion",
        "The same graph and set S. Only neighbors outside S count: D and E, in accent, "
        "each touch the triangle. Two outside neighbors over three members gives "
        "boundary expansion 2 / 3 = 0.667.")
def boundary_expansion(s: Scene) -> None:
    right = _set(s, "g.boundary_expansion(S)", cut=False, nodes="DE")
    s.card_centered(right, ["outside  D, E", "|S|      3", ("2 / 3    0.667", "accent")],
                    title="|N(S) − S| / |S|")


@figure("node_boundary", "connectivity", "Node boundary",
        "The same graph and set S. The nodes outside S with a neighbor inside it are D and "
        "E, in accent; F and G are further away. The call returns their node ids, 4 and 5.")
def node_boundary(s: Scene) -> None:
    right = _set(s, "g.node_boundary(S)", cut=False, nodes="DE")
    s.card_centered(right, ["boundary  D, E", ("ids       [4, 5]", "accent")])


@figure("edge_boundary", "connectivity", "Edge boundary",
        "The same graph and set S. The edges with exactly one end in S are B–D, C–D and "
        "C–E, in accent and labelled with their edge ids. The call returns those ids, 4, 5 "
        "and 6.")
def edge_boundary(s: Scene) -> None:
    right = _set(s, "g.edge_boundary(S)", edge_ids=True)
    s.card_centered(right, ["edges  B–D, C–D, C–E", ("ids    [4, 5, 6]", "accent")])


# --- spectral bisection on the two-triangle graph -------------------------------

# PAIR from graph_metrics.py: the graph fiedler_vector shades. The Fiedler
# sign gives the sides: negative D, E, F (g2, as in fiedler_vector) and
# non-negative A, B, C (g3); accent stays for the single cut edge C–D.


@figure("spectral_bisection", "connectivity", "Spectral bisection",
        "Two triangles joined by the edge C–D. Splitting the nodes by the sign of their "
        "Fiedler vector entries cuts only that edge (accent): the negative side D, E, F "
        "(teal) and the non-negative side A, B, C (violet), returned as the id lists "
        "[4, 5, 6] and [1, 2, 3].")
def spectral_bisection(s: Scene) -> None:
    left, right = compute(s, "g", "g.spectral_bisection()")
    sides = {**{k: "g3" for k in "ABC"}, **{k: "g2" for k in "DEF"}}
    s.graph(place(PAIR, left, unit=UNIT),
            [(a, b, {"style": sides[a] if sides[a] == sides[b] else "accent"})
             for a, b in PAIR_EDGES], styles=sides)
    s.hull(list("DEF"), "g2")
    s.hull(list("ABC"), "g3")
    s.card_centered(right, [("[4, 5, 6]  D, E, F", "g2"), ("[1, 2, 3]  A, B, C", "g3"),
                            ("cut       C–D", "accent")],
                    title="(negative, non-negative side)")
