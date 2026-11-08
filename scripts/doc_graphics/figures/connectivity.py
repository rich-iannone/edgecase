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
(the base graph has no bridges); `maximum_flow` uses a small directed network;
`node_percolation` returns a Graph, so it is a `transform` (before → after).

Later additions follow the same rules: `maximum_flow_value` and `minimum_cost_flow`
reuse the `maximum_flow` network; `edge_percolation` mirrors `node_percolation`;
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

The edge-connectivity partitions also use PAIR, whose bridge C–D is the
weak link: `chain_decomposition` colors each chain (nodes and edges) accent /
`g2` with the bridge `muted` (it is in no chain), and `k_edge_components`
shades the two 2-edge-connected triangles as accent / `g2` hulls with the
bridge muted. `k_edge_subgraphs` uses `contrast` on its docstring's K2,3
(hubs u, v; middle nodes a, b, c) to set the two notions side by side for
k = 3: the three edge-disjoint u–v paths (accent / g2 / g3) join u and v into
one component, but no edge runs inside {u, v}, so every subgraph is a single
node. `is_k_edge_connected` and `average_node_connectivity` use `contrast`
on six nodes like graph_metrics' algebraic_connectivity: the cycle (any one
edge can go; the two accent edges are a cut of size 2) and the cycle beside
the path (two node-disjoint A–D routes in accent / g2, then one, its missing
closing edge `removed`). Each figure computes its result with edgecase and
asserts it, so a library change breaks the build.
"""

from core import SIZE_NOTE, Box, Scene, compute, contrast, figure, place, ring, transform

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


@figure("articulation_points", "connectivity", "Articulation points",
        "Two triangles joined by the path C, D, E. The nodes C, D, and E are in the accent "
        "color: removing any one of them splits the graph. The card lists the articulation "
        "points C, D, E.")
def articulation_points(s: Scene) -> None:
    left, right = compute(s, "g", "g.articulation_points()")
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


@figure("edge_connectivity", "connectivity", "Edge connectivity",
        "Two diamonds that share the hub node D. The two accent edges B–D and C–D form a "
        "smallest edge cut: removing both disconnects the graph and no single edge does. "
        "The card shows that cut and the edge connectivity 2.")
def edge_connectivity(s: Scene) -> None:
    left, right = compute(s, "g", "g.edge_connectivity()")
    _base(s, left, edges=EDGE_CUT)
    s.card_centered(right, ["cut    B–D, C–D", ("value  2", "accent")])


@figure("node_connectivity", "connectivity", "Node connectivity",
        "Two diamonds that share the hub node D, drawn in the accent color. Removing D alone "
        "disconnects the graph, so the node connectivity is 1, smaller than the edge "
        "connectivity of 2. The card shows the cut D and the value 1.")
def node_connectivity(s: Scene) -> None:
    left, right = compute(s, "g", "g.node_connectivity()")
    _base(s, left, nodes=NODE_CUT)
    s.card_centered(right, ["cut    D", ("value  1", "accent")])


@figure("adhesion", "connectivity", "Adhesion",
        "Two diamonds that share the hub node D. The two accent edges B–D and C–D form a "
        "smallest edge cut, so the adhesion (edge connectivity) is 2. The card shows that "
        "cut and the value 2.")
def adhesion(s: Scene) -> None:
    left, right = compute(s, "g", "g.adhesion()")
    _base(s, left, edges=EDGE_CUT)
    s.card_centered(right, ["cut    B–D, C–D", ("value  2", "accent")])


@figure("cohesion", "connectivity", "Cohesion",
        "Two triangles joined through the two middle nodes D and E, drawn in the accent "
        "color. No single node disconnects the graph, but removing D and E together does, "
        "so the cohesion (node connectivity) is 2. The card shows the cut D, E and the value 2.")
def cohesion(s: Scene) -> None:
    # A 2-connected example, so this alias doesn't just repeat the node
    # connectivity figure (which shows the base graph's single cut node).
    left, right = compute(s, "g", "g.cohesion()")
    coords = {"A": (0, 0.6), "B": (1, 0), "C": (1, 1.2), "D": (2, 0), "E": (2, 1.2), "F": (3, 0),
              "G": (3, 1.2), "H": (4, 0.6)}
    edges = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"), ("D", "F"),
             ("E", "G"), ("F", "G"), ("F", "H"), ("G", "H")]
    s.graph(place(coords, left, unit=UNIT), edges, styles={"D": "accent", "E": "accent"})
    s.card_centered(right, ["cut    D, E", ("value  2", "accent")])


# --- pairwise (s, t) versions: always B and E on the base graph -------------


@figure("minimum_cut_value", "connectivity", "Minimum cut between two nodes",
        "Two diamonds that share the hub node D, with B and E outlined as source and sink. "
        "The two accent edges B–D and C–D are a smallest set of edges whose removal "
        "separates B from E. The card shows that cut and its size 2.")
def minimum_cut_value(s: Scene) -> None:
    left, right = compute(s, "g", 'g.minimum_cut_value("B", "E")')
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


@figure("node_connectivity_between", "connectivity", "Node connectivity between two nodes",
        "Two diamonds that share the hub node D, with B and E outlined as source and target. "
        "Every path from B to E passes through the accent hub D, so one node removal "
        "separates them. The card shows the cut D and the value 1.")
def node_connectivity_between(s: Scene) -> None:
    left, right = compute(s, "g", 'g.node_connectivity_between("B","E")')
    _base(s, left, nodes=NODE_CUT, pair=True)
    s.card_centered(right, ["cut    D", ("value  1", "accent")])


# --- flow ------------------------------------------------------------------


@figure("maximum_flow", "connectivity", "Maximum flow",
        "A directed network from source S to sink T through A and B. Each edge is labeled "
        "flow over capacity, and every edge carries flow, drawn in the accent color with "
        "width growing with the flow. The card lists the flow per edge, which totals 5.")
def maximum_flow(s: Scene) -> None:
    left, right = compute(s, "g", 'g.maximum_flow("S", "T")')
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

# The maximum_flow network: (from, to, capacity).
FLOW = {"S": (0, 0.8), "A": (1.5, 0), "B": (1.5, 1.6), "T": (3, 0.8)}
FLOW_NET = [("S", "A", 4), ("S", "B", 2), ("A", "B", 2), ("A", "T", 2), ("B", "T", 3)]


@figure("maximum_flow_value", "connectivity", "Maximum flow value",
        "The directed network from the maximum_flow figure, each edge labeled with its capacity. "
        "The accent edges A→T and B→T form a minimum cut separating source S from sink T, "
        "with capacity 2 + 3. The card shows that cut and the returned maximum flow value 5.0.")
def maximum_flow_value(s: Scene) -> None:
    left, right = compute(s, "g", 'g.maximum_flow_value("S", "T")')
    cut = {("A", "T"), ("B", "T")}  # S-side {S, A, B}: the only cut of capacity 5
    s.graph(place(FLOW, left, unit=UNIT),
            [(a, b, {"style": "accent", "weight": c}) if (a, b) in cut else (a, b, {"weight": c})
             for a, b, c in FLOW_NET],
            directed=True, styles={"S": "soft", "T": "soft"})
    s.card_centered(right, ["cut    A→T, B→T", ("value  2 + 3 = 5.0", "accent")])


@figure("minimum_cost_flow", "connectivity", "Minimum cost flow",
        "The maximum_flow network with a per-unit cost on each edge, labeled flow/capacity "
        "then cost. Sending a demand of 4 from S to T at the lowest cost fills S→A, then "
        "splits at A between the cheap detour A→B→T and the direct A→T; the costly S→B "
        "stays unused and faded. The card lists flow times cost per edge and the totals: "
        "flow 4, cost 14.")
def minimum_cost_flow(s: Scene) -> None:
    left, right = compute(s, "g", 'g.minimum_cost_flow("S", "T", demand=4)')
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


# --- later additions: maximum cut ------------------------------------------
#
# maximum_cut rebuilds its docstring example in edgecase and asserts the result.
# The sides of a 6-cycle alternate, so they are node colors (g2 / g3) rather
# than hulls; the cut edges are `accent`, and here every edge crosses.
HEX = ring(list("ABCDEF"), radius=1.0)
HEX_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "A")]


@figure("maximum_cut", "connectivity", "Maximum cut",
        "A six-node cycle A to F. The split puts alternate nodes on opposite sides: A, C, E "
        "on side S (teal) and B, D, F on side T (violet). Every one of the six edges then "
        "runs between the two sides, so all six are cut edges in the accent color. The card "
        "gives the cut value 6.0 and the two sides.")
def maximum_cut(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.maximum_cut(seed=1)")
    g = ec.Graph().add_nodes(list(HEX))
    g.add_edges(HEX_EDGES)
    value, (S, T) = g.maximum_cut(seed=1)
    side_s = sorted(g.get_label_from_id(n) for n in S)
    side_t = sorted(g.get_label_from_id(n) for n in T)
    assert value == 6.0 and side_s == ["A", "C", "E"] and side_t == ["B", "D", "F"]
    cut = [(a, b) for a, b in HEX_EDGES if (a in side_s) != (b in side_s)]
    assert len(cut) == value
    s.graph(place(HEX, left, unit=UNIT), [(a, b, {"style": "accent"}) for a, b in cut],
            styles={k: "g2" if k in side_s else "g3" for k in HEX})
    s.card_centered(right, [(f"value  {value:.1f}", "accent"),
                            ("S      " + ", ".join(side_s), "g2"),
                            ("T      " + ", ".join(side_t), "g3"),
                            f"cut    {len(cut)} of {len(HEX_EDGES)} edges"],
                    title="(value, (S, T))")


# --- later additions: edge-connectivity partitions -------------------------
#
# chain_decomposition and k_edge_components rebuild PAIR (the docstrings' two
# triangles joined by the bridge C–D) in edgecase and assert their results.


def _pair_graph():
    import edgecase as ec

    g = ec.Graph().add_nodes(list(PAIR))
    g.add_edges(PAIR_EDGES)
    return g


def _labels(g, ids) -> list[str]:
    return [g.get_label_from_id(n) for n in ids]


@figure("chain_decomposition", "connectivity", "Chain decomposition",
        "Two triangles joined by the bridge C–D. The depth-first search splits the edges into "
        "two chains, each a closed loop: A, C, B back to A (accent) and D, F, E back to D "
        "(teal). The bridge C–D, faded, lies on no cycle and so belongs to no chain. The card "
        "lists both chains as edge pairs in traversal order.")
def chain_decomposition(s: Scene) -> None:
    g = _pair_graph()
    chains = [[tuple(_labels(g, e)) for e in c] for c in g.chain_decomposition()]
    assert chains == [[("A", "C"), ("C", "B"), ("B", "A")], [("D", "F"), ("F", "E"), ("E", "D")]]
    left, right = compute(s, "g", "g.chain_decomposition()")
    colors = ("accent", "g2")
    edge_color = {frozenset(e): c for c, chain in zip(colors, chains) for e in chain}
    node_color = {k: c for c, chain in zip(colors, chains) for e in chain for k in e}
    s.graph(place(PAIR, left, unit=UNIT),
            [(a, b, {"style": edge_color.get(frozenset((a, b)), "muted")}) for a, b in PAIR_EDGES],
            styles=node_color)
    s.card_centered(right, [("[" + ", ".join(f"({a}, {b})" for a, b in chain) + "]", c)
                            for c, chain in zip(colors, chains)] + ["C–D  in no chain"],
                    title="chains, as (u, v) pairs")


@figure("k_edge_components", "connectivity", "k-edge-connected components",
        "Two triangles joined by the bridge C–D. With k = 2, nodes belong together when two "
        "edge-disjoint paths join them: the triangle A, B, C (accent region) and the triangle "
        "D, E, F (teal region) qualify, while the faded bridge offers only one path between "
        "them. The card lists the two components; with k = 3 all 6 nodes are separate.")
def k_edge_components(s: Scene) -> None:
    g = _pair_graph()
    comps = [_labels(g, c) for c in g.k_edge_components(2)]
    assert comps == [["A", "B", "C"], ["D", "E", "F"]]
    assert len(g.k_edge_components(3)) == 6
    left, right = compute(s, "g", "g.k_edge_components(2)")
    colors = dict(zip(("ABC", "DEF"), ("accent", "g2")))
    group = {k: c for ks, c in colors.items() for k in ks}
    s.graph(place(PAIR, left, unit=UNIT),
            [(a, b, {"style": group[a] if group[a] == group[b] else "muted"})
             for a, b in PAIR_EDGES], styles=group)
    for ks, c in colors.items():
        s.hull(list(ks), c)
    s.card_centered(right, [(f"[{', '.join(c)}]", colors["".join(c)]) for c in comps]
                    + ["k = 3  6 singletons"], title="k = 2 components")


# k_edge_subgraphs docstring: hubs u and v joined through a, b and c (K2,3).
K23 = {"u": (0, 1), "a": (1, 0), "b": (1, 1), "c": (1, 2), "v": (2, 1)}
K23_EDGES = [("u", "a"), ("a", "v"), ("u", "b"), ("b", "v"), ("u", "c"), ("c", "v")]


@figure("k_edge_subgraphs", "connectivity", "k-edge-connected subgraphs",
        "Both sides show hubs u and v joined through a, b and c, with k = 3. Left: "
        "k_edge_components(3) counts paths through the whole graph, and the three "
        "edge-disjoint u–v paths (accent, teal, violet) put u and v in one component, with "
        "a, b and c alone. Right: k_edge_subgraphs(3) counts only edges inside each group, "
        "and u and v are not even adjacent, so every subgraph is a single node.")
def k_edge_subgraphs(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(["u", "v", "a", "b", "c"])  # the docstring's node order
    g.add_edges(K23_EDGES)
    comps = [_labels(g, c) for c in g.k_edge_components(3)]
    subs = [_labels(g, c) for c in g.k_edge_subgraphs(3)]
    assert comps == [["u", "v"], ["a"], ["b"], ["c"]]
    assert subs == [["u"], ["v"], ["a"], ["b"], ["c"]]
    left, right = contrast(s, "g.k_edge_components(3)", "g.k_edge_subgraphs(3)",
                           "[[u, v], [a], [b], [c]]", "5 singletons")
    paths = dict(zip("abc", ("accent", "g2", "g3")))
    for box, p, note in ((left, "l.", "3 edge-disjoint u–v paths"),
                         (right, "r.", "no edge inside {u, v}")):
        s.text(box.cx, box.y0 + 8, note, size=SIZE_NOTE, opacity=0.6)
        gbox = Box(box.x0, box.y0 + 22, box.x1, box.y1)
        if p == "l.":
            edges = [(a, b, {"style": paths[a if a in paths else b]}) for a, b in K23_EDGES]
            styles = {"u": "accent", "v": "accent"}
        else:
            edges, styles = K23_EDGES, {}
        s.graph(place(K23, gbox, unit=UNIT), edges, styles=styles, prefix=p)


# is_k_edge_connected docstring: the six-node cycle, for k = 2 and k = 3.
HEX_CUT = {("B", "C"), ("E", "F")}  # two opposite edges: a cut of size 2


@figure("is_k_edge_connected", "connectivity", "Is k-edge-connected",
        "The same six-node cycle on both sides. Left: removing any single edge leaves a path, "
        "so the cycle is 2-edge-connected and is_k_edge_connected(2) returns True. Right: "
        "removing the two accent edges B–C and E–F splits it in two, so fewer than 3 edge "
        "removals disconnect it and is_k_edge_connected(3) returns False.")
def is_k_edge_connected(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(list(HEX))
    g.add_edges(HEX_EDGES)
    assert g.is_k_edge_connected(2) is True and g.is_k_edge_connected(3) is False
    assert g.edge_connectivity() == len(HEX_CUT)
    left, right = contrast(s, "g.is_k_edge_connected(2)", "g.is_k_edge_connected(3)")
    for box, p, note, cut in ((left, "l.", "any 1 edge can go", set()),
                              (right, "r.", "2 edges cut it in two", HEX_CUT)):
        s.text(box.cx, box.y0 + 8, note, size=SIZE_NOTE, opacity=0.6)
        gbox = Box(box.x0, box.y0 + 22, box.x1, box.y1)
        s.graph(place(HEX, gbox, unit=UNIT),
                [(a, b, {"style": "accent"}) if (a, b) in cut else (a, b) for a, b in HEX_EDGES],
                prefix=p)


@figure("average_node_connectivity", "connectivity", "Average node connectivity",
        "Left: a six-node cycle, where every pair of nodes, such as A and D (outlined), is "
        "joined by two routes that share no other node (accent and teal), so the average "
        "node connectivity is 2.0. Right: the same nodes without the closing edge F–A "
        "(dashed), a path, where every pair has only one route and the average is 1.0.")
def average_node_connectivity(s: Scene) -> None:
    import edgecase as ec

    cycle = ec.Graph().add_cycle_graph(6)
    path = ec.Graph().add_path_graph(6)
    assert cycle.average_node_connectivity() == 2.0 and path.average_node_connectivity() == 1.0
    assert ec.Graph().add_full_graph(5).average_node_connectivity() == 4.0
    left, right = contrast(s, "g.average_node_connectivity()", "g.average_node_connectivity()",
                           "2.0", "1.0")
    upper = {("A", "B"), ("B", "C"), ("C", "D")}
    for box, p, note in ((left, "l.", "A to D: 2 disjoint routes"),
                         (right, "r.", "A to D: 1 route")):
        s.text(box.cx, box.y0 + 8, note, size=SIZE_NOTE, opacity=0.6)
        gbox = Box(box.x0, box.y0 + 22, box.x1, box.y1)
        edges = []
        for a, b in HEX_EDGES:
            if (a, b) in upper:
                st = "accent"
            elif (a, b) == ("F", "A") and p == "r.":
                st = "removed"
            else:
                st = "g2" if p == "l." else "muted"
            edges.append((a, b, {"style": st}))
        s.graph(place(HEX, gbox, unit=UNIT), edges,
                styles={"A": "soft", "D": "soft"}, prefix=p)


# --- later additions: pairwise tables, every minimum cut, and augmentation ------------
#
# `all_pairs_node_connectivity` draws its docstring's square with one diagonal beside the
# table it returns (the diagonal's ends, the only pair with three disjoint routes, in
# accent). `all_node_cuts` uses a strip of four triangles, whose three minimum cuts slide
# along it: one miniature per cut, its nodes `accent`, the edges they would take with them
# `removed`, and a card listing the cuts. `is_locally_k_edge_connected` is a `contrast` on
# PAIR like `is_k_edge_connected`: two edge-disjoint routes inside a triangle (accent / g2)
# against the bridge every route across must use. `k_edge_augmentation` is a `transform` on
# a binary tree: the edges it adds are `ghost` (dashed accent) and the leaves they pair up,
# which are the bridge tree's leaves here, are `soft`. Each figure asserts its result.

SQUARE = {"A": (0, 0), "B": (1.2, 0), "C": (1.2, 1.2), "D": (0, 1.2)}
SQUARE_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]


@figure("all_pairs_node_connectivity", "connectivity", "All-pairs node connectivity",
        "A square A, B, C, D with the diagonal A–C (accent). The table gives the local node "
        "connectivity of every pair: A and C are joined directly and through B and through "
        "D, three routes that share no other node (accent cells), while every other pair has "
        "two.")
def all_pairs_node_connectivity(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(list(SQUARE)).add_edges(SQUARE_EDGES)
    lab = g.get_label_from_id
    table = {lab(u): {lab(v): k for v, k in row.items()} for u, row in
             g.all_pairs_node_connectivity().items()}
    assert table["A"]["C"] == 3 and all(
        k == 2 for u, row in table.items() for v, k in row.items() if {u, v} != {"A", "C"})
    left, right = compute(s, "g", "g.all_pairs_node_connectivity()")
    s.graph(place(SQUARE, left, unit=UNIT),
            [(a, b, {"style": "accent"}) if (a, b) == ("A", "C") else (a, b)
             for a, b in SQUARE_EDGES], styles={"A": "soft", "C": "soft"})
    names = list(SQUARE)
    rows = [[""] + names] + [[u] + ["–" if u == v else str(table[u][v]) for v in names]
                             for u in names]
    hl = {(1, 3): "accent", (3, 1): "accent"}
    s.table_centered(right, rows, [36] * 5, highlight=hl)


TRI = {"A": (0, 1), "B": (0.5, 0), "C": (1, 1), "D": (1.5, 0), "E": (2, 1), "F": (2.5, 0)}
TRI_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("C", "E"),
             ("D", "E"), ("D", "F"), ("E", "F")]


@figure("all_node_cuts", "connectivity", "All minimum node cuts",
        "Three copies of a strip of four triangles on A to F, whose node connectivity is 2. "
        "Each copy shows one minimum node cut in the accent color, with the edges it removes "
        "dashed: B, C cuts off A; C, D separates A, B from E, F; and D, E cuts off F. The "
        "card lists the three cuts.")
def all_node_cuts(s: Scene) -> None:
    import edgecase as ec
    from core import R_SMALL, SIZE_CARD

    g = ec.Graph().add_nodes(list(TRI)).add_edges(TRI_EDGES)
    cuts = [[g.get_label_from_id(n) for n in c] for c in g.all_node_cuts()]
    assert cuts == [["B", "C"], ["C", "D"], ["D", "E"]]
    left, right = compute(s, "g", "g.all_node_cuts()")
    pitch = left.w / 3
    for i, cut in enumerate(cuts):
        cx = left.x0 + pitch * (i + 0.5)
        cell = Box(cx - pitch / 2 + 4, left.y0 + 24, cx + pitch / 2 - 4, left.y1 - 30)
        edges = [(a, b, {"style": "removed"}) if {a, b} & set(cut) else (a, b)
                 for a, b in TRI_EDGES]
        s.graph(place(TRI, cell, unit=44, r=R_SMALL, pad=2), edges,
                styles={k: "accent" for k in cut}, r=R_SMALL, prefix=f"m{i}.")
        s.text(cx, left.y1 - 14, f"[{', '.join(cut)}]", size=SIZE_CARD, color="accent",
               weight=600, role="card")
    s.card_centered(right, [(f"[{', '.join(c)}]", "accent") for c in cuts],
                    title="3 cuts of size 2")


@figure("is_locally_k_edge_connected", "connectivity", "Is locally k-edge-connected",
        "Two triangles joined by the bridge C–D, twice. Left: A and C (outlined) are joined "
        "by the edge A–C (accent) and by the route A, B, C (teal), two paths with no edge in "
        "common, so is_locally_k_edge_connected(A, C, 2) returns True. Right: every route "
        "from A to E crosses the bridge C–D (accent), so a single edge separates them and the "
        "call returns False.")
def is_locally_k_edge_connected(s: Scene) -> None:
    g = _pair_graph()
    assert g.is_locally_k_edge_connected("A", "C", 2) is True
    assert g.is_locally_k_edge_connected("A", "E", 2) is False
    left, right = contrast(s, "g.is_locally_k_edge_connected(A, C, 2)",
                           "g.is_locally_k_edge_connected(A, E, 2)")
    route = {("A", "C"): "accent", ("A", "B"): "g2", ("B", "C"): "g2"}
    for box, p, note, ends, styles in (
            (left, "l.", "2 edge-disjoint A–C paths", ("A", "C"), route),
            (right, "r.", "every A–E path uses C–D", ("A", "E"), {("C", "D"): "accent"})):
        s.text(box.cx, box.y0 + 8, note, size=SIZE_NOTE, opacity=0.6)
        gbox = Box(box.x0, box.y0 + 22, box.x1, box.y1)
        s.graph(place(PAIR, gbox, unit=UNIT),
                [(a, b, {"style": styles[(a, b)]}) if (a, b) in styles else (a, b)
                 for a, b in PAIR_EDGES],
                styles={k: "soft" for k in ends}, prefix=p)


BTREE = {"A": (1.5, 0), "B": (0.5, 1), "C": (2.5, 1), "D": (0, 2), "E": (1, 2), "F": (2, 2),
         "G": (3, 2)}
BTREE_EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")]


@figure("k_edge_augmentation", "connectivity", "k-edge augmentation",
        "Left: a binary tree on seven nodes, in which every edge is a bridge. Right: "
        "k_edge_augmentation(2) adds the two dashed accent edges D–F and E–G, pairing up the "
        "four leaves (outlined), after which every edge lies on a cycle and no single edge "
        "removal disconnects the tree; with four leaves, two new edges is the fewest "
        "possible.")
def k_edge_augmentation(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(list(BTREE)).add_edges(BTREE_EDGES)
    added = [tuple(g.get_label_from_id(n) for n in e) for e in g.k_edge_augmentation(2)]
    assert added == [("D", "F"), ("E", "G")]
    left, right = transform(s, "g", "g.k_edge_augmentation(2)")
    s.graph(place(BTREE, Box(left.x0, left.y0, left.x1, left.y1 - 24), unit=UNIT),
            BTREE_EDGES, prefix="l.")
    s.graph(place(BTREE, Box(right.x0, right.y0, right.x1, right.y1 - 24), unit=UNIT),
            BTREE_EDGES, styles={k: "soft" for k in "DEFG"}, prefix="r.")
    s.edge("r.D", "r.F", style="ghost", bend=26)
    s.edge("r.E", "r.G", style="ghost", bend=26)
