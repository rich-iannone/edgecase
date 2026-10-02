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
"""

from core import Scene, compute, figure, place, transform

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
    # Not the base graph: get_cohesion() returns the minimum degree (an upper
    # bound), which is 2 on the base graph although its node connectivity is 1.
    # Here the bound is tight, so the picture and the returned value agree.
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
