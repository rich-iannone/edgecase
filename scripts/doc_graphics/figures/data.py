"""Data access: lookups, ids, tables, and structural edge counts.

Template: `compute` (lookups that return a bool use `contrast`: a True
query beside a False one). Left: a small graph; right: the returned value as
a card, or a compact table with the real DataFrame column names.

Every id/label/lookup figure uses one shared four-node diamond, so the
id <-> label mapping is the same everywhere: node ids are outlined pills
just outside each node, edge ids are small boxed numbers on the edges. Ids
are ink when they are context and accent when they are what the call is
about. The query (a node, an edge, an id) is `soft`; what comes back is
`accent`. The counting methods use a row of four nodes (self-loops and
parallel edges) or the directed diamond (reciprocal pairs), with the counted
edges in `accent` and the count or list in the card. The attribute getters
keep the id pills / edge-id boxes (the result is keyed by id) and show the
values only in the card; elements lacking the attribute are `muted`.
"""

import math

from core import EMPH_STROKE, SIZE_NOTE, Box, Scene, compute, contrast, figure, place

UNIT = 100  # one graph scale for the whole family

# Shared id/label graph: two triangles sharing B–C. Node ids follow insertion
# order (A=1 … D=4); edge ids follow the edge list order (1 … 5).
DIAMOND = {"A": (0, 0.5), "B": (1, 0), "C": (1, 1), "D": (2, 0.5)}
D_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D")]
NODE_ID = {"A": 1, "B": 2, "C": 3, "D": 4}
BADGE_AT = {"A": "w", "B": "n", "C": "s", "D": "e"}  # radial, clear of every edge

# Row graph for self-loops and parallel edges.
ROW = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0)}


def _ids(s: Scene, accent: str = "", prefix: str = "") -> None:
    """Node-id pills; nodes named in `accent` get accent pills, the rest ink."""
    for k, where in BADGE_AT.items():
        if prefix + k in s.nodes:
            s.badge(prefix + k, str(NODE_ID[k]), where=where,
                    color="accent" if k in accent else "ink")


def _edge_label(s: Scene, i: int, text: str, accent: bool) -> None:
    """Small boxed edge id at the midpoint (apex, for bent edges) of edge i."""
    x, y = s.edge_point(s.edges[i], 0.5)
    s.text(x, y, text, size=SIZE_NOTE, boxed=True, role="weight", owner=f"edge{i}",
           color="accent" if accent else "ink", weight=600 if accent else 400)


def _diamond(s: Scene, box, *, styles=None, edge_styles=None, directed=False,
             extra=(), coords=DIAMOND, prefix: str = "") -> dict:
    """The shared diamond; `extra` appends edges (a, b, opts) after the base five."""
    edge_styles = edge_styles or {}
    edges = [(a, b, {"style": edge_styles.get((a, b), "default")}) for a, b in D_EDGES]
    pos = place(coords, box, unit=UNIT)
    pos = {k: v for k, v in pos.items() if k in DIAMOND}
    s.graph(pos, edges + list(extra), directed=directed, styles=styles or {}, prefix=prefix)
    return pos


def _edge_ids(s: Scene, accent=(), start: int = 0) -> None:
    """Label edges start.. with ids 1..; ids in `accent` are highlighted."""
    for i in range(start, len(s.edges)):
        eid = i - start + 1
        _edge_label(s, i, str(eid), eid in accent)


# -- self-loops (not drawable at an angle by core, so drawn here) -------------


def _loop(s: Scene, nid: str, angle: float = 0, label: str | None = None) -> None:
    """Accent self-loop teardrop on `nid`, tilted `angle` degrees from straight up.

    Same shape as core's self-loop edge, but rotatable so a node can carry two
    loops side by side.
    """
    n = s.nodes[nid]
    r, lr = n.r, n.r * 0.95
    pts = [(-0.55 * r, -0.83 * r), (-1.9 * lr, -(r + 2.3 * lr)),
           (1.9 * lr, -(r + 2.3 * lr)), (0.55 * r, -0.83 * r)]
    a = math.radians(angle)
    rot = [(n.x + x * math.cos(a) - y * math.sin(a), n.y + x * math.sin(a) + y * math.cos(a))
           for x, y in pts]
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = rot
    s.path(f"M {x0:.2f} {y0:.2f} C {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f}",
           color="accent", width=EMPH_STROKE)
    if label is not None:
        ax = 0.125 * x0 + 0.375 * x1 + 0.375 * x2 + 0.125 * x3
        ay = 0.125 * y0 + 0.375 * y1 + 0.375 * y2 + 0.125 * y3
        s.text(ax, ay, label, size=SIZE_NOTE, boxed=True, color="accent", weight=600)


def _loops_graph(s: Scene, box, *, ids: bool = False, edge_ids: bool = False) -> None:
    """Path A–B–C–D; two loops on A, one on C (edge ids 4, 5, 6)."""
    low = Box(box.x0, box.y0 + 36, box.x1, box.y1)  # loops rise ~40px; balance them
    s.graph(place(ROW, low, unit=UNIT), [("A", "B"), ("B", "C"), ("C", "D")])
    if edge_ids:
        _edge_ids(s)
    _loop(s, "A", -42, "4" if edge_ids else None)
    _loop(s, "A", 42, "5" if edge_ids else None)
    _loop(s, "C", 0, "6" if edge_ids else None)
    if ids:
        for k in "ABCD":
            s.badge(k, str(NODE_ID[k]), where="s", color="accent" if k in "AC" else "ink")


# Parallel edges: A–B twice (ids 1, 2), B–C three times (3, 4, 5), C–D once (6).
MULTI = [("A", "B", 12), ("A", "B", -12), ("B", "C", 20), ("B", "C", 0), ("B", "C", -20),
         ("C", "D", 0)]


def _multi_graph(s: Scene, box, styles: list[str], *, ids: bool = False,
                 edge_ids: bool = False) -> None:
    edges = [(a, b, {"bend": bend, "style": st}) for (a, b, bend), st in zip(MULTI, styles)]
    s.graph(place(ROW, box, unit=UNIT), edges)
    if edge_ids:
        _edge_ids(s, accent={i + 1 for i, st in enumerate(styles) if st == "accent"})
    if ids:
        for k in "ABCD":
            s.badge(k, str(NODE_ID[k]), where="s", color="accent" if k in "ABC" else "ink")


# Directed diamond with two reciprocal pairs: B→A (id 6) and D→C (id 7) added.
RECIP = [("B", "A", {"bend": 14}), ("D", "C", {"bend": 14})]
MUTUAL = {("A", "B"), ("C", "D")}


def _mutual_graph(s: Scene, box, *, mutual: str, oneway: str, edge_ids: bool = False) -> None:
    edge_styles = {e: (mutual if e in MUTUAL else oneway) for e in D_EDGES}
    extra = [(a, b, {**o, "style": mutual}) for a, b, o in RECIP]
    _diamond(s, box, directed=True, edge_styles=edge_styles, extra=extra)
    # the forward half of each reciprocal pair bows the other way
    for e in s.edges:
        if (e.a, e.b) in MUTUAL:
            e.bend = 14
    if edge_ids:
        _edge_ids(s, accent={1, 5, 6, 7})


# -- tables -------------------------------------------------------------------


@figure("get_node_df", "data", "Node DataFrame",
        "The shared four-node graph A, B, C, D, each node tagged with its integer id 1 to 4. "
        "The result is a table with the columns id, label, type, and degree: one row per "
        "node sorted by id, type empty (null), and degrees 2, 3, 3, 2.")
def get_node_df(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_node_df()")
    _diamond(s, left)
    _ids(s)
    rows = [["id", "label", "type", "degree"]]
    rows += [[str(NODE_ID[k]), k, "null", str(d)] for k, d in zip("ABCD", (2, 3, 3, 2))]
    s.table_centered(right, rows, [40, 52, 52, 60])


@figure("get_edge_df", "data", "Edge DataFrame",
        "The shared four-node graph with node ids 1 to 4 on the nodes and edge ids 1 to 5 on "
        "the edges. The result is a table with the columns id, source, target, and rel: one "
        "row per edge sorted by id, with endpoints given as node ids and rel empty (null).")
def get_edge_df(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_edge_df()")
    _diamond(s, left)
    _edge_ids(s)
    _ids(s)
    rows = [["id", "source", "target", "rel"]]
    rows += [[str(i), str(NODE_ID[a]), str(NODE_ID[b]), "null"]
             for i, (a, b) in enumerate(D_EDGES, 1)]
    s.table_centered(right, rows, [40, 60, 60, 48])


@figure("get_edges_as_tuples", "data", "Edges as tuples",
        "The shared four-node graph with node ids 1 to 4 and edge ids 1 to 5. The card lists "
        "one (id, source, target) tuple per edge, from (1, 1, 2) for edge 1 between A and B "
        "to (5, 3, 4) for edge 5 between C and D.")
def get_edges_as_tuples(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_edges_as_tuples()")
    _diamond(s, left)
    _edge_ids(s)
    _ids(s)
    s.card_centered(right, [f"({i}, {NODE_ID[a]}, {NODE_ID[b]})"
                            for i, (a, b) in enumerate(D_EDGES, 1)],
                    title="(id, source, target)")


@figure("get_edges_str_repr", "data", "Edges as strings",
        "The shared four-node graph with node ids 1 to 4 in pills; the edge between B and C "
        "is outlined in the accent color. The table lists one string per edge, 'source--target', "
        "using node ids by default ('1--2' to '3--4') and labels with use_labels=True ('A--B' "
        "to 'C--D'); the B–C row, '2--3' and 'B--C', is highlighted.")
def get_edges_str_repr(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_edges_str_repr()")
    _diamond(s, left, edge_styles={("B", "C"): "soft"})
    _ids(s, accent="BC")
    rows = [["default", "use_labels=True"]] + [
        [f"'{NODE_ID[a]}--{NODE_ID[b]}'", f"'{a}--{b}'"] for a, b in D_EDGES
    ]
    bc = D_EDGES.index(("B", "C")) + 1
    s.table_centered(right, rows, [100, 120], highlight={(bc, 0): "accent", (bc, 1): "accent"})


@figure("get_graph_name", "data", "Graph name",
        "The shared four-node graph, tagged with the name it was created with, "
        "graph_name roads, shown as a tag in the accent color. The card returns that name, "
        "'roads', and a note below it says a graph without a name returns None.")
def get_graph_name(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_graph_name()")
    pos = _diamond(s, left)
    top = min(y for _, y in pos.values())
    s.text(left.cx, top - 34, 'graph_name="roads"', size=SIZE_NOTE, boxed=True,
           color="accent", weight=600)
    card = s.card_centered(right, [("'roads'", "accent")])
    s.text(right.cx, card.y + card.h + 16, "None if the graph has no name", size=SIZE_NOTE,
           opacity=0.6)


# -- id lists -----------------------------------------------------------------


@figure("get_node_ids", "data", "Node ids",
        "The shared four-node graph with each node's integer id in an accent pill: A is 1, "
        "B is 2, C is 3, D is 4. The card returns all four ids in internal storage order, "
        "3, 1, 4, 2, which is not sorted.")
def get_node_ids(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_node_ids()")
    _diamond(s, left)
    _ids(s, accent="ABCD")
    s.card_centered(right, [("[3, 1, 4, 2]", "accent")], title="node ids")


@figure("get_edge_ids", "data", "Edge ids",
        "The shared four-node graph with each edge's integer id, 1 to 5, in the accent color "
        "on the edge. The card returns the list 1, 2, 3, 4, 5.")
def get_edge_ids(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_edge_ids()")
    _diamond(s, left)
    _edge_ids(s, accent={1, 2, 3, 4, 5})
    s.card_centered(right, [("[1, 2, 3, 4, 5]", "accent")], title="edge ids")


@figure("get_node_creation_order", "data", "Node creation order",
        "The shared four-node graph, added in the order A, B, C, D, with each node's id in an "
        "accent pill. The card returns the node ids from oldest to newest: 1, 2, 3, 4.")
def get_node_creation_order(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_node_creation_order()")
    _diamond(s, left)
    _ids(s, accent="ABCD")
    s.card_centered(right, [("[1, 2, 3, 4]", "accent")], title="oldest → newest")


@figure("get_edge_creation_order", "data", "Edge creation order",
        "The shared four-node graph with each edge's id, 1 to 5, in the accent color; edge 1 "
        "(A to B) was added first and edge 5 (C to D) last. The card returns the edge ids "
        "from oldest to newest: 1, 2, 3, 4, 5.")
def get_edge_creation_order(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_edge_creation_order()")
    _diamond(s, left)
    _edge_ids(s, accent={1, 2, 3, 4, 5})
    s.card_centered(right, [("[1, 2, 3, 4, 5]", "accent")], title="oldest → newest")


# -- single lookups -----------------------------------------------------------


@figure("get_node", "data", "Get node",
        "The shared four-node graph with node ids in pills. Node B, the query, is outlined in "
        "the accent color with its id 2 highlighted. The card returns the Node object "
        "Node(id=2, label='B').")
def get_node(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_node("B")')
    _diamond(s, left, styles={"B": "soft"})
    _ids(s, accent="B")
    s.card_centered(right, [("Node(id=2, label='B')", "accent")])


@figure("get_edge", "data", "Get edge",
        "The shared four-node graph with node ids in pills and edge ids on the edges. The "
        "queried edge 3, between B and C, is in the accent color, as are the ids 2 and 3 of "
        "its endpoints. The card returns the Edge object Edge(id=3, 2 -> 3).")
def get_edge(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_edge(3)")
    _diamond(s, left, edge_styles={("B", "C"): "soft"})
    _edge_ids(s, accent={3})
    _ids(s, accent="BC")
    s.card_centered(right, [("Edge(id=3, 2 -> 3)", "accent")])


@figure("get_node_ids_by_edge_id", "data", "Node ids by edge id",
        "The shared four-node graph with node ids in pills and edge ids on the edges. The "
        "queried edge 3, between B and C, is in the accent color, and so are its endpoints B "
        "and C with ids 2 and 3. The card returns the (source, target) pair (2, 3).")
def get_node_ids_by_edge_id(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_node_ids_by_edge_id(3)")
    _diamond(s, left, styles={"B": "accent", "C": "accent"}, edge_styles={("B", "C"): "soft"})
    _edge_ids(s, accent={3})
    _ids(s, accent="BC")
    s.card_centered(right, [("(2, 3)", "accent")], title="(source, target)")


@figure("get_id_from_label", "data", "Id from label",
        "The shared four-node graph with node ids in pills. The queried label C is outlined in "
        "the accent color and its id pill, 3, is highlighted. The card returns 3.")
def get_id_from_label(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_id_from_label("C")')
    _diamond(s, left, styles={"C": "soft"})
    _ids(s, accent="C")
    s.card_centered(right, [("3", "accent")], title="node id")


@figure("get_label_from_id", "data", "Label from id",
        "The shared four-node graph with node ids in pills. The queried id 2 is highlighted, "
        "and the node it belongs to, B, is filled in the accent color. The card returns the "
        "label B.")
def get_label_from_id(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_label_from_id(2)")
    _diamond(s, left, styles={"B": "accent"})
    _ids(s, accent="B")
    s.card_centered(right, [("'B'", "accent")], title="label")


@figure("find_edge_ids", "data", "Find edge ids",
        "The shared four-node graph with a second, parallel edge between A and B. The query "
        "nodes A and B are outlined in the accent color, and both edges between them, ids 1 "
        "and 6, are highlighted. The card returns 1 and 6.")
def find_edge_ids(s: Scene) -> None:
    left, right = compute(s, "g", 'g.find_edge_ids("A", "B")')
    _diamond(s, left, styles={"A": "soft", "B": "soft"}, edge_styles={("A", "B"): "accent"},
             extra=[("A", "B", {"style": "accent", "bend": -12})])
    s.edges[0].bend = 12
    _edge_ids(s, accent={1, 6})
    s.card_centered(right, [("[1, 6]", "accent")], title="edge ids")


# -- membership (bool) --------------------------------------------------------


@figure("has_node", "data", "Has node",
        "Left: the shared four-node graph queried for C, which is outlined in the accent color; "
        "has_node returns True. Right: the same graph queried for E, drawn as a dashed outline "
        "where it would be; there is no such node, so has_node returns False.")
def has_node(s: Scene) -> None:
    left, right = contrast(s, 'g.has_node("C")', 'g.has_node("E")')
    coords = {**DIAMOND, "E": (2.7, 0.5)}  # place with E so both diamonds line up
    _diamond(s, left, styles={"C": "soft"}, coords=coords, prefix="l.")
    _diamond(s, right, coords=coords, prefix="r.")
    s.node("r.E", *place(coords, right, unit=UNIT)["E"], label="E", style="removed")


@figure("has_edge", "data", "Has edge",
        "The shared graph, directed. Left: querying A to B finds the accent edge from A to B, "
        "so has_edge returns True. Right: querying B to A, the reverse direction, finds only a "
        "dashed outline where that edge would be, so has_edge returns False.")
def has_edge(s: Scene) -> None:
    left, right = contrast(s, 'g.has_edge("A", "B")', 'g.has_edge("B", "A")')
    q = {"A": "soft", "B": "soft"}
    _diamond(s, left, styles=q, directed=True, edge_styles={("A", "B"): "accent"}, prefix="l.")
    _diamond(s, right, styles=q, directed=True, prefix="r.",
             extra=[("B", "A", {"style": "removed", "bend": 14})])
    s.edges[5].bend = 14  # the existing A→B bows away from the missing B→A


@figure("has_edge_id", "data", "Has edge id",
        "The shared four-node graph with edge ids 1 to 5 on the edges. Left: querying edge id "
        "3 finds the accent edge between B and C, so has_edge_id returns True. Right: no edge "
        "carries id 9, so has_edge_id returns False.")
def has_edge_id(s: Scene) -> None:
    left, right = contrast(s, "g.has_edge_id(3)", "g.has_edge_id(9)")
    _diamond(s, left, edge_styles={("B", "C"): "accent"}, prefix="l.")
    _edge_ids(s, accent={3})
    n = len(s.edges)
    _diamond(s, right, prefix="r.")
    _edge_ids(s, start=n)


# -- self-loops ---------------------------------------------------------------


@figure("count_loops", "data", "Count loops per node",
        "Four nodes in a row, A to D, with node ids 1 to 4 below them. Node A carries two "
        "self-loops and node C one, all in the accent color. The card maps node id to loop "
        "count: node 3 has 1 and node 1 has 2.")
def count_loops(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_loops()")
    _loops_graph(s, left, ids=True)
    s.card_centered(right, [("{3: 1, 1: 2}", "accent")], title="node id → loops")


@figure("count_all_loops", "data", "Count all loops",
        "Four nodes in a row, A to D, with two self-loops on A and one on C, all in the accent "
        "color. The card counts every self-loop edge in the graph: 2 + 1 = 3.")
def count_all_loops(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_all_loops()")
    _loops_graph(s, left)
    s.card_centered(right, ["A  2", "C  1", ("total  3", "accent")], title="self-loops")


@figure("which_loops", "data", "Which loops",
        "Four nodes in a row joined by edges 1, 2, 3, with self-loops 4 and 5 on A and 6 on C "
        "in the accent color. The card returns the self-loop edge ids 4, 5, 6.")
def which_loops(s: Scene) -> None:
    left, right = compute(s, "g", "g.which_loops()")
    _loops_graph(s, left, edge_ids=True)
    s.card_centered(right, [("[4, 5, 6]", "accent")], title="edge ids")


# -- parallel edges -----------------------------------------------------------


@figure("count_multi_edges", "data", "Count multi-edges",
        "Four nodes in a row with two parallel edges between A and B, three between B and C, "
        "and one between C and D. For each pair the first edge is ink and the extras are in "
        "the accent color; the card counts the extras: 1 + 2 = 3.")
def count_multi_edges(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_multi_edges()")
    _multi_graph(s, left, ["default", "accent", "accent", "default", "accent", "default"])
    s.card_centered(right, ["A–B  1 extra", "B–C  2 extra", ("total  3", "accent")])


@figure("count_multiple", "data", "Count multiple edges",
        "Four nodes in a row with node ids 1 to 4: two parallel edges between A and B and three "
        "between B and C, in the accent color, and a single edge between C and D. The card maps "
        "each repeated (source, target) id pair to its edge count: (2, 3) has 3, (1, 2) has 2.")
def count_multiple(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_multiple()")
    _multi_graph(s, left, ["accent"] * 5 + ["default"], ids=True)
    s.card_centered(right, [("{(2, 3): 3,", "accent"), ("  (1, 2): 2}", "accent")],
                    title="(source, target) → count")


@figure("which_multiple", "data", "Which multiple edges",
        "Four nodes in a row with edge ids on every edge: edges 1 and 2 run between A and B, "
        "edges 3, 4, 5 between B and C, all in the accent color, and edge 6 alone between C "
        "and D. The card returns every edge in a repeated pair, first ones included: 1 to 5.")
def which_multiple(s: Scene) -> None:
    left, right = compute(s, "g", "g.which_multiple()")
    _multi_graph(s, left, ["accent"] * 5 + ["default"], edge_ids=True)
    s.card_centered(right, [("[1, 2, 3, 4, 5]", "accent")], title="edge ids")


# -- reciprocity --------------------------------------------------------------


@figure("which_mutual", "data", "Which mutual",
        "The shared graph, directed, with edge ids on every edge. A and B are joined both ways "
        "by edges 1 and 6, and C and D by edges 5 and 7, all in the accent color; the other "
        "edges are one-way. The card returns the mutual edge id pairs (1, 6) and (5, 7).")
def which_mutual(s: Scene) -> None:
    left, right = compute(s, "g", "g.which_mutual()")
    _mutual_graph(s, left, mutual="accent", oneway="default", edge_ids=True)
    s.card_centered(right, [("[(1, 6), (5, 7)]", "accent")], title="edge id pairs")


@figure("count_mutual_node_pairs", "data", "Count mutual node pairs",
        "The shared graph, directed. A and B point at each other, and so do C and D; those "
        "reciprocal arcs are in the accent color and the one-way edges are faded. The card "
        "counts the two mutual pairs.")
def count_mutual_node_pairs(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_mutual_node_pairs()")
    _mutual_graph(s, left, mutual="accent", oneway="muted")
    s.card_centered(right, ["A ⇄ B", "C ⇄ D", ("count  2", "accent")])


@figure("count_asymmetric_node_pairs", "data", "Count asymmetric node pairs",
        "The shared graph, directed. The one-way edges A to C, B to C, and B to D are in the "
        "accent color; the reciprocal pairs between A and B and between C and D are faded. "
        "The card counts the three one-way pairs.")
def count_asymmetric_node_pairs(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_asymmetric_node_pairs()")
    _mutual_graph(s, left, mutual="muted", oneway="accent")
    s.card_centered(right, ["A → C", "B → C", "B → D", ("count  3", "accent")])


# -- attribute getters --------------------------------------------------------
#
# The returned dict is keyed by id, so the diamond keeps its id pills / edge-id
# boxes and the values appear in the card. Elements that lack the attribute are
# `muted` and missing from the result; ids that come back are accent.


@figure("get_node_attrs", "data", "Get node attributes",
        "The shared four-node graph with node ids in pills. Nodes A, B, and D have an attribute "
        "x and their ids 1, 2, 4 are highlighted; node C has no x and is faded. The card maps "
        "node id to value, in storage order, for the three nodes that have it: 1 is 3, 4 is 4, "
        "2 is 5.")
def get_node_attrs(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_node_attrs("x")')
    _diamond(s, left, styles={"C": "muted"})
    _ids(s, accent="ABD")
    s.card_centered(right, [("{1: 3, 4: 4, 2: 5}", "accent")], title="node id → x")


@figure("get_edge_attrs", "data", "Get edge attributes",
        "The shared four-node graph with edge ids on the edges. Edges 1, 3, and 5 have an "
        "attribute w and are in the accent color; edges 2 and 4 have no w and are faded. The "
        "card maps edge id to value for the three edges that have it: 1 is 2, 3 is 5, 5 is 1.")
def get_edge_attrs(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_edge_attrs("w")')
    _diamond(s, left, edge_styles={e: "muted" if e in (("A", "C"), ("B", "D")) else "accent"
                                   for e in D_EDGES})
    _edge_ids(s, accent={1, 3, 5})
    s.card_centered(right, [("{1: 2, 3: 5, 5: 1}", "accent")], title="edge id → w")


@figure("get_edge_attrs_by_id", "data", "Get edge attributes by id",
        "The shared four-node graph with edge ids on the edges; every edge has an attribute w. "
        "The queried edges 2 (A to C) and 5 (C to D) are outlined in the accent color with "
        "their ids highlighted. The card maps just those two ids to their values: 2 is 4 and "
        "5 is 1.")
def get_edge_attrs_by_id(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_edge_attrs_by_id("w", [2, 5])')
    _diamond(s, left, edge_styles={("A", "C"): "soft", ("C", "D"): "soft"})
    _edge_ids(s, accent={2, 5})
    s.card_centered(right, [("{2: 4, 5: 1}", "accent")], title="edge id → w")


# -- totals -------------------------------------------------------------------


@figure("count_nodes", "data", "Count nodes",
        "The shared four-node graph with every node, A to D, in the accent color and its id, "
        "1 to 4, in a pill. The card returns the node count, 4.")
def count_nodes(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_nodes()")
    _diamond(s, left, styles={k: "accent" for k in DIAMOND})
    _ids(s, accent="ABCD")
    s.card_centered(right, [("4", "accent")], title="nodes")


@figure("count_edges", "data", "Count edges",
        "The shared graph, directed, with edge ids on every edge; all six edges are in the "
        "accent color, including the reverse edge 6 from B back to A. In a directed graph A to "
        "B and B to A are separate edges, so the card returns 6.")
def count_edges(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_edges()")
    _diamond(s, left, directed=True, edge_styles={e: "accent" for e in D_EDGES},
             extra=[("B", "A", {"style": "accent", "bend": 14})])
    s.edges[0].bend = 14  # A→B bows away from its reverse
    _edge_ids(s, accent={1, 2, 3, 4, 5, 6})
    s.card_centered(right, [("6", "accent")], title="edges")


# --- summary ---------------------------------------------------------------------

# describe() prints nine "key : value" lines; the result panel is too narrow for the full
# keys at card size, so they're shown as a two-column table with short keys, in order.
DESCRIBE_ROWS = [
    ["nodes", "4"], ["edges", "5"], ["directed", "False"], ["density", "0.83"],
    ["avg degree (min, max)", "2.50 (2, 3)"], ["connected", "True"], ["components", "1"],
    ["tree", "False"], ["bipartite", "False"],
]


@figure("describe", "data", "Describe",
        "The shared four-node graph, two triangles sharing the edge B–C, beside the summary "
        "describe() prints for it: 4 nodes and 5 edges (both highlighted), undirected, "
        "density 0.83, average degree 2.50 ranging from 2 to 3, connected as a single "
        "component, and neither a tree nor bipartite.")
def describe(s: Scene) -> None:
    left, right = compute(s, "g", "g.describe()")
    _diamond(s, left)
    s.table_centered(right, DESCRIBE_ROWS, [152, 88], row_h=17, header=False,
                     highlight={(0, 1): "accent", (1, 1): "accent"})
