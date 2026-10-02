"""Selection: marking nodes and edges for later operations, and reading the marks back.

Template: `transform` for anything that changes the selection (select_*,
deselect_*, clear_selection, invert_*): the left panel is the current
selection, the right panel the selection after the call. `compute` for the
read-only queries (count_selected_*, get_selected_*_ids): the graph with its
selection, then a card with the returned value.

Encoding: selected nodes/edges are `accent`, unselected ones `default`. A node
the call names without selecting it (a seed, an anchor) is `soft`. Everything
is drawn on one shared 2x3 grid (`BASE`, `BASE_EDGES`, `UNIT`), the same base
used by the `_ws` working-selection figures, so the whole family reads as one.
Ids, creation order, and per-node values the call depends on are shown as
badges (nodes) or edge labels (edges) in both panels.
"""

from core import Box, Scene, compute, figure, place, transform

# Shared base layout for the selection families (also used by selection_actions.py).
BASE = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (0, 1), "E": (1, 1), "F": (2, 1)}
BASE_EDGES = [("A", "B"), ("B", "C"), ("A", "D"), ("B", "E"), ("C", "F"), ("D", "E"), ("E", "F")]
UNIT = 72

# edgecase assigns ids from 1 in creation order: nodes A..F -> 1..6, and the
# edges in BASE_EDGES order -> 1..7.
NODE_ID = {k: i for i, k in enumerate(BASE, 1)}
EDGE_ID = {e: i for i, e in enumerate(BASE_EDGES, 1)}


def _panel(s: Scene, box: Box, prefix: str, *, nodes=(), edges=(), soft=(),
           edge_labels: dict | None = None, labels=True, coords=None, edge_list=None) -> None:
    """Draw the base graph in `box` with the given nodes/edges selected (`accent`)."""
    coords = coords or BASE
    edge_list = edge_list or BASE_EDGES
    styles = {k: "accent" for k in nodes} | {k: "soft" for k in soft}
    items = []
    for e in edge_list:
        kw = {"style": "accent"} if e in edges else {}
        if edge_labels is not None:
            kw["weight"] = edge_labels[e]
        items.append((e[0], e[1], kw))
    s.graph(place(coords, box, unit=UNIT), items, styles=styles, labels=labels, prefix=prefix)


def _badges(s: Scene, prefix: str, values: dict, *, filled: bool = False) -> None:
    """Badges point away from the grid: top row north, bottom row south.

    A badge takes its node's selection color (accent if selected, ink if not), so
    an unselected 'before' panel never looks highlighted.
    """
    for k, v in values.items():
        color = "accent" if s.nodes[prefix + k].style == "accent" else "ink"
        s.badge(prefix + k, str(v), filled=filled, color=color,
                where="n" if BASE[k][1] == 0 else "s")


ID_LABELS = {k: str(v) for k, v in NODE_ID.items()}  # draw nodes by integer id


# --- selecting --------------------------------------------------------------

W_ATTR = {"A": 3, "B": 8, "C": 6, "D": 2, "E": 9, "F": 4}


@figure("select_nodes", "selection", "Select nodes by predicate",
        "Six nodes A to F, each with a badge giving its w attribute (3, 8, 6, 2, 9, 4). After "
        "selecting with the predicate w greater than 5, nodes B, C, and E, whose values are 8, "
        "6, and 9, are selected and drawn in the accent color.")
def select_nodes(s: Scene) -> None:
    left, right = transform(s, "g", 'g.select_nodes(lambda n: n.attributes["w"]>5)')
    _panel(s, left, "l.")
    _badges(s, "l.", W_ATTR)
    sel = [k for k, v in W_ATTR.items() if v > 5]
    _panel(s, right, "r.", nodes=sel)
    _badges(s, "r.", W_ATTR)


WEIGHT = dict(zip(BASE_EDGES, ["0.3", "0.8", "0.6", "0.2", "0.9", "0.4", "0.7"]))


@figure("select_edges", "selection", "Select edges by predicate",
        "A six-node grid whose seven edges are labeled with weights. After selecting with the "
        "predicate weight greater than 0.5, the four edges weighing 0.8, 0.6, 0.9, and 0.7 "
        "are selected and drawn in the accent color; the rest stay plain.")
def select_edges(s: Scene) -> None:
    left, right = transform(s, "g", "g.select_edges(lambda e: e.weight > 0.5)")
    _panel(s, left, "l.", edge_labels=WEIGHT)
    _panel(s, right, "r.", edges=[e for e, w in WEIGHT.items() if float(w) > 0.5],
           edge_labels=WEIGHT)


@figure("select_nodes_by_id", "selection", "Select nodes by id",
        "Six nodes labeled by their integer ids 1 to 6. After select_nodes_by_id with ids "
        "1, 3, and 5, those three nodes are selected and drawn in the accent color.")
def select_nodes_by_id(s: Scene) -> None:
    left, right = transform(s, "g", "g.select_nodes_by_id([1, 3, 5])")
    _panel(s, left, "l.", labels=ID_LABELS)
    _panel(s, right, "r.", nodes=["A", "C", "E"], labels=ID_LABELS)


@figure("select_edges_by_id", "selection", "Select edges by id",
        "A six-node grid whose seven edges are labeled with their integer ids 1 to 7. After "
        "select_edges_by_id with ids 2 and 5, the edges B–C and C–F are selected and drawn "
        "in the accent color.")
def select_edges_by_id(s: Scene) -> None:
    left, right = transform(s, "g", "g.select_edges_by_id([2, 5])")
    _panel(s, left, "l.", edge_labels=EDGE_ID)
    _panel(s, right, "r.", edges=[e for e, i in EDGE_ID.items() if i in (2, 5)],
           edge_labels=EDGE_ID)


@figure("select_edges_by_node", "selection", "Select edges by node",
        "A six-node grid with no selection. After select_edges_by_node on B, the three edges "
        "touching B (A–B, B–C, and B–E) are selected in the accent color; B itself is outlined "
        "as the named node but is not selected.")
def select_edges_by_node(s: Scene) -> None:
    left, right = transform(s, "g", 'g.select_edges_by_node(["B"])')
    _panel(s, left, "l.")
    _panel(s, right, "r.", soft=["B"], edges=[e for e in BASE_EDGES if "B" in e])


DEGREE = {"A": 2, "B": 3, "C": 2, "D": 2, "E": 3, "F": 2}


@figure("select_nodes_by_degree", "selection", "Select nodes by degree",
        "Six nodes, each with a badge giving its degree: B and E have degree 3, the others 2. "
        "After select_nodes_by_degree with min_degree 3, only B and E are selected and drawn "
        "in the accent color.")
def select_nodes_by_degree(s: Scene) -> None:
    left, right = transform(s, "g", "g.select_nodes_by_degree(min_degree=3)")
    _panel(s, left, "l.")
    _badges(s, "l.", DEGREE)
    _panel(s, right, "r.", nodes=["B", "E"])
    _badges(s, "r.", DEGREE)


# The base grid plus one diagonal B–D, so PageRank has a single clear winner
# (B 0.24) instead of the B/E tie the plain grid gives (which would echo the
# degree figure).
CENT_EDGES = BASE_EDGES + [("B", "D")]
PAGERANK = {"A": "0.13", "B": "0.24", "C": "0.13", "D": "0.18", "E": "0.18", "F": "0.13"}


@figure("select_nodes_by_centrality", "selection", "Select nodes by centrality",
        "The six-node grid with an extra diagonal edge B–D, each node badged with its PageRank "
        "score. B scores highest at 0.24, ahead of D and E at 0.18, so selecting the top one "
        "node by PageRank selects B, drawn in the accent color.")
def select_nodes_by_centrality(s: Scene) -> None:
    left, right = transform(s, "g", 'g.select_nodes_by_centrality("pagerank", k=1)')
    _panel(s, left, "l.", edge_list=CENT_EDGES)
    _badges(s, "l.", PAGERANK)
    _panel(s, right, "r.", nodes=["B"], edge_list=CENT_EDGES)
    _badges(s, "r.", PAGERANK)


@figure("select_nodes_in_neighborhood", "selection", "Select nodes in neighborhood",
        "A six-node grid with no selection. After select_nodes_in_neighborhood from seed A with "
        "k=2, badges give each node's hop distance from A, and every node within two hops is "
        "selected in the accent color: A itself, B and D at one hop, and C and E at two. F, "
        "three hops away, is not selected.")
def select_nodes_in_neighborhood(s: Scene) -> None:
    # edgecase's implementation adds the seed itself to the selection (verified),
    # so A is drawn selected.
    left, right = transform(s, "g", 'g.select_nodes_in_neighborhood("A", k=2)')
    _panel(s, left, "l.")
    _panel(s, right, "r.", nodes=list("ABCDE"))
    _badges(s, "r.", {"A": 0, "B": 1, "D": 1, "C": 2, "E": 2, "F": 3})


@figure("select_last_nodes_created", "selection", "Select last nodes created",
        "Six nodes A to F with filled badges 1 to 6 giving the order in which they were created. "
        "select_last_nodes_created(2) selects the two most recent, E and F, drawn in the "
        "accent color.")
def select_last_nodes_created(s: Scene) -> None:
    left, right = transform(s, "g", "g.select_last_nodes_created(2)")
    order = {k: i for i, k in enumerate(BASE, 1)}
    _panel(s, left, "l.")
    _badges(s, "l.", order, filled=True)
    _panel(s, right, "r.", nodes=["E", "F"])
    _badges(s, "r.", order, filled=True)


@figure("select_last_edges_created", "selection", "Select last edges created",
        "A six-node grid whose seven edges are labeled 1 to 7 in the order they were created. "
        "select_last_edges_created(2) selects the two most recent, D–E and E–F, drawn in the "
        "accent color.")
def select_last_edges_created(s: Scene) -> None:
    left, right = transform(s, "g", "g.select_last_edges_created(2)")
    _panel(s, left, "l.", edge_labels=EDGE_ID)
    _panel(s, right, "r.", edges=[("D", "E"), ("E", "F")], edge_labels=EDGE_ID)


# --- changing an existing selection -----------------------------------------


@figure("deselect_nodes", "selection", "Deselect nodes",
        "Before: nodes A, B, C, and E are selected in the accent color. After deselect_nodes "
        "on A and C, those two return to plain and only B and E remain selected.")
def deselect_nodes(s: Scene) -> None:
    left, right = transform(s, "g", 'g.deselect_nodes(["A", "C"])')
    _panel(s, left, "l.", nodes=list("ABCE"))
    _panel(s, right, "r.", nodes=["B", "E"])


@figure("deselect_edges", "selection", "Deselect edges",
        "A grid whose edges are labeled with their ids 1 to 7. Before: edges 1, 2, and 4 "
        "(A–B, B–C, B–E) are selected in the accent color. After deselect_edges with id 2, "
        "edge B–C returns to plain and edges 1 and 4 stay selected.")
def deselect_edges(s: Scene) -> None:
    left, right = transform(s, "g", "g.deselect_edges([2])")
    _panel(s, left, "l.", edges=[("A", "B"), ("B", "C"), ("B", "E")], edge_labels=EDGE_ID)
    _panel(s, right, "r.", edges=[("A", "B"), ("B", "E")], edge_labels=EDGE_ID)


@figure("clear_selection", "selection", "Clear selection",
        "Before: nodes B and E and the edge between them are selected in the accent color. "
        "After clear_selection, nothing is selected: every node and edge is plain.")
def clear_selection(s: Scene) -> None:
    left, right = transform(s, "g", "g.clear_selection()")
    _panel(s, left, "l.", nodes=["B", "E"], edges=[("B", "E")])
    _panel(s, right, "r.")


@figure("invert_node_selection", "selection", "Invert node selection",
        "Before: nodes B and E are selected in the accent color. After invert_node_selection "
        "the selection flips: A, C, D, and F are selected and B and E are plain.")
def invert_node_selection(s: Scene) -> None:
    left, right = transform(s, "g", "g.invert_node_selection()")
    _panel(s, left, "l.", nodes=["B", "E"])
    _panel(s, right, "r.", nodes=list("ACDF"))


INV_EDGES = [("A", "B"), ("B", "E"), ("E", "F")]


@figure("invert_edge_selection", "selection", "Invert edge selection",
        "Before: edges A–B, B–E, and E–F are selected in the accent color. After "
        "invert_edge_selection the selection flips: the other four edges are selected and "
        "those three are plain.")
def invert_edge_selection(s: Scene) -> None:
    left, right = transform(s, "g", "g.invert_edge_selection()")
    _panel(s, left, "l.", edges=INV_EDGES)
    _panel(s, right, "r.", edges=[e for e in BASE_EDGES if e not in INV_EDGES])


# --- querying the selection -------------------------------------------------


@figure("count_selected_nodes", "selection", "Count selected nodes",
        "A six-node grid with nodes B, D, and F selected in the accent color. "
        "count_selected_nodes() returns 3.")
def count_selected_nodes(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_selected_nodes()")
    _panel(s, left, "l.", nodes=["B", "D", "F"])
    s.card_centered(right, [("3", "accent")], title="selected nodes")


@figure("count_selected_edges", "selection", "Count selected edges",
        "A six-node grid with edges A–B, B–E, and E–F selected in the accent color. "
        "count_selected_edges() returns 3.")
def count_selected_edges(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_selected_edges()")
    _panel(s, left, "l.", edges=INV_EDGES)
    s.card_centered(right, [("3", "accent")], title="selected edges")


@figure("get_selected_node_ids", "selection", "Get selected node ids",
        "Six nodes labeled by their integer ids 1 to 6, with nodes 2, 4, and 6 selected in the "
        "accent color. get_selected_node_ids() returns the list of those ids, 6, 4, 2, in no "
        "guaranteed order.")
def get_selected_node_ids(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_selected_node_ids()")
    _panel(s, left, "l.", nodes=["B", "D", "F"], labels=ID_LABELS)
    s.card_centered(right, [("[6, 4, 2]", "accent")], title="node ids")


@figure("get_selected_edge_ids", "selection", "Get selected edge ids",
        "A grid whose edges are labeled with their ids 1 to 7, with edges 1, 4, and 7 "
        "(A–B, B–E, E–F) selected in the accent color. get_selected_edge_ids() returns the "
        "list of those ids, 7, 4, 1, in no guaranteed order.")
def get_selected_edge_ids(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_selected_edge_ids()")
    _panel(s, left, "l.", edges=INV_EDGES, edge_labels=EDGE_ID)
    s.card_centered(right, [("[7, 4, 1]", "accent")], title="edge ids")
