"""Selection actions: methods that act on the working selection (`*_ws`).

Template: `transform` (`compute` for the DataFrame exports). Left: the graph
with the current selection in `accent`. Right: what the action does to the
selection, and only to it: deleted elements become `removed` (dashed), new
edges appear in `accent`, and attribute values (badges on nodes, labels on
edges) change only where the selection is. The DataFrame exports keep the
same left panel and return a small table of just the selected rows.

Every figure shares one 2 x 3 base layout (`BASE`, `BASE_EDGES`, `UNIT`), also
used by selection.py, so selecting and acting on a selection read as one
family. The reverse-direction methods use the same layout with directed edges.
"""

from core import Scene, compute, figure, place, transform

BASE = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (0, 1), "E": (1, 1), "F": (2, 1)}
BASE_EDGES = [("A", "B"), ("B", "C"), ("A", "D"), ("B", "E"), ("C", "F"), ("D", "E"), ("E", "F")]
UNIT = 72

# Attribute values used by the setter/mutator figures (checked with edgecase).
NODE_X = {"A": 3, "B": 5, "C": 2, "D": 4, "E": 6, "F": 1}
EDGE_W = dict(zip(BASE_EDGES, [3, 1, 2, 4, 6, 2, 3]))

# The two shared selections: three nodes, and three edges.
SEL_NODES = ("B", "D", "F")
SEL_EDGES = {("B", "C"), ("B", "E"), ("D", "E")}


def _touches(e: tuple[str, str], nodes) -> bool:
    return e[0] in nodes or e[1] in nodes


def _node_values(s: Scene, prefix: str, values: dict[str, int], changed) -> None:
    """Value badges above the top row and below the bottom row; selection in accent."""
    for k, v in values.items():
        s.badge(prefix + k, str(v), where="n" if BASE[k][1] == 0 else "s",
                color="accent" if k in changed else "ink")


def _weighted(values: dict[tuple[str, str], int], sel) -> list:
    return [(a, b, {"weight": values[(a, b)], "style": "accent" if (a, b) in sel else "default"})
            for a, b in BASE_EDGES]


# --- structural actions ------------------------------------------------------


@figure("delete_nodes_ws", "selection", "Delete selected nodes",
        "Left: a six-node graph with nodes B and E selected in the accent color. Right: B "
        "and E are gone along with all five of their incident edges, shown as dashed "
        "outlines; only A, C, D, F and the edges A–D and C–F remain.")
def delete_nodes_ws(s: Scene) -> None:
    left, right = transform(s, "g", "g.delete_nodes_ws()")
    sel = ("B", "E")
    s.graph(place(BASE, left, unit=UNIT), BASE_EDGES,
            styles={k: "accent" for k in sel}, prefix="l.")
    s.graph(place(BASE, right, unit=UNIT),
            [(a, b, {"style": "removed"}) if _touches((a, b), sel) else (a, b) for a, b in BASE_EDGES],
            styles={k: "removed" for k in sel}, prefix="r.")


@figure("delete_edges_ws", "selection", "Delete selected edges",
        "Left: a six-node graph with edges B–C and D–E selected in the accent color. Right: "
        "those two edges are gone, shown dashed, while all six nodes and the other five "
        "edges stay in place.")
def delete_edges_ws(s: Scene) -> None:
    left, right = transform(s, "g", "g.delete_edges_ws()")
    sel = {("B", "C"), ("D", "E")}
    s.graph(place(BASE, left, unit=UNIT),
            [(a, b, {"style": "accent" if (a, b) in sel else "default"}) for a, b in BASE_EDGES],
            prefix="l.")
    s.graph(place(BASE, right, unit=UNIT),
            [(a, b, {"style": "removed" if (a, b) in sel else "default"}) for a, b in BASE_EDGES],
            prefix="r.")


@figure("fully_connect_nodes_ws", "selection", "Fully connect selected nodes",
        "Left: nodes A, B, D, and E are selected in the accent color; they already form a "
        "square of four edges. Right: the two missing pairs get new accent edges, the "
        "diagonals A–E and B–D, so the four selected nodes form a clique; C and F are untouched.")
def fully_connect_nodes_ws(s: Scene) -> None:
    left, right = transform(s, "g", "g.fully_connect_nodes_ws()")
    sel = {k: "accent" for k in "ABDE"}
    s.graph(place(BASE, left, unit=UNIT), BASE_EDGES, styles=sel, prefix="l.")
    new = [("A", "E", {"style": "accent"}), ("B", "D", {"style": "accent"})]
    s.graph(place(BASE, right, unit=UNIT), BASE_EDGES + new, styles=sel, prefix="r.")


@figure("add_reverse_edges_ws", "selection", "Add reverse edges for the selection",
        "Left: a directed six-node graph with edges B→C and D→E selected in the accent "
        "color. Right: each selected edge gains a reverse partner, C→B and E→D, drawn in "
        "the accent color; the original edges stay, so those pairs now point both ways.")
def add_reverse_edges_ws(s: Scene) -> None:
    left, right = transform(s, "g", "g.add_reverse_edges_ws()")
    sel = {("B", "C"), ("D", "E")}
    s.graph(place(BASE, left, unit=UNIT),
            [(a, b, {"style": "accent" if (a, b) in sel else "default"}) for a, b in BASE_EDGES],
            directed=True, prefix="l.")
    # Reciprocal pairs bow apart (same bend, opposite directions -> opposite sides).
    edges = [(a, b, {"bend": 7}) if (a, b) in sel else (a, b) for a, b in BASE_EDGES]
    edges += [(b, a, {"bend": 7, "style": "accent"}) for a, b in sorted(sel)]
    s.graph(place(BASE, right, unit=UNIT), edges, directed=True, prefix="r.")


@figure("rev_edge_dir_ws", "selection", "Reverse selected edges",
        "Left: a directed six-node graph with edges B→C and D→E selected in the accent "
        "color. Right: the same two edges now point the other way, C→B and E→D, still in "
        "the accent color; every other edge keeps its direction.")
def rev_edge_dir_ws(s: Scene) -> None:
    left, right = transform(s, "g", "g.rev_edge_dir_ws()")
    sel = {("B", "C"), ("D", "E")}
    s.graph(place(BASE, left, unit=UNIT),
            [(a, b, {"style": "accent" if (a, b) in sel else "default"}) for a, b in BASE_EDGES],
            directed=True, prefix="l.")
    s.graph(place(BASE, right, unit=UNIT),
            [(b, a, {"style": "accent"}) if (a, b) in sel else (a, b) for a, b in BASE_EDGES],
            directed=True, prefix="r.")


@figure("transform_to_subgraph_ws", "selection", "Subgraph from the selection",
        "Left: nodes A, B, C, and E are selected in the accent color. Right: the new graph "
        "keeps just those four nodes and the three edges among them, A–B, B–C, and B–E; "
        "nodes D and F and every edge touching them are dropped, shown dashed.")
def transform_to_subgraph_ws(s: Scene) -> None:
    left, right = transform(s, "g", "g.transform_to_subgraph_ws()")
    sel = set("ABCE")
    s.graph(place(BASE, left, unit=UNIT), BASE_EDGES, styles={k: "accent" for k in sel}, prefix="l.")
    s.graph(place(BASE, right, unit=UNIT),
            [(a, b) if {a, b} <= sel else (a, b, {"style": "removed"}) for a, b in BASE_EDGES],
            styles={k: "accent" if k in sel else "removed" for k in BASE}, prefix="r.")


# --- attribute setters & mutators ---------------------------------------------


@figure("set_node_attrs_ws", "selection", "Set attributes on selected nodes",
        "Every node carries a value x in a badge (A 3, B 5, C 2, D 4, E 6, F 1). Nodes B, "
        "D, and F are selected in the accent color; after the call all three read 7, "
        "whatever they held before, while the other nodes keep their values.")
def set_node_attrs_ws(s: Scene) -> None:
    left, right = transform(s, "g", "g.set_node_attrs_ws(x=7)")
    styles = {k: "accent" for k in SEL_NODES}
    s.graph(place(BASE, left, unit=UNIT), BASE_EDGES, styles=styles, prefix="l.")
    s.graph(place(BASE, right, unit=UNIT), BASE_EDGES, styles=styles, prefix="r.")
    _node_values(s, "l.", NODE_X, SEL_NODES)
    _node_values(s, "r.", {k: 7 if k in SEL_NODES else v for k, v in NODE_X.items()}, SEL_NODES)


@figure("mutate_node_attrs_ws", "selection", "Mutate an attribute on selected nodes",
        "Every node carries a value x in a badge. Nodes B, D, and F are selected in the "
        "accent color with values 5, 4, and 1; applying the function doubles each to 10, 8, "
        "and 2, while the unselected nodes keep 3, 2, and 6.")
def mutate_node_attrs_ws(s: Scene) -> None:
    left, right = transform(s, "g", 'g.mutate_node_attrs_ws("x", lambda x: 2*x)')
    styles = {k: "accent" for k in SEL_NODES}
    s.graph(place(BASE, left, unit=UNIT), BASE_EDGES, styles=styles, prefix="l.")
    s.graph(place(BASE, right, unit=UNIT), BASE_EDGES, styles=styles, prefix="r.")
    _node_values(s, "l.", NODE_X, SEL_NODES)
    _node_values(s, "r.", {k: 2 * v if k in SEL_NODES else v for k, v in NODE_X.items()}, SEL_NODES)


@figure("set_edge_attrs_ws", "selection", "Set attributes on selected edges",
        "Every edge carries a value w. Edges B–C, B–E, and D–E are selected in the accent "
        "color with values 1, 4, and 2; after the call all three read 5, while the other "
        "four edges keep their values.")
def set_edge_attrs_ws(s: Scene) -> None:
    left, right = transform(s, "g", "g.set_edge_attrs_ws(w=5)")
    s.graph(place(BASE, left, unit=UNIT), _weighted(EDGE_W, SEL_EDGES), prefix="l.")
    after = {e: 5 if e in SEL_EDGES else v for e, v in EDGE_W.items()}
    s.graph(place(BASE, right, unit=UNIT), _weighted(after, SEL_EDGES), prefix="r.")


@figure("mutate_edge_attrs_ws", "selection", "Mutate an attribute on selected edges",
        "Every edge carries a value w. Edges B–C, B–E, and D–E are selected in the accent "
        "color with values 1, 4, and 2; applying the function doubles each to 2, 8, and 4, "
        "while the other four edges keep their values.")
def mutate_edge_attrs_ws(s: Scene) -> None:
    left, right = transform(s, "g", 'g.mutate_edge_attrs_ws("w", lambda w: 2*w)')
    s.graph(place(BASE, left, unit=UNIT), _weighted(EDGE_W, SEL_EDGES), prefix="l.")
    after = {e: 2 * v if e in SEL_EDGES else v for e, v in EDGE_W.items()}
    s.graph(place(BASE, right, unit=UNIT), _weighted(after, SEL_EDGES), prefix="r.")


# --- DataFrame exports ---------------------------------------------------------


@figure("get_node_df_ws", "selection", "Selected nodes as a DataFrame",
        "A six-node graph with nodes B, D, and F selected in the accent color. The returned "
        "table has one row per selected node only: ids 2, 4, 6 with labels B, D, F and "
        "degrees 3, 2, 2.")
def get_node_df_ws(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_node_df_ws()")
    s.graph(place(BASE, left, unit=UNIT), BASE_EDGES,
            styles={k: "accent" for k in SEL_NODES})
    rows = [["id", "label", "degree"], ["2", "B", "3"], ["4", "D", "2"], ["6", "F", "2"]]
    s.table_centered(right, rows, [44, 52, 60])


@figure("get_edge_df_ws", "selection", "Selected edges as a DataFrame",
        "A six-node graph whose badges give the node ids 1 to 6, with edges B–C and D–E "
        "selected in the accent color. The returned table has one row per selected edge "
        "only: edge 2 from node 2 to 3, and edge 6 from node 4 to 5.")
def get_edge_df_ws(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_edge_df_ws()")
    sel = {("B", "C"), ("D", "E")}
    s.graph(place(BASE, left, unit=UNIT),
            [(a, b, {"style": "accent" if (a, b) in sel else "default"}) for a, b in BASE_EDGES])
    # node ids, so the source/target columns can be read off the picture
    for i, k in enumerate(BASE, 1):
        s.badge(k, str(i), where="n" if BASE[k][1] == 0 else "s", color="ink")
    rows = [["id", "source", "target"], ["2", "2", "3"], ["6", "4", "5"]]
    s.table_centered(right, rows, [44, 60, 60])
