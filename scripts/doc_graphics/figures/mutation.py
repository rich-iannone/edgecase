"""Mutations: the graph before → after one change.

Template: `transform`. The left panel is the graph before the call, the right
panel after. The element being added is `accent` on the right; an element
being removed is `accent` on the left and `removed` (dashed) on the right, so
the reader sees *what* goes and *where it was*. Both panels share one layout
so nothing moves except the thing that changed.

Two base layouts on one grid (columns 1.2 apart, rows 1 apart) keep siblings
comparable:

- `SQUARE` (A–D, a 4-cycle) for node/edge add/delete (single, bulk, and by
  id; the by-id figures add node-id pills at the outer corners) and the
  clean-up trio (`simplify`, `remove_selfloops`, `remove_duplicate_edges`),
  which all start from the same multigraph: a self-loop on B and a doubled
  C–D edge.
- `H` (A–F, two rows joined by the B–E rung) for contractions and the
  attribute mutations. In contractions the surviving node `u` is `soft` in
  both panels; the absorbed node and its edges are `accent` → `removed`; the
  rerouted edges (and any self-loop the merge creates) are `accent` on the
  right.

Attribute mutations keep the structure fixed and change node appearance:
the left panel shows the source values as ink badges, the right panel the
written result (colors / sizes) with the badges in the result colors. The
plain setters (`set_node_attrs`, `set_edge_attrs`) mirror the selection
family's `*_ws` setters: the elements named in the call are `soft` in both
panels and only their values (badges / edge labels) change, turning accent.
`relabel_nodes` is the label version: renamed nodes `soft` → `accent` with
their new labels, ink id pills on both sides to show that ids survive.
"""

from core import SIZE_NOTE, Box, Scene, figure, place, transform

UNIT = 80  # one density for the whole family
BEND = 15  # parallel pairs bow symmetrically by this much

SQUARE = {"A": (0, 0), "B": (1.2, 0), "C": (0, 1), "D": (1.2, 1)}
SQUARE_EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]

H = {"A": (0, 0), "B": (1.2, 0), "C": (2.4, 0), "D": (0, 1), "E": (1.2, 1), "F": (2.4, 1)}
H_EDGES = [("A", "B"), ("B", "C"), ("B", "E"), ("D", "E"), ("E", "F")]


def _place(coords: dict, box: Box) -> dict:
    """Place on the family grid, nudged down a little to leave room for self-loops."""
    return place(coords, Box(box.x0, box.y0 + 8, box.x1, box.y1), unit=UNIT)


def _styled(edges, styles: dict) -> list:
    """Attach per-edge styles keyed by frozenset({a, b}) (or (a, a) for loops)."""
    out = []
    for e in edges:
        a, b = e[0], e[1]
        kw = dict(e[2]) if len(e) > 2 else {}
        st = styles.get(frozenset((a, b)))
        if st:
            kw["style"] = st
        out.append((a, b, kw))
    return out


# --- nodes and edges --------------------------------------------------------


@figure("add_node", "mutation", "Add node",
        "Before: a four-node cycle A, B, D, C. After: a fifth node E appears in the accent "
        "color beside it, with no edges; nothing else changes.")
def add_node(s: Scene) -> None:
    left, right = transform(s, "g", 'g.add_node(label="E")')
    coords = {**SQUARE, "E": (2.4, 0.5)}
    before = _place(coords, left)  # place with E so A–D line up across panels
    del before["E"]
    s.graph(before, SQUARE_EDGES, prefix="l.")
    s.graph(_place(coords, right), SQUARE_EDGES, styles={"E": "accent"}, prefix="r.")


@figure("delete_node", "mutation", "Delete node",
        "Before: in a four-node cycle, node D is highlighted in the accent color along with "
        "its two incident edges. After: D and both of its edges are gone, shown as dashed "
        "outlines, leaving the path C, A, B.")
def delete_node(s: Scene) -> None:
    left, right = transform(s, "g", 'g.delete_node("D")')
    for box, p, st in ((left, "l.", "accent"), (right, "r.", "removed")):
        s.graph(_place(SQUARE, box),
                [(a, b, {"style": st}) if "D" in (a, b) else (a, b) for a, b in SQUARE_EDGES],
                styles={"D": st}, prefix=p)


@figure("add_edge", "mutation", "Add edge",
        "Before: four nodes joined by three edges, C to A to B to D. After: a new edge between "
        "the existing nodes C and D, in the accent color, closes the cycle; no nodes are "
        "added.")
def add_edge(s: Scene) -> None:
    left, right = transform(s, "g", 'g.add_edge("C", "D")')
    s.graph(_place(SQUARE, left), SQUARE_EDGES[:3], prefix="l.")
    s.graph(_place(SQUARE, right), SQUARE_EDGES[:3] + [("C", "D", {"style": "accent"})],
            prefix="r.")


@figure("delete_edge", "mutation", "Delete edge",
        "Before: a four-node cycle with the edge between A and C highlighted in the accent "
        "color. After: that edge is gone, shown dashed, while all four nodes and the other "
        "three edges stay in place.")
def delete_edge(s: Scene) -> None:
    left, right = transform(s, "g", 'g.delete_edge("A", "C")')
    for box, p, st in ((left, "l.", "accent"), (right, "r.", "removed")):
        s.graph(_place(SQUARE, box), _styled(SQUARE_EDGES, {frozenset("AC"): st}), prefix=p)


@figure("delete_edge_by_id", "mutation", "Delete edge by ID",
        "Before: a four-node cycle whose edges are labeled with their edge IDs 1 to 5; A and B "
        "are joined by two parallel edges, 1 and 5, and edge 5 is highlighted in the accent "
        "color. After: only edge 5 is gone, shown dashed, while its parallel twin, edge 1, "
        "remains.")
def delete_edge_by_id(s: Scene) -> None:
    left, right = transform(s, "g", "g.delete_edge_by_id(5)")
    for box, p, st in ((left, "l.", "accent"), (right, "r.", "removed")):
        edges = [("A", "B", {"weight": "1", "bend": BEND}), ("A", "C", {"weight": "2"}),
                 ("B", "D", {"weight": "3"}), ("C", "D", {"weight": "4"}),
                 ("A", "B", {"weight": "5", "bend": -BEND, "style": st})]
        s.graph(_place(SQUARE, box), edges, prefix=p)


# --- clean-up: self-loops and parallel edges ---------------------------------
#
# One multigraph for all three: SQUARE plus a self-loop on B and a second C–D
# edge. Each method removes its target (accent → removed) and leaves the
# other one untouched, so the three figures can be compared directly.


def _cleanup(s: Scene, call: str, *, loop: bool, dup: bool) -> None:
    left, right = transform(s, "g", call)
    for box, p, st in ((left, "l.", "accent"), (right, "r.", "removed")):
        edges = [e for e in SQUARE_EDGES if e != ("C", "D")] + [
            ("C", "D", {"bend": -BEND}),  # the first C–D edge (lowest id) is the survivor
            ("C", "D", {"bend": BEND, **({"style": st} if dup else {})}),
            ("B", "B", {"style": st} if loop else {}),
        ]
        s.graph(_place(SQUARE, box), edges, prefix=p)


@figure("simplify", "mutation", "Simplify",
        "Before: a four-node cycle with a self-loop on B and a second, parallel edge between "
        "C and D, both in the accent color. After simplify(), the self-loop and the duplicate "
        "edge are gone, shown dashed, leaving a simple four-node cycle.")
def simplify(s: Scene) -> None:
    _cleanup(s, "g.simplify()", loop=True, dup=True)


@figure("remove_selfloops", "mutation", "Remove self-loops",
        "Before: a four-node cycle with a self-loop on B, in the accent color, and a pair of "
        "parallel edges between C and D. After: the self-loop is gone, shown dashed, while "
        "both parallel C to D edges remain.")
def remove_selfloops(s: Scene) -> None:
    _cleanup(s, "g.remove_selfloops()", loop=True, dup=False)


@figure("remove_duplicate_edges", "mutation", "Remove duplicate edges",
        "Before: a four-node cycle with two parallel edges between C and D, the second in the "
        "accent color, and a self-loop on B. After: only one C to D edge remains and the "
        "duplicate is shown dashed, while the self-loop is left in place.")
def remove_duplicate_edges(s: Scene) -> None:
    _cleanup(s, "g.remove_duplicate_edges()", loop=False, dup=True)


# --- contractions -------------------------------------------------------------


def _contract(s: Scene, call: str, u: str, gone: str, new_edges: list, *,
              hull: bool = False) -> None:
    """`gone` nodes merge into `u`; their incident edges reroute as `new_edges`."""
    left, right = transform(s, "g", call)
    touched = {frozenset((a, b)): True for a, b in H_EDGES if a in gone or b in gone}
    for box, p, st in ((left, "l.", "accent"), (right, "r.", "removed")):
        edges = _styled(H_EDGES, {k: st for k in touched})
        if p == "r.":
            edges += [(a, b, {"style": "accent"}) for a, b in new_edges]
        s.graph(_place(H, box), edges, styles={u: "soft", **{k: st for k in gone}}, prefix=p)
        if hull and p == "l.":
            s.hull([p + u] + [p + k for k in gone])


@figure("contract_nodes", "mutation", "Contract nodes",
        "Before: a six-node graph in two rows; node A is outlined as the survivor and node D, "
        "which is not adjacent to A, is in the accent color with its edge to E. After: D is "
        "gone, shown dashed, and its edge has been rerouted to A, so a new accent edge joins "
        "A and E.")
def contract_nodes(s: Scene) -> None:
    _contract(s, 'g.contract_nodes("A", "D")', "A", "D", [("A", "E")])


@figure("contracted_edge", "mutation", "Contract an edge",
        "Before: a six-node graph in two rows; B is outlined as the survivor and the edge B to E "
        "is contracted, so E and its three edges are in the accent color. After: E is gone, "
        "shown dashed; its edges to D and F now attach to B, and the contracted edge itself "
        "has become a self-loop on B, all in the accent color.")
def contracted_edge(s: Scene) -> None:
    _contract(s, 'g.contracted_edge("B", "E")', "B", "E",
              [("B", "D"), ("B", "F"), ("B", "B")])


@figure("contract_nodes_multi", "mutation", "Contract several nodes",
        "Before: a six-node graph in two rows with B, E, and F grouped in a tinted region; B, "
        "first in the list, is outlined as the survivor and E and F are in the accent color. "
        "After: E and F are gone, shown dashed. Their outside edge from D now attaches to B, "
        "and the edges inside the group collapse into one self-loop on B.")
def contract_nodes_multi(s: Scene) -> None:
    _contract(s, 'g.contract_nodes_multi(["B", "E", "F"])', "B", "EF",
              [("B", "D"), ("B", "B")], hull=True)


# --- attribute mutations --------------------------------------------------------

COMMUNITY = {"A": 0, "B": 1, "C": 0, "D": 2, "E": 1, "F": 2}
GROUP = {0: "accent", 1: "g2", 2: "g3"}  # palette colors in order of first appearance


@figure("colorize_node_attrs", "mutation", "Colorize node attributes",
        "Before: a six-node graph whose nodes carry a community attribute, shown as badges "
        "with the values 0, 1, and 2. After: each node has a color written from its value, "
        "so the two nodes with 0 are in the accent color, the two with 1 in teal, and the two "
        "with 2 in violet.")
def colorize_node_attrs(s: Scene) -> None:
    left, right = transform(s, "g", 'g.colorize_node_attrs("community")')
    s.graph(_place(H, left), H_EDGES, prefix="l.")
    s.graph(_place(H, right), H_EDGES, prefix="r.",
            styles={k: GROUP[v] for k, v in COMMUNITY.items()})
    for k, v in COMMUNITY.items():
        s.badge("l." + k, str(v), color="ink")
        s.badge("r." + k, str(v), color=GROUP[v])


SCORE = {"A": 2, "B": 10, "C": 6, "D": 4, "E": 8, "F": 0}
SIZE = {"A": 0.6, "B": 1.0, "C": 0.8, "D": 0.7, "E": 0.9, "F": 0.5}  # edgecase output


@figure("rescale_node_attrs", "mutation", "Rescale node attributes",
        "Before: a six-node graph whose nodes carry raw scores from 0 to 10, shown as badges, "
        "all drawn at one size. After: the scores are rescaled linearly into the default "
        "range 0.5 to 1.0 and written as each node's size, so F becomes 0.5 and B 1.0, with "
        "every node drawn in proportion.")
def rescale_node_attrs(s: Scene) -> None:
    left, right = transform(s, "g", 'g.rescale_node_attrs("score")')
    s.graph(_place(H, left), H_EDGES, prefix="l.")
    s.graph(_place(H, right), H_EDGES, prefix="r.")
    for k, v in SIZE.items():
        s.nodes["r." + k].r = 20 * v  # radius ∝ the written size (10 to 20 px)
    for k in H:
        s.badge("l." + k, str(SCORE[k]), color="ink")
        s.badge("r." + k, f"{SIZE[k]:.1f}")


# --- bulk and by-id adders -------------------------------------------------------
#
# Same SQUARE as add_node/add_edge. The by-id variants tag every node with its
# integer id (A=1 … D=4, insertion order) in a pill at the outer corner; the
# ids passed to the call are accent pills in both panels.

ID_AT = {"A": ("1", "nw"), "B": ("2", "ne"), "C": ("3", "sw"), "D": ("4", "se")}


def _node_ids(s: Scene, prefix: str, accent: str) -> None:
    for k, (i, where) in ID_AT.items():
        s.badge(prefix + k, i, where=where, color="accent" if k in accent else "ink")


@figure("add_nodes", "mutation", "Add several nodes",
        "Before: a four-node cycle A, B, D, C. After: two new nodes, E and F, appear in the "
        "accent color beside it in one call, with no edges; nothing else changes.")
def add_nodes(s: Scene) -> None:
    left, right = transform(s, "g", 'g.add_nodes(["E", "F"])')
    coords = {**SQUARE, "E": (2.4, 0), "F": (2.4, 1)}
    before = _place(coords, left)  # place with E, F so A–D line up across panels
    del before["E"], before["F"]
    s.graph(before, SQUARE_EDGES, prefix="l.")
    s.graph(_place(coords, right), SQUARE_EDGES, styles={"E": "accent", "F": "accent"},
            prefix="r.")


@figure("add_edges", "mutation", "Add several edges",
        "Before: four nodes with two edges, B to A to C. After: two new edges, B to D and C to "
        "D, appear in the accent color in one call and close the cycle; no nodes are added.")
def add_edges(s: Scene) -> None:
    left, right = transform(s, "g", 'g.add_edges([("B", "D"), ("C", "D")])')
    s.graph(_place(SQUARE, left), SQUARE_EDGES[:2], prefix="l.")
    s.graph(_place(SQUARE, right),
            SQUARE_EDGES[:2] + [(a, b, {"style": "accent"}) for a, b in SQUARE_EDGES[2:]],
            prefix="r.")


@figure("add_edge_by_id", "mutation", "Add edge by ID",
        "Before: four nodes tagged with their integer ids, A 1, B 2, C 3, D 4, joined by three "
        "edges; the ids 3 and 4 are highlighted. After: a new accent edge joins node 3 (C) and "
        "node 4 (D), closing the cycle; no nodes are added.")
def add_edge_by_id(s: Scene) -> None:
    left, right = transform(s, "g", "g.add_edge_by_id(3, 4)")
    s.graph(_place(SQUARE, left), SQUARE_EDGES[:3], prefix="l.")
    s.graph(_place(SQUARE, right), SQUARE_EDGES[:3] + [("C", "D", {"style": "accent"})],
            prefix="r.")
    for p in ("l.", "r."):
        _node_ids(s, p, "CD")


@figure("add_edges_by_id", "mutation", "Add several edges by ID",
        "Before: four nodes tagged with their integer ids, A 1, B 2, C 3, D 4, with two edges, "
        "B to A to C; the ids 2, 3, and 4 are highlighted. After: the id pairs (2, 4) and (3, 4) "
        "become two new accent edges, B to D and C to D, closing the cycle.")
def add_edges_by_id(s: Scene) -> None:
    left, right = transform(s, "g", "g.add_edges_by_id([(2, 4), (3, 4)])")
    s.graph(_place(SQUARE, left), SQUARE_EDGES[:2], prefix="l.")
    s.graph(_place(SQUARE, right),
            SQUARE_EDGES[:2] + [(a, b, {"style": "accent"}) for a, b in SQUARE_EDGES[2:]],
            prefix="r.")
    for p in ("l.", "r."):
        _node_ids(s, p, "BCD")


# --- attribute setters -----------------------------------------------------------
#
# H layout, structure fixed. Values sit in ink badges (nodes) or edge labels;
# the elements named in the call are `soft` in both panels and their written
# values turn accent on the right. Same values as set_*_attrs_ws, so the
# selection versions read as the same operation.

X = {"A": 3, "B": 5, "C": 2, "D": 4, "E": 6, "F": 1}
W = dict(zip(H_EDGES, [3, 1, 4, 2, 6]))


def _x_badges(s: Scene, prefix: str, values: dict, changed) -> None:
    for k, v in values.items():
        s.badge(prefix + k, str(v), where="n" if H[k][1] == 0 else "s",
                color="accent" if k in changed else "ink")


@figure("set_node_attrs", "mutation", "Set node attributes",
        "Every node of a six-node graph carries a value x in a badge (A 3, B 5, C 2, D 4, E 6, "
        "F 1). The call names nodes B and F, outlined in the accent color; after it both read "
        "7, while the other four nodes keep their values.")
def set_node_attrs(s: Scene) -> None:
    left, right = transform(s, "g", 'g.set_node_attrs("x", 7, nodes=["B", "F"])')
    named = ("B", "F")
    styles = {k: "soft" for k in named}
    s.graph(_place(H, left), H_EDGES, styles=styles, prefix="l.")
    s.graph(_place(H, right), H_EDGES, styles=styles, prefix="r.")
    _x_badges(s, "l.", X, ())
    _x_badges(s, "r.", {k: 7 if k in named else v for k, v in X.items()}, named)


@figure("set_edge_attrs", "mutation", "Set edge attributes",
        "Every edge of a six-node graph carries a value w on its label. The call names the "
        "edge B to E, drawn in a light accent stroke, which holds 4; after it that edge reads "
        "5, in the accent color, while the other four edges keep their values.")
def set_edge_attrs(s: Scene) -> None:
    left, right = transform(s, "g", 'g.set_edge_attrs("w", 5, edges=[("B", "E")])')
    named = ("B", "E")
    s.graph(_place(H, left),
            [(a, b, {"weight": v, **({"style": "soft"} if (a, b) == named else {})})
             for (a, b), v in W.items()], prefix="l.")
    s.graph(_place(H, right),
            [(a, b, {"weight": 5, "style": "accent"}) if (a, b) == named
             else (a, b, {"weight": v}) for (a, b), v in W.items()], prefix="r.")


NAMES = {"A": "Ann", "B": "Bob", "C": "Cat", "D": "Dan", "E": "Eve", "F": "Fay"}


@figure("set_node_attr_to_display", "mutation", "Set display attribute",
        "The same six-node graph on both sides, every node carrying a name in a badge (Ann, "
        "Bob, Cat, Dan, Eve, Fay). The call changes no node data, so the badges are identical; "
        "it only records a display preference, shown as a tag above each graph that reads "
        "display: None before and display: name, in the accent color, after.")
def set_node_attr_to_display(s: Scene) -> None:
    left, right = transform(s, "g", 'g.set_node_attr_to_display("name")')
    for box, prefix in ((left, "l."), (right, "r.")):
        pos = _place(H, box)
        s.graph(pos, H_EDGES, prefix=prefix)
        for k, v in NAMES.items():
            s.badge(prefix + k, v, where="n" if H[k][1] == 0 else "s", color="ink")
    top = min(y for _, y in _place(H, left).values())
    s.text(left.cx, top - 44, "display: None", size=SIZE_NOTE, opacity=0.6)
    s.text(right.cx, top - 44, 'display: "name"', size=SIZE_NOTE, color="accent", weight=600)


# --- renaming ----------------------------------------------------------------
#
# relabel_nodes follows the plain setters: the renamed nodes are `soft` on the
# left and `accent` (with their new labels) on the right. Every node carries
# its integer id in an ink pill on both sides, since ids are what survive.

H_ID = {k: str(i) for i, k in enumerate(H, 1)}  # insertion order: A=1 … F=6


@figure("relabel_nodes", "mutation", "Relabel nodes",
        "Before: a six-node graph with each node's id in a pill, 1 to 6. The call renames B "
        "to X and E to Y; after it those two nodes carry their new labels in the accent "
        "color, while their ids 2 and 5, the edges, and the other labels are unchanged.")
def relabel_nodes(s: Scene) -> None:
    left, right = transform(s, "g", 'g.relabel_nodes({"B": "X", "E": "Y"})')
    new = {"B": "X", "E": "Y"}
    s.graph(_place(H, left), H_EDGES, styles={k: "soft" for k in new}, prefix="l.")
    s.graph(_place(H, right), H_EDGES, styles={k: "accent" for k in new}, labels=new,
            prefix="r.")
    for prefix in ("l.", "r."):
        for k, i in H_ID.items():
            s.badge(prefix + k, i, where="n" if H[k][1] == 0 else "s", color="ink")
