"""Mutations: the graph before → after one change.

Template: `transform`. The left panel is the graph before the call, the right
panel after. The element being added is `accent` on the right; an element
being removed is `accent` on the left and `removed` (dashed) on the right, so
the reader sees *what* goes and *where it was*. Both panels share one layout
so nothing moves except the thing that changed.

Two base layouts on one grid (columns 1.2 apart, rows 1 apart) keep siblings
comparable:

- `SQUARE` (A–D, a 4-cycle) for node/edge add/delete and the clean-up trio
  (`simplify`, `remove_selfloops`, `remove_duplicate_edges`), which all start
  from the same multigraph: a self-loop on B and a doubled C–D edge.
- `H` (A–F, two rows joined by the B–E rung) for contractions and the two
  attribute mutations. In contractions the surviving node `u` is `soft` in
  both panels; the absorbed node and its edges are `accent` → `removed`; the
  rerouted edges (and any self-loop the merge creates) are `accent` on the
  right.

Attribute mutations keep the structure fixed and change node appearance:
the left panel shows the source values as ink badges, the right panel the
written result (colors / sizes) with the badges in the result colors.
"""

from core import Box, Scene, figure, place, transform

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
