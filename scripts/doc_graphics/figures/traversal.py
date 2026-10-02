"""Traversals and neighborhood queries: what a search reaches from a node.

Template: `compute` for methods that return nodes. Left: the graph, start or
query node `soft`, the edges the search follows `accent`, everything else
plain. Visit orders carry filled badges (1 = start). Right: a card with the
returned nodes, in the order edgecase returns them.

The `trav_*` methods return nothing: they *move the selection* along edges.
They use `transform` instead: the selection (`accent`) on the left, the new
selection on the right. Node-to-node variants select nodes (edges stay ink);
the `*_edge` variants select edges (nodes stay ink), so the two kinds read
differently at a glance.

Shared layouts keep siblings comparable: `bfs`/`dfs` use one undirected graph,
and every `trav_*` method and neighbor query uses one directed graph (`DIR`)
around the node C, which has in-neighbors A and E and out-neighbors B and D.
"""

from core import SIZE_NOTE, Box, Scene, compute, figure, place, transform

UNIT = 64  # one density for the whole family

# --- bfs / dfs: undirected, columns = BFS depth from A ----------------------

SEARCH = {"A": (0, 0.8), "B": (1.4, 0), "C": (1.4, 1.6), "D": (2.8, 0), "E": (2.8, 1.6),
          "F": (4.2, 0.8)}
SEARCH_EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "E"), ("D", "F"),
                ("E", "F")]


def _search(s: Scene, call: str, order: str, tree: set) -> dict:
    left, right = compute(s, "g", call)
    box = Box(left.x0, left.y0 + 22, left.x1, left.y1)  # leave a row for d= labels
    pos = place(SEARCH, box, unit=UNIT)
    s.graph(pos, [(a, b, {"style": "accent" if (a, b) in tree else "default"})
                  for a, b in SEARCH_EDGES], styles={"A": "soft"})
    for i, k in enumerate(order, 1):
        s.badge(k, str(i), filled=True)
    s.card_centered(right, [("  ".join(order), "accent")], title="visit order")
    return pos


@figure("bfs", "traversal", "Breadth-first search",
        "A six-node graph searched breadth-first from A, with nodes in columns by distance "
        "from A (0 to 3). Numbered badges give the visit order, which finishes each column "
        "before starting the next, and the search tree's edges are in the accent color. "
        "The result card lists the order A, B, C, D, E, F.")
def bfs(s: Scene) -> None:
    pos = _search(s, 'g.bfs(start="A", return_list=True)', "ABCDEF",
                  {("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("D", "F")})
    for depth, k in enumerate("ABDF"):
        s.text(pos[k][0], 20, f"d={depth}", size=SIZE_NOTE, opacity=0.55)


@figure("dfs", "traversal", "Depth-first search",
        "The same six-node graph searched depth-first from A. Numbered badges show the "
        "search running down one branch, A, B, D, F, E, before backing up to reach C last; "
        "the accent edges form a single long path. The result card lists the order A, B, "
        "D, F, E, C.")
def dfs(s: Scene) -> None:
    _search(s, 'g.dfs(start="A", return_list=True)', "ABDFEC",
            {("A", "B"), ("B", "D"), ("D", "F"), ("E", "F"), ("C", "E")})


# --- shared directed graph for trav_* and neighbor queries --------------------

DIR = {"A": (0, 0), "B": (0, 1.6), "C": (1.05, 0.8), "D": (2.1, 0), "E": (2.1, 1.6),
       "F": (3.15, 0.8), "G": (4.2, 0.8)}
DIR_EDGES = [("A", "C"), ("C", "B"), ("C", "D"), ("E", "C"), ("D", "F"), ("F", "E"),
             ("F", "G")]
VALUES = {"A": 6, "B": 2, "C": 1, "D": 3, "E": 4, "F": 7, "G": 9}  # for the *_until pair


def _dir(s: Scene, box: Box, prefix: str, *, nodes=(), edges=(), style="accent",
         extra=()) -> dict:
    """Draw DIR in `box`; `nodes`/`edges` get `style` (edges as (a, b) pairs)."""
    pos = place(DIR, box, unit=UNIT)
    es = [(a, b, {"style": style}) if (a, b) in edges else (a, b) for a, b in DIR_EDGES]
    s.graph(pos, es + list(extra), directed=True, styles={k: style for k in nodes},
            prefix=prefix)
    return pos


def _trav_nodes(s: Scene, call: str, picked: str) -> None:
    left, right = transform(s, "g", call)
    _dir(s, left, "l.", nodes="C")
    _dir(s, right, "r.", nodes=picked)


def _trav_edges(s: Scene, call: str, picked: set) -> None:
    left, right = transform(s, "g", call)
    _dir(s, left, "l.", nodes="C")
    _dir(s, right, "r.", edges=picked)


@figure("trav_out", "traversal", "Outward traversal",
        "Left: in a seven-node directed graph, node C is selected (accent). Right: after "
        "trav_out() the selection has moved one step along C's outgoing edges, to B and D; "
        "C is no longer selected, and A and E, whose edges point into C, are not reached.")
def trav_out(s: Scene) -> None:
    _trav_nodes(s, "g.trav_out()", "BD")


@figure("trav_in", "traversal", "Inward traversal",
        "Left: in a seven-node directed graph, node C is selected (accent). Right: after "
        "trav_in() the selection has moved one step backward along the edges pointing into "
        "C, to A and E; C is no longer selected, and its successors B and D are not reached.")
def trav_in(s: Scene) -> None:
    _trav_nodes(s, "g.trav_in()", "AE")


@figure("trav_both", "traversal", "Bidirectional traversal",
        "Left: in a seven-node directed graph, node C is selected (accent). Right: after "
        "trav_both() the selection is every node one edge away from C in either direction, "
        "A, B, D, and E; C itself and the more distant F and G are not selected.")
def trav_both(s: Scene) -> None:
    _trav_nodes(s, "g.trav_both()", "ABDE")


@figure("trav_out_edge", "traversal", "Traverse to outbound edges",
        "Left: in a seven-node directed graph, node C is selected (accent). Right: after "
        "trav_out_edge() the node selection is gone and the two edges leaving C, C to B and "
        "C to D, are selected instead (accent).")
def trav_out_edge(s: Scene) -> None:
    _trav_edges(s, "g.trav_out_edge()", {("C", "B"), ("C", "D")})


@figure("trav_in_edge", "traversal", "Traverse to inbound edges",
        "Left: in a seven-node directed graph, node C is selected (accent). Right: after "
        "trav_in_edge() the node selection is gone and the two edges pointing into C, A to C "
        "and E to C, are selected instead (accent).")
def trav_in_edge(s: Scene) -> None:
    _trav_edges(s, "g.trav_in_edge()", {("A", "C"), ("E", "C")})


@figure("trav_reverse_edge", "traversal", "Traverse to reverse edges",
        "Left: two edges are selected in the accent color, C to D and A to C; C and D are "
        "also joined by a second edge from D back to C. Right: after trav_reverse_edge() "
        "only D to C is selected, the reverse of C to D; A to C has no reverse edge, so it "
        "contributes nothing.")
def trav_reverse_edge(s: Scene) -> None:
    left, right = transform(s, "g", "g.trav_reverse_edge()")
    # The C-D pair is reciprocal: bow both edges apart (same bend, opposite sides).
    for box, p, sel in ((left, "l.", {("A", "C"), ("C", "D")}), (right, "r.", {("D", "C")})):
        pos = place(DIR, box, unit=UNIT)
        es = [(a, b, {"style": "accent" if (a, b) in sel else "default",
                      **({"bend": 9} if {a, b} == {"C", "D"} else {})})
              for a, b in DIR_EDGES + [("D", "C")]]
        s.graph(pos, es, directed=True, prefix=p)


def _until(s: Scene, call: str, picked: str) -> None:
    left, right = transform(s, "g", call)
    pos = _dir(s, left, "l.", nodes="C")
    # Values sit above each node (below for the bottom row) so they read as a set.
    for k, v in VALUES.items():
        s.badge("l." + k, str(v), where="s" if k in "BE" else "n")
    # The condition goes in the free corner under F and G.
    s.text((pos["F"][0] + pos["G"][0]) / 2, 172, "cond: value ≥ 5", size=SIZE_NOTE,
           opacity=0.6)
    _dir(s, right, "r.", nodes=picked)


@figure("trav_out_until", "traversal", "Outward traversal until a condition",
        "Left: node C is selected in a seven-node directed graph whose nodes carry values; "
        "the condition is value at least 5. Right: walking forward from C through B and D, "
        "the first match is F (value 7), which becomes the selection. G also matches but is "
        "never reached, because the walk stops at F; A matches but lies against the arrows.")
def trav_out_until(s: Scene) -> None:
    _until(s, "g.trav_out_until(cond)", "F")


@figure("trav_in_until", "traversal", "Inward traversal until a condition",
        "Left: node C is selected in a seven-node directed graph whose nodes carry values; "
        "the condition is value at least 5. Right: walking backward against the arrows from "
        "C, the walk stops at A (value 6) on one branch and at F (value 7), via E, on the "
        "other, so A and F become the selection.")
def trav_in_until(s: Scene) -> None:
    _until(s, "g.trav_in_until(cond)", "AF")


# --- neighbor queries on DIR ------------------------------------------------


def _query(s: Scene, call: str, found: str, edges: set, title: str) -> None:
    left, right = compute(s, "g", call)
    pos = place(DIR, left, unit=UNIT)
    es = [(a, b, {"style": "accent"}) if (a, b) in edges else (a, b) for a, b in DIR_EDGES]
    s.graph(pos, es, directed=True, styles={"C": "soft", **{k: "accent" for k in found}})
    s.card_centered(right, [("  ".join(found), "accent")], title=title)


@figure("neighbors", "traversal", "Neighbors",
        "A seven-node directed graph with the query node C outlined. With mode all, every "
        "node joined to C by an edge in either direction is returned, so B, D, A, and E and "
        "their edges to C are in the accent color. The card lists B, D, A, E: successors "
        "first, then predecessors.")
def neighbors(s: Scene) -> None:
    _query(s, 'g.neighbors("C", mode="all")', "BDAE",
           {("A", "C"), ("C", "B"), ("C", "D"), ("E", "C")}, "neighbors of C")


@figure("get_successors", "traversal", "Successors",
        "A seven-node directed graph with the query node C outlined. The two edges leaving "
        "C and their targets, B and D, are in the accent color; A and E point into C and "
        "are not successors. The card lists B, D.")
def get_successors(s: Scene) -> None:
    _query(s, 'g.get_successors("C")', "BD", {("C", "B"), ("C", "D")}, "successors of C")


@figure("get_predecessors", "traversal", "Predecessors",
        "A seven-node directed graph with the query node C outlined. The two edges pointing "
        "into C and their sources, A and E, are in the accent color; B and D are targets of "
        "C, not predecessors. The card lists A, E.")
def get_predecessors(s: Scene) -> None:
    _query(s, 'g.get_predecessors("C")', "AE", {("A", "C"), ("E", "C")},
           "predecessors of C")


@figure("get_non_neighbors", "traversal", "Non-neighbors",
        "A seven-node directed graph with the query node C outlined. C is joined to A, B, "
        "D, and E in one direction or the other, so the only nodes returned are F and G, "
        "shown in the accent color. The card lists F, G.")
def get_non_neighbors(s: Scene) -> None:
    _query(s, 'g.get_non_neighbors("C")', "FG", set(), "non-neighbors of C")


# --- reachability and ordering ----------------------------------------------


@figure("get_all_connected_nodes", "traversal", "All connected nodes",
        "A directed graph in two pieces, with the start node A outlined. Ignoring arrow "
        "direction, A reaches B, C, and D, shown in the accent color with the edges used; "
        "the separate pair E and F is not reached. The card lists B, C, D, without A itself.")
def get_all_connected_nodes(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_all_connected_nodes("A")')
    coords = {"A": (0, 0), "B": (1.4, 0), "C": (1.4, 1.6), "D": (2.8, 1.6),
              "E": (4.2, 0), "F": (4.2, 1.6)}
    pos = place(coords, left, unit=UNIT)
    s.graph(pos, [("A", "B", {"style": "accent"}), ("C", "B", {"style": "accent"}),
                  ("C", "D", {"style": "accent"}), ("E", "F")], directed=True,
            styles={"A": "soft", "B": "accent", "C": "accent", "D": "accent"})
    s.card_centered(right, [("B  C  D", "accent")], title="connected to A")


@figure("get_topological_sort_order", "traversal", "Topological sort",
        "A six-node directed acyclic graph laid out left to right. Numbered badges give the "
        "topological order, in which every node comes after all nodes with arrows into it. "
        "The result card lists A, B, D, C, E, F.")
def get_topological_sort_order(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_topological_sort_order()")
    coords = {"A": (0, 0), "B": (0, 1.6), "C": (1.4, 1.6), "D": (1.4, 0), "E": (2.8, 0.8),
              "F": (4.2, 0.8)}
    pos = place(coords, left, unit=UNIT)
    s.graph(pos, [("A", "C"), ("B", "C"), ("A", "D"), ("C", "E"), ("D", "E"), ("E", "F")],
            directed=True)
    order = "ABDCEF"
    for i, k in enumerate(order, 1):
        s.badge(k, str(i), filled=True)
    s.card_centered(right, [("  ".join(order), "accent")], title="topological order")
