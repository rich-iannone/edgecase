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

DAG queries (ancestors, descendants, lowest_common_ancestor,
topological_generations) share one seven-node DAG (`DAG`) laid out left to
right by generation. The query node(s) are `soft`, the returned nodes and
the edges that reach them `accent`; topological_generations instead gives
each generation a group color and hull, with a card listing the layers. all_topological_sorts needs a DAG with only a few orderings, so
it uses a five-node bowtie (`BOWTIE`, two sources, one hub, two sinks) in the
same left-to-right style, with every ordering listed on the card.

Control flow and causal DAGs follow their docstring examples, with the node
names (too long for a circle) set beside the nodes. The dominator pair shares
one if/else control-flow graph (`CFG`) laid out left to right from `entry`
(`soft`). immediate_dominators draws the dominator tree over it: CFG edges
that are tree edges in accent, the two branch-to-merge edges muted, and the
tree edge if -> merge, which is no CFG edge, as a dashed accent (`ghost`)
arrow; the card lists every node's immediate dominator. dominance_frontiers
outlines the two branches and fills their frontier, merge, in accent along
the edges that reach it. The d-separation methods color x `g2` and y `g3`
and name the conditioning set z in a note: the two predicates use
`contrast`, z `soft`, with the open path (or the redundant member of z) in
accent on the False side; find_minimal_d_separator returns z, so it is in
accent. Every figure asserts its result with edgecase.
"""

from core import SIZE_NOTE, Box, Scene, compute, contrast, figure, place, transform

from figures.similarity import _name

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


# --- DAG queries on one shared DAG -------------------------------------------

# Seven nodes in four topological generations, laid out left to right by
# generation: [A, B], [C, D], [E, F], [G]. Shared with the DAG longest-path
# figures in paths.py (which add durations as weights).
DAG = {"A": (0, 0), "B": (0, 1.4), "C": (1.2, 0), "D": (1.2, 1.4), "E": (2.4, 0),
       "F": (2.4, 1.4), "G": (3.6, 0.7)}
DAG_EDGES = [("A", "C"), ("A", "D"), ("B", "D"), ("C", "E"), ("D", "E"), ("D", "F"),
             ("E", "G"), ("F", "G")]
GENERATIONS = [list("AB"), list("CD"), list("EF"), ["G"]]


def _dag_query(s: Scene, call: str, *, query: str, found: str, edges: set,
               lines: list, title: str) -> None:
    left, right = compute(s, "g", call)
    pos = place(DAG, left, unit=UNIT)
    s.graph(pos, [(a, b, {"style": "accent"}) if (a, b) in edges else (a, b)
                  for a, b in DAG_EDGES], directed=True,
            styles={**{k: "soft" for k in query}, **{k: "accent" for k in found}})
    s.card_centered(right, lines, title=title)


@figure("ancestors", "traversal", "Ancestors",
        "A seven-node DAG flowing left to right, with the query node F outlined. Following "
        "arrows backward from F reaches its parent D and, through D, both A and B, shown in "
        "the accent color with the edges used. C and E cannot reach F. The card lists A, B, D.")
def ancestors(s: Scene) -> None:
    _dag_query(s, 'g.ancestors("F")', query="F", found="ABD",
               edges={("A", "D"), ("B", "D"), ("D", "F")},
               lines=[("A  B  D", "accent")], title="nodes that can reach F")


@figure("descendants", "traversal", "Descendants",
        "A seven-node DAG flowing left to right, with the query node C outlined. Following "
        "arrows forward from C reaches E and then G, shown in the accent color with the edges "
        "used; D and F lie on another branch and are not reached. The card lists E, G.")
def descendants(s: Scene) -> None:
    _dag_query(s, 'g.descendants("C")', query="C", found="EG",
               edges={("C", "E"), ("E", "G")},
               lines=[("E  G", "accent")], title="nodes reachable from C")


@figure("lowest_common_ancestor", "traversal", "Lowest common ancestor",
        "A seven-node DAG with the query nodes E and F outlined. A, B, and D all have paths "
        "to both, but D is the closest, so it is the lowest common ancestor, shown in the "
        "accent color with its edges to E and F. The card lists the common ancestors A, B, D "
        "and the answer D.")
def lowest_common_ancestor(s: Scene) -> None:
    _dag_query(s, 'g.lowest_common_ancestor("E", "F")', query="EF", found="D",
               edges={("D", "E"), ("D", "F")},
               lines=["common  A  B  D", ("lowest  D", "accent")], title="ancestors of E and F")


@figure("topological_generations", "traversal", "Topological generations",
        "A seven-node DAG laid out in four columns, one per generation, each in its own color "
        "and tinted region: the sources A, B (accent), then C, D (teal), E, F (violet), and "
        "finally G (gold). Every arrow points to a later generation. The card lists the "
        "generations 0 to 3.")
def topological_generations(s: Scene) -> None:
    left, right = compute(s, "g", "g.topological_generations()")
    colors = ("accent", "g2", "g3", "g4")
    grp = {k: colors[i] for i, gen in enumerate(GENERATIONS) for k in gen}
    s.graph(place(DAG, left, unit=UNIT), DAG_EDGES, directed=True, styles=grp)
    for i, gen in enumerate(GENERATIONS):
        s.hull(gen, colors[i])
    s.card_centered(right, [(f"{i} → {', '.join(gen)}", colors[i])
                            for i, gen in enumerate(GENERATIONS)], title="generation")


# A bowtie: A and B feed C, which feeds D and E. Only the two pairs can swap,
# so it has exactly 2 x 2 = 4 orderings.
BOWTIE = {"A": (0, 0), "B": (0, 1.4), "C": (1.2, 0.7), "D": (2.4, 0), "E": (2.4, 1.4)}
BOWTIE_EDGES = [("A", "C"), ("B", "C"), ("C", "D"), ("C", "E")]
BOWTIE_SORTS = ["ABCDE", "ABCED", "BACDE", "BACED"]


@figure("all_topological_sorts", "traversal", "All topological sorts",
        "A five-node DAG flowing left to right: A and B both point to C, which points to D "
        "and E. A and B can come in either order, and so can D and E, while C is always "
        "third. The card lists all four orderings in lexicographic order, from A, B, C, D, E "
        "to B, A, C, E, D.")
def all_topological_sorts(s: Scene) -> None:
    left, right = compute(s, "g", "g.all_topological_sorts()")
    s.graph(place(BOWTIE, left, unit=UNIT), BOWTIE_EDGES, directed=True)
    s.card_centered(right, [("  ".join(o), "accent") for o in BOWTIE_SORTS],
                    title="every valid order")


# --- search trees: edges, trees, layers, postorder ------------------------------

# The bfs_edges / dfs_edges docstring graph: a triangle A, B, C with a tail C-D,
# plus a separate pair E-F that only a source-less search reaches. Edge insertion
# order (A-B before A-C) fixes the order neighbors are explored.
TRI = {"A": (0, 0.8), "B": (1.3, 0), "C": (1.3, 1.6), "D": (2.6, 1.6), "E": (4.0, 0),
       "F": (4.0, 1.6)}
TRI_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("E", "F")]


def _tree_edges(s: Scene, call: str, tree: list, *, start: str = "", unreached: str = "") -> None:
    """TRI with the returned (parent, child) edges numbered in discovery order."""
    left, right = compute(s, "g", call)
    rank = {e: i for i, e in enumerate(tree, 1)}
    edges = []
    for a, b in TRI_EDGES:
        e = (a, b) if (a, b) in rank else (b, a) if (b, a) in rank else None
        if e:
            edges.append((e[0], e[1], {"style": "accent", "weight": str(rank[e])}))
        else:
            edges.append((a, b, {"style": "muted" if a in unreached else "default"}))
    s.graph(place(TRI, left, unit=UNIT), edges,
            styles={**({start: "soft"} if start else {}), **{k: "muted" for k in unreached}})
    s.card_centered(right, [(f"{i}  ({a}, {b})", "accent") for (a, b), i in rank.items()],
                    title="(parent, child) pairs")


@figure("bfs_edges", "traversal", "BFS edges",
        "A triangle A, B, C with a tail C to D, searched breadth-first from A (outlined). "
        "The tree edges are numbered in discovery order: A finds both B and C before moving "
        "on, so C is reached directly from A and the edge B-C is not used; the separate pair "
        "E-F is never reached. The card lists (A, B), (A, C), (C, D).")
def bfs_edges(s: Scene) -> None:
    _tree_edges(s, 'g.bfs_edges("A")', [("A", "B"), ("A", "C"), ("C", "D")], start="A",
                unreached="EF")


@figure("dfs_edges", "traversal", "DFS edges",
        "The same graph searched depth-first with no source. The tree edges are numbered in "
        "discovery order: from A the search dives to B and then C, so C is reached through B "
        "rather than from A, and the edge A-C is not used. It then restarts at E for the "
        "separate pair. The card lists (A, B), (B, C), (C, D), (E, F).")
def dfs_edges(s: Scene) -> None:
    _tree_edges(s, "g.dfs_edges()", [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")])


def _search_tree(s: Scene, call: str, tree: list, depth: int) -> None:
    """SEARCH → the returned tree on the same layout: tree arcs accent, the rest dropped."""
    left, right = transform(s, "g", call)
    for box, p in ((left, "l."), (right, "r.")):
        gb = Box(box.x0, box.y0 + 22, box.x1, box.y1)  # leave a row for the depth note
        pos = place(SEARCH, gb, unit=UNIT)
        if p == "l.":
            s.graph(pos, SEARCH_EDGES, styles={"A": "soft"}, prefix=p)
            continue
        dropped = [(a, b, {"style": "removed"}) for a, b in SEARCH_EDGES
                   if (a, b) not in tree and (b, a) not in tree]
        s.graph(pos, [(a, b, {"style": "accent", "directed": True}) for a, b in tree] + dropped,
                styles={"A": "soft"}, prefix=p)
        s.text(box.cx, 20, f"tree depth {depth}", size=SIZE_NOTE, opacity=0.6)


@figure("bfs_tree", "traversal", "BFS tree",
        "Left: a six-node graph with the root A outlined. Right: bfs_tree() returns a new "
        "directed graph keeping only the edges the breadth-first search discovered, A to B, "
        "A to C, B to D, B to E, and D to F, as accent arrows; the unused edges C-E and E-F "
        "are dashed. Every root path is a fewest-hops path, so the tree has depth 3.")
def bfs_tree(s: Scene) -> None:
    _search_tree(s, 'g.bfs_tree("A")', [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"),
                                        ("D", "F")], 3)


@figure("dfs_tree", "traversal", "DFS tree",
        "Left: the same six-node graph with the root A outlined. Right: dfs_tree() keeps the "
        "edges the depth-first search discovered, which snake through the graph as one long "
        "path, A to B, D, F, E, and finally C (accent arrows); A-C and B-E are dashed. The "
        "tree has depth 5, against 3 for the BFS tree.")
def dfs_tree(s: Scene) -> None:
    _search_tree(s, 'g.dfs_tree("A")', [("A", "B"), ("B", "D"), ("D", "F"), ("F", "E"),
                                        ("E", "C")], 5)


# Two sources at opposite ends, so the layers grow inward and meet in the middle.
SPINE = {"A": (0, 0.8), "B": (1, 0), "C": (1, 1.6), "D": (2, 0.8), "E": (3, 0.8),
         "F": (4, 0), "G": (4, 1.6), "H": (5, 0.8)}
SPINE_EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E"), ("E", "F"),
               ("E", "G"), ("F", "H"), ("G", "H")]
LAYERS = [list("AH"), list("BCFG"), list("DE")]  # g.bfs_layers(["A", "H"])


@figure("bfs_layers", "traversal", "BFS layers",
        "An eight-node graph searched from two sources at opposite ends, A and H. Each node "
        "is colored by its hop distance to the nearer source: A and H form layer 0 (accent), "
        "B, C, F, G layer 1 (teal), and D, E, where the two searches meet, layer 2 (violet). "
        "The card lists the three layers.")
def bfs_layers(s: Scene) -> None:
    left, right = compute(s, "g", 'g.bfs_layers(["A", "H"])')
    colors = ("accent", "g2", "g3")
    grp = {k: colors[i] for i, layer in enumerate(LAYERS) for k in layer}
    s.graph(place(SPINE, left, unit=UNIT), SPINE_EDGES, styles=grp)
    # one tinted region per side: a layer is split between the two sources
    for i, layer in enumerate(LAYERS):
        for side in ("ABCD", "EFGH"):
            part = [k for k in layer if k in side]
            if part:
                s.hull(part, colors[i])
    s.card_centered(right, [(f"{i} → {', '.join(layer)}", colors[i])
                            for i, layer in enumerate(LAYERS)], title="layer")


# A binary tree drawn top-down; postorder lists leaves before their parents.
BTREE = {"A": (1.5, 0), "B": (0.5, 1), "C": (2.5, 1), "D": (0, 2), "E": (1, 2), "F": (2, 2),
         "G": (3, 2)}
BTREE_EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")]
POSTORDER = "DEBFGCA"  # g.dfs_postorder_nodes("A"); preorder (dfs) is ABDECFG


@figure("dfs_postorder_nodes", "traversal", "DFS postorder",
        "A seven-node tree searched depth-first from the root A (outlined). Numbered badges "
        "give the order nodes finish: D and E finish before their parent B, F and G before C, "
        "and the root A finishes last. The card lists D, E, B, F, G, C, A, beside the "
        "preorder A, B, D, E, C, F, G that dfs() returns.")
def dfs_postorder_nodes(s: Scene) -> None:
    left, right = compute(s, "g", 'g.dfs_postorder_nodes("A")')
    s.graph(place(BTREE, left, unit=UNIT), [(a, b, {"style": "accent"}) for a, b in BTREE_EDGES],
            styles={"A": "soft"})
    for i, k in enumerate(POSTORDER, 1):
        s.badge(k, str(i), filled=True, where="e" if k == "A" else "auto")
    s.card_centered(right, [("  ".join(POSTORDER), "accent"), "dfs()  A B D E C F G"],
                    title="finish order")


# --- later additions: dominators -------------------------------------------
#
# The dominator docstrings' if/else control-flow graph, left to right.
CFG = {"entry": (0, 0.5), "if": (1, 0.5), "then": (2, 0), "else": (2, 1), "merge": (3, 0.5),
       "exit": (4, 0.5)}
CFG_EDGES = [("entry", "if"), ("if", "then"), ("if", "else"), ("then", "merge"),
             ("else", "merge"), ("merge", "exit")]
CFG_NAME_AT = {"then": "n"}  # every other name sits below its node


def _cfg_graph():
    import edgecase as ec

    g = ec.Graph(directed=True).add_nodes(list(CFG))
    g.add_edges(CFG_EDGES)
    return g


def _cfg(s: Scene, box: Box, edges: list, styles: dict) -> None:
    gbox = Box(box.x0, box.y0 + 14, box.x1, box.y1 - 14)  # rows for names above and below
    s.graph(place(CFG, gbox, unit=UNIT * 1.3), edges, directed=True, styles=styles, labels=False)
    for k in CFG:
        st = styles.get(k, "default")
        _name(s, k, k, CFG_NAME_AT.get(k, "s"), "ink" if st == "default" else "accent")


@figure("immediate_dominators", "traversal", "Immediate dominators",
        "An if/else control-flow graph: entry (outlined) leads to if, which branches to then "
        "and else; both rejoin at merge, which leads to exit. The dominator tree is drawn in "
        "the accent color: every path to merge passes through if but through neither branch, "
        "so the dashed accent arrow if to merge replaces the faded branch edges. The card "
        "maps each node to its immediate dominator: if for then, else and merge, merge for "
        "exit.")
def immediate_dominators(s: Scene) -> None:
    g = _cfg_graph()
    idom = {g.get_label_from_id(n): g.get_label_from_id(d)
            for n, d in g.immediate_dominators("entry").items()}
    assert idom == {"entry": "entry", "if": "entry", "then": "if", "else": "if", "merge": "if",
                    "exit": "merge"}
    left, right = compute(s, "g", 'g.immediate_dominators("entry")')
    tree = {(d, n) for n, d in idom.items() if n != d}
    edges = [(a, b, {"style": "accent" if (a, b) in tree else "muted"}) for a, b in CFG_EDGES]
    edges += [(d, n, {"style": "ghost"}) for d, n in sorted(tree) if (d, n) not in CFG_EDGES]
    _cfg(s, left, edges, {"entry": "soft"})
    s.card_centered(right, [(f"{n:<6} {d}", "accent" if n == "merge" else "ink")
                            for n, d in idom.items()], title="node   immediate dominator")


@figure("dominance_frontiers", "traversal", "Dominance frontiers",
        "The same if/else control-flow graph from entry. The branches then and else are "
        "outlined: each dominates only itself, and its edge into merge (accent) leaves its "
        "region, so merge, filled in accent, is the dominance frontier of both. The card "
        "lists every node's frontier: merge for then and else, and empty for the others.")
def dominance_frontiers(s: Scene) -> None:
    g = _cfg_graph()
    df = {g.get_label_from_id(n): sorted(g.get_label_from_id(f) for f in fs)
          for n, fs in g.dominance_frontiers("entry").items()}
    assert df == {"entry": [], "if": [], "then": ["merge"], "else": ["merge"], "merge": [],
                  "exit": []}
    left, right = compute(s, "g", 'g.dominance_frontiers("entry")')
    hot = {("then", "merge"), ("else", "merge")}
    _cfg(s, left, [(a, b, {"style": "accent"}) if (a, b) in hot else (a, b) for a, b in CFG_EDGES],
         {"then": "soft", "else": "soft", "merge": "accent"})
    s.card_centered(right, [(f"{n:<6} {{{', '.join(f)}}}" if f else f"{n:<6} {{}}",
                             "accent" if f else "ink") for n, f in df.items()],
                    title="node   dominance frontier")


# --- later additions: d-separation ------------------------------------------
#
# Causal DAGs from the docstrings; x is g2, y is g3, z is named in a note.
SMOKING = {"smoking": (0, 0), "tar": (1.2, 0), "cancer": (2.4, 0), "fingers": (1.2, 1)}
SMOKING_EDGES = [("smoking", "tar"), ("tar", "cancer"), ("smoking", "fingers")]


def _dag_graph(nodes, edges):
    import edgecase as ec

    g = ec.Graph(directed=True).add_nodes(list(nodes))
    g.add_edges(edges)
    return g


def _named_dag(s: Scene, box: Box, coords: dict, edges: list, styles: dict, prefix: str,
               below: set, unit: float) -> None:
    """A causal DAG with names beside its nodes (`below` under them, the rest above)."""
    gbox = Box(box.x0, box.y0 + 16, box.x1, box.y1 - 16)
    s.graph(place(coords, gbox, unit=unit), edges, directed=True, styles=styles, labels=False,
            prefix=prefix)
    for k in coords:
        st = styles.get(k, "default")
        color = {"g2": "g2", "g3": "g3", "default": "ink"}.get(st, "accent")
        _name(s, prefix + k, k, "s" if k in below else "n", color)


@figure("is_d_separator", "traversal", "Is d-separator",
        "A causal graph: smoking (teal) causes tar, tar causes cancer (violet), and smoking "
        "also causes yellow fingers. Left: conditioning on tar (outlined) blocks the only "
        "path from smoking to cancer, so is_d_separator returns True. Right: with nothing "
        "conditioned on, the chain smoking, tar, cancer (accent) is open, so smoking and "
        "cancer are dependent and is_d_separator returns False.")
def is_d_separator(s: Scene) -> None:
    g = _dag_graph(SMOKING, SMOKING_EDGES)
    assert g.is_d_separator("smoking", "cancer", "tar") is True
    assert g.is_d_separator("smoking", "cancer", []) is False
    left, right = contrast(s, 'g.is_d_separator("smoking", "cancer", z)',
                           'g.is_d_separator("smoking", "cancer", z)')
    xy = {"smoking": "g2", "cancer": "g3"}
    chain = {("smoking", "tar"), ("tar", "cancer")}
    for box, p, z in ((left, "l.", ["tar"]), (right, "r.", [])):
        s.text(box.cx, box.y0 + 8, f"z = [{', '.join(z)}]", size=SIZE_NOTE, opacity=0.6)
        gbox = Box(box.x0, box.y0 + 14, box.x1, box.y1)
        edges = [(a, b, {"style": "accent"}) if not z and (a, b) in chain else (a, b)
                 for a, b in SMOKING_EDGES]
        _named_dag(s, gbox, SMOKING, edges, {**xy, **dict.fromkeys(z, "soft")}, p,
                   below={"fingers"}, unit=UNIT * 1.4)


CHAIN4 = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0)}
CHAIN4_EDGES = [("A", "B"), ("B", "C"), ("C", "D")]


@figure("is_minimal_d_separator", "traversal", "Is minimal d-separator",
        "The chain A to B to C to D, with A in teal and D in violet. Left: z = [B] (outlined) "
        "blocks the chain, and dropping B would open it, so is_minimal_d_separator returns "
        "True. Right: z = [B, C] also separates A from D, but C (accent) is not needed, since "
        "B alone already blocks the chain, so the set is not minimal and the call returns "
        "False.")
def is_minimal_d_separator(s: Scene) -> None:
    g = _dag_graph(CHAIN4, CHAIN4_EDGES)
    assert g.is_minimal_d_separator("A", "D", ["B"]) is True
    assert g.is_minimal_d_separator("A", "D", ["B", "C"]) is False
    assert g.is_d_separator("A", "D", ["B", "C"]) is True
    left, right = contrast(s, 'g.is_minimal_d_separator("A", "D", z)',
                           'g.is_minimal_d_separator("A", "D", z)')
    for box, p, z in ((left, "l.", "B"), (right, "r.", "BC")):
        s.text(box.cx, box.y0 + 8, f"z = [{', '.join(z)}]", size=SIZE_NOTE, opacity=0.6)
        gbox = Box(box.x0, box.y0 + 22, box.x1, box.y1)
        styles = {"A": "g2", "D": "g3", "B": "soft"}
        if "C" in z:
            styles["C"] = "accent"
        s.graph(place(CHAIN4, gbox, unit=UNIT * 1.3), CHAIN4_EDGES, directed=True, styles=styles,
                prefix=p)


# find_minimal_d_separator docstring: age acts on heart health only through exercise.
HEART = {"age": (0, 0.8), "exercise": (1.3, 0.8), "weight": (2.3, 0), "heart": (3.3, 0.8)}
HEART_EDGES = [("age", "exercise"), ("exercise", "weight"), ("weight", "heart"),
               ("exercise", "heart")]


@figure("find_minimal_d_separator", "traversal", "Find minimal d-separator",
        "A causal graph: age (teal) affects exercise, exercise affects weight, and both "
        "exercise and weight affect heart health (violet). Every path from age to heart "
        "starts with the edge into exercise, so conditioning on exercise alone, in the "
        "accent color, blocks them all. The card returns the minimal separator [exercise].")
def find_minimal_d_separator(s: Scene) -> None:
    g = _dag_graph(HEART, HEART_EDGES)
    z = [g.get_label_from_id(n) for n in g.find_minimal_d_separator("age", "heart")]
    assert z == ["exercise"]
    g.add_edge("age", "heart")
    assert g.find_minimal_d_separator("age", "heart") is None
    left, right = compute(s, "g", 'g.find_minimal_d_separator("age", "heart")')
    _named_dag(s, left, HEART, HEART_EDGES,
               {"age": "g2", "heart": "g3", "exercise": "accent"}, "",
               below={"age", "exercise", "heart"}, unit=UNIT * 1.4)
    s.card_centered(right, [(f"[{', '.join(z)}]", "accent"), "blocks all age–heart paths"],
                    title="minimal separator z")
