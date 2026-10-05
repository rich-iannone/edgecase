"""Paths: shortest paths, path enumeration, trails, walks, and cycles.

Template: `compute`. Left: the graph, the endpoints you pass in `soft`, the
returned path's nodes and edges `accent`, everything else `muted`. When the
method returns several paths or cycles they are drawn in accent / g2 / g3;
an edge used by two of them is split into a bowed pair so both colors stay
visible. Right: a card listing each path in its matching color (with the
cost when the method is weighted), or a compact matrix for all-pairs and
walk counts. Paths read `A → B → C`; cycles, returned without repeating the
start node, read `[A, B, C]`.

Shared layouts keep siblings comparable:
- SP: one directed, weighted 6-node graph (A to F by three disjoint routes)
  for Bellman-Ford, ADA*, k-shortest, simple paths, the AdaStar planner, and
  the hop-count BFS shortest_path.
- SQUARE: a 4-node graph for the all-pairs matrices and walk counts.
- HOUSE: a 5-node graph (a triangle on a square) for every cycle method; its
  directed version (DHOUSE_EDGES) also hosts both random walks, whose visit
  positions (0 = start) are filled badges on nodes or labels on edges.
- SANTA_EDGES: HOUSE plus both diagonals (the house of Santa Claus) for
  eulerian_path, which mirrors eulerian_circuit: every edge `accent` and
  numbered in trail order, the two odd ends `soft`. predicates.py reuses it
  for has_eulerian_path, transforms.py reuses HOUSE for eulerize.
- TRAIL: a spine A-B-C-D with a directed loop at B, where trails, paths,
  and walks differ.
- SP also hosts the shortest-path queries: shortest_path_length (weighted
  route + hop count on the card), all_shortest_paths (unweighted, two tied
  routes in accent / g2), and the single-source pair, which draw the
  Dijkstra tree from the `soft` source in accent (distances as badges for
  the *_length variant). bidirectional_dijkstra shades the nodes each end
  actually settles (forward g2, backward g3) behind the accent path.
- LADDER: a 2 x 4 ladder with sources A and G for the multi-source pair.
  The path variant colors every node and tree edge by its nearest source's
  catchment (accent / g2), marking each source with a halo (a one-node
  hull); the *_length variant keeps the sources `soft` with distance badges.
- DAG (from traversal.py) with durations as weights for dag_longest_path
  and dag_longest_path_length: the critical path in accent, the rest muted.
- BOWTIE: two triangles sharing C plus a detour A - F - E, for the Menger
  pair. edge_disjoint_paths finds three routes from A to E (two meet at C);
  node_disjoint_paths keeps only two, and the route through B, D stays muted.
"""

import math

from core import SIZE_NOTE, Box, Scene, compute, figure, place, ring

UNIT = 72  # one graph scale for the whole family
COLORS = ("accent", "g2", "g3")
GAP = 5  # apex offset (px) of each stroke when two routes share an edge

# --- shared layouts ----------------------------------------------------------

SP = {"A": (0, 0.85), "B": (1.2, 0), "C": (2.4, 0), "D": (1.8, 0.85), "E": (1.8, 1.7),
      "F": (3.6, 0.85)}
SP_EDGES = [("A", "B", 1), ("B", "C", 2), ("C", "F", 1), ("A", "D", 3), ("D", "F", 3),
            ("A", "E", 2), ("E", "F", 3)]
TOP, MID, LOW = list("ABCF"), list("ADF"), list("AEF")  # costs 4, 6, 5

SQUARE = {"A": (0, 0), "B": (1.3, 0), "C": (1.3, 1.3), "D": (0, 1.3)}
AP_EDGES = [("A", "B", 1), ("B", "C", 2), ("A", "C", 4), ("C", "D", 1), ("D", "A", 2)]

HOUSE = {"A": (0.75, 0), "B": (0, 0.85), "C": (1.5, 0.85), "D": (0, 1.75), "E": (1.5, 1.75)}
HOUSE_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E")]
# The "house of Santa Claus": HOUSE plus both diagonals of the square, in the
# eulerian_path docstring's insertion order (bl-br, bl-tl, br-tr, tl-tr, ...).
SANTA_EDGES = [("D", "E"), ("D", "B"), ("E", "C"), ("B", "C"), ("B", "A"), ("C", "A"),
               ("D", "C"), ("E", "B")]

TRAIL = {"A": (0, 1), "B": (1, 1), "C": (2, 1), "D": (3, 1), "E": (0.5, 0), "F": (1.5, 0)}
TRAIL_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("B", "E"), ("E", "F"), ("F", "B")]


def bracket(nodes: list[str]) -> str:
    return "[" + ", ".join(nodes) + "]"


# --- drawing helper ----------------------------------------------------------


def _steps(route: list[str], closed: bool) -> list[tuple[str, str]]:
    pairs = list(zip(route, route[1:]))
    return pairs + [(route[-1], route[0])] if closed else pairs


def _area(pts: list[tuple[float, float]]) -> float:
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))) / 2


def routes(s: Scene, pos: dict, edges: list, rts: list[list[str]], *, directed: bool,
           closed: bool = False, ends: tuple[str, ...] = (), rest: str = "muted",
           colors: tuple[str, ...] = COLORS, sides: list[tuple[float, float]] | None = None,
           weights: bool = False) -> None:
    """Draw `edges` with each route (node list) in its own color.

    An edge used by one route takes that route's color; an edge shared by two
    routes is drawn twice, bowed apart, each stroke on the side of its own
    route (its centroid, or the anchor point in `sides`). If both routes lie
    on the same side, the one enclosing the larger area moves outward.
    Nodes on exactly one route take its color, `ends` are `soft`.
    """
    def uses(rt: list[str], a: str, b: str) -> bool:
        st = _steps(rt, closed)
        return (a, b) in st or (not directed and (b, a) in st)

    anchors = sides or [(sum(pos[k][0] for k in rt) / len(rt), sum(pos[k][1] for k in rt) / len(rt))
                        for rt in rts]
    areas = [_area([pos[k] for k in rt]) for rt in rts]
    styles = {}
    for k in pos:
        on = [i for i, rt in enumerate(rts) if k in rt]
        styles[k] = "soft" if k in ends else colors[on[0]] if len(on) == 1 else (
            "default" if on or rest != "muted" else "muted")
    s.graph(pos, [], styles=styles)
    for e in edges:
        a, b = e[0], e[1]
        kw = {"directed": directed}
        if weights:
            kw["weight"] = str(e[2]).replace("-", "−")
        users = [i for i, rt in enumerate(rts) if uses(rt, a, b)]
        if not users:
            s.edge(a, b, style=rest, **kw)
        elif len(users) == 1:
            s.edge(a, b, style=colors[users[0]], **kw)
        else:
            (ax, ay), (bx, by) = pos[a], pos[b]
            d = math.hypot(bx - ax, by - ay)
            px, py = -(by - ay) / d, (bx - ax) / d  # core's positive-bend direction
            mx, my = (ax + bx) / 2, (ay + by) / 2
            side = {i: (1 if (anchors[i][0] - mx) * px + (anchors[i][1] - my) * py >= 0 else -1)
                    for i in users}
            i, j = sorted(users, key=lambda u: areas[u])[:2]
            if side[i] == side[j]:
                side[j] = -side[i]
            for u in (i, j):
                s.edge(a, b, style=colors[u], bend=side[u] * GAP, **kw)


def sp_panel(s: Scene, box: Box, rts: list[list[str]], *, weights: bool = True,
             edges: list | None = None) -> None:
    routes(s, place(SP, box, unit=UNIT), edges or SP_EDGES, rts, directed=True,
           ends=("A", "F"), weights=weights)


def matrix_note(s: Scene, box: Box, top: float, text: str) -> None:
    s.text(box.cx, top - 12, text, size=SIZE_NOTE, opacity=0.6)


# --- shortest paths ----------------------------------------------------------


@figure("bellman_ford_shortest_path", "paths", "Bellman-Ford shortest path",
        "A directed weighted graph with three routes from A to F, one of whose edges, D to F, "
        "has weight minus 2. Bellman-Ford handles the negative weight, so the returned path "
        "A, D, F is in the accent color. The card reads cost 3 + (−2) = 1.")
def bellman_ford_shortest_path(s: Scene) -> None:
    left, right = compute(s, "g", 'g.bellman_ford_shortest_path("A", "F")')
    neg = [e if e[:2] != ("D", "F") else ("D", "F", -2) for e in SP_EDGES]
    sp_panel(s, left, [MID], edges=neg)
    s.card_centered(right, [("A → D → F", "accent"), "cost  3 + (−2) = 1"],
                    title="shortest path, A to F")


@figure("ada_star_search", "paths", "ADA* search",
        "A directed weighted graph with three routes from A to F, costing 4, 6, and 5. "
        "With epsilon 1.0 ADA* behaves like A* and returns the optimal path A, B, C, F, "
        "drawn in the accent color. The card gives its cost, 1 + 2 + 1 = 4.")
def ada_star_search(s: Scene) -> None:
    left, right = compute(s, "g", 'g.ada_star_search("A", "F")')
    sp_panel(s, left, [TOP])
    s.card_centered(right, [("A → B → C → F", "accent"), "cost  1 + 2 + 1 = 4",
                            "ε = 1.0: optimal"], title="path")


@figure("k_shortest_paths", "paths", "k shortest paths",
        "A directed weighted graph with three routes from A to F. With k = 2, Yen's algorithm "
        "returns the cheapest path A, B, C, F (cost 4) in the accent color and the runner-up "
        "A, E, F (cost 5) in teal; the third route, through D (cost 6), stays faint.")
def k_shortest_paths(s: Scene) -> None:
    left, right = compute(s, "g", 'g.k_shortest_paths("A", "F", k=2)')
    sp_panel(s, left, [TOP, LOW])
    s.card_centered(right, [("A → B → C → F   4", "accent"), ("A → E → F       5", "g2")],
                    title="(path, cost), cheapest first")


@figure("get_all_simple_paths", "paths", "All simple paths",
        "A directed graph with three routes from A to F, none repeating a node. All three "
        "simple paths are returned and drawn in their own colors: A, B, C, F in the accent "
        "color, A, D, F in teal, and A, E, F in violet, matching the card.")
def get_all_simple_paths(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_all_simple_paths("A", "F")')
    sp_panel(s, left, [TOP, MID, LOW], weights=False)
    s.card_centered(right, [("A → B → C → F", "accent"), ("A → D → F", "g2"),
                            ("A → E → F", "g3")], title="simple paths")


@figure("compute_or_improve_path", "paths", "Compute or improve path",
        "An AdaStar planner from A to F, with a heuristic h, on a directed weighted graph. A first call with "
        "epsilon 4 quickly returns the suboptimal path A, D, F (cost 6, teal); calling again "
        "with epsilon 1.0 refines the same search to the optimal path A, B, C, F (cost 4, "
        "accent color).")
def compute_or_improve_path(s: Scene) -> None:
    left, right = compute(s, 'p = AdaStar(g, "A", "F", heuristic=h)', "p.compute_or_improve_path(1.0)")
    sp_panel(s, left, [TOP, MID])
    s.card_centered(right, [("ε = 4   A → D → F       6", "g2"),
                            ("ε = 1   A → B → C → F   4", "accent")], title="call, path, cost")


@figure("extract_path", "paths", "Extract path",
        "An AdaStar planner from A to F that has already searched. extract_path reads the "
        "stored path A, B, C, F, drawn in the accent color, without running the search again.")
def extract_path(s: Scene) -> None:
    left, right = compute(s, 'p = AdaStar(g, "A", "F")', "p.extract_path()")
    sp_panel(s, left, [TOP])
    s.card_centered(right, [("A → B → C → F", "accent")], title="stored path, no new search")


def _ap_panel(s: Scene, box: Box) -> None:
    # A to C goes through B (1 + 2 = 3) instead of the direct edge (4).
    pos = place(SQUARE, box, unit=UNIT)
    routes(s, pos, AP_EDGES, [list("ABC")], directed=True, ends=("A", "C"), rest="default",
           weights=True)
    for e in s.edges:
        if (e.a, e.b) == ("A", "C"):
            e.style = "muted"


@figure("all_pairs_shortest_path_length", "paths", "All-pairs shortest path lengths",
        "A four-node directed weighted cycle A, B, C, D with a direct shortcut A to C of "
        "weight 4. The distance matrix lists every source-target cost; the highlighted cell "
        "A to C is 3, because the route through B (1 + 2, in the accent color) beats the "
        "direct edge.")
def all_pairs_shortest_path_length(s: Scene) -> None:
    left, right = compute(s, "g", "g.all_pairs_shortest_path_length()")
    _ap_panel(s, left)
    rows = [["", "A", "B", "C", "D"], ["A", "0", "1", "3", "4"], ["B", "5", "0", "2", "3"],
            ["C", "3", "4", "0", "1"], ["D", "2", "3", "5", "0"]]
    x0, y0, _, _ = s.table_centered(right, rows, [28, 38, 38, 38, 38],
                                    highlight={(1, 3): "accent"})
    matrix_note(s, right, y0, "result[source][target]")


@figure("all_pairs_shortest_path", "paths", "All-pairs shortest paths",
        "A four-node directed weighted cycle A, B, C, D with a direct shortcut A to C of "
        "weight 4. The matrix gives the shortest path for every source-target pair, written "
        "as node letters; the highlighted cell A to C is A, B, C, the route drawn in the "
        "accent color.")
def all_pairs_shortest_path(s: Scene) -> None:
    left, right = compute(s, "g", "g.all_pairs_shortest_path()")
    _ap_panel(s, left)
    rows = [["", "A", "B", "C", "D"], ["A", "A", "AB", "ABC", "ABCD"],
            ["B", "BCDA", "B", "BC", "BCD"], ["C", "CDA", "CDAB", "C", "CD"],
            ["D", "DA", "DAB", "DABC", "D"]]
    x0, y0, _, _ = s.table_centered(right, rows, [28, 48, 48, 48, 48],
                                    highlight={(1, 3): "accent"})
    matrix_note(s, right, y0, "result[source][target]")


# --- trails and walks --------------------------------------------------------

TRAIL_SIDES = [(1.5, 1.6), (1.0, 0.0)]  # simple path bows below the spine, the loop trail above


def _trail_panel(s: Scene, box: Box) -> None:
    routes(s, place(TRAIL, box, unit=UNIT), TRAIL_EDGES, [list("ABCD"), list("ABEFBCD")],
           directed=True, ends=("A", "D"), sides=TRAIL_SIDES)


@figure("count_trails", "paths", "Count trails",
        "A directed path A, B, C, D with a loop B, E, F back to B. Two trails reach D from A: "
        "the plain path in the accent color, and a teal one that goes around the loop and "
        "passes B twice without reusing an edge. The card lists both and the count, 2.")
def count_trails(s: Scene) -> None:
    left, right = compute(s, "g", 'g.count_trails("A", "D")')
    _trail_panel(s, left)
    s.card_centered(right, [("A → B → C → D", "accent"), ("A → B → E → F → B → C → D", "g2"),
                            "count  2"], title="trails: no edge twice")


@figure("enumerate_trails", "paths", "Enumerate trails",
        "A directed path A, B, C, D with a loop B, E, F back to B. Both trails from A to D are "
        "returned: one goes around the loop (teal) and visits B twice, which a trail allows "
        "because no edge repeats; the other is the plain path (accent color).")
def enumerate_trails(s: Scene) -> None:
    left, right = compute(s, "g", 'g.enumerate_trails("A", "D")')
    _trail_panel(s, left)
    s.card_centered(right, [("[A, B, E, F, B, C, D]", "g2"), ("[A, B, C, D]", "accent")],
                    title="B repeats, no edge does")


@figure("count_walks", "paths", "Count walks",
        "A triangle A, B, C with a pendant node D on C. The matrix counts walks of length 2 "
        "between every pair. The highlighted entry A to A is 2: the walks A, B, A (accent "
        "color) and A, C, A (teal), which go out and back along the same edge, as walks may.")
def count_walks(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_walks(walk_length=2)")
    pos = place(SQUARE, left, unit=UNIT)
    s.graph(pos, [], styles={"A": "soft"})
    for a, b, c in (("A", "B", "accent"), ("A", "C", "g2")):
        s.edge(a, b, directed=True, style=c, bend=GAP + 1)
        s.edge(b, a, directed=True, style=c, bend=GAP + 1)
    s.edge("B", "C")
    s.edge("C", "D")
    rows = [["", "A", "B", "C", "D"], ["A", "2", "1", "1", "1"], ["B", "1", "2", "1", "1"],
            ["C", "1", "1", "3", "0"], ["D", "1", "1", "0", "1"]]
    box = Box(right.x0, right.y0, right.x1, right.y1 - 18)
    x0, y0, x1, y1 = s.table_centered(box, rows, [28, 38, 38, 38, 38], highlight={(1, 1): "accent"})
    s.text(right.cx - 6, y1 + 14, "A → B → A", size=SIZE_NOTE, anchor="end", color="accent")
    s.text(right.cx + 6, y1 + 14, "A → C → A", size=SIZE_NOTE, anchor="start", color="g2")


@figure("eulerian_circuit", "paths", "Eulerian circuit",
        "Two triangles sharing node C, so every node has even degree. The Eulerian circuit "
        "starts and ends at C and uses each of the six edges exactly once; edges are numbered "
        "in the order the circuit takes them, and the card lists C, E, D, C, A, B, C.")
def eulerian_circuit(s: Scene) -> None:
    left, right = compute(s, "g", "g.eulerian_circuit()")
    coords = {"A": (0, 0), "B": (0, 1.6), "C": (1.2, 0.8), "D": (2.4, 0), "E": (2.4, 1.6)}
    pos = place(coords, left, unit=UNIT)
    s.graph(pos, [(a, b, {"style": "accent", "weight": str(i)}) for i, (a, b) in enumerate(
        [("C", "E"), ("E", "D"), ("D", "C"), ("C", "A"), ("A", "B"), ("B", "C")], 1)])
    s.card_centered(right, [("C → E → D → C → A → B → C", "accent")],
                    title="every edge once, back to start")


SANTA_TRAIL = list("DECBDCABE")  # g.eulerian_path()


@figure("eulerian_path", "paths", "Eulerian path",
        "The house of Santa Claus: a square with both diagonals and a roof. Only the bottom "
        "corners D and E (outlined) have odd degree, so the Eulerian path runs from one to "
        "the other, using each of the eight edges once; edges are numbered in the order the "
        "path takes them, and the card lists D, E, C, B, D, C, A, B, E.")
def eulerian_path(s: Scene) -> None:
    left, right = compute(s, "g", "g.eulerian_path()")
    pos = place(HOUSE, left, unit=UNIT)
    steps = list(zip(SANTA_TRAIL, SANTA_TRAIL[1:]))
    # the two diagonals cross mid-square: label them near their lower ends instead
    t = {("D", "C"): 0.28, ("B", "E"): 0.72}
    s.graph(pos, [(a, b, {"style": "accent", "weight": str(i), "weight_t": t.get((a, b), 0.5)})
                  for i, (a, b) in enumerate(steps, 1)],
            styles={SANTA_TRAIL[0]: "soft", SANTA_TRAIL[-1]: "soft"})
    s.card_centered(right, [(" → ".join(SANTA_TRAIL[:5]), "accent"),
                            ("→ " + " → ".join(SANTA_TRAIL[5:]), "accent")],
                    title="every edge once, odd to odd")


# --- cycles ------------------------------------------------------------------


def _house(s: Scene, box: Box, cycles: list[list[str]], *, directed: bool = False,
           edges: list | None = None, ends: tuple[str, ...] = ()) -> None:
    routes(s, place(HOUSE, box, unit=UNIT), edges or HOUSE_EDGES, cycles, directed=directed,
           closed=True, rest="default", ends=ends)


@figure("get_cycles", "paths", "Elementary cycles (directed)",
        "A directed graph shaped like a house. Johnson's algorithm finds the two cycles that "
        "follow the edge directions: the triangle C, A, B in the accent color and the outer "
        "loop C, A, B, D, E in teal. The square B, C, E, D is not a directed cycle.")
def get_cycles(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_cycles()")
    _house(s, left, [list("CAB"), list("CABDE")], directed=True,
           edges=[("A", "B"), ("B", "C"), ("C", "A"), ("B", "D"), ("D", "E"), ("E", "C")])
    s.card_centered(right, [(bracket(list("CAB")), "accent"), (bracket(list("CABDE")), "g2")],
                    title="directed cycles")


@figure("simple_cycles", "paths", "Simple cycles",
        "An undirected house-shaped graph: a triangle on top of a square. It has three simple "
        "cycles, each drawn in its own color: the triangle C, A, B (accent color), the outer "
        "loop C, A, B, D, E (teal), and the square C, B, D, E (violet).")
def simple_cycles(s: Scene) -> None:
    left, right = compute(s, "g", "g.simple_cycles()")
    cyc = [list("CAB"), list("CABDE"), list("CBDE")]
    _house(s, left, cyc)
    s.card_centered(right, [(bracket(c), col) for c, col in zip(cyc, COLORS)],
                    title="simple cycles")


@figure("cycle_basis", "paths", "Cycle basis",
        "The house-shaped graph has 6 edges and 5 nodes, so a cycle basis has 6 − 5 + 1 = 2 "
        "cycles. Building the BFS tree from root A (node id 1) gives the fundamental cycles "
        "B, A, C (accent color) and D, B, A, C, E (teal), each closed by one non-tree edge.")
def cycle_basis(s: Scene) -> None:
    left, right = compute(s, "g", "g.cycle_basis(root=1)")
    _house(s, left, [list("BAC"), list("DBACE")], ends=("A",))
    s.badge("A", "id 1", where="e")
    s.card_centered(right, [(bracket(list("BAC")), "accent"), (bracket(list("DBACE")), "g2")],
                    title="E − V + 1 = 2 cycles")


@figure("minimum_cycle_basis", "paths", "Minimum cycle basis",
        "The house-shaped graph again. Its minimum cycle basis is the triangle A, C, B (length "
        "3, accent color) and the square B, C, E, D (length 4, teal), sharing edge B to C; the "
        "card totals 7, the smallest possible for two independent cycles.")
def minimum_cycle_basis(s: Scene) -> None:
    left, right = compute(s, "g", "g.minimum_cycle_basis()")
    _house(s, left, [list("ACB"), list("BCED")])
    s.card_centered(right, [("[A, C, B]      3", "accent"), ("[B, C, E, D]   4", "g2"),
                            "total          7"], title="cycle, length")


@figure("get_girth", "paths", "Girth",
        "The house-shaped graph has cycles of length 3, 4, and 5. The shortest is the "
        "triangle A, B, C, drawn in the accent color, so the girth is 3.")
def get_girth(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_girth()")
    _house(s, left, [list("ABC")])
    s.card_centered(right, ["shortest  [A, B, C]", ("girth     3", "accent")])


# The house with directed edges, as in get_cycles (one strongly connected component).
DHOUSE_EDGES = [("A", "B"), ("B", "C"), ("C", "A"), ("B", "D"), ("D", "E"), ("E", "C")]


@figure("directed_cycle_basis", "paths", "Directed cycle basis",
        "The house-shaped graph with directed edges, one strongly connected component. With 6 "
        "edges and 5 nodes the basis has 6 − 5 + 1 = 2 cycles, each following the arrows: the "
        "triangle A, B, C in the accent color and the outer loop A, B, D, E, C in teal, each "
        "listed from its lowest node.")
def directed_cycle_basis(s: Scene) -> None:
    left, right = compute(s, "g", "g.directed_cycle_basis()")
    _house(s, left, [list("ABC"), list("ABDEC")], directed=True, edges=DHOUSE_EDGES)
    s.card_centered(right, [(bracket(list("ABC")), "accent"), (bracket(list("ABDEC")), "g2")],
                    title="E − V + SCCs = 2 cycles")


# --- shortest path by hops -----------------------------------------------------


@figure("shortest_path", "paths", "Shortest path (fewest hops)",
        "The same directed weighted graph with three routes from A to F. shortest_path runs a "
        "breadth-first search, so it ignores the weights and returns the two-hop path A, D, F "
        "in the accent color, with filled badges giving each node's position 0, 1, 2.")
def shortest_path(s: Scene) -> None:
    left, right = compute(s, "g", 'g.shortest_path("A", "F")')
    sp_panel(s, left, [MID])
    for i, k in enumerate(MID):
        s.badge(k, str(i), filled=True)
    s.card_centered(right, [("A → D → F", "accent"), "2 hops, weights ignored"],
                    title="shortest_path_order")


# --- random walks ----------------------------------------------------------------

# random_walk("A", steps=6, seed=8) on DHOUSE_EDGES; random_edge_walk with the
# same seed draws from the same random stream, so it takes the same route.
WALK = list("ABCABDE")  # random_walk("A", steps=6, seed=8)


def _walk_panel(s: Scene, box: Box, *, step_labels: bool) -> None:
    steps = list(zip(WALK, WALK[1:]))
    order: dict[tuple[str, str], list[int]] = {}
    for i, e in enumerate(steps):
        order.setdefault(e, []).append(i)
    edges = []
    for a, b in DHOUSE_EDGES:
        kw = {"directed": True, "style": "accent" if (a, b) in order else "muted"}
        if step_labels and (a, b) in order:
            kw["weight"] = ",".join(map(str, order[(a, b)]))
        edges.append((a, b, kw))
    styles = {k: "accent" for k in WALK} if not step_labels else {}
    styles["A"] = "soft"
    s.graph(place(HOUSE, box, unit=UNIT), edges, styles=styles)


@figure("random_walk", "paths", "Random walk",
        "A directed house-shaped graph walked from A for 6 steps with a fixed seed. Filled "
        "badges give each node's positions in the walk, 0 for the start; A and B are visited "
        "twice. The card lists the 7 visited nodes, A, B, C, A, B, D, E; edge E to C is unused.")
def random_walk(s: Scene) -> None:
    left, right = compute(s, "g", 'g.random_walk("A", steps=6, seed=8)')
    _walk_panel(s, left, step_labels=False)
    pos: dict[str, list[int]] = {}
    for i, k in enumerate(WALK):
        pos.setdefault(k, []).append(i)
    for k, v in pos.items():
        s.badge(k, ",".join(map(str, v)), filled=True)
    s.card_centered(right, [(" → ".join(WALK), "accent")], title="steps + 1 = 7 nodes")


@figure("random_edge_walk", "paths", "Random edge walk",
        "The same directed house walked from A for 6 steps with the same seed, recording edges. "
        "Each traversed edge, in the accent color, is labeled with its step numbers from 0; "
        "A to B is taken twice, at steps 0 and 3. The card lists the 6 edges in order.")
def random_edge_walk(s: Scene) -> None:
    left, right = compute(s, "g", 'g.random_edge_walk("A", steps=6, seed=8)')
    _walk_panel(s, left, step_labels=True)
    hops = [f"{a}→{b}" for a, b in zip(WALK, WALK[1:])]
    s.card_centered(right, [("  ".join(hops[:3]), "accent"), ("  ".join(hops[3:]), "accent")],
                    title="(edge_id, source, target) × 6")


# --- shortest-path queries on the shared SP graph ------------------------------


@figure("shortest_path_length", "paths", "Shortest path length",
        "The directed weighted graph with three routes from A to F, costing 4, 6, and 5. On "
        "a weighted graph the distance is the cheapest total weight, so the route A, B, C, F "
        "is drawn in the accent color and the card reads 1 + 2 + 1 = 4.0; on the unweighted "
        "graph the same call counts hops and returns 2.")
def shortest_path_length(s: Scene) -> None:
    left, right = compute(s, "g", 'g.shortest_path_length("A", "F")')
    sp_panel(s, left, [TOP])
    s.card_centered(right, [("1 + 2 + 1 = 4.0", "accent"), "unweighted g:  2 hops"],
                    title="weighted distance, A to F")


@figure("all_shortest_paths", "paths", "All shortest paths",
        "The directed graph with three routes from A to F, unweighted. Two routes tie at "
        "two hops and both are returned: A, D, F in the accent color and A, E, F in teal. "
        "The three-hop route through B and C stays faint.")
def all_shortest_paths(s: Scene) -> None:
    left, right = compute(s, "g", 'g.all_shortest_paths("A", "F")')
    sp_panel(s, left, [MID, LOW], weights=False)
    s.card_centered(right, [("A → D → F", "accent"), ("A → E → F", "g2")],
                    title="every 2-hop path")


# bidirectional_dijkstra("A", "F") on the weighted SP graph. Replaying the Rust
# loop (expand whichever side has the smaller tentative distance, stop once
# top_f + top_b >= best) settles A, B forward and F, C backward, in the order
# A, F, B, C; best = 4 is found when B relaxes B -> C. D and E get tentative
# distances but are never settled.
BIDI_FWD, BIDI_BWD = list("AB"), list("FC")


@figure("bidirectional_dijkstra", "paths", "Bidirectional Dijkstra",
        "The directed weighted graph with three routes from A to F. One search grows "
        "forward from A and settles A and B (teal region), the other grows backward from F "
        "and settles F and C (violet region); they meet across the edge B to C, and D and E "
        "are never settled. The returned path A, B, C, F is in the accent color, with "
        "length 1 + 2 + 1 = 4.0.")
def bidirectional_dijkstra(s: Scene) -> None:
    left, right = compute(s, "g", 'g.bidirectional_dijkstra("A", "F")')
    sp_panel(s, left, [TOP])
    s.hull(BIDI_FWD, "g2")
    s.hull(BIDI_BWD, "g3")
    s.card_centered(right, [("forward   A, B", "g2"), ("backward  F, C", "g3"),
                            ("A → B → C → F", "accent"), ("length  1 + 2 + 1 = 4.0", "accent")],
                    title="settled from each end, path")


# single_source_shortest_path("A") on the weighted SP graph (Dijkstra).
SSSP = {"A": list("A"), "B": list("AB"), "C": list("ABC"), "D": list("AD"), "E": list("AE"),
        "F": list("ABCF")}
SSSP_TREE = {("A", "B"), ("B", "C"), ("C", "F"), ("A", "D"), ("A", "E")}
SSSP_DIST = {"A": "0.0", "B": "1.0", "C": "3.0", "D": "3.0", "E": "2.0", "F": "4.0"}


def _tree_panel(s: Scene, box: Box) -> None:
    """SP with the shortest-path tree from A in accent and A soft."""
    pos = place(SP, box, unit=UNIT)
    s.graph(pos, [(a, b, {"style": "accent" if (a, b) in SSSP_TREE else "muted",
                          "weight": str(w)}) for a, b, w in SP_EDGES],
            directed=True, styles={k: "soft" if k == "A" else "accent" for k in SP})


@figure("single_source_shortest_path", "paths", "Single-source shortest paths",
        "The directed weighted graph with the source A outlined. The cheapest path from A to "
        "every other node together forms a tree, drawn in the accent color: B, D, and E hang "
        "off A directly, C and F are reached through B. The edges D to F and E to F are not "
        "used. The card lists each node's path.")
def single_source_shortest_path(s: Scene) -> None:
    left, right = compute(s, "g", 'g.single_source_shortest_path("A")')
    _tree_panel(s, left)
    s.card_centered(right, [f"{k}  {' → '.join(p)}" for k, p in SSSP.items()],
                    title="target   path")


@figure("single_source_shortest_path_length", "paths", "Single-source shortest path lengths",
        "The directed weighted graph with the source A outlined and each node's distance "
        "from A in a badge: A 0.0, B 1.0, E 2.0, C and D 3.0, and F 4.0, reached along the "
        "accent edges. The table lists the same distances from nearest to farthest.")
def single_source_shortest_path_length(s: Scene) -> None:
    left, right = compute(s, "g", 'g.single_source_shortest_path_length("A")')
    _tree_panel(s, left)
    for k, v in SSSP_DIST.items():
        s.badge(k, v)
    order = sorted(SSSP_DIST, key=lambda k: (float(SSSP_DIST[k]), k))
    groups: dict[str, list[str]] = {}
    for k in order:
        groups.setdefault(SSSP_DIST[k], []).append(k)
    rows = [["node", "distance"]] + [[", ".join(ks), v] for v, ks in groups.items()]
    s.table_centered(right, rows, [64, 72])


# --- multi-source shortest paths: catchments on a ladder ------------------------

# A 2 x 4 ladder with sources A and G. They are an odd number of hops apart on a
# bipartite graph, so no node is equally close to both (no ties to break).
LADDER = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0), "E": (0, 1.1), "F": (1, 1.1),
          "G": (2, 1.1), "H": (3, 1.1)}
LADDER_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F"), ("F", "G"), ("G", "H"),
                ("A", "E"), ("B", "F"), ("C", "G"), ("D", "H")]
SOURCES = ("A", "G")
# multi_source_shortest_path(["A", "G"]): path per node, starting at its nearest source
MSP = {"A": list("A"), "B": list("AB"), "E": list("AE"), "G": list("G"), "C": list("GC"),
       "F": list("GF"), "H": list("GH"), "D": list("GCD")}
MSP_DIST = {k: len(p) - 1 for k, p in MSP.items()}
SRC_CAPTION = 'g,  src = ["A", "G"]'


@figure("multi_source_shortest_path", "paths", "Multi-source shortest paths",
        "A two-by-four ladder with sources A and G. Each node is reached from whichever "
        "source is nearer, which splits the graph into two catchment areas: A serves B and E "
        "(accent color), G serves C, D, F, and H (teal). The shortest-path tree edges take "
        "their catchment's color, and each source has a halo. The card lists each catchment "
        "and the path to D, [G, C, D].")
def multi_source_shortest_path(s: Scene) -> None:
    left, right = compute(s, SRC_CAPTION, "g.multi_source_shortest_path(src)")
    color = {k: "accent" if p[0] == "A" else "g2" for k, p in MSP.items()}
    tree = {(p[-2], p[-1]) for p in MSP.values() if len(p) > 1}
    edges = []
    for a, b in LADDER_EDGES:
        on = (a, b) in tree or (b, a) in tree
        edges.append((a, b, {"style": color[a] if on else "muted"}))
    s.graph(place(LADDER, left, unit=UNIT), edges, styles=color)
    # catchment hulls would cut across the ladder diagonally; node and tree-edge
    # colors carry the catchments, and a halo marks each source
    for src, c in zip(SOURCES, ("accent", "g2")):
        s.hull([src], c, pad=7, opacity=0.3)
    s.card_centered(right, [("A:  A, B, E", "accent"), ("G:  C, D, F, G, H", "g2"),
                            "path to D  [G, C, D]"], title="nearest source:  nodes")


@figure("multi_source_shortest_path_length", "paths", "Multi-source shortest path lengths",
        "A two-by-four ladder with sources A and G outlined, each node's distance to the "
        "nearer source in a badge. The sources are at 0; B, C, E, F, and H are one hop from "
        "a source; D, in the far corner, is the only node at distance 2. The table groups "
        "the nodes by distance.")
def multi_source_shortest_path_length(s: Scene) -> None:
    left, right = compute(s, SRC_CAPTION, "g.multi_source_shortest_path_length(src)")
    s.graph(place(LADDER, left, unit=UNIT), LADDER_EDGES,
            styles={k: "soft" for k in SOURCES})
    for k in LADDER:
        s.badge(k, str(MSP_DIST[k]))
    groups: dict[int, list[str]] = {}
    for k in sorted(LADDER):
        groups.setdefault(MSP_DIST[k], []).append(k)
    rows = [["distance", "nodes"]] + [[str(d), ", ".join(groups[d])] for d in sorted(groups)]
    s.table_centered(right, rows, [72, 112])


# --- longest paths in a DAG (the shared DAG from traversal.py) ------------------

# Task durations on the DAG's edges; the critical path is B → D → F → G.
DAG_W = {("A", "C"): 2, ("A", "D"): 1, ("B", "D"): 3, ("C", "E"): 2, ("D", "E"): 1,
         ("D", "F"): 4, ("E", "G"): 1, ("F", "G"): 2}
CRITICAL = list("BDFG")


def _dag_panel(s: Scene, box: Box) -> None:
    from figures.traversal import DAG, DAG_EDGES

    routes(s, place(DAG, box, unit=UNIT), [(a, b, DAG_W[(a, b)]) for a, b in DAG_EDGES],
           [CRITICAL], directed=True, weights=True)


@figure("dag_longest_path", "paths", "DAG longest path",
        "A seven-node DAG whose edges carry task durations. The heaviest chain of tasks, the "
        "critical path B, D, F, G, is drawn in the accent color: its durations 3 + 4 + 2 add "
        "up to 9.0, more than any other route from a source to G.")
def dag_longest_path(s: Scene) -> None:
    left, right = compute(s, "g", "g.dag_longest_path()")
    _dag_panel(s, left)
    s.card_centered(right, [("B → D → F → G", "accent"), "weight  3 + 4 + 2 = 9.0"],
                    title="critical path")


@figure("dag_longest_path_length", "paths", "DAG longest path length",
        "The seven-node DAG with task durations on its edges. The longest path, B, D, F, G "
        "in the accent color, has total weight 3 + 4 + 2 = 9.0, which the card gives as the "
        "result; ignoring the weights, the longest path has 3 edges.")
def dag_longest_path_length(s: Scene) -> None:
    left, right = compute(s, "g", "g.dag_longest_path_length()")
    _dag_panel(s, left)
    s.card_centered(right, ["B → D → F → G", ("3 + 4 + 2 = 9.0", "accent"),
                            "unweighted g:  3"], title="longest path length")


# --- disjoint paths (Menger): the bowtie plus a detour through F ----------------

# Two triangles sharing C (the edge_disjoint_paths docstring bowtie), plus a
# third route A - F - E that avoids C, so both variants find more than one path.
BOWTIE = {"A": (0, 0.8), "B": (1, 0), "C": (2, 0.8), "D": (3, 0), "E": (4, 0.8), "F": (2, 1.7)}
BOWTIE_EDGES = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "C"),
                ("A", "F"), ("F", "E")]
EDGE_DISJ = [list("ACE"), list("AFE"), list("ABCDE")]  # edge_disjoint_paths("A", "E")
NODE_DISJ = [list("ACE"), list("AFE")]  # node_disjoint_paths("A", "E")


def arrows(rt: list[str]) -> str:
    return " → ".join(rt)


@figure("edge_disjoint_paths", "paths", "Edge-disjoint paths",
        "Two triangles sharing node C, plus a detour from A to E through F. Three paths from "
        "A to E use no edge twice: A, C, E in the accent color, A, F, E in teal, and the long "
        "way A, B, C, D, E in violet. Two of them pass through C, which edge-disjoint paths "
        "may share.")
def edge_disjoint_paths(s: Scene) -> None:
    left, right = compute(s, "g", 'g.edge_disjoint_paths("A", "E")')
    routes(s, place(BOWTIE, left, unit=UNIT), BOWTIE_EDGES, EDGE_DISJ, directed=False,
           ends=("A", "E"))
    s.card_centered(right, [(arrows(rt), c) for rt, c in zip(EDGE_DISJ, COLORS)],
                    title="3 paths, no shared edge")


@figure("node_disjoint_paths", "paths", "Node-disjoint paths",
        "The same graph: two triangles sharing node C, plus a detour from A to E through F. "
        "Only two paths from A to E avoid each other's inner nodes, A, C, E in the accent "
        "color and A, F, E in teal; the third edge-disjoint route, A, B, C, D, E, would "
        "reuse C, so B and D stay faint.")
def node_disjoint_paths(s: Scene) -> None:
    left, right = compute(s, "g", 'g.node_disjoint_paths("A", "E")')
    routes(s, place(BOWTIE, left, unit=UNIT), BOWTIE_EDGES, NODE_DISJ, directed=False,
           ends=("A", "E"))
    s.card_centered(right, [(arrows(rt), c) for rt, c in zip(NODE_DISJ, COLORS)]
                    + ["(route via B, D reuses C)"], title="2 paths, no shared node")


# --- later additions: round trips ---------------------------------------------
#
# traveling_salesman rebuilds its docstring example in edgecase and asserts the
# returned route, so a library change breaks the build instead of the picture.
# SITES: five sites on a pentagon with two chords (A–C, B–E) the tour skips.
SITES = ring(list("ABCDE"), radius=1.0)
SITES_EDGES = [("A", "B", 2), ("B", "C", 2), ("C", "D", 3), ("D", "E", 2), ("E", "A", 3),
               ("A", "C", 4), ("B", "E", 3)]


@figure("traveling_salesman", "paths", "Traveling salesman",
        "Five sites A to E on a pentagon, joined by roads weighted with travel times: A–B 2, "
        "B–C 2, C–D 3, D–E 2, E–A 3, and the two chords A–C 4 and B–E 3. The round trip "
        "starts at A (outlined) and follows the accent arrows A, B, C, D, E, back to A, "
        "visiting every site once; the chords are faded. The card shows the route and its "
        "cost 2 + 2 + 3 + 2 + 3 = 12.")
def traveling_salesman(s: Scene) -> None:
    import edgecase as ec

    left, right = compute(s, "g", "g.traveling_salesman()")
    g = ec.Graph(weighted=True).add_nodes(list(SITES))
    g.add_edges([(a, b, float(w)) for a, b, w in SITES_EDGES])
    tour = [g.get_label_from_id(n) for n in g.traveling_salesman()]
    assert tour == ["A", "B", "C", "D", "E", "A"]
    w = {frozenset((a, b)): c for a, b, c in SITES_EDGES}
    legs = list(zip(tour, tour[1:]))
    cost = [w[frozenset(leg)] for leg in legs]
    assert sum(cost) == 12
    on = {frozenset(leg) for leg in legs}
    # chord weights sit off the crossing point so the two labels stay apart
    t = {frozenset("AC"): 0.72, frozenset("BE"): 0.75}
    specs = [(a, b, {"style": "accent", "weight": c, "directed": True}) for (a, b), c in
             zip(legs, cost)]
    specs += [(a, b, {"style": "muted", "weight": c, "weight_t": t[frozenset((a, b))]})
              for a, b, c in SITES_EDGES if frozenset((a, b)) not in on]
    s.graph(place(SITES, left, unit=UNIT), specs,
            styles={k: "soft" if k == tour[0] else "accent" for k in SITES})
    s.card_centered(right, [(arrows(tour), "accent"),
                            "cost  " + " + ".join(str(c) for c in cost) + f" = {sum(cost)}"],
                    title="visit every site, return to A")
