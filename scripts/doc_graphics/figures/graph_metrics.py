"""Whole-graph metrics: one number that summarizes the entire graph.

Template: `compute`. Left: the graph, with whatever the metric counts drawn
in `accent` (and what it *could* count but doesn't as `ghost`/`removed`).
Per-node ingredients (degree, eccentricity, local clustering) ride on the
nodes as outline badges. Right: a result card that shows the arithmetic as
aligned `name  value` lines, ending in the value in accent. Distributions
(degree histogram, adjacency spectrum) swap the card for a small bar chart.

Siblings share a few base graphs so they can be compared directly:

- `DIST`  six nodes, a 4-cycle with a tail on each side (distance metrics)
- `TRI`   two triangles sharing an edge plus a pendant (triangles, cliques,
          clustering, spectrum, degree statistics)
- `DIR`   a five-node digraph with two reciprocal pairs (directed degree
          aggregates, reciprocity)
- `PAIR`  two clusters joined by one edge (assortativity)

Every number on a card is what edgecase returns for the depicted graph.
"""

from core import SIZE_NOTE, Box, Scene, compute, figure, place, ring

UNIT = 84  # one grid step for every graph in the family

# --- base graphs -------------------------------------------------------------

# Distance family. Eccentricities: A 4, B 3, C 2, D 2, E 3, F 4.
DIST = {"A": (0, 0.5), "B": (1, 0.5), "C": (2, 0), "D": (2, 1), "E": (3, 0.5), "F": (4, 0.5)}
DIST_EDGES = [("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"), ("E", "F")]
DIST_ECC = {"A": 4, "B": 3, "C": 2, "D": 2, "E": 3, "F": 4}

# Triangle family: triangles ABC and BCD share the edge B–C; E hangs off D.
TRI = {"A": (0, 0.5), "B": (1, 0), "C": (1, 1), "D": (2, 0.5), "E": (3, 0.5)}
TRI_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
TRI_DEG = {"A": 2, "B": 3, "C": 3, "D": 3, "E": 1}
TRI_ABC = {("A", "B"), ("A", "C"), ("B", "C")}
TRI_BCD = {("B", "C"), ("B", "D"), ("C", "D")}

# Directed family: A⇄B and C⇄D are reciprocal pairs.
DIR = {"A": (0, 0), "B": (0, 1.3), "C": (1.2, 0.65), "D": (2.4, 0.65), "E": (3.6, 0.65)}
DIR_ARCS = [("A", "B"), ("B", "A"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "C"), ("D", "E")]
DIR_MUTUAL = {("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")}
BEND = 9  # bow for each arc of a reciprocal pair

# Assortativity family: two sides joined through C–D.
PAIR = {"A": (0, 0), "B": (0, 1.3), "C": (1, 0.65), "D": (2.2, 0.65), "E": (3.2, 0), "F": (3.2, 1.3)}
PAIR_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("D", "F"), ("E", "F")]


def _undirected(s: Scene, coords, edges, box: Box, *, accent=frozenset(), styles=None,
                weights=None, heat=None) -> None:
    pos = place(coords, box, unit=UNIT)
    weights = weights or {}
    out = []
    for a, b in edges:
        kw = {}
        if (a, b) in accent:
            kw["style"] = "accent"
        if (a, b) in weights:
            kw["weight"] = weights[(a, b)]
        out.append((a, b, kw))
    s.graph(pos, out, styles=styles, heat=heat)


def _directed(s: Scene, box: Box, *, accent=frozenset(), styles=None) -> None:
    pos = place(DIR, box, unit=UNIT)
    arcs = []
    for a, b in DIR_ARCS:
        kw = {"directed": True}
        if (b, a) in DIR_ARCS:
            kw["bend"] = BEND
        if (a, b) in accent:
            kw["style"] = "accent"
        arcs.append((a, b, kw))
    s.graph(pos, arcs, styles=styles)


def _badges(s: Scene, values: dict, where: dict | None = None) -> None:
    where = where or {}
    for k, v in values.items():
        s.badge(k, str(v), where=where.get(k, "auto"))


def _bars(s: Scene, box: Box, values: list[float], labels: list[str], ticks: list[str] | None,
          *, accent: set[int], bar_w: float = 30, gap: float = 14, scale: float = 1.0,
          base_y: float | None = None) -> None:
    """Minimal bar chart: bars rise (or drop) from a shared baseline.

    `values` in data units, `scale` px per unit. `labels` sit beyond each
    bar's end; `ticks` (optional) sit under the baseline.
    """
    n = len(values)
    total = n * bar_w + (n - 1) * gap
    x0 = box.cx - total / 2
    y0 = box.cy if base_y is None else base_y
    s.line(x0 - 10, y0, x0 + total + 10, y0, opacity=0.5, width=1, role="axis")
    for i, v in enumerate(values):
        x = x0 + i * (bar_w + gap)
        h = abs(v) * scale
        top = y0 - h if v >= 0 else y0
        color = "accent" if i in accent else "ink"
        s.rect(x, top, bar_w, h, rx=2, stroke=None, solid=True, fill_color=color,
               fill_opacity=1 if i in accent else 0.3, role="bar")
        ly = top - 9 if v >= 0 else top + h + 9
        s.text(x + bar_w / 2, ly, labels[i], size=SIZE_NOTE, color=color,
               weight=600 if i in accent else 400, opacity=1 if i in accent else 0.75)
        if ticks:
            s.text(x + bar_w / 2, y0 + 11, ticks[i], size=SIZE_NOTE, opacity=0.6)


# --- density & degree --------------------------------------------------------


@figure("get_density", "graph-metrics", "Graph density",
        "A five-node graph with four edges drawn solid and the six missing node pairs "
        "drawn as faint dashed lines. The result card reads 4 of 10 possible edges, "
        "density 0.4.")
def get_density(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_density()")
    ids = list("ABCDE")
    pos = place(ring(ids), left, unit=UNIT)
    present = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "E")]
    missing = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
               if (a, b) not in present]
    s.graph(pos, [(a, b, {"style": "accent"}) for a, b in present]
            + [(a, b, {"style": "removed"}) for a, b in missing])
    s.card_centered(right, ["edges     4", "possible  10", ("density   0.4", "accent")])


@figure("get_average_degree", "graph-metrics", "Average degree",
        "A five-node undirected graph with each node's degree in a badge: A 2, B 3, C 3, "
        "D 3, E 1. The card adds the degrees to 12, divides by 5 nodes, and gives the "
        "average degree 2.4.")
def get_average_degree(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_average_degree()")
    _undirected(s, TRI, TRI_EDGES, left)
    _badges(s, TRI_DEG)
    s.card_centered(right, ["Σ degree  12", "nodes     5", ("average   2.4", "accent")])


@figure("get_agg_degree_in", "graph-metrics", "Aggregate in-degree",
        "A five-node directed graph with each node's in-degree in a badge. Node C receives "
        "three arcs, drawn in the accent color, more than any other node. The card lists "
        "the in-degrees 1, 1, 3, 1, 1 and their max, 3.0.")
def get_agg_degree_in(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_agg_degree_in("max")')
    _directed(s, left, accent={("A", "C"), ("B", "C"), ("D", "C")}, styles={"C": "accent"})
    _badges(s, {"A": 1, "B": 1, "C": 3, "D": 1, "E": 1})
    s.card_centered(right, ["in-degree  1 1 3 1 1", ("max        3.0", "accent")])


@figure("get_agg_degree_out", "graph-metrics", "Aggregate out-degree",
        "The same five-node directed graph with each node's out-degree in a badge: A 2, "
        "B 2, C 1, D 2, E 0. The card sums them to 7 over 5 nodes and gives the mean "
        "out-degree 1.4.")
def get_agg_degree_out(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_agg_degree_out("mean")')
    _directed(s, left)
    _badges(s, {"A": 2, "B": 2, "C": 1, "D": 2, "E": 0})
    s.card_centered(right, ["out-degree  2 2 1 2 0", "sum / n     7 / 5",
                            ("mean        1.4", "accent")])


@figure("get_agg_degree_total", "graph-metrics", "Aggregate total degree",
        "The same five-node directed graph with each node's total degree (in plus out) in "
        "a badge: A 3, B 3, C 4, D 3, E 1. The card sorts them as 1, 3, 3, 3, 4 and gives "
        "the median, 3.0.")
def get_agg_degree_total(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_agg_degree_total("median")')
    _directed(s, left)
    _badges(s, {"A": 3, "B": 3, "C": 4, "D": 3, "E": 1})
    s.card_centered(right, ["sorted  1 3 3 3 4", ("median  3.0", "accent")])


@figure("get_degree_distribution", "graph-metrics", "Degree distribution",
        "A five-node graph with each node's degree in a badge, and a bar chart of how many "
        "nodes have each degree: one node of degree 1, one of degree 2, and three of "
        "degree 3.")
def get_degree_distribution(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_degree_distribution()")
    _undirected(s, TRI, TRI_EDGES, left)
    _badges(s, TRI_DEG)
    _bars(s, right, [1, 1, 3], ["1", "1", "3"], ["deg 1", "deg 2", "deg 3"],
          accent={0, 1, 2}, bar_w=36, gap=22, scale=34, base_y=150)


# --- triangles & clustering --------------------------------------------------


@figure("count_triangles", "graph-metrics", "Count triangles",
        "Two triangles, A B C (accent) and B C D (teal), share the edge B to C; E hangs off "
        "D. The table counts the triangles each node belongs to: B and C are in 2, A and D "
        "in 1, E in none.")
def count_triangles(s: Scene) -> None:
    left, right = compute(s, "g", "g.count_triangles()")
    pos = place(TRI, left, unit=UNIT)
    edges = []
    for a, b in TRI_EDGES:
        if (a, b) == ("B", "C"):
            edges.append((a, b))
        elif (a, b) in TRI_ABC:
            edges.append((a, b, {"style": "accent"}))
        elif (a, b) in TRI_BCD:
            edges.append((a, b, {"style": "g2"}))
        else:
            edges.append((a, b))
    s.graph(pos, edges)
    rows = [["node", "triangles"], ["B", "2"], ["C", "2"], ["A", "1"], ["D", "1"], ["E", "0"]]
    s.table_centered(right, rows, [56, 84])


@figure("get_transitivity", "graph-metrics", "Transitivity",
        "A five-node graph with two triangles in the accent color. Badges count the "
        "connected triples centered on each node (1, 3, 3, 3, 0), ten in all. The card "
        "computes 3 times 2 triangles over 10 triples, transitivity 0.6.")
def get_transitivity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_transitivity()")
    _undirected(s, TRI, TRI_EDGES, left, accent=TRI_ABC | TRI_BCD)
    _badges(s, {"A": 1, "B": 3, "C": 3, "D": 3, "E": 0})
    s.card_centered(right, ["triangles  2", "triples    10", ("3·2 / 10   0.6", "accent")])


@figure("get_avg_clustering_coefficient", "graph-metrics", "Average clustering coefficient",
        "A five-node graph with two triangles in the accent color. Badges give each node's "
        "local clustering coefficient: A 1, B 0.67, C 0.67, D 0.33, E 0. The card sums "
        "them to 2.667 over 5 nodes, an average of 0.533.")
def get_avg_clustering_coefficient(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_avg_clustering_coefficient()")
    _undirected(s, TRI, TRI_EDGES, left, accent=TRI_ABC | TRI_BCD)
    _badges(s, {"A": "1", "B": ".67", "C": ".67", "D": ".33", "E": "0"})
    s.card_centered(right, ["Σ C(v)   2.667", "nodes    5", ("average  0.533", "accent")])


@figure("clique_number", "graph-metrics", "Clique number",
        "A five-node graph whose largest cliques have three nodes. One of them, A B C, is "
        "drawn in the accent color; B C D ties it, and no four nodes are all linked. The "
        "card gives the clique number 3.")
def clique_number(s: Scene) -> None:
    left, right = compute(s, "g", "g.clique_number()")
    _undirected(s, TRI, TRI_EDGES, left, accent=TRI_ABC,
                styles={"A": "accent", "B": "accent", "C": "accent"})
    s.card_centered(right, ["largest  A, B, C", "tie      B, C, D", ("size     3", "accent")])


@figure("chromatic_number_upper_bound", "graph-metrics", "Chromatic number upper bound",
        "A five-node graph colored greedily so that no edge joins two nodes of the same "
        "color: C and E get color 0 (accent), A and D color 1 (teal), B color 2 (violet). "
        "The card counts three colors used, the upper bound 3.")
def chromatic_number_upper_bound(s: Scene) -> None:
    left, right = compute(s, "g", "g.chromatic_number_upper_bound()")
    colors = {"C": "accent", "E": "accent", "A": "g2", "D": "g2", "B": "g3"}
    _undirected(s, TRI, TRI_EDGES, left, styles=colors)
    s.card_centered(right, [("0 → C, E", "accent"), ("1 → A, D", "g2"), ("2 → B", "g3"),
                            "colors used  3"], title="greedy coloring")


@figure("rich_club_coefficient", "graph-metrics", "Rich-club coefficient",
        "A five-node graph whose three degree-3 nodes B, C, D are grouped in the accent "
        "color and fully linked to each other. The table gives, for each degree k, the "
        "nodes above k, the edges among them, and phi(k): 0.6, 0.833, and 1.0 at k = 2.")
def rich_club_coefficient(s: Scene) -> None:
    left, right = compute(s, "g", "g.rich_club_coefficient()")
    club = ["B", "C", "D"]
    _undirected(s, TRI, TRI_EDGES, left, accent=TRI_BCD, styles={k: "accent" for k in club})
    s.hull(club, "accent")
    rows = [["k", "nodes", "edges", "φ(k)"], ["0", "5", "6", "0.6"], ["1", "4", "5", "0.833"],
            ["2", "3", "3", "1.0"]]
    s.table_centered(right, rows, [36, 54, 54, 64],
                     highlight={(3, j): "accent" for j in range(4)})


# --- spectrum ----------------------------------------------------------------

TRI_EIG = [2.641, 0.724, -0.589, -1.0, -1.776]


@figure("get_spectrum", "graph-metrics", "Adjacency spectrum",
        "A five-node graph and a bar chart of its five adjacency eigenvalues, sorted from "
        "largest to smallest: 2.641, 0.724, -0.589, -1.000, and -1.776. The largest "
        "eigenvalue is in the accent color.")
def get_spectrum(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_spectrum()")
    _undirected(s, TRI, TRI_EDGES, left)
    _bars(s, right, TRI_EIG, [f"{v:.3f}" for v in TRI_EIG], None, accent={0},
          bar_w=30, gap=14, scale=26, base_y=104)


@figure("get_estrada_index", "graph-metrics", "Estrada index",
        "A five-node graph and a card that exponentiates each of its five adjacency "
        "eigenvalues, from e to the 2.641 (14.03) down to e to the -1.776 (0.17), and adds "
        "them to the Estrada index 17.18.")
def get_estrada_index(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_estrada_index()")
    _undirected(s, TRI, TRI_EDGES, left)
    terms = ["14.03", "2.06", "0.55", "0.37", "0.17"]
    lines = [f"e^{v:<8.3f}{t}" for v, t in zip(TRI_EIG, terms)]
    s.card_centered(right, lines + [("Σ         17.18", "accent")])


# --- distances ---------------------------------------------------------------


@figure("get_diameter", "graph-metrics", "Diameter",
        "A six-node graph: a four-node cycle with a tail on each side. The longest "
        "shortest path runs from A to F in four hops (shown in the accent color along one "
        "of the two equal routes), so the diameter is 4.")
def get_diameter(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_diameter()")
    path = {("A", "B"), ("B", "C"), ("C", "E"), ("E", "F")}
    _undirected(s, DIST, DIST_EDGES, left, accent=path,
                styles={"A": "accent", "F": "accent"})
    s.card_centered(right, ["farthest  A, F", "hops      4", ("diameter  4", "accent")])


@figure("get_max_eccentricity", "graph-metrics", "Maximum eccentricity",
        "A six-node graph with each node's eccentricity in a badge: A 4, B 3, C 2, D 2, "
        "E 3, F 4. The end nodes A and F have the largest value and are drawn in the "
        "accent color; the card gives the maximum eccentricity 4.")
def get_max_eccentricity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_max_eccentricity()")
    _undirected(s, DIST, DIST_EDGES, left, styles={"A": "accent", "F": "accent"})
    _badges(s, DIST_ECC)
    s.card_centered(right, ["ecc      2, 3, 4", ("max ecc  4", "accent")])


@figure("get_min_eccentricity", "graph-metrics", "Minimum eccentricity",
        "A six-node graph with each node's eccentricity in a badge: A 4, B 3, C 2, D 2, "
        "E 3, F 4. The middle nodes C and D have the smallest value and are drawn in the "
        "accent color; the card gives the minimum eccentricity 2.")
def get_min_eccentricity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_min_eccentricity()")
    _undirected(s, DIST, DIST_EDGES, left, styles={"C": "accent", "D": "accent"})
    _badges(s, DIST_ECC)
    s.card_centered(right, ["ecc      2, 3, 4", ("min ecc  2", "accent")])


@figure("get_radius", "graph-metrics", "Radius",
        "A six-node graph with each node's eccentricity in a badge. The center nodes C and "
        "D, drawn in the accent color, reach every other node within 2 hops, so the "
        "radius is 2.")
def get_radius(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_radius()")
    _undirected(s, DIST, DIST_EDGES, left, styles={"C": "accent", "D": "accent"})
    _badges(s, DIST_ECC)
    s.card_centered(right, ["center  C, D", "ecc     2", ("radius  2", "accent")])


@figure("get_periphery", "graph-metrics", "Periphery",
        "A six-node graph with each node's eccentricity in a badge. The end nodes A and F "
        "have the maximum eccentricity 4 and are drawn in the accent color. The card "
        "returns their node ids, 1 and 6.")
def get_periphery(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_periphery()")
    _undirected(s, DIST, DIST_EDGES, left, styles={"A": "accent", "F": "accent"})
    _badges(s, DIST_ECC)
    s.card_centered(right, ["max ecc  4", "nodes    A, F", ("ids      [1, 6]", "accent")])


@figure("get_wiener_index", "graph-metrics", "Wiener index",
        "A six-node graph and a card that groups its 15 node pairs by shortest-path "
        "distance: 6 pairs at distance 1, 6 at 2, 2 at 3, and 1 at 4. Weighting each "
        "group by its distance sums to the Wiener index 28.0.")
def get_wiener_index(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_wiener_index()")
    _undirected(s, DIST, DIST_EDGES, left)
    s.card_centered(right, ["d=1  6 × 1    6", "d=2  6 × 2    12", "d=3  2 × 3    6",
                            "d=4  1 × 4    4", ("Σ d           28.0", "accent")])


@figure("get_mean_distance", "graph-metrics", "Mean distance",
        "A six-node graph and a card that adds the shortest-path distances of all 15 node "
        "pairs to 28 and divides by the 15 pairs, giving a mean distance of 1.867.")
def get_mean_distance(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_mean_distance()")
    _undirected(s, DIST, DIST_EDGES, left)
    s.card_centered(right, ["Σ d    28", "pairs  15", ("mean   1.867", "accent")])


@figure("global_efficiency", "graph-metrics", "Global efficiency",
        "A six-node graph and a card that groups its 15 node pairs by distance and adds "
        "one over the distance for each: 6 pairs at 1, 6 at one half, 2 at one third, 1 at "
        "one quarter, 9.917 in all. Divided by 15 pairs this gives efficiency 0.661.")
def global_efficiency(s: Scene) -> None:
    left, right = compute(s, "g", "g.global_efficiency()")
    _undirected(s, DIST, DIST_EDGES, left)
    s.card_centered(right, ["d=1  6 × 1      6", "d=2  6 × 1/2    3", "d=3  2 × 1/3    0.667",
                            "d=4  1 × 1/4    0.25", ("9.917 / 15      0.661", "accent")])


@figure("get_kirchhoff_index", "graph-metrics", "Kirchhoff index",
        "A six-node graph labeled with the resistance distance across each edge: 1 on the "
        "two tail edges, but 0.75 on the four cycle edges (accent), where a second route "
        "runs in parallel. Summing resistance over all 15 pairs gives the Kirchhoff index "
        "21.0.")
def get_kirchhoff_index(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_kirchhoff_index()")
    cycle = {("B", "C"), ("B", "D"), ("C", "E"), ("D", "E")}
    weights = {e: ".75" if e in cycle else "1" for e in DIST_EDGES}
    _undirected(s, DIST, DIST_EDGES, left, accent=cycle, weights=weights)
    s.card_centered(right, ["Ω tail edge   1", "Ω cycle edge  0.75", "pairs summed  15",
                            ("Σ Ω           21.0", "accent")])


# --- directed & assortativity ------------------------------------------------


@figure("get_reciprocity", "graph-metrics", "Reciprocity",
        "A five-node directed graph with seven arcs. The two reciprocal pairs, A and B and "
        "C and D, are drawn in the accent color: four of the seven arcs have a partner "
        "pointing back, so reciprocity is 4 / 7 = 0.571.")
def get_reciprocity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_reciprocity()")
    _directed(s, left, accent=DIR_MUTUAL)
    s.card_centered(right, ["arcs          7", "reciprocated  4", ("4 / 7         0.571", "accent")])


@figure("get_assortativity_degree", "graph-metrics", "Degree assortativity",
        "Two hubs C and D, each with degree 3, linked to each other and to two leaves of "
        "degree 1 apiece. Four of the five edges join a high-degree node to a low-degree "
        "one (accent), so mixing is disassortative and r = -0.667.")
def get_assortativity_degree(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_assortativity_degree()")
    edges = [("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("D", "F")]
    _undirected(s, PAIR, edges, left, accent={e for e in edges if e != ("C", "D")})
    _badges(s, {"A": 1, "B": 1, "C": 3, "D": 3, "E": 1, "F": 1})
    s.card_centered(right, ["hub–leaf  4 edges", "hub–hub   1 edge", ("r         -0.667", "accent")])


@figure("get_assortativity", "graph-metrics", "Numeric assortativity",
        "Two triangles joined by one edge, each node shaded by a numeric attribute shown "
        "in its badge: 1, 2, 3 on the left and 7, 8, 9 on the right. Low values link to "
        "low and high to high, so the correlation across edges is positive, r = 0.774.")
def get_assortativity(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_assortativity("age")')
    vals = {"A": 1, "B": 2, "C": 3, "D": 7, "E": 8, "F": 9}
    _undirected(s, PAIR, PAIR_EDGES, left, heat={k: v / 9 for k, v in vals.items()})
    _badges(s, vals)
    s.card_centered(right, ["low–low    3 edges", "high–high  3 edges", "low–high   1 edge",
                            ("r          0.774", "accent")])


@figure("get_assortativity_nominal", "graph-metrics", "Nominal assortativity",
        "Two triangles joined by one edge, with the nodes split into two categories: A, B, "
        "C in the accent color and D, E, F in teal. Six of the seven edges join nodes of "
        "the same category, so mixing is assortative and r = 0.714.")
def get_assortativity_nominal(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_assortativity_nominal("tag")')
    grp = {**{k: "accent" for k in "ABC"}, **{k: "g2" for k in "DEF"}}
    pos = place(PAIR, left, unit=UNIT)
    s.graph(pos, [(a, b, {"style": grp[a]} if grp[a] == grp[b] else {}) for a, b in PAIR_EDGES],
            styles=grp)
    s.card_centered(right, ["same tag   6 edges", "cross tag  1 edge",
                            ("r          0.714", "accent")])


# --- Laplacian spectrum ---------------------------------------------------------

TRI_LAP = [0.0, 0.830, 2.689, 4.0, 4.481]  # L = D − A, ascending


@figure("get_laplacian_spectrum", "graph-metrics", "Laplacian spectrum",
        "A five-node graph, two triangles sharing an edge plus a pendant, and a bar chart "
        "of the five eigenvalues of its Laplacian D minus A, sorted from smallest to largest: 0.000, 0.830, 2.689, 4.000, and 4.481. "
        "The single zero means the graph is connected, and the second-smallest eigenvalue, "
        "the algebraic connectivity 0.830, is in the accent color.")
def get_laplacian_spectrum(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_laplacian_spectrum()")
    _undirected(s, TRI, TRI_EDGES, left)
    _bars(s, right, TRI_LAP, [f"{v:.3f}" for v in TRI_LAP], [f"λ{i}" for i in range(1, 6)],
          accent={1}, bar_w=30, gap=14, scale=26, base_y=160)
