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
          clustering, spectrum, degree statistics; `generalized_degree`
          labels each edge with its triangle count)
- `DIR`   a five-node digraph with two reciprocal pairs (directed degree
          aggregates, reciprocity)
- `PAIR`  two clusters joined by one edge (assortativity). Its hub-and-leaf
          variant (`HUB_EDGES`, degree badges, hub–leaf edges accent) is
          shared by every degree-mixing method: assortativity, Pearson r,
          mixing dict / matrix, average degree connectivity. The attribute
          mixing methods draw the full PAIR with the two input teams as
          `g2` / `g3` and the cross edge in accent; matrices are small
          tables with their row/column labels in a note line above
- `STAR`  a hub with three leaves and a three-node tail, where the
          center and the barycenter differ (get_barycenter; badges give each
          node's total distance)

Directed hierarchy scalars borrow graphs from node_metrics_local.py so the
per-node and whole-graph versions match: `global_reaching_centrality` uses
`contrast` (the out-tree `TREE`, its local reaching values as badges, beside
a directed 4-cycle where every node reaches everyone), and
`trophic_incoherence_parameter` reuses the food web with its edges labelled
by level difference and a card computing their standard deviation.

Graph matrices (`laplacian_matrix`, `normalized_laplacian_matrix`,
`bethe_hessian_matrix`, `directed_laplacian_matrix`, `modularity_matrix`)
swap the card for a labelled matrix table (`_matrix`) with the formula in a
note line; the Laplacian family uses TRI with degree badges, modularity uses
PAIR (two triangles), whose positive cells form one block per triangle. The
modularity and Bethe Hessian spectra are bar charts on PAIR, like
`get_spectrum`. `algebraic_connectivity` and `kemeny_constant` use `contrast`
on six nodes: the cycle beside the path (its closing edge accent / removed)
and the complete graph beside the cycle (the added chords accent).

The small-world coefficients share `SW`, an eight-node ring whose every
second node also joins a square of chords, so it has four triangles. Its
reference values come from edgecase's internal `small_world()` (the pieces
the step narration shows): the card gives each ratio's numbers and ends in
sigma or omega in accent; the seed is in the caption. transforms.py draws
random_reference and lattice_reference on the same ring.

Every number on a card is what edgecase returns for the depicted graph.
"""

from core import R_SMALL, SIZE_NOTE, Box, Scene, compute, contrast, figure, place, ring

from figures.node_metrics_local import TREE, TREE_ARCS, food_web

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


@figure("get_center", "graph-metrics", "Center",
        "A six-node graph with each node's eccentricity in a badge. The middle nodes C and D "
        "have the minimum eccentricity 2 and are drawn in the accent color. The card returns "
        "their node ids, 3 and 4.")
def get_center(s: Scene) -> None:
    # The mirror image of get_periphery: same graph, badges, and card layout.
    left, right = compute(s, "g", "g.get_center()")
    _undirected(s, DIST, DIST_EDGES, left, styles={"C": "accent", "D": "accent"})
    _badges(s, DIST_ECC)
    s.card_centered(right, ["min ecc  2", "nodes    C, D", ("ids      [3, 4]", "accent")])


# A star with a tail: leaves A, B, C on hub D, then the chain E–F–G. The tail
# pulls the center (min eccentricity) to E; the leaves hold the barycenter
# (min total distance) at D. Badges: each node's total distance to the rest.
STAR = {"A": (0, -0.05), "B": (0, 0.6), "C": (0, 1.25), "D": (1, 0.6), "E": (2, 0.6),
        "F": (3, 0.6), "G": (4, 0.6)}
STAR_EDGES = [("A", "D"), ("B", "D"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G")]
STAR_SUM = {"A": 14, "B": 14, "C": 14, "D": 9, "E": 10, "F": 13, "G": 18}


@figure("get_barycenter", "graph-metrics", "Barycenter",
        "A hub D with three leaves A, B, C and a tail E, F, G, each node badged with its "
        "total distance to all the others. D has the smallest total, 9, so it is the "
        "barycenter, in the accent color; the card returns its id 4. The center, which "
        "minimizes the farthest distance instead, is E, one step down the tail.")
def get_barycenter(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_barycenter()")
    _undirected(s, STAR, STAR_EDGES, left, styles={"D": "accent"})
    _badges(s, STAR_SUM, where={"A": "n", "B": "w", "C": "s"})
    s.card_centered(right, ["min Σd   9", "nodes    D", ("ids      [4]", "accent"),
                            "center   E (ecc 2)"])


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


# --- hierarchy in directed graphs ---------------------------------------------


@figure("global_reaching_centrality", "graph-metrics", "Global reaching centrality",
        "Left, a five-node out-tree, each node badged with the share of other nodes it "
        "reaches: A 1, B 0.5, and the leaves 0. Averaging each node's gap to the top value "
        "gives (0 + 0.5 + 1 + 1 + 1) / 4 = 0.875, a strong hierarchy. Right, in a directed "
        "4-cycle every node reaches all others, there is no gap, and the score is 0.0.")
def global_reaching_centrality(s: Scene) -> None:
    left, right = contrast(s, "g.global_reaching_centrality()",
                           "g.global_reaching_centrality()", "0.875", "0.0")
    s.graph(place(TREE, left, unit=UNIT), TREE_ARCS, directed=True, prefix="l.",
            styles={"A": "accent"})
    for k, v in {"A": "1", "B": "0.5", "C": "0", "D": "0", "E": "0"}.items():
        s.badge("l." + k, v, color="accent" if k == "A" else "ink")
    cyc = {"A": (0, 0), "B": (1.2, 0), "C": (1.2, 1.2), "D": (0, 1.2)}
    s.graph(place(cyc, right, unit=UNIT), [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
            directed=True, prefix="r.")
    for k in cyc:
        s.badge("r." + k, "1", color="ink")


@figure("trophic_incoherence_parameter", "graph-metrics", "Trophic incoherence parameter",
        "The food web of grass, rabbit, mouse, fox and owl, laid out by trophic level with "
        "each edge labelled by its level difference: four edges of exactly 1, mouse to owl "
        "1.5 and fox to owl 0.5. The card takes their mean, 1.0, and their standard "
        "deviation, q = 0.29; a perfectly layered web would score 0.")
def trophic_incoherence_parameter(s: Scene) -> None:
    left, right = compute(s, "g", "g.trophic_incoherence_parameter()")
    food_web(s, left, diffs=True)
    # population std of [1, 1, 1, 1, 1.5, 0.5] = sqrt(0.5 / 6)
    s.card_centered(right, ["diffs  1 1 1 1 1.5 0.5", "mean   1.0",
                            ("q = σ  0.29", "accent")])


# --- degree and attribute mixing -------------------------------------------------

# Degree mixing reuses the hub-and-leaf variant of PAIR from
# get_assortativity_degree (hubs C, D of degree 3, four leaves of degree 1), so
# the dict, matrix, k_nn table and Pearson r all describe that one picture.
HUB_EDGES = [("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("D", "F")]
HUB_DEG = {"A": 1, "B": 1, "C": 3, "D": 3, "E": 1, "F": 1}
HUB_LEAF = {e for e in HUB_EDGES if e != ("C", "D")}


def _hubs(s: Scene, box: Box) -> None:
    """The hub-and-leaf graph, hub–leaf edges in accent, degrees as badges."""
    _undirected(s, PAIR, HUB_EDGES, box, accent=HUB_LEAF)
    _badges(s, HUB_DEG)


@figure("average_degree_connectivity", "graph-metrics", "Average degree connectivity",
        "Two hubs C and D of degree 3, linked to each other and to two leaves of degree 1 "
        "apiece. Every leaf's neighbor is a hub, so degree 1 maps to 3.0, while the hubs' "
        "six neighbors have degrees 1, 1, 3, 3, 1, 1, so degree 3 maps to 1.667: a falling "
        "curve, the mark of disassortative mixing.")
def average_degree_connectivity(s: Scene) -> None:
    left, right = compute(s, "g", "g.average_degree_connectivity()")
    _hubs(s, left)
    rows = [["k", "nodes", "nbr degrees", "k_nn"], ["1", "A B E F", "3 3 3 3", "3.0"],
            ["3", "C D", "1 1 3 3 1 1", "1.667"]]
    s.table_centered(right, rows, [28, 64, 90, 52],
                     highlight={(1, 3): "accent", (2, 3): "accent"})


@figure("degree_mixing_dict", "graph-metrics", "Degree mixing dict",
        "The hub-and-leaf graph with each degree in a badge. The four accent edges join "
        "degree 1 to degree 3, and the hub edge C–D joins 3 to 3. Counting every edge from "
        "both ends gives {1: {3: 4}, 3: {1: 4, 3: 2}}.")
def degree_mixing_dict(s: Scene) -> None:
    left, right = compute(s, "g", "g.degree_mixing_dict()")
    _hubs(s, left)
    s.card_centered(right, [("1: {3: 4}", "accent"), ("3: {1: 4, 3: 2}", "accent"),
                            "C–D counts 3→3 twice"], title="{deg u: {deg v: edge ends}}")


@figure("degree_mixing_matrix", "graph-metrics", "Degree mixing matrix",
        "The hub-and-leaf graph with each degree in a badge, and its 2 by 2 mixing matrix "
        "over the degrees 1 and 3. Of the ten edge ends, eight join a leaf to a hub, 0.4 in "
        "each off-diagonal cell (accent), and two join hub to hub, 0.2; no leaf touches a "
        "leaf.")
def degree_mixing_matrix(s: Scene) -> None:
    left, right = compute(s, "g", "g.degree_mixing_matrix()")
    _hubs(s, left)
    rows = [["", "1", "3"], ["1", "0.0", "0.4"], ["3", "0.4", "0.2"]]
    _, y0, _, _ = s.table_centered(right, rows, [40, 52, 52],
                                   highlight={(1, 2): "accent", (2, 1): "accent"})
    s.text(right.cx, y0 - 12, "degrees = [1, 3]", size=SIZE_NOTE, opacity=0.6)


@figure("degree_pearson_correlation_coefficient", "graph-metrics", "Degree Pearson correlation",
        "The hub-and-leaf graph with each degree in a badge. Over the ten edge ends the "
        "mean degree is 2.2; the degrees at the two ends of an edge have covariance -0.64 "
        "and variance 0.96, so their correlation is r = -0.667: hubs link to leaves.")
def degree_pearson_correlation_coefficient(s: Scene) -> None:
    left, right = compute(s, "g", "g.degree_pearson_correlation_coefficient()")
    _hubs(s, left)
    # pairs (1, 3) and (3, 1) four times each, (3, 3) twice
    s.card_centered(right, ["mean deg    2.2", "cov         -0.64", "var         0.96",
                            ("cov / var   -0.667", "accent")], title="over 10 edge ends")


# Attribute mixing reuses PAIR with a categorical "team": A, B, C are x (g2)
# and D, E, F are y (g3) -- input groups, so accent stays free for the result.
TEAM = {**{k: "g2" for k in "ABC"}, **{k: "g3" for k in "DEF"}}


def _teams(s: Scene, box: Box) -> None:
    pos = place(PAIR, box, unit=UNIT)
    s.graph(pos, [(a, b, {"style": TEAM[a]} if TEAM[a] == TEAM[b] else {"style": "accent"})
                  for a, b in PAIR_EDGES], styles=TEAM)
    # name each team beside its outer edge (A–B, E–F), at mid-height
    my = (pos["A"][1] + pos["B"][1]) / 2
    s.text(pos["A"][0] - 12, my, "team x", size=SIZE_NOTE, anchor="end", color="g2")
    s.text(pos["E"][0] + 12, my, "team y", size=SIZE_NOTE, anchor="start", color="g3")


@figure("attribute_mixing_dict", "graph-metrics", "Attribute mixing dict",
        "Two triangles joined by one edge, with team x on A, B, C (teal) and team y on D, "
        "E, F (violet). Each triangle's three edges count from both ends, 6 within each "
        "team, and the one accent edge C–D links x to y once each way: {x: {x: 6, y: 1}, "
        "y: {x: 1, y: 6}}.")
def attribute_mixing_dict(s: Scene) -> None:
    left, right = compute(s, "g", 'g.attribute_mixing_dict("team")')
    _teams(s, left)
    s.card_centered(right, [("x: {x: 6, y: 1}", "g2"), ("y: {x: 1, y: 6}", "g3"),
                            "C–D gives x→y and y→x"], title="{team u: {team v: edge ends}}")


@figure("attribute_mixing_matrix", "graph-metrics", "Attribute mixing matrix",
        "The same two teams, x on A, B, C (teal) and y on D, E, F (violet), and their 2 by "
        "2 mixing matrix. Twelve of the fourteen edge ends stay within a team, 0.43 in each "
        "diagonal cell (accent); the single cross edge C–D puts 0.07 in each off-diagonal "
        "cell.")
def attribute_mixing_matrix(s: Scene) -> None:
    left, right = compute(s, "g", 'g.attribute_mixing_matrix("team")')
    _teams(s, left)
    rows = [["", "x", "y"], ["x", "0.43", "0.07"], ["y", "0.07", "0.43"]]
    _, y0, _, _ = s.table_centered(right, rows, [40, 52, 52],
                                   highlight={(1, 1): "accent", (2, 2): "accent"})
    s.text(right.cx, y0 - 12, "values = [x, y]", size=SIZE_NOTE, opacity=0.6)


# --- triangles per edge -------------------------------------------------------------

TRI_K = {("A", "B"): 1, ("A", "C"): 1, ("B", "C"): 2, ("B", "D"): 1, ("C", "D"): 1,
         ("D", "E"): 0}  # triangles through each edge


@figure("generalized_degree", "graph-metrics", "Generalized degree",
        "Two triangles sharing the edge B–C, plus E hanging off D, with each edge labelled "
        "by the number of triangles it lies in. D's three edges, in accent, lie in 1, 1 and "
        "0 triangles, so D maps to {0: 1, 1: 2}; the shared edge B–C lies in 2, giving B "
        "and C {1: 2, 2: 1}.")
def generalized_degree(s: Scene) -> None:
    left, right = compute(s, "g", "g.generalized_degree()")
    d_edges = {e for e in TRI_EDGES if "D" in e}
    _undirected(s, TRI, TRI_EDGES, left, accent=d_edges, styles={"D": "soft"},
                weights={e: str(k) for e, k in TRI_K.items()})
    rows = [["node", "{k: edges}"], ["A", "{1: 2}"], ["B", "{1: 2, 2: 1}"],
            ["C", "{1: 2, 2: 1}"], ["D", "{0: 1, 1: 2}"], ["E", "{0: 1}"]]
    s.table_centered(right, rows, [52, 110], highlight={(4, 0): "accent", (4, 1): "accent"})


# --- graph matrices ------------------------------------------------------------
#
# Matrix-valued methods return (matrix, node_ids). They draw the matrix as a
# compact table with the node labels as row / column headers (ids follow the
# labels: A = 1, B = 2, ...) and the formula in a note line above. Cells use
# the signed encoding: the entries the figure is about are tinted accent (the
# diagonal, positive pairs), and negative entries that matter are tinted g2.
# The Laplacian family (L, normalized L, Bethe Hessian) shares TRI with degree
# badges, so the diagonal can be read off the graph and matches
# get_laplacian_spectrum; modularity uses PAIR (two triangles), whose positive
# cells form one block per triangle.


def _num(v: float, digits: int = 2) -> str:
    """Matrix entry: integers bare, otherwise `digits` decimals; never -0."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)) + 0)
    out = f"{v:.{digits}f}"
    return "0" if float(out) == 0 else out


def _matrix(s: Scene, box: Box, labels: list[str], values: list[list[float]], note: str,
            *, cols: list[str] | None = None, highlight: dict | None = None,
            col_w: float = 40, head_w: float = 28, row_h: float = 20,
            digits: int = 2) -> None:
    """Matrix as a labelled table with a note line above; `highlight` maps
    (row, col) of the matrix (0-based, headers excluded) to a color."""
    cols = cols or labels
    rows = [[""] + cols] + [[lab] + [_num(v, digits) for v in row]
                            for lab, row in zip(labels, values)]
    hl = {(i + 1, j + 1): c for (i, j), c in (highlight or {}).items()}
    th = row_h * len(rows)
    inner = Box(box.x0, box.y0 + 12, box.x1, box.y1)  # room for the note
    _, y0, _, _ = s.table(inner.cx - (head_w + col_w * len(cols)) / 2, inner.cy - th / 2, rows,
                          [head_w] + [col_w] * len(cols), row_h=row_h, highlight=hl)
    s.text(box.cx, y0 - 12, note, size=SIZE_NOTE, opacity=0.6)


TRI_IDS = list("ABCDE")
TRI_LAPM = [[2, -1, -1, 0, 0], [-1, 3, -1, -1, 0], [-1, -1, 3, -1, 0], [0, -1, -1, 3, -1],
            [0, 0, 0, -1, 1]]
DIAG5 = {(i, i): "accent" for i in range(5)}


def _tri_degrees(s: Scene, box: Box, styles=None) -> None:
    _undirected(s, TRI, TRI_EDGES, box, styles=styles)
    _badges(s, TRI_DEG)


@figure("laplacian_matrix", "graph-metrics", "Laplacian matrix",
        "Two triangles sharing the edge B–C plus a pendant E, each node badged with its "
        "degree, and the 5 by 5 Laplacian D minus A. The degrees 2, 3, 3, 3, 1 run down the "
        "diagonal (accent), each edge puts -1 in its two cells, and every row sums to 0.")
def laplacian_matrix(s: Scene) -> None:
    left, right = compute(s, "g", "g.laplacian_matrix()")
    _tri_degrees(s, left)
    _matrix(s, right, TRI_IDS, TRI_LAPM, "L = D − A, rows sum to 0", highlight=DIAG5)


R3, R6 = 3 ** -0.5, 6 ** -0.5  # 1/√(3·1) for D–E, 1/√(2·3) for A–B, A–C


@figure("normalized_laplacian_matrix", "graph-metrics", "Normalized Laplacian matrix",
        "The same five-node graph with degree badges and its normalized Laplacian: 1 on the "
        "diagonal and, for each edge, minus one over the square root of its two degrees. "
        "Edges at A (degrees 2 and 3) give -0.41, edges between degree-3 nodes -0.33, and "
        "the highlighted D–E (3 and 1) -0.58.")
def normalized_laplacian_matrix(s: Scene) -> None:
    left, right = compute(s, "g", "g.normalized_laplacian_matrix()")
    _tri_degrees(s, left, styles={"D": "soft", "E": "soft"})
    t = 1 / 3
    vals = [[1, -R6, -R6, 0, 0], [-R6, 1, -t, -t, 0], [-R6, -t, 1, -t, 0],
            [0, -t, -t, 1, -R3], [0, 0, 0, -R3, 1]]
    _matrix(s, right, TRI_IDS, vals, "−1 / √(deg u · deg v)",
            highlight={(3, 4): "accent", (4, 3): "accent"})


@figure("bethe_hessian_matrix", "graph-metrics", "Bethe Hessian matrix",
        "The same five-node graph with degree badges and its Bethe Hessian at the default "
        "r = 1.67 (32 / 12 - 1, from the degrees). Each edge gives -r = -1.67 and the "
        "diagonal (accent) holds r² - 1 + degree: 3.78 for A, 4.78 for B, C and D, and 2.78 "
        "for E. At r = 1 it would be the Laplacian.")
def bethe_hessian_matrix(s: Scene) -> None:
    left, right = compute(s, "g", "g.bethe_hessian_matrix()")
    _tri_degrees(s, left)
    r = 5 / 3  # Σd² / Σd − 1 = 32 / 12 − 1
    vals = [[(r * r - 1 + TRI_DEG[a]) if a == b else (-r if v else 0)
             for b, v in zip(TRI_IDS, row)] for a, row in zip(TRI_IDS, TRI_LAPM)]
    _matrix(s, right, TRI_IDS, vals, "r = 1.67: (r² − 1) I − r A + D", highlight=DIAG5)


# Directed Laplacian: a directed 4-cycle with a chord, strongly connected and
# aperiodic (cycles of length 4 and 3), so the walk is a plain random walk.
DSQ = {"A": (0, 0), "B": (1.3, 0), "C": (1.3, 1.3), "D": (0, 1.3)}
DSQ_ARCS = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]


@figure("directed_laplacian_matrix", "graph-metrics", "Directed Laplacian matrix",
        "A directed four-cycle A, B, C, D with a chord from A to C, and its directed "
        "Laplacian, built from a random walk on the arcs. The matrix is symmetric even "
        "though the graph is not: the one-way arc A to B gives -0.35 in both the A, B and "
        "B, A cells (highlighted), the chord -0.25, and B and D, with no arc, 0.")
def directed_laplacian_matrix(s: Scene) -> None:
    left, right = compute(s, "g", "g.directed_laplacian_matrix()")
    pos = place(DSQ, left, unit=UNIT)
    s.graph(pos, [(a, b, {"style": "soft"}) if (a, b) == ("A", "B") else (a, b)
                  for a, b in DSQ_ARCS], directed=True, styles={"A": "soft", "B": "soft"})
    h = 0.5 ** 1.5  # 0.354
    vals = [[1, -h, -0.25, -0.5], [-h, 1, -h, 0], [-0.25, -h, 1, -0.5], [-0.5, 0, -0.5, 1]]
    _matrix(s, right, list("ABCD"), vals, "A → B is one-way; L is symmetric",
            col_w=46, highlight={(0, 1): "accent", (1, 0): "accent"})


PAIR_MOD = [[-0.286, 0.714, 0.571, -0.429, -0.286, -0.286],
            [0.714, -0.286, 0.571, -0.429, -0.286, -0.286],
            [0.571, 0.571, -0.643, 0.357, -0.429, -0.429],
            [-0.429, -0.429, 0.357, -0.643, 0.571, 0.571],
            [-0.286, -0.286, -0.429, 0.571, -0.286, 0.714],
            [-0.286, -0.286, -0.429, 0.571, 0.714, -0.286]]
PAIR_DEG = {"A": 2, "B": 2, "C": 3, "D": 3, "E": 2, "F": 2}


@figure("modularity_matrix", "graph-metrics", "Modularity matrix",
        "Two triangles A, B, C and D, E, F joined by the edge C–D, each node badged with its "
        "degree, and the 6 by 6 modularity matrix: actual edges minus those expected from "
        "the degrees, rounded to one decimal. The positive cells (accent) form one block per "
        "triangle plus C–D at 0.4; pairs across the triangles, such as C and E at -0.4, and "
        "the diagonal are negative.")
def modularity_matrix(s: Scene) -> None:
    left, right = compute(s, "g", "g.modularity_matrix()")
    _undirected(s, PAIR, PAIR_EDGES, left)
    _badges(s, PAIR_DEG)
    hl = {(i, j): "accent" for i, row in enumerate(PAIR_MOD) for j, v in enumerate(row) if v > 0}
    _matrix(s, right, list("ABCDEF"), PAIR_MOD, "B = A − k kᵀ / 2m,  m = 7", highlight=hl,
            col_w=36, head_w=24, digits=1)


PAIR_MODSPEC = [-1.73, -1.0, -1.0, -0.43, 0.0, 1.73]  # ±√3; 2 decimals


@figure("modularity_spectrum", "graph-metrics", "Modularity spectrum",
        "Two triangles joined by the edge C–D, and a bar chart of the six eigenvalues of "
        "its modularity matrix in increasing order: -1.73, -1.00, -1.00, -0.43, 0.00 and "
        "1.73. The single large positive eigenvalue, in the accent color, signals a "
        "two-way community split.")
def modularity_spectrum(s: Scene) -> None:
    left, right = compute(s, "g", "g.modularity_spectrum()")
    _undirected(s, PAIR, PAIR_EDGES, left)
    _bars(s, right, PAIR_MODSPEC, [f"{v:.2f}" for v in PAIR_MODSPEC], None, accent={5},
          bar_w=30, gap=12, scale=33, base_y=104)


PAIR_BH = [0.031, 0.748, 4.194, 4.469, 4.469, 6.334]  # default r = 34 / 14 − 1


@figure("bethe_hessian_spectrum", "graph-metrics", "Bethe Hessian spectrum",
        "Two triangles joined by the edge C–D, and a bar chart of the six eigenvalues of "
        "its Bethe Hessian at the default r = 1.43, in increasing order: 0.031, 0.748, "
        "4.194, 4.469, 4.469 and 6.334. The two small eigenvalues, in the accent color, "
        "stand well apart from the rest, matching the two triangle communities.")
def bethe_hessian_spectrum(s: Scene) -> None:
    left, right = compute(s, "g", "g.bethe_hessian_spectrum()")
    _undirected(s, PAIR, PAIR_EDGES, left)
    _bars(s, right, PAIR_BH, [f"{v:.3f}" for v in PAIR_BH], [f"λ{i}" for i in range(1, 7)],
          accent={0, 1}, bar_w=30, gap=12, scale=19, base_y=160)
    s.text(right.cx - 120, 24, "r = 1.43", size=SIZE_NOTE, anchor="start", opacity=0.6)


# --- connectivity of the whole graph: two contrasts on six nodes ------------------

RING6 = list("ABCDEF")
C6_EDGES = list(zip(RING6, RING6[1:] + RING6[:1]))  # A–B … F–A


@figure("algebraic_connectivity", "graph-metrics", "Algebraic connectivity",
        "Left, a six-node cycle whose closing edge F–A is in the accent color; right, the "
        "same nodes without that edge (dashed), a path. The cycle's algebraic connectivity "
        "is 1.0 and the path's only 0.268: the path is much easier to cut in two.")
def algebraic_connectivity(s: Scene) -> None:
    left, right = contrast(s, "g.algebraic_connectivity()", "g.algebraic_connectivity()",
                           "1.0", "0.268")
    coords = ring(RING6)
    for box, p, st in ((left, "l.", "accent"), (right, "r.", "removed")):
        s.graph(place(coords, box, unit=UNIT), [(a, b, {"style": st}) if (a, b) == ("F", "A")
                                                else (a, b) for a, b in C6_EDGES], prefix=p)


@figure("kemeny_constant", "graph-metrics", "Kemeny constant",
        "Left, the complete graph on six nodes: the nine edges it adds to the six-node cycle "
        "are in the accent color, and a random walk reaches a target drawn from its "
        "stationary distribution in 4.167 steps on average. Right, the cycle alone needs "
        "5.833. The value does not depend on the start node; smaller is better connected.")
def kemeny_constant(s: Scene) -> None:
    left, right = contrast(s, "g.kemeny_constant()", "g.kemeny_constant()", "4.167", "5.833")
    coords = ring(RING6)
    chords = [(a, b) for i, a in enumerate(RING6) for b in RING6[i + 2:]
              if (b, a) not in C6_EDGES and (a, b) not in C6_EDGES]
    s.graph(place(coords, left, unit=UNIT),
            C6_EDGES + [(a, b, {"style": "accent"}) for a, b in chords], prefix="l.")
    s.graph(place(coords, right, unit=UNIT), C6_EDGES, prefix="r.")


# --- later additions: small-world coefficients ------------------------------
#
# SW: a ring of eight nodes plus the chords A-C, C-E, E-G, G-A, which close
# four triangles around the ring. Random references (degree-preserving) lose
# most of them; path lengths barely change. Values use seed=1 and NetworkX's
# default niter / nrand.
SW_NODES = list("ABCDEFGH")
# Small nodes on a regular octagon (clockwise from A at the top) keep every
# chord, including the references' rewired ones, clear of the nodes it skips.
SW = ring(SW_NODES)
SW_EDGES = (list(zip(SW_NODES, SW_NODES[1:] + SW_NODES[:1]))
            + [("A", "C"), ("C", "E"), ("E", "G"), ("G", "A")])


def sw_graph(edges=SW_EDGES):
    import edgecase as ec

    g = ec.Graph().add_nodes(SW_NODES)
    g.add_edges(edges)
    return g


def _small_world(kind: str, niter: int) -> tuple[float, tuple[float, float, float, float]]:
    from edgecase import _core

    g = sw_graph()
    value, parts = _core.small_world(g._graph, kind, niter, 10, 1)
    method = getattr(g, f"small_world_{kind}")
    assert method(seed=1) == value  # the public call returns the same number
    return value, parts


@figure("small_world_sigma", "graph-metrics", "Small-world coefficient sigma",
        "An eight-node ring with four chords that close four triangles. Its transitivity C is "
        "0.43, against 0.354 averaged over 10 degree-preserving random references, while its "
        "average path length L, 1.643, is close to the references' 1.707. The card divides "
        "the two ratios, (0.43 / 0.354) / (1.643 / 1.707), for sigma = 1.260: above 1, "
        "so the graph counts as small-world.")
def small_world_sigma(s: Scene) -> None:
    value, (c, length, cr, lr) = _small_world("sigma", 100)
    assert (round(c, 3), round(length, 3), round(cr, 3), round(lr, 3)) == (
        0.429, 1.643, 0.354, 1.707)
    assert round(value, 3) == 1.26
    left, right = compute(s, "g", "g.small_world_sigma(seed=1)")
    s.graph(place(SW, left, unit=UNIT, r=R_SMALL), SW_EDGES, r=R_SMALL)
    s.card_centered(right, [f"C / C_r  {c:.3f} / {cr:.3f}", f"L / L_r  {length:.3f} / {lr:.3f}",
                            (f"sigma    {value:.3f}", "accent")],
                    title="(C / C_r) / (L / L_r)")


@figure("small_world_omega", "graph-metrics", "Small-world coefficient omega",
        "An eight-node ring with four chords that close four triangles. The random references "
        "average a path length L_r of 1.689 against the graph's 1.643, and the best lattice "
        "reference clusters no more than the graph itself, C_l = C = 0.667. The card computes "
        "1.689 / 1.643 minus 0.667 / 0.667, omega = 0.028: near 0, the small-world middle "
        "between a lattice (-1) and a random graph (1).")
def small_world_omega(s: Scene) -> None:
    value, (c, length, cl, lr) = _small_world("omega", 5)
    assert (round(c, 3), round(length, 3), round(cl, 3), round(lr, 3)) == (
        0.667, 1.643, 0.667, 1.689)
    assert round(value, 3) == 0.028
    left, right = compute(s, "g", "g.small_world_omega(seed=1)")
    s.graph(place(SW, left, unit=UNIT, r=R_SMALL), SW_EDGES, r=R_SMALL)
    s.card_centered(right, [f"L_r / L  {lr:.3f} / {length:.3f}", f"C / C_l  {c:.3f} / {cl:.3f}",
                            (f"omega    {value:.3f}", "accent")],
                    title="L_r / L − C / C_l")
