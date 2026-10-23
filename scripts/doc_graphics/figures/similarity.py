"""Similarity and link prediction: how alike two nodes are.

Template: `compute`. Left: the graph with the queried pair `soft` and the
witnesses the score is built from (shared neighbors, counted edges, current
paths) in `accent`; neighbors of only one side stay `default`. Right: a card
with the formula's actual numbers, ending in the value in accent (or a compact
table when the method returns many pairs or a matrix).

Every neighbor-based measure uses the SAME 7-node graph and the same pair
(B, E), so the reader can compare formulas on identical inputs:
N(B) = {A, C, D}, N(E) = {C, D, F, G}; the two shared neighbors have different
degrees (C: 3, D: 2), which is what separates Adamic-Adar and resource
allocation from the plain count. SimRank* and communicability reuse it too.
The two graph-vs-graph comparisons (edge overlap, edit distance) draw G and H
side by side in the input panel, with the matched or edited parts in accent.
The exact edit-path methods do the same with graphs built and checked in
edgecase: optimal_edit_paths colors the first path's node pairs g2 / g3 / g4
in both graphs and the card, with the inserted edge dashed (`ghost`);
optimize_graph_edit_distance lists the falling costs, ending in the distance;
and optimize_edit_paths uses `contrast` on its docstring star, roots `soft`.
The isomorphism and hashing methods follow their docstring examples and use
node-group colors for classes: find_isomorphism gives each matched pair one
color (accent / g2 / g3) in both graphs and the card, the Weisfeiler-Lehman
figures color nodes by hash class, and simrank_similarity draws its
students and courses as two columns with names beside the nodes.

The later link-prediction scores follow their docstring examples, each
asserted with edgecase. common_neighbor_centrality scores the pairs of the
path A-B-C-D: the pair A, C `soft` with a dashed (`ghost`) candidate link,
their shared neighbor B in accent, and a pair / cn / d / score table with the
formula in a note line. The three community-aware scores
(cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster)
share one four-node square (`CMTY`): community 0 (A, B, D) and community 1
(C) are `g2` / `g3` hulls, the pair A, D is `soft` with a dashed candidate
link, and the common neighbors the score counts are accent (C stays plain
where the score ignores it); the card shows the formula's numbers.
"""

from core import SIZE_LABEL, SIZE_NOTE, Box, Scene, compute, contrast, figure, place, ring

UNIT = 60  # one graph scale for the whole family

# The shared graph: B and E (the pair) with shared neighbors C (top) and D
# (bottom); A hangs off B, F and G off E, and C–F gives C a third edge.
SHARED = {"A": (0, 1), "B": (1, 1), "C": (2, 0), "D": (2, 2), "E": (3, 1),
          "F": (4, 0), "G": (4, 2)}
SHARED_EDGES = [("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"),
                ("E", "F"), ("E", "G"), ("C", "F")]
WITNESS_EDGES = {("B", "C"), ("B", "D"), ("C", "E"), ("D", "E")}

PAIRS = "pairs = [(B, E)]"  # card title for the ebunch-style methods
CARD_W = 224  # one card width for every shared-graph figure, so siblings line up


def _shared(s: Scene, box, *, witnesses: bool = True, styles: dict | None = None,
            accent_edges: set | None = None, muted_rest: bool = False) -> None:
    """Draw the shared graph: pair soft, shared neighbors and their edges accent."""
    st = {"B": "soft", "E": "soft"}
    if witnesses:
        st.update({"C": "accent", "D": "accent"})
    st.update(styles or {})
    hot = WITNESS_EDGES if accent_edges is None else accent_edges
    rest = {"style": "muted"} if muted_rest else {}
    s.graph(place(SHARED, box, unit=UNIT),
            [(a, b, {"style": "accent"} if (a, b) in hot else rest) for a, b in SHARED_EDGES],
            styles=st)


# --- shared-neighbor counts ------------------------------------------------


@figure("get_common_neighbors", "similarity", "Common neighbors",
        "Nodes B and E are outlined in the accent color; their two shared neighbors C and "
        "D are filled in accent, while A (only next to B) and F and G (only next to E) stay "
        "plain. The card lists N(B) = A, C, D and N(E) = C, D, F, G, and returns C and D.")
def get_common_neighbors(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_common_neighbors("B", "E")')
    _shared(s, left)
    s.card_centered(right, ["N(B) = {A, C, D}", "N(E) = {C, D, F, G}", ("[C, D]", "accent")], w=CARD_W)


@figure("common_neighbors", "similarity", "Common neighbor count",
        "Nodes B and E are outlined in the accent color and their two shared neighbors C "
        "and D are filled in accent. The card shows the intersection of the two "
        "neighborhoods, C and D, giving a common-neighbor count of 2 for the pair.")
def common_neighbors(s: Scene) -> None:
    left, right = compute(s, "g", "g.common_neighbors(pairs)")
    _shared(s, left)
    s.card_centered(right, ["N(B) ∩ N(E) = {C, D}", ("count = 2", "accent")], w=CARD_W, title=PAIRS)


# --- normalized overlaps ---------------------------------------------------


@figure("jaccard_coefficient", "similarity", "Jaccard coefficient",
        "Nodes B and E are outlined in the accent color with their shared neighbors C and "
        "D filled in accent. A table scores three requested pairs by shared over total "
        "distinct neighbors: B, E is 2 of 5 = 0.4, C, D is 2 of 3 = 0.67, and A, E is 0.")
def jaccard_coefficient(s: Scene) -> None:
    left, right = compute(s, "g", "g.jaccard_coefficient(pairs)")
    _shared(s, left)
    rows = [["pair", "∩", "∪", "score"], ["B, E", "2", "5", "0.4"], ["C, D", "2", "3", "0.67"],
            ["A, E", "0", "5", "0"]]
    s.table_centered(right, rows, [56, 36, 36, 56],
                     highlight={(1, j): "accent" for j in range(4)})


@figure("get_jaccard_similarity", "similarity", "Jaccard similarity",
        "Nodes B and E are outlined in the accent color with their shared neighbors C and "
        "D filled in accent; A, F, and G each touch only one side. The card divides the 2 "
        "shared neighbors by the 5 distinct neighbors: Jaccard similarity 0.4.")
def get_jaccard_similarity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_jaccard_similarity()")
    _shared(s, left)
    s.card_centered(right, ["|N(B) ∩ N(E)| = 2", "|N(B) ∪ N(E)| = 5", ("jaccard = 0.4", "accent")],
                    w=CARD_W, title="sim[B][E]")


@figure("get_dice_similarity", "similarity", "Dice similarity",
        "Nodes B and E are outlined in the accent color with their shared neighbors C and "
        "D filled in accent. The card doubles the 2 shared neighbors to 4 and divides by "
        "the summed degrees 3 + 4 = 7: Dice similarity 0.571.")
def get_dice_similarity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_dice_similarity()")
    _shared(s, left)
    s.card_centered(right, ["2·|N(B) ∩ N(E)| = 4", "|N(B)| + |N(E)| = 7", ("dice = 0.571", "accent")],
                    w=CARD_W, title="sim[B][E]")


@figure("get_cosine_similarity", "similarity", "Cosine similarity",
        "Nodes B and E are outlined in the accent color with their shared neighbors C and "
        "D filled in accent. The card divides the 2 shared neighbors by the square root of "
        "3 times 4, about 3.464: cosine similarity 0.577.")
def get_cosine_similarity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_cosine_similarity()")
    _shared(s, left)
    s.card_centered(right, ["|N(B) ∩ N(E)| = 2", "√(|N(B)|·|N(E)|) = 3.464",
                            ("cosine = 0.577", "accent")], w=CARD_W, title="sim[B][E]")


@figure("get_overlap_similarity", "similarity", "Overlap similarity",
        "Nodes B and E are outlined in the accent color with their shared neighbors C and "
        "D filled in accent. The card divides the 2 shared neighbors by the smaller "
        "neighborhood, B with 3: overlap coefficient 0.667.")
def get_overlap_similarity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_overlap_similarity()")
    _shared(s, left)
    s.card_centered(right, ["|N(B) ∩ N(E)| = 2", "min(|N(B)|, |N(E)|) = 3",
                            ("overlap = 0.667", "accent")], w=CARD_W, title="sim[B][E]")


# --- degree-weighted link prediction ---------------------------------------


@figure("adamic_adar_index", "similarity", "Adamic-Adar index",
        "Nodes B and E are outlined in the accent color with their shared neighbors C and "
        "D filled in accent; badges give C degree 3 and D degree 2. The card sums 1 over "
        "the log of each degree, 0.910 + 1.443, for an Adamic-Adar index of 2.353.")
def adamic_adar_index(s: Scene) -> None:
    left, right = compute(s, "g", "g.adamic_adar_index(pairs)")
    _shared(s, left)
    s.badge("C", "deg 3")
    s.badge("D", "deg 2")
    s.card_centered(right, ["1/ln 3 + 1/ln 2", "0.910 + 1.443", ("AA = 2.353", "accent")],
                    w=CARD_W, title=PAIRS)


@figure("resource_allocation_index", "similarity", "Resource allocation index",
        "Nodes B and E are outlined in the accent color with their shared neighbors C and "
        "D filled in accent; badges give C degree 3 and D degree 2. The card sums 1 over "
        "each degree, 1/3 + 1/2, for a resource allocation index of 0.833.")
def resource_allocation_index(s: Scene) -> None:
    left, right = compute(s, "g", "g.resource_allocation_index(pairs)")
    _shared(s, left)
    s.badge("C", "deg 3")
    s.badge("D", "deg 2")
    s.card_centered(right, ["1/3 + 1/2", "0.333 + 0.500", ("RA = 0.833", "accent")],
                    w=CARD_W, title=PAIRS)


@figure("preferential_attachment", "similarity", "Preferential attachment",
        "Nodes B and E are outlined in the accent color and every edge touching them is in "
        "accent; badges give B degree 3 and E degree 4. Shared neighbors play no role: the "
        "card multiplies the two degrees, 3 times 4, for a score of 12.")
def preferential_attachment(s: Scene) -> None:
    left, right = compute(s, "g", "g.preferential_attachment(pairs)")
    incident = {(a, b) for a, b in SHARED_EDGES if "B" in (a, b) or "E" in (a, b)}
    _shared(s, left, witnesses=False, accent_edges=incident)
    s.badge("B", "deg 3")
    s.badge("E", "deg 4")
    s.card_centered(right, ["deg(B) = 3", "deg(E) = 4", ("PA = 3 × 4 = 12", "accent")],
                    w=CARD_W, title=PAIRS)


# --- many pairs at once ----------------------------------------------------


@figure("get_pairwise_similarity", "similarity", "Pairwise similarity matrix",
        "Four requested nodes B, C, D, and E are outlined in the accent color. A 4 by 4 "
        "table gives their Jaccard similarities, with 1 on the diagonal; the B, E cells "
        "read 0.4 and C, D is the most similar pair at 0.67.")
def get_pairwise_similarity(s: Scene) -> None:
    left, right = compute(s, "g", "g.get_pairwise_similarity(nodes)")
    _shared(s, left, witnesses=False, accent_edges=set(), styles={"C": "soft", "D": "soft"})
    ids = list("BCDE")
    m = {("B", "E"): "0.4", ("C", "D"): "0.67", ("C", "E"): "0.17"}
    val = lambda a, b: "1" if a == b else m.get((a, b)) or m.get((b, a)) or "0"  # noqa: E731
    rows = [[""] + ids] + [[a] + [val(a, b) for b in ids] for a in ids]
    s.table_centered(right, rows, [30, 42, 42, 42, 42],
                     highlight={(1, 4): "accent", (4, 1): "accent"})


# --- attribute similarity among neighbors ---------------------------------


@figure("get_similar_neighbors", "similarity", "Similar neighbors",
        "Node E is outlined in the accent color and nodes B through G carry badges with "
        "their value of x. Of E's neighbors, C and F match E's value of 3 and are filled in "
        "accent; D and G differ, and B matches but is not a neighbor, so it is faded.")
def get_similar_neighbors(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_similar_neighbors("E", "x")')
    near = {("C", "E"), ("D", "E"), ("E", "F"), ("E", "G")}
    s.graph(place(SHARED, left, unit=UNIT),
            [(a, b, {"style": "accent"} if (a, b) in {("C", "E"), ("E", "F")}
              else {} if (a, b) in near else {"style": "muted"}) for a, b in SHARED_EDGES],
            styles={"E": "soft", "C": "accent", "F": "accent", "A": "muted", "B": "muted"})
    for k, v in {"B": 3, "C": 3, "D": 5, "E": 3, "F": 3, "G": 1}.items():
        s.badge(k, f"x={v}")
    s.card_centered(right, ["x(E) = 3", "neighbors: C D F G", ("[C, F]", "accent")], w=CARD_W)


# --- embeddedness ----------------------------------------------------------


@figure("dispersion", "similarity", "Dispersion",
        "Nodes A and B are outlined in the accent color and share three neighbors C, D, "
        "and E, filled in accent. C and D are linked, but E touches neither, so two of the "
        "three shared-neighbor pairs are unlinked: dispersion 2 / 3 = 0.667.")
def dispersion(s: Scene) -> None:
    left, right = compute(s, "g", 'g.dispersion("A", "B")')
    # D sits above the A–B line so its two spokes don't read as one A–B edge
    coords = {"A": (0, 1.35), "C": (1.5, 0), "D": (1.5, 0.85), "E": (1.5, 2.1), "B": (3, 1.35)}
    spokes = [(p, w, {"style": "accent"}) for p in "AB" for w in "CDE"]
    s.graph(place(coords, left, unit=UNIT), spokes + [("C", "D")],
            styles={"A": "soft", "B": "soft", "C": "accent", "D": "accent", "E": "accent"})
    s.card_centered(right, ["common: C, D, E", "unlinked: C–E, D–E", ("2 / 3 = 0.667", "accent")],
                    w=CARD_W)


# --- resistance ------------------------------------------------------------

# A 5-cycle seen as two parallel routes from A to B: A–C–B (2 Ω) above and
# A–D–E–B (3 Ω) below.
RING = {"A": (0, 1), "C": (1.5, 0), "D": (0.8, 2), "E": (2.2, 2), "B": (3, 1)}
RING_EDGES = [("A", "C"), ("C", "B"), ("A", "D"), ("D", "E"), ("E", "B")]


@figure("get_resistance_distance", "similarity", "Resistance distance",
        "Nodes A and B are outlined in the accent color and joined by two routes drawn in "
        "accent, A–C–B and A–D–E–B, with each edge a 1 ohm resistor. The card combines the "
        "2 and 3 ohm routes in parallel, 2·3 / (2+3), for a resistance of 1.2.")
def get_resistance_distance(s: Scene) -> None:
    left, right = compute(s, "g", 'g.get_resistance_distance("A", "B")')
    s.graph(place(RING, left, unit=UNIT), [(a, b, {"style": "accent"}) for a, b in RING_EDGES],
            styles={"A": "soft", "B": "soft"})
    # the call is too long for the caption with literal labels; the card names them
    s.card_centered(right, ["A–C–B     2 Ω", "A–D–E–B   3 Ω", "2·3 / (2+3)", ("R = 1.2", "accent")],
                    title="1 Ω per edge")


@figure("resistance_distance_matrix", "similarity", "Resistance distance matrix",
        "A five-node cycle A–C–B–E–D. The table gives the resistance distance for every "
        "pair: 0 on the diagonal, 0.8 for adjacent nodes, and 1.2 for nodes two steps "
        "apart; A and B are outlined in the accent color and their 1.2 cells are highlighted.")
def resistance_distance_matrix(s: Scene) -> None:
    left, right = compute(s, "g", "g.resistance_distance_matrix()")
    s.graph(place(RING, left, unit=UNIT), RING_EDGES, styles={"A": "soft", "B": "soft"})
    ids = list("ABCDE")
    adj = {frozenset(e) for e in RING_EDGES}
    val = lambda a, b: "0" if a == b else "0.8" if frozenset((a, b)) in adj else "1.2"  # noqa: E731
    rows = [[""] + ids] + [[a] + [val(a, b) for b in ids] for a in ids]
    s.table_centered(right, rows, [28, 38, 38, 38, 38, 38],
                     highlight={(1, 2): "accent", (2, 1): "accent"})


# --- SimRank* and communicability: every walk, not just shared neighbors -----


def _sub_table(s: Scene, box, vals: dict, note: str) -> None:
    """4 × 4 table over B, C, D, E (like get_pairwise_similarity), B–E highlighted."""
    ids = list("BCDE")
    val = lambda a, b: "1" if a == b else vals.get((a, b)) or vals[(b, a)]  # noqa: E731
    rows = [[""] + ids] + [[a] + [val(a, b) for b in ids] for a in ids]
    x0, y0, _, _ = s.table_centered(box, rows, [30, 42, 42, 42, 42],
                                    highlight={(1, 4): "accent", (4, 1): "accent"})
    s.text(box.cx, y0 - 12, note, size=SIZE_NOTE, opacity=0.6)


SIMRANK = {("B", "C"): "0.44", ("B", "D"): "0.49", ("B", "E"): "0.30", ("C", "D"): "0.36",
           ("C", "E"): "0.47", ("D", "E"): "0.47"}


@figure("simrank_star_similarity", "similarity", "SimRank* similarity",
        "Nodes B and E are outlined in the accent color, with shared neighbors C and D in "
        "accent. A 4 by 4 table, part of the full all-pairs result, gives SimRank* scores for "
        "B, C, D, and E with 1 on the diagonal; the highlighted B, E cells read 0.30.")
def simrank_star_similarity(s: Scene) -> None:
    left, right = compute(s, "g", "g.simrank_star_similarity()")
    _shared(s, left)
    _sub_table(s, right, SIMRANK, "simrank_star[u][v], C = 0.8")


@figure("simrank_star_similarity_pair", "similarity", "SimRank* similarity for one pair",
        "Nodes B and E are outlined in the accent color; the short paths between them through "
        "C and D are in accent. SimRank* counts paths of every length, decayed by the factor "
        "C = 0.8 per step, and the card returns a score of 0.297 for the pair.")
def simrank_star_similarity_pair(s: Scene) -> None:
    left, right = compute(s, "g", 'g.simrank_star_similarity_pair("B","E")')
    _shared(s, left)
    s.card_centered(right, ["in-link paths via C, D, …", "each step decays by C",
                            ("s*(B, E) = 0.297", "accent")], w=CARD_W, title="importance_factor C = 0.8")


@figure("communicability", "similarity", "Communicability",
        "Nodes B and E are outlined in the accent color, and the two walks of length 2 between "
        "them, through C and D, are in accent. Communicability sums the walks of every length k "
        "divided by k factorial, 2/2! + 1/3! + 15/4! and so on, giving 2.069 for B and E.")
def communicability(s: Scene) -> None:
    left, right = compute(s, "g", "g.communicability()")
    _shared(s, left)
    s.card_centered(right, ["Σ (k-step walks) / k!", "2/2! + 1/3! + 15/4! + …",
                            ("C[B][E] = 2.069", "accent")], w=CARD_W, title="result[B][E]")


# --- comparing two graphs --------------------------------------------------

# G and H on the same four node labels (as in set-ops): both have A–B and B–D.
SQ = {"A": (0, 0), "B": (1.5, 0), "C": (0, 1.5), "D": (1.5, 1.5)}


def _pair(s: Scene, left, g_edges: list, h_edges: list, *, g_styles: dict | None = None,
          h_styles: dict | None = None, h_pos: dict | None = None) -> None:
    """G in the left half of the input panel, H in the right half, each captioned."""
    mid = left.cx
    gbox = Box(left.x0, left.y0, mid, left.y1)
    hbox = Box(mid, left.y0, left.x1, left.y1)
    s.graph(place(SQ, gbox, unit=UNIT), g_edges, styles=g_styles, prefix="g.")
    s.graph(place(h_pos or SQ, hbox, unit=UNIT), h_edges, styles=h_styles, prefix="h.")
    # the template's left caption ("G") moves under G; H gets its own
    cap = next(t for t in s.texts if t.role == "caption" and t.text == "G")
    cap.x = gbox.cx
    s.caption(hbox.cx, "H", max_w=hbox.w)


@figure("edge_overlap", "similarity", "Edge overlap",
        "Two graphs on nodes A to D. G and H share edges A–B and B–D, drawn in accent; A–C is "
        "only in G and C–D only in H. The card counts 2 shared edges out of 4 in the union, "
        "one unique to each side, for a Jaccard index of 0.5.")
def edge_overlap(s: Scene) -> None:
    left, right = compute(s, "G", "G.edge_overlap(H)")
    hot = {"style": "accent"}
    _pair(s, left, [("A", "B", hot), ("B", "D", hot), ("A", "C")],
          [("A", "B", hot), ("B", "D", hot), ("C", "D")])
    s.card_centered(right, ["overlap = 2   union = 4", "g1_only = 1   g2_only = 1",
                            ("jaccard = 2 / 4 = 0.5", "accent")], w=CARD_W, title="edges matched by label")


@figure("graph_edit_distance", "similarity", "Graph edit distance",
        "Graph G has nodes A, B, C, D and H has A, B, D, E; both share edges A–B and B–D. "
        "Turning G into H deletes node C and edge A–C (accent in G) and inserts node E and "
        "edge D–E (accent in H): four edits at cost 1 each, so the distance is 4.")
def graph_edit_distance(s: Scene) -> None:
    left, right = compute(s, "G", "G.graph_edit_distance(H)")
    hot = {"style": "accent"}
    _pair(s, left, [("A", "B"), ("B", "D"), ("A", "C", hot)], [("A", "B"), ("B", "D"), ("D", "E", hot)],
          g_styles={"C": "accent"}, h_styles={"E": "accent"},
          h_pos={"A": (0, 0), "B": (1.5, 0), "E": (0, 1.5), "D": (1.5, 1.5)})
    s.card_centered(right, ["delete  node C, edge A–C", "insert  node E, edge D–E",
                            ("distance = 2 + 2 = 4", "accent")], w=CARD_W, title="cost 1 per edit")


# --- exact edit paths ------------------------------------------------------
#
# The exact search (optimal_edit_paths, optimize_graph_edit_distance,
# optimize_edit_paths) works on node and edge ids, so these figures build the
# graphs with edgecase, assert what it returns, and draw from that.

# optimal_edit_paths docstring: a three-node path laid onto a triangle.
PATH3 = {"A": (0, 1.2), "B": (0.8, 0), "C": (1.6, 1.2)}
TRI3 = {"X": (0, 1.2), "Y": (0.8, 0), "Z": (1.6, 1.2)}


def _ids(g, pairs) -> list[tuple]:
    """Node-id pairs of an edit path as label pairs (None stays None)."""
    return [tuple(None if x is None else gr.get_label_from_id(x) for x, gr in zip(p, g))
            for p in pairs]


@figure("optimal_edit_paths", "similarity", "Optimal edit paths",
        "A three-node path A–B–C beside a triangle X, Y, Z. The first of the six optimal edit "
        "paths, each costing 1, maps A to X, B to Y and C to Z (matching colors in both "
        "graphs and the card); the path's two edges land on X–Y and Y–Z, and the triangle's "
        "third edge Z–X, dashed in accent, is the one insertion.")
def optimal_edit_paths(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(list(PATH3)).add_edges([("A", "B"), ("B", "C")])
    h = ec.Graph().add_nodes(list(TRI3)).add_edges([("X", "Y"), ("Y", "Z"), ("Z", "X")])
    paths, cost = g.optimal_edit_paths(h)
    assert (len(paths), cost) == (6, 1.0)
    node_path, edge_path = paths[0]
    assert _ids((g, h), node_path) == [("A", "X"), ("B", "Y"), ("C", "Z")]
    inserted = [f for e, f in edge_path if e is None]
    assert [h.get_node_ids_by_edge_id(f) for f in inserted] == [(3, 1)]  # Z–X

    left, right = compute(s, "path", "path.optimal_edit_paths(triangle)")
    pbox, tbox = _halves(s, left, "path", "triangle")
    colors = dict(zip(PATH3, ("g2", "g3", "g4")))
    s.graph(place(PATH3, pbox, unit=UNIT), [("A", "B"), ("B", "C")], styles=colors, prefix="g.")
    s.graph(place(TRI3, tbox, unit=UNIT), [("X", "Y"), ("Y", "Z"), ("Z", "X", {"style": "ghost"})],
            styles={v: colors[u] for u, v in _ids((g, h), node_path)}, prefix="h.")
    s.card_centered(right, [(f"{u}  →  {v}", colors[u]) for u, v in _ids((g, h), node_path)]
                    + [("insert edge Z–X", "accent")], w=CARD_W,
                    title=f"paths[0] of {len(paths)}, cost = {cost}")


# optimize_graph_edit_distance: a diamond against a relabeled diamond with a
# pendant, edges listed in an order that sends the search through four
# upper bounds before the distance (asserted below).
DIA_G = {"A": (1, 0), "B": (1, 1.5), "C": (0, 0.75), "D": (2, 0.75)}
DIA_G_EDGES = [("B", "C"), ("A", "D"), ("A", "C"), ("B", "D"), ("A", "B")]
DIA_H = {"A": (0, 0.75), "B": (2, 1.5), "C": (2, 0.75), "D": (1, 0), "E": (1, 1.5)}
DIA_H_EDGES = [("C", "D"), ("A", "E"), ("C", "E"), ("D", "E"), ("A", "D"), ("B", "E")]


@figure("optimize_graph_edit_distance", "similarity", "Improving edit distance estimates",
        "Graph G is a diamond, two triangles sharing edge A–B; H is the same diamond on "
        "other labels, sharing D–E, with an extra node B hanging off E (accent). The search "
        "yields the costs 10, 8, 6 and 4 as it finds cheaper edit paths, and ends at 2, the "
        "exact distance: insert node B and edge E–B.")
def optimize_graph_edit_distance(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(list("ABCD")).add_edges(DIA_G_EDGES)
    h = ec.Graph().add_nodes(list("ABCDE")).add_edges(DIA_H_EDGES)
    costs = list(g.optimize_graph_edit_distance(h))
    assert costs == [10.0, 8.0, 6.0, 4.0, 2.0]
    *_, (node_path, _, _) = g.optimize_edit_paths(h)
    assert ("A", "B") not in _ids((g, h), node_path) and (None, "B") in _ids((g, h), node_path)

    left, right = compute(s, "G", "list(G.optimize_graph_edit_distance(H))")
    gbox, hbox = _halves(s, left, "G", "H")
    s.graph(place(DIA_G, gbox, unit=UNIT), DIA_G_EDGES, prefix="g.")
    s.graph(place(DIA_H, hbox, unit=UNIT),
            [(a, b, {"style": "accent"}) if "B" in (a, b) else (a, b) for a, b in DIA_H_EDGES],
            styles={"B": "accent"}, prefix="h.")
    first, *mid, last = costs
    s.card_centered(right, [f"first  {first}", "then   " + "  ".join(map(str, mid)),
                            (f"last   {last} = distance", "accent")],
                    w=CARD_W, title="each cost below the last")


# optimize_edit_paths docstring: a six-node star against itself, center 1.
STAR6 = {"1": (0, 0), **ring([str(i) for i in range(2, 7)])}
STAR_UNIT = 58  # two stars side by side in each half of the contrast


@figure("optimize_edit_paths", "similarity", "Edit paths with forced roots",
        "A six-node star, center 1 and leaves 2 to 6, compared with itself twice, each "
        "comparison drawing the star and its copy side by side. On the left the roots pair "
        "the two centers (outlined in accent) and the cheapest edit path costs "
        "0. On the right the roots force leaf 2 of the first star onto the center of the "
        "second, and the cheapest path costs 8: four edges deleted and four inserted to "
        "rewire the star around its new center.")
def optimize_edit_paths(s: Scene) -> None:
    import edgecase as ec

    star = ec.Graph().add_star_graph(6)
    assert [(a, b) for _, a, b in star.get_edges_as_tuples()] == [(1, i) for i in range(2, 7)]
    best = {r: [c for *_, c in star.optimize_edit_paths(star, roots=r)][-1] for r in ((1, 1), (2, 1))}
    assert best == {(1, 1): 0.0, (2, 1): 8.0}

    left, right = contrast(s, "optimize_edit_paths(star, roots=(1, 1))",
                           "optimize_edit_paths(star, roots=(2, 1))",
                           str(best[1, 1]), str(best[2, 1]))
    edges = [("1", str(i)) for i in range(2, 7)]
    for box, (u, v), tag in ((left, (1, 1), "l"), (right, (2, 1), "r")):
        a, b = Box(box.x0, box.y0, box.cx, box.y1), Box(box.cx, box.y0, box.x1, box.y1)
        s.graph(place(STAR6, a, unit=STAR_UNIT), edges, styles={str(u): "soft"}, prefix=f"{tag}g.")
        s.graph(place(STAR6, b, unit=STAR_UNIT), edges, styles={str(v): "soft"}, prefix=f"{tag}h.")


# --- isomorphism and structural hashes -------------------------------------


def _name(s: Scene, node_id: str, text: str, where: str, color: str = "ink") -> None:
    """A node name too long for its circle, set just outside it (n, s, e or w)."""
    n = s.nodes[node_id]
    dx = {"w": -(n.r + 5), "e": n.r + 5}.get(where, 0)
    dy = {"n": -(n.r + 10), "s": n.r + 10}.get(where, 0)
    anchor = {"w": "end", "e": "start"}.get(where, "middle")
    s.text(n.x + dx, n.y + dy, text, size=SIZE_LABEL, mono=False, weight=600, anchor=anchor,
           color=color, role="name")


def _halves(s: Scene, left: Box, a_cap: str, b_cap: str) -> tuple[Box, Box]:
    """Split the input panel into two graphs, each captioned (the template's caption moves)."""
    abox, bbox = Box(left.x0, left.y0, left.cx, left.y1), Box(left.cx, left.y0, left.x1, left.y1)
    cap = next(t for t in s.texts if t.role == "caption" and t.text == a_cap)
    cap.x = abox.cx
    s.caption(bbox.cx, b_cap, max_w=bbox.w)
    return abox, bbox


# find_isomorphism docstring: one workflow exported twice under different names.
FLOW_A = {"fetch": (0, 1), "build": (1, 0), "test": (2, 1)}
FLOW_A_EDGES = [("fetch", "build"), ("build", "test"), ("fetch", "test")]
FLOW_B = {"s1": (0, 0.15), "s2": (1.7, 0), "s3": (0.8, 1.2)}
FLOW_B_EDGES = [("s2", "s3"), ("s1", "s3"), ("s1", "s2")]


@figure("find_isomorphism", "similarity", "Find isomorphism",
        "Two directed graphs drawn differently. Graph a is a workflow where fetch points to "
        "build and test, and build points to test; graph b has nodes s1, s2, s3 with s1 "
        "pointing to s2 and s3, and s2 pointing to s3. Matching colors pair each node of a "
        "with its partner in b, and the card returns the mapping fetch to s1 (accent), build "
        "to s2 (teal), and test to s3 (violet), under which every edge of a lands on an edge "
        "of b.")
def find_isomorphism(s: Scene) -> None:
    import edgecase as ec

    a = ec.Graph(directed=True).add_nodes(list(FLOW_A))
    a.add_edges(FLOW_A_EDGES)
    b = ec.Graph(directed=True).add_nodes(list(FLOW_B))
    b.add_edges(FLOW_B_EDGES)
    got = {a.get_label_from_id(k): b.get_label_from_id(v) for k, v in a.find_isomorphism(b).items()}
    assert got == {"fetch": "s1", "build": "s2", "test": "s3"}
    assert a.find_isomorphism(ec.Graph(directed=True).add_path_graph(3)) is None
    left, right = compute(s, "a", "a.find_isomorphism(b)")
    abox, bbox = _halves(s, left, "a", "b")
    colors = dict(zip(FLOW_A, ("accent", "g2", "g3")))
    s.graph(place(FLOW_A, Box(abox.x0, abox.y0, abox.x1, abox.y1 - 14), unit=UNIT),
            FLOW_A_EDGES, directed=True, styles=colors, labels=False, prefix="a.")
    for k, where in zip(FLOW_A, ("s", "n", "s")):
        _name(s, "a." + k, k, where, colors[k])
    s.graph(place(FLOW_B, bbox, unit=UNIT), FLOW_B_EDGES, directed=True,
            styles={got[k]: c for k, c in colors.items()}, prefix="b.")
    s.card_centered(right, [(f"{k:<5}  →  {v}", colors[k]) for k, v in got.items()],
                    w=CARD_W, title="node of a → node of b")


# weisfeiler_lehman_graph_hash docstring: a triangle with a tail, relabeled.
TAIL_A = {"A": (0, 0), "B": (0, 1.4), "C": (1.1, 0.7), "D": (2.2, 0.7)}
TAIL_A_EDGES = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")]
TAIL_B = {"w": (0.7, 0), "x": (0.7, 1), "y": (0, 2), "z": (1.4, 2)}
TAIL_B_EDGES = [("z", "y"), ("y", "x"), ("x", "z"), ("w", "x")]


@figure("weisfeiler_lehman_graph_hash", "similarity", "Weisfeiler-Lehman graph hash",
        "Two drawings of a triangle with a one-edge tail, one labeled A to D and one labeled "
        "w to z. Node colors mark the classes that neighborhood aggregation finds: the "
        "degree-3 node (C, x) in accent, the two other triangle nodes (A, B and y, z) in "
        "teal, and the tail end (D, w) in violet. Both graphs reduce to the same class counts, "
        "so the card shows the same hash, starting c0454391, for each, and equal is True.")
def weisfeiler_lehman_graph_hash(s: Scene) -> None:
    import edgecase as ec

    a = ec.Graph().add_nodes(list(TAIL_A))
    a.add_edges(TAIL_A_EDGES)
    b = ec.Graph().add_nodes(list(TAIL_B))
    b.add_edges(TAIL_B_EDGES)
    ha, hb = a.weisfeiler_lehman_graph_hash(), b.weisfeiler_lehman_graph_hash()
    assert ha == hb and ha.startswith("c0454391")
    hexagon = ec.Graph().add_cycle_graph(6)
    two = ec.Graph().add_cycle_graph(3).add_cycle_graph(3)
    assert hexagon.weisfeiler_lehman_graph_hash() == two.weisfeiler_lehman_graph_hash()
    left, right = compute(s, "a", "a.weisfeiler_lehman_graph_hash()")
    abox, bbox = _halves(s, left, "a", "b")
    s.graph(place(TAIL_A, abox, unit=UNIT), TAIL_A_EDGES,
            styles={"C": "accent", "A": "g2", "B": "g2", "D": "g3"}, prefix="a.")
    s.graph(place(TAIL_B, bbox, unit=UNIT * 0.8), TAIL_B_EDGES,
            styles={"x": "accent", "y": "g2", "z": "g2", "w": "g3"}, prefix="b.")
    s.card_centered(right, [f"a  {ha[:8]}…", f"b  {hb[:8]}…", ("equal  True", "accent")],
                    w=CARD_W, title="hash, first 8 hex digits")


# weisfeiler_lehman_subgraph_hashes docstring: a five-node path.
PATH5 = {k: (i, 0) for i, k in enumerate("ABCDE")}
PATH5_EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]


@figure("weisfeiler_lehman_subgraph_hashes", "similarity", "Weisfeiler-Lehman subgraph hashes",
        "A path of five nodes, A to E, colored by their second-round hashes: the two ends A "
        "and E share one hash (accent), the inner nodes B and D share another (teal), and the "
        "middle node C has its own (violet). The card lists the first six hex digits of each "
        "class in matching colors: 68f978 for A and E, b51b1d for B and D, and 0ea5aa for C.")
def weisfeiler_lehman_subgraph_hashes(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(list(PATH5))
    g.add_edges(PATH5_EDGES)
    second = {g.get_label_from_id(n): h[1][:6]
              for n, h in g.weisfeiler_lehman_subgraph_hashes(iterations=2).items()}
    classes = {"accent": "AE", "g2": "BD", "g3": "C"}
    for nodes in classes.values():
        assert len({second[k] for k in nodes}) == 1
    assert len(set(second.values())) == 3
    assert (second["A"], second["B"], second["C"]) == ("68f978", "b51b1d", "0ea5aa")
    left, right = compute(s, "g", "g.weisfeiler_lehman_subgraph_hashes()")
    s.graph(place(PATH5, left, unit=UNIT * 1.3), PATH5_EDGES,
            styles={k: c for c, nodes in classes.items() for k in nodes})
    s.card_centered(right, [(f"{', '.join(nodes):<4}  {second[nodes[0]]}", c)
                            for c, nodes in classes.items()],
                    w=CARD_W, title="round 2, first 6 hex digits")


# simrank_similarity docstring: students and the courses they take.
STUDENTS = ["ann", "bob", "cy"]
COURSES = ["math", "art", "bio"]
ENROLL = [("ann", "math"), ("ann", "art"), ("bob", "math"), ("bob", "art"), ("cy", "bio")]


@figure("simrank_similarity", "similarity", "SimRank similarity",
        "Students ann, bob, and cy in a column on the left, courses math, art, and bio on the "
        "right. ann and bob, outlined in the accent color, take the same two courses, math and "
        "art (accent), while cy only takes bio. ann and bob are never linked directly, yet "
        "SimRank with C = 0.9 rates them 0.818 because their neighbors are alike; ann and cy "
        "share nothing and score 0.0.")
def simrank_similarity(s: Scene) -> None:
    import edgecase as ec

    g = ec.Graph().add_nodes(STUDENTS + COURSES)
    g.add_edges(ENROLL)
    ab, ac = g.simrank_similarity("ann", "bob"), g.simrank_similarity("ann", "cy")
    assert round(ab, 3) == 0.818 and ac == 0.0
    left, right = compute(s, "g", 'g.simrank_similarity("ann", "bob")')
    coords = {k: (0, i) for i, k in enumerate(STUDENTS)} | {k: (2.6, i) for i, k in
                                                            enumerate(COURSES)}
    styles = {"ann": "soft", "bob": "soft", "math": "accent", "art": "accent"}
    s.graph(place(coords, left, unit=UNIT), [(a, b, {"style": "accent"}) if b != "bio" else
                                            (a, b) for a, b in ENROLL],
            styles=styles, labels=False)
    for k in coords:
        _name(s, k, k, "w" if k in STUDENTS else "e", "accent" if k in styles else "ink")
    s.card_centered(right, [(f"s(ann, bob) = {ab:.3f}", "accent"), f"s(ann, cy)  = {ac:.1f}"],
                    w=CARD_W, title="importance_factor C = 0.9")


# --- later additions: link prediction with distance and communities --------


def _link_graph(nodes, edges):
    import edgecase as ec

    g = ec.Graph().add_nodes(nodes)
    g.add_edges(edges)
    return g


def _scores(g, method: str, *args) -> dict:
    return {(g.get_label_from_id(u), g.get_label_from_id(v)): sc
            for u, v, sc in getattr(g, method)(*args)}


PATH4 = {"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0)}
PATH4_EDGES = [("A", "B"), ("B", "C"), ("C", "D")]


@figure("common_neighbor_centrality", "similarity", "Common neighbor centrality",
        "A path A, B, C, D. The candidate link A–C (dashed) joins nodes that share the "
        "neighbor B (accent) and are 2 steps apart. With alpha 0.8 and 4 nodes, the table "
        "scores every non-adjacent pair as 0.8 times the shared neighbors plus 0.2 times 4 "
        "over the distance: A–C and B–D score 1.2, and A–D, with no shared neighbor and 3 "
        "steps apart, only 0.267.")
def common_neighbor_centrality(s: Scene) -> None:
    g = _link_graph(list(PATH4), PATH4_EDGES)
    got = {k: round(v, 3) for k, v in _scores(g, "common_neighbor_centrality").items()}
    assert got == {("A", "C"): 1.2, ("A", "D"): 0.267, ("B", "D"): 1.2}
    left, right = compute(s, "g", "g.common_neighbor_centrality()")
    gbox = Box(left.x0, left.y0 + 30, left.x1, left.y1)  # room for the candidate arc
    s.graph(place(PATH4, gbox, unit=UNIT * 1.5),
            [("A", "B", {"style": "accent"}), ("B", "C", {"style": "accent"}), ("C", "D"),
             ("A", "C", {"style": "ghost", "bend": -30})],
            styles={"A": "soft", "C": "soft", "B": "accent"})
    rows = [["pair", "cn", "d", "score"], ["A, C", "1", "2", "1.2"], ["A, D", "0", "3", "0.267"],
            ["B, D", "1", "2", "1.2"]]
    top = right.cy - 40 + 8
    s.text(right.cx, top - 12, "0.8·cn + 0.2·4 / d", size=SIZE_NOTE, opacity=0.6)
    s.table(right.cx - 98, top, rows, [56, 36, 36, 68],
            highlight={(1, j): "accent" for j in range(4)})


# The community docstrings' square: A, B, D in community 0 and C in community 1.
CMTY = {"A": (0, 0.7), "B": (1, 0), "C": (1, 1.6), "D": (2, 0.7)}
CMTY_EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
CMTY_OF = {"A": 0, "B": 0, "C": 1, "D": 0}


def _cmty_graph():
    import edgecase as ec

    g = ec.Graph().add_nodes([(k, {"community": c}) for k, c in CMTY_OF.items()])
    g.add_edges(CMTY_EDGES)
    return g


def _cmty(s: Scene, box: Box, counted: str) -> None:
    """The community square: pair A, D soft, the `counted` common neighbors accent."""
    pos = place(CMTY, box, unit=UNIT * 1.15)
    s.graph(pos, [(a, b, {"style": "accent"}) if (set(counted) & {a, b}) else (a, b)
                  for a, b in CMTY_EDGES] + [("A", "D", {"style": "ghost"})],
            styles={"A": "soft", "D": "soft", **dict.fromkeys(counted, "accent")})
    s.hull(["A", "B", "D"], "g2")
    s.hull(["C"], "g3")
    bx, by = pos["B"]
    s.text(bx + 30, by, "community 0", size=SIZE_NOTE, anchor="start", color="g2")
    cx, cy = pos["C"]
    s.text(cx + 30, cy, "community 1", size=SIZE_NOTE, anchor="start", color="g3")


@figure("cn_soundarajan_hopcroft", "similarity", "Common neighbors with community bonus",
        "Four nodes in a square: A, B and D form community 0 (teal region) and C is community "
        "1 (violet region). The candidate link A–D (dashed) has two common neighbors, both in "
        "the accent color. B shares the pair's community, so it counts twice; C does not, so "
        "it counts once. The card adds 2 + 1 for a score of 3.0.")
def cn_soundarajan_hopcroft(s: Scene) -> None:
    g = _cmty_graph()
    got = _scores(g, "cn_soundarajan_hopcroft", [("A", "D")])
    assert got == {("A", "D"): 3.0}
    left, right = compute(s, "g", 'g.cn_soundarajan_hopcroft(pairs)')
    _cmty(s, left, "BC")
    s.card_centered(right, ["B  same community   2", "C  other community  1",
                            ("score  2 + 1 = 3.0", "accent")], w=CARD_W, title="pairs = [(A, D)]")


@figure("ra_index_soundarajan_hopcroft", "similarity", "Resource allocation with communities",
        "Four nodes in a square: A, B and D form community 0 (teal region) and C is community "
        "1 (violet region). Of the two common neighbors of the candidate link A–D (dashed), "
        "only B (accent) is in the pair's community, so only B counts, with 1 over its degree "
        "2. C is ignored. The card gives the score 0.5.")
def ra_index_soundarajan_hopcroft(s: Scene) -> None:
    g = _cmty_graph()
    got = _scores(g, "ra_index_soundarajan_hopcroft", [("A", "D")])
    assert got == {("A", "D"): 0.5}
    left, right = compute(s, "g", 'g.ra_index_soundarajan_hopcroft(pairs)')
    _cmty(s, left, "B")
    s.card_centered(right, ["B  1 / deg 2   0.5", "C  other community  0",
                            ("score  0.5", "accent")], w=CARD_W, title="pairs = [(A, D)]")


@figure("within_inter_cluster", "similarity", "Within-inter-cluster ratio",
        "Four nodes in a square: A, B and D form community 0 (teal region) and C is community "
        "1 (violet region). The candidate link A–D (dashed) has one common neighbor inside "
        "its community, B (accent), and one outside, C. The card divides 1 within by 1 inter "
        "plus delta 0.001, a score of 0.999.")
def within_inter_cluster(s: Scene) -> None:
    g = _cmty_graph()
    got = {k: round(v, 3) for k, v in _scores(g, "within_inter_cluster", [("A", "D")]).items()}
    assert got == {("A", "D"): 0.999}
    left, right = compute(s, "g", 'g.within_inter_cluster(pairs)')
    _cmty(s, left, "B")
    s.card_centered(right, ["within  {B}  1", "inter   {C}  1",
                            ("1 / (1 + 0.001) = 0.999", "accent")], w=CARD_W, title="pairs = [(A, D)]")
