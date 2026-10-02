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
allocation from the plain count.
"""

from core import Scene, compute, figure, place

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
