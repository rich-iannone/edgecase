"""Querying: the matches of a Cypher MATCH pattern, as a table, a selection, or a count.

Template: `compute` for match() (graph → table) and count_matches() (graph →
card), `transform` for select_match() (no selection → selection), as in the
selection family.

All three figures run one query, `MATCH (a)-->(b)-->(a)` (a 2-cycle), on one
directed graph: the selection family's 2x3 grid (`BASE`) with arrows, where
B and C, and D and E, point at each other and A and F are in no cycle. Each
2-cycle matches twice, once from each end, which is the Cypher rule the
figures teach. match() shows node ids as badges (it returns ids), its first
row and that match in `accent`; select_match() selects every node of every
match in `accent`; count_matches() draws the matched edges in `accent` and
lists the four matches on the card with the count.
"""

from core import Scene, compute, figure, place, transform

from figures.selection import BASE, NODE_ID

UNIT = 72  # the selection family's grid step

QUERY = "MATCH (a)-->(b)-->(a)"
EDGES = [("A", "B"), ("B", "C"), ("C", "B"), ("A", "D"), ("B", "E"), ("C", "F"),
         ("D", "E"), ("E", "D"), ("E", "F")]
MUTUAL = {("B", "C"), ("C", "B"), ("D", "E"), ("E", "D")}


def _graph():
    import edgecase as ec

    g = ec.Graph(directed=True).add_nodes(list(BASE))
    g.add_edges(EDGES)
    return g


def _matches() -> list[tuple[str, str]]:
    """The (a, b) labels of each match, in match() row order, checked against edgecase."""
    g = _graph()
    found = [tuple(row) for row in g.match(QUERY, use_labels=True).iter_rows()]
    assert found == [("B", "C"), ("C", "B"), ("D", "E"), ("E", "D")]
    assert g.count_matches(QUERY) == len(found)
    return found


def _draw(s: Scene, box, prefix: str, *, nodes=(), edges=()) -> None:
    """The query graph in `box`, with `nodes` and `edges` in `accent`."""
    items = []
    for e in EDGES:
        kw = {"bend": 14} if e in MUTUAL else {}
        if e in edges:
            kw["style"] = "accent"
        items.append((e[0], e[1], kw))
    s.graph(place(BASE, box, unit=UNIT), items, directed=True,
            styles=dict.fromkeys(nodes, "accent"), prefix=prefix)


@figure("match", "querying", "Match a pattern",
        "A directed six-node grid A to F, each node tagged with its integer id 1 to 6, where "
        "B and C point at each other and so do D and E. The query MATCH (a)-->(b)-->(a) finds "
        "the 2-cycles, and the result is a table with columns a and b holding node ids, one "
        "row per match: 2 3, 3 2, 4 5, and 5 4. Each cycle matches twice, once from each "
        "end. The first row and its match, B and C with both edges, are in the accent color.")
def match(s: Scene) -> None:
    found = _matches()
    left, right = compute(s, "g", f'g.match("{QUERY}")')
    a, b = found[0]
    _draw(s, left, "", nodes=(a, b), edges=((a, b), (b, a)))
    for k in BASE:
        s.badge(k, str(NODE_ID[k]), where="n" if BASE[k][1] == 0 else "s",
                color="accent" if k in (a, b) else "ink")
    rows = [["a", "b"]] + [[str(NODE_ID[x]), str(NODE_ID[y])] for x, y in found]
    s.table_centered(right, rows, [48, 48], highlight={(1, 0): "accent", (1, 1): "accent"})


@figure("select_match", "querying", "Select the matches of a pattern",
        "A directed six-node grid A to F with no selection, where B and C point at each other "
        "and so do D and E. After select_match with MATCH (a)-->(b)-->(a), every node in a "
        "2-cycle is selected: B, C, D, and E, in the accent color. A and F are in no cycle "
        "and stay plain.")
def select_match(s: Scene) -> None:
    found = _matches()
    g = _graph()
    g.select_match(QUERY)
    selected = sorted(g.get_label_from_id(i) for i in g.get_selected_node_ids())
    assert selected == sorted({k for m in found for k in m}) == ["B", "C", "D", "E"]
    left, right = transform(s, "g", f'g.select_match("{QUERY}")')
    _draw(s, left, "l.")
    _draw(s, right, "r.", nodes=selected)


@figure("count_matches", "querying", "Count the matches of a pattern",
        "A directed six-node grid A to F, where B and C point at each other and so do D and "
        "E; those four edges are in the accent color. The query MATCH (a)-->(b)-->(a) "
        "matches each 2-cycle once from each end, so the card lists B→C→B, C→B→C, D→E→D, "
        "and E→D→E, and the count is 4.")
def count_matches(s: Scene) -> None:
    found = _matches()
    left, right = compute(s, "g", f'g.count_matches("{QUERY}")')
    _draw(s, left, "", edges=MUTUAL)
    paths = [f"{a}→{b}→{a}" for a, b in found]
    s.card_centered(right, [f"{paths[0]}  {paths[1]}", f"{paths[2]}  {paths[3]}",
                            (f"count  {len(found)}", "accent")], title="matches")
