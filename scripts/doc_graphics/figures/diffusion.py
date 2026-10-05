"""Diffusion: spreading a signal from seed nodes over the network.

Template: `compute`. Left: the graph with the seed nodes you pass in `soft`
(accent outline) and the spread shown as `heat` on every node (darker =
higher). Right: a table of node → score sorted high to low, tied nodes sharing
a row, the top row highlighted, exactly as in the node-metrics family.
`network_smooth` rewrites the scores it is given, so it uses `transform`
instead: raw scores as ink badges before → smoothed scores as heat + accent
badges after. Selection-style methods (`influence_maximization`,
`expand_seed_set`) show the chosen or added nodes in `accent`; `voterank`
also numbers its picks with filled order badges, as traversals do.

All six methods use ONE shared graph (`TWO_*`: a diamond A–D joined by the
bridge D–E to the triangle E, F, G) and, where they take seeds, the same seed
A, so the spreading methods can be compared directly: heat diffusion and
random walk with restart both rank A > B, C > D > E > F, G, and the bridge
throttles how much reaches the right cluster.

Seeds are keyed by node id in the API, so captions show the real call
(`seeds={1: 1.0}`) and the seed carries an `id 1` pill to tie it to A. Every number is
what edgecase returns for the depicted graph (rounded to 2 decimals).
"""

from core import Scene, compute, figure, place, transform

UNIT = 80  # one scale for the whole family (matches node_metrics)

# --- the shared two-cluster graph ----------------------------------------------

TWO_IDS = {"A": 1}  # edgecase node ids for the seeds shown (nodes added A..G)
TWO_COORDS = {"A": (0, 0.6), "B": (0.7, 0), "C": (0.7, 1.2), "D": (1.4, 0.6),
              "E": (2.4, 0.6), "F": (3.1, 0), "G": (3.1, 1.2)}
TWO_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("D", "E"),
             ("E", "F"), ("E", "G"), ("F", "G")]


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _rows(scores: dict[str, float]) -> list[list[str]]:
    """Table body sorted high→low; nodes with the same rounded score share a row."""
    groups: dict[str, list[str]] = {}
    for k, v in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
        groups.setdefault(_fmt(v), []).append(k)
    return [[", ".join(ks), v] for v, ks in groups.items()]


def _spread(s: Scene, call: str, header: str, seeds: list[str],
            scores: dict[str, float]) -> None:
    """The family template: seeds soft, heat-shaded graph → sorted score table."""
    left, right = compute(s, "g", call)
    top = max(scores.values())
    s.graph(place(TWO_COORDS, left, unit=UNIT), TWO_EDGES,
            heat={k: v / top for k, v in scores.items()},
            styles={k: "soft" for k in seeds})
    for k in seeds:
        s.badge(k, f"id {TWO_IDS[k]}")
    rows = [["node", header]] + _rows(scores)
    s.table_centered(right, rows, [64, 64], highlight={(1, 0): "accent", (1, 1): "accent"})


# --- spreading from a seed -------------------------------------------------------


@figure("heat_diffusion", "diffusion", "Heat diffusion from seed nodes",
        "A seven-node graph, a diamond A to D joined by the bridge D-E to the triangle E, F, "
        "G, with the seed A (node id 1) outlined and every node shaded by the heat it holds after "
        "diffusing for time 1. The table reads A 0.31, B and C 0.24, D 0.15, E 0.04, and F "
        "and G 0.01: heat fades with distance from the seed and barely crosses the bridge.")
def heat_diffusion(s: Scene) -> None:
    _spread(s, "g.heat_diffusion(seeds={1: 1.0})", "heat", ["A"],
            {"A": 0.31497, "B": 0.236, "C": 0.236, "D": 0.149, "E": 0.0405,
             "F": 0.0119, "G": 0.0119})


@figure("random_walk_with_restart", "diffusion", "Random walk with restart",
        "The same seven-node two-cluster graph with the seed A (node id 1) outlined and every node "
        "shaded by its steady-state visit probability for a walker that jumps back to A with "
        "probability 0.15 at each step. The table reads A 0.27, B and C 0.22, D 0.14, E 0.07, "
        "and F and G 0.03.")
def random_walk_with_restart(s: Scene) -> None:
    _spread(s, "g.random_walk_with_restart(seeds={1: 1.0})", "prob", ["A"],
            {"A": 0.2748, "B": 0.2202, "C": 0.2202, "D": 0.1448, "E": 0.0706,
             "F": 0.0348, "G": 0.0348})


# raw scores (C is a dropout inside the scored cluster) and edgecase's output
RAW = {"A": 1, "B": 1, "C": 0, "D": 1, "E": 0, "F": 0, "G": 0}
SMOOTH = {"A": 0.722, "B": 0.88, "C": 0.451, "D": 0.745, "E": 0.14, "F": 0.031, "G": 0.031}
# fixed badge spots so each node's value sits in the same place in both panels
SPOT = {"A": "nw", "B": "ne", "C": "se", "D": "ne", "E": "s", "F": "ne", "G": "se"}


@figure("network_smooth", "diffusion", "Smooth node scores over the network",
        "Before: the shared seven-node graph with raw scores as badges, 1 on A, B and D and "
        "0 elsewhere, the scored nodes outlined. After: the smoothed scores, shaded as heat. "
        "C, a 0 surrounded by scored neighbors, rises to 0.45, while the far triangle E, F, G "
        "stays near 0.")
def network_smooth(s: Scene) -> None:
    left, right = transform(s, "g", "g.network_smooth(scores)")
    seeds = {k: "soft" for k, v in RAW.items() if v}
    s.graph(place(TWO_COORDS, left, unit=UNIT), TWO_EDGES, prefix="l.", styles=seeds)
    s.graph(place(TWO_COORDS, right, unit=UNIT), TWO_EDGES, prefix="r.",
            heat={k: v / max(SMOOTH.values()) for k, v in SMOOTH.items()})
    for k in TWO_COORDS:
        s.badge("l." + k, str(RAW[k]), where=SPOT[k], color="ink")
        s.badge("r." + k, _fmt(SMOOTH[k]), where=SPOT[k])


# --- choosing and growing seed sets -----------------------------------------------


@figure("influence_maximization", "diffusion", "Influence maximization",
        "The shared seven-node graph with the two nodes chosen by the greedy search, D and "
        "G, filled in accent: one seed in each cluster. The table lists their marginal gains "
        "in expected spread under the independent cascade model, 1.37 for D and 1.17 for G, "
        "for an expected spread of 2.54.")
def influence_maximization(s: Scene) -> None:
    left, right = compute(s, "g", "g.influence_maximization(k=2, seed=3)")
    s.graph(place(TWO_COORDS, left, unit=UNIT), TWO_EDGES,
            styles={"D": "accent", "G": "accent"})
    # edgecase returns [(4, 1.37), (7, 1.17)]: (node id, marginal gain) in pick order
    rows = [["node", "gain"], ["D", "1.37"], ["G", "1.17"], ["spread", "2.54"]]
    s.table_centered(right, rows, [64, 64],
                     highlight={(i, j): "accent" for i in (1, 2) for j in (0, 1)})


@figure("expand_seed_set", "diffusion", "Expand a seed set into a module",
        "The shared seven-node graph with the seeds A and B outlined and the two nodes the "
        "expansion adds, C and D, filled in accent, all four inside a shaded module. Each "
        "joins with 2 of its 3 edges into the module, a score of 2/3, and the size cap of 4 "
        "stops the growth before E. The card returns A, B, C, D.")
def expand_seed_set(s: Scene) -> None:
    left, right = compute(s, "g", "g.expand_seed_set(seeds, max_size=4)")
    s.hull(["A", "B", "C", "D"])
    s.graph(place(TWO_COORDS, left, unit=UNIT), TWO_EDGES,
            styles={"A": "soft", "B": "soft", "C": "accent", "D": "accent"})
    # score = edges into the current module / degree; C first, then D
    s.card_centered(right, ["seeds   A, B", "+ C     2/3", "+ D     2/3",
                            ("[A, B, C, D]", "accent")], title="links in / degree")


@figure("voterank", "diffusion", "VoteRank spreaders",
        "The shared seven-node graph with the two spreaders VoteRank elects filled in accent "
        "and numbered in election order: first B, which ties on three votes and has the "
        "lowest id, then E in the other cluster, because B's neighbors lose voting power "
        "once B is elected. The card returns B, E.")
def voterank(s: Scene) -> None:
    left, right = compute(s, "g", "g.voterank(count=2)")
    s.graph(place(TWO_COORDS, left, unit=UNIT), TWO_EDGES,
            styles={"B": "accent", "E": "accent"})
    s.badge("B", "1", filled=True)
    s.badge("E", "2", filled=True)
    s.card_centered(right, ["round 1  B", "round 2  E", ("[B, E]", "accent")],
                    title="elected in order")
