"""Layout: methods that give every node an (x, y) position.

Template: `transform`. The same graph twice: on the left scattered at random
(captioned `g`), on the right at the positions the method computes, so the
reader sees what the layout brings out. Both panels are drawn from live
edgecase output (`_positions`): the scatter is `layout_random()` with the seed
in `SCATTER`, stretched to the panel's 2:1 shape; the result is the captioned
call; and edgecase's y axis (up) is flipped for the canvas (down). Nodes are
small and labeled with their ids, since several layouts place nodes in id
order.

Nothing is highlighted except what the method keys on, in both panels: the
groups that `layout_shell()`, `layout_bipartite()` and `layout_multipartite()`
place apart are colored `accent`, `g2`, `g3` (edges inside a group take its
color), and the BFS start node is `soft`. The scatter for `layout_random()`
itself is swapped for a circle, so its output is the scattered side.
"""

import math

from core import MUTED, R_SMALL, Scene, figure, place, transform

UNIT = 96  # the layouts span about 2 units; this fills the panel height

GROUP_STYLES = ("accent", "g2", "g3", "g4")

# Seed of the output in the `layout_random()` figure, picked for a lint-clean drawing
RANDOM_SEED = 71

# Seed of the input scatter in each figure, picked for a lint-clean drawing
SCATTER = {
    "layout_circular": 32,
    "layout_shell": 296,
    "layout_spiral": 32,
    "layout_bipartite": 20,
    "layout_multipartite": 37,
    "layout_bfs": 82,
    "layout_spectral": 146,
    "layout_spring": 101,
    "layout_kamada_kawai": 146,
    "layout_forceatlas2": 307,
    "layout_arf": 107,
}


def _graph(build: str):
    """Run `g.<build>` on an empty edgecase graph."""
    import edgecase as ec

    return eval(f"g.{build}", {"g": ec.Graph()})


def _positions(g, call: str) -> dict[str, tuple[float, float]]:
    """Run `g.<call>` (a `layout_*()` call) and return its positions, y flipped."""
    eval(f"g.{call}", {"g": g})
    attr = "pos"
    return {str(nid): (float(x), -float(y)) for nid, (x, y) in g.get_node_attrs(attr).items()}


def _edges(g) -> list[tuple[str, str]]:
    return [(str(a), str(b)) for _, a, b in g.get_edges_as_tuples()]


def _group_styles(g, attr: str) -> dict[str, str]:
    """Node id -> group style, groups in sorted value order."""
    values = g.get_node_attrs(attr)
    order = sorted(set(values.values()))
    return {str(nid): GROUP_STYLES[order.index(v)] for nid, v in values.items()}


def _colored(edges, styles) -> list:
    return [(a, b, {"style": styles[a]}) if styles.get(a) and styles.get(a) == styles.get(b)
            else (a, b) for a, b in edges]


def _before_after(s: Scene, method: str, g, call: str, *, styles=None,
                  before: str | None = None, guide=None) -> None:
    """Draw `g` scattered (or at `before`, a layout call) on the left and after `call` on the right.

    `guide` is a polyline in the right panel's layout coords (y flipped), drawn faintly
    under the graph.
    """
    styles = styles or {}
    edges = _edges(g)
    drawn = _colored(edges, styles)
    left, right = transform(s, "g", f"g.{call}")
    if before:
        start = _positions(g, before)
    else:
        # stretched to the panel's 2:1 shape, so the scatter isn't crammed into a square
        scatter = _positions(g, f"layout_random(seed={SCATTER[method]})")
        start = {k: (2 * x, y) for k, (x, y) in scatter.items()}
    end = _positions(g, call)
    for box, coords, prefix in ((left, start, "a."), (right, end, "b.")):
        extra = {f"~{i}": xy for i, xy in enumerate(guide or [])} if prefix == "b." else {}
        pos = place({**coords, **extra}, box, unit=UNIT, r=R_SMALL)
        if extra:
            pts = [pos.pop(k) for k in extra]
            s.path("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts), opacity=MUTED,
                   width=1)
        s.graph(pos, drawn, prefix=prefix, r=R_SMALL, styles=styles)


def _spiral_guide(g, resolution: float) -> list[tuple[float, float]]:
    """The spiral that `layout_spiral(resolution, equidistant=True)` puts the nodes on, center out.

    Fitted rather than assumed: the unscaled positions (edgecase's recurrence) are matched to the output to recover its rescaling, and `r = a * angle + b`
    is fitted to the nodes' polar coordinates, then traced in from the last node to `r = 0`.
    """
    out = g.layout_spiral(resolution, equidistant=True, return_dict=True)
    raw, theta = [], resolution + 1 / (0.5 * resolution)
    for _ in out:
        r = 0.5 * theta
        theta += 1 / r
        raw.append((math.cos(theta) * r, math.sin(theta) * r))
    got = [out[nid] for nid in sorted(out)]
    n = len(raw)
    rx, ry = sum(x for x, _ in raw) / n, sum(y for _, y in raw) / n
    ox, oy = sum(x for x, _ in got) / n, sum(y for _, y in got) / n
    k = (sum((a - rx) * (c - ox) + (b - ry) * (d - oy) for (a, b), (c, d) in zip(raw, got))
         / sum((a - rx) ** 2 + (b - ry) ** 2 for a, b in raw))
    tx, ty = ox - k * rx, oy - k * ry
    assert all(math.hypot(k * a + tx - c, k * b + ty - d) < 1e-9
               for (a, b), (c, d) in zip(raw, got)), "spiral rescaling isn't a plain zoom"

    angles = [math.atan2(b, a) for a, b in raw]
    for i in range(1, n):  # unwrap
        while angles[i] < angles[i - 1]:
            angles[i] += 2 * math.pi
    radii = [math.hypot(a, b) for a, b in raw]
    ma, mr = sum(angles) / n, sum(radii) / n
    slope = (sum((t - ma) * (r - mr) for t, r in zip(angles, radii))
             / sum((t - ma) ** 2 for t in angles))
    start = ma - mr / slope  # angle where the fitted radius reaches 0
    pts = []
    for i in range(121):
        t = start + (angles[-1] - start) * i / 120
        r = slope * (t - ma) + mr
        pts.append((k * r * math.cos(t) + tx, -(k * r * math.sin(t) + ty)))
    return pts


@figure("layout_circular", "layout", "Circular layout",
        "On the left, an eight-node graph scattered at random: a ring of eight nodes with two "
        "chords. On the right, layout_circular() puts the nodes evenly around a circle in id "
        "order, so the ring becomes a regular octagon and the two chords cross its middle.")
def layout_circular(s: Scene) -> None:
    g = _graph("add_cycle_graph(8).add_edges([(1, 5), (3, 7)])")
    _before_after(s, "layout_circular", g, "layout_circular()")


@figure("layout_shell", "layout", "Shell layout",
        "On the left, a ten-node graph scattered at random, its nodes colored by their shell "
        "attribute: a hub in the accent color, its three neighbors in teal and six outer nodes "
        "in violet, two on each neighbor. On the right, layout_shell(nlist='shell') puts the "
        "hub at the center, its neighbors on a small circle around it and the outer nodes on a "
        "larger circle outside.")
def layout_shell(s: Scene) -> None:
    g = _graph("add_star_graph(4).add_nodes(['5', '6', '7', '8', '9', '10'])")
    g.add_edges([("2", "5"), ("2", "10"), ("3", "6"), ("3", "7"), ("4", "8"), ("4", "9")])
    g.set_node_attrs("shell", {n: 0 if n == 1 else 1 if n <= 4 else 2 for n in range(1, 11)})
    _before_after(s, "layout_shell", g, 'layout_shell(nlist="shell")',
                  styles=_group_styles(g, "shell"))


@figure("layout_random", "layout", "Random layout",
        "On the left, a ring of eight nodes with two chords, drawn on a circle. On the right, "
        f"layout_random(seed={RANDOM_SEED}) drops each node at a uniformly random point in a square, "
        "ignoring the edges, so the ring is scrambled.")
def layout_random(s: Scene) -> None:
    g = _graph("add_cycle_graph(8).add_edges([(1, 5), (3, 7)])")
    _before_after(s, "layout_random", g, f"layout_random(seed={RANDOM_SEED})",
                  before="layout_circular()")


@figure("layout_spiral", "layout", "Spiral layout",
        "On the left, a path of nine nodes scattered at random. On the right, "
        "layout_spiral(1.4, equidistant=True) sets the nodes evenly spaced along a spiral in id "
        "order, so the path curls around half a turn; a faint line traces the spiral on in to "
        "its center.")
def layout_spiral(s: Scene) -> None:
    g = _graph("add_path_graph(9)")
    _before_after(s, "layout_spiral", g, "layout_spiral(1.4, equidistant=True)",
                  guide=_spiral_guide(g, 1.4))


@figure("layout_bipartite", "layout", "Bipartite layout",
        "On the left, a bipartite graph of seven nodes scattered at random, nodes 1 to 3 in the "
        "accent color and nodes 4 to 7 in teal. On the right, layout_bipartite(nodes=[1, 2, 3]) "
        "puts nodes 1 to 3 in a left column and the other four in a right column, so every edge "
        "runs across between the columns.")
def layout_bipartite(s: Scene) -> None:
    g = _graph("add_nodes(['1', '2', '3', '4', '5', '6', '7'])")
    g.add_edges([("1", "4"), ("1", "5"), ("2", "5"), ("2", "6"), ("3", "6"), ("3", "7"),
                 ("1", "7")])
    g.set_node_attrs("side", {n: int(n > 3) for n in range(1, 8)})
    _before_after(s, "layout_bipartite", g, "layout_bipartite(nodes=[1, 2, 3])",
                  styles=_group_styles(g, "side"))


@figure("layout_multipartite", "layout", "Multipartite layout",
        "On the left, a seven-node graph in three layers scattered at random, the layers "
        "colored accent, teal and violet. On the right, layout_multipartite('layer') puts "
        "each layer in its own column, left to right, so the edges step from one column to "
        "the next.")
def layout_multipartite(s: Scene) -> None:
    g = _graph("add_nodes(['1', '2', '3', '4', '5', '6', '7'])")
    g.add_edges([("1", "3"), ("1", "4"), ("2", "4"), ("2", "5"), ("3", "6"), ("4", "6"),
                 ("4", "7"), ("5", "7")])
    g.set_node_attrs("layer", {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2})
    _before_after(s, "layout_multipartite", g, 'layout_multipartite("layer")',
                  styles=_group_styles(g, "layer"))


@figure("layout_bfs", "layout", "BFS layout",
        "On the left, an eight-node graph scattered at random, node 1 outlined as the start. "
        "On the right, layout_bfs(start=1) puts node 1 in the leftmost column and every other "
        "node in a column by its distance from node 1, so the graph reads as layers of a "
        "breadth-first search.")
def layout_bfs(s: Scene) -> None:
    g = _graph("add_nodes(['1', '2', '3', '4', '5', '6', '7', '8'])")
    g.add_edges([("1", "2"), ("1", "3"), ("1", "4"), ("2", "5"), ("3", "5"), ("3", "6"),
                 ("4", "6"), ("5", "7"), ("6", "8"), ("7", "8")])
    _before_after(s, "layout_bfs", g, "layout_bfs(start=1)", styles={"1": "soft"})


@figure("layout_spectral", "layout", "Spectral layout",
        "On the left, a three-by-three grid graph scattered at random. On the right, "
        "layout_spectral() places the nodes by two eigenvectors of the graph's Laplacian, which "
        "recovers the grid: nine nodes in three rows of three, joined to their neighbors.")
def layout_spectral(s: Scene) -> None:
    g = _graph("add_grid_graph([3, 3])")
    _before_after(s, "layout_spectral", g, "layout_spectral()")


@figure("layout_spring", "layout", "Spring layout",
        "On the left, the cube graph (eight nodes, each joined to three others) scattered at "
        "random. On the right, layout_spring(seed=1) treats edges as springs and nodes as "
        "repelling charges, and the cube settles into one square inside another.")
def layout_spring(s: Scene) -> None:
    g = _graph("add_prism_graph(4)")
    _before_after(s, "layout_spring", g, "layout_spring(seed=1)")


@figure("layout_kamada_kawai", "layout", "Kamada-Kawai layout",
        "On the left, a three-by-three grid graph scattered at random. On the right, "
        "layout_kamada_kawai() sets the distance between every two nodes to match their "
        "number of hops, which straightens the grid into three rows of three.")
def layout_kamada_kawai(s: Scene) -> None:
    g = _graph("add_grid_graph([3, 3])")
    _before_after(s, "layout_kamada_kawai", g, "layout_kamada_kawai()")


@figure("layout_forceatlas2", "layout", "ForceAtlas2 layout",
        "On the left, a ring of three triangles, each joined to the next by one edge, scattered "
        "at random. On the right, layout_forceatlas2(seed=1) pulls each triangle together and "
        "pushes the triangles apart, so they sit as three separate clusters linked in a ring.")
def layout_forceatlas2(s: Scene) -> None:
    g = _graph("add_ring_of_cliques(3, 3)")
    _before_after(s, "layout_forceatlas2", g, "layout_forceatlas2(seed=1)")


@figure("layout_arf", "layout", "ARF layout",
        "On the left, a barbell graph scattered at random: two triangles, nodes 1 to 3 and 5 to "
        "7, joined through node 4. On the right, layout_arf(seed=19) pulls joined nodes together "
        "and pushes every pair apart, so the two triangles sit apart with node 4 bridging them.")
def layout_arf(s: Scene) -> None:
    g = _graph("add_barbell_graph(3, 1)")
    _before_after(s, "layout_arf", g, "layout_arf(seed=19)")
