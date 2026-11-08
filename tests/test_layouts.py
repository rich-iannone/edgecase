"""Tests for the `layout_*()` methods and plotting from stored positions.

The deterministic layouts are checked coordinate for coordinate against NetworkX; the iterative
ones are checked for the properties they guarantee (reproducibility, scale and center, fixed
nodes, edges drawn shorter than random pairs), and every layout for argument validation, step
recording and `return_dict=`.
"""

import math
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_MUTATE

np = pytest.importorskip("numpy")


def to_nx(g):
    G = nx.DiGraph() if g.is_directed() else nx.Graph()
    G.add_nodes_from(sorted(g.get_node_ids(limit=None)))
    G.add_edges_from((s, t) for _, s, t in g.get_edges_as_tuples())
    return G


def max_diff(ours, theirs):
    assert set(ours) == set(theirs)
    return max(max(abs(ours[k][0] - theirs[k][0]), abs(ours[k][1] - theirs[k][1])) for k in ours)


def weighted_graph(edges):
    """A weighted graph on nodes 1..n from `(source, target, weight)` triples."""
    g = ec.Graph(weighted=True)
    for _ in range(max(max(s, t) for s, t, _ in edges)):
        g.add_node()
    for s, t, w in edges:
        g.add_edge(s, t, weight=w)
    return g


def extent(pos):
    """Largest absolute coordinate."""
    return max(max(abs(x), abs(y)) for x, y in pos.values())


def mean_edge_ratio(g, pos):
    """Mean edge length over the mean distance between random pairs of nodes."""

    def d(a, b):
        return math.dist(pos[a], pos[b])

    edges = [(s, t) for _, s, t in g.get_edges_as_tuples() if s != t]
    ids = list(pos)
    rng = random.Random(0)
    pairs = [tuple(rng.sample(ids, 2)) for _ in range(2000)]
    return (sum(d(*e) for e in edges) / len(edges)) / (sum(d(*p) for p in pairs) / len(pairs))


ALL_LAYOUTS = [
    "layout_circular",
    "layout_shell",
    "layout_random",
    "layout_spiral",
    "layout_spectral",
    "layout_spring",
    "layout_kamada_kawai",
    "layout_forceatlas2",
    "layout_arf",
]
SEEDED = {"layout_random", "layout_spring", "layout_forceatlas2", "layout_arf"}


def call(g, method, **kwargs):
    if method in SEEDED:
        kwargs.setdefault("seed", 1)
    return getattr(g, method)(**kwargs)


# =============================================================================
# NetworkX parity for the deterministic layouts
# =============================================================================


class TestNetworkXParity:
    # NetworkX computes circle angles in float32, so its coordinates are off by up to ~6e-7
    TOL = 1e-6

    def test_circular(self):
        g = ec.karate_club()
        G = to_nx(g)
        assert max_diff(g.layout_circular(return_dict=True), nx.circular_layout(G)) < self.TOL
        ours = g.layout_circular(scale=2, center=(1, -1), return_dict=True)
        assert max_diff(ours, nx.circular_layout(G, scale=2, center=(1, -1))) < self.TOL

    @pytest.mark.parametrize(
        "nlist",
        [
            None,
            [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10], list(range(11, 35))],
            [[1], list(range(2, 35))],
        ],
    )
    def test_shell(self, nlist):
        g = ec.karate_club()
        ours = g.layout_shell(nlist=nlist, scale=3, return_dict=True)
        assert max_diff(ours, nx.shell_layout(to_nx(g), nlist=nlist, scale=3)) < 3 * self.TOL

    def test_shell_rotate(self):
        g = ec.Graph().add_cycle_graph(9)
        nlist = [[1, 2, 3], [4, 5, 6, 7, 8, 9]]
        ours = g.layout_shell(nlist=nlist, rotate=0.3, return_dict=True)
        assert max_diff(ours, nx.shell_layout(to_nx(g), nlist=nlist, rotate=0.3)) < self.TOL

    @pytest.mark.parametrize("equidistant", [False, True])
    def test_spiral(self, equidistant):
        g = ec.karate_club()
        ours = g.layout_spiral(equidistant=equidistant, return_dict=True)
        theirs = nx.spiral_layout(to_nx(g), equidistant=equidistant)
        assert max_diff(ours, theirs) < self.TOL

    @pytest.mark.parametrize("align", ["vertical", "horizontal"])
    def test_bipartite(self, align):
        g = ec.Graph().add_complete_bipartite_graph(3, 4)
        ours = g.layout_bipartite(nodes=[1, 2, 3], align=align, return_dict=True)
        theirs = nx.bipartite_layout(to_nx(g), [1, 2, 3], align=align)
        assert max_diff(ours, theirs) < 1e-12

    @pytest.mark.parametrize("align", ["vertical", "horizontal"])
    def test_bfs(self, align):
        g = ec.Graph().add_balanced_tree(2, 3)
        ours = g.layout_bfs(1, align=align, return_dict=True)
        assert max_diff(ours, nx.bfs_layout(to_nx(g), 1, align=align)) < 1e-12

    def test_multipartite(self):
        g = ec.Graph().add_complete_bipartite_graph(3, 4)
        G = to_nx(g)
        nx.set_node_attributes(G, g.get_node_attrs("part"), "part")
        ours = g.layout_multipartite("part", return_dict=True)
        assert max_diff(ours, nx.multipartite_layout(G, "part")) < 1e-12

    def test_multipartite_dict(self):
        g = ec.Graph().add_path_graph(6)
        layers = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
        G = to_nx(g)
        ours = g.layout_multipartite(subset_key=layers, return_dict=True)
        assert max_diff(ours, nx.multipartite_layout(G, layers)) < 1e-12

    @pytest.mark.parametrize(
        "make",
        [
            lambda: ec.Graph().add_lollipop_graph(5, 4),
            ec.karate_club,
            lambda: ec.Graph().add_path_graph(7),
        ],
    )
    def test_spectral_up_to_sign(self, make):
        g = make()
        ours = g.layout_spectral(return_dict=True)
        theirs = nx.spectral_layout(to_nx(g))
        ids = sorted(ours)
        for axis in range(2):
            a = np.array([ours[k][axis] for k in ids])
            b = np.array([theirs[k][axis] for k in ids])
            assert min(np.abs(a - b).max(), np.abs(a + b).max()) < 1e-9

    def test_spectral_sign_is_fixed(self):
        pos = ec.Graph().add_path_graph(7).layout_spectral(return_dict=True)
        ids = sorted(pos)
        for axis in range(2):
            first = next(pos[k][axis] for k in ids if abs(pos[k][axis]) > 1e-9)
            assert first > 0

    def test_kamada_kawai_stress_matches_networkx(self):
        def stress(G, pos):
            d = dict(nx.shortest_path_length(G))
            nodes = list(G)
            total = 0.0
            for i, u in enumerate(nodes):
                for v in nodes[i + 1 :]:
                    total += (math.dist(pos[u], pos[v]) - d[u][v]) ** 2 / d[u][v] ** 2
            return total

        g = ec.Graph().add_grid_graph([5, 5])
        G = to_nx(g)
        ours = stress(G, g.layout_kamada_kawai(scale=None, return_dict=True))
        # NetworkX's result is rescaled, so compare against its unscaled solver output
        from networkx.drawing.layout import _kamada_kawai_solve

        nodes = list(G)
        dist = dict(nx.shortest_path_length(G))
        D = np.array([[dist[a][b] for b in nodes] for a in nodes], float)
        start = nx.circular_layout(G)
        raw = _kamada_kawai_solve(D, np.array([start[k] for k in nodes]), 2)
        theirs = stress(G, dict(zip(nodes, map(tuple, raw))))
        assert ours <= theirs * 1.01


# =============================================================================
# Shared behavior
# =============================================================================


class TestShared:
    @pytest.mark.parametrize("method", ALL_LAYOUTS)
    def test_stores_positions_and_chains(self, method):
        g = ec.karate_club()
        assert call(g, method) is g
        pos = g.get_node_attrs("pos")
        assert set(pos) == set(g.get_node_ids(limit=None))
        assert all(isinstance(p, list) and len(p) == 2 for p in pos.values())

    @pytest.mark.parametrize("method", ALL_LAYOUTS)
    def test_return_dict_matches_stored(self, method):
        g = ec.karate_club()
        d = call(g, method, return_dict=True)
        assert not g.get_node_attrs("pos")
        call(g, method, attr_name="xy")
        stored = g.get_node_attrs("xy")
        assert all(isinstance(v, tuple) for v in d.values())
        assert max_diff(d, {k: tuple(v) for k, v in stored.items()}) == 0

    @pytest.mark.parametrize("method", ALL_LAYOUTS)
    def test_scale_and_center(self, method):
        g = ec.karate_club()
        pos = call(g, method, scale=2.5, center=(10, -4), return_dict=True)
        shifted = {k: (x - 10, y + 4) for k, (x, y) in pos.items()}
        if method == "layout_random":
            assert extent(shifted) <= 2.5
            return
        assert extent(shifted) == pytest.approx(2.5, rel=1e-9)
        if method != "layout_shell":
            # Rescaled layouts are centered on the mean
            assert sum(x for x, _ in pos.values()) / len(pos) == pytest.approx(10)
            assert sum(y for _, y in pos.values()) / len(pos) == pytest.approx(-4)

    @pytest.mark.parametrize("method", ALL_LAYOUTS)
    def test_single_node_and_empty(self, method):
        one = ec.Graph().add_node(label="a")
        pos = call(one, method, center=(3, 4), return_dict=True)
        assert list(pos) == [1]
        assert all(math.isfinite(v) for v in pos[1])
        assert call(ec.Graph(), method, return_dict=True) == {}

    @pytest.mark.parametrize("method", ALL_LAYOUTS)
    def test_records_a_mutation_step(self, method):
        g = ec.karate_club()
        before = len(g.steps)
        call(g, method, attr_name="where")
        step = g.steps[-1]
        assert len(g.steps) == before + 1
        assert step.kind == KIND_MUTATE
        assert step.method == method
        assert step.code.startswith(f"G.{method}(")
        assert "attr_name='where'" in step.code
        assert step.delta.attrs_changed == {"where": 34}
        assert g._computations[-1][0] == method

    @pytest.mark.parametrize(
        "method, kwargs",
        [
            ("layout_bipartite", {}),
            ("layout_bfs", {"source": 1}),
            ("layout_multipartite", {"subset_key": "part"}),
        ],
    )
    def test_structured_layouts_record_steps(self, method, kwargs):
        g = ec.Graph().add_complete_bipartite_graph(2, 3)
        getattr(g, method)(**kwargs)
        assert g.steps[-1].method == method
        assert g.steps[-1].delta.attrs_changed == {"pos": 5}
        assert len(g.get_node_attrs("pos")) == 5

    @pytest.mark.parametrize("method", ALL_LAYOUTS)
    def test_return_dict_records_nothing(self, method):
        g = ec.karate_club()
        before = (len(g.steps), len(g._computations))
        call(g, method, return_dict=True)
        assert (len(g.steps), len(g._computations)) == before

    @pytest.mark.parametrize("method", ALL_LAYOUTS)
    @pytest.mark.parametrize("scale", [0, -1, float("nan"), "big", True])
    def test_bad_scale(self, method, scale):
        with pytest.raises(ValueError, match="scale"):
            call(ec.karate_club(), method, scale=scale)

    @pytest.mark.parametrize("method", ALL_LAYOUTS)
    @pytest.mark.parametrize("center", [(1,), "ab", (0, float("inf")), None])
    def test_bad_center(self, method, center):
        with pytest.raises(ValueError, match="center"):
            call(ec.karate_club(), method, center=center)

    def test_works_beyond_1000_nodes(self):
        g = ec.Graph().add_cycle_graph(1200)
        assert len(g.layout_circular(return_dict=True)) == 1200
        assert len(g.layout_spectral(return_dict=True)) == 1200

    def test_labels_accepted_where_nodes_are_named(self):
        g = ec.Graph().add_node(label="a").add_node(label="b").add_node(label="c")
        g.add_edge("a", "b").add_edge("b", "c")
        by_label = g.layout_shell(nlist=[["b"], ["a", "c"]], return_dict=True)
        by_id = g.layout_shell(nlist=[[2], [1, 3]], return_dict=True)
        assert by_label == by_id
        assert g.layout_bfs("b", return_dict=True) == g.layout_bfs(2, return_dict=True)


# =============================================================================
# Simple layouts
# =============================================================================


class TestSimpleLayouts:
    def test_random_is_reproducible_and_bounded(self):
        g = ec.Graph().add_path_graph(200)
        a = g.layout_random(seed=7, return_dict=True)
        assert a == g.layout_random(seed=7, return_dict=True)
        assert a != g.layout_random(seed=8, return_dict=True)
        assert extent(a) <= 1
        b = g.layout_random(seed=7, scale=3, center=(5, 5), return_dict=True)
        assert all(2 <= x <= 8 and 2 <= y <= 8 for x, y in b.values())

    def test_shell_by_attribute(self):
        g = ec.karate_club().coreness(attr_name="core")
        pos = g.layout_shell(nlist="core", return_dict=True)
        core = g.get_node_attrs("core")
        radius = {nid: math.hypot(*p) for nid, p in pos.items()}
        # Higher cores sit on outer shells (sorted value order, innermost first)
        for u in pos:
            for v in pos:
                if core[u] < core[v]:
                    assert radius[u] < radius[v] + 1e-12

    def test_shell_errors(self):
        g = ec.Graph().add_path_graph(4)
        with pytest.raises(ValueError, match="more than one shell"):
            g.layout_shell(nlist=[[1, 2], [2, 3, 4]])
        with pytest.raises(ValueError, match="Every node must be in a shell"):
            g.layout_shell(nlist=[[1, 2], [3]])
        with pytest.raises(ValueError, match="lack the attribute"):
            g._graph.set_node_attr(1, "grp", 0)
            g.layout_shell(nlist="grp")
        with pytest.raises(ValueError, match="rotate"):
            g.layout_shell(rotate=float("nan"))

    def test_spiral_resolution(self):
        with pytest.raises(ValueError, match="resolution"):
            ec.Graph().add_path_graph(5).layout_spiral(resolution=0)

    def test_bipartite_two_colors_by_default(self):
        g = ec.Graph().add_complete_bipartite_graph(2, 3)
        pos = g.layout_bipartite(return_dict=True)
        xs = {nid: round(p[0], 9) for nid, p in pos.items()}
        assert xs[1] == xs[2] != xs[3] == xs[4] == xs[5]

    def test_bipartite_errors(self):
        with pytest.raises(ValueError):
            ec.Graph().add_cycle_graph(5).layout_bipartite()
        with pytest.raises(ValueError, match="align"):
            ec.Graph().add_path_graph(3).layout_bipartite(align="diagonal")
        with pytest.raises(ValueError, match="aspect_ratio"):
            ec.Graph().add_path_graph(3).layout_bipartite(aspect_ratio=0)

    def test_multipartite_errors(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError, match="lack the attribute"):
            g.layout_multipartite("layer")
        with pytest.raises(TypeError):
            g.layout_multipartite(42)

    def test_bfs_layers_follow_distance(self):
        g = ec.Graph().add_balanced_tree(3, 2)
        pos = g.layout_bfs(1, return_dict=True)
        depth = {1: 0, **{n: 1 for n in (2, 3, 4)}, **{n: 2 for n in range(5, 14)}}
        xs = {nid: round(p[0], 9) for nid, p in pos.items()}
        for u in pos:
            for v in pos:
                assert (xs[u] == xs[v]) == (depth[u] == depth[v])

    def test_bfs_unreachable(self):
        g = ec.Graph().add_path_graph(3).add_node(label="island")
        with pytest.raises(ValueError):
            g.layout_bfs(1)

    def test_bfs_directed_follows_successors(self):
        g = ec.Graph(directed=True).add_path_graph(3)
        g.layout_bfs(1)
        with pytest.raises(ValueError):
            g.layout_bfs(3)


# =============================================================================
# Iterative layouts
# =============================================================================


class TestIterativeLayouts:
    @pytest.mark.parametrize(
        "method", ["layout_spring", "layout_forceatlas2", "layout_arf", "layout_kamada_kawai"]
    )
    def test_reproducible(self, method):
        g = ec.karate_club()
        assert call(g, method, return_dict=True) == call(g, method, return_dict=True)

    @pytest.mark.parametrize("method", ["layout_spring", "layout_forceatlas2", "layout_arf"])
    def test_seed_changes_layout(self, method):
        g = ec.karate_club()
        assert call(g, method, seed=1, return_dict=True) != call(
            g, method, seed=2, return_dict=True
        )

    @pytest.mark.parametrize(
        "method, limit",
        [
            ("layout_spring", 0.5),
            ("layout_kamada_kawai", 0.5),
            ("layout_forceatlas2", 0.5),
            ("layout_spectral", 0.5),
            # ARF fills a disc evenly by design; with the default `a=1.1` edges are barely
            # shorter than random pairs (as in NetworkX), so pull them in harder
            ("layout_arf", 0.7),
        ],
    )
    def test_edges_shorter_than_random_pairs(self, method, limit):
        g = ec.Graph().add_grid_graph([12, 12])
        kwargs = {"a": 2.0} if method == "layout_arf" else {}
        assert mean_edge_ratio(g, call(g, method, return_dict=True, **kwargs)) < limit

    def test_kamada_kawai_path_distances(self):
        pos = ec.Graph().add_path_graph(6).layout_kamada_kawai(scale=None, return_dict=True)
        assert math.dist(pos[1], pos[6]) == pytest.approx(5, abs=1e-3)
        assert math.dist(pos[1], pos[2]) == pytest.approx(1, abs=1e-3)

    def test_kamada_kawai_weighted_lengths(self):
        g = weighted_graph([(1, 2, 1.0), (2, 3, 4.0)])
        pos = g.layout_kamada_kawai(scale=None, return_dict=True)
        assert math.dist(pos[2], pos[3]) == pytest.approx(4, abs=1e-3)
        pos = g.layout_kamada_kawai(weighted=False, scale=None, return_dict=True)
        assert math.dist(pos[2], pos[3]) == pytest.approx(1, abs=1e-3)

    def test_kamada_kawai_disconnected(self):
        g = ec.Graph().add_path_graph(4).add_path_graph(4)
        pos = g.layout_kamada_kawai(return_dict=True)
        assert len(pos) == 8 and all(math.isfinite(c) for p in pos.values() for c in p)

    def test_kamada_kawai_rejects_zero_weight(self):
        g = weighted_graph([(1, 2, 0.0), (2, 3, 1.0)])
        with pytest.raises(ValueError):
            g.layout_kamada_kawai()

    def test_spring_fixed_nodes_stay_put(self):
        g = ec.Graph().add_lollipop_graph(6, 4).layout_circular(attr_name="start")
        start = g.get_node_attrs("start")
        pos = g.layout_spring(initial="start", fixed=[1, 3], seed=2, return_dict=True)
        for nid in (1, 3):
            assert pos[nid] == pytest.approx(tuple(start[nid]))
        assert pos[5] != pytest.approx(tuple(start[5]))

    def test_spring_fixed_needs_initial(self):
        with pytest.raises(ValueError):
            ec.Graph().add_path_graph(4).layout_spring(fixed=[1])

    def test_initial_positions_used(self):
        g = ec.Graph().add_path_graph(10)
        init = {nid: (nid, 0.0) for nid in range(1, 11)}
        a = g.layout_spring(initial=init, iterations=1, return_dict=True)
        b = g.layout_spring(initial=init, iterations=1, seed=99, return_dict=True)
        # Fully specified starting positions leave nothing to the seed
        assert a == b

    @pytest.mark.parametrize(
        "method", ["layout_spring", "layout_kamada_kawai", "layout_forceatlas2", "layout_arf"]
    )
    def test_bad_initial(self, method):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError, match="Initial position"):
            call(g, method, initial={1: (0.0, float("nan"))})
        with pytest.raises(ValueError, match="Initial position"):
            call(g, method, initial={1: "here"})
        with pytest.raises(ValueError, match="which no node has"):
            call(g, method, initial="nowhere")

    def test_spring_validation(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError):
            g.layout_spring(k=0)
        with pytest.raises(ValueError):
            g.layout_spring(threshold=-1)

    def test_forceatlas2_options(self):
        g = ec.karate_club()
        base = g.layout_forceatlas2(seed=3, return_dict=True)
        for kw in (
            dict(linlog=True),
            dict(dissuade_hubs=True),
            dict(strong_gravity=True),
            dict(node_size={1: 3.0}),
            dict(node_mass={1: 10.0}),
        ):
            pos = g.layout_forceatlas2(seed=3, return_dict=True, **kw)
            assert pos != base
            assert all(math.isfinite(c) for p in pos.values() for c in p)

    def test_forceatlas2_overlap_prevention(self):
        g = ec.Graph().add_full_graph(12)
        size = {nid: 1.0 for nid in range(1, 13)}
        pos = g.layout_forceatlas2(
            node_size=size, max_iter=300, seed=1, scale=None, return_dict=True
        )
        ids = sorted(pos)
        closest = min(math.dist(pos[u], pos[v]) for i, u in enumerate(ids) for v in ids[i + 1 :])
        assert closest > 1.5

    def test_forceatlas2_validation(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError):
            g.layout_forceatlas2(scaling_ratio=0)
        with pytest.raises(ValueError):
            g.layout_forceatlas2(jitter_tolerance=0)
        with pytest.raises(ValueError):
            g.layout_forceatlas2(gravity=-1)
        with pytest.raises(ValueError):
            g.layout_forceatlas2(node_mass={1: 0.0})
        with pytest.raises(ValueError, match="must be a number"):
            g.layout_forceatlas2(node_mass={1: "heavy"})
        with pytest.raises(ValueError):
            g.layout_forceatlas2(node_size={1: -1.0})

    def test_arf_validation(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError):
            g.layout_arf(a=1.0)
        with pytest.raises(ValueError):
            g.layout_arf(scaling=0)
        with pytest.raises(ValueError):
            g.layout_arf(dt=0)
        with pytest.raises(ValueError):
            g.layout_arf(etol=-1)

    def test_negative_weights_rejected(self):
        g = weighted_graph([(1, 2, -1.0), (2, 3, 1.0)])
        for method in ("layout_spring", "layout_forceatlas2", "layout_spectral"):
            with pytest.raises(ValueError):
                call(g, method)

    def test_weights_change_spring(self):
        g = weighted_graph([(1, 2, 10.0), (2, 3, 0.1), (3, 1, 0.1)])
        weighted = g.layout_spring(seed=1, scale=None, return_dict=True)
        plain = g.layout_spring(seed=1, scale=None, weighted=False, return_dict=True)
        assert weighted != plain
        assert math.dist(weighted[1], weighted[2]) < math.dist(weighted[2], weighted[3])

    def test_large_graph_runs_in_parallel_path(self):
        g = ec.Graph().add_newman_watts_strogatz_graph(400, 4, 0.1, seed=1)
        for method in ("layout_spring", "layout_forceatlas2", "layout_arf"):
            pos = call(g, method, return_dict=True)
            assert len(pos) == 400
            assert extent(pos) == pytest.approx(1.0)


# =============================================================================
# Plotting stored positions
# =============================================================================


class TestStoredLayoutPlotting:
    def test_plot_uses_stored_positions(self):
        g = ec.Graph().add_path_graph(3).layout_circular()
        svg = g.plot(layout="pos").svg
        assert svg.startswith("<svg")

    def test_plot_with_custom_attribute(self):
        g = ec.Graph().add_path_graph(3)
        g.layout_bfs(1, attr_name="tree")
        assert g.plot(layout="tree").svg.startswith("<svg")

    def test_y_axis_points_up(self):
        from edgecase.svg_renderer import _stored_layout

        g = ec.Graph().add_node(label="low").add_node(label="high")
        g._graph.set_node_attr(1, "pos", [0.0, 0.0])
        g._graph.set_node_attr(2, "pos", [0.0, 1.0])
        pos = _stored_layout(g, [1, 2], "pos", 600.0, 400.0)
        assert pos["2"][1] < pos["1"][1]  # SVG y grows downward

    def test_unknown_layout_message(self):
        with pytest.raises(ValueError, match="name of a node attribute"):
            ec.Graph().add_path_graph(3).plot(layout="nope")

    def test_partially_stored_positions(self):
        g = ec.Graph().add_path_graph(3)
        g._graph.set_node_attr(1, "pos", [0.0, 0.0])
        with pytest.raises(ValueError, match="2 of 3 nodes"):
            g.plot(layout="pos")

    def test_invalid_stored_values(self):
        g = ec.Graph().add_path_graph(2)
        g._graph.set_node_attr(1, "pos", [0.0, 0.0])
        g._graph.set_node_attr(2, "pos", "far away")
        with pytest.raises(ValueError):
            g.plot(layout="pos")

    def test_traversal_plot_with_stored_positions(self):
        g = ec.Graph().add_grid_graph([3, 3]).layout_spectral()
        order = g.bfs(1, return_list=True)
        result = g.plot_traversal(visit_order=order, source=1, layout="pos")
        assert "<svg" in result.svg
