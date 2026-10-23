"""Tests for is_planar(), check_planarity() and layout_planar(), checked against NetworkX."""

import math
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_MUTATE


def from_nx(G: nx.Graph) -> ec.Graph:
    """An edgecase graph with nodes 1..n for NetworkX's nodes 0..n-1 (in order)."""
    nodes = list(G)
    index = {v: i + 1 for i, v in enumerate(nodes)}
    g = ec.Graph().add_nodes([str(i + 1) for i in range(len(nodes))])
    g.add_edges([(str(index[u]), str(index[v])) for u, v in G.edges()])
    return g


def to_nx(g: ec.Graph) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_nodes_from(g.get_node_ids(limit=None))
    G.add_edges_from((s, t) for _, s, t in g.get_edges_as_tuples())
    return G


def triangulation(n: int, rng: random.Random) -> nx.Graph:
    """A random maximal planar graph: keep splitting a random triangle with a new node."""
    G = nx.Graph([(0, 1), (1, 2), (0, 2)])
    faces = [(0, 1, 2)]
    for v in range(3, n):
        a, b, c = faces.pop(rng.randrange(len(faces)))
        G.add_edges_from([(v, a), (v, b), (v, c)])
        faces += [(a, b, v), (b, c, v), (a, c, v)]
    return G


def random_graphs(count: int, seed: int):
    """A mix of sparse random graphs (mostly planar) and dense ones (mostly not)."""
    rng = random.Random(seed)
    for i in range(count):
        n = rng.randint(0, 13)
        if i % 3 == 0 and n >= 4:
            G = triangulation(n, rng)
            drop = rng.sample(sorted(G.edges), rng.randint(0, G.number_of_edges() // 2))
            G.remove_edges_from(drop)
        else:
            G = nx.gnp_random_graph(n, rng.choice([0.15, 0.3, 0.45, 0.6]), seed=rng.randrange(2**31))
        yield G


def faces(embedding: dict) -> int:
    """Count the faces of a rotation system by walking every half-edge once."""
    seen = set()
    count = 0
    for v, nbrs in embedding.items():
        for w in nbrs:
            if (v, w) in seen:
                continue
            count += 1
            a, b = v, w
            while (a, b) not in seen:
                seen.add((a, b))
                # next half-edge on the face: from b, the neighbor just before a, clockwise
                around = embedding[b]
                a, b = b, around[(around.index(a) - 1) % len(around)]
    return count


def crossing(p, q, r, s) -> bool:
    """Whether segments pq and rs cross at a point inside both."""

    def side(a, b, c):
        d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (d > 1e-12) - (d < -1e-12)

    return side(p, q, r) * side(p, q, s) < 0 and side(r, s, p) * side(r, s, q) < 0


def touches(p, q, r) -> bool:
    """Whether point r lies on segment pq, other than at its ends."""
    d = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    inside = min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(
        p[1], q[1]
    )
    return abs(d) < 1e-12 and inside and r not in (p, q)


def assert_crossing_free(g: ec.Graph, pos: dict) -> None:
    edges = sorted({(min(s, t), max(s, t)) for _, s, t in g.get_edges_as_tuples() if s != t})
    assert len(set(pos.values())) == len(pos), "two nodes share a position"
    for i, (a, b) in enumerate(edges):
        for c in pos:
            assert c in (a, b) or not touches(pos[a], pos[b], pos[c]), (a, b, c)
        for c, d in edges[i + 1 :]:
            if len({a, b, c, d}) == 4:
                assert not crossing(pos[a], pos[b], pos[c], pos[d]), ((a, b), (c, d))


# =============================================================================
# is_planar
# =============================================================================


class TestIsPlanar:
    def test_matches_networkx_on_random_graphs(self):
        for G in random_graphs(600, seed=1):
            assert from_nx(G).is_planar() == nx.is_planar(G)

    @pytest.mark.parametrize(
        "make, expected",
        [
            (lambda: ec.Graph().add_complete_bipartite_graph(3, 3), False),
            (lambda: ec.Graph().add_complete_bipartite_graph(2, 40), True),
            (lambda: ec.Graph().add_named_graph("petersen"), False),
            (lambda: ec.Graph().add_named_graph("dodecahedral"), True),
            (lambda: ec.Graph().add_named_graph("icosahedral"), True),
            (lambda: ec.Graph().add_full_graph(n=4), True),
            (lambda: ec.Graph().add_full_graph(n=5), False),
        ],
    )
    def test_classic_graphs(self, make, expected):
        assert make().is_planar() is expected

    def test_passes_edge_bound_yet_not_planar(self):
        # A subdivided K5 keeps well inside e <= 3n - 6 (and has no triangles)
        G = nx.complete_graph(5)
        H = nx.Graph()
        for u, v in G.edges():
            H.add_edges_from([(u, (u, v)), ((u, v), v)])
        g = from_nx(H)
        assert g.count_edges() <= 2 * g.count_nodes() - 4
        assert not g.is_planar()

    def test_ignores_parallel_edges_and_self_loops(self):
        g = ec.Graph().add_cycle_graph(3)
        g.add_edge(1, 2).add_edge(2, 3).add_edge(1, 1)
        assert g.is_planar()
        k4 = ec.Graph().add_full_graph(n=4)
        for s, t in [(1, 2), (2, 1), (3, 4), (4, 4)]:
            k4.add_edge(s, t)
        assert k4.is_planar()

    def test_ignores_direction(self):
        g = ec.Graph(directed=True).add_full_graph(n=5)
        assert not g.is_planar()
        d = ec.Graph(directed=True).add_full_graph(n=4)
        assert d.is_planar()

    def test_disconnected(self):
        g = ec.Graph().add_full_graph(n=4).add_full_graph(n=4)
        assert g.is_planar()
        g.add_full_graph(n=5)
        assert not g.is_planar()

    def test_after_removing_nodes(self):
        g = ec.Graph().add_full_graph(n=5)
        g.delete_node(3)
        assert g.is_planar()


# =============================================================================
# check_planarity
# =============================================================================


class TestCheckPlanarity:
    def test_matches_networkx_and_embedding_is_valid(self):
        for G in random_graphs(400, seed=2):
            g = from_nx(G)
            planar, embedding = g.check_planarity()
            assert planar == nx.check_planarity(G)[0]
            if not planar:
                assert embedding is None
                continue
            assert set(embedding) == set(g.get_node_ids(limit=None))
            simple = nx.Graph(to_nx(g))
            for v, nbrs in embedding.items():
                assert sorted(nbrs) == sorted(simple[v])
            # Euler: V - E + F = 2 for each component with edges, each with its own outer face
            parts = [c for c in nx.connected_components(simple) if len(c) > 1]
            v = sum(len(c) for c in parts)
            assert faces(embedding) == simple.number_of_edges() - v + 2 * len(parts)

    def test_embedding_order_of_a_wheel(self):
        planar, embedding = ec.Graph().add_wheel_graph(6).check_planarity()
        assert planar
        rim = embedding[1]
        # the hub's neighbors run around the rim, in one direction or the other
        start = rim.index(2)
        turned = rim[start:] + rim[:start]
        assert turned in ([2, 3, 4, 5, 6], [2, 6, 5, 4, 3])

    def test_empty_and_isolated(self):
        assert ec.Graph().check_planarity() == (True, {})
        g = ec.Graph().add_nodes(["a", "b"])
        assert g.check_planarity() == (True, {1: [], 2: []})

    def test_embedding_ignores_parallel_edges_and_loops(self):
        g = ec.Graph().add_cycle_graph(4)
        g.add_edge(1, 2).add_edge(3, 3)
        planar, embedding = g.check_planarity()
        assert planar
        assert {v: sorted(n) for v, n in embedding.items()} == {
            1: [2, 4],
            2: [1, 3],
            3: [2, 4],
            4: [1, 3],
        }

    def test_counterexample_off_by_default(self):
        assert ec.Graph().add_full_graph(n=5).check_planarity() == (False, None)

    def test_counterexample_is_minimal_and_non_planar(self):
        for G in random_graphs(300, seed=3):
            if nx.is_planar(G):
                continue
            g = from_nx(G)
            planar, sub = g.check_planarity(counterexample=True)
            assert not planar and isinstance(sub, ec.Graph)
            H = nx.Graph(to_nx(sub))
            assert H.number_of_edges() == sub.count_edges()
            assert not nx.is_planar(H)
            for e in list(H.edges):
                K = H.copy()
                K.remove_edge(*e)
                assert nx.is_planar(K)
            # a subdivision of K5 or K3,3: branch nodes of degree 4 or 3, the rest degree 2
            degrees = sorted(d for _, d in H.degree() if d != 2)
            assert degrees in ([4] * 5, [3] * 6)

    def test_counterexample_of_petersen(self):
        planar, sub = ec.Graph().add_named_graph("petersen").check_planarity(counterexample=True)
        assert not planar
        assert (sub.count_nodes(), sub.count_edges()) == (9, 12)

    def test_counterexample_keeps_ids_and_attributes(self):
        g = ec.Graph().add_full_graph(n=5)
        ids = g.get_edge_ids(limit=None)
        g.set_edge_attrs("tag", {eid: f"e{eid}" for eid in ids})
        # a parallel copy of one edge: the counterexample uses the lower ID
        g.add_edge(1, 2)
        _, sub = g.check_planarity(counterexample=True)
        assert sorted(sub.get_edge_ids(limit=None)) == sorted(ids)
        assert sub.get_edge_attrs_by_id("tag") == {eid: f"e{eid}" for eid in ids}

    def test_counterexample_ignores_direction(self):
        _, sub = ec.Graph(directed=True).add_full_graph(n=5).check_planarity(counterexample=True)
        assert sub.count_edges() == 10

    def test_does_not_modify_graph(self):
        g = ec.Graph().add_full_graph(n=5)
        before = (g.count_nodes(), g.count_edges(), len(g.steps))
        g.check_planarity(counterexample=True)
        assert (g.count_nodes(), g.count_edges(), len(g.steps)) == before


# =============================================================================
# layout_planar
# =============================================================================


class TestLayoutPlanar:
    def test_no_crossings_on_random_planar_graphs(self):
        checked = 0
        for G in random_graphs(500, seed=4):
            if not nx.is_planar(G):
                continue
            g = from_nx(G)
            assert_crossing_free(g, g.layout_planar(return_dict=True))
            checked += 1
        assert checked > 300

    @pytest.mark.parametrize("n", [20, 60, 150])
    def test_no_crossings_on_triangulations(self, n):
        G = triangulation(n, random.Random(n))
        g = from_nx(G)
        assert_crossing_free(g, g.layout_planar(scale=None, return_dict=True))

    @pytest.mark.parametrize(
        "make",
        [
            lambda: ec.Graph().add_grid_graph([6, 7]),
            lambda: ec.Graph().add_named_graph("dodecahedral"),
            lambda: ec.Graph().add_named_graph("icosahedral"),
            lambda: ec.Graph().add_wheel_graph(12),
            lambda: ec.Graph().add_balanced_tree(3, 3),
            lambda: ec.Graph().add_complete_bipartite_graph(2, 9),
            lambda: ec.Graph().add_full_graph(n=4).add_cycle_graph(5).add_nodes(["x"]),
        ],
    )
    def test_no_crossings_on_classic_graphs(self, make):
        g = make()
        assert_crossing_free(g, g.layout_planar(return_dict=True))

    def test_grid_coordinates(self):
        G = triangulation(40, random.Random(7))
        g = from_nx(G)
        pos = g.layout_planar(scale=None, return_dict=True)
        n = g.count_nodes()
        assert all(x == int(x) and y == int(y) for x, y in pos.values())
        assert min(x for x, _ in pos.values()) == 0 and min(y for _, y in pos.values()) == 0
        assert max(x for x, _ in pos.values()) <= 2 * n - 4
        assert max(y for _, y in pos.values()) <= n - 2

    def test_small_graphs_use_the_triangle(self):
        assert ec.Graph().add_path_graph(3).layout_planar(scale=None, return_dict=True) == {
            1: (0.0, 0.0),
            2: (2.0, 0.0),
            3: (1.0, 1.0),
        }
        assert ec.Graph().add_path_graph(2).layout_planar(return_dict=True) == {
            1: (-1.0, 0.0),
            2: (1.0, 0.0),
        }

    def test_matches_networkx_on_small_graphs(self):
        # with fewer than four nodes there is no embedding choice, so NetworkX agrees exactly
        for n in range(1, 4):
            g = ec.Graph().add_path_graph(n)
            theirs = nx.planar_layout(nx.path_graph(range(1, n + 1)), scale=2, center=(1, 1))
            ours = g.layout_planar(scale=2, center=(1, 1), return_dict=True)
            for k in ours:
                assert ours[k] == pytest.approx(tuple(theirs[k]))

    def test_rescaled_like_networkx(self):
        g = ec.Graph().add_named_graph("dodecahedral")
        raw = g.layout_planar(scale=None, return_dict=True)
        pos = g.layout_planar(scale=2.5, center=(10, -4), return_dict=True)
        expected = nx.rescale_layout_dict(raw, scale=2.5)
        for k in pos:
            assert pos[k] == pytest.approx((expected[k][0] + 10, expected[k][1] - 4))

    def test_not_planar_raises(self):
        with pytest.raises(ValueError, match="not planar"):
            ec.Graph().add_full_graph(n=5).layout_planar()
        with pytest.raises(ValueError, match="not planar"):
            ec.Graph().add_named_graph("petersen").layout_planar(return_dict=True)

    def test_deterministic(self):
        g = from_nx(triangulation(30, random.Random(3)))
        assert g.layout_planar(return_dict=True) == g.layout_planar(return_dict=True)

    def test_ignores_parallel_edges_loops_and_direction(self):
        g = ec.Graph(directed=True).add_wheel_graph(7)
        base = g.layout_planar(return_dict=True)
        g.add_edge(2, 1).add_edge(3, 3).add_edge(1, 2)
        assert g.layout_planar(return_dict=True) == base

    def test_stores_positions_and_records_a_step(self):
        g = ec.Graph().add_grid_graph([3, 4])
        before = len(g.steps)
        assert g.layout_planar(attr_name="where") is g
        pos = g.get_node_attrs("where")
        assert set(pos) == set(g.get_node_ids(limit=None))
        assert all(isinstance(p, list) and len(p) == 2 for p in pos.values())
        step = g.steps[-1]
        assert len(g.steps) == before + 1
        assert step.kind == KIND_MUTATE
        assert step.method == "layout_planar"
        assert step.code == "G.layout_planar(attr_name='where')"
        assert step.delta.attrs_changed == {"where": 12}

    def test_return_dict_matches_stored_and_records_nothing(self):
        g = ec.Graph().add_grid_graph([3, 4])
        before = len(g.steps)
        d = g.layout_planar(scale=3, return_dict=True)
        assert len(g.steps) == before and not g.get_node_attrs("pos")
        g.layout_planar(scale=3)
        assert {k: tuple(v) for k, v in g.get_node_attrs("pos").items()} == d
        assert g.steps[-1].code == "G.layout_planar(scale=3, attr_name='pos')"

    def test_single_node_and_empty(self):
        one = ec.Graph().add_node(label="a")
        assert one.layout_planar(center=(3, 4), return_dict=True) == {1: (3.0, 4.0)}
        assert ec.Graph().layout_planar(return_dict=True) == {}

    @pytest.mark.parametrize("scale", [0, -1, float("nan"), "big", True])
    def test_bad_scale(self, scale):
        with pytest.raises(ValueError, match="scale"):
            ec.Graph().add_cycle_graph(5).layout_planar(scale=scale)

    @pytest.mark.parametrize("center", [(1,), "ab", (0, float("inf")), None])
    def test_bad_center(self, center):
        with pytest.raises(ValueError, match="center"):
            ec.Graph().add_cycle_graph(5).layout_planar(center=center)

    def test_large_graph(self):
        g = ec.Graph().add_grid_graph([60, 60])
        pos = g.layout_planar(return_dict=True)
        assert len(pos) == 3600
        assert max(max(abs(x), abs(y)) for x, y in pos.values()) == pytest.approx(1.0)
        assert all(math.isfinite(c) for p in pos.values() for c in p)
