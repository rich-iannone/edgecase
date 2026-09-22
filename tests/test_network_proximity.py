import math
import pytest
import edgecase as ec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chain(n=5):
    """A-B-C-D-E chain graph."""
    g = ec.Graph()
    labels = [chr(65 + i) for i in range(n)]
    for lbl in labels:
        g.add_node(label=lbl)
    for i in range(n - 1):
        g.add_edge(source=labels[i], target=labels[i + 1])
    return g


def ids(g, *labels):
    return [g.get_id_from_label(lbl) for lbl in labels]


def make_two_cliques():
    """Two 4-cliques connected by a bridge."""
    g = ec.Graph()
    for i in range(8):
        g.add_node(label=str(i))
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(source=str(i), target=str(j))
    for i in range(4, 8):
        for j in range(i + 1, 8):
            g.add_edge(source=str(i), target=str(j))
    g.add_edge(source="3", target="4")
    return g


# ===========================================================================
# network_proximity — closest
# ===========================================================================


class TestProximityClosest:
    def test_adjacent_nodes(self):
        g = make_chain()
        a, b = ids(g, "A", "B")
        r = g.network_proximity([a], [b], method="closest")
        assert r["proximity"] == 1.0

    def test_distant_nodes(self):
        g = make_chain()
        a, e = ids(g, "A", "E")
        r = g.network_proximity([a], [e], method="closest")
        assert r["proximity"] == 4.0

    def test_multi_set_takes_minimum(self):
        g = make_chain()
        a, b, d, e = ids(g, "A", "B", "D", "E")
        r = g.network_proximity([a, b], [d, e], method="closest")
        # Closest pair is B-D = 2 hops
        assert r["proximity"] == 2.0

    def test_same_node_in_both_sets(self):
        g = make_chain()
        a, b = ids(g, "A", "B")
        r = g.network_proximity([a], [a, b], method="closest")
        assert r["proximity"] == 0.0

    def test_reachable_pairs_count(self):
        g = make_chain()
        a, b, d, e = ids(g, "A", "B", "D", "E")
        r = g.network_proximity([a, b], [d, e], method="closest")
        assert r["reachable_pairs"] == 4.0
        assert r["total_pairs"] == 4.0

    def test_disconnected_graph(self):
        g = ec.Graph()
        g.add_node(label="A")
        g.add_node(label="B")
        a, b = ids(g, "A", "B")
        r = g.network_proximity([a], [b], method="closest")
        assert r["proximity"] == float("inf")
        assert r["reachable_pairs"] == 0.0


# ===========================================================================
# network_proximity — shortest (average)
# ===========================================================================


class TestProximityShortest:
    def test_average_distance(self):
        g = make_chain()
        a, b, d, e = ids(g, "A", "B", "D", "E")
        r = g.network_proximity([a, b], [d, e], method="shortest")
        # A-D=3, A-E=4, B-D=2, B-E=3 → avg = 12/4 = 3.0
        assert r["proximity"] == 3.0

    def test_single_pair(self):
        g = make_chain()
        a, c = ids(g, "A", "C")
        r = g.network_proximity([a], [c], method="shortest")
        assert r["proximity"] == 2.0

    def test_same_set(self):
        g = make_chain()
        a, b = ids(g, "A", "B")
        r = g.network_proximity([a, b], [a, b], method="shortest")
        # A-A=0, A-B=1, B-A=1, B-B=0 → avg = 0.5
        assert r["proximity"] == 0.5


# ===========================================================================
# network_proximity — kernel
# ===========================================================================


class TestProximityKernel:
    def test_adjacent_high_similarity(self):
        g = make_chain()
        a, b = ids(g, "A", "B")
        r = g.network_proximity([a], [b], method="kernel")
        # exp(-1) ≈ 0.368
        assert abs(r["proximity"] - math.exp(-1)) < 1e-6

    def test_distant_low_similarity(self):
        g = make_chain()
        a, e = ids(g, "A", "E")
        r = g.network_proximity([a], [e], method="kernel")
        # exp(-4) ≈ 0.018
        assert r["proximity"] < 0.02

    def test_kernel_between_zero_and_one(self):
        g = make_chain()
        a, b, d, e = ids(g, "A", "B", "D", "E")
        r = g.network_proximity([a, b], [d, e], method="kernel")
        assert 0 < r["proximity"] < 1


# ===========================================================================
# network_proximity — separation
# ===========================================================================


class TestProximitySeparation:
    def test_overlapping_sets_negative(self):
        g = make_two_cliques()
        set_a = ids(g, "0", "1", "2", "3")
        set_b = ids(g, "0", "1", "2", "3")
        r = g.network_proximity(set_a, set_b, method="separation")
        # Same sets: d_ab includes self-pairs (distance 0) while d_aa excludes them,
        # so separation is negative (the sets overlap in network space)
        assert r["separation"] < 0

    def test_separated_cliques_positive(self):
        g = make_two_cliques()
        set_a = ids(g, "0", "1", "2")
        set_b = ids(g, "5", "6", "7")
        r = g.network_proximity(set_a, set_b, method="separation")
        assert r["separation"] > 0

    def test_returns_all_keys(self):
        g = make_chain()
        a, b, d, e = ids(g, "A", "B", "D", "E")
        r = g.network_proximity([a, b], [d, e], method="separation")
        assert "separation" in r
        assert "d_ab" in r
        assert "d_aa" in r
        assert "d_bb" in r

    def test_single_node_sets(self):
        g = make_chain()
        a, e = ids(g, "A", "E")
        r = g.network_proximity([a], [e], method="separation")
        # d_aa = 0, d_bb = 0, d_ab = 4.0 → separation = 4.0
        assert r["d_aa"] == 0.0
        assert r["d_bb"] == 0.0
        assert r["separation"] == r["d_ab"]


# ===========================================================================
# network_proximity — error handling
# ===========================================================================


class TestProximityErrors:
    def test_empty_set_a(self):
        g = make_chain()
        b = ids(g, "B")[0]
        with pytest.raises(ValueError, match="non-empty"):
            g.network_proximity([], [b])

    def test_empty_set_b(self):
        g = make_chain()
        a = ids(g, "A")[0]
        with pytest.raises(ValueError, match="non-empty"):
            g.network_proximity([a], [])

    def test_invalid_node(self):
        g = make_chain()
        with pytest.raises(ValueError, match="not found"):
            g.network_proximity([99999], [ids(g, "A")[0]])

    def test_invalid_method(self):
        g = make_chain()
        a, b = ids(g, "A", "B")
        with pytest.raises(ValueError, match="Unknown method"):
            g.network_proximity([a], [b], method="invalid")


# ===========================================================================
# network_proximity — weighted graph
# ===========================================================================


class TestProximityWeighted:
    def test_weighted_closest(self):
        g = ec.Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="B", weight=1.0)
        g.add_edge(source="B", target="C", weight=5.0)
        a, c = ids(g, "A", "C")
        r = g.network_proximity([a], [c], method="closest")
        assert r["proximity"] == 6.0  # 1 + 5

    def test_weighted_shortest_path_choice(self):
        g = ec.Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="C", weight=10.0)
        g.add_edge(source="A", target="B", weight=1.0)
        g.add_edge(source="B", target="C", weight=2.0)
        a, c = ids(g, "A", "C")
        r = g.network_proximity([a], [c], method="closest")
        assert r["proximity"] == 3.0  # A->B->C = 1+2 < A->C = 10


# ===========================================================================
# network_proximity — directed graph
# ===========================================================================


class TestProximityDirected:
    def test_directed_asymmetric(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge(source="A", target="B")
        a, b = ids(g, "A", "B")
        r_ab = g.network_proximity([a], [b], method="closest")
        r_ba = g.network_proximity([b], [a], method="closest")
        assert r_ab["proximity"] == 1.0
        assert r_ba["proximity"] == float("inf")
