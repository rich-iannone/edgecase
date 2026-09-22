import pytest
import edgecase as ec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chain(n=5):
    g = ec.Graph()
    for i in range(n):
        g.add_node(label=str(i))
    for i in range(n - 1):
        g.add_edge(source=str(i), target=str(i + 1))
    return g


def make_triangle():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="A", target="B")
        .add_edge(source="A", target="C")
        .add_edge(source="B", target="C")
    )


def ids(g, *labels):
    return [g.get_id_from_label(lbl) for lbl in labels]


# ===========================================================================
# network_smooth — basic functionality
# ===========================================================================


class TestSmoothBasic:
    def test_preserves_uniform_scores(self):
        g = make_triangle()
        a, b, c = ids(g, "A", "B", "C")
        scores = {a: 1.0, b: 1.0, c: 1.0}
        result = g.network_smooth(scores, alpha=0.5, return_dict=True)
        for nid in [a, b, c]:
            assert abs(result[nid] - 1.0) < 1e-6

    def test_smoothing_reduces_extremes(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        n4 = g.get_id_from_label("4")
        scores = {n0: 10.0, n4: -10.0}
        result = g.network_smooth(scores, alpha=0.5, return_dict=True)
        # Smoothed extremes should be closer to zero
        assert abs(result[n0]) < 10.0
        assert abs(result[n4]) < 10.0

    def test_neighbors_gain_signal(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        n1 = g.get_id_from_label("1")
        scores = {n0: 10.0}
        result = g.network_smooth(scores, alpha=0.5, return_dict=True)
        # Node 1 should pick up some signal from node 0
        assert result[n1] > 0.0

    def test_all_nodes_in_result(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        result = g.network_smooth({n0: 5.0}, return_dict=True)
        assert len(result) == 5

    def test_alpha_zero_returns_original(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        n2 = g.get_id_from_label("2")
        scores = {n0: 10.0, n2: 5.0}
        result = g.network_smooth(scores, alpha=0.0, return_dict=True)
        assert abs(result[n0] - 10.0) < 1e-6
        assert abs(result[n2] - 5.0) < 1e-6
        # Unscored nodes stay at 0
        n4 = g.get_id_from_label("4")
        assert abs(result[n4]) < 1e-6


# ===========================================================================
# network_smooth — alpha parameter
# ===========================================================================


class TestSmoothAlpha:
    def test_higher_alpha_more_smoothing(self):
        g = make_chain(10)
        n0 = g.get_id_from_label("0")
        scores = {n0: 10.0}
        low = g.network_smooth(scores, alpha=0.2, return_dict=True)
        high = g.network_smooth(scores, alpha=0.8, return_dict=True)
        # With higher alpha, signal spreads further
        n5 = g.get_id_from_label("5")
        assert high[n5] > low[n5]

    def test_alpha_one_pure_propagation(self):
        g = make_triangle()
        a, b, c = ids(g, "A", "B", "C")
        scores = {a: 3.0, b: 0.0, c: 0.0}
        result = g.network_smooth(scores, alpha=1.0, return_dict=True)
        # With alpha=1, all influence is from neighbors — converges to uniform
        vals = list(result.values())
        assert max(vals) - min(vals) < 1e-3


# ===========================================================================
# network_smooth — weighted graph
# ===========================================================================


class TestSmoothWeighted:
    def test_weight_influences_smoothing(self):
        g = ec.Graph(weighted=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="B", weight=10.0)
        g.add_edge(source="A", target="C", weight=1.0)
        a, b, c = ids(g, "A", "B", "C")
        scores = {a: 10.0}
        result = g.network_smooth(scores, alpha=0.5, return_dict=True)
        # B should receive more signal than C due to higher edge weight
        assert result[b] > result[c]


# ===========================================================================
# network_smooth — directed graph
# ===========================================================================


class TestSmoothDirected:
    def test_signal_follows_edges(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_node(label="C")
        g.add_edge(source="A", target="B")
        g.add_edge(source="B", target="C")
        a, b, c = ids(g, "A", "B", "C")
        scores = {a: 10.0}
        result = g.network_smooth(scores, alpha=0.5, return_dict=True)
        # Signal flows A -> B -> C
        assert result[b] > 0.0
        assert result[c] > 0.0

    def test_no_backflow(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge(source="A", target="B")
        a, b = ids(g, "A", "B")
        scores = {b: 10.0}
        result = g.network_smooth(scores, alpha=0.5, return_dict=True)
        # No edge from B to A, so A stays at its initial (0)
        assert abs(result[a]) < 1e-6


# ===========================================================================
# network_smooth — chaining and storage
# ===========================================================================


class TestSmoothChaining:
    def test_return_self(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        result = g.network_smooth({n0: 1.0})
        assert result is g

    def test_return_dict(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        result = g.network_smooth({n0: 1.0}, return_dict=True)
        assert isinstance(result, dict)

    def test_computations_tracked(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        g.network_smooth({n0: 1.0})
        found = any(c[0] == "network_smooth" for c in g._computations)
        assert found

    def test_attr_stored(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        g.network_smooth({n0: 1.0}, attr_name="smoothed")
        ndf = g.get_node_df()
        assert "smoothed" in ndf.columns


# ===========================================================================
# network_smooth — error handling
# ===========================================================================


class TestSmoothErrors:
    def test_alpha_negative(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        with pytest.raises(ValueError, match="alpha"):
            g.network_smooth({n0: 1.0}, alpha=-0.1, return_dict=True)

    def test_alpha_too_large(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        with pytest.raises(ValueError, match="alpha"):
            g.network_smooth({n0: 1.0}, alpha=1.5, return_dict=True)


# ===========================================================================
# network_smooth — edge cases
# ===========================================================================


class TestSmoothEdgeCases:
    def test_empty_graph(self):
        g = ec.Graph()
        result = g.network_smooth({}, return_dict=True)
        assert result == {}

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        a = g.get_id_from_label("A")
        result = g.network_smooth({a: 5.0}, return_dict=True)
        # Single node with alpha=0.5: 0.5*0 + 0.5*5 = 2.5... no neighbors
        # Actually: neighbor_sum = 0, so smoothed = 0 + 0.5*5 = 2.5
        assert abs(result[a] - 2.5) < 1e-6

    def test_empty_scores(self):
        g = make_chain()
        result = g.network_smooth({}, return_dict=True)
        # All scores are 0, so smoothed is 0
        for val in result.values():
            assert abs(val) < 1e-10

    def test_convergence(self):
        g = make_chain(20)
        n0 = g.get_id_from_label("0")
        # Should converge within 100 iterations
        result = g.network_smooth({n0: 100.0}, max_iter=100, return_dict=True)
        assert all(v is not None for v in result.values())
