"""Tests for bridging centrality and Laplacian centrality."""

import pytest
import edgecase as ec


@pytest.fixture
def barbell():
    """Two triangles connected by a single bridge edge (3-4)."""
    g = ec.Graph(directed=False)
    for i in range(1, 7):
        g.add_node(label=str(i))
    g.add_edge("1", "2").add_edge("2", "3").add_edge("3", "1")
    g.add_edge("3", "4")
    g.add_edge("4", "5").add_edge("5", "6").add_edge("6", "4")
    return g


@pytest.fixture
def star():
    """Star graph: hub (1) connected to 5 leaves."""
    g = ec.Graph(directed=False)
    g.add_node(label="hub")
    for i in range(1, 6):
        g.add_node(label=str(i))
        g.add_edge("hub", str(i))
    return g


@pytest.fixture
def triangle():
    """Simple triangle: 1-2, 2-3, 3-1."""
    g = ec.Graph(directed=False)
    for i in range(1, 4):
        g.add_node(label=str(i))
    g.add_edge("1", "2").add_edge("2", "3").add_edge("3", "1")
    return g


@pytest.fixture
def path():
    """Path graph: 1-2-3-4-5."""
    g = ec.Graph(directed=False)
    for i in range(1, 6):
        g.add_node(label=str(i))
    for i in range(1, 5):
        g.add_edge(str(i), str(i + 1))
    return g


# ============================================================
# Bridging Centrality
# ============================================================


class TestBridgingCentrality:
    def test_returns_dict(self, barbell):
        result = barbell.bridging_centrality(return_dict=True)
        assert isinstance(result, dict)
        assert len(result) == 6

    def test_bridge_nodes_score_highest(self, barbell):
        result = barbell.bridging_centrality(return_dict=True)
        id3 = barbell.get_id_from_label("3")
        id4 = barbell.get_id_from_label("4")
        id1 = barbell.get_id_from_label("1")
        assert result[id3] > result[id1]
        assert result[id4] > result[id1]

    def test_normalized_max_is_one(self, barbell):
        result = barbell.bridging_centrality(normalized=True, return_dict=True)
        assert max(result.values()) == pytest.approx(1.0)

    def test_unnormalized(self, barbell):
        result = barbell.bridging_centrality(normalized=False, return_dict=True)
        assert max(result.values()) > 0.0

    def test_values_nonnegative(self, barbell):
        result = barbell.bridging_centrality(return_dict=True)
        for val in result.values():
            assert val >= -1e-10, f"Negative bridging centrality: {val}"

    def test_empty_graph(self):
        g = ec.Graph(directed=False)
        result = g.bridging_centrality(return_dict=True)
        assert result == {}

    def test_single_node(self):
        g = ec.Graph(directed=False).add_node(label="A")
        result = g.bridging_centrality(return_dict=True)
        assert len(result) == 1
        assert list(result.values())[0] == 0.0

    def test_triangle_all_equal(self, triangle):
        result = triangle.bridging_centrality(return_dict=True)
        vals = list(result.values())
        assert all(abs(v - vals[0]) < 1e-10 for v in vals)

    def test_path_inner_nodes_higher(self, path):
        result = path.bridging_centrality(return_dict=True)
        id1 = path.get_id_from_label("1")
        id3 = path.get_id_from_label("3")
        assert result[id3] >= result[id1]

    def test_returns_graph_chaining(self, barbell):
        result = barbell.bridging_centrality()
        assert isinstance(result, ec.Graph)

    def test_does_not_mutate(self, barbell):
        n = barbell.count_nodes()
        e = barbell.count_edges()
        barbell.bridging_centrality()
        assert barbell.count_nodes() == n
        assert barbell.count_edges() == e


# ============================================================
# Laplacian Centrality
# ============================================================


class TestLaplacianCentrality:
    def test_returns_dict(self, barbell):
        result = barbell.laplacian_centrality(return_dict=True)
        assert isinstance(result, dict)
        assert len(result) == 6

    def test_normalized_sums_to_one(self, barbell):
        result = barbell.laplacian_centrality(normalized=True, return_dict=True)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_unnormalized_values(self, triangle):
        """For a triangle (all degree 2): ΔE(v) = 4 + 2 + 2*(2+2) = 14."""
        result = triangle.laplacian_centrality(normalized=False, return_dict=True)
        for val in result.values():
            assert val == pytest.approx(14.0)

    def test_star_hub_highest(self, star):
        result = star.laplacian_centrality(return_dict=True)
        hub_id = star.get_id_from_label("hub")
        hub_val = result[hub_id]
        for nid, val in result.items():
            if nid != hub_id:
                assert hub_val > val

    def test_star_leaves_equal(self, star):
        result = star.laplacian_centrality(return_dict=True)
        hub_id = star.get_id_from_label("hub")
        leaf_vals = [v for nid, v in result.items() if nid != hub_id]
        assert all(abs(v - leaf_vals[0]) < 1e-10 for v in leaf_vals)

    def test_values_nonnegative(self, barbell):
        result = barbell.laplacian_centrality(return_dict=True)
        for val in result.values():
            assert val >= -1e-10

    def test_empty_graph(self):
        g = ec.Graph(directed=False)
        result = g.laplacian_centrality(return_dict=True)
        assert result == {}

    def test_single_node(self):
        g = ec.Graph(directed=False).add_node(label="A")
        result = g.laplacian_centrality(return_dict=True)
        assert len(result) == 1
        assert list(result.values())[0] == 0.0

    def test_isolated_node_zero(self):
        g = ec.Graph(directed=False)
        g.add_node(label="A").add_node(label="B").add_node(label="C")
        g.add_edge("A", "B")
        result = g.laplacian_centrality(return_dict=True)
        id_c = g.get_id_from_label("C")
        assert result[id_c] == 0.0

    def test_bridge_nodes_higher(self, barbell):
        result = barbell.laplacian_centrality(return_dict=True)
        id3 = barbell.get_id_from_label("3")
        id1 = barbell.get_id_from_label("1")
        assert result[id3] > result[id1]

    def test_returns_graph_chaining(self, barbell):
        result = barbell.laplacian_centrality()
        assert isinstance(result, ec.Graph)

    def test_does_not_mutate(self, barbell):
        n = barbell.count_nodes()
        e = barbell.count_edges()
        barbell.laplacian_centrality()
        assert barbell.count_nodes() == n
        assert barbell.count_edges() == e

    def test_unnormalized_formula(self, star):
        """Verify the closed-form: ΔE(v) = d(v)² + d(v) + 2·Σ d(u)."""
        result = star.laplacian_centrality(normalized=False, return_dict=True)
        hub_id = star.get_id_from_label("hub")
        # Hub: d=5, neighbors all have d=1
        # ΔE = 25 + 5 + 2*(1+1+1+1+1) = 40
        assert result[hub_id] == pytest.approx(40.0)
        # Leaf: d=1, neighbor (hub) has d=5
        # ΔE = 1 + 1 + 2*5 = 12
        leaf_id = star.get_id_from_label("1")
        assert result[leaf_id] == pytest.approx(12.0)
