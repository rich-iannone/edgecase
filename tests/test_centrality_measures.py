"""Tests for centrality measures: bridging, Laplacian, eccentricity, radiality, closeness
vitality, coreness, HITS, alpha, and leverage centrality."""

import edgecase as ec
import pytest


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


# ============================================================
# Eccentricity, Radiality, Vitality, Coreness, HITS, Alpha, Leverage
# ============================================================


def _by_label(G, scores):
    """Re-key a node-ID-keyed result dict by node label for readable assertions."""
    return {G.get_label_from_id(nid): v for nid, v in scores.items()}


class TestEccentricity:
    """Test eccentricity method."""

    def test_path_graph_eccentricity(self):
        """Test eccentricity on a path graph."""
        G = ec.Graph()
        G.add_path_graph(5)
        ecc = _by_label(G, G.eccentricity(return_dict=True))

        # Path: 1-2-3-4-5
        # Node 3 (middle) has max distance 2 to endpoints
        assert ecc["3"] == 2
        # Endpoints have max distance 4
        assert ecc["1"] == 4
        assert ecc["5"] == 4

    def test_star_graph_eccentricity(self):
        """Test eccentricity on a star graph."""
        G = ec.Graph()
        G.add_star_graph(5)
        ecc = _by_label(G, G.eccentricity(return_dict=True))

        # Hub (node 1) has distance 1 to all leaves
        assert ecc["1"] == 1
        # Leaves have distance 2 to other leaves (through hub)
        assert ecc["2"] == 2
        assert ecc["3"] == 2

    def test_disconnected_graph_eccentricity(self):
        """Test eccentricity is measured within each component of a disconnected graph."""
        G = ec.Graph()
        G.add_nodes(["A", "B", "C", "D"])
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        ecc = _by_label(G, G.eccentricity(return_dict=True))

        # Isolated A reaches nothing; B-C-D is a path of length 2
        assert ecc["A"] == 0
        assert ecc["B"] == 2
        assert ecc["C"] == 1

    def test_single_node_eccentricity(self):
        """Test eccentricity for single node."""
        G = ec.Graph()
        G.add_node("A")
        ecc = _by_label(G, G.eccentricity(return_dict=True))

        # Single node has eccentricity 0
        assert ecc["A"] == 0


class TestRadiality:
    """Test radiality method."""

    def test_path_graph_radiality(self):
        """Test radiality on a path graph."""
        G = ec.Graph()
        G.add_path_graph(5)
        rad = _by_label(G, G.radiality(return_dict=True))

        # Middle node (3) should have higher radiality
        # Formula: (diameter + 1 - eccentricity) / (n - 1)
        # diameter = 4, n = 5
        # Node 3: (4 + 1 - 2) / 4 = 0.75
        # Node 1: (4 + 1 - 4) / 4 = 0.25
        assert rad["3"] == pytest.approx(0.75)
        assert rad["1"] == pytest.approx(0.25)
        assert rad["5"] == pytest.approx(0.25)

    def test_star_graph_radiality(self):
        """Test radiality on a star graph."""
        G = ec.Graph()
        G.add_star_graph(5)
        rad = _by_label(G, G.radiality(return_dict=True))

        # Hub should have maximum radiality
        # diameter = 2, n = 5
        # Hub: (2 + 1 - 1) / 4 = 0.5
        # Leaves: (2 + 1 - 2) / 4 = 0.25
        assert rad["1"] == pytest.approx(0.5)
        assert rad["2"] == pytest.approx(0.25)

    def test_disconnected_radiality(self):
        """Test radiality returns None for disconnected graph."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        rad = _by_label(G, G.radiality(return_dict=True))

        assert rad["A"] is None
        assert rad["B"] is None


class TestClosenessVitality:
    """Test closeness_vitality method."""

    def test_star_graph_vitality(self):
        """Test vitality on a star graph - hub should be critical."""
        G = ec.Graph()
        G.add_star_graph(4)
        vitality = G.closeness_vitality(return_dict=True)

        # Hub removal disconnects all leaves (huge vitality)
        # Leaf removal only affects paths through that leaf (low vitality)
        hub_id = G.get_id_from_label("1")
        leaf_id = G.get_id_from_label("2")

        assert vitality[hub_id] > vitality[leaf_id]
        assert vitality[hub_id] > 0

    def test_path_graph_vitality(self):
        """Test vitality on a path graph."""
        G = ec.Graph()
        G.add_path_graph(5)
        vitality = G.closeness_vitality(return_dict=True)

        # Middle node (3) should have highest vitality
        # Removing it increases path lengths the most
        node_3_id = G.get_id_from_label("3")
        node_1_id = G.get_id_from_label("1")
        node_5_id = G.get_id_from_label("5")

        assert vitality[node_3_id] > vitality[node_1_id]
        assert vitality[node_3_id] > vitality[node_5_id]

    def test_bridge_node_vitality(self):
        """Test vitality recognizes bridge nodes."""
        G = ec.Graph()
        # Create two triangles connected by a bridge
        # Add nodes first
        for label in ["A", "B", "C", "D", "E", "F"]:
            G.add_node(label)

        # Add edges for first triangle
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")
        G.add_edge("C", "D")  # Bridge
        # Add edges for second triangle
        G.add_edge("D", "E")
        G.add_edge("E", "F")
        G.add_edge("F", "D")

        vitality = G.closeness_vitality(return_dict=True)

        # Bridge nodes C and D should have high vitality
        c_id = G.get_id_from_label("C")
        d_id = G.get_id_from_label("D")
        a_id = G.get_id_from_label("A")
        e_id = G.get_id_from_label("E")

        assert vitality[c_id] > vitality[a_id]
        assert vitality[d_id] > vitality[e_id]


class TestCoreness:
    """Test coreness method."""

    def test_complete_graph_coreness(self):
        """Test coreness on complete graph."""
        G = ec.Graph()
        G.add_full_graph(5)
        coreness = _by_label(G, G.coreness(return_dict=True))

        # Complete graph K5: all nodes have degree 4
        # All nodes belong to 4-core
        assert all(k == 4 for k in coreness.values())

    def test_star_graph_coreness(self):
        """Test coreness on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)
        coreness = _by_label(G, G.coreness(return_dict=True))

        # Leaves have degree 1, hub has degree 5
        # Leaves removed first (coreness 1), hub last (coreness 1)
        # Actually all nodes have coreness 1
        assert all(k == 1 for k in coreness.values())

    def test_nested_cores(self):
        """Test coreness with nested k-cores."""
        G = ec.Graph()
        G.add_nodes(["A", "B", "C", "D"])
        # Create structure with different core levels
        # Central triangle (3-core)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        # Add nodes to make 2-core
        G.add_edge("A", "D")
        G.add_edge("D", "B")

        coreness = _by_label(G, G.coreness(return_dict=True))

        # Triangle nodes should have higher coreness
        assert coreness["A"] >= 2
        assert coreness["B"] >= 2
        assert coreness["C"] >= 2


class TestHITSCentrality:
    """Test HITS authority and hub centrality methods."""

    def test_authority_hub_directed(self):
        """Test HITS on simple directed graph."""
        G = ec.Graph(directed=True)
        G.add_nodes(["A", "B", "C"])
        # A points to B and C (good hub)
        # B and C are authorities
        G.add_edge("A", "B")
        G.add_edge("A", "C")

        auth = _by_label(G, G.authority_centrality(return_dict=True))
        hub = _by_label(G, G.hub_centrality(return_dict=True))

        # B and C should have higher authority
        assert auth["B"] > auth["A"]
        assert auth["C"] > auth["A"]

        # A should have higher hub score
        assert hub["A"] > hub["B"]
        assert hub["A"] > hub["C"]

    def test_hits_convergence(self):
        """Test HITS converges with different parameters."""
        G = ec.Graph(directed=True)
        G.add_nodes(["A", "B", "C", "D"])
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        # Should converge with default parameters
        auth1 = _by_label(G, G.authority_centrality(max_iter=100, tol=1e-6, return_dict=True))
        auth2 = _by_label(G, G.authority_centrality(max_iter=200, tol=1e-6, return_dict=True))

        # Results should be similar
        for node in auth1:
            assert auth1[node] == pytest.approx(auth2[node], abs=1e-5)

    def test_hits_undirected_raises(self):
        """Test HITS raises error on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_nodes(["A", "B"])
        G.add_edge("A", "B")

        with pytest.raises(ValueError, match="directed"):
            G.authority_centrality(return_dict=True)

        with pytest.raises(ValueError, match="directed"):
            G.hub_centrality(return_dict=True)

    def test_hits_mutual_reinforcement(self):
        """Test HITS captures mutual reinforcement."""
        G = ec.Graph(directed=True)
        G.add_nodes(["hub1", "auth1", "auth2", "hub2"])
        # Classic web structure:
        # Good hubs point to good authorities
        G.add_edge("hub1", "auth1")
        G.add_edge("hub1", "auth2")
        G.add_edge("hub2", "auth1")
        G.add_edge("hub2", "auth2")

        auth = _by_label(G, G.authority_centrality(return_dict=True))
        hub = _by_label(G, G.hub_centrality(return_dict=True))

        # Authorities should have similar high scores
        assert auth["auth1"] == pytest.approx(auth["auth2"], rel=0.1)

        # Hubs should have similar high scores
        assert hub["hub1"] == pytest.approx(hub["hub2"], rel=0.1)


class TestAlphaCentrality:
    """Test alpha_centrality method."""

    def test_alpha_star_graph(self):
        """Test alpha centrality on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)

        alpha_cent = _by_label(G, G.alpha_centrality(alpha=0.1, return_dict=True))

        # Hub should have higher alpha centrality
        assert alpha_cent["1"] > alpha_cent["2"]
        assert alpha_cent["1"] > alpha_cent["3"]

    def test_alpha_parameter_effect(self):
        """Test effect of alpha parameter."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Lower alpha emphasizes local structure
        alpha_low = _by_label(G, G.alpha_centrality(alpha=0.01, return_dict=True))
        # Higher alpha emphasizes global structure
        alpha_high = _by_label(G, G.alpha_centrality(alpha=0.5, return_dict=True))

        # Both should complete without error
        assert len(alpha_low) == 5
        assert len(alpha_high) == 5

    def test_alpha_convergence(self):
        """Test alpha centrality convergence."""
        G = ec.Graph()
        G.add_full_graph(4)

        alpha1 = _by_label(G, G.alpha_centrality(alpha=0.1, max_iter=50, tol=1e-6, return_dict=True))
        alpha2 = _by_label(G, G.alpha_centrality(alpha=0.1, max_iter=100, tol=1e-6, return_dict=True))

        # Results should be similar after convergence
        for node in alpha1:
            assert alpha1[node] == pytest.approx(alpha2[node], rel=1e-4)


class TestLeverageCentrality:
    """Test leverage_centrality method."""

    def test_leverage_star_graph(self):
        """Test leverage on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)
        leverage = _by_label(G, G.leverage_centrality(return_dict=True))

        # Hub has higher degree than neighbors (positive leverage)
        assert leverage["1"] > 0

        # Leaves have lower degree than hub (negative leverage)
        assert leverage["2"] < 0
        assert leverage["3"] < 0

    def test_leverage_balanced_graph(self):
        """Test leverage on graph with uniform degrees."""
        G = ec.Graph()
        G.add_cycle_graph(5)
        leverage = _by_label(G, G.leverage_centrality(return_dict=True))

        # All nodes have same degree (2)
        # Leverage should be 0 or near 0
        for node in leverage:
            assert leverage[node] == pytest.approx(0.0, abs=1e-10)

    def test_leverage_mixed_structure(self):
        """Test leverage captures degree differences."""
        G = ec.Graph()
        G.add_nodes(["hub", "A", "B", "C", "D"])
        # Create structure with varying degrees
        G.add_edge("hub", "A")
        G.add_edge("hub", "B")
        G.add_edge("hub", "C")
        G.add_edge("hub", "D")
        G.add_edge("A", "B")

        leverage = _by_label(G, G.leverage_centrality(return_dict=True))

        # Hub has highest degree, should have positive leverage
        assert leverage["hub"] > 0

        # A and B connected to hub but also each other
        # Should have different leverage than C and D
        assert leverage["A"] != leverage["C"]


class TestCentralityIntegration:
    """Integration tests combining multiple centrality measures."""

    def test_multiple_measures_consistency(self):
        """Test multiple centrality measures on same graph."""
        G = ec.Graph()
        G.add_star_graph(5)

        # Get all centrality measures
        ecc = G.eccentricity(return_dict=True)
        rad = G.radiality(return_dict=True)
        vitality = G.closeness_vitality(return_dict=True)
        coreness = G.coreness(return_dict=True)
        alpha = G.alpha_centrality(return_dict=True)
        leverage = G.leverage_centrality(return_dict=True)

        # All should return same node set (node IDs)
        nodes = set(ecc.keys())
        assert set(rad.keys()) == nodes
        assert set(vitality.keys()) == nodes
        assert set(coreness.keys()) == nodes
        assert set(alpha.keys()) == nodes
        assert set(leverage.keys()) == nodes

        # Hub should be central by multiple measures
        hub_id = G.get_id_from_label("1")
        leaf_id = G.get_id_from_label("2")

        assert vitality[hub_id] > vitality[leaf_id]
        assert alpha[hub_id] > alpha[leaf_id]
        assert leverage[hub_id] > leverage[leaf_id]

    def test_centrality_on_empty_graph(self):
        """Test centrality measures on empty graph."""
        G = ec.Graph()

        ecc = G.eccentricity(return_dict=True)
        rad = G.radiality(return_dict=True)
        vitality = G.closeness_vitality(return_dict=True)
        coreness = G.coreness(return_dict=True)
        alpha = G.alpha_centrality(return_dict=True)
        leverage = G.leverage_centrality(return_dict=True)

        # All should return empty dicts
        assert len(ecc) == 0
        assert len(rad) == 0
        assert len(vitality) == 0
        assert len(coreness) == 0
        assert len(alpha) == 0
        assert len(leverage) == 0

    def test_centrality_single_node(self):
        """Test centrality measures on single node."""
        G = ec.Graph()
        G.add_node("A")

        ecc = G.eccentricity(return_dict=True)
        rad = G.radiality(return_dict=True)
        vitality = G.closeness_vitality(return_dict=True)
        coreness = G.coreness(return_dict=True)
        alpha = G.alpha_centrality(return_dict=True)
        leverage = G.leverage_centrality(return_dict=True)

        # Should handle single node gracefully
        a_id = G.get_id_from_label("A")
        assert a_id in ecc
        assert a_id in rad
        assert a_id in vitality
        assert a_id in coreness
        assert a_id in alpha
        assert a_id in leverage
