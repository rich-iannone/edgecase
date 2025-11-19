"""Tests for centrality measure methods."""

import edgecase as ec
import pytest


class TestEccentricity:
    """Test get_eccentricity method."""

    def test_path_graph_eccentricity(self):
        """Test eccentricity on a path graph."""
        G = ec.Graph()
        G.add_path_graph(5)
        ecc = G.eccentricity(return_dict=True)

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
        ecc = G.eccentricity(return_dict=True)

        # Hub (node 1) has distance 1 to all leaves
        assert ecc["1"] == 1
        # Leaves have distance 2 to other leaves (through hub)
        assert ecc["2"] == 2
        assert ecc["3"] == 2

    def test_disconnected_graph_eccentricity(self):
        """Test eccentricity returns None for disconnected graph."""
        G = ec.Graph()
        G.add_node("A")
        G.add_node("B")
        ecc = G.eccentricity(return_dict=True)

        # Disconnected nodes should have None eccentricity
        assert ecc["A"] is None
        assert ecc["B"] is None

    def test_single_node_eccentricity(self):
        """Test eccentricity for single node."""
        G = ec.Graph()
        G.add_node("A")
        ecc = G.eccentricity(return_dict=True)

        # Single node has eccentricity 0
        assert ecc["A"] == 0


class TestRadiality:
    """Test get_radiality method."""

    def test_path_graph_radiality(self):
        """Test radiality on a path graph."""
        G = ec.Graph()
        G.add_path_graph(5)
        rad = G.radiality(return_dict=True)

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
        rad = G.radiality(return_dict=True)

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
        rad = G.radiality(return_dict=True)

        assert rad["A"] is None
        assert rad["B"] is None


class TestClosenessVitality:
    """Test get_closeness_vitality method."""

    def test_star_graph_vitality(self):
        """Test vitality on a star graph - hub should be critical."""
        G = ec.Graph()
        G.add_star_graph(4)
        vitality = G.get_closeness_vitality()

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
        vitality = G.get_closeness_vitality()

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

        vitality = G.get_closeness_vitality()

        # Bridge nodes C and D should have high vitality
        c_id = G.get_id_from_label("C")
        d_id = G.get_id_from_label("D")
        a_id = G.get_id_from_label("A")
        e_id = G.get_id_from_label("E")

        assert vitality[c_id] > vitality[a_id]
        assert vitality[d_id] > vitality[e_id]


class TestCoreness:
    """Test get_coreness method."""

    def test_complete_graph_coreness(self):
        """Test coreness on complete graph."""
        G = ec.Graph()
        G.add_full_graph(5)
        coreness = G.coreness(return_dict=True)

        # Complete graph K5: all nodes have degree 4
        # All nodes belong to 4-core
        assert all(k == 4 for k in coreness.values())

    def test_star_graph_coreness(self):
        """Test coreness on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)
        coreness = G.coreness(return_dict=True)

        # Leaves have degree 1, hub has degree 5
        # Leaves removed first (coreness 1), hub last (coreness 1)
        # Actually all nodes have coreness 1
        assert all(k == 1 for k in coreness.values())

    def test_nested_cores(self):
        """Test coreness with nested k-cores."""
        G = ec.Graph()
        # Create structure with different core levels
        # Central triangle (3-core)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "A")

        # Add nodes to make 2-core
        G.add_edge("A", "D")
        G.add_edge("D", "B")

        coreness = G.coreness(return_dict=True)

        # Triangle nodes should have higher coreness
        assert coreness["A"] >= 2
        assert coreness["B"] >= 2
        assert coreness["C"] >= 2


class TestHITSCentrality:
    """Test HITS authority and hub centrality methods."""

    def test_authority_hub_directed(self):
        """Test HITS on simple directed graph."""
        G = ec.Graph(directed=True)
        # A points to B and C (good hub)
        # B and C are authorities
        G.add_edge("A", "B")
        G.add_edge("A", "C")

        auth = G.get_authority_centrality()
        hub = G.get_hub_centrality()

        # B and C should have higher authority
        assert auth["B"] > auth["A"]
        assert auth["C"] > auth["A"]

        # A should have higher hub score
        assert hub["A"] > hub["B"]
        assert hub["A"] > hub["C"]

    def test_hits_convergence(self):
        """Test HITS converges with different parameters."""
        G = ec.Graph(directed=True)
        G.add_edge("A", "B")
        G.add_edge("A", "C")
        G.add_edge("B", "D")
        G.add_edge("C", "D")

        # Should converge with default parameters
        auth1 = G.get_authority_centrality(max_iter=100, tol=1e-6)
        auth2 = G.get_authority_centrality(max_iter=200, tol=1e-6)

        # Results should be similar
        for node in auth1:
            assert auth1[node] == pytest.approx(auth2[node], abs=1e-5)

    def test_hits_undirected_raises(self):
        """Test HITS raises error on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_edge("A", "B")

        with pytest.raises(ValueError, match="directed"):
            G.get_authority_centrality()

        with pytest.raises(ValueError, match="directed"):
            G.get_hub_centrality()

    def test_hits_mutual_reinforcement(self):
        """Test HITS captures mutual reinforcement."""
        G = ec.Graph(directed=True)
        # Classic web structure:
        # Good hubs point to good authorities
        G.add_edge("hub1", "auth1")
        G.add_edge("hub1", "auth2")
        G.add_edge("hub2", "auth1")
        G.add_edge("hub2", "auth2")

        auth = G.get_authority_centrality()
        hub = G.get_hub_centrality()

        # Authorities should have similar high scores
        assert auth["auth1"] == pytest.approx(auth["auth2"], rel=0.1)

        # Hubs should have similar high scores
        assert hub["hub1"] == pytest.approx(hub["hub2"], rel=0.1)


class TestAlphaCentrality:
    """Test get_alpha_centrality method."""

    def test_alpha_star_graph(self):
        """Test alpha centrality on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)

        alpha_cent = G.get_alpha_centrality(alpha=0.1)

        # Hub should have higher alpha centrality
        assert alpha_cent["1"] > alpha_cent["2"]
        assert alpha_cent["1"] > alpha_cent["3"]

    def test_alpha_parameter_effect(self):
        """Test effect of alpha parameter."""
        G = ec.Graph()
        G.add_path_graph(5)

        # Lower alpha emphasizes local structure
        alpha_low = G.get_alpha_centrality(alpha=0.01)
        # Higher alpha emphasizes global structure
        alpha_high = G.get_alpha_centrality(alpha=0.5)

        # Both should complete without error
        assert len(alpha_low) == 5
        assert len(alpha_high) == 5

    def test_alpha_convergence(self):
        """Test alpha centrality convergence."""
        G = ec.Graph()
        G.add_full_graph(4)

        alpha1 = G.get_alpha_centrality(alpha=0.1, max_iter=50, tol=1e-6)
        alpha2 = G.get_alpha_centrality(alpha=0.1, max_iter=100, tol=1e-6)

        # Results should be similar after convergence
        for node in alpha1:
            assert alpha1[node] == pytest.approx(alpha2[node], rel=1e-4)


class TestLeverageCentrality:
    """Test get_leverage_centrality method."""

    def test_leverage_star_graph(self):
        """Test leverage on star graph."""
        G = ec.Graph()
        G.add_star_graph(5)
        leverage = G.get_leverage_centrality()

        # Hub has higher degree than neighbors (positive leverage)
        assert leverage["1"] > 0

        # Leaves have lower degree than hub (negative leverage)
        assert leverage["2"] < 0
        assert leverage["3"] < 0

    def test_leverage_balanced_graph(self):
        """Test leverage on graph with uniform degrees."""
        G = ec.Graph()
        G.add_cycle_graph(5)
        leverage = G.get_leverage_centrality()

        # All nodes have same degree (2)
        # Leverage should be 0 or near 0
        for node in leverage:
            assert leverage[node] == pytest.approx(0.0, abs=1e-10)

    def test_leverage_mixed_structure(self):
        """Test leverage captures degree differences."""
        G = ec.Graph()
        # Create structure with varying degrees
        G.add_edge("hub", "A")
        G.add_edge("hub", "B")
        G.add_edge("hub", "C")
        G.add_edge("hub", "D")
        G.add_edge("A", "B")

        leverage = G.get_leverage_centrality()

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
        vitality = G.get_closeness_vitality()
        coreness = G.coreness(return_dict=True)
        alpha = G.get_alpha_centrality()
        leverage = G.get_leverage_centrality()

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
        vitality = G.get_closeness_vitality()
        coreness = G.coreness(return_dict=True)
        alpha = G.get_alpha_centrality()
        leverage = G.get_leverage_centrality()

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
        vitality = G.get_closeness_vitality()
        coreness = G.coreness(return_dict=True)
        alpha = G.get_alpha_centrality()
        leverage = G.get_leverage_centrality()

        # Should handle single node gracefully
        a_id = G.get_id_from_label("A")
        assert a_id in ecc
        assert a_id in rad
        assert a_id in vitality
        assert a_id in coreness
        assert a_id in alpha
        assert a_id in leverage
