"""Tests for centrality measures: bridging, Laplacian, eccentricity, radiality, closeness
vitality, coreness, HITS, alpha, and leverage centrality."""

import itertools
import math
import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_METRIC, KIND_QUERY


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


# Subgraph, current-flow, group, VoteRank, trophic, reaching, subset betweenness and
# structural-hole centralities, validated against NetworkX (and brute force where NetworkX's
# group betweenness is wrong).


def _connected_cases():
    for seed in range(12):
        rnd = random.Random(seed)
        U = nx.connected_watts_strogatz_graph(14, 4, 0.3, seed=seed)
        U = nx.Graph(sorted(U.edges))
        for u, v in U.edges:
            U[u][v]["weight"] = rnd.uniform(0.5, 3.0)
        yield U, _build(U), _build(U, weighted=True)


def _random_cases(directed: bool):
    for seed in range(12):
        rnd = random.Random(seed)
        G0 = nx.gnp_random_graph(12, 0.25, seed=seed, directed=directed)
        G = (nx.DiGraph if directed else nx.Graph)()
        G.add_nodes_from(range(12))
        for u, v in G0.edges:
            G.add_edge(u, v, weight=rnd.choice([1, 2, 3]))
        yield G, _build(G), _build(G, weighted=True), rnd


def _build(G: nx.Graph, weighted: bool = False) -> ec.Graph:
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in sorted(G.nodes)])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


def _lab(g: ec.Graph, scores: dict) -> dict:
    return {int(g.get_label_from_id(k)): v for k, v in scores.items()}


def _ids(g: ec.Graph, nodes) -> list[int]:
    return [g.get_id_from_label(str(v)) for v in nodes]


def _brute_group_betweenness(G, group, weight=None, endpoints=False):
    S, total = set(group), 0.0
    for s, t in itertools.permutations(G.nodes, 2):
        if (s in S or t in S) and not endpoints:
            continue
        if not nx.has_path(G, s, t):
            continue
        paths = list(nx.all_shortest_paths(G, s, t, weight=weight))
        hits = sum(1 for p in paths if (S & set(p) if endpoints else S & set(p[1:-1])))
        total += hits / len(paths)
    return total if G.is_directed() else total / 2


class TestSpectralAndElectrical:
    def test_subgraph_centrality(self):
        for U, g, _ in _connected_cases():
            ours = _lab(g, g.subgraph_centrality(return_dict=True))
            assert ours == pytest.approx(nx.subgraph_centrality(U))
        assert sum(
            ec.karate_club().subgraph_centrality(return_dict=True).values()
        ) == pytest.approx(ec.karate_club().get_estrada_index())

    @pytest.mark.parametrize("normalized", [True, False])
    def test_current_flow_betweenness(self, normalized):
        for U, g, gw in _connected_cases():
            ours = _lab(g, g.current_flow_betweenness_centrality(normalized, return_dict=True))
            assert ours == pytest.approx(nx.current_flow_betweenness_centrality(U, normalized))
            ours = _lab(gw, gw.current_flow_betweenness_centrality(normalized, return_dict=True))
            ref = nx.current_flow_betweenness_centrality(U, normalized, weight="weight")
            assert ours == pytest.approx(ref)

    def test_current_flow_closeness(self):
        for U, g, gw in _connected_cases():
            assert _lab(g, g.current_flow_closeness_centrality(return_dict=True)) == pytest.approx(
                nx.current_flow_closeness_centrality(U)
            )
            assert _lab(
                gw, gw.current_flow_closeness_centrality(return_dict=True)
            ) == pytest.approx(nx.information_centrality(U, weight="weight"))

    def test_requirements(self):
        split = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("C", "D")])
        with pytest.raises(ValueError, match="connected"):
            split.current_flow_betweenness_centrality()
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).subgraph_centrality()


class TestGroupCentrality:
    def test_against_brute_force_and_networkx(self):
        for directed in (False, True):
            for G, g, gw, rnd in _random_cases(directed):
                grp = rnd.sample(range(12), 3)
                for endpoints in (False, True):
                    ours = g.group_betweenness_centrality(grp_ids := _ids(g, grp), False, endpoints)
                    assert ours == pytest.approx(
                        _brute_group_betweenness(G, grp, endpoints=endpoints)
                    )
                ours_w = gw.group_betweenness_centrality(grp_ids, False)
                assert ours_w == pytest.approx(_brute_group_betweenness(G, grp, weight="weight"))
                # (NetworkX's group betweenness is not compared: it deviates from the
                # definition on some graphs, connected ones included; brute force is the
                # reference above)
                assert g.group_closeness_centrality(grp_ids) == pytest.approx(
                    nx.group_closeness_centrality(G, grp)
                )
                assert gw.group_closeness_centrality(grp_ids) == pytest.approx(
                    nx.group_closeness_centrality(G, grp, weight="weight")
                )
                if directed:
                    assert g.group_degree_centrality(grp_ids, mode="out") == pytest.approx(
                        nx.group_out_degree_centrality(G, grp)
                    )
                    assert g.group_degree_centrality(grp_ids, mode="in") == pytest.approx(
                        nx.group_in_degree_centrality(G, grp)
                    )
                else:
                    assert g.group_degree_centrality(grp_ids) == pytest.approx(
                        nx.group_degree_centrality(G, grp)
                    )

    def test_examples_and_errors(self):
        g = ec.Graph().add_nodes(list("ABCDEF"))
        g.add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
        assert g.group_betweenness_centrality(["C", "D"]) == pytest.approx(2 / 3)
        assert g.group_betweenness_centrality(["A", "B"]) == 0.0
        path = ec.Graph().add_path_graph(7)
        assert path.group_closeness_centrality([2, 6]) > path.group_closeness_centrality([3, 4])
        with pytest.raises(ValueError, match="at least one"):
            g.group_degree_centrality([])
        with pytest.raises(ValueError, match="mode"):
            g.group_degree_centrality(["A"], mode="both")


class TestVoteRankAndTrophic:
    def test_voterank_matches_networkx(self):
        for directed in (False, True):
            for G, g, _, _ in _random_cases(directed):
                assert [int(g.get_label_from_id(n)) for n in g.voterank()] == nx.voterank(G)
                assert [int(g.get_label_from_id(n)) for n in g.voterank(3)] == nx.voterank(G, 3)
        assert ec.Graph().voterank() == []

    def test_trophic_matches_networkx(self):
        for G, _, _, _ in _random_cases(True):
            D = nx.DiGraph([(u, v, d) for u, v, d in G.edges(data=True) if u < v])
            D.add_nodes_from(range(12))
            g, gw = _build(D), _build(D, weighted=True)
            assert _lab(g, g.trophic_levels(return_dict=True)) == pytest.approx(
                nx.trophic_levels(D, weight=None)
            )
            assert _lab(gw, gw.trophic_levels(return_dict=True)) == pytest.approx(
                nx.trophic_levels(D)
            )
            assert gw.trophic_incoherence_parameter() == pytest.approx(
                nx.trophic_incoherence_parameter(D)
            )
            ours = {
                (int(gw.get_label_from_id(s)), int(gw.get_label_from_id(t))): d
                for (_, s, t), d in zip(gw.get_edges_as_tuples(), gw.trophic_differences().values())
            }
            assert ours == pytest.approx(nx.trophic_differences(D))

    def test_food_web_and_cannibalism(self):
        web = ec.Graph(directed=True).add_nodes(["grass", "rabbit", "mouse", "fox", "owl"])
        web.add_edges(
            [("grass", "rabbit"), ("grass", "mouse"), ("rabbit", "fox"),
             ("mouse", "fox"), ("mouse", "owl"), ("fox", "owl")]
        )  # fmt: skip
        levels = {
            web.get_label_from_id(n): v for n, v in web.trophic_levels(return_dict=True).items()
        }
        assert levels == pytest.approx({"grass": 1, "rabbit": 2, "mouse": 2, "fox": 3, "owl": 3.5})
        D = nx.DiGraph([(0, 1), (1, 2), (0, 2), (2, 2)])
        g = _build(D)
        for cannibalism in (False, True):
            assert g.trophic_incoherence_parameter(cannibalism=cannibalism) == pytest.approx(
                nx.trophic_incoherence_parameter(D, weight=None, cannibalism=cannibalism)
            )
        with pytest.raises(ValueError, match="basal"):
            ec.Graph(directed=True).add_nodes(["A", "B"]).add_edges(
                [("A", "B"), ("B", "A")]
            ).trophic_levels()


class TestReachingSubsetStructuralHoles:
    def test_local_and_global_reaching(self):
        for U, g, gw in _connected_cases():
            ref = {v: nx.local_reaching_centrality(U, v) for v in U}
            assert _lab(g, g.local_reaching_centrality(return_dict=True)) == pytest.approx(ref)
            ref_w = {v: nx.local_reaching_centrality(U, v, weight="weight") for v in U}
            assert _lab(gw, gw.local_reaching_centrality(return_dict=True)) == pytest.approx(ref_w)
            assert gw.global_reaching_centrality() == pytest.approx(
                nx.global_reaching_centrality(U, weight="weight")
            )
        for G, g, _, _ in _random_cases(True):
            if G.number_of_edges():
                assert g.global_reaching_centrality() == pytest.approx(
                    nx.global_reaching_centrality(G)
                )

    @pytest.mark.parametrize("directed", [False, True])
    def test_subset_betweenness(self, directed):
        for G, g, gw, rnd in _random_cases(directed):
            srcs, tgts = rnd.sample(range(12), 4), rnd.sample(range(12), 4)
            for normalized in (False, True):
                ours = g.betweenness_centrality_subset(
                    _ids(g, srcs), _ids(g, tgts), normalized, return_dict=True
                )
                ref = nx.betweenness_centrality_subset(G, srcs, tgts, normalized=normalized)
                assert _lab(g, ours) == pytest.approx(ref)
            ours = gw.betweenness_centrality_subset(
                _ids(gw, srcs), _ids(gw, tgts), return_dict=True
            )
            assert _lab(gw, ours) == pytest.approx(
                nx.betweenness_centrality_subset(G, srcs, tgts, weight="weight")
            )

    def test_effective_size_and_local_constraint(self):
        for U, g, gw in _connected_cases():
            assert _lab(g, g.effective_size(return_dict=True)) == pytest.approx(
                nx.effective_size(U)
            )
            # NetworkX's sparse shortcut is wrong for weights; its per-node formula is the reference
            ref_w = nx.effective_size(U, nodes=list(U), weight="weight")
            assert _lab(gw, gw.effective_size(return_dict=True)) == pytest.approx(ref_w)
            for u, v in list(U.edges)[:6]:
                assert gw.local_constraint(str(u), str(v)) == pytest.approx(
                    nx.local_constraint(U, u, v, weight="weight")
                )
        g = ec.Graph().add_nodes(["A", "B", "Z"]).add_edge("A", "B")
        assert g.effective_size(return_dict=True)[g.get_id_from_label("Z")] is None
        d = nx.gnp_random_graph(12, 0.3, seed=3, directed=True)
        gd = _build(d)
        ours = _lab(gd, gd.effective_size(return_dict=True))
        ref = nx.effective_size(d, nodes=list(d))
        for k, v in ref.items():
            if math.isnan(v):
                # NetworkX gives NaN to every node without out-edges; EdgeCase only to nodes
                # without any contacts, since in-ties count as contacts too
                assert (ours[k] is None) == (d.degree(k) == 0)
            else:
                assert ours[k] == pytest.approx(v)


class TestCentralityStepHooks:
    def test_metric_and_query_steps(self):
        g = ec.Graph().add_path_graph(5)
        g.current_flow_closeness_centrality()
        step = g.steps[-1]
        assert step.kind == KIND_METRIC and set(step.delta.metric_values) == {
            "1",
            "2",
            "3",
            "4",
            "5",
        }
        g.group_degree_centrality([2, 4])
        step = g.steps[-1]
        assert step.kind == KIND_QUERY and step.delta.nodes_selected == ["2", "4"]
        g.voterank(count=2)
        assert g.steps[-1].delta.node_groups == {"2": 0, "4": 1}
        n = len(g.steps)
        g.effective_size(return_dict=True)
        assert len(g.steps) == n
