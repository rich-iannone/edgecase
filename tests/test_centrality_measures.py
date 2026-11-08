"""Tests for centrality measures: bridging, Laplacian, eccentricity, radiality, closeness
vitality, coreness, HITS, alpha, and leverage centrality."""

import itertools
import math
import random

import edgecase as ec
import networkx as nx
import numpy as np
import pytest
from edgecase import Graph
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

    def test_normalized_divides_by_laplacian_energy(self):
        """Path a-b-c has energy 1 + 4 + 1 + 2 * 2 = 10; b's drop is all of it.

        The old code divided by the sum of the drops (22), giving 3/11 and 5/11.
        """
        g = ec.Graph(directed=False).add_nodes(list("abc")).add_edges([("a", "b"), ("b", "c")])
        result = _by_label(g, g.laplacian_centrality(return_dict=True))
        assert result == pytest.approx({"a": 0.6, "b": 1.0, "c": 0.6})
        raw = _by_label(g, g.laplacian_centrality(normalized=False, return_dict=True))
        assert raw == pytest.approx({"a": 6.0, "b": 10.0, "c": 6.0})

    def test_star_hub_takes_all_energy(self, star):
        result = star.laplacian_centrality(return_dict=True)
        assert result[star.get_id_from_label("hub")] == pytest.approx(1.0)

    @pytest.mark.parametrize("normalized", [True, False])
    def test_matches_networkx(self, normalized):
        for G in _random_graphs(False):
            g = _from_nx(G)
            ours = _by_int_label(g, g.laplacian_centrality(normalized=normalized, return_dict=True))
            if G.number_of_edges() == 0:
                continue
            ref = nx.laplacian_centrality(G, normalized=normalized, weight=None)
            assert ours == pytest.approx(ref, abs=1e-9)

    @pytest.mark.parametrize("normalized", [True, False])
    def test_weighted_matches_networkx(self, normalized):
        for G, g in _weighted_cases():
            if G.is_directed():
                continue
            ours = _by_int_label(g, g.laplacian_centrality(normalized=normalized, return_dict=True))
            ref = nx.laplacian_centrality(G, normalized=normalized, weight="weight")
            assert ours == pytest.approx(ref, abs=1e-9)

    def test_weighted_flag(self):
        g = ec.Graph(directed=False, weighted=True).add_nodes(list("abc"))
        g.add_edges([("a", "b", 3.0), ("b", "c", 1.0)])
        # Strengths 3, 4, 1; energy 9 + 16 + 1 + 2 * (9 + 1) = 46
        result = _by_label(g, g.laplacian_centrality(return_dict=True))
        assert result == pytest.approx({"a": 42 / 46, "b": 1.0, "c": 10 / 46})
        unweighted = _by_label(g, g.laplacian_centrality(weighted=False, return_dict=True))
        assert unweighted == pytest.approx({"a": 0.6, "b": 1.0, "c": 0.6})
        g.laplacian_centrality(weighted=False)
        assert g._computations[-1] == (
            "laplacian_centrality",
            {"normalized": True, "weighted": False},
            "laplacian_centrality",
        )

    def test_parallel_edges_sum_and_self_loops_ignored(self):
        """Matches NetworkX on the multigraph, where parallel edges add into one Laplacian entry."""
        edges = [(0, 1), (0, 1), (1, 2), (2, 2), (2, 3)]
        g = ec.Graph(directed=False).add_nodes(["0", "1", "2", "3"])
        g.add_edges([(str(u), str(v)) for u, v in edges])
        ours = _by_int_label(g, g.laplacian_centrality(return_dict=True))
        ref = nx.laplacian_centrality(nx.MultiGraph(edges), weight=None)
        assert ours == pytest.approx(ref)

    def test_directed_ignores_direction(self):
        d = ec.Graph(directed=True).add_nodes(list("abc")).add_edges([("a", "b"), ("c", "b")])
        u = ec.Graph(directed=False).add_nodes(list("abc")).add_edges([("a", "b"), ("b", "c")])
        assert _by_label(d, d.laplacian_centrality(return_dict=True)) == pytest.approx(
            _by_label(u, u.laplacian_centrality(return_dict=True))
        )

    def test_rejects_negative_weights(self):
        g = ec.Graph(directed=False, weighted=True).add_nodes(list("ab"))
        g.add_edge("a", "b", weight=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            g.laplacian_centrality()
        assert g.laplacian_centrality(weighted=False, return_dict=True)

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


class TestHITSWeighted:
    """weighted= on HITS, checked against NetworkX's hits() (which always reads weights)."""

    @staticmethod
    def _sum_to_one(G, scores):
        total = sum(scores.values())
        return {G.get_label_from_id(nid): v / total for nid, v in scores.items()}

    @pytest.mark.parametrize("seed", range(30))
    def test_matches_networkx(self, seed):
        rnd = random.Random(seed)
        n = rnd.randint(3, 10)
        nxg = nx.MultiDiGraph() if seed % 3 == 0 else nx.DiGraph()
        nxg.add_nodes_from(range(n))
        for _ in range(rnd.randint(2, 3 * n)):
            u, v = rnd.randrange(n), rnd.randrange(n)
            nxg.add_edge(u, v, weight=rnd.choice([0.5, 1.0, 2.0, 3.5]))
        # With a repeated top singular value the scores are not unique, and NetworkX's SVD
        # may return any vector in that space; power iteration from all ones picks one. A
        # near tie is unique but converges too slowly for a tight comparison
        sv = np.linalg.svd(nx.to_numpy_array(nxg, weight="weight"), compute_uv=False)
        if sv[1] > 0.95 * sv[0]:
            pytest.skip("top singular value is (nearly) repeated")
        try:
            nx_hub, nx_auth = nx.hits(nxg, max_iter=1000, tol=1e-12)
        except nx.PowerIterationFailedConvergence:
            pytest.skip("NetworkX did not converge")

        G = ec.Graph(directed=True, weighted=True)
        G.add_nodes([str(v) for v in nxg.nodes()])
        G.add_edges([(str(u), str(v), w) for u, v, w in nxg.edges(data="weight")])
        hub = self._sum_to_one(G, G.hub_centrality(1000, 1e-12, weighted=True, return_dict=True))
        auth = self._sum_to_one(
            G, G.authority_centrality(1000, 1e-12, weighted=True, return_dict=True)
        )
        for v in nxg.nodes():
            assert hub[str(v)] == pytest.approx(nx_hub[v], abs=1e-6)
            assert auth[str(v)] == pytest.approx(nx_auth[v], abs=1e-6)

    def test_weight_shifts_scores(self):
        G = ec.Graph(directed=True, weighted=True)
        G.add_nodes(["A", "B", "C"])
        G.add_edges([("A", "B", 3.0), ("A", "C", 1.0)])

        plain = _by_label(G, G.authority_centrality(weighted=False, return_dict=True))
        heavy = _by_label(G, G.authority_centrality(return_dict=True))  # follows the graph

        assert plain["B"] == pytest.approx(plain["C"])
        assert heavy["B"] == pytest.approx(3 * heavy["C"])

    def test_parallel_edges_add_up(self):
        """Two A->B edges count like one edge of weight 2, for hubs and authorities alike."""
        multi = ec.Graph(directed=True)
        multi.add_nodes(["a", "b", "c"])
        multi.add_edges([("a", "b"), ("a", "b"), ("a", "c"), ("c", "b")])
        single = ec.Graph(directed=True, weighted=True)
        single.add_nodes(["a", "b", "c"])
        single.add_edges([("a", "b", 2.0), ("a", "c", 1.0), ("c", "b", 1.0)])

        for method in ("hub_centrality", "authority_centrality"):
            m = _by_label(multi, getattr(multi, method)(return_dict=True))
            s = _by_label(single, getattr(single, method)(weighted=True, return_dict=True))
            for v in "abc":
                assert m[v] == pytest.approx(s[v], abs=1e-6)

    def test_negative_weight_raises(self):
        G = ec.Graph(directed=True, weighted=True)
        G.add_nodes(["A", "B"])
        G.add_edges([("A", "B", -1.0)])

        with pytest.raises(ValueError, match="non-negative"):
            G.hub_centrality(return_dict=True)
        # unweighted runs ignore the weight
        G.hub_centrality(weighted=False, return_dict=True)

    def test_weighted_is_tracked(self):
        G = ec.Graph(directed=True)
        G.add_nodes(["A", "B"])
        G.add_edge("A", "B")
        G.hub_centrality(weighted=True)
        assert G._computations[-1] == ("hub_centrality", {"weighted": True}, "hub_centrality")


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
        ) == pytest.approx(ec.karate_club().estrada_index())

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
                # normalized over ordered pairs: those outside the group, or all n(n - 1)
                # with endpoints (the NetworkX 3.7 scale)
                n, scale = len(G), 1 if directed else 2
                for endpoints, pairs in ((False, (n - 3) * (n - 4)), (True, n * (n - 1))):
                    ours = g.group_betweenness_centrality(grp_ids, endpoints=endpoints)
                    ref = _brute_group_betweenness(G, grp, endpoints=endpoints) * scale / pairs
                    assert ours == pytest.approx(ref)
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

    def test_trophic_without_non_basal_nodes(self):
        # Every node basal: the empty linear system used to abort the process
        for nodes in ([], ["A"], ["A", "B", "C"]):
            g = ec.Graph(directed=True).add_nodes(nodes)
            assert g.trophic_levels(return_dict=True) == {n: 1.0 for n in g.get_node_ids()}
            assert g.trophic_differences() == {}
            assert g.trophic_incoherence_parameter() == 0.0


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


# Closeness and degree centrality, validated against NetworkX on random graphs (directed and
# undirected, including disconnected ones) and on hand-computed cases.


def _random_graphs(directed: bool):
    # p is low enough that many of these graphs are disconnected
    for seed in range(12):
        yield nx.gnp_random_graph(20, 0.12, seed=seed, directed=directed)


def _strip() -> Graph:
    """The seven-node graph used by the doc figures."""
    return (
        Graph()
        .add_nodes(list("ABCDEFG"))
        .add_edges(
            [
                ("A", "B"),
                ("A", "C"),
                ("B", "C"),
                ("B", "D"),
                ("C", "D"),
                ("C", "E"),
                ("D", "E"),
                ("E", "F"),
                ("F", "G"),
            ]
        )
    )


def _from_nx(G: nx.Graph) -> Graph:
    g = Graph(directed=G.is_directed())
    g.add_nodes([str(n) for n in G.nodes])
    if G.number_of_edges():
        g.add_edges([(str(u), str(v)) for u, v in G.edges])
    return g


def _by_int_label(g: Graph, scores: dict[int, float]) -> dict[int, float]:
    return {int(g.get_label_from_id(nid)): v for nid, v in scores.items()}


TOL = 1e-9


def _weighted_cases():
    for seed in range(15):
        rnd = random.Random(seed)
        for directed in (False, True):
            G = nx.gnp_random_graph(15, 0.22, seed=seed, directed=directed)
            for u, v in G.edges:
                G[u][v]["weight"] = rnd.choice([0.5, 1, 1.5, 2, 3])
            yield G, _build_in_order(G, True)


def _triangle_detour() -> ec.Graph:
    """A-C is heavy, so the cheapest A-C route detours through B."""
    return (
        ec.Graph(weighted=True)
        .add_nodes(["A", "B", "C"])
        .add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 5.0)])
    )


def _build_in_order(G: nx.Graph, weighted: bool) -> ec.Graph:
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G.nodes])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


class TestClosenessCentrality:
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("wf_improved", [True, False])
    def test_matches_networkx(self, directed, wf_improved):
        for G in _random_graphs(directed):
            g = _from_nx(G)
            # NetworkX measures incoming distance on digraphs, i.e. mode="in"
            mode = "in" if directed else "out"
            ours = _by_int_label(
                g, g.closeness_centrality(mode=mode, wf_improved=wf_improved, return_dict=True)
            )
            ref = nx.closeness_centrality(G, wf_improved=wf_improved)
            for n, v in ref.items():
                assert ours[n] == pytest.approx(v, abs=TOL)

    def test_out_mode_matches_networkx_on_reverse(self):
        for G in _random_graphs(True):
            g = _from_nx(G)
            ours = _by_int_label(g, g.closeness_centrality(mode="out", return_dict=True))
            ref = nx.closeness_centrality(G.reverse())
            for n, v in ref.items():
                assert ours[n] == pytest.approx(v, abs=TOL)

    def test_all_mode_ignores_direction(self):
        for G in _random_graphs(True):
            g = _from_nx(G)
            ours = _by_int_label(g, g.closeness_centrality(mode="all", return_dict=True))
            ref = nx.closeness_centrality(G.to_undirected())
            for n, v in ref.items():
                assert ours[n] == pytest.approx(v, abs=TOL)

    def test_path_hand_computed(self):
        g = Graph().add_nodes(list("ABCDE"))
        g.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        c = g.closeness_centrality(return_dict=True)
        lab = {g.get_label_from_id(k): v for k, v in c.items()}
        # C: distances 2+1+1+2 = 6 -> 4/6; A: 1+2+3+4 = 10 -> 4/10
        assert lab["C"] == pytest.approx(4 / 6)
        assert lab["A"] == pytest.approx(0.4)
        assert lab["A"] == lab["E"] and lab["B"] == lab["D"]

    def test_strip_values_used_in_figure(self):
        g = _strip()
        lab = {
            g.get_label_from_id(k): round(v, 2)
            for k, v in g.closeness_centrality(return_dict=True).items()
        }
        assert lab == {"A": 0.46, "B": 0.5, "C": 0.67, "D": 0.6, "E": 0.67, "F": 0.5, "G": 0.35}

    def test_wf_correction_on_disconnected_graph(self):
        g = Graph().add_nodes(["Hub", "A", "B", "C", "X", "Y"])
        g.add_edges([("Hub", "A"), ("Hub", "B"), ("Hub", "C"), ("X", "Y")])
        corrected = g.closeness_centrality(return_dict=True)
        raw = g.closeness_centrality(wf_improved=False, return_dict=True)
        x, hub = g.get_id_from_label("X"), g.get_id_from_label("Hub")
        assert raw[x] == pytest.approx(1.0)
        assert corrected[x] == pytest.approx(0.2)  # 1 * (1 / 5)
        assert corrected[hub] > corrected[x]

    def test_empty_single_and_isolated(self):
        assert Graph().closeness_centrality(return_dict=True) == {}
        g = Graph().add_node(label="A")
        assert list(g.closeness_centrality(return_dict=True).values()) == [0.0]
        g = Graph().add_nodes(["A", "B", "C"]).add_edge("A", "B")
        assert g.closeness_centrality(return_dict=True)[g.get_id_from_label("C")] == 0.0

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            _strip().closeness_centrality(mode="sideways")

    def test_stores_attribute_and_chains(self):
        g = _strip().closeness_centrality()
        assert "closeness_centrality" in g.get_node_df().columns
        g.closeness_centrality(attr_name="close", mode="all")
        assert "close" in g.get_node_df().columns
        assert ("closeness_centrality", {"mode": "all"}, "close") in g._computations

    def test_return_dict_does_not_store(self):
        g = _strip()
        g.closeness_centrality(return_dict=True)
        assert "closeness_centrality" not in g.get_node_df().columns

    def test_steps(self):
        g = _strip().closeness_centrality()
        step = g.steps[-1]
        assert step.kind == KIND_METRIC and step.method == "closeness_centrality"
        assert step.code == "G.closeness_centrality(attr_name='closeness_centrality')"
        assert round(step.delta.metric_values["E"], 2) == 0.67
        assert set(step.delta.metric_values) == set("ABCDEFG")
        n = len(g.steps)
        g.closeness_centrality(return_dict=True)
        assert len(g.steps) == n
        assert '"mv"' in g.play()._html  # node heat

    def test_non_default_params_appear_in_code(self):
        g = Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        g.closeness_centrality(mode="in", wf_improved=False)
        assert g.steps[-1].code == (
            "G.closeness_centrality(mode='in', wf_improved=False, attr_name='closeness_centrality')"
        )


class TestDegreeCentrality:
    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        for G in _random_graphs(directed):
            g = _from_nx(G)
            ours = _by_int_label(g, g.degree_centrality(return_dict=True))
            for n, v in nx.degree_centrality(G).items():
                assert ours[n] == pytest.approx(v, abs=TOL)

    def test_in_and_out_match_networkx(self):
        for G in _random_graphs(True):
            g = _from_nx(G)
            ours_in = _by_int_label(g, g.degree_centrality(mode="in", return_dict=True))
            ours_out = _by_int_label(g, g.degree_centrality(mode="out", return_dict=True))
            for n, v in nx.in_degree_centrality(G).items():
                assert ours_in[n] == pytest.approx(v, abs=TOL)
            for n, v in nx.out_degree_centrality(G).items():
                assert ours_out[n] == pytest.approx(v, abs=TOL)

    def test_mode_ignored_for_undirected(self):
        g = _strip()
        assert g.degree_centrality(mode="in", return_dict=True) == g.degree_centrality(
            return_dict=True
        )

    def test_star(self):
        g = Graph().add_nodes(["Hub", "A", "B", "C", "D"])
        g.add_edges([("Hub", x) for x in "ABCD"])
        c = g.degree_centrality(return_dict=True)
        assert c[g.get_id_from_label("Hub")] == pytest.approx(1.0)
        assert c[g.get_id_from_label("A")] == pytest.approx(0.25)

    def test_self_loop_counts_twice_like_networkx(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 2)])
        g = _from_nx(G)
        ours = _by_int_label(g, g.degree_centrality(return_dict=True))
        for n, v in nx.degree_centrality(G).items():
            assert ours[n] == pytest.approx(v)

    def test_empty_and_single(self):
        assert Graph().degree_centrality(return_dict=True) == {}
        g = Graph().add_node(label="A")
        assert list(g.degree_centrality(return_dict=True).values()) == [1.0]

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            _strip().degree_centrality(mode="both")

    def test_stores_attribute(self):
        g = Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        g.degree_centrality(mode="in", attr_name="indeg")
        df = g.get_node_df().sort("label")
        assert df["indeg"].to_list() == [0.0, 1.0]

    def test_steps(self):
        g = _strip().degree_centrality(mode="all")
        step = g.steps[-1]
        assert step.kind == KIND_METRIC and step.method == "degree_centrality"
        assert step.delta.metric_values["C"] == pytest.approx(4 / 6)
        assert "highest: C" in step.narration
        n = len(g.steps)
        g.degree_centrality(return_dict=True)
        assert len(g.steps) == n


# Weighted betweenness, edge betweenness, closeness and harmonic centrality (weighted=None
# follows the graph, like shortest_path()), validated against NetworkX.


class TestWeightedCentralitiesAgainstNetworkX:
    def test_betweenness(self):
        for G, g in _weighted_cases():
            ours = _by_int_label(g, g.betweenness_centrality(return_dict=True))
            ref = nx.betweenness_centrality(G, weight="weight")
            assert ours == pytest.approx(ref, abs=1e-12)

    @pytest.mark.parametrize("normalized", [True, False])
    def test_edge_betweenness(self, normalized):
        for G, g in _weighted_cases():
            scores = g.edge_betweenness_centrality(normalized=normalized, return_dict=True)
            ours = {}
            for eid, value in scores.items():
                s, t = g.get_node_ids_by_edge_id(eid)
                ours[(int(g.get_label_from_id(s)), int(g.get_label_from_id(t)))] = value
            ref = nx.edge_betweenness_centrality(G, normalized=normalized, weight="weight")
            for (u, v), val in ref.items():
                got = ours.get((u, v))
                if got is None and not G.is_directed():
                    got = ours[(v, u)]
                assert got == pytest.approx(val, abs=1e-12)

    @pytest.mark.parametrize("wf_improved", [True, False])
    def test_closeness(self, wf_improved):
        for G, g in _weighted_cases():
            mode = "in" if G.is_directed() else "out"  # NetworkX uses incoming distances
            ours = _by_int_label(
                g, g.closeness_centrality(mode=mode, wf_improved=wf_improved, return_dict=True)
            )
            ref = nx.closeness_centrality(G, distance="weight", wf_improved=wf_improved)
            assert ours == pytest.approx(ref, abs=1e-12)

    def test_harmonic(self):
        for G, g in _weighted_cases():
            ours = _by_int_label(g, g.harmonic_centrality(return_dict=True))
            # NetworkX sums over incoming distances; mode="out" matches it on the reverse
            ref = nx.harmonic_centrality(G.reverse() if G.is_directed() else G, distance="weight")
            assert ours == pytest.approx(ref, abs=1e-12)

    def test_pagerank(self):
        for G, g in _weighted_cases():
            ours = _by_int_label(g, g.pagerank(tol=1e-13, max_iter=1000, return_dict=True))
            ref = nx.pagerank(G, tol=1e-15, max_iter=1000)
            assert ours == pytest.approx(ref, abs=1e-9)

    def test_load(self):
        """Distinct random weights, so no shortest paths tie (see `test_load_ties_split_evenly`)."""
        rnd = random.Random(4)
        for seed in range(20):
            G = nx.gnp_random_graph(14, 0.25, seed=seed, directed=seed % 2 == 1)
            for u, v in G.edges:
                G[u][v]["weight"] = rnd.uniform(0.1, 5.0)
            g = _build_in_order(G, True)
            ours = _by_int_label(g, g.load_centrality(return_dict=True))
            assert ours == pytest.approx(nx.load_centrality(G, weight="weight"), abs=1e-12)
            cutoff = rnd.uniform(1.0, 6.0)
            ours = _by_int_label(g, g.load_centrality(cutoff=cutoff, return_dict=True))
            ref = nx.load_centrality(G, cutoff=cutoff, weight="weight")
            assert ours == pytest.approx(ref, abs=1e-12)


class TestCentralityWeightedFlag:
    def test_follows_graph_by_default_and_can_be_overridden(self):
        g = _triangle_detour()
        b = g.get_id_from_label("B")
        assert g.betweenness_centrality(return_dict=True)[b] == 1.0
        assert g.betweenness_centrality(weighted=False, return_dict=True)[b] == 0.0

    def test_weighted_true_on_unweighted_graph_uses_stored_weights(self):
        g = (
            ec.Graph()
            .add_nodes(["A", "B", "C"])
            .add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 5.0)])
        )
        b = g.get_id_from_label("B")
        assert g.betweenness_centrality(return_dict=True)[b] == 0.0
        assert g.betweenness_centrality(weighted=True, return_dict=True)[b] == 1.0

    def test_consistent_with_shortest_path_length(self):
        g = _triangle_detour()
        close = g.closeness_centrality(return_dict=True)
        a = g.get_id_from_label("A")
        total = g.shortest_path_length("A", "B") + g.shortest_path_length("A", "C")
        assert close[a] == pytest.approx(2 / total)

    def test_unweighted_results_unchanged_on_weighted_graph_when_disabled(self):
        G = nx.gnp_random_graph(20, 0.2, seed=3)
        for u, v in G.edges:
            G[u][v]["weight"] = 2.5
        weighted, plain = _build_in_order(G, True), _build_in_order(G, False)
        assert weighted.betweenness_centrality(
            weighted=False, return_dict=True
        ) == plain.betweenness_centrality(return_dict=True)
        assert weighted.closeness_centrality(
            weighted=False, return_dict=True
        ) == plain.closeness_centrality(return_dict=True)

    def test_parallel_edges_only_cheapest_carries_paths(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"])
        g.add_edges([("A", "B", 1.0), ("A", "B", 3.0)])
        cheap, dear = g.find_edge_ids("A", "B")
        raw = g.edge_betweenness_centrality(normalized=False, return_dict=True)
        assert raw[cheap] == pytest.approx(1.0) and raw[dear] == 0.0

    def test_harmonic_cutoff_is_a_distance(self):
        g = _triangle_detour()
        h = g.harmonic_centrality(cutoff=1, return_dict=True)
        a = g.get_id_from_label("A")
        assert h[a] == pytest.approx(1.0)  # only B (distance 1) is within the cutoff

    def test_negative_weights_rejected(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=-1.0)
        for method in (
            "betweenness_centrality",
            "edge_betweenness_centrality",
            "closeness_centrality",
            "harmonic_centrality",
        ):
            with pytest.raises(ValueError, match="non-negative"):
                getattr(g, method)(return_dict=True)

    def test_pagerank_reads_weights_as_tie_strengths(self):
        """Path `a - b` (10), `b - c` (1) used to give `a` and `c` the same rank."""
        g = ec.Graph(weighted=True).add_nodes(list("abc"))
        g.add_edges([("a", "b", 10.0), ("b", "c", 1.0)])
        ranks = g.pagerank(return_dict=True)
        assert ranks[1] == pytest.approx(0.426, abs=5e-4)
        assert ranks[3] == pytest.approx(0.088, abs=5e-4)
        plain = g.pagerank(weighted=False, return_dict=True)
        assert plain[1] == pytest.approx(plain[3])
        assert g.pagerank(weighted=True, return_dict=True) == ranks

    def test_load_follows_weights(self):
        g = _triangle_detour()
        b = g.get_id_from_label("B")
        assert g.load_centrality(return_dict=True)[b] == 1.0
        assert g.load_centrality(weighted=False, return_dict=True)[b] == 0.0
        # A cutoff is a total weight: A -> B -> C costs 2
        assert g.load_centrality(cutoff=1.5, return_dict=True)[b] == 0.0

    def test_load_ties_split_evenly(self):
        """Two equal routes from A to C, direct and through B: B gets half of each pair."""
        g = ec.Graph(weighted=True).add_nodes(["A", "B", "C"])
        g.add_edges([("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 2.0)])
        raw = g.load_centrality(normalized=False, return_dict=True)
        assert raw == {1: 0.0, 2: 1.0, 3: 0.0}

    @pytest.mark.parametrize("weighted", [False, True])
    def test_load_parallel_edges_count_once(self, weighted):
        """A doubled `a - b` edge used to give `a` and `b` more load than `c` and `d`."""
        g = ec.Graph(weighted=weighted).add_nodes(list("abcd"))
        g.add_edges([("a", "b"), ("a", "b"), ("b", "d"), ("a", "c"), ("c", "d")])
        assert g.load_centrality(return_dict=True) == pytest.approx(
            {1: 1 / 6, 2: 1 / 6, 3: 1 / 6, 4: 1 / 6}
        )

    @pytest.mark.parametrize("bad", [-1.0, float("inf")])
    def test_pagerank_rejects_bad_weights(self, bad):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=bad)
        with pytest.raises(ValueError, match="non-negative"):
            g.pagerank(return_dict=True)
        assert len(g.pagerank(weighted=False, return_dict=True)) == 2

    def test_load_rejects_negative_weights(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            g.load_centrality(return_dict=True)

    def test_harmonic_has_no_weights_argument(self):
        with pytest.raises(TypeError, match="weights"):
            _triangle_detour().harmonic_centrality(weights="weight")

    def test_explicit_flag_recorded_in_computations(self):
        g = _triangle_detour().betweenness_centrality(weighted=False).closeness_centrality()
        assert ("betweenness_centrality", {"weighted": False}, "betweenness_centrality") in (
            g._computations
        )
        assert ("closeness_centrality", {}, "closeness_centrality") in g._computations

    def test_weighted_closeness_step_code(self):
        g = _triangle_detour().closeness_centrality(weighted=False)
        assert g.steps[-1].code == (
            "G.closeness_centrality(weighted=False, attr_name='closeness_centrality')"
        )


# Current-flow subsets and edge variants, edge betweenness on subsets, edge load and the most
# prominent group, validated against NetworkX and (where NetworkX deviates) brute force.


def _edge_lab(g: ec.Graph, scores: dict) -> dict:
    out = {}
    for eid, v in scores.items():
        u, w = sorted(int(g.get_label_from_id(n)) for n in g.get_node_ids_by_edge_id(eid))
        out[(u, w)] = v
    return out


def _undirected_ref(ref: dict) -> dict:
    return {tuple(sorted(e)): v for e, v in ref.items()}


class TestCurrentFlowVariants:
    @pytest.mark.parametrize("normalized", [True, False])
    def test_edge_current_flow_matches_networkx(self, normalized):
        for U, g, gw in _connected_cases():
            ours = _edge_lab(
                g, g.edge_current_flow_betweenness_centrality(normalized, return_dict=True)
            )
            ref = nx.edge_current_flow_betweenness_centrality(U, normalized)
            assert ours == pytest.approx(_undirected_ref(ref))
            ours = _edge_lab(
                gw, gw.edge_current_flow_betweenness_centrality(normalized, return_dict=True)
            )
            ref = nx.edge_current_flow_betweenness_centrality(U, normalized, weight="weight")
            assert ours == pytest.approx(_undirected_ref(ref))

    @pytest.mark.parametrize("normalized", [True, False])
    def test_edge_subset_matches_networkx(self, normalized):
        for seed, (U, g, gw) in enumerate(_connected_cases()):
            rnd = random.Random(seed)
            S, T = rnd.sample(range(14), 3), rnd.sample(range(14), 4)
            for graph, weight in ((g, None), (gw, "weight")):
                ours = graph.edge_current_flow_betweenness_centrality_subset(
                    _ids(graph, S), _ids(graph, T), normalized, return_dict=True
                )
                ref = nx.edge_current_flow_betweenness_centrality_subset(
                    U, S, T, normalized, weight=weight
                )
                assert _edge_lab(graph, ours) == pytest.approx(_undirected_ref(ref))

    @pytest.mark.parametrize("normalized", [True, False])
    def test_node_subset_against_direct_currents(self, normalized):
        import numpy as np

        for seed, (U, g, _) in enumerate(_connected_cases()):
            rnd = random.Random(seed)
            S, T = rnd.sample(range(14), 3), rnd.sample(range(14), 3)
            n = len(U)
            pinv = np.linalg.pinv(nx.laplacian_matrix(U, nodelist=range(n), weight=None).toarray())
            expected = dict.fromkeys(U, 0.0)
            for s in S:
                for t in T:
                    if s == t:
                        continue
                    p = pinv[:, s] - pinv[:, t]
                    for v in U:
                        if v not in (s, t):
                            expected[v] += 0.5 * sum(abs(p[v] - p[w]) for w in U[v])
            scale = 1 / ((n - 1) * (n - 2)) if normalized else 0.5
            ours = g.current_flow_betweenness_centrality_subset(
                _ids(g, S), _ids(g, T), normalized, return_dict=True
            )
            assert _lab(g, ours) == pytest.approx({v: x * scale for v, x in expected.items()})
            # every node as a source and a target gives the full measure (NetworkX agrees there)
            everyone = _ids(g, range(n))
            full = g.current_flow_betweenness_centrality(normalized, return_dict=True)
            assert g.current_flow_betweenness_centrality_subset(
                everyone, everyone, normalized, return_dict=True
            ) == pytest.approx(full)
        if normalized:
            ref = nx.current_flow_betweenness_centrality_subset(U, list(U), list(U), weight=None)
            assert _lab(g, full) == pytest.approx(ref)

    def test_approximation(self):
        for U, g, gw in _connected_cases():
            for graph in (g, gw):
                exact = graph.current_flow_betweenness_centrality(return_dict=True)
                approx = graph.approximate_current_flow_betweenness_centrality(
                    epsilon=0.1, kmax=100_000, seed=7, return_dict=True
                )
                assert max(abs(approx[n] - exact[n]) for n in exact) < 0.1
                again = graph.approximate_current_flow_betweenness_centrality(
                    epsilon=0.1, kmax=100_000, seed=7, return_dict=True
                )
                assert again == approx
        raw = g.approximate_current_flow_betweenness_centrality(
            normalized=False, seed=1, return_dict=True
        )
        normed = g.approximate_current_flow_betweenness_centrality(seed=1, return_dict=True)
        assert raw == pytest.approx({k: v * 13 * 12 / 2 for k, v in normed.items()})
        with pytest.raises(ValueError, match="kmax"):
            g.approximate_current_flow_betweenness_centrality(epsilon=0.01, kmax=100)

    def test_storage_and_errors(self):
        g = ec.Graph().add_cycle_graph(5)
        g.edge_current_flow_betweenness_centrality()
        stored = g.get_edge_df()["edge_current_flow_betweenness_centrality"].to_list()
        ring = nx.edge_current_flow_betweenness_centrality(nx.cycle_graph(5))
        assert stored == pytest.approx([next(iter(ring.values()))] * 5)
        step = g.steps[-1]
        assert step.kind == KIND_METRIC and len(step.delta.edge_metric_values) == 5
        split = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("C", "D")])
        with pytest.raises(ValueError, match="connected"):
            split.edge_current_flow_betweenness_centrality()
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_path_graph(3).current_flow_betweenness_centrality_subset(
                [1], [3]
            )
        with pytest.raises(ValueError, match="does not exist"):
            g.edge_current_flow_betweenness_centrality_subset([1], [99])


class TestEdgeSubsetAndLoad:
    def test_edge_betweenness_subset_brute_force(self):
        for directed in (False, True):
            for G, g, gw, rnd in _random_cases(directed):
                S, T = rnd.sample(range(12), 3), rnd.sample(range(12), 3)
                for graph, weight in ((g, None), (gw, "weight")):
                    ours = graph.edge_betweenness_centrality_subset(
                        _ids(graph, S), _ids(graph, T), return_dict=True
                    )
                    brute: dict = {}
                    for s in S:
                        for t in T:
                            if s == t or not nx.has_path(G, s, t):
                                continue
                            paths = list(nx.all_shortest_paths(G, s, t, weight=weight))
                            for p in paths:
                                for e in zip(p, p[1:]):
                                    e = e if directed else tuple(sorted(e))
                                    brute[e] = brute.get(e, 0.0) + 1 / len(paths)
                    scale = 1.0 if directed else 0.5
                    for eid, value in ours.items():
                        u, v = (
                            int(graph.get_label_from_id(n))
                            for n in graph.get_node_ids_by_edge_id(eid)
                        )
                        key = (u, v) if directed else tuple(sorted((u, v)))
                        assert value == pytest.approx(brute.get(key, 0.0) * scale)

    def test_edge_betweenness_subset_tree_matches_networkx(self):
        # on a tree every pair has one shortest path, where NetworkX's split is also right
        T = (
            nx.random_labeled_tree(15, seed=3)
            if hasattr(nx, "random_labeled_tree")
            else (nx.random_tree(15, seed=3))
        )
        g = _build(T)
        S, D = [0, 4, 9], [2, 7, 11, 14]
        for normalized in (False, True):
            ours = g.edge_betweenness_centrality_subset(
                _ids(g, S), _ids(g, D), normalized, return_dict=True
            )
            ref = nx.edge_betweenness_centrality_subset(T, S, D, normalized=normalized)
            assert _edge_lab(g, ours) == pytest.approx(_undirected_ref(ref))

    def test_edge_load_matches_networkx(self):
        for directed in (False, True):
            for G, g, _, _ in _random_cases(directed):
                for cutoff in (None, 2):
                    ours = g.edge_load_centrality(cutoff, return_dict=True)
                    ref = nx.edge_load_centrality(G, cutoff=cutoff or False)
                    assert len(ours) == G.number_of_edges()
                    for eid, x in ours.items():
                        u, v = (int(g.get_label_from_id(n)) for n in g.get_node_ids_by_edge_id(eid))
                        assert x == pytest.approx(ref[(u, v)]) and x == pytest.approx(ref[(v, u)])
        path = ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("B", "C")])
        with pytest.raises(ValueError, match="cutoff"):
            path.edge_load_centrality(0)

    def test_steps(self):
        g = ec.Graph().add_path_graph(4)
        g.edge_betweenness_centrality_subset([1], [4], attr_name="flow")
        assert set(g.get_edge_df()["flow"]) == {0.5}
        assert g.steps[-1].kind == KIND_METRIC
        g.edge_load_centrality()
        assert set(g.get_edge_df()["edge_load_centrality"]) == {6.0, 8.0}
        assert g.steps[-1].kind == KIND_METRIC


class TestProminentGroup:
    def test_optimal_against_exhaustive_search(self):
        for directed in (False, True):
            for G, g, gw, rnd in _random_cases(directed):
                for graph, weighted in ((g, False), (gw, True)):
                    ids = graph.get_node_ids(limit=None)
                    for k in (1, 2, 3):
                        for endpoints in (False, True):
                            # raw scores: with endpoints, prominent_group() and
                            # group_betweenness_centrality() normalize differently (as in
                            # NetworkX 3.7)
                            value, group = graph.prominent_group(
                                k, endpoints=endpoints, normalized=False
                            )
                            best = max(
                                graph.group_betweenness_centrality(
                                    list(c), normalized=False, endpoints=endpoints
                                )
                                for c in itertools.combinations(ids, k)
                            )
                            assert value == pytest.approx(best)
                            assert graph.group_betweenness_centrality(
                                group, normalized=False, endpoints=endpoints
                            ) == pytest.approx(best)

    def test_matches_networkx_on_connected_graphs(self):
        for U, g, _ in list(_connected_cases())[:4]:
            for k in (2, 3):
                value, group = g.prominent_group(k)
                ref_value, ref_group = nx.prominent_group(U, k)
                assert round(value, 2) == ref_value
                assert value == pytest.approx(nx.group_betweenness_centrality(U, ref_group))

    def test_options(self):
        g = ec.Graph().add_nodes(list("ABCDEFG"))
        g.add_edges(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "A"),
                ("C", "D"),
                ("D", "E"),
                ("E", "F"),
                ("F", "G"),
                ("G", "E"),
            ]
        )
        value, group = g.prominent_group(2)
        assert {g.get_label_from_id(n) for n in group} == {"C", "E"}
        _, without = g.prominent_group(2, exclude=["C"])
        assert g.get_id_from_label("C") not in without
        gv, gg = g.prominent_group(2, greedy=True)
        assert gv == pytest.approx(g.group_betweenness_centrality(gg)) and gv <= value
        raw, _ = g.prominent_group(2, normalized=False)
        assert raw == pytest.approx(value * 5 * 4 / 2)
        assert g.steps[-1].delta.nodes_selected
        with pytest.raises(ValueError, match="at least 1"):
            g.prominent_group(0)
        with pytest.raises(ValueError, match="larger"):
            g.prominent_group(7, exclude=["A"])
