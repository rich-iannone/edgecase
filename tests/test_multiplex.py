import pytest
import edgecase as ec
from edgecase.multiplex import MultiplexGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_two_layer_4node():
    """Two layers on 4 nodes with complementary edges."""
    layer_a = (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge(source="A", target="B")
        .add_edge(source="C", target="D")
    )
    layer_b = (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge(source="A", target="C")
        .add_edge(source="B", target="D")
    )
    return MultiplexGraph({"a": layer_a, "b": layer_b})


def make_three_layers():
    """Three layers on 3 nodes."""
    labels = ["X", "Y", "Z"]
    g1 = ec.Graph()
    g2 = ec.Graph()
    g3 = ec.Graph()
    for lbl in labels:
        g1.add_node(label=lbl)
        g2.add_node(label=lbl)
        g3.add_node(label=lbl)
    g1.add_edge(source="X", target="Y")
    g2.add_edge(source="Y", target="Z")
    g3.add_edge(source="X", target="Z")
    return MultiplexGraph({"ppi": g1, "coex": g2, "genetic": g3})


def make_identical_layers():
    """Two identical layers (same edges)."""

    def mk():
        return (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
        )

    return MultiplexGraph({"layer1": mk(), "layer2": mk()})


def make_single_layer():
    """A multiplex with only one layer."""
    g = ec.Graph().add_node(label="A").add_node(label="B").add_edge(source="A", target="B")
    return MultiplexGraph({"only": g})


def make_directed_layers():
    """Two directed layers."""
    g1 = (
        ec.Graph(directed=True)
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="A", target="B")
        .add_edge(source="B", target="C")
    )
    g2 = (
        ec.Graph(directed=True)
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="C", target="A")
    )
    return MultiplexGraph({"forward": g1, "backward": g2})


def make_cluster_layers():
    """Two layers that both reinforce the same 2-community structure."""
    nodes = [str(i) for i in range(8)]

    def mk_layer(intra_edges, inter_edges):
        g = ec.Graph()
        for n in nodes:
            g.add_node(label=n)
        for s, t in intra_edges:
            g.add_edge(source=str(s), target=str(t))
        for s, t in inter_edges:
            g.add_edge(source=str(s), target=str(t))
        return g

    # Cluster 1: {0,1,2,3}, Cluster 2: {4,5,6,7}
    intra1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    intra2 = [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
    g1 = mk_layer(intra1 + intra2, [(3, 4)])
    g2 = mk_layer(intra1 + intra2, [(3, 4)])
    return MultiplexGraph({"layer1": g1, "layer2": g2})


# ===========================================================================
# MultiplexGraph — Construction
# ===========================================================================


class TestMultiplexConstruction:
    def test_basic_creation(self):
        mpx = make_two_layer_4node()
        assert mpx.count_nodes() == 4
        assert mpx.count_layers() == 2

    def test_layer_names(self):
        mpx = make_two_layer_4node()
        assert set(mpx.layer_names) == {"a", "b"}

    def test_get_layer(self):
        mpx = make_two_layer_4node()
        layer = mpx.get_layer("a")
        assert isinstance(layer, ec.Graph)
        assert layer.count_edges() == 2

    def test_get_layer_missing(self):
        mpx = make_two_layer_4node()
        with pytest.raises(KeyError, match="no_such"):
            mpx.get_layer("no_such")

    def test_node_labels(self):
        mpx = make_two_layer_4node()
        assert mpx.node_labels() == {"A", "B", "C", "D"}

    def test_repr(self):
        mpx = make_two_layer_4node()
        r = repr(mpx)
        assert "MultiplexGraph" in r
        assert "4 nodes" in r

    def test_layers_property_returns_copy(self):
        mpx = make_two_layer_4node()
        layers = mpx.layers
        layers["new"] = ec.Graph()
        assert "new" not in mpx.layers

    def test_three_layers(self):
        mpx = make_three_layers()
        assert mpx.count_layers() == 3
        assert mpx.count_nodes() == 3

    def test_single_layer(self):
        mpx = make_single_layer()
        assert mpx.count_layers() == 1
        assert mpx.count_nodes() == 2


# ===========================================================================
# MultiplexGraph — Validation errors
# ===========================================================================


class TestMultiplexValidation:
    def test_empty_layers(self):
        with pytest.raises(ValueError, match="At least one layer"):
            MultiplexGraph({})

    def test_mismatched_nodes(self):
        g1 = ec.Graph().add_node(label="A").add_node(label="B")
        g2 = ec.Graph().add_node(label="A").add_node(label="C")
        with pytest.raises(ValueError, match="do not match"):
            MultiplexGraph({"g1": g1, "g2": g2})

    def test_mismatched_node_count(self):
        g1 = ec.Graph().add_node(label="A").add_node(label="B")
        g2 = ec.Graph().add_node(label="A")
        with pytest.raises(ValueError, match="do not match"):
            MultiplexGraph({"g1": g1, "g2": g2})

    def test_three_layers_one_mismatched(self):
        g1 = ec.Graph().add_node(label="A").add_node(label="B")
        g2 = ec.Graph().add_node(label="A").add_node(label="B")
        g3 = ec.Graph().add_node(label="A").add_node(label="X")
        with pytest.raises(ValueError, match="do not match"):
            MultiplexGraph({"g1": g1, "g2": g2, "g3": g3})


# ===========================================================================
# multiplex_pagerank — basic
# ===========================================================================


class TestMultiplexPageRankBasic:
    def test_returns_dict(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_pagerank(return_dict=True)
        assert isinstance(result, dict)
        assert len(result) == 4

    def test_scores_sum_to_one(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_pagerank(return_dict=True)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_nodes_present(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_pagerank(return_dict=True)
        assert set(result.keys()) == {"A", "B", "C", "D"}

    def test_scores_positive(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_pagerank(return_dict=True)
        for score in result.values():
            assert score > 0

    def test_chaining(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_pagerank()
        assert result is mpx

    def test_return_dict_false(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_pagerank(return_dict=False)
        assert isinstance(result, MultiplexGraph)


# ===========================================================================
# multiplex_pagerank — layer weights
# ===========================================================================


class TestMultiplexPageRankWeights:
    def test_equal_weights(self):
        mpx = make_two_layer_4node()
        r1 = mpx.multiplex_pagerank(return_dict=True)
        r2 = mpx.multiplex_pagerank(layer_weights={"a": 1.0, "b": 1.0}, return_dict=True)
        for k in r1:
            assert abs(r1[k] - r2[k]) < 1e-10

    def test_zero_weight_ignores_layer(self):
        mpx = make_two_layer_4node()
        # Only layer "a" matters (A-B, C-D)
        r = mpx.multiplex_pagerank(layer_weights={"a": 1.0, "b": 0.0}, return_dict=True)
        # A and B should have similar scores, C and D similar
        assert abs(r["A"] - r["B"]) < 0.01
        assert abs(r["C"] - r["D"]) < 0.01

    def test_unequal_weights_changes_result(self):
        # Use asymmetric layers so zeroing one vs. the other gives different results
        g1 = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="A", target="C")
        )
        g2 = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="B", target="C")
        )
        mpx = MultiplexGraph({"x": g1, "y": g2})
        r1 = mpx.multiplex_pagerank(layer_weights={"x": 1.0, "y": 0.0}, return_dict=True)
        r2 = mpx.multiplex_pagerank(layer_weights={"x": 0.0, "y": 1.0}, return_dict=True)
        diffs = [abs(r1[k] - r2[k]) for k in r1]
        assert max(diffs) > 0.001

    def test_high_weight_layer_dominates(self):
        mpx = make_two_layer_4node()
        # Layer a: A-B, C-D; layer b: A-C, B-D; weight b 100x
        r = mpx.multiplex_pagerank(layer_weights={"a": 0.01, "b": 100.0}, return_dict=True)
        # All nodes roughly equal (layer b has symmetric structure)
        scores = list(r.values())
        assert max(scores) - min(scores) < 0.05


# ===========================================================================
# multiplex_pagerank — parameters
# ===========================================================================


class TestMultiplexPageRankParams:
    def test_damping_effect(self):
        mpx = make_two_layer_4node()
        r_low = mpx.multiplex_pagerank(damping=0.5, return_dict=True)
        r_high = mpx.multiplex_pagerank(damping=0.99, return_dict=True)
        # Higher damping means more influence from graph structure
        # Lower damping means scores converge toward uniform
        scores_low = sorted(r_low.values())
        scores_high = sorted(r_high.values())
        range_low = scores_low[-1] - scores_low[0]
        range_high = scores_high[-1] - scores_high[0]
        assert range_high >= range_low

    def test_three_layers(self):
        mpx = make_three_layers()
        result = mpx.multiplex_pagerank(return_dict=True)
        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_single_layer(self):
        mpx = make_single_layer()
        result = mpx.multiplex_pagerank(return_dict=True)
        assert len(result) == 2
        assert abs(sum(result.values()) - 1.0) < 1e-6


# ===========================================================================
# multiplex_pagerank — edge cases
# ===========================================================================


class TestMultiplexPageRankEdgeCases:
    def test_identical_layers(self):
        mpx = make_identical_layers()
        result = mpx.multiplex_pagerank(return_dict=True)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_directed_layers(self):
        mpx = make_directed_layers()
        result = mpx.multiplex_pagerank(return_dict=True)
        assert len(result) == 3
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_no_edges_in_one_layer(self):
        g1 = ec.Graph().add_node(label="A").add_node(label="B").add_edge(source="A", target="B")
        g2 = ec.Graph().add_node(label="A").add_node(label="B")
        mpx = MultiplexGraph({"edges": g1, "empty": g2})
        result = mpx.multiplex_pagerank(return_dict=True)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_single_node_layers(self):
        g1 = ec.Graph().add_node(label="A")
        g2 = ec.Graph().add_node(label="A")
        mpx = MultiplexGraph({"g1": g1, "g2": g2})
        result = mpx.multiplex_pagerank(return_dict=True)
        assert result == {"A": 1.0}


# ===========================================================================
# multiplex_community_detection — basic
# ===========================================================================


class TestMultiplexCommunityBasic:
    def test_returns_dict(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_community_detection(return_dict=True)
        assert isinstance(result, dict)
        assert len(result) == 4

    def test_all_nodes_assigned(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_community_detection(return_dict=True)
        assert set(result.keys()) == {"A", "B", "C", "D"}

    def test_community_ids_are_ints(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_community_detection(return_dict=True)
        for v in result.values():
            assert isinstance(v, int)

    def test_chaining(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_community_detection()
        assert result is mpx

    def test_return_dict_false(self):
        mpx = make_two_layer_4node()
        result = mpx.multiplex_community_detection(return_dict=False)
        assert isinstance(result, MultiplexGraph)


# ===========================================================================
# multiplex_community_detection — methods
# ===========================================================================


class TestMultiplexCommunityMethods:
    def test_louvain_default(self):
        mpx = make_cluster_layers()
        result = mpx.multiplex_community_detection(method="louvain", return_dict=True)
        assert len(result) == 8

    def test_walktrap(self):
        mpx = make_cluster_layers()
        result = mpx.multiplex_community_detection(method="walktrap", return_dict=True)
        assert len(result) == 8

    def test_invalid_method(self):
        mpx = make_two_layer_4node()
        with pytest.raises(ValueError, match="Unknown method"):
            mpx.multiplex_community_detection(method="invalid", return_dict=True)

    def test_two_clusters_detected(self):
        mpx = make_cluster_layers()
        result = mpx.multiplex_community_detection(return_dict=True)
        cluster_0 = {k for k, v in result.items() if v == result["0"]}
        cluster_1 = {k for k, v in result.items() if v != result["0"]}
        expected_0 = {"0", "1", "2", "3"}
        expected_1 = {"4", "5", "6", "7"}
        # The community IDs may be swapped, but the partition should match
        assert (cluster_0 == expected_0 and cluster_1 == expected_1) or (
            cluster_0 == expected_1 and cluster_1 == expected_0
        )


# ===========================================================================
# multiplex_community_detection — layer weights
# ===========================================================================


class TestMultiplexCommunityWeights:
    def test_equal_weights(self):
        mpx = make_cluster_layers()
        r1 = mpx.multiplex_community_detection(return_dict=True)
        r2 = mpx.multiplex_community_detection(
            layer_weights={"layer1": 1.0, "layer2": 1.0}, return_dict=True
        )
        assert r1 == r2

    def test_zero_weight_layer(self):
        mpx = make_cluster_layers()
        result = mpx.multiplex_community_detection(
            layer_weights={"layer1": 1.0, "layer2": 0.0}, return_dict=True
        )
        assert len(result) == 8


# ===========================================================================
# multiplex_community_detection — resolution
# ===========================================================================


class TestMultiplexCommunityResolution:
    def test_high_resolution_more_communities(self):
        mpx = make_cluster_layers()
        r_low = mpx.multiplex_community_detection(resolution=0.1, return_dict=True)
        r_high = mpx.multiplex_community_detection(resolution=10.0, return_dict=True)
        n_low = len(set(r_low.values()))
        n_high = len(set(r_high.values()))
        assert n_high >= n_low


# ===========================================================================
# multiplex_community_detection — edge cases
# ===========================================================================


class TestMultiplexCommunityEdgeCases:
    def test_single_node(self):
        g1 = ec.Graph().add_node(label="A")
        g2 = ec.Graph().add_node(label="A")
        mpx = MultiplexGraph({"g1": g1, "g2": g2})
        result = mpx.multiplex_community_detection(return_dict=True)
        assert result == {"A": 0}

    def test_two_isolated_nodes(self):
        g1 = ec.Graph().add_node(label="A").add_node(label="B")
        g2 = ec.Graph().add_node(label="A").add_node(label="B")
        mpx = MultiplexGraph({"g1": g1, "g2": g2})
        result = mpx.multiplex_community_detection(return_dict=True)
        assert len(result) == 2

    def test_no_edges(self):
        g1 = ec.Graph().add_node(label="A").add_node(label="B").add_node(label="C")
        g2 = ec.Graph().add_node(label="A").add_node(label="B").add_node(label="C")
        mpx = MultiplexGraph({"g1": g1, "g2": g2})
        result = mpx.multiplex_community_detection(return_dict=True)
        assert len(result) == 3

    def test_three_layers_community(self):
        mpx = make_three_layers()
        result = mpx.multiplex_community_detection(return_dict=True)
        assert len(result) == 3

    def test_directed_layers_community(self):
        mpx = make_directed_layers()
        result = mpx.multiplex_community_detection(return_dict=True)
        assert len(result) == 3

    def test_walktrap_edge_case(self):
        g1 = ec.Graph().add_node(label="A").add_node(label="B").add_edge(source="A", target="B")
        g2 = ec.Graph().add_node(label="A").add_node(label="B")
        mpx = MultiplexGraph({"g1": g1, "g2": g2})
        result = mpx.multiplex_community_detection(method="walktrap", return_dict=True)
        assert len(result) == 2


# ===========================================================================
# MultiplexGraph — integration
# ===========================================================================


class TestMultiplexIntegration:
    def test_pagerank_then_community(self):
        mpx = make_cluster_layers()
        mpx.multiplex_pagerank()
        result = mpx.multiplex_community_detection(return_dict=True)
        assert len(result) == 8

    def test_community_then_pagerank(self):
        mpx = make_cluster_layers()
        mpx.multiplex_community_detection()
        result = mpx.multiplex_pagerank(return_dict=True)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_access_via_edgecase_module(self):
        g = ec.Graph().add_node(label="A")
        mpx = ec.MultiplexGraph({"g": g})
        assert isinstance(mpx, MultiplexGraph)

    def test_large_multiplex(self):
        """10 layers with 20 nodes each."""
        nodes = [str(i) for i in range(20)]
        layers = {}
        for layer_idx in range(10):
            g = ec.Graph()
            for n in nodes:
                g.add_node(label=n)
            for i in range(0, 20, 2):
                g.add_edge(source=str(i), target=str((i + 1 + layer_idx) % 20))
            layers[f"layer_{layer_idx}"] = g
        mpx = MultiplexGraph(layers)
        assert mpx.count_layers() == 10
        assert mpx.count_nodes() == 20
        pr = mpx.multiplex_pagerank(return_dict=True)
        assert len(pr) == 20
        assert abs(sum(pr.values()) - 1.0) < 1e-6
