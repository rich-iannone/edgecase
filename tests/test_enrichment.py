"""Tests for network enrichment testing: subgraph enrichment, module annotation enrichment."""

import math

import pytest
import edgecase as ec


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def annotated_graph():
    g = ec.Graph()
    for label in ["G1", "G2", "G3", "G4", "G5", "G6"]:
        g.add_node(label=label)
    g.add_edge("G1", "G2").add_edge("G2", "G3").add_edge("G3", "G1")
    g.add_edge("G4", "G5").add_edge("G5", "G6").add_edge("G6", "G4")

    for label, pathway in [
        ("G1", "apoptosis"),
        ("G2", "apoptosis"),
        ("G3", "apoptosis"),
        ("G4", "signaling"),
        ("G5", "signaling"),
        ("G6", "apoptosis"),
    ]:
        nid = g.get_id_from_label(label)
        g.set_node_attrs("pathway", pathway, nodes=[nid])
    return g


@pytest.fixture
def large_annotated():
    """Larger graph with clear enrichment signal."""
    g = ec.Graph()
    for i in range(20):
        g.add_node(label=f"N{i}")
    # Cluster 1: N0-N9, mostly pathway X
    for i in range(9):
        g.add_edge(f"N{i}", f"N{i+1}")
    g.add_edge("N9", "N0")

    # Cluster 2: N10-N19, mostly pathway Y
    for i in range(10, 19):
        g.add_edge(f"N{i}", f"N{i+1}")
    g.add_edge("N19", "N10")

    # Bridge
    g.add_edge("N9", "N10")

    # Assign pathways
    for i in range(10):
        nid = g.get_id_from_label(f"N{i}")
        pw = "X" if i < 8 else "Y"
        g.set_node_attrs("pathway", pw, nodes=[nid])

    for i in range(10, 20):
        nid = g.get_id_from_label(f"N{i}")
        pw = "Y" if i < 18 else "X"
        g.set_node_attrs("pathway", pw, nodes=[nid])

    return g


# =============================================================================
# Subgraph Enrichment
# =============================================================================


class TestSubgraphEnrichment:
    def test_enriched_subgraph(self, annotated_graph):
        g = annotated_graph
        sub = [g.get_id_from_label(label) for label in ["G1", "G2", "G3"]]
        result = g.subgraph_enrichment(sub, "pathway", "apoptosis")
        assert result["k"] == 3
        assert result["n"] == 3
        assert result["fold_enrichment"] > 1.0
        assert 0.0 <= result["p_value"] <= 1.0

    def test_no_enrichment(self, annotated_graph):
        g = annotated_graph
        sub = [g.get_id_from_label(label) for label in ["G4", "G5"]]
        result = g.subgraph_enrichment(sub, "pathway", "apoptosis")
        assert result["k"] == 0
        assert result["fold_enrichment"] == 0.0

    def test_all_values_mode(self, annotated_graph):
        g = annotated_graph
        sub = [g.get_id_from_label(label) for label in ["G1", "G2", "G3"]]
        results = g.subgraph_enrichment(sub, "pathway")
        assert "apoptosis" in results
        assert "signaling" in results
        assert results["apoptosis"]["fold_enrichment"] > results["signaling"]["fold_enrichment"]

    def test_p_values_in_range(self, annotated_graph):
        g = annotated_graph
        sub = [g.get_id_from_label(label) for label in ["G1", "G2"]]
        results = g.subgraph_enrichment(sub, "pathway")
        for value, result in results.items():
            assert 0.0 <= result["p_value"] <= 1.0

    def test_subgraph_equals_full_graph(self, annotated_graph):
        g = annotated_graph
        all_ids = g.get_node_ids()
        result = g.subgraph_enrichment(all_ids, "pathway", "apoptosis")
        assert result["fold_enrichment"] == pytest.approx(1.0, abs=0.01)

    def test_empty_subgraph(self, annotated_graph):
        g = annotated_graph
        result = g.subgraph_enrichment([], "pathway", "apoptosis")
        assert result["k"] == 0
        assert result["n"] == 0

    def test_missing_attribute(self, annotated_graph):
        g = annotated_graph
        sub = [g.get_id_from_label("G1")]
        result = g.subgraph_enrichment(sub, "nonexistent_attr", "anything")
        assert result["K"] == 0
        assert result["k"] == 0

    def test_chi_squared_method(self, annotated_graph):
        g = annotated_graph
        sub = [g.get_id_from_label(label) for label in ["G1", "G2", "G3"]]
        result = g.subgraph_enrichment(sub, "pathway", "apoptosis", method="chi_squared")
        assert 0.0 <= result["p_value"] <= 1.0

    def test_chi_squared_is_one_sided(self):
        # 10 nodes, 5 tagged "A". Subgraph of 4: E = 2. k = 4 is enrichment, k = 0 is depletion
        # by the same distance, which the old |k - E| statistic scored identically
        g = ec.Graph().add_nodes([f"N{i}" for i in range(10)])
        ids = g.get_node_ids()
        g.set_node_attrs("tag", {nid: ("A" if i < 5 else "B") for i, nid in enumerate(ids)})

        enriched = g.subgraph_enrichment(ids[:4], "tag", "A", method="chi_squared")
        depleted = g.subgraph_enrichment(ids[6:], "tag", "A", method="chi_squared")
        neutral = g.subgraph_enrichment(ids[3:7], "tag", "A", method="chi_squared")

        z = 2 / math.sqrt(2)
        assert enriched["p_value"] == pytest.approx(0.5 * math.erfc(z / math.sqrt(2)))
        assert depleted["p_value"] == pytest.approx(0.5 * math.erfc(-z / math.sqrt(2)))
        assert depleted["p_value"] > 0.5 > enriched["p_value"]
        assert neutral["p_value"] == pytest.approx(0.5)
        assert enriched["p_value"] + depleted["p_value"] == pytest.approx(1.0)

    def test_invalid_method_raises(self, annotated_graph):
        g = annotated_graph
        sub = [g.get_id_from_label("G1")]
        with pytest.raises(ValueError):
            g.subgraph_enrichment(sub, "pathway", "apoptosis", method="invalid")

    def test_result_dict_keys(self, annotated_graph):
        g = annotated_graph
        sub = [g.get_id_from_label("G1")]
        result = g.subgraph_enrichment(sub, "pathway", "apoptosis")
        assert set(result.keys()) == {"value", "k", "K", "n", "N", "p_value", "fold_enrichment"}

    def test_larger_graph_enrichment(self, large_annotated):
        g = large_annotated
        sub = [g.get_id_from_label(f"N{i}") for i in range(10)]
        result = g.subgraph_enrichment(sub, "pathway", "X")
        assert result["fold_enrichment"] > 1.0
        assert result["k"] == 8
        assert result["K"] == 10


# =============================================================================
# Module Annotation Enrichment
# =============================================================================


class TestModuleAnnotationEnrichment:
    def test_returns_results_per_community(self, annotated_graph):
        g = annotated_graph
        communities = g.cmty_louvain(return_dict=True)
        results = g.module_annotation_enrichment(communities, "pathway")
        assert len(results) > 0
        for cmty_id, vals in results.items():
            assert isinstance(vals, dict)

    def test_bonferroni_correction(self, annotated_graph):
        g = annotated_graph
        communities = g.cmty_louvain(return_dict=True)
        results = g.module_annotation_enrichment(
            communities, "pathway", correction="bonferroni"
        )
        for cmty_id, vals in results.items():
            for value, result in vals.items():
                assert "p_adjusted" in result
                assert result["p_adjusted"] >= result["p_value"]
                assert result["p_adjusted"] <= 1.0

    def test_fdr_correction(self, annotated_graph):
        g = annotated_graph
        communities = g.cmty_louvain(return_dict=True)
        results = g.module_annotation_enrichment(
            communities, "pathway", correction="fdr"
        )
        for cmty_id, vals in results.items():
            for value, result in vals.items():
                assert "p_adjusted" in result
                assert result["p_adjusted"] <= 1.0

    def test_fdr_monotonicity(self, large_annotated):
        g = large_annotated
        communities = g.cmty_louvain(return_dict=True)
        results = g.module_annotation_enrichment(
            communities, "pathway", correction="fdr"
        )
        # Collect all adjusted p-values and verify they are <= 1.0
        for cmty_id, vals in results.items():
            for value, result in vals.items():
                assert 0.0 <= result["p_adjusted"] <= 1.0

    def test_invalid_correction_raises(self, annotated_graph):
        g = annotated_graph
        communities = g.cmty_louvain(return_dict=True)
        with pytest.raises(ValueError):
            g.module_annotation_enrichment(
                communities, "pathway", correction="invalid"
            )

    def test_single_community(self, annotated_graph):
        g = annotated_graph
        all_ids = g.get_node_ids()
        communities = {nid: 0 for nid in all_ids}
        results = g.module_annotation_enrichment(communities, "pathway")
        assert 0 in results

    def test_empty_communities(self, annotated_graph):
        g = annotated_graph
        results = g.module_annotation_enrichment({}, "pathway")
        assert results == {}
