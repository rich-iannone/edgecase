import pytest
import edgecase as ec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_three_cliques():
    """Three 4-cliques connected by bridges."""
    g = ec.Graph()
    for i in range(12):
        g.add_node(label=str(i))
    for base in [0, 4, 8]:
        for i in range(base, base + 4):
            for j in range(i + 1, base + 4):
                g.add_edge(source=str(i), target=str(j))
    g.add_edge(source="3", target="4")
    g.add_edge(source="7", target="8")
    return g


def make_chain(n=10):
    g = ec.Graph()
    for i in range(n):
        g.add_node(label=str(i))
    for i in range(n - 1):
        g.add_edge(source=str(i), target=str(i + 1))
    return g


# ===========================================================================
# gwas_pipeline — basic functionality
# ===========================================================================


class TestGwasPipelineBasic:
    def test_returns_all_keys(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        result = g.gwas_pipeline(scores)

        assert "smoothed_scores" in result
        assert "communities" in result
        assert "module_scores" in result

    def test_smoothed_scores_for_all_nodes(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        result = g.gwas_pipeline(scores)

        assert len(result["smoothed_scores"]) == 12

    def test_communities_cover_all_nodes(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        result = g.gwas_pipeline(scores)
        all_nodes = set()
        for members in result["communities"].values():
            all_nodes.update(members)

        assert all_nodes == set(g.get_node_ids())

    def test_module_scores_sorted_descending(self):
        g = make_three_cliques()
        scores = {}
        for i in range(4):
            scores[g.get_id_from_label(str(i))] = 10.0
        for i in range(4, 12):
            scores[g.get_id_from_label(str(i))] = 0.1
        result = g.gwas_pipeline(scores)
        ms = list(result["module_scores"].values())
        for i in range(1, len(ms)):
            assert ms[i] <= ms[i - 1]

    def test_high_score_module_ranks_first(self):
        g = make_three_cliques()
        scores = {}
        for i in range(4):
            scores[g.get_id_from_label(str(i))] = 100.0
        for i in range(4, 12):
            scores[g.get_id_from_label(str(i))] = 0.0
        result = g.gwas_pipeline(scores)

        # The top module should have a high mean score
        top_score = list(result["module_scores"].values())[0]

        assert top_score > 1.0


# ===========================================================================
# gwas_pipeline — community methods
# ===========================================================================


class TestGwasPipelineMethods:
    def test_louvain(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        result = g.gwas_pipeline(scores, community_method="louvain")

        assert len(result["communities"]) >= 1

    def test_walktrap(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        result = g.gwas_pipeline(scores, community_method="walktrap")

        assert len(result["communities"]) >= 1

    def test_invalid_method(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        with pytest.raises(ValueError, match="Unknown community method"):
            g.gwas_pipeline(scores, community_method="invalid")


# ===========================================================================
# gwas_pipeline — smoothing parameters
# ===========================================================================


class TestGwasPipelineSmoothing:
    def test_alpha_zero_preserves_signal(self):
        g = make_three_cliques()
        n0 = g.get_id_from_label("0")
        scores = {n0: 100.0}
        result = g.gwas_pipeline(scores, smooth_alpha=0.0)

        # With alpha=0, smoothed = original
        assert abs(result["smoothed_scores"][n0] - 100.0) < 1e-6

    def test_higher_alpha_spreads_signal(self):
        g = make_chain()
        n0 = g.get_id_from_label("0")
        n5 = g.get_id_from_label("5")
        scores = {n0: 100.0}
        low = g.gwas_pipeline(scores, smooth_alpha=0.2)
        high = g.gwas_pipeline(scores, smooth_alpha=0.8)

        assert high["smoothed_scores"][n5] > low["smoothed_scores"][n5]


# ===========================================================================
# gwas_pipeline — enrichment
# ===========================================================================


class TestGwasPipelineEnrichment:
    def test_enrichment_returned(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        positive = [g.get_id_from_label(str(i)) for i in range(4)]
        result = g.gwas_pipeline(scores, enrichment_attr="gene_set", enrichment_labels=positive)

        assert "enrichment" in result
        assert len(result["enrichment"]) == len(result["communities"])

    def test_enrichment_values_positive(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        positive = [g.get_id_from_label(str(i)) for i in range(4)]
        result = g.gwas_pipeline(scores, enrichment_attr="gene_set", enrichment_labels=positive)
        for val in result["enrichment"].values():
            assert val >= 0.0

    def test_no_enrichment_without_params(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        result = g.gwas_pipeline(scores)

        assert "enrichment" not in result

    def test_concentrated_set_has_high_enrichment(self):
        g = make_three_cliques()
        scores = {nid: 1.0 for nid in g.get_node_ids()}

        # All positive nodes in first clique
        positive = [g.get_id_from_label(str(i)) for i in range(4)]
        result = g.gwas_pipeline(scores, enrichment_attr="gene_set", enrichment_labels=positive)

        # At least one community should be enriched (ratio > 1.0)
        max_enrichment = max(result["enrichment"].values())

        assert max_enrichment > 1.0


# ===========================================================================
# gwas_pipeline — edge cases
# ===========================================================================


class TestGwasPipelineEdgeCases:
    def test_empty_scores(self):
        g = make_three_cliques()
        result = g.gwas_pipeline({})

        assert len(result["smoothed_scores"]) == 12

        for val in result["smoothed_scores"].values():
            assert abs(val) < 1e-10

    def test_all_same_scores(self):
        g = make_three_cliques()
        scores = {nid: 5.0 for nid in g.get_node_ids()}
        result = g.gwas_pipeline(scores)
        vals = list(result["module_scores"].values())

        # All modules should have roughly equal scores
        assert max(vals) - min(vals) < 1.0

    def test_single_community_graph(self):
        g = ec.Graph()
        for i in range(4):
            g.add_node(label=str(i))
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(source=str(i), target=str(j))
        scores = {nid: 1.0 for nid in g.get_node_ids()}
        result = g.gwas_pipeline(scores)

        assert len(result["communities"]) >= 1
