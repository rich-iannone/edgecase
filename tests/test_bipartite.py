"""Tests for bipartite graph algorithms: sets, projection, centrality measures."""

import pytest
import edgecase as ec


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_bipartite():
    """Gene1-Disease1, Gene1-Disease2, Gene2-Disease1."""
    g = ec.Graph()
    for label in ["Gene1", "Gene2", "Gene3", "Disease1", "Disease2"]:
        g.add_node(label=label)
    g.add_edge("Gene1", "Disease1")
    g.add_edge("Gene1", "Disease2")
    g.add_edge("Gene2", "Disease1")
    g.add_edge("Gene2", "Disease2")
    g.add_edge("Gene3", "Disease1")
    return g


@pytest.fixture
def complete_bipartite():
    """K_{3,2}: every gene connected to every disease."""
    g = ec.Graph()
    genes = ["G1", "G2", "G3"]
    diseases = ["D1", "D2"]
    for label in genes + diseases:
        g.add_node(label=label)
    for gene in genes:
        for disease in diseases:
            g.add_edge(gene, disease)
    return g


@pytest.fixture
def disconnected_bipartite():
    """Two bipartite components with no edges between them."""
    g = ec.Graph()
    for label in ["A1", "A2", "B1", "B2", "X1", "X2", "Y1"]:
        g.add_node(label=label)
    g.add_edge("A1", "B1").add_edge("A2", "B2")
    g.add_edge("X1", "Y1").add_edge("X2", "Y1")
    return g


def _gene_set_index(graph, set_0, set_1, gene_label="Gene1"):
    """Return 0 or 1 depending on which set contains the gene."""
    gene_id = graph.get_id_from_label(gene_label)
    return 0 if gene_id in set_0 else 1


# =============================================================================
# Bipartite Sets
# =============================================================================


class TestBipartiteSets:
    def test_returns_two_sets(self, simple_bipartite):
        s0, s1 = simple_bipartite.bipartite_sets()
        assert len(s0) + len(s1) == 5

    def test_sets_are_disjoint(self, simple_bipartite):
        s0, s1 = simple_bipartite.bipartite_sets()
        assert len(set(s0) & set(s1)) == 0

    def test_partitions_correct_sizes(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        sizes = sorted([len(s0), len(s1)])
        assert sizes == [2, 3]

    def test_non_bipartite_raises(self):
        g = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "A")
        )
        with pytest.raises(Exception):
            g.bipartite_sets()

    def test_empty_graph(self):
        g = ec.Graph()
        s0, s1 = g.bipartite_sets()
        assert s0 == [] and s1 == []

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        s0, s1 = g.bipartite_sets()
        assert len(s0) + len(s1) == 1

    def test_single_edge(self):
        g = ec.Graph().add_node(label="A").add_node(label="B").add_edge("A", "B")
        s0, s1 = g.bipartite_sets()
        assert sorted([len(s0), len(s1)]) == [1, 1]

    def test_disconnected_bipartite(self, disconnected_bipartite):
        s0, s1 = disconnected_bipartite.bipartite_sets()
        assert len(s0) + len(s1) == 7

    def test_complete_bipartite(self, complete_bipartite):
        s0, s1 = complete_bipartite.bipartite_sets()
        sizes = sorted([len(s0), len(s1)])
        assert sizes == [2, 3]


# =============================================================================
# Bipartite Projection
# =============================================================================

    def test_partition_zero_holds_lowest_id(self):
        """Partition 0 is the side containing the lowest node ID, on every call."""
        g = ec.Graph()
        for label in "ABCDEFG":
            g.add_node(label=label)
        for u, v in [("A", "E"), ("B", "E"), ("B", "F"), ("C", "F"), ("D", "F"), ("D", "G")]:
            g.add_edge(u, v)
        for _ in range(3):
            s0, s1 = g.bipartite_sets()
            assert min(s0 + s1) in s0
            assert sorted(g.get_label_from_id(n) for n in s0) == ["A", "B", "C", "D"]


class TestBipartiteProjection:
    def test_gene_projection_node_count(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        proj = g.bipartite_projection(which=gene_set)
        assert proj.count_nodes() == 3

    def test_gene_projection_has_edges(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        proj = g.bipartite_projection(which=gene_set)
        assert proj.count_edges() > 0

    def test_shared_neighbor_weight(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        proj = g.bipartite_projection(which=gene_set)

        # Gene1 and Gene2 share Disease1 and Disease2 -> weight 2
        edf = proj.get_edge_df()
        g1_id = proj.get_id_from_label("Gene1")
        g2_id = proj.get_id_from_label("Gene2")

        edge_weights = {}
        for row in edf.iter_rows(named=True):
            src = row.get("from") or row.get("source")
            tgt = row.get("to") or row.get("target")
            w = row.get("weight", 1.0)
            edge_weights[(src, tgt)] = w
            edge_weights[(tgt, src)] = w

        assert edge_weights.get((g1_id, g2_id), 0) == 2.0

    def test_complete_bipartite_projection(self, complete_bipartite):
        g = complete_bipartite
        s0, s1 = g.bipartite_sets()
        g1_id = g.get_id_from_label("G1")
        gene_set = 0 if g1_id in s0 else 1
        proj = g.bipartite_projection(which=gene_set)
        # K_{3,2} projected onto 3-set: complete graph on 3 nodes
        assert proj.count_nodes() == 3
        assert proj.count_edges() == 3

    def test_disease_projection(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        disease_set = 1 - gene_set
        proj = g.bipartite_projection(which=disease_set)
        assert proj.count_nodes() == 2

    def test_invalid_which_raises(self, simple_bipartite):
        with pytest.raises(Exception):
            simple_bipartite.bipartite_projection(which=2)

    def test_projection_preserves_labels(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        proj = g.bipartite_projection(which=gene_set)
        assert proj.has_node("Gene1")
        assert proj.has_node("Gene2")
        assert proj.has_node("Gene3")


# =============================================================================
# Bipartite Centrality
# =============================================================================


class TestBipartiteDegreeCentrality:
    def test_fully_connected_node_has_centrality_one(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        bc = g.bipartite_degree_centrality(which=gene_set, return_dict=True)
        gene1_id = g.get_id_from_label("Gene1")
        assert bc[gene1_id] == 1.0

    def test_partially_connected_node(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        bc = g.bipartite_degree_centrality(which=gene_set, return_dict=True)
        gene3_id = g.get_id_from_label("Gene3")
        assert bc[gene3_id] == 0.5

    def test_values_in_range(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        bc = g.bipartite_degree_centrality(which=gene_set, return_dict=True)
        assert all(0.0 <= v <= 1.0 for v in bc.values())

    def test_chaining(self, simple_bipartite):
        result = simple_bipartite.bipartite_degree_centrality(which=0)
        assert result is simple_bipartite


class TestBipartiteClosenessCentrality:
    def test_returns_correct_count(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        cc = g.bipartite_closeness_centrality(which=gene_set, return_dict=True)
        assert len(cc) == 3

    def test_values_non_negative(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        cc = g.bipartite_closeness_centrality(which=gene_set, return_dict=True)
        assert all(v >= 0 for v in cc.values())

    def test_fully_connected_higher_closeness(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        cc = g.bipartite_closeness_centrality(which=gene_set, return_dict=True)
        gene1_id = g.get_id_from_label("Gene1")
        gene3_id = g.get_id_from_label("Gene3")
        assert cc[gene1_id] >= cc[gene3_id]


class TestBipartiteBetweennessCentrality:
    def test_returns_correct_count(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        bb = g.bipartite_betweenness_centrality(which=gene_set, return_dict=True)
        assert len(bb) == 3

    def test_values_non_negative(self, simple_bipartite):
        g = simple_bipartite
        s0, s1 = g.bipartite_sets()
        gene_set = _gene_set_index(g, s0, s1)
        bb = g.bipartite_betweenness_centrality(which=gene_set, return_dict=True)
        assert all(v >= 0 for v in bb.values())
