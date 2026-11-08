"""Tests for bipartite graph algorithms: sets, projection, centrality measures."""

import random

import edgecase as ec
import networkx as nx
import pytest
from networkx.algorithms import bipartite as nx_bp

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

    @pytest.mark.parametrize("reverse", [False, True])
    def test_shortest_distances_on_cycle(self, reverse):
        """Distances are true shortest paths, whatever order the edges were added in.

        On the 8-cycle A-x-B-y-C-z-D-w-A, A is 1 hop from x and w and 3 hops from y
        and z, so its closeness is 4 / (1 + 3 + 3 + 1) = 0.5 (a depth-first walk
        around the cycle used to report 5 hops to one of them).
        """
        ring = ["A", "x", "B", "y", "C", "z", "D", "w"]
        edges = [(ring[i], ring[(i + 1) % 8]) for i in range(8)]
        if reverse:
            edges = [(t, s) for s, t in reversed(edges)]

        g = ec.Graph()
        for label in ring:
            g.add_node(label=label)
        for source, target in edges:
            g.add_edge(source, target)

        s0, s1 = g.bipartite_sets()
        which = 0 if g.get_id_from_label("A") in s0 else 1
        cc = g.bipartite_closeness_centrality(which=which, return_dict=True)

        assert len(cc) == 4
        assert all(score == pytest.approx(0.5) for score in cc.values())

    def test_matches_networkx_shortest_paths(self):
        """Scores match the documented formula computed from NetworkX BFS distances."""
        import random

        rng = random.Random(7)
        for seed in range(40):
            reference = nx.bipartite.random_graph(
                rng.randint(2, 8), rng.randint(2, 8), 0.4, seed=seed
            )
            edges = list(reference.edges())
            rng.shuffle(edges)

            g = ec.Graph()
            for node in reference.nodes():
                g.add_node(label=str(node))
            for u, v in edges:
                g.add_edge(str(u), str(v))
            label_of = {g.get_id_from_label(str(node)): node for node in reference.nodes()}

            s0, s1 = g.bipartite_sets()
            for which, (targets, others) in enumerate([(s0, s1), (s1, s0)]):
                cc = g.bipartite_closeness_centrality(which=which, return_dict=True)
                for node_id in targets:
                    dist = nx.single_source_shortest_path_length(reference, label_of[node_id])
                    reached = [dist[label_of[o]] for o in others if label_of[o] in dist]
                    expected = len(reached) / sum(reached) if reached else 0.0
                    assert cc[node_id] == pytest.approx(expected)


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


class TestBipartiteDirected:
    """Regression: on directed graphs the coloring only followed out-edges, so bipartite_sets()
    could raise "not bipartite" when is_bipartite() was True, and the projection and
    centralities only saw outgoing edges."""

    EDGES = [("g2", "d1"), ("g1", "d1"), ("g3", "d2"), ("g2", "d2"), ("d3", "g3")]

    def _pair(self):
        nodes = ["d1", "d2", "d3", "g1", "g2", "g3"]
        directed = ec.Graph(directed=True).add_nodes(nodes).add_edges(self.EDGES)
        undirected = ec.Graph().add_nodes(nodes).add_edges(self.EDGES)
        return directed, undirected

    def test_edge_toward_lower_id_does_not_raise(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B", "C"]).add_edges([("B", "A"), ("C", "B")])
        assert g.is_bipartite()
        a, b, c = (g.get_id_from_label(x) for x in "ABC")
        assert g.bipartite_sets() == ([a, c], [b])

    def test_matches_undirected(self):
        directed, undirected = self._pair()
        assert directed.is_bipartite() and undirected.is_bipartite()
        assert directed.bipartite_sets() == undirected.bipartite_sets()
        for which in (0, 1):
            for method in (
                "bipartite_degree_centrality",
                "bipartite_closeness_centrality",
                "bipartite_betweenness_centrality",
            ):
                d = getattr(directed, method)(which=which, return_dict=True)
                u = getattr(undirected, method)(which=which, return_dict=True)
                assert d == pytest.approx(u), method
            dp = directed.bipartite_projection(which=which)
            up = undirected.bipartite_projection(which=which)
            assert sorted(dp.get_edges_as_tuples()) == sorted(up.get_edges_as_tuples())

    def test_directed_odd_cycle_and_self_loop_not_bipartite(self):
        tri = ec.Graph(directed=True).add_nodes(["A", "B", "C"])
        tri.add_edges([("A", "B"), ("B", "C"), ("A", "C")])
        assert not tri.is_bipartite()
        with pytest.raises(ValueError, match="not bipartite"):
            tri.bipartite_sets()
        loop = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edges([("A", "B"), ("B", "B")])
        assert not loop.is_bipartite()


def _random_bipartite(count=80):
    for seed in range(count):
        rnd = random.Random(seed)
        a, b = rnd.randint(1, 7), rnd.randint(1, 7)
        B = nx_bp.random_graph(a, b, rnd.choice([0.3, 0.5, 0.8]), seed=seed)
        for u, v in B.edges:
            B[u][v]["weight"] = rnd.uniform(0.5, 2)
        g = ec.Graph(weighted=True).add_nodes([str(n) for n in B])
        for u, v, d in B.edges(data=True):
            g.add_edge(str(u), str(v), weight=d["weight"])
        yield B, g, a, rnd


def _relabel(g, d):
    return {int(g.get_label_from_id(k)): v for k, v in d.items()}


class TestBipartiteExtras:
    """Latapy and Robins-Alexander clustering, redundancy, spectral bipartivity, BiRank,
    weighted projections, biadjacency matrices and node-set checks, against NetworkX."""

    def test_clustering_and_redundancy(self):
        for B, g, _, _ in _random_bipartite():
            for mode in ("dot", "min", "max"):
                assert _relabel(g, g.latapy_clustering(mode=mode)) == pytest.approx(
                    nx_bp.latapy_clustering(B, mode=mode)
                )
            assert g.robins_alexander_clustering() == pytest.approx(
                nx_bp.robins_alexander_clustering(B)
            )
            busy = [n for n in B if B.degree(n) >= 2]
            if busy:
                assert _relabel(g, g.node_redundancy([str(n) for n in busy])) == pytest.approx(
                    nx_bp.node_redundancy(B, busy)
                )

    @pytest.mark.skipif(not hasattr(nx_bp, "birank"), reason="NetworkX 3.5+")
    def test_spectral_bipartivity_and_birank(self):
        for B, g, a, _ in _random_bipartite(50):
            for weighted in (False, True):
                assert g.spectral_bipartivity(weighted=weighted) == pytest.approx(
                    nx_bp.spectral_bipartivity(B, weight="weight" if weighted else None)
                )
            top = list(range(a))
            if len(top) < len(B):
                ref = nx_bp.birank(B, top, top_personalization={0: 1.0}, weight="weight")
                ours = g.birank(
                    [str(x) for x in top], top_personalization={"0": 1.0}, weighted=True
                )
                assert _relabel(g, ours) == pytest.approx(ref)
        square = ec.Graph().add_cycle_graph(4)
        assert square.spectral_bipartivity() == pytest.approx(1.0)
        assert ec.Graph().add_cycle_graph(3).spectral_bipartivity() < 1

    def test_projections_matrix_and_node_sets(self):
        for B, g, a, rnd in _random_bipartite():
            side = {int(g.get_label_from_id(x)) for x in g.bipartite_sets()[0]}
            if len(side) < len(B):
                for weighting, ref_fn in [
                    ("count", nx_bp.weighted_projected_graph),
                    ("collaboration", nx_bp.collaboration_weighted_projected_graph),
                    ("jaccard", nx_bp.overlap_weighted_projected_graph),
                    (
                        "overlap",
                        lambda G, n: nx_bp.overlap_weighted_projected_graph(G, n, jaccard=False),
                    ),
                ]:
                    proj = g.bipartite_projection(which=0, weighting=weighting)
                    ours = {
                        frozenset(
                            (int(proj.get_label_from_id(s)), int(proj.get_label_from_id(t)))
                        ): proj.get_edge(e).weight
                        for e, s, t in proj.get_edges_as_tuples()
                    }
                    ref = {
                        frozenset((u, v)): d["weight"]
                        for u, v, d in ref_fn(B, side).edges(data=True)
                    }
                    assert ours.keys() == ref.keys()
                    assert all(ours[k] == pytest.approx(ref[k]) for k in ref)
            matrix, rows, cols = g.biadjacency_matrix(
                row_nodes=[str(x) for x in range(a)], weighted=True
            )
            ref = nx_bp.biadjacency_matrix(
                B, list(range(a)), [int(g.get_label_from_id(c)) for c in cols], weight="weight"
            ).toarray()
            assert sum(matrix, []) == pytest.approx(ref.flatten().tolist())
            for nodes in (list(range(a)), rnd.sample(list(B), max(1, len(B) // 2))):
                assert g.is_bipartite_node_set(
                    [str(x) for x in nodes]
                ) == nx_bp.is_bipartite_node_set(B, nodes)

    def test_examples_and_errors(self):
        g = ec.Graph().add_nodes(["ann", "bob", "cy", "party", "talk", "game"])
        g.add_edges(
            [
                ("ann", "party"),
                ("ann", "talk"),
                ("bob", "party"),
                ("bob", "talk"),
                ("cy", "talk"),
                ("cy", "game"),
            ]
        )
        red = g.node_redundancy(nodes=["ann", "cy"])
        assert list(red.values()) == [1.0, 0.0]
        with pytest.raises(ValueError, match="fewer than two"):
            g.node_redundancy(nodes=["game"])
        with pytest.raises(ValueError, match="mode"):
            g.latapy_clustering(mode="median")
        with pytest.raises(ValueError, match="not bipartite"):
            ec.Graph().add_cycle_graph(3).latapy_clustering()
        with pytest.raises(ValueError, match="duplicates"):
            g.is_bipartite_node_set(["ann", "ann"])
        with pytest.raises(ValueError, match="weighting"):
            g.bipartite_projection(weighting="cosine")
        with pytest.raises(ValueError, match="each side"):
            g.birank(top_nodes=list(g.get_node_ids(limit=None)))
