"""Tests for Girvan-Newman community detection."""

import pytest
import edgecase as ec
import networkx as nx


@pytest.fixture
def two_triangles():
    """Two triangles connected by a single bridge edge."""
    return (
        ec.Graph(directed=False)
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_node(label="E")
        .add_node(label="F")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "A")
        .add_edge("C", "D")  # bridge
        .add_edge("D", "E")
        .add_edge("E", "F")
        .add_edge("F", "D")
    )


@pytest.fixture
def two_cliques():
    """Two 5-cliques connected by a single bridge edge."""
    g = ec.Graph()
    for i in range(10):
        g.add_node(label=f"N{i}")
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(f"N{i}", f"N{j}")
    for i in range(5, 10):
        for j in range(i + 1, 10):
            g.add_edge(f"N{i}", f"N{j}")
    g.add_edge("N4", "N5")
    return g


class TestCmtyGirvanNewman:
    def test_returns_dict(self, two_triangles):
        result = two_triangles.cmty_girvan_newman(return_dict=True)
        assert isinstance(result, dict)
        assert len(result) == 6

    def test_finds_two_communities(self, two_triangles):
        result = two_triangles.cmty_girvan_newman(return_dict=True)
        assert len(set(result.values())) == 2

    def test_correct_split(self, two_triangles):
        result = two_triangles.cmty_girvan_newman(return_dict=True)
        labeled = {two_triangles.get_label_from_id(nid): cid for nid, cid in result.items()}
        # A, B, C should be in the same community
        assert labeled["A"] == labeled["B"] == labeled["C"]
        # D, E, F should be in the same community
        assert labeled["D"] == labeled["E"] == labeled["F"]
        # The two groups should be in different communities
        assert labeled["A"] != labeled["D"]

    def test_chaining_returns_self(self, two_triangles):
        result = two_triangles.cmty_girvan_newman()
        assert result is two_triangles

    def test_stores_attribute(self, two_triangles):
        two_triangles.cmty_girvan_newman()
        ndf = two_triangles.get_node_df()
        assert "cmty_girvan_newman" in ndf.columns

    def test_custom_attr_name(self, two_triangles):
        two_triangles.cmty_girvan_newman(attr_name="my_community")
        ndf = two_triangles.get_node_df()
        assert "my_community" in ndf.columns

    def test_two_cliques(self, two_cliques):
        result = two_cliques.cmty_girvan_newman(return_dict=True)
        assert len(set(result.values())) == 2

        clique1_ids = {two_cliques.get_id_from_label(f"N{i}") for i in range(5)}
        clique2_ids = {two_cliques.get_id_from_label(f"N{i}") for i in range(5, 10)}
        clique1_blocks = {result[nid] for nid in clique1_ids}
        clique2_blocks = {result[nid] for nid in clique2_ids}
        assert len(clique1_blocks) == 1
        assert len(clique2_blocks) == 1
        assert clique1_blocks != clique2_blocks

    def test_empty_graph(self):
        g = ec.Graph()
        result = g.cmty_girvan_newman(return_dict=True)
        assert result == {}

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        result = g.cmty_girvan_newman(return_dict=True)
        assert len(result) == 1

    def test_no_edges(self):
        g = ec.Graph().add_node(label="A").add_node(label="B")
        result = g.cmty_girvan_newman(return_dict=True)
        assert len(result) == 2

    def test_directed_raises(self):
        g = ec.Graph(directed=True)
        with pytest.raises(ValueError, match="undirected"):
            g.cmty_girvan_newman(return_dict=True)

    def test_all_node_ids_present(self, two_triangles):
        result = two_triangles.cmty_girvan_newman(return_dict=True)
        node_ids = set(two_triangles.get_node_ids())
        assert set(result.keys()) == node_ids

    def test_does_not_mutate_graph(self, two_triangles):
        edges_before = two_triangles.count_edges()
        nodes_before = two_triangles.count_nodes()
        two_triangles.cmty_girvan_newman(return_dict=True)
        assert two_triangles.count_edges() == edges_before
        assert two_triangles.count_nodes() == nodes_before

    def test_always_agrees_with_cmty_edge_btwns(self):
        """Both names run one implementation with deterministic tie-breaking."""
        import random

        rng = random.Random(5)
        for _ in range(60):
            n = rng.randint(6, 11)
            ref = nx.gnp_random_graph(n, 0.35, seed=rng.randint(0, 10**6))
            if not nx.is_connected(ref):
                continue
            g = ec.Graph()
            for v in range(n):
                g.add_node(label=str(v))
            for u, v in ref.edges():
                g.add_edge(str(u), str(v))
            assert g.cmty_girvan_newman(return_dict=True) == g.cmty_edge_btwns(return_dict=True)

    def test_communities_numbered_by_smallest_node(self):
        """Community 0 is the one holding the smallest node ID."""
        g = ec.Graph()
        for label in "ABCDEFG":
            g.add_node(label=label)
        for u, v in [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E"), ("D", "F"),
                     ("E", "F"), ("E", "G"), ("F", "G")]:
            g.add_edge(u, v)
        d = g.cmty_girvan_newman(return_dict=True)
        assert d[g.get_id_from_label("A")] == 0
        assert d[g.get_id_from_label("G")] == 1


class TestGirvanNewmanDendrogram:
    """Test the raw dendrogram output from the Rust core."""

    def test_dendrogram_structure(self, two_triangles):
        from edgecase._core import girvan_newman

        dendrogram = girvan_newman(two_triangles._graph)
        assert isinstance(dendrogram, list)
        assert len(dendrogram) >= 2

        # First entry is best-modularity partition
        best = dendrogram[0]
        assert len(best) == 2

    def test_dendrogram_all_nodes_each_level(self, two_triangles):
        from edgecase._core import girvan_newman

        dendrogram = girvan_newman(two_triangles._graph)
        expected_nodes = set(two_triangles.get_node_ids())

        for level in dendrogram:
            level_nodes = set()
            for community in level:
                level_nodes.update(community)
            assert level_nodes == expected_nodes

    def test_dendrogram_last_level_singletons(self, two_triangles):
        from edgecase._core import girvan_newman

        dendrogram = girvan_newman(two_triangles._graph)
        last = dendrogram[-1]
        for community in last:
            assert len(community) == 1
