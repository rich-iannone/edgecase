"""Tests for seed module discovery: prize-collecting Steiner tree, seed set expansion."""

import pytest
import edgecase as ec


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ppi_graph():
    g = ec.Graph(weighted=True)
    for label in ["BRCA1", "TP53", "ATM", "CHEK2", "MDM2", "CDK2"]:
        g.add_node(label=label)
    g.add_edge("BRCA1", "ATM", weight=1.0)
    g.add_edge("ATM", "CHEK2", weight=1.0)
    g.add_edge("ATM", "TP53", weight=2.0)
    g.add_edge("TP53", "MDM2", weight=1.0)
    g.add_edge("MDM2", "CDK2", weight=3.0)
    g.add_edge("BRCA1", "CHEK2", weight=1.5)
    return g


@pytest.fixture
def cluster_graph():
    """Two tight clusters connected by a single bridge."""
    g = ec.Graph()
    for label in ["A1", "A2", "A3", "Bridge", "B1", "B2", "B3"]:
        g.add_node(label=label)
    g.add_edge("A1", "A2").add_edge("A2", "A3").add_edge("A3", "A1")
    g.add_edge("A3", "Bridge")
    g.add_edge("Bridge", "B1")
    g.add_edge("B1", "B2").add_edge("B2", "B3").add_edge("B3", "B1")
    return g


@pytest.fixture
def disconnected_graph():
    g = ec.Graph(weighted=True)
    for label in ["A", "B", "X", "Y"]:
        g.add_node(label=label)
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("X", "Y", weight=1.0)
    return g


# =============================================================================
# Prize-Collecting Steiner Tree
# =============================================================================


class TestPrizeCollectingSteinerTree:
    def test_connects_high_prize_nodes(self, ppi_graph):
        tree = ppi_graph.prize_collecting_steiner_tree(
            prizes={"BRCA1": 10.0, "TP53": 8.0, "CHEK2": 9.0}
        )
        assert tree.count_nodes() >= 3
        assert tree.count_edges() >= 2

    def test_excludes_low_prize_distant_nodes(self, ppi_graph):
        edges = ppi_graph.prize_collecting_steiner_tree(
            prizes={"BRCA1": 10.0, "CDK2": 0.5}, edge_penalty=5.0, return_subgraph=False
        )
        connected_labels = set()
        for s, t, _ in edges:
            connected_labels.add(s)
            connected_labels.add(t)
        assert "CDK2" not in connected_labels

    def test_single_prize_node_empty_tree(self, ppi_graph):
        edges = ppi_graph.prize_collecting_steiner_tree(
            prizes={"BRCA1": 10.0}, return_subgraph=False
        )
        assert isinstance(edges, list)
        assert len(edges) == 0

    def test_return_subgraph(self, ppi_graph):
        tree = ppi_graph.prize_collecting_steiner_tree(
            prizes={"BRCA1": 10.0, "CHEK2": 8.0}
        )
        assert isinstance(tree, ec.Graph)
        assert tree.count_edges() >= 1

    def test_return_edge_list(self, ppi_graph):
        edges = ppi_graph.prize_collecting_steiner_tree(
            prizes={"BRCA1": 10.0, "CHEK2": 8.0}, return_subgraph=False
        )
        assert isinstance(edges, list)
        assert all(len(e) == 3 for e in edges)

    def test_nonexistent_node_raises(self, ppi_graph):
        with pytest.raises(Exception):
            ppi_graph.prize_collecting_steiner_tree(prizes={"NONEXISTENT": 5.0})

    def test_empty_prizes_returns_empty_graph(self, ppi_graph):
        tree = ppi_graph.prize_collecting_steiner_tree(prizes={})
        assert isinstance(tree, ec.Graph)
        assert tree.count_nodes() == 0

    def test_high_edge_penalty_shrinks_tree(self, ppi_graph):
        tree_low = ppi_graph.prize_collecting_steiner_tree(
            prizes={"BRCA1": 5.0, "TP53": 5.0, "CDK2": 5.0}, edge_penalty=0.1
        )
        tree_high = ppi_graph.prize_collecting_steiner_tree(
            prizes={"BRCA1": 5.0, "TP53": 5.0, "CDK2": 5.0}, edge_penalty=10.0
        )
        assert tree_low.count_edges() >= tree_high.count_edges()

    def test_disconnected_seeds_partial_tree(self, disconnected_graph):
        edges = disconnected_graph.prize_collecting_steiner_tree(
            prizes={"A": 10.0, "X": 10.0}, return_subgraph=False
        )
        # Seeds are in different components, tree can only reach within components
        connected_labels = set()
        for s, t, _ in edges:
            connected_labels.add(s)
            connected_labels.add(t)
        # Can't connect A and X since they're in different components
        assert not ({"A", "X"}.issubset(connected_labels))


# =============================================================================
# Expand Seed Set
# =============================================================================


class TestExpandSeedSet:
    def test_includes_seeds(self, cluster_graph):
        module = cluster_graph.expand_seed_set(seeds=["A1", "A2"])
        a1_id = cluster_graph.get_id_from_label("A1")
        a2_id = cluster_graph.get_id_from_label("A2")
        assert a1_id in module
        assert a2_id in module

    def test_expands_within_cluster(self, cluster_graph):
        module = cluster_graph.expand_seed_set(seeds=["A1", "A2"])
        a3_id = cluster_graph.get_id_from_label("A3")
        assert a3_id in module

    def test_respects_max_size(self):
        g = ec.Graph()
        for i in range(10):
            g.add_node(label=f"N{i}")
        for i in range(9):
            g.add_edge(f"N{i}", f"N{i+1}")
        module = g.expand_seed_set(seeds=["N0"], max_size=3)
        assert len(module) <= 3

    def test_respects_min_score(self):
        g = (
            ec.Graph()
            .add_node(label="S1")
            .add_node(label="Hub")
            .add_node(label="Far1")
            .add_node(label="Far2")
            .add_node(label="Far3")
            .add_edge("S1", "Hub")
            .add_edge("Hub", "Far1")
            .add_edge("Hub", "Far2")
            .add_edge("Hub", "Far3")
        )
        # Hub has 1 edge to module, 4 total edges -> score = 0.25
        module = g.expand_seed_set(seeds=["S1"], min_score=0.5)
        hub_id = g.get_id_from_label("Hub")
        assert hub_id not in module

    def test_tightly_connected_preferred(self):
        g = (
            ec.Graph()
            .add_node(label="S1")
            .add_node(label="S2")
            .add_node(label="Tight")
            .add_node(label="Loose")
            .add_node(label="LooseNeighbor")
            .add_edge("S1", "Tight")
            .add_edge("S2", "Tight")
            .add_edge("S1", "Loose")
            .add_edge("Loose", "LooseNeighbor")
        )
        # Tight: 2 edges to module / 2 total = 1.0
        # Loose: 1 edge to module / 2 total = 0.5
        module = g.expand_seed_set(seeds=["S1", "S2"], max_size=3)
        tight_id = g.get_id_from_label("Tight")
        assert tight_id in module

    def test_nonexistent_seed_raises(self, ppi_graph):
        with pytest.raises(Exception):
            ppi_graph.expand_seed_set(seeds=["NONEXISTENT"])

    def test_empty_seeds_raises(self, ppi_graph):
        with pytest.raises(Exception):
            ppi_graph.expand_seed_set(seeds=[])

    def test_seed_is_entire_graph(self):
        g = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_edge("A", "B")
        )
        module = g.expand_seed_set(seeds=["A", "B"])
        assert len(module) == 2

    def test_returns_sorted_ids(self, cluster_graph):
        module = cluster_graph.expand_seed_set(seeds=["A1"])
        assert module == sorted(module)
