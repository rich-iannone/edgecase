"""Tests for null model generators: configuration model, degree-preserving rewire,
expected degree graph, stochastic block model generation."""

import edgecase as ec
import networkx as nx
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_graph():
    g = ec.Graph()
    for label in ["A", "B", "C", "D", "E"]:
        g.add_node(label=label)
    g.add_edge("A", "B").add_edge("B", "C").add_edge("C", "D")
    g.add_edge("D", "E").add_edge("E", "A").add_edge("A", "C")
    return g


@pytest.fixture
def triangle_graph():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge("A", "B")
        .add_edge("B", "C")
        .add_edge("C", "A")
    )


# =============================================================================
# Configuration Model
# =============================================================================


class TestConfigurationModel:
    def test_correct_node_count(self):
        g = ec.Graph()
        cm = g.configuration_model([2, 2, 2], seed=23)
        assert cm.count_nodes() == 3

    def test_correct_edge_count(self):
        g = ec.Graph()
        cm = g.configuration_model([2, 2, 2], seed=23)
        assert cm.count_edges() == 3

    def test_empty_degree_sequence(self):
        g = ec.Graph()
        cm = g.configuration_model([], seed=23)
        assert cm.count_nodes() == 0
        assert cm.count_edges() == 0

    def test_all_zero_degrees(self):
        g = ec.Graph()
        cm = g.configuration_model([0, 0, 0], seed=23)
        assert cm.count_nodes() == 3
        assert cm.count_edges() == 0

    def test_odd_sum_raises(self):
        g = ec.Graph()
        with pytest.raises(Exception):
            g.configuration_model([1, 2, 2], seed=23)

    def test_seed_reproducibility(self):
        g = ec.Graph()
        cm1 = g.configuration_model([2, 2, 2, 2], seed=123)
        cm2 = g.configuration_model([2, 2, 2, 2], seed=123)
        assert cm1.count_edges() == cm2.count_edges()

    def test_directed_graph(self):
        g = ec.Graph()
        cm = g.configuration_model([1, 2, 1], directed=True, seed=23)
        assert cm.count_nodes() == 3

    def test_large_degree_sequence(self):
        g = ec.Graph()
        seq = [4] * 10
        cm = g.configuration_model(seq, seed=23)
        assert cm.count_nodes() == 10
        assert cm.count_edges() == 20

    def test_returns_new_graph(self, simple_graph):
        cm = simple_graph.configuration_model([2, 2], seed=23)
        assert cm is not simple_graph
        assert isinstance(cm, ec.Graph)


# =============================================================================
# Degree-Preserving Rewire
# =============================================================================


class TestDegreePreservingRewire:
    def test_preserves_node_count(self, simple_graph):
        rw = simple_graph.degree_preserving_rewire(seed=23)
        assert rw.count_nodes() == simple_graph.count_nodes()

    def test_preserves_edge_count(self, simple_graph):
        rw = simple_graph.degree_preserving_rewire(seed=23)
        assert rw.count_edges() == simple_graph.count_edges()

    def test_returns_new_graph(self, simple_graph):
        rw = simple_graph.degree_preserving_rewire(seed=23)
        assert rw is not simple_graph

    def test_seed_reproducibility(self, simple_graph):
        rw1 = simple_graph.degree_preserving_rewire(seed=23)
        rw2 = simple_graph.degree_preserving_rewire(seed=23)
        edges1 = sorted(rw1._graph.edges())
        edges2 = sorted(rw2._graph.edges())
        assert edges1 == edges2

    def test_single_edge_graph(self):
        g = ec.Graph().add_node(label="A").add_node(label="B").add_edge("A", "B")
        rw = g.degree_preserving_rewire(seed=23)
        assert rw.count_nodes() == 2
        assert rw.count_edges() == 1

    def test_empty_graph(self):
        g = ec.Graph()
        rw = g.degree_preserving_rewire(seed=23)
        assert rw.count_nodes() == 0
        assert rw.count_edges() == 0

    def test_custom_n_swaps(self, simple_graph):
        rw = simple_graph.degree_preserving_rewire(n_swaps=1, seed=23)
        assert rw.count_edges() == simple_graph.count_edges()

    def test_preserves_degree_sequence(self, simple_graph):
        rw = simple_graph.degree_preserving_rewire(n_swaps=100, seed=23)
        orig_degrees = sorted(simple_graph.degrees(return_dict=True).values())
        rw_degrees = sorted(rw.degrees(return_dict=True).values())
        assert orig_degrees == rw_degrees


# =============================================================================
# Expected Degree Graph (Chung-Lu)
# =============================================================================


class TestExpectedDegreeGraph:
    def test_correct_node_count(self):
        g = ec.Graph()
        edg = g.expected_degree_graph([3.0, 3.0, 2.0, 2.0], seed=23)
        assert edg.count_nodes() == 4

    def test_empty_weights(self):
        g = ec.Graph()
        edg = g.expected_degree_graph([], seed=23)
        assert edg.count_nodes() == 0

    def test_negative_weight_raises(self):
        g = ec.Graph()
        with pytest.raises(Exception):
            g.expected_degree_graph([-1.0, 2.0], seed=23)

    def test_all_zero_weights_raises(self):
        g = ec.Graph()
        with pytest.raises(Exception):
            g.expected_degree_graph([0.0, 0.0], seed=23)

    def test_seed_reproducibility(self):
        g = ec.Graph()
        e1 = g.expected_degree_graph([3.0, 3.0, 2.0], seed=99)
        e2 = g.expected_degree_graph([3.0, 3.0, 2.0], seed=99)
        assert e1.count_edges() == e2.count_edges()

    def test_high_weights_produce_more_edges(self):
        g = ec.Graph()
        sparse = g.expected_degree_graph([0.5, 0.5, 0.5, 0.5], seed=23)
        dense = g.expected_degree_graph([10.0, 10.0, 10.0, 10.0], seed=23)
        assert dense.count_edges() >= sparse.count_edges()

    def test_returns_undirected(self):
        g = ec.Graph()
        edg = g.expected_degree_graph([2.0, 2.0, 2.0], seed=23)
        assert not edg.is_directed()

    def test_returns_new_graph(self, simple_graph):
        edg = simple_graph.expected_degree_graph([2.0, 2.0], seed=23)
        assert edg is not simple_graph


# =============================================================================
# Stochastic Block Model Generation
# =============================================================================


class TestGenerateSBM:
    def test_correct_node_count(self):
        g = ec.Graph()
        sbm = g.generate_stochastic_block_model([5, 5], [0.8, 0.1, 0.1, 0.8], seed=23)
        assert sbm.count_nodes() == 10

    def test_empty_sizes(self):
        g = ec.Graph()
        sbm = g.generate_stochastic_block_model([], [], seed=23)
        assert sbm.count_nodes() == 0

    def test_wrong_p_matrix_size_raises(self):
        g = ec.Graph()
        with pytest.raises(Exception):
            g.generate_stochastic_block_model([5, 5], [0.5], seed=23)

    def test_invalid_probability_raises(self):
        g = ec.Graph()
        with pytest.raises(Exception):
            g.generate_stochastic_block_model([3], [1.5], seed=23)

    def test_negative_probability_raises(self):
        g = ec.Graph()
        with pytest.raises(Exception):
            g.generate_stochastic_block_model([3], [-0.1], seed=23)

    def test_seed_reproducibility(self):
        g = ec.Graph()
        s1 = g.generate_stochastic_block_model([5, 5], [0.8, 0.1, 0.1, 0.8], seed=23)
        s2 = g.generate_stochastic_block_model([5, 5], [0.8, 0.1, 0.1, 0.8], seed=23)
        assert s1.count_edges() == s2.count_edges()

    def test_dense_within_sparse_between(self):
        g = ec.Graph()
        sbm = g.generate_stochastic_block_model([10, 10], [1.0, 0.0, 0.0, 1.0], seed=23)
        # With p_within=1.0, each block of 10 has C(10,2)=45 edges
        assert sbm.count_edges() == 90

    def test_single_block(self):
        g = ec.Graph()
        sbm = g.generate_stochastic_block_model([5], [0.8], seed=23)
        assert sbm.count_nodes() == 5

    def test_three_blocks(self):
        g = ec.Graph()
        p = [0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8]
        sbm = g.generate_stochastic_block_model([4, 4, 4], p, seed=23)
        assert sbm.count_nodes() == 12

    def test_zero_probability_no_edges(self):
        g = ec.Graph()
        sbm = g.generate_stochastic_block_model([5, 5], [0.0, 0.0, 0.0, 0.0], seed=23)
        assert sbm.count_edges() == 0


class TestSmallWorldReferences:
    """Degree-preserving random and lattice references and the sigma / omega coefficients."""

    @staticmethod
    def _degrees(g):
        return sorted(g.get_degree(n) for n in g.get_node_ids(limit=None))

    def test_references_preserve_degrees_and_connectivity(self):
        g = ec.Graph().add_smallworld_graph(n=40, k=4, p=0.2, seed=1)
        for method in (g.random_reference, g.lattice_reference):
            r = method(niter=3, seed=2)
            assert self._degrees(r) == self._degrees(g) and r.is_connected()
            assert sorted(r.get_node_ids(limit=None)) == sorted(g.get_node_ids(limit=None))
            again = method(niter=3, seed=2)
            assert r.get_edges_str_repr() == again.get_edges_str_repr()

    def test_lattice_raises_and_random_lowers_clustering(self):
        ws = ec.Graph().add_smallworld_graph(n=60, k=6, p=0.05, seed=1)
        er = ec.Graph().add_gnp_graph(n=60, p=0.12, seed=2)
        assert ws.random_reference(niter=5, seed=3).get_avg_clustering_coefficient() < (
            ws.get_avg_clustering_coefficient()
        )
        if er.is_connected():
            assert er.lattice_reference(niter=10, seed=4).get_avg_clustering_coefficient() > (
                er.get_avg_clustering_coefficient()
            )

    def test_sigma_and_omega_agree_with_networkx_in_range(self):
        G = nx.connected_watts_strogatz_graph(80, 6, 0.1, seed=1)
        g = ec.Graph().add_nodes([str(n) for n in G])
        g.add_edges([(str(u), str(v)) for u, v in G.edges])
        sigma = g.small_world_sigma(niter=10, nrand=5, seed=3)
        # NetworkX reference values (nx.sigma / nx.omega with the same arguments, seed=3), which
        # take ~20 s to recompute: sigma 5.730, omega -0.2423
        assert sigma == pytest.approx(5.730, rel=0.2)
        assert sigma > 1
        omega = g.small_world_omega(niter=5, nrand=5, seed=3)
        assert omega == pytest.approx(-0.2423, abs=0.15)
        assert -1 < omega < 1

    def test_errors(self):
        with pytest.raises(ValueError, match="four nodes"):
            ec.Graph().add_path_graph(3).random_reference()
        split = ec.Graph().add_cycle_graph(4).add_cycle_graph(4)
        with pytest.raises(ValueError, match="connected"):
            split.small_world_sigma(niter=1, nrand=1, seed=1)
