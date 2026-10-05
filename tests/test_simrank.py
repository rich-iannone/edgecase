"""Tests for SimRank and SimRank* similarity."""

import random

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_METRIC


@pytest.fixture
def diamond_directed():
    """Diamond DAG: 1->2, 1->3, 2->4, 3->4."""
    g = ec.Graph(directed=True)
    for i in range(1, 5):
        g.add_node(label=str(i))
    g.add_edge("1", "2").add_edge("1", "3")
    g.add_edge("2", "4").add_edge("3", "4")
    return g


@pytest.fixture
def simple_directed():
    """Two nodes pointing to a third: 1->3, 2->3."""
    g = ec.Graph(directed=True)
    for i in range(1, 4):
        g.add_node(label=str(i))
    g.add_edge("1", "3").add_edge("2", "3")
    return g


@pytest.fixture
def triangle_undirected():
    """Triangle: 1-2, 2-3, 3-1."""
    g = ec.Graph(directed=False)
    for i in range(1, 4):
        g.add_node(label=str(i))
    g.add_edge("1", "2").add_edge("2", "3").add_edge("3", "1")
    return g


@pytest.fixture
def chain_directed():
    """Chain: 1->2->3->4->5."""
    g = ec.Graph(directed=True)
    for i in range(1, 6):
        g.add_node(label=str(i))
    for i in range(1, 5):
        g.add_edge(str(i), str(i + 1))
    return g


class TestSimRankStarSimilarity:
    def test_returns_dict(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        assert isinstance(result, dict)
        assert len(result) == 4

    def test_self_similarity_is_one(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        for nid, inner in result.items():
            assert abs(inner.get(nid, 0.0) - 1.0) < 1e-10

    def test_symmetric_nodes_positive(self):
        """Nodes 1 and 2 are both linked from 3; they should have positive similarity."""
        g = ec.Graph(directed=True)
        for i in range(1, 4):
            g.add_node(label=str(i))
        g.add_edge("3", "1").add_edge("3", "2")
        result = g.simrank_star_similarity(return_dict=True)
        id1 = g.get_id_from_label("1")
        id2 = g.get_id_from_label("2")
        sim = result[id1].get(id2, 0.0)
        assert sim > 0.0, f"Expected positive similarity, got {sim}"

    def test_shared_out_neighbor_alone_gives_zero(self, simple_directed):
        """SimRank* follows in-links: 1 and 2 have none, so pointing at 3 doesn't make them similar."""
        result = simple_directed.simrank_star_similarity(return_dict=True)
        id1 = simple_directed.get_id_from_label("1")
        id2 = simple_directed.get_id_from_label("2")
        assert result[id1].get(id2, 0.0) == 0.0

    def test_siblings_match_yu_et_al(self):
        """Hub -> {P1, P2} -> Q: the siblings' score matches a direct solve of Yu et al.'s equation."""
        np = pytest.importorskip("numpy")
        labels = ["Hub", "P1", "P2", "Q"]
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        g = ec.Graph(directed=True)
        for label in labels:
            g.add_node(label=label)
        for u, v in edges:
            g.add_edge(labels[u], labels[v])

        # S = C/2 (Q S + S Q^T) + (1 - C) I with Q[j][i] = 1/|In(j)| for i in In(j)
        n, c = len(labels), 0.8
        q = np.zeros((n, n))
        for j in range(n):
            ins = [u for u, v in edges if v == j]
            for i in ins:
                q[j, i] = 1 / len(ins)
        eye = np.eye(n)
        a = np.eye(n * n) - c / 2 * (np.kron(eye, q) + np.kron(q, eye))
        s = np.linalg.solve(a, ((1 - c) * eye).flatten(order="F")).reshape((n, n), order="F")
        expected = s[1, 2] / np.sqrt(s[1, 1] * s[2, 2])

        p1, p2 = g.get_id_from_label("P1"), g.get_id_from_label("P2")
        for method in ["closed_form", "iterative"]:
            result = g.simrank_star_similarity(
                return_dict=True, method=method, max_iterations=1000, tolerance=1e-12
            )
            assert result[p1][p2] == pytest.approx(expected, abs=1e-9)

    def test_closed_form_matches_iterative_on_directed_graphs(self):
        """Non-symmetric Q gives complex Schur blocks; both solvers must still agree."""
        import random

        rng = random.Random(2)
        for _ in range(30):
            n = rng.randint(3, 8)
            g = ec.Graph(directed=True)
            for i in range(n):
                g.add_node(label=str(i))
            for u in range(n):
                for v in range(n):
                    if u != v and rng.random() < 0.35:
                        g.add_edge(str(u), str(v))
            closed = g.simrank_star_similarity(return_dict=True)
            iterative = g.simrank_star_similarity(
                return_dict=True, method="iterative", max_iterations=2000, tolerance=1e-13
            )
            for a, row in iterative.items():
                for b, value in row.items():
                    assert closed[a].get(b, 0.0) == pytest.approx(value, abs=1e-9)

    def test_default_storage_works_with_node_df(self, simple_directed):
        """return_dict=False stores records that get_node_df() can turn into a column."""
        simple_directed.simrank_star_similarity()

        df = simple_directed.get_node_df()
        assert "simrank_star" in df.columns
        id3 = simple_directed.get_id_from_label("3")
        row = df.filter(df["id"] == id3)["simrank_star"].to_list()[0]
        assert row[0] == {"node": id3, "similarity": 1.0}
        similarities = [record["similarity"] for record in row]
        assert similarities == sorted(similarities, reverse=True)

    def test_diamond_nodes_2_3_similar(self, diamond_directed):
        """Nodes 2 and 3 share in-neighbor (1) and out-neighbor (4)."""
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        id2 = diamond_directed.get_id_from_label("2")
        id3 = diamond_directed.get_id_from_label("3")
        sim = result[id2].get(id3, 0.0)
        assert sim > 0.0, f"Expected positive similarity for symmetric nodes, got {sim}"

    def test_values_in_unit_range(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        for inner in result.values():
            for val in inner.values():
                assert -1e-10 <= val <= 1.0 + 1e-10, f"Value out of range: {val}"

    def test_symmetry(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity(return_dict=True)
        ids = list(result.keys())
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                val_ab = result[a].get(b, 0.0)
                val_ba = result[b].get(a, 0.0)
                assert abs(val_ab - val_ba) < 1e-10, (
                    f"Not symmetric: sim({a},{b})={val_ab} vs sim({b},{a})={val_ba}"
                )

    def test_methods_agree(self, diamond_directed):
        cf = diamond_directed.simrank_star_similarity(method="closed_form", return_dict=True)
        it = diamond_directed.simrank_star_similarity(
            method="iterative", max_iterations=500, tolerance=1e-8, return_dict=True
        )
        for a in cf:
            for b in cf[a]:
                v_cf = cf[a][b]
                v_it = it.get(a, {}).get(b, 0.0)
                assert abs(v_cf - v_it) < 1e-3, f"Methods disagree for ({a},{b}): {v_cf} vs {v_it}"

    def test_triangle_equal_similarity(self, triangle_undirected):
        result = triangle_undirected.simrank_star_similarity(return_dict=True)
        ids = sorted(result.keys())
        sim_01 = result[ids[0]].get(ids[1], 0.0)
        sim_02 = result[ids[0]].get(ids[2], 0.0)
        sim_12 = result[ids[1]].get(ids[2], 0.0)
        assert abs(sim_01 - sim_02) < 1e-6
        assert abs(sim_02 - sim_12) < 1e-6

    def test_empty_graph(self):
        g = ec.Graph(directed=True)
        result = g.simrank_star_similarity(return_dict=True)
        assert result == {}

    def test_single_node(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A")
        result = g.simrank_star_similarity(return_dict=True)
        assert len(result) == 1
        nid = list(result.keys())[0]
        assert abs(result[nid][nid] - 1.0) < 1e-10

    def test_disconnected_nodes(self):
        g = ec.Graph(directed=True)
        g.add_node(label="A").add_node(label="B")
        result = g.simrank_star_similarity(return_dict=True)
        id_a = g.get_id_from_label("A")
        id_b = g.get_id_from_label("B")
        sim = result[id_a].get(id_b, 0.0)
        assert sim == 0.0, f"Disconnected nodes should have 0 similarity, got {sim}"

    def test_chain_decay(self, chain_directed):
        """Similarity should decrease with distance."""
        result = chain_directed.simrank_star_similarity(return_dict=True)
        id1 = chain_directed.get_id_from_label("1")
        id2 = chain_directed.get_id_from_label("2")
        id5 = chain_directed.get_id_from_label("5")
        sim_12 = result[id1].get(id2, 0.0)
        sim_15 = result[id1].get(id5, 0.0)
        assert sim_12 >= sim_15, (
            f"Closer nodes should be more similar: sim(1,2)={sim_12}, sim(1,5)={sim_15}"
        )


class TestSimRankStarSimilarityPair:
    def test_pair_matches_full(self, diamond_directed):
        full = diamond_directed.simrank_star_similarity(return_dict=True)
        id2 = diamond_directed.get_id_from_label("2")
        id3 = diamond_directed.get_id_from_label("3")
        pair = diamond_directed.simrank_star_similarity_pair("2", "3")
        expected = full[id2].get(id3, 0.0)
        assert abs(pair - expected) < 1e-10

    def test_self_pair(self, diamond_directed):
        assert diamond_directed.simrank_star_similarity_pair("1", "1") == 1.0

    def test_invalid_node(self, diamond_directed):
        with pytest.raises(Exception):
            diamond_directed.simrank_star_similarity_pair("1", "999")


class TestSimRankStarGraphMethod:
    def test_returns_graph(self, diamond_directed):
        result = diamond_directed.simrank_star_similarity()
        assert isinstance(result, ec.Graph)

    def test_does_not_mutate_structure(self, diamond_directed):
        n = diamond_directed.count_nodes()
        e = diamond_directed.count_edges()
        diamond_directed.simrank_star_similarity()
        assert diamond_directed.count_nodes() == n
        assert diamond_directed.count_edges() == e

    def test_importance_factor_validation(self, diamond_directed):
        with pytest.raises(Exception):
            diamond_directed.simrank_star_similarity(importance_factor=1.0, return_dict=True)
        with pytest.raises(Exception):
            diamond_directed.simrank_star_similarity(importance_factor=-0.1, return_dict=True)

    def test_invalid_method(self, diamond_directed):
        with pytest.raises(Exception):
            diamond_directed.simrank_star_similarity(method="bad", return_dict=True)

    def test_different_importance_factors(self, diamond_directed):
        low = diamond_directed.simrank_star_similarity(importance_factor=0.3, return_dict=True)
        high = diamond_directed.simrank_star_similarity(importance_factor=0.9, return_dict=True)
        id2 = diamond_directed.get_id_from_label("2")
        id3 = diamond_directed.get_id_from_label("3")
        assert low[id2].get(id3, 0.0) < high[id2].get(id3, 0.0), (
            "Higher importance factor should give higher similarity"
        )


class TestPlainSimRank:
    """SimRank (Jeh and Widom), validated against NetworkX's simrank_similarity."""

    def test_against_networkx(self):
        for seed in range(50):
            rnd = random.Random(seed)
            directed = seed % 2 == 0
            weighted = seed % 3 == 0
            G = nx.gnp_random_graph(rnd.randint(1, 14), 0.3, seed=seed, directed=directed)
            for u, v in G.edges:
                G[u][v]["weight"] = rnd.uniform(0.5, 3) if weighted else 1
            g = ec.Graph(directed=directed, weighted=weighted).add_nodes([str(n) for n in G])
            for u, v in G.edges:
                g.add_edge(str(u), str(v), weight=G[u][v]["weight"])
            try:
                ref = nx.simrank_similarity(G)
            except nx.ExceededMaxIterations:
                continue
            label = {g.get_id_from_label(str(n)): n for n in G}
            for a, row in g.simrank_similarity().items():
                for b, value in row.items():
                    assert value == pytest.approx(ref[label[a]][label[b]], abs=1e-9)

    def test_source_target_and_steps(self):
        g = ec.Graph().add_nodes(["ann", "bob", "cy", "math", "art", "bio"])
        g.add_edges(
            [("ann", "math"), ("ann", "art"), ("bob", "math"), ("bob", "art"), ("cy", "bio")]
        )
        assert g.simrank_similarity("ann", "bob") == pytest.approx(0.818, abs=1e-3)
        assert g.simrank_similarity("ann", "cy") == 0.0
        row = g.simrank_similarity("math")
        assert row[g.get_id_from_label("math")] == 1.0
        step = g.steps[-1]
        assert step.kind == KIND_METRIC and step.delta.nodes_selected == ["math"]
        with pytest.raises(ValueError, match="converge"):
            g.simrank_similarity(max_iterations=1)
