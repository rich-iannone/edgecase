import random

import edgecase as ec
import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_triangle():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="A", target="B")
        .add_edge(source="B", target="C")
        .add_edge(source="A", target="C")
    )


def make_path():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="A", target="B")
        .add_edge(source="B", target="C")
    )


def make_k4():
    g = ec.Graph()
    for label in "ABCD":
        g.add_node(label=label)
    for i, a in enumerate("ABCD"):
        for b in "ABCD"[i + 1 :]:
            g.add_edge(source=a, target=b)
    return g


def make_star():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge(source="A", target="B")
        .add_edge(source="A", target="C")
        .add_edge(source="A", target="D")
    )


def make_cycle4():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge(source="A", target="B")
        .add_edge(source="B", target="C")
        .add_edge(source="C", target="D")
        .add_edge(source="D", target="A")
    )


def make_triangle_with_tail():
    return (
        ec.Graph()
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_node(label="D")
        .add_edge(source="A", target="B")
        .add_edge(source="B", target="C")
        .add_edge(source="A", target="C")
        .add_edge(source="C", target="D")
    )


def make_directed_triangle():
    return (
        ec.Graph(directed=True)
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="A", target="B")
        .add_edge(source="B", target="C")
        .add_edge(source="A", target="C")
    )


def make_directed_cycle():
    return (
        ec.Graph(directed=True)
        .add_node(label="A")
        .add_node(label="B")
        .add_node(label="C")
        .add_edge(source="A", target="B")
        .add_edge(source="B", target="C")
        .add_edge(source="C", target="A")
    )


def make_two_cliques_bridge():
    """Two 5-cliques connected by a single bridge edge."""
    g = ec.Graph()
    for i in range(10):
        g.add_node(label=str(i))
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(source=str(i), target=str(j))
    for i in range(5, 10):
        for j in range(i + 1, 10):
            g.add_edge(source=str(i), target=str(j))
    g.add_edge(source="4", target="5")
    return g


# ===========================================================================
# motif_census — size 3
# ===========================================================================


class TestMotifCensusSize3:
    def test_pure_triangle(self):
        census = make_triangle().motif_census(size=3, return_dict=True)
        assert census["triangle"] == 1
        assert "path" not in census

    def test_pure_path(self):
        census = make_path().motif_census(size=3, return_dict=True)
        assert census["path"] == 1
        assert "triangle" not in census

    def test_triangle_plus_tail(self):
        census = make_triangle_with_tail().motif_census(size=3, return_dict=True)
        assert census["triangle"] == 1
        assert census["path"] == 2

    def test_k4_all_triangles(self):
        census = make_k4().motif_census(size=3, return_dict=True)
        assert census["triangle"] == 4

    def test_star_all_paths(self):
        census = make_star().motif_census(size=3, return_dict=True)
        assert census["path"] == 3
        assert "triangle" not in census

    def test_cycle4_paths(self):
        census = make_cycle4().motif_census(size=3, return_dict=True)
        assert census["path"] == 4
        assert "triangle" not in census

    def test_empty_graph(self):
        g = ec.Graph()
        census = g.motif_census(size=3, return_dict=True)
        assert census == {}

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        census = g.motif_census(size=3, return_dict=True)
        assert census == {}

    def test_two_nodes(self):
        g = ec.Graph().add_node(label="A").add_node(label="B").add_edge(source="A", target="B")
        census = g.motif_census(size=3, return_dict=True)
        assert census == {}

    def test_disconnected_pair_and_singleton(self):
        g = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
        )
        census = g.motif_census(size=3, return_dict=True)
        assert census == {}

    def test_two_cliques_bridge(self):
        g = make_two_cliques_bridge()
        census = g.motif_census(size=3, return_dict=True)
        # Each 5-clique has C(5,3) = 10 triangles
        assert census["triangle"] == 20
        # Paths crossing the bridge: center nodes and bridge neighbors
        assert census["path"] > 0

    def test_total_count_triangle_tail(self):
        census = make_triangle_with_tail().motif_census(size=3, return_dict=True)
        total = sum(census.values())
        assert total == 3  # 1 triangle + 2 paths


# ===========================================================================
# motif_census — size 4
# ===========================================================================


class TestMotifCensusSize4:
    def test_k4_single_clique(self):
        census = make_k4().motif_census(size=4, return_dict=True)
        assert census["clique_4"] == 1

    def test_star_single_motif(self):
        census = make_star().motif_census(size=4, return_dict=True)
        assert len(census) == 1
        total = sum(census.values())
        assert total == 1

    def test_cycle4_single_motif(self):
        census = make_cycle4().motif_census(size=4, return_dict=True)
        assert len(census) == 1
        total = sum(census.values())
        assert total == 1

    def test_too_few_nodes(self):
        census = make_triangle().motif_census(size=4, return_dict=True)
        assert census == {}

    def test_empty_graph(self):
        g = ec.Graph()
        census = g.motif_census(size=4, return_dict=True)
        assert census == {}

    def test_k4_plus_pendant(self):
        g = make_k4()
        g.add_node(label="E")
        g.add_edge(source="D", target="E")
        census = g.motif_census(size=4, return_dict=True)
        assert census["clique_4"] == 1
        total = sum(census.values())
        # K4 + pendant: {A,B,C,D}=clique, then {A,B,D,E}, {A,C,D,E}, {B,C,D,E}
        assert total > 1

    def test_path_of_4(self):
        g = (
            ec.Graph()
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
        )
        census = g.motif_census(size=4, return_dict=True)
        total = sum(census.values())
        assert total == 1  # Only one connected 4-subgraph (the whole graph)


# ===========================================================================
# motif_census — directed graphs
# ===========================================================================


class TestMotifCensusDirected:
    def test_directed_triangle_and_cycle_differ(self):
        tri = make_directed_triangle().motif_census(size=3, return_dict=True)
        cyc = make_directed_cycle().motif_census(size=3, return_dict=True)
        # Both have 1 motif but the motif class should differ
        assert sum(tri.values()) == 1
        assert sum(cyc.values()) == 1
        tri_keys = set(tri.keys())
        cyc_keys = set(cyc.keys())
        assert tri_keys != cyc_keys

    def test_directed_feedforward_loop(self):
        g = (
            ec.Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="A", target="C")
            .add_edge(source="B", target="C")
        )
        census = g.motif_census(size=3, return_dict=True)
        assert sum(census.values()) == 1

    def test_directed_empty(self):
        g = ec.Graph(directed=True)
        census = g.motif_census(size=3, return_dict=True)
        assert census == {}

    def test_directed_two_nodes(self):
        g = (
            ec.Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_edge(source="A", target="B")
        )
        census = g.motif_census(size=3, return_dict=True)
        assert census == {}

    def test_directed_mutual_edge(self):
        g = (
            ec.Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="A")
            .add_edge(source="B", target="C")
        )
        census = g.motif_census(size=3, return_dict=True)
        assert sum(census.values()) == 1

    def test_directed_four_node(self):
        g = (
            ec.Graph(directed=True)
            .add_node(label="A")
            .add_node(label="B")
            .add_node(label="C")
            .add_node(label="D")
            .add_edge(source="A", target="B")
            .add_edge(source="B", target="C")
            .add_edge(source="C", target="D")
            .add_edge(source="D", target="A")
        )
        census = g.motif_census(size=4, return_dict=True)
        assert sum(census.values()) == 1


# ===========================================================================
# motif_census — error handling
# ===========================================================================


class TestMotifCensusErrors:
    def test_size_too_small(self):
        g = make_triangle()
        with pytest.raises(ValueError, match="3 or 4"):
            g.motif_census(size=2, return_dict=True)

    def test_size_too_large(self):
        g = make_triangle()
        with pytest.raises(ValueError, match="3 or 4"):
            g.motif_census(size=5, return_dict=True)

    def test_size_one(self):
        g = make_triangle()
        with pytest.raises(ValueError, match="3 or 4"):
            g.motif_census(size=1, return_dict=True)


# ===========================================================================
# motif_census — chaining and return_dict
# ===========================================================================


class TestMotifCensusChaining:
    def test_return_self(self):
        g = make_triangle()
        result = g.motif_census(size=3)
        assert result is g

    def test_return_dict(self):
        g = make_triangle()
        result = g.motif_census(size=3, return_dict=True)
        assert isinstance(result, dict)

    def test_computations_tracked(self):
        g = make_triangle()
        g.motif_census(size=3)
        found = any(c[0] == "motif_census" for c in g._computations)
        assert found

    def test_computations_not_tracked_with_return_dict(self):
        g = make_triangle()
        before = len(g._computations)
        g.motif_census(size=3, return_dict=True)
        assert len(g._computations) == before


# ===========================================================================
# motif_significance — basic functionality
# ===========================================================================


class TestMotifSignificanceBasic:
    def test_returns_dict(self):
        g = make_triangle_with_tail()
        result = g.motif_significance(size=3, n_random=5, seed=23, return_dict=True)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_keys_present(self):
        g = make_triangle_with_tail()
        result = g.motif_significance(size=3, n_random=5, seed=23, return_dict=True)
        for _class, stats in result.items():
            assert "count" in stats
            assert "mean_random" in stats
            assert "std_random" in stats
            assert "z_score" in stats
            assert "p_value" in stats

    def test_count_matches_census(self):
        g = make_triangle_with_tail()
        census = g.motif_census(size=3, return_dict=True)
        sig = g.motif_significance(size=3, n_random=5, seed=23, return_dict=True)
        for motif_class, count in census.items():
            assert sig[motif_class]["count"] == float(count)

    def test_p_value_range(self):
        g = make_triangle_with_tail()
        result = g.motif_significance(size=3, n_random=10, seed=23, return_dict=True)
        for stats in result.values():
            assert 0.0 <= stats["p_value"] <= 1.0

    def test_std_non_negative(self):
        g = make_triangle_with_tail()
        result = g.motif_significance(size=3, n_random=10, seed=23, return_dict=True)
        for stats in result.values():
            assert stats["std_random"] >= 0.0


# ===========================================================================
# motif_significance — statistical behavior
# ===========================================================================


class TestMotifSignificanceStatistical:
    def test_clustered_graph_triangle_significant(self):
        g = make_two_cliques_bridge()
        sig = g.motif_significance(size=3, n_random=20, seed=23, return_dict=True)
        # Triangles should be significantly overrepresented
        assert sig["triangle"]["z_score"] > 2.0
        assert sig["triangle"]["p_value"] < 0.05

    def test_clustered_graph_paths_underrepresented(self):
        g = make_two_cliques_bridge()
        sig = g.motif_significance(size=3, n_random=20, seed=23, return_dict=True)
        # Paths should be underrepresented
        assert sig["path"]["z_score"] < -2.0

    def test_seed_reproducibility(self):
        g = make_triangle_with_tail()
        r1 = g.motif_significance(size=3, n_random=10, seed=23, return_dict=True)
        r2 = g.motif_significance(size=3, n_random=10, seed=23, return_dict=True)
        for cls in r1:
            assert r1[cls]["z_score"] == r2[cls]["z_score"]
            assert r1[cls]["p_value"] == r2[cls]["p_value"]

    def test_different_seeds_may_differ(self):
        g = make_two_cliques_bridge()
        r1 = g.motif_significance(size=3, n_random=5, seed=1, return_dict=True)
        r2 = g.motif_significance(size=3, n_random=5, seed=999, return_dict=True)
        # With different seeds, at least mean_random values may differ
        any_diff = False
        for cls in r1:
            if cls in r2 and r1[cls]["mean_random"] != r2[cls]["mean_random"]:
                any_diff = True
        assert any_diff


# ===========================================================================
# motif_significance — error handling
# ===========================================================================


class TestMotifSignificanceErrors:
    def test_invalid_size(self):
        g = make_triangle()
        with pytest.raises(ValueError, match="3 or 4"):
            g.motif_significance(size=5, n_random=5, return_dict=True)

    def test_zero_random(self):
        g = make_triangle()
        with pytest.raises(ValueError, match="n_random"):
            g.motif_significance(size=3, n_random=0, return_dict=True)


# ===========================================================================
# motif_significance — chaining
# ===========================================================================


class TestMotifSignificanceChaining:
    def test_return_self(self):
        g = make_triangle_with_tail()
        result = g.motif_significance(size=3, n_random=5, seed=23)
        assert result is g

    def test_return_dict(self):
        g = make_triangle_with_tail()
        result = g.motif_significance(size=3, n_random=5, seed=23, return_dict=True)
        assert isinstance(result, dict)

    def test_computations_tracked(self):
        g = make_triangle_with_tail()
        g.motif_significance(size=3, n_random=5, seed=23)
        found = any(c[0] == "motif_significance" for c in g._computations)
        assert found


# ===========================================================================
# motif_significance — edge cases
# ===========================================================================


class TestMotifSignificanceEdgeCases:
    def test_empty_graph(self):
        g = ec.Graph()
        result = g.motif_significance(size=3, n_random=5, seed=23, return_dict=True)
        assert result == {}

    def test_graph_too_small(self):
        g = ec.Graph().add_node(label="A").add_node(label="B").add_edge(source="A", target="B")
        result = g.motif_significance(size=3, n_random=5, seed=23, return_dict=True)
        assert result == {}

    def test_size_4(self):
        g = make_k4()
        g.add_node(label="E")
        g.add_edge(source="D", target="E")
        result = g.motif_significance(size=4, n_random=5, seed=23, return_dict=True)
        assert len(result) > 0
        for stats in result.values():
            assert "count" in stats
            assert "z_score" in stats

    def test_no_seed(self):
        g = make_triangle_with_tail()
        result = g.motif_significance(size=3, n_random=5, return_dict=True)
        assert isinstance(result, dict)
        assert len(result) > 0


# ===========================================================================
# motif_census — larger graphs
# ===========================================================================


class TestMotifCensusLarger:
    def test_petersen_like(self):
        """5-cycle with pentagram inside (Petersen-like)."""
        g = ec.Graph()
        for i in range(10):
            g.add_node(label=str(i))
        # Outer 5-cycle
        for i in range(5):
            g.add_edge(source=str(i), target=str((i + 1) % 5))
        # Inner pentagram
        for i in range(5):
            g.add_edge(source=str(i), target=str(i + 5))
        for i in range(5):
            g.add_edge(source=str(i + 5), target=str((i + 2) % 5 + 5))
        census = g.motif_census(size=3, return_dict=True)
        total = sum(census.values())
        assert total > 0

    def test_complete_bipartite_k33(self):
        """K3,3 has no triangles (bipartite)."""
        g = ec.Graph()
        for i in range(6):
            g.add_node(label=str(i))
        for i in range(3):
            for j in range(3, 6):
                g.add_edge(source=str(i), target=str(j))
        census = g.motif_census(size=3, return_dict=True)
        assert "triangle" not in census
        # All 3-node subgraphs should be paths
        assert census.get("path", 0) > 0

    def test_20_node_chain(self):
        g = ec.Graph()
        for i in range(20):
            g.add_node(label=str(i))
        for i in range(19):
            g.add_edge(source=str(i), target=str(i + 1))
        census = g.motif_census(size=3, return_dict=True)
        assert census["path"] == 18  # Each internal node forms a path with its neighbors
        assert "triangle" not in census


# Directed triadic census, validated against NetworkX.


def _random(seed: int, directed: bool, n: int = 12):
    rnd = random.Random(seed)
    G0 = nx.gnp_random_graph(n, rnd.choice([0.2, 0.3, 0.45]), seed=seed, directed=directed)
    G = (nx.DiGraph if directed else nx.Graph)()
    G.add_nodes_from(range(n))
    for u, v in G0.edges:
        G.add_edge(u, v, capacity=rnd.choice([1, 2, 3, 5]))
    g = ec.Graph(directed=directed).add_nodes([str(v) for v in range(n)])
    for u, v, d in G.edges(data=True):
        g.add_edge(str(u), str(v), capacity=d["capacity"])
    return G, g


class TestTriadicCensus:
    def test_matches_networkx(self):
        for seed in range(20):
            G, g = _random(seed, True)
            assert g.triadic_census() == nx.triadic_census(G)
        D = nx.gnp_random_graph(60, 0.08, seed=1, directed=True)
        big = ec.Graph(directed=True).add_nodes([str(v) for v in D])
        big.add_edges([(str(u), str(v)) for u, v in D.edges])
        assert big.triadic_census() == nx.triadic_census(D)

    def test_order_total_and_examples(self):
        cycle = ec.Graph(directed=True).add_nodes(list("ABC"))
        cycle.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        census = cycle.triadic_census()
        assert list(census) == [
            "003", "012", "102", "021D", "021U", "021C", "111D", "111U",
            "030T", "030C", "201", "120D", "120U", "120C", "210", "300",
        ]  # fmt: skip
        assert census["030C"] == 1 and sum(census.values()) == 1
        g = ec.Graph(directed=True).add_nodes(list("ABCDE"))
        assert g.triadic_census()["003"] == 10

    def test_parallel_edges_and_undirected(self):
        g = ec.Graph(directed=True).add_nodes(list("ABC"))
        g.add_edges([("A", "B"), ("A", "B"), ("B", "B")])
        assert g.triadic_census()["012"] == 1
        with pytest.raises(ValueError, match="directed"):
            ec.Graph().add_nodes(list("ABC")).triadic_census()

    def test_triadic_census_step(self):
        cycle = ec.Graph(directed=True).add_nodes(list("ABC"))
        cycle.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        cycle.triadic_census()
        step = cycle.steps[-1]
        assert step.narration == "Triadic census of 1 triads; most common connected types: 030C ×1."
