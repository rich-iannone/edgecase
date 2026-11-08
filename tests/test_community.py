"""Tests for community detection: stochastic block model fitting."""

import random
from collections import defaultdict

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_METRIC
from networkx.algorithms import community as nx_comm

# =============================================================================
# Fixtures
# =============================================================================


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


@pytest.fixture
def sbm_graph():
    """Graph generated from a known 2-block SBM."""
    g = ec.Graph()
    sbm = g.generate_stochastic_block_model([10, 10], [0.9, 0.05, 0.05, 0.9], seed=23)
    return sbm


# =============================================================================
# SBM Fitting
# =============================================================================


class TestCmtyStochasticBlockModel:
    def test_returns_dict(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        assert isinstance(result, dict)
        assert len(result) == 10

    def test_correct_number_of_blocks(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        assert len(set(result.values())) <= 2

    def test_chaining_returns_self(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, seed=23)
        assert result is two_cliques

    def test_stores_attribute(self, two_cliques):
        two_cliques.cmty_stochastic_block_model(k=2, seed=23)
        ndf = two_cliques.get_node_df()
        assert "sbm_block" in ndf.columns

    def test_custom_attr_name(self, two_cliques):
        two_cliques.cmty_stochastic_block_model(k=2, attr_name="my_block", seed=23)
        ndf = two_cliques.get_node_df()
        assert "my_block" in ndf.columns

    def test_recovers_clear_structure(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        clique1_ids = {two_cliques.get_id_from_label(f"N{i}") for i in range(5)}
        clique2_ids = {two_cliques.get_id_from_label(f"N{i}") for i in range(5, 10)}
        clique1_blocks = {result[nid] for nid in clique1_ids}
        clique2_blocks = {result[nid] for nid in clique2_ids}
        # Each clique should be mostly in one block
        assert len(clique1_blocks) <= 2
        assert len(clique2_blocks) <= 2

    def test_sbm_generated_graph(self, sbm_graph):
        result = sbm_graph.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        assert len(result) == 20
        assert len(set(result.values())) <= 2

    def test_three_blocks(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=3, return_dict=True, seed=23)
        assert len(set(result.values())) <= 3

    def test_single_block(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=1, return_dict=True, seed=23)
        assert all(v == 0 for v in result.values())

    def test_empty_graph(self):
        g = ec.Graph()
        result = g.cmty_stochastic_block_model(k=2, return_dict=True)
        assert result == {}

    def test_single_node(self):
        g = ec.Graph().add_node(label="A")
        result = g.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        assert len(result) == 1

    def test_seed_reproducibility(self, two_cliques):
        r1 = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=99)
        r2 = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=99)
        assert r1 == r2

    def test_all_node_ids_present(self, two_cliques):
        result = two_cliques.cmty_stochastic_block_model(k=2, return_dict=True, seed=23)
        node_ids = set(two_cliques.get_node_ids())
        assert set(result.keys()) == node_ids


# Modularity, Leiden, label propagation and k-clique communities, validated against NetworkX.


def _random_cases():
    """(networkx graph, edgecase graph, weight key) over many random configurations."""
    for seed in range(15):
        rnd = random.Random(seed)
        for directed in (False, True):
            for weighted in (False, True):
                G = nx.gnp_random_graph(14, 0.2, seed=seed, directed=directed)
                for u, v in G.edges:
                    G[u][v]["weight"] = rnd.choice([0.5, 1, 2, 3])
                yield G, _build(G, weighted), ("weight" if weighted else None)


def _two_triangles() -> ec.Graph:
    return (
        ec.Graph()
        .add_nodes(list("ABCDEF"))
        .add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
    )


def _build(G: nx.Graph, weighted: bool = False) -> ec.Graph:
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G.nodes])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


def _labels(g: ec.Graph, ids) -> list[str]:
    return [g.get_label_from_id(n) for n in ids]


def _groups(membership: dict[int, int]) -> list[set[int]]:
    out = defaultdict(set)
    for n, c in membership.items():
        out[c].add(n)
    return list(out.values())


def _lab(g: ec.Graph, nid: int) -> int:
    return int(g.get_label_from_id(nid))


class TestModularity:
    def test_matches_networkx(self):
        for G, g, w in _random_cases():
            rnd = random.Random(len(G.edges))
            part = {n: rnd.randrange(3) for n in G.nodes}
            groups = [{n for n in G.nodes if part[n] == k} for k in range(3)]
            groups = [x for x in groups if x]
            for res in (1.0, 0.6):
                ours = g.modularity({str(n): c for n, c in part.items()}, resolution=res)
                ref = nx.community.modularity(G, groups, weight=w or "unused", resolution=res)
                assert ours == pytest.approx(ref, abs=1e-12)

    def test_input_forms_agree(self):
        g = _two_triangles()
        by_list = g.modularity([["A", "B", "C"], ["D", "E", "F"]])
        a, d = g.get_id_from_label("A"), g.get_id_from_label("D")
        by_ids = g.modularity([[a, a + 1, a + 2], [d, d + 1, d + 2]])
        by_dict = g.modularity({lbl: int(lbl in "DEF") for lbl in "ABCDEF"})
        g.set_node_attrs(
            "grp", {g.get_id_from_label(x): ("x" if x in "ABC" else "y") for x in "ABCDEF"}
        )
        by_attr = g.modularity("grp")
        assert by_list == pytest.approx(5 / 14)
        assert by_list == by_ids == by_dict == by_attr

    def test_one_community_scores_zero_and_bad_split_negative(self):
        g = _two_triangles()
        assert g.modularity([list("ABCDEF")]) == pytest.approx(0.0)
        assert g.modularity([["A", "D"], ["B", "E"], ["C", "F"]]) < 0

    def test_invalid_partitions(self):
        g = _two_triangles()
        with pytest.raises(ValueError, match="more than one"):
            g.modularity([["A", "B", "C"], ["C", "D", "E", "F"]])
        with pytest.raises(ValueError, match="missing"):
            g.modularity([["A", "B", "C"]])
        with pytest.raises(ValueError, match="No node attribute"):
            g.modularity("nope")

    def test_edgeless_graph(self):
        assert ec.Graph().add_nodes(["A", "B"]).modularity([["A"], ["B"]]) == 0.0


class TestCommunityDetection:
    @staticmethod
    def _connected(g, membership):
        G = nx.Graph([(s, t) for _, s, t in g.get_edges_as_tuples()])
        G.add_nodes_from(membership)
        return all(nx.is_connected(G.subgraph(c)) for c in _groups(membership))

    def test_leiden_karate_reaches_optimum_and_connected(self):
        karate = ec.karate_club()
        for seed in range(5):
            m = karate.cmty_leiden(seed=seed, return_dict=True)
            assert karate.modularity(m) == pytest.approx(0.4198, abs=1e-4)
            assert self._connected(karate, m)

    @pytest.mark.parametrize("method", ["cmty_leiden", "cmty_label_propagation"])
    def test_recovers_planted_partition(self, method):
        P = nx.planted_partition_graph(6, 30, 0.4, 0.01, seed=2)
        g = _build(P)
        m = getattr(g, method)(seed=1, return_dict=True)
        found = sorted(sorted(_lab(g, n) for n in c) for c in _groups(m))
        truth = sorted(sorted(range(i * 30, (i + 1) * 30)) for i in range(6))
        assert found == truth

    def test_seeds_reproducible_and_labels_numbered(self):
        g = _build(nx.powerlaw_cluster_graph(300, 3, 0.3, seed=1))
        for method in ("cmty_leiden", "cmty_label_propagation"):
            a = getattr(g, method)(seed=7, return_dict=True)
            assert a == getattr(g, method)(seed=7, return_dict=True)
            assert set(a.values()) == set(range(len(set(a.values()))))
            assert a[min(a)] == 0

    def test_resolution_controls_granularity(self):
        karate = ec.karate_club()
        fine = karate.cmty_leiden(resolution=2.0, seed=1, return_dict=True)
        coarse = karate.cmty_leiden(resolution=0.5, seed=1, return_dict=True)
        assert len(set(fine.values())) > len(set(coarse.values()))

    def test_leiden_rejects_directed_and_handles_trivial(self):
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).cmty_leiden()
        assert ec.Graph().cmty_leiden(return_dict=True) == {}
        iso = ec.Graph().add_nodes(["A", "B"]).cmty_leiden(return_dict=True)
        assert len(set(iso.values())) == 2

    def test_stores_attributes(self):
        g = _two_triangles().cmty_leiden(seed=1).cmty_label_propagation(seed=1, attr_name="lpa")
        df = g.get_node_df().sort("label")
        assert df["cmty_leiden"].to_list() == [0, 0, 0, 1, 1, 1]
        assert df["lpa"].n_unique() == 2

    def test_k_clique_matches_networkx(self):
        for seed in range(15):
            U = nx.gnp_random_graph(16, 0.35, seed=seed)
            g = _build(U)
            for k in (2, 3, 4):
                ours = sorted(sorted(_lab(g, x) for x in c) for c in g.cmty_k_clique(k))
                ref = sorted(sorted(c) for c in nx.community.k_clique_communities(U, k))
                assert ours == ref

    def test_k_clique_overlap_and_errors(self):
        g = ec.Graph().add_nodes(list("ABCDEFG"))
        g.add_edges(
            [
                ("A", "B"),
                ("A", "C"),
                ("B", "C"),
                ("B", "D"),
                ("C", "D"),
                ("D", "E"),
                ("D", "F"),
                ("E", "F"),
                ("F", "G"),
            ]
        )
        comms = [_labels(g, c) for c in g.cmty_k_clique(3)]
        assert comms == [["A", "B", "C", "D"], ["D", "E", "F"]]
        with pytest.raises(ValueError, match="at least 2"):
            g.cmty_k_clique(1)
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).cmty_k_clique()

    def test_community_steps(self):
        g = _two_triangles().cmty_leiden(seed=1)
        step = g.steps[-1]
        assert step.kind == KIND_METRIC
        assert step.delta.node_groups == {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}
        assert step.narration == "cmty_leiden() found 2 communities."
        g.modularity("cmty_leiden")
        assert g.steps[-1].narration == "Modularity of 2 communities: Q = 0.357."
        g.cmty_k_clique(3)
        assert g.steps[-1].narration == "2 3-clique communities."

    def test_return_dict_records_no_step(self):
        g = _two_triangles()
        n = len(g.steps)
        g.cmty_leiden(return_dict=True)
        g.cmty_label_propagation(return_dict=True)
        assert len(g.steps) == n

    def test_label_propagation_with_infinite_weights(self):
        # inf - inf is NaN, which used to abort the tie filter
        g = ec.Graph(weighted=True).add_nodes(list("ABCD"))
        g.add_edges([("A", "B", float("inf")), ("B", "C", 1.0), ("C", "D", float("inf"))])
        labels = g.cmty_label_propagation(return_dict=True, seed=1)
        a, b, c, d = (labels[g.get_id_from_label(x)] for x in "ABCD")
        assert a == b and c == d

    def test_player_serializes_groups_and_selection(self):
        g = _two_triangles().cmty_leiden(seed=1)
        g.all_shortest_paths("A", "F")
        html = g.play()._html
        assert '"ng": {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}' in html
        assert '"se": [["A", "C"], ["C", "D"], ["D", "F"]]' in html
        assert "groupColor(" in html


# Partition quality (coverage and performance), validated against NetworkX.


class TestPartitionQuality:
    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        for seed in range(15):
            rnd = random.Random(seed)
            G = nx.gnp_random_graph(14, 0.25, seed=seed, directed=directed)
            g = ec.Graph(directed=directed).add_nodes([str(n) for n in G.nodes])
            if G.number_of_edges():
                g.add_edges([(str(u), str(v)) for u, v in G.edges])
            part = {n: rnd.randrange(3) for n in G.nodes}
            groups = [x for x in ({n for n in G if part[n] == k} for k in range(3)) if x]
            ours = g.partition_quality({str(n): c for n, c in part.items()})
            assert ours == pytest.approx(nx.community.partition_quality(G, groups))

    def test_two_triangles(self):
        g = ec.Graph().add_nodes(list("ABCDEF"))
        g.add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
        coverage, performance = g.partition_quality([["A", "B", "C"], ["D", "E", "F"]])
        assert coverage == pytest.approx(6 / 7)
        assert performance == pytest.approx(14 / 15)  # only the C-D pair is misclassified
        assert g.partition_quality([list("ABCDEF")]) == pytest.approx((1.0, 7 / 15))

    def test_edge_cases_and_errors(self):
        assert ec.Graph().add_nodes(["A", "B"]).partition_quality([["A"], ["B"]]) == (0.0, 1.0)
        assert ec.Graph().add_nodes(["A"]).partition_quality([["A"]]) == (0.0, 0.0)
        g = ec.Graph().add_nodes(["A", "B"]).add_edge("A", "B")
        with pytest.raises(ValueError, match="missing"):
            g.partition_quality([["A"]])

    def test_partition_quality_step(self):
        g = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("C", "D")])
        g.partition_quality([["A", "B"], ["C", "D"]])
        step = g.steps[-1]
        assert step.delta.node_groups == {"A": 0, "B": 0, "C": 1, "D": 1}
        assert step.narration == "2 communities: coverage 1.000, performance 1.000."


def _two_groups():
    g = ec.Graph().add_nodes(list("ABCDEFGH"))
    return g.add_edges(
        [
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
            ("E", "F"),
            ("E", "G"),
            ("E", "H"),
            ("F", "G"),
            ("F", "H"),
            ("G", "H"),
            ("D", "E"),
        ]
    )


def _from_nx(G, weighted=False):
    g = ec.Graph(weighted=weighted).add_nodes([str(n) for n in G])
    for u, v, d in G.edges(data=True):
        g.add_edge(str(u), str(v), weight=d.get("weight", 1.0))
    return g


class TestMoreCommunityMethods:
    """Fluid communities, fast label propagation, Louvain/Leiden levels, Kernighan-Lin, Lukes and
    greedy source expansion (exact NetworkX parity where the algorithm is deterministic)."""

    def test_fluid_and_fast_label_propagation(self):
        G = nx.planted_partition_graph(4, 20, 0.6, 0.02, seed=3)
        g = _from_nx(G)
        fluid = g.cmty_fluid(4, seed=1, return_dict=True)
        assert set(fluid.values()) == {0, 1, 2, 3}
        fast = g.cmty_fast_label_propagation(seed=1, return_dict=True)
        groups = {}
        for n, c in fast.items():
            groups.setdefault(c, set()).add(int(g.get_label_from_id(n)))
        assert nx_comm.modularity(G, groups.values()) > 0.6
        assert g.cmty_fluid(4, seed=5, return_dict=True) == g.cmty_fluid(
            4, seed=5, return_dict=True
        )
        g.cmty_fluid(4, seed=1)
        assert set(g.get_node_attrs("cmty_fluid").values()) == {0, 1, 2, 3}
        with pytest.raises(ValueError, match="connected"):
            ec.Graph().add_nodes(["A", "B"]).cmty_fluid(1)
        with pytest.raises(ValueError, match="exceed"):
            _two_groups().cmty_fluid(9)

    def test_levels(self):
        G = nx.connected_watts_strogatz_graph(300, 6, 0.05, seed=2)
        g = _from_nx(G)
        for levels in (g.cmty_louvain_levels(seed=42), g.cmty_leiden_levels(seed=42)):
            counts = [len(set(level.values())) for level in levels]
            assert counts == sorted(counts, reverse=True) and len(levels) >= 2
        assert g.cmty_louvain_levels(seed=42)[-1] == g.cmty_louvain(return_dict=True)
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_path_graph(3).cmty_louvain_levels()

    def test_kernighan_lin(self):
        matches = 0
        for seed in range(40):
            rnd = random.Random(seed)
            G = nx.gnp_random_graph(12, 0.35, seed=seed)
            for u, v in G.edges:
                G[u][v]["weight"] = rnd.random()
            g = _from_nx(G, True)
            start = set(rnd.sample(list(G), 6))
            a, b = g.kernighan_lin_bisection(
                partition=([str(x) for x in start], [str(x) for x in set(G) - start])
            )
            side = {int(g.get_label_from_id(x)) for x in a}
            assert len(a) == len(b) == 6
            assert (
                nx.cut_size(G, side, weight="weight")
                <= nx.cut_size(G, start, weight="weight") + 1e-12
            )
            ra, _ = nx_comm.kernighan_lin_bisection(G, partition=(start, set(G) - start), seed=1)
            ours_cut = nx.cut_size(G, side, weight="weight")
            matches += abs(ours_cut - nx.cut_size(G, ra, weight="weight")) < 1e-9
        assert matches >= 35  # NetworkX sometimes continues after a rounding-zero pass
        left, right = _two_groups().kernighan_lin_bisection(seed=3)
        assert len(left) == len(right) == 4
        with pytest.raises(ValueError, match="split every node"):
            _two_groups().kernighan_lin_bisection(partition=(["A"], ["B"]))

    def test_lukes(self):
        for seed in range(60):
            rnd = random.Random(seed)
            T = nx.random_labeled_tree(rnd.randint(2, 12), seed=seed)
            for n in T:
                T.nodes[n]["w"] = rnd.randint(1, 3)
            for u, v in T.edges:
                T[u][v]["e"] = rnd.randint(1, 5)
            g = ec.Graph().add_nodes([(str(n), {"w": T.nodes[n]["w"]}) for n in T])
            g.add_edges([(str(u), str(v), {"e": T[u][v]["e"]}) for u, v in T.edges])
            k = max(T.nodes[n]["w"] for n in T) + rnd.randint(0, 4)
            ours = [
                {int(g.get_label_from_id(x)) for x in c} for c in g.lukes_partitioning(k, "w", "e")
            ]
            ref = nx_comm.lukes_partitioning(T, k, node_weight="w", edge_weight="e")

            def kept(parts):
                return sum(
                    T[u][v]["e"] for c in parts for u in c for v in c if u < v and T.has_edge(u, v)
                )

            assert kept(ours) == kept(ref)
            assert all(sum(T.nodes[x]["w"] for x in c) <= k for c in ours)
            assert sorted(x for c in ours for x in c) == sorted(T)
        with pytest.raises(ValueError, match="tree"):
            ec.Graph().add_cycle_graph(4).lukes_partitioning(2)
        # The weight field is the default, as it is for weight_attr="weight"
        for heavy in ("A", "C"):
            path = ec.Graph(weighted=True).add_nodes(["A", "B", "C"])
            path.add_edges([("A", "B", 5.0 if heavy == "A" else 1.0)])
            path.add_edges([("B", "C", 5.0 if heavy == "C" else 1.0)])
            parts = path.lukes_partitioning(2, weight_attr="weight")
            labels = sorted(sorted(path.get_label_from_id(x) for x in c) for c in parts)
            assert sorted(["B", heavy]) in labels
            assert path.lukes_partitioning(2) == parts
        # A missing edge attribute raises, as a missing node attribute does
        path = ec.Graph().add_nodes(["A", "B", "C"])
        path.add_edges([("A", "B", {"e": 1.0}), ("B", "C")])
        with pytest.raises(ValueError, match="no numeric 'e' attribute, which weight_attr='e'"):
            path.lukes_partitioning(2, weight_attr="e")

    @pytest.mark.skipif(not hasattr(nx_comm, "greedy_source_expansion"), reason="NetworkX 3.5+")
    def test_greedy_source_expansion(self):
        G = nx.karate_club_graph()
        g = _from_nx(G)
        found = {int(g.get_label_from_id(x)) for x in g.greedy_source_expansion("0")}
        assert found == nx_comm.greedy_source_expansion(G, source=0)
        assert len(g.greedy_source_expansion("0", cutoff=3)) == 3
        two = _two_groups()
        assert {two.get_label_from_id(x) for x in two.greedy_source_expansion("A")} == set("ABCDE")
