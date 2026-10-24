"""Tests for CPM Leiden and the community quality functions added in NetworkX 3.6/3.7.

The reference implementations below follow NetworkX 3.7's `constant_potts_model`,
`overlapping_modularity`, `is_cover` and `bipartite.modularity` (the installed NetworkX may be
older and lack them).
"""

import random
from collections import defaultdict

import edgecase as ec
import networkx as nx
import pytest

# =============================================================================
# Reference implementations (NetworkX 3.7)
# =============================================================================


def _ref_cpm(G, communities, resolution=1.0, weight="weight", node_weight="node_weight"):
    factor = 1.0 if G.is_directed() else 0.5
    total = 0.0
    for c in communities:
        comm = set(c)
        e_c = sum(wt for _, v, wt in G.edges(comm, data=weight, default=1) if v in comm)
        n_c = sum(G.nodes[u].get(node_weight, 1) for u in c)
        total += e_c - resolution * factor * n_c**2
    return total


def _ref_overlapping(G, communities, resolution=1.0, weight="weight"):
    membership = defaultdict(int)
    for c in communities:
        for u in c:
            membership[u] += 1
    degree = dict(G.degree(weight=weight))
    deg_sum = sum(degree.values())
    if deg_sum == 0:
        return 0.0
    total = 0.0
    for c in communities:
        comm = set(c)
        lt = sum(
            wt / (membership[u] * membership[v])
            for u, v, wt in G.edges(comm, data=weight, default=1)
            if v in comm
        )
        kt = sum(degree[u] / membership[u] for u in comm)
        total += 2 * lt / deg_sum - resolution * (kt / deg_sum) ** 2
    return total


def _ref_bipartite(G, communities, red, resolution=1.0, weight="weight"):
    red = set(red)
    blue = set(G) - red
    degree = dict(G.degree(weight=weight))
    m = sum(degree[v] for v in red)
    if m == 0:
        return 0.0
    total = 0.0
    for c in communities:
        comm = set(c)
        l_c = sum(wt for _, v, wt in G.edges(comm, data=weight, default=1) if v in comm)
        k_c = sum(degree[u] for u in comm & red)
        d_c = sum(degree[u] for u in comm & blue)
        total += l_c / m - resolution * k_c * d_c / m**2
    return total


# =============================================================================
# Helpers
# =============================================================================


def _build(G, weighted=False):
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G.nodes])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


def _lab(groups):
    return [[str(n) for n in c] for c in groups]


def _random_partition(G, k, rnd):
    part = [[] for _ in range(k)]
    for n in G:
        part[rnd.randrange(k)].append(n)
    return [p for p in part if p]


def _cases():
    """(networkx graph with weights, edgecase graph, weight key) over random graphs."""
    for seed in range(12):
        rnd = random.Random(seed)
        for directed in (False, True):
            for weighted in (False, True):
                G = nx.gnp_random_graph(16, 0.2, seed=seed, directed=directed)
                if seed % 3 == 0:
                    G.add_edge(2, 2)
                for u, v in G.edges:
                    G[u][v]["weight"] = rnd.choice([0.5, 1, 2, 3])
                yield G, _build(G, weighted), ("weight" if weighted else "unused")


def _two_triangles():
    return (
        ec.Graph()
        .add_nodes(list("ABCDEF"))
        .add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
    )


def _bowtie():
    """Two triangles sharing node C."""
    return (
        ec.Graph()
        .add_nodes(list("ABCDE"))
        .add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "C")])
    )


def _groups(membership):
    out = defaultdict(set)
    for n, c in membership.items():
        out[c].add(n)
    return list(out.values())


# =============================================================================
# Constant Potts Model
# =============================================================================


class TestConstantPottsModel:
    def test_matches_networkx(self):
        for G, g, w in _cases():
            rnd = random.Random(G.number_of_edges())
            part = _random_partition(G, rnd.randint(1, 5), rnd)
            for res in (1.0, 0.3, 0.05):
                ours = g.constant_potts_model(_lab(part), resolution=res)
                assert ours == pytest.approx(_ref_cpm(G, part, res, weight=w), abs=1e-9)

    def test_two_triangles(self):
        g = _two_triangles()
        # Each triangle: 3 edges - 0.5 * 3^2 / 2 = 0.75
        assert g.constant_potts_model([list("ABC"), list("DEF")], resolution=0.5) == 1.5
        assert g.constant_potts_model([list("ABCDEF")], resolution=0.5) == -2.0
        assert g.constant_potts_model([[x] for x in "ABCDEF"], resolution=0.5) == -1.5

    def test_node_weight(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 3), (3, 4)])
        nx.set_node_attributes(G, {0: 2.5, 3: 4.0}, "nw")
        g = _build(G)
        g.set_node_attrs("nw", {g.get_id_from_label("0"): 2.5, g.get_id_from_label("3"): 4.0})
        part = [[0, 1, 2], [3, 4]]
        ours = g.constant_potts_model(_lab(part), resolution=0.4, node_weight="nw")
        assert ours == pytest.approx(_ref_cpm(G, part, 0.4, node_weight="nw"))
        with pytest.raises(ValueError, match="No node attribute"):
            g.constant_potts_model(_lab(part), node_weight="missing")

    def test_invalid_partition(self):
        g = _two_triangles()
        with pytest.raises(ValueError, match="missing"):
            g.constant_potts_model([list("ABC")])
        with pytest.raises(ValueError, match="more than one"):
            g.constant_potts_model([list("ABC"), list("CDEF")])

    def test_step(self):
        g = _two_triangles()
        g.constant_potts_model([list("ABC"), list("DEF")], resolution=0.5)
        step = g.steps[-1]
        assert step.code == "G.constant_potts_model(communities, resolution=0.5)"
        assert step.delta.node_groups == {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}
        assert step.narration == "Constant Potts Model score of 2 communities: 1.500."


# =============================================================================
# Leiden with objective="cpm"
# =============================================================================


class TestLeidenCPM:
    # Best CPM score NetworkX 3.7's leiden_communities(metric="cpm") reaches on the unweighted
    # karate club over seeds 0-4
    NX_BEST = {0.05: 53.55, 0.1: 41.4, 0.2: 27.2, 0.5: 5.5}

    @pytest.mark.parametrize("res", sorted(NX_BEST))
    def test_karate_matches_networkx_quality(self, res):
        karate = ec.karate_club()
        best = max(
            karate.constant_potts_model(
                karate.cmty_leiden(objective="cpm", resolution=res, seed=seed, return_dict=True),
                resolution=res,
            )
            for seed in range(5)
        )
        assert best >= self.NX_BEST[res] - 1e-9

    def test_communities_connected_and_dense(self):
        karate = ec.karate_club()
        G = nx.Graph([(s, t) for _, s, t in karate.get_edges_as_tuples()])
        for res in (0.1, 0.3, 0.6):
            m = karate.cmty_leiden(objective="cpm", resolution=res, seed=1, return_dict=True)
            for c in _groups(m):
                sub = G.subgraph(c)
                assert len(c) == 1 or nx.is_connected(sub)
                if len(c) > 1:
                    # Splitting a community off never pays: its density exceeds the threshold
                    assert nx.density(sub) >= res

    def test_granularity_grows_with_resolution(self):
        karate = ec.karate_club()
        counts = [
            len(set(karate.cmty_leiden(objective="cpm", resolution=r, seed=1,
                                       return_dict=True).values()))
            for r in (0.05, 0.2, 0.5, 1.0)
        ]
        assert counts == sorted(counts)
        assert counts[-1] == 34  # at density threshold 1, no merge pays off

    def test_recovers_planted_partition(self):
        P = nx.planted_partition_graph(5, 25, 0.5, 0.01, seed=3)
        g = _build(P)
        m = g.cmty_leiden(objective="cpm", resolution=0.1, seed=1, return_dict=True)
        found = sorted(sorted(int(g.get_label_from_id(n)) for n in c) for c in _groups(m))
        assert found == [list(range(i * 25, (i + 1) * 25)) for i in range(5)]

    def test_weighted(self):
        G = nx.les_miserables_graph()
        g = ec.Graph(weighted=True).add_nodes(list(G))
        for u, v, d in G.edges(data=True):
            g.add_edge(u, v, weight=d["weight"])
        m = g.cmty_leiden(objective="cpm", resolution=1.0, seed=1, return_dict=True)
        q = g.constant_potts_model(m, resolution=1.0)
        assert q > g.constant_potts_model({n: 0 for n in m}, resolution=1.0)
        assert q > g.constant_potts_model({n: i for i, n in enumerate(m)}, resolution=1.0)

    def test_theta_and_seed(self):
        g = nx.powerlaw_cluster_graph(200, 3, 0.3, seed=2)
        eg = _build(g)
        a = eg.cmty_leiden(objective="cpm", resolution=0.05, theta=0.2, seed=4, return_dict=True)
        b = eg.cmty_leiden(objective="cpm", resolution=0.05, theta=0.2, seed=4, return_dict=True)
        assert a == b

    def test_modularity_objective_unchanged(self):
        karate = ec.karate_club()
        a = karate.cmty_leiden(seed=3, return_dict=True)
        b = karate.cmty_leiden(seed=3, objective="modularity", theta=0.01, return_dict=True)
        assert a == b

    def test_errors(self):
        g = _two_triangles()
        with pytest.raises(ValueError, match="objective"):
            g.cmty_leiden(objective="barber")
        for theta in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match="theta"):
                g.cmty_leiden(theta=theta)
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).cmty_leiden(objective="cpm")

    def test_tracked_params(self):
        g = _two_triangles().cmty_leiden(objective="cpm", resolution=0.5, theta=0.05, seed=1)
        assert g.steps[-1].code == (
            "G.cmty_leiden(resolution=0.5, seed=1, objective='cpm', theta=0.05, "
            "attr_name='cmty_leiden')"
        )
        df = g.get_node_df().sort("label")
        assert df["cmty_leiden"].to_list() == [0, 0, 0, 1, 1, 1]


# =============================================================================
# Covers and overlapping modularity
# =============================================================================


class TestIsCover:
    def test_basic(self):
        g = _bowtie()
        assert g.is_cover([list("ABC"), list("CDE")])
        assert not g.is_cover([list("ABC")])
        assert g.is_cover([list("ABCDE")])
        assert not g.is_cover([])

    def test_unknown_nodes_ignored(self):
        g = _bowtie()
        assert g.is_cover([["A", "B", "C", "Z"], ["D", "E", 999]])
        assert not g.is_cover([["Z"]])

    def test_empty_graph(self):
        assert ec.Graph().is_cover([])

    def test_rejects_non_list(self):
        with pytest.raises(TypeError):
            _bowtie().is_cover("cmty")


class TestOverlappingModularity:
    def test_matches_networkx(self):
        for G, g, w in _cases():
            if G.is_directed():
                continue
            rnd = random.Random(G.number_of_edges())
            part = _random_partition(G, rnd.randint(1, 4), rnd)
            cover = [sorted(set(p) | set(rnd.sample(list(G), 3))) for p in part]
            for res in (1.0, 0.7):
                ours = g.overlapping_modularity(_lab(cover), resolution=res)
                assert ours == pytest.approx(_ref_overlapping(G, cover, res, w), abs=1e-12)

    def test_equals_modularity_on_partition(self):
        karate = ec.karate_club()
        m = karate.cmty_leiden(seed=1, return_dict=True)
        groups = [sorted(c) for c in _groups(m)]
        assert karate.overlapping_modularity(groups) == pytest.approx(karate.modularity(m))

    def test_bowtie(self):
        g = _bowtie()
        both = g.overlapping_modularity([list("ABC"), list("CDE")])
        forced = g.overlapping_modularity([list("ABC"), list("DE")])
        assert both == pytest.approx(1 / 6)
        assert forced == pytest.approx(g.modularity([list("ABC"), list("DE")]))
        assert both > forced

    def test_duplicates_in_a_group_count_once(self):
        g = _bowtie()
        assert g.overlapping_modularity([list("ABCA"), list("CDE")]) == pytest.approx(1 / 6)

    def test_edgeless_and_errors(self):
        assert ec.Graph().add_nodes(["A", "B"]).overlapping_modularity([["A", "B"]]) == 0.0
        g = _bowtie()
        with pytest.raises(ValueError, match="not in any community"):
            g.overlapping_modularity([list("ABC")])
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).overlapping_modularity([["A"]])
        with pytest.raises(TypeError):
            g.overlapping_modularity({"A": 0})

    def test_step(self):
        g = _bowtie()
        g.overlapping_modularity([list("ABC"), list("CDE")], resolution=0.5)
        step = g.steps[-1]
        assert step.code == "G.overlapping_modularity(communities, resolution=0.5)"
        assert step.narration.startswith("Extended modularity of 2 overlapping communities")


# =============================================================================
# Bipartite (Barber) modularity
# =============================================================================


class TestBipartiteModularity:
    def test_matches_networkx(self):
        for seed in range(12):
            rnd = random.Random(seed)
            B = nx.bipartite.random_graph(9, 12, 0.3, seed=seed)
            for u, v in B.edges:
                B[u][v]["weight"] = rnd.choice([0.5, 1, 2, 3])
            top = [n for n, side in B.nodes(data="bipartite") if side == 0]
            part = _random_partition(B, rnd.randint(1, 4), rnd)
            for weighted in (False, True):
                g = _build(B, weighted)
                w = "weight" if weighted else "unused"
                for res in (1.0, 0.6):
                    ours = g.bipartite_modularity(
                        _lab(part), top_nodes=[str(t) for t in top], resolution=res
                    )
                    assert ours == pytest.approx(_ref_bipartite(B, part, top, res, w), abs=1e-12)

    def test_auto_sides_on_connected_graph(self):
        for seed in range(6):
            B = nx.bipartite.random_graph(8, 10, 0.4, seed=seed)
            if not nx.is_connected(B):
                continue
            g = _build(B)
            top = [str(n) for n, side in B.nodes(data="bipartite") if side == 0]
            part = _lab(_random_partition(B, 3, random.Random(seed)))
            # On a connected graph, swapping the two sides leaves the score unchanged
            assert g.bipartite_modularity(part) == pytest.approx(
                g.bipartite_modularity(part, top_nodes=top)
            )

    def test_two_components(self):
        # NetworkX docstring example: two disconnected edges
        g = ec.Graph().add_nodes(["0", "1", "2", "3"]).add_edges([("0", "2"), ("1", "3")])
        assert g.bipartite_modularity([["0", "2"], ["1", "3"]], top_nodes=["0", "1"]) == 0.5

    def test_lumped_scores_zero(self):
        B = nx.complete_bipartite_graph(3, 4)
        g = _build(B)
        assert g.bipartite_modularity([g.get_node_ids()]) == pytest.approx(0.0)

    def test_edgeless(self):
        g = ec.Graph().add_nodes(["A", "B"])
        assert g.bipartite_modularity([["A"], ["B"]]) == 0.0

    def test_errors(self):
        tri = ec.Graph().add_nodes(list("ABC")).add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        with pytest.raises(ValueError, match="not bipartite"):
            tri.bipartite_modularity([list("ABC")])
        loop = ec.Graph().add_nodes(["A", "B"]).add_edges([("A", "B"), ("A", "A")])
        with pytest.raises(ValueError, match="self-loop"):
            loop.bipartite_modularity([["A", "B"]])
        path = ec.Graph().add_nodes(list("ABC")).add_edges([("A", "B"), ("B", "C")])
        with pytest.raises(ValueError, match="does not split"):
            path.bipartite_modularity([list("ABC")], top_nodes=["A", "B"])
        with pytest.raises(ValueError, match="missing"):
            path.bipartite_modularity([["A", "B"]])
        with pytest.raises(ValueError, match="undirected"):
            ec.Graph(directed=True).add_nodes(["A"]).bipartite_modularity([["A"]])

    def test_step(self):
        g = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("C", "D")])
        g.bipartite_modularity([["A", "B"], ["C", "D"]], top_nodes=["A", "C"])
        step = g.steps[-1]
        assert step.code == "G.bipartite_modularity(communities, top_nodes=top_nodes)"
        assert step.delta.node_groups == {"A": 0, "B": 0, "C": 1, "D": 1}
        assert step.narration == "Bipartite modularity of 2 communities: Q = 0.500."
