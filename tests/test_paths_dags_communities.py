"""Tests for the shortest-path queries, DAG methods, and community methods added together:

- paths: shortest_path_length, has_path, all_shortest_paths, single_source_shortest_path(_length),
  multi_source_shortest_path(_length)
- DAGs: ancestors, descendants, topological_generations, dag_longest_path(_length),
  condensation, lowest_common_ancestor
- communities: modularity, cmty_leiden, cmty_label_propagation, cmty_k_clique

Results are validated against NetworkX on random graphs (directed/undirected, weighted/unweighted,
often disconnected), checked on hand-computed cases, and exercised through the step log/player.
"""

import itertools
import math
import random
from collections import defaultdict

import edgecase as ec
import networkx as nx
import pytest
from edgecase.steps import KIND_METRIC, KIND_QUERY, KIND_TRANSFORM

# --- helpers ----------------------------------------------------------------------


def _build(G: nx.Graph, weighted: bool = False) -> ec.Graph:
    g = ec.Graph(directed=G.is_directed(), weighted=weighted)
    g.add_nodes([str(n) for n in G.nodes])
    for u, v, d in G.edges(data=True):
        if weighted:
            g.add_edge(str(u), str(v), weight=d["weight"])
        else:
            g.add_edge(str(u), str(v))
    return g


def _lab(g: ec.Graph, nid: int) -> int:
    return int(g.get_label_from_id(nid))


def _id(g: ec.Graph, n) -> int:
    return g.get_id_from_label(str(n))


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


def _random_dags():
    for seed in range(15):
        rnd = random.Random(seed)
        D = nx.gnp_random_graph(14, 0.25, seed=seed, directed=True)
        D = nx.DiGraph([(u, v) for u, v in D.edges if u < v])
        D.add_nodes_from(range(14))
        for u, v in D.edges:
            D[u][v]["weight"] = rnd.choice([1, 2, 3])
        for weighted in (False, True):
            yield D, _build(D, weighted), weighted


def _two_triangles() -> ec.Graph:
    return (
        ec.Graph()
        .add_nodes(list("ABCDEF"))
        .add_edges(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "D")]
        )
    )


def _pipeline() -> ec.Graph:
    return (
        ec.Graph(directed=True)
        .add_nodes(["fetch", "compile", "test", "package", "docs"])
        .add_edges(
            [
                ("fetch", "compile"),
                ("compile", "test"),
                ("test", "package"),
                ("compile", "package"),
                ("fetch", "docs"),
            ]
        )
    )


def _labels(g: ec.Graph, ids) -> list[str]:
    return [g.get_label_from_id(n) for n in ids]


# --- shortest-path queries --------------------------------------------------------


class TestPathQueriesAgainstNetworkX:
    def test_single_source_lengths(self):
        for G, g, w in _random_cases():
            s = 0
            ours = {
                _lab(g, k): v
                for k, v in g.single_source_shortest_path_length("0", return_dict=True).items()
            }
            ref = (
                nx.single_source_dijkstra_path_length(G, s, weight=w)
                if w
                else nx.single_source_shortest_path_length(G, s)
            )
            assert ours.keys() == ref.keys()
            for k, v in ref.items():
                assert ours[k] == pytest.approx(v)

    def test_cutoff(self):
        for G, g, w in _random_cases():
            ours = set(
                _lab(g, k)
                for k in g.single_source_shortest_path_length("0", cutoff=2, return_dict=True)
            )
            ref = (
                nx.single_source_dijkstra_path_length(G, 0, cutoff=2, weight=w)
                if w
                else nx.single_source_shortest_path_length(G, 0, cutoff=2)
            )
            assert ours == set(ref)

    def test_single_source_paths_are_shortest(self):
        for G, g, w in _random_cases():
            ref = nx.single_source_dijkstra_path_length(G, 0, weight=w or (lambda u, v, d: 1))
            paths = g.single_source_shortest_path("0")
            assert {_lab(g, t) for t in paths} == set(ref)
            for t, p in paths.items():
                p = [_lab(g, x) for x in p]
                assert p[0] == 0 and p[-1] == _lab(g, t)
                cost = sum(G[a][b]["weight"] if w else 1 for a, b in zip(p, p[1:]))
                assert cost == pytest.approx(ref[p[-1]])

    def test_multi_source(self):
        for G, g, w in _random_cases():
            srcs = [0, 1, 2]
            ref = nx.multi_source_dijkstra_path_length(
                G, set(srcs), weight=w or (lambda u, v, d: 1)
            )
            ours = g.multi_source_shortest_path_length([str(s) for s in srcs], return_dict=True)
            assert {_lab(g, k): v for k, v in ours.items()} == pytest.approx(ref)
            for t, p in g.multi_source_shortest_path([str(s) for s in srcs]).items():
                p = [_lab(g, x) for x in p]
                assert p[0] in srcs and p[-1] == _lab(g, t)
                cost = sum(G[a][b]["weight"] if w else 1 for a, b in zip(p, p[1:]))
                assert cost == pytest.approx(ref[p[-1]])

    def test_pairwise_length_has_path_and_all_shortest_paths(self):
        for G, g, w in _random_cases():
            for a, b in itertools.islice(itertools.permutations(G.nodes, 2), 0, 80, 9):
                reachable = nx.has_path(G, a, b)
                assert g.has_path(str(a), str(b)) is reachable
                length = g.shortest_path_length(str(a), str(b))
                paths = g.all_shortest_paths(str(a), str(b))
                if not reachable:
                    assert length is None and paths == []
                    continue
                assert length == pytest.approx(nx.shortest_path_length(G, a, b, weight=w))
                ours = sorted([_lab(g, x) for x in p] for p in paths)
                assert ours == sorted(nx.all_shortest_paths(G, a, b, weight=w))


class TestPathQueries:
    def test_unweighted_lengths_are_ints(self):
        g = _two_triangles()
        assert g.shortest_path_length("A", "F") == 3
        assert isinstance(g.shortest_path_length("A", "F"), int)
        dist = g.single_source_shortest_path_length("A", return_dict=True)
        assert all(isinstance(v, int) for v in dist.values())

    def test_weighted_lengths_are_floats(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=2.5)
        assert g.shortest_path_length("A", "B") == 2.5

    def test_agrees_with_shortest_path(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(list("ABCDF"))
        g.add_edges(
            [("A", "B", 1.0), ("B", "C", 2.0), ("C", "F", 1.0), ("A", "D", 3.0), ("D", "F", 3.0)]
        )
        path = g.shortest_path("A", "F", return_list=True)
        assert g.shortest_path_length("A", "F") == 4.0
        assert path in g.all_shortest_paths("A", "F")

    def test_self_distance_and_self_path(self):
        g = _two_triangles()
        assert g.shortest_path_length("A", "A") == 0
        assert g.has_path("A", "A")
        a = g.get_id_from_label("A")
        assert g.all_shortest_paths("A", "A") == [[a]]
        assert g.single_source_shortest_path("A")[a] == [a]

    def test_directed_has_no_way_back(self):
        g = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edge("A", "B")
        assert g.has_path("A", "B") and not g.has_path("B", "A")
        assert g.shortest_path_length("B", "A") is None

    def test_grid_has_binomial_number_of_paths_and_max_paths(self):
        grid = ec.Graph().add_grid_2d(rows=4, cols=4)
        assert len(grid.all_shortest_paths(1, 16)) == math.comb(6, 3)
        assert len(grid.all_shortest_paths(1, 16, max_paths=5)) == 5
        with pytest.raises(ValueError, match="max_paths"):
            grid.all_shortest_paths(1, 16, max_paths=0)

    def test_negative_weights_rejected(self):
        g = ec.Graph(weighted=True).add_nodes(["A", "B"]).add_edge("A", "B", weight=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            g.shortest_path_length("A", "B")

    def test_errors(self):
        g = _two_triangles()
        with pytest.raises(ValueError):
            g.has_path("A", "nope")
        with pytest.raises(ValueError, match="source"):
            g.multi_source_shortest_path([])

    def test_length_attributes_stored_with_nulls(self):
        g = ec.Graph().add_nodes(["A", "B", "C"]).add_edge("A", "B")
        g.single_source_shortest_path_length("A", attr_name="d")
        df = g.get_node_df().sort("label")
        assert df["d"].to_list() == [0, 1, None]
        g.multi_source_shortest_path_length(["A", "C"])
        df = g.get_node_df().sort("label")
        assert df["multi_source_shortest_path_length"].to_list() == [0, 1, 0]


# --- DAGs -------------------------------------------------------------------------


class TestDagAgainstNetworkX:
    def test_ancestors_descendants_any_graph(self):
        for G, g, _ in _random_cases():
            for x in list(G.nodes)[:5]:
                assert {_lab(g, k) for k in g.ancestors(str(x))} == nx.ancestors(G, x)
                assert {_lab(g, k) for k in g.descendants(str(x))} == nx.descendants(G, x)

    def test_generations_longest_path_lca(self):
        for D, d, weighted in _random_dags():
            gens = [sorted(_lab(d, x) for x in gen) for gen in d.topological_generations()]
            assert gens == [sorted(x) for x in nx.topological_generations(D)]
            path = [_lab(d, x) for x in d.dag_longest_path()]
            assert all(D.has_edge(a, b) for a, b in zip(path, path[1:]))
            ref = nx.dag_longest_path_length(
                D, weight="weight" if weighted else "unused", default_weight=1
            )
            assert d.dag_longest_path_length() == pytest.approx(ref)
            for a, b in itertools.islice(itertools.combinations(range(14), 2), 0, 91, 4):
                ours = d.lowest_common_ancestor(str(a), str(b))
                if nx.lowest_common_ancestor(D, a, b) is None:
                    assert ours is None
                    continue
                common = (nx.ancestors(D, a) | {a}) & (nx.ancestors(D, b) | {b})
                o = _lab(d, ours)
                assert o in common and not any(c in common for c in D.successors(o))

    def test_condensation(self):
        for G, g, _ in _random_cases():
            if not G.is_directed():
                continue
            ours = g.condensation()
            ref = nx.condensation(G)
            members = sorted(
                sorted(int(x) for x in m) for m in ours.get_node_attrs("members").values()
            )
            assert members == sorted(sorted(ref.nodes[k]["members"]) for k in ref)
            assert ours.count_edges() == ref.number_of_edges()
            assert ours.is_dag()
            # nodes were added in topological order
            assert all(s < t for _, s, t in ours.get_edges_as_tuples())


class TestDag:
    def test_pipeline(self):
        g = _pipeline()
        assert _labels(g, g.ancestors("package")) == ["fetch", "compile", "test"]
        assert _labels(g, g.descendants("compile")) == ["test", "package"]
        gens = [_labels(g, gen) for gen in g.topological_generations()]
        assert gens == [["fetch"], ["compile", "docs"], ["test"], ["package"]]
        assert _labels(g, g.dag_longest_path()) == ["fetch", "compile", "test", "package"]
        assert g.dag_longest_path_length() == 3

    def test_weighted_longest_path(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(list("ABCD"))
        g.add_edges([("A", "B", 1.0), ("B", "D", 1.0), ("A", "C", 5.0), ("C", "D", 0.5)])
        assert _labels(g, g.dag_longest_path()) == ["A", "C", "D"]
        assert g.dag_longest_path_length() == 5.5

    def test_cycles_and_undirected_rejected(self):
        cyc = ec.Graph(directed=True).add_nodes(["A", "B"]).add_edges([("A", "B"), ("B", "A")])
        loop = ec.Graph(directed=True).add_nodes(["A"]).add_edge("A", "A")
        for g in (cyc, loop):
            for method in ("topological_generations", "dag_longest_path"):
                with pytest.raises(ValueError, match="cycle"):
                    getattr(g, method)()
            with pytest.raises(ValueError, match="cycle"):
                g.lowest_common_ancestor("A", "A")
        und = _two_triangles()
        for method in ("topological_generations", "dag_longest_path", "condensation"):
            with pytest.raises(ValueError, match="directed"):
                getattr(und, method)()

    def test_ancestors_on_undirected_is_component(self):
        g = _two_triangles().add_node("Z")
        assert _labels(g, g.ancestors("A")) == ["B", "C", "D", "E", "F"]
        assert g.descendants("Z") == []

    def test_empty_and_edgeless(self):
        assert ec.Graph(directed=True).dag_longest_path() == []
        assert ec.Graph(directed=True).topological_generations() == []
        g = ec.Graph(directed=True).add_nodes(["A", "B"])
        assert len(g.dag_longest_path()) == 1 and g.dag_longest_path_length() == 0

    def test_lca_taxonomy_and_self_ancestry(self):
        t = ec.Graph(directed=True).add_nodes(["M", "C", "R", "cat", "dog", "mouse"])
        t.add_edges([("M", "C"), ("M", "R"), ("C", "cat"), ("C", "dog"), ("R", "mouse")])
        lab = t.get_label_from_id
        assert lab(t.lowest_common_ancestor("cat", "dog")) == "C"
        assert lab(t.lowest_common_ancestor("cat", "mouse")) == "M"
        assert lab(t.lowest_common_ancestor("C", "dog")) == "C"
        two_roots = ec.Graph(directed=True).add_nodes(["A", "B"])
        assert two_roots.lowest_common_ancestor("A", "B") is None

    def test_condensation_labels_and_attributes(self):
        g = ec.Graph(directed=True).add_nodes(list("ABCDE"))
        g.add_edges([("A", "B"), ("B", "A"), ("B", "C"), ("C", "D"), ("D", "C")])
        dag = g.condensation()
        df = dag.get_node_df()
        assert df["label"].to_list() == ["{A,B}", "{C,D}", "E"]
        assert df["members"].to_list() == [["A", "B"], ["C", "D"], ["E"]]
        assert df["size"].to_list() == [2, 2, 1]
        assert dag.get_edges_str_repr(use_labels=True) == ["{A,B}->{C,D}"]
        assert g.count_nodes() == 5  # original untouched


# --- communities ------------------------------------------------------------------


def _groups(membership: dict[int, int]) -> list[set[int]]:
    out = defaultdict(set)
    for n, c in membership.items():
        out[c].add(n)
    return list(out.values())


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


# --- step log and player ----------------------------------------------------------


class TestStepHooks:
    def test_path_queries_highlight_results(self):
        g = _two_triangles()
        g.all_shortest_paths("A", "F")
        step = g.steps[-1]
        assert step.kind == KIND_QUERY and step.method == "all_shortest_paths"
        assert step.delta.nodes_selected == ["A", "C", "D", "F"]
        assert step.delta.edges_selected == [("A", "C"), ("C", "D"), ("D", "F")]
        assert step.narration == "1 shortest path from A to F."

        g.has_path("A", "F")
        assert g.steps[-1].narration == "F is reachable from A."
        g.shortest_path_length("A", "F")
        assert g.steps[-1].result == 3
        assert g.steps[-1].delta.nodes_selected == ["A", "F"]

    def test_distance_methods_record_metric_heat(self):
        g = _two_triangles().single_source_shortest_path_length("A")
        step = g.steps[-1]
        assert step.kind == KIND_METRIC
        assert step.delta.metric_values == {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3, "F": 3}
        assert "farthest: E (3)" in step.narration

    def test_multi_source_path_records_catchments(self):
        g = _two_triangles()
        g.multi_source_shortest_path(["A", "F"])
        d = g.steps[-1].delta
        assert d.node_groups == {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}
        assert d.nodes_selected == [] and len(d.edges_selected) == 4

    def test_dag_steps(self):
        g = _pipeline()
        g.ancestors("package")
        assert g.steps[-1].delta.nodes_selected == ["fetch", "compile", "test"]
        g.topological_generations()
        assert g.steps[-1].delta.node_groups == {
            "fetch": 0,
            "compile": 1,
            "docs": 1,
            "test": 2,
            "package": 3,
        }
        g.dag_longest_path()
        assert g.steps[-1].narration == "Longest path: fetch → compile → test → package (length 3)."
        g.lowest_common_ancestor("test", "docs")
        assert g.steps[-1].delta.nodes_selected == ["fetch"]

    def test_condensation_records_on_both_graphs(self):
        g = ec.Graph(directed=True).add_nodes(list("ABC")).add_edges([("A", "B"), ("B", "A")])
        dag = g.condensation()
        step = g.steps[-1]
        assert step.kind == KIND_TRANSFORM
        assert step.delta.node_groups == {"A": 0, "B": 0, "C": 1}
        assert len(dag.steps) == 1 and dag.steps[0].delta.nodes_added == ["{A,B}", "C"]

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
        g.single_source_shortest_path_length("A", return_dict=True)
        g.multi_source_shortest_path_length(["A"], return_dict=True)
        g.cmty_leiden(return_dict=True)
        g.cmty_label_propagation(return_dict=True)
        assert len(g.steps) == n

    def test_player_serializes_groups_and_selection(self):
        g = _two_triangles().cmty_leiden(seed=1)
        g.all_shortest_paths("A", "F")
        html = g.play()._html
        assert '"ng": {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}' in html
        assert '"se": [["A", "C"], ["C", "D"], ["D", "F"]]' in html
        assert "groupColor(" in html

    def test_add_edges_by_id_records_labels_and_respects_compounds(self):
        g = ec.Graph().add_nodes(["A", "B"])
        g.add_edges_by_id([(1, 2)])
        assert g.steps[-1].delta.edges_added == [("A", "B")]
