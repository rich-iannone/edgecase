"""NumPy, SciPy sparse and pandas conversions, adjacency lists, and JSON adjacency/tree data."""

import json

import edgecase as ec
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from edgecase.steps import KIND_BUILD_NODES

from tests.test_io_interop import assert_same, edges_of, nodes_of, nx_pairs, pairs, rich_graph

# =============================================================================
# Helpers
# =============================================================================


def multi_graph(directed=False, note="x # y"):
    """Space-free labels, parallel weighted edges, a self-loop, an isolated node, attributes."""
    g = ec.Graph(directed=directed, weighted=True)
    g.add_nodes(["a", "b", "c", "d", "lonely"])
    g.add_edge("a", "b", weight=2.0, rel="knows")
    g.add_edge("b", "a", weight=3.0)
    g.add_edge("b", "c", weight=1.5, note=note, tags=[1, "two"])
    g.add_edge("c", "c", weight=0.5)
    g.add_edge("d", "a", weight=4.0)
    return g


def to_nx(g, multi=True):
    """The graph as a NetworkX (multi)graph keyed by label, with `weight` set per edge."""
    if multi:
        G = (nx.MultiDiGraph if g.is_directed() else nx.MultiGraph)()
    else:
        G = (nx.DiGraph if g.is_directed() else nx.Graph)()
    for label, attrs in nodes_of(g):
        G.add_node(label, **attrs)
    for s, t, w, attrs in edges_of(g):
        G.add_edge(s, t, **({} if w is None else {"weight": w}), **attrs)
    return G


def single_step(g, method):
    assert len(g._steps) == 1
    assert g._steps[0].kind == KIND_BUILD_NODES
    assert g._steps[0].method == method


# =============================================================================
# NumPy, SciPy and pandas adjacency matrices
# =============================================================================


class TestMatrices:
    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("multigraph_weight", ["sum", "min", "max"])
    def test_matches_networkx(self, directed, multigraph_weight):
        g = multi_graph(directed)
        G = to_nx(g)
        nodelist = [label for label, _ in nodes_of(g)]
        if multigraph_weight == "sum":
            expected = nx.to_numpy_array(G, nodelist=nodelist)
        else:
            func = {"min": min, "max": max}[multigraph_weight]
            expected = nx.to_numpy_array(G, nodelist=nodelist, multigraph_weight=func)
        A = ec.to_numpy_array(g, multigraph_weight=multigraph_weight)
        np.testing.assert_array_equal(A, expected)
        S = ec.to_scipy_sparse_array(g, multigraph_weight=multigraph_weight)
        np.testing.assert_array_equal(S.toarray(), expected)
        df = ec.to_pandas_adjacency(g, multigraph_weight=multigraph_weight)
        assert list(df.index) == nodelist and list(df.columns) == nodelist
        np.testing.assert_array_equal(df.to_numpy(), expected)

    def test_unweighted_counts_edges(self):
        g = multi_graph()
        A = ec.to_numpy_array(g, weighted=False)
        assert A[0, 1] == A[1, 0] == 2
        assert A[2, 2] == 1
        np.testing.assert_array_equal(A, A.T)

    def test_unweighted_graph_ignores_missing_weights(self):
        g = ec.Graph().add_path_graph(3)
        np.testing.assert_array_equal(ec.to_numpy_array(g), [[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def test_nodelist_order_and_subset(self):
        g = multi_graph(directed=True)
        A = ec.to_numpy_array(g, nodelist=["c", "b"])
        np.testing.assert_array_equal(A, [[0.5, 0.0], [1.5, 0.0]])
        ids = [g.get_id_from_label("c"), g.get_id_from_label("b")]
        np.testing.assert_array_equal(ec.to_numpy_array(g, nodelist=ids), A)
        assert list(ec.to_pandas_adjacency(g, nodelist=["c", "b"]).index) == ["c", "b"]
        with pytest.raises(ValueError, match="more than once"):
            ec.to_numpy_array(g, nodelist=["a", "a"])
        with pytest.raises(ValueError):
            ec.to_numpy_array(g, nodelist=["nope"])

    def test_nonedge_and_dtype(self):
        g = ec.Graph().add_path_graph(3)
        A = ec.to_numpy_array(g, nonedge=np.inf)
        assert A[0, 2] == np.inf and A[0, 1] == 1
        B = ec.to_numpy_array(g, dtype=int)
        assert B.dtype == int
        S = ec.to_scipy_sparse_array(g, dtype=np.int32, format="coo")
        assert S.dtype == np.int32 and S.format == "coo"

    def test_bad_multigraph_weight(self):
        with pytest.raises(ValueError, match="multigraph_weight"):
            ec.to_numpy_array(ec.Graph(), multigraph_weight="mean")

    def test_empty_graph(self):
        assert ec.to_numpy_array(ec.Graph()).shape == (0, 0)
        assert ec.to_scipy_sparse_array(ec.Graph()).shape == (0, 0)
        assert ec.from_numpy_array(np.zeros((0, 0))).count_nodes() == 0

    def test_unlabeled_nodes_named_by_id(self):
        g = ec.Graph()
        g.add_node()
        g.add_node(label="x")
        assert list(ec.to_pandas_adjacency(g).index) == ["1", "x"]

    @pytest.mark.parametrize("directed", [False, True])
    def test_numpy_round_trip(self, directed):
        g = ec.Graph(directed=directed, weighted=True).add_nodes(["0", "1", "2", "3"])
        g.add_edge("0", "1", weight=2.0).add_edge("1", "2", weight=0.5)
        g.add_edge("3", "3", weight=7.0)
        h = ec.from_numpy_array(ec.to_numpy_array(g), directed=directed)
        assert pairs(h) == pairs(g)
        assert sorted(w for _, _, w, _ in edges_of(h)) == [0.5, 2.0, 7.0]
        assert h.is_weighted()
        single_step(h, "from_numpy_array")

    @pytest.mark.parametrize("directed", [False, True])
    def test_parallel_edges_round_trip(self, directed):
        g = multi_graph(directed)
        A = ec.to_numpy_array(g, weighted=False)
        labels = ["a", "b", "c", "d", "lonely"]
        h = ec.from_numpy_array(A, directed=directed, parallel_edges=True, node_labels=labels)
        assert pairs(h) == pairs(g)
        assert not h.is_weighted()
        S = ec.to_scipy_sparse_array(g, weighted=False)
        h = ec.from_scipy_sparse_array(S, directed=directed, parallel_edges=True)
        assert h.count_edges() == g.count_edges()
        df = ec.to_pandas_adjacency(g, weighted=False)
        assert pairs(ec.from_pandas_adjacency(df, directed, parallel_edges=True)) == pairs(g)

    def test_parallel_edges_needs_whole_numbers(self):
        with pytest.raises(ValueError, match="whole number"):
            ec.from_numpy_array(np.array([[0, 1.5], [1.5, 0]]), parallel_edges=True)

    def test_from_numpy_matches_networkx(self):
        rng = np.random.default_rng(3)
        A = rng.integers(0, 3, size=(8, 8)).astype(float)
        D = ec.from_numpy_array(A, directed=True)
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        assert pairs(D) == nx_pairs(G)
        weights = {(s, t): w for s, t, w, _ in edges_of(D)}
        assert weights == {(str(u), str(v)): d["weight"] for u, v, d in G.edges(data=True)}
        U = A + A.T
        assert pairs(ec.from_numpy_array(U)) == nx_pairs(nx.from_numpy_array(U))

    def test_weighted_detection(self):
        assert not ec.from_numpy_array(np.array([[0, 1], [1, 0]])).is_weighted()
        assert ec.from_numpy_array(np.array([[0, 2], [2, 0]])).is_weighted()
        g = ec.from_numpy_array(np.array([[0, 2], [2, 0]]), weighted=False)
        assert not g.is_weighted() and edges_of(g)[0][2] is None

    def test_asymmetric_undirected_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            ec.from_numpy_array(np.array([[0, 1], [0, 0]]))
        with pytest.raises(ValueError, match="symmetric"):
            ec.from_numpy_array(np.array([[0, 1], [2, 0]]))
        with pytest.raises(ValueError, match="symmetric"):
            ec.from_scipy_sparse_array(sp.csr_array(np.array([[0, 1], [0, 0]])))
        assert ec.from_numpy_array(np.array([[0, 1], [0, 0]]), directed=True).count_edges() == 1

    def test_bad_matrices(self):
        with pytest.raises(ValueError, match="square"):
            ec.from_numpy_array(np.zeros((2, 3)))
        with pytest.raises(TypeError):
            ec.from_numpy_array(np.array([["a", "b"], ["c", "d"]]))
        with pytest.raises(TypeError):
            ec.from_numpy_array(np.array([[0, 1j], [1j, 0]]))
        with pytest.raises(ValueError, match="non-finite"):
            ec.from_numpy_array(np.array([[0, np.inf], [np.inf, 0]]))
        with pytest.raises(TypeError):
            ec.from_scipy_sparse_array(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="square"):
            ec.from_scipy_sparse_array(sp.csr_array((2, 3)))

    def test_custom_nonedge(self):
        A = np.array([[np.inf, 0.0], [0.0, np.inf]])
        g = ec.from_numpy_array(A, nonedge=np.inf)
        assert pairs(g) == [("0", "1")] and edges_of(g)[0][2] == 0.0
        B = np.array([[np.nan, 2.0], [2.0, np.nan]])
        assert pairs(ec.from_numpy_array(B, nonedge=np.nan)) == [("0", "1")]

    def test_node_labels(self):
        g = ec.from_numpy_array(np.eye(2), node_labels=["x", 7])
        assert [label for label, _ in nodes_of(g)] == ["x", "7"]
        with pytest.raises(ValueError, match="2 nodes"):
            ec.from_numpy_array(np.eye(2), node_labels=["x"])
        with pytest.raises(ValueError, match="more than once"):
            ec.from_numpy_array(np.eye(2), node_labels=["x", "x"])

    def test_sparse_explicit_zeros_and_duplicates(self):
        coo = sp.coo_array(([0.0, 1.0, 2.0, 2.0], ([0, 0, 1, 1], [1, 2, 0, 0])), shape=(3, 3))
        g = ec.from_scipy_sparse_array(coo, directed=True)
        assert {(s, t): w for s, t, w, _ in edges_of(g)} == {("0", "2"): 1.0, ("1", "0"): 4.0}
        single_step(g, "from_scipy_sparse_array")

    def test_sparse_formats_and_matrix_class(self):
        g = multi_graph(directed=True)
        dense = ec.to_numpy_array(g)
        for fmt in ("csr", "csc", "coo", "lil", "dok"):
            S = ec.to_scipy_sparse_array(g, format=fmt)
            assert S.format == fmt
            np.testing.assert_array_equal(S.toarray(), dense)
        old = sp.csr_matrix(dense)
        h = ec.from_scipy_sparse_array(old, directed=True)
        np.testing.assert_array_equal(ec.to_numpy_array(h), dense)

    def test_sparse_matches_networkx(self):
        G = nx.gnm_random_graph(40, 120, seed=2, directed=True)
        for u, v in G.edges():
            G[u][v]["weight"] = (u * 7 + v) % 5 + 1
        S = nx.to_scipy_sparse_array(G, nodelist=sorted(G))
        g = ec.from_scipy_sparse_array(S, directed=True)
        assert pairs(g) == nx_pairs(G)
        H = nx.from_scipy_sparse_array(ec.to_scipy_sparse_array(g), create_using=nx.DiGraph)
        assert nx_pairs(H) == nx_pairs(G)
        assert all(H[u][v]["weight"] == G[u][v]["weight"] for u, v in G.edges())

    def test_pandas_adjacency_round_trip(self):
        g = multi_graph(directed=True)
        df = ec.to_pandas_adjacency(g, multigraph_weight="max")
        h = ec.from_pandas_adjacency(df, directed=True)
        assert [label for label, _ in nodes_of(h)] == list(df.index)
        np.testing.assert_array_equal(ec.to_numpy_array(h), df.to_numpy())
        single_step(h, "from_pandas_adjacency")
        G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        assert nx_pairs(G) == pairs(h)

    def test_pandas_adjacency_columns_reordered(self):
        df = pd.DataFrame([[0, 1], [0, 0]], index=["a", "b"], columns=["b", "a"])
        g = ec.from_pandas_adjacency(df, directed=True)
        assert pairs(g) == [("a", "a")]
        with pytest.raises(ValueError, match="same node names"):
            ec.from_pandas_adjacency(pd.DataFrame([[0]], index=["a"], columns=["b"]))
        with pytest.raises(ValueError, match="same node names"):
            ec.from_pandas_adjacency(
                pd.DataFrame([[0, 0], [0, 0]], index=["a", "a"], columns=["a", "a"])
            )


# =============================================================================
# pandas edge lists
# =============================================================================


class TestPandasEdgelist:
    def test_columns(self):
        g = multi_graph()
        df = ec.to_pandas_edgelist(g)
        assert list(df.columns) == ["source", "target", "weight", "rel", "note", "tags"]
        assert df["source"].tolist() == ["a", "b", "b", "c", "d"]
        assert df["weight"].tolist() == [2.0, 3.0, 1.5, 0.5, 4.0]
        # pandas 3 stores the missing string as NaN rather than None
        assert df["rel"].tolist()[0] == "knows" and pd.isna(df["rel"].tolist()[1])
        assert df["tags"].tolist()[2] == [1, "two"]

    def test_no_weight_column_without_weights(self):
        df = ec.to_pandas_edgelist(ec.Graph().add_path_graph(3), source="u", target="v")
        assert list(df.columns) == ["u", "v"]
        with pytest.raises(ValueError, match="different"):
            ec.to_pandas_edgelist(ec.Graph(), source="x", target="x")

    def test_empty_graph(self):
        df = ec.to_pandas_edgelist(ec.Graph().add_nodes(["a"]))
        assert len(df) == 0 and list(df.columns) == ["source", "target"]

    @pytest.mark.parametrize("directed", [False, True])
    def test_round_trip(self, directed):
        g = multi_graph(directed)
        g.delete_node("lonely")  # an edge list can't hold isolated nodes
        h = ec.from_pandas_edgelist(ec.to_pandas_edgelist(g), edge_attr=True, directed=directed)
        assert edges_of(h) == edges_of(g)
        assert h.is_weighted()
        single_step(h, "from_pandas_edgelist")

    def test_edge_attr_choices(self):
        df = pd.DataFrame({"s": ["a", "b"], "t": ["b", "c"], "weight": [2.0, None], "k": [1, 2]})
        g = ec.from_pandas_edgelist(df, "s", "t")
        assert [(w, a) for _, _, w, a in edges_of(g)] == [(None, {}), (None, {})]
        assert not g.is_weighted()
        g = ec.from_pandas_edgelist(df, "s", "t", edge_attr="k")
        assert [a for _, _, _, a in edges_of(g)] == [{"k": 1}, {"k": 2}]
        g = ec.from_pandas_edgelist(df, "s", "t", edge_attr=["weight", "k"])
        assert [(w, a) for _, _, w, a in edges_of(g)] == [(2.0, {"k": 1}), (None, {"k": 2})]
        assert g.is_weighted()
        with pytest.raises(ValueError, match="no column"):
            ec.from_pandas_edgelist(df, "s", "t", edge_attr="nope")
        with pytest.raises(ValueError, match="no column"):
            ec.from_pandas_edgelist(df)

    def test_non_numeric_weight_is_attribute(self):
        df = pd.DataFrame({"source": ["a"], "target": ["b"], "weight": ["heavy"]})
        g = ec.from_pandas_edgelist(df, edge_attr=True)
        assert edges_of(g) == [("a", "b", None, {"weight": "heavy"})]

    def test_missing_endpoint_raises(self):
        df = pd.DataFrame({"source": ["a", None], "target": ["b", "c"]})
        with pytest.raises(ValueError, match="missing"):
            ec.from_pandas_edgelist(df)

    def test_node_values_and_order(self):
        df = pd.DataFrame({"source": [3, 1], "target": [1, 2]})
        g = ec.from_pandas_edgelist(df, directed=True)
        assert [label for label, _ in nodes_of(g)] == ["3", "1", "2"]
        assert pairs(g) == [("1", "2"), ("3", "1")]

    def test_polars_dataframe(self):
        import polars as pl

        df = pl.DataFrame({"source": ["a", "b"], "target": ["b", "c"], "w": [1, None]})
        g = ec.from_pandas_edgelist(df, edge_attr=True)
        assert [a for _, _, _, a in edges_of(g)] == [{"w": 1}, {}]

    def test_networkx_interop(self):
        g = multi_graph(directed=True)
        df = ec.to_pandas_edgelist(g)
        G = nx.from_pandas_edgelist(df, edge_attr=True, create_using=nx.MultiDiGraph, edge_key=None)
        assert nx_pairs(G) == pairs(g)
        df2 = nx.to_pandas_edgelist(G)
        h = ec.from_pandas_edgelist(df2, edge_attr=["weight", "rel"], directed=True)
        assert pairs(h) == pairs(g)
        assert sorted(w for _, _, w, _ in edges_of(h)) == sorted(w for _, _, w, _ in edges_of(g))


# =============================================================================
# Adjacency lists
# =============================================================================


class TestAdjlist:
    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx_writer(self, directed):
        G = nx.relabel_nodes(nx.gnm_random_graph(30, 60, seed=5, directed=directed), str)
        G.add_edge("3", "3")
        g = ec.Graph(directed=directed).add_nodes(list(G))
        g.add_edges(list(G.edges()))
        assert ec.to_adjlist(g).splitlines() == list(nx.generate_adjlist(G))
        assert ec.to_multiline_adjlist(g).splitlines() == list(nx.generate_multiline_adjlist(G))

    @pytest.mark.parametrize("directed", [False, True])
    def test_round_trip(self, tmp_path, directed):
        g = multi_graph(directed)
        text = ec.to_adjlist(g, tmp_path / "g.adjlist")
        assert (tmp_path / "g.adjlist").read_text() == text
        h = ec.from_adjlist(tmp_path / "g.adjlist", directed=directed, parallel_edges=True)
        assert [label for label, _ in nodes_of(h)] == [label for label, _ in nodes_of(g)]
        assert pairs(h) == pairs(g)
        assert not h.is_weighted()
        single_step(h, "from_adjlist")
        h = ec.from_adjlist(str(tmp_path / "g.adjlist"), directed=directed)
        assert pairs(h) == sorted(set(pairs(g)))

    def test_networkx_reads_ours(self, tmp_path):
        g = multi_graph()
        ec.to_adjlist(g, tmp_path / "g.adjlist")
        G = nx.read_adjlist(tmp_path / "g.adjlist", create_using=nx.MultiGraph)
        assert nx_pairs(G) == pairs(g)
        D = multi_graph(directed=True)
        ec.to_adjlist(D, tmp_path / "d.adjlist")
        G = nx.read_adjlist(tmp_path / "d.adjlist", create_using=nx.MultiDiGraph)
        assert nx_pairs(G) == pairs(D)
        assert set(G) == {"a", "b", "c", "d", "lonely"}

    def test_reading_networkx_files(self, tmp_path):
        G = nx.les_miserables_graph()
        G = nx.relabel_nodes(G, {n: n.replace(" ", "_") for n in G})
        nx.write_adjlist(G, tmp_path / "g.adjlist")
        g = ec.from_adjlist(tmp_path / "g.adjlist")
        assert pairs(g) == nx_pairs(G)
        assert [label for label, _ in nodes_of(g)] == list(G)

    def test_text_comments_and_delimiter(self):
        g = ec.from_adjlist("# header\na,b,c  # trailing\n\nb,c,\n", delimiter=",")
        assert pairs(g) == [("a", "b"), ("a", "c"), ("b", "c")]
        g = ec.from_adjlist(["a b", "b a"], comments=None)
        assert pairs(g) == [("a", "b")]
        g = ec.from_adjlist(b"a b\nb a\n", parallel_edges=True)
        assert pairs(g) == [("a", "b"), ("a", "b")]
        g = ec.from_adjlist("a b\nb a\n", directed=True)
        assert pairs(g) == [("a", "b"), ("b", "a")]
        text = ec.to_adjlist(ec.Graph().add_path_graph(3), delimiter=";")
        assert text == "1;2\n2;3\n3\n"
        assert pairs(ec.from_adjlist(text, delimiter=";")) == [("1", "2"), ("2", "3")]

    def test_single_line_string_is_a_path(self):
        with pytest.raises(FileNotFoundError):
            ec.from_adjlist("a b c")

    def test_empty_graph_round_trip(self):
        # "" used to be read as the path "." (IsADirectoryError)
        for write, read in [
            (ec.to_adjlist, ec.from_adjlist),
            (ec.to_multiline_adjlist, ec.from_multiline_adjlist),
        ]:
            text = write(ec.Graph())
            assert text == ""
            g = read(text)
            assert g.count_nodes() == 0 and g.count_edges() == 0

    @pytest.mark.parametrize("label", ["has space", "hash#tag", "tab\there", ""])
    def test_bad_labels(self, label):
        g = ec.Graph().add_nodes(["ok", label])
        with pytest.raises(ValueError, match="can't be written"):
            ec.to_adjlist(g)
        with pytest.raises(ValueError, match="can't be written"):
            ec.to_multiline_adjlist(g)

    def test_label_with_space_ok_for_other_delimiter(self):
        g = ec.Graph().add_nodes(["new york", "boston"]).add_edge("new york", "boston")
        text = ec.to_adjlist(g, delimiter=",")
        assert pairs(ec.from_adjlist(text, delimiter=",")) == [("boston", "new york")]
        with pytest.raises(ValueError):
            ec.to_adjlist(g)


class TestMultilineAdjlist:
    @pytest.mark.parametrize("directed", [False, True])
    def test_round_trip(self, tmp_path, directed):
        g = multi_graph(directed)
        text = ec.to_multiline_adjlist(g, tmp_path / "g.txt")
        h = ec.from_multiline_adjlist(tmp_path / "g.txt", directed=directed, parallel_edges=True)

        def norm(graph):
            return sorted(
                ((s, t) if directed else tuple(sorted((s, t))), w, repr(a))
                for s, t, w, a in edges_of(graph)
            )

        assert norm(h) == norm(g)
        assert [label for label, _ in nodes_of(h)] == [label for label, _ in nodes_of(g)]
        assert h.is_weighted()
        assert ec.from_multiline_adjlist(text, directed=directed).count_edges() == (
            len(set(pairs(g)))
        )
        single_step(h, "from_multiline_adjlist")

    def test_layout(self):
        g = ec.Graph().add_nodes(["a", "b"]).add_edge("a", "b").add_edge("b", "b")
        assert ec.to_multiline_adjlist(g) == "a 1\nb {}\nb 1\nb {}\n"

    def test_networkx_reads_ours(self, tmp_path):
        # NetworkX's reader cuts lines at '#' and drops the spaces in edge data
        g = multi_graph(note="x_y")
        ec.to_multiline_adjlist(g, tmp_path / "g.txt")
        G = nx.read_multiline_adjlist(tmp_path / "g.txt", create_using=nx.MultiGraph)
        assert nx_pairs(G) == pairs(g)
        assert sorted(d["weight"] for _, _, d in G.edges(data=True)) == [0.5, 1.5, 2.0, 3.0, 4.0]

    def test_reading_networkx_files(self, tmp_path):
        G = nx.MultiDiGraph()
        G.add_edge("a", "b", weight=2, label="two words")
        G.add_edge("a", "b", weight=3)
        G.add_edge("b", "c")
        G.add_node("solo")
        nx.write_multiline_adjlist(G, tmp_path / "g.txt")
        g = ec.from_multiline_adjlist(tmp_path / "g.txt", directed=True, parallel_edges=True)
        assert edges_of(g) == [
            ("a", "b", 2.0, {"label": "two words"}),
            ("a", "b", 3.0, {}),
            ("b", "c", None, {}),
        ]
        assert [label for label, _ in nodes_of(g)] == ["a", "b", "c", "solo"]

    def test_comment_inside_data_kept(self):
        text = "a 1\nb {'note': 'x # y'}  # trailing comment\nb 0\n"
        g = ec.from_multiline_adjlist(text)
        assert edges_of(g) == [("a", "b", None, {"note": "x # y"})]

    def test_bare_number_is_weight(self):
        g = ec.from_multiline_adjlist("a 1\nb 2.5\n")
        assert edges_of(g) == [("a", "b", 2.5, {})] and g.is_weighted()

    def test_errors(self):
        with pytest.raises(ValueError, match="neighbor count"):
            ec.from_multiline_adjlist("a b c\n")
        with pytest.raises(ValueError, match="ends before"):
            ec.from_multiline_adjlist("a 2\nb {}\n")
        with pytest.raises(ValueError, match="can't read"):
            ec.from_multiline_adjlist("a 1\nb {oops\n")
        with pytest.raises(ValueError, match="must be a dict"):
            ec.from_multiline_adjlist("a 1\nb 'text'\n")
        g = ec.Graph(weighted=True).add_nodes(["a", "b"]).add_edge("a", "b", weight=float("inf"))
        with pytest.raises(ValueError, match="plain values"):
            ec.to_multiline_adjlist(g)


# =============================================================================
# JSON adjacency data
# =============================================================================


class TestAdjacencyData:
    @pytest.mark.parametrize("directed", [False, True])
    def test_round_trip(self, tmp_path, directed):
        g = rich_graph(directed)
        data = ec.to_adjacency_data(g, tmp_path / "g.json")
        assert data["multigraph"] is True
        for source in (data, tmp_path / "g.json", json.dumps(data)):
            h = ec.from_adjacency_data(source)
            assert nodes_of(h) == nodes_of(g)
            assert sorted(edges_of(h), key=repr) == sorted(edges_of(g), key=repr)
            assert h.get_graph_name() == "Rich graph"
            assert h.is_directed() == directed and h.is_weighted()
        single_step(h, "from_adjacency_data")

    def test_simple_graph_round_trip_exact(self):
        g = ec.Graph(weighted=True, graph_name="g").add_nodes(["a", "b", "c"])
        g.add_edge("a", "b", weight=1.0, k=1).add_edge("b", "c", weight=2.0)
        assert_same(ec.from_adjacency_data(ec.to_adjacency_data(g)), g)
        # An undirected edge comes back from its first listing's side
        g.add_edge("c", "a")
        h = ec.from_adjacency_data(ec.to_adjacency_data(g))
        assert ("a", "c", None, {}) in edges_of(h)

    @pytest.mark.parametrize("directed", [False, True])
    def test_matches_networkx(self, directed):
        g = multi_graph(directed)
        ours = ec.to_adjacency_data(g)
        G = to_nx(g, multi=ours["multigraph"])
        theirs = nx.adjacency_data(G)
        assert ours["nodes"] == theirs["nodes"]
        assert ours["directed"] == theirs["directed"]
        assert ours["multigraph"] == theirs["multigraph"]
        assert ours["adjacency"] == theirs["adjacency"]

    @pytest.mark.parametrize("directed", [False, True])
    @pytest.mark.parametrize("multigraph", [False, True])
    def test_reading_networkx_data(self, directed, multigraph):
        cls = {
            (False, False): nx.Graph,
            (False, True): nx.MultiGraph,
            (True, False): nx.DiGraph,
            (True, True): nx.MultiDiGraph,
        }[(directed, multigraph)]
        G = cls(name="nx")
        G.add_edge("a", "b", weight=2.0, color="red")
        G.add_edge("b", "c")
        G.add_edge("c", "c", weight=1.0)
        if multigraph:
            G.add_edge("a", "b", weight=5.0)
        G.add_node("solo", size=3)
        g = ec.from_adjacency_data(json.loads(json.dumps(nx.adjacency_data(G))))
        assert pairs(g) == nx_pairs(G)
        assert g.get_graph_name() == "nx"
        assert nodes_of(g)[-1] == ("solo", {"size": 3})
        assert any(
            {s, t} == {"a", "b"} and w == 2.0 and a == {"color": "red"}
            for s, t, w, a in edges_of(g)
        )

    def test_networkx_reads_ours(self):
        for directed in (False, True):
            g = multi_graph(directed)
            G = nx.adjacency_graph(ec.to_adjacency_data(g))
            assert G.is_multigraph() == (not directed) and G.is_directed() == directed
            assert nx_pairs(G) == pairs(g)
            assert set(G) == {"a", "b", "c", "d", "lonely"}

    def test_one_sided_undirected_listing(self):
        data = {"nodes": [{"id": 1}, {"id": 2}], "adjacency": [[{"id": 2}], []]}
        g = ec.from_adjacency_data(data)
        assert pairs(g) == [("1", "2")] and not g.is_weighted()

    def test_graph_field_as_dict(self):
        data = {"graph": {"name": "dict"}, "nodes": [], "adjacency": []}
        assert ec.from_adjacency_data(data).get_graph_name() == "dict"

    def test_errors(self):
        with pytest.raises(ValueError, match="'nodes' and 'adjacency'"):
            ec.from_adjacency_data({"nodes": []})
        with pytest.raises(ValueError, match="one 'adjacency' list per node"):
            ec.from_adjacency_data({"nodes": [{"id": 1}], "adjacency": []})
        with pytest.raises(ValueError, match="needs an 'id'"):
            ec.from_adjacency_data({"nodes": [{}], "adjacency": [[]]})
        with pytest.raises(ValueError, match="neighbor"):
            ec.from_adjacency_data({"nodes": [{"id": 1}], "adjacency": [[{}]]})


# =============================================================================
# JSON tree data
# =============================================================================


def tree(directed=True):
    g = ec.Graph(directed=directed).add_nodes(["r", "a", "b", "c", "d"])
    g.set_node_attrs("size", 2, nodes=["a"])
    g.add_edges([("r", "a"), ("r", "b"), ("a", "c"), ("a", "d")])
    return g


class TestTreeData:
    def test_layout(self):
        assert ec.to_tree_data(tree(), "r") == {
            "id": "r",
            "children": [
                {"id": "a", "size": 2, "children": [{"id": "c"}, {"id": "d"}]},
                {"id": "b"},
            ],
        }

    def test_matches_networkx(self):
        g = tree()
        G = nx.DiGraph()
        for label, attrs in nodes_of(g):
            G.add_node(label, **attrs)
        G.add_edges_from((s, t) for s, t, _, _ in edges_of(g))
        assert ec.to_tree_data(g, "r") == nx.tree_data(G, "r")
        assert ec.to_tree_data(g, "r", ident="name", children="kids") == nx.tree_data(
            G, "r", ident="name", children="kids"
        )

    def test_round_trip(self, tmp_path):
        g = tree()
        data = ec.to_tree_data(g, "r", tmp_path / "t.json")
        for source in (data, tmp_path / "t.json", json.dumps(data)):
            h = ec.from_tree_data(source)
            # Nodes come back in depth-first order
            assert nodes_of(h) == [nodes_of(g)[i] for i in (0, 1, 3, 4, 2)]
            assert pairs(h) == pairs(g)
        single_step(h, "from_tree_data")
        u = ec.from_tree_data(data, directed=False)
        assert not u.is_directed() and pairs(u) == sorted(tuple(sorted(p)) for p in pairs(g))

    def test_undirected_tree_oriented_from_root(self):
        g = tree(directed=False)
        data = ec.to_tree_data(g, "a")
        assert data == {
            "id": "a",
            "size": 2,
            "children": [{"id": "r", "children": [{"id": "b"}]}, {"id": "c"}, {"id": "d"}],
        }
        assert ec.to_tree_data(g, g.get_id_from_label("a")) == data

    def test_reading_networkx_data(self):
        G = nx.random_labeled_tree(40, seed=1)
        T = nx.bfs_tree(G, 0)
        g = ec.from_tree_data(nx.tree_data(T, 0))
        assert pairs(g) == nx_pairs(T)
        expected = [str(n) for n in nx.tree_graph(nx.tree_data(T, 0))]
        assert [label for label, _ in nodes_of(g)] == expected

    def test_networkx_reads_ours(self):
        T = nx.tree_graph(ec.to_tree_data(tree(), "r"))
        assert nx_pairs(T) == pairs(tree())

    def test_deep_tree_no_recursion_limit(self):
        g = ec.Graph(directed=True).add_path_graph(5000)
        data = ec.to_tree_data(g, "1")
        assert ec.from_tree_data(data).count_edges() == 4999

    def test_not_a_tree(self):
        g = tree()
        with pytest.raises(ValueError, match="rooted at"):
            ec.to_tree_data(g, "a")
        g.add_edge("b", "c")
        with pytest.raises(ValueError, match="isn't a tree"):
            ec.to_tree_data(g, "r")
        cyc = ec.Graph().add_cycle_graph(4)
        cyc.delete_node("4")
        cyc.add_edge("1", "1")
        with pytest.raises(ValueError, match="isn't a tree"):
            ec.to_tree_data(cyc, "1")
        forest = ec.Graph().add_nodes(["a", "b", "c", "d"])
        forest.add_edges([("a", "b"), ("c", "d"), ("c", "d")])
        with pytest.raises(ValueError, match="isn't a tree"):
            ec.to_tree_data(forest, "a")
        with pytest.raises(ValueError):
            ec.to_tree_data(tree(), "nope")

    def test_field_clashes(self):
        g = tree()
        g.set_node_attrs("children", 1, nodes=["b"])
        with pytest.raises(ValueError, match="'children'"):
            ec.to_tree_data(g, "r")
        assert ec.to_tree_data(g, "r", children="kids")["kids"][1] == {"id": "b", "children": 1}
        with pytest.raises(ValueError, match="different"):
            ec.to_tree_data(g, "r", ident="x", children="x")
        with pytest.raises(ValueError, match="different"):
            ec.from_tree_data({}, ident="x", children="x")

    def test_bad_tree_data(self):
        with pytest.raises(ValueError, match="'id' field"):
            ec.from_tree_data({"children": []})
        with pytest.raises(ValueError, match="must be a list"):
            ec.from_tree_data({"id": 1, "children": {"id": 2}})
        with pytest.raises(ValueError, match="more than once"):
            ec.from_tree_data({"id": 1, "children": [{"id": 2}, {"id": 2}]})


def test_single_node_tree():
    g = ec.Graph().add_nodes(["only"])
    assert ec.to_tree_data(g, "only") == {"id": "only"}
    assert nodes_of(ec.from_tree_data({"id": "only"})) == [("only", {})]
