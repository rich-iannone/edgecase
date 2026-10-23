"""Tests for LEDA graph I/O and LaTeX (TikZ) output, with NetworkX parity."""

import random

import edgecase as ec
import networkx as nx
import pytest


def _labeled_edges(graph):
    """Edges as (source label, target label, weight, label attribute), in ID order."""
    out = []
    for eid in sorted(graph.get_edge_ids(limit=None)):
        e = graph.get_edge(eid)
        attrs = e.attributes or {}
        out.append(
            (
                graph.get_label_from_id(e.source),
                graph.get_label_from_id(e.target),
                e.weight,
                attrs.get("label"),
            )
        )
    return out


def _labels(graph):
    return [graph.get_label_from_id(n) for n in sorted(graph.get_node_ids(limit=None))]


def _random_graph(rng, directed, weighted=False, labels=False):
    n = rng.randint(1, 9)
    g = ec.Graph(directed=directed, weighted=weighted).add_nodes([f"v{i}" for i in range(n)])
    for _ in range(rng.randint(0, 14)):
        s, t = f"v{rng.randrange(n)}", f"v{rng.randrange(n)}"
        if weighted:
            g.add_edges([(s, t, float(rng.randint(-5, 9)))])
        else:
            eid = g.add_edge(s, t).find_edge_ids(s, t)[-1]
            if labels and rng.random() < 0.6:
                g.set_edge_attrs("label", f"e{eid}", edge_ids=[eid])
    return g


# =============================================================================
# from_leda
# =============================================================================

LEDA_SAMPLE = """\
#header section
LEDA.GRAPH
string
int
-1
#nodes section
5
|{v1}|
|{v2}|
|{v3}|
|{v4}|
|{v5}|

#edges section
7
1 2 0 |{4}|
1 3 0 |{3}|
2 3 0 |{2}|
3 4 0 |{3}|
3 5 0 |{7}|
4 5 0 |{6}|
5 1 0 |{foo}|
"""


class TestFromLeda:
    def test_networkx_docs_sample_parity(self):
        # NetworkX keeps edge data as a string label; numeric data is a weight here, and
        # non-numeric data under an int type is an error
        with pytest.raises(ValueError, match="isn't a number"):
            ec.from_leda(LEDA_SAMPLE)
        text = LEDA_SAMPLE.replace("|{foo}|", "|{1}|")
        g = ec.from_leda(text)
        G = nx.parse_leda(text.splitlines())
        assert g.is_directed() and G.is_directed()
        assert sorted(_labels(g)) == sorted(G.nodes)
        ours = sorted((s, t, w) for s, t, w, _ in _labeled_edges(g))
        theirs = sorted((u, v, float(d["label"])) for u, v, d in G.edges(data=True))
        assert ours == theirs
        assert g.is_weighted()

    def test_undirected_and_string_edges(self):
        text = "LEDA.GRAPH\nstring\nstring\n-2\n3\n|{a}|\n|{b}|\n|{}|\n2\n1 2 0 |{x}|\n2 3 0 |{}|\n"
        g = ec.from_leda(text)
        G = nx.parse_leda(text)
        assert not g.is_directed() and not G.is_directed()
        assert not g.is_weighted()
        # an empty node label falls back to the node number, as in NetworkX
        assert sorted(_labels(g)) == sorted(map(str, G.nodes))
        assert _labeled_edges(g) == [("a", "b", None, "x"), ("b", "3", None, None)]

    def test_missing_direction_line_is_directed(self):
        g = ec.from_leda("LEDA.GRAPH\nvoid\nvoid\n2\n|{}|\n|{}|\n1\n1 2 0 |{}|\n")
        assert g.is_directed()
        assert g.count_nodes() == 2 and g.count_edges() == 1

    def test_directed_override(self):
        g = ec.from_leda(
            "LEDA.GRAPH\nvoid\nvoid\n-1\n2\n|{}|\n|{}|\n1\n1 2 0 |{}|\n", directed=False
        )
        assert not g.is_directed()

    def test_parallel_edges_and_self_loops_kept(self):
        text = "LEDA.GRAPH\nvoid\nvoid\n-1\n2\n|{}|\n|{}|\n3\n1 2 0 |{}|\n1 2 0 |{}|\n2 2 0 |{}|\n"
        g = ec.from_leda(text)
        assert g.count_edges() == 3

    def test_list_of_lines_and_file(self, tmp_path):
        text = "LEDA.GRAPH\nvoid\nvoid\n-2\n2\n|{p}|\n|{q}|\n1\n1 2 0 |{}|\n"
        path = tmp_path / "g.gw"
        path.write_text(text)
        for source in (text.splitlines(), path, str(path)):
            g = ec.from_leda(source)
            assert sorted(_labels(g)) == ["p", "q"]
            assert g.count_edges() == 1

    @pytest.mark.parametrize(
        "text, message",
        [
            ("GRAPH\nvoid\nvoid\n-1\n0\n0\n", "LEDA.GRAPH header"),
            ("LEDA.GRAPH\nvoid\nvoid\n-3\n0\n0\n", "direction line"),
            ("LEDA.GRAPH\nvoid\nvoid\n-1\nx\n", "number of nodes"),
            ("LEDA.GRAPH\nvoid\nvoid\n-1\n2\n|{}|\n", "Can't read node 2"),
            ("LEDA.GRAPH\nvoid\nvoid\n-1\n1\n|{}|\n1\n1 2 0 |{}|\n", "outside 1 to 1"),
            ("LEDA.GRAPH\nvoid\nvoid\n-1\n1\n|{}|\n1\n1 1 |{}|\n", "Can't read edge 1"),
            ("LEDA.GRAPH\nvoid\nvoid\n-1\n1\n|{}|\n", "number of edges"),
        ],
    )
    def test_malformed(self, text, message):
        with pytest.raises(ValueError, match=message):
            ec.from_leda(text)


# =============================================================================
# to_leda
# =============================================================================


class TestToLeda:
    def test_docstring_example(self):
        g = ec.Graph().add_path_graph(3)
        assert ec.to_leda(g) == (
            "LEDA.GRAPH\nstring\nvoid\n-2\n3\n|{1}|\n|{2}|\n|{3}|\n2\n1 2 0 |{}|\n2 3 0 |{}|\n"
        )

    def test_edge_types(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["a", "b"])
        g.add_edges([("a", "b", 2.5)])
        lines = ec.to_leda(g).splitlines()
        assert lines[2] == "double" and lines[3] == "-1" and lines[-1] == "1 2 0 |{2.5}|"

        g = ec.Graph().add_nodes(["a", "b", "c"]).add_edge("a", "b").add_edge("b", "c")
        g.set_edge_attrs("label", "road", edge_ids=g.find_edge_ids("a", "b"))
        lines = ec.to_leda(g).splitlines()
        assert lines[2] == "string"
        assert lines[-2:] == ["1 2 0 |{road}|", "2 3 0 |{}|"]

    def test_unwritable_labels(self):
        g = ec.Graph().add_nodes(["bad}|label"])
        with pytest.raises(ValueError, match="can't be written"):
            ec.to_leda(g)
        g = ec.Graph().add_nodes(["two\nlines"])
        with pytest.raises(ValueError, match="can't be written"):
            ec.to_leda(g)

    def test_filepath(self, tmp_path):
        g = ec.Graph().add_cycle_graph(4)
        path = tmp_path / "out.gw"
        text = ec.to_leda(g, filepath=path)
        assert path.read_text() == text

    @pytest.mark.parametrize("seed", range(20))
    def test_round_trip(self, seed):
        rng = random.Random(seed)
        directed = rng.random() < 0.5
        weighted = rng.random() < 0.4
        g = _random_graph(rng, directed, weighted=weighted, labels=not weighted)
        back = ec.from_leda(ec.to_leda(g))
        assert back.is_directed() == directed
        assert _labels(back) == _labels(g)
        expected = _labeled_edges(g)
        if not weighted:
            assert back.is_weighted() is False
        assert _labeled_edges(back) == expected

    @pytest.mark.parametrize("seed", range(10))
    def test_networkx_reads_our_output(self, seed):
        rng = random.Random(100 + seed)
        directed = rng.random() < 0.5
        g = _random_graph(rng, directed, labels=True)
        G = nx.parse_leda(ec.to_leda(g))
        assert G.is_directed() == directed
        assert sorted(G.nodes) == sorted(_labels(g))
        # NetworkX's simple graphs merge parallel edges; compare the node pairs
        ours = {(s, t) if directed else frozenset((s, t)) for s, t, _, _ in _labeled_edges(g)}
        theirs = {(u, v) if directed else frozenset((u, v)) for u, v in G.edges}
        assert ours == theirs


# =============================================================================
# LaTeX
# =============================================================================


def _nx_copy(g):
    """A NetworkX graph whose node names are the EdgeCase node IDs."""
    G = nx.DiGraph() if g.is_directed() else nx.Graph()
    G.add_nodes_from(sorted(g.get_node_ids()))
    for _, s, t in g.get_edges_as_tuples():
        G.add_edge(s, t)
    return G


def _body_lines(text):
    """Split TikZ code into node lines and the multiset of edge lines."""
    lines = text.splitlines()
    edges = sorted(line for line in lines if line.strip().startswith("\\draw") and " to" in line)
    others = [line for line in lines if line not in edges]
    return others, edges


class TestToLatexRaw:
    def test_docstring_example(self):
        g = ec.Graph(directed=True).add_path_graph(3)
        expected = (
            "  \\begin{tikzpicture}\n"
            "      \\draw\n"
            "        (0.0:2) node (1){1}\n"
            "        (120.0:2) node (2){2}\n"
            "        (240.0:2) node (3){3};\n"
            "      \\begin{scope}[->]\n"
            "        \\draw (1) to (2);\n"
            "        \\draw (2) to (3);\n"
            "      \\end{scope}\n"
            "    \\end{tikzpicture}\n"
        )
        assert ec.to_latex_raw(g) == expected

    @pytest.mark.parametrize("directed", [False, True])
    def test_parity_default_circle(self, directed):
        g = ec.Graph(directed=directed).add_path_graph(5)
        assert ec.to_latex_raw(g) == nx.to_latex_raw(_nx_copy(g))

    def test_parity_with_options(self):
        g = ec.Graph().add_star_graph(4)
        G = _nx_copy(g)
        ids = sorted(g.get_node_ids())
        pos = {n: (i * 1.23456, -i) for i, n in enumerate(ids)}
        node_opts = {ids[0]: "red", ids[2]: "[fill=blue!20]"}
        node_text = {ids[1]: "$v_1$"}
        pairs = [(s, t) for _, s, t in g.get_edges_as_tuples()]
        edge_opts = {pairs[0]: "bend left"}
        edge_text = {pairs[1]: "x", pairs[2]: "y"}
        edge_lopts = {pairs[1]: "above"}
        kwargs = dict(
            tikz_options="[scale=2]",
            default_node_options="circle, draw",
            default_edge_options="thick",
        )
        ours = ec.to_latex_raw(
            g,
            pos=pos,
            node_options=node_opts,
            node_label=node_text,
            edge_options=edge_opts,
            edge_label=edge_text,
            edge_label_options=edge_lopts,
            **kwargs,
        )
        theirs = nx.to_latex_raw(
            G,
            pos=pos,
            node_options=node_opts,
            node_label=node_text,
            edge_options=edge_opts,
            edge_label=edge_text,
            edge_label_options=edge_lopts,
            **kwargs,
        )
        assert _body_lines(ours) == _body_lines(theirs)

    def test_attribute_settings_and_layout_positions(self):
        g = ec.Graph().add_path_graph(4).layout_circular()
        G = _nx_copy(g)
        for nid in g.get_node_ids():
            G.nodes[nid]["pos"] = tuple(g.get_node(nid).attributes["pos"])
        ours = ec.to_latex_raw(g)
        theirs = nx.to_latex_raw(G, pos="pos", node_label="nolabel")
        # NetworkX writes the node name as text when there's no label; ours is the label
        assert _body_lines(ours) == _body_lines(theirs)

    def test_self_loop_and_labels(self):
        g = ec.Graph(directed=True).add_nodes(["a", "b"]).add_edge("a", "a").add_edge("a", "b")
        text = ec.to_latex_raw(g, node_label={"b": "B"}, edge_label={("a", "b"): "w"})
        assert "\\draw[loop,] (1) to (1);" in text
        assert "\\draw (1) to node[] {w} (2);" in text
        assert "node (2){B}" in text and "node (1){a}" in text

    def test_partial_positions_raise(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError, match="has no position"):
            ec.to_latex_raw(g, pos={1: (0, 0)})
        with pytest.raises(ValueError, match="isn't an"):
            ec.to_latex_raw(g, pos={1: 5, 2: (0, 0), 3: (1, 1)})

    def test_bad_edge_keys(self):
        g = ec.Graph().add_path_graph(3)
        with pytest.raises(ValueError, match="has no edge"):
            ec.to_latex_raw(g, edge_label={(1, 3): "x"})
        with pytest.raises(ValueError, match="isn't in the graph"):
            ec.to_latex_raw(g, edge_label={99: "x"})
        with pytest.raises(TypeError):
            ec.to_latex_raw(g, edge_label={"x": "x"})


class TestToLatex:
    def test_single_parity(self):
        g = ec.Graph().add_path_graph(4)
        G = _nx_copy(g)
        for as_document in (True, False):
            ours = ec.to_latex(g, caption="A path", latex_label="fig:p", as_document=as_document)
            theirs = nx.to_latex(G, caption="A path", latex_label="fig:p", as_document=as_document)
            assert ours == theirs

    def test_subfigures_parity(self):
        gs = [ec.Graph().add_path_graph(3), ec.Graph(directed=True).add_path_graph(4)]
        Gs = [_nx_copy(g) for g in gs]
        kwargs = dict(sub_captions=["one", "two"], sub_labels=["a", "b"], n_rows=2)
        assert ec.to_latex(gs, **kwargs) == nx.to_latex(Gs, **kwargs)

    def test_list_of_positions(self):
        gs = [ec.Graph().add_path_graph(2), ec.Graph().add_path_graph(2)]
        text = ec.to_latex(gs, pos=[{1: (0, 0), 2: (1, 0)}, {1: (0, 1), 2: (1, 1)}])
        assert "(0, 0) node (1)" in text and "(1, 1) node (2)" in text

    def test_length_mismatch(self):
        gs = [ec.Graph().add_path_graph(2), ec.Graph().add_path_graph(2)]
        with pytest.raises(ValueError, match="same length"):
            ec.to_latex(gs, sub_captions=["only one"])

    def test_filepath(self, tmp_path):
        path = tmp_path / "fig.tex"
        text = ec.to_latex(ec.Graph().add_path_graph(3), filepath=path)
        assert path.read_text() == text
        assert text.startswith("\\documentclass{report}")
