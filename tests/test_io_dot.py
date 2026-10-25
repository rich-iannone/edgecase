"""Tests for `from_dot()` and `to_dot()`."""

import shutil
import subprocess

import edgecase as ec
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _nodes(g):
    """`{label: attributes}` for every node, in node ID order."""
    return {g.get_node(n).label: dict(g.get_node(n).attributes or {})
            for n in sorted(g.get_node_ids(limit=None))}


def _edges(g):
    """`(source label, target label, weight, attributes)` for every edge, in edge ID order."""
    out = []
    for e in sorted(g.get_edge_ids(limit=None)):
        x = g.get_edge(e)
        s, t = g.get_label_from_id(x.source), g.get_label_from_id(x.target)
        out.append((s, t, x.weight, dict(x.attributes or {})))
    return out


def _pairs(g):
    return g.get_edges_str_repr(use_labels=True)


def _same(g, h):
    assert g.is_directed() == h.is_directed()
    assert g.is_weighted() == h.is_weighted()
    assert g.get_graph_name() == h.get_graph_name()
    assert _nodes(g) == _nodes(h)
    assert _edges(g) == _edges(h)


# =============================================================================
# Reading
# =============================================================================


class TestGraphs:
    def test_digraph_and_graph(self):
        g = ec.from_dot("digraph { a -> b }")
        assert g.is_directed() and _pairs(g) == ["a->b"]
        h = ec.from_dot("graph { a -- b }")
        assert not h.is_directed() and _pairs(h) == ["a--b"]

    def test_keywords_ignore_case(self):
        g = ec.from_dot("STRICT DiGraph G { NODE [shape=box]; a -> b; SubGraph { c } }")
        assert g.is_directed() and g.get_graph_name() == "G"
        assert _nodes(g) == {"a": {"shape": "box"}, "b": {"shape": "box"}, "c": {"shape": "box"}}

    def test_graph_name(self):
        assert ec.from_dot('digraph "my graph" {}').get_graph_name() == "my graph"
        assert ec.from_dot("digraph 42 {}").get_graph_name() == "42"
        assert ec.from_dot("digraph {}").get_graph_name() is None
        assert ec.from_dot("digraph G {}", graph_name="other").get_graph_name() == "other"

    def test_empty_graph(self):
        g = ec.from_dot("graph {}")
        assert g.count_nodes() == 0 and not g.is_directed()

    def test_several_graphs(self):
        graphs = ec.from_dot("digraph one { a -> b } graph two { x -- y -- z }")
        assert [g.get_graph_name() for g in graphs] == ["one", "two"]
        assert [_pairs(g) for g in graphs] == [["a->b"], ["x--y", "y--z"]]

    def test_steps(self):
        g = ec.from_dot("digraph { a -> b }")
        assert len(g.steps) == 1
        assert g.steps[-1].code == "from_dot('digraph { a -> b }')"


class TestStatements:
    def test_nodes_in_order_of_appearance(self):
        g = ec.from_dot("digraph { c; a -> b; d [k=v]; b -> c }")
        assert list(_nodes(g)) == ["c", "a", "b", "d"]
        assert _pairs(g) == ["a->b", "b->c"]

    def test_chain_with_attributes(self):
        g = ec.from_dot("digraph { a -> b -> c [color=red] }")
        assert _edges(g) == [("a", "b", None, {"color": "red"}), ("b", "c", None, {"color": "red"})]

    def test_node_statements_update_attributes(self):
        g = ec.from_dot("graph { a [x=1]; a [y=2, x=3] [z=4]; a -- b }")
        assert _nodes(g)["a"] == {"x": "3", "y": "2", "z": "4"}

    def test_attribute_list_separators(self):
        g = ec.from_dot("graph { a [x=1; y=2, z=3 w=4] }")
        assert _nodes(g)["a"] == {"x": "1", "y": "2", "z": "3", "w": "4"}

    def test_attribute_without_value_is_true(self):
        g = ec.from_dot("graph { a [bold] }")
        assert _nodes(g)["a"] == {"bold": "true"}

    def test_parallel_edges_and_self_loops(self):
        g = ec.from_dot("digraph { a -> b; a -> b; a -> a }")
        assert _pairs(g) == ["a->b", "a->b", "a->a"]

    def test_graph_attributes_are_skipped(self):
        g = ec.from_dot("digraph { rankdir=LR; graph [bgcolor=white]; label=\"x\"; a -> b }")
        assert list(_nodes(g)) == ["a", "b"] and _nodes(g)["a"] == {}

    def test_ports_are_ignored(self):
        g = ec.from_dot('digraph { a:out:n -> b:sw; c:"p 1" -> a:e }')
        assert _pairs(g) == ["a->b", "c->a"]

    def test_semicolons_optional(self):
        g = ec.from_dot("digraph { a -> b c -> d; e }")
        assert _pairs(g) == ["a->b", "c->d"] and g.count_nodes() == 5


class TestSubgraphs:
    def test_edges_to_and_from_subgraphs(self):
        g = ec.from_dot("digraph { a -> {b c}; {d e} -> f; {g h} -> {i j} }")
        assert _pairs(g) == [
            "a->b", "a->c", "d->f", "e->f", "g->i", "g->j", "h->i", "h->j",
        ]

    def test_chain_through_subgraph(self):
        g = ec.from_dot("digraph { parse -> {lint check} -> done }")
        assert _pairs(g) == ["parse->lint", "parse->check", "lint->done", "check->done"]

    def test_subgraph_holds_nodes_of_nested_subgraphs_and_edges(self):
        g = ec.from_dot("digraph { x -> subgraph s { a -> b; { c } } }")
        assert _pairs(g) == ["a->b", "x->a", "x->b", "x->c"]

    def test_clusters_keep_their_nodes(self):
        g = ec.from_dot(
            "digraph { subgraph cluster_0 { label=\"zero\"; a -> b } subgraph cluster_1 { c } "
            "b -> c }"
        )
        assert list(_nodes(g)) == ["a", "b", "c"]
        assert _pairs(g) == ["a->b", "b->c"]


class TestDefaults:
    def test_defaults_apply_to_later_items(self):
        g = ec.from_dot(
            "digraph { a; node [shape=box]; edge [style=dashed]; b; a -> c; a -> b [style=bold] }"
        )
        assert _nodes(g) == {"a": {}, "b": {"shape": "box"}, "c": {"shape": "box"}}
        assert _edges(g) == [("a", "c", None, {"style": "dashed"}),
                             ("a", "b", None, {"style": "bold"})]

    def test_defaults_skip_existing_nodes(self):
        g = ec.from_dot("digraph { a; node [color=red]; a -> b }")
        assert _nodes(g) == {"a": {}, "b": {"color": "red"}}

    def test_defaults_stay_in_their_subgraph(self):
        g = ec.from_dot(
            "digraph { node [shape=box]; { node [color=red]; a; { node [shape=oval]; b } } c }"
        )
        assert _nodes(g) == {
            "a": {"shape": "box", "color": "red"},
            "b": {"shape": "oval", "color": "red"},
            "c": {"shape": "box"},
        }

    def test_defaults_update(self):
        g = ec.from_dot("digraph { node [k=1, j=2]; node [k=3]; a }")
        assert _nodes(g)["a"] == {"k": "3", "j": "2"}


class TestStrict:
    def test_strict_digraph_merges_repeated_edges(self):
        g = ec.from_dot("strict digraph { a -> b [x=1]; a -> b [y=2]; b -> a; a -> a; a -> a }")
        assert _edges(g) == [
            ("a", "b", None, {"x": "1", "y": "2"}),
            ("b", "a", None, {}),
            ("a", "a", None, {}),
        ]

    def test_strict_graph_merges_both_directions(self):
        g = ec.from_dot("strict graph { a -- b [x=1]; b -- a [x=2]; a -- {b c} }")
        assert _edges(g) == [("a", "b", None, {"x": "2"}), ("a", "c", None, {})]

    def test_strict_merge_skips_defaults(self):
        g = ec.from_dot("strict digraph { a -> b; edge [color=red]; a -> b [x=1] }")
        assert _edges(g) == [("a", "b", None, {"x": "1"})]


class TestIDs:
    def test_kinds_of_id(self):
        g = ec.from_dot(
            'digraph { _a1 -> "with space"; -1.5 -> .5; 7 -> é; "edge" -> "Node";'
            " café -> naïve }"
        )
        assert list(_nodes(g)) == [
            "_a1", "with space", "-1.5", ".5", "7", "é", "edge", "Node", "café", "naïve",
        ]

    def test_quoted_strings(self):
        text = r'digraph { a [x="say \"hi\"", y="a\nb\\c", z="line \
continued"] }'
        assert _nodes(ec.from_dot(text))["a"] == {
            "x": 'say "hi"', "y": "a\\nb\\\\c", "z": "line continued",
        }

    def test_backslash_then_quote(self):
        # as in Graphviz, \\" is a backslash and an escaped quote, so the string goes on
        g = ec.from_dot(r'digraph { a [x="1\\" y=2"] }')
        assert _nodes(g)["a"] == {"x": '1\\" y=2'}

    def test_concatenation(self):
        g = ec.from_dot('digraph { a [label="one, " + "two" + \n " three"] }')
        assert _nodes(g)["a"] == {"label": "one, two three"}

    def test_newlines_inside_quotes(self):
        g = ec.from_dot('digraph { a [label="two\nlines"] }')
        assert _nodes(g)["a"] == {"label": "two\nlines"}

    def test_html_strings(self):
        g = ec.from_dot(
            "digraph { a [label=<<b>bold</b> &amp; <i>x</i>>]; <node<br/>id> -> a }"
        )
        assert _nodes(g) == {"a": {"label": "<<b>bold</b> &amp; <i>x</i>>"}, "<node<br/>id>": {}}

    def test_comments(self):
        g = ec.from_dot(
            "# preprocessor line\n"
            "digraph { // to the end\n"
            "  a /* inline */ -> b; /* spanning\n lines */ c\n"
            "# another\n"
            "}"
        )
        assert list(_nodes(g)) == ["a", "b", "c"]

    def test_hash_only_starts_a_comment_at_line_start(self):
        with pytest.raises(ValueError, match="unexpected character '#'"):
            ec.from_dot("digraph { a # b }")

    def test_edge_ops_without_spaces(self):
        assert _pairs(ec.from_dot("digraph{a->b->-1}")) == ["a->b", "b->-1"]
        assert _pairs(ec.from_dot("graph{1--2--x}")) == ["1--2", "2--x"]


class TestWeights:
    def test_numeric_weights(self):
        g = ec.from_dot('digraph { a -> b [weight=2]; b -> c [weight="1e-3"]; c -> a }')
        assert g.is_weighted()
        assert _edges(g) == [("a", "b", 2.0, {}), ("b", "c", 0.001, {}), ("c", "a", None, {})]

    def test_non_numeric_weight_stays_an_attribute(self):
        g = ec.from_dot('digraph { a -> b [weight=heavy]; b -> c [weight="nan"] }')
        assert not g.is_weighted()
        assert [e[3] for e in _edges(g)] == [{"weight": "heavy"}, {"weight": "nan"}]

    def test_weight_from_edge_defaults(self):
        g = ec.from_dot("digraph { edge [weight=3]; a -> b; b -> c [weight=1] }")
        assert [e[2] for e in _edges(g)] == [3.0, 1.0]

    def test_weighted_argument(self):
        text = "digraph { a -> b [weight=2]; b -> c }"
        g = ec.from_dot(text, weighted=False)
        assert not g.is_weighted() and _edges(g)[0][3] == {"weight": "2"}
        g = ec.from_dot("digraph { a -> b }", weighted=True)
        assert g.is_weighted() and _edges(g)[0][2] is None

    def test_node_weight_is_an_attribute(self):
        g = ec.from_dot("digraph { a [weight=2] }")
        assert not g.is_weighted() and _nodes(g)["a"] == {"weight": "2"}


class TestSources:
    def test_path_string_bytes(self, tmp_path):
        path = tmp_path / "g.dot"
        path.write_text("digraph { a -> b }", encoding="utf-8")
        for source in (path, str(path), b"digraph { a -> b }"):
            assert _pairs(ec.from_dot(source)) == ["a->b"]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ec.from_dot(str(tmp_path / "missing.dot"))

    def test_bad_source_type(self):
        with pytest.raises(TypeError, match="DOT text or a path"):
            ec.from_dot(42)


class TestErrors:
    @pytest.mark.parametrize(
        "text, message",
        [
            ("digraph { a -- b }", r"a digraph joins nodes with ->, not -- \(line 1, column 13\)"),
            ("graph {\n  a -> b }", r"a graph joins nodes with --, not -> \(line 2, column 5\)"),
            ("diagram { a }", r"expected 'graph' or 'digraph', found 'diagram' "
                              r"\(line 1, column 1\)"),
            ("strict", r"expected 'graph' or 'digraph', but the text ended"),
            ("// nothing", r"the text holds no graph"),
            ("digraph { a -> b", r"expected '\}', but the text ended"),
            ("digraph { a [x=1 }", r"expected an attribute name or '\]', found '\}'"),
            ("digraph { a [x=] }", r"expected an attribute value, found '\]'"),
            ('digraph { a [x="open] }', r"a quoted string that never ends \(line 1, column 16\)"),
            ("digraph { a [x=<<b>] }", r"an HTML string that never ends"),
            ("digraph { a /* open", r"comment that never ends"),
            ("digraph { node }", r"expected '\[' after 'node'"),
            ("digraph { a -> node }", r"expected a node, found the keyword 'node'"),
            ('digraph { a [x="1" + y] }', r"only quoted strings can be joined with '\+'"),
            ("digraph { a & b }", r"unexpected character '&'"),
            ("digraph { a -> }", r"expected a node, found '\}'"),
            ("digraph G", r"expected '\{', but the text ended"),
        ],
    )
    def test_errors(self, text, message):
        # as bytes, since a string without "{" is a file name
        with pytest.raises(ValueError, match=message):
            ec.from_dot(text.encode())

    def test_empty_statement(self):
        # Graphviz allows one ';' after a statement, but no empty statements
        with pytest.raises(ValueError, match=r"expected a statement, found ';'"):
            ec.from_dot("digraph { a;; b }")


# =============================================================================
# Writing
# =============================================================================


class TestToDot:
    def test_layout(self):
        g = ec.Graph(directed=True, weighted=True, graph_name="G")
        g.add_nodes([("a", {"color": "red", "size": 3}), "b"])
        g.add_edge("a", "b", weight=2.0, style="dashed")
        assert ec.to_dot(g) == (
            "digraph G {\n"
            "  a [color=red, size=3];\n"
            "  b;\n"
            "  a -> b [weight=2.0, style=dashed];\n"
            "}\n"
        )

    def test_undirected_and_empty(self):
        g = ec.Graph().add_nodes(["x", "y"])
        g.add_edges([("x", "y")])
        assert ec.to_dot(g) == "graph {\n  x;\n  y;\n  x -- y;\n}\n"
        assert ec.to_dot(ec.Graph()) == "graph {\n}\n"

    def test_quoting(self):
        g = ec.Graph(directed=True, graph_name="my graph")
        g.add_nodes([
            ("plain_1", {}), ("two words", {}), ("node", {}), ("Graph", {}), ("-2.5", {}),
            ("1e5", {}), ("é", {}), ("", {}), ('say "hi"', {}), ("<<b>x</b>>", {}),
            ("<b>x</b>", {}), ("<x", {}),
        ])
        lines = ec.to_dot(g).splitlines()
        assert lines[0] == 'digraph "my graph" {'
        assert lines[1:-1] == [
            "  plain_1;", '  "two words";', '  "node";', '  "Graph";', "  -2.5;", '  "1e5";',
            "  é;", '  "";', '  "say \\"hi\\"";', "  <<b>x</b>>;", '  "<b>x</b>";',
            '  "<x";',
        ]

    def test_values(self):
        g = ec.Graph().add_nodes([("a", {
            "t": True, "f": False, "i": -4, "x": 0.5, "big": 1e20, "s": "a\\nb", "n": None,
        })])
        assert ec.to_dot(g).splitlines()[1] == (
            '  a [t=true, f=false, i=-4, x=0.5, big="1e+20", s="a\\nb"];'
        )

    def test_odd_attribute_names(self):
        g = ec.Graph().add_nodes([("a", {"my key": 1, "edge": 2})])
        assert ec.to_dot(g).splitlines()[1] == '  a ["my key"=1, "edge"=2];'

    def test_unwritable_values(self):
        for value, message in [
            ([1, 2], "is a list"), ({"k": 1}, "is a dict"), (float("inf"), "is inf"),
            ("ends\\", "backslash at the end of a line"),
            ("a\\\nb", "backslash at the end of a line"),
        ]:
            g = ec.Graph().add_nodes([("a", {"v": value})])
            with pytest.raises(ValueError, match=message):
                ec.to_dot(g)

    def test_weight_clash(self):
        # only a graph built from parsed data can have both
        g = ec.io._build_graph([], [("a", "b", 1.0, {"weight": "x"})], True, True, None, "t()")
        with pytest.raises(ValueError, match="'weight' attribute as well as its weight"):
            ec.to_dot(g)

    def test_unweighted_graph_keeps_weight_attribute(self):
        g = ec.from_dot("digraph { a -> b [weight=5] }", weighted=False)
        assert "a -> b [weight=5];" in ec.to_dot(g)

    def test_writes_file(self, tmp_path):
        g = ec.Graph(directed=True).add_nodes(["a", "b"])
        g.add_edges([("a", "b")])
        path = tmp_path / "out.dot"
        text = ec.to_dot(g, path)
        assert path.read_text(encoding="utf-8") == text
        _same(ec.from_dot(path), g)

    @pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz isn't installed")
    def test_graphviz_reads_it(self):
        g = ec.Graph(directed=True, weighted=True, graph_name="odd name")
        g.add_nodes([("a b", {"tip": 'say "hi"'}), ("node", {"tip": "<<b>x</b>>"}), "é"])
        g.add_edge("a b", "node", weight=0.5)
        g.add_edge("é", "é", weight=1e-9)
        result = subprocess.run(["dot", "-Tcanon"], input=ec.to_dot(g), capture_output=True,
                                text=True, check=True)
        assert ec.from_dot(result.stdout).count_edges() == 2


# =============================================================================
# Round trips
# =============================================================================


class TestRoundTrip:
    def test_strings_survive(self):
        g = ec.Graph(directed=True, weighted=True, graph_name="name with \"quotes\"")
        g.add_nodes([
            ("a b", {"path": "C:\\dir\\file", "multi": "two\nlines"}),
            ("node", {"html": "<<b>x</b>>", "back": 'a\\"b'}),
            ("ü", {}), ("-3", {}), ("", {"empty": ""}),
        ])
        g.set_node_attrs("label", {g.get_id_from_label("a b"): 'say "hi"'})
        g.add_edge("a b", "node", weight=-1.5, style="dashed")
        g.add_edge("a b", "node", weight=2.0)
        g.add_edge("ü", "ü", weight=1e-12)
        g.add_edge("-3", "", weight=7.0)
        _same(ec.from_dot(ec.to_dot(g)), g)

    def test_undirected_multigraph(self):
        g = ec.Graph().add_nodes(["a", "b", "c", "isolated"])
        g.add_edges([("a", "b"), ("b", "a"), ("c", "b"), ("c", "c")])
        _same(ec.from_dot(ec.to_dot(g)), g)

    def test_values_come_back_as_strings(self):
        g = ec.Graph().add_nodes([("a", {"n": 3, "x": 0.5, "t": True})])
        assert _nodes(ec.from_dot(ec.to_dot(g)))["a"] == {"n": "3", "x": "0.5", "t": "true"}

    def test_generators(self):
        for g in (ec.Graph().add_cycle_graph(7), ec.Graph(directed=True).add_star_graph(5)):
            h = ec.from_dot(ec.to_dot(g))
            assert h.is_isomorphic(g)
            assert _pairs(h) == _pairs(g)

    def test_dot_text_round_trip(self):
        text = """
        strict digraph flow {
            node [shape=box];
            start -> {left right} [weight=2];
            left -> end; right -> end [label="done"];
        }
        """
        g = ec.from_dot(text)
        _same(ec.from_dot(ec.to_dot(g)), g)
