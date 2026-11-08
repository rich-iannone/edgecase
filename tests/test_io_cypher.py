"""Tests for `from_cypher()` and `to_cypher()`."""

import edgecase as ec
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _nodes(g):
    """`{label: attributes}` for every node."""
    return {g.get_node(n).label: dict(g.get_node(n).attributes or {})
            for n in g.get_node_ids(limit=None)}


def _edges(g):
    """`(source label, target label, weight, attributes)` for every edge, in edge ID order."""
    out = []
    for e in sorted(g.get_edge_ids(limit=None)):
        x = g.get_edge(e)
        s, t = g.get_label_from_id(x.source), g.get_label_from_id(x.target)
        out.append((s, t, x.weight, dict(x.attributes or {})))
    return out


def _same(g, h):
    assert g.is_directed() == h.is_directed()
    assert g.is_weighted() == h.is_weighted()
    assert _nodes(g) == _nodes(h)
    assert _edges(g) == _edges(h)


MOVIES = """
// A corner of the Neo4j movie graph
CREATE (TheMatrix:Movie {title: 'The Matrix', released: 1999, tagline: 'Welcome to the Real World'})
CREATE (Keanu:Person {name: 'Keanu Reeves', born: 1964})
CREATE (Carrie:Person {name: 'Carrie-Anne Moss', born: 1967})
CREATE (Lana:Person {name: 'Lana Wachowski', born: 1965})
CREATE
  (Keanu)-[:ACTED_IN {roles: ['Neo']}]->(TheMatrix),
  (Carrie)-[:ACTED_IN {roles: ['Trinity']}]->(TheMatrix),
  (Lana)-[:DIRECTED]->(TheMatrix);
"""


# =============================================================================
# Reading
# =============================================================================


class TestFromCypher:
    def test_movie_graph(self):
        g = ec.from_cypher(MOVIES)
        assert g.is_directed() and not g.is_weighted()
        nodes = _nodes(g)
        # the movie has no `name`, so it's labeled by its variable
        assert nodes["TheMatrix"] == {
            "type": "Movie", "title": "The Matrix", "released": 1999,
            "tagline": "Welcome to the Real World",
        }
        assert nodes["Keanu Reeves"] == {"type": "Person", "born": 1964}
        assert _edges(g) == [
            ("Keanu Reeves", "TheMatrix", None, {"rel": "ACTED_IN", "roles": ["Neo"]}),
            ("Carrie-Anne Moss", "TheMatrix", None, {"rel": "ACTED_IN", "roles": ["Trinity"]}),
            ("Lana Wachowski", "TheMatrix", None, {"rel": "DIRECTED"}),
        ]

    def test_chains_and_commas(self):
        g = ec.from_cypher("CREATE (a)-[:R]->(b)-[:S]->(c), (c)-[:T]->(a)")
        assert g.get_edges_str_repr(use_labels=True) == ["a->b", "b->c", "c->a"]
        assert [e[3]["rel"] for e in _edges(g)] == ["R", "S", "T"]

    def test_backward_arrows(self):
        g = ec.from_cypher("CREATE (a)<-[:R]-(b)<--(c)")
        assert g.get_edges_str_repr(use_labels=True) == ["b->a", "c->b"]
        assert _edges(g)[1][3] == {}

    def test_bare_relationships(self):
        g = ec.from_cypher("CREATE (a)-->(b), (b)-[]->(c), (c)-[r]->(a)")
        assert g.get_edges_str_repr(use_labels=True) == ["a->b", "b->c", "c->a"]
        assert all(e[3] == {} for e in _edges(g))

    def test_undirected(self):
        g = ec.from_cypher("CREATE (a)-[:R]-(b)--(c)<-[:S]-(d)", directed=False)
        assert not g.is_directed()
        assert g.get_edges_str_repr(use_labels=True) == ["a--b", "b--c", "d--c"]

    def test_anonymous_nodes_avoid_used_names(self):
        g = ec.from_cypher("CREATE (n1)-[:R]->(), ()-[:R]->({name: 'n3'})")
        # the anonymous nodes are n2 and n4, skipping the n3 already taken by a name
        assert g.get_edges_str_repr(use_labels=True) == ["n1->n2", "n4->n3"]

    def test_variables_shared_across_clauses(self):
        g = ec.from_cypher("CREATE (a {name: 'A'}); CREATE (b {name: 'B'}) CREATE (a)-[:R]->(b)")
        assert g.count_nodes() == 2
        assert g.get_edges_str_repr(use_labels=True) == ["A->B"]

    def test_parallel_edges_and_self_loops(self):
        g = ec.from_cypher("CREATE (a)-[:R]->(b), (a)-[:R]->(b), (a)-[:LOOP]->(a)")
        assert g.get_edges_str_repr(use_labels=True) == ["a->b", "a->b", "a->a"]

    def test_values(self):
        g = ec.from_cypher(
            "CREATE ({name: 'x', i: 42, neg: -7, f: 1.5, e: 2e3, dot: .5, t: TRUE, b: false,"
            " gone: null, l: [1, 'two', [3.0]], empty: [], s: \"dq\"})"
        )
        assert _nodes(g)["x"] == {
            "i": 42, "neg": -7, "f": 1.5, "e": 2000.0, "dot": 0.5, "t": True, "b": False,
            "l": [1, "two", [3.0]], "empty": [], "s": "dq",
        }

    def test_string_escapes(self):
        text = r"""CREATE ({name: 'it\'s', a: "say \"hi\"", t: 'a\tb\nc\\d', u: 'é\U0001F600'})"""
        attrs = _nodes(ec.from_cypher(text))["it's"]
        assert attrs == {"a": 'say "hi"', "t": "a\tb\nc\\d", "u": "é\U0001F600"}

    def test_backtick_names_and_unicode(self):
        g = ec.from_cypher(
            "CREATE (`my node`:`Odd Label` {`full name`: 'Zoë', `tick``s`: 1})"
            "-[:`HAS PART`]->(ü:Ünïcode)"
        )
        assert _nodes(g) == {
            "my node": {"type": "Odd Label", "full name": "Zoë", "tick`s": 1},
            "ü": {"type": "Ünïcode"},
        }
        assert _edges(g)[0][3] == {"rel": "HAS PART"}

    def test_several_labels(self):
        g = ec.from_cypher("CREATE (a:Person:Actor:Person {name: 'A'})")
        assert _nodes(g)["A"] == {"type": ["Person", "Actor", "Person"]}

    def test_comments_and_keyword_case(self):
        g = ec.from_cypher("/* header\n spanning lines */ create (a) // trailing\n Create (b)")
        assert sorted(_nodes(g)) == ["a", "b"]

    def test_weights(self):
        g = ec.from_cypher(
            "CREATE (a)-[:R {weight: 2}]->(b), (b)-[:R {weight: 'heavy'}]->(c), (c)-[:R]->(a),"
            " (a)-[:R {weight: true}]->(c)"
        )
        assert g.is_weighted()
        weights = [(e[2], e[3].get("weight")) for e in _edges(g)]
        assert weights == [(2.0, None), (None, "heavy"), (None, None), (None, True)]
        assert not ec.from_cypher("CREATE (a)-[:R]->(b)").is_weighted()

    def test_label_key(self):
        text = "CREATE (a {name: 'A', id: 7})-[:R]->(b {name: 'B', id: 8})"
        g = ec.from_cypher(text, label_key="id")
        assert _nodes(g) == {"7": {"name": "A"}, "8": {"name": "B"}}
        g = ec.from_cypher(text, label_key=None)
        assert _nodes(g) == {"a": {"name": "A", "id": 7}, "b": {"name": "B", "id": 8}}

    def test_duplicate_names_fall_back_to_variables(self):
        text = "CREATE (a {name: 'Sam'})-[:R]->(b {name: 'Sam'})"
        with pytest.warns(UserWarning, match="'Sam' is used by more than one node"):
            g = ec.from_cypher(text)
        assert _nodes(g) == {"a": {"name": "Sam"}, "b": {"name": "Sam"}}

    def test_empty_text(self):
        for text in ("", "  // nothing\n", ";;"):
            g = ec.from_cypher(text)
            assert g.count_nodes() == 0 and g.is_directed()

    def test_path_and_graph_name(self, tmp_path):
        path = tmp_path / "g.cypher"
        path.write_text("CREATE (a)-[:R]->(b)", encoding="utf-8")
        g = ec.from_cypher(path, graph_name="small")
        assert g.get_edges_str_repr(use_labels=True) == ["a->b"]
        assert g.get_graph_name() == "small"
        # a string is always the text, never a file name (the path's first separator is `/`, or
        # `\` on Windows)
        with pytest.raises(ValueError, match="Cypher: unexpected character"):
            ec.from_cypher(str(path))

    def test_steps(self):
        g = ec.from_cypher("CREATE (a)-[:R]->(b)")
        assert len(g.steps) == 1
        assert g.steps[-1].code == "from_cypher('CREATE (a)-[:R]->(b)')"
        long = ec.from_cypher("CREATE " + ", ".join(f"(v{i})" for i in range(30)))
        assert long.steps[-1].code == "from_cypher(<str>)"


# =============================================================================
# Errors
# =============================================================================


class TestFromCypherErrors:
    @pytest.mark.parametrize(
        "text, message",
        [
            ("MATCH (a) RETURN a", r"only CREATE clauses can build a graph, found 'MATCH'; to "
                                   r"query a graph, use Graph.match\(\), select_match\(\) or "
                                   r"count_matches\(\) \(line 1, column 1\)"),
            ("CREATE (a) match (b)", r"found 'match'; to query a graph, use Graph.match\(\)"),
            ("CREATE (a)\nRETURN a", r"found 'RETURN' \(line 2, column 1\)"),
            ("CREATE (a)--(b)", r"needs a direction .* \(line 1, column 11\)"),
            ("CREATE\n  (a)<-->(b)", r"can't point both ways \(line 2, column 6\)"),
            ("CREATE (a {x: 1, x: 2})", r"'x' is given twice \(line 1, column 18\)"),
            ("CREATE (a {s: 'x})", r"the quote ' is never closed \(line 1, column 15\)"),
            ("CREATE (a) /* open", r"comment that never ends"),
            ("CREATE (a {m: {k: 1}})", r"can't be a map"),
            ("CREATE (a {s: '\\q'})", r"unknown escape \\q"),
            ("CREATE (a {s: '\\u12'})", r"bad \\u escape"),
            ("CREATE (a", r"expected '\)' to end the node, but the text ended"),
            ("CREATE (a)-[:R->(b)", r"expected '\]'"),
            ("CREATE (a {x 1})", r"expected ':' after the property name"),
            ("CREATE (a {x: })", r"expected a value, found '\}'"),
            ("CREATE (a:Person {name: 'A'}), (a:Person)", r"`a` is already declared"),
            ("CREATE (a), (a {k: 1})", r"`a` is already declared"),
            ("CREATE (a:Person {type: 'x'})", r"both labels and a 'type' property"),
            ("CREATE (a)-[:R {rel: 'x'}]->(b)", r"also has a 'rel' property"),
            ("CREATE (a) & (b)", r"unexpected character '&'"),
            ("CREATE", r"expected '\(' to start a node, but the text ended"),
        ],
    )
    def test_errors(self, text, message):
        with pytest.raises(ValueError, match=message):
            ec.from_cypher(text)

    def test_repeating_a_variable_is_fine(self):
        g = ec.from_cypher("CREATE (a:P {k: 1}), (a), (a)-[:R]->(a)")
        assert g.count_nodes() == 1 and g.count_edges() == 1

    def test_bad_source_type(self):
        with pytest.raises(TypeError, match="Cypher text or a Path"):
            ec.from_cypher(42)


# =============================================================================
# Writing
# =============================================================================


class TestToCypher:
    def test_layout(self):
        g = ec.Graph(directed=True).add_nodes([("A", {"type": "Person", "age": 31}), "B"])
        g.add_edges([("A", "B")])
        assert ec.to_cypher(g) == (
            "CREATE\n"
            "  (n1:Person {name: 'A', age: 31}),\n"
            "  (n2 {name: 'B'}),\n"
            "  (n1)-[:RELATED_TO]->(n2);\n"
        )

    def test_empty_graph(self):
        assert ec.to_cypher(ec.Graph()) == ""

    def test_relationship_types_and_weights(self):
        g = ec.Graph(directed=True, weighted=True).add_nodes(["a", "b"])
        g.add_edge("a", "b", weight=2.5, rel="KNOWS", since=2020)
        g.add_edge("b", "a", weight=1.0)
        assert ec.to_cypher(g, default_rel="LINK").splitlines()[-2:] == [
            "  (n1)-[:KNOWS {weight: 2.5, since: 2020}]->(n2),",
            "  (n2)-[:LINK {weight: 1.0}]->(n1);",
        ]

    def test_values(self):
        g = ec.Graph().add_nodes([("x", {
            "s": "it's a\\b\n", "t": True, "f": 1e20, "g": 1e-7, "l": [1, "a", [False]],
            "tu": (1, 2),
        })])
        assert ec.to_cypher(g).splitlines()[1] == (
            "  (n1 {name: 'x', s: 'it\\'s a\\\\b\\n', t: true, f: 1e20, g: 1e-07, "
            "l: [1, 'a', [false]], tu: [1, 2]});"
        )

    def test_odd_names_are_backticked(self):
        g = ec.Graph(directed=True).add_nodes([("x", {"type": ["Big Cat", "a`b"], "my key": 1})])
        g.add_edge("x", "x", rel="SELF LOOP")
        assert ec.to_cypher(g) == (
            "CREATE\n"
            "  (n1:`Big Cat`:`a``b` {name: 'x', `my key`: 1}),\n"
            "  (n1)-[:`SELF LOOP`]->(n1);\n"
        )

    def test_non_string_type_is_a_property(self):
        g = ec.Graph().add_nodes([("x", {"type": 3}), ("y", {"type": []})])
        lines = ec.to_cypher(g).splitlines()
        assert lines[1:] == ["  (n1 {name: 'x', type: 3}),", "  (n2 {name: 'y', type: []});"]

    def test_unwritable_values(self):
        for value, message in [({"k": 1}, "is a dict"), (float("nan"), "is nan"),
                               ([1, None], "holding None")]:
            g = ec.Graph().add_nodes([("x", {"v": value})])
            with pytest.raises(ValueError, match=message):
                ec.to_cypher(g)

    def test_label_key_clash(self):
        g = ec.Graph().add_nodes([("x", {"name": "other"})])
        with pytest.raises(ValueError, match="choose another label_key"):
            ec.to_cypher(g)
        assert "{label: 'x', name: 'other'}" in ec.to_cypher(g, label_key="label")

    def test_writes_file(self, tmp_path):
        g = ec.Graph(directed=True).add_nodes(["a", "b"])
        g.add_edge("a", "b", rel="R")
        path = tmp_path / "out.cypher"
        text = ec.to_cypher(g, path)
        assert path.read_text(encoding="utf-8") == text
        _same(ec.from_cypher(path), g)


# =============================================================================
# Round trips
# =============================================================================


class TestRoundTrip:
    def test_movie_graph(self):
        g = ec.from_cypher(MOVIES)
        _same(ec.from_cypher(ec.to_cypher(g)), g)

    def test_weighted_multigraph_with_loops(self):
        g = ec.Graph(directed=True, weighted=True)
        g.add_nodes([("a", {"type": ["P", "Q"], "k": [1.5, "x"]}), ("b c", {"z": False}), "ü"])
        g.add_edge("a", "b c", weight=-3.0, rel="R")
        g.add_edge("a", "b c", weight=0.25)
        g.add_edge("ü", "ü", weight=1e-12, note="it's")
        h = ec.from_cypher(ec.to_cypher(g))
        assert _edges(h)[1][3] == {"rel": "RELATED_TO"}
        g.set_edge_attrs("rel", {sorted(g.get_edge_ids(limit=None))[1]: "RELATED_TO",
                                 sorted(g.get_edge_ids(limit=None))[2]: "RELATED_TO"})
        _same(h, g)

    def test_undirected(self):
        g = ec.Graph().add_nodes(["a", "b", "c"])
        g.add_edges([("a", "b"), ("c", "b")])
        h = ec.from_cypher(ec.to_cypher(g, default_rel="R"), directed=False)
        assert h.get_edges_str_repr(use_labels=True) == g.get_edges_str_repr(use_labels=True)

    def test_built_in_generator(self):
        g = ec.Graph().add_cycle_graph(6)
        h = ec.from_cypher(ec.to_cypher(g), directed=False)
        assert h.count_edges() == 6 and h.is_isomorphic(g)
