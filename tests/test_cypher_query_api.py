"""Tests for the Cypher query methods on `Graph`: `match()`, `select_match()` and
`count_matches()`."""

import edgecase as ec
import polars as pl
import pytest

MOVIES = """
CREATE (keanu:Person {name: 'Keanu Reeves', born: 1964}),
       (carrie:Person {name: 'Carrie-Anne Moss', born: 1967}),
       (lana:Person {name: 'Lana Wachowski', born: 1965}),
       (matrix:Movie {title: 'The Matrix', released: 1999}),
       (wick:Movie {title: 'John Wick', released: 2014}),
       (keanu)-[:ACTED_IN {roles: ['Neo']}]->(matrix),
       (carrie)-[:ACTED_IN {roles: ['Trinity']}]->(matrix),
       (keanu)-[:ACTED_IN {roles: ['John Wick']}]->(wick),
       (lana)-[:DIRECTED]->(matrix)
"""

ACTED = "MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)"


@pytest.fixture
def g():
    return ec.from_cypher(MOVIES)


def node(g, label):
    return g.get_id_from_label(label)


def labels(g, ids):
    return sorted(g.get_label_from_id(i) for i in ids)


# =============================================================================
# match()
# =============================================================================


class TestMatchColumns:
    def test_named_variables_are_id_columns(self, g):
        df = g.match(ACTED)
        assert df.columns == ["p", "r", "m"]
        assert df.dtypes == [pl.Int64] * 3
        assert df.height == 3
        for p, r, m in df.iter_rows():
            edge = g.get_edge(r)
            assert (edge.source, edge.target) == (p, m)

    def test_ids_join_onto_the_node_table(self, g):
        df = g.match(ACTED).join(g.get_node_df(), left_on="m", right_on="id")
        assert sorted(df["title"].to_list()) == ["John Wick", "The Matrix", "The Matrix"]

    def test_anonymous_elements_have_no_column(self, g):
        df = g.match("MATCH (p:Person)-->()")
        assert df.columns == ["p"]
        assert labels(g, df["p"]) == [
            "Carrie-Anne Moss", "Keanu Reeves", "Keanu Reeves", "Lana Wachowski"]

    def test_use_labels(self, g):
        df = g.match(ACTED, use_labels=True)
        assert df.dtypes == [pl.String, pl.Int64, pl.String]
        assert sorted(zip(df["p"], df["m"])) == [
            ("Carrie-Anne Moss", "matrix"), ("Keanu Reeves", "matrix"), ("Keanu Reeves", "wick")]

    def test_no_matches_keeps_the_columns(self, g):
        df = g.match(ACTED + " WHERE m.released > 3000")
        assert df.columns == ["p", "r", "m"] and df.height == 0
        assert df.dtypes == [pl.Int64] * 3
        assert g.match(ACTED + " WHERE false", use_labels=True).dtypes == [
            pl.String, pl.Int64, pl.String]

    def test_needs_a_named_variable(self, g):
        with pytest.raises(ValueError, match=r"names no variables.*count_matches\(\)"):
            g.match("MATCH ()-->()")


class TestMatchReturn:
    def test_properties_and_order(self, g):
        df = g.match(ACTED + " RETURN m.title, p.name, r.roles")
        assert df.columns == ["m.title", "p.name", "r.roles"]
        assert df.schema["r.roles"] == pl.List(pl.String)
        assert sorted(df.iter_rows()) == [
            ("John Wick", "Keanu Reeves", ["John Wick"]),
            ("The Matrix", "Carrie-Anne Moss", ["Trinity"]),
            ("The Matrix", "Keanu Reeves", ["Neo"]),
        ]

    def test_variables_and_labels(self, g):
        df = g.match(ACTED + " RETURN m, p.born", use_labels=True)
        assert df.columns == ["m", "p.born"]
        assert df.dtypes == [pl.String, pl.Int64]

    def test_missing_properties_are_null(self, g):
        df = g.match("MATCH (n) RETURN n.born")
        assert sorted(df["n.born"].to_list(), key=str) == [1964, 1965, 1967, None, None]

    def test_mixed_types(self):
        g = ec.Graph().add_nodes([("a", {"v": 1}), ("b", {"v": "x"})])
        df = g.match("MATCH (n) RETURN n.v")
        assert df.height == 2

    def test_distinct(self, g):
        df = g.match(ACTED + " RETURN DISTINCT p.name")
        assert df["p.name"].to_list() == ["Keanu Reeves", "Carrie-Anne Moss"]
        assert g.match(ACTED + " RETURN DISTINCT r.roles").height == 3
        assert g.match("MATCH (n) RETURN DISTINCT n.born").height == 4  # one null row
        assert g.match(ACTED + " RETURN DISTINCT m", use_labels=True)["m"].to_list() == [
            "matrix", "wick"]

    def test_distinct_lists_and_maps(self):
        g = ec.Graph().add_nodes([
            ("a", {"v": [1, 2]}), ("b", {"v": [1, 2]}), ("c", {"v": [True]}), ("d", {"v": [1]}),
        ])
        assert g.match("MATCH (n) RETURN DISTINCT n.v").height == 3


class TestMatchOrder:
    def test_sorted_by_named_ids_in_order_of_appearance(self, g):
        df = g.match("MATCH (m:Movie)<-[r]-(p)")
        assert df.columns == ["m", "r", "p"]
        assert df.rows() == sorted(df.rows())
        df = g.match("MATCH (p)-[r]->(m) RETURN m, p")
        keys = [(p, m) for m, p in df.iter_rows()]
        assert keys == sorted(keys)

    def test_the_same_query_gives_the_same_table(self, g):
        text = "MATCH (a)--(b)--(c)"
        assert g.match(text).equals(g.match(text))


class TestMatchLimit:
    def test_limit(self, g):
        assert g.match(ACTED, limit=2).height == 2
        assert g.match(ACTED, limit=10).height == 3
        df = g.match(ACTED, limit=0)
        assert df.height == 0 and df.columns == ["p", "r", "m"]
        assert g.match(ACTED, limit=2)["p"].is_sorted()

    def test_limit_counts_distinct_rows(self, g):
        assert g.match(ACTED + " RETURN DISTINCT m.title", limit=2).height == 2

    @pytest.mark.parametrize("limit", [-1, 1.5, True, "2"])
    def test_bad_limit(self, g, limit):
        with pytest.raises(ValueError, match="limit= must be a non-negative integer"):
            g.match(ACTED, limit=limit)


class TestMatchOptions:
    def test_params(self, g):
        df = g.match("MATCH (p:Person) WHERE p.born >= $y RETURN p.name", params={"y": 1965})
        assert sorted(df["p.name"]) == ["Carrie-Anne Moss", "Lana Wachowski"]

    def test_distinct_nodes(self):
        g = ec.Graph(directed=True).add_nodes(["a", "b"]).add_edges([("a", "b"), ("b", "a")])
        assert g.match("MATCH (x)-->(y)-->(z)").height == 2
        assert g.match("MATCH (x)-->(y)-->(z)", distinct_nodes=True).height == 0

    def test_label_key(self, g):
        assert g.match("MATCH (m {name: 'matrix'})").height == 1
        assert g.match("MATCH (m {name: 'matrix'})", label_key=None).height == 0
        assert g.match("MATCH (m {title: 'wick'})", label_key="title").height == 0
        assert g.match("MATCH (m {title: 'John Wick'})", label_key="title").height == 1

    def test_undirected(self):
        g = ec.Graph(directed=False).add_nodes(["a", "b"]).add_edge("a", "b")
        for text in ("MATCH (x)-->(y)", "MATCH (x)<--(y)", "MATCH (x)--(y)"):
            assert sorted(g.match(text, use_labels=True).rows()) == [("a", "b"), ("b", "a")]

    def test_empty_graph(self):
        df = ec.Graph().match("MATCH (a)-[r]->(b)")
        assert df.columns == ["a", "r", "b"] and df.height == 0


class TestErrors:
    @pytest.mark.parametrize("method", ["match", "count_matches", "select_match"])
    def test_argument_types(self, g, method):
        f = getattr(g, method)
        with pytest.raises(TypeError, match="query= must be a string"):
            f(3)
        with pytest.raises(TypeError, match="params= must be a dict"):
            f(ACTED, params=[1])
        with pytest.raises(TypeError, match="label_key= must be"):
            f(ACTED, label_key=1)

    @pytest.mark.parametrize("method", ["match", "count_matches", "select_match"])
    def test_query_errors(self, g, method):
        f = getattr(g, method)
        with pytest.raises(ValueError, match=r"^Cypher: .*line 1, column"):
            f("MATCH (a")
        with pytest.raises(ValueError, match=r"\$y isn't given"):
            f("MATCH (a) WHERE a.born = $y")

    def test_params_can_be_any_mapping(self, g):
        from types import MappingProxyType
        assert g.count_matches("MATCH (p) WHERE p.born = $y", MappingProxyType({"y": 1964})) == 1

    def test_errors_record_no_step(self, g):
        n = len(g.steps)
        with pytest.raises(ValueError):
            g.match("MATCH (a")
        assert len(g.steps) == n


# =============================================================================
# count_matches()
# =============================================================================


class TestCountMatches:
    def test_counts_rows_of_match(self, g):
        for text in (ACTED, "MATCH (a)--(b)", "MATCH (a)-->(b)<--(c)", "MATCH (n:Nope)",
                     "MATCH (p:Person) WHERE p.born > 1964"):
            assert g.count_matches(text) == g.match(text).height

    def test_anonymous_patterns(self, g):
        assert g.count_matches("MATCH ()-->()") == 4
        assert g.count_matches("MATCH ()") == 5

    def test_return_distinct(self, g):
        assert g.count_matches(ACTED + " RETURN p") == 3
        assert g.count_matches(ACTED + " RETURN DISTINCT p") == 2
        assert g.count_matches(ACTED + " RETURN DISTINCT m.title") == 2

    def test_cycles(self):
        g = (
            ec.Graph(directed=True)
            .add_nodes(["A", "B", "C", "D"])
            .add_edges([("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "C")])
        )
        assert g.count_matches("MATCH (a)-->(b)-->(c)-->(a)") == 3
        assert g.count_matches("MATCH (a)-->(b)-->(a)") == 2
        assert g.count_matches("MATCH (a)-->(b)-->(c)", distinct_nodes=True) == 5

    def test_empty_graph(self):
        assert ec.Graph().count_matches("MATCH (a)") == 0


# =============================================================================
# select_match()
# =============================================================================


class TestSelectMatch:
    def test_every_node_by_default(self, g):
        assert g.select_match("MATCH (m:Movie {title: 'John Wick'})--()") is g
        assert labels(g, g.get_selected_node_ids()) == ["Keanu Reeves", "wick"]

    def test_select_one_or_several(self, g):
        g.select_match(ACTED, select="p")
        assert labels(g, g.get_selected_node_ids()) == ["Carrie-Anne Moss", "Keanu Reeves"]
        g.select_match(ACTED, select=["p", "m"], set_op="replace")
        assert labels(g, g.get_selected_node_ids()) == [
            "Carrie-Anne Moss", "Keanu Reeves", "matrix", "wick"]

    def test_select_edges(self, g):
        g.select_nodes_by_id([node(g, "Lana Wachowski")])
        g.select_match(ACTED + " WHERE m.released < 2000", select="r")
        assert g.count_selected_nodes() == 0
        ids = g.get_selected_edge_ids()
        assert sorted(g.get_edge(e).source for e in ids) == sorted(
            [node(g, "Keanu Reeves"), node(g, "Carrie-Anne Moss")])

    def test_set_ops(self, g):
        g.select_match("MATCH (p:Person)")
        g.select_match("MATCH (p) WHERE p.born > 1964", set_op="intersect")
        assert labels(g, g.get_selected_node_ids()) == ["Carrie-Anne Moss", "Lana Wachowski"]
        g.select_match("MATCH (p {born: 1967})", set_op="difference")
        assert labels(g, g.get_selected_node_ids()) == ["Lana Wachowski"]
        g.select_match("MATCH (m:Movie)")
        assert g.count_selected_nodes() == 3
        g.select_match("MATCH (m:Movie)", set_op="replace")
        assert labels(g, g.get_selected_node_ids()) == ["matrix", "wick"]

    def test_no_matches(self, g):
        g.select_match("MATCH (m:Movie)")
        g.select_match("MATCH (n:Nope)")
        assert g.count_selected_nodes() == 2
        g.select_match("MATCH (n:Nope)", set_op="replace")
        assert g.count_selected_nodes() == 0
        g.select_match("MATCH ()-[r:NOPE]->()", select="r", set_op="replace")
        assert g.count_selected_edges() == 0

    def test_then_act_on_the_selection(self, g):
        g.select_match(ACTED, select="p").set_node_attrs_ws(actor=True)
        df = g.get_node_df().filter(pl.col("actor"))
        assert sorted(df["label"]) == ["Carrie-Anne Moss", "Keanu Reeves"]

    def test_return_is_rejected(self, g):
        with pytest.raises(ValueError, match=r"select_match\(\) doesn't take RETURN.*select=.*"
                                             r"\(line 1, column 42\)"):
            g.select_match(ACTED + " RETURN p")

    def test_bad_select(self, g):
        g.select_match("MATCH (m:Movie)")
        with pytest.raises(ValueError, match=r"select= names 'x', which the query doesn't bind; "
                                             r"its variables are 'p', 'r', 'm'"):
            g.select_match(ACTED, select="x")
        with pytest.raises(ValueError, match="it names no variables"):
            g.select_match("MATCH ()", select="x")
        with pytest.raises(ValueError, match="both node and relationship variables"):
            g.select_match(ACTED, select=["p", "r"])
        for bad in (3, [], ["p", 3], {"p"}):
            with pytest.raises(TypeError, match="select= must be"):
                g.select_match(ACTED, select=bad)
        with pytest.raises(ValueError, match="set_op"):
            g.select_match(ACTED, set_op="nope")
        assert labels(g, g.get_selected_node_ids()) == ["matrix", "wick"]

    def test_errors_while_matching_keep_the_selection(self, g):
        g.select_match("MATCH (m:Movie)")
        with pytest.raises(ValueError, match="WHERE needs true, false or null"):
            g.select_match("MATCH (p) WHERE p.name", set_op="replace")
        assert labels(g, g.get_selected_node_ids()) == ["matrix", "wick"]

    def test_undirected_and_empty(self):
        u = ec.Graph(directed=False).add_nodes(["a", "b", "c"]).add_edge("a", "b")
        u.select_match("MATCH (x)-->(y)", select="y")
        assert labels(u, u.get_selected_node_ids()) == ["a", "b"]
        e = ec.Graph()
        e.select_match("MATCH (x)")
        assert e.count_selected_nodes() == 0


# =============================================================================
# Steps
# =============================================================================


class TestSteps:
    def test_match_step(self, g):
        g.match(ACTED + " WHERE m.released < 2000")
        step = g.steps[-1]
        assert (step.kind, step.method, step.result) == ("query", "match", 2)
        assert step.code == (
            "G.match('MATCH (p:Person)-[r:ACTED_IN]->(m:Movie) WHERE m.released...')"
            "  # 2 rows"
        )
        assert step.narration == "match(): 2 rows."
        # the first match is highlighted
        assert step.delta.nodes_selected == ["Keanu Reeves", "matrix"]
        assert step.delta.edges_selected == [("Keanu Reeves", "matrix")]

    def test_match_arguments_in_the_code(self, g):
        g.match("MATCH (p {born: $y})", params={"y": 1964}, limit=0, use_labels=True,
                distinct_nodes=True, label_key=None)
        step = g.steps[-1]
        assert step.code == (
            "G.match('MATCH (p {born: $y})', params=..., limit=0, use_labels=True, "
            "distinct_nodes=True, label_key=None)  # 0 rows"
        )
        assert step.delta.nodes_selected == []
        g.match("MATCH (p)\n   WHERE p.born = 1964")
        assert g.steps[-1].code == "G.match('MATCH (p) WHERE p.born = 1964')  # 1 row"
        assert g.steps[-1].narration == "match(): 1 row."

    def test_count_matches_step(self, g):
        assert g.count_matches("MATCH (a)-[r]->(a)") == 0
        step = g.steps[-1]
        assert (step.kind, step.method, step.result) == ("query", "count_matches", 0)
        assert step.code == "G.count_matches('MATCH (a)-[r]->(a)')"
        assert step.narration == "count_matches(): 0 matches."
        g.count_matches("MATCH (:Movie)", label_key="title")
        assert g.steps[-1].code == "G.count_matches('MATCH (:Movie)', label_key='title')"
        assert g.steps[-1].narration == "count_matches(): 2 matches."

    def test_select_match_step(self, g):
        g.select_match(ACTED, select=["p"], set_op="replace")
        step = g.steps[-1]
        assert (step.kind, step.method) == ("select", "select_match")
        assert step.code == (
            "G.select_match('MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)', select=['p'], "
            "set_op='replace')  # 2 selected"
        )
        assert sorted(step.delta.nodes_selected) == ["Carrie-Anne Moss", "Keanu Reeves"]
        g.select_match(ACTED, select="r")
        assert g.steps[-1].code.endswith("select='r')  # 3 selected")
        assert len(g.steps[-1].delta.edges_selected) == 3

    def test_inside_other_methods_no_step(self, g):
        n = len(g.steps)
        g._recording_depth += 1
        try:
            g.match(ACTED)
            g.count_matches(ACTED)
            g.select_match(ACTED)
        finally:
            g._recording_depth -= 1
        assert len(g.steps) == n
